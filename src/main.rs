//! ZeroClaw with Openclaw-Scribe extensions.
//!
//! Minimal binary entry point for the cross-compiled aarch64 build.
//! On a Raspberry Pi Zero 2W this would be wired to the full agent
//! loop; here it just validates that the library links correctly.

use openclaw_scribe::memory::markdown::MarkdownMemory;
use openclaw_scribe::memory::session::SessionManager;
use openclaw_scribe::memory::context::ContextManager;
use openclaw_scribe::memory::heartbeat::HeartbeatManager;
use openclaw_scribe::memory::startup::StartupManager;
use openclaw_scribe::memory::structured_log::{
    self, EventType, LogEvent, LogLevel, LoggingConfig, StructuredLogger,
    parse_logging_config,
};
use openclaw_scribe::memory::subagent::{SubagentManager, spawn_heartbeat_subagent};
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let workspace = std::env::var("ZEROCLAW_WORKSPACE")
        .unwrap_or_else(|_| ".".to_string());

    eprintln!("[zeroclaw] Starting with workspace: {workspace}");

    // Load config and set up structured logger
    let logging_config = load_logging_config(&workspace);
    let logger = StructuredLogger::new(logging_config);

    // Best-effort cleanup of old log files (non-blocking)
    {
        let cleanup_logger = logger.clone();
        std::thread::spawn(move || cleanup_logger.cleanup_old_logs());
    }

    // Generate a loop ID for this agent invocation
    let loop_id = format!(
        "{}-{}",
        chrono::Local::now().format("%Y%m%d-%H%M%S"),
        &uuid::Uuid::new_v4().to_string()[..8]
    );

    let md_mem = Arc::new(MarkdownMemory::new(&workspace));
    let session_mgr = Arc::new(Mutex::new(SessionManager::new(&workspace)));

    // Load markdown context for system prompt
    let context = md_mem.load_session_context().await?;
    logger.log_and_stderr(
        &LogEvent::new(&loop_id, 0, EventType::SessionEvent, LogLevel::Info)
            .with_detail(&format!("loaded {} bytes of session context", context.len())),
        &format!("[zeroclaw] Loaded {} bytes of session context", context.len()),
    );

    // Set up context window management
    let mut ctx_mgr = ContextManager::for_provider("anthropic");
    ctx_mgr.set_system_prompt_tokens(context.len() / 4);

    // Set up subagent manager for background task execution
    let notify: openclaw_scribe::memory::subagent::NotifyCallback =
        Arc::new(|msg| eprintln!("{msg}"));
    let subagent_mgr = Arc::new(SubagentManager::new(md_mem.clone(), notify));

    // Set up heartbeat and startup managers
    let _heartbeat_mgr = Arc::new(HeartbeatManager::new(md_mem.clone()));
    let startup_mgr = StartupManager::new(md_mem.clone());

    // Check for morning summary on startup
    if let Some(_prompt) = startup_mgr.check_and_generate().await? {
        logger.log_and_stderr(
            &LogEvent::new(&loop_id, 0, EventType::SessionEvent, LogLevel::Info)
                .with_detail("morning summary prompt ready"),
            "[zeroclaw] Morning summary prompt ready",
        );
    }

    // Spawn heartbeat as a background subagent (non-blocking)
    match spawn_heartbeat_subagent(&subagent_mgr, &md_mem, Vec::new()).await {
        Ok(Some(_handle)) => {
            logger.log_and_stderr(
                &LogEvent::new(&loop_id, 0, EventType::SessionEvent, LogLevel::Info)
                    .with_detail("heartbeat spawned as background subagent"),
                "[zeroclaw] Heartbeat spawned as background subagent",
            );
        }
        Ok(None) => {
            eprintln!("[zeroclaw] No heartbeat tasks found");
        }
        Err(e) => {
            logger.log_and_stderr(
                &LogEvent::new(&loop_id, 0, EventType::Error, LogLevel::Error)
                    .with_detail(&format!("heartbeat spawn error: {e}")),
                &format!("[zeroclaw] Heartbeat spawn error: {e}"),
            );
        }
    }

    // Start or resume session
    {
        let mut mgr = session_mgr.lock().await;
        if let Some(path) = mgr.find_todays_latest().await? {
            let turns = mgr.resume_session(&path).await?;
            ctx_mgr.recount_tokens(&turns);
            let msg = format!(
                "resumed session with {} turns ({} tokens)",
                turns.len(),
                ctx_mgr.current_tokens()
            );
            logger.log_and_stderr(
                &LogEvent::new(&loop_id, 0, EventType::SessionEvent, LogLevel::Info)
                    .with_detail(&msg),
                &format!("[zeroclaw] Resumed session with {} turns ({} tokens)",
                    turns.len(), ctx_mgr.current_tokens()),
            );
        } else {
            let id = mgr.new_session().await?;
            structured_log::log_session_event(
                &logger, &loop_id, 0,
                &format!("started new session: {id}"),
            );
            eprintln!("[zeroclaw] Started new session: {id}");
        }
    }

    eprintln!("[zeroclaw] Ready. Context usage: {:.1}%", ctx_mgr.usage_ratio() * 100.0);

    // Flush any buffered log events before exiting
    logger.flush();

    Ok(())
}

/// Load the logging configuration from config.toml.
///
/// Checks the workspace root first, then ~/.zeroclaw/config.toml.
/// Returns defaults if no config file is found.
fn load_logging_config(workspace: &str) -> LoggingConfig {
    // Try workspace config first
    let workspace_config = std::path::Path::new(workspace).join("config.toml");
    if let Ok(content) = std::fs::read_to_string(&workspace_config) {
        if let Ok(config) = parse_logging_config(&content) {
            return config;
        }
    }

    // Try ~/.zeroclaw/config.toml
    if let Ok(home) = std::env::var("HOME") {
        let global_config = std::path::Path::new(&home).join(".zeroclaw").join("config.toml");
        if let Ok(content) = std::fs::read_to_string(&global_config) {
            if let Ok(config) = parse_logging_config(&content) {
                return config;
            }
        }
    }

    LoggingConfig::default()
}

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
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let workspace = std::env::var("ZEROCLAW_WORKSPACE")
        .unwrap_or_else(|_| ".".to_string());

    eprintln!("[zeroclaw] Starting with workspace: {workspace}");

    let md_mem = Arc::new(MarkdownMemory::new(&workspace));
    let session_mgr = Arc::new(Mutex::new(SessionManager::new(&workspace)));

    // Load markdown context for system prompt
    let context = md_mem.load_session_context().await?;
    eprintln!("[zeroclaw] Loaded {} bytes of session context", context.len());

    // Set up context window management
    let mut ctx_mgr = ContextManager::for_provider("anthropic");
    ctx_mgr.set_system_prompt_tokens(context.len() / 4);

    // Set up heartbeat and startup managers
    let heartbeat_mgr = Arc::new(HeartbeatManager::new(md_mem.clone()));
    let startup_mgr = StartupManager::new(md_mem.clone());

    // Check for morning summary on startup
    if let Some(_prompt) = startup_mgr.check_and_generate().await? {
        eprintln!("[zeroclaw] Morning summary prompt ready");
    }

    // Check for heartbeat tasks
    if let Some(_prompt) = heartbeat_mgr.run_cycle().await? {
        eprintln!("[zeroclaw] Heartbeat tasks ready");
    }

    // Start or resume session
    {
        let mut mgr = session_mgr.lock().await;
        if let Some(path) = mgr.find_todays_latest().await? {
            let turns = mgr.resume_session(&path).await?;
            ctx_mgr.recount_tokens(&turns);
            eprintln!(
                "[zeroclaw] Resumed session with {} turns ({} tokens)",
                turns.len(),
                ctx_mgr.current_tokens()
            );
        } else {
            let id = mgr.new_session().await?;
            eprintln!("[zeroclaw] Started new session: {id}");
        }
    }

    eprintln!("[zeroclaw] Ready. Context usage: {:.1}%", ctx_mgr.usage_ratio() * 100.0);

    Ok(())
}

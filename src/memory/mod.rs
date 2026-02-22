//! Markdown-based memory, session persistence, and context management for ZeroClaw.
//!
//! This module provides a plain-file alternative to the SQLite memory backend.
//! It uses human-readable markdown files for all memory storage, making it
//! easy to inspect, edit, and version-control agent memory.
//!
//! It also provides session persistence so conversations survive process
//! restarts, writing each turn as JSONL to `~/.zeroclaw/sessions/`.
//!
//! Context window management prevents overflow by tracking token usage,
//! pruning old tool results, and auto-compacting conversations when they
//! approach the provider's context limit.
//!
//! ## Silent turns
//!
//! The `silent` module provides a `NO_REPLY` token that allows the agent
//! to perform background operations (heartbeat tasks, memory flush, startup
//! summaries) without producing user-visible output. Silent turn responses
//! are still persisted in session history.
//!
//! ## Heartbeat
//!
//! The `heartbeat` module reads recurring tasks from `HEARTBEAT.md` and
//! executes them as silent turns. Results are written to today's daily note,
//! and user-relevant output (prefixed with `NOTIFY:`) goes to a dedicated
//! `## Notifications` section. A `/heartbeat` command allows manual triggering.
//!
//! ## Startup summary
//!
//! The `startup` module detects the first session of a new day and, if
//! yesterday's daily note exists, injects a silent turn asking the agent
//! to write a `## Morning Summary` to today's daily note — giving an e-ink
//! display a fresh summary each morning without user action.
//!
//! ## Async subagent spawning
//!
//! The `subagent` module provides non-blocking background task execution.
//! When a heartbeat fires, it spawns a subagent via `tokio::spawn` instead
//! of running inline. The subagent gets its own ephemeral conversation
//! history, shares read access to markdown memory, and writes results to
//! daily notes using file-locked atomic appends. A `/tasks` command shows
//! running subagents, last completion time, and last error.
//!
//! ## Telegram bot
//!
//! The `telegram` module provides a Telegram bot channel using HTTP
//! long-polling (not webhooks). It polls `getUpdates` at a configurable
//! interval (default 5 seconds), parses incoming messages, and routes
//! them through the same agent loop as the CLI. Responses are sent back
//! via `sendMessage`. The Telegram listener shares the same session
//! persistence, memory, and tools as the CLI — both can run simultaneously.
//!
//! ## File Layout
//!
//! ```text
//! workspace/
//! ├── SOUL.md              — Agent personality & behavior rules
//! ├── USER.md              — User preferences & project context
//! ├── MEMORY.md            — Curated long-term knowledge
//! ├── HEARTBEAT.md         — Recurring heartbeat tasks
//! ├── memory/
//! │   ├── 2026-02-13.md    — Yesterday's daily note
//! │   └── 2026-02-14.md    — Today's daily note
//! └── sessions/
//!     ├── default/
//!     │   ├── session.jsonl — Default session turns
//!     │   └── session.json  — Session metadata
//!     └── scribe-hardware/
//!         ├── session.jsonl — Named session turns
//!         └── session.json  — Session metadata
//! ```
//!
//! ## Named Sessions
//!
//! Sessions are identified by slugified names (lowercase, hyphens).
//! Local commands (`/new`, `/switch`, `/sessions`, `/rename`) are
//! intercepted in the input handler before reaching the LLM.
//! On startup, the most recently active session across all names
//! is resumed automatically.
//!
//! ## Usage
//!
//! ```rust
//! use crate::memory::markdown::MarkdownMemory;
//! use crate::memory::markdown_tools;
//! use crate::memory::session::SessionManager;
//! use crate::memory::session_tools;
//! use crate::memory::context::{ContextManager, ContextConfig};
//! use crate::memory::silent::{NO_REPLY, is_silent_response};
//! use crate::memory::heartbeat::HeartbeatManager;
//! use crate::memory::startup::StartupManager;
//! use std::sync::Arc;
//! use tokio::sync::Mutex;
//!
//! let md_mem = Arc::new(MarkdownMemory::new("./workspace"));
//! let session_mgr = Arc::new(Mutex::new(SessionManager::new("./workspace")));
//!
//! // Load context for system prompt injection (with session name)
//! let session_name = session_mgr.lock().await.session_name().map(|s| s.to_string());
//! let context = md_mem.load_session_context_named(session_name.as_deref()).await?;
//!
//! // Set up context window management
//! let mut ctx_mgr = ContextManager::for_provider("anthropic");
//! ctx_mgr.set_system_prompt_tokens(context.len() / 4);
//!
//! // Set up heartbeat and startup managers
//! let heartbeat_mgr = Arc::new(HeartbeatManager::new(md_mem.clone()));
//! let startup_mgr = StartupManager::new(md_mem.clone());
//!
//! // Check for morning summary on startup
//! if let Some(prompt) = startup_mgr.check_and_generate().await? {
//!     // Inject prompt as a silent turn in the agent loop
//! }
//!
//! // Register tools in the agent loop
//! let mut tools = markdown_tools::all_markdown_tools(md_mem.clone());
//! tools.extend(session_tools::all_session_tools(session_mgr.clone()));
//! tools.extend(web_tools::all_web_tools(config.brave_api_key.clone()));
//! tools.push(heartbeat::heartbeat_tool(heartbeat_mgr));
//!
//! // In the agent response handler:
//! // if is_silent_response(&response) {
//! //     // Persist turn but don't display to user
//! // }
//!
//! // Optional: start Telegram listener alongside CLI
//! use crate::memory::telegram::{TelegramConfig, TelegramListener};
//! let tg_config = TelegramConfig::new(config.telegram_bot_token.clone());
//! let tg_listener = TelegramListener::new(
//!     tg_config,
//!     md_mem.clone(),
//!     session_mgr.clone(),
//! ).unwrap();
//! let (mut tg_rx, _tg_handle) = tg_listener.start_polling();
//! // Process tg_rx messages through the same agent loop
//! ```

pub mod agent;
pub mod agent_tools;
pub mod composio;
pub mod context;
pub mod failover;
pub mod file_tools;
pub mod heartbeat;
pub mod markdown;
pub mod markdown_tools;
pub mod memory_growth;
pub mod provider_tools;
pub mod recursion_guard;
pub mod sandbox;
pub mod session;
pub mod session_tools;
pub mod silent;
pub mod startup;
pub mod structured_log;
pub mod subagent;
pub mod telegram;
pub mod telegram_rich;
pub mod web_tools;

// ── Memory trait and types ──────────────────────────────────────────
//
// These are defined here so that MarkdownMemory can implement the
// Memory trait as a drop-in replacement for the SQLite backend.

use anyhow::Result;
use async_trait::async_trait;

/// Category tag for a memory entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryCategory {
    General,
    Decision,
    Discovery,
    Todo,
    Bug,
    Context,
    Custom(String),
}

/// A single memory entry returned by recall/list.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub key: String,
    pub content: String,
    pub category: MemoryCategory,
    pub score: Option<f64>,
}

/// Trait for memory backends (SQLite, Markdown, etc.).
#[async_trait]
pub trait Memory: Send + Sync {
    /// Store a memory entry.
    async fn store(&self, key: &str, content: &str, category: &str) -> Result<()>;
    /// Recall memories matching a query.
    async fn recall(&self, query: &str) -> Result<Vec<MemoryEntry>>;
    /// Get a specific entry by key.
    async fn get(&self, key: &str) -> Result<Option<String>>;
    /// Forget (delete) an entry by key.
    async fn forget(&self, key: &str) -> Result<()>;
    /// List entries in a category.
    async fn list(&self, category: &str) -> Result<Vec<MemoryEntry>>;
}

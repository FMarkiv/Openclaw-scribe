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
//! ## File Layout
//!
//! ```text
//! workspace/
//! ├── SOUL.md              — Agent personality & behavior rules
//! ├── USER.md              — User preferences & project context
//! ├── MEMORY.md            — Curated long-term knowledge
//! ├── memory/
//! │   ├── 2026-02-13.md    — Yesterday's daily note
//! │   └── 2026-02-14.md    — Today's daily note
//! └── sessions/
//!     ├── 2026-02-13_a1b2c3d4.jsonl  — Previous session
//!     └── 2026-02-14_e5f6g7h8.jsonl  — Today's session
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use crate::memory::markdown::MarkdownMemory;
//! use crate::memory::markdown_tools;
//! use crate::memory::session::SessionManager;
//! use crate::memory::session_tools;
//! use crate::memory::context::{ContextManager, ContextConfig};
//! use std::sync::Arc;
//! use tokio::sync::Mutex;
//!
//! let md_mem = Arc::new(MarkdownMemory::new("./workspace"));
//! let session_mgr = Arc::new(Mutex::new(SessionManager::new("./workspace")));
//!
//! // Load context for system prompt injection
//! let context = md_mem.load_session_context().await?;
//!
//! // Set up context window management
//! let mut ctx_mgr = ContextManager::for_provider("anthropic");
//! ctx_mgr.set_system_prompt_tokens(context.len() / 4);
//!
//! // Register tools in the agent loop
//! let mut tools = markdown_tools::all_markdown_tools(md_mem);
//! tools.extend(session_tools::all_session_tools(session_mgr));
//! tools.extend(web_tools::all_web_tools(config.brave_api_key.clone()));
//! ```

pub mod context;
pub mod markdown;
pub mod markdown_tools;
pub mod session;
pub mod session_tools;
pub mod web_tools;

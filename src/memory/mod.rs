//! Markdown-based memory and session persistence for ZeroClaw.
//!
//! This module provides a plain-file alternative to the SQLite memory backend.
//! It uses human-readable markdown files for all memory storage, making it
//! easy to inspect, edit, and version-control agent memory.
//!
//! It also provides session persistence so conversations survive process
//! restarts, writing each turn as JSONL to `~/.zeroclaw/sessions/`.
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
//! use std::sync::Arc;
//! use tokio::sync::Mutex;
//!
//! let md_mem = Arc::new(MarkdownMemory::new("./workspace"));
//! let session_mgr = Arc::new(Mutex::new(SessionManager::new("./workspace")));
//!
//! // Load context for system prompt injection
//! let context = md_mem.load_session_context().await?;
//!
//! // Register tools in the agent loop
//! let mut tools = markdown_tools::all_markdown_tools(md_mem);
//! tools.extend(session_tools::all_session_tools(session_mgr));
//! ```

pub mod markdown;
pub mod markdown_tools;
pub mod session;
pub mod session_tools;

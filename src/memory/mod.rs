//! Markdown-based memory system for ZeroClaw.
//!
//! This module provides a plain-file alternative to the SQLite memory backend.
//! It uses human-readable markdown files for all memory storage, making it
//! easy to inspect, edit, and version-control agent memory.
//!
//! ## File Layout
//!
//! ```text
//! workspace/
//! ├── SOUL.md              — Agent personality & behavior rules
//! ├── USER.md              — User preferences & project context
//! ├── MEMORY.md            — Curated long-term knowledge
//! └── memory/
//!     ├── 2026-02-13.md    — Yesterday's daily note
//!     └── 2026-02-14.md    — Today's daily note
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use crate::memory::markdown::MarkdownMemory;
//! use crate::memory::markdown_tools;
//! use std::sync::Arc;
//!
//! let md_mem = Arc::new(MarkdownMemory::new("./workspace"));
//!
//! // Load context for system prompt injection
//! let context = md_mem.load_session_context().await?;
//!
//! // Register tools in the agent loop
//! let tools = markdown_tools::all_markdown_tools(md_mem);
//! ```

pub mod markdown;
pub mod markdown_tools;

//! Markdown-based memory system for ZeroClaw.
//!
//! Provides file-based memory using plain markdown files:
//! - `SOUL.md`   — agent personality (read-only, loaded at session start)
//! - `USER.md`   — user preferences (read-only, loaded at session start)
//! - `MEMORY.md` — curated long-term memory (read/write via promote)
//! - `memory/YYYY-MM-DD.md` — daily append-only session notes
//!
//! This module implements the `Memory` trait so it can serve as a drop-in
//! replacement (or supplement) for the SQLite memory backend.

use anyhow::{Context, Result};
use chrono::{Local, NaiveDate};
use std::path::{Path, PathBuf};
use tokio::fs;

use crate::memory::memory_growth::{MemoryConfig, MemoryGrowthController};

/// A single search hit when recalling across markdown files.
#[derive(Debug, Clone)]
pub struct MarkdownSearchResult {
    /// Which file the match came from.
    pub source: String,
    /// The matching line(s) with surrounding context.
    pub content: String,
    /// Line number of the first match in the file.
    pub line_number: usize,
}

/// File-based markdown memory backend.
pub struct MarkdownMemory {
    base_dir: PathBuf,
    /// Memory growth controller (lazily references the same base_dir).
    memory_config: MemoryConfig,
}

impl MarkdownMemory {
    /// Create a new MarkdownMemory rooted at `base_dir`.
    ///
    /// Expected directory layout:
    /// ```text
    /// base_dir/
    /// ├── SOUL.md
    /// ├── USER.md
    /// ├── MEMORY.md
    /// └── memory/
    ///     ├── 2026-02-14.md
    ///     └── ...
    /// ```
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
            memory_config: MemoryConfig::default(),
        }
    }

    /// Create a new MarkdownMemory with custom memory growth configuration.
    pub fn with_config(base_dir: impl Into<PathBuf>, config: MemoryConfig) -> Self {
        Self {
            base_dir: base_dir.into(),
            memory_config: config,
        }
    }

    /// Get a MemoryGrowthController for this memory instance.
    pub fn growth_controller(&self) -> MemoryGrowthController {
        MemoryGrowthController::new(&self.base_dir, self.memory_config.clone())
    }

    /// Get the memory growth configuration.
    pub fn memory_config(&self) -> &MemoryConfig {
        &self.memory_config
    }

    // ── Path helpers ─────────────────────────────────────────────

    pub fn soul_path(&self) -> PathBuf {
        self.base_dir.join("SOUL.md")
    }

    pub fn user_path(&self) -> PathBuf {
        self.base_dir.join("USER.md")
    }

    pub fn memory_path(&self) -> PathBuf {
        self.base_dir.join("MEMORY.md")
    }

    pub fn daily_dir(&self) -> PathBuf {
        self.base_dir.join("memory")
    }

    pub fn daily_note_path(&self, date: NaiveDate) -> PathBuf {
        self.daily_dir().join(format!("{}.md", date.format("%Y-%m-%d")))
    }

    pub fn today_note_path(&self) -> PathBuf {
        self.daily_note_path(Local::now().date_naive())
    }

    pub fn yesterday_note_path(&self) -> PathBuf {
        let yesterday = Local::now().date_naive() - chrono::Duration::days(1);
        self.daily_note_path(yesterday)
    }

    /// The base directory for all markdown memory files.
    pub fn base_dir(&self) -> &std::path::Path {
        &self.base_dir
    }

    // ── Session start: build system prompt context ───────────────

    /// Load all markdown context for injection into the system prompt.
    ///
    /// Returns a string containing SOUL.md + USER.md + MEMORY.md +
    /// yesterday's daily note + today's daily note, each clearly
    /// delimited with headers.
    ///
    /// If `session_name` is provided, it is included as a session
    /// context header so the LLM knows which project context it's in.
    pub async fn load_session_context(&self) -> Result<String> {
        self.load_session_context_named(None).await
    }

    /// Load session context with an optional session name injected.
    pub async fn load_session_context_named(
        &self,
        session_name: Option<&str>,
    ) -> Result<String> {
        let mut context = String::new();

        // Session name — so the LLM knows which project context it's in
        if let Some(name) = session_name {
            context.push_str(&format!("=== Active Session: {name} ===\n\n"));
        }

        // SOUL.md — always loaded
        if let Some(soul) = self.read_file_if_exists(&self.soul_path()).await? {
            context.push_str("=== SOUL (Agent Identity) ===\n");
            context.push_str(&soul);
            context.push_str("\n\n");
        }

        // USER.md — always loaded
        if let Some(user) = self.read_file_if_exists(&self.user_path()).await? {
            context.push_str("=== USER (Preferences & Context) ===\n");
            context.push_str(&user);
            context.push_str("\n\n");
        }

        // MEMORY.md — long-term knowledge (selective loading)
        let memory_content = self.load_memory_selective(&[]).await?;
        if !memory_content.is_empty() {
            context.push_str("=== MEMORY (Long-Term Knowledge) ===\n");
            context.push_str(&memory_content);
            context.push_str("\n\n");
        }

        // Yesterday's daily note — recent context
        if let Some(yesterday) = self
            .read_file_if_exists(&self.yesterday_note_path())
            .await?
        {
            let date = (Local::now().date_naive() - chrono::Duration::days(1))
                .format("%Y-%m-%d")
                .to_string();
            context.push_str(&format!("=== Daily Note ({date} — yesterday) ===\n"));
            context.push_str(&yesterday);
            context.push_str("\n\n");
        }

        // Today's daily note — current session context
        if let Some(today) = self.read_file_if_exists(&self.today_note_path()).await? {
            let date = Local::now().date_naive().format("%Y-%m-%d").to_string();
            context.push_str(&format!("=== Daily Note ({date} — today) ===\n"));
            context.push_str(&today);
            context.push_str("\n\n");
        }

        Ok(context)
    }

    /// Load session context with selective memory loading based on recent messages.
    ///
    /// This is the preferred method when recent conversation messages are available,
    /// as it enables keyword-based section matching for leaner system prompts.
    pub async fn load_session_context_with_messages(
        &self,
        session_name: Option<&str>,
        recent_messages: &[String],
    ) -> Result<String> {
        let mut context = String::new();

        if let Some(name) = session_name {
            context.push_str(&format!("=== Active Session: {name} ===\n\n"));
        }

        if let Some(soul) = self.read_file_if_exists(&self.soul_path()).await? {
            context.push_str("=== SOUL (Agent Identity) ===\n");
            context.push_str(&soul);
            context.push_str("\n\n");
        }

        if let Some(user) = self.read_file_if_exists(&self.user_path()).await? {
            context.push_str("=== USER (Preferences & Context) ===\n");
            context.push_str(&user);
            context.push_str("\n\n");
        }

        // MEMORY.md — selective loading based on conversation context
        let memory_content = self.load_memory_selective(recent_messages).await?;
        if !memory_content.is_empty() {
            context.push_str("=== MEMORY (Long-Term Knowledge) ===\n");
            context.push_str(&memory_content);
            context.push_str("\n\n");
        }

        if let Some(yesterday) = self.read_file_if_exists(&self.yesterday_note_path()).await? {
            let date = (Local::now().date_naive() - chrono::Duration::days(1))
                .format("%Y-%m-%d")
                .to_string();
            context.push_str(&format!("=== Daily Note ({date} — yesterday) ===\n"));
            context.push_str(&yesterday);
            context.push_str("\n\n");
        }

        if let Some(today) = self.read_file_if_exists(&self.today_note_path()).await? {
            let date = Local::now().date_naive().format("%Y-%m-%d").to_string();
            context.push_str(&format!("=== Daily Note ({date} — today) ===\n"));
            context.push_str(&today);
            context.push_str("\n\n");
        }

        Ok(context)
    }

    /// Load MEMORY.md content using selective loading.
    ///
    /// Delegates to the growth controller for keyword-based section matching.
    /// Falls back to full content for small files or legacy format.
    async fn load_memory_selective(&self, recent_messages: &[String]) -> Result<String> {
        let gc = self.growth_controller();
        gc.load_selective(recent_messages).await
    }

    // ── Daily notes: append-only log ─────────────────────────────

    /// Append an entry to today's daily note.
    ///
    /// Creates the file (with a date header) if it doesn't exist yet.
    /// Each entry is timestamped and separated by a blank line.
    pub async fn append_daily_note(&self, content: &str) -> Result<()> {
        let path = self.today_note_path();
        self.ensure_daily_dir().await?;

        let timestamp = Local::now().format("%H:%M:%S");
        let entry = if path.exists() {
            format!("\n### {timestamp}\n\n{content}\n")
        } else {
            let date = Local::now().date_naive().format("%Y-%m-%d");
            format!("# Daily Note — {date}\n\n### {timestamp}\n\n{content}\n")
        };

        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await
            .with_context(|| format!("Failed to open daily note: {}", path.display()))?
            .write_all(entry.as_bytes())
            .await
            .with_context(|| format!("Failed to append to daily note: {}", path.display()))?;

        Ok(())
    }

    // ── Search: grep across all markdown files ───────────────────

    /// Search across all markdown files for lines containing `query`.
    ///
    /// Performs case-insensitive substring matching. Returns matches
    /// with surrounding context lines (1 line before and after).
    pub async fn search(&self, query: &str) -> Result<Vec<MarkdownSearchResult>> {
        let query_lower = query.to_lowercase();
        let mut results = Vec::new();

        // Search MEMORY.md
        self.search_file(&self.memory_path(), &query_lower, &mut results)
            .await?;

        // Search daily notes (most recent first)
        let mut daily_files = self.list_daily_notes().await?;
        daily_files.sort_by(|a, b| b.cmp(a)); // newest first

        for path in daily_files {
            self.search_file(&path, &query_lower, &mut results)
                .await?;
        }

        // Search SOUL.md and USER.md (less likely to match but complete)
        self.search_file(&self.soul_path(), &query_lower, &mut results)
            .await?;
        self.search_file(&self.user_path(), &query_lower, &mut results)
            .await?;

        Ok(results)
    }

    /// Search a single file for lines matching the query.
    async fn search_file(
        &self,
        path: &Path,
        query_lower: &str,
        results: &mut Vec<MarkdownSearchResult>,
    ) -> Result<()> {
        let content = match self.read_file_if_exists(path).await? {
            Some(c) => c,
            None => return Ok(()),
        };

        let lines: Vec<&str> = content.lines().collect();
        let source = path
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| path.display().to_string());

        for (i, line) in lines.iter().enumerate() {
            if line.to_lowercase().contains(query_lower) {
                // Grab 1 line of context before and after
                let start = i.saturating_sub(1);
                let end = (i + 2).min(lines.len());
                let context_lines: Vec<&str> = lines[start..end].to_vec();

                results.push(MarkdownSearchResult {
                    source: source.clone(),
                    content: context_lines.join("\n"),
                    line_number: i + 1, // 1-indexed
                });
            }
        }

        Ok(())
    }

    // ── Promote: daily note → MEMORY.md ──────────────────────────

    /// Promote content from a daily note (or freeform text) into MEMORY.md.
    ///
    /// Appends the content under the appropriate section. The caller
    /// should specify which section to add under (e.g., "Project Facts",
    /// "Decisions", "Patterns", "Known Issues") or it defaults to a
    /// general "Promoted" section.
    pub async fn promote(&self, content: &str, section: Option<&str>) -> Result<()> {
        let path = self.memory_path();
        let mut existing = self
            .read_file_if_exists(&path)
            .await?
            .unwrap_or_default();

        let date = Local::now().date_naive().format("%Y-%m-%d");
        let entry = format!("\n- [{date}] {content}");

        if let Some(section_name) = section {
            // Try to find the section header and append after it
            let header = format!("## {section_name}");
            if let Some(pos) = existing.find(&header) {
                // Find the end of the header line
                let after_header = pos + header.len();
                let insert_pos = existing[after_header..]
                    .find('\n')
                    .map(|p| after_header + p)
                    .unwrap_or(existing.len());

                // Find where the next section starts (or end of file)
                let next_section = existing[insert_pos + 1..]
                    .find("\n## ")
                    .map(|p| insert_pos + 1 + p)
                    .unwrap_or(existing.len());

                // Insert before the next section
                existing.insert_str(next_section, &entry);
            } else {
                // Section doesn't exist — append a new one
                existing.push_str(&format!("\n\n## {section_name}\n{entry}\n"));
            }
        } else {
            // No section specified — append under a generic "Promoted" section
            let promoted_header = "## Promoted";
            if existing.contains(promoted_header) {
                let pos = existing.find(promoted_header).unwrap();
                let after_header = pos + promoted_header.len();
                let next_section = existing[after_header..]
                    .find("\n## ")
                    .map(|p| after_header + p)
                    .unwrap_or(existing.len());
                existing.insert_str(next_section, &entry);
            } else {
                existing.push_str(&format!("\n\n{promoted_header}\n{entry}\n"));
            }
        }

        fs::write(&path, &existing)
            .await
            .with_context(|| format!("Failed to write MEMORY.md: {}", path.display()))?;

        // Enforce section size caps after writing
        let gc = self.growth_controller();
        gc.enforce_section_caps().await?;

        Ok(())
    }

    // ── Flush: write session summary before exit ─────────────────

    /// Write a session summary to today's daily note.
    ///
    /// Called before session end to capture what was accomplished,
    /// what's pending, and any important context for the next session.
    pub async fn flush(&self, summary: &str) -> Result<()> {
        let entry = format!(
            "---\n\n## Session Summary\n\n{summary}\n\n---"
        );
        self.append_daily_note(&entry).await
    }

    // ── Rebuild helpers ───────────────────────────────────────────

    /// Load daily notes spanning a date range (inclusive).
    ///
    /// Returns the concatenated content of all daily notes in the range,
    /// or `None` if no daily notes exist for any date in the range.
    pub async fn load_daily_notes_range(
        &self,
        from: NaiveDate,
        to: NaiveDate,
    ) -> Result<Option<String>> {
        let mut combined = String::new();
        let mut date = from;
        while date <= to {
            let path = self.daily_note_path(date);
            if let Some(content) = self.read_file_if_exists(&path).await? {
                if !combined.is_empty() {
                    combined.push_str("\n\n---\n\n");
                }
                combined.push_str(&content);
            }
            date += chrono::Duration::days(1);
        }
        if combined.is_empty() {
            Ok(None)
        } else {
            Ok(Some(combined))
        }
    }

    /// Load raw sources for a full conversation rebuild.
    ///
    /// Returns `(daily_notes_content, memory_content)`. Either may be empty
    /// if the respective source files don't exist.
    pub async fn load_rebuild_sources(
        &self,
        session_created: NaiveDate,
    ) -> Result<(String, String)> {
        let today = Local::now().date_naive();
        let daily_notes = self
            .load_daily_notes_range(session_created, today)
            .await?
            .unwrap_or_default();

        let memory_content = self
            .read_file_if_exists(&self.memory_path())
            .await?
            .unwrap_or_default();

        Ok((daily_notes, memory_content))
    }

    // ── Utility helpers ──────────────────────────────────────────

    /// Read a file if it exists, returning None if it doesn't.
    async fn read_file_if_exists(&self, path: &Path) -> Result<Option<String>> {
        match fs::read_to_string(path).await {
            Ok(content) => Ok(Some(content)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e).with_context(|| format!("Failed to read: {}", path.display())),
        }
    }

    /// Ensure the `memory/` directory exists.
    async fn ensure_daily_dir(&self) -> Result<()> {
        let dir = self.daily_dir();
        if !dir.exists() {
            fs::create_dir_all(&dir)
                .await
                .with_context(|| format!("Failed to create memory dir: {}", dir.display()))?;
        }
        Ok(())
    }

    /// List all daily note files in the memory/ directory.
    async fn list_daily_notes(&self) -> Result<Vec<PathBuf>> {
        let dir = self.daily_dir();
        if !dir.exists() {
            return Ok(Vec::new());
        }

        let mut entries = fs::read_dir(&dir)
            .await
            .with_context(|| format!("Failed to read memory dir: {}", dir.display()))?;

        let mut files = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().map(|e| e == "md").unwrap_or(false) {
                files.push(path);
            }
        }

        Ok(files)
    }
}

// ── Implement the Memory trait for MarkdownMemory ────────────────
//
// This allows MarkdownMemory to be used as a drop-in replacement
// for SqliteMemory anywhere the Memory trait is expected.

use crate::memory::{Memory, MemoryCategory, MemoryEntry};
use async_trait::async_trait;

#[async_trait]
impl Memory for MarkdownMemory {
    /// Store a memory entry by appending it to today's daily note.
    async fn store(&self, key: &str, content: &str, category: &str) -> Result<()> {
        let entry = format!("**[{category}] {key}**: {content}");
        self.append_daily_note(&entry).await
    }

    /// Recall memories by searching across all markdown files.
    async fn recall(&self, query: &str) -> Result<Vec<MemoryEntry>> {
        let results = self.search(query).await?;
        Ok(results
            .into_iter()
            .map(|r| MemoryEntry {
                key: r.source.clone(),
                content: r.content,
                category: MemoryCategory::General,
                score: Some(1.0), // flat score for substring matches
            })
            .collect())
    }

    /// Get a specific entry by key — searches for exact key match.
    async fn get(&self, key: &str) -> Result<Option<String>> {
        let results = self.search(key).await?;
        Ok(results.into_iter().next().map(|r| r.content))
    }

    /// Forget is a no-op for append-only markdown memory.
    /// Daily notes are immutable logs. Use memory_promote to curate
    /// MEMORY.md, and manually edit MEMORY.md to remove entries.
    async fn forget(&self, _key: &str) -> Result<()> {
        // Append-only: log the intent but don't delete
        self.append_daily_note(&format!("_Forget requested for key: {_key} (no-op in markdown memory)_"))
            .await
    }

    /// List entries in a category — searches for the category tag.
    async fn list(&self, category: &str) -> Result<Vec<MemoryEntry>> {
        self.recall(category).await
    }
}

// ── Import for tokio::io::AsyncWriteExt ──────────────────────────
use tokio::io::AsyncWriteExt;

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn setup() -> (TempDir, MarkdownMemory) {
        let tmp = TempDir::new().unwrap();
        let mem = MarkdownMemory::new(tmp.path());

        // Create the memory/ subdirectory
        fs::create_dir_all(mem.daily_dir()).await.unwrap();

        // Write minimal SOUL.md and USER.md
        fs::write(mem.soul_path(), "# SOUL\nYou are ZeroClaw.\n")
            .await
            .unwrap();
        fs::write(mem.user_path(), "# USER\nName: Test User\n")
            .await
            .unwrap();
        fs::write(
            mem.memory_path(),
            "# MEMORY\n\n## Project Facts\n\n## Decisions\n",
        )
        .await
        .unwrap();

        (tmp, mem)
    }

    #[tokio::test]
    async fn load_session_context_includes_all_files() {
        let (_tmp, mem) = setup().await;

        let ctx = mem.load_session_context().await.unwrap();
        assert!(ctx.contains("SOUL (Agent Identity)"));
        assert!(ctx.contains("You are ZeroClaw"));
        assert!(ctx.contains("USER (Preferences & Context)"));
        assert!(ctx.contains("Test User"));
        assert!(ctx.contains("MEMORY (Long-Term Knowledge)"));
    }

    #[tokio::test]
    async fn append_daily_note_creates_and_appends() {
        let (_tmp, mem) = setup().await;

        mem.append_daily_note("First entry").await.unwrap();
        let content = fs::read_to_string(mem.today_note_path()).await.unwrap();
        assert!(content.contains("# Daily Note"));
        assert!(content.contains("First entry"));

        mem.append_daily_note("Second entry").await.unwrap();
        let content = fs::read_to_string(mem.today_note_path()).await.unwrap();
        assert!(content.contains("First entry"));
        assert!(content.contains("Second entry"));
    }

    #[tokio::test]
    async fn search_finds_matches_across_files() {
        let (_tmp, mem) = setup().await;

        mem.append_daily_note("The build system uses Cargo")
            .await
            .unwrap();
        mem.append_daily_note("Tests pass with cargo test")
            .await
            .unwrap();

        let results = mem.search("cargo").await.unwrap();
        assert!(results.len() >= 2);
        assert!(results.iter().any(|r| r.content.contains("Cargo")));
    }

    #[tokio::test]
    async fn promote_adds_to_memory_section() {
        let (_tmp, mem) = setup().await;

        mem.promote("Zeroclaw uses Rust 2021 edition", Some("Project Facts"))
            .await
            .unwrap();

        let content = fs::read_to_string(mem.memory_path()).await.unwrap();
        assert!(content.contains("Zeroclaw uses Rust 2021 edition"));
        assert!(content.contains("## Project Facts"));
    }

    #[tokio::test]
    async fn promote_creates_section_if_missing() {
        let (_tmp, mem) = setup().await;

        mem.promote("Always run clippy", Some("Coding Standards"))
            .await
            .unwrap();

        let content = fs::read_to_string(mem.memory_path()).await.unwrap();
        assert!(content.contains("## Coding Standards"));
        assert!(content.contains("Always run clippy"));
    }

    #[tokio::test]
    async fn flush_writes_session_summary() {
        let (_tmp, mem) = setup().await;

        mem.flush("Implemented markdown memory system. Tests passing.")
            .await
            .unwrap();

        let content = fs::read_to_string(mem.today_note_path()).await.unwrap();
        assert!(content.contains("Session Summary"));
        assert!(content.contains("Implemented markdown memory system"));
    }

    #[tokio::test]
    async fn search_returns_empty_for_no_matches() {
        let (_tmp, mem) = setup().await;
        let results = mem.search("xyznonexistent").await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn load_context_handles_missing_files() {
        let tmp = TempDir::new().unwrap();
        let mem = MarkdownMemory::new(tmp.path());

        // No files exist — should return empty context without error
        let ctx = mem.load_session_context().await.unwrap();
        assert!(ctx.is_empty());
    }

    #[tokio::test]
    async fn load_session_context_named_includes_session_name() {
        let (_tmp, mem) = setup().await;

        let ctx = mem
            .load_session_context_named(Some("scribe-hardware"))
            .await
            .unwrap();
        assert!(ctx.contains("=== Active Session: scribe-hardware ==="));
        assert!(ctx.contains("SOUL (Agent Identity)"));
    }

    #[tokio::test]
    async fn load_session_context_named_none_matches_unnamed() {
        let (_tmp, mem) = setup().await;

        let ctx_unnamed = mem.load_session_context().await.unwrap();
        let ctx_none = mem.load_session_context_named(None).await.unwrap();
        assert_eq!(ctx_unnamed, ctx_none);
    }

    // ── Rebuild source tests ──────────────────────────────────────

    #[tokio::test]
    async fn load_daily_notes_range_returns_content() {
        let (_tmp, mem) = setup().await;

        // Write a note for today
        mem.append_daily_note("Today's work notes").await.unwrap();

        let today = Local::now().date_naive();
        let result = mem.load_daily_notes_range(today, today).await.unwrap();
        assert!(result.is_some());
        assert!(result.unwrap().contains("Today's work notes"));
    }

    #[tokio::test]
    async fn load_daily_notes_range_returns_none_when_empty() {
        let (_tmp, mem) = setup().await;

        // No notes exist for the range
        let from = NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
        let to = NaiveDate::from_ymd_opt(2020, 1, 3).unwrap();
        let result = mem.load_daily_notes_range(from, to).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn load_rebuild_sources_returns_daily_notes_and_memory() {
        let (_tmp, mem) = setup().await;

        // Write a note for today
        mem.append_daily_note("Important decision made").await.unwrap();

        let today = Local::now().date_naive();
        let (daily, memory) = mem.load_rebuild_sources(today).await.unwrap();

        assert!(daily.contains("Important decision made"));
        assert!(memory.contains("MEMORY")); // from setup
    }

    #[tokio::test]
    async fn load_rebuild_sources_handles_missing_daily_notes() {
        let (_tmp, mem) = setup().await;

        // No daily notes for a past date range
        let old_date = NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
        let (daily, memory) = mem.load_rebuild_sources(old_date).await.unwrap();

        // daily_notes should still contain today's content (empty since no note for today)
        // but it should not fail; memory should still have content
        assert!(memory.contains("MEMORY"));
        // daily might be empty or contain today's note if it exists
        let _ = daily; // no panic
    }
}

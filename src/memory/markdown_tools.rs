//! Agent tools for the markdown memory system.
//!
//! These tools wrap `MarkdownMemory` methods and implement the `Tool` trait
//! so they can be registered in the agent loop alongside existing tools.
//!
//! Tools provided:
//! - `memory_store`   — append a note to today's daily log
//! - `memory_recall`  — search across all markdown memory files
//! - `memory_flush`   — write a session summary before exit
//! - `memory_promote` — move important content to long-term MEMORY.md

use crate::memory::markdown::MarkdownMemory;
use crate::memory::memory_growth::parse_memory_content;
use crate::memory::semantic_search::SemanticScorer;
use crate::tools::{Tool, ToolExecutionResult};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;

// ── memory_store ─────────────────────────────────────────────────

/// Tool: Append a note to today's daily log.
///
/// This replaces (or supplements) the SQLite `memory_store` tool.
/// Instead of writing to a database, it appends timestamped entries
/// to `memory/YYYY-MM-DD.md`.
pub struct MemoryStoreTool {
    memory: Arc<MarkdownMemory>,
}

impl MemoryStoreTool {
    pub fn new(memory: Arc<MarkdownMemory>) -> Self {
        Self { memory }
    }
}

#[async_trait]
impl Tool for MemoryStoreTool {
    fn name(&self) -> &str {
        "memory_store"
    }

    fn description(&self) -> &str {
        "Store a note in today's daily log. Use this to record important context, \
         decisions, discoveries, or anything worth remembering. Entries are timestamped \
         and appended to today's daily note (memory/YYYY-MM-DD.md)."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The note to store. Be specific and include enough context \
                                    to be useful when read later."
                },
                "category": {
                    "type": "string",
                    "description": "Optional category tag (e.g., 'decision', 'discovery', \
                                    'todo', 'bug', 'context'). Helps with later retrieval.",
                    "default": "note"
                }
            },
            "required": ["content"]
        })
    }

    async fn execute(&self, args: Value) -> Result<ToolExecutionResult> {
        let content = args["content"]
            .as_str()
            .unwrap_or("(empty)")
            .to_string();
        let category = args["category"]
            .as_str()
            .unwrap_or("note")
            .to_string();

        let entry = format!("**[{category}]** {content}");

        match self.memory.append_daily_note(&entry).await {
            Ok(()) => Ok(ToolExecutionResult {
                success: true,
                output: format!("Stored in today's daily note: [{category}] {content}"),
                error: None,
            }),
            Err(e) => Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to store note: {e}")),
            }),
        }
    }
}

// ── memory_recall ────────────────────────────────────────────────

/// Tool: Search across all markdown memory files.
///
/// Performs a two-phase search:
/// - **Phase 1** (keyword grep): case-insensitive substring search across
///   MEMORY.md, daily notes, SOUL.md, USER.md.
/// - **Phase 2** (semantic scoring): if enabled, sends Phase 1 results to
///   an LLM for relevance scoring (0-10). Only entries scoring 7+ are returned.
///
/// Phase 2 is skipped when: semantic search is disabled, there are 1-2 exact
/// token matches, or the scoring API fails (graceful degradation to Phase 1).
pub struct MemoryRecallTool {
    memory: Arc<MarkdownMemory>,
    /// Optional semantic scorer for Phase 2 LLM-based relevance scoring.
    scorer: Option<Arc<SemanticScorer>>,
}

impl MemoryRecallTool {
    pub fn new(memory: Arc<MarkdownMemory>) -> Self {
        Self {
            memory,
            scorer: None,
        }
    }

    /// Create a MemoryRecallTool with semantic scoring enabled.
    pub fn with_scorer(memory: Arc<MarkdownMemory>, scorer: Arc<SemanticScorer>) -> Self {
        Self {
            memory,
            scorer: Some(scorer),
        }
    }

    /// Update the reference tracker for entries matching a query.
    ///
    /// This tracks "last referenced" timestamps for stale entry scoring.
    async fn touch_matching_entries(&self, query: &str) -> Result<()> {
        let gc = self.memory.growth_controller();
        let memory_content = match tokio::fs::read_to_string(gc.memory_path()).await {
            Ok(c) => c,
            Err(_) => return Ok(()), // No MEMORY.md — nothing to track
        };

        let (_, sections) = parse_memory_content(&memory_content);
        let all_entries: Vec<_> = sections.iter()
            .flat_map(|s| s.entries())
            .collect();

        if all_entries.is_empty() {
            return Ok(());
        }

        let mut tracker = gc.load_refs().await?;
        tracker.touch_matching(&all_entries, query);
        gc.save_refs(&tracker).await?;

        Ok(())
    }

    /// Update the reference tracker for specific entry texts returned by Phase 2.
    async fn touch_scored_entries(&self, entry_texts: &[String]) -> Result<()> {
        let gc = self.memory.growth_controller();
        let mut tracker = gc.load_refs().await?;
        for text in entry_texts {
            tracker.touch(text);
        }
        gc.save_refs(&tracker).await?;
        Ok(())
    }

    /// Run Phase 2 semantic scoring on keyword grep results.
    ///
    /// Returns the formatted output with scored results, or None to fall
    /// back to Phase 1 results.
    async fn run_phase2_on_results(
        &self,
        query: &str,
        results: &[crate::memory::markdown::MarkdownSearchResult],
    ) -> Option<String> {
        let scorer = self.scorer.as_ref()?;
        if !scorer.is_enabled() {
            return None;
        }

        // Check if Phase 2 should be skipped (exact match optimization)
        let contents: Vec<&str> = results.iter().map(|r| r.content.as_str()).collect();
        if SemanticScorer::should_skip_phase2(query, &contents) {
            return None;
        }

        // Build entries text from Phase 1 results
        let entries_text: String = results
            .iter()
            .enumerate()
            .map(|(i, r)| format!("[{}] ({}:L{}) {}", i + 1, r.source, r.line_number, r.content))
            .collect::<Vec<_>>()
            .join("\n");

        // Score via LLM
        let scored = scorer.score_entries(query, &entries_text).await?;
        if scored.is_empty() {
            return None;
        }

        // Update reference tracker for scored entries
        let scored_texts: Vec<String> = scored.iter().map(|s| s.entry.clone()).collect();
        if let Err(e) = self.touch_scored_entries(&scored_texts).await {
            eprintln!("[memory_recall] Failed to update reference tracker for scored entries: {e}");
        }

        // Format scored results
        let mut output = format!(
            "Found {} semantically relevant match(es) for \"{query}\" (scored 7+/10):\n\n",
            scored.len()
        );
        for (i, entry) in scored.iter().enumerate() {
            output.push_str(&format!(
                "--- Match {} (score: {}/10) ---\n{}\n\n",
                i + 1,
                entry.score,
                entry.entry
            ));
        }
        Some(output)
    }

    /// Run Phase 2 manifest-based search when keyword grep returns nothing.
    ///
    /// Sends the MEMORY.md manifest to the LLM to identify relevant topics,
    /// then loads and scores those topic files.
    async fn run_phase2_manifest_fallback(&self, query: &str) -> Option<String> {
        let scorer = self.scorer.as_ref()?;
        if !scorer.is_enabled() {
            return None;
        }

        // Load the manifest/index from MEMORY.md
        let manifest = self.memory.load_manifest().await.ok()?;
        if manifest.is_empty() {
            return None;
        }

        // Ask LLM to identify relevant topics from the manifest
        let relevant_topics = scorer.identify_relevant_topics(query, &manifest).await?;
        if relevant_topics.is_empty() {
            return None;
        }

        // For each relevant topic, try to load its file and score entries
        let mut all_topic_entries = String::new();
        for topic in &relevant_topics {
            // The "entry" field from manifest scoring might contain the filename
            // Try to extract it
            let topic_file = extract_topic_filename(&topic.entry);
            if let Some(filename) = topic_file {
                if let Ok(Some(content)) = self.memory.load_topic_file(&filename).await {
                    if !all_topic_entries.is_empty() {
                        all_topic_entries.push('\n');
                    }
                    all_topic_entries.push_str(&content);
                }
            }
        }

        if all_topic_entries.is_empty() {
            return None;
        }

        // Score the topic file entries
        let scored = scorer.score_entries(query, &all_topic_entries).await?;
        if scored.is_empty() {
            return None;
        }

        // Update reference tracker for scored entries
        let scored_texts: Vec<String> = scored.iter().map(|s| s.entry.clone()).collect();
        if let Err(e) = self.touch_scored_entries(&scored_texts).await {
            eprintln!("[memory_recall] Failed to update reference tracker for scored entries: {e}");
        }

        let mut output = format!(
            "Found {} semantically relevant match(es) for \"{query}\" from archived topics (scored 7+/10):\n\n",
            scored.len()
        );
        for (i, entry) in scored.iter().enumerate() {
            output.push_str(&format!(
                "--- Match {} (score: {}/10) ---\n{}\n\n",
                i + 1,
                entry.score,
                entry.entry
            ));
        }
        Some(output)
    }
}

#[async_trait]
impl Tool for MemoryRecallTool {
    fn name(&self) -> &str {
        "memory_recall"
    }

    fn description(&self) -> &str {
        "Search your memory (daily notes, long-term memory, and context files) \
         for information matching a query. Uses case-insensitive substring matching. \
         Returns matching lines with surrounding context and source file names."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Can be a keyword, phrase, or topic. \
                                    Case-insensitive substring matching is used."
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, args: Value) -> Result<ToolExecutionResult> {
        let query = args["query"]
            .as_str()
            .unwrap_or("")
            .to_string();

        if query.is_empty() {
            return Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some("Query cannot be empty.".to_string()),
            });
        }

        // ── Phase 1: Keyword grep ────────────────────────────────
        match self.memory.search(&query).await {
            Ok(results) => {
                // Track references for stale entry scoring (Phase 1)
                if let Err(e) = self.touch_matching_entries(&query).await {
                    eprintln!("[memory_recall] Failed to update reference tracker: {e}");
                }

                if results.is_empty() {
                    // Phase 1 returned nothing — try Phase 2 manifest fallback
                    if let Some(phase2_output) = self.run_phase2_manifest_fallback(&query).await {
                        return Ok(ToolExecutionResult {
                            success: true,
                            output: phase2_output,
                            error: None,
                        });
                    }
                    Ok(ToolExecutionResult {
                        success: true,
                        output: format!("No matches found for: \"{query}\""),
                        error: None,
                    })
                } else {
                    // ── Phase 2: Semantic scoring (if enabled) ───
                    // Try Phase 2 scoring on keyword grep results.
                    // Falls back to Phase 1 results on failure.
                    if let Some(phase2_output) = self.run_phase2_on_results(&query, &results).await {
                        return Ok(ToolExecutionResult {
                            success: true,
                            output: phase2_output,
                            error: None,
                        });
                    }

                    // Phase 2 skipped or failed — return Phase 1 results
                    let mut output = format!("Found {} match(es) for \"{query}\":\n\n", results.len());
                    for (i, result) in results.iter().enumerate() {
                        output.push_str(&format!(
                            "--- Match {} ({}:L{}) ---\n{}\n\n",
                            i + 1,
                            result.source,
                            result.line_number,
                            result.content
                        ));
                    }
                    Ok(ToolExecutionResult {
                        success: true,
                        output,
                        error: None,
                    })
                }
            }
            Err(e) => Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some(format!("Search failed: {e}")),
            }),
        }
    }
}

// ── memory_flush ─────────────────────────────────────────────────

/// Tool: Write a session summary before session end.
///
/// Captures what was accomplished, what's pending, and any important
/// context for the next session. Should be called before the session
/// closes to preserve continuity.
pub struct MemoryFlushTool {
    memory: Arc<MarkdownMemory>,
}

impl MemoryFlushTool {
    pub fn new(memory: Arc<MarkdownMemory>) -> Self {
        Self { memory }
    }
}

#[async_trait]
impl Tool for MemoryFlushTool {
    fn name(&self) -> &str {
        "memory_flush"
    }

    fn description(&self) -> &str {
        "Write a session summary to today's daily note before the session ends. \
         Include: (1) what was accomplished, (2) what's still pending, \
         (3) any important context or decisions for the next session. \
         Call this before ending a session to preserve continuity."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "accomplished": {
                    "type": "string",
                    "description": "Summary of what was accomplished in this session."
                },
                "pending": {
                    "type": "string",
                    "description": "What remains to be done or any blockers."
                },
                "context": {
                    "type": "string",
                    "description": "Important context or decisions for the next session."
                }
            },
            "required": ["accomplished"]
        })
    }

    async fn execute(&self, args: Value) -> Result<ToolExecutionResult> {
        let accomplished = args["accomplished"]
            .as_str()
            .unwrap_or("(no summary provided)")
            .to_string();
        let pending = args["pending"].as_str().unwrap_or("").to_string();
        let context = args["context"].as_str().unwrap_or("").to_string();

        let mut summary = format!("**Accomplished:** {accomplished}");
        if !pending.is_empty() {
            summary.push_str(&format!("\n\n**Pending:** {pending}"));
        }
        if !context.is_empty() {
            summary.push_str(&format!("\n\n**Context for next session:** {context}"));
        }

        match self.memory.flush(&summary).await {
            Ok(()) => Ok(ToolExecutionResult {
                success: true,
                output: "Session summary written to today's daily note.".to_string(),
                error: None,
            }),
            Err(e) => Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to flush session summary: {e}")),
            }),
        }
    }
}

// ── memory_promote ───────────────────────────────────────────────

/// Tool: Promote important information to long-term MEMORY.md.
///
/// Moves significant facts, decisions, patterns, or issues from
/// daily notes into the curated MEMORY.md file for permanent retention.
pub struct MemoryPromoteTool {
    memory: Arc<MarkdownMemory>,
}

impl MemoryPromoteTool {
    pub fn new(memory: Arc<MarkdownMemory>) -> Self {
        Self { memory }
    }
}

#[async_trait]
impl Tool for MemoryPromoteTool {
    fn name(&self) -> &str {
        "memory_promote"
    }

    fn description(&self) -> &str {
        "Promote important information to long-term memory (MEMORY.md). \
         Use this to preserve key facts, decisions, patterns, or known issues \
         that should persist beyond daily notes. Specify which section to file \
         the entry under: 'Project Facts', 'Decisions', 'Patterns', or 'Known Issues'."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to promote to long-term memory. \
                                    Be concise but complete."
                },
                "section": {
                    "type": "string",
                    "description": "Which MEMORY.md section to file under.",
                    "enum": ["Project Facts", "Decisions", "Patterns", "Known Issues"]
                }
            },
            "required": ["content", "section"]
        })
    }

    async fn execute(&self, args: Value) -> Result<ToolExecutionResult> {
        let content = args["content"]
            .as_str()
            .unwrap_or("(empty)")
            .to_string();
        let section = args["section"].as_str().map(|s| s.to_string());

        match self
            .memory
            .promote(&content, section.as_deref())
            .await
        {
            Ok(()) => {
                let section_name = section.as_deref().unwrap_or("Promoted");
                Ok(ToolExecutionResult {
                    success: true,
                    output: format!(
                        "Promoted to MEMORY.md under \"{section_name}\": {content}"
                    ),
                    error: None,
                })
            }
            Err(e) => Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to promote: {e}")),
            }),
        }
    }
}

// ── Tool registration helper ─────────────────────────────────────

/// Create all markdown memory tools, ready to register in the agent loop.
///
/// Usage in `tools::all_tools()`:
/// ```rust
/// let md_mem = Arc::new(MarkdownMemory::new(&config.workspace));
/// let mut tools = tools::all_tools(&security, mem.clone(), composio_key, &config.browser);
/// tools.extend(markdown_tools::all_markdown_tools(md_mem, None));
/// ```
pub fn all_markdown_tools(
    memory: Arc<MarkdownMemory>,
    scorer: Option<Arc<SemanticScorer>>,
) -> Vec<Box<dyn Tool>> {
    let recall_tool: Box<dyn Tool> = match scorer {
        Some(s) => Box::new(MemoryRecallTool::with_scorer(memory.clone(), s)),
        None => Box::new(MemoryRecallTool::new(memory.clone())),
    };
    vec![
        Box::new(MemoryStoreTool::new(memory.clone())),
        recall_tool,
        Box::new(MemoryFlushTool::new(memory.clone())),
        Box::new(MemoryPromoteTool::new(memory)),
    ]
}

/// Extract a topic filename from a manifest entry string.
///
/// Manifest entries look like: `**filename.md** (N entries): summary`
/// This extracts the filename portion.
fn extract_topic_filename(entry: &str) -> Option<String> {
    // Try to find **filename.md** pattern
    if let Some(start) = entry.find("**") {
        let rest = &entry[start + 2..];
        if let Some(end) = rest.find("**") {
            let filename = &rest[..end];
            if filename.ends_with(".md") {
                return Some(filename.to_string());
            }
        }
    }
    // Fall back: if the entry itself looks like a filename
    let trimmed = entry.trim();
    if trimmed.ends_with(".md") && !trimmed.contains(' ') {
        return Some(trimmed.to_string());
    }
    None
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tokio::fs;

    async fn setup() -> (TempDir, Arc<MarkdownMemory>) {
        let tmp = TempDir::new().unwrap();
        let mem = Arc::new(MarkdownMemory::new(tmp.path()));

        fs::create_dir_all(mem.daily_dir()).await.unwrap();
        fs::write(mem.soul_path(), "# SOUL\nYou are ZeroClaw.\n")
            .await
            .unwrap();
        fs::write(mem.user_path(), "# USER\nName: Test User\n")
            .await
            .unwrap();
        fs::write(
            mem.memory_path(),
            "# MEMORY\n\n## Project Facts\n\n## Decisions\n\n## Patterns\n\n## Known Issues\n",
        )
        .await
        .unwrap();

        (tmp, mem)
    }

    #[tokio::test]
    async fn memory_store_tool_appends_to_daily_note() {
        let (_tmp, mem) = setup().await;
        let tool = MemoryStoreTool::new(mem.clone());

        let result = tool
            .execute(json!({"content": "Found a bug in parser", "category": "bug"}))
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("bug"));

        let content = fs::read_to_string(mem.today_note_path()).await.unwrap();
        assert!(content.contains("Found a bug in parser"));
        assert!(content.contains("[bug]"));
    }

    #[tokio::test]
    async fn memory_recall_tool_searches_files() {
        let (_tmp, mem) = setup().await;

        // Store some content first
        mem.append_daily_note("The Rust compiler found 3 errors")
            .await
            .unwrap();

        let tool = MemoryRecallTool::new(mem.clone());
        let result = tool
            .execute(json!({"query": "Rust compiler"}))
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("Rust compiler"));
    }

    #[tokio::test]
    async fn memory_recall_tool_handles_no_matches() {
        let (_tmp, mem) = setup().await;
        let tool = MemoryRecallTool::new(mem.clone());

        let result = tool
            .execute(json!({"query": "nonexistent_xyz"}))
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("No matches"));
    }

    #[tokio::test]
    async fn memory_flush_tool_writes_summary() {
        let (_tmp, mem) = setup().await;
        let tool = MemoryFlushTool::new(mem.clone());

        let result = tool
            .execute(json!({
                "accomplished": "Implemented markdown memory",
                "pending": "Need to add vector search",
                "context": "SQLite backend still works as fallback"
            }))
            .await
            .unwrap();
        assert!(result.success);

        let content = fs::read_to_string(mem.today_note_path()).await.unwrap();
        assert!(content.contains("Session Summary"));
        assert!(content.contains("Implemented markdown memory"));
        assert!(content.contains("Need to add vector search"));
    }

    #[tokio::test]
    async fn memory_promote_tool_adds_to_section() {
        let (_tmp, mem) = setup().await;
        let tool = MemoryPromoteTool::new(mem.clone());

        let result = tool
            .execute(json!({
                "content": "Project uses Rust 2021 edition",
                "section": "Project Facts"
            }))
            .await
            .unwrap();
        assert!(result.success);

        let content = fs::read_to_string(mem.memory_path()).await.unwrap();
        assert!(content.contains("Rust 2021 edition"));
    }

    #[tokio::test]
    async fn all_markdown_tools_returns_four_tools() {
        let (_tmp, mem) = setup().await;
        let tools = all_markdown_tools(mem, None);
        assert_eq!(tools.len(), 4);

        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"memory_store"));
        assert!(names.contains(&"memory_recall"));
        assert!(names.contains(&"memory_flush"));
        assert!(names.contains(&"memory_promote"));
    }

    #[tokio::test]
    async fn all_markdown_tools_with_scorer_returns_four_tools() {
        let (_tmp, mem) = setup().await;
        let scorer = Arc::new(SemanticScorer::new(
            crate::memory::semantic_search::SemanticSearchConfig {
                enabled: true,
                scoring_model: String::new(),
                provider: "anthropic".to_string(),
                primary_model: "claude-sonnet-4-20250514".to_string(),
                api_key: "test".to_string(),
                api_base_url: None,
            },
        ));
        let tools = all_markdown_tools(mem, Some(scorer));
        assert_eq!(tools.len(), 4);
    }

    #[tokio::test]
    async fn memory_recall_without_scorer_returns_phase1_results() {
        let (_tmp, mem) = setup().await;
        mem.append_daily_note("Docker sandbox is configured")
            .await
            .unwrap();

        let tool = MemoryRecallTool::new(mem);
        let result = tool.execute(json!({"query": "Docker"})).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("Docker"));
        // Should be Phase 1 format (no score mentions)
        assert!(!result.output.contains("score:"));
    }

    #[tokio::test]
    async fn memory_recall_with_disabled_scorer_returns_phase1() {
        let (_tmp, mem) = setup().await;
        mem.append_daily_note("Docker sandbox is configured")
            .await
            .unwrap();

        let scorer = Arc::new(SemanticScorer::new(
            crate::memory::semantic_search::SemanticSearchConfig {
                enabled: false,
                scoring_model: String::new(),
                provider: "anthropic".to_string(),
                primary_model: "claude-sonnet-4-20250514".to_string(),
                api_key: "test".to_string(),
                api_base_url: None,
            },
        ));
        let tool = MemoryRecallTool::with_scorer(mem, scorer);
        let result = tool.execute(json!({"query": "Docker"})).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("Docker"));
        // Disabled scorer should fall back to Phase 1
        assert!(!result.output.contains("score:"));
    }

    #[test]
    fn test_extract_topic_filename_markdown_bold() {
        assert_eq!(
            extract_topic_filename("**docker-setup.md** (5 entries): Docker configuration"),
            Some("docker-setup.md".to_string())
        );
    }

    #[test]
    fn test_extract_topic_filename_plain() {
        assert_eq!(
            extract_topic_filename("docker-setup.md"),
            Some("docker-setup.md".to_string())
        );
    }

    #[test]
    fn test_extract_topic_filename_no_match() {
        assert_eq!(extract_topic_filename("just some text"), None);
    }
}

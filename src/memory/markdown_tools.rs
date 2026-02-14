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
use crate::tools::Tool;
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;

/// Result type returned by tool execution.
pub struct ToolExecutionResult {
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
}

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
/// Performs case-insensitive substring search across MEMORY.md,
/// all daily notes, SOUL.md, and USER.md. Returns matching lines
/// with context.
pub struct MemoryRecallTool {
    memory: Arc<MarkdownMemory>,
}

impl MemoryRecallTool {
    pub fn new(memory: Arc<MarkdownMemory>) -> Self {
        Self { memory }
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

        match self.memory.search(&query).await {
            Ok(results) => {
                if results.is_empty() {
                    Ok(ToolExecutionResult {
                        success: true,
                        output: format!("No matches found for: \"{query}\""),
                        error: None,
                    })
                } else {
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
/// tools.extend(markdown_tools::all_markdown_tools(md_mem));
/// ```
pub fn all_markdown_tools(memory: Arc<MarkdownMemory>) -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(MemoryStoreTool::new(memory.clone())),
        Box::new(MemoryRecallTool::new(memory.clone())),
        Box::new(MemoryFlushTool::new(memory.clone())),
        Box::new(MemoryPromoteTool::new(memory)),
    ]
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
        let tools = all_markdown_tools(mem);
        assert_eq!(tools.len(), 4);

        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"memory_store"));
        assert!(names.contains(&"memory_recall"));
        assert!(names.contains(&"memory_flush"));
        assert!(names.contains(&"memory_promote"));
    }
}

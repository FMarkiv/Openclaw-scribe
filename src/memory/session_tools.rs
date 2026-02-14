//! Agent tools for session persistence.
//!
//! These tools wrap `SessionManager` and implement the `Tool` trait
//! so they can be registered in the agent loop alongside existing tools.
//!
//! Tools provided:
//! - `session_new`      — start a fresh session (the `/new` command)
//! - `session_list`     — list recent sessions (the `/sessions` command)

use crate::memory::session::{SessionInfo, SessionManager};
use crate::tools::Tool;
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Result type returned by tool execution (matches markdown_tools convention).
pub struct ToolExecutionResult {
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
}

// ── session_new ──────────────────────────────────────────────────

/// Tool: Start a fresh session without loading previous history.
///
/// This implements the `/new` command. It creates a new session file
/// and clears the current conversation context so the agent starts
/// fresh. Memory files (SOUL.md, USER.md, MEMORY.md) are still loaded.
pub struct SessionNewTool {
    session_mgr: Arc<Mutex<SessionManager>>,
}

impl SessionNewTool {
    pub fn new(session_mgr: Arc<Mutex<SessionManager>>) -> Self {
        Self { session_mgr }
    }
}

#[async_trait]
impl Tool for SessionNewTool {
    fn name(&self) -> &str {
        "session_new"
    }

    fn description(&self) -> &str {
        "Start a fresh conversation session. Creates a new session file and clears \
         the current conversation history. Memory files (SOUL.md, USER.md, MEMORY.md) \
         remain loaded. Use this when you want a clean slate without previous context. \
         This is the /new command."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }

    async fn execute(&self, _args: Value) -> Result<ToolExecutionResult> {
        let mut mgr = self.session_mgr.lock().await;
        match mgr.new_session().await {
            Ok(id) => Ok(ToolExecutionResult {
                success: true,
                output: format!(
                    "Started fresh session: {id}\n\
                     Previous conversation history cleared.\n\
                     Memory files (SOUL.md, USER.md, MEMORY.md) remain available."
                ),
                error: None,
            }),
            Err(e) => Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to create new session: {e}")),
            }),
        }
    }
}

// ── session_list ─────────────────────────────────────────────────

/// Tool: List recent sessions with date and message count.
///
/// This implements the `/sessions` command. Shows the most recent
/// sessions so the user can see their conversation history.
pub struct SessionListTool {
    session_mgr: Arc<Mutex<SessionManager>>,
}

impl SessionListTool {
    pub fn new(session_mgr: Arc<Mutex<SessionManager>>) -> Self {
        Self { session_mgr }
    }
}

#[async_trait]
impl Tool for SessionListTool {
    fn name(&self) -> &str {
        "session_list"
    }

    fn description(&self) -> &str {
        "List recent conversation sessions with their dates, message counts, \
         and session IDs. Shows the most recent sessions first. Use this to \
         review past sessions or find a session to reference. \
         This is the /sessions command."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of sessions to list (default: 10, max: 50).",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50
                }
            },
            "required": []
        })
    }

    async fn execute(&self, args: Value) -> Result<ToolExecutionResult> {
        let limit = args["limit"]
            .as_u64()
            .unwrap_or(10)
            .min(50) as usize;

        let mgr = self.session_mgr.lock().await;
        match mgr.list_sessions(limit).await {
            Ok(sessions) => {
                if sessions.is_empty() {
                    return Ok(ToolExecutionResult {
                        success: true,
                        output: "No sessions found.".to_string(),
                        error: None,
                    });
                }

                let mut output = format!("Recent sessions ({} found):\n\n", sessions.len());
                output.push_str(&format!(
                    "{:<12} {:<10} {:>6}  {}\n",
                    "DATE", "SESSION", "TURNS", "FILE"
                ));
                output.push_str(&format!("{}\n", "-".repeat(60)));

                for info in &sessions {
                    let filename = info
                        .path
                        .file_name()
                        .and_then(|f| f.to_str())
                        .unwrap_or("unknown");
                    output.push_str(&format!(
                        "{:<12} {:<10} {:>6}  {}\n",
                        info.date.format("%Y-%m-%d"),
                        &info.session_id,
                        info.turn_count,
                        filename,
                    ));
                }

                // Add current session indicator
                let current_id = mgr.session_id();
                if let Some(id) = current_id {
                    output.push_str(&format!("\nActive session: {id}"));
                }

                Ok(ToolExecutionResult {
                    success: true,
                    output,
                    error: None,
                })
            }
            Err(e) => Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to list sessions: {e}")),
            }),
        }
    }
}

// ── Tool registration helper ─────────────────────────────────────

/// Create all session persistence tools, ready to register in the agent loop.
///
/// Usage:
/// ```rust
/// let session_mgr = Arc::new(Mutex::new(SessionManager::new(&config.workspace)));
/// let mut tools = tools::all_tools(...);
/// tools.extend(session_tools::all_session_tools(session_mgr.clone()));
/// ```
pub fn all_session_tools(session_mgr: Arc<Mutex<SessionManager>>) -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(SessionNewTool::new(session_mgr.clone())),
        Box::new(SessionListTool::new(session_mgr)),
    ]
}

// ── Formatting helper ────────────────────────────────────────────

/// Format a session info for display (used by startup resume prompt).
pub fn format_session_summary(info: &SessionInfo) -> String {
    format!(
        "Session {} from {} — {} turn(s)",
        info.session_id,
        info.date.format("%Y-%m-%d"),
        info.turn_count,
    )
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::session::SessionManager;
    use tempfile::TempDir;

    async fn setup() -> (TempDir, Arc<Mutex<SessionManager>>) {
        let tmp = TempDir::new().unwrap();
        let mgr = Arc::new(Mutex::new(SessionManager::new(tmp.path())));
        (tmp, mgr)
    }

    #[tokio::test]
    async fn session_new_tool_creates_session() {
        let (_tmp, mgr) = setup().await;
        let tool = SessionNewTool::new(mgr.clone());

        let result = tool.execute(json!({})).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("Started fresh session"));

        // Verify session was created
        let locked = mgr.lock().await;
        assert!(locked.session_id().is_some());
        assert!(locked.session_path().is_some());
    }

    #[tokio::test]
    async fn session_list_tool_empty() {
        let (_tmp, mgr) = setup().await;
        let tool = SessionListTool::new(mgr.clone());

        let result = tool.execute(json!({})).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("No sessions found"));
    }

    #[tokio::test]
    async fn session_list_tool_shows_sessions() {
        let (_tmp, mgr) = setup().await;

        // Create a session with some turns
        {
            let mut locked = mgr.lock().await;
            locked.new_session().await.unwrap();
            let turn = SessionManager::user_turn("Hello");
            locked.append_turn(&turn).await.unwrap();
            let turn = SessionManager::assistant_turn(Some("Hi there!"), vec![]);
            locked.append_turn(&turn).await.unwrap();
        }

        let tool = SessionListTool::new(mgr.clone());
        let result = tool.execute(json!({})).await.unwrap();

        assert!(result.success);
        assert!(result.output.contains("1 found"));
        assert!(result.output.contains("2")); // 2 turns
    }

    #[tokio::test]
    async fn session_list_tool_respects_limit() {
        let (_tmp, mgr) = setup().await;

        // Create 3 sessions
        for _ in 0..3 {
            let mut locked = mgr.lock().await;
            locked.new_session().await.unwrap();
            let turn = SessionManager::user_turn("Hello");
            locked.append_turn(&turn).await.unwrap();
        }

        let tool = SessionListTool::new(mgr.clone());
        let result = tool.execute(json!({"limit": 2})).await.unwrap();

        assert!(result.success);
        assert!(result.output.contains("2 found"));
    }

    #[tokio::test]
    async fn all_session_tools_returns_two_tools() {
        let (_tmp, mgr) = setup().await;
        let tools = all_session_tools(mgr);
        assert_eq!(tools.len(), 2);

        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"session_new"));
        assert!(names.contains(&"session_list"));
    }

    #[tokio::test]
    async fn format_session_summary_produces_readable_output() {
        use chrono::Local;
        let info = SessionInfo {
            session_id: "abc12345".to_string(),
            date: Local::now().date_naive(),
            turn_count: 42,
            path: std::path::PathBuf::from("/tmp/test.jsonl"),
        };

        let summary = format_session_summary(&info);
        assert!(summary.contains("abc12345"));
        assert!(summary.contains("42 turn(s)"));
    }
}

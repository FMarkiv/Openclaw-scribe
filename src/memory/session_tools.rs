//! Agent tools for session persistence.
//!
//! These tools wrap `SessionManager` and implement the `Tool` trait
//! so they can be registered in the agent loop alongside existing tools.
//!
//! Tools provided:
//! - `session_new`      — start a fresh named session (the `/new` command)
//! - `session_switch`   — switch to an existing session (the `/switch` command)
//! - `session_list`     — list all sessions (the `/sessions` command)
//! - `session_rename`   — rename the current session (the `/rename` command)
//!
//! ## Local command parsing
//!
//! The `parse_session_command()` function intercepts user input before
//! it reaches the LLM, handling `/new`, `/switch`, `/sessions`, and
//! `/rename` as local commands that manipulate session state directly.

use crate::memory::session::{SessionInfo, SessionManager, slugify};
use crate::tools::{Tool, ToolExecutionResult};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Mutex;

// ── Local command parsing ────────────────────────────────────────

/// A parsed session command intercepted from user input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionCommand {
    /// /new <name> — create a new named session
    New(String),
    /// /switch <name> — switch to an existing session
    Switch(String),
    /// /sessions — list all sessions
    List,
    /// /rename <name> — rename the current session
    Rename(String),
}

/// Parse user input for session commands.
///
/// Returns `Some(SessionCommand)` if the input is a session command,
/// `None` if it should be passed through to the LLM.
///
/// Commands are case-insensitive and the argument is slugified.
pub fn parse_session_command(input: &str) -> Option<SessionCommand> {
    let trimmed = input.trim();

    if trimmed.eq_ignore_ascii_case("/sessions") {
        return Some(SessionCommand::List);
    }

    if let Some(rest) = strip_command_prefix(trimmed, "/new") {
        let name = rest.trim();
        if name.is_empty() {
            return None; // /new without a name is not a session command
        }
        return Some(SessionCommand::New(name.to_string()));
    }

    if let Some(rest) = strip_command_prefix(trimmed, "/switch") {
        let name = rest.trim();
        if name.is_empty() {
            return None;
        }
        return Some(SessionCommand::Switch(name.to_string()));
    }

    if let Some(rest) = strip_command_prefix(trimmed, "/rename") {
        let name = rest.trim();
        if name.is_empty() {
            return None;
        }
        return Some(SessionCommand::Rename(name.to_string()));
    }

    None
}

/// Strip a command prefix (case-insensitive) and return the rest.
fn strip_command_prefix<'a>(input: &'a str, prefix: &str) -> Option<&'a str> {
    let lower = input.to_lowercase();
    if lower.starts_with(prefix) {
        let rest = &input[prefix.len()..];
        // Must be followed by whitespace or end of string
        if rest.is_empty() || rest.starts_with(char::is_whitespace) {
            return Some(rest);
        }
    }
    None
}

/// Execute a parsed session command against the session manager.
///
/// Returns a human-readable response string to display to the user.
pub async fn execute_session_command(
    cmd: &SessionCommand,
    session_mgr: &Arc<Mutex<SessionManager>>,
) -> Result<String> {
    match cmd {
        SessionCommand::New(name) => {
            let mut mgr = session_mgr.lock().await;
            let slug = mgr.new_named_session(name).await?;
            Ok(format!(
                "Created new session: {slug}\n\
                 Previous conversation history cleared.\n\
                 Memory files (SOUL.md, USER.md, MEMORY.md) remain available."
            ))
        }
        SessionCommand::Switch(name) => {
            let slug = slugify(name);
            let mut mgr = session_mgr.lock().await;
            let turns = mgr.switch_session(&slug).await?;
            Ok(format!(
                "Switched to session: {slug}\n\
                 Loaded {} turn(s) from history.",
                turns.len()
            ))
        }
        SessionCommand::List => {
            let mgr = session_mgr.lock().await;
            let sessions = mgr.list_sessions(50).await?;

            if sessions.is_empty() {
                return Ok("No sessions found.".to_string());
            }

            let current_name = mgr.session_name().map(|s| s.to_string());

            let mut output = format!("Sessions ({} found):\n\n", sessions.len());
            output.push_str(&format!(
                "  {:<20} {:>6}  {}\n",
                "NAME", "TURNS", "LAST ACTIVE"
            ));
            output.push_str(&format!("  {}\n", "-".repeat(56)));

            for info in &sessions {
                let marker = if current_name.as_deref() == Some(&info.name) {
                    "* "
                } else {
                    "  "
                };
                let last_active_short = truncate_timestamp(&info.last_active);
                output.push_str(&format!(
                    "{}{:<20} {:>6}  {}\n",
                    marker, info.name, info.turn_count, last_active_short,
                ));
            }

            if let Some(name) = &current_name {
                output.push_str(&format!("\n* = active session ({name})"));
            }

            Ok(output)
        }
        SessionCommand::Rename(new_name) => {
            let mut mgr = session_mgr.lock().await;
            let new_slug = mgr.rename_session(new_name).await?;
            Ok(format!("Session renamed to: {new_slug}"))
        }
    }
}

/// Truncate an ISO-8601 timestamp to just date + time for display.
fn truncate_timestamp(ts: &str) -> &str {
    // RFC 3339: "2026-02-15T10:30:00+00:00" → take first 19 chars
    if ts.len() >= 19 {
        &ts[..19]
    } else {
        ts
    }
}

// ── session_new ──────────────────────────────────────────────────

/// Tool: Start a fresh named session.
///
/// This implements the `/new` command. It creates a new session directory
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
        "Start a fresh named conversation session. Creates a new session directory \
         and clears the current conversation history. Memory files (SOUL.md, USER.md, \
         MEMORY.md) remain loaded. Provide a name for the session (will be slugified). \
         Use this when you want a clean slate without previous context. \
         This is the /new command."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name for the new session (will be slugified to lowercase-hyphens)."
                }
            },
            "required": []
        })
    }

    async fn execute(&self, args: Value) -> Result<ToolExecutionResult> {
        let name = args["name"].as_str().unwrap_or("");

        let mut mgr = self.session_mgr.lock().await;
        let result = if name.is_empty() {
            mgr.new_session().await
        } else {
            mgr.new_named_session(name).await
        };

        match result {
            Ok(slug) => Ok(ToolExecutionResult {
                success: true,
                output: format!(
                    "Started fresh session: {slug}\n\
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

// ── session_switch ──────────────────────────────────────────────

/// Tool: Switch to an existing named session.
///
/// This implements the `/switch` command. Saves the current session
/// and loads the target session's history.
pub struct SessionSwitchTool {
    session_mgr: Arc<Mutex<SessionManager>>,
}

impl SessionSwitchTool {
    pub fn new(session_mgr: Arc<Mutex<SessionManager>>) -> Self {
        Self { session_mgr }
    }
}

#[async_trait]
impl Tool for SessionSwitchTool {
    fn name(&self) -> &str {
        "session_switch"
    }

    fn description(&self) -> &str {
        "Switch to an existing named session. Saves the current session first, \
         then loads the target session's history and memory context. \
         This is the /switch command."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the session to switch to."
                }
            },
            "required": ["name"]
        })
    }

    async fn execute(&self, args: Value) -> Result<ToolExecutionResult> {
        let name = match args["name"].as_str() {
            Some(n) => n,
            None => {
                return Ok(ToolExecutionResult {
                    success: false,
                    output: String::new(),
                    error: Some("Session name is required.".to_string()),
                });
            }
        };

        let mut mgr = self.session_mgr.lock().await;
        match mgr.switch_session(name).await {
            Ok(turns) => Ok(ToolExecutionResult {
                success: true,
                output: format!(
                    "Switched to session: {}\nLoaded {} turn(s) from history.",
                    slugify(name),
                    turns.len()
                ),
                error: None,
            }),
            Err(e) => Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to switch session: {e}")),
            }),
        }
    }
}

// ── session_list ─────────────────────────────────────────────────

/// Tool: List all sessions with metadata.
///
/// This implements the `/sessions` command. Shows all sessions with
/// name, last active timestamp, turn count, and a marker for the
/// currently active session.
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
        "List all conversation sessions with their names, last active timestamps, \
         turn counts, and a marker for the currently active session. \
         This is the /sessions command."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of sessions to list (default: 50, max: 100).",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 100
                }
            },
            "required": []
        })
    }

    async fn execute(&self, args: Value) -> Result<ToolExecutionResult> {
        let limit = args["limit"]
            .as_u64()
            .unwrap_or(50)
            .min(100) as usize;

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

                let current_name = mgr.session_name().map(|s| s.to_string());

                let mut output = format!("Sessions ({} found):\n\n", sessions.len());
                output.push_str(&format!(
                    "  {:<20} {:>6}  {}\n",
                    "NAME", "TURNS", "LAST ACTIVE"
                ));
                output.push_str(&format!("  {}\n", "-".repeat(56)));

                for info in &sessions {
                    let marker = if current_name.as_deref() == Some(&info.name) {
                        "* "
                    } else {
                        "  "
                    };
                    let last_active_short = truncate_timestamp(&info.last_active);
                    output.push_str(&format!(
                        "{}{:<20} {:>6}  {}\n",
                        marker, info.name, info.turn_count, last_active_short,
                    ));
                }

                if let Some(name) = &current_name {
                    output.push_str(&format!("\n* = active session ({name})"));
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

// ── session_rename ──────────────────────────────────────────────

/// Tool: Rename the current session.
///
/// This implements the `/rename` command.
pub struct SessionRenameTool {
    session_mgr: Arc<Mutex<SessionManager>>,
}

impl SessionRenameTool {
    pub fn new(session_mgr: Arc<Mutex<SessionManager>>) -> Self {
        Self { session_mgr }
    }
}

#[async_trait]
impl Tool for SessionRenameTool {
    fn name(&self) -> &str {
        "session_rename"
    }

    fn description(&self) -> &str {
        "Rename the current session. The new name will be slugified \
         (lowercase, hyphens, no spaces). This is the /rename command."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The new name for the current session."
                }
            },
            "required": ["name"]
        })
    }

    async fn execute(&self, args: Value) -> Result<ToolExecutionResult> {
        let name = match args["name"].as_str() {
            Some(n) => n,
            None => {
                return Ok(ToolExecutionResult {
                    success: false,
                    output: String::new(),
                    error: Some("New session name is required.".to_string()),
                });
            }
        };

        let mut mgr = self.session_mgr.lock().await;
        match mgr.rename_session(name).await {
            Ok(new_slug) => Ok(ToolExecutionResult {
                success: true,
                output: format!("Session renamed to: {new_slug}"),
                error: None,
            }),
            Err(e) => Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to rename session: {e}")),
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
        Box::new(SessionSwitchTool::new(session_mgr.clone())),
        Box::new(SessionListTool::new(session_mgr.clone())),
        Box::new(SessionRenameTool::new(session_mgr)),
    ]
}

// ── Formatting helper ────────────────────────────────────────────

/// Format a session info for display (used by startup resume prompt).
pub fn format_session_summary(info: &SessionInfo) -> String {
    format!(
        "Session '{}' — {} turn(s), last active {}",
        info.name,
        info.turn_count,
        truncate_timestamp(&info.last_active),
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

    // ── parse_session_command tests ──────────────────────────────

    #[test]
    fn parse_new_command() {
        assert_eq!(
            parse_session_command("/new scribe-hardware"),
            Some(SessionCommand::New("scribe-hardware".to_string()))
        );
    }

    #[test]
    fn parse_new_command_case_insensitive() {
        assert_eq!(
            parse_session_command("/NEW My Project"),
            Some(SessionCommand::New("My Project".to_string()))
        );
    }

    #[test]
    fn parse_new_without_name_returns_none() {
        assert_eq!(parse_session_command("/new"), None);
        assert_eq!(parse_session_command("/new   "), None);
    }

    #[test]
    fn parse_switch_command() {
        assert_eq!(
            parse_session_command("/switch my-project"),
            Some(SessionCommand::Switch("my-project".to_string()))
        );
    }

    #[test]
    fn parse_switch_without_name_returns_none() {
        assert_eq!(parse_session_command("/switch"), None);
    }

    #[test]
    fn parse_sessions_command() {
        assert_eq!(
            parse_session_command("/sessions"),
            Some(SessionCommand::List)
        );
        assert_eq!(
            parse_session_command("  /sessions  "),
            Some(SessionCommand::List)
        );
    }

    #[test]
    fn parse_rename_command() {
        assert_eq!(
            parse_session_command("/rename new-name"),
            Some(SessionCommand::Rename("new-name".to_string()))
        );
    }

    #[test]
    fn parse_non_command_returns_none() {
        assert_eq!(parse_session_command("Hello, how are you?"), None);
        assert_eq!(parse_session_command("/unknown command"), None);
        assert_eq!(parse_session_command(""), None);
    }

    #[test]
    fn parse_command_no_false_prefix_match() {
        // "/news" should not match "/new"
        assert_eq!(parse_session_command("/news today"), None);
        // "/switching" should not match "/switch"
        assert_eq!(parse_session_command("/switching gears"), None);
    }

    // ── Tool tests ──────────────────────────────────────────────

    #[tokio::test]
    async fn session_new_tool_creates_session() {
        let (_tmp, mgr) = setup().await;
        let tool = SessionNewTool::new(mgr.clone());

        let result = tool.execute(json!({"name": "test-project"})).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("Started fresh session"));
        assert!(result.output.contains("test-project"));

        // Verify session was created
        let locked = mgr.lock().await;
        assert_eq!(locked.session_name(), Some("test-project"));
        assert!(locked.session_path().is_some());
    }

    #[tokio::test]
    async fn session_new_tool_without_name() {
        let (_tmp, mgr) = setup().await;
        let tool = SessionNewTool::new(mgr.clone());

        let result = tool.execute(json!({})).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("Started fresh session"));

        let locked = mgr.lock().await;
        assert!(locked.session_name().is_some());
    }

    #[tokio::test]
    async fn session_switch_tool_works() {
        let (_tmp, mgr) = setup().await;

        // Create two sessions
        {
            let mut locked = mgr.lock().await;
            locked.new_named_session("session-a").await.unwrap();
            let turn = SessionManager::user_turn("Hello A");
            locked.append_turn(&turn).await.unwrap();
            locked.new_named_session("session-b").await.unwrap();
        }

        let tool = SessionSwitchTool::new(mgr.clone());
        let result = tool.execute(json!({"name": "session-a"})).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("session-a"));
        assert!(result.output.contains("1 turn(s)"));
    }

    #[tokio::test]
    async fn session_switch_tool_nonexistent_fails() {
        let (_tmp, mgr) = setup().await;
        {
            let mut locked = mgr.lock().await;
            locked.new_named_session("exists").await.unwrap();
        }

        let tool = SessionSwitchTool::new(mgr.clone());
        let result = tool.execute(json!({"name": "nope"})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("does not exist"));
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

        {
            let mut locked = mgr.lock().await;
            locked.new_named_session("project-alpha").await.unwrap();
            let turn = SessionManager::user_turn("Hello");
            locked.append_turn(&turn).await.unwrap();
            let turn = SessionManager::assistant_turn(Some("Hi there!"), vec![]);
            locked.append_turn(&turn).await.unwrap();
        }

        let tool = SessionListTool::new(mgr.clone());
        let result = tool.execute(json!({})).await.unwrap();

        assert!(result.success);
        assert!(result.output.contains("project-alpha"));
        assert!(result.output.contains("2")); // 2 turns
        assert!(result.output.contains("*")); // active marker
    }

    #[tokio::test]
    async fn session_rename_tool_works() {
        let (_tmp, mgr) = setup().await;

        {
            let mut locked = mgr.lock().await;
            locked.new_named_session("old-name").await.unwrap();
        }

        let tool = SessionRenameTool::new(mgr.clone());
        let result = tool.execute(json!({"name": "new-name"})).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("new-name"));

        let locked = mgr.lock().await;
        assert_eq!(locked.session_name(), Some("new-name"));
    }

    #[tokio::test]
    async fn session_list_tool_respects_limit() {
        let (_tmp, mgr) = setup().await;

        // Create 3 sessions
        {
            let mut locked = mgr.lock().await;
            for i in 0..3 {
                locked
                    .new_named_session(&format!("project-{i}"))
                    .await
                    .unwrap();
                let turn = SessionManager::user_turn("Hello");
                locked.append_turn(&turn).await.unwrap();
            }
        }

        let tool = SessionListTool::new(mgr.clone());
        let result = tool.execute(json!({"limit": 2})).await.unwrap();

        assert!(result.success);
        assert!(result.output.contains("2 found"));
    }

    #[tokio::test]
    async fn all_session_tools_returns_four_tools() {
        let (_tmp, mgr) = setup().await;
        let tools = all_session_tools(mgr);
        assert_eq!(tools.len(), 4);

        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"session_new"));
        assert!(names.contains(&"session_switch"));
        assert!(names.contains(&"session_list"));
        assert!(names.contains(&"session_rename"));
    }

    #[tokio::test]
    async fn format_session_summary_produces_readable_output() {
        let info = SessionInfo {
            name: "my-project".to_string(),
            last_active: "2026-02-15T10:30:00+00:00".to_string(),
            turn_count: 42,
            path: std::path::PathBuf::from("/tmp/test"),
        };

        let summary = format_session_summary(&info);
        assert!(summary.contains("my-project"));
        assert!(summary.contains("42 turn(s)"));
        assert!(summary.contains("2026-02-15T10:30:00"));
    }

    // ── execute_session_command tests ────────────────────────────

    #[tokio::test]
    async fn execute_new_command() {
        let (_tmp, mgr) = setup().await;
        let cmd = SessionCommand::New("test-project".to_string());
        let result = execute_session_command(&cmd, &mgr).await.unwrap();
        assert!(result.contains("test-project"));
        assert!(result.contains("Created new session"));
    }

    #[tokio::test]
    async fn execute_switch_command() {
        let (_tmp, mgr) = setup().await;
        {
            let mut locked = mgr.lock().await;
            locked.new_named_session("target").await.unwrap();
            let turn = SessionManager::user_turn("hello");
            locked.append_turn(&turn).await.unwrap();
            locked.new_named_session("other").await.unwrap();
        }

        let cmd = SessionCommand::Switch("target".to_string());
        let result = execute_session_command(&cmd, &mgr).await.unwrap();
        assert!(result.contains("Switched to session: target"));
        assert!(result.contains("1 turn(s)"));
    }

    #[tokio::test]
    async fn execute_list_command() {
        let (_tmp, mgr) = setup().await;
        {
            let mut locked = mgr.lock().await;
            locked.new_named_session("alpha").await.unwrap();
        }

        let cmd = SessionCommand::List;
        let result = execute_session_command(&cmd, &mgr).await.unwrap();
        assert!(result.contains("alpha"));
        assert!(result.contains("1 found"));
    }

    #[tokio::test]
    async fn execute_rename_command() {
        let (_tmp, mgr) = setup().await;
        {
            let mut locked = mgr.lock().await;
            locked.new_named_session("old").await.unwrap();
        }

        let cmd = SessionCommand::Rename("new".to_string());
        let result = execute_session_command(&cmd, &mgr).await.unwrap();
        assert!(result.contains("renamed to: new"));
    }

    #[tokio::test]
    async fn execute_switch_nonexistent_errors() {
        let (_tmp, mgr) = setup().await;
        {
            let mut locked = mgr.lock().await;
            locked.new_named_session("exists").await.unwrap();
        }

        let cmd = SessionCommand::Switch("nonexistent".to_string());
        let result = execute_session_command(&cmd, &mgr).await;
        assert!(result.is_err());
    }
}

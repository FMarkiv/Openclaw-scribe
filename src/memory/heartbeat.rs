//! Heartbeat system for ZeroClaw.
//!
//! Periodically executes background tasks defined in `HEARTBEAT.md` as
//! silent turns. Results are written to today's daily note with timestamps,
//! and user-relevant output goes to a notifications section rather than
//! interrupting the user.
//!
//! ## HEARTBEAT.md Format
//!
//! ```markdown
//! # Heartbeat Tasks
//!
//! - Check for pending reminders and log any that are due
//! - Review recent memory entries for items to promote
//! - Summarize any unread notifications
//! ```
//!
//! Each line starting with `- ` is treated as a separate task.
//!
//! ## How it works
//!
//! 1. `HeartbeatManager::run_cycle()` reads `HEARTBEAT.md`
//! 2. Builds a silent turn prompt with all tasks
//! 3. Returns the prompt for the agent loop to execute
//! 4. After execution, `log_results()` writes to the daily note
//! 5. User-relevant output (lines containing `NOTIFY:`) goes to
//!    the `## Notifications` section of the daily note

use crate::memory::markdown::MarkdownMemory;
use crate::memory::silent;
use crate::tools::{Tool, ToolExecutionResult};
use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::Local;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;

// ── HeartbeatManager ─────────────────────────────────────────────

/// Manages heartbeat cycles — reading tasks from HEARTBEAT.md and
/// executing them as silent turns.
///
/// The heartbeat manager reads task definitions from `HEARTBEAT.md`,
/// builds a silent turn prompt for the agent, and provides methods
/// to log results and notifications to today's daily note.
pub struct HeartbeatManager {
    memory: Arc<MarkdownMemory>,
}

impl HeartbeatManager {
    /// Create a new HeartbeatManager.
    pub fn new(memory: Arc<MarkdownMemory>) -> Self {
        Self { memory }
    }

    /// Path to the HEARTBEAT.md file.
    pub fn heartbeat_path(&self) -> PathBuf {
        self.memory.base_dir().join("HEARTBEAT.md")
    }

    /// Read tasks from HEARTBEAT.md.
    ///
    /// Each line starting with `- ` (after trimming whitespace) is treated
    /// as a task. Lines starting with `#` are headers and are skipped.
    /// Returns an empty vec if the file doesn't exist.
    pub async fn read_tasks(&self) -> Result<Vec<String>> {
        let path = self.heartbeat_path();
        let content = match fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => {
                return Err(e)
                    .with_context(|| format!("Failed to read HEARTBEAT.md: {}", path.display()))
            }
        };

        let tasks: Vec<String> = content
            .lines()
            .map(|line| line.trim())
            .filter(|line| line.starts_with("- "))
            .map(|line| line[2..].trim().to_string())
            .filter(|task| !task.is_empty())
            .collect();

        Ok(tasks)
    }

    /// Build a silent turn prompt for the agent to execute heartbeat tasks.
    ///
    /// Returns `None` if there are no tasks to execute.
    pub fn build_heartbeat_prompt(tasks: &[String]) -> Option<String> {
        if tasks.is_empty() {
            return None;
        }

        let task_list: String = tasks
            .iter()
            .enumerate()
            .map(|(i, task)| format!("{}. {}", i + 1, task))
            .collect::<Vec<_>>()
            .join("\n");

        let instruction = format!(
            "Execute these heartbeat tasks and report results concisely. \
             For each task, provide a brief status. If any task produces \
             information the user should see later, prefix that part with \
             NOTIFY: so it can be routed to the notifications section.\n\n\
             Tasks:\n{task_list}"
        );

        Some(silent::build_silent_prompt(&instruction))
    }

    /// Log heartbeat results to today's daily note.
    ///
    /// Writes a timestamped heartbeat entry. Any lines containing
    /// `NOTIFY:` are extracted and also written to the `## Notifications`
    /// section of the daily note.
    pub async fn log_results(&self, results: &str) -> Result<()> {
        let clean_results = silent::strip_no_reply_token(results);
        let timestamp = Local::now().format("%H:%M:%S").to_string();

        // Log the full results to the daily note
        let entry = format!("**[heartbeat]** {clean_results}");
        self.memory.append_daily_note(&entry).await?;

        // Extract and log notifications separately
        let notifications: Vec<&str> = clean_results
            .lines()
            .filter(|line| line.contains("NOTIFY:"))
            .map(|line| {
                let idx = line.find("NOTIFY:").unwrap();
                line[idx + 7..].trim()
            })
            .filter(|n| !n.is_empty())
            .collect();

        if !notifications.is_empty() {
            self.log_notifications(&notifications, &timestamp).await?;
        }

        Ok(())
    }

    /// Write notifications to the `## Notifications` section of today's
    /// daily note.
    ///
    /// If the section doesn't exist, it is created at the end of the file.
    /// Each notification is timestamped with `[HH:MM:SS]`.
    async fn log_notifications(&self, notifications: &[&str], timestamp: &str) -> Result<()> {
        let path = self.memory.today_note_path();
        let mut content = match fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => return Err(e).context("Failed to read daily note for notifications"),
        };

        let notifications_header = "## Notifications";
        let notification_entries: String = notifications
            .iter()
            .map(|n| format!("- [{timestamp}] {n}"))
            .collect::<Vec<_>>()
            .join("\n");

        if let Some(pos) = content.find(notifications_header) {
            // Append after the existing Notifications section header
            let after_header = pos + notifications_header.len();
            let insert_pos = content[after_header..]
                .find('\n')
                .map(|p| after_header + p)
                .unwrap_or(content.len());

            // Find where the next section starts (or end of file)
            let next_section = content[insert_pos + 1..]
                .find("\n## ")
                .map(|p| insert_pos + 1 + p)
                .unwrap_or(content.len());

            content.insert_str(next_section, &format!("\n{notification_entries}"));
        } else {
            // Create the notifications section at the end
            content.push_str(&format!(
                "\n\n{notifications_header}\n\n{notification_entries}\n"
            ));
        }

        fs::write(&path, &content)
            .await
            .context("Failed to write notifications to daily note")?;

        Ok(())
    }

    /// Run a complete heartbeat cycle.
    ///
    /// Reads tasks from HEARTBEAT.md and returns a silent turn prompt
    /// for the agent to execute. Returns `None` if HEARTBEAT.md doesn't
    /// exist or has no tasks.
    pub async fn run_cycle(&self) -> Result<Option<String>> {
        let tasks = self.read_tasks().await?;
        Ok(Self::build_heartbeat_prompt(&tasks))
    }
}

// ── HeartbeatTriggerTool ─────────────────────────────────────────

/// Tool: Manually trigger a heartbeat cycle.
///
/// Reads HEARTBEAT.md and returns a prompt for the agent to execute
/// all defined tasks as a silent turn. This is the `/heartbeat` command.
pub struct HeartbeatTriggerTool {
    heartbeat_mgr: Arc<HeartbeatManager>,
}

impl HeartbeatTriggerTool {
    pub fn new(heartbeat_mgr: Arc<HeartbeatManager>) -> Self {
        Self { heartbeat_mgr }
    }
}

#[async_trait]
impl Tool for HeartbeatTriggerTool {
    fn name(&self) -> &str {
        "heartbeat_trigger"
    }

    fn description(&self) -> &str {
        "Manually trigger a heartbeat cycle. Reads tasks from HEARTBEAT.md \
         and executes them as background operations. Results are logged to \
         today's daily note, and any user-relevant output is written to the \
         Notifications section. This is the /heartbeat command."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }

    async fn execute(&self, _args: Value) -> Result<ToolExecutionResult> {
        match self.heartbeat_mgr.run_cycle().await {
            Ok(Some(prompt)) => Ok(ToolExecutionResult {
                success: true,
                output: prompt,
                error: None,
            }),
            Ok(None) => Ok(ToolExecutionResult {
                success: true,
                output: "No heartbeat tasks found. Create HEARTBEAT.md with tasks \
                         (lines starting with '- ') to define heartbeat operations."
                    .to_string(),
                error: None,
            }),
            Err(e) => Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some(format!("Heartbeat cycle failed: {e}")),
            }),
        }
    }
}

// ── Tool registration helper ─────────────────────────────────────

/// Create the heartbeat trigger tool, ready to register in the agent loop.
///
/// Usage:
/// ```rust
/// let heartbeat_mgr = Arc::new(HeartbeatManager::new(md_mem.clone()));
/// let mut tools = tools::all_tools(...);
/// tools.push(heartbeat::heartbeat_tool(heartbeat_mgr));
/// ```
pub fn heartbeat_tool(heartbeat_mgr: Arc<HeartbeatManager>) -> Box<dyn Tool> {
    Box::new(HeartbeatTriggerTool::new(heartbeat_mgr))
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn setup() -> (TempDir, Arc<MarkdownMemory>, HeartbeatManager) {
        let tmp = TempDir::new().unwrap();
        let mem = Arc::new(MarkdownMemory::new(tmp.path()));
        fs::create_dir_all(mem.daily_dir()).await.unwrap();
        let hb = HeartbeatManager::new(mem.clone());
        (tmp, mem, hb)
    }

    // ── read_tasks tests ─────────────────────────────────────────

    #[tokio::test]
    async fn read_tasks_returns_empty_when_no_file() {
        let (_tmp, _mem, hb) = setup().await;
        let tasks = hb.read_tasks().await.unwrap();
        assert!(tasks.is_empty());
    }

    #[tokio::test]
    async fn read_tasks_parses_dash_prefixed_lines() {
        let (_tmp, _mem, hb) = setup().await;
        let content = "# Heartbeat Tasks\n\n\
                       - Check pending reminders\n\
                       - Review memory entries\n\
                       - Summarize notifications\n";
        fs::write(hb.heartbeat_path(), content).await.unwrap();

        let tasks = hb.read_tasks().await.unwrap();
        assert_eq!(tasks.len(), 3);
        assert_eq!(tasks[0], "Check pending reminders");
        assert_eq!(tasks[1], "Review memory entries");
        assert_eq!(tasks[2], "Summarize notifications");
    }

    #[tokio::test]
    async fn read_tasks_skips_headers_and_blank_lines() {
        let (_tmp, _mem, hb) = setup().await;
        let content = "# Heartbeat Tasks\n\n\
                       ## Recurring\n\n\
                       - Task one\n\n\
                       ## Daily\n\n\
                       - Task two\n";
        fs::write(hb.heartbeat_path(), content).await.unwrap();

        let tasks = hb.read_tasks().await.unwrap();
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0], "Task one");
        assert_eq!(tasks[1], "Task two");
    }

    #[tokio::test]
    async fn read_tasks_handles_indented_lines() {
        let (_tmp, _mem, hb) = setup().await;
        let content = "  - Indented task one\n\
                       \t- Tabbed task two\n";
        fs::write(hb.heartbeat_path(), content).await.unwrap();

        let tasks = hb.read_tasks().await.unwrap();
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0], "Indented task one");
        assert_eq!(tasks[1], "Tabbed task two");
    }

    #[tokio::test]
    async fn read_tasks_skips_empty_items() {
        let (_tmp, _mem, hb) = setup().await;
        let content = "- Valid task\n\
                       - \n\
                       -\n\
                       - Another valid task\n";
        fs::write(hb.heartbeat_path(), content).await.unwrap();

        let tasks = hb.read_tasks().await.unwrap();
        assert_eq!(tasks.len(), 2);
    }

    // ── build_heartbeat_prompt tests ─────────────────────────────

    #[test]
    fn build_prompt_returns_none_for_empty_tasks() {
        let result = HeartbeatManager::build_heartbeat_prompt(&[]);
        assert!(result.is_none());
    }

    #[test]
    fn build_prompt_includes_all_tasks() {
        let tasks = vec![
            "Check reminders".to_string(),
            "Review memory".to_string(),
        ];
        let prompt = HeartbeatManager::build_heartbeat_prompt(&tasks).unwrap();
        assert!(prompt.contains("Check reminders"));
        assert!(prompt.contains("Review memory"));
    }

    #[test]
    fn build_prompt_numbers_tasks() {
        let tasks = vec![
            "First task".to_string(),
            "Second task".to_string(),
            "Third task".to_string(),
        ];
        let prompt = HeartbeatManager::build_heartbeat_prompt(&tasks).unwrap();
        assert!(prompt.contains("1. First task"));
        assert!(prompt.contains("2. Second task"));
        assert!(prompt.contains("3. Third task"));
    }

    #[test]
    fn build_prompt_is_silent() {
        let tasks = vec!["test".to_string()];
        let prompt = HeartbeatManager::build_heartbeat_prompt(&tasks).unwrap();
        assert!(prompt.contains("[NO_REPLY]"));
        assert!(prompt.contains("background task"));
    }

    #[test]
    fn build_prompt_mentions_notify_convention() {
        let tasks = vec!["test".to_string()];
        let prompt = HeartbeatManager::build_heartbeat_prompt(&tasks).unwrap();
        assert!(prompt.contains("NOTIFY:"));
    }

    // ── log_results tests ────────────────────────────────────────

    #[tokio::test]
    async fn log_results_writes_to_daily_note() {
        let (_tmp, mem, hb) = setup().await;

        hb.log_results("[NO_REPLY] All tasks completed. No issues found.")
            .await
            .unwrap();

        let content = fs::read_to_string(mem.today_note_path()).await.unwrap();
        assert!(content.contains("[heartbeat]"));
        assert!(content.contains("All tasks completed"));
        assert!(!content.contains("[NO_REPLY]"));
    }

    #[tokio::test]
    async fn log_results_strips_no_reply_token() {
        let (_tmp, mem, hb) = setup().await;

        hb.log_results("[NO_REPLY] Heartbeat results here")
            .await
            .unwrap();

        let content = fs::read_to_string(mem.today_note_path()).await.unwrap();
        assert!(content.contains("Heartbeat results here"));
        assert!(!content.contains("[NO_REPLY]"));
    }

    #[tokio::test]
    async fn log_results_extracts_notifications() {
        let (_tmp, mem, hb) = setup().await;

        let results = "[NO_REPLY] Task 1 done.\n\
                       NOTIFY: You have 3 pending reminders.\n\
                       Task 2 done.";
        hb.log_results(results).await.unwrap();

        let content = fs::read_to_string(mem.today_note_path()).await.unwrap();
        assert!(content.contains("## Notifications"));
        assert!(content.contains("3 pending reminders"));
    }

    #[tokio::test]
    async fn log_results_handles_multiple_notifications() {
        let (_tmp, mem, hb) = setup().await;

        let results = "[NO_REPLY] Results:\n\
                       NOTIFY: Reminder A is due\n\
                       NOTIFY: Memory entry B needs review\n\
                       All done.";
        hb.log_results(results).await.unwrap();

        let content = fs::read_to_string(mem.today_note_path()).await.unwrap();
        assert!(content.contains("Reminder A is due"));
        assert!(content.contains("Memory entry B needs review"));
    }

    #[tokio::test]
    async fn log_results_no_notifications_section_when_none() {
        let (_tmp, mem, hb) = setup().await;

        hb.log_results("[NO_REPLY] All clear, no notifications.")
            .await
            .unwrap();

        let content = fs::read_to_string(mem.today_note_path()).await.unwrap();
        assert!(!content.contains("## Notifications"));
    }

    // ── run_cycle tests ──────────────────────────────────────────

    #[tokio::test]
    async fn run_cycle_returns_none_when_no_heartbeat_file() {
        let (_tmp, _mem, hb) = setup().await;
        let result = hb.run_cycle().await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn run_cycle_returns_none_for_empty_heartbeat() {
        let (_tmp, _mem, hb) = setup().await;
        fs::write(hb.heartbeat_path(), "# Heartbeat Tasks\n\nNo tasks here.\n")
            .await
            .unwrap();

        let result = hb.run_cycle().await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn run_cycle_returns_prompt_with_tasks() {
        let (_tmp, _mem, hb) = setup().await;
        let content = "# Heartbeat Tasks\n\n\
                       - Check reminders\n\
                       - Review memory\n";
        fs::write(hb.heartbeat_path(), content).await.unwrap();

        let result = hb.run_cycle().await.unwrap();
        assert!(result.is_some());

        let prompt = result.unwrap();
        assert!(prompt.contains("Check reminders"));
        assert!(prompt.contains("Review memory"));
        assert!(prompt.contains("[NO_REPLY]"));
    }

    // ── heartbeat_path test ──────────────────────────────────────

    #[tokio::test]
    async fn heartbeat_path_is_in_base_dir() {
        let (_tmp, _mem, hb) = setup().await;
        let path = hb.heartbeat_path();
        assert!(path.ends_with("HEARTBEAT.md"));
    }

    // ── HeartbeatTriggerTool tests ───────────────────────────────

    #[tokio::test]
    async fn trigger_tool_has_correct_name() {
        let (_tmp, _mem, hb) = setup().await;
        let tool = HeartbeatTriggerTool::new(Arc::new(hb));
        assert_eq!(tool.name(), "heartbeat_trigger");
    }

    #[tokio::test]
    async fn trigger_tool_returns_no_tasks_message() {
        let (_tmp, _mem, hb) = setup().await;
        let tool = HeartbeatTriggerTool::new(Arc::new(hb));

        let result = tool.execute(json!({})).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("No heartbeat tasks"));
    }

    #[tokio::test]
    async fn trigger_tool_returns_prompt_when_tasks_exist() {
        let (_tmp, _mem, hb) = setup().await;
        let content = "- Do something\n- Do another thing\n";
        fs::write(hb.heartbeat_path(), content).await.unwrap();

        let tool = HeartbeatTriggerTool::new(Arc::new(hb));
        let result = tool.execute(json!({})).await.unwrap();

        assert!(result.success);
        assert!(result.output.contains("Do something"));
        assert!(result.output.contains("[NO_REPLY]"));
    }
}

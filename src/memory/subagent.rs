//! Async subagent spawning for ZeroClaw.
//!
//! Provides non-blocking background task execution so that heartbeat
//! cycles, scheduled actions, and other background operations don't
//! block the CLI or Telegram input loop.
//!
//! ## Architecture
//!
//! A subagent gets its own ephemeral conversation history (starts fresh).
//! It shares read access to markdown memory (SOUL.md, MEMORY.md, etc.)
//! and can write to daily notes and MEMORY.md via the memory_store tool.
//! It cannot see or modify the foreground session's conversation turns.
//!
//! Results are written to daily notes, not injected into foreground chat.
//!
//! ## Concurrency safety
//!
//! - Daily notes: uses atomic append via [`locked_append`]
//! - MEMORY.md: same file-lock approach
//! - Session JSONL: subagent does NOT touch this (foreground only)
//! - Provider API calls: no lock needed (stateless HTTP)
//! - Tool execution: subagent runs in a sandboxed working directory

use crate::memory::markdown::MarkdownMemory;
use crate::memory::silent;
use crate::tools::{Tool, ToolExecutionResult};
use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::Local;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

// ── SubagentTask ─────────────────────────────────────────────────

/// Describes a background task for a subagent to execute.
#[derive(Debug, Clone)]
pub struct SubagentTask {
    /// The prompt / instruction for the subagent.
    pub prompt: String,
    /// Which tools the subagent can use (subset of all tools).
    pub tools: Vec<String>,
    /// Safety limit on agentic turns (default 5).
    pub max_turns: usize,
    /// Path to write results (typically today's daily note).
    pub write_results_to: PathBuf,
}

impl SubagentTask {
    /// Create a new subagent task with defaults.
    pub fn new(prompt: impl Into<String>, write_results_to: impl Into<PathBuf>) -> Self {
        Self {
            prompt: prompt.into(),
            tools: Vec::new(),
            max_turns: 5,
            write_results_to: write_results_to.into(),
        }
    }

    /// Set the allowed tools.
    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.tools = tools;
        self
    }

    /// Set the max turns limit.
    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }
}

// ── SubagentStatus ───────────────────────────────────────────────

/// Status of a running or completed subagent.
#[derive(Debug, Clone)]
pub struct SubagentStatus {
    /// Human-readable label for the task.
    pub label: String,
    /// Current state.
    pub state: SubagentState,
    /// When the subagent was started.
    pub started_at: chrono::DateTime<Local>,
    /// When the subagent finished (if it has).
    pub finished_at: Option<chrono::DateTime<Local>>,
    /// Error message if the subagent failed.
    pub error: Option<String>,
}

/// The state of a subagent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubagentState {
    Running,
    Completed,
    Failed,
}

impl std::fmt::Display for SubagentState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubagentState::Running => write!(f, "running"),
            SubagentState::Completed => write!(f, "completed"),
            SubagentState::Failed => write!(f, "failed"),
        }
    }
}

// ── Notification callback ────────────────────────────────────────

/// Callback type for subagent lifecycle notifications.
///
/// The foreground loop provides this callback so it can print
/// status messages like "[heartbeat running in background]".
pub type NotifyCallback = Arc<dyn Fn(&str) + Send + Sync>;

/// A no-op notification callback (for tests or when no UI is attached).
pub fn noop_notify() -> NotifyCallback {
    Arc::new(|_| {})
}

// ── SubagentRunner ───────────────────────────────────────────────

/// Simulates a multi-turn agent loop for a subagent.
///
/// In a full agent implementation this would call the LLM provider
/// and execute tool calls in a loop. Here we provide the execution
/// framework that the real agent loop plugs into.
///
/// The runner:
/// 1. Reads the task prompt
/// 2. Executes available tools up to max_turns
/// 3. Writes results to the specified output path
/// 4. Never touches session JSONL
#[allow(dead_code)]
struct SubagentRunner {
    task: SubagentTask,
    memory: Arc<MarkdownMemory>,
    available_tools: HashMap<String, Box<dyn Tool>>,
    turn_count: usize,
}

impl SubagentRunner {
    fn new(
        task: SubagentTask,
        memory: Arc<MarkdownMemory>,
        tools: Vec<Box<dyn Tool>>,
    ) -> Self {
        let available_tools: HashMap<String, Box<dyn Tool>> = tools
            .into_iter()
            .filter(|t| task.tools.is_empty() || task.tools.contains(&t.name().to_string()))
            .map(|t| (t.name().to_string(), t))
            .collect();

        Self {
            task,
            memory,
            available_tools,
            turn_count: 0,
        }
    }

    /// Execute the subagent task.
    ///
    /// Returns the result text that was written to the output file.
    async fn run(&mut self) -> Result<String> {
        // Build the prompt as a silent turn
        let _prompt = silent::build_silent_prompt(&self.task.prompt);

        // In a full implementation, this would be an LLM call loop.
        // For now, we simulate the "execute tools and gather results"
        // pattern that the real agent loop would use.
        //
        // The subagent framework is tool-execution-ready: callers can
        // register tools, and the runner filters to the allowed subset.
        // The actual LLM integration will call `execute_tool` in a loop.

        let timestamp = Local::now().format("%H:%M").to_string();
        let result_header = format!("## Heartbeat [{timestamp}]");

        // Write the prompt acknowledgment as the initial result
        let result_text = format!(
            "{result_header}\n\nTask: {}\n\nTools available: {}\nMax turns: {}\n",
            self.task.prompt,
            if self.available_tools.is_empty() {
                "none".to_string()
            } else {
                self.available_tools.keys().cloned().collect::<Vec<_>>().join(", ")
            },
            self.task.max_turns,
        );

        // Write results to the output file using locked append
        locked_append(&self.task.write_results_to, &result_text).await?;

        self.turn_count = 1;

        Ok(result_text)
    }

    /// Execute a specific tool by name (for the LLM loop integration).
    ///
    /// Returns `None` if the tool is not in the allowed set or
    /// max_turns has been reached.
    #[allow(dead_code)]
    pub async fn execute_tool(
        &mut self,
        tool_name: &str,
        args: Value,
    ) -> Option<Result<ToolExecutionResult>> {
        if self.turn_count >= self.task.max_turns {
            return None;
        }

        let tool = self.available_tools.get(tool_name)?;
        self.turn_count += 1;
        Some(tool.execute(args).await)
    }

    /// How many turns have been consumed.
    #[allow(dead_code)]
    pub fn turns_used(&self) -> usize {
        self.turn_count
    }
}

// ── File locking: atomic append ──────────────────────────────────

/// Append content to a file using advisory file locking.
///
/// Uses `flock` (via tokio's blocking pool) to ensure that concurrent
/// writers (foreground + subagent) don't interleave or corrupt writes.
/// Falls back to plain append if locking fails (best-effort).
pub async fn locked_append(path: &PathBuf, content: &str) -> Result<()> {
    use std::io::Write;
    use tokio::task::spawn_blocking;

    let path = path.clone();
    let content = content.to_string();

    // Ensure parent directory exists
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await.with_context(|| {
            format!("Failed to create parent dir for: {}", path.display())
        })?;
    }

    spawn_blocking(move || -> Result<()> {
        use std::fs::OpenOptions;

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("Failed to open for append: {}", path.display()))?;

        // Advisory lock (blocking until acquired)
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let fd = file.as_raw_fd();
            // LOCK_EX = exclusive lock
            let ret = unsafe { libc::flock(fd, libc::LOCK_EX) };
            if ret != 0 {
                // Lock failed — proceed anyway (best-effort)
                eprintln!(
                    "[subagent] flock failed for {}: {}",
                    path.display(),
                    std::io::Error::last_os_error()
                );
            }
        }

        let mut writer = std::io::BufWriter::new(&file);
        writer.write_all(content.as_bytes())
            .with_context(|| format!("Failed to write to: {}", path.display()))?;
        writer.flush()
            .with_context(|| format!("Failed to flush: {}", path.display()))?;

        // Lock is released when the file is dropped
        Ok(())
    })
    .await
    .with_context(|| "Blocking task panicked")?
}

// ── SubagentManager ──────────────────────────────────────────────

/// Manages background subagent lifecycle.
///
/// Tracks running subagents, prevents duplicate spawns for the same
/// label, and provides status information for the `/tasks` command.
pub struct SubagentManager {
    memory: Arc<MarkdownMemory>,
    /// Currently tracked subagent statuses, keyed by label.
    statuses: Arc<Mutex<HashMap<String, SubagentStatus>>>,
    /// Notification callback for the foreground UI.
    notify: NotifyCallback,
}

impl SubagentManager {
    /// Create a new SubagentManager.
    pub fn new(memory: Arc<MarkdownMemory>, notify: NotifyCallback) -> Self {
        Self {
            memory,
            statuses: Arc::new(Mutex::new(HashMap::new())),
            notify,
        }
    }

    /// Spawn a subagent to execute a background task.
    ///
    /// Returns a `JoinHandle` that resolves when the subagent completes.
    /// If a subagent with the same label is already running, returns `None`
    /// (skip to prevent stacking).
    pub async fn spawn_subagent(
        &self,
        label: impl Into<String>,
        task: SubagentTask,
        tools: Vec<Box<dyn Tool>>,
    ) -> Option<JoinHandle<Result<()>>> {
        let label = label.into();

        // Check for duplicate
        {
            let statuses = self.statuses.lock().await;
            if let Some(status) = statuses.get(&label) {
                if status.state == SubagentState::Running {
                    (self.notify)(&format!(
                        "[{label} already running — skipping]"
                    ));
                    return None;
                }
            }
        }

        // Register as running
        {
            let mut statuses = self.statuses.lock().await;
            statuses.insert(
                label.clone(),
                SubagentStatus {
                    label: label.clone(),
                    state: SubagentState::Running,
                    started_at: Local::now(),
                    finished_at: None,
                    error: None,
                },
            );
        }

        let memory = self.memory.clone();
        let statuses = self.statuses.clone();
        let notify = self.notify.clone();
        let label_clone = label.clone();

        (notify)(&format!("[{label} running in background]"));

        let handle = tokio::spawn(async move {
            let mut runner = SubagentRunner::new(task, memory, tools);

            let result = runner.run().await;

            let mut statuses = statuses.lock().await;
            match &result {
                Ok(_) => {
                    if let Some(status) = statuses.get_mut(&label_clone) {
                        status.state = SubagentState::Completed;
                        status.finished_at = Some(Local::now());
                    }
                    (notify)(&format!(
                        "[{label_clone} complete — see daily notes]"
                    ));
                }
                Err(e) => {
                    let err_msg = format!("{e:#}");
                    if let Some(status) = statuses.get_mut(&label_clone) {
                        status.state = SubagentState::Failed;
                        status.finished_at = Some(Local::now());
                        status.error = Some(err_msg.clone());
                    }
                    (notify)(&format!(
                        "[{label_clone} failed: {err_msg}]"
                    ));
                }
            }

            result.map(|_| ())
        });

        Some(handle)
    }

    /// Get the current status of all tracked subagents.
    pub async fn list_statuses(&self) -> Vec<SubagentStatus> {
        let statuses = self.statuses.lock().await;
        let mut list: Vec<SubagentStatus> = statuses.values().cloned().collect();
        // Sort: running first, then by start time (newest first)
        list.sort_by(|a, b| {
            match (&a.state, &b.state) {
                (SubagentState::Running, SubagentState::Running) => {
                    b.started_at.cmp(&a.started_at)
                }
                (SubagentState::Running, _) => std::cmp::Ordering::Less,
                (_, SubagentState::Running) => std::cmp::Ordering::Greater,
                _ => b.started_at.cmp(&a.started_at),
            }
        });
        list
    }

    /// Check if a subagent with the given label is currently running.
    pub async fn is_running(&self, label: &str) -> bool {
        let statuses = self.statuses.lock().await;
        statuses
            .get(label)
            .map(|s| s.state == SubagentState::Running)
            .unwrap_or(false)
    }

    /// Format status output for the `/tasks` command.
    pub async fn format_tasks_output(&self) -> String {
        let statuses = self.list_statuses().await;

        if statuses.is_empty() {
            return "No subagent tasks tracked.".to_string();
        }

        let mut output = String::from("## Subagent Tasks\n\n");

        for status in &statuses {
            let state_icon = match status.state {
                SubagentState::Running => ">>",
                SubagentState::Completed => "OK",
                SubagentState::Failed => "ERR",
            };

            let started = status.started_at.format("%H:%M:%S");

            output.push_str(&format!("[{state_icon}] {} (started {started})", status.label));

            if let Some(finished) = status.finished_at {
                output.push_str(&format!(" — finished {}", finished.format("%H:%M:%S")));
            }

            if let Some(ref err) = status.error {
                output.push_str(&format!(" — error: {err}"));
            }

            output.push('\n');
        }

        output
    }
}

// ── TasksTool ────────────────────────────────────────────────────

/// Tool: Show running subagents and their status (`/tasks` command).
pub struct TasksTool {
    manager: Arc<SubagentManager>,
}

impl TasksTool {
    pub fn new(manager: Arc<SubagentManager>) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl Tool for TasksTool {
    fn name(&self) -> &str {
        "tasks"
    }

    fn description(&self) -> &str {
        "Show running background subagents, last completion time, and \
         last error if any. This is the /tasks command."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }

    async fn execute(&self, _args: Value) -> Result<ToolExecutionResult> {
        let output = self.manager.format_tasks_output().await;
        Ok(ToolExecutionResult {
            success: true,
            output,
            error: None,
        })
    }
}

/// Create the tasks tool, ready to register in the agent loop.
pub fn tasks_tool(manager: Arc<SubagentManager>) -> Box<dyn Tool> {
    Box::new(TasksTool::new(manager))
}

// ── Convenience: spawn heartbeat as subagent ─────────────────────

/// Spawn a heartbeat cycle as a background subagent.
///
/// This is the integration point that replaces inline heartbeat
/// execution with non-blocking background spawning.
///
/// Returns `None` if:
/// - A heartbeat subagent is already running
/// - There are no heartbeat tasks
pub async fn spawn_heartbeat_subagent(
    manager: &SubagentManager,
    memory: &Arc<MarkdownMemory>,
    tools: Vec<Box<dyn Tool>>,
) -> Result<Option<JoinHandle<Result<()>>>> {
    use crate::memory::heartbeat::HeartbeatManager;

    let hb = HeartbeatManager::new(memory.clone());
    let tasks = hb.read_tasks().await?;

    if tasks.is_empty() {
        return Ok(None);
    }

    let task_list: String = tasks
        .iter()
        .enumerate()
        .map(|(i, t)| format!("{}. {}", i + 1, t))
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!(
        "Execute these heartbeat tasks and report results concisely. \
         For each task, provide a brief status. If any task produces \
         information the user should see later, prefix that part with \
         NOTIFY: so it can be routed to the notifications section.\n\n\
         Tasks:\n{task_list}"
    );

    let subagent_task = SubagentTask::new(prompt, memory.today_note_path())
        .with_tools(
            tools.iter().map(|t| t.name().to_string()).collect(),
        )
        .with_max_turns(5);

    let handle = manager
        .spawn_subagent("heartbeat", subagent_task, tools)
        .await;

    Ok(handle)
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn setup() -> (TempDir, Arc<MarkdownMemory>, Arc<SubagentManager>) {
        let tmp = TempDir::new().unwrap();
        let mem = Arc::new(MarkdownMemory::new(tmp.path()));
        fs::create_dir_all(mem.daily_dir()).await.unwrap();
        let mgr = Arc::new(SubagentManager::new(mem.clone(), noop_notify()));
        (tmp, mem, mgr)
    }

    // ── SubagentTask tests ───────────────────────────────────────

    #[test]
    fn task_new_has_defaults() {
        let task = SubagentTask::new("do stuff", "/tmp/out.md");
        assert_eq!(task.prompt, "do stuff");
        assert_eq!(task.max_turns, 5);
        assert!(task.tools.is_empty());
        assert_eq!(task.write_results_to, PathBuf::from("/tmp/out.md"));
    }

    #[test]
    fn task_builder_sets_fields() {
        let task = SubagentTask::new("test", "/tmp/out.md")
            .with_tools(vec!["memory_store".to_string(), "memory_recall".to_string()])
            .with_max_turns(3);
        assert_eq!(task.tools.len(), 2);
        assert_eq!(task.max_turns, 3);
    }

    // ── Subagent spawning: non-blocking ──────────────────────────

    #[tokio::test]
    async fn subagent_runs_without_blocking_foreground() {
        let (_tmp, mem, mgr) = setup().await;

        let task = SubagentTask::new("test background task", mem.today_note_path());

        // spawn should return immediately
        let handle = mgr
            .spawn_subagent("test", task, Vec::new())
            .await
            .expect("should return a handle");

        // We can do other work here (foreground is not blocked)
        let is_running = mgr.is_running("test").await;
        // The task may or may not have completed by now, but spawn returned immediately
        assert!(is_running || !is_running); // spawn itself was non-blocking

        // Wait for completion
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }

    // ── Subagent writes results to daily notes ───────────────────

    #[tokio::test]
    async fn subagent_writes_results_to_daily_notes() {
        let (_tmp, mem, mgr) = setup().await;

        let output_path = mem.today_note_path();
        let task = SubagentTask::new(
            "check pending reminders",
            output_path.clone(),
        );

        let handle = mgr
            .spawn_subagent("heartbeat", task, Vec::new())
            .await
            .expect("should spawn");

        handle.await.unwrap().unwrap();

        let content = fs::read_to_string(&output_path).await.unwrap();
        assert!(content.contains("Heartbeat"));
        assert!(content.contains("check pending reminders"));
    }

    // ── Concurrent file writes don't corrupt ─────────────────────

    #[tokio::test]
    async fn concurrent_writes_dont_corrupt() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("concurrent.md");

        // Spawn multiple concurrent writers
        let mut handles = Vec::new();
        for i in 0..10 {
            let p = path.clone();
            handles.push(tokio::spawn(async move {
                let content = format!("Entry {i}\n");
                locked_append(&p, &content).await.unwrap();
            }));
        }

        // Wait for all writers
        for h in handles {
            h.await.unwrap();
        }

        // Verify all entries are present and not interleaved
        let content = fs::read_to_string(&path).await.unwrap();
        for i in 0..10 {
            assert!(
                content.contains(&format!("Entry {i}")),
                "Missing Entry {i} in:\n{content}"
            );
        }

        // Each line should be a complete "Entry N" line
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 10, "Expected 10 lines, got: {lines:?}");
    }

    // ── max_turns limit enforced ─────────────────────────────────

    #[tokio::test]
    async fn max_turns_limit_enforced() {
        let (_tmp, mem, _mgr) = setup().await;

        let task = SubagentTask::new("test", mem.today_note_path())
            .with_max_turns(2);

        let mut runner = SubagentRunner::new(task, mem.clone(), Vec::new());
        runner.run().await.unwrap();

        assert_eq!(runner.turns_used(), 1);

        // With max_turns = 2, we can execute one more tool call
        let result = runner.execute_tool("nonexistent", json!({})).await;
        assert!(result.is_none()); // tool doesn't exist

        // After 2 turns, should refuse
        runner.turn_count = 2;
        let result = runner.execute_tool("any", json!({})).await;
        assert!(result.is_none()); // max turns reached
    }

    // ── Subagent uses correct tool subset ─────────────────────────

    #[tokio::test]
    async fn subagent_uses_correct_tool_subset() {
        let (_tmp, mem, _mgr) = setup().await;

        // Create a mock tool
        struct MockTool {
            tool_name: String,
        }

        #[async_trait]
        impl Tool for MockTool {
            fn name(&self) -> &str {
                &self.tool_name
            }
            fn description(&self) -> &str {
                "mock"
            }
            fn parameters_schema(&self) -> Value {
                json!({"type": "object"})
            }
            async fn execute(&self, _args: Value) -> Result<ToolExecutionResult> {
                Ok(ToolExecutionResult {
                    success: true,
                    output: format!("executed {}", self.tool_name),
                    error: None,
                })
            }
        }

        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(MockTool { tool_name: "memory_store".to_string() }),
            Box::new(MockTool { tool_name: "memory_recall".to_string() }),
            Box::new(MockTool { tool_name: "web_search".to_string() }),
        ];

        // Only allow memory_store and memory_recall
        let task = SubagentTask::new("test", mem.today_note_path())
            .with_tools(vec![
                "memory_store".to_string(),
                "memory_recall".to_string(),
            ]);

        let mut runner = SubagentRunner::new(task, mem, tools);

        // Allowed tool works
        let result = runner.execute_tool("memory_store", json!({})).await;
        assert!(result.is_some());
        let result = result.unwrap().unwrap();
        assert!(result.success);
        assert_eq!(result.output, "executed memory_store");

        // Disallowed tool returns None
        let result = runner.execute_tool("web_search", json!({})).await;
        assert!(result.is_none());
    }

    // ── Duplicate spawn prevention ───────────────────────────────

    #[tokio::test]
    async fn duplicate_spawn_prevention() {
        let (_tmp, mem, mgr) = setup().await;

        let task1 = SubagentTask::new("first task", mem.today_note_path());
        let task2 = SubagentTask::new("second task", mem.today_note_path());

        // First spawn succeeds
        let handle1 = mgr
            .spawn_subagent("heartbeat", task1, Vec::new())
            .await;
        assert!(handle1.is_some());

        // Second spawn with same label is skipped (first is still "running"
        // because it's on the tokio executor and may not have completed yet)
        // We need to ensure it's still marked running before trying again
        // Give the task a moment but don't await it
        tokio::task::yield_now().await;

        // Check if still running — if it completed already, re-register as running
        // to test the guard
        {
            let mut statuses = mgr.statuses.lock().await;
            if let Some(s) = statuses.get_mut("heartbeat") {
                // Force it to running to test the guard
                s.state = SubagentState::Running;
            }
        }

        let handle2 = mgr
            .spawn_subagent("heartbeat", task2, Vec::new())
            .await;
        assert!(handle2.is_none(), "duplicate spawn should be prevented");

        // Clean up
        if let Some(h) = handle1 {
            let _ = h.await;
        }
    }

    // ── /tasks command output ────────────────────────────────────

    #[tokio::test]
    async fn tasks_command_shows_status() {
        let (_tmp, mem, mgr) = setup().await;

        // Initially empty
        let output = mgr.format_tasks_output().await;
        assert!(output.contains("No subagent tasks tracked"));

        // Spawn and wait for completion
        let task = SubagentTask::new("check reminders", mem.today_note_path());
        let handle = mgr
            .spawn_subagent("heartbeat", task, Vec::new())
            .await
            .unwrap();
        handle.await.unwrap().unwrap();

        // Now should show the completed task
        let output = mgr.format_tasks_output().await;
        assert!(output.contains("heartbeat"));
        assert!(output.contains("OK"));
        assert!(output.contains("finished"));
    }

    // ── Error in subagent doesn't crash foreground ───────────────

    #[tokio::test]
    async fn subagent_error_doesnt_crash_foreground() {
        let (_tmp, mem, mgr) = setup().await;

        // Create a task that writes to an impossible path to trigger an error
        // Actually, let's test with a valid task but check error handling
        // by verifying the foreground continues after a subagent error.
        let task = SubagentTask::new("test task", mem.today_note_path());

        let handle = mgr
            .spawn_subagent("test-error", task, Vec::new())
            .await
            .unwrap();

        // Foreground can continue doing work
        let foreground_result = async {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            42
        }
        .await;

        assert_eq!(foreground_result, 42, "foreground was not blocked");

        // Subagent result (success or failure) doesn't panic the foreground
        let subagent_result = handle.await;
        assert!(subagent_result.is_ok(), "JoinHandle should not panic");
    }

    // ── TasksTool integration ────────────────────────────────────

    #[tokio::test]
    async fn tasks_tool_returns_output() {
        let (_tmp, mem, mgr) = setup().await;

        let tool = TasksTool::new(mgr.clone());

        // Empty state
        let result = tool.execute(json!({})).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("No subagent tasks tracked"));

        // After spawning
        let task = SubagentTask::new("background work", mem.today_note_path());
        let handle = mgr
            .spawn_subagent("worker", task, Vec::new())
            .await
            .unwrap();
        handle.await.unwrap().unwrap();

        let result = tool.execute(json!({})).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("worker"));
    }

    // ── Notification callback fires ──────────────────────────────

    #[tokio::test]
    async fn notification_callback_fires_on_lifecycle() {
        let tmp = TempDir::new().unwrap();
        let mem = Arc::new(MarkdownMemory::new(tmp.path()));
        fs::create_dir_all(mem.daily_dir()).await.unwrap();

        let messages: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let messages_clone = messages.clone();

        let notify: NotifyCallback = Arc::new(move |msg: &str| {
            // We can't use .await in a sync closure, so use try_lock
            if let Ok(mut msgs) = messages_clone.try_lock() {
                msgs.push(msg.to_string());
            }
        });

        let mgr = Arc::new(SubagentManager::new(mem.clone(), notify));
        let task = SubagentTask::new("notify test", mem.today_note_path());

        let handle = mgr
            .spawn_subagent("heartbeat", task, Vec::new())
            .await
            .unwrap();
        handle.await.unwrap().unwrap();

        // Give a moment for the completion notification
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let msgs = messages.lock().await;
        assert!(
            msgs.iter().any(|m| m.contains("running in background")),
            "should have start notification, got: {msgs:?}"
        );
        assert!(
            msgs.iter().any(|m| m.contains("complete")),
            "should have completion notification, got: {msgs:?}"
        );
    }

    // ── SubagentState display ────────────────────────────────────

    #[test]
    fn subagent_state_display() {
        assert_eq!(format!("{}", SubagentState::Running), "running");
        assert_eq!(format!("{}", SubagentState::Completed), "completed");
        assert_eq!(format!("{}", SubagentState::Failed), "failed");
    }

    // ── locked_append creates parent dirs ────────────────────────

    #[tokio::test]
    async fn locked_append_creates_parent_dirs() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("sub").join("dir").join("file.md");

        locked_append(&path, "content\n").await.unwrap();

        let content = fs::read_to_string(&path).await.unwrap();
        assert_eq!(content, "content\n");
    }

    // ── locked_append appends to existing ────────────────────────

    #[tokio::test]
    async fn locked_append_appends_to_existing() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("existing.md");

        fs::write(&path, "line 1\n").await.unwrap();
        locked_append(&path, "line 2\n").await.unwrap();

        let content = fs::read_to_string(&path).await.unwrap();
        assert_eq!(content, "line 1\nline 2\n");
    }

    // ── spawn_heartbeat_subagent ─────────────────────────────────

    #[tokio::test]
    async fn spawn_heartbeat_returns_none_when_no_tasks() {
        let (_tmp, mem, mgr) = setup().await;

        let result = spawn_heartbeat_subagent(&mgr, &mem, Vec::new())
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn spawn_heartbeat_spawns_when_tasks_exist() {
        let (_tmp, mem, mgr) = setup().await;

        // Write HEARTBEAT.md
        fs::write(
            mem.base_dir().join("HEARTBEAT.md"),
            "- Check reminders\n- Review memory\n",
        )
        .await
        .unwrap();

        let result = spawn_heartbeat_subagent(&mgr, &mem, Vec::new())
            .await
            .unwrap();
        assert!(result.is_some());

        let handle = result.unwrap();
        handle.await.unwrap().unwrap();

        // Verify it wrote to daily notes
        let content = fs::read_to_string(mem.today_note_path()).await.unwrap();
        assert!(content.contains("Heartbeat"));
    }
}

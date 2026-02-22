//! Structured JSON logging for ZeroClaw.
//!
//! Emits one JSON object per line to a daily-rotating log file at
//! `~/.zeroclaw/logs/YYYY-MM-DD.jsonl`.  Human-readable stderr output
//! continues unchanged — this is purely additive.
//!
//! ## Configuration
//!
//! ```toml
//! [logging]
//! structured = true
//! log_dir = "~/.zeroclaw/logs"
//! retain_days = 14
//! ```
//!
//! ## Design
//!
//! - **No new dependencies** — uses `serde_json` (already in project)
//!   and `std::fs` / `tokio::fs` for file I/O.
//! - **Non-blocking** — writes are buffered in memory and flushed
//!   periodically or on important events.
//! - **Fault-tolerant** — disk-full / permission errors are swallowed
//!   with a stderr warning; the agent never crashes due to logging.
//! - **Daily rotation** — a new file is created each day automatically.
//! - **Retain cleanup** — on startup, files older than `retain_days`
//!   are deleted best-effort without blocking.

use chrono::{Local, NaiveDate};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

// ── LogLevel ─────────────────────────────────────────────────────

/// Severity level for a log event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Info,
    Warn,
    Error,
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Info => write!(f, "info"),
            LogLevel::Warn => write!(f, "warn"),
            LogLevel::Error => write!(f, "error"),
        }
    }
}

impl std::str::FromStr for LogLevel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "info" => Ok(LogLevel::Info),
            "warn" => Ok(LogLevel::Warn),
            "error" => Ok(LogLevel::Error),
            _ => Err(format!("unknown log level: {s}")),
        }
    }
}

// ── EventType ────────────────────────────────────────────────────

/// Classification of a structured log event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventType {
    ToolCall,
    ApiRequest,
    ApiResponse,
    Compaction,
    Failover,
    RecursionDetected,
    MemoryGrowth,
    Error,
    SessionEvent,
}

impl std::fmt::Display for EventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            EventType::ToolCall => "tool_call",
            EventType::ApiRequest => "api_request",
            EventType::ApiResponse => "api_response",
            EventType::Compaction => "compaction",
            EventType::Failover => "failover",
            EventType::RecursionDetected => "recursion_detected",
            EventType::MemoryGrowth => "memory_growth",
            EventType::Error => "error",
            EventType::SessionEvent => "session_event",
        };
        write!(f, "{s}")
    }
}

impl std::str::FromStr for EventType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "tool_call" => Ok(EventType::ToolCall),
            "api_request" => Ok(EventType::ApiRequest),
            "api_response" => Ok(EventType::ApiResponse),
            "compaction" => Ok(EventType::Compaction),
            "failover" => Ok(EventType::Failover),
            "recursion_detected" => Ok(EventType::RecursionDetected),
            "memory_growth" => Ok(EventType::MemoryGrowth),
            "error" => Ok(EventType::Error),
            "session_event" => Ok(EventType::SessionEvent),
            _ => Err(format!("unknown event type: {s}")),
        }
    }
}

// ── LogEvent ─────────────────────────────────────────────────────

/// A single structured log event written as one JSON line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEvent {
    /// ISO 8601 timestamp.
    pub timestamp: String,
    /// Unique identifier for this agent loop invocation.
    pub loop_id: String,
    /// Turn number within the current loop.
    pub turn_number: u32,
    /// What kind of event this is.
    pub event_type: EventType,
    /// LLM provider name (e.g., "anthropic", "openai").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Model name (e.g., "claude-sonnet-4-20250514").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Duration of the operation in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    /// Number of input tokens consumed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    /// Number of output tokens produced.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
    /// Tool name for ToolCall events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// Free-form context string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// Severity level.
    pub level: LogLevel,
}

impl LogEvent {
    /// Create a new LogEvent with required fields; optional fields are None.
    pub fn new(
        loop_id: &str,
        turn_number: u32,
        event_type: EventType,
        level: LogLevel,
    ) -> Self {
        Self {
            timestamp: Local::now().to_rfc3339(),
            loop_id: loop_id.to_string(),
            turn_number,
            event_type,
            provider: None,
            model: None,
            duration_ms: None,
            input_tokens: None,
            output_tokens: None,
            tool_name: None,
            detail: None,
            level,
        }
    }

    /// Builder: set provider.
    pub fn with_provider(mut self, provider: &str) -> Self {
        self.provider = Some(provider.to_string());
        self
    }

    /// Builder: set model.
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }

    /// Builder: set duration_ms.
    pub fn with_duration_ms(mut self, ms: u64) -> Self {
        self.duration_ms = Some(ms);
        self
    }

    /// Builder: set token counts.
    pub fn with_tokens(mut self, input: u32, output: u32) -> Self {
        self.input_tokens = Some(input);
        self.output_tokens = Some(output);
        self
    }

    /// Builder: set tool_name.
    pub fn with_tool_name(mut self, name: &str) -> Self {
        self.tool_name = Some(name.to_string());
        self
    }

    /// Builder: set detail.
    pub fn with_detail(mut self, detail: &str) -> Self {
        self.detail = Some(detail.to_string());
        self
    }

    /// Serialize this event to a single JSON line (no trailing newline).
    pub fn to_json_line(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

// ── LoggingConfig ────────────────────────────────────────────────

/// Configuration for structured logging, loaded from `[logging]` in config.toml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Whether structured JSON logging is enabled.
    #[serde(default = "default_structured")]
    pub structured: bool,
    /// Directory for log files. Supports `~` expansion.
    #[serde(default = "default_log_dir")]
    pub log_dir: String,
    /// Number of days to retain log files.
    #[serde(default = "default_retain_days")]
    pub retain_days: u32,
}

fn default_structured() -> bool {
    true
}

fn default_log_dir() -> String {
    "~/.zeroclaw/logs".to_string()
}

fn default_retain_days() -> u32 {
    14
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            structured: default_structured(),
            log_dir: default_log_dir(),
            retain_days: default_retain_days(),
        }
    }
}

impl LoggingConfig {
    /// Resolve the log directory path, expanding `~` to the home directory.
    pub fn resolved_log_dir(&self) -> PathBuf {
        expand_tilde(&self.log_dir)
    }
}

/// Expand a leading `~` to the user's home directory.
fn expand_tilde(path: &str) -> PathBuf {
    if path.starts_with("~/") || path == "~" {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(&path[2..]);
        }
    }
    PathBuf::from(path)
}

/// Parse a LoggingConfig from the full config.toml contents.
///
/// Returns defaults if the `[logging]` section is absent.
pub fn parse_logging_config(toml_str: &str) -> Result<LoggingConfig, toml::de::Error> {
    #[derive(Deserialize)]
    struct Wrapper {
        #[serde(default)]
        logging: Option<LoggingConfig>,
    }

    let wrapper: Wrapper = toml::from_str(toml_str)?;
    Ok(wrapper.logging.unwrap_or_default())
}

// ── StructuredLogger ─────────────────────────────────────────────

/// Inner state protected by a mutex.
struct LoggerInner {
    /// Buffered events waiting to be flushed.
    buffer: Vec<String>,
    /// The date string of the currently-open log file.
    current_date: String,
    /// Resolved log directory path.
    log_dir: PathBuf,
    /// Whether structured logging is enabled.
    enabled: bool,
}

/// Thread-safe structured logger with buffered writes and daily rotation.
///
/// Clone-friendly via `Arc` — share across the agent loop, tool executors,
/// and background tasks.
#[derive(Clone)]
pub struct StructuredLogger {
    inner: Arc<Mutex<LoggerInner>>,
    config: LoggingConfig,
}

impl StructuredLogger {
    /// Create a new logger from configuration.
    ///
    /// Creates the log directory if it doesn't exist.
    /// Errors during directory creation are logged to stderr but don't
    /// prevent the logger from being created (writes will just fail silently).
    pub fn new(config: LoggingConfig) -> Self {
        let log_dir = config.resolved_log_dir();
        let enabled = config.structured;

        if enabled {
            if let Err(e) = std::fs::create_dir_all(&log_dir) {
                eprintln!(
                    "[zeroclaw] Warning: could not create log directory {}: {e}",
                    log_dir.display()
                );
            }
        }

        let today = Local::now().date_naive().format("%Y-%m-%d").to_string();

        Self {
            inner: Arc::new(Mutex::new(LoggerInner {
                buffer: Vec::new(),
                current_date: today,
                log_dir,
                enabled,
            })),
            config,
        }
    }

    /// Create a disabled logger (for testing or when structured=false).
    pub fn disabled() -> Self {
        Self::new(LoggingConfig {
            structured: false,
            ..Default::default()
        })
    }

    /// Whether this logger is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.structured
    }

    /// Log an event.
    ///
    /// Serializes the event to JSON and buffers it. If the buffer
    /// reaches a threshold, triggers an automatic flush.
    pub fn log(&self, event: &LogEvent) {
        let json = match event.to_json_line() {
            Ok(j) => j,
            Err(e) => {
                eprintln!("[zeroclaw] Warning: failed to serialize log event: {e}");
                return;
            }
        };

        let should_flush;
        {
            let mut inner = match self.inner.lock() {
                Ok(g) => g,
                Err(_) => return, // Poisoned mutex — silently skip
            };

            if !inner.enabled {
                return;
            }

            // Check for day rollover
            let today = Local::now().date_naive().format("%Y-%m-%d").to_string();
            if today != inner.current_date {
                // Flush old day's buffer before rolling
                let _ = Self::flush_locked(&mut inner);
                inner.current_date = today;
            }

            inner.buffer.push(json);
            should_flush = inner.buffer.len() >= 16
                || event.level == LogLevel::Error
                || event.event_type == EventType::Compaction
                || event.event_type == EventType::Failover;
        }

        if should_flush {
            self.flush();
        }
    }

    /// Convenience: log an event and also emit a human-readable message to stderr.
    pub fn log_and_stderr(&self, event: &LogEvent, stderr_msg: &str) {
        eprintln!("{stderr_msg}");
        self.log(event);
    }

    /// Flush all buffered events to disk.
    ///
    /// Errors are reported to stderr but never propagated.
    pub fn flush(&self) {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        let _ = Self::flush_locked(&mut inner);
    }

    /// Internal flush with the lock already held.
    fn flush_locked(inner: &mut LoggerInner) -> std::io::Result<()> {
        if inner.buffer.is_empty() || !inner.enabled {
            return Ok(());
        }

        let path = inner.log_dir.join(format!("{}.jsonl", inner.current_date));

        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            Ok(mut file) => {
                let mut write_err = None;
                for line in &inner.buffer {
                    if let Err(e) = writeln!(file, "{line}") {
                        eprintln!(
                            "[zeroclaw] Warning: log write failed ({}): {e}",
                            path.display()
                        );
                        write_err = Some(e);
                        break;
                    }
                }
                inner.buffer.clear();
                match write_err {
                    Some(e) => Err(e),
                    None => Ok(()),
                }
            }
            Err(e) => {
                eprintln!(
                    "[zeroclaw] Warning: could not open log file {}: {e}",
                    path.display()
                );
                inner.buffer.clear(); // Drop buffered events to avoid OOM
                Err(e)
            }
        }
    }

    /// The log directory path.
    pub fn log_dir(&self) -> PathBuf {
        self.config.resolved_log_dir()
    }

    /// Delete log files older than `retain_days`.
    ///
    /// Best-effort — errors are logged to stderr but don't propagate.
    /// Call this on startup in a non-blocking way.
    pub fn cleanup_old_logs(&self) {
        if !self.config.structured {
            return;
        }

        let log_dir = self.config.resolved_log_dir();
        let retain_days = self.config.retain_days;

        let today = Local::now().date_naive();
        let cutoff = today - chrono::Duration::days(retain_days as i64);

        let entries = match std::fs::read_dir(&log_dir) {
            Ok(e) => e,
            Err(_) => return, // Directory doesn't exist or can't be read
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            let filename = match path.file_stem().and_then(|s| s.to_str()) {
                Some(f) => f,
                None => continue,
            };

            // Only delete files matching YYYY-MM-DD.jsonl pattern
            let ext = path.extension().and_then(|e| e.to_str());
            if ext != Some("jsonl") {
                continue;
            }

            if let Ok(file_date) = NaiveDate::parse_from_str(filename, "%Y-%m-%d") {
                if file_date < cutoff {
                    if let Err(e) = std::fs::remove_file(&path) {
                        eprintln!(
                            "[zeroclaw] Warning: failed to delete old log {}: {e}",
                            path.display()
                        );
                    }
                }
            }
        }
    }

    /// Read the last N log entries from today's (or most recent) log file,
    /// optionally filtered by event type.
    ///
    /// Returns entries in chronological order (oldest first).
    pub fn tail_entries(
        &self,
        count: usize,
        event_type_filter: Option<EventType>,
    ) -> Vec<LogEvent> {
        let log_dir = self.config.resolved_log_dir();

        // Flush first to ensure we read the latest entries
        self.flush();

        // Find log files, sorted by date descending
        let mut log_files = match Self::list_log_files(&log_dir) {
            Ok(files) => files,
            Err(_) => return Vec::new(),
        };
        log_files.sort_by(|a, b| b.cmp(a));

        let mut results = Vec::new();

        for file_path in log_files {
            let content = match std::fs::read_to_string(&file_path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let mut file_events: Vec<LogEvent> = content
                .lines()
                .filter(|l| !l.trim().is_empty())
                .filter_map(|line| serde_json::from_str::<LogEvent>(line).ok())
                .filter(|evt| {
                    event_type_filter
                        .map(|f| evt.event_type == f)
                        .unwrap_or(true)
                })
                .collect();

            // Prepend these (we're going backwards through files)
            file_events.append(&mut results);
            results = file_events;

            if results.len() >= count {
                break;
            }
        }

        // Take the last `count` entries
        let start = results.len().saturating_sub(count);
        results[start..].to_vec()
    }

    /// List all .jsonl log files in the log directory.
    fn list_log_files(log_dir: &Path) -> std::io::Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        for entry in std::fs::read_dir(log_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file()
                && path.extension().and_then(|e| e.to_str()) == Some("jsonl")
            {
                files.push(path);
            }
        }
        Ok(files)
    }

    /// Format log entries for human-readable display.
    pub fn format_entries_for_display(entries: &[LogEvent]) -> String {
        if entries.is_empty() {
            return "No log entries found.".to_string();
        }

        let mut output = String::new();

        for event in entries {
            let ts = if event.timestamp.len() >= 19 {
                &event.timestamp[..19]
            } else {
                &event.timestamp
            };

            let level_tag = match event.level {
                LogLevel::Info => "INFO ",
                LogLevel::Warn => "WARN ",
                LogLevel::Error => "ERROR",
            };

            output.push_str(&format!(
                "{ts}  {level_tag}  {:<20}",
                event.event_type.to_string()
            ));

            if let Some(ref provider) = event.provider {
                output.push_str(&format!("  provider={provider}"));
            }
            if let Some(ref model) = event.model {
                output.push_str(&format!("  model={model}"));
            }
            if let Some(ref tool) = event.tool_name {
                output.push_str(&format!("  tool={tool}"));
            }
            if let Some(ms) = event.duration_ms {
                output.push_str(&format!("  {ms}ms"));
            }
            if let Some(input) = event.input_tokens {
                output.push_str(&format!("  in={input}"));
            }
            if let Some(out) = event.output_tokens {
                output.push_str(&format!("  out={out}"));
            }
            if let Some(ref detail) = event.detail {
                let truncated = if detail.len() > 80 {
                    format!("{}...", &detail[..80])
                } else {
                    detail.clone()
                };
                output.push_str(&format!("  \"{truncated}\""));
            }

            output.push('\n');
        }

        output
    }
}

impl Drop for StructuredLogger {
    fn drop(&mut self) {
        // Best-effort flush on drop
        self.flush();
    }
}

// ── /logs command parsing ────────────────────────────────────────

/// A parsed /logs command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogsCommand {
    /// Number of entries to show.
    pub count: usize,
    /// Optional event type filter.
    pub event_type_filter: Option<EventType>,
}

/// Parse user input for the /logs command.
///
/// Syntax: `/logs [N] [event_type]`
///
/// Examples:
///   `/logs`           — last 20 entries, no filter
///   `/logs 50`        — last 50 entries, no filter
///   `/logs 20 error`  — last 20 error events
///   `/logs error`     — last 20 error events
///
/// Returns `None` if the input is not a /logs command.
pub fn parse_logs_command(input: &str) -> Option<LogsCommand> {
    let trimmed = input.trim();
    let lower = trimmed.to_lowercase();

    if !lower.starts_with("/logs") {
        return None;
    }

    let rest = &trimmed[5..];
    if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
        return None; // e.g., "/logging" should not match
    }

    let parts: Vec<&str> = rest.split_whitespace().collect();

    let mut count = 20usize;
    let mut event_type_filter = None;

    for part in &parts {
        if let Ok(n) = part.parse::<usize>() {
            count = n.min(1000); // Cap at 1000 to prevent abuse
        } else if let Ok(et) = part.parse::<EventType>() {
            event_type_filter = Some(et);
        }
        // Unknown tokens are silently ignored
    }

    Some(LogsCommand {
        count,
        event_type_filter,
    })
}

/// Execute a /logs command against the logger.
pub fn execute_logs_command(cmd: &LogsCommand, logger: &StructuredLogger) -> String {
    let entries = logger.tail_entries(cmd.count, cmd.event_type_filter);

    if entries.is_empty() {
        let filter_desc = cmd
            .event_type_filter
            .map(|et| format!(" (filter: {})", et))
            .unwrap_or_default();
        return format!("No log entries found{filter_desc}.");
    }

    let header = {
        let filter_desc = cmd
            .event_type_filter
            .map(|et| format!(" [{}]", et))
            .unwrap_or_default();
        format!(
            "Last {} log entries{}:\n\n",
            entries.len(),
            filter_desc
        )
    };

    format!("{header}{}", StructuredLogger::format_entries_for_display(&entries))
}

// ── Convenience helpers for instrumentation points ──────────────

/// Log a tool call start event.
pub fn log_tool_call(
    logger: &StructuredLogger,
    loop_id: &str,
    turn: u32,
    tool_name: &str,
    duration_ms: Option<u64>,
    success: bool,
    error_detail: Option<&str>,
) {
    let level = if success { LogLevel::Info } else { LogLevel::Warn };
    let mut event = LogEvent::new(loop_id, turn, EventType::ToolCall, level)
        .with_tool_name(tool_name);

    if let Some(ms) = duration_ms {
        event = event.with_duration_ms(ms);
    }

    if let Some(detail) = error_detail {
        event = event.with_detail(detail);
    } else if !success {
        event = event.with_detail("tool call failed");
    }

    logger.log(&event);
}

/// Log an LLM API request event.
pub fn log_api_request(
    logger: &StructuredLogger,
    loop_id: &str,
    turn: u32,
    provider: &str,
    model: &str,
) {
    let event = LogEvent::new(loop_id, turn, EventType::ApiRequest, LogLevel::Info)
        .with_provider(provider)
        .with_model(model);
    logger.log(&event);
}

/// Log an LLM API response event.
pub fn log_api_response(
    logger: &StructuredLogger,
    loop_id: &str,
    turn: u32,
    provider: &str,
    model: &str,
    duration_ms: u64,
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
) {
    let mut event = LogEvent::new(loop_id, turn, EventType::ApiResponse, LogLevel::Info)
        .with_provider(provider)
        .with_model(model)
        .with_duration_ms(duration_ms);

    if let (Some(inp), Some(out)) = (input_tokens, output_tokens) {
        event = event.with_tokens(inp, out);
    }

    logger.log(&event);
}

/// Log a compaction event.
pub fn log_compaction(
    logger: &StructuredLogger,
    loop_id: &str,
    turn: u32,
    tokens_before: usize,
    tokens_after: usize,
    depth: u32,
    emergency: bool,
) {
    let level = if emergency { LogLevel::Warn } else { LogLevel::Info };
    let kind = if emergency { "emergency" } else { "auto" };
    let detail = format!(
        "{kind} compaction depth={depth}: {tokens_before} -> {tokens_after} tokens"
    );
    let event = LogEvent::new(loop_id, turn, EventType::Compaction, level)
        .with_detail(&detail);
    logger.log(&event);
}

/// Log a failover event.
pub fn log_failover(
    logger: &StructuredLogger,
    loop_id: &str,
    turn: u32,
    from_provider: &str,
    to_provider: &str,
    reason: &str,
) {
    let detail = format!("{from_provider} -> {to_provider}: {reason}");
    let event = LogEvent::new(loop_id, turn, EventType::Failover, LogLevel::Warn)
        .with_provider(from_provider)
        .with_detail(&detail);
    logger.log(&event);
}

/// Log a recursion detection event.
pub fn log_recursion_detected(
    logger: &StructuredLogger,
    loop_id: &str,
    turn: u32,
    tool_name: &str,
    kind: &str,
    forced_stop: bool,
) {
    let level = if forced_stop { LogLevel::Warn } else { LogLevel::Info };
    let detail = format!("{kind}{}", if forced_stop { " (force stop)" } else { " (warn)" });
    let event = LogEvent::new(loop_id, turn, EventType::RecursionDetected, level)
        .with_tool_name(tool_name)
        .with_detail(&detail);
    logger.log(&event);
}

/// Log a memory growth event (splits, compaction, archival).
pub fn log_memory_growth(
    logger: &StructuredLogger,
    loop_id: &str,
    turn: u32,
    detail: &str,
) {
    let event = LogEvent::new(loop_id, turn, EventType::MemoryGrowth, LogLevel::Info)
        .with_detail(detail);
    logger.log(&event);
}

/// Log a session event (create, switch, resume).
pub fn log_session_event(
    logger: &StructuredLogger,
    loop_id: &str,
    turn: u32,
    detail: &str,
) {
    let event = LogEvent::new(loop_id, turn, EventType::SessionEvent, LogLevel::Info)
        .with_detail(detail);
    logger.log(&event);
}

/// Log an error event.
pub fn log_error(
    logger: &StructuredLogger,
    loop_id: &str,
    turn: u32,
    detail: &str,
) {
    let event = LogEvent::new(loop_id, turn, EventType::Error, LogLevel::Error)
        .with_detail(detail);
    logger.log(&event);
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_config(dir: &Path) -> LoggingConfig {
        LoggingConfig {
            structured: true,
            log_dir: dir.to_string_lossy().to_string(),
            retain_days: 14,
        }
    }

    // ── LogEvent serialization round-trip ─────────────────────

    #[test]
    fn log_event_serialization_round_trip() {
        let event = LogEvent::new("test-loop-1", 5, EventType::ToolCall, LogLevel::Info)
            .with_provider("anthropic")
            .with_model("claude-sonnet-4-20250514")
            .with_duration_ms(1234)
            .with_tokens(100, 200)
            .with_tool_name("memory_store")
            .with_detail("stored 3 entries");

        let json = event.to_json_line().unwrap();
        let parsed: LogEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.loop_id, "test-loop-1");
        assert_eq!(parsed.turn_number, 5);
        assert_eq!(parsed.event_type, EventType::ToolCall);
        assert_eq!(parsed.level, LogLevel::Info);
        assert_eq!(parsed.provider.as_deref(), Some("anthropic"));
        assert_eq!(parsed.model.as_deref(), Some("claude-sonnet-4-20250514"));
        assert_eq!(parsed.duration_ms, Some(1234));
        assert_eq!(parsed.input_tokens, Some(100));
        assert_eq!(parsed.output_tokens, Some(200));
        assert_eq!(parsed.tool_name.as_deref(), Some("memory_store"));
        assert_eq!(parsed.detail.as_deref(), Some("stored 3 entries"));
    }

    #[test]
    fn log_event_minimal_serialization() {
        let event = LogEvent::new("loop-2", 0, EventType::Error, LogLevel::Error);
        let json = event.to_json_line().unwrap();
        let parsed: LogEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.event_type, EventType::Error);
        assert_eq!(parsed.level, LogLevel::Error);
        assert!(parsed.provider.is_none());
        assert!(parsed.model.is_none());
        assert!(parsed.duration_ms.is_none());
        assert!(parsed.input_tokens.is_none());
        assert!(parsed.output_tokens.is_none());
        assert!(parsed.tool_name.is_none());
        assert!(parsed.detail.is_none());
    }

    #[test]
    fn log_event_produces_valid_json() {
        let event = LogEvent::new("loop-1", 1, EventType::ApiResponse, LogLevel::Info)
            .with_provider("openai")
            .with_model("gpt-4o")
            .with_duration_ms(500)
            .with_tokens(50, 75);

        let json = event.to_json_line().unwrap();

        // Must be valid JSON
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(value.is_object());
        assert_eq!(value["event_type"], "api_response");
        assert_eq!(value["level"], "info");
        assert_eq!(value["provider"], "openai");
        assert_eq!(value["duration_ms"], 500);
    }

    #[test]
    fn log_event_optional_fields_omitted_from_json() {
        let event = LogEvent::new("loop-1", 0, EventType::SessionEvent, LogLevel::Info);
        let json = event.to_json_line().unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        // Optional fields should not be present in the JSON
        assert!(value.get("provider").is_none());
        assert!(value.get("model").is_none());
        assert!(value.get("duration_ms").is_none());
        assert!(value.get("input_tokens").is_none());
        assert!(value.get("output_tokens").is_none());
        assert!(value.get("tool_name").is_none());
        assert!(value.get("detail").is_none());
    }

    // ── EventType / LogLevel parsing ─────────────────────────

    #[test]
    fn event_type_round_trip() {
        let types = [
            EventType::ToolCall,
            EventType::ApiRequest,
            EventType::ApiResponse,
            EventType::Compaction,
            EventType::Failover,
            EventType::RecursionDetected,
            EventType::MemoryGrowth,
            EventType::Error,
            EventType::SessionEvent,
        ];
        for et in &types {
            let s = et.to_string();
            let parsed: EventType = s.parse().unwrap();
            assert_eq!(*et, parsed);
        }
    }

    #[test]
    fn log_level_round_trip() {
        for level in &[LogLevel::Info, LogLevel::Warn, LogLevel::Error] {
            let s = level.to_string();
            let parsed: LogLevel = s.parse().unwrap();
            assert_eq!(*level, parsed);
        }
    }

    #[test]
    fn event_type_parse_unknown_errors() {
        assert!("unknown_type".parse::<EventType>().is_err());
    }

    #[test]
    fn log_level_parse_unknown_errors() {
        assert!("debug".parse::<LogLevel>().is_err());
    }

    // ── LoggingConfig ────────────────────────────────────────

    #[test]
    fn config_defaults() {
        let config = LoggingConfig::default();
        assert!(config.structured);
        assert_eq!(config.log_dir, "~/.zeroclaw/logs");
        assert_eq!(config.retain_days, 14);
    }

    #[test]
    fn config_deserialization() {
        let toml_str = r#"
            structured = false
            log_dir = "/tmp/zeroclaw-logs"
            retain_days = 7
        "#;
        let config: LoggingConfig = toml::from_str(toml_str).unwrap();
        assert!(!config.structured);
        assert_eq!(config.log_dir, "/tmp/zeroclaw-logs");
        assert_eq!(config.retain_days, 7);
    }

    #[test]
    fn config_partial_deserialization_uses_defaults() {
        let toml_str = r#"
            retain_days = 30
        "#;
        let config: LoggingConfig = toml::from_str(toml_str).unwrap();
        assert!(config.structured); // default
        assert_eq!(config.log_dir, "~/.zeroclaw/logs"); // default
        assert_eq!(config.retain_days, 30);
    }

    #[test]
    fn parse_logging_config_from_full_toml() {
        let toml_str = r#"
[provider]
primary = "anthropic"

[logging]
structured = true
log_dir = "/var/log/zeroclaw"
retain_days = 7
"#;
        let config = parse_logging_config(toml_str).unwrap();
        assert!(config.structured);
        assert_eq!(config.log_dir, "/var/log/zeroclaw");
        assert_eq!(config.retain_days, 7);
    }

    #[test]
    fn parse_logging_config_missing_section_defaults() {
        let toml_str = r#"
[provider]
primary = "anthropic"
"#;
        let config = parse_logging_config(toml_str).unwrap();
        assert!(config.structured);
        assert_eq!(config.log_dir, "~/.zeroclaw/logs");
        assert_eq!(config.retain_days, 14);
    }

    #[test]
    fn expand_tilde_expands_home() {
        std::env::set_var("HOME", "/home/testuser");
        let p = expand_tilde("~/.zeroclaw/logs");
        assert_eq!(p, PathBuf::from("/home/testuser/.zeroclaw/logs"));
    }

    #[test]
    fn expand_tilde_no_tilde_unchanged() {
        let p = expand_tilde("/tmp/logs");
        assert_eq!(p, PathBuf::from("/tmp/logs"));
    }

    // ── Daily file rotation logic ────────────────────────────

    #[test]
    fn daily_file_rotation_creates_correct_filename() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(tmp.path());
        let logger = StructuredLogger::new(config);

        let event = LogEvent::new("loop-1", 0, EventType::SessionEvent, LogLevel::Info)
            .with_detail("test rotation");
        logger.log(&event);
        logger.flush();

        let today = Local::now().date_naive().format("%Y-%m-%d").to_string();
        let expected_path = tmp.path().join(format!("{today}.jsonl"));
        assert!(expected_path.exists(), "Log file should exist: {}", expected_path.display());

        let content = std::fs::read_to_string(&expected_path).unwrap();
        assert!(!content.is_empty());

        // Each line should be valid JSON
        for line in content.lines() {
            let _: serde_json::Value = serde_json::from_str(line)
                .expect("Each line should be valid JSON");
        }
    }

    #[test]
    fn multiple_events_written_to_same_file() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(tmp.path());
        let logger = StructuredLogger::new(config);

        for i in 0..5 {
            let event = LogEvent::new("loop-1", i, EventType::ToolCall, LogLevel::Info)
                .with_tool_name(&format!("tool_{i}"));
            logger.log(&event);
        }
        logger.flush();

        let today = Local::now().date_naive().format("%Y-%m-%d").to_string();
        let path = tmp.path().join(format!("{today}.jsonl"));
        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 5);
    }

    // ── retain_days cleanup ──────────────────────────────────

    #[test]
    fn cleanup_deletes_old_files() {
        let tmp = TempDir::new().unwrap();
        let config = LoggingConfig {
            structured: true,
            log_dir: tmp.path().to_string_lossy().to_string(),
            retain_days: 7,
        };

        // Create fake log files: one old, one recent
        let old_date = (Local::now().date_naive() - chrono::Duration::days(10))
            .format("%Y-%m-%d")
            .to_string();
        let recent_date = (Local::now().date_naive() - chrono::Duration::days(3))
            .format("%Y-%m-%d")
            .to_string();
        let today = Local::now().date_naive().format("%Y-%m-%d").to_string();

        std::fs::write(tmp.path().join(format!("{old_date}.jsonl")), "old data\n").unwrap();
        std::fs::write(tmp.path().join(format!("{recent_date}.jsonl")), "recent data\n").unwrap();
        std::fs::write(tmp.path().join(format!("{today}.jsonl")), "today data\n").unwrap();

        let logger = StructuredLogger::new(config);
        logger.cleanup_old_logs();

        // Old file should be deleted
        assert!(
            !tmp.path().join(format!("{old_date}.jsonl")).exists(),
            "Old log file should be deleted"
        );
        // Recent and today files should remain
        assert!(
            tmp.path().join(format!("{recent_date}.jsonl")).exists(),
            "Recent log file should remain"
        );
        assert!(
            tmp.path().join(format!("{today}.jsonl")).exists(),
            "Today's log file should remain"
        );
    }

    #[test]
    fn cleanup_ignores_non_jsonl_files() {
        let tmp = TempDir::new().unwrap();
        let config = LoggingConfig {
            structured: true,
            log_dir: tmp.path().to_string_lossy().to_string(),
            retain_days: 0, // Delete everything
        };

        let old_date = (Local::now().date_naive() - chrono::Duration::days(5))
            .format("%Y-%m-%d")
            .to_string();

        std::fs::write(tmp.path().join(format!("{old_date}.jsonl")), "data\n").unwrap();
        std::fs::write(tmp.path().join(format!("{old_date}.txt")), "other\n").unwrap();
        std::fs::write(tmp.path().join("config.toml"), "config\n").unwrap();

        let logger = StructuredLogger::new(config);
        logger.cleanup_old_logs();

        // Only the .jsonl file should be deleted
        assert!(!tmp.path().join(format!("{old_date}.jsonl")).exists());
        assert!(tmp.path().join(format!("{old_date}.txt")).exists());
        assert!(tmp.path().join("config.toml").exists());
    }

    #[test]
    fn cleanup_handles_nonexistent_dir() {
        let config = LoggingConfig {
            structured: true,
            log_dir: "/nonexistent/path/logs".to_string(),
            retain_days: 7,
        };
        let logger = StructuredLogger::new(LoggingConfig {
            structured: false,
            ..config.clone()
        });
        // Should not panic
        logger.cleanup_old_logs();
    }

    // ── log_event helper produces valid JSON ─────────────────

    #[test]
    fn log_helper_produces_valid_json_file() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(tmp.path());
        let logger = StructuredLogger::new(config);

        log_tool_call(&logger, "loop-1", 1, "memory_store", Some(42), true, None);
        log_api_request(&logger, "loop-1", 1, "anthropic", "claude-sonnet-4-20250514");
        log_api_response(&logger, "loop-1", 1, "anthropic", "claude-sonnet-4-20250514", 1500, Some(100), Some(200));
        log_compaction(&logger, "loop-1", 2, 150000, 30000, 1, false);
        log_failover(&logger, "loop-1", 2, "anthropic", "openai", "503 Service Unavailable");
        log_recursion_detected(&logger, "loop-1", 3, "file_read", "identical_call", false);
        log_memory_growth(&logger, "loop-1", 3, "split MEMORY.md into 3 topic files");
        log_session_event(&logger, "loop-1", 0, "created session: my-project");
        log_error(&logger, "loop-1", 4, "connection timeout after 30s");

        logger.flush();

        let today = Local::now().date_naive().format("%Y-%m-%d").to_string();
        let path = tmp.path().join(format!("{today}.jsonl"));
        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().filter(|l| !l.is_empty()).collect();

        assert_eq!(lines.len(), 9);

        // Every line must be valid JSON and deserialize to LogEvent
        for (i, line) in lines.iter().enumerate() {
            let event: LogEvent = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("Line {i} is not valid LogEvent JSON: {e}"));
            assert!(!event.timestamp.is_empty());
            assert!(!event.loop_id.is_empty());
        }
    }

    // ── Disabled logger ──────────────────────────────────────

    #[test]
    fn disabled_logger_does_not_write() {
        let tmp = TempDir::new().unwrap();
        let config = LoggingConfig {
            structured: false,
            log_dir: tmp.path().to_string_lossy().to_string(),
            retain_days: 14,
        };
        let logger = StructuredLogger::new(config);

        let event = LogEvent::new("loop-1", 0, EventType::Error, LogLevel::Error)
            .with_detail("should not be written");
        logger.log(&event);
        logger.flush();

        // No files should be created
        let files: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .collect();
        assert!(files.is_empty(), "Disabled logger should not create files");
    }

    // ── Tail entries ─────────────────────────────────────────

    #[test]
    fn tail_entries_returns_last_n() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(tmp.path());
        let logger = StructuredLogger::new(config);

        for i in 0..10u32 {
            let event = LogEvent::new("loop-1", i, EventType::ToolCall, LogLevel::Info)
                .with_tool_name(&format!("tool_{i}"));
            logger.log(&event);
        }
        logger.flush();

        let entries = logger.tail_entries(5, None);
        assert_eq!(entries.len(), 5);
        // Should be the last 5 (turns 5-9)
        assert_eq!(entries[0].turn_number, 5);
        assert_eq!(entries[4].turn_number, 9);
    }

    #[test]
    fn tail_entries_with_filter() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(tmp.path());
        let logger = StructuredLogger::new(config);

        // Mix of event types
        logger.log(&LogEvent::new("loop-1", 0, EventType::ToolCall, LogLevel::Info));
        logger.log(&LogEvent::new("loop-1", 1, EventType::Error, LogLevel::Error));
        logger.log(&LogEvent::new("loop-1", 2, EventType::ToolCall, LogLevel::Info));
        logger.log(&LogEvent::new("loop-1", 3, EventType::Error, LogLevel::Error));
        logger.log(&LogEvent::new("loop-1", 4, EventType::ApiResponse, LogLevel::Info));
        logger.flush();

        let errors = logger.tail_entries(10, Some(EventType::Error));
        assert_eq!(errors.len(), 2);
        assert!(errors.iter().all(|e| e.event_type == EventType::Error));
    }

    #[test]
    fn tail_entries_empty_log() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(tmp.path());
        let logger = StructuredLogger::new(config);

        let entries = logger.tail_entries(10, None);
        assert!(entries.is_empty());
    }

    // ── Display formatting ───────────────────────────────────

    #[test]
    fn format_entries_for_display_produces_readable_output() {
        let events = vec![
            LogEvent::new("loop-1", 0, EventType::SessionEvent, LogLevel::Info)
                .with_detail("created session: test"),
            LogEvent::new("loop-1", 1, EventType::ApiResponse, LogLevel::Info)
                .with_provider("anthropic")
                .with_model("claude-sonnet-4-20250514")
                .with_duration_ms(1500)
                .with_tokens(100, 200),
            LogEvent::new("loop-1", 2, EventType::Error, LogLevel::Error)
                .with_detail("connection timeout"),
        ];

        let output = StructuredLogger::format_entries_for_display(&events);
        assert!(output.contains("session_event"));
        assert!(output.contains("api_response"));
        assert!(output.contains("ERROR"));
        assert!(output.contains("anthropic"));
        assert!(output.contains("1500ms"));
        assert!(output.contains("connection timeout"));
    }

    #[test]
    fn format_entries_empty() {
        let output = StructuredLogger::format_entries_for_display(&[]);
        assert_eq!(output, "No log entries found.");
    }

    // ── /logs command parsing ────────────────────────────────

    #[test]
    fn parse_logs_command_basic() {
        let cmd = parse_logs_command("/logs").unwrap();
        assert_eq!(cmd.count, 20);
        assert!(cmd.event_type_filter.is_none());
    }

    #[test]
    fn parse_logs_command_with_count() {
        let cmd = parse_logs_command("/logs 50").unwrap();
        assert_eq!(cmd.count, 50);
        assert!(cmd.event_type_filter.is_none());
    }

    #[test]
    fn parse_logs_command_with_filter() {
        let cmd = parse_logs_command("/logs error").unwrap();
        assert_eq!(cmd.count, 20);
        assert_eq!(cmd.event_type_filter, Some(EventType::Error));
    }

    #[test]
    fn parse_logs_command_with_count_and_filter() {
        let cmd = parse_logs_command("/logs 30 error").unwrap();
        assert_eq!(cmd.count, 30);
        assert_eq!(cmd.event_type_filter, Some(EventType::Error));
    }

    #[test]
    fn parse_logs_command_filter_first() {
        let cmd = parse_logs_command("/logs failover 10").unwrap();
        assert_eq!(cmd.count, 10);
        assert_eq!(cmd.event_type_filter, Some(EventType::Failover));
    }

    #[test]
    fn parse_logs_command_non_command_returns_none() {
        assert!(parse_logs_command("hello").is_none());
        assert!(parse_logs_command("/logging").is_none());
        assert!(parse_logs_command("").is_none());
        assert!(parse_logs_command("/new something").is_none());
    }

    #[test]
    fn parse_logs_command_caps_at_1000() {
        let cmd = parse_logs_command("/logs 9999").unwrap();
        assert_eq!(cmd.count, 1000);
    }

    #[test]
    fn parse_logs_command_all_event_types() {
        for et_str in &[
            "tool_call", "api_request", "api_response", "compaction",
            "failover", "recursion_detected", "memory_growth", "error",
            "session_event",
        ] {
            let input = format!("/logs {et_str}");
            let cmd = parse_logs_command(&input).unwrap();
            assert!(cmd.event_type_filter.is_some(), "Should parse {et_str}");
        }
    }

    // ── Execute /logs command ────────────────────────────────

    #[test]
    fn execute_logs_command_with_entries() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(tmp.path());
        let logger = StructuredLogger::new(config);

        for i in 0..5 {
            let event = LogEvent::new("loop-1", i, EventType::ToolCall, LogLevel::Info)
                .with_tool_name(&format!("tool_{i}"));
            logger.log(&event);
        }

        let cmd = LogsCommand {
            count: 3,
            event_type_filter: None,
        };
        let output = execute_logs_command(&cmd, &logger);
        assert!(output.contains("3 log entries"));
        assert!(output.contains("tool_call"));
    }

    #[test]
    fn execute_logs_command_empty() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(tmp.path());
        let logger = StructuredLogger::new(config);

        let cmd = LogsCommand {
            count: 10,
            event_type_filter: None,
        };
        let output = execute_logs_command(&cmd, &logger);
        assert!(output.contains("No log entries found"));
    }

    #[test]
    fn execute_logs_command_with_filter_no_match() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(tmp.path());
        let logger = StructuredLogger::new(config);

        // Only tool_call events
        logger.log(&LogEvent::new("loop-1", 0, EventType::ToolCall, LogLevel::Info));
        logger.flush();

        let cmd = LogsCommand {
            count: 10,
            event_type_filter: Some(EventType::Error),
        };
        let output = execute_logs_command(&cmd, &logger);
        assert!(output.contains("No log entries found"));
        assert!(output.contains("error"));
    }

    // ── Thread safety ────────────────────────────────────────

    #[test]
    fn logger_is_clone_and_send() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(tmp.path());
        let logger = StructuredLogger::new(config);

        // Clone and use from multiple threads
        let handles: Vec<_> = (0..4)
            .map(|thread_id| {
                let logger = logger.clone();
                std::thread::spawn(move || {
                    for i in 0..10 {
                        let event = LogEvent::new(
                            &format!("thread-{thread_id}"),
                            i,
                            EventType::ToolCall,
                            LogLevel::Info,
                        );
                        logger.log(&event);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        logger.flush();

        let today = Local::now().date_naive().format("%Y-%m-%d").to_string();
        let path = tmp.path().join(format!("{today}.jsonl"));
        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 40); // 4 threads * 10 events
    }
}

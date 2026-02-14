//! Session persistence for ZeroClaw.
//!
//! Writes every completed conversation turn to a JSONL file so that
//! sessions survive process restarts. Each session file lives at:
//!
//! ```text
//! ~/.zeroclaw/sessions/YYYY-MM-DD_{session_id}.jsonl
//! ```
//!
//! Each line is a self-contained JSON object with:
//! `role`, `content`, `tool_calls`, `tool_results`, `timestamp`
//!
//! On startup the most recent session from today can be resumed,
//! restoring full message history into the agent loop.

use anyhow::{Context, Result};
use chrono::{Local, NaiveDate};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

// ── Persisted turn types ─────────────────────────────────────────

/// A single tool call as persisted to the session file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// A single tool result as persisted to the session file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedToolResult {
    pub tool_call_id: String,
    pub content: String,
}

/// One line in the JSONL session file — a completed conversation turn.
///
/// Designed to be human-readable when inspected with `cat` or `grep`:
/// each field is present but may be null/empty when not applicable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTurn {
    /// "user", "assistant", or "tool_result"
    pub role: String,
    /// Text content of the turn (may be null for pure tool-call turns).
    pub content: Option<String>,
    /// Tool calls made by the assistant (empty for user/tool_result turns).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<PersistedToolCall>,
    /// Tool results returned to the assistant (empty except for tool_result turns).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_results: Vec<PersistedToolResult>,
    /// ISO-8601 timestamp of when the turn completed.
    pub timestamp: String,
}

/// Metadata about a session file, used by the `/sessions` listing.
#[derive(Debug, Clone)]
pub struct SessionInfo {
    /// The session ID (UUID portion of the filename).
    pub session_id: String,
    /// Date the session was created.
    pub date: NaiveDate,
    /// Number of turns (lines) in the file.
    pub turn_count: usize,
    /// Full path to the session file.
    pub path: PathBuf,
}

// ── SessionManager ───────────────────────────────────────────────

/// Manages reading and writing session JSONL files.
///
/// Each session gets a unique file:
/// `{sessions_dir}/YYYY-MM-DD_{session_id}.jsonl`
///
/// The manager can:
/// - Create new sessions
/// - Append turns to the current session
/// - Find and load the most recent session for today
/// - List recent sessions with metadata
pub struct SessionManager {
    /// Directory where session files are stored (typically ~/.zeroclaw/sessions/).
    sessions_dir: PathBuf,
    /// Current session ID (set on new_session or resume).
    session_id: Option<String>,
    /// Current session file path.
    session_path: Option<PathBuf>,
}

impl SessionManager {
    /// Create a new SessionManager.
    ///
    /// `base_dir` is the zeroclaw workspace root (e.g., `~/.zeroclaw`).
    /// Session files go into `{base_dir}/sessions/`.
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            sessions_dir: base_dir.into().join("sessions"),
            session_id: None,
            session_path: None,
        }
    }

    /// The directory where all session files live.
    pub fn sessions_dir(&self) -> &Path {
        &self.sessions_dir
    }

    /// The current session ID, if any.
    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    /// The current session file path, if any.
    pub fn session_path(&self) -> Option<&Path> {
        self.session_path.as_deref()
    }

    // ── Directory setup ──────────────────────────────────────────

    /// Ensure the sessions directory exists.
    async fn ensure_sessions_dir(&self) -> Result<()> {
        if !self.sessions_dir.exists() {
            fs::create_dir_all(&self.sessions_dir)
                .await
                .with_context(|| {
                    format!(
                        "Failed to create sessions dir: {}",
                        self.sessions_dir.display()
                    )
                })?;
        }
        Ok(())
    }

    // ── Session lifecycle ────────────────────────────────────────

    /// Start a brand-new session. Generates a new UUID and creates the file.
    pub async fn new_session(&mut self) -> Result<String> {
        self.ensure_sessions_dir().await?;

        let id = Uuid::new_v4().to_string()[..8].to_string(); // short ID
        let date = Local::now().date_naive().format("%Y-%m-%d");
        let filename = format!("{date}_{id}.jsonl");
        let path = self.sessions_dir.join(&filename);

        // Touch the file so it exists
        fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(&path)
            .await
            .with_context(|| format!("Failed to create session file: {}", path.display()))?;

        self.session_id = Some(id.clone());
        self.session_path = Some(path);

        Ok(id)
    }

    /// Resume an existing session by loading its turns.
    ///
    /// Sets this session as the active one and returns all persisted turns.
    pub async fn resume_session(&mut self, path: &Path) -> Result<Vec<SessionTurn>> {
        let filename = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        let id = extract_session_id(filename);

        self.session_id = Some(id.to_string());
        self.session_path = Some(path.to_path_buf());

        self.load_turns(path).await
    }

    /// Append a completed turn to the current session file.
    ///
    /// Each turn is written as a single JSON line followed by a newline.
    /// This is called after every completed turn (not mid-stream).
    pub async fn append_turn(&self, turn: &SessionTurn) -> Result<()> {
        let path = self
            .session_path
            .as_ref()
            .context("No active session — call new_session() or resume_session() first")?;

        let mut line = serde_json::to_string(turn)
            .context("Failed to serialize session turn")?;
        line.push('\n');

        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await
            .with_context(|| format!("Failed to open session file: {}", path.display()))?;

        file.write_all(line.as_bytes())
            .await
            .with_context(|| format!("Failed to write to session file: {}", path.display()))?;

        Ok(())
    }

    // ── Convenience builders for SessionTurn ─────────────────────

    /// Build a SessionTurn for a user message.
    pub fn user_turn(content: &str) -> SessionTurn {
        SessionTurn {
            role: "user".to_string(),
            content: Some(content.to_string()),
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            timestamp: Local::now().to_rfc3339(),
        }
    }

    /// Build a SessionTurn for an assistant response (with optional tool calls).
    pub fn assistant_turn(
        content: Option<&str>,
        tool_calls: Vec<PersistedToolCall>,
    ) -> SessionTurn {
        SessionTurn {
            role: "assistant".to_string(),
            content: content.map(|s| s.to_string()),
            tool_calls,
            tool_results: Vec::new(),
            timestamp: Local::now().to_rfc3339(),
        }
    }

    /// Build a SessionTurn for tool results.
    pub fn tool_result_turn(results: Vec<PersistedToolResult>) -> SessionTurn {
        SessionTurn {
            role: "tool_result".to_string(),
            content: None,
            tool_calls: Vec::new(),
            tool_results: results,
            timestamp: Local::now().to_rfc3339(),
        }
    }

    // ── Session discovery ────────────────────────────────────────

    /// Find the most recent session file from today, if one exists.
    ///
    /// Returns `None` if there are no sessions from today.
    pub async fn find_todays_latest(&self) -> Result<Option<PathBuf>> {
        let today = Local::now().date_naive().format("%Y-%m-%d").to_string();
        let sessions = self.list_session_files().await?;

        // Files are named YYYY-MM-DD_{id}.jsonl — filter to today, pick latest by mtime
        let mut todays: Vec<PathBuf> = sessions
            .into_iter()
            .filter(|p| {
                p.file_name()
                    .and_then(|f| f.to_str())
                    .map(|f| f.starts_with(&today))
                    .unwrap_or(false)
            })
            .collect();

        if todays.is_empty() {
            return Ok(None);
        }

        // Sort by modification time (most recent last)
        todays.sort_by_key(|p| {
            std::fs::metadata(p)
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        });

        Ok(todays.last().cloned())
    }

    /// List recent sessions with metadata (for the `/sessions` command).
    ///
    /// Returns up to `limit` sessions, most recent first.
    pub async fn list_sessions(&self, limit: usize) -> Result<Vec<SessionInfo>> {
        let mut files = self.list_session_files().await?;

        // Sort by modification time, most recent first
        files.sort_by_key(|p| {
            std::cmp::Reverse(
                std::fs::metadata(p)
                    .and_then(|m| m.modified())
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
            )
        });

        files.truncate(limit);

        let mut infos = Vec::new();
        for path in files {
            if let Some(info) = self.session_info(&path).await? {
                infos.push(info);
            }
        }

        Ok(infos)
    }

    /// Check whether a session file is from today.
    pub fn is_from_today(path: &Path) -> bool {
        let today = Local::now().date_naive().format("%Y-%m-%d").to_string();
        path.file_name()
            .and_then(|f| f.to_str())
            .map(|f| f.starts_with(&today))
            .unwrap_or(false)
    }

    // ── Internal helpers ─────────────────────────────────────────

    /// Load all turns from a session JSONL file.
    async fn load_turns(&self, path: &Path) -> Result<Vec<SessionTurn>> {
        let content = fs::read_to_string(path)
            .await
            .with_context(|| format!("Failed to read session file: {}", path.display()))?;

        let mut turns = Vec::new();
        for (line_num, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let turn: SessionTurn = serde_json::from_str(line).with_context(|| {
                format!(
                    "Failed to parse session turn at line {} in {}",
                    line_num + 1,
                    path.display()
                )
            })?;
            turns.push(turn);
        }

        Ok(turns)
    }

    /// List all .jsonl files in the sessions directory.
    async fn list_session_files(&self) -> Result<Vec<PathBuf>> {
        if !self.sessions_dir.exists() {
            return Ok(Vec::new());
        }

        let mut entries = fs::read_dir(&self.sessions_dir)
            .await
            .with_context(|| {
                format!(
                    "Failed to read sessions dir: {}",
                    self.sessions_dir.display()
                )
            })?;

        let mut files = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                files.push(path);
            }
        }

        Ok(files)
    }

    /// Build SessionInfo metadata for a single session file.
    async fn session_info(&self, path: &Path) -> Result<Option<SessionInfo>> {
        let filename = match path.file_stem().and_then(|s| s.to_str()) {
            Some(f) => f,
            None => return Ok(None),
        };

        // Parse date from YYYY-MM-DD_{id}
        let date = if filename.len() >= 10 {
            NaiveDate::parse_from_str(&filename[..10], "%Y-%m-%d").ok()
        } else {
            None
        };

        let date = match date {
            Some(d) => d,
            None => return Ok(None),
        };

        let session_id = extract_session_id(filename).to_string();

        // Count lines (turns) in the file
        let content = fs::read_to_string(path).await.unwrap_or_default();
        let turn_count = content.lines().filter(|l| !l.trim().is_empty()).count();

        Ok(Some(SessionInfo {
            session_id,
            date,
            turn_count,
            path: path.to_path_buf(),
        }))
    }
}

/// Extract the session ID from a filename like `2026-02-14_a1b2c3d4`.
///
/// Returns the portion after the date and underscore separator.
fn extract_session_id(filename: &str) -> &str {
    // Filename: YYYY-MM-DD_SESSIONID
    // The date is always 10 chars, followed by underscore
    if filename.len() > 11 && filename.as_bytes()[10] == b'_' {
        &filename[11..]
    } else {
        filename
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn setup() -> (TempDir, SessionManager) {
        let tmp = TempDir::new().unwrap();
        let mgr = SessionManager::new(tmp.path());
        (tmp, mgr)
    }

    #[tokio::test]
    async fn new_session_creates_file() {
        let (_tmp, mut mgr) = setup().await;
        let id = mgr.new_session().await.unwrap();

        assert!(!id.is_empty());
        assert!(mgr.session_path().unwrap().exists());
        assert!(mgr
            .session_path()
            .unwrap()
            .to_str()
            .unwrap()
            .ends_with(".jsonl"));
    }

    #[tokio::test]
    async fn append_and_load_turns() {
        let (_tmp, mut mgr) = setup().await;
        mgr.new_session().await.unwrap();

        // Write a user turn
        let user = SessionManager::user_turn("Hello, ZeroClaw!");
        mgr.append_turn(&user).await.unwrap();

        // Write an assistant turn
        let assistant = SessionManager::assistant_turn(Some("Hi! How can I help?"), vec![]);
        mgr.append_turn(&assistant).await.unwrap();

        // Write an assistant turn with tool calls
        let tc = PersistedToolCall {
            id: "tc_1".to_string(),
            name: "memory_store".to_string(),
            arguments: serde_json::json!({"content": "test"}),
        };
        let assistant_tc = SessionManager::assistant_turn(None, vec![tc]);
        mgr.append_turn(&assistant_tc).await.unwrap();

        // Write a tool result turn
        let tr = PersistedToolResult {
            tool_call_id: "tc_1".to_string(),
            content: "Stored successfully.".to_string(),
        };
        let tool_result = SessionManager::tool_result_turn(vec![tr]);
        mgr.append_turn(&tool_result).await.unwrap();

        // Load turns back
        let path = mgr.session_path().unwrap().to_path_buf();
        let turns = mgr.load_turns(&path).await.unwrap();

        assert_eq!(turns.len(), 4);
        assert_eq!(turns[0].role, "user");
        assert_eq!(turns[0].content.as_deref(), Some("Hello, ZeroClaw!"));
        assert_eq!(turns[1].role, "assistant");
        assert_eq!(turns[1].content.as_deref(), Some("Hi! How can I help?"));
        assert_eq!(turns[2].role, "assistant");
        assert_eq!(turns[2].tool_calls.len(), 1);
        assert_eq!(turns[2].tool_calls[0].name, "memory_store");
        assert_eq!(turns[3].role, "tool_result");
        assert_eq!(turns[3].tool_results.len(), 1);
    }

    #[tokio::test]
    async fn find_todays_latest_returns_most_recent() {
        let (_tmp, mut mgr) = setup().await;

        // Create two sessions
        let _id1 = mgr.new_session().await.unwrap();
        let path1 = mgr.session_path().unwrap().to_path_buf();

        // Small delay to ensure different mtime
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let id2 = mgr.new_session().await.unwrap();
        let path2 = mgr.session_path().unwrap().to_path_buf();

        // Write something to path2 so it has a later mtime
        let turn = SessionManager::user_turn("hello");
        mgr.append_turn(&turn).await.unwrap();

        let latest = mgr.find_todays_latest().await.unwrap();
        assert!(latest.is_some());
        assert_eq!(latest.unwrap(), path2);
    }

    #[tokio::test]
    async fn list_sessions_returns_metadata() {
        let (_tmp, mut mgr) = setup().await;

        mgr.new_session().await.unwrap();
        let turn = SessionManager::user_turn("hello");
        mgr.append_turn(&turn).await.unwrap();
        let turn = SessionManager::assistant_turn(Some("world"), vec![]);
        mgr.append_turn(&turn).await.unwrap();

        let sessions = mgr.list_sessions(10).await.unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].turn_count, 2);
        assert_eq!(sessions[0].date, Local::now().date_naive());
    }

    #[tokio::test]
    async fn resume_session_loads_turns() {
        let (_tmp, mut mgr) = setup().await;
        mgr.new_session().await.unwrap();

        let user = SessionManager::user_turn("First message");
        mgr.append_turn(&user).await.unwrap();
        let path = mgr.session_path().unwrap().to_path_buf();

        // Create a new manager and resume
        let mut mgr2 = SessionManager::new(_tmp.path());
        let turns = mgr2.resume_session(&path).await.unwrap();

        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].content.as_deref(), Some("First message"));
        assert!(mgr2.session_id().is_some());
    }

    #[tokio::test]
    async fn session_files_are_human_readable() {
        let (_tmp, mut mgr) = setup().await;
        mgr.new_session().await.unwrap();

        let turn = SessionManager::user_turn("Can you help me debug this?");
        mgr.append_turn(&turn).await.unwrap();

        // Read raw file and verify it's grep-friendly JSONL
        let content = fs::read_to_string(mgr.session_path().unwrap())
            .await
            .unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 1);

        // Each line is valid JSON
        let parsed: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(parsed["role"], "user");
        assert!(parsed["content"]
            .as_str()
            .unwrap()
            .contains("debug this"));
        assert!(parsed["timestamp"].as_str().is_some());
    }

    #[tokio::test]
    async fn extract_session_id_works() {
        assert_eq!(extract_session_id("2026-02-14_a1b2c3d4"), "a1b2c3d4");
        assert_eq!(extract_session_id("2026-02-14_longid99"), "longid99");
        assert_eq!(extract_session_id("short"), "short"); // fallback
    }

    #[tokio::test]
    async fn is_from_today_checks_date() {
        let today = Local::now().date_naive().format("%Y-%m-%d").to_string();
        let today_file = PathBuf::from(format!("{today}_abc123.jsonl"));
        let old_file = PathBuf::from("2020-01-01_abc123.jsonl");

        assert!(SessionManager::is_from_today(&today_file));
        assert!(!SessionManager::is_from_today(&old_file));
    }

    #[tokio::test]
    async fn empty_session_has_zero_turns() {
        let (_tmp, mut mgr) = setup().await;
        mgr.new_session().await.unwrap();

        let sessions = mgr.list_sessions(10).await.unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].turn_count, 0);
    }
}

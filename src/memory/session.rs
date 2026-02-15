//! Session persistence for ZeroClaw.
//!
//! Writes every completed conversation turn to a JSONL file so that
//! sessions survive process restarts. Each named session lives at:
//!
//! ```text
//! ~/.zeroclaw/sessions/<name>/session.jsonl
//! ```
//!
//! Each line is a self-contained JSON object with:
//! `role`, `content`, `tool_calls`, `tool_results`, `timestamp`
//!
//! On startup the most recently active session (across all names) is
//! resumed. Named sessions can be created, switched, renamed, and
//! listed via local commands.

use anyhow::{bail, Context, Result};
use chrono::Local;
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

/// Metadata about a named session, used by `/sessions` listing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    /// The session name (slug).
    pub name: String,
    /// ISO-8601 timestamp of when the session was created.
    pub created_at: String,
    /// ISO-8601 timestamp of the most recent activity.
    pub last_active: String,
    /// Optional provider name (e.g. "anthropic").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Optional model name (e.g. "claude-3-opus").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Information about a session for listing purposes.
#[derive(Debug, Clone)]
pub struct SessionInfo {
    /// The session name (slug).
    pub name: String,
    /// ISO-8601 timestamp of last activity.
    pub last_active: String,
    /// Number of turns (lines) in the session file.
    pub turn_count: usize,
    /// Full path to the session directory.
    pub path: PathBuf,
}

// ── Slug helper ──────────────────────────────────────────────────

/// Slugify a session name: lowercase, replace non-alphanumeric with
/// hyphens, collapse multiple hyphens, trim leading/trailing hyphens.
pub fn slugify(name: &str) -> String {
    let slug: String = name
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect();

    // Collapse multiple hyphens and trim
    let mut result = String::new();
    let mut prev_hyphen = true; // treat start as hyphen to trim leading
    for c in slug.chars() {
        if c == '-' {
            if !prev_hyphen {
                result.push('-');
            }
            prev_hyphen = true;
        } else {
            result.push(c);
            prev_hyphen = false;
        }
    }

    // Trim trailing hyphen
    if result.ends_with('-') {
        result.pop();
    }

    if result.is_empty() {
        // Fallback for purely non-alphanumeric input
        "unnamed".to_string()
    } else {
        result
    }
}

// ── SessionManager ───────────────────────────────────────────────

/// Manages reading and writing session JSONL files.
///
/// Named sessions use the directory layout:
/// `{sessions_dir}/<name>/session.jsonl`
/// `{sessions_dir}/<name>/session.json`  (metadata)
///
/// The manager can:
/// - Create new named sessions
/// - Switch between sessions
/// - Append turns to the current session
/// - Find and resume the most recently active session
/// - List all sessions with metadata
/// - Rename sessions
pub struct SessionManager {
    /// Directory where session directories are stored (typically ~/.zeroclaw/sessions/).
    sessions_dir: PathBuf,
    /// Current session name (set on new_session or resume).
    session_name: Option<String>,
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
            session_name: None,
            session_path: None,
        }
    }

    /// The directory where all session directories live.
    pub fn sessions_dir(&self) -> &Path {
        &self.sessions_dir
    }

    /// The current session name, if any.
    pub fn session_name(&self) -> Option<&str> {
        self.session_name.as_deref()
    }

    /// The current session ID (returns session name for compatibility).
    pub fn session_id(&self) -> Option<&str> {
        self.session_name.as_deref()
    }

    /// The current session file path, if any.
    pub fn session_path(&self) -> Option<&Path> {
        self.session_path.as_deref()
    }

    /// The directory for a given session name.
    pub fn session_dir_for(&self, name: &str) -> PathBuf {
        self.sessions_dir.join(name)
    }

    /// The JSONL file path for a given session name.
    fn session_jsonl_for(&self, name: &str) -> PathBuf {
        self.sessions_dir.join(name).join("session.jsonl")
    }

    /// The metadata file path for a given session name.
    fn session_meta_path_for(&self, name: &str) -> PathBuf {
        self.sessions_dir.join(name).join("session.json")
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

    /// Start a brand-new named session.
    ///
    /// Creates the session directory, empty JSONL file, and metadata.
    /// Returns an error if a session with that name already exists.
    pub async fn new_named_session(&mut self, name: &str) -> Result<String> {
        let slug = slugify(name);
        self.ensure_sessions_dir().await?;

        let dir = self.session_dir_for(&slug);
        if dir.exists() {
            bail!("Session '{}' already exists", slug);
        }

        fs::create_dir_all(&dir)
            .await
            .with_context(|| format!("Failed to create session dir: {}", dir.display()))?;

        let jsonl_path = self.session_jsonl_for(&slug);
        fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(&jsonl_path)
            .await
            .with_context(|| format!("Failed to create session file: {}", jsonl_path.display()))?;

        let now = Local::now().to_rfc3339();
        let meta = SessionMeta {
            name: slug.clone(),
            created_at: now.clone(),
            last_active: now,
            provider: None,
            model: None,
        };
        self.write_meta(&slug, &meta).await?;

        self.session_name = Some(slug.clone());
        self.session_path = Some(jsonl_path);

        Ok(slug)
    }

    /// Start a brand-new session with an auto-generated ID (backwards compat).
    ///
    /// Uses UUID-based naming: creates session under "default" name if
    /// no name is given, or generates a unique fallback.
    pub async fn new_session(&mut self) -> Result<String> {
        self.ensure_sessions_dir().await?;

        // If "default" doesn't exist, create it; otherwise generate a unique name
        let name = if !self.session_dir_for("default").exists() {
            "default".to_string()
        } else {
            let id = Uuid::new_v4().to_string()[..8].to_string();
            format!("session-{id}")
        };

        match self.new_named_session(&name).await {
            Ok(slug) => Ok(slug),
            Err(_) => {
                // Name collision, try with a uuid suffix
                let id = Uuid::new_v4().to_string()[..8].to_string();
                let fallback = format!("session-{id}");
                self.new_named_session(&fallback).await
            }
        }
    }

    /// Resume an existing session by name.
    ///
    /// Sets this session as the active one and returns all persisted turns.
    pub async fn resume_named_session(&mut self, name: &str) -> Result<Vec<SessionTurn>> {
        let slug = slugify(name);
        let dir = self.session_dir_for(&slug);
        if !dir.exists() {
            bail!("Session '{}' does not exist", slug);
        }

        let jsonl_path = self.session_jsonl_for(&slug);
        let turns = self.load_turns(&jsonl_path).await?;

        // Update last_active
        self.touch_meta(&slug).await?;

        self.session_name = Some(slug);
        self.session_path = Some(jsonl_path);

        Ok(turns)
    }

    /// Resume an existing session from a file path (backwards compat).
    ///
    /// Handles both new-style directory sessions and old-style flat JSONL files.
    pub async fn resume_session(&mut self, path: &Path) -> Result<Vec<SessionTurn>> {
        // Check if this is a new-style directory session
        if path.file_name().and_then(|f| f.to_str()) == Some("session.jsonl") {
            if let Some(parent) = path.parent() {
                if let Some(name) = parent.file_name().and_then(|f| f.to_str()) {
                    return self.resume_named_session(name).await;
                }
            }
        }

        // Old-style flat file: extract name from filename
        let filename = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        let id = extract_session_id(filename);
        let turns = self.load_turns(path).await?;

        self.session_name = Some(id.to_string());
        self.session_path = Some(path.to_path_buf());

        Ok(turns)
    }

    /// Switch to a different named session. Saves the current session
    /// (flushes pending JSONL) before loading the new one.
    ///
    /// Returns the loaded turns from the target session.
    pub async fn switch_session(&mut self, name: &str) -> Result<Vec<SessionTurn>> {
        let slug = slugify(name);

        // Touch current session's last_active before leaving
        if let Some(current) = &self.session_name {
            let _ = self.touch_meta(current).await;
        }

        self.resume_named_session(&slug).await
    }

    /// Rename the current session.
    pub async fn rename_session(&mut self, new_name: &str) -> Result<String> {
        let new_slug = slugify(new_name);

        let current_name = self
            .session_name
            .as_ref()
            .context("No active session to rename")?
            .clone();

        if current_name == new_slug {
            return Ok(new_slug);
        }

        let old_dir = self.session_dir_for(&current_name);
        let new_dir = self.session_dir_for(&new_slug);

        if new_dir.exists() {
            bail!("Session '{}' already exists", new_slug);
        }

        fs::rename(&old_dir, &new_dir)
            .await
            .with_context(|| format!("Failed to rename session directory"))?;

        // Update metadata
        if let Ok(mut meta) = self.read_meta(&new_slug).await {
            meta.name = new_slug.clone();
            let _ = self.write_meta(&new_slug, &meta).await;
        }

        let new_jsonl = self.session_jsonl_for(&new_slug);
        self.session_name = Some(new_slug.clone());
        self.session_path = Some(new_jsonl);

        Ok(new_slug)
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

        // Update last_active timestamp
        if let Some(name) = &self.session_name {
            let _ = self.touch_meta(name).await;
        }

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

    /// Find the most recently active session across all names.
    ///
    /// Returns `None` if there are no sessions.
    pub async fn find_most_recent(&self) -> Result<Option<PathBuf>> {
        let sessions = self.list_sessions(1).await?;
        Ok(sessions.into_iter().next().map(|info| {
            self.session_jsonl_for(&info.name)
        }))
    }

    /// Find the most recent session file from today, if one exists.
    ///
    /// Now checks named sessions and also falls back to legacy flat files.
    pub async fn find_todays_latest(&self) -> Result<Option<PathBuf>> {
        // First try: find the most recently active named session
        if let Some(path) = self.find_most_recent().await? {
            return Ok(Some(path));
        }

        // Fallback: check for legacy flat JSONL files
        let today = Local::now().date_naive().format("%Y-%m-%d").to_string();
        let legacy_files = self.list_legacy_session_files().await?;

        let mut todays: Vec<PathBuf> = legacy_files
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

        todays.sort_by_key(|p| {
            std::fs::metadata(p)
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        });

        Ok(todays.last().cloned())
    }

    /// List all sessions with metadata, most recently active first.
    ///
    /// Returns up to `limit` sessions.
    pub async fn list_sessions(&self, limit: usize) -> Result<Vec<SessionInfo>> {
        if !self.sessions_dir.exists() {
            return Ok(Vec::new());
        }

        let mut infos = Vec::new();

        let mut entries = fs::read_dir(&self.sessions_dir)
            .await
            .with_context(|| {
                format!(
                    "Failed to read sessions dir: {}",
                    self.sessions_dir.display()
                )
            })?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            let name = match path.file_name().and_then(|f| f.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };

            let jsonl_path = path.join("session.jsonl");
            if !jsonl_path.exists() {
                continue;
            }

            // Read metadata
            let (last_active, turn_count) = match self.read_meta(&name).await {
                Ok(meta) => {
                    let content = fs::read_to_string(&jsonl_path).await.unwrap_or_default();
                    let tc = content.lines().filter(|l| !l.trim().is_empty()).count();
                    (meta.last_active, tc)
                }
                Err(_) => {
                    // No metadata file — use file mtime
                    let mtime = std::fs::metadata(&jsonl_path)
                        .and_then(|m| m.modified())
                        .ok();
                    let content = fs::read_to_string(&jsonl_path).await.unwrap_or_default();
                    let tc = content.lines().filter(|l| !l.trim().is_empty()).count();
                    let last = mtime
                        .map(|t| {
                            let dt: chrono::DateTime<Local> = t.into();
                            dt.to_rfc3339()
                        })
                        .unwrap_or_else(|| Local::now().to_rfc3339());
                    (last, tc)
                }
            };

            infos.push(SessionInfo {
                name,
                last_active,
                turn_count,
                path,
            });
        }

        // Sort by last_active descending
        infos.sort_by(|a, b| b.last_active.cmp(&a.last_active));
        infos.truncate(limit);

        Ok(infos)
    }

    /// Check whether a session with the given name exists.
    pub fn session_exists(&self, name: &str) -> bool {
        let slug = slugify(name);
        self.session_dir_for(&slug).exists()
    }

    /// Check whether a session file is from today (legacy compat).
    pub fn is_from_today(path: &Path) -> bool {
        let today = Local::now().date_naive().format("%Y-%m-%d").to_string();
        path.file_name()
            .and_then(|f| f.to_str())
            .map(|f| f.starts_with(&today))
            .unwrap_or(false)
    }

    // ── Metadata helpers ─────────────────────────────────────────

    /// Write session metadata to disk.
    async fn write_meta(&self, name: &str, meta: &SessionMeta) -> Result<()> {
        let path = self.session_meta_path_for(name);
        let json = serde_json::to_string_pretty(meta)
            .context("Failed to serialize session metadata")?;
        fs::write(&path, json)
            .await
            .with_context(|| format!("Failed to write metadata: {}", path.display()))?;
        Ok(())
    }

    /// Read session metadata from disk.
    async fn read_meta(&self, name: &str) -> Result<SessionMeta> {
        let path = self.session_meta_path_for(name);
        let content = fs::read_to_string(&path)
            .await
            .with_context(|| format!("Failed to read metadata: {}", path.display()))?;
        let meta: SessionMeta = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse metadata: {}", path.display()))?;
        Ok(meta)
    }

    /// Update the last_active timestamp in metadata.
    async fn touch_meta(&self, name: &str) -> Result<()> {
        match self.read_meta(name).await {
            Ok(mut meta) => {
                meta.last_active = Local::now().to_rfc3339();
                self.write_meta(name, &meta).await
            }
            Err(_) => {
                // Create metadata if it doesn't exist
                let now = Local::now().to_rfc3339();
                let meta = SessionMeta {
                    name: name.to_string(),
                    created_at: now.clone(),
                    last_active: now,
                    provider: None,
                    model: None,
                };
                self.write_meta(name, &meta).await
            }
        }
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

    /// List legacy flat .jsonl files in the sessions directory.
    async fn list_legacy_session_files(&self) -> Result<Vec<PathBuf>> {
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
            if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                files.push(path);
            }
        }

        Ok(files)
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
        let name = mgr.new_session().await.unwrap();

        assert!(!name.is_empty());
        assert!(mgr.session_path().unwrap().exists());
        assert!(mgr
            .session_path()
            .unwrap()
            .to_str()
            .unwrap()
            .ends_with(".jsonl"));
    }

    #[tokio::test]
    async fn new_named_session_creates_directory() {
        let (_tmp, mut mgr) = setup().await;
        let name = mgr.new_named_session("my-project").await.unwrap();

        assert_eq!(name, "my-project");
        assert!(mgr.session_path().unwrap().exists());
        assert!(mgr.session_dir_for("my-project").exists());

        // Check metadata exists
        let meta_path = mgr.session_meta_path_for("my-project");
        assert!(meta_path.exists());
    }

    #[tokio::test]
    async fn duplicate_name_errors() {
        let (_tmp, mut mgr) = setup().await;
        mgr.new_named_session("test").await.unwrap();
        let result = mgr.new_named_session("test").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }

    #[tokio::test]
    async fn append_and_load_turns() {
        let (_tmp, mut mgr) = setup().await;
        mgr.new_named_session("test").await.unwrap();

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
    async fn find_most_recent_returns_latest() {
        let (_tmp, mut mgr) = setup().await;

        mgr.new_named_session("first").await.unwrap();
        let turn = SessionManager::user_turn("hello");
        mgr.append_turn(&turn).await.unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        mgr.new_named_session("second").await.unwrap();
        let turn = SessionManager::user_turn("world");
        mgr.append_turn(&turn).await.unwrap();

        let latest = mgr.find_most_recent().await.unwrap();
        assert!(latest.is_some());
        let latest_path = latest.unwrap();
        assert!(latest_path.to_str().unwrap().contains("second"));
    }

    #[tokio::test]
    async fn find_todays_latest_returns_most_recent() {
        let (_tmp, mut mgr) = setup().await;

        mgr.new_named_session("first").await.unwrap();
        let turn = SessionManager::user_turn("hello");
        mgr.append_turn(&turn).await.unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        mgr.new_named_session("second").await.unwrap();
        let turn = SessionManager::user_turn("world");
        mgr.append_turn(&turn).await.unwrap();

        let latest = mgr.find_todays_latest().await.unwrap();
        assert!(latest.is_some());
    }

    #[tokio::test]
    async fn list_sessions_returns_metadata() {
        let (_tmp, mut mgr) = setup().await;

        mgr.new_named_session("project-a").await.unwrap();
        let turn = SessionManager::user_turn("hello");
        mgr.append_turn(&turn).await.unwrap();
        let turn = SessionManager::assistant_turn(Some("world"), vec![]);
        mgr.append_turn(&turn).await.unwrap();

        let sessions = mgr.list_sessions(10).await.unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].name, "project-a");
        assert_eq!(sessions[0].turn_count, 2);
    }

    #[tokio::test]
    async fn resume_named_session_loads_turns() {
        let (_tmp, mut mgr) = setup().await;
        mgr.new_named_session("test-resume").await.unwrap();

        let user = SessionManager::user_turn("First message");
        mgr.append_turn(&user).await.unwrap();

        // Create a new manager and resume
        let mut mgr2 = SessionManager::new(_tmp.path());
        let turns = mgr2.resume_named_session("test-resume").await.unwrap();

        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].content.as_deref(), Some("First message"));
        assert_eq!(mgr2.session_name(), Some("test-resume"));
    }

    #[tokio::test]
    async fn switch_session_works() {
        let (_tmp, mut mgr) = setup().await;

        mgr.new_named_session("session-a").await.unwrap();
        let turn = SessionManager::user_turn("Message in A");
        mgr.append_turn(&turn).await.unwrap();

        mgr.new_named_session("session-b").await.unwrap();
        let turn = SessionManager::user_turn("Message in B");
        mgr.append_turn(&turn).await.unwrap();

        // Switch back to A
        let turns = mgr.switch_session("session-a").await.unwrap();
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].content.as_deref(), Some("Message in A"));
        assert_eq!(mgr.session_name(), Some("session-a"));
    }

    #[tokio::test]
    async fn switch_to_nonexistent_errors() {
        let (_tmp, mut mgr) = setup().await;
        mgr.new_named_session("exists").await.unwrap();
        let result = mgr.switch_session("nonexistent").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    #[tokio::test]
    async fn rename_session_works() {
        let (_tmp, mut mgr) = setup().await;
        mgr.new_named_session("old-name").await.unwrap();
        let turn = SessionManager::user_turn("hello");
        mgr.append_turn(&turn).await.unwrap();

        let new_name = mgr.rename_session("new-name").await.unwrap();
        assert_eq!(new_name, "new-name");
        assert_eq!(mgr.session_name(), Some("new-name"));

        // Old dir should be gone, new dir should exist
        assert!(!mgr.session_dir_for("old-name").exists());
        assert!(mgr.session_dir_for("new-name").exists());

        // Should still be able to load turns
        let turns = mgr.load_turns(mgr.session_path().unwrap()).await.unwrap();
        assert_eq!(turns.len(), 1);
    }

    #[tokio::test]
    async fn rename_to_existing_name_errors() {
        let (_tmp, mut mgr) = setup().await;
        mgr.new_named_session("first").await.unwrap();
        mgr.new_named_session("second").await.unwrap();
        let result = mgr.rename_session("first").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }

    #[tokio::test]
    async fn session_files_are_human_readable() {
        let (_tmp, mut mgr) = setup().await;
        mgr.new_named_session("readable").await.unwrap();

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
        mgr.new_named_session("empty").await.unwrap();

        let sessions = mgr.list_sessions(10).await.unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].turn_count, 0);
    }

    #[tokio::test]
    async fn slugify_basic() {
        assert_eq!(slugify("my-project"), "my-project");
        assert_eq!(slugify("My Project!"), "my-project");
        assert_eq!(slugify("  Hello World  "), "hello-world");
        assert_eq!(slugify("foo---bar"), "foo-bar");
        assert_eq!(slugify("CamelCase"), "camelcase");
        assert_eq!(slugify("with_underscores"), "with-underscores");
        assert_eq!(slugify("!!!"), "unnamed");
    }

    #[tokio::test]
    async fn slugify_preserves_numbers() {
        assert_eq!(slugify("project-42"), "project-42");
        assert_eq!(slugify("v2.0"), "v2-0");
    }

    #[tokio::test]
    async fn startup_resumes_most_recent_across_names() {
        let (_tmp, mut mgr) = setup().await;

        // Create several sessions with different last_active times
        mgr.new_named_session("old-project").await.unwrap();
        let turn = SessionManager::user_turn("old message");
        mgr.append_turn(&turn).await.unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        mgr.new_named_session("new-project").await.unwrap();
        let turn = SessionManager::user_turn("new message");
        mgr.append_turn(&turn).await.unwrap();

        // A fresh manager should find the most recent
        let fresh_mgr = SessionManager::new(_tmp.path());
        let latest = fresh_mgr.find_most_recent().await.unwrap();
        assert!(latest.is_some());
        assert!(latest.unwrap().to_str().unwrap().contains("new-project"));
    }

    #[tokio::test]
    async fn session_meta_is_persisted() {
        let (_tmp, mut mgr) = setup().await;
        mgr.new_named_session("meta-test").await.unwrap();

        let meta = mgr.read_meta("meta-test").await.unwrap();
        assert_eq!(meta.name, "meta-test");
        assert!(!meta.created_at.is_empty());
        assert!(!meta.last_active.is_empty());
    }

    #[tokio::test]
    async fn legacy_flat_file_resume_works() {
        let (tmp, mut mgr) = setup().await;
        // Create a legacy-style flat JSONL file
        let sessions_dir = tmp.path().join("sessions");
        fs::create_dir_all(&sessions_dir).await.unwrap();

        let legacy_path = sessions_dir.join("2026-02-15_abc12345.jsonl");
        let turn = SessionTurn {
            role: "user".to_string(),
            content: Some("Legacy message".to_string()),
            tool_calls: vec![],
            tool_results: vec![],
            timestamp: "2026-02-15T10:00:00+00:00".to_string(),
        };
        let line = serde_json::to_string(&turn).unwrap() + "\n";
        fs::write(&legacy_path, &line).await.unwrap();

        let turns = mgr.resume_session(&legacy_path).await.unwrap();
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].content.as_deref(), Some("Legacy message"));
        assert_eq!(mgr.session_id(), Some("abc12345"));
    }
}

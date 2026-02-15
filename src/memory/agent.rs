//! Multi-agent support for ZeroClaw.
//!
//! Each agent is a named persona with its own:
//! - `SOUL.md` (personality, instructions, tone)
//! - Workspace directory (isolated file operations)
//! - Tool allowlist (which tools it can use)
//! - Provider/model override (optional)
//! - Own memory files (USER.md, MEMORY.md, daily notes)
//! - Own session history (separate JSONL)
//!
//! Agents share the same running process. The user switches between
//! agents like switching sessions.
//!
//! ## Directory Layout
//!
//! ```text
//! ~/.zeroclaw/agents/
//! +-- default/
//! |   +-- SOUL.md
//! |   +-- USER.md
//! |   +-- MEMORY.md
//! |   +-- memory/
//! |   +-- sessions/
//! |   +-- workspace/
//! +-- coder/
//! |   +-- SOUL.md
//! |   +-- tools.toml
//! |   +-- agent.toml
//! |   +-- ...
//! +-- researcher/
//!     +-- SOUL.md
//!     +-- tools.toml
//!     +-- ...
//! ```

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::fs;

// ── Agent configuration types ────────────────────────────────────

/// Per-agent configuration loaded from `agent.toml`.
///
/// All fields are optional. If omitted, the agent uses global defaults.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Agent name (must match directory name).
    #[serde(default)]
    pub name: Option<String>,
    /// Human-readable description for `/agents` listing.
    #[serde(default)]
    pub description: Option<String>,
    /// Override provider (e.g. "anthropic", "openai").
    #[serde(default)]
    pub provider: Option<String>,
    /// Override model (e.g. "claude-sonnet-4-20250514").
    #[serde(default)]
    pub model: Option<String>,
}

/// Per-agent tool allowlist loaded from `tools.toml`.
///
/// If `tools.toml` is absent, the agent gets ALL tools.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolsConfig {
    /// List of allowed tool names. If empty or absent, all tools allowed.
    #[serde(default)]
    pub allowed: Vec<String>,
}

impl ToolsConfig {
    /// Check whether a tool is allowed for this agent.
    ///
    /// If the allowlist is empty, all tools are allowed.
    pub fn is_tool_allowed(&self, tool_name: &str) -> bool {
        self.allowed.is_empty() || self.allowed.contains(&tool_name.to_string())
    }
}

/// Information about an agent for listing purposes.
#[derive(Debug, Clone)]
pub struct AgentInfo {
    /// The agent name (directory name).
    pub name: String,
    /// Human-readable description from agent.toml.
    pub description: Option<String>,
    /// Whether this is the currently active agent.
    pub active: bool,
    /// Number of sessions for this agent.
    pub session_count: usize,
}

// ── Template SOUL.md ─────────────────────────────────────────────

/// Generate a template SOUL.md for a new agent.
pub fn template_soul_md(name: &str) -> String {
    format!(
        "# {name}\n\
         \n\
         You are {name}, an AI assistant.\n\
         \n\
         ## Personality\n\
         [Describe your personality and communication style]\n\
         \n\
         ## Capabilities\n\
         [What you're good at and how you approach tasks]\n\
         \n\
         ## Boundaries\n\
         [What you should avoid or decline]\n"
    )
}

// ── AgentManager ─────────────────────────────────────────────────

/// Manages agents: creation, switching, deletion, listing.
///
/// The `agents_dir` is typically `~/.zeroclaw/agents/`.
/// Each subdirectory is an agent with its own files.
pub struct AgentManager {
    /// Base directory for the zeroclaw installation (e.g. ~/.zeroclaw).
    base_dir: PathBuf,
    /// Directory containing all agent directories.
    agents_dir: PathBuf,
    /// Currently active agent name.
    active_agent: String,
}

impl AgentManager {
    /// Create a new AgentManager.
    ///
    /// `base_dir` is the zeroclaw root (e.g. `~/.zeroclaw`).
    /// The agents directory is `{base_dir}/agents/`.
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        let base_dir = base_dir.into();
        let agents_dir = base_dir.join("agents");
        Self {
            base_dir,
            agents_dir,
            active_agent: "default".to_string(),
        }
    }

    /// The base zeroclaw directory.
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// The agents root directory.
    pub fn agents_dir(&self) -> &Path {
        &self.agents_dir
    }

    /// The currently active agent name.
    pub fn active_agent(&self) -> &str {
        &self.active_agent
    }

    /// Directory for a specific agent.
    pub fn agent_dir(&self, name: &str) -> PathBuf {
        self.agents_dir.join(name)
    }

    /// The workspace directory for an agent (file operations scoped here).
    pub fn agent_workspace(&self, name: &str) -> PathBuf {
        self.agents_dir.join(name).join("workspace")
    }

    /// The sessions directory for an agent.
    pub fn agent_sessions_dir(&self, name: &str) -> PathBuf {
        self.agents_dir.join(name).join("sessions")
    }

    /// The memory directory for an agent (daily notes).
    pub fn agent_memory_dir(&self, name: &str) -> PathBuf {
        self.agents_dir.join(name).join("memory")
    }

    /// Path to agent.toml for an agent.
    pub fn agent_config_path(&self, name: &str) -> PathBuf {
        self.agents_dir.join(name).join("agent.toml")
    }

    /// Path to tools.toml for an agent.
    pub fn agent_tools_config_path(&self, name: &str) -> PathBuf {
        self.agents_dir.join(name).join("tools.toml")
    }

    /// Path to SOUL.md for an agent.
    pub fn agent_soul_path(&self, name: &str) -> PathBuf {
        self.agents_dir.join(name).join("SOUL.md")
    }

    /// Path to USER.md for an agent.
    pub fn agent_user_path(&self, name: &str) -> PathBuf {
        self.agents_dir.join(name).join("USER.md")
    }

    /// Path to MEMORY.md for an agent.
    pub fn agent_memory_path(&self, name: &str) -> PathBuf {
        self.agents_dir.join(name).join("MEMORY.md")
    }

    // ── Agent lifecycle ─────────────────────────────────────────

    /// Check if an agent exists.
    pub fn agent_exists(&self, name: &str) -> bool {
        self.agent_dir(name).exists()
    }

    /// Create a new agent with the given name.
    ///
    /// Creates the directory structure and a template SOUL.md.
    /// Returns an error if the agent already exists.
    pub async fn create_agent(&self, name: &str) -> Result<()> {
        let slug = slugify_agent_name(name);

        if slug.is_empty() {
            bail!("Agent name cannot be empty");
        }

        let dir = self.agent_dir(&slug);
        if dir.exists() {
            bail!("Agent '{}' already exists", slug);
        }

        // Create directory structure
        fs::create_dir_all(&dir)
            .await
            .with_context(|| format!("Failed to create agent dir: {}", dir.display()))?;

        fs::create_dir_all(self.agent_workspace(&slug))
            .await
            .with_context(|| "Failed to create agent workspace")?;

        fs::create_dir_all(self.agent_sessions_dir(&slug))
            .await
            .with_context(|| "Failed to create agent sessions dir")?;

        fs::create_dir_all(self.agent_memory_dir(&slug))
            .await
            .with_context(|| "Failed to create agent memory dir")?;

        // Write template SOUL.md
        let soul_content = template_soul_md(&slug);
        fs::write(self.agent_soul_path(&slug), &soul_content)
            .await
            .with_context(|| "Failed to write template SOUL.md")?;

        // Write default agent.toml
        let agent_config = AgentConfig {
            name: Some(slug.clone()),
            description: Some(format!("{slug} agent")),
            provider: None,
            model: None,
        };
        let toml_str = toml::to_string_pretty(&agent_config)
            .with_context(|| "Failed to serialize agent.toml")?;
        fs::write(self.agent_config_path(&slug), &toml_str)
            .await
            .with_context(|| "Failed to write agent.toml")?;

        Ok(())
    }

    /// Delete an agent.
    ///
    /// Cannot delete the "default" agent.
    /// Returns an error if the agent doesn't exist.
    pub async fn delete_agent(&self, name: &str) -> Result<()> {
        let slug = slugify_agent_name(name);

        if slug == "default" {
            bail!("Cannot delete the default agent");
        }

        if !self.agent_exists(&slug) {
            bail!("Agent '{}' does not exist", slug);
        }

        if self.active_agent == slug {
            bail!(
                "Cannot delete the currently active agent '{}'. Switch to another agent first.",
                slug
            );
        }

        let dir = self.agent_dir(&slug);
        fs::remove_dir_all(&dir)
            .await
            .with_context(|| format!("Failed to delete agent directory: {}", dir.display()))?;

        Ok(())
    }

    /// Switch to a different agent.
    ///
    /// Returns the agent's config and tools config.
    pub async fn switch_agent(
        &mut self,
        name: &str,
    ) -> Result<(AgentConfig, ToolsConfig)> {
        let slug = slugify_agent_name(name);

        if !self.agent_exists(&slug) {
            bail!("Agent '{}' does not exist", slug);
        }

        self.active_agent = slug.clone();

        let config = self.load_agent_config(&slug).await?;
        let tools_config = self.load_tools_config(&slug).await?;

        Ok((config, tools_config))
    }

    /// List all agents with metadata.
    pub async fn list_agents(&self) -> Result<Vec<AgentInfo>> {
        let mut agents = Vec::new();

        if !self.agents_dir.exists() {
            return Ok(agents);
        }

        let mut entries = fs::read_dir(&self.agents_dir)
            .await
            .with_context(|| {
                format!(
                    "Failed to read agents dir: {}",
                    self.agents_dir.display()
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

            let config = self.load_agent_config(&name).await.unwrap_or_default();
            let session_count = self.count_sessions(&name).await.unwrap_or(0);

            agents.push(AgentInfo {
                name: name.clone(),
                description: config.description,
                active: name == self.active_agent,
                session_count,
            });
        }

        // Sort alphabetically, but "default" first
        agents.sort_by(|a, b| {
            if a.name == "default" {
                std::cmp::Ordering::Less
            } else if b.name == "default" {
                std::cmp::Ordering::Greater
            } else {
                a.name.cmp(&b.name)
            }
        });

        Ok(agents)
    }

    /// Load agent.toml for a given agent.
    pub async fn load_agent_config(&self, name: &str) -> Result<AgentConfig> {
        let path = self.agent_config_path(name);
        match fs::read_to_string(&path).await {
            Ok(content) => {
                let config: AgentConfig = toml::from_str(&content)
                    .with_context(|| format!("Failed to parse agent.toml: {}", path.display()))?;
                Ok(config)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(AgentConfig::default()),
            Err(e) => Err(e).with_context(|| format!("Failed to read: {}", path.display())),
        }
    }

    /// Load tools.toml for a given agent.
    pub async fn load_tools_config(&self, name: &str) -> Result<ToolsConfig> {
        let path = self.agent_tools_config_path(name);
        match fs::read_to_string(&path).await {
            Ok(content) => {
                let config: ToolsConfig = toml::from_str(&content)
                    .with_context(|| format!("Failed to parse tools.toml: {}", path.display()))?;
                Ok(config)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(ToolsConfig::default()),
            Err(e) => Err(e).with_context(|| format!("Failed to read: {}", path.display())),
        }
    }

    /// Count sessions for a given agent.
    async fn count_sessions(&self, name: &str) -> Result<usize> {
        let dir = self.agent_sessions_dir(name);
        if !dir.exists() {
            return Ok(0);
        }

        let mut count = 0;
        let mut entries = fs::read_dir(&dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            if entry.path().is_dir() {
                count += 1;
            }
        }

        Ok(count)
    }

    // ── Migration ───────────────────────────────────────────────

    /// Migrate from the old flat layout to the agents/default/ structure.
    ///
    /// If `~/.zeroclaw/agents/` already exists, this is a no-op.
    /// Otherwise, moves existing files into `agents/default/`.
    pub async fn migrate_if_needed(&self) -> Result<bool> {
        // Already migrated if agents dir exists with a default agent
        if self.agent_dir("default").join("SOUL.md").exists()
            || self.agent_dir("default").join("sessions").exists()
        {
            return Ok(false);
        }

        // Check if there are old-style files to migrate
        let old_soul = self.base_dir.join("SOUL.md");
        let old_user = self.base_dir.join("USER.md");
        let old_memory = self.base_dir.join("MEMORY.md");
        let old_memory_dir = self.base_dir.join("memory");
        let old_sessions_dir = self.base_dir.join("sessions");
        let old_heartbeat = self.base_dir.join("HEARTBEAT.md");

        let has_old_files = old_soul.exists()
            || old_user.exists()
            || old_memory.exists()
            || old_memory_dir.exists()
            || old_sessions_dir.exists()
            || old_heartbeat.exists();

        if !has_old_files {
            // Nothing to migrate, just create the default agent
            self.ensure_default_agent().await?;
            return Ok(false);
        }

        // Create the default agent directory structure
        let default_dir = self.agent_dir("default");
        fs::create_dir_all(&default_dir).await?;
        fs::create_dir_all(self.agent_workspace("default")).await?;

        // Move files into default agent
        if old_soul.exists() {
            move_file(&old_soul, &self.agent_soul_path("default")).await?;
        }
        if old_user.exists() {
            move_file(&old_user, &self.agent_user_path("default")).await?;
        }
        if old_memory.exists() {
            move_file(&old_memory, &self.agent_memory_path("default")).await?;
        }
        if old_heartbeat.exists() {
            let dest = default_dir.join("HEARTBEAT.md");
            move_file(&old_heartbeat, &dest).await?;
        }

        // Move memory directory
        if old_memory_dir.exists() {
            let dest = self.agent_memory_dir("default");
            move_dir(&old_memory_dir, &dest).await?;
        }

        // Move sessions directory
        if old_sessions_dir.exists() {
            let dest = self.agent_sessions_dir("default");
            move_dir(&old_sessions_dir, &dest).await?;
        }

        Ok(true)
    }

    /// Ensure the default agent directory exists with minimal structure.
    pub async fn ensure_default_agent(&self) -> Result<()> {
        let default_dir = self.agent_dir("default");
        if default_dir.exists() {
            return Ok(());
        }

        fs::create_dir_all(&default_dir).await?;
        fs::create_dir_all(self.agent_workspace("default")).await?;
        fs::create_dir_all(self.agent_sessions_dir("default")).await?;
        fs::create_dir_all(self.agent_memory_dir("default")).await?;

        // Write a default SOUL.md if none exists
        let soul_path = self.agent_soul_path("default");
        if !soul_path.exists() {
            let content = "# Default Agent\n\nYou are ZeroClaw, a helpful AI assistant.\n";
            fs::write(&soul_path, content).await?;
        }

        Ok(())
    }
}

// ── Helpers ──────────────────────────────────────────────────────

/// Slugify an agent name: lowercase, replace non-alphanumeric with
/// hyphens, collapse multiple hyphens, trim.
pub fn slugify_agent_name(name: &str) -> String {
    let slug: String = name
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect();

    let mut result = String::new();
    let mut prev_hyphen = true;
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

    if result.ends_with('-') {
        result.pop();
    }

    result
}

/// Move a file from src to dst, creating parent dirs as needed.
async fn move_file(src: &Path, dst: &Path) -> Result<()> {
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent).await?;
    }
    // Try rename first (same filesystem), fall back to copy+remove
    match fs::rename(src, dst).await {
        Ok(()) => Ok(()),
        Err(_) => {
            fs::copy(src, dst).await.with_context(|| {
                format!("Failed to copy {} to {}", src.display(), dst.display())
            })?;
            fs::remove_file(src).await.with_context(|| {
                format!("Failed to remove original: {}", src.display())
            })?;
            Ok(())
        }
    }
}

/// Move a directory from src to dst.
async fn move_dir(src: &Path, dst: &Path) -> Result<()> {
    if dst.exists() {
        // Merge: copy contents instead of replacing
        copy_dir_recursive(src, dst).await?;
        fs::remove_dir_all(src).await?;
    } else {
        match fs::rename(src, dst).await {
            Ok(()) => {}
            Err(_) => {
                // Cross-filesystem: copy recursively then remove
                if let Some(parent) = dst.parent() {
                    fs::create_dir_all(parent).await?;
                }
                copy_dir_recursive(src, dst).await?;
                fs::remove_dir_all(src).await?;
            }
        }
    }
    Ok(())
}

/// Recursively copy a directory's contents.
fn copy_dir_recursive<'a>(
    src: &'a Path,
    dst: &'a Path,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
    Box::pin(async move {
        fs::create_dir_all(dst).await?;

        let mut entries = fs::read_dir(src).await?;
        while let Some(entry) = entries.next_entry().await? {
            let src_path = entry.path();
            let file_name = entry.file_name();
            let dst_path = dst.join(file_name);

            if src_path.is_dir() {
                copy_dir_recursive(&src_path, &dst_path).await?;
            } else {
                fs::copy(&src_path, &dst_path).await?;
            }
        }

        Ok(())
    })
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn setup() -> (TempDir, AgentManager) {
        let tmp = TempDir::new().unwrap();
        let mgr = AgentManager::new(tmp.path());
        (tmp, mgr)
    }

    // ── slugify_agent_name ──────────────────────────────────────

    #[test]
    fn slugify_basic() {
        assert_eq!(slugify_agent_name("coder"), "coder");
        assert_eq!(slugify_agent_name("My Coder"), "my-coder");
        assert_eq!(slugify_agent_name("  Research Bot!  "), "research-bot");
        assert_eq!(slugify_agent_name("foo---bar"), "foo-bar");
        assert_eq!(slugify_agent_name("CamelCase"), "camelcase");
    }

    #[test]
    fn slugify_preserves_numbers() {
        assert_eq!(slugify_agent_name("agent-42"), "agent-42");
        assert_eq!(slugify_agent_name("v2.0"), "v2-0");
    }

    #[test]
    fn slugify_empty_input() {
        assert_eq!(slugify_agent_name(""), "");
        assert_eq!(slugify_agent_name("!!!"), "");
    }

    // ── ToolsConfig ─────────────────────────────────────────────

    #[test]
    fn tools_config_empty_allows_all() {
        let config = ToolsConfig::default();
        assert!(config.is_tool_allowed("any_tool"));
        assert!(config.is_tool_allowed("shell"));
    }

    #[test]
    fn tools_config_allowlist_filters() {
        let config = ToolsConfig {
            allowed: vec![
                "shell".to_string(),
                "file_read".to_string(),
            ],
        };
        assert!(config.is_tool_allowed("shell"));
        assert!(config.is_tool_allowed("file_read"));
        assert!(!config.is_tool_allowed("web_search"));
        assert!(!config.is_tool_allowed("memory_store"));
    }

    // ── template_soul_md ────────────────────────────────────────

    #[test]
    fn template_soul_has_agent_name() {
        let soul = template_soul_md("coder");
        assert!(soul.contains("# coder"));
        assert!(soul.contains("You are coder"));
        assert!(soul.contains("## Personality"));
        assert!(soul.contains("## Capabilities"));
        assert!(soul.contains("## Boundaries"));
    }

    // ── AgentManager: create_agent ──────────────────────────────

    #[tokio::test]
    async fn create_agent_creates_directory_structure() {
        let (_tmp, mgr) = setup().await;

        mgr.create_agent("coder").await.unwrap();

        assert!(mgr.agent_dir("coder").exists());
        assert!(mgr.agent_workspace("coder").exists());
        assert!(mgr.agent_sessions_dir("coder").exists());
        assert!(mgr.agent_memory_dir("coder").exists());
        assert!(mgr.agent_soul_path("coder").exists());
        assert!(mgr.agent_config_path("coder").exists());
    }

    #[tokio::test]
    async fn create_agent_writes_template_soul() {
        let (_tmp, mgr) = setup().await;
        mgr.create_agent("researcher").await.unwrap();

        let content = fs::read_to_string(mgr.agent_soul_path("researcher"))
            .await
            .unwrap();
        assert!(content.contains("# researcher"));
        assert!(content.contains("You are researcher"));
    }

    #[tokio::test]
    async fn create_agent_writes_agent_toml() {
        let (_tmp, mgr) = setup().await;
        mgr.create_agent("coder").await.unwrap();

        let content = fs::read_to_string(mgr.agent_config_path("coder"))
            .await
            .unwrap();
        let config: AgentConfig = toml::from_str(&content).unwrap();
        assert_eq!(config.name.as_deref(), Some("coder"));
        assert!(config.description.is_some());
    }

    #[tokio::test]
    async fn create_agent_slugifies_name() {
        let (_tmp, mgr) = setup().await;
        mgr.create_agent("My Agent!").await.unwrap();

        assert!(mgr.agent_exists("my-agent"));
        assert!(!mgr.agent_exists("My Agent!"));
    }

    #[tokio::test]
    async fn create_duplicate_agent_errors() {
        let (_tmp, mgr) = setup().await;
        mgr.create_agent("coder").await.unwrap();

        let result = mgr.create_agent("coder").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }

    #[tokio::test]
    async fn create_agent_empty_name_errors() {
        let (_tmp, mgr) = setup().await;
        let result = mgr.create_agent("").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be empty"));
    }

    // ── AgentManager: delete_agent ──────────────────────────────

    #[tokio::test]
    async fn delete_agent_removes_directory() {
        let (_tmp, mgr) = setup().await;
        mgr.create_agent("coder").await.unwrap();
        assert!(mgr.agent_exists("coder"));

        mgr.delete_agent("coder").await.unwrap();
        assert!(!mgr.agent_exists("coder"));
    }

    #[tokio::test]
    async fn cannot_delete_default_agent() {
        let (_tmp, mgr) = setup().await;
        mgr.ensure_default_agent().await.unwrap();

        let result = mgr.delete_agent("default").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Cannot delete the default"));
    }

    #[tokio::test]
    async fn delete_nonexistent_agent_errors() {
        let (_tmp, mgr) = setup().await;
        let result = mgr.delete_agent("nonexistent").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    #[tokio::test]
    async fn cannot_delete_active_agent() {
        let (_tmp, mut mgr) = setup().await;
        mgr.create_agent("coder").await.unwrap();
        mgr.switch_agent("coder").await.unwrap();

        let result = mgr.delete_agent("coder").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("currently active"));
    }

    // ── AgentManager: switch_agent ──────────────────────────────

    #[tokio::test]
    async fn switch_agent_changes_active() {
        let (_tmp, mut mgr) = setup().await;
        mgr.ensure_default_agent().await.unwrap();
        mgr.create_agent("coder").await.unwrap();

        assert_eq!(mgr.active_agent(), "default");

        let (config, _tools) = mgr.switch_agent("coder").await.unwrap();
        assert_eq!(mgr.active_agent(), "coder");
        assert_eq!(config.name.as_deref(), Some("coder"));
    }

    #[tokio::test]
    async fn switch_to_nonexistent_errors() {
        let (_tmp, mut mgr) = setup().await;
        let result = mgr.switch_agent("nonexistent").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    #[tokio::test]
    async fn switch_preserves_previous_agent() {
        let (_tmp, mut mgr) = setup().await;
        mgr.ensure_default_agent().await.unwrap();
        mgr.create_agent("coder").await.unwrap();
        mgr.create_agent("researcher").await.unwrap();

        mgr.switch_agent("coder").await.unwrap();
        assert_eq!(mgr.active_agent(), "coder");

        mgr.switch_agent("researcher").await.unwrap();
        assert_eq!(mgr.active_agent(), "researcher");

        // Previous agent directories still exist
        assert!(mgr.agent_exists("coder"));
        assert!(mgr.agent_exists("default"));
    }

    // ── AgentManager: list_agents ───────────────────────────────

    #[tokio::test]
    async fn list_agents_returns_all() {
        let (_tmp, mgr) = setup().await;
        mgr.ensure_default_agent().await.unwrap();
        mgr.create_agent("coder").await.unwrap();
        mgr.create_agent("researcher").await.unwrap();

        let agents = mgr.list_agents().await.unwrap();
        assert_eq!(agents.len(), 3);

        let names: Vec<&str> = agents.iter().map(|a| a.name.as_str()).collect();
        assert!(names.contains(&"default"));
        assert!(names.contains(&"coder"));
        assert!(names.contains(&"researcher"));

        // Default should be first
        assert_eq!(agents[0].name, "default");
    }

    #[tokio::test]
    async fn list_agents_marks_active() {
        let (_tmp, mut mgr) = setup().await;
        mgr.ensure_default_agent().await.unwrap();
        mgr.create_agent("coder").await.unwrap();
        mgr.switch_agent("coder").await.unwrap();

        let agents = mgr.list_agents().await.unwrap();

        let default = agents.iter().find(|a| a.name == "default").unwrap();
        let coder = agents.iter().find(|a| a.name == "coder").unwrap();

        assert!(!default.active);
        assert!(coder.active);
    }

    #[tokio::test]
    async fn list_agents_empty_when_no_agents_dir() {
        let (_tmp, mgr) = setup().await;
        let agents = mgr.list_agents().await.unwrap();
        assert!(agents.is_empty());
    }

    // ── AgentManager: config loading ────────────────────────────

    #[tokio::test]
    async fn load_agent_config_reads_toml() {
        let (_tmp, mgr) = setup().await;
        mgr.create_agent("coder").await.unwrap();

        // Write a custom agent.toml
        let custom_config = r#"
name = "coder"
description = "Senior software engineer with full shell access"
provider = "anthropic"
model = "claude-sonnet-4-20250514"
"#;
        fs::write(mgr.agent_config_path("coder"), custom_config)
            .await
            .unwrap();

        let config = mgr.load_agent_config("coder").await.unwrap();
        assert_eq!(config.name.as_deref(), Some("coder"));
        assert_eq!(
            config.description.as_deref(),
            Some("Senior software engineer with full shell access")
        );
        assert_eq!(config.provider.as_deref(), Some("anthropic"));
        assert_eq!(config.model.as_deref(), Some("claude-sonnet-4-20250514"));
    }

    #[tokio::test]
    async fn load_agent_config_returns_default_when_missing() {
        let (_tmp, mgr) = setup().await;
        mgr.ensure_default_agent().await.unwrap();

        // Remove agent.toml
        let path = mgr.agent_config_path("default");
        if path.exists() {
            fs::remove_file(&path).await.unwrap();
        }

        let config = mgr.load_agent_config("default").await.unwrap();
        assert!(config.name.is_none());
        assert!(config.provider.is_none());
    }

    #[tokio::test]
    async fn load_tools_config_reads_toml() {
        let (_tmp, mgr) = setup().await;
        mgr.create_agent("coder").await.unwrap();

        let tools_toml = r#"allowed = ["shell", "file_read", "file_write", "str_replace"]"#;
        fs::write(mgr.agent_tools_config_path("coder"), tools_toml)
            .await
            .unwrap();

        let tools = mgr.load_tools_config("coder").await.unwrap();
        assert_eq!(tools.allowed.len(), 4);
        assert!(tools.is_tool_allowed("shell"));
        assert!(tools.is_tool_allowed("file_read"));
        assert!(!tools.is_tool_allowed("web_search"));
    }

    #[tokio::test]
    async fn load_tools_config_returns_default_when_missing() {
        let (_tmp, mgr) = setup().await;
        mgr.create_agent("coder").await.unwrap();

        let tools = mgr.load_tools_config("coder").await.unwrap();
        assert!(tools.allowed.is_empty());
        assert!(tools.is_tool_allowed("anything"));
    }

    // ── AgentManager: ensure_default_agent ──────────────────────

    #[tokio::test]
    async fn ensure_default_creates_structure() {
        let (_tmp, mgr) = setup().await;
        mgr.ensure_default_agent().await.unwrap();

        assert!(mgr.agent_dir("default").exists());
        assert!(mgr.agent_workspace("default").exists());
        assert!(mgr.agent_sessions_dir("default").exists());
        assert!(mgr.agent_memory_dir("default").exists());
        assert!(mgr.agent_soul_path("default").exists());
    }

    #[tokio::test]
    async fn ensure_default_is_idempotent() {
        let (_tmp, mgr) = setup().await;
        mgr.ensure_default_agent().await.unwrap();

        // Write something to SOUL.md
        fs::write(mgr.agent_soul_path("default"), "custom soul\n")
            .await
            .unwrap();

        // Call again — should not overwrite
        mgr.ensure_default_agent().await.unwrap();

        let content = fs::read_to_string(mgr.agent_soul_path("default"))
            .await
            .unwrap();
        assert_eq!(content, "custom soul\n");
    }

    // ── Migration ───────────────────────────────────────────────

    #[tokio::test]
    async fn migration_moves_old_files_to_default_agent() {
        let (tmp, mgr) = setup().await;

        // Create old-style flat files
        fs::write(tmp.path().join("SOUL.md"), "old soul\n")
            .await
            .unwrap();
        fs::write(tmp.path().join("USER.md"), "old user\n")
            .await
            .unwrap();
        fs::write(tmp.path().join("MEMORY.md"), "old memory\n")
            .await
            .unwrap();
        fs::write(tmp.path().join("HEARTBEAT.md"), "- task 1\n")
            .await
            .unwrap();

        // Create old memory dir
        let old_memory = tmp.path().join("memory");
        fs::create_dir_all(&old_memory).await.unwrap();
        fs::write(old_memory.join("2026-02-14.md"), "daily note\n")
            .await
            .unwrap();

        // Create old sessions dir
        let old_sessions = tmp.path().join("sessions");
        fs::create_dir_all(old_sessions.join("my-session"))
            .await
            .unwrap();
        fs::write(
            old_sessions.join("my-session").join("session.jsonl"),
            "{}\n",
        )
        .await
        .unwrap();

        // Run migration
        let migrated = mgr.migrate_if_needed().await.unwrap();
        assert!(migrated);

        // Old files should be moved
        assert!(!tmp.path().join("SOUL.md").exists());
        assert!(!tmp.path().join("USER.md").exists());
        assert!(!tmp.path().join("MEMORY.md").exists());
        assert!(!tmp.path().join("HEARTBEAT.md").exists());
        assert!(!tmp.path().join("memory").exists());
        assert!(!tmp.path().join("sessions").exists());

        // New locations should have the content
        let soul = fs::read_to_string(mgr.agent_soul_path("default"))
            .await
            .unwrap();
        assert_eq!(soul, "old soul\n");

        let user = fs::read_to_string(mgr.agent_user_path("default"))
            .await
            .unwrap();
        assert_eq!(user, "old user\n");

        let mem = fs::read_to_string(mgr.agent_memory_path("default"))
            .await
            .unwrap();
        assert_eq!(mem, "old memory\n");

        let hb = fs::read_to_string(mgr.agent_dir("default").join("HEARTBEAT.md"))
            .await
            .unwrap();
        assert_eq!(hb, "- task 1\n");

        let daily = fs::read_to_string(
            mgr.agent_memory_dir("default").join("2026-02-14.md"),
        )
        .await
        .unwrap();
        assert_eq!(daily, "daily note\n");

        let session = fs::read_to_string(
            mgr.agent_sessions_dir("default")
                .join("my-session")
                .join("session.jsonl"),
        )
        .await
        .unwrap();
        assert_eq!(session, "{}\n");
    }

    #[tokio::test]
    async fn migration_is_noop_when_already_migrated() {
        let (_tmp, mgr) = setup().await;
        mgr.ensure_default_agent().await.unwrap();

        // Already has agents/default, so no migration needed
        let migrated = mgr.migrate_if_needed().await.unwrap();
        assert!(!migrated);
    }

    #[tokio::test]
    async fn migration_creates_default_when_no_old_files() {
        let (_tmp, mgr) = setup().await;

        let migrated = mgr.migrate_if_needed().await.unwrap();
        assert!(!migrated);

        // Default agent should still be created
        assert!(mgr.agent_dir("default").exists());
    }

    // ── Session isolation ───────────────────────────────────────

    #[tokio::test]
    async fn each_agent_has_independent_sessions_dir() {
        let (_tmp, mgr) = setup().await;
        mgr.create_agent("coder").await.unwrap();
        mgr.create_agent("researcher").await.unwrap();

        let coder_sessions = mgr.agent_sessions_dir("coder");
        let researcher_sessions = mgr.agent_sessions_dir("researcher");

        assert_ne!(coder_sessions, researcher_sessions);
        assert!(coder_sessions.exists());
        assert!(researcher_sessions.exists());
    }

    // ── Workspace scoping ───────────────────────────────────────

    #[tokio::test]
    async fn workspace_is_scoped_to_agent() {
        let (_tmp, mgr) = setup().await;
        mgr.create_agent("coder").await.unwrap();
        mgr.create_agent("researcher").await.unwrap();

        let coder_ws = mgr.agent_workspace("coder");
        let researcher_ws = mgr.agent_workspace("researcher");

        assert_ne!(coder_ws, researcher_ws);
        assert!(coder_ws.to_str().unwrap().contains("coder"));
        assert!(researcher_ws.to_str().unwrap().contains("researcher"));
    }

    // ── Provider/model override ─────────────────────────────────

    #[tokio::test]
    async fn provider_model_override_from_agent_toml() {
        let (_tmp, mgr) = setup().await;
        mgr.create_agent("cheap").await.unwrap();

        let config_toml = r#"
name = "cheap"
description = "Uses a cheaper model"
provider = "openai"
model = "gpt-4o-mini"
"#;
        fs::write(mgr.agent_config_path("cheap"), config_toml)
            .await
            .unwrap();

        let config = mgr.load_agent_config("cheap").await.unwrap();
        assert_eq!(config.provider.as_deref(), Some("openai"));
        assert_eq!(config.model.as_deref(), Some("gpt-4o-mini"));
    }
}

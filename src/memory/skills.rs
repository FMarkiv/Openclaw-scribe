//! Runtime skills system for ZeroClaw.
//!
//! A "skill" is a directory containing:
//! - `skill.toml`  — metadata, parameters, and execution config
//! - `SKILL.md`    — instructions injected into the system prompt
//! - `run.sh`      — (optional) shell script implementing the skill
//! - `run.py`      — (optional) Python script alternative
//!
//! Skills are loaded from `~/.zeroclaw/skills/` at startup and can
//! be reloaded at runtime via `/skills-reload`.
//!
//! Each skill becomes an LLM tool: the tool name, description, and
//! parameter schema are derived from `skill.toml`. When the LLM calls
//! the tool, the agent validates parameters, sets environment variables,
//! pipes JSON params via stdin, and captures stdout as the result.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

// ── skill.toml types ─────────────────────────────────────────────

/// Top-level skill configuration from `skill.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillConfig {
    /// Skill name (used as the tool name).
    pub name: String,
    /// Human-readable description shown to the LLM.
    pub description: String,
    /// Skill version.
    #[serde(default = "default_version")]
    pub version: String,
    /// Whether the skill is enabled.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Execution configuration.
    #[serde(default)]
    pub execution: ExecutionConfig,
    /// Parameter definitions.
    #[serde(default)]
    pub parameters: Vec<ParameterDef>,
}

/// Execution section of skill.toml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    /// Runner type: "shell" or "python".
    #[serde(default = "default_runner")]
    pub runner: String,
    /// Script file name relative to skill directory.
    #[serde(default = "default_script")]
    pub script: String,
    /// Timeout in seconds for script execution.
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,
    /// Working directory: "workspace" (agent's) or "skill" (skill dir).
    #[serde(default = "default_working_dir")]
    pub working_dir: String,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            runner: "shell".to_string(),
            script: "run.sh".to_string(),
            timeout_seconds: 30,
            working_dir: "workspace".to_string(),
        }
    }
}

/// A parameter definition from [[parameters]] in skill.toml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDef {
    /// Parameter name.
    pub name: String,
    /// Parameter type (string, number, boolean, etc.).
    #[serde(rename = "type", default = "default_param_type")]
    pub param_type: String,
    /// Human-readable description of the parameter.
    #[serde(default)]
    pub description: String,
    /// Whether the parameter is required.
    #[serde(default)]
    pub required: bool,
}

fn default_version() -> String {
    "1.0".to_string()
}
fn default_true() -> bool {
    true
}
fn default_runner() -> String {
    "shell".to_string()
}
fn default_script() -> String {
    "run.sh".to_string()
}
fn default_timeout() -> u64 {
    30
}
fn default_working_dir() -> String {
    "workspace".to_string()
}
fn default_param_type() -> String {
    "string".to_string()
}

// ── Loaded skill ─────────────────────────────────────────────────

/// A fully loaded skill ready for execution.
#[derive(Debug, Clone)]
pub struct LoadedSkill {
    /// Parsed skill.toml configuration.
    pub config: SkillConfig,
    /// Absolute path to the skill directory.
    pub dir: PathBuf,
    /// Content of SKILL.md (empty string if absent).
    pub skill_md: String,
    /// Whether the skill is currently enabled.
    pub enabled: bool,
}

impl LoadedSkill {
    /// Absolute path to the script file.
    pub fn script_path(&self) -> PathBuf {
        self.dir.join(&self.config.execution.script)
    }

    /// Build a JSON Schema from the parameter definitions.
    pub fn parameters_schema(&self) -> Value {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for param in &self.config.parameters {
            let json_type = match param.param_type.as_str() {
                "number" | "integer" => "number",
                "boolean" | "bool" => "boolean",
                _ => "string",
            };
            properties.insert(
                param.name.clone(),
                json!({
                    "type": json_type,
                    "description": param.description,
                }),
            );
            if param.required {
                required.push(Value::String(param.name.clone()));
            }
        }

        json!({
            "type": "object",
            "properties": properties,
            "required": required,
        })
    }

    /// Validate arguments against parameter definitions.
    ///
    /// Returns Ok(()) if all required parameters are present,
    /// or an error listing the missing ones.
    pub fn validate_params(&self, args: &Value) -> Result<()> {
        let mut missing = Vec::new();
        for param in &self.config.parameters {
            if param.required {
                let val = &args[&param.name];
                if val.is_null() {
                    missing.push(param.name.clone());
                }
            }
        }
        if !missing.is_empty() {
            bail!("Missing required parameters: {}", missing.join(", "));
        }
        Ok(())
    }
}

// ── Skill execution ──────────────────────────────────────────────

/// Result of executing a skill script.
#[derive(Debug, Clone)]
pub struct SkillExecutionResult {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
}

/// Execute a loaded skill with the given arguments.
///
/// 1. Validates required parameters.
/// 2. Sets SKILL_PARAM_{NAME} env vars for each parameter.
/// 3. Passes args as JSON via stdin.
/// 4. Runs the script with the configured runner.
/// 5. Enforces the timeout.
/// 6. Captures stdout (tool result) and stderr (warnings).
pub async fn execute_skill(
    skill: &LoadedSkill,
    args: &Value,
    agent_workspace: Option<&Path>,
) -> Result<SkillExecutionResult> {
    skill.validate_params(args)?;

    let script_path = skill.script_path();
    if !script_path.exists() {
        bail!(
            "Script not found: {} (skill: {})",
            script_path.display(),
            skill.config.name,
        );
    }

    // Determine working directory
    let work_dir = match skill.config.execution.working_dir.as_str() {
        "skill" => skill.dir.clone(),
        _ => agent_workspace
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| skill.dir.clone()),
    };

    // Build the command
    let mut cmd = match skill.config.execution.runner.as_str() {
        "python" => {
            let mut c = Command::new("python3");
            c.arg(&script_path);
            c
        }
        _ => {
            // shell runner
            let mut c = Command::new("bash");
            c.arg(&script_path);
            c
        }
    };

    cmd.current_dir(&work_dir);
    cmd.stdin(std::process::Stdio::piped());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    // Set environment variables for each parameter
    if let Some(obj) = args.as_object() {
        for (key, value) in obj {
            let env_name = format!("SKILL_PARAM_{}", key.to_uppercase());
            let env_value = match value {
                Value::String(s) => s.clone(),
                Value::Null => String::new(),
                other => other.to_string(),
            };
            cmd.env(&env_name, &env_value);
        }
    }

    // Spawn the process
    let mut child = cmd.spawn().with_context(|| {
        format!(
            "Failed to spawn script for skill '{}'",
            skill.config.name,
        )
    })?;

    // Write JSON args to stdin
    if let Some(mut stdin) = child.stdin.take() {
        let json_bytes = serde_json::to_vec(args).unwrap_or_default();
        // Ignore write errors (script might not read stdin)
        let _ = stdin.write_all(&json_bytes).await;
        drop(stdin);
    }

    // Grab the child's pid so we can kill on timeout
    let child_id = child.id();

    // Wait with timeout
    let timeout = Duration::from_secs(skill.config.execution.timeout_seconds);
    let result = tokio::time::timeout(timeout, child.wait_with_output()).await;

    match result {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let success = output.status.success();

            Ok(SkillExecutionResult {
                success,
                stdout,
                stderr,
            })
        }
        Ok(Err(e)) => {
            bail!(
                "Script execution error for skill '{}': {}",
                skill.config.name,
                e,
            );
        }
        Err(_) => {
            // Timeout — try to kill the process by pid
            if let Some(pid) = child_id {
                #[cfg(unix)]
                unsafe {
                    libc::kill(pid as i32, libc::SIGKILL);
                }
            }
            Ok(SkillExecutionResult {
                success: false,
                stdout: String::new(),
                stderr: format!(
                    "Skill '{}' timed out after {} seconds",
                    skill.config.name,
                    skill.config.execution.timeout_seconds,
                ),
            })
        }
    }
}

// ── SkillManager ─────────────────────────────────────────────────

/// Manages the lifecycle of runtime skills.
///
/// Loads skills from a directory, supports enable/disable, reload,
/// and provides access to loaded skills for tool registration and
/// system prompt injection.
pub struct SkillManager {
    /// Directory containing skill subdirectories.
    skills_dir: PathBuf,
    /// Currently loaded skills, keyed by skill name.
    skills: HashMap<String, LoadedSkill>,
    /// Warnings from the last load/reload.
    warnings: Vec<String>,
}

impl SkillManager {
    /// Create a new SkillManager.
    ///
    /// `skills_dir` is typically `~/.zeroclaw/skills/`.
    pub fn new(skills_dir: impl Into<PathBuf>) -> Self {
        Self {
            skills_dir: skills_dir.into(),
            skills: HashMap::new(),
            warnings: Vec::new(),
        }
    }

    /// The skills root directory.
    pub fn skills_dir(&self) -> &Path {
        &self.skills_dir
    }

    /// Load all skills from the skills directory.
    ///
    /// Scans each subdirectory for `skill.toml`, parses it, and
    /// loads the optional `SKILL.md`. Skills with missing or invalid
    /// `skill.toml` are skipped with a warning.
    pub async fn load_skills(&mut self) -> Result<()> {
        self.skills.clear();
        self.warnings.clear();

        if !self.skills_dir.exists() {
            return Ok(());
        }

        let mut entries = fs::read_dir(&self.skills_dir)
            .await
            .with_context(|| {
                format!(
                    "Failed to read skills directory: {}",
                    self.skills_dir.display(),
                )
            })?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            match self.load_single_skill(&path).await {
                Ok(skill) => {
                    self.skills.insert(skill.config.name.clone(), skill);
                }
                Err(e) => {
                    let dir_name = path
                        .file_name()
                        .and_then(|f| f.to_str())
                        .unwrap_or("?");
                    self.warnings.push(format!(
                        "Skipped skill '{}': {}",
                        dir_name, e,
                    ));
                }
            }
        }

        Ok(())
    }

    /// Load a single skill from a directory.
    async fn load_single_skill(&self, dir: &Path) -> Result<LoadedSkill> {
        let toml_path = dir.join("skill.toml");
        if !toml_path.exists() {
            bail!("skill.toml not found");
        }

        let toml_content = fs::read_to_string(&toml_path)
            .await
            .with_context(|| format!("Failed to read {}", toml_path.display()))?;

        let config: SkillConfig = toml::from_str(&toml_content)
            .with_context(|| format!("Failed to parse {}", toml_path.display()))?;

        if config.name.is_empty() {
            bail!("Skill name cannot be empty");
        }

        // Load optional SKILL.md
        let skill_md_path = dir.join("SKILL.md");
        let skill_md = match fs::read_to_string(&skill_md_path).await {
            Ok(content) => content,
            Err(_) => String::new(),
        };

        let enabled = config.enabled;

        Ok(LoadedSkill {
            config,
            dir: dir.to_path_buf(),
            skill_md,
            enabled,
        })
    }

    /// Reload all skills from disk.
    ///
    /// This is the same as `load_skills` but preserves the
    /// enabled/disabled state of skills that were previously loaded.
    pub async fn reload_skills(&mut self) -> Result<()> {
        // Remember enabled states
        let prev_states: HashMap<String, bool> = self
            .skills
            .iter()
            .map(|(name, s)| (name.clone(), s.enabled))
            .collect();

        self.load_skills().await?;

        // Restore enabled states for skills that still exist
        for (name, enabled) in &prev_states {
            if let Some(skill) = self.skills.get_mut(name) {
                skill.enabled = *enabled;
            }
        }

        Ok(())
    }

    /// Get all loaded skills (including disabled).
    pub fn all_skills(&self) -> Vec<&LoadedSkill> {
        let mut skills: Vec<&LoadedSkill> = self.skills.values().collect();
        skills.sort_by(|a, b| a.config.name.cmp(&b.config.name));
        skills
    }

    /// Get only enabled skills.
    pub fn enabled_skills(&self) -> Vec<&LoadedSkill> {
        let mut skills: Vec<&LoadedSkill> = self
            .skills
            .values()
            .filter(|s| s.enabled)
            .collect();
        skills.sort_by(|a, b| a.config.name.cmp(&b.config.name));
        skills
    }

    /// Get a skill by name.
    pub fn get_skill(&self, name: &str) -> Option<&LoadedSkill> {
        self.skills.get(name)
    }

    /// Enable a skill by name. Returns true if found.
    pub fn enable_skill(&mut self, name: &str) -> bool {
        if let Some(skill) = self.skills.get_mut(name) {
            skill.enabled = true;
            true
        } else {
            false
        }
    }

    /// Disable a skill by name. Returns true if found.
    pub fn disable_skill(&mut self, name: &str) -> bool {
        if let Some(skill) = self.skills.get_mut(name) {
            skill.enabled = false;
            true
        } else {
            false
        }
    }

    /// Warnings from the last load/reload operation.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Build the "## Available Skills" system prompt section.
    ///
    /// Returns the content to append to the system prompt,
    /// including SKILL.md content for each enabled skill.
    pub fn build_system_prompt_section(&self) -> String {
        let enabled = self.enabled_skills();
        if enabled.is_empty() {
            return String::new();
        }

        let mut section = String::from("\n## Available Skills\n\n");

        for skill in &enabled {
            section.push_str(&format!("### {}\n", skill.config.name));
            section.push_str(&format!("{}\n", skill.config.description));

            if !skill.config.parameters.is_empty() {
                section.push_str("Parameters:\n");
                for param in &skill.config.parameters {
                    let req = if param.required {
                        " (required)"
                    } else {
                        " (optional)"
                    };
                    section.push_str(&format!(
                        "- `{}` ({}){}: {}\n",
                        param.name, param.param_type, req, param.description,
                    ));
                }
            }

            if !skill.skill_md.is_empty() {
                section.push('\n');
                section.push_str(&skill.skill_md);
                if !skill.skill_md.ends_with('\n') {
                    section.push('\n');
                }
            }
            section.push('\n');
        }

        section
    }

    /// Check if a skill is allowed by an agent's tools config.
    ///
    /// Skills are filtered the same way as built-in tools: if the
    /// allowlist is empty all skills are available; otherwise only
    /// skills named in the list are.
    pub fn filter_by_tools_config(
        &self,
        tools_config: &crate::memory::agent::ToolsConfig,
    ) -> Vec<&LoadedSkill> {
        self.enabled_skills()
            .into_iter()
            .filter(|s| tools_config.is_tool_allowed(&s.config.name))
            .collect()
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Helper: create a minimal skill directory with skill.toml.
    async fn create_skill_dir(
        base: &Path,
        name: &str,
        toml_content: &str,
        skill_md: Option<&str>,
        script: Option<&str>,
    ) {
        let dir = base.join(name);
        fs::create_dir_all(&dir).await.unwrap();
        fs::write(dir.join("skill.toml"), toml_content)
            .await
            .unwrap();
        if let Some(md) = skill_md {
            fs::write(dir.join("SKILL.md"), md).await.unwrap();
        }
        if let Some(sc) = script {
            fs::write(dir.join("run.sh"), sc).await.unwrap();
            // Make executable on Unix
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let perms = std::fs::Permissions::from_mode(0o755);
                std::fs::set_permissions(dir.join("run.sh"), perms).unwrap();
            }
        }
    }

    fn valid_toml(name: &str) -> String {
        format!(
            r#"
name = "{name}"
description = "Test skill {name}"
version = "1.0"
enabled = true

[execution]
runner = "shell"
script = "run.sh"
timeout_seconds = 5
working_dir = "skill"

[[parameters]]
name = "input"
type = "string"
description = "Input text"
required = true

[[parameters]]
name = "verbose"
type = "boolean"
description = "Enable verbose output"
required = false
"#
        )
    }

    // ── SkillConfig parsing ─────────────────────────────────────

    #[test]
    fn parse_valid_skill_toml() {
        let toml_str = r#"
name = "git-commit"
description = "Stage and commit changes"
version = "1.0"
enabled = true

[execution]
runner = "shell"
script = "run.sh"
timeout_seconds = 30
working_dir = "workspace"

[[parameters]]
name = "message"
type = "string"
description = "Commit message"
required = false

[[parameters]]
name = "files"
type = "string"
description = "Files to stage"
required = true
"#;
        let config: SkillConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.name, "git-commit");
        assert_eq!(config.description, "Stage and commit changes");
        assert_eq!(config.version, "1.0");
        assert!(config.enabled);
        assert_eq!(config.execution.runner, "shell");
        assert_eq!(config.execution.script, "run.sh");
        assert_eq!(config.execution.timeout_seconds, 30);
        assert_eq!(config.execution.working_dir, "workspace");
        assert_eq!(config.parameters.len(), 2);
        assert_eq!(config.parameters[0].name, "message");
        assert!(!config.parameters[0].required);
        assert_eq!(config.parameters[1].name, "files");
        assert!(config.parameters[1].required);
    }

    #[test]
    fn parse_minimal_skill_toml() {
        let toml_str = r#"
name = "hello"
description = "Say hello"
"#;
        let config: SkillConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.name, "hello");
        assert_eq!(config.version, "1.0");
        assert!(config.enabled);
        assert_eq!(config.execution.runner, "shell");
        assert_eq!(config.execution.script, "run.sh");
        assert_eq!(config.execution.timeout_seconds, 30);
        assert!(config.parameters.is_empty());
    }

    #[test]
    fn parse_invalid_toml_errors() {
        let toml_str = "this is not valid toml {{{";
        let result: Result<SkillConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn parse_missing_required_field_errors() {
        // Missing name field
        let toml_str = r#"
description = "No name"
"#;
        let result: Result<SkillConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }

    // ── LoadedSkill: parameters_schema ──────────────────────────

    #[test]
    fn parameters_schema_correct() {
        let skill = LoadedSkill {
            config: SkillConfig {
                name: "test".to_string(),
                description: "test".to_string(),
                version: "1.0".to_string(),
                enabled: true,
                execution: ExecutionConfig::default(),
                parameters: vec![
                    ParameterDef {
                        name: "input".to_string(),
                        param_type: "string".to_string(),
                        description: "Input text".to_string(),
                        required: true,
                    },
                    ParameterDef {
                        name: "count".to_string(),
                        param_type: "number".to_string(),
                        description: "Count".to_string(),
                        required: false,
                    },
                    ParameterDef {
                        name: "verbose".to_string(),
                        param_type: "boolean".to_string(),
                        description: "Verbose".to_string(),
                        required: false,
                    },
                ],
            },
            dir: PathBuf::from("/tmp/test"),
            skill_md: String::new(),
            enabled: true,
        };

        let schema = skill.parameters_schema();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["input"]["type"], "string");
        assert_eq!(schema["properties"]["count"]["type"], "number");
        assert_eq!(schema["properties"]["verbose"]["type"], "boolean");
        assert_eq!(schema["required"], json!(["input"]));
    }

    // ── LoadedSkill: validate_params ────────────────────────────

    #[test]
    fn validate_params_accepts_valid() {
        let skill = LoadedSkill {
            config: SkillConfig {
                name: "test".to_string(),
                description: "test".to_string(),
                version: "1.0".to_string(),
                enabled: true,
                execution: ExecutionConfig::default(),
                parameters: vec![ParameterDef {
                    name: "input".to_string(),
                    param_type: "string".to_string(),
                    description: "".to_string(),
                    required: true,
                }],
            },
            dir: PathBuf::from("/tmp/test"),
            skill_md: String::new(),
            enabled: true,
        };

        let args = json!({"input": "hello"});
        assert!(skill.validate_params(&args).is_ok());
    }

    #[test]
    fn validate_params_rejects_missing_required() {
        let skill = LoadedSkill {
            config: SkillConfig {
                name: "test".to_string(),
                description: "test".to_string(),
                version: "1.0".to_string(),
                enabled: true,
                execution: ExecutionConfig::default(),
                parameters: vec![
                    ParameterDef {
                        name: "input".to_string(),
                        param_type: "string".to_string(),
                        description: "".to_string(),
                        required: true,
                    },
                    ParameterDef {
                        name: "output".to_string(),
                        param_type: "string".to_string(),
                        description: "".to_string(),
                        required: true,
                    },
                ],
            },
            dir: PathBuf::from("/tmp/test"),
            skill_md: String::new(),
            enabled: true,
        };

        let args = json!({});
        let err = skill.validate_params(&args).unwrap_err();
        assert!(err.to_string().contains("input"));
        assert!(err.to_string().contains("output"));
    }

    #[test]
    fn validate_params_allows_missing_optional() {
        let skill = LoadedSkill {
            config: SkillConfig {
                name: "test".to_string(),
                description: "test".to_string(),
                version: "1.0".to_string(),
                enabled: true,
                execution: ExecutionConfig::default(),
                parameters: vec![ParameterDef {
                    name: "verbose".to_string(),
                    param_type: "boolean".to_string(),
                    description: "".to_string(),
                    required: false,
                }],
            },
            dir: PathBuf::from("/tmp/test"),
            skill_md: String::new(),
            enabled: true,
        };

        let args = json!({});
        assert!(skill.validate_params(&args).is_ok());
    }

    // ── SkillManager: load_skills ───────────────────────────────

    #[tokio::test]
    async fn load_skills_from_directory() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        fs::create_dir_all(&skills_dir).await.unwrap();

        create_skill_dir(
            &skills_dir,
            "hello",
            &valid_toml("hello"),
            Some("Use hello to greet.\n"),
            Some("#!/bin/bash\necho hello"),
        )
        .await;

        create_skill_dir(
            &skills_dir,
            "bye",
            &valid_toml("bye"),
            None,
            Some("#!/bin/bash\necho bye"),
        )
        .await;

        let mut mgr = SkillManager::new(&skills_dir);
        mgr.load_skills().await.unwrap();

        assert_eq!(mgr.all_skills().len(), 2);
        assert!(mgr.get_skill("hello").is_some());
        assert!(mgr.get_skill("bye").is_some());
        assert!(mgr.warnings().is_empty());
    }

    #[tokio::test]
    async fn load_skills_skips_invalid_with_warning() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        fs::create_dir_all(&skills_dir).await.unwrap();

        // Valid skill
        create_skill_dir(
            &skills_dir,
            "good",
            &valid_toml("good"),
            None,
            Some("#!/bin/bash\necho ok"),
        )
        .await;

        // Invalid skill (bad TOML)
        let bad_dir = skills_dir.join("bad");
        fs::create_dir_all(&bad_dir).await.unwrap();
        fs::write(bad_dir.join("skill.toml"), "not {{{ valid toml")
            .await
            .unwrap();

        // Missing skill.toml
        let empty_dir = skills_dir.join("empty");
        fs::create_dir_all(&empty_dir).await.unwrap();

        let mut mgr = SkillManager::new(&skills_dir);
        mgr.load_skills().await.unwrap();

        assert_eq!(mgr.all_skills().len(), 1);
        assert!(mgr.get_skill("good").is_some());
        assert_eq!(mgr.warnings().len(), 2);
    }

    #[tokio::test]
    async fn load_skills_empty_directory() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        fs::create_dir_all(&skills_dir).await.unwrap();

        let mut mgr = SkillManager::new(&skills_dir);
        mgr.load_skills().await.unwrap();

        assert!(mgr.all_skills().is_empty());
        assert!(mgr.warnings().is_empty());
    }

    #[tokio::test]
    async fn load_skills_nonexistent_directory() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("does-not-exist");

        let mut mgr = SkillManager::new(&skills_dir);
        mgr.load_skills().await.unwrap();

        assert!(mgr.all_skills().is_empty());
    }

    #[tokio::test]
    async fn load_skill_with_skill_md() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        fs::create_dir_all(&skills_dir).await.unwrap();

        create_skill_dir(
            &skills_dir,
            "documented",
            &valid_toml("documented"),
            Some("# Documented Skill\n\nUse this skill for documentation.\n"),
            Some("#!/bin/bash\necho doc"),
        )
        .await;

        let mut mgr = SkillManager::new(&skills_dir);
        mgr.load_skills().await.unwrap();

        let skill = mgr.get_skill("documented").unwrap();
        assert!(skill.skill_md.contains("Documented Skill"));
    }

    // ── SkillManager: enable/disable ────────────────────────────

    #[tokio::test]
    async fn enable_disable_skill() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        fs::create_dir_all(&skills_dir).await.unwrap();

        create_skill_dir(
            &skills_dir,
            "toggle",
            &valid_toml("toggle"),
            None,
            Some("#!/bin/bash\necho toggle"),
        )
        .await;

        let mut mgr = SkillManager::new(&skills_dir);
        mgr.load_skills().await.unwrap();

        assert_eq!(mgr.enabled_skills().len(), 1);

        // Disable
        assert!(mgr.disable_skill("toggle"));
        assert_eq!(mgr.enabled_skills().len(), 0);
        assert_eq!(mgr.all_skills().len(), 1);

        // Enable
        assert!(mgr.enable_skill("toggle"));
        assert_eq!(mgr.enabled_skills().len(), 1);
    }

    #[tokio::test]
    async fn enable_nonexistent_returns_false() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        fs::create_dir_all(&skills_dir).await.unwrap();

        let mut mgr = SkillManager::new(&skills_dir);
        mgr.load_skills().await.unwrap();

        assert!(!mgr.enable_skill("nope"));
        assert!(!mgr.disable_skill("nope"));
    }

    #[tokio::test]
    async fn disabled_skills_not_in_enabled_list() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        fs::create_dir_all(&skills_dir).await.unwrap();

        // Skill with enabled = false in toml
        let toml_str = r#"
name = "off"
description = "Disabled skill"
enabled = false
"#;
        create_skill_dir(&skills_dir, "off", toml_str, None, None).await;

        let mut mgr = SkillManager::new(&skills_dir);
        mgr.load_skills().await.unwrap();

        assert_eq!(mgr.all_skills().len(), 1);
        assert_eq!(mgr.enabled_skills().len(), 0);
    }

    // ── SkillManager: reload preserves states ───────────────────

    #[tokio::test]
    async fn reload_preserves_enabled_state() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        fs::create_dir_all(&skills_dir).await.unwrap();

        create_skill_dir(
            &skills_dir,
            "alpha",
            &valid_toml("alpha"),
            None,
            Some("#!/bin/bash\necho alpha"),
        )
        .await;

        let mut mgr = SkillManager::new(&skills_dir);
        mgr.load_skills().await.unwrap();

        // Disable alpha
        mgr.disable_skill("alpha");
        assert_eq!(mgr.enabled_skills().len(), 0);

        // Reload — alpha should still be disabled
        mgr.reload_skills().await.unwrap();
        let skill = mgr.get_skill("alpha").unwrap();
        assert!(!skill.enabled);
    }

    #[tokio::test]
    async fn reload_picks_up_new_skills() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        fs::create_dir_all(&skills_dir).await.unwrap();

        create_skill_dir(
            &skills_dir,
            "first",
            &valid_toml("first"),
            None,
            Some("#!/bin/bash\necho first"),
        )
        .await;

        let mut mgr = SkillManager::new(&skills_dir);
        mgr.load_skills().await.unwrap();
        assert_eq!(mgr.all_skills().len(), 1);

        // Add a new skill at runtime
        create_skill_dir(
            &skills_dir,
            "second",
            &valid_toml("second"),
            None,
            Some("#!/bin/bash\necho second"),
        )
        .await;

        mgr.reload_skills().await.unwrap();
        assert_eq!(mgr.all_skills().len(), 2);
        assert!(mgr.get_skill("second").is_some());
    }

    // ── SkillManager: system prompt section ─────────────────────

    #[tokio::test]
    async fn system_prompt_section_includes_enabled() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        fs::create_dir_all(&skills_dir).await.unwrap();

        create_skill_dir(
            &skills_dir,
            "greeter",
            &valid_toml("greeter"),
            Some("Use greeter to say hi.\n"),
            Some("#!/bin/bash\necho hi"),
        )
        .await;

        let mut mgr = SkillManager::new(&skills_dir);
        mgr.load_skills().await.unwrap();

        let section = mgr.build_system_prompt_section();
        assert!(section.contains("## Available Skills"));
        assert!(section.contains("### greeter"));
        assert!(section.contains("Test skill greeter"));
        assert!(section.contains("Use greeter to say hi."));
        assert!(section.contains("`input` (string) (required)"));
    }

    #[tokio::test]
    async fn system_prompt_section_empty_when_no_skills() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        fs::create_dir_all(&skills_dir).await.unwrap();

        let mut mgr = SkillManager::new(&skills_dir);
        mgr.load_skills().await.unwrap();

        let section = mgr.build_system_prompt_section();
        assert!(section.is_empty());
    }

    #[tokio::test]
    async fn system_prompt_excludes_disabled() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        fs::create_dir_all(&skills_dir).await.unwrap();

        create_skill_dir(
            &skills_dir,
            "active",
            &valid_toml("active"),
            Some("Active skill.\n"),
            Some("#!/bin/bash\necho active"),
        )
        .await;

        create_skill_dir(
            &skills_dir,
            "inactive",
            &valid_toml("inactive"),
            Some("Inactive skill.\n"),
            Some("#!/bin/bash\necho inactive"),
        )
        .await;

        let mut mgr = SkillManager::new(&skills_dir);
        mgr.load_skills().await.unwrap();
        mgr.disable_skill("inactive");

        let section = mgr.build_system_prompt_section();
        assert!(section.contains("### active"));
        assert!(!section.contains("### inactive"));
    }

    // ── Per-agent tool filtering ────────────────────────────────

    #[tokio::test]
    async fn filter_by_tools_config_empty_allows_all() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        fs::create_dir_all(&skills_dir).await.unwrap();

        create_skill_dir(
            &skills_dir,
            "alpha",
            &valid_toml("alpha"),
            None,
            Some("#!/bin/bash\necho a"),
        )
        .await;
        create_skill_dir(
            &skills_dir,
            "beta",
            &valid_toml("beta"),
            None,
            Some("#!/bin/bash\necho b"),
        )
        .await;

        let mut mgr = SkillManager::new(&skills_dir);
        mgr.load_skills().await.unwrap();

        let tools_config = crate::memory::agent::ToolsConfig::default();
        let filtered = mgr.filter_by_tools_config(&tools_config);
        assert_eq!(filtered.len(), 2);
    }

    #[tokio::test]
    async fn filter_by_tools_config_restricts() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        fs::create_dir_all(&skills_dir).await.unwrap();

        create_skill_dir(
            &skills_dir,
            "alpha",
            &valid_toml("alpha"),
            None,
            Some("#!/bin/bash\necho a"),
        )
        .await;
        create_skill_dir(
            &skills_dir,
            "beta",
            &valid_toml("beta"),
            None,
            Some("#!/bin/bash\necho b"),
        )
        .await;

        let mut mgr = SkillManager::new(&skills_dir);
        mgr.load_skills().await.unwrap();

        let tools_config = crate::memory::agent::ToolsConfig {
            allowed: vec!["alpha".to_string(), "shell".to_string()],
        };
        let filtered = mgr.filter_by_tools_config(&tools_config);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].config.name, "alpha");
    }

    // ── Script execution ────────────────────────────────────────

    #[tokio::test]
    async fn execute_skill_captures_stdout() {
        let tmp = TempDir::new().unwrap();
        let skill_dir = tmp.path().join("echo-skill");
        fs::create_dir_all(&skill_dir).await.unwrap();

        let toml_str = r#"
name = "echo-skill"
description = "Echo input"

[execution]
runner = "shell"
script = "run.sh"
timeout_seconds = 5
working_dir = "skill"
"#;
        fs::write(skill_dir.join("skill.toml"), toml_str)
            .await
            .unwrap();

        let script = "#!/bin/bash\necho \"Hello from skill\"";
        fs::write(skill_dir.join("run.sh"), script).await.unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(
                skill_dir.join("run.sh"),
                std::fs::Permissions::from_mode(0o755),
            )
            .unwrap();
        }

        let skill = LoadedSkill {
            config: toml::from_str(toml_str).unwrap(),
            dir: skill_dir.clone(),
            skill_md: String::new(),
            enabled: true,
        };

        let result = execute_skill(&skill, &json!({}), None).await.unwrap();
        assert!(result.success);
        assert_eq!(result.stdout.trim(), "Hello from skill");
        assert!(result.stderr.is_empty());
    }

    #[tokio::test]
    async fn execute_skill_sets_env_vars() {
        let tmp = TempDir::new().unwrap();
        let skill_dir = tmp.path().join("env-skill");
        fs::create_dir_all(&skill_dir).await.unwrap();

        let toml_str = r#"
name = "env-skill"
description = "Check env"

[execution]
runner = "shell"
script = "run.sh"
timeout_seconds = 5
working_dir = "skill"

[[parameters]]
name = "message"
type = "string"
description = "A message"
required = true
"#;
        fs::write(skill_dir.join("skill.toml"), toml_str)
            .await
            .unwrap();

        let script = "#!/bin/bash\necho \"msg=$SKILL_PARAM_MESSAGE\"";
        fs::write(skill_dir.join("run.sh"), script).await.unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(
                skill_dir.join("run.sh"),
                std::fs::Permissions::from_mode(0o755),
            )
            .unwrap();
        }

        let skill = LoadedSkill {
            config: toml::from_str(toml_str).unwrap(),
            dir: skill_dir.clone(),
            skill_md: String::new(),
            enabled: true,
        };

        let args = json!({"message": "world"});
        let result = execute_skill(&skill, &args, None).await.unwrap();
        assert!(result.success);
        assert_eq!(result.stdout.trim(), "msg=world");
    }

    #[tokio::test]
    async fn execute_skill_receives_json_stdin() {
        let tmp = TempDir::new().unwrap();
        let skill_dir = tmp.path().join("stdin-skill");
        fs::create_dir_all(&skill_dir).await.unwrap();

        let toml_str = r#"
name = "stdin-skill"
description = "Read stdin"

[execution]
runner = "shell"
script = "run.sh"
timeout_seconds = 5
working_dir = "skill"

[[parameters]]
name = "data"
type = "string"
description = "Data"
required = true
"#;
        fs::write(skill_dir.join("skill.toml"), toml_str)
            .await
            .unwrap();

        // Script reads stdin and echoes it
        let script = "#!/bin/bash\ncat";
        fs::write(skill_dir.join("run.sh"), script).await.unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(
                skill_dir.join("run.sh"),
                std::fs::Permissions::from_mode(0o755),
            )
            .unwrap();
        }

        let skill = LoadedSkill {
            config: toml::from_str(toml_str).unwrap(),
            dir: skill_dir.clone(),
            skill_md: String::new(),
            enabled: true,
        };

        let args = json!({"data": "test-value"});
        let result = execute_skill(&skill, &args, None).await.unwrap();
        assert!(result.success);
        // stdout should be the JSON
        let parsed: Value = serde_json::from_str(result.stdout.trim()).unwrap();
        assert_eq!(parsed["data"], "test-value");
    }

    #[tokio::test]
    async fn execute_skill_captures_stderr_as_warning() {
        let tmp = TempDir::new().unwrap();
        let skill_dir = tmp.path().join("warn-skill");
        fs::create_dir_all(&skill_dir).await.unwrap();

        let toml_str = r#"
name = "warn-skill"
description = "Warns"

[execution]
runner = "shell"
script = "run.sh"
timeout_seconds = 5
working_dir = "skill"
"#;
        fs::write(skill_dir.join("skill.toml"), toml_str)
            .await
            .unwrap();

        let script = "#!/bin/bash\necho \"result\" && echo \"warning!\" >&2";
        fs::write(skill_dir.join("run.sh"), script).await.unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(
                skill_dir.join("run.sh"),
                std::fs::Permissions::from_mode(0o755),
            )
            .unwrap();
        }

        let skill = LoadedSkill {
            config: toml::from_str(toml_str).unwrap(),
            dir: skill_dir.clone(),
            skill_md: String::new(),
            enabled: true,
        };

        let result = execute_skill(&skill, &json!({}), None).await.unwrap();
        assert!(result.success);
        assert_eq!(result.stdout.trim(), "result");
        assert!(result.stderr.contains("warning!"));
    }

    #[tokio::test]
    async fn execute_skill_timeout_enforcement() {
        let tmp = TempDir::new().unwrap();
        let skill_dir = tmp.path().join("slow-skill");
        fs::create_dir_all(&skill_dir).await.unwrap();

        let toml_str = r#"
name = "slow-skill"
description = "Sleeps"

[execution]
runner = "shell"
script = "run.sh"
timeout_seconds = 1
working_dir = "skill"
"#;
        fs::write(skill_dir.join("skill.toml"), toml_str)
            .await
            .unwrap();

        // Script sleeps longer than timeout
        let script = "#!/bin/bash\nsleep 60\necho done";
        fs::write(skill_dir.join("run.sh"), script).await.unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(
                skill_dir.join("run.sh"),
                std::fs::Permissions::from_mode(0o755),
            )
            .unwrap();
        }

        let skill = LoadedSkill {
            config: toml::from_str(toml_str).unwrap(),
            dir: skill_dir.clone(),
            skill_md: String::new(),
            enabled: true,
        };

        let result = execute_skill(&skill, &json!({}), None).await.unwrap();
        assert!(!result.success);
        assert!(result.stderr.contains("timed out"));
    }

    #[tokio::test]
    async fn execute_skill_missing_script_errors() {
        let tmp = TempDir::new().unwrap();
        let skill_dir = tmp.path().join("no-script");
        fs::create_dir_all(&skill_dir).await.unwrap();

        let toml_str = r#"
name = "no-script"
description = "Missing script"

[execution]
script = "run.sh"
"#;
        fs::write(skill_dir.join("skill.toml"), toml_str)
            .await
            .unwrap();
        // Note: no run.sh created

        let skill = LoadedSkill {
            config: toml::from_str(toml_str).unwrap(),
            dir: skill_dir.clone(),
            skill_md: String::new(),
            enabled: true,
        };

        let result = execute_skill(&skill, &json!({}), None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Script not found"));
    }

    #[tokio::test]
    async fn execute_skill_validates_required_params() {
        let tmp = TempDir::new().unwrap();
        let skill_dir = tmp.path().join("req-skill");
        fs::create_dir_all(&skill_dir).await.unwrap();

        let toml_str = r#"
name = "req-skill"
description = "Requires params"

[execution]
script = "run.sh"

[[parameters]]
name = "files"
type = "string"
description = "Files"
required = true
"#;
        fs::write(skill_dir.join("skill.toml"), toml_str)
            .await
            .unwrap();
        fs::write(skill_dir.join("run.sh"), "#!/bin/bash\necho ok")
            .await
            .unwrap();

        let skill = LoadedSkill {
            config: toml::from_str(toml_str).unwrap(),
            dir: skill_dir.clone(),
            skill_md: String::new(),
            enabled: true,
        };

        // Missing required param
        let result = execute_skill(&skill, &json!({}), None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("files"));
    }

    #[tokio::test]
    async fn execute_python_skill() {
        let tmp = TempDir::new().unwrap();
        let skill_dir = tmp.path().join("py-skill");
        fs::create_dir_all(&skill_dir).await.unwrap();

        let toml_str = r#"
name = "py-skill"
description = "Python skill"

[execution]
runner = "python"
script = "run.py"
timeout_seconds = 5
working_dir = "skill"
"#;
        fs::write(skill_dir.join("skill.toml"), toml_str)
            .await
            .unwrap();

        let script = r#"
import os
import sys
msg = os.environ.get("SKILL_PARAM_NAME", "world")
print(f"hello {msg}")
"#;
        fs::write(skill_dir.join("run.py"), script).await.unwrap();

        let skill = LoadedSkill {
            config: toml::from_str(toml_str).unwrap(),
            dir: skill_dir.clone(),
            skill_md: String::new(),
            enabled: true,
        };

        let args = json!({"name": "zeroclaw"});
        let result = execute_skill(&skill, &args, None).await.unwrap();
        assert!(result.success);
        assert_eq!(result.stdout.trim(), "hello zeroclaw");
    }

    // ── Skill with empty name rejected ──────────────────────────

    #[tokio::test]
    async fn skill_with_empty_name_rejected() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        fs::create_dir_all(&skills_dir).await.unwrap();

        let toml_str = r#"
name = ""
description = "Empty name"
"#;
        create_skill_dir(&skills_dir, "empty-name", toml_str, None, None).await;

        let mut mgr = SkillManager::new(&skills_dir);
        mgr.load_skills().await.unwrap();

        assert!(mgr.all_skills().is_empty());
        assert_eq!(mgr.warnings().len(), 1);
        assert!(mgr.warnings()[0].contains("empty"));
    }
}

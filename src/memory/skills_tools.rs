//! Skill tools and CLI commands for the runtime skills system.
//!
//! This module provides:
//! - `SkillTool`: wraps a loaded skill as an LLM-callable tool
//! - Slash commands: `/skills`, `/skill-info`, `/skills-reload`,
//!   `/skill-enable`, `/skill-disable`
//! - `all_skill_tools()`: registers enabled skills as tools
//!
//! ## Slash commands
//!
//! Commands are intercepted before reaching the LLM:
//! - `/skills`           — list all skills (name, description, status)
//! - `/skill-info <n>`   — show full details of a skill
//! - `/skills-reload`    — rescan skills directory and reload
//! - `/skill-enable <n>` — enable a disabled skill
//! - `/skill-disable <n>`— disable an enabled skill

use crate::memory::skills::{execute_skill, LoadedSkill, SkillManager};
use crate::tools::{Tool, ToolExecutionResult};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;

// ── SkillTool ────────────────────────────────────────────────────

/// Wraps a `LoadedSkill` as an LLM tool implementing the `Tool` trait.
///
/// When the LLM invokes this tool, the skill script is executed with
/// parameter validation, env vars, and JSON stdin piping.
pub struct SkillTool {
    skill: LoadedSkill,
    agent_workspace: Option<std::path::PathBuf>,
}

impl SkillTool {
    /// Create a new SkillTool from a loaded skill.
    pub fn new(skill: LoadedSkill, agent_workspace: Option<std::path::PathBuf>) -> Self {
        Self {
            skill,
            agent_workspace,
        }
    }
}

#[async_trait]
impl Tool for SkillTool {
    fn name(&self) -> &str {
        &self.skill.config.name
    }

    fn description(&self) -> &str {
        &self.skill.config.description
    }

    fn parameters_schema(&self) -> Value {
        self.skill.parameters_schema()
    }

    async fn execute(&self, args: Value) -> Result<ToolExecutionResult> {
        match execute_skill(
            &self.skill,
            &args,
            self.agent_workspace.as_deref(),
        )
        .await
        {
            Ok(result) => {
                let mut output = result.stdout;

                // Append stderr as warning if non-empty
                if !result.stderr.is_empty() {
                    if !output.is_empty() && !output.ends_with('\n') {
                        output.push('\n');
                    }
                    output.push_str(&format!("[warning] {}", result.stderr));
                }

                Ok(ToolExecutionResult {
                    success: result.success,
                    output,
                    error: if result.success {
                        None
                    } else {
                        Some(result.stderr)
                    },
                })
            }
            Err(e) => Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some(format!("Skill execution error: {e}")),
            }),
        }
    }
}

// ── Local command parsing ────────────────────────────────────────

/// A parsed skills command intercepted from user input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkillsCommand {
    /// /skills — list all skills
    List,
    /// /skill-info <name> — show skill details
    Info(String),
    /// /skills-reload — reload skills from disk
    Reload,
    /// /skill-enable <name> — enable a skill
    Enable(String),
    /// /skill-disable <name> — disable a skill
    Disable(String),
}

/// Parse user input for skill commands.
///
/// Returns `Some(SkillsCommand)` if the input is a skills command,
/// `None` if it should be passed through to the LLM.
pub fn parse_skills_command(input: &str) -> Option<SkillsCommand> {
    let trimmed = input.trim();

    // /skills-reload must come before /skills
    if eq_ignore_case(trimmed, "/skills-reload") {
        return Some(SkillsCommand::Reload);
    }

    // /skills (no argument)
    if eq_ignore_case(trimmed, "/skills") {
        return Some(SkillsCommand::List);
    }

    // /skill-info <name>
    if let Some(rest) = strip_cmd(trimmed, "/skill-info") {
        let name = rest.trim();
        if !name.is_empty() {
            return Some(SkillsCommand::Info(name.to_string()));
        }
        return None;
    }

    // /skill-enable <name>
    if let Some(rest) = strip_cmd(trimmed, "/skill-enable") {
        let name = rest.trim();
        if !name.is_empty() {
            return Some(SkillsCommand::Enable(name.to_string()));
        }
        return None;
    }

    // /skill-disable <name>
    if let Some(rest) = strip_cmd(trimmed, "/skill-disable") {
        let name = rest.trim();
        if !name.is_empty() {
            return Some(SkillsCommand::Disable(name.to_string()));
        }
        return None;
    }

    None
}

fn eq_ignore_case(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

fn strip_cmd<'a>(input: &'a str, prefix: &str) -> Option<&'a str> {
    let lower = input.to_lowercase();
    if lower.starts_with(prefix) {
        let rest = &input[prefix.len()..];
        if rest.is_empty() || rest.starts_with(char::is_whitespace) {
            return Some(rest);
        }
    }
    None
}

/// Execute a parsed skills command.
///
/// Returns a human-readable string for the user.
pub async fn execute_skills_command(
    cmd: &SkillsCommand,
    skill_mgr: &Arc<Mutex<SkillManager>>,
) -> Result<String> {
    match cmd {
        SkillsCommand::List => {
            let mgr = skill_mgr.lock().await;
            let skills = mgr.all_skills();

            if skills.is_empty() {
                return Ok("No skills found. Add skills to ~/.zeroclaw/skills/".to_string());
            }

            let mut output = format!("Skills ({} found):\n\n", skills.len());
            output.push_str(&format!(
                "  {:<20} {:<36} {:>8}\n",
                "NAME", "DESCRIPTION", "STATUS",
            ));
            output.push_str(&format!("  {}\n", "-".repeat(66)));

            for skill in &skills {
                let status = if skill.enabled { "enabled" } else { "disabled" };
                let desc: String = skill
                    .config
                    .description
                    .chars()
                    .take(34)
                    .collect();
                output.push_str(&format!(
                    "  {:<20} {:<36} {:>8}\n",
                    skill.config.name, desc, status,
                ));
            }

            if !mgr.warnings().is_empty() {
                output.push_str(&format!("\nWarnings ({}):\n", mgr.warnings().len()));
                for w in mgr.warnings() {
                    output.push_str(&format!("  - {w}\n"));
                }
            }

            Ok(output)
        }
        SkillsCommand::Info(name) => {
            let mgr = skill_mgr.lock().await;
            match mgr.get_skill(name) {
                Some(skill) => {
                    let mut output = format!("Skill: {}\n", skill.config.name);
                    output.push_str(&format!("Description: {}\n", skill.config.description));
                    output.push_str(&format!("Version: {}\n", skill.config.version));
                    output.push_str(&format!(
                        "Status: {}\n",
                        if skill.enabled { "enabled" } else { "disabled" },
                    ));
                    output.push_str(&format!("Directory: {}\n", skill.dir.display()));
                    output.push_str(&format!(
                        "Runner: {}\n",
                        skill.config.execution.runner,
                    ));
                    output.push_str(&format!(
                        "Script: {}\n",
                        skill.config.execution.script,
                    ));
                    output.push_str(&format!(
                        "Timeout: {}s\n",
                        skill.config.execution.timeout_seconds,
                    ));
                    output.push_str(&format!(
                        "Working dir: {}\n",
                        skill.config.execution.working_dir,
                    ));

                    if !skill.config.parameters.is_empty() {
                        output.push_str("\nParameters:\n");
                        for p in &skill.config.parameters {
                            let req = if p.required { "required" } else { "optional" };
                            output.push_str(&format!(
                                "  {} ({}, {}): {}\n",
                                p.name, p.param_type, req, p.description,
                            ));
                        }
                    }

                    if !skill.skill_md.is_empty() {
                        output.push_str("\n--- SKILL.md ---\n");
                        output.push_str(&skill.skill_md);
                        if !skill.skill_md.ends_with('\n') {
                            output.push('\n');
                        }
                    }

                    Ok(output)
                }
                None => Ok(format!("Skill '{}' not found.", name)),
            }
        }
        SkillsCommand::Reload => {
            let mut mgr = skill_mgr.lock().await;
            mgr.reload_skills().await?;

            let count = mgr.all_skills().len();
            let enabled = mgr.enabled_skills().len();
            let mut output = format!(
                "Skills reloaded: {} found, {} enabled.\n",
                count, enabled,
            );

            if !mgr.warnings().is_empty() {
                output.push_str(&format!("Warnings ({}):\n", mgr.warnings().len()));
                for w in mgr.warnings() {
                    output.push_str(&format!("  - {w}\n"));
                }
            }

            Ok(output)
        }
        SkillsCommand::Enable(name) => {
            let mut mgr = skill_mgr.lock().await;
            if mgr.enable_skill(name) {
                Ok(format!("Skill '{}' enabled.", name))
            } else {
                Ok(format!("Skill '{}' not found.", name))
            }
        }
        SkillsCommand::Disable(name) => {
            let mut mgr = skill_mgr.lock().await;
            if mgr.disable_skill(name) {
                Ok(format!("Skill '{}' disabled.", name))
            } else {
                Ok(format!("Skill '{}' not found.", name))
            }
        }
    }
}

// ── Tool registration helper ─────────────────────────────────────

/// Create Tool objects for all enabled skills.
///
/// Each enabled skill becomes a `SkillTool` that can be registered
/// in the agent loop alongside built-in tools.
pub fn all_skill_tools(
    skill_mgr: &SkillManager,
    agent_workspace: Option<&Path>,
) -> Vec<Box<dyn Tool>> {
    skill_mgr
        .enabled_skills()
        .into_iter()
        .map(|skill| {
            Box::new(SkillTool::new(
                skill.clone(),
                agent_workspace.map(|p| p.to_path_buf()),
            )) as Box<dyn Tool>
        })
        .collect()
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::skills::SkillManager;
    use tempfile::TempDir;
    use tokio::fs;

    /// Create a skill directory for testing.
    async fn create_test_skill(
        skills_dir: &Path,
        name: &str,
        enabled: bool,
        with_script: bool,
        with_md: bool,
    ) {
        let dir = skills_dir.join(name);
        fs::create_dir_all(&dir).await.unwrap();

        let toml_content = format!(
            r#"
name = "{name}"
description = "Test skill {name}"
version = "1.0"
enabled = {enabled}

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
"#,
        );
        fs::write(dir.join("skill.toml"), &toml_content)
            .await
            .unwrap();

        if with_script {
            let script = format!("#!/bin/bash\necho \"output from {name}\"");
            fs::write(dir.join("run.sh"), &script).await.unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(
                    dir.join("run.sh"),
                    std::fs::Permissions::from_mode(0o755),
                )
                .unwrap();
            }
        }

        if with_md {
            fs::write(
                dir.join("SKILL.md"),
                format!("# {name}\nUse this skill for {name} tasks.\n"),
            )
            .await
            .unwrap();
        }
    }

    async fn setup_mgr(skills_dir: &Path) -> Arc<Mutex<SkillManager>> {
        let mut mgr = SkillManager::new(skills_dir);
        mgr.load_skills().await.unwrap();
        Arc::new(Mutex::new(mgr))
    }

    // ── parse_skills_command ────────────────────────────────────

    #[test]
    fn parse_skills_list() {
        assert_eq!(
            parse_skills_command("/skills"),
            Some(SkillsCommand::List),
        );
        assert_eq!(
            parse_skills_command("  /SKILLS  "),
            Some(SkillsCommand::List),
        );
    }

    #[test]
    fn parse_skills_reload() {
        assert_eq!(
            parse_skills_command("/skills-reload"),
            Some(SkillsCommand::Reload),
        );
    }

    #[test]
    fn parse_skill_info() {
        assert_eq!(
            parse_skills_command("/skill-info git-commit"),
            Some(SkillsCommand::Info("git-commit".to_string())),
        );
    }

    #[test]
    fn parse_skill_info_no_name() {
        assert_eq!(parse_skills_command("/skill-info"), None);
        assert_eq!(parse_skills_command("/skill-info   "), None);
    }

    #[test]
    fn parse_skill_enable() {
        assert_eq!(
            parse_skills_command("/skill-enable git-commit"),
            Some(SkillsCommand::Enable("git-commit".to_string())),
        );
    }

    #[test]
    fn parse_skill_disable() {
        assert_eq!(
            parse_skills_command("/skill-disable translate"),
            Some(SkillsCommand::Disable("translate".to_string())),
        );
    }

    #[test]
    fn parse_non_skill_command_returns_none() {
        assert_eq!(parse_skills_command("hello"), None);
        assert_eq!(parse_skills_command("/unknown"), None);
        assert_eq!(parse_skills_command(""), None);
    }

    #[test]
    fn parse_no_false_prefix_match() {
        // /skills-reload should not match /skills
        assert_eq!(
            parse_skills_command("/skills-reload"),
            Some(SkillsCommand::Reload),
        );
        // /skillset should not match /skills
        assert_eq!(parse_skills_command("/skillset"), None);
    }

    // ── execute_skills_command: /skills ──────────────────────────

    #[tokio::test]
    async fn execute_list_shows_skills() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        fs::create_dir_all(&skills_dir).await.unwrap();

        create_test_skill(&skills_dir, "alpha", true, true, false).await;
        create_test_skill(&skills_dir, "beta", true, true, false).await;

        let mgr = setup_mgr(&skills_dir).await;
        let result = execute_skills_command(&SkillsCommand::List, &mgr)
            .await
            .unwrap();

        assert!(result.contains("2 found"));
        assert!(result.contains("alpha"));
        assert!(result.contains("beta"));
        assert!(result.contains("enabled"));
    }

    #[tokio::test]
    async fn execute_list_empty() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        fs::create_dir_all(&skills_dir).await.unwrap();

        let mgr = setup_mgr(&skills_dir).await;
        let result = execute_skills_command(&SkillsCommand::List, &mgr)
            .await
            .unwrap();

        assert!(result.contains("No skills found"));
    }

    // ── execute_skills_command: /skill-info ──────────────────────

    #[tokio::test]
    async fn execute_info_shows_details() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        fs::create_dir_all(&skills_dir).await.unwrap();

        create_test_skill(&skills_dir, "detailed", true, true, true).await;

        let mgr = setup_mgr(&skills_dir).await;
        let result = execute_skills_command(
            &SkillsCommand::Info("detailed".to_string()),
            &mgr,
        )
        .await
        .unwrap();

        assert!(result.contains("Skill: detailed"));
        assert!(result.contains("Description:"));
        assert!(result.contains("Version: 1.0"));
        assert!(result.contains("enabled"));
        assert!(result.contains("Runner: shell"));
        assert!(result.contains("input (string, required)"));
        assert!(result.contains("SKILL.md"));
        assert!(result.contains("Use this skill for detailed tasks"));
    }

    #[tokio::test]
    async fn execute_info_not_found() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        fs::create_dir_all(&skills_dir).await.unwrap();

        let mgr = setup_mgr(&skills_dir).await;
        let result = execute_skills_command(
            &SkillsCommand::Info("nope".to_string()),
            &mgr,
        )
        .await
        .unwrap();

        assert!(result.contains("not found"));
    }

    // ── execute_skills_command: /skills-reload ───────────────────

    #[tokio::test]
    async fn execute_reload_picks_up_new() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        fs::create_dir_all(&skills_dir).await.unwrap();

        create_test_skill(&skills_dir, "first", true, true, false).await;

        let mgr = setup_mgr(&skills_dir).await;

        // Add another skill
        create_test_skill(&skills_dir, "second", true, true, false).await;

        let result = execute_skills_command(&SkillsCommand::Reload, &mgr)
            .await
            .unwrap();

        assert!(result.contains("2 found"));
        assert!(result.contains("2 enabled"));
    }

    // ── execute_skills_command: /skill-enable, /skill-disable ───

    #[tokio::test]
    async fn execute_enable_disable() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        fs::create_dir_all(&skills_dir).await.unwrap();

        create_test_skill(&skills_dir, "toggler", true, true, false).await;

        let mgr = setup_mgr(&skills_dir).await;

        // Disable
        let result = execute_skills_command(
            &SkillsCommand::Disable("toggler".to_string()),
            &mgr,
        )
        .await
        .unwrap();
        assert!(result.contains("disabled"));

        {
            let locked = mgr.lock().await;
            assert_eq!(locked.enabled_skills().len(), 0);
        }

        // Enable
        let result = execute_skills_command(
            &SkillsCommand::Enable("toggler".to_string()),
            &mgr,
        )
        .await
        .unwrap();
        assert!(result.contains("enabled"));

        {
            let locked = mgr.lock().await;
            assert_eq!(locked.enabled_skills().len(), 1);
        }
    }

    #[tokio::test]
    async fn execute_enable_not_found() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        fs::create_dir_all(&skills_dir).await.unwrap();

        let mgr = setup_mgr(&skills_dir).await;
        let result = execute_skills_command(
            &SkillsCommand::Enable("nope".to_string()),
            &mgr,
        )
        .await
        .unwrap();
        assert!(result.contains("not found"));
    }

    // ── SkillTool integration ───────────────────────────────────

    #[tokio::test]
    async fn skill_tool_executes() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        fs::create_dir_all(&skills_dir).await.unwrap();

        create_test_skill(&skills_dir, "echo-test", true, true, false).await;

        let mut mgr = SkillManager::new(&skills_dir);
        mgr.load_skills().await.unwrap();

        let skill = mgr.get_skill("echo-test").unwrap().clone();
        let tool = SkillTool::new(skill, None);

        assert_eq!(tool.name(), "echo-test");
        assert!(tool.description().contains("echo-test"));

        let result = tool
            .execute(serde_json::json!({"input": "hello"}))
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("output from echo-test"));
    }

    #[tokio::test]
    async fn skill_tool_missing_required_param() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        fs::create_dir_all(&skills_dir).await.unwrap();

        create_test_skill(&skills_dir, "req-test", true, true, false).await;

        let mut mgr = SkillManager::new(&skills_dir);
        mgr.load_skills().await.unwrap();

        let skill = mgr.get_skill("req-test").unwrap().clone();
        let tool = SkillTool::new(skill, None);

        let result = tool.execute(serde_json::json!({})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("input"));
    }

    // ── all_skill_tools ─────────────────────────────────────────

    #[tokio::test]
    async fn all_skill_tools_returns_enabled_only() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        fs::create_dir_all(&skills_dir).await.unwrap();

        create_test_skill(&skills_dir, "on", true, true, false).await;
        create_test_skill(&skills_dir, "off", false, true, false).await;

        let mut mgr = SkillManager::new(&skills_dir);
        mgr.load_skills().await.unwrap();

        let tools = all_skill_tools(&mgr, None);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "on");
    }

    #[tokio::test]
    async fn all_skill_tools_empty_when_none() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        fs::create_dir_all(&skills_dir).await.unwrap();

        let mut mgr = SkillManager::new(&skills_dir);
        mgr.load_skills().await.unwrap();

        let tools = all_skill_tools(&mgr, None);
        assert!(tools.is_empty());
    }

    // ── SkillTool stderr becomes warning ────────────────────────

    #[tokio::test]
    async fn skill_tool_stderr_appended_as_warning() {
        let tmp = TempDir::new().unwrap();
        let skill_dir = tmp.path().join("warn-skill");
        fs::create_dir_all(&skill_dir).await.unwrap();

        let toml_str = r#"
name = "warn-tool"
description = "Tool with warning"

[execution]
runner = "shell"
script = "run.sh"
timeout_seconds = 5
working_dir = "skill"
"#;
        fs::write(skill_dir.join("skill.toml"), toml_str)
            .await
            .unwrap();

        let script = "#!/bin/bash\necho result && echo caution >&2";
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

        let config: crate::memory::skills::SkillConfig =
            toml::from_str(toml_str).unwrap();
        let skill = LoadedSkill {
            config,
            dir: skill_dir.clone(),
            skill_md: String::new(),
            enabled: true,
        };
        let tool = SkillTool::new(skill, None);

        let result = tool.execute(serde_json::json!({})).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("result"));
        assert!(result.output.contains("[warning]"));
        assert!(result.output.contains("caution"));
    }
}

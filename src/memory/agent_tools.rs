//! Agent tools for multi-agent management.
//!
//! These tools wrap `AgentManager` and implement the `Tool` trait
//! so they can be registered in the agent loop alongside existing tools.
//!
//! Tools provided:
//! - `agent_switch`  — switch to a named agent (`/agent <name>`)
//! - `agent_list`    — list all agents (`/agents`)
//! - `agent_new`     — create a new agent (`/agent-new <name>`)
//! - `agent_delete`  — delete an agent (`/agent-delete <name>`)
//!
//! ## Local command parsing
//!
//! The `parse_agent_command()` function intercepts user input before
//! it reaches the LLM, handling `/agent`, `/agents`, `/agent-new`,
//! and `/agent-delete` as local commands.

use crate::memory::agent::{AgentManager, slugify_agent_name};
use crate::tools::{Tool, ToolExecutionResult};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Mutex;

// ── Local command parsing ────────────────────────────────────────

/// A parsed agent command intercepted from user input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentCommand {
    /// /agent <name> — switch to an agent
    Switch(String),
    /// /agents — list all agents
    List,
    /// /agent-new <name> — create a new agent
    New(String),
    /// /agent-delete <name> — delete an agent
    Delete(String),
}

/// Parse user input for agent commands.
///
/// Returns `Some(AgentCommand)` if the input is an agent command,
/// `None` if it should be passed through to the LLM.
pub fn parse_agent_command(input: &str) -> Option<AgentCommand> {
    let trimmed = input.trim();

    if trimmed.eq_ignore_ascii_case("/agents") {
        return Some(AgentCommand::List);
    }

    // /agent-new must be checked before /agent to avoid prefix collision
    if let Some(rest) = strip_command_prefix(trimmed, "/agent-new") {
        let name = rest.trim();
        if name.is_empty() {
            return None;
        }
        return Some(AgentCommand::New(name.to_string()));
    }

    // /agent-delete
    if let Some(rest) = strip_command_prefix(trimmed, "/agent-delete") {
        let name = rest.trim();
        if name.is_empty() {
            return None;
        }
        return Some(AgentCommand::Delete(name.to_string()));
    }

    // /agent <name> — switch to agent (must come after /agent-new and /agent-delete)
    if let Some(rest) = strip_command_prefix(trimmed, "/agent") {
        let name = rest.trim();
        if name.is_empty() {
            return None; // bare /agent without a name is not valid
        }
        return Some(AgentCommand::Switch(name.to_string()));
    }

    None
}

/// Strip a command prefix (case-insensitive) and return the rest.
fn strip_command_prefix<'a>(input: &'a str, prefix: &str) -> Option<&'a str> {
    let lower = input.to_lowercase();
    if lower.starts_with(prefix) {
        let rest = &input[prefix.len()..];
        if rest.is_empty() || rest.starts_with(char::is_whitespace) {
            return Some(rest);
        }
    }
    None
}

/// Execute a parsed agent command against the agent manager.
///
/// Returns a human-readable response string to display to the user.
pub async fn execute_agent_command(
    cmd: &AgentCommand,
    agent_mgr: &Arc<Mutex<AgentManager>>,
) -> Result<String> {
    match cmd {
        AgentCommand::Switch(name) => {
            let slug = slugify_agent_name(name);
            let mut mgr = agent_mgr.lock().await;
            let (config, tools_config) = mgr.switch_agent(&slug).await?;

            let mut output = format!("Switched to agent: {slug}\n");
            if let Some(desc) = &config.description {
                output.push_str(&format!("  {desc}\n"));
            }
            if let Some(provider) = &config.provider {
                output.push_str(&format!("  Provider: {provider}\n"));
            }
            if let Some(model) = &config.model {
                output.push_str(&format!("  Model: {model}\n"));
            }
            if !tools_config.allowed.is_empty() {
                output.push_str(&format!(
                    "  Tools: {}\n",
                    tools_config.allowed.join(", ")
                ));
            }

            Ok(output)
        }
        AgentCommand::List => {
            let mgr = agent_mgr.lock().await;
            let agents = mgr.list_agents().await?;

            if agents.is_empty() {
                return Ok("No agents found.".to_string());
            }

            let mut output = format!("Agents ({} found):\n\n", agents.len());
            output.push_str(&format!(
                "  {:<16} {:<30} {:>8}\n",
                "NAME", "DESCRIPTION", "SESSIONS"
            ));
            output.push_str(&format!("  {}\n", "-".repeat(58)));

            for info in &agents {
                let marker = if info.active { "* " } else { "  " };
                let desc = info
                    .description
                    .as_deref()
                    .unwrap_or("")
                    .chars()
                    .take(28)
                    .collect::<String>();
                output.push_str(&format!(
                    "{}{:<16} {:<30} {:>8}\n",
                    marker, info.name, desc, info.session_count,
                ));
            }

            let active = mgr.active_agent().to_string();
            output.push_str(&format!("\n* = active agent ({active})"));

            Ok(output)
        }
        AgentCommand::New(name) => {
            let mgr = agent_mgr.lock().await;
            let slug = slugify_agent_name(name);
            mgr.create_agent(name).await?;

            Ok(format!(
                "Created agent: {slug}\n\
                 Directory: agents/{slug}/\n\
                 Edit SOUL.md to customize personality.\n\
                 Use /agent {slug} to switch."
            ))
        }
        AgentCommand::Delete(name) => {
            let mgr = agent_mgr.lock().await;
            let slug = slugify_agent_name(name);
            mgr.delete_agent(&slug).await?;

            Ok(format!("Deleted agent: {slug}"))
        }
    }
}

// ── agent_switch ─────────────────────────────────────────────────

/// Tool: Switch to a named agent (`/agent <name>`).
pub struct AgentSwitchTool {
    agent_mgr: Arc<Mutex<AgentManager>>,
}

impl AgentSwitchTool {
    pub fn new(agent_mgr: Arc<Mutex<AgentManager>>) -> Self {
        Self { agent_mgr }
    }
}

#[async_trait]
impl Tool for AgentSwitchTool {
    fn name(&self) -> &str {
        "agent_switch"
    }

    fn description(&self) -> &str {
        "Switch to a different agent persona. Each agent has its own \
         SOUL.md, workspace, tool allowlist, and session history. \
         This is the /agent command."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the agent to switch to."
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
                    error: Some("Agent name is required.".to_string()),
                });
            }
        };

        let cmd = AgentCommand::Switch(name.to_string());
        match execute_agent_command(&cmd, &self.agent_mgr).await {
            Ok(output) => Ok(ToolExecutionResult {
                success: true,
                output,
                error: None,
            }),
            Err(e) => Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to switch agent: {e}")),
            }),
        }
    }
}

// ── agent_list ───────────────────────────────────────────────────

/// Tool: List all agents (`/agents`).
pub struct AgentListTool {
    agent_mgr: Arc<Mutex<AgentManager>>,
}

impl AgentListTool {
    pub fn new(agent_mgr: Arc<Mutex<AgentManager>>) -> Self {
        Self { agent_mgr }
    }
}

#[async_trait]
impl Tool for AgentListTool {
    fn name(&self) -> &str {
        "agent_list"
    }

    fn description(&self) -> &str {
        "List all available agents with their names, descriptions, \
         active marker, and session counts. This is the /agents command."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }

    async fn execute(&self, _args: Value) -> Result<ToolExecutionResult> {
        let cmd = AgentCommand::List;
        match execute_agent_command(&cmd, &self.agent_mgr).await {
            Ok(output) => Ok(ToolExecutionResult {
                success: true,
                output,
                error: None,
            }),
            Err(e) => Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to list agents: {e}")),
            }),
        }
    }
}

// ── agent_new ────────────────────────────────────────────────────

/// Tool: Create a new agent (`/agent-new <name>`).
pub struct AgentNewTool {
    agent_mgr: Arc<Mutex<AgentManager>>,
}

impl AgentNewTool {
    pub fn new(agent_mgr: Arc<Mutex<AgentManager>>) -> Self {
        Self { agent_mgr }
    }
}

#[async_trait]
impl Tool for AgentNewTool {
    fn name(&self) -> &str {
        "agent_new"
    }

    fn description(&self) -> &str {
        "Create a new agent with its own SOUL.md, workspace, and session history. \
         The agent will be created with a template SOUL.md for customization. \
         This is the /agent-new command."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name for the new agent (will be slugified)."
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
                    error: Some("Agent name is required.".to_string()),
                });
            }
        };

        let cmd = AgentCommand::New(name.to_string());
        match execute_agent_command(&cmd, &self.agent_mgr).await {
            Ok(output) => Ok(ToolExecutionResult {
                success: true,
                output,
                error: None,
            }),
            Err(e) => Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to create agent: {e}")),
            }),
        }
    }
}

// ── agent_delete ─────────────────────────────────────────────────

/// Tool: Delete an agent (`/agent-delete <name>`).
pub struct AgentDeleteTool {
    agent_mgr: Arc<Mutex<AgentManager>>,
}

impl AgentDeleteTool {
    pub fn new(agent_mgr: Arc<Mutex<AgentManager>>) -> Self {
        Self { agent_mgr }
    }
}

#[async_trait]
impl Tool for AgentDeleteTool {
    fn name(&self) -> &str {
        "agent_delete"
    }

    fn description(&self) -> &str {
        "Delete an agent and all its files (SOUL.md, workspace, sessions, memory). \
         Cannot delete the 'default' agent. \
         This is the /agent-delete command."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the agent to delete."
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
                    error: Some("Agent name is required.".to_string()),
                });
            }
        };

        let cmd = AgentCommand::Delete(name.to_string());
        match execute_agent_command(&cmd, &self.agent_mgr).await {
            Ok(output) => Ok(ToolExecutionResult {
                success: true,
                output,
                error: None,
            }),
            Err(e) => Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to delete agent: {e}")),
            }),
        }
    }
}

// ── Tool registration helper ─────────────────────────────────────

/// Create all agent management tools, ready to register in the agent loop.
pub fn all_agent_tools(agent_mgr: Arc<Mutex<AgentManager>>) -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(AgentSwitchTool::new(agent_mgr.clone())),
        Box::new(AgentListTool::new(agent_mgr.clone())),
        Box::new(AgentNewTool::new(agent_mgr.clone())),
        Box::new(AgentDeleteTool::new(agent_mgr)),
    ]
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::agent::AgentManager;
    use tempfile::TempDir;

    async fn setup() -> (TempDir, Arc<Mutex<AgentManager>>) {
        let tmp = TempDir::new().unwrap();
        let mgr = AgentManager::new(tmp.path());
        mgr.ensure_default_agent().await.unwrap();
        let mgr = Arc::new(Mutex::new(mgr));
        (tmp, mgr)
    }

    // ── parse_agent_command tests ───────────────────────────────

    #[test]
    fn parse_agent_switch() {
        assert_eq!(
            parse_agent_command("/agent coder"),
            Some(AgentCommand::Switch("coder".to_string()))
        );
    }

    #[test]
    fn parse_agent_switch_case_insensitive() {
        assert_eq!(
            parse_agent_command("/AGENT researcher"),
            Some(AgentCommand::Switch("researcher".to_string()))
        );
    }

    #[test]
    fn parse_agent_without_name_returns_none() {
        assert_eq!(parse_agent_command("/agent"), None);
        assert_eq!(parse_agent_command("/agent   "), None);
    }

    #[test]
    fn parse_agents_list() {
        assert_eq!(
            parse_agent_command("/agents"),
            Some(AgentCommand::List)
        );
        assert_eq!(
            parse_agent_command("  /agents  "),
            Some(AgentCommand::List)
        );
    }

    #[test]
    fn parse_agent_new() {
        assert_eq!(
            parse_agent_command("/agent-new coder"),
            Some(AgentCommand::New("coder".to_string()))
        );
    }

    #[test]
    fn parse_agent_new_without_name_returns_none() {
        assert_eq!(parse_agent_command("/agent-new"), None);
    }

    #[test]
    fn parse_agent_delete() {
        assert_eq!(
            parse_agent_command("/agent-delete coder"),
            Some(AgentCommand::Delete("coder".to_string()))
        );
    }

    #[test]
    fn parse_agent_delete_without_name_returns_none() {
        assert_eq!(parse_agent_command("/agent-delete"), None);
    }

    #[test]
    fn parse_non_agent_command_returns_none() {
        assert_eq!(parse_agent_command("Hello, how are you?"), None);
        assert_eq!(parse_agent_command("/unknown command"), None);
        assert_eq!(parse_agent_command(""), None);
    }

    #[test]
    fn parse_no_false_prefix_match() {
        // "/agents" should not match "/agent" switch
        // (it's caught by the /agents check first)
        assert_eq!(
            parse_agent_command("/agents"),
            Some(AgentCommand::List)
        );
        // "/agenting" should not match
        assert_eq!(parse_agent_command("/agenting"), None);
    }

    #[test]
    fn parse_agent_new_before_agent_switch() {
        // /agent-new should be parsed as New, not as Switch with name "-new ..."
        assert_eq!(
            parse_agent_command("/agent-new test"),
            Some(AgentCommand::New("test".to_string()))
        );
    }

    // ── execute_agent_command tests ─────────────────────────────

    #[tokio::test]
    async fn execute_new_command() {
        let (_tmp, mgr) = setup().await;
        let cmd = AgentCommand::New("coder".to_string());
        let result = execute_agent_command(&cmd, &mgr).await.unwrap();
        assert!(result.contains("Created agent: coder"));
        assert!(result.contains("Edit SOUL.md"));
    }

    #[tokio::test]
    async fn execute_list_command() {
        let (_tmp, mgr) = setup().await;

        // Create an agent
        {
            let locked = mgr.lock().await;
            locked.create_agent("coder").await.unwrap();
        }

        let cmd = AgentCommand::List;
        let result = execute_agent_command(&cmd, &mgr).await.unwrap();
        assert!(result.contains("default"));
        assert!(result.contains("coder"));
        assert!(result.contains("2 found"));
    }

    #[tokio::test]
    async fn execute_switch_command() {
        let (_tmp, mgr) = setup().await;

        // Create an agent to switch to
        {
            let locked = mgr.lock().await;
            locked.create_agent("coder").await.unwrap();
        }

        let cmd = AgentCommand::Switch("coder".to_string());
        let result = execute_agent_command(&cmd, &mgr).await.unwrap();
        assert!(result.contains("Switched to agent: coder"));

        let locked = mgr.lock().await;
        assert_eq!(locked.active_agent(), "coder");
    }

    #[tokio::test]
    async fn execute_switch_nonexistent_errors() {
        let (_tmp, mgr) = setup().await;
        let cmd = AgentCommand::Switch("nonexistent".to_string());
        let result = execute_agent_command(&cmd, &mgr).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn execute_delete_command() {
        let (_tmp, mgr) = setup().await;

        {
            let locked = mgr.lock().await;
            locked.create_agent("coder").await.unwrap();
        }

        let cmd = AgentCommand::Delete("coder".to_string());
        let result = execute_agent_command(&cmd, &mgr).await.unwrap();
        assert!(result.contains("Deleted agent: coder"));

        let locked = mgr.lock().await;
        assert!(!locked.agent_exists("coder"));
    }

    #[tokio::test]
    async fn execute_delete_default_errors() {
        let (_tmp, mgr) = setup().await;
        let cmd = AgentCommand::Delete("default".to_string());
        let result = execute_agent_command(&cmd, &mgr).await;
        assert!(result.is_err());
    }

    // ── Tool tests ──────────────────────────────────────────────

    #[tokio::test]
    async fn agent_new_tool_creates_agent() {
        let (_tmp, mgr) = setup().await;
        let tool = AgentNewTool::new(mgr.clone());

        let result = tool.execute(json!({"name": "researcher"})).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("Created agent: researcher"));

        let locked = mgr.lock().await;
        assert!(locked.agent_exists("researcher"));
    }

    #[tokio::test]
    async fn agent_new_tool_missing_name() {
        let (_tmp, mgr) = setup().await;
        let tool = AgentNewTool::new(mgr.clone());

        let result = tool.execute(json!({})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("name is required"));
    }

    #[tokio::test]
    async fn agent_switch_tool_works() {
        let (_tmp, mgr) = setup().await;

        {
            let locked = mgr.lock().await;
            locked.create_agent("coder").await.unwrap();
        }

        let tool = AgentSwitchTool::new(mgr.clone());
        let result = tool.execute(json!({"name": "coder"})).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("Switched to agent: coder"));
    }

    #[tokio::test]
    async fn agent_switch_tool_nonexistent() {
        let (_tmp, mgr) = setup().await;
        let tool = AgentSwitchTool::new(mgr.clone());

        let result = tool.execute(json!({"name": "nope"})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("does not exist"));
    }

    #[tokio::test]
    async fn agent_list_tool_works() {
        let (_tmp, mgr) = setup().await;

        {
            let locked = mgr.lock().await;
            locked.create_agent("coder").await.unwrap();
        }

        let tool = AgentListTool::new(mgr.clone());
        let result = tool.execute(json!({})).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("default"));
        assert!(result.output.contains("coder"));
    }

    #[tokio::test]
    async fn agent_delete_tool_works() {
        let (_tmp, mgr) = setup().await;

        {
            let locked = mgr.lock().await;
            locked.create_agent("coder").await.unwrap();
        }

        let tool = AgentDeleteTool::new(mgr.clone());
        let result = tool.execute(json!({"name": "coder"})).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("Deleted agent: coder"));
    }

    #[tokio::test]
    async fn agent_delete_tool_default_fails() {
        let (_tmp, mgr) = setup().await;
        let tool = AgentDeleteTool::new(mgr.clone());

        let result = tool.execute(json!({"name": "default"})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("Cannot delete the default"));
    }

    #[tokio::test]
    async fn agent_delete_tool_missing_name() {
        let (_tmp, mgr) = setup().await;
        let tool = AgentDeleteTool::new(mgr.clone());

        let result = tool.execute(json!({})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("name is required"));
    }

    // ── all_agent_tools ──────────────────────────────────────────

    #[tokio::test]
    async fn all_agent_tools_returns_four_tools() {
        let (_tmp, mgr) = setup().await;
        let tools = all_agent_tools(mgr);
        assert_eq!(tools.len(), 4);

        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"agent_switch"));
        assert!(names.contains(&"agent_list"));
        assert!(names.contains(&"agent_new"));
        assert!(names.contains(&"agent_delete"));
    }

    // ── Tool filtering: agent with tools.toml only gets listed tools ──

    #[tokio::test]
    async fn tool_filtering_with_tools_toml() {
        let (_tmp, mgr) = setup().await;

        // Create an agent with restricted tools
        {
            let locked = mgr.lock().await;
            locked.create_agent("restricted").await.unwrap();

            let tools_toml = r#"allowed = ["shell", "str_replace"]"#;
            tokio::fs::write(
                locked.agent_tools_config_path("restricted"),
                tools_toml,
            )
            .await
            .unwrap();
        }

        // Switch to the agent and verify tool config
        {
            let mut locked = mgr.lock().await;
            let (_config, tools_config) = locked.switch_agent("restricted").await.unwrap();

            assert!(tools_config.is_tool_allowed("shell"));
            assert!(tools_config.is_tool_allowed("str_replace"));
            assert!(!tools_config.is_tool_allowed("web_search"));
            assert!(!tools_config.is_tool_allowed("memory_store"));
        }
    }
}

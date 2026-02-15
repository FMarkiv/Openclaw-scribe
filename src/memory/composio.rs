//! Composio integration for connecting to external apps via OAuth.
//!
//! Composio (composio.dev) handles OAuth flows and provides a unified API
//! for 250+ apps (Gmail, Google Calendar, GitHub, Notion, etc.). Instead
//! of implementing OAuth for each service ourselves, we call Composio's
//! API which manages tokens, refresh flows, and standardized action
//! endpoints.
//!
//! ## Architecture
//!
//! - `ComposioConfig` — configuration from `[composio]` section in config.toml
//! - `ComposioClient` — HTTP client wrapping Composio's REST API
//! - `ComposioTool` — LLM-callable tool wrapping a single Composio action
//! - `ConnectionState` — local cache of connected apps (source of truth is Composio)
//! - `ComposioManager` — orchestrates client, tools, and connection state
//!
//! ## Commands
//!
//! - `/connect <app>` — start OAuth flow, show authorization URL
//! - `/connect-check <app>` — verify if app is connected
//! - `/connections` — list all connected apps
//! - `/disconnect <app>` — remove app connection
//! - `/composio-actions <app>` — list available actions for an app
//!
//! ## Tool Registration
//!
//! When Composio is enabled, connected apps' actions are registered as
//! LLM-callable tools with names like `composio_gmail_send_email`.
//! Tools are dynamically added on `/connect` and removed on `/disconnect`.

use crate::tools::{Tool, ToolExecutionResult};
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

// ── Configuration ───────────────────────────────────────────────

/// Configuration for the Composio integration, from `[composio]` in config.toml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposioConfig {
    /// Whether Composio integration is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// Composio API key. Can also be set via COMPOSIO_API_KEY env var.
    #[serde(default)]
    pub api_key: String,
    /// Base URL for the Composio API.
    #[serde(default = "default_base_url")]
    pub base_url: String,
}

fn default_base_url() -> String {
    "https://backend.composio.dev/api/v2".to_string()
}

impl Default for ComposioConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            api_key: String::new(),
            base_url: default_base_url(),
        }
    }
}

impl ComposioConfig {
    /// Resolve the API key from the config or the environment.
    pub fn resolve_api_key(&self) -> Option<String> {
        if !self.api_key.is_empty() {
            Some(self.api_key.clone())
        } else {
            std::env::var("COMPOSIO_API_KEY").ok()
        }
    }

    /// Check whether the integration is enabled and has an API key.
    pub fn is_usable(&self) -> bool {
        self.enabled && self.resolve_api_key().is_some()
    }
}

// ── API response types ──────────────────────────────────────────

/// Information about an available app from the Composio API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppInfo {
    /// Unique app identifier (e.g. "gmail", "github").
    #[serde(default)]
    pub key: String,
    /// Human-readable name (e.g. "Gmail", "GitHub").
    #[serde(default)]
    pub name: String,
    /// Description of the app.
    #[serde(default)]
    pub description: String,
}

/// Information about an available action from the Composio API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionInfo {
    /// Unique action identifier (e.g. "GMAIL_SEND_EMAIL").
    #[serde(default)]
    pub name: String,
    /// Human-readable display name.
    #[serde(default, alias = "displayName")]
    pub display_name: String,
    /// Description of what the action does.
    #[serde(default)]
    pub description: String,
    /// JSON Schema for the action's parameters.
    #[serde(default)]
    pub parameters: Value,
}

/// Result from executing an action via the Composio API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionResult {
    /// Whether the action succeeded.
    #[serde(default)]
    pub success: bool,
    /// The action's output data.
    #[serde(default)]
    pub data: Value,
    /// Error message if the action failed.
    #[serde(default)]
    pub error: Option<String>,
}

/// Response from the auth URL endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthUrlResponse {
    /// The OAuth URL for the user to visit.
    #[serde(default, alias = "redirectUrl")]
    pub redirect_url: String,
}

/// Response from the auth status endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthStatusResponse {
    /// Whether the app is currently connected/authorized.
    #[serde(default)]
    pub status: String,
}

// ── Connection state ────────────────────────────────────────────

/// Local cache of connected apps. Source of truth is Composio's server;
/// this file at `~/.zeroclaw/composio/connections.json` is just a cache.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConnectionState {
    /// Map of app key to connection info.
    #[serde(default)]
    pub connections: HashMap<String, ConnectionInfo>,
}

/// Per-app connection info cached locally.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionInfo {
    /// App key (e.g. "gmail").
    pub app: String,
    /// When the connection was established (ISO-8601).
    pub connected_at: String,
    /// Available action names for this app.
    #[serde(default)]
    pub actions: Vec<String>,
}

impl ConnectionState {
    /// Load connection state from disk, or return default if file doesn't exist.
    pub async fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = tokio::fs::read_to_string(path)
            .await
            .context("reading connections.json")?;
        let state: Self = serde_json::from_str(&content)
            .context("parsing connections.json")?;
        Ok(state)
    }

    /// Save connection state to disk.
    pub async fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .context("creating composio directory")?;
        }
        let content = serde_json::to_string_pretty(self)
            .context("serializing connections.json")?;
        tokio::fs::write(path, content)
            .await
            .context("writing connections.json")?;
        Ok(())
    }

    /// Check if an app is connected.
    pub fn is_connected(&self, app: &str) -> bool {
        self.connections.contains_key(app)
    }

    /// Add a connection for an app.
    pub fn add_connection(&mut self, app: &str, actions: Vec<String>) {
        self.connections.insert(
            app.to_string(),
            ConnectionInfo {
                app: app.to_string(),
                connected_at: chrono::Utc::now().to_rfc3339(),
                actions,
            },
        );
    }

    /// Remove a connection for an app.
    pub fn remove_connection(&mut self, app: &str) -> bool {
        self.connections.remove(app).is_some()
    }

    /// List all connected app keys.
    pub fn connected_apps(&self) -> Vec<String> {
        let mut apps: Vec<String> = self.connections.keys().cloned().collect();
        apps.sort();
        apps
    }
}

// ── ComposioClient ──────────────────────────────────────────────

/// HTTP client that wraps Composio's REST API.
///
/// All API calls go through this client with proper error handling
/// and timeouts (10s default).
pub struct ComposioClient {
    client: Client,
    base_url: String,
    api_key: String,
}

impl ComposioClient {
    /// Create a new Composio API client.
    pub fn new(base_url: &str, api_key: &str) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap_or_else(|_| Client::new());
        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
        }
    }

    /// Build the full URL for an API endpoint.
    fn url(&self, path: &str) -> String {
        format!("{}/{}", self.base_url, path.trim_start_matches('/'))
    }

    /// List available apps from the Composio API.
    pub async fn list_apps(&self) -> Result<Vec<AppInfo>> {
        let resp = self
            .client
            .get(&self.url("/apps"))
            .header("x-api-key", &self.api_key)
            .send()
            .await
            .context("Composio API request failed")?;

        Self::check_rate_limit(&resp);

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("Composio API returned HTTP {status}: {body}");
        }

        let apps: Vec<AppInfo> = resp.json().await
            .context("Failed to parse apps response")?;
        Ok(apps)
    }

    /// List available actions for a specific app.
    pub async fn list_actions(&self, app: &str) -> Result<Vec<ActionInfo>> {
        let resp = self
            .client
            .get(&self.url("/actions"))
            .header("x-api-key", &self.api_key)
            .query(&[("appNames", app)])
            .send()
            .await
            .context("Composio API request failed")?;

        Self::check_rate_limit(&resp);

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("Composio API returned HTTP {status}: {body}");
        }

        let actions: Vec<ActionInfo> = resp.json().await
            .context("Failed to parse actions response")?;
        Ok(actions)
    }

    /// Get an OAuth URL for the user to authorize an app.
    pub async fn get_auth_url(&self, app: &str) -> Result<String> {
        let resp = self
            .client
            .post(&self.url("/connectedAccounts"))
            .header("x-api-key", &self.api_key)
            .json(&json!({ "appName": app }))
            .send()
            .await
            .context("Composio API request failed")?;

        Self::check_rate_limit(&resp);

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("Composio API returned HTTP {status}: {body}");
        }

        let auth_resp: AuthUrlResponse = resp.json().await
            .context("Failed to parse auth URL response")?;
        Ok(auth_resp.redirect_url)
    }

    /// Check if an app is currently connected/authorized.
    pub async fn check_auth(&self, app: &str) -> Result<bool> {
        let resp = self
            .client
            .get(&self.url("/connectedAccounts"))
            .header("x-api-key", &self.api_key)
            .query(&[("appNames", app), ("status", "ACTIVE")])
            .send()
            .await
            .context("Composio API request failed")?;

        Self::check_rate_limit(&resp);

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("Composio API returned HTTP {status}: {body}");
        }

        // If we get accounts back, the app is connected
        let body: Value = resp.json().await
            .context("Failed to parse auth status response")?;

        let is_connected = body.as_array()
            .map(|arr| !arr.is_empty())
            .or_else(|| body.get("items").and_then(|v| v.as_array()).map(|arr| !arr.is_empty()))
            .unwrap_or(false);

        Ok(is_connected)
    }

    /// Execute an action via the Composio API.
    pub async fn execute_action(
        &self,
        action: &str,
        params: Value,
    ) -> Result<ActionResult> {
        let resp = self
            .client
            .post(&self.url(&format!("/actions/{}/execute", action)))
            .header("x-api-key", &self.api_key)
            .json(&json!({
                "input": params,
            }))
            .send()
            .await
            .context("Composio API request failed")?;

        Self::check_rate_limit(&resp);

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Ok(ActionResult {
                success: false,
                data: Value::Null,
                error: Some(format!("Composio API returned HTTP {status}: {body}")),
            });
        }

        let result: ActionResult = resp.json().await.unwrap_or_else(|_| ActionResult {
            success: true,
            data: Value::Null,
            error: None,
        });
        Ok(result)
    }

    /// Log a warning if rate-limited (Retry-After header present).
    fn check_rate_limit(resp: &reqwest::Response) {
        if resp.status().as_u16() == 429 {
            if let Some(retry_after) = resp.headers().get("retry-after") {
                if let Ok(secs) = retry_after.to_str().unwrap_or("").parse::<u64>() {
                    eprintln!(
                        "[composio] Rate limited. Retry after {secs}s"
                    );
                }
            }
        }
    }
}

// ── ComposioTool ────────────────────────────────────────────────

/// An LLM-callable tool that wraps a single Composio action.
///
/// Tool name format: `composio_{app}_{action}`
/// (e.g. `composio_gmail_send_email`, `composio_github_create_issue`)
pub struct ComposioTool {
    /// Tool name in format composio_{app}_{action_suffix}.
    tool_name: String,
    /// Human-readable description.
    tool_description: String,
    /// JSON Schema for parameters.
    schema: Value,
    /// The Composio action name to execute.
    action_name: String,
    /// Reference to the shared Composio client.
    client: Arc<ComposioClient>,
}

impl ComposioTool {
    /// Create a new ComposioTool from an action's metadata.
    pub fn new(
        app: &str,
        action: &ActionInfo,
        client: Arc<ComposioClient>,
    ) -> Self {
        let tool_name = generate_tool_name(app, &action.name);

        let description = if action.description.is_empty() {
            format!(
                "Execute {} action via Composio ({})",
                action.display_name, app
            )
        } else {
            format!(
                "{} (via Composio {})",
                action.description, app
            )
        };

        let schema = if action.parameters.is_null() || action.parameters.is_object() && action.parameters.as_object().map_or(true, |o| o.is_empty()) {
            json!({
                "type": "object",
                "properties": {
                    "params": {
                        "type": "object",
                        "description": "Parameters for this Composio action."
                    }
                },
                "required": []
            })
        } else {
            action.parameters.clone()
        };

        Self {
            tool_name,
            tool_description: description,
            schema,
            action_name: action.name.clone(),
            client,
        }
    }
}

#[async_trait]
impl Tool for ComposioTool {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn description(&self) -> &str {
        &self.tool_description
    }

    fn parameters_schema(&self) -> Value {
        self.schema.clone()
    }

    async fn execute(&self, args: Value) -> Result<ToolExecutionResult> {
        match self.client.execute_action(&self.action_name, args).await {
            Ok(result) => {
                if result.success {
                    let output = if result.data.is_null() {
                        "Action executed successfully.".to_string()
                    } else {
                        serde_json::to_string_pretty(&result.data)
                            .unwrap_or_else(|_| result.data.to_string())
                    };
                    Ok(ToolExecutionResult {
                        success: true,
                        output,
                        error: None,
                    })
                } else {
                    Ok(ToolExecutionResult {
                        success: false,
                        output: String::new(),
                        error: result.error.or_else(|| Some("Action failed.".to_string())),
                    })
                }
            }
            Err(e) => Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some(format!("Composio action failed: {e}")),
            }),
        }
    }
}

// ── Tool name generation ────────────────────────────────────────

/// Generate a tool name from an app key and action name.
///
/// Format: `composio_{app}_{action_suffix}`
///
/// The action name from Composio is typically like "GMAIL_SEND_EMAIL".
/// We normalize it to lowercase and strip the app prefix if present.
pub fn generate_tool_name(app: &str, action_name: &str) -> String {
    let app_lower = app.to_lowercase();
    let action_lower = action_name.to_lowercase();

    // Strip app prefix if present (e.g. "GMAIL_SEND_EMAIL" → "send_email")
    let action_suffix = if action_lower.starts_with(&format!("{}_", app_lower)) {
        &action_lower[app_lower.len() + 1..]
    } else {
        &action_lower
    };

    format!("composio_{}_{}", app_lower, action_suffix)
}

// ── Command parsing ─────────────────────────────────────────────

/// A parsed Composio command intercepted from user input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComposioCommand {
    /// /connect <app> — start OAuth flow
    Connect(String),
    /// /connect-check <app> — verify if app is connected
    ConnectCheck(String),
    /// /connections — list all connected apps
    Connections,
    /// /disconnect <app> — remove app connection
    Disconnect(String),
    /// /composio-actions <app> — list available actions
    Actions(String),
}

/// Parse user input for Composio commands.
///
/// Returns `Some(ComposioCommand)` if the input is a Composio command,
/// `None` if it should be passed through to the LLM.
pub fn parse_composio_command(input: &str) -> Option<ComposioCommand> {
    let trimmed = input.trim();

    if trimmed.eq_ignore_ascii_case("/connections") {
        return Some(ComposioCommand::Connections);
    }

    // /connect-check must come before /connect to avoid prefix collision
    if let Some(rest) = strip_command_prefix(trimmed, "/connect-check") {
        let app = rest.trim();
        if app.is_empty() {
            return None;
        }
        return Some(ComposioCommand::ConnectCheck(app.to_lowercase()));
    }

    if let Some(rest) = strip_command_prefix(trimmed, "/connect") {
        let app = rest.trim();
        if app.is_empty() {
            return None;
        }
        return Some(ComposioCommand::Connect(app.to_lowercase()));
    }

    if let Some(rest) = strip_command_prefix(trimmed, "/disconnect") {
        let app = rest.trim();
        if app.is_empty() {
            return None;
        }
        return Some(ComposioCommand::Disconnect(app.to_lowercase()));
    }

    if let Some(rest) = strip_command_prefix(trimmed, "/composio-actions") {
        let app = rest.trim();
        if app.is_empty() {
            return None;
        }
        return Some(ComposioCommand::Actions(app.to_lowercase()));
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

// ── ComposioManager ─────────────────────────────────────────────

/// Orchestrates the Composio client, tools, and connection state.
///
/// The manager is shared via `Arc<Mutex<ComposioManager>>` in the
/// agent loop, similar to how `SessionManager` and `AgentManager`
/// are shared.
pub struct ComposioManager {
    /// The Composio HTTP client.
    client: Arc<ComposioClient>,
    /// Path to connections.json cache file.
    connections_path: PathBuf,
    /// Cached connection state.
    state: ConnectionState,
    /// Currently registered Composio tools (keyed by tool name).
    registered_tools: HashMap<String, ComposioTool>,
}

impl ComposioManager {
    /// Create a new ComposioManager.
    ///
    /// `base_dir` is typically `~/.zeroclaw`.
    pub async fn new(config: &ComposioConfig, base_dir: &Path) -> Result<Self> {
        let api_key = config
            .resolve_api_key()
            .unwrap_or_default();

        let client = Arc::new(ComposioClient::new(&config.base_url, &api_key));
        let connections_path = base_dir.join("composio").join("connections.json");
        let state = ConnectionState::load(&connections_path).await?;

        Ok(Self {
            client,
            connections_path,
            state,
            registered_tools: HashMap::new(),
        })
    }

    /// Create a manager for testing with a given base directory.
    #[cfg(test)]
    pub async fn new_test(base_dir: &Path) -> Result<Self> {
        let config = ComposioConfig::default();
        Self::new(&config, base_dir).await
    }

    /// Execute a parsed Composio command.
    pub async fn execute_command(&mut self, cmd: &ComposioCommand) -> Result<String> {
        match cmd {
            ComposioCommand::Connect(app) => {
                match self.client.get_auth_url(app).await {
                    Ok(url) => Ok(format!(
                        "To connect {app}, open this URL in your browser:\n\n\
                         {url}\n\n\
                         After authorizing, run /connect-check {app} to verify."
                    )),
                    Err(e) => Ok(format!("Failed to get auth URL for {app}: {e}")),
                }
            }
            ComposioCommand::ConnectCheck(app) => {
                match self.client.check_auth(app).await {
                    Ok(true) => {
                        // Fetch actions for the newly connected app
                        let actions = self.client.list_actions(app).await
                            .unwrap_or_default();
                        let action_names: Vec<String> = actions.iter()
                            .map(|a| a.name.clone())
                            .collect();

                        // Register tools for this app
                        for action in &actions {
                            let tool = ComposioTool::new(app, action, self.client.clone());
                            self.registered_tools.insert(tool.tool_name.clone(), tool);
                        }

                        // Update connection state
                        self.state.add_connection(app, action_names.clone());
                        self.state.save(&self.connections_path).await?;

                        let tool_names: Vec<String> = actions.iter()
                            .map(|a| generate_tool_name(app, &a.name))
                            .collect();

                        Ok(format!(
                            "{app} is connected!\n\
                             {} action(s) available:\n  {}",
                            actions.len(),
                            tool_names.join("\n  ")
                        ))
                    }
                    Ok(false) => {
                        Ok(format!(
                            "{app} is not connected. Run /connect {app} to start OAuth flow."
                        ))
                    }
                    Err(e) => {
                        Ok(format!("Failed to check auth status for {app}: {e}"))
                    }
                }
            }
            ComposioCommand::Connections => {
                let apps = self.state.connected_apps();
                if apps.is_empty() {
                    return Ok(
                        "No connected apps.\n\
                         Use /connect <app> to connect an app (e.g. /connect gmail)."
                            .to_string(),
                    );
                }

                let mut output = format!("Connected apps ({}):\n\n", apps.len());
                for app in &apps {
                    if let Some(info) = self.state.connections.get(app) {
                        output.push_str(&format!(
                            "  {} — {} action(s), connected {}\n",
                            app,
                            info.actions.len(),
                            &info.connected_at[..10.min(info.connected_at.len())],
                        ));
                    }
                }
                Ok(output)
            }
            ComposioCommand::Disconnect(app) => {
                let removed = self.state.remove_connection(app);
                if removed {
                    // Remove registered tools for this app
                    let prefix = format!("composio_{}_", app);
                    self.registered_tools
                        .retain(|name, _| !name.starts_with(&prefix));

                    self.state.save(&self.connections_path).await?;
                    Ok(format!("Disconnected {app}. Tools removed."))
                } else {
                    Ok(format!("{app} was not connected."))
                }
            }
            ComposioCommand::Actions(app) => {
                match self.client.list_actions(app).await {
                    Ok(actions) => {
                        if actions.is_empty() {
                            return Ok(format!("No actions found for {app}."));
                        }

                        let mut output = format!(
                            "Actions for {app} ({} found):\n\n",
                            actions.len()
                        );
                        for action in &actions {
                            let tool_name = generate_tool_name(app, &action.name);
                            let desc = if action.description.is_empty() {
                                &action.display_name
                            } else {
                                &action.description
                            };
                            output.push_str(&format!("  {tool_name}\n    {desc}\n\n"));
                        }
                        Ok(output)
                    }
                    Err(e) => Ok(format!("Failed to list actions for {app}: {e}")),
                }
            }
        }
    }

    /// Get all currently registered Composio tools as boxed trait objects.
    ///
    /// This is used during tool registration in the agent loop.
    pub fn get_tools(&self) -> Vec<Box<dyn Tool>> {
        // We need to rebuild tools from our state since we can't clone
        // the registered_tools directly. Use the stored action info.
        let mut tools: Vec<Box<dyn Tool>> = Vec::new();
        for (_name, tool) in &self.registered_tools {
            let new_tool = ComposioTool {
                tool_name: tool.tool_name.clone(),
                tool_description: tool.tool_description.clone(),
                schema: tool.schema.clone(),
                action_name: tool.action_name.clone(),
                client: tool.client.clone(),
            };
            tools.push(Box::new(new_tool));
        }
        tools
    }

    /// Get tool names for a specific app.
    pub fn tools_for_app(&self, app: &str) -> Vec<String> {
        let prefix = format!("composio_{}_", app);
        self.registered_tools
            .keys()
            .filter(|name| name.starts_with(&prefix))
            .cloned()
            .collect()
    }

    /// Get the connection state (for system prompt generation).
    pub fn connection_state(&self) -> &ConnectionState {
        &self.state
    }

    /// Initialize tools for all currently connected apps.
    ///
    /// Called on startup when Composio is enabled. Tries to fetch
    /// actions for each connected app. If the Composio API is
    /// unreachable, logs a warning and continues — the agent works
    /// fine without external app access.
    pub async fn initialize_tools(&mut self) {
        let apps = self.state.connected_apps();
        if apps.is_empty() {
            return;
        }

        for app in &apps {
            match self.client.list_actions(app).await {
                Ok(actions) => {
                    for action in &actions {
                        let tool = ComposioTool::new(app, action, self.client.clone());
                        self.registered_tools.insert(tool.tool_name.clone(), tool);
                    }
                    eprintln!("[composio] Loaded {} actions for {app}", actions.len());
                }
                Err(e) => {
                    eprintln!("[composio] Warning: could not load actions for {app}: {e}");
                }
            }
        }
    }
}

// ── System prompt integration ───────────────────────────────────

/// Generate the system prompt section for connected Composio apps.
///
/// When apps are connected, returns a section like:
/// ```text
/// ## Connected Apps
/// You have access to the following external services:
/// - Gmail: composio_gmail_send_email, composio_gmail_read_emails
/// - GitHub: composio_github_create_issue, composio_github_list_repos
/// Use the composio_* tools to interact with these services.
/// ```
///
/// Returns `None` if no apps are connected.
pub fn generate_system_prompt_section(state: &ConnectionState) -> Option<String> {
    let apps = state.connected_apps();
    if apps.is_empty() {
        return None;
    }

    let mut section = String::from("## Connected Apps\nYou have access to the following external services:\n");

    for app in &apps {
        if let Some(info) = state.connections.get(app) {
            let action_tools: Vec<String> = info
                .actions
                .iter()
                .map(|a| generate_tool_name(app, a))
                .collect();

            if action_tools.is_empty() {
                section.push_str(&format!("- {app}: (connected, no actions cached)\n"));
            } else {
                section.push_str(&format!("- {app}: {}\n", action_tools.join(", ")));
            }
        }
    }

    section.push_str("Use the composio_* tools to interact with these services.\n");
    Some(section)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ── ComposioConfig tests ────────────────────────────────────

    #[test]
    fn config_default_values() {
        let config = ComposioConfig::default();
        assert!(!config.enabled);
        assert!(config.api_key.is_empty());
        assert_eq!(config.base_url, "https://backend.composio.dev/api/v2");
    }

    #[test]
    fn config_is_usable_requires_enabled_and_key() {
        let config = ComposioConfig::default();
        assert!(!config.is_usable());

        let config = ComposioConfig {
            enabled: true,
            api_key: String::new(),
            ..Default::default()
        };
        // Without env var set, not usable
        std::env::remove_var("COMPOSIO_API_KEY");
        assert!(!config.is_usable());

        let config = ComposioConfig {
            enabled: true,
            api_key: "test-key".to_string(),
            ..Default::default()
        };
        assert!(config.is_usable());

        let config = ComposioConfig {
            enabled: false,
            api_key: "test-key".to_string(),
            ..Default::default()
        };
        assert!(!config.is_usable());
    }

    #[test]
    fn config_resolve_api_key_prefers_config_over_env() {
        let config = ComposioConfig {
            api_key: "from-config".to_string(),
            ..Default::default()
        };
        assert_eq!(config.resolve_api_key(), Some("from-config".to_string()));
    }

    #[test]
    fn config_resolve_api_key_empty_falls_to_env() {
        std::env::remove_var("COMPOSIO_API_KEY");
        let config = ComposioConfig::default();
        assert!(config.resolve_api_key().is_none());
    }

    #[test]
    fn config_parses_from_toml() {
        let toml_str = r#"
            enabled = true
            api_key = "my-key"
            base_url = "https://custom.api/v2"
        "#;
        let config: ComposioConfig = toml::from_str(toml_str).unwrap();
        assert!(config.enabled);
        assert_eq!(config.api_key, "my-key");
        assert_eq!(config.base_url, "https://custom.api/v2");
    }

    #[test]
    fn config_parses_from_toml_with_defaults() {
        let toml_str = r#"
            enabled = false
        "#;
        let config: ComposioConfig = toml::from_str(toml_str).unwrap();
        assert!(!config.enabled);
        assert!(config.api_key.is_empty());
        assert_eq!(config.base_url, "https://backend.composio.dev/api/v2");
    }

    #[test]
    fn config_parses_disabled_from_empty() {
        let toml_str = "";
        let config: ComposioConfig = toml::from_str(toml_str).unwrap();
        assert!(!config.enabled);
    }

    #[test]
    fn config_missing_api_key_not_usable() {
        let toml_str = r#"
            enabled = true
        "#;
        std::env::remove_var("COMPOSIO_API_KEY");
        let config: ComposioConfig = toml::from_str(toml_str).unwrap();
        assert!(!config.is_usable());
    }

    // ── ConnectionState tests ───────────────────────────────────

    #[tokio::test]
    async fn connection_state_load_missing_file() {
        let state = ConnectionState::load(Path::new("/nonexistent/path.json"))
            .await
            .unwrap();
        assert!(state.connections.is_empty());
    }

    #[tokio::test]
    async fn connection_state_save_and_load() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("composio").join("connections.json");

        let mut state = ConnectionState::default();
        state.add_connection("gmail", vec!["GMAIL_SEND_EMAIL".to_string()]);
        state.add_connection("github", vec!["GITHUB_CREATE_ISSUE".to_string()]);
        state.save(&path).await.unwrap();

        let loaded = ConnectionState::load(&path).await.unwrap();
        assert!(loaded.is_connected("gmail"));
        assert!(loaded.is_connected("github"));
        assert!(!loaded.is_connected("slack"));
        assert_eq!(loaded.connected_apps().len(), 2);
    }

    #[test]
    fn connection_state_add_remove() {
        let mut state = ConnectionState::default();

        state.add_connection("gmail", vec!["SEND".to_string()]);
        assert!(state.is_connected("gmail"));
        assert!(!state.is_connected("github"));

        state.add_connection("github", vec!["CREATE_ISSUE".to_string()]);
        assert_eq!(state.connected_apps().len(), 2);

        assert!(state.remove_connection("gmail"));
        assert!(!state.is_connected("gmail"));
        assert!(!state.remove_connection("gmail")); // already removed
        assert_eq!(state.connected_apps().len(), 1);
    }

    #[test]
    fn connection_state_connected_apps_sorted() {
        let mut state = ConnectionState::default();
        state.add_connection("github", vec![]);
        state.add_connection("gmail", vec![]);
        state.add_connection("notion", vec![]);

        let apps = state.connected_apps();
        assert_eq!(apps, vec!["github", "gmail", "notion"]);
    }

    // ── Tool name generation tests ──────────────────────────────

    #[test]
    fn generate_tool_name_strips_app_prefix() {
        assert_eq!(
            generate_tool_name("gmail", "GMAIL_SEND_EMAIL"),
            "composio_gmail_send_email"
        );
    }

    #[test]
    fn generate_tool_name_no_prefix() {
        assert_eq!(
            generate_tool_name("github", "CREATE_ISSUE"),
            "composio_github_create_issue"
        );
    }

    #[test]
    fn generate_tool_name_case_insensitive() {
        assert_eq!(
            generate_tool_name("GitHub", "GITHUB_CREATE_PR"),
            "composio_github_create_pr"
        );
    }

    #[test]
    fn generate_tool_name_format() {
        // Verify the format is always composio_{app}_{action}
        let name = generate_tool_name("notion", "NOTION_CREATE_PAGE");
        assert!(name.starts_with("composio_"));
        assert!(name.starts_with("composio_notion_"));
        assert_eq!(name, "composio_notion_create_page");
    }

    // ── Command parsing tests ───────────────────────────────────

    #[test]
    fn parse_connect_command() {
        assert_eq!(
            parse_composio_command("/connect gmail"),
            Some(ComposioCommand::Connect("gmail".to_string()))
        );
    }

    #[test]
    fn parse_connect_command_case_insensitive() {
        assert_eq!(
            parse_composio_command("/CONNECT Gmail"),
            Some(ComposioCommand::Connect("gmail".to_string()))
        );
    }

    #[test]
    fn parse_connect_without_app_returns_none() {
        assert_eq!(parse_composio_command("/connect"), None);
        assert_eq!(parse_composio_command("/connect   "), None);
    }

    #[test]
    fn parse_connect_check_command() {
        assert_eq!(
            parse_composio_command("/connect-check gmail"),
            Some(ComposioCommand::ConnectCheck("gmail".to_string()))
        );
    }

    #[test]
    fn parse_connect_check_before_connect() {
        // /connect-check should be parsed before /connect
        assert_eq!(
            parse_composio_command("/connect-check github"),
            Some(ComposioCommand::ConnectCheck("github".to_string()))
        );
    }

    #[test]
    fn parse_connect_check_without_app_returns_none() {
        assert_eq!(parse_composio_command("/connect-check"), None);
    }

    #[test]
    fn parse_connections_command() {
        assert_eq!(
            parse_composio_command("/connections"),
            Some(ComposioCommand::Connections)
        );
        assert_eq!(
            parse_composio_command("  /connections  "),
            Some(ComposioCommand::Connections)
        );
    }

    #[test]
    fn parse_disconnect_command() {
        assert_eq!(
            parse_composio_command("/disconnect gmail"),
            Some(ComposioCommand::Disconnect("gmail".to_string()))
        );
    }

    #[test]
    fn parse_disconnect_without_app_returns_none() {
        assert_eq!(parse_composio_command("/disconnect"), None);
    }

    #[test]
    fn parse_composio_actions_command() {
        assert_eq!(
            parse_composio_command("/composio-actions github"),
            Some(ComposioCommand::Actions("github".to_string()))
        );
    }

    #[test]
    fn parse_composio_actions_without_app_returns_none() {
        assert_eq!(parse_composio_command("/composio-actions"), None);
    }

    #[test]
    fn parse_non_composio_command_returns_none() {
        assert_eq!(parse_composio_command("Hello"), None);
        assert_eq!(parse_composio_command("/unknown"), None);
        assert_eq!(parse_composio_command(""), None);
        assert_eq!(parse_composio_command("/new session"), None);
    }

    #[test]
    fn parse_no_false_prefix_match() {
        // "/connecting" should not match "/connect"
        assert_eq!(parse_composio_command("/connecting stuff"), None);
        // "/disconnecting" should not match "/disconnect"
        assert_eq!(parse_composio_command("/disconnecting stuff"), None);
    }

    // ── ComposioClient tests ────────────────────────────────────

    #[test]
    fn client_url_construction() {
        let client = ComposioClient::new("https://backend.composio.dev/api/v2", "test-key");
        assert_eq!(
            client.url("/apps"),
            "https://backend.composio.dev/api/v2/apps"
        );
        assert_eq!(
            client.url("/actions"),
            "https://backend.composio.dev/api/v2/actions"
        );
        assert_eq!(
            client.url("/actions/GMAIL_SEND_EMAIL/execute"),
            "https://backend.composio.dev/api/v2/actions/GMAIL_SEND_EMAIL/execute"
        );
    }

    #[test]
    fn client_url_strips_trailing_slash() {
        let client = ComposioClient::new("https://api.example.com/v2/", "key");
        assert_eq!(
            client.url("/apps"),
            "https://api.example.com/v2/apps"
        );
    }

    #[test]
    fn client_url_handles_leading_slash() {
        let client = ComposioClient::new("https://api.example.com/v2", "key");
        assert_eq!(
            client.url("apps"),
            "https://api.example.com/v2/apps"
        );
        assert_eq!(
            client.url("/apps"),
            "https://api.example.com/v2/apps"
        );
    }

    #[test]
    fn client_stores_api_key() {
        let client = ComposioClient::new("https://api.example.com", "my-key-123");
        assert_eq!(client.api_key, "my-key-123");
    }

    // ── ComposioTool tests ──────────────────────────────────────

    #[test]
    fn composio_tool_name_format() {
        let client = Arc::new(ComposioClient::new("https://api.example.com", "key"));
        let action = ActionInfo {
            name: "GMAIL_SEND_EMAIL".to_string(),
            display_name: "Send Email".to_string(),
            description: "Send an email via Gmail".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"}
                },
                "required": ["to", "subject", "body"]
            }),
        };

        let tool = ComposioTool::new("gmail", &action, client);
        assert_eq!(tool.name(), "composio_gmail_send_email");
        assert!(tool.description().contains("Gmail"));
    }

    #[test]
    fn composio_tool_default_schema_for_empty_params() {
        let client = Arc::new(ComposioClient::new("https://api.example.com", "key"));
        let action = ActionInfo {
            name: "GITHUB_LIST_REPOS".to_string(),
            display_name: "List Repos".to_string(),
            description: String::new(),
            parameters: Value::Null,
        };

        let tool = ComposioTool::new("github", &action, client);
        let schema = tool.parameters_schema();
        assert_eq!(schema["type"], "object");
    }

    #[test]
    fn composio_tool_passes_custom_schema() {
        let client = Arc::new(ComposioClient::new("https://api.example.com", "key"));
        let custom_schema = json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        });
        let action = ActionInfo {
            name: "GITHUB_SEARCH_REPOS".to_string(),
            display_name: "Search Repos".to_string(),
            description: "Search GitHub repos".to_string(),
            parameters: custom_schema.clone(),
        };

        let tool = ComposioTool::new("github", &action, client);
        assert_eq!(tool.parameters_schema(), custom_schema);
    }

    // ── System prompt integration tests ─────────────────────────

    #[test]
    fn system_prompt_no_connections() {
        let state = ConnectionState::default();
        assert!(generate_system_prompt_section(&state).is_none());
    }

    #[test]
    fn system_prompt_with_connections() {
        let mut state = ConnectionState::default();
        state.add_connection(
            "gmail",
            vec![
                "GMAIL_SEND_EMAIL".to_string(),
                "GMAIL_READ_EMAILS".to_string(),
            ],
        );
        state.add_connection(
            "github",
            vec!["GITHUB_CREATE_ISSUE".to_string()],
        );

        let section = generate_system_prompt_section(&state).unwrap();
        assert!(section.contains("## Connected Apps"));
        assert!(section.contains("gmail"));
        assert!(section.contains("github"));
        assert!(section.contains("composio_gmail_send_email"));
        assert!(section.contains("composio_gmail_read_emails"));
        assert!(section.contains("composio_github_create_issue"));
        assert!(section.contains("composio_*"));
    }

    #[test]
    fn system_prompt_with_empty_actions() {
        let mut state = ConnectionState::default();
        state.add_connection("slack", vec![]);

        let section = generate_system_prompt_section(&state).unwrap();
        assert!(section.contains("slack"));
        assert!(section.contains("no actions cached"));
    }

    // ── ComposioManager tests ───────────────────────────────────

    #[tokio::test]
    async fn manager_creates_with_default_config() {
        let tmp = TempDir::new().unwrap();
        let config = ComposioConfig::default();
        let mgr = ComposioManager::new(&config, tmp.path()).await.unwrap();
        assert!(mgr.state.connections.is_empty());
        assert!(mgr.registered_tools.is_empty());
    }

    #[tokio::test]
    async fn manager_loads_existing_connections() {
        let tmp = TempDir::new().unwrap();

        // Pre-create connections.json
        let conn_dir = tmp.path().join("composio");
        tokio::fs::create_dir_all(&conn_dir).await.unwrap();
        let conn_path = conn_dir.join("connections.json");
        let mut state = ConnectionState::default();
        state.add_connection("gmail", vec!["GMAIL_SEND_EMAIL".to_string()]);
        state.save(&conn_path).await.unwrap();

        let config = ComposioConfig::default();
        let mgr = ComposioManager::new(&config, tmp.path()).await.unwrap();
        assert!(mgr.state.is_connected("gmail"));
        assert!(!mgr.state.is_connected("github"));
    }

    #[tokio::test]
    async fn manager_disconnect_removes_tools_and_connection() {
        let tmp = TempDir::new().unwrap();
        let config = ComposioConfig::default();
        let mut mgr = ComposioManager::new(&config, tmp.path()).await.unwrap();

        // Manually add a connection and tool
        mgr.state.add_connection("gmail", vec!["GMAIL_SEND_EMAIL".to_string()]);
        let action = ActionInfo {
            name: "GMAIL_SEND_EMAIL".to_string(),
            display_name: "Send Email".to_string(),
            description: "Send email".to_string(),
            parameters: Value::Null,
        };
        let tool = ComposioTool::new("gmail", &action, mgr.client.clone());
        mgr.registered_tools.insert(tool.tool_name.clone(), tool);

        assert!(mgr.state.is_connected("gmail"));
        assert!(!mgr.registered_tools.is_empty());

        let cmd = ComposioCommand::Disconnect("gmail".to_string());
        let result = mgr.execute_command(&cmd).await.unwrap();
        assert!(result.contains("Disconnected gmail"));
        assert!(!mgr.state.is_connected("gmail"));
        assert!(mgr.registered_tools.is_empty());
    }

    #[tokio::test]
    async fn manager_disconnect_nonexistent_app() {
        let tmp = TempDir::new().unwrap();
        let config = ComposioConfig::default();
        let mut mgr = ComposioManager::new(&config, tmp.path()).await.unwrap();

        let cmd = ComposioCommand::Disconnect("slack".to_string());
        let result = mgr.execute_command(&cmd).await.unwrap();
        assert!(result.contains("was not connected"));
    }

    #[tokio::test]
    async fn manager_connections_empty() {
        let tmp = TempDir::new().unwrap();
        let config = ComposioConfig::default();
        let mut mgr = ComposioManager::new(&config, tmp.path()).await.unwrap();

        let cmd = ComposioCommand::Connections;
        let result = mgr.execute_command(&cmd).await.unwrap();
        assert!(result.contains("No connected apps"));
    }

    #[tokio::test]
    async fn manager_connections_shows_apps() {
        let tmp = TempDir::new().unwrap();
        let config = ComposioConfig::default();
        let mut mgr = ComposioManager::new(&config, tmp.path()).await.unwrap();

        mgr.state.add_connection(
            "gmail",
            vec!["GMAIL_SEND_EMAIL".to_string(), "GMAIL_READ_EMAILS".to_string()],
        );

        let cmd = ComposioCommand::Connections;
        let result = mgr.execute_command(&cmd).await.unwrap();
        assert!(result.contains("gmail"));
        assert!(result.contains("2 action(s)"));
    }

    #[tokio::test]
    async fn manager_get_tools_returns_registered_tools() {
        let tmp = TempDir::new().unwrap();
        let config = ComposioConfig::default();
        let mut mgr = ComposioManager::new(&config, tmp.path()).await.unwrap();

        // Register some tools
        let action1 = ActionInfo {
            name: "GMAIL_SEND_EMAIL".to_string(),
            display_name: "Send Email".to_string(),
            description: "Send email".to_string(),
            parameters: Value::Null,
        };
        let action2 = ActionInfo {
            name: "GMAIL_READ_EMAILS".to_string(),
            display_name: "Read Emails".to_string(),
            description: "Read emails".to_string(),
            parameters: Value::Null,
        };

        let tool1 = ComposioTool::new("gmail", &action1, mgr.client.clone());
        let tool2 = ComposioTool::new("gmail", &action2, mgr.client.clone());
        mgr.registered_tools.insert(tool1.tool_name.clone(), tool1);
        mgr.registered_tools.insert(tool2.tool_name.clone(), tool2);

        let tools = mgr.get_tools();
        assert_eq!(tools.len(), 2);

        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"composio_gmail_send_email"));
        assert!(names.contains(&"composio_gmail_read_emails"));
    }

    #[tokio::test]
    async fn manager_tools_for_app() {
        let tmp = TempDir::new().unwrap();
        let config = ComposioConfig::default();
        let mut mgr = ComposioManager::new(&config, tmp.path()).await.unwrap();

        let action = ActionInfo {
            name: "GMAIL_SEND_EMAIL".to_string(),
            display_name: "Send Email".to_string(),
            description: "Send email".to_string(),
            parameters: Value::Null,
        };
        let tool = ComposioTool::new("gmail", &action, mgr.client.clone());
        mgr.registered_tools.insert(tool.tool_name.clone(), tool);

        let action2 = ActionInfo {
            name: "GITHUB_CREATE_ISSUE".to_string(),
            display_name: "Create Issue".to_string(),
            description: "Create issue".to_string(),
            parameters: Value::Null,
        };
        let tool2 = ComposioTool::new("github", &action2, mgr.client.clone());
        mgr.registered_tools.insert(tool2.tool_name.clone(), tool2);

        let gmail_tools = mgr.tools_for_app("gmail");
        assert_eq!(gmail_tools.len(), 1);
        assert!(gmail_tools[0].contains("gmail"));

        let github_tools = mgr.tools_for_app("github");
        assert_eq!(github_tools.len(), 1);
        assert!(github_tools[0].contains("github"));

        let slack_tools = mgr.tools_for_app("slack");
        assert!(slack_tools.is_empty());
    }

    // ── Per-agent tool filtering applies to composio_* tools ────

    #[test]
    fn agent_tool_filtering_applies_to_composio_tools() {
        use crate::memory::agent::ToolsConfig;

        let tools_config = ToolsConfig {
            allowed: vec![
                "shell".to_string(),
                "composio_gmail_send_email".to_string(),
            ],
        };

        assert!(tools_config.is_tool_allowed("composio_gmail_send_email"));
        assert!(!tools_config.is_tool_allowed("composio_github_create_issue"));
        assert!(tools_config.is_tool_allowed("shell"));
        assert!(!tools_config.is_tool_allowed("web_search"));
    }

    #[test]
    fn agent_tool_filtering_empty_allows_all_composio() {
        use crate::memory::agent::ToolsConfig;

        let tools_config = ToolsConfig {
            allowed: vec![],
        };

        assert!(tools_config.is_tool_allowed("composio_gmail_send_email"));
        assert!(tools_config.is_tool_allowed("composio_github_create_issue"));
    }

    // ── Graceful degradation test ───────────────────────────────

    #[tokio::test]
    async fn graceful_degradation_composio_unreachable() {
        // Use a fake URL that will fail to connect
        let client = Arc::new(ComposioClient::new(
            "http://localhost:1",
            "fake-key",
        ));

        // list_actions should return an error, not panic
        let result = client.list_actions("gmail").await;
        assert!(result.is_err());

        // list_apps should return an error, not panic
        let result = client.list_apps().await;
        assert!(result.is_err());

        // get_auth_url should return an error, not panic
        let result = client.get_auth_url("gmail").await;
        assert!(result.is_err());

        // check_auth should return an error, not panic
        let result = client.check_auth("gmail").await;
        assert!(result.is_err());
    }

    // ── API response deserialization tests ───────────────────────

    #[test]
    fn app_info_deserializes() {
        let json_str = r#"{
            "key": "gmail",
            "name": "Gmail",
            "description": "Google email service"
        }"#;
        let info: AppInfo = serde_json::from_str(json_str).unwrap();
        assert_eq!(info.key, "gmail");
        assert_eq!(info.name, "Gmail");
    }

    #[test]
    fn app_info_deserializes_with_defaults() {
        let json_str = r#"{}"#;
        let info: AppInfo = serde_json::from_str(json_str).unwrap();
        assert!(info.key.is_empty());
        assert!(info.name.is_empty());
    }

    #[test]
    fn action_info_deserializes() {
        let json_str = r#"{
            "name": "GMAIL_SEND_EMAIL",
            "displayName": "Send Email",
            "description": "Send an email via Gmail",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"}
                }
            }
        }"#;
        let info: ActionInfo = serde_json::from_str(json_str).unwrap();
        assert_eq!(info.name, "GMAIL_SEND_EMAIL");
        assert_eq!(info.display_name, "Send Email");
        assert!(!info.parameters.is_null());
    }

    #[test]
    fn action_info_deserializes_with_defaults() {
        let json_str = r#"{"name": "TEST_ACTION"}"#;
        let info: ActionInfo = serde_json::from_str(json_str).unwrap();
        assert_eq!(info.name, "TEST_ACTION");
        assert!(info.description.is_empty());
    }

    #[test]
    fn action_result_deserializes() {
        let json_str = r#"{
            "success": true,
            "data": {"messageId": "abc123"},
            "error": null
        }"#;
        let result: ActionResult = serde_json::from_str(json_str).unwrap();
        assert!(result.success);
        assert!(result.error.is_none());
    }

    #[test]
    fn action_result_deserializes_failure() {
        let json_str = r#"{
            "success": false,
            "data": null,
            "error": "Invalid credentials"
        }"#;
        let result: ActionResult = serde_json::from_str(json_str).unwrap();
        assert!(!result.success);
        assert_eq!(result.error.unwrap(), "Invalid credentials");
    }

    // ── Parameter passthrough test ──────────────────────────────

    #[test]
    fn composio_tool_preserves_action_name_for_execution() {
        let client = Arc::new(ComposioClient::new("https://api.example.com", "key"));
        let action = ActionInfo {
            name: "GMAIL_SEND_EMAIL".to_string(),
            display_name: "Send Email".to_string(),
            description: "Send email".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"}
                }
            }),
        };

        let tool = ComposioTool::new("gmail", &action, client);
        // The action_name should be preserved for API calls
        assert_eq!(tool.action_name, "GMAIL_SEND_EMAIL");
        // The tool name should be the formatted version
        assert_eq!(tool.tool_name, "composio_gmail_send_email");
    }

    // ── Connection state persistence tests ──────────────────────

    #[tokio::test]
    async fn connection_state_creates_parent_directory() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("deep").join("nested").join("connections.json");

        let mut state = ConnectionState::default();
        state.add_connection("test", vec![]);
        state.save(&path).await.unwrap();

        assert!(path.exists());
        let loaded = ConnectionState::load(&path).await.unwrap();
        assert!(loaded.is_connected("test"));
    }

    #[tokio::test]
    async fn connection_state_roundtrip_preserves_data() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("connections.json");

        let mut state = ConnectionState::default();
        state.add_connection(
            "gmail",
            vec!["SEND".to_string(), "READ".to_string()],
        );
        state.add_connection(
            "github",
            vec!["CREATE_ISSUE".to_string()],
        );
        state.save(&path).await.unwrap();

        let loaded = ConnectionState::load(&path).await.unwrap();
        assert_eq!(loaded.connected_apps().len(), 2);

        let gmail = loaded.connections.get("gmail").unwrap();
        assert_eq!(gmail.actions.len(), 2);
        assert!(gmail.actions.contains(&"SEND".to_string()));
        assert!(gmail.actions.contains(&"READ".to_string()));

        let github = loaded.connections.get("github").unwrap();
        assert_eq!(github.actions.len(), 1);
    }
}

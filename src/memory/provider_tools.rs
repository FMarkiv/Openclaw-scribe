//! Agent tools for provider failover management.
//!
//! Provides the `/provider` command that shows the current primary and
//! fallback provider status, plus any recent failover events.
//!
//! ## Local command parsing
//!
//! The `parse_provider_command()` function intercepts `/provider` from
//! user input before it reaches the LLM.

use crate::memory::failover::FailoverManager;
use crate::tools::{Tool, ToolExecutionResult};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Mutex;

// ── Local command parsing ────────────────────────────────────────

/// Parse user input for the /provider command.
///
/// Returns `true` if the input is the `/provider` command,
/// `false` if it should be passed through to the LLM.
pub fn is_provider_command(input: &str) -> bool {
    input.trim().eq_ignore_ascii_case("/provider")
}

/// Execute the /provider command and return the formatted output.
pub async fn execute_provider_command(
    failover_mgr: &Arc<Mutex<FailoverManager>>,
) -> Result<String> {
    let mgr = failover_mgr.lock().await;
    Ok(mgr.format_provider_status())
}

// ── provider_status tool ─────────────────────────────────────────

/// Tool: Show provider configuration and failover status.
///
/// This implements the `/provider` command. It displays the primary
/// provider, all configured fallbacks, and any recent failover events.
pub struct ProviderStatusTool {
    failover_mgr: Arc<Mutex<FailoverManager>>,
}

impl ProviderStatusTool {
    pub fn new(failover_mgr: Arc<Mutex<FailoverManager>>) -> Self {
        Self { failover_mgr }
    }
}

#[async_trait]
impl Tool for ProviderStatusTool {
    fn name(&self) -> &str {
        "provider_status"
    }

    fn description(&self) -> &str {
        "Show the current LLM provider configuration including the primary provider, \
         configured fallback providers, and any recent failover events. \
         This is the /provider command."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }

    async fn execute(&self, _args: Value) -> Result<ToolExecutionResult> {
        let mgr = self.failover_mgr.lock().await;
        let status = mgr.format_provider_status();

        Ok(ToolExecutionResult {
            success: true,
            output: status,
            error: None,
        })
    }
}

/// Create the provider status tool, ready to register in the agent loop.
pub fn provider_status_tool(
    failover_mgr: Arc<Mutex<FailoverManager>>,
) -> Box<dyn Tool> {
    Box::new(ProviderStatusTool::new(failover_mgr))
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::failover::{ProviderChain, ProviderConfig};

    fn make_failover_mgr(with_fallbacks: bool) -> Arc<Mutex<FailoverManager>> {
        let chain = if with_fallbacks {
            ProviderChain {
                primary: ProviderConfig {
                    provider: "anthropic".to_string(),
                    model: "claude-sonnet-4-20250514".to_string(),
                },
                fallbacks: vec![
                    ProviderConfig {
                        provider: "openai".to_string(),
                        model: "gpt-4o".to_string(),
                    },
                    ProviderConfig {
                        provider: "openrouter".to_string(),
                        model: "anthropic/claude-sonnet-4-20250514".to_string(),
                    },
                ],
            }
        } else {
            ProviderChain::primary_only("anthropic", "claude-sonnet-4-20250514")
        };

        Arc::new(Mutex::new(FailoverManager::new(chain)))
    }

    // ── Command parsing tests ───────────────────────────────────

    #[test]
    fn parse_provider_command_matches() {
        assert!(is_provider_command("/provider"));
        assert!(is_provider_command("  /provider  "));
        assert!(is_provider_command("/PROVIDER"));
        assert!(is_provider_command("/Provider"));
    }

    #[test]
    fn parse_provider_command_no_false_matches() {
        assert!(!is_provider_command("/providers"));
        assert!(!is_provider_command("/provider openai"));
        assert!(!is_provider_command("What is the provider?"));
        assert!(!is_provider_command(""));
        assert!(!is_provider_command("/new"));
    }

    // ── Tool tests ──────────────────────────────────────────────

    #[tokio::test]
    async fn provider_status_tool_shows_primary() {
        let mgr = make_failover_mgr(false);
        let tool = ProviderStatusTool::new(mgr);

        let result = tool.execute(json!({})).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("anthropic"));
        assert!(result.output.contains("claude-sonnet-4-20250514"));
        assert!(result.output.contains("Fallbacks: none"));
    }

    #[tokio::test]
    async fn provider_status_tool_shows_fallbacks() {
        let mgr = make_failover_mgr(true);
        let tool = ProviderStatusTool::new(mgr);

        let result = tool.execute(json!({})).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("anthropic"));
        assert!(result.output.contains("openai"));
        assert!(result.output.contains("openrouter"));
        assert!(result.output.contains("Fallbacks (in order)"));
    }

    #[tokio::test]
    async fn execute_provider_command_returns_status() {
        let mgr = make_failover_mgr(true);
        let result = execute_provider_command(&mgr).await.unwrap();
        assert!(result.contains("Provider Configuration"));
        assert!(result.contains("anthropic"));
    }

    #[tokio::test]
    async fn provider_status_tool_registered_correctly() {
        let mgr = make_failover_mgr(false);
        let tool = provider_status_tool(mgr);
        assert_eq!(tool.name(), "provider_status");
        assert!(!tool.description().is_empty());
    }
}

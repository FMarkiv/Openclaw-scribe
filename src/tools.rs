//! Tool trait and common types for ZeroClaw agent tools.
//!
//! All agent tools implement the `Tool` trait, which provides a name,
//! description, JSON Schema for parameters, and an async execute method.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

/// Result returned by tool execution.
pub struct ToolExecutionResult {
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
}

/// Trait that all agent tools must implement.
///
/// Tools are registered in the agent loop and exposed to the LLM as
/// callable functions. The LLM chooses which tool to call based on
/// the name, description, and parameter schema.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The tool's unique name (used in API tool_call messages).
    fn name(&self) -> &str;

    /// Human-readable description shown to the LLM.
    fn description(&self) -> &str;

    /// JSON Schema describing the tool's parameters.
    fn parameters_schema(&self) -> Value;

    /// Execute the tool with the given arguments.
    async fn execute(&self, args: Value) -> Result<ToolExecutionResult>;
}

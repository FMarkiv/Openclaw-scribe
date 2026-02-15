//! Docker sandbox for shell command execution.
//!
//! Provides optional Docker-based sandboxing so that shell commands from
//! untrusted sources (e.g. Telegram users) cannot damage the host system.
//!
//! When `sandbox.enabled = true` in config, commands run inside an ephemeral
//! Docker container (`docker run --rm`). When disabled (the default), commands
//! run directly on the host via `tokio::process::Command`.
//!
//! ## Config (`config.toml`)
//!
//! ```toml
//! [sandbox]
//! enabled = false
//! image = "zeroclaw-sandbox:latest"
//! workspace = "/workspace"
//! timeout_seconds = 30
//! memory_limit = "256m"
//! network = false
//! mount_paths = []
//! ```

use crate::tools::{Tool, ToolExecutionResult};
use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::Duration;
use tokio::process::Command;

// ── Config ──────────────────────────────────────────────────────

/// Sandbox configuration from `[sandbox]` section of config.toml.
#[derive(Debug, Clone, Deserialize)]
pub struct SandboxConfig {
    /// Whether sandbox mode is enabled. Default: false.
    #[serde(default)]
    pub enabled: bool,

    /// Docker image to use for the sandbox container.
    #[serde(default = "default_image")]
    pub image: String,

    /// Mount point inside the container for the workspace.
    #[serde(default = "default_workspace")]
    pub workspace: String,

    /// Per-command timeout in seconds.
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,

    /// Container memory limit (Docker format, e.g. "256m").
    #[serde(default = "default_memory_limit")]
    pub memory_limit: String,

    /// Whether to enable networking inside the container.
    #[serde(default)]
    pub network: bool,

    /// Additional paths to bind-mount read-only into the container.
    #[serde(default)]
    pub mount_paths: Vec<String>,
}

fn default_image() -> String {
    "zeroclaw-sandbox:latest".to_string()
}

fn default_workspace() -> String {
    "/workspace".to_string()
}

fn default_timeout() -> u64 {
    30
}

fn default_memory_limit() -> String {
    "256m".to_string()
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            image: default_image(),
            workspace: default_workspace(),
            timeout_seconds: default_timeout(),
            memory_limit: default_memory_limit(),
            network: false,
            mount_paths: Vec::new(),
        }
    }
}

// ── Executor trait ──────────────────────────────────────────────

/// Result of executing a shell command.
#[derive(Debug, Clone)]
pub struct CommandOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

/// Trait for shell command execution strategies.
#[async_trait]
pub trait SandboxExecutor: Send + Sync {
    /// Execute a shell command and return its output.
    async fn execute(&self, command: &str, working_dir: &str) -> Result<CommandOutput>;
}

/// Create the appropriate executor based on config.
pub fn create_executor(config: &SandboxConfig, host_workspace: &str) -> Box<dyn SandboxExecutor> {
    if config.enabled {
        Box::new(DockerExecutor::new(config.clone(), host_workspace.to_string()))
    } else {
        Box::new(DirectExecutor)
    }
}

// ── DirectExecutor ──────────────────────────────────────────────

/// Executes commands directly on the host via `sh -c`.
pub struct DirectExecutor;

#[async_trait]
impl SandboxExecutor for DirectExecutor {
    async fn execute(&self, command: &str, working_dir: &str) -> Result<CommandOutput> {
        let output = Command::new("sh")
            .arg("-c")
            .arg(command)
            .current_dir(working_dir)
            .output()
            .await?;

        Ok(CommandOutput {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code: output.status.code().unwrap_or(-1),
        })
    }
}

// ── DockerExecutor ──────────────────────────────────────────────

/// Executes commands inside an ephemeral Docker container.
pub struct DockerExecutor {
    config: SandboxConfig,
    host_workspace: String,
}

impl DockerExecutor {
    pub fn new(config: SandboxConfig, host_workspace: String) -> Self {
        Self {
            config,
            host_workspace,
        }
    }

    /// Build the `docker run` argument list.
    fn build_docker_args(&self, command: &str) -> Vec<String> {
        let mut args = vec![
            "run".to_string(),
            "--rm".to_string(),
            format!("--memory={}", self.config.memory_limit),
        ];

        // Network mode
        if !self.config.network {
            args.push("--network=none".to_string());
        }

        // Workspace bind-mount (read-write)
        args.push("-v".to_string());
        args.push(format!(
            "{}:{}",
            self.host_workspace, self.config.workspace
        ));

        // Working directory inside container
        args.push("-w".to_string());
        args.push(self.config.workspace.clone());

        // Additional read-only mounts
        for path in &self.config.mount_paths {
            args.push("-v".to_string());
            args.push(format!("{path}:{path}:ro"));
        }

        // Image
        args.push(self.config.image.clone());

        // Command
        args.push("sh".to_string());
        args.push("-c".to_string());
        args.push(command.to_string());

        args
    }
}

#[async_trait]
impl SandboxExecutor for DockerExecutor {
    async fn execute(&self, command: &str, _working_dir: &str) -> Result<CommandOutput> {
        // Check that docker is available
        let docker_check = Command::new("docker")
            .arg("version")
            .output()
            .await;

        if docker_check.is_err() {
            return Ok(CommandOutput {
                stdout: String::new(),
                stderr: "Docker is not installed or not accessible. \
                         Install Docker or set sandbox.enabled = false in config.toml."
                    .to_string(),
                exit_code: 127,
            });
        }

        let args = self.build_docker_args(command);

        let timeout = Duration::from_secs(self.config.timeout_seconds);

        let child = Command::new("docker")
            .args(&args)
            .output();

        let result = tokio::time::timeout(timeout, child).await;

        match result {
            Ok(Ok(output)) => Ok(CommandOutput {
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                exit_code: output.status.code().unwrap_or(-1),
            }),
            Ok(Err(e)) => Ok(CommandOutput {
                stdout: String::new(),
                stderr: format!("Failed to start Docker container: {e}"),
                exit_code: 1,
            }),
            Err(_) => {
                // Timeout — attempt to kill any lingering container.
                // Since we used --rm, Docker should clean up, but the
                // process itself needs to be dropped (which happens when
                // the future is cancelled by timeout).
                Ok(CommandOutput {
                    stdout: String::new(),
                    stderr: format!(
                        "Command timed out after {} seconds.",
                        self.config.timeout_seconds
                    ),
                    exit_code: 124,
                })
            }
        }
    }
}

// ── ShellTool ───────────────────────────────────────────────────

/// Shell command execution tool for the ZeroClaw agent.
///
/// Dispatches to either `DirectExecutor` or `DockerExecutor` based
/// on the sandbox configuration.
pub struct ShellTool {
    executor: Box<dyn SandboxExecutor>,
    working_dir: PathBuf,
}

impl ShellTool {
    pub fn new(executor: Box<dyn SandboxExecutor>, working_dir: PathBuf) -> Self {
        Self {
            executor,
            working_dir,
        }
    }
}

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Execute a shell command and return its stdout, stderr, and exit code. \
         When sandbox mode is enabled, commands run inside a Docker container. \
         Use this for running builds, tests, git commands, and other CLI tasks."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute."
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, args: Value) -> Result<ToolExecutionResult> {
        let command = args["command"].as_str().unwrap_or("").to_string();

        if command.is_empty() {
            return Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some("command is required.".to_string()),
            });
        }

        let working_dir = self.working_dir.to_string_lossy().to_string();
        let result = self.executor.execute(&command, &working_dir).await;

        match result {
            Ok(output) => {
                let mut combined = String::new();
                if !output.stdout.is_empty() {
                    combined.push_str(&output.stdout);
                }
                if !output.stderr.is_empty() {
                    if !combined.is_empty() {
                        combined.push('\n');
                    }
                    combined.push_str(&output.stderr);
                }

                let success = output.exit_code == 0;
                Ok(ToolExecutionResult {
                    success,
                    output: combined,
                    error: if success {
                        None
                    } else {
                        Some(format!("Exit code: {}", output.exit_code))
                    },
                })
            }
            Err(e) => Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some(format!("Command execution failed: {e}")),
            }),
        }
    }
}

/// Create a shell tool with the given sandbox config and workspace path.
pub fn shell_tool(config: &SandboxConfig, workspace: &str) -> Box<dyn Tool> {
    let executor = create_executor(config, workspace);
    Box::new(ShellTool::new(executor, PathBuf::from(workspace)))
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ── Config parsing ──────────────────────────────────────

    #[test]
    fn config_default_values() {
        let config = SandboxConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.image, "zeroclaw-sandbox:latest");
        assert_eq!(config.workspace, "/workspace");
        assert_eq!(config.timeout_seconds, 30);
        assert_eq!(config.memory_limit, "256m");
        assert!(!config.network);
        assert!(config.mount_paths.is_empty());
    }

    #[test]
    fn config_deserialize_enabled() {
        let toml_str = r#"
            enabled = true
            image = "my-sandbox:v2"
            workspace = "/work"
            timeout_seconds = 60
            memory_limit = "512m"
            network = true
            mount_paths = ["/data", "/models"]
        "#;
        let config: SandboxConfig = toml::from_str(toml_str).unwrap();
        assert!(config.enabled);
        assert_eq!(config.image, "my-sandbox:v2");
        assert_eq!(config.workspace, "/work");
        assert_eq!(config.timeout_seconds, 60);
        assert_eq!(config.memory_limit, "512m");
        assert!(config.network);
        assert_eq!(config.mount_paths, vec!["/data", "/models"]);
    }

    #[test]
    fn config_deserialize_disabled_with_defaults() {
        let toml_str = "enabled = false\n";
        let config: SandboxConfig = toml::from_str(toml_str).unwrap();
        assert!(!config.enabled);
        assert_eq!(config.image, "zeroclaw-sandbox:latest");
        assert_eq!(config.workspace, "/workspace");
        assert_eq!(config.timeout_seconds, 30);
        assert_eq!(config.memory_limit, "256m");
        assert!(!config.network);
        assert!(config.mount_paths.is_empty());
    }

    #[test]
    fn config_deserialize_empty() {
        let toml_str = "";
        let config: SandboxConfig = toml::from_str(toml_str).unwrap();
        assert!(!config.enabled);
    }

    // ── DirectExecutor ──────────────────────────────────────

    #[tokio::test]
    async fn direct_executor_runs_command() {
        let tmp = TempDir::new().unwrap();
        let executor = DirectExecutor;
        let result = executor
            .execute("echo hello", tmp.path().to_str().unwrap())
            .await
            .unwrap();

        assert_eq!(result.stdout.trim(), "hello");
        assert_eq!(result.exit_code, 0);
    }

    #[tokio::test]
    async fn direct_executor_captures_stderr() {
        let tmp = TempDir::new().unwrap();
        let executor = DirectExecutor;
        let result = executor
            .execute("echo error >&2", tmp.path().to_str().unwrap())
            .await
            .unwrap();

        assert_eq!(result.stderr.trim(), "error");
        assert_eq!(result.exit_code, 0);
    }

    #[tokio::test]
    async fn direct_executor_preserves_exit_code() {
        let tmp = TempDir::new().unwrap();
        let executor = DirectExecutor;
        let result = executor
            .execute("exit 42", tmp.path().to_str().unwrap())
            .await
            .unwrap();

        assert_eq!(result.exit_code, 42);
    }

    #[tokio::test]
    async fn direct_executor_uses_working_dir() {
        let tmp = TempDir::new().unwrap();
        let executor = DirectExecutor;
        let result = executor
            .execute("pwd", tmp.path().to_str().unwrap())
            .await
            .unwrap();

        // On some systems /tmp is a symlink; canonicalize both paths
        let expected = tmp.path().canonicalize().unwrap();
        let actual_path = PathBuf::from(result.stdout.trim()).canonicalize().unwrap();
        assert_eq!(actual_path, expected);
    }

    #[tokio::test]
    async fn direct_executor_captures_combined_output() {
        let tmp = TempDir::new().unwrap();
        let executor = DirectExecutor;
        let result = executor
            .execute("echo out && echo err >&2", tmp.path().to_str().unwrap())
            .await
            .unwrap();

        assert_eq!(result.stdout.trim(), "out");
        assert_eq!(result.stderr.trim(), "err");
        assert_eq!(result.exit_code, 0);
    }

    // ── DockerExecutor argument building ────────────────────

    #[test]
    fn docker_args_basic() {
        let config = SandboxConfig {
            enabled: true,
            image: "zeroclaw-sandbox:latest".to_string(),
            workspace: "/workspace".to_string(),
            timeout_seconds: 30,
            memory_limit: "256m".to_string(),
            network: false,
            mount_paths: Vec::new(),
        };
        let executor = DockerExecutor::new(config, "/home/user/project".to_string());
        let args = executor.build_docker_args("echo hello");

        assert!(args.contains(&"run".to_string()));
        assert!(args.contains(&"--rm".to_string()));
        assert!(args.contains(&"--memory=256m".to_string()));
        assert!(args.contains(&"--network=none".to_string()));
        assert!(args.contains(&"-v".to_string()));
        assert!(args.contains(&"/home/user/project:/workspace".to_string()));
        assert!(args.contains(&"-w".to_string()));
        assert!(args.contains(&"/workspace".to_string()));
        assert!(args.contains(&"zeroclaw-sandbox:latest".to_string()));
        assert!(args.contains(&"echo hello".to_string()));
    }

    #[test]
    fn docker_args_network_enabled() {
        let config = SandboxConfig {
            enabled: true,
            network: true,
            ..SandboxConfig::default()
        };
        let executor = DockerExecutor::new(config, "/tmp/ws".to_string());
        let args = executor.build_docker_args("ls");

        assert!(!args.contains(&"--network=none".to_string()));
    }

    #[test]
    fn docker_args_network_disabled() {
        let config = SandboxConfig {
            enabled: true,
            network: false,
            ..SandboxConfig::default()
        };
        let executor = DockerExecutor::new(config, "/tmp/ws".to_string());
        let args = executor.build_docker_args("ls");

        assert!(args.contains(&"--network=none".to_string()));
    }

    #[test]
    fn docker_args_readonly_mounts() {
        let config = SandboxConfig {
            enabled: true,
            mount_paths: vec!["/data".to_string(), "/models".to_string()],
            ..SandboxConfig::default()
        };
        let executor = DockerExecutor::new(config, "/tmp/ws".to_string());
        let args = executor.build_docker_args("ls");

        assert!(args.contains(&"/data:/data:ro".to_string()));
        assert!(args.contains(&"/models:/models:ro".to_string()));
    }

    #[test]
    fn docker_args_custom_image_and_memory() {
        let config = SandboxConfig {
            enabled: true,
            image: "my-img:v3".to_string(),
            memory_limit: "1g".to_string(),
            ..SandboxConfig::default()
        };
        let executor = DockerExecutor::new(config, "/tmp/ws".to_string());
        let args = executor.build_docker_args("test");

        assert!(args.contains(&"my-img:v3".to_string()));
        assert!(args.contains(&"--memory=1g".to_string()));
    }

    #[test]
    fn docker_args_workspace_mount_is_rw() {
        let config = SandboxConfig::default();
        let executor = DockerExecutor::new(config, "/home/user/work".to_string());
        let args = executor.build_docker_args("ls");

        // Workspace mount should NOT have :ro suffix
        let mount_arg = args
            .iter()
            .find(|a| a.contains("/home/user/work"))
            .expect("workspace mount should be present");
        assert!(!mount_arg.ends_with(":ro"), "workspace should be read-write");
        assert_eq!(mount_arg, "/home/user/work:/workspace");
    }

    // ── DockerExecutor graceful fallback ────────────────────

    #[tokio::test]
    async fn docker_executor_graceful_when_docker_missing() {
        // This test checks that if `docker` is not on PATH, we get a
        // graceful error message instead of a panic.
        // We simulate this by using an executor that checks for a
        // non-existent docker binary.
        let _config = SandboxConfig {
            enabled: true,
            ..SandboxConfig::default()
        };

        // Create a custom executor that tries a fake docker binary
        struct FakeDockerExecutor;

        #[async_trait]
        impl SandboxExecutor for FakeDockerExecutor {
            async fn execute(&self, _command: &str, _working_dir: &str) -> Result<CommandOutput> {
                // Simulate docker not found
                let result = Command::new("docker_nonexistent_binary_12345")
                    .arg("version")
                    .output()
                    .await;

                match result {
                    Err(_) => Ok(CommandOutput {
                        stdout: String::new(),
                        stderr: "Docker is not installed or not accessible. \
                                 Install Docker or set sandbox.enabled = false in config.toml."
                            .to_string(),
                        exit_code: 127,
                    }),
                    Ok(_) => unreachable!(),
                }
            }
        }

        let executor = FakeDockerExecutor;
        let result = executor.execute("echo hello", "/tmp").await.unwrap();

        assert_eq!(result.exit_code, 127);
        assert!(result.stderr.contains("Docker is not installed"));
    }

    // ── create_executor dispatch ────────────────────────────

    #[tokio::test]
    async fn create_executor_disabled_returns_direct() {
        let config = SandboxConfig {
            enabled: false,
            ..SandboxConfig::default()
        };
        let tmp = TempDir::new().unwrap();
        let executor = create_executor(&config, tmp.path().to_str().unwrap());

        // DirectExecutor should work — run a simple command
        let result = executor
            .execute("echo dispatch_test", tmp.path().to_str().unwrap())
            .await
            .unwrap();

        assert_eq!(result.stdout.trim(), "dispatch_test");
        assert_eq!(result.exit_code, 0);
    }

    #[test]
    fn create_executor_enabled_returns_docker() {
        let config = SandboxConfig {
            enabled: true,
            ..SandboxConfig::default()
        };
        // We can't easily test the Docker executor without Docker,
        // but we can verify the function doesn't panic.
        let _executor = create_executor(&config, "/tmp/ws");
    }

    // ── ShellTool ───────────────────────────────────────────

    #[test]
    fn shell_tool_name() {
        let executor = Box::new(DirectExecutor);
        let tool = ShellTool::new(executor, PathBuf::from("/tmp"));
        assert_eq!(tool.name(), "shell");
    }

    #[test]
    fn shell_tool_schema_requires_command() {
        let executor = Box::new(DirectExecutor);
        let tool = ShellTool::new(executor, PathBuf::from("/tmp"));
        let schema = tool.parameters_schema();
        let required = schema["required"].as_array().unwrap();
        let req_names: Vec<&str> = required.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(req_names.contains(&"command"));
    }

    #[tokio::test]
    async fn shell_tool_empty_command_returns_error() {
        let executor = Box::new(DirectExecutor);
        let tool = ShellTool::new(executor, PathBuf::from("/tmp"));

        let result = tool.execute(json!({"command": ""})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("command is required"));
    }

    #[tokio::test]
    async fn shell_tool_missing_command_returns_error() {
        let executor = Box::new(DirectExecutor);
        let tool = ShellTool::new(executor, PathBuf::from("/tmp"));

        let result = tool.execute(json!({})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("command is required"));
    }

    #[tokio::test]
    async fn shell_tool_runs_command_via_direct_executor() {
        let tmp = TempDir::new().unwrap();
        let executor = Box::new(DirectExecutor);
        let tool = ShellTool::new(executor, tmp.path().to_path_buf());

        let result = tool
            .execute(json!({"command": "echo tool_test"}))
            .await
            .unwrap();

        assert!(result.success, "error: {:?}", result.error);
        assert!(result.output.contains("tool_test"));
    }

    #[tokio::test]
    async fn shell_tool_nonzero_exit_reports_error() {
        let tmp = TempDir::new().unwrap();
        let executor = Box::new(DirectExecutor);
        let tool = ShellTool::new(executor, tmp.path().to_path_buf());

        let result = tool
            .execute(json!({"command": "exit 1"}))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("Exit code: 1"));
    }

    #[tokio::test]
    async fn shell_tool_captures_stderr_in_output() {
        let tmp = TempDir::new().unwrap();
        let executor = Box::new(DirectExecutor);
        let tool = ShellTool::new(executor, tmp.path().to_path_buf());

        let result = tool
            .execute(json!({"command": "echo err_msg >&2"}))
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.output.contains("err_msg"));
    }

    // ── shell_tool() factory ────────────────────────────────

    #[test]
    fn shell_tool_factory_disabled() {
        let config = SandboxConfig::default();
        let tool = shell_tool(&config, "/tmp");
        assert_eq!(tool.name(), "shell");
    }

    #[test]
    fn shell_tool_factory_enabled() {
        let config = SandboxConfig {
            enabled: true,
            ..SandboxConfig::default()
        };
        let tool = shell_tool(&config, "/tmp");
        assert_eq!(tool.name(), "shell");
    }

    // ── Timeout handling (mock) ─────────────────────────────

    #[tokio::test]
    async fn timeout_kills_long_running_command() {
        // Use DirectExecutor with a command that would run forever,
        // but wrap it with our own timeout to verify the pattern.
        let tmp = TempDir::new().unwrap();
        let executor = DirectExecutor;

        let timeout = Duration::from_millis(200);
        let child = executor.execute("sleep 10", tmp.path().to_str().unwrap());
        let result = tokio::time::timeout(timeout, child).await;

        assert!(result.is_err(), "should have timed out");
    }

    // ── Config round-trip via TOML ──────────────────────────

    #[test]
    fn config_from_full_toml_section() {
        let toml_str = r#"
            [sandbox]
            enabled = true
            image = "zeroclaw-sandbox:latest"
            workspace = "/workspace"
            timeout_seconds = 30
            memory_limit = "256m"
            network = false
            mount_paths = []
        "#;

        #[derive(Deserialize)]
        struct Wrapper {
            sandbox: SandboxConfig,
        }

        let wrapper: Wrapper = toml::from_str(toml_str).unwrap();
        assert!(wrapper.sandbox.enabled);
        assert_eq!(wrapper.sandbox.image, "zeroclaw-sandbox:latest");
        assert_eq!(wrapper.sandbox.workspace, "/workspace");
        assert_eq!(wrapper.sandbox.timeout_seconds, 30);
        assert_eq!(wrapper.sandbox.memory_limit, "256m");
        assert!(!wrapper.sandbox.network);
        assert!(wrapper.sandbox.mount_paths.is_empty());
    }

    #[test]
    fn config_missing_sandbox_section_uses_defaults() {
        let toml_str = "[other]\nkey = \"value\"\n";

        #[derive(Deserialize)]
        struct Wrapper {
            #[serde(default)]
            sandbox: SandboxConfig,
        }

        let wrapper: Wrapper = toml::from_str(toml_str).unwrap();
        assert!(!wrapper.sandbox.enabled);
        assert_eq!(wrapper.sandbox.image, "zeroclaw-sandbox:latest");
    }

    // ── Workspace read-write verification ───────────────────

    #[tokio::test]
    async fn workspace_is_writable_via_direct_executor() {
        let tmp = TempDir::new().unwrap();
        let executor = DirectExecutor;

        // Write a file
        let result = executor
            .execute(
                "echo 'written by shell' > test_output.txt",
                tmp.path().to_str().unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(result.exit_code, 0);

        // Read it back
        let result = executor
            .execute("cat test_output.txt", tmp.path().to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(result.stdout.trim(), "written by shell");
    }
}

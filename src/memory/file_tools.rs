//! File editing tools for the ZeroClaw agent.
//!
//! Provides surgical file editing via `str_replace`, which replaces a single
//! unique occurrence of a string in a file rather than rewriting the entire
//! file contents.

use crate::tools::{Tool, ToolExecutionResult};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::path::Path;
use tokio::fs;

/// Tool for surgical string replacement in files.
///
/// Finds exactly one occurrence of `old_str` in the file at `path` and
/// replaces it with `new_str`. If `old_str` appears zero times or more
/// than once the tool returns an error so the caller can add context to
/// disambiguate.
pub struct StrReplaceTool;

impl StrReplaceTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for StrReplaceTool {
    fn name(&self) -> &str {
        "str_replace"
    }

    fn description(&self) -> &str {
        "Replace a single unique occurrence of a string in a file. \
         Prefer this over file_write for editing existing files. \
         Use file_write only for creating new files or complete rewrites. \
         The old_str must appear exactly once in the file — if it appears \
         zero times or more than once an error is returned."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to edit."
                },
                "old_str": {
                    "type": "string",
                    "description": "The exact string to find in the file. Must appear exactly once."
                },
                "new_str": {
                    "type": "string",
                    "description": "The replacement string. Omit or pass empty string to delete the match.",
                    "default": ""
                }
            },
            "required": ["path", "old_str"]
        })
    }

    async fn execute(&self, args: Value) -> anyhow::Result<ToolExecutionResult> {
        let path_str = args["path"].as_str().unwrap_or("").to_string();
        let old_str = args["old_str"].as_str().unwrap_or("").to_string();
        let new_str = args["new_str"].as_str().unwrap_or("").to_string();

        // --- Validate inputs ---

        if path_str.is_empty() {
            return Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some("path is required.".to_string()),
            });
        }

        if old_str.is_empty() {
            return Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some("old_str is required and cannot be empty.".to_string()),
            });
        }

        let path = Path::new(&path_str);

        if !path.exists() {
            return Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some(format!("File not found: {path_str}")),
            });
        }

        if !path.is_file() {
            return Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some(format!("Path is not a file: {path_str}")),
            });
        }

        // --- Read file ---

        let contents = match fs::read_to_string(path).await {
            Ok(c) => c,
            Err(e) => {
                return Ok(ToolExecutionResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("Failed to read file: {e}")),
                });
            }
        };

        // --- Count occurrences ---

        let match_count = contents.matches(&old_str).count();

        if match_count == 0 {
            return Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some("No match found for the provided old_str".to_string()),
            });
        }

        if match_count > 1 {
            return Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Found {match_count} matches for old_str — it must be unique. \
                     Add more surrounding context to disambiguate."
                )),
            });
        }

        // --- Replace and write ---

        let new_contents = contents.replacen(&old_str, &new_str, 1);

        if let Err(e) = fs::write(path, &new_contents).await {
            return Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to write file: {e}")),
            });
        }

        // --- Build context output ---

        let output = build_context_output(&new_contents, &new_str, &old_str, &path_str);

        Ok(ToolExecutionResult {
            success: true,
            output,
            error: None,
        })
    }
}

/// Build a success message showing a few lines of context around the edit.
///
/// Shows 3 lines before and 3 lines after the replaced region so the caller
/// can verify the edit was applied correctly.
fn build_context_output(
    new_contents: &str,
    new_str: &str,
    old_str: &str,
    path: &str,
) -> String {
    let lines: Vec<&str> = new_contents.lines().collect();

    // Find the line range that was affected by the replacement.
    // We locate where new_str starts in the new contents.
    let (start_line, end_line) = if new_str.is_empty() {
        // Deletion: find the position where old_str was removed.
        // The edit point is right at the boundary. We use the byte
        // offset of the first difference from the original.
        // Since old_str was unique and we used replacen, we can find
        // the offset by searching for where old_str *would* have been.
        let offset = new_contents.len(); // fallback
        let before = new_contents
            .find(old_str)
            .unwrap_or(offset.min(new_contents.len()));
        // Actually for deletion, old_str is gone. Use byte offset math:
        // the replacement happened at the position of old_str in the
        // original file. In the new file that position still exists.
        let orig_with_old = format!("{}{}", &new_contents[..before.min(new_contents.len())], old_str);
        let _ = orig_with_old; // just for reasoning
        // Count newlines up to `before` offset to find the line number.
        let edit_line = new_contents[..before.min(new_contents.len())]
            .matches('\n')
            .count();
        (edit_line, edit_line)
    } else {
        // Find the new_str in new_contents to locate the edit.
        match new_contents.find(new_str) {
            Some(byte_offset) => {
                let start = new_contents[..byte_offset].matches('\n').count();
                let end = start + new_str.matches('\n').count();
                (start, end)
            }
            None => (0, 0),
        }
    };

    let context_lines = 3;
    let from = start_line.saturating_sub(context_lines);
    let to = (end_line + context_lines + 1).min(lines.len());

    let mut output = format!("Successfully edited {path}\n\n");
    for (i, line) in lines[from..to].iter().enumerate() {
        let line_num = from + i + 1;
        output.push_str(&format!("{line_num:>4} | {line}\n"));
    }

    output
}

/// Create the str_replace file-editing tool.
pub fn file_edit_tool() -> Box<dyn Tool> {
    Box::new(StrReplaceTool::new())
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use tempfile::TempDir;

    /// Helper: create a temp directory and write a test file.
    async fn setup(content: &str) -> (TempDir, String) {
        let tmp = TempDir::new().unwrap();
        let file_path = tmp.path().join("test.txt");
        fs::write(&file_path, content).await.unwrap();
        (tmp, file_path.to_string_lossy().to_string())
    }

    // ── Successful single-match replacement ──────────────────

    #[tokio::test]
    async fn single_match_replacement() {
        let (_tmp, path) = setup("Hello, world!\nThis is a test.\nGoodbye.\n").await;
        let tool = StrReplaceTool::new();

        let result = tool
            .execute(json!({
                "path": path,
                "old_str": "This is a test.",
                "new_str": "This has been edited."
            }))
            .await
            .unwrap();

        assert!(result.success, "Expected success, got error: {:?}", result.error);
        assert!(result.output.contains("Successfully edited"));

        let content = fs::read_to_string(&path).await.unwrap();
        assert!(content.contains("This has been edited."));
        assert!(!content.contains("This is a test."));
    }

    // ── No match found → error ───────────────────────────────

    #[tokio::test]
    async fn no_match_returns_error() {
        let (_tmp, path) = setup("Hello, world!\n").await;
        let tool = StrReplaceTool::new();

        let result = tool
            .execute(json!({
                "path": path,
                "old_str": "nonexistent string",
                "new_str": "replacement"
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert_eq!(
            result.error.as_deref(),
            Some("No match found for the provided old_str")
        );
    }

    // ── Multiple matches found → error with count ────────────

    #[tokio::test]
    async fn multiple_matches_returns_error_with_count() {
        let (_tmp, path) = setup("foo bar foo baz foo\n").await;
        let tool = StrReplaceTool::new();

        let result = tool
            .execute(json!({
                "path": path,
                "old_str": "foo",
                "new_str": "qux"
            }))
            .await
            .unwrap();

        assert!(!result.success);
        let err = result.error.unwrap();
        assert!(err.contains("3 matches"), "Expected '3 matches' in: {err}");
        assert!(err.contains("must be unique"));
    }

    // ── Empty new_str deletes the match ──────────────────────

    #[tokio::test]
    async fn empty_new_str_deletes_match() {
        let (_tmp, path) = setup("Keep this. Remove this. Keep that.\n").await;
        let tool = StrReplaceTool::new();

        let result = tool
            .execute(json!({
                "path": path,
                "old_str": "Remove this. "
            }))
            .await
            .unwrap();

        assert!(result.success, "Expected success, got error: {:?}", result.error);

        let content = fs::read_to_string(&path).await.unwrap();
        assert_eq!(content, "Keep this. Keep that.\n");
    }

    #[tokio::test]
    async fn explicit_empty_new_str_deletes_match() {
        let (_tmp, path) = setup("alpha beta gamma\n").await;
        let tool = StrReplaceTool::new();

        let result = tool
            .execute(json!({
                "path": path,
                "old_str": " beta",
                "new_str": ""
            }))
            .await
            .unwrap();

        assert!(result.success);

        let content = fs::read_to_string(&path).await.unwrap();
        assert_eq!(content, "alpha gamma\n");
    }

    // ── Preserves file permissions ───────────────────────────

    #[tokio::test]
    async fn preserves_file_permissions() {
        let (_tmp, path) = setup("original content\n").await;

        // Set a non-default permission (e.g., 0o755)
        let perms = std::fs::Permissions::from_mode(0o755);
        std::fs::set_permissions(&path, perms).unwrap();

        let tool = StrReplaceTool::new();
        let result = tool
            .execute(json!({
                "path": path,
                "old_str": "original",
                "new_str": "modified"
            }))
            .await
            .unwrap();

        assert!(result.success);

        let metadata = std::fs::metadata(&path).unwrap();
        let mode = metadata.permissions().mode() & 0o777;
        assert_eq!(mode, 0o755, "File permissions should be preserved");
    }

    // ── Preserves encoding (UTF-8 with special chars) ────────

    #[tokio::test]
    async fn preserves_utf8_encoding() {
        let content = "こんにちは世界\nRust 🦀 is great\nüöä\n";
        let (_tmp, path) = setup(content).await;
        let tool = StrReplaceTool::new();

        let result = tool
            .execute(json!({
                "path": path,
                "old_str": "Rust 🦀 is great",
                "new_str": "Rust 🦀 is awesome"
            }))
            .await
            .unwrap();

        assert!(result.success);

        let new_content = fs::read_to_string(&path).await.unwrap();
        assert!(new_content.contains("こんにちは世界"));
        assert!(new_content.contains("Rust 🦀 is awesome"));
        assert!(new_content.contains("üöä"));
    }

    // ── Context lines shown in success output ────────────────

    #[tokio::test]
    async fn context_lines_in_output() {
        let content = "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\nline 8\nline 9\nline 10\n";
        let (_tmp, path) = setup(content).await;
        let tool = StrReplaceTool::new();

        let result = tool
            .execute(json!({
                "path": path,
                "old_str": "line 5",
                "new_str": "EDITED LINE"
            }))
            .await
            .unwrap();

        assert!(result.success);
        // Should show context around the edit (lines 2-8 roughly)
        assert!(result.output.contains("EDITED LINE"));
        // Should include line numbers
        assert!(result.output.contains(" | "));
    }

    // ── Path validation: file must exist ─────────────────────

    #[tokio::test]
    async fn file_not_found_error() {
        let tool = StrReplaceTool::new();

        let result = tool
            .execute(json!({
                "path": "/tmp/nonexistent_file_12345.txt",
                "old_str": "hello",
                "new_str": "world"
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("File not found"));
    }

    // ── Path validation: must be a file, not a directory ─────

    #[tokio::test]
    async fn directory_path_returns_error() {
        let tmp = TempDir::new().unwrap();
        let dir_path = tmp.path().to_string_lossy().to_string();
        let tool = StrReplaceTool::new();

        let result = tool
            .execute(json!({
                "path": dir_path,
                "old_str": "hello",
                "new_str": "world"
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("not a file"));
    }

    // ── Empty path returns error ─────────────────────────────

    #[tokio::test]
    async fn empty_path_returns_error() {
        let tool = StrReplaceTool::new();

        let result = tool
            .execute(json!({
                "path": "",
                "old_str": "hello",
                "new_str": "world"
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("path is required"));
    }

    // ── Empty old_str returns error ──────────────────────────

    #[tokio::test]
    async fn empty_old_str_returns_error() {
        let (_tmp, path) = setup("some content\n").await;
        let tool = StrReplaceTool::new();

        let result = tool
            .execute(json!({
                "path": path,
                "old_str": "",
                "new_str": "world"
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("old_str is required"));
    }

    // ── Multi-line replacement works ─────────────────────────

    #[tokio::test]
    async fn multi_line_replacement() {
        let content = "fn main() {\n    println!(\"hello\");\n}\n";
        let (_tmp, path) = setup(content).await;
        let tool = StrReplaceTool::new();

        let result = tool
            .execute(json!({
                "path": path,
                "old_str": "    println!(\"hello\");",
                "new_str": "    println!(\"goodbye\");\n    println!(\"world\");"
            }))
            .await
            .unwrap();

        assert!(result.success);

        let new_content = fs::read_to_string(&path).await.unwrap();
        assert!(new_content.contains("goodbye"));
        assert!(new_content.contains("world"));
        assert!(!new_content.contains("hello"));
    }

    // ── Tool metadata ────────────────────────────────────────

    #[test]
    fn tool_name_is_str_replace() {
        let tool = StrReplaceTool::new();
        assert_eq!(tool.name(), "str_replace");
    }

    #[test]
    fn tool_schema_has_required_fields() {
        let tool = StrReplaceTool::new();
        let schema = tool.parameters_schema();
        let required = schema["required"].as_array().unwrap();
        let req_names: Vec<&str> = required.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(req_names.contains(&"path"));
        assert!(req_names.contains(&"old_str"));
    }

    #[test]
    fn file_edit_tool_returns_str_replace() {
        let tool = file_edit_tool();
        assert_eq!(tool.name(), "str_replace");
    }
}

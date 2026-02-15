//! Startup daily summary for ZeroClaw.
//!
//! On the first session of a new day, if yesterday's daily note exists,
//! generates a silent turn that asks the agent to read yesterday's notes
//! and write a brief "morning summary" to today's daily note.
//!
//! This gives an e-ink display a fresh summary each morning without
//! requiring user action.
//!
//! ## How it works
//!
//! 1. At session start, `StartupManager::check_and_generate()` is called
//! 2. If yesterday's daily note exists and today's doesn't yet have a
//!    `## Morning Summary` section, a silent turn prompt is returned
//! 3. The agent loop injects this as a user message, the agent responds
//!    with `[NO_REPLY]` and writes the summary via `memory_store`
//! 4. The summary appears in today's daily note for the e-ink display

use crate::memory::markdown::MarkdownMemory;
use crate::memory::silent;
use anyhow::Result;
use chrono::Local;
use std::sync::Arc;
use tokio::fs;

// ── StartupManager ───────────────────────────────────────────────

/// Manages startup daily summary generation.
///
/// Detects the first session of a new day and creates a silent turn
/// prompt to generate a morning summary from yesterday's notes.
pub struct StartupManager {
    memory: Arc<MarkdownMemory>,
}

impl StartupManager {
    /// Create a new StartupManager.
    pub fn new(memory: Arc<MarkdownMemory>) -> Self {
        Self { memory }
    }

    /// Check whether a morning summary should be generated.
    ///
    /// Returns `true` if:
    /// 1. Yesterday's daily note exists (there's something to summarize)
    /// 2. Today's daily note either doesn't exist or doesn't already
    ///    contain a `## Morning Summary` section
    pub async fn should_generate_summary(&self) -> Result<bool> {
        let yesterday_path = self.memory.yesterday_note_path();
        if !yesterday_path.exists() {
            return Ok(false);
        }

        let today_path = self.memory.today_note_path();
        if today_path.exists() {
            let content = fs::read_to_string(&today_path).await?;
            if content.contains("## Morning Summary") {
                return Ok(false); // Already generated
            }
        }

        Ok(true)
    }

    /// Build a silent turn prompt for the morning summary.
    ///
    /// Reads yesterday's daily note and constructs a prompt that asks
    /// the agent to summarize it and write the summary to today's
    /// daily note.
    ///
    /// Returns `None` if no summary should be generated.
    pub async fn build_summary_prompt(&self) -> Result<Option<String>> {
        if !self.should_generate_summary().await? {
            return Ok(None);
        }

        let yesterday_path = self.memory.yesterday_note_path();
        let yesterday_content = fs::read_to_string(&yesterday_path).await?;

        let yesterday_date = (Local::now().date_naive() - chrono::Duration::days(1))
            .format("%Y-%m-%d")
            .to_string();

        let instruction = format!(
            "Read yesterday's notes ({yesterday_date}) and write a brief morning \
             summary to today's daily note using the memory_store tool with \
             category 'morning-summary'. The summary should include:\n\
             1. Key accomplishments from yesterday\n\
             2. Any pending tasks or blockers\n\
             3. Important context for today\n\n\
             Yesterday's notes:\n\
             ---\n\
             {yesterday_content}\n\
             ---\n\n\
             Format the summary as a concise markdown section:\n\
             ## Morning Summary\n\
             - [key point]\n\
             - [key point]\n\
             ...\n\n\
             Keep it to 3-5 bullet points. Be specific and actionable."
        );

        Ok(Some(silent::build_silent_prompt(&instruction)))
    }

    /// Check and generate a morning summary if needed.
    ///
    /// Main entry point called at session startup. Returns a silent
    /// turn prompt if a morning summary should be generated, or `None`
    /// if no summary is needed (no yesterday note, or summary already
    /// exists in today's note).
    pub async fn check_and_generate(&self) -> Result<Option<String>> {
        self.build_summary_prompt().await
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn setup() -> (TempDir, Arc<MarkdownMemory>, StartupManager) {
        let tmp = TempDir::new().unwrap();
        let mem = Arc::new(MarkdownMemory::new(tmp.path()));
        fs::create_dir_all(mem.daily_dir()).await.unwrap();
        let startup = StartupManager::new(mem.clone());
        (tmp, mem, startup)
    }

    // ── should_generate_summary tests ────────────────────────────

    #[tokio::test]
    async fn should_generate_returns_false_when_no_yesterday() {
        let (_tmp, _mem, startup) = setup().await;
        // No yesterday note exists
        assert!(!startup.should_generate_summary().await.unwrap());
    }

    #[tokio::test]
    async fn should_generate_returns_true_when_yesterday_exists() {
        let (_tmp, mem, startup) = setup().await;

        // Create yesterday's note
        let yesterday_path = mem.yesterday_note_path();
        fs::write(
            &yesterday_path,
            "# Daily Note — yesterday\n\n### 10:00:00\n\nWorked on feature X.\n",
        )
        .await
        .unwrap();

        assert!(startup.should_generate_summary().await.unwrap());
    }

    #[tokio::test]
    async fn should_generate_returns_false_when_summary_already_exists() {
        let (_tmp, mem, startup) = setup().await;

        // Create yesterday's note
        let yesterday_path = mem.yesterday_note_path();
        fs::write(&yesterday_path, "# Yesterday\n\nSome content.\n")
            .await
            .unwrap();

        // Create today's note with existing morning summary
        let today_path = mem.today_note_path();
        fs::write(
            &today_path,
            "# Daily Note\n\n## Morning Summary\n\n- Did stuff yesterday.\n",
        )
        .await
        .unwrap();

        assert!(!startup.should_generate_summary().await.unwrap());
    }

    #[tokio::test]
    async fn should_generate_returns_true_when_today_exists_without_summary() {
        let (_tmp, mem, startup) = setup().await;

        // Create yesterday's note
        let yesterday_path = mem.yesterday_note_path();
        fs::write(&yesterday_path, "# Yesterday\n\nSome content.\n")
            .await
            .unwrap();

        // Create today's note WITHOUT morning summary
        let today_path = mem.today_note_path();
        fs::write(
            &today_path,
            "# Daily Note\n\n### 10:00:00\n\nSome other content.\n",
        )
        .await
        .unwrap();

        assert!(startup.should_generate_summary().await.unwrap());
    }

    // ── build_summary_prompt tests ───────────────────────────────

    #[tokio::test]
    async fn build_prompt_returns_none_when_no_yesterday() {
        let (_tmp, _mem, startup) = setup().await;
        let result = startup.build_summary_prompt().await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn build_prompt_returns_silent_prompt_with_yesterday_content() {
        let (_tmp, mem, startup) = setup().await;

        let yesterday_content = "# Daily Note\n\n### 14:00:00\n\n\
                                  **[note]** Implemented heartbeat system.\n\n\
                                  ### 16:00:00\n\n\
                                  **[decision]** Use JSONL for session persistence.\n";
        let yesterday_path = mem.yesterday_note_path();
        fs::write(&yesterday_path, yesterday_content).await.unwrap();

        let result = startup.build_summary_prompt().await.unwrap();
        assert!(result.is_some());

        let prompt = result.unwrap();
        assert!(prompt.contains("[NO_REPLY]"));
        assert!(prompt.contains("heartbeat system"));
        assert!(prompt.contains("JSONL"));
        assert!(prompt.contains("Morning Summary"));
        assert!(prompt.contains("memory_store"));
    }

    #[tokio::test]
    async fn build_prompt_includes_yesterday_date() {
        let (_tmp, mem, startup) = setup().await;

        let yesterday_path = mem.yesterday_note_path();
        fs::write(&yesterday_path, "Some content").await.unwrap();

        let result = startup.build_summary_prompt().await.unwrap();
        let prompt = result.unwrap();

        let yesterday_date = (Local::now().date_naive() - chrono::Duration::days(1))
            .format("%Y-%m-%d")
            .to_string();
        assert!(prompt.contains(&yesterday_date));
    }

    // ── check_and_generate tests ─────────────────────────────────

    #[tokio::test]
    async fn check_and_generate_delegates_to_build_prompt() {
        let (_tmp, mem, startup) = setup().await;

        // No yesterday → None
        let result = startup.check_and_generate().await.unwrap();
        assert!(result.is_none());

        // With yesterday → Some
        let yesterday_path = mem.yesterday_note_path();
        fs::write(&yesterday_path, "Yesterday's notes").await.unwrap();

        let result = startup.check_and_generate().await.unwrap();
        assert!(result.is_some());
    }

    #[tokio::test]
    async fn check_and_generate_returns_none_after_summary_written() {
        let (_tmp, mem, startup) = setup().await;

        // Create yesterday's note
        let yesterday_path = mem.yesterday_note_path();
        fs::write(&yesterday_path, "Yesterday content").await.unwrap();

        // First call — should return Some
        let result = startup.check_and_generate().await.unwrap();
        assert!(result.is_some());

        // Simulate the agent writing the summary
        let today_path = mem.today_note_path();
        fs::write(
            &today_path,
            "# Daily Note\n\n## Morning Summary\n\n- Summary point.\n",
        )
        .await
        .unwrap();

        // Second call — should return None (already summarized)
        let result = startup.check_and_generate().await.unwrap();
        assert!(result.is_none());
    }
}

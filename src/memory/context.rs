//! Context window management and auto-compaction for ZeroClaw.
//!
//! Prevents context overflow by tracking token usage, pruning old tool
//! results, and auto-compacting the conversation when it approaches
//! the provider's context window limit.
//!
//! ## How it works
//!
//! 1. **Token counting** — After each turn, the running token total is
//!    updated using a simple heuristic (characters / 4).
//!
//! 2. **Session pruning** — Before each API call, tool results older
//!    than 10 turns are truncated if they exceed 500 tokens.
//!
//! 3. **Auto-compaction** — At 75% context capacity the manager:
//!    - Flushes important context to `memory/YYYY-MM-DD.md`
//!    - Asks the LLM to summarize the conversation
//!    - Replaces all but the last 5 turns with the summary
//!    - Persists the compacted state to the session JSONL file
//!    - Logs the compaction event to today's daily note
//!
//! 4. **Overflow recovery** — If the provider returns a context overflow
//!    error, an emergency compaction keeps only the last 3 turns and
//!    retries once.

use crate::memory::markdown::MarkdownMemory;
use crate::memory::session::{SessionManager, SessionTurn};
use anyhow::{Context as AnyhowContext, Result};
use chrono::Local;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

// ── Provider context window defaults ────────────────────────────

/// Default context window sizes (in tokens) for known providers.
pub const DEFAULT_ANTHROPIC_CONTEXT: usize = 200_000;
pub const DEFAULT_OPENAI_CONTEXT: usize = 128_000;
pub const DEFAULT_OLLAMA_CONTEXT: usize = 8_000;
pub const DEFAULT_OPENROUTER_CONTEXT: usize = 128_000;
pub const DEFAULT_COMPATIBLE_CONTEXT: usize = 32_000;

/// Ratio of context capacity at which auto-compaction triggers.
pub const COMPACTION_THRESHOLD: f64 = 0.75;

/// Number of recent turns to preserve during normal auto-compaction.
pub const COMPACT_KEEP_TURNS: usize = 5;

/// Number of recent turns to preserve during emergency compaction.
pub const EMERGENCY_KEEP_TURNS: usize = 3;

/// Tool results older than this many turns get pruned.
pub const PRUNE_AGE_TURNS: usize = 10;

/// Tool results larger than this (in tokens) get truncated during pruning.
pub const PRUNE_TOKEN_THRESHOLD: usize = 500;

/// How many leading characters of a pruned tool result to keep in the summary line.
pub const PRUNE_PREVIEW_CHARS: usize = 100;

// ── Token counting ──────────────────────────────────────────────

/// Approximate token count for a string using the chars/4 heuristic.
///
/// This is intentionally simple — it doesn't need to be exact because
/// we use it for threshold checks, not billing. The real provider will
/// reject if we actually exceed the limit (handled by overflow recovery).
pub fn estimate_tokens(text: &str) -> usize {
    // chars / 4, minimum 1 for non-empty strings
    let chars = text.len();
    if chars == 0 {
        0
    } else {
        (chars / 4).max(1)
    }
}

/// Estimate token count for a single session turn.
pub fn estimate_turn_tokens(turn: &SessionTurn) -> usize {
    let mut tokens = 0;

    // Role tag overhead (~4 tokens)
    tokens += 4;

    // Content
    if let Some(ref content) = turn.content {
        tokens += estimate_tokens(content);
    }

    // Tool calls
    for tc in &turn.tool_calls {
        tokens += estimate_tokens(&tc.name);
        tokens += estimate_tokens(&tc.arguments.to_string());
        tokens += 4; // structural overhead
    }

    // Tool results
    for tr in &turn.tool_results {
        tokens += estimate_tokens(&tr.content);
        tokens += 4; // structural overhead
    }

    tokens
}

// ── ContextConfig ───────────────────────────────────────────────

/// Provider-specific context window configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    /// Maximum context window size in tokens.
    pub max_tokens: usize,
    /// Ratio (0.0–1.0) at which auto-compaction triggers.
    pub compaction_threshold: f64,
    /// Number of recent turns to keep during normal compaction.
    pub compact_keep_turns: usize,
    /// Number of recent turns to keep during emergency compaction.
    pub emergency_keep_turns: usize,
    /// Tool results older than this many turns get pruned.
    pub prune_age_turns: usize,
    /// Token threshold above which old tool results get truncated.
    pub prune_token_threshold: usize,
}

impl ContextConfig {
    /// Create a config for a known provider name.
    ///
    /// Recognized providers (case-insensitive):
    /// - `anthropic` → 200k
    /// - `openai` → 128k
    /// - `ollama` → 8k
    /// - `openrouter` → 128k
    /// - `compatible` → 32k
    ///
    /// Unknown providers default to 32k.
    pub fn for_provider(provider: &str) -> Self {
        let max_tokens = match provider.to_lowercase().as_str() {
            "anthropic" => DEFAULT_ANTHROPIC_CONTEXT,
            "openai" => DEFAULT_OPENAI_CONTEXT,
            "ollama" => DEFAULT_OLLAMA_CONTEXT,
            "openrouter" => DEFAULT_OPENROUTER_CONTEXT,
            "compatible" => DEFAULT_COMPATIBLE_CONTEXT,
            _ => DEFAULT_COMPATIBLE_CONTEXT,
        };

        Self {
            max_tokens,
            compaction_threshold: COMPACTION_THRESHOLD,
            compact_keep_turns: COMPACT_KEEP_TURNS,
            emergency_keep_turns: EMERGENCY_KEEP_TURNS,
            prune_age_turns: PRUNE_AGE_TURNS,
            prune_token_threshold: PRUNE_TOKEN_THRESHOLD,
        }
    }

    /// Create a config with a custom context window size.
    pub fn with_max_tokens(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            compaction_threshold: COMPACTION_THRESHOLD,
            compact_keep_turns: COMPACT_KEEP_TURNS,
            emergency_keep_turns: EMERGENCY_KEEP_TURNS,
            prune_age_turns: PRUNE_AGE_TURNS,
            prune_token_threshold: PRUNE_TOKEN_THRESHOLD,
        }
    }

    /// The token count at which compaction should trigger.
    pub fn compaction_trigger(&self) -> usize {
        (self.max_tokens as f64 * self.compaction_threshold) as usize
    }
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self::for_provider("anthropic")
    }
}

// ── CompactionEvent ─────────────────────────────────────────────

/// Record of a compaction event, for logging and diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionEvent {
    /// ISO-8601 timestamp of the compaction.
    pub timestamp: String,
    /// Whether this was an emergency compaction.
    pub emergency: bool,
    /// Token count before compaction.
    pub tokens_before: usize,
    /// Token count after compaction.
    pub tokens_after: usize,
    /// Number of turns before compaction.
    pub turns_before: usize,
    /// Number of turns after compaction (summary + kept turns).
    pub turns_after: usize,
}

// ── ContextManager ──────────────────────────────────────────────

/// Manages context window usage, pruning, and compaction.
///
/// Sits between the agent loop and the provider, inspecting and
/// modifying the message history to stay within context limits.
pub struct ContextManager {
    /// Provider-specific context configuration.
    config: ContextConfig,
    /// Running token estimate for the current conversation.
    current_tokens: usize,
    /// Token estimate for the system prompt (constant per session).
    system_prompt_tokens: usize,
    /// History of compaction events this session.
    compaction_history: Vec<CompactionEvent>,
}

impl ContextManager {
    /// Create a new ContextManager with the given configuration.
    pub fn new(config: ContextConfig) -> Self {
        Self {
            config,
            current_tokens: 0,
            system_prompt_tokens: 0,
            compaction_history: Vec::new(),
        }
    }

    /// Create a ContextManager for a named provider.
    pub fn for_provider(provider: &str) -> Self {
        Self::new(ContextConfig::for_provider(provider))
    }

    /// The current configuration.
    pub fn config(&self) -> &ContextConfig {
        &self.config
    }

    /// Current estimated token usage.
    pub fn current_tokens(&self) -> usize {
        self.current_tokens
    }

    /// Set the system prompt token count (call once at session start).
    pub fn set_system_prompt_tokens(&mut self, tokens: usize) {
        self.system_prompt_tokens = tokens;
    }

    /// History of compaction events this session.
    pub fn compaction_history(&self) -> &[CompactionEvent] {
        &self.compaction_history
    }

    /// The percentage of context window currently in use.
    pub fn usage_ratio(&self) -> f64 {
        let total = self.system_prompt_tokens + self.current_tokens;
        total as f64 / self.config.max_tokens as f64
    }

    /// Whether auto-compaction should trigger based on current usage.
    pub fn should_compact(&self) -> bool {
        self.usage_ratio() >= self.config.compaction_threshold
    }

    // ── Token tracking ──────────────────────────────────────────

    /// Recount tokens from the full turn history.
    ///
    /// Call this after loading/resuming a session to sync the counter.
    pub fn recount_tokens(&mut self, turns: &[SessionTurn]) {
        self.current_tokens = turns.iter().map(estimate_turn_tokens).sum();
    }

    /// Update the running token count after appending a turn.
    pub fn track_turn(&mut self, turn: &SessionTurn) {
        self.current_tokens += estimate_turn_tokens(turn);
    }

    // ── Session pruning ─────────────────────────────────────────

    /// Prune old tool results in-place.
    ///
    /// For any `tool_result` turn older than `prune_age_turns` from the
    /// end of the history, if any individual tool result content exceeds
    /// `prune_token_threshold` tokens, replace it with a one-line summary.
    ///
    /// The tool call turn itself is kept intact — only the output is trimmed.
    ///
    /// Returns the number of tool results that were truncated.
    pub fn prune_old_tool_results(&mut self, turns: &mut Vec<SessionTurn>) -> usize {
        let total = turns.len();
        if total <= self.config.prune_age_turns {
            return 0;
        }

        let cutoff = total - self.config.prune_age_turns;
        let mut truncated_count = 0;

        for turn in turns[..cutoff].iter_mut() {
            if turn.role != "tool_result" {
                continue;
            }

            for tr in turn.tool_results.iter_mut() {
                let token_est = estimate_tokens(&tr.content);
                if token_est > self.config.prune_token_threshold {
                    let preview: String = tr.content.chars().take(PRUNE_PREVIEW_CHARS).collect();
                    let old_content = std::mem::replace(
                        &mut tr.content,
                        format!("[Tool result truncated: {preview}...]"),
                    );
                    // Update token count: subtract old, add new
                    let old_tokens = estimate_tokens(&old_content);
                    let new_tokens = estimate_tokens(&tr.content);
                    if old_tokens > new_tokens {
                        self.current_tokens = self.current_tokens.saturating_sub(old_tokens - new_tokens);
                    }
                    truncated_count += 1;
                }
            }
        }

        truncated_count
    }

    // ── Auto-compaction ─────────────────────────────────────────

    /// Build the system message instructing the agent to flush memory.
    ///
    /// This is injected as a system turn before compaction so the agent
    /// can save important context to `memory/YYYY-MM-DD.md`.
    pub fn build_memory_flush_instruction() -> String {
        "SYSTEM: Context window is approaching capacity. Before compaction, \
         save any important context, decisions, and current task state to \
         memory using the memory_store tool. Write to memory/YYYY-MM-DD.md. \
         Include: key decisions made, current task progress, important facts \
         discovered, and any pending work."
            .to_string()
    }

    /// Build the prompt that asks the LLM to summarize the conversation.
    ///
    /// This is sent as a one-shot request to get a concise summary that
    /// will replace the older portion of the conversation.
    pub fn build_compaction_prompt(turns: &[SessionTurn]) -> String {
        let mut conversation = String::new();
        for turn in turns {
            conversation.push_str(&format!("[{}] ", turn.role));
            if let Some(ref content) = turn.content {
                conversation.push_str(content);
            }
            for tc in &turn.tool_calls {
                conversation.push_str(&format!(" → tool_call: {}({})", tc.name, tc.arguments));
            }
            for tr in &turn.tool_results {
                let preview: String = tr.content.chars().take(200).collect();
                conversation.push_str(&format!(" → result: {preview}"));
                if tr.content.len() > 200 {
                    conversation.push_str("...");
                }
            }
            conversation.push('\n');
        }

        format!(
            "Summarize this conversation into a concise context summary \
             preserving key decisions, facts, and current task state. \
             Be thorough but brief — this summary will replace the original \
             messages in the context window.\n\n\
             CONVERSATION:\n{conversation}\n\n\
             SUMMARY:"
        )
    }

    /// Perform auto-compaction on the turn history.
    ///
    /// Replaces all turns older than the last `keep_turns` with a single
    /// system message containing the provided summary.
    ///
    /// Returns a `CompactionEvent` describing what happened.
    pub fn compact(
        &mut self,
        turns: &mut Vec<SessionTurn>,
        summary: &str,
        keep_turns: usize,
    ) -> CompactionEvent {
        let tokens_before = self.current_tokens;
        let turns_before = turns.len();

        // Calculate how many turns to remove
        let remove_count = if turns.len() > keep_turns {
            turns.len() - keep_turns
        } else {
            0
        };

        if remove_count == 0 {
            return CompactionEvent {
                timestamp: Local::now().to_rfc3339(),
                emergency: keep_turns <= EMERGENCY_KEEP_TURNS,
                tokens_before,
                tokens_after: tokens_before,
                turns_before,
                turns_after: turns.len(),
            };
        }

        // Remove old turns and prepend the summary
        let kept_turns: Vec<SessionTurn> = turns.drain(remove_count..).collect();

        turns.clear();

        // Insert summary as a system message
        let summary_turn = SessionTurn {
            role: "system".to_string(),
            content: Some(format!(
                "[Compacted conversation summary]\n\n{summary}"
            )),
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            timestamp: Local::now().to_rfc3339(),
        };

        turns.push(summary_turn);
        turns.extend(kept_turns);

        // Recount tokens
        self.recount_tokens(turns);

        let event = CompactionEvent {
            timestamp: Local::now().to_rfc3339(),
            emergency: keep_turns <= EMERGENCY_KEEP_TURNS,
            tokens_before,
            tokens_after: self.current_tokens,
            turns_before,
            turns_after: turns.len(),
        };

        self.compaction_history.push(event.clone());

        event
    }

    /// Perform emergency compaction — more aggressive, keeps only last 3 turns.
    ///
    /// Used when the provider returns a context overflow error.
    pub fn emergency_compact(
        &mut self,
        turns: &mut Vec<SessionTurn>,
        summary: &str,
    ) -> CompactionEvent {
        self.compact(turns, summary, self.config.emergency_keep_turns)
    }

    /// Build a compaction log entry for the daily note.
    pub fn format_compaction_log(event: &CompactionEvent) -> String {
        let kind = if event.emergency {
            "Emergency compaction"
        } else {
            "Auto-compaction"
        };

        format!(
            "**[compaction]** {kind}: {before_turns} turns ({before_tok} tokens) → \
             {after_turns} turns ({after_tok} tokens)",
            before_turns = event.turns_before,
            before_tok = event.tokens_before,
            after_turns = event.turns_after,
            after_tok = event.tokens_after,
        )
    }

    // ── Full pre-call pipeline ──────────────────────────────────

    /// Run the full pre-API-call context management pipeline.
    ///
    /// This is the main entry point called before each provider API call.
    /// It performs:
    /// 1. Session pruning (truncate old tool results)
    /// 2. Check if compaction is needed
    ///
    /// Returns `Some(CompactionNeeded)` if compaction should be triggered,
    /// or `None` if the context is within limits.
    pub fn pre_call_check(
        &mut self,
        turns: &mut Vec<SessionTurn>,
    ) -> PreCallResult {
        // Step 1: Prune old tool results
        let pruned = self.prune_old_tool_results(turns);

        // Step 2: Check if compaction is needed
        if self.should_compact() {
            PreCallResult::CompactionNeeded { pruned_results: pruned }
        } else {
            PreCallResult::Ok { pruned_results: pruned }
        }
    }

    // ── Session persistence helpers ─────────────────────────────

    /// Persist the compacted conversation state to a session file.
    ///
    /// Rewrites the session JSONL file with the compacted turns.
    /// This is a destructive operation — the original turns are lost
    /// from the file (but the summary preserves key information).
    pub async fn persist_compacted_session(
        turns: &[SessionTurn],
        session_mgr: &Arc<Mutex<SessionManager>>,
    ) -> Result<()> {
        let mgr = session_mgr.lock().await;
        let path = mgr
            .session_path()
            .context("No active session for compaction persistence")?
            .to_path_buf();

        // Rewrite the entire session file with compacted turns
        let mut content = String::new();
        for turn in turns {
            let line = serde_json::to_string(turn)
                .context("Failed to serialize compacted turn")?;
            content.push_str(&line);
            content.push('\n');
        }

        tokio::fs::write(&path, &content)
            .await
            .with_context(|| format!("Failed to write compacted session: {}", path.display()))?;

        Ok(())
    }

    /// Log a compaction event to today's daily note.
    pub async fn log_compaction_to_daily_note(
        event: &CompactionEvent,
        memory: &Arc<MarkdownMemory>,
    ) -> Result<()> {
        let log_entry = Self::format_compaction_log(event);
        memory.append_daily_note(&log_entry).await
    }

    // ── Overflow error detection ────────────────────────────────

    /// Check if an error message indicates a context overflow.
    ///
    /// Different providers return different error messages/codes for
    /// context overflow. This checks for common patterns.
    pub fn is_context_overflow_error(error_msg: &str) -> bool {
        let lower = error_msg.to_lowercase();
        lower.contains("context_length_exceeded")
            || lower.contains("maximum context length")
            || lower.contains("context window")
            || lower.contains("too many tokens")
            || lower.contains("max_tokens")
            || lower.contains("token limit")
            || lower.contains("context length")
            || lower.contains("request too large")
            || lower.contains("prompt is too long")
    }
}

// ── PreCallResult ───────────────────────────────────────────────

/// Result of the pre-API-call context management check.
#[derive(Debug)]
pub enum PreCallResult {
    /// Context is within limits. Proceed with the API call.
    Ok {
        /// Number of tool results that were pruned in this pass.
        pruned_results: usize,
    },
    /// Context is at or above the compaction threshold.
    /// The caller should trigger auto-compaction before the API call.
    CompactionNeeded {
        /// Number of tool results that were pruned in this pass.
        pruned_results: usize,
    },
}

impl PreCallResult {
    /// Whether compaction is needed.
    pub fn needs_compaction(&self) -> bool {
        matches!(self, PreCallResult::CompactionNeeded { .. })
    }

    /// Number of tool results that were pruned.
    pub fn pruned_count(&self) -> usize {
        match self {
            PreCallResult::Ok { pruned_results } => *pruned_results,
            PreCallResult::CompactionNeeded { pruned_results } => *pruned_results,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::session::{PersistedToolCall, PersistedToolResult, SessionTurn};
    use tempfile::TempDir;

    // ── Helper builders ─────────────────────────────────────────

    fn user_turn(content: &str) -> SessionTurn {
        SessionTurn {
            role: "user".to_string(),
            content: Some(content.to_string()),
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            timestamp: "2026-02-14T10:00:00Z".to_string(),
        }
    }

    fn assistant_turn(content: &str) -> SessionTurn {
        SessionTurn {
            role: "assistant".to_string(),
            content: Some(content.to_string()),
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            timestamp: "2026-02-14T10:00:01Z".to_string(),
        }
    }

    fn tool_call_turn(name: &str) -> SessionTurn {
        SessionTurn {
            role: "assistant".to_string(),
            content: None,
            tool_calls: vec![PersistedToolCall {
                id: "tc_1".to_string(),
                name: name.to_string(),
                arguments: serde_json::json!({"path": "src/main.rs"}),
            }],
            tool_results: Vec::new(),
            timestamp: "2026-02-14T10:00:02Z".to_string(),
        }
    }

    fn tool_result_turn(content: &str) -> SessionTurn {
        SessionTurn {
            role: "tool_result".to_string(),
            content: None,
            tool_calls: Vec::new(),
            tool_results: vec![PersistedToolResult {
                tool_call_id: "tc_1".to_string(),
                content: content.to_string(),
            }],
            timestamp: "2026-02-14T10:00:03Z".to_string(),
        }
    }

    fn large_tool_result(size_chars: usize) -> SessionTurn {
        let content = "x".repeat(size_chars);
        tool_result_turn(&content)
    }

    // ── Token counting tests ────────────────────────────────────

    #[test]
    fn estimate_tokens_basic() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("a"), 1); // min 1 for non-empty
        assert_eq!(estimate_tokens("hello world!"), 3); // 12 chars / 4
        assert_eq!(estimate_tokens("abcd"), 1); // 4 / 4
        assert_eq!(estimate_tokens("abcdefgh"), 2); // 8 / 4
    }

    #[test]
    fn estimate_tokens_longer_text() {
        let text = "a".repeat(1000);
        assert_eq!(estimate_tokens(&text), 250); // 1000 / 4
    }

    #[test]
    fn estimate_turn_tokens_user() {
        let turn = user_turn("Hello, how are you?");
        let tokens = estimate_turn_tokens(&turn);
        // 4 (overhead) + 19/4 (content) = 4 + 4 = 8
        assert!(tokens > 0);
        assert!(tokens < 20);
    }

    #[test]
    fn estimate_turn_tokens_with_tool_results() {
        let turn = large_tool_result(4000); // 4000 chars = ~1000 tokens
        let tokens = estimate_turn_tokens(&turn);
        assert!(tokens >= 1000);
    }

    // ── ContextConfig tests ─────────────────────────────────────

    #[test]
    fn config_for_known_providers() {
        let anthropic = ContextConfig::for_provider("anthropic");
        assert_eq!(anthropic.max_tokens, 200_000);

        let openai = ContextConfig::for_provider("openai");
        assert_eq!(openai.max_tokens, 128_000);

        let ollama = ContextConfig::for_provider("ollama");
        assert_eq!(ollama.max_tokens, 8_000);

        let openrouter = ContextConfig::for_provider("openrouter");
        assert_eq!(openrouter.max_tokens, 128_000);
    }

    #[test]
    fn config_for_unknown_provider_defaults() {
        let unknown = ContextConfig::for_provider("mystery_llm");
        assert_eq!(unknown.max_tokens, DEFAULT_COMPATIBLE_CONTEXT);
    }

    #[test]
    fn config_case_insensitive() {
        let config = ContextConfig::for_provider("Anthropic");
        assert_eq!(config.max_tokens, 200_000);

        let config = ContextConfig::for_provider("OPENAI");
        assert_eq!(config.max_tokens, 128_000);
    }

    #[test]
    fn config_compaction_trigger() {
        let config = ContextConfig::for_provider("anthropic");
        assert_eq!(config.compaction_trigger(), 150_000); // 200k * 0.75
    }

    #[test]
    fn config_with_custom_max_tokens() {
        let config = ContextConfig::with_max_tokens(50_000);
        assert_eq!(config.max_tokens, 50_000);
        assert_eq!(config.compaction_trigger(), 37_500);
    }

    // ── ContextManager basic tests ──────────────────────────────

    #[test]
    fn manager_tracks_tokens() {
        let mut mgr = ContextManager::for_provider("anthropic");
        assert_eq!(mgr.current_tokens(), 0);

        let turn = user_turn("Hello world");
        mgr.track_turn(&turn);
        assert!(mgr.current_tokens() > 0);
    }

    #[test]
    fn manager_recount_tokens() {
        let mut mgr = ContextManager::for_provider("anthropic");

        let turns = vec![
            user_turn("Hello"),
            assistant_turn("Hi there, how can I help?"),
            user_turn("Write some code"),
        ];

        mgr.recount_tokens(&turns);
        assert!(mgr.current_tokens() > 0);

        // Recount should give same result
        let tokens_first = mgr.current_tokens();
        mgr.recount_tokens(&turns);
        assert_eq!(mgr.current_tokens(), tokens_first);
    }

    #[test]
    fn manager_usage_ratio() {
        let config = ContextConfig::with_max_tokens(1000);
        let mut mgr = ContextManager::new(config);
        mgr.set_system_prompt_tokens(100);

        // Add enough tokens to reach ~50%
        let big_turn = user_turn(&"x".repeat(1600)); // ~400 tokens
        mgr.track_turn(&big_turn);

        let ratio = mgr.usage_ratio();
        assert!(ratio > 0.4);
        assert!(ratio < 0.6);
    }

    #[test]
    fn manager_should_compact_at_threshold() {
        let config = ContextConfig::with_max_tokens(100);
        let mut mgr = ContextManager::new(config);

        // Below threshold
        let small = user_turn("hi");
        mgr.track_turn(&small);
        assert!(!mgr.should_compact());

        // Push above 75%
        let big = user_turn(&"x".repeat(400)); // ~100 tokens, way over for 100-token window
        mgr.track_turn(&big);
        assert!(mgr.should_compact());
    }

    // ── Session pruning tests ───────────────────────────────────

    #[test]
    fn prune_skips_recent_tool_results() {
        let mut mgr = ContextManager::for_provider("anthropic");

        // Create 5 turns — all within prune_age_turns (10)
        let mut turns = vec![
            user_turn("hello"),
            tool_call_turn("file_read"),
            large_tool_result(4000), // big result but recent
            assistant_turn("done"),
            user_turn("thanks"),
        ];

        mgr.recount_tokens(&turns);
        let truncated = mgr.prune_old_tool_results(&mut turns);

        assert_eq!(truncated, 0);
        // Tool result should be unchanged
        assert!(!turns[2].tool_results[0].content.starts_with("[Tool result truncated:"));
    }

    #[test]
    fn prune_truncates_old_large_tool_results() {
        let mut mgr = ContextManager::for_provider("anthropic");

        // Create turns: 1 old tool result + 11 more turns to push it past the age limit
        let mut turns = vec![
            user_turn("read file"),
            tool_call_turn("file_read"),
            large_tool_result(4000), // old, large result
        ];

        // Add 11 more turns to push the tool result past prune_age_turns (10)
        for i in 0..11 {
            turns.push(user_turn(&format!("message {i}")));
        }

        mgr.recount_tokens(&turns);
        let tokens_before = mgr.current_tokens();
        let truncated = mgr.prune_old_tool_results(&mut turns);

        assert_eq!(truncated, 1);
        assert!(turns[2].tool_results[0].content.starts_with("[Tool result truncated:"));
        assert!(turns[2].tool_results[0].content.ends_with("...]"));
        assert!(mgr.current_tokens() < tokens_before);
    }

    #[test]
    fn prune_preserves_small_tool_results() {
        let mut mgr = ContextManager::for_provider("anthropic");

        // Small tool result that's old but under threshold
        let mut turns = vec![
            user_turn("check"),
            tool_call_turn("status"),
            tool_result_turn("OK"), // small result
        ];

        for i in 0..11 {
            turns.push(user_turn(&format!("msg {i}")));
        }

        mgr.recount_tokens(&turns);
        let truncated = mgr.prune_old_tool_results(&mut turns);

        assert_eq!(truncated, 0);
        assert_eq!(turns[2].tool_results[0].content, "OK");
    }

    #[test]
    fn prune_keeps_tool_calls_intact() {
        let mut mgr = ContextManager::for_provider("anthropic");

        let mut turns = vec![
            user_turn("read"),
            tool_call_turn("file_read"),
            large_tool_result(4000),
        ];

        for i in 0..11 {
            turns.push(user_turn(&format!("msg {i}")));
        }

        mgr.recount_tokens(&turns);
        mgr.prune_old_tool_results(&mut turns);

        // Tool call turn should be completely unchanged
        assert_eq!(turns[1].tool_calls.len(), 1);
        assert_eq!(turns[1].tool_calls[0].name, "file_read");
    }

    // ── Compaction tests ────────────────────────────────────────

    #[test]
    fn compact_replaces_old_turns_with_summary() {
        let mut mgr = ContextManager::for_provider("anthropic");

        let mut turns = vec![
            user_turn("first message"),
            assistant_turn("first response"),
            user_turn("second message"),
            assistant_turn("second response"),
            user_turn("third message"),
            assistant_turn("third response"),
            user_turn("fourth message"),
            assistant_turn("fourth response"),
        ];

        mgr.recount_tokens(&turns);
        let event = mgr.compact(&mut turns, "Summary of earlier conversation.", 3);

        // Should have: 1 summary + 3 kept turns = 4 total
        assert_eq!(turns.len(), 4);
        assert_eq!(turns[0].role, "system");
        assert!(turns[0].content.as_ref().unwrap().contains("Summary of earlier conversation"));
        assert_eq!(event.turns_before, 8);
        assert_eq!(event.turns_after, 4);
        assert!(event.tokens_after < event.tokens_before);
    }

    #[test]
    fn compact_with_fewer_turns_than_keep_is_noop() {
        let mut mgr = ContextManager::for_provider("anthropic");

        let mut turns = vec![
            user_turn("only message"),
            assistant_turn("only response"),
        ];

        mgr.recount_tokens(&turns);
        let event = mgr.compact(&mut turns, "Summary", 5);

        assert_eq!(turns.len(), 2); // unchanged
        assert_eq!(event.turns_before, 2);
        assert_eq!(event.turns_after, 2);
    }

    #[test]
    fn emergency_compact_keeps_only_3_turns() {
        let mut mgr = ContextManager::for_provider("anthropic");

        let mut turns: Vec<SessionTurn> = (0..20)
            .map(|i| user_turn(&format!("message {i}")))
            .collect();

        mgr.recount_tokens(&turns);
        let event = mgr.emergency_compact(&mut turns, "Emergency summary");

        // 1 summary + 3 kept = 4
        assert_eq!(turns.len(), 4);
        assert!(event.emergency);
        assert_eq!(event.turns_before, 20);
    }

    #[test]
    fn compact_records_event_in_history() {
        let mut mgr = ContextManager::for_provider("anthropic");
        assert!(mgr.compaction_history().is_empty());

        let mut turns: Vec<SessionTurn> = (0..10)
            .map(|i| user_turn(&format!("msg {i}")))
            .collect();

        mgr.recount_tokens(&turns);
        mgr.compact(&mut turns, "Summary", 3);

        assert_eq!(mgr.compaction_history().len(), 1);
        assert!(mgr.compaction_history()[0].emergency);
    }

    // ── Compaction prompt & log tests ───────────────────────────

    #[test]
    fn build_compaction_prompt_includes_conversation() {
        let turns = vec![
            user_turn("What is Rust?"),
            assistant_turn("Rust is a systems programming language."),
        ];

        let prompt = ContextManager::build_compaction_prompt(&turns);
        assert!(prompt.contains("What is Rust?"));
        assert!(prompt.contains("systems programming language"));
        assert!(prompt.contains("Summarize this conversation"));
        assert!(prompt.contains("SUMMARY:"));
    }

    #[test]
    fn build_memory_flush_instruction_is_clear() {
        let instruction = ContextManager::build_memory_flush_instruction();
        assert!(instruction.contains("Context window"));
        assert!(instruction.contains("memory_store"));
        assert!(instruction.contains("memory/YYYY-MM-DD.md"));
    }

    #[test]
    fn format_compaction_log_normal() {
        let event = CompactionEvent {
            timestamp: "2026-02-14T10:00:00Z".to_string(),
            emergency: false,
            tokens_before: 150_000,
            tokens_after: 30_000,
            turns_before: 100,
            turns_after: 6,
        };

        let log = ContextManager::format_compaction_log(&event);
        assert!(log.contains("Auto-compaction"));
        assert!(log.contains("150000"));
        assert!(log.contains("30000"));
        assert!(log.contains("100 turns"));
        assert!(log.contains("6 turns"));
    }

    #[test]
    fn format_compaction_log_emergency() {
        let event = CompactionEvent {
            timestamp: "2026-02-14T10:00:00Z".to_string(),
            emergency: true,
            tokens_before: 200_000,
            tokens_after: 5_000,
            turns_before: 150,
            turns_after: 4,
        };

        let log = ContextManager::format_compaction_log(&event);
        assert!(log.contains("Emergency compaction"));
    }

    // ── Pre-call pipeline tests ─────────────────────────────────

    #[test]
    fn pre_call_check_ok_when_under_threshold() {
        let config = ContextConfig::with_max_tokens(100_000);
        let mut mgr = ContextManager::new(config);

        let mut turns = vec![
            user_turn("hello"),
            assistant_turn("hi"),
        ];

        mgr.recount_tokens(&turns);
        let result = mgr.pre_call_check(&mut turns);

        assert!(!result.needs_compaction());
        assert_eq!(result.pruned_count(), 0);
    }

    #[test]
    fn pre_call_check_signals_compaction_when_over_threshold() {
        let config = ContextConfig::with_max_tokens(100); // tiny window
        let mut mgr = ContextManager::new(config);

        let mut turns = vec![
            user_turn(&"x".repeat(400)), // ~100 tokens, fills the window
        ];

        mgr.recount_tokens(&turns);
        let result = mgr.pre_call_check(&mut turns);

        assert!(result.needs_compaction());
    }

    #[test]
    fn pre_call_check_prunes_and_checks() {
        let config = ContextConfig::with_max_tokens(1_000_000); // large window
        let mut mgr = ContextManager::new(config);

        // Old large tool result + enough turns to trigger pruning
        let mut turns = vec![
            user_turn("read file"),
            tool_call_turn("file_read"),
            large_tool_result(4000),
        ];
        for i in 0..11 {
            turns.push(user_turn(&format!("msg {i}")));
        }

        mgr.recount_tokens(&turns);
        let result = mgr.pre_call_check(&mut turns);

        assert!(!result.needs_compaction());
        assert_eq!(result.pruned_count(), 1);
    }

    // ── Overflow detection tests ────────────────────────────────

    #[test]
    fn detects_context_overflow_errors() {
        assert!(ContextManager::is_context_overflow_error(
            "context_length_exceeded: max 200000 tokens"
        ));
        assert!(ContextManager::is_context_overflow_error(
            "This model's maximum context length is 128000 tokens"
        ));
        assert!(ContextManager::is_context_overflow_error(
            "Error: too many tokens in request"
        ));
        assert!(ContextManager::is_context_overflow_error(
            "request too large for model"
        ));
        assert!(ContextManager::is_context_overflow_error(
            "The prompt is too long. Please reduce."
        ));
    }

    #[test]
    fn does_not_detect_unrelated_errors() {
        assert!(!ContextManager::is_context_overflow_error("rate limit exceeded"));
        assert!(!ContextManager::is_context_overflow_error("invalid api key"));
        assert!(!ContextManager::is_context_overflow_error("internal server error"));
        assert!(!ContextManager::is_context_overflow_error("network timeout"));
    }

    // ── Session persistence tests ───────────────────────────────

    #[tokio::test]
    async fn persist_compacted_session_writes_file() {
        let tmp = TempDir::new().unwrap();
        let session_mgr = Arc::new(Mutex::new(SessionManager::new(tmp.path())));

        // Create a session
        {
            let mut mgr = session_mgr.lock().await;
            mgr.new_session().await.unwrap();
            let turn = SessionManager::user_turn("original message");
            mgr.append_turn(&turn).await.unwrap();
        }

        // Compact and persist
        let compacted_turns = vec![
            SessionTurn {
                role: "system".to_string(),
                content: Some("[Compacted summary]".to_string()),
                tool_calls: Vec::new(),
                tool_results: Vec::new(),
                timestamp: "2026-02-14T10:00:00Z".to_string(),
            },
            user_turn("latest message"),
        ];

        ContextManager::persist_compacted_session(&compacted_turns, &session_mgr)
            .await
            .unwrap();

        // Verify the file was rewritten
        let mgr = session_mgr.lock().await;
        let path = mgr.session_path().unwrap();
        let content = tokio::fs::read_to_string(path).await.unwrap();
        let lines: Vec<&str> = content.lines().filter(|l| !l.is_empty()).collect();

        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("Compacted summary"));
        assert!(lines[1].contains("latest message"));
    }

    #[tokio::test]
    async fn log_compaction_to_daily_note_appends() {
        let tmp = TempDir::new().unwrap();
        let memory = Arc::new(MarkdownMemory::new(tmp.path()));
        tokio::fs::create_dir_all(memory.daily_dir()).await.unwrap();

        let event = CompactionEvent {
            timestamp: "2026-02-14T10:00:00Z".to_string(),
            emergency: false,
            tokens_before: 150_000,
            tokens_after: 30_000,
            turns_before: 100,
            turns_after: 6,
        };

        ContextManager::log_compaction_to_daily_note(&event, &memory)
            .await
            .unwrap();

        let content = tokio::fs::read_to_string(memory.today_note_path())
            .await
            .unwrap();
        assert!(content.contains("[compaction]"));
        assert!(content.contains("Auto-compaction"));
        assert!(content.contains("150000"));
    }

    // ── Integration-style tests ─────────────────────────────────

    #[test]
    fn full_lifecycle_prune_then_compact() {
        let config = ContextConfig::with_max_tokens(500);
        let mut mgr = ContextManager::new(config);

        // Build up a conversation with old tool results
        let mut turns: Vec<SessionTurn> = Vec::new();

        // Old tool interactions
        turns.push(user_turn("read the big file"));
        turns.push(tool_call_turn("file_read"));
        turns.push(large_tool_result(4000));
        turns.push(assistant_turn("I see the file contents."));

        // More conversation
        for i in 0..12 {
            turns.push(user_turn(&format!("question {i}")));
            turns.push(assistant_turn(&format!("answer {i}")));
        }

        mgr.recount_tokens(&turns);

        // Step 1: Pre-call check — should prune and signal compaction
        let result = mgr.pre_call_check(&mut turns);
        assert_eq!(result.pruned_count(), 1); // the old large tool result
        assert!(!result.needs_compaction()); // pruning reduced tokens below threshold

        // Step 2: Compact
        let event = mgr.compact(&mut turns, "Summary of file reading and Q&A.", 5);

        assert!(event.turns_after <= 6); // summary + 5 kept
        assert!(mgr.current_tokens() < 500);
        assert_eq!(mgr.compaction_history().len(), 1);
    }

    #[test]
    fn emergency_compact_after_overflow() {
        let config = ContextConfig::with_max_tokens(200);
        let mut mgr = ContextManager::new(config);

        let mut turns: Vec<SessionTurn> = (0..15)
            .map(|i| user_turn(&format!("msg {i} with some content to fill tokens")))
            .collect();

        mgr.recount_tokens(&turns);

        // Simulate overflow recovery
        let event = mgr.emergency_compact(&mut turns, "Emergency: key state preserved.");

        assert!(event.emergency);
        assert_eq!(turns.len(), 4); // summary + 3
        assert_eq!(turns[0].role, "system");
    }
}

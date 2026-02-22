//! Persistent context window usage tracker for ZeroClaw.
//!
//! Tracks actual token usage as reported by LLM API responses and provides:
//!
//! - **Per-session metrics** persisted in session metadata (survives restarts)
//! - **Status line output** after every agent response (stderr with CTX_STATUS prefix)
//! - **Telegram footer** appended to messages when usage > 30%
//! - **`/context` command** for detailed breakdown on demand
//!
//! ## Design principles
//!
//! - **Zero new API calls** — piggybacks entirely on `usage.prompt_tokens`
//!   and `usage.completion_tokens` already returned by every LLM response
//! - **No added latency** — status computation is trivial arithmetic
//! - **Backward compatible** — sessions without ContextMeter data initialize
//!   on next turn (all fields default to zero)
//! - **Trust the API** — token counts are model-specific but API-reported
//!   counts are always accurate for the model being used

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Known model context windows ─────────────────────────────────

/// Default context window size when a model is not in the known table.
pub const DEFAULT_CONTEXT_WINDOW: u32 = 128_000;

/// Build the known models table with context window sizes.
///
/// These can be overridden via `[models]` in config.toml.
pub fn known_models() -> HashMap<String, u32> {
    let mut m = HashMap::new();

    // Anthropic (Claude)
    m.insert("claude-opus-4-6".to_string(), 200_000);
    m.insert("claude-opus-4-6-extended".to_string(), 1_000_000);
    m.insert("claude-sonnet-4-6".to_string(), 200_000);

    // OpenAI (GPT)
    m.insert("gpt-5.2".to_string(), 400_000);
    m.insert("gpt-5.2-chat".to_string(), 128_000);

    // Google (Gemini)
    m.insert("gemini-3-pro".to_string(), 1_000_000);
    m.insert("gemini-3-flash".to_string(), 1_000_000);

    m
}

/// Look up context window for a model, with optional config overrides.
///
/// Returns the context window size and whether a warning should be logged
/// (true when the model was not found in any table).
pub fn context_window_for_model(
    model: &str,
    config_overrides: Option<&HashMap<String, u32>>,
) -> (u32, bool) {
    // Check config overrides first
    if let Some(overrides) = config_overrides {
        if let Some(&size) = overrides.get(model) {
            return (size, false);
        }
    }

    // Check known models table
    let known = known_models();
    if let Some(&size) = known.get(model) {
        return (size, false);
    }

    // Default with warning
    (DEFAULT_CONTEXT_WINDOW, true)
}

// ── ContextMeter ────────────────────────────────────────────────

/// Persistent context window usage tracker.
///
/// Updated after every LLM response using API-reported token counts.
/// Persisted in session metadata so it survives process restarts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextMeter {
    /// Total context window size for the current model (tokens).
    pub model_context_window: u32,
    /// Token count for the system prompt (measured at session start,
    /// updated on memory/skill changes).
    pub system_prompt_tokens: u32,
    /// Conversation tokens (API prompt_tokens minus system_prompt_tokens).
    pub conversation_tokens: u32,
    /// Completion tokens from the last API response.
    pub last_completion_tokens: u32,
    /// Token count at which compaction triggers (context_window * compact_pct).
    pub compaction_threshold_tokens: u32,
    /// Current compaction depth (from session metadata).
    pub compaction_depth: u32,
    /// Number of completed turns in this session.
    pub turn_count: u32,
    /// Rolling average tokens per turn.
    pub avg_tokens_per_turn: u32,
}

impl Default for ContextMeter {
    fn default() -> Self {
        Self {
            model_context_window: DEFAULT_CONTEXT_WINDOW,
            system_prompt_tokens: 0,
            conversation_tokens: 0,
            last_completion_tokens: 0,
            compaction_threshold_tokens: (DEFAULT_CONTEXT_WINDOW as f64 * 0.75) as u32,
            compaction_depth: 0,
            turn_count: 0,
            avg_tokens_per_turn: 0,
        }
    }
}

impl ContextMeter {
    /// Create a new ContextMeter for a given model.
    ///
    /// `compact_pct` is the compaction threshold as a fraction (e.g. 0.75).
    pub fn new(model: &str, compact_pct: f64, config_overrides: Option<&HashMap<String, u32>>) -> (Self, bool) {
        let (context_window, warn) = context_window_for_model(model, config_overrides);
        let threshold = (context_window as f64 * compact_pct) as u32;

        let meter = Self {
            model_context_window: context_window,
            system_prompt_tokens: 0,
            conversation_tokens: 0,
            last_completion_tokens: 0,
            compaction_threshold_tokens: threshold,
            compaction_depth: 0,
            turn_count: 0,
            avg_tokens_per_turn: 0,
        };

        (meter, warn)
    }

    /// Set the system prompt token count.
    ///
    /// Called at session start (using the first API response's prompt_tokens
    /// as baseline when conversation history is minimal) and updated on
    /// memory reload, skill changes, or agent switches.
    pub fn set_system_prompt_tokens(&mut self, tokens: u32) {
        self.system_prompt_tokens = tokens;
    }

    /// Update the compaction depth.
    pub fn set_compaction_depth(&mut self, depth: u32) {
        self.compaction_depth = depth;
    }

    /// Update the meter from an LLM API response.
    ///
    /// `prompt_tokens` and `completion_tokens` come directly from the
    /// API response's `usage` object (all 5 providers return these).
    pub fn update_from_api_response(&mut self, prompt_tokens: u32, completion_tokens: u32) {
        // Update conversation tokens = prompt_tokens - system_prompt_tokens
        self.conversation_tokens = prompt_tokens.saturating_sub(self.system_prompt_tokens);
        self.last_completion_tokens = completion_tokens;

        // Update turn count and rolling average
        self.turn_count += 1;
        if self.turn_count == 1 {
            self.avg_tokens_per_turn = self.conversation_tokens;
        } else {
            // Exponential moving average with alpha = 2/(N+1), capped at N=20
            // for responsiveness. This weights recent turns more heavily.
            // Rolling average: avg = avg + (2/(n+1)) * (new_observation - avg)
            // Using integer math to avoid floating point.
            // Capped at N=20 for responsiveness (weights recent turns more).
            let n = self.turn_count.min(20) as i64;
            let current = self.avg_tokens_per_turn as i64;
            let observation = self.conversation_tokens as i64;
            let delta = (2 * (observation - current)) / (n + 1);
            self.avg_tokens_per_turn = (current + delta).max(1) as u32;
        }
    }

    // ── Computed metrics ────────────────────────────────────────

    /// Total tokens currently in use (system + conversation).
    pub fn total_tokens(&self) -> u32 {
        self.system_prompt_tokens + self.conversation_tokens
    }

    /// Usage percentage (0–100).
    pub fn usage_pct(&self) -> u32 {
        if self.model_context_window == 0 {
            return 0;
        }
        ((self.total_tokens() as u64 * 100) / self.model_context_window as u64) as u32
    }

    /// Usage ratio (0.0–1.0).
    pub fn usage_ratio(&self) -> f64 {
        if self.model_context_window == 0 {
            return 0.0;
        }
        self.total_tokens() as f64 / self.model_context_window as f64
    }

    /// Compaction threshold percentage (e.g. 75).
    pub fn compact_at_pct(&self) -> u32 {
        if self.model_context_window == 0 {
            return 75;
        }
        ((self.compaction_threshold_tokens as u64 * 100) / self.model_context_window as u64) as u32
    }

    /// Estimated turns remaining before compaction triggers.
    ///
    /// Returns 0 if already at or past the threshold, or if avg_tokens_per_turn is 0.
    pub fn estimated_turns_remaining(&self) -> u32 {
        if self.avg_tokens_per_turn == 0 {
            return 0;
        }

        let total = self.total_tokens();
        if total >= self.compaction_threshold_tokens {
            return 0;
        }

        let headroom = self.compaction_threshold_tokens - total;
        headroom / self.avg_tokens_per_turn
    }

    /// Remaining tokens before hitting the context window limit.
    pub fn headroom(&self) -> u32 {
        self.model_context_window.saturating_sub(self.total_tokens())
    }

    // ── Status line formatting ──────────────────────────────────

    /// Format the context window size for display (e.g. "128k", "200k", "1M").
    pub fn format_context_size(tokens: u32) -> String {
        if tokens >= 1_000_000 {
            if tokens % 1_000_000 == 0 {
                format!("{}M", tokens / 1_000_000)
            } else {
                format!("{:.1}M", tokens as f64 / 1_000_000.0)
            }
        } else {
            format!("{}k", tokens / 1_000)
        }
    }

    /// Emit the machine-parseable CTX_STATUS JSON line.
    ///
    /// Format: `CTX_STATUS:{"used":21267,"max":128000,"pct":17,...}`
    pub fn status_json(&self) -> String {
        format!(
            "CTX_STATUS:{{\"used\":{},\"max\":{},\"pct\":{},\"turns_left\":{},\"compact_at_pct\":{},\"depth\":{},\"system\":{},\"conversation\":{}}}",
            self.total_tokens(),
            self.model_context_window,
            self.usage_pct(),
            self.estimated_turns_remaining(),
            self.compact_at_pct(),
            self.compaction_depth,
            self.system_prompt_tokens,
            self.conversation_tokens,
        )
    }

    /// Emit the human-readable status line.
    ///
    /// Examples:
    /// - `[ctx 21,267/128k (17%) | ~84 turns left | compaction at 75% | depth 1]`
    /// - `[ctx 68,400/128k (53%) | ~22 turns left | ⚠ compaction at 75% | depth 1]`
    /// - `[ctx 92,100/128k (72%) | ~6 turns left | 🔴 compaction imminent | depth 2]`
    pub fn status_line(&self) -> String {
        let pct = self.usage_pct();
        let total = self.total_tokens();
        let max_display = Self::format_context_size(self.model_context_window);
        let turns_left = self.estimated_turns_remaining();

        let compaction_part = if pct >= 70 {
            format!("\u{1f534} compaction imminent")
        } else if pct >= 50 {
            format!("\u{26a0} compaction at {}%", self.compact_at_pct())
        } else {
            format!("compaction at {}%", self.compact_at_pct())
        };

        format!(
            "[ctx {}/{} ({}%) | ~{} turns left | {} | depth {}]",
            format_number(total),
            max_display,
            pct,
            turns_left,
            compaction_part,
            self.compaction_depth,
        )
    }

    /// Emit both status lines to stderr.
    ///
    /// Writes the machine-parseable JSON line first, then the human-readable line.
    pub fn emit_status_to_stderr(&self) {
        eprintln!("{}", self.status_json());
        eprintln!("{}", self.status_line());
    }

    // ── Telegram footer ─────────────────────────────────────────

    /// Generate a Telegram footer line, or None if usage is <= 30%.
    ///
    /// Format: `─── ctx 17% | ~84 turns ───`
    pub fn telegram_footer(&self) -> Option<String> {
        let pct = self.usage_pct();
        if pct <= 30 {
            return None;
        }

        let turns_left = self.estimated_turns_remaining();
        Some(format!(
            "\u{2500}\u{2500}\u{2500} ctx {}% | ~{} turns \u{2500}\u{2500}\u{2500}",
            pct, turns_left
        ))
    }

    // ── /context command output ─────────────────────────────────

    /// Format the detailed /context command output.
    ///
    /// `system_breakdown` is an optional list of (name, tokens) for system
    /// prompt components (SOUL.md, USER.md, MEMORY.md, Tools, etc.).
    /// `session_name` is the current session name for display.
    pub fn format_context_detail(
        &self,
        system_breakdown: Option<&[(String, u32)]>,
        session_name: Option<&str>,
    ) -> String {
        let mut out = String::new();

        // System prompt section
        out.push_str(&format!(
            "System prompt:    {} tokens\n",
            format_number(self.system_prompt_tokens),
        ));

        if let Some(breakdown) = system_breakdown {
            for (name, tokens) in breakdown {
                out.push_str(&format!(
                    "  {}:{}{} \n",
                    name,
                    " ".repeat(16usize.saturating_sub(name.len() + 1)),
                    format_number(*tokens),
                ));
            }
        }

        // Conversation section
        out.push_str(&format!(
            "Conversation:    {} tokens ({} turns)\n",
            format_number(self.conversation_tokens),
            self.turn_count,
        ));

        // Total
        let total = self.total_tokens();
        let pct = if self.model_context_window > 0 {
            (total as f64 / self.model_context_window as f64) * 100.0
        } else {
            0.0
        };
        out.push_str(&format!(
            "Total:           {} / {} ({:.1}%)\n",
            format_number(total),
            format_number(self.model_context_window),
            pct,
        ));

        // Headroom
        out.push_str(&format!(
            "Headroom:       {} tokens\n",
            format_number(self.headroom()),
        ));

        // Compaction threshold
        out.push_str(&format!(
            "Compaction at:   {} ({}%)\n",
            format_number(self.compaction_threshold_tokens),
            self.compact_at_pct(),
        ));

        // Estimated turns
        if self.avg_tokens_per_turn > 0 {
            out.push_str(&format!(
                "Est. turns left: ~{} (avg {} tokens/turn)\n",
                self.estimated_turns_remaining(),
                format_number(self.avg_tokens_per_turn),
            ));
        } else {
            out.push_str("Est. turns left: — (no data yet)\n");
        }

        // Compaction depth
        out.push_str(&format!(
            "Compaction depth: {}\n",
            self.compaction_depth,
        ));

        // Session name
        if let Some(name) = session_name {
            out.push_str(&format!("Session:         {}\n", name));
        }

        out
    }
}

// ── Config parsing ──────────────────────────────────────────────

/// Parse `[models]` section from config.toml content.
///
/// Expected format:
/// ```toml
/// [models]
/// "claude-opus-4-6" = 200000
/// "gpt-5.2" = 400000
/// ```
///
/// Returns a map of model name -> context window size.
pub fn parse_models_config(toml_content: &str) -> Option<HashMap<String, u32>> {
    let table: toml::Value = toml_content.parse().ok()?;
    let models = table.get("models")?.as_table()?;

    let mut result = HashMap::new();
    for (key, value) in models {
        if let Some(size) = value.as_integer() {
            if size > 0 {
                result.insert(key.clone(), size as u32);
            }
        }
    }

    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

// ── Helpers ─────────────────────────────────────────────────────

/// Format a number with comma separators (e.g. 21267 -> "21,267").
pub fn format_number(n: u32) -> String {
    let s = n.to_string();
    let len = s.len();
    if len <= 3 {
        return s;
    }

    let mut result = String::with_capacity(len + len / 3);
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            result.push(',');
        }
        result.push(ch);
    }
    result
}

// ── /context command parsing ────────────────────────────────────

/// Check if user input is the /context command.
pub fn is_context_command(input: &str) -> bool {
    input.trim().eq_ignore_ascii_case("/context")
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ContextMeter construction ──────────────────────────────

    #[test]
    fn new_meter_for_known_model() {
        let (meter, warn) = ContextMeter::new("claude-opus-4-6", 0.75, None);
        assert_eq!(meter.model_context_window, 200_000);
        assert_eq!(meter.compaction_threshold_tokens, 150_000);
        assert!(!warn);
    }

    #[test]
    fn new_meter_for_unknown_model_warns() {
        let (meter, warn) = ContextMeter::new("mystery-llm-3", 0.75, None);
        assert_eq!(meter.model_context_window, DEFAULT_CONTEXT_WINDOW);
        assert!(warn);
    }

    #[test]
    fn new_meter_with_config_override() {
        let mut overrides = HashMap::new();
        overrides.insert("custom-model".to_string(), 500_000);

        let (meter, warn) = ContextMeter::new("custom-model", 0.75, Some(&overrides));
        assert_eq!(meter.model_context_window, 500_000);
        assert_eq!(meter.compaction_threshold_tokens, 375_000);
        assert!(!warn);
    }

    #[test]
    fn config_override_takes_precedence_over_known() {
        let mut overrides = HashMap::new();
        overrides.insert("claude-opus-4-6".to_string(), 300_000);

        let (meter, warn) = ContextMeter::new("claude-opus-4-6", 0.75, Some(&overrides));
        assert_eq!(meter.model_context_window, 300_000);
        assert!(!warn);
    }

    // ── Default meter ──────────────────────────────────────────

    #[test]
    fn default_meter_is_zeroed() {
        let meter = ContextMeter::default();
        assert_eq!(meter.model_context_window, DEFAULT_CONTEXT_WINDOW);
        assert_eq!(meter.system_prompt_tokens, 0);
        assert_eq!(meter.conversation_tokens, 0);
        assert_eq!(meter.last_completion_tokens, 0);
        assert_eq!(meter.turn_count, 0);
        assert_eq!(meter.avg_tokens_per_turn, 0);
        assert_eq!(meter.compaction_depth, 0);
    }

    // ── Update from API response ───────────────────────────────

    #[test]
    fn update_from_api_response_first_turn() {
        let mut meter = ContextMeter::default();
        meter.set_system_prompt_tokens(2_000);

        // Simulate: API reports 5000 prompt tokens total
        meter.update_from_api_response(5_000, 500);

        assert_eq!(meter.conversation_tokens, 3_000); // 5000 - 2000
        assert_eq!(meter.last_completion_tokens, 500);
        assert_eq!(meter.turn_count, 1);
        assert_eq!(meter.avg_tokens_per_turn, 3_000);
    }

    #[test]
    fn update_from_api_response_multiple_turns() {
        let mut meter = ContextMeter::default();
        meter.set_system_prompt_tokens(1_000);

        // Turn 1: 3000 prompt tokens
        meter.update_from_api_response(3_000, 200);
        assert_eq!(meter.conversation_tokens, 2_000);
        assert_eq!(meter.turn_count, 1);
        assert_eq!(meter.avg_tokens_per_turn, 2_000);

        // Turn 2: 6000 prompt tokens (conversation grew)
        meter.update_from_api_response(6_000, 300);
        assert_eq!(meter.conversation_tokens, 5_000);
        assert_eq!(meter.turn_count, 2);
        // Rolling average should move toward 5000
        assert!(meter.avg_tokens_per_turn > 2_000);
        assert!(meter.avg_tokens_per_turn <= 5_000);
    }

    #[test]
    fn update_saturates_when_system_exceeds_prompt() {
        let mut meter = ContextMeter::default();
        meter.set_system_prompt_tokens(10_000);

        // If prompt_tokens < system_prompt_tokens (shouldn't happen, but defensive)
        meter.update_from_api_response(5_000, 100);
        assert_eq!(meter.conversation_tokens, 0); // saturating_sub
    }

    // ── Percentage calculation ──────────────────────────────────

    #[test]
    fn usage_pct_calculation() {
        let mut meter = ContextMeter::default();
        meter.model_context_window = 100_000;
        meter.system_prompt_tokens = 5_000;
        meter.conversation_tokens = 12_000;

        // Total = 17000, pct = 17%
        assert_eq!(meter.usage_pct(), 17);
        assert_eq!(meter.total_tokens(), 17_000);
    }

    #[test]
    fn usage_pct_zero_window() {
        let mut meter = ContextMeter::default();
        meter.model_context_window = 0;
        assert_eq!(meter.usage_pct(), 0);
    }

    #[test]
    fn usage_ratio_calculation() {
        let mut meter = ContextMeter::default();
        meter.model_context_window = 200_000;
        meter.system_prompt_tokens = 2_000;
        meter.conversation_tokens = 98_000;

        let ratio = meter.usage_ratio();
        assert!((ratio - 0.5).abs() < 0.001);
    }

    // ── Turns remaining estimation ─────────────────────────────

    #[test]
    fn estimated_turns_remaining_basic() {
        let mut meter = ContextMeter::default();
        meter.model_context_window = 128_000;
        meter.compaction_threshold_tokens = 96_000; // 75%
        meter.system_prompt_tokens = 2_847;
        meter.conversation_tokens = 18_420;
        meter.avg_tokens_per_turn = 1_268;

        // headroom = 96000 - (2847 + 18420) = 74733
        // turns = 74733 / 1268 = 58
        let turns = meter.estimated_turns_remaining();
        assert_eq!(turns, 58);
    }

    #[test]
    fn estimated_turns_remaining_already_past_threshold() {
        let mut meter = ContextMeter::default();
        meter.model_context_window = 128_000;
        meter.compaction_threshold_tokens = 96_000;
        meter.system_prompt_tokens = 50_000;
        meter.conversation_tokens = 50_000;
        meter.avg_tokens_per_turn = 1_000;

        // total = 100000 > 96000
        assert_eq!(meter.estimated_turns_remaining(), 0);
    }

    #[test]
    fn estimated_turns_remaining_zero_avg() {
        let meter = ContextMeter::default();
        assert_eq!(meter.estimated_turns_remaining(), 0);
    }

    // ── Status line formatting ─────────────────────────────────

    #[test]
    fn status_line_under_50_pct() {
        let mut meter = ContextMeter::default();
        meter.model_context_window = 128_000;
        meter.compaction_threshold_tokens = 96_000;
        meter.system_prompt_tokens = 2_847;
        meter.conversation_tokens = 18_420;
        meter.avg_tokens_per_turn = 1_268;
        meter.compaction_depth = 1;

        let line = meter.status_line();
        assert!(line.contains("ctx 21,267/128k"));
        // 21267 * 100 / 128000 = 16 (integer division)
        assert!(line.contains("16%"));
        assert!(line.contains("compaction at 75%"));
        assert!(line.contains("depth 1"));
        // Should NOT have warning emoji
        assert!(!line.contains('\u{26a0}'));
        assert!(!line.contains('\u{1f534}'));
    }

    #[test]
    fn status_line_over_50_pct_has_warning() {
        let mut meter = ContextMeter::default();
        meter.model_context_window = 128_000;
        meter.compaction_threshold_tokens = 96_000;
        meter.system_prompt_tokens = 2_000;
        meter.conversation_tokens = 66_400;
        meter.avg_tokens_per_turn = 2_000;
        meter.compaction_depth = 1;

        let line = meter.status_line();
        assert!(line.contains("53%"));
        assert!(line.contains('\u{26a0}')); // warning emoji
    }

    #[test]
    fn status_line_over_70_pct_has_imminent() {
        let mut meter = ContextMeter::default();
        meter.model_context_window = 128_000;
        meter.compaction_threshold_tokens = 96_000;
        meter.system_prompt_tokens = 2_000;
        meter.conversation_tokens = 90_100;
        meter.avg_tokens_per_turn = 3_000;
        meter.compaction_depth = 2;

        let line = meter.status_line();
        assert!(line.contains("71%") || line.contains("72%"));
        assert!(line.contains('\u{1f534}')); // red circle
        assert!(line.contains("compaction imminent"));
        assert!(line.contains("depth 2"));
    }

    // ── Threshold emoji changes ────────────────────────────────

    #[test]
    fn threshold_emoji_changes_at_boundaries() {
        let mut meter = ContextMeter::default();
        meter.model_context_window = 100;
        meter.compaction_threshold_tokens = 75;
        meter.avg_tokens_per_turn = 1;
        meter.compaction_depth = 0;

        // 49% - no emoji
        meter.system_prompt_tokens = 0;
        meter.conversation_tokens = 49;
        let line = meter.status_line();
        assert!(!line.contains('\u{26a0}'));
        assert!(!line.contains('\u{1f534}'));

        // 50% - warning emoji
        meter.conversation_tokens = 50;
        let line = meter.status_line();
        assert!(line.contains('\u{26a0}'));

        // 69% - still warning
        meter.conversation_tokens = 69;
        let line = meter.status_line();
        assert!(line.contains('\u{26a0}'));

        // 70% - red/imminent
        meter.conversation_tokens = 70;
        let line = meter.status_line();
        assert!(line.contains('\u{1f534}'));
        assert!(line.contains("compaction imminent"));
    }

    // ── CTX_STATUS JSON ────────────────────────────────────────

    #[test]
    fn status_json_is_parseable() {
        let mut meter = ContextMeter::default();
        meter.model_context_window = 128_000;
        meter.compaction_threshold_tokens = 96_000;
        meter.system_prompt_tokens = 2_847;
        meter.conversation_tokens = 18_420;
        meter.avg_tokens_per_turn = 1_268;
        meter.compaction_depth = 1;

        let json_line = meter.status_json();
        assert!(json_line.starts_with("CTX_STATUS:"));

        // Parse the JSON portion
        let json_str = &json_line["CTX_STATUS:".len()..];
        let parsed: serde_json::Value = serde_json::from_str(json_str).unwrap();

        assert_eq!(parsed["used"], 21_267);
        assert_eq!(parsed["max"], 128_000);
        assert_eq!(parsed["pct"], 16);
        assert_eq!(parsed["compact_at_pct"], 75);
        assert_eq!(parsed["depth"], 1);
        assert_eq!(parsed["system"], 2_847);
        assert_eq!(parsed["conversation"], 18_420);
    }

    // ── Telegram footer ────────────────────────────────────────

    #[test]
    fn telegram_footer_none_under_30_pct() {
        let mut meter = ContextMeter::default();
        meter.model_context_window = 100_000;
        meter.system_prompt_tokens = 1_000;
        meter.conversation_tokens = 10_000;

        assert!(meter.telegram_footer().is_none());
    }

    #[test]
    fn telegram_footer_some_over_30_pct() {
        let mut meter = ContextMeter::default();
        meter.model_context_window = 100_000;
        meter.system_prompt_tokens = 5_000;
        meter.conversation_tokens = 30_000;
        meter.avg_tokens_per_turn = 1_000;

        let footer = meter.telegram_footer();
        assert!(footer.is_some());
        let text = footer.unwrap();
        assert!(text.contains("ctx 35%"));
        assert!(text.contains("turns"));
        assert!(text.contains("\u{2500}")); // box-drawing dash
    }

    // ── /context command output ────────────────────────────────

    #[test]
    fn context_detail_output() {
        let mut meter = ContextMeter::default();
        meter.model_context_window = 128_000;
        meter.compaction_threshold_tokens = 96_000;
        meter.system_prompt_tokens = 2_847;
        meter.conversation_tokens = 18_420;
        meter.avg_tokens_per_turn = 1_268;
        meter.turn_count = 34;
        meter.compaction_depth = 1;

        let breakdown = vec![
            ("SOUL.md".to_string(), 400u32),
            ("USER.md".to_string(), 310),
            ("MEMORY.md".to_string(), 1_200),
            ("Tools".to_string(), 937),
        ];

        let output = meter.format_context_detail(
            Some(&breakdown),
            Some("scribe-hardware"),
        );

        assert!(output.contains("System prompt:"));
        assert!(output.contains("2,847"));
        assert!(output.contains("SOUL.md"));
        assert!(output.contains("400"));
        assert!(output.contains("USER.md"));
        assert!(output.contains("MEMORY.md"));
        assert!(output.contains("Conversation:"));
        assert!(output.contains("18,420"));
        assert!(output.contains("34 turns"));
        assert!(output.contains("Total:"));
        assert!(output.contains("21,267"));
        assert!(output.contains("128,000"));
        assert!(output.contains("Headroom:"));
        assert!(output.contains("Compaction at:"));
        assert!(output.contains("96,000"));
        assert!(output.contains("75%"));
        assert!(output.contains("Est. turns left:"));
        assert!(output.contains("1,268 tokens/turn"));
        assert!(output.contains("Compaction depth: 1"));
        assert!(output.contains("Session:"));
        assert!(output.contains("scribe-hardware"));
    }

    #[test]
    fn context_detail_without_breakdown() {
        let meter = ContextMeter::default();
        let output = meter.format_context_detail(None, None);

        assert!(output.contains("System prompt:"));
        assert!(output.contains("Conversation:"));
        assert!(output.contains("Total:"));
        assert!(!output.contains("Session:"));
    }

    // ── Serialization / deserialization ─────────────────────────

    #[test]
    fn serialize_deserialize_round_trip() {
        let mut meter = ContextMeter::default();
        meter.model_context_window = 200_000;
        meter.system_prompt_tokens = 3_000;
        meter.conversation_tokens = 25_000;
        meter.last_completion_tokens = 800;
        meter.compaction_threshold_tokens = 150_000;
        meter.compaction_depth = 2;
        meter.turn_count = 15;
        meter.avg_tokens_per_turn = 1_500;

        let json = serde_json::to_string(&meter).unwrap();
        let deserialized: ContextMeter = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.model_context_window, 200_000);
        assert_eq!(deserialized.system_prompt_tokens, 3_000);
        assert_eq!(deserialized.conversation_tokens, 25_000);
        assert_eq!(deserialized.last_completion_tokens, 800);
        assert_eq!(deserialized.compaction_threshold_tokens, 150_000);
        assert_eq!(deserialized.compaction_depth, 2);
        assert_eq!(deserialized.turn_count, 15);
        assert_eq!(deserialized.avg_tokens_per_turn, 1_500);
    }

    #[test]
    fn deserialize_missing_fields_defaults_to_zero() {
        // Simulate an old session metadata that doesn't have ContextMeter fields
        let json = "{}";
        let meter: ContextMeter = serde_json::from_str(json).unwrap_or_default();

        assert_eq!(meter.model_context_window, DEFAULT_CONTEXT_WINDOW);
        assert_eq!(meter.system_prompt_tokens, 0);
        assert_eq!(meter.conversation_tokens, 0);
        assert_eq!(meter.turn_count, 0);
    }

    // ── Number formatting ──────────────────────────────────────

    #[test]
    fn format_number_basic() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(1), "1");
        assert_eq!(format_number(42), "42");
        assert_eq!(format_number(999), "999");
        assert_eq!(format_number(1_000), "1,000");
        assert_eq!(format_number(21_267), "21,267");
        assert_eq!(format_number(128_000), "128,000");
        assert_eq!(format_number(1_000_000), "1,000,000");
    }

    // ── Context size formatting ────────────────────────────────

    #[test]
    fn format_context_size_basic() {
        assert_eq!(ContextMeter::format_context_size(128_000), "128k");
        assert_eq!(ContextMeter::format_context_size(200_000), "200k");
        assert_eq!(ContextMeter::format_context_size(1_000_000), "1M");
        assert_eq!(ContextMeter::format_context_size(8_000), "8k");
    }

    // ── Known models table ─────────────────────────────────────

    #[test]
    fn known_models_contains_expected() {
        let table = known_models();
        assert_eq!(table["claude-opus-4-6"], 200_000);
        assert_eq!(table["claude-opus-4-6-extended"], 1_000_000);
        assert_eq!(table["claude-sonnet-4-6"], 200_000);
        assert_eq!(table["gpt-5.2"], 400_000);
        assert_eq!(table["gpt-5.2-chat"], 128_000);
        assert_eq!(table["gemini-3-pro"], 1_000_000);
        assert_eq!(table["gemini-3-flash"], 1_000_000);
    }

    // ── Config parsing ─────────────────────────────────────────

    #[test]
    fn parse_models_config_basic() {
        let toml = r#"
[models]
"claude-opus-4-6" = 250000
"my-custom-model" = 64000
"#;
        let result = parse_models_config(toml);
        assert!(result.is_some());
        let map = result.unwrap();
        assert_eq!(map["claude-opus-4-6"], 250_000);
        assert_eq!(map["my-custom-model"], 64_000);
    }

    #[test]
    fn parse_models_config_missing_section() {
        let toml = r#"
[provider]
primary = "anthropic"
"#;
        assert!(parse_models_config(toml).is_none());
    }

    #[test]
    fn parse_models_config_empty_section() {
        let toml = r#"
[models]
"#;
        assert!(parse_models_config(toml).is_none());
    }

    // ── /context command parsing ───────────────────────────────

    #[test]
    fn is_context_command_matches() {
        assert!(is_context_command("/context"));
        assert!(is_context_command("  /context  "));
        assert!(is_context_command("/CONTEXT"));
        assert!(is_context_command("/Context"));
    }

    #[test]
    fn is_context_command_no_false_matches() {
        assert!(!is_context_command("/contexts"));
        assert!(!is_context_command("/context window"));
        assert!(!is_context_command("What is the context?"));
        assert!(!is_context_command(""));
        assert!(!is_context_command("/new"));
    }

    // ── Headroom calculation ───────────────────────────────────

    #[test]
    fn headroom_calculation() {
        let mut meter = ContextMeter::default();
        meter.model_context_window = 128_000;
        meter.system_prompt_tokens = 3_000;
        meter.conversation_tokens = 25_000;

        assert_eq!(meter.headroom(), 100_000);
    }

    #[test]
    fn headroom_saturates_at_zero() {
        let mut meter = ContextMeter::default();
        meter.model_context_window = 100;
        meter.system_prompt_tokens = 50;
        meter.conversation_tokens = 60;

        assert_eq!(meter.headroom(), 0); // 100 - 110 saturates to 0
    }

    // ── Compact at pct ─────────────────────────────────────────

    #[test]
    fn compact_at_pct_basic() {
        let mut meter = ContextMeter::default();
        meter.model_context_window = 128_000;
        meter.compaction_threshold_tokens = 96_000;

        assert_eq!(meter.compact_at_pct(), 75);
    }
}

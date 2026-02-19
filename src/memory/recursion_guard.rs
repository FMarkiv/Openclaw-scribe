//! Tool recursion and loop detection for the ZeroClaw agent loop.
//!
//! Detects three classes of unproductive repetition:
//!
//! 1. **Identical tool calls** — the same tool with the same canonical
//!    parameters repeated N times (default 3).
//! 2. **Oscillation patterns** — A→B→A→B or A→B→C→A→B→C cycles in the
//!    most recent tool calls (default depth 4).
//! 3. **Memory store spam** — `memory_store` called 3+ times in one
//!    loop with substantially similar content (>80% token overlap).
//!
//! On detection the guard emits an [`Intervention`] rather than aborting
//! the loop outright, giving the agent one more chance to break the
//! cycle.  If the agent *still* repeats after intervention, a forced
//! stop is triggered.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use crate::memory::recursion_guard::{RecursionGuard, RecursionGuardConfig};
//!
//! let config = RecursionGuardConfig::default();
//! let mut guard = RecursionGuard::new(config);
//!
//! // After each tool call in the agent loop:
//! if let Some(intervention) = guard.record_and_check("memory_store", &args_json) {
//!     match intervention {
//!         Intervention::Warn { message, .. } => {
//!             // Inject `message` as a system message
//!         }
//!         Intervention::ForceStop { message, .. } => {
//!             // End the tool loop, return message to user
//!         }
//!     }
//! }
//!
//! // Reset when the user sends a new message:
//! guard.reset();
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

// ── Configuration ───────────────────────────────────────────────

/// Configuration for the recursion guard, loadable from `[agent]` in
/// `config.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecursionGuardConfig {
    /// Number of identical tool calls before the first warning.
    pub max_identical_tool_calls: usize,
    /// Length of the trailing window checked for oscillation patterns.
    pub max_oscillation_depth: usize,
    /// Maximum number of entries kept in the fingerprint window.
    pub window_size: usize,
    /// Similarity threshold (0.0–1.0) for fuzzy memory_store spam
    /// detection.  Two contents are "substantially similar" when their
    /// token overlap ratio exceeds this value.
    pub memory_spam_similarity: f64,
    /// How many similar `memory_store` calls trigger a spam warning.
    pub memory_spam_threshold: usize,
}

impl Default for RecursionGuardConfig {
    fn default() -> Self {
        Self {
            max_identical_tool_calls: 3,
            max_oscillation_depth: 4,
            window_size: 50,
            memory_spam_similarity: 0.80,
            memory_spam_threshold: 3,
        }
    }
}

// ── Fingerprint ─────────────────────────────────────────────────

/// A fingerprint is a u64 hash of `tool_name + canonical(params)`.
///
/// Canonical form: JSON keys sorted, whitespace trimmed, so
/// `{"b":"2","a":"1"}` and `{ "a": "1", "b": "2" }` produce the
/// same hash.
type Fingerprint = u64;

/// Compute the canonical fingerprint for a tool call.
pub fn fingerprint(tool_name: &str, params: &Value) -> Fingerprint {
    let canonical = canonicalize(params);
    let mut hasher = DefaultHasher::new();
    tool_name.hash(&mut hasher);
    canonical.hash(&mut hasher);
    hasher.finish()
}

/// Produce a canonical JSON string: sorted keys, no extraneous whitespace.
fn canonicalize(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let entries: Vec<String> = keys
                .iter()
                .map(|k| format!("\"{}\":{}", k, canonicalize(&map[k.as_str()])))
                .collect();
            format!("{{{}}}", entries.join(","))
        }
        Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(canonicalize).collect();
            format!("[{}]", items.join(","))
        }
        Value::String(s) => format!("\"{}\"", s.trim()),
        _ => value.to_string(),
    }
}

// ── Window entry ────────────────────────────────────────────────

/// A single entry in the rolling fingerprint window.
#[derive(Debug, Clone)]
struct WindowEntry {
    tool_name: String,
    fingerprint: Fingerprint,
    /// Raw content for memory_store calls (for fuzzy matching).
    memory_content: Option<String>,
}

// ── Intervention ────────────────────────────────────────────────

/// What the guard wants the agent loop to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Intervention {
    /// Inject a warning system message but let the loop continue.
    Warn {
        message: String,
        tool_name: String,
        kind: DetectionKind,
    },
    /// Force-stop the tool loop and surface the message to the user.
    ForceStop {
        message: String,
        tool_name: String,
        kind: DetectionKind,
    },
}

/// Which detector fired.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DetectionKind {
    IdenticalCall,
    Oscillation,
    MemoryStoreSpam,
}

impl std::fmt::Display for DetectionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DetectionKind::IdenticalCall => write!(f, "identical_call"),
            DetectionKind::Oscillation => write!(f, "oscillation"),
            DetectionKind::MemoryStoreSpam => write!(f, "memory_store_spam"),
        }
    }
}

// ── Detection event (for structured logging) ────────────────────

/// A structured log record emitted when detection fires.
#[derive(Debug, Clone, Serialize)]
pub struct DetectionEvent {
    pub kind: DetectionKind,
    pub tool_name: String,
    pub count: usize,
    pub forced_stop: bool,
    pub message: String,
}

// ── RecursionGuard ──────────────────────────────────────────────

/// In-memory guard that tracks tool calls within an agent loop
/// invocation and detects unproductive repetition.
pub struct RecursionGuard {
    config: RecursionGuardConfig,
    /// Rolling window of recent tool call fingerprints.
    window: Vec<WindowEntry>,
    /// Count of each fingerprint seen in the current window.
    counts: HashMap<Fingerprint, usize>,
    /// Whether a warning has already been issued (to know when to
    /// escalate to ForceStop).
    warned: HashMap<Fingerprint, bool>,
    /// Whether an oscillation warning has been issued.
    oscillation_warned: bool,
    /// Whether a memory spam warning has been issued.
    memory_spam_warned: bool,
    /// All detection events this loop (for structured logging).
    events: Vec<DetectionEvent>,
}

impl RecursionGuard {
    /// Create a new guard with the given configuration.
    pub fn new(config: RecursionGuardConfig) -> Self {
        Self {
            config,
            window: Vec::new(),
            counts: HashMap::new(),
            warned: HashMap::new(),
            oscillation_warned: false,
            memory_spam_warned: false,
            events: Vec::new(),
        }
    }

    /// Create a guard with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(RecursionGuardConfig::default())
    }

    /// Record a tool call and check for all recursion patterns.
    ///
    /// Returns `Some(Intervention)` if a pattern was detected, or
    /// `None` if everything looks fine.
    ///
    /// The caller should:
    /// - On `Warn`: inject `message` as a system message and continue.
    /// - On `ForceStop`: end the tool loop and surface `message`.
    pub fn record_and_check(
        &mut self,
        tool_name: &str,
        params: &Value,
    ) -> Option<Intervention> {
        let fp = fingerprint(tool_name, params);

        // Extract memory_store content for spam detection.
        let memory_content = if tool_name == "memory_store" {
            params["content"].as_str().map(|s| s.to_string())
        } else {
            None
        };

        // Push entry into window.
        self.window.push(WindowEntry {
            tool_name: tool_name.to_string(),
            fingerprint: fp,
            memory_content,
        });

        // Update count.
        let count = self.counts.entry(fp).or_insert(0);
        *count += 1;
        let current_count = *count;

        // Enforce window size cap.
        self.enforce_window_cap();

        // Check detectors in priority order.

        // 1. Identical call detection.
        if let Some(intervention) = self.check_identical(tool_name, fp, current_count) {
            return Some(intervention);
        }

        // 2. Oscillation detection.
        if let Some(intervention) = self.check_oscillation() {
            return Some(intervention);
        }

        // 3. Memory store spam detection.
        if tool_name == "memory_store" {
            if let Some(intervention) = self.check_memory_spam() {
                return Some(intervention);
            }
        }

        None
    }

    /// Reset all state.  Call when the user sends a new message,
    /// a session switch occurs, or compaction happens.
    pub fn reset(&mut self) {
        self.window.clear();
        self.counts.clear();
        self.warned.clear();
        self.oscillation_warned = false;
        self.memory_spam_warned = false;
        // Keep events for session-level logging.
    }

    /// All detection events emitted so far (for structured logging).
    pub fn events(&self) -> &[DetectionEvent] {
        &self.events
    }

    /// Current window size.
    pub fn window_len(&self) -> usize {
        self.window.len()
    }

    /// The guard's configuration.
    pub fn config(&self) -> &RecursionGuardConfig {
        &self.config
    }

    // ── Private detectors ──────────────────────────────────────

    /// Enforce the rolling window size cap.  When the window exceeds
    /// the configured size, the oldest entry is removed and its count
    /// decremented.
    fn enforce_window_cap(&mut self) {
        while self.window.len() > self.config.window_size {
            let removed = self.window.remove(0);
            if let Some(c) = self.counts.get_mut(&removed.fingerprint) {
                *c = c.saturating_sub(1);
                if *c == 0 {
                    self.counts.remove(&removed.fingerprint);
                    self.warned.remove(&removed.fingerprint);
                }
            }
        }
    }

    /// Check whether the same fingerprint has appeared too many times.
    fn check_identical(
        &mut self,
        tool_name: &str,
        fp: Fingerprint,
        count: usize,
    ) -> Option<Intervention> {
        let threshold = self.config.max_identical_tool_calls;

        if count < threshold {
            return None;
        }

        let already_warned = self.warned.get(&fp).copied().unwrap_or(false);

        // First time hitting threshold → warn.
        // After warning (threshold + 2 total) → force stop.
        if !already_warned {
            self.warned.insert(fp, true);
            let message = format!(
                "You have called `{tool_name}` with identical parameters {count} times. \
                 This appears to be a loop. Try a different approach, ask the user for \
                 clarification, or explain why you're stuck."
            );
            let event = DetectionEvent {
                kind: DetectionKind::IdenticalCall,
                tool_name: tool_name.to_string(),
                count,
                forced_stop: false,
                message: message.clone(),
            };
            self.events.push(event);
            Some(Intervention::Warn {
                message,
                tool_name: tool_name.to_string(),
                kind: DetectionKind::IdenticalCall,
            })
        } else if count >= threshold + 2 {
            let message = format!(
                "I was stuck in a loop calling `{tool_name}` repeatedly and stopped \
                 automatically."
            );
            let event = DetectionEvent {
                kind: DetectionKind::IdenticalCall,
                tool_name: tool_name.to_string(),
                count,
                forced_stop: true,
                message: message.clone(),
            };
            self.events.push(event);
            Some(Intervention::ForceStop {
                message,
                tool_name: tool_name.to_string(),
                kind: DetectionKind::IdenticalCall,
            })
        } else {
            None
        }
    }

    /// Check whether the most recent tool calls form an oscillation
    /// pattern (repeating cycle of length 2 or 3).
    fn check_oscillation(&mut self) -> Option<Intervention> {
        let depth = self.config.max_oscillation_depth;
        let len = self.window.len();
        if len < depth {
            return None;
        }

        let tail: Vec<Fingerprint> = self.window[len - depth..]
            .iter()
            .map(|e| e.fingerprint)
            .collect();

        // Check cycle of length 2: A B A B …
        if depth >= 4 && self.is_repeating_cycle(&tail, 2) {
            return self.emit_oscillation(2);
        }

        // Check cycle of length 3: A B C A B C …
        if depth >= 6 && self.is_repeating_cycle(&tail, 3) {
            return self.emit_oscillation(3);
        }

        None
    }

    /// Returns true if `seq` is a repeating cycle of the given length
    /// and the cycle contains at least 2 distinct fingerprints (otherwise
    /// it's just identical-call repetition, not oscillation).
    fn is_repeating_cycle(&self, seq: &[Fingerprint], cycle_len: usize) -> bool {
        if seq.len() < cycle_len * 2 {
            return false;
        }
        let pattern = &seq[..cycle_len];

        // Require at least 2 distinct fingerprints — a cycle of all
        // identical calls is handled by the identical-call detector.
        let mut seen = std::collections::HashSet::new();
        for fp in pattern {
            seen.insert(fp);
        }
        if seen.len() < 2 {
            return false;
        }

        // Verify at least two full repetitions of the pattern exist
        // within the sequence.
        seq.chunks(cycle_len)
            .all(|chunk| chunk == pattern)
    }

    /// Emit an oscillation intervention (warn or force-stop).
    fn emit_oscillation(&mut self, cycle_len: usize) -> Option<Intervention> {
        let tool_names: Vec<String> = self.window
            [self.window.len().saturating_sub(cycle_len)..]
            .iter()
            .map(|e| e.tool_name.clone())
            .collect();
        let cycle_desc = tool_names.join(" → ");

        if !self.oscillation_warned {
            self.oscillation_warned = true;
            let message = format!(
                "Your last tool calls form a repeating pattern ({cycle_desc}). \
                 This appears to be an oscillation loop. Try a different approach, \
                 ask the user for clarification, or explain why you're stuck."
            );
            let event = DetectionEvent {
                kind: DetectionKind::Oscillation,
                tool_name: cycle_desc.clone(),
                count: cycle_len,
                forced_stop: false,
                message: message.clone(),
            };
            self.events.push(event);
            Some(Intervention::Warn {
                message,
                tool_name: cycle_desc,
                kind: DetectionKind::Oscillation,
            })
        } else {
            let message =
                "I was stuck in an oscillation loop and stopped automatically.".to_string();
            let event = DetectionEvent {
                kind: DetectionKind::Oscillation,
                tool_name: cycle_desc.clone(),
                count: cycle_len,
                forced_stop: true,
                message: message.clone(),
            };
            self.events.push(event);
            Some(Intervention::ForceStop {
                message,
                tool_name: cycle_desc,
                kind: DetectionKind::Oscillation,
            })
        }
    }

    /// Check whether `memory_store` has been called with substantially
    /// similar content multiple times.
    fn check_memory_spam(&mut self) -> Option<Intervention> {
        let contents: Vec<&str> = self
            .window
            .iter()
            .filter(|e| e.tool_name == "memory_store")
            .filter_map(|e| e.memory_content.as_deref())
            .collect();

        if contents.len() < self.config.memory_spam_threshold {
            return None;
        }

        // Count how many pairs exceed the similarity threshold.
        let mut similar_count = 0usize;
        // Compare the latest entry against all previous ones.
        let latest = match contents.last() {
            Some(c) => *c,
            None => return None,
        };

        for &earlier in &contents[..contents.len() - 1] {
            if token_overlap(latest, earlier) >= self.config.memory_spam_similarity {
                similar_count += 1;
            }
        }

        // Need (threshold - 1) similar earlier entries for a cluster.
        if similar_count + 1 < self.config.memory_spam_threshold {
            return None;
        }

        if !self.memory_spam_warned {
            self.memory_spam_warned = true;
            let message = format!(
                "You have called `memory_store` {} times with substantially similar \
                 content. This appears to be redundant. The information is already \
                 stored — try a different approach or move on.",
                contents.len()
            );
            let event = DetectionEvent {
                kind: DetectionKind::MemoryStoreSpam,
                tool_name: "memory_store".to_string(),
                count: contents.len(),
                forced_stop: false,
                message: message.clone(),
            };
            self.events.push(event);
            Some(Intervention::Warn {
                message,
                tool_name: "memory_store".to_string(),
                kind: DetectionKind::MemoryStoreSpam,
            })
        } else {
            let message =
                "I was stuck in a loop storing similar memories and stopped automatically."
                    .to_string();
            let event = DetectionEvent {
                kind: DetectionKind::MemoryStoreSpam,
                tool_name: "memory_store".to_string(),
                count: contents.len(),
                forced_stop: true,
                message: message.clone(),
            };
            self.events.push(event);
            Some(Intervention::ForceStop {
                message,
                tool_name: "memory_store".to_string(),
                kind: DetectionKind::MemoryStoreSpam,
            })
        }
    }
}

// ── Fuzzy matching ──────────────────────────────────────────────

/// Compute the token-level overlap ratio between two strings.
///
/// Tokenization: split on whitespace, lowercased.  Ratio is
/// `|intersection| / |union|` (Jaccard similarity).
///
/// Returns a value in `[0.0, 1.0]`.
pub fn token_overlap(a: &str, b: &str) -> f64 {
    let tokens_a: std::collections::HashSet<&str> =
        a.split_whitespace().map(|t| t.trim()).collect();
    let tokens_b: std::collections::HashSet<&str> =
        b.split_whitespace().map(|t| t.trim()).collect();

    if tokens_a.is_empty() && tokens_b.is_empty() {
        return 1.0;
    }

    let intersection = tokens_a.intersection(&tokens_b).count();
    let union = tokens_a.union(&tokens_b).count();

    if union == 0 {
        return 1.0;
    }

    intersection as f64 / union as f64
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── Fingerprinting tests ────────────────────────────────────

    #[test]
    fn canonical_sorts_keys() {
        let a = json!({"b": 2, "a": 1});
        let b = json!({"a": 1, "b": 2});
        assert_eq!(canonicalize(&a), canonicalize(&b));
    }

    #[test]
    fn canonical_trims_string_whitespace() {
        let a = json!({"key": "  hello  "});
        let b = json!({"key": "hello"});
        assert_eq!(canonicalize(&a), canonicalize(&b));
    }

    #[test]
    fn canonical_handles_nested_objects() {
        let a = json!({"outer": {"z": 1, "a": 2}});
        let b = json!({"outer": {"a": 2, "z": 1}});
        assert_eq!(canonicalize(&a), canonicalize(&b));
    }

    #[test]
    fn canonical_handles_arrays() {
        let a = json!({"items": [1, 2, 3]});
        let b = json!({"items": [1, 2, 3]});
        assert_eq!(canonicalize(&a), canonicalize(&b));

        // Array order matters.
        let c = json!({"items": [3, 2, 1]});
        assert_ne!(canonicalize(&a), canonicalize(&c));
    }

    #[test]
    fn fingerprint_same_for_semantically_identical() {
        let fp1 = fingerprint("file_read", &json!({"path": "src/main.rs"}));
        let fp2 = fingerprint("file_read", &json!({"path": "src/main.rs"}));
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn fingerprint_differs_for_different_tool() {
        let fp1 = fingerprint("file_read", &json!({"path": "src/main.rs"}));
        let fp2 = fingerprint("file_write", &json!({"path": "src/main.rs"}));
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn fingerprint_differs_for_different_params() {
        let fp1 = fingerprint("file_read", &json!({"path": "src/main.rs"}));
        let fp2 = fingerprint("file_read", &json!({"path": "src/lib.rs"}));
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn fingerprint_key_order_irrelevant() {
        let fp1 = fingerprint("tool", &json!({"a": 1, "b": 2}));
        let fp2 = fingerprint("tool", &json!({"b": 2, "a": 1}));
        assert_eq!(fp1, fp2);
    }

    // ── Identical call detection ────────────────────────────────

    #[test]
    fn no_intervention_below_threshold() {
        let mut guard = RecursionGuard::with_defaults();
        let params = json!({"path": "src/main.rs"});

        // 2 calls — below default threshold of 3.
        assert!(guard.record_and_check("file_read", &params).is_none());
        assert!(guard.record_and_check("file_read", &params).is_none());
    }

    #[test]
    fn warns_at_threshold() {
        let mut guard = RecursionGuard::with_defaults();
        let params = json!({"path": "src/main.rs"});

        guard.record_and_check("file_read", &params);
        guard.record_and_check("file_read", &params);
        let result = guard.record_and_check("file_read", &params); // 3rd

        match result {
            Some(Intervention::Warn { tool_name, kind, .. }) => {
                assert_eq!(tool_name, "file_read");
                assert_eq!(kind, DetectionKind::IdenticalCall);
            }
            other => panic!("Expected Warn, got {other:?}"),
        }
    }

    #[test]
    fn force_stops_after_warning_plus_two() {
        let mut guard = RecursionGuard::with_defaults();
        let params = json!({"path": "src/main.rs"});

        // Calls 1-2: nothing.
        guard.record_and_check("file_read", &params);
        guard.record_and_check("file_read", &params);

        // Call 3: warn.
        let r3 = guard.record_and_check("file_read", &params);
        assert!(matches!(r3, Some(Intervention::Warn { .. })));

        // Call 4: nothing (warned but under threshold+2).
        let r4 = guard.record_and_check("file_read", &params);
        assert!(r4.is_none());

        // Call 5: force stop (threshold=3 + 2 = 5).
        let r5 = guard.record_and_check("file_read", &params);
        match r5 {
            Some(Intervention::ForceStop { kind, .. }) => {
                assert_eq!(kind, DetectionKind::IdenticalCall);
            }
            other => panic!("Expected ForceStop, got {other:?}"),
        }
    }

    #[test]
    fn different_tools_tracked_independently() {
        let mut guard = RecursionGuard::with_defaults();

        guard.record_and_check("file_read", &json!({"path": "a"}));
        guard.record_and_check("file_read", &json!({"path": "a"}));
        guard.record_and_check("shell_exec", &json!({"cmd": "ls"}));
        guard.record_and_check("shell_exec", &json!({"cmd": "ls"}));

        // Neither has hit 3 yet.
        assert_eq!(guard.events().len(), 0);
    }

    #[test]
    fn configurable_threshold() {
        let config = RecursionGuardConfig {
            max_identical_tool_calls: 2,
            ..Default::default()
        };
        let mut guard = RecursionGuard::new(config);
        let params = json!({"x": 1});

        guard.record_and_check("t", &params);
        let r = guard.record_and_check("t", &params); // 2nd = threshold
        assert!(matches!(r, Some(Intervention::Warn { .. })));
    }

    // ── Oscillation detection ───────────────────────────────────

    #[test]
    fn detects_ab_ab_oscillation() {
        let mut guard = RecursionGuard::with_defaults(); // depth=4

        let params_a = json!({"path": "a"});
        let params_b = json!({"path": "b"});

        guard.record_and_check("tool_a", &params_a);
        guard.record_and_check("tool_b", &params_b);
        guard.record_and_check("tool_a", &params_a);
        let result = guard.record_and_check("tool_b", &params_b); // A B A B

        match result {
            Some(Intervention::Warn { kind, .. }) => {
                assert_eq!(kind, DetectionKind::Oscillation);
            }
            other => panic!("Expected oscillation Warn, got {other:?}"),
        }
    }

    #[test]
    fn no_oscillation_with_varied_calls() {
        let mut guard = RecursionGuard::with_defaults();

        guard.record_and_check("tool_a", &json!({"x": 1}));
        guard.record_and_check("tool_b", &json!({"x": 2}));
        guard.record_and_check("tool_c", &json!({"x": 3}));
        let result = guard.record_and_check("tool_d", &json!({"x": 4}));

        assert!(result.is_none());
    }

    #[test]
    fn oscillation_force_stops_after_warning() {
        let mut guard = RecursionGuard::with_defaults();
        let pa = json!({"x": 1});
        let pb = json!({"x": 2});

        // First cycle: A B A B → warn.
        guard.record_and_check("a", &pa);
        guard.record_and_check("b", &pb);
        guard.record_and_check("a", &pa);
        let r1 = guard.record_and_check("b", &pb);
        assert!(matches!(r1, Some(Intervention::Warn { .. })));

        // Second cycle: A B A B → force stop.
        guard.record_and_check("a", &pa);
        guard.record_and_check("b", &pb);
        guard.record_and_check("a", &pa);
        let r2 = guard.record_and_check("b", &pb);
        assert!(matches!(r2, Some(Intervention::ForceStop { .. })));
    }

    #[test]
    fn detects_abc_abc_oscillation() {
        let config = RecursionGuardConfig {
            max_oscillation_depth: 6,
            ..Default::default()
        };
        let mut guard = RecursionGuard::new(config);
        let pa = json!({"x": 1});
        let pb = json!({"x": 2});
        let pc = json!({"x": 3});

        guard.record_and_check("a", &pa);
        guard.record_and_check("b", &pb);
        guard.record_and_check("c", &pc);
        guard.record_and_check("a", &pa);
        guard.record_and_check("b", &pb);
        let result = guard.record_and_check("c", &pc); // A B C A B C

        match result {
            Some(Intervention::Warn { kind, .. }) => {
                assert_eq!(kind, DetectionKind::Oscillation);
            }
            other => panic!("Expected oscillation Warn, got {other:?}"),
        }
    }

    // ── Memory store spam detection ─────────────────────────────

    #[test]
    fn detects_similar_memory_store_calls() {
        let mut guard = RecursionGuard::with_defaults();

        // All three share high Jaccard overlap (>80%).
        let r1 = guard.record_and_check(
            "memory_store",
            &json!({"content": "The project uses Rust and Tokio for async IO operations here"}),
        );
        assert!(r1.is_none());

        let r2 = guard.record_and_check(
            "memory_store",
            &json!({"content": "The project uses Rust and Tokio for async IO operations now"}),
        );
        assert!(r2.is_none());

        let r3 = guard.record_and_check(
            "memory_store",
            &json!({"content": "The project uses Rust and Tokio for async IO operations today"}),
        );

        match r3 {
            Some(Intervention::Warn { kind, .. }) => {
                assert_eq!(kind, DetectionKind::MemoryStoreSpam);
            }
            other => panic!("Expected memory spam Warn, got {other:?}"),
        }
    }

    #[test]
    fn no_spam_for_different_content() {
        let mut guard = RecursionGuard::with_defaults();

        guard.record_and_check(
            "memory_store",
            &json!({"content": "The project uses Rust"}),
        );
        guard.record_and_check(
            "memory_store",
            &json!({"content": "User prefers dark mode in the terminal"}),
        );
        let r = guard.record_and_check(
            "memory_store",
            &json!({"content": "Build system requires cargo and make"}),
        );

        assert!(r.is_none());
    }

    #[test]
    fn memory_spam_force_stops_after_warning() {
        let mut guard = RecursionGuard::with_defaults();

        // 3 similar calls (>80% Jaccard overlap) → warn.
        guard.record_and_check(
            "memory_store",
            &json!({"content": "Rust project uses cargo for builds and dependency management yes"}),
        );
        guard.record_and_check(
            "memory_store",
            &json!({"content": "Rust project uses cargo for builds and dependency management now"}),
        );
        let r3 = guard.record_and_check(
            "memory_store",
            &json!({"content": "Rust project uses cargo for builds and dependency management too"}),
        );
        assert!(matches!(r3, Some(Intervention::Warn { .. })));

        // 4th similar call → force stop.
        let r4 = guard.record_and_check(
            "memory_store",
            &json!({"content": "Rust project uses cargo for builds and dependency management here"}),
        );
        assert!(matches!(r4, Some(Intervention::ForceStop { .. })));
    }

    // ── Intervention message injection ──────────────────────────

    #[test]
    fn warn_message_contains_tool_name_and_count() {
        let mut guard = RecursionGuard::with_defaults();
        let params = json!({"q": "test"});

        guard.record_and_check("search", &params);
        guard.record_and_check("search", &params);
        let result = guard.record_and_check("search", &params);

        if let Some(Intervention::Warn { message, .. }) = result {
            assert!(message.contains("search"));
            assert!(message.contains("3 times"));
            assert!(message.contains("different approach"));
        } else {
            panic!("Expected Warn intervention");
        }
    }

    #[test]
    fn force_stop_message_explains_loop() {
        let mut guard = RecursionGuard::with_defaults();
        let params = json!({"q": "test"});

        for _ in 0..4 {
            guard.record_and_check("search", &params);
        }
        let result = guard.record_and_check("search", &params); // 5th

        if let Some(Intervention::ForceStop { message, .. }) = result {
            assert!(message.contains("stuck in a loop"));
            assert!(message.contains("stopped automatically"));
        } else {
            panic!("Expected ForceStop intervention");
        }
    }

    // ── Reset on new user message ───────────────────────────────

    #[test]
    fn reset_clears_all_state() {
        let mut guard = RecursionGuard::with_defaults();
        let params = json!({"path": "test"});

        guard.record_and_check("file_read", &params);
        guard.record_and_check("file_read", &params);
        assert_eq!(guard.window_len(), 2);

        guard.reset();

        assert_eq!(guard.window_len(), 0);

        // Should not warn — count reset to 0.
        let r = guard.record_and_check("file_read", &params);
        assert!(r.is_none());
        assert_eq!(guard.window_len(), 1);
    }

    #[test]
    fn reset_allows_same_calls_again() {
        let mut guard = RecursionGuard::with_defaults();
        let params = json!({"x": 1});

        // Hit threshold.
        guard.record_and_check("t", &params);
        guard.record_and_check("t", &params);
        let r = guard.record_and_check("t", &params);
        assert!(matches!(r, Some(Intervention::Warn { .. })));

        // Simulate new user message.
        guard.reset();

        // Same calls again — should not trigger until threshold again.
        assert!(guard.record_and_check("t", &params).is_none());
        assert!(guard.record_and_check("t", &params).is_none());
    }

    // ── Window size cap ─────────────────────────────────────────

    #[test]
    fn window_respects_size_cap() {
        let config = RecursionGuardConfig {
            window_size: 5,
            max_identical_tool_calls: 10, // high so we don't trigger
            ..Default::default()
        };
        let mut guard = RecursionGuard::new(config);

        for i in 0..10 {
            guard.record_and_check("tool", &json!({"i": i}));
        }

        assert_eq!(guard.window_len(), 5);
    }

    #[test]
    fn evicted_entries_decrement_counts() {
        let config = RecursionGuardConfig {
            window_size: 3,
            max_identical_tool_calls: 4, // won't trigger with window=3
            ..Default::default()
        };
        let mut guard = RecursionGuard::new(config);
        let params = json!({"x": 1});

        // Fill window with same call.
        guard.record_and_check("t", &params); // [t]
        guard.record_and_check("t", &params); // [t, t]
        guard.record_and_check("t", &params); // [t, t, t]

        // Push new call — oldest evicted, count drops to 2.
        guard.record_and_check("other", &json!({"y": 2})); // [t, t, other]

        // Count for "t" with params should now be 2, not 3.
        assert_eq!(*guard.counts.get(&fingerprint("t", &params)).unwrap(), 2);
    }

    // ── Token overlap tests ─────────────────────────────────────

    #[test]
    fn token_overlap_identical() {
        assert!((token_overlap("hello world", "hello world") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn token_overlap_no_overlap() {
        assert!((token_overlap("hello world", "foo bar")).abs() < f64::EPSILON);
    }

    #[test]
    fn token_overlap_partial() {
        let ratio = token_overlap("the project uses rust", "the project uses python");
        // 3 shared out of 5 unique = 0.6
        assert!(ratio > 0.5);
        assert!(ratio < 0.8);
    }

    #[test]
    fn token_overlap_empty_strings() {
        assert!((token_overlap("", "") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn token_overlap_one_empty() {
        assert!((token_overlap("hello", "")).abs() < f64::EPSILON);
    }

    // ── Structured logging tests ────────────────────────────────

    #[test]
    fn events_recorded_on_detection() {
        let mut guard = RecursionGuard::with_defaults();
        let params = json!({"p": 1});

        guard.record_and_check("t", &params);
        guard.record_and_check("t", &params);
        guard.record_and_check("t", &params);

        assert_eq!(guard.events().len(), 1);
        assert_eq!(guard.events()[0].kind, DetectionKind::IdenticalCall);
        assert!(!guard.events()[0].forced_stop);
    }

    #[test]
    fn events_preserved_across_reset() {
        let mut guard = RecursionGuard::with_defaults();
        let params = json!({"p": 1});

        guard.record_and_check("t", &params);
        guard.record_and_check("t", &params);
        guard.record_and_check("t", &params);

        guard.reset();

        // Events are kept for session-level logging even after reset.
        assert_eq!(guard.events().len(), 1);
    }

    // ── Edge cases ──────────────────────────────────────────────

    #[test]
    fn empty_params() {
        let mut guard = RecursionGuard::with_defaults();
        let params = json!({});

        guard.record_and_check("t", &params);
        guard.record_and_check("t", &params);
        let r = guard.record_and_check("t", &params);
        assert!(matches!(r, Some(Intervention::Warn { .. })));
    }

    #[test]
    fn null_params() {
        let mut guard = RecursionGuard::with_defaults();
        let params = Value::Null;

        guard.record_and_check("t", &params);
        guard.record_and_check("t", &params);
        let r = guard.record_and_check("t", &params);
        assert!(matches!(r, Some(Intervention::Warn { .. })));
    }

    #[test]
    fn memory_store_without_content_field() {
        let mut guard = RecursionGuard::with_defaults();
        let params = json!({"category": "note"}); // no content field

        // Should not panic, just skip spam detection.
        guard.record_and_check("memory_store", &params);
        guard.record_and_check("memory_store", &params);
        let r = guard.record_and_check("memory_store", &params);
        // Will trigger identical call detection, not spam.
        assert!(matches!(r, Some(Intervention::Warn { kind: DetectionKind::IdenticalCall, .. })));
    }
}

//! LLM-based semantic search for memory recall (Phase 2).
//!
//! After keyword grep (Phase 1), matching entries are sent to an LLM
//! for relevance scoring. Only entries scoring 7+ out of 10 are returned,
//! ordered by score.
//!
//! When Phase 1 returns no results, the MEMORY.md manifest/index is sent
//! to the LLM to identify relevant topics, which are then loaded and scored.
//!
//! ## Caching
//!
//! Scoring results are cached in-memory (HashMap) with a 1-hour TTL and
//! max 100 entries with LRU eviction. This avoids repeated API calls for
//! the same recall query during a session.
//!
//! ## Graceful degradation
//!
//! If the scoring API call fails (timeout, rate limit, error), Phase 2
//! falls back silently to keyword-only results. If no LLM provider is
//! configured, Phase 2 is skipped entirely.
//!
//! ## Cost
//!
//! Scoring calls count toward normal API usage/billing. The input is
//! capped at 4KB per call to limit cost.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::memory::memory_growth::extract_keywords;

/// Maximum bytes of memory entries sent to the scoring LLM per call.
const MAX_SCORING_INPUT_BYTES: usize = 4096;

/// Cache TTL: 1 hour.
const CACHE_TTL: Duration = Duration::from_secs(3600);

/// Maximum number of cached scoring results (LRU eviction beyond this).
const CACHE_MAX_ENTRIES: usize = 100;

/// System prompt for the scoring LLM call.
pub const SCORING_SYSTEM_PROMPT: &str = "You are a memory relevance scorer. \
    Given a query and a list of memory entries, score each 0-10 for relevance. \
    Return JSON: [{\"entry\": \"...\", \"score\": N}]. \
    Only include entries scoring 7 or above. Be generous — partial relevance counts.";

// ── Scored entry ─────────────────────────────────────────────────

/// A single memory entry with its relevance score from the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredEntry {
    /// The entry text (or a key identifying it).
    pub entry: String,
    /// Relevance score 0-10.
    pub score: u8,
}

// ── LRU Cache ────────────────────────────────────────────────────

/// In-memory LRU cache for scoring results.
///
/// Keys are hashes of (query + entries). Values are scored results.
/// Max 100 entries, 1-hour TTL, LRU eviction.
pub struct ScoringCache {
    entries: HashMap<u64, CacheEntry>,
    /// Tracks insertion/access order for LRU eviction. Most recent at end.
    access_order: Vec<u64>,
}

struct CacheEntry {
    results: Vec<ScoredEntry>,
    created_at: Instant,
}

impl ScoringCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            access_order: Vec::new(),
        }
    }

    /// Look up cached results. Returns None on miss or expiry.
    pub fn get(&mut self, key: u64) -> Option<Vec<ScoredEntry>> {
        if let Some(entry) = self.entries.get(&key) {
            if entry.created_at.elapsed() < CACHE_TTL {
                // Move to end (most recently used)
                self.access_order.retain(|k| *k != key);
                self.access_order.push(key);
                return Some(entry.results.clone());
            } else {
                // Expired — remove
                self.entries.remove(&key);
                self.access_order.retain(|k| *k != key);
            }
        }
        None
    }

    /// Insert results into the cache, evicting LRU entries if at capacity.
    pub fn put(&mut self, key: u64, results: Vec<ScoredEntry>) {
        // If key already exists, remove old position
        if self.entries.contains_key(&key) {
            self.access_order.retain(|k| *k != key);
        }

        // Evict LRU entries if at capacity
        while self.entries.len() >= CACHE_MAX_ENTRIES && !self.access_order.is_empty() {
            let oldest_key = self.access_order.remove(0);
            self.entries.remove(&oldest_key);
        }

        self.entries.insert(
            key,
            CacheEntry {
                results,
                created_at: Instant::now(),
            },
        );
        self.access_order.push(key);
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

// ── Semantic scorer ──────────────────────────────────────────────

/// Configuration for semantic search scoring.
#[derive(Debug, Clone)]
pub struct SemanticSearchConfig {
    /// Whether semantic search is enabled.
    pub enabled: bool,
    /// Model to use for scoring (empty = use primary model).
    pub scoring_model: String,
    /// Provider identifier (e.g., "anthropic", "openai", "openrouter").
    pub provider: String,
    /// Primary model name (used when scoring_model is empty).
    pub primary_model: String,
    /// API key for the provider.
    pub api_key: String,
    /// Optional base URL for compatible/custom providers.
    pub api_base_url: Option<String>,
}

/// LLM-based semantic scorer for memory entries.
///
/// Handles: LLM API calls, caching, input capping, and response parsing.
/// Thread-safe via interior Mutex on the cache.
pub struct SemanticScorer {
    config: SemanticSearchConfig,
    cache: Mutex<ScoringCache>,
    client: reqwest::Client,
}

impl SemanticScorer {
    /// Create a new scorer with the given configuration.
    pub fn new(config: SemanticSearchConfig) -> Self {
        Self {
            config,
            cache: Mutex::new(ScoringCache::new()),
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
        }
    }

    /// Create a scorer with a custom reqwest client (for testing).
    #[cfg(test)]
    pub fn with_client(config: SemanticSearchConfig, client: reqwest::Client) -> Self {
        Self {
            config,
            cache: Mutex::new(ScoringCache::new()),
            client,
        }
    }

    /// Whether semantic search is enabled in the config.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// The model used for scoring calls.
    pub fn model(&self) -> &str {
        if self.config.scoring_model.is_empty() {
            &self.config.primary_model
        } else {
            &self.config.scoring_model
        }
    }

    /// Check if Phase 2 should be skipped for these keyword grep results.
    ///
    /// Skip Phase 2 when there are 1-2 results and each contains an exact
    /// token match from the query. This avoids unnecessary API calls for
    /// obvious keyword matches.
    pub fn should_skip_phase2(query: &str, result_contents: &[&str]) -> bool {
        if result_contents.is_empty() || result_contents.len() > 2 {
            return false;
        }

        // Extract meaningful keywords from the query
        let keywords = extract_keywords(&[query.to_string()]);
        if keywords.is_empty() {
            return false;
        }

        // Check if every keyword appears in at least one result
        keywords.iter().all(|kw| {
            result_contents.iter().any(|content| {
                content.to_lowercase().contains(kw.as_str())
            })
        })
    }

    /// Score entries using the LLM. Returns scored entries (7+), or None on failure.
    ///
    /// Checks cache first. On API failure, returns None (graceful degradation).
    pub async fn score_entries(
        &self,
        query: &str,
        entries_text: &str,
    ) -> Option<Vec<ScoredEntry>> {
        if !self.is_enabled() || self.config.api_key.is_empty() {
            return None;
        }

        // Cap input size
        let capped = cap_input(entries_text, MAX_SCORING_INPUT_BYTES);

        // Check cache
        let cache_key = compute_cache_key(query, &capped);
        if let Ok(mut cache) = self.cache.lock() {
            if let Some(cached) = cache.get(cache_key) {
                return Some(cached);
            }
        }

        // Build user message and call LLM
        let user_message = build_scoring_user_message(query, &capped);
        let result = self.call_scoring_llm(&user_message).await;

        match result {
            Ok(scored) => {
                // Cache the results
                if let Ok(mut cache) = self.cache.lock() {
                    cache.put(cache_key, scored.clone());
                }
                Some(scored)
            }
            Err(e) => {
                eprintln!("[semantic_search] Scoring API call failed: {e}");
                None
            }
        }
    }

    /// Identify relevant topics from a manifest using the LLM.
    ///
    /// Used when keyword grep returns no results: the manifest is sent to
    /// the LLM to identify which topic files might contain relevant entries.
    pub async fn identify_relevant_topics(
        &self,
        query: &str,
        manifest_text: &str,
    ) -> Option<Vec<ScoredEntry>> {
        self.score_entries(query, manifest_text).await
    }

    /// Make the scoring LLM call, dispatching to the correct provider API.
    async fn call_scoring_llm(&self, user_message: &str) -> Result<Vec<ScoredEntry>> {
        let response_text = match self.config.provider.as_str() {
            "anthropic" => self.call_anthropic(user_message).await?,
            _ => self.call_openai_compatible(user_message).await?,
        };
        parse_scoring_response(&response_text)
    }

    /// Call the Anthropic Messages API.
    async fn call_anthropic(&self, user_message: &str) -> Result<String> {
        let body = serde_json::json!({
            "model": self.model(),
            "max_tokens": 1024,
            "system": SCORING_SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": user_message}
            ]
        });

        let resp = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Anthropic API error {status}: {text}");
        }

        let json: serde_json::Value = resp.json().await?;
        Ok(json["content"][0]["text"]
            .as_str()
            .unwrap_or("")
            .to_string())
    }

    /// Call an OpenAI-compatible chat completions API.
    ///
    /// Supports: openai, openrouter, ollama, compatible providers.
    async fn call_openai_compatible(&self, user_message: &str) -> Result<String> {
        let base_url = self
            .config
            .api_base_url
            .as_deref()
            .unwrap_or(match self.config.provider.as_str() {
                "openai" => "https://api.openai.com/v1",
                "openrouter" => "https://openrouter.ai/api/v1",
                "ollama" => "http://localhost:11434/v1",
                _ => "https://api.openai.com/v1",
            });

        let body = serde_json::json!({
            "model": self.model(),
            "max_tokens": 1024,
            "messages": [
                {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        });

        let resp = self
            .client
            .post(format!("{base_url}/chat/completions"))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("API error {status}: {text}");
        }

        let json: serde_json::Value = resp.json().await?;
        Ok(json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string())
    }
}

// ── Helper functions ─────────────────────────────────────────────

/// Build the user message for the scoring prompt.
pub fn build_scoring_user_message(query: &str, entries: &str) -> String {
    format!("Query: {query}\n\nMemory entries:\n{entries}")
}

/// Compute a cache key from query + entries text.
pub fn compute_cache_key(query: &str, entries: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    query.hash(&mut hasher);
    entries.hash(&mut hasher);
    hasher.finish()
}

/// Cap the input text at `max_bytes`, truncating at a line boundary.
pub fn cap_input(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let mut truncated = String::new();
    for line in text.lines() {
        if truncated.len() + line.len() + 1 > max_bytes {
            break;
        }
        if !truncated.is_empty() {
            truncated.push('\n');
        }
        truncated.push_str(line);
    }
    truncated
}

/// Parse the LLM's scoring response into `ScoredEntry` items.
///
/// The LLM is expected to return JSON: `[{"entry": "...", "score": N}]`.
/// Handles markdown code blocks and extra text around the JSON.
/// Filters to entries scoring 7+.
pub fn parse_scoring_response(response: &str) -> Result<Vec<ScoredEntry>> {
    let json_str = extract_json_array(response);
    let entries: Vec<ScoredEntry> = serde_json::from_str(&json_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse scoring response as JSON: {e}"))?;
    Ok(entries.into_iter().filter(|e| e.score >= 7).collect())
}

/// Extract a JSON array from text that may contain markdown or extra content.
fn extract_json_array(text: &str) -> String {
    // Try to find JSON array brackets
    if let Some(start) = text.find('[') {
        if let Some(end) = text.rfind(']') {
            if end > start {
                return text[start..=end].to_string();
            }
        }
    }
    text.to_string()
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Scoring prompt construction tests ─────────────────────

    #[test]
    fn test_build_scoring_user_message() {
        let msg = build_scoring_user_message("docker setup", "- [2026-01-01] Docker config\n- [2026-01-02] Rust build");
        assert!(msg.contains("Query: docker setup"));
        assert!(msg.contains("Memory entries:"));
        assert!(msg.contains("Docker config"));
        assert!(msg.contains("Rust build"));
    }

    #[test]
    fn test_scoring_system_prompt_contains_required_elements() {
        assert!(SCORING_SYSTEM_PROMPT.contains("score each 0-10"));
        assert!(SCORING_SYSTEM_PROMPT.contains("7 or above"));
        assert!(SCORING_SYSTEM_PROMPT.contains("JSON"));
        assert!(SCORING_SYSTEM_PROMPT.contains("generous"));
    }

    // ── Input capping tests ──────────────────────────────────

    #[test]
    fn test_cap_input_under_limit() {
        let text = "line one\nline two\nline three";
        let capped = cap_input(text, 4096);
        assert_eq!(capped, text);
    }

    #[test]
    fn test_cap_input_over_limit() {
        let line = "a".repeat(100);
        let text = format!("{line}\n{line}\n{line}\n{line}\n{line}");
        let capped = cap_input(&text, 250);
        // Should fit ~2 lines (100 + 1 + 100 = 201, third would be 302)
        assert!(capped.len() <= 250);
        assert!(capped.contains(&"a".repeat(100)));
    }

    #[test]
    fn test_cap_input_truncates_at_line_boundary() {
        let text = "short line\nthis is a much longer line that pushes us over the limit";
        let capped = cap_input(text, 20);
        assert_eq!(capped, "short line");
    }

    // ── Response parsing tests ───────────────────────────────

    #[test]
    fn test_parse_scoring_response_valid_json() {
        let response = r#"[{"entry": "Docker config", "score": 9}, {"entry": "Rust build", "score": 5}]"#;
        let results = parse_scoring_response(response).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entry, "Docker config");
        assert_eq!(results[0].score, 9);
    }

    #[test]
    fn test_parse_scoring_response_filters_below_7() {
        let response = r#"[
            {"entry": "High relevance", "score": 10},
            {"entry": "Medium relevance", "score": 7},
            {"entry": "Low relevance", "score": 6},
            {"entry": "No relevance", "score": 2}
        ]"#;
        let results = parse_scoring_response(response).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.score >= 7));
    }

    #[test]
    fn test_parse_scoring_response_with_markdown_wrapper() {
        let response = "Here are the scores:\n```json\n[{\"entry\": \"Docker\", \"score\": 8}]\n```\nDone.";
        let results = parse_scoring_response(response).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entry, "Docker");
    }

    #[test]
    fn test_parse_scoring_response_empty_array() {
        let response = "[]";
        let results = parse_scoring_response(response).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_scoring_response_invalid_json() {
        let response = "not json at all";
        let result = parse_scoring_response(response);
        assert!(result.is_err());
    }

    // ── Cache tests ──────────────────────────────────────────

    #[test]
    fn test_cache_miss_returns_none() {
        let mut cache = ScoringCache::new();
        assert!(cache.get(12345).is_none());
    }

    #[test]
    fn test_cache_hit_returns_results() {
        let mut cache = ScoringCache::new();
        let entries = vec![ScoredEntry {
            entry: "test".to_string(),
            score: 8,
        }];
        cache.put(42, entries.clone());
        let cached = cache.get(42).unwrap();
        assert_eq!(cached.len(), 1);
        assert_eq!(cached[0].entry, "test");
    }

    #[test]
    fn test_cache_lru_eviction() {
        let mut cache = ScoringCache::new();

        // Fill cache to capacity
        for i in 0..CACHE_MAX_ENTRIES {
            cache.put(
                i as u64,
                vec![ScoredEntry {
                    entry: format!("entry-{i}"),
                    score: 8,
                }],
            );
        }
        assert_eq!(cache.len(), CACHE_MAX_ENTRIES);

        // Adding one more should evict the oldest (key=0)
        cache.put(
            999,
            vec![ScoredEntry {
                entry: "new".to_string(),
                score: 9,
            }],
        );
        assert_eq!(cache.len(), CACHE_MAX_ENTRIES);
        assert!(cache.get(0).is_none(), "Oldest entry should be evicted");
        assert!(cache.get(999).is_some(), "New entry should be present");
    }

    #[test]
    fn test_cache_lru_access_updates_order() {
        let mut cache = ScoringCache::new();

        // Fill cache to capacity
        for i in 0..CACHE_MAX_ENTRIES {
            cache.put(
                i as u64,
                vec![ScoredEntry {
                    entry: format!("entry-{i}"),
                    score: 8,
                }],
            );
        }

        // Access key=0, making it most-recently-used
        cache.get(0);

        // Adding a new entry should evict key=1 (now the LRU), not key=0
        cache.put(
            999,
            vec![ScoredEntry {
                entry: "new".to_string(),
                score: 9,
            }],
        );

        assert!(cache.get(0).is_some(), "Recently accessed entry should survive");
        assert!(cache.get(1).is_none(), "LRU entry should be evicted");
    }

    #[test]
    fn test_cache_key_computation() {
        let key1 = compute_cache_key("query1", "entries1");
        let key2 = compute_cache_key("query1", "entries1");
        let key3 = compute_cache_key("query2", "entries1");

        assert_eq!(key1, key2, "Same inputs should produce same key");
        assert_ne!(key1, key3, "Different inputs should produce different keys");
    }

    // ── Skip Phase 2 tests ──────────────────────────────────

    #[test]
    fn test_should_skip_phase2_exact_match() {
        // 1 result that contains the query keyword → skip
        let contents = vec!["Docker sandbox has slow startup on Pi Zero"];
        assert!(SemanticScorer::should_skip_phase2("docker sandbox", &contents));
    }

    #[test]
    fn test_should_skip_phase2_two_results_exact() {
        // 2 results both covering query keywords → skip
        let contents = vec![
            "Add Docker sandbox support",
            "Docker sandbox has slow startup",
        ];
        assert!(SemanticScorer::should_skip_phase2("docker sandbox", &contents));
    }

    #[test]
    fn test_should_skip_phase2_too_many_results() {
        // >2 results → don't skip (worth scoring)
        let contents = vec!["result1 docker", "result2 docker", "result3 docker"];
        assert!(!SemanticScorer::should_skip_phase2("docker", &contents));
    }

    #[test]
    fn test_should_skip_phase2_empty_results() {
        // 0 results → don't skip (need manifest fallback)
        let contents: Vec<&str> = vec![];
        assert!(!SemanticScorer::should_skip_phase2("docker", &contents));
    }

    #[test]
    fn test_should_skip_phase2_partial_match() {
        // 1 result that only partially matches → don't skip
        let contents = vec!["Docker sandbox has slow startup"];
        assert!(!SemanticScorer::should_skip_phase2("docker memory", &contents));
    }

    // ── Scorer config tests ─────────────────────────────────

    #[test]
    fn test_scorer_model_uses_scoring_model_when_set() {
        let config = SemanticSearchConfig {
            enabled: true,
            scoring_model: "claude-haiku-3".to_string(),
            provider: "anthropic".to_string(),
            primary_model: "claude-sonnet-4-20250514".to_string(),
            api_key: "test".to_string(),
            api_base_url: None,
        };
        let scorer = SemanticScorer::new(config);
        assert_eq!(scorer.model(), "claude-haiku-3");
    }

    #[test]
    fn test_scorer_model_falls_back_to_primary() {
        let config = SemanticSearchConfig {
            enabled: true,
            scoring_model: String::new(),
            provider: "anthropic".to_string(),
            primary_model: "claude-sonnet-4-20250514".to_string(),
            api_key: "test".to_string(),
            api_base_url: None,
        };
        let scorer = SemanticScorer::new(config);
        assert_eq!(scorer.model(), "claude-sonnet-4-20250514");
    }

    #[test]
    fn test_scorer_disabled_when_config_false() {
        let config = SemanticSearchConfig {
            enabled: false,
            scoring_model: String::new(),
            provider: "anthropic".to_string(),
            primary_model: "claude-sonnet-4-20250514".to_string(),
            api_key: "test".to_string(),
            api_base_url: None,
        };
        let scorer = SemanticScorer::new(config);
        assert!(!scorer.is_enabled());
    }

    // ── Graceful degradation test (no API key) ──────────────

    #[tokio::test]
    async fn test_score_entries_returns_none_when_disabled() {
        let config = SemanticSearchConfig {
            enabled: false,
            scoring_model: String::new(),
            provider: "anthropic".to_string(),
            primary_model: "claude-sonnet-4-20250514".to_string(),
            api_key: "test-key".to_string(),
            api_base_url: None,
        };
        let scorer = SemanticScorer::new(config);
        let result = scorer
            .score_entries("docker", "- Docker config entry")
            .await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_score_entries_returns_none_when_no_api_key() {
        let config = SemanticSearchConfig {
            enabled: true,
            scoring_model: String::new(),
            provider: "anthropic".to_string(),
            primary_model: "claude-sonnet-4-20250514".to_string(),
            api_key: String::new(),
            api_base_url: None,
        };
        let scorer = SemanticScorer::new(config);
        let result = scorer
            .score_entries("docker", "- Docker config entry")
            .await;
        assert!(result.is_none());
    }

    // ── Graceful degradation on API failure ──────────────────

    #[tokio::test]
    async fn test_score_entries_returns_none_on_api_failure() {
        // Use a non-existent URL to simulate connection failure
        let config = SemanticSearchConfig {
            enabled: true,
            scoring_model: String::new(),
            provider: "anthropic".to_string(),
            primary_model: "claude-sonnet-4-20250514".to_string(),
            api_key: "fake-key".to_string(),
            api_base_url: None,
        };
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(100))
            .build()
            .unwrap();
        let scorer = SemanticScorer::with_client(config, client);
        let result = scorer
            .score_entries("docker", "- Docker config entry")
            .await;
        // Should return None, not panic or propagate error
        assert!(result.is_none());
    }

    // ── Extract JSON array tests ─────────────────────────────

    #[test]
    fn test_extract_json_array_clean() {
        let text = r#"[{"entry": "test", "score": 8}]"#;
        assert_eq!(extract_json_array(text), text);
    }

    #[test]
    fn test_extract_json_array_with_surrounding_text() {
        let text = "Here are the results:\n[{\"entry\": \"test\", \"score\": 8}]\nEnd.";
        let extracted = extract_json_array(text);
        assert!(extracted.starts_with('['));
        assert!(extracted.ends_with(']'));
    }

    #[test]
    fn test_extract_json_array_no_brackets() {
        let text = "no json here";
        assert_eq!(extract_json_array(text), text);
    }
}

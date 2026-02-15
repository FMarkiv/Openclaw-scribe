//! Model failover for ZeroClaw.
//!
//! Provides automatic failover across multiple LLM providers when the
//! primary is down or rate-limited. Fallbacks are tried in order; on the
//! next user turn the primary is tried again first.
//!
//! ## Config
//!
//! ```toml
//! [provider]
//! primary = "anthropic"
//! model = "claude-sonnet-4-20250514"
//!
//! [[provider.fallbacks]]
//! provider = "openai"
//! model = "gpt-4o"
//!
//! [[provider.fallbacks]]
//! provider = "openrouter"
//! model = "anthropic/claude-sonnet-4-20250514"
//! ```
//!
//! ## Error classification
//!
//! | HTTP status       | Classification  | Action                |
//! |-------------------|-----------------|-----------------------|
//! | 429               | Retryable       | Try next fallback     |
//! | 500, 502, 503, 529 | Retryable     | Try next fallback     |
//! | 401, 403          | Non-retryable   | Skip provider         |
//! | Connection/timeout| Retryable       | Try next fallback     |
//! | Context overflow  | Non-retryable   | Trigger compaction    |

use crate::memory::context::ContextManager;
use serde::{Deserialize, Serialize};
use std::fmt;

// ── Provider configuration ──────────────────────────────────────

/// Configuration for a single LLM provider.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProviderConfig {
    /// Provider identifier (e.g., "anthropic", "openai", "openrouter").
    pub provider: String,
    /// Model name (e.g., "claude-sonnet-4-20250514", "gpt-4o").
    pub model: String,
}

impl fmt::Display for ProviderConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.provider, self.model)
    }
}

/// Full provider chain: primary + ordered fallbacks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderChain {
    /// The primary provider — tried first on every turn.
    pub primary: ProviderConfig,
    /// Ordered fallback providers — tried in sequence when primary fails.
    #[serde(default)]
    pub fallbacks: Vec<ProviderConfig>,
}

impl ProviderChain {
    /// Create a chain with only a primary (no fallbacks).
    pub fn primary_only(provider: &str, model: &str) -> Self {
        Self {
            primary: ProviderConfig {
                provider: provider.to_string(),
                model: model.to_string(),
            },
            fallbacks: Vec::new(),
        }
    }

    /// Total number of providers in the chain (primary + fallbacks).
    pub fn len(&self) -> usize {
        1 + self.fallbacks.len()
    }

    /// Whether any fallbacks are configured.
    pub fn has_fallbacks(&self) -> bool {
        !self.fallbacks.is_empty()
    }

    /// Iterate over all providers: primary first, then fallbacks in order.
    pub fn iter(&self) -> impl Iterator<Item = &ProviderConfig> {
        std::iter::once(&self.primary).chain(self.fallbacks.iter())
    }
}

// ── Error classification ────────────────────────────────────────

/// Classification of an API call error for failover purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    /// Error is retryable via failover (rate limit, server error, network).
    Retryable,
    /// Error is an auth failure — skip this provider entirely.
    AuthFailure,
    /// Error is a context overflow — trigger compaction, not failover.
    ContextOverflow,
    /// Error is non-retryable for other reasons.
    NonRetryable,
}

/// An API error with enough information for failover decisions.
#[derive(Debug, Clone)]
pub struct ApiError {
    /// HTTP status code, if available.
    pub status_code: Option<u16>,
    /// Error message from the provider.
    pub message: String,
    /// Whether this was a connection/timeout error (no HTTP response).
    pub is_connection_error: bool,
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(code) = self.status_code {
            write!(f, "HTTP {} — {}", code, self.message)
        } else if self.is_connection_error {
            write!(f, "Connection error — {}", self.message)
        } else {
            write!(f, "{}", self.message)
        }
    }
}

impl ApiError {
    /// Classify this error for failover routing.
    pub fn classify(&self) -> ErrorClass {
        // Context overflow takes priority — never failover, always compact.
        if ContextManager::is_context_overflow_error(&self.message) {
            return ErrorClass::ContextOverflow;
        }

        // Auth failures are non-retryable for this provider.
        if let Some(code) = self.status_code {
            if code == 401 || code == 403 {
                return ErrorClass::AuthFailure;
            }
        }

        // Retryable HTTP status codes.
        if let Some(code) = self.status_code {
            if matches!(code, 429 | 500 | 502 | 503 | 529) {
                return ErrorClass::Retryable;
            }
        }

        // Connection/network errors are retryable.
        if self.is_connection_error {
            return ErrorClass::Retryable;
        }

        ErrorClass::NonRetryable
    }

    /// A short human-readable reason string for log messages.
    pub fn short_reason(&self) -> String {
        if let Some(code) = self.status_code {
            let label = match code {
                429 => "Rate Limited",
                500 => "Internal Server Error",
                502 => "Bad Gateway",
                503 => "Service Unavailable",
                529 => "Overloaded",
                401 => "Unauthorized",
                403 => "Forbidden",
                _ => "HTTP Error",
            };
            format!("{} {}", code, label)
        } else if self.is_connection_error {
            "Connection Error".to_string()
        } else {
            "Unknown Error".to_string()
        }
    }
}

// ── Failover event ──────────────────────────────────────────────

/// Record of a failover event, for logging and diagnostics.
#[derive(Debug, Clone)]
pub struct FailoverEvent {
    /// The provider that failed.
    pub from_provider: String,
    /// The provider we're failing over to.
    pub to_provider: String,
    /// Short reason for the failover.
    pub reason: String,
}

impl fmt::Display for FailoverEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Failover: {} -> {} ({})",
            self.from_provider, self.to_provider, self.reason
        )
    }
}

/// Format a failover event for the daily note log.
pub fn format_failover_log(event: &FailoverEvent) -> String {
    format!(
        "\u{26a0}\u{fe0f} Failover: {} \u{2192} {} ({})",
        event.from_provider, event.to_provider, event.reason
    )
}

// ── Failover result ─────────────────────────────────────────────

/// The outcome of a failover-wrapped API call attempt.
#[derive(Debug)]
pub enum FailoverOutcome<T> {
    /// Call succeeded (possibly after failover).
    Success {
        /// The result of the successful call.
        result: T,
        /// Which provider handled the call.
        provider: ProviderConfig,
        /// Failover events that occurred before success (empty if primary succeeded).
        failover_events: Vec<FailoverEvent>,
    },
    /// All providers failed.
    AllFailed {
        /// The last error encountered.
        last_error: ApiError,
        /// All failover events that occurred.
        failover_events: Vec<FailoverEvent>,
    },
    /// Context overflow detected — caller should trigger compaction.
    ContextOverflow {
        /// The original error.
        error: ApiError,
    },
}

// ── FailoverManager ─────────────────────────────────────────────

/// Manages failover across a chain of LLM providers.
///
/// On each user turn, the manager resets to the primary provider.
/// When an API call fails with a retryable error, it advances to the
/// next provider in the chain. Auth failures skip the provider
/// entirely. Context overflow errors are routed to compaction instead.
pub struct FailoverManager {
    /// The configured provider chain.
    chain: ProviderChain,
    /// Failover events from the current turn.
    current_turn_events: Vec<FailoverEvent>,
}

impl FailoverManager {
    /// Create a new FailoverManager for the given provider chain.
    pub fn new(chain: ProviderChain) -> Self {
        Self {
            chain,
            current_turn_events: Vec::new(),
        }
    }

    /// The configured provider chain.
    pub fn chain(&self) -> &ProviderChain {
        &self.chain
    }

    /// The primary provider.
    pub fn primary(&self) -> &ProviderConfig {
        &self.chain.primary
    }

    /// Reset to primary for the next user turn.
    ///
    /// Call this at the start of each new user turn so we always
    /// try the primary first (don't permanently switch to fallback).
    pub fn reset_for_new_turn(&mut self) {
        self.current_turn_events.clear();
    }

    /// Failover events from the current turn.
    pub fn current_turn_events(&self) -> &[FailoverEvent] {
        &self.current_turn_events
    }

    /// Attempt an API call with automatic failover.
    ///
    /// The `call_fn` closure receives a `&ProviderConfig` and returns
    /// `Result<T, ApiError>`. The manager tries each provider in the
    /// chain until one succeeds or all fail.
    ///
    /// Returns a `FailoverOutcome` describing what happened.
    pub async fn call_with_failover<T, F, Fut>(
        &mut self,
        call_fn: F,
    ) -> FailoverOutcome<T>
    where
        F: Fn(&ProviderConfig) -> Fut,
        Fut: std::future::Future<Output = Result<T, ApiError>>,
    {
        let providers: Vec<ProviderConfig> = self.chain.iter().cloned().collect();
        let mut last_error: Option<ApiError> = None;

        for (i, provider) in providers.iter().enumerate() {
            match call_fn(provider).await {
                Ok(result) => {
                    return FailoverOutcome::Success {
                        result,
                        provider: provider.clone(),
                        failover_events: self.current_turn_events.clone(),
                    };
                }
                Err(err) => {
                    match err.classify() {
                        ErrorClass::ContextOverflow => {
                            return FailoverOutcome::ContextOverflow { error: err };
                        }
                        ErrorClass::Retryable | ErrorClass::AuthFailure => {
                            // Log the failover event if there's a next provider.
                            if i + 1 < providers.len() {
                                let event = FailoverEvent {
                                    from_provider: provider.provider.clone(),
                                    to_provider: providers[i + 1].provider.clone(),
                                    reason: err.short_reason(),
                                };
                                eprintln!(
                                    "[failover] Primary provider {} failed ({}), falling back to {}",
                                    provider.provider,
                                    err.short_reason(),
                                    providers[i + 1].provider
                                );
                                self.current_turn_events.push(event);
                            }
                            last_error = Some(err);
                        }
                        ErrorClass::NonRetryable => {
                            // Non-retryable and not context overflow — stop trying.
                            return FailoverOutcome::AllFailed {
                                last_error: err,
                                failover_events: self.current_turn_events.clone(),
                            };
                        }
                    }
                }
            }
        }

        // All providers exhausted.
        FailoverOutcome::AllFailed {
            last_error: last_error.unwrap_or(ApiError {
                status_code: None,
                message: "No providers configured".to_string(),
                is_connection_error: false,
            }),
            failover_events: self.current_turn_events.clone(),
        }
    }

    /// Format the current provider status for the /provider command.
    pub fn format_provider_status(&self) -> String {
        let mut output = String::new();

        output.push_str("Provider Configuration\n");
        output.push_str(&"=".repeat(40));
        output.push('\n');

        output.push_str(&format!(
            "\nPrimary:  {} (model: {})\n",
            self.chain.primary.provider, self.chain.primary.model
        ));

        if self.chain.fallbacks.is_empty() {
            output.push_str("Fallbacks: none\n");
        } else {
            output.push_str("\nFallbacks (in order):\n");
            for (i, fb) in self.chain.fallbacks.iter().enumerate() {
                output.push_str(&format!(
                    "  {}. {} (model: {})\n",
                    i + 1,
                    fb.provider,
                    fb.model
                ));
            }
        }

        if !self.current_turn_events.is_empty() {
            output.push_str("\nRecent failover events (this turn):\n");
            for event in &self.current_turn_events {
                output.push_str(&format!("  - {}\n", event));
            }
        }

        output
    }
}

// ── Config parsing ──────────────────────────────────────────────

/// TOML-compatible structure for the [provider] section with fallbacks.
#[derive(Debug, Clone, Deserialize)]
pub struct ProviderTomlConfig {
    /// Primary provider name (e.g., "anthropic").
    #[serde(default)]
    pub primary: Option<String>,
    /// Alias for backwards-compat: "name" in old config.
    #[serde(default)]
    pub name: Option<String>,
    /// API key (or set via env var).
    #[serde(default)]
    pub api_key: Option<String>,
    /// Primary model name.
    #[serde(default)]
    pub model: Option<String>,
    /// Ordered list of fallback providers.
    #[serde(default)]
    pub fallbacks: Vec<FallbackTomlConfig>,
}

/// TOML-compatible structure for a [[provider.fallbacks]] entry.
#[derive(Debug, Clone, Deserialize)]
pub struct FallbackTomlConfig {
    /// Provider name.
    pub provider: String,
    /// Model name.
    pub model: String,
}

impl ProviderTomlConfig {
    /// Convert the TOML config into a ProviderChain.
    ///
    /// Falls back to defaults when fields are missing:
    /// - Provider defaults to "anthropic"
    /// - Model defaults to "claude-sonnet-4-20250514"
    pub fn into_chain(self) -> ProviderChain {
        let provider_name = self
            .primary
            .or(self.name)
            .unwrap_or_else(|| "anthropic".to_string());

        let model_name = self
            .model
            .unwrap_or_else(|| "claude-sonnet-4-20250514".to_string());

        let fallbacks: Vec<ProviderConfig> = self
            .fallbacks
            .into_iter()
            .map(|fb| ProviderConfig {
                provider: fb.provider,
                model: fb.model,
            })
            .collect();

        ProviderChain {
            primary: ProviderConfig {
                provider: provider_name,
                model: model_name,
            },
            fallbacks,
        }
    }
}

/// Parse a ProviderChain from a TOML config string.
///
/// Expects the full config file contents. If the `[provider]` section
/// is missing or empty, returns a default Anthropic chain with no
/// fallbacks.
pub fn parse_provider_chain(toml_str: &str) -> Result<ProviderChain, toml::de::Error> {
    /// Wrapper to extract the [provider] section from the full config.
    #[derive(Deserialize)]
    struct Wrapper {
        #[serde(default)]
        provider: Option<ProviderTomlConfig>,
    }

    let wrapper: Wrapper = toml::from_str(toml_str)?;

    Ok(match wrapper.provider {
        Some(cfg) => cfg.into_chain(),
        None => ProviderChain::primary_only("anthropic", "claude-sonnet-4-20250514"),
    })
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ProviderConfig / ProviderChain tests ────────────────────

    #[test]
    fn provider_config_display() {
        let cfg = ProviderConfig {
            provider: "anthropic".to_string(),
            model: "claude-sonnet-4-20250514".to_string(),
        };
        assert_eq!(cfg.to_string(), "anthropic:claude-sonnet-4-20250514");
    }

    #[test]
    fn chain_primary_only() {
        let chain = ProviderChain::primary_only("anthropic", "claude-sonnet-4-20250514");
        assert_eq!(chain.len(), 1);
        assert!(!chain.has_fallbacks());
        assert_eq!(chain.primary.provider, "anthropic");
    }

    #[test]
    fn chain_with_fallbacks() {
        let chain = ProviderChain {
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
        };

        assert_eq!(chain.len(), 3);
        assert!(chain.has_fallbacks());

        let names: Vec<&str> = chain.iter().map(|p| p.provider.as_str()).collect();
        assert_eq!(names, vec!["anthropic", "openai", "openrouter"]);
    }

    // ── Error classification tests ──────────────────────────────

    #[test]
    fn classify_rate_limit_as_retryable() {
        let err = ApiError {
            status_code: Some(429),
            message: "Rate limit exceeded".to_string(),
            is_connection_error: false,
        };
        assert_eq!(err.classify(), ErrorClass::Retryable);
    }

    #[test]
    fn classify_server_errors_as_retryable() {
        for code in [500, 502, 503, 529] {
            let err = ApiError {
                status_code: Some(code),
                message: format!("Server error {code}"),
                is_connection_error: false,
            };
            assert_eq!(err.classify(), ErrorClass::Retryable, "HTTP {code} should be retryable");
        }
    }

    #[test]
    fn classify_auth_errors_as_auth_failure() {
        for code in [401, 403] {
            let err = ApiError {
                status_code: Some(code),
                message: format!("Auth error {code}"),
                is_connection_error: false,
            };
            assert_eq!(err.classify(), ErrorClass::AuthFailure, "HTTP {code} should be auth failure");
        }
    }

    #[test]
    fn classify_connection_error_as_retryable() {
        let err = ApiError {
            status_code: None,
            message: "Connection timed out".to_string(),
            is_connection_error: true,
        };
        assert_eq!(err.classify(), ErrorClass::Retryable);
    }

    #[test]
    fn classify_context_overflow_not_retryable() {
        let err = ApiError {
            status_code: Some(400),
            message: "context_length_exceeded: max 200000 tokens".to_string(),
            is_connection_error: false,
        };
        assert_eq!(err.classify(), ErrorClass::ContextOverflow);
    }

    #[test]
    fn classify_context_overflow_takes_priority_over_status_code() {
        // Even if the status code is retryable (e.g., a weird 500 with overflow msg)
        let err = ApiError {
            status_code: Some(500),
            message: "too many tokens in request".to_string(),
            is_connection_error: false,
        };
        assert_eq!(err.classify(), ErrorClass::ContextOverflow);
    }

    #[test]
    fn classify_unknown_error_as_non_retryable() {
        let err = ApiError {
            status_code: Some(400),
            message: "Invalid request body".to_string(),
            is_connection_error: false,
        };
        assert_eq!(err.classify(), ErrorClass::NonRetryable);
    }

    #[test]
    fn short_reason_for_known_codes() {
        assert!(ApiError {
            status_code: Some(429),
            message: String::new(),
            is_connection_error: false,
        }
        .short_reason()
        .contains("Rate Limited"));

        assert!(ApiError {
            status_code: Some(503),
            message: String::new(),
            is_connection_error: false,
        }
        .short_reason()
        .contains("Service Unavailable"));

        assert!(ApiError {
            status_code: None,
            message: String::new(),
            is_connection_error: true,
        }
        .short_reason()
        .contains("Connection Error"));
    }

    // ── Failover event formatting ───────────────────────────────

    #[test]
    fn failover_event_display() {
        let event = FailoverEvent {
            from_provider: "anthropic".to_string(),
            to_provider: "openai".to_string(),
            reason: "503 Service Unavailable".to_string(),
        };
        let s = event.to_string();
        assert!(s.contains("anthropic"));
        assert!(s.contains("openai"));
        assert!(s.contains("503"));
    }

    #[test]
    fn format_failover_log_for_daily_note() {
        let event = FailoverEvent {
            from_provider: "anthropic".to_string(),
            to_provider: "openai".to_string(),
            reason: "503 Service Unavailable".to_string(),
        };
        let log = format_failover_log(&event);
        assert!(log.contains("Failover"));
        assert!(log.contains("anthropic"));
        assert!(log.contains("openai"));
        assert!(log.contains("503"));
    }

    // ── FailoverManager tests ───────────────────────────────────

    #[tokio::test]
    async fn successful_primary_no_failover() {
        let chain = ProviderChain::primary_only("anthropic", "claude-sonnet-4-20250514");
        let mut mgr = FailoverManager::new(chain);

        let outcome = mgr
            .call_with_failover(|_provider| async { Ok::<&str, ApiError>("response") })
            .await;

        match outcome {
            FailoverOutcome::Success {
                result,
                provider,
                failover_events,
            } => {
                assert_eq!(result, "response");
                assert_eq!(provider.provider, "anthropic");
                assert!(failover_events.is_empty());
            }
            _ => panic!("Expected Success"),
        }
    }

    #[tokio::test]
    async fn failover_from_primary_to_first_fallback() {
        let chain = ProviderChain {
            primary: ProviderConfig {
                provider: "anthropic".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            fallbacks: vec![ProviderConfig {
                provider: "openai".to_string(),
                model: "gpt-4o".to_string(),
            }],
        };
        let mut mgr = FailoverManager::new(chain);

        let outcome = mgr
            .call_with_failover(|provider| {
                let provider_name = provider.provider.clone();
                async move {
                    if provider_name == "anthropic" {
                        Err(ApiError {
                            status_code: Some(503),
                            message: "Service Unavailable".to_string(),
                            is_connection_error: false,
                        })
                    } else {
                        Ok("fallback response")
                    }
                }
            })
            .await;

        match outcome {
            FailoverOutcome::Success {
                result,
                provider,
                failover_events,
            } => {
                assert_eq!(result, "fallback response");
                assert_eq!(provider.provider, "openai");
                assert_eq!(failover_events.len(), 1);
                assert_eq!(failover_events[0].from_provider, "anthropic");
                assert_eq!(failover_events[0].to_provider, "openai");
            }
            _ => panic!("Expected Success"),
        }
    }

    #[tokio::test]
    async fn skips_auth_failure_to_next_fallback() {
        let chain = ProviderChain {
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
        };
        let mut mgr = FailoverManager::new(chain);

        let outcome = mgr
            .call_with_failover(|provider| {
                let name = provider.provider.clone();
                async move {
                    match name.as_str() {
                        "anthropic" => Err(ApiError {
                            status_code: Some(401),
                            message: "Unauthorized".to_string(),
                            is_connection_error: false,
                        }),
                        "openai" => Err(ApiError {
                            status_code: Some(403),
                            message: "Forbidden".to_string(),
                            is_connection_error: false,
                        }),
                        _ => Ok("openrouter works"),
                    }
                }
            })
            .await;

        match outcome {
            FailoverOutcome::Success {
                result,
                provider,
                failover_events,
            } => {
                assert_eq!(result, "openrouter works");
                assert_eq!(provider.provider, "openrouter");
                assert_eq!(failover_events.len(), 2);
            }
            _ => panic!("Expected Success"),
        }
    }

    #[tokio::test]
    async fn all_providers_fail_returns_last_error() {
        let chain = ProviderChain {
            primary: ProviderConfig {
                provider: "anthropic".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            fallbacks: vec![ProviderConfig {
                provider: "openai".to_string(),
                model: "gpt-4o".to_string(),
            }],
        };
        let mut mgr = FailoverManager::new(chain);

        let outcome: FailoverOutcome<&str> = mgr
            .call_with_failover(|provider| {
                let name = provider.provider.clone();
                async move {
                    Err(ApiError {
                        status_code: Some(503),
                        message: format!("{name} is down"),
                        is_connection_error: false,
                    })
                }
            })
            .await;

        match outcome {
            FailoverOutcome::AllFailed {
                last_error,
                failover_events,
            } => {
                assert!(last_error.message.contains("openai"));
                assert_eq!(failover_events.len(), 1);
            }
            _ => panic!("Expected AllFailed"),
        }
    }

    #[tokio::test]
    async fn resets_to_primary_on_next_turn() {
        let chain = ProviderChain {
            primary: ProviderConfig {
                provider: "anthropic".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            fallbacks: vec![ProviderConfig {
                provider: "openai".to_string(),
                model: "gpt-4o".to_string(),
            }],
        };
        let mut mgr = FailoverManager::new(chain);

        // First turn: primary fails, fallback succeeds
        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cc = call_count.clone();

        let _outcome = mgr
            .call_with_failover(|provider| {
                let name = provider.provider.clone();
                let cc = cc.clone();
                async move {
                    cc.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if name == "anthropic" {
                        Err(ApiError {
                            status_code: Some(429),
                            message: "rate limited".to_string(),
                            is_connection_error: false,
                        })
                    } else {
                        Ok("ok")
                    }
                }
            })
            .await;

        assert_eq!(mgr.current_turn_events().len(), 1);

        // Reset for new turn
        mgr.reset_for_new_turn();
        assert!(mgr.current_turn_events().is_empty());

        // Second turn: tries primary again first
        let providers_tried = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let pt = providers_tried.clone();

        let _outcome = mgr
            .call_with_failover(|provider| {
                let name = provider.provider.clone();
                let pt = pt.clone();
                async move {
                    pt.lock().unwrap().push(name.clone());
                    Ok::<&str, ApiError>("ok")
                }
            })
            .await;

        let tried = providers_tried.lock().unwrap();
        // Should have tried primary first (and succeeded)
        assert_eq!(tried[0], "anthropic");
        assert_eq!(tried.len(), 1); // Only primary was needed
    }

    #[tokio::test]
    async fn context_overflow_triggers_compaction_not_failover() {
        let chain = ProviderChain {
            primary: ProviderConfig {
                provider: "anthropic".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            fallbacks: vec![ProviderConfig {
                provider: "openai".to_string(),
                model: "gpt-4o".to_string(),
            }],
        };
        let mut mgr = FailoverManager::new(chain);

        let outcome: FailoverOutcome<&str> = mgr
            .call_with_failover(|_provider| async {
                Err(ApiError {
                    status_code: Some(400),
                    message: "context_length_exceeded: max 200000 tokens".to_string(),
                    is_connection_error: false,
                })
            })
            .await;

        match outcome {
            FailoverOutcome::ContextOverflow { error } => {
                assert!(error.message.contains("context_length_exceeded"));
            }
            _ => panic!("Expected ContextOverflow, got {:?}", std::mem::discriminant(&outcome)),
        }

        // No failover events should have been generated
        assert!(mgr.current_turn_events().is_empty());
    }

    #[tokio::test]
    async fn no_fallbacks_original_behavior() {
        let chain = ProviderChain::primary_only("anthropic", "claude-sonnet-4-20250514");
        let mut mgr = FailoverManager::new(chain);

        // Success case: works normally
        let outcome = mgr
            .call_with_failover(|_p| async { Ok::<_, ApiError>("ok") })
            .await;
        assert!(matches!(outcome, FailoverOutcome::Success { .. }));

        // Failure case: returns error immediately (no fallback)
        let outcome: FailoverOutcome<&str> = mgr
            .call_with_failover(|_p| async {
                Err(ApiError {
                    status_code: Some(503),
                    message: "down".to_string(),
                    is_connection_error: false,
                })
            })
            .await;
        match outcome {
            FailoverOutcome::AllFailed {
                failover_events, ..
            } => {
                assert!(failover_events.is_empty());
            }
            _ => panic!("Expected AllFailed"),
        }
    }

    #[tokio::test]
    async fn non_retryable_error_stops_immediately() {
        let chain = ProviderChain {
            primary: ProviderConfig {
                provider: "anthropic".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            fallbacks: vec![ProviderConfig {
                provider: "openai".to_string(),
                model: "gpt-4o".to_string(),
            }],
        };
        let mut mgr = FailoverManager::new(chain);

        let providers_tried = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let pt = providers_tried.clone();

        let outcome: FailoverOutcome<&str> = mgr
            .call_with_failover(|provider| {
                let name = provider.provider.clone();
                let pt = pt.clone();
                async move {
                    pt.lock().unwrap().push(name);
                    Err(ApiError {
                        status_code: Some(400),
                        message: "Invalid request body".to_string(),
                        is_connection_error: false,
                    })
                }
            })
            .await;

        let tried = providers_tried.lock().unwrap();
        // Only primary should have been tried — non-retryable stops immediately
        assert_eq!(tried.len(), 1);
        assert_eq!(tried[0], "anthropic");
        assert!(matches!(outcome, FailoverOutcome::AllFailed { .. }));
    }

    // ── Config parsing tests ────────────────────────────────────

    #[test]
    fn parse_full_config_with_fallbacks() {
        let toml = r#"
[provider]
primary = "anthropic"
model = "claude-sonnet-4-20250514"

[[provider.fallbacks]]
provider = "openai"
model = "gpt-4o"

[[provider.fallbacks]]
provider = "openrouter"
model = "anthropic/claude-sonnet-4-20250514"
"#;

        let chain = parse_provider_chain(toml).unwrap();
        assert_eq!(chain.primary.provider, "anthropic");
        assert_eq!(chain.primary.model, "claude-sonnet-4-20250514");
        assert_eq!(chain.fallbacks.len(), 2);
        assert_eq!(chain.fallbacks[0].provider, "openai");
        assert_eq!(chain.fallbacks[0].model, "gpt-4o");
        assert_eq!(chain.fallbacks[1].provider, "openrouter");
        assert_eq!(chain.fallbacks[1].model, "anthropic/claude-sonnet-4-20250514");
    }

    #[test]
    fn parse_config_no_fallbacks() {
        let toml = r#"
[provider]
primary = "anthropic"
model = "claude-sonnet-4-20250514"
"#;

        let chain = parse_provider_chain(toml).unwrap();
        assert_eq!(chain.primary.provider, "anthropic");
        assert!(!chain.has_fallbacks());
    }

    #[test]
    fn parse_config_legacy_name_field() {
        let toml = r#"
[provider]
name = "openai"
model = "gpt-4o"
"#;

        let chain = parse_provider_chain(toml).unwrap();
        assert_eq!(chain.primary.provider, "openai");
        assert_eq!(chain.primary.model, "gpt-4o");
    }

    #[test]
    fn parse_config_no_provider_section_defaults() {
        let toml = r#"
[workspace]
root = "."
"#;

        let chain = parse_provider_chain(toml).unwrap();
        assert_eq!(chain.primary.provider, "anthropic");
        assert_eq!(chain.primary.model, "claude-sonnet-4-20250514");
        assert!(!chain.has_fallbacks());
    }

    #[test]
    fn parse_empty_config_defaults() {
        let chain = parse_provider_chain("").unwrap();
        assert_eq!(chain.primary.provider, "anthropic");
        assert!(!chain.has_fallbacks());
    }

    #[test]
    fn parse_config_primary_takes_precedence_over_name() {
        let toml = r#"
[provider]
primary = "openai"
name = "anthropic"
model = "gpt-4o"
"#;

        let chain = parse_provider_chain(toml).unwrap();
        assert_eq!(chain.primary.provider, "openai");
    }

    // ── Provider status formatting ──────────────────────────────

    #[test]
    fn format_provider_status_primary_only() {
        let chain = ProviderChain::primary_only("anthropic", "claude-sonnet-4-20250514");
        let mgr = FailoverManager::new(chain);
        let status = mgr.format_provider_status();

        assert!(status.contains("Primary:  anthropic"));
        assert!(status.contains("claude-sonnet-4-20250514"));
        assert!(status.contains("Fallbacks: none"));
    }

    #[test]
    fn format_provider_status_with_fallbacks() {
        let chain = ProviderChain {
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
        };
        let mgr = FailoverManager::new(chain);
        let status = mgr.format_provider_status();

        assert!(status.contains("Primary:  anthropic"));
        assert!(status.contains("1. openai"));
        assert!(status.contains("2. openrouter"));
        assert!(!status.contains("Fallbacks: none"));
    }

    #[test]
    fn provider_command_output() {
        let chain = ProviderChain {
            primary: ProviderConfig {
                provider: "anthropic".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
            },
            fallbacks: vec![
                ProviderConfig {
                    provider: "openai".to_string(),
                    model: "gpt-4o".to_string(),
                },
            ],
        };
        let mgr = FailoverManager::new(chain);
        let output = mgr.format_provider_status();

        // Verify it's well-formed output suitable for the /provider command
        assert!(output.contains("Provider Configuration"));
        assert!(output.contains("Primary:"));
        assert!(output.contains("Fallbacks (in order):"));
    }
}

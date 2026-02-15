//! Telegram bot channel for ZeroClaw.
//!
//! Provides a Telegram bot that uses HTTP long-polling (not webhooks)
//! to receive messages, routes them through the same agent pipeline as
//! the CLI, and sends responses back via the Telegram sendMessage API.
//!
//! ## Why long-polling?
//!
//! Long-polling is simpler than webhooks — no public IP or TLS certificate
//! is needed. The bot polls `getUpdates` at a configurable interval
//! (default 5 seconds), which keeps CPU cost low while being responsive
//! enough for messaging.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────┐
//! │ TelegramListener                         │
//! │                                          │
//! │  ┌─── poll loop (tokio::spawn) ───────┐  │
//! │  │ GET /getUpdates?offset=N&timeout=T │  │
//! │  │         ↓                          │  │
//! │  │ parse Update[]                     │  │
//! │  │         ↓                          │  │
//! │  │ for each text message:             │  │
//! │  │   send to agent_tx channel         │  │
//! │  └────────────────────────────────────┘  │
//! │                                          │
//! │  agent_rx ──→ agent loop processes msg   │
//! │            ──→ response sent via          │
//! │                 POST /sendMessage         │
//! └──────────────────────────────────────────┘
//! ```
//!
//! ## Session persistence
//!
//! Messages from Telegram use the same `SessionManager` and
//! `MarkdownMemory` as the CLI — turns are persisted to the JSONL
//! session file, interactions appear in today's daily note, and all
//! tools (memory, web, heartbeat) are available.
//!
//! ## Configuration
//!
//! Add to `config.toml`:
//!
//! ```toml
//! [telegram]
//! bot_token = "123456:ABC-DEF..."
//! enabled = true
//! poll_interval_secs = 5
//! ```
//!
//! Or set the `TELEGRAM_BOT_TOKEN` environment variable and pass
//! `--telegram` on the command line.

use anyhow::{Context, Result};
use chrono::Local;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex};

use crate::memory::markdown::MarkdownMemory;
use crate::memory::session::SessionManager;

// ── Telegram API types ──────────────────────────────────────────

/// A Telegram Update object (subset of fields we need).
#[derive(Debug, Clone, Deserialize)]
pub struct TelegramUpdate {
    /// Unique identifier for this update.
    pub update_id: i64,
    /// New incoming message (present when the update is a message).
    pub message: Option<TelegramMessage>,
}

/// A Telegram Message object (subset of fields we need).
#[derive(Debug, Clone, Deserialize)]
pub struct TelegramMessage {
    /// Unique message identifier within this chat.
    pub message_id: i64,
    /// Sender of the message.
    pub from: Option<TelegramUser>,
    /// Chat the message belongs to.
    pub chat: TelegramChat,
    /// Actual text of the message (if it is a text message).
    pub text: Option<String>,
    /// Date the message was sent (Unix timestamp).
    pub date: i64,
}

/// A Telegram User object (subset).
#[derive(Debug, Clone, Deserialize)]
pub struct TelegramUser {
    pub id: i64,
    pub first_name: String,
    #[serde(default)]
    pub last_name: Option<String>,
    #[serde(default)]
    pub username: Option<String>,
}

/// A Telegram Chat object (subset).
#[derive(Debug, Clone, Deserialize)]
pub struct TelegramChat {
    pub id: i64,
    #[serde(rename = "type")]
    pub chat_type: String,
}

/// Response from Telegram's getUpdates API.
#[derive(Debug, Deserialize)]
pub struct GetUpdatesResponse {
    pub ok: bool,
    #[serde(default)]
    pub result: Vec<TelegramUpdate>,
    #[serde(default)]
    pub description: Option<String>,
}

/// Response from Telegram's sendMessage API.
#[derive(Debug, Deserialize)]
pub struct SendMessageResponse {
    pub ok: bool,
    #[serde(default)]
    pub description: Option<String>,
}

/// Request body for Telegram's sendMessage API.
#[derive(Debug, Serialize)]
struct SendMessageRequest {
    chat_id: i64,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parse_mode: Option<String>,
}

// ── Incoming message from Telegram ──────────────────────────────

/// A processed incoming message ready for the agent loop.
#[derive(Debug, Clone)]
pub struct IncomingTelegramMessage {
    /// The chat ID to reply to.
    pub chat_id: i64,
    /// The text content of the message.
    pub text: String,
    /// Display name of the sender.
    pub sender_name: String,
    /// Telegram user ID of the sender.
    pub sender_id: i64,
    /// Original Telegram message ID (for threading/logging).
    pub message_id: i64,
}

// ── Telegram bot configuration ──────────────────────────────────

/// Configuration for the Telegram bot.
#[derive(Debug, Clone)]
pub struct TelegramConfig {
    /// The bot token from @BotFather.
    pub bot_token: String,
    /// Whether the Telegram listener is enabled.
    pub enabled: bool,
    /// How often to poll getUpdates (seconds).
    pub poll_interval_secs: u64,
    /// Timeout for long-polling requests to Telegram (seconds).
    /// This is the `timeout` parameter sent to getUpdates — Telegram
    /// holds the connection open for this long before returning an
    /// empty response. Set lower than `poll_interval_secs` for best
    /// responsiveness.
    pub long_poll_timeout_secs: u64,
}

impl Default for TelegramConfig {
    fn default() -> Self {
        Self {
            bot_token: String::new(),
            enabled: false,
            poll_interval_secs: 5,
            long_poll_timeout_secs: 30,
        }
    }
}

impl TelegramConfig {
    /// Create a TelegramConfig from a bot token with default settings.
    pub fn new(bot_token: String) -> Self {
        Self {
            bot_token,
            enabled: true,
            ..Default::default()
        }
    }

    /// Resolve the bot token from the config field or the environment.
    pub fn resolve_token(&self) -> Option<String> {
        if !self.bot_token.is_empty() {
            Some(self.bot_token.clone())
        } else {
            std::env::var("TELEGRAM_BOT_TOKEN").ok()
        }
    }

    /// Build the base URL for Telegram Bot API calls.
    pub fn api_base_url(&self) -> Option<String> {
        self.resolve_token()
            .map(|token| format!("https://api.telegram.org/bot{token}"))
    }
}

// ── Telegram API client ─────────────────────────────────────────

/// Low-level client for Telegram Bot API operations.
pub struct TelegramApi {
    client: Client,
    base_url: String,
}

impl TelegramApi {
    /// Create a new TelegramApi client.
    pub fn new(bot_token: &str) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .context("Failed to build HTTP client for Telegram API")?;

        Ok(Self {
            client,
            base_url: format!("https://api.telegram.org/bot{bot_token}"),
        })
    }

    /// Poll for new updates using long-polling.
    ///
    /// `offset` is the ID of the first update to return. Use the
    /// last received `update_id + 1` to acknowledge previous updates.
    /// `timeout` is how long Telegram should hold the connection open
    /// (in seconds) before returning an empty response.
    pub async fn get_updates(
        &self,
        offset: Option<i64>,
        timeout: u64,
    ) -> Result<Vec<TelegramUpdate>> {
        let url = format!("{}/getUpdates", self.base_url);

        let mut params: Vec<(&str, String)> = vec![
            ("timeout", timeout.to_string()),
        ];
        if let Some(off) = offset {
            params.push(("offset", off.to_string()));
        }

        let response = self
            .client
            .get(&url)
            .query(&params)
            .send()
            .await
            .context("Failed to send getUpdates request")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!(
                "Telegram getUpdates returned HTTP {status}: {body}"
            );
        }

        let parsed: GetUpdatesResponse = response
            .json()
            .await
            .context("Failed to parse getUpdates response")?;

        if !parsed.ok {
            anyhow::bail!(
                "Telegram getUpdates returned ok=false: {}",
                parsed.description.unwrap_or_default()
            );
        }

        Ok(parsed.result)
    }

    /// Send a text message to a chat.
    ///
    /// Splits messages longer than Telegram's 4096-char limit into
    /// multiple messages automatically.
    pub async fn send_message(
        &self,
        chat_id: i64,
        text: &str,
    ) -> Result<()> {
        const MAX_MESSAGE_LEN: usize = 4096;

        // Split long messages at the 4096-char boundary
        let chunks = split_message(text, MAX_MESSAGE_LEN);

        for chunk in chunks {
            let url = format!("{}/sendMessage", self.base_url);
            let body = SendMessageRequest {
                chat_id,
                text: chunk,
                parse_mode: None,
            };

            let response = self
                .client
                .post(&url)
                .json(&body)
                .send()
                .await
                .context("Failed to send sendMessage request")?;

            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                anyhow::bail!(
                    "Telegram sendMessage returned HTTP {status}: {body}"
                );
            }

            let parsed: SendMessageResponse = response
                .json()
                .await
                .context("Failed to parse sendMessage response")?;

            if !parsed.ok {
                anyhow::bail!(
                    "Telegram sendMessage returned ok=false: {}",
                    parsed.description.unwrap_or_default()
                );
            }
        }

        Ok(())
    }
}

// ── TelegramListener ────────────────────────────────────────────

/// The Telegram polling listener.
///
/// Runs in a background tokio task, polls getUpdates, parses incoming
/// messages, and sends them through an mpsc channel for the agent loop
/// to process. Responses are sent back via the Telegram API.
///
/// The listener shares the same `MarkdownMemory` and `SessionManager`
/// as the CLI, so all interactions are persisted and visible in session
/// history and daily notes.
pub struct TelegramListener {
    config: TelegramConfig,
    api: Arc<TelegramApi>,
    memory: Arc<MarkdownMemory>,
    session_mgr: Arc<Mutex<SessionManager>>,
}

impl TelegramListener {
    /// Create a new TelegramListener.
    pub fn new(
        config: TelegramConfig,
        memory: Arc<MarkdownMemory>,
        session_mgr: Arc<Mutex<SessionManager>>,
    ) -> Result<Self> {
        let token = config
            .resolve_token()
            .context("Telegram bot token not set. Set TELEGRAM_BOT_TOKEN env var \
                      or add telegram_bot_token to config.toml.")?;

        let api = Arc::new(TelegramApi::new(&token)?);

        Ok(Self {
            config,
            api,
            memory,
            session_mgr,
        })
    }

    /// Get a reference to the Telegram API client (for sending messages
    /// from outside the listener, e.g., from the agent loop).
    pub fn api(&self) -> &Arc<TelegramApi> {
        &self.api
    }

    /// Start the polling loop in a background task.
    ///
    /// Returns an `mpsc::Receiver` that yields incoming messages.
    /// The caller is responsible for processing messages and sending
    /// responses via `self.api().send_message()`.
    ///
    /// The returned `JoinHandle` can be used to monitor or cancel the
    /// polling task.
    pub fn start_polling(
        &self,
    ) -> (
        mpsc::Receiver<IncomingTelegramMessage>,
        tokio::task::JoinHandle<()>,
    ) {
        let (tx, rx) = mpsc::channel::<IncomingTelegramMessage>(64);
        let api = self.api.clone();
        let poll_interval = Duration::from_secs(self.config.poll_interval_secs);
        let long_poll_timeout = self.config.long_poll_timeout_secs;
        let memory = self.memory.clone();

        let handle = tokio::spawn(async move {
            let mut offset: Option<i64> = None;

            loop {
                match api.get_updates(offset, long_poll_timeout).await {
                    Ok(updates) => {
                        for update in updates {
                            // Advance the offset to acknowledge this update
                            offset = Some(update.update_id + 1);

                            // Only process text messages
                            if let Some(msg) = update.message {
                                if let Some(text) = msg.text {
                                    let sender_name = msg
                                        .from
                                        .as_ref()
                                        .map(|u| {
                                            let mut name = u.first_name.clone();
                                            if let Some(ref last) = u.last_name {
                                                name.push(' ');
                                                name.push_str(last);
                                            }
                                            name
                                        })
                                        .unwrap_or_else(|| "Unknown".to_string());

                                    let sender_id = msg
                                        .from
                                        .as_ref()
                                        .map(|u| u.id)
                                        .unwrap_or(0);

                                    // Log to daily note
                                    let log_entry = format!(
                                        "**[telegram]** Message from {sender_name}: {text}"
                                    );
                                    if let Err(e) = memory.append_daily_note(&log_entry).await {
                                        eprintln!(
                                            "[telegram] Failed to log to daily note: {e}"
                                        );
                                    }

                                    let incoming = IncomingTelegramMessage {
                                        chat_id: msg.chat.id,
                                        text,
                                        sender_name,
                                        sender_id,
                                        message_id: msg.message_id,
                                    };

                                    if tx.send(incoming).await.is_err() {
                                        // Receiver dropped — agent loop shut down
                                        eprintln!(
                                            "[telegram] Message channel closed, stopping poller"
                                        );
                                        return;
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("[telegram] getUpdates error: {e}");
                        // Back off on errors to avoid hammering the API
                        tokio::time::sleep(Duration::from_secs(10)).await;
                        continue;
                    }
                }

                // Wait before the next poll cycle
                tokio::time::sleep(poll_interval).await;
            }
        });

        (rx, handle)
    }

    /// Run the Telegram listener with an integrated agent message handler.
    ///
    /// This is the high-level entry point that:
    /// 1. Starts the polling loop
    /// 2. Reads incoming messages from the channel
    /// 3. Passes each message through the provided `agent_handler`
    /// 4. Sends the response back to the Telegram chat
    /// 5. Persists both the user message and assistant response to the session
    /// 6. Logs interactions to today's daily note
    ///
    /// `agent_handler` is a closure that takes a user message string and
    /// returns the agent's response. This is the same processing pipeline
    /// used by the CLI.
    pub async fn run<F, Fut>(
        &self,
        agent_handler: F,
    ) -> Result<()>
    where
        F: Fn(String) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<String>> + Send,
    {
        let (mut rx, _handle) = self.start_polling();
        let api = self.api.clone();
        let session_mgr = self.session_mgr.clone();
        let memory = self.memory.clone();

        eprintln!("[telegram] Bot started, polling for messages...");

        while let Some(msg) = rx.recv().await {
            eprintln!(
                "[telegram] Message from {} (chat {}): {}",
                msg.sender_name, msg.chat_id, msg.text
            );

            // Persist the user turn
            {
                let mgr = session_mgr.lock().await;
                let user_turn = SessionManager::user_turn(&format!(
                    "[telegram:{}] {}",
                    msg.sender_name, msg.text
                ));
                if let Err(e) = mgr.append_turn(&user_turn).await {
                    eprintln!("[telegram] Failed to persist user turn: {e}");
                }
            }

            // Process through the agent loop
            let response = match agent_handler(msg.text.clone()).await {
                Ok(resp) => resp,
                Err(e) => {
                    let error_msg = format!("Error processing message: {e}");
                    eprintln!("[telegram] {error_msg}");
                    error_msg
                }
            };

            // Persist the assistant turn
            {
                let mgr = session_mgr.lock().await;
                let assistant_turn =
                    SessionManager::assistant_turn(Some(&response), vec![]);
                if let Err(e) = mgr.append_turn(&assistant_turn).await {
                    eprintln!("[telegram] Failed to persist assistant turn: {e}");
                }
            }

            // Log response to daily note
            let log_entry = format!(
                "**[telegram-response]** To {}: {}",
                msg.sender_name,
                truncate_for_log(&response, 200)
            );
            if let Err(e) = memory.append_daily_note(&log_entry).await {
                eprintln!("[telegram] Failed to log response to daily note: {e}");
            }

            // Send response back to Telegram
            if let Err(e) = api.send_message(msg.chat_id, &response).await {
                eprintln!(
                    "[telegram] Failed to send response to chat {}: {e}",
                    msg.chat_id
                );
            }
        }

        Ok(())
    }
}

// ── Utility functions ───────────────────────────────────────────

/// Split a message into chunks that fit within Telegram's character limit.
///
/// Tries to split at newline boundaries for readability. Falls back to
/// splitting at the character limit if no newline is found.
fn split_message(text: &str, max_len: usize) -> Vec<String> {
    if text.len() <= max_len {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        if remaining.len() <= max_len {
            chunks.push(remaining.to_string());
            break;
        }

        // Try to split at the last newline before the limit
        let split_at = remaining[..max_len]
            .rfind('\n')
            .map(|pos| pos + 1)  // include the newline in this chunk
            .unwrap_or(max_len); // fallback: hard split at limit

        chunks.push(remaining[..split_at].to_string());
        remaining = &remaining[split_at..];
    }

    chunks
}

/// Truncate text for daily note logging (to avoid bloating the log).
fn truncate_for_log(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        text.to_string()
    } else {
        format!("{}...", &text[..max_chars])
    }
}

/// Parse a TOML config section for Telegram settings.
///
/// Expected format:
/// ```toml
/// [telegram]
/// bot_token = "123456:ABC-DEF..."
/// enabled = true
/// poll_interval_secs = 5
/// ```
pub fn parse_telegram_config(toml_value: &toml::Value) -> TelegramConfig {
    let telegram_section = match toml_value.get("telegram") {
        Some(section) => section,
        None => return TelegramConfig::default(),
    };

    let bot_token = telegram_section
        .get("bot_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let enabled = telegram_section
        .get("enabled")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let poll_interval_secs = telegram_section
        .get("poll_interval_secs")
        .and_then(|v| v.as_integer())
        .map(|v| v.max(1) as u64)
        .unwrap_or(5);

    let long_poll_timeout_secs = telegram_section
        .get("long_poll_timeout_secs")
        .and_then(|v| v.as_integer())
        .map(|v| v.max(1) as u64)
        .unwrap_or(30);

    TelegramConfig {
        bot_token,
        enabled,
        poll_interval_secs,
        long_poll_timeout_secs,
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── TelegramConfig tests ────────────────────────────────────

    #[test]
    fn default_config_has_sensible_defaults() {
        let config = TelegramConfig::default();
        assert!(config.bot_token.is_empty());
        assert!(!config.enabled);
        assert_eq!(config.poll_interval_secs, 5);
        assert_eq!(config.long_poll_timeout_secs, 30);
    }

    #[test]
    fn new_config_sets_token_and_enables() {
        let config = TelegramConfig::new("test_token".to_string());
        assert_eq!(config.bot_token, "test_token");
        assert!(config.enabled);
        assert_eq!(config.poll_interval_secs, 5);
    }

    #[test]
    fn resolve_token_prefers_struct_field() {
        let config = TelegramConfig::new("from_config".to_string());
        assert_eq!(config.resolve_token().unwrap(), "from_config");
    }

    #[test]
    fn resolve_token_falls_back_to_env() {
        let config = TelegramConfig::default();
        // Clear any existing env var
        std::env::remove_var("TELEGRAM_BOT_TOKEN");
        assert!(config.resolve_token().is_none());

        // Set env var
        std::env::set_var("TELEGRAM_BOT_TOKEN", "from_env");
        assert_eq!(config.resolve_token().unwrap(), "from_env");
        std::env::remove_var("TELEGRAM_BOT_TOKEN");
    }

    #[test]
    fn api_base_url_builds_correctly() {
        let config = TelegramConfig::new("123456:ABC-DEF".to_string());
        assert_eq!(
            config.api_base_url().unwrap(),
            "https://api.telegram.org/bot123456:ABC-DEF"
        );
    }

    #[test]
    fn api_base_url_returns_none_without_token() {
        std::env::remove_var("TELEGRAM_BOT_TOKEN");
        let config = TelegramConfig::default();
        assert!(config.api_base_url().is_none());
    }

    // ── split_message tests ─────────────────────────────────────

    #[test]
    fn split_message_short_text_unchanged() {
        let result = split_message("Hello, world!", 4096);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "Hello, world!");
    }

    #[test]
    fn split_message_empty_text() {
        let result = split_message("", 4096);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "");
    }

    #[test]
    fn split_message_splits_at_newline() {
        let text = "Line 1\nLine 2\nLine 3";
        let result = split_message(text, 10);
        // "Line 1\n" is 7 chars, "Line 2\n" is 7 more = 14 > 10
        // So first chunk should be "Line 1\n" (split at last newline before 10)
        assert!(result.len() >= 2);
        assert!(result[0].ends_with('\n') || result[0].len() <= 10);
    }

    #[test]
    fn split_message_hard_splits_without_newline() {
        let text = "a".repeat(100);
        let result = split_message(&text, 30);
        assert!(result.len() > 1);
        for chunk in &result {
            assert!(chunk.len() <= 30);
        }
    }

    #[test]
    fn split_message_preserves_all_content() {
        let text = "Hello\nWorld\nThis is a test\nMore text here";
        let result = split_message(text, 15);
        let rejoined: String = result.join("");
        assert_eq!(rejoined, text);
    }

    // ── truncate_for_log tests ──────────────────────────────────

    #[test]
    fn truncate_short_text_unchanged() {
        assert_eq!(truncate_for_log("short", 100), "short");
    }

    #[test]
    fn truncate_long_text_adds_ellipsis() {
        let text = "a".repeat(50);
        let result = truncate_for_log(&text, 20);
        assert_eq!(result.len(), 23); // 20 + "..."
        assert!(result.ends_with("..."));
    }

    // ── Telegram API type deserialization tests ─────────────────

    #[test]
    fn deserialize_get_updates_response() {
        let json = r#"{
            "ok": true,
            "result": [
                {
                    "update_id": 123456789,
                    "message": {
                        "message_id": 42,
                        "from": {
                            "id": 100,
                            "first_name": "John",
                            "last_name": "Doe",
                            "username": "johndoe"
                        },
                        "chat": {
                            "id": 100,
                            "type": "private"
                        },
                        "text": "Hello bot!",
                        "date": 1707900000
                    }
                }
            ]
        }"#;

        let resp: GetUpdatesResponse = serde_json::from_str(json).unwrap();
        assert!(resp.ok);
        assert_eq!(resp.result.len(), 1);

        let update = &resp.result[0];
        assert_eq!(update.update_id, 123456789);

        let msg = update.message.as_ref().unwrap();
        assert_eq!(msg.message_id, 42);
        assert_eq!(msg.text.as_deref(), Some("Hello bot!"));

        let from = msg.from.as_ref().unwrap();
        assert_eq!(from.first_name, "John");
        assert_eq!(from.last_name.as_deref(), Some("Doe"));
        assert_eq!(from.username.as_deref(), Some("johndoe"));

        assert_eq!(msg.chat.id, 100);
        assert_eq!(msg.chat.chat_type, "private");
    }

    #[test]
    fn deserialize_get_updates_empty() {
        let json = r#"{"ok": true, "result": []}"#;
        let resp: GetUpdatesResponse = serde_json::from_str(json).unwrap();
        assert!(resp.ok);
        assert!(resp.result.is_empty());
    }

    #[test]
    fn deserialize_get_updates_error() {
        let json = r#"{"ok": false, "description": "Unauthorized"}"#;
        let resp: GetUpdatesResponse = serde_json::from_str(json).unwrap();
        assert!(!resp.ok);
        assert_eq!(resp.description.as_deref(), Some("Unauthorized"));
    }

    #[test]
    fn deserialize_message_without_from() {
        let json = r#"{
            "update_id": 1,
            "message": {
                "message_id": 1,
                "chat": {"id": 1, "type": "private"},
                "text": "anonymous message",
                "date": 1707900000
            }
        }"#;
        let update: TelegramUpdate = serde_json::from_str(json).unwrap();
        let msg = update.message.unwrap();
        assert!(msg.from.is_none());
        assert_eq!(msg.text.as_deref(), Some("anonymous message"));
    }

    #[test]
    fn deserialize_update_without_message() {
        // Updates can be callback queries, inline queries, etc.
        let json = r#"{"update_id": 999}"#;
        let update: TelegramUpdate = serde_json::from_str(json).unwrap();
        assert_eq!(update.update_id, 999);
        assert!(update.message.is_none());
    }

    #[test]
    fn deserialize_message_without_text() {
        // Messages can be photos, stickers, etc.
        let json = r#"{
            "update_id": 1,
            "message": {
                "message_id": 1,
                "chat": {"id": 1, "type": "group"},
                "date": 1707900000
            }
        }"#;
        let update: TelegramUpdate = serde_json::from_str(json).unwrap();
        let msg = update.message.unwrap();
        assert!(msg.text.is_none());
        assert_eq!(msg.chat.chat_type, "group");
    }

    #[test]
    fn deserialize_send_message_response_ok() {
        let json = r#"{"ok": true}"#;
        let resp: SendMessageResponse = serde_json::from_str(json).unwrap();
        assert!(resp.ok);
    }

    #[test]
    fn deserialize_send_message_response_error() {
        let json = r#"{"ok": false, "description": "Bad Request: chat not found"}"#;
        let resp: SendMessageResponse = serde_json::from_str(json).unwrap();
        assert!(!resp.ok);
        assert!(resp.description.unwrap().contains("chat not found"));
    }

    // ── parse_telegram_config tests ─────────────────────────────

    #[test]
    fn parse_config_with_all_fields() {
        let toml_str = r#"
            [telegram]
            bot_token = "123456:ABC"
            enabled = true
            poll_interval_secs = 10
            long_poll_timeout_secs = 45
        "#;
        let value: toml::Value = toml::from_str(toml_str).unwrap();
        let config = parse_telegram_config(&value);

        assert_eq!(config.bot_token, "123456:ABC");
        assert!(config.enabled);
        assert_eq!(config.poll_interval_secs, 10);
        assert_eq!(config.long_poll_timeout_secs, 45);
    }

    #[test]
    fn parse_config_with_defaults() {
        let toml_str = r#"
            [telegram]
            bot_token = "test"
        "#;
        let value: toml::Value = toml::from_str(toml_str).unwrap();
        let config = parse_telegram_config(&value);

        assert_eq!(config.bot_token, "test");
        assert!(!config.enabled); // default false
        assert_eq!(config.poll_interval_secs, 5); // default 5
        assert_eq!(config.long_poll_timeout_secs, 30); // default 30
    }

    #[test]
    fn parse_config_missing_section() {
        let toml_str = r#"
            [other]
            key = "value"
        "#;
        let value: toml::Value = toml::from_str(toml_str).unwrap();
        let config = parse_telegram_config(&value);

        assert!(config.bot_token.is_empty());
        assert!(!config.enabled);
    }

    #[test]
    fn parse_config_enforces_minimum_interval() {
        let toml_str = r#"
            [telegram]
            bot_token = "test"
            poll_interval_secs = 0
        "#;
        let value: toml::Value = toml::from_str(toml_str).unwrap();
        let config = parse_telegram_config(&value);
        assert_eq!(config.poll_interval_secs, 1); // clamped to minimum 1
    }

    // ── IncomingTelegramMessage tests ───────────────────────────

    #[test]
    fn incoming_message_fields() {
        let msg = IncomingTelegramMessage {
            chat_id: 12345,
            text: "Hello!".to_string(),
            sender_name: "Alice".to_string(),
            sender_id: 100,
            message_id: 42,
        };
        assert_eq!(msg.chat_id, 12345);
        assert_eq!(msg.text, "Hello!");
        assert_eq!(msg.sender_name, "Alice");
        assert_eq!(msg.sender_id, 100);
        assert_eq!(msg.message_id, 42);
    }
}

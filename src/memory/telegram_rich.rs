//! Rich Telegram messaging: MarkdownV2 formatting, message editing,
//! file attachments, reactions, and graceful degradation.
//!
//! ## Design principles
//!
//! Every rich feature is **fire-and-forget**: if any API call fails
//! (missing permissions, network blip, unsupported chat type), we log a
//! warning and continue. The bot must never crash or block on cosmetic
//! features.
//!
//! ## `TelegramRichClient`
//!
//! Wraps the raw Telegram Bot API HTTP calls and adds:
//!
//! - **MarkdownV2 escaping** — `telegram_escape()` handles all 18
//!   special characters required by Telegram's MarkdownV2 mode.
//! - **Message editing** — `edit_message()` updates a previously sent
//!   message in-place, useful for progress updates during tool-use chains.
//! - **Reactions** — `set_reaction()` sends emoji reactions to acknowledge
//!   receipt, completion, or errors.
//! - **File sending** — `send_file()` uploads a local file via the
//!   `sendDocument` multipart API.
//! - **File downloading** — `download_file()` fetches a file from Telegram
//!   (sent by a user) and saves it to the workspace.
//! - **Message splitting** — Long MarkdownV2 messages are split at the
//!   4096-char boundary while preserving formatting.

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

// ── MarkdownV2 escaping ─────────────────────────────────────────

/// Characters that must be escaped in Telegram's MarkdownV2 mode.
///
/// See <https://core.telegram.org/bots/api#markdownv2-style>.
const MARKDOWNV2_SPECIAL_CHARS: &[char] = &[
    '_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!',
];

/// Escape a plain-text string for safe inclusion in a MarkdownV2 message.
///
/// Every special character listed in the Telegram Bot API docs is
/// prefixed with a backslash.
///
/// ```
/// use openclaw_scribe::memory::telegram_rich::telegram_escape;
/// assert_eq!(telegram_escape("Hello *world*!"), r"Hello \*world\*\!");
/// ```
pub fn telegram_escape(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len() + text.len() / 4);
    for ch in text.chars() {
        if MARKDOWNV2_SPECIAL_CHARS.contains(&ch) {
            escaped.push('\\');
        }
        escaped.push(ch);
    }
    escaped
}

// ── API request / response types ────────────────────────────────

/// Request body for `editMessageText`.
#[derive(Debug, Serialize)]
struct EditMessageRequest {
    chat_id: i64,
    message_id: i64,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parse_mode: Option<String>,
}

/// Minimal response from edit / send operations.
#[derive(Debug, Deserialize)]
struct ApiResponse {
    ok: bool,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    result: Option<serde_json::Value>,
}

/// Response from `sendMessage` that includes the sent message object.
#[derive(Debug, Deserialize)]
struct SendMessageFullResponse {
    ok: bool,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    result: Option<SentMessage>,
}

/// A sent message (subset of fields we need).
#[derive(Debug, Deserialize)]
struct SentMessage {
    message_id: i64,
}

/// Request body for `sendMessage` with MarkdownV2 support.
#[derive(Debug, Serialize)]
struct SendMessageRequest {
    chat_id: i64,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parse_mode: Option<String>,
}

/// Request body for `setMessageReaction`.
#[derive(Debug, Serialize)]
struct SetReactionRequest {
    chat_id: i64,
    message_id: i64,
    reaction: Vec<ReactionType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    is_big: Option<bool>,
}

/// A reaction type (emoji).
#[derive(Debug, Serialize)]
struct ReactionType {
    #[serde(rename = "type")]
    reaction_type: String,
    emoji: String,
}

/// Response from `getFile`.
#[derive(Debug, Deserialize)]
struct GetFileResponse {
    ok: bool,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    result: Option<TelegramFile>,
}

/// A Telegram File object.
#[derive(Debug, Clone, Deserialize)]
pub struct TelegramFile {
    pub file_id: String,
    #[serde(default)]
    pub file_unique_id: Option<String>,
    #[serde(default)]
    pub file_size: Option<i64>,
    #[serde(default)]
    pub file_path: Option<String>,
}

/// Document attachment from an incoming message.
#[derive(Debug, Clone, Deserialize)]
pub struct TelegramDocument {
    pub file_id: String,
    #[serde(default)]
    pub file_unique_id: Option<String>,
    #[serde(default)]
    pub file_name: Option<String>,
    #[serde(default)]
    pub mime_type: Option<String>,
    #[serde(default)]
    pub file_size: Option<i64>,
}

/// Photo size from an incoming message.
#[derive(Debug, Clone, Deserialize)]
pub struct TelegramPhotoSize {
    pub file_id: String,
    #[serde(default)]
    pub file_unique_id: Option<String>,
    pub width: i32,
    pub height: i32,
    #[serde(default)]
    pub file_size: Option<i64>,
}

/// Voice message from an incoming message.
#[derive(Debug, Clone, Deserialize)]
pub struct TelegramVoice {
    pub file_id: String,
    #[serde(default)]
    pub file_unique_id: Option<String>,
    pub duration: i32,
    #[serde(default)]
    pub mime_type: Option<String>,
    #[serde(default)]
    pub file_size: Option<i64>,
}

// ── Reaction emoji constants ────────────────────────────────────

/// Reaction sent when the bot starts processing a message.
pub const REACTION_PROCESSING: &str = "\u{1F440}"; // 👀
/// Reaction sent on successful completion.
pub const REACTION_SUCCESS: &str = "\u{2705}"; // ✅
/// Reaction sent on error.
pub const REACTION_ERROR: &str = "\u{274C}"; // ❌

// ── TelegramRichClient ──────────────────────────────────────────

/// High-level Telegram client with rich messaging capabilities.
///
/// Wraps raw Bot API HTTP calls and adds MarkdownV2 formatting,
/// message editing, reactions, and file transfer. All rich features
/// degrade gracefully — failures are logged but never propagated.
pub struct TelegramRichClient {
    client: Client,
    base_url: String,
    bot_token: String,
    /// Tracks the last message_id sent per chat for edit-in-place.
    last_message_ids: Arc<Mutex<HashMap<i64, i64>>>,
    /// Workspace root for saving downloaded files.
    workspace_root: PathBuf,
}

impl TelegramRichClient {
    /// Create a new `TelegramRichClient`.
    ///
    /// `workspace_root` is used as the base directory for saving
    /// files downloaded from Telegram.
    pub fn new(bot_token: &str, workspace_root: &Path) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .context("Failed to build HTTP client for TelegramRichClient")?;

        Ok(Self {
            client,
            base_url: format!("https://api.telegram.org/bot{bot_token}"),
            bot_token: bot_token.to_string(),
            last_message_ids: Arc::new(Mutex::new(HashMap::new())),
            workspace_root: workspace_root.to_path_buf(),
        })
    }

    /// Create a `TelegramRichClient` with a custom base URL (for testing).
    #[cfg(test)]
    pub fn with_base_url(
        bot_token: &str,
        base_url: &str,
        workspace_root: &Path,
    ) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .context("Failed to build HTTP client")?;

        Ok(Self {
            client,
            base_url: base_url.to_string(),
            bot_token: bot_token.to_string(),
            last_message_ids: Arc::new(Mutex::new(HashMap::new())),
            workspace_root: workspace_root.to_path_buf(),
        })
    }

    // ── Sending messages (MarkdownV2) ───────────────────────────

    /// Send a plain-text message. The text is auto-escaped for MarkdownV2.
    ///
    /// Returns the message_id of the sent message (or the last chunk
    /// if the message was split).
    pub async fn send_markdown_message(
        &self,
        chat_id: i64,
        text: &str,
    ) -> Result<i64> {
        let escaped = telegram_escape(text);
        self.send_raw_markdown(chat_id, &escaped).await
    }

    /// Send a pre-formatted MarkdownV2 message (caller is responsible
    /// for correct escaping of literal text).
    ///
    /// Returns the message_id of the sent message (last chunk if split).
    pub async fn send_raw_markdown(
        &self,
        chat_id: i64,
        markdown_text: &str,
    ) -> Result<i64> {
        let chunks = split_message(markdown_text, 4096);
        let mut last_id: i64 = 0;

        for chunk in &chunks {
            let url = format!("{}/sendMessage", self.base_url);
            let body = SendMessageRequest {
                chat_id,
                text: chunk.clone(),
                parse_mode: Some("MarkdownV2".to_string()),
            };

            let resp = self
                .client
                .post(&url)
                .json(&body)
                .send()
                .await
                .context("Failed to send MarkdownV2 message")?;

            let status = resp.status();
            let parsed: SendMessageFullResponse = resp
                .json()
                .await
                .context("Failed to parse sendMessage response")?;

            if !status.is_success() || !parsed.ok {
                // Fallback: try without parse_mode
                let fallback_id = self.send_plain_message(chat_id, chunk).await?;
                last_id = fallback_id;
                continue;
            }

            if let Some(msg) = parsed.result {
                last_id = msg.message_id;
            }
        }

        // Track last message ID for this chat
        self.last_message_ids.lock().await.insert(chat_id, last_id);

        Ok(last_id)
    }

    /// Send a plain-text message (no formatting). Used as fallback.
    ///
    /// Returns the message_id.
    pub async fn send_plain_message(
        &self,
        chat_id: i64,
        text: &str,
    ) -> Result<i64> {
        let chunks = split_message(text, 4096);
        let mut last_id: i64 = 0;

        for chunk in &chunks {
            let url = format!("{}/sendMessage", self.base_url);
            let body = SendMessageRequest {
                chat_id,
                text: chunk.clone(),
                parse_mode: None,
            };

            let resp = self
                .client
                .post(&url)
                .json(&body)
                .send()
                .await
                .context("Failed to send plain message")?;

            let parsed: SendMessageFullResponse = resp
                .json()
                .await
                .context("Failed to parse sendMessage response")?;

            if !parsed.ok {
                anyhow::bail!(
                    "sendMessage failed: {}",
                    parsed.description.unwrap_or_default()
                );
            }

            if let Some(msg) = parsed.result {
                last_id = msg.message_id;
            }
        }

        self.last_message_ids.lock().await.insert(chat_id, last_id);
        Ok(last_id)
    }

    // ── Message editing ─────────────────────────────────────────

    /// Edit a previously sent message in-place.
    ///
    /// Used during tool-use chains to update progress instead of
    /// spamming new messages. The text is sent as plain text (no parse mode)
    /// to avoid escaping issues with intermediate status messages.
    pub async fn edit_message(
        &self,
        chat_id: i64,
        message_id: i64,
        text: &str,
    ) -> Result<()> {
        let url = format!("{}/editMessageText", self.base_url);
        let body = EditMessageRequest {
            chat_id,
            message_id,
            text: text.to_string(),
            parse_mode: None,
        };

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Failed to edit message")?;

        let parsed: ApiResponse = resp
            .json()
            .await
            .context("Failed to parse editMessageText response")?;

        if !parsed.ok {
            anyhow::bail!(
                "editMessageText failed: {}",
                parsed.description.unwrap_or_default()
            );
        }

        Ok(())
    }

    /// Edit the last message sent to a chat. Returns `Ok(true)` if a
    /// message was found and edited, `Ok(false)` if no tracked message
    /// exists for that chat.
    pub async fn edit_last_message(
        &self,
        chat_id: i64,
        text: &str,
    ) -> Result<bool> {
        let msg_id = {
            let ids = self.last_message_ids.lock().await;
            ids.get(&chat_id).copied()
        };

        match msg_id {
            Some(id) => {
                self.edit_message(chat_id, id, text).await?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Get the last tracked message_id for a chat.
    pub async fn last_message_id(&self, chat_id: i64) -> Option<i64> {
        self.last_message_ids.lock().await.get(&chat_id).copied()
    }

    /// Manually set the last message_id for a chat (e.g., after sending
    /// through the basic API).
    pub async fn track_message(&self, chat_id: i64, message_id: i64) {
        self.last_message_ids.lock().await.insert(chat_id, message_id);
    }

    // ── Reactions ───────────────────────────────────────────────

    /// Set a reaction emoji on a message.
    ///
    /// Fire-and-forget: returns `Ok(())` even if the API rejects the
    /// reaction (e.g., bot lacks permission, chat doesn't support reactions).
    pub async fn set_reaction(
        &self,
        chat_id: i64,
        message_id: i64,
        emoji: &str,
    ) -> Result<()> {
        let url = format!("{}/setMessageReaction", self.base_url);
        let body = SetReactionRequest {
            chat_id,
            message_id,
            reaction: vec![ReactionType {
                reaction_type: "emoji".to_string(),
                emoji: emoji.to_string(),
            }],
            is_big: None,
        };

        match self.client.post(&url).json(&body).send().await {
            Ok(resp) => {
                // We don't care about the response — reactions are best-effort.
                if !resp.status().is_success() {
                    eprintln!(
                        "[telegram-rich] Reaction {emoji} failed (HTTP {}), ignoring",
                        resp.status()
                    );
                }
            }
            Err(e) => {
                eprintln!("[telegram-rich] Reaction {emoji} request failed: {e}, ignoring");
            }
        }

        Ok(())
    }

    /// Acknowledge message receipt (eyes emoji).
    pub async fn react_processing(&self, chat_id: i64, message_id: i64) {
        let _ = self.set_reaction(chat_id, message_id, REACTION_PROCESSING).await;
    }

    /// Mark message as successfully processed.
    pub async fn react_success(&self, chat_id: i64, message_id: i64) {
        let _ = self.set_reaction(chat_id, message_id, REACTION_SUCCESS).await;
    }

    /// Mark message as failed.
    pub async fn react_error(&self, chat_id: i64, message_id: i64) {
        let _ = self.set_reaction(chat_id, message_id, REACTION_ERROR).await;
    }

    // ── File sending ────────────────────────────────────────────

    /// Send a local file to a Telegram chat via `sendDocument`.
    ///
    /// The file is uploaded as a multipart form. Common types (text,
    /// code, images, PDFs) are supported natively by Telegram.
    pub async fn send_file(
        &self,
        chat_id: i64,
        file_path: &Path,
        caption: Option<&str>,
    ) -> Result<()> {
        if !file_path.exists() {
            anyhow::bail!("File not found: {}", file_path.display());
        }
        if !file_path.is_file() {
            anyhow::bail!("Path is not a file: {}", file_path.display());
        }

        let file_name = file_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "file".to_string());

        let file_bytes = tokio::fs::read(file_path)
            .await
            .context("Failed to read file for upload")?;

        let url = format!("{}/sendDocument", self.base_url);

        let file_part = reqwest::multipart::Part::bytes(file_bytes)
            .file_name(file_name)
            .mime_str("application/octet-stream")
            .context("Failed to set MIME type")?;

        let mut form = reqwest::multipart::Form::new()
            .text("chat_id", chat_id.to_string())
            .part("document", file_part);

        if let Some(cap) = caption {
            form = form.text("caption", cap.to_string());
        }

        let resp = self
            .client
            .post(&url)
            .multipart(form)
            .send()
            .await
            .context("Failed to send document")?;

        let parsed: ApiResponse = resp
            .json()
            .await
            .context("Failed to parse sendDocument response")?;

        if !parsed.ok {
            anyhow::bail!(
                "sendDocument failed: {}",
                parsed.description.unwrap_or_default()
            );
        }

        Ok(())
    }

    // ── File downloading ────────────────────────────────────────

    /// Download a file from Telegram using its `file_id`.
    ///
    /// Calls `getFile` to resolve the server-side path, then downloads
    /// the raw bytes and saves them to `{workspace_root}/downloads/{filename}`.
    ///
    /// Returns the local path where the file was saved.
    pub async fn download_file(
        &self,
        file_id: &str,
        suggested_name: Option<&str>,
    ) -> Result<PathBuf> {
        // Step 1: Call getFile to get the file_path
        let get_file_url = format!("{}/getFile", self.base_url);
        let resp = self
            .client
            .get(&get_file_url)
            .query(&[("file_id", file_id)])
            .send()
            .await
            .context("Failed to call getFile")?;

        let parsed: GetFileResponse = resp
            .json()
            .await
            .context("Failed to parse getFile response")?;

        if !parsed.ok {
            anyhow::bail!(
                "getFile failed: {}",
                parsed.description.unwrap_or_default()
            );
        }

        let tg_file = parsed
            .result
            .context("getFile returned ok but no result")?;

        let server_path = tg_file
            .file_path
            .context("getFile result has no file_path")?;

        // Step 2: Download the file bytes
        let download_url = format!(
            "https://api.telegram.org/file/bot{}/{}",
            self.bot_token, server_path
        );

        let file_bytes = self
            .client
            .get(&download_url)
            .send()
            .await
            .context("Failed to download file from Telegram")?
            .bytes()
            .await
            .context("Failed to read file bytes")?;

        // Step 3: Save to workspace
        let downloads_dir = self.workspace_root.join("downloads");
        tokio::fs::create_dir_all(&downloads_dir)
            .await
            .context("Failed to create downloads directory")?;

        let filename = suggested_name
            .map(|s| s.to_string())
            .or_else(|| {
                server_path
                    .rsplit('/')
                    .next()
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| format!("file_{file_id}"));

        let save_path = downloads_dir.join(&filename);
        tokio::fs::write(&save_path, &file_bytes)
            .await
            .context("Failed to save downloaded file")?;

        Ok(save_path)
    }

    /// Process an incoming document attachment: download and return info.
    pub async fn handle_incoming_document(
        &self,
        doc: &TelegramDocument,
    ) -> Result<(PathBuf, String)> {
        let filename = doc.file_name.as_deref().unwrap_or("document");
        let path = self.download_file(&doc.file_id, Some(filename)).await?;

        let size_str = doc
            .file_size
            .map(|s| format_file_size(s))
            .unwrap_or_else(|| "unknown size".to_string());

        let info = format!(
            "User sent file: {} ({}). Saved to {}",
            filename,
            size_str,
            path.display()
        );

        Ok((path, info))
    }

    /// Process an incoming photo: download the largest version.
    pub async fn handle_incoming_photo(
        &self,
        photos: &[TelegramPhotoSize],
    ) -> Result<(PathBuf, String)> {
        // Pick the largest photo (last in the array)
        let photo = photos
            .last()
            .context("Empty photo array")?;

        let filename = format!("photo_{}.jpg", &photo.file_id[..8.min(photo.file_id.len())]);
        let path = self.download_file(&photo.file_id, Some(&filename)).await?;

        let size_str = photo
            .file_size
            .map(|s| format_file_size(s))
            .unwrap_or_else(|| "unknown size".to_string());

        let info = format!(
            "User sent photo: {} ({}x{}, {}). Saved to {}",
            filename,
            photo.width,
            photo.height,
            size_str,
            path.display()
        );

        Ok((path, info))
    }

    /// Process an incoming voice message: download it.
    pub async fn handle_incoming_voice(
        &self,
        voice: &TelegramVoice,
    ) -> Result<(PathBuf, String)> {
        let ext = voice
            .mime_type
            .as_deref()
            .and_then(mime_to_extension)
            .unwrap_or("ogg");
        let filename = format!("voice_{}.{}", &voice.file_id[..8.min(voice.file_id.len())], ext);
        let path = self.download_file(&voice.file_id, Some(&filename)).await?;

        let size_str = voice
            .file_size
            .map(|s| format_file_size(s))
            .unwrap_or_else(|| "unknown size".to_string());

        let info = format!(
            "User sent voice message: {} ({}s, {}). Saved to {}",
            filename,
            voice.duration,
            size_str,
            path.display()
        );

        Ok((path, info))
    }
}

// ── SendFileTool ────────────────────────────────────────────────

use crate::tools::{Tool, ToolExecutionResult};
use async_trait::async_trait;

/// Agent tool for sending a file from the workspace to the current
/// Telegram chat.
pub struct SendFileTool {
    rich_client: Arc<TelegramRichClient>,
    /// The chat_id to send to (set when processing a Telegram message).
    chat_id: Arc<Mutex<Option<i64>>>,
}

impl SendFileTool {
    pub fn new(
        rich_client: Arc<TelegramRichClient>,
        chat_id: Arc<Mutex<Option<i64>>>,
    ) -> Self {
        Self {
            rich_client,
            chat_id,
        }
    }
}

#[async_trait]
impl Tool for SendFileTool {
    fn name(&self) -> &str {
        "send_file"
    }

    fn description(&self) -> &str {
        "Send a file from the workspace to the current Telegram chat. \
         Supports text, code, images, PDFs, and other common file types. \
         The file must exist on disk."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to send."
                },
                "caption": {
                    "type": "string",
                    "description": "Optional caption to include with the file."
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<ToolExecutionResult> {
        let path_str = args["path"].as_str().unwrap_or("").to_string();
        let caption = args["caption"].as_str().map(|s| s.to_string());

        if path_str.is_empty() {
            return Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some("path is required.".to_string()),
            });
        }

        let chat_id = {
            let id = self.chat_id.lock().await;
            match *id {
                Some(cid) => cid,
                None => {
                    return Ok(ToolExecutionResult {
                        success: false,
                        output: String::new(),
                        error: Some(
                            "No active Telegram chat. This tool only works \
                             when processing a Telegram message."
                                .to_string(),
                        ),
                    });
                }
            }
        };

        let file_path = Path::new(&path_str);

        match self
            .rich_client
            .send_file(chat_id, file_path, caption.as_deref())
            .await
        {
            Ok(()) => Ok(ToolExecutionResult {
                success: true,
                output: format!("File sent to Telegram chat: {path_str}"),
                error: None,
            }),
            Err(e) => Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to send file: {e}")),
            }),
        }
    }
}

// ── Utility functions ───────────────────────────────────────────

/// Split a message into chunks that fit within a character limit.
///
/// Tries to split at newline boundaries for readability. Falls back
/// to splitting at the character limit if no newline is found.
pub fn split_message(text: &str, max_len: usize) -> Vec<String> {
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

        let split_at = remaining[..max_len]
            .rfind('\n')
            .map(|pos| pos + 1)
            .unwrap_or(max_len);

        chunks.push(remaining[..split_at].to_string());
        remaining = &remaining[split_at..];
    }

    chunks
}

/// Format a file size in bytes into a human-readable string.
fn format_file_size(bytes: i64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// Map a MIME type to a file extension.
fn mime_to_extension(mime: &str) -> Option<&str> {
    match mime {
        "audio/ogg" => Some("ogg"),
        "audio/mpeg" => Some("mp3"),
        "audio/wav" => Some("wav"),
        "image/jpeg" => Some("jpg"),
        "image/png" => Some("png"),
        "image/gif" => Some("gif"),
        "video/mp4" => Some("mp4"),
        "application/pdf" => Some("pdf"),
        _ => None,
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ── telegram_escape tests ───────────────────────────────────

    #[test]
    fn escape_all_special_chars() {
        // Every special char should be escaped with a backslash
        let input = "_*[]()~`>#+-=|{}.!";
        let result = telegram_escape(input);
        for ch in MARKDOWNV2_SPECIAL_CHARS {
            let escaped = format!("\\{ch}");
            assert!(
                result.contains(&escaped),
                "Expected {escaped} in result: {result}"
            );
        }
    }

    #[test]
    fn escape_plain_text_unchanged() {
        let input = "Hello world 123";
        assert_eq!(telegram_escape(input), "Hello world 123");
    }

    #[test]
    fn escape_mixed_content() {
        assert_eq!(
            telegram_escape("Hello *world*!"),
            r"Hello \*world\*\!"
        );
    }

    #[test]
    fn escape_empty_string() {
        assert_eq!(telegram_escape(""), "");
    }

    #[test]
    fn escape_only_special_chars() {
        let result = telegram_escape("*_~");
        assert_eq!(result, r"\*\_\~");
    }

    #[test]
    fn escape_code_snippet() {
        let input = "Use `println!()` in Rust";
        let result = telegram_escape(input);
        assert_eq!(result, r"Use \`println\!\(\)\` in Rust");
    }

    #[test]
    fn escape_markdown_link() {
        let input = "Visit [Google](https://google.com)";
        let result = telegram_escape(input);
        assert_eq!(
            result,
            r"Visit \[Google\]\(https://google\.com\)"
        );
    }

    #[test]
    fn escape_preserves_newlines_and_spaces() {
        let input = "Line 1\nLine 2\n  indented";
        let result = telegram_escape(input);
        assert_eq!(result, "Line 1\nLine 2\n  indented");
    }

    #[test]
    fn escape_heading_with_hash() {
        let input = "# Heading";
        assert_eq!(telegram_escape(input), r"\# Heading");
    }

    #[test]
    fn escape_pipe_in_table() {
        let input = "col1 | col2";
        assert_eq!(telegram_escape(input), r"col1 \| col2");
    }

    #[test]
    fn escape_braces() {
        let input = "fn main() { }";
        assert_eq!(telegram_escape(input), r"fn main\(\) \{ \}");
    }

    #[test]
    fn escape_tilde() {
        let input = "~strikethrough~";
        assert_eq!(telegram_escape(input), r"\~strikethrough\~");
    }

    #[test]
    fn escape_equals_and_plus() {
        let input = "a + b = c";
        assert_eq!(telegram_escape(input), r"a \+ b \= c");
    }

    #[test]
    fn escape_dot_in_url() {
        let input = "example.com";
        assert_eq!(telegram_escape(input), r"example\.com");
    }

    // ── split_message tests ─────────────────────────────────────

    #[test]
    fn split_short_message_unchanged() {
        let result = split_message("Hello!", 4096);
        assert_eq!(result, vec!["Hello!"]);
    }

    #[test]
    fn split_empty_message() {
        let result = split_message("", 4096);
        assert_eq!(result, vec![""]);
    }

    #[test]
    fn split_at_newline_boundary() {
        let text = "Line 1\nLine 2\nLine 3";
        let result = split_message(text, 10);
        assert!(result.len() >= 2);
        for chunk in &result {
            assert!(chunk.len() <= 10, "Chunk too long: {}", chunk.len());
        }
    }

    #[test]
    fn split_hard_split_without_newline() {
        let text = "a".repeat(100);
        let result = split_message(&text, 30);
        assert!(result.len() > 1);
        for chunk in &result {
            assert!(chunk.len() <= 30);
        }
    }

    #[test]
    fn split_preserves_all_content() {
        let text = "Hello\nWorld\nThis is a test\nMore text here";
        let result = split_message(text, 15);
        let rejoined: String = result.join("");
        assert_eq!(rejoined, text);
    }

    #[test]
    fn split_at_4096_boundary() {
        let text = "x".repeat(8192);
        let result = split_message(&text, 4096);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 4096);
        assert_eq!(result[1].len(), 4096);
    }

    #[test]
    fn split_just_under_limit() {
        let text = "x".repeat(4096);
        let result = split_message(&text, 4096);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn split_just_over_limit() {
        let text = "x".repeat(4097);
        let result = split_message(&text, 4096);
        assert_eq!(result.len(), 2);
    }

    // ── format_file_size tests ──────────────────────────────────

    #[test]
    fn format_bytes() {
        assert_eq!(format_file_size(500), "500 B");
    }

    #[test]
    fn format_kilobytes() {
        assert_eq!(format_file_size(2048), "2.0 KB");
    }

    #[test]
    fn format_megabytes() {
        assert_eq!(format_file_size(5 * 1024 * 1024), "5.0 MB");
    }

    // ── mime_to_extension tests ─────────────────────────────────

    #[test]
    fn mime_ogg() {
        assert_eq!(mime_to_extension("audio/ogg"), Some("ogg"));
    }

    #[test]
    fn mime_unknown() {
        assert_eq!(mime_to_extension("application/x-custom"), None);
    }

    // ── Message editing: tracks message_id ──────────────────────

    #[tokio::test]
    async fn track_message_id() {
        let tmp = TempDir::new().unwrap();
        let client = TelegramRichClient::new("test_token", tmp.path()).unwrap();

        assert!(client.last_message_id(123).await.is_none());

        client.track_message(123, 456).await;
        assert_eq!(client.last_message_id(123).await, Some(456));

        // Different chat should be independent
        assert!(client.last_message_id(789).await.is_none());
    }

    #[tokio::test]
    async fn track_message_overwrites_per_chat() {
        let tmp = TempDir::new().unwrap();
        let client = TelegramRichClient::new("test_token", tmp.path()).unwrap();

        client.track_message(100, 1).await;
        client.track_message(100, 2).await;
        assert_eq!(client.last_message_id(100).await, Some(2));
    }

    // ── Reaction emoji constants ────────────────────────────────

    #[test]
    fn reaction_constants_are_valid_emoji() {
        assert_eq!(REACTION_PROCESSING, "\u{1F440}");
        assert_eq!(REACTION_SUCCESS, "\u{2705}");
        assert_eq!(REACTION_ERROR, "\u{274C}");
    }

    // ── TelegramDocument deserialization ─────────────────────────

    #[test]
    fn deserialize_document() {
        let json = r#"{
            "file_id": "BQACAgIAAxkBAAI",
            "file_unique_id": "AgAD",
            "file_name": "report.pdf",
            "mime_type": "application/pdf",
            "file_size": 125000
        }"#;
        let doc: TelegramDocument = serde_json::from_str(json).unwrap();
        assert_eq!(doc.file_id, "BQACAgIAAxkBAAI");
        assert_eq!(doc.file_name.as_deref(), Some("report.pdf"));
        assert_eq!(doc.mime_type.as_deref(), Some("application/pdf"));
        assert_eq!(doc.file_size, Some(125000));
    }

    #[test]
    fn deserialize_document_minimal() {
        let json = r#"{"file_id": "abc123"}"#;
        let doc: TelegramDocument = serde_json::from_str(json).unwrap();
        assert_eq!(doc.file_id, "abc123");
        assert!(doc.file_name.is_none());
        assert!(doc.mime_type.is_none());
        assert!(doc.file_size.is_none());
    }

    // ── TelegramPhotoSize deserialization ────────────────────────

    #[test]
    fn deserialize_photo_size() {
        let json = r#"{
            "file_id": "AgACAgIAAxkBAAI",
            "file_unique_id": "AQADAgAT",
            "width": 1280,
            "height": 720,
            "file_size": 98765
        }"#;
        let photo: TelegramPhotoSize = serde_json::from_str(json).unwrap();
        assert_eq!(photo.width, 1280);
        assert_eq!(photo.height, 720);
        assert_eq!(photo.file_size, Some(98765));
    }

    // ── TelegramVoice deserialization ───────────────────────────

    #[test]
    fn deserialize_voice() {
        let json = r#"{
            "file_id": "AwACAgIAAxkBAAI",
            "duration": 5,
            "mime_type": "audio/ogg",
            "file_size": 12345
        }"#;
        let voice: TelegramVoice = serde_json::from_str(json).unwrap();
        assert_eq!(voice.duration, 5);
        assert_eq!(voice.mime_type.as_deref(), Some("audio/ogg"));
    }

    // ── GetFileResponse deserialization ─────────────────────────

    #[test]
    fn deserialize_get_file_response() {
        let json = r#"{
            "ok": true,
            "result": {
                "file_id": "abc123",
                "file_unique_id": "def456",
                "file_size": 1024,
                "file_path": "documents/file_0.pdf"
            }
        }"#;
        let resp: GetFileResponse = serde_json::from_str(json).unwrap();
        assert!(resp.ok);
        let file = resp.result.unwrap();
        assert_eq!(file.file_id, "abc123");
        assert_eq!(file.file_path.as_deref(), Some("documents/file_0.pdf"));
        assert_eq!(file.file_size, Some(1024));
    }

    #[test]
    fn deserialize_get_file_error() {
        let json = r#"{"ok": false, "description": "Bad Request: invalid file_id"}"#;
        let resp: GetFileResponse = serde_json::from_str(json).unwrap();
        assert!(!resp.ok);
        assert!(resp.description.unwrap().contains("invalid file_id"));
    }

    // ── SendFileTool metadata ───────────────────────────────────

    #[tokio::test]
    async fn send_file_tool_name_and_schema() {
        let tmp = TempDir::new().unwrap();
        let client = Arc::new(TelegramRichClient::new("test", tmp.path()).unwrap());
        let chat_id = Arc::new(Mutex::new(None));
        let tool = SendFileTool::new(client, chat_id);

        assert_eq!(tool.name(), "send_file");
        let schema = tool.parameters_schema();
        let required = schema["required"].as_array().unwrap();
        let names: Vec<&str> = required.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(names.contains(&"path"));
    }

    #[tokio::test]
    async fn send_file_tool_no_chat_id_returns_error() {
        let tmp = TempDir::new().unwrap();
        let client = Arc::new(TelegramRichClient::new("test", tmp.path()).unwrap());
        let chat_id = Arc::new(Mutex::new(None)); // No active chat
        let tool = SendFileTool::new(client, chat_id);

        let result = tool
            .execute(serde_json::json!({"path": "/tmp/test.txt"}))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("No active Telegram chat"));
    }

    #[tokio::test]
    async fn send_file_tool_empty_path_returns_error() {
        let tmp = TempDir::new().unwrap();
        let client = Arc::new(TelegramRichClient::new("test", tmp.path()).unwrap());
        let chat_id = Arc::new(Mutex::new(Some(123i64)));
        let tool = SendFileTool::new(client, chat_id);

        let result = tool
            .execute(serde_json::json!({"path": ""}))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("path is required"));
    }

    // ── Graceful degradation: reaction failure doesn't block ────

    #[tokio::test]
    async fn reaction_to_invalid_server_does_not_panic() {
        let tmp = TempDir::new().unwrap();
        // Point to a non-existent server — the reaction call must not panic
        let client = TelegramRichClient::new("invalid_token", tmp.path()).unwrap();

        // These should all return Ok(()) even though they'll fail internally
        client.react_processing(123, 1).await;
        client.react_success(123, 1).await;
        client.react_error(123, 1).await;
        // If we get here without panic, the test passes
    }

    // ── Serialization of API request types ──────────────────────

    #[test]
    fn edit_message_request_serializes() {
        let req = EditMessageRequest {
            chat_id: 123,
            message_id: 456,
            text: "Updated text".to_string(),
            parse_mode: Some("MarkdownV2".to_string()),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["chat_id"], 123);
        assert_eq!(json["message_id"], 456);
        assert_eq!(json["text"], "Updated text");
        assert_eq!(json["parse_mode"], "MarkdownV2");
    }

    #[test]
    fn edit_message_request_skips_none_parse_mode() {
        let req = EditMessageRequest {
            chat_id: 1,
            message_id: 2,
            text: "hello".to_string(),
            parse_mode: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("parse_mode").is_none());
    }

    #[test]
    fn set_reaction_request_serializes() {
        let req = SetReactionRequest {
            chat_id: 100,
            message_id: 200,
            reaction: vec![ReactionType {
                reaction_type: "emoji".to_string(),
                emoji: "\u{1F440}".to_string(),
            }],
            is_big: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["chat_id"], 100);
        assert_eq!(json["message_id"], 200);
        let reactions = json["reaction"].as_array().unwrap();
        assert_eq!(reactions.len(), 1);
        assert_eq!(reactions[0]["type"], "emoji");
    }

    // ── send_file validation ────────────────────────────────────

    #[tokio::test]
    async fn send_file_nonexistent_path_returns_error() {
        let tmp = TempDir::new().unwrap();
        let client = TelegramRichClient::new("test_token", tmp.path()).unwrap();

        let result = client
            .send_file(123, Path::new("/nonexistent/file.txt"), None)
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("File not found"));
    }

    #[tokio::test]
    async fn send_file_directory_returns_error() {
        let tmp = TempDir::new().unwrap();
        let client = TelegramRichClient::new("test_token", tmp.path()).unwrap();

        let result = client
            .send_file(123, tmp.path(), None)
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not a file"));
    }

    // ── Markdown formatting preserved in messages ───────────────

    #[test]
    fn escaped_markdown_preserves_structure() {
        // Bold: *text* in MarkdownV2
        // If we escape user input, the stars become literal
        let user_input = "This is *important*";
        let escaped = telegram_escape(user_input);
        assert_eq!(escaped, r"This is \*important\*");

        // To actually send bold, caller would build:
        let bold_text = format!("This is *{}*", telegram_escape("important"));
        assert_eq!(bold_text, r"This is *important*");
    }

    #[test]
    fn escaped_code_block_structure() {
        // To send a code block, only the content inside should be escaped
        // (actually in MarkdownV2, inside ``` blocks only ` and \ need escaping,
        // but for simplicity we test the general escape behavior)
        let code = "fn main() {}";
        let escaped = telegram_escape(code);
        assert!(escaped.contains(r"\{"));
        assert!(escaped.contains(r"\}"));
        assert!(escaped.contains(r"\("));
        assert!(escaped.contains(r"\)"));
    }
}

//! Agent tools for web search and page fetching.
//!
//! These tools give the agent research capabilities by allowing it to
//! search the web via the Brave Search API and fetch full page content.
//!
//! Tools provided:
//! - `web_search`  — search the web using the Brave Search API
//! - `web_fetch`   — fetch a URL and return its plain-text content
//!
//! ## Configuration
//!
//! The Brave Search API key is read from:
//! 1. The `BRAVE_API_KEY` environment variable, or
//! 2. The `brave_api_key` field in `~/.zeroclaw/config.toml`
//!
//! Get a key at <https://brave.com/search/api/>.

use crate::tools::{Tool, ToolExecutionResult};
use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use std::time::Duration;

// ── Web fetch configuration ─────────────────────────────────

/// Configuration for the web fetch tool.
///
/// Loaded from the `[web]` section of `config.toml`. Missing fields
/// use sensible defaults; a missing `[web]` section is fine.
#[derive(Debug, Clone, Deserialize)]
pub struct WebFetchConfig {
    /// Maximum response body size in bytes before truncation (default: 5 MB).
    #[serde(default = "default_max_fetch_bytes")]
    pub max_fetch_bytes: usize,
    /// Maximum number of HTTP redirects to follow (default: 5).
    #[serde(default = "default_max_redirects")]
    pub max_redirects: usize,
    /// Total request timeout in seconds (default: 30).
    #[serde(default = "default_fetch_timeout_secs")]
    pub fetch_timeout_secs: u64,
}

fn default_max_fetch_bytes() -> usize {
    5 * 1024 * 1024
}
fn default_max_redirects() -> usize {
    5
}
fn default_fetch_timeout_secs() -> u64 {
    30
}

impl Default for WebFetchConfig {
    fn default() -> Self {
        Self {
            max_fetch_bytes: default_max_fetch_bytes(),
            max_redirects: default_max_redirects(),
            fetch_timeout_secs: default_fetch_timeout_secs(),
        }
    }
}

/// Parse the `[web]` section from a TOML config string.
///
/// Returns defaults if the section is missing.
pub fn parse_web_config(toml_str: &str) -> Result<WebFetchConfig, toml::de::Error> {
    #[derive(Deserialize)]
    struct Wrapper {
        #[serde(default)]
        web: Option<WebFetchConfig>,
    }
    let wrapper: Wrapper = toml::from_str(toml_str)?;
    Ok(wrapper.web.unwrap_or_default())
}

/// Returns `true` if the content type represents binary content that
/// should not be downloaded (images, video, audio, archives, etc.).
fn is_binary_content_type(content_type: &str) -> bool {
    let ct = content_type.to_lowercase();
    let media_type = ct.split(';').next().unwrap_or(&ct).trim();
    media_type.starts_with("image/")
        || media_type.starts_with("video/")
        || media_type.starts_with("audio/")
        || media_type == "application/octet-stream"
        || media_type == "application/zip"
        || media_type == "application/gzip"
        || media_type == "application/x-gzip"
        || media_type == "application/x-tar"
        || media_type == "application/x-rar-compressed"
        || media_type == "application/x-7z-compressed"
        || media_type == "application/wasm"
        || media_type == "application/x-executable"
        || media_type == "application/x-sharedlib"
}

/// Approximate token count by splitting on whitespace.
///
/// This is a fast heuristic — not a true tokenizer — but sufficient for
/// truncation to avoid context bloat.
fn approx_token_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Truncate text to approximately `max_tokens` tokens (whitespace-split).
fn truncate_to_tokens(text: &str, max_tokens: usize) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() <= max_tokens {
        return text.to_string();
    }
    let truncated = words[..max_tokens].join(" ");
    format!("{truncated}\n\n[... truncated to {max_tokens} tokens]")
}

/// Strip HTML tags from a string, returning plain text.
///
/// This is a simple state-machine approach — it does not handle
/// CDATA, script content, or malformed HTML perfectly, but it is
/// good enough for extracting readable text from web pages.
fn strip_html_tags(html: &str) -> String {
    let mut output = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;
    let mut tag_buf = String::new();

    for ch in html.chars() {
        match ch {
            '<' => {
                in_tag = true;
                tag_buf.clear();
            }
            '>' if in_tag => {
                in_tag = false;
                let tag_lower = tag_buf.to_lowercase();
                if tag_lower.starts_with("script") {
                    in_script = true;
                } else if tag_lower.starts_with("/script") {
                    in_script = false;
                } else if tag_lower.starts_with("style") {
                    in_style = true;
                } else if tag_lower.starts_with("/style") {
                    in_style = false;
                }
                // Block-level tags get a newline
                if tag_lower.starts_with('p')
                    || tag_lower.starts_with("/p")
                    || tag_lower.starts_with("br")
                    || tag_lower.starts_with("div")
                    || tag_lower.starts_with("/div")
                    || tag_lower.starts_with("h")
                    || tag_lower.starts_with("/h")
                    || tag_lower.starts_with("li")
                    || tag_lower.starts_with("tr")
                {
                    output.push('\n');
                }
            }
            _ if in_tag => {
                tag_buf.push(ch);
            }
            _ if in_script || in_style => {
                // Skip content inside <script> and <style>
            }
            _ => {
                output.push(ch);
            }
        }
    }

    // Decode common HTML entities
    let output = output
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ");

    // Collapse excessive whitespace: multiple blank lines → single blank line
    let mut collapsed = String::with_capacity(output.len());
    let mut consecutive_newlines = 0u32;
    for ch in output.chars() {
        if ch == '\n' {
            consecutive_newlines += 1;
            if consecutive_newlines <= 2 {
                collapsed.push(ch);
            }
        } else {
            consecutive_newlines = 0;
            collapsed.push(ch);
        }
    }

    collapsed.trim().to_string()
}

// ── Brave Search API response types ─────────────────────────

#[derive(Debug, Deserialize)]
struct BraveSearchResponse {
    #[serde(default)]
    web: Option<BraveWebResults>,
}

#[derive(Debug, Deserialize)]
struct BraveWebResults {
    #[serde(default)]
    results: Vec<BraveWebResult>,
}

#[derive(Debug, Deserialize)]
struct BraveWebResult {
    title: String,
    url: String,
    #[serde(default)]
    description: String,
}

// ── web_search ──────────────────────────────────────────────

/// Tool: Search the web using the Brave Search API.
///
/// Sends a query to the Brave Search API and returns formatted
/// results with title, URL, and snippet for each hit.
pub struct WebSearchTool {
    client: Client,
    api_key: Option<String>,
}

impl WebSearchTool {
    pub fn new(api_key: Option<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| Client::new());
        Self { client, api_key }
    }

    /// Resolve the API key from the struct field or the environment.
    fn resolve_api_key(&self) -> Option<String> {
        self.api_key
            .clone()
            .or_else(|| std::env::var("BRAVE_API_KEY").ok())
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web for information using the Brave Search API. \
         Returns a list of results with titles, URLs, and snippets. \
         Use this when you need up-to-date information, facts you're \
         unsure about, or documentation references. Follow up with \
         web_fetch to read full page content when snippets aren't enough."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific and include relevant \
                                    keywords for better results."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5, max: 20).",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, args: Value) -> Result<ToolExecutionResult> {
        let query = args["query"]
            .as_str()
            .unwrap_or("")
            .to_string();

        if query.is_empty() {
            return Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some("Search query cannot be empty.".to_string()),
            });
        }

        let max_results = args["max_results"]
            .as_u64()
            .unwrap_or(5)
            .min(20) as usize;

        let api_key = match self.resolve_api_key() {
            Some(key) => key,
            None => {
                return Ok(ToolExecutionResult {
                    success: false,
                    output: String::new(),
                    error: Some(
                        "Brave API key not set. Set BRAVE_API_KEY env var \
                         or add brave_api_key to ~/.zeroclaw/config.toml."
                            .to_string(),
                    ),
                });
            }
        };

        let response = self
            .client
            .get("https://api.search.brave.com/res/v1/web/search")
            .header("Accept", "application/json")
            .header("X-Subscription-Token", &api_key)
            .query(&[
                ("q", query.as_str()),
                ("count", &max_results.to_string()),
            ])
            .send()
            .await;

        let response = match response {
            Ok(r) => r,
            Err(e) => {
                return Ok(ToolExecutionResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("Search request failed: {e}")),
                });
            }
        };

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Brave Search API returned HTTP {status}: {body}"
                )),
            });
        }

        let search_response: BraveSearchResponse = match response.json().await {
            Ok(r) => r,
            Err(e) => {
                return Ok(ToolExecutionResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("Failed to parse search response: {e}")),
                });
            }
        };

        let results = search_response
            .web
            .map(|w| w.results)
            .unwrap_or_default();

        if results.is_empty() {
            return Ok(ToolExecutionResult {
                success: true,
                output: format!("No results found for: \"{query}\""),
                error: None,
            });
        }

        let mut output = format!(
            "Search results for \"{query}\" ({} result(s)):\n\n",
            results.len()
        );
        for (i, result) in results.iter().enumerate() {
            output.push_str(&format!(
                "{}. **{}**\n   {}\n   {}\n\n",
                i + 1,
                result.title,
                result.url,
                result.description,
            ));
        }

        Ok(ToolExecutionResult {
            success: true,
            output,
            error: None,
        })
    }
}

// ── web_fetch ───────────────────────────────────────────────

/// Maximum number of tokens to return from a fetched page.
const MAX_FETCH_TOKENS: usize = 4000;

/// Tool: Fetch a web page and return its plain-text content.
///
/// Strips HTML tags, removes script/style blocks, and truncates
/// to avoid context window bloat.
pub struct WebFetchTool {
    client: Client,
    config: WebFetchConfig,
}

impl WebFetchTool {
    pub fn new() -> Self {
        Self::with_config(WebFetchConfig::default())
    }

    pub fn with_config(config: WebFetchConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.fetch_timeout_secs))
            .redirect(reqwest::redirect::Policy::limited(config.max_redirects))
            .user_agent("ZeroClaw/1.0 (research agent)")
            .build()
            .unwrap_or_else(|_| Client::new());
        Self { client, config }
    }
}

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch a web page and return its plain-text content. Use this after \
         web_search when you need more detail than the search snippet provides. \
         HTML is stripped and content is truncated to ~4000 tokens to avoid \
         context bloat. Works best for articles, documentation, and text-heavy pages."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch. Must be a valid HTTP or HTTPS URL."
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, args: Value) -> Result<ToolExecutionResult> {
        let url = args["url"]
            .as_str()
            .unwrap_or("")
            .to_string();

        if url.is_empty() {
            return Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some("URL cannot be empty.".to_string()),
            });
        }

        // Basic URL validation
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Invalid URL: \"{url}\". Must start with http:// or https://."
                )),
            });
        }

        let mut response = match self.client.get(&url).send().await {
            Ok(r) => r,
            Err(e) => {
                let msg = if e.is_timeout() {
                    format!("Request timed out fetching: {url}")
                } else if e.is_redirect() {
                    format!("Too many redirects (>{})", self.config.max_redirects)
                } else if e.is_connect() {
                    format!("Could not connect to: {url}")
                } else {
                    format!("Failed to fetch {url}: {e}")
                };
                return Ok(ToolExecutionResult {
                    success: false,
                    output: String::new(),
                    error: Some(msg),
                });
            }
        };

        let status = response.status();
        if !status.is_success() {
            return Ok(ToolExecutionResult {
                success: false,
                output: String::new(),
                error: Some(format!("HTTP {status} fetching {url}")),
            });
        }

        // Reject binary content types without downloading the body
        if let Some(content_type) = response.headers().get("content-type") {
            if let Ok(ct) = content_type.to_str() {
                if is_binary_content_type(ct) {
                    let content_length = response
                        .headers()
                        .get("content-length")
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("unknown");
                    return Ok(ToolExecutionResult {
                        success: true,
                        output: format!(
                            "Binary content ({ct}, {content_length} bytes). Not fetched."
                        ),
                        error: None,
                    });
                }
            }
        }

        // Read response body with size limit
        let max_bytes = self.config.max_fetch_bytes;
        let mut body_bytes: Vec<u8> = Vec::new();
        let mut size_truncated = false;

        loop {
            match response.chunk().await {
                Ok(Some(chunk)) => {
                    body_bytes.extend_from_slice(&chunk);
                    if body_bytes.len() > max_bytes {
                        body_bytes.truncate(max_bytes);
                        size_truncated = true;
                        break;
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    return Ok(ToolExecutionResult {
                        success: false,
                        output: String::new(),
                        error: Some(format!("Failed to read response body: {e}")),
                    });
                }
            }
        }

        let body = String::from_utf8_lossy(&body_bytes).to_string();
        let plain_text = strip_html_tags(&body);

        if plain_text.is_empty() {
            return Ok(ToolExecutionResult {
                success: true,
                output: format!("Page at {url} returned no readable text content."),
                error: None,
            });
        }

        let total_tokens = approx_token_count(&plain_text);
        let content = truncate_to_tokens(&plain_text, MAX_FETCH_TOKENS);

        let mut output = format!("Content from {url} (~{total_tokens} tokens):\n\n{content}");
        if total_tokens > MAX_FETCH_TOKENS {
            output.push_str(&format!(
                "\n\n(Showing ~{MAX_FETCH_TOKENS} of ~{total_tokens} tokens)"
            ));
        }
        if size_truncated {
            let size_str = if max_bytes >= 1024 * 1024 {
                format!("{}MB", max_bytes / (1024 * 1024))
            } else {
                format!("{} bytes", max_bytes)
            };
            output.push_str(&format!(
                "\n\n[Content truncated at {size_str}. Full page is larger.]"
            ));
        }

        Ok(ToolExecutionResult {
            success: true,
            output,
            error: None,
        })
    }
}

// ── Tool registration helper ────────────────────────────────

/// Create all web research tools, ready to register in the agent loop.
///
/// Usage in `tools::all_tools()`:
/// ```rust
/// use crate::memory::web_tools;
///
/// let web_config = web_tools::parse_web_config(&config_toml).unwrap_or_default();
/// let mut tools = tools::all_tools(&security, mem.clone(), composio_key, &config.browser);
/// tools.extend(web_tools::all_web_tools(config.brave_api_key.clone(), web_config));
/// ```
pub fn all_web_tools(
    brave_api_key: Option<String>,
    web_config: WebFetchConfig,
) -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(WebSearchTool::new(brave_api_key)),
        Box::new(WebFetchTool::with_config(web_config)),
    ]
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Unit tests for HTML stripping ────────────────────────

    #[test]
    fn strip_html_tags_removes_basic_tags() {
        let html = "<p>Hello <b>world</b></p>";
        let text = strip_html_tags(html);
        assert!(text.contains("Hello"));
        assert!(text.contains("world"));
        assert!(!text.contains("<p>"));
        assert!(!text.contains("<b>"));
    }

    #[test]
    fn strip_html_tags_removes_script_content() {
        let html = "<p>Before</p><script>alert('xss');</script><p>After</p>";
        let text = strip_html_tags(html);
        assert!(text.contains("Before"));
        assert!(text.contains("After"));
        assert!(!text.contains("alert"));
        assert!(!text.contains("xss"));
    }

    #[test]
    fn strip_html_tags_removes_style_content() {
        let html = "<style>body { color: red; }</style><p>Visible</p>";
        let text = strip_html_tags(html);
        assert!(text.contains("Visible"));
        assert!(!text.contains("color"));
        assert!(!text.contains("red"));
    }

    #[test]
    fn strip_html_tags_decodes_entities() {
        let html = "Tom &amp; Jerry &lt;3 &quot;cheese&quot;";
        let text = strip_html_tags(html);
        assert_eq!(text, "Tom & Jerry <3 \"cheese\"");
    }

    #[test]
    fn strip_html_tags_handles_empty_string() {
        assert_eq!(strip_html_tags(""), "");
    }

    #[test]
    fn strip_html_tags_handles_plain_text() {
        let text = "No HTML here, just plain text.";
        assert_eq!(strip_html_tags(text), text);
    }

    #[test]
    fn strip_html_tags_collapses_whitespace() {
        let html = "<p>First</p>\n\n\n\n\n<p>Second</p>";
        let text = strip_html_tags(html);
        // Should not have more than 2 consecutive newlines
        assert!(!text.contains("\n\n\n"));
    }

    // ── Unit tests for token truncation ─────────────────────

    #[test]
    fn truncate_to_tokens_short_text_unchanged() {
        let text = "hello world foo bar";
        let result = truncate_to_tokens(text, 100);
        assert_eq!(result, text);
    }

    #[test]
    fn truncate_to_tokens_long_text_truncated() {
        let words: Vec<&str> = (0..100).map(|_| "word").collect();
        let text = words.join(" ");
        let result = truncate_to_tokens(&text, 10);
        assert!(result.contains("[... truncated to 10 tokens]"));
        // Should have exactly 10 "word" occurrences before the truncation marker
        let before_marker = result.split("[...").next().unwrap();
        assert_eq!(before_marker.split_whitespace().count(), 10);
    }

    #[test]
    fn approx_token_count_counts_words() {
        assert_eq!(approx_token_count("hello world"), 2);
        assert_eq!(approx_token_count("one two three four"), 4);
        assert_eq!(approx_token_count(""), 0);
        assert_eq!(approx_token_count("   "), 0);
    }

    // ── Unit tests for WebSearchTool ────────────────────────

    #[test]
    fn web_search_tool_has_correct_name() {
        let tool = WebSearchTool::new(None);
        assert_eq!(tool.name(), "web_search");
    }

    #[test]
    fn web_search_tool_schema_requires_query() {
        let tool = WebSearchTool::new(None);
        let schema = tool.parameters_schema();
        let required = schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v.as_str() == Some("query")));
    }

    #[tokio::test]
    async fn web_search_tool_rejects_empty_query() {
        let tool = WebSearchTool::new(Some("fake_key".to_string()));
        let result = tool.execute(json!({"query": ""})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("empty"));
    }

    #[tokio::test]
    async fn web_search_tool_requires_api_key() {
        // Clear env var if set
        std::env::remove_var("BRAVE_API_KEY");
        let tool = WebSearchTool::new(None);
        let result = tool
            .execute(json!({"query": "rust programming"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("API key"));
    }

    #[tokio::test]
    async fn web_search_tool_uses_env_api_key() {
        let tool = WebSearchTool::new(None);
        // When no key is set in struct or env, resolve_api_key returns None
        std::env::remove_var("BRAVE_API_KEY");
        assert!(tool.resolve_api_key().is_none());

        // When key is set in struct, it takes precedence
        let tool_with_key = WebSearchTool::new(Some("test_key".to_string()));
        assert_eq!(tool_with_key.resolve_api_key().unwrap(), "test_key");
    }

    // ── Unit tests for WebFetchTool ─────────────────────────

    #[test]
    fn web_fetch_tool_has_correct_name() {
        let tool = WebFetchTool::new();
        assert_eq!(tool.name(), "web_fetch");
    }

    #[test]
    fn web_fetch_tool_schema_requires_url() {
        let tool = WebFetchTool::new();
        let schema = tool.parameters_schema();
        let required = schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v.as_str() == Some("url")));
    }

    #[tokio::test]
    async fn web_fetch_tool_rejects_empty_url() {
        let tool = WebFetchTool::new();
        let result = tool.execute(json!({"url": ""})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("empty"));
    }

    #[tokio::test]
    async fn web_fetch_tool_rejects_invalid_url() {
        let tool = WebFetchTool::new();
        let result = tool
            .execute(json!({"url": "not-a-url"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("Invalid URL"));
    }

    #[tokio::test]
    async fn web_fetch_tool_rejects_ftp_url() {
        let tool = WebFetchTool::new();
        let result = tool
            .execute(json!({"url": "ftp://example.com/file"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("Invalid URL"));
    }

    // ── Registration helper test ────────────────────────────

    #[test]
    fn all_web_tools_returns_two_tools() {
        let tools = all_web_tools(None, WebFetchConfig::default());
        assert_eq!(tools.len(), 2);

        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"web_search"));
        assert!(names.contains(&"web_fetch"));
    }

    // ── WebFetchConfig tests ────────────────────────────────

    #[test]
    fn config_defaults_are_sensible() {
        let config = WebFetchConfig::default();
        assert_eq!(config.max_fetch_bytes, 5 * 1024 * 1024);
        assert_eq!(config.max_redirects, 5);
        assert_eq!(config.fetch_timeout_secs, 30);
    }

    #[test]
    fn parse_web_config_missing_section_returns_defaults() {
        let toml_str = r#"
[logging]
structured = true
"#;
        let config = parse_web_config(toml_str).unwrap();
        assert_eq!(config.max_fetch_bytes, 5 * 1024 * 1024);
        assert_eq!(config.max_redirects, 5);
        assert_eq!(config.fetch_timeout_secs, 30);
    }

    #[test]
    fn parse_web_config_partial_section_fills_defaults() {
        let toml_str = r#"
[web]
max_fetch_bytes = 1048576
"#;
        let config = parse_web_config(toml_str).unwrap();
        assert_eq!(config.max_fetch_bytes, 1_048_576);
        assert_eq!(config.max_redirects, 5);
        assert_eq!(config.fetch_timeout_secs, 30);
    }

    #[test]
    fn parse_web_config_full_section() {
        let toml_str = r#"
[web]
max_fetch_bytes = 2097152
max_redirects = 3
fetch_timeout_secs = 10
"#;
        let config = parse_web_config(toml_str).unwrap();
        assert_eq!(config.max_fetch_bytes, 2_097_152);
        assert_eq!(config.max_redirects, 3);
        assert_eq!(config.fetch_timeout_secs, 10);
    }

    // ── Binary content-type detection ───────────────────────

    #[test]
    fn binary_content_type_detects_images() {
        assert!(is_binary_content_type("image/png"));
        assert!(is_binary_content_type("image/jpeg"));
        assert!(is_binary_content_type("image/gif"));
        assert!(is_binary_content_type("Image/PNG")); // case-insensitive
    }

    #[test]
    fn binary_content_type_detects_media() {
        assert!(is_binary_content_type("video/mp4"));
        assert!(is_binary_content_type("audio/mpeg"));
    }

    #[test]
    fn binary_content_type_detects_archives() {
        assert!(is_binary_content_type("application/zip"));
        assert!(is_binary_content_type("application/gzip"));
        assert!(is_binary_content_type("application/octet-stream"));
    }

    #[test]
    fn binary_content_type_ignores_charset_params() {
        assert!(is_binary_content_type("image/png; charset=utf-8"));
        assert!(is_binary_content_type("application/octet-stream; name=file.bin"));
    }

    #[test]
    fn binary_content_type_allows_text() {
        assert!(!is_binary_content_type("text/html"));
        assert!(!is_binary_content_type("text/plain"));
        assert!(!is_binary_content_type("text/html; charset=utf-8"));
        assert!(!is_binary_content_type("application/json"));
        assert!(!is_binary_content_type("application/xml"));
        assert!(!is_binary_content_type("application/javascript"));
    }

    // ── Integration tests with mock server ──────────────────

    #[tokio::test]
    async fn fetch_truncates_at_size_limit() {
        let mut server = mockito::Server::new_async().await;
        // Create a body larger than our small limit
        let big_body = "x".repeat(500);
        let mock = server
            .mock("GET", "/big")
            .with_status(200)
            .with_header("content-type", "text/plain")
            .with_body(&big_body)
            .create_async()
            .await;

        let config = WebFetchConfig {
            max_fetch_bytes: 100,
            ..Default::default()
        };
        let tool = WebFetchTool::with_config(config);
        let result = tool
            .execute(json!({"url": format!("{}/big", server.url())}))
            .await
            .unwrap();

        assert!(result.success);
        assert!(
            result.output.contains("[Content truncated at 100 bytes. Full page is larger.]"),
            "output was: {}",
            result.output
        );
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn fetch_does_not_truncate_small_response() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/small")
            .with_status(200)
            .with_header("content-type", "text/plain")
            .with_body("Hello world")
            .create_async()
            .await;

        let tool = WebFetchTool::new();
        let result = tool
            .execute(json!({"url": format!("{}/small", server.url())}))
            .await
            .unwrap();

        assert!(result.success);
        assert!(!result.output.contains("[Content truncated"));
        assert!(result.output.contains("Hello world"));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn fetch_rejects_binary_content_type() {
        let mut server = mockito::Server::new_async().await;
        let body = "fake png data that is not real";
        let mock = server
            .mock("GET", "/image.png")
            .with_status(200)
            .with_header("content-type", "image/png")
            .with_header("content-length", &body.len().to_string())
            .with_body(body)
            .create_async()
            .await;

        let tool = WebFetchTool::new();
        let result = tool
            .execute(json!({"url": format!("{}/image.png", server.url())}))
            .await
            .unwrap();

        assert!(result.success);
        assert!(
            result.output.contains("Binary content"),
            "output was: {}",
            result.output
        );
        assert!(result.output.contains("image/png"));
        assert!(result.output.contains("Not fetched"));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn fetch_rejects_octet_stream() {
        let mut server = mockito::Server::new_async().await;
        let body = "binary data";
        let mock = server
            .mock("GET", "/file.bin")
            .with_status(200)
            .with_header("content-type", "application/octet-stream")
            .with_header("content-length", &body.len().to_string())
            .with_body(body)
            .create_async()
            .await;

        let tool = WebFetchTool::new();
        let result = tool
            .execute(json!({"url": format!("{}/file.bin", server.url())}))
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.output.contains("Binary content"));
        assert!(result.output.contains("application/octet-stream"));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn fetch_redirect_limit_exceeded() {
        let mut server = mockito::Server::new_async().await;
        // Create a redirect loop: /redir -> /redir (self-redirect)
        let url = format!("{}/redir", server.url());
        // reqwest sends 1 original request + max_redirects follow-ups
        let _mock = server
            .mock("GET", "/redir")
            .expect_at_least(1)
            .with_status(302)
            .with_header("location", &url)
            .create_async()
            .await;

        let config = WebFetchConfig {
            max_redirects: 2,
            ..Default::default()
        };
        let tool = WebFetchTool::with_config(config);
        let result = tool
            .execute(json!({"url": url}))
            .await
            .unwrap();

        assert!(!result.success);
        let err = result.error.unwrap();
        assert!(
            err.contains("Too many redirects (>2)"),
            "error was: {err}"
        );
    }

    // ── Brave API response deserialization ───────────────────

    #[test]
    fn brave_response_deserializes_with_results() {
        let json_str = r#"{
            "web": {
                "results": [
                    {
                        "title": "Rust Programming Language",
                        "url": "https://www.rust-lang.org/",
                        "description": "A systems programming language."
                    },
                    {
                        "title": "Rust by Example",
                        "url": "https://doc.rust-lang.org/rust-by-example/",
                        "description": "Learn Rust by example."
                    }
                ]
            }
        }"#;
        let resp: BraveSearchResponse = serde_json::from_str(json_str).unwrap();
        let results = resp.web.unwrap().results;
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Rust Programming Language");
        assert_eq!(results[1].url, "https://doc.rust-lang.org/rust-by-example/");
    }

    #[test]
    fn brave_response_deserializes_empty() {
        let json_str = r#"{}"#;
        let resp: BraveSearchResponse = serde_json::from_str(json_str).unwrap();
        assert!(resp.web.is_none());
    }

    #[test]
    fn brave_response_deserializes_empty_results() {
        let json_str = r#"{"web": {"results": []}}"#;
        let resp: BraveSearchResponse = serde_json::from_str(json_str).unwrap();
        assert!(resp.web.unwrap().results.is_empty());
    }

    // ── Integration-style test with complex HTML ────────────

    #[test]
    fn strip_html_handles_real_world_page() {
        let html = r#"<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
    <style>
        body { font-family: sans-serif; }
        .hidden { display: none; }
    </style>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            console.log('loaded');
        });
    </script>
</head>
<body>
    <h1>Main Heading</h1>
    <p>This is the first paragraph with <a href="/link">a link</a>.</p>
    <div class="content">
        <h2>Section Two</h2>
        <p>Second paragraph with &amp; special &lt;chars&gt;.</p>
        <ul>
            <li>Item one</li>
            <li>Item two</li>
        </ul>
    </div>
    <script>var x = 42;</script>
    <p>Final paragraph.</p>
</body>
</html>"#;

        let text = strip_html_tags(html);
        // Should contain visible text
        assert!(text.contains("Main Heading"));
        assert!(text.contains("first paragraph"));
        assert!(text.contains("a link"));
        assert!(text.contains("Section Two"));
        assert!(text.contains("& special <chars>"));
        assert!(text.contains("Item one"));
        assert!(text.contains("Item two"));
        assert!(text.contains("Final paragraph"));

        // Should NOT contain script or style content
        assert!(!text.contains("console.log"));
        assert!(!text.contains("DOMContentLoaded"));
        assert!(!text.contains("font-family"));
        assert!(!text.contains("display: none"));
        assert!(!text.contains("var x = 42"));
    }
}

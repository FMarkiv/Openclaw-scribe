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
}

impl WebFetchTool {
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::limited(5))
            .user_agent("ZeroClaw/1.0 (research agent)")
            .build()
            .unwrap_or_else(|_| Client::new());
        Self { client }
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

        let response = match self.client.get(&url).send().await {
            Ok(r) => r,
            Err(e) => {
                let msg = if e.is_timeout() {
                    format!("Request timed out fetching: {url}")
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

        let body = match response.text().await {
            Ok(text) => text,
            Err(e) => {
                return Ok(ToolExecutionResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("Failed to read response body: {e}")),
                });
            }
        };

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
/// let mut tools = tools::all_tools(&security, mem.clone(), composio_key, &config.browser);
/// tools.extend(web_tools::all_web_tools(config.brave_api_key.clone()));
/// ```
pub fn all_web_tools(brave_api_key: Option<String>) -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(WebSearchTool::new(brave_api_key)),
        Box::new(WebFetchTool::new()),
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
        let tools = all_web_tools(None);
        assert_eq!(tools.len(), 2);

        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"web_search"));
        assert!(names.contains(&"web_fetch"));
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

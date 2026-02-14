//! Silent turn support for ZeroClaw.
//!
//! Allows the agent to perform background operations (memory flush,
//! heartbeat tasks, startup summaries) without producing user-visible
//! output. A response that starts with the `NO_REPLY` token is still
//! persisted in session history but suppressed from the display.
//!
//! ## How it works
//!
//! 1. The agent loop checks each assistant response with `is_silent_response()`
//! 2. If silent, the response is persisted to the session JSONL but not
//!    displayed to the user
//! 3. The `NO_REPLY` prefix can be stripped via `strip_no_reply_token()`
//!    before logging to daily notes
//!
//! ## Usage
//!
//! ```rust
//! use crate::memory::silent::{NO_REPLY, is_silent_response, strip_no_reply_token};
//!
//! let response = "[NO_REPLY] Memory flush completed, 3 entries saved.";
//! assert!(is_silent_response(response));
//! assert_eq!(
//!     strip_no_reply_token(response),
//!     "Memory flush completed, 3 entries saved."
//! );
//! ```

/// The NO_REPLY token. When an agent response starts with this token,
/// the response is suppressed from user-visible output but still
/// persisted in session history.
///
/// This enables background operations like memory flush, heartbeat
/// tasks, and startup summaries to run without interrupting the user.
pub const NO_REPLY: &str = "[NO_REPLY]";

/// Check whether an agent response is a silent turn.
///
/// Returns `true` if the response content starts with the `NO_REPLY` token
/// (ignoring leading whitespace).
pub fn is_silent_response(content: &str) -> bool {
    content.trim_start().starts_with(NO_REPLY)
}

/// Strip the `NO_REPLY` token from a response, returning the remaining content.
///
/// If the response does not start with `NO_REPLY`, returns the original
/// content unchanged.
pub fn strip_no_reply_token(content: &str) -> &str {
    let trimmed = content.trim_start();
    if trimmed.starts_with(NO_REPLY) {
        trimmed[NO_REPLY.len()..].trim_start()
    } else {
        content
    }
}

/// Build a silent turn system prompt that instructs the agent to respond
/// with the NO_REPLY prefix.
///
/// Wraps the given instruction in a system message that tells the agent
/// to prefix its response with `[NO_REPLY]` so it won't be shown to the
/// user.
pub fn build_silent_prompt(instruction: &str) -> String {
    format!(
        "SYSTEM: This is a background task. Prefix your entire response with \
         {NO_REPLY} so it is not shown to the user. The response will still be \
         saved to session history and daily notes.\n\n\
         Task: {instruction}"
    )
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── NO_REPLY constant ────────────────────────────────────────

    #[test]
    fn no_reply_constant_has_expected_value() {
        assert_eq!(NO_REPLY, "[NO_REPLY]");
    }

    #[test]
    fn no_reply_constant_is_bracketed() {
        assert!(NO_REPLY.starts_with('['));
        assert!(NO_REPLY.ends_with(']'));
    }

    // ── is_silent_response ───────────────────────────────────────

    #[test]
    fn is_silent_detects_no_reply_prefix() {
        assert!(is_silent_response("[NO_REPLY] some content"));
        assert!(is_silent_response("[NO_REPLY]"));
        assert!(is_silent_response("[NO_REPLY] "));
    }

    #[test]
    fn is_silent_detects_no_reply_with_content() {
        assert!(is_silent_response(
            "[NO_REPLY] Memory flush completed, 3 entries saved."
        ));
        assert!(is_silent_response(
            "[NO_REPLY] Heartbeat: all tasks completed successfully."
        ));
    }

    #[test]
    fn is_silent_ignores_leading_whitespace() {
        assert!(is_silent_response("  [NO_REPLY] content"));
        assert!(is_silent_response("\n[NO_REPLY] content"));
        assert!(is_silent_response("\t[NO_REPLY] content"));
        assert!(is_silent_response("  \n\t  [NO_REPLY] content"));
    }

    #[test]
    fn is_silent_rejects_normal_responses() {
        assert!(!is_silent_response("Hello, how can I help?"));
        assert!(!is_silent_response(""));
        assert!(!is_silent_response("NO_REPLY without brackets"));
        assert!(!is_silent_response("Some text [NO_REPLY] in middle"));
    }

    #[test]
    fn is_silent_rejects_partial_matches() {
        assert!(!is_silent_response("[NO_REPLY"));
        assert!(!is_silent_response("NO_REPLY]"));
        assert!(!is_silent_response("[no_reply] lowercase"));
        assert!(!is_silent_response("[No_Reply] mixed case"));
    }

    // ── strip_no_reply_token ─────────────────────────────────────

    #[test]
    fn strip_no_reply_extracts_content() {
        assert_eq!(
            strip_no_reply_token("[NO_REPLY] Memory flush completed."),
            "Memory flush completed."
        );
        assert_eq!(
            strip_no_reply_token("[NO_REPLY] Heartbeat done."),
            "Heartbeat done."
        );
    }

    #[test]
    fn strip_no_reply_handles_no_space_after_token() {
        assert_eq!(
            strip_no_reply_token("[NO_REPLY]content"),
            "content"
        );
    }

    #[test]
    fn strip_no_reply_handles_leading_whitespace() {
        assert_eq!(
            strip_no_reply_token("  [NO_REPLY]   some content  "),
            "some content  "
        );
    }

    #[test]
    fn strip_no_reply_preserves_non_silent_content() {
        let content = "Normal response without token";
        assert_eq!(strip_no_reply_token(content), content);
    }

    #[test]
    fn strip_no_reply_handles_empty_after_token() {
        assert_eq!(strip_no_reply_token("[NO_REPLY]"), "");
        assert_eq!(strip_no_reply_token("[NO_REPLY] "), "");
        assert_eq!(strip_no_reply_token("[NO_REPLY]  "), "");
    }

    #[test]
    fn strip_no_reply_preserves_multiline_content() {
        let content = "[NO_REPLY] Line one\nLine two\nLine three";
        assert_eq!(
            strip_no_reply_token(content),
            "Line one\nLine two\nLine three"
        );
    }

    // ── build_silent_prompt ──────────────────────────────────────

    #[test]
    fn build_silent_prompt_includes_instruction() {
        let prompt = build_silent_prompt("Flush memory to daily note");
        assert!(prompt.contains("Flush memory to daily note"));
    }

    #[test]
    fn build_silent_prompt_includes_no_reply_token() {
        let prompt = build_silent_prompt("test task");
        assert!(prompt.contains("[NO_REPLY]"));
    }

    #[test]
    fn build_silent_prompt_marks_as_background() {
        let prompt = build_silent_prompt("test");
        assert!(prompt.contains("background task"));
        assert!(prompt.contains("not shown to the user"));
    }

    #[test]
    fn build_silent_prompt_includes_system_prefix() {
        let prompt = build_silent_prompt("test");
        assert!(prompt.starts_with("SYSTEM:"));
    }

    #[test]
    fn build_silent_prompt_includes_persistence_note() {
        let prompt = build_silent_prompt("test");
        assert!(prompt.contains("session history"));
        assert!(prompt.contains("daily notes"));
    }
}

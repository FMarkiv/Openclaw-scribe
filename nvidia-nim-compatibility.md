# NVIDIA NIM Compatibility Analysis for Zeroclaw's Compatible Provider

**Date**: 2026-02-14
**Target**: NVIDIA NIM API at `https://integrate.api.nvidia.com/v1/`
**Provider**: `OpenAiCompatibleProvider` (`custom:https://integrate.api.nvidia.com/v1`)

---

## TL;DR

**Basic chat completions will work out of the box. Tool use (function calling)
will NOT work — zeroclaw has no tool_use/function_calling implementation at all.**

This is not a NIM-specific problem. It's an architectural gap in zeroclaw: the
agent loop never sends tool definitions to any provider and never parses
tool_call responses. Tools exist in code but are invoked only via prompt
engineering (the LLM is told about tools in the system prompt and expected to
emit text that looks like a tool invocation, which the agent... doesn't parse
either).

---

## Detailed Analysis

### 1. URL Construction — WILL DOUBLE `/v1/`

The Compatible provider constructs URLs like this (`compatible.rs:112`):

```rust
let url = format!("{}/v1/chat/completions", self.base_url);
```

If you configure:
```
custom:https://integrate.api.nvidia.com/v1/
```

The trailing slash is stripped (`compatible.rs:36`):
```rust
base_url: base_url.trim_end_matches('/').to_string(),
```

So `base_url` becomes `https://integrate.api.nvidia.com/v1`, and the final URL
becomes:

```
https://integrate.api.nvidia.com/v1/v1/chat/completions
                                   ^^^^ DOUBLED
```

**Fix**: Configure as `custom:https://integrate.api.nvidia.com` (without `/v1`).

### 2. Auth — WORKS

NVIDIA NIM uses `Authorization: Bearer <key>`, which is the default `AuthStyle::Bearer`
used by the `custom:` factory (`providers/mod.rs:102`). No issues here.

### 3. Request Format — WORKS (for plain chat)

The `ChatRequest` struct (`compatible.rs:48-53`):
```rust
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f64,
}
```

This serializes to standard OpenAI format:
```json
{
  "model": "meta/llama-3.1-70b-instruct",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "temperature": 0.7
}
```

NIM accepts this. Plain chat completions will work.

### 4. Response Format — WORKS (for plain chat)

The `ResponseMessage` struct (`compatible.rs:71-74`):
```rust
struct ResponseMessage {
    content: String,
}
```

NIM returns standard OpenAI format:
```json
{
  "choices": [{"message": {"content": "Hello!", "role": "assistant"}}]
}
```

The `content: String` field will deserialize fine. Extra fields like `role` are
ignored by serde's default behavior. This works.

### 5. Tool Use / Function Calling — DOES NOT WORK (fundamental gap)

This is the critical finding. Here's what's missing:

#### a) No tool definitions are sent in API requests

The `ChatRequest` struct has no `tools` or `functions` field. The agent loop
(`agent/loop_.rs:62`) creates tool objects but only uses them to build a text
description in the system prompt:

```rust
let _tools = tools::all_tools(&security, mem.clone(), composio_key, &config.browser);
// ^^^ stored but NEVER passed to the provider
```

Instead, tool names/descriptions are injected as plain text in the system prompt
(`agent/loop_.rs:85-98`):
```rust
let mut tool_descs: Vec<(&str, &str)> = vec![
    ("shell", "Execute terminal commands"),
    ("file_read", "Read file contents"),
    ...
];
```

This becomes:
```
## Tools
- **shell**: Execute terminal commands
- **file_read**: Read file contents
```

There is **no structured tool/function definition** sent to any API.

#### b) No tool_call response parsing

The Provider trait returns `String`:
```rust
async fn chat_with_system(...) -> anyhow::Result<String>;
```

When the LLM responds with a `tool_calls` array (as NIM and OpenAI do for
function calling), the response looks like:

```json
{
  "choices": [{
    "message": {
      "content": null,
      "tool_calls": [{"function": {"name": "shell", "arguments": "{\"command\": \"ls\"}"}}]
    }
  }]
}
```

Zeroclaw's `ResponseMessage` only has `content: String`. When `content` is
`null` (which it is during tool_calls), **serde deserialization will fail**
because `String` can't deserialize from JSON `null`. The request will error out.

#### c) No tool execution loop

Even if you could parse tool_calls, there's no loop that:
1. Receives a tool_call from the LLM
2. Executes the tool
3. Sends the result back as a `tool` message
4. Gets the LLM's final response

The agent loop (`agent/loop_.rs:125-128`) is a single request-response:
```rust
let response = provider
    .chat_with_system(Some(&system_prompt), &enriched, model_name, temperature)
    .await?;
println!("{response}");
```

One shot. No iteration.

### 6. What About the Tool Trait Infrastructure?

Zeroclaw has a well-built `Tool` trait with `ToolSpec`, `ToolResult`, JSON
schemas, and real implementations (shell, file_read, etc.). But this
infrastructure is **never connected to the provider API**. The tools are defined
and can execute, but:

- `ToolSpec` is never serialized into an OpenAI `tools` array
- `Tool::execute()` is never called from the agent loop
- `ToolResult` is never formatted as a tool-role message

The tool infrastructure appears to be built for future use but is not wired up
to the agent loop or any provider.

---

## Summary: What Works, What Doesn't

| Feature | NVIDIA NIM | Status |
|---|---|---|
| HTTPS + Bearer auth | Yes | WORKS |
| Chat completions (plain text) | Yes | WORKS (fix URL) |
| System prompt | Yes | WORKS |
| Temperature | Yes | WORKS |
| Tool/function definitions in request | Required for tool use | NOT SENT |
| Parsing `tool_calls` from response | Required for tool use | NOT IMPLEMENTED |
| Tool execution loop | Required for agentic behavior | NOT IMPLEMENTED |
| `content: null` responses | Happens during tool_calls | WILL CRASH (serde) |

## To Make It Work as a Chat Bot (No Tools)

Configure as:
```toml
provider = "custom:https://integrate.api.nvidia.com"
model = "meta/llama-3.1-70b-instruct"
api_key = "nvapi-..."
```

Note: use `https://integrate.api.nvidia.com` **without** `/v1` to avoid the
doubled path. This will give you a working chat agent with memory context
injection, but no actual tool execution.

## To Make Tool Use Work (What Would Need to Change)

This is a significant architectural change, not a small patch:

1. **Provider trait**: Return a structured response type instead of `String`
   (with variants for text content and tool_calls)
2. **Request structs**: Add optional `tools` field with JSON schema definitions
3. **Response structs**: Handle `content: Option<String>` and
   `tool_calls: Option<Vec<ToolCall>>`
4. **Agent loop**: Implement a loop that executes tools and feeds results back
5. **Message history**: Track multi-turn conversation with tool results

This is the difference between "chat wrapper with memory" and "agentic tool-use
framework". Zeroclaw currently is the former with the infrastructure to become
the latter.

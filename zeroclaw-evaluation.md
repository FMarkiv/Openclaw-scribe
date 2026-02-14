# Zeroclaw Evaluation Report

**Repository**: https://github.com/theonlyhennygod/zeroclaw
**Date**: 2026-02-14
**Verdict**: **FUNCTIONAL — This is a real, deployable project, not scaffolding.**

---

## 1. cargo build --release

**Result**: SUCCESS — clean build, zero errors, zero warnings.

Compiles 150+ crates including reqwest, rusqlite, tokio, hyper, and produces a
release binary. Build took ~1m27s.

---

## 2. cargo test

**Result**: **648 tests passed, 0 failed, 0 ignored.**

| Test Suite | Passed |
|---|---|
| Unit tests (lib) | 641 |
| Integration tests (memory_comparison) | 7 |
| Doc tests | 0 (none written) |

**Tests are real assertions, not stubs.** Examples:
- `shell_executes_allowed_command` — actually runs a shell command and checks output
- `file_read_existing_file` / `file_write_creates_file` — real filesystem I/O
- `store_core` / `recall_finds_match` — writes to SQLite and queries back
- `encrypt_decrypt_roundtrip` — tests actual crypto operations
- `compare_recall_quality` / `compare_recall_speed` — integration benchmarks for memory

---

## 3. src/ Structure — Code vs Boilerplate

**60 Rust source files, ~18,500 lines total.**

| Category | Lines | % |
|---|---|---|
| Implementation logic | ~13,200 | 71% |
| Traits/structs/imports/comments | ~5,300 | 29% |

Major modules by size:
- `onboard/wizard.rs` — 2,410 lines (interactive setup wizard)
- `memory/sqlite.rs` — 1,365 lines (full memory engine)
- `config/schema.rs` — 1,116 lines (configuration)
- `integrations/registry.rs` — 856 lines (integration dispatch)
- `security/policy.rs` — 814 lines (security sandboxing)
- `channels/mod.rs` — 780 lines (channel routing)
- `skills/mod.rs` — 633 lines (skills/actions system)

This is not a project with a big README and empty src/. The code-to-boilerplate
ratio is healthy.

---

## 4. Provider Trait — Real HTTP Implementations

**YES — all providers make actual HTTP calls via reqwest.**

| Provider | Endpoint | Auth | Timeout |
|---|---|---|---|
| Anthropic | `https://api.anthropic.com/v1/messages` | x-api-key header | 120s |
| OpenAI | `https://api.openai.com/v1/chat/completions` | Bearer token | 120s |
| Ollama | `http://localhost:11434/api/chat` | None (local) | 300s |
| OpenRouter | OpenAI-compatible | Bearer token | 120s |
| Compatible | Configurable base URL | Bearer token | 120s |

Each provider:
- Constructs proper JSON request bodies with system/user messages
- Sets correct headers (content-type, auth, anthropic-version)
- Parses structured JSON responses
- Returns errors via anyhow for non-200 status codes

**Not stubs.** These are complete API clients.

---

## 5. Memory System — SQLite + FTS5 + Vector Search

**FULLY IMPLEMENTED.**

The SQLite memory engine (`memory/sqlite.rs`, 1,365 lines) includes:

- **Schema**: `memories` table with id, key, content, category, embedding (BLOB),
  timestamps. Plus `memories_fts` FTS5 virtual table with auto-sync triggers.
- **FTS5 search**: BM25-ranked full-text search with proper escaping
- **Vector search**: Cosine similarity over stored embeddings, loaded from BLOBs
- **Hybrid merge**: Weighted fusion of FTS5 + vector scores with normalization
- **Embedding cache**: LRU-evicted cache table to avoid redundant API calls
- **Embedding providers**: OpenAI embedding API (real HTTP calls) + noop fallback
- **Recall pipeline**: Query embedding → FTS5 search → vector search → hybrid
  merge → LIKE fallback if empty → return scored results
- **Full CRUD**: store, recall, get, list, forget, count, health_check

The vector operations (`memory/vector.rs`, 402 lines) include actual cosine
similarity computation, score normalization, and serialization to/from byte
arrays for BLOB storage.

**Not stubs.** This is a working semantic memory system.

---

## 6. Tools — Actually Execute

**ALL TOOLS EXECUTE REAL OPERATIONS.**

| Tool | Implementation | Evidence |
|---|---|---|
| shell | `tokio::process::Command` with `sh -c` | 60s timeout, stdout/stderr capture, exit code |
| file_read | `tokio::fs::read_to_string` | Path validation, workspace sandboxing |
| file_write | `tokio::fs::write` | Creates parent dirs, reports bytes written |
| memory_store | Calls `memory.store()` → SQLite INSERT + embedding | Full pipeline |
| memory_recall | Calls `memory.recall()` → hybrid search | FTS5 + vector + fallback |
| memory_forget | Calls `memory.forget()` → SQLite DELETE | Confirmed by tests |
| browser_open | URL validation, domain allowlist, SSRF protection | 465 lines |
| composio | External integration platform | API client |

Security is enforced at the tool level:
- Shell commands checked against allowlist before execution
- File paths validated against workspace root (no traversal)
- Rate limiting per action type
- Readonly mode blocks all mutations

---

## Verdict

**Zeroclaw is a real, functional AI agent framework.** It is not scaffolding.

**What works**:
- Compiles cleanly with zero warnings
- 648 tests pass with real assertions (file I/O, shell execution, SQLite queries, crypto)
- 5 LLM providers with actual HTTP API calls (Anthropic, OpenAI, Ollama, OpenRouter, compatible)
- Full semantic memory with SQLite + FTS5 + vector cosine similarity + hybrid search
- All core tools (shell, file_read, file_write, memory_store/recall/forget) execute real operations
- Security sandboxing (command allowlists, path validation, rate limiting)
- Multi-channel support (CLI, Discord, Slack, Telegram, Matrix, iMessage)
- ~18,500 lines of Rust, 71% implementation logic

**What it needs to run on a Raspberry Pi**:
- An API key for at least one provider (Anthropic, OpenAI) — or a local Ollama instance
- ARM cross-compilation (standard Rust cross-compile, no exotic deps beyond SQLite)
- A `zeroclaw.toml` config file (the onboarding wizard generates one interactively)

**Recommendation**: This is worth a live test. Set up a provider key and run it.

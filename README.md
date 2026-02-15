# Openclaw-Scribe

**A lightweight AI agent in 1.5MB — OpenClaw reimagined for embedded devices.**

[![Rust](https://img.shields.io/badge/Rust-2021_edition-orange?logo=rust)](https://www.rust-lang.org/)
[![ARM64](https://img.shields.io/badge/ARM64-aarch64--musl-blue)](https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/)
[![MIT](https://img.shields.io/badge/License-MIT-green)](#license)

---

## Acknowledgments

This project is inspired by and builds upon concepts from:

- **[OpenClaw](https://github.com/openclaw/openclaw)** — the open-source AI agent platform with 145K+ stars that pioneered multi-channel AI agents with memory, tool use, and heartbeat systems.
- **[ZeroClaw](https://github.com/theonlyhennygod/zeroclaw)** — the original Rust-based lightweight agent that provided the foundation and starting architecture for this project.

Openclaw-Scribe is **not affiliated with either project**. It is an independent reimplementation optimized for single-user embedded devices.

---

## Why This Exists

OpenClaw is ~430K lines of TypeScript requiring Node.js — too heavy for 512MB RAM devices. Openclaw-Scribe delivers ~80% of OpenClaw's core functionality in a **1.5MB static ARM binary** with **<10MB runtime memory**.

Designed for the **Scribe device**: a portable e-ink developer terminal built on Raspberry Pi Zero 2W.

| | OpenClaw | Openclaw-Scribe |
|---|---|---|
| Language | TypeScript | Rust |
| Binary size | ~200MB (Node.js + deps) | ~1.5MB (static musl) |
| Runtime memory | ~300MB+ | <10MB |
| Dependencies | Node.js, npm, PostgreSQL | None (static binary) |
| Target hardware | Cloud servers | Raspberry Pi Zero 2W |

---

## Features

### Agent Core
- Multi-turn tool use with automatic tool-call → result looping
- 5 LLM providers: Anthropic, OpenAI, Ollama, OpenRouter, Compatible (any OpenAI-compatible API)
- Automatic model failover with ordered fallback chain
- Context window management with auto-compaction at 75% capacity

### Memory System
- Markdown-based memory files — human-readable, git-friendly
- `SOUL.md` — agent identity and behavior rules
- `USER.md` — user preferences and project context
- `MEMORY.md` — curated long-term knowledge
- `memory/YYYY-MM-DD.md` — daily session notes (append-only)
- Automatic morning summaries on first session of each day

### Session Management
- Named sessions with `/new`, `/switch`, `/sessions`, `/rename`
- JSONL persistence — each turn saved to `sessions/<name>/session.jsonl`
- Automatic session resume on startup
- Context compaction: summarize and prune when approaching token limits

### Tools
| Tool | Description |
|------|-------------|
| `shell` | Execute shell commands (with optional Docker sandbox) |
| `str_replace` | Surgical find-and-replace in files |
| `web_search` | Search the web via Brave Search API |
| `web_fetch` | Fetch and parse URLs to plaintext |
| `memory_store` | Append timestamped note to today's daily log |
| `memory_recall` | Search across all markdown memory files |
| `memory_flush` | Write session summary before exit |
| `memory_promote` | Move content from daily notes to MEMORY.md |

### Multi-Agent
- Separate agent personas in `~/.zeroclaw/agents/<name>/`
- Per-agent `SOUL.md`, `agent.toml`, memory, sessions, and workspace
- Optional `tools.toml` to restrict which tools an agent can use
- Switch between agents with `/agent <name>`

### Telegram Bot
- HTTP long-polling — no public IP or webhooks required
- Shared agent loop, memory, and session persistence with CLI
- MarkdownV2 formatting with message splitting at 4096 chars
- File upload/download, emoji reactions, message editing during tool chains

### Background Tasks
- Heartbeat system: recurring tasks defined in `HEARTBEAT.md`
- Async subagent spawning via `tokio::spawn` — non-blocking
- Silent turns with `[NO_REPLY]` token for background operations
- `/tasks` command to monitor running/completed subagents

### Sandbox
- Optional Docker isolation for shell commands
- Alpine-based container with bash, git, python3, build tools
- Configurable memory limits, network access, and timeouts
- Ideal for untrusted inputs (e.g., Telegram users)

---

## Quick Start

### Prerequisites

- **To build from source:** [Rust toolchain](https://rustup.rs/) (1.70+)
- **To run a prebuilt binary:** just `scp` it to your device — no dependencies

### Build and Deploy

```bash
# Clone the repo
git clone https://github.com/FMarkiv/Openclaw-scribe.git
cd Openclaw-scribe

# Build for ARM64 (Raspberry Pi Zero 2W)
cargo build --release --target aarch64-unknown-linux-musl

# Deploy to your Pi
scp target/aarch64-unknown-linux-musl/release/zeroclaw pi@device:~/

# Or build natively on the Pi
cargo build --release
```

### Configure

Copy the example config and add your API key:

```bash
mkdir -p ~/.zeroclaw
cp config.toml ~/.zeroclaw/config.toml
```

Minimal `~/.zeroclaw/config.toml`:

```toml
[provider]
primary = "anthropic"
api_key = "sk-ant-..."
model = "claude-sonnet-4-20250514"
```

### Run

```bash
./zeroclaw
```

Or set the workspace path:

```bash
ZEROCLAW_WORKSPACE=~/my-project ./zeroclaw
```

---

## Configuration Reference

Complete annotated `config.toml`:

```toml
# ── Provider ─────────────────────────────────────────────────────
[provider]
primary = "anthropic"          # anthropic | openai | ollama | openrouter | compatible
api_key = "sk-ant-..."         # Or set ANTHROPIC_API_KEY / OPENAI_API_KEY env var
model = "claude-sonnet-4-20250514"

# ── Fallback Providers ──────────────────────────────────────────
# Ordered list — tried in sequence when the primary fails with a
# retryable error (429, 5xx, connection timeout).
# Auth errors (401, 403) skip the provider entirely.
# On the next user turn, the primary is always tried first.

[[provider.fallbacks]]
provider = "openai"
model = "gpt-4o"

[[provider.fallbacks]]
provider = "openrouter"
model = "anthropic/claude-sonnet-4-20250514"

# ── Workspace ────────────────────────────────────────────────────
[workspace]
root = "."                     # Contains SOUL.md, MEMORY.md, etc.

# ── Telegram Bot ─────────────────────────────────────────────────
[telegram]
bot_token = "123456789:AAF-your-bot-token-here"
enabled = true
poll_interval_secs = 5         # Polling frequency (default: 5s)
long_poll_timeout_secs = 30    # Telegram long-poll hold time (default: 30s)

# ── Sandbox ──────────────────────────────────────────────────────
[sandbox]
enabled = false                # Opt-in, default off
image = "zeroclaw-sandbox:latest"
workspace = "/workspace"       # Mount point inside container
timeout_seconds = 30           # Per-command timeout
memory_limit = "256m"          # Container memory cap
network = false                # Disable network in sandbox
mount_paths = []               # Additional read-only mounts
```

---

## Architecture

### System Overview

```
┌──────────────────┐     ┌───────────────────────┐     ┌────────────┐
│  Display          │     │  Openclaw-Scribe       │     │  LLM APIs  │
│  (Python/IT8951)  │◄──► │  (Rust, ~1.5MB)        │◄──► │  Anthropic │
│  E-ink refresh    │     │                         │     │  OpenAI    │
└──────────────────┘     │  ┌─────────────────┐   │     │  Ollama    │
        ▲                 │  │ Agent Loop       │   │     │  OpenRouter│
        │                 │  │ receive → tools  │   │     │  Compatible│
        ▼                 │  │ → respond        │   │     └────────────┘
┌──────────────────┐     │  └─────────────────┘   │
│  tmux             │◄──► │  ┌─────────────────┐   │
│  (session mux)    │     │  │ Memory (markdown) │   │
└──────────────────┘     │  │ Sessions (JSONL)  │   │
                          │  │ Context manager   │   │
┌──────────────────┐     │  └─────────────────┘   │
│  Telegram Bot     │◄──► │  ┌─────────────────┐   │
│  (long-polling)   │     │  │ Subagent manager  │   │
└──────────────────┘     │  │ Heartbeat         │   │
                          │  │ Sandbox (Docker)  │   │
                          │  └─────────────────┘   │
                          └───────────────────────┘
```

### Agent Loop

```
User input (CLI or Telegram)
    │
    ├─► Slash command? ──► Handle locally (/new, /switch, /agent, etc.)
    │
    └─► Send to LLM with system prompt + conversation history
            │
            ├─► Tool calls? ──► Execute tools ──► Append results ──► Loop back to LLM
            │
            └─► Text response ──► Display to user ──► Persist to session JSONL
```

### Workspace Layout

```
~/.zeroclaw/
├── config.toml              — Configuration file
├── SOUL.md                  — Agent identity & behavior rules
├── USER.md                  — User preferences & project context
├── MEMORY.md                — Curated long-term knowledge
├── HEARTBEAT.md             — Recurring background tasks
├── memory/
│   ├── 2026-02-14.md        — Yesterday's daily note
│   └── 2026-02-15.md        — Today's daily note
├── sessions/
│   ├── default/
│   │   ├── session.jsonl    — Conversation turns
│   │   └── session.json     — Session metadata
│   └── my-project/
│       ├── session.jsonl
│       └── session.json
└── agents/
    └── researcher/
        ├── SOUL.md           — Agent-specific personality
        ├── agent.toml        — Name, description, provider override
        ├── tools.toml        — Tool allowlist
        ├── MEMORY.md         — Isolated memory
        ├── memory/           — Isolated daily notes
        └── sessions/         — Isolated sessions
```

### Source Layout

```
src/
├── main.rs                  — Binary entry point, initialization
├── lib.rs                   — Library exports (pub mod memory, tools)
├── tools.rs                 — Tool trait: name, description, parameters_schema, execute
└── memory/
    ├── mod.rs               — Module declarations, Memory trait, MemoryCategory
    ├── markdown.rs          — MarkdownMemory: load context, append daily notes, search
    ├── markdown_tools.rs    — memory_store, memory_recall, memory_flush, memory_promote
    ├── session.rs           — SessionManager: JSONL persistence, session metadata
    ├── session_tools.rs     — /new, /switch, /sessions, /rename commands
    ├── context.rs           — ContextManager: token counting, auto-compaction, pruning
    ├── agent.rs             — AgentManager: multi-agent support, per-agent config
    ├── agent_tools.rs       — /agent, /agents, /agent-new, /agent-delete commands
    ├── failover.rs          — ProviderChain: primary + fallbacks, error classification
    ├── provider_tools.rs    — /provider command, failover status display
    ├── heartbeat.rs         — HeartbeatManager: read HEARTBEAT.md, execute as silent turns
    ├── subagent.rs          — SubagentManager: async background tasks, /tasks command
    ├── silent.rs            — [NO_REPLY] token for background operations
    ├── startup.rs           — StartupManager: morning summaries on first session of day
    ├── telegram.rs          — TelegramListener: long-polling, message routing
    ├── telegram_rich.rs     — MarkdownV2 formatting, file ops, reactions, message editing
    ├── web_tools.rs         — web_search (Brave API), web_fetch (HTML→plaintext)
    ├── file_tools.rs        — str_replace tool: surgical find-and-replace
    └── sandbox.rs           — Docker sandbox: shell tool, container lifecycle
```

---

## Commands Reference

All slash commands are handled locally before reaching the LLM.

### Session Commands

| Command | Description |
|---------|-------------|
| `/new <name>` | Start a new named session |
| `/switch <name>` | Resume an existing session |
| `/sessions` | List all sessions with metadata |
| `/rename <name>` | Rename the current session |

### Agent Commands

| Command | Description |
|---------|-------------|
| `/agent <name>` | Switch to a different agent |
| `/agents` | List all agents with descriptions |
| `/agent-new <name>` | Create a new agent |
| `/agent-delete <name>` | Delete an agent and its data |

### Other Commands

| Command | Description |
|---------|-------------|
| `/provider` | Show provider chain and recent failover events |
| `/heartbeat` | Manually trigger heartbeat cycle |
| `/tasks` | List running/completed background subagents |

---

## Building from Source

### Native Build

```bash
cargo build --release
```

The binary is at `target/release/zeroclaw`.

### Cross-Compilation for ARM64

The project includes cross-compilation config for `aarch64-unknown-linux-musl` (Raspberry Pi Zero 2W). Prerequisites: `clang`, `llvm-ar`, and the Rust target.

```bash
# Install the target
rustup target add aarch64-unknown-linux-musl

# Build (uses .cargo/config.toml for linker/compiler settings)
cargo build --release --target aarch64-unknown-linux-musl
```

The static binary is at `target/aarch64-unknown-linux-musl/release/zeroclaw`.

### Docker Sandbox Image

If using the sandbox feature:

```bash
docker build -f Dockerfile.sandbox -t zeroclaw-sandbox:latest .
```

### Running Tests

```bash
cargo test --lib
```

---

## License

MIT

---

## Contributing

Contributions are welcome. Please [open an issue](https://github.com/FMarkiv/Openclaw-scribe/issues) for bug reports, feature requests, or questions before submitting large changes.

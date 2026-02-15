# Openclaw-scribe

Patches, documentation, and extensions for the [Zeroclaw](https://github.com/theonlyhennygod/zeroclaw) AI agent framework.

## Contents

- `zeroclaw-evaluation.md` — Evaluation report of the Zeroclaw project
- `nvidia-nim-compatibility.md` — NVIDIA NIM compatibility analysis
- `config.toml` — Example configuration file (copy to `~/.zeroclaw/config.toml`)
- `zeroclaw-patches/` — Patches for Zeroclaw
  - `tool-use-wiring.patch` — Multi-turn tool-use support for all providers
  - `markdown-memory-integration.patch` — Agent loop integration for markdown memory
  - `session-persistence.patch` — JSONL session persistence
  - `telegram-bot-integration.patch` — Telegram bot channel (long-polling)

## Markdown Memory System

An OpenClaw-style file-based memory system that supplements (or replaces) Zeroclaw's SQLite memory backend. All memory is stored in human-readable, version-controllable markdown files.

### File Layout

```
workspace/
├── SOUL.md              — Agent personality & behavior rules
├── USER.md              — User preferences & project context
├── MEMORY.md            — Curated long-term knowledge
└── memory/
    ├── 2026-02-13.md    — Yesterday's daily note
    └── 2026-02-14.md    — Today's daily note
```

### Files

| File | Purpose | Loaded at session start | Agent can write |
|------|---------|------------------------|-----------------|
| `SOUL.md` | Defines agent identity and behavior | Yes | No |
| `USER.md` | User preferences and project context | Yes | No |
| `MEMORY.md` | Curated long-term knowledge | Yes | Yes (via `memory_promote`) |
| `memory/YYYY-MM-DD.md` | Daily session logs | Today + yesterday | Yes (append-only) |

### Tools

| Tool | Description |
|------|-------------|
| `memory_store` | Append a timestamped note to today's daily log |
| `memory_recall` | Search across all markdown files (case-insensitive substring) |
| `memory_flush` | Write a session summary before exit (accomplished/pending/context) |
| `memory_promote` | Move important information from daily notes to MEMORY.md |

### Session Lifecycle

1. **Start**: Load `SOUL.md` + `USER.md` + `MEMORY.md` + yesterday's note + today's note → inject into system prompt
2. **During**: Agent uses `memory_store` to record decisions, discoveries, context to daily note
3. **End**: Agent calls `memory_flush` to write session summary
4. **Curation**: Agent (or user) calls `memory_promote` to move important facts to `MEMORY.md`

### Rust Source

The implementation lives in `src/memory/`:

- `markdown.rs` — Core `MarkdownMemory` struct (implements the `Memory` trait)
- `markdown_tools.rs` — Tool wrappers (`MemoryStoreTool`, `MemoryRecallTool`, `MemoryFlushTool`, `MemoryPromoteTool`)
- `mod.rs` — Module declarations

### Integration

Apply `zeroclaw-patches/markdown-memory-integration.patch` to wire the markdown memory into Zeroclaw's agent loop. The patch:

1. Adds `chrono` dependency to `Cargo.toml`
2. Registers `markdown` and `markdown_tools` modules
3. Initializes `MarkdownMemory` at session start
4. Loads markdown context into the system prompt
5. Registers all four markdown memory tools alongside existing tools

The SQLite memory backend remains available — the markdown tools are additive. Both systems can coexist.

## Telegram Bot Channel

A Telegram bot that uses HTTP long-polling (not webhooks) to receive messages and route them through the same agent pipeline as the CLI. No public IP or TLS certificate needed.

### Setup

1. Message [@BotFather](https://t.me/BotFather) on Telegram and create a bot
2. Copy the bot token
3. Configure via one of:
   - `config.toml`: set `[telegram] bot_token = "..."` and `enabled = true`
   - Environment variable: `export TELEGRAM_BOT_TOKEN="..."`
   - CLI flag: `zeroclaw run --telegram`

### Configuration

```toml
[telegram]
bot_token = "123456789:AAF-your-bot-token-here"
enabled = true
poll_interval_secs = 5         # How often to poll (default: 5s)
long_poll_timeout_secs = 30    # Telegram long-poll timeout (default: 30s)
```

### Features

- **Long-polling**: Polls `getUpdates` every N seconds — low CPU, no webhooks
- **Shared agent**: Uses the same agent loop, tools, and memory as the CLI
- **Session persistence**: Telegram turns are written to the same JSONL session files
- **Daily notes**: All Telegram interactions appear in today's daily note
- **Simultaneous operation**: Telegram listener and CLI run concurrently (shared agent, shared memory, separate message streams)
- **Message splitting**: Long responses are automatically split at Telegram's 4096-char limit

### Rust Source

- `src/memory/telegram.rs` — `TelegramListener`, `TelegramApi`, `TelegramConfig`, Telegram API types

### Integration

Apply `zeroclaw-patches/telegram-bot-integration.patch` to wire the Telegram listener into Zeroclaw's agent loop. The patch:

1. Adds the `--telegram` CLI flag
2. Adds `[telegram]` config section parsing
3. Starts the polling listener as a background tokio task
4. Routes incoming Telegram messages through the same `agent_loop()`
5. Sends responses back via `sendMessage`

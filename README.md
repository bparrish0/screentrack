# ScreenTrack

Continuous screen capture and AI-powered activity summarization for macOS.

ScreenTrack runs as a menubar app (or CLI daemon) that captures what's on your screen using macOS Accessibility APIs and OCR, then builds a tiered hierarchy of LLM-powered summaries — from per-minute micro-summaries up to weekly rollups. Query your activity history in natural language.

## Features

- **Accessibility-first capture** — extracts structured text from application windows via the macOS Accessibility API with zero visual overhead. Falls back to OCR (Apple Vision or LLM-based) for apps that render as graphics (remote desktop, terminals with custom renderers, etc.)
- **Tiered summarization** — micro (1-min) → hourly → daily → weekly summaries, each tier building on the previous. Preserves specific details like names, URLs, error messages, and time breakdowns
- **Smart Query** — agentic LLM queries that autonomously decide what data to fetch from your history across multiple rounds of reasoning
- **Keystroke and click tracking** — captures typing speed, typed text, and click targets with accessibility hit-testing for richer activity context
- **Native macOS menubar app** — live capture log, settings UI, and summary viewer built with AppKit (no Electron, no web views)
- **Multi-machine support** — run in hub/client mode to aggregate captures from multiple Macs over HTTP
- **Diff capture** — for chat apps (Messages, Teams, Slack, Discord) and terminals, only stores new/changed content instead of re-capturing entire conversation history
- **Adaptive OCR** — automatically promotes apps to OCR when accessibility text becomes stale or repetitive
- **Auto-update** — detects when the binary is replaced and relaunches seamlessly
- **Privacy-first** — all data stays local in a SQLite database; the only network calls go to your own LLM server
- **Export** — smart query results can be exported to Markdown, HTML, or PDF

## Requirements

- **macOS 13.0+** (Ventura or later)
- **Rust toolchain** — install via [rustup](https://rustup.rs)
- **An OpenAI-compatible LLM server** — for example [llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.com), [vLLM](https://github.com/vllm-project/vllm), or any cloud API (OpenAI, Anthropic via proxy, etc.)

## Permissions

ScreenTrack requires three macOS permissions (grant in System Settings → Privacy & Security):

| Permission | Purpose |
|---|---|
| **Accessibility** | Read text content from application windows |
| **Screen Recording** | OCR fallback for apps without accessible text |
| **Input Monitoring** | Keystroke and click event capture |

## Build & Install

```bash
# Build the release binary
cargo build --release

# Option A: Create macOS app bundle and install
./bundle.sh
cp -r ScreenTrack.app /Applications/
open /Applications/ScreenTrack.app

# Option B: Install from the built binary (builds app bundle in-place)
./target/release/screentrack install

# Option C: Run as a CLI daemon (no menubar UI)
./target/release/screentrack start
```

### Code Signing

By default, `screentrack install` uses ad-hoc signing (`-`), which works without an Apple Developer certificate. If you have a Developer ID, set it via environment variable:

```bash
export SCREENTRACK_SIGN_IDENTITY="Developer ID Application: Your Name (TEAM_ID)"
./target/release/screentrack install
```

## Configuration

On first launch, configure your LLM connection in the menubar settings panel. Settings are saved to `~/.screentrack/gui_config.json`.

You can also pass settings as CLI flags:

| Setting | CLI Flag | Default |
|---|---|---|
| LLM URL | `--llm-url` | `http://localhost:8080` |
| Model | `--model` | *(empty — must be configured)* |
| API Key | `--llm-api-key` | *(none)* |
| Database | `--db` | `~/.screentrack/db.sqlite` |

### Environment Variables

| Variable | Description |
|---|---|
| `SCREENTRACK_SIGN_IDENTITY` | Code signing identity for the `install` command (default: `-` for ad-hoc) |
| `RUST_LOG` | Log level filter (e.g. `info`, `debug`, `screentrack=debug`) |

## Usage

### Capture Modes

```bash
# Standalone: capture + summarize locally
screentrack start

# Hub: capture locally + receive from other machines + summarize
screentrack serve --listen 0.0.0.0:7878

# Client: capture locally + push frames to a hub
screentrack push --server http://hub:7878

# GUI: launch as menubar app (default when no subcommand given)
screentrack
```

### Querying

```bash
# Natural language query
screentrack query "what was I working on this morning?"

# Filter by app, tab, or machine
screentrack query "last 2 hours" --app Firefox --detail frames

# Smart query (agentic — LLM decides what data to fetch)
screentrack smart-query "summarize my day"

# Follow-up on previous smart query
screentrack smart-query -f "what about the meeting?"

# Export to file
screentrack smart-query "weekly report" -o report.md
screentrack smart-query "weekly report" -o report.pdf
```

### Other Commands

```bash
screentrack status                       # Capture statistics
screentrack list apps                    # Apps captured today
screentrack list tabs                    # Browser tabs today
screentrack list time                    # Time per app
screentrack list machines                # Known machines
screentrack summarize all                # Run all summarization tiers
screentrack profile                      # View extracted user profile
screentrack llm-test --sizes 4000,8000   # Test LLM context limits
```

## Architecture

```
screentrack/
├── crates/
│   ├── capture/       # Screen capture engine
│   │                  #   Accessibility API text extraction
│   │                  #   Apple Vision + LLM OCR backends
│   │                  #   CGEvent tap (keystrokes, clicks, scrolls)
│   │                  #   Frame deduplication and diff capture
│   │
│   ├── store/         # SQLite database layer
│   │                  #   Schema, migrations, FTS5 full-text search
│   │                  #   Frames, summaries, typing bursts, clicks
│   │
│   ├── summarizer/    # LLM integration
│   │                  #   OpenAI-compatible API client
│   │                  #   Tiered summarization prompts
│   │                  #   Token usage tracking
│   │
│   └── daemon/        # Application layer
│                      #   CLI (clap) + native menubar GUI (AppKit)
│                      #   Smart query engine, HTTP server
│                      #   Scheduler, auto-update, benchmarks
│
├── bundle.sh          # Build macOS .app bundle
├── Cargo.toml         # Workspace root
└── Cargo.lock
```

## Data Storage

All data is stored locally under `~/.screentrack/`:

| File | Purpose |
|---|---|
| `db.sqlite` | Main database (frames, summaries, typing, clicks) |
| `gui_config.json` | Menubar app settings |
| `smartquery_state.json` | Conversation state for follow-up queries |
| `screenshots/` | OCR screenshots (optional, with `--save-screenshots`) |

## License

[MIT](LICENSE)

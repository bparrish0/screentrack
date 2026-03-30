# ScreenTrack

## Build
```
cargo build --release
./target/release/screentrack install
```

## Architecture
Workspace with 4 crates: `capture`, `store`, `summarizer`, `daemon`.
The `daemon` crate is the main binary (`screentrack`) with CLI + native macOS menubar GUI.

## LLM Integration
Uses any OpenAI-compatible API. Configure via `--llm-url` and `--model` CLI flags
or via `~/.screentrack/gui_config.json` (SavedSettings loaded at startup).

## Version bumping
Bump the version in `crates/daemon/Cargo.toml` on every compile. Run `screentrack install` after every compile.

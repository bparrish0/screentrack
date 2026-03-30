use anyhow::Result;
use chrono::{Local, TimeZone, Utc};
use colored::Colorize;
use screentrack_store::{Database, FrameFilter};
use screentrack_summarizer::client::{ChatMessage, LlmClient};
use screentrack_summarizer::prompts;
use serde::Deserialize;
use std::path::PathBuf;

use crate::query::infer_time_range;

/// Options for the smart query command.
pub struct SmartQueryOpts {
    pub question: String,
    pub follow_up: bool,
    pub raw: bool,
    pub max_rounds: usize,
}

/// Saved conversation state for follow-up queries.
#[derive(serde::Serialize, serde::Deserialize)]
struct ConversationState {
    messages: Vec<ChatMessage>,
    #[serde(default)]
    start: i64,
    #[serde(default)]
    end: i64,
}

fn state_path() -> PathBuf {
    let dir = dirs_next::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".screentrack");
    dir.join("smartquery_state.json")
}

fn save_state(state: &ConversationState) -> Result<()> {
    let json = serde_json::to_string(state)?;
    std::fs::write(state_path(), json)?;
    Ok(())
}

fn load_state() -> Result<ConversationState> {
    let json = std::fs::read_to_string(state_path())?;
    let state: ConversationState = serde_json::from_str(&json)?;
    Ok(state)
}

/// Render Markdown text to the terminal with ANSI formatting.
fn print_markdown(text: &str) {
    let rendered = crate::md_render::render(text);
    print!("{rendered}");
}

/// Rough token estimate: ~4 chars per token for English text.
fn estimate_tokens(text: &str) -> usize {
    text.len() / 4
}

/// Estimate total tokens across all messages.
fn estimate_conversation_tokens(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .map(|m| estimate_tokens(&m.content) + 4)
        .sum() // +4 per message overhead
}

/// Max context budget (~149k actual tokens verified via llmtest).
/// Leave room for the LLM's response.
const CONTEXT_BUDGET: usize = 134_000;
const RESPONSE_RESERVE: usize = 8_000;
const MAX_DATA_TOKENS_PER_FETCH: usize = 50_000;

/// Truncate text to fit within a token budget, adding a note about truncation.
fn truncate_to_budget(text: &str, max_tokens: usize) -> String {
    let max_chars = max_tokens * 4;
    if text.len() <= max_chars {
        return text.to_string();
    }
    // Find a char boundary
    let mut end = max_chars;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}...\n\n(truncated — {} more characters not shown. Use a narrower search or smaller limit to see specific data.)",
        &text[..end],
        text.len() - end,
    )
}

fn build_system_prompt() -> String {
    let local_now = Utc::now().with_timezone(&Local);
    let tz_abbrev = local_now.format("%Z").to_string();
    let current_time = local_now.format("%A, %B %-d, %Y at %I:%M %p").to_string();
    let yesterday_str = (local_now - chrono::Duration::days(1))
        .format("%A, %B %-d, %Y")
        .to_string();

    format!(
        "/no_think\n\
You are an intelligent assistant answering questions about the user's computer activity. \
Right now it is {now} {tz}. Yesterday was {yesterday}. \
All timestamps are in the user's local timezone ({tz}).\n\n\
You have access to a database of screen captures organized in tiers:\n\
- **frames**: Raw screen text captures (most detailed, includes OCR text, app names, window titles)\n\
- **micro**: 1-minute summaries of activity\n\
- **hourly**: Hourly rollup summaries\n\
- **daily**: Daily rollup summaries\n\
- **weekly**: Weekly rollup summaries\n\
- **typing_bursts**: Keystroke logs with typed text and WPM\n\
- **click_events**: UI element clicks (buttons, links, tabs)\n\n\
You will be given a data catalog showing what data exists and how far back it goes. \
YOU decide what time range and data to fetch. Respond with a JSON command. You can make multiple \
requests to drill down into the data before answering.\n\n\
EVERY command MUST include a \"time_range\" field to specify what period to query. Examples:\n\
- \"time_range\": \"last 2 hours\"\n\
- \"time_range\": \"today\"\n\
- \"time_range\": \"yesterday\"\n\
- \"time_range\": \"last week\"\n\
- \"time_range\": \"2026-03-25\"\n\
- \"time_range\": \"last 30 minutes\"\n\
- \"time_range\": \"this morning\"\n\n\
IMPORTANT — TIMESTAMPS:\n\
- All times shown use 12-hour format with AM/PM (e.g. \"2:30 PM\", \"11:45 AM\").\n\
- Gaps between consecutive summaries indicate periods of inactivity (screen locked, sleeping, away).\n\
- When asked about sleep/wake times, look for the longest gap — sleep starts after the LAST summary \
before the gap, wake time is the FIRST summary after the gap.\n\n\
IMPORTANT — CONTEXT LIMITS:\n\
- You have a large context window (~130k tokens). Each fetch consumes part of it.\n\
- The data includes a \"tokens_remaining\" field — plan your fetches to stay within budget.\n\
- Summaries are compact (~50-200 tokens each); frames are larger (~200-500 tokens each).\n\
- You can fetch more data than needed — don't be overly conservative.\n\
- Frame fetches can use limits up to 5000. Use larger limits (500+) for thorough coverage.\n\
- Don't fetch the same tier/search/time_range combination twice.\n\n\
Available commands (respond with ONLY ONE JSON command per message, no markdown fences, no explanation):\n\
{{\"action\": \"fetch_summaries\", \"tier\": \"hourly\", \"time_range\": \"today\"}} — fetch summaries of a tier\n\
{{\"action\": \"search_summaries\", \"search\": \"Angie\", \"time_range\": \"last week\"}} — search summaries by keyword\n\
{{\"action\": \"fetch_app_time\", \"time_range\": \"today\"}} — time spent per application\n\
{{\"action\": \"fetch_typing_speed\", \"time_range\": \"last 2 hours\"}} — keystroke data with full typed text and WPM\n\
{{\"action\": \"search_typing\", \"search\": \"password\", \"time_range\": \"today\"}} — search keystroke history\n\
{{\"action\": \"fetch_clicks\", \"time_range\": \"last hour\"}} — what UI elements were clicked\n\
{{\"action\": \"search_clicks\", \"search\": \"Bookmarks\", \"time_range\": \"today\"}} — search click events\n\
{{\"action\": \"fetch_frames\", \"limit\": 500, \"time_range\": \"last 30 minutes\"}} — raw screen captures\n\
{{\"action\": \"fetch_frames\", \"app\": \"Firefox\", \"limit\": 500, \"time_range\": \"today\"}} — frames filtered by app\n\
{{\"action\": \"fetch_frames\", \"search\": \"docker\", \"limit\": 500, \"time_range\": \"today\"}} — frames by text search\n\
{{\"action\": \"answer\", \"text\": \"Your answer here in markdown\"}} — provide the final answer\n\n\
STRATEGY:\n\
1. **Choose the right time range** based on the question. \"What did I do today?\" → today. \
\"Who did I email last week?\" → last week. \"What was I just doing?\" → last 30 minutes.\n\
2. **Start with summaries.** They're cheap and tell you WHAT happened and WHEN. \
For a full day, use hourly. For multiple days, use daily. For under an hour, use micro.\n\
3. **Use search_summaries for specific topics.** If asking about a person/project/event, \
search across all tiers rather than fetching everything.\n\
4. **Fetch frames only for exact details** (email text, error messages, URLs).\n\
5. **Use fetch_typing_speed to see what the user actually typed** — helpful for Messages, \
passwords, commands, etc.\n\
6. **Answer as soon as you have enough.** Don't over-fetch.",
        tz = tz_abbrev,
        now = current_time,
        yesterday = yesterday_str,
    )
}

#[derive(Debug, Deserialize)]
struct LlmCommand {
    action: String,
    tier: Option<String>,
    app: Option<String>,
    search: Option<String>,
    limit: Option<i64>,
    text: Option<String>,
    /// Optional date/time override, e.g. "yesterday", "2026-03-27", "last week"
    /// Lets the LLM request data outside the original time range.
    time_range: Option<String>,
}

/// Format typing speed data for the LLM.
fn format_typing_speed(
    stats: &[(Option<String>, i64, i64, f64)],
    bursts: &[(i64, i64, i64, i64, f64, Option<String>, Option<String>)],
) -> String {
    let mut out = String::from("**Typing Speed Data:**\n\n");

    // Overall stats
    let total_chars: i64 = stats.iter().map(|s| s.1).sum();
    let total_ms: i64 = stats.iter().map(|s| s.2).sum();
    let overall_wpm = if total_ms > 0 {
        (total_chars as f64 / 5.0) / (total_ms as f64 / 60_000.0)
    } else {
        0.0
    };
    out.push_str(&format!(
        "Overall: {total_chars} chars typed, {:.0} WPM average, {} typing bursts\n\n",
        overall_wpm,
        bursts.len(),
    ));

    // Per-app breakdown
    out.push_str("**Per-app breakdown:**\n");
    for (app, chars, ms, avg_wpm) in stats {
        let app_name = app.as_deref().unwrap_or("Unknown");
        let minutes = *ms as f64 / 60_000.0;
        out.push_str(&format!(
            "- {app_name}: {chars} chars, {:.1} min typing, {avg_wpm:.0} WPM avg\n",
            minutes,
        ));
    }

    // Individual bursts timeline with typed text
    if !bursts.is_empty() {
        out.push_str(&format!("\n**Typing bursts ({}):**\n", bursts.len()));
        for (start, _end, chars, dur_ms, wpm, app, typed) in bursts {
            let ts = chrono::Utc
                .timestamp_millis_opt(*start)
                .unwrap()
                .with_timezone(&chrono::Local);
            let app_name = app.as_deref().unwrap_or("?");
            let text_preview = match typed {
                Some(t) if !t.is_empty() => format!(" \"{}\"", t),
                _ => String::new(),
            };
            out.push_str(&format!(
                "- {} [{}]: {} chars in {}ms ({:.0} WPM){}\n",
                ts.format("%I:%M:%S %p"),
                app_name,
                chars,
                dur_ms,
                wpm,
                text_preview,
            ));
        }
    }

    out
}

/// Export the answer to a file. Format is determined by file extension:
/// - `.md` — beautifully formatted Markdown
/// - `.html` — styled HTML document
/// - `.pdf` — PDF via macOS native conversion
pub fn export_answer(question: &str, answer: &str, path: &str) -> Result<()> {
    let path = std::path::Path::new(path);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("md")
        .to_lowercase();

    let now = Local::now();
    let timestamp = now.format("%Y-%m-%d %I:%M %p").to_string();

    match ext.as_str() {
        "md" => {
            let content = format_markdown_export(question, answer, &timestamp);
            std::fs::write(path, &content)?;
            eprintln!(
                "{} {}",
                "✓ Saved:".green().bold(),
                path.display(),
            );
        }
        "html" => {
            let html = format_html_export(question, answer, &timestamp);
            std::fs::write(path, &html)?;
            eprintln!(
                "{} {}",
                "✓ Saved:".green().bold(),
                path.display(),
            );
        }
        "pdf" => {
            let html = format_html_export(question, answer, &timestamp);
            let html_tmp = path.with_extension("_tmp.html");
            // Use absolute path for Chrome's file:// URL
            let html_abs = std::fs::canonicalize(
                &std::path::Path::new("."),
            )?
            .join(html_tmp.file_name().unwrap());
            std::fs::write(&html_abs, &html)?;
            let file_url = format!("file://{}", html_abs.display());
            let pdf_abs = std::fs::canonicalize(std::path::Path::new("."))?
                .join(path.file_name().unwrap_or(path.as_os_str()));

            // Try converters in order of preference
            let converters: Vec<(&str, Vec<String>)> = vec![
                // Google Chrome headless
                (
                    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                    vec![
                        "--headless".into(),
                        "--disable-gpu".into(),
                        "--no-sandbox".into(),
                        "--no-pdf-header-footer".into(),
                        "--run-all-compositor-stages-before-draw".into(),
                        format!("--print-to-pdf={}", pdf_abs.display()),
                        file_url.clone(),
                    ],
                ),
                // wkhtmltopdf
                (
                    "wkhtmltopdf",
                    vec![
                        "--quiet".into(),
                        "--enable-local-file-access".into(),
                        html_abs.display().to_string(),
                        pdf_abs.display().to_string(),
                    ],
                ),
            ];

            let mut converted = false;
            for (cmd, args) in &converters {
                let result = std::process::Command::new(cmd)
                    .args(args)
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status();
                if let Ok(status) = result {
                    if status.success() {
                        converted = true;
                        break;
                    }
                }
            }

            // Clean up temp HTML
            let _ = std::fs::remove_file(&html_abs);

            if converted {
                // Move to requested path if different from pdf_abs
                if pdf_abs != path {
                    let _ = std::fs::rename(&pdf_abs, path);
                }
                eprintln!(
                    "{} {}",
                    "✓ Saved:".green().bold(),
                    path.display(),
                );
            } else {
                let html_path = path.with_extension("html");
                let html_content = format_html_export(question, answer, &timestamp);
                std::fs::write(&html_path, &html_content)?;
                eprintln!(
                    "{} No PDF converter found (tried Chrome, wkhtmltopdf)",
                    "⚠".yellow(),
                );
                eprintln!(
                    "{} {}",
                    "✓ Saved as HTML instead:".green().bold(),
                    html_path.display(),
                );
            }
        }
        other => {
            anyhow::bail!(
                "Unsupported format '.{other}'. Use .md, .html, or .pdf"
            );
        }
    }

    Ok(())
}

/// Format a beautifully styled Markdown export.
fn format_markdown_export(question: &str, answer: &str, timestamp: &str) -> String {
    format!(
        r#"---
title: ScreenTrack Query
date: {timestamp}
---

# ScreenTrack Query

> **Q:** {question}

---

{answer}

---

*Generated by ScreenTrack on {timestamp}*
"#,
    )
}

/// Format a beautifully styled HTML document.
fn format_html_export(question: &str, answer: &str, timestamp: &str) -> String {
    // Convert markdown answer to HTML
    let parser = pulldown_cmark::Parser::new(answer);
    let mut answer_html = String::new();
    pulldown_cmark::html::push_html(&mut answer_html, parser);

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ScreenTrack — {question_esc}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {{
    --bg: #ffffff;
    --fg: #1a1a2e;
    --muted: #6b7280;
    --accent: #3b82f6;
    --accent-light: #dbeafe;
    --border: #e5e7eb;
    --code-bg: #f3f4f6;
    --blockquote-bg: #f0f9ff;
    --blockquote-border: #3b82f6;
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: var(--fg);
    background: var(--bg);
    line-height: 1.7;
    max-width: 800px;
    margin: 0 auto;
    padding: 48px 32px;
  }}

  .header {{
    border-bottom: 2px solid var(--accent);
    padding-bottom: 24px;
    margin-bottom: 32px;
  }}

  .header h1 {{
    font-size: 14px;
    font-weight: 600;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
  }}

  .question {{
    background: var(--blockquote-bg);
    border-left: 4px solid var(--blockquote-border);
    padding: 16px 20px;
    border-radius: 0 8px 8px 0;
    margin-bottom: 32px;
    font-size: 18px;
    font-weight: 500;
    color: var(--fg);
  }}

  .answer h1 {{ font-size: 24px; font-weight: 700; margin: 32px 0 16px; }}
  .answer h2 {{ font-size: 20px; font-weight: 600; margin: 28px 0 12px; color: var(--fg); }}
  .answer h3 {{ font-size: 16px; font-weight: 600; margin: 24px 0 8px; }}

  .answer p {{ margin: 12px 0; }}

  .answer ul, .answer ol {{
    margin: 12px 0;
    padding-left: 24px;
  }}

  .answer li {{ margin: 6px 0; }}

  .answer blockquote {{
    background: var(--blockquote-bg);
    border-left: 4px solid var(--blockquote-border);
    padding: 12px 16px;
    margin: 16px 0;
    border-radius: 0 6px 6px 0;
  }}

  .answer code {{
    font-family: 'JetBrains Mono', monospace;
    background: var(--code-bg);
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.9em;
  }}

  .answer pre {{
    background: var(--code-bg);
    padding: 16px;
    border-radius: 8px;
    overflow-x: auto;
    margin: 16px 0;
    border: 1px solid var(--border);
  }}

  .answer pre code {{
    background: none;
    padding: 0;
  }}

  .answer table {{
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
  }}

  .answer th, .answer td {{
    border: 1px solid var(--border);
    padding: 10px 14px;
    text-align: left;
  }}

  .answer th {{
    background: var(--code-bg);
    font-weight: 600;
  }}

  .answer strong {{ font-weight: 600; }}

  .answer hr {{
    border: none;
    border-top: 1px solid var(--border);
    margin: 24px 0;
  }}

  .footer {{
    margin-top: 48px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
    font-size: 12px;
    color: var(--muted);
    text-align: center;
  }}

  @media print {{
    body {{ padding: 24px; }}
    .header {{ page-break-after: avoid; }}
  }}
</style>
</head>
<body>
  <div class="header">
    <h1>ScreenTrack Query</h1>
  </div>

  <div class="question">{question_esc}</div>

  <div class="answer">
    {answer_html}
  </div>

  <div class="footer">
    Generated by ScreenTrack &middot; {timestamp}
  </div>
</body>
</html>"#,
        question_esc = html_escape(question),
        answer_html = answer_html,
        timestamp = timestamp,
    )
}

/// Simple HTML entity escaping.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// If the text looks like a raw `{"action": "answer", "text": "..."}` JSON
/// wrapper, extract just the text field. Handles cases where the LLM returns
/// the answer JSON as the content instead of plain markdown.
fn unwrap_answer_json(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.starts_with('{') && trimmed.contains("\"action\"") {
        if let Ok(cmd) = serde_json::from_str::<LlmCommand>(trimmed) {
            if cmd.action == "answer" {
                if let Some(inner) = cmd.text {
                    return inner;
                }
            }
        }
    }
    text.to_string()
}

/// Resolve time range from a command's `time_range` field.
/// If not specified, defaults to the last 24 hours.
fn resolve_time_range(cmd: &LlmCommand) -> (i64, i64) {
    if let Some(ref tr) = cmd.time_range {
        infer_time_range(tr)
    } else {
        // Default: last 24 hours
        let now = Utc::now();
        let start = now - chrono::Duration::days(1);
        (start.timestamp_millis(), now.timestamp_millis())
    }
}

/// Format a resolved time range as a human-readable string for LLM feedback.
fn describe_time_range(start: i64, end: i64) -> String {
    let s = Utc.timestamp_millis_opt(start).unwrap().with_timezone(&Local);
    let e = Utc.timestamp_millis_opt(end).unwrap().with_timezone(&Local);
    if s.date_naive() == e.date_naive() {
        format!(
            "{} {} → {}",
            s.format("%a %b %-d"),
            s.format("%I:%M %p"),
            e.format("%I:%M %p"),
        )
    } else {
        format!(
            "{} → {}",
            s.format("%a %b %-d %I:%M %p"),
            e.format("%a %b %-d %I:%M %p"),
        )
    }
}

/// Format data availability into a human-readable catalog for the LLM.
fn format_data_availability(availability: &[(String, i64, i64, i64)]) -> String {
    let mut out = String::from("**Available data in the database:**\n\n");
    out.push_str("| Data Type | Count | Earliest | Latest |\n");
    out.push_str("|-----------|-------|----------|--------|\n");
    for (tier, count, earliest, latest) in availability {
        if *count == 0 {
            continue;
        }
        let earliest_dt = Utc
            .timestamp_millis_opt(*earliest)
            .unwrap()
            .with_timezone(&Local);
        let latest_dt = Utc
            .timestamp_millis_opt(*latest)
            .unwrap()
            .with_timezone(&Local);
        out.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            tier,
            count,
            earliest_dt.format("%a %b %-d, %Y %I:%M %p"),
            latest_dt.format("%a %b %-d, %Y %I:%M %p"),
        ));
    }
    out
}

/// Handle a smart query with an agentic LLM loop.
pub async fn handle_smart_query(
    db: &Database,
    client: &LlmClient,
    opts: &SmartQueryOpts,
) -> Result<()> {
    handle_smart_query_inner(db, db, client, opts).await
}

/// Handle a smart query with separate DBs for summaries and frames.
/// Used by benchmark mode: summaries from benchmark DB, frames from production DB.
pub async fn handle_smart_query_hybrid(
    summary_db: &Database,
    frame_db: &Database,
    client: &LlmClient,
    opts: &SmartQueryOpts,
) -> Result<()> {
    handle_smart_query_inner(summary_db, frame_db, client, opts).await
}

/// Inner implementation that prints the answer to stdout.
async fn handle_smart_query_inner(
    summary_db: &Database,
    frame_db: &Database,
    client: &LlmClient,
    opts: &SmartQueryOpts,
) -> Result<()> {
    let answer = smart_query_get_answer(summary_db, frame_db, client, opts).await?;
    if opts.raw {
        println!("{answer}");
    } else {
        print_markdown(&answer);
    }
    Ok(())
}

/// Run the agentic smart query loop and return the final answer as a markdown string.
/// This is the core logic used by both CLI and GUI.
pub async fn smart_query_get_answer(
    summary_db: &Database,
    frame_db: &Database,
    client: &LlmClient,
    opts: &SmartQueryOpts,
) -> Result<String> {
    let mut messages: Vec<ChatMessage>;

    if opts.follow_up {
        // Load previous conversation state
        let prev = load_state().map_err(|_| {
            anyhow::anyhow!("No previous smart-query conversation found. Run without -f first.")
        })?;
        messages = prev.messages;

        let used = estimate_conversation_tokens(&messages);
        eprintln!(
            "{} {} {} {}",
            "▸".dimmed(),
            "Follow-up:".dimmed(),
            "resuming previous conversation".bright_green().bold(),
            format!("(~{}k tokens of prior context)", used / 1000).bright_black(),
        );

        // Append the follow-up question
        messages.push(ChatMessage {
            role: "user".into(),
            content: format!(
                "**Follow-up question:** {}\n\n\
                 You still have access to the same data commands. Use the data already in this \
                 conversation if it answers the question — only fetch more if needed.",
                opts.question
            ),
        });
    } else {
        // Query data availability from the database
        let availability = summary_db.get_data_availability().await?;

        let catalog = format!(
            "{}\n**User question:** {}",
            format_data_availability(&availability),
            opts.question,
        );

        // Print availability info
        eprintln!(
            "{} {} {}",
            "▸".dimmed(),
            "Mode:".dimmed(),
            "smart query (agentic)".bright_green().bold(),
        );
        for (tier, count, _, _) in &availability {
            if *count > 0 {
                eprintln!(
                    "{}   {} {}",
                    "▸".dimmed(),
                    crate::query::tier_label(tier),
                    format!("({count})").dimmed(),
                );
            }
        }

        // Start the conversation
        messages = vec![
            ChatMessage {
                role: "system".into(),
                content: build_system_prompt(),
            },
            ChatMessage {
                role: "user".into(),
                content: catalog,
            },
        ];
    }

    for round in 0..opts.max_rounds {
        let used_tokens = estimate_conversation_tokens(&messages);
        let remaining = CONTEXT_BUDGET
            .saturating_sub(used_tokens)
            .saturating_sub(RESPONSE_RESERVE);

        eprintln!(
            "\n{} {} {} {}",
            "▸".dimmed(),
            format!("Round {}:", round + 1).dimmed(),
            "thinking...".bright_black(),
            format!(
                "(~{}k tokens used, ~{}k remaining)",
                used_tokens / 1000,
                remaining / 1000
            )
            .bright_black(),
        );

        // If we're almost out of budget, force the answer
        if remaining < 1000 {
            eprintln!(
                "{} {}",
                "▸".dimmed(),
                "Context nearly full, requesting final answer...".yellow(),
            );
            messages.push(ChatMessage {
                role: "user".into(),
                content: "Context window is nearly full. Please answer now with the data you have. Respond with: {\"action\": \"answer\", \"text\": \"your answer\"}".into(),
            });
        }

        let response = client.chat_messages(&messages).await?;

        // Try to parse as JSON command
        let response_trimmed = response.trim();
        let cmd: Option<LlmCommand> = parse_llm_command(response_trimmed);

        match cmd {
            Some(cmd) if cmd.action == "answer" => {
                let answer = cmd.text.unwrap_or_else(|| response.clone());
                // Safety: if the LLM wrapped the answer in JSON again, unwrap it
                let answer = unwrap_answer_json(&answer);
                tracing::info!(
                    "smart_query: answer ready after {} round(s), {} bytes",
                    round + 1,
                    answer.len()
                );
                eprintln!(
                    "{} {}\n",
                    "▸".dimmed(),
                    format!(
                        "Answer ready (after {} round{})",
                        round + 1,
                        if round == 0 { "" } else { "s" }
                    )
                    .green(),
                );
                // Save conversation for follow-ups
                messages.push(ChatMessage {
                    role: "assistant".into(),
                    content: response,
                });
                save_state(&ConversationState {
                    messages,
                    start: 0,
                    end: 0,
                })
                .ok();
                return Ok(answer);
            }
            Some(cmd) if cmd.action == "fetch_summaries" => {
                let tier = cmd.tier.as_deref().unwrap_or("micro");
                let (q_start, q_end) = resolve_time_range(&cmd);
                tracing::info!("smart_query: round {} fetching {} summaries", round + 1, tier);
                eprintln!(
                    "{} {} {}",
                    "▸".dimmed(),
                    "Fetching:".dimmed(),
                    format!("{tier} summaries").cyan(),
                );

                let summaries = summary_db.get_summaries(tier, q_start, q_end).await?;

                let mut data = String::new();
                if summaries.is_empty() {
                    data.push_str(&format!(
                        "No {tier} summaries found in range: {}. Try a different time_range.\n",
                        describe_time_range(q_start, q_end),
                    ));
                } else {
                    data.push_str(&format!("**{} {} summaries:**\n\n", summaries.len(), tier));
                    let mut prev_end: Option<i64> = None;
                    for s in &summaries {
                        // Show gap between summaries (indicates inactivity)
                        if let Some(pe) = prev_end {
                            let gap_mins = (s.start_time - pe) / 60_000;
                            if gap_mins >= 5 {
                                let gap_desc = if gap_mins >= 60 {
                                    format!("{} hr {} min", gap_mins / 60, gap_mins % 60)
                                } else {
                                    format!("{} min", gap_mins)
                                };
                                data.push_str(&format!(
                                    "⸻ **GAP: {} of inactivity** ⸻\n\n",
                                    gap_desc
                                ));
                            }
                        }
                        prev_end = Some(s.end_time);

                        let s_start = Utc
                            .timestamp_millis_opt(s.start_time)
                            .unwrap()
                            .with_timezone(&Local);
                        let s_end = Utc
                            .timestamp_millis_opt(s.end_time)
                            .unwrap()
                            .with_timezone(&Local);
                        data.push_str(&format!(
                            "**{} → {}**\n",
                            s_start.format("%a %b %-d, %Y %I:%M %p"),
                            s_end.format("%I:%M %p"),
                        ));
                        if let Some(ref apps) = s.apps_referenced {
                            data.push_str(&format!("Apps: {apps}\n"));
                        }
                        data.push_str(&s.summary);
                        data.push_str("\n\n---\n\n");
                    }
                }

                // Truncate to fit budget
                let budget = MAX_DATA_TOKENS_PER_FETCH.min(remaining.saturating_sub(500));
                data = truncate_to_budget(&data, budget);

                let data_tokens = estimate_tokens(&data);
                eprintln!(
                    "{} {} {} {} {} {}",
                    "▸".dimmed(),
                    "Provided:".dimmed(),
                    crate::query::tier_label(tier),
                    format!("({}", summaries.len()).dimmed(),
                    format!(
                        "{})",
                        if summaries.len() == 1 {
                            "summary"
                        } else {
                            "summaries"
                        }
                    )
                    .dimmed(),
                    format!("~{}k tokens", data_tokens / 1000).bright_black(),
                );

                messages.push(ChatMessage {
                    role: "assistant".into(),
                    content: response.clone(),
                });

                // Append budget info to help the LLM plan
                let new_remaining = remaining
                    .saturating_sub(data_tokens)
                    .saturating_sub(estimate_tokens(&response));
                data.push_str(&format!("\n\n(tokens_remaining: ~{})", new_remaining));

                messages.push(ChatMessage {
                    role: "user".into(),
                    content: data,
                });
            }
            Some(cmd) if cmd.action == "search_summaries" => {
                let search = cmd.search.as_deref().unwrap_or("");
                let tier = cmd.tier.as_deref();
                let tier_desc = tier.unwrap_or("all tiers");
                let (q_start, q_end) = resolve_time_range(&cmd);
                tracing::info!("smart_query: round {} searching '{}' in {}", round + 1, search, tier_desc);
                eprintln!(
                    "{} {} {}",
                    "▸".dimmed(),
                    "Searching:".dimmed(),
                    format!("summaries for \"{search}\" in {tier_desc}").cyan(),
                );

                let summaries = summary_db
                    .search_summaries(search, tier, q_start, q_end)
                    .await?;

                let mut data = String::new();
                if summaries.is_empty() {
                    data.push_str(&format!(
                        "No summaries matching \"{search}\" found in {tier_desc} for range: {}. Try a different time_range.\n",
                        describe_time_range(q_start, q_end)
                    ));
                } else {
                    data.push_str(&format!(
                        "**{} summaries matching \"{}\":**\n\n",
                        summaries.len(),
                        search
                    ));
                    for s in &summaries {
                        let s_start = Utc
                            .timestamp_millis_opt(s.start_time)
                            .unwrap()
                            .with_timezone(&Local);
                        let s_end = Utc
                            .timestamp_millis_opt(s.end_time)
                            .unwrap()
                            .with_timezone(&Local);
                        data.push_str(&format!(
                            "**[{}] {} → {}**\n",
                            s.tier,
                            s_start.format("%a %b %-d, %Y %I:%M %p"),
                            s_end.format("%I:%M %p"),
                        ));
                        if let Some(ref apps) = s.apps_referenced {
                            data.push_str(&format!("Apps: {apps}\n"));
                        }
                        data.push_str(&s.summary);
                        data.push_str("\n\n---\n\n");
                    }
                }

                // Truncate to fit budget
                let budget = MAX_DATA_TOKENS_PER_FETCH.min(remaining.saturating_sub(500));
                data = truncate_to_budget(&data, budget);

                let data_tokens = estimate_tokens(&data);
                eprintln!(
                    "{} {} {} {} {}",
                    "▸".dimmed(),
                    "Found:".dimmed(),
                    format!("{}", summaries.len()).bold(),
                    format!("matching summaries").dimmed(),
                    format!("~{}k tokens", data_tokens / 1000).bright_black(),
                );

                messages.push(ChatMessage {
                    role: "assistant".into(),
                    content: response.clone(),
                });

                let new_remaining = remaining
                    .saturating_sub(data_tokens)
                    .saturating_sub(estimate_tokens(&response));
                data.push_str(&format!("\n\n(tokens_remaining: ~{})", new_remaining));

                messages.push(ChatMessage {
                    role: "user".into(),
                    content: data,
                });
            }
            Some(cmd) if cmd.action == "fetch_app_time" => {
                let (q_start, q_end) = resolve_time_range(&cmd);
                tracing::info!("smart_query: round {} fetching app time", round + 1);
                eprintln!(
                    "{} {} {}",
                    "▸".dimmed(),
                    "Fetching:".dimmed(),
                    "app time breakdown".cyan(),
                );

                let app_times = frame_db.get_app_time(q_start, q_end).await?;
                let mut data = if app_times.is_empty() {
                    "No app time data available for this time range.\n".to_string()
                } else {
                    prompts::format_app_time(&app_times)
                };

                let data_tokens = estimate_tokens(&data);
                eprintln!(
                    "{} {} {} {} {}",
                    "▸".dimmed(),
                    "Provided:".dimmed(),
                    format!("{}", app_times.len()).bold(),
                    "apps with time data".dimmed(),
                    format!("~{} tokens", data_tokens).bright_black(),
                );

                messages.push(ChatMessage {
                    role: "assistant".into(),
                    content: response.clone(),
                });

                let new_remaining = remaining
                    .saturating_sub(data_tokens)
                    .saturating_sub(estimate_tokens(&response));
                data.push_str(&format!("\n\n(tokens_remaining: ~{})", new_remaining));

                messages.push(ChatMessage {
                    role: "user".into(),
                    content: data,
                });
            }
            Some(cmd) if cmd.action == "fetch_typing_speed" => {
                let (q_start, q_end) = resolve_time_range(&cmd);
                tracing::info!("smart_query: round {} fetching typing speed", round + 1);
                eprintln!(
                    "{} {} {}",
                    "▸".dimmed(),
                    "Fetching:".dimmed(),
                    "typing speed metrics".cyan(),
                );

                let stats = frame_db.get_typing_speed(q_start, q_end).await?;
                let bursts = frame_db.get_typing_bursts(q_start, q_end).await?;

                let mut data = if stats.is_empty() {
                    "No typing data available for this time range.\n".to_string()
                } else {
                    format_typing_speed(&stats, &bursts)
                };

                let data_tokens = estimate_tokens(&data);
                eprintln!(
                    "{} {} {} {} {}",
                    "▸".dimmed(),
                    "Provided:".dimmed(),
                    format!("{} bursts", bursts.len()).bold(),
                    "typing speed data".dimmed(),
                    format!("~{} tokens", data_tokens).bright_black(),
                );

                messages.push(ChatMessage {
                    role: "assistant".into(),
                    content: response.clone(),
                });

                let new_remaining = remaining
                    .saturating_sub(data_tokens)
                    .saturating_sub(estimate_tokens(&response));
                data.push_str(&format!("\n\n(tokens_remaining: ~{})", new_remaining));

                messages.push(ChatMessage {
                    role: "user".into(),
                    content: data,
                });
            }
            Some(cmd) if cmd.action == "search_typing" => {
                let search = cmd.search.as_deref().unwrap_or("");
                let (q_start, q_end) = resolve_time_range(&cmd);
                tracing::info!("smart_query: round {} searching typing for '{}'", round + 1, search);
                eprintln!(
                    "{} {} {}",
                    "▸".dimmed(),
                    "Searching:".dimmed(),
                    format!("typed text for \"{}\"", search).cyan(),
                );

                let results = frame_db.search_typing(search, q_start, q_end).await?;

                let mut data = if results.is_empty() {
                    format!("No typed text matching \"{}\" found in this time range.\n", search)
                } else {
                    let mut s = format!("**{} typing bursts matching \"{}\":**\n\n", results.len(), search);
                    for (start_t, _end_t, chars, dur_ms, wpm, app, typed) in &results {
                        let ts = chrono::Utc
                            .timestamp_millis_opt(*start_t)
                            .unwrap()
                            .with_timezone(&chrono::Local);
                        let app_name = app.as_deref().unwrap_or("?");
                        let text = typed.as_deref().unwrap_or("");
                        s.push_str(&format!(
                            "- {} [{}]: ({} chars, {}ms, {:.0} WPM) \"{}\"\n",
                            ts.format("%I:%M:%S %p"),
                            app_name,
                            chars,
                            dur_ms,
                            wpm,
                            text,
                        ));
                    }
                    s
                };

                let data_tokens = estimate_tokens(&data);
                eprintln!(
                    "{} {} {} {} {}",
                    "▸".dimmed(),
                    "Found:".dimmed(),
                    format!("{}", results.len()).bold(),
                    "matching typing bursts".dimmed(),
                    format!("~{}k tokens", data_tokens / 1000).bright_black(),
                );

                messages.push(ChatMessage {
                    role: "assistant".into(),
                    content: response.clone(),
                });

                let new_remaining = remaining
                    .saturating_sub(data_tokens)
                    .saturating_sub(estimate_tokens(&response));
                data.push_str(&format!("\n\n(tokens_remaining: ~{})", new_remaining));

                messages.push(ChatMessage {
                    role: "user".into(),
                    content: data,
                });
            }
            Some(cmd) if cmd.action == "fetch_clicks" || cmd.action == "search_clicks" => {
                let (q_start, q_end) = resolve_time_range(&cmd);
                let is_search = cmd.action == "search_clicks";
                let search = cmd.search.as_deref().unwrap_or("");

                if is_search {
                    tracing::info!("smart_query: round {} searching clicks for '{}'", round + 1, search);
                    eprintln!(
                        "{} {} {}",
                        "▸".dimmed(),
                        "Searching:".dimmed(),
                        format!("clicks for \"{}\"", search).cyan(),
                    );
                } else {
                    tracing::info!("smart_query: round {} fetching clicks", round + 1);
                    eprintln!(
                        "{} {} {}",
                        "▸".dimmed(),
                        "Fetching:".dimmed(),
                        "click events".cyan(),
                    );
                }

                let results = if is_search {
                    frame_db.search_clicks(search, q_start, q_end).await?
                } else {
                    frame_db.get_click_events(q_start, q_end).await?
                };

                let mut data = if results.is_empty() {
                    if is_search {
                        format!("No clicks matching \"{}\" found.\n", search)
                    } else {
                        "No click events found in this time range.\n".to_string()
                    }
                } else {
                    let mut s = format!("**{} click events:**\n\n", results.len());
                    for (ts, _x, _y, app, role, title, desc, _value) in &results {
                        let dt = chrono::Utc
                            .timestamp_millis_opt(*ts)
                            .unwrap()
                            .with_timezone(&chrono::Local);
                        let app_name = app.as_deref().unwrap_or("?");
                        let role_str = role.as_deref().unwrap_or("");
                        let label = title
                            .as_deref()
                            .or(desc.as_deref())
                            .unwrap_or("(no label)");
                        s.push_str(&format!(
                            "- {} [{}] {}: \"{}\"\n",
                            dt.format("%I:%M:%S %p"),
                            app_name,
                            role_str,
                            label,
                        ));
                    }
                    s
                };

                let data_tokens = estimate_tokens(&data);
                eprintln!(
                    "{} {} {} {} {}",
                    "▸".dimmed(),
                    "Found:".dimmed(),
                    format!("{}", results.len()).bold(),
                    "click events".dimmed(),
                    format!("~{}k tokens", data_tokens / 1000).bright_black(),
                );

                messages.push(ChatMessage {
                    role: "assistant".into(),
                    content: response.clone(),
                });

                let new_remaining = remaining
                    .saturating_sub(data_tokens)
                    .saturating_sub(estimate_tokens(&response));
                data.push_str(&format!("\n\n(tokens_remaining: ~{})", new_remaining));

                messages.push(ChatMessage {
                    role: "user".into(),
                    content: data,
                });
            }
            Some(cmd) if cmd.action == "fetch_frames" => {
                let (q_start, q_end) = resolve_time_range(&cmd);
                tracing::info!("smart_query: round {} fetching frames", round + 1);
                // Clamp limit to avoid blowing the budget
                let requested_limit = cmd.limit.unwrap_or(500);
                // Each frame is roughly 300-500 tokens; cap based on remaining budget
                let max_frames_by_budget = (remaining.saturating_sub(500) / 400).max(1) as i64;
                let limit = requested_limit.min(max_frames_by_budget).min(5000); // hard cap at 5000

                let filter_desc = if let Some(ref app) = cmd.app {
                    format!("frames (app: {app}, limit: {limit})")
                } else if let Some(ref search) = cmd.search {
                    format!("frames (search: \"{search}\", limit: {limit})")
                } else {
                    format!("frames (limit: {limit})")
                };
                eprintln!(
                    "{} {} {}",
                    "▸".dimmed(),
                    "Fetching:".dimmed(),
                    filter_desc.cyan(),
                );

                if requested_limit != limit {
                    eprintln!(
                        "{} {}",
                        "▸".dimmed(),
                        format!(
                            "(clamped from {requested_limit} to {limit} to fit context budget)"
                        )
                        .bright_black(),
                    );
                }

                let filter = FrameFilter {
                    start: Some(q_start),
                    end: Some(q_end),
                    app_name: cmd.app.clone(),
                    search_text: cmd.search.clone(),
                    limit: Some(limit),
                    ..Default::default()
                };

                let frames = frame_db.get_frames_filtered(&filter).await?;

                let mut data = String::new();
                if frames.is_empty() {
                    data.push_str(&format!(
                        "No frames found matching the filter for range: {}. Try a different time_range.\n",
                        describe_time_range(q_start, q_end),
                    ));
                } else {
                    data.push_str(&format!("**{} frames:**\n\n", frames.len()));
                    // Truncate individual frame text more aggressively for smartquery
                    let frame_data: Vec<prompts::FrameData> = frames
                        .iter()
                        .map(|f| {
                            let text = if f.text_content.len() > 800 {
                                let mut end = 800;
                                while end > 0 && !f.text_content.is_char_boundary(end) {
                                    end -= 1;
                                }
                                format!("{}… (truncated)", &f.text_content[..end])
                            } else {
                                f.text_content.clone()
                            };
                            prompts::FrameData {
                                machine: Some(f.machine_id.clone()),
                                app: f.app_name.clone().unwrap_or_else(|| "Unknown".into()),
                                window: f.window_title.clone(),
                                browser_tab: f.browser_tab.clone(),
                                text,
                                timestamp: Some(
                                    Utc.timestamp_millis_opt(f.timestamp)
                                        .unwrap()
                                        .with_timezone(&Local)
                                        .format("%I:%M:%S %p")
                                        .to_string(),
                                ),
                            }
                        })
                        .collect();
                    data.push_str(&prompts::format_frames_for_micro(&frame_data));
                }

                // Truncate total data to fit budget
                let budget = MAX_DATA_TOKENS_PER_FETCH.min(remaining.saturating_sub(500));
                data = truncate_to_budget(&data, budget);

                let data_tokens = estimate_tokens(&data);
                eprintln!(
                    "{} {} {} {} {} {}",
                    "▸".dimmed(),
                    "Provided:".dimmed(),
                    crate::query::tier_label("frames"),
                    format!("({}", frames.len()).dimmed(),
                    format!("{})", if frames.len() == 1 { "frame" } else { "frames" }).dimmed(),
                    format!("~{}k tokens", data_tokens / 1000).bright_black(),
                );

                messages.push(ChatMessage {
                    role: "assistant".into(),
                    content: response.clone(),
                });

                let new_remaining = remaining
                    .saturating_sub(data_tokens)
                    .saturating_sub(estimate_tokens(&response));
                data.push_str(&format!("\n\n(tokens_remaining: ~{})", new_remaining));

                messages.push(ChatMessage {
                    role: "user".into(),
                    content: data,
                });
            }
            _ => {
                // LLM didn't return valid JSON — treat the whole response as the answer
                tracing::info!(
                    "smart_query: round {} direct answer (no JSON), {} bytes",
                    round + 1,
                    response.len()
                );
                eprintln!(
                    "{} {}\n",
                    "▸".dimmed(),
                    "Direct answer (no data requests):".green(),
                );
                messages.push(ChatMessage {
                    role: "assistant".into(),
                    content: response.clone(),
                });
                save_state(&ConversationState {
                    messages,
                    start: 0,
                    end: 0,
                })
                .ok();
                return Ok(unwrap_answer_json(&response));
            }
        }
    }

    // Hit max rounds — ask LLM for final answer
    eprintln!(
        "{} {}",
        "▸".dimmed(),
        format!(
            "Reached max rounds ({}), requesting final answer...",
            opts.max_rounds
        )
        .yellow(),
    );
    messages.push(ChatMessage {
        role: "user".into(),
        content: "You've reached the maximum number of data requests. Please provide your best answer now using the data you've gathered. Respond with: {\"action\": \"answer\", \"text\": \"your answer\"}".into(),
    });

    let response = client.chat_messages(&messages).await?;
    let cmd = parse_llm_command(response.trim());

    let answer = match cmd {
        Some(cmd) if cmd.action == "answer" => {
            unwrap_answer_json(&cmd.text.unwrap_or(response.clone()))
        }
        _ => unwrap_answer_json(&response),
    };

    // Save conversation for follow-ups
    messages.push(ChatMessage {
        role: "assistant".into(),
        content: response,
    });
    save_state(&ConversationState {
        messages,
        start: 0,
        end: 0,
    })
    .ok();

    eprintln!("{} {}\n", "▸".dimmed(), "Answer:".green());
    Ok(answer)
}

/// Try to parse a JSON command from the LLM response.
/// Handles cases where the LLM wraps JSON in text or markdown fences,
/// or returns multiple JSON commands on separate lines (takes the first).
fn parse_llm_command(text: &str) -> Option<LlmCommand> {
    // Try direct parse first
    if let Ok(cmd) = serde_json::from_str::<LlmCommand>(text) {
        return Some(cmd);
    }

    // Try each line individually — LLM may return multiple commands on separate lines
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('{') && line.ends_with('}') {
            if let Ok(cmd) = serde_json::from_str::<LlmCommand>(line) {
                return Some(cmd);
            }
        }
    }

    // Try to find the first complete JSON object in the text
    if let Some(start) = text.find('{') {
        // Find the matching closing brace (not the last one — there may be multiple objects)
        let bytes = text.as_bytes();
        let mut depth = 0;
        for (i, &b) in bytes[start..].iter().enumerate() {
            match b {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        let end = start + i;
                        if let Ok(cmd) = serde_json::from_str::<LlmCommand>(&text[start..=end]) {
                            return Some(cmd);
                        }
                        break;
                    }
                }
                _ => {}
            }
        }
    }

    // Legacy fallback: first { to last }
    if let Some(start) = text.find('{') {
        if let Some(end) = text.rfind('}') {
            if let Ok(cmd) = serde_json::from_str::<LlmCommand>(&text[start..=end]) {
                return Some(cmd);
            }
        }
    }

    None
}

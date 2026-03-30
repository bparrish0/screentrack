//! Needle-in-a-haystack context length test for the local LLM.
//!
//! Embeds a unique secret phrase at the middle of increasingly large blocks of
//! real frame data, then asks the LLM to find it. Reports pass/fail at each
//! context size to help determine the effective context limit.

use anyhow::Result;
use chrono::{Local, TimeZone, Utc};
use colored::Colorize;
use screentrack_store::Database;
use screentrack_summarizer::client::LlmClient;
use std::time::Instant;

/// The secret phrase embedded in the haystack.
const NEEDLE: &str = "SECRET_PINEAPPLE_TSUNAMI_42";

/// Build a haystack of frame text at approximately `target_tokens` size,
/// with the needle embedded at the middle.
/// Returns (haystack_text, needle_position_percent).
fn build_haystack(
    frames: &[screentrack_store::Frame],
    target_tokens: usize,
) -> Option<(String, usize)> {
    // Screen capture data tokenizes at varying ratios. Use a conservative
    // ratio to avoid overshooting the target context size.
    let target_chars = target_tokens * 3;

    // Format frames into text blocks. Cycle through frames multiple times
    // if needed to reach the target size.
    let mut blocks: Vec<String> = Vec::new();
    let mut total_chars = 0;
    let needed = target_chars + 2000;

    for pass in 0..5 {
        for (i, f) in frames.iter().enumerate() {
            let ts = Utc
                .timestamp_millis_opt(f.timestamp)
                .unwrap()
                .with_timezone(&Local);
            let app = f.app_name.as_deref().unwrap_or("?");
            let window = f.window_title.as_deref().unwrap_or("");
            // On repeat passes, add a pass marker so content isn't identical
            let prefix = if pass > 0 {
                format!("[pass {}] ", pass + 1)
            } else {
                String::new()
            };
            let block = format!(
                "{}[{} | {} — {}]\n{}\n---\n",
                prefix,
                ts.format("%I:%M:%S %p"),
                app,
                window,
                &f.text_content,
            );
            total_chars += block.len();
            blocks.push(block);

            if total_chars >= needed {
                break;
            }
        }
        if total_chars >= needed {
            break;
        }
    }

    if total_chars < target_chars / 2 {
        return None; // not enough frames even after cycling
    }

    // Truncate to target size, inserting needle at the middle
    let mut haystack = String::with_capacity(target_chars + 200);
    let mut chars_so_far = 0;
    let midpoint = target_chars / 2;
    let mut needle_inserted = false;
    let needle_line = format!(
        "\n[INTERNAL NOTE] The secret verification code is: {NEEDLE}\n\n"
    );

    for block in &blocks {
        if !needle_inserted && chars_so_far + block.len() >= midpoint {
            // Insert needle at this point
            haystack.push_str(&needle_line);
            needle_inserted = true;
        }
        haystack.push_str(block);
        chars_so_far += block.len();
        if chars_so_far >= target_chars {
            break;
        }
    }

    if !needle_inserted {
        haystack.push_str(&needle_line);
    }

    let position_pct = if haystack.contains(NEEDLE) {
        let pos = haystack.find(NEEDLE).unwrap();
        (pos as f64 / haystack.len() as f64 * 100.0) as usize
    } else {
        50
    };

    Some((haystack, position_pct))
}

pub async fn run_context_length_test(
    db: &Database,
    client: &LlmClient,
    sizes: &[usize],
    start: i64,
    end: i64,
) -> Result<()> {
    println!(
        "\n{}\n",
        "━━━ LLM Context Length Test (Needle in a Haystack) ━━━".bold()
    );
    println!(
        "  {} {}",
        "Needle:".dimmed(),
        NEEDLE.yellow()
    );
    println!(
        "  {} {}",
        "LLM:".dimmed(),
        client.base_url().unwrap_or("?").cyan()
    );
    println!(
        "  {} {}\n",
        "Sizes:".dimmed(),
        sizes
            .iter()
            .map(|s| format!("{}k", s / 1000))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Fetch a large pool of frames to build haystacks from.
    // Need enough raw text to fill the largest target size at ~3 chars/token.
    let max_size = *sizes.iter().max().unwrap_or(&256_000);
    let max_frames = (max_size as i64 * 3 / 200).max(10000); // ~200 chars avg per frame
    let filter = screentrack_store::FrameFilter {
        start: Some(start),
        end: Some(end),
        limit: Some(max_frames),
        ..Default::default()
    };
    let frames = db.get_frames_filtered(&filter).await?;

    if frames.is_empty() {
        println!("{}", "No frames found in the time range. Try --range this-week".red());
        return Ok(());
    }

    println!(
        "  {} {} frames available as haystack material\n",
        "Pool:".dimmed(),
        frames.len()
    );

    let system = "/no_think\nYou are a verification assistant. The user will give you a large block of screen capture data. Hidden somewhere in the data is a secret verification code. Find and return ONLY the secret code, nothing else. The code looks like a phrase of random words and numbers (e.g., SECRET_SOMETHING_123). Return just the code with no other text.";

    // Override max_tokens for the test — the answer is < 50 tokens, so cap
    // generation to prevent runaway output when the model loses focus at
    // large context sizes (which is itself a failure signal).
    let test_client = {
        let mut cfg = screentrack_summarizer::client::LlmConfig {
            base_url: client.base_url().unwrap_or("http://localhost:8080").to_string(),
            max_tokens: 512,
            timeout_secs: 300,
            ..Default::default()
        };
        // Copy model and API key from the real client's config via saved settings
        if let Some(saved) = crate::menubar::SavedSettings::load() {
            cfg.model = saved.model;
            cfg.api_key = saved.llm_api_key;
            cfg.base_url = saved.llm_url;
        }
        LlmClient::new(cfg)
    };

    let mut results: Vec<(usize, usize, bool, f64)> = Vec::new(); // (target, actual_tokens, pass, secs)

    for &size in sizes {
        let Some((haystack, needle_pct)) = build_haystack(&frames, size) else {
            println!(
                "  {} {:>6}k  {}",
                "⊘".yellow(),
                size / 1000,
                "SKIP — not enough frame data".yellow(),
            );
            continue;
        };

        let user_msg = format!(
            "Find the secret verification code hidden in this screen capture data:\n\n{haystack}"
        );

        print!(
            "  {} {:>6}k  ",
            "…".dimmed(),
            size / 1000,
        );
        // Flush so the user sees progress before the LLM responds
        use std::io::Write;
        let _ = std::io::stdout().flush();

        let start_time = Instant::now();
        let result = test_client.chat_with_usage(system, &user_msg).await;
        let elapsed = start_time.elapsed().as_secs_f64();

        // Clear the progress line
        print!("\r");

        match result {
            Ok((response, usage)) => {
                let found = !response.is_empty() && response.contains(NEEDLE);
                let actual_prompt = usage.prompt_tokens;
                results.push((size, actual_prompt, found, elapsed));

                let info = format!(
                    "actual: {}k tokens, needle at ~{}%, {:.0} tok/s",
                    actual_prompt / 1000,
                    needle_pct,
                    actual_prompt as f64 / elapsed,
                );

                if found {
                    println!(
                        "  {} {:>6}k  {}  {:>5.1}s  ({})",
                        "✓".green().bold(),
                        size / 1000,
                        "PASS".green().bold(),
                        elapsed,
                        info.dimmed(),
                    );
                } else {
                    let preview: String = response.chars().take(80).collect();
                    let fail_reason = if response.is_empty() {
                        "FAIL (empty response — model may have exhausted tokens on thinking)"
                    } else {
                        "FAIL (wrong answer)"
                    };
                    println!(
                        "  {} {:>6}k  {}  {:>5.1}s  ({})",
                        "✗".red().bold(),
                        size / 1000,
                        fail_reason.red().bold(),
                        elapsed,
                        info.dimmed(),
                    );
                    if !response.is_empty() {
                        println!(
                            "           {} \"{}\"",
                            "Got:".dimmed(),
                            preview.bright_black(),
                        );
                    }
                }
            }
            Err(e) => {
                results.push((size, 0, false, elapsed));
                println!(
                    "  {} {:>6}k  {}  {:>5.1}s  {}",
                    "✗".red().bold(),
                    size / 1000,
                    "ERROR".red().bold(),
                    elapsed,
                    format!("{e}").bright_black(),
                );
            }
        }
    }

    // Summary
    println!("\n{}", "━━━ Results ━━━".bold());
    let last_pass = results.iter().filter(|(_, _, ok, _)| *ok).last();
    let first_fail = results.iter().filter(|(_, _, ok, _)| !*ok).next();

    if let Some((target, actual, _, _)) = last_pass {
        println!(
            "  {} {}k target ({}k actual tokens)",
            "Max working context:".green(),
            target / 1000,
            actual / 1000,
        );
    }
    if let Some((target, actual, _, _)) = first_fail {
        println!(
            "  {} {}k target ({}k actual tokens)",
            "First failure at:".red(),
            target / 1000,
            actual / 1000,
        );
    }
    if let Some((_, actual, _, _)) = last_pass {
        let recommended = (*actual as f64 * 0.9) as usize;
        println!(
            "\n  {} Set CONTEXT_BUDGET to ~{}k in smartquery.rs",
            "Recommendation:".cyan(),
            recommended / 1000,
        );
    }

    println!();
    Ok(())
}

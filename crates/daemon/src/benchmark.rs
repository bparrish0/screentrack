use anyhow::Result;
use chrono::{Local, TimeZone, Utc};
use colored::Colorize;
use screentrack_store::Database;
use screentrack_summarizer::client::LlmClient;
use screentrack_summarizer::prompts;
use std::path::PathBuf;
use std::time::Instant;

/// Estimate tokens from text (~4 chars per token).
fn estimate_tokens(text: &str) -> usize {
    text.len() / 4
}

/// Format a duration nicely.
fn fmt_duration(d: std::time::Duration) -> String {
    if d.as_secs() >= 60 {
        format!(
            "{}m {:.1}s",
            d.as_secs() / 60,
            (d.as_secs() % 60) as f64 + d.subsec_millis() as f64 / 1000.0
        )
    } else {
        format!("{:.1}s", d.as_secs_f64())
    }
}

/// Format milliseconds as a readable timestamp.
fn fmt_ts(millis: i64) -> String {
    Utc.timestamp_millis_opt(millis)
        .unwrap()
        .with_timezone(&Local)
        .format("%Y-%m-%d %H:%M")
        .to_string()
}

pub struct BenchmarkRunOpts {
    pub context_length: usize,
    pub micro_frames: i64,
    pub micro_period: String,
    pub benchmark_db_path: PathBuf,
}

pub struct BenchmarkQueryOpts {
    pub question: String,
    pub raw: bool,
    pub max_rounds: usize,
    pub benchmark_db_path: PathBuf,
}

pub struct BenchmarkListOpts {
    pub tier: Option<String>,
    pub benchmark_db_path: PathBuf,
}

/// A single benchmark result for display.
struct BenchResult {
    name: String,
    input_tokens: usize,
    output_tokens: usize,
    duration: std::time::Duration,
    summary_preview: String,
}

/// Run the full benchmark suite.
pub async fn run_benchmark(
    prod_db: &Database,
    client: &LlmClient,
    opts: &BenchmarkRunOpts,
) -> Result<()> {
    // Open/create the benchmark database
    let bench_db = Database::new(&opts.benchmark_db_path).await?;

    println!("{}", "━━━ Screentrack LLM Benchmark ━━━".bold());
    println!(
        "  {} {}",
        "Context length:".dimmed(),
        opts.context_length.to_string().cyan()
    );
    println!(
        "  {} {}",
        "Micro frames:".dimmed(),
        opts.micro_frames.to_string().cyan()
    );
    println!(
        "  {} {}",
        "Micro period:".dimmed(),
        opts.micro_period.cyan()
    );
    println!(
        "  {} {}",
        "Benchmark DB:".dimmed(),
        opts.benchmark_db_path.display().to_string().cyan()
    );
    println!(
        "  {} {}",
        "LLM endpoint:".dimmed(),
        client.base_url().unwrap_or("unknown").cyan()
    );
    println!();

    let mut results: Vec<BenchResult> = Vec::new();

    // ── Test 1: Raw frames → micro summary ──
    println!("{}", "▸ Test 1: Raw frames → micro summary".bold());
    match run_micro_test(prod_db, &bench_db, client, opts).await {
        Ok(r) => results.push(r),
        Err(e) => eprintln!("  {} {}", "FAILED:".red().bold(), e),
    }

    // ── Test 2: Micro summaries → hourly summary ──
    println!("\n{}", "▸ Test 2: Micro summaries → hourly rollup".bold());
    match run_hourly_test(prod_db, &bench_db, client, opts).await {
        Ok(r) => results.push(r),
        Err(e) => eprintln!("  {} {}", "FAILED:".red().bold(), e),
    }

    // ── Test 3: Hourly summaries → daily summary ──
    println!("\n{}", "▸ Test 3: Hourly summaries → daily rollup".bold());
    match run_daily_test(prod_db, &bench_db, client, opts).await {
        Ok(r) => results.push(r),
        Err(e) => eprintln!("  {} {}", "FAILED:".red().bold(), e),
    }

    // ── Results table ──
    println!("\n{}", "━━━ Results ━━━".bold());
    println!(
        "  {:<35} {:>12} {:>12} {:>10} {:>10}",
        "Test".bold(),
        "Input tok".bold(),
        "Output tok".bold(),
        "Total tok".bold(),
        "Time".bold(),
    );
    println!("  {}", "─".repeat(82));

    for r in &results {
        let total = r.input_tokens + r.output_tokens;
        let pct = if opts.context_length > 0 {
            format!("({}%)", total * 100 / opts.context_length)
        } else {
            String::new()
        };
        println!(
            "  {:<35} {:>12} {:>12} {:>10} {:>10}",
            r.name,
            format!("~{}", r.input_tokens).dimmed(),
            format!("~{}", r.output_tokens).dimmed(),
            format!("~{} {}", total, pct).cyan(),
            fmt_duration(r.duration).yellow(),
        );
    }

    println!();
    println!(
        "  {} {}",
        "Benchmark summaries saved to:".dimmed(),
        opts.benchmark_db_path.display().to_string().green(),
    );
    println!(
        "  {} {}",
        "View with:".dimmed(),
        "screentrack benchmark list".cyan(),
    );
    println!(
        "  {} {}",
        "Query with:".dimmed(),
        "screentrack benchmark query \"question\"".cyan(),
    );

    Ok(())
}

/// Test 1: Raw frames → micro summary.
async fn run_micro_test(
    prod_db: &Database,
    bench_db: &Database,
    client: &LlmClient,
    opts: &BenchmarkRunOpts,
) -> Result<BenchResult> {
    let (start, end) = crate::query::infer_time_range(&opts.micro_period);

    println!(
        "  {} {} → {}",
        "Period:".dimmed(),
        fmt_ts(start).yellow(),
        fmt_ts(end).yellow(),
    );

    // Pull frames from production DB
    let filter = screentrack_store::FrameFilter {
        start: Some(start),
        end: Some(end),
        limit: Some(opts.micro_frames),
        ..Default::default()
    };
    let frames = prod_db.get_frames_filtered(&filter).await?;

    if frames.is_empty() {
        anyhow::bail!(
            "No frames found in the specified period. Try a wider --micro-period (e.g. \"today\")"
        );
    }

    if (frames.len() as i64) < opts.micro_frames {
        println!(
            "  {} only {} frames available (requested {}). Try a wider --micro-period",
            "Note:".yellow().bold(),
            frames.len(),
            opts.micro_frames,
        );
    }

    println!(
        "  {} {} frames loaded",
        "Input:".dimmed(),
        frames.len().to_string().bold(),
    );

    // Format exactly like the real summarizer
    let frame_data: Vec<prompts::FrameData> = frames
        .iter()
        .map(|f| prompts::FrameData {
            machine: Some(f.machine_id.clone()),
            app: f.app_name.clone().unwrap_or_else(|| "Unknown".into()),
            window: f.window_title.clone(),
            browser_tab: f.browser_tab.clone(),
            text: f.text_content.clone(),
            timestamp: Some(
                Utc.timestamp_millis_opt(f.timestamp)
                    .unwrap()
                    .with_timezone(&Local)
                    .format("%H:%M:%S")
                    .to_string(),
            ),
        })
        .collect();

    let mut user_input = prompts::format_frames_with_context(&frame_data, None);
    let input_tokens = estimate_tokens(prompts::MICRO_SYSTEM) + estimate_tokens(&user_input);

    if input_tokens > opts.context_length {
        println!(
            "  {} input (~{} tokens) exceeds context length ({}), truncating",
            "Warning:".yellow().bold(),
            input_tokens,
            opts.context_length,
        );
        let max_chars = (opts.context_length - estimate_tokens(prompts::MICRO_SYSTEM) - 1000) * 4;
        if user_input.len() > max_chars {
            let mut end = max_chars;
            while end > 0 && !user_input.is_char_boundary(end) {
                end -= 1;
            }
            user_input = format!("{}… (truncated to fit context)", &user_input[..end]);
        }
    }

    println!(
        "  {} ~{} tokens",
        "Est. input:".dimmed(),
        input_tokens.to_string().cyan(),
    );

    // Run the LLM
    let timer = Instant::now();
    let summary_text = client.chat(prompts::MICRO_SYSTEM, &user_input).await?;
    let duration = timer.elapsed();
    let output_tokens = estimate_tokens(&summary_text);

    // Preview
    let preview = if summary_text.len() > 200 {
        format!("{}…", &summary_text[..200])
    } else {
        summary_text.clone()
    };
    println!("  {} {}", "Time:".dimmed(), fmt_duration(duration).yellow());
    println!(
        "  {} ~{} tokens",
        "Output:".dimmed(),
        output_tokens.to_string().cyan()
    );
    println!("  {} {}", "Preview:".dimmed(), preview.bright_black());

    // Save to benchmark DB
    let mut apps: Vec<String> = frames.iter().filter_map(|f| f.app_name.clone()).collect();
    apps.sort();
    apps.dedup();

    let summary = screentrack_store::NewSummary {
        tier: "micro".into(),
        start_time: frames.first().unwrap().timestamp,
        end_time: frames.last().unwrap().timestamp,
        summary: summary_text,
        apps_referenced: Some(apps),
        source_frame_ids: Some(frames.iter().map(|f| f.id).collect()),
        source_summary_ids: None,
    };
    bench_db.insert_summary(summary).await?;

    Ok(BenchResult {
        name: format!("Micro ({} frames)", frames.len()),
        input_tokens,
        output_tokens,
        duration,
        summary_preview: preview,
    })
}

/// Test 2: Micro summaries → hourly rollup.
async fn run_hourly_test(
    prod_db: &Database,
    bench_db: &Database,
    client: &LlmClient,
    opts: &BenchmarkRunOpts,
) -> Result<BenchResult> {
    // Find a recent hour with micro summaries
    let now = Utc::now().timestamp_millis();
    let one_day_ago = now - (24 * 60 * 60 * 1000);
    let micros = prod_db.get_summaries("micro", one_day_ago, now).await?;

    if micros.is_empty() {
        anyhow::bail!("No micro summaries found in the last 24 hours");
    }

    // Take up to 60 micros from the most recent contiguous block (1 per minute for an hour)
    let take = micros.len().min(60);
    let selected = &micros[micros.len() - take..];
    let start_time = selected.first().unwrap().start_time;
    let end_time = selected.last().unwrap().end_time;

    println!(
        "  {} {} → {}",
        "Period:".dimmed(),
        fmt_ts(start_time).yellow(),
        fmt_ts(end_time).yellow(),
    );
    println!(
        "  {} {} micro summaries",
        "Input:".dimmed(),
        selected.len().to_string().bold(),
    );

    // Format exactly like the real rollup
    let summary_data: Vec<(String, String)> = selected
        .iter()
        .map(|s| {
            let time_range = format!("{}–{}", fmt_ts(s.start_time), fmt_ts(s.end_time));
            (time_range, s.summary.clone())
        })
        .collect();

    let mut user_input = prompts::format_summaries_for_rollup(&summary_data);

    // Add app time data like the real rollup does
    if let Ok(app_times) = prod_db.get_app_time(start_time, end_time).await {
        user_input.push_str(&prompts::format_app_time(&app_times));
    }

    let input_tokens = estimate_tokens(prompts::HOURLY_SYSTEM) + estimate_tokens(&user_input);

    if input_tokens > opts.context_length {
        println!(
            "  {} input (~{} tokens) exceeds context length ({}), truncating",
            "Warning:".yellow().bold(),
            input_tokens,
            opts.context_length,
        );
        let max_chars = (opts.context_length - estimate_tokens(prompts::HOURLY_SYSTEM) - 1000) * 4;
        if user_input.len() > max_chars {
            let mut end = max_chars;
            while end > 0 && !user_input.is_char_boundary(end) {
                end -= 1;
            }
            user_input = format!("{}… (truncated to fit context)", &user_input[..end]);
        }
    }

    println!(
        "  {} ~{} tokens",
        "Est. input:".dimmed(),
        input_tokens.to_string().cyan(),
    );

    let timer = Instant::now();
    let summary_text = client.chat(prompts::HOURLY_SYSTEM, &user_input).await?;
    let duration = timer.elapsed();
    let output_tokens = estimate_tokens(&summary_text);

    let preview = if summary_text.len() > 200 {
        format!("{}…", &summary_text[..200])
    } else {
        summary_text.clone()
    };
    println!("  {} {}", "Time:".dimmed(), fmt_duration(duration).yellow());
    println!(
        "  {} ~{} tokens",
        "Output:".dimmed(),
        output_tokens.to_string().cyan()
    );
    println!("  {} {}", "Preview:".dimmed(), preview.bright_black());

    // Collect apps
    let mut apps: Vec<String> = selected
        .iter()
        .filter_map(|s| s.apps_referenced.as_ref())
        .flat_map(|a| serde_json::from_str::<Vec<String>>(a).unwrap_or_default())
        .collect();
    apps.sort();
    apps.dedup();

    let summary = screentrack_store::NewSummary {
        tier: "hourly".into(),
        start_time,
        end_time,
        summary: summary_text,
        apps_referenced: Some(apps),
        source_frame_ids: None,
        source_summary_ids: Some(selected.iter().map(|s| s.id).collect()),
    };
    bench_db.insert_summary(summary).await?;

    Ok(BenchResult {
        name: format!("Hourly ({} micros)", selected.len()),
        input_tokens,
        output_tokens,
        duration,
        summary_preview: preview,
    })
}

/// Test 3: Hourly summaries → daily rollup.
async fn run_daily_test(
    prod_db: &Database,
    bench_db: &Database,
    client: &LlmClient,
    opts: &BenchmarkRunOpts,
) -> Result<BenchResult> {
    // Find hourlies from the most recent full day
    let now = Utc::now().timestamp_millis();
    let two_days_ago = now - (2 * 24 * 60 * 60 * 1000);
    let hourlies = prod_db.get_summaries("hourly", two_days_ago, now).await?;

    if hourlies.is_empty() {
        anyhow::bail!("No hourly summaries found in the last 2 days");
    }

    // Take up to 24 most recent hourlies (one day's worth)
    let take = hourlies.len().min(24);
    let hourlies = &hourlies[hourlies.len() - take..];
    let start_time = hourlies.first().unwrap().start_time;
    let end_time = hourlies.last().unwrap().end_time;

    println!(
        "  {} {} → {}",
        "Period:".dimmed(),
        fmt_ts(start_time).yellow(),
        fmt_ts(end_time).yellow(),
    );
    println!(
        "  {} {} hourly summaries",
        "Input:".dimmed(),
        hourlies.len().to_string().bold(),
    );

    let summary_data: Vec<(String, String)> = hourlies
        .iter()
        .map(|s| {
            let time_range = format!("{}–{}", fmt_ts(s.start_time), fmt_ts(s.end_time));
            (time_range, s.summary.clone())
        })
        .collect();

    let mut user_input = prompts::format_summaries_for_rollup(&summary_data);

    if let Ok(app_times) = prod_db.get_app_time(start_time, end_time).await {
        user_input.push_str(&prompts::format_app_time(&app_times));
    }

    let input_tokens = estimate_tokens(prompts::DAILY_SYSTEM) + estimate_tokens(&user_input);
    println!(
        "  {} ~{} tokens",
        "Est. input:".dimmed(),
        input_tokens.to_string().cyan(),
    );

    // Check if input exceeds context budget
    if input_tokens > opts.context_length {
        println!(
            "  {} input ({} tokens) exceeds context length ({} tokens), truncating",
            "Warning:".yellow().bold(),
            input_tokens,
            opts.context_length,
        );
        let max_chars = (opts.context_length - estimate_tokens(prompts::DAILY_SYSTEM) - 1000) * 4;
        if user_input.len() > max_chars {
            let mut end = max_chars;
            while end > 0 && !user_input.is_char_boundary(end) {
                end -= 1;
            }
            user_input = format!("{}… (truncated to fit context)", &user_input[..end]);
        }
    }

    let timer = Instant::now();
    let summary_text = client.chat(prompts::DAILY_SYSTEM, &user_input).await?;
    let duration = timer.elapsed();
    let output_tokens = estimate_tokens(&summary_text);

    let preview = if summary_text.len() > 200 {
        format!("{}…", &summary_text[..200])
    } else {
        summary_text.clone()
    };
    println!("  {} {}", "Time:".dimmed(), fmt_duration(duration).yellow());
    println!(
        "  {} ~{} tokens",
        "Output:".dimmed(),
        output_tokens.to_string().cyan()
    );
    println!("  {} {}", "Preview:".dimmed(), preview.bright_black());

    let mut apps: Vec<String> = hourlies
        .iter()
        .filter_map(|s| s.apps_referenced.as_ref())
        .flat_map(|a| serde_json::from_str::<Vec<String>>(a).unwrap_or_default())
        .collect();
    apps.sort();
    apps.dedup();

    let summary = screentrack_store::NewSummary {
        tier: "daily".into(),
        start_time,
        end_time,
        summary: summary_text,
        apps_referenced: Some(apps),
        source_frame_ids: None,
        source_summary_ids: Some(hourlies.iter().map(|s| s.id).collect()),
    };
    bench_db.insert_summary(summary).await?;

    Ok(BenchResult {
        name: format!("Daily ({} hourlies)", hourlies.len()),
        input_tokens,
        output_tokens,
        duration,
        summary_preview: preview,
    })
}

/// List summaries in the benchmark database.
pub async fn list_benchmark(opts: &BenchmarkListOpts) -> Result<()> {
    if !opts.benchmark_db_path.exists() {
        println!(
            "{}",
            "No benchmark database found. Run 'screentrack benchmark run' first.".yellow()
        );
        return Ok(());
    }

    let bench_db = Database::new(&opts.benchmark_db_path).await?;

    let tiers = if let Some(ref tier) = opts.tier {
        vec![tier.as_str()]
    } else {
        vec!["micro", "hourly", "daily", "weekly"]
    };

    let mut found = false;
    for tier in tiers {
        let summaries = bench_db.get_summaries(tier, 0, i64::MAX).await?;
        if summaries.is_empty() {
            continue;
        }
        found = true;

        let tier_colored = match tier {
            "micro" => "micro".magenta().bold(),
            "hourly" => "hourly".cyan().bold(),
            "daily" => "daily".blue().bold(),
            "weekly" => "weekly".bright_blue().bold(),
            _ => tier.white().bold(),
        };
        println!(
            "\n{} {} {}",
            tier_colored,
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
        );

        for s in &summaries {
            println!(
                "\n  {} {} → {}",
                "▸".dimmed(),
                fmt_ts(s.start_time).yellow(),
                fmt_ts(s.end_time).yellow(),
            );
            if let Some(ref apps) = s.apps_referenced {
                if let Ok(app_list) = serde_json::from_str::<Vec<String>>(apps) {
                    let colored: Vec<String> =
                        app_list.iter().map(|a| a.green().to_string()).collect();
                    println!("    {} {}", "Apps:".dimmed(), colored.join(", "));
                }
            }
            // Render markdown
            let rendered = crate::md_render::render(&s.summary);
            for line in rendered.lines() {
                println!("    {line}");
            }
        }
    }

    if !found {
        println!(
            "{}",
            "No benchmark summaries found. Run 'screentrack benchmark run' first.".yellow()
        );
    }

    Ok(())
}

/// Smart query against benchmark summaries + production frames.
pub async fn query_benchmark(
    prod_db: &Database,
    client: &LlmClient,
    opts: &BenchmarkQueryOpts,
) -> Result<()> {
    if !opts.benchmark_db_path.exists() {
        anyhow::bail!("No benchmark database found. Run 'screentrack benchmark run' first.");
    }

    let bench_db = Database::new(&opts.benchmark_db_path).await?;

    println!(
        "{} {} {}",
        "▸".dimmed(),
        "Benchmark query:".dimmed(),
        "summaries from benchmark DB, frames from production DB".bright_green(),
    );

    // Use the smart query system but with a hybrid DB wrapper
    crate::smartquery::handle_smart_query_hybrid(
        &bench_db,
        prod_db,
        client,
        &crate::smartquery::SmartQueryOpts {
            question: opts.question.clone(),
            follow_up: false,
            raw: opts.raw,
            max_rounds: opts.max_rounds,
        },
    )
    .await
}

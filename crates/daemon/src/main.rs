mod auto_update;
mod benchmark;
mod llmtest;
mod client_push;
mod md_attributed;
mod md_render;
mod menubar;
mod query;
mod scheduler;
mod server;
mod smartquery;
mod viewer;

use anyhow::Result;
use chrono::TimeZone;
use clap::{Parser, Subcommand};
use colored::Colorize;
use screentrack_store::Database;
use screentrack_summarizer::client::{LlmClient, LlmConfig};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "screentrack",
    version,
    about = "Screen capture and memory daemon"
)]
struct Cli {
    /// Path to the database file
    #[arg(long, global = true, default_value = "~/.screentrack/db.sqlite")]
    db: String,

    /// LLM server URL (OpenAI-compatible)
    #[arg(long, global = true, default_value = "http://localhost:8080")]
    llm_url: String,

    /// LLM model name
    #[arg(long, global = true, default_value = "")]
    model: String,

    /// LLM API key (Bearer token)
    #[arg(long, global = true)]
    llm_api_key: Option<String>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the capture daemon (local-only mode)
    Start {
        /// Save screenshots alongside captured text
        #[arg(long)]
        save_screenshots: bool,
    },
    /// Start as server: capture locally + receive from clients + summarize
    Serve {
        /// Save screenshots alongside captured text
        #[arg(long)]
        save_screenshots: bool,

        /// Address to listen on
        #[arg(long, default_value = "0.0.0.0:7878")]
        listen: String,
    },
    /// Start as client: capture locally + push frames to server
    Push {
        /// Save screenshots alongside captured text
        #[arg(long)]
        save_screenshots: bool,

        /// Server URL to push frames to (e.g. http://server:7878)
        #[arg(long)]
        server: String,
    },
    /// Show capture statistics
    Status,
    /// Query your activity history
    Query {
        /// The question to ask (or time expression like "last 20 seconds")
        question: String,

        /// Detail level: micro, hourly, daily, weekly, frames
        #[arg(long, default_value = "auto")]
        detail: String,

        /// Filter by application name
        #[arg(long)]
        app: Option<String>,

        /// Filter by browser tab title (substring match)
        #[arg(long)]
        tab: Option<String>,

        /// Full-text search within frame content
        #[arg(long)]
        search: Option<String>,

        /// Filter by machine name
        #[arg(long)]
        machine: Option<String>,

        /// Show raw data without LLM interpretation
        #[arg(long)]
        raw: bool,

        /// Maximum number of results
        #[arg(long, default_value = "100")]
        limit: i64,
    },
    /// List captured apps or browser tabs
    List {
        /// What to list: apps, tabs, machines
        what: String,

        /// Time range: today, yesterday, this-week (default: today)
        #[arg(long, default_value = "today")]
        range: String,
    },
    /// View your user profile (interests, projects, reminders, etc.)
    Profile {
        /// Filter by category: interest, frustration, joy, project, remember, relationship
        #[arg(long)]
        category: Option<String>,

        /// Show archived entries too
        #[arg(long)]
        archived: bool,
    },
    /// Smart query: LLM chooses what data to fetch (agentic)
    SmartQuery {
        /// The question to ask
        question: String,

        /// Follow-up: continue the previous smart-query conversation
        #[arg(short, long)]
        follow_up: bool,

        /// Show raw output without markdown rendering
        #[arg(long)]
        raw: bool,

        /// Maximum number of LLM rounds before forcing an answer
        #[arg(long, default_value = "10")]
        max_rounds: usize,

        /// Save output to a file (supports .md, .pdf, .html)
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Benchmark LLM performance against real screentrack data
    Benchmark {
        #[command(subcommand)]
        action: BenchmarkAction,

        /// Path to the benchmark database
        #[arg(long, default_value = "~/.screentrack/benchmark.sqlite")]
        benchmark_db: String,
    },
    /// Run summarization manually
    Summarize {
        /// Tier to run: micro, hourly, daily, weekly, all
        #[arg(default_value = "micro")]
        tier: String,
    },
    /// Install ScreenTrack.app to /Applications
    Install,
    /// Test LLM context length limits with needle-in-a-haystack
    LlmTest {
        /// Context sizes to test (tokens). Defaults to common sizes.
        #[arg(long, value_delimiter = ',')]
        sizes: Option<Vec<usize>>,

        /// Time range to pull frames from
        #[arg(long, default_value = "today")]
        range: String,
    },
}

#[derive(Subcommand)]
enum BenchmarkAction {
    /// Run the benchmark suite
    Run {
        /// Maximum context window length (tokens) for the LLM
        #[arg(long, default_value = "48000")]
        context_length: usize,

        /// Number of raw frames to use for the micro summary test
        #[arg(long, default_value = "200")]
        micro_frames: i64,

        /// Time period for micro summary test frames (e.g. "last 12 hours", "today")
        #[arg(long, default_value = "today")]
        micro_period: String,
    },
    /// Smart-query against benchmark summaries (frames from production DB)
    Query {
        /// The question to ask
        question: String,

        /// Show raw output without markdown rendering
        #[arg(long)]
        raw: bool,

        /// Maximum LLM rounds
        #[arg(long, default_value = "10")]
        max_rounds: usize,
    },
    /// List summaries in the benchmark database
    List {
        /// Filter by tier: micro, hourly, daily
        #[arg(long)]
        tier: Option<String>,
    },
    /// Clear the benchmark database
    Clear,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.command.is_none() {
        // GUI mode: per-layer filtering so debug log captures more than terminal
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;
        use tracing_subscriber::Layer as _;
        tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::layer().with_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            ))
            .with(
                menubar::GuiLogLayer
                    .with_filter(EnvFilter::new("debug,sqlx=warn,hyper=info,reqwest=info")),
            )
            .init();

        menubar::run_menubar_app(cli.db, cli.llm_url, cli.model, cli.llm_api_key);
        return Ok(());
    }

    // CLI mode: normal fmt subscriber
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    // CLI mode → create tokio runtime
    tokio::runtime::Runtime::new()?.block_on(async_main(cli))
}

async fn async_main(cli: Cli) -> Result<()> {
    // Load saved GUI settings — these take priority over CLI defaults.
    // CLI flags explicitly passed by the user still override saved settings,
    // but clap can't distinguish "user passed --llm-api-key X" from "default value",
    // so we always prefer saved settings when they exist.
    let saved = menubar::SavedSettings::load();

    let db_raw = saved.as_ref().map(|s| s.db_path.as_str()).unwrap_or(&cli.db);
    let llm_url = saved.as_ref().map(|s| s.llm_url.as_str()).unwrap_or(&cli.llm_url);
    let model = saved.as_ref().map(|s| s.model.as_str()).unwrap_or(&cli.model);
    let api_key = saved
        .as_ref()
        .and_then(|s| s.llm_api_key.clone())
        .or(cli.llm_api_key);

    let db_path = expand_tilde(db_raw);
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let db = Arc::new(Database::new(&db_path).await?);

    let llm_config = LlmConfig {
        base_url: llm_url.to_string(),
        model: model.to_string(),
        api_key: api_key,
        ..Default::default()
    };

    match cli.command.unwrap() {
        Commands::Start { save_screenshots } => {
            info!(
                "Starting screentrack v{} (local mode) built {}",
                env!("CARGO_PKG_VERSION"),
                env!("BUILD_TIMESTAMP")
            );
            info!("Machine ID: {}", db.machine_id);

            let capture_config = make_capture_config(save_screenshots, &db_path);
            let llm_client = Arc::new(LlmClient::new(llm_config));

            // Start summarization scheduler in background
            let sched_db = db.clone();
            let sched_client = llm_client.clone();
            tokio::spawn(async move {
                scheduler::run_scheduler(sched_db, sched_client).await;
            });

            // Run capture loop (blocks)
            screentrack_capture::capture::run_capture_loop(capture_config, db).await?;
        }

        Commands::Serve {
            save_screenshots,
            listen,
        } => {
            info!(
                "Starting screentrack v{} (server mode) built {}",
                env!("CARGO_PKG_VERSION"),
                env!("BUILD_TIMESTAMP")
            );
            info!("  Machine ID: {}", db.machine_id);
            info!("  Database: {}", db_path.display());
            info!("  Listen: {listen}");
            info!("  Endpoints:");
            info!("    GET  /api/v1/health");
            info!("    POST /api/v1/frames");

            let capture_config = make_capture_config(save_screenshots, &db_path);

            // Start HTTP server in background
            let state = Arc::new(server::AppState {
                db: db.clone(),
                llm_config: Some(llm_config.clone()),
            });

            let llm_client = Arc::new(LlmClient::new(llm_config));

            // Start summarization scheduler in background
            let sched_db = db.clone();
            let sched_client = llm_client.clone();
            tokio::spawn(async move {
                scheduler::run_scheduler(sched_db, sched_client).await;
            });
            let app = server::router(state);
            let listener = tokio::net::TcpListener::bind(&listen).await?;
            info!("HTTP server listening on {listen}");
            tokio::spawn(async move {
                if let Err(e) = axum::serve(listener, app).await {
                    tracing::error!("HTTP server error: {e}");
                }
            });

            // Run capture loop (blocks)
            screentrack_capture::capture::run_capture_loop(capture_config, db).await?;
        }

        Commands::Push {
            save_screenshots,
            server: server_url,
        } => {
            info!(
                "Starting screentrack v{} (push mode) built {}",
                env!("CARGO_PKG_VERSION"),
                env!("BUILD_TIMESTAMP")
            );
            info!("  Machine ID: {}", db.machine_id);
            info!("  Server: {server_url}");

            let capture_config = make_capture_config(save_screenshots, &db_path);

            // Start push loop in background (no summarization scheduler)
            let push_db = db.clone();
            tokio::spawn(async move {
                client_push::run_push_loop(push_db, server_url).await;
            });

            // Run capture loop (blocks)
            screentrack_capture::capture::run_capture_loop(capture_config, db).await?;
        }

        Commands::Status => {
            let stats = db.get_capture_stats().await?;
            let (_, min_ts, max_ts) = db.get_stats().await?;

            println!("{}", "━━━ Capture Stats ━━━".bold());
            println!("  {} {}", "Machine ID:".dimmed(), db.machine_id.cyan());
            println!(
                "  {} {}",
                "Frames captured:".dimmed(),
                stats.frames_total.to_string().bold()
            );
            println!(
                "    {} {}",
                "via accessibility:".dimmed(),
                stats.frames_accessibility.to_string().green()
            );
            println!(
                "    {} {}",
                "via OCR (local):".dimmed(),
                stats.frames_ocr_local.to_string().green()
            );
            println!(
                "    {} {}",
                "via OCR (remote):".dimmed(),
                stats.frames_ocr_remote.to_string().green()
            );

            // Per-machine breakdown
            let machine_stats = db.get_machine_stats().await?;
            if machine_stats.len() > 1 {
                println!();
                println!("  {}", "Frames by machine:".dimmed());
                for (machine, count) in &machine_stats {
                    println!("    {:>20}  {}", machine.cyan(), count.to_string().bold());
                }
            }

            println!();
            println!("  {}", "Frames skipped:".dimmed());
            println!(
                "    {} {}",
                "unchanged (diff):".dimmed(),
                stats.frames_skipped_unchanged.to_string().yellow()
            );
            println!(
                "    {} {}",
                "deduplicated:".dimmed(),
                stats.frames_skipped_dedup.to_string().yellow()
            );
            let total_seen =
                stats.frames_total + stats.frames_skipped_unchanged + stats.frames_skipped_dedup;
            if total_seen > 0 {
                let skip_pct = ((stats.frames_skipped_unchanged + stats.frames_skipped_dedup)
                    as f64
                    / total_seen as f64)
                    * 100.0;
                println!(
                    "    {} {}",
                    "skip rate:".dimmed(),
                    format!("{skip_pct:.1}%").yellow()
                );
            }

            if let (Some(min), Some(max)) = (min_ts, max_ts) {
                println!();
                let min_dt = chrono::Utc
                    .timestamp_millis_opt(min)
                    .unwrap()
                    .with_timezone(&chrono::Local);
                let max_dt = chrono::Utc
                    .timestamp_millis_opt(max)
                    .unwrap()
                    .with_timezone(&chrono::Local);
                println!(
                    "  {} {} {} {}",
                    "Time range:".dimmed(),
                    min_dt.format("%Y-%m-%d %H:%M:%S").to_string().yellow(),
                    "→".dimmed(),
                    max_dt.format("%Y-%m-%d %H:%M:%S").to_string().yellow(),
                );
            }

            println!();
            println!("{}", "━━━ Summaries ━━━".bold());
            for tier in &["micro", "hourly", "daily", "weekly"] {
                let summaries = db.get_summaries(tier, 0, i64::MAX).await?;
                let tier_colored = match *tier {
                    "micro" => tier.magenta().bold(),
                    "hourly" => tier.cyan().bold(),
                    "daily" => tier.blue().bold(),
                    "weekly" => tier.bright_blue().bold(),
                    _ => tier.white().bold(),
                };
                println!(
                    "  {:>16}  {}",
                    tier_colored,
                    summaries.len().to_string().bold()
                );
            }
        }

        Commands::Query {
            question,
            detail,
            app,
            tab,
            search,
            machine,
            raw,
            limit,
        } => {
            let client = LlmClient::new(llm_config);
            query::handle_query(
                &db,
                &client,
                &query::QueryOpts {
                    question,
                    detail,
                    app,
                    tab,
                    search,
                    machine,
                    raw,
                    limit,
                },
            )
            .await?;
        }

        Commands::SmartQuery {
            question,
            follow_up,
            raw,
            max_rounds,
            output,
        } => {
            let client = LlmClient::new(llm_config);
            let opts = smartquery::SmartQueryOpts {
                question: question.clone(),
                follow_up,
                raw,
                max_rounds,
            };
            if let Some(ref path) = output {
                let answer = smartquery::smart_query_get_answer(&db, &db, &client, &opts).await?;
                smartquery::export_answer(&question, &answer, path)?;
            } else {
                smartquery::handle_smart_query(&db, &client, &opts).await?;
            }
        }

        Commands::Benchmark {
            action,
            benchmark_db,
        } => {
            let bench_path = expand_tilde(&benchmark_db);
            let client = LlmClient::new(llm_config);
            match action {
                BenchmarkAction::Run {
                    context_length,
                    micro_frames,
                    micro_period,
                } => {
                    benchmark::run_benchmark(
                        &db,
                        &client,
                        &benchmark::BenchmarkRunOpts {
                            context_length,
                            micro_frames,
                            micro_period,
                            benchmark_db_path: bench_path,
                        },
                    )
                    .await?;
                }
                BenchmarkAction::Query {
                    question,
                    raw,
                    max_rounds,
                } => {
                    benchmark::query_benchmark(
                        &db,
                        &client,
                        &benchmark::BenchmarkQueryOpts {
                            question,
                            raw,
                            max_rounds,
                            benchmark_db_path: bench_path,
                        },
                    )
                    .await?;
                }
                BenchmarkAction::List { tier } => {
                    benchmark::list_benchmark(&benchmark::BenchmarkListOpts {
                        tier,
                        benchmark_db_path: bench_path,
                    })
                    .await?;
                }
                BenchmarkAction::Clear => {
                    if bench_path.exists() {
                        std::fs::remove_file(&bench_path)?;
                        println!("{} {}", "Cleared:".green(), bench_path.display());
                    } else {
                        println!("{}", "No benchmark database to clear.".yellow());
                    }
                }
            }
        }

        Commands::List { what, range } => {
            let (start, end) = query::parse_range(&range);
            match what.as_str() {
                "apps" => {
                    let apps = db.get_app_names(start, end).await?;
                    if apps.is_empty() {
                        println!("{}", "No apps captured in this time range.".yellow());
                    } else {
                        println!(
                            "{} {}\n",
                            "Apps seen".bold(),
                            format!("({})", apps.len()).dimmed()
                        );
                        for app in &apps {
                            println!("  {}", app.green());
                        }
                    }
                }
                "tabs" => {
                    let tabs = db.get_browser_tabs(start, end).await?;
                    if tabs.is_empty() {
                        println!(
                            "{}",
                            "No browser tabs captured in this time range.".yellow()
                        );
                    } else {
                        println!(
                            "{} {}\n",
                            "Browser tabs seen".bold(),
                            format!("({})", tabs.len()).dimmed()
                        );
                        for tab in &tabs {
                            println!("  {}", tab.cyan());
                        }
                    }
                }
                "machines" => {
                    let machines = db.get_machines().await?;
                    if machines.is_empty() {
                        println!("{}", "No machines captured.".yellow());
                    } else {
                        println!(
                            "{} {}\n",
                            "Machines seen".bold(),
                            format!("({})", machines.len()).dimmed()
                        );
                        for m in &machines {
                            println!("  {}", m.cyan());
                        }
                    }
                }
                "time" => {
                    let app_times = db.get_app_time(start, end).await?;
                    if app_times.is_empty() {
                        println!("{}", "No active window data in this time range.".yellow());
                    } else {
                        println!(
                            "{} {}\n",
                            "Time per app".bold(),
                            format!("({range})").dimmed()
                        );
                        for (app, ms) in &app_times {
                            println!(
                                "  {}  {}",
                                format!("{:>8}", format_duration(*ms)).cyan().bold(),
                                app.green()
                            );

                            // Show tab breakdown for browser apps
                            if is_browser_app(app) {
                                let windows = db
                                    .get_window_time(app, start, end)
                                    .await
                                    .unwrap_or_default();
                                for (title, wms) in &windows {
                                    if *wms < 1000 {
                                        continue;
                                    } // skip sub-second entries
                                    let tab = clean_browser_title(
                                        title.as_deref().unwrap_or("(untitled)"),
                                    );
                                    println!(
                                        "  {}    {}",
                                        format!("{:>8}", format_duration(*wms)).dimmed(),
                                        tab
                                    );
                                }
                            }
                        }
                        let total: i64 = app_times.iter().map(|(_, ms)| ms).sum();
                        println!(
                            "\n  {}  {}",
                            format!("{:>8}", format_duration(total)).cyan().bold(),
                            "TOTAL".bold()
                        );
                    }
                }
                _ => eprintln!(
                    "{} Unknown list type: {}. Use: apps, tabs, machines, time",
                    "error:".red().bold(),
                    what
                ),
            }
        }

        Commands::Profile {
            category,
            archived: _,
        } => {
            let entries = if let Some(ref cat) = category {
                db.get_profile_by_category(cat).await?
            } else {
                db.get_active_profile().await?
            };

            if entries.is_empty() {
                println!(
                    "{}",
                    "No profile entries yet. Profile is built during daily summarization.".yellow()
                );
                println!(
                    "{}",
                    "Run 'screentrack summarize daily' to generate one now.".dimmed()
                );
            } else {
                let category_labels: &[(&str, &str)] = &[
                    ("interest", "Interests"),
                    ("project", "Projects"),
                    ("remember", "Remember"),
                    ("relationship", "Relationships"),
                    ("joy", "Joys"),
                    ("frustration", "Frustrations"),
                ];

                for &(cat_key, cat_label) in category_labels {
                    let cat_entries: Vec<_> =
                        entries.iter().filter(|e| e.category == cat_key).collect();
                    if cat_entries.is_empty() {
                        continue;
                    }

                    let colored_label = match cat_key {
                        "interest" => cat_label.cyan().bold(),
                        "project" => cat_label.blue().bold(),
                        "remember" => cat_label.yellow().bold(),
                        "relationship" => cat_label.green().bold(),
                        "joy" => cat_label.magenta().bold(),
                        "frustration" => cat_label.red().bold(),
                        _ => cat_label.white().bold(),
                    };
                    println!("\n{colored_label}");
                    for entry in cat_entries {
                        let age = format_relative_age(entry.last_seen);
                        println!(
                            "  {} {} {}",
                            "•".dimmed(),
                            entry.content,
                            format!("({})", age).dimmed()
                        );
                    }
                }
                println!();
            }
        }

        Commands::Summarize { tier } => {
            let client = LlmClient::new(llm_config);
            match tier.as_str() {
                "micro" => {
                    let n = screentrack_summarizer::tiers::summarize_micro(&db, &client).await?;
                    println!(
                        "Created {} {} {}",
                        n.to_string().bold(),
                        "micro".magenta().bold(),
                        if n == 1 { "summary" } else { "summaries" }
                    );
                }
                "hourly" => {
                    let n = screentrack_summarizer::tiers::summarize_hourly(&db, &client).await?;
                    println!(
                        "Created {} {} {}",
                        n.to_string().bold(),
                        "hourly".cyan().bold(),
                        if n == 1 { "summary" } else { "summaries" }
                    );
                }
                "daily" => {
                    let n = screentrack_summarizer::tiers::summarize_daily(&db, &client).await?;
                    println!(
                        "Created {} {} {}",
                        n.to_string().bold(),
                        "daily".blue().bold(),
                        if n == 1 { "summary" } else { "summaries" }
                    );
                }
                "weekly" => {
                    let n = screentrack_summarizer::tiers::summarize_weekly(&db, &client).await?;
                    println!(
                        "Created {} {} {}",
                        n.to_string().bold(),
                        "weekly".bright_blue().bold(),
                        if n == 1 { "summary" } else { "summaries" }
                    );
                }
                "all" => {
                    let n1 = screentrack_summarizer::tiers::summarize_micro(&db, &client).await?;
                    let n2 = screentrack_summarizer::tiers::summarize_hourly(&db, &client).await?;
                    let n3 = screentrack_summarizer::tiers::summarize_daily(&db, &client).await?;
                    let n4 = screentrack_summarizer::tiers::summarize_weekly(&db, &client).await?;
                    println!(
                        "Created summaries: {} {}, {} {}, {} {}, {} {}",
                        n1.to_string().bold(),
                        "micro".magenta(),
                        n2.to_string().bold(),
                        "hourly".cyan(),
                        n3.to_string().bold(),
                        "daily".blue(),
                        n4.to_string().bold(),
                        "weekly".bright_blue(),
                    );
                }
                _ => {
                    eprintln!(
                        "{} Unknown tier: {}. Use: micro, hourly, daily, weekly, all",
                        "error:".red().bold(),
                        tier
                    );
                }
            }
        }

        Commands::Install => {
            install_app()?;
        }

        Commands::LlmTest { sizes, range } => {
            let sizes = sizes.unwrap_or_else(|| vec![4_000, 8_000, 16_000, 32_000, 64_000, 128_000, 192_000, 256_000]);
            let (start, end) = query::parse_range(&range);
            let client = LlmClient::new(llm_config);
            llmtest::run_context_length_test(&db, &client, &sizes, start, end).await?;
        }
    }

    Ok(())
}

fn install_app() -> Result<()> {
    let version = env!("CARGO_PKG_VERSION");
    let app_dir = PathBuf::from("/Applications/ScreenTrack.app");
    let contents = app_dir.join("Contents");
    let macos = contents.join("MacOS");
    let resources = contents.join("Resources");

    // Get path to the currently running binary
    let self_bin = std::env::current_exe()?;

    println!("Installing ScreenTrack v{version} to /Applications...");

    // Remove old install if present
    if app_dir.exists() {
        std::fs::remove_dir_all(&app_dir)?;
    }

    std::fs::create_dir_all(&macos)?;
    std::fs::create_dir_all(&resources)?;

    // Copy binary
    std::fs::copy(&self_bin, macos.join("screentrack"))?;

    // Write Info.plist
    let plist = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>ScreenTrack</string>
    <key>CFBundleDisplayName</key>
    <string>ScreenTrack</string>
    <key>CFBundleIdentifier</key>
    <string>com.screentrack.app</string>
    <key>CFBundleVersion</key>
    <string>{version}</string>
    <key>CFBundleShortVersionString</key>
    <string>{version}</string>
    <key>CFBundleExecutable</key>
    <string>screentrack</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSUIElement</key>
    <true/>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>"#
    );
    std::fs::write(contents.join("Info.plist"), plist)?;

    // Codesign the app bundle.
    // Use SCREENTRACK_SIGN_IDENTITY env var if set, otherwise ad-hoc sign ("-").
    // Ad-hoc signing works without an Apple Developer certificate and is
    // sufficient for local use. Allow in System Settings > Privacy & Security.
    let sign_identity = std::env::var("SCREENTRACK_SIGN_IDENTITY")
        .unwrap_or_else(|_| "-".to_string());

    println!("Signing app bundle with identity: {sign_identity}");
    let sign_status = std::process::Command::new("codesign")
        .args([
            "--force",
            "--sign",
            &sign_identity,
            "--deep",
            "--options",
            "runtime",
            "/Applications/ScreenTrack.app",
        ])
        .status()?;

    if sign_status.success() {
        println!(
            "{} Signed and installed to /Applications/ScreenTrack.app",
            "✓".green().bold()
        );
    } else {
        println!(
            "{} Installed but signing failed (the app may still work unsigned)",
            "⚠".yellow().bold()
        );
    }
    println!("  Run: {}", "open /Applications/ScreenTrack.app".cyan());

    Ok(())
}

fn expand_tilde(path: &str) -> PathBuf {
    if path.starts_with("~/") {
        if let Some(home) = dirs_next::home_dir() {
            return home.join(&path[2..]);
        }
    }
    PathBuf::from(path)
}

fn format_duration(ms: i64) -> String {
    if ms >= 3_600_000 {
        format!("{:.1}h", ms as f64 / 3_600_000.0)
    } else if ms >= 60_000 {
        format!("{}m", ms / 60_000)
    } else {
        format!("{}s", ms / 1000)
    }
}

const BROWSER_APPS_LIST: &[&str] = &[
    "Safari",
    "Google Chrome",
    "Firefox",
    "Arc",
    "Brave Browser",
    "Microsoft Edge",
    "Chromium",
    "Opera",
    "Vivaldi",
    "Orion",
];

fn is_browser_app(app: &str) -> bool {
    BROWSER_APPS_LIST.iter().any(|&b| b == app)
}

/// Clean a browser window title into a readable tab name.
/// Strips browser name suffixes and tab count prefixes like "(699) ".
fn clean_browser_title(title: &str) -> String {
    let suffixes = &[
        " - Mozilla Firefox",
        " - Firefox",
        " - Google Chrome",
        " - Brave Browser",
        " - Brave",
        " - Microsoft Edge",
        " - Chromium",
        " - Opera",
        " - Vivaldi",
        " - Orion",
        " — Arc",
        " - Arc",
        " - Safari",
        " - YouTube",
    ];

    let mut cleaned = title.to_string();
    for suffix in suffixes {
        if cleaned.ends_with(suffix) {
            cleaned = cleaned[..cleaned.len() - suffix.len()].to_string();
            break;
        }
    }

    // Strip leading tab count like "(699) "
    if cleaned.starts_with('(') {
        if let Some(end) = cleaned.find(") ") {
            let inside = &cleaned[1..end];
            if inside.chars().all(|c| c.is_ascii_digit()) {
                cleaned = cleaned[end + 2..].to_string();
            }
        }
    }

    cleaned
}

fn format_relative_age(timestamp_ms: i64) -> String {
    let now = chrono::Utc::now().timestamp_millis();
    let age_ms = now - timestamp_ms;
    let hours = age_ms / 3_600_000;
    if hours < 1 {
        "just now".into()
    } else if hours < 24 {
        format!("{}h ago", hours)
    } else {
        let days = hours / 24;
        if days == 1 {
            "yesterday".into()
        } else {
            format!("{}d ago", days)
        }
    }
}

fn make_capture_config(
    save_screenshots: bool,
    db_path: &std::path::Path,
) -> screentrack_capture::capture::CaptureConfig {
    let force_ocr_apps_path = db_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join("force_ocr_apps.json");
    let force_ocr_apps = screentrack_capture::capture::load_force_ocr_apps(&force_ocr_apps_path);

    screentrack_capture::capture::CaptureConfig {
        save_screenshots,
        screenshot_dir: Some(db_path.parent().unwrap().join("screenshots")),
        force_ocr_apps,
        force_ocr_apps_path: Some(force_ocr_apps_path),
        ..Default::default()
    }
}

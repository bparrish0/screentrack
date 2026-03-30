use anyhow::Result;
use chrono::{Duration, Local, NaiveTime, TimeZone, Utc};
use colored::Colorize;
use screentrack_store::{Database, FrameFilter};
use screentrack_summarizer::client::LlmClient;
use screentrack_summarizer::prompts;
/// Render Markdown text to the terminal with ANSI formatting.
fn print_markdown(text: &str) {
    let rendered = crate::md_render::render(text);
    print!("{rendered}");
}

pub struct QueryOpts {
    pub question: String,
    pub detail: String,
    pub app: Option<String>,
    pub tab: Option<String>,
    pub search: Option<String>,
    pub machine: Option<String>,
    pub raw: bool,
    pub limit: i64,
}

/// Handle a user query about their activity.
pub async fn handle_query(db: &Database, client: &LlmClient, opts: &QueryOpts) -> Result<()> {
    let (start, end) = infer_time_range(&opts.question);
    let has_filters =
        opts.app.is_some() || opts.tab.is_some() || opts.search.is_some() || opts.machine.is_some();

    // Show the time range being queried
    let start_dt = Utc
        .timestamp_millis_opt(start)
        .unwrap()
        .with_timezone(&Local);
    let end_dt = Utc.timestamp_millis_opt(end).unwrap().with_timezone(&Local);
    let end_fmt = if start_dt.date_naive() == end_dt.date_naive() {
        end_dt.format("%H:%M").to_string()
    } else {
        end_dt.format("%Y-%m-%d %H:%M").to_string()
    };
    eprintln!(
        "{} {} {} {}",
        "▸".dimmed(),
        "Time range:".dimmed(),
        start_dt.format("%Y-%m-%d %H:%M").to_string().yellow(),
        format!("→ {}", end_fmt).yellow(),
    );

    // Determine detail level
    let detail = if opts.detail == "auto" {
        let d = auto_detail_level(start, end, has_filters);
        eprintln!(
            "{} {} {}",
            "▸".dimmed(),
            "Auto detail level:".dimmed(),
            tier_label(&d),
        );
        d
    } else {
        opts.detail.clone()
    };

    // If detail is "frames" or we have specific filters, query raw frames
    if detail == "frames" || has_filters {
        return handle_frames_query(db, client, opts, start, end).await;
    }

    // Try summaries, supplementing with finer tiers to fill gaps.
    // For example, if we want hourly summaries for a 2-hour window but only 1 hourly
    // exists (covering the first hour), we also pull micro summaries for the uncovered
    // second hour so the LLM has the full picture.
    let summaries = db.get_summaries(&detail, start, end).await?;

    let supplement_tiers: &[&str] = match detail.as_str() {
        "weekly" => &["daily", "hourly", "micro"],
        "daily" => &["hourly", "micro"],
        "hourly" => &["micro"],
        _ => &[],
    };

    // Collect the primary summaries
    let mut all_summaries: Vec<screentrack_store::Summary> = summaries;
    let mut supplement_tier_used: Option<&str> = None;
    let mut supplement_count = 0usize;

    // Check if the primary tier has sparse coverage — if summaries exist but don't
    // cover the full time range, fill gaps with finer-grained summaries
    if !all_summaries.is_empty() && !supplement_tiers.is_empty() {
        // Find the time ranges NOT covered by the primary summaries
        let covered_end = all_summaries
            .iter()
            .map(|s| s.end_time)
            .max()
            .unwrap_or(start);
        let covered_start = all_summaries
            .iter()
            .map(|s| s.start_time)
            .min()
            .unwrap_or(end);

        // Look for supplements in gaps (before first summary and after last summary)
        for &tier in supplement_tiers {
            let mut gap_summaries = Vec::new();
            // Gap before the first primary summary
            if covered_start > start {
                let before = db
                    .get_summaries(tier, start, covered_start)
                    .await
                    .unwrap_or_default();
                gap_summaries.extend(before);
            }
            // Gap after the last primary summary
            if covered_end < end {
                let after = db
                    .get_summaries(tier, covered_end, end)
                    .await
                    .unwrap_or_default();
                gap_summaries.extend(after);
            }
            if !gap_summaries.is_empty() {
                supplement_count = gap_summaries.len();
                supplement_tier_used = Some(tier);
                all_summaries.extend(gap_summaries);
                all_summaries.sort_by_key(|s| s.start_time);
                break; // Use the finest tier that has data
            }
        }
    }

    if !all_summaries.is_empty() {
        let primary_count = all_summaries.len() - supplement_count;
        if supplement_count > 0 {
            eprintln!(
                "{} {} {} {} {} {} {} {} {}\n",
                "▸".dimmed(),
                "Source:".dimmed(),
                tier_label(&detail),
                format!("({primary_count}").dimmed(),
                format!(
                    "{})",
                    if primary_count == 1 {
                        "summary"
                    } else {
                        "summaries"
                    }
                )
                .dimmed(),
                "+".dimmed(),
                tier_label(supplement_tier_used.unwrap()),
                format!("({supplement_count}").dimmed(),
                format!(
                    "{} filling gaps)",
                    if supplement_count == 1 {
                        "summary"
                    } else {
                        "summaries"
                    }
                )
                .dimmed(),
            );
        } else {
            print_source_banner(&detail, all_summaries.len(), None);
        }
        return display_summaries(client, &opts.question, &all_summaries, &detail, opts.raw).await;
    }

    // No summaries at primary tier — try falling back entirely to finer tiers
    for &tier in supplement_tiers {
        let summaries = db.get_summaries(tier, start, end).await?;
        if !summaries.is_empty() {
            print_source_banner(tier, summaries.len(), Some(&detail));
            return display_summaries(client, &opts.question, &summaries, tier, opts.raw).await;
        }
    }

    // No summaries at all — fall back to raw frames
    eprintln!(
        "{} {}",
        "▸".dimmed(),
        "No summaries found, falling back to raw frames...".yellow(),
    );
    handle_frames_query(db, client, opts, start, end).await
}

/// Print a colored banner showing what data source is being used.
fn print_source_banner(tier: &str, count: usize, fell_back_from: Option<&str>) {
    if let Some(original) = fell_back_from {
        eprintln!(
            "{} No {} summaries found, fell back to {}",
            "▸".dimmed(),
            tier_label(original),
            tier_label(tier),
        );
    }
    eprintln!(
        "{} {} {} {} {}\n",
        "▸".dimmed(),
        "Source:".dimmed(),
        tier_label(tier),
        format!("({count}").dimmed(),
        format!("{})", if count == 1 { "summary" } else { "summaries" }).dimmed(),
    );
}

/// Return a colorized tier label.
pub fn tier_label(tier: &str) -> String {
    match tier {
        "micro" => "micro".magenta().bold().to_string(),
        "hourly" => "hourly".cyan().bold().to_string(),
        "daily" => "daily".blue().bold().to_string(),
        "weekly" => "weekly".bright_blue().bold().to_string(),
        "frames" => "frames".green().bold().to_string(),
        _ => tier.white().bold().to_string(),
    }
}

async fn handle_frames_query(
    db: &Database,
    client: &LlmClient,
    opts: &QueryOpts,
    start: i64,
    end: i64,
) -> Result<()> {
    let filter = FrameFilter {
        start: Some(start),
        end: Some(end),
        app_name: opts.app.clone(),
        browser_tab_contains: opts.tab.clone(),
        source: None,
        search_text: opts.search.clone(),
        machine_id: opts.machine.clone(),
        limit: Some(opts.limit),
    };

    let frames = db.get_frames_filtered(&filter).await?;

    if frames.is_empty() {
        println!("{}", "No frames found for this query.".yellow());
        return Ok(());
    }

    eprintln!(
        "{} {} {} {} {}\n",
        "▸".dimmed(),
        "Source:".dimmed(),
        tier_label("frames"),
        format!("({}", frames.len()).dimmed(),
        format!("{})", if frames.len() == 1 { "frame" } else { "frames" }).dimmed(),
    );

    if opts.raw {
        // Only show machine name if frames span multiple machines
        let machines: std::collections::HashSet<_> = frames.iter().map(|f| &f.machine_id).collect();
        let show_machine = machines.len() > 1;

        println!("{}\n", format!("━━━ {} frames ━━━", frames.len()).bold());
        for f in &frames {
            let ts = Utc
                .timestamp_millis_opt(f.timestamp)
                .unwrap()
                .with_timezone(&Local);
            let app = f.app_name.as_deref().unwrap_or("?");
            let window = f.window_title.as_deref().unwrap_or("");

            // Build the header line with colors
            let mut header = format!("{}", ts.format("%H:%M:%S").to_string().yellow());
            header.push_str(&format!(" [{}", app.green()));
            if !window.is_empty() {
                header.push_str(&format!(" — {}", window.white()));
            }
            if let Some(ref tab) = f.browser_tab {
                header.push_str(&format!(" {} {}", "|".dimmed(), tab.cyan()));
            }
            if show_machine {
                header.push_str(&format!(" {}", format!("@{}", f.machine_id).bright_black()));
            }
            header.push_str(&format!("] {}", format!("({})", f.source).dimmed()));
            println!("{header}");

            // Truncate very long text for display
            let text = if f.text_content.len() > 500 {
                let mut end = 500;
                while end > 0 && !f.text_content.is_char_boundary(end) {
                    end -= 1;
                }
                format!("{}…", &f.text_content[..end])
            } else {
                f.text_content.clone()
            };
            println!("{text}");
            println!("{}", "─".repeat(60).dimmed());
        }
        return Ok(());
    }

    // Send frames to LLM for interpretation
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

    let context = prompts::format_frames_for_micro(&frame_data);
    let user_msg = format!(
        "Here is raw screen capture data ({} frames):\n\n{context}\n\nQuestion: {}",
        frames.len(),
        opts.question
    );

    let answer = client.chat(prompts::QUERY_SYSTEM, &user_msg).await?;
    print_markdown(&answer);

    Ok(())
}

async fn display_summaries(
    client: &LlmClient,
    question: &str,
    summaries: &[screentrack_store::Summary],
    tier: &str,
    raw: bool,
) -> Result<()> {
    if raw {
        println!(
            "{}\n",
            format!("━━━ {} summaries ({}) ━━━", tier, summaries.len()).bold()
        );
        for s in summaries {
            let start = Utc
                .timestamp_millis_opt(s.start_time)
                .unwrap()
                .with_timezone(&Local);
            let end = Utc
                .timestamp_millis_opt(s.end_time)
                .unwrap()
                .with_timezone(&Local);
            println!(
                "{} {} {}",
                start.format("%Y-%m-%d %H:%M").to_string().yellow(),
                "→".dimmed(),
                end.format("%H:%M").to_string().yellow(),
            );
            if let Some(ref apps) = s.apps_referenced {
                if let Ok(app_list) = serde_json::from_str::<Vec<String>>(apps) {
                    let colored_apps: Vec<String> =
                        app_list.iter().map(|a| a.green().to_string()).collect();
                    println!("{} {}", "Apps:".dimmed(), colored_apps.join(", "));
                } else {
                    println!("{} {}", "Apps:".dimmed(), apps);
                }
            }
            print_markdown(&s.summary);
            println!("\n{}", "─".repeat(60).dimmed());
        }
        return Ok(());
    }

    let summary_data: Vec<(String, String)> = summaries
        .iter()
        .map(|s| {
            let start = Utc
                .timestamp_millis_opt(s.start_time)
                .unwrap()
                .with_timezone(&Local);
            let end = Utc
                .timestamp_millis_opt(s.end_time)
                .unwrap()
                .with_timezone(&Local);
            let time_range = format!(
                "{} to {}",
                start.format("%Y-%m-%d %H:%M"),
                end.format("%H:%M")
            );
            (time_range, s.summary.clone())
        })
        .collect();

    let context = prompts::format_summaries_for_rollup(&summary_data);
    let user_msg = format!("Here are my activity summaries:\n\n{context}\n\nQuestion: {question}");

    let answer = client.chat(prompts::QUERY_SYSTEM, &user_msg).await?;
    print_markdown(&answer);

    Ok(())
}

/// Auto-select the best detail level based on the time range.
fn auto_detail_level(start: i64, end: i64, has_filters: bool) -> String {
    if has_filters {
        return "frames".into();
    }

    let duration_secs = (end - start) / 1000;
    match duration_secs {
        0..=300 => "frames".into(),       // ≤5 min: raw frames
        301..=3600 => "micro".into(),     // 5 min–1 hour: micro summaries
        3601..=86400 => "hourly".into(),  // 1 hour–1 day: hourly
        86401..=604800 => "daily".into(), // 1–7 days: daily
        _ => "weekly".into(),             // >7 days: weekly
    }
}

/// Infer a time range from natural language in the question.
/// Returns (start_millis, end_millis).
pub fn infer_time_range(question: &str) -> (i64, i64) {
    let now = Utc::now();
    let q = question.to_lowercase();

    // Check for "N seconds/minutes/hours ago" patterns
    if let Some(duration) = parse_relative_duration(&q) {
        let start = now - duration;
        return (start.timestamp_millis(), now.timestamp_millis());
    }

    if q.contains("today") {
        let start_of_day = now
            .with_timezone(&Local)
            .date_naive()
            .and_time(NaiveTime::from_hms_opt(0, 0, 0).unwrap());
        let start = Local
            .from_local_datetime(&start_of_day)
            .unwrap()
            .with_timezone(&Utc);
        (start.timestamp_millis(), now.timestamp_millis())
    } else if q.contains("yesterday") {
        let yesterday = now - Duration::days(1);
        let start_of_day = yesterday
            .with_timezone(&Local)
            .date_naive()
            .and_time(NaiveTime::from_hms_opt(0, 0, 0).unwrap());
        let end_of_day = now
            .with_timezone(&Local)
            .date_naive()
            .and_time(NaiveTime::from_hms_opt(0, 0, 0).unwrap());
        let start = Local
            .from_local_datetime(&start_of_day)
            .unwrap()
            .with_timezone(&Utc);
        let end = Local
            .from_local_datetime(&end_of_day)
            .unwrap()
            .with_timezone(&Utc);
        (start.timestamp_millis(), end.timestamp_millis())
    } else if q.contains("this week") || q.contains("past week") || q.contains("last week") {
        let start = now - Duration::days(7);
        (start.timestamp_millis(), now.timestamp_millis())
    } else if q.contains("this morning") {
        let start_of_day = now
            .with_timezone(&Local)
            .date_naive()
            .and_time(NaiveTime::from_hms_opt(0, 0, 0).unwrap());
        let noon = now
            .with_timezone(&Local)
            .date_naive()
            .and_time(NaiveTime::from_hms_opt(12, 0, 0).unwrap());
        let start = Local
            .from_local_datetime(&start_of_day)
            .unwrap()
            .with_timezone(&Utc);
        let end = Local
            .from_local_datetime(&noon)
            .unwrap()
            .with_timezone(&Utc);
        (
            start.timestamp_millis(),
            end.timestamp_millis().min(now.timestamp_millis()),
        )
    } else if q.contains("this afternoon") {
        let noon = now
            .with_timezone(&Local)
            .date_naive()
            .and_time(NaiveTime::from_hms_opt(12, 0, 0).unwrap());
        let start = Local
            .from_local_datetime(&noon)
            .unwrap()
            .with_timezone(&Utc);
        (start.timestamp_millis(), now.timestamp_millis())
    } else if let Some((start_ms, end_ms)) = parse_time_of_day_range(&q, now) {
        (start_ms, end_ms)
    } else if let Some((start_ms, end_ms)) = parse_date_string(&q) {
        (start_ms, end_ms)
    } else {
        // Default: last 24 hours
        let start = now - Duration::days(1);
        (start.timestamp_millis(), now.timestamp_millis())
    }
}

/// Parse relative duration expressions like "20 seconds ago", "last 5 minutes", "past 2.5 hours".
/// Supports integer, decimal, and English number words (e.g. "two hours", "1.5 days").
fn parse_relative_duration(q: &str) -> Option<Duration> {
    // Replace English number words with digits before matching
    let q = replace_number_words(q);

    // Each pattern captures a number (integer or decimal) and maps to seconds-per-unit
    let patterns: &[(&str, f64)] = &[
        (r"(\d+(?:\.\d+)?)\s*seconds?\s*ago", 1.0),
        (r"(\d+(?:\.\d+)?)\s*minutes?\s*ago", 60.0),
        (r"(\d+(?:\.\d+)?)\s*hours?\s*ago", 3600.0),
        (r"(\d+(?:\.\d+)?)\s*days?\s*ago", 86400.0),
        (r"(?:last|past)\s+(\d+(?:\.\d+)?)\s*seconds?", 1.0),
        (r"(?:last|past)\s+(\d+(?:\.\d+)?)\s*minutes?", 60.0),
        (r"(?:last|past)\s+(\d+(?:\.\d+)?)\s*hours?", 3600.0),
        (r"(?:last|past)\s+(\d+(?:\.\d+)?)\s*days?", 86400.0),
    ];

    for (pattern, secs_per_unit) in patterns {
        if let Some(caps) = regex_lite::Regex::new(pattern).ok()?.captures(&q) {
            if let Some(n) = caps.get(1).and_then(|m| m.as_str().parse::<f64>().ok()) {
                let total_secs = (n * secs_per_unit) as i64;
                return Some(Duration::seconds(total_secs));
            }
        }
    }

    // "last hour", "past hour"
    if q.contains("last hour") || q.contains("past hour") {
        return Some(Duration::hours(1));
    }

    None
}

/// Parse an explicit time-of-day range like "from 4AM to 6AM", "between 2pm and 5pm",
/// "4am to 6am", "4am-6am". Returns millisecond timestamps in the user's local timezone.
/// If the end time is still in the future, uses today's date; otherwise uses today
/// (or yesterday if both times are in the future).
fn parse_time_of_day_range(
    q: &str,
    now: chrono::DateTime<Utc>,
) -> Option<(i64, i64)> {
    // Match patterns like "4am to 6am", "from 4 AM to 6 PM", "between 2pm and 5pm",
    // "4am-6am", "from 10:30am to 2pm", "4 am to 6 am"
    let time_pattern = r"(\d{1,2})(?::(\d{2}))?\s*([ap]\.?m\.?)\s*(?:to|[-–—]|and|through|until)\s*(\d{1,2})(?::(\d{2}))?\s*([ap]\.?m\.?)";
    let re = regex_lite::Regex::new(time_pattern).ok()?;
    let caps = re.captures(q)?;

    let start_hour_raw: u32 = caps.get(1)?.as_str().parse().ok()?;
    let start_min: u32 = caps.get(2).and_then(|m| m.as_str().parse().ok()).unwrap_or(0);
    let start_ampm = caps.get(3)?.as_str().to_lowercase().replace('.', "");

    let end_hour_raw: u32 = caps.get(4)?.as_str().parse().ok()?;
    let end_min: u32 = caps.get(5).and_then(|m| m.as_str().parse().ok()).unwrap_or(0);
    let end_ampm = caps.get(6)?.as_str().to_lowercase().replace('.', "");

    let to_24h = |hour: u32, ampm: &str| -> Option<u32> {
        if hour > 12 || hour == 0 { return None; }
        if ampm.starts_with('a') {
            Some(if hour == 12 { 0 } else { hour })
        } else {
            Some(if hour == 12 { 12 } else { hour + 12 })
        }
    };

    let start_h = to_24h(start_hour_raw, &start_ampm)?;
    let end_h = to_24h(end_hour_raw, &end_ampm)?;

    let local_now = now.with_timezone(&Local);
    let today = local_now.date_naive();

    // Build the time-of-day range on today first
    let start_time = NaiveTime::from_hms_opt(start_h, start_min, 0)?;
    let end_time = NaiveTime::from_hms_opt(end_h, end_min, 0)?;

    // Determine which date to use. If the end time is in the future, the user
    // likely means today. If both times are in the past, also today.
    // If both are in the future (e.g., asking at 2am about "4am to 6am"),
    // they probably mean yesterday.
    let date = if start_time > local_now.time() && end_time > local_now.time() {
        today - chrono::Duration::days(1)
    } else {
        today
    };

    let start_dt = date.and_time(start_time);
    let end_dt = date.and_time(end_time);

    let start_utc = Local.from_local_datetime(&start_dt).unwrap().with_timezone(&Utc);
    let end_utc = Local.from_local_datetime(&end_dt).unwrap().with_timezone(&Utc);

    Some((start_utc.timestamp_millis(), end_utc.timestamp_millis()))
}

/// Parse an explicit date string like "2026-03-27" and return the full day range
/// (midnight to midnight) in the user's local timezone.
fn parse_date_string(q: &str) -> Option<(i64, i64)> {
    let re = regex_lite::Regex::new(r"(\d{4}-\d{2}-\d{2})").ok()?;
    let caps = re.captures(q)?;
    let date_str = caps.get(1)?.as_str();
    let date = chrono::NaiveDate::parse_from_str(date_str, "%Y-%m-%d").ok()?;

    let start_dt = date.and_time(NaiveTime::from_hms_opt(0, 0, 0)?);
    let end_dt = (date + Duration::days(1)).and_time(NaiveTime::from_hms_opt(0, 0, 0)?);

    let start = Local.from_local_datetime(&start_dt).unwrap().with_timezone(&Utc);
    let end = Local.from_local_datetime(&end_dt).unwrap().with_timezone(&Utc);

    Some((start.timestamp_millis(), end.timestamp_millis()))
}

/// Replace English number words with digits so duration parsing can match them.
/// Uses word-boundary-aware replacement to avoid corrupting other words
/// (e.g. "a" in "last" or "have").
fn replace_number_words(q: &str) -> String {
    let words: &[(&str, &str)] = &[
        ("twenty", "20"),
        ("nineteen", "19"),
        ("eighteen", "18"),
        ("seventeen", "17"),
        ("sixteen", "16"),
        ("fifteen", "15"),
        ("fourteen", "14"),
        ("thirteen", "13"),
        ("twelve", "12"),
        ("eleven", "11"),
        ("ten", "10"),
        ("nine", "9"),
        ("eight", "8"),
        ("seven", "7"),
        ("six", "6"),
        ("five", "5"),
        ("four", "4"),
        ("three", "3"),
        ("two", "2"),
        ("one", "1"),
        ("a couple", "2"),
        ("a few", "3"),
        ("half an", "0.5"),
        ("half a", "0.5"),
        ("a half", "0.5"),
        ("an", "1"),
        ("a", "1"),
    ];
    let mut result = q.to_string();
    for &(word, digit) in words {
        // Use regex word boundaries to avoid replacing "a" inside "last", "have", etc.
        if let Ok(re) = regex_lite::Regex::new(&format!(r"\b{}\b", regex_lite::escape(word))) {
            result = re.replace_all(&result, digit).into_owned();
        }
    }
    result
}

/// Parse a named range for the `list` command.
pub fn parse_range(range: &str) -> (i64, i64) {
    let now = Utc::now();
    match range {
        "today" => {
            let start_of_day = now
                .with_timezone(&Local)
                .date_naive()
                .and_time(NaiveTime::from_hms_opt(0, 0, 0).unwrap());
            let start = Local
                .from_local_datetime(&start_of_day)
                .unwrap()
                .with_timezone(&Utc);
            (start.timestamp_millis(), now.timestamp_millis())
        }
        "yesterday" => {
            let yesterday = now - Duration::days(1);
            let start = yesterday
                .with_timezone(&Local)
                .date_naive()
                .and_time(NaiveTime::from_hms_opt(0, 0, 0).unwrap());
            let end = now
                .with_timezone(&Local)
                .date_naive()
                .and_time(NaiveTime::from_hms_opt(0, 0, 0).unwrap());
            let s = Local
                .from_local_datetime(&start)
                .unwrap()
                .with_timezone(&Utc);
            let e = Local.from_local_datetime(&end).unwrap().with_timezone(&Utc);
            (s.timestamp_millis(), e.timestamp_millis())
        }
        "this-week" => {
            let start = now - Duration::days(7);
            (start.timestamp_millis(), now.timestamp_millis())
        }
        _ => {
            // Default: all time
            (0, now.timestamp_millis())
        }
    }
}

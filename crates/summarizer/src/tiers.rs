use anyhow::Result;
use chrono::{DateTime, Duration, TimeZone, Utc};
use screentrack_store::{Database, NewSummary};
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::client::LlmClient;
use crate::prompts;

/// Run micro-level summarization for recent unsummarized frames.
pub async fn summarize_micro(db: &Database, client: &LlmClient) -> Result<u32> {
    // Look back 5 minutes for any unsummarized frames
    let since = Utc::now().timestamp_millis() - (5 * 60 * 1000);
    let frames = db.get_unsummarized_frames(since).await?;

    if frames.is_empty() {
        debug!("No unsummarized frames");
        return Ok(0);
    }

    // Group frames into 1-minute windows
    let mut windows: Vec<Vec<&screentrack_store::Frame>> = Vec::new();
    let window_ms: i64 = 60 * 1000;
    let mut current_window_start = frames[0].timestamp;
    let mut current_window = Vec::new();

    for frame in &frames {
        if frame.timestamp - current_window_start > window_ms {
            if !current_window.is_empty() {
                windows.push(current_window);
                current_window = Vec::new();
            }
            current_window_start = frame.timestamp;
        }
        current_window.push(frame);
    }
    if !current_window.is_empty() {
        windows.push(current_window);
    }

    let mut count = 0;

    // Fetch the most recent micro summary for sliding context.
    // This lets the LLM link activities across micro boundaries
    // (e.g. "accessing 10.0.0.137" → "checking DHCP for the same device").
    let mut prev_summary_text: Option<String> =
        db.get_latest_summary("micro").await?.map(|s| s.summary);

    for window in windows {
        let start_time = window.first().unwrap().timestamp;
        let end_time = window.last().unwrap().timestamp;
        let frame_ids: Vec<i64> = window.iter().map(|f| f.id).collect();
        let frame_machine_id = window.first().unwrap().machine_id.clone();

        // Collect apps referenced
        let mut apps: Vec<String> = window.iter().filter_map(|f| f.app_name.clone()).collect();
        apps.sort();
        apps.dedup();

        // Format frames for the LLM, including previous summary as context
        let frame_data: Vec<prompts::FrameData> = window
            .iter()
            .map(|f| prompts::FrameData {
                machine: Some(f.machine_id.clone()),
                app: f.app_name.clone().unwrap_or_else(|| "Unknown".into()),
                window: f.window_title.clone(),
                browser_tab: f.browser_tab.clone(),
                text: f.text_content.clone(),
                timestamp: Some(format_timestamp(f.timestamp)),
            })
            .collect();

        let user_input =
            prompts::format_frames_with_context(&frame_data, prev_summary_text.as_deref());

        match client.chat(prompts::MICRO_SYSTEM, &user_input).await {
            Ok(summary_text) if summary_text.trim().is_empty() => {
                warn!("LLM returned empty micro summary for {start_time}..{end_time}, skipping");
            }
            Ok(summary_text) => {
                let summary = NewSummary {
                    tier: "micro".into(),
                    start_time,
                    end_time,
                    summary: summary_text.clone(),
                    apps_referenced: Some(apps),
                    source_frame_ids: Some(frame_ids.clone()),
                    source_summary_ids: None,
                };
                db.insert_summary(summary).await?;
                // Mark source frames as summarized
                db.mark_frames_summarized(&frame_machine_id, &frame_ids)
                    .await?;
                count += 1;
                info!("Created micro summary for {start_time}..{end_time}");

                // Slide the context window forward
                prev_summary_text = Some(summary_text);
            }
            Err(e) => {
                warn!("Failed to generate micro summary: {e}");
            }
        }
    }

    Ok(count)
}

/// Roll up micro summaries into an hourly summary.
pub async fn summarize_hourly(db: &Database, client: &LlmClient) -> Result<u32> {
    rollup(db, client, "micro", "hourly", 60 * 60 * 1000).await
}

/// Roll up hourly summaries into a daily summary.
pub async fn summarize_daily(db: &Database, client: &LlmClient) -> Result<u32> {
    rollup(db, client, "hourly", "daily", 24 * 60 * 60 * 1000).await
}

/// Roll up daily summaries into a weekly summary.
pub async fn summarize_weekly(db: &Database, client: &LlmClient) -> Result<u32> {
    rollup(db, client, "daily", "weekly", 7 * 24 * 60 * 60 * 1000).await
}

async fn rollup(
    db: &Database,
    client: &LlmClient,
    child_tier: &str,
    parent_tier: &str,
    window_ms: i64,
) -> Result<u32> {
    let since = Utc::now().timestamp_millis() - (window_ms * 2);
    let children = db
        .get_unrolled_summaries(child_tier, parent_tier, since)
        .await?;

    if children.is_empty() {
        debug!("No unrolled {child_tier} summaries for {parent_tier}");
        return Ok(0);
    }

    // Group children into windows.
    // For daily rollups, align to calendar days (local time) so a "daily" covers
    // one actual day regardless of gaps. For other tiers, group by time proximity.
    let windows: Vec<Vec<&screentrack_store::Summary>> = if parent_tier == "daily" {
        group_by_calendar_day(&children)
    } else if parent_tier == "weekly" {
        group_by_calendar_week(&children)
    } else {
        // Hourly: group by clock-aligned hour
        group_by_calendar_hour(&children)
    };

    // For daily/weekly, don't roll up the current (incomplete) day/week.
    // Only process windows where the period has ended.
    let now = Utc::now().timestamp_millis();
    let min_children = match parent_tier {
        "daily" => 3,  // Need at least 3 hourlies to make a meaningful daily
        "weekly" => 2, // Need at least 2 dailies for a weekly
        _ => 1,
    };

    let system_prompt = match parent_tier {
        "hourly" => prompts::HOURLY_SYSTEM,
        "daily" => prompts::DAILY_SYSTEM,
        "weekly" => prompts::WEEKLY_SYSTEM,
        _ => prompts::HOURLY_SYSTEM,
    };

    let mut count = 0;
    let mut last_summary_text: Option<String> = None;

    for window in windows {
        let start_time = window.first().unwrap().start_time;
        let end_time = window.last().unwrap().end_time;

        // Skip windows that are too small (not enough child summaries)
        if window.len() < min_children {
            debug!(
                "Skipping {parent_tier} window {start_time}..{end_time}: only {} {child_tier} summaries (need {min_children})",
                window.len()
            );
            continue;
        }

        // For hourly, skip if the hour hasn't ended yet.
        if parent_tier == "hourly" {
            let hour_ms = 60 * 60 * 1000;
            let current_hour_start = now - (now % hour_ms);
            if start_time >= current_hour_start {
                debug!(
                    "Skipping hourly window {start_time}..{end_time}: hour still in progress"
                );
                continue;
            }
        }

        // For daily, skip if the day hasn't ended yet (we haven't passed the next 4:30 AM).
        // For weekly, skip if any summary is from the current day period.
        if parent_tier == "daily" {
            let boundary = last_day_boundary_ms();
            if end_time > boundary {
                debug!(
                    "Skipping daily window {start_time}..{end_time}: day still in progress (boundary: {boundary})"
                );
                continue;
            }
        } else if parent_tier == "weekly" && end_time > now - (24 * 60 * 60 * 1000) {
            debug!("Skipping weekly window {start_time}..{end_time}: week still in progress");
            continue;
        }

        let summary_ids: Vec<i64> = window.iter().map(|s| s.id).collect();

        // Collect all apps
        let mut apps: Vec<String> = window
            .iter()
            .filter_map(|s| s.apps_referenced.as_ref())
            .flat_map(|a| serde_json::from_str::<Vec<String>>(a).unwrap_or_default())
            .collect();
        apps.sort();
        apps.dedup();

        // Format for LLM
        let summary_data: Vec<(String, String)> = window
            .iter()
            .map(|s| {
                let time_range = format!(
                    "{}–{}",
                    format_timestamp(s.start_time),
                    format_timestamp(s.end_time)
                );
                (time_range, s.summary.clone())
            })
            .collect();

        let mut user_input = prompts::format_summaries_for_rollup(&summary_data);

        // Inject measured app time data for hourly and daily rollups
        if matches!(parent_tier, "hourly" | "daily") {
            if let Ok(app_times) = db.get_app_time(start_time, end_time).await {
                user_input.push_str(&prompts::format_app_time(&app_times));
            }
        }

        match client.chat(system_prompt, &user_input).await {
            Ok(summary_text) if summary_text.trim().is_empty() => {
                warn!("LLM returned empty {parent_tier} summary for {start_time}..{end_time}, skipping");
            }
            Ok(summary_text) => {
                let summary = NewSummary {
                    tier: parent_tier.into(),
                    start_time,
                    end_time,
                    summary: summary_text.clone(),
                    apps_referenced: Some(apps),
                    source_frame_ids: None,
                    source_summary_ids: Some(summary_ids),
                };
                db.insert_summary(summary).await?;
                count += 1;
                info!("Created {parent_tier} summary for {start_time}..{end_time}");
                last_summary_text = Some(summary_text);
            }
            Err(e) => {
                warn!("Failed to generate {parent_tier} summary: {e}");
            }
        }
    }

    // Update user profile from daily and weekly summaries only.
    // Micro/hourly are too granular and flood the profile with noise.
    if matches!(parent_tier, "daily" | "weekly") {
        if let Some(ref text) = last_summary_text {
            if let Err(e) = update_profile(db, client, text).await {
                warn!("Failed to update user profile from {parent_tier}: {e}");
            }
        }
    }

    Ok(count)
}

/// Update the user profile based on a summary.
/// Called after every tier of summarization with the most recent summary text.
async fn update_profile(db: &Database, client: &LlmClient, summary_text: &str) -> Result<()> {
    let profile_entries = db.get_active_profile().await?;
    let profile_tuples: Vec<(i64, String, String)> = profile_entries
        .iter()
        .map(|e| (e.id, e.category.clone(), e.content.clone()))
        .collect();

    let user_input = prompts::format_profile_update_input(&profile_tuples, summary_text);

    let response = client
        .chat(prompts::PROFILE_UPDATE_SYSTEM, &user_input)
        .await?;

    // Strip markdown code fences if the LLM wraps the JSON
    let json_str = response
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    let delta: ProfileDelta = serde_json::from_str(json_str).map_err(|e| {
        warn!("Profile update LLM returned invalid JSON: {json_str}");
        e
    })?;

    let mut added = 0u32;
    let mut updated = 0u32;
    let mut archived = 0u32;

    for entry in &delta.add {
        if !VALID_CATEGORIES.contains(&entry.category.as_str()) {
            warn!("Profile: ignoring unknown category '{}'", entry.category);
            continue;
        }
        db.insert_profile_entry(&entry.category, &entry.content, None)
            .await?;
        added += 1;
    }

    for entry in &delta.update {
        db.update_profile_entry(entry.id, &entry.content, None)
            .await?;
        updated += 1;
    }

    for &id in &delta.archive {
        db.archive_profile_entry(id).await?;
        archived += 1;
    }

    if added > 0 || updated > 0 || archived > 0 {
        info!(
            "Profile updated: +{added} added, ~{updated} updated, -{archived} archived ({} total active)",
            profile_entries.len() as i64 + added as i64 - archived as i64
        );
    } else {
        debug!("Profile: no changes from summary");
    }

    Ok(())
}

const VALID_CATEGORIES: &[&str] = &[
    "interest",
    "frustration",
    "joy",
    "project",
    "remember",
    "relationship",
];

#[derive(serde::Deserialize)]
struct ProfileDelta {
    #[serde(default)]
    add: Vec<ProfileAdd>,
    #[serde(default)]
    update: Vec<ProfileUpdate>,
    #[serde(default)]
    archive: Vec<i64>,
}

#[derive(serde::Deserialize)]
struct ProfileAdd {
    category: String,
    content: String,
}

#[derive(serde::Deserialize)]
struct ProfileUpdate {
    id: i64,
    content: String,
}

fn format_timestamp(millis: i64) -> String {
    let dt = Utc.timestamp_millis_opt(millis).unwrap();
    dt.format("%H:%M").to_string()
}

/// The hour and minute that defines the boundary between "days" for daily rollups.
/// Activity before this time belongs to the previous day's summary.
/// 4:30 AM local time — the user is almost certainly asleep.
pub const DAY_BOUNDARY_HOUR: u32 = 4;
pub const DAY_BOUNDARY_MINUTE: u32 = 30;

/// Group summaries by "screen day" (local time, split at 4:30 AM).
/// Activity at 2 AM on March 27 belongs to the "March 26" day.
fn group_by_calendar_day<'a>(
    summaries: &'a [screentrack_store::Summary],
) -> Vec<Vec<&'a screentrack_store::Summary>> {
    use chrono::{Duration, Local};
    use std::collections::BTreeMap;

    // Offset to shift times back so that 4:30 AM becomes midnight
    let boundary_offset =
        Duration::hours(DAY_BOUNDARY_HOUR as i64) + Duration::minutes(DAY_BOUNDARY_MINUTE as i64);

    let mut by_date: BTreeMap<chrono::NaiveDate, Vec<&screentrack_store::Summary>> =
        BTreeMap::new();

    for s in summaries {
        let local_dt = Utc
            .timestamp_millis_opt(s.start_time)
            .unwrap()
            .with_timezone(&Local);
        // Subtract the boundary offset so that e.g. 2:00 AM on Mar 27 → 9:30 PM on Mar 26
        let shifted = local_dt - boundary_offset;
        let date = shifted.date_naive();
        by_date.entry(date).or_default().push(s);
    }

    by_date.into_values().collect()
}

/// Compute the most recent 4:30 AM boundary in local time as a UTC millis timestamp.
/// If it's currently before 4:30 AM, returns yesterday's 4:30 AM.
pub fn last_day_boundary_ms() -> i64 {
    use chrono::{Local, NaiveTime, TimeZone};

    let now_local = Utc::now().with_timezone(&Local);
    let boundary_time = NaiveTime::from_hms_opt(DAY_BOUNDARY_HOUR, DAY_BOUNDARY_MINUTE, 0).unwrap();
    let today_boundary = now_local.date_naive().and_time(boundary_time);
    let today_boundary_local = Local.from_local_datetime(&today_boundary).unwrap();

    let boundary = if now_local < today_boundary_local {
        // Before today's 4:30 AM — use yesterday's
        today_boundary_local - chrono::Duration::days(1)
    } else {
        today_boundary_local
    };

    boundary.with_timezone(&Utc).timestamp_millis()
}

/// Group summaries by clock-aligned hour (local time).
/// All summaries whose start_time falls in the same calendar hour go into one group.
fn group_by_calendar_hour<'a>(
    summaries: &'a [screentrack_store::Summary],
) -> Vec<Vec<&'a screentrack_store::Summary>> {
    use chrono::{Local, Timelike};
    use std::collections::BTreeMap;

    let mut by_hour: BTreeMap<(chrono::NaiveDate, u32), Vec<&screentrack_store::Summary>> =
        BTreeMap::new();

    for s in summaries {
        let local_dt = Utc
            .timestamp_millis_opt(s.start_time)
            .unwrap()
            .with_timezone(&Local);
        let key = (local_dt.date_naive(), local_dt.hour());
        by_hour.entry(key).or_default().push(s);
    }

    by_hour.into_values().collect()
}

/// Group summaries by calendar week (local time, ISO week).
/// All summaries whose start_time falls in the same ISO week go into one group.
fn group_by_calendar_week<'a>(
    summaries: &'a [screentrack_store::Summary],
) -> Vec<Vec<&'a screentrack_store::Summary>> {
    use chrono::{Datelike, Local};
    use std::collections::BTreeMap;

    let mut by_week: BTreeMap<(i32, u32), Vec<&screentrack_store::Summary>> = BTreeMap::new();

    for s in summaries {
        let local_dt = Utc
            .timestamp_millis_opt(s.start_time)
            .unwrap()
            .with_timezone(&Local);
        let iso = local_dt.iso_week();
        by_week.entry((iso.year(), iso.week())).or_default().push(s);
    }

    by_week.into_values().collect()
}

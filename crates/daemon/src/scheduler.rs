use chrono::{Local, NaiveTime, TimeZone, Utc};
use screentrack_store::Database;
use screentrack_summarizer::client::LlmClient;
use screentrack_summarizer::tiers;
use std::sync::Arc;
use tokio::time::{interval, sleep, Duration};
use tracing::{info, warn};

/// Run the summarization scheduler. This runs in the background and triggers
/// summarization at appropriate intervals.
///
/// On startup, immediately catches up any overdue summaries (micro → hourly →
/// daily → weekly) so that restarts don't leave gaps. Then enters the normal
/// interval loop.
///
/// Daily summaries are triggered at 4:30 AM local time, which is the boundary
/// between "days". If the app wasn't running at 4:30 AM, the startup catch-up
/// handles it.
pub async fn run_scheduler(db: Arc<Database>, client: Arc<LlmClient>) {
    info!("Summarization scheduler started");

    // Catch-up: run all tiers immediately on startup to fill any gaps
    // from downtime (sleep/wake, restarts, etc.)
    info!("Scheduler: running startup catch-up...");
    run_all_tiers(&db, &client).await;
    info!("Scheduler: catch-up complete, entering interval loop");

    let mut micro_interval = interval(Duration::from_secs(60)); // Every 1 minute
    let mut hourly_interval = interval(Duration::from_secs(60 * 60)); // Every hour

    // Skip the first immediate tick (catch-up already ran)
    micro_interval.tick().await;
    hourly_interval.tick().await;

    // Spawn the daily/weekly trigger on its own task, aligned to 4:30 AM
    let daily_db = db.clone();
    let daily_client = client.clone();
    tokio::spawn(async move {
        loop {
            let wait = duration_until_next_daily_boundary();
            info!(
                "Scheduler: next daily rollup in {:.1}h (at 4:30 AM local)",
                wait.as_secs_f64() / 3600.0
            );
            sleep(wait).await;

            // It's 4:30 AM — yesterday's data is complete, roll it up
            info!("Scheduler: 4:30 AM — running daily + weekly rollup");
            match tiers::summarize_daily(&daily_db, &daily_client).await {
                Ok(n) if n > 0 => info!("Scheduler: created {n} daily summaries"),
                Ok(_) => {}
                Err(e) => warn!("Scheduler: daily summarization failed: {e}"),
            }
            match tiers::summarize_weekly(&daily_db, &daily_client).await {
                Ok(n) if n > 0 => info!("Scheduler: created {n} weekly summaries"),
                Ok(_) => {}
                Err(e) => warn!("Scheduler: weekly summarization failed: {e}"),
            }

            // Sleep a bit to avoid re-triggering if the clock hasn't moved past 4:30
            sleep(Duration::from_secs(120)).await;
        }
    });

    loop {
        tokio::select! {
            _ = micro_interval.tick() => {
                match tiers::summarize_micro(&db, &client).await {
                    Ok(n) if n > 0 => info!("Scheduler: created {n} micro summaries"),
                    Ok(_) => {},
                    Err(e) => warn!("Scheduler: micro summarization failed: {e}"),
                }
            }
            _ = hourly_interval.tick() => {
                match tiers::summarize_hourly(&db, &client).await {
                    Ok(n) if n > 0 => info!("Scheduler: created {n} hourly summaries"),
                    Ok(_) => {},
                    Err(e) => warn!("Scheduler: hourly summarization failed: {e}"),
                }
            }
        }
    }
}

/// Run all summarization tiers in order. Used for startup catch-up.
async fn run_all_tiers(db: &Database, client: &LlmClient) {
    match tiers::summarize_micro(db, client).await {
        Ok(n) if n > 0 => info!("Catch-up: created {n} micro summaries"),
        Ok(_) => {}
        Err(e) => warn!("Catch-up: micro failed: {e}"),
    }
    match tiers::summarize_hourly(db, client).await {
        Ok(n) if n > 0 => info!("Catch-up: created {n} hourly summaries"),
        Ok(_) => {}
        Err(e) => warn!("Catch-up: hourly failed: {e}"),
    }
    match tiers::summarize_daily(db, client).await {
        Ok(n) if n > 0 => info!("Catch-up: created {n} daily summaries"),
        Ok(_) => {}
        Err(e) => warn!("Catch-up: daily failed: {e}"),
    }
    match tiers::summarize_weekly(db, client).await {
        Ok(n) if n > 0 => info!("Catch-up: created {n} weekly summaries"),
        Ok(_) => {}
        Err(e) => warn!("Catch-up: weekly failed: {e}"),
    }
}

/// Calculate how long until the next 4:30 AM local time.
fn duration_until_next_daily_boundary() -> Duration {
    let now_local = Utc::now().with_timezone(&Local);
    let boundary_time =
        NaiveTime::from_hms_opt(tiers::DAY_BOUNDARY_HOUR, tiers::DAY_BOUNDARY_MINUTE, 0).unwrap();

    let today_boundary = now_local.date_naive().and_time(boundary_time);
    let today_boundary_local = Local.from_local_datetime(&today_boundary).unwrap();

    let next_boundary = if now_local >= today_boundary_local {
        // Already past today's 4:30 AM — target tomorrow
        today_boundary_local + chrono::Duration::days(1)
    } else {
        today_boundary_local
    };

    let until = next_boundary.signed_duration_since(now_local);
    Duration::from_secs(until.num_seconds().max(60) as u64)
}

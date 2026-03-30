use screentrack_store::Database;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::time::{interval, Duration};
use tracing::{debug, info, warn};

#[derive(Debug, Serialize)]
struct FrameBatch {
    machine_id: String,
    frames: Vec<RemoteFrame>,
}

#[derive(Debug, Serialize)]
struct RemoteFrame {
    client_id: i64,
    timestamp: i64,
    app_name: Option<String>,
    window_title: Option<String>,
    browser_tab: Option<String>,
    text_content: String,
    source: String,
}

#[derive(Debug, Deserialize)]
struct FrameBatchResponse {
    accepted: u32,
    deduplicated: u32,
    #[allow(dead_code)]
    last_client_id: Option<i64>,
}

/// Run the push loop, periodically sending unsynced frames to the server.
pub async fn run_push_loop(db: Arc<Database>, server_url: String) {
    info!("Push loop started");
    info!("  Server: {server_url}");
    info!("  Machine ID: {}", db.machine_id);
    info!("  Push interval: 10s");

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to build HTTP client");

    // Startup health check
    let health_url = format!("{server_url}/api/v1/health");
    info!("Checking server connectivity...");
    match client.get(&health_url).send().await {
        Ok(resp) if resp.status().is_success() => {
            info!("Server is reachable at {server_url}");
        }
        Ok(resp) => {
            warn!("Server returned {} — will keep retrying", resp.status());
        }
        Err(e) => {
            warn!("Cannot reach server at {server_url}: {e}");
            warn!("Frames will be buffered locally and pushed when server becomes available");
        }
    }

    let mut tick = interval(Duration::from_secs(10));
    let mut consecutive_failures = 0u32;

    loop {
        tick.tick().await;

        let unsynced = match db.get_unsynced_frames(500).await {
            Ok(frames) => frames,
            Err(e) => {
                warn!("Failed to get unsynced frames: {e}");
                continue;
            }
        };

        if unsynced.is_empty() {
            debug!("No unsynced frames to push");
            continue;
        }

        let count = unsynced.len();
        let ids: Vec<i64> = unsynced.iter().map(|f| f.id).collect();
        let batch = FrameBatch {
            machine_id: db.machine_id.clone(),
            frames: unsynced
                .iter()
                .map(|f| RemoteFrame {
                    client_id: f.id,
                    timestamp: f.timestamp,
                    app_name: f.app_name.clone(),
                    window_title: f.window_title.clone(),
                    browser_tab: f.browser_tab.clone(),
                    text_content: f.text_content.clone(),
                    source: f.source.clone(),
                })
                .collect(),
        };

        info!("Pushing {count} frames to {server_url}...");
        let url = format!("{server_url}/api/v1/frames");
        match client.post(&url).json(&batch).send().await {
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<FrameBatchResponse>().await {
                    Ok(result) => {
                        info!(
                            "Push OK: {} accepted, {} deduplicated",
                            result.accepted, result.deduplicated
                        );
                        if let Err(e) = db.mark_frames_synced(&ids).await {
                            warn!("Failed to mark frames as synced: {e}");
                        }
                        consecutive_failures = 0;
                    }
                    Err(e) => warn!("Failed to parse server response: {e}"),
                }
            }
            Ok(resp) => {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                warn!("Server returned {status}: {body}");
                consecutive_failures += 1;
            }
            Err(e) => {
                consecutive_failures += 1;
                if consecutive_failures <= 3 {
                    warn!("Push failed: {e} (attempt {consecutive_failures}, will retry)");
                } else if consecutive_failures % 30 == 0 {
                    // Log every 5 minutes (30 * 10s) when server is down
                    warn!(
                        "Push still failing after {} attempts ({} frames buffered): {e}",
                        consecutive_failures, count
                    );
                }
            }
        }
    }
}

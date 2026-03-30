use axum::{
    extract::{DefaultBodyLimit, Query, State},
    routing::{get, post},
    Json, Router,
};
use base64::Engine as _;
use screentrack_store::{Database, NewFrame};
use screentrack_summarizer::client::{LlmClient, LlmConfig};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, warn};

pub struct AppState {
    pub db: Arc<Database>,
    pub llm_config: Option<LlmConfig>,
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/v1/health", get(health))
        .route("/api/v1/frames", post(receive_frames))
        .route("/api/v1/summaries", get(get_summaries))
        .route("/api/v1/stats", get(get_stats))
        .route("/api/v1/smart-query", post(smart_query))
        .route("/api/v1/vision/analyze", post(vision_analyze))
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024)) // 64MB
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}

// ---------------------------------------------------------------------------
// Frame ingestion (existing)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct FrameBatch {
    pub machine_id: String,
    pub frames: Vec<RemoteFrame>,
}

#[derive(Debug, Deserialize)]
pub struct RemoteFrame {
    pub client_id: i64,
    pub timestamp: i64,
    pub app_name: Option<String>,
    pub window_title: Option<String>,
    pub browser_tab: Option<String>,
    pub text_content: String,
    pub source: String,
}

#[derive(Debug, Serialize)]
pub struct FrameBatchResponse {
    pub accepted: u32,
    pub deduplicated: u32,
    pub last_client_id: Option<i64>,
}

async fn receive_frames(
    State(state): State<Arc<AppState>>,
    Json(batch): Json<FrameBatch>,
) -> Json<FrameBatchResponse> {
    let mut accepted = 0u32;
    let mut deduplicated = 0u32;
    let mut errors = 0u32;
    let total = batch.frames.len();

    info!(
        "Receiving {} frames from machine '{}'",
        total, batch.machine_id
    );

    for remote in &batch.frames {
        let frame = NewFrame {
            app_name: remote.app_name.clone(),
            window_title: remote.window_title.clone(),
            browser_tab: remote.browser_tab.clone(),
            text_content: remote.text_content.clone(),
            source: remote.source.clone(),
            screenshot_path: None,
        };

        match state
            .db
            .insert_frame_for_machine(&batch.machine_id, frame, Some(remote.timestamp))
            .await
        {
            Ok(Some(_)) => accepted += 1,
            Ok(None) => deduplicated += 1,
            Err(e) => {
                warn!(
                    "Failed to insert remote frame from '{}': {e}",
                    batch.machine_id
                );
                errors += 1;
            }
        }
    }

    if errors > 0 {
        info!(
            "Batch from '{}': {} accepted, {} deduplicated, {} errors (of {} total)",
            batch.machine_id, accepted, deduplicated, errors, total
        );
    } else {
        info!(
            "Batch from '{}': {} accepted, {} deduplicated",
            batch.machine_id, accepted, deduplicated
        );
    }

    Json(FrameBatchResponse {
        accepted,
        deduplicated,
        last_client_id: batch.frames.last().map(|f| f.client_id),
    })
}

// ---------------------------------------------------------------------------
// Query endpoints (for push clients)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct SummaryQuery {
    tier: String,
    start: i64,
    end: i64,
}

#[derive(Debug, Serialize)]
struct SummaryResponse {
    pub id: i64,
    pub tier: String,
    pub start_time: i64,
    pub end_time: i64,
    pub summary: String,
    pub apps_referenced: Option<String>,
}

async fn get_summaries(
    State(state): State<Arc<AppState>>,
    Query(q): Query<SummaryQuery>,
) -> Json<Vec<SummaryResponse>> {
    let summaries = state
        .db
        .get_summaries(&q.tier, q.start, q.end)
        .await
        .unwrap_or_default();

    Json(
        summaries
            .into_iter()
            .map(|s| SummaryResponse {
                id: s.id,
                tier: s.tier,
                start_time: s.start_time,
                end_time: s.end_time,
                summary: s.summary,
                apps_referenced: s.apps_referenced,
            })
            .collect(),
    )
}

#[derive(Debug, Serialize)]
struct StatsResponse {
    frames_total: i64,
    frames_accessibility: i64,
    frames_ocr_local: i64,
    frames_ocr_remote: i64,
    frames_skipped_unchanged: i64,
    frames_skipped_dedup: i64,
    summary_counts: Vec<(String, i64)>,
}

async fn get_stats(State(state): State<Arc<AppState>>) -> Json<StatsResponse> {
    let capture = state.db.get_capture_stats().await.unwrap_or_default();
    let counts = state
        .db
        .get_summary_counts(0, i64::MAX)
        .await
        .unwrap_or_default();

    Json(StatsResponse {
        frames_total: capture.frames_total,
        frames_accessibility: capture.frames_accessibility,
        frames_ocr_local: capture.frames_ocr_local,
        frames_ocr_remote: capture.frames_ocr_remote,
        frames_skipped_unchanged: capture.frames_skipped_unchanged,
        frames_skipped_dedup: capture.frames_skipped_dedup,
        summary_counts: counts,
    })
}

#[derive(Debug, Deserialize)]
struct SmartQueryRequest {
    question: String,
    #[serde(default = "default_max_rounds")]
    max_rounds: usize,
}

fn default_max_rounds() -> usize {
    10
}

#[derive(Debug, Serialize)]
struct SmartQueryResponse {
    answer: String,
}

async fn smart_query(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SmartQueryRequest>,
) -> Json<SmartQueryResponse> {
    let Some(ref llm_config) = state.llm_config else {
        return Json(SmartQueryResponse {
            answer: "LLM not configured on this server.".into(),
        });
    };

    let client = LlmClient::new(llm_config.clone());
    let opts = crate::smartquery::SmartQueryOpts {
        question: req.question,
        follow_up: false,
        raw: true,
        max_rounds: req.max_rounds,
    };

    let answer = crate::smartquery::smart_query_get_answer(&state.db, &state.db, &client, &opts)
        .await
        .unwrap_or_else(|e| format!("Error: {e}"));

    Json(SmartQueryResponse { answer })
}

#[derive(Debug, Deserialize)]
struct VisionAnalyzeRequest {
    /// Base64-encoded image payload; accepts either raw base64 or a data URL.
    image_base64: String,
    /// Optional prompt for non-text analysis.
    prompt: Option<String>,
    /// Optional model override.
    model: Option<String>,
    /// Optional max tokens override.
    max_tokens: Option<u32>,
}

#[derive(Debug, Serialize)]
struct VisionAnalyzeResponse {
    text: Option<String>,
    model: Option<String>,
    error: Option<String>,
}

fn strip_data_url_prefix(input: &str) -> &str {
    let trimmed = input.trim();
    if let Some((prefix, rest)) = trimmed.split_once(',') {
        if prefix.starts_with("data:") {
            return rest;
        }
    }
    trimmed
}

async fn vision_analyze(
    State(state): State<Arc<AppState>>,
    Json(req): Json<VisionAnalyzeRequest>,
) -> Json<VisionAnalyzeResponse> {
    let Some(ref llm_config) = state.llm_config else {
        return Json(VisionAnalyzeResponse {
            text: None,
            model: None,
            error: Some("LLM not configured on this server.".into()),
        });
    };

    let image_bytes = match base64::engine::general_purpose::STANDARD
        .decode(strip_data_url_prefix(&req.image_base64))
    {
        Ok(v) => v,
        Err(e) => {
            return Json(VisionAnalyzeResponse {
                text: None,
                model: None,
                error: Some(format!("Invalid base64 image payload: {e}")),
            });
        }
    };

    let image = match image::load_from_memory(&image_bytes) {
        Ok(img) => img,
        Err(e) => {
            return Json(VisionAnalyzeResponse {
                text: None,
                model: None,
                error: Some(format!("Invalid image data: {e}")),
            });
        }
    };

    let ocr_config = screentrack_capture::ocr::LlmOcrConfig {
        base_url: llm_config.base_url.clone(),
        model: req
            .model
            .clone()
            .unwrap_or_else(|| llm_config.model.clone()),
        api_key: llm_config.api_key.clone(),
        timeout_secs: llm_config.timeout_secs,
        max_tokens: req.max_tokens.unwrap_or(llm_config.max_tokens),
        prompt: None,
    };
    let prompt = req.prompt.clone();
    let model_name = ocr_config.model.clone();

    let result = tokio::task::spawn_blocking(move || {
        screentrack_capture::ocr::analyze_image_with_llm(&image, &ocr_config, prompt.as_deref())
    })
    .await;

    match result {
        Ok(Ok(out)) => Json(VisionAnalyzeResponse {
            text: Some(out.text),
            model: Some(model_name),
            error: None,
        }),
        Ok(Err(e)) => Json(VisionAnalyzeResponse {
            text: None,
            model: Some(model_name),
            error: Some(format!("Vision analysis failed: {e}")),
        }),
        Err(e) => Json(VisionAnalyzeResponse {
            text: None,
            model: Some(model_name),
            error: Some(format!("Vision analysis task failed: {e}")),
        }),
    }
}

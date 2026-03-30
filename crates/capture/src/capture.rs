use anyhow::Result;
use screentrack_store::{Database, NewFrame};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

use crate::accessibility;
use crate::events::{CaptureEvent, EventConfig};
use crate::frame_compare::{FrameCompareConfig, FrameComparer};
use crate::monitor;

/// Apps that show historical conversation content — only capture full text
/// if the user is actively typing in them, not just viewing.
const PASSIVE_CONTENT_APPS: &[&str] = &[
    "Messages",
    "Slack",
    "Discord",
    "WhatsApp",
    "Telegram",
    "Signal",
    "Microsoft Teams",
];

/// How recently the user must have typed in a passive content app for full text capture.
const PASSIVE_INPUT_TIMEOUT_SECS: u64 = 15;

fn is_passive_content_app(app_name: &str) -> bool {
    PASSIVE_CONTENT_APPS.iter().any(|&a| a == app_name)
}

/// Apps whose accessibility tree doesn't capture the main content (e.g. remote desktop
/// sessions render as a graphics surface). Use OCR instead.
const OCR_PREFERRED_APPS: &[&str] = &["Royal TSX", "Codex", "ScreenConnect"];

pub fn default_force_ocr_apps() -> Vec<String> {
    OCR_PREFERRED_APPS.iter().map(|s| s.to_string()).collect()
}

/// If accessibility character count repeats for this many captures, probe OCR.
const ADAPTIVE_OCR_REPEAT_STREAK: usize = 6;
/// Ignore tiny accessibility payloads for adaptive promotion checks.
const ADAPTIVE_OCR_MIN_CHARS: usize = 120;
/// Promote app to OCR-preferred when similarity falls below this threshold.
const ADAPTIVE_OCR_SIMILARITY_THRESHOLD: f64 = 0.35;

#[derive(Default, Clone, Copy)]
struct RepeatLengthState {
    last_len: usize,
    streak: usize,
}

fn normalize_app_name(app_name: &str) -> String {
    app_name.trim().to_string()
}

fn app_name_matches(a: &str, b: &str) -> bool {
    a.trim().eq_ignore_ascii_case(b.trim())
}

fn force_ocr_contains(force_ocr_apps: &HashSet<String>, app_name: &str) -> bool {
    force_ocr_apps.iter().any(|a| app_name_matches(a, app_name))
}

fn force_ocr_insert(force_ocr_apps: &mut HashSet<String>, app_name: &str) -> bool {
    if force_ocr_contains(force_ocr_apps, app_name) {
        return false;
    }
    force_ocr_apps.insert(normalize_app_name(app_name));
    true
}

fn sorted_force_ocr_apps(force_ocr_apps: &HashSet<String>) -> Vec<String> {
    let mut apps: Vec<String> = force_ocr_apps
        .iter()
        .map(|a| a.trim().to_string())
        .filter(|a| !a.is_empty())
        .collect();
    apps.sort_by_key(|a| a.to_ascii_lowercase());
    apps.dedup_by(|a, b| a.eq_ignore_ascii_case(b));
    apps
}

fn token_jaccard_similarity(a: &str, b: &str) -> f64 {
    let tokenize = |s: &str| -> HashSet<String> {
        s.split_whitespace()
            .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|t| !t.is_empty())
            .map(|t| t.to_ascii_lowercase())
            .collect()
    };
    let a_tokens = tokenize(a);
    let b_tokens = tokenize(b);
    if a_tokens.is_empty() && b_tokens.is_empty() {
        return 1.0;
    }
    if a_tokens.is_empty() || b_tokens.is_empty() {
        return 0.0;
    }
    let intersection = a_tokens.intersection(&b_tokens).count() as f64;
    let union = a_tokens.union(&b_tokens).count() as f64;
    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

fn is_substantially_different(accessibility_text: &str, ocr_text: &str) -> bool {
    let a = accessibility_text.trim();
    let b = ocr_text.trim();
    if b.is_empty() {
        return false;
    }
    if a.is_empty() {
        return b.len() >= ADAPTIVE_OCR_MIN_CHARS;
    }

    let similarity = token_jaccard_similarity(a, b);
    let min_len = a.len().min(b.len()) as f64;
    let max_len = a.len().max(b.len()) as f64;
    let len_ratio = if max_len == 0.0 {
        0.0
    } else {
        min_len / max_len
    };

    similarity < ADAPTIVE_OCR_SIMILARITY_THRESHOLD || (similarity < 0.55 && len_ratio < 0.5)
}

fn update_repeat_length_state(
    repeat_state: &mut HashMap<String, RepeatLengthState>,
    app_name: &str,
    text_len: usize,
) -> usize {
    let app_key = normalize_app_name(app_name);
    if text_len < ADAPTIVE_OCR_MIN_CHARS {
        repeat_state.remove(&app_key);
        return 0;
    }

    let state = repeat_state.entry(app_key).or_default();
    if state.last_len == text_len {
        state.streak += 1;
    } else {
        state.last_len = text_len;
        state.streak = 1;
    }
    state.streak
}

/// Load force-OCR applications from disk.
pub fn load_force_ocr_apps(path: &std::path::Path) -> Vec<String> {
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };
    let parsed = serde_json::from_str::<Vec<String>>(&data).unwrap_or_default();
    let set: HashSet<String> = parsed
        .into_iter()
        .map(|a| a.trim().to_string())
        .filter(|a| !a.is_empty())
        .collect();
    sorted_force_ocr_apps(&set)
}

/// Save force-OCR applications to disk.
pub fn save_force_ocr_apps(path: &std::path::Path, apps: &[String]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let set: HashSet<String> = apps
        .iter()
        .map(|a| a.trim().to_string())
        .filter(|a| !a.is_empty())
        .collect();
    let sorted = sorted_force_ocr_apps(&set);
    let json = serde_json::to_string_pretty(&sorted).unwrap_or_else(|_| "[]".to_string());
    std::fs::write(path, json)
}

fn run_ocr_with_timing(
    image: &image::DynamicImage,
    app_name: Option<&str>,
    phase: &str,
) -> Result<crate::ocr::OcrResult> {
    let started = Instant::now();
    let result = crate::ocr::ocr_image(image);
    let elapsed_ms = started.elapsed().as_millis();
    let app = app_name.unwrap_or("unknown");

    match &result {
        Ok(ocr) => {
            debug!(
                "OCR {} [apple] for '{}' took {}ms ({} chars)",
                phase,
                app,
                elapsed_ms,
                ocr.text.len()
            );
        }
        Err(err) => {
            debug!(
                "OCR {} [apple] for '{}' failed after {}ms: {}",
                phase, app, elapsed_ms, err
            );
        }
    }

    result
}

fn maybe_probe_ocr_text(app_name: &str) -> Option<String> {
    let ocr_image = monitor::capture_focused_window().ok()?;
    let ocr_image = if let Some((l, t, r, b)) = ocr_crop_ratios(app_name) {
        crop_for_ocr(ocr_image, l, t, r, b)
    } else {
        ocr_image
    };
    let ocr_result = run_ocr_with_timing(&ocr_image, Some(app_name), "probe").ok()?;
    let text = ocr_result.text.trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// Per-app crop ratios for OCR. Crops away UI chrome (sidebars, toolbars) so OCR
/// focuses on the main content area. Values are fractions of the window dimension.
/// (left_crop, top_crop, right_crop, bottom_crop)
fn ocr_crop_ratios(app_name: &str) -> Option<(f64, f64, f64, f64)> {
    match app_name {
        // Royal TSX: crop the left sidebar (~20% of window width)
        "Royal TSX" => Some((0.20, 0.0, 0.0, 0.0)),
        _ => None,
    }
}

/// Crop an image according to the given ratios (fractions of width/height to remove).
fn crop_for_ocr(
    image: image::DynamicImage,
    left: f64,
    top: f64,
    right: f64,
    bottom: f64,
) -> image::DynamicImage {
    let w = image.width();
    let h = image.height();
    let x = (w as f64 * left) as u32;
    let y = (h as f64 * top) as u32;
    let cw = w
        .saturating_sub(x)
        .saturating_sub((w as f64 * right) as u32);
    let ch = h
        .saturating_sub(y)
        .saturating_sub((h as f64 * bottom) as u32);
    if cw == 0 || ch == 0 {
        return image;
    }
    image.crop_imm(x, y, cw, ch)
}

/// Apps where we diff against previous capture to avoid storing stale conversation history.
const DIFF_CAPTURE_APPS: &[&str] = &["Microsoft Teams", "Messages", "Ghostty", "Terminal", "Firefox"];

fn is_diff_capture_app(app_name: &str) -> bool {
    DIFF_CAPTURE_APPS.iter().any(|&a| a == app_name)
}

/// Extract only the new lines from `current` that don't appear in `previous`.
/// Uses a set of lines from the previous capture to filter.
fn diff_text(previous: &str, current: &str) -> String {
    let prev_lines: HashSet<&str> = previous.lines().collect();
    let new_lines: Vec<&str> = current
        .lines()
        .filter(|line| !line.trim().is_empty() && !prev_lines.contains(line))
        .collect();
    new_lines.join("\n")
}

/// Configuration for the capture orchestrator.
pub struct CaptureConfig {
    pub frame_compare: FrameCompareConfig,
    pub event_config: EventConfig,
    /// Force capture even if frame looks unchanged after this many seconds (safety valve).
    pub force_capture_secs: u64,
    pub save_screenshots: bool,
    pub screenshot_dir: Option<std::path::PathBuf>,
    /// Optional path to runtime settings JSON (GUI config) to allow live OCR screenshot toggles.
    pub runtime_settings_path: Option<std::path::PathBuf>,
    /// Poll interval for runtime settings reloading.
    pub runtime_settings_poll_secs: u64,
    /// Additional apps to always OCR (user-configured and auto-promoted).
    pub force_ocr_apps: Vec<String>,
    /// Optional path for persisting force-OCR apps.
    pub force_ocr_apps_path: Option<std::path::PathBuf>,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            frame_compare: FrameCompareConfig::default(),
            event_config: EventConfig::default(),
            force_capture_secs: 10,
            save_screenshots: false,
            screenshot_dir: None,
            runtime_settings_path: None,
            runtime_settings_poll_secs: 2,
            force_ocr_apps: Vec::new(),
            force_ocr_apps_path: None,
        }
    }
}

/// Internal options passed to do_capture (avoids partial move of CaptureConfig).
struct CaptureOpts {
    save_screenshots: bool,
    screenshot_dir: Option<std::path::PathBuf>,
}

#[derive(serde::Deserialize)]
struct RuntimeSettingsFile {
    #[serde(default)]
    retain_ocr_screenshots: bool,
    #[serde(default)]
    ocr_screenshot_dir: Option<String>,
}

fn expand_tilde(path: &str) -> std::path::PathBuf {
    let trimmed = path.trim();
    if trimmed == "~" {
        if let Some(home) = std::env::var_os("HOME") {
            return std::path::PathBuf::from(home);
        }
        return std::path::PathBuf::from(trimmed);
    }
    if let Some(rest) = trimmed.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return std::path::PathBuf::from(home).join(rest);
        }
    }
    std::path::PathBuf::from(trimmed)
}

fn maybe_reload_runtime_capture_opts(
    opts: &mut CaptureOpts,
    base_save_screenshots: bool,
    base_screenshot_dir: Option<&std::path::PathBuf>,
    runtime_settings_path: Option<&std::path::Path>,
    last_runtime_mtime: &mut Option<SystemTime>,
) {
    let Some(path) = runtime_settings_path else {
        return;
    };

    let metadata = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(_) => return,
    };
    let modified = metadata.modified().ok();
    if modified.is_some() && *last_runtime_mtime == modified {
        return;
    }

    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(e) => {
            debug!("Failed reading runtime settings {}: {e}", path.display());
            return;
        }
    };
    let parsed: RuntimeSettingsFile = match serde_json::from_str(&data) {
        Ok(v) => v,
        Err(e) => {
            debug!("Failed parsing runtime settings {}: {e}", path.display());
            return;
        }
    };

    *last_runtime_mtime = modified;

    let dynamic_dir = parsed
        .ocr_screenshot_dir
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(expand_tilde);

    let old_save = opts.save_screenshots;
    let old_dir = opts.screenshot_dir.clone();

    opts.save_screenshots = base_save_screenshots || parsed.retain_ocr_screenshots;
    opts.screenshot_dir = dynamic_dir.or_else(|| base_screenshot_dir.cloned());

    if opts.save_screenshots {
        if let Some(ref dir) = opts.screenshot_dir {
            if let Err(e) = std::fs::create_dir_all(dir) {
                warn!("Failed to create OCR screenshot dir {}: {e}", dir.display());
            }
        }
    }

    if old_save != opts.save_screenshots || old_dir != opts.screenshot_dir {
        info!(
            "Runtime OCR screenshot setting updated: retain={}, dir={}",
            opts.save_screenshots,
            opts.screenshot_dir
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "<default>".to_string())
        );
    }
}

/// Run the capture loop. This is the main entry point for screen capture.
///
/// Capture is triggered in two ways:
/// 1. **Event-driven**: CGEvent tap detects clicks, typing pauses, scroll stops → immediate capture.
/// 2. **Periodic baseline**: Every N seconds (default 2s), check if the screen changed via frame diff.
///
/// Both paths run frame comparison to skip unchanged screens. The force-capture safety valve
/// (default 10s) ensures we never go too long without capturing, even if the diff is too aggressive.
pub async fn run_capture_loop(config: CaptureConfig, db: Arc<Database>) -> Result<()> {
    info!("Starting capture loop");
    info!(
        "Periodic check: every {}ms, force capture: every {}s",
        config.event_config.periodic_check_ms, config.force_capture_secs
    );

    // Extract fields before moving event_config
    let frame_compare_config = config.frame_compare;
    let force_capture_secs = config.force_capture_secs;
    let save_screenshots = config.save_screenshots;
    let screenshot_dir = config.screenshot_dir;
    let runtime_settings_path = config.runtime_settings_path;
    let runtime_settings_poll_secs = config.runtime_settings_poll_secs.max(1);
    let force_ocr_apps_path = config.force_ocr_apps_path;

    let mut frame_comparer = FrameComparer::new(frame_compare_config);
    let mut last_capture = Instant::now();
    let force_interval = Duration::from_secs(force_capture_secs);

    // Track which app the user last typed in (for passive content filtering)
    let mut last_typing_app: Option<String> = None;
    let mut last_typing_time = Instant::now() - Duration::from_secs(999);

    // Active window tracking state
    let mut current_focus: Option<(String, Option<String>)> = None;
    let focus_idle_timeout_ms: u64 = 90_000; // 90 seconds

    // Last captured text per app+window for diff-based dedup (Teams, Messages)
    let mut last_text_by_app: HashMap<(String, Option<String>), String> = HashMap::new();
    // Tracks repeated accessibility char-count patterns for adaptive OCR promotion.
    let mut repeated_accessibility: HashMap<String, RepeatLengthState> = HashMap::new();

    // Build the initial force-OCR set from configured list + persisted list.
    let mut force_ocr_apps: HashSet<String> = HashSet::new();
    let mut loaded_from_persisted_file = false;
    for app in &config.force_ocr_apps {
        let _ = force_ocr_insert(&mut force_ocr_apps, app);
    }
    if let Some(ref path) = force_ocr_apps_path {
        loaded_from_persisted_file = path.exists();
        for app in load_force_ocr_apps(path) {
            let _ = force_ocr_insert(&mut force_ocr_apps, &app);
        }
    }
    if force_ocr_apps.is_empty() && !loaded_from_persisted_file {
        for app in OCR_PREFERRED_APPS {
            let _ = force_ocr_insert(&mut force_ocr_apps, app);
        }
    }

    let mut capture_opts = CaptureOpts {
        save_screenshots,
        screenshot_dir: screenshot_dir.clone(),
    };
    let runtime_settings_poll_interval = Duration::from_secs(runtime_settings_poll_secs);
    let mut last_runtime_settings_poll = Instant::now()
        .checked_sub(runtime_settings_poll_interval)
        .unwrap_or_else(Instant::now);
    let mut last_runtime_settings_mtime: Option<SystemTime> = None;

    // Start event listener (CGEvent tap + periodic timer)
    let (event_rx, last_activity_ms, keystroke_buffer, keystroke_times, keystroke_chars) =
        crate::events::start_event_listener(config.event_config)?;

    // Main loop: wait for events
    loop {
        if last_runtime_settings_poll.elapsed() >= runtime_settings_poll_interval {
            maybe_reload_runtime_capture_opts(
                &mut capture_opts,
                save_screenshots,
                screenshot_dir.as_ref(),
                runtime_settings_path.as_deref(),
                &mut last_runtime_settings_mtime,
            );
            last_runtime_settings_poll = Instant::now();
        }

        let event = match event_rx.recv_timeout(Duration::from_secs(1)) {
            Ok(event) => event,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // No event, but still track active window if user is present
                let now_epoch_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64;
                let last_act = last_activity_ms.load(Ordering::Acquire);
                let activity_age_ms = now_epoch_ms.saturating_sub(last_act);
                let is_present = activity_age_ms < focus_idle_timeout_ms;

                if is_present {
                    if let Ok(Some(app)) = accessibility::get_focused_app_name() {
                        let window = accessibility::get_focused_window_title();
                        let focus = (app.clone(), window.clone());
                        if current_focus.as_ref() != Some(&focus) {
                            current_focus = Some(focus);
                        }
                        let _ = db
                            .upsert_active_window(
                                &db.machine_id,
                                &app,
                                window.as_deref(),
                                now_epoch_ms as i64,
                            )
                            .await;
                    }
                } else if current_focus.is_some() {
                    let _ = db
                        .finalize_active_window(&db.machine_id, last_act as i64)
                        .await;
                    current_focus = None;
                }
                continue;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                warn!("Event listener disconnected");
                break;
            }
        };

        let now = Instant::now();
        let force = now.duration_since(last_capture) >= force_interval;

        // Check if user is present (any input including mouse movement within 30s)
        let now_epoch_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let last_act = last_activity_ms.load(Ordering::Acquire);
        let activity_age_ms = now_epoch_ms.saturating_sub(last_act);
        let is_present = activity_age_ms < focus_idle_timeout_ms;

        // Track active window
        if is_present {
            if let Ok(Some(app)) = accessibility::get_focused_app_name() {
                let window = accessibility::get_focused_window_title();
                let focus = (app.clone(), window.clone());
                if current_focus.as_ref() != Some(&focus) {
                    current_focus = Some(focus);
                }
                let _ = db
                    .upsert_active_window(
                        &db.machine_id,
                        &app,
                        window.as_deref(),
                        now_epoch_ms as i64,
                    )
                    .await;
            }
        } else if current_focus.is_some() {
            let _ = db
                .finalize_active_window(&db.machine_id, last_act as i64)
                .await;
            current_focus = None;
        }

        match event {
            CaptureEvent::Keystroke => {
                // Record which app received keyboard input (fires immediately on KeyDown).
                if let Ok(Some(app)) = accessibility::get_focused_app_name() {
                    let should_capture = is_passive_content_app(&app);
                    last_typing_app = Some(app);
                    last_typing_time = now;
                    if should_capture {
                        do_capture(
                            &db,
                            &mut frame_comparer,
                            true,
                            &capture_opts,
                            last_typing_app.as_deref(),
                            last_typing_time,
                            &keystroke_buffer,
                            false,
                            &mut last_text_by_app,
                            &mut force_ocr_apps,
                            &mut repeated_accessibility,
                            force_ocr_apps_path.as_deref(),
                        )
                        .await;
                        last_capture = now;
                    }
                }
                continue;
            }
            CaptureEvent::TypingPause => {
                debug!("Capture triggered by TypingPause");
                // Drain keystroke timestamps + chars and store typing burst
                let times: Vec<i64> = std::mem::take(&mut *keystroke_times.lock().unwrap());
                let chars: Vec<char> = std::mem::take(&mut *keystroke_chars.lock().unwrap());
                if times.len() >= 2 {
                    let app = last_typing_app.clone();
                    let burst_db = db.clone();
                    let typed: String = chars.into_iter().collect();
                    tokio::spawn(async move {
                        if let Err(e) = burst_db.insert_typing_burst(&times, app.as_deref(), &typed).await {
                            tracing::warn!("Failed to insert typing burst: {e}");
                        }
                    });
                } else if !times.is_empty() {
                    // Single keystroke — not enough for speed calc, just clear
                    debug!("Single keystroke, skipping typing burst");
                }
                if do_capture(
                    &db,
                    &mut frame_comparer,
                    force,
                    &capture_opts,
                    last_typing_app.as_deref(),
                    last_typing_time,
                    &keystroke_buffer,
                    true,
                    &mut last_text_by_app,
                    &mut force_ocr_apps,
                    &mut repeated_accessibility,
                    force_ocr_apps_path.as_deref(),
                )
                .await
                {
                    last_capture = now;
                }
            }
            CaptureEvent::Click { x, y } => {
                debug!("Capture triggered by Click at ({x}, {y})");
                // Record what was clicked via accessibility hit-test
                let click_db = db.clone();
                let click_x = x;
                let click_y = y;
                tokio::spawn(async move {
                    if let Some(target) = crate::accessibility::get_click_target(click_x, click_y) {
                        let ts = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_millis() as i64;
                        if let Err(e) = click_db
                            .insert_click_event(
                                ts,
                                click_x,
                                click_y,
                                Some(&target.app_name),
                                target.role.as_deref(),
                                target.title.as_deref(),
                                target.description.as_deref(),
                                target.value.as_deref(),
                            )
                            .await
                        {
                            tracing::warn!("Failed to insert click event: {e}");
                        }
                    }
                });
                if do_capture(
                    &db,
                    &mut frame_comparer,
                    force,
                    &capture_opts,
                    last_typing_app.as_deref(),
                    last_typing_time,
                    &keystroke_buffer,
                    true,
                    &mut last_text_by_app,
                    &mut force_ocr_apps,
                    &mut repeated_accessibility,
                    force_ocr_apps_path.as_deref(),
                )
                .await
                {
                    last_capture = now;
                }
            }
            CaptureEvent::ScrollStop | CaptureEvent::AppSwitch { .. } => {
                debug!("Capture triggered by {event:?}");
                if do_capture(
                    &db,
                    &mut frame_comparer,
                    force,
                    &capture_opts,
                    last_typing_app.as_deref(),
                    last_typing_time,
                    &keystroke_buffer,
                    true,
                    &mut last_text_by_app,
                    &mut force_ocr_apps,
                    &mut repeated_accessibility,
                    force_ocr_apps_path.as_deref(),
                )
                .await
                {
                    last_capture = now;
                }
            }
            CaptureEvent::Periodic => {
                // Skip captures when user is idle — no point capturing an unattended screen
                if !is_present {
                    continue;
                }
                if do_capture(
                    &db,
                    &mut frame_comparer,
                    force,
                    &capture_opts,
                    last_typing_app.as_deref(),
                    last_typing_time,
                    &keystroke_buffer,
                    false,
                    &mut last_text_by_app,
                    &mut force_ocr_apps,
                    &mut repeated_accessibility,
                    force_ocr_apps_path.as_deref(),
                )
                .await
                {
                    last_capture = now;
                }
            }
            CaptureEvent::Idle => {
                // One final capture when going idle, then stop until activity resumes
                debug!("Idle — final capture");
                do_capture(
                    &db,
                    &mut frame_comparer,
                    true,
                    &capture_opts,
                    last_typing_app.as_deref(),
                    last_typing_time,
                    &keystroke_buffer,
                    true,
                    &mut last_text_by_app,
                    &mut force_ocr_apps,
                    &mut repeated_accessibility,
                    force_ocr_apps_path.as_deref(),
                )
                .await;
                last_capture = now;
                // Keep diff cache across idle so reopening a conversation doesn't
                // re-capture old message history as new content.
                repeated_accessibility.clear();
            }
        }
    }

    Ok(())
}

/// Perform a single capture cycle. Returns true if a frame was actually captured.
async fn do_capture(
    db: &Database,
    frame_comparer: &mut FrameComparer,
    force: bool,
    opts: &CaptureOpts,
    last_typing_app: Option<&str>,
    last_typing_time: Instant,
    keystroke_buffer: &std::sync::Mutex<String>,
    drain_keystrokes: bool,
    last_text_by_app: &mut HashMap<(String, Option<String>), String>,
    force_ocr_apps: &mut HashSet<String>,
    repeated_accessibility: &mut HashMap<String, RepeatLengthState>,
    force_ocr_apps_path: Option<&std::path::Path>,
) -> bool {
    // Try to capture the focused window (needed for change detection and OCR fallback).
    // This can fail if screen recording permission isn't granted or no focused window is
    // available — that's OK, accessibility captures still work without it.
    let image = monitor::capture_focused_window().ok();

    // Check if frame changed (skip if no screenshot available)
    if let Some(ref img) = image {
        if !force && !frame_comparer.has_changed(img) {
            if let Err(e) = db.increment_stat("frames_skipped_unchanged").await {
                debug!("Failed to increment skip stat: {e}");
            }
            return false;
        }
    }

    // Try accessibility tree first
    let need_ocr = match accessibility::capture_accessibility_text() {
        Ok(Some(mut result)) => {
            // Accessibility succeeded — drain keystroke buffer so it doesn't accumulate
            keystroke_buffer.lock().unwrap().clear();
            // Skip login screen — treat as away from PC
            if result.app_name == "loginwindow" {
                debug!("Login screen detected, skipping (user away)");
                return false;
            }

            // Skip windows that just duplicate data already in the DB
            if let Some(ref title) = result.window_title {
                if title == "ScreenTrack Debug Log" {
                    debug!("Skipping ScreenTrack Debug Log window");
                    return false;
                }
            }

            // Apps with graphical content (or auto-promoted apps) use OCR.
            if force_ocr_contains(force_ocr_apps, &result.app_name) {
                debug!(
                    "OCR-preferred app '{}', skipping accessibility",
                    result.app_name
                );
                true
            } else {
                // For passive content apps (Messages, Slack, etc.), only capture full text
                // if the user recently typed in this app. Otherwise just note they viewed it.
                if is_passive_content_app(&result.app_name) {
                    let recently_typed = last_typing_app == Some(result.app_name.as_str())
                        && last_typing_time.elapsed()
                            < Duration::from_secs(PASSIVE_INPUT_TIMEOUT_SECS);
                    if !recently_typed {
                        debug!(
                            "Passive content app '{}' without recent typing, capturing minimal",
                            result.app_name
                        );
                        result.full_text = format!(
                            "[Viewing conversation with {}]",
                            result.window_title.as_deref().unwrap_or("unknown")
                        );
                        result.elements.clear();
                    }
                }

                // Diff capture: apps that accumulate history (chat apps, terminals).
                // Only store new lines compared to the previous capture. On first
                // capture, seed the cache without dumping stale history.
                if is_diff_capture_app(&result.app_name)
                    && !result.full_text.starts_with('[')
                {
                    let key = (result.app_name.clone(), result.window_title.clone());
                    let has_prev = last_text_by_app.contains_key(&key);
                    if !has_prev {
                        debug!(
                            "Diff capture app '{}': first capture of '{}', seeding ({} chars)",
                            result.app_name,
                            result.window_title.as_deref().unwrap_or("?"),
                            result.full_text.len(),
                        );
                        last_text_by_app.insert(key, result.full_text.clone());
                        result.full_text = format!(
                            "[Using {}]",
                            result.window_title.as_deref().unwrap_or(&result.app_name)
                        );
                        result.elements.clear();
                    } else {
                        let new_text = diff_text(
                            last_text_by_app.get(&key).unwrap(),
                            &result.full_text,
                        );
                        if new_text.is_empty() {
                            debug!(
                                "Diff capture app '{}' has no new content, skipping",
                                result.app_name
                            );
                            return false;
                        }
                        debug!(
                            "Diff capture app '{}': {} new chars (was {} total)",
                            result.app_name,
                            new_text.len(),
                            result.full_text.len(),
                        );
                        last_text_by_app.insert(key, result.full_text.clone());
                        result.full_text = new_text;
                    }
                }

                // Detect suspiciously repetitive accessibility payload lengths; if OCR output is
                // substantially different, auto-promote this app to force-OCR.
                let streak = update_repeat_length_state(
                    repeated_accessibility,
                    &result.app_name,
                    result.full_text.len(),
                );
                if streak >= ADAPTIVE_OCR_REPEAT_STREAK {
                    if let Some(ocr_probe_text) = maybe_probe_ocr_text(&result.app_name) {
                        if is_substantially_different(&result.full_text, &ocr_probe_text) {
                            // Log the detection but do NOT auto-promote — only
                            // the hardcoded OCR_PREFERRED_APPS list and manually
                            // configured apps should use OCR.
                            if !force_ocr_contains(force_ocr_apps, &result.app_name) {
                                info!(
                                    frame_event = "ocr_candidate",
                                    source = "ocr_adaptive",
                                    app = %result.app_name,
                                    chars = ocr_probe_text.len(),
                                    "App {} detected as OCR candidate (not auto-promoting)",
                                    result.app_name,
                                );
                            }
                            repeated_accessibility.remove(&normalize_app_name(&result.app_name));
                            return true;
                        }
                    }
                    // Avoid continuously probing every frame when not different enough.
                    repeated_accessibility.remove(&normalize_app_name(&result.app_name));
                }

                debug!(
                    "Got accessibility text from {} ({} chars)",
                    result.app_name,
                    result.full_text.len()
                );
                info!(
                    frame_event = "captured",
                    source = "accessibility",
                    app = %result.app_name,
                    chars = result.full_text.len(),
                    "Frame: accessibility @ {}",
                    result.app_name,
                );
                let frame = NewFrame {
                    app_name: Some(result.app_name),
                    window_title: result.window_title,
                    browser_tab: result.browser_tab,
                    text_content: result.full_text,
                    source: "accessibility".into(),
                    screenshot_path: None,
                };
                if let Err(e) = db.insert_frame(frame).await {
                    warn!("Failed to insert frame: {e}");
                }
                false
            }
        }
        Ok(None) => {
            debug!("No accessibility text, checking keystroke buffer");
            true
        }
        Err(e) => {
            debug!("Accessibility error: {e}, checking keystroke buffer");
            true
        }
    };

    if !need_ocr {
        return true;
    }

    // Drain the keystroke buffer — only when user has stopped typing
    let typed_text = if drain_keystrokes {
        let mut buf = keystroke_buffer.lock().unwrap();
        std::mem::take(&mut *buf)
    } else {
        String::new()
    };

    // Try to get the focused app name
    let app_name = accessibility::get_focused_app_name().ok().flatten();
    let window_title = accessibility::get_focused_window_title();

    if !typed_text.is_empty() {
        // Use keystroke buffer as content — more accurate than OCR for typed text
        debug!(
            "Using keystroke buffer ({} chars) for {}",
            typed_text.len(),
            app_name.as_deref().unwrap_or("unknown")
        );
        info!(
            frame_event = "captured",
            source = "keystrokes",
            app = app_name.as_deref().unwrap_or("unknown"),
            chars = typed_text.len(),
            "Frame: keystrokes @ {}",
            app_name.as_deref().unwrap_or("unknown"),
        );
        let frame = NewFrame {
            app_name,
            window_title,
            browser_tab: None,
            text_content: format!("[Typed text] {typed_text}"),
            source: "keystrokes".into(),
            screenshot_path: None,
        };
        if let Err(e) = db.insert_frame(frame).await {
            warn!("Failed to insert frame: {e}");
        }
        return true;
    }

    // Fallback: Apple Vision OCR — capture focused window for cleaner results.
    // If screenshot capture isn't available (no screen recording permission), skip OCR.
    let ocr_image = match monitor::capture_focused_window().ok().or(image) {
        Some(img) => img,
        None => {
            debug!("No screenshot available for OCR (screen recording not granted?)");
            return false;
        }
    };

    // Apply per-app cropping to remove sidebars/chrome before OCR
    let ocr_image = if let Some(ref name) = app_name {
        if let Some((l, t, r, b)) = ocr_crop_ratios(name) {
            crop_for_ocr(ocr_image, l, t, r, b)
        } else {
            ocr_image
        }
    } else {
        ocr_image
    };

    match run_ocr_with_timing(&ocr_image, app_name.as_deref(), "capture") {
        Ok(result) if !result.text.is_empty() => {
            info!(
                frame_event = "captured",
                source = "ocr_local",
                app = app_name.as_deref().unwrap_or("unknown"),
                chars = result.text.len(),
                "Frame: ocr @ {}",
                app_name.as_deref().unwrap_or("unknown"),
            );
            let screenshot_path = if opts.save_screenshots {
                save_screenshot(&ocr_image, opts.screenshot_dir.as_deref())
            } else {
                None
            };

            let frame = NewFrame {
                app_name,
                window_title,
                browser_tab: None,
                text_content: result.text,
                source: "ocr_local".into(),
                screenshot_path,
            };
            if let Err(e) = db.insert_frame(frame).await {
                warn!("Failed to insert frame: {e}");
            }
            true
        }
        Ok(_) => {
            debug!("OCR returned no text");
            false
        }
        Err(e) => {
            warn!("OCR failed: {e}");
            false
        }
    }
}

fn save_screenshot(image: &image::DynamicImage, dir: Option<&std::path::Path>) -> Option<String> {
    let dir = dir.unwrap_or_else(|| std::path::Path::new("/tmp/screentrack"));
    std::fs::create_dir_all(dir).ok()?;
    let filename = format!(
        "capture_{}.jpg",
        chrono::Utc::now().format("%Y%m%d_%H%M%S_%3f")
    );
    let path = dir.join(&filename);
    image.save(&path).ok()?;
    Some(path.to_string_lossy().into_owned())
}

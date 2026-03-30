use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use chrono::{Local, TimeZone, Utc};
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2::{define_class, msg_send, AllocAnyThread, DefinedClass, MainThreadMarker};
use objc2_app_kit::{
    NSAlert, NSAlertFirstButtonReturn, NSApplication, NSApplicationActivationPolicy,
    NSAutoresizingMaskOptions, NSBackingStoreType, NSButton, NSColor, NSControlStateValueOff,
    NSControlStateValueOn, NSFont, NSMenu, NSMenuItem, NSPopUpButton, NSScrollView, NSStatusBar,
    NSStatusItem, NSTextField, NSTextView, NSView, NSWindow, NSWindowStyleMask,
};
use objc2_foundation::{
    ns_string, NSMutableAttributedString, NSObject, NSObjectProtocol, NSPoint, NSRect, NSRunLoop,
    NSRunLoopCommonModes, NSSize, NSString, NSTimer,
};
use serde::{Deserialize, Serialize};

use screentrack_store::Database;
use screentrack_summarizer::client::{LlmClient, LlmConfig};

use crate::viewer::ViewerWindow;

// ---------------------------------------------------------------------------
// Persistent settings
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedSettings {
    pub db_path: String,
    pub llm_url: String,
    pub model: String,
    pub llm_api_key: Option<String>,
    /// "serve", "push", "local", or null (don't auto-start)
    pub mode: Option<String>,
    /// For serve mode
    pub listen: Option<String>,
    /// For push mode
    pub server_url: Option<String>,
    /// Retain OCR screenshots for debugging
    #[serde(default)]
    pub retain_ocr_screenshots: bool,
    /// Optional custom directory for OCR screenshots
    #[serde(default)]
    pub ocr_screenshot_dir: Option<String>,
}

impl SavedSettings {
    fn path() -> std::path::PathBuf {
        dirs_next::home_dir()
            .unwrap_or_else(|| ".".into())
            .join(".screentrack")
            .join("gui_config.json")
    }

    pub fn load() -> Option<Self> {
        let data = std::fs::read_to_string(Self::path()).ok()?;
        serde_json::from_str(&data).ok()
    }

    fn save(&self) {
        if let Some(parent) = Self::path().parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(Self::path(), json);
        }
    }
}

// ---------------------------------------------------------------------------
// Debug log ring buffer
// ---------------------------------------------------------------------------

/// Shared log buffer that captures tracing output for display in the GUI.
pub static DEBUG_LOG: std::sync::LazyLock<Mutex<DebugLog>> =
    std::sync::LazyLock::new(|| Mutex::new(DebugLog::new(500)));
static DEBUG_LOG_VERSION: AtomicU64 = AtomicU64::new(0);

pub struct DebugLog {
    lines: std::collections::VecDeque<String>,
    capacity: usize,
}

impl DebugLog {
    fn new(capacity: usize) -> Self {
        Self {
            lines: std::collections::VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Push a frame event line.
    /// `source_key` is like "accessibility", "ocr_local", "keystrokes", "skip_unchanged".
    /// `app` is the app name. `timestamp` is the formatted time prefix.
    pub fn push_frame_event(&mut self, source_key: &str, app: &str, chars: usize, timestamp: &str) {
        let line = format!("{timestamp} FRAME [{source_key}] {app} ({chars} chars)");
        if self.lines.len() >= self.capacity {
            self.lines.pop_front();
        }
        self.lines.push_back(line);
        DEBUG_LOG_VERSION.fetch_add(1, Ordering::AcqRel);
    }

    pub fn push(&mut self, line: String) {
        if self.lines.len() >= self.capacity {
            self.lines.pop_front();
        }
        self.lines.push_back(line);
        DEBUG_LOG_VERSION.fetch_add(1, Ordering::AcqRel);
    }
}

/// A tracing layer that writes formatted log lines into the DEBUG_LOG buffer.
pub struct GuiLogLayer;

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for GuiLogLayer {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let now = chrono::Local::now().format("%H:%M:%S%.3f").to_string();

        // Check if this is a frame event by extracting structured fields
        struct FrameVisitor {
            frame_event: Option<String>,
            source: Option<String>,
            app: Option<String>,
            chars: Option<usize>,
            message: String,
        }
        impl tracing::field::Visit for FrameVisitor {
            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                match field.name() {
                    "frame_event" => self.frame_event = Some(value.to_string()),
                    "source" => self.source = Some(value.to_string()),
                    "app" => self.app = Some(value.to_string()),
                    "message" => self.message = value.to_string(),
                    _ => {}
                }
            }
            fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
                if field.name() == "chars" {
                    self.chars = Some(value as usize);
                }
            }
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                let mut debug_value = format!("{:?}", value);
                if debug_value.starts_with('"')
                    && debug_value.ends_with('"')
                    && debug_value.len() >= 2
                {
                    debug_value = debug_value[1..debug_value.len() - 1].to_string();
                }
                match field.name() {
                    // Some tracing field encodings (including `%...`) can arrive via `record_debug`.
                    "frame_event" => self.frame_event = Some(debug_value),
                    "source" => self.source = Some(debug_value),
                    "app" => self.app = Some(debug_value),
                    "chars" => {
                        if let Ok(v) = debug_value.parse::<usize>() {
                            self.chars = Some(v);
                        }
                    }
                    "message" => self.message = debug_value,
                    _ => {}
                }
            }
        }

        let mut visitor = FrameVisitor {
            frame_event: None,
            source: None,
            app: None,
            chars: None,
            message: String::new(),
        };
        event.record(&mut visitor);

        if let Ok(mut log) = DEBUG_LOG.lock() {
            if visitor.frame_event.is_some() {
                let source = visitor
                    .source
                    .as_deref()
                    .or(visitor.frame_event.as_deref())
                    .unwrap_or("unknown");
                let app = visitor.app.as_deref().unwrap_or("unknown");
                let chars = visitor.chars.unwrap_or(0);
                log.push_frame_event(source, app, chars, &now);
            } else {
                use std::fmt::Write;
                let meta = event.metadata();
                let mut msg = String::new();
                let _ = write!(msg, "{now} {:<5} ", meta.level());
                if let Some(module) = meta.module_path() {
                    let short = module.rsplit("::").next().unwrap_or(module);
                    let _ = write!(msg, "[{short}] ");
                }
                let _ = write!(msg, "{}", visitor.message);
                log.push(msg);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// App state types
// ---------------------------------------------------------------------------

struct DaemonState {
    runtime: Option<tokio::runtime::Runtime>,
    db: Option<Arc<Database>>,
    mode: Option<String>,
    running: Arc<AtomicBool>,
}

#[derive(Default, Clone)]
struct DisplayStats {
    frames_total: i64,
    frames_accessibility: i64,
    frames_ocr: i64,
    frames_keystrokes: i64,
    micro: i64,
    hourly: i64,
    daily: i64,
    weekly: i64,
}

struct StatusItems {
    frames: Retained<NSMenuItem>,
    accessibility: Retained<NSMenuItem>,
    ocr: Retained<NSMenuItem>,
    keystrokes: Retained<NSMenuItem>,
    micro: Retained<NSMenuItem>,
    hourly: Retained<NSMenuItem>,
    daily: Retained<NSMenuItem>,
    weekly: Retained<NSMenuItem>,
    mode_item: Retained<NSMenuItem>,
    stop_item: Retained<NSMenuItem>,
    serve_item: Retained<NSMenuItem>,
    push_item: Retained<NSMenuItem>,
    local_item: Retained<NSMenuItem>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DebugViewMode {
    LiveLog,
    RawFrames,
    MicroSummaries,
    HourlySummaries,
    DailySummaries,
    WeeklySummaries,
}

impl DebugViewMode {
    fn from_index(idx: isize) -> Self {
        match idx {
            0 => Self::LiveLog,
            1 => Self::RawFrames,
            2 => Self::MicroSummaries,
            3 => Self::HourlySummaries,
            4 => Self::DailySummaries,
            5 => Self::WeeklySummaries,
            _ => Self::LiveLog,
        }
    }

    fn tier(self) -> Option<&'static str> {
        match self {
            Self::MicroSummaries => Some("micro"),
            Self::HourlySummaries => Some("hourly"),
            Self::DailySummaries => Some("daily"),
            Self::WeeklySummaries => Some("weekly"),
            _ => None,
        }
    }

    fn summary_limit(self) -> i64 {
        match self {
            Self::MicroSummaries => 60,
            Self::HourlySummaries => 24,
            Self::DailySummaries => 14,
            Self::WeeklySummaries => 8,
            _ => 0,
        }
    }
}

struct DebugLogWindow {
    window: Retained<NSWindow>,
    text_view: Retained<NSTextView>,
    pause_button: Retained<NSButton>,
    mode_popup: Retained<NSPopUpButton>,
    refresh_button: Retained<NSButton>,
    _scroll_view: Retained<NSScrollView>,
    mode: Cell<DebugViewMode>,
    auto_follow: Cell<bool>,
    last_rendered_version: Cell<u64>,
}

#[derive(Clone)]
struct AppConfig {
    db_path: String,
    llm_url: String,
    model: String,
    llm_api_key: Option<String>,
    listen: String,
    server_url: String,
    retain_ocr_screenshots: bool,
    ocr_screenshot_dir: String,
}

struct AppDelegateIvars {
    daemon: RefCell<DaemonState>,
    stats: RefCell<DisplayStats>,
    status_items: RefCell<Option<StatusItems>>,
    config: RefCell<AppConfig>,
    viewer: RefCell<Option<ViewerWindow>>,
    query_answer: Arc<Mutex<Option<String>>>,
    query_pending: Arc<AtomicBool>,
    viewer_db: RefCell<Option<Arc<Database>>>,
    viewer_runtime: RefCell<Option<tokio::runtime::Runtime>>,
    debug_window: RefCell<Option<DebugLogWindow>>,
}

// ---------------------------------------------------------------------------
// ObjC class definition
// ---------------------------------------------------------------------------

define_class!(
    #[unsafe(super(NSObject))]
    #[name = "STAppDelegate"]
    #[ivars = AppDelegateIvars]
    struct AppDelegate;

    impl AppDelegate {
        #[unsafe(method(startServe:))]
        fn _start_serve(&self, _sender: *mut NSObject) { self.show_serve_dialog(); }

        #[unsafe(method(startPush:))]
        fn _start_push(&self, _sender: *mut NSObject) { self.show_push_dialog(); }

        #[unsafe(method(startLocal:))]
        fn _start_local(&self, _sender: *mut NSObject) {
            self.start_daemon("local", None, false);
            self.save_settings();
        }

        #[unsafe(method(stopDaemon:))]
        fn _stop_daemon_action(&self, _sender: *mut NSObject) {
            self.stop_daemon();
            // Clear saved auto-start mode
            self.save_settings_with_mode(None);
        }

        #[unsafe(method(quit:))]
        fn _quit(&self, _sender: *mut NSObject) {
            self.stop_daemon();
            let app = NSApplication::sharedApplication(MainThreadMarker::new().unwrap());
            unsafe { app.terminate(None) };
        }

        #[unsafe(method(updateStatus:))]
        fn _update_status(&self, _timer: *mut NSObject) {
            self.refresh_stats();
        }

        #[unsafe(method(refreshDebugLog:))]
        fn _refresh_debug_log(&self, _timer: *mut NSObject) {
            self.poll_smart_query_answer();
            self.refresh_debug_log();
        }

        #[unsafe(method(openViewer:))]
        fn _open_viewer(&self, _sender: *mut NSObject) { self.open_viewer(); }

        #[unsafe(method(segmentChanged:))]
        fn _segment_changed(&self, _sender: *mut NSObject) { self.handle_segment_change(); }

        #[unsafe(method(askQuery:))]
        fn _ask_query(&self, _sender: *mut NSObject) { self.handle_ask_query(); }

        #[unsafe(method(showSettings:))]
        fn _show_settings(&self, _sender: *mut NSObject) { self.show_settings_dialog(); }

        #[unsafe(method(showDebugLog:))]
        fn _show_debug_log(&self, _sender: *mut NSObject) { self.show_debug_log(); }

        #[unsafe(method(toggleDebugAutoScroll:))]
        fn _toggle_debug_auto_scroll(&self, _sender: *mut NSObject) { self.toggle_debug_auto_scroll(); }

        #[unsafe(method(debugViewModeChanged:))]
        fn _debug_view_mode_changed(&self, _sender: *mut NSObject) { self.debug_view_mode_changed(); }

        #[unsafe(method(refreshDebugData:))]
        fn _refresh_debug_data(&self, _sender: *mut NSObject) { self.refresh_debug_data(); }

    }
);

unsafe impl Send for AppDelegate {}
unsafe impl Sync for AppDelegate {}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl AppDelegate {
    fn new(config: AppConfig) -> Retained<Self> {
        let this = Self::alloc().set_ivars(AppDelegateIvars {
            daemon: RefCell::new(DaemonState {
                runtime: None,
                db: None,
                mode: None,
                running: Arc::new(AtomicBool::new(false)),
            }),
            stats: RefCell::new(DisplayStats::default()),
            status_items: RefCell::new(None),
            config: RefCell::new(config),
            viewer: RefCell::new(None),
            query_answer: Arc::new(Mutex::new(None)),
            query_pending: Arc::new(AtomicBool::new(false)),
            viewer_db: RefCell::new(None),
            viewer_runtime: RefCell::new(None),
            debug_window: RefCell::new(None),
        });
        unsafe { msg_send![super(this), init] }
    }

    // -- Settings persistence -----------------------------------------------

    fn save_settings(&self) {
        let daemon = self.ivars().daemon.borrow();
        self.save_settings_with_mode(daemon.mode.clone());
    }

    fn save_settings_with_mode(&self, mode: Option<String>) {
        let config = self.ivars().config.borrow().clone();
        let settings = SavedSettings {
            db_path: config.db_path,
            llm_url: config.llm_url,
            model: config.model,
            llm_api_key: config.llm_api_key,
            mode,
            listen: Some(config.listen),
            server_url: Some(config.server_url),
            retain_ocr_screenshots: config.retain_ocr_screenshots,
            ocr_screenshot_dir: if config.ocr_screenshot_dir.trim().is_empty() {
                None
            } else {
                Some(config.ocr_screenshot_dir)
            },
        };
        settings.save();
    }

    // -- Settings dialog ----------------------------------------------------

    fn show_settings_dialog(&self) {
        let mtm = MainThreadMarker::new().unwrap();
        let config = self.ivars().config.borrow().clone();

        let alert = NSAlert::new(mtm);
        alert.setMessageText(&NSString::from_str("ScreenTrack Settings"));
        alert.setInformativeText(&NSString::from_str("Capture settings apply while running."));
        alert.addButtonWithTitle(&NSString::from_str("Save"));
        alert.addButtonWithTitle(&NSString::from_str("Cancel"));

        let row_h = 24.0_f64;
        let label_h = 18.0;
        let gap = 6.0;
        // 6 labeled text rows + 1 checkbox row + 1 labeled text row (OCR directory)
        let total_h = 6.0 * (row_h + label_h + gap) + (row_h + gap) + (row_h + label_h + gap) + gap;

        let view = NSView::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(400.0, total_h)),
        );

        let fields: Vec<(String, String)> = vec![
            ("Database path:".into(), config.db_path.clone()),
            ("LLM URL:".into(), config.llm_url.clone()),
            ("Model:".into(), config.model.clone()),
            (
                "API Key:".into(),
                config.llm_api_key.clone().unwrap_or_default(),
            ),
            ("Listen address (serve):".into(), config.listen.clone()),
            ("Server URL (push):".into(), config.server_url.clone()),
        ];

        let mut text_fields: Vec<Retained<NSTextField>> = Vec::new();
        let mut y = total_h;

        for (label_text, default_val) in &fields {
            y -= label_h + gap;
            let label = NSTextField::wrappingLabelWithString(&NSString::from_str(label_text), mtm);
            label.setFrame(NSRect::new(
                NSPoint::new(0.0, y),
                NSSize::new(400.0, label_h),
            ));
            view.addSubview(&label);

            y -= row_h;
            let field = NSTextField::initWithFrame(
                mtm.alloc(),
                NSRect::new(NSPoint::new(0.0, y), NSSize::new(400.0, row_h)),
            );
            field.setStringValue(&NSString::from_str(default_val));
            view.addSubview(&field);
            text_fields.push(field);
        }

        y -= row_h + gap;
        let retain_checkbox = unsafe {
            NSButton::checkboxWithTitle_target_action(
                &NSString::from_str("Retain OCR screenshots for debugging"),
                None,
                None,
                mtm,
            )
        };
        retain_checkbox.setFrame(NSRect::new(NSPoint::new(0.0, y), NSSize::new(400.0, row_h)));
        retain_checkbox.setState(if config.retain_ocr_screenshots {
            NSControlStateValueOn
        } else {
            NSControlStateValueOff
        });
        view.addSubview(&retain_checkbox);

        y -= label_h + gap;
        let ocr_dir_label = NSTextField::wrappingLabelWithString(
            &NSString::from_str("OCR screenshot directory (optional):"),
            mtm,
        );
        ocr_dir_label.setFrame(NSRect::new(
            NSPoint::new(0.0, y),
            NSSize::new(400.0, label_h),
        ));
        view.addSubview(&ocr_dir_label);

        y -= row_h;
        let ocr_dir_field = NSTextField::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::new(0.0, y), NSSize::new(400.0, row_h)),
        );
        ocr_dir_field.setStringValue(&NSString::from_str(&config.ocr_screenshot_dir));
        view.addSubview(&ocr_dir_field);
        text_fields.push(ocr_dir_field);

        alert.setAccessoryView(Some(&view));

        let response = alert.runModal();
        if response == NSAlertFirstButtonReturn {
            let mut config = self.ivars().config.borrow_mut();
            config.db_path = text_fields[0].stringValue().to_string();
            config.llm_url = text_fields[1].stringValue().to_string();
            config.model = text_fields[2].stringValue().to_string();
            let key = text_fields[3].stringValue().to_string();
            config.llm_api_key = if key.is_empty() { None } else { Some(key) };
            config.listen = text_fields[4].stringValue().to_string();
            config.server_url = text_fields[5].stringValue().to_string();
            config.retain_ocr_screenshots = retain_checkbox.state() == NSControlStateValueOn;
            config.ocr_screenshot_dir = text_fields[6].stringValue().to_string();
            drop(config);

            // Reset viewer DB so it reconnects with new path if changed
            *self.ivars().viewer_db.borrow_mut() = None;
            *self.ivars().viewer_runtime.borrow_mut() = None;

            // Save to disk
            self.save_settings();
        }
    }

    // -- Mode dialogs -------------------------------------------------------

    fn show_serve_dialog(&self) {
        let mtm = MainThreadMarker::new().unwrap();
        let config = self.ivars().config.borrow().clone();

        let alert = NSAlert::new(mtm);
        alert.setMessageText(&NSString::from_str("Start Serve Mode"));
        alert.setInformativeText(&NSString::from_str(
            "Capture locally + receive from clients + summarize",
        ));
        alert.addButtonWithTitle(&NSString::from_str("Start"));
        alert.addButtonWithTitle(&NSString::from_str("Cancel"));

        let view = NSView::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(300.0, 54.0)),
        );
        let label =
            NSTextField::wrappingLabelWithString(&NSString::from_str("Listen address:"), mtm);
        label.setFrame(NSRect::new(
            NSPoint::new(0.0, 30.0),
            NSSize::new(300.0, 20.0),
        ));
        let field = NSTextField::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(300.0, 24.0)),
        );
        field.setStringValue(&NSString::from_str(&config.listen));
        view.addSubview(&label);
        view.addSubview(&field);
        alert.setAccessoryView(Some(&view));

        let response = alert.runModal();
        if response == NSAlertFirstButtonReturn {
            let listen = field.stringValue().to_string();
            self.ivars().config.borrow_mut().listen = listen.clone();
            self.start_daemon("serve", Some(listen), false);
            self.save_settings();
        }
    }

    fn show_push_dialog(&self) {
        let mtm = MainThreadMarker::new().unwrap();
        let config = self.ivars().config.borrow().clone();

        let alert = NSAlert::new(mtm);
        alert.setMessageText(&NSString::from_str("Start Push Mode"));
        alert.setInformativeText(&NSString::from_str(
            "Capture locally + push frames to server",
        ));
        alert.addButtonWithTitle(&NSString::from_str("Start"));
        alert.addButtonWithTitle(&NSString::from_str("Cancel"));

        let view = NSView::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(300.0, 54.0)),
        );
        let label = NSTextField::wrappingLabelWithString(&NSString::from_str("Server URL:"), mtm);
        label.setFrame(NSRect::new(
            NSPoint::new(0.0, 30.0),
            NSSize::new(300.0, 20.0),
        ));
        let field = NSTextField::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(300.0, 24.0)),
        );
        field.setStringValue(&NSString::from_str(&config.server_url));
        if config.server_url.is_empty() {
            field.setPlaceholderString(Some(&NSString::from_str("http://server:7878")));
        }
        view.addSubview(&label);
        view.addSubview(&field);
        alert.setAccessoryView(Some(&view));

        let response = alert.runModal();
        if response == NSAlertFirstButtonReturn {
            let server_url = field.stringValue().to_string();
            if server_url.is_empty() {
                let err = NSAlert::new(mtm);
                err.setMessageText(&NSString::from_str("Server URL is required"));
                err.runModal();
                return;
            }
            self.ivars().config.borrow_mut().server_url = server_url.clone();
            self.start_daemon("push", Some(server_url), false);
            self.save_settings();
        }
    }

    // -- Daemon lifecycle ---------------------------------------------------

    fn start_daemon(&self, mode: &str, arg: Option<String>, save_screenshots: bool) {
        let config = self.ivars().config.borrow().clone();
        let mode_str = mode.to_string();

        let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");

        let db_path = super::expand_tilde(&config.db_path);
        if let Some(parent) = db_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let db: Arc<Database> = match rt.block_on(Database::new(&db_path)) {
            Ok(db) => Arc::new(db),
            Err(e) => {
                eprintln!("Failed to open database: {e}");
                return;
            }
        };

        let llm_config = LlmConfig {
            base_url: config.llm_url.clone(),
            model: config.model.clone(),
            api_key: config.llm_api_key.clone(),
            ..Default::default()
        };

        let running = {
            let daemon = self.ivars().daemon.borrow();
            daemon.running.clone()
        };
        running.store(true, Ordering::Release);

        let should_save_screenshots = save_screenshots || config.retain_ocr_screenshots;
        let mut capture_config = super::make_capture_config(should_save_screenshots, &db_path);
        capture_config.runtime_settings_path = Some(SavedSettings::path());
        capture_config.runtime_settings_poll_secs = 2;
        if should_save_screenshots {
            let custom_dir = config.ocr_screenshot_dir.trim();
            if !custom_dir.is_empty() {
                let dir = super::expand_tilde(custom_dir);
                let _ = std::fs::create_dir_all(&dir);
                capture_config.screenshot_dir = Some(dir);
            }
        }

        match mode {
            "serve" => {
                let listen = arg.unwrap_or_else(|| config.listen.clone());
                let state = Arc::new(super::server::AppState {
                    db: db.clone(),
                    llm_config: Some(llm_config.clone()),
                });
                let llm_client = Arc::new(LlmClient::new(llm_config));

                let sched_db = db.clone();
                let sched_client = llm_client.clone();
                rt.spawn(async move {
                    super::scheduler::run_scheduler(sched_db, sched_client).await;
                });
                let app = super::server::router(state);
                let db_capture = db.clone();
                rt.spawn(async move {
                    let listener = match tokio::net::TcpListener::bind(&listen).await {
                        Ok(l) => l,
                        Err(e) => {
                            eprintln!("Failed to bind {listen}: {e}");
                            return;
                        }
                    };
                    let _ = axum::serve(listener, app).await;
                });

                rt.spawn(async move {
                    let _ =
                        screentrack_capture::capture::run_capture_loop(capture_config, db_capture)
                            .await;
                });
            }
            "push" => {
                let server_url = arg.unwrap_or_else(|| config.server_url.clone());
                let push_db = db.clone();
                let db_capture = db.clone();

                rt.spawn(async move {
                    super::client_push::run_push_loop(push_db, server_url).await;
                });

                rt.spawn(async move {
                    let _ =
                        screentrack_capture::capture::run_capture_loop(capture_config, db_capture)
                            .await;
                });
            }
            _ => {
                let llm_client = Arc::new(LlmClient::new(llm_config));
                let sched_db = db.clone();
                let sched_client = llm_client.clone();
                let db_capture = db.clone();

                rt.spawn(async move {
                    super::scheduler::run_scheduler(sched_db, sched_client).await;
                });

                rt.spawn(async move {
                    let _ =
                        screentrack_capture::capture::run_capture_loop(capture_config, db_capture)
                            .await;
                });
            }
        }

        // Watch for binary updates and auto-relaunch
        super::auto_update::spawn_watcher(&rt);

        {
            let mut daemon = self.ivars().daemon.borrow_mut();
            daemon.runtime = Some(rt);
            daemon.db = Some(db);
            daemon.mode = Some(mode_str.clone());
        }

        if let Some(ref items) = *self.ivars().status_items.borrow() {
            let mode_label = match mode_str.as_str() {
                "serve" => "Mode: Serve",
                "push" => "Mode: Push",
                _ => "Mode: Local",
            };
            items.mode_item.setTitle(&NSString::from_str(mode_label));
            items.stop_item.setEnabled(true);
            items.serve_item.setEnabled(false);
            items.push_item.setEnabled(false);
            items.local_item.setEnabled(false);
        }
    }

    fn stop_daemon(&self) {
        let mut daemon = self.ivars().daemon.borrow_mut();
        daemon.running.store(false, Ordering::Release);

        if let Some(rt) = daemon.runtime.take() {
            rt.shutdown_background();
        }
        daemon.db = None;
        daemon.mode = None;

        drop(daemon);
        if let Some(ref items) = *self.ivars().status_items.borrow() {
            items
                .mode_item
                .setTitle(&NSString::from_str("Mode: Not running"));
            items.stop_item.setEnabled(false);
            items.serve_item.setEnabled(true);
            items.push_item.setEnabled(true);
            items.local_item.setEnabled(true);

            items.frames.setTitle(&NSString::from_str("  Frames: --"));
            items
                .accessibility
                .setTitle(&NSString::from_str("    Accessibility: --"));
            items.ocr.setTitle(&NSString::from_str("    OCR: --"));
            items
                .keystrokes
                .setTitle(&NSString::from_str("    Keystrokes: --"));
            items.micro.setTitle(&NSString::from_str("  Micro: --"));
            items.hourly.setTitle(&NSString::from_str("  Hourly: --"));
            items.daily.setTitle(&NSString::from_str("  Daily: --"));
            items.weekly.setTitle(&NSString::from_str("  Weekly: --"));
        }
    }

    // -- Debug log window ---------------------------------------------------

    fn show_debug_log(&self) {
        let mtm = MainThreadMarker::new().unwrap();
        let target: &AnyObject = unsafe { std::mem::transmute(self as &AppDelegate) };

        {
            let mut dw = self.ivars().debug_window.borrow_mut();
            if let Some(ref w) = *dw {
                if w.window.isVisible() {
                    w.window.close();
                    return;
                }
            } else {
                *dw = Some(create_debug_log_window(target, mtm));
            }
        }

        {
            let dw = self.ivars().debug_window.borrow();
            if let Some(ref w) = *dw {
                w.window.makeKeyAndOrderFront(None);
                let app = NSApplication::sharedApplication(mtm);
                unsafe { app.activate() };
            }
        }
        // Immediate refresh so content shows right away
        let is_raw_mode = {
            let dw = self.ivars().debug_window.borrow();
            matches!(
                dw.as_ref().map(|w| w.mode.get()),
                Some(DebugViewMode::RawFrames)
            )
        };
        if is_raw_mode {
            self.refresh_debug_raw_frames();
        } else {
            self.refresh_debug_log();
        }
    }

    fn toggle_debug_auto_scroll(&self) {
        let should_jump_to_tail = {
            let dw = self.ivars().debug_window.borrow();
            let Some(ref w) = *dw else { return };
            if w.mode.get() != DebugViewMode::LiveLog {
                return;
            }
            let enabled = !w.auto_follow.get();
            w.auto_follow.set(enabled);
            set_debug_autoscroll_button_title(&w.pause_button, enabled);
            enabled
        };
        if should_jump_to_tail {
            // Apply any buffered updates immediately when resuming.
            self.refresh_debug_log();
        }
    }

    fn debug_view_mode_changed(&self) {
        let next_mode = {
            let dw = self.ivars().debug_window.borrow();
            let Some(ref w) = *dw else { return };
            let idx = unsafe { w.mode_popup.indexOfSelectedItem() };
            let next = DebugViewMode::from_index(idx);
            w.mode.set(next);
            if next == DebugViewMode::LiveLog {
                w.auto_follow.set(true);
                set_debug_autoscroll_button_title(&w.pause_button, true);
                w.last_rendered_version.set(0);
            }
            set_debug_view_controls(w);
            next
        };

        match next_mode {
            DebugViewMode::LiveLog => self.refresh_debug_log(),
            DebugViewMode::RawFrames => self.refresh_debug_raw_frames(),
            _ => self.refresh_debug_summaries(),
        }
    }

    fn load_recent_raw_frames(&self, limit: i64) -> Vec<screentrack_store::Frame> {
        {
            let daemon = self.ivars().daemon.borrow();
            if let (Some(rt), Some(db)) = (daemon.runtime.as_ref(), daemon.db.as_ref()) {
                let db = db.clone();
                return rt
                    .block_on(async { db.get_recent_frames(limit).await })
                    .unwrap_or_default();
            }
        }

        if !self.ensure_viewer_db() {
            return Vec::new();
        }
        let db = self.ivars().viewer_db.borrow();
        let rt = self.ivars().viewer_runtime.borrow();
        let Some(db) = db.as_ref() else {
            return Vec::new();
        };
        let Some(rt) = rt.as_ref() else {
            return Vec::new();
        };
        let db = db.clone();
        rt.block_on(async { db.get_recent_frames(limit).await })
            .unwrap_or_default()
    }

    fn refresh_debug_raw_frames(&self) {
        let dw = self.ivars().debug_window.borrow();
        let Some(ref w) = *dw else { return };
        if !w.window.isVisible() || w.mode.get() != DebugViewMode::RawFrames {
            return;
        }
        drop(dw);

        let frames = self.load_recent_raw_frames(10);
        let text = format_recent_raw_frames(&frames);

        let dw = self.ivars().debug_window.borrow();
        let Some(ref w) = *dw else { return };
        if !w.window.isVisible() || w.mode.get() != DebugViewMode::RawFrames {
            return;
        }
        unsafe { w.text_view.setString(&NSString::from_str(&text)) };
        w.text_view
            .scrollRangeToVisible(objc2_foundation::NSRange::new(0, 0));
    }

    fn load_recent_summaries(&self, tier: &str, limit: i64) -> Vec<screentrack_store::Summary> {
        {
            let daemon = self.ivars().daemon.borrow();
            if let (Some(rt), Some(db)) = (daemon.runtime.as_ref(), daemon.db.as_ref()) {
                let db = db.clone();
                let tier = tier.to_string();
                return rt
                    .block_on(async { db.get_recent_summaries(&tier, limit).await })
                    .unwrap_or_default();
            }
        }

        if !self.ensure_viewer_db() {
            return Vec::new();
        }
        let db = self.ivars().viewer_db.borrow();
        let rt = self.ivars().viewer_runtime.borrow();
        let Some(db) = db.as_ref() else {
            return Vec::new();
        };
        let Some(rt) = rt.as_ref() else {
            return Vec::new();
        };
        let db = db.clone();
        let tier = tier.to_string();
        rt.block_on(async { db.get_recent_summaries(&tier, limit).await })
            .unwrap_or_default()
    }

    fn refresh_debug_summaries(&self) {
        let mode = {
            let dw = self.ivars().debug_window.borrow();
            let Some(ref w) = *dw else { return };
            if !w.window.isVisible() {
                return;
            }
            w.mode.get()
        };

        let Some(tier) = mode.tier() else { return };
        let limit = mode.summary_limit();
        let summaries = self.load_recent_summaries(tier, limit);
        let text = format_recent_summaries(tier, &summaries);

        let dw = self.ivars().debug_window.borrow();
        let Some(ref w) = *dw else { return };
        if !w.window.isVisible() || w.mode.get() != mode {
            return;
        }
        unsafe { w.text_view.setString(&NSString::from_str(&text)) };
        w.text_view
            .scrollRangeToVisible(objc2_foundation::NSRange::new(0, 0));
    }

    fn refresh_debug_data(&self) {
        let mode = {
            let dw = self.ivars().debug_window.borrow();
            let Some(ref w) = *dw else { return };
            w.mode.get()
        };
        match mode {
            DebugViewMode::RawFrames => self.refresh_debug_raw_frames(),
            DebugViewMode::LiveLog => {}
            _ => self.refresh_debug_summaries(),
        }
    }

    fn refresh_debug_log(&self) {
        let dw = self.ivars().debug_window.borrow();
        let Some(ref w) = *dw else { return };
        if !w.window.isVisible() || w.mode.get() != DebugViewMode::LiveLog {
            return;
        }
        let version = DEBUG_LOG_VERSION.load(Ordering::Acquire);
        if version == w.last_rendered_version.get() {
            return;
        }

        // If the user scrolls up or explicitly pauses, freeze redraw so selection/copy state is stable.
        if w.auto_follow.get() && !debug_log_is_near_bottom(w) {
            w.auto_follow.set(false);
            set_debug_autoscroll_button_title(&w.pause_button, false);
            return;
        }
        if !w.auto_follow.get() {
            // Keep data flowing into DEBUG_LOG, but don't mutate text view while paused.
            return;
        }

        let text = if let Ok(log) = DEBUG_LOG.lock() {
            log.lines
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            return;
        };

        unsafe { w.text_view.setString(&NSString::from_str(&text)) };
        scroll_text_view_to_end(&w.text_view);

        w.last_rendered_version.set(version);
    }

    // -- Viewer window ------------------------------------------------------

    /// If we're in push mode, return the server URL to query against.
    fn push_server_url(&self) -> Option<String> {
        let daemon = self.ivars().daemon.borrow();
        if daemon.mode.as_deref() == Some("push") {
            let config = self.ivars().config.borrow();
            if !config.server_url.is_empty() {
                return Some(config.server_url.clone());
            }
        }
        None
    }

    fn ensure_viewer_db(&self) -> bool {
        if self.ivars().viewer_db.borrow().is_some() {
            return true;
        }
        let config = self.ivars().config.borrow().clone();
        let db_path = super::expand_tilde(&config.db_path);
        if let Some(parent) = db_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let rt = match tokio::runtime::Runtime::new() {
            Ok(rt) => rt,
            Err(e) => {
                tracing::error!("Viewer: failed to create runtime: {e}");
                return false;
            }
        };
        let db = match rt.block_on(Database::new(&db_path)) {
            Ok(db) => Arc::new(db),
            Err(e) => {
                tracing::error!(
                    "Viewer: failed to open database at {}: {e}",
                    db_path.display()
                );
                return false;
            }
        };
        tracing::info!("Viewer: opened database at {}", db_path.display());
        *self.ivars().viewer_db.borrow_mut() = Some(db);
        *self.ivars().viewer_runtime.borrow_mut() = Some(rt);
        true
    }

    fn open_viewer(&self) {
        let mtm = MainThreadMarker::new().unwrap();

        {
            let mut viewer = self.ivars().viewer.borrow_mut();
            if viewer.is_none() {
                let target: &AnyObject = unsafe { std::mem::transmute(self as &AppDelegate) };
                *viewer = Some(crate::viewer::create_viewer_window(target, mtm));
            }
        }

        let viewer = self.ivars().viewer.borrow();
        if let Some(ref v) = *viewer {
            v.window.makeKeyAndOrderFront(None);
            let app = NSApplication::sharedApplication(mtm);
            unsafe { app.activate() };
        }
        drop(viewer);

        self.handle_segment_change();
    }

    fn handle_segment_change(&self) {
        let mtm = MainThreadMarker::new().unwrap();
        let segment = {
            let viewer = self.ivars().viewer.borrow();
            let Some(ref v) = *viewer else { return };

            // Default query controls for smart-query tab.
            v.ask_button.setTitle(&NSString::from_str("Ask"));
            v.query_field
                .setPlaceholderString(Some(&NSString::from_str("Ask about your activity...")));

            let segment = v.segment_control.selectedSegment();
            match segment {
                0 => {
                    set_viewer_text(
                        &v.text_view,
                        "## Smart Query\n\nType a question below and click **Ask**.\n\nThe AI will search your activity data to find the answer.",
                        mtm,
                    );
                }
                1 => {
                    v.ask_button.setTitle(&NSString::from_str("Apply"));
                    v.query_field.setPlaceholderString(Some(&NSString::from_str(
                        "App name to add, or -AppName to remove",
                    )));
                }
                _ => {}
            }
            segment
        };
        // Call show_ocr_config_tab after dropping the viewer borrow
        if segment == 1 {
            self.show_ocr_config_tab(None);
        }
    }

    fn force_ocr_apps_path(&self) -> std::path::PathBuf {
        let db_path = super::expand_tilde(&self.ivars().config.borrow().db_path);
        db_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("force_ocr_apps.json")
    }

    fn load_force_ocr_apps(&self) -> Vec<String> {
        let path = self.force_ocr_apps_path();
        if path.exists() {
            screentrack_capture::capture::load_force_ocr_apps(&path)
        } else {
            screentrack_capture::capture::default_force_ocr_apps()
        }
    }

    fn save_force_ocr_apps(&self, apps: &[String]) -> bool {
        let path = self.force_ocr_apps_path();
        match screentrack_capture::capture::save_force_ocr_apps(&path, apps) {
            Ok(()) => true,
            Err(e) => {
                tracing::warn!("Failed to save force OCR apps to {}: {e}", path.display());
                false
            }
        }
    }

    fn show_ocr_config_tab(&self, status: Option<&str>) {
        let mtm = MainThreadMarker::new().unwrap();
        let viewer = self.ivars().viewer.borrow();
        let Some(ref v) = *viewer else { return };

        let apps = self.load_force_ocr_apps();
        let mut md = String::from("## OCR Config\n\n");
        md.push_str(
            "Apps in this list always use OCR capture instead of accessibility capture.\n\n",
        );
        md.push_str(
            "Changes apply to new captures immediately when saved; existing daemon mode may need a restart in some cases.\n\n",
        );
        if let Some(s) = status {
            md.push_str(&format!("**{}**\n\n", s));
        }
        md.push_str("### Force OCR Apps\n");
        if apps.is_empty() {
            md.push_str("- _(none)_\n");
        } else {
            for app in &apps {
                md.push_str(&format!("- `{}`\n", app));
            }
        }
        md.push_str(
            "\n### Edit\nType an app name below and click **Apply** to add it.\nType `-AppName` and click **Apply** to remove it.\n",
        );

        set_viewer_text(&v.text_view, &md, mtm);
    }

    fn handle_ocr_config_action(&self) {
        let mut input = {
            let viewer = self.ivars().viewer.borrow();
            let Some(ref v) = *viewer else { return };
            v.query_field.stringValue().to_string()
        };
        input = input.trim().to_string();
        if input.is_empty() {
            return;
        }

        let mut apps = self.load_force_ocr_apps();
        let status = if let Some(remove_target) = input.strip_prefix('-') {
            let target = remove_target.trim();
            let before = apps.len();
            apps.retain(|a| !a.eq_ignore_ascii_case(target));
            if apps.len() < before {
                format!("Removed `{}` from force OCR list.", target)
            } else {
                format!("`{}` was not in the force OCR list.", target)
            }
        } else if apps.iter().any(|a| a.eq_ignore_ascii_case(&input)) {
            format!("`{}` is already in the force OCR list.", input)
        } else {
            apps.push(input.clone());
            format!("Added `{}` to force OCR list.", input)
        };

        let _ = self.save_force_ocr_apps(&apps);

        {
            let viewer = self.ivars().viewer.borrow();
            if let Some(ref v) = *viewer {
                v.query_field.setStringValue(&NSString::from_str(""));
            }
        }

        self.show_ocr_config_tab(Some(&status));
    }

    fn handle_ask_query(&self) {
        let is_config = {
            let viewer = self.ivars().viewer.borrow();
            let Some(ref v) = *viewer else { return };
            v.segment_control.selectedSegment() == 1
        };
        if is_config {
            self.handle_ocr_config_action();
            return;
        }

        let mtm = MainThreadMarker::new().unwrap();

        let question = {
            let viewer = self.ivars().viewer.borrow();
            let Some(ref v) = *viewer else { return };
            let q = v.query_field.stringValue().to_string();
            if q.is_empty() {
                return;
            }
            unsafe { v.spinner.startAnimation(None) };
            v.ask_button.setEnabled(false);

            v.segment_control.setSelectedSegment(0);
            set_viewer_text(
                &v.text_view,
                &format!(
                    "## Smart Query\n\n**Question:** {}\n\nSearching your activity data...",
                    q
                ),
                mtm,
            );
            q
        };

        let config = self.ivars().config.borrow().clone();
        let server_url = self.push_server_url();
        let answer_slot = self.ivars().query_answer.clone();
        let pending_flag = self.ivars().query_pending.clone();

        let query_start = std::time::Instant::now();
        let question_for_log = question.clone();

        std::thread::spawn(move || {
            tracing::info!("smart_query thread: starting");

            let answer_text = if let Some(ref server_url) = server_url {
                tracing::info!("smart_query thread: using server {}", server_url);
                run_smart_query_via_server(server_url, &question)
            } else {
                tracing::info!("smart_query thread: running locally");
                let rt = tokio::runtime::Runtime::new().expect("Failed to create runtime");
                let db_path = super::expand_tilde(&config.db_path);

                let db = match rt.block_on(Database::new(&db_path)) {
                    Ok(db) => Arc::new(db),
                    Err(e) => {
                        tracing::error!("smart_query thread: db error: {e}");
                        *answer_slot.lock().unwrap() = Some(format!("## Error\n\n{}", e));
                        pending_flag.store(true, Ordering::Release);
                        return;
                    }
                };

                let llm_config = LlmConfig {
                    base_url: config.llm_url.clone(),
                    model: config.model.clone(),
                    api_key: config.llm_api_key.clone(),
                    ..Default::default()
                };
                tracing::info!(
                    "smart_query thread: llm_url={}, model={}",
                    config.llm_url,
                    config.model
                );
                let client = LlmClient::new(llm_config);
                let opts = crate::smartquery::SmartQueryOpts {
                    question,
                    follow_up: false,
                    raw: true,
                    max_rounds: 10,
                };

                match rt.block_on(async {
                    crate::smartquery::smart_query_get_answer(&db, &db, &client, &opts).await
                }) {
                    Ok(text) => {
                        tracing::info!("smart_query thread: got answer ({} bytes)", text.len());
                        text
                    }
                    Err(e) => {
                        tracing::error!("smart_query thread: query error: {e}");
                        format!("## Error\n\n{}", e)
                    }
                }
            };

            let elapsed = query_start.elapsed();
            log_smart_query(&question_for_log, &answer_text, elapsed);

            *answer_slot.lock().unwrap() = Some(answer_text);
            pending_flag.store(true, Ordering::Release);
            tracing::info!("smart_query thread: answer ready, flag set");
        });
    }

    /// Called by the 0.3s timer to check if a smart query answer is ready.
    fn poll_smart_query_answer(&self) {
        if !self.ivars().query_pending.load(Ordering::Acquire) {
            return;
        }
        self.ivars().query_pending.store(false, Ordering::Release);

        let mtm = MainThreadMarker::new().unwrap();

        let answer = self.ivars().query_answer.lock().unwrap().take();
        let Some(answer_md) = answer else {
            tracing::warn!("poll_smart_query: flag was set but answer was None");
            let viewer = self.ivars().viewer.borrow();
            if let Some(ref v) = *viewer {
                unsafe { v.spinner.stopAnimation(None) };
                v.ask_button.setEnabled(true);
            }
            return;
        };

        tracing::info!(
            "poll_smart_query: displaying answer ({} bytes)",
            answer_md.len()
        );

        let viewer = self.ivars().viewer.borrow();
        let Some(ref v) = *viewer else { return };

        unsafe { v.spinner.stopAnimation(None) };
        v.ask_button.setEnabled(true);

        set_viewer_text(&v.text_view, &answer_md, mtm);
    }

    // -- Stats refresh ------------------------------------------------------

    fn refresh_stats(&self) {
        let db = {
            let daemon = self.ivars().daemon.borrow();
            match daemon.db {
                Some(ref db) => db.clone(),
                None => return,
            }
        };

        let stats_result = {
            let daemon = self.ivars().daemon.borrow();
            match daemon.runtime {
                Some(ref rt) => {
                    let db2 = db.clone();
                    rt.block_on(async {
                        let capture_stats = db2.get_capture_stats().await.ok();
                        let summary_counts = db2.get_summary_counts(0, i64::MAX).await.ok();
                        (capture_stats, summary_counts)
                    })
                }
                None => return,
            }
        };

        let (capture_stats, summary_counts) = stats_result;
        let mut display = self.ivars().stats.borrow_mut();

        if let Some(cs) = capture_stats {
            display.frames_total = cs.frames_total;
            display.frames_accessibility = cs.frames_accessibility;
            display.frames_ocr = cs.frames_ocr_local + cs.frames_ocr_remote;
            display.frames_keystrokes = cs.frames_total
                - cs.frames_accessibility
                - cs.frames_ocr_local
                - cs.frames_ocr_remote;
        }

        if let Some(sc) = summary_counts {
            for (tier, count) in &sc {
                match tier.as_str() {
                    "micro" => display.micro = *count,
                    "hourly" => display.hourly = *count,
                    "daily" => display.daily = *count,
                    "weekly" => display.weekly = *count,
                    _ => {}
                }
            }
        }

        if let Some(ref items) = *self.ivars().status_items.borrow() {
            items.frames.setTitle(&NSString::from_str(&format!(
                "  Frames: {}",
                display.frames_total
            )));
            items.accessibility.setTitle(&NSString::from_str(&format!(
                "    Accessibility: {}",
                display.frames_accessibility
            )));
            items.ocr.setTitle(&NSString::from_str(&format!(
                "    OCR: {}",
                display.frames_ocr
            )));
            items.keystrokes.setTitle(&NSString::from_str(&format!(
                "    Keystrokes: {}",
                display.frames_keystrokes
            )));
            items
                .micro
                .setTitle(&NSString::from_str(&format!("  Micro: {}", display.micro)));
            items.hourly.setTitle(&NSString::from_str(&format!(
                "  Hourly: {}",
                display.hourly
            )));
            items
                .daily
                .setTitle(&NSString::from_str(&format!("  Daily: {}", display.daily)));
            items.weekly.setTitle(&NSString::from_str(&format!(
                "  Weekly: {}",
                display.weekly
            )));
        }
    }
}

// ---------------------------------------------------------------------------
// Debug log window
// ---------------------------------------------------------------------------

fn create_debug_log_window(target: &AnyObject, mtm: MainThreadMarker) -> DebugLogWindow {
    let rect = NSRect::new(NSPoint::new(100.0, 100.0), NSSize::new(700.0, 400.0));
    let style = NSWindowStyleMask::Titled
        .union(NSWindowStyleMask::Closable)
        .union(NSWindowStyleMask::Resizable)
        .union(NSWindowStyleMask::Miniaturizable);
    let window = unsafe {
        NSWindow::initWithContentRect_styleMask_backing_defer(
            mtm.alloc(),
            rect,
            style,
            NSBackingStoreType::Buffered,
            false,
        )
    };
    window.setTitle(&NSString::from_str("ScreenTrack Debug Log"));
    window.center();
    unsafe { window.setReleasedWhenClosed(false) };

    let content = window.contentView().unwrap();
    let f = content.frame();
    let top_bar_h = 34.0;

    let scroll = NSScrollView::initWithFrame(
        mtm.alloc(),
        NSRect::new(
            NSPoint::new(0.0, 0.0),
            NSSize::new(f.size.width, f.size.height - top_bar_h),
        ),
    );
    scroll.setHasVerticalScroller(true);
    scroll.setAutoresizingMask(
        NSAutoresizingMaskOptions::ViewWidthSizable | NSAutoresizingMaskOptions::ViewHeightSizable,
    );

    let text_view = NSTextView::initWithFrame(
        mtm.alloc(),
        NSRect::new(
            NSPoint::new(0.0, 0.0),
            NSSize::new(f.size.width, f.size.height - top_bar_h),
        ),
    );
    text_view.setEditable(false);
    text_view.setSelectable(true);
    text_view.setMinSize(NSSize::new(0.0, f.size.height));
    text_view.setMaxSize(NSSize::new(f64::MAX, f64::MAX));
    text_view.setVerticallyResizable(true);
    text_view.setHorizontallyResizable(false);
    text_view.setFont(Some(&NSFont::monospacedSystemFontOfSize_weight(11.0, 0.0)));
    text_view.setAutoresizingMask(NSAutoresizingMaskOptions::ViewWidthSizable);
    if let Some(container) = unsafe { text_view.textContainer() } {
        unsafe {
            container.setContainerSize(NSSize::new(f.size.width, f64::MAX));
            container.setWidthTracksTextView(true);
            container.setHeightTracksTextView(false);
        };
    }

    // Dark background for log readability
    unsafe { text_view.setBackgroundColor(&NSColor::textBackgroundColor()) };

    scroll.setDocumentView(Some(&text_view));
    content.addSubview(&scroll);

    let pause_button = unsafe {
        NSButton::buttonWithTitle_target_action(
            &NSString::from_str("Pause Auto-Scroll"),
            Some(target),
            Some(objc2::sel!(toggleDebugAutoScroll:)),
            mtm,
        )
    };
    let button_h = 24.0;

    // Dropdown for selecting view mode
    let mode_popup = unsafe {
        let popup = NSPopUpButton::initWithFrame_pullsDown(
            mtm.alloc(),
            NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(180.0, button_h)),
            false,
        );
        popup.addItemWithTitle(&NSString::from_str("Live Log"));
        popup.addItemWithTitle(&NSString::from_str("Raw Frames"));
        popup.addItemWithTitle(&NSString::from_str("Micro Summaries"));
        popup.addItemWithTitle(&NSString::from_str("Hourly Summaries"));
        popup.addItemWithTitle(&NSString::from_str("Daily Summaries"));
        popup.addItemWithTitle(&NSString::from_str("Weekly Summaries"));
        popup.setTarget(Some(target));
        popup.setAction(Some(objc2::sel!(debugViewModeChanged:)));
        popup
    };
    let mode_w = 180.0;
    mode_popup.setFrame(NSRect::new(
        NSPoint::new(10.0, f.size.height - button_h - 6.0),
        NSSize::new(mode_w, button_h),
    ));
    mode_popup.setAutoresizingMask(
        NSAutoresizingMaskOptions::ViewMaxXMargin | NSAutoresizingMaskOptions::ViewMinYMargin,
    );
    content.addSubview(&mode_popup);

    let refresh_button = unsafe {
        NSButton::buttonWithTitle_target_action(
            &NSString::from_str("Refresh"),
            Some(target),
            Some(objc2::sel!(refreshDebugData:)),
            mtm,
        )
    };
    let refresh_w = 80.0;
    refresh_button.setFrame(NSRect::new(
        NSPoint::new(10.0 + mode_w + 8.0, f.size.height - button_h - 6.0),
        NSSize::new(refresh_w, button_h),
    ));
    refresh_button.setAutoresizingMask(
        NSAutoresizingMaskOptions::ViewMaxXMargin | NSAutoresizingMaskOptions::ViewMinYMargin,
    );
    refresh_button.setHidden(true);
    content.addSubview(&refresh_button);

    let button_w = 170.0;
    pause_button.setFrame(NSRect::new(
        NSPoint::new(
            f.size.width - button_w - 10.0,
            f.size.height - button_h - 6.0,
        ),
        NSSize::new(button_w, button_h),
    ));
    pause_button.setAutoresizingMask(
        NSAutoresizingMaskOptions::ViewMinXMargin | NSAutoresizingMaskOptions::ViewMinYMargin,
    );
    content.addSubview(&pause_button);

    let w = DebugLogWindow {
        window,
        text_view,
        pause_button,
        mode_popup,
        refresh_button,
        _scroll_view: scroll,
        mode: Cell::new(DebugViewMode::LiveLog),
        auto_follow: Cell::new(true),
        last_rendered_version: Cell::new(0),
    };
    set_debug_view_controls(&w);
    w
}

/// Set the viewer text view content from markdown, forcing a display update.
fn set_viewer_text(text_view: &NSTextView, markdown: &str, mtm: MainThreadMarker) {
    let attributed = crate::md_attributed::markdown_to_attributed_string(markdown, mtm);
    if let Some(storage) = unsafe { text_view.textStorage() } {
        storage.beginEditing();
        let len = storage.length();
        unsafe {
            let _: () = msg_send![
                &*storage,
                replaceCharactersInRange: objc2_foundation::NSRange::new(0, len),
                withAttributedString: &*attributed
            ];
        }
        storage.endEditing();
    }
    // Force the layout manager to fully compute layout for the new content
    // before we scroll or display — otherwise display() can race the layout
    // and render stale content for longer texts.
    unsafe {
        if let Some(lm) = text_view.layoutManager() {
            if let Some(tc) = text_view.textContainer() {
                lm.ensureLayoutForTextContainer(&tc);
            }
        }
    }
    text_view.scrollRangeToVisible(objc2_foundation::NSRange::new(0, 0));
    unsafe { text_view.display() };
}

fn set_debug_autoscroll_button_title(button: &NSButton, auto_follow: bool) {
    let title = if auto_follow {
        "Pause Auto-Scroll"
    } else {
        "Resume Auto-Scroll"
    };
    button.setTitle(&NSString::from_str(title));
}

fn set_debug_view_controls(w: &DebugLogWindow) {
    let mode = w.mode.get();
    match mode {
        DebugViewMode::LiveLog => {
            w.pause_button.setHidden(false);
            w.refresh_button.setHidden(true);
        }
        _ => {
            // All data views: hide pause, show refresh
            w.pause_button.setHidden(true);
            w.refresh_button.setHidden(false);
        }
    }
}

fn format_recent_raw_frames(frames: &[screentrack_store::Frame]) -> String {
    if frames.is_empty() {
        return "Last 10 raw frames (newest first)\n\nNo frames found.".to_string();
    }

    let mut out = String::from("Last 10 raw frames (newest first)\n");
    for (idx, frame) in frames.iter().enumerate() {
        let ts = Utc
            .timestamp_millis_opt(frame.timestamp)
            .single()
            .map(|t| t.with_timezone(&Local).format("%H:%M:%S%.3f").to_string())
            .unwrap_or_else(|| frame.timestamp.to_string());
        out.push_str("\n");
        out.push_str(&format!("--- Frame {} ---\n", idx + 1));
        out.push_str(&format!("time: {ts}\n"));
        out.push_str(&format!(
            "app: {}\n",
            frame.app_name.as_deref().unwrap_or("unknown")
        ));
        out.push_str(&format!(
            "window: {}\n",
            frame.window_title.as_deref().unwrap_or("")
        ));
        out.push_str(&format!(
            "tab: {}\n",
            frame.browser_tab.as_deref().unwrap_or("")
        ));
        out.push_str(&format!("source: {}\n", frame.source));
        out.push_str("text:\n");
        out.push_str(&frame.text_content);
        out.push_str("\n");
    }
    out
}

fn format_recent_summaries(tier: &str, summaries: &[screentrack_store::Summary]) -> String {
    let label = match tier {
        "micro" => "Micro",
        "hourly" => "Hourly",
        "daily" => "Daily",
        "weekly" => "Weekly",
        other => other,
    };

    if summaries.is_empty() {
        return format!("{label} Summaries (newest first)\n\nNo summaries found.");
    }

    let mut out = format!("{label} Summaries (newest first) — {} found\n", summaries.len());
    for (idx, s) in summaries.iter().enumerate() {
        let start = Utc
            .timestamp_millis_opt(s.start_time)
            .single()
            .map(|t| t.with_timezone(&Local));
        let end = Utc
            .timestamp_millis_opt(s.end_time)
            .single()
            .map(|t| t.with_timezone(&Local));

        let time_str = match (start, end) {
            (Some(s), Some(e)) => {
                if tier == "daily" || tier == "weekly" {
                    format!("{} — {}", s.format("%Y-%m-%d %H:%M"), e.format("%Y-%m-%d %H:%M"))
                } else {
                    format!("{} — {}", s.format("%H:%M:%S"), e.format("%H:%M:%S"))
                }
            }
            _ => format!("{}..{}", s.start_time, s.end_time),
        };

        out.push_str(&format!("\n--- {} Summary {} ---\n", label, idx + 1));
        out.push_str(&format!("time: {time_str}\n"));
        if let Some(ref apps) = s.apps_referenced {
            out.push_str(&format!("apps: {apps}\n"));
        }
        out.push('\n');
        out.push_str(&s.summary);
        out.push('\n');
    }
    out
}

fn scroll_text_view_to_end(text_view: &NSTextView) {
    let len = text_view.string().len();
    text_view.setSelectedRange(objc2_foundation::NSRange::new(len, 0));
    unsafe {
        let _: () = objc2::msg_send![
            text_view,
            scrollToEndOfDocument: std::ptr::null::<AnyObject>()
        ];
    }
}

fn debug_log_is_near_bottom(w: &DebugLogWindow) -> bool {
    let clip = w._scroll_view.contentView();
    let visible = clip.bounds();
    let doc_height = w
        .text_view
        .bounds()
        .size
        .height
        .max(w.text_view.frame().size.height);
    let visible_bottom = visible.origin.y + visible.size.height;
    doc_height <= visible_bottom + 8.0
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn log_smart_query(question: &str, answer: &str, elapsed: std::time::Duration) {
    use std::io::Write;

    let log_path = dirs_next::home_dir()
        .unwrap_or_default()
        .join(".screentrack")
        .join("smart_query_log.txt");

    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
    let secs = elapsed.as_secs_f64();

    let entry = format!(
        "=== QUERY at {timestamp} (took {secs:.2}s, answer {answer_len} bytes) ===\n\
         Q: {question}\n\
         \n\
         A:\n\
         {answer}\n\
         \n\
         === END ===\n\n",
        answer_len = answer.len(),
    );

    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
    {
        let _ = f.write_all(entry.as_bytes());
    }
}

fn make_disabled_item(title: &str, mtm: MainThreadMarker) -> Retained<NSMenuItem> {
    let item = unsafe {
        NSMenuItem::initWithTitle_action_keyEquivalent(
            mtm.alloc(),
            &NSString::from_str(title),
            None,
            ns_string!(""),
        )
    };
    item.setEnabled(false);
    item
}

fn make_action_item(
    title: &str,
    action: objc2::runtime::Sel,
    target: &AppDelegate,
    mtm: MainThreadMarker,
) -> Retained<NSMenuItem> {
    let item = unsafe {
        NSMenuItem::initWithTitle_action_keyEquivalent(
            mtm.alloc(),
            &NSString::from_str(title),
            Some(action),
            ns_string!(""),
        )
    };
    let target_obj: &AnyObject = unsafe { std::mem::transmute(target as &AppDelegate) };
    unsafe { item.setTarget(Some(target_obj)) };
    item
}

/// Run a smart query via the server's HTTP API.
fn run_smart_query_via_server(server_url: &str, question: &str) -> String {
    #[derive(serde::Serialize)]
    struct Req<'a> {
        question: &'a str,
        max_rounds: usize,
    }
    #[derive(serde::Deserialize)]
    struct Resp {
        answer: String,
    }

    let url = format!("{}/api/v1/smart-query", server_url);
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .unwrap();
    let resp = match client
        .post(&url)
        .json(&Req {
            question,
            max_rounds: 10,
        })
        .send()
    {
        Ok(r) => r,
        Err(e) => return format!("## Error\n\nCould not reach server: {e}"),
    };
    match resp.json::<Resp>() {
        Ok(r) => r.answer,
        Err(e) => format!("## Error\n\nBad response from server: {e}"),
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Entry point for the menu bar application. Does not return.
pub fn run_menubar_app(
    db_path: String,
    llm_url: String,
    model: String,
    llm_api_key: Option<String>,
) {
    if let Ok(mut log) = DEBUG_LOG.lock() {
        log.push(format!(
            "ScreenTrack v{} (built {})",
            env!("CARGO_PKG_VERSION"),
            env!("BUILD_TIMESTAMP"),
        ));
        log.push(String::new());
    }
    tracing::info!("ScreenTrack GUI starting");

    let mtm = MainThreadMarker::new().expect("Must be called from the main thread");
    let app = NSApplication::sharedApplication(mtm);
    app.setActivationPolicy(NSApplicationActivationPolicy::Accessory);

    // Load saved settings, merging with CLI defaults
    let saved = SavedSettings::load();
    let config = if let Some(ref s) = saved {
        AppConfig {
            db_path: s.db_path.clone(),
            llm_url: s.llm_url.clone(),
            model: s.model.clone(),
            llm_api_key: s.llm_api_key.clone(),
            listen: s.listen.clone().unwrap_or_else(|| "0.0.0.0:7878".into()),
            server_url: s.server_url.clone().unwrap_or_default(),
            retain_ocr_screenshots: s.retain_ocr_screenshots,
            ocr_screenshot_dir: s.ocr_screenshot_dir.clone().unwrap_or_default(),
        }
    } else {
        AppConfig {
            db_path,
            llm_url,
            model,
            llm_api_key,
            listen: "0.0.0.0:7878".into(),
            server_url: String::new(),
            retain_ocr_screenshots: false,
            ocr_screenshot_dir: String::new(),
        }
    };

    let delegate = AppDelegate::new(config);
    let delegate_obj: &AnyObject = unsafe { std::mem::transmute(&*delegate as &AppDelegate) };

    // Create status bar item
    let status_bar = NSStatusBar::systemStatusBar();
    let status_item = status_bar.statusItemWithLength(-1.0);

    if let Some(button) = status_item.button(mtm) {
        button.setTitle(&NSString::from_str("ST"));
    }

    // Build menu
    let menu = NSMenu::new(mtm);

    let header = make_disabled_item(&format!("ScreenTrack v{}", env!("CARGO_PKG_VERSION")), mtm);
    menu.addItem(&header);
    menu.addItem(&NSMenuItem::separatorItem(mtm));

    let mode_item = make_disabled_item("Mode: Not running", mtm);
    menu.addItem(&mode_item);
    menu.addItem(&NSMenuItem::separatorItem(mtm));

    let serve_item = make_action_item(
        "Start Serve Mode...",
        objc2::sel!(startServe:),
        &delegate,
        mtm,
    );
    menu.addItem(&serve_item);

    let push_item = make_action_item(
        "Start Push Mode...",
        objc2::sel!(startPush:),
        &delegate,
        mtm,
    );
    menu.addItem(&push_item);

    let local_item = make_action_item("Start Local Mode", objc2::sel!(startLocal:), &delegate, mtm);
    menu.addItem(&local_item);

    menu.addItem(&NSMenuItem::separatorItem(mtm));

    // Status - Frames
    let status_header = make_disabled_item("--- Frames ---", mtm);
    menu.addItem(&status_header);
    let frames_item = make_disabled_item("  Frames: --", mtm);
    menu.addItem(&frames_item);
    let acc_item = make_disabled_item("    Accessibility: --", mtm);
    menu.addItem(&acc_item);
    let ocr_item = make_disabled_item("    OCR: --", mtm);
    menu.addItem(&ocr_item);
    let keys_item = make_disabled_item("    Keystrokes: --", mtm);
    menu.addItem(&keys_item);

    menu.addItem(&NSMenuItem::separatorItem(mtm));

    // Status - Summaries
    let summaries_header = make_disabled_item("--- Summaries ---", mtm);
    menu.addItem(&summaries_header);
    let micro_item = make_disabled_item("  Micro: --", mtm);
    menu.addItem(&micro_item);
    let hourly_item = make_disabled_item("  Hourly: --", mtm);
    menu.addItem(&hourly_item);
    let daily_item = make_disabled_item("  Daily: --", mtm);
    menu.addItem(&daily_item);
    let weekly_item = make_disabled_item("  Weekly: --", mtm);
    menu.addItem(&weekly_item);

    menu.addItem(&NSMenuItem::separatorItem(mtm));

    // Browse
    let browse_item = make_action_item("Browse...", objc2::sel!(openViewer:), &delegate, mtm);
    menu.addItem(&browse_item);

    menu.addItem(&NSMenuItem::separatorItem(mtm));

    // Settings & Debug
    let settings_item = make_action_item("Settings...", objc2::sel!(showSettings:), &delegate, mtm);
    menu.addItem(&settings_item);

    let debug_item = make_action_item("Debug Log...", objc2::sel!(showDebugLog:), &delegate, mtm);
    menu.addItem(&debug_item);

    menu.addItem(&NSMenuItem::separatorItem(mtm));

    // Stop
    let stop_item = make_action_item("Stop", objc2::sel!(stopDaemon:), &delegate, mtm);
    stop_item.setEnabled(false);
    menu.addItem(&stop_item);

    // Quit
    let quit_item = make_action_item("Quit ScreenTrack", objc2::sel!(quit:), &delegate, mtm);
    menu.addItem(&quit_item);

    status_item.setMenu(Some(&menu));

    // Store menu item refs
    {
        *delegate.ivars().status_items.borrow_mut() = Some(StatusItems {
            frames: frames_item,
            accessibility: acc_item,
            ocr: ocr_item,
            keystrokes: keys_item,
            micro: micro_item,
            hourly: hourly_item,
            daily: daily_item,
            weekly: weekly_item,
            mode_item,
            stop_item,
            serve_item,
            push_item,
            local_item,
        });
    }

    // Auto-start from saved settings
    if let Some(ref s) = saved {
        if let Some(ref mode) = s.mode {
            let arg = match mode.as_str() {
                "serve" => s.listen.clone(),
                "push" => s.server_url.clone(),
                _ => None,
            };
            delegate.start_daemon(mode, arg, false);
        }
    }

    // Timers in common run-loop modes so they keep firing during menu/window interactions.
    let _stats_timer = unsafe {
        NSTimer::timerWithTimeInterval_target_selector_userInfo_repeats(
            2.0,
            delegate_obj,
            objc2::sel!(updateStatus:),
            None,
            true,
        )
    };
    let _debug_timer = unsafe {
        NSTimer::timerWithTimeInterval_target_selector_userInfo_repeats(
            0.3,
            delegate_obj,
            objc2::sel!(refreshDebugLog:),
            None,
            true,
        )
    };
    let runloop = NSRunLoop::mainRunLoop();
    unsafe {
        runloop.addTimer_forMode(&_stats_timer, NSRunLoopCommonModes);
        runloop.addTimer_forMode(&_debug_timer, NSRunLoopCommonModes);
    }

    std::mem::forget(status_item);
    std::mem::forget(delegate);
    std::mem::forget(_stats_timer);
    std::mem::forget(_debug_timer);

    unsafe { app.run() };
}

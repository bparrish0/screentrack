use anyhow::Result;
use core_foundation::runloop::{kCFRunLoopDefaultMode, CFRunLoop};
use core_graphics::event::{
    CGEvent, CGEventTap, CGEventTapLocation, CGEventTapOptions, CGEventTapPlacement,
    CGEventTapProxy, CGEventType,
};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

/// Events that trigger a screen capture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CaptureEvent {
    /// User switched to a different application.
    AppSwitch { app_name: String },
    /// Mouse click at screen coordinates.
    Click { x: f64, y: f64 },
    /// User stopped typing for a configured duration.
    TypingPause,
    /// A key was pressed (fires immediately, used to track which app received input).
    Keystroke,
    /// User stopped scrolling for a configured duration.
    ScrollStop,
    /// No user activity for a long time.
    Idle,
    /// Periodic check interval fired (baseline polling).
    Periodic,
}

/// Configuration for the event listener.
pub struct EventConfig {
    /// How long after the last keystroke to emit TypingPause (ms).
    pub typing_pause_ms: u64,
    /// How long after the last scroll to emit ScrollStop (ms).
    pub scroll_stop_ms: u64,
    /// How long with no activity to emit Idle (ms).
    pub idle_timeout_ms: u64,
    /// Minimum time between emitted events (ms).
    pub debounce_ms: u64,
    /// Baseline periodic check interval (ms). Screen is checked at least this often.
    pub periodic_check_ms: u64,
}

impl Default for EventConfig {
    fn default() -> Self {
        Self {
            typing_pause_ms: 500,
            scroll_stop_ms: 300,
            idle_timeout_ms: 30_000,
            debounce_ms: 200,
            periodic_check_ms: 2_000, // Check every 2 seconds baseline
        }
    }
}

/// Shared state between the CGEvent tap callback and the timer thread.
struct TapState {
    last_key: std::sync::Mutex<Option<Instant>>,
    last_scroll: std::sync::Mutex<Option<Instant>>,
    last_activity: std::sync::Mutex<Instant>,
    /// Epoch ms of last user input (including mouse movement). Lock-free, shared with capture loop.
    last_activity_ms: Arc<AtomicU64>,
    got_click: AtomicBool,
    got_keystroke: AtomicBool,
    click_x: std::sync::Mutex<f64>,
    click_y: std::sync::Mutex<f64>,
    /// Buffer of characters typed since last drain. Used as fallback when accessibility fails.
    keystroke_buffer: Arc<std::sync::Mutex<String>>,
    /// Per-keystroke timestamps (epoch ms) for typing speed analysis.
    keystroke_times: Arc<std::sync::Mutex<Vec<i64>>>,
    /// Per-keystroke characters, paired with keystroke_times. NOT cleared by
    /// accessibility captures — only drained on TypingPause for burst storage.
    keystroke_chars: Arc<std::sync::Mutex<Vec<char>>>,
}

/// Start the event listener. Spawns two threads:
/// 1. A CGEvent tap thread that intercepts input events via a CFRunLoop.
/// 2. A timer thread that checks for typing pauses, scroll stops, idle, and periodic ticks.
///
/// Returns a receiver for capture events and an atomic tracking the last activity time
/// (epoch ms, includes mouse movement) for use in active window tracking.
pub fn start_event_listener(
    config: EventConfig,
) -> Result<(
    mpsc::Receiver<CaptureEvent>,
    Arc<AtomicU64>,
    Arc<std::sync::Mutex<String>>,
    Arc<std::sync::Mutex<Vec<i64>>>,
    Arc<std::sync::Mutex<Vec<char>>>,
)> {
    let (tx, rx) = mpsc::channel();

    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let last_activity_ms = Arc::new(AtomicU64::new(now_ms));
    let keystroke_buffer = Arc::new(std::sync::Mutex::new(String::new()));
    let keystroke_times = Arc::new(std::sync::Mutex::new(Vec::<i64>::new()));
    let keystroke_chars = Arc::new(std::sync::Mutex::new(Vec::<char>::new()));

    let state = Arc::new(TapState {
        last_key: std::sync::Mutex::new(None),
        last_scroll: std::sync::Mutex::new(None),
        last_activity: std::sync::Mutex::new(Instant::now()),
        last_activity_ms: last_activity_ms.clone(),
        got_click: AtomicBool::new(false),
        got_keystroke: AtomicBool::new(false),
        click_x: std::sync::Mutex::new(0.0),
        click_y: std::sync::Mutex::new(0.0),
        keystroke_buffer: keystroke_buffer.clone(),
        keystroke_times: keystroke_times.clone(),
        keystroke_chars: keystroke_chars.clone(),
    });

    // Thread 1: CGEvent tap with CFRunLoop
    let tap_state = state.clone();
    std::thread::Builder::new()
        .name("screentrack-cgevent-tap".into())
        .spawn(move || {
            run_event_tap(tap_state);
        })?;

    // Thread 2: Timer that checks state and emits CaptureEvents
    let timer_state = state.clone();
    std::thread::Builder::new()
        .name("screentrack-event-timer".into())
        .spawn(move || {
            run_timer_loop(config, timer_state, tx);
        })?;

    Ok((rx, last_activity_ms, keystroke_buffer, keystroke_times, keystroke_chars))
}

/// Set up the CGEvent tap and run the CFRunLoop.
/// This intercepts mouse clicks, key presses, and scroll events system-wide.
fn run_event_tap(state: Arc<TapState>) {
    let events_of_interest = vec![
        CGEventType::LeftMouseDown,
        CGEventType::RightMouseDown,
        CGEventType::KeyDown,
        CGEventType::ScrollWheel,
        CGEventType::MouseMoved,
    ];

    let tap = CGEventTap::new(
        CGEventTapLocation::HID,
        CGEventTapPlacement::HeadInsertEventTap,
        CGEventTapOptions::ListenOnly,
        events_of_interest,
        move |_proxy: CGEventTapProxy,
              event_type: CGEventType,
              event: &CGEvent|
              -> Option<CGEvent> {
            let now = Instant::now();
            *state.last_activity.lock().unwrap() = now;
            // Update lock-free epoch ms for the capture loop's presence detection
            let epoch_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
            state.last_activity_ms.store(epoch_ms, Ordering::Release);

            match event_type {
                CGEventType::LeftMouseDown | CGEventType::RightMouseDown => {
                    let loc = event.location();
                    *state.click_x.lock().unwrap() = loc.x;
                    *state.click_y.lock().unwrap() = loc.y;
                    state.got_click.store(true, Ordering::Release);
                }
                CGEventType::KeyDown => {
                    *state.last_key.lock().unwrap() = Some(now);
                    state.got_keystroke.store(true, Ordering::Release);

                    // Extract typed character and buffer it
                    if let Some(ch) = get_event_unicode(event) {
                        let mut buf = state.keystroke_buffer.lock().unwrap();
                        buf.push(ch);
                        // Record timestamp + char for typing burst storage
                        state.keystroke_times.lock().unwrap().push(epoch_ms as i64);
                        state.keystroke_chars.lock().unwrap().push(ch);
                    }
                }
                CGEventType::ScrollWheel => {
                    *state.last_scroll.lock().unwrap() = Some(now);
                }
                // MouseMoved: just updates last_activity (handled above)
                _ => {}
            }

            // ListenOnly tap: always return the event unchanged
            None
        },
    );

    match tap {
        Ok(tap) => {
            tap.enable();
            let source = tap
                .mach_port
                .create_runloop_source(0)
                .expect("Failed to create run loop source from event tap");
            let run_loop = CFRunLoop::get_current();
            run_loop.add_source(&source, unsafe { kCFRunLoopDefaultMode });
            info!("CGEvent tap active — listening for input events");
            // This blocks forever, processing events via the run loop.
            CFRunLoop::run_current();
        }
        Err(()) => {
            warn!("Failed to create CGEvent tap. Accessibility permissions may not be granted.");
            warn!("The periodic baseline capture will still work.");
            // Block forever so the thread doesn't exit
            loop {
                std::thread::sleep(Duration::from_secs(3600));
            }
        }
    }
}

/// Timer loop that monitors TapState and emits CaptureEvents.
fn run_timer_loop(config: EventConfig, state: Arc<TapState>, tx: mpsc::Sender<CaptureEvent>) {
    let mut last_emit_time = Instant::now();
    let mut last_periodic = Instant::now();
    let mut typing_pause_sent = false;
    let mut scroll_stop_sent = false;
    let mut idle_sent = false;

    let debounce = Duration::from_millis(config.debounce_ms);
    let typing_pause = Duration::from_millis(config.typing_pause_ms);
    let scroll_stop = Duration::from_millis(config.scroll_stop_ms);
    let idle_timeout = Duration::from_millis(config.idle_timeout_ms);
    let periodic = Duration::from_millis(config.periodic_check_ms);

    loop {
        std::thread::sleep(Duration::from_millis(50));
        let now = Instant::now();

        // Check for keystroke (fires immediately, no debounce — used for input tracking only)
        if state.got_keystroke.swap(false, Ordering::AcqRel) {
            let _ = tx.send(CaptureEvent::Keystroke);
        }

        // Check for click
        if state.got_click.swap(false, Ordering::AcqRel) {
            if now.duration_since(last_emit_time) >= debounce {
                let x = *state.click_x.lock().unwrap();
                let y = *state.click_y.lock().unwrap();
                let _ = tx.send(CaptureEvent::Click { x, y });
                last_emit_time = now;
                debug!("Event: Click at ({x:.0}, {y:.0})");
            }
            idle_sent = false;
        }

        // Check for typing pause
        let key_time = *state.last_key.lock().unwrap();
        if let Some(kt) = key_time {
            if now.duration_since(kt) >= typing_pause {
                if !typing_pause_sent && now.duration_since(last_emit_time) >= debounce {
                    let _ = tx.send(CaptureEvent::TypingPause);
                    last_emit_time = now;
                    typing_pause_sent = true;
                    debug!("Event: TypingPause");
                }
            } else {
                // Still typing
                typing_pause_sent = false;
            }
            idle_sent = false;
        }

        // Check for scroll stop
        let scroll_time = *state.last_scroll.lock().unwrap();
        if let Some(st) = scroll_time {
            if now.duration_since(st) >= scroll_stop {
                if !scroll_stop_sent && now.duration_since(last_emit_time) >= debounce {
                    let _ = tx.send(CaptureEvent::ScrollStop);
                    last_emit_time = now;
                    scroll_stop_sent = true;
                    debug!("Event: ScrollStop");
                }
            } else {
                scroll_stop_sent = false;
            }
            idle_sent = false;
        }

        // Check for idle
        let last_activity = *state.last_activity.lock().unwrap();
        if now.duration_since(last_activity) >= idle_timeout && !idle_sent {
            let _ = tx.send(CaptureEvent::Idle);
            idle_sent = true;
            debug!("Event: Idle");
        }

        // Periodic baseline check
        if now.duration_since(last_periodic) >= periodic {
            let _ = tx.send(CaptureEvent::Periodic);
            last_periodic = now;
        }
    }
}

/// Extract the unicode character from a CGEvent KeyDown event.
fn get_event_unicode(event: &CGEvent) -> Option<char> {
    use foreign_types_shared::ForeignType;

    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGEventKeyboardGetUnicodeString(
            event: core_graphics::sys::CGEventRef,
            max_len: u32,
            actual_len: *mut u32,
            buf: *mut u16,
        );
    }

    let mut buf = [0u16; 4];
    let mut actual_len: u32 = 0;
    unsafe {
        CGEventKeyboardGetUnicodeString(
            event.as_ptr(),
            buf.len() as u32,
            &mut actual_len,
            buf.as_mut_ptr(),
        );
    }
    if actual_len == 0 {
        return None;
    }
    let ch = char::decode_utf16(buf[..actual_len as usize].iter().copied())
        .next()?
        .ok()?;

    // Skip non-printable control characters (arrow keys, function keys, etc.)
    // but keep newline, tab, and regular printable chars
    if ch == '\n' || ch == '\r' || ch == '\t' || !ch.is_control() {
        Some(ch)
    } else {
        None
    }
}

/// Check if the current process has accessibility permissions.
pub fn check_accessibility_permissions() -> bool {
    #[link(name = "ApplicationServices", kind = "framework")]
    extern "C" {
        fn AXIsProcessTrusted() -> bool;
    }
    unsafe { AXIsProcessTrusted() }
}

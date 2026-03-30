use anyhow::{Context, Result};
use image::DynamicImage;
use xcap::{Monitor, Window};

pub struct MonitorInfo {
    pub id: u32,
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub is_primary: bool,
}

/// Enumerate all connected monitors.
pub fn list_monitors() -> Result<Vec<MonitorInfo>> {
    let monitors = Monitor::all().context("Failed to enumerate monitors")?;
    let mut infos = Vec::new();

    for m in monitors.iter() {
        infos.push(MonitorInfo {
            id: m.id()?,
            name: m.name()?,
            width: m.width()?,
            height: m.height()?,
            is_primary: m.is_primary()?,
        });
    }

    Ok(infos)
}

/// Capture a screenshot from the primary monitor.
pub fn capture_primary() -> Result<DynamicImage> {
    let monitors = Monitor::all().context("Failed to enumerate monitors")?;
    let primary = monitors
        .into_iter()
        .find(|m| m.is_primary().unwrap_or(false))
        .context("No primary monitor found")?;

    let image = primary
        .capture_image()
        .context("Failed to capture screenshot")?;

    Ok(DynamicImage::ImageRgba8(image))
}

/// Capture a screenshot of the focused window only.
/// Captures from the focused window's monitor and crops to window bounds. This ensures we get
/// the fully composited pixels including child rendering surfaces (e.g. FreeRDP in
/// Royal TSX) while also handling multi-monitor coordinate spaces correctly.
/// Returns an error if no focused window can be captured.
pub fn capture_focused_window() -> Result<DynamicImage> {
    let windows = Window::all().context("Failed to enumerate windows")?;
    let focused = windows
        .iter()
        .find(|w| w.is_focused().unwrap_or(false) && !w.is_minimized().unwrap_or(false))
        .or_else(|| windows.iter().find(|w| w.is_focused().unwrap_or(false)));

    let Some(win) = focused else {
        anyhow::bail!("No focused window found");
    };

    let win_x = win.x().unwrap_or(0);
    let win_y = win.y().unwrap_or(0);
    let win_w = win.width().unwrap_or(0) as i32;
    let win_h = win.height().unwrap_or(0) as i32;

    if win_w <= 0 || win_h <= 0 {
        anyhow::bail!("Focused window had invalid bounds");
    }

    let monitor = match win.current_monitor() {
        Ok(m) => m,
        Err(_) => {
            // Fallback: direct window capture if monitor resolution fails.
            return win
                .capture_image()
                .map(DynamicImage::ImageRgba8)
                .context("Failed to capture focused window image");
        }
    };

    let mon_x = monitor.x().unwrap_or(0);
    let mon_y = monitor.y().unwrap_or(0);
    let mon_w = monitor.width().unwrap_or(0) as i32;
    let mon_h = monitor.height().unwrap_or(0) as i32;

    if mon_w <= 0 || mon_h <= 0 {
        return win
            .capture_image()
            .map(DynamicImage::ImageRgba8)
            .context("Failed to capture focused window image");
    }

    // Convert global window coordinates to monitor-local region and clamp to bounds.
    let mut local_x = win_x - mon_x;
    let mut local_y = win_y - mon_y;
    let mut local_w = win_w;
    let mut local_h = win_h;

    if local_x < 0 {
        local_w += local_x;
        local_x = 0;
    }
    if local_y < 0 {
        local_h += local_y;
        local_y = 0;
    }
    local_w = local_w.min(mon_w.saturating_sub(local_x));
    local_h = local_h.min(mon_h.saturating_sub(local_y));

    if local_w <= 0 || local_h <= 0 {
        return monitor
            .capture_image()
            .map(DynamicImage::ImageRgba8)
            .context("Failed to capture focused monitor image");
    }

    monitor
        .capture_region(
            local_x as u32,
            local_y as u32,
            local_w as u32,
            local_h as u32,
        )
        .map(DynamicImage::ImageRgba8)
        .or_else(|_| {
            // Last-resort fallback: direct window capture.
            win.capture_image().map(DynamicImage::ImageRgba8)
        })
        .context("Failed to capture focused window region")
}

/// Capture a screenshot from a specific monitor by id.
pub fn capture_monitor(monitor_id: u32) -> Result<DynamicImage> {
    let monitors = Monitor::all().context("Failed to enumerate monitors")?;
    let monitor = monitors
        .into_iter()
        .find(|m| m.id().unwrap_or(0) == monitor_id)
        .with_context(|| format!("Monitor {monitor_id} not found"))?;

    let image = monitor
        .capture_image()
        .context("Failed to capture screenshot")?;

    Ok(DynamicImage::ImageRgba8(image))
}

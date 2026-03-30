use anyhow::Result;
use core_foundation::base::{CFType, TCFType};
use core_foundation::string::CFString;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tracing::debug;

// AXUIElement FFI bindings
#[link(name = "ApplicationServices", kind = "framework")]
extern "C" {
    fn AXUIElementCreateSystemWide() -> AXUIElementRef;
    fn AXUIElementCreateApplication(pid: i32) -> AXUIElementRef;
    fn AXUIElementCopyAttributeValue(
        element: AXUIElementRef,
        attribute: core_foundation::string::CFStringRef,
        value: *mut core_foundation::base::CFTypeRef,
    ) -> i32;
    fn AXUIElementCopyAttributeNames(
        element: AXUIElementRef,
        names: *mut core_foundation::base::CFTypeRef,
    ) -> i32;
    fn AXUIElementCopyElementAtPosition(
        application: AXUIElementRef,
        x: f32,
        y: f32,
        element: *mut AXUIElementRef,
    ) -> i32;
}

type AXUIElementRef = core_foundation::base::CFTypeRef;

const AX_ERROR_SUCCESS: i32 = 0;
const ELEMENT_TIMEOUT: Duration = Duration::from_millis(100);
const WALK_TIMEOUT: Duration = Duration::from_secs(30);

/// Text extracted from a single UI element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessibilityElement {
    pub role: Option<String>,
    pub title: Option<String>,
    pub value: Option<String>,
    pub description: Option<String>,
}

/// Result of walking the accessibility tree for the focused application.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessibilityResult {
    pub app_name: String,
    pub window_title: Option<String>,
    /// For browser apps, the tab title extracted from the window title
    /// (with browser name suffix stripped). None for non-browser apps.
    pub browser_tab: Option<String>,
    pub elements: Vec<AccessibilityElement>,
    pub full_text: String,
    pub truncated: bool,
}

/// Known browser app names — used to detect browsers and strip suffixes from window titles.
const BROWSER_SUFFIXES: &[&str] = &[
    " - Google Chrome",
    " - Brave",
    " - Brave Browser",
    " - Mozilla Firefox",
    " - Firefox",
    " - Microsoft Edge",
    " - Chromium",
    " - Opera",
    " - Vivaldi",
    " — Arc", // Arc uses em dash
    " - Arc",
    " - Orion",
    " - Safari", // Safari Technology Preview
];

const BROWSER_APPS: &[&str] = &[
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

/// Extract the browser tab title from a window title by stripping the browser name suffix.
/// Safari doesn't append its name, so for Safari the window title IS the tab title.
fn extract_browser_tab(app_name: &str, window_title: &str) -> String {
    for suffix in BROWSER_SUFFIXES {
        if window_title.ends_with(suffix) {
            return window_title[..window_title.len() - suffix.len()].to_string();
        }
    }
    // Safari and some others don't append their name — the window title is the tab title
    window_title.to_string()
}

/// Roles to skip when walking the tree (decorative, chrome, or slow).
/// Skipping a role also skips its entire subtree.
const SKIP_ROLES: &[&str] = &[
    "AXScrollBar",
    "AXImage",
    "AXToolbar",
    "AXSecureTextField",
    "AXProgressIndicator",
    "AXSplitter",
    "AXGrowArea",
    // Menus and menu items (context menus, menu bar, tab context menus)
    "AXMenu",
    "AXMenuBar",
    "AXMenuBarItem",
    "AXMenuItem",
    // Browser/app chrome
    "AXTabGroup",
    "AXBanner",     // site nav banners
    "AXNavigation", // nav bars
];

/// Text content to filter out — common browser UI strings that aren't page content.
/// Matched as exact lines after trimming.
const SKIP_TEXT_PREFIXES: &[&str] = &[
    "New Tab",
    "Close Tab",
    "Reload Tab",
    "Mute Tab",
    "Pin Tab",
    "Unload Tab",
    "Duplicate Tab",
    "Bookmark Tab",
    "Move Tab",
    "Select All Tabs",
    "Close Multiple Tabs",
    "Close Duplicate Tabs",
    "Reopen Closed Tab",
    "Open Link in",
    "Open Image in",
    "Save Link As",
    "Copy Link",
    "Copy Clean Link",
    "Send Link to",
    "Send to Device",
    "Share",
    "Add Tab to",
    "Open in New Container",
    "Exit Full Screen",
    "Enter Full Screen",
    "Bookmark Link",
];

/// Get the string value of an AX attribute.
unsafe fn get_ax_string(element: AXUIElementRef, attr: &str) -> Option<String> {
    let cf_attr = CFString::new(attr);
    let mut value: core_foundation::base::CFTypeRef = std::ptr::null();
    let result = AXUIElementCopyAttributeValue(element, cf_attr.as_concrete_TypeRef(), &mut value);
    if result != AX_ERROR_SUCCESS || value.is_null() {
        return None;
    }
    let cf_type: CFType = TCFType::wrap_under_create_rule(value);
    // Try to convert to string
    let type_id = core_foundation::base::CFGetTypeID(value);
    let string_type_id = core_foundation::string::CFString::type_id();
    if type_id == string_type_id {
        let cf_string: CFString = TCFType::wrap_under_get_rule(value as _);
        Some(cf_string.to_string())
    } else {
        None
    }
}

/// Get the children of an AX element as a CFArray.
unsafe fn get_ax_children(element: AXUIElementRef) -> Vec<AXUIElementRef> {
    let cf_attr = CFString::new("AXChildren");
    let mut value: core_foundation::base::CFTypeRef = std::ptr::null();
    let result = AXUIElementCopyAttributeValue(element, cf_attr.as_concrete_TypeRef(), &mut value);
    if result != AX_ERROR_SUCCESS || value.is_null() {
        return Vec::new();
    }

    let type_id = core_foundation::base::CFGetTypeID(value);
    let array_type_id = core_foundation::array::CFArray::<CFType>::type_id();
    if type_id != array_type_id {
        core_foundation::base::CFRelease(value);
        return Vec::new();
    }

    let array: core_foundation::array::CFArray<CFType> =
        TCFType::wrap_under_create_rule(value as _);
    let count = array.len();
    let mut children = Vec::with_capacity(count as usize);
    for i in 0..count {
        let child = array.get(i).unwrap().as_CFTypeRef();
        core_foundation::base::CFRetain(child);
        children.push(child);
    }
    children
}

/// Walk the accessibility tree of the focused application and extract text.
/// Lightweight query to get just the focused app name without walking the tree.
/// Falls back to AppleScript if the accessibility API fails (e.g. on second monitors).
pub fn get_focused_app_name() -> Result<Option<String>> {
    let result: Option<String> = unsafe {
        let system_wide = AXUIElementCreateSystemWide();
        if system_wide.is_null() {
            None
        } else {
            let cf_attr = CFString::new("AXFocusedApplication");
            let mut value: core_foundation::base::CFTypeRef = std::ptr::null();
            let ax_result = AXUIElementCopyAttributeValue(
                system_wide,
                cf_attr.as_concrete_TypeRef(),
                &mut value,
            );
            core_foundation::base::CFRelease(system_wide);
            if ax_result != AX_ERROR_SUCCESS || value.is_null() {
                None
            } else {
                let app_name = get_ax_string(value, "AXTitle");
                core_foundation::base::CFRelease(value);
                app_name
            }
        }
    };

    if result.is_some() {
        return Ok(result);
    }

    // Fallback: use AppleScript to get the frontmost app
    get_frontmost_app_via_appscript()
}

/// Get the frontmost application name via AppleScript (reliable cross-monitor fallback).
/// Uses "displayed name" to get the proper capitalized app name (e.g. "Ghostty" not "ghostty").
fn get_frontmost_app_via_appscript() -> Result<Option<String>> {
    let output = std::process::Command::new("osascript")
        .args(["-e", "tell application \"System Events\" to get displayed name of first application process whose frontmost is true"])
        .output();
    match output {
        Ok(out) if out.status.success() => {
            let name = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if name.is_empty() {
                Ok(None)
            } else {
                Ok(Some(name))
            }
        }
        _ => Ok(None),
    }
}

/// Get the frontmost application's PID via AppleScript.
fn get_frontmost_app_pid() -> Option<i32> {
    let output = std::process::Command::new("osascript")
        .args(["-e", "tell application \"System Events\" to get unix id of first application process whose frontmost is true"])
        .output()
        .ok()?;
    if output.status.success() {
        String::from_utf8_lossy(&output.stdout).trim().parse().ok()
    } else {
        None
    }
}

/// Lightweight query to get the focused window's title without walking the tree.
pub fn get_focused_window_title() -> Option<String> {
    unsafe {
        let system_wide = AXUIElementCreateSystemWide();
        if system_wide.is_null() {
            return None;
        }

        let focused_app_ref = {
            let cf_attr = CFString::new("AXFocusedApplication");
            let mut value: core_foundation::base::CFTypeRef = std::ptr::null();
            let result = AXUIElementCopyAttributeValue(
                system_wide,
                cf_attr.as_concrete_TypeRef(),
                &mut value,
            );
            core_foundation::base::CFRelease(system_wide);
            if result != AX_ERROR_SUCCESS || value.is_null() {
                return None;
            }
            value
        };

        let focused_window = {
            let cf_attr = CFString::new("AXFocusedWindow");
            let mut value: core_foundation::base::CFTypeRef = std::ptr::null();
            let result = AXUIElementCopyAttributeValue(
                focused_app_ref,
                cf_attr.as_concrete_TypeRef(),
                &mut value,
            );
            core_foundation::base::CFRelease(focused_app_ref);
            if result != AX_ERROR_SUCCESS || value.is_null() {
                return None;
            }
            value
        };

        let title = get_ax_string(focused_window, "AXTitle");
        core_foundation::base::CFRelease(focused_window);
        title
    }
}

pub fn capture_accessibility_text() -> Result<Option<AccessibilityResult>> {
    unsafe {
        let system_wide = AXUIElementCreateSystemWide();
        if system_wide.is_null() {
            return Ok(None);
        }

        // Get the focused application — try AX API first, fall back to PID-based approach
        let (focused_app_ref, app_name) = {
            let cf_attr = CFString::new("AXFocusedApplication");
            let mut value: core_foundation::base::CFTypeRef = std::ptr::null();
            let result = AXUIElementCopyAttributeValue(
                system_wide,
                cf_attr.as_concrete_TypeRef(),
                &mut value,
            );
            core_foundation::base::CFRelease(system_wide);
            if result == AX_ERROR_SUCCESS && !value.is_null() {
                let name = get_ax_string(value, "AXTitle").unwrap_or_default();
                (value, name)
            } else {
                // AXFocusedApplication failed — fall back to getting PID via AppleScript
                // and creating an AXUIElement directly from it
                debug!(
                    "AXFocusedApplication failed (error {}), trying PID fallback",
                    result
                );
                let pid = match get_frontmost_app_pid() {
                    Some(pid) => pid,
                    None => return Ok(None),
                };
                let app_ref = AXUIElementCreateApplication(pid);
                if app_ref.is_null() {
                    return Ok(None);
                }
                let name = get_ax_string(app_ref, "AXTitle")
                    .or_else(|| get_frontmost_app_via_appscript().ok().flatten())
                    .unwrap_or_default();
                (app_ref, name)
            }
        };

        // Get focused window
        let focused_window = {
            let cf_attr = CFString::new("AXFocusedWindow");
            let mut value: core_foundation::base::CFTypeRef = std::ptr::null();
            let result = AXUIElementCopyAttributeValue(
                focused_app_ref,
                cf_attr.as_concrete_TypeRef(),
                &mut value,
            );
            if result == AX_ERROR_SUCCESS && !value.is_null() {
                Some(value)
            } else {
                None
            }
        };

        let window_title = focused_window.and_then(|w| get_ax_string(w, "AXTitle"));

        // For browser apps, extract the tab title from the window title
        let is_browser = BROWSER_APPS.iter().any(|b| app_name.contains(b));
        let browser_tab = if is_browser {
            window_title
                .as_ref()
                .map(|wt| extract_browser_tab(&app_name, wt))
        } else {
            None
        };

        // Walk the tree from the focused window (or app if no window)
        let root = focused_window.unwrap_or(focused_app_ref);
        let start = Instant::now();
        let mut elements = Vec::new();
        let mut text_parts = Vec::new();
        let mut truncated = false;

        walk_tree(
            root,
            &mut elements,
            &mut text_parts,
            &start,
            0,
            20, // max depth
            &mut truncated,
        );

        // Clean up
        if let Some(w) = focused_window {
            core_foundation::base::CFRelease(w);
        }
        core_foundation::base::CFRelease(focused_app_ref);

        let full_text = text_parts.join("\n");
        if full_text.trim().is_empty() {
            debug!(
                "Accessibility tree for '{}' yielded {} elements but no text (truncated: {})",
                app_name,
                elements.len(),
                truncated
            );
            return Ok(None);
        }

        Ok(Some(AccessibilityResult {
            app_name,
            window_title,
            browser_tab,
            elements,
            full_text,
            truncated,
        }))
    }
}

unsafe fn walk_tree(
    element: AXUIElementRef,
    elements: &mut Vec<AccessibilityElement>,
    text_parts: &mut Vec<String>,
    start: &Instant,
    depth: usize,
    max_depth: usize,
    truncated: &mut bool,
) {
    if start.elapsed() > WALK_TIMEOUT {
        *truncated = true;
        return;
    }
    if depth > max_depth {
        *truncated = true;
        return;
    }
    if elements.len() > 5000 {
        *truncated = true;
        return;
    }

    let role = get_ax_string(element, "AXRole");

    // Skip decorative roles
    if let Some(ref r) = role {
        if SKIP_ROLES.iter().any(|skip| r == skip) {
            return;
        }
    }

    let title = get_ax_string(element, "AXTitle");
    let value = get_ax_string(element, "AXValue");
    let description = get_ax_string(element, "AXDescription");

    // Collect text
    let text = match role.as_deref() {
        Some("AXStaticText") | Some("AXTextField") | Some("AXTextArea") => {
            value.clone().or_else(|| title.clone())
        }
        Some("AXButton") | Some("AXLink") | Some("AXTab") | Some("AXMenuItem") => {
            title.clone().or_else(|| description.clone())
        }
        _ => title.clone().or_else(|| value.clone()),
    };

    if let Some(ref t) = text {
        let trimmed = t.trim();
        if !trimmed.is_empty() {
            // Skip browser UI chrome text
            let is_chrome = SKIP_TEXT_PREFIXES.iter().any(|p| trimmed.starts_with(p));
            // Deduplicate consecutive identical text (parent element + child text node)
            let dominated = text_parts.last().map_or(false, |prev| prev == trimmed);
            if !is_chrome && !dominated {
                text_parts.push(trimmed.to_string());
            }
        }
    }

    if role.is_some() || title.is_some() || value.is_some() || description.is_some() {
        elements.push(AccessibilityElement {
            role,
            title,
            value,
            description,
        });
    }

    // Recurse into children
    let children = get_ax_children(element);
    for child in children {
        walk_tree(
            child,
            elements,
            text_parts,
            start,
            depth + 1,
            max_depth,
            truncated,
        );
        core_foundation::base::CFRelease(child);
    }
}

/// Describes the UI element that was clicked.
#[derive(Debug, Clone)]
pub struct ClickTarget {
    pub app_name: String,
    pub role: Option<String>,
    pub title: Option<String>,
    pub description: Option<String>,
    pub value: Option<String>,
}

impl ClickTarget {
    /// Human-readable label for this click target.
    pub fn label(&self) -> String {
        // Prefer title, then description, then value, then role
        if let Some(ref t) = self.title {
            if !t.is_empty() {
                return t.clone();
            }
        }
        if let Some(ref d) = self.description {
            if !d.is_empty() {
                return d.clone();
            }
        }
        if let Some(ref v) = self.value {
            if !v.is_empty() {
                let preview: String = v.chars().take(80).collect();
                return preview;
            }
        }
        self.role.clone().unwrap_or_else(|| "unknown".into())
    }
}

/// Get the accessibility element at a screen coordinate.
/// Uses the focused app's AX ref to hit-test. Returns info about what was clicked.
pub fn get_click_target(x: f64, y: f64) -> Option<ClickTarget> {
    unsafe {
        let system_wide = AXUIElementCreateSystemWide();
        if system_wide.is_null() {
            return None;
        }

        // Get focused application
        let cf_attr = CFString::new("AXFocusedApplication");
        let mut app_value: core_foundation::base::CFTypeRef = std::ptr::null();
        let result = AXUIElementCopyAttributeValue(
            system_wide,
            cf_attr.as_concrete_TypeRef(),
            &mut app_value,
        );
        core_foundation::base::CFRelease(system_wide);
        if result != AX_ERROR_SUCCESS || app_value.is_null() {
            return None;
        }

        let app_name = get_ax_string(app_value, "AXTitle").unwrap_or_default();

        // Hit-test at the click position
        let mut element: AXUIElementRef = std::ptr::null();
        let hit_result =
            AXUIElementCopyElementAtPosition(app_value, x as f32, y as f32, &mut element);
        core_foundation::base::CFRelease(app_value);

        if hit_result != AX_ERROR_SUCCESS || element.is_null() {
            return None;
        }

        let role = get_ax_string(element, "AXRole");
        let title = get_ax_string(element, "AXTitle");
        let description = get_ax_string(element, "AXDescription");
        let value = get_ax_string(element, "AXValue");
        core_foundation::base::CFRelease(element);

        Some(ClickTarget {
            app_name,
            role,
            title,
            description,
            value,
        })
    }
}

//! Native macOS viewer window for browsing summaries and running smart queries.

use objc2::rc::Retained;
use objc2::MainThreadMarker;
use objc2_app_kit::*;
use objc2_foundation::*;

/// Retained references to viewer UI elements.
pub struct ViewerWindow {
    pub window: Retained<NSWindow>,
    pub text_view: Retained<NSTextView>,
    pub segment_control: Retained<NSSegmentedControl>,
    pub query_field: Retained<NSTextField>,
    pub ask_button: Retained<NSButton>,
    pub spinner: Retained<NSProgressIndicator>,
    pub _scroll_view: Retained<NSScrollView>,
}

/// Create the viewer window with all UI elements.
pub fn create_viewer_window(
    target: &objc2::runtime::AnyObject,
    mtm: MainThreadMarker,
) -> ViewerWindow {
    let window_rect = NSRect::new(NSPoint::new(200.0, 200.0), NSSize::new(800.0, 600.0));

    let style = NSWindowStyleMask::Titled
        .union(NSWindowStyleMask::Closable)
        .union(NSWindowStyleMask::Resizable)
        .union(NSWindowStyleMask::Miniaturizable);

    let window = unsafe {
        NSWindow::initWithContentRect_styleMask_backing_defer(
            mtm.alloc(),
            window_rect,
            style,
            NSBackingStoreType::Buffered,
            false,
        )
    };
    window.setTitle(&NSString::from_str("ScreenTrack"));
    window.center();
    unsafe { window.setReleasedWhenClosed(false) };

    let content_view = window.contentView().unwrap();
    let content_frame = content_view.frame();
    let w = content_frame.size.width;
    let h = content_frame.size.height;

    // --- Segmented control at top ---
    let seg_height = 30.0;
    let seg_y = h - seg_height - 8.0;
    let seg_rect = NSRect::new(NSPoint::new(10.0, seg_y), NSSize::new(w - 20.0, seg_height));
    let segment_control = NSSegmentedControl::initWithFrame(mtm.alloc(), seg_rect);
    segment_control.setSegmentCount(2);
    unsafe {
        segment_control.setLabel_forSegment(&NSString::from_str("Smart Query"), 0);
        segment_control.setLabel_forSegment(&NSString::from_str("Config"), 1);
    }
    segment_control.setSelectedSegment(0); // default to smart query
    unsafe {
        segment_control.setTarget(Some(target));
        segment_control.setAction(Some(objc2::sel!(segmentChanged:)));
    }
    // Autoresizing: flexible width, stick to top
    segment_control.setAutoresizingMask(
        NSAutoresizingMaskOptions::ViewWidthSizable | NSAutoresizingMaskOptions::ViewMinYMargin,
    );
    content_view.addSubview(&segment_control);

    // --- Query bar at bottom ---
    let bar_height = 36.0;
    let bar_y = 4.0;

    let spinner = NSProgressIndicator::initWithFrame(
        mtm.alloc(),
        NSRect::new(NSPoint::new(w - 36.0, bar_y + 6.0), NSSize::new(24.0, 24.0)),
    );
    spinner.setStyle(NSProgressIndicatorStyle::Spinning);
    unsafe { spinner.setDisplayedWhenStopped(false) };
    spinner.setAutoresizingMask(NSAutoresizingMaskOptions::ViewMinXMargin);
    content_view.addSubview(&spinner);

    let ask_button = unsafe {
        NSButton::buttonWithTitle_target_action(
            &NSString::from_str("Ask"),
            Some(target),
            Some(objc2::sel!(askQuery:)),
            mtm,
        )
    };
    let btn_width = 60.0;
    ask_button.setFrame(NSRect::new(
        NSPoint::new(w - btn_width - 40.0, bar_y + 2.0),
        NSSize::new(btn_width, bar_height - 4.0),
    ));
    ask_button.setAutoresizingMask(NSAutoresizingMaskOptions::ViewMinXMargin);
    content_view.addSubview(&ask_button);

    let field_width = w - btn_width - 60.0;
    let query_field = NSTextField::initWithFrame(
        mtm.alloc(),
        NSRect::new(
            NSPoint::new(10.0, bar_y + 4.0),
            NSSize::new(field_width, 26.0),
        ),
    );
    query_field.setPlaceholderString(Some(&NSString::from_str("Ask about your activity...")));
    query_field.setAutoresizingMask(NSAutoresizingMaskOptions::ViewWidthSizable);
    unsafe {
        query_field.setTarget(Some(target));
        query_field.setAction(Some(objc2::sel!(askQuery:)));
    }
    content_view.addSubview(&query_field);

    // --- Scroll view + text view in the middle ---
    let scroll_y = bar_height + 4.0;
    let scroll_height = seg_y - scroll_y - 8.0;
    let scroll_rect = NSRect::new(NSPoint::new(0.0, scroll_y), NSSize::new(w, scroll_height));

    let scroll_view = NSScrollView::initWithFrame(mtm.alloc(), scroll_rect);
    scroll_view.setHasVerticalScroller(true);
    scroll_view.setHasHorizontalScroller(false);
    scroll_view.setAutoresizingMask(
        NSAutoresizingMaskOptions::ViewWidthSizable | NSAutoresizingMaskOptions::ViewHeightSizable,
    );

    let text_rect = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(w, scroll_height));
    let text_view = NSTextView::initWithFrame(mtm.alloc(), text_rect);
    text_view.setEditable(false);
    text_view.setSelectable(true);
    unsafe {
        text_view.setTextContainerInset(NSSize::new(10.0, 10.0));
    }
    // Make text view resize with scroll view
    text_view.setAutoresizingMask(NSAutoresizingMaskOptions::ViewWidthSizable);
    // Allow text to wrap
    if let Some(container) = unsafe { text_view.textContainer() } {
        unsafe {
            container.setWidthTracksTextView(true);
        }
    }

    scroll_view.setDocumentView(Some(&text_view));
    content_view.addSubview(&scroll_view);

    // Set initial placeholder text
    let initial = crate::md_attributed::markdown_to_attributed_string(
        "## ScreenTrack Viewer\n\nSelect a tab above to browse summaries, or type a question below.",
        mtm,
    );
    if let Some(storage) = unsafe { text_view.textStorage() } {
        storage.setAttributedString(&initial);
    }

    ViewerWindow {
        window,
        text_view,
        segment_control,
        query_field,
        ask_button,
        spinner,
        _scroll_view: scroll_view,
    }
}

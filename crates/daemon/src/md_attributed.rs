//! Convert Markdown text to NSAttributedString for rich text display in AppKit.
//! Parallel to md_render.rs but targets NSAttributedString instead of ANSI terminal codes.

use objc2::rc::Retained;
use objc2::{AllocAnyThread, MainThreadMarker};
use objc2_app_kit::{NSColor, NSFont};
use objc2_foundation::{NSMutableAttributedString, NSString};
use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag, TagEnd};

/// Convert a markdown string to a styled NSMutableAttributedString.
pub fn markdown_to_attributed_string(
    markdown: &str,
    _mtm: MainThreadMarker,
) -> Retained<NSMutableAttributedString> {
    let result = NSMutableAttributedString::new();

    let opts = Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TABLES;
    let parser = Parser::new_ext(markdown, opts);

    let body_font = NSFont::systemFontOfSize(13.0);
    let body_color = NSColor::labelColor();

    // State tracking
    let mut heading_level: Option<HeadingLevel> = None;
    let mut in_strong = false;
    let mut in_emphasis = false;
    let mut in_code_block = false;
    let mut in_inline_code = false;
    let mut list_depth: usize = 0;
    let mut list_ordered_counter: Vec<Option<u64>> = Vec::new(); // None = unordered, Some(n) = ordered
    let mut in_block_quote = false;

    for event in parser {
        match event {
            Event::Start(tag) => match tag {
                Tag::Heading { level, .. } => {
                    heading_level = Some(level);
                }
                Tag::Strong => in_strong = true,
                Tag::Emphasis => in_emphasis = true,
                Tag::CodeBlock(_) => {
                    in_code_block = true;
                    append_text(&result, "\n", &mono_font(12.0), &secondary_color());
                }
                Tag::List(start) => {
                    list_depth += 1;
                    list_ordered_counter.push(start);
                }
                Tag::Item => {
                    let indent = "  ".repeat(list_depth);
                    let bullet = if let Some(counter) = list_ordered_counter.last_mut() {
                        if let Some(n) = counter {
                            let b = format!("{indent}{n}. ");
                            *n += 1;
                            b
                        } else {
                            format!("{indent}  \u{2022} ")
                        }
                    } else {
                        format!("{indent}\u{2022} ")
                    };
                    append_text(&result, &bullet, &body_font, &secondary_color());
                }
                Tag::BlockQuote(_) => {
                    in_block_quote = true;
                    append_text(&result, "\u{2502} ", &body_font, &secondary_color());
                }
                Tag::Paragraph => {}
                Tag::Link { dest_url, .. } => {
                    // We'll just show the text, not the URL
                    let _ = dest_url;
                }
                _ => {}
            },
            Event::End(tag_end) => match tag_end {
                TagEnd::Heading(_) => {
                    append_text(&result, "\n\n", &body_font, &body_color);
                    heading_level = None;
                }
                TagEnd::Strong => in_strong = false,
                TagEnd::Emphasis => in_emphasis = false,
                TagEnd::CodeBlock => {
                    in_code_block = false;
                    append_text(&result, "\n", &body_font, &body_color);
                }
                TagEnd::List(_) => {
                    list_depth = list_depth.saturating_sub(1);
                    list_ordered_counter.pop();
                    if list_depth == 0 {
                        append_text(&result, "\n", &body_font, &body_color);
                    }
                }
                TagEnd::Item => {
                    append_text(&result, "\n", &body_font, &body_color);
                }
                TagEnd::BlockQuote(_) => {
                    in_block_quote = false;
                    append_text(&result, "\n", &body_font, &body_color);
                }
                TagEnd::Paragraph => {
                    append_text(&result, "\n\n", &body_font, &body_color);
                }
                _ => {}
            },
            Event::Text(text) => {
                let (font, color) = resolve_style(
                    &heading_level,
                    in_strong,
                    in_emphasis,
                    in_code_block,
                    in_inline_code,
                    in_block_quote,
                );
                append_text(&result, &text, &font, &color);
            }
            Event::Code(code) => {
                // Inline code
                in_inline_code = true;
                let font = mono_font(12.0);
                let color = inline_code_color();
                append_text(&result, &format!("`{code}`"), &font, &color);
                in_inline_code = false;
            }
            Event::SoftBreak => {
                append_text(&result, " ", &body_font, &body_color);
            }
            Event::HardBreak => {
                append_text(&result, "\n", &body_font, &body_color);
            }
            Event::Rule => {
                let rule = "\u{2500}".repeat(50);
                append_text(
                    &result,
                    &format!("\n{rule}\n\n"),
                    &body_font,
                    &secondary_color(),
                );
            }
            _ => {}
        }
    }

    result
}

fn resolve_style(
    heading: &Option<HeadingLevel>,
    strong: bool,
    _emphasis: bool,
    code_block: bool,
    _inline_code: bool,
    _block_quote: bool,
) -> (Retained<NSFont>, Retained<NSColor>) {
    if code_block {
        return (mono_font(12.0), secondary_color());
    }

    if let Some(level) = heading {
        return match level {
            HeadingLevel::H1 => (bold_font(20.0), NSColor::systemBlueColor()),
            HeadingLevel::H2 => (bold_font(16.0), NSColor::labelColor()),
            HeadingLevel::H3 => (bold_font(14.0), NSColor::secondaryLabelColor()),
            _ => (bold_font(13.0), NSColor::secondaryLabelColor()),
        };
    }

    if strong {
        return (bold_font(13.0), NSColor::labelColor());
    }

    (NSFont::systemFontOfSize(13.0), NSColor::labelColor())
}

fn bold_font(size: f64) -> Retained<NSFont> {
    NSFont::boldSystemFontOfSize(size)
}

fn mono_font(size: f64) -> Retained<NSFont> {
    unsafe {
        NSFont::monospacedSystemFontOfSize_weight(size, 0.0) // NSFontWeightRegular = 0.0
    }
}

fn secondary_color() -> Retained<NSColor> {
    NSColor::secondaryLabelColor()
}

fn inline_code_color() -> Retained<NSColor> {
    NSColor::systemOrangeColor()
}

/// Append styled text to an NSMutableAttributedString.
fn append_text(result: &NSMutableAttributedString, text: &str, font: &NSFont, color: &NSColor) {
    if text.is_empty() {
        return;
    }

    unsafe {
        let ns_text = NSString::from_str(text);
        let piece =
            NSMutableAttributedString::initWithString(NSMutableAttributedString::alloc(), &ns_text);

        let range = objc2_foundation::NSRange::new(0, ns_text.len());

        // Set font attribute
        let font_key: &NSString = objc2::msg_send![
            objc2::class!(NSString),
            stringWithUTF8String: c"NSFont".as_ptr()
        ];
        piece.addAttribute_value_range(font_key, font, range);

        // Set color attribute
        let color_key: &NSString = objc2::msg_send![
            objc2::class!(NSString),
            stringWithUTF8String: c"NSColor".as_ptr()
        ];
        piece.addAttribute_value_range(color_key, color, range);

        result.appendAttributedString(&piece);
    }
}

//! Lightweight terminal Markdown renderer using pulldown-cmark.
//!
//! Converts GitHub-Flavored Markdown to ANSI-colored terminal output.
//! Handles headers, bold, italic, code, lists, blockquotes, and horizontal rules.

use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag, TagEnd};

// ANSI escape codes
const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const ITALIC: &str = "\x1b[3m";
const UNDERLINE: &str = "\x1b[4m";
const STRIKETHROUGH: &str = "\x1b[9m";

// Colors
const CYAN: &str = "\x1b[36m";
const YELLOW: &str = "\x1b[33m";
const _GREEN: &str = "\x1b[32m";
const MAGENTA: &str = "\x1b[35m";
const BRIGHT_BLACK: &str = "\x1b[90m"; // dim gray

/// Render a Markdown string to ANSI-formatted terminal text.
pub fn render(markdown: &str) -> String {
    let options = Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TABLES;
    let parser = Parser::new_ext(markdown, options);

    let mut out = String::new();
    let mut list_stack: Vec<Option<u64>> = Vec::new(); // None = unordered, Some(n) = ordered at n
    let mut in_heading = false;
    let mut heading_level = HeadingLevel::H1;
    let mut in_code_block = false;
    let mut in_block_quote = false;
    let mut pending_newline = false;

    for event in parser {
        match event {
            // --- Block-level ---
            Event::Start(Tag::Heading { level, .. }) => {
                if pending_newline {
                    out.push('\n');
                    pending_newline = false;
                }
                in_heading = true;
                heading_level = level;
                match level {
                    HeadingLevel::H1 => out.push_str(&format!("{BOLD}{UNDERLINE}{CYAN}")),
                    HeadingLevel::H2 => out.push_str(&format!("{BOLD}{CYAN}")),
                    HeadingLevel::H3 => out.push_str(&format!("{BOLD}{YELLOW}")),
                    _ => out.push_str(BOLD),
                }
            }
            Event::End(TagEnd::Heading(_)) => {
                out.push_str(RESET);
                out.push('\n');
                // Add an extra blank line after H1/H2
                if matches!(heading_level, HeadingLevel::H1 | HeadingLevel::H2) {
                    out.push('\n');
                }
                in_heading = false;
                pending_newline = false;
            }

            Event::Start(Tag::Paragraph) => {
                if pending_newline {
                    out.push('\n');
                    pending_newline = false;
                }
                if in_block_quote {
                    out.push_str(&format!("{BRIGHT_BLACK}│{RESET} "));
                }
            }
            Event::End(TagEnd::Paragraph) => {
                out.push('\n');
                pending_newline = true;
            }

            Event::Start(Tag::BlockQuote(_)) => {
                in_block_quote = true;
                if pending_newline {
                    out.push('\n');
                    pending_newline = false;
                }
            }
            Event::End(TagEnd::BlockQuote(_)) => {
                in_block_quote = false;
                pending_newline = true;
            }

            Event::Start(Tag::List(first_num)) => {
                if pending_newline && list_stack.is_empty() {
                    // Don't add blank line if we're starting a nested list
                    out.push('\n');
                }
                pending_newline = false;
                list_stack.push(first_num);
            }
            Event::End(TagEnd::List(_)) => {
                list_stack.pop();
                if list_stack.is_empty() {
                    pending_newline = true;
                }
            }

            Event::Start(Tag::Item) => {
                let _ = pending_newline;
                pending_newline = false;
                let depth = list_stack.len().saturating_sub(1);
                let indent = "  ".repeat(depth);

                if let Some(current) = list_stack.last_mut() {
                    match current {
                        Some(n) => {
                            out.push_str(&format!("{indent}  {BRIGHT_BLACK}{n}.{RESET} "));
                            *n += 1;
                        }
                        None => {
                            out.push_str(&format!("{indent}  {BRIGHT_BLACK}•{RESET} "));
                        }
                    }
                }
            }
            Event::End(TagEnd::Item) => {
                // Newline is already added by paragraph end or text
                if !out.ends_with('\n') {
                    out.push('\n');
                }
            }

            Event::Start(Tag::CodeBlock(_)) => {
                if pending_newline {
                    out.push('\n');
                    pending_newline = false;
                }
                in_code_block = true;
                out.push_str(&format!("{DIM}"));
            }
            Event::End(TagEnd::CodeBlock) => {
                out.push_str(RESET);
                in_code_block = false;
                pending_newline = true;
            }

            // --- Inline ---
            Event::Start(Tag::Strong) => {
                out.push_str(BOLD);
            }
            Event::End(TagEnd::Strong) => {
                out.push_str(RESET);
                // Restore any outer formatting context
                if in_heading {
                    match heading_level {
                        HeadingLevel::H1 => out.push_str(&format!("{BOLD}{UNDERLINE}{CYAN}")),
                        HeadingLevel::H2 => out.push_str(&format!("{BOLD}{CYAN}")),
                        HeadingLevel::H3 => out.push_str(&format!("{BOLD}{YELLOW}")),
                        _ => out.push_str(BOLD),
                    }
                }
            }

            Event::Start(Tag::Emphasis) => {
                out.push_str(ITALIC);
            }
            Event::End(TagEnd::Emphasis) => {
                out.push_str(RESET);
                if in_heading {
                    match heading_level {
                        HeadingLevel::H1 => out.push_str(&format!("{BOLD}{UNDERLINE}{CYAN}")),
                        HeadingLevel::H2 => out.push_str(&format!("{BOLD}{CYAN}")),
                        HeadingLevel::H3 => out.push_str(&format!("{BOLD}{YELLOW}")),
                        _ => out.push_str(BOLD),
                    }
                }
            }

            Event::Start(Tag::Strikethrough) => {
                out.push_str(STRIKETHROUGH);
            }
            Event::End(TagEnd::Strikethrough) => {
                out.push_str(RESET);
            }

            Event::Start(Tag::Link { .. }) => {
                out.push_str(&format!("{UNDERLINE}{CYAN}"));
            }
            Event::End(TagEnd::Link) => {
                out.push_str(RESET);
            }

            Event::Code(code) => {
                out.push_str(&format!("{MAGENTA}`{code}`{RESET}"));
            }

            Event::Text(text) => {
                if in_code_block {
                    // Indent code block lines
                    for line in text.lines() {
                        out.push_str("    ");
                        out.push_str(line);
                        out.push('\n');
                    }
                } else {
                    out.push_str(&text);
                }
            }

            Event::SoftBreak => {
                out.push('\n');
                if in_block_quote {
                    out.push_str(&format!("{BRIGHT_BLACK}│{RESET} "));
                }
            }
            Event::HardBreak => {
                out.push('\n');
            }

            Event::Rule => {
                if pending_newline {
                    out.push('\n');
                }
                out.push_str(&format!("{BRIGHT_BLACK}{}{RESET}\n", "─".repeat(60)));
                pending_newline = true;
            }

            // Tables — simple rendering
            Event::Start(Tag::Table(_)) => {
                if pending_newline {
                    out.push('\n');
                    pending_newline = false;
                }
            }
            Event::End(TagEnd::Table) => {
                pending_newline = true;
            }
            Event::Start(Tag::TableHead) => {
                out.push_str(BOLD);
            }
            Event::End(TagEnd::TableHead) => {
                out.push_str(RESET);
                out.push('\n');
            }
            Event::Start(Tag::TableRow) => {}
            Event::End(TagEnd::TableRow) => {
                out.push('\n');
            }
            Event::Start(Tag::TableCell) => {
                out.push_str("  ");
            }
            Event::End(TagEnd::TableCell) => {
                out.push_str("  ");
            }

            _ => {}
        }
    }

    // Trim trailing blank lines but keep one final newline
    while out.ends_with("\n\n") {
        out.pop();
    }
    if !out.ends_with('\n') {
        out.push('\n');
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bold_in_list() {
        let md = "* Item with **bold** text\n* Another item\n";
        let rendered = render(md);
        // Should contain ANSI bold sequence
        assert!(rendered.contains(BOLD));
        assert!(rendered.contains("bold"));
        assert!(rendered.contains("•"));
    }

    #[test]
    fn test_heading_colors() {
        let md = "# Title\n\nSome text\n";
        let rendered = render(md);
        assert!(rendered.contains(CYAN));
        assert!(rendered.contains("Title"));
    }

    #[test]
    fn test_inline_code() {
        let md = "Use `cargo build` to compile.\n";
        let rendered = render(md);
        assert!(rendered.contains(MAGENTA));
        assert!(rendered.contains("`cargo build`"));
    }

    #[test]
    fn test_nested_list() {
        let md = "* Top\n  * Nested\n  * Nested 2\n* Top 2\n";
        let rendered = render(md);
        assert!(rendered.contains("Top"));
        assert!(rendered.contains("Nested"));
    }
}

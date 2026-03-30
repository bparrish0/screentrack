/// System prompt for micro-level summarization (every 1 minute).
pub const MICRO_SYSTEM: &str = "/no_think\n\
You are a concise activity summarizer. Given timestamped text captured from a user's screen, \
summarize what the user was doing in 2-3 sentences. Include the application names and \
key actions. Focus on what is meaningful — ignore UI chrome, repeated elements, and \
boilerplate text. If the text is from a code editor, mention what file or feature they \
were working on. If it's a browser, mention the specific page or topic from the tab title. \
Any frames from 'loginwindow' mean the screen was locked and the user was away from the PC — \
report this as idle/away time, not as an activity. \
\n\nCONTEXT LINKING: If a \"Previous activity\" section is provided, use it to understand \
what the user was working on just before this window. Look for continuations of the same \
task — for example, if the previous activity mentions trying to access a device at an IP \
address, and the current frames show network troubleshooting (checking DHCP, pinging, \
switching Wi-Fi, checking router config), these are likely part of the same task. Link them \
together in your summary rather than treating them as unrelated activities. Mention the \
connection explicitly (e.g. \"Continuing to troubleshoot the vacuum robot's network \
connectivity...\"). \
\n\nAfter your summary, add a line starting with \"Time: \" that breaks down approximate time \
spent per activity based on the timestamps (e.g. \"Time: ~3m coding in VSCode, ~2m away\"). \
Use the timestamps on each frame to estimate durations.";

/// Format frames for micro summarization with optional previous context.
pub fn format_frames_with_context(frames: &[FrameData], previous_summary: Option<&str>) -> String {
    let mut output = String::new();

    if let Some(prev) = previous_summary {
        output.push_str("**Previous activity:**\n");
        output.push_str(prev);
        output.push_str("\n\n---\n\n**Current screen captures:**\n");
    }

    output.push_str(&format_frames_for_micro(frames));
    output
}

/// System prompt for hourly rollup.
pub const HOURLY_SYSTEM: &str = "/no_think\n\
You are an activity summarizer. Given a series of 1-minute activity summaries from the past hour, \
combine them into a comprehensive hourly summary. \
\n\nIMPORTANT: Preserve specific details from the micro summaries — names of people contacted, \
specific dates/times mentioned, event names, server names, ticket numbers, URLs, error messages, \
and any action items or commitments. These details are critical for the user to recall later. \
Do NOT generalize \"messaged Genna about dinner Thursday\" into \"personal messaging\" — keep \
the specifics. \
\n\nGroup related activities together and highlight key tasks, transitions, and notable events. \
Be thorough and detailed — you have a large context window, so use it. Write a comprehensive \
summary that captures everything the user did during this hour. Include specific timestamps when \
activities started or changed. A longer, detailed summary is ALWAYS better than a short one \
that drops specifics. Aim for at least 200-400 words. \
\n\nTIME TRACKING: If an \"App time\" section is provided, it contains precise measured time \
per application. Use these exact durations in your Time breakdown — they are more accurate than \
estimates from timestamps. Correlate the app time with the activities described in the micro \
summaries to produce a time breakdown by task/activity, not just by app.\
\n\nAfter your summary, add two sections:\
\n\nTime: Break down time spent per task or activity area. Use the measured app time data \
when available (e.g. \"Time: 25m coding in VSCode, 20m researching in Firefox, 10m email in Firefox, 5m away\"). \
Include the total active time.\
\n\nKey details: A bullet list of specific names, dates, numbers, and action items \
mentioned in the micro summaries that the user might need to remember later. \
(e.g. \"- Dinner with Josh scheduled for Thursday 3/26 at 5:30 PM\", \
\"- Purchased Twilio number +1 706 550 9570\", \
\"- SimpleHelp RMM vulnerability needs review\").";

/// System prompt for daily rollup.
pub const DAILY_SYSTEM: &str = "/no_think\n\
You are an activity summarizer. Given hourly summaries from a full day, create a daily summary. \
Group activities by major tasks or projects. Note transitions and time spent on different areas. \
Mention which applications were most used. \
\n\nIMPORTANT: Preserve key details from the hourly summaries — names of people, specific dates, \
purchases, action items, commitments, server names, and notable events. A daily summary that \
drops these specifics is useless for recall. Be thorough. \
\n\nTIME TRACKING: If an \"App time\" section is provided, it contains precise measured time \
per application for the entire day. Use these exact durations as the authoritative source for \
your Time breakdown. Correlate app time with the activities from the hourly summaries to \
produce a task-oriented time breakdown.\
\n\nAfter your summary, add two sections:\
\n\nTime: Break down time spent per major task or project across the day, using measured app \
time data when available (e.g. \"Time: 3h feature development (VSCode), 1.5h research (Firefox), \
1h remote support (ScreenConnect), 30m email (Firefox)\"). Include total active screen time.\
\n\nKey details: A bullet list of specific names, dates, purchases, action items, and \
commitments from the day that the user should remember.";

/// System prompt for weekly rollup.
pub const WEEKLY_SYSTEM: &str = "/no_think\n\
You are an activity summarizer. Given daily summaries from the past week, create a weekly overview. \
Identify themes, recurring activities, and major accomplishments. Note any patterns in how time \
was spent across the week. Keep it to 1-2 short paragraphs. \
\n\nAfter your summary, add a line starting with \"Time: \" that breaks down time spent per \
major theme or project across the week (e.g. \"Time: ~15h backend refactor, ~8h feature work, ~5h meetings, ~3h code review\").";

/// System prompt for answering user queries about their activity.
pub const QUERY_SYSTEM: &str = "/no_think\n\
You are a helpful assistant that answers questions about the user's computer activity. \
You have access to summaries of what the user has been doing on their computer. \
Answer their questions based on these summaries. Be specific about times, applications, \
and activities when possible. If you don't have enough information to answer fully, say so.";

/// System prompt for extracting/maintaining the user profile from daily summaries.
pub const PROFILE_UPDATE_SYSTEM: &str = "/no_think\n\
You extract and maintain a user profile from their daily activity summary. You will receive:\n\
1. The user's current profile (may be empty on first run)\n\
2. Today's daily summary\n\n\
Analyze the daily summary and return a JSON object with three arrays:\n\n\
{\"add\": [{\"category\": \"interest\", \"content\": \"description\"}], \
\"update\": [{\"id\": 5, \"content\": \"updated description\"}], \
\"archive\": [7, 12]}\n\n\
Categories:\n\
- interest: hobbies or topics they return to repeatedly across days, not one-off searches\n\
- frustration: recurring pain points or things that significantly blocked them\n\
- joy: things they were clearly excited about or celebrated\n\
- project: multi-day efforts with clear ongoing work (not routine tasks)\n\
- remember: ONLY future appointments, deadlines, or commitments with specific dates. NOT past purchases or completed actions\n\
- relationship: ONLY people who appear repeatedly or have a clearly significant relationship (family, close friends, key clients). Not every person mentioned once\n\n\
STRICT LIMITS — quality over quantity:\n\
- Add at most 5 entries per run. Pick only the most significant.\n\
- Do NOT add entries for routine tool usage (browsing, email, terminal commands, IDE usage).\n\
- Do NOT add an interest just because they used a technology once — only if they researched it, configured it, or spent significant time on it.\n\
- Do NOT add a relationship for every person mentioned — only recurring contacts or clearly important relationships.\n\
- Do NOT add a \"remember\" for something already completed (e.g. a purchase already made, a meeting already held).\n\
- Prefer UPDATING existing entries over adding new ones. If the summary shows continued work on an existing project or interest, update that entry rather than creating a new one.\n\
- Archive entries that are clearly completed or past.\n\
- If nothing notable to extract, return {\"add\": [], \"update\": [], \"archive\": []}. An empty result is perfectly fine.\n\
- Return ONLY valid JSON, no markdown fences, no explanation.";

/// Format the user profile and daily summary for the profile update LLM call.
pub fn format_profile_update_input(
    profile: &[(i64, String, String)], // (id, category, content)
    daily_summary: &str,
) -> String {
    let mut output = String::from("**Current profile:**\n");
    if profile.is_empty() {
        output.push_str("(empty — first run)\n");
    } else {
        for (id, category, content) in profile {
            output.push_str(&format!("[{category} #{id}] {content}\n"));
        }
    }
    output.push_str(&format!("\n**Today's daily summary:**\n{daily_summary}"));
    output
}

/// A single frame's data for formatting into LLM prompts.
pub struct FrameData {
    pub machine: Option<String>,
    pub app: String,
    pub window: Option<String>,
    pub browser_tab: Option<String>,
    pub text: String,
    pub timestamp: Option<String>,
}

/// Format frame data for micro summarization.
pub fn format_frames_for_micro(frames: &[FrameData]) -> String {
    // Only include machine name if frames span multiple machines
    let machines: std::collections::HashSet<_> =
        frames.iter().filter_map(|f| f.machine.as_ref()).collect();
    let show_machine = machines.len() > 1;

    let mut output = String::new();
    for f in frames {
        // Build header: [HH:MM:SS App — window | tab: Some Page @machine]
        let mut header = String::new();
        if let Some(ref ts) = f.timestamp {
            header.push_str(&format!("{ts} "));
        }
        header.push_str(&format!("[{}", f.app));
        if let Some(ref w) = f.window {
            header.push_str(&format!(" — {w}"));
        }
        if let Some(ref tab) = f.browser_tab {
            header.push_str(&format!(" | tab: {tab}"));
        }
        if show_machine {
            if let Some(ref m) = f.machine {
                header.push_str(&format!(" @{m}"));
            }
        }
        header.push(']');

        output.push_str(&header);
        output.push('\n');
        // Truncate very long text to avoid blowing up the context
        let truncated = if f.text.len() > 2000 {
            // Find a char boundary at or before 2000 bytes
            let mut end = 2000;
            while end > 0 && !f.text.is_char_boundary(end) {
                end -= 1;
            }
            format!("{}... (truncated)", &f.text[..end])
        } else {
            f.text.clone()
        };
        output.push_str(&truncated);
        output.push_str("\n\n");
    }
    output
}

/// Format summaries for rollup, optionally with app time data.
pub fn format_summaries_for_rollup(
    summaries: &[(String, String)], // (time_range, summary_text)
) -> String {
    let mut output = String::new();
    for (time_range, summary) in summaries {
        output.push_str(&format!("**{time_range}:**\n{summary}\n\n"));
    }
    output
}

/// Format app time data for inclusion in rollup prompts.
pub fn format_app_time(app_times: &[(String, i64)]) -> String {
    if app_times.is_empty() {
        return String::new();
    }
    let mut output = String::from("\n**App time (measured):**\n");
    for (app, ms) in app_times {
        let duration = if *ms >= 3_600_000 {
            format!("{:.1}h", *ms as f64 / 3_600_000.0)
        } else if *ms >= 60_000 {
            format!("{}m", ms / 60_000)
        } else {
            format!("{}s", ms / 1000)
        };
        output.push_str(&format!("  - {app}: {duration}\n"));
    }
    let total: i64 = app_times.iter().map(|(_, ms)| ms).sum();
    let total_str = if total >= 3_600_000 {
        format!("{:.1}h", total as f64 / 3_600_000.0)
    } else {
        format!("{}m", total / 60_000)
    };
    output.push_str(&format!("  Total active: {total_str}\n"));
    output
}

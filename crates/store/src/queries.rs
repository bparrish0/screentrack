use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::FromRow;

use crate::Database;

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Frame {
    pub id: i64,
    pub machine_id: String,
    pub timestamp: i64,
    pub app_name: Option<String>,
    pub window_title: Option<String>,
    pub browser_tab: Option<String>,
    pub text_content: String,
    pub source: String,
    pub content_hash: Option<String>,
    pub simhash: Option<i64>,
    pub screenshot_path: Option<String>,
    pub synced: i64,
    pub summarized: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewFrame {
    pub app_name: Option<String>,
    pub window_title: Option<String>,
    pub browser_tab: Option<String>,
    pub text_content: String,
    pub source: String,
    pub screenshot_path: Option<String>,
}

/// Filter options for querying frames.
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct ProfileEntry {
    pub id: i64,
    pub category: String,
    pub content: String,
    pub first_seen: i64,
    pub last_seen: i64,
    pub status: String,
    pub source_summary_id: Option<i64>,
}

#[derive(Debug, Clone, Default)]
pub struct FrameFilter {
    pub start: Option<i64>,
    pub end: Option<i64>,
    pub app_name: Option<String>,
    pub browser_tab_contains: Option<String>,
    pub source: Option<String>,
    pub search_text: Option<String>,
    pub machine_id: Option<String>,
    pub limit: Option<i64>,
}

/// Capture statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CaptureStats {
    pub frames_total: i64,
    pub frames_accessibility: i64,
    pub frames_ocr_local: i64,
    pub frames_ocr_remote: i64,
    pub frames_skipped_unchanged: i64,
    pub frames_skipped_dedup: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Summary {
    pub id: i64,
    pub machine_id: String,
    pub tier: String,
    pub start_time: i64,
    pub end_time: i64,
    pub summary: String,
    pub apps_referenced: Option<String>,
    pub source_frame_ids: Option<String>,
    pub source_summary_ids: Option<String>,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewSummary {
    pub tier: String,
    pub start_time: i64,
    pub end_time: i64,
    pub summary: String,
    pub apps_referenced: Option<Vec<String>>,
    pub source_frame_ids: Option<Vec<i64>>,
    pub source_summary_ids: Option<Vec<i64>>,
}

const DEDUP_WINDOW_SECS: i64 = 45;

/// Sanitize a user search string for FTS5 MATCH.
/// Wraps each token in double quotes so special characters (hyphens, colons, etc.)
/// are treated as literals rather than FTS operators.
fn sanitize_fts_query(query: &str) -> String {
    let tokens: Vec<String> = query
        .split_whitespace()
        .map(|t| format!("\"{}\"", t.replace('"', "")))
        .collect();
    tokens.join(" ")
}
/// Maximum hamming distance between simhashes to consider frames near-duplicates.
const SIMHASH_THRESHOLD: u32 = 3;

fn compute_hash(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hex::encode(hasher.finalize())
}

/// Compute a 64-bit simhash fingerprint for fuzzy deduplication.
/// Similar texts produce fingerprints with few differing bits.
fn compute_simhash(text: &str) -> i64 {
    let mut counts = [0i32; 64];

    // Hash overlapping 3-word shingles
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return 0;
    }

    let shingle_size = 3.min(words.len());
    for window in words.windows(shingle_size) {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for w in window {
            std::hash::Hash::hash(w, &mut hasher);
        }
        let h = std::hash::Hasher::finish(&hasher);

        for i in 0..64 {
            if (h >> i) & 1 == 1 {
                counts[i] += 1;
            } else {
                counts[i] -= 1;
            }
        }
    }

    let mut fingerprint: u64 = 0;
    for i in 0..64 {
        if counts[i] > 0 {
            fingerprint |= 1 << i;
        }
    }
    fingerprint as i64
}

impl Database {
    /// Insert a frame, deduplicating against recent identical or near-identical content.
    /// Uses exact hash for identical content, and simhash (hamming distance) for fuzzy matching.
    /// Returns None if the frame was deduplicated (skipped).
    pub async fn insert_frame(&self, frame: NewFrame) -> Result<Option<i64>> {
        self.insert_frame_for_machine(&self.machine_id.clone(), frame, None)
            .await
    }

    /// Insert a frame for a specific machine (used by the server when receiving remote frames).
    /// Optionally accepts a pre-set timestamp (for remote frames).
    pub async fn insert_frame_for_machine(
        &self,
        machine_id: &str,
        frame: NewFrame,
        timestamp: Option<i64>,
    ) -> Result<Option<i64>> {
        let now = timestamp.unwrap_or_else(|| Utc::now().timestamp_millis());
        let hash = compute_hash(&frame.text_content);
        let simhash = compute_simhash(&frame.text_content);
        let dedup_cutoff = now - (DEDUP_WINDOW_SECS * 1000);

        // Check for exact duplicate (scoped to machine)
        let existing: Option<(i64,)> = sqlx::query_as(
            "SELECT id FROM frames
             WHERE machine_id = ? AND content_hash = ? AND app_name IS ? AND window_title IS ? AND timestamp > ?
             ORDER BY timestamp DESC LIMIT 1",
        )
        .bind(machine_id)
        .bind(&hash)
        .bind(&frame.app_name)
        .bind(&frame.window_title)
        .bind(dedup_cutoff)
        .fetch_optional(&self.pool)
        .await?;

        if existing.is_some() {
            self.increment_stat("frames_skipped_dedup").await?;
            return Ok(None);
        }

        // Check for near-duplicate via simhash hamming distance (scoped to machine).
        let recent: Option<(i64,)> = sqlx::query_as(
            "SELECT simhash FROM frames
             WHERE machine_id = ? AND app_name IS ? AND window_title IS ? AND simhash IS NOT NULL AND timestamp > ?
             ORDER BY timestamp DESC LIMIT 1",
        )
        .bind(machine_id)
        .bind(&frame.app_name)
        .bind(&frame.window_title)
        .bind(dedup_cutoff)
        .fetch_optional(&self.pool)
        .await?;

        if let Some((prev_simhash,)) = recent {
            let distance = (simhash ^ prev_simhash).count_ones();
            if distance <= SIMHASH_THRESHOLD {
                self.increment_stat("frames_skipped_dedup").await?;
                return Ok(None);
            }
        }

        let id = sqlx::query_scalar::<_, i64>(
            "INSERT INTO frames (machine_id, timestamp, app_name, window_title, browser_tab, text_content, source, content_hash, simhash, screenshot_path)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
             RETURNING id",
        )
        .bind(machine_id)
        .bind(now)
        .bind(&frame.app_name)
        .bind(&frame.window_title)
        .bind(&frame.browser_tab)
        .bind(&frame.text_content)
        .bind(&frame.source)
        .bind(&hash)
        .bind(simhash)
        .bind(&frame.screenshot_path)
        .fetch_one(&self.pool)
        .await?;

        Ok(Some(id))
    }

    /// Get frames in a time range.
    pub async fn get_frames(&self, start: i64, end: i64) -> Result<Vec<Frame>> {
        let frames = sqlx::query_as::<_, Frame>(
            "SELECT * FROM frames WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC",
        )
        .bind(start)
        .bind(end)
        .fetch_all(&self.pool)
        .await?;

        Ok(frames)
    }

    /// Get frames that haven't been included in any micro summary yet.
    pub async fn get_unsummarized_frames(&self, since: i64) -> Result<Vec<Frame>> {
        let frames = sqlx::query_as::<_, Frame>(
            "SELECT * FROM frames WHERE timestamp >= ? AND summarized = 0 ORDER BY timestamp ASC",
        )
        .bind(since)
        .fetch_all(&self.pool)
        .await?;

        Ok(frames)
    }

    /// Mark frames as summarized.
    pub async fn mark_frames_summarized(&self, machine_id: &str, ids: &[i64]) -> Result<()> {
        for id in ids {
            sqlx::query("UPDATE frames SET summarized = 1 WHERE id = ? AND machine_id = ?")
                .bind(id)
                .bind(machine_id)
                .execute(&self.pool)
                .await?;
        }
        Ok(())
    }

    /// Get frames that haven't been synced to the server yet.
    pub async fn get_unsynced_frames(&self, limit: i64) -> Result<Vec<Frame>> {
        let frames = sqlx::query_as::<_, Frame>(
            "SELECT * FROM frames WHERE synced = 0 ORDER BY timestamp ASC LIMIT ?",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(frames)
    }

    /// Mark frames as synced to the server.
    pub async fn mark_frames_synced(&self, ids: &[i64]) -> Result<()> {
        for id in ids {
            sqlx::query("UPDATE frames SET synced = 1 WHERE id = ? AND machine_id = ?")
                .bind(id)
                .bind(&self.machine_id)
                .execute(&self.pool)
                .await?;
        }
        Ok(())
    }

    /// Full-text search on frame content.
    pub async fn search_frames(&self, query: &str, limit: i64) -> Result<Vec<Frame>> {
        let escaped = sanitize_fts_query(query);
        let frames = sqlx::query_as::<_, Frame>(
            "SELECT f.* FROM frames f
             JOIN frames_fts fts ON f.id = fts.rowid
             WHERE frames_fts MATCH ?
             ORDER BY rank
             LIMIT ?",
        )
        .bind(&escaped)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(frames)
    }

    /// Insert a summary.
    pub async fn insert_summary(&self, summary: NewSummary) -> Result<i64> {
        let now = Utc::now().timestamp_millis();
        let apps_json = summary
            .apps_referenced
            .map(|a| serde_json::to_string(&a).unwrap_or_default());
        let frame_ids_json = summary
            .source_frame_ids
            .map(|ids| serde_json::to_string(&ids).unwrap_or_default());
        let summary_ids_json = summary
            .source_summary_ids
            .map(|ids| serde_json::to_string(&ids).unwrap_or_default());

        let id = sqlx::query_scalar::<_, i64>(
            "INSERT INTO summaries (machine_id, tier, start_time, end_time, summary, apps_referenced, source_frame_ids, source_summary_ids, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
             RETURNING id",
        )
        .bind(&self.machine_id)
        .bind(&summary.tier)
        .bind(summary.start_time)
        .bind(summary.end_time)
        .bind(&summary.summary)
        .bind(&apps_json)
        .bind(&frame_ids_json)
        .bind(&summary_ids_json)
        .bind(now)
        .fetch_one(&self.pool)
        .await?;

        Ok(id)
    }

    /// Get summaries for a tier in a time range.
    pub async fn get_summaries(&self, tier: &str, start: i64, end: i64) -> Result<Vec<Summary>> {
        let summaries = sqlx::query_as::<_, Summary>(
            "SELECT * FROM summaries
             WHERE tier = ? AND start_time >= ? AND end_time <= ?
             ORDER BY start_time ASC",
        )
        .bind(tier)
        .bind(start)
        .bind(end)
        .fetch_all(&self.pool)
        .await?;

        Ok(summaries)
    }

    /// Get child summaries that haven't been rolled up into the next tier yet.
    pub async fn get_unrolled_summaries(
        &self,
        child_tier: &str,
        parent_tier: &str,
        since: i64,
    ) -> Result<Vec<Summary>> {
        let summaries = sqlx::query_as::<_, Summary>(
            "SELECT s.* FROM summaries s
             WHERE s.tier = ? AND s.start_time >= ?
               AND s.id NOT IN (
                   SELECT value FROM summaries p, json_each(p.source_summary_ids)
                   WHERE p.tier = ?
               )
             ORDER BY s.start_time ASC",
        )
        .bind(child_tier)
        .bind(since)
        .bind(parent_tier)
        .fetch_all(&self.pool)
        .await?;

        Ok(summaries)
    }

    /// Get the latest summary for a given tier.
    pub async fn get_latest_summary(&self, tier: &str) -> Result<Option<Summary>> {
        let summary = sqlx::query_as::<_, Summary>(
            "SELECT * FROM summaries WHERE tier = ? ORDER BY end_time DESC LIMIT 1",
        )
        .bind(tier)
        .fetch_optional(&self.pool)
        .await?;

        Ok(summary)
    }

    /// Get the most recent summaries for a given tier, newest first.
    pub async fn get_recent_summaries(&self, tier: &str, limit: i64) -> Result<Vec<Summary>> {
        let summaries = sqlx::query_as::<_, Summary>(
            "SELECT * FROM summaries WHERE tier = ? ORDER BY end_time DESC LIMIT ?",
        )
        .bind(tier)
        .bind(limit.max(1))
        .fetch_all(&self.pool)
        .await?;
        Ok(summaries)
    }

    /// Query frames with flexible filters.
    pub async fn get_frames_filtered(&self, filter: &FrameFilter) -> Result<Vec<Frame>> {
        let start = filter.start.unwrap_or(0);
        let end = filter.end.unwrap_or(i64::MAX);
        let limit = filter.limit.unwrap_or(500);

        // Build query dynamically based on which filters are set
        let mut sql = String::from("SELECT * FROM frames WHERE timestamp >= ? AND timestamp <= ?");
        if filter.machine_id.is_some() {
            sql.push_str(" AND machine_id = ?");
        }
        if filter.app_name.is_some() {
            sql.push_str(" AND app_name = ?");
        }
        if filter.browser_tab_contains.is_some() {
            sql.push_str(" AND browser_tab LIKE ?");
        }
        if filter.source.is_some() {
            sql.push_str(" AND source = ?");
        }
        if filter.search_text.is_some() {
            sql.push_str(" AND id IN (SELECT rowid FROM frames_fts WHERE frames_fts MATCH ?)");
        }
        sql.push_str(" ORDER BY timestamp ASC LIMIT ?");

        let mut q = sqlx::query_as::<_, Frame>(&sql).bind(start).bind(end);

        if let Some(ref mid) = filter.machine_id {
            q = q.bind(mid);
        }
        if let Some(ref app) = filter.app_name {
            q = q.bind(app);
        }
        if let Some(ref url) = filter.browser_tab_contains {
            q = q.bind(format!("%{url}%"));
        }
        if let Some(ref source) = filter.source {
            q = q.bind(source);
        }
        if let Some(ref text) = filter.search_text {
            q = q.bind(sanitize_fts_query(text));
        }
        q = q.bind(limit);

        let frames = q.fetch_all(&self.pool).await?;
        Ok(frames)
    }

    /// Get the most recent frames, newest-first.
    pub async fn get_recent_frames(&self, limit: i64) -> Result<Vec<Frame>> {
        let limit = limit.max(1);
        let frames =
            sqlx::query_as::<_, Frame>("SELECT * FROM frames ORDER BY timestamp DESC LIMIT ?")
                .bind(limit)
                .fetch_all(&self.pool)
                .await?;
        Ok(frames)
    }

    /// Get distinct app names seen in a time range.
    pub async fn get_app_names(&self, start: i64, end: i64) -> Result<Vec<String>> {
        let rows: Vec<(String,)> = sqlx::query_as(
            "SELECT DISTINCT app_name FROM frames
             WHERE app_name IS NOT NULL AND timestamp >= ? AND timestamp <= ?
             ORDER BY app_name",
        )
        .bind(start)
        .bind(end)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|r| r.0).collect())
    }

    /// Get distinct browser tab titles seen in a time range.
    pub async fn get_browser_tabs(&self, start: i64, end: i64) -> Result<Vec<String>> {
        let rows: Vec<(String,)> = sqlx::query_as(
            "SELECT DISTINCT browser_tab FROM frames
             WHERE browser_tab IS NOT NULL AND timestamp >= ? AND timestamp <= ?
             ORDER BY browser_tab",
        )
        .bind(start)
        .bind(end)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|r| r.0).collect())
    }

    /// Increment a capture stat counter.
    pub async fn increment_stat(&self, key: &str) -> Result<()> {
        sqlx::query(
            "INSERT INTO capture_stats (key, value) VALUES (?, 1)
             ON CONFLICT(key) DO UPDATE SET value = value + 1",
        )
        .bind(key)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Get all capture statistics.
    pub async fn get_capture_stats(&self) -> Result<CaptureStats> {
        // Frame counts by source from actual data
        let total: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM frames")
            .fetch_one(&self.pool)
            .await?;
        let accessibility: (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM frames WHERE source = 'accessibility'")
                .fetch_one(&self.pool)
                .await?;
        let ocr_local: (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM frames WHERE source = 'ocr_local'")
                .fetch_one(&self.pool)
                .await?;
        let ocr_remote: (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM frames WHERE source = 'ocr_remote'")
                .fetch_one(&self.pool)
                .await?;

        // Skip counters from stats table
        let skipped_unchanged = self.get_stat("frames_skipped_unchanged").await?;
        let skipped_dedup = self.get_stat("frames_skipped_dedup").await?;

        Ok(CaptureStats {
            frames_total: total.0,
            frames_accessibility: accessibility.0,
            frames_ocr_local: ocr_local.0,
            frames_ocr_remote: ocr_remote.0,
            frames_skipped_unchanged: skipped_unchanged,
            frames_skipped_dedup: skipped_dedup,
        })
    }

    async fn get_stat(&self, key: &str) -> Result<i64> {
        let row: Option<(i64,)> = sqlx::query_as("SELECT value FROM capture_stats WHERE key = ?")
            .bind(key)
            .fetch_optional(&self.pool)
            .await?;
        Ok(row.map(|r| r.0).unwrap_or(0))
    }

    /// Get frame count and time range stats.
    pub async fn get_stats(&self) -> Result<(i64, Option<i64>, Option<i64>)> {
        let row: (i64, Option<i64>, Option<i64>) =
            sqlx::query_as("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM frames")
                .fetch_one(&self.pool)
                .await?;

        Ok(row)
    }

    /// Get frame counts grouped by machine_id.
    pub async fn get_machine_stats(&self) -> Result<Vec<(String, i64)>> {
        let rows: Vec<(String, i64)> = sqlx::query_as(
            "SELECT machine_id, COUNT(*) FROM frames GROUP BY machine_id ORDER BY machine_id",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }

    /// Get distinct machine IDs in the database.
    pub async fn get_machines(&self) -> Result<Vec<String>> {
        let rows: Vec<(String,)> =
            sqlx::query_as("SELECT DISTINCT machine_id FROM frames ORDER BY machine_id")
                .fetch_all(&self.pool)
                .await?;

        Ok(rows.into_iter().map(|r| r.0).collect())
    }

    /// Update or create an active window span. If the most recent span for this machine
    /// has the same app/window and ended recently, extend it. Otherwise start a new span.
    pub async fn upsert_active_window(
        &self,
        machine_id: &str,
        app_name: &str,
        window_title: Option<&str>,
        now_ms: i64,
    ) -> Result<()> {
        // Find the most recent span for this machine
        let recent: Option<(i64, String, Option<String>)> = sqlx::query_as(
            "SELECT id, app_name, window_title FROM active_windows
             WHERE machine_id = ? AND end_time >= ? - 5000
             ORDER BY end_time DESC LIMIT 1",
        )
        .bind(machine_id)
        .bind(now_ms)
        .fetch_optional(&self.pool)
        .await?;

        if let Some((id, prev_app, prev_window)) = recent {
            if prev_app == app_name && prev_window.as_deref() == window_title {
                // Same app/window — extend the span
                sqlx::query(
                    "UPDATE active_windows SET end_time = ?, duration_ms = ? - start_time WHERE id = ?",
                )
                .bind(now_ms)
                .bind(now_ms)
                .bind(id)
                .execute(&self.pool)
                .await?;
                return Ok(());
            }
        }

        // Different app/window or no recent span — start new
        sqlx::query(
            "INSERT INTO active_windows (machine_id, app_name, window_title, start_time, end_time, duration_ms)
             VALUES (?, ?, ?, ?, ?, 0)",
        )
        .bind(machine_id)
        .bind(app_name)
        .bind(window_title)
        .bind(now_ms)
        .bind(now_ms)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Finalize the current active window span (set its end_time).
    pub async fn finalize_active_window(&self, machine_id: &str, end_ms: i64) -> Result<()> {
        sqlx::query(
            "UPDATE active_windows SET end_time = ?, duration_ms = ? - start_time
             WHERE machine_id = ? AND id = (
                 SELECT id FROM active_windows WHERE machine_id = ?
                 ORDER BY end_time DESC LIMIT 1
             ) AND end_time < ?",
        )
        .bind(end_ms)
        .bind(end_ms)
        .bind(machine_id)
        .bind(machine_id)
        .bind(end_ms)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Get total time per app in a time range, sorted by duration descending.
    /// Excludes apps with zero total time.
    pub async fn get_app_time(&self, start: i64, end: i64) -> Result<Vec<(String, i64)>> {
        let rows: Vec<(String, i64)> = sqlx::query_as(
            "SELECT app_name, SUM(duration_ms) as total_ms
             FROM active_windows
             WHERE start_time >= ? AND start_time <= ? AND duration_ms > 0
             GROUP BY app_name
             HAVING total_ms > 0
             ORDER BY total_ms DESC",
        )
        .bind(start)
        .bind(end)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }

    /// Get summary counts per tier in a time range.
    pub async fn get_summary_counts(&self, start: i64, end: i64) -> Result<Vec<(String, i64)>> {
        let rows: Vec<(String, i64)> = sqlx::query_as(
            "SELECT tier, COUNT(*) FROM summaries
             WHERE start_time >= ? AND end_time <= ?
             GROUP BY tier ORDER BY tier",
        )
        .bind(start)
        .bind(end)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    /// Get the number of frames in a time range.
    pub async fn get_frame_count(&self, start: i64, end: i64) -> Result<i64> {
        let (count,): (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM frames WHERE timestamp >= ? AND timestamp <= ?")
                .bind(start)
                .bind(end)
                .fetch_one(&self.pool)
                .await?;
        Ok(count)
    }

    /// Search summaries by text, optionally filtered by tier and time range.
    pub async fn search_summaries(
        &self,
        query: &str,
        tier: Option<&str>,
        start: i64,
        end: i64,
    ) -> Result<Vec<Summary>> {
        let escaped = sanitize_fts_query(query);

        let (sql, has_tier) = if tier.is_some() {
            (
                "SELECT s.* FROM summaries s
                 JOIN summaries_fts fts ON s.id = fts.rowid
                 WHERE summaries_fts MATCH ?
                   AND s.tier = ?
                   AND s.start_time >= ? AND s.end_time <= ?
                 ORDER BY s.start_time ASC",
                true,
            )
        } else {
            (
                "SELECT s.* FROM summaries s
                 JOIN summaries_fts fts ON s.id = fts.rowid
                 WHERE summaries_fts MATCH ?
                   AND s.start_time >= ? AND s.end_time <= ?
                 ORDER BY s.start_time ASC",
                false,
            )
        };

        let mut q = sqlx::query_as::<_, Summary>(sql).bind(&escaped);
        if has_tier {
            q = q.bind(tier.unwrap());
        }
        q = q.bind(start).bind(end);

        let summaries = q.fetch_all(&self.pool).await?;
        Ok(summaries)
    }

    // ── Typing Speed ──

    /// Insert a typing burst record from a sequence of keystroke timestamps and chars.
    pub async fn insert_typing_burst(&self, times: &[i64], app_name: Option<&str>, typed_text: &str) -> Result<()> {
        if times.len() < 2 {
            return Ok(());
        }
        let start_time = times[0];
        let end_time = *times.last().unwrap();
        let duration_ms = end_time - start_time;
        let char_count = times.len() as i64;

        // Compute WPM: standard is 5 chars per word
        let words = char_count as f64 / 5.0;
        let minutes = duration_ms as f64 / 60_000.0;
        let wpm = if minutes > 0.0 { words / minutes } else { 0.0 };

        // Compute inter-key intervals
        let intervals: Vec<i64> = times.windows(2).map(|w| w[1] - w[0]).collect();
        let intervals_json = serde_json::to_string(&intervals).unwrap_or_default();

        sqlx::query(
            "INSERT INTO typing_bursts (machine_id, start_time, end_time, char_count, duration_ms, wpm, app_name, intervals, typed_text)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&self.machine_id)
        .bind(start_time)
        .bind(end_time)
        .bind(char_count)
        .bind(duration_ms)
        .bind(wpm)
        .bind(app_name)
        .bind(&intervals_json)
        .bind(typed_text)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Get typing speed statistics for a time range.
    /// Returns (overall_wpm, total_chars, total_duration_ms, per-app stats).
    pub async fn get_typing_speed(
        &self,
        start: i64,
        end: i64,
    ) -> Result<Vec<(Option<String>, i64, i64, f64)>> {
        // Returns (app_name, total_chars, total_duration_ms, avg_wpm) per app
        let rows: Vec<(Option<String>, i64, i64, f64)> = sqlx::query_as(
            "SELECT app_name, SUM(char_count) as total_chars, SUM(duration_ms) as total_ms,
                    AVG(wpm) as avg_wpm
             FROM typing_bursts
             WHERE start_time >= ? AND start_time <= ?
             GROUP BY app_name
             ORDER BY total_chars DESC",
        )
        .bind(start)
        .bind(end)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }

    /// Get raw typing burst records for detailed analysis.
    pub async fn get_typing_bursts(
        &self,
        start: i64,
        end: i64,
    ) -> Result<Vec<(i64, i64, i64, i64, f64, Option<String>, Option<String>)>> {
        // Returns (start_time, end_time, char_count, duration_ms, wpm, app_name, typed_text)
        let rows: Vec<(i64, i64, i64, i64, f64, Option<String>, Option<String>)> = sqlx::query_as(
            "SELECT start_time, end_time, char_count, duration_ms, wpm, app_name, typed_text
             FROM typing_bursts
             WHERE start_time >= ? AND start_time <= ?
             ORDER BY start_time ASC",
        )
        .bind(start)
        .bind(end)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }

    /// Get data availability: earliest/latest timestamps and counts per summary tier,
    /// plus frame count and date range. Used by smart-query to tell the LLM what's available.
    pub async fn get_data_availability(&self) -> Result<Vec<(String, i64, i64, i64)>> {
        // Returns (tier_or_type, count, earliest_ms, latest_ms)
        let rows: Vec<(String, i64, i64, i64)> = sqlx::query_as(
            "SELECT tier, COUNT(*), MIN(start_time), MAX(end_time)
             FROM summaries GROUP BY tier
             UNION ALL
             SELECT 'frames', COUNT(*), MIN(timestamp), MAX(timestamp)
             FROM frames
             UNION ALL
             SELECT 'typing_bursts', COUNT(*), MIN(start_time), MAX(end_time)
             FROM typing_bursts
             UNION ALL
             SELECT 'click_events', COUNT(*), MIN(timestamp), MAX(timestamp)
             FROM click_events
             ORDER BY 3 ASC",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }

    // ── Click Events ──

    /// Insert a click event with target element info.
    pub async fn insert_click_event(
        &self,
        timestamp: i64,
        x: f64,
        y: f64,
        app_name: Option<&str>,
        role: Option<&str>,
        title: Option<&str>,
        description: Option<&str>,
        value: Option<&str>,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO click_events (machine_id, timestamp, x, y, app_name, element_role, element_title, element_description, element_value)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&self.machine_id)
        .bind(timestamp)
        .bind(x)
        .bind(y)
        .bind(app_name)
        .bind(role)
        .bind(title)
        .bind(description)
        .bind(value)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Get click events in a time range.
    pub async fn get_click_events(
        &self,
        start: i64,
        end: i64,
    ) -> Result<Vec<(i64, f64, f64, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>)>> {
        let rows = sqlx::query_as(
            "SELECT timestamp, x, y, app_name, element_role, element_title, element_description, element_value
             FROM click_events
             WHERE timestamp >= ? AND timestamp <= ?
             ORDER BY timestamp ASC",
        )
        .bind(start)
        .bind(end)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }

    /// Search click events by element title/description/value.
    pub async fn search_clicks(
        &self,
        query: &str,
        start: i64,
        end: i64,
    ) -> Result<Vec<(i64, f64, f64, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>)>> {
        let pattern = format!("%{}%", query);
        let rows = sqlx::query_as(
            "SELECT timestamp, x, y, app_name, element_role, element_title, element_description, element_value
             FROM click_events
             WHERE (element_title LIKE ? OR element_description LIKE ? OR element_value LIKE ?)
               AND timestamp >= ? AND timestamp <= ?
             ORDER BY timestamp ASC",
        )
        .bind(&pattern)
        .bind(&pattern)
        .bind(&pattern)
        .bind(start)
        .bind(end)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }

    /// Search typing bursts by typed text content.
    pub async fn search_typing(
        &self,
        query: &str,
        start: i64,
        end: i64,
    ) -> Result<Vec<(i64, i64, i64, i64, f64, Option<String>, Option<String>)>> {
        let pattern = format!("%{}%", query);
        let rows: Vec<(i64, i64, i64, i64, f64, Option<String>, Option<String>)> = sqlx::query_as(
            "SELECT start_time, end_time, char_count, duration_ms, wpm, app_name, typed_text
             FROM typing_bursts
             WHERE typed_text LIKE ? AND start_time >= ? AND start_time <= ?
             ORDER BY start_time ASC",
        )
        .bind(&pattern)
        .bind(start)
        .bind(end)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }

    // ── User Profile ──

    /// Get all active profile entries, ordered by category then recency.
    pub async fn get_active_profile(&self) -> Result<Vec<ProfileEntry>> {
        let rows = sqlx::query_as::<_, ProfileEntry>(
            "SELECT * FROM user_profile WHERE status = 'active' ORDER BY category, last_seen DESC",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    /// Get active profile entries for a specific category.
    pub async fn get_profile_by_category(&self, category: &str) -> Result<Vec<ProfileEntry>> {
        let rows = sqlx::query_as::<_, ProfileEntry>(
            "SELECT * FROM user_profile WHERE category = ? AND status = 'active' ORDER BY last_seen DESC",
        )
        .bind(category)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    /// Insert a new profile entry. Returns the new row ID.
    pub async fn insert_profile_entry(
        &self,
        category: &str,
        content: &str,
        summary_id: Option<i64>,
    ) -> Result<i64> {
        let now = Utc::now().timestamp_millis();
        let id = sqlx::query_scalar::<_, i64>(
            "INSERT INTO user_profile (category, content, first_seen, last_seen, status, source_summary_id)
             VALUES (?, ?, ?, ?, 'active', ?)
             RETURNING id",
        )
        .bind(category)
        .bind(content)
        .bind(now)
        .bind(now)
        .bind(summary_id)
        .fetch_one(&self.pool)
        .await?;
        Ok(id)
    }

    /// Update an existing profile entry's content and refresh last_seen.
    pub async fn update_profile_entry(
        &self,
        id: i64,
        content: &str,
        summary_id: Option<i64>,
    ) -> Result<()> {
        let now = Utc::now().timestamp_millis();
        sqlx::query(
            "UPDATE user_profile SET content = ?, last_seen = ?, source_summary_id = ? WHERE id = ?",
        )
        .bind(content)
        .bind(now)
        .bind(summary_id)
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Archive a profile entry (mark as no longer active).
    pub async fn archive_profile_entry(&self, id: i64) -> Result<()> {
        sqlx::query("UPDATE user_profile SET status = 'archived' WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Get time breakdown by window title for a specific app, sorted by duration descending.
    /// Excludes entries with zero total time.
    pub async fn get_window_time(
        &self,
        app_name: &str,
        start: i64,
        end: i64,
    ) -> Result<Vec<(Option<String>, i64)>> {
        let rows: Vec<(Option<String>, i64)> = sqlx::query_as(
            "SELECT window_title, SUM(duration_ms) as total_ms
             FROM active_windows
             WHERE app_name = ? AND start_time >= ? AND start_time <= ? AND duration_ms > 0
             GROUP BY window_title
             HAVING total_ms > 0
             ORDER BY total_ms DESC",
        )
        .bind(app_name)
        .bind(start)
        .bind(end)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_insert_and_query_frames() {
        let db = Database::in_memory().await.unwrap();

        let frame = NewFrame {
            app_name: Some("Terminal".into()),
            window_title: Some("zsh".into()),
            browser_tab: None,
            text_content: "$ cargo build\nCompiling screentrack v0.1.0".into(),
            source: "accessibility".into(),
            screenshot_path: None,
        };

        let id = db.insert_frame(frame).await.unwrap();
        assert!(id.is_some());

        let now = Utc::now().timestamp_millis();
        let frames = db.get_frames(now - 5000, now + 5000).await.unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].app_name.as_deref(), Some("Terminal"));
    }

    #[tokio::test]
    async fn test_dedup() {
        let db = Database::in_memory().await.unwrap();

        let frame = NewFrame {
            app_name: Some("Terminal".into()),
            window_title: Some("zsh".into()),
            browser_tab: None,
            text_content: "same content".into(),
            source: "accessibility".into(),
            screenshot_path: None,
        };

        let id1 = db.insert_frame(frame.clone()).await.unwrap();
        assert!(id1.is_some());

        // Same content within dedup window should be skipped
        let id2 = db.insert_frame(frame).await.unwrap();
        assert!(id2.is_none());
    }

    #[tokio::test]
    async fn test_fts_search() {
        let db = Database::in_memory().await.unwrap();

        let frame = NewFrame {
            app_name: Some("VSCode".into()),
            window_title: Some("main.rs".into()),
            browser_tab: None,
            text_content: "fn main() { println!(\"hello world\"); }".into(),
            source: "ocr_local".into(),
            screenshot_path: None,
        };
        db.insert_frame(frame).await.unwrap();

        let results = db.search_frames("hello world", 10).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_summaries() {
        let db = Database::in_memory().await.unwrap();

        let now = Utc::now().timestamp_millis();
        let summary = NewSummary {
            tier: "micro".into(),
            start_time: now - 300_000,
            end_time: now,
            summary: "User was editing Rust code in VSCode.".into(),
            apps_referenced: Some(vec!["VSCode".into()]),
            source_frame_ids: Some(vec![1, 2, 3]),
            source_summary_ids: None,
        };

        let id = db.insert_summary(summary).await.unwrap();
        assert!(id > 0);

        let summaries = db
            .get_summaries("micro", now - 600_000, now + 1000)
            .await
            .unwrap();
        assert_eq!(summaries.len(), 1);
    }

    #[tokio::test]
    async fn test_filtered_query() {
        let db = Database::in_memory().await.unwrap();

        // Insert frames from different apps
        for (app, tab, text, source) in [
            (
                "Safari",
                Some("rust-lang/rust: The Rust Programming Language"),
                "Rust repo page",
                "accessibility",
            ),
            (
                "Google Chrome",
                Some("tokio - Rust async runtime"),
                "Tokio docs",
                "accessibility",
            ),
            ("Terminal", None, "cargo build output", "accessibility"),
            ("VSCode", None, "fn main() {}", "ocr_local"),
        ] {
            db.insert_frame(NewFrame {
                app_name: Some(app.into()),
                window_title: None,
                browser_tab: tab.map(|t| t.into()),
                text_content: text.into(),
                source: source.into(),
                screenshot_path: None,
            })
            .await
            .unwrap();
        }

        let now = Utc::now().timestamp_millis();

        // Filter by app
        let frames = db
            .get_frames_filtered(&FrameFilter {
                start: Some(now - 5000),
                end: Some(now + 5000),
                app_name: Some("Safari".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(frames.len(), 1);

        // Filter by browser tab
        let frames = db
            .get_frames_filtered(&FrameFilter {
                start: Some(now - 5000),
                end: Some(now + 5000),
                browser_tab_contains: Some("rust".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(frames.len(), 2); // Both browser tabs contain "rust"

        // Filter by source
        let frames = db
            .get_frames_filtered(&FrameFilter {
                start: Some(now - 5000),
                end: Some(now + 5000),
                source: Some("ocr_local".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(frames.len(), 1);
    }

    #[tokio::test]
    async fn test_capture_stats() {
        let db = Database::in_memory().await.unwrap();

        // Insert some frames
        db.insert_frame(NewFrame {
            app_name: Some("Terminal".into()),
            window_title: None,
            browser_tab: None,
            text_content: "frame 1".into(),
            source: "accessibility".into(),
            screenshot_path: None,
        })
        .await
        .unwrap();

        db.insert_frame(NewFrame {
            app_name: None,
            window_title: None,
            browser_tab: None,
            text_content: "frame 2".into(),
            source: "ocr_local".into(),
            screenshot_path: None,
        })
        .await
        .unwrap();

        // Simulate some skips
        db.increment_stat("frames_skipped_unchanged").await.unwrap();
        db.increment_stat("frames_skipped_unchanged").await.unwrap();
        db.increment_stat("frames_skipped_unchanged").await.unwrap();

        let stats = db.get_capture_stats().await.unwrap();
        assert_eq!(stats.frames_total, 2);
        assert_eq!(stats.frames_accessibility, 1);
        assert_eq!(stats.frames_ocr_local, 1);
        assert_eq!(stats.frames_skipped_unchanged, 3);
    }
}

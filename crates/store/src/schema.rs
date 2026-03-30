use anyhow::Result;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::SqlitePool;
use std::path::Path;
use std::str::FromStr;

pub struct Database {
    pub pool: SqlitePool,
    pub machine_id: String,
}

/// Get a stable hostname for this machine.
/// On macOS, prefers LocalHostName (Bonjour name) since the POSIX hostname
/// often returns an unstable reverse-DNS name from DHCP (e.g. "Mac.bp.local").
pub fn get_machine_id() -> String {
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("scutil")
            .args(["--get", "LocalHostName"])
            .output()
        {
            let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !name.is_empty() {
                return name;
            }
        }
    }

    gethostname::gethostname().to_string_lossy().into_owned()
}

impl Database {
    pub async fn new(path: &Path) -> Result<Self> {
        let url = format!("sqlite:{}?mode=rwc", path.display());
        let options = SqliteConnectOptions::from_str(&url)?
            .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal)
            .synchronous(sqlx::sqlite::SqliteSynchronous::Normal)
            .busy_timeout(std::time::Duration::from_secs(30));

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await?;

        let machine_id = get_machine_id();
        let db = Self { pool, machine_id };
        db.run_migrations().await?;
        Ok(db)
    }

    pub async fn in_memory() -> Result<Self> {
        let options = SqliteConnectOptions::from_str("sqlite::memory:")?
            .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal)
            .synchronous(sqlx::sqlite::SqliteSynchronous::Normal);

        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await?;

        let machine_id = get_machine_id();
        let db = Self { pool, machine_id };
        db.run_migrations().await?;
        Ok(db)
    }

    async fn run_migrations(&self) -> Result<()> {
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                app_name TEXT,
                window_title TEXT,
                browser_tab TEXT,
                text_content TEXT NOT NULL,
                source TEXT NOT NULL,
                content_hash TEXT,
                simhash INTEGER,
                screenshot_path TEXT
            )",
        )
        .execute(&self.pool)
        .await?;

        // Migrations: add columns if missing (for existing databases)
        sqlx::query("ALTER TABLE frames ADD COLUMN browser_tab TEXT")
            .execute(&self.pool)
            .await
            .ok();
        sqlx::query("ALTER TABLE frames ADD COLUMN simhash INTEGER")
            .execute(&self.pool)
            .await
            .ok();

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS capture_stats (
                key TEXT PRIMARY KEY,
                value INTEGER NOT NULL DEFAULT 0
            )",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tier TEXT NOT NULL,
                start_time INTEGER NOT NULL,
                end_time INTEGER NOT NULL,
                summary TEXT NOT NULL,
                apps_referenced TEXT,
                source_frame_ids TEXT,
                source_summary_ids TEXT,
                created_at INTEGER NOT NULL
            )",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_frames_timestamp ON frames(timestamp)")
            .execute(&self.pool)
            .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_frames_content_hash ON frames(content_hash)")
            .execute(&self.pool)
            .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_summaries_tier_time ON summaries(tier, start_time)",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_frames_app_name ON frames(app_name)")
            .execute(&self.pool)
            .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_frames_browser_tab ON frames(browser_tab)")
            .execute(&self.pool)
            .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_frames_source ON frames(source)")
            .execute(&self.pool)
            .await?;

        // FTS5 for full-text search on frame text
        sqlx::query(
            "CREATE VIRTUAL TABLE IF NOT EXISTS frames_fts USING fts5(
                text_content,
                content=frames,
                content_rowid=id
            )",
        )
        .execute(&self.pool)
        .await?;

        // Triggers to keep FTS in sync
        sqlx::query(
            "CREATE TRIGGER IF NOT EXISTS frames_ai AFTER INSERT ON frames BEGIN
                INSERT INTO frames_fts(rowid, text_content) VALUES (new.id, new.text_content);
            END",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE TRIGGER IF NOT EXISTS frames_ad AFTER DELETE ON frames BEGIN
                INSERT INTO frames_fts(frames_fts, rowid, text_content) VALUES('delete', old.id, old.text_content);
            END",
        )
        .execute(&self.pool)
        .await?;

        // Multi-machine support columns
        sqlx::query("ALTER TABLE frames ADD COLUMN machine_id TEXT NOT NULL DEFAULT 'local'")
            .execute(&self.pool)
            .await
            .ok();
        sqlx::query("ALTER TABLE frames ADD COLUMN synced INTEGER NOT NULL DEFAULT 0")
            .execute(&self.pool)
            .await
            .ok();
        sqlx::query("ALTER TABLE frames ADD COLUMN summarized INTEGER NOT NULL DEFAULT 0")
            .execute(&self.pool)
            .await
            .ok();
        sqlx::query("ALTER TABLE summaries ADD COLUMN machine_id TEXT NOT NULL DEFAULT 'local'")
            .execute(&self.pool)
            .await
            .ok();

        // Migrate existing 'local' rows to actual hostname
        sqlx::query("UPDATE frames SET machine_id = ? WHERE machine_id = 'local'")
            .bind(&self.machine_id)
            .execute(&self.pool)
            .await
            .ok();
        sqlx::query("UPDATE summaries SET machine_id = ? WHERE machine_id = 'local'")
            .bind(&self.machine_id)
            .execute(&self.pool)
            .await
            .ok();

        // Consolidate any POSIX hostname (e.g. "Mac.bp.local") to the stable machine_id
        // This handles the case where the hostname source changed (e.g. from gethostname to scutil)
        let posix_hostname = gethostname::gethostname().to_string_lossy().into_owned();
        if posix_hostname != self.machine_id {
            let updated = sqlx::query("UPDATE frames SET machine_id = ? WHERE machine_id = ?")
                .bind(&self.machine_id)
                .bind(&posix_hostname)
                .execute(&self.pool)
                .await
                .ok();
            if let Some(result) = updated {
                if result.rows_affected() > 0 {
                    tracing::info!(
                        "Migrated {} frames from '{}' to '{}'",
                        result.rows_affected(),
                        posix_hostname,
                        self.machine_id
                    );
                }
            }
            sqlx::query("UPDATE summaries SET machine_id = ? WHERE machine_id = ?")
                .bind(&self.machine_id)
                .bind(&posix_hostname)
                .execute(&self.pool)
                .await
                .ok();
        }

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_frames_machine_timestamp ON frames(machine_id, timestamp)",
        )
        .execute(&self.pool)
        .await?;

        // Active window time tracking
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS active_windows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id TEXT NOT NULL,
                app_name TEXT NOT NULL,
                window_title TEXT,
                start_time INTEGER NOT NULL,
                end_time INTEGER NOT NULL,
                duration_ms INTEGER NOT NULL
            )",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_active_windows_machine_time ON active_windows(machine_id, start_time)",
        )
        .execute(&self.pool)
        .await?;

        // User profile / insights — built from daily summaries
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS user_profile (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                first_seen INTEGER NOT NULL,
                last_seen INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                source_summary_id INTEGER
            )",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_user_profile_category_status ON user_profile(category, status)",
        )
        .execute(&self.pool)
        .await?;

        // Typing burst records for typing speed analysis
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS typing_bursts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id TEXT NOT NULL DEFAULT 'local',
                start_time INTEGER NOT NULL,
                end_time INTEGER NOT NULL,
                char_count INTEGER NOT NULL,
                duration_ms INTEGER NOT NULL,
                wpm REAL NOT NULL,
                app_name TEXT,
                intervals TEXT,
                typed_text TEXT
            )",
        )
        .execute(&self.pool)
        .await?;

        // Migration: add typed_text column if missing
        sqlx::query("ALTER TABLE typing_bursts ADD COLUMN typed_text TEXT")
            .execute(&self.pool)
            .await
            .ok();

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_typing_bursts_time ON typing_bursts(start_time)",
        )
        .execute(&self.pool)
        .await?;

        // Click events with target element info
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS click_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id TEXT NOT NULL DEFAULT 'local',
                timestamp INTEGER NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                app_name TEXT,
                element_role TEXT,
                element_title TEXT,
                element_description TEXT,
                element_value TEXT
            )",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_click_events_time ON click_events(timestamp)",
        )
        .execute(&self.pool)
        .await?;

        // FTS5 for full-text search on summaries
        sqlx::query(
            "CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts USING fts5(
                summary,
                content=summaries,
                content_rowid=id
            )",
        )
        .execute(&self.pool)
        .await?;

        // Triggers to keep summaries FTS in sync
        sqlx::query(
            "CREATE TRIGGER IF NOT EXISTS summaries_ai AFTER INSERT ON summaries BEGIN
                INSERT INTO summaries_fts(rowid, summary) VALUES (new.id, new.summary);
            END",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE TRIGGER IF NOT EXISTS summaries_ad AFTER DELETE ON summaries BEGIN
                INSERT INTO summaries_fts(summaries_fts, rowid, summary) VALUES('delete', old.id, old.summary);
            END",
        )
        .execute(&self.pool)
        .await?;

        // Backfill: index any existing summaries that predate the FTS table
        sqlx::query(
            "INSERT OR IGNORE INTO summaries_fts(rowid, summary)
             SELECT id, summary FROM summaries",
        )
        .execute(&self.pool)
        .await
        .ok();

        Ok(())
    }
}

//! Watch the running executable for changes and relaunch when updated.

use std::path::PathBuf;
use std::time::SystemTime;
use tokio::time::{interval, Duration};
use tracing::{info, warn};

/// Poll interval for checking binary modification time.
const CHECK_INTERVAL: Duration = Duration::from_secs(5);

/// Get the path to the currently running executable.
fn exe_path() -> Option<PathBuf> {
    std::env::current_exe().ok()
}

/// Get the modification time of a file.
fn mtime(path: &PathBuf) -> Option<SystemTime> {
    std::fs::metadata(path).ok()?.modified().ok()
}

/// Spawn a task that watches the running binary and relaunches if it changes.
/// This enables zero-downtime upgrades: compile + install, and the running
/// instance will detect the change and relaunch itself within a few seconds.
pub fn spawn_watcher(rt: &tokio::runtime::Runtime) {
    let Some(path) = exe_path() else {
        warn!("auto_update: could not determine executable path, skipping");
        return;
    };
    let Some(initial_mtime) = mtime(&path) else {
        warn!("auto_update: could not read executable mtime, skipping");
        return;
    };

    info!("auto_update: watching {} for changes", path.display());

    rt.spawn(async move {
        let mut ticker = interval(CHECK_INTERVAL);
        ticker.tick().await; // skip first immediate tick

        loop {
            ticker.tick().await;

            let Some(current_mtime) = mtime(&path) else {
                continue;
            };

            if current_mtime != initial_mtime {
                info!(
                    "auto_update: executable changed, relaunching ScreenTrack..."
                );

                // Use a shell one-liner that:
                // 1. Waits for THIS process to exit (by PID)
                // 2. Then launches the new version
                // This avoids the race where `open` tries to activate the
                // still-running instance, and ensures the new app launches
                // after the old one is fully gone.
                let pid = std::process::id();
                let _ = std::process::Command::new("/bin/sh")
                    .arg("-c")
                    .arg(format!(
                        "while kill -0 {} 2>/dev/null; do sleep 0.5; done; sleep 1; open /Applications/ScreenTrack.app",
                        pid
                    ))
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .spawn();

                // Give the shell a moment to start, then exit
                tokio::time::sleep(Duration::from_millis(500)).await;
                std::process::exit(0);
            }
        }
    });
}

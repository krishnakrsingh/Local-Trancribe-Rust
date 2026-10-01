//! `VaaniDesk.exe`: the desktop application.
//!
//! This is the whole program a user downloads. It opens a native window, wires
//! the real pipeline behind it, and hands control to Slint's event loop. There
//! is no console, no web server, and no second process.
//!
//! # The three things that can fail before a window appears
//!
//! 1. **The database cannot be opened.** Without it there is nowhere to put a
//!    transcript, so there is no product. This exits with a message.
//! 2. **The model is missing.** It ships in `models/`, and a corrupt or absent
//!    bundle is reported *in the window* rather than on exit, because the app
//!    still opens, still explains, and still browses previous work.
//! 3. **No display.** Reported plainly, because a GUI app that exits silently
//!    is indistinguishable from a broken installer.
//!
//! In other words: only a missing database is fatal. Everything else the user
//! can act on belongs in the window's banner, which is what it is for.

#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]
#![warn(missing_docs)]

use std::sync::Arc;

use vaani_desktop::host::DesktopHost;

use vaani_app::backend::AppBackend;
use vaani_storage::Store;

fn main() {
    // Logging is initialised first so that anything after this point is
    // recorded. A GUI app has no console, so the file log is the only record
    // that exists, and "it didn't do anything" is the least useful bug report.
    init_logging();

    if let Err(message) = run() {
        // Nothing fatal is expected here, but "it vanished with no message" is
        // the worst possible outcome, so a fatal path prints somewhere a user
        // or a support engineer can find it.
        eprintln!("VaaniDesk could not start: {message}");
        std::process::exit(1);
    }
}

/// Build the backend, the window, and run the event loop.
fn run() -> Result<(), String> {
    // The one genuinely fatal dependency: somewhere to keep results.
    let store = Store::open_default().map_err(|error| {
        format!(
            "local storage could not be opened ({}). {}",
            error.code(),
            error.user.detail
        )
    })?;
    tracing::info!(db = %store.path().display(), "storage open");

    // Clear out anything a previous run was killed before it could clean up.
    //
    // This is a crash-recovery step, not housekeeping: a job's work directory
    // holds decoded audio and partial exports, `run_job` removes it on every
    // exit it can reach, and a power cut or a task-manager kill reaches none of
    // them. On a machine that was interrupted mid-job those directories can be
    // hundreds of megabytes, and nothing else would ever remove them.
    //
    // Before the backend exists, so no live job can be using one.
    let swept = vaani_app::orchestrator::Orchestrator::sweep_stale_work_dirs(
        &vaani_app::config::default_work_dir(),
    );
    if swept > 0 {
        tracing::info!(
            directories = swept,
            "removed stale work directories from a previous run"
        );
    }

    // The backend resolves the bundled model, picks the execution provider and
    // owns the job workers. Everything the window shows comes from it.
    let backend = AppBackend::new(store).map_err(|error| {
        format!(
            "the transcription engine could not start: {}",
            error.user.detail
        )
    })?;

    // Report what the machine and the bundle actually offer, once, at startup.
    // `doctor`'s per-user claims are the reason this is worth logging: a user
    // asking "why is it slow" is answered by the provider line.
    let runtime = vaani_inference::RuntimeReport::collect();
    tracing::info!(
        runtime = %runtime.summary(),
        gpu = runtime.machine.gpu_acceleration_available(),
        provider = ?runtime.machine.inference_adapter().map(|a| a.name.as_str()),
        "runtime ready"
    );

    // The one blocking, user-visible call in the process.
    vaani_ui::run(backend, Arc::new(DesktopHost::new()))
        .map_err(|error| format!("the window could not be created: {error}"))
}

/// Send `tracing` to a file beside the database, and nowhere else.
///
/// There is no console to print to, and a user who reports "it didn't work"
/// needs something to attach. The log sits in the same directory as the
/// database (`dirs::data_dir()`/`VaaniDesk`), which is where the store already
/// puts itself, so there is exactly one folder to look in. The log is truncated
/// on each run so it never grows without bound in a place the user cannot
/// easily find.
fn init_logging() {
    use tracing_subscriber::EnvFilter;

    let filter = EnvFilter::try_from_env("VAANI_LOG").unwrap_or_else(|_| EnvFilter::new("info"));

    // `dirs::data_dir()` is the same root `vaani_storage::Store::default_path`
    // uses, so the log and the database never end up in different trees.
    let log_path = dirs::data_dir().map(|dir| dir.join("VaaniDesk").join("vaanidesk.log"));
    // Kept separately because `log_path` is moved by the match below and the
    // error handler still needs to know where the file is.
    let log_path_for_errors = log_path.clone();

    // A plain marker with no tracing machinery. If the file appears with only
    // this line, the path works and the subscriber setup is at fault; if it
    // does not appear at all, the path itself is wrong. One write, two
    // failure modes distinguished.
    if let Some(ref path) = log_path {
        let _ = std::fs::write(
            path,
            format!("VaaniDesk {} starting\n", env!("CARGO_PKG_VERSION")),
        );
    }

    let builder = || {
        tracing_subscriber::fmt()
            .with_env_filter(filter.clone())
            .with_target(true)
            .with_ansi(false)
    };

    // One builder *per attempt*, not one shared across the arms below: both
    // `try_init` and `with_writer` consume the builder by value, so a single
    // value cannot appear in more than one arm of these matches. Building it
    // costs nothing, and doing it per arm is what lets "log to the file if we
    // can, otherwise to stderr" be expressed at all.
    let result = match log_path {
        Some(path) => match path.parent().map(std::fs::create_dir_all) {
            Some(Ok(())) => std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&path)
                .map_or_else(
                    |_| builder().try_init(),
                    |file| {
                        builder()
                            .with_writer(std::sync::Mutex::new(file))
                            .try_init()
                    },
                ),
            _ => builder().try_init(),
        },
        _ => builder().try_init(),
    };

    if let Err(error) = &result {
        // There is no console to print to, so a failure here would vanish
        // entirely. Append it to the log file itself, which the marker above
        // proved is writable.
        let msg = format!("note: logging to file unavailable ({error}); using stderr\n");
        eprintln!("{msg}");
        if let Some(ref path) = log_path_for_errors {
            use std::io::Write as _;
            if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(path) {
                let _ = f.write_all(msg.as_bytes());
            }
        }
    }
}

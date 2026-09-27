//! Launch the real window against the in-memory demo backend.
//!
//! This is the binary A5 used to verify by hand that a window appears and
//! responds, and it is what the architect's first smoke test should run before
//! the production backend exists. There is nothing to install:
//!
//! ```text
//! cargo run -p vaani-ui --example window
//! ```
//!
//! The window it opens carries a permanent "Demo mode" banner, and every
//! transcript it shows is prefixed `[sample segment N]`, so a screenshot of this
//! run can never be mistaken for a screenshot of the product.
//!
//! It prints its own wall-clock heartbeat so that a run left open in the
//! background is visibly alive. Process CPU time is deliberately *not* measured
//! here: reading it needs either platform FFI in an example or an external
//! sampler, and an external sampler is both safer and more honest. The A5
//! report quotes the number `Get-Process` produced.

use std::sync::Arc;
use std::time::{Duration, Instant};

use slint::ComponentHandle as _;
use vaani_ui::HostServices;
use vaani_ui::backend::ExportTarget;
use vaani_ui::backend::{JobController as _, Theme};
use vaani_ui::fake::FakeController;

/// A host that answers the media picker with a path that does not exist.
///
/// That is deliberate. The demo job never opens a file: it runs a scripted
/// sequence and produces labelled sample text, so pointing it at a real
/// recording would imply it reads one. Press F2, get a job, watch the phases go
/// by, then open the transcript and rename a speaker.
#[derive(Debug, Default)]
struct DemoHost;

impl HostServices for DemoHost {
    fn pick_media_files(&self) -> Vec<std::path::PathBuf> {
        vec![std::path::PathBuf::from(
            "C:/recordings/quarterly-review.mkv",
        )]
    }

    fn pick_export_target(&self, suggested_name: &str) -> Option<ExportTarget> {
        Some(ExportTarget::File(
            std::env::temp_dir().join(suggested_name),
        ))
    }

    fn set_clipboard_text(&self, _text: &str) {}

    fn host_name(&self) -> String {
        "demo host (no dialogs yet)".to_owned()
    }
}

fn main() -> Result<(), slint::PlatformError> {
    std::thread::Builder::new()
        .name("vaani-ui-heartbeat".to_owned())
        .spawn(heartbeat)
        .expect("the OS refused to start the heartbeat thread");

    println!("VaaniDesk demo window.");
    println!("  F2 add a file    F5 transcribe    Ctrl+Esc cancel    F6 export");
    println!("  F3 / Shift+F3 next / previous match    Ctrl+C copy    Ctrl+A select all");
    println!("  Up / Down walk the transcript, Shift extends the selection.");
    println!("Every transcript is clearly-labelled sample data. Close the window to quit.");

    let backend = FakeController::for_demo();
    // A theme override, so a light-scheme screenshot is one environment variable
    // away rather than an afternoon of changing the system setting.
    let theme = match std::env::var("VAANI_DEMO_THEME").as_deref() {
        Ok("light") => Some(Theme::Light),
        Ok("dark") => Some(Theme::Dark),
        _ => None,
    };
    if theme.is_some() {
        println!("forcing the {theme:?} colour scheme");
    }
    if std::env::var_os("VAANI_DEMO_AUTORUN").is_some() {
        println!("VAANI_DEMO_AUTORUN is set: queueing a job from a worker in two seconds.");
        autorun(backend.clone());
    }

    let session = vaani_ui::build(backend, Arc::new(DemoHost))?;
    if let Some(theme) = theme {
        session.model.set_theme_index(match theme {
            Theme::Light => 1,
            _ => 2,
        });
    }
    session.window.show()?;
    slint::run_event_loop()
}

/// Queue a job from a plain thread.
///
/// This doubles as the smallest possible demonstration of the threading
/// contract. The thread below is not the event loop, it has no handle to the
/// window and no way to touch one, and the job still appears — because the
/// backend is `Send + Sync` and the UI picks the job up from the event channel
/// on its next tick. The production backend does exactly this from its own
/// worker.
fn autorun(backend: Arc<FakeController>) {
    std::thread::Builder::new()
        .name("vaani-ui-autorun".to_owned())
        .spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(2));
            match backend.enqueue(std::path::Path::new("C:/recordings/autorun.mkv")) {
                Ok(id) => println!("queued {id} from another thread"),
                Err(error) => println!("could not queue: {error}"),
            }
        })
        .expect("the OS refused to start the autorun thread");
}

/// Print a heartbeat so a backgrounded run is visibly alive.
fn heartbeat() {
    let start = Instant::now();
    loop {
        std::thread::sleep(Duration::from_secs(5));
        println!("still running, {} s elapsed", start.elapsed().as_secs());
    }
}

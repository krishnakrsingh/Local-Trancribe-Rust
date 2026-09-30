//! Shared test scaffolding: one thread that owns Slint, and a window harness.
//!
//! # Why one thread
//!
//! winit permits exactly **one** `EventLoop` per process, and Slint's winit
//! backend creates one when the platform is installed. `cargo test` runs the
//! tests inside a binary on many threads in parallel, so twenty tests each
//! building a window would fail twenty times with "EventLoop can't be
//! recreated". That is not a limitation of the UI under test; it is a
//! limitation of the toolkit, and the honest response is to run the tests in
//! sequence on one thread rather than to pretend the problem is not there.
//!
//! Each test hands its body to this thread and waits for it, so the tests are
//! serialised but otherwise ordinary: they build the real window, show it, and
//! assert on real state.

#![allow(dead_code)]

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, OnceLock, mpsc};

use slint::ComponentHandle as _;
pub use vaani_ui::fake::{FakeController, RecordingHostServices};
use vaani_ui::slint_api::AppWindow;
use vaani_ui::{HostServices, JobController as _, UiModel, UiSession};

type Job = Box<dyn FnOnce() + Send + 'static>;

struct UiThread {
    jobs: Sender<Job>,
}

fn ui_thread() -> &'static UiThread {
    static THREAD: OnceLock<UiThread> = OnceLock::new();
    THREAD.get_or_init(|| {
        let (tx, rx) = channel::<Job>();
        std::thread::Builder::new()
            .name("vaani-ui-test-ui-thread".to_owned())
            .spawn(move || {
                for job in rx {
                    job();
                }
            })
            .expect("the OS refused to start the UI test thread");
        UiThread { jobs: tx }
    })
}

/// Run `body` on the shared UI thread and wait for it.
///
/// A panic inside `body` is caught and re-raised on the calling thread, so one
/// failing test does not take the thread (and therefore every later test) with
/// it.
pub fn on_ui_thread<F>(body: F)
where
    F: FnOnce() + Send + 'static,
{
    let (done_tx, done_rx) = channel::<std::thread::Result<()>>();
    let job: Job = Box::new(move || {
        let result = catch_unwind(AssertUnwindSafe(body));
        let _ = done_tx.send(result);
    });
    ui_thread()
        .jobs
        .send(job)
        .expect("the UI test thread died; that is a bug in this file");
    match done_rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(payload)) => std::panic::resume_unwind(payload),
        Err(_) => panic!("the UI test thread dropped the job without running it"),
    }
}

/// A window, the model driving it, and the two test doubles.
pub struct Harness {
    /// The built window and controller.
    pub session: UiSession<Arc<FakeController>>,
    /// The recording host, for asserting on dialogs and the clipboard.
    pub host: Arc<RecordingHostServices>,
    /// The backend, for asserting on what the UI asked it to do.
    pub backend: Arc<FakeController>,
}

impl Harness {
    /// A harness with the standard fast backend and a host that answers nothing.
    pub fn new() -> Self {
        Self::with_backend(
            FakeController::for_tests(),
            RecordingHostServices::cancelling(),
        )
    }

    /// A harness with a backend and host of the caller's choosing.
    pub fn with_backend(backend: Arc<FakeController>, host: Arc<RecordingHostServices>) -> Self {
        let host_for_model: Arc<dyn HostServices> = host.clone();
        let session = vaani_ui::build(backend.clone(), host_for_model)
            .expect("the winit backend must be able to create a window adapter");
        Self {
            session,
            host,
            backend,
        }
    }

    /// Build and show, which every test needs because a hidden window is never
    /// laid out and the transcript viewport would report a height of zero.
    pub fn shown() -> Self {
        let harness = Self::new();
        harness.session.window.show().expect("show");
        harness
    }

    /// The window.
    pub fn window(&self) -> &AppWindow {
        &self.session.window
    }

    /// The controller.
    pub fn model(&self) -> &UiModel<Arc<FakeController>> {
        &self.session.model
    }

    /// Drain the backend's events, the way the 33 ms timer would.
    pub fn pump(&self) {
        self.model().pump();
    }

    /// Pump until `predicate` holds, or fail with a useful message.
    pub fn pump_until(&self, what: &str, mut predicate: impl FnMut(&Self) -> bool) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        while std::time::Instant::now() < deadline {
            self.pump();
            if predicate(self) {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        panic!("timed out waiting for {what}");
    }

    /// The job row the window is showing for `id`.
    pub fn job_row(&self, id: &vaani_ui::JobId) -> Option<vaani_ui::slint_api::JobRow> {
        use slint::Model as _;
        let model = self.window().get_jobs();
        (0..model.row_count())
            .filter_map(|i| model.row_data(i))
            .find(|row| row.id.as_str() == id.as_str())
    }

    /// The status text the window is showing for `id`.
    pub fn job_status(&self, id: &vaani_ui::JobId) -> String {
        self.job_row(id)
            .map(|r| r.status_text.to_string())
            .unwrap_or_default()
    }

    /// The id of the job the window has selected, if any.
    pub fn selected_job_id(&self) -> Option<vaani_ui::JobId> {
        let index = self.window().get_selected_job();
        let rows = self.model().job_list().rows();
        if index < 0 {
            return None;
        }
        rows.get(index as usize)
            .map(|r| vaani_ui::JobId::from(r.id.as_str()))
    }

    /// Select the job that has a transcript, and return its row index.
    ///
    /// The seeded history is complete, failed, cancelled in that order, and the
    /// list is newest-first, so the finished job is *not* row 0. Looking it up
    /// beats hard-coding an index that silently rots the moment the history
    /// changes.
    pub fn select_finished(&self) -> i32 {
        let rows = self.model().job_list().rows();
        let index = rows
            .iter()
            .position(|r| {
                self.backend
                    .job(&vaani_ui::JobId::from(r.id.as_str()))
                    .is_some_and(|j| j.has_transcript)
            })
            .expect("the seeded history must contain a finished job");
        self.window()
            .invoke_select_job_requested(i32::try_from(index).unwrap());
        i32::try_from(index).unwrap()
    }
    /// Resize the window and let it lay out.
    ///
    /// The window's own minimums are not the interesting number; the width a
    /// person actually has on a laptop is. Tests that only ever run at the
    /// default size pass on a layout that overflows at 1280px, which is most
    /// laptops.
    pub fn resized(&self, width: i32, height: i32) {
        self.window()
            .window()
            .set_size(slint::LogicalSize::new(width as f32, height as f32));
        // Two pumps: one to apply the new size, one to run the viewport
        // callback that the transcript layout depends on.
        self.pump();
        self.pump();
    }

    /// The window's logical size.
    pub fn size(&self) -> (i32, i32) {
        let s = self.window().window().size();
        (s.width as i32, s.height as i32)
    }
}

/// Declare a test that runs on the shared UI thread.
///
/// Every window test in this crate uses it, because every window test needs a
/// window, and there can only be one event loop.
#[macro_export]
macro_rules! ui_test {
    ($name:ident, $body:block) => {
        #[test]
        fn $name() {
            $crate::common::on_ui_thread(|| $body);
        }
    };
}

/// A channel pair for the tests that need to hear from a worker thread.
pub fn worker_channel<T>() -> (mpsc::Sender<T>, mpsc::Receiver<T>) {
    mpsc::channel()
}

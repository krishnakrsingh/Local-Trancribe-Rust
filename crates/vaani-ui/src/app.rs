//! Starting the window.
//!
//! # Why the entry point is a function and not a `main`
//!
//! This crate is a library. The binary belongs to whichever crate the architect
//! wires the production backend into, because only that crate knows which
//! [`JobController`] and which [`HostServices`] are real. What lives here is the
//! part that is the same either way: build the window, hand it to a
//! [`UiModel`], show it, run the event loop, and survive a backend that fails.
//!
//! # The threading contract, in one paragraph
//!
//! `build` and `run` must be called on the thread that owns the window, which on
//! Windows and macOS is the process's main thread and on Linux is whichever
//! thread initialised the display connection. Everything the UI does happens
//! there. Worker threads never touch Slint: they send [`ControllerEvent`]s down
//! the controller's channel, and [`UiModel::pump`] applies them on the next
//! event-loop tick. A worker that tried to set a property would trip
//! [`UiModel::assert_on_loop`] and panic with a message that says what to do.

use std::rc::Rc;
use std::sync::Arc;

use slint::ComponentHandle;

use crate::backend::{HostServices, JobController};
use crate::fake::FakeController;
use crate::slint_api::AppWindow;
use crate::ui_model::UiModel;

/// A built, not-yet-shown window and the model driving it.
///
/// Returned rather than hidden inside `run` so a test can stand up the real
/// window, drive it, and assert on its state without an event loop.
pub struct UiSession<C: JobController> {
    /// The window. Show it, or not.
    pub window: Rc<AppWindow>,
    /// The controller.
    pub model: Rc<UiModel<C>>,
}

impl<C: JobController> std::fmt::Debug for UiSession<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UiSession")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

/// Build the window and wire it to `controller`.
///
/// Does not show the window and does not run an event loop. Must be called on
/// the event-loop thread.
pub fn build<C: JobController>(
    controller: C,
    host: Arc<dyn HostServices>,
) -> Result<UiSession<C>, slint::PlatformError> {
    let window = Rc::new(AppWindow::new()?);
    let model = UiModel::new(controller, host, &window);
    Ok(UiSession { window, model })
}

/// Build the window, show it, and run the event loop until the window closes.
///
/// Must be called on the thread that owns the window. A backend error surfaces
/// in the window's error banner; it does not take the window down, because a
/// transcription tool that disappears when one file fails is a tool nobody
/// trusts with a folder of recordings.
pub fn run<C: JobController>(
    controller: C,
    host: Arc<dyn HostServices>,
) -> Result<(), slint::PlatformError> {
    let session = build(controller, host)?;
    session.window.show()?;
    slint::run_event_loop()
}

/// Build and show the window against the in-memory demo backend.
///
/// This is what `examples/window.rs` and the architect's first smoke test use.
/// The window carries a permanent "Demo mode" banner, because a run that produced
/// sample data must never be mistaken for a run that produced a transcript.
pub fn run_demo() -> Result<(), slint::PlatformError> {
    run(FakeController::for_demo(), Arc::new(UnavailableHost))
}

/// The [`HostServices`] used by [`run_demo`].
///
/// Dialogs are not available until the architect wires `rfd`, so this one fills
/// the file field with a sample path and leaves the rest to the user. It exists
/// so the demo is fully exercisable end to end without a dialog library.
#[derive(Debug, Default)]
struct UnavailableHost;

impl HostServices for UnavailableHost {
    fn pick_media_files(&self) -> Vec<std::path::PathBuf> {
        Vec::new()
    }

    fn pick_export_target(&self, _suggested_name: &str) -> Option<crate::backend::ExportTarget> {
        None
    }

    fn set_clipboard_text(&self, _text: &str) {}

    fn host_name(&self) -> String {
        "no dialogs yet".to_owned()
    }
}

/// The window's title, for a host that wants to set its own.
pub const WINDOW_TITLE: &str = "VaaniDesk";

//! The native Slint user interface for VaaniDesk.
//!
//! Owned by agent **A5**. The `Cargo.toml` and `build.rs` in this directory are
//! frozen by the architect; `ui/*.slint` and the rest of `src/` are mine.
//!
//! This is a real native window. There is no web view, no localhost server and
//! no browser involved, because a transcription tool that runs a local HTTP
//! server to draw its own interface is a worse tool with more ways to break: it
//! needs a port, it has a security surface, and it fails in ways a native
//! window does not.
//!
//! # Module map
//!
//! | module | owns | must not know about |
//! |---|---|---|
//! | [`backend`] | the [`JobController`](backend::JobController) contract and its DTOs | Slint, the pipeline |
//! | [`view`] | the state the window shows, as plain structs | Slint, the backend |
//! | [`ui_model`] | the Rust/Slint boundary and the threading | nothing above it |
//! | [`fake`] | a deterministic in-memory backend and host | Slint |
//! | [`sample`] | clearly-labelled sample transcripts | everything |
//! | [`app`] | building and running the window | the pipeline |
//!
//! The dependency arrow points one way, from `app` down through `ui_model` to
//! `view` and `backend`. `ui_model` is the only module allowed to name a
//! `slint` type, and `view` is not allowed to name anything from `ui_model`.
//!
//! # How the threading works, and why
//!
//! Slint components are not `Send`, and a property written from a worker thread
//! is undefined behaviour that usually looks fine right up until it is not. So
//! there is exactly one place state changes: [`UiModel`], on the thread that
//! constructed it, which is the event-loop thread. It records that thread and
//! asserts it on every write, so a violation is a panic with a useful message
//! rather than a corrupted property.
//!
//! A job's progress arrives on a worker thread as a
//! [`ControllerEvent`](backend::ControllerEvent) on a crossbeam channel. A
//! 33 ms timer on the event loop drains the channel, collapses the burst down to
//! one update per job ([`view::coalesce`]) and pushes the result into the
//! window. The consequences worth stating:
//!
//! * The event loop never blocks. Not on I/O, not on a mutex a worker might be
//!   holding, not on model loading. All of that is the backend's, on its own
//!   threads.
//! * A fast producer cannot flood the UI. 400 progress events in one frame
//!   become one property write.
//! * Cancellation is immediate in the only sense a user can see: the row says
//!   "Cancelling" on the frame the button was pressed, because that update
//!   happens in the click handler, not on the next poll.
//!
//! # Read these first
//!
//! `docs/agent/PARALLEL_OWNERSHIP.md`, then [`backend`]'s module documentation,
//! which is the contract the production backend is written against.

#![warn(missing_docs)]

// Note on lints: the workspace sets `clippy::all` and leaves it there. This
// crate does not add `pedantic` on top, on purpose. A crate that fails a gate
// on lint levels the rest of the workspace does not enforce trains everyone to
// ignore the gate, and the levels that would fire here are almost all
// `cast_precision_loss` on `u32 -> f32` in windowing arithmetic, where the
// precision loss is irrelevant next to a 22-pixel line height. The casts that
// matter are annotated where they appear.

/// Crate version, from the manifest.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The Slint-generated types.
///
/// Public because the host has to name [`slint_api::AppWindow`] to call
/// [`app::run`], and because the properties and callbacks declared in
/// `ui/app-window.slint` are the integration surface.
///
/// The generated code carries no documentation of its own — the documentation
/// for these types is the `.slint` file, which is where they are written — so
/// the lints are relaxed here rather than fought with attributes in generated
/// code we do not own.
pub mod slint_api {
    #![allow(missing_docs, missing_debug_implementations, clippy::all)]
    slint::include_modules!();
}

pub mod app;
pub mod backend;
pub mod fake;
pub mod sample;
pub mod ui_model;
pub mod view;

pub use app::{UiSession, build, run, run_demo};
pub use backend::{
    ALL_PHASES, AppSettings, Choice, ControllerEvent, ExportFormat, ExportRequest, ExportTarget,
    HostServices, JobController, JobId, JobSummary, SettingsPatch, Severity, SpeakerSummary, Theme,
    UnavailableHostServices,
};
pub use slint_api::AppWindow;
pub use ui_model::{PUMP_INTERVAL, TRANSCRIPT_LINE_HEIGHT_PX, TranscriptLayout, UiModel};

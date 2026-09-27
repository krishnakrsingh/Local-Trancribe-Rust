//! The A2 job pipeline: orchestration plus the UI backend adapter.
//!
//! Owned by agent **A2**. See `docs/agent/PARALLEL_OWNERSHIP.md`: this crate
//! owns `crates/vaani-app/**` and nothing else.
//!
//! # What lives here
//!
//! * [`config`] — [`AppConfig`](config::AppConfig), the pipeline's settings.
//! * [`stages`] — one [`PipelineStage`](vaani_core::pipeline::PipelineStage)
//!   per pipeline step: media, VAD, ASR, diarization, storage, export.
//! * [`orchestrator`] — the stage runner. It owns `StageArtifacts` and
//!   `StageContext` construction, advances the job state machine, fans
//!   progress out, cancels through `CancelToken::child`, retries transient
//!   decode failures, and resumes completed work by content hash.
//! * [`backend`] — [`AppBackend`](backend::AppBackend), the thin
//!   [`JobController`](vaani_ui::backend::JobController) adapter. Every method
//!   returns immediately; all blocking work happens on worker threads.
//!
//! # Rules this crate follows (from `INTEGRATION_CONTRACT.md` §14)
//!
//! * Services never call each other. Only the orchestrator constructs stage
//!   contexts and advances phases.
//! * Diarization takes the audio buffer, never ASR output. Attribution is a
//!   separate step inside the diarization stage.
//! * `Transcript::push_chunk` is the single rebase point for chunk-local ASR
//!   times. Nothing here adds a chunk offset to a `Time`.
//! * Diarization off (or with no engine) skips with a reason, never fails.
//! * Nothing here blocks the Slint event-loop thread.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod backend;
pub mod config;
pub mod orchestrator;
pub mod stages;

#[cfg(test)]
mod fakes;

pub use backend::AppBackend;
pub use config::{AppConfig, DEFAULT_MODEL};
pub use orchestrator::{AsrProvider, EngineProviders, Fanout, JobOutcome, Orchestrator};
pub use stages::Diarize;

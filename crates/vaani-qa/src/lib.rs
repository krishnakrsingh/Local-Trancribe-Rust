//! Cross-crate conformance, regression and end-to-end tests for VaaniDesk.
//!
//! Owned by agent **A10**. This crate exists to be depended on by nothing: it
//! is a test harness, and its whole value is that it is the one place that
//! knows what every crate promised. `cargo test -p vaani-qa` is the
//! cross-crate gate.
//!
//! The most important thing in here is not a test. It is the **conformance
//! harness** in [`conformance`]: a trait that any stage can implement, so that
//! "does this stage honour the contract" is a reusable question with one answer
//! per implementation rather than a bespoke test written each time. The whole
//! contract is seven guarantees — cancellation, empty input, progress,
//! adversarial input, actionable errors, determinism, and the time coordinate
//! space — and one line buys all seven:
//!
//! ```ignore
//! assert_conforms(&EnergyVadStage::new());
//! ```
//!
//! # Modules
//!
//! * [`conformance`] — the [`ConformanceStage`](conformance::ConformanceStage)
//!   trait, the seven `check_*` functions, and adapters over
//!   `vaani_media`, `vaani_vad`, `vaani_diarization` and every `vaani_app`
//!   `PipelineStage`.
//! * [`golden`] — validation of the ASR golden fixtures, with the values
//!   pinned independently of the files they are read from, so a regenerated
//!   golden file cannot silently become the new truth.
//! * [`fixtures`] — synthetic audio and WAV writers. No binary is committed and
//!   no network is touched; every byte of audio under test is synthesised here
//!   from first principles.
//! * [`skip`] — the only four reasons a test may not run, and the machinery that
//!   makes a real failure impossible to turn into a skip.
//!
//! # Privacy
//!
//! The private development recording is never read by the default suite and no
//! test writes transcript content — real or synthesised — to any file, snapshot
//! or fixture. Every fixture is generated in-process, and every assertion is on
//! aggregate structure: counts, timings and invariants. `scratch/` is
//! gitignored and this crate does not defeat that.

pub mod conformance;
pub mod fixtures;
pub mod golden;
pub mod skip;

/// Crate version, from the manifest.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

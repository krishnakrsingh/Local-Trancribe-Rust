//! VaaniDesk accuracy evaluation harness.
//!
//! The project is structurally validated: timings are monotone, chunks are
//! rebased correctly, the transcript round-trips through SQLite, the exports
//! parse. None of that says the words are right, and this crate is the part
//! that would say so — if it were given something true to measure against.
//!
//! # The three error classes, kept apart
//!
//! | class | means | metric |
//! |---|---|---|
//! | **ASR error** | wrong words | WER, CER, and the S/D/I split |
//! | **diarization error** | right words, wrong speaker | DER, JER, speaker count, turn errors |
//! | **timestamp error** | right words and speaker, wrong time | deviation statistics |
//!
//! The harness never combines them. See [`report::BlendRefusal`].
//!
//! # The rule about ground truth
//!
//! Nothing in this crate can produce a reference, and nothing in it will accept
//! a model-produced one. [`hypothesis::Hypothesis::is_reference_grade`] is
//! hard-wired to `false`, so a future code path that tries to score a
//! hypothesis against a hypothesis has to delete a function that says no
//! first. A reference is something a person wrote down after listening to the
//! audio; that is not something this repository can synthesize, and a number
//! derived from one is a self-consistency check with a decimal point on it.
//!
//! # Layout
//!
//! * [`normalize`] — the orthographic rules, numbered and reported.
//! * [`edit`] — Levenshtein with a traceback and a fixed tie-break.
//! * [`wer`] — WER, CER over scalars and over grapheme clusters.
//! * [`rttm`] — NIST RTTM parsing and writing, in integer microseconds.
//! * [`der`] — DER with an exact maximum-weight speaker permutation, JER, turn
//!   metrics.
//! * [`timing`] — timestamp deviation, or `unavailable` with a reason.
//! * [`hypothesis`] — reading the system's own export as a hypothesis.
//! * [`taxonomy`] — the detectable signatures behind
//!   `docs/evaluation/ERROR_TAXONOMY.md`.
//! * [`report`] — the output schema.
//! * [`wav`] — minimal WAV I/O, for cutting clips.
//! * [`probe`] — acoustic structure, for choosing clips without asking the
//!   model what was said.
//! * [`set`] — the evaluation set on disk, and the one scoring entry point.

#![forbid(unsafe_code)]
#![deny(missing_debug_implementations)]

pub mod der;
pub mod edit;
pub mod hypothesis;
pub mod normalize;
pub mod probe;
pub mod report;
pub mod rttm;
pub mod set;
pub mod taxonomy;
pub mod timing;
pub mod wav;
pub mod wer;

/// Crate version, embedded in every report.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

//! Cross-crate conformance and regression tests.
//!
//! Owned by agent **A10**. This crate exists to be depended on by nothing: it
//! is a test harness, and its whole value is that it is the one place that
//! knows what every crate promised. `cargo test -p vaani-qa` is the end-to-end
//! gate.
//!
//! The most important thing in here is not a test. It is the **conformance
//! harness**: a set of traits that any `vaani-*` stage crate can be run against,
//! so "does this stage honour the contract" is a reusable question with one
//! answer per implementation rather than a bespoke test written each time.
//!
//! Read `docs/agent/PARALLEL_OWNERSHIP.md` before the first edit.
//!
//! # Modules
//!
//! * [`conformance`] — the four stage traits plus `run_all_conformance()`.
//! * [`fixtures`] — the tiny synthetic WAV writer. No binary is committed and
//!   no network is touched; every byte of audio under test is synthesised here
//!   from first principles (PCM S16 WAV only).

pub mod conformance;
pub mod fixtures;
pub mod skip;

/// Crate version, from the manifest.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

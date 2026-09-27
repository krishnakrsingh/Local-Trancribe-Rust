//! Grouping recognised words into readable, attributable segments.
//!
//! Owned by agent **A11**. Read `docs/agent/PARALLEL_OWNERSHIP.md` first.
//!
//! This crate exists because the ASR adapter emits **one `Segment` per word**.
//! That is correct at the model boundary and useless as a product: a
//! 14-minute meeting came out as 1,913 words in 1,913 segments, and its SRT had
//! a single-word cue every 436 ms. It also made speaker attribution 1,913
//! independent per-word votes instead of a handful of turn-level decisions, so
//! one word landing in the wrong place could outvote its neighbours.
//!
//! It is a separate crate rather than a step inside `vaani-app` because the CLI
//! and the desktop app both run the pipeline, and a grouping policy that existed
//! in only one of them would give the two products different transcripts for the
//! same recording.

/// Crate version, from the manifest.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

//! The command line entry point and headless job runner.
//!
//! Owned by agent **A8** (`crates/vaani-cli/**`). See
//! `docs/agent/PARALLEL_OWNERSHIP.md` before the first edit.
//!
//! # The two binaries
//!
//! * `vaani.exe` (`src/bin/vaani.rs`) — the GUI launcher. Normal launch opens
//!   the Slint window with **no console** (`#![windows_subsystem = "windows"]`
//!   on that bin only); `--debug` (and any other CLI invocation) reattaches to
//!   the parent console via `AttachConsole`, falling back to `AllocConsole`
//!   under `--debug`. With no arguments it opens the GUI; with arguments it
//!   behaves exactly like `vaani-cli.exe`.
//! * `vaani-cli.exe` (`src/bin/vaani-cli.rs`) — headless. Always has a
//!   console, suitable for scripts and for `benchmark --json` pipelines.
//!
//! Both bins are thin wrappers over [`cli::run_cli`]; all behaviour lives here
//! so it is unit-testable without spawning a process.
//!
//! # The vaani-app seam (read this when vaani-app lands)
//!
//! Until `vaani-app` exists, transcription and benchmarking are implemented by
//! [`pipeline::DirectPipeline`], which wires `vaani-core` + `vaani-media` +
//! `vaani-vad` + `vaani-asr` + `vaani-diarization` + `vaani-storage` +
//! `vaani-export` + `vaani-models` directly. The CLI surface ([`cli::Cli`],
//! [`pipeline::TranscribeRequest`], [`pipeline::BenchmarkReport`]) is defined
//! against the [`pipeline::Pipeline`] trait, **not** against `DirectPipeline`.
//!
//! Swapping to `vaani-app` is a one-module change:
//!
//! 1. Add `vaani-app = { workspace = true }` to this crate's `Cargo.toml`.
//! 2. Implement `Pipeline` for a `vaani-app` adapter (construct
//!    `StageArtifacts` + `StageContext`, advance phases there — the trait
//!    method signatures already match that shape).
//! 3. Change the single construction site in [`cli::run_cli`] from
//!    `DirectPipeline` to the adapter. No subcommand, flag, JSON field, or
//!    test changes.
//!
//! `doctor`, `models`, `jobs`, and `export` already talk to the service crates
//! (`vaani-inference`, `vaani-models`, `vaani-storage`, `vaani-export`)
//! directly and are unaffected by the swap.
//!
//! # Error contract
//!
//! Tests and scripts assert on [`vaani_core::ErrorCode`] (`code()` / `as_str()`),
//! never on message text. Process exit codes: `0` success, `2` CLI usage error
//! (clap), `3` a `VaaniError` reached the top (its `code` is printed as JSON to
//! stderr so scripts can match on it).

/// Crate version, from the workspace.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Product version. Single in-binary source is `vaani_core::APP_VERSION`;
/// re-exported here so `--version` output and JSON reports agree with it.
pub use vaani_core::APP_VERSION;

pub mod cli;
pub mod console;
pub mod pipeline;
pub mod service;

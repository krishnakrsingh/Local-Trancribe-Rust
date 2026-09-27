//! The native Slint user interface.
//!
//! Owned by agent **A5**. The `Cargo.toml` and `build.rs` in this directory are
//! frozen by the architect; `ui/*.slint` and the rest of `src/` are yours.
//!
//! This is a real native window. There is no web view, no localhost server and
//! no browser involved, because a transcription tool that runs a local HTTP
//! server to draw its own interface is a worse tool with more ways to break.
//!
//! Read `docs/agent/PARALLEL_OWNERSHIP.md` before the first edit.

#![warn(missing_docs)]

/// Crate version, from the manifest.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

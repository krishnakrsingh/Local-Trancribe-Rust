//! The command line entry point and headless job runner.
//!
//! Owned by agent **A8**. The `Cargo.toml` in this directory is frozen by
//! the architect and is part of the integration surface; the contents of this
//! file are yours.
//!
//! If you need a dependency that is not in the manifest, do not edit the
//! manifest. Report it under `INTEGRATION NOTES` and the architect will add it
//! in one commit that every agent rebases onto.
//!
//! Read `docs/agent/PARALLEL_OWNERSHIP.md` before the first edit.

/// Crate version, from the workspace.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

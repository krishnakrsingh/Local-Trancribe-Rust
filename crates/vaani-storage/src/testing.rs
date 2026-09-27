//! Test-support helpers.
//!
//! # Why this lives in the library
//!
//! `tempfile` is not in this crate's frozen dependency list and the manifest is
//! architect-owned, so a test cannot create a scratch directory from a crate.
//! The only two options were duplicating a ~40-line helper into every test
//! file, or putting it here once. It is `#[doc(hidden)]` and has no
//! dependencies, so it costs a user of the crate nothing.
//!
//! Everything is `Drop`-scoped: a `TempDir` deletes its directory and
//! everything in it when it goes out of scope, so a failing test cannot leave a
//! 40 MB database behind in `%TEMP%`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A uniquely named directory that deletes itself.
#[derive(Debug)]
pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    /// Create a directory named after `label`, the process id, and a counter so
    /// that two tests with the same label cannot collide.
    pub fn new(label: &str) -> Self {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("vaani-a6-{label}-{}-{n}", std::process::id()));
        // A leftover directory from a killed test run must not make this one
        // fail; remove it first, then create fresh.
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("temp dir must be creatable");
        Self { path }
    }

    /// The directory.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// A path inside the directory. The file is not created.
    ///
    /// Takes anything path-like, so a test can join a `format!`ed name as
    /// easily as a literal.
    pub fn join(&self, name: impl AsRef<Path>) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        // Best effort: a locked file on Windows must not fail a test that has
        // already passed its assertions.
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

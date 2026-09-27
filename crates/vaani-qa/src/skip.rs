//! Loud skips: tests requiring a model or a runtime must SKIP with a printed
//! reason, never silently pass and never fail for lack of a 2.7 GB download.
//!
//! Every helper here prints `SKIP: <test> — <reason>` to stdout (visible with
//! `cargo test -- --nocapture`) and returns a `bool` the caller checks with an
//! early return. The print is the point: a skip nobody can see is a pass that
//! lies.

/// Print a loud skip reason. Call this right before an early `return`.
pub fn skip(test: &str, reason: &str) {
    println!("SKIP: {test} — {reason}");
}

/// Whether the SraVaani artifacts are present under `dir`.
pub fn artifacts_present(dir: &std::path::Path) -> bool {
    vaani_asr::artifacts_in(dir).is_complete()
}

/// Default artifact directory the harness probes for a real model.
pub fn default_artifact_dir() -> std::path::PathBuf {
    vaani_asr::default_dir()
}

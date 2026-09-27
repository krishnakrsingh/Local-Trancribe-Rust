//! Dry-run contract: missing input is a clean error, present input without a
//! live pipeline is a graceful skip. Never touches real media.

use std::path::PathBuf;

fn scratch_file(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("vaani-bench-{name}-{}", std::process::id()));
    std::fs::write(&path, b"not real media").expect("scratch file must be writable");
    path
}

#[test]
fn missing_file_is_a_clean_error() {
    let missing = PathBuf::from("vaani-bench-definitely-not-here-12345.mp4");
    assert!(!missing.exists());
    let err = vaani_bench::run_benchmark(&missing).expect_err("missing file must fail");
    let text = err.to_string();
    assert!(
        text.contains("vaani-bench-definitely-not-here-12345.mp4"),
        "{text}"
    );
    assert!(text.contains("not found"), "{text}");
    assert!(!text.contains("panicked"), "{text}");
}

#[test]
fn present_file_skips_gracefully_without_runtime_or_pipeline() {
    let path = scratch_file("dry-run");
    let report = vaani_bench::run_benchmark(&path).expect("present file must not error");
    std::fs::remove_file(&path).ok();
    assert_eq!(report.status, "skip");
    let note = report.note.as_ref().expect("a skip must print its reason");
    assert!(note.starts_with("SKIP:"), "{note}");
    assert_eq!(report.failures, 0, "a skip is not a failure");
    assert_eq!(report.segments, 0);
    assert_eq!(report.speakers, 0);
    assert!(!report.rtf_is_valid(), "no audio duration was measured");
    assert!(report.wall_ms >= report.startup_ms);
    assert!(report.input.contains("dry-run"));
}

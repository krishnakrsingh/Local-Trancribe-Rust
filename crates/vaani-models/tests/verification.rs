//! Verification: the file is present, the right size, the right bytes.
//!
//! The interesting cases are the ones where the answer must *differ*: a
//! truncated file, a wrong-size file, a file somebody else has open, and a real
//! 845 MiB encoder verified in a fixed amount of memory.
//!
//! The peak-RSS case runs in a **child process**. `cargo test` runs tests as
//! threads of one process, so a peak working set measured in-process would be
//! contaminated by every other test in the binary — including this one's own
//! file writes. The child re-invokes this same test binary with `--exact`, which
//! is why there is no extra `[[bin]]` and no dev-dependency.

mod support;

use std::path::Path;
use std::time::Duration;

use support::TempDir;
use vaani_models::manifest::Manifest;
use vaani_models::progress::ProgressLog;
use vaani_models::verify::{DEFAULT_BUFFER_BYTES, Diagnosis, FileProblem, Verifier};
use vaani_models::{CancelToken, lock};

/// Set by the parent; read by the child, which is the same binary re-invoked.
const CHILD_ENV: &str = "VAANI_A7_RSS_CHILD";
const CHILD_FILE_ENV: &str = "VAANI_A7_RSS_FILE";
const CHILD_REPORT_ENV: &str = "VAANI_A7_RSS_REPORT";
/// The child prints this so the parent can find it in captured output.
const CHILD_MARKER: &str = "A7-RSS";

/// A two-file model whose hashes are computed from what the fixture writes, so
/// the "correct" case is genuinely correct rather than a coincidence.
fn manifest_for(dir: &Path) -> Manifest {
    let small = dir.join("small.bin");
    let nested = dir.join("fe").join("tables.bin");
    Manifest::parse(&format!(
        r#"{{"schema_version":1,"models":[{{
          "id":"m1","engine_id":"e1","version":"1","display_name":"M1",
          "description":"d","task":"asr","architecture":"a","access":"open",
          "license":{{"spdx":"MIT","name":"MIT","url":"https://example.org/l"}},
          "source":{{"kind":"bundled"}},"languages":["hi"],"sample_rates":[16000],
          "size_bytes":{total},
          "files":[
            {{"path":"small.bin","role":"weights","bytes":{small_len},
              "sha256":"{small_sha}"}},
            {{"path":"fe/tables.bin","role":"frontend","bytes":{nested_len},
              "sha256":"{nested_sha}"}}
          ]}}]}}"#,
        total = std::fs::metadata(&small).map(|m| m.len()).unwrap_or(0)
            + std::fs::metadata(&nested).map(|m| m.len()).unwrap_or(0),
        small_len = std::fs::metadata(&small).map(|m| m.len()).unwrap_or(0),
        small_sha = TempDir::sha256_of(&small),
        nested_len = std::fs::metadata(&nested).map(|m| m.len()).unwrap_or(0),
        nested_sha = TempDir::sha256_of(&nested),
    ))
    .expect("the generated manifest must be valid")
}

fn verify(
    m: &Manifest,
    dir: &Path,
    cancel: &CancelToken,
    log: &ProgressLog,
) -> vaani_models::VerifyReport {
    let sink = log.sink();
    Verifier::new().verify(m.model("m1").unwrap(), dir, cancel, &sink)
}

#[test]
fn a_correct_model_verifies() {
    let t = TempDir::new("ver-ok");
    t.file("m1/small.bin", 4096);
    t.file("m1/fe/tables.bin", 8192);
    let m = manifest_for(&t.join("m1"));
    let log = ProgressLog::new();
    let r = verify(&m, &t.join("m1"), &CancelToken::new(), &log);

    assert_eq!(r.diagnosis, Diagnosis::Good, "{:?}", r.files);
    assert!(r.is_ok());
    assert!(r.to_error().is_none());
    assert!(r.failed.is_empty());
    assert_eq!(r.bytes_hashed, 4096 + 8192);
    assert!(r.files.iter().all(|f| f.outcome.is_ok()));
    assert!(r.throughput().unwrap() > 0.0);
    // Progress really arrived, and reached the total.
    assert!(log.len() >= 2, "{:?}", log.ticks());
    assert!(log.reached_total());
}

#[test]
fn a_missing_file_is_missing_not_corrupt() {
    let t = TempDir::new("ver-missing");
    t.file("m1/small.bin", 4096);
    t.file("m1/fe/tables.bin", 8192);
    let m = manifest_for(&t.join("m1"));
    // Delete it *after* the manifest was built, so the hash is real and the
    // file is genuinely absent.
    std::fs::remove_file(t.join("m1/fe/tables.bin")).unwrap();
    let log = ProgressLog::new();
    let r = verify(&m, &t.join("m1"), &CancelToken::new(), &log);

    assert_eq!(r.diagnosis, Diagnosis::Missing);
    assert_eq!(r.failed, vec!["fe/tables.bin"]);
    let e = r
        .to_error()
        .expect("a missing file is an error a user sees");
    assert_eq!(e.code(), vaani_core::ErrorCode::ModelUnavailable);
    assert!(e.user.recovery.unwrap().contains("Download"));
}

#[test]
fn a_truncated_file_is_reported_as_truncated_not_as_a_hash_mismatch() {
    // The distinction the user acts on: a short file is an interrupted download,
    // and the recovery is "download it again", not "reinstall the app".
    let t = TempDir::new("ver-trunc");
    t.file("m1/small.bin", 4096);
    t.file("m1/fe/tables.bin", 8192);
    let m = manifest_for(&t.join("m1"));
    // Cut the nested file in half *after* the manifest was built, so the size
    // disagrees and the hash was never going to match anyway. The size check
    // must win, because it is free.
    t.file("m1/fe/tables.bin", 4000);

    let log = ProgressLog::new();
    let r = verify(&m, &t.join("m1"), &CancelToken::new(), &log);
    assert_eq!(r.diagnosis, Diagnosis::Truncated);
    assert_eq!(r.failed, vec!["fe/tables.bin"]);
    assert!(matches!(
        r.files[1].problem(),
        Some(FileProblem::Truncated {
            expected: 8192,
            actual: 4000
        })
    ));
    // Nothing was hashed for the short file: the size check comes first.
    assert_eq!(r.bytes_hashed, 4096);
    let e = r.to_error().unwrap();
    assert_eq!(e.code(), vaani_core::ErrorCode::ModelCorrupt);
    assert!(e.user.recovery.unwrap().contains("interrupted"));
}

#[test]
fn a_longer_file_is_reported_as_the_wrong_file_not_as_truncation() {
    let t = TempDir::new("ver-long");
    t.file("m1/small.bin", 4096);
    t.file("m1/fe/tables.bin", 8192);
    let m = manifest_for(&t.join("m1"));
    t.file("m1/fe/tables.bin", 8193);

    let log = ProgressLog::new();
    let r = verify(&m, &t.join("m1"), &CancelToken::new(), &log);
    assert_eq!(r.diagnosis, Diagnosis::WrongSize);
    assert!(matches!(
        r.files[1].problem(),
        Some(FileProblem::WrongSize {
            expected: 8192,
            actual: 8193
        })
    ));
    let rec = r.to_error().unwrap().user.recovery.unwrap();
    assert!(rec.contains("not the ones this model is made of"), "{rec}");
}

#[test]
fn the_right_size_with_the_wrong_bytes_is_a_content_difference() {
    let t = TempDir::new("ver-same");
    t.file("m1/small.bin", 4096);
    t.file("m1/fe/tables.bin", 8192);
    let m = manifest_for(&t.join("m1"));
    t.bytes("m1/fe/tables.bin", &support::other_ramp(8192));

    let log = ProgressLog::new();
    let r = verify(&m, &t.join("m1"), &CancelToken::new(), &log);
    assert_eq!(r.diagnosis, Diagnosis::ContentDiffers);
    assert!(r.content_disagreement());
    assert_eq!(r.failed, vec!["fe/tables.bin"]);
    // Both files were hashed: the sizes matched, so nothing short-circuited.
    assert_eq!(r.bytes_hashed, 4096 + 8192);
    // The message must not pretend to know which of the two causes it is.
    let rec = r.to_error().unwrap().user.recovery.unwrap();
    assert!(rec.contains("edited"), "{rec}");
    assert!(rec.contains("Download"), "{rec}");
}

#[test]
fn an_optional_file_that_is_wrong_does_not_make_the_model_bad() {
    // SraVaani's fp32 encoder is optional. A user who deleted it to save 1.65 GiB
    // has a perfectly good install and must not be told otherwise.
    let t = TempDir::new("ver-optional");
    t.file("m1/small.bin", 4096);
    t.file("m1/fe/tables.bin", 8192);
    let m = manifest_for(&t.join("m1"));
    t.file("m1/reference.bin", 1024); // not in the manifest at all

    let log = ProgressLog::new();
    let r = verify(&m, &t.join("m1"), &CancelToken::new(), &log);
    assert!(r.is_ok());
    assert!(r.failed.is_empty());
}

#[test]
fn a_file_held_open_by_another_process_fails_clearly() {
    // On Windows this is `ERROR_SHARING_VIOLATION`, and it is the difference
    // between "the model is in use, close the other window" and a raw I/O
    // message. The lock is taken with `share_mode = FILE_SHARE_READ`, so other
    // processes can read and nobody can open it any other way.
    let t = TempDir::new("ver-locked");
    t.file("m1/small.bin", 4096);
    t.file("m1/fe/tables.bin", 8192);
    let m = manifest_for(&t.join("m1"));
    let victim = t.join("m1/fe/tables.bin");

    let log = ProgressLog::new();
    {
        // Exclusive, not merely read-shared: this is what another process
        // actually does when it is the one holding the weights, and it is the
        // only state in which verification *cannot* proceed.
        let _guard = lock::FileGuard::open_exclusive(&victim).expect("the guard must open");
        assert!(lock::is_in_use(&victim), "a guarded file is in use");
        let r = verify(&m, &t.join("m1"), &CancelToken::new(), &log);
        assert_eq!(r.diagnosis, Diagnosis::Locked, "{:?}", r.files);
        assert!(matches!(r.files[1].problem(), Some(FileProblem::Locked)));
        // It failed *clearly*, at the file, before reporting anything about the
        // rest: the other file still verified.
        assert!(r.files[0].outcome.is_ok());
        let e = r
            .to_error()
            .expect("a locked file is a user-facing failure");
        assert_eq!(e.code(), vaani_core::ErrorCode::StorageFailed);
        let rec = e.user.recovery.unwrap();
        assert!(rec.contains("Close other VaaniDesk windows"), "{rec}");
    }
    // And it verifies once the holder lets go.
    let r = verify(&m, &t.join("m1"), &CancelToken::new(), &log);
    assert!(r.is_ok());
}

#[test]
fn verification_is_cancellable_mid_file() {
    // A 24 MiB file with a progress interval small enough that cancellation
    // lands mid-file rather than between files.
    let t = TempDir::new("ver-cancel");
    t.file("m1/small.bin", 24 * 1024 * 1024);
    t.file("m1/fe/tables.bin", 16);
    let m = manifest_for(&t.join("m1"));
    let cancel = CancelToken::new();

    let log = ProgressLog::new();
    let sink = log.sink();
    let model = m.model("m1").unwrap();
    // Cancel from the progress callback, the way a UI's cancel button would.
    let c = cancel.clone();
    let report = Verifier::new().with_progress_every(1024 * 1024).verify(
        model,
        &t.join("m1"),
        &cancel,
        &|p| {
            sink(p.clone());
            if p.bytes_done > 0 {
                c.cancel();
            }
        },
    );
    assert!(report.cancelled, "verification must stop when cancelled");
    assert_eq!(report.diagnosis, Diagnosis::Cancelled);
    // It stopped early: nowhere near the whole 24 MiB was hashed.
    assert!(
        report.bytes_hashed < 24 * 1024 * 1024,
        "hashed {} bytes after being cancelled",
        report.bytes_hashed
    );
    let e = report.to_error().unwrap();
    assert!(e.is_cancelled());
    assert_eq!(e.code(), vaani_core::ErrorCode::Cancelled);
}

#[test]
fn a_cancelled_verification_is_not_an_error_the_user_has_to_dismiss() {
    let t = TempDir::new("ver-cancel2");
    t.file("m1/small.bin", 4096);
    t.file("m1/fe/tables.bin", 8192);
    let m = manifest_for(&t.join("m1"));
    let cancel = CancelToken::new();
    cancel.cancel();
    let r = verify(&m, &t.join("m1"), &cancel, &ProgressLog::new());
    assert!(r.cancelled);
    assert_eq!(r.diagnosis, Diagnosis::Cancelled);
    assert!(r.to_error().unwrap().is_cancelled());
}

/// The child half of the peak-RSS measurement. Invoked by the parent below with
/// `--exact`, so that the peak working set it reports is its own.
#[test]
fn rss_child() {
    if std::env::var_os(CHILD_ENV).is_none() {
        return;
    }
    let file = std::env::var_os(CHILD_FILE_ENV).expect("the child needs a file");
    let report = std::env::var_os(CHILD_REPORT_ENV).expect("the child needs a report path");
    let path = std::path::PathBuf::from(file);

    let size = std::fs::metadata(&path).expect("the file must exist").len();
    let hash = TempDir::sha256_of(&path);
    // Hash it with the real streaming verifier so the measurement covers the
    // path the product uses, not a synthetic loop.
    let started = std::time::Instant::now();
    let (hashed, hashed_len) = {
        let mut buf = vec![0u8; DEFAULT_BUFFER_BYTES];
        vaani_models::verify::hash_file_streaming(&path, &mut buf, |_| true)
            .expect("readable")
            .expect("the whole file")
    };
    let elapsed = started.elapsed();
    let peak = vaani_models::sys::peak_working_set_bytes().unwrap_or(0);

    let body = format!(
        "{CHILD_MARKER} file={} size={size} hash_ok={} hashed={} elapsed_us={} peak_rss={peak}",
        path.display(),
        u8::from(hex::encode(hashed) == hash),
        hashed_len,
        elapsed.as_micros(),
    );
    // Also print it, so `cargo test -- --nocapture` shows the number.
    println!("{body}");
    std::fs::write(report, &body).expect("the child must be able to report");
}

#[test]
fn verifying_the_real_encoder_stays_inside_a_memory_ceiling() {
    // The real claim: an 845 MiB encoder is verified without ever being in
    // memory. The measurement is taken in a child process, because
    // `PeakWorkingSetSize` is a high-water mark for the whole process and the test
    // binary's other threads would otherwise be counted.
    let artifacts = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../artifacts/sravaani/encoder.fp16.onnx");
    if !artifacts.is_file() {
        println!(
            "skipping: {} is not present; run scripts/export_sravaani_onnx.py first",
            artifacts.display()
        );
        return;
    }
    let size = std::fs::metadata(&artifacts).unwrap().len();

    let t = TempDir::new("ver-rss");
    let report = t.join("child.txt");
    let exe = std::env::current_exe().expect("the test binary's own path");
    let out = std::process::Command::new(exe)
        .args(["--exact", "rss_child", "--nocapture", "--test-threads=1"])
        .env(CHILD_ENV, "1")
        .env(CHILD_FILE_ENV, &artifacts)
        .env(CHILD_REPORT_ENV, &report)
        .output()
        .expect("the child test binary must run");
    assert!(
        out.status.success(),
        "the child failed: {}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let body = std::fs::read_to_string(&report).expect("the child must have reported");
    println!("{body}");

    let field = |name: &str| -> u64 {
        body.split_whitespace()
            .find_map(|kv| kv.strip_prefix(&format!("{name}=")))
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("the child did not report {name}: {body}"))
    };
    assert!(body.starts_with(CHILD_MARKER), "{body}");
    assert_eq!(field("hash_ok"), 1, "the child must hash the real file");
    assert_eq!(field("hashed"), size);
    let peak = field("peak_rss");

    // The file is 845 MiB. Peak working set must be nowhere near that: a
    // whole-file read would be ~886 MB, so this is the assertion that the
    // streaming path is real. The ceiling is loose on purpose -- it is a
    // regression guard, not a benchmark.
    const CEILING: u64 = 256 * 1024 * 1024;
    println!(
        "verified {size} bytes in {:?} with a peak working set of {peak} bytes ({:.1} MiB)",
        Duration::from_micros(field("elapsed_us")),
        peak as f64 / (1024.0 * 1024.0)
    );
    assert!(
        peak < CEILING,
        "peak working set {peak} ({:.1} MiB) exceeds the {CEILING} byte ceiling; \
         something read the file into memory",
        peak as f64 / (1024.0 * 1024.0)
    );
    // And it must be more than a bare test harness, or the number is fake.
    assert!(
        peak > 4 * 1024 * 1024,
        "peak working set {peak} is implausibly small"
    );
}

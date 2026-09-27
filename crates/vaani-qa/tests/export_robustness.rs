//! Export robustness: determinism, read-only / missing targets, overwrite
//! policy, JSON round-trip, all five formats.

use vaani_core::{EngineStamp, ErrorCode, Segment, Time, Transcript, Word};
use vaani_export::{ExportFormat, ExportOptions, ExportTarget, export_to_file};
use vaani_qa::conformance::sample_transcript;

fn rich_transcript() -> Transcript {
    let mut t = Transcript::new(EngineStamp::asr("qa-export", "1.0"));
    for i in 0..6 {
        let start = i * 3000;
        let mut seg = Segment::new(Time(start), Time(start + 2500));
        seg.id = format!("seg-{i}");
        seg.words = vec![
            Word::new("namaste", Time(start), Time(start + 1200)),
            Word::new("dost", Time(start + 1200), Time(start + 2500)),
        ];
        seg.speaker = Some(i as usize % 2);
        t.segments.push(seg);
    }
    t.add_speaker(vaani_core::Speaker::new(0));
    t.add_speaker(vaani_core::Speaker::new(1));
    t
}

#[test]
fn all_five_formats_write_and_reread() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let t = rich_transcript();
    let options = ExportOptions::default().with_source_name("meeting.m4a");
    for format in ExportFormat::ALL {
        let written = export_to_file(
            &t,
            format,
            &ExportTarget::Directory(dir.path().to_path_buf()),
            &options,
        )
        .unwrap_or_else(|e| panic!("{format}: {e}"));
        assert!(written.bytes > 0, "{format} wrote nothing");
        assert!(written.path.is_file(), "{format} missing");
    }
    let mut names: Vec<String> = std::fs::read_dir(dir.path())
        .expect("read dir")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "meeting.json",
            "meeting.md",
            "meeting.srt",
            "meeting.txt",
            "meeting.vtt"
        ]
    );
}

#[test]
fn export_is_deterministic_for_fixed_options() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let t = sample_transcript();
    let options = ExportOptions::default().with_source_name("a.wav");
    for format in ExportFormat::ALL {
        let target = ExportTarget::Directory(dir.path().to_path_buf());
        let first = export_to_file(&t, format, &target, &options).expect("first");
        let first_bytes = std::fs::read(&first.path).expect("read first");
        let overwrite = ExportOptions {
            overwrite: true,
            ..options.clone()
        };
        let second = export_to_file(&t, format, &target, &overwrite).expect("second");
        let second_bytes = std::fs::read(&second.path).expect("read second");
        assert_eq!(first_bytes, second_bytes, "{format} nondeterministic");
    }
}

#[test]
fn json_export_round_trips_to_equal_transcript() {
    let t = rich_transcript();
    let text = vaani_export::render(&t, ExportFormat::Json, &ExportOptions::default())
        .expect("render json");
    let back = vaani_export::read_json_export(&text).expect("read json");
    assert_eq!(back, t);
}

#[test]
fn export_failure_into_missing_parent_is_export_failed() {
    let t = sample_transcript();
    let missing = std::path::PathBuf::from("Z:\\vaani-qa-definitely-missing-dir-9f3\\out.srt");
    // If that drive exists on the machine, fall back to a file-as-parent.
    let target = if missing.parent().is_some_and(|p| p.exists()) {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let blocker = dir.path().join("blocker");
        std::fs::write(&blocker, "block").expect("blocker");
        ExportTarget::File(blocker.join("out.srt"))
    } else {
        ExportTarget::File(missing)
    };
    let err = export_to_file(&t, ExportFormat::Srt, &target, &ExportOptions::default())
        .expect_err("must fail");
    assert_eq!(err.code(), ErrorCode::ExportFailed);
}

#[test]
fn export_failure_when_parent_is_a_file() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let blocker = dir.path().join("blocker");
    std::fs::write(&blocker, "block").expect("blocker");
    let t = sample_transcript();
    let target = ExportTarget::File(blocker.join("out.txt"));
    let err = export_to_file(&t, ExportFormat::Text, &target, &ExportOptions::default())
        .expect_err("must fail");
    assert_eq!(err.code(), ErrorCode::ExportFailed);
}

#[test]
fn existing_file_is_not_clobbered_by_default() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let t = sample_transcript();
    let options = ExportOptions::default().with_source_name("a.wav");
    let target = ExportTarget::Directory(dir.path().to_path_buf());
    export_to_file(&t, ExportFormat::Text, &target, &options).expect("first");
    let path = dir.path().join("a.txt");
    std::fs::write(&path, "mine").expect("overwrite fixture");
    let err = export_to_file(&t, ExportFormat::Text, &target, &options).expect_err("must refuse");
    assert_eq!(err.code(), ErrorCode::ExportFailed);
    assert_eq!(std::fs::read_to_string(&path).expect("read"), "mine");
    // Explicit overwrite succeeds.
    let overwrite = ExportOptions {
        overwrite: true,
        ..options
    };
    export_to_file(&t, ExportFormat::Text, &target, &overwrite).expect("overwrite");
    assert_ne!(std::fs::read_to_string(&path).expect("read"), "mine");
}

#[test]
fn insufficient_disk_is_reported_not_panicked() {
    // Mock a full disk with a zero-byte free-space probe: the models
    // downloader's precheck must map it to InsufficientDisk.
    use std::sync::Arc;
    use vaani_models::download::{Downloader, FixedFreeSpace};
    let downloader = Downloader::with_policy(
        vaani_models::download::DownloadPolicy::single_attempt(),
        Arc::new(FixedFreeSpace(Some(0))),
    );
    let dir = tempfile::TempDir::new().expect("tempdir");
    let outcome = downloader.precheck(dir.path(), 1024 * 1024);
    match outcome {
        Ok(_) => panic!("a full disk must not precheck clean"),
        Err(e) => {
            let vaani = e.to_vaani_error();
            assert_eq!(vaani.code(), ErrorCode::InsufficientDisk);
        }
    }
}

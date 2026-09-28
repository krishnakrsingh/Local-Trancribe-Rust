//! Writing files: atomicity, overwrite policy, and staying inside the directory
//! the user chose.
//!
//! These are the properties that turn "export" from a print statement into
//! something a user can trust with an hour of work. Each test is written as the
//! thing that would go wrong, not as the function that would be called.

use std::path::{Path, PathBuf};

use vaani_core::{ErrorCode, MediaLength};
use vaani_export::{
    ExportFormat, ExportOptions, ExportTarget, export_all, export_to_file, safe_file_name,
    write_atomic,
};
use vaani_storage::fixtures;
use vaani_storage::testing::TempDir;

fn transcript() -> vaani_core::Transcript {
    fixtures::transcript(0x5AFE, 30)
}

fn files_in(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .map_err(|e| panic!("{}: {e}", dir.display()))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

#[test]
fn an_export_never_leaves_a_temporary_file_behind() {
    let dir = TempDir::new("no-temp");
    let o = ExportOptions::default().with_source_name("standup.m4a");
    for f in ExportFormat::ALL {
        let target = ExportTarget::Directory(dir.path().to_path_buf());
        export_to_file(&transcript(), f, &target, &o).unwrap();
    }
    let names = files_in(dir.path());
    assert_eq!(names.len(), 5, "{names:?}");
    for name in &names {
        assert!(!name.ends_with(".tmp"), "left a temp file: {name}");
        assert!(!name.starts_with('.'), "left a hidden temp file: {name}");
    }
}

#[test]
fn a_failed_export_does_not_disturb_the_files_already_there() {
    let dir = TempDir::new("failed-keeps");
    let o = ExportOptions::default().with_source_name("a.wav");
    let target = ExportTarget::Directory(dir.path().to_path_buf());
    export_to_file(&transcript(), ExportFormat::Srt, &target, &o).unwrap();
    let before = std::fs::read(dir.join("a.srt")).unwrap();

    // The second export cannot write: the target is a directory.
    std::fs::create_dir(dir.join("a.txt")).unwrap();
    let e = export_to_file(&transcript(), ExportFormat::Text, &target, &o).unwrap_err();
    assert_eq!(e.code(), ErrorCode::ExportFailed);
    assert!(e.user.recovery.is_some());
    // The subtitle file is byte-identical to what it was.
    assert_eq!(std::fs::read(dir.join("a.srt")).unwrap(), before);
    // And no temp file survived the failure.
    assert!(!files_in(dir.path()).iter().any(|n| n.ends_with(".tmp")));
}

#[test]
fn an_existing_file_is_replaced_only_when_asked() {
    let dir = TempDir::new("replace");
    let path = dir.join("out.srt");
    write_atomic(&path, "the user's own file", false).unwrap();

    let t = transcript();
    let target = ExportTarget::File(path.clone());
    let o = ExportOptions::default();

    let e = export_to_file(&t, ExportFormat::Srt, &target, &o).unwrap_err();
    assert_eq!(e.code(), ErrorCode::ExportFailed);
    assert!(
        e.user.detail.contains("out.srt"),
        "the message must name the file"
    );
    assert!(e.user.recovery.unwrap().contains("different name"));
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        "the user's own file",
        "refusing must not have touched the file"
    );

    let written = export_to_file(
        &t,
        ExportFormat::Srt,
        &target,
        &o.clone().with_overwrite(true),
    )
    .unwrap();
    assert!(
        std::fs::read_to_string(&written.path)
            .unwrap()
            .contains("-->")
    );
}

#[test]
fn a_crafted_source_name_cannot_write_outside_the_target_directory() {
    // The name comes from the *media file*, which the user (or a download) can
    // call anything. Every one of these must land inside the directory.
    let dir = TempDir::new("escape");
    let outside = TempDir::new("escape-outside");
    let marker = outside.join("pwned.txt");
    let t = transcript();

    for hostile in [
        "../../pwned.txt",
        "..\\..\\pwned.txt",
        "/etc/pwned.txt",
        "C:\\Windows\\pwned.txt",
        "..",
        ".",
        "....//....//pwned.txt",
        "subdir/../../pwned.txt",
    ] {
        // Overwrite is enabled so that several hostile names which sanitise to
        // the same file do not collide with each other; the property under test
        // is *where* the file lands, not whether it was allowed to exist.
        let o = ExportOptions::default()
            .with_source_name(hostile)
            .with_overwrite(true);
        let target = ExportTarget::Directory(dir.path().to_path_buf());
        let written = export_to_file(&t, ExportFormat::Text, &target, &o)
            .unwrap_or_else(|e| panic!("{hostile}: {e:?}"));
        let parent = written
            .path
            .parent()
            .expect("every export has a parent directory");
        assert_eq!(
            parent.canonicalize().unwrap(),
            dir.path().canonicalize().unwrap(),
            "{hostile} wrote outside the directory: {}",
            written.path.display()
        );
    }
    assert!(!marker.exists(), "a file was written outside the target");
    // Every hostile name collapsed onto the same safe file, and nothing else is
    // in the directory.
    let names = files_in(dir.path());
    assert!(
        names
            .iter()
            .all(|n| n == "pwned.txt" || n == "transcript.txt"),
        "{names:?}"
    );
    assert!(files_in(outside.path()).is_empty());
}

#[test]
fn an_explicit_file_target_is_the_users_decision_even_if_it_looks_odd() {
    // A save dialog is the user saying "exactly here". A name with no extension,
    // or a different one, is their call — the safety rules apply to names *we*
    // invent, not to a path the user chose.
    let dir = TempDir::new("explicit");
    let path = dir.join("no-extension");
    let target = ExportTarget::File(path.clone());
    let written = export_to_file(
        &transcript(),
        ExportFormat::Srt,
        &target,
        &ExportOptions::default(),
    )
    .unwrap();
    assert_eq!(written.path, path);
    assert!(path.exists());
}

#[test]
fn a_file_name_is_always_something_writable() {
    // The sanitiser has to produce a name for every input, including the ones
    // that are not names at all.
    for hostile in [
        "",
        ".",
        "..",
        "...",
        "///",
        "////",
        "con",
        "NUL.txt",
        "a".repeat(400).leak(),
        "\u{0}\u{1}\u{2}",
        "trailing...",
        "trailing   ",
        "<>|:*?\"",
        "résumé — 2026",
    ] {
        let safe = safe_file_name(hostile);
        assert!(!safe.is_empty(), "{hostile:?} produced nothing");
        assert!(
            !safe.contains(['/', '\\', ':', '*', '?', '"', '<', '>', '|']),
            "{hostile:?} produced {safe:?}, which Windows will not accept"
        );
        assert!(!safe.contains(".."), "{hostile:?} produced {safe:?}");
        assert!(!safe.ends_with(['.', ' ']), "{hostile:?} produced {safe:?}");
        assert!(safe.chars().count() <= 130, "{hostile:?} produced {safe:?}");
    }
}

#[test]
fn exporting_every_format_stops_at_the_first_failure() {
    // "Export to…" offering all five formats must not leave four new files and
    // one error: the user has to be able to trust what is on disk.
    let dir = TempDir::new("all-or-nothing");
    let t = transcript();
    let o = ExportOptions::default().with_source_name("a.wav");
    // Block the *first* format by occupying its name with a directory.
    std::fs::create_dir(dir.join("a.txt")).unwrap();
    let e = export_all(&t, &ExportFormat::ALL, dir.path(), &o).unwrap_err();
    assert_eq!(e.code(), ErrorCode::ExportFailed);
    let names = files_in(dir.path());
    assert_eq!(
        names,
        ["a.txt"],
        "nothing else should have been written: {names:?}"
    );
}

#[test]
fn the_export_log_records_what_a_pipeline_should_record() {
    // What the pipeline writes into `exports` after a successful export, and
    // that the record matches the file on disk.
    let dir = TempDir::new("log");
    let store = vaani_storage::Store::open(dir.join("v.db")).unwrap();
    let t = transcript();
    let job = vaani_storage::JobRecord::new("j", dir.join("a.wav"));
    store.upsert_job(&job).unwrap();
    store.replace_transcript("j", &t).unwrap();

    let o = ExportOptions::default().with_source_name("a.wav");
    for f in ExportFormat::ALL {
        let target = ExportTarget::Directory(dir.path().to_path_buf());
        let written = export_to_file(&t, f, &target, &o).unwrap();
        let record =
            vaani_storage::ExportRecord::new(Some("j"), &written.path, f.as_str(), written.bytes);
        store.record_export(&record).unwrap();
    }
    let log = store.list_exports(Some("j")).unwrap();
    assert_eq!(log.len(), 5);
    for entry in &log {
        let format = ExportFormat::parse(&entry.format).expect("a known format");
        assert!(entry.path.ends_with(&format!(".{}", format.extension())));
        let on_disk = std::fs::metadata(entry.path()).unwrap().len();
        assert_eq!(
            entry.bytes as u64, on_disk,
            "the log lies about {}",
            entry.format
        );
        assert!(entry.created_at_ms > 0);
    }
}

#[test]
fn a_very_long_transcript_exports_in_reasonable_time_and_space() {
    // Not a benchmark: a guard against an accidental quadratic in the writer.
    let dir = TempDir::new("long");
    let t = fixtures::transcript(1, 2_000);
    let o = ExportOptions::default().with_source_name("long.wav");
    let target = ExportTarget::Directory(dir.path().to_path_buf());
    let began = std::time::Instant::now();
    let srt = export_to_file(&t, ExportFormat::Srt, &target, &o).unwrap();
    let took = began.elapsed();
    assert!(srt.cues > 100, "expected a real number of cues");
    assert!(took.as_secs() < 10, "exporting took {took:?}");
    let bytes = std::fs::metadata(&srt.path).unwrap().len();
    assert!(bytes > 10_000, "the file looks too small: {bytes}");
}

#[test]
fn an_export_of_a_transcript_with_no_media_name_uses_a_neutral_filename() {
    // A transcript pasted from a chat window has no source file. It still needs a
    // name, and it must not be an empty one.
    let dir = TempDir::new("no-name");
    let target = ExportTarget::Directory(dir.path().to_path_buf());
    let written = export_to_file(
        &transcript(),
        ExportFormat::Markdown,
        &target,
        &ExportOptions::default(),
    )
    .unwrap();
    assert_eq!(written.path.file_name().unwrap(), "transcript.md");
    assert_eq!(written.path.parent().unwrap(), dir.path());
}

#[test]
fn writing_to_a_directory_that_does_not_exist_says_so() {
    let dir = TempDir::new("missing");
    let target = ExportTarget::Directory(dir.join("nope/deeper"));
    let e = export_to_file(
        &transcript(),
        ExportFormat::Srt,
        &target,
        &ExportOptions::default().with_source_name("a.wav"),
    )
    .unwrap_err();
    assert_eq!(e.code(), ErrorCode::ExportFailed);
    assert!(e.user.recovery.is_some());
}

#[test]
fn the_written_file_is_utf8_and_ends_with_a_newline() {
    // Two small things players and `cat` care about, and which are easy to lose.
    let dir = TempDir::new("utf8");
    let t = fixtures::single_word();
    let o = ExportOptions::default().with_source_name("a.wav");
    for f in [
        ExportFormat::Text,
        ExportFormat::Srt,
        ExportFormat::Vtt,
        ExportFormat::Markdown,
    ] {
        let target = ExportTarget::Directory(dir.path().to_path_buf());
        let written = export_to_file(&t, f, &target, &o).unwrap();
        let bytes = std::fs::read(&written.path).unwrap();
        let text = String::from_utf8(bytes).expect("the file must be valid UTF-8");
        assert!(!text.contains('\r'), "{f} contains a carriage return");
        assert!(text.ends_with('\n'), "{f} does not end with a newline");
        assert!(!text.starts_with('\u{feff}'), "{f} has a byte-order mark");
    }
}

#[test]
fn export_returns_the_cue_count_the_pipeline_can_log() {
    let dir = TempDir::new("cues");
    let t = transcript();
    let o = ExportOptions::default().with_source_name("a.wav");
    let target = ExportTarget::Directory(dir.path().to_path_buf());
    let srt = export_to_file(&t, ExportFormat::Srt, &target, &o).unwrap();
    let expected = vaani_export::cues::build_cues(&t, &o.cue_options()).len();
    assert_eq!(srt.cues, expected);
    assert_eq!(
        srt.cues,
        vaani_export::verify::parse_srt(&std::fs::read_to_string(&srt.path).unwrap())
            .unwrap()
            .len()
    );
}

#[test]
fn timestamps_in_the_file_match_the_transcript_exactly() {
    // The subtitle times are the viewer's times, to the millisecond. If a future
    // change rounds them, this fails.
    let dir = TempDir::new("times");
    let mut t = fixtures::transcript(2, 3);
    t.duration = Some(MediaLength::from_millis(7_825_400));
    let o = ExportOptions::default().with_source_name("a.wav");
    let target = ExportTarget::Directory(dir.path().to_path_buf());
    let srt = export_to_file(&t, ExportFormat::Srt, &target, &o).unwrap();
    let parsed =
        vaani_export::verify::parse_srt(&std::fs::read_to_string(&srt.path).unwrap()).unwrap();
    let built = vaani_export::cues::build_cues(&t, &o.cue_options());
    assert_eq!(parsed.len(), built.len());
    for (p, b) in parsed.iter().zip(built.iter()) {
        assert_eq!(p.start.as_millis(), b.start.as_millis());
        assert_eq!(p.end.as_millis(), b.end.as_millis());
    }
}

#[test]
fn a_second_process_cannot_interleave_into_a_half_written_file() {
    // Two exports of the same target at once: the loser's write must not produce
    // a file that is a mixture of the two. Either one complete file or the
    // other, never a splice, because each write is a rename over the target.
    let dir = TempDir::new("race");
    let path: PathBuf = dir.join("out.srt");
    let big = fixtures::transcript(1, 800);
    let small = fixtures::single_word();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let handles: Vec<_> = [big, small]
        .into_iter()
        .map(|t| {
            let path = path.clone();
            let barrier = std::sync::Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                write_atomic(
                    &path,
                    &vaani_export::render(&t, ExportFormat::Srt, &ExportOptions::default())
                        .unwrap(),
                    true,
                )
                .unwrap();
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let final_text = std::fs::read_to_string(&path).unwrap();
    // Whichever won, the file is one complete export, so it parses cleanly.
    if !final_text.is_empty() {
        vaani_export::verify::validate_srt(&final_text, 2)
            .unwrap_or_else(|e| panic!("a spliced file was produced: {e:?}"));
    }
}

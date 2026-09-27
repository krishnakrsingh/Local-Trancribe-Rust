//! Export robustness: all five formats, determinism, targets, overwrite policy,
//! and the subtitle files read back by the reference parsers.
//!
//! # What changed, and why
//!
//! * `insufficient_disk_is_reported_not_panicked` was in this file. It tests
//!   `vaani_models::download`'s free-space precheck, which has nothing to do with
//!   exporting a transcript. Misfiled tests are how suites grow without growing
//!   coverage; it has moved to `tests/models_runtime.rs`, where the crate it
//!   exercises is named in the filename.
//! * `export_failure_into_missing_parent_is_export_failed` and
//!   `export_failure_when_parent_is_a_file` were the same test twice, and the
//!   first reached for a `Z:\` drive that may or may not exist on the machine.
//!   They are now one test that always uses a temp directory, so it means the
//!   same thing everywhere.
//! * The subtitle exports are now checked with `vaani_export::verify`, the
//!   reference parsers, rather than by asserting a byte count. `docs/JOURNEY.md`
//!   still lists "no SRT or VTT parser has validated the export" as open, and a
//!   byte count is the reason it was open.

use vaani_core::{EngineStamp, ErrorCode, Segment, Time, Transcript, Word};
use vaani_export::{ExportFormat, ExportOptions, ExportTarget, export_to_file};
use vaani_qa::conformance::{check_errors_are_actionable, check_export_stage, sample_transcript};

/// A transcript with segment ids, multi-word segments, two speakers and a
/// deliberate gap, so the renderers are not all fed one uniform shape.
fn rich_transcript() -> Transcript {
    let mut t = Transcript::new(EngineStamp::asr("qa-export", "1.0"));
    for i in 0..6i64 {
        let start = i * 3_000;
        let mut seg = Segment::new(Time(start), Time(start + 2_500));
        seg.id = format!("seg-{i}");
        seg.words = vec![
            Word::new("alpha", Time(start), Time(start + 1_200)),
            Word::new("bravo", Time(start + 1_200), Time(start + 2_500)),
        ];
        seg.speaker = Some((i % 2) as usize);
        t.push_chunk(Time(0), vec![seg]);
    }
    t.add_speaker(vaani_core::Speaker::new(0));
    t.add_speaker(vaani_core::Speaker::new(1));
    t
}

#[test]
fn all_five_formats_write_a_non_empty_file_named_after_the_source() {
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
        .unwrap_or_else(|e| panic!("{format}: {}", e.debug_chain()));
        assert!(written.bytes > 0, "{format} wrote nothing");
        assert!(
            written.path.is_file(),
            "{format}: {}",
            written.path.display()
        );
        assert_eq!(
            written.path.file_name().unwrap().to_string_lossy(),
            format!("meeting.{}", format.extension()),
            "{format} was not named after the source media"
        );
        // A file of the right length that is not readable text is not an export.
        let text = std::fs::read_to_string(&written.path)
            .unwrap_or_else(|e| panic!("{format} is not UTF-8: {e}"));
        assert!(!text.trim().is_empty(), "{format} is whitespace");
        assert_eq!(
            format.is_cue_based(),
            written.cues > 0,
            "{format} reports {} cues",
            written.cues
        );
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
        ],
        "exactly the five formats, nothing else"
    );
}

#[test]
fn the_subtitle_formats_parse_under_the_published_rules() {
    // `vaani_export::verify` holds hand-written parsers implementing the SubRip
    // and WebVTT specifications. This proves the bytes this crate writes parse,
    // including the parts people get wrong: the comma against the dot, cue
    // numbering from 1, blank-line termination, and one cue ending before the next
    // begins.
    //
    // It is a weaker guarantee than a real player, and says so: see the release
    // checklist item in `docs/JOURNEY.md`. What it does prove is that no export
    // is malformed in a way a strict parser would reject.
    let dir = tempfile::TempDir::new().expect("tempdir");
    let t = rich_transcript();
    let options = ExportOptions::default().with_source_name("a.wav");
    let target = ExportTarget::Directory(dir.path().to_path_buf());

    let srt = export_to_file(&t, ExportFormat::Srt, &target, &options).expect("srt");
    let cues = vaani_export::verify::validate_srt(
        &std::fs::read_to_string(&srt.path).expect("read srt"),
        2,
    )
    .unwrap_or_else(|e| panic!("srt does not parse: {}", e.debug_chain()));
    assert!(!cues.is_empty());
    assert_eq!(
        cues.first().and_then(|c| c.index),
        Some(1),
        "cue numbering starts at 1"
    );
    for (i, c) in cues.iter().enumerate() {
        assert!(c.end > c.start, "cue {} has end <= start", i + 1);
        assert!(
            c.lines.len() <= 2,
            "cue {} has {} lines; the readability budget is two",
            i + 1,
            c.lines.len()
        );
    }
    for w in cues.windows(2) {
        assert!(
            w[0].end <= w[1].start,
            "cues overlap: {} ends at {} and the next starts at {}",
            w[0].end.as_millis(),
            w[0].end.as_millis(),
            w[1].start.as_millis()
        );
    }
    // The comma is the SubRip millisecond separator; a dot makes the file
    // unparseable by strict readers even though players often accept it.
    let raw = std::fs::read_to_string(&srt.path).expect("read srt");
    assert!(
        raw.contains("-->") && raw.contains(',') && !raw.contains('.'),
        "the SRT timestamp line must use a comma before the milliseconds"
    );

    let vtt = export_to_file(&t, ExportFormat::Vtt, &target, &options).expect("vtt");
    let vtt_text = std::fs::read_to_string(&vtt.path).expect("read vtt");
    assert!(
        vtt_text.starts_with("WEBVTT"),
        "a WebVTT file must open with the signature; got {:?}",
        &vtt_text[..vtt_text.len().min(20)]
    );
    let vcues = vaani_export::verify::validate_vtt(&vtt_text, 2)
        .unwrap_or_else(|e| panic!("vtt does not parse: {}", e.debug_chain()));
    assert_eq!(
        vcues.len(),
        cues.len(),
        "the two subtitle formats must describe the same cues"
    );
    println!(
        "srt: {} cues, vtt: {} cues, both parse",
        cues.len(),
        vcues.len()
    );
}

#[test]
fn exporting_twice_with_the_same_options_produces_identical_bytes() {
    // An export that embedded a timestamp or a random id would break every diff,
    // every content-addressed cache above it, and every "did this change?" question
    // a user asks.
    let dir = tempfile::TempDir::new().expect("tempdir");
    let t = rich_transcript();
    let options = ExportOptions::default().with_source_name("a.wav");
    let overwrite = ExportOptions {
        overwrite: true,
        ..options.clone()
    };
    for format in ExportFormat::ALL {
        let target = ExportTarget::Directory(dir.path().to_path_buf());
        let first = export_to_file(&t, format, &target, &options).expect("first");
        let first_bytes = std::fs::read(&first.path).expect("read first");
        let second = export_to_file(&t, format, &target, &overwrite).expect("second");
        let second_bytes = std::fs::read(&second.path).expect("read second");
        assert_eq!(first_bytes, second_bytes, "{format} is not deterministic");
    }
}

#[test]
fn the_json_export_round_trips_to_an_equal_transcript() {
    // Not "a similar transcript": an equal one. The JSON export *is*
    // `vaani_core::Transcript`'s own serialisation, so any drift between them is a
    // bug in the domain type rather than in the exporter.
    let t = rich_transcript();
    let text = vaani_export::render(&t, ExportFormat::Json, &ExportOptions::default())
        .expect("render json");
    let back = vaani_export::read_json_export(&text).expect("read json");
    assert_eq!(
        back,
        t,
        "the JSON round trip lost something: {}",
        vaani_storage::fixtures::diff(&t, &back).unwrap_or_default()
    );
}

#[test]
fn the_export_conformance_checks_all_pass_on_a_rich_transcript() {
    // Determinism, the five extensions, the JSON round trip and the two
    // subtitle parsers, as reusable checks rather than as assertions written out
    // again here.
    let report = check_export_stage(&rich_transcript());
    assert!(report.ok(), "{}", report.render());
    let sample = check_export_stage(&sample_transcript());
    assert!(sample.ok(), "{}", sample.render());
    println!("{}", report.render());
}

#[test]
fn a_typed_file_target_whose_parent_is_missing_is_an_error_not_an_invitation() {
    // Silently creating the parent of a path a user typed turns a typo into a
    // stray directory tree, and the export lands somewhere they will not look.
    // A `Directory` target is the opposite case and is created — see
    // `a_new_directory_target_is_created`.
    let dir = tempfile::TempDir::new().expect("tempdir");
    let blocker = dir.path().join("a-file");
    std::fs::write(&blocker, b"not a directory").expect("blocker");

    // The parent is a file, so the write cannot succeed.
    let target = ExportTarget::File(blocker.join("out.srt"));
    let err = export_to_file(
        &sample_transcript(),
        ExportFormat::Srt,
        &target,
        &ExportOptions::default(),
    )
    .expect_err("must fail");
    assert_eq!(err.code(), ErrorCode::ExportFailed);
    let report = check_errors_are_actionable(&[("parent_is_a_file", &err)]);
    assert!(report.ok(), "{}", report.render());

    // A missing parent is refused and nothing is created.
    let missing = dir.path().join("nope").join("out.txt");
    let err = export_to_file(
        &sample_transcript(),
        ExportFormat::Text,
        &ExportTarget::File(missing.clone()),
        &ExportOptions::default(),
    )
    .expect_err("a missing parent is an error");
    assert_eq!(err.code(), ErrorCode::ExportFailed);
    assert!(
        !missing.parent().expect("parent").exists(),
        "nothing may be created for a refused File target"
    );
}

#[test]
fn a_new_directory_target_is_the_callers_job_not_the_exporters() {
    // **An asymmetry worth naming, and the reason this test exists.**
    // `vaani_app`'s `ExportStage::run` creates a `Directory` target that does not
    // exist yet, because a save dialog hands one back routinely (JOURNEY's
    // wave-2 item 3). The `vaani_export::export_to_file` function does *not*: it
    // resolves a path and writes it, and writing into a folder that is not there
    // fails.
    //
    // Both are correct, and the split is invisible until someone calls the
    // library directly — which is exactly what this test does. What must not
    // happen is the failure being silent or unclassified.
    let dir = tempfile::TempDir::new().expect("tempdir");
    let target = dir.path().join("brand").join("new");
    assert!(!target.exists(), "the fixture must start absent");
    let err = export_to_file(
        &sample_transcript(),
        ExportFormat::Text,
        &ExportTarget::Directory(target.clone()),
        &ExportOptions::default().with_source_name("standup"),
    )
    .expect_err("export_to_file does not create directories; ExportStage does");
    assert_eq!(err.code(), ErrorCode::ExportFailed);
    let report = check_errors_are_actionable(&[("directory_absent", &err)]);
    assert!(report.ok(), "{}", report.render());
    assert!(
        !target.exists(),
        "a refused write must not leave a directory behind"
    );
}

#[test]
fn an_existing_file_is_never_clobbered_unless_the_caller_says_so() {
    // "Nothing is overwritten silently" is one of the exporter's five stated
    // properties, and it is the one a user notices when it breaks: they had
    // edited that file.
    let dir = tempfile::TempDir::new().expect("tempdir");
    let t = sample_transcript();
    let options = ExportOptions::default().with_source_name("a.wav");
    let target = ExportTarget::Directory(dir.path().to_path_buf());
    export_to_file(&t, ExportFormat::Text, &target, &options).expect("first");
    let path = dir.path().join("a.txt");
    std::fs::write(&path, "mine").expect("the user's own file");
    let err = export_to_file(&t, ExportFormat::Text, &target, &options)
        .expect_err("the second export must refuse");
    assert_eq!(err.code(), ErrorCode::ExportFailed);
    assert_eq!(
        std::fs::read_to_string(&path).expect("read"),
        "mine",
        "the existing file was overwritten"
    );
    let report = check_errors_are_actionable(&[("already_there", &err)]);
    assert!(report.ok(), "{}", report.render());

    // And an explicit overwrite replaces it.
    export_to_file(
        &t,
        ExportFormat::Text,
        &target,
        &ExportOptions {
            overwrite: true,
            ..options
        },
    )
    .expect("overwrite");
    assert_ne!(std::fs::read_to_string(&path).expect("read"), "mine");
}

#[test]
fn a_crafted_source_name_cannot_escape_the_directory_the_user_chose() {
    // The name comes from the media file, which a user can call anything. This
    // is the one place where a bug is a security bug, so it is checked here as
    // well as inside the crate.
    let dir = tempfile::TempDir::new().expect("tempdir");
    for hostile in [
        "../../evil.txt",
        "..\\..\\evil.txt",
        "/absolute/evil.txt",
        "sub/../../evil.txt",
        "C:\\windows\\evil.txt",
    ] {
        let written = export_to_file(
            &sample_transcript(),
            ExportFormat::Text,
            &ExportTarget::Directory(dir.path().to_path_buf()),
            &ExportOptions {
                // Two of these names sanitise to the same file, so the default
                // "do not clobber" policy would refuse the second for an
                // unrelated reason. Overwriting is safe here precisely because
                // the containment check is what is under test.
                overwrite: true,
                ..ExportOptions::default().with_source_name(hostile)
            },
        )
        .unwrap_or_else(|e| panic!("{hostile}: {}", e.debug_chain()));
        assert_eq!(
            written.path.parent().expect("a parent"),
            dir.path(),
            "{hostile} escaped the target directory to {}",
            written.path.display()
        );

        // The outcome above is protected by two independent layers: the file
        // name is sanitised, and the writer separately refuses any path whose
        // parent is not the chosen directory. Asserting only the outcome means
        // a test cannot tell which layer did the work -- the mutation harness
        // found this by disabling the sanitiser, watching the suite stay green,
        // and correctly concluding that nothing here pins the *mechanism*.
        //
        // So pin the mechanism too. The name the writer chose must be a single
        // path component: no separator, no parent reference, and it must still
        // carry the hostile input's own characters, because a sanitiser that
        // silently produced a constant would satisfy every assertion above.
        let chosen = written
            .path
            .file_name()
            .expect("a file name")
            .to_string_lossy()
            .into_owned();
        assert!(
            !chosen.contains('/') && !chosen.contains('\\'),
            "{hostile} produced the multi-component name {chosen:?}"
        );
        assert!(
            !chosen.contains(".."),
            "{hostile} produced the name {chosen:?}, which still contains a parent reference"
        );
        assert!(
            !chosen.is_empty() && chosen != "." && chosen != "..",
            "{hostile} produced the empty or degenerate name {chosen:?}"
        );
        assert!(
            chosen.contains("evil"),
            "{hostile} produced {chosen:?}, which discarded the source name entirely rather \
             than sanitising it"
        );
    }
    // And nothing landed outside.
    let parent = dir.path().parent().expect("a parent");
    assert!(
        !parent.join("evil.txt").exists(),
        "a crafted name wrote outside the target directory"
    );
}

#[test]
fn a_transcript_with_no_segments_still_exports_to_something_readable() {
    // An empty transcript is a complete state, not a failure: a silent recording
    // must still produce a file a user can attach to an email.
    //
    // The subtitle formats are the exception and are allowed to be *empty*,
    // because an SRT with no cues has no correct content — a placeholder cue at
    // 00:00:00,000 would be a fabrication, and a player would show it.
    let empty = Transcript::new(EngineStamp::asr("qa-export", "1.0"));
    let dir = tempfile::TempDir::new().expect("tempdir");
    let options = ExportOptions::default().with_source_name("silence.wav");
    for format in ExportFormat::ALL {
        let written = export_to_file(
            &empty,
            format,
            &ExportTarget::Directory(dir.path().to_path_buf()),
            &options,
        )
        .unwrap_or_else(|e| panic!("{format}: {}", e.debug_chain()));
        let text = std::fs::read_to_string(&written.path).expect("read");
        if format.is_cue_based() {
            // No cues, so no cue content. A placeholder at 00:00:00,000 would be
            // a fabrication and a player would show it. WebVTT still carries
            // its mandatory `WEBVTT` signature, because a file without one is
            // not a VTT file at all.
            assert_eq!(written.cues, 0, "{format} reported {} cues", written.cues);
            if format == ExportFormat::Vtt {
                assert!(
                    text.starts_with("WEBVTT") && text.trim() == "WEBVTT",
                    "a VTT with no cues must be exactly the signature: {text:?}"
                );
            } else {
                assert!(
                    text.trim().is_empty(),
                    "{format} has no cues, so it must be empty rather than containing a \
                     placeholder: {text:?}"
                );
            }
        } else if format == ExportFormat::Text {
            // Plain text has nowhere to put metadata, so a transcript with
            // nothing in it is an empty file. That is the honest answer: a
            // banner would be content the user did not say.
            assert!(
                text.trim().is_empty(),
                "{format} invented content for an empty transcript: {text:?}"
            );
        } else {
            // Markdown and JSON carry a header — a title, a metadata block, an
            // envelope — which is *metadata*, not content, and is what makes the
            // file useful to a support conversation. What must never appear is a
            // word the user did not say.
            assert_eq!(
                written.bytes as usize,
                text.len(),
                "{format} reported {} bytes but wrote {}",
                written.bytes,
                text.len()
            );
            assert!(
                !text.trim().is_empty(),
                "{format} carries metadata, so even an empty transcript produces a file a user \
                 can attach and a support conversation can start from"
            );
            if format == ExportFormat::Json {
                let back = vaani_export::read_json_export(&text)
                    .unwrap_or_else(|e| panic!("{format}: {}", e.debug_chain()));
                assert!(back.segments.is_empty(), "{format} invented a segment");
            } else {
                let body: Vec<&str> = text
                    .lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with('-'))
                    .collect();
                assert!(
                    body.is_empty(),
                    "{format} produced body content for an empty transcript: {body:?}"
                );
            }
        }
    }
    // And the JSON round trip still works on an empty transcript, which is the
    // case a "skip the file if it is empty" shortcut would silently lose.
    let json_text = std::fs::read_to_string(dir.path().join("silence.json")).expect("read json");
    let json =
        vaani_export::read_json_export(&json_text).expect("an empty transcript must round-trip");
    assert!(json.segments.is_empty());
}

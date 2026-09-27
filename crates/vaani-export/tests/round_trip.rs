//! Round trips: export, read back, and get the same bytes out again.
//!
//! # What "round trip" means per format
//!
//! A round trip is only meaningful if the format carries the information being
//! recovered, so each is checked against what it actually holds:
//!
//! * **JSON** carries the whole transcript, so the round trip is exact: the
//!   `Transcript` that comes back must **equal** the one that went in. This is
//!   the single most important property in the crate, because a JSON export is
//!   what other tools consume.
//! * **SRT and VTT** carry cues — times and text — and not word-level timings or
//!   speaker attribution structure. So the round trip is checked at the level the
//!   format has: parse the file, re-serialise from the parsed cues, and require
//!   **byte-identical** output. That proves the written bytes are fully
//!   recoverable from the format's own structure, with no hidden state — which is
//!   the property that catches a serialiser that drops or reorders a field.
//! * **Text and Markdown** are one-way human formats, so what is checked is
//!   determinism (the same input gives the same bytes) and recoverability of the
//!   words, since a user will copy them out of the file.
//!
//! Idempotence catches more than a golden file: a golden file breaks when a
//! format is intentionally changed, and passes when a change is wrong.

use vaani_core::{EngineStamp, Segment, Speaker, Time, Transcript, Word};
use vaani_export::{
    ExportFormat, ExportOptions, ExportTarget, cues, export_to_file, render, verify,
};
use vaani_storage::fixtures;
use vaani_storage::testing::TempDir;

fn options() -> ExportOptions {
    ExportOptions::default()
        .with_source_name("standup.m4a")
        .with_timestamp_ms(1_700_000_000_000)
}

/// Re-serialise a parsed subtitle file and require the same bytes.
fn subtitle_round_trip(format: ExportFormat, text: &str) {
    match format {
        ExportFormat::Srt => {
            if text.is_empty() {
                return; // an empty transcript has no cues to recover
            }
            let parsed = verify::validate_srt(text, 2).expect("the file must be valid");
            let rebuilt: Vec<cues::Cue> = parsed
                .iter()
                .map(|c| cues::Cue {
                    start: c.start,
                    end: c.end,
                    lines: c.lines.clone(),
                    segment_id: String::new(),
                })
                .collect();
            assert_eq!(
                cues::render_srt(&rebuilt, true),
                text,
                "an SRT file does not survive a re-serialisation of its own cues"
            );
            // Every cue's text and times are recoverable, which is what makes the
            // file usable by another tool.
            assert!(!parsed.is_empty());
        }
        ExportFormat::Vtt => {
            if text == "WEBVTT\n\n" {
                return;
            }
            let parsed = verify::validate_vtt(text, 2).expect("the file must be valid");
            let rebuilt: Vec<cues::Cue> = parsed
                .iter()
                .map(|c| cues::Cue {
                    start: c.start,
                    end: c.end,
                    // The reader unescapes, so re-escape on the way out: the two
                    // must cancel, which is what proves the escape is reversible.
                    lines: c.lines.iter().map(|l| cues::escape_vtt_text(l)).collect(),
                    segment_id: String::new(),
                })
                .collect();
            assert_eq!(
                cues::render_vtt(&rebuilt),
                text,
                "a VTT file does not survive a re-serialisation of its own cues"
            );
        }
        _ => unreachable!("not a subtitle format"),
    }
}

/// Everything that can be recovered from a format, recovered.
fn check(format: ExportFormat, t: &Transcript, o: &ExportOptions) {
    let first = render(t, format, o).unwrap();
    // Determinism: the renderer reads no clock and no environment.
    assert_eq!(
        first,
        render(t, format, o).unwrap(),
        "{format} is not deterministic"
    );
    match format {
        ExportFormat::Srt | ExportFormat::Vtt => subtitle_round_trip(format, &first),
        ExportFormat::Json => {
            let back = vaani_export::read_json_export(&first).unwrap();
            assert!(
                fixtures::diff(t, &back).is_none(),
                "the transcript did not survive the JSON round trip"
            );
            // ...and re-exporting what came back gives the same bytes, which is
            // the idempotence claim in its strongest form.
            assert_eq!(render(&back, format, o).unwrap(), first);
        }
        ExportFormat::Text | ExportFormat::Markdown => {
            // Every segment's words are all still there, verbatim, in order. The
            // check is "the text is present", not "the text is the whole file",
            // because both formats add structure around it: timestamps, speaker
            // backslash escapes and hard wraps. Both sides are therefore
            // normalised the same way before comparing: escapes removed, all
            // whitespace collapsed to single spaces. A hard wrap inside a
            // sentence is a layout decision, not a change of text, and a
            // backslash is an escape marker rather than part of the words.
            let haystack = normalise(&first.replace('\\', ""));
            for seg in &t.segments {
                let text = normalise(&seg.text().replace('\\', ""));
                if text.is_empty() {
                    continue;
                }
                assert!(
                    haystack.contains(&text),
                    "{format} lost a segment: {text:?}"
                );
            }
        }
    }
}

/// A spread of transcript shapes, including the ones that break exporters.
fn cases() -> Vec<Transcript> {
    let mut out: Vec<Transcript> = vec![
        fixtures::empty(),
        fixtures::single_word(),
        fixtures::one_long_segment(200),
    ];
    for seed in 0..20u64 {
        out.push(fixtures::transcript(seed, 15));
    }
    // Speakers, some renamed and some not, so labels are exercised.
    let mut diarized = fixtures::transcript(99, 10);
    diarized.speakers = vec![
        Speaker {
            index: 0,
            name: Some("Krishna Anand".into()),
            share: Some(0.6),
        },
        Speaker {
            index: 1,
            name: None,
            share: Some(0.4),
        },
    ];
    out.push(diarized);
    // A transcript with no duration, which is what a live source gives.
    let mut no_duration = fixtures::transcript(7, 5);
    no_duration.duration = None;
    out.push(no_duration);
    // A transcript with nothing but silence.
    let mut silent = fixtures::transcript(3, 8);
    for s in &mut silent.segments {
        s.words.clear();
    }
    out.push(silent);
    out
}

#[test]
fn every_format_round_trips_for_every_awkward_shape() {
    for t in cases() {
        for f in ExportFormat::ALL {
            check(f, &t, &options());
        }
    }
}

#[test]
fn every_format_round_trips_for_a_hundred_generated_transcripts() {
    for seed in 0..100u64 {
        let t = fixtures::transcript(seed.wrapping_mul(0x1234_5678), 20);
        for f in ExportFormat::ALL {
            check(f, &t, &options());
        }
    }
}

#[test]
fn json_without_a_timestamp_is_reproducible_and_with_one_is_not() {
    // The reason `timestamp_ms` is an option rather than a clock read: an export
    // that embeds the current time can never be compared with a previous one.
    let t = fixtures::transcript(5, 8);
    let a = render(&t, ExportFormat::Json, &ExportOptions::default()).unwrap();
    let b = render(&t, ExportFormat::Json, &ExportOptions::default()).unwrap();
    assert_eq!(a, b);
    assert!(!a.contains("exported_at"));

    let first = options().with_timestamp_ms(1_000);
    let second = options().with_timestamp_ms(2_000);
    assert_ne!(
        render(&t, ExportFormat::Json, &first).unwrap(),
        render(&t, ExportFormat::Json, &second).unwrap()
    );
}

#[test]
fn a_transcript_survives_a_full_store_export_read_back_cycle() {
    // The end-to-end property the product actually promises: what is in the
    // database is what the user gets in the file, and reading the file gives back
    // the same transcript that went in.
    let dir = TempDir::new("cycle");
    let store = vaani_storage::Store::open(dir.join("v.db")).unwrap();
    let t = fixtures::transcript(0xBEEF, 60);
    let job = vaani_storage::JobRecord::new("j", dir.join("standup.m4a"));
    store.upsert_job(&job).unwrap();
    store.replace_transcript("j", &t).unwrap();

    let loaded = store.load_transcript("j").unwrap();
    let o = options();
    for f in ExportFormat::ALL {
        let target = ExportTarget::Directory(dir.path().to_path_buf());
        let written = export_to_file(&loaded, f, &target, &o).unwrap();
        let contents = std::fs::read_to_string(&written.path).unwrap();
        assert_eq!(
            contents,
            render(&loaded, f, &o).unwrap(),
            "{f}: the file on disk differs from the renderer"
        );
        if f == ExportFormat::Json {
            let back = vaani_export::read_json_export(&contents).unwrap();
            assert!(
                fixtures::diff(&t, &back).is_none(),
                "the transcript changed on the way through the database and a file"
            );
        }
    }
}

/// Collapse every run of whitespace to a single space, so a hard wrap inside a
/// sentence is not mistaken for a change of text.
fn normalise(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[test]
fn a_hand_edited_srt_still_parses_and_re_serialises() {
    // The property is only useful if it survives contact with files this crate did
    // not write, so: take a valid file, change a time, and check that it is still
    // valid and that re-serialising it is stable.
    let t = fixtures::single_word();
    let srt = render(&t, ExportFormat::Srt, &ExportOptions::default()).unwrap();
    let edited = srt.replace("00:00:00,500", "00:00:00,750");
    assert_ne!(
        edited, srt,
        "the fixture does not contain the expected timing"
    );
    let cues = verify::validate_srt(&edited, 2).unwrap();
    assert_eq!(cues[0].end, Time(750));
    let rebuilt: Vec<cues::Cue> = cues
        .iter()
        .map(|c| cues::Cue {
            start: c.start,
            end: c.end,
            lines: c.lines.clone(),
            segment_id: String::new(),
        })
        .collect();
    assert_eq!(cues::render_srt(&rebuilt, true), edited);
}

#[test]
fn a_hand_edited_file_that_breaks_a_rule_is_still_refused() {
    // The mirror image: an edit that makes the file invalid must be reported,
    // not accepted. Moving a cue's start past its end is exactly what a careless
    // hand edit does.
    let t = fixtures::single_word();
    let srt = render(&t, ExportFormat::Srt, &ExportOptions::default()).unwrap();
    let broken = srt.replace("00:00:00,000", "00:00:00,900");
    let e = verify::validate_srt(&broken, 2).unwrap_err();
    assert!(
        e.debug_chain().contains("positive_duration"),
        "{}",
        e.debug_chain()
    );
}

#[test]
fn a_cue_times_a_millisecond_apart_do_not_lose_either() {
    // 1 ms cues are the pathological case for a serialiser that rounds: two cues
    // 1 ms apart must stay 1 ms apart, not collapse into one or overlap.
    let mut t = Transcript::new(EngineStamp::asr("m", "1"));
    for i in 0..4 {
        let start = i;
        let mut seg = Segment::new(Time(start), Time(start + 1));
        seg.id = format!("s{i}");
        seg.words = vec![Word::new("x", Time(start), Time(start + 1))];
        t.segments.push(seg);
    }
    let srt = render(&t, ExportFormat::Srt, &ExportOptions::default()).unwrap();
    let parsed = verify::validate_srt(&srt, 2).unwrap();
    assert_eq!(parsed.len(), 4, "{srt}");
    for (i, c) in parsed.iter().enumerate() {
        assert_eq!(c.start.as_millis(), i as i64, "cue {i} moved: {srt}");
        assert!(c.end > c.start, "cue {i} has no duration");
    }
}

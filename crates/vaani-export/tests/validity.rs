//! Is the output actually valid, for every format, on the transcripts that break
//! exporters?
//!
//! # How validity was checked
//!
//! There is **no offline SubRip or WebVTT parser available to this crate**, and
//! adding one would mean a new third-party dependency, which is the architect's
//! decision and not this agent's. So SRT and VTT are validated against
//! hand-written parsers in [`vaani_export::verify`] that implement the published
//! rules, and the tests below say which rule each assertion is checking.
//!
//! What that proves: the bytes this crate writes parse as SubRip and as WebVTT
//! according to those rules, including the details people get wrong — the comma
//! versus the dot, sequential indices, blank-line termination, at most two lines
//! per cue, no overlaps, no zero-length cues, no mid-word breaks.
//!
//! What it does **not** prove: that every player in the world renders the file
//! identically. That needs a real player, and it is in the report as an open item
//! rather than dressed up as a pass.
//!
//! The awkward inputs are the point of this file. An exporter that has only ever
//! seen a tidy three-word segment has not been tested.

use vaani_core::{EngineStamp, Segment, Speaker, Time, Transcript, Word};
use vaani_export::{ExportFormat, ExportOptions, cues, render, verify};
use vaani_storage::fixtures;

/// A transcript with the shapes that break exporters.
fn awkward() -> Transcript {
    let mut t = fixtures::transcript(0xA11CE, 40);
    t.duration = Some(Time(783_400));
    t
}

fn one_segment(words: &[&str], start: i64, end: i64) -> Transcript {
    let mut t = Transcript::new(EngineStamp::asr("m", "1"));
    let span = ((end - start) / (words.len() as i64 + 1)).max(1);
    let mut seg = Segment::new(Time(start), Time(end));
    seg.id = "s0".into();
    seg.words = words
        .iter()
        .enumerate()
        .map(|(i, w)| {
            Word::new(
                *w,
                Time(start + span * i as i64),
                Time(start + span * (i as i64 + 1)),
            )
        })
        .collect();
    t.segments.push(seg);
    t
}

fn render_all(t: &Transcript, o: &ExportOptions) -> Vec<(ExportFormat, String)> {
    ExportFormat::ALL
        .iter()
        .map(|f| (*f, render(t, *f, o).unwrap()))
        .collect()
}

#[test]
fn an_empty_transcript_produces_valid_files_in_every_format() {
    let t = fixtures::empty();
    let o = ExportOptions::default();
    for (f, text) in render_all(&t, &o) {
        match f {
            // An empty transcript has no cues, and an SRT file with no cues is
            // not a valid SubRip file — so the exporter writes an empty file
            // rather than a malformed one. Asserted here so the choice is
            // deliberate rather than accidental.
            ExportFormat::Srt => assert_eq!(text, "", "{f} should be empty"),
            ExportFormat::Vtt => assert_eq!(text, "WEBVTT\n\n", "{f} keeps its header"),
            ExportFormat::Text => assert_eq!(text, ""),
            ExportFormat::Markdown => assert!(text.starts_with("# Transcript")),
            ExportFormat::Json => assert!(text.contains("\"segments\": []")),
        }
    }
}

#[test]
fn a_single_word_transcript_is_valid_everywhere() {
    let t = fixtures::single_word();
    let o = ExportOptions::default();
    for (f, text) in render_all(&t, &o) {
        match f {
            ExportFormat::Srt => {
                let cues = verify::validate_srt(&text, 2).unwrap();
                assert_eq!(cues.len(), 1);
                assert_eq!(cues[0].text(), "हाँ");
            }
            ExportFormat::Vtt => {
                let cues = verify::validate_vtt(&text, 2).unwrap();
                assert_eq!(cues.len(), 1);
            }
            ExportFormat::Text => assert_eq!(text.trim(), "[00:00] हाँ"),
            _ => {}
        }
    }
}

#[test]
fn a_segment_far_too_long_for_one_cue_is_split_not_truncated() {
    // 120 words at 42 columns is roughly 14 lines, so seven cues of two lines.
    let words: Vec<String> = (0..120).map(|i| format!("word{i}")).collect();
    let refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
    let t = one_segment(&refs, 0, 120_000);
    let o = ExportOptions::default();
    let srt = render(&t, ExportFormat::Srt, &o).unwrap();
    let cues = verify::validate_srt(&srt, 2).unwrap();
    assert!(cues.len() >= 6, "expected several cues, got {}", cues.len());

    // Nothing was lost: every word appears exactly once, in order.
    let spoken: Vec<String> = cues
        .iter()
        .flat_map(|c| {
            let text = c.text();
            text.split_whitespace()
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(spoken, words, "words were lost or reordered");

    // Every cue is within the rules, which `validate_srt` has already checked;
    // assert the two that matter most explicitly, because they are the two
    // people get wrong.
    for (i, c) in cues.iter().enumerate() {
        assert!(c.lines.len() <= 2, "cue {i} has {} lines", c.lines.len());
        assert!(c.end > c.start, "cue {i} has no duration");
        for line in &c.lines {
            assert!(line.chars().count() <= 42, "{line:?} is over 42 characters");
        }
        if i > 0 {
            assert!(
                c.start >= cues[i - 1].end,
                "cue {i} overlaps the previous one"
            );
        }
    }
    // The split points are word times, not a proportional guess: consecutive
    // cues are contiguous, because the words they share a boundary with are.
    for i in 1..cues.len() {
        assert_eq!(cues[i - 1].end, cues[i].start, "gap or overlap at cue {i}");
    }
}

#[test]
fn a_segment_longer_than_two_lines_with_no_word_times_is_still_valid() {
    // No word timings: the exporter falls back to dividing the span by character
    // count. The result is approximate, and the test asserts that it is
    // *valid* rather than that it is *right*, because "right" is not available
    // without timings.
    let mut t = Transcript::new(EngineStamp::asr("m", "1"));
    let mut seg = Segment::new(Time(0), Time(10_000));
    seg.id = "s0".into();
    seg.words = (0..30)
        .map(|i| Word::new(format!("w{i}"), Time(0), Time(0)))
        .collect();
    t.segments.push(seg);
    let srt = render(&t, ExportFormat::Srt, &ExportOptions::default()).unwrap();
    let cues = verify::validate_srt(&srt, 2).unwrap();
    assert!(cues.len() > 1);
    assert_eq!(cues[0].start, Time(0));
    assert_eq!(cues[cues.len() - 1].end, Time(10_000));
}

#[test]
fn back_to_back_segments_with_a_one_millisecond_gap_stay_valid() {
    let mut t = Transcript::new(EngineStamp::asr("m", "1"));
    for i in 0..5 {
        let mut seg = Segment::new(Time(i * 1_001), Time(i * 1_001 + 1_000));
        seg.id = format!("s{i}");
        seg.words = vec![Word::new("word", Time(i * 1_001), Time(i * 1_001 + 1_000))];
        t.segments.push(seg);
    }
    let srt = render(&t, ExportFormat::Srt, &ExportOptions::default()).unwrap();
    let cues = verify::validate_srt(&srt, 2).unwrap();
    assert_eq!(cues.len(), 5);
    for i in 1..cues.len() {
        assert!(cues[i].start >= cues[i - 1].end, "overlap at {i}");
        // The 1 ms gaps survive, because the cue ends are the segments' ends.
        assert_eq!(cues[i].start.as_millis() - cues[i - 1].end.as_millis(), 1);
    }
}

#[test]
fn an_all_silence_transcript_produces_no_cues_and_no_crash() {
    // Every segment empty: a transcript of a recording with no speech in it.
    let mut t = fixtures::transcript(1, 10);
    for s in &mut t.segments {
        s.words.clear();
    }
    let o = ExportOptions::default();
    let srt = render(&t, ExportFormat::Srt, &o).unwrap();
    assert_eq!(srt, "", "no speech means no cues");
    let vtt = render(&t, ExportFormat::Vtt, &o).unwrap();
    assert_eq!(vtt, "WEBVTT\n\n", "the header survives");
    let txt = render(&t, ExportFormat::Text, &o).unwrap();
    assert_eq!(txt, "");
    // JSON is still a complete, parseable document.
    let json = render(&t, ExportFormat::Json, &o).unwrap();
    assert_eq!(
        vaani_export::read_json_export(&json)
            .unwrap()
            .segments
            .len(),
        10
    );
}

#[test]
fn non_ascii_and_mixed_script_text_survives_every_format() {
    let t = one_segment(
        &[
            "नमस्ते",
            "مرحبا",
            "你好",
            "こんにちは",
            "안녕하세요",
            "café",
            "naïve",
            "ß",
            "½",
            "🎧",
            "👩‍💻",
        ],
        0,
        11_000,
    );
    let o = ExportOptions::default();
    for (f, text) in render_all(&t, &o) {
        match f {
            ExportFormat::Srt => {
                let cues = verify::validate_srt(&text, 2).unwrap();
                assert!(cues[0].text().contains("नमस्ते"), "{text}");
                assert!(cues[0].text().contains("👩‍💻"), "{text}");
            }
            ExportFormat::Vtt => {
                let cues = verify::validate_vtt(&text, 2).unwrap();
                assert!(cues[0].text().contains("नमस्ते"));
            }
            ExportFormat::Text => assert!(text.contains("مرحبا")),
            ExportFormat::Json => {
                let back = vaani_export::read_json_export(&text).unwrap();
                assert_eq!(back, t, "{f} lost the text");
            }
            ExportFormat::Markdown => assert!(text.contains("नमस्ते")),
        }
    }
}

#[test]
fn characters_that_are_special_in_each_format_are_handled() {
    // SRT: a blank line would end the cue early, and "-->" is legal but the VTT
    // parser chokes on it. Markdown: `#`, `*`, `_`, backtick, brackets. JSON:
    // quotes and backslashes.
    let t = one_segment(
        &[
            "a --> b",
            "#hashtag",
            "*star*",
            "_under_",
            "`tick`",
            "[link](x)",
            "quote\"inside",
            "back\\slash",
            "line sep",
        ],
        0,
        9_000,
    );
    let o = ExportOptions::default().with_source_name("meeting.m4a");

    let srt = render(&t, ExportFormat::Srt, &o).unwrap();
    let cues = verify::validate_srt(&srt, 2).unwrap();
    let text = cues.iter().map(|c| c.text()).collect::<Vec<_>>().join(" ");
    for needed in ["-->", "#hashtag", "*star*", "quote\"inside", "back\\slash"] {
        assert!(text.contains(needed), "SRT lost {needed:?}: {srt}");
    }
    // A cue's text can never contain a blank line, because a blank line ends it.
    for c in &cues {
        assert!(!c.text().contains('\n'), "a cue contains a newline");
    }

    let vtt = render(&t, ExportFormat::Vtt, &o).unwrap();
    // The arrow is escaped everywhere it appears as *text*, and only there: the
    // timing line's own separator is of course a literal arrow, so the check is
    // on the payload lines rather than on the whole file.
    let mut in_timing = false;
    for line in vtt.lines() {
        if line.contains("-->") && !in_timing {
            in_timing = true;
            continue;
        }
        if line.trim().is_empty() {
            in_timing = false;
            continue;
        }
        assert!(
            !line.contains("-->"),
            "the cue payload still contains an arrow: {line:?} in {vtt}"
        );
    }
    let cues = verify::validate_vtt(&vtt, 2).unwrap();
    assert!(cues[0].text().contains("-->"), "escaping was not reversed");

    let md = render(&t, ExportFormat::Markdown, &o).unwrap();
    // No *body* line may begin with a character that would make it a block. The
    // document's own heading and metadata lines start with `#` and `-` on
    // purpose, so the check is on the paragraph lines.
    let mut in_paragraph = false;
    for line in md.lines() {
        if line.starts_with("**[") {
            in_paragraph = true;
        } else if in_paragraph && line.trim().is_empty() {
            in_paragraph = false;
        } else if in_paragraph {
            let first = line.trim_start().chars().next().unwrap_or(' ');
            assert!(
                !matches!(first, '#' | '>' | '=' | '-' | '+' | '|'),
                "a body line begins with a block marker: {line:?}"
            );
        }
    }
    // `*` and `_` mean something anywhere in a line, so they are escaped
    // wherever they appear. `#` only means something at the start of a line, so
    // mid-sentence it is left alone and the text reads as it was said.
    assert!(md.contains("\\*star\\*"), "{md}");
    assert!(md.contains("\\_under\\_"), "{md}");
    assert!(md.contains("\\`tick\\`"), "{md}");
    assert!(md.contains("#hashtag"), "{md}");
    assert!(md.contains("\\[link\\]"), "{md}");

    let json = render(&t, ExportFormat::Json, &o).unwrap();
    let back = vaani_export::read_json_export(&json).unwrap();
    assert_eq!(back, t, "JSON lost or mangled a special character");
    // The file is valid JSON, not merely parseable by our own reader.
    let _: serde_json::Value = serde_json::from_str(&json).unwrap();
}

#[test]
fn a_word_longer_than_the_line_is_never_broken() {
    let long = "x".repeat(200);
    let t = one_segment(&["before", long.as_str(), "after"], 0, 3_000);
    let srt = render(&t, ExportFormat::Srt, &ExportOptions::default()).unwrap();
    let cues = verify::validate_srt(&srt, 2).unwrap();
    let joined: Vec<String> = cues.iter().flat_map(|c| c.lines.clone()).collect();
    assert!(
        joined.iter().any(|l| l == &long),
        "the long word was broken: {:?}",
        joined.iter().map(|l| l.chars().count()).collect::<Vec<_>>()
    );
    // Every line is either the long word or short; nothing overflows except that
    // one line, which is the documented exception.
    for line in &joined {
        assert!(
            line.chars().count() <= 200,
            "a line exceeded the longest word"
        );
    }
}

#[test]
fn overlapping_segments_are_serialised_rather_than_emitted_as_overlaps() {
    let mut t = Transcript::new(EngineStamp::asr("m", "1"));
    for (i, (start, end)) in [(0i64, 5_000i64), (2_000, 7_000), (6_000, 6_500)]
        .into_iter()
        .enumerate()
    {
        let mut seg = Segment::new(Time(start), Time(end));
        seg.id = format!("s{i}");
        seg.words = vec![Word::new("text", Time(start), Time(end))];
        t.segments.push(seg);
    }
    let srt = render(&t, ExportFormat::Srt, &ExportOptions::default()).unwrap();
    let cues = verify::validate_srt(&srt, 2).unwrap();
    assert_eq!(cues.len(), 3, "no text is dropped");
    for i in 1..cues.len() {
        assert!(
            cues[i].start >= cues[i - 1].end,
            "cue {i} starts at {} but the previous ends at {}",
            cues[i].start.as_millis(),
            cues[i - 1].end.as_millis()
        );
    }
    // The cost of serialising is stated rather than hidden: a caption can appear
    // after the speech it holds. That is the deliberate trade.
    assert!(cues[1].start.as_millis() >= 5_000);
}

#[test]
fn cue_numbers_are_sequential_and_can_be_turned_off() {
    let t = awkward();
    let o = ExportOptions::default();
    let numbered = render(&t, ExportFormat::Srt, &o).unwrap();
    let cues = verify::parse_srt(&numbered).unwrap();
    for (i, c) in cues.iter().enumerate() {
        assert_eq!(
            c.index,
            Some(i as u64 + 1),
            "cue {} has index {:?}",
            i + 1,
            c.index
        );
    }
    let bare = render(
        &t,
        ExportFormat::Srt,
        &ExportOptions {
            include_cue_numbers: false,
            ..ExportOptions::default()
        },
    )
    .unwrap();
    // Without indices the file is still parseable by a lenient reader, which is
    // what a player does.
    let cues = verify::parse_srt(&bare).unwrap();
    assert_eq!(cues.len(), numbered_cue_count(&numbered));
    assert!(cues.iter().all(|c| c.index.is_none()));
}

fn numbered_cue_count(srt: &str) -> usize {
    srt.lines()
        .filter(|l| l.trim().parse::<u64>().is_ok() && *l != "0")
        .count()
}

#[test]
fn all_three_speaker_states_are_handled() {
    // 1. A renamed speaker, 2. an unnamed cluster, 3. no diarization at all.
    let mut t = one_segment(&["hello", "there"], 0, 2_000);
    t.segments[0].speaker = Some(0);
    t.speakers = vec![
        Speaker {
            index: 0,
            name: Some("Krishna".into()),
            share: Some(1.0),
        },
        Speaker {
            index: 1,
            name: None,
            share: None,
        },
    ];
    t.segments.push({
        let mut s = Segment::new(Time(2_000), Time(4_000));
        s.id = "s1".into();
        s.speaker = Some(1);
        s.words = vec![Word::new("unnamed", Time(2_000), Time(4_000))];
        s
    });
    let o = ExportOptions::default();
    let srt = render(&t, ExportFormat::Srt, &o).unwrap();
    let cues = verify::validate_srt(&srt, 2).unwrap();
    assert_eq!(cues[0].text(), "Krishna: hello there");
    assert_eq!(cues[1].text(), "Speaker 2: unnamed");
    let md = render(&t, ExportFormat::Markdown, &o).unwrap();
    assert!(md.contains("## Krishna"), "{md}");
    assert!(md.contains("## Speaker 2"), "{md}");
    // And with diarization the turn is not labelled twice.
    assert!(!md.contains("**[00:00]** Krishna:"), "{md}");

    // 3. No diarization: no labels anywhere, and no invented speaker.
    let plain = one_segment(&["just", "words"], 0, 2_000);
    let srt = render(&plain, ExportFormat::Srt, &o).unwrap();
    let cues = verify::validate_srt(&srt, 2).unwrap();
    assert_eq!(cues[0].text(), "just words");
    let md = render(&plain, ExportFormat::Markdown, &o).unwrap();
    assert!(
        !md.contains("## "),
        "an undiarized transcript has no sections: {md}"
    );
    assert!(!md.contains("Speaker"), "{md}");
}

#[test]
fn a_hundred_random_transcripts_all_produce_valid_subtitles() {
    // The property behind the per-case tests above: for a wide spread of shapes,
    // the output parses and obeys the rules.
    for seed in 0..100u64 {
        let t = fixtures::transcript(seed.wrapping_mul(0x9E37_79B9), 25);
        let o = ExportOptions::default().with_source_name("a.wav");
        let srt = render(&t, ExportFormat::Srt, &o)
            .unwrap_or_else(|e| panic!("seed {seed}: SRT failed: {e:?}"));
        let vtt = render(&t, ExportFormat::Vtt, &o)
            .unwrap_or_else(|e| panic!("seed {seed}: VTT failed: {e:?}"));
        if !srt.is_empty() {
            verify::validate_srt(&srt, 2)
                .unwrap_or_else(|e| panic!("seed {seed}: invalid SRT: {e:?}"));
        }
        if vtt != "WEBVTT\n\n" {
            verify::validate_vtt(&vtt, 2)
                .unwrap_or_else(|e| panic!("seed {seed}: invalid VTT: {e:?}"));
        }
    }
}

#[test]
fn the_cue_builder_is_the_single_source_of_both_subtitle_formats() {
    // SRT and VTT are the same cues with a different serialiser, so the two
    // must agree cue for cue. If a format ever grows its own timing logic, this
    // fails.
    let t = awkward();
    let o = ExportOptions::default();
    let built = cues::build_cues(&t, &o.cue_options());
    let srt = verify::parse_srt(&render(&t, ExportFormat::Srt, &o).unwrap()).unwrap();
    let vtt = verify::parse_vtt(&render(&t, ExportFormat::Vtt, &o).unwrap()).unwrap();
    assert_eq!(srt.len(), built.len());
    assert_eq!(vtt.len(), built.len());
    for i in 0..built.len() {
        assert_eq!(srt[i].start, built[i].start, "cue {i}");
        assert_eq!(srt[i].end, built[i].end, "cue {i}");
        assert_eq!(vtt[i].start, built[i].start, "cue {i}");
        assert_eq!(vtt[i].end, built[i].end, "cue {i}");
    }
}

#[test]
fn timestamps_are_in_source_media_coordinates() {
    // The times a user sees in the viewer and the times in the subtitle file are
    // the same numbers: no chunk offsets, no rounding, no rescaling.
    let mut t = Transcript::new(EngineStamp::asr("m", "1"));
    let mut seg = Segment::new(Time(3_661_500), Time(3_662_000));
    seg.id = "s0".into();
    seg.words = vec![Word::new("late", Time(3_661_500), Time(3_662_000))];
    t.segments.push(seg);
    let srt = render(&t, ExportFormat::Srt, &ExportOptions::default()).unwrap();
    assert!(srt.contains("01:01:01,500 --> 01:01:02,000"), "{srt}");
    let vtt = render(&t, ExportFormat::Vtt, &ExportOptions::default()).unwrap();
    assert!(vtt.contains("01:01:01.500 --> 01:01:02.000"), "{vtt}");
    // The text format shows the viewer's own form of the same instant.
    let txt = render(&t, ExportFormat::Text, &ExportOptions::default()).unwrap();
    assert!(txt.contains("[01:01:01]"), "{txt}");
}

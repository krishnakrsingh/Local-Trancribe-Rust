//! Boundary conditions, one test per rule, each written to fail first.
//!
//! Every fixture here is **synthesised**: word material is a numbered token
//! like `w0007`, never anything resembling a real sentence. That is not
//! fastidiousness. `scratch/clip.wav` is a real private meeting, and a test
//! fixture containing its words — or any fragment of its word *sequence* —
//! would be the meeting leaking into the repository. A numbered token carries
//! no content, and it carries a second benefit: the expected output can be
//! written down by hand, because the ground truth is known exactly.

use vaani_core::pipeline::{DiarizationResult, SpeakerSpan};
use vaani_core::{EngineStamp, Segment, Time, Transcript, Word};
use vaani_segment::{GroupingPolicy, group, group_with_diarization, plan_groups};

fn t(ms: i64) -> Time {
    Time::from_millis(ms)
}

/// A transcript of one-word segments, which is exactly what the ASR adapter
/// emits. `words` is `(text, start, end)`.
fn one_word_per_segment(words: &[(&str, i64, i64)]) -> Transcript {
    let mut tr = Transcript::new(EngineStamp::asr("sravaani", "1.0"));
    for (i, (text, start, end)) in words.iter().enumerate() {
        let mut seg =
            Segment::new(t(*start), t(*end)).with_words(vec![Word::new(*text, t(*start), t(*end))]);
        seg.id = format!("s{i:04}");
        tr.segments.push(seg);
    }
    tr
}

/// A flat run of words with a fixed 20 ms gap and 300 ms per word, so a test
/// only has to state the gaps it cares about.
fn run(count: usize, first_start: i64) -> Transcript {
    let words: Vec<(String, i64, i64)> = (0..count)
        .map(|i| {
            let start = first_start + (i as i64) * 320;
            (format!("w{i:04}"), start, start + 300)
        })
        .collect();
    let borrowed: Vec<(&str, i64, i64)> =
        words.iter().map(|(a, b, c)| (a.as_str(), *b, *c)).collect();
    one_word_per_segment(&borrowed)
}

fn policy() -> GroupingPolicy {
    GroupingPolicy::default()
}

fn text_of(tr: &Transcript) -> Vec<String> {
    tr.segments.iter().map(|s| s.text()).collect()
}

fn report_words(tr: &Transcript) -> usize {
    tr.segments.iter().map(|s| s.words.len()).sum()
}

// ---------------------------------------------------------------- empty input

#[test]
fn an_empty_transcript_groups_to_an_empty_transcript() {
    let mut tr = Transcript::new(EngineStamp::asr("sravaani", "1.0"));
    let report = group(&mut tr, &policy()).unwrap();
    assert!(tr.segments.is_empty());
    assert_eq!(report.words_in, 0);
    assert_eq!(report.words_out, 0);
    assert_eq!(report.segments_in, 0);
    assert_eq!(report.segments_out, 0);
    assert_eq!(report.total_breaks(), 0);
    assert!(report.input_was_well_formed());
}

#[test]
fn a_transcript_of_only_wordless_segments_produces_no_segments() {
    // A segment with no words is a blank screen for its duration. It cannot
    // become an output segment, and it must not become an *empty* one either.
    let mut tr = Transcript::new(EngineStamp::asr("sravaani", "1.0"));
    tr.segments.push(Segment::new(t(0), t(1_000)));
    tr.segments.push(Segment::new(t(1_000), t(2_000)));
    let report = group(&mut tr, &policy()).unwrap();
    assert!(tr.segments.is_empty());
    assert_eq!(report.source_segments_without_words, 2);
    assert_eq!(report.words_in, 0);
}

// --------------------------------------------------------------- single word

#[test]
fn a_single_word_survives_as_a_single_segment() {
    let mut tr = one_word_per_segment(&[("w0000", 0, 300)]);
    let report = group(&mut tr, &policy()).unwrap();
    assert_eq!(tr.segments.len(), 1);
    assert_eq!(tr.segments[0].text(), "w0000");
    // The span is the word's own span, not shortened by the minimum-duration
    // floor, because 300 ms already exceeds it.
    assert_eq!(tr.segments[0].start, t(0));
    assert_eq!(tr.segments[0].end, t(300));
    assert_eq!(report.words_in, 1);
    assert_eq!(report.words_out, 1);
    // One word per source segment means the id survives verbatim.
    assert_eq!(tr.segments[0].id, "s0000");
}

#[test]
fn a_single_zero_length_word_still_gets_a_non_empty_segment() {
    // `push_chunk` clamps a degenerate word to end == start, so this really
    // happens. The segment must still be selectable and exportable.
    let mut tr = one_word_per_segment(&[("w0000", 500, 500)]);
    group(&mut tr, &policy()).unwrap();
    assert_eq!(tr.segments.len(), 1);
    assert!(
        tr.segments[0].end > tr.segments[0].start,
        "zero-length segment"
    );
    assert_eq!(
        tr.segments[0].words[0].end,
        t(500),
        "the word's time is untouched"
    );
}

// ------------------------------------------------------------- the threshold

#[test]
fn a_gap_exactly_at_the_threshold_breaks() {
    // The off-by-one case. `>=` is the documented rule, so 500 breaks and 499
    // does not. Getting this wrong is invisible in aggregate and produces one
    // segment too many or too few, forever.
    let mut tr = one_word_per_segment(&[("w0000", 0, 300), ("w0001", 800, 1_100)]);
    let report = group(&mut tr, &policy()).unwrap();
    assert_eq!(
        tr.segments.len(),
        2,
        "800 - 300 == 500, exactly the threshold"
    );
    assert_eq!(report.breaks_pause, 1);
    assert_eq!(report.breaks_hard_gap, 0);
}

#[test]
fn a_gap_one_millisecond_below_the_threshold_does_not_break() {
    let mut tr = one_word_per_segment(&[("w0000", 0, 300), ("w0001", 799, 1_099)]);
    let report = group(&mut tr, &policy()).unwrap();
    assert_eq!(tr.segments.len(), 1, "499 ms of silence is not a pause");
    assert_eq!(report.breaks_pause, 0);
    assert_eq!(report.total_breaks(), 0);
}

#[test]
fn a_gap_of_zero_does_not_break() {
    let mut tr = one_word_per_segment(&[("w0000", 0, 300), ("w0001", 300, 600)]);
    let report = group(&mut tr, &policy()).unwrap();
    assert_eq!(tr.segments.len(), 1);
    assert_eq!(text_of(&tr), vec!["w0000 w0001"]);
    assert_eq!(report.total_breaks(), 0);
}

#[test]
fn a_negative_gap_does_not_break_and_does_not_panic() {
    // Overlapping words: a chunk overlap can produce a word starting before the
    // previous one ended. `gap` is negative, which is below every threshold.
    let mut tr = one_word_per_segment(&[("w0000", 0, 500), ("w0001", 200, 700)]);
    let report = group(&mut tr, &policy()).unwrap();
    assert_eq!(tr.segments.len(), 1);
    assert_eq!(report.total_breaks(), 0);
    // Non-overlap is still enforced on the *segment* span.
    assert!(tr.segments[0].end >= tr.segments[0].start);
}

#[test]
fn a_gap_far_larger_than_any_cap_breaks_on_the_hard_ceiling() {
    // Ten seconds of silence, with every cap switched off, so only the hard
    // ceiling can fire. This is the rule that stops a segment's stated duration
    // from being mostly silence.
    let p = GroupingPolicy {
        max_words: 0,
        max_chars: 0,
        max_duration_ms: 0,
        ..GroupingPolicy::default()
    };
    let mut tr = one_word_per_segment(&[("w0000", 0, 300), ("w0001", 10_300, 10_600)]);
    let report = group(&mut tr, &p).unwrap();
    assert_eq!(tr.segments.len(), 2);
    assert_eq!(report.breaks_hard_gap, 1);
    assert_eq!(
        report.breaks_pause, 0,
        "the pause rule is subsumed by the ceiling"
    );
}

#[test]
fn a_gap_below_the_hard_ceiling_breaks_on_the_pause_rule_and_is_counted_as_such() {
    let mut tr = one_word_per_segment(&[("w0000", 0, 300), ("w0001", 1_000, 1_300)]);
    let report = group(&mut tr, &policy()).unwrap();
    assert_eq!(tr.segments.len(), 2);
    assert_eq!(report.breaks_pause, 1);
    assert_eq!(report.breaks_hard_gap, 0);
}

// ----------------------------------------------------------------- the caps

#[test]
fn the_word_cap_breaks_a_run_of_evenly_spaced_words() {
    // 20 ms gaps, so no pause rule can fire and the cap is the only break.
    let words: Vec<(String, i64, i64)> = (0..30)
        .map(|i| {
            let s = i as i64 * 120;
            (format!("w{i:04}"), s, s + 100)
        })
        .collect();
    let borrowed: Vec<(&str, i64, i64)> =
        words.iter().map(|(a, b, c)| (a.as_str(), *b, *c)).collect();
    let mut tr = one_word_per_segment(&borrowed);
    let p = GroupingPolicy {
        max_words: 10,
        min_segment_words: 0,
        ..GroupingPolicy::default()
    };
    let report = group(&mut tr, &p).unwrap();
    assert_eq!(report.breaks_word_cap, 2);
    assert_eq!(tr.segments.len(), 3);
    assert_eq!(tr.segments[0].words.len(), 10);
    assert_eq!(tr.segments[2].words.len(), 10);
    assert_eq!(report.words_in, 30);
    assert_eq!(report.words_out, 30);
}

#[test]
fn a_zero_cap_means_no_cap() {
    let p = GroupingPolicy {
        max_words: 0,
        max_chars: 0,
        max_duration_ms: 0,
        min_segment_words: 0,
        ..GroupingPolicy::default()
    };
    // One long unbroken run: 100 words, 20 ms apart, never pausing.
    let mut tr = run(100, 0);
    let report = group(&mut tr, &p).unwrap();
    assert_eq!(
        tr.segments.len(),
        1,
        "with every cap off nothing can break it"
    );
    assert_eq!(
        report.breaks_word_cap + report.breaks_char_cap + report.breaks_duration_cap,
        0
    );
}

#[test]
fn the_char_cap_breaks_before_exceeding_the_budget() {
    // Five-character tokens plus a space: a 17-word group is 103 characters, so
    // the 84-character cap must fire well before the word cap would.
    let words: Vec<(String, i64, i64)> = (0..40)
        .map(|i| {
            let s = i as i64 * 120;
            (format!("w{i:04}"), s, s + 100)
        })
        .collect();
    let borrowed: Vec<(&str, i64, i64)> =
        words.iter().map(|(a, b, c)| (a.as_str(), *b, *c)).collect();
    let mut tr = one_word_per_segment(&borrowed);
    // The runt guard off, so this measures the cap itself and nothing else.
    let p = GroupingPolicy {
        max_words: 0,
        max_chars: 84,
        min_segment_words: 0,
        ..GroupingPolicy::default()
    };
    let report = group(&mut tr, &p).unwrap();
    assert!(report.breaks_char_cap > 0, "{report:?}");
    assert_eq!(report.cap_breaks_deferred, 0);
    assert_eq!(report.segments_over_cap, 0);
    for seg in &tr.segments {
        assert!(
            seg.text().chars().count() <= 84,
            "{} chars in {:?}",
            seg.text().chars().count(),
            seg.id
        );
    }
}

#[test]
fn a_cap_that_would_be_deferred_can_exceed_the_budget_and_says_so() {
    // Seven 5-character words make 41 characters, one over the 40-character cap.
    // With a pause immediately after the seventh word, honouring the cap would
    // produce a six-word group and then a one-word group; the guard defers to the
    // pause instead, and the report says the group went over.
    let mut words: Vec<(String, i64, i64)> = Vec::new();
    let mut clock = 0i64;
    for _block in 0..3 {
        for i in 0..7 {
            let start = clock + i * 120;
            words.push((format!("w{i:04}"), start, start + 100));
        }
        clock += 7 * 120 + 900;
    }
    let borrowed: Vec<(&str, i64, i64)> =
        words.iter().map(|(a, b, c)| (a.as_str(), *b, *c)).collect();
    let mut tr = one_word_per_segment(&borrowed);
    let p = GroupingPolicy {
        max_words: 0,
        max_chars: 40,
        min_segment_words: 3,
        ..GroupingPolicy::default()
    };
    let report = group(&mut tr, &p).unwrap();
    // The first two blocks defer to the pause that follows them. The last block
    // has no pause after it, so there is nothing to defer to and the cap fires:
    // a four-group result, not three, and the runt guard does not paper over it.
    assert_eq!(tr.segments.len(), 4, "{report:?}");
    for i in 0..2 {
        assert_eq!(
            tr.segments[i].words.len(),
            7,
            "group {i} deferred the cap to the pause"
        );
        assert_eq!(tr.segments[i].text().chars().count(), 41);
    }
    assert_eq!(
        tr.segments[2].words.len(),
        6,
        "no boundary ahead, so the cap fired"
    );
    assert_eq!(tr.segments[3].words.len(), 1);
    assert_eq!(report.cap_breaks_deferred, 2, "{report:?}");
    assert_eq!(
        report.breaks_char_cap, 1,
        "one cap break was actually taken"
    );
    assert_eq!(report.breaks_pause, 2);
    assert_eq!(
        report.segments_over_cap, 2,
        "the cost of the guard is counted, not hidden"
    );
}

#[test]
fn a_single_word_longer_than_the_char_cap_is_never_split() {
    // Half a word on screen is unreadable, so a token that cannot fit still
    // forms a segment of its own.
    let long = "x".repeat(200);
    let mut tr = one_word_per_segment(&[("a", 0, 100), (&long, 120, 220)]);
    let report = group(&mut tr, &policy()).unwrap();
    assert_eq!(tr.segments.len(), 2);
    assert_eq!(tr.segments[1].words.len(), 1);
    assert_eq!(tr.segments[1].words[0].text.len(), 200);
    assert!(report.segments_over_cap >= 1);
}

#[test]
fn the_duration_cap_splits_a_long_sentence_with_no_pause() {
    // The "no punctuation, no pause, one sentence" case: 60 words 100 ms apart
    // is 6 s of unbroken speech, so only the duration cap can break it.
    let words: Vec<(String, i64, i64)> = (0..60)
        .map(|i| {
            let s = i as i64 * 100;
            (format!("w{i:04}"), s, s + 100)
        })
        .collect();
    let borrowed: Vec<(&str, i64, i64)> =
        words.iter().map(|(a, b, c)| (a.as_str(), *b, *c)).collect();
    let mut tr = one_word_per_segment(&borrowed);
    let p = GroupingPolicy {
        max_words: 0,
        max_chars: 0,
        max_duration_ms: 2_000,
        min_segment_words: 0,
        ..GroupingPolicy::default()
    };
    let report = group(&mut tr, &p).unwrap();
    assert!(report.breaks_duration_cap > 0, "{report:?}");
    for seg in &tr.segments {
        assert!(seg.duration().as_millis() <= 2_000, "{:?}", seg.duration());
    }
}

// -------------------------------------------------------------- the runt guard

#[test]
fn a_cap_one_word_before_a_pause_is_deferred_rather_than_making_a_runt() {
    // max_words = 4 fires at word 5, and the pause is at word 6. Honouring the
    // cap would produce a one-word segment; the guard defers to the pause.
    let mut tr = one_word_per_segment(&[
        ("w0000", 0, 300),
        ("w0001", 320, 620),
        ("w0002", 640, 940),
        ("w0003", 960, 1_260),
        ("w0004", 1_280, 1_580),
        ("w0005", 3_000, 3_300),
    ]);
    let p = GroupingPolicy {
        max_words: 4,
        min_segment_words: 3,
        ..GroupingPolicy::default()
    };
    let report = group(&mut tr, &p).unwrap();
    assert_eq!(tr.segments.len(), 2, "{:?}", text_of(&tr));
    assert_eq!(tr.segments[0].words.len(), 5, "the cap was deferred");
    assert_eq!(tr.segments[1].words.len(), 1);
    assert_eq!(report.cap_breaks_deferred, 1);
    assert_eq!(
        report.segments_over_cap, 1,
        "and the cost is counted, not hidden"
    );
}

#[test]
fn the_guard_does_not_defer_when_the_next_boundary_is_far_away() {
    // Same cap, but the next pause is 30 words later. There is no runt to avoid.
    let mut tr = one_word_per_segment(&[
        ("w0000", 0, 300),
        ("w0001", 320, 620),
        ("w0002", 640, 940),
        ("w0003", 960, 1_260),
        ("w0004", 1_280, 1_580),
        ("w0005", 1_600, 1_900),
        ("w0006", 1_920, 2_220),
    ]);
    let p = GroupingPolicy {
        max_words: 4,
        min_segment_words: 3,
        ..GroupingPolicy::default()
    };
    let report = group(&mut tr, &p).unwrap();
    assert_eq!(report.cap_breaks_deferred, 0);
    assert_eq!(tr.segments[0].words.len(), 4);
    assert_eq!(tr.segments[1].words.len(), 3);
}

#[test]
fn a_guard_with_no_natural_boundary_ahead_honours_the_cap() {
    // Nothing to defer to, so the cap fires. Without this the last group of a
    // transcript could grow without bound.
    let mut tr = one_word_per_segment(&[
        ("w0000", 0, 300),
        ("w0001", 320, 620),
        ("w0002", 640, 940),
        ("w0003", 960, 1_260),
        ("w0004", 1_280, 1_580),
    ]);
    let p = GroupingPolicy {
        max_words: 4,
        min_segment_words: 3,
        ..GroupingPolicy::default()
    };
    let report = group(&mut tr, &p).unwrap();
    assert_eq!(report.cap_breaks_deferred, 0);
    assert_eq!(tr.segments[0].words.len(), 4);
    assert_eq!(tr.segments[1].words.len(), 1);
}

#[test]
fn a_guard_larger_than_the_word_cap_is_rejected_rather_than_silently_ignored() {
    // `min_segment_words > max_words` means the word cap can never be honoured
    // without a runt appearing anyway, so the policy is incoherent. Refusing it
    // is better than shipping a policy whose word cap does nothing.
    let p = GroupingPolicy {
        max_words: 2,
        min_segment_words: 3,
        ..GroupingPolicy::default()
    };
    let mut tr = run(10, 0);
    assert!(group(&mut tr, &p).is_err());
}

// ------------------------------------------------------------- speaker turns

fn diarization(spans: &[(usize, i64, i64)], num: usize) -> DiarizationResult {
    DiarizationResult {
        spans: spans
            .iter()
            .map(|(s, a, b)| SpeakerSpan {
                speaker: *s,
                start: t(*a),
                end: t(*b),
                score: Some(1.0),
            })
            .collect(),
        num_speakers: num,
        elapsed_ms: 0,
    }
}

#[test]
fn a_speaker_turn_breaks_the_segment() {
    let mut tr = one_word_per_segment(&[
        ("w0000", 0, 300),
        ("w0001", 320, 620),
        ("w0002", 640, 940),
        ("w0003", 960, 1_260),
    ]);
    let d = diarization(&[(0, 0, 800), (1, 800, 2_000)], 2);
    let report = group_with_diarization(&mut tr, &d, &policy()).unwrap();
    assert_eq!(report.breaks_speaker_turn, 1);
    assert_eq!(tr.segments.len(), 2);
    assert_eq!(tr.segments[0].speaker, Some(0));
    assert_eq!(tr.segments[1].speaker, Some(1));
}

#[test]
fn a_turn_with_the_turn_rule_off_does_not_break() {
    let mut tr = one_word_per_segment(&[("w0000", 0, 300), ("w0001", 320, 620)]);
    let d = diarization(&[(0, 0, 200), (1, 200, 1_000)], 2);
    let p = GroupingPolicy {
        break_on_speaker_turn: false,
        ..GroupingPolicy::default()
    };
    let report = group_with_diarization(&mut tr, &d, &p).unwrap();
    assert_eq!(report.breaks_speaker_turn, 0);
    assert_eq!(tr.segments.len(), 1);
    // Both words are labelled: the midpoint of word 0 is 150 (speaker 0) and of
    // word 1 is 470 (speaker 1). A tie in the group goes to the lower index, and
    // the turn rule being off is what let them end up in one group at all.
    assert_eq!(tr.segments[0].speaker, Some(0));
}

#[test]
fn a_word_outside_every_span_does_not_break_the_segment() {
    // Unknown is not "a change". A hole in diarization coverage must not shatter
    // a sentence into pieces.
    let mut tr =
        one_word_per_segment(&[("w0000", 0, 300), ("w0001", 320, 620), ("w0002", 640, 940)]);
    let d = diarization(&[(0, 0, 400), (0, 900, 1_200)], 1);
    let report = group_with_diarization(&mut tr, &d, &policy()).unwrap();
    assert_eq!(report.breaks_speaker_turn, 0);
    assert_eq!(tr.segments.len(), 1);
    assert_eq!(tr.segments[0].speaker, Some(0));
}

#[test]
fn the_midpoint_decides_a_word_that_straddles_a_turn() {
    // A word from 700 to 900 with a boundary at 800: exactly 100 ms each way.
    // The midpoint lands on the boundary instant, and
    // `DiarizationResult::speaker_at` resolves an ambiguous instant to the
    // *shortest* containing span; both spans are 800 ms long here, so the first
    // listed wins. What the test pins is that a break happens at all — the
    // word is not silently assigned to whichever voice started it.
    let mut tr = one_word_per_segment(&[("w0000", 0, 300), ("w0001", 700, 900)]);
    let d = diarization(&[(1, 0, 800), (0, 800, 1_000)], 2);
    let report = group_with_diarization(&mut tr, &d, &policy()).unwrap();
    assert_eq!(report.breaks_speaker_turn, 1, "{report:?}");
    assert_eq!(tr.segments.len(), 2);
}

#[test]
fn a_word_mostly_in_the_second_voice_is_labelled_with_it() {
    // Word 700..900 with the boundary at 750: 50 ms of speaker 0, 150 ms of
    // speaker 1. The midpoint at 800 is inside speaker 1's span, so the group
    // turns there. A start-time lookup would have said speaker 0 and put the
    // break a word later.
    let mut tr = one_word_per_segment(&[("w0000", 0, 300), ("w0001", 700, 900)]);
    let d = diarization(&[(0, 0, 750), (1, 750, 1_000)], 2);
    let report = group_with_diarization(&mut tr, &d, &policy()).unwrap();
    assert_eq!(report.breaks_speaker_turn, 1, "{report:?}");
    assert_eq!(tr.segments.len(), 2);
    assert_eq!(tr.segments[0].speaker, Some(0));
    assert_eq!(tr.segments[1].speaker, Some(1));
}

#[test]
fn a_word_whose_midpoint_is_in_the_first_voice_keeps_the_first_voice() {
    // The mirror of the previous test: 700..900 with the boundary at 850. Only
    // 50 ms of speaker 1, and the midpoint at 800 is still speaker 0's.
    let mut tr = one_word_per_segment(&[("w0000", 0, 300), ("w0001", 700, 900)]);
    let d = diarization(&[(0, 0, 850), (1, 850, 1_000)], 2);
    let report = group_with_diarization(&mut tr, &d, &policy()).unwrap();
    assert_eq!(report.breaks_speaker_turn, 0, "{report:?}");
    assert_eq!(tr.segments.len(), 1);
    assert_eq!(tr.segments[0].speaker, Some(0));
}

#[test]
fn grouping_registers_every_speaker_it_uses() {
    let mut tr = one_word_per_segment(&[("w0000", 0, 300), ("w0001", 5_000, 5_300)]);
    let d = diarization(&[(0, 0, 1_000), (1, 5_000, 6_000)], 2);
    group_with_diarization(&mut tr, &d, &policy()).unwrap();
    for seg in &tr.segments {
        let s = seg.speaker.expect("both words are covered");
        assert!(tr.speaker(s).is_some(), "speaker {s} does not resolve");
    }
    assert_eq!(tr.speakers.len(), 2);
    let sum: f32 = tr.speakers.iter().filter_map(|s| s.share).sum();
    assert!((sum - 1.0).abs() < 1e-5, "shares summed to {sum}");
}

#[test]
fn a_user_rename_survives_grouping() {
    let mut tr = one_word_per_segment(&[("w0000", 0, 300), ("w0001", 5_000, 5_300)]);
    tr.add_speaker(vaani_core::Speaker::new(0));
    tr.speaker_mut(0).name = Some("Krishna".into());
    let d = diarization(&[(0, 0, 1_000), (1, 5_000, 6_000)], 2);
    group_with_diarization(&mut tr, &d, &policy()).unwrap();
    assert_eq!(
        tr.speaker(0).unwrap().name.as_deref(),
        Some("Krishna"),
        "regrouping must not throw away a name the user typed"
    );
}

#[test]
fn an_empty_diarization_result_attributes_nothing_and_does_not_panic() {
    // `--no-diarize` reaches here, and so does a diarizer that found no
    // speakers. Both are complete product states, not failures.
    let mut tr = one_word_per_segment(&[("w0000", 0, 300), ("w0001", 5_000, 5_300)]);
    let report = group_with_diarization(&mut tr, &DiarizationResult::default(), &policy()).unwrap();
    assert_eq!(report.attributed, 0);
    assert_eq!(report.unattributed, tr.segments.len());
    assert_eq!(report.breaks_speaker_turn, 0);
    assert!(tr.segments.iter().all(|s| s.speaker.is_none()));
    assert_eq!(report.words_in, report.words_out);
}

#[test]
fn a_span_naming_an_unregistered_speaker_is_ignored_not_fatal() {
    let mut tr = one_word_per_segment(&[("w0000", 0, 300), ("w0001", 5_000, 5_300)]);
    let d = diarization(&[(7, 0, 10_000)], 1);
    let report = group_with_diarization(&mut tr, &d, &policy()).unwrap();
    assert_eq!(report.attributed, 0, "{report:?}");
    assert!(tr.segments.iter().all(|s| s.speaker.is_none()));
}

#[test]
fn grouping_without_diarization_honours_existing_speakers() {
    // The ASR-only call site still has to be safe on a transcript that was
    // already attributed, or an orchestrator that attributes before grouping
    // would throw the attribution away.
    let mut tr = one_word_per_segment(&[("w0000", 0, 300), ("w0001", 5_000, 5_300)]);
    tr.segments[0].speaker = Some(0);
    tr.segments[1].speaker = Some(1);
    group(&mut tr, &policy()).unwrap();
    assert_eq!(tr.segments[0].speaker, Some(0));
    assert_eq!(tr.segments[1].speaker, Some(1));
}

// ---------------------------------------------------------------- ordering

#[test]
fn output_segments_are_sorted_and_non_overlapping() {
    let mut tr = one_word_per_segment(&[
        ("w0000", 0, 300),
        ("w0001", 5_000, 5_300),
        ("w0002", 200, 500),
        ("w0003", 900, 1_200),
    ]);
    group(&mut tr, &policy()).unwrap();
    for pair in tr.segments.windows(2) {
        assert!(
            pair[0].end <= pair[1].start,
            "{}..{} overlaps {}..{}",
            pair[0].start.as_millis(),
            pair[0].end.as_millis(),
            pair[1].start.as_millis(),
            pair[1].end.as_millis()
        );
    }
    let starts: Vec<i64> = tr.segments.iter().map(|s| s.start.as_millis()).collect();
    let mut sorted = starts.clone();
    sorted.sort_unstable();
    assert_eq!(starts, sorted);
}

#[test]
fn words_out_of_order_input_comes_out_sorted_and_is_reported() {
    let mut tr = one_word_per_segment(&[
        ("w0002", 900, 1_200),
        ("w0000", 0, 300),
        ("w0001", 400, 700),
    ]);
    let report = group(&mut tr, &policy()).unwrap();
    // Every one of the three positions holds a different word after the sort, so
    // all three are reported as moved. The count is of *positions*, not of
    // inversions, because that is the quantity a reader can check by eye.
    assert_eq!(report.reordered_words, 3, "{report:?}");
    assert_eq!(text_of(&tr), vec!["w0000 w0001 w0002"]);
    assert!(!report.input_was_well_formed());
}

#[test]
fn already_sorted_input_reports_no_reordering() {
    let mut tr = run(40, 0);
    let report = group(&mut tr, &policy()).unwrap();
    assert_eq!(report.reordered_words, 0);
    assert!(report.input_was_well_formed(), "{report:?}");
}

#[test]
fn words_with_identical_times_keep_their_input_order() {
    // A stable sort is the only thing standing between a duplicate timestamp and
    // a transcript whose words are silently reordered.
    let mut tr = one_word_per_segment(&[
        ("w0000", 100, 200),
        ("w0001", 100, 200),
        ("w0002", 100, 200),
    ]);
    group(&mut tr, &policy()).unwrap();
    assert_eq!(text_of(&tr), vec!["w0000 w0001 w0002"]);
}

// ------------------------------------------------------------- media bounds

#[test]
fn segments_are_clamped_into_the_media_bounds() {
    // `push_chunk` has no upper clamp, so a word can land past the end of the
    // recording. The buffer is the authority on where the media ends, and the
    // caller states that window explicitly — see `MediaBounds`.
    let mut tr = one_word_per_segment(&[("w0000", 9_800, 10_023)]);
    let report = vaani_segment::group_within(
        &mut tr,
        vaani_segment::MediaBounds::of_duration(t(10_000)),
        &policy(),
    )
    .unwrap();
    assert_eq!(tr.segments[0].end, t(10_000));
    assert_eq!(report.segments_clamped_to_media, 1);
    assert!(!report.input_was_well_formed());
    // The word's own time is untouched: grouping does not move words.
    assert_eq!(tr.segments[0].words[0].end, t(10_023));
}

#[test]
fn without_bounds_nothing_is_clamped() {
    // The zero-based entry points do not know where the media starts, so they
    // clamp nothing rather than guessing — a length is not an absolute time.
    let mut tr = one_word_per_segment(&[("w0000", 9_800, 10_023)]);
    tr.duration = Some(t(10_000));
    let report = group(&mut tr, &policy()).unwrap();
    assert_eq!(tr.segments[0].end, t(10_023), "nothing to clamp against");
    assert_eq!(report.segments_clamped_to_media, 0);
}

#[test]
fn a_transcript_that_does_not_start_at_zero_is_not_pinned_to_the_origin() {
    // Regression. A10's conformance suite caught this: the media clamp compared
    // an **absolute** segment end against `Transcript::duration`, which is a
    // **length**. For a transcript whose times run 300 000..600 000 with a
    // declared duration of 300 000, that crushed all three segments to
    // 300 000..300 001 and left every word outside its own segment. Three
    // conformance checks failed at once.
    let mut tr = Transcript::new(EngineStamp::asr("x", "0"));
    tr.duration = Some(t(300_000));
    let half = 150_000;
    tr.push_chunk(
        t(300_000),
        vec![
            Segment::new(t(0), t(half)).with_words(vec![
                Word::new("alpha", t(0), t(half / 2)),
                Word::new("bravo", t(half / 2), t(half)),
            ]),
            Segment::new(t(half), t(300_000)).with_words(vec![Word::new(
                "charlie",
                t(half),
                t(300_000),
            )]),
        ],
    );
    // No bounds: the transcript says what it says and grouping does not guess.
    let report = group(&mut tr, &policy()).unwrap();
    assert_eq!(report.segments_clamped_to_media, 0);
    for seg in &tr.segments {
        assert!(seg.start >= t(300_000), "{:?}", seg.start);
        for w in &seg.words {
            assert!(
                w.start >= seg.start && w.end <= seg.end,
                "a word fell outside its own segment"
            );
        }
    }
    assert_eq!(report.words_in, 3);
    assert_eq!(report.words_out, 3);
}

#[test]
fn a_transcript_with_no_known_duration_is_not_clamped() {
    let mut tr = one_word_per_segment(&[("w0000", 0, 300)]);
    assert!(tr.duration.is_none());
    let report = group(&mut tr, &policy()).unwrap();
    assert_eq!(tr.segments[0].end, t(300));
    assert_eq!(report.segments_clamped_to_media, 0);
}

#[test]
fn a_negative_word_time_is_clamped_to_zero_without_moving_the_word() {
    let mut tr = one_word_per_segment(&[("w0000", -40, 300)]);
    group(&mut tr, &policy()).unwrap();
    assert_eq!(tr.segments[0].start, t(0));
    assert_eq!(
        tr.segments[0].words[0].start,
        t(-40),
        "the word keeps its time"
    );
}

// ----------------------------------------------------------------- metadata

/// One word's identity, for the field-by-field comparison that proves grouping
/// moved nothing.
type WordKey = (String, i64, i64, Option<f32>, Option<String>);

fn word_keys(t: &Transcript) -> Vec<WordKey> {
    t.segments
        .iter()
        .flat_map(|s| s.words.iter())
        .map(|w| {
            (
                w.text.clone(),
                w.start.as_millis(),
                w.end.as_millis(),
                w.confidence,
                w.normalized.clone(),
            )
        })
        .collect()
}

#[test]
fn word_times_are_never_changed() {
    // The single most damaging thing this crate could do. Every word, in order,
    // compared field by field before and after.
    let before = word_keys(&run(40, 0));
    let mut tr = run(40, 0);
    group(&mut tr, &policy()).unwrap();
    assert_eq!(before, word_keys(&tr));
}

#[test]
fn confidence_becomes_the_mean_of_the_groups_words() {
    let mut tr = one_word_per_segment(&[("w0000", 0, 300), ("w0001", 320, 620)]);
    tr.segments[0].words[0].confidence = Some(0.4);
    tr.segments[1].words[0].confidence = Some(0.8);
    group(&mut tr, &policy()).unwrap();
    let c = tr.segments[0].confidence.expect("both words reported one");
    assert!((c - 0.6).abs() < 1e-6, "mean was {c}");
}

#[test]
fn a_group_whose_words_report_no_confidence_has_none() {
    let mut tr = one_word_per_segment(&[("w0000", 0, 300), ("w0001", 320, 620)]);
    group(&mut tr, &policy()).unwrap();
    assert!(tr.segments[0].confidence.is_none());
}

#[test]
fn the_first_non_empty_language_wins() {
    let mut tr = one_word_per_segment(&[("w0000", 0, 300), ("w0001", 320, 620)]);
    tr.segments[0].language = Some("hinglish".into());
    tr.segments[1].language = Some("hi".into());
    group(&mut tr, &policy()).unwrap();
    assert_eq!(tr.segments[0].language.as_deref(), Some("hinglish"));
}

#[test]
fn grouping_leaves_the_rest_of_the_transcript_alone() {
    let mut tr = one_word_per_segment(&[("w0000", 0, 300), ("w0001", 5_000, 5_300)]);
    let id = tr.id.clone();
    tr.duration = Some(t(60_000));
    tr.mode = vaani_core::TranscriptMode::Clean;
    tr.edited = true;
    tr.engine.diarization_model = Some("heuristic-v1".into());
    group(&mut tr, &policy()).unwrap();
    assert_eq!(tr.id, id);
    assert_eq!(tr.duration, Some(t(60_000)));
    assert_eq!(tr.mode, vaani_core::TranscriptMode::Clean);
    assert!(tr.edited);
    assert_eq!(tr.engine.diarization_model.as_deref(), Some("heuristic-v1"));
}

// --------------------------------------------------------------------- ids

#[test]
fn a_no_op_grouping_leaves_every_id_untouched() {
    // Three isolated words, each already its own segment: grouping cannot merge
    // them, so nothing should be renamed.
    let mut tr = one_word_per_segment(&[
        ("w0000", 0, 300),
        ("w0001", 5_000, 5_300),
        ("w0002", 9_000, 9_300),
    ]);
    let before: Vec<String> = tr.segments.iter().map(|s| s.id.clone()).collect();
    group(&mut tr, &policy()).unwrap();
    let after: Vec<String> = tr.segments.iter().map(|s| s.id.clone()).collect();
    assert_eq!(before, after);
}

#[test]
fn a_group_that_merges_two_source_segments_keeps_the_first_segments_id() {
    let mut tr = one_word_per_segment(&[("w0000", 0, 300), ("w0001", 320, 620)]);
    group(&mut tr, &policy()).unwrap();
    assert_eq!(tr.segments.len(), 1);
    assert_eq!(tr.segments[0].id, "s0000");
}

#[test]
fn a_group_that_splits_a_multi_word_source_segment_names_its_offset() {
    // One source segment holding three words, split into two groups. The first
    // group starts at the source's word 0 so it keeps the bare id; the second
    // starts at word 2, so it says so. Two groups out of one source segment can
    // never collide, because only one of them starts at word 0.
    let mut tr = Transcript::new(EngineStamp::asr("sravaani", "1.0"));
    let mut seg = Segment::new(t(0), t(900)).with_words(vec![
        Word::new("w0000", t(0), t(300)),
        Word::new("w0001", t(320), t(620)),
        Word::new("w0002", t(640), t(900)),
    ]);
    seg.id = "multi".into();
    tr.segments.push(seg);
    let p = GroupingPolicy {
        max_words: 2,
        min_segment_words: 0,
        ..GroupingPolicy::default()
    };
    group(&mut tr, &p).unwrap();
    assert_eq!(tr.segments.len(), 2);
    assert_eq!(tr.segments[0].id, "multi");
    assert_eq!(tr.segments[1].id, "multi#2");
}

#[test]
fn regrouping_is_idempotent_including_ids() {
    let mut tr = run(60, 0);
    let p = GroupingPolicy {
        max_words: 7,
        min_segment_words: 2,
        ..GroupingPolicy::default()
    };
    group(&mut tr, &p).unwrap();
    let once = tr.clone();
    group(&mut tr, &p).unwrap();
    assert_eq!(
        tr.segments.iter().map(|s| s.id.clone()).collect::<Vec<_>>(),
        once.segments
            .iter()
            .map(|s| s.id.clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(tr.segments.len(), once.segments.len());
    assert!(vaani_segment::same_grouping(&tr, &once));
}

// ------------------------------------------------------ punctuation, honestly

#[test]
fn synthesised_fixtures_carry_no_punctuation() {
    // The claim "the policy does not use punctuation because there is none" only
    // means something if the fixtures themselves would have shown it.
    let tr = run(20, 0);
    assert!(vaani_segment::word_text_contains_no_punctuation(&tr));
}

#[test]
fn the_punctuation_check_would_notice_punctuation() {
    // Otherwise the check above is vacuous: it must actually be able to fail.
    let mut tr = one_word_per_segment(&[("w0000.", 0, 300)]);
    assert!(!vaani_segment::word_text_contains_no_punctuation(&tr));
    tr = one_word_per_segment(&[("w0000?", 0, 300)]);
    assert!(!vaani_segment::word_text_contains_no_punctuation(&tr));
    // The Devanagari danda, which is what an Indic model *would* emit if it
    // emitted punctuation at all.
    tr = one_word_per_segment(&[("w0000\u{0964}", 0, 300)]);
    assert!(!vaani_segment::word_text_contains_no_punctuation(&tr));
    // A comma, a full stop, a question mark and a bracket.
    for punct in [",", ".", "!", "?", ";", ":", "\"", "(", ")", "-"] {
        let tr = one_word_per_segment(&[(punct, 0, 300)]);
        assert!(
            !vaani_segment::word_text_contains_no_punctuation(&tr),
            "{punct} was not detected"
        );
    }
    // And it must not fire on letters, digits, marks or emoji.
    for clean in ["abc", "123", "w0000", "\u{093e}", "caf\u{e9}", "\u{1f3a7}"] {
        let tr = one_word_per_segment(&[(clean, 0, 300)]);
        assert!(
            vaani_segment::word_text_contains_no_punctuation(&tr),
            "{clean} was wrongly called punctuation"
        );
    }
}

// ---------------------------------------------------------------- the plan

#[test]
fn planning_does_not_mutate_the_transcript() {
    let tr = run(30, 0);
    let plan = plan_groups(&tr, None, None, &policy()).unwrap();
    assert_eq!(tr.segments.len(), 30, "the input is untouched");
    assert!(!plan.groups.is_empty());
    assert_eq!(plan.report.words_in, 30);
    assert_eq!(plan.report.words_out, 30);
}

#[test]
fn an_invalid_policy_is_an_error_rather_than_a_silent_guess() {
    let mut tr = run(10, 0);
    let p = GroupingPolicy {
        pause_break_ms: 900,
        max_gap_ms: 300,
        ..GroupingPolicy::default()
    };
    let err = group(&mut tr, &p).unwrap_err();
    assert!(err.debug_chain().contains("not self-consistent"), "{err:?}");
    assert_eq!(tr.segments.len(), 10, "and nothing was changed");
}

#[test]
fn the_word_count_is_the_duplicate_check() {
    // A test that only compared the *text* of the output would pass a
    // duplicated word that happens to be textually identical to its neighbour,
    // which is the only way a duplicate can hide. Comparing counts cannot.
    let mut tr = one_word_per_segment(&[
        ("w0000", 0, 300),
        ("w0001", 320, 620),
        ("w0002", 640, 940),
        ("w0003", 5_000, 5_300),
    ]);
    let report = group(&mut tr, &policy()).unwrap();
    let flat: Vec<&Word> = tr.segments.iter().flat_map(|s| s.words.iter()).collect();
    assert_eq!(flat.len(), 4, "each word exactly once");
    assert_eq!(report.words_in, 4);
    assert_eq!(report.words_out, flat.len());
    // And the text, for the case a reader actually cares about.
    assert_eq!(text_of(&tr), vec!["w0000 w0001 w0002", "w0003"]);
}

#[test]
fn two_identical_words_at_identical_times_are_both_kept() {
    // The adversarial version: a duplicate would be invisible to any text-based
    // check, because the two words are indistinguishable.
    let mut tr = one_word_per_segment(&[("w0000", 100, 200), ("w0000", 100, 200)]);
    let report = group(&mut tr, &policy()).unwrap();
    let flat: Vec<&Word> = tr.segments.iter().flat_map(|s| s.words.iter()).collect();
    assert_eq!(flat.len(), 2, "{report:?}");
    assert_eq!(report.words_in, 2);
    assert_eq!(report.words_out, 2);
}

#[test]
fn a_regroup_over_a_differently_grouped_transcript_chains_the_id_suffix() {
    // The documented cost of derived ids, pinned with a constructed case rather
    // than left to whichever random seed produces it. Two passes with
    // *different* policies: the first merges three words, the second splits them
    // again, so the second pass's source segment is the first pass's derived id
    // and the suffix appends rather than replaces.
    let mut tr =
        one_word_per_segment(&[("w0000", 0, 300), ("w0001", 320, 620), ("w0002", 640, 940)]);
    let merged = GroupingPolicy {
        max_words: 24,
        min_segment_words: 0,
        ..GroupingPolicy::default()
    };
    group(&mut tr, &merged).unwrap();
    assert_eq!(tr.segments.len(), 1);
    assert_eq!(tr.segments[0].id, "s0000");

    let split = GroupingPolicy {
        max_words: 2,
        min_segment_words: 0,
        ..GroupingPolicy::default()
    };
    group(&mut tr, &split).unwrap();
    assert_eq!(tr.segments.len(), 2);
    assert_eq!(
        tr.segments[0].id, "s0000",
        "starts at the group's own first word"
    );
    assert_eq!(tr.segments[1].id, "s0000#2", "and the suffix chains");
    // Words and boundaries are still exact.
    assert_eq!(tr.segments[0].words.len(), 2);
    assert_eq!(tr.segments[1].words.len(), 1);
    assert_eq!(report_words(&tr), 3);
}

#[test]
fn a_regroup_can_change_a_segments_language_and_that_is_documented() {
    // `Word` has no language field, so a group's language is a function of the
    // segments its words came from. Re-grouping therefore cannot always
    // reproduce it. This test states the limitation so that a future change which
    // fixes it has to delete this test deliberately rather than discover it.
    let mut tr = one_word_per_segment(&[("w0000", 0, 300), ("w0001", 320, 620)]);
    tr.segments[0].language = Some("hinglish".into());
    tr.segments[1].language = None;
    let p = GroupingPolicy {
        max_words: 24,
        min_segment_words: 0,
        ..GroupingPolicy::default()
    };
    group(&mut tr, &p).unwrap();
    assert_eq!(tr.segments[0].language.as_deref(), Some("hinglish"));

    // Split it again: the first group starts on the tagged word, the second on
    // the untagged one, and the first pass's `hinglish` is gone from the words.
    let split = GroupingPolicy {
        max_words: 1,
        min_segment_words: 0,
        ..GroupingPolicy::default()
    };
    group(&mut tr, &split).unwrap();
    // Both halves now report `hinglish`, including the half whose only word came
    // from a segment that was never tagged. That is the leak: the first pass
    // stamped the *merged* segment, and the second pass can only see the merged
    // segment. It cannot arise in production — the pipeline groups once, and the
    // per-segment language is the ASR stage's to set — but it is the reason the
    // cross-ordering property test compares segmentation and not language.
    assert_eq!(tr.segments[0].language.as_deref(), Some("hinglish"));
    assert_eq!(tr.segments[1].language.as_deref(), Some("hinglish"));
}

// ------------------------------------------------------- the gap measurement

#[test]
fn the_gap_distribution_measures_the_silence_between_words() {
    // The measurement the pause threshold is chosen from, on a fixture whose
    // gaps are known exactly. Gaps: 200, 500, 1500.
    let tr = one_word_per_segment(&[
        ("w0000", 0, 300),
        ("w0001", 500, 800),
        ("w0002", 1_300, 1_600),
        ("w0003", 3_100, 3_400),
    ]);
    let d = vaani_segment::gap_distribution(&tr);
    assert_eq!(d.gaps, 3);
    assert_eq!(d.negative, 0);
    assert_eq!(d.zero, 0);
    assert_eq!(d.sorted_gaps(), vec![200, 500, 1_500]);
    // Nearest rank over three values: p50 is the middle one.
    assert_eq!(d.percentile(0.5), 500);
    assert_eq!(d.percentile(0.9), 1_500);
    assert!((d.fraction_at_or_above(500) - 2.0 / 3.0).abs() < 1e-9);
    assert!((d.fraction_at_or_above(2_000) - 0.0).abs() < 1e-9);
    assert!((d.fraction_at_or_above(0) - 1.0).abs() < 1e-9);
    // The pre-computed arrays line up with the query methods.
    assert_eq!(d.percentiles.len(), vaani_segment::PERCENTILES.len());
    assert_eq!(d.at_or_above.len(), vaani_segment::THRESHOLD_SWEEP.len());
    for (t, f) in vaani_segment::THRESHOLD_SWEEP
        .iter()
        .zip(d.at_or_above.iter())
    {
        assert!(
            (f - d.fraction_at_or_above(*t)).abs() < 1e-12,
            "threshold {t}"
        );
    }
}

#[test]
fn overlapping_and_abutting_words_are_counted_not_hidden() {
    // A chunk overlap produces a negative gap and `push_chunk` can produce an
    // abutting one. Both are real, and a distribution that hid them would make
    // the threshold look better chosen than the data supports.
    let tr = one_word_per_segment(&[
        ("w0000", 0, 500),
        ("w0001", 300, 800),
        ("w0002", 800, 1_100),
    ]);
    let d = vaani_segment::gap_distribution(&tr);
    assert_eq!(d.gaps, 2);
    assert_eq!(d.negative, 1);
    assert_eq!(d.zero, 1);
    assert_eq!(d.sorted_gaps(), vec![-200, 0]);
}

#[test]
fn a_transcript_with_fewer_than_two_words_has_no_gaps() {
    for tr in [
        Transcript::new(EngineStamp::asr("t", "1")),
        one_word_per_segment(&[("w0000", 0, 300)]),
    ] {
        let d = vaani_segment::gap_distribution(&tr);
        assert_eq!(d.gaps, 0);
        assert_eq!(d.percentile(0.5), 0);
        assert_eq!(d.fraction_at_or_above(0), 0.0);
    }
}

#[test]
fn the_duration_cap_is_what_catches_an_unbroken_run_of_abutting_words() {
    // The pathology the real measurement found, reproduced in miniature: 1,889 of
    // the 1,911 inter-word gaps on the real meeting are exactly zero, because the
    // ASR gives each word the frame immediately after the previous one. A
    // continuous run of speech is therefore *one unbroken span* to the grouper,
    // and the pause rule can never fire inside it. Only the duration cap ends it.
    //
    // This is the single most important fixture in the file: it is why
    // `max_duration_ms` is a rule and not a nicety.
    let words: Vec<(String, i64, i64)> = (0..200)
        .map(|i| {
            // Exactly abutting: word i+1 starts the millisecond word i ends.
            let s = i as i64 * 100;
            (format!("w{i:04}"), s, s + 100)
        })
        .collect();
    let borrowed: Vec<(&str, i64, i64)> =
        words.iter().map(|(a, b, c)| (a.as_str(), *b, *c)).collect();
    let mut tr = one_word_per_segment(&borrowed);
    let d = vaani_segment::gap_distribution(&tr);
    assert_eq!(d.gaps, 199);
    assert_eq!(d.zero, 199, "every gap is exactly zero");
    assert_eq!(
        d.fraction_at_or_above(1),
        0.0,
        "and the pause rule cannot fire at all"
    );

    let p = GroupingPolicy {
        max_words: 0,
        max_chars: 0,
        max_duration_ms: 5_000,
        min_segment_words: 0,
        ..GroupingPolicy::default()
    };
    let report = group(&mut tr, &p).unwrap();
    assert_eq!(report.breaks_pause, 0);
    assert_eq!(report.breaks_hard_gap, 0);
    assert!(report.breaks_duration_cap > 0, "{report:?}");
    for seg in &tr.segments {
        assert!(seg.duration().as_millis() <= 5_000, "{:?}", seg.duration());
    }
    // And with the duration cap off, the whole run is one segment — which is the
    // bug this rule exists to prevent.
    let mut uncapped = one_word_per_segment(&borrowed);
    let report = group(
        &mut uncapped,
        &GroupingPolicy {
            max_words: 0,
            max_chars: 0,
            max_duration_ms: 0,
            min_segment_words: 0,
            ..GroupingPolicy::default()
        },
    )
    .unwrap();
    assert_eq!(report.segments_out, 1, "20 s of speech in one segment");
}

#[test]
fn an_overlapping_word_is_never_orphaned_outside_its_own_segment() {
    // Regression. A10's conformance suite caught this on the diarization stage:
    // when a cut landed between two *overlapping* words, the two groups' spans
    // overlapped, and the repair — pushing the later segment's start forward to
    // the earlier one's end — left that segment's own first word **outside its
    // own segment**. Three conformance checks failed at once.
    //
    // The fix is upstream of the repair: the cut is moved back over the
    // offending word, so each group's word range is disjoint from the next and
    // no repair is needed. Here word 1 (100..400) overlaps word 0 (0..300), and
    // the pause at word 2 would have cut between them.
    let mut tr = one_word_per_segment(&[
        ("w0000", 0, 300),
        ("w0001", 100, 400),
        ("w0002", 5_000, 5_300),
    ]);
    let report = group(&mut tr, &policy()).unwrap();
    // Word 1 overlaps word 0, so the pause before word 2 cannot be honoured:
    // the first group would span 0..400 and the second would start at 100.
    // All three words therefore share one segment, and the later break is the
    // one that survives.
    assert_eq!(tr.segments.len(), 2, "{:?}", text_of(&tr));
    assert_eq!(tr.segments[0].words.len(), 2);
    assert_eq!(tr.segments[1].words.len(), 1);
    assert_eq!(tr.segments[0].end.as_millis(), 400);
    assert_eq!(tr.segments[1].start.as_millis(), 5_000);
    for seg in &tr.segments {
        for w in &seg.words {
            assert!(w.start >= seg.start && w.end <= seg.end);
        }
    }
    assert_eq!(report.words_in, report.words_out);
    assert_eq!(
        report.segments_pushed_for_overlap, 0,
        "no repair was needed"
    );
}

#[test]
fn the_first_two_words_must_share_a_segment_when_they_overlap() {
    // The degenerate case: a cut between words 0 and 1 cannot be made disjoint
    // without emptying the first group, so it is dropped. Two overlapping words
    // can only be represented non-overlapping by putting them in one segment.
    let mut tr = one_word_per_segment(&[
        ("w0000", 0, 500),
        ("w0001", 100, 900),
        ("w0002", 5_000, 5_300),
    ]);
    let report = group(&mut tr, &policy()).unwrap();
    assert_eq!(tr.segments.len(), 2, "{:?}", text_of(&tr));
    assert_eq!(tr.segments[0].words.len(), 2);
    for seg in &tr.segments {
        for w in &seg.words {
            assert!(w.start >= seg.start && w.end <= seg.end);
        }
    }
    assert_eq!(report.words_in, report.words_out);
    assert_eq!(
        report.segments_pushed_for_overlap, 0,
        "no repair was needed"
    );
}

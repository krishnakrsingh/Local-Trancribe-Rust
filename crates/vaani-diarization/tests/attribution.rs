//! Speaker attribution onto a transcript: the majority vote, its edge cases,
//! and the "no speaker" mode.
//!
//! # The cases that matter
//!
//! * A segment that covers two speakers must land on the one that owns more of
//!   it, by **overlap duration**, not by whichever span happens to come first
//!   and not by a lookup at the segment's start.
//! * A segment that covers nothing must be `None`, and that is a normal
//!   complete product mode, not an error and not a panic.
//! * A segment whose ends land exactly on span boundaries is half-open, so the
//!   boundary instant belongs to the later span and contributes no time to the
//!   earlier one.
//! * A `DiarizationResult` whose `num_speakers` disagrees with its spans must
//!   resolve deterministically and must not panic: a span naming an
//!   unregistered speaker is ignored, exactly as
//!   `DiarizationResult::speaker_durations` ignores it.
//! * The reverse direction: `Vec<Speaker>` with talk-time shares.

use vaani_core::Time;
use vaani_core::pipeline::{DiarizationResult, SpeakerSpan};
use vaani_core::transcript::{EngineStamp, Segment, Speaker, Transcript, Word};
use vaani_diarization::attribution::{self, AttributionReport};

fn t(ms: i64) -> Time {
    Time::from_millis(ms)
}

fn span(speaker: usize, start: i64, end: i64) -> SpeakerSpan {
    SpeakerSpan {
        speaker,
        start: t(start),
        end: t(end),
        score: Some(1.0),
    }
}

fn result(spans: Vec<SpeakerSpan>, num_speakers: usize) -> DiarizationResult {
    DiarizationResult {
        spans,
        num_speakers,
        elapsed_ms: 0,
    }
}

/// A transcript with one word per segment, so `to_plain_text` is readable and
/// a test failure can be identified by index.
fn transcript(ranges: &[(i64, i64)]) -> Transcript {
    let mut tr = Transcript::new(EngineStamp::asr("test", "1"));
    for (i, (a, b)) in ranges.iter().enumerate() {
        let seg =
            Segment::new(t(*a), t(*b)).with_words(vec![Word::new(format!("w{i}"), t(*a), t(*b))]);
        tr.segments.push(seg);
    }
    tr
}

#[test]
fn a_segment_covering_two_speakers_lands_on_the_majority() {
    // 300 ms of speaker 0, 700 ms of speaker 1.
    let d = result(vec![span(0, 0, 300), span(1, 300, 1_000)], 2);
    assert_eq!(attribution::majority_speaker(t(0), t(1_000), &d), Some(1));
}

#[test]
fn the_majority_is_by_duration_not_by_order() {
    // The same spans in the other order must give the same answer. An
    // implementation that walked the spans and took the first overlapping one
    // would answer 0 here.
    let a = result(vec![span(0, 0, 300), span(1, 300, 1_000)], 2);
    let b = result(vec![span(1, 300, 1_000), span(0, 0, 300)], 2);
    assert_eq!(
        attribution::majority_speaker(t(0), t(1_000), &a),
        attribution::majority_speaker(t(0), t(1_000), &b)
    );
    assert_eq!(attribution::majority_speaker(t(0), t(1_000), &b), Some(1));
}

#[test]
fn a_segment_covering_nothing_is_none_and_is_not_an_error() {
    let d = result(vec![span(0, 0, 1_000)], 1);
    assert_eq!(attribution::majority_speaker(t(5_000), t(6_000), &d), None);
    let mut tr = transcript(&[(0, 1_000), (5_000, 6_000)]);
    let r = attribution::attribute_speakers(&mut tr, &d);
    assert_eq!(r.attributed, 1);
    assert_eq!(r.unattributed, 1);
    assert_eq!(tr.segments[1].speaker, None);
}

#[test]
fn an_empty_result_leaves_everything_as_none() {
    // The complete ASR-only product mode.
    let mut tr = transcript(&[(0, 1_000), (1_000, 2_000)]);
    let r = attribution::attribute_speakers(&mut tr, &DiarizationResult::default());
    assert_eq!(r.attributed, 0);
    assert_eq!(r.unattributed, 2);
    assert!(tr.speakers.is_empty());
    assert!(tr.segments.iter().all(|s| s.speaker.is_none()));
    assert_eq!(r.attributed_ratio(), 0.0);
}

#[test]
fn boundary_instants_are_half_open() {
    // A segment ending exactly where a span starts owns none of that span, and
    // a segment starting there owns all of the next one. Getting this wrong
    // double-counts the boundary and can flip a marginal vote.
    let d = result(vec![span(0, 0, 1_000), span(1, 1_000, 2_000)], 2);
    assert_eq!(attribution::majority_speaker(t(500), t(1_000), &d), Some(0));
    assert_eq!(
        attribution::majority_speaker(t(1_000), t(1_500), &d),
        Some(1)
    );
    // A segment that starts and ends on the boundary owns nothing at all.
    assert_eq!(attribution::majority_speaker(t(1_000), t(1_000), &d), None);
}

#[test]
fn a_tie_is_resolved_towards_the_lower_index_every_time() {
    let d = result(vec![span(1, 0, 500), span(0, 500, 1_000)], 2);
    for _ in 0..64 {
        assert_eq!(attribution::majority_speaker(t(0), t(1_000), &d), Some(0));
    }
    // And the margin is exactly zero, which is how a caller can see it was a
    // tie rather than a narrow win.
    assert_eq!(attribution::majority_margin(t(0), t(1_000), &d), 0.0);
}

#[test]
fn the_margin_reports_how_decided_the_vote_was() {
    let d = result(vec![span(0, 0, 250), span(1, 250, 1_000)], 2);
    let (s, m) = attribution::overlap_vote(t(0), t(1_000), &d);
    assert_eq!(s, Some(1));
    assert!((m - 0.5).abs() < 1e-9, "margin {m}");
    // A unanimous segment has a margin of 1.
    let (s, m) = attribution::overlap_vote(t(300), t(900), &d);
    assert_eq!(s, Some(1));
    assert!((m - 1.0).abs() < 1e-9, "margin {m}");
}

#[test]
fn a_degenerate_segment_is_none_rather_than_a_panic() {
    // Real transcripts contain these: a chunk whose last word straddles the
    // boundary produces an inverted or zero-length segment.
    let d = result(vec![span(0, 0, 1_000)], 1);
    assert_eq!(attribution::majority_speaker(t(500), t(500), &d), None);
    assert_eq!(attribution::majority_speaker(t(500), t(100), &d), None);
    let mut tr = transcript(&[(500, 500)]);
    let r = attribution::attribute_speakers(&mut tr, &d);
    assert_eq!(r.unattributed, 1);
}

#[test]
fn a_span_naming_an_unregistered_speaker_is_ignored_not_panicked_on() {
    // The core tests this for `speaker_durations`; this is the vote, and the
    // requirement is that neither panics nor invents a speaker.
    let d = result(vec![span(7, 0, 10_000)], 1);
    assert_eq!(attribution::majority_speaker(t(0), t(10_000), &d), None);
    let mut tr = transcript(&[(0, 10_000)]);
    let r = attribution::attribute_speakers(&mut tr, &d);
    assert_eq!(r.ignored_spans, 1);
    assert_eq!(r.attributed, 0);
    assert_eq!(tr.segments[0].speaker, None);
    assert_eq!(tr.speakers.len(), 1, "the registry is num_speakers wide");
}

#[test]
fn a_stray_span_does_not_outvote_a_registered_one() {
    // The stray is ten times longer. It must still be ignored, because a
    // phantom index cannot win a vote.
    let d = result(
        vec![
            span(0, 0, 100),
            span(9, 100, 10_000),
            span(1, 10_000, 10_100),
        ],
        2,
    );
    assert_eq!(attribution::majority_speaker(t(0), t(10_100), &d), Some(0));
}

#[test]
fn every_attributed_index_resolves_against_the_registry() {
    let d = result(vec![span(0, 0, 1_000), span(1, 1_000, 2_000)], 2);
    let mut tr = transcript(&[(0, 1_000), (1_000, 2_000), (2_000, 3_000)]);
    attribution::attribute_speakers(&mut tr, &d);
    for (i, seg) in tr.segments.iter().enumerate() {
        if let Some(s) = seg.speaker {
            assert!(
                tr.speaker(s).is_some(),
                "segment {i} names speaker {s}, which is not registered"
            );
        }
    }
    assert_eq!(
        tr.speakers.iter().map(|s| s.index).collect::<Vec<_>>(),
        vec![0, 1]
    );
}

#[test]
fn a_speaker_who_says_nothing_still_gets_a_registry_row() {
    // The UI needs a row for a registered speaker even if they are silent, so
    // the settings screen can show "Speaker 3: 0%". The share is 0, not None.
    let d = result(vec![span(0, 0, 1_000)], 3);
    let mut tr = transcript(&[(0, 1_000)]);
    attribution::attribute_speakers(&mut tr, &d);
    assert_eq!(tr.speakers.len(), 3);
    assert_eq!(tr.speakers[2].share, Some(0.0));
}

#[test]
fn a_user_rename_survives_re_attribution() {
    // Re-running diarization on an edited transcript must not throw away the
    // names the user typed. `Transcript::add_speaker` replaces the whole
    // record, so this is a real risk rather than a theoretical one.
    let mut tr = transcript(&[(0, 1_000)]);
    attribution::attribute_speakers(&mut tr, &result(vec![span(0, 0, 1_000)], 1));
    tr.speaker_mut(0).name = Some("Priya".into());
    let d2 = result(vec![span(0, 0, 500), span(1, 500, 1_000)], 2);
    attribution::attribute_speakers(&mut tr, &d2);
    assert_eq!(tr.speaker(0).unwrap().name.as_deref(), Some("Priya"));
    assert_eq!(tr.speaker(1).unwrap().name, None);
    assert!(tr.to_plain_text().contains("Priya"));
}

#[test]
fn shares_are_recomputed_and_sum_to_one() {
    let d = result(vec![span(0, 0, 250), span(1, 250, 1_000)], 2);
    let mut tr = transcript(&[(0, 250), (250, 1_000)]);
    attribution::attribute_speakers(&mut tr, &d);
    let sum: f32 = tr.speakers.iter().filter_map(|s| s.share).sum();
    assert!((sum - 1.0).abs() < 1e-5, "shares summed to {sum}");
    assert!((tr.speakers[0].share.unwrap() - 0.25).abs() < 1e-5);
}

#[test]
fn the_report_flags_a_contested_segment() {
    let d = result(vec![span(0, 0, 500), span(1, 500, 1_000)], 2);
    let mut tr = transcript(&[(0, 1_000)]);
    let r: AttributionReport = attribution::attribute_speakers(&mut tr, &d);
    assert_eq!(r.attributed, 1);
    assert_eq!(r.contested, 1, "a 50/50 segment is not a clear win");
    assert_eq!(r.speakers_used, 1, "the loser owns no segment");
    assert!((r.attributed_ratio() - 1.0).abs() < 1e-9);
}

#[test]
fn a_clear_winner_is_not_flagged_as_contested() {
    let d = result(vec![span(0, 0, 100), span(1, 100, 1_000)], 2);
    let mut tr = transcript(&[(0, 1_000)]);
    let r = attribution::attribute_speakers(&mut tr, &d);
    assert_eq!(r.contested, 0);
}

#[test]
fn the_reverse_direction_gives_talk_time_shares() {
    let d = result(vec![span(0, 0, 3_000), span(1, 4_000, 5_000)], 2);
    let speakers: Vec<Speaker> = attribution::speakers_from_diarization(&d);
    assert_eq!(speakers.len(), 2);
    assert_eq!(speakers[0].index, 0);
    assert_eq!(speakers[1].index, 1);
    assert!((speakers[0].share.unwrap() - 0.75).abs() < 1e-6);
    assert!((speakers[1].share.unwrap() - 0.25).abs() < 1e-6);
    // Usable as-is in a transcript, which is the point of building them here.
    let mut tr = transcript(&[(0, 4_000)]);
    attribution::apply(&mut tr, &d, &speakers);
    assert_eq!(tr.speaker(0).unwrap().name, None);
    assert!(tr.to_plain_text().contains("Speaker 1"));
}

#[test]
fn apply_preserves_supplied_names_and_ignores_extra_speakers() {
    let d = result(vec![span(0, 0, 1_000), span(1, 1_000, 2_000)], 2);
    let mut tr = transcript(&[(0, 2_000)]);
    attribution::apply(
        &mut tr,
        &d,
        &[
            Speaker {
                index: 0,
                name: Some("A".into()),
                share: None,
            },
            Speaker {
                index: 7,
                name: Some("ghost".into()),
                share: None,
            },
        ],
    );
    assert_eq!(tr.speaker(0).unwrap().name.as_deref(), Some("A"));
    assert!(tr.speaker(7).is_none(), "a phantom speaker was registered");
    assert_eq!(tr.speakers.len(), 2);
}

#[test]
fn attribute_segment_writes_the_vote_and_returns_the_margin() {
    let d = result(vec![span(0, 0, 250), span(1, 250, 1_000)], 2);
    let mut seg = Segment::new(t(0), t(1_000));
    let m = attribution::attribute_segment(&mut seg, &d);
    assert_eq!(seg.speaker, Some(1));
    assert!((m - 0.5).abs() < 1e-9, "margin {m}");
}

#[test]
fn attribute_segment_overwrites_a_stale_label() {
    // Re-running diarization must replace, not accumulate. A stale label from
    // an earlier run would be worse than no label.
    let d = result(vec![span(1, 0, 1_000)], 2);
    let mut seg = Segment::new(t(0), t(1_000)).with_speaker(Some(0));
    attribution::attribute_segment(&mut seg, &d);
    assert_eq!(seg.speaker, Some(1));
}

#[test]
fn an_empty_transcript_reports_a_zero_ratio() {
    let mut tr = Transcript::new(EngineStamp::asr("test", "1"));
    let r = attribution::attribute_speakers(&mut tr, &result(vec![span(0, 0, 10)], 1));
    assert_eq!(r.attributed, 0);
    assert_eq!(r.attributed_ratio(), 0.0);
}

#[test]
fn attribution_is_deterministic_across_runs() {
    // Speaker indices are persisted in a transcript, so a non-deterministic
    // vote means labels move between runs. Run it a hundred times.
    let d = result(
        vec![span(0, 0, 400), span(1, 400, 900), span(2, 900, 1_000)],
        3,
    );
    let ranges = [(0, 500), (400, 800), (700, 1_000)];
    let first: Vec<Option<usize>> = {
        let mut tr = transcript(&ranges);
        attribution::attribute_speakers(&mut tr, &d);
        tr.segments.iter().map(|s| s.speaker).collect()
    };
    for _ in 0..100 {
        let mut tr = transcript(&ranges);
        attribution::attribute_speakers(&mut tr, &d);
        let got: Vec<Option<usize>> = tr.segments.iter().map(|s| s.speaker).collect();
        assert_eq!(got, first, "the vote is not deterministic");
    }
}

#[test]
fn a_real_result_from_the_engine_attributes_without_gaps() {
    // The end-to-end path: run the engine, then attribute, then check that
    // every attributed index resolves and the shares are sane. This is the
    // sequence the pipeline performs.
    use vaani_core::CancelToken;
    use vaani_core::pipeline::DiarizationEngine;
    use vaani_diarization::engine::HeuristicDiarizer;
    use vaani_diarization::synth;

    let voices = synth::two_voices();
    let (samples, _) = synth::render(
        &synth::alternate(&[0, 1], 0, 4_000, 15),
        &voices,
        60_000,
        16_000,
        0.003,
        31,
    );
    let audio = vaani_core::media::AudioBuffer::mono(samples, 16_000, Time::ZERO);
    let d = HeuristicDiarizer::new()
        .diarize(&audio, &CancelToken::new(), &|_| {})
        .unwrap();
    assert!(!d.spans.is_empty());

    // ASR segments are coarser than diarization spans: 7 s each across 60 s.
    let mut tr = transcript(&[(0, 7_000), (7_000, 15_000), (15_000, 24_000)]);
    let r = attribution::attribute_speakers(&mut tr, &d);
    println!(
        "engine -> transcript: {} speakers, {} spans, {} of {} segments attributed, \
         {} contested",
        d.num_speakers,
        d.spans.len(),
        r.attributed,
        r.attributed + r.unattributed,
        r.contested
    );
    for seg in &tr.segments {
        if let Some(s) = seg.speaker {
            assert!(s < tr.speakers.len());
            assert!(tr.speaker(s).is_some());
        }
    }
    assert!(r.attributed > 0);
}

//! Mapping a `DiarizationResult` onto a `Transcript`.
//!
//! # Why this is not a lookup
//!
//! ASR segments and diarization spans are produced by different stages from
//! different grids and do not line up. An ASR segment is usually 4-15 s and
//! covers several 1.5 s diarization windows and at least one speaker change;
//! a diarization span is a turn. Asking "which speaker is this segment" with a
//! single call to `DiarizationResult::speaker_at(segment.start)` throws away
//! the whole segment and is wrong whenever the segment starts in the last
//! sentence of one turn and continues into the next.
//!
//! So attribution is a **time-overlap-weighted majority vote**: every span that
//! overlaps the segment contributes its overlap *duration*, the largest total
//! wins, and the segment is attributed there. Ties go to the lower speaker
//! index so that the same input always produces the same transcript.
//!
//! # What comes back
//!
//! `None`, always, without comment, when:
//!
//! * no span overlaps the segment (it is silence, or diarization found
//!   nothing there), or
//! * the only overlapping spans name speakers this result did not register.
//!
//! This is a **normal, complete product mode**. `Segment::speaker` is
//! `Option<usize>` and ASR-only is a first-class state, so "no speaker" is an
//! answer, not a failure. Nothing in this module returns a `Result` for that
//! reason; the only thing it can get wrong is a panicking, and it does not.
//!
//! # When `num_speakers` disagrees with the spans
//!
//! `DiarizationResult` documents that a span may name a speaker the result
//! never registered, and `DiarizationResult::speaker_durations` ignores such a
//! span rather than panicking. This module does the same: a span with
//! `speaker >= num_speakers` contributes nothing to any vote and is counted in
//! [`AttributionReport::ignored_spans`].
//!
//! The alternative — growing the registry to cover the stray span — was
//! rejected deliberately. A span naming speaker 7 out of a result that
//! registered 2 speakers is either a clustering bug or a caller that
//! hand-assembled a result; in both cases materialising a phantom
//! `Speaker 8` in the UI is worse than ignoring the span. The engine itself
//! never produces this, and [`assert_consistent`] checks that.

use vaani_core::pipeline::DiarizationResult;
use vaani_core::transcript::{Segment, Speaker, Time, Transcript};

/// The number of speakers a `DiarizationResult` claims, for attribution
/// purposes.
///
/// This is `num_speakers`, and it is the only authority. A span whose speaker
/// index is at or above it is ignored, exactly as
/// `DiarizationResult::speaker_durations` ignores it.
#[must_use]
pub fn registered_speaker_count(d: &DiarizationResult) -> usize {
    d.num_speakers
}

/// The highest speaker index any span names, ignoring unregistered ones.
#[must_use]
pub fn max_registered_speaker(d: &DiarizationResult) -> Option<usize> {
    d.spans
        .iter()
        .filter(|s| s.speaker < d.num_speakers)
        .map(|s| s.speaker)
        .max()
}

/// The winner of an overlap-weighted vote over `[start, end)`, or `None`.
///
/// Returns `(speaker, margin)` where `margin` is
/// `(winner_ms - runner_up_ms) / segment_ms`, in `[0, 1]`. A margin near zero
/// means the segment is split almost evenly between two speakers, which is
/// worth knowing and which a bare `Option<usize>` throws away. Ties go to the
/// lower index.
#[must_use]
pub fn overlap_vote(start: Time, end: Time, d: &DiarizationResult) -> (Option<usize>, f64) {
    let lo = start.as_millis();
    let hi = end.as_millis();
    let total = (hi - lo).max(0);
    if total == 0 {
        return (None, 0.0);
    }
    let mut votes: Vec<i64> = vec![0; d.num_speakers];
    for s in &d.spans {
        // A span naming an unregistered speaker is ignored, not panicked on.
        let Some(slot) = votes.get_mut(s.speaker) else {
            continue;
        };
        let o_lo = s.start.as_millis().max(lo);
        let o_hi = s.end.as_millis().min(hi);
        if o_hi > o_lo {
            *slot += o_hi - o_lo;
        }
    }
    let best = votes
        .iter()
        .enumerate()
        .fold(None::<(usize, i64)>, |acc, (i, v)| match acc {
            Some((_, b)) if *v <= b => acc,
            _ => Some((i, *v)),
        });
    match best {
        // No span overlapped, or every overlapping span was unregistered.
        Some((_, 0)) | None => (None, 0.0),
        Some((winner, w)) => {
            let runner = votes
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != winner)
                .map(|(_, v)| *v)
                .max()
                .unwrap_or(0);
            let margin = (w - runner) as f64 / total as f64;
            (Some(winner), margin.clamp(0.0, 1.0))
        }
    }
}

/// The majority speaker for one time range, or `None`.
#[must_use]
pub fn majority_speaker(start: Time, end: Time, d: &DiarizationResult) -> Option<usize> {
    overlap_vote(start, end, d).0
}

/// The margin by which the majority won, in `[0, 1]`. Zero when there is no
/// majority.
#[must_use]
pub fn majority_margin(start: Time, end: Time, d: &DiarizationResult) -> f64 {
    overlap_vote(start, end, d).1
}

/// Set one segment's speaker from a diarization result, and report why.
///
/// `None` is written for a segment with no overlapping span. That is the
/// ASR-only state, and it is written without comment.
pub fn attribute_segment(seg: &mut Segment, d: &DiarizationResult) -> f64 {
    let (speaker, margin) = overlap_vote(seg.start, seg.end, d);
    seg.speaker = speaker;
    margin
}

/// What [`attribute_speakers`] did, for logging and for the benchmark report.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AttributionReport {
    /// Segments given a speaker.
    pub attributed: usize,
    /// Segments left as `None` because nothing overlapped them.
    pub unattributed: usize,
    /// Segments whose winning margin was below `0.34`, i.e. more than a third
    /// of the segment belonged to the runner-up.
    pub contested: usize,
    /// Spans whose speaker was at or above `num_speakers` and were therefore
    /// ignored.
    pub ignored_spans: usize,
    /// Distinct speakers that ended up owning at least one segment.
    pub speakers_used: usize,
}

impl AttributionReport {
    /// Fraction of segments that got a speaker, in `[0, 1]`.
    #[must_use]
    pub fn attributed_ratio(&self) -> f64 {
        let total = self.attributed + self.unattributed;
        if total == 0 {
            return 0.0;
        }
        self.attributed as f64 / total as f64
    }
}

/// Attribute every segment of `transcript` from `d`.
///
/// Also registers `0..num_speakers` in `transcript.speakers` so every
/// attributed index resolves, and fills each speaker's `share`.
///
/// Existing `Segment::speaker` values are **overwritten**: this is a
/// diarization pass, and leaving a stale label behind from an earlier run
/// would be worse than replacing it. Existing `Speaker::name` values are
/// **preserved**, because a user who has renamed Speaker 1 must not lose that
/// when the job is re-run.
///
/// Passing a `DiarizationResult` with no spans attributes nothing and leaves
/// every segment as `None`, which is the correct ASR-only result.
pub fn attribute_speakers(transcript: &mut Transcript, d: &DiarizationResult) -> AttributionReport {
    let mut report = AttributionReport {
        ignored_spans: d
            .spans
            .iter()
            .filter(|s| s.speaker >= d.num_speakers)
            .count(),
        ..AttributionReport::default()
    };

    // Register the speakers first, so an index that a vote produces always
    // resolves. `add_speaker` replaces the whole record by index, so a name
    // the user set is carried across explicitly rather than left to chance.
    let mut existing_names: Vec<Option<String>> = vec![None; d.num_speakers];
    for s in &transcript.speakers {
        if s.index < d.num_speakers {
            existing_names[s.index] = s.name.clone();
        }
    }
    for (i, name) in existing_names.into_iter().enumerate() {
        transcript.add_speaker(Speaker {
            index: i,
            name,
            share: None,
        });
    }

    let mut used = vec![false; d.num_speakers];
    for seg in &mut transcript.segments {
        let (speaker, margin) = overlap_vote(seg.start, seg.end, d);
        seg.speaker = speaker;
        match speaker {
            Some(s) => {
                report.attributed += 1;
                if let Some(flag) = used.get_mut(s) {
                    *flag = true;
                }
                if margin < 0.34 {
                    report.contested += 1;
                }
            }
            None => report.unattributed += 1,
        }
    }
    report.speakers_used = used.iter().filter(|x| **x).count();
    transcript.recompute_speaker_shares();
    report
}

/// Build the `Vec<Speaker>` for a diarization result, with talk-time shares.
///
/// `share` is each speaker's fraction of total claimed speech time, which is
/// what a UI shows next to a name. The indices are contiguous from 0 and
/// match `DiarizationResult::num_speakers`, so `Segment::speaker` values
/// resolve against this vector directly.
#[must_use]
pub fn speakers_from_diarization(d: &DiarizationResult) -> Vec<Speaker> {
    let durations = d.speaker_durations();
    let total: i64 = durations.iter().sum();
    (0..d.num_speakers)
        .map(|i| Speaker {
            index: i,
            name: None,
            share: Some(if total > 0 {
                durations[i] as f32 / total as f32
            } else {
                0.0
            }),
        })
        .collect()
}

/// Attribute a transcript and register its speakers in one call.
///
/// Equivalent to [`speakers_from_diarization`] followed by
/// [`attribute_speakers`], except that it does not clobber names the user has
/// already set on `transcript`.
pub fn apply(
    transcript: &mut Transcript,
    d: &DiarizationResult,
    existing: &[Speaker],
) -> AttributionReport {
    for s in existing {
        if s.index < d.num_speakers {
            transcript.add_speaker(s.clone());
        }
    }
    attribute_speakers(transcript, d)
}

/// Check that a result is internally consistent: every span names a
/// registered speaker, and every registered speaker has at least one span.
///
/// A result that fails is a bug in whatever produced it. The engine asserts
/// this on its own output in release builds too, because the alternative — a
/// transcript whose speaker labels quietly do not resolve — is exactly the
/// class of defect this crate exists to not have.
#[must_use]
pub fn assert_consistent(d: &DiarizationResult) -> Vec<String> {
    let mut problems = Vec::new();
    for (i, s) in d.spans.iter().enumerate() {
        if s.speaker >= d.num_speakers {
            problems.push(format!(
                "span {i} names speaker {} but num_speakers is {}",
                s.speaker, d.num_speakers
            ));
        }
        if s.end.as_millis() <= s.start.as_millis() {
            problems.push(format!("span {i} is empty or inverted"));
        }
    }
    let mut seen = vec![false; d.num_speakers];
    for s in &d.spans {
        if s.speaker < d.num_speakers {
            seen[s.speaker] = true;
        }
    }
    for (i, ok) in seen.iter().enumerate() {
        if !*ok {
            problems.push(format!("speaker {i} is registered but owns no span"));
        }
    }
    problems
}

#[cfg(test)]
mod tests {
    use super::*;
    use vaani_core::transcript::{EngineStamp, Word};

    fn t(ms: i64) -> Time {
        Time::from_millis(ms)
    }

    fn result(spans: &[(usize, i64, i64)], n: usize) -> DiarizationResult {
        DiarizationResult {
            spans: spans
                .iter()
                .map(|(s, a, b)| vaani_core::pipeline::SpeakerSpan {
                    speaker: *s,
                    start: t(*a),
                    end: t(*b),
                    score: Some(1.0),
                })
                .collect(),
            num_speakers: n,
            elapsed_ms: 0,
        }
    }

    fn transcript(segments: &[(i64, i64)]) -> Transcript {
        let mut tr = Transcript::new(EngineStamp::asr("test", "1"));
        for (i, (a, b)) in segments.iter().enumerate() {
            tr.segments
                .push(Segment::new(t(*a), t(*b)).with_words(vec![Word::new("x", t(*a), t(*b))]));
            tr.segments[i].id = format!("seg{i}");
        }
        tr
    }

    #[test]
    fn a_segment_covering_one_speaker_gets_that_speaker() {
        let d = result(&[(0, 0, 1_000), (1, 1_000, 2_000)], 2);
        assert_eq!(majority_speaker(t(0), t(1_000), &d), Some(0));
        assert_eq!(majority_speaker(t(1_000), t(2_000), &d), Some(1));
    }

    #[test]
    fn a_segment_covering_two_speakers_lands_on_the_majority() {
        // 300 ms of speaker 0 and 700 ms of speaker 1: the vote must weight
        // by duration, not by whichever span happens to come first.
        let d = result(&[(0, 0, 300), (1, 300, 1_000)], 2);
        assert_eq!(majority_speaker(t(0), t(1_000), &d), Some(1));
        let (s, m) = overlap_vote(t(0), t(1_000), &d);
        assert_eq!(s, Some(1));
        assert!((m - 0.4).abs() < 1e-9, "margin {m}");
    }

    #[test]
    fn the_order_of_the_spans_does_not_matter() {
        let a = result(&[(0, 0, 300), (1, 300, 1_000)], 2);
        let b = result(&[(1, 300, 1_000), (0, 0, 300)], 2);
        assert_eq!(
            majority_speaker(t(0), t(1_000), &a),
            majority_speaker(t(0), t(1_000), &b)
        );
    }

    #[test]
    fn a_segment_covering_nothing_is_none_and_not_an_error() {
        let d = result(&[(0, 0, 1_000)], 1);
        assert_eq!(majority_speaker(t(5_000), t(6_000), &d), None);
        assert_eq!(majority_margin(t(5_000), t(6_000), &d), 0.0);
    }

    #[test]
    fn an_empty_result_attributes_nothing_and_does_not_panic() {
        let d = DiarizationResult::default();
        assert_eq!(majority_speaker(t(0), t(1_000), &d), None);
        let mut tr = transcript(&[(0, 1_000)]);
        let r = attribute_speakers(&mut tr, &d);
        assert_eq!(r.attributed, 0);
        assert_eq!(r.unattributed, 1);
        assert_eq!(tr.segments[0].speaker, None);
    }

    #[test]
    fn a_degenerate_segment_is_none_rather_than_a_panic() {
        let d = result(&[(0, 0, 1_000)], 1);
        // Inverted and zero-length segments both appear in real transcripts
        // from a chunk whose last word straddles the boundary.
        assert_eq!(majority_speaker(t(500), t(500), &d), None);
        assert_eq!(majority_speaker(t(500), t(100), &d), None);
    }

    #[test]
    fn boundary_instants_are_half_open() {
        // A segment ending exactly where a span starts owns none of that span.
        let d = result(&[(0, 0, 1_000), (1, 1_000, 2_000)], 2);
        assert_eq!(majority_speaker(t(500), t(1_000), &d), Some(0));
        assert_eq!(majority_speaker(t(1_000), t(1_500), &d), Some(1));
    }

    #[test]
    fn a_tie_goes_to_the_lower_index_deterministically() {
        let d = result(&[(1, 0, 500), (0, 500, 1_000)], 2);
        for _ in 0..16 {
            assert_eq!(majority_speaker(t(0), t(1_000), &d), Some(0));
        }
    }

    #[test]
    fn a_span_naming_an_unregistered_speaker_is_ignored_not_counted() {
        // The core tests this for `speaker_durations`; this is the same rule
        // on the vote, and the requirement is that it must not panic.
        let d = result(&[(7, 0, 10_000)], 1);
        assert_eq!(majority_speaker(t(0), t(10_000), &d), None);
        let mut tr = transcript(&[(0, 1_000)]);
        let r = attribute_speakers(&mut tr, &d);
        assert_eq!(r.ignored_spans, 1);
        assert_eq!(r.attributed, 0);
        assert_eq!(tr.segments[0].speaker, None);
    }

    #[test]
    fn a_stray_registered_span_does_not_shadow_a_real_one() {
        let d = result(&[(0, 0, 100), (9, 0, 9_000), (1, 100, 200)], 2);
        assert_eq!(majority_speaker(t(0), t(200), &d), Some(0));
    }

    #[test]
    fn registration_covers_every_attributed_index() {
        let d = result(&[(0, 0, 1_000), (1, 1_000, 2_000)], 2);
        let mut tr = transcript(&[(0, 1_000), (1_000, 2_000)]);
        attribute_speakers(&mut tr, &d);
        for seg in &tr.segments {
            if let Some(s) = seg.speaker {
                assert!(tr.speaker(s).is_some(), "speaker {s} was not registered");
            }
        }
        assert_eq!(tr.speakers.len(), 2);
        assert_eq!(tr.speakers[0].index, 0);
        assert_eq!(tr.speakers[1].index, 1);
    }

    #[test]
    fn a_user_rename_survives_re_attribution() {
        // Re-running diarization on an edited transcript must not throw away
        // the names the user typed.
        let d = result(&[(0, 0, 1_000)], 1);
        let mut tr = transcript(&[(0, 1_000)]);
        attribute_speakers(&mut tr, &d);
        tr.speaker_mut(0).name = Some("Priya".into());
        let d2 = result(&[(0, 0, 500), (1, 500, 1_000)], 2);
        attribute_speakers(&mut tr, &d2);
        assert_eq!(tr.speaker(0).unwrap().name.as_deref(), Some("Priya"));
        assert_eq!(tr.speaker(1).unwrap().name, None);
    }

    #[test]
    fn shares_are_computed_after_attribution() {
        let d = result(&[(0, 0, 250), (1, 250, 1_000)], 2);
        let mut tr = transcript(&[(0, 250), (250, 1_000)]);
        attribute_speakers(&mut tr, &d);
        let sum: f32 = tr.speakers.iter().filter_map(|s| s.share).sum();
        assert!((sum - 1.0).abs() < 1e-5, "shares summed to {sum}");
        assert!((tr.speakers[0].share.unwrap() - 0.25).abs() < 1e-5);
    }

    #[test]
    fn the_report_counts_contested_segments() {
        // 50/50 between two speakers: attributed, but flagged as contested.
        let d = result(&[(0, 0, 500), (1, 500, 1_000)], 2);
        let mut tr = transcript(&[(0, 1_000)]);
        let r = attribute_speakers(&mut tr, &d);
        assert_eq!(r.attributed, 1);
        assert_eq!(r.contested, 1, "a 50/50 segment must be flagged");
        assert_eq!(r.speakers_used, 1);
        assert!((r.attributed_ratio() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_clear_winner_is_not_contested() {
        let d = result(&[(0, 0, 100), (1, 100, 1_000)], 2);
        let mut tr = transcript(&[(0, 1_000)]);
        let r = attribute_speakers(&mut tr, &d);
        assert_eq!(r.contested, 0);
    }

    #[test]
    fn attribution_reports_zero_ratio_for_an_empty_transcript() {
        let mut tr = Transcript::new(EngineStamp::asr("test", "1"));
        let r = attribute_speakers(&mut tr, &result(&[(0, 0, 10)], 1));
        assert_eq!(r.attributed_ratio(), 0.0);
    }

    #[test]
    fn speakers_from_diarization_carries_talk_time_shares() {
        let d = result(&[(0, 0, 3_000), (1, 4_000, 5_000)], 2);
        let s = speakers_from_diarization(&d);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].index, 0);
        assert!((s[0].share.unwrap() - 0.75).abs() < 1e-6);
        assert!((s[1].share.unwrap() - 0.25).abs() < 1e-6);
        assert!(s.iter().all(|x| x.name.is_none()));
        assert_eq!(s[0].display_name(), "Speaker 1");
    }

    #[test]
    fn speakers_from_an_empty_result_has_no_share_and_no_panic() {
        let s = speakers_from_diarization(&DiarizationResult::default());
        assert!(s.is_empty());
    }

    #[test]
    fn a_speaker_with_no_time_gets_a_zero_share_not_none() {
        // Registered but silent: the UI needs a row for it, with a zero bar.
        let d = result(&[(0, 0, 1_000)], 3);
        let s = speakers_from_diarization(&d);
        assert_eq!(s.len(), 3);
        assert_eq!(s[2].share, Some(0.0));
    }

    #[test]
    fn apply_preserves_supplied_speaker_records() {
        let d = result(&[(0, 0, 1_000), (1, 1_000, 2_000)], 2);
        let mut tr = transcript(&[(0, 2_000)]);
        let given = vec![Speaker {
            index: 0,
            name: Some("A".into()),
            share: None,
        }];
        apply(&mut tr, &d, &given);
        assert_eq!(tr.speaker(0).unwrap().name.as_deref(), Some("A"));
        assert_eq!(tr.segments[0].speaker, Some(0));
    }

    #[test]
    fn apply_ignores_a_speaker_beyond_the_result() {
        let d = result(&[(0, 0, 1_000)], 1);
        let mut tr = transcript(&[(0, 1_000)]);
        apply(&mut tr, &d, &[Speaker::new(5)]);
        assert!(tr.speaker(5).is_none(), "a phantom speaker was registered");
    }

    #[test]
    fn attribute_segment_writes_the_vote_and_returns_the_margin() {
        let d = result(&[(0, 0, 250), (1, 250, 1_000)], 2);
        let mut seg = Segment::new(t(0), t(1_000));
        let m = attribute_segment(&mut seg, &d);
        assert_eq!(seg.speaker, Some(1));
        assert!((m - 0.5).abs() < 1e-9, "margin {m}");
    }

    #[test]
    fn consistency_check_finds_each_kind_of_broken_result() {
        let mut probs = assert_consistent(&result(&[(3, 0, 10)], 2));
        assert!(
            probs.iter().any(|p| p.contains("num_speakers is 2")),
            "{probs:?}"
        );

        probs = assert_consistent(&result(&[(0, 10, 10)], 1));
        assert!(
            probs.iter().any(|p| p.contains("empty or inverted")),
            "{probs:?}"
        );

        probs = assert_consistent(&result(&[(0, 0, 10)], 3));
        assert!(
            probs.iter().any(|p| p.contains("owns no span")),
            "{probs:?}"
        );

        assert!(assert_consistent(&result(&[(0, 0, 10), (1, 20, 30)], 2)).is_empty());
    }

    #[test]
    fn max_registered_speaker_sees_only_valid_indices() {
        assert_eq!(
            max_registered_speaker(&result(&[(0, 0, 1), (4, 1, 2)], 2)),
            Some(0)
        );
        assert_eq!(max_registered_speaker(&DiarizationResult::default()), None);
    }

    #[test]
    fn registered_speaker_count_is_just_num_speakers() {
        assert_eq!(registered_speaker_count(&result(&[(0, 0, 1)], 5)), 5);
        assert_eq!(registered_speaker_count(&DiarizationResult::default()), 0);
    }
}

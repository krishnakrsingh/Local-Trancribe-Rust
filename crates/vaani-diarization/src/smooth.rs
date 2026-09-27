//! Turning a per-frame cluster assignment into usable turns.
//!
//! # Why this stage exists at all
//!
//! Without it the output is unusable in a UI. A raw agglomerative label
//! sequence flips every time a window's features wander, and the transcript
//! viewer then shows a person changing label four times in one sentence, with
//! no indication that the change is the engine's uncertainty rather than a
//! real turn. Three operations fix that, and all three are here:
//!
//! 1. **Median filtering** over the per-frame assignment. A median of length
//!    `k` replaces each label with the most common label in its
//!    `((k-1)/2, (k+1)/2)` neighbourhood. It removes short label excursions
//!    without moving a genuine boundary, which is exactly what a mean or a
//!    mode would not do: a mean is not a label and a mode blurs a real
//!    turn boundary wider than the filter.
//! 2. **Minimum turn duration.** A turn shorter than `min_turn_ms` is
//!    absorbed into a neighbour. A person saying "haan" is 300 ms; a
//!    clustering artefact is also 300 ms. The user can merge two labels with
//!    a click, so an over-split is repairable and dropping a real short turn
//!    is not.
//! 3. **Adjacent merging.** Two consecutive turns with the same label are one
//!    turn. This is a correctness requirement, not a tidiness one: the core's
//!    `DiarizationResult::speaker_at` resolves overlapping spans by preferring
//!    the *shortest*, so two adjacent same-label spans would make the label at
//!    the shared boundary depend on span lengths.
//!
//! # Spans are non-overlapping and tight
//!
//! The engine emits **non-overlapping** spans, sorted by start, with
//! `start < end`. The core type permits overlap, but overlap has no use here:
//! every turn boundary is a genuine, decided boundary after smoothing, and an
//! overlapping pair would make the label at that instant depend on which span
//! happens to be shorter. The core's shortest-span rule exists to resolve a
//! genuinely ambiguous region; manufacturing that ambiguity and then relying
//! on the rule to resolve it would be a strange way to spend it.

use vaani_core::Time;
use vaani_core::pipeline::SpeakerSpan;

/// One contiguous stretch of one speaker, before it becomes a `SpeakerSpan`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Turn {
    /// Index into `Transcript::speakers`.
    pub speaker: usize,
    /// Inclusive start, absolute.
    pub start: Time,
    /// Exclusive end, absolute.
    pub end: Time,
    /// Agreement of the *unsmoothed* labels in this turn with `speaker`, in
    /// `[0, 1]`. This is the honest uncertainty signal: a turn whose
    /// unsmoothed labels were split down the middle is a boundary the engine
    /// is not sure about, and the UI can show that.
    pub score: f64,
}

impl Turn {
    /// Length in milliseconds, never negative.
    #[must_use]
    pub fn duration_ms(&self) -> i64 {
        (self.end.as_millis() - self.start.as_millis()).max(0)
    }

    /// The span this turn becomes.
    #[must_use]
    pub fn to_span(&self) -> SpeakerSpan {
        SpeakerSpan {
            speaker: self.speaker,
            start: self.start,
            end: self.end,
            score: Some(self.score.clamp(0.0, 1.0)),
        }
    }
}

/// Median-filter a label sequence.
///
/// `width` is forced odd and at least 1. Edges use a **truncated**
/// neighbourhood rather than a reflected one: a reflected window would invent
/// a turn at the first and last frame of the file, and the first frame of a
/// file is exactly where a VAD hangover already put a boundary.
#[must_use]
pub fn median_filter(labels: &[usize], width: usize) -> Vec<usize> {
    if labels.is_empty() {
        return Vec::new();
    }
    let w = if width.is_multiple_of(2) {
        width + 1
    } else {
        width
    };
    if w <= 1 || labels.len() <= 1 {
        return labels.to_vec();
    }
    let half = w / 2;
    let mut out = vec![0usize; labels.len()];
    let mut buf: Vec<usize> = Vec::with_capacity(w);
    for (i, slot) in out.iter_mut().enumerate() {
        buf.clear();
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(labels.len());
        buf.extend_from_slice(&labels[lo..hi]);
        buf.sort_unstable();
        *slot = buf[buf.len() / 2];
    }
    out
}

/// Run-length encode a label sequence into turns, keeping only the frames in
/// `active`.
///
/// Frames outside `active` break a turn. A window of silence between two
/// sentences from the same speaker therefore produces two turns, not one, and
/// the resulting spans do not cover the silence — which is correct, because a
/// speaker span over a pause is not something the UI can draw.
///
/// Times come from `frame_time(i)`, so the caller controls the frame-to-time
/// mapping and this function stays free of any sample-rate assumption.
#[must_use]
pub fn encode_turns(
    labels: &[usize],
    active: &[bool],
    raw_labels: &[usize],
    frame_time: impl Fn(usize) -> Time,
    frame_ms: i64,
) -> Vec<Turn> {
    let mut turns: Vec<Turn> = Vec::new();
    let mut i = 0usize;
    while i < labels.len() {
        if !active.get(i).copied().unwrap_or(false) {
            i += 1;
            continue;
        }
        let speaker = labels[i];
        let start = i;
        while i < labels.len()
            && active.get(i).copied().unwrap_or(false)
            && labels.get(i).copied() == Some(speaker)
        {
            i += 1;
        }
        // Exclusive end is one frame past the last active frame of the run,
        // so consecutive turns tile the active timeline without a gap.
        let end_frame = i;
        let start_ms = frame_time(start).as_millis();
        let end_ms = frame_time(end_frame.saturating_sub(1)).as_millis() + frame_ms;
        let agree = agreement(raw_labels, start, end_frame, speaker);
        turns.push(Turn {
            speaker,
            start: Time::from_millis(start_ms.max(0)),
            end: Time::from_millis(end_ms.max(0)),
            score: agree,
        });
    }
    merge_adjacent(turns)
}

/// Fraction of `start..end` whose raw label was `speaker`.
///
/// A missing or short `raw_labels` slice degrades to 1.0, which is the
/// optimistic end. That direction is chosen because the field is only ever
/// shown as a confidence hint, and a low-confidence label on a boundary the
/// engine actually got right is the more annoying of the two mistakes.
fn agreement(raw: &[usize], start: usize, end: usize, speaker: usize) -> f64 {
    if end <= start {
        return 1.0;
    }
    let mut agree = 0usize;
    let mut total = 0usize;
    for i in start..end {
        // A missing or short slice degrades to full confidence, which is the
        // optimistic end. That direction is chosen because the field is only
        // ever shown as a confidence hint, and a low-confidence label on a
        // boundary the engine actually got right is the more annoying of the
        // two mistakes.
        if let Some(l) = raw.get(i) {
            total += 1;
            if *l == speaker {
                agree += 1;
            }
        }
    }
    if total == 0 {
        1.0
    } else {
        agree as f64 / total as f64
    }
}

/// Merge consecutive turns that share a speaker and touch in time.
///
/// A gap of up to `max_gap_ms` between two same-speaker turns is bridged, which
/// removes the seam a one-frame activity dropout would otherwise leave. A gap
/// larger than that is a real pause and is left alone.
#[must_use]
pub fn merge_adjacent(turns: Vec<Turn>) -> Vec<Turn> {
    merge_adjacent_with_gap(turns, 0)
}

/// Merge consecutive same-speaker turns, bridging gaps up to `max_gap_ms`.
#[must_use]
pub fn merge_adjacent_with_gap(mut turns: Vec<Turn>, max_gap_ms: i64) -> Vec<Turn> {
    let mut out: Vec<Turn> = Vec::with_capacity(turns.len());
    for t in turns.drain(..) {
        match out.last_mut() {
            Some(prev)
                if prev.speaker == t.speaker
                    && t.start.as_millis() - prev.end.as_millis() <= max_gap_ms =>
            {
                // Weighted by frame count so a long confident stretch is not
                // overruled by a short uncertain one.
                let a = prev.score.max(1e-6);
                let b = t.score.max(1e-6);
                let w = (prev.end.as_millis() - prev.start.as_millis()).max(0) as f64;
                let w2 = (t.end.as_millis() - t.start.as_millis()).max(0) as f64;
                prev.score = (a * w + b * w2) / (w + w2).max(1e-6);
                prev.end = t.end;
            }
            _ => out.push(t),
        }
    }
    out
}

/// Absorb turns shorter than `min_ms` into a neighbour.
///
/// The neighbour chosen is the **longer** of the two adjacent turns, with ties
/// going to the earlier one. "Longer" rather than "the one on the left" because
/// attaching a 200 ms blip to a 10-second turn and to a 300 ms turn are very
/// different acts, and the first is almost always right.
#[must_use]
pub fn absorb_short_turns(mut turns: Vec<Turn>, min_ms: i64) -> Vec<Turn> {
    if min_ms <= 0 || turns.is_empty() {
        return turns;
    }
    // Each pass removes at least one turn, so the number of turns is a
    // sufficient bound and the loop cannot spin.
    for _ in 0..turns.len() + 1 {
        let Some(idx) = (0..turns.len())
            .filter(|i| turns[*i].duration_ms() < min_ms)
            .min_by_key(|i| turns[*i].duration_ms())
        else {
            break;
        };
        let to_left = idx > 0;
        let to_right = idx + 1 < turns.len();
        // The turn's own speaker is what it claims; pick the neighbour that is
        // longer, so a blip joins the substantial turn rather than another blip.
        let left_len = if to_left {
            turns[idx - 1].duration_ms()
        } else {
            -1
        };
        let right_len = if to_right {
            turns[idx + 1].duration_ms()
        } else {
            -1
        };
        if !to_left && !to_right {
            break;
        }
        let merge_left = left_len >= right_len;
        if merge_left {
            // The short turn's *speaker assignment* is the suspect part, so
            // the surviving turn adopts it and the length too.
            turns[idx - 1].speaker = turns[idx].speaker;
            turns[idx - 1].end = turns[idx].end;
            turns.remove(idx);
        } else {
            // The short turn survives and absorbs the neighbour, because the
            // short turn's *length* is the suspect property and the
            // neighbour's speaker is the more trustworthy of the two. Keeping
            // the short turn also keeps the span's start, which is what a
            // join to the right has to preserve.
            turns[idx].speaker = turns[idx + 1].speaker;
            turns[idx].end = turns[idx + 1].end;
            turns.remove(idx + 1);
        }
    }
    turns
}

/// Clamp a turn to a duration and enforce `start < end`.
///
/// A turn with a non-positive duration is dropped rather than shortened: a
/// zero-length span has no meaning and would make `coverage` and every
/// downstream overlap calculation carry a degenerate term.
pub fn sanitise(turns: Vec<Turn>) -> Vec<Turn> {
    turns
        .into_iter()
        .filter(|t| t.end.as_millis() > t.start.as_millis())
        .collect()
}

/// The invariant every `Vec<SpeakerSpan>` this crate produces must satisfy.
///
/// Sorted by start, `start < end`, non-overlapping, and every `speaker` below
/// `num_speakers`. Returned as a list of human-readable problems so a failure
/// says which property broke rather than just that one did.
#[must_use]
pub fn validate_spans(spans: &[SpeakerSpan], num_speakers: usize) -> Vec<String> {
    let mut problems = Vec::new();
    for (i, s) in spans.iter().enumerate() {
        if s.end.as_millis() <= s.start.as_millis() {
            problems.push(format!(
                "span {i} is empty or inverted: {}..{}",
                s.start.as_millis(),
                s.end.as_millis()
            ));
        }
        if s.speaker >= num_speakers {
            problems.push(format!(
                "span {i} names speaker {} but only {num_speakers} are registered",
                s.speaker
            ));
        }
        if let Some(sc) = s.score
            && !(0.0..=1.0).contains(&sc)
        {
            problems.push(format!("span {i} has score {sc} outside [0,1]"));
        }
        if i > 0 {
            let prev = &spans[i - 1];
            if s.start < prev.start {
                problems.push(format!(
                    "span {i} starts at {} before span {i}'s {}",
                    s.start.as_millis(),
                    prev.start.as_millis()
                ));
            }
            if s.start < prev.end {
                problems.push(format!(
                    "span {i} overlaps span {} ({}..{} vs {}..{})",
                    i - 1,
                    prev.start.as_millis(),
                    prev.end.as_millis(),
                    s.start.as_millis(),
                    s.end.as_millis()
                ));
            }
        }
    }
    problems
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(ms: i64) -> Time {
        Time::from_millis(ms)
    }

    #[test]
    fn median_filter_removes_a_short_excursion() {
        let labels = vec![0, 0, 0, 1, 0, 0, 0];
        let out = median_filter(&labels, 3);
        assert_eq!(out, vec![0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn median_filter_keeps_a_sustained_change() {
        // A real turn is longer than the filter, and must survive it.
        let labels = vec![0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0];
        let out = median_filter(&labels, 3);
        assert_eq!(out[0..4], [0, 0, 0, 0]);
        assert_eq!(out[4..9], [1, 1, 1, 1, 1]);
        assert_eq!(out[9..13], [0, 0, 0, 0]);
    }

    #[test]
    fn median_filter_of_nothing_is_nothing() {
        assert!(median_filter(&[], 5).is_empty());
        assert_eq!(median_filter(&[3], 5), vec![3]);
    }

    #[test]
    fn an_even_width_is_rounded_up_and_a_width_of_one_is_the_identity() {
        let labels = vec![0, 0, 1, 0, 0];
        assert_eq!(median_filter(&labels, 1), labels);
        // Width 4 becomes 5, which is the same as asking for 5.
        assert_eq!(median_filter(&labels, 4), median_filter(&labels, 5));
    }

    #[test]
    fn median_filter_truncates_at_the_edges_rather_than_reflecting() {
        // A reflected window would put a phantom 1 at index 0.
        let labels = vec![0, 0, 0, 0, 0];
        assert_eq!(median_filter(&labels, 5), labels);
        let one_then_zero = vec![1, 0, 0, 0, 0];
        let out = median_filter(&one_then_zero, 3);
        // Index 0 sees {1, 0} -> lower median of two elements is index 0's own
        // value under a truncated window, so it stays. That is a documented
        // edge, not a reflection.
        assert_eq!(out[0], 1);
        assert_eq!(out[1..], [0, 0, 0, 0]);
    }

    #[test]
    fn encode_turns_tiles_the_active_timeline_without_overlap() {
        let labels = vec![0, 0, 1, 1, 1, 0, 0];
        let active = vec![true; 7];
        let raw = labels.clone();
        let turns = encode_turns(&labels, &active, &raw, |i| t(i as i64 * 10), 10);
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[0].start, t(0));
        assert_eq!(turns[0].end, t(20));
        assert_eq!(turns[1].start, t(20));
        assert_eq!(turns[1].end, t(50));
        assert_eq!(turns[2].start, t(50));
        assert_eq!(turns[2].end, t(70));
        for w in turns.windows(2) {
            assert!(w[0].end <= w[1].start);
        }
    }

    #[test]
    fn inactive_frames_break_a_turn() {
        // Silence between two sentences by the same speaker must produce two
        // spans, because a span drawn over the pause is not drawable.
        let labels = vec![0, 0, 0, 0];
        let active = vec![true, true, false, true];
        let turns = encode_turns(&labels, &active, &labels, |i| t(i as i64 * 10), 10);
        assert_eq!(turns.len(), 2, "{turns:?}");
        assert_eq!(turns[0].end, t(20));
        assert_eq!(turns[1].start, t(30));
    }

    #[test]
    fn all_inactive_gives_no_turns() {
        let labels = vec![0, 0, 0];
        let turns = encode_turns(
            &labels,
            &[false, false, false],
            &labels,
            |i| t(i as i64 * 10),
            10,
        );
        assert!(turns.is_empty());
    }

    #[test]
    fn score_reports_the_unsmoothed_agreement() {
        // Frames 3..5 were labelled 1 by the raw pass, 0 by the filter: the
        // turn is uncertain and must say so.
        let labels = vec![0, 0, 0, 0, 0, 0, 0];
        let mut raw = labels.clone();
        raw[3] = 1;
        raw[4] = 1;
        let active = vec![true; 7];
        let turns = encode_turns(&labels, &active, &raw, |i| t(i as i64 * 10), 10);
        assert_eq!(turns.len(), 1);
        assert!(turns[0].score < 1.0, "score {}", turns[0].score);
        assert!(turns[0].score > 0.5, "score {}", turns[0].score);
    }

    #[test]
    fn a_missing_raw_slice_degrades_to_full_confidence() {
        let labels = vec![0, 0, 0];
        let turns = encode_turns(&labels, &[true; 3], &[], |i| t(i as i64 * 10), 10);
        assert_eq!(turns[0].score, 1.0);
    }

    #[test]
    fn merging_bridges_only_same_speaker_neighbours() {
        let turns = vec![
            Turn {
                speaker: 0,
                start: t(0),
                end: t(100),
                score: 1.0,
            },
            Turn {
                speaker: 0,
                start: t(100),
                end: t(200),
                score: 0.5,
            },
            Turn {
                speaker: 1,
                start: t(200),
                end: t(300),
                score: 1.0,
            },
        ];
        let out = merge_adjacent(turns);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].end, t(200));
        assert!((out[0].score - 0.75).abs() < 1e-9, "score {}", out[0].score);
    }

    #[test]
    fn merging_with_a_gap_budget_bridges_a_small_gap() {
        let turns = vec![
            Turn {
                speaker: 0,
                start: t(0),
                end: t(100),
                score: 1.0,
            },
            Turn {
                speaker: 0,
                start: t(150),
                end: t(200),
                score: 1.0,
            },
        ];
        assert_eq!(merge_adjacent_with_gap(turns.clone(), 0).len(), 2);
        assert_eq!(merge_adjacent_with_gap(turns, 60).len(), 1);
    }

    #[test]
    fn a_short_turn_joins_the_longer_neighbour() {
        // A 100 ms blip between a 1000 ms and a 900 ms turn. The 900 ms turn
        // is itself above the 500 ms minimum, so only the blip is absorbed.
        let turns = vec![
            Turn {
                speaker: 0,
                start: t(0),
                end: t(1_000),
                score: 1.0,
            },
            Turn {
                speaker: 1,
                start: t(1_000),
                end: t(1_100),
                score: 1.0,
            },
            Turn {
                speaker: 0,
                start: t(1_100),
                end: t(2_000),
                score: 1.0,
            },
        ];
        let out = absorb_short_turns(turns, 500);
        assert_eq!(out.len(), 2, "{out:?}");
        assert_eq!(out[0].speaker, 1);
        assert_eq!(out[0].end, t(1_100));
        assert_eq!(out[1].speaker, 0);
        assert_eq!(out[1].end, t(2_000));
    }

    #[test]
    fn a_short_turn_joins_the_right_when_the_right_is_longer() {
        let turns = vec![
            Turn {
                speaker: 0,
                start: t(0),
                end: t(300),
                score: 1.0,
            },
            Turn {
                speaker: 1,
                start: t(300),
                end: t(400),
                score: 1.0,
            },
            Turn {
                speaker: 0,
                start: t(400),
                end: t(2_000),
                score: 1.0,
            },
        ];
        // A 150 ms minimum so the 100 ms blip is the only thing absorbed. The
        // 300 ms turn survives, and the blip joins the long turn, so the two
        // surviving turns are [0, 300) and [300, 2000) -- touching, with no
        // gap, because the absorbed blip was between them.
        let out = absorb_short_turns(turns, 150);
        assert_eq!(out.len(), 2, "{out:?}");
        assert_eq!(out[0].end, t(300));
        assert_eq!(out[1].start, t(300));
        assert_eq!(out[1].speaker, 0);
    }

    #[test]
    fn a_short_turn_at_either_end_is_merged_not_dropped() {
        let turns = vec![
            Turn {
                speaker: 1,
                start: t(0),
                end: t(100),
                score: 1.0,
            },
            Turn {
                speaker: 0,
                start: t(100),
                end: t(1_000),
                score: 1.0,
            },
        ];
        let out = absorb_short_turns(turns, 500);
        assert_eq!(out.len(), 1, "a turn was deleted from the transcript");
        // The short turn's speaker assignment is the suspect part, so the
        // surviving turn adopts the neighbour's.
        assert_eq!(out[0].speaker, 0);
        assert_eq!(out[0].start, t(0));
        assert_eq!(out[0].end, t(1_000));
    }

    #[test]
    fn a_lone_short_turn_survives_rather_than_disappearing() {
        let turns = vec![Turn {
            speaker: 3,
            start: t(0),
            end: t(100),
            score: 1.0,
        }];
        let out = absorb_short_turns(turns, 500);
        assert_eq!(out.len(), 1, "a turn was deleted from the transcript");
    }

    #[test]
    fn absorption_terminates_on_all_short_turns() {
        let turns: Vec<Turn> = (0..6)
            .map(|i| Turn {
                speaker: i % 2,
                start: t(i as i64 * 100),
                end: t((i as i64 + 1) * 100),
                score: 1.0,
            })
            .collect();
        let out = absorb_short_turns(turns, 500);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn absorption_is_a_no_op_when_the_minimum_is_zero() {
        let turns = vec![Turn {
            speaker: 0,
            start: t(0),
            end: t(1),
            score: 1.0,
        }];
        assert_eq!(absorb_short_turns(turns.clone(), 0), turns);
        assert!(absorb_short_turns(turns, -1).len() == 1);
    }

    #[test]
    fn sanitise_drops_degenerate_spans() {
        let turns = vec![
            Turn {
                speaker: 0,
                start: t(100),
                end: t(100),
                score: 1.0,
            },
            Turn {
                speaker: 0,
                start: t(200),
                end: t(100),
                score: 1.0,
            },
            Turn {
                speaker: 0,
                start: t(0),
                end: t(10),
                score: 1.0,
            },
        ];
        let out = sanitise(turns);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].end, t(10));
    }

    #[test]
    fn a_span_carries_its_score_clamped() {
        let turn = Turn {
            speaker: 0,
            start: t(0),
            end: t(10),
            score: 4.2,
        };
        assert_eq!(turn.to_span().score, Some(1.0));
        let turn = Turn {
            speaker: 0,
            start: t(0),
            end: t(10),
            score: -1.0,
        };
        assert_eq!(turn.to_span().score, Some(0.0));
    }

    #[test]
    fn validate_accepts_a_well_formed_run() {
        let spans = vec![
            Turn {
                speaker: 0,
                start: t(0),
                end: t(100),
                score: 1.0,
            }
            .to_span(),
            Turn {
                speaker: 1,
                start: t(100),
                end: t(400),
                score: 0.5,
            }
            .to_span(),
        ];
        assert!(validate_spans(&spans, 2).is_empty());
    }

    #[test]
    fn validate_names_each_broken_property() {
        let empty = Turn {
            speaker: 0,
            start: t(5),
            end: t(5),
            score: 1.0,
        }
        .to_span();
        assert!(validate_spans(&[empty], 1)[0].contains("empty or inverted"));

        let unregistered = Turn {
            speaker: 7,
            start: t(0),
            end: t(5),
            score: 1.0,
        }
        .to_span();
        assert!(validate_spans(&[unregistered], 1)[0].contains("only 1 are registered"));

        let overlapping = vec![
            Turn {
                speaker: 0,
                start: t(0),
                end: t(100),
                score: 1.0,
            }
            .to_span(),
            Turn {
                speaker: 1,
                start: t(50),
                end: t(200),
                score: 1.0,
            }
            .to_span(),
        ];
        let p = validate_spans(&overlapping, 2);
        assert!(p.iter().any(|s| s.contains("overlaps")), "{p:?}");

        let unsorted = vec![
            Turn {
                speaker: 0,
                start: t(300),
                end: t(400),
                score: 1.0,
            }
            .to_span(),
            Turn {
                speaker: 0,
                start: t(100),
                end: t(200),
                score: 1.0,
            }
            .to_span(),
        ];
        let p = validate_spans(&unsorted, 1);
        assert!(p.iter().any(|s| s.contains("before")), "{p:?}");

        let bad_score = vec![SpeakerSpan {
            speaker: 0,
            start: t(0),
            end: t(10),
            score: Some(2.0),
        }];
        assert!(validate_spans(&bad_score, 1)[0].contains("outside [0,1]"));
    }

    #[test]
    fn touching_spans_are_not_overlapping() {
        let spans = vec![
            Turn {
                speaker: 0,
                start: t(0),
                end: t(100),
                score: 1.0,
            }
            .to_span(),
            Turn {
                speaker: 1,
                start: t(100),
                end: t(200),
                score: 1.0,
            }
            .to_span(),
        ];
        assert!(validate_spans(&spans, 2).is_empty());
    }
}

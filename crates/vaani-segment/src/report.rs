//! What grouping did, in numbers.
//!
//! Every count here is an aggregate: a caller can put the whole report in a log
//! line, a settings panel or a benchmark file without leaking a single word of
//! the recording. That is a requirement, not a convenience — the transcripts
//! this crate processes are real private meetings, and a diagnostic that quotes
//! one is a privacy incident.
//!
//! The report is also the *evidence* for the policy. "Is 500 ms the right pause?"
//! is not answerable by argument; it is answerable by
//! [`breaks_pause`](Self::breaks_pause) and
//! [`segments_out`](Self::segments_out) on a real recording.

use serde::{Deserialize, Serialize};

/// Why a segment ended.
///
/// Public because a settings panel showing "this threshold produced 214
/// segments, 180 of them at a pause and 34 at a speaker turn" is the only way a
/// user can tell whether a number they are about to change means anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[allow(missing_docs)] // each variant is documented on the variant
pub enum BreakReason {
    /// The silence before the word was at least `max_gap_ms`.
    HardGap,
    /// The silence before the word was at least `pause_break_ms`.
    Pause,
    /// The word's speaker differs from the previous word's.
    SpeakerTurn,
    /// Adding the word would exceed `max_words`.
    WordCap,
    /// Adding the word would exceed `max_chars`.
    CharCap,
    /// Adding the word would exceed `max_duration_ms`.
    DurationCap,
}

impl BreakReason {
    /// Every reason, in the order the rules are evaluated. A settings panel
    /// renders them in this order so the list matches the documented rules.
    pub const ALL: [BreakReason; 6] = [
        BreakReason::HardGap,
        BreakReason::Pause,
        BreakReason::SpeakerTurn,
        BreakReason::WordCap,
        BreakReason::CharCap,
        BreakReason::DurationCap,
    ];

    /// A stable identifier for reports and settings files.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            BreakReason::HardGap => "hard_gap",
            BreakReason::Pause => "pause",
            BreakReason::SpeakerTurn => "speaker_turn",
            BreakReason::WordCap => "word_cap",
            BreakReason::CharCap => "char_cap",
            BreakReason::DurationCap => "duration_cap",
        }
    }
}

/// Counts of every break reason, every repair, and the before/after totals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
#[allow(missing_docs)] // each field is documented on the field
pub struct GroupingReport {
    /// Words in the input. Always equal to [`words_out`](Self::words_out): the
    /// whole point of the crate, and the one number that must never differ.
    pub words_in: usize,
    /// Words in the output. Equal to `words_in` by construction.
    pub words_out: usize,
    /// Segments in the input.
    pub segments_in: usize,
    /// Segments in the output.
    pub segments_out: usize,

    /// Source segments that carried no words. They cannot become output
    /// segments — a segment with no words is a blank screen for its duration —
    /// so they are counted rather than silently dropped.
    pub source_segments_without_words: usize,
    /// How many source segments were absorbed into other segments. This is
    /// `segments_in - segments_out` for a transcript whose every segment had at
    /// least one word.
    pub source_segments_merged: usize,

    /// Words the stable sort actually moved. Zero on any transcript whose words
    /// were already in time order, which is every transcript this pipeline
    /// produces; a non-zero value on real output means the input was not sorted
    /// and the report is telling the truth about it.
    pub reordered_words: usize,

    /// Breaks caused by a gap of at least `max_gap_ms`.
    pub breaks_hard_gap: usize,
    /// Breaks caused by a gap of at least `pause_break_ms`.
    pub breaks_pause: usize,
    /// Breaks caused by the speaker changing.
    pub breaks_speaker_turn: usize,
    /// Breaks caused by `max_words`.
    pub breaks_word_cap: usize,
    /// Breaks caused by `max_chars`.
    pub breaks_char_cap: usize,
    /// Breaks caused by `max_duration_ms`.
    pub breaks_duration_cap: usize,

    /// Breaks the policy asked for that were **moved back** to keep an
    /// overlapping word inside the segment that owns it. Non-zero only for
    /// input with overlapping words, which chunk overlap produces.
    pub breaks_moved_for_overlap: usize,
    /// Breaks the policy asked for that were **dropped entirely** because the
    /// words on either side overlap and no position between them can be both
    /// disjoint and non-empty. Those words have to share a segment.
    ///
    /// Together with [`breaks_moved_for_overlap`](Self::breaks_moved_for_overlap)
    /// this is the honest cost of the "no word outside its own segment" rule,
    /// and the reason `total_breaks()` still equals `segments_out - 1`: a break
    /// that did not happen is not counted as one.
    pub breaks_dropped_for_overlap: usize,

    /// Cap breaks the runt guard declined to honour because a natural boundary
    /// was closer than `min_segment_words`. Each one produced a group longer
    /// than the cap that asked for the break, and the break happened at the
    /// natural boundary instead.
    pub cap_breaks_deferred: usize,
    /// Finished groups that violate an enabled cap because of a deferral. The
    /// honest cost of the runt guard: a non-zero value here with a non-zero
    /// `cap_breaks_deferred` is the guard working, and a non-zero value here
    /// with a *zero* deferral count is a bug in `cap_break`.
    pub segments_over_cap: usize,

    /// Segments whose start was pushed forward so they would not overlap the
    /// previous one. Zero on well-formed input.
    pub segments_pushed_for_overlap: usize,
    /// Segments whose span was clamped into the media duration, because
    /// `push_chunk` has no upper clamp and a word can land past the end of the
    /// recording.
    pub segments_clamped_to_media: usize,
    /// Segments whose span was extended to [`MIN_SEGMENT_MS`].
    /// [`crate::grouping::MIN_SEGMENT_MS`]. Non-zero only for degenerate input.
    pub segments_extended_to_minimum: usize,

    /// Groups that ended up with a speaker.
    pub attributed: usize,
    /// Groups left as `None` — the complete ASR-only state, not a failure.
    pub unattributed: usize,
}

impl GroupingReport {
    /// Words per output segment, or 0 for an empty transcript.
    #[must_use]
    pub fn words_per_segment(&self) -> f64 {
        if self.segments_out == 0 {
            0.0
        } else {
            self.words_out as f64 / self.segments_out as f64
        }
    }

    /// Segments removed per segment added: `segments_in / segments_out`, or 0
    /// when there is no output. The headline "1.00 words per segment became
    /// 7.4" number.
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        if self.segments_out == 0 {
            0.0
        } else {
            self.segments_in as f64 / self.segments_out as f64
        }
    }

    /// Fraction of groups attributed to a speaker, in `[0, 1]`.
    #[must_use]
    pub fn attributed_ratio(&self) -> f64 {
        let total = self.attributed + self.unattributed;
        if total == 0 {
            0.0
        } else {
            self.attributed as f64 / total as f64
        }
    }

    /// Total breaks, which is `segments_out - 1` unless the transcript was
    /// empty.
    #[must_use]
    pub fn total_breaks(&self) -> usize {
        self.breaks_hard_gap
            + self.breaks_pause
            + self.breaks_speaker_turn
            + self.breaks_word_cap
            + self.breaks_char_cap
            + self.breaks_duration_cap
    }

    /// Breaks by reason, in the order the rules are evaluated, as a fixed-size
    /// array. The `breaks_*` fields are the flat form; both exist because a
    /// struct of six named fields serialises into a settings-friendly report
    /// while an array iterates in the documented order without a `Vec`
    /// allocation on a per-job path.
    #[must_use]
    pub fn breaks(&self) -> [(BreakReason, usize); 6] {
        [
            (BreakReason::HardGap, self.breaks_hard_gap),
            (BreakReason::Pause, self.breaks_pause),
            (BreakReason::SpeakerTurn, self.breaks_speaker_turn),
            (BreakReason::WordCap, self.breaks_word_cap),
            (BreakReason::CharCap, self.breaks_char_cap),
            (BreakReason::DurationCap, self.breaks_duration_cap),
        ]
    }

    /// Record one break that was actually taken.
    pub(crate) fn record_break(&mut self, reason: BreakReason) {
        match reason {
            BreakReason::HardGap => self.breaks_hard_gap += 1,
            BreakReason::Pause => self.breaks_pause += 1,
            BreakReason::SpeakerTurn => self.breaks_speaker_turn += 1,
            BreakReason::WordCap => self.breaks_word_cap += 1,
            BreakReason::CharCap => self.breaks_char_cap += 1,
            BreakReason::DurationCap => self.breaks_duration_cap += 1,
        }
    }

    /// Record one cap break the runt guard declined. The break was *not* taken
    /// here, so it is deliberately not counted in `breaks_*`: the number of
    /// breaks and the number of groups must agree.
    pub(crate) fn deferred_by_reason(&mut self, reason: BreakReason) {
        debug_assert!(
            matches!(
                reason,
                BreakReason::WordCap | BreakReason::CharCap | BreakReason::DurationCap
            ),
            "only a cap can be deferred; a natural break is not a choice"
        );
        self.cap_breaks_deferred += 1;
    }

    /// Whether anything about the input was not in the shape the pipeline is
    /// supposed to produce: words out of order, a segment with no words, an
    /// overlap, a span past the media, a zero-length span.
    ///
    /// A clean run on a clean transcript reports `true`. A clean run on a real
    /// recording that reports `false` has found something, and the individual
    /// counters say what. Overlapping words do **not** clear this: they are
    /// handled without losing a word or a time, and
    /// [`breaks_moved_for_overlap`](Self::breaks_moved_for_overlap) is where
    /// that shows up.
    #[must_use]
    pub fn input_was_well_formed(&self) -> bool {
        self.reordered_words == 0
            && self.source_segments_without_words == 0
            && self.segments_pushed_for_overlap == 0
            && self.segments_clamped_to_media == 0
            && self.segments_extended_to_minimum == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratios_are_guarded_against_an_empty_transcript() {
        let r = GroupingReport::default();
        assert_eq!(r.words_per_segment(), 0.0);
        assert_eq!(r.compression_ratio(), 0.0);
        assert_eq!(r.attributed_ratio(), 0.0);
        assert_eq!(r.total_breaks(), 0);
    }

    #[test]
    fn the_ratios_are_the_documented_arithmetic() {
        let r = GroupingReport {
            words_in: 1913,
            words_out: 1913,
            segments_in: 1913,
            segments_out: 250,
            attributed: 240,
            unattributed: 10,
            breaks_pause: 200,
            breaks_hard_gap: 10,
            breaks_speaker_turn: 39,
            ..GroupingReport::default()
        };
        assert!((r.words_per_segment() - 7.652).abs() < 1e-3);
        assert!((r.compression_ratio() - 7.652).abs() < 1e-3);
        assert!((r.attributed_ratio() - 0.96).abs() < 1e-9);
        assert_eq!(r.total_breaks(), 249);
    }

    #[test]
    fn a_clean_run_on_clean_input_is_well_formed() {
        let r = GroupingReport {
            words_in: 10,
            words_out: 10,
            segments_in: 10,
            segments_out: 2,
            breaks_pause: 1,
            attributed: 2,
            ..GroupingReport::default()
        };
        assert!(r.input_was_well_formed());
    }

    #[test]
    fn every_repair_flag_makes_the_input_not_well_formed() {
        let base = GroupingReport {
            words_in: 10,
            words_out: 10,
            segments_in: 10,
            segments_out: 2,
            breaks_pause: 1,
            attributed: 2,
            ..GroupingReport::default()
        };
        for broken in [
            GroupingReport {
                reordered_words: 1,
                ..base
            },
            GroupingReport {
                source_segments_without_words: 1,
                ..base
            },
            GroupingReport {
                segments_pushed_for_overlap: 1,
                ..base
            },
            GroupingReport {
                segments_clamped_to_media: 1,
                ..base
            },
            GroupingReport {
                segments_extended_to_minimum: 1,
                ..base
            },
        ] {
            assert!(!broken.input_was_well_formed(), "{broken:?}");
        }
    }
}

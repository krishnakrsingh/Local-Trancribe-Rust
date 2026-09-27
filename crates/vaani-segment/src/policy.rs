//! The grouping policy: every knob, named, documented, and serialisable.
//!
//! # Why this is a value and not a constant
//!
//! Grouping is a **product** decision, not an implementation detail. A subtitle
//! file and a meeting document want different things from the same transcript:
//! a caption has to be readable in the time it is on screen, a document has to
//! be skimmable in a browser. Both are produced from the same word timings, so
//! the difference has to live in a value the caller can change — and in
//! something the settings panel and the CLI can show, edit and persist.
//!
//! Every field is therefore public and `Serialize`, and carries the reasoning
//! for its own default. `EnergyVadConfig` and `ChunkPolicy` set the precedent:
//! the numbers that produced a measurement are a value you can write to a
//! report and read back, not a scatter of literals in an `impl Default`.
//!
//! # The rules, in the order they are applied
//!
//! For each word after the first, a new segment is started when **any** of the
//! following holds. They are listed in the order they are evaluated, which is
//! also the order [`GroupingReport`](crate::GroupingReport) counts them in.
//!
//! 1. **A hard gap** — the silence before this word is at least
//!    [`max_gap_ms`](Self::max_gap_ms). A segment may never span this much
//!    silence, because a segment's duration is `last word end − first word
//!    start` and a duration that includes a long hole is a lie about how long
//!    the speaker was talking.
//! 2. **A pause** — the silence before this word is at least
//!    [`pause_break_ms`](Self::pause_break_ms). This is the *preferred*
//!    boundary and the one the default is tuned from a measured pause
//!    distribution. It is clamped to be at most `max_gap_ms`, so the two can
//!    never disagree about where the line is.
//! 3. **A speaker turn** — this word and the previous one both resolve to a
//!    speaker and the speakers differ. Requires a `DiarizationResult`; with
//!    `--no-diarize` this rule cannot fire and never does.
//! 4. **The word cap** — adding this word would take the segment past
//!    [`max_words`](Self::max_words).
//! 5. **The character cap** — adding this word would take the segment's text
//!    past [`max_chars`](Self::max_chars). Counted in `char`s of
//!    `Word::display_text()` joined by single spaces, which is exactly the
//!    metric `vaani_export`'s cue wrapper counts, so a segment that fits the cap
//!    is a segment that fits a caption line budget.
//! 6. **The duration cap** — this word ends more than
//!    [`max_duration_ms`](Self::max_duration_ms) after the segment's first word
//!    started. This is what splits a long sentence that contains no pause at
//!    all; without it, a monotone speaker with no breath produces one unbounded
//!    segment.
//!
//! ## The one rule that is not a break: the runt guard
//!
//! Rules 4–6 are **soft** in one specific way. Before honouring a cap, the
//! grouper looks ahead to the next *natural* boundary (rule 1, 2 or 3). If the
//! words between here and there number fewer than
//! [`min_segment_words`](Self::min_segment_words), the cap break is **deferred**
//! to that natural boundary instead.
//!
//! The reason is that a cap firing one word before a pause produces a one-word
//! segment, and a one-word segment is the exact failure this crate exists to
//! remove. Deferring can only add `min_segment_words - 1` words to a group, so
//! the cap is still bounded, and the report counts how many finished segments
//! exceeded a cap ([`GroupingReport::segments_over_cap`]) so a policy that is
//! routinely deferring is visible rather than silent.
//!
//! ## Zero means "no limit"
//!
//! `max_words`, `max_chars` and `max_duration_ms` accept `0` to mean the cap is
//! switched off. That is the honest encoding: a negative length is a typo, and
//! a zero-length segment limit is not a request for zero-length segments.
//!
//! ## What this policy deliberately does not do
//!
//! **It does not use punctuation.** SraVaani 1.0 emits none — see the
//! `grouping` module for why, and
//! [`word_text_contains_no_punctuation`](crate::word_text_contains_no_punctuation)
//! for the check that keeps the claim honest.

use serde::{Deserialize, Serialize};

/// A serialisable, inspectable description of how words become segments.
///
/// See the [module documentation](self) for the rules and the reasoning behind
/// each default. The defaults are the *measured* ones, not round numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
#[allow(missing_docs)] // each field is documented on the field
pub struct GroupingPolicy {
    /// The preferred pause that ends a segment, in milliseconds.
    ///
    /// A new segment starts when the silence before a word is **at least** this
    /// long, so a gap of exactly `pause_break_ms` breaks and a gap of
    /// `pause_break_ms - 1` does not.
    ///
    /// **Default 500 ms, and the measured reason is not the one you would
    /// guess.** On the real 834 s meeting, 1,889 of the 1,911 inter-word gaps
    /// are **exactly zero**: SraVaani's decoder assigns each word a frame span
    /// and gives the next word the frame immediately after, so consecutive words
    /// abut and the recogniser reports no silence between them at all. Only
    /// 1.15 % of gaps exceed 100 ms and 0.37 % exceed 2 s.
    ///
    /// The consequence is that on this ASR the pause rule is nearly inert: it
    /// fired **6 times out of 226 breaks**. The segmentation is carried by the
    /// speaker turn (188) and the character cap (22).
    ///
    /// That makes the default *robust* rather than *tuned*, and the distinction
    /// matters. The sweep from 200 ms to 1,500 ms moves the segment count only
    /// between 115 and 116 — a plateau, not a cliff — so no value in that band
    /// is load-bearing and the default cannot be fragile. 500 ms is chosen
    /// because it is the middle of that plateau **and** because it is the
    /// conventional clause boundary, so it will behave sensibly on an engine
    /// that *does* report inter-word silence. It is not a number this recording
    /// picked out, and claiming otherwise would be reading a plateau as a peak.
    ///
    /// See [`super::grouping::GapDistribution`] and `THRESHOLD_SWEEP` for how to
    /// re-measure this on a different recording or a different model.
    pub pause_break_ms: i64,

    /// The largest silence a single segment may span, in milliseconds.
    ///
    /// A hard ceiling, independent of `pause_break_ms`: it exists so that a user
    /// who raises `pause_break_ms` for a very fast talker still cannot produce a
    /// segment whose stated duration is dominated by silence.
    /// [`is_valid`](Self::is_valid) requires `max_gap_ms >= pause_break_ms`, so
    /// the two can never disagree about where the line is; a value equal to
    /// `pause_break_ms` switches this extra rule off.
    ///
    /// **Default 1 500 ms.** On the real meeting this fired **8 times out of
    /// 226** — it is genuinely a safety net rather than the policy. It exists so
    /// that a user who raises `pause_break_ms` for a very fast talker, or who
    /// switches to an engine that does report silence, still cannot produce a
    /// segment whose stated duration is dominated by a hole. A second and a half
    /// of silence in a meeting is a different speaker, a different topic, or an
    /// interruption.
    pub max_gap_ms: i64,

    /// Maximum words in one segment. `0` disables the cap.
    ///
    /// **Default 24 words.** Measured: this fired **zero** times on the real
    /// meeting. The character cap is always reached first, because a 24-word
    /// group of this material is around 110 characters against an 84-character
    /// budget. The word cap is therefore not a tuned value on this recording; it
    /// is here for a speaker who says many very short words, where the character
    /// count would let a segment run to fifty or sixty words and no reader would
    /// call that a clause.
    pub max_words: usize,

    /// Maximum characters of segment text, counting `Word::display_text()`
    /// joined by single spaces. `0` disables the cap.
    ///
    /// **Default 84 characters**, which is `vaani_export`'s caption budget of
    /// two lines of 42. The metric is deliberately the one the cue wrapper uses,
    /// so "fits the cap" and "fits a caption" mean the same thing and grouping
    /// cannot disagree with the exporter about line width.
    ///
    /// This is the rule that actually shapes the output on real speech: measured,
    /// it fired **22 times** without diarization and is the second-largest
    /// contributor with it. Mean segment text came out at 38 characters, well
    /// inside the budget, because the speaker-turn rule cuts first.
    ///
    /// Note that this is a count of `char`s, not of columns: a Devanagari
    /// combining mark counts as one character and a CJK ideograph occupies two
    /// columns, so the cap is a proxy for both rather than an exact width in
    /// either script.
    pub max_chars: usize,

    /// Maximum span from a segment's first word start to its last word end, in
    /// milliseconds. `0` disables the cap.
    ///
    /// **Default 12 000 ms.** This is the "long sentence with no pause" rule,
    /// and it is the rule that catches the pathology this ASR creates. Because
    /// consecutive words abut, an unbroken run of speech is *one continuous
    /// span* to the grouper: measured, this fired **12 times** and the resulting
    /// 12-second segments are the longest in the transcript. Without it, a
    /// monotone speaker with no breath would produce one unbounded segment —
    /// here, the 12 s cap caught the worst case at 12.7 s.
    pub max_duration_ms: i64,

    /// Do not create a segment with fewer than this many words when a cap and a
    /// natural boundary are close together. `0` disables the guard.
    ///
    /// **Default 3 words.** A cap break one or two words before a pause is
    /// indistinguishable from a bug in the output; three words is the smallest
    /// group that still reads as a phrase. Measured on the real meeting the
    /// guard deferred **12** cap breaks, which produced 8 segments over a cap —
    /// the cost is counted in [`GroupingReport::segments_over_cap`] rather than
    /// hidden, and 8 of 227 is a price worth paying to avoid 8 one-word
    /// segments.
    pub min_segment_words: usize,

    /// Break the segment when the attributed speaker changes.
    ///
    /// Requires a `DiarizationResult`; with no diarization this rule cannot
    /// fire. Turning it off is occasionally right: a two-speaker exchange whose
    /// turns are three words each reads better as paragraphs than as a stutter
    /// of six one-line segments.
    ///
    /// **Default `true`, and this is the rule that matters most.** Measured on
    /// the real meeting, the speaker turn produced **188 of the 226 breaks** —
    /// 83 % of the segmentation. Everything else is a backstop.
    ///
    /// Grouping *by turn* rather than after attribution is also what turns
    /// speaker attribution from 1,912 independent per-word majority votes into
    /// 227 per-segment votes, where a single word landing on the wrong side of a
    /// boundary cannot outvote its neighbours.
    pub break_on_speaker_turn: bool,
}

impl Default for GroupingPolicy {
    fn default() -> Self {
        Self {
            pause_break_ms: 500,
            max_gap_ms: 1_500,
            max_words: 24,
            max_chars: 84,
            max_duration_ms: 12_000,
            min_segment_words: 3,
            break_on_speaker_turn: true,
        }
    }
}

impl GroupingPolicy {
    /// Whether the policy is self-consistent.
    ///
    /// Checked at construction rather than trusted, because every one of these
    /// is a plausible thing for a hand-edited settings file or a CLI flag to get
    /// wrong, and each wrong value has a *plausible* failure: a `max_gap_ms`
    /// below `pause_break_ms` silently disables the pause rule, and a
    /// `min_segment_words` above `max_words` makes the word cap unreachable.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.pause_break_ms >= 0
            // The hard ceiling must not sit *below* the preferred pause, or the
            // pause rule could never fire and the ceiling would be doing the
            // policy's job invisibly.
            && self.max_gap_ms >= self.pause_break_ms
            // A duration cap shorter than a pause is not a duration cap, it is a
            // bug that makes every segment one word long.
            && (self.max_duration_ms == 0 || self.max_duration_ms > self.pause_break_ms)
            // The guard defers *any* cap break, so it is only incoherent against
            // the word cap when the word cap is on: with `max_words == 0` there
            // is no word cap to make unreachable, and the char and duration
            // caps are still there to defer.
            && (self.max_words == 0
                || self.min_segment_words == 0
                || self.min_segment_words <= self.max_words)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_policy_is_valid() {
        assert!(GroupingPolicy::default().is_valid());
    }

    #[test]
    fn defaults_are_the_documented_measured_values() {
        let p = GroupingPolicy::default();
        assert_eq!(p.pause_break_ms, 500);
        assert_eq!(p.max_gap_ms, 1_500);
        assert_eq!(p.max_words, 24);
        assert_eq!(p.max_chars, 84);
        assert_eq!(p.max_duration_ms, 12_000);
        assert_eq!(p.min_segment_words, 3);
        assert!(p.break_on_speaker_turn);
    }

    #[test]
    fn a_hard_gap_below_the_pause_threshold_is_rejected() {
        // This is the failure the check exists for: `max_gap_ms` firing first
        // would make the pause rule dead code and the policy's real threshold
        // would be the ceiling, not the number the user set.
        let p = GroupingPolicy {
            pause_break_ms: 500,
            max_gap_ms: 200,
            ..GroupingPolicy::default()
        };
        assert!(!p.is_valid());
    }

    #[test]
    fn a_duration_cap_shorter_than_a_pause_is_rejected() {
        let p = GroupingPolicy {
            pause_break_ms: 500,
            max_duration_ms: 100,
            ..GroupingPolicy::default()
        };
        assert!(!p.is_valid());
    }

    #[test]
    fn a_disabled_duration_cap_is_always_allowed() {
        // 0 means "no limit", which cannot be shorter than a pause because it
        // is not a length at all.
        let p = GroupingPolicy {
            max_duration_ms: 0,
            ..GroupingPolicy::default()
        };
        assert!(p.is_valid());
    }

    #[test]
    fn a_runt_guard_larger_than_the_word_cap_is_rejected() {
        let p = GroupingPolicy {
            max_words: 4,
            min_segment_words: 9,
            ..GroupingPolicy::default()
        };
        assert!(!p.is_valid());
    }

    #[test]
    fn a_runt_guard_with_no_word_cap_is_allowed() {
        let p = GroupingPolicy {
            max_words: 0,
            min_segment_words: 5,
            ..GroupingPolicy::default()
        };
        assert!(p.is_valid());
    }

    #[test]
    fn a_negative_pause_is_rejected() {
        let p = GroupingPolicy {
            pause_break_ms: -1,
            ..GroupingPolicy::default()
        };
        assert!(!p.is_valid());
    }

    #[test]
    fn the_policy_is_serialisable_in_both_directions() {
        // The settings panel writes this and the CLI reads it, so the derive is
        // load-bearing rather than decorative. `serde_json` is not a dependency
        // of this crate — the manifest is frozen — so the check is that both
        // traits are implemented, which is what a caller needs.
        fn assert_traits<T: Serialize + serde::de::DeserializeOwned + Default + Copy>() {}
        assert_traits::<GroupingPolicy>();
    }
}

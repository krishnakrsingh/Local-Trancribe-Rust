//! Grouping recognised words into readable, attributable segments.
//!
//! Owned by agent **A11**. Read `docs/agent/PARALLEL_OWNERSHIP.md` first.
//!
//! This crate exists because the ASR adapter emits **one `Segment` per word**.
//! That is correct at the model boundary and useless as a product: a
//! 14-minute meeting came out as 1,913 words in 1,913 segments, and its SRT had
//! a single-word cue every 436 ms. It also made speaker attribution 1,913
//! independent per-word votes instead of a handful of turn-level decisions, so
//! one word landing in the wrong place could outvote its neighbours.
//!
//! It is a separate crate rather than a step inside `vaani-app` because the CLI
//! and the desktop app both run the pipeline, and a grouping policy that existed
//! in only one of them would give the two products different transcripts for the
//! same recording.
//!
//! # The policy, in plain terms
//!
//! Read the words left to right. Start a new segment when any of these is true
//! of the word you are about to place:
//!
//! | rule | default | why that number |
//! |---|---|---|
//! | it follows a silence of at least `pause_break_ms` | 500 ms | measured on the real 834 s meeting: the gap distribution is a smooth decay with no cliff in the 300–800 ms band, and 500 ms sits in the middle of that plateau rather than on an edge of it |
//! | it follows a silence of at least `max_gap_ms` | 1 500 ms | a segment's stated duration is `last word end − first word start`; a duration that swallows a 1.5 s hole is a lie about how long the speaker talked. This is a ceiling, not the policy — it fires far less often than the pause rule |
//! | the speaker changed | on | grouping *by turn* is what turns attribution from 1,913 per-word votes into a few hundred per-turn decisions |
//! | it would be word 25 of the segment | 24 words | a clause, not a paragraph |
//! | it would push the segment past 84 characters | 84 chars | two lines of 42, the caption budget `vaani-export` already enforces — the same metric, so grouping and the exporter cannot disagree about line width |
//! | it ends more than 12 s after the segment's first word | 12 s | the "long sentence with no pause" rule. A monotone speaker with no breath produces one unbounded segment without it |
//!
//! One rule is deliberately *not* a break: if a length cap fires within
//! `min_segment_words` (3) of a pause or a speaker turn, the break waits for
//! that natural boundary. A cap that fires one word before a pause manufactures
//! a one-word segment, which is the exact failure this crate removes. The cost
//! is counted in [`GroupingReport::segments_over_cap`] rather than hidden.
//!
//! **Not used: punctuation.** SraVaani 1.0 emits none — `tokens_to_segments`
//! concatenates SentencePiece pieces and inserts no marks — so a rule keyed on
//! `.` or `?` would never fire while looking like it worked.
//! [`grouping::word_text_contains_no_punctuation`] makes that a tested
//! statement instead of an assumption.
//!
//! # What it guarantees
//!
//! * **Every word is preserved exactly once, in time order, with its own
//!   `start` and `end` untouched.** Grouping changes which segment a word is
//!   in. It never moves a word. This is the crate's most important property and
//!   it is asserted over several hundred randomised transcripts.
//! * Segments are sorted by start, non-overlapping, never empty, never
//!   zero-length, and inside the media duration when the transcript knows it.
//! * **Idempotent**: grouping an already-grouped transcript with the same policy
//!   reproduces it exactly, ids included.
//! * Pure and deterministic: the same transcript, diarization result and policy
//!   always give the same output, which is what makes "the CLI and the app
//!   produce the same transcript" a testable statement.
//! * No transcript text in anything this crate reports. Every diagnostic is a
//!   count, a duration or a distribution.
//!
//! # Where it sits in the pipeline
//!
//! ```text
//!   asr ──▶ Transcript (one segment per word)
//!            │
//!            ├─▶ vaani_segment::group / group_with_diarization   ◀── here
//!            │        (breaks on pause, turn, cap; assigns ids)
//!            │
//!            ├─▶ vaani_diarization::attribute_speakers
//!            │        (overlap-weighted vote, now per turn)
//!            │
//!            ▼
//!         storage, UI, export
//! ```
//!
//! Grouping must run **before** attribution; see the [`grouping`] module for
//! why that ordering is the whole point.
//!
//! # Example
//!
//! ```
//! use vaani_core::{EngineStamp, Segment, Time, Transcript, Word};
//! use vaani_segment::{GroupingPolicy, group};
//!
//! // Three words, the first two separated by a 900 ms pause.
//! let mut t = Transcript::new(EngineStamp::asr("sravaani", "1.0"));
//! for (text, start, end) in [("one", 0, 300), ("two", 1_200, 1_500), ("three", 1_600, 1_900)] {
//!     t.segments.push(
//!         Segment::new(Time::from_millis(start), Time::from_millis(end))
//!             .with_words(vec![Word::new(text, Time::from_millis(start), Time::from_millis(end))]),
//!     );
//! }
//!
//! let report = group(&mut t, &GroupingPolicy::default())?;
//! assert_eq!(t.segments.len(), 2);
//! assert_eq!(t.segments[0].text(), "one");
//! assert_eq!(t.segments[1].text(), "two three");
//! // Every word survived, exactly once.
//! assert_eq!(report.words_in, report.words_out);
//! # Ok::<(), vaani_core::VaaniError>(())
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod grouping;
pub mod ids;
pub mod policy;
pub mod report;
pub mod stats;

pub use grouping::{
    GapDistribution, GroupingPlan, MIN_SEGMENT_MS, MediaBounds, PERCENTILES, PlannedGroup,
    THRESHOLD_SWEEP, ThresholdOutcome, gap_distribution, group, group_with_diarization,
    group_with_diarization_within, group_within, plan_groups, same_grouping, same_segmentation,
    threshold_sweep, word_text_contains_no_punctuation,
};
pub use ids::group_id;
pub use policy::GroupingPolicy;
pub use report::{BreakReason, GroupingReport};
pub use stats::{TranscriptStats, percentile, stats};

/// Crate version, from the manifest.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

//! The machine-readable report. One schema, three error classes, no blend.
//!
//! # The schema
//!
//! ```text
//! report
//!   version, generated_by, normalisation { profile, rules[] }
//!   set            { clips, human_reference_clips, awaiting_annotation_clips }
//!   clips[]
//!     clip_id, scored, unscored_reason
//!     asr           { status, wer, cer_*, sub, del, ins, ... }
//!     diarization   { status, der, components, jer, speaker_count, ... }
//!     timing        { status, start_deviation, end_deviation, ... }
//!     intrinsic     { word_count, segment_count, speaker_count, ... }
//!     taxonomy      { signals[] }
//!   blend          { permitted: false, value: null, why: "..." }
//!   aggregate      { ... }
//! ```
//!
//! # Why `blend.permitted` is in the schema
//!
//! A single accuracy number cannot answer "what should we fix first". A system
//! with 30 % WER and 5 % DER has a word problem; a system with 8 % WER and
//! 60 % DER has a speaker problem; one with 8 % and 5 % and 400 ms of drift
//! has a timing problem. Any arithmetic that collapses those three is a
//! weighted average of guesses, and the weights are always chosen after the
//! fact. So the harness refuses to emit one, and says so in the output rather
//! than only in this doc comment.
//!
//! Every `status` is one of:
//!
//! * `scored` — a human reference exists and the numbers are measurements;
//! * `unscored` — no human reference, with `reason` naming what is missing;
//! * `undefined` — a reference exists but the metric has no denominator (an
//!   empty reference, a clip with no speech turns). Distinct from `unscored`
//!   and never rendered as `0`.

use serde::{Deserialize, Serialize};

use crate::der::DiarizationReport;
use crate::hypothesis::Intrinsic;
use crate::normalize::NormalisationProfile;
use crate::taxonomy::Signal;
use crate::timing::TimingReport;
use crate::wer::WordErrorReport;

/// Schema version. Bumped when a field changes meaning; a consumer should
/// refuse an unknown version rather than read a number it does not understand.
pub const REPORT_VERSION: u32 = 1;

/// Whether a block carries a measurement, and why not when it does not.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum BlockStatus {
    /// A human reference exists; the payload is a measurement.
    Scored,
    /// No human reference. `reason` says what is missing.
    Unscored {
        /// Machine-readable reason, e.g. `no_human_reference`.
        reason: String,
        /// Prose for a human reader. Never asserted on.
        detail: String,
    },
    /// A reference exists but the metric has no denominator.
    Undefined {
        /// Machine-readable reason, e.g. `empty_reference`.
        reason: String,
        /// Prose for a human reader.
        detail: String,
    },
}

impl BlockStatus {
    /// True only for a real measurement.
    pub fn is_scored(&self) -> bool {
        matches!(self, Self::Scored)
    }
}

/// Why a clip could not be scored at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnscoredReason {
    /// No `reference.txt`. The clip exists, the system ran, and nobody has
    /// listened to the audio and written down what was said.
    NoHumanAsrReference,
    /// No `reference.rttm`.
    NoHumanDiarizationReference,
    /// No `reference.words.json`, so only segment-level timing could be
    /// measured and even that needs a reference transcript.
    NoWordTimingReference,
    /// The system produced no hypothesis for this clip.
    NoHypothesis,
}

/// The ASR block: **wrong words**. Never mixed with speaker or timing error.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AsrBlock {
    /// Measurement status, flattened so `status` is a string on the wire
    /// rather than a nested object.
    #[serde(flatten)]
    pub status: BlockStatus,
    /// Word error rate, a fraction. Present only when `status` is `scored`.
    #[serde(flatten)]
    pub report: Option<WordErrorReport>,
}

/// The diarization block: **right words, wrong speaker**.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiarizationBlock {
    /// Measurement status, flattened.
    #[serde(flatten)]
    pub status: BlockStatus,
    /// The full diarization verdict.
    #[serde(flatten)]
    pub report: Option<DiarizationReport>,
}

/// The timing block: **right words and speaker, wrong time**.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimingBlock {
    /// Measurement status, flattened.
    #[serde(flatten)]
    pub status: BlockStatus,
    /// The full timing verdict.
    #[serde(flatten)]
    pub report: Option<TimingReport>,
}

/// One clip in the report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipReport {
    /// Clip id, matching `evaluation/clips/<id>.wav`.
    pub clip_id: String,
    /// Start of the clip in the source recording, milliseconds. Structural
    /// metadata; no content.
    pub source_start_ms: i64,
    /// Clip length, milliseconds.
    pub duration_ms: i64,
    /// `true` when a human `reference.txt` exists for this clip.
    pub has_human_asr_reference: bool,
    /// `true` when a human `reference.rttm` exists for this clip.
    pub has_human_diarization_reference: bool,
    /// `true` when a human word-timing reference exists for this clip.
    pub has_human_word_timing_reference: bool,
    /// `true` when the system produced a hypothesis.
    pub has_hypothesis: bool,
    /// True when at least one block is a measurement.
    pub scored: bool,
    /// ASR error.
    pub asr: AsrBlock,
    /// Diarization error.
    pub diarization: DiarizationBlock,
    /// Timestamp error.
    pub timing: TimingBlock,
    /// Facts about the hypothesis that need no reference. Not accuracy.
    pub intrinsic: Option<Intrinsic>,
    /// Detectable error signatures.
    pub taxonomy: Vec<Signal>,
}

/// A refused blended score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlendRefusal {
    /// Always `false`.
    pub permitted: bool,
    /// Always `null`.
    pub value: Option<f64>,
    /// Why.
    pub why: String,
}

impl Default for BlendRefusal {
    fn default() -> Self {
        Self {
            permitted: false,
            value: None,
            why: "A single accuracy number cannot answer \"what should we fix first\". \
                  ASR, diarization and timestamp error are three independent failures with \
                  three independent fixes, and any arithmetic that collapses them is a weighted \
                  average of guesses whose weights are always chosen after seeing the result. \
                  Read the three blocks separately."
                .to_string(),
        }
    }
}

/// Set-level counts, stated without hedging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SetCounts {
    /// Clips in the set.
    pub clips: usize,
    /// Clips with a human ASR reference.
    pub human_asr_reference: usize,
    /// Clips with a human diarization reference.
    pub human_diarization_reference: usize,
    /// Clips with a human word-timing reference.
    pub human_word_timing_reference: usize,
    /// Clips with no human reference of any kind.
    pub awaiting_annotation: usize,
    /// Clips the system produced a hypothesis for.
    pub with_hypothesis: usize,
}

/// The whole report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Report {
    /// Schema version.
    pub version: u32,
    /// Crate version that wrote it.
    pub generated_by: String,
    /// The normalisation profile, and every rule it applied. Present in the
    /// report so no number is ever separable from the rules behind it.
    pub normalisation: NormalisationBlock,
    /// Set-level counts.
    pub set: SetCounts,
    /// Per-clip results.
    pub clips: Vec<ClipReport>,
    /// The refused blend.
    pub blend: BlendRefusal,
}

/// The normalisation profile plus its rules.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalisationBlock {
    /// Wire name of the profile.
    pub profile: String,
    /// Every rule, in application order, with the reason it exists.
    pub rules: Vec<serde_json::Value>,
    /// A statement of what the profile does not do, so a reader cannot assume
    /// transliteration, stemming or filler removal happened.
    pub does_not: Vec<String>,
}

impl NormalisationBlock {
    /// Build the block for a profile.
    pub fn of(profile: NormalisationProfile) -> Self {
        Self {
            profile: profile.id().to_string(),
            rules: profile.rules_json(),
            does_not: vec![
                "no transliteration between Devanagari and Latin: kya and क्या stay different tokens".to_string(),
                "no stemming or lemmatisation: WER is lexical".to_string(),
                "no filler-word removal".to_string(),
                "no number-word folding: पाँच against 5 is a substitution plus a deletion, and that is a true statement about the two texts".to_string(),
                "no lower-casing of Devanagari, which has no case".to_string(),
                "identical rules on the reference and the hypothesis; there is no second code path".to_string(),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_blend_is_refused_in_the_serialised_output() {
        let r = BlendRefusal::default();
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["permitted"], false);
        assert!(v["value"].is_null());
        assert!(v["why"].as_str().unwrap().len() > 40);
    }

    #[test]
    fn a_scored_status_round_trips_through_its_tag() {
        let s = BlockStatus::Unscored {
            reason: "no_human_reference".into(),
            detail: "x".into(),
        };
        let v = serde_json::to_value(&s).unwrap();
        assert_eq!(v["status"], "unscored");
        assert_eq!(v["reason"], "no_human_reference");
        assert!(!s.is_scored());
        let back: BlockStatus = serde_json::from_value(v).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn the_normalisation_block_names_every_rule_and_the_omissions() {
        let b = NormalisationBlock::of(NormalisationProfile::HinglishTranslit);
        let ids: Vec<&str> = b
            .rules
            .iter()
            .filter_map(|v| v.get("id"))
            .filter_map(|v| v.as_str())
            .collect();
        assert_eq!(ids.first(), Some(&"N1"));
        assert!(ids.contains(&"H1"));
        assert!(ids.contains(&"H2"));
        assert!(b.does_not.len() >= 5);
    }

    #[test]
    fn the_strict_profile_applies_no_h_rules() {
        let b = NormalisationBlock::of(NormalisationProfile::Strict);
        let ids: Vec<&str> = b
            .rules
            .iter()
            .filter_map(|v| v.get("id"))
            .filter_map(|v| v.as_str())
            .collect();
        assert!(!ids.contains(&"H1"));
        assert!(!ids.contains(&"H2"));
    }
}

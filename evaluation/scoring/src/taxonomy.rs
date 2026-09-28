//! Turning `docs/evaluation/ERROR_TAXONOMY.md` into something that runs.
//!
//! A taxonomy whose categories cannot be detected is a list of intentions. So
//! every category in the document has, here, a machine-checkable **signature**:
//! an id, the error class it belongs to, a count, a unit, and — when the
//! signature could not be evaluated — a stated reason.
//!
//! # The honesty rule this module exists to enforce
//!
//! Several categories are **not** detectable from a hypothesis alone, and this
//! module says so rather than emitting a proxy that looks like a measurement:
//!
//! * `asr.hallucinated_word`, `asr.omitted_word` and `asr.substituted_word`
//!   need a human reference. Without one they report `unavailable`, and the
//!   count is `None` — not `0`, because "none found" and "not looked" are
//!   different claims.
//! * `asr.named_entity_error` needs a human reference **and** a list of the
//!   names that matter. Neither exists yet.
//! * `timing.*` need reference word timings.
//! * `diarization.wrong_speaker`, `.missed_turn` and `.false_turn` are not
//!   re-counted here; they are `components.confusion`, `.missed_detection` and
//!   `.false_alarm` of the diarization block, and duplicating the arithmetic
//!   would only create a second number that can drift away from the first.
//!
//! Four signatures *are* exactly measurable without a reference, because they
//! are claims about the hypothesis alone, and they are reported as `measured`
//! with the class they belong to: repeated decode loops, punctuation-only
//! tokens, mixed-script tokens, and reference overlap time.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use crate::edit::align;
use crate::hypothesis::Hypothesis;
use crate::normalize::{NormalisationProfile, ScriptMix, normalise, script_mix};
use crate::rttm::RttmRecord;

/// Which of the three error classes a signal belongs to. There is no fourth
/// class and no blended one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorClass {
    /// Wrong words.
    Asr,
    /// Right words, wrong speaker.
    Diarization,
    /// Right words and speaker, wrong time.
    Timing,
}

/// Whether a signal was measured, approximated, or could not be evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalStatus {
    /// Counted exactly.
    ///
    /// Either from a reference and a hypothesis, or from the hypothesis alone
    /// when the signature is a claim about the hypothesis.
    Measured,
    /// Counted without a reference, and known to be an approximation.
    Proxy,
    /// Not evaluated. `note` says what is missing and `count` is `None`.
    Unavailable,
}

/// One detectable error signature.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Signal {
    /// Stable id, matching a row in `ERROR_TAXONOMY.md`.
    pub id: String,
    /// The error class. Never blended.
    pub class: ErrorClass,
    /// How the count was obtained.
    pub status: SignalStatus,
    /// The count. `0` when evaluated and the signature was absent; `None`
    /// when `Unavailable`, because zero and unknown are not the same number.
    pub count: Option<i64>,
    /// What the count is a count of.
    pub unit: String,
    /// The threshold used, when there is one.
    pub threshold: Option<String>,
    /// Why it is unavailable, or what a proxy's blind spot is.
    pub note: String,
}

impl Signal {
    fn measured(
        id: &str,
        class: ErrorClass,
        count: i64,
        unit: &str,
        threshold: Option<String>,
    ) -> Self {
        Self {
            id: id.to_string(),
            class,
            status: SignalStatus::Measured,
            count: Some(count),
            unit: unit.to_string(),
            threshold,
            note: String::new(),
        }
    }
    fn proxy(id: &str, class: ErrorClass, count: i64, unit: &str, note: &str) -> Self {
        Self {
            id: id.to_string(),
            class,
            status: SignalStatus::Proxy,
            count: Some(count),
            unit: unit.to_string(),
            threshold: None,
            note: note.to_string(),
        }
    }
    fn unavailable(id: &str, class: ErrorClass, unit: &str, note: &str) -> Self {
        Self {
            id: id.to_string(),
            class,
            status: SignalStatus::Unavailable,
            count: None,
            unit: unit.to_string(),
            threshold: None,
            note: note.to_string(),
        }
    }
}

/// Thresholds for the timing signatures, all in milliseconds. Collected here
/// and serialised so a report quotes the exact number behind every fired
/// threshold.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimingThresholds {
    /// Beyond this, a word is "shifted" rather than jittery.
    pub shifted_ms: i64,
    /// Beyond this, a word is "drifted".
    pub drift_ms: i64,
    /// Hypothesis span longer than this multiple of the reference span.
    pub long_ratio: f64,
    /// Hypothesis span shorter than this multiple of the reference span.
    pub short_ratio: f64,
}

impl Default for TimingThresholds {
    fn default() -> Self {
        Self {
            shifted_ms: 500,
            drift_ms: 200,
            long_ratio: 1.5,
            short_ratio: 0.667,
        }
    }
}

/// The reference side of the taxonomy, or the fact that there is none.
#[derive(Debug)]
pub enum ReferenceView<'a> {
    /// A reference a person produced by listening. A reference derived from any
    /// model output is not this variant, and the ASR signals then refuse to
    /// count it — a system scored against its own output is a
    /// self-consistency check, not a measurement.
    Human {
        /// Normalised reference tokens.
        tokens: &'a [String],
        /// Normalised hypothesis tokens.
        hypothesis_tokens: &'a [String],
    },
    /// No usable reference.
    None,
}

/// Timing signature counts, computed by the caller that holds the alignment
/// and both sets of word times.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimingCounts {
    /// Aligned words starting more than `shifted_ms` away.
    pub shifted: i64,
    /// Aligned words starting more than `drift_ms` away.
    pub drifted: i64,
    /// Aligned words whose hypothesis span exceeds `long_ratio`.
    pub too_long: i64,
    /// Aligned words whose hypothesis span falls below `short_ratio`.
    pub too_short: i64,
}

/// Evaluate every signature.
pub fn evaluate(
    hypothesis: &Hypothesis,
    reference: ReferenceView<'_>,
    hypothesis_rttm: &[RttmRecord],
    reference_rttm: &[RttmRecord],
    timing_counts: Option<TimingCounts>,
    thresholds: TimingThresholds,
) -> Vec<Signal> {
    let mut s = Vec::new();
    let strict = NormalisationProfile::Strict;
    let i = hypothesis.intrinsic();

    // ---- ASR: exactly measurable from the hypothesis alone ----------------
    s.push(Signal::measured(
        "asr.duplicated_word",
        ErrorClass::Asr,
        i.duplicate_runs as i64,
        "contiguous runs of >= 2 adjacent identical normalised words",
        None,
    ));
    let punct = hypothesis
        .words
        .iter()
        .filter(|w| !w.text.is_empty() && w.text.chars().all(crate::normalize::is_punct_or_symbol))
        .count() as i64;
    s.push(Signal::measured(
        "asr.punctuation_error",
        ErrorClass::Asr,
        punct,
        "hypothesis words that are pure punctuation; the normaliser deletes these, so WER is blind to them and this counter is the only thing that sees them",
        None,
    ));
    let mixed_hyp = hypothesis
        .words
        .iter()
        .filter(|w| {
            let joined: String = normalise(&w.text, strict)
                .tokens
                .iter()
                .flat_map(|t| t.chars())
                .collect();
            script_mix(&joined) == ScriptMix::Mixed
        })
        .count() as i64;
    s.push(Signal::measured(
        "asr.code_switch_boundary_error",
        ErrorClass::Asr,
        mixed_hyp,
        "hypothesis words mixing Devanagari and Latin in one token",
        None,
    ));

    // ---- ASR: needs a human reference -------------------------------------
    match reference {
        ReferenceView::Human {
            tokens,
            hypothesis_tokens,
        } => {
            let a = align(tokens, hypothesis_tokens);
            s.push(Signal::measured(
                "asr.hallucinated_word",
                ErrorClass::Asr,
                a.insertions as i64,
                "hypothesis tokens with no aligned reference token",
                None,
            ));
            s.push(Signal::measured(
                "asr.omitted_word",
                ErrorClass::Asr,
                a.deletions as i64,
                "reference tokens with no aligned hypothesis token",
                None,
            ));
            s.push(Signal::measured(
                "asr.substituted_word",
                ErrorClass::Asr,
                a.substitutions as i64,
                "aligned pairs whose two tokens differ",
                None,
            ));
            s.push(Signal::measured(
                "asr.mixed_script_reference_token",
                ErrorClass::Asr,
                tokens.iter().filter(|t| script_mix(t) == ScriptMix::Mixed).count() as i64,
                "reference tokens mixing Devanagari and Latin; a reference-side code-switch boundary signal, distinct from the hypothesis-side count above",
                None,
            ));
        }
        ReferenceView::None => {
            s.push(Signal::unavailable(
                "asr.hallucinated_word",
                ErrorClass::Asr,
                "hypothesis tokens with no aligned reference token",
                "no human reference for this clip",
            ));
            s.push(Signal::unavailable(
                "asr.omitted_word",
                ErrorClass::Asr,
                "reference tokens with no aligned reference",
                "no human reference for this clip",
            ));
            s.push(Signal::unavailable(
                "asr.substituted_word",
                ErrorClass::Asr,
                "aligned pairs whose two tokens differ",
                "no human reference for this clip",
            ));
            s.push(Signal::proxy(
                "asr.hallucinated_word_proxy",
                ErrorClass::Asr,
                i.adjacent_duplicate_words as i64,
                "adjacent identical hypothesis word pairs",
                "a repeat is what a decoding loop looks like, but a genuinely repeated word is indistinguishable from one; it is a candidate count, not a hallucination count",
            ));
        }
    }
    s.push(Signal::unavailable(
        "asr.named_entity_error",
        ErrorClass::Asr,
        "aligned name tokens that differ",
        "needs a human reference AND evaluation/metadata/entities.txt listing the names that matter; the list does not exist yet, so this category is documented but not measured",
    ));

    // ---- Diarization ------------------------------------------------------
    let has_ref_diar = reference_rttm.iter().any(|r| r.is_turn());
    if has_ref_diar {
        let ref_speakers = distinct_names(reference_rttm);
        let hyp_speakers = distinct_names(hypothesis_rttm);
        // Best-match map from each reference speaker to a hypothesis label.
        let mut mapped: Vec<&String> = Vec::new();
        for rs in &ref_speakers {
            if let Some(h) = hypothesis_rttm
                .iter()
                .filter(|h| h.is_turn())
                .max_by_key(|h| pair_overlap_us(reference_rttm, rs, h))
            {
                mapped.push(&h.name);
            }
        }
        let merged = ref_speakers.len() - mapped.iter().collect::<BTreeSet<_>>().len();
        s.push(Signal::measured(
            "diarization.speaker_merge",
            ErrorClass::Diarization,
            merged as i64,
            "reference speakers whose best-matching hypothesis label is the same speaker's",
            None,
        ));
        s.push(Signal::measured(
            "diarization.speaker_split",
            ErrorClass::Diarization,
            hyp_speakers.len().saturating_sub(ref_speakers.len()) as i64,
            "hypothesis labels in excess of the reference speaker count",
            None,
        ));
        s.push(Signal::measured(
            "diarization.overlap_failure",
            ErrorClass::Diarization,
            reference_overlap_ms(reference_rttm),
            "milliseconds in which two reference speakers were active at once; a non-zero value means the reference contains overlap and a diarizer that emits one speaker per frame cannot represent it",
            None,
        ));
        for (id, unit) in [
            ("diarization.wrong_speaker", "milliseconds of confusion"),
            (
                "diarization.missed_turn",
                "milliseconds of missed detection",
            ),
            ("diarization.false_turn", "milliseconds of false alarm"),
        ] {
            s.push(Signal::unavailable(
                id,
                ErrorClass::Diarization,
                unit,
                "reported once, as a component of the diarization error rate; not re-counted here so the two numbers cannot drift apart",
            ));
        }
    } else {
        for (id, unit) in [
            (
                "diarization.speaker_merge",
                "reference speakers mapped onto one hypothesis label",
            ),
            (
                "diarization.speaker_split",
                "hypothesis labels beyond the reference count",
            ),
            (
                "diarization.overlap_failure",
                "milliseconds of reference overlap",
            ),
            ("diarization.wrong_speaker", "milliseconds of confusion"),
            (
                "diarization.missed_turn",
                "milliseconds of missed detection",
            ),
            ("diarization.false_turn", "milliseconds of false alarm"),
        ] {
            s.push(Signal::unavailable(
                id,
                ErrorClass::Diarization,
                unit,
                "no human diarization reference for this clip",
            ));
        }
    }

    // ---- Timing -----------------------------------------------------------
    match timing_counts {
        Some(c) => {
            s.push(Signal::measured(
                "timing.segment_shifted",
                ErrorClass::Timing,
                c.shifted,
                "aligned words whose start differs from the reference by more than the shift threshold",
                Some(format!("> {} ms", thresholds.shifted_ms)),
            ));
            s.push(Signal::measured(
                "timing.word_timestamp_drift",
                ErrorClass::Timing,
                c.drifted,
                "aligned words whose start differs by more than the drift threshold",
                Some(format!("> {} ms", thresholds.drift_ms)),
            ));
            s.push(Signal::measured(
                "timing.segment_too_long",
                ErrorClass::Timing,
                c.too_long,
                "aligned words whose hypothesis span exceeds the reference span by more than the long ratio",
                Some(format!("> {}x", thresholds.long_ratio)),
            ));
            s.push(Signal::measured(
                "timing.segment_too_short",
                ErrorClass::Timing,
                c.too_short,
                "aligned words whose hypothesis span falls below the reference span by more than the short ratio",
                Some(format!("< {}x", thresholds.short_ratio)),
            ));
        }
        None => {
            for (id, unit) in [
                (
                    "timing.segment_shifted",
                    "aligned words starting past the shift threshold",
                ),
                (
                    "timing.word_timestamp_drift",
                    "aligned words starting past the drift threshold",
                ),
                (
                    "timing.segment_too_long",
                    "aligned words whose span exceeds the long ratio",
                ),
                (
                    "timing.segment_too_short",
                    "aligned words whose span falls below the short ratio",
                ),
            ] {
                s.push(Signal::unavailable(
                    id,
                    ErrorClass::Timing,
                    unit,
                    "no human word-timing reference for this clip",
                ));
            }
        }
    }

    s
}

fn distinct_names(rs: &[RttmRecord]) -> Vec<String> {
    let mut v: Vec<String> = rs
        .iter()
        .filter(|r| r.is_turn())
        .map(|r| r.name.clone())
        .collect();
    v.sort();
    v.dedup();
    v
}

fn pair_overlap_us(refs: &[RttmRecord], a: &str, b: &RttmRecord) -> i64 {
    refs.iter()
        .filter(|r| r.is_turn() && r.name == a)
        .map(|r| (r.end_us().min(b.end_us()) - r.start_us.max(b.start_us)).max(0))
        .sum()
}

/// Total milliseconds in which at least two reference speakers were active.
///
/// Swept from the reference boundaries, so it is a property of the reference
/// alone and can be reported before any hypothesis exists. A non-zero result
/// is the single most important fact about a diarization reference: a system
/// that emits at most one speaker per instant cannot represent it, and its DER
/// is bounded below accordingly.
pub fn reference_overlap_ms(reference: &[RttmRecord]) -> i64 {
    let turns: Vec<(&str, i64, i64)> = reference
        .iter()
        .filter(|r| r.is_turn())
        .map(|r| (r.name.as_str(), r.start_us, r.end_us()))
        .collect();
    let mut points: Vec<i64> = turns.iter().flat_map(|(_, a, b)| [*a, *b]).collect();
    points.sort_unstable();
    points.dedup();
    let mut total = 0i64;
    for w in points.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        if hi <= lo {
            continue;
        }
        // Every turn boundary is a point, so a turn either covers this whole
        // elementary interval or none of it. Overlap means two *distinct
        // speakers*, not two turns: one speaker pausing and resuming does not
        // create overlap.
        let n = turns
            .iter()
            .filter(|t| t.1 <= lo && t.2 >= hi)
            .map(|&(name, _, _)| name)
            .collect::<BTreeSet<_>>()
            .len();
        if n >= 2 {
            total += hi - lo;
        }
    }
    total / 1000
}

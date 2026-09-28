//! Timestamp error, separated from every other error class.
//!
//! # What is measurable, and what is not
//!
//! Word-level timestamp accuracy needs a reference that carries **word
//! timings**. An orthographic `reference.txt` does not, and no honest number
//! can be produced from one. So [`score`] takes an optional list of
//! [`ReferenceWord`]s, and when they are absent it returns
//! [`TimingReport::unavailable`] with the reason rather than a zero. A zero
//! here would be indistinguishable from "the system places every word at the
//! right millisecond", which is precisely the claim this module exists to
//! check.
//!
//! # Metrics
//!
//! Words are paired through the **same token alignment** that produces WER, so
//! a substitution is still a *timing* comparison: the system said the right
//! word at the wrong time, and that is a timestamp error, not a word error. The
//! two classes are reported separately and neither is derived from the other.
//!
//! * `mean_start_deviation_ms`, `median_...`, `p90_...`, `max_abs_...` over
//!   `hypothesis_start - reference_start`.
//! * the same for end times.
//! * `overlapping_pairs` / `negative_duration_pairs`: pairs whose span has a
//!   negative or zero length, which is a structural defect.
//! * `matched_pairs` next to every mean, always.

use serde::{Deserialize, Serialize};

use crate::edit::{Op, align};
use crate::hypothesis::ReferenceWord;
use crate::normalize::NormalisedText;

/// Why no timestamp number could be produced.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", content = "detail", rename_all = "snake_case")]
pub enum TimingUnavailable {
    /// The clip has no `reference.words.json`, so no reference word times.
    NoReferenceWordTimings,
    /// The reference has words but the alignment paired none of them with a
    /// hypothesis word that also carries timings.
    NoAlignedPairs,
}

/// Distribution of a signed or absolute deviation series.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Deviation {
    /// Number of samples.
    pub count: usize,
    /// Arithmetic mean, milliseconds.
    pub mean_ms: f64,
    /// Median, milliseconds.
    pub median_ms: f64,
    /// 90th percentile by the nearest-rank method, milliseconds.
    pub p90_ms: f64,
    /// Largest absolute sample, milliseconds.
    pub max_abs_ms: f64,
}

/// The timestamp verdict for one clip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimingReport {
    /// `true` when deviation figures were computed.
    pub defined: bool,
    /// Why they were not, when they were not.
    pub unavailable: Option<TimingUnavailable>,
    /// `hyp_start - ref_start` across aligned pairs.
    pub start_deviation: Option<Deviation>,
    /// `hyp_end - ref_end` across aligned pairs.
    pub end_deviation: Option<Deviation>,
    /// Aligned pairs carrying timings on both sides.
    pub matched_pairs: usize,
    /// Aligned pairs whose hypothesis span is non-positive.
    pub negative_duration_pairs: usize,
    /// Pairs where the hypothesis span overlaps the reference span by less
    /// than half its own length — a word placed inside the wrong word.
    pub poorly_aligned_pairs: usize,
    /// Offset that would minimise the mean absolute start error, milliseconds.
    /// A systematic lag is a very different bug from jitter, and the mean
    /// alone cannot tell them apart.
    pub best_constant_offset_ms: f64,
}

/// The report for a clip with no reference word timings.
pub fn unavailable(reason: TimingUnavailable) -> TimingReport {
    TimingReport {
        defined: false,
        unavailable: Some(reason),
        start_deviation: None,
        end_deviation: None,
        matched_pairs: 0,
        negative_duration_pairs: 0,
        poorly_aligned_pairs: 0,
        best_constant_offset_ms: 0.0,
    }
}

/// Score hypothesis word timings against reference word timings.
///
/// `reference` must already be normalised by the caller, using the same
/// profile as the reference text, so that the alignment indices line up with
/// [`reference_tokens`].
pub fn score(
    reference_tokens: &NormalisedText,
    reference_words: &[ReferenceWord],
    hypothesis: &NormalisedText,
    hypothesis_words: &[(i64, i64)],
) -> TimingReport {
    if reference_words.is_empty() {
        return unavailable(TimingUnavailable::NoReferenceWordTimings);
    }
    let a = align(&reference_tokens.tokens, &hypothesis.tokens);
    let mut starts: Vec<f64> = Vec::new();
    let mut ends: Vec<f64> = Vec::new();
    let mut matched = 0usize;
    let mut negative = 0usize;
    let mut poor = 0usize;

    for (k, op) in a.ops.iter().enumerate() {
        if !matches!(op, Op::Match | Op::Sub) {
            continue;
        }
        let ri = match a.ref_index[k] {
            Some(i) if i < reference_words.len() => i,
            _ => continue,
        };
        let hi = match a.hyp_index[k] {
            Some(i) if i < hypothesis_words.len() => i,
            _ => continue,
        };
        let (hs, he) = hypothesis_words[hi];
        let (rs, re) = (reference_words[ri].start_ms, reference_words[ri].end_ms);
        matched += 1;
        if he <= hs {
            negative += 1;
        }
        let inter = (he.min(re) - hs.max(rs)).max(0);
        let union = (he.max(re) - hs.min(rs)).max(1);
        if inter * 2 < union {
            poor += 1;
        }
        starts.push((hs - rs) as f64);
        ends.push((he - re) as f64);
    }

    if matched == 0 {
        return unavailable(TimingUnavailable::NoAlignedPairs);
    }

    TimingReport {
        defined: true,
        unavailable: None,
        start_deviation: Some(deviation(&starts)),
        end_deviation: Some(deviation(&ends)),
        matched_pairs: matched,
        negative_duration_pairs: negative,
        poorly_aligned_pairs: poor,
        best_constant_offset_ms: best_offset(&starts),
    }
}

fn deviation(xs: &[f64]) -> Deviation {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    let mean = s.iter().sum::<f64>() / n as f64;
    let median = if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    };
    // Nearest-rank p90: the smallest sample at or above the 90th percentile
    // position. Chosen over linear interpolation because the samples are
    // millisecond differences and interpolating between two of them invents a
    // value that was never measured.
    let rank = ((0.9 * n as f64).ceil() as usize).clamp(1, n);
    Deviation {
        count: n,
        mean_ms: round3(mean),
        median_ms: round3(median),
        p90_ms: round3(s[rank - 1]),
        max_abs_ms: round3(s.iter().map(|x| x.abs()).fold(0.0f64, f64::max)),
    }
}

/// The offset that minimises the mean absolute error: the median, because the
/// absolute loss is minimised by a median and the squared loss by a mean.
fn best_offset(xs: &[f64]) -> f64 {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    let m = if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    };
    round3(m)
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize::{NormalisationProfile, normalise};

    fn n(s: &str) -> NormalisedText {
        normalise(s, NormalisationProfile::Strict)
    }

    #[test]
    fn no_reference_timings_is_unavailable_and_never_zero() {
        let r = score(&n("a b"), &[], &n("a b"), &[(0, 100), (100, 200)]);
        assert!(!r.defined);
        assert_eq!(
            r.unavailable,
            Some(TimingUnavailable::NoReferenceWordTimings)
        );
        assert!(r.start_deviation.is_none());
    }

    #[test]
    fn a_constant_lag_shows_up_as_a_median_and_a_tiny_spread() {
        let refw = vec![
            ReferenceWord {
                text: "a".into(),
                start_ms: 0,
                end_ms: 100,
            },
            ReferenceWord {
                text: "b".into(),
                start_ms: 200,
                end_ms: 300,
            },
        ];
        let r = score(&n("a b"), &refw, &n("a b"), &[(120, 220), (320, 420)]);
        assert!(r.defined);
        let d = r.start_deviation.unwrap();
        assert_eq!(d.count, 2);
        assert!((d.mean_ms - 120.0).abs() < 1e-9, "{d:?}");
        assert!((d.max_abs_ms - 120.0).abs() < 1e-9, "{d:?}");
        assert!((r.best_constant_offset_ms - 120.0).abs() < 1e-9);
    }

    #[test]
    fn a_substitution_is_still_a_timing_comparison() {
        // Reference says "a b", hypothesis says "a c". The second pair is a
        // substitution, and its timing is still measured: wrong word AND wrong
        // time are two separate facts.
        let refw = vec![
            ReferenceWord {
                text: "a".into(),
                start_ms: 0,
                end_ms: 100,
            },
            ReferenceWord {
                text: "b".into(),
                start_ms: 200,
                end_ms: 300,
            },
        ];
        let r = score(&n("a b"), &refw, &n("a c"), &[(0, 100), (260, 360)]);
        assert_eq!(r.matched_pairs, 2);
        assert!((r.start_deviation.unwrap().mean_ms - 30.0).abs() < 1e-9);
    }

    #[test]
    fn a_deleted_word_contributes_no_timing_sample() {
        let refw = vec![
            ReferenceWord {
                text: "a".into(),
                start_ms: 0,
                end_ms: 100,
            },
            ReferenceWord {
                text: "b".into(),
                start_ms: 200,
                end_ms: 300,
            },
        ];
        let r = score(&n("a b"), &refw, &n("a"), &[(0, 100)]);
        assert_eq!(r.matched_pairs, 1);
    }

    #[test]
    fn no_alignment_means_no_alignment() {
        let refw = vec![ReferenceWord {
            text: "a".into(),
            start_ms: 0,
            end_ms: 100,
        }];
        let r = score(&n("a"), &refw, &n("b"), &[]);
        assert!(!r.defined);
        assert_eq!(r.unavailable, Some(TimingUnavailable::NoAlignedPairs));
    }

    #[test]
    fn a_negative_hypothesis_span_is_counted() {
        let refw = vec![ReferenceWord {
            text: "a".into(),
            start_ms: 100,
            end_ms: 200,
        }];
        let r = score(&n("a"), &refw, &n("a"), &[(200, 100)]);
        assert_eq!(r.negative_duration_pairs, 1);
    }

    #[test]
    fn a_word_placed_in_the_wrong_place_is_counted_as_poorly_aligned() {
        let refw = vec![ReferenceWord {
            text: "a".into(),
            start_ms: 0,
            end_ms: 100,
        }];
        let r = score(&n("a"), &refw, &n("a"), &[(5_000, 5_100)]);
        assert_eq!(r.poorly_aligned_pairs, 1);
    }

    #[test]
    fn the_percentile_is_nearest_rank_and_never_invents_a_value() {
        let xs: Vec<f64> = (1..=10).map(|i| i as f64 * 10.0).collect();
        let d = deviation(&xs);
        assert_eq!(d.count, 10);
        assert!((d.mean_ms - 55.0).abs() < 1e-9);
        assert!((d.median_ms - 55.0).abs() < 1e-9);
        assert!((d.p90_ms - 90.0).abs() < 1e-9);
        assert!((d.max_abs_ms - 100.0).abs() < 1e-9);
    }
}

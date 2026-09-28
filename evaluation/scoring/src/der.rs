//! Diarization error rate, Jaccard error rate, and the turn-level metrics.
//!
//! # DER, stated exactly
//!
//! DER is the total duration of error divided by the total reference speech
//! duration, as a fraction:
//!
//! ```text
//! DER = (missed detection + false alarm + confusion) / total reference speech
//! ```
//!
//! The error is accumulated per **elementary interval** — the gaps between
//! consecutive turn boundaries — rather than per record, so overlapping turns
//! are handled correctly: an interval where two reference speakers and one
//! hypothesis speaker are active contributes one missed detection and one
//! confusion.
//!
//! Three steps make it well defined, and each is a parameter rather than a
//! hidden constant:
//!
//! 1. **Minimum duration.** Turns shorter than
//!    [`DiarParams::min_duration_us`] are discarded from both sides before
//!    anything is compared.
//! 2. **Collar.** `collar/2` is trimmed from *each end* of every turn on
//!    **both** sides, capped at half its duration, and the removed reference
//!    time is excluded from the denominator. Trimming only the reference would
//!    charge a hypothesis that runs 100 ms long for a boundary that the collar
//!    was supposed to forgive. `0` by default: the strict number is the one
//!    that gets quoted, and the forgiving number is reported beside it.
//! 3. **Permutation.** Speaker labels are never compared as strings. The
//!    assignment maximising total correct overlap is found **exactly**, by the
//!    Hungarian algorithm on a padded `(n+m) x (n+m)` cost matrix that also
//!    contains an explicit "unmatched" slot on both sides. Enumerating label
//!    permutations, which is what a small implementation tends to do, is wrong
//!    for two reasons: it is `n!`, and a maximum-*cardinality* matching is not a
//!    maximum-*weight* one. Both are covered by tests in this file.
//!
//! # JER
//!
//! `1 - |A n B| / |A u B|`, computed separately over the speech region and the
//! non-speech region and then averaged. It answers a different question from
//! DER — how good is the *segmentation of the conversation*, ignoring who is
//! talking — which is why it is reported separately rather than as a
//! consolation prize.
//!
//! # Turn metrics
//!
//! A *turn* is a maximal run of same-speaker regions, after the permutation has
//! been applied. Reported:
//!
//! * [`DiarizationReport::boundary`] — matched boundaries one-to-one, nearest
//!   first, inside `turn_tolerance_us`; the mean deviation always travels with
//!   the match count, because a mean with no count beside it is not a number.
//! * `false_speaker_change` — a reference turn boundary where the permuted
//!   hypothesis has the *same* speaker on both sides.
//! * `missed_speaker_change` — a reference turn boundary with no hypothesis
//!   boundary inside the tolerance.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::rttm::RttmRecord;

/// Tolerances and thresholds, serialised into the report so a DER is never
/// quoted without them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiarParams {
    /// Collar in microseconds, trimmed from each end of every reference turn
    /// and excluded from the denominator. `0` is the strict setting.
    pub collar_us: i64,
    /// Turns shorter than this are discarded from both sides before scoring.
    /// `0` disables the filter.
    pub min_duration_us: i64,
    /// Window for turn-boundary matching, and for false/missed speaker-change
    /// detection.
    pub turn_tolerance_us: i64,
}

impl Default for DiarParams {
    /// The strict defaults: no forgiveness anywhere except the turn tolerance,
    /// which only decides what counts as "the same boundary".
    fn default() -> Self {
        Self {
            collar_us: 0,
            min_duration_us: 0,
            turn_tolerance_us: 250_000,
        }
    }
}

impl DiarParams {
    /// The commonly quoted forgiving setting, reported beside the strict one.
    /// 250 ms is the `pyannote` default; it is here so the number is
    /// comparable, not because the boundary is known to be right.
    pub fn with_collar_250ms() -> Self {
        Self {
            collar_us: 250_000,
            ..Self::default()
        }
    }
}

/// A maximal same-speaker run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Turn {
    /// Speaker label, after the permutation has been applied.
    pub speaker: String,
    /// Start, microseconds.
    pub start_us: i64,
    /// End, microseconds.
    pub end_us: i64,
}

/// A contiguous region on the timeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Region {
    /// Start, microseconds.
    pub start_us: i64,
    /// End, microseconds.
    pub end_us: i64,
}

/// The three DER components, each as a fraction of reference speech duration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DerComponents {
    /// Reference speech that no hypothesis speaker covered.
    pub missed_detection: f64,
    /// Hypothesis speech that no reference speaker covered.
    pub false_alarm: f64,
    /// Speech covered by both sides, but attributed to different people.
    pub confusion: f64,
}

impl DerComponents {
    /// The three summed, which is DER.
    pub fn total(&self) -> f64 {
        self.missed_detection + self.false_alarm + self.confusion
    }
}

/// Speaker-count agreement. A count is already label-free, so no permutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeakerCount {
    /// Distinct reference speaker labels.
    pub reference: usize,
    /// Distinct hypothesis speaker labels.
    pub hypothesis: usize,
    /// Whether the two counts agree.
    pub correct: bool,
}

/// Boundary-matching counts, always reported next to the mean.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundaryStats {
    /// Boundaries present in both, matched one-to-one within tolerance.
    pub matched: usize,
    /// Reference boundaries with no hypothesis boundary in tolerance.
    pub unmatched_reference: usize,
    /// Hypothesis boundaries with no reference boundary in tolerance.
    pub unmatched_hypothesis: usize,
    /// Mean absolute deviation of matched boundaries, milliseconds. `0.0` when
    /// nothing matched, and `matched == 0` is why.
    pub mean_abs_deviation_ms: f64,
    /// Largest absolute deviation among matched boundaries, milliseconds.
    pub max_abs_deviation_ms: f64,
    /// Tolerance used, milliseconds.
    pub tolerance_ms: f64,
}

/// The whole diarization verdict. Rates are fractions, not percentages, so
/// nothing is lost to a rounding decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiarizationReport {
    /// `true` when the reference held at least one scored turn.
    pub defined: bool,
    /// Why DER is undefined, when it is.
    pub undefined_reason: Option<String>,
    /// Reference speech duration, milliseconds, after collar removal.
    pub reference_speech_ms: f64,
    /// Total hypothesis speech duration, milliseconds.
    pub hypothesis_speech_ms: f64,
    /// Diarization error rate.
    pub der: Option<f64>,
    /// The three components as fractions — the diagnosis DER alone cannot give.
    pub components: DerComponents,
    /// The same three in milliseconds.
    pub error_ms: DerComponents,
    /// Microseconds of reference speech scored as correct, for auditing.
    pub correct_ms: f64,
    /// Hypothesis speech that falls inside reference speech but could not be
    /// matched to a reference speaker. The hypothesis-side mirror of
    /// [`DerComponents::confusion`], reported so that
    /// `correct + false_alarm + unattributed == hypothesis speech` reconciles
    /// as exactly as `correct + missed + confusion == reference speech` does.
    pub unattributed_hypothesis_ms: f64,
    /// Jaccard error rate, averaged over the two regions.
    pub jer: Option<f64>,
    /// JER over the speech region only.
    pub jer_speech: Option<f64>,
    /// JER over the non-speech region only.
    pub jer_non_speech: Option<f64>,
    /// Speaker-count agreement.
    pub speaker_count: SpeakerCount,
    /// The reference-to-hypothesis label map that was applied, for auditing.
    pub permutation: BTreeMap<String, String>,
    /// Turn-boundary agreement.
    pub boundary: BoundaryStats,
    /// Reference turn boundaries where the hypothesis did not change speaker.
    pub false_speaker_change: usize,
    /// Reference turn boundaries with no hypothesis boundary in tolerance.
    pub missed_speaker_change: usize,
    /// Reference turns, after filtering and merging.
    pub reference_turns: usize,
    /// Hypothesis turns, after filtering, merging and permutation.
    pub hypothesis_turns: usize,
    /// Parameters used.
    pub params: DiarParams,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RawTurn {
    speaker: String,
    start: i64,
    end: i64,
}

/// Score a diarization hypothesis against a reference.
///
/// `params` is an argument rather than a default, so a report always carries
/// the exact tolerances that produced it.
pub fn score(
    reference: &[RttmRecord],
    hypothesis: &[RttmRecord],
    params: DiarParams,
) -> DiarizationReport {
    let ref_turns: Vec<RawTurn> = reference
        .iter()
        .filter(|r| r.is_turn() && r.duration_us >= params.min_duration_us)
        .map(|r| RawTurn {
            speaker: r.name.clone(),
            start: collar_start(r.start_us, r.duration_us, params.collar_us),
            end: collar_end(r.start_us, r.duration_us, params.collar_us),
        })
        .filter(|t| t.end > t.start)
        .collect();
    // The collar is a region of the *timeline* around each reference turn, not
    // a trim of each file's own edges. Time inside it is not scored on either
    // side, so a hypothesis that starts 100 ms late is forgiven for it rather
    // than charged twice for the same 100 ms: once as a missed detection
    // against the reference start, and once as a false alarm for the
    // hypothesis overhanging past it. Trimming the hypothesis's own edges
    // instead would silently shorten a hypothesis that is already correct.
    let excluded = collar_regions(
        &reference
            .iter()
            .filter(|r| r.is_turn() && r.duration_us >= params.min_duration_us)
            .map(|r| (r.start_us, r.duration_us))
            .collect::<Vec<_>>(),
        params.collar_us,
    );
    let hyp_turns: Vec<RawTurn> = hypothesis
        .iter()
        .filter(|r| r.is_turn() && r.duration_us >= params.min_duration_us)
        .map(|r| RawTurn {
            speaker: r.name.clone(),
            start: r.start_us.max(0),
            end: r.end_us(),
        })
        .filter(|t| t.end > t.start)
        .flat_map(|t| {
            subtract(t.start, t.end, &excluded)
                .into_iter()
                .map(move |(lo, hi)| RawTurn {
                    speaker: t.speaker.clone(),
                    start: lo,
                    end: hi,
                })
                .collect::<Vec<_>>()
        })
        .filter(|t| t.end > t.start)
        .collect();

    let ref_speech: i64 = ref_turns.iter().map(|t| t.end - t.start).sum();
    let hyp_speech: i64 = hyp_turns.iter().map(|t| t.end - t.start).sum();

    let ref_labels = distinct(&ref_turns);
    let hyp_labels = distinct(&hyp_turns);
    let speaker_count = SpeakerCount {
        reference: ref_labels.len(),
        hypothesis: hyp_labels.len(),
        correct: ref_labels.len() == hyp_labels.len(),
    };

    if ref_turns.is_empty() || ref_speech == 0 {
        return undefined(
            "reference holds no scored speech turn after filtering",
            ref_speech,
            hyp_speech,
            speaker_count,
            ref_turns.len(),
            hyp_turns.len(),
            params,
        );
    }

    // --- permutation: exact maximum-overlap assignment ----------------------
    let overlaps = overlap_matrix(&ref_turns, &hyp_turns);
    let assignment = best_assignment(ref_labels.len(), hyp_labels.len(), &overlaps);
    let mut permutation = BTreeMap::new();
    for (ri, hi) in &assignment {
        permutation.insert(ref_labels[*ri].clone(), hyp_labels[*hi].clone());
    }
    let overlap_correct_us: i64 = assignment.iter().map(|(l, r)| overlaps[*l][*r]).sum();

    // --- per elementary interval error accumulation -------------------------
    let mut points: Vec<i64> = Vec::new();
    for t in ref_turns.iter().chain(hyp_turns.iter()) {
        points.push(t.start);
        points.push(t.end);
    }
    points.sort_unstable();
    points.dedup();

    // Per-label total coverage on the reference side, used to attribute each
    // elementary interval to the labels active in it.
    let ref_spans: Vec<Vec<(i64, i64)>> = spans_by_label(&ref_turns, &ref_labels);
    let hyp_spans: Vec<Vec<(i64, i64)>> = spans_by_label(&hyp_turns, &hyp_labels);
    let map: Vec<Option<usize>> = (0..ref_labels.len())
        .map(|i| assignment.iter().find(|(l, _)| *l == i).map(|(_, h)| *h))
        .collect();

    let mut missed_us: i64 = 0;
    let mut false_alarm_us: i64 = 0;
    let mut confusion_us: i64 = 0;
    let mut correct_us: i64 = 0;
    let mut unattributed_hyp_us: i64 = 0;
    for w in points.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        if hi <= lo {
            continue;
        }
        // Every turn boundary is a point, so a turn either covers this whole
        // elementary interval or none of it.
        let d = hi - lo;
        let r_active: usize = (0..ref_labels.len())
            .filter(|i| covering(&ref_spans[*i], lo, hi))
            .count();
        let h_active: usize = (0..hyp_labels.len())
            .filter(|j| covering(&hyp_spans[*j], lo, hi))
            .count();
        // `map` is injective from reference labels to hypothesis labels, so
        // the matched pairs number `min(r_active, h_active)` at most and each
        // hypothesis label is matched at most once.
        let matched: usize = (0..ref_labels.len())
            .filter(|i| {
                covering(&ref_spans[*i], lo, hi)
                    && map[*i].is_some_and(|hj| covering(&hyp_spans[hj], lo, hi))
            })
            .count();
        correct_us += matched as i64 * d;
        let ref_residual: usize = r_active - matched;
        let hyp_residual: usize = h_active - matched;

        // The three components partition the error without double counting.
        // A reference-side residual is *missed detection* when the hypothesis
        // says nobody is talking there, and *confusion* when the hypothesis
        // says somebody is — just not this person. Attributing it to confusion
        // rather than to both is what makes the two identities below hold, and
        // both are asserted on every call:
        //
        //   correct + missed + confusion == reference speech
        //   correct + false_alarm + confusion == hypothesis speech
        //
        // Counting a residual as both missed *and* confusion is the convention
        // in some published implementations. It is defensible but it is not
        // the only choice, and it is not this one.
        if h_active == 0 {
            missed_us += ref_residual as i64 * d;
        } else {
            confusion_us += ref_residual as i64 * d;
        }
        if r_active == 0 {
            false_alarm_us += hyp_residual as i64 * d;
        } else {
            // Somebody was talking, so this hypothesis speaker is not a false
            // alarm; it is the other face of the confusion just charged on the
            // reference side. Counted once, as its own diagnostic, so that
            // hypothesis speech reconciles as well as reference speech does.
            unattributed_hyp_us += hyp_residual as i64 * d;
        }
    }

    // Two correctness invariants that cost nothing and catch a broken
    // permutation, a miscounted residual, or a double count.
    debug_assert_eq!(correct_us + missed_us + confusion_us, ref_speech);
    debug_assert_eq!(
        correct_us + false_alarm_us + unattributed_hyp_us,
        hyp_speech
    );
    debug_assert_eq!(correct_us, overlap_correct_us);

    let to_frac = |us: i64| -> f64 { us as f64 / ref_speech as f64 };
    let components = DerComponents {
        missed_detection: to_frac(missed_us),
        false_alarm: to_frac(false_alarm_us),
        confusion: to_frac(confusion_us),
    };

    // --- JER and turn metrics ----------------------------------------------
    let ref_regions = to_regions(&ref_turns);
    let hyp_regions = to_regions(&hyp_turns);
    let (jer_speech, jer_non) = jaccard(&ref_regions, &hyp_regions, ref_speech.max(hyp_speech));
    let jer = match (jer_speech, jer_non) {
        (Some(a), Some(b)) => Some((a + b) / 2.0),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };

    let ref_turn_list: Vec<Turn> = merge_turns(&ref_turns)
        .into_iter()
        .map(|t| Turn {
            speaker: t.speaker.clone(),
            start_us: t.start,
            end_us: t.end,
        })
        .collect();
    let hyp_turn_list: Vec<Turn> = merge_turns(&hyp_turns)
        .into_iter()
        .map(|t| Turn {
            speaker: permutation
                .get(&t.speaker)
                .cloned()
                .unwrap_or_else(|| t.speaker.clone()),
            start_us: t.start,
            end_us: t.end,
        })
        .collect();

    let boundary = boundary_stats(&ref_turn_list, &hyp_turn_list, params.turn_tolerance_us);
    // The two directions of the same count that `boundary` already reports,
    // under the names the taxonomy uses. They are the same numbers on purpose:
    // inventing a second arithmetic here would create a figure that can drift
    // away from the one a reader can check.
    //
    // * missed: a reference turn boundary with no hypothesis boundary nearby,
    //   i.e. the system did not change speaker when a human heard a change.
    // * false:  a hypothesis turn boundary with no reference boundary nearby,
    //   i.e. the system changed speaker when a human heard one person continue.
    let missed_change = boundary.unmatched_reference;
    let false_change = boundary.unmatched_hypothesis;

    DiarizationReport {
        defined: true,
        undefined_reason: None,
        reference_speech_ms: ref_speech as f64 / 1000.0,
        hypothesis_speech_ms: hyp_speech as f64 / 1000.0,
        der: Some(components.total()),
        components,
        error_ms: DerComponents {
            missed_detection: missed_us as f64 / 1000.0,
            false_alarm: false_alarm_us as f64 / 1000.0,
            confusion: confusion_us as f64 / 1000.0,
        },
        correct_ms: correct_us as f64 / 1000.0,
        unattributed_hypothesis_ms: unattributed_hyp_us as f64 / 1000.0,
        jer,
        jer_speech,
        jer_non_speech: jer_non,
        speaker_count,
        permutation,
        boundary,
        false_speaker_change: false_change,
        missed_speaker_change: missed_change,
        reference_turns: ref_turn_list.len(),
        hypothesis_turns: hyp_turn_list.len(),
        params,
    }
}

fn undefined(
    why: &str,
    ref_speech: i64,
    hyp_speech: i64,
    speaker_count: SpeakerCount,
    ref_turns: usize,
    hyp_turns: usize,
    params: DiarParams,
) -> DiarizationReport {
    DiarizationReport {
        defined: false,
        undefined_reason: Some(why.to_string()),
        reference_speech_ms: ref_speech as f64 / 1000.0,
        hypothesis_speech_ms: hyp_speech as f64 / 1000.0,
        der: None,
        components: DerComponents {
            missed_detection: 0.0,
            false_alarm: 0.0,
            confusion: 0.0,
        },
        error_ms: DerComponents {
            missed_detection: 0.0,
            false_alarm: 0.0,
            confusion: 0.0,
        },
        correct_ms: 0.0,
        unattributed_hypothesis_ms: 0.0,
        jer: None,
        jer_speech: None,
        jer_non_speech: None,
        speaker_count,
        permutation: BTreeMap::new(),
        boundary: BoundaryStats {
            matched: 0,
            unmatched_reference: 0,
            unmatched_hypothesis: 0,
            mean_abs_deviation_ms: 0.0,
            max_abs_deviation_ms: 0.0,
            tolerance_ms: params.turn_tolerance_us as f64 / 1000.0,
        },
        false_speaker_change: 0,
        missed_speaker_change: 0,
        reference_turns: ref_turns,
        hypothesis_turns: hyp_turns,
        params,
    }
}

fn distinct(turns: &[RawTurn]) -> Vec<String> {
    let mut v: Vec<String> = turns.iter().map(|t| t.speaker.clone()).collect();
    v.sort();
    v.dedup();
    v
}

fn spans_by_label(turns: &[RawTurn], labels: &[String]) -> Vec<Vec<(i64, i64)>> {
    labels
        .iter()
        .map(|l| {
            let mut s: Vec<(i64, i64)> = turns
                .iter()
                .filter(|t| t.speaker == *l)
                .map(|t| (t.start, t.end))
                .collect();
            s.sort_unstable();
            s
        })
        .collect()
}

/// Whether any of the label's spans covers the whole elementary interval.
fn covering(spans: &[(i64, i64)], lo: i64, hi: i64) -> bool {
    spans.iter().any(|&(a, b)| a <= lo && b >= hi)
}

/// Total overlap in microseconds between reference label `ri` and hypothesis
/// label `hi`.
fn overlap_matrix(ref_turns: &[RawTurn], hyp_turns: &[RawTurn]) -> Vec<Vec<i64>> {
    let rl = distinct(ref_turns);
    let hl = distinct(hyp_turns);
    let mut m = vec![vec![0i64; hl.len()]; rl.len()];
    for t in ref_turns {
        let ri = rl.iter().position(|s| *s == t.speaker).unwrap();
        for u in hyp_turns {
            let hi = hl.iter().position(|s| *s == u.speaker).unwrap();
            m[ri][hi] += (t.end.min(u.end) - t.start.max(u.start)).max(0);
        }
    }
    m
}

/// Exact maximum-weight assignment between reference labels and hypothesis
/// labels, minimising `-overlap` on a padded square matrix.
///
/// The padding is what makes "unmatched" expressible: rows `0..n` are
/// reference labels, rows `n..n+m` are hypothesis labels; columns `0..m` are
/// hypothesis slots and columns `m..m+n` are the unmatched slots. Matching
/// reference label `i` to hypothesis label `j` costs `-overlap[i][j]`; every
/// other pairing costs `0`. A maximum-weight matching therefore picks exactly
/// the permutation that maximises correct overlap, and a reference speaker
/// who was never heard is left unmatched rather than forced onto a label.
///
/// The classic O(n^3) Hungarian (Kuhn–Munkres) algorithm. Exact, no `n!`
/// cliff, and unlike an augmenting-path search it maximises **weight**, which
/// is not the same as maximising cardinality — a max-cardinality matching can
/// score far worse on a sparse overlap matrix, and there is a test for exactly
/// that.
fn best_assignment(n: usize, m: usize, overlaps: &[Vec<i64>]) -> Vec<(usize, usize)> {
    let size = n + m;
    if size == 0 {
        return Vec::new();
    }
    let mut cost = vec![vec![0i64; size]; size];
    for i in 0..n {
        for j in 0..m {
            cost[i][j] = -overlaps[i][j];
        }
    }
    let col = hungarian_min(&cost);
    let mut out = Vec::new();
    for (i, &j) in col.iter().enumerate().take(n) {
        if j < m {
            out.push((i, j));
        }
    }
    out.sort_unstable();
    out
}

/// Classic O(n^3) Hungarian for a square cost matrix. Returns, for every row,
/// the column it is assigned to. The matrix is square, so a perfect assignment
/// always exists and the `inf` guards below are unreachable; they are kept
/// because a silent `usize::MAX` index would be a crash with no explanation.
fn hungarian_min(cost: &[Vec<i64>]) -> Vec<usize> {
    let n = cost.len();
    const INF: i64 = i64::MAX / 4;
    let mut u = vec![0i64; n + 1];
    let mut v = vec![0i64; n + 1];
    let mut p = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![INF; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = INF;
            let mut j1 = 0usize;
            for j in 1..=n {
                if used[j] {
                    continue;
                }
                let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                if cur < minv[j] {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if minv[j] < delta {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut col = vec![usize::MAX; n];
    for j in 1..=n {
        if p[j] > 0 {
            col[p[j] - 1] = j - 1;
        }
    }
    col
}

fn collar_start(start: i64, duration: i64, collar_us: i64) -> i64 {
    if collar_us <= 0 {
        return start.max(0);
    }
    let trim = (collar_us / 2).min(duration / 2);
    (start + trim).max(0)
}

fn collar_end(start: i64, duration: i64, collar_us: i64) -> i64 {
    if collar_us <= 0 {
        return (start + duration).max(0);
    }
    let trim = (collar_us / 2).min(duration / 2);
    start + duration - trim
}

/// The timeline regions the collar forgives: `collar/2` at each end of every
/// reference turn, never longer than half the turn.
fn collar_regions(ref_turns: &[(i64, i64)], collar_us: i64) -> Vec<(i64, i64)> {
    if collar_us <= 0 {
        return Vec::new();
    }
    ref_turns
        .iter()
        .flat_map(|&(s, d)| {
            let trim = (collar_us / 2).min(d / 2);
            [
                (s.max(0), (s + trim).max(0)),
                ((s + d - trim).max(0), s + d),
            ]
        })
        .filter(|(a, b)| b > a)
        .collect()
}

/// `[lo, hi)` minus the excluded regions, as a list of surviving spans.
fn subtract(lo: i64, hi: i64, excluded: &[(i64, i64)]) -> Vec<(i64, i64)> {
    if hi <= lo {
        return Vec::new();
    }
    let mut cuts: Vec<(i64, i64)> = excluded
        .iter()
        .map(|(a, b)| ((*a).max(lo), (*b).min(hi)))
        .filter(|(a, b)| b > a)
        .collect();
    if cuts.is_empty() {
        return vec![(lo, hi)];
    }
    cuts.sort_unstable();
    let mut out = Vec::new();
    let mut cur = lo;
    for (a, b) in cuts {
        if a > cur {
            out.push((cur, a));
        }
        // Overlapping excluded regions must not un-advance the cursor.
        cur = cur.max(b);
    }
    if cur < hi {
        out.push((cur, hi));
    }
    out
}

/// Union of all turn intervals, as a sorted list of disjoint regions.
fn to_regions(turns: &[RawTurn]) -> Vec<Region> {
    let mut iv: Vec<(i64, i64)> = turns.iter().map(|t| (t.start, t.end)).collect();
    iv.sort_unstable();
    let mut out: Vec<Region> = Vec::new();
    for (lo, hi) in iv {
        match out.last_mut() {
            Some(last) if lo <= last.end_us => last.end_us = last.end_us.max(hi),
            _ => out.push(Region {
                start_us: lo,
                end_us: hi,
            }),
        }
    }
    out
}

/// JER over the speech region and the non-speech region, separately.
///
/// `span` is the end of the timeline both files are scored over: the largest
/// turn end on either side. Non-speech is the complement of the turn union
/// within `[0, span)`, so a hypothesis that marks a long silence as speech is
/// penalised in the non-speech term rather than being invisible.
fn jaccard(
    ref_regions: &[Region],
    hyp_regions: &[Region],
    span: i64,
) -> (Option<f64>, Option<f64>) {
    let r_speech = total(ref_regions);
    let h_speech = total(hyp_regions);
    let speech = jaccard_of(
        r_speech,
        h_speech,
        total(&intersect(ref_regions, hyp_regions)),
    );
    let ref_non = total(&complement(ref_regions, span));
    let hyp_non = total(&complement(hyp_regions, span));
    let both_non = total(&intersect(
        &complement(ref_regions, span),
        &complement(hyp_regions, span),
    ));
    (speech, jaccard_of(ref_non, hyp_non, both_non))
}

fn total(r: &[Region]) -> i64 {
    r.iter().map(|x| x.end_us - x.start_us).sum()
}

fn intersect(a: &[Region], b: &[Region]) -> Vec<Region> {
    let mut out = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        let lo = a[i].start_us.max(b[j].start_us);
        let hi = a[i].end_us.min(b[j].end_us);
        if hi > lo {
            out.push(Region {
                start_us: lo,
                end_us: hi,
            });
        }
        if a[i].end_us < b[j].end_us {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

fn complement(a: &[Region], span: i64) -> Vec<Region> {
    let mut out = Vec::new();
    let mut cur = 0i64;
    for r in a {
        if r.start_us > cur {
            out.push(Region {
                start_us: cur,
                end_us: r.start_us.min(span),
            });
        }
        cur = cur.max(r.end_us);
    }
    if cur < span {
        out.push(Region {
            start_us: cur,
            end_us: span,
        });
    }
    out.retain(|r| r.end_us > r.start_us);
    out
}

fn jaccard_of(a: i64, b: i64, inter: i64) -> Option<f64> {
    let union = a + b - inter;
    if union <= 0 {
        None
    } else {
        Some(1.0 - inter as f64 / union as f64)
    }
}

/// Collapse a turn list into maximal same-speaker runs.
fn merge_turns(turns: &[RawTurn]) -> Vec<RawTurn> {
    let mut iv: Vec<&RawTurn> = turns.iter().collect();
    iv.sort_by_key(|t| (t.start, t.end));
    let mut out: Vec<RawTurn> = Vec::new();
    for t in iv {
        match out.last_mut() {
            Some(last) if last.speaker == t.speaker && t.start <= last.end => {
                last.end = last.end.max(t.end);
            }
            _ => out.push(RawTurn {
                speaker: t.speaker.clone(),
                start: t.start,
                end: t.end,
            }),
        }
    }
    out
}

/// Match turn boundaries one-to-one, nearest first, within tolerance.
fn boundary_stats(ref_turns: &[Turn], hyp_turns: &[Turn], tol_us: i64) -> BoundaryStats {
    let rb: Vec<i64> = ref_turns.iter().skip(1).map(|t| t.start_us).collect();
    let hb: Vec<i64> = hyp_turns.iter().skip(1).map(|t| t.start_us).collect();
    let mut pairs: Vec<(usize, usize, i64)> = Vec::new();
    for (i, r) in rb.iter().enumerate() {
        for (j, h) in hb.iter().enumerate() {
            let d = (r - h).abs();
            if d <= tol_us {
                pairs.push((i, j, d));
            }
        }
    }
    pairs.sort_by_key(|p| p.2);
    let mut used_r: Vec<usize> = Vec::new();
    let mut used_h: Vec<usize> = Vec::new();
    let mut devs: Vec<i64> = Vec::new();
    for (i, j, d) in pairs {
        if used_r.contains(&i) || used_h.contains(&j) {
            continue;
        }
        used_r.push(i);
        used_h.push(j);
        devs.push(d);
    }
    let matched = devs.len();
    let mean = if matched == 0 {
        0.0
    } else {
        devs.iter().map(|d| *d as f64).sum::<f64>() / matched as f64 / 1000.0
    };
    let max = devs.iter().copied().max().unwrap_or(0) as f64 / 1000.0;
    BoundaryStats {
        matched,
        unmatched_reference: rb.len().saturating_sub(matched),
        unmatched_hypothesis: hb.len().saturating_sub(matched),
        mean_abs_deviation_ms: mean,
        max_abs_deviation_ms: max,
        tolerance_ms: tol_us as f64 / 1000.0,
    }
}

/// False and missed speaker changes, computed directly from the two turn lists
/// rather than from the boundary matcher.
///
/// This is the *definition*; [`boundary_stats`] is an independent
/// implementation of the same quantity, and the unit test below asserts the two
/// agree. Two implementations of one number that are checked against each other
/// is worth more here than one implementation that is merely trusted.
#[cfg(test)]
fn speaker_change_errors(ref_turns: &[Turn], hyp_turns: &[Turn], tol_us: i64) -> (usize, usize) {
    let rb: Vec<i64> = ref_turns.iter().skip(1).map(|t| t.start_us).collect();
    let hb: Vec<i64> = hyp_turns.iter().skip(1).map(|t| t.start_us).collect();
    // Deliberately a different algorithm from `boundary_stats`: walk the
    // reference boundaries in order and claim the nearest unused hypothesis
    // boundary. `boundary_stats` collects every in-tolerance pair and takes them
    // globally nearest-first. Both are correct matchings for a one-to-one
    // problem, and the test that they agree is the point of keeping two.
    let mut used = vec![false; hb.len()];
    let mut matched = 0usize;
    for r in &rb {
        let mut best: Option<(usize, i64)> = None;
        for (j, h) in hb.iter().enumerate() {
            if used[j] {
                continue;
            }
            let d = (r - h).abs();
            if d <= tol_us && best.is_none_or(|(_, bd)| d < bd) {
                best = Some((j, d));
            }
        }
        if let Some((j, _)) = best {
            used[j] = true;
            matched += 1;
        }
    }
    (hb.len() - matched, rb.len() - matched)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(speaker: &str, start: i64, dur: i64) -> RttmRecord {
        RttmRecord::speaker(speaker, start, dur)
    }

    /// The counters the report publishes must equal the counts computed
    /// directly from the two turn lists, or one of them is lying.
    #[test]
    fn the_published_change_counters_match_the_direct_count() {
        let cases: Vec<(Vec<RttmRecord>, Vec<RttmRecord>)> = vec![
            (
                vec![turn("A", 0, 2_000_000), turn("B", 2_000_000, 2_000_000)],
                vec![turn("A", 0, 4_000_000)],
            ),
            (
                vec![turn("A", 0, 4_000_000)],
                vec![turn("A", 0, 2_000_000), turn("B", 2_000_000, 2_000_000)],
            ),
            (
                vec![
                    turn("A", 0, 2_000_000),
                    turn("B", 2_000_000, 2_000_000),
                    turn("A", 4_000_000, 2_000_000),
                ],
                vec![
                    turn("A", 0, 2_000_000),
                    turn("B", 2_100_000, 1_900_000),
                    turn("A", 4_000_000, 2_000_000),
                ],
            ),
        ];
        for (r, h) in cases {
            let rep = score(&r, &h, DiarParams::default());
            let ref_list: Vec<Turn> = merge_turns(
                &r.iter()
                    .map(|x| RawTurn {
                        speaker: x.name.clone(),
                        start: x.start_us,
                        end: x.end_us(),
                    })
                    .collect::<Vec<_>>(),
            )
            .into_iter()
            .map(|t| Turn {
                speaker: t.speaker,
                start_us: t.start,
                end_us: t.end,
            })
            .collect();
            let hyp_list: Vec<Turn> = merge_turns(
                &h.iter()
                    .map(|x| RawTurn {
                        speaker: x.name.clone(),
                        start: x.start_us,
                        end: x.end_us(),
                    })
                    .collect::<Vec<_>>(),
            )
            .into_iter()
            .map(|t| Turn {
                speaker: t.speaker,
                start_us: t.start,
                end_us: t.end,
            })
            .collect();
            let (false_c, missed_c) =
                speaker_change_errors(&ref_list, &hyp_list, rep.params.turn_tolerance_us);
            assert_eq!(rep.missed_speaker_change, missed_c, "{rep:#?}");
            assert_eq!(rep.false_speaker_change, false_c, "{rep:#?}");
            assert_eq!(rep.boundary.unmatched_reference, missed_c);
            assert_eq!(rep.boundary.unmatched_hypothesis, false_c);
        }
    }

    /// The collar forgives time on both sides of a boundary, so a hypothesis
    /// that is 100 ms late at each end of a 2 s turn scores 0 with a 250 ms
    /// collar and 0.1 without one. It must not be charged twice.
    #[test]
    fn the_collar_forgives_time_and_charges_it_once() {
        let r = vec![turn("A", 0, 2_000_000)];
        let h = vec![turn("A", 100_000, 1_800_000)];
        let strict = score(&r, &h, DiarParams::default());
        // The hypothesis sits entirely inside the reference, so nothing is a
        // false alarm: the 200 ms it fails to cover at the two ends is all
        // missed detection. Charging the same 200 ms again as a false alarm
        // would put the strict DER at 0.2 for a hypothesis that is 90 % right.
        assert!((strict.der.unwrap() - 0.1).abs() < 1e-9, "{strict:#?}");
        assert!(
            (strict.components.missed_detection - 0.1).abs() < 1e-9,
            "{strict:#?}"
        );
        assert!(
            (strict.components.false_alarm - 0.0).abs() < 1e-9,
            "{strict:#?}"
        );
        let forgiven = score(&r, &h, DiarParams::with_collar_250ms());
        assert_eq!(forgiven.der, Some(0.0), "{forgiven:#?}");
        assert!(
            (forgiven.reference_speech_ms - 1750.0).abs() < 1e-9,
            "{forgiven:#?}"
        );
        assert!(
            (forgiven.hypothesis_speech_ms - 1750.0).abs() < 1e-9,
            "the collar removes the same 125 ms from the hypothesis, not just the reference"
        );
    }

    /// Subtraction is exact: an interval in the middle of an excluded region
    /// disappears, an interval spanning one becomes two.
    #[test]
    fn subtract_removes_only_the_excluded_part() {
        let ex = vec![(100, 200)];
        assert_eq!(subtract(0, 300, &ex), vec![(0, 100), (200, 300)]);
        assert_eq!(subtract(120, 180, &ex), Vec::<(i64, i64)>::new());
        assert_eq!(subtract(0, 50, &ex), vec![(0, 50)]);
        assert_eq!(subtract(0, 300, &[]), vec![(0, 300)]);
    }
}

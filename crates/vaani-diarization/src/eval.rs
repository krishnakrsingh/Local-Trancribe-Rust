//! Ground-truth labelling and measured quality, shared by every caller.
//!
//! # Why this module exists
//!
//! The synthetic end-to-end test and the tuning example both need to answer
//! "how far apart are same-speaker windows from different-speaker windows, and
//! did the clustering get it right". When those two answers are computed in two
//! places they *will* drift, and when they drift the only visible symptom is
//! that the diagnostic and the test disagree -- which reads as "the engine is
//! behaving inconsistently" and sends you tuning against a fiction.
//!
//! That happened. The test reimplemented labelling from the utterance list,
//! discarding the per-frame reference [`crate::synth::render`] had already
//! handed back, while the example inferred speaker identity from turn parity.
//! They excluded different sets of windows, so the example's `different p50`
//! of exactly 0.0 was an artefact of boundary windows being labelled by the
//! turn they *started* in. The engine was fine; the two measurements were not
//! measuring the same thing.
//!
//! So the labelling happens here, once, from the synthesiser's own per-frame
//! reference, and both callers go through it.
//!
//! # The rule
//!
//! A window is **pure** when every labelled frame inside it belongs to one
//! speaker. Windows that straddle a turn boundary have no single correct answer
//! and are excluded from every geometry and quality statistic -- not because
//! they are inconvenient but because scoring a window against a label it does
//! not have measures the fixture, not the engine.
//!
//! Silence is *not* a reason to exclude a window. A window that starts before
//! the speech does is a completely ordinary window and is kept, because the
//! engine sees it too and has to cope with it.

use crate::cosine_distance;
use crate::engine::LabelledWindow;

/// The label `crate::synth::render` writes for a frame with no speech.
pub const NO_SPEAKER: i32 = -1;

/// A window's ground-truth speaker, and whether it is unambiguous.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowLabel {
    /// The speaker owning the window's labelled frames, or [`NO_SPEAKER`].
    pub speaker: i32,
    /// Labelled frames in the window.
    pub labelled_frames: usize,
    /// Of those, how many agree with `speaker`.
    pub agreeing_frames: usize,
    /// True when the window holds exactly one speaker and at least one frame of
    /// that speaker. Only these windows are scored.
    pub pure: bool,
}

impl WindowLabel {
    /// Fraction of the window's labelled frames that agree with `speaker`.
    #[must_use]
    pub fn agreement(&self) -> f64 {
        if self.labelled_frames == 0 {
            return 0.0;
        }
        self.agreeing_frames as f64 / self.labelled_frames as f64
    }
}

/// Label each window from the per-frame reference.
///
/// `truth` is the second return value of [`crate::synth::render`]: one label per
/// `frame_ms` of audio from the start of the buffer, [`NO_SPEAKER`] where
/// nothing was rendered. Windows are matched to frames by their absolute
/// start time, so a buffer with a non-zero `Time` origin still lines up as long
/// as the reference was built for the same buffer.
#[must_use]
pub fn label_windows(
    windows: &[LabelledWindow],
    truth: &[i32],
    frame_ms: i64,
    start_ms: i64,
) -> Vec<WindowLabel> {
    windows
        .iter()
        .map(|w| {
            let lo = ((w.start.as_millis() - start_ms) / frame_ms.max(1)).max(0) as usize;
            let hi = (((w.end.as_millis() - start_ms) / frame_ms.max(1)).max(0) as usize)
                .min(truth.len())
                .max(lo);
            let mut counts: std::collections::BTreeMap<i32, usize> =
                std::collections::BTreeMap::new();
            for t in truth.get(lo..hi).unwrap_or(&[]) {
                if *t != NO_SPEAKER {
                    *counts.entry(*t).or_default() += 1;
                }
            }
            let labelled: usize = counts.values().sum();
            // Ties break toward the lower speaker id, so labelling is
            // deterministic and independent of iteration order.
            let best = counts
                .iter()
                .max_by_key(|(s, c)| (**c, -(**s)))
                .map(|(s, c)| (*s, *c))
                .unwrap_or((NO_SPEAKER, 0));
            WindowLabel {
                speaker: best.0,
                labelled_frames: labelled,
                agreeing_frames: best.1,
                pure: labelled > 0 && best.1 == labelled,
            }
        })
        .collect()
}

/// The five-number summary plus a mean, for a population of distances.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Percentiles {
    /// Number of samples.
    pub n: usize,
    /// Smallest sample.
    pub min: f64,
    /// 10th percentile.
    pub p10: f64,
    /// Median.
    pub p50: f64,
    /// 90th percentile.
    pub p90: f64,
    /// Largest sample.
    pub max: f64,
    /// Arithmetic mean.
    pub mean: f64,
}

impl Percentiles {
    /// Summarise an unsorted population. Empty input is all `NAN`, which
    /// compares false to everything and so cannot silently pass a threshold.
    #[must_use]
    pub fn of(mut values: Vec<f64>) -> Self {
        if values.is_empty() {
            return Self {
                n: 0,
                min: f64::NAN,
                p10: f64::NAN,
                p50: f64::NAN,
                p90: f64::NAN,
                max: f64::NAN,
                mean: f64::NAN,
            };
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let at = |p: f64| {
            values[((p * (values.len() - 1) as f64).round() as usize).min(values.len() - 1)]
        };
        Self {
            n: values.len(),
            min: values[0],
            p10: at(0.10),
            p50: at(0.50),
            p90: at(0.90),
            max: values[values.len() - 1],
            mean: values.iter().sum::<f64>() / values.len() as f64,
        }
    }
}

/// How separable two populations of windows are.
#[derive(Debug, Clone, PartialEq)]
pub struct Geometry {
    /// Pure windows held.
    pub pure_windows: usize,
    /// Windows dropped for straddling a turn boundary.
    pub mixed_windows: usize,
    /// Same-speaker pairwise distances.
    pub same: Percentiles,
    /// Different-speaker pairwise distances.
    pub different: Percentiles,
    /// `different.p10 - same.p90`. Positive means the two populations are
    /// separated by a gap, and its size is the honest headroom the merge
    /// distance has to fit into. This is the number to look at.
    pub margin: f64,
}

/// Measure the same/different distance populations over pure windows only.
#[must_use]
pub fn measure_geometry(windows: &[LabelledWindow], labels: &[WindowLabel]) -> Geometry {
    let pure: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter(|(_, l)| l.pure)
        .map(|(i, _)| i)
        .collect();
    let mut same = Vec::new();
    let mut different = Vec::new();
    for (x, &i) in pure.iter().enumerate() {
        for &j in &pure[x + 1..] {
            if windows[i].vector.len() != windows[j].vector.len() {
                continue;
            }
            let d = cosine_distance(&windows[i].vector, &windows[j].vector);
            if labels[i].speaker == labels[j].speaker {
                same.push(d);
            } else {
                different.push(d);
            }
        }
    }
    let s = Percentiles::of(same);
    let d = Percentiles::of(different);
    Geometry {
        pure_windows: pure.len(),
        mixed_windows: labels.len() - pure.len(),
        margin: d.p10 - s.p90,
        same: s,
        different: d,
    }
}

/// Quality of a clustering against the ground truth, over pure windows only.
#[derive(Debug, Clone, PartialEq)]
pub struct Quality {
    /// Windows scored.
    pub scored: usize,
    /// Fraction of scored windows in their own speaker's dominant cluster.
    pub purity: f64,
    /// Fraction of scored windows whose speaker is in any cluster at all.
    pub recall: f64,
    /// Fraction of scored windows whose cluster is *uniquely* the best cluster
    /// for their speaker. The strictest figure here.
    pub complete_accuracy: f64,
    /// Complete-linkage separation: the largest same-group cross pair, the
    /// largest cross-group pair, their 10th percentile, and the mean.
    pub complete_link_max: f64,
    /// Smallest cross-speaker distance.
    pub complete_link_min: f64,
    /// 10th percentile of cross-speaker distances.
    pub complete_link_p10: f64,
    /// Mean cross-speaker distance.
    pub complete_link_mean: f64,
    /// 90th percentile of the same-speaker population.
    pub within_speaker_p90: f64,
    /// Geometry, so one call reports both the outcome and the evidence.
    pub geometry: Geometry,
}

/// Score the clustering the engine produced against the ground truth.
#[must_use]
pub fn measure_quality(windows: &[LabelledWindow], labels: &[WindowLabel]) -> Quality {
    let scored: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter(|(_, l)| l.pure)
        .map(|(i, _)| i)
        .collect();
    let n = scored.len();
    let empty = f64::NAN;
    if n == 0 {
        return Quality {
            scored: 0,
            purity: empty,
            recall: empty,
            complete_accuracy: empty,
            complete_link_max: empty,
            complete_link_min: empty,
            complete_link_p10: empty,
            complete_link_mean: empty,
            within_speaker_p90: empty,
            geometry: measure_geometry(windows, labels),
        };
    }

    let mut speakers: Vec<i32> = scored.iter().map(|&i| labels[i].speaker).collect();
    speakers.sort_unstable();
    speakers.dedup();
    let k = speakers.len();
    let cluster_count = windows.iter().map(|w| w.cluster).max().map_or(0, |m| m + 1);
    let sidx = |s: i32| speakers.iter().position(|&x| x == s).unwrap_or(0);
    // conf[cluster][speaker]
    let mut conf = vec![vec![0usize; k]; cluster_count];
    for &i in &scored {
        conf[windows[i].cluster][sidx(labels[i].speaker)] += 1;
    }

    let purity = (0..k)
        .map(|s| {
            let total: usize = (0..cluster_count).map(|c| conf[c][s]).sum();
            let best = (0..cluster_count).map(|c| conf[c][s]).max().unwrap_or(0);
            if total == 0 {
                0.0
            } else {
                best as f64 / total as f64
            }
        })
        .sum::<f64>()
        / k as f64;

    let recall = (0..k)
        .map(|s| {
            let total: usize = (0..cluster_count).map(|c| conf[c][s]).sum();
            let present = (0..cluster_count).filter(|&c| conf[c][s] > 0).count();
            if total == 0 {
                0.0
            } else {
                present as f64 / cluster_count as f64
            }
        })
        .sum::<f64>()
        / k as f64;

    let complete_accuracy = scored
        .iter()
        .filter(|&&i| {
            let c = windows[i].cluster;
            let s = sidx(labels[i].speaker);
            let col_best = (0..k).map(|o| conf[c][o]).max().unwrap_or(0);
            let row_best = (0..cluster_count).map(|r| conf[r][s]).max().unwrap_or(0);
            conf[c][s] == col_best && conf[c][s] == row_best
        })
        .count() as f64
        / n as f64;

    let rows: Vec<Vec<f32>> = windows.iter().map(|w| w.vector.clone()).collect();
    let group = |s: i32| -> Vec<usize> {
        scored
            .iter()
            .copied()
            .filter(|&i| labels[i].speaker == s)
            .collect()
    };
    let mut within = Vec::new();
    for s in &speakers {
        let g = group(*s);
        for (x, &i) in g.iter().enumerate() {
            for &j in &g[x + 1..] {
                within.push(cosine_distance(&rows[i], &rows[j]));
            }
        }
    }
    let within_p90 = Percentiles::of(within).p90;

    let mut cross_all = Vec::new();
    let (mut cl_max, mut cl_min, mut cl_p10, mut cl_mean) = (empty, empty, empty, empty);
    if speakers.len() >= 2 {
        for x in 0..speakers.len() {
            for y in (x + 1)..speakers.len() {
                let a = group(speakers[x]);
                let b = group(speakers[y]);
                if a.is_empty() || b.is_empty() {
                    continue;
                }
                if let (Some(worst), Some(mean)) = (
                    crate::cluster::complete_link_distance(&a, &b, &rows),
                    crate::cluster::complete_link_mean(&a, &b, &rows),
                ) {
                    cl_max = cl_max.max(worst);
                    cl_mean = if cl_mean.is_nan() {
                        mean
                    } else {
                        (cl_mean + mean) / 2.0
                    };
                }
                let mut cross = Vec::new();
                for &i in &a {
                    for &j in &b {
                        let d = cosine_distance(&rows[i], &rows[j]);
                        cross.push(d);
                        cross_all.push(d);
                    }
                }
                let s = Percentiles::of(cross);
                cl_min = if cl_min.is_nan() {
                    s.min
                } else {
                    cl_min.min(s.min)
                };
                cl_p10 = if cl_p10.is_nan() {
                    s.p10
                } else {
                    (cl_p10 + s.p10) / 2.0
                };
            }
        }
    }
    let _ = cross_all;

    Quality {
        scored: n,
        purity,
        recall,
        complete_accuracy,
        complete_link_max: cl_max,
        complete_link_min: cl_min,
        complete_link_p10: cl_p10,
        complete_link_mean: cl_mean,
        within_speaker_p90: within_p90,
        geometry: measure_geometry(windows, labels),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::LabelledWindow;
    use vaani_core::Time;

    fn window(lo: i64, hi: i64, v: f32) -> LabelledWindow {
        LabelledWindow {
            start: Time(lo),
            end: Time(hi),
            speech_ms: hi - lo,
            vector: vec![v, 1.0],
            raw: vec![v, 1.0],
            cluster: 0,
        }
    }

    #[test]
    fn a_window_within_one_speaker_is_pure() {
        // 1.5 s window over frames 0..=149 at 10 ms each, all speaker 0.
        let truth = vec![0i32; 300];
        let w = [window(0, 150, 1.0)];
        let l = label_windows(&w, &truth, 10, 0);
        assert!(l[0].pure);
        assert_eq!(l[0].speaker, 0);
        assert_eq!(l[0].labelled_frames, 15);
    }

    #[test]
    fn a_window_straddling_a_turn_is_not_pure() {
        // Speaker 0 until frame 100, speaker 1 after. A window covering frames
        // 95..=110 touches both, so it has no single correct label.
        let mut truth = vec![0i32; 300];
        truth[100..].fill(1);
        let w = [window(950, 1100, 1.0)];
        let l = label_windows(&w, &truth, 10, 0);
        assert!(!l[0].pure, "{l:?}");
        assert!(l[0].agreement() < 1.0);
    }

    #[test]
    fn silence_inside_a_window_does_not_exclude_it() {
        // Speech from frame 50, silence before and after. This is an ordinary
        // window -- the engine sees it -- so it must be kept.
        let mut truth = vec![-1i32; 300];
        truth[50..100].fill(0);
        let w = [window(400, 1000, 1.0)];
        let l = label_windows(&w, &truth, 10, 0);
        assert!(l[0].pure, "{l:?}");
        // Frames 40..=99 are inside the window, of which 50..=99 are speech.
        assert_eq!(l[0].labelled_frames, 50, "{l:?}");
    }

    #[test]
    fn a_window_with_no_speech_at_all_is_not_pure() {
        let truth = vec![-1i32; 300];
        let w = [window(0, 150, 1.0)];
        let l = label_windows(&w, &truth, 10, 0);
        assert!(!l[0].pure);
        assert_eq!(l[0].speaker, NO_SPEAKER);
    }

    #[test]
    fn an_empty_population_is_nan_not_zero() {
        let s = Percentiles::of(Vec::new());
        assert!(s.p90.is_nan());
        // The point of NAN: it must not satisfy a threshold by accident.
        assert!(
            s.p90
                .partial_cmp(&0.70)
                .is_none_or(std::cmp::Ordering::is_gt),
            "NAN must not compare as less than a threshold"
        );
    }

    #[test]
    fn percentiles_index_the_sorted_population() {
        let s = Percentiles::of((0..=100).map(|i| i as f64).collect());
        assert_eq!(s.n, 101);
        assert_eq!(s.min, 0.0);
        assert_eq!(s.max, 100.0);
        assert_eq!(s.p50, 50.0);
        assert_eq!(s.p10, 10.0);
        assert_eq!(s.p90, 90.0);
    }

    #[test]
    fn a_two_speaker_geometry_reports_a_positive_margin_when_separated() {
        // 6 s of reference: speaker 0 for the first 3 s, speaker 1 after. Five
        // 600 ms windows on each side, so every window sits wholly inside one
        // speaker's territory and all ten are pure.
        let mut truth = vec![0i32; 600];
        truth[300..].fill(1);
        let w: Vec<LabelledWindow> = (0..10)
            .map(|i| window(i * 600, i * 600 + 600, if i < 5 { 0.1 } else { 0.9 }))
            .collect();
        let l = label_windows(&w, &truth, 10, 0);
        let g = measure_geometry(&w, &l);
        assert_eq!(g.pure_windows, 10, "{g:?}");
        assert_eq!(g.mixed_windows, 0, "{g:?}");
        assert!(g.margin > 0.0, "{g:?}");
    }

    #[test]
    fn a_one_speaker_recording_reports_no_different_population() {
        // The degenerate case has to be visible, not silently zero. An empty
        // cross-speaker population is NAN, so a flattering reading can never
        // come from there being nothing to separate.
        let truth = vec![0i32; 600];
        let w: Vec<LabelledWindow> = (0..5)
            .map(|i| window(i * 600, i * 600 + 600, 0.1))
            .collect();
        let l = label_windows(&w, &truth, 10, 0);
        let g = measure_geometry(&w, &l);
        assert_eq!(g.different.n, 0);
        assert!(g.different.p10.is_nan());
        assert!(g.margin.is_nan());
    }
}

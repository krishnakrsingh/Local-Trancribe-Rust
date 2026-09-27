//! Voice activity detection, owned by this crate.
//!
//! # Why this exists and why it is not `vaani-vad`
//!
//! The product has a VAD stage (agent A3, `vaani-vad`) and the pipeline runs
//! `Vad` before `Diarization` in `JobPhase` order. This module is a
//! **fallback**, not a competitor, and it exists for three concrete reasons:
//!
//! 1. `vaani-diarization` cannot depend on `vaani-vad` without a cycle: A3's
//!    crate is specified to take the same `AudioBuffer` and produce the same
//!    `SpeechTimeline`, and two crates both owning "find the speech" is worse
//!    than one of them having a documented internal fallback. Editing A3's
//!    crate to break the cycle is not mine to do.
//! 2. Diarization has a *different* need from ASR chunking. The VAD stage
//!    optimises for chunk boundaries good for the decoder; this one optimises
//!    for frames whose features are not noise, and it benefits from seeing
//!    the whole recording's statistics at once.
//! 3. The engine must be testable in isolation. A diarization unit test
//!    should not require a VAD model to be present.
//!
//! [`crate::engine::HeuristicDiarizer::diarize_with_timeline`] takes an
//! externally supplied timeline and uses that instead, so in the real pipeline
//! A3's output wins. This code runs only when nothing else supplied one.
//!
//! # The detector
//!
//! Three per-frame quantities, all from one 256-point transform per frame:
//!
//! * **Energy**, in dB. The floor and the ceiling are *percentiles of this
//!   recording's own frame energies*, not absolute constants, because a
//!   -30 dBFS recording and a -6 dBFS recording of the same meeting need
//!   different thresholds and no absolute number serves both.
//! * **Spectral flux**: the sum of positive frame-to-frame changes in the log
//!   magnitude spectrum. Speech is not stationary; a steady hiss is.
//! * **Spectral flatness**: the geometric mean over the arithmetic mean of the
//!   power spectrum, in `[0, 1]`. White noise is near 1.0, speech is below
//!   about 0.3. **This is the dimension that makes a noise-only recording
//!   produce zero speakers rather than eight**, and it is the reason the
//!   engine has an adversarial white-noise test that passes.
//!
//! A frame is active when it is clearly above the floor
//! (`energy_hi_ratio`), or when it is marginally above it *and* is changing
//! (`energy_lo_ratio` and flux). It is never active when its flatness says it
//! is not speech, whatever its energy — a loud air conditioner is loud and
//! flat, and that is the case a pure energy gate gets wrong.
//!
//! Runs are then given hangover, interior gaps are filled, and short runs are
//! discarded. Those three operations are what turn a frame mask into a
//! `SpeechTimeline` with usable boundaries.

use vaani_core::Time;
use vaani_core::pipeline::{SpeechSegment, SpeechTimeline};

use crate::config::ActivityParams;
use crate::dsp::{FrontEnd, percentile};

/// Per-frame activity quantities, from pass 1.
#[derive(Debug, Clone, PartialEq)]
pub struct ActivityFrames {
    /// Frame index of each entry. Always `0..activity.frames`.
    pub frames: usize,
    /// Broadband energy in dB, one per frame.
    pub energy_db: Vec<f32>,
    /// Sum of positive log-magnitude changes, one per frame.
    pub flux: Vec<f32>,
    /// Geometric-over-arithmetic power ratio, one per frame.
    pub flatness: Vec<f32>,
}

impl ActivityFrames {
    /// Whether any frame was recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames == 0
    }
}

/// The thresholds derived from a recording's own statistics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActivityThresholds {
    /// Above this, a frame is speech outright.
    pub energy_hi_db: f32,
    /// Above this, a frame needs flux to agree.
    pub energy_lo_db: f32,
    /// The flux a marginal frame must exceed.
    pub flux: f32,
    /// Spectral flatness above which a frame is not speech at all.
    pub max_flatness: f32,
    /// Whether the recording had enough dynamic range to threshold at all.
    pub usable: bool,
}

/// The percentile used as the *loud* reference when deriving thresholds.
///
/// **This number is the single most consequential one in the module, and an
/// earlier version of this code got it wrong.** The natural choice is the 90th
/// percentile of frame energy, and it works on a recording that is mostly
/// speech. It fails completely on a recording that is mostly *silence*, which
/// is most real recordings: a 10-minute buffer carrying 10 seconds of speech is
/// 98% silence, so the 90th percentile of frame energy is a silence frame, the
/// derived dynamic range is zero, the recording is declared uniform, and the
/// engine reports zero speakers on a file that plainly has some. A 3-hour
/// meeting with 20 minutes of speech is 89% silence and fails the same way.
///
/// p99.9 is the fix. It is above the silence population for any recording
/// where speech is at least 0.1% of the duration, which every recording that
/// contains any speech at all satisfies.
const LOUD_PERCENTILE: f32 = 99.9;

/// Derive thresholds from a recording's own frame statistics.
///
/// `usable == false` means the recording is uniform: either digital silence
/// or a single constant level with no speech structure. In that case the
/// honest answer is zero speech, and thresholding anyway would invent it.
#[must_use]
pub fn derive_thresholds(a: &ActivityFrames, p: &ActivityParams) -> ActivityThresholds {
    let e_lo = percentile(&a.energy_db, 10.0);
    let e_hi = percentile(&a.energy_db, LOUD_PERCENTILE);
    let range = e_hi - e_lo;
    let f_med = percentile(&a.flux, 50.0);
    let f_hi = percentile(&a.flux, LOUD_PERCENTILE);
    ActivityThresholds {
        energy_hi_db: e_lo + p.energy_hi_ratio * range,
        energy_lo_db: e_lo + p.energy_lo_ratio * range,
        flux: f_med + p.flux_ratio * (f_hi - f_med).max(0.0),
        max_flatness: p.max_flatness,
        usable: range >= p.min_dynamic_range_db,
    }
}

/// Turn per-frame quantities into a boolean activity mask.
#[must_use]
pub fn activity_mask(a: &ActivityFrames, t: &ActivityThresholds) -> Vec<bool> {
    if !t.usable {
        return vec![false; a.frames];
    }
    (0..a.frames)
        .map(|i| {
            if a.flatness[i] > t.max_flatness {
                return false;
            }
            let e = a.energy_db[i];
            e > t.energy_hi_db || (e > t.energy_lo_db && a.flux[i] > t.flux)
        })
        .collect()
}

/// Run pass 1 over the whole buffer.
///
/// Returns the frame count, the three per-frame quantities, and (when
/// `progress` is supplied) a progress fraction in `[0, 1]` at most every
/// `poll` frames.
pub fn scan(
    fe: &mut FrontEnd,
    buf: &[f32],
    mut cancelled: impl FnMut() -> bool,
    mut progress: impl FnMut(f64),
    poll: usize,
) -> ActivityFrames {
    let frames = fe.usable_frames(buf.len());
    let count = frames.len();
    let mut out = ActivityFrames {
        frames: 0,
        energy_db: Vec::with_capacity(count),
        flux: Vec::with_capacity(count),
        flatness: Vec::with_capacity(count),
    };
    let mut scratch = [0f32; 3];
    let since = poll.max(1);
    for (i, f) in frames.enumerate() {
        if i % since == 0 {
            if cancelled() {
                break;
            }
            progress(if count == 0 {
                1.0
            } else {
                i as f64 / count as f64
            });
        }
        fe.frame_activity(buf, f, &mut scratch);
        out.energy_db.push(scratch[0]);
        out.flux.push(scratch[1]);
        out.flatness.push(scratch[2]);
    }
    // The count is what was actually visited, not what the buffer could have
    // offered. A cancelled scan that returns a full-length array of zeros
    // would let the caller threshold silence into a confident answer.
    out.frames = out.energy_db.len();
    out
}

/// Turn a frame mask into a `SpeechTimeline` in source-media time.
///
/// Applies, in order: hangover (pad each run by `hangover_frames` on both
/// sides), gap filling (bridge interior gaps shorter than `max_gap_frames`),
/// and minimum length (drop runs shorter than `min_speech_frames`).
///
/// The output is sorted, non-overlapping, and merged to the detector's minimum
/// gap — the contract `SpeechTimeline` documents. Times are absolute, i.e.
/// `buf_start_ms + frame_time_ms`, because a `SpeechTimeline` that came out of
/// a chunk must not need the caller to know the chunk's offset.
#[must_use]
pub fn mask_to_timeline(
    mask: &[bool],
    frame_ms: i64,
    buf_start_ms: i64,
    p: &ActivityParams,
) -> SpeechTimeline {
    let mut segments: Vec<SpeechSegment> = Vec::new();
    let mut run: Option<(usize, usize)> = None;
    let close = |run: (usize, usize), segments: &mut Vec<SpeechSegment>| {
        // `run` is `[lo, hi)`, so `hi` is already the exclusive end and only
        // needs the hangover added.
        let lo = run.0.saturating_sub(p.hangover_frames);
        let hi = run.1 + p.hangover_frames;
        if hi <= lo {
            return;
        }
        let start = buf_start_ms + lo as i64 * frame_ms;
        let end = buf_start_ms + hi as i64 * frame_ms;
        if end > start {
            segments.push(SpeechSegment {
                start: Time::from_millis(start.max(0)),
                end: Time::from_millis(end.max(0)),
                confidence: None,
            });
        }
    };

    for (i, on) in mask.iter().copied().enumerate() {
        match (on, run) {
            (true, None) => run = Some((i, i + 1)),
            (true, Some((lo, _))) => run = Some((lo, i + 1)),
            (false, Some(r)) => {
                close(r, &mut segments);
                run = None;
            }
            (false, None) => {}
        }
    }
    if let Some(r) = run {
        close(r, &mut segments);
    }

    // Bridge short interior gaps, then drop short runs. Doing it in this order
    // matters: dropping first would leave a gap that bridging then reopens.
    // `max_gap_frames` is converted to milliseconds once, here. Comparing a
    // millisecond gap against a frame count is the kind of unit slip that
    // silently disables a feature -- the comparison is always false, the
    // bridging never happens, and nothing warns.
    let max_gap_ms = p.max_gap_frames as i64 * frame_ms;
    let mut merged: Vec<SpeechSegment> = Vec::with_capacity(segments.len());
    for s in segments {
        match merged.last_mut() {
            Some(prev) if s.start.as_millis() - prev.end.as_millis() <= max_gap_ms => {
                prev.end = s.end;
            }
            _ => merged.push(s),
        }
    }
    let min_ms = p.min_speech_frames as i64 * frame_ms;
    merged.retain(|s| s.end.as_millis() - s.start.as_millis() >= min_ms);

    let speech_duration_ms = merged
        .iter()
        .map(|s| s.end.as_millis() - s.start.as_millis())
        .sum();
    SpeechTimeline {
        segments: merged,
        speech_duration_ms,
    }
}

/// Whether a frame index falls inside any segment of `tl`, given that segment
/// times are absolute and the frame's time is `frame_start + i*frame_ms`.
#[must_use]
pub fn frame_is_active(tl: &SpeechTimeline, at_ms: i64, frame_ms: i64, i: usize) -> bool {
    let t = at_ms + i as i64 * frame_ms;
    tl.segments
        .iter()
        .any(|s| s.start.as_millis() <= t && t < s.end.as_millis())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p() -> ActivityParams {
        ActivityParams::default()
    }

    /// A mask with the given runs, e.g. `[(2, 5)]` is frames 2..5 active.
    fn mask(runs: &[(usize, usize)], n: usize) -> Vec<bool> {
        let mut m = vec![false; n];
        for (lo, hi) in runs {
            for (i, slot) in m.iter_mut().enumerate() {
                if i >= *lo && i < *hi {
                    *slot = true;
                }
            }
        }
        m
    }

    #[test]
    fn silence_yields_no_segments() {
        let _fe = FrontEnd::new(
            crate::config::FeatureParams::default(),
            crate::config::ActivityParams { n_fft: 64, ..p() },
        );
        let tl = mask_to_timeline(&[false; 100], 10, 0, &p());
        assert!(tl.segments.is_empty());
        assert_eq!(tl.speech_duration_ms, 0);
    }

    #[test]
    fn a_uniform_recording_is_reported_as_unusable_rather_than_thresholded() {
        // The "all frames at the same energy" case: a constant tone or a
        // perfectly level file. Thresholding it would produce a mask at the
        // mercy of rounding.
        let a = ActivityFrames {
            frames: 100,
            energy_db: vec![-40.0; 100],
            flux: vec![0.0; 100],
            flatness: vec![0.1; 100],
        };
        let t = derive_thresholds(&a, &p());
        assert!(!t.usable);
        assert!(activity_mask(&a, &t).iter().all(|m| !*m));
    }

    #[test]
    fn a_typical_two_level_recording_is_usable() {
        let a = ActivityFrames {
            frames: 200,
            energy_db: (0..200)
                .map(|i| if i < 100 { -60.0 } else { -20.0 })
                .collect(),
            flux: vec![0.1; 200],
            flatness: vec![0.1; 200],
        };
        let t = derive_thresholds(&a, &p());
        assert!(t.usable);
        let m = activity_mask(&a, &t);
        assert!(!m[..100].iter().any(|x| *x), "the quiet half is silence");
        assert!(m[100..].iter().all(|x| *x), "the loud half is speech");
    }

    #[test]
    fn a_recording_that_is_mostly_silence_still_finds_its_speech() {
        // THE regression test for the loud-percentile choice. 10 minutes of
        // audio carrying 10 seconds of speech is 98.3% silence, so every
        // ordinary percentile -- including p90 -- lands in the silence and the
        // derived dynamic range is zero.
        let n = 60_000usize; // 10 minutes of 10 ms frames
        let speech_frames = 1_000; // 10 seconds
        let mut energy = vec![-120.0f32; n];
        let mut flat = vec![0.9f32; n];
        for i in 0..speech_frames {
            energy[n - speech_frames + i] = -20.0;
            flat[n - speech_frames + i] = 0.002;
        }
        let a = ActivityFrames {
            frames: n,
            energy_db: energy,
            flux: vec![0.0; n],
            flatness: flat,
        };
        let t = derive_thresholds(&a, &p());
        assert!(
            t.usable,
            "a 98%-silent recording was declared uniform: range must be measured against p99.9"
        );
        let m = activity_mask(&a, &t);
        let active = m.iter().filter(|x| **x).count();
        assert_eq!(active, speech_frames, "found {active} of {speech_frames}");
    }

    #[test]
    fn two_percent_of_speech_is_still_found() {
        let n = 60_000usize;
        let speech_frames = n / 50; // 2%
        let mut energy = vec![-120.0f32; n];
        let mut flat = vec![0.9f32; n];
        for i in 0..speech_frames {
            energy[i] = -25.0;
            flat[i] = 0.002;
        }
        let a = ActivityFrames {
            frames: n,
            energy_db: energy,
            flux: vec![0.0; n],
            flatness: flat,
        };
        let t = derive_thresholds(&a, &p());
        assert!(t.usable);
        let m = activity_mask(&a, &t);
        let active = m.iter().filter(|x| **x).count();
        assert!(
            active > speech_frames * 9 / 10,
            "found {active} of {speech_frames}"
        );
    }

    #[test]
    fn flat_noise_is_never_active_however_loud_it_is() {
        // The white-noise adversarial case, at the unit level: maximum energy
        // dynamic range, so the energy gate is wide open, and still nothing
        // passes because the flatness gate refuses it.
        let a = ActivityFrames {
            frames: 100,
            energy_db: (0..100).map(|i| -60.0 + i as f32).collect(),
            flux: (0..100).map(|i| i as f32 * 3.0).collect(),
            flatness: vec![0.95; 100],
        };
        let t = derive_thresholds(&a, &p());
        assert!(t.usable);
        assert!(
            activity_mask(&a, &t).iter().all(|m| !*m),
            "flat noise was reported as speech"
        );
    }

    #[test]
    fn flatness_beats_energy_when_they_disagree() {
        // A loud, high-flatness frame next to a quiet, speech-shaped one.
        let a = ActivityFrames {
            frames: 2,
            energy_db: vec![-5.0, -30.0],
            flux: vec![0.0, 5.0],
            flatness: vec![0.9, 0.1],
        };
        let mut t = derive_thresholds(&a, &p());
        t.usable = true;
        t.energy_hi_db = -20.0;
        t.energy_lo_db = -40.0;
        t.flux = 1.0;
        let m = activity_mask(&a, &t);
        assert!(!m[0], "the hiss must be rejected");
        assert!(m[1], "the speech-shaped frame must be accepted");
    }

    #[test]
    fn hangover_extends_a_run_on_both_sides() {
        let _fe = FrontEnd::new(crate::config::FeatureParams::default(), p());
        let a = ActivityParams {
            hangover_frames: 3,
            max_gap_frames: 0,
            min_speech_frames: 0,
            ..p()
        };
        let tl = mask_to_timeline(&mask(&[(10, 12)], 40), 10, 0, &a);
        assert_eq!(tl.segments.len(), 1);
        // 7..15, i.e. 70 ms to 150 ms.
        assert_eq!(tl.segments[0].start, Time(70));
        assert_eq!(tl.segments[0].end, Time(150));
    }

    #[test]
    fn short_interior_gaps_are_bridged_and_long_ones_are_not() {
        let _fe = FrontEnd::new(crate::config::FeatureParams::default(), p());
        let a = ActivityParams {
            hangover_frames: 0,
            max_gap_frames: 5,
            min_speech_frames: 0,
            ..p()
        };
        // A 3-frame gap is bridged.
        let tl = mask_to_timeline(&mask(&[(5, 8), (11, 14)], 30), 10, 0, &a);
        assert_eq!(tl.segments.len(), 1, "{:?}", tl.segments);
        assert_eq!(tl.segments[0].start, Time(50));
        assert_eq!(tl.segments[0].end, Time(140));
        // A 10-frame gap is not.
        let tl = mask_to_timeline(&mask(&[(5, 8), (18, 21)], 40), 10, 0, &a);
        assert_eq!(tl.segments.len(), 2);
    }

    #[test]
    fn runs_shorter_than_the_minimum_are_discarded() {
        let _fe = FrontEnd::new(crate::config::FeatureParams::default(), p());
        let a = ActivityParams {
            hangover_frames: 0,
            max_gap_frames: 0,
            min_speech_frames: 20,
            ..p()
        };
        // A 5-frame blip is dropped; a 30-frame run survives.
        let tl = mask_to_timeline(&mask(&[(5, 10), (20, 50)], 80), 10, 0, &a);
        assert_eq!(tl.segments.len(), 1, "{:?}", tl.segments);
        assert_eq!(tl.segments[0].start, Time(200));
    }

    #[test]
    fn segments_are_sorted_and_non_overlapping() {
        let _fe = FrontEnd::new(crate::config::FeatureParams::default(), p());
        let a = ActivityParams {
            hangover_frames: 2,
            max_gap_frames: 4,
            min_speech_frames: 0,
            ..p()
        };
        let tl = mask_to_timeline(&mask(&[(5, 8), (14, 17), (40, 44)], 60), 10, 0, &a);
        for w in tl.segments.windows(2) {
            assert!(w[0].end <= w[1].start, "{:?}", tl.segments);
        }
        for s in &tl.segments {
            assert!(s.start < s.end);
        }
    }

    #[test]
    fn timeline_times_are_absolute_not_chunk_local() {
        // A buffer that starts 5 minutes into the media must report absolute
        // times, or the transcript drifts by five minutes.
        let _fe = FrontEnd::new(crate::config::FeatureParams::default(), p());
        let a = ActivityParams {
            hangover_frames: 0,
            max_gap_frames: 0,
            min_speech_frames: 0,
            ..p()
        };
        let tl = mask_to_timeline(&mask(&[(10, 20)], 40), 10, 300_000, &a);
        assert_eq!(tl.segments[0].start, Time(300_100));
    }

    #[test]
    fn speech_duration_is_the_sum_of_the_segments() {
        let _fe = FrontEnd::new(crate::config::FeatureParams::default(), p());
        let a = ActivityParams {
            hangover_frames: 0,
            max_gap_frames: 0,
            min_speech_frames: 0,
            ..p()
        };
        let tl = mask_to_timeline(&mask(&[(0, 10), (20, 30)], 40), 10, 0, &a);
        assert_eq!(tl.speech_duration_ms, 200);
        assert!((tl.speech_ratio(Time(400)) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn a_run_touching_the_end_of_the_buffer_does_not_overrun_it() {
        let _fe = FrontEnd::new(crate::config::FeatureParams::default(), p());
        let tl = mask_to_timeline(&mask(&[(38, 40)], 40), 10, 0, &p());
        // Hangover extends past the last frame; the engine clamps against the
        // buffer duration later, so here we only require a valid segment.
        for s in &tl.segments {
            assert!(s.start < s.end);
        }
    }

    #[test]
    fn frame_is_active_respects_absolute_times() {
        let tl = SpeechTimeline {
            segments: vec![SpeechSegment {
                start: Time(1_000),
                end: Time(2_000),
                confidence: None,
            }],
            speech_duration_ms: 1_000,
        };
        assert!(frame_is_active(&tl, 0, 10, 100)); // t = 1000
        assert!(!frame_is_active(&tl, 0, 10, 99)); // t = 990
        assert!(!frame_is_active(&tl, 0, 10, 200)); // t = 2000, half-open
        // A chunk that starts five minutes into the media must still agree with
        // the absolute segment times.
        let late = SpeechTimeline {
            segments: vec![SpeechSegment {
                start: Time(300_000),
                end: Time(301_000),
                confidence: None,
            }],
            speech_duration_ms: 1_000,
        };
        assert!(frame_is_active(&late, 300_000, 10, 50));
        assert!(!frame_is_active(&late, 299_000, 10, 50));
    }

    #[test]
    fn scan_stops_when_cancelled() {
        let mut fe = FrontEnd::new(crate::config::FeatureParams::default(), p());
        let buf: Vec<f32> = (0..16_000).map(|i| ((i as f32) * 0.01).sin()).collect();
        let a = scan(&mut fe, &buf, || true, |_| {}, 10);
        assert!(a.is_empty(), "cancellation did not stop the scan");
    }

    #[test]
    fn scan_reports_monotone_progress() {
        let mut fe = FrontEnd::new(crate::config::FeatureParams::default(), p());
        let buf: Vec<f32> = (0..32_000).map(|i| ((i as f32) * 0.01).sin()).collect();
        let mut seen: Vec<f64> = Vec::new();
        let a = scan(&mut fe, &buf, || false, |f| seen.push(f), 20);
        assert!(!a.is_empty());
        assert!(seen.windows(2).all(|w| w[1] >= w[0]), "{seen:?}");
        assert!(seen.iter().all(|f| (0.0..=1.0).contains(f)));
    }

    #[test]
    fn an_empty_buffer_scans_to_nothing() {
        let mut fe = FrontEnd::new(crate::config::FeatureParams::default(), p());
        let a = scan(&mut fe, &[], || false, |_| {}, 10);
        assert!(a.is_empty());
        assert!(activity_mask(&a, &derive_thresholds(&a, &p())).is_empty());
    }
}

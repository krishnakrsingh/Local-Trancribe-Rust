//! Acoustic structure probing, for choosing evaluation clips without
//! listening and without asking the system what was said.
//!
//! # Why this exists
//!
//! The evaluation set has to cover speech characteristics — pauses, overlap,
//! speaker changes, fast speech — and the only tool available for choosing
//! clip boundaries is a measurement. But the obvious measurement is the wrong
//! one: running the recogniser over the whole recording and picking windows
//! where it looks interesting selects the set using the thing under test, and
//! every number computed on that set is then optimistic by construction. So
//! clip selection in this harness uses **only** these three signals, all
//! computed from the raw samples:
//!
//! * frame RMS energy in dBFS;
//! * a zero-crossing rate, which separates voiced speech from clicks and
//!   fan noise;
//! * an adaptive noise-floor estimate, because a fixed threshold on a
//!   recording with a moving floor calls half a meeting silent.
//!
//! None of it knows a word. The output is a structural summary — speech
//! fraction, longest silence, silence count, loudness percentiles — and that
//! summary contains no transcript content and is safe to commit. The
//! frame-level detail is not committed.
//!
//! # What it cannot do, stated plainly
//!
//! None of this can tell you whether a stretch is Hindi, Hinglish or English,
//! whether it contains a product name, or whether two people are talking over
//! each other. Those are *perceptual* facts, and this module produces no claim
//! about them. Every such field in `metadata/<clip>.meta.json` is therefore
//! marked `unverified` unless a human annotator has confirmed it. A taxonomy
//! that cannot detect its own categories is a wish list.

use serde::{Deserialize, Serialize};

use crate::wav::Audio;

/// Analysis frame length, milliseconds.
pub const FRAME_MS: i64 = 20;
/// Analysis hop, milliseconds.
pub const HOP_MS: i64 = 10;

/// One analysis frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Frame {
    /// Frame centre, milliseconds.
    pub t_ms: i64,
    /// Root-mean-square level, dBFS. Silence floors at about -120.
    pub rms_dbfs: f32,
    /// Zero crossings per second. Voiced speech sits well above 0 and well
    /// below ~6000; a click is a single spike.
    pub zcr: f32,
    /// `true` when the frame is above the adaptive floor.
    pub speech: bool,
}

/// A contiguous non-speech span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Silence {
    /// Start, milliseconds.
    pub start_ms: i64,
    /// End, milliseconds.
    pub end_ms: i64,
}

impl Silence {
    /// Length, milliseconds.
    pub fn len_ms(&self) -> i64 {
        self.end_ms - self.start_ms
    }
}

/// A structural summary of a piece of audio. Contains no content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Structure {
    /// Sample rate, hertz.
    pub sample_rate: u32,
    /// Duration, milliseconds.
    pub duration_ms: i64,
    /// Fraction of frames classified as speech, in [0, 1].
    pub speech_fraction: f64,
    /// Number of speech runs longer than 200 ms.
    pub speech_runs: usize,
    /// Number of silence spans longer than 300 ms.
    pub silence_spans: usize,
    /// Longest silence, milliseconds.
    pub longest_silence_ms: i64,
    /// Longest contiguous speech run, milliseconds.
    pub longest_speech_ms: i64,
    /// Silences of at least 700 ms, i.e. pauses a listener would notice.
    pub notable_pauses: Vec<Silence>,
    /// 5th percentile frame level, dBFS.
    pub rms_p05_dbfs: f32,
    /// Median frame level, dBFS.
    pub rms_p50_dbfs: f32,
    /// 95th percentile frame level, dBFS.
    pub rms_p95_dbfs: f32,
    /// Median zero-crossing rate over speech frames, crossings per second.
    /// A crude speaking-rate proxy: a rising median means brighter, faster
    /// articulation. It is a proxy and is labelled as one everywhere.
    pub speech_zcr_median: f32,
    /// The adaptive noise floor the classifier settled on, dBFS.
    pub noise_floor_dbfs: f32,
    /// Frames analysed.
    pub frame_count: usize,
}

/// Frames plus the structural summary, so clip selection can look at both.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Analysis {
    /// Structural summary.
    pub structure: Structure,
    /// Per-frame detail. Never committed: it is a dense map of the recording.
    #[serde(default)]
    pub frames: Vec<Frame>,
}

/// Compute the frame sequence and the structural summary.
pub fn analyse(audio: &Audio) -> Analysis {
    let rate = audio.sample_rate.max(1) as i64;
    let frame_len = (FRAME_MS * rate / 1000).max(1) as usize;
    let hop = (HOP_MS * rate / 1000).max(1) as usize;

    let mut frames: Vec<Frame> = Vec::new();
    let mut at = 0usize;
    while at + frame_len <= audio.samples.len() {
        let w = &audio.samples[at..at + frame_len];
        let sum_sq: f64 = w.iter().map(|s| (*s as f64) * (*s as f64)).sum();
        let rms = (sum_sq / w.len() as f64).sqrt() as f32;
        let mut crossings = 0usize;
        for pair in w.windows(2) {
            if (pair[0] >= 0.0) != (pair[1] >= 0.0) {
                crossings += 1;
            }
        }
        let secs = frame_len as f32 / rate as f32;
        frames.push(Frame {
            t_ms: (at as i64 * 1000) / rate,
            rms_dbfs: dbfs(rms),
            zcr: crossings as f32 / secs.max(1e-6),
            speech: false,
        });
        at += hop;
    }
    if frames.is_empty() {
        return Analysis {
            structure: empty_structure(rate),
            frames,
        };
    }

    // Adaptive floor: the 20th percentile of frame level, which is robust to
    // both a loud meeting and a room with a hum in it. A fixed threshold is
    // the classic way to call half a recording silent.
    let mut levels: Vec<f32> = frames.iter().map(|f| f.rms_dbfs).collect();
    levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let floor = percentile(&levels, 0.20);
    // A speech frame must clear the floor by 8 dB. A fixed margin is used
    // rather than a scaled one because the meeting was recorded at one gain.
    let threshold = floor + 8.0;
    for f in frames.iter_mut() {
        f.speech = f.rms_dbfs > threshold;
    }

    let structure = Structure {
        sample_rate: audio.sample_rate,
        duration_ms: audio.duration_ms(),
        speech_fraction: frames.iter().filter(|f| f.speech).count() as f64 / frames.len() as f64,
        speech_runs: runs(&frames, true, 200),
        silence_spans: runs(&frames, false, 300),
        longest_silence_ms: longest(&frames, false),
        longest_speech_ms: longest(&frames, true),
        notable_pauses: pauses(&frames, 700),
        rms_p05_dbfs: round1(percentile(&levels, 0.05)),
        rms_p50_dbfs: round1(percentile(&levels, 0.50)),
        rms_p95_dbfs: round1(percentile(&levels, 0.95)),
        speech_zcr_median: round1(median_of_speech(&frames)),
        noise_floor_dbfs: round1(floor),
        frame_count: frames.len(),
    };
    Analysis { structure, frames }
}

fn empty_structure(rate: i64) -> Structure {
    Structure {
        sample_rate: rate as u32,
        duration_ms: 0,
        speech_fraction: 0.0,
        speech_runs: 0,
        silence_spans: 0,
        longest_silence_ms: 0,
        longest_speech_ms: 0,
        notable_pauses: Vec::new(),
        rms_p05_dbfs: -120.0,
        rms_p50_dbfs: -120.0,
        rms_p95_dbfs: -120.0,
        speech_zcr_median: 0.0,
        noise_floor_dbfs: -120.0,
        frame_count: 0,
    }
}

fn dbfs(rms: f32) -> f32 {
    if rms <= 1e-7 {
        -120.0
    } else {
        20.0 * rms.log10()
    }
}

fn percentile(sorted: &[f32], q: f64) -> f32 {
    if sorted.is_empty() {
        return -120.0;
    }
    let r = ((q * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    sorted[r - 1]
}

fn median_of_speech(frames: &[Frame]) -> f32 {
    let mut v: Vec<f32> = frames.iter().filter(|f| f.speech).map(|f| f.zcr).collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

fn runs(frames: &[Frame], want_speech: bool, min_ms: i64) -> usize {
    let mut n = 0usize;
    let mut start: Option<i64> = None;
    let mut last = 0i64;
    for f in frames {
        if f.speech == want_speech {
            if start.is_none() {
                start = Some(f.t_ms);
            }
            last = f.t_ms;
        } else if let Some(s) = start.take()
            && last + crate::probe::HOP_MS - s >= min_ms
        {
            n += 1;
        }
    }
    if let Some(s) = start
        && last + HOP_MS - s >= min_ms
    {
        n += 1;
    }
    n
}

fn longest(frames: &[Frame], want_speech: bool) -> i64 {
    let mut best = 0i64;
    let mut start: Option<i64> = None;
    let mut last = 0i64;
    for f in frames {
        if f.speech == want_speech {
            if start.is_none() {
                start = Some(f.t_ms);
            }
            last = f.t_ms;
        } else if let Some(s) = start.take() {
            best = best.max(last + HOP_MS - s);
        }
    }
    if let Some(s) = start {
        best = best.max(last + HOP_MS - s);
    }
    best
}

fn pauses(frames: &[Frame], min_ms: i64) -> Vec<Silence> {
    let mut out = Vec::new();
    let mut start: Option<i64> = None;
    let mut last = 0i64;
    for f in frames {
        if !f.speech {
            if start.is_none() {
                start = Some(f.t_ms);
            }
            last = f.t_ms;
        } else if let Some(s) = start.take() {
            let e = last + HOP_MS;
            if e - s >= min_ms {
                out.push(Silence {
                    start_ms: s,
                    end_ms: e,
                });
            }
        }
    }
    if let Some(s) = start {
        let e = last + HOP_MS;
        if e - s >= min_ms {
            out.push(Silence {
                start_ms: s,
                end_ms: e,
            });
        }
    }
    out
}

fn round1(x: f32) -> f32 {
    (x * 10.0).round() / 10.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(samples: &[f32], rate: u32) -> Audio {
        Audio {
            sample_rate: rate,
            samples: samples.to_vec(),
        }
    }

    #[test]
    fn silence_is_reported_as_silence() {
        let a = build(&vec![0.0; 16_000], 16_000); // 1 s of digital silence
        let s = analyse(&a).structure;
        assert_eq!(s.speech_fraction, 0.0);
        // 99 frames at a 10 ms hop, so the last frame centre is 980 ms and the
        // run closes one hop later.
        assert_eq!(s.frame_count, 99);
        assert_eq!(s.longest_silence_ms, 990);
        assert!(s.noise_floor_dbfs <= -119.0);
        assert_eq!(s.duration_ms, 1000);
    }

    #[test]
    fn a_tone_over_silence_finds_the_run_and_the_gap() {
        let mut v = vec![0.0f32; 8_000]; // 0.5 s silence
        v.extend((0..8_000).map(|i| ((i as f32 / 20.0).sin()) * 0.4));
        v.extend(vec![0.0f32; 8_000]); // 0.5 s silence
        let s = analyse(&build(&v, 16_000)).structure;
        assert!(s.speech_fraction > 0.3, "{s:?}");
        assert_eq!(s.speech_runs, 1, "{s:?}");
        assert_eq!(s.silence_spans, 2, "{s:?}");
        // 0.5 s of gap is a real gap but not a pause a listener would call out,
        // and the 700 ms threshold is what says so. Both facts are asserted,
        // because a threshold nobody checks is a constant, not a policy.
        assert_eq!(s.notable_pauses.len(), 0, "{s:?}");
        assert!(
            s.longest_silence_ms >= 480 && s.longest_silence_ms <= 500,
            "{s:?}"
        );
        assert!(s.longest_speech_ms >= 500, "{s:?}");
        assert!(s.noise_floor_dbfs <= -119.0, "{s:?}");
    }

    #[test]
    fn a_gap_longer_than_the_pause_threshold_is_reported_as_a_pause() {
        let mut v = vec![0.0f32; 16_000]; // 1.0 s
        v.extend((0..8_000).map(|i| ((i as f32 / 20.0).sin()) * 0.4));
        v.extend(vec![0.0f32; 8_000]); // 0.5 s
        let s = analyse(&build(&v, 16_000)).structure;
        assert_eq!(s.notable_pauses.len(), 1, "{s:?}");
        assert!(s.longest_silence_ms >= 950, "{s:?}");
    }

    #[test]
    fn an_empty_file_does_not_panic_and_reports_zero() {
        let s = analyse(&build(&[], 16_000)).structure;
        assert_eq!(s.frame_count, 0);
        assert_eq!(s.duration_ms, 0);
        assert_eq!(s.speech_fraction, 0.0);
    }

    #[test]
    fn a_very_short_file_yields_at_most_one_frame() {
        let s = analyse(&build(&[0.1, 0.2, 0.3], 16_000)).structure;
        assert!(s.frame_count <= 1);
    }

    #[test]
    fn the_summary_contains_no_sample_values() {
        // The committed artefact is `Structure`; a serialised summary must be
        // expressible without the frame list. Checked by construction below.
        let v = vec![0.0f32; 16_000];
        let a = analyse(&build(&v, 16_000));
        let json = serde_json::to_value(&a.structure).unwrap();
        assert!(json.get("frames").is_none());
    }
}

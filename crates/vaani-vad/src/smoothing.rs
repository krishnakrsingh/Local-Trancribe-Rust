//! Turning a per-frame score into a [`SpeechTimeline`].
//!
//! Everything in this module is a pure function over small vectors. That is a
//! deliberate structural choice, not a stylistic one: the hysteresis state
//! machine, the two duration rules and the padding are the part of a VAD that
//! is hardest to get right and easiest to get subtly wrong, and the only way
//! to test "a 30 ms click must not become a segment" against audio is to
//! synthesise 30 ms of audio every time. Against a `Vec<bool>` of 3 frames it
//! is a one-line test.
//!
//! # The three rules, and why each exists
//!
//! ```text
//!  score  --hysteresis-->  state  --min silence-->  state  --min speech--> segments
//! ```
//!
//! **Hysteresis** (a higher threshold to enter speech than to leave) exists
//! because a single threshold on a noisy score chatters. Without it, a frame
//! hovering at the threshold flips the state on alternate frames, and the
//! duration rules then see a 10 ms "segment" 40 times a second. The gap
//! between the two thresholds is the hysteresis band; it is 0.16 in score units
//! here, which is wider than the frame-to-frame spread of a stationary signal
//! and narrower than the spread between a vowel and the pause after it. Both
//! endpoints are pinned by measurement rather than taste: the leave threshold
//! must sit above the loudest adversary's score and the enter threshold above
//! the leave threshold, and `tests/calibration.rs` prints the numbers.
//!
//! **Minimum silence** (300 ms) exists because natural inter-word gaps are
//! 50-250 ms. Cutting at those produces a segment per word, and every cut
//! costs a decoder warm-up and a chance to mis-decode the word that straddles
//! the boundary. A 50 ms gap is a stop closure, not a segment boundary. The
//! cost is that two genuinely separate utterances 200 ms apart merge into one,
//! which costs a little context and nothing else -- the decoder sees both.
//!
//! **Minimum speech** (200 ms) exists because there is no word shorter than
//! about 150 ms and a 30 ms click is not a word. This is also the backstop
//! against keyboard noise, a cough, and paper rustle, all of which pass the
//! per-frame rule easily and all of which are 10-80 ms.
//!
//! **Padding** (60 ms each side) exists because the window is 25 ms and the
//! hop is 10 ms, so a frame is scored up to 25 ms after the energy arrived and
//! the *last* frame of an utterance overlaps the last 25 ms of it. Unpadded
//! boundaries are systematically late on onsets and early on offsets, i.e.
//! biased in both directions, and a systematic early offset clips the last
//! phoneme. 60 ms covers 2.5 frames, which is more than the 2.5 frames of
//! quantisation plus the 25 ms of window smear.
//!
//! # Order of operations
//!
//! The two duration rules are applied **silence first, then speech**, and then
//! iterated to a fixed point. The order matters and it is not cosmetic:
//!
//! * Absorbing a short silence *lengthens* speech runs, so it can only help a
//!   run clear the minimum-speech floor. Doing it first is therefore safe.
//! * Dropping a short speech run *widens* the silence around it. If the widened
//!   silence is still shorter than the minimum silence, the two neighbouring
//!   utterances should now merge -- which is a second absorption pass.
//!
//! One pass of each is not enough, and a fixed point of at most 8 iterations is
//! provably reached: every pass strictly reduces the number of runs, and a
//! finite sequence of runs cannot decrease forever. The bound is a backstop
//! against a bug turning that argument into a hang; the tests assert the
//! fixed point is reached in at most 3 passes on the fixtures.

use serde::{Deserialize, Serialize};
use vaani_core::Time;
use vaani_core::pipeline::{SpeechSegment, SpeechTimeline};

use crate::dsp::HOP_MS;
use crate::features::{frames_to_ms_at, ms_to_frames_at_ceil};

/// The decision rule's smoothing parameters.
///
/// Separate from [`crate::config::EnergyVadConfig`] because the neural engine
/// uses exactly these numbers on exactly the same post-processing, and a single
/// copy of them is the only way "the neural path applies the same smoothing"
/// can be a true statement rather than an aspiration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct SmoothingConfig {
    /// Score at or above which a frame may start a speech run.
    pub enter_threshold: f32,
    /// Score below which a frame ends a speech run. Must be **strictly less**
    /// than `enter_threshold`; a violation is a bug and is reported, not
    /// silently normalised, because a silent normalisation makes the
    /// hysteresis band zero and the chatter comes back.
    pub leave_threshold: f32,
    /// A speech run shorter than this is discarded. In milliseconds.
    pub min_speech_ms: i64,
    /// A silence shorter than this does not end a speech run. In milliseconds.
    pub min_silence_ms: i64,
    /// Padding added either side of every segment, in milliseconds.
    pub pad_ms: i64,
}

impl SmoothingConfig {
    /// Whether the hysteresis band is actually a band.
    ///
    /// `enter == leave` is not a degenerate configuration, it is a bug: it
    /// makes the decision a function of `score >= t` with no memory, which is
    /// precisely the chattering behaviour the hysteresis exists to prevent.
    pub fn is_valid(&self) -> bool {
        self.leave_threshold < self.enter_threshold
            && self.min_speech_ms >= 0
            && self.min_silence_ms >= 0
            && self.pad_ms >= 0
            && self.enter_threshold.is_finite()
            && self.leave_threshold.is_finite()
    }

    /// Minimum speech in whole frames on the energy detector's 10 ms grid.
    #[must_use]
    pub fn min_speech_frames(&self) -> usize {
        self.min_speech_frames_on(HOP_MS)
    }

    /// Minimum silence in whole frames on the energy detector's 10 ms grid.
    #[must_use]
    pub fn min_silence_frames(&self) -> usize {
        self.min_silence_frames_on(HOP_MS)
    }

    /// Minimum speech in whole frames on an explicit `hop_ms` grid.
    ///
    /// The two engines run on different grids (10 ms and 16 ms) and share this
    /// one policy, so the conversion has to be against the caller's grid. Doing
    /// it against a single hard-coded hop is how a "minimum speech 200 ms" rule
    /// quietly becomes 160 ms in one engine.
    #[must_use]
    pub fn min_speech_frames_on(&self, hop_ms: i64) -> usize {
        ms_to_frames_at_ceil(self.min_speech_ms, hop_ms)
    }

    /// Minimum silence in whole frames on an explicit `hop_ms` grid.
    #[must_use]
    pub fn min_silence_frames_on(&self, hop_ms: i64) -> usize {
        ms_to_frames_at_ceil(self.min_silence_ms, hop_ms)
    }
}

impl Default for SmoothingConfig {
    /// The tuned defaults.
    fn default() -> Self {
        Self {
            // Measured against the score distributions in
            // `tests/calibration.rs`: the loudest adversary (a steady tone or a
            // 50 Hz hum, both fully speech-*shaped* and only separated by the
            // steady-source and out-of-band vetoes) scores 0.25 and 0.19, and
            // broadband white noise scores 0.00. So the leave threshold has to
            // sit above 0.25 or a hum holds a segment open indefinitely, and the
            // enter threshold has to be well clear of the leave threshold for
            // the hysteresis band to do its job.
            enter_threshold: 0.50,
            leave_threshold: 0.34,
            min_speech_ms: 200,
            min_silence_ms: 300,
            pad_ms: 60,
        }
    }
}

/// One maximal run of frames sharing a state, in frame indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Run {
    /// First frame of the run, inclusive.
    pub start: usize,
    /// Number of frames in the run; always at least 1.
    pub len: usize,
    /// Whether the run is speech.
    pub speech: bool,
}

impl Run {
    /// First frame past the end of the run.
    #[must_use]
    pub fn end(&self) -> usize {
        self.start + self.len
    }
}

/// Apply the hysteresis rule to a frame score sequence.
///
/// The state machine is a two-state automaton with one latch: once in speech,
/// the state is held until a frame falls **below** the leave threshold. A frame
/// between the two thresholds therefore keeps whatever state it is in, which
/// is what makes a vowel-to-pause transition not split a segment: the pause has
/// to be genuinely quiet (below `leave`), not merely quietish.
#[must_use]
pub fn hysteresis(scores: &[f32], config: &SmoothingConfig) -> Vec<bool> {
    let mut out = Vec::with_capacity(scores.len());
    let mut in_speech = false;
    for &s in scores {
        if in_speech {
            if s < config.leave_threshold {
                in_speech = false;
            }
        } else if s >= config.enter_threshold {
            in_speech = true;
        }
        out.push(in_speech);
    }
    out
}

/// Split a state sequence into maximal runs.
///
/// A sequence of length 0 has no runs; a sequence of length 1 has exactly one.
/// Zero-length runs are never emitted, which is what lets
/// [`absorb_short_silences`] and [`drop_short_speech`] be total functions over
/// run lists.
#[must_use]
pub fn runs(states: &[bool]) -> Vec<Run> {
    let mut out: Vec<Run> = Vec::new();
    for (i, &s) in states.iter().enumerate() {
        match out.last_mut() {
            Some(last) if last.speech == s => last.len += 1,
            _ => out.push(Run {
                start: i,
                len: 1,
                speech: s,
            }),
        }
    }
    out
}

/// Merge speech runs across silences shorter than `min_silence_frames`.
///
/// A short silence with speech on **both** sides is absorbed: the two runs
/// become one. A short silence at the start or end of the buffer is left
/// alone, because there is nothing to merge it with and shortening the silence
/// would gain nothing.
#[must_use]
pub fn absorb_short_silences(mut rs: Vec<Run>, min_silence_frames: usize) -> Vec<Run> {
    if min_silence_frames <= 1 || rs.len() < 3 {
        return rs;
    }
    let mut i = 1usize;
    while i < rs.len() {
        // i >= 1, so `rs[i - 1]` is in bounds: a silence can only be an
        // *inner* silence if there is a run before it, and run 0 being a
        // silence is exactly the buffer-edge case this leaves alone.
        let absorbable = !rs[i].speech
            && rs[i - 1].speech
            && i + 1 < rs.len()
            && rs[i + 1].speech
            && rs[i].len < min_silence_frames;
        if absorbable {
            let merged = Run {
                start: rs[i - 1].start,
                len: rs[i + 1].end() - rs[i - 1].start,
                speech: true,
            };
            rs[i - 1] = merged;
            rs.drain(i..i + 2);
            // Step back one: the run now at i-1 may itself be a short silence
            // sitting next to a short silence.
            i = i.saturating_sub(1);
        } else {
            i += 1;
        }
    }
    rs
}

/// Merge adjacent runs of the same state back into maximal runs.
///
/// `drop_short_speech` removes elements from the middle of a list, which can
/// leave two adjacent silence runs. That is invisible in the final result --
/// [`smoothing`] filters to speech runs only -- but it violates the invariant
/// that a run list contains *maximal* runs, and an invariant that is violated
/// in one place is an invariant nothing can rely on.
#[must_use]
pub fn coalesce_runs(rs: Vec<Run>) -> Vec<Run> {
    let mut out: Vec<Run> = Vec::with_capacity(rs.len());
    for r in rs {
        match out.last_mut() {
            Some(last) if last.speech == r.speech => last.len += r.len,
            _ => out.push(r),
        }
    }
    out
}

/// Discard speech runs shorter than `min_speech_frames`.
///
/// A discarded run becomes **silence**, not a hole. This distinction is the
/// whole point: a run list with a gap in it makes every downstream duration and
/// frame-index calculation short by exactly the dropped frames, and it hides
/// the fact that the silence on either side of the dropped run is now
/// contiguous -- which is the run that `min_silence_frames` has to judge. (A
/// `retain`-based version of this left a 5-frame hole in a 305-frame timeline
/// and split one 105-frame silence in two.)
#[must_use]
pub fn drop_short_speech(rs: Vec<Run>, min_speech_frames: usize) -> Vec<Run> {
    if min_speech_frames == 0 {
        return rs;
    }
    let mut out: Vec<Run> = Vec::with_capacity(rs.len());
    for r in rs {
        if r.speech && r.len < min_speech_frames {
            match out.last_mut() {
                // Runs are contiguous, so the run we are inside is the one
                // whose `end` equals this run's `start`: lengthening it is
                // exactly "the silence is 5 frames longer now".
                Some(last) if !last.speech => last.len += r.len,
                _ => out.push(Run {
                    start: r.start,
                    len: r.len,
                    speech: false,
                }),
            }
        } else {
            out.push(r);
        }
    }
    coalesce_runs(out)
}

/// Absorb short silences, then drop short speech runs, to a fixed point.
///
/// Returns the speech runs only, sorted, non-overlapping, and non-adjacent
/// (consecutive runs are always separated by at least `min_silence_ms`).
#[must_use]
pub fn smoothing(rs: Vec<Run>, config: &SmoothingConfig) -> Vec<Run> {
    smoothing_passes(rs, config, HOP_MS).speech
}

/// The confidence reported for a segment.
///
/// # Why this is honest rather than a made-up probability
///
/// `SpeechSegment::confidence` is an `Option<f32>` precisely so a detector that
/// cannot produce a calibrated number can say so. This one *can* produce a
/// meaningful ranking, and this is what it is:
///
/// ```text
/// confidence = mean over the segment's frames of
///              clamp01((score - leave_threshold) / (1 - leave_threshold))
/// ```
///
/// i.e. **the average distance the segment's frames sat above the threshold at
/// which the detector would have given up on it.** A segment that only just
/// held together sits near 0; a segment that scored 0.9 throughout sits near
/// 0.75. That is a real, reproducible, per-frame-computable quantity.
///
/// What it is *not*: a posterior. It is not calibrated against a labelled
/// corpus, and `0.8` does not mean "80 % likely to be speech". It is an
/// ordering key -- "which of these two segments is more solidly speech" --
/// and the doc comment on the field says `Option::is_none` is available for a
/// detector that has no such ordering. `energy.rs` documents the same caveat at
/// the engine level and the neural path's number is on the model's own scale
/// rather than this one, which is a real inconsistency and is called out in the
/// final report.
#[must_use]
pub fn segment_confidence(scores: &[f32], run: Run, leave_threshold: f32) -> Option<f32> {
    if scores.is_empty() || run.len == 0 {
        return None;
    }
    let hi = 1.0 - leave_threshold;
    // `partial_cmp` rather than `<= 0.0` because a NaN threshold has to land
    // here too: `!(hi > 0.0)` is true for NaN, `hi <= 0.0` is not, and a NaN
    // confidence is worse than no confidence.
    if hi.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return None;
    }
    let mut acc = 0f64;
    let mut n = 0usize;
    for s in scores
        .iter()
        .take(run.end().min(scores.len()))
        .skip(run.start)
    {
        let margin = (s - leave_threshold) / hi;
        acc += margin.clamp(0.0, 1.0) as f64;
        n += 1;
    }
    if n == 0 {
        return None;
    }
    Some((acc / n as f64) as f32)
}

/// Build a [`SpeechTimeline`] from frame scores, on the energy detector's grid.
#[must_use]
pub fn build_timeline(
    scores: &[f32],
    config: &SmoothingConfig,
    buffer_start_ms: i64,
    buffer_len_ms: i64,
) -> SpeechTimeline {
    build_timeline_at(scores, config, HOP_MS, buffer_start_ms, buffer_len_ms)
}

/// Build a [`SpeechTimeline`] from frame scores on an explicit `hop_ms` grid.
///
/// `buffer_start_ms` is `AudioBuffer::start_time`; every emitted time is
/// absolute in source-media coordinates, which is the contract
/// `pipeline.rs:40-45` states and the only thing that keeps a 40-minute
/// recording from producing a transcript that is 40 minutes out of sync.
///
/// Padding is applied and then **clipped against the neighbour**, so padded
/// segments can never overlap. The final clamp to `[buffer_start,
/// buffer_start + buffer_len_ms]` means a segment can never claim audio that
/// does not exist even if the frame count is a frame or two optimistic at the
/// tail.
#[must_use]
pub fn build_timeline_at(
    scores: &[f32],
    config: &SmoothingConfig,
    hop_ms: i64,
    buffer_start_ms: i64,
    buffer_len_ms: i64,
) -> SpeechTimeline {
    let states = hysteresis(scores, config);
    let runs = smoothing_passes(runs(&states), config, hop_ms).speech;
    segments_from_runs(
        &runs,
        scores,
        config,
        hop_ms,
        buffer_start_ms,
        buffer_len_ms,
    )
}

/// Padding, clipping and timeline assembly. Split out so the chunker and the
/// tests can build a timeline from runs they already have.
#[must_use]
pub fn segments_from_runs(
    runs: &[Run],
    scores: &[f32],
    config: &SmoothingConfig,
    hop_ms: i64,
    buffer_start_ms: i64,
    buffer_len_ms: i64,
) -> SpeechTimeline {
    let buffer_end = buffer_start_ms.saturating_add(buffer_len_ms.max(0));
    let pad = config.pad_ms;

    let mut segments: Vec<SpeechSegment> = Vec::with_capacity(runs.len());
    let mut cursor = buffer_start_ms;
    for r in runs {
        let mut start = buffer_start_ms + frames_to_ms_at(r.start, hop_ms) - pad;
        let mut end = buffer_start_ms + frames_to_ms_at(r.end(), hop_ms) + pad;
        // Never reach back over the previous segment, and never past the end of
        // the buffer.
        start = start.max(cursor).max(buffer_start_ms);
        end = end.min(buffer_end).max(start);
        if end <= start {
            continue;
        }
        cursor = end;
        segments.push(SpeechSegment {
            start: Time(start),
            end: Time(end),
            confidence: segment_confidence(scores, *r, config.leave_threshold),
        });
    }
    let speech_duration_ms = segments
        .iter()
        .map(|s| s.end.as_millis() - s.start.as_millis())
        .sum();
    SpeechTimeline {
        segments,
        speech_duration_ms,
    }
}

/// One segmentation of a score sequence, for tests and for the tuning harness.
#[derive(Debug, Clone, PartialEq)]
pub struct Segmentation {
    /// How many times the fixed-point loop ran before the run list stopped
    /// changing. Useful as a canary on the smoothing rules.
    pub passes: usize,
    /// The surviving speech runs, in frame indices.
    pub speech: Vec<Run>,
}

/// [`smoothing`], but on an explicit `hop_ms` grid and reporting how many
/// passes it took.
///
/// Identical behaviour to [`smoothing`]; the pass count is the only addition
/// and it exists so a test can assert the fixed point is reached in a bounded
/// number of iterations rather than taking the bound on trust.
#[must_use]
pub fn smoothing_passes(rs: Vec<Run>, config: &SmoothingConfig, hop_ms: i64) -> Segmentation {
    let min_silence = config.min_silence_frames_on(hop_ms);
    let min_speech = config.min_speech_frames_on(hop_ms);
    let mut current = rs;
    let mut passes = 0usize;
    for _ in 0..8 {
        passes += 1;
        let before = current.len();
        current = absorb_short_silences(current, min_silence);
        current = drop_short_speech(current, min_speech);
        if current.len() == before {
            break;
        }
    }
    let speech = current.into_iter().filter(|r| r.speech).collect();
    Segmentation { passes, speech }
}

/// Total speech time implied by a set of runs, in milliseconds.
///
/// Independent of padding and of the buffer bounds, so a test can check the
/// arithmetic of the duration rules without involving either.
#[must_use]
pub fn runs_duration_ms(rs: &[Run]) -> i64 {
    runs_duration_ms_at(rs, HOP_MS)
}

/// As [`runs_duration_ms`], on an explicit `hop_ms` grid.
#[must_use]
pub fn runs_duration_ms_at(rs: &[Run], hop_ms: i64) -> i64 {
    rs.iter()
        .filter(|r| r.speech)
        .map(|r| frames_to_ms_at(r.len, hop_ms))
        .sum()
}

/// The nominal frame hop in milliseconds, re-exported so callers do not have
/// to import `features` for one constant.
pub const FRAME_HOP_MS: i64 = HOP_MS;

#[cfg(test)]
mod tests {
    use super::*;

    /// A state sequence: `s` for speech, `.` for silence.
    fn states(spec: &str) -> Vec<bool> {
        spec.chars().map(|c| c == 's').collect()
    }

    /// Turn a state spec into a run list with runs of at least one frame,
    /// interleaving silence runs of one frame where the spec has no gap.
    fn run_list(speech_lens: &[usize], gap: usize) -> Vec<Run> {
        let mut out = Vec::new();
        let mut at = 0usize;
        for (i, &l) in speech_lens.iter().enumerate() {
            if i > 0 {
                out.push(Run {
                    start: at,
                    len: gap,
                    speech: false,
                });
                at += gap;
            }
            out.push(Run {
                start: at,
                len: l,
                speech: true,
            });
            at += l;
        }
        out
    }

    fn cfg(min_speech: i64, min_silence: i64) -> SmoothingConfig {
        SmoothingConfig {
            min_speech_ms: min_speech,
            min_silence_ms: min_silence,
            ..SmoothingConfig::default()
        }
    }

    #[test]
    fn hysteresis_uses_two_thresholds_not_one() {
        let c = SmoothingConfig::default();
        // Expressed *relative to the thresholds* rather than as literals, so
        // this states the rule instead of the current tuning constants.
        let mid = (c.enter_threshold + c.leave_threshold) / 2.0;
        let just_under_enter = c.enter_threshold - 0.01;
        let just_over_leave = c.leave_threshold + 0.01;
        // Below enter, above leave: does not start speech.
        assert_eq!(hysteresis(&[just_under_enter], &c), vec![false]);
        // At or above enter: starts.
        assert_eq!(hysteresis(&[c.enter_threshold], &c), vec![true]);
        // Inside the band, having entered: holds.
        assert_eq!(
            hysteresis(&[1.0, mid, just_over_leave], &c),
            vec![true, true, true],
            "a frame between the thresholds must not end a run"
        );
        // Below leave: ends.
        assert_eq!(
            hysteresis(&[1.0, c.leave_threshold - 0.01], &c),
            vec![true, false]
        );
        // And then a frame inside the band does not restart.
        assert_eq!(
            hysteresis(&[1.0, c.leave_threshold - 0.01, mid], &c),
            vec![true, false, false]
        );
    }

    #[test]
    fn hysteresis_on_an_empty_or_constant_sequence() {
        let c = SmoothingConfig::default();
        assert!(hysteresis(&[], &c).is_empty());
        assert_eq!(hysteresis(&vec![1.0; 100], &c), vec![true; 100]);
        assert_eq!(hysteresis(&vec![0.0; 100], &c), vec![false; 100]);
    }

    #[test]
    fn equal_thresholds_are_reported_as_invalid() {
        // A zero-width hysteresis band is the chattering configuration; it
        // must be refused rather than silently corrected, because a silent
        // correction is how a "why is this VAD producing 40 segments a second"
        // bug gets in.
        let c = SmoothingConfig {
            enter_threshold: 0.5,
            leave_threshold: 0.5,
            ..SmoothingConfig::default()
        };
        assert!(!c.is_valid());
        assert!(SmoothingConfig::default().is_valid());
        let c = SmoothingConfig {
            enter_threshold: 0.4,
            leave_threshold: 0.6,
            ..SmoothingConfig::default()
        };
        assert!(!c.is_valid(), "leave above enter is inverted, not valid");
    }

    #[test]
    fn runs_splits_a_state_sequence_into_maximal_runs() {
        let r = runs(&states("ss.ss"));
        assert_eq!(r.len(), 3);
        assert_eq!(
            r[0],
            Run {
                start: 0,
                len: 2,
                speech: true
            }
        );
        assert_eq!(
            r[1],
            Run {
                start: 2,
                len: 1,
                speech: false
            }
        );
        assert_eq!(
            r[2],
            Run {
                start: 3,
                len: 2,
                speech: true
            }
        );
        assert!(runs(&[]).is_empty());
        assert_eq!(runs(&[true]).len(), 1);
    }

    #[test]
    fn a_30ms_click_does_not_become_a_segment() {
        // 3 frames at a 10 ms hop. The frame rule fires -- the click is
        // loud and it is broadband -- and only the duration floor stops it.
        let c = SmoothingConfig::default();
        let mut scores = vec![0.0f32; 100];
        for s in scores.iter_mut().take(3).skip(1) {
            *s = 0.95;
        }
        let tl = build_timeline(&scores, &c, 0, 1_000);
        assert!(
            tl.segments.is_empty(),
            "a 30 ms click produced {:?}",
            tl.segments
        );
        assert_eq!(tl.speech_duration_ms, 0);
    }

    #[test]
    fn a_50ms_gap_does_not_split_a_segment() {
        // Two 2-second bursts separated by 50 ms of silence. Real speech does
        // this between words, and cutting there is what produces one ASR chunk
        // per word.
        let c = SmoothingConfig::default();
        let mut scores = vec![0.9f32; 200 + 5 + 200];
        for s in scores.iter_mut().take(205).skip(200) {
            *s = 0.0;
        }
        let tl = build_timeline(&scores, &c, 0, 5_000);
        assert_eq!(tl.segments.len(), 1, "50 ms gap split the segment");
        let seg = tl.segments[0];
        // 4.05 s of speech (405 frames), padded by 60 ms at the end. The
        // leading pad is clamped at the buffer start, so it is not double
        // counted.
        assert_eq!(seg.start.as_millis(), 0);
        assert_eq!(seg.end.as_millis(), 4_050 + 60);
    }

    #[test]
    fn a_five_second_pause_does_split_a_segment() {
        let c = SmoothingConfig::default();
        let mut scores = vec![0.9f32; 200 + 500 + 200];
        for s in scores.iter_mut().take(700).skip(200) {
            *s = 0.0;
        }
        let tl = build_timeline(&scores, &c, 0, 10_000);
        assert_eq!(tl.segments.len(), 2, "a 5 s pause did not split");
        assert!(tl.segments[0].end.as_millis() <= 2_060);
        assert!(tl.segments[1].start.as_millis() >= 7_000 - 60);
    }

    #[test]
    fn short_silence_absorption_then_short_speech_dropping_reaches_a_fixed_point() {
        // Speech 100 / gap 5 / speech 1 / gap 5 / speech 100. The middle
        // 1-frame run is below the 20-frame speech floor, so dropping it
        // widens the two 5-frame gaps to 11 frames -- still below the 30-frame
        // silence floor, so a second absorption pass is *required* to get the
        // right answer. One pass of each rule would emit two segments here.
        let c = cfg(200, 300);
        let rs = run_list(&[100, 1, 100], 5);
        let seg = smoothing_passes(rs, &c, HOP_MS);
        assert_eq!(seg.speech.len(), 1, "expected one merged segment");
        assert!(
            seg.passes <= 3,
            "fixed point took {} passes; the bound is 8",
            seg.passes
        );
    }

    #[test]
    fn a_short_silence_at_the_buffer_edges_is_left_alone() {
        // There is nothing to merge it with, and shortening it gains nothing.
        let rs = vec![
            Run {
                start: 0,
                len: 3,
                speech: false,
            },
            Run {
                start: 3,
                len: 100,
                speech: true,
            },
            Run {
                start: 103,
                len: 2,
                speech: false,
            },
        ];
        let out = absorb_short_silences(rs, 30);
        // All three survive. The leading silence has no speech run *before* it
        // to be absorbed into, and `absorb_short_silences` starts its scan at
        // index 1 precisely so that index 0 cannot be mistaken for one. The
        // trailing 2-frame silence likewise has no speech run after it.
        assert_eq!(out.len(), 3, "{out:?}");
        assert!(!out[0].speech && out[0].len == 3);
        assert!(out[1].speech && out[1].len == 100);
        assert!(!out[2].speech && out[2].len == 2);
    }

    #[test]
    fn dropping_a_short_speech_run_widens_the_silences_around_it() {
        // 100 speech / 50 silence / 5 speech / 50 silence / 100 speech. The
        // middle run is below the 20-frame floor, so dropping it leaves one
        // 105-frame silence spanning all of it -- the dropped run's frames
        // become silence rather than a hole, which is the behaviour `retain`
        // got wrong. Merging two runs into one *segment* is `smoothing`'s job;
        // here we only assert that the two silences are now one maximal run.
        let rs = run_list(&[100, 5, 100], 50);
        let out = drop_short_speech(rs, 20);
        assert_eq!(out.len(), 3, "{out:?}");
        assert!(out[0].speech && out[0].len == 100);
        assert!(!out[1].speech && out[1].len == 105, "{out:?}");
        assert!(out[2].speech && out[2].len == 100);
        assert_eq!(out[2].start, 205, "{out:?}");
        // And the run list is still maximal: no two adjacent runs share a state.
        for w in out.windows(2) {
            assert_ne!(w[0].speech, w[1].speech, "{out:?}");
            assert_eq!(w[0].end(), w[1].start, "{out:?}");
        }
    }

    #[test]
    fn coalesce_runs_merges_adjacent_same_state_runs() {
        let rs = vec![
            Run {
                start: 0,
                len: 5,
                speech: true,
            },
            Run {
                start: 5,
                len: 3,
                speech: false,
            },
            Run {
                start: 8,
                len: 2,
                speech: false,
            },
            Run {
                start: 10,
                len: 7,
                speech: true,
            },
        ];
        let out = coalesce_runs(rs);
        assert_eq!(out.len(), 3);
        assert_eq!(
            out[1],
            Run {
                start: 5,
                len: 5,
                speech: false
            }
        );
        assert_eq!(
            out[2],
            Run {
                start: 10,
                len: 7,
                speech: true
            }
        );
        assert!(coalesce_runs(vec![]).is_empty());
    }

    #[test]
    fn the_fixed_point_loop_terminates_on_a_pathological_run_list() {
        // Alternating long speech and single-frame silences: every absorption
        // pass removes runs, so the loop must terminate. Assert the bound
        // rather than trusting it.
        let mut rs = Vec::new();
        let mut at = 0usize;
        for i in 0..200 {
            if i > 0 {
                rs.push(Run {
                    start: at,
                    len: 1,
                    speech: false,
                });
                at += 1;
            }
            rs.push(Run {
                start: at,
                len: 25,
                speech: true,
            });
            at += 25;
        }
        let seg = smoothing_passes(rs, &cfg(200, 300), HOP_MS);
        assert!(seg.passes <= 8, "took {} passes", seg.passes);
        assert_eq!(seg.speech.len(), 1);
    }

    #[test]
    fn padding_is_clipped_against_the_neighbour_so_segments_never_overlap() {
        let c = SmoothingConfig::default();
        // Two 200 ms bursts 400 ms apart, so 2 x 60 ms of padding cannot
        // overlap. The gap has to be long enough that the smoothing rules do
        // not absorb it first, or the test passes for the wrong reason.
        let mut scores = vec![0.0f32; 200];
        for s in scores.iter_mut().take(20) {
            *s = 0.9;
        }
        for s in scores.iter_mut().take(80).skip(60) {
            *s = 0.9;
        }
        let tl = build_timeline(&scores, &c, 0, 2_000);
        assert_eq!(tl.segments.len(), 2, "{tl:?}");
        assert!(
            tl.segments[0].end <= tl.segments[1].start,
            "{:?} overlaps {:?}",
            tl.segments[0],
            tl.segments[1]
        );
        // Both are padded by 60 ms, and the padding is not clipped because the
        // gap is 400 ms.
        assert_eq!(tl.segments[0].start.as_millis(), 0);
        assert_eq!(tl.segments[0].end.as_millis(), 260);
        assert_eq!(tl.segments[1].start.as_millis(), 540);
    }

    #[test]
    fn segments_are_clamped_to_the_buffer_and_never_have_end_at_or_before_start() {
        let c = SmoothingConfig::default();
        // A 200-frame run would nominally end at 2000 ms + 60 ms of padding,
        // past a 1000 ms buffer.
        let scores = vec![0.9f32; 200];
        let tl = build_timeline(&scores, &c, 0, 1_000);
        assert_eq!(tl.segments.len(), 1);
        assert_eq!(tl.segments[0].end.as_millis(), 1_000);
        // And a zero-length buffer produces nothing at all.
        let tl = build_timeline(&scores, &c, 0, 0);
        assert!(tl.segments.is_empty());
    }

    #[test]
    fn times_are_absolute_and_rebase_with_the_buffer_start() {
        // The 40-minute-out-of-sync bug. A buffer that starts at 5 minutes must
        // produce times near 300 000, not near zero.
        let c = SmoothingConfig::default();
        let scores = vec![0.9f32; 200];
        let tl = build_timeline(&scores, &c, 300_000, 2_000);
        assert_eq!(tl.segments.len(), 1);
        assert_eq!(tl.segments[0].start.as_millis(), 300_000);
        assert_eq!(tl.segments[0].end.as_millis(), 302_000);
    }

    #[test]
    fn speech_duration_is_the_sum_of_the_segments_and_the_ratio_is_in_range() {
        let c = SmoothingConfig::default();
        let mut scores = vec![0.0f32; 600];
        for s in scores.iter_mut().take(100) {
            *s = 0.9;
        }
        for s in scores.iter_mut().take(300).skip(200) {
            *s = 0.9;
        }
        let tl = build_timeline(&scores, &c, 0, 6_000);
        let sum: i64 = tl
            .segments
            .iter()
            .map(|s| s.end.as_millis() - s.start.as_millis())
            .sum();
        assert_eq!(tl.speech_duration_ms, sum);
        let r = tl.speech_ratio(Time(6_000));
        assert!((0.0..=1.0).contains(&r), "ratio {r}");
        assert!(
            r > 0.2 && r < 0.5,
            "ratio {r} is not plausible for 2 s of 6 s"
        );
    }

    #[test]
    fn confidence_ranks_a_solid_segment_above_a_marginal_one() {
        // This is the entire contract of the number: an ordering key.
        let c = SmoothingConfig::default();
        let solid = vec![0.95f32; 100];
        let marginal = vec![0.39f32; 100];
        let rs = [Run {
            start: 0,
            len: 100,
            speech: true,
        }];
        let a = segment_confidence(&solid, rs[0], c.leave_threshold).unwrap();
        let b = segment_confidence(&marginal, rs[0], c.leave_threshold).unwrap();
        assert!(a > b, "solid {a} should beat marginal {b}");
        assert!((0.0..=1.0).contains(&a));
        // A frame exactly at the leave threshold has zero margin.
        let at_leave = vec![c.leave_threshold; 100];
        assert_eq!(
            segment_confidence(&at_leave, rs[0], c.leave_threshold),
            Some(0.0)
        );
    }

    #[test]
    fn confidence_is_none_rather_than_a_guess_when_there_is_nothing_to_say() {
        let rs = [Run {
            start: 0,
            len: 10,
            speech: true,
        }];
        assert_eq!(segment_confidence(&[], rs[0], 0.38), None);
        // leave_threshold == 1 leaves no headroom to measure a margin in.
        assert_eq!(segment_confidence(&[0.9; 10], rs[0], 1.0), None);
        // A run that lies entirely past the end of the score array.
        let past = [Run {
            start: 50,
            len: 10,
            speech: true,
        }];
        assert_eq!(segment_confidence(&[0.9; 10], past[0], 0.38), None);
    }

    #[test]
    fn build_timeline_is_total_over_adversarial_score_input() {
        let c = SmoothingConfig::default();
        let cases: Vec<Vec<f32>> = vec![
            vec![],
            vec![0.0; 1],
            vec![1.0; 1],
            vec![f32::NAN; 50],
            vec![f32::INFINITY; 50],
            vec![-1.0; 50],
            vec![1.0; 1000],
            {
                let mut v = vec![0.9f32; 1000];
                for s in v.iter_mut().step_by(7) {
                    *s = 0.0;
                }
                v
            },
        ];
        for scores in cases {
            let tl = build_timeline(&scores, &c, 0, 10_000);
            for w in tl.segments.windows(2) {
                assert!(w[0].end <= w[1].start, "overlap: {:?} {:?}", w[0], w[1]);
            }
            for s in &tl.segments {
                assert!(s.start < s.end, "empty segment {s:?}");
                assert!(s.start.as_millis() >= 0);
                assert!(s.end.as_millis() <= 10_000);
            }
            let r = tl.speech_ratio(Time(10_000));
            assert!((0.0..=1.0).contains(&r), "ratio {r}");
        }
    }

    #[test]
    fn runs_duration_is_the_arithmetic_sum_of_the_speech_runs() {
        let rs = run_list(&[100, 200], 50);
        // 1000 + 2000 = 3000 ms of speech, 500 ms of silence not counted.
        assert_eq!(runs_duration_ms(&rs), 3_000);
        assert_eq!(runs_duration_ms(&[]), 0);
        assert_eq!(FRAME_HOP_MS, 10);
    }
}

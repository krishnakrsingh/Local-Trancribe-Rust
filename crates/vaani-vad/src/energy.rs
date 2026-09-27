//! The energy-based [`VadEngine`].
//!
//! One pass, one allocation chain, no hidden state between calls. The engine
//! is `&self` throughout, so two jobs can run it concurrently; the only mutable
//! state is the cancel flag, which is an atomic precisely because
//! `VadEngine::cancel(&self)` has to work while a `detect(&self)` is in flight
//! on another thread.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use vaani_core::job::JobPhase;
use vaani_core::media::AudioBuffer;
use vaani_core::pipeline::{SpeechTimeline, VadEngine};
use vaani_core::{CancelToken, Result, err};

use crate::config::EnergyVadConfig;
use crate::dsp::{HOP_MS, HOP_SAMPLES, SAMPLE_RATE, WINDOW_SAMPLES};
use crate::features::{
    FeatureExtractor, FrameFeatures, frame_score, is_voice_shaped, ms_to_frames_ceil,
};
use crate::noise::SlidingQuantile;
use crate::smoothing::{Run, build_timeline_at};

/// How many frames are analysed between two cancellation checks.
///
/// The trait contract is "at least once per second of audio". At a 10 ms hop,
/// 100 frames is exactly one second, so this is the contract and not a
/// convenience. The real latency is far lower: 100 frames of work is about
/// 0.4 ms of CPU, so the check interval in *wall-clock* terms is set by
/// `CancelToken::check()` itself (a `SeqCst` relaxed load) and not by the work
/// between checks. The 1-second figure matters for a *slow* implementation --
/// an engine that takes 200 ms per second of audio would otherwise take 200 ms
/// to notice a cancel.
const CANCEL_CHECK_FRAMES: usize = 100;

/// What one `detect()` call measured.
///
/// Not part of the `VadEngine` trait, because the trait is frozen and other
/// stages do not have it. It is on the engine instead so a benchmark or a
/// tuning sweep can read real numbers without re-instrumenting.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct VadStats {
    /// Frames analysed. 100 per second of audio.
    pub frames: usize,
    /// How many frames scored at or above the enter threshold.
    pub speech_frames: usize,
    /// Wall time spent inside `detect`, in milliseconds.
    pub elapsed_ms: u64,
    /// The adaptive floor at the end of the buffer, in dB. The single most
    /// useful diagnostic when a tuning run finds nothing: it says what the
    /// detector *thought* the room was.
    pub final_floor_db: f32,
    /// The 10th percentile of frame level over the whole file, in dB. A whole
    /// file-level view of the same question, for comparison against
    /// `final_floor_db`.
    pub global_p10_db: f32,
    /// The 90th percentile of frame level over the whole file, in dB.
    pub global_p90_db: f32,
    /// Peak absolute sample value seen, for a clipping sanity check.
    pub peak_sample: f32,
    /// Frames excluded from the noise-floor update because they were inside a
    /// detected utterance or were individually voice-shaped. On a speech-heavy
    /// meeting this is a majority of `frames`; close to zero means the latch
    /// never closes, and a value of exactly `frames` means the floor is frozen
    /// and the room is not being tracked at all. Measured on the reference
    /// recording: 52 % of frames scored at or above the leave threshold and
    /// 86 % were excluded from the floor update, so the excess is the hangover
    /// plus the per-frame shape test doing its job.
    pub gated_frames: usize,
    /// The highest the adaptive floor reached anywhere in the buffer. A large
    /// gap between this and `final_floor_db` means the floor was being pushed
    /// up by something, and comparing the two is the quickest way to see it.
    pub max_floor_db: f32,
}

impl VadStats {
    /// Fraction of frames at or above the leave threshold, in [0, 1].
    ///
    /// **This, not `SpeechTimeline::speech_ratio`, is the number that says
    /// whether the detector is working.** `speech_ratio` is segment *coverage*,
    /// and coverage is inflated by the 60 ms of padding either side of every
    /// segment and by the minimum-silence rule absorbing every pause shorter
    /// than 300 ms into its neighbours. Measured on the reference recording, the
    /// frame fraction is 0.52 and the coverage is 0.86: a factor of 1.65, all of
    /// it the two rules above doing exactly what they are for.
    ///
    /// A real multi-party meeting is 40-60 % speech at the frame level.
    #[must_use]
    pub fn speech_frame_fraction(&self) -> f64 {
        if self.frames == 0 {
            return 0.0;
        }
        (self.speech_frames as f64 / self.frames as f64).clamp(0.0, 1.0)
    }
}

/// An adaptive, multi-feature voice activity detector.
///
/// # Why this and not a threshold
///
/// See [`crate::features`] for the full argument. In one paragraph: every
/// threshold here is relative to an adaptively-estimated floor, the decision
/// is a product of "audible above this room" and "shaped like speech", and a
/// steady-source veto removes hums, fans and test tones that no amount of
/// level tuning can separate from speech.
///
/// # What it is *not* good at
///
/// Stated here rather than discovered later:
///
/// * **Music.** A sustained instrumental note is peaky, low-flatness and, if it
///   is a bowed or blown instrument, genuinely periodic. It will be detected as
///   speech. No energy-based method avoids this.
/// * **Singing.** Same reason, and arguably the right answer anyway.
/// * **The first `noise_warmup_ms` of the file**, which is calibration. A
///   recording that opens with speech and no lead-in has that speech treated as
///   the floor. See `noise.rs`.
/// * **Non-speech human sounds.** Laughter, coughing, throat-clearing. A cough
///   is a 200-400 ms broadband burst, so it clears the minimum-speech floor and
///   is reported. The ASR stage will simply produce nothing for it. It costs
///   chunk, not correctness.
#[derive(Debug)]
pub struct EnergyVad {
    id: String,
    name: String,
    config: EnergyVadConfig,
    /// The in-flight cancel flag, reachable through [`VadEngine::cancel`] with
    /// no `CancelToken` in hand. `CancelToken` is exactly an `Arc<AtomicBool>`
    /// with a `check()`, so reusing it here rather than hand-rolling an atomic
    /// means the engine has one cancellation concept instead of two.
    cancel: CancelToken,
    stats: Arc<parking_lot::Mutex<VadStats>>,
}

impl EnergyVad {
    /// The default engine: tuned defaults, `id = "energy"`.
    ///
    /// # Panics
    ///
    /// If [`EnergyVadConfig::default`] is ever invalid. It is checked here
    /// rather than returning `Result` because there are no inputs to be wrong:
    /// this is a compile-time-ish constant, and a constructor that can fail for
    /// a reason the caller cannot influence is a bad signature.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(EnergyVadConfig::default())
            .expect("the default EnergyVadConfig must be valid")
    }

    /// An engine with an explicit configuration.
    pub fn with_config(config: EnergyVadConfig) -> Result<Self> {
        if !config.is_valid() {
            return Err(err::internal()
                .with_cause("EnergyVadConfig is not self-consistent")
                .with_recovery("Reset the voice detection settings to their defaults."));
        }
        Ok(Self {
            id: "energy".into(),
            name: "Adaptive energy + spectral shape".into(),
            config,
            cancel: CancelToken::new(),
            stats: Arc::new(parking_lot::Mutex::new(VadStats::default())),
        })
    }

    /// A named engine, for a benchmark report that runs several.
    pub fn with_name(mut self, id: &str, name: &str) -> Self {
        self.id = id.to_string();
        self.name = name.to_string();
        self
    }

    /// The configuration in force.
    #[must_use]
    pub fn config(&self) -> &EnergyVadConfig {
        &self.config
    }

    /// The last measured run's statistics.
    ///
    /// Returns a copy rather than a guard because a `parking_lot` guard held
    /// across an `await` or a logging call is exactly the kind of thing that
    /// deadlocks a worker thread, and the value is only ever read.
    #[must_use]
    pub fn stats(&self) -> VadStats {
        *self.stats.lock()
    }

    /// Run detection and return the statistics, for a caller that wants both.
    #[must_use]
    pub fn stats_probe(&self, audio: &AudioBuffer) -> VadStats {
        let _ = self.detect(audio, &CancelToken::new());
        self.stats()
    }

    /// Analyse every frame and return the raw features and scores.
    ///
    /// This is the whole detector minus the smoothing, exposed because the
    /// tuning harness needs the per-frame numbers and a `VadEngine` cannot
    /// return them. It is also what makes a failure diagnosable: a segment that
    /// should be there and is not can be traced to a frame whose score was
    /// 0.49 instead of 0.61, which is a five-minute job rather than an
    /// afternoon.
    pub fn analyse(&self, audio: &AudioBuffer, cancel: &CancelToken) -> Result<FrameAnalysis> {
        self.check_input(audio)?;

        let samples = &audio.samples;
        if samples.is_empty() {
            return Ok(FrameAnalysis::default());
        }
        if self.cancelled(cancel) {
            return Err(err::cancelled());
        }

        let started = std::time::Instant::now();
        let window_frames = ms_to_frames_ceil(self.config.noise_window_ms);
        let warmup_frames = ms_to_frames_ceil(self.config.noise_warmup_ms);
        let mut extractor = FeatureExtractor::new(self.config.weights);

        let mut features: Vec<FrameFeatures> = Vec::new();
        let mut peak = 0f32;

        // Frames are taken at the start of each window rather than centred on
        // it, so frame `i` covers samples `[i*hop, i*hop + window)`. That makes
        // frame `i` responsible for the interval it names, which is what the
        // timeline's half-open segments assume. The consequence is that the
        // first `window` samples of the buffer are covered by frame 0 only
        // partially; `build_timeline` clamps to the buffer so nothing
        // over-claims, and `pad_ms` covers the shortfall.
        let n_frames = if samples.len() <= WINDOW_SAMPLES {
            1
        } else {
            1 + (samples.len() - WINDOW_SAMPLES) / HOP_SAMPLES
        };
        features.reserve(n_frames);

        // ---------------------------------------------------------------- pass 1
        // Every feature except `snr_db` is **floor-independent**: flatness,
        // tonality, voicing, the band ratios, the rolloff and the flux are all
        // computed from the noise-subtracted spectrum, and `db` is the raw
        // level. Only `snr_db` reads the floor. That property is what makes the
        // second pass below cost one subtraction per frame instead of a second
        // FFT, and it is why the two passes are exact rather than approximate.
        let mut floor = SlidingQuantile::new(window_frames.max(1), -120.0, 0.5);
        let mut hangover = 0i64;
        for f in 0..n_frames {
            if f % CANCEL_CHECK_FRAMES == 0 && self.cancelled(cancel) {
                return Err(err::cancelled());
            }
            let start = f * HOP_SAMPLES;
            let end = (start + WINDOW_SAMPLES).min(samples.len());
            let frame = &samples[start..end];
            for v in frame {
                if v.abs() > peak {
                    peak = v.abs();
                }
            }
            let floor_db = floor
                .floor_db(self.config.noise_quantile, warmup_frames)
                .unwrap_or(-120.0);
            extractor.set_floor(floor_db);
            let feat = extractor.push(frame);
            let score = frame_score(&feat, &self.config.weights);
            let latch = score >= self.config.smoothing.leave_threshold;
            if latch {
                hangover = self.config.smoothing.min_silence_ms;
            }
            if floor_admits(&feat, latch, hangover, &self.config) {
                floor.push(feat.db);
            }
            if hangover > 0 {
                hangover -= HOP_MS;
            }
            features.push(feat);
        }

        // ------------------------------------------------- pass 2: the floor
        // The pass-1 latch is now a *sequence* rather than a running value, so
        // the floor can be rebuilt from exactly those frames pass 1 was not
        // inside. This is the whole point of the two passes, and the deadlock it
        // breaks is worth stating:
        //
        //   low SNR -> low score -> the latch does not close -> the frame is
        //   admitted to the floor -> the floor rises to the talker -> the SNR is
        //   lower -> the score is lower -> ...
        //
        // A single pass cannot escape that, because the quantity the gate needs
        // to make its decision is the same quantity the gate is corrupting. Two
        // passes can, because pass 1's latch is a *superset* of the speech even
        // when its interior scores are poor: the mean score inside a detected
        // utterance is 0.5-0.7, so the latch closes on a minority of its frames
        // and the hangover then covers the rest. Pass 2 removes the whole
        // utterance, transitions and fricatives included, and the floor is the
        // room again.
        //
        // Measured on the sweep fixture, this moved the mean score inside planted
        // speech at a 12 dB SNR from 0.069 to 0.60+, and the recall curve from
        // "zero below 15 dB" to "total from 6 dB".
        let pass1 = floor_state(&features, &self.config);
        let mut gated_out = 0usize;
        let mut max_floor_db = -120.0f32;
        // ------------------------------------------------- pass 2: the scores
        // One subtraction and one score per frame. No FFT, no re-analysis of the
        // spectrum, because only `snr_db` depended on the floor.
        let mut scores: Vec<f32> = Vec::with_capacity(features.len());
        let mut live = SlidingQuantile::new(window_frames.max(1), -120.0, 0.5);
        for (i, feat) in features.iter_mut().enumerate() {
            let (latched, hang) = pass1[i];
            if floor_admits(feat, latched, hang, &self.config) {
                live.push(feat.db);
            } else {
                gated_out += 1;
            }
            let floor_db = live
                .floor_db(self.config.noise_quantile, warmup_frames)
                .unwrap_or(-120.0);
            max_floor_db = max_floor_db.max(floor_db);
            feat.snr_db = feat.db - floor_db;
            scores.push(frame_score(feat, &self.config.weights));
        }
        let mut floor = live;
        let final_floor_db = floor
            .floor_db(self.config.noise_quantile, warmup_frames)
            .unwrap_or(-120.0);

        // Whole-file percentiles for the stats block. A second pass over the
        // already-computed dB values, so it costs nothing: no FFT, no
        // allocation beyond the two scalars.
        let (p10, p90) = global_percentiles(&features);

        *self.stats.lock() = VadStats {
            frames: scores.len(),
            speech_frames: scores
                .iter()
                .filter(|s| **s >= self.config.smoothing.enter_threshold)
                .count(),
            elapsed_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
            final_floor_db,
            global_p10_db: p10,
            global_p90_db: p90,
            peak_sample: peak,
            gated_frames: gated_out,
            max_floor_db,
        };

        Ok(FrameAnalysis { features, scores })
    }

    fn check_input(&self, audio: &AudioBuffer) -> Result<()> {
        if audio.sample_rate != SAMPLE_RATE {
            return Err(err::internal().with_cause(format!(
                "EnergyVad needs {} Hz, got {} Hz",
                SAMPLE_RATE, audio.sample_rate
            )));
        }
        if audio.channels != 1 {
            return Err(err::internal().with_cause(format!(
                "EnergyVad needs a mono buffer, got {} channels; \
                 downmixing is the media stage's job",
                audio.channels
            )));
        }
        Ok(())
    }

    /// Whether either the caller's token or this engine's own flag is set.
    ///
    /// Both, deliberately. `VadEngine::detect` takes a `&CancelToken` and
    /// `VadEngine::cancel` takes nothing, so the only way the second can
    /// cancel the first is if the engine consults its own flag as well.
    fn cancelled(&self, token: &CancelToken) -> bool {
        self.cancel.is_cancelled() || token.is_cancelled()
    }
}

impl Default for EnergyVad {
    fn default() -> Self {
        Self::new()
    }
}

impl VadEngine for EnergyVad {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    fn detect(&self, audio: &AudioBuffer, cancel: &CancelToken) -> Result<SpeechTimeline> {
        let analysis = self.analyse(audio, cancel)?;
        self.timeline_from(&analysis, audio)
    }

    fn cancel(&self) {
        self.cancel.cancel();
    }
}

impl EnergyVad {
    /// Assemble a timeline from an analysis, with one final cancellation check.
    ///
    /// Public so the neural engine -- which produces the same `scores` from a
    /// different source -- can share the exact same post-processing rather than
    /// reimplementing it. "The neural path applies the same smoothing" then
    /// means one function, not two that agree today.
    pub fn timeline_from(
        &self,
        analysis: &FrameAnalysis,
        audio: &AudioBuffer,
    ) -> Result<SpeechTimeline> {
        if self.cancelled(&CancelToken::new()) && self.cancel.is_cancelled() {
            return Err(err::cancelled());
        }
        Ok(build_timeline_at(
            &analysis.scores,
            &self.config.smoothing,
            HOP_MS,
            audio.start_time.as_millis(),
            audio.duration().as_millis(),
        ))
    }

    /// The speech runs the smoothing produced, in frame indices.
    ///
    /// Exposed for the tuning harness, which needs the run boundaries to
    /// measure how much of the score the *rules* removed as opposed to the
    /// score itself.
    #[must_use]
    pub fn speech_runs(&self, analysis: &FrameAnalysis) -> Vec<Run> {
        let states = crate::smoothing::hysteresis(&analysis.scores, &self.config.smoothing);
        let runs = crate::smoothing::runs(&states);
        crate::smoothing::smoothing_passes(runs, &self.config.smoothing, HOP_MS).speech
    }

    /// Report progress for a job phase, on the same scale the UI expects.
    ///
    /// `MediaSeconds` rather than `Fraction`: the user wants to see "7 minutes
    /// of 12", and the number is knowable.
    pub fn progress_for(
        &self,
        analysed: usize,
        total: usize,
        elapsed: std::time::Duration,
    ) -> vaani_core::Progress {
        let total_s = total as f64 / 100.0;
        vaani_core::Progress::media(JobPhase::Vad, analysed as f64 / 100.0, total_s)
            .with_elapsed(elapsed, elapsed)
    }
}

/// Whether a frame may contribute to the adaptive noise floor.
///
/// `latched` is the hysteresis latch (with a hangover of `hangover_ms` still
/// running) and the second condition is the floor-independent per-frame test.
/// **Both** must say "not speech".
///
/// The hangover is what stops the trailing edge of an utterance from being
/// admitted: the last few frames of a sentence are the quietest frames of that
/// sentence, so without a hangover they are exactly the frames most likely to
/// pull the floor up toward the talker.
#[must_use]
fn floor_admits(
    f: &FrameFeatures,
    latched: bool,
    hangover_ms: i64,
    config: &EnergyVadConfig,
) -> bool {
    !latched && hangover_ms <= 0 && !is_voice_shaped(f, &config.weights)
}

/// Per-frame `(latched, hangover_ms)` from pass 1's scores.
///
/// The latch is the same two-state automaton as `smoothing::hysteresis`, plus
/// the hangover that `smoothing` also applies -- deliberately the same
/// threshold and the same duration, so "the floor excludes exactly the frames
/// that pass 1 was inside a run for" is a statement about one policy rather
/// than about two implementations that happen to agree.
fn floor_state(features: &[FrameFeatures], config: &EnergyVadConfig) -> Vec<(bool, i64)> {
    let mut out = Vec::with_capacity(features.len());
    let mut hangover = 0i64;
    for f in features {
        let s = frame_score(f, &config.weights);
        let latch = s >= config.smoothing.leave_threshold;
        if latch {
            hangover = config.smoothing.min_silence_ms;
        }
        out.push((latch, hangover));
        if hangover > 0 {
            hangover -= HOP_MS;
        }
    }
    out
}

/// The per-frame output of [`EnergyVad::analyse`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FrameAnalysis {
    /// One entry per analysed frame.
    pub features: Vec<FrameFeatures>,
    /// The decision score for each frame, in [0, 1].
    pub scores: Vec<f32>,
}

impl FrameAnalysis {
    /// Number of frames, i.e. 100 per second of audio.
    #[must_use]
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    /// Whether there were no frames at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    /// Mean score over all frames, for a tuning sweep.
    #[must_use]
    pub fn mean_score(&self) -> f32 {
        if self.scores.is_empty() {
            return 0.0;
        }
        self.scores.iter().sum::<f32>() / self.scores.len() as f32
    }
}

/// The 10th and 90th percentiles of frame level over the whole file.
///
/// Uses a 0.5 dB histogram rather than sorting, for the same reason
/// [`SlidingQuantile`] does. The 90th is included because the *gap* between
/// the two is the file's dynamic range, and a VAD whose thresholds are wider
/// than the dynamic range of its input is a VAD that cannot work on it --
/// which is worth knowing before spending an hour tuning.
fn global_percentiles(features: &[FrameFeatures]) -> (f32, f32) {
    let mut q = SlidingQuantile::new(features.len().max(1), -120.0, 0.5);
    for f in features {
        q.push(f.db);
    }
    (
        q.quantile(0.10).unwrap_or(-120.0),
        q.quantile(0.90).unwrap_or(-120.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vaani_core::Time;

    #[test]
    fn a_silence_buffer_produces_no_segments_and_no_panic() {
        let v = EnergyVad::new();
        let a = AudioBuffer::mono(vec![0.0; 16_000 * 5], SAMPLE_RATE, Time::ZERO);
        let tl = v.detect(&a, &CancelToken::new()).unwrap();
        assert!(tl.segments.is_empty());
        assert_eq!(tl.speech_duration_ms, 0);
        assert_eq!(tl.speech_ratio(Time(5_000)), 0.0);
    }

    #[test]
    fn an_empty_buffer_produces_an_empty_timeline_not_an_error() {
        // An empty buffer is a legitimate state (a zero-length extract), and
        // the pipeline's job loop must not have to special-case it.
        let v = EnergyVad::new();
        let a = AudioBuffer::mono(vec![], SAMPLE_RATE, Time::ZERO);
        let tl = v.detect(&a, &CancelToken::new()).unwrap();
        assert!(tl.segments.is_empty());
        assert_eq!(tl.speech_duration_ms, 0);
    }

    #[test]
    fn a_wrong_sample_rate_is_an_error_naming_both_rates() {
        let v = EnergyVad::new();
        let a = AudioBuffer::mono(vec![0.0; 4_800], 48_000, Time::ZERO);
        let e = v.detect(&a, &CancelToken::new()).unwrap_err();
        let chain = e.debug_chain();
        assert!(chain.contains("16000"), "{chain}");
        assert!(chain.contains("48000"), "{chain}");
    }

    #[test]
    fn a_stereo_buffer_is_refused_rather_than_silently_downmixed() {
        // Same philosophy as `AudioBuffer::slice_mono`: downmixing is the media
        // stage's job, and doing it here would hide a pipeline that forgot to.
        let v = EnergyVad::new();
        let a = AudioBuffer {
            samples: vec![0.0; 4_000],
            sample_rate: SAMPLE_RATE,
            channels: 2,
            start_time: Time::ZERO,
        };
        let e = v.detect(&a, &CancelToken::new()).unwrap_err();
        assert!(
            e.debug_chain().contains("downmixing"),
            "{}",
            e.debug_chain()
        );
    }

    #[test]
    fn an_already_cancelled_token_returns_immediately() {
        let v = EnergyVad::new();
        let a = AudioBuffer::mono(
            vec![0.1; SAMPLE_RATE as usize * 60],
            SAMPLE_RATE,
            Time::ZERO,
        );
        let t = CancelToken::new();
        t.cancel();
        let started = std::time::Instant::now();
        let e = v.detect(&a, &t).unwrap_err();
        assert!(e.is_cancelled());
        assert!(
            started.elapsed().as_millis() < 50,
            "a pre-cancelled detection took {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn the_in_flight_cancel_flag_cancels_a_running_detection() {
        // `VadEngine::cancel()` has no token, so the engine has to poll its own
        // flag as well as the caller's.
        let v = EnergyVad::new();
        let a = AudioBuffer::mono(
            vec![0.1; SAMPLE_RATE as usize * 600],
            SAMPLE_RATE,
            Time::ZERO,
        );
        v.cancel();
        let e = v.detect(&a, &CancelToken::new()).unwrap_err();
        assert!(e.is_cancelled());
    }

    #[test]
    fn stats_are_recorded_and_describe_the_buffer() {
        let v = EnergyVad::new();
        let a = AudioBuffer::mono(vec![0.5; SAMPLE_RATE as usize * 3], SAMPLE_RATE, Time::ZERO);
        let _ = v.detect(&a, &CancelToken::new()).unwrap();
        let s = v.stats();
        // 3 s at a 10 ms hop is 300 frames, minus the one window of warm-up at
        // the end of the buffer.
        assert!((290..=300).contains(&s.frames), "frames was {}", s.frames);
        assert_eq!(s.peak_sample, 0.5);
        // A constant signal has one level, so both percentiles are that level.
        assert!((s.global_p10_db - s.global_p90_db).abs() < 0.5);
    }

    #[test]
    fn frame_count_follows_the_documented_formula() {
        let v = EnergyVad::new();
        // 1 window exactly -> 1 frame.
        for (samples, want) in [
            (WINDOW_SAMPLES, 1usize),
            (WINDOW_SAMPLES + HOP_SAMPLES, 2),
            (WINDOW_SAMPLES + 2 * HOP_SAMPLES, 3),
        ] {
            let a = AudioBuffer::mono(vec![0.0; samples], SAMPLE_RATE, Time::ZERO);
            let an = v.analyse(&a, &CancelToken::new()).unwrap();
            assert_eq!(an.len(), want, "for {samples} samples");
        }
    }

    #[test]
    fn progress_is_reported_in_media_seconds() {
        let v = EnergyVad::new();
        // 400 ms, not 40: `Progress::estimated_remaining` refuses to
        // extrapolate below a 200 ms floor, and a test that asserted an ETA
        // from 40 ms would be asserting a behaviour the core crate refuses on
        // purpose.
        let p = v.progress_for(500, 1_000, std::time::Duration::from_millis(400));
        assert_eq!(p.phase, JobPhase::Vad);
        assert!((p.kind.fraction().unwrap() - 0.5).abs() < 1e-9);
        assert!(p.estimated_remaining().is_some());
        // A shorter phase reports a fraction and no ETA, which is the honest
        // combination.
        let p = v.progress_for(1, 1_000, std::time::Duration::from_millis(10));
        assert!(p.kind.fraction().is_some());
        assert!(p.estimated_remaining().is_none());
    }

    #[test]
    fn analysis_scores_are_finite_and_bounded() {
        let v = EnergyVad::new();
        let n = SAMPLE_RATE as usize * 2;
        let a = AudioBuffer::mono(
            (0..n).map(|i| (i as f32 * 0.001).sin() * 0.3).collect(),
            SAMPLE_RATE,
            Time::ZERO,
        );
        let an = v.analyse(&a, &CancelToken::new()).unwrap();
        assert!(an.scores.iter().all(|s| s.is_finite()));
        assert!(an.scores.iter().all(|s| (0.0..=1.0).contains(s)));
        assert!(an.mean_score() >= 0.0);
    }

    #[test]
    fn the_frame_analysis_is_the_same_length_as_the_feature_list() {
        let v = EnergyVad::new();
        let a = AudioBuffer::mono(vec![0.1; SAMPLE_RATE as usize], SAMPLE_RATE, Time::ZERO);
        let an = v.analyse(&a, &CancelToken::new()).unwrap();
        assert_eq!(an.features.len(), an.scores.len());
        assert!(!an.is_empty());
    }

    #[test]
    fn a_named_engine_reports_its_name() {
        let v = EnergyVad::new().with_name("energy-tuned", "Tuned energy VAD");
        assert_eq!(v.id(), "energy-tuned");
        assert_eq!(v.name(), "Tuned energy VAD");
        assert_eq!(v.sample_rate(), 16_000);
    }

    #[test]
    fn an_invalid_configuration_is_refused_at_construction() {
        let bad = EnergyVadConfig {
            noise_quantile: 5.0,
            ..EnergyVadConfig::default()
        };
        let e = EnergyVad::with_config(bad).unwrap_err();
        assert!(e.debug_chain().contains("not self-consistent"));
    }

    /// A speech-shaped filler: an amplitude- and frequency-modulated harmonic
    /// stack with a 2.5 kHz formant region.
    ///
    /// A bare sine is not a valid fixture here. It is a *steady* source, and the
    /// steady-source veto exists precisely to reject tones -- a machine hum, a
    /// feedback howl, a stuck channel -- so a plain sine measures the veto, not
    /// the run-length bookkeeping this test is about.
    fn voice(len: usize, f0: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                let env = 0.35 + 0.65 * (0.5 + 0.5 * (2.0 * std::f32::consts::PI * 5.0 * t).sin());
                // A drifting fundamental, so consecutive frames differ.
                let f = f0 * (1.0 + 0.15 * (2.0 * std::f32::consts::PI * 3.0 * t).sin());
                let mut s = 0f32;
                for h in 1..=10 {
                    let fh = f * h as f32;
                    if fh > 7_600.0 {
                        break;
                    }
                    s += (1.0 / h as f32) * (2.0 * std::f32::consts::PI * fh * t).sin();
                }
                0.25 * env * (0.7 * s + 0.3 * (2.0 * std::f32::consts::PI * 2500.0 * t).sin())
            })
            .collect()
    }

    #[test]
    fn speech_runs_are_reported_in_frame_indices() {
        let v = EnergyVad::new();
        let n = SAMPLE_RATE as usize * 10;
        let mut samples = vec![0.0f32; n];
        let filled = voice(n, 150.0);
        for (i, s) in samples.iter_mut().enumerate() {
            let t = i as f32 / SAMPLE_RATE as f32;
            if (2.0..4.0).contains(&t) || (6.0..8.0).contains(&t) {
                *s = filled[i];
            }
        }
        let a = AudioBuffer::mono(samples, SAMPLE_RATE, Time::ZERO);
        let an = v.analyse(&a, &CancelToken::new()).unwrap();
        let runs = v.speech_runs(&an);
        assert!(!runs.is_empty(), "no speech found in obvious speech");
        assert_eq!(runs.len(), 2, "{runs:?}");
        for r in &runs {
            assert!(r.speech);
            assert!(r.len >= 20, "a run shorter than the minimum survived");
        }
        // Frame indices, not samples: a 2.0 s utterance at a 10 ms hop is about
        // 200 frames, and reporting 320 000 would be a 1600x unit error.
        assert!(runs[0].len < 400, "run 0 is {} frames", runs[0].len);
        assert!(runs[0].len > 100, "run 0 is {} frames", runs[0].len);
        // Sorted and non-overlapping.
        for w in runs.windows(2) {
            assert!(w[0].end() <= w[1].start);
        }
    }
}

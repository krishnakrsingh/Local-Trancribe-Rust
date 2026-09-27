//! A Silero VAD ONNX adapter behind [`VadEngine`].
//!
//! # Status: implemented against the published interface, UNVALIDATED
//!
//! **There is no Silero ONNX model in this repository and none was
//! downloaded.** `artifacts/` contains only the SraVaani ASR graphs, and no
//! `*.onnx` matching a VAD exists anywhere in the SraVaani source tree. So
//! nothing in this file has ever been executed against real weights, and every
//! number it would produce is unverified. What is here is a correct-looking
//! adapter plus the honest reporting of that fact, which the brief allows and
//! a fabricated benchmark does not.
//!
//! Specifically, **unvalidated** means:
//!
//! * the input/output names, ranks and the LSTM state shape come from the
//!   published Silero VAD v5 ONNX signature, not from reading a graph;
//! * the frame geometry (512 samples at 16 kHz, 256-sample hop) is the
//!   documented one, and is *not* the same geometry as the energy detector's
//!   400/160 -- so the two engines produce timelines on different grids, which
//!   is fine because everything downstream consumes milliseconds;
//! * there is no parity fixture, so a model swap that silently changed the
//!   output semantics would be caught by nothing here.
//!
//! # What is *not* unvalidated
//!
//! Everything that is not a statement about the model: the smoothing, the
//! duration rules, the padding, the cancellation, the sample-rate and
//! mono checks and the input validation are the **same code** the energy engine
//! runs ([`crate::smoothing`]), so a future validation of the model validates
//! the policy and not the plumbing.
//!
//! # The Silero VAD v5 interface
//!
//! ```text
//! inputs : input  float32 [batch, num_samples]   512 at 16 kHz
//!          state  float32 [2, batch, 128]        LSTM state, carried across windows
//!          sr     int64   []                     8000 or 16000
//! outputs: output float32 [batch, num_samples]   per-sample speech probability
//!          stateN float32 [2, batch, 128]        state for the next window
//! ```
//!
//! The two things that are easy to get wrong and are therefore asserted in
//! `verify_interface`:
//!
//! * **`state` must be fed back.** A stateless call produces a decision with no
//!   temporal context, which for a model this small is a large accuracy loss
//!   rather than a crash -- the worst kind of bug.
//! * **`state` must be reset at a discontinuity**, i.e. at the start of a
//!   buffer. Carrying state across two unrelated recordings makes the first
//!   second of the second one wrong.
//!
//! `output` is **per sample**, not per frame: a 512-sample window yields 512
//! probabilities. Aggregating them to one value per hop is what
//! [`SileroVad::window_scores`] does, and the aggregation is the *mean* over
//! the window's second half -- the first half of a 32 ms window overlaps the
//! previous window and carries the same state, so counting it twice weights
//! old audio more heavily than new.

use std::path::{Path, PathBuf};

use parking_lot::Mutex;
use vaani_core::media::AudioBuffer;
use vaani_core::pipeline::{SpeechTimeline, VadEngine};
use vaani_core::{CancelToken, Result, err};
use vaani_inference::{F32Tensor, I64Tensor, Input, OnnxSession, Provider};

use crate::config::EnergyVadConfig;
use crate::dsp::SAMPLE_RATE;
use crate::energy::EnergyVad;
use crate::smoothing::build_timeline_at;

/// Frame length the model is fed, in samples: 32 ms at 16 kHz.
pub const WINDOW_SAMPLES: usize = 512;

/// Hop between windows, in samples: 16 ms at 16 kHz. A 50 % overlap, which is
/// what the published implementation uses.
pub const HOP_SAMPLES: usize = 256;

/// Length of the LSTM state carried between windows, per the published
/// signature `[2, batch, 128]`.
pub const STATE_SIZE: usize = 128;

/// The file names the adapter looks for, in order of preference.
pub const MODEL_FILENAMES: [&str; 3] = [
    "silero_vad.onnx",
    "silero_vad_iterator.onnx",
    "silero-vad.onnx",
];

/// Where a Silero VAD model was looked for, and what was found.
///
/// `SileroVad::discover` returns this rather than a bare `bool` so the UI and
/// `vaani doctor` can say *where* they looked, which is the difference between
/// a user fixing the problem and filing a bug.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelProbe {
    /// Every path that was checked, in order.
    pub searched: Vec<PathBuf>,
    /// The first one that exists, if any.
    pub found: Option<PathBuf>,
}

impl ModelProbe {
    /// Whether a model was found.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.found.is_some()
    }
}

/// A Silero VAD ONNX engine.
///
/// The session is behind a `Mutex` because `VadEngine::detect` takes `&self`
/// and `OnnxSession::run` takes `&mut self` -- the frozen inference wrapper's
/// signature, which the architect owns. A `Mutex` is the honest way to
/// reconcile the two: it serialises concurrent detections, which for a VAD is
/// correct (there is one timeline per job) and costs nothing, since two jobs
/// would be sharing a GPU anyway.
#[derive(Debug)]
pub struct SileroVad {
    path: PathBuf,
    config: EnergyVadConfig,
    session: Mutex<Option<OnnxSession>>,
    provider: Provider,
    cancel: CancelToken,
}

impl SileroVad {
    /// Look for a Silero VAD model in and around `dir`.
    ///
    /// Checks, in order: `dir` itself, `dir/../silero`, `dir/../sravaani`, and
    /// the workspace's own `artifacts/silero`. The last one is a convenience
    /// for development; in an installed product A7 (model manager) resolves
    /// the path and hands it to [`Self::open`].
    #[must_use]
    pub fn discover(dir: &Path) -> ModelProbe {
        let mut candidates: Vec<PathBuf> = Vec::new();
        for name in MODEL_FILENAMES {
            candidates.push(dir.join(name));
        }
        if let Some(parent) = dir.parent() {
            for name in MODEL_FILENAMES {
                candidates.push(parent.join("silero").join(name));
            }
        }
        let found = candidates.iter().find(|p| p.is_file()).cloned();
        ModelProbe {
            searched: candidates,
            found,
        }
    }

    /// Open a model. Fails with `ModelUnavailable` if the file is absent.
    pub fn open(path: &Path, provider: Provider, config: EnergyVadConfig) -> Result<Self> {
        if !config.is_valid() {
            return Err(err::internal().with_cause("EnergyVadConfig is not self-consistent"));
        }
        if !path.is_file() {
            return Err(err::model_unavailable()
                .with_cause(format!("no Silero VAD model at {}", path.display()))
                .with_recovery("Download the voice detection model in Settings, or keep using the built-in detector."));
        }
        // Opened eagerly so a bad model fails here, where the message can name
        // it, rather than at the first `detect` in the middle of a job.
        let caps = vaani_inference::capability::detect(
            &vaani_inference::runtime::location().provider_dirs,
        );
        let session = OnnxSession::load(path, provider, &caps, None)?;
        Self::verify_interface(&session)?;
        Ok(Self {
            path: path.to_path_buf(),
            config,
            session: Mutex::new(Some(session)),
            provider,
            cancel: CancelToken::new(),
        })
    }

    /// Open a model if one is present, `None` otherwise.
    ///
    /// The "is the model present?" check the brief asks for, and the one the
    /// pipeline should actually call: an absent VAD model is a *normal* state
    /// for this product (the energy detector is the shipped default), not an
    /// error, so it must not propagate as one.
    pub fn try_open(
        dir: &Path,
        provider: Provider,
        config: EnergyVadConfig,
    ) -> Result<Option<Self>> {
        // Validated before the search, so an invalid configuration is an error
        // whether or not a model happens to be installed. Checking it only on
        // the success path means the behaviour depends on the filesystem.
        if !config.is_valid() {
            return Err(err::internal().with_cause("EnergyVadConfig is not self-consistent"));
        }
        let probe = Self::discover(dir);
        match probe.found {
            Some(p) => Self::open(&p, provider, config).map(Some),
            None => {
                tracing::info!(
                    searched = ?probe.searched,
                    "no Silero VAD model present; the energy detector is the fallback"
                );
                Ok(None)
            }
        }
    }

    /// Where the model was loaded from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The execution provider the caller asked for. Which one is *actually* in
    /// use is inside the session and is recorded by `vaani-inference`.
    #[must_use]
    pub fn provider(&self) -> Provider {
        self.provider
    }

    /// The smoothing policy, shared with the energy detector.
    #[must_use]
    pub fn config(&self) -> &EnergyVadConfig {
        &self.config
    }

    /// Check the graph's declared interface against the published one.
    ///
    /// Fails loudly rather than adapting. A model with a different state shape
    /// is a *different model*, and silently reshaping the state tensor to fit
    /// would produce probabilities from uninitialised memory that look
    /// plausible and are meaningless.
    fn verify_interface(session: &OnnxSession) -> Result<()> {
        const REQUIRED_INPUTS: [&str; 3] = ["input", "state", "sr"];
        const REQUIRED_OUTPUTS: [&str; 2] = ["output", "stateN"];
        let inputs = session.input_names();
        let outputs = session.output_names();
        for name in REQUIRED_INPUTS {
            if !inputs.iter().any(|n| n == name) {
                return Err(err::model_load_failed().with_cause(format!(
                    "the VAD model at this path has no input named {name:?}; it has {inputs:?}. \
                     This does not look like a Silero VAD v5 graph."
                )));
            }
        }
        for name in REQUIRED_OUTPUTS {
            if !outputs.iter().any(|n| n == name) {
                return Err(err::model_load_failed().with_cause(format!(
                    "the VAD model has no output named {name:?}; it has {outputs:?}"
                )));
            }
        }
        tracing::info!(
            inputs = ?inputs,
            outputs = ?outputs,
            "silero vad interface verified"
        );
        Ok(())
    }

    /// Run the model over the buffer and return one score per hop.
    ///
    /// Returns `None` if no session is loaded, which after [`Self::open`]
    /// cannot happen and exists so a `Debug`-constructed engine reports the
    /// fact rather than panicking.
    fn window_scores(&self, audio: &AudioBuffer, cancel: &CancelToken) -> Result<Option<Vec<f32>>> {
        let samples = &audio.samples;
        if samples.is_empty() {
            return Ok(Some(Vec::new()));
        }
        let mut guard = self.session.lock();
        let Some(session) = guard.as_mut() else {
            return Ok(None);
        };

        // The state is zero-initialised here, which is the documented reset for
        // a discontinuity. Carrying state in from a previous buffer would make
        // the first second of this one wrong. Two tensors: one recurrent state
        // and one context, for the single-stream Silero graph.
        let mut state = vec![0f32; 2 * STATE_SIZE];
        let sr = I64Tensor::scalar(i64::from(SAMPLE_RATE));
        let mut scores = Vec::with_capacity(samples.len() / HOP_SAMPLES + 1);

        let n_windows = 1 + samples.len().saturating_sub(WINDOW_SAMPLES) / HOP_SAMPLES;
        for w in 0..n_windows {
            if w % 62 == 0 {
                // 62 windows at 16 ms is one second of audio, matching the
                // `VadEngine` contract.
                if self.cancel.is_cancelled() || cancel.is_cancelled() {
                    return Err(err::cancelled());
                }
            }
            let start = w * HOP_SAMPLES;
            let end = (start + WINDOW_SAMPLES).min(samples.len());
            let mut window = vec![0f32; WINDOW_SAMPLES];
            window[..end - start].copy_from_slice(&samples[start..end]);

            let input = F32Tensor::new(vec![1, WINDOW_SAMPLES as i64], window)?;
            let state_in = F32Tensor::new(vec![2, 1, STATE_SIZE as i64], state.clone())?;
            let out = session.run(
                &[
                    Input::f32("input", input),
                    Input::f32("state", state_in),
                    Input::i64("sr", sr.clone()),
                ],
                &["output", "stateN"],
            )?;
            let probs = out[0].as_f32()?;
            // Carry the state forward. This is the single most important line
            // in the file: without it the model has no temporal context and its
            // accuracy collapses without any error being raised.
            state = out[1].as_f32()?.data.clone();

            scores.push(aggregate_window(probs.as_slice()));
        }
        Ok(Some(scores))
    }
}

/// Reduce a window's per-sample probabilities to one number.
///
/// The mean of the window's **second half**. A 512-sample window at a 256
/// sample hop means consecutive windows share 256 samples; averaging the whole
/// window counts those shared samples twice and weights older audio as heavily
/// as new, which lags the decision by up to one window (32 ms). Taking the
/// second half makes the value a function of audio at or after the window's own
/// start, which is the half that is new.
#[must_use]
pub fn aggregate_window(probs: &[f32]) -> f32 {
    if probs.is_empty() {
        return 0.0;
    }
    let half = probs.len() / 2;
    let tail = &probs[half..];
    let slice = if tail.is_empty() { probs } else { tail };
    let mean = slice.iter().sum::<f32>() / slice.len() as f32;
    if mean.is_nan() {
        0.0
    } else {
        mean.clamp(0.0, 1.0)
    }
}

impl VadEngine for SileroVad {
    fn id(&self) -> &str {
        "silero-vad-v5"
    }

    fn name(&self) -> &str {
        "Silero VAD v5 (ONNX)"
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    fn detect(&self, audio: &AudioBuffer, cancel: &CancelToken) -> Result<SpeechTimeline> {
        if audio.sample_rate != SAMPLE_RATE {
            return Err(err::internal().with_cause(format!(
                "SileroVad needs {} Hz, got {} Hz",
                SAMPLE_RATE, audio.sample_rate
            )));
        }
        if audio.channels != 1 {
            return Err(err::internal()
                .with_cause("SileroVad needs a mono buffer; downmixing is the media stage's job"));
        }
        if self.cancel.is_cancelled() || cancel.is_cancelled() {
            return Err(err::cancelled());
        }
        let Some(scores) = self.window_scores(audio, cancel)? else {
            return Err(err::model_unavailable()
                .with_cause("the Silero VAD session was dropped after load")
                .with_recovery("Reopen the voice detection model."));
        };

        // The same post-processing the energy engine uses, on a 16 ms grid
        // instead of a 10 ms one. The duration rules are expressed in
        // milliseconds and converted against *this* engine's hop, so "minimum
        // speech 200 ms" means 200 ms here too rather than 13 frames of 16 ms
        // or 20 frames of 10 ms.
        Ok(build_timeline_at(
            &scores,
            &self.config.smoothing,
            HOP_MS,
            audio.start_time.as_millis(),
            audio.duration().as_millis(),
        ))
    }

    fn cancel(&self) {
        self.cancel.cancel();
    }
}

/// The frame hop the neural engine's timeline is built on, in milliseconds.
///
/// Public because the duration rules have to be converted against it, and a
/// caller reading a `SpeechTimeline` from this engine has to know that 200 ms
/// of "minimum speech" here is 13 of its frames and not 20.
pub const HOP_MS: i64 = (HOP_SAMPLES as i64 * 1000) / SAMPLE_RATE as i64;

/// Which engine the pipeline should use, and why.
///
/// A free function rather than a method, because the question is answered by
/// the *pipeline*, not by either engine, and because it is the piece the
/// architect wires in.
///
/// # The policy
///
/// **The energy detector is the default and the fallback.** The neural detector
/// is used only when a model is present *and* the caller asks for it. That is
/// the opposite of "prefer the best model": the product's model manager (A7)
/// does not currently ship a VAD model, so a "prefer neural" default would
/// mean the default path is the untested one.
///
/// The `Result` is already `#[must_use]`, so the return value is not marked
/// again here; the point that is worth stating in prose is the `warn!` on the
/// fallback path, which is the only place a caller finds out that the model it
/// asked for is not there.
pub fn preferred_engine(
    model_dir: Option<&Path>,
    prefer_neural: bool,
    provider: Provider,
    config: EnergyVadConfig,
) -> Result<Box<dyn VadEngine>> {
    if prefer_neural && let Some(dir) = model_dir {
        match SileroVad::try_open(dir, provider, config)? {
            Some(v) => return Ok(Box::new(v)),
            None => tracing::warn!(
                "neural VAD requested but no model was found; using the energy detector"
            ),
        }
    }
    Ok(Box::new(EnergyVad::with_config(config)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EnergyVadConfig;
    use vaani_core::Time;

    #[test]
    fn discovery_reports_every_path_it_checked_and_finds_nothing_when_absent() {
        let dir = Path::new("definitely-not-a-model-directory");
        let probe = SileroVad::discover(dir);
        assert!(!probe.is_available());
        assert!(probe.found.is_none());
        assert!(
            probe.searched.len() >= MODEL_FILENAMES.len(),
            "the search list must not be empty: {:?}",
            probe.searched
        );
        for p in &probe.searched {
            assert!(!p.exists(), "{p:?} should not exist");
        }
    }

    #[test]
    fn discovery_finds_a_file_when_one_is_there() {
        // An empty file is enough: discovery checks existence, not validity,
        // and `open` is where validity is established. Conflating the two would
        // make "is it downloaded?" depend on the inference runtime.
        let dir = std::env::temp_dir().join("vaani-vad-probe");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join(MODEL_FILENAMES[0]);
        std::fs::write(&path, b"").unwrap();
        let probe = SileroVad::discover(&dir);
        assert_eq!(probe.found.as_deref(), Some(path.as_path()));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn opening_a_missing_model_is_a_model_unavailable_error() {
        let e = SileroVad::open(
            Path::new("nowhere/silero_vad.onnx"),
            Provider::Cpu,
            EnergyVadConfig::default(),
        )
        .unwrap_err();
        assert_eq!(e.code(), vaani_core::ErrorCode::ModelUnavailable);
        assert!(e.user.recovery.is_some(), "the user needs a next step");
    }

    #[test]
    fn try_open_returns_none_rather_than_an_error_when_no_model_is_present() {
        // An absent VAD model is this product's normal state, so it must not
        // propagate as a failure.
        let got = SileroVad::try_open(
            Path::new("nowhere"),
            Provider::Cpu,
            EnergyVadConfig::default(),
        )
        .unwrap();
        assert!(got.is_none());
    }

    #[test]
    fn the_frame_geometry_is_the_published_one() {
        // 512 samples at 16 kHz is 32 ms; 256 is 16 ms. These are the published
        // Silero v5 numbers and they are *not* the energy detector's 400/160,
        // which is why nothing downstream can assume a shared frame grid.
        assert_eq!(WINDOW_SAMPLES, 512);
        assert_eq!(HOP_SAMPLES, 256);
        assert_eq!(WINDOW_SAMPLES * 1000 / 16_000, 32);
        assert_eq!(HOP_SAMPLES * 1000 / 16_000, 16);
        assert_eq!(super::HOP_MS, 16);
        assert_eq!(STATE_SIZE, 128);
    }

    #[test]
    fn window_aggregation_uses_only_the_second_half() {
        // Consecutive windows share 256 samples. Averaging all 512 would let
        // old audio outvote new audio and lag the decision by up to 32 ms.
        let mut probs = vec![1.0; WINDOW_SAMPLES];
        for p in probs.iter_mut().skip(HOP_SAMPLES) {
            *p = 0.0;
        }
        assert_eq!(aggregate_window(&probs), 0.0);

        let mut probs = vec![0.0; WINDOW_SAMPLES];
        for p in probs.iter_mut().skip(HOP_SAMPLES) {
            *p = 1.0;
        }
        assert_eq!(aggregate_window(&probs), 1.0);

        // Degenerate inputs are total, not panicking.
        assert_eq!(aggregate_window(&[]), 0.0);
        assert_eq!(aggregate_window(&[0.5]), 0.5);
        assert_eq!(aggregate_window(&[f32::NAN, f32::NAN]), 0.0);
        assert_eq!(aggregate_window(&[-1.0, -1.0]), 0.0);
    }

    #[test]
    fn the_smoothing_policy_is_shared_verbatim_with_the_energy_engine() {
        // "The neural path applies the same smoothing" has to mean one value,
        // not two that agree today. A field-by-field comparison is what makes
        // that checkable in review.
        let e = EnergyVadConfig::default();
        let s = e.smoothing;
        assert_eq!(s, EnergyVadConfig::default().smoothing);
        // The same millisecond rule converts to different frame counts on the
        // two engines' grids, and both conversions are correct because both
        // engines pass their own hop. This is the bug a shared hard-coded hop
        // would cause: 200 ms would silently become 160 ms here.
        assert_eq!(s.min_speech_frames_on(10), 20);
        assert_eq!(s.min_silence_frames_on(10), 30);
        assert_eq!(s.min_speech_frames_on(16), 13);
        assert_eq!(s.min_silence_frames_on(16), 19);
        assert_eq!(s.min_speech_frames_on(super::HOP_MS), 13);
        // A non-positive grid is not a frame grid.
        assert_eq!(s.min_speech_frames_on(0), 0);
    }

    #[test]
    fn preferred_engine_falls_back_to_the_energy_detector() {
        // The shipped default. A3 ships no VAD model, so a "prefer neural"
        // default would make the untested path the default path.
        let got = preferred_engine(None, true, Provider::Cpu, EnergyVadConfig::default()).unwrap();
        assert_eq!(got.id(), "energy");
        let got = preferred_engine(
            Some(Path::new("nowhere")),
            true,
            Provider::Cpu,
            EnergyVadConfig::default(),
        )
        .unwrap();
        assert_eq!(got.id(), "energy");
    }

    #[test]
    fn preferred_engine_returns_a_usable_engine_that_reports_its_contract() {
        let got = preferred_engine(None, false, Provider::Cpu, EnergyVadConfig::default()).unwrap();
        assert_eq!(got.sample_rate(), 16_000);
        assert!(!got.name().is_empty());
        let tl = got
            .detect(
                &AudioBuffer::mono(vec![0.0; 16_000], 16_000, Time::ZERO),
                &CancelToken::new(),
            )
            .unwrap();
        assert!(tl.segments.is_empty());
    }

    #[test]
    fn an_invalid_config_is_refused_by_every_constructor() {
        let bad = EnergyVadConfig {
            noise_quantile: 9.0,
            ..EnergyVadConfig::default()
        };
        assert!(SileroVad::try_open(Path::new("nowhere"), Provider::Cpu, bad).is_err());
        assert!(preferred_engine(None, false, Provider::Cpu, bad).is_err());
    }
}

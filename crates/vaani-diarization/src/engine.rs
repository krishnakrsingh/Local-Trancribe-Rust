//! The `DiarizationEngine` implementation and the pipeline that fills it in.
//!
//! # What one `diarize` call does
//!
//! ```text
//!  AudioBuffer
//!      |
//!  [1] activity pass          one 256-point transform per frame, streaming.
//!      |                      two f32 per frame is the only per-frame array
//!      |                      the engine ever holds for the whole file.
//!      v
//!  SpeechTimeline             thresholds from this recording's own percentiles
//!      |                      plus a spectral-flatness gate.
//!      |
//!  [2] feature pass           full cepstra, pitch and spectrum per frame, but
//!      |                      reduced to one vector per 1.5 s window as it
//!      |                      goes, so peak state is one window.
//!      v
//!  Vec<WindowEmbedding>       median over frames -> z-score with a
//!      |                      MinSpread floor -> block weights -> L2
//!      v
//!  clustering                 agglomerative (average linkage, cosine) down to
//!      |                      a single cluster, then a stopping rule, then the
//!      |                      min-speech guard, then Lloyd refinement.
//!      v
//!  per-frame labels           every active frame assigned to a centroid
//!      |
//!  [3] smoothing              median filter, minimum turn, adjacent merge
//!      v
//!  DiarizationResult          sorted, non-overlapping, tight spans
//! ```
//!
//! # Memory
//!
//! Nothing here holds a full-resolution spectrogram, and that is a constraint
//! rather than an accident. The only per-frame array is the activity pass's
//! two floats per frame: 8.6 MB for a three-hour recording. The feature pass
//! holds one window's worth of per-frame state (75 frames x 83 floats =
//! 25 kB), the window vectors cost 83 x 4 B x `N` (about 4.8 MB for a
//! three-hour meeting), and the clustering's distance matrix is capped at
//! `max_cluster_windows^2 x 4 B` — 16 MB at the default cap. Total additional
//! allocation is single-digit megabytes on top of the `AudioBuffer` the caller
//! already holds, which is why a 14-minute meeting and a three-hour one cost
//! about the same.
//!
//! # Cancellation
//!
//! The stage contract asks for at least one cancellation check per second of
//! audio. `cancel_poll_frames` defaults to 50 frames, which is 0.5 s, and
//! every long loop checks there. `cancel()` sets an internal flag as well as
//! cancelling any token, so the UI's cancel button works even when no job
//! token is reachable.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use vaani_core::media::AudioBuffer;
use vaani_core::pipeline::{DiarizationEngine, DiarizationResult, SpeakerSpan, SpeechTimeline};
use vaani_core::{CancelToken, Result, Time, err};
use vaani_inference::{MachineCapabilities, Provider};

use crate::config::{DiarizationConfig, REQUIRED_SAMPLE_RATE, SpeakerCount};
use crate::dsp::FrontEnd;
use crate::features::{Layout, WindowAccumulator, WindowEmbedding, normalise_in_place};
use crate::{attribution, cluster, embed, smooth, vad};

/// What the engine actually is, for the settings panel and the benchmark
/// report.
///
/// This exists because `DiarizationEngine::id` returns a `&str` and there is
/// no honest way to say "a spectral heuristic that happens to be running
/// because there are no weights installed" in a single token. The id is
/// `vaani-diarization-heuristic`; this struct says the rest.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct DiarizationCapabilities {
    /// Whether a trained speaker embedder is loaded. False in every build that
    /// ships today, because no diarization weights exist in this repository.
    pub neural_embeddings: bool,
    /// The model file behind the neural path, when one is loaded.
    pub neural_model: Option<String>,
    /// Whether the speaker count was estimated rather than supplied.
    pub speaker_count_estimated: bool,
    /// Whether the built-in activity detector is in use. False when the VAD
    /// stage supplied a timeline.
    pub internal_activity_detector: bool,
}

/// Timing and shape information from one run, for the benchmark report.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct DiarizationTimings {
    /// Pass 1: the activity scan, in milliseconds.
    pub activity_ms: u64,
    /// Pass 2: features, in milliseconds.
    pub features_ms: u64,
    /// Agglomeration, stopping rule, guard and refinement, in milliseconds.
    pub cluster_ms: u64,
    /// Median filtering and turn assembly, in milliseconds.
    pub smooth_ms: u64,
    /// Analysis frames the engine visited.
    pub frames: usize,
    /// Windows with enough speech to be embedded.
    pub windows: usize,
    /// Active frames, from the activity pass.
    pub active_frames: usize,
    /// Clusters produced, before the min-speech guard.
    pub clusters: usize,
    /// Which stopping rule fired.
    pub rule: &'static str,
    /// Wall time for the whole call.
    pub total_ms: u64,
}

impl DiarizationTimings {
    /// The realtime factor: wall time over audio duration. Lower is faster.
    #[must_use]
    pub fn rtf(&self, audio_ms: i64) -> f64 {
        if audio_ms <= 0 {
            return 0.0;
        }
        self.total_ms as f64 / audio_ms as f64
    }
}

/// The heuristic diarization engine.
#[derive(Debug)]
pub struct HeuristicDiarizer {
    config: DiarizationConfig,
    cancel_flag: AtomicBool,
    loaded: AtomicBool,
    embedder: parking_lot::Mutex<Option<embed::NeuralEmbedder>>,
    provider: Provider,
    caps: MachineCapabilities,
    state: parking_lot::Mutex<EngineState>,
}

#[derive(Debug, Default)]
struct EngineState {
    timings: Option<DiarizationTimings>,
    capabilities: Option<DiarizationCapabilities>,
}

impl HeuristicDiarizer {
    /// The stable id, matching the capability this build provides.
    pub const ID: &'static str = "vaani-diarization-heuristic";
    /// The human name.
    pub const NAME: &'static str = "Speaker separation (spectral heuristic)";

    /// An engine with the default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(DiarizationConfig::default())
    }

    /// An engine with an explicit configuration.
    #[must_use]
    pub fn with_config(config: DiarizationConfig) -> Self {
        let caps = DiarizationCapabilities {
            neural_embeddings: false,
            neural_model: None,
            speaker_count_estimated: !matches!(config.clusters.count, SpeakerCount::Fixed(_)),
            internal_activity_detector: true,
        };
        Self {
            config,
            cancel_flag: AtomicBool::new(false),
            loaded: AtomicBool::new(false),
            embedder: parking_lot::Mutex::new(None),
            provider: Provider::Auto,
            caps: vaani_inference::capability::detect(&[]),
            state: parking_lot::Mutex::new(EngineState {
                timings: None,
                capabilities: Some(caps),
            }),
        }
    }

    /// An engine that will use a GPU when one is usable.
    #[must_use]
    pub fn with_provider(mut self, provider: Provider, caps: MachineCapabilities) -> Self {
        self.provider = provider;
        self.caps = caps;
        self
    }

    /// The configuration in force.
    #[must_use]
    pub fn config(&self) -> &DiarizationConfig {
        &self.config
    }

    /// Timings from the most recent successful `diarize`, if any.
    #[must_use]
    pub fn last_timings(&self) -> Option<DiarizationTimings> {
        self.state.lock().timings
    }

    /// What the engine is actually running, for the settings panel.
    #[must_use]
    pub fn capabilities(&self) -> DiarizationCapabilities {
        self.state
            .lock()
            .capabilities
            .clone()
            .unwrap_or(DiarizationCapabilities {
                neural_embeddings: false,
                neural_model: None,
                speaker_count_estimated: true,
                internal_activity_detector: true,
            })
    }

    /// Whether a trained speaker embedder is loaded.
    #[must_use]
    pub fn has_neural_embedder(&self) -> bool {
        self.embedder.lock().is_some()
    }

    /// Diarize, using the built-in activity detector.
    pub fn diarize_with_activity(
        &self,
        audio: &AudioBuffer,
        cancel: &CancelToken,
        progress: &dyn Fn(f64),
    ) -> Result<DiarizationResult> {
        self.run(audio, cancel, progress, None)
    }

    /// Diarize, using a `SpeechTimeline` the VAD stage already produced.
    ///
    /// This is the path the real pipeline takes: `JobPhase::Vad` runs before
    /// `JobPhase::Diarization`, and A3's output is tuned for the decoder
    /// rather than for the feature stability this stage needs. Passing `None`
    /// falls back to the built-in detector.
    pub fn diarize_with_timeline(
        &self,
        audio: &AudioBuffer,
        timeline: Option<&SpeechTimeline>,
        cancel: &CancelToken,
        progress: &dyn Fn(f64),
    ) -> Result<DiarizationResult> {
        self.run(audio, cancel, progress, timeline)
    }

    /// The whole pipeline, given an optional external activity timeline.
    fn run(
        &self,
        audio: &AudioBuffer,
        cancel: &CancelToken,
        progress: &dyn Fn(f64),
        external: Option<&SpeechTimeline>,
    ) -> Result<DiarizationResult> {
        let started = Instant::now();
        self.cancel_flag.store(false, Ordering::SeqCst);
        self.check(cancel)?;

        if audio.channels != 1 {
            return Err(err::internal().with_cause(format!(
                "diarization needs a mono buffer, got {} channels; downmixing is the media stage's job",
                audio.channels
            )));
        }
        if audio.sample_rate != REQUIRED_SAMPLE_RATE {
            // Not resampled here on purpose. A resampler would change the
            // pitch feature by a factor the caller does not know about, and
            // the bug it would hide ("the pipeline handed me 48 kHz") is more
            // expensive to find than this error is to read.
            return Err(err::internal().with_cause(format!(
                "diarization needs {REQUIRED_SAMPLE_RATE} Hz, got {} Hz",
                audio.sample_rate
            )));
        }
        if audio.samples.is_empty() {
            return Ok(DiarizationResult::default());
        }

        let cfg = self.config.clone();
        let frame_ms = cfg.features.frame_ms();
        let start_ms = audio.start_time.as_millis();
        let mut fe = FrontEnd::new(cfg.features, cfg.activity);
        let layout = Layout::of(&cfg.features);
        let poll = cfg.cancel_poll_frames.max(1);
        let usable = fe.usable_frames(audio.samples.len());
        let n_usable = usable.len();
        let mut timings = DiarizationTimings {
            frames: n_usable,
            ..DiarizationTimings::default()
        };

        // --- [1] activity ---------------------------------------------------
        let activity_started = Instant::now();
        let (timeline, internal_detector) = match external {
            Some(tl) => (tl.clone(), false),
            None => {
                let frames = vad::scan(
                    &mut fe,
                    &audio.samples,
                    || self.is_cancelled(cancel),
                    |f| progress(f * 0.15),
                    poll,
                );
                let thresholds = vad::derive_thresholds(&frames, &cfg.activity);
                let mask = vad::activity_mask(&frames, &thresholds);
                (
                    vad::mask_to_timeline(&mask, frame_ms, start_ms, &cfg.activity),
                    true,
                )
            }
        };
        self.check(cancel)?;
        timings.activity_ms = activity_started.elapsed().as_millis() as u64;
        if let Some(cap) = self.state.lock().capabilities.as_mut() {
            cap.internal_activity_detector = internal_detector;
        }

        // Both paths produce a timeline in absolute time with the detector's
        // own hangover already applied, so the per-frame mask is derived from
        // it once, identically, in both cases. Doing it only for an external
        // timeline -- which is what this did for a while -- left the internal
        // path with an all-inactive mask and produced zero windows on a
        // recording that was 60% speech.
        let mut active = vec![false; n_usable];
        let mut seg = 0usize;
        for (i, f) in usable.clone().enumerate() {
            let at = start_ms + fe.frame_time_ms(f);
            while seg < timeline.segments.len() && timeline.segments[seg].end.as_millis() <= at {
                seg += 1;
            }
            if seg < timeline.segments.len() && timeline.segments[seg].start.as_millis() <= at {
                active[i] = true;
            }
        }
        timings.active_frames = active.iter().filter(|a| **a).count();

        // Silence is a complete, correct answer: zero speakers, no spans. It
        // is emphatically not an error, and `Segment::speaker` staying `None`
        // is a normal product state.
        if timeline.segments.is_empty() || n_usable == 0 || !active.iter().any(|a| *a) {
            return self.finish(started, timings, DiarizationResult::default(), progress);
        }

        // --- [2] features --------------------------------------------------
        let features_started = Instant::now();
        let windows = self.embed_windows(
            &mut fe, audio, &active, &usable, frame_ms, start_ms, layout, cancel, progress, poll,
        )?;
        self.check(cancel)?;
        timings.features_ms = features_started.elapsed().as_millis() as u64;
        timings.windows = windows.len();
        if windows.is_empty() {
            return self.finish(started, timings, DiarizationResult::default(), progress);
        }

        // --- [3] clustering ------------------------------------------------
        let cluster_started = Instant::now();
        let mut rows: Vec<Vec<f32>> = windows.iter().map(|w| w.embedding.vector.clone()).collect();
        let speech: Vec<i64> = windows.iter().map(|w| w.embedding.speech_ms).collect();
        // Smooth the feature trajectory *before* normalisation. A 1.5 s window
        // is mostly a description of the five syllables in it, so without this
        // the phoneme-level variation is the same size as the speaker-level
        // variation once z-scored, and the clustering finds twelve speakers in
        // a conversation. See `features::temporal_smooth`.
        crate::features::temporal_smooth(&mut rows, cfg.clusters.feature_smooth_half_windows);
        normalise_in_place(&mut rows, &layout, &cfg.features.weights, &cfg.min_spread);

        // Thin to the clustering cap. The stride is deterministic, so two runs
        // over the same recording cluster the same rows and produce identical
        // speaker indices.
        let cap = cfg.clusters.max_cluster_windows.max(2);
        let stride = windows.len().div_ceil(cap).max(1);
        let sampled: Vec<Vec<f32>> = rows.iter().step_by(stride).cloned().collect();
        let sampled_speech: Vec<i64> = speech.iter().step_by(stride).copied().collect();

        let outcome = cluster::cluster(
            &sampled,
            &sampled_speech,
            &cfg.clusters,
            cfg.effective_min_cluster_speech_ms(),
            cfg.effective_merge_distance(),
        );
        timings.rule = outcome.rule;
        timings.clusters = outcome.num_clusters;
        if outcome.num_clusters == 0 {
            timings.cluster_ms = cluster_started.elapsed().as_millis() as u64;
            return self.finish(started, timings, DiarizationResult::default(), progress);
        }
        progress(0.75);

        // Project every window, sampled or not, onto the resulting centroids.
        let frame_labels = project_frames(&windows, &rows, &outcome.centroids, &active);
        self.check(cancel)?;
        timings.cluster_ms = cluster_started.elapsed().as_millis() as u64;

        // --- [4] smoothing and turns ---------------------------------------
        let smooth_started = Instant::now();
        let filtered = smooth::median_filter_cancellable(
            &frame_labels,
            cfg.effective_median_filter_frames(),
            || self.is_cancelled(cancel),
            poll,
        );
        let turns = smooth::encode_turns_cancellable(
            &filtered,
            &active,
            &frame_labels,
            |i| Time::from_millis((start_ms + fe.frame_time_ms(usable.start + i)).max(0)),
            frame_ms,
            || self.is_cancelled(cancel),
            poll,
        );
        let turns = smooth::absorb_short_turns(turns, cfg.clusters.min_turn_ms);
        let turns = smooth::sanitise(turns);
        let spans: Vec<SpeakerSpan> = turns.iter().map(smooth::Turn::to_span).collect();
        // The two filters above stop early on a cancel and hand back a short
        // `filtered` and a partial `turns`. Without this check a cancelled run
        // would sail on and return a confident transcript of the first half of
        // the meeting, which is the one outcome worse than an error.
        self.check(cancel)?;
        timings.smooth_ms = smooth_started.elapsed().as_millis() as u64;
        progress(0.95);

        let problems = smooth::validate_spans(&spans, outcome.num_clusters);
        if !problems.is_empty() {
            // A bug in this crate, not in the input. Refusing to return the
            // result is louder than returning a transcript whose speaker
            // labels do not resolve.
            return Err(err::internal().with_cause(format!(
                "diarization produced invalid spans: {}",
                problems.join("; ")
            )));
        }

        let result = DiarizationResult {
            spans,
            num_speakers: outcome.num_clusters,
            elapsed_ms: started.elapsed().as_millis() as u64,
        };
        // The engine's own invariant: every registered speaker owns a span.
        let inconsistencies = attribution::assert_consistent(&result);
        if !inconsistencies.is_empty() {
            return Err(err::internal().with_cause(format!(
                "diarization result is inconsistent: {}",
                inconsistencies.join("; ")
            )));
        }

        tracing::info!(
            windows = windows.len(),
            stride,
            speakers = outcome.num_clusters,
            spans = result.spans.len(),
            rule = outcome.rule,
            total_ms = result.elapsed_ms,
            "diarization complete"
        );
        self.finish(started, timings, result, progress)
    }

    /// Record timings, set the result's own elapsed time, and report progress.
    fn finish(
        &self,
        started: Instant,
        mut timings: DiarizationTimings,
        mut result: DiarizationResult,
        progress: &dyn Fn(f64),
    ) -> Result<DiarizationResult> {
        timings.total_ms = started.elapsed().as_millis() as u64;
        result.elapsed_ms = timings.total_ms;
        self.state.lock().timings = Some(timings);
        progress(1.0);
        Ok(result)
    }

    /// Build one embedding per analysis window over the active frames.
    ///
    /// The pitch fallback for windows with no voiced frame needs the whole
    /// recording's pitch statistics, which are not known until every window
    /// has been visited. So each window keeps a one-float summary of how many
    /// of its frames were voiced, and the pitch dimension is patched
    /// afterwards. Storing the per-frame rows for a second pass would cost
    /// 360 MB on a three-hour recording; this costs nothing.
    #[allow(clippy::too_many_arguments)]
    fn embed_windows(
        &self,
        fe: &mut FrontEnd,
        audio: &AudioBuffer,
        active: &[bool],
        usable: &std::ops::Range<usize>,
        frame_ms: i64,
        start_ms: i64,
        layout: Layout,
        cancel: &CancelToken,
        progress: &dyn Fn(f64),
        poll: usize,
    ) -> Result<Vec<Window>> {
        let cfg = &self.config;
        let w = cfg.clusters.window_frames.max(1);
        let hop = cfg.clusters.window_hop_frames.max(1).min(w);
        let min_active =
            (cfg.clusters.min_window_speech_ratio.clamp(0.0, 1.0) * w as f32).ceil() as usize;
        let n = active.len();
        let mut out: Vec<Window> = Vec::new();
        let mut pitch_sum = 0f64;
        let mut pitch_n = 0usize;

        let mut power = vec![0f32; fe.n_freq()];
        let mut log_mel = vec![0f32; cfg.features.n_mel];
        let mut cepstra = vec![0f32; cfg.features.n_cepstra];
        let mut frame = vec![0f32; layout.total];
        let mut window_vec = vec![0f32; layout.total];

        let mut lo = 0usize;
        // Frames walked since the last cancellation check, not windows. The
        // distinction is the whole bug this comment exists to prevent:
        // `cancel_poll_frames` is 50 *frames* -- 0.5 s of audio -- and counting
        // windows instead made the real interval 50 windows, which at a 1.5 s
        // window and a 0.75 s hop is 37.5 s of audio. On a 90-second fixture
        // that is two checks in the entire run, and a cancel waited 5.4 s to be
        // noticed while the stage's own documentation promised half a second.
        let mut since = 0usize;
        while lo < n {
            let hi = (lo + w).min(n);
            let mut acc = WindowAccumulator::new(layout, cfg.features.delta_width);
            let mut act = 0usize;
            for (i, is_active) in active[lo..hi].iter().enumerate() {
                if !*is_active {
                    continue;
                }
                act += 1;
                since += 1;
                if since >= poll {
                    since = 0;
                    // `Err`, never an empty `Vec`: an empty window list is this
                    // function's "the recording is silence" answer, and a
                    // cancelled run must not be reported as one.
                    self.check(cancel)?;
                }
                // `i` is the offset *within this window's slice* of `active`,
                // so the absolute frame index needs `lo` added. Leaving it off
                // makes every window after the first analyse the audio under
                // the *first* window: the whole feature vector becomes a copy
                // of turn zero's, for every turn in the file. That is
                // invisible in a speaker count -- a recording of one speaker
                // still reports one speaker, because a copy of one speaker is
                // one speaker -- and it is why the pitch dimension measured
                // 131 Hz on every window of a two-voice fixture whose second
                // voice renders at 217 Hz, and why the same-speaker distance
                // distribution had a spike at exactly zero and a tail out to
                // 1.9 at the same time.
                let frame_index = usable.start + lo + i;
                let (log2, conf) = fe.frame_pitch(&audio.samples, frame_index);
                crate::features::frame_vector(
                    fe,
                    &audio.samples,
                    frame_index,
                    &mut power,
                    &mut log_mel,
                    &mut cepstra,
                    &mut frame,
                );
                if conf > 0.0 && log2 != 0.0 {
                    pitch_sum += f64::from(log2);
                    pitch_n += 1;
                }
                acc.push(&frame, log2, conf);
            }
            // A window is kept when it holds at least `min_active` voiced-or-
            // not frames. The final window of a file is usually short; it is
            // still evaluated, because a 0.6 s of speech is a real turn and
            // dropping it would delete somebody's last sentence.
            if acc.len() >= min_active.max(1) {
                acc.reduce(0.0, &mut window_vec);
                let f_lo = usable.start + lo;
                let f_hi = usable.start + hi.saturating_sub(1);
                out.push(Window {
                    lo,
                    hi,
                    voiced: acc.voiced(),
                    embedding: WindowEmbedding {
                        start: Time::from_millis((start_ms + fe.frame_time_ms(f_lo)).max(0)),
                        end: Time::from_millis(
                            (start_ms + fe.frame_time_ms(f_hi) + frame_ms).max(0),
                        ),
                        speech_ms: (act as i64 * frame_ms).max(0),
                        vector: window_vec.clone(),
                    },
                });
            }
            progress(0.15 + 0.55 * lo as f64 / n.max(1) as f64);
            lo += hop;
        }

        // The fallback for windows with no voiced frame: the mean of every
        // voiced pitch this recording showed. Without it those windows sit at
        // log2(0) = 0, which z-scoring turns into a strong negative outlier,
        // and every fricative becomes its own speaker.
        let fallback = if pitch_n > 0 {
            (pitch_sum / pitch_n as f64) as f32
        } else {
            0.0
        };
        for win in &mut out {
            if win.voiced == 0 {
                win.embedding.vector[layout.pitch] = fallback;
            }
        }
        Ok(out)
    }

    /// `Err(Cancelled)` if either the token or `cancel()` has fired.
    fn check(&self, cancel: &CancelToken) -> Result<()> {
        if self.cancel_flag.load(Ordering::SeqCst) || cancel.is_cancelled() {
            return Err(err::cancelled());
        }
        Ok(())
    }

    /// Whether cancellation has been requested, from either source.
    #[must_use]
    pub fn is_cancelled(&self, cancel: &CancelToken) -> bool {
        self.cancel_flag.load(Ordering::SeqCst) || cancel.is_cancelled()
    }
}

impl Default for HeuristicDiarizer {
    fn default() -> Self {
        Self::new()
    }
}

/// One analysis window, with the bookkeeping the projection step needs.
#[derive(Debug, Clone)]
struct Window {
    /// First active-frame index this window covers, in the engine's frame
    /// numbering (`0..active.len()`).
    lo: usize,
    /// One past the last.
    hi: usize,
    /// How many of its frames had a confident autocorrelation peak.
    voiced: usize,
    embedding: WindowEmbedding,
}

/// A window as the clustering sees it: the same normalised vector the
/// agglomeration was given, plus the cluster it was assigned.
///
/// Returned by [`HeuristicDiarizer::debug_windows`] so a labelled reference
/// can be compared against the engine's own features. This is what makes the
/// synthetic end-to-end test able to report a measured purity and a measured
/// complete-link separation instead of only a speaker count: the test knows
/// which speaker owns which time, and this says which cluster owns which
/// window.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelledWindow {
    /// Window start, absolute.
    pub start: vaani_core::Time,
    /// Window end, absolute.
    pub end: vaani_core::Time,
    /// Active speech inside the window, in milliseconds.
    pub speech_ms: i64,
    /// The L2-normalised, weighted, z-scored feature vector, carrier included.
    pub vector: Vec<f32>,
    /// The same window's feature vector *before* normalisation, i.e. the
    /// median over its frames. Kept so a diagnostic can see what the
    /// normalisation did, rather than having to infer it from the output.
    pub raw: Vec<f32>,
    /// The cluster the window was assigned. Contiguous from 0.
    pub cluster: usize,
}

impl HeuristicDiarizer {
    /// The window embeddings and their cluster assignments, for a labelled
    /// evaluation.
    ///
    /// This runs the whole feature and clustering pipeline and stops before
    /// smoothing, so the caller sees the raw per-window decisions. It exists
    /// for tests and for the tuning example; the product calls [`DiarizationEngine::diarize`]
    /// and never this, because a per-window view is not something a transcript
    /// can display.
    pub fn debug_windows(
        &self,
        audio: &AudioBuffer,
        cancel: &CancelToken,
    ) -> Result<Vec<LabelledWindow>> {
        let cfg = self.config.clone();
        self.check(cancel)?;
        if audio.channels != 1 || audio.sample_rate != REQUIRED_SAMPLE_RATE {
            return Err(err::internal().with_cause("debug_windows: mono 16 kHz required"));
        }
        let frame_ms = cfg.features.frame_ms();
        let start_ms = audio.start_time.as_millis();
        let mut fe = FrontEnd::new(cfg.features, cfg.activity);
        let layout = Layout::of(&cfg.features);
        let poll = cfg.cancel_poll_frames.max(1);
        let usable = fe.usable_frames(audio.samples.len());
        let n = usable.len();

        let frames = vad::scan(
            &mut fe,
            &audio.samples,
            || self.is_cancelled(cancel),
            |_| {},
            poll,
        );
        let thresholds = vad::derive_thresholds(&frames, &cfg.activity);
        let mask = vad::activity_mask(&frames, &thresholds);
        let timeline = vad::mask_to_timeline(&mask, frame_ms, start_ms, &cfg.activity);

        let mut active = vec![false; n];
        let mut seg = 0usize;
        for (i, f) in usable.clone().enumerate() {
            let at = start_ms + fe.frame_time_ms(f);
            while seg < timeline.segments.len() && timeline.segments[seg].end.as_millis() <= at {
                seg += 1;
            }
            if seg < timeline.segments.len() && timeline.segments[seg].start.as_millis() <= at {
                active[i] = true;
            }
        }
        if !active.iter().any(|a| *a) {
            return Ok(Vec::new());
        }

        let windows = self.embed_windows(
            &mut fe,
            audio,
            &active,
            &usable,
            frame_ms,
            start_ms,
            layout,
            cancel,
            &|_| {},
            poll,
        )?;
        self.check(cancel)?;
        if windows.is_empty() {
            return Ok(Vec::new());
        }

        let mut rows: Vec<Vec<f32>> = windows.iter().map(|w| w.embedding.vector.clone()).collect();
        let raw = rows.clone();
        let speech: Vec<i64> = windows.iter().map(|w| w.embedding.speech_ms).collect();
        crate::features::temporal_smooth(&mut rows, cfg.clusters.feature_smooth_half_windows);
        normalise_in_place(&mut rows, &layout, &cfg.features.weights, &cfg.min_spread);

        let cap = cfg.clusters.max_cluster_windows.max(2);
        let stride = windows.len().div_ceil(cap).max(1);
        let sampled: Vec<Vec<f32>> = rows.iter().step_by(stride).cloned().collect();
        let sampled_speech: Vec<i64> = speech.iter().step_by(stride).copied().collect();
        let outcome = cluster::cluster(
            &sampled,
            &sampled_speech,
            &cfg.clusters,
            cfg.effective_min_cluster_speech_ms(),
            cfg.effective_merge_distance(),
        );
        // The window's **own** cluster, from its own vector and the resulting
        // centroids. It is tempting to read the frame labels here, but those
        // are indexed by frame and assigned by "the latest window containing
        // this frame", so `frame_labels[i]` is the label of frame `i`, not of
        // window `i`. Reading it here attributed every window to whatever
        // cluster happened to own the frame at the same ordinal, which made a
        // perfectly good clustering look like noise.
        let labels: Vec<usize> = rows
            .iter()
            .map(|r| nearest_centroid(r, &outcome.centroids))
            .collect();

        Ok(windows
            .iter()
            .enumerate()
            .map(|(i, w)| LabelledWindow {
                start: w.embedding.start,
                end: w.embedding.end,
                speech_ms: w.embedding.speech_ms,
                vector: rows[i].clone(),
                raw: raw[i].clone(),
                cluster: labels[i],
            })
            .collect())
    }
}

/// Assign every active frame a cluster index.
///
/// For each frame the **latest** window containing it wins. "Latest" rather
/// than "majority" because the windows overlap by 50% and are laid out on a
/// fixed stride, so the latest one is the one whose decision was made with the
/// most evidence about where the recording was heading. Every active frame is
/// covered by at least one window because the stride never exceeds the window
/// length, which the constructor enforces.
fn project_frames(
    windows: &[Window],
    rows: &[Vec<f32>],
    centroids: &[Vec<f32>],
    active: &[bool],
) -> Vec<usize> {
    let n = active.len();
    let mut labels = vec![0usize; n];
    if windows.is_empty() || centroids.is_empty() {
        return labels;
    }
    let mut j = 0usize;
    for (i, slot) in labels.iter_mut().enumerate() {
        if !active[i] {
            continue;
        }
        while j + 1 < windows.len() && windows[j + 1].lo <= i {
            j += 1;
        }
        if i >= windows[j].lo && i < windows[j].hi {
            let row = rows.get(j).unwrap_or_else(|| &windows[j].embedding.vector);
            *slot = nearest_centroid(row, centroids);
        } else if i >= windows[j].hi {
            // Past the last window: the last window is the best information
            // available.
            *slot = nearest_centroid(&rows[windows.len() - 1], centroids);
        }
    }
    labels
}

fn nearest_centroid(row: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best = 0usize;
    let mut best_cost = f64::INFINITY;
    for (c, cen) in centroids.iter().enumerate() {
        let cost = crate::features::cosine_distance(row, cen);
        if cost < best_cost {
            best_cost = cost;
            best = c;
        }
    }
    best
}

impl DiarizationEngine for HeuristicDiarizer {
    fn id(&self) -> &str {
        Self::ID
    }

    fn name(&self) -> &str {
        Self::NAME
    }

    /// Idempotent, and honest about what "loaded" means.
    ///
    /// For the heuristic path there are no weights, so this only probes for an
    /// optional neural embedder and clears the cancel flag. It never creates a
    /// CUDA context, because a user who opens the app to browse history must
    /// not pay for one and `vaani-inference` documents that as a rule.
    fn load(&self, cancel: &CancelToken) -> Result<()> {
        cancel.check()?;
        self.cancel_flag.store(false, Ordering::SeqCst);
        let mut model: Option<String> = None;
        if let Some(dir) = &self.config.artifacts_dir {
            let a = embed::probe(dir);
            if a.is_available() {
                let caps = self.caps.clone();
                match embed::NeuralEmbedder::open(
                    &a.path,
                    embed::EmbedderSpec::default(),
                    self.provider,
                    &caps,
                    None,
                ) {
                    Ok(m) => {
                        model = Some(a.path.display().to_string());
                        *self.embedder.lock() = Some(m);
                    }
                    Err(e) => {
                        // A present-but-unloadable model is reported and then
                        // ignored, so the product still produces speaker labels
                        // from the heuristic. The transcript is the product.
                        tracing::warn!(
                            path = %a.path.display(),
                            error = %e,
                            "a neural embedder is present but would not load; using the spectral heuristic"
                        );
                    }
                }
            } else if let Some(p) = &a.problem {
                tracing::debug!(path = %a.path.display(), "{p}");
            }
        }
        if let Some(cap) = self.state.lock().capabilities.as_mut() {
            cap.neural_embeddings = model.is_some();
            cap.neural_model = model;
        }
        self.loaded.store(true, Ordering::SeqCst);
        cancel.check()
    }

    fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::SeqCst)
    }

    /// Safe when nothing was loaded, and safe to call twice.
    fn unload(&self) {
        if let Some(m) = self.embedder.lock().take() {
            m.close();
        }
        self.loaded.store(false, Ordering::SeqCst);
        if let Some(cap) = self.state.lock().capabilities.as_mut() {
            cap.neural_embeddings = false;
            cap.neural_model = None;
        }
    }

    fn diarize(
        &self,
        audio: &AudioBuffer,
        cancel: &CancelToken,
        progress: &dyn Fn(f64),
    ) -> Result<DiarizationResult> {
        self.diarize_with_activity(audio, cancel, progress)
    }

    fn cancel(&self) {
        self.cancel_flag.store(true, Ordering::SeqCst);
    }
}

/// The engine the pipeline should construct by default.
///
/// A free function rather than a registry, because there is exactly one engine
/// in this build and a registry for one entry is indirection. The architect
/// adds a registry when there is a second.
#[must_use]
pub fn default_engine() -> Arc<HeuristicDiarizer> {
    Arc::new(HeuristicDiarizer::new())
}

/// Box an engine as the trait object the pipeline holds.
#[must_use]
pub fn as_engine(e: HeuristicDiarizer) -> Box<dyn DiarizationEngine> {
    Box::new(e)
}

/// Coverage of a result against a buffer's duration, clamped to `[0, 1]`.
///
/// A wrapper over `DiarizationResult::coverage` so a caller does not have to
/// remember to handle a zero duration.
#[must_use]
pub fn coverage_of(d: &DiarizationResult, audio: &AudioBuffer) -> f64 {
    d.coverage(audio.duration())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone_turns(n_ms: i64, f0: f32) -> Vec<f32> {
        (0..(n_ms * 16))
            .map(|i| {
                let t = i as f32 / 16_000.0;
                (2.0 * std::f32::consts::PI * f0 * t).sin() * 0.4
            })
            .collect()
    }

    /// A speech-shaped signal: two alternating voices with a pitch contour, so
    /// the recording has real energy dynamics and more than one spectral
    /// character. A constant tone is deliberately *not* used for cancellation
    /// tests: it is uniform, so the activity detector correctly reports no
    /// speech and the run finishes before a cancel could land.
    fn speech_shaped(total_ms: i64) -> Vec<f32> {
        let voices = crate::synth::two_voices();
        let utterances =
            crate::synth::alternate(&[0, 1], 0, 2_000, (total_ms / 2_000).max(1) as usize);
        let (buf, _) = crate::synth::render(&utterances, &voices, total_ms, 16_000, 0.002, 7);
        buf
    }

    fn buf(samples: Vec<f32>) -> AudioBuffer {
        AudioBuffer::mono(samples, 16_000, Time::ZERO)
    }

    #[test]
    fn an_empty_buffer_is_an_empty_result_not_an_error() {
        let e = HeuristicDiarizer::new();
        let r = e
            .diarize(&buf(Vec::new()), &CancelToken::new(), &|_| {})
            .unwrap();
        assert_eq!(r.spans.len(), 0);
        assert_eq!(r.num_speakers, 0);
    }

    #[test]
    fn digital_silence_produces_no_speakers_and_no_error() {
        let e = HeuristicDiarizer::new();
        let r = e
            .diarize(&buf(vec![0.0; 160_000]), &CancelToken::new(), &|_| {})
            .unwrap();
        assert_eq!(r.num_speakers, 0, "{:?}", r.spans);
        assert!(r.spans.is_empty());
    }

    #[test]
    fn a_stereo_buffer_is_refused_with_a_reason() {
        let e = HeuristicDiarizer::new();
        let audio = AudioBuffer {
            samples: vec![0.0; 1_000],
            sample_rate: 16_000,
            channels: 2,
            start_time: Time::ZERO,
        };
        let err = e.diarize(&audio, &CancelToken::new(), &|_| {}).unwrap_err();
        assert!(err.debug_chain().contains("mono"));
    }

    #[test]
    fn a_wrong_sample_rate_is_refused_rather_than_resampled() {
        let e = HeuristicDiarizer::new();
        let audio = AudioBuffer::mono(vec![0.0; 48_000], 48_000, Time::ZERO);
        let err = e.diarize(&audio, &CancelToken::new(), &|_| {}).unwrap_err();
        assert!(err.debug_chain().contains("16000 Hz"));
    }

    #[test]
    fn progress_is_monotone_and_ends_at_one() {
        let e = HeuristicDiarizer::new();
        let seen = std::cell::RefCell::new(Vec::<f64>::new());
        let _ = e.diarize(&buf(tone_turns(4_000, 180.0)), &CancelToken::new(), &|f| {
            seen.borrow_mut().push(f)
        });
        let seen = seen.into_inner();
        assert!(!seen.is_empty());
        assert!(seen.windows(2).all(|w| w[1] >= w[0]), "{seen:?}");
        assert_eq!(seen.last().copied(), Some(1.0));
        assert!(seen.iter().all(|f| (0.0..=1.0).contains(f)));
    }

    #[test]
    fn a_cancelled_token_fails_before_any_work() {
        let e = HeuristicDiarizer::new();
        let token = CancelToken::new();
        token.cancel();
        let err = e
            .diarize(&buf(tone_turns(4_000, 180.0)), &token, &|_| {})
            .unwrap_err();
        assert!(err.is_cancelled());
    }

    #[test]
    fn load_is_idempotent_and_unload_is_safe_when_nothing_was_loaded() {
        let e = HeuristicDiarizer::new();
        assert!(!e.is_loaded());
        e.unload();
        e.load(&CancelToken::new()).unwrap();
        e.load(&CancelToken::new()).unwrap();
        assert!(e.is_loaded());
        e.unload();
        e.unload();
        assert!(!e.is_loaded());
    }

    #[test]
    fn capabilities_say_the_truth_about_a_build_with_no_weights() {
        let e = HeuristicDiarizer::new();
        e.load(&CancelToken::new()).unwrap();
        let c = e.capabilities();
        assert!(
            !c.neural_embeddings,
            "no diarization weights exist in this repo"
        );
        assert!(c.neural_model.is_none());
        assert!(c.speaker_count_estimated);
        assert!(!e.has_neural_embedder());
    }

    #[test]
    fn a_configured_artifacts_dir_with_no_weights_stays_on_the_heuristic() {
        let e = HeuristicDiarizer::with_config(
            DiarizationConfig::default().with_artifacts_dir("/definitely/not/here"),
        );
        e.load(&CancelToken::new()).unwrap();
        assert!(!e.has_neural_embedder());
        assert!(!e.capabilities().neural_embeddings);
    }

    #[test]
    fn a_fixed_speaker_count_is_reported_as_not_estimated() {
        let e = HeuristicDiarizer::with_config(DiarizationConfig::default().with_speaker_count(2));
        assert!(!e.capabilities().speaker_count_estimated);
    }

    #[test]
    fn every_window_analyses_its_own_audio_not_the_first_windows() {
        // The regression test for the frame-index bug: `frame_index` omitted
        // `lo`, so every window after the first extracted features from the
        // audio under window zero.
        //
        // Two speakers nearly an octave apart, alternating in long turns, so
        // the pitch dimension has to differ between a window in the first turn
        // and a window in the fourth. With the bug the two are identical to
        // within a hundredth of a log2 unit.
        let voices = crate::synth::two_voices();
        let utterances = crate::synth::alternate(&[0, 1], 0, 4_000, 8);
        let (buf, _) = crate::synth::render(&utterances, &voices, 32_000, 16_000, 0.003, 2024);
        let audio = AudioBuffer::mono(buf, 16_000, Time::ZERO);
        let e = HeuristicDiarizer::new();
        let windows = e
            .debug_windows(&audio, &CancelToken::new())
            .expect("windows");
        assert!(windows.len() >= 4, "only {} windows", windows.len());

        // Group the windows by the median of their raw pitch, which must be
        // bimodal with roughly an octave between the modes.
        let mut pitches: Vec<f32> = windows
            .iter()
            .map(|w| w.raw[crate::features::Layout::of(&e.config.features).pitch])
            .collect();
        pitches.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lo = pitches[pitches.len() / 10];
        let hi = pitches[pitches.len() * 9 / 10];
        let octaves = hi - lo;
        assert!(
            octaves > 0.4,
            "the pitch dimension spans only {octaves:.3} octaves over {} windows: {pitches:?}",
            pitches.len()
        );
    }

    #[test]
    fn an_external_timeline_is_used_in_preference_to_the_internal_detector() {
        let e = HeuristicDiarizer::new();
        let audio = buf(tone_turns(6_000, 200.0));
        let tl = SpeechTimeline {
            segments: vec![vaani_core::pipeline::SpeechSegment {
                start: Time(0),
                end: Time(6_000),
                confidence: None,
            }],
            speech_duration_ms: 6_000,
        };
        let r = e
            .diarize_with_timeline(&audio, Some(&tl), &CancelToken::new(), &|_| {})
            .unwrap();
        assert!(!e.capabilities().internal_activity_detector);
        // Six seconds of one steady tone is one speaker and no more.
        assert_eq!(r.num_speakers, 1, "spans {:?}", r.spans);
    }

    #[test]
    fn an_empty_external_timeline_produces_no_speakers() {
        let e = HeuristicDiarizer::new();
        let audio = buf(tone_turns(6_000, 200.0));
        let r = e
            .diarize_with_timeline(
                &audio,
                Some(&SpeechTimeline::default()),
                &CancelToken::new(),
                &|_| {},
            )
            .unwrap();
        assert_eq!(r.num_speakers, 0);
    }

    #[test]
    fn a_whitespace_only_buffer_short_of_one_window_produces_nothing() {
        // 1 s of audio is shorter than the 1.5 s analysis window, so no window
        // can be emitted. That is a real and common case: a 5 ms click in a
        // long file, or the tail of a chunk.
        let e = HeuristicDiarizer::new();
        let r = e
            .diarize(&buf(tone_turns(1_000, 200.0)), &CancelToken::new(), &|_| {})
            .unwrap();
        assert_eq!(r.num_speakers, 0);
    }

    #[test]
    fn timings_are_recorded_and_bounded_by_the_call() {
        let e = HeuristicDiarizer::new();
        let _ = e.diarize(&buf(tone_turns(8_000, 200.0)), &CancelToken::new(), &|_| {});
        let t = e.last_timings().expect("timings must be recorded");
        // 8 s at 10 ms is 800 frame centres, of which frame 0 (its window
        // lies before the media) and frame 800 (its centre is past the last
        // sample) are excluded.
        assert_eq!(t.frames, 799, "8 s of 10 ms frames");
        assert!(t.total_ms > 0);
        assert!(t.rtf(8_000) > 0.0);
        assert_eq!(t.rtf(0), 0.0, "a zero duration must not divide by zero");
    }

    #[test]
    fn the_engine_is_nameable_through_the_trait() {
        let e: Box<dyn DiarizationEngine> = as_engine(HeuristicDiarizer::new());
        assert_eq!(e.id(), HeuristicDiarizer::ID);
        assert!(!e.name().is_empty());
        assert!(!default_engine().is_loaded());
    }

    #[test]
    fn cancel_fires_even_without_a_cancelled_token() {
        // The audio is rendered *before* the thread starts. Building 10
        // minutes of it takes a couple of seconds, and a cancel issued while
        // the worker is still rendering is cleared by `diarize` at the start
        // of the call -- which is correct behaviour and makes the test a race
        // unless the fixture is prepared first.
        let audio = buf(speech_shaped(600_000));
        let e = std::sync::Arc::new(HeuristicDiarizer::new());
        let token = CancelToken::new();
        let e3 = e.clone();
        let handle = std::thread::spawn(move || e3.diarize(&audio, &token, &|_| {}));
        // Cancel repeatedly rather than once: the flag is cleared at the start
        // of every call, so a single cancel that lands before the worker is
        // actually inside `diarize` is discarded. Looping makes the assertion
        // independent of thread scheduling, which is the only way it is stable
        // under load.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
        while std::time::Instant::now() < deadline {
            e.cancel();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let joined = std::time::Instant::now();
        let r = handle.join().unwrap();
        assert!(
            r.unwrap_err().is_cancelled(),
            "a 10-minute job finished before the cancel landed"
        );
        assert!(
            joined.elapsed() < std::time::Duration::from_secs(5),
            "cancellation took {:?} to take effect",
            joined.elapsed()
        );
    }

    #[test]
    fn a_cancel_token_alone_also_stops_a_long_job() {
        let audio = buf(speech_shaped(600_000));
        let e = HeuristicDiarizer::new();
        let token = CancelToken::new();
        let t2 = token.clone();
        let handle = std::thread::spawn(move || e.diarize(&audio, &t2, &|_| {}));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
        while std::time::Instant::now() < deadline {
            token.cancel();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(handle.join().unwrap().unwrap_err().is_cancelled());
    }

    #[test]
    fn a_second_run_after_a_cancel_starts_clean() {
        // The cancel flag is per-call, not sticky. An engine that latched it
        // would fail every subsequent job in the process.
        let e = HeuristicDiarizer::new();
        e.cancel();
        assert!(e.is_cancelled(&CancelToken::new()), "the flag was set");
        let r = e
            .diarize(&buf(speech_shaped(8_000)), &CancelToken::new(), &|_| {})
            .expect("a cancelled engine must recover on the next call");
        assert!(r.num_speakers >= 1);
    }

    #[test]
    fn coverage_of_a_silent_recording_is_zero_and_does_not_divide_by_zero() {
        let e = HeuristicDiarizer::new();
        let audio = buf(vec![0.0; 16_000]);
        let r = e.diarize(&audio, &CancelToken::new(), &|_| {}).unwrap();
        assert_eq!(coverage_of(&r, &audio), 0.0);
    }
}

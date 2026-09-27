//! The pipeline contract: the interfaces every stage is written against.
//!
//! # Why this lives in `vaani-core`
//!
//! Eight subsystems are being implemented in parallel by different agents. The
//! only thing keeping that from being a random walk is that they all compile
//! against *these exact signatures*, in a crate with no dependencies. If the
//! media stage, the VAD stage and the UI each invented their own idea of "a
//! buffer of audio", integration would become a negotiation instead of a
//! merge.
//!
//! This module is owned by the architect and is **not** edited by feature
//! agents. A stage that needs a different contract raises it; it does not
//! define it locally.
//!
//! # The shape of a pipeline
//!
//! ```text
//!   Path ─▶ media ─▶ AudioBuffer ─▶ vad ─▶ SpeechTimeline ─▶ chunk
//!                                                              │
//!                     ┌────────────────────────────────────────┘
//!                     ▼
//!                 asr ─▶ Transcript (chunk-local times)
//!                     │
//!        diarization ─▶ SpeakerSpan[]  (independent of asr)
//!                     │
//!                 postprocess ─▶ Transcript (cleaned, in place)
//!                     │
//!!                    structuring ─▶ Transcript (outlined)
//!                     │
//!                    storage ─▶ persisted job
//!                     │
//!          export ─▶ Vec<PathBuf>
//! ```
//!
//! Two properties are load-bearing and are enforced by types rather than by
//! convention:
//!
//! * **Times are absolute once they leave a stage.** `AudioBuffer::start_time`
//!   and `SpeechTimeline`'s segments are all in source-media coordinates. The
//!   ASR stage is the single exception and says so: it works in chunk-local
//!   coordinates because the decoder's frame indices are chunk-relative, and
//!   `vaani_core::Transcript::push_chunk` does the rebasing.
//! * **Diarization never depends on ASR output.** It takes an `AudioBuffer` and
//!   produces spans. Speaker attribution onto transcript segments is a later,
//!   separate step, so a diarization model can be swapped or turned off without
//!   touching a line of ASR code.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::Result;
use crate::job::{CancelToken, JobPhase, Progress};
use crate::media::AudioBuffer;

/// A contiguous stretch of detected speech, in source-media coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct SpeechSegment {
    pub start: crate::Time,
    pub end: crate::Time,
    /// Mean speech probability over the segment, in [0,1]. `None` for
    /// energy-based detectors that do not produce one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

/// The result of voice activity detection over a whole buffer.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SpeechTimeline {
    /// Sorted, non-overlapping, merged to the detector's minimum gap.
    pub segments: Vec<SpeechSegment>,
    /// Total speech time, for reporting "this meeting is 18% speech".
    #[serde(default)]
    pub speech_duration_ms: i64,
}

impl SpeechTimeline {
    /// Fraction of the buffer that is speech, in [0,1]. The single most
    /// useful number for deciding whether VAD is earning its keep.
    #[must_use]
    pub fn speech_ratio(&self, total: crate::Time) -> f64 {
        if total.as_millis() <= 0 {
            return 0.0;
        }
        (self.speech_duration_ms as f64 / total.as_millis() as f64).clamp(0.0, 1.0)
    }

    /// Segments overlapping `[start, end)`, for chunk extraction.
    pub fn overlapping(
        &self,
        start: crate::Time,
        end: crate::Time,
    ) -> impl Iterator<Item = &SpeechSegment> {
        self.segments
            .iter()
            .filter(move |s| s.start < end && s.end > start)
    }
}

/// Voice activity detection.
pub trait VadEngine: Send + Sync + std::fmt::Debug {
    /// Stable id, used in the benchmark report and the settings panel.
    fn id(&self) -> &str;

    /// Human name for the UI.
    fn name(&self) -> &str;

    /// The sample rate this detector requires. The pipeline resamples before
    /// calling; a detector must never be handed audio it cannot use.
    fn sample_rate(&self) -> u32;

    /// Detect speech. Must poll `cancel` at least once per second of audio.
    fn detect(&self, audio: &AudioBuffer, cancel: &CancelToken) -> Result<SpeechTimeline>;

    /// Cancel an in-flight detection, for a cancel button that fires while no
    /// job token is reachable from the UI.
    fn cancel(&self);
}

/// One speaker's occupancy of the timeline.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct SpeakerSpan {
    /// Cluster index. 0-based, contiguous from 0. The UI may rename it; the
    /// index itself never changes.
    pub speaker: usize,
    pub start: crate::Time,
    pub end: crate::Time,
    /// Clustering confidence for this span, in [0,1]. `None` when the engine
    /// cannot produce one, which is not an error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
}

impl SpeakerSpan {
    /// Whether `t` falls inside the span, half-open.
    #[must_use]
    pub fn contains(&self, t: crate::Time) -> bool {
        t >= self.start && t < self.end
    }
}

/// The output of diarization, independent of any transcript.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct DiarizationResult {
    /// Sorted by start time. May overlap when a clustering boundary is
    /// uncertain; the attribution step resolves that.
    pub spans: Vec<SpeakerSpan>,
    /// Number of distinct speakers discovered.
    pub num_speakers: usize,
    /// Total time spent computing, for the benchmark report.
    pub elapsed_ms: u64,
}

impl DiarizationResult {
    /// Fraction of `total` covered by exactly one speaker, in [0,1].
    #[must_use]
    pub fn coverage(&self, total: crate::Time) -> f64 {
        if total.as_millis() <= 0 {
            return 0.0;
        }
        let covered: i64 = self
            .spans
            .iter()
            .map(|s| (s.end.as_millis() - s.start.as_millis()).max(0))
            .sum();
        (covered as f64 / total.as_millis() as f64).clamp(0.0, 1.0)
    }

    /// The speaker at time `t`, preferring the shortest containing span so an
    /// ambiguous overlap resolves to the more specific claim.
    #[must_use]
    pub fn speaker_at(&self, t: crate::Time) -> Option<usize> {
        self.spans
            .iter()
            .filter(|s| s.contains(t))
            .min_by_key(|s| s.end.as_millis() - s.start.as_millis())
            .map(|s| s.speaker)
    }

    /// Speech time per speaker, in cluster order.
    #[must_use]
    pub fn speaker_durations(&self) -> Vec<i64> {
        let mut out = vec![0i64; self.num_speakers];
        for s in &self.spans {
            if let Some(slot) = out.get_mut(s.speaker) {
                *slot += (s.end.as_millis() - s.start.as_millis()).max(0);
            }
        }
        out
    }
}

/// Speaker diarization.
pub trait DiarizationEngine: Send + Sync + std::fmt::Debug {
    /// Stable id for the benchmark report.
    fn id(&self) -> &str;

    /// Human name for the UI.
    fn name(&self) -> &str;

    /// Load models. Idempotent, and must be callable off the UI thread.
    fn load(&self, cancel: &CancelToken) -> Result<()>;

    /// Whether the models are currently loaded.
    fn is_loaded(&self) -> bool;

    /// Release models and any accelerator memory. Must be safe when only
    /// partially loaded.
    fn unload(&self);

    /// Diarize. `progress` is called with a fraction in [0,1] and must be
    /// cheap enough to call per chunk.
    fn diarize(
        &self,
        audio: &AudioBuffer,
        cancel: &CancelToken,
        progress: &dyn Fn(f64),
    ) -> Result<DiarizationResult>;

    /// Cancel an in-flight diarization.
    fn cancel(&self);
}

/// Progress sink handed to a stage. Object-safe so a stage can hold it without
/// generics, which keeps the stage traits usable as trait objects.
pub type ProgressSink = dyn Fn(Progress) + Send + Sync;

/// Everything a stage is given, and everything it is allowed to put back.
///
/// The split matters: `&self` inputs cannot be mutated by a stage, so two
/// stages cannot fight over the same buffer, and `artifacts` is a plain struct
/// so the orchestration order is visible in the type rather than implied by
/// field access order.
#[allow(missing_docs)] // the first four fields are documented; the rest are self-describing handles
pub struct StageContext<'a> {
    /// The source file, for stages that need to re-open it.
    pub input: &'a Path,
    /// Per-job output directory. Stages that write intermediate files use this
    /// and nothing else; never the source file's directory.
    pub work_dir: &'a Path,
    pub cancel: &'a CancelToken,
    pub progress: &'a ProgressSink,
    pub artifacts: &'a mut StageArtifacts,
}

impl std::fmt::Debug for StageContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StageContext")
            .field("input", &self.input)
            .field("work_dir", &self.work_dir)
            .field("cancelled", &self.cancel.is_cancelled())
            .field("has_audio", &self.artifacts.audio.is_some())
            .field("has_speech", &self.artifacts.speech.is_some())
            .field("has_transcript", &self.artifacts.transcript.is_some())
            .field("has_speakers", &self.artifacts.speakers.is_some())
            .finish_non_exhaustive()
    }
}

impl StageContext<'_> {
    /// Report progress for a phase, if the job has not been cancelled.
    pub fn report(&self, phase: JobPhase, fraction: Option<f64>, detail: Option<&str>) {
        if self.cancel.is_cancelled() {
            return;
        }
        let mut p = match fraction {
            Some(f) => Progress::media(phase, f.clamp(0.0, 1.0), 1.0),
            None => Progress::indeterminate(phase),
        };
        if let Some(d) = detail {
            p = p.with_detail(d);
        }
        (self.progress)(p);
    }

    /// `Err(Cancelled)` if the job has been cancelled. Call at every chunk
    /// boundary; it is a relaxed atomic load.
    pub fn check(&self) -> Result<()> {
        self.cancel.check()
    }
}

/// The blackboard stages hand results through.
///
/// One struct, one owner (`vaani-app`), so there is exactly one place where the
/// pipeline's data flow is written down.
#[derive(Debug, Default)]
#[allow(missing_docs)] // field names are the documentation
pub struct StageArtifacts {
    /// Set by the media stage.
    pub audio: Option<AudioBuffer>,
    /// Set by the VAD stage.
    pub speech: Option<SpeechTimeline>,
    /// Accumulated by the ASR stage across chunks.
    pub transcript: Option<crate::Transcript>,
    /// Set by the diarization stage.
    pub speakers: Option<DiarizationResult>,
    /// Appended by the export stage.
    pub exported: Vec<PathBuf>,
}

impl StageArtifacts {
    /// The decoded audio, or a user-facing error naming what is missing.
    pub fn audio(&self) -> Result<&AudioBuffer> {
        self.audio
            .as_ref()
            .ok_or_else(|| crate::err::internal().with_cause("the media stage produced no audio"))
    }

    /// The accumulated transcript, or a user-facing error naming what is missing.
    pub fn transcript(&self) -> Result<&crate::Transcript> {
        self.transcript.as_ref().ok_or_else(|| {
            crate::err::internal().with_cause("the ASR stage produced no transcript")
        })
    }
}

/// One step in the pipeline.
pub trait PipelineStage: Send + Sync {
    /// Stable name, used in logs and the benchmark report.
    fn name(&self) -> &'static str;

    /// The lifecycle phase this stage occupies. A stage may span several
    /// phases; this is the one the UI shows while it runs.
    fn phase(&self) -> JobPhase;

    /// Whether the stage can run. A stage that needs a model reports `false`
    /// with a reason, so the orchestration layer can skip it and say why
    /// rather than failing the job.
    fn is_available(&self) -> std::result::Result<(), String> {
        Ok(())
    }

    /// Run. Must poll `ctx.check()` at least once per second of work.
    fn run(&self, ctx: &mut StageContext<'_>) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Time;

    fn t(ms: i64) -> Time {
        Time(ms)
    }

    #[test]
    fn speech_ratio_is_guarded_against_zero_duration() {
        let tl = SpeechTimeline {
            segments: vec![],
            speech_duration_ms: 0,
        };
        assert_eq!(tl.speech_ratio(t(0)), 0.0);
        assert_eq!(tl.speech_ratio(t(1000)), 0.0);
        let tl = SpeechTimeline {
            segments: vec![],
            speech_duration_ms: 250,
        };
        assert!((tl.speech_ratio(t(1000)) - 0.25).abs() < 1e-9);
    }

    #[test]
    fn overlapping_returns_only_the_spans_that_touch_the_range() {
        let tl = SpeechTimeline {
            segments: vec![
                SpeechSegment {
                    start: t(0),
                    end: t(1000),
                    confidence: None,
                },
                SpeechSegment {
                    start: t(2000),
                    end: t(3000),
                    confidence: None,
                },
                SpeechSegment {
                    start: t(5000),
                    end: t(6000),
                    confidence: None,
                },
            ],
            speech_duration_ms: 3000,
        };
        let hits: Vec<_> = tl.overlapping(t(900), t(2100)).collect();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].start, t(0));
        assert_eq!(hits[1].start, t(2000));
    }

    fn span(speaker: usize, start: i64, end: i64) -> SpeakerSpan {
        SpeakerSpan {
            speaker,
            start: t(start),
            end: t(end),
            score: None,
        }
    }

    #[test]
    fn speaker_at_prefers_the_shortest_containing_span() {
        // An ambiguous overlap must resolve to the more specific claim, not to
        // whichever happened to be listed first.
        let r = DiarizationResult {
            spans: vec![span(0, 0, 5000), span(1, 1000, 2000)],
            num_speakers: 2,
            elapsed_ms: 0,
        };
        assert_eq!(r.speaker_at(t(1500)), Some(1), "the tight span should win");
        assert_eq!(r.speaker_at(t(3000)), Some(0));
        assert_eq!(r.speaker_at(t(9000)), None, "past the end of everything");
    }

    #[test]
    fn span_contains_is_half_open() {
        // Half-open matters: adjacent segments from one speaker must not both
        // claim the boundary instant, or the attribution double-counts it.
        let s = span(0, 1000, 2000);
        assert!(!s.contains(t(999)));
        assert!(s.contains(t(1000)));
        assert!(s.contains(t(1999)));
        assert!(!s.contains(t(2000)));
    }

    #[test]
    fn coverage_and_durations_use_span_lengths() {
        let r = DiarizationResult {
            spans: vec![span(0, 0, 3000), span(1, 4000, 6000)],
            num_speakers: 2,
            elapsed_ms: 0,
        };
        assert!((r.coverage(t(6000)) - 0.8333).abs() < 1e-3);
        assert_eq!(r.speaker_durations(), vec![3000, 2000]);
    }

    #[test]
    fn coverage_is_guarded_against_zero_duration() {
        let r = DiarizationResult::default();
        assert_eq!(r.coverage(t(0)), 0.0);
    }

    #[test]
    fn a_span_naming_an_unregistered_speaker_is_ignored_not_panicking() {
        // A clustering engine that reports a higher index than `num_speakers`
        // would otherwise panic on the tight loop below.
        let r = DiarizationResult {
            spans: vec![span(7, 0, 1000)],
            num_speakers: 1,
            elapsed_ms: 0,
        };
        assert_eq!(r.speaker_durations(), vec![0]);
    }

    #[test]
    fn artifacts_report_a_useful_error_when_a_stage_has_not_run() {
        let a = StageArtifacts::default();
        let e = a.audio().unwrap_err();
        assert!(e.debug_chain().contains("media stage"));
        assert!(a.transcript().is_err());
    }
}

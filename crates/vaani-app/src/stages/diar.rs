//! The diarization stage: independent audio path plus attribution.
//!
//! Diarizes the decoded [`AudioBuffer`](vaani_core::media::AudioBuffer)
//! directly — never ASR output — reusing the VAD timeline when one exists,
//! then attributes the resulting spans onto the transcript with
//! [`attribute_speakers`](vaani_diarization::attribution::attribute_speakers).
//! Attribution is part of this stage (not of diarization itself) so the
//! diarizer stays transcript-free per `INTEGRATION_CONTRACT.md` §4.
//!
//! When no engine is available (diarization disabled, or no engine built for
//! this run) the stage reports itself unavailable with a reason and the
//! orchestrator skips it: a transcript without speakers is a complete answer.

use std::sync::Arc;

use vaani_core::media::AudioBuffer;
use vaani_core::pipeline::{
    DiarizationResult, PipelineStage, SpeechTimeline, StageContext,
};
use vaani_core::{err, CancelToken, JobPhase, Result};
use vaani_diarization::HeuristicDiarizer;

/// How the stage diarizes without depending on a concrete engine.
///
/// The production implementation delegates to
/// [`HeuristicDiarizer::diarize_with_timeline`]; tests inject synthetic spans.
pub trait Diarize: Send + Sync {
    /// Stable engine id recorded in `EngineStamp::diarization_model`.
    fn engine_id(&self) -> &'static str;

    /// Diarize `audio`, reusing `speech` when present. Takes the audio
    /// buffer, never transcript segments.
    fn diarize(
        &self,
        audio: &AudioBuffer,
        speech: Option<&SpeechTimeline>,
        cancel: &CancelToken,
        progress: &dyn Fn(f64),
    ) -> Result<DiarizationResult>;
}

impl Diarize for HeuristicDiarizer {
    fn engine_id(&self) -> &'static str {
        Self::ID
    }

    fn diarize(
        &self,
        audio: &AudioBuffer,
        speech: Option<&SpeechTimeline>,
        cancel: &CancelToken,
        progress: &dyn Fn(f64),
    ) -> Result<DiarizationResult> {
        self.diarize_with_timeline(audio, speech, cancel, progress)
    }
}

/// Speaker separation plus speaker attribution onto the transcript.
pub struct DiarStage {
    engine: Option<Arc<dyn Diarize>>,
    unavailable_reason: Option<String>,
}

impl DiarStage {
    /// A stage that will diarize with `engine`.
    pub fn new(engine: Arc<dyn Diarize>) -> Self {
        Self {
            engine: Some(engine),
            unavailable_reason: None,
        }
    }

    /// A stage that reports itself unavailable with `reason`, so the
    /// orchestrator skips diarization instead of failing the job.
    pub fn without_engine(reason: impl Into<String>) -> Self {
        Self {
            engine: None,
            unavailable_reason: Some(reason.into()),
        }
    }
}

impl std::fmt::Debug for DiarStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiarStage")
            .field("available", &self.engine.is_some())
            .field("unavailable_reason", &self.unavailable_reason)
            .finish()
    }
}

impl PipelineStage for DiarStage {
    fn name(&self) -> &'static str {
        "diarization"
    }

    fn phase(&self) -> JobPhase {
        JobPhase::Diarization
    }

    fn is_available(&self) -> std::result::Result<(), String> {
        match (&self.engine, &self.unavailable_reason) {
            (Some(_), _) => Ok(()),
            (None, Some(reason)) => Err(reason.clone()),
            (None, None) => Err("speaker separation is unavailable".to_owned()),
        }
    }

    fn run(&self, ctx: &mut StageContext<'_>) -> Result<()> {
        ctx.check()?;
        let engine = self.engine.as_ref().ok_or_else(|| {
            err::internal().with_cause("the diarization stage ran while unavailable")
        })?;
        let audio = ctx.artifacts.audio()?;
        let speech = ctx.artifacts.speech.clone();
        let result = engine.diarize(audio, speech.as_ref(), ctx.cancel, &|fraction| {
            ctx.report(
                JobPhase::Diarization,
                Some(fraction.clamp(0.0, 1.0)),
                Some("separating speakers"),
            );
        })?;
        let transcript =
            ctx.artifacts.transcript.as_mut().ok_or_else(|| {
                err::internal().with_cause("the ASR stage produced no transcript")
            })?;
        vaani_diarization::attribute_speakers(transcript, &result);
        transcript.engine.diarization_model = Some(engine.engine_id().to_owned());
        ctx.artifacts.speakers = Some(result);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fakes::FakeDiar;
    use std::path::Path;
    use vaani_core::media::AudioBuffer;
    use vaani_core::pipeline::StageArtifacts;
    use vaani_core::{CancelToken, EngineStamp, Time, Transcript};

    fn transcript_with_segments() -> Transcript {
        let mut transcript = Transcript::new(EngineStamp::asr("fake-asr", "test.0"));
        transcript.push_chunk(
            Time::ZERO,
            vec![
                crate::fakes::segment("hello world", 0, 2000),
                crate::fakes::segment("goodbye world", 2000, 4000),
            ],
        );
        transcript
    }

    #[test]
    fn unavailable_stage_names_its_reason() {
        let stage = DiarStage::without_engine("speaker separation is turned off in settings");
        assert_eq!(stage.name(), "diarization");
        assert_eq!(stage.phase(), JobPhase::Diarization);
        assert_eq!(
            stage.is_available(),
            Err("speaker separation is turned off in settings".to_owned())
        );
    }

    #[test]
    fn attribution_assigns_speakers_from_synthetic_spans() {
        let stage = DiarStage::new(Arc::new(FakeDiar::halves()));
        let mut artifacts = StageArtifacts {
            audio: Some(AudioBuffer::mono(vec![0.5; 64_000], 16_000, Time::ZERO)),
            transcript: Some(transcript_with_segments()),
            ..StageArtifacts::default()
        };
        let cancel = CancelToken::new();
        let dir = std::env::temp_dir();
        let mut ctx = StageContext {
            input: Path::new("input.wav"),
            work_dir: &dir,
            cancel: &cancel,
            progress: &|_| {},
            artifacts: &mut artifacts,
        };
        stage.run(&mut ctx).unwrap();
        let transcript = ctx.artifacts.transcript.as_ref().unwrap();
        // The fake splits 4 s of audio into halves; each 2 s segment falls
        // wholly inside one half.
        assert_eq!(transcript.segments[0].speaker, Some(0));
        assert_eq!(transcript.segments[1].speaker, Some(1));
        assert_eq!(transcript.speakers.len(), 2);
        assert_eq!(
            transcript.engine.diarization_model.as_deref(),
            Some("fake-diar")
        );
        assert!(ctx.artifacts.speakers.is_some());
    }
}

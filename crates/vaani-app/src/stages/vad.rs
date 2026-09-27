//! The VAD stage: detect speech in the decoded buffer.
//!
//! Runs a [`VadEngine`](vaani_core::pipeline::VadEngine) over the mono 16 kHz
//! audio the media stage produced and stores the absolute-time
//! [`SpeechTimeline`](vaani_core::pipeline::SpeechTimeline). The stage never
//! resamples: decode already produced the sample rate the engine requires.

use std::sync::Arc;

use vaani_core::pipeline::{PipelineStage, StageContext, VadEngine};
use vaani_core::{err, JobPhase, Result};

/// Voice activity detection over the whole buffer.
#[derive(Debug)]
pub struct VadStage {
    engine: Arc<dyn VadEngine>,
}

impl VadStage {
    /// A stage running `engine`.
    pub fn new(engine: Arc<dyn VadEngine>) -> Self {
        Self { engine }
    }

    /// The engine in use.
    #[must_use]
    pub fn engine(&self) -> &Arc<dyn VadEngine> {
        &self.engine
    }
}

impl PipelineStage for VadStage {
    fn name(&self) -> &'static str {
        "vad"
    }

    fn phase(&self) -> JobPhase {
        JobPhase::Vad
    }

    fn run(&self, ctx: &mut StageContext<'_>) -> Result<()> {
        ctx.check()?;
        let audio = ctx.artifacts.audio()?;
        if audio.sample_rate != self.engine.sample_rate() {
            return Err(err::internal().with_cause(format!(
                "VAD engine {} needs {} Hz, decoded audio is {} Hz; resampling is the media stage's job",
                self.engine.id(),
                self.engine.sample_rate(),
                audio.sample_rate
            )));
        }
        let timeline = self.engine.detect(audio, ctx.cancel)?;
        let seconds = timeline.speech_duration_ms as f64 / 1000.0;
        ctx.report(
            JobPhase::Vad,
            Some(1.0),
            Some(&format!(
                "{} speech regions, {seconds:.1}s of speech",
                timeline.segments.len()
            )),
        );
        ctx.artifacts.speech = Some(timeline);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fakes::FakeVad;
    use std::path::Path;
    use vaani_core::media::AudioBuffer;
    use vaani_core::pipeline::StageArtifacts;
    use vaani_core::{CancelToken, Time};

    #[test]
    fn detect_stores_an_absolute_timeline() {
        let stage = VadStage::new(Arc::new(FakeVad::speech()));
        let mut artifacts = StageArtifacts {
            audio: Some(AudioBuffer::mono(vec![0.5; 32_000], 16_000, Time::ZERO)),
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
        let timeline = ctx.artifacts.speech.as_ref().unwrap();
        assert_eq!(timeline.segments.len(), 1);
        assert_eq!(timeline.segments[0].start, Time::ZERO);
        assert_eq!(timeline.segments[0].end, Time(2000));
    }

    #[test]
    fn missing_audio_is_an_internal_error() {
        let stage = VadStage::new(Arc::new(FakeVad::speech()));
        let mut artifacts = StageArtifacts::default();
        let cancel = CancelToken::new();
        let dir = std::env::temp_dir();
        let mut ctx = StageContext {
            input: Path::new("input.wav"),
            work_dir: &dir,
            cancel: &cancel,
            progress: &|_| {},
            artifacts: &mut artifacts,
        };
        let error = stage.run(&mut ctx).unwrap_err();
        assert!(error.debug_chain().contains("media stage"));
    }
}

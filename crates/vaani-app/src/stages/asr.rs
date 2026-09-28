//! The ASR stage: plan chunks, transcribe each one, accumulate the transcript.
//!
//! Plans [`Chunk`](vaani_vad::chunking::Chunk)s from the VAD timeline with
//! [`Chunker::plan`](vaani_vad::chunking::Chunker::plan), slices samples with
//! [`AudioBuffer::slice_mono`](vaani_core::media::AudioBuffer::slice_mono),
//! transcribes each chunk with an
//! [`AsrEngine`](vaani_asr::engine::AsrEngine), and accumulates with
//! [`Transcript::push_chunk`](vaani_core::Transcript::push_chunk) — the single
//! rebase point where chunk-local times become media times. Nothing here adds
//! a chunk offset to a [`Time`](vaani_core::Time).
//!
//! Silence (no chunks) still produces a transcript: an empty one with the
//! media duration, so the storage stage has something to persist.

use vaani_asr::engine::{AsrEngine, AudioChunk};
use vaani_core::pipeline::{PipelineStage, StageContext};
use vaani_core::{err, EngineStamp, JobPhase, Language, Result, Transcript};
use vaani_media::TARGET_SAMPLE_RATE;
use vaani_vad::{ChunkPolicy, Chunker};

/// Chunked speech recognition into one accumulated [`Transcript`].
#[derive(Debug)]
pub struct AsrStage {
    engine: Box<dyn AsrEngine>,
    language: Language,
    policy: ChunkPolicy,
}

impl AsrStage {
    /// A stage transcribing with `engine`, hinting `language`, chunked by
    /// `policy`.
    pub fn new(engine: Box<dyn AsrEngine>, language: Language, policy: ChunkPolicy) -> Self {
        Self {
            engine,
            language,
            policy,
        }
    }
}

impl PipelineStage for AsrStage {
    fn name(&self) -> &'static str {
        "asr"
    }

    fn phase(&self) -> JobPhase {
        JobPhase::Asr
    }

    fn run(&self, ctx: &mut StageContext<'_>) -> Result<()> {
        ctx.check()?;
        let speech = ctx.artifacts.speech.clone().ok_or_else(|| {
            err::internal().with_cause("the VAD stage produced no speech timeline")
        })?;
        let total = ctx.artifacts.audio()?.duration();
        let chunks = Chunker::new(self.policy).plan(&speech, total);

        let mut transcript = Transcript::new(EngineStamp::asr(
            self.engine.model_name().to_owned(),
            self.engine.model_version().to_owned(),
        ));
        transcript.duration = Some(total);

        let total_chunks = chunks.len();
        for (index, chunk) in chunks.iter().enumerate() {
            ctx.check()?;
            let (samples, sample_rate) = {
                let audio = ctx.artifacts.audio()?;
                if audio.sample_rate != TARGET_SAMPLE_RATE {
                    return Err(err::internal().with_cause(format!(
                        "ASR needs {TARGET_SAMPLE_RATE} Hz, decoded audio is {} Hz; resampling is the media stage's job",
                        audio.sample_rate
                    )));
                }
                (audio.slice_mono(chunk.start, chunk.end), audio.sample_rate)
            };
            if samples.is_empty() {
                continue;
            }
            let input =
                AudioChunk::new(samples, sample_rate, chunk.start).with_language(self.language);
            // The result's times are chunk-local by contract; `push_chunk_within`
            // does the one rebasing, clamped to the media so the last frame
            // cannot land past the end of the recording. The decoded buffer
            // starts at zero, so `total` is the media end.
            let result = self.engine.transcribe(&input, ctx.cancel)?;
            transcript.push_chunk_within(chunk.start, result.segments, total);
            ctx.report(
                JobPhase::Asr,
                Some((index + 1) as f64 / total_chunks as f64),
                Some(&format!("chunk {}/{}", index + 1, total_chunks)),
            );
        }
        ctx.artifacts.transcript = Some(transcript);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fakes::FakeAsr;
    use std::path::Path;
    use vaani_core::media::AudioBuffer;
    use vaani_core::pipeline::{SpeechSegment, SpeechTimeline, StageArtifacts};
    use vaani_core::{CancelToken, Time};

    fn two_seconds_of_speech() -> (AudioBuffer, SpeechTimeline) {
        let audio = AudioBuffer::mono(vec![0.5; 32_000], 16_000, Time::ZERO);
        let timeline = SpeechTimeline {
            segments: vec![SpeechSegment {
                start: Time::ZERO,
                end: Time(2000),
                confidence: None,
            }],
            speech_duration_ms: 2000,
        };
        (audio, timeline)
    }

    fn run_stage(engine: FakeAsr, audio: AudioBuffer, speech: SpeechTimeline) -> Transcript {
        let stage = AsrStage::new(Box::new(engine), Language::Auto, ChunkPolicy::default());
        let mut artifacts = StageArtifacts {
            audio: Some(audio),
            speech: Some(speech),
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
        ctx.artifacts.transcript.clone().unwrap()
    }

    #[test]
    fn transcribe_accumulates_one_transcript_with_media_times() {
        let (audio, speech) = two_seconds_of_speech();
        let transcript = run_stage(FakeAsr::new(), audio, speech);
        assert!(!transcript.segments.is_empty());
        assert_eq!(transcript.engine.asr_model, "fake-asr");
        // Times left the fake chunk-local and arrived rebased onto the media.
        for segment in &transcript.segments {
            assert!(segment.start >= Time::ZERO);
            assert!(segment.end >= segment.start);
            for word in &segment.words {
                assert!(word.start >= Time::ZERO);
                assert!(word.end >= word.start);
            }
        }
    }

    #[test]
    fn silence_still_produces_an_empty_transcript() {
        let audio = AudioBuffer::mono(vec![0.0; 16_000], 16_000, Time::ZERO);
        let speech = SpeechTimeline::default();
        let transcript = run_stage(FakeAsr::new(), audio, speech);
        assert!(transcript.segments.is_empty());
        assert_eq!(transcript.duration, Some(Time(1000)));
    }

    #[test]
    fn missing_timeline_is_an_internal_error() {
        let stage = AsrStage::new(
            Box::new(FakeAsr::new()),
            Language::Auto,
            ChunkPolicy::default(),
        );
        let mut artifacts = StageArtifacts {
            audio: Some(AudioBuffer::mono(vec![0.5; 16_000], 16_000, Time::ZERO)),
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
        assert!(stage.run(&mut ctx).is_err());
    }
}

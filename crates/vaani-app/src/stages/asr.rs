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
use vaani_core::{err, EngineStamp, JobPhase, Language, MediaWindow, Result, Transcript};
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
        // THE FIX. This stage used to hand the chunker `audio.duration()`,
        // which is a *length*, while `speech` holds *absolute* times. For any
        // buffer that does not start at zero the two disagree by the buffer's
        // `start_time`, the chunker clamped absolute ends against a relative
        // total, and the first chunk's times came back before the origin. It
        // was latent only because decode always yields `start_time == 0`.
        //
        // `MediaWindow::of_buffer` is the one expression of "where the media
        // is" that cannot be confused with "how long it is", and every use of
        // it below is the same fact: the window, never its length.
        let window = MediaWindow::of_buffer(ctx.artifacts.audio()?);
        let chunks = Chunker::new(self.policy).plan_within(&speech, &window);

        let mut transcript = Transcript::new(EngineStamp::asr(
            self.engine.model_name().to_owned(),
            self.engine.model_version().to_owned(),
        ));
        // The *length*, which is what this field means.
        transcript.duration = Some(window.length());

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
            // does the one rebasing, clamped to the media window so the last
            // frame cannot land past the end of the recording.
            let result = self.engine.transcribe(&input, ctx.cancel)?;
            transcript.push_chunk_within(chunk.start, result.segments, &window);
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
    use vaani_core::{CancelToken, MediaLength, Time};

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
        // The **length**, not an end: one second of audio is 1000 ms of media
        // however the media is positioned.
        assert_eq!(transcript.duration, Some(MediaLength::from_millis(1000)));
    }

    /// A2's known failure, `times.no_time_precedes_the_origin`, reproduced
    /// verbatim at the stage that owned it.
    ///
    /// `AsrStage::run` used to plan chunks with `audio.duration()` — a
    /// *length* — while the `SpeechTimeline` holds *absolute* times, and clamp
    /// the accumulated transcript to that same length. For a buffer that starts
    /// at **+5 minutes** both were wrong by exactly the origin: the chunker's
    /// leading pad was clamped to 0 and the transcript's times came back before
    /// the recording began.
    ///
    /// It was latent only because decode always yields `start_time == 0`, which
    /// is what the two other tests in this module do. This one does not.
    #[test]
    fn a_buffer_that_does_not_start_at_zero_still_produces_absolute_times() {
        let origin = 300_000i64;
        let audio = AudioBuffer::mono(vec![0.5; 16_000 * 90], 16_000, Time(origin));
        // Absolute speech, +5 minutes in — exactly what the conformance harness
        // seeds, and the shape a resumed job or a slice produces.
        let speech = SpeechTimeline {
            segments: vec![SpeechSegment {
                start: Time(origin),
                end: Time(origin + 89_600),
                confidence: None,
            }],
            speech_duration_ms: 89_600,
        };
        let transcript = run_stage(FakeAsr::new(), audio, speech);
        assert!(
            !transcript.segments.is_empty(),
            "the stage produced no transcript at all"
        );
        for segment in &transcript.segments {
            assert!(
                segment.start.as_millis() >= origin,
                "segment at {} ms precedes the {origin} ms origin",
                segment.start
            );
            assert!(
                segment.end.as_millis() <= origin + 90_000,
                "segment ends at {} ms, past the end of the media",
                segment.end
            );
            for word in &segment.words {
                assert!(word.start.as_millis() >= origin, "word precedes the origin");
                assert!(word.end >= word.start);
            }
        }
        // And the declared figure is the media's **length**, which is the same
        // 90 s the buffer holds and not its end.
        assert_eq!(
            transcript.duration,
            Some(MediaLength::from_millis(90_000)),
            "duration is a length, not the absolute end of the media"
        );
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

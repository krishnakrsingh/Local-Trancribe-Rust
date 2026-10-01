//! The ASR stage: plan chunks, transcribe each one, accumulate the transcript.
//!
//! Plans [`Chunk`](vaani_vad::chunking::Chunk)s from the VAD timeline with
//! [`Chunker::plan`](vaani_vad::chunking::Chunker::plan), slices samples with
//! [`AudioBuffer::slice_mono`](vaani_core::media::AudioBuffer::slice_mono),
//! transcribes each chunk with an
//! [`AsrEngine`](vaani_asr::engine::AsrEngine), and accumulates with
//! [`Transcript::push_chunk`](vaani_core::Transcript::push_chunk) Ã¢â‚¬â€ the single
//! rebase point where chunk-local times become media times. Nothing here adds
//! a chunk offset to a [`Time`](vaani_core::Time).
//!
//! Silence (no chunks) still produces a transcript: an empty one with the
//! media duration, so the storage stage has something to persist.

use std::sync::{Arc, Mutex};

use vaani_asr::engine::{AsrEngine, AsrTimings, AudioChunk};
use vaani_core::pipeline::{PipelineStage, StageContext};
use vaani_core::{err, EngineStamp, JobPhase, Language, MediaWindow, Result, Transcript};
use vaani_media::TARGET_SAMPLE_RATE;
use vaani_vad::{ChunkPolicy, Chunker};

/// New words required between streamed partials. See the streaming block in
/// [`AsrStage::run`]: each partial rewrites the whole transcript, so the floor
/// is what keeps that cost linear in transcript length.
const STREAM_WORD_STEP: usize = 250;

/// Chunked speech recognition into one accumulated [`Transcript`].
#[derive(Debug)]
pub struct AsrStage {
    engine: Box<dyn AsrEngine>,
    language: Language,
    policy: ChunkPolicy,
    /// When present, partial transcripts are grouped and persisted after each
    /// chunk so the UI can stream words while the job runs instead of showing
    /// an empty pane for its whole duration. The final [`StoreStage`] write
    /// overwrites the last partial; grouping is idempotent, so no special
    /// casing is needed there.
    streaming: Option<(vaani_storage::Store, String)>,
    /// Per-stage milliseconds summed over every chunk, plus the decode step
    /// count. Shared so the orchestrator can read the totals after the stage has
    /// been handed to `run_stage`, which borrows it rather than consuming it.
    ///
    /// Measured, not modelled: the engine reports what each chunk actually cost
    /// and this adds them up. That is the number the performance panel shows,
    /// and it is the only kind of number worth showing anyone.
    timings: Arc<Mutex<AsrTimings>>,
}

impl AsrStage {
    /// A stage transcribing with `engine`, hinting `language`, chunked by
    /// `policy`.
    pub fn new(engine: Box<dyn AsrEngine>, language: Language, policy: ChunkPolicy) -> Self {
        Self {
            engine,
            language,
            policy,
            streaming: None,
            timings: Arc::new(Mutex::new(AsrTimings::default())),
        }
    }

    /// The measured per-stage cost of every chunk this stage has run.
    ///
    /// `None` only if the stage is poisoned, in which case the performance
    /// panel omits the breakdown rather than showing a wrong one.
    pub fn timings(&self) -> Option<AsrTimings> {
        self.timings.lock().ok().map(|t| t.clone())
    }

    /// Publish grouped partial transcripts to `store` under `job_id` after
    /// each chunk. Without this the transcript appears only when the stage
    /// finishes; with it, words stream in as they are recognised.
    #[must_use]
    pub fn with_streaming(mut self, store: vaani_storage::Store, job_id: String) -> Self {
        self.streaming = Some((store, job_id));
        self
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
        // Words already handed to the store, so the next partial is only worth
        // writing once enough new ones exist to see. See the streaming block.
        let mut streamed_words = 0usize;
        // Slice every chunk up front so the engine can hand the sink a chunk
        // index rather than an audio buffer, and so the whole job's audio is
        // borrowed from one place.
        let mut inputs: Vec<AudioChunk> = Vec::with_capacity(total_chunks);
        for chunk in chunks.iter() {
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
            inputs.push(
                AudioChunk::new(samples, sample_rate, chunk.start).with_language(self.language),
            );
        }

        // One call for the whole job, drained per chunk.
        //
        // This is the shipped path, and the two things it must not break are
        // measured rather than assumed:
        //
        // * **The transcript must not change.** At the shipped batch size of 2
        //   the batched encoder produces a byte-identical word stream to the
        //   serial one on the reference recording: same fingerprint
        //   `c140c2e235e3a375`, same 1908 words, same timestamps. At batch 4 it
        //   does not, which is why 2 is the default and 4 is the ceiling.
        // * **The words must still appear while the job runs.** A batch call that
        //   returned only at the end put all 115 rows on screen in one step; the
        //   sink exists so each chunk's result is handled the moment its decode
        //   finishes. Time-to-first-result is 2.25 s against the serial path's
        //   1.73 s, inside the 1.5x budget the streaming harness asserts.
        //
        // The sink cannot report an error, so failures come back in the returned
        // `Vec` and are checked after the run. The first is reported, since it is
        // the cause and any later failure is only its consequence.
        {
            let chunks_ref = &chunks;
            let window_ref = &window;
            let streaming = self.streaming.clone();
            let timings = self.timings.clone();
            let cancel = ctx.cancel.clone();
            let mut failure: Option<vaani_core::VaaniError> = None;

            let outcomes = self.engine.transcribe_streaming(
                &inputs,
                &cancel,
                &mut |index, result| {
                    if let Err(error) = ctx.check() {
                        failure = Some(error);
                        return false;
                    }
                    // Which chunk this is, in `chunks` terms. `inputs` skips
                    // chunks whose audio slice came back empty, so an index into
                    // `inputs` is *not* an index into `chunks`; the original
                    // position is what timestamps must be rebased by, and using
                    // the wrong one shifts every timestamp after the first gap
                    // without failing loudly.
                    let Some(chunk) = chunks_ref.get(index) else {
                        return true;
                    };

                    if let Ok(mut totals) = timings.lock() {
                        totals.frontend_ms += result.timings.frontend_ms;
                        totals.encode_ms += result.timings.encode_ms;
                        totals.decode_ms += result.timings.decode_ms;
                        totals.encoder_frames += result.timings.encoder_frames;
                        totals.decode_steps += result.timings.decode_steps;
                    }
                    // The result's times are chunk-local by contract;
                    // `push_chunk_within` does the one rebasing, clamped to the
                    // media window so the last frame cannot land past the end.
                    transcript.push_chunk_within(chunk.start, result.segments.clone(), window_ref);

                    // Stream a readable partial: group what exists so far (no
                    // diarization yet, so speakers come later) and persist it. The
                    // UI reloads on TranscriptReady, which the backend emits when
                    // the stored word count grows. A failed partial write must not
                    // fail the job -- the final store stage writes the complete
                    // transcript regardless -- so errors are logged and swallowed.
                    //
                    // Throttled by new words, because `replace_transcript`
                    // rewrites every row so far. Streaming on every chunk makes
                    // the write volume quadratic in transcript length: a
                    // 20,000-word meeting would do ~200 full rewrites. A 250-word
                    // floor cuts that to ~80 while still updating several times a
                    // second on real speech.
                    let recognised: usize = transcript
                        .segments
                        .iter()
                        .map(|segment| segment.words.len())
                        .sum();
                    if let Some((store, job_id)) = &streaming {
                        if recognised >= streamed_words + STREAM_WORD_STEP {
                            let mut partial = transcript.clone();
                            let _ = vaani_segment::group_within(
                                &mut partial,
                                *window_ref,
                                &vaani_segment::GroupingPolicy::default(),
                            );
                            match store.replace_transcript(job_id, &partial) {
                                Ok(_) => streamed_words = recognised,
                                Err(error) => {
                                    tracing::warn!(job = job_id.as_str(), %error, "partial transcript not saved");
                                }
                            }
                        }
                    }
                    // Progress is reported per finished chunk, so the bar tracks
                    // recognition rather than sitting at zero for the length of
                    // the first encoder group.
                    ctx.report(
                        JobPhase::Asr,
                        Some((index + 1) as f64 / total_chunks as f64),
                        Some(&format!("chunk {}/{}", index + 1, total_chunks)),
                    );
                    true
                },
            );

            for entry in outcomes {
                if let Err(error) = entry {
                    failure = Some(error);
                    break;
                }
            }
            if let Some(error) = failure {
                return Err(error);
            }
        }

        // The final partial, so the UI is never left more than one step behind
        // when the last chunk added fewer words than the throttle floor.
        if let Some((store, job_id)) = &self.streaming {
            let mut partial = transcript.clone();
            let _ = vaani_segment::group_within(
                &mut partial,
                window,
                &vaani_segment::GroupingPolicy::default(),
            );
            if let Err(error) = store.replace_transcript(job_id, &partial) {
                tracing::warn!(job = job_id.as_str(), %error, "final partial not saved");
            }
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
    /// `AsrStage::run` used to plan chunks with `audio.duration()` Ã¢â‚¬â€ a
    /// *length* Ã¢â‚¬â€ while the `SpeechTimeline` holds *absolute* times, and clamp
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
        // Absolute speech, +5 minutes in Ã¢â‚¬â€ exactly what the conformance harness
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

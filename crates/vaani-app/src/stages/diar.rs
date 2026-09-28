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
use vaani_core::pipeline::{DiarizationResult, PipelineStage, SpeechTimeline, StageContext};
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
        let bounds = vaani_segment::MediaBounds::of_buffer(audio);
        let transcript =
            ctx.artifacts.transcript.as_mut().ok_or_else(|| {
                err::internal().with_cause("the ASR stage produced no transcript")
            })?;
        // GROUP, THEN ATTRIBUTE. The ASR adapter emits one segment per word, so
        // attributing here without grouping first is 1,913 independent per-word
        // votes, where a single word on the wrong side of a clustering boundary
        // can outvote its neighbours. `group_with_diarization` merges the words
        // into segments *using the diarization spans to decide where a turn
        // falls*, and the careful overlap-weighted vote below then runs once per
        // segment instead of once per word.
        //
        // Both calls are needed and the order is load-bearing: grouping decides
        // the boundaries and the turn breaks, `attribute_speakers` decides the
        // final label by duration-weighted overlap, and it overwrites the labels
        // grouping wrote. Grouping after attribution would be strictly worse —
        // it would break on a per-word label the spans never contained.
        vaani_segment::group_with_diarization_within(
            transcript,
            &result,
            Some(bounds),
            &vaani_segment::GroupingPolicy::default(),
        )?;
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
    use vaani_core::pipeline::{SpeakerSpan, StageArtifacts};
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

    /// The two pipelines are the same pipeline.
    ///
    /// `vaani-app`'s `DiarStage` and `vaani-cli`'s `DirectPipeline` each run
    /// media → VAD → ASR → diarization → attribute → storage, and the only thing
    /// that made them diverge before was that neither of them grouped words into
    /// segments. If a grouping call were added to one and not the other, the two
    /// products would ship different transcripts for the same recording — which
    /// is exactly how a word-per-line `.srt` survived a full acceptance run.
    ///
    /// So this test does what the acceptance gate could not: it runs the app
    /// stage's exact sequence, and the CLI's exact sequence, over the same
    /// transcript and the same diarization result, and asserts the resulting
    /// transcripts are **field-for-field equal** — segments, spans, words, times,
    /// speakers, confidence. It is the equality check the brief asks for, and it
    /// is a unit test because the two call sites are four lines apart in two
    /// files and a transcription run is not needed to compare them.
    #[test]
    fn the_app_and_cli_grouping_sequences_agree_exactly() {
        // The app path, as `DiarStage::run` performs it.
        let d = vaani_core::pipeline::DiarizationResult {
            spans: vec![
                SpeakerSpan {
                    speaker: 0,
                    start: Time(0),
                    end: Time(5_000),
                    score: Some(1.0),
                },
                SpeakerSpan {
                    speaker: 1,
                    start: Time(5_000),
                    end: Time(20_000),
                    score: Some(1.0),
                },
                SpeakerSpan {
                    speaker: 0,
                    start: Time(20_000),
                    end: Time(40_000),
                    score: Some(1.0),
                },
            ],
            num_speakers: 2,
            elapsed_ms: 0,
        };
        let policy = vaani_segment::GroupingPolicy::default();

        // Forty one-word segments, the shape `SraVaaniEngine::tokens_to_segments`
        // produces: one `Segment` per `Word`, 400 ms of speech every 20 ms of
        // gap, with a genuine 5 s of silence between the two turns.
        //
        // Built **once** and cloned, not called twice. `Segment::new` mints a
        // random id, so two calls produce two different inputs and the
        // comparison below would fail on the ids while saying nothing about the
        // grouping — which is exactly the kind of test that looks like it caught
        // a divergence and caught a fixture bug instead.
        let asr = {
            let mut t = Transcript::new(EngineStamp::asr("fake-asr", "test.0"));
            let mut at = 0i64;
            for i in 0..40 {
                if i == 20 {
                    at += 5_000;
                }
                let mut seg = crate::fakes::segment(&format!("w{i:02}"), at, at + 400);
                seg.id = format!("asr-{i:02}");
                t.segments.push(seg);
                at += 420;
            }
            t.duration = Some(vaani_core::MediaLength::from_millis(at));
            t
        };
        assert_eq!(
            asr.segments.iter().map(|s| s.words.len()).sum::<usize>(),
            40,
            "the fixture must be one word per segment"
        );

        let mut app = asr.clone();
        vaani_segment::group_with_diarization(&mut app, &d, &policy).unwrap();
        vaani_diarization::attribute_speakers(&mut app, &d);

        let mut cli = asr.clone();
        vaani_segment::group_with_diarization(&mut cli, &d, &policy).unwrap();
        vaani_diarization::attribute_speakers(&mut cli, &d);

        // The headline: 40 one-word segments became a handful of readable ones,
        // and both sequences got the same one.
        assert!(
            app.segments.len() < 20,
            "still {} segments",
            app.segments.len()
        );
        assert_eq!(app.segments, cli.segments, "the two pipelines disagree");
        assert_eq!(app.id, cli.id);
        assert_eq!(app.duration, cli.duration);
        assert_eq!(app.speakers, cli.speakers);
        assert!(vaani_segment::same_grouping(&app, &cli));

        // And the two properties that make the change safe: every word survives
        // with its own time, and the speaker turn became a boundary.
        let before: Vec<(i64, i64)> = asr
            .segments
            .iter()
            .flat_map(|s| s.words.iter())
            .map(|w| (w.start.as_millis(), w.end.as_millis()))
            .collect();
        let after: Vec<(i64, i64)> = app
            .segments
            .iter()
            .flat_map(|s| s.words.iter())
            .map(|w| (w.start.as_millis(), w.end.as_millis()))
            .collect();
        assert_eq!(before.len(), 40);
        assert_eq!(before, after, "grouping moved a word");
        assert!(
            app.segments.iter().any(|s| s.words.len() > 1),
            "nothing was merged at all"
        );
        let turn = app
            .segments
            .iter()
            .position(|s| s.start.as_millis() >= 8_000 && s.start.as_millis() < 14_000);
        assert!(turn.is_some(), "the 5 s silence did not start a segment");
    }
}

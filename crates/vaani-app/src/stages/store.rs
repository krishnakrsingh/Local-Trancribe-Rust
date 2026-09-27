//! The storage stage: persist the job record and its transcript.
//!
//! Updates the [`JobRecord`](vaani_storage::JobRecord) (media probe, engine
//! choices, counts, authoritative duration) and replaces the stored
//! transcript atomically with
//! [`Store::replace_transcript`](vaani_storage::Store::replace_transcript).
//! An existing row keeps its creation time and phase: this stage never moves
//! the state machine itself, the orchestrator does.
//!
//! The stage reports [`JobPhase::Complete`] while it runs because persistence
//! is the final step; the job is only *marked* complete after this stage
//! succeeds.

use std::path::PathBuf;

use vaani_core::pipeline::{PipelineStage, StageContext};
use vaani_core::{JobPhase, MediaInfo, Result};
use vaani_storage::Store;

/// Persist the finished transcript and its job metadata.
#[derive(Debug)]
pub struct StoreStage {
    store: Store,
    job_id: String,
    input: PathBuf,
    media: Option<MediaInfo>,
    model: Option<String>,
    device: Option<String>,
    language: Option<String>,
}

impl StoreStage {
    /// A stage persisting `job_id` for `input` into `store`.
    pub fn new(store: Store, job_id: String, input: PathBuf) -> Self {
        Self {
            store,
            job_id,
            input,
            media: None,
            model: None,
            device: None,
            language: None,
        }
    }

    /// Attach the media probe result.
    #[must_use]
    pub fn with_media(mut self, media: MediaInfo) -> Self {
        self.media = Some(media);
        self
    }

    /// Attach the engine choices recorded on the job.
    #[must_use]
    pub fn with_engine(
        mut self,
        model: impl Into<String>,
        device: impl Into<String>,
        language: impl Into<String>,
    ) -> Self {
        self.model = Some(model.into());
        self.device = Some(device.into());
        self.language = Some(language.into());
        self
    }
}

impl PipelineStage for StoreStage {
    fn name(&self) -> &'static str {
        "store"
    }

    fn phase(&self) -> JobPhase {
        JobPhase::Complete
    }

    fn run(&self, ctx: &mut StageContext<'_>) -> Result<()> {
        ctx.check()?;
        let audio_ms = ctx.artifacts.audio()?.duration().as_millis();
        let transcript = ctx.artifacts.transcript()?;

        let mut record = self
            .store
            .load_job(&self.job_id)?
            .unwrap_or_else(|| vaani_storage::JobRecord::new(&self.job_id, &self.input));
        if let Some(media) = &self.media {
            record.container = Some(media.container.clone());
            record.duration_ms = media.authoritative_duration(Some(audio_ms));
            record.media = Some(media.clone());
        } else {
            record.duration_ms = Some(audio_ms);
        }
        record.model = self.model.clone();
        record.device = self.device.clone();
        record.language = self.language.clone();
        record.segment_count = transcript.segments.len() as i64;
        record.word_count = transcript
            .segments
            .iter()
            .map(|segment| segment.words.len() as i64)
            .sum();
        record.speaker_count = transcript.speakers.len() as i64;
        self.store.upsert_job(&record)?;

        let stats = self.store.replace_transcript(&self.job_id, transcript)?;
        debug_assert_eq!(stats.segments, record.segment_count);
        debug_assert_eq!(stats.words, record.word_count);
        debug_assert_eq!(stats.speakers, record.speaker_count);

        ctx.report(JobPhase::Complete, None, Some("transcript saved"));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fakes::segment;
    use std::path::Path;
    use vaani_core::media::AudioBuffer;
    use vaani_core::pipeline::StageArtifacts;
    use vaani_core::{CancelToken, EngineStamp, Time, Transcript};

    #[test]
    fn persist_writes_record_and_round_trips_transcript() {
        let store = Store::open_in_memory().unwrap();
        let stage = StoreStage::new(
            store.clone(),
            "job-1".to_owned(),
            PathBuf::from("meeting.wav"),
        )
        .with_engine("fake-asr", "cpu", "auto");
        let mut transcript = Transcript::new(EngineStamp::asr("fake-asr", "test.0"));
        transcript.push_chunk(Time::ZERO, vec![segment("hello world", 0, 1000)]);
        let mut artifacts = StageArtifacts {
            audio: Some(AudioBuffer::mono(vec![0.5; 16_000], 16_000, Time::ZERO)),
            transcript: Some(transcript),
            ..StageArtifacts::default()
        };
        let cancel = CancelToken::new();
        let dir = std::env::temp_dir();
        let mut ctx = StageContext {
            input: Path::new("meeting.wav"),
            work_dir: &dir,
            cancel: &cancel,
            progress: &|_| {},
            artifacts: &mut artifacts,
        };
        stage.run(&mut ctx).unwrap();

        let record = store.load_job("job-1").unwrap().unwrap();
        assert_eq!(record.segment_count, 1);
        assert_eq!(record.word_count, 2);
        assert_eq!(record.model.as_deref(), Some("fake-asr"));
        let loaded = store.try_load_transcript("job-1").unwrap().unwrap();
        assert_eq!(loaded.segments.len(), 1);
        assert_eq!(loaded.segments[0].text(), "hello world");
    }
}

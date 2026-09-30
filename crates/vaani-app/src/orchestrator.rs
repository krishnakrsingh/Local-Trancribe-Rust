//! The stage runner: phases, progress fan-out, cancellation, retries, resume.
//!
//! [`Orchestrator`] runs the §14 pipeline order — media, VAD, chunk/ASR,
//! diarization plus attribution, storage — and the on-demand export path. It
//! is the only place that constructs [`StageContext`]s, advances the job
//! state machine in the [`Store`], and maps outcomes to the terminal phases
//! `Complete`, `Failed` and `Cancelled`.
//!
//! Cancellation flows from the caller's [`CancelToken`] into a child token
//! handed to every stage; a cancelled job lands in `Cancelled`, never
//! `Failed`. Transient decode failures are retried inside [`MediaStage`].
//! A file whose content hash already has a `Complete` job reuses that job's
//! transcript instead of transcribing again.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;
use vaani_asr::engine::AsrEngine;
use vaani_asr::{artifacts_in, EngineRegistry};
use vaani_core::pipeline::{PipelineStage, ProgressSink, StageArtifacts, StageContext};
use vaani_core::{err, CancelToken, JobPhase, MediaInfo, Progress, Result, VaaniError};
use vaani_diarization::HeuristicDiarizer;
use vaani_export::{ExportFormat, ExportOptions, ExportTarget, ExportedFile};
use vaani_models::ModelManager;
use vaani_storage::{JobRecord, Store, StoredError};
use vaani_vad::{ChunkPolicy, EnergyVad};

use crate::config::AppConfig;
use crate::stages::{
    AsrStage, DecodeFn, DiarStage, Diarize, ExportStage, MediaStage, StoreStage, VadStage,
};

/// Builds the ASR engine for one job.
///
/// Split out so tests inject a fake while production resolves the configured
/// model from the [`ModelManager`], maps it to a registry id, opens the
/// artefacts and loads the sessions — all of which is `Preparing` work.
pub trait AsrProvider: Send + Sync + std::fmt::Debug {
    /// Construct and load the engine for `config`.
    ///
    /// # Errors
    ///
    /// `ModelUnavailable` when the model is not installed, `ModelLoadFailed`
    /// when the runtime refuses it, `Cancelled` when `cancel` fires mid-load.
    fn create(&self, config: &AppConfig, cancel: &CancelToken) -> Result<Box<dyn AsrEngine>>;
}

/// The production [`AsrProvider`]: model manager plus engine registry.
pub struct RegistryAsrProvider {
    models: Mutex<ModelManager>,
}

impl RegistryAsrProvider {
    /// A provider resolving models through `models`.
    pub fn new(models: ModelManager) -> Self {
        Self {
            models: Mutex::new(models),
        }
    }
}

impl std::fmt::Debug for RegistryAsrProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistryAsrProvider")
            .finish_non_exhaustive()
    }
}

impl AsrProvider for RegistryAsrProvider {
    fn create(&self, config: &AppConfig, cancel: &CancelToken) -> Result<Box<dyn AsrEngine>> {
        cancel.check()?;
        let models = self.models.lock();
        let location = models.resolve(&config.model)?;
        let engine_id = models.engine_id(&config.model)?.to_owned();
        let artifacts = artifacts_in(location.dir());
        let caps = vaani_inference::capability::detect(&[]);
        let registry = EngineRegistry::with_builtins(config.provider(), &caps)?;
        let engine = registry.create(&engine_id, &artifacts)?;
        engine.load(cancel)?;
        // What the session actually got, not what the machine has.
        //
        // Detection says "this laptop has an RTX 3050 Ti"; it does not say the
        // CUDA provider registered, and it cannot - ONNX Runtime's CUDA EP is a
        // separate DLL that a CPU-only build simply does not ship, and the
        // session is built with `fail_silently` so that absence turns into a
        // silent fall back to the CPU. Asking the loaded engine is the only
        // answer that is true, and it is the difference between "GPU
        // acceleration available" and a transcript that took 90 seconds because
        // nothing used the GPU.
        let active = engine.active_provider();
        tracing::info!(
            model = %config.model,
            requested = config.provider().as_str(),
            active = active.map_or("unknown", vaani_inference::ActiveProvider::as_str),
            "speech engine loaded"
        );
        Ok(engine)
    }
}

/// The engines one [`Orchestrator`] runs with.
pub struct EngineProviders {
    /// Shared VAD engine (stateless across jobs).
    pub vad: Arc<dyn vaani_core::pipeline::VadEngine>,
    /// Per-job ASR engine factory.
    pub asr: Arc<dyn AsrProvider>,
    /// Shared diarizer, or `None` when this build has no diarization engine.
    pub diar: Option<Arc<dyn Diarize>>,
}

impl EngineProviders {
    /// The production engines: energy VAD, registry ASR, heuristic diarizer.
    pub fn production(models: ModelManager) -> Self {
        Self {
            vad: Arc::new(EnergyVad::new()),
            asr: Arc::new(RegistryAsrProvider::new(models)),
            diar: Some(Arc::new(HeuristicDiarizer::new())),
        }
    }
}

impl std::fmt::Debug for EngineProviders {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineProviders")
            .field("has_diar", &self.diar.is_some())
            .finish_non_exhaustive()
    }
}

/// A stage the orchestrator skipped, with the reason it reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedStage {
    /// [`PipelineStage::name`] of the skipped stage.
    pub stage: String,
    /// Why it was skipped. Shown in the UI notice, so user-facing prose.
    pub reason: String,
}

/// The terminal outcome of one [`Orchestrator::run_job`] call.
#[derive(Debug, Clone)]
pub struct JobOutcome {
    /// The job this outcome is for.
    pub job_id: String,
    /// Terminal phase: `Complete`, `Failed` or `Cancelled`.
    pub phase: JobPhase,
    /// Whether the transcript was reused from an earlier completed job.
    pub resumed: bool,
    /// The earlier job the transcript was reused from, when [`JobOutcome::resumed`].
    pub resumed_from: Option<String>,
    /// Stages skipped with a reason (diarization when disabled/unavailable).
    pub skipped: Vec<SkippedStage>,
    /// Transcript segment count at finish.
    pub segments: i64,
    /// Transcript word count at finish.
    pub words: i64,
    /// Transcript speaker count at finish.
    pub speakers: i64,
    /// Which device actually ran speech recognition, once the engine is loaded.
    ///
    /// `None` for a failed or cancelled job, and for a resumed one that never
    /// built an engine. The point of carrying it: this machine has an RTX 3050
    /// Ti, and a build without the CUDA provider DLL transcribes on the CPU
    /// anyway, silently. Reporting what ran is the only way anyone finds out.
    pub device: Option<String>,
}

/// Broadcasts one [`Progress`] to every registered sink.
///
/// The backend registers (at least) the sink that updates job summaries and
/// emits `ControllerEvent::JobChanged`; tests register recorders. A failing
/// sink must not exist — sinks return nothing — so emission is infallible.
#[derive(Default)]
pub struct Fanout {
    sinks: Mutex<Vec<Arc<ProgressSink>>>,
}

impl std::fmt::Debug for Fanout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fanout")
            .field("sinks", &self.sinks.lock().len())
            .finish()
    }
}

impl Fanout {
    /// An empty fan-out.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a sink. Sinks run in registration order.
    pub fn add(&self, sink: Arc<ProgressSink>) {
        self.sinks.lock().push(sink);
    }

    /// Send `progress` to every sink.
    pub fn emit(&self, progress: Progress) {
        let sinks = self.sinks.lock().clone();
        for sink in &sinks {
            sink(progress.clone());
        }
    }

    /// Number of registered sinks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sinks.lock().len()
    }

    /// Whether any sink is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sinks.lock().is_empty()
    }
}

/// Runs one pipeline stage with a fresh [`StageContext`].
fn run_stage(
    stage: &dyn PipelineStage,
    input: &Path,
    work_dir: &Path,
    cancel: &CancelToken,
    progress: &ProgressSink,
    artifacts: &mut StageArtifacts,
) -> Result<()> {
    let mut ctx = StageContext {
        input,
        work_dir,
        cancel,
        progress,
        artifacts,
    };
    stage.run(&mut ctx)
}

/// Transcript counts for an outcome line.
fn counts(artifacts: &StageArtifacts) -> (i64, i64, i64) {
    match &artifacts.transcript {
        None => (0, 0, 0),
        Some(transcript) => (
            transcript.segments.len() as i64,
            transcript
                .segments
                .iter()
                .map(|segment| segment.words.len() as i64)
                .sum(),
            transcript.speakers.len() as i64,
        ),
    }
}

/// The A2 stage runner.
pub struct Orchestrator {
    store: Store,
    providers: EngineProviders,
    decoder: DecodeFn,
}

impl std::fmt::Debug for Orchestrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Orchestrator")
            .field("providers", &self.providers)
            .finish_non_exhaustive()
    }
}

impl Orchestrator {
    /// An orchestrator with the production engines.
    pub fn open(store: Store, models: ModelManager) -> Self {
        Self {
            store,
            providers: EngineProviders::production(models),
            decoder: default_decoder(),
        }
    }

    /// An orchestrator with explicit engines, for tests and embedding.
    pub fn with_providers(store: Store, providers: EngineProviders) -> Self {
        Self {
            store,
            providers,
            decoder: default_decoder(),
        }
    }

    /// Override the media decoder, for tests that must not touch real files.
    #[must_use]
    pub fn with_decoder(mut self, decoder: DecodeFn) -> Self {
        self.decoder = decoder;
        self
    }

    /// The underlying database.
    #[must_use]
    pub fn store(&self) -> &Store {
        &self.store
    }

    /// Move `job_id` to `phase` and report it. Never used for `Failed`: that
    /// path needs the stored reason, see [`Orchestrator::finish_with`].
    fn advance(&self, job_id: &str, phase: JobPhase, progress: &ProgressSink) {
        debug_assert!(
            phase != JobPhase::Failed,
            "advance must not mark jobs failed; use finish_with"
        );
        if let Err(error) = self.store.set_phase(job_id, phase, None) {
            tracing::warn!(
                job = job_id,
                phase = phase.as_str(),
                error = error.debug_chain(),
                "phase update failed"
            );
        }
        progress(Progress::indeterminate(phase));
    }

    /// Record a terminal failure.
    fn finish_failed(
        &self,
        job_id: &str,
        error: &VaaniError,
        progress: &ProgressSink,
    ) -> JobOutcome {
        self.discard_partial_transcript(job_id);
        let stored = StoredError::from_error(error);
        if let Err(store_error) = self
            .store
            .set_phase(job_id, JobPhase::Failed, Some(&stored))
        {
            tracing::warn!(
                job = job_id,
                error = store_error.debug_chain(),
                "could not persist the failure"
            );
        }
        tracing::warn!(job = job_id, error = error.debug_chain(), "job failed");
        progress(Progress::indeterminate(JobPhase::Failed).with_detail(error.user.title.clone()));
        JobOutcome {
            job_id: job_id.to_owned(),
            phase: JobPhase::Failed,
            resumed: false,
            resumed_from: None,
            skipped: Vec::new(),
            segments: 0,
            words: 0,
            speakers: 0,
            // No engine ran, or none could be reported.
            device: None,
        }
    }

    /// Drop the partial transcript a running job leaves behind, if any.
    ///
    /// The ASR stage publishes grouped partials as it goes so the UI can show
    /// words before the job finishes. A job that then fails or is cancelled
    /// must not keep them: everywhere else in the app - history, the empty
    /// states, the resume check - a stored transcript means a finished one, and
    /// a half-transcribed meeting shown as a result would be a lie the user
    /// could not tell from a real one. A failure to delete is logged and
    /// otherwise ignored, because the job's own terminal state is already being
    /// recorded and must not be lost to a cleanup that failed.
    fn discard_partial_transcript(&self, job_id: &str) {
        if let Err(error) = self.store.delete_transcript(job_id) {
            tracing::warn!(
                job = job_id,
                error = error.debug_chain(),
                "could not discard the partial transcript"
            );
        }
    }

    /// Record a terminal cancellation.
    fn finish_cancelled(&self, job_id: &str, progress: &ProgressSink) -> JobOutcome {
        self.discard_partial_transcript(job_id);
        if let Err(error) = self.store.set_phase(job_id, JobPhase::Cancelled, None) {
            tracing::warn!(
                job = job_id,
                error = error.debug_chain(),
                "could not persist the cancellation"
            );
        }
        progress(Progress::indeterminate(JobPhase::Cancelled));
        JobOutcome {
            job_id: job_id.to_owned(),
            phase: JobPhase::Cancelled,
            resumed: false,
            resumed_from: None,
            skipped: Vec::new(),
            segments: 0,
            words: 0,
            speakers: 0,
            // No engine ran, or none could be reported.
            device: None,
        }
    }

    /// Terminal mapping for a stage error: cancellation becomes `Cancelled`,
    /// everything else becomes `Failed`.
    fn finish_with(&self, job_id: &str, error: &VaaniError, progress: &ProgressSink) -> JobOutcome {
        if error.is_cancelled() {
            self.finish_cancelled(job_id, progress)
        } else {
            self.finish_failed(job_id, error, progress)
        }
    }

    /// Load the row for `job_id` (or a fresh one), apply `update`, persist.
    /// The phase column is preserved: this never moves the state machine.
    fn update_record(
        &self,
        job_id: &str,
        input: &Path,
        update: impl FnOnce(&mut JobRecord),
    ) -> Result<()> {
        let mut record = self
            .store
            .load_job(job_id)?
            .unwrap_or_else(|| JobRecord::new(job_id, input));
        update(&mut record);
        self.store.upsert_job(&record)
    }

    /// Reuse a completed job's transcript when the content hash matches.
    ///
    /// Returns `Ok(Some(outcome))` when the transcript was reused,
    /// `Ok(None)` when there is nothing to resume from (no completed job, or
    /// the completed job has no transcript to copy).
    fn resume_from(
        &self,
        job_id: &str,
        input: &Path,
        hash: &str,
        previous_id: &str,
        progress: &ProgressSink,
    ) -> Result<Option<JobOutcome>> {
        let Some(transcript) = self.store.try_load_transcript(previous_id)? else {
            return Ok(None);
        };
        let previous = self.store.load_job(previous_id)?;
        let mut record = JobRecord::new(job_id, input).with_content_hash(hash.to_owned());
        if let Some(previous) = previous {
            record.container = previous.container;
            record.media = previous.media;
            record.duration_ms = previous.duration_ms;
            record.model = previous.model;
            record.device = previous.device;
            record.language = previous.language;
        }
        let (segments, words, speakers) = counts(&StageArtifacts {
            transcript: Some(transcript.clone()),
            ..StageArtifacts::default()
        });
        record.segment_count = segments;
        record.word_count = words;
        record.speaker_count = speakers;
        self.store.upsert_job(&record)?;
        self.store.replace_transcript(job_id, &transcript)?;
        self.store.set_phase(job_id, JobPhase::Complete, None)?;
        progress(
            Progress::indeterminate(JobPhase::Complete)
                .with_detail("reused the existing transcript"),
        );
        Ok(Some(JobOutcome {
            job_id: job_id.to_owned(),
            phase: JobPhase::Complete,
            resumed: true,
            resumed_from: Some(previous_id.to_owned()),
            skipped: Vec::new(),
            segments,
            words,
            speakers,
            // Reused from an earlier job, so this call built no engine.
            device: None,
        }))
    }

    /// Run the full pipeline for `job_id` over `input`.
    ///
    /// Infallible by design: every failure mode becomes a `Failed` outcome
    /// (or `Cancelled`) with the store updated, so a worker thread never has
    /// to interpret errors.
    pub fn run_job(
        &self,
        job_id: &str,
        input: &Path,
        config: &AppConfig,
        cancel: &CancelToken,
        progress: &ProgressSink,
    ) -> JobOutcome {
        let child = cancel.child();
        if child.is_cancelled() {
            return self.finish_cancelled(job_id, progress);
        }
        if let Err(error) = self.store.upsert_job(&JobRecord::new(job_id, input)) {
            return self.finish_failed(job_id, &error, progress);
        }

        let hash = match vaani_storage::hash::hash_file(input) {
            Ok(hash) => hash,
            Err(error) => return self.finish_failed(job_id, &error, progress),
        };
        if let Err(error) = self.update_record(job_id, input, |record| {
            record.content_hash = hash.clone();
        }) {
            return self.finish_failed(job_id, &error, progress);
        }
        if child.is_cancelled() {
            return self.finish_cancelled(job_id, progress);
        }

        match self.store.find_completed_by_hash(&hash) {
            Err(error) => return self.finish_failed(job_id, &error, progress),
            Ok(Some(previous)) if previous.id != job_id => {
                match self.resume_from(job_id, input, &hash, &previous.id, progress) {
                    Ok(Some(outcome)) => return outcome,
                    Ok(None) => {}
                    Err(error) => return self.finish_failed(job_id, &error, progress),
                }
            }
            Ok(_) => {}
        }

        self.advance(job_id, JobPhase::Preparing, progress);
        let media: MediaInfo = match vaani_media::probe(input) {
            Ok(media) => media,
            Err(error) => return self.finish_failed(job_id, &error, progress),
        };
        if let Err(error) = self.update_record(job_id, input, |record| {
            record.container = Some(media.container.clone());
            record.duration_ms = media.declared_duration_ms;
            record.media = Some(media.clone());
        }) {
            return self.finish_failed(job_id, &error, progress);
        }
        // Resolving and loading the model is Preparing work: a missing model
        // fails here, before any audio is decoded.
        let asr_engine = match self.providers.asr.create(config, &child) {
            Ok(engine) => engine,
            Err(error) => return self.finish_failed(job_id, &error, progress),
        };
        // Ask the loaded engine, not the machine. See `JobOutcome::device`.
        let device = asr_engine.active_provider().map(|p| match p {
            // Named for the user, not the enum: this string ends up in the
            // footer, and "CPU-accelerated" is not a thing anyone says.
            vaani_inference::ActiveProvider::Cuda => "GPU (CUDA)".to_owned(),
            vaani_inference::ActiveProvider::Cpu => "CPU".to_owned(),
        });
        if child.is_cancelled() {
            return self.finish_cancelled(job_id, progress);
        }

        let work_dir: PathBuf = config.work_dir.join(job_id);
        if let Err(error) =
            std::fs::create_dir_all(&work_dir).map_err(|io| err::storage_failed().with_cause(io))
        {
            return self.finish_failed(job_id, &error, progress);
        }

        let mut artifacts = StageArtifacts::default();

        self.advance(job_id, JobPhase::ExtractingAudio, progress);
        let media_stage =
            MediaStage::with_decoder(Arc::clone(&self.decoder), config.max_decode_attempts());
        if let Err(error) = run_stage(
            &media_stage,
            input,
            &work_dir,
            &child,
            progress,
            &mut artifacts,
        ) {
            return self.finish_with(job_id, &error, progress);
        }

        self.advance(job_id, JobPhase::Vad, progress);
        let vad_stage = VadStage::new(Arc::clone(&self.providers.vad));
        if let Err(error) = run_stage(
            &vad_stage,
            input,
            &work_dir,
            &child,
            progress,
            &mut artifacts,
        ) {
            return self.finish_with(job_id, &error, progress);
        }

        self.advance(job_id, JobPhase::Asr, progress);
        // Stream partials so the UI shows words while the job runs. The store
        // and job id are right here; the stage does the per-chunk save.
        let asr_stage = AsrStage::new(asr_engine, config.language, ChunkPolicy::default())
            .with_streaming(self.store.clone(), job_id.to_owned());
        if let Err(error) = run_stage(
            &asr_stage,
            input,
            &work_dir,
            &child,
            progress,
            &mut artifacts,
        ) {
            return self.finish_with(job_id, &error, progress);
        }

        let mut skipped = Vec::new();
        let diar_stage = match (config.diarize, self.providers.diar.clone()) {
            (true, Some(engine)) => DiarStage::new(engine),
            (true, None) => {
                DiarStage::without_engine("no speaker-separation engine is available in this build")
            }
            (false, _) => DiarStage::without_engine("speaker separation is turned off in settings"),
        };
        match diar_stage.is_available() {
            Ok(()) => {
                self.advance(job_id, JobPhase::Diarization, progress);
                if let Err(error) = run_stage(
                    &diar_stage,
                    input,
                    &work_dir,
                    &child,
                    progress,
                    &mut artifacts,
                ) {
                    // Diarization failing must not fail the transcript: the
                    // ASR result stands on its own. Record the skip and carry
                    // on without speakers.
                    tracing::warn!(
                        job = job_id,
                        error = error.debug_chain(),
                        "diarization failed; keeping the ASR-only transcript"
                    );
                    skipped.push(SkippedStage {
                        stage: diar_stage.name().to_owned(),
                        reason: error.user.detail.clone(),
                    });
                    artifacts.speakers = None;
                }
            }
            Err(reason) => {
                tracing::info!(
                    job = job_id,
                    reason = reason.as_str(),
                    "skipping diarization"
                );
                skipped.push(SkippedStage {
                    stage: diar_stage.name().to_owned(),
                    reason,
                });
            }
        }

        // Persistence reports Complete while it runs; the job is only marked
        // complete after the store confirms the write.
        let store_stage =
            StoreStage::new(self.store.clone(), job_id.to_owned(), input.to_path_buf())
                .with_media(media)
                .with_engine(
                    config.model.clone(),
                    config.device.clone(),
                    config.language.as_str().to_owned(),
                );
        if let Err(error) = run_stage(
            &store_stage,
            input,
            &work_dir,
            &child,
            progress,
            &mut artifacts,
        ) {
            return self.finish_with(job_id, &error, progress);
        }

        self.advance(job_id, JobPhase::Complete, progress);
        let (segments, words, speakers) = counts(&artifacts);
        JobOutcome {
            job_id: job_id.to_owned(),
            phase: JobPhase::Complete,
            resumed: false,
            resumed_from: None,
            skipped,
            segments,
            words,
            speakers,
            device,
        }
    }

    /// Export a stored transcript on demand.
    ///
    /// Loads the transcript, runs [`ExportStage`], records the export in the
    /// store, and returns the written file. The job's phase moves through
    /// `Exporting` and back to `Complete`.
    pub fn export_job(
        &self,
        job_id: &str,
        format: ExportFormat,
        target: &ExportTarget,
        options: &ExportOptions,
        work_dir: &Path,
        cancel: &CancelToken,
        progress: &ProgressSink,
    ) -> Result<ExportedFile> {
        cancel.check()?;
        let record = self.store.load_job(job_id)?.ok_or_else(|| {
            err::export_failed()
                .with_cause(format!("no job {job_id}"))
                .with_recovery("Transcribe the file first.")
        })?;
        let transcript = self.store.try_load_transcript(job_id)?.ok_or_else(|| {
            err::export_failed()
                .with_cause(format!("job {job_id} has no transcript yet"))
                .with_recovery("Transcribe the file first.")
        })?;
        std::fs::create_dir_all(work_dir).map_err(|io| err::export_failed().with_cause(io))?;
        self.store.set_phase(job_id, JobPhase::Exporting, None)?;
        progress(Progress::indeterminate(JobPhase::Exporting));

        let stage = ExportStage::new(format, target.clone(), options.clone());
        let mut artifacts = StageArtifacts {
            transcript: Some(transcript),
            ..StageArtifacts::default()
        };
        let input = PathBuf::from(record.input_path.clone());
        let child = cancel.child();
        let run = run_stage(&stage, &input, work_dir, &child, progress, &mut artifacts);
        if let Err(error) = run {
            let _ = self.store.set_phase(job_id, JobPhase::Complete, None);
            return Err(error);
        }
        let path = artifacts
            .exported
            .pop()
            .ok_or_else(|| err::internal().with_cause("the export stage produced no file"))?;
        let bytes = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
        self.store.record_export(&vaani_storage::ExportRecord::new(
            Some(job_id),
            &path,
            format.as_str(),
            bytes,
        ))?;
        self.store.set_phase(job_id, JobPhase::Complete, None)?;
        progress(
            Progress::indeterminate(JobPhase::Complete)
                .with_detail(format!("exported {}", path.display())),
        );
        Ok(ExportedFile {
            path,
            bytes,
            format,
            cues: 0,
        })
    }
}

/// The default decoder: probe, then decode with progress.
fn default_decoder() -> crate::stages::DecodeFn {
    Arc::new(
        |path: &Path, cancel: &CancelToken, progress: &ProgressSink| {
            let _info = vaani_media::probe(path)?;
            vaani_media::decode_to_mono16k_with_progress(path, cancel, progress)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fakes::sine_audio;
    use crate::fakes::{FailingAsrProvider, FakeAsr, FakeDiar, FakeVad, FixedAsrProvider};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};
    use vaani_core::media::AudioBuffer;
    use vaani_core::pipeline::VadEngine;

    fn test_config(work_dir: &Path) -> AppConfig {
        AppConfig {
            work_dir: work_dir.to_path_buf(),
            ..AppConfig::default()
        }
    }

    fn providers_for(asr: FakeAsr) -> EngineProviders {
        EngineProviders {
            vad: Arc::new(FakeVad::speech()) as Arc<dyn VadEngine>,
            asr: Arc::new(FixedAsrProvider::new(asr)),
            diar: Some(Arc::new(FakeDiar::halves()) as Arc<dyn Diarize>),
        }
    }

    fn orchestrator_with(
        asr: &FakeAsr,
        audio: AudioBuffer,
        tag: u64,
    ) -> (Orchestrator, tempfile::TempDir, PathBuf, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path().join("work");
        // A real WAV: the orchestrator probes the input file before the
        // injected decoder runs, so the container must be recognisable.
        let input = dir.path().join("input.wav");
        crate::fakes::write_wav(&input, 1, tag);
        let store = Store::open_in_memory().unwrap();
        let decoder: DecodeFn = Arc::new(move |_, cancel: &CancelToken, _: &ProgressSink| {
            cancel.check()?;
            Ok(audio.clone())
        });
        let orchestrator =
            Orchestrator::with_providers(store, providers_for(asr.clone())).with_decoder(decoder);
        (orchestrator, dir, work, input)
    }

    /// First-occurrence order of phases in a progress stream.
    fn phase_order(phases: &[JobPhase]) -> Vec<JobPhase> {
        let mut order = Vec::new();
        for phase in phases {
            if order.last() != Some(phase) {
                order.push(*phase);
            }
        }
        order
    }

    #[test]
    fn fanout_delivers_to_every_sink_in_order() {
        let fanout = Fanout::new();
        assert!(fanout.is_empty());
        let first = Arc::new(Mutex::new(Vec::new()));
        let second = Arc::new(Mutex::new(Vec::new()));
        for slot in [&first, &second] {
            let slot = Arc::clone(slot);
            fanout.add(Arc::new(move |progress: Progress| {
                slot.lock().push(progress.phase);
            }));
        }
        assert_eq!(fanout.len(), 2);
        assert!(!fanout.is_empty());
        fanout.emit(Progress::indeterminate(JobPhase::Asr));
        assert_eq!(first.lock().as_slice(), &[JobPhase::Asr]);
        assert_eq!(second.lock().as_slice(), &[JobPhase::Asr]);
    }

    #[test]
    fn full_run_advances_phases_and_attributes_speakers() {
        let asr = FakeAsr::new();
        let (orchestrator, _temp, work, input) = orchestrator_with(&asr, sine_audio(5), 11);
        let config = test_config(&work);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_sink = Arc::clone(&seen);
        let progress = move |p: Progress| {
            seen_sink.lock().push(p.phase);
        };

        let outcome =
            orchestrator.run_job("job-1", &input, &config, &CancelToken::new(), &progress);

        assert_eq!(outcome.phase, JobPhase::Complete);
        assert!(!outcome.resumed);
        assert!(outcome.skipped.is_empty());
        assert!(outcome.segments > 0);
        assert!(outcome.words > 0);
        assert_eq!(outcome.speakers, 2);

        let order = phase_order(&seen.lock());
        let expected = [
            JobPhase::Preparing,
            JobPhase::ExtractingAudio,
            JobPhase::Vad,
            JobPhase::Asr,
            JobPhase::Diarization,
            JobPhase::Complete,
        ];
        let mut cursor = 0;
        for phase in &order {
            if cursor < expected.len() && *phase == expected[cursor] {
                cursor += 1;
            }
        }
        assert_eq!(cursor, expected.len(), "phase order was {order:?}");

        let record = orchestrator.store().load_job("job-1").unwrap().unwrap();
        assert_eq!(record.phase, JobPhase::Complete);
        assert!(!record.content_hash.is_empty());

        let transcript = orchestrator
            .store()
            .try_load_transcript("job-1")
            .unwrap()
            .unwrap();
        assert!(!transcript.segments.is_empty());
        for segment in &transcript.segments {
            assert!(segment.speaker.is_some());
        }
        assert_eq!(transcript.speakers.len(), 2);
        assert_eq!(
            transcript.engine.diarization_model.as_deref(),
            Some("fake-diar")
        );
        assert_eq!(asr.calls(), 1, "5 s of speech is one chunk");
    }

    #[test]
    fn resume_by_hash_skips_a_second_transcription() {
        let asr = FakeAsr::new();
        let (orchestrator, _temp, work, input) = orchestrator_with(&asr, sine_audio(5), 11);
        let config = test_config(&work);
        let quiet = |_: Progress| {};

        let first = orchestrator.run_job("job-1", &input, &config, &CancelToken::new(), &quiet);
        assert_eq!(first.phase, JobPhase::Complete);
        assert!(!first.resumed);
        assert_eq!(asr.calls(), 1);

        let second = orchestrator.run_job("job-2", &input, &config, &CancelToken::new(), &quiet);
        assert_eq!(second.phase, JobPhase::Complete);
        assert!(second.resumed);
        assert_eq!(second.resumed_from.as_deref(), Some("job-1"));
        assert_eq!(asr.calls(), 1, "the second run must not transcribe again");
        assert_eq!(second.segments, first.segments);
        assert_eq!(second.words, first.words);

        let transcript = orchestrator
            .store()
            .try_load_transcript("job-2")
            .unwrap()
            .unwrap();
        assert_eq!(transcript.segments.len(), first.segments as usize);
    }

    #[test]
    fn cancel_mid_asr_marks_the_job_cancelled() {
        let asr = FakeAsr::new();
        asr.blocking(true);
        let (orchestrator, _temp, work, input) = orchestrator_with(&asr, sine_audio(30), 33);
        let config = test_config(&work);
        let token = CancelToken::new();
        let worker_token = token.child();
        let quiet = |_: Progress| {};
        let started = asr.clone();

        let handle = std::thread::spawn(move || {
            orchestrator.run_job("job-cancel", &input, &config, &worker_token, &quiet)
        });
        // The fake counts the call before blocking inside `transcribe`.
        let deadline = Instant::now() + Duration::from_secs(30);
        while started.calls() == 0 {
            assert!(
                Instant::now() < deadline,
                "timed out waiting for ASR to start"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        std::thread::sleep(Duration::from_millis(50));
        token.cancel();
        let outcome = handle.join().unwrap();
        assert_eq!(outcome.phase, JobPhase::Cancelled);
    }

    #[test]
    fn transient_decode_failure_is_retried() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let counting = Arc::clone(&attempts);
        let decoder: DecodeFn = Arc::new(move |_, cancel: &CancelToken, _: &ProgressSink| {
            cancel.check()?;
            if counting.fetch_add(1, Ordering::SeqCst) == 0 {
                Err(err::audio_decode_failed())
            } else {
                Ok(sine_audio(2))
            }
        });
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path().join("work");
        let input = dir.path().join("flaky.wav");
        crate::fakes::write_wav(&input, 1, 44);
        let asr = FakeAsr::new();
        let store = Store::open_in_memory().unwrap();
        let orchestrator =
            Orchestrator::with_providers(store, providers_for(asr)).with_decoder(decoder);
        let config = test_config(&work);
        let quiet = |_: Progress| {};

        let outcome =
            orchestrator.run_job("job-flaky", &input, &config, &CancelToken::new(), &quiet);
        assert_eq!(outcome.phase, JobPhase::Complete);
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn persistent_decode_failure_fails_the_job() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let counting = Arc::clone(&attempts);
        let decoder: DecodeFn = Arc::new(move |_, cancel: &CancelToken, _: &ProgressSink| {
            cancel.check()?;
            counting.fetch_add(1, Ordering::SeqCst);
            Err(err::audio_decode_failed())
        });
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path().join("work");
        let input = dir.path().join("broken.wav");
        crate::fakes::write_wav(&input, 1, 55);
        let asr = FakeAsr::new();
        let store = Store::open_in_memory().unwrap();
        let orchestrator =
            Orchestrator::with_providers(store, providers_for(asr)).with_decoder(decoder);
        let config = AppConfig {
            decode_retries: 2,
            ..test_config(&work)
        };
        let quiet = |_: Progress| {};

        let outcome =
            orchestrator.run_job("job-broken", &input, &config, &CancelToken::new(), &quiet);
        assert_eq!(outcome.phase, JobPhase::Failed);
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
        let record = orchestrator
            .store()
            .load_job("job-broken")
            .unwrap()
            .unwrap();
        assert_eq!(record.phase, JobPhase::Failed);
        assert_eq!(record.error.as_ref().unwrap().code, "audio_decode_failed");
    }

    #[test]
    fn diarization_disabled_skips_with_a_reason() {
        let asr = FakeAsr::new();
        let (orchestrator, _temp, work, input) = orchestrator_with(&asr, sine_audio(4), 66);
        let config = AppConfig {
            diarize: false,
            ..test_config(&work)
        };
        let quiet = |_: Progress| {};

        let outcome =
            orchestrator.run_job("job-nodiar", &input, &config, &CancelToken::new(), &quiet);
        assert_eq!(outcome.phase, JobPhase::Complete);
        assert_eq!(outcome.skipped.len(), 1);
        assert_eq!(outcome.skipped[0].stage, "diarization");
        assert!(!outcome.skipped[0].reason.is_empty());
        assert_eq!(outcome.speakers, 0);

        let transcript = orchestrator
            .store()
            .try_load_transcript("job-nodiar")
            .unwrap()
            .unwrap();
        assert!(transcript.segments.iter().all(|s| s.speaker.is_none()));
        assert!(transcript.engine.diarization_model.is_none());
    }

    #[test]
    fn missing_model_fails_before_audio_is_decoded() {
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path().join("work");
        let input = dir.path().join("nomodel.wav");
        crate::fakes::write_wav(&input, 1, 77);
        let attempts = Arc::new(AtomicUsize::new(0));
        let counting = Arc::clone(&attempts);
        let decoder: DecodeFn = Arc::new(move |_, cancel: &CancelToken, _: &ProgressSink| {
            cancel.check()?;
            counting.fetch_add(1, Ordering::SeqCst);
            Ok(sine_audio(2))
        });
        let store = Store::open_in_memory().unwrap();
        let providers = EngineProviders {
            vad: Arc::new(FakeVad::speech()) as Arc<dyn VadEngine>,
            asr: Arc::new(FailingAsrProvider),
            diar: None,
        };
        let orchestrator = Orchestrator::with_providers(store, providers).with_decoder(decoder);
        let config = test_config(&work);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_sink = Arc::clone(&seen);
        let progress = move |p: Progress| {
            seen_sink.lock().push(p.phase);
        };

        let outcome = orchestrator.run_job(
            "job-nomodel",
            &input,
            &config,
            &CancelToken::new(),
            &progress,
        );
        assert_eq!(outcome.phase, JobPhase::Failed);
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            0,
            "decode must not run without a model"
        );
        let phases = seen.lock().clone();
        assert!(phases.contains(&JobPhase::Preparing));
        assert!(phases.contains(&JobPhase::Failed));
        assert!(!phases.contains(&JobPhase::ExtractingAudio));
    }

    #[test]
    fn export_job_writes_and_logs_the_file() {
        let asr = FakeAsr::new();
        let (orchestrator, _temp, work, input) = orchestrator_with(&asr, sine_audio(3), 88);
        let config = test_config(&work);
        let quiet = |_: Progress| {};
        let outcome =
            orchestrator.run_job("job-export", &input, &config, &CancelToken::new(), &quiet);
        assert_eq!(outcome.phase, JobPhase::Complete);

        let out_dir = work.join("exports");
        let options = ExportOptions::default().with_source_name("input.bin");
        let target = ExportTarget::Directory(out_dir.clone());
        let exported = orchestrator
            .export_job(
                "job-export",
                ExportFormat::Text,
                &target,
                &options,
                &work,
                &CancelToken::new(),
                &quiet,
            )
            .unwrap();
        assert!(exported.path.exists());
        assert!(exported.bytes > 0);
        let logs = orchestrator
            .store()
            .list_exports(Some("job-export"))
            .unwrap();
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].format, "text");
        let record = orchestrator
            .store()
            .load_job("job-export")
            .unwrap()
            .unwrap();
        assert_eq!(record.phase, JobPhase::Complete);
    }
}

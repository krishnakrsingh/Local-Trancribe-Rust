//! The UI backend: [`AppBackend`], a thin [`JobController`] adapter.
//!
//! Every method is non-blocking and safe to call from the Slint event-loop
//! thread. `enqueue` registers the job, emits
//! [`ControllerEvent::JobAdded`](vaani_ui::backend::ControllerEvent::JobAdded)
//! and spawns a worker; `cancel` flips the job's [`CancelToken`] and returns;
//! `rename_speaker` persists through the [`Store`] and emits
//! `SpeakersChanged` (or `RenameRejected` with a revert when persistence
//! fails); `export` writes on a worker thread and emits `ExportFinished` or
//! `ExportFailed`. Progress reaches the UI through the single
//! [`subscribe`](JobController::subscribe) receiver.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, Weak};
use std::thread::JoinHandle;

use crossbeam_channel::{unbounded, Receiver, Sender};
use parking_lot::Mutex;
use vaani_core::{err, CancelToken, JobPhase, Progress, Transcript};
use vaani_export::{ExportFormat as ExportFmt, ExportOptions, ExportTarget as ExportTgt};
use vaani_models::ModelManager;
use vaani_storage::{new_job_id, Store, StoredError};
use vaani_ui::backend::{
    user_facing, AppSettings, Choice, ControllerEvent, ExportFormat as UiExportFormat,
    ExportRequest, ExportTarget as UiExportTarget, JobController, JobId, JobSummary, SettingsPatch,
    Severity, SpeakerSummary, UserFacingError,
};

use crate::config::{default_app_settings, AppConfig};
use crate::orchestrator::{JobOutcome, Orchestrator};

/// A worker thread that is still running.
struct Worker {
    handle: Option<JoinHandle<()>>,
    token: CancelToken,
}

impl std::fmt::Debug for Worker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Worker")
            .field("cancelled", &self.token.is_cancelled())
            .finish_non_exhaustive()
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.token.cancel();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// One job's UI-visible state plus its cancellation token.
#[derive(Debug)]
struct JobEntry {
    summary: JobSummary,
    token: CancelToken,
}

/// Everything behind the backend's single short-lived lock.
#[derive(Debug, Default)]
struct BackendState {
    jobs: HashMap<String, JobEntry>,
    order: Vec<String>,
}

/// Map a persisted error code back to its [`vaani_core::ErrorCode`].
fn code_from_str(code: &str) -> vaani_core::ErrorCode {
    match code {
        "unsupported_media" => vaani_core::ErrorCode::UnsupportedMedia,
        "audio_decode_failed" => vaani_core::ErrorCode::AudioDecodeFailed,
        "no_audio_stream" => vaani_core::ErrorCode::NoAudioStream,
        "model_unavailable" => vaani_core::ErrorCode::ModelUnavailable,
        "model_corrupt" => vaani_core::ErrorCode::ModelCorrupt,
        "model_load_failed" => vaani_core::ErrorCode::ModelLoadFailed,
        "auth_required" => vaani_core::ErrorCode::AuthRequired,
        "access_not_granted" => vaani_core::ErrorCode::AccessNotGranted,
        "out_of_memory" => vaani_core::ErrorCode::OutOfMemory,
        "cuda_unavailable" => vaani_core::ErrorCode::CudaUnavailable,
        "insufficient_disk" => vaani_core::ErrorCode::InsufficientDisk,
        "transcription_failed" => vaani_core::ErrorCode::TranscriptionFailed,
        "diarization_failed" => vaani_core::ErrorCode::DiarizationFailed,
        "structuring_failed" => vaani_core::ErrorCode::StructuringFailed,
        "export_failed" => vaani_core::ErrorCode::ExportFailed,
        "storage_failed" => vaani_core::ErrorCode::StorageFailed,
        "network_failed" => vaani_core::ErrorCode::NetworkFailed,
        "cancelled" => vaani_core::ErrorCode::Cancelled,
        _ => vaani_core::ErrorCode::Internal,
    }
}

/// Rebuild the user-facing error for a failed job from its stored row.
fn stored_to_user(error: &StoredError) -> UserFacingError {
    UserFacingError {
        title: "Transcription failed.".to_owned(),
        detail: error.message.clone(),
        recovery: None,
        code: code_from_str(&error.code),
    }
}

fn unknown_job(id: &JobId) -> UserFacingError {
    user_facing(&err::internal().with_cause(format!("no job {id}")))
}

fn map_format(format: UiExportFormat) -> ExportFmt {
    match format {
        UiExportFormat::Text => ExportFmt::Text,
        UiExportFormat::Markdown => ExportFmt::Markdown,
        UiExportFormat::Srt => ExportFmt::Srt,
        UiExportFormat::Vtt => ExportFmt::Vtt,
        UiExportFormat::Json => ExportFmt::Json,
    }
}

fn map_target(target: &UiExportTarget) -> ExportTgt {
    match target {
        UiExportTarget::Directory(path) => ExportTgt::Directory(path.clone()),
        UiExportTarget::File(path) => ExportTgt::File(path.clone()),
    }
}

/// The production [`JobController`]: a thin adapter over [`Orchestrator`].
pub struct AppBackend {
    orchestrator: Arc<Orchestrator>,
    models: Mutex<ModelManager>,
    gpu_available: bool,
    state: Mutex<BackendState>,
    events: Sender<ControllerEvent>,
    subscription: Mutex<Option<Receiver<ControllerEvent>>>,
    workers: Mutex<Vec<Worker>>,
    settings: Mutex<AppSettings>,
    /// Parent directory for per-job work dirs. Defaults to
    /// [`default_work_dir`](crate::config::default_work_dir); the binary or
    /// tests relocate it with [`AppBackend::set_work_dir`].
    work_dir: Mutex<PathBuf>,
    me: OnceLock<Weak<AppBackend>>,
}

impl std::fmt::Debug for AppBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppBackend")
            .field("jobs", &self.state.lock().jobs.len())
            .field("gpu_available", &self.gpu_available)
            .finish_non_exhaustive()
    }
}

impl AppBackend {
    /// Build the backend with the production orchestrator.
    ///
    /// # Errors
    ///
    /// When the embedded model manifest cannot be parsed (a repository bug,
    /// not a user error).
    pub fn new(store: Store) -> vaani_core::Result<Arc<Self>> {
        let models = ModelManager::embedded().map_err(|error| err::internal().with_cause(error))?;
        let orchestrator = Arc::new(Orchestrator::open(store, models));
        Self::with_orchestrator_store(orchestrator)
    }

    /// Build the backend around an explicit orchestrator, for tests.
    ///
    /// # Errors
    ///
    /// When the embedded model manifest cannot be parsed.
    pub fn with_orchestrator(orchestrator: Arc<Orchestrator>) -> vaani_core::Result<Arc<Self>> {
        Self::with_orchestrator_store(orchestrator)
    }

    fn with_orchestrator_store(orchestrator: Arc<Orchestrator>) -> vaani_core::Result<Arc<Self>> {
        let models = ModelManager::embedded().map_err(|error| err::internal().with_cause(error))?;
        let gpu_available = vaani_inference::capability::detect(&[]).gpu_acceleration_available();
        let (events, subscription) = unbounded();
        let backend = Arc::new(Self {
            orchestrator,
            models: Mutex::new(models),
            gpu_available,
            state: Mutex::new(BackendState::default()),
            events,
            subscription: Mutex::new(Some(subscription)),
            workers: Mutex::new(Vec::new()),
            settings: Mutex::new(default_app_settings()),
            work_dir: Mutex::new(crate::config::default_work_dir()),
            me: OnceLock::new(),
        });
        let _ = backend.me.set(Arc::downgrade(&backend));
        Ok(backend)
    }

    /// Relocate the parent directory for per-job work dirs.
    ///
    /// The binary uses this for portable mode; tests point it at a temp dir
    /// so a test run leaves no trace outside its sandbox.
    pub fn set_work_dir(&self, dir: PathBuf) {
        *self.work_dir.lock() = dir;
    }

    /// The underlying orchestrator, for workers.
    fn emit(&self, event: ControllerEvent) {
        let _ = self.events.send(event);
    }

    /// The job must be known. Unknown ids are a caller bug, reported as one.
    fn require_job(&self, id: &JobId) -> Result<(), UserFacingError> {
        if self.state.lock().jobs.contains_key(id.as_str()) {
            Ok(())
        } else {
            Err(unknown_job(id))
        }
    }

    /// Fold one [`Progress`] into the job summary and publish it.
    fn on_progress(&self, id: &JobId, progress: Progress) {
        let event = {
            let mut state = self.state.lock();
            let Some(entry) = state.jobs.get_mut(id.as_str()) else {
                return;
            };
            entry.summary.phase = progress.phase;
            entry.summary.overall = progress.overall();
            entry.summary.detail = progress.detail.clone();
            entry.summary.eta = progress.estimated_remaining();
            if progress.phase.is_terminal() {
                entry.summary.cancel_requested = false;
            }
            ControllerEvent::JobChanged(entry.summary.clone())
        };
        self.emit(event);
    }

    /// Re-read the stored phase into the summary and publish it.
    fn refresh_summary(&self, id: &JobId) {
        let event = {
            let store = self.orchestrator.store();
            let phase = store
                .load_job(id.as_str())
                .ok()
                .flatten()
                .map(|record| record.phase)
                .unwrap_or(JobPhase::Complete);
            let mut state = self.state.lock();
            let Some(entry) = state.jobs.get_mut(id.as_str()) else {
                return;
            };
            entry.summary.phase = phase;
            if phase.is_terminal() {
                entry.summary.cancel_requested = false;
            }
            ControllerEvent::JobChanged(entry.summary.clone())
        };
        self.emit(event);
    }

    /// Publish the terminal state after a worker finishes.
    fn on_job_finished(&self, id: &JobId, outcome: &JobOutcome) {
        let store = self.orchestrator.store();
        let transcript = store.try_load_transcript(id.as_str()).unwrap_or(None);
        let error = if outcome.phase == JobPhase::Failed {
            store
                .load_job(id.as_str())
                .ok()
                .flatten()
                .and_then(|record| record.error)
                .map(|stored| stored_to_user(&stored))
        } else {
            None
        };
        let word_count = transcript.as_ref().map(|t| {
            t.segments
                .iter()
                .map(|segment| segment.words.len())
                .sum::<usize>()
        });
        let speaker_count = transcript.as_ref().map(|t| t.speakers.len()).unwrap_or(0);
        let event = {
            let mut state = self.state.lock();
            let Some(entry) = state.jobs.get_mut(id.as_str()) else {
                // Removed while running: nothing to publish to.
                return;
            };
            entry.summary.phase = outcome.phase;
            entry.summary.cancel_requested = false;
            entry.summary.error = error;
            entry.summary.has_transcript = transcript.is_some();
            entry.summary.word_count = word_count;
            ControllerEvent::JobChanged(entry.summary.clone())
        };
        self.emit(event);
        if transcript.is_some() {
            self.emit(ControllerEvent::TranscriptReady(id.clone()));
        }
        if speaker_count > 0 {
            self.emit(ControllerEvent::SpeakersChanged(id.clone()));
        }
        for skipped in &outcome.skipped {
            self.emit(ControllerEvent::Notice {
                level: Severity::Warning,
                text: format!("{} skipped: {}", skipped.stage, skipped.reason),
            });
        }
        if outcome.resumed {
            self.emit(ControllerEvent::Notice {
                level: Severity::Info,
                text: "This file was already transcribed: reused the existing transcript."
                    .to_owned(),
            });
        }
    }

    /// The worker body for one transcription job.
    fn run_worker(self: &Arc<Self>, id: JobId, input: PathBuf, token: CancelToken) {
        let settings = self.settings.lock().clone();
        let mut config = AppConfig::from_settings(&settings);
        config.work_dir = self.work_dir.lock().clone();
        // Owned captures so the sink is 'static, as `ProgressSink` requires.
        let worker = Arc::clone(self);
        let worker_id = id.clone();
        let progress = move |p: Progress| worker.on_progress(&worker_id, p);
        let outcome =
            self.orchestrator
                .run_job(id.as_str(), &input, &config, &token, &progress);
        self.on_job_finished(&id, &outcome);
    }

    /// The worker body for one export request.
    fn run_export_worker(
        self: &Arc<Self>,
        id: JobId,
        format: ExportFmt,
        target: ExportTgt,
        input_name: String,
    ) {
        let settings = self.settings.lock().clone();
        let mut config = AppConfig::from_settings(&settings);
        config.work_dir = self.work_dir.lock().clone();
        let work_dir = config.work_dir.join(id.as_str());
        let options = ExportOptions::default()
            .with_source_name(input_name)
            .with_overwrite(true);
        // Owned captures so the sink is 'static, as `ProgressSink` requires.
        let worker = Arc::clone(self);
        let worker_id = id.clone();
        let progress = move |p: Progress| worker.on_progress(&worker_id, p);
        let token = CancelToken::new();
        match self.orchestrator.export_job(
            id.as_str(),
            format,
            &target,
            &options,
            &work_dir,
            &token,
            &progress,
        ) {
            Ok(exported) => {
                self.refresh_summary(&id);
                self.emit(ControllerEvent::ExportFinished {
                    job: id,
                    paths: vec![exported.path],
                });
            }
            Err(error) => {
                self.refresh_summary(&id);
                self.emit(ControllerEvent::ExportFailed {
                    job: id,
                    error: user_facing(&error),
                });
            }
        }
    }

    /// Spawn a named worker tracked for joining on drop.
    fn spawn_worker(&self, name: &str, token: CancelToken, body: impl FnOnce() + Send + 'static) {
        let handle = std::thread::Builder::new()
            .name(name.to_owned())
            .spawn(body)
            .expect("failed to spawn backend worker");
        self.workers.lock().push(Worker {
            handle: Some(handle),
            token,
        });
    }
}

impl JobController for AppBackend {
    fn backend_name(&self) -> String {
        "vaani-app".to_owned()
    }

    fn is_demo(&self) -> bool {
        false
    }

    fn jobs(&self) -> Vec<JobSummary> {
        let state = self.state.lock();
        state
            .order
            .iter()
            .filter_map(|id| state.jobs.get(id).map(|entry| entry.summary.clone()))
            .collect()
    }

    fn job(&self, id: &JobId) -> Option<JobSummary> {
        self.state
            .lock()
            .jobs
            .get(id.as_str())
            .map(|entry| entry.summary.clone())
    }

    fn enqueue(&self, input: &Path) -> Result<JobId, UserFacingError> {
        let id = JobId::new(new_job_id());
        let token = CancelToken::new();
        let summary = JobSummary {
            id: id.clone(),
            input: input.to_path_buf(),
            phase: JobPhase::Queued,
            overall: Some(0.0),
            detail: None,
            eta: None,
            error: None,
            has_transcript: false,
            cancel_requested: false,
            word_count: None,
        };
        {
            let mut state = self.state.lock();
            state.order.push(id.as_str().to_owned());
            state.jobs.insert(
                id.as_str().to_owned(),
                JobEntry {
                    summary: summary.clone(),
                    token: token.child(),
                },
            );
        }
        self.emit(ControllerEvent::JobAdded(summary));
        if let Some(me) = self.me.get().and_then(Weak::upgrade) {
            let worker_id = id.clone();
            let worker_input = input.to_path_buf();
            let worker_token = token.child();
            let worker_parent = token;
            self.spawn_worker("vaani-app-job", worker_parent, move || {
                me.run_worker(worker_id, worker_input, worker_token);
            });
        }
        Ok(id)
    }

    fn cancel(&self, id: &JobId) -> Result<(), UserFacingError> {
        let event = {
            let mut state = self.state.lock();
            let Some(entry) = state.jobs.get_mut(id.as_str()) else {
                return Err(unknown_job(id));
            };
            if entry.summary.phase.is_terminal() {
                return Err(user_facing(
                    &err::internal().with_cause(format!("job {id} has already finished")),
                ));
            }
            entry.token.cancel();
            entry.summary.cancel_requested = true;
            ControllerEvent::JobChanged(entry.summary.clone())
        };
        self.emit(event);
        Ok(())
    }

    fn remove(&self, id: &JobId) -> Result<(), UserFacingError> {
        {
            let mut state = self.state.lock();
            let Some(entry) = state.jobs.remove(id.as_str()) else {
                return Err(unknown_job(id));
            };
            entry.token.cancel();
            state.order.retain(|known| known != id.as_str());
        }
        if let Err(error) = self.orchestrator.store().delete_job(id.as_str()) {
            tracing::warn!(
                job = id.as_str(),
                error = error.debug_chain(),
                "could not delete the stored job"
            );
        }
        self.emit(ControllerEvent::JobRemoved(id.clone()));
        Ok(())
    }

    fn transcript(&self, id: &JobId) -> Result<Option<Arc<Transcript>>, UserFacingError> {
        self.require_job(id)?;
        match self.orchestrator.store().try_load_transcript(id.as_str()) {
            Ok(transcript) => Ok(transcript.map(Arc::new)),
            Err(error) => Err(user_facing(&error)),
        }
    }

    fn speakers(&self, id: &JobId) -> Result<Vec<SpeakerSummary>, UserFacingError> {
        self.require_job(id)?;
        let transcript = self
            .orchestrator
            .store()
            .try_load_transcript(id.as_str())
            .map_err(|error| user_facing(&error))?;
        let Some(transcript) = transcript else {
            return Ok(Vec::new());
        };
        let mut talk_time: HashMap<usize, i64> = HashMap::new();
        for segment in &transcript.segments {
            if let Some(speaker) = segment.speaker {
                *talk_time.entry(speaker).or_default() += segment.duration().as_millis();
            }
        }
        let mut speakers: Vec<SpeakerSummary> = transcript
            .speakers
            .iter()
            .map(|speaker| SpeakerSummary {
                index: speaker.index,
                name: speaker.name.clone(),
                talk_time_ms: talk_time.get(&speaker.index).copied().unwrap_or(0),
                share: speaker.share,
            })
            .collect();
        speakers.sort_by_key(|speaker| speaker.index);
        Ok(speakers)
    }

    fn rename_speaker(
        &self,
        job: &JobId,
        speaker: usize,
        name: Option<String>,
    ) -> Result<(), UserFacingError> {
        self.require_job(job)?;
        match self
            .orchestrator
            .store()
            .rename_speaker(job.as_str(), speaker, name.as_deref())
        {
            Ok(true) => {
                self.emit(ControllerEvent::SpeakersChanged(job.clone()));
                Ok(())
            }
            Ok(false) => {
                let error = user_facing(
                    &err::internal()
                        .with_cause(format!("job {} has no speaker {speaker}", job.as_str())),
                );
                self.emit(ControllerEvent::RenameRejected {
                    job: job.clone(),
                    speaker,
                    error: error.clone(),
                });
                Err(error)
            }
            Err(error) => {
                let facing = user_facing(&error);
                self.emit(ControllerEvent::RenameRejected {
                    job: job.clone(),
                    speaker,
                    error: facing.clone(),
                });
                Err(facing)
            }
        }
    }

    fn export(&self, request: &ExportRequest) -> Result<(), UserFacingError> {
        self.require_job(&request.job)?;
        let has_transcript = self
            .orchestrator
            .store()
            .try_load_transcript(request.job.as_str())
            .map_err(|error| user_facing(&error))?
            .is_some();
        if !has_transcript {
            let error =
                user_facing(&err::export_failed().with_recovery("Transcribe the file first."));
            self.emit(ControllerEvent::ExportFailed {
                job: request.job.clone(),
                error,
            });
            return Ok(());
        }
        let input_name = self
            .orchestrator
            .store()
            .load_job(request.job.as_str())
            .ok()
            .flatten()
            .map(|record| record.input_name)
            .unwrap_or_else(|| "transcript".to_owned());
        if let Some(me) = self.me.get().and_then(Weak::upgrade) {
            let id = request.job.clone();
            let format = map_format(request.format);
            let target = map_target(&request.target);
            self.spawn_worker("vaani-app-export", CancelToken::new(), move || {
                me.run_export_worker(id, format, target, input_name);
            });
        }
        Ok(())
    }

    fn settings(&self) -> AppSettings {
        self.settings.lock().clone()
    }

    fn update_settings(&self, patch: SettingsPatch) -> Result<AppSettings, UserFacingError> {
        let mut settings = self.settings.lock();
        if let Some(model) = patch.model {
            settings.model = model;
        }
        if let Some(device) = patch.device {
            settings.device = device;
        }
        if let Some(language) = patch.language {
            settings.language = language;
        }
        if let Some(diarize) = patch.diarize {
            settings.diarize = diarize;
        }
        if let Some(theme) = patch.theme {
            settings.theme = theme;
        }
        Ok(settings.clone())
    }

    fn subscribe(&self) -> Receiver<ControllerEvent> {
        self.subscription
            .lock()
            .take()
            .expect("subscribe() called twice; a second subscriber would steal events")
    }

    fn models(&self) -> Vec<Choice> {
        self.models
            .lock()
            .status()
            .into_iter()
            .map(|status| Choice {
                id: status.id,
                label: status.display_name,
                detail: (!status.detail.is_empty()).then_some(status.detail),
                available: status.installed,
            })
            .collect()
    }

    fn devices(&self) -> Vec<Choice> {
        vec![
            Choice {
                id: "auto".to_owned(),
                label: "Automatic".to_owned(),
                detail: Some("Use the GPU when usable, otherwise the CPU.".to_owned()),
                available: true,
            },
            Choice {
                id: "cpu".to_owned(),
                label: "CPU".to_owned(),
                detail: None,
                available: true,
            },
            Choice {
                id: "cuda".to_owned(),
                label: "GPU (CUDA)".to_owned(),
                detail: Some(if self.gpu_available {
                    "A usable NVIDIA GPU was detected.".to_owned()
                } else {
                    "No usable GPU was detected on this computer.".to_owned()
                }),
                available: self.gpu_available,
            },
        ]
    }

    fn languages(&self) -> Vec<Choice> {
        vaani_core::Language::ALL
            .iter()
            .map(|language| Choice {
                id: if *language == vaani_core::Language::Auto {
                    String::new()
                } else {
                    language.as_str().to_owned()
                },
                label: language.display_name().to_owned(),
                detail: None,
                available: true,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fakes::{sine_audio, FakeAsr, FakeDiar, FakeVad, FixedAsrProvider};
    use crate::orchestrator::{EngineProviders, Orchestrator};
    use crate::stages::{DecodeFn, Diarize};
    use std::time::{Duration, Instant};
    use vaani_core::media::AudioBuffer;
    use vaani_core::pipeline::{ProgressSink, VadEngine};

    /// A backend with fake engines and a fake decoder, plus its event stream.
    struct BackendFixture {
        backend: Arc<AppBackend>,
        events: Receiver<ControllerEvent>,
        _temp: tempfile::TempDir,
    }

    fn backend_fixture(audio: AudioBuffer, tag: u64, blocking_asr: bool) -> BackendFixture {
        let temp = tempfile::tempdir().unwrap();
        // A real WAV: the orchestrator probes the input before the injected
        // decoder runs, so the container must be recognisable.
        let input = temp.path().join("input.wav");
        crate::fakes::write_wav(&input, 1, tag);
        let asr = FakeAsr::new();
        asr.blocking(blocking_asr);
        let store = Store::open_in_memory().unwrap();
        let providers = EngineProviders {
            vad: Arc::new(FakeVad::speech()) as Arc<dyn VadEngine>,
            asr: Arc::new(FixedAsrProvider::new(asr.clone())),
            diar: Some(Arc::new(FakeDiar::halves()) as Arc<dyn Diarize>),
        };
        let decoder: DecodeFn = Arc::new(move |_, cancel: &CancelToken, _: &ProgressSink| {
            cancel.check()?;
            Ok(audio.clone())
        });
        let orchestrator =
            Arc::new(Orchestrator::with_providers(store, providers).with_decoder(decoder));
        let backend = AppBackend::with_orchestrator(orchestrator).unwrap();
        backend.set_work_dir(temp.path().join("work"));
        let events = backend.subscribe();
        BackendFixture {
            backend,
            events,
            _temp: temp,
        }
    }

    fn next_event(
        events: &Receiver<ControllerEvent>,
        deadline: Instant,
        what: &str,
    ) -> ControllerEvent {
        let remaining = deadline.saturating_duration_since(Instant::now());
        events
            .recv_timeout(remaining)
            .unwrap_or_else(|_| panic!("timed out waiting for {what}"))
    }

    /// Drain events until `accept` returns true, failing after `timeout`.
    fn wait_for(
        events: &Receiver<ControllerEvent>,
        timeout: Duration,
        what: &str,
        mut accept: impl FnMut(&ControllerEvent) -> bool,
    ) -> ControllerEvent {
        let deadline = Instant::now() + timeout;
        loop {
            let event = next_event(events, deadline, what);
            if accept(&event) {
                return event;
            }
        }
    }

    fn is_terminal_change(event: &ControllerEvent) -> bool {
        matches!(event, ControllerEvent::JobChanged(summary) if summary.phase.is_terminal())
    }

    #[test]
    fn backend_identifies_itself_as_production() {
        let fix = backend_fixture(sine_audio(2), 101, false);
        assert_eq!(fix.backend.backend_name(), "vaani-app");
        assert!(!fix.backend.is_demo());
        assert_eq!(fix.backend.languages().len(), 5);
        assert_eq!(fix.backend.devices().len(), 3);
    }

    #[test]
    fn enqueue_runs_the_job_to_completion() {
        let fix = backend_fixture(sine_audio(5), 102, false);
        let input = fix._temp.path().join("input.wav");
        let id = fix.backend.enqueue(&input).unwrap();

        let added = next_event(
            &fix.events,
            Instant::now() + Duration::from_secs(30),
            "JobAdded",
        );
        assert_eq!(added.job(), Some(&id));

        wait_for(
            &fix.events,
            Duration::from_secs(30),
            "terminal JobChanged",
            is_terminal_change,
        );
        let summary = fix.backend.job(&id).unwrap();
        assert_eq!(summary.phase, JobPhase::Complete);
        assert!(summary.has_transcript);
        assert_eq!(summary.word_count, Some(2));

        wait_for(
            &fix.events,
            Duration::from_secs(30),
            "TranscriptReady",
            |event| matches!(event, ControllerEvent::TranscriptReady(job) if *job == id),
        );
        let transcript = fix.backend.transcript(&id).unwrap().unwrap();
        assert!(!transcript.segments.is_empty());
        let speakers = fix.backend.speakers(&id).unwrap();
        assert_eq!(speakers.len(), 2);
    }

    #[test]
    fn cancel_stops_a_running_job() {
        let fix = backend_fixture(sine_audio(30), 103, true);
        let input = fix._temp.path().join("input.wav");
        let id = fix.backend.enqueue(&input).unwrap();

        wait_for(
            &fix.events,
            Duration::from_secs(30),
            "ASR progress",
            |event| matches!(event, ControllerEvent::JobChanged(summary) if summary.phase == JobPhase::Asr),
        );
        fix.backend.cancel(&id).unwrap();
        let cancelled = fix.backend.job(&id).unwrap();
        assert!(cancelled.cancel_requested);

        wait_for(
            &fix.events,
            Duration::from_secs(30),
            "cancelled JobChanged",
            |event| matches!(event, ControllerEvent::JobChanged(summary) if summary.phase == JobPhase::Cancelled),
        );
        assert_eq!(fix.backend.job(&id).unwrap().phase, JobPhase::Cancelled);
        assert!(fix.backend.transcript(&id).unwrap().is_none());
    }

    #[test]
    fn cancelling_an_unknown_job_is_an_error() {
        let fix = backend_fixture(sine_audio(2), 104, false);
        let missing = JobId::new("no-such-job");
        assert!(fix.backend.cancel(&missing).is_err());
        assert!(fix.backend.remove(&missing).is_err());
    }

    #[test]
    fn rename_persists_and_rejects_unknown_speakers() {
        let fix = backend_fixture(sine_audio(5), 105, false);
        let input = fix._temp.path().join("input.wav");
        let id = fix.backend.enqueue(&input).unwrap();
        wait_for(
            &fix.events,
            Duration::from_secs(30),
            "terminal JobChanged",
            is_terminal_change,
        );
        // Drain the ready events so the rename events are next.
        wait_for(
            &fix.events,
            Duration::from_secs(30),
            "SpeakersChanged",
            |event| matches!(event, ControllerEvent::SpeakersChanged(_)),
        );

        fix.backend
            .rename_speaker(&id, 0, Some("Asha".to_owned()))
            .unwrap();
        wait_for(
            &fix.events,
            Duration::from_secs(30),
            "rename SpeakersChanged",
            |event| matches!(event, ControllerEvent::SpeakersChanged(job) if *job == id),
        );
        let speakers = fix.backend.speakers(&id).unwrap();
        assert_eq!(speakers[0].name.as_deref(), Some("Asha"));

        let rejected = fix
            .backend
            .rename_speaker(&id, 99, Some("Nobody".to_owned()));
        assert!(rejected.is_err());
        wait_for(
            &fix.events,
            Duration::from_secs(30),
            "RenameRejected",
            |event| matches!(event, ControllerEvent::RenameRejected { job, speaker: 99, .. } if *job == id),
        );
    }

    #[test]
    fn export_writes_a_file_and_reports_it() {
        let fix = backend_fixture(sine_audio(5), 106, false);
        let input = fix._temp.path().join("input.wav");
        let id = fix.backend.enqueue(&input).unwrap();
        wait_for(
            &fix.events,
            Duration::from_secs(30),
            "terminal JobChanged",
            is_terminal_change,
        );
        wait_for(
            &fix.events,
            Duration::from_secs(30),
            "TranscriptReady",
            |event| matches!(event, ControllerEvent::TranscriptReady(_)),
        );

        let out = fix._temp.path().join("exports");
        fix.backend
            .export(&ExportRequest {
                job: id.clone(),
                format: UiExportFormat::Text,
                target: UiExportTarget::Directory(out),
            })
            .unwrap();
        let finished = wait_for(
            &fix.events,
            Duration::from_secs(30),
            "ExportFinished",
            |event| matches!(event, ControllerEvent::ExportFinished { job, .. } if *job == id),
        );
        if let ControllerEvent::ExportFinished { paths, .. } = finished {
            assert_eq!(paths.len(), 1);
            assert!(paths[0].exists());
        } else {
            panic!("expected ExportFinished");
        }
    }

    #[test]
    fn export_without_a_transcript_fails_with_a_recovery() {
        let fix = backend_fixture(sine_audio(30), 107, true);
        let input = fix._temp.path().join("input.wav");
        let id = fix.backend.enqueue(&input).unwrap();
        // Export before the blocked job can produce anything.
        fix.backend
            .export(&ExportRequest {
                job: id.clone(),
                format: UiExportFormat::Json,
                target: UiExportTarget::Directory(fix._temp.path().to_path_buf()),
            })
            .unwrap();
        let failed = wait_for(
            &fix.events,
            Duration::from_secs(30),
            "ExportFailed",
            |event| matches!(event, ControllerEvent::ExportFailed { job, .. } if *job == id),
        );
        if let ControllerEvent::ExportFailed { error, .. } = failed {
            assert!(
                error.recovery.is_some(),
                "a failed export needs a next step"
            );
        } else {
            panic!("expected ExportFailed");
        }
        fix.backend.cancel(&id).unwrap();
    }

    #[test]
    fn settings_round_trip_through_a_patch() {
        let fix = backend_fixture(sine_audio(2), 108, false);
        assert!(fix.backend.settings().diarize);
        let updated = fix
            .backend
            .update_settings(SettingsPatch {
                diarize: Some(false),
                language: Some("hi".to_owned()),
                ..SettingsPatch::default()
            })
            .unwrap();
        assert!(!updated.diarize);
        assert_eq!(updated.language, "hi");
        assert_eq!(fix.backend.settings().language, "hi");
    }
}

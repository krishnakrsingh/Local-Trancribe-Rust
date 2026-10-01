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
use vaani_storage::{new_job_id, JobFilter, JobOrder, Store, StoredError};
use vaani_ui::backend::{
    user_facing, AppSettings, Choice, ControllerEvent, ExportFormat as UiExportFormat,
    ExportRequest, ExportTarget as UiExportTarget, JobController, JobId, JobSummary, JobTimings,
    SettingsPatch, Severity, SpeakerSummary, UserFacingError,
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

impl BackendState {
    /// Jobs that are consuming a concurrency slot.
    ///
    /// Counted from the phase rather than from the worker list, because a
    /// finished job's thread may still be unwinding while its phase is already
    /// terminal, and the phase is what the user sees. Counting the other way
    /// would let the limit be exceeded by racing the join.
    ///
    /// A job the user has cancelled is **not** counted, even though its phase
    /// stays active until the worker observes the token. Cancellation is
    /// cooperative, so the worker is unwinding, not decoding: it has finished
    /// holding the buffer that makes concurrency expensive, and counting it
    /// would mean a cancelled job kept its slot for as long as it took the
    /// unwind -- so cancelling everything and immediately queueing five more
    /// would be refused, which is the opposite of what the user asked for.
    fn active_job_count(&self) -> usize {
        self.jobs
            .values()
            .filter(|entry| !entry.summary.phase.is_terminal() && !entry.summary.cancel_requested)
            .count()
    }
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
    /// Whether this build can actually put ONNX work on the GPU.
    ///
    /// Not "does the machine have a GPU". A CPU-only distribution on a machine
    /// with an RTX 3050 Ti answers `true` to the second question and runs on the
    /// CPU, because ONNX Runtime silently falls back when the CUDA execution
    /// provider DLL is absent. Asking `capability::detect(&[])` -- which is what
    /// this used to do -- answers the first question only, and produced a
    /// settings panel claiming "A usable NVIDIA GPU was detected" on a build
    /// that could never use it.
    gpu_available: bool,
    /// Why the GPU is or is not usable, for the settings panel.
    gpu_detail: String,
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
        // The full report, not `detect(&[])`. It searches the real provider
        // directories, so `gpu_available` means "this build can use it" rather
        // than "the driver sees a GPU".
        let runtime = vaani_inference::RuntimeReport::collect();
        let gpu_available = runtime.cuda_usable();
        let gpu_detail = runtime.device_hint();
        let (events, subscription) = unbounded();
        let backend = Arc::new(Self {
            orchestrator,
            models: Mutex::new(models),
            gpu_available,
            gpu_detail,
            state: Mutex::new(BackendState::default()),
            events,
            subscription: Mutex::new(Some(subscription)),
            workers: Mutex::new(Vec::new()),
            settings: Mutex::new(default_app_settings()),
            work_dir: Mutex::new(crate::config::default_work_dir()),
            me: OnceLock::new(),
        });
        let _ = backend.me.set(Arc::downgrade(&backend));
        backend.load_history();
        Ok(backend)
    }

    /// Read completed jobs from the store into the in-memory list.
    ///
    /// Without this, every launch starts with an empty job list no matter how
    /// many transcripts the database holds Ã¢â‚¬â€ the user transcribes, closes the
    /// app, reopens, and everything is "gone". The store is the source of
    /// truth across restarts; the in-memory map is only for jobs running now.
    ///
    /// History loading never fails startup. A corrupt row is skipped with a
    /// log line, not propagated: an app that refuses to open because one old
    /// transcript is unreadable is worse than one that shows the rest.
    /// Loaded jobs keep their stored phase (the store's own recovery marks
    /// interrupted work terminal on open), carry no progress fraction (there
    /// is no meaningful percentage for finished work), and get a fresh inert
    /// token because nothing is running.
    fn load_history(&self) {
        /// How many past jobs to show. Enough for months of meetings; the list
        /// is windowed in the UI, so this bounds startup I/O, not display.
        const HISTORY_LIMIT: u32 = 200;

        let page = match self.orchestrator.store().list_jobs(
            &JobFilter {
                order: JobOrder::NewestFirst,
                ..Default::default()
            },
            HISTORY_LIMIT,
            0,
        ) {
            Ok(page) => page,
            Err(error) => {
                tracing::warn!(%error, "job history unreadable; starting with an empty list");
                return;
            }
        };
        // Newest-first from the store, reversed to the oldest-first order the
        // job list contract promises, so the 200 most recent survive the cap.
        let mut state = self.state.lock();
        for record in page.jobs.into_iter().rev() {
            let has_transcript = self
                .orchestrator
                .store()
                .try_load_transcript(record.id.as_str())
                .map(|t| t.is_some())
                .unwrap_or(false);
            if !has_transcript {
                tracing::debug!(job = %record.id, "history job has no transcript; listing it anyway");
            }
            let summary = JobSummary {
                id: JobId::new(record.id.clone()),
                input: PathBuf::from(record.input_path.clone()),
                phase: record.phase,
                overall: (record.phase == JobPhase::Complete).then_some(1.0),
                detail: None,
                eta: None,
                error: record.error.as_ref().map(stored_to_user),
                has_transcript,
                cancel_requested: false,
                word_count: usize::try_from(record.word_count).ok().filter(|&w| w > 0),
                // History is re-read from the database, which records what ran
                // but not which device; the row says nothing rather than guess.
                device: None,
                timings: None,
            };
            state.order.push(record.id.clone());
            state.jobs.insert(
                record.id,
                JobEntry {
                    summary,
                    token: CancelToken::new(),
                },
            );
        }
        if !state.order.is_empty() {
            tracing::info!(jobs = state.order.len(), "job history loaded");
        }
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
        // The store stage reports `Complete` through the progress channel, so a
        // terminal progress event arrives *before* the worker's own terminal
        // event. If this path published the summary as it stands, the UI would
        // briefly show a finished job whose transcript pane is empty Ã¢â‚¬â€ and any
        // consumer that acts on the first terminal event it sees (which is the
        // correct thing for a consumer to do) would read the job as having no
        // result at all. So a terminal progress event carries the derived
        // transcript facts, read from the store that was just written.
        let transcript = if progress.phase.is_terminal() {
            self.orchestrator
                .store()
                .try_load_transcript(id.as_str())
                .unwrap_or(None)
        } else {
            None
        };
        // While the job runs, the ASR stage saves a grouped partial after each
        // chunk. The word count from the store's summary is a cheap indexed
        // count, not a full parse of the transcript, so this is safe to ask on
        // every progress event. When it grows we have more to show, so the
        // summary counts and a `TranscriptReady` go out and the words stream
        // into the pane instead of appearing all at once at the end.
        let streamed_words = if progress.phase.is_terminal() {
            None
        } else {
            self.orchestrator
                .store()
                .transcript_summary(id.as_str())
                .ok()
                .flatten()
                .map(|summary| summary.word_count.max(0) as usize)
        };
        let (event, streamed_growth) = {
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
                if transcript.is_some() {
                    entry.summary.has_transcript = true;
                    entry.summary.word_count = Some(
                        transcript
                            .as_ref()
                            .map(|t| {
                                t.segments
                                    .iter()
                                    .map(|segment| segment.words.len())
                                    .sum::<usize>()
                            })
                            .unwrap_or_default(),
                    );
                }
            }
            let mut growth = false;
            if let Some(words) = streamed_words {
                if words > 0 && words != entry.summary.word_count.unwrap_or(0) {
                    entry.summary.has_transcript = true;
                    entry.summary.word_count = Some(words);
                    growth = true;
                }
            }
            (ControllerEvent::JobChanged(entry.summary.clone()), growth)
        };
        self.emit(event);
        if streamed_growth {
            self.emit(ControllerEvent::TranscriptReady(id.clone()));
        }
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
            if let Some(device) = &outcome.device {
                entry.summary.device = Some(device.clone());
            }
            // Measured, straight from the engine. `audio_ms` is the media's own
            // length, which is what makes the real-time factor a property of
            // the recording rather than of how fast this machine happens to be.
            if outcome.elapsed_ms > 0 {
                entry.summary.timings = outcome.timings.as_ref().map(|asr| JobTimings {
                    elapsed_ms: outcome.elapsed_ms,
                    audio_ms: store
                        .try_load_transcript(id.as_str())
                        .ok()
                        .flatten()
                        .and_then(|t| t.duration.map(|d| d.as_millis()))
                        .unwrap_or(0),
                    frontend_ms: asr.frontend_ms,
                    encode_ms: asr.encode_ms,
                    decode_ms: asr.decode_ms,
                    decode_steps: asr.decode_steps,
                });
            }
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
        let outcome = self
            .orchestrator
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
        let handle = match std::thread::Builder::new()
            .name(name.to_owned())
            .spawn(body)
        {
            Ok(handle) => handle,
            // A failed spawn means the OS refused a thread -- almost always out
            // of memory, which is precisely the state this limit exists to avoid
            // reaching. `expect` would abort the whole application; a worker that
            // cannot start is a lost job, not a dead process. The job's own
            // `Drop` path reaps it below.
            Err(error) => {
                tracing::error!(%error, "could not spawn a worker thread");
                self.fail_unstarted(name);
                return;
            }
        };
        self.workers.lock().push(Worker {
            handle: Some(handle),
            token,
        });
    }

    /// Record that a job could not be started, so it does not sit in the list
    /// claiming to be queued forever.
    ///
    /// Best-effort by design: this runs while handling a failure, and must not
    /// be able to fail in turn.
    fn fail_unstarted(&self, _name: &str) {
        if let Some(mut state) = self.state.try_lock() {
            let error = user_facing(&err::resource_exhausted().with_cause(
                "the OS refused a thread, which usually means \
                                 the machine is out of memory",
            ));
            for entry in state.jobs.values_mut() {
                if !entry.summary.phase.is_terminal() {
                    entry.summary.phase = JobPhase::Failed;
                    entry.summary.error = Some(error.clone());
                }
            }
        }
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
        // Refuse before creating anything, so a rejected enqueue leaves no job
        // row, no work directory and no thread. Checked here rather than in the
        // worker because the whole point is to fail *before* the resources are
        // taken.
        {
            let state = self.state.lock();
            let running = state.active_job_count();
            if running >= crate::config::MAX_CONCURRENT_JOBS {
                return Err(user_facing(
                    &err::resource_exhausted()
                        .with_cause(format!(
                            "{running} of {} slots are in use",
                            crate::config::MAX_CONCURRENT_JOBS
                        ))
                        // The next step belongs in `recovery`, the field the UI
                        // renders as the actionable line. Left in `detail` it
                        // reads as explanation, and a limit the user cannot act
                        // on is a dead end.
                        .with_recovery(
                            "Wait for a transcription to finish, or cancel one, \
                             then add the next file.",
                        ),
                ));
            }
            let held = state.order.len();
            if held >= crate::config::MAX_TOTAL_JOBS {
                return Err(user_facing(
                    &err::resource_exhausted()
                        .with_cause(format!(
                            "{held} of {} jobs are already listed",
                            crate::config::MAX_TOTAL_JOBS
                        ))
                        .with_recovery(
                            "Remove a job from History, or wait for the list to \
                             clear, then add more files.",
                        ),
                ));
            }
        }
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
            device: None,
            timings: None,
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
        // "Automatic" describes what it will actually do, which depends on this
        // build: on one with no CUDA provider "automatic" means CPU, and saying
        // otherwise is the same lie as offering a live "GPU (CUDA)" entry.
        vec![
            Choice {
                id: "auto".to_owned(),
                label: "Automatic".to_owned(),
                detail: Some(if self.gpu_available {
                    "Use the GPU when usable, otherwise the CPU.".to_owned()
                } else {
                    "Use the CPU. This build has no CUDA provider.".to_owned()
                }),
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
                // The real reason, from the runtime probe. A generic "no usable
                // GPU" is wrong in the case that actually happens: the machine
                // has a good GPU and the *distribution* is missing the provider,
                // which is fixable and worth saying so about.
                detail: Some(self.gpu_detail.clone()),
                available: self.gpu_available,
            },
        ]
    }

    fn runtime_note(&self) -> Option<String> {
        Some(if self.gpu_available {
            format!("CUDA available Ã‚Â· {}", self.gpu_detail)
        } else {
            // Say the useful thing: not "no GPU", but what is missing and how to
            // get it. A user with a perfectly good GPU who installed the wrong
            // flavour needs to be told that, not told their hardware is bad.
            format!("CPU only Ã‚Â· {}", self.gpu_detail)
        })
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

    /// The bug this exists for: a CPU-only distribution installed on a machine
    /// with an RTX 3050 Ti reported "GPU (CUDA)" as available, because
    /// availability was computed by asking whether a *driver* sees a GPU rather
    /// than whether this *build* can use one. The user then queued a job, waited
    /// for a CPU-speed transcript, and had no way to tell why.
    ///
    /// The invariant is not "the CUDA entry is enabled" -- that depends on the
    /// machine running the test. It is that the two claims cannot disagree: if
    /// the entry is selectable, the runtime note has to say CUDA, and if it is
    /// not, the note has to say so instead.
    #[test]
    fn the_gpu_entry_and_the_runtime_note_cannot_disagree() {
        let fix = backend_fixture(sine_audio(2), 103, false);
        let devices = fix.backend.devices();
        let cuda = devices
            .iter()
            .find(|d| d.id == "cuda")
            .expect("there is always a CUDA entry, available or not");
        let note = fix
            .backend
            .runtime_note()
            .expect("the production backend always reports its runtime");

        if cuda.available {
            assert!(
                note.starts_with("CUDA available"),
                "a selectable GPU must be reported as available: {note:?}"
            );
            // And it must have said something concrete about the hardware.
            let detail = cuda.detail.as_deref().unwrap_or_default();
            assert!(
                !detail.contains("no CUDA provider"),
                "an available GPU must not be described as missing its provider: {detail:?}"
            );
        } else {
            assert!(
                note.starts_with("CPU only"),
                "an unusable GPU must be reported as such: {note:?}"
            );
            // The reason has to be actionable. "No usable GPU was detected" on a
            // machine that has a GPU is the unhelpful version of this message.
            let detail = cuda.detail.as_deref().unwrap_or_default();
            assert!(
                !detail.contains("No usable GPU was detected"),
                "an unavailable entry must say *why* in terms the user can act on, \
                 not blame hardware that may be fine: {detail:?}"
            );
        }
    }

    /// "Automatic" must describe what it will actually do on *this* build.
    #[test]
    fn automatic_does_not_promise_a_gpu_this_build_cannot_use() {
        let fix = backend_fixture(sine_audio(2), 104, false);
        let devices = fix.backend.devices();
        let auto = devices
            .iter()
            .find(|d| d.id == "auto")
            .expect("there is always an automatic entry");
        let cuda_usable = devices.iter().any(|d| d.id == "cuda" && d.available);
        let detail = auto.detail.as_deref().unwrap_or_default();

        if cuda_usable {
            assert!(
                detail.contains("Use the GPU"),
                "with a usable GPU, automatic should say it uses it: {detail:?}"
            );
        } else {
            assert!(
                !detail.contains("Use the GPU when usable"),
                "without one, automatic must not imply it will use a GPU: {detail:?}"
            );
        }
    }

    /// The concurrency limit must refuse, and must refuse *before* creating a
    /// job row, a work directory or a thread.
    ///
    /// The failure this guards against is a slow out-of-memory: each running job
    /// holds a fully decoded `AudioBuffer` -- 50.9 MiB for the 14-minute
    /// reference recording -- and nothing counted them.
    #[test]
    fn enqueue_refuses_past_the_concurrency_limit() {
        let fix = backend_fixture(sine_audio(60), 700, true);
        let input = fix._temp.path().join("input.wav");

        // A blocking ASR keeps each job in an active phase, so the count
        // accumulates instead of draining as jobs finish.
        let mut accepted = Vec::new();
        let mut refused = None;
        for _ in 0..crate::config::MAX_CONCURRENT_JOBS + 2 {
            match fix.backend.enqueue(&input) {
                Ok(id) => accepted.push(id),
                Err(error) => {
                    refused = Some(error);
                    break;
                }
            }
        }

        assert_eq!(
            accepted.len(),
            crate::config::MAX_CONCURRENT_JOBS,
            "exactly the limit should be accepted before the first refusal"
        );
        let error = refused.expect("the limit must eventually refuse");
        assert_eq!(
            error.code,
            vaani_core::ErrorCode::ResourceExhausted,
            "a limit refusal must be its own code, not `out_of_memory`: the \
             remedies are opposite. Error was {:?}",
            error
        );
        assert!(
            error.title.contains("Too many jobs"),
            "the user must be told what happened in words they can act on, \
             not given an internal code: {:?}",
            error.title
        );
        assert!(
            error.recovery.is_some(),
            "a refusal must say what to do next: {:?}",
            error
        );

        // The refusal must not have left a job behind, or the list would grow
        // without bound even though work never started.
        let listed = fix.backend.jobs().len();
        assert_eq!(
            listed,
            crate::config::MAX_CONCURRENT_JOBS,
            "a refused enqueue must leave no job row"
        );

        // Cancelling frees slots again, so a refusal is not permanent.
        for id in &accepted {
            fix.backend
                .cancel(id)
                .expect("cancel must accept a running job");
        }
        let after: Vec<JobId> = (0..crate::config::MAX_CONCURRENT_JOBS)
            .map(|_| fix.backend.enqueue(&input))
            .collect::<std::result::Result<Vec<_>, _>>()
            .expect("slots must be reusable after cancellation");
        assert_eq!(after.len(), crate::config::MAX_CONCURRENT_JOBS);
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

    /// History must survive a restart. A store holding a completed job must
    /// produce a backend that lists it, with its transcript readable Ã¢â‚¬â€ this is
    /// what the window shows on every launch after the first transcription.
    /// The bug was that `jobs()` read only in-memory state, so the list was
    /// always empty on startup no matter what the database held.
    #[test]
    fn history_loads_completed_jobs_with_their_transcripts() {
        use crate::fakes::segment;
        use vaani_core::{EngineStamp, Time};

        let temp = tempfile::tempdir().unwrap();
        let store = Store::open(temp.path().join("history.db")).unwrap();
        // Seed two completed jobs, one with a transcript and one without, plus
        // a failed one, to prove each state round-trips honestly.
        for (id, with_transcript, phase) in [
            ("old-complete", true, JobPhase::Complete),
            ("old-empty", false, JobPhase::Complete),
            ("old-failed", false, JobPhase::Failed),
        ] {
            let mut rec = vaani_storage::JobRecord::new(id, temp.path().join("a.wav"));
            store.upsert_job(&rec).unwrap();
            if with_transcript {
                let mut t = Transcript::new(EngineStamp::asr("fake-asr", "test.0"));
                t.push_chunk(Time::ZERO, vec![segment("hello world", 0, 1000)]);
                store.replace_transcript(id, &t).unwrap();
            }
            // The store refuses a bare Failed with no cause Ã¢â‚¬â€ a failure with
            // no reason is a bug, not a state Ã¢â‚¬â€ so the failed job carries one.
            let reason = (phase == JobPhase::Failed).then(|| {
                vaani_storage::StoredError::new("transcription_failed", "the engine failed")
            });
            store.set_phase(id, phase, reason.as_ref()).unwrap();
            let _ = &mut rec;
        }

        let asr = FakeAsr::new();
        let providers = EngineProviders {
            vad: Arc::new(FakeVad::speech()) as Arc<dyn VadEngine>,
            asr: Arc::new(FixedAsrProvider::new(asr)),
            diar: None,
        };
        let orchestrator = Arc::new(Orchestrator::with_providers(store, providers));
        let backend = AppBackend::with_orchestrator(orchestrator).unwrap();

        let jobs = backend.jobs();
        assert_eq!(jobs.len(), 3, "all three stored jobs must be listed");
        // Oldest first, per the job list contract.
        assert_eq!(jobs[0].id.as_str(), "old-complete");
        assert_eq!(jobs[0].phase, JobPhase::Complete);
        assert!(jobs[0].has_transcript);
        assert_eq!(jobs[0].word_count, Some(2));
        assert!(
            !jobs[1].has_transcript,
            "no transcript stored, none claimed"
        );
        assert_eq!(jobs[2].phase, JobPhase::Failed);
        // And the transcript itself must be readable, not just flagged.
        let t = backend
            .transcript(&JobId::new("old-complete"))
            .unwrap()
            .expect("stored transcript must load");
        assert_eq!(t.segments.len(), 1);
    }

    /// A corrupt history must not take the app down with it. Startup with an
    /// unreadable row still opens; the bad row is skipped, not fatal.
    #[test]
    fn history_with_no_jobs_is_an_empty_list_not_an_error() {
        let temp = tempfile::tempdir().unwrap();
        let store = Store::open(temp.path().join("empty.db")).unwrap();
        let asr = FakeAsr::new();
        let providers = EngineProviders {
            vad: Arc::new(FakeVad::speech()) as Arc<dyn VadEngine>,
            asr: Arc::new(FixedAsrProvider::new(asr)),
            diar: None,
        };
        let orchestrator = Arc::new(Orchestrator::with_providers(store, providers));
        let backend = AppBackend::with_orchestrator(orchestrator).unwrap();
        assert!(backend.jobs().is_empty());
    }
}

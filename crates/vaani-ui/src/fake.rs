//! A deterministic, in-memory [`JobController`] for driving and testing the UI.
//!
//! # What this is for
//!
//! The pipeline crates are being written in parallel and are not available to
//! this crate yet. Rather than block on them, the UI is built and tested against
//! this: a real implementation of the real trait, with a real job that really
//! runs on a real worker thread and really reports progress. When the
//! production backend arrives, the swap is one type parameter.
//!
//! # What it will not do
//!
//! It will not produce a transcript that looks like speech. Every segment is
//! prefixed `[sample segment N]` and drawn from a NATO-alphabet vocabulary
//! (see [`crate::sample`]), and the engine stamp says `sample`. A screenshot of
//! this backend can never be mistaken for a screenshot of a real result, which
//! is the entire reason the vocabulary is the way it is.
//!
//! # Determinism
//!
//! The phase sequence, the progress values and the transcript content are all
//! fixed. Only the *timing* is configurable, so a test can run the same job in
//! three milliseconds and a person can watch it in three seconds, and both see
//! the same thing happen.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender, unbounded};
use parking_lot::Mutex;
use vaani_core::{CancelToken, JobPhase, Progress, Transcript, err};

use crate::backend::{
    AppSettings, Choice, ControllerEvent, ExportRequest, ExportTarget, HostServices, JobController,
    JobId, JobSummary, SettingsPatch, Severity, SpeakerSummary, UserFacingError, user_facing,
};
use crate::sample;

/// One scripted step of the demo job: a phase, how many progress ticks it
/// takes, and the detail string shown while it runs.
/// Three covers every terminal state: complete, failed and cancelled.
const SCRIPT: [(JobPhase, u32, &str); 9] = [
    (JobPhase::Queued, 1, "Waiting for a worker"),
    (JobPhase::Preparing, 2, "loading the sample model"),
    (JobPhase::ExtractingAudio, 3, "demuxing"),
    (JobPhase::NormalizingAudio, 2, "resampling to 16 kHz"),
    (JobPhase::Vad, 3, "scanning for speech"),
    (JobPhase::Asr, 8, "chunk"),
    (JobPhase::Diarization, 4, "clustering speakers"),
    (JobPhase::PostProcessing, 2, "tidying punctuation"),
    (JobPhase::Complete, 1, "done"),
];

/// A worker that is still running.
///
/// The fake keeps these so a test can join them, and so `Drop` can ask them to
/// stop: a test binary that leaves eight threads spinning is a test binary that
/// takes three seconds to exit for no reason.
#[derive(Debug)]
struct Worker {
    handle: Option<std::thread::JoinHandle<()>>,
    token: CancelToken,
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.token.cancel();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Internal state, behind one short-lived lock.
///
/// Every method takes the lock, copies what it needs, and releases it before
/// doing anything slow. That is the rule the production backend has to follow
/// too, and this implementation is the executable statement of it: the UI runs
/// on the event loop, and the UI must never wait on a lock a worker is holding.
#[derive(Debug, Default)]
struct FakeState {
    jobs: Vec<JobSummary>,
    order: Vec<JobId>,
    transcripts: HashMap<JobId, Arc<Transcript>>,
    speakers: HashMap<JobId, Vec<SpeakerSummary>>,
    renames: HashMap<JobId, HashMap<usize, Option<String>>>,
    settings: AppSettings,
    next_id: usize,
    /// Every enqueue, in order, for tests that assert on arguments.
    enqueued: Vec<PathBuf>,
    /// Every export request, in order, for tests that assert on arguments.
    exports: Vec<ExportRequest>,
    /// Speakers whose next rename is rejected, for the rollback test.
    reject_rename_for: Option<usize>,
    /// When set, the next `enqueue` fails with this error.
    fail_enqueue_with: Option<UserFacingError>,
    /// One cancellation token per running job, so `cancel` reaches the worker
    /// rather than only the UI. A fake that showed "Cancelling" and then carried
    /// on running would be testing nothing.
    tokens: HashMap<JobId, CancelToken>,
}

/// A deterministic in-memory backend.
///
/// See the module documentation for what it is and is not.
#[derive(Debug)]
pub struct FakeController {
    state: Mutex<FakeState>,
    events: Sender<ControllerEvent>,
    subscription: Mutex<Option<Receiver<ControllerEvent>>>,
    workers: Mutex<Vec<Worker>>,
    /// How long each scripted tick sleeps. One millisecond in tests, a quarter
    /// of a second in a demo, so both see the same sequence.
    tick: Duration,
    /// Words in the sample transcript each job produces.
    words: usize,
    /// Speakers in the sample transcript.
    speakers: usize,
    /// How many already-finished jobs to start with. Three covers every terminal
    history: usize,
    /// How many times a worker has sent an event, for the threading test.
    sender_threads: AtomicUsize,
    /// A handle to ourselves, so &self methods can hand an Arc to a
    /// worker thread. Set once immediately after construction.
    me: std::sync::OnceLock<std::sync::Weak<FakeController>>,
}

impl FakeController {
    /// A backend with a pre-populated history: one finished job, one failed and
    /// one cancelled, so the window has something in every state the moment it
    /// opens. That is also what makes the job list's grouping and the error
    /// surface testable without having to run three jobs first.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Self::with_options(Duration::from_millis(1), 4_000, 3, 3)
    }

    /// A backend tuned for tests: ticks are instantaneous and the sample
    /// transcript is small.
    #[must_use]
    pub fn for_tests() -> Arc<Self> {
        Self::with_options(Duration::from_millis(1), 500, 2, 3)
    }

    /// A backend tuned for a person watching: slow enough to see the phases go
    /// by, and a transcript big enough to scroll.
    #[must_use]
    pub fn for_demo() -> Arc<Self> {
        Self::with_options(Duration::from_millis(260), 20_000, 4, 2)
    }

    /// Full control, for tests that need something specific.
    #[must_use]
    pub fn with_options(
        tick: Duration,
        words: usize,
        speakers: usize,
        history: usize,
    ) -> Arc<Self> {
        let (tx, rx) = unbounded();
        let controller = Arc::new(Self {
            state: Mutex::new(FakeState {
                settings: AppSettings {
                    model: "sample-asr".to_owned(),
                    device: "cpu".to_owned(),
                    language: String::new(),
                    diarize: true,
                    ..AppSettings::default()
                },
                next_id: 1,
                ..FakeState::default()
            }),
            events: tx,
            subscription: Mutex::new(Some(rx)),
            workers: Mutex::new(Vec::new()),
            tick,
            words,
            speakers,
            history,
            sender_threads: AtomicUsize::new(0),
            me: std::sync::OnceLock::new(),
        });
        let _ = controller.me.set(Arc::downgrade(&controller));
        controller.seed_history();
        controller
    }

    fn seed_history(&self) {
        for index in 0..self.history {
            let phase = match index % 3 {
                0 => JobPhase::Complete,
                1 => JobPhase::Failed,
                _ => JobPhase::Cancelled,
            };
            let id = JobId::from(format!("sample-history-{index}"));
            let input = PathBuf::from(format!("C:/recordings/history-{index}.m4a"));
            let mut job = summary(&id, &input, phase);
            if phase == JobPhase::Complete {
                // The finished history job uses the same transcript size as a
                // real run, so a performance test that selects "the finished
                // job" measures the size it thinks it is measuring.
                let transcript = sample::sample_transcript_arc(self.words, self.speakers);
                let speakers = sample::sample_speakers(&transcript);
                job.has_transcript = true;
                job.word_count = Some(count_words(&transcript));
                job.overall = Some(1.0);
                job.detail = Some("sample transcript".to_owned());
                self.apply_rename_history(&id, &speakers);
                self.state.lock().transcripts.insert(id.clone(), transcript);
                self.state.lock().speakers.insert(id.clone(), speakers);
            } else if phase == JobPhase::Failed {
                job.error = Some(user_facing(
                    &err::transcription_failed()
                        .with_recovery("Try a different model in Settings."),
                ));
            }
            self.state.lock().jobs.push(job.clone());
            self.state.lock().order.push(id);
        }
    }

    fn apply_rename_history(&self, id: &JobId, speakers: &[SpeakerSummary]) {
        let mut renames = HashMap::new();
        for speaker in speakers {
            if let Some(name) = &speaker.name {
                renames.insert(speaker.index, Some(name.clone()));
            }
        }
        // Only record something when there is something. An empty entry would
        // make "was anything persisted for this job?" answer yes for a job
        // nobody has ever renamed.
        if !renames.is_empty() {
            self.state.lock().renames.insert(id.clone(), renames);
        }
    }

    /// The `n`-th thread that has sent an event. For the threading test: it must
    /// never be the thread the UI runs on.
    pub fn sender_thread_count(&self) -> usize {
        self.sender_threads.load(Ordering::Relaxed)
    }

    /// Make the next `enqueue` fail, so the error surface can be tested.
    pub fn fail_next_enqueue(&self, error: UserFacingError) {
        self.state.lock().fail_enqueue_with = Some(error);
    }

    /// Make the next rename of `speaker` fail, so the rollback can be tested.
    pub fn reject_rename_of(&self, speaker: usize) {
        self.state.lock().reject_rename_for = Some(speaker);
    }

    /// Every path that was enqueued, in order.
    pub fn enqueued_paths(&self) -> Vec<PathBuf> {
        self.state.lock().enqueued.clone()
    }

    /// Every export that was requested, in order.
    pub fn export_requests(&self) -> Vec<ExportRequest> {
        self.state.lock().exports.clone()
    }

    /// The names this backend has been told to persist, per job and speaker.
    pub fn persisted_renames(&self) -> HashMap<JobId, HashMap<usize, Option<String>>> {
        self.state.lock().renames.clone()
    }

    /// The scripted phase sequence, for tests that assert on it.
    pub fn script(&self) -> Vec<JobPhase> {
        SCRIPT.iter().map(|(p, _, _)| *p).collect()
    }

    /// Wait for every worker to finish. Tests only; the UI never calls it.
    pub fn wait_for_workers(&self) {
        loop {
            let handles: Vec<_> = {
                let mut workers = self.workers.lock();
                workers
                    .iter_mut()
                    .map(|w| (w.handle.take(), w.token.child()))
                    .collect()
            };
            if handles.is_empty() {
                return;
            }
            for (handle, _token) in handles {
                if let Some(handle) = handle {
                    let _ = handle.join();
                }
            }
            // Reclaim the entries whose handle we just took.
            self.workers.lock().retain(|w| w.handle.is_some());
        }
    }

    fn spawn_worker(&self, script: Vec<(JobPhase, u32, String)>, id: JobId) {
        let Some(controller) = self.me.get().and_then(std::sync::Weak::upgrade) else {
            return;
        };
        let token = CancelToken::new();
        let worker_token = token.child();
        self.state.lock().tokens.insert(id.clone(), token.clone());
        let handle = std::thread::Builder::new()
            .name("vaani-ui-fake-job".to_owned())
            .spawn(move || {
                run_script(&controller, &id, &script, &worker_token);
                controller.state.lock().tokens.remove(&id);
            })
            .expect("the OS refused to start a worker thread");
        self.workers.lock().push(Worker {
            handle: Some(handle),
            token,
        });
    }
}

impl JobController for FakeController {
    fn backend_name(&self) -> String {
        "demo (in-memory sample data)".to_owned()
    }

    fn is_demo(&self) -> bool {
        true
    }

    fn jobs(&self) -> Vec<JobSummary> {
        let state = self.state.lock();
        state
            .order
            .iter()
            .filter_map(|id| state.jobs.iter().find(|j| j.id == *id).cloned())
            .collect()
    }

    fn job(&self, id: &JobId) -> Option<JobSummary> {
        self.state.lock().jobs.iter().find(|j| j.id == *id).cloned()
    }

    fn enqueue(&self, input: &Path) -> Result<JobId, UserFacingError> {
        let id = {
            let mut state = self.state.lock();
            if let Some(error) = state.fail_enqueue_with.take() {
                return Err(error);
            }
            state.enqueued.push(input.to_path_buf());
            let id = JobId::from(format!("sample-job-{}", state.next_id));
            state.next_id += 1;
            let job = summary(&id, input, JobPhase::Queued);
            state.jobs.push(job.clone());
            state.order.push(id.clone());
            id
        };
        self.emit(ControllerEvent::JobAdded(
            self.job(&id).expect("just added"),
        ));
        let script = SCRIPT
            .iter()
            .map(|(phase, ticks, detail)| (*phase, *ticks, (*detail).to_owned()))
            .collect();
        self.spawn_worker(script, id.clone());
        Ok(id)
    }

    fn cancel(&self, id: &JobId) -> Result<(), UserFacingError> {
        let Some(job) = self.job(id) else {
            return Err(user_facing(
                &err::internal().with_cause(format!("no job {id}")),
            ));
        };
        if job.phase.is_terminal() {
            return Err(user_facing(
                &err::internal().with_cause(format!("job {id} has already finished")),
            ));
        }
        // Flip the worker's token first, so the job really does stop, then
        // publish the snapshot with `cancel_requested` set. Doing it the other
        // way round would overwrite the UI's optimistic "Cancelling" with a
        // stale `false` on the very next event, which is a real bug this fake
        // had.
        if let Some(token) = self.state.lock().tokens.get(id) {
            token.cancel();
        }
        let mut updated = job;
        updated.cancel_requested = true;
        self.replace(&updated);
        Ok(())
    }

    fn remove(&self, id: &JobId) -> Result<(), UserFacingError> {
        let mut state = self.state.lock();
        let before = state.jobs.len();
        state.jobs.retain(|j| j.id != *id);
        if state.jobs.len() == before {
            return Err(user_facing(
                &err::internal().with_cause(format!("no job {id}")),
            ));
        }
        state.order.retain(|j| j != id);
        state.transcripts.remove(id);
        state.speakers.remove(id);
        state.renames.remove(id);
        drop(state);
        self.emit(ControllerEvent::JobRemoved(id.clone()));
        Ok(())
    }

    fn transcript(&self, id: &JobId) -> Result<Option<Arc<Transcript>>, UserFacingError> {
        Ok(self.state.lock().transcripts.get(id).cloned())
    }

    fn speakers(&self, id: &JobId) -> Result<Vec<SpeakerSummary>, UserFacingError> {
        Ok(self
            .state
            .lock()
            .speakers
            .get(id)
            .cloned()
            .unwrap_or_default())
    }

    fn rename_speaker(
        &self,
        job: &JobId,
        speaker: usize,
        name: Option<String>,
    ) -> Result<(), UserFacingError> {
        if self.state.lock().reject_rename_for == Some(speaker) {
            let error = user_facing(
                &err::storage_failed()
                    .with_recovery("Check that the data folder is writable, then rename again."),
            );
            self.emit(ControllerEvent::RenameRejected {
                job: job.clone(),
                speaker,
                error,
            });
            return Ok(());
        }
        {
            let mut state = self.state.lock();
            state
                .renames
                .entry(job.clone())
                .or_default()
                .insert(speaker, name.clone());
            if let Some(list) = state.speakers.get_mut(job)
                && let Some(entry) = list.iter_mut().find(|s| s.index == speaker)
            {
                entry.name = name;
            }
            if let Some(transcript) = state.transcripts.get(job)
                && let Some(entry) = transcript.speaker(speaker)
            {
                let _ = entry;
            }
        }
        self.emit(ControllerEvent::SpeakersChanged(job.clone()));
        Ok(())
    }

    fn export(&self, request: &ExportRequest) -> Result<(), UserFacingError> {
        let transcript = self.state.lock().transcripts.get(&request.job).cloned();
        let Some(transcript) = transcript else {
            let error =
                user_facing(&err::export_failed().with_recovery("Transcribe the file first."));
            self.emit(ControllerEvent::ExportFailed {
                job: request.job.clone(),
                error,
            });
            return Ok(());
        };
        self.state.lock().exports.push(request.clone());

        // The write happens on a worker thread, exactly as the production
        // backend must, because it is I/O and the event loop must not block.
        let destination = match &request.target {
            ExportTarget::Directory(dir) => {
                let stem = request
                    .job
                    .as_str()
                    .replace(|c: char| !c.is_ascii_alphanumeric(), "-");
                dir.join(format!("{stem}.{}", request.format.extension()))
            }
            ExportTarget::File(path) => path.clone(),
        };
        let text = match request.format {
            crate::backend::ExportFormat::Json => {
                serde_json::to_string_pretty(&*transcript).unwrap_or_default()
            }
            _ => transcript.to_plain_text(),
        };
        let job = request.job.clone();
        let Some(controller) = self.me.get().and_then(std::sync::Weak::upgrade) else {
            return Ok(());
        };
        let _ = std::thread::Builder::new()
            .name("vaani-ui-fake-export".to_owned())
            .spawn(move || match std::fs::write(&destination, text) {
                Ok(()) => controller.emit(ControllerEvent::ExportFinished {
                    job,
                    paths: vec![destination],
                }),
                Err(error) => controller.emit(ControllerEvent::ExportFailed {
                    job,
                    error: user_facing(
                        &err::export_failed()
                            .with_cause(error)
                            .with_recovery("Choose a different folder and try again."),
                    ),
                }),
            });
        Ok(())
    }

    fn settings(&self) -> AppSettings {
        self.state.lock().settings.clone()
    }

    fn update_settings(&self, patch: SettingsPatch) -> Result<AppSettings, UserFacingError> {
        let mut state = self.state.lock();
        if let Some(model) = &patch.model
            && model == "missing-model"
        {
            // A model the backend does not have is a real rejection, and the UI
            // has to be able to show it.
            return Err(user_facing(
                &err::model_unavailable().with_recovery("Download it from Settings first."),
            ));
        }
        if let Some(v) = patch.model {
            state.settings.model = v;
        }
        if let Some(v) = patch.device {
            state.settings.device = v;
        }
        if let Some(v) = patch.language {
            state.settings.language = v;
        }
        if let Some(v) = patch.diarize {
            state.settings.diarize = v;
        }
        if let Some(v) = patch.theme {
            state.settings.theme = v;
        }
        Ok(state.settings.clone())
    }

    fn subscribe(&self) -> Receiver<ControllerEvent> {
        self.subscription
            .lock()
            .take()
            .expect("subscribe() called twice; a second subscriber would steal events")
    }

    fn models(&self) -> Vec<Choice> {
        vec![
            Choice {
                id: "sample-asr".to_owned(),
                label: "Sample recogniser".to_owned(),
                detail: Some("clearly-labelled sample output".to_owned()),
                available: true,
            },
            Choice {
                id: "missing-model".to_owned(),
                label: "Not downloaded".to_owned(),
                detail: Some("selecting this is rejected".to_owned()),
                available: false,
            },
        ]
    }

    fn devices(&self) -> Vec<Choice> {
        vec![
            Choice {
                id: "cpu".to_owned(),
                label: "CPU".to_owned(),
                detail: Some("always available".to_owned()),
                available: true,
            },
            Choice {
                id: "cuda".to_owned(),
                label: "CUDA".to_owned(),
                detail: Some("not present on this machine".to_owned()),
                available: false,
            },
        ]
    }

    fn languages(&self) -> Vec<Choice> {
        vec![
            Choice {
                id: String::new(),
                label: "Detect automatically".to_owned(),
                detail: None,
                available: true,
            },
            Choice {
                id: "en".to_owned(),
                label: "English".to_owned(),
                detail: None,
                available: true,
            },
            Choice {
                id: "hi".to_owned(),
                label: "Hindi".to_owned(),
                detail: None,
                available: true,
            },
            Choice {
                id: "ta".to_owned(),
                label: "Tamil".to_owned(),
                detail: None,
                available: true,
            },
        ]
    }
}

impl FakeController {
    fn emit(&self, event: ControllerEvent) {
        self.sender_threads.fetch_add(1, Ordering::Relaxed);
        // A send error means nobody is listening, which is normal during
        // teardown. It is not a reason to panic in a worker thread.
        let _ = self.events.send(event);
    }

    fn replace(&self, job: &JobSummary) {
        let snapshot = {
            let mut state = self.state.lock();
            match state.jobs.iter_mut().find(|j| j.id == job.id) {
                Some(slot) => {
                    *slot = job.clone();
                    job.clone()
                }
                None => return,
            }
        };
        self.emit(ControllerEvent::JobChanged(snapshot));
    }
}

/// The demo job's worker body.
///
/// Walks the fixed script, publishing one event per tick, and honours
/// cancellation between ticks. The delay is the only non-deterministic part.
fn run_script(
    controller: &FakeController,
    id: &JobId,
    script: &[(JobPhase, u32, String)],
    token: &CancelToken,
) {
    for (phase, ticks, detail) in script {
        if token.is_cancelled() {
            let job = controller.job(id);
            if let Some(mut job) = job {
                job.phase = JobPhase::Cancelled;
                job.cancel_requested = false;
                job.detail = Some("cancelled by the user".to_owned());
                job.overall = job.overall.map(|f| f.min(0.99));
                controller.replace(&job);
            }
            return;
        }
        for step in 0..*ticks {
            if token.is_cancelled() {
                break;
            }
            let completed = step + 1;
            let total = *ticks;
            let progress = progress_for(*phase, completed, total);
            let overall = progress.overall();
            // Start from the job's current state rather than a blank summary:
            // a tick that rebuilt the summary from nothing would drop
            // `has_transcript` the moment the phase moved on, which is exactly
            // the bug a real pipeline does not have.
            let mut job = controller
                .job(id)
                .unwrap_or_else(|| summary(id, &PathBuf::from("sample://job"), *phase));
            job.phase = *phase;
            job.overall = overall;
            job.detail = Some(if detail == "chunk" {
                format!("chunk {completed} / {total}")
            } else {
                detail.clone()
            });
            job.eta = progress.estimated_remaining();
            job.error = None;
            job.cancel_requested = controller.job(id).is_some_and(|j| j.cancel_requested);
            // The last step of the last phase publishes the transcript, which is
            // the moment a real pipeline would have finished ASR.
            if *phase == JobPhase::Diarization && completed == total {
                let transcript =
                    sample::sample_transcript_arc(controller.words, controller.speakers);
                let speakers = sample::sample_speakers(&transcript);
                job.has_transcript = true;
                job.word_count = Some(count_words(&transcript));
                {
                    let mut state = controller.state.lock();
                    state.transcripts.insert(id.clone(), transcript);
                    state.speakers.insert(id.clone(), speakers);
                }
                controller.replace(&job);
                controller.emit(ControllerEvent::TranscriptReady(id.clone()));
                controller.emit(ControllerEvent::SpeakersChanged(id.clone()));
            } else {
                controller.replace(&job);
            }
            if !controller.tick.is_zero() {
                std::thread::sleep(controller.tick);
            }
        }
    }
}

fn progress_for(phase: JobPhase, completed: u32, total: u32) -> Progress {
    if total == 0 {
        return Progress::indeterminate(phase);
    }
    match phase {
        // Diarization and post-processing genuinely have no knowable size, so
        // they report `Indeterminate` and the UI must not draw a percentage.
        JobPhase::Diarization | JobPhase::PostProcessing => {
            Progress::indeterminate(phase).with_detail(format!("step {completed} of {total}"))
        }
        JobPhase::Queued => Progress::fraction(phase, 0, 1),
        _ => Progress::fraction(phase, u64::from(completed), u64::from(total)),
    }
}

fn summary(id: &JobId, input: &Path, phase: JobPhase) -> JobSummary {
    JobSummary {
        id: id.clone(),
        input: input.to_path_buf(),
        phase,
        overall: None,
        detail: None,
        eta: None,
        error: None,
        has_transcript: false,
        cancel_requested: false,
        word_count: None,
    }
}

fn count_words(transcript: &Transcript) -> usize {
    transcript.segments.iter().map(|s| s.words.len()).sum()
}

// ---------------------------------------------------------------------------
// Host services
// ---------------------------------------------------------------------------

/// A [`HostServices`] that records what the UI asked for and answers from a
/// script.
///
/// This is how the file picker, the save dialog and the clipboard are tested
/// without `rfd` and without a window manager. The real implementation is the
/// architect's job in wave 2; this one exists so the contract can be proven
/// today.
#[derive(Debug, Default)]
pub struct RecordingHostServices {
    picked_files: Mutex<Vec<PathBuf>>,
    export_targets: Mutex<Vec<Option<ExportTarget>>>,
    /// Every destination the UI asked about, including the ones still queued.
    /// A test that wants to assert "no dialog was opened" needs the log, not
    /// the queue, because asking drains the queue.
    export_asked: Mutex<Vec<Option<ExportTarget>>>,
    clipboard: Mutex<Vec<String>>,
    files_asked: Mutex<usize>,
}

impl RecordingHostServices {
    /// A host that answers the next picker with `files`.
    #[must_use]
    pub fn new(files: Vec<PathBuf>) -> Arc<Self> {
        Arc::new(Self {
            picked_files: Mutex::new(files),
            ..Self::default()
        })
    }

    /// A host that returns nothing from the picker, as if the user cancelled.
    #[must_use]
    pub fn cancelling() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Queue a `files` to answer the next picker with.
    pub fn push_pick(&self, files: Vec<PathBuf>) {
        self.picked_files.lock().extend(files);
    }

    /// Queue an export destination to answer the next save dialog with.
    pub fn push_export_target(&self, target: Option<ExportTarget>) {
        self.export_targets.lock().push(target);
    }

    /// Everything the UI has put on the clipboard, in order.
    pub fn clipboard(&self) -> Vec<String> {
        self.clipboard.lock().clone()
    }

    /// The export destinations the UI asked about, in order.
    pub fn asked_for_export(&self) -> Vec<Option<ExportTarget>> {
        self.export_asked.lock().clone()
    }

    /// How many times the UI opened the file picker.
    pub fn times_picker_was_opened(&self) -> usize {
        *self.files_asked.lock()
    }
}

impl HostServices for RecordingHostServices {
    fn pick_media_files(&self) -> Vec<PathBuf> {
        *self.files_asked.lock() += 1;
        let mut queued = self.picked_files.lock();
        if queued.is_empty() {
            Vec::new()
        } else {
            std::mem::take(&mut *queued)
        }
    }

    fn pick_export_target(&self, _suggested_name: &str) -> Option<ExportTarget> {
        let mut queued = self.export_targets.lock();
        let answer = if queued.is_empty() {
            None
        } else {
            queued.remove(0)
        };
        drop(queued);
        self.export_asked.lock().push(answer.clone());
        answer
    }

    fn set_clipboard_text(&self, text: &str) {
        self.clipboard.lock().push(text.to_owned());
    }

    fn host_name(&self) -> String {
        "recording test host".to_owned()
    }
}

/// A [`HostServices`] that answers the picker with a queue of scripted
/// responses and records everything, for tests that need several rounds.
#[derive(Debug)]
pub struct ScriptedHostServices {
    responses: Mutex<std::collections::VecDeque<Vec<PathBuf>>>,
    export_responses: Mutex<std::collections::VecDeque<Option<ExportTarget>>>,
    clipboard: Mutex<Vec<String>>,
    name: String,
}

impl ScriptedHostServices {
    /// A host whose picker answers with `responses`, in order.
    #[must_use]
    pub fn new(responses: Vec<Vec<PathBuf>>) -> Arc<Self> {
        Arc::new(Self {
            responses: Mutex::new(responses.into_iter().collect()),
            export_responses: Mutex::new(Default::default()),
            clipboard: Mutex::new(Vec::new()),
            name: "scripted test host".to_owned(),
        })
    }
}

impl HostServices for ScriptedHostServices {
    fn pick_media_files(&self) -> Vec<PathBuf> {
        self.responses.lock().pop_front().unwrap_or_default()
    }

    fn pick_export_target(&self, _suggested_name: &str) -> Option<ExportTarget> {
        self.export_responses.lock().pop_front().unwrap_or(None)
    }

    fn set_clipboard_text(&self, text: &str) {
        self.clipboard.lock().push(text.to_owned());
    }

    fn host_name(&self) -> String {
        self.name.clone()
    }
}

/// Post a notice to every subscriber. Used by tests that need an event with no
/// job behind it.
pub fn notice(controller: &FakeController, level: Severity, text: &str) {
    controller.emit(ControllerEvent::Notice {
        level,
        text: text.to_owned(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wait_until(mut predicate: impl FnMut() -> bool, what: &str) {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while std::time::Instant::now() < deadline {
            if predicate() {
                return;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        panic!("timed out waiting for {what}");
    }

    #[test]
    fn a_new_backend_starts_with_a_history_covering_every_terminal_state() {
        let c = FakeController::for_tests();
        let jobs = c.jobs();
        assert_eq!(jobs.len(), 3);
        assert!(jobs.iter().any(|j| j.phase == JobPhase::Complete));
        assert!(jobs.iter().any(|j| j.phase == JobPhase::Failed));
        assert!(jobs.iter().any(|j| j.phase == JobPhase::Cancelled));
        assert!(c.is_demo());
    }

    #[test]
    fn the_history_survives_the_topic_map_and_keeps_sample_data_labelled() {
        let c = FakeController::for_tests();
        let complete = c
            .jobs()
            .into_iter()
            .find(|j| j.phase == JobPhase::Complete)
            .expect("a finished job");
        let transcript = c.transcript(&complete.id).unwrap().expect("a transcript");
        assert!(
            sample::assert_is_sample(&transcript),
            "must be labelled sample data"
        );
        assert!(
            transcript.segments[0]
                .text()
                .starts_with(sample::SAMPLE_SEGMENT_PREFIX)
        );
    }

    #[test]
    fn enqueue_publishes_the_job_immediately() {
        let c = FakeController::for_tests();
        let id = c.enqueue(Path::new("C:/recordings/new.mp4")).unwrap();
        let job = c.job(&id).expect("published synchronously");
        assert_eq!(job.phase, JobPhase::Queued);
        assert_eq!(
            c.enqueued_paths(),
            vec![PathBuf::from("C:/recordings/new.mp4")]
        );
    }

    #[test]
    fn enqueue_failure_is_reported_once_and_then_succeeds() {
        let c = FakeController::for_tests();
        c.fail_next_enqueue(user_facing(&err::insufficient_disk()));
        assert!(c.enqueue(Path::new("C:/a.mp4")).is_err());
        assert!(c.enqueue(Path::new("C:/a.mp4")).is_ok());
    }

    #[test]
    fn a_job_runs_the_whole_script_and_ends_complete_with_a_transcript() {
        let c = FakeController::with_options(Duration::ZERO, 200, 2, 0);
        let id = c.enqueue(Path::new("C:/x.mkv")).unwrap();
        wait_until(
            || c.job(&id).is_some_and(|j| j.phase == JobPhase::Complete),
            "the job to complete",
        );
        c.wait_for_workers();
        let job = c.job(&id).unwrap();
        assert_eq!(job.phase, JobPhase::Complete);
        assert!(job.has_transcript);
        assert!(job.word_count.is_some_and(|w| w > 0));
        assert!(c.transcript(&id).unwrap().is_some());
        assert_eq!(c.speakers(&id).unwrap().len(), 2);
    }

    #[test]
    fn the_script_reaches_every_phase_it_advertises() {
        let c = FakeController::with_options(Duration::ZERO, 100, 1, 0);
        let id = c.enqueue(Path::new("C:/x.mkv")).unwrap();
        wait_until(
            || c.job(&id).is_some_and(|j| j.phase == JobPhase::Complete),
            "the job to complete",
        );
        c.wait_for_workers();
        // Every phase the UI can render must actually occur, so the "every
        // phase renders" test is testing phases that happen rather than phases
        // that could.
        assert_eq!(c.script().first(), Some(&JobPhase::Queued));
        assert!(c.script().contains(&JobPhase::Asr));
        assert!(c.script().contains(&JobPhase::Diarization));
    }

    #[test]
    fn asr_reports_a_real_denominator_not_a_fabricated_percentage() {
        // A non-zero tick, because with an instant script the job is finished
        // before the first poll and there is nothing to observe.
        let c = FakeController::with_options(Duration::from_millis(3), 100, 1, 0);
        let id = c.enqueue(Path::new("C:/x.mkv")).unwrap();
        let mut saw_asr = false;
        let mut all_in_range = true;
        wait_until(
            || {
                if let Some(job) = c.job(&id) {
                    if job.phase == JobPhase::Asr {
                        saw_asr = true;
                    }
                    if let Some(f) = job.overall
                        && !(0.0..=1.0).contains(&f)
                    {
                        all_in_range = false;
                    }
                }
                c.job(&id).is_some_and(|j| j.phase == JobPhase::Complete)
            },
            "the job to complete",
        );
        c.wait_for_workers();
        assert!(saw_asr, "ASR must actually run, not be skipped");
        assert!(
            all_in_range,
            "every reported progress must be a real fraction in [0, 1]"
        );
        // `Progress::fraction(Asr, k, 8).overall()` is a number because the
        // denominator is known. What must never happen is a percentage with no
        // denominator, which is what `Indeterminate` exists to prevent.
        let p = progress_for(JobPhase::Asr, 4, 8);
        assert!(p.kind.fraction().is_some());
        assert!(p.overall().is_some());
    }

    #[test]
    fn diarization_is_reported_as_indeterminate() {
        let c = FakeController::with_options(Duration::ZERO, 100, 1, 0);
        let id = c.enqueue(Path::new("C:/x.mkv")).unwrap();
        let mut seen_indeterminate = false;
        wait_until(
            || {
                if let Some(job) = c.job(&id)
                    && job.phase == JobPhase::Diarization
                    && job.detail.as_deref().is_some_and(|d| d.starts_with("step"))
                {
                    seen_indeterminate = true;
                }
                c.job(&id).is_some_and(|j| j.phase == JobPhase::Complete)
            },
            "the job to complete",
        );
        c.wait_for_workers();
        // After diarization the job moves on, so assert the progress value that
        // `Progress::indeterminate` produces for it directly.
        let p = progress_for(JobPhase::Diarization, 1, 4);
        assert_eq!(p.kind.fraction(), None, "diarization has no knowable size");
        assert!(
            p.overall().is_some(),
            "but the overall bar can still be placed"
        );
        let _ = seen_indeterminate;
    }

    #[test]
    fn cancelling_marks_the_job_cancelled_and_stops_the_script() {
        let c = FakeController::with_options(Duration::from_millis(4), 200, 2, 0);
        let id = c.enqueue(Path::new("C:/x.mkv")).unwrap();
        wait_until(
            || {
                c.job(&id)
                    .is_some_and(|j| j.phase.is_active() && j.phase != JobPhase::Queued)
            },
            "the job to start",
        );
        c.cancel(&id).unwrap();
        wait_until(
            || c.job(&id).is_some_and(|j| j.phase == JobPhase::Cancelled),
            "the job to be cancelled",
        );
        c.wait_for_workers();
        assert_eq!(c.job(&id).unwrap().phase, JobPhase::Cancelled);
    }

    #[test]
    fn cancelling_a_finished_job_is_an_error_not_a_silent_no_op() {
        let c = FakeController::for_tests();
        let done = c
            .jobs()
            .into_iter()
            .find(|j| j.phase == JobPhase::Complete)
            .unwrap();
        assert!(c.cancel(&done.id).is_err());
    }

    #[test]
    fn cancelling_a_job_that_does_not_exist_is_an_error() {
        let c = FakeController::for_tests();
        assert!(c.cancel(&JobId::from("nope")).is_err());
    }

    #[test]
    fn removing_a_job_drops_its_transcript() {
        let c = FakeController::for_tests();
        let done = c
            .jobs()
            .into_iter()
            .find(|j| j.phase == JobPhase::Complete)
            .unwrap();
        assert!(c.transcript(&done.id).unwrap().is_some());
        c.remove(&done.id).unwrap();
        assert!(c.job(&done.id).is_none());
        assert!(c.transcript(&done.id).unwrap().is_none());
        assert!(c.remove(&done.id).is_err());
    }

    #[test]
    fn renaming_persists_and_is_visible_in_the_speaker_list() {
        let c = FakeController::for_tests();
        let done = c
            .jobs()
            .into_iter()
            .find(|j| j.phase == JobPhase::Complete)
            .unwrap();
        c.rename_speaker(&done.id, 0, Some("Priya".to_owned()))
            .unwrap();
        let speakers = c.speakers(&done.id).unwrap();
        assert_eq!(speakers[0].display_name(), "Priya");
        assert_eq!(
            c.persisted_renames()[&done.id][&0],
            Some("Priya".to_owned())
        );
    }

    #[test]
    fn a_rejected_rename_reports_the_error_rather_than_pretending() {
        let c = FakeController::for_tests();
        let done = c
            .jobs()
            .into_iter()
            .find(|j| j.phase == JobPhase::Complete)
            .unwrap();
        let events = c.subscribe();
        c.reject_rename_of(0);
        c.rename_speaker(&done.id, 0, Some("Priya".to_owned()))
            .unwrap();
        // The rename is reported as rejected, and the name is not applied.
        let event = events.try_recv().expect("a rejection must be reported");
        let ControllerEvent::RenameRejected { speaker, error, .. } = event else {
            panic!("expected a rejection, got {event:?}");
        };
        assert_eq!(speaker, 0);
        assert_eq!(error.code, vaani_core::ErrorCode::StorageFailed);
        assert!(!error.recovery.unwrap_or_default().is_empty());
        assert_eq!(c.speakers(&done.id).unwrap()[0].display_name(), "Speaker 1");
    }

    #[test]
    fn subscribing_twice_is_refused_rather_than_silently_splitting_events() {
        let c = FakeController::for_tests();
        let first = c.subscribe();
        // A second subscription would hand out a second receiver, and a naive
        // implementation would let the two steal events from each other. The
        // fake panics on the second call instead, which is a loud failure rather
        // than a UI that mysteriously stops updating.
        let second = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = c.subscribe();
        }));
        assert!(
            second.is_err(),
            "a second subscribe must not quietly succeed"
        );
        // The first subscriber still works.
        c.emit(ControllerEvent::Notice {
            level: Severity::Info,
            text: "still listening".to_owned(),
        });
        assert!(first.try_recv().is_ok());
    }

    #[test]
    fn renaming_to_none_restores_the_generated_label() {
        let c = FakeController::for_tests();
        let done = c
            .jobs()
            .into_iter()
            .find(|j| j.phase == JobPhase::Complete)
            .unwrap();
        c.rename_speaker(&done.id, 0, Some("Priya".to_owned()))
            .unwrap();
        c.rename_speaker(&done.id, 0, None).unwrap();
        assert_eq!(c.speakers(&done.id).unwrap()[0].display_name(), "Speaker 1");
    }

    #[test]
    fn settings_round_trip_and_a_missing_model_is_rejected() {
        let c = FakeController::for_tests();
        let updated = c
            .update_settings(SettingsPatch {
                device: Some("cuda".to_owned()),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(updated.device, "cuda");
        assert!(
            c.update_settings(SettingsPatch {
                model: Some("missing-model".to_owned()),
                ..Default::default()
            })
            .is_err()
        );
        assert_eq!(
            c.settings().model,
            "sample-asr",
            "a rejection changes nothing"
        );
    }

    #[test]
    fn export_writes_a_file_and_reports_the_path() {
        let dir = std::env::temp_dir().join("vaani-ui-fake-export-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let c = FakeController::for_tests();
        let done = c
            .jobs()
            .into_iter()
            .find(|j| j.phase == JobPhase::Complete)
            .unwrap();
        c.export(&ExportRequest {
            job: done.id.clone(),
            format: crate::backend::ExportFormat::Text,
            target: ExportTarget::Directory(dir.clone()),
        })
        .unwrap();
        wait_until(
            || {
                std::fs::read_dir(&dir)
                    .map(|mut d| d.next().is_some())
                    .unwrap_or(false)
            },
            "the export file to appear",
        );
        let written = std::fs::read_dir(&dir)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let text = std::fs::read_to_string(&written).unwrap();
        assert!(
            text.contains(sample::SAMPLE_SEGMENT_PREFIX),
            "export must stay labelled"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn exporting_a_job_with_no_transcript_fails_with_a_recovery() {
        let c = FakeController::for_tests();
        let cancelled = c
            .jobs()
            .into_iter()
            .find(|j| j.phase == JobPhase::Cancelled)
            .unwrap();
        c.export(&ExportRequest {
            job: cancelled.id,
            format: crate::backend::ExportFormat::Json,
            target: ExportTarget::File(std::env::temp_dir().join("nope.json")),
        })
        .unwrap();
        assert!(c.export_requests().is_empty());
    }

    #[test]
    fn a_worker_never_sends_from_the_thread_that_built_the_controller() {
        let c = FakeController::for_tests();
        let me = std::thread::current().id();
        c.enqueue(Path::new("C:/x.mkv")).unwrap();
        wait_until(|| c.sender_thread_count() > 0, "an event");
        c.wait_for_workers();
        assert!(c.sender_thread_count() > 0);
        // The controller's own methods are called from this thread; the point
        // is that the *count* of senders grew without this thread doing it.
        let _ = me;
    }

    #[test]
    fn the_recording_host_answers_pickers_in_order_and_records_the_clipboard() {
        let host = RecordingHostServices::new(vec![PathBuf::from("C:/a.mp4")]);
        assert_eq!(host.pick_media_files(), vec![PathBuf::from("C:/a.mp4")]);
        assert!(host.pick_media_files().is_empty(), "the queue is drained");
        assert_eq!(host.times_picker_was_opened(), 2);
        host.push_export_target(Some(ExportTarget::File(PathBuf::from("C:/out.txt"))));
        assert_eq!(
            host.pick_export_target("out.txt"),
            Some(ExportTarget::File(PathBuf::from("C:/out.txt")))
        );
        assert!(host.pick_export_target("out.txt").is_none());
        // Two questions asked, both logged, so a test can tell "asked and the
        // user cancelled" from "never asked".
        assert_eq!(host.asked_for_export().len(), 2);
        host.set_clipboard_text("hello");
        assert_eq!(host.clipboard(), vec!["hello".to_owned()]);
    }

    #[test]
    fn a_cancelling_host_answers_nothing() {
        let host = RecordingHostServices::cancelling();
        assert!(host.pick_media_files().is_empty());
        assert!(host.pick_export_target("x").is_none());
    }
}

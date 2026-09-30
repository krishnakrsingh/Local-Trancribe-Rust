//! The contract between the UI and whatever actually transcribes things.
//!
//! # Why a trait, and why these method names
//!
//! The pipeline crates are being written in parallel and are not available to
//! this crate yet. Rather than guess at their shape, the UI states what it
//! *needs* and lets the production side fill it in. The trait below is that
//! statement, and it is deliberately narrow:
//!
//! * **Every method is non-blocking and runs on the event-loop thread.**
//!   Nothing here decodes audio, loads a model, or touches the database. A
//!   method that did would freeze the window, and a frozen window is the
//!   failure mode this whole design exists to avoid.
//! * **Work is reported back through [`ControllerEvent`], not return values.**
//!   A job takes minutes. A return value cannot span that, and a callback that
//!   the UI must remember to marshal is a bug waiting to happen. The
//!   controller owns an MPSC channel; the UI drains it on the event loop.
//! * **The UI keeps a cache and the controller keeps the truth.** Read
//!   accessors exist so the UI can rebuild from scratch (on startup, after a
//!   backend swap) without a race; the event stream is how it stays current.
//!
//! The concrete methods are named after what the *user* does, not after what
//! the pipeline does internally, so a reorganisation of the pipeline cannot
//! ripple into the UI.
//!
//! # What the production implementation must guarantee
//!
//! 1. `enqueue` returns as soon as the job is accepted, having published it
//!    through the event channel.
//! 2. `cancel` flips the job's [`vaani_core::CancelToken`] and returns. The UI
//!    does not wait for the job to actually stop; it shows "Cancelling" and
//!    waits for a terminal event.
//! 3. `rename_speaker` applies in memory immediately and persists
//!    asynchronously. `name: None` restores the generated `Speaker N` label.
//!    If persistence fails, the backend must emit
//!    [`ControllerEvent::RenameRejected`] and revert.
//! 4. `export` writes on a worker thread and emits
//!    [`ControllerEvent::ExportFinished`] or
//!    [`ControllerEvent::ExportFailed`].
//! 5. Every method must be safe to call from the event-loop thread only, and
//!    must not hold a lock that a worker thread needs for longer than a few
//!    microseconds.

use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crossbeam_channel::Receiver;
use vaani_core::{JobPhase, Transcript, VaaniError};

pub use vaani_core::UserFacingError;

/// A stable identifier for one transcription job.
///
/// Wraps a `String` rather than aliasing one so a `JobId` can never be passed
/// where a `SpeakerIndex` or a `PathBuf` is expected. The inner value is
/// whatever the backend uses in its own database; the UI only ever compares it
/// for equality and shows it in diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JobId(String);

impl JobId {
    /// Wrap an existing identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The identifier as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for JobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for JobId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for JobId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

/// A file format the transcript can be written out as.
///
/// Mirrors `vaani_core::EXPORT_EXTENSIONS` but is a closed enum here, because
/// a `ComboBox` needs a stable, ordered list and a `&[&str]` pair list does
/// not give the UI a compile error when a format is added without a label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ExportFormat {
    /// One paragraph per speaker turn.
    #[default]
    Text,
    /// Markdown with a heading per turn.
    Markdown,
    /// SubRip subtitles.
    Srt,
    /// WebVTT subtitles.
    Vtt,
    /// The full `vaani_core::Transcript` as JSON.
    Json,
}

impl ExportFormat {
    /// Every format, in the order the export control shows them.
    pub const ALL: [ExportFormat; 5] =
        [Self::Text, Self::Markdown, Self::Srt, Self::Vtt, Self::Json];

    /// File extension, without the dot.
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Text => "txt",
            Self::Markdown => "md",
            Self::Srt => "srt",
            Self::Vtt => "vtt",
            Self::Json => "json",
        }
    }

    /// Short label for the export control.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Text => "Plain text (.txt)",
            Self::Markdown => "Markdown (.md)",
            Self::Srt => "SubRip subtitles (.srt)",
            Self::Vtt => "WebVTT subtitles (.vtt)",
            Self::Json => "JSON (.json)",
        }
    }
}

impl fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Where an export should be written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportTarget {
    /// Write the format's default file name into this directory.
    Directory(PathBuf),
    /// Write exactly this path. The extension is the backend's problem, not
    /// the UI's: appending it twice is a worse bug than a missing one.
    File(PathBuf),
}

/// A request to write a transcript out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportRequest {
    /// Which job to export.
    pub job: JobId,
    /// Which format to write.
    pub format: ExportFormat,
    /// Where to write it.
    pub target: ExportTarget,
}

/// Everything the job list needs to render one row, with no further queries.
///
/// Deliberately a flat snapshot: a list of twenty jobs re-read twenty times
/// per frame is how a job list ends up stuttering, and the alternative (a row
/// type with a back-pointer to a live job) makes the UI hold locks.
#[derive(Debug, Clone, PartialEq)]
pub struct JobSummary {
    /// Stable identity.
    pub id: JobId,
    /// The file the user chose. Shown in full in the row's tooltip-equivalent
    /// accessible description, and by file name in the row itself.
    pub input: PathBuf,
    /// Where the job is in its lifecycle.
    pub phase: JobPhase,
    /// Overall progress in `[0, 1]`, or `None` when the backend cannot justify
    /// a number. `None` must render as an indeterminate indicator, never as
    /// `0%` and never as a fabricated percentage.
    pub overall: Option<f64>,
    /// Free-form detail, e.g. `chunk 12 / 48`.
    pub detail: Option<String>,
    /// Measured remaining time, if the backend can estimate one.
    pub eta: Option<Duration>,
    /// Set when the job failed. Cleared on any other phase change.
    pub error: Option<UserFacingError>,
    /// Whether a transcript can be shown for this job yet.
    pub has_transcript: bool,
    /// Whether cancellation has been requested but not yet observed. Lets the
    /// row say "Cancelling" instead of pretending the click did nothing.
    pub cancel_requested: bool,
    /// Word count, once a transcript exists.
    pub word_count: Option<usize>,
    /// Which device actually ran the speech recognition, once the engine is
    /// loaded.
    ///
    /// The distinction this preserves is the one that bit us: the machine can
    /// name an RTX 3050 Ti and still transcribe on the CPU, because ONNX
    /// Runtime's CUDA provider is a separate DLL that a build may not ship, and
    /// its absence is a silent fall back rather than an error. So the row and
    /// the footer report what ran, not what was detected.
    pub device: Option<String>,
    /// Measured cost of the finished job.
    ///
    /// `None` until a job completes, and permanently for one that failed or was
    /// cancelled: there is no speed to report for work that did not finish. The
    /// performance panel derives the real-time factor and the stage split from
    /// here rather than from an estimate, because a benchmark nobody measured is
    /// a guess with a decimal point in it.
    pub timings: Option<JobTimings>,
}

/// Measured cost of one finished job, in milliseconds.
///
/// The unit is milliseconds throughout, matching what the engine measures, and
/// the UI converts to seconds only for display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct JobTimings {
    /// Wall time for the whole job, model load included.
    pub elapsed_ms: u64,
    /// Length of the audio recognised, so real-time factor is audio over
    /// elapsed.
    pub audio_ms: i64,
    /// Feature extraction.
    pub frontend_ms: u64,
    /// The encoder session.
    pub encode_ms: u64,
    /// The decode loop.
    pub decode_ms: u64,
    /// Transducer emission steps, which is what the decode cost is measured per.
    pub decode_steps: usize,
}

impl JobTimings {
    /// Real-time factor: audio seconds per wall second. Below 1.0 is faster
    /// than realtime.
    ///
    /// `None` when there is no elapsed time or no audio, which is the honest
    /// answer for a zero-length or failed job rather than a division by zero
    /// dressed up as a number.
    #[must_use]
    pub fn real_time_factor(&self) -> Option<f64> {
        if self.elapsed_ms == 0 || self.audio_ms <= 0 {
            return None;
        }
        Some(self.audio_ms as f64 / 1000.0 / (self.elapsed_ms as f64 / 1000.0))
    }

    /// How many times faster than realtime this ran, i.e. `1 / rtf`.
    #[must_use]
    pub fn speed_multiple(&self) -> Option<f64> {
        self.real_time_factor()
            .filter(|r| *r > 0.0)
            .map(|r| 1.0 / r)
    }

    /// Cost per decode step in microseconds: the number that says whether the
    /// decode loop is doing arithmetic or paying launch overhead.
    #[must_use]
    pub fn microseconds_per_step(&self) -> Option<f64> {
        if self.decode_steps == 0 {
            return None;
        }
        Some(self.decode_ms as f64 * 1000.0 / self.decode_steps as f64)
    }
}

/// One speaker as the speaker panel shows them.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerSummary {
    /// Cluster index, stable for the life of the job. 0-based; the UI displays
    /// `index + 1` because humans count from one.
    pub index: usize,
    /// The user's name, if they set one.
    pub name: Option<String>,
    /// Total attributed speech time in milliseconds.
    pub talk_time_ms: i64,
    /// Share of total attributed speech, in `[0, 1]`.
    pub share: Option<f32>,
}

impl SpeakerSummary {
    /// What to render: the user's name, or the generated label.
    pub fn display_name(&self) -> String {
        match self.name.as_deref() {
            Some(name) if !name.trim().is_empty() => name.to_owned(),
            _ => format!("Speaker {}", self.index + 1),
        }
    }
}

/// One entry in a settings drop-down.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Choice {
    /// The value the backend understands. Never shown to the user.
    pub id: String,
    /// What the user sees.
    pub label: String,
    /// Second line of the drop-down entry, e.g. a model size or a device name.
    pub detail: Option<String>,
    /// `false` for an entry that is known but not usable right now, e.g. a
    /// model that has not been downloaded. The UI must say *why* rather than
    /// hiding it, so the user can go and fix it.
    pub available: bool,
}

/// Which colour scheme the window should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Theme {
    /// Follow the operating system.
    #[default]
    System,
    /// Always light.
    Light,
    /// Always dark.
    Dark,
}

/// The settings the UI reads and writes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AppSettings {
    /// `Choice::id` of the selected ASR model.
    pub model: String,
    /// `Choice::id` of the selected compute device.
    pub device: String,
    /// BCP-47-ish language tag, or the empty string for "detect".
    pub language: String,
    /// Whether to run speaker separation.
    pub diarize: bool,
    /// Colour scheme.
    pub theme: Theme,
}

/// A partial settings update. `None` means "leave this alone", which is why
/// this is a struct of options and not an `AppSettings`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SettingsPatch {
    /// New model id.
    pub model: Option<String>,
    /// New device id.
    pub device: Option<String>,
    /// New language tag.
    pub language: Option<String>,
    /// Whether to diarize.
    pub diarize: Option<bool>,
    /// New theme.
    pub theme: Option<Theme>,
}

impl SettingsPatch {
    /// `true` when the patch would change nothing.
    pub fn is_empty(&self) -> bool {
        self.model.is_none()
            && self.device.is_none()
            && self.language.is_none()
            && self.diarize.is_none()
            && self.theme.is_none()
    }
}

/// How loudly to say something in the notice area.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Something worked.
    Info,
    /// Something the user may want to know about, but the job carried on.
    Warning,
    /// Something failed.
    Error,
}

/// Something the backend wants the UI to know about.
///
/// Every variant carries everything needed to render it, so applying an event
/// never needs a second query and can never observe a torn state.
#[derive(Debug, Clone, PartialEq)]
pub enum ControllerEvent {
    /// A job appeared in the list.
    JobAdded(JobSummary),
    /// A job's summary changed: progress, phase, error, or the whole thing.
    ///
    /// A full replacement rather than a delta. The UI coalesces a burst of
    /// these down to the last one per job, which is only sound because each
    /// event is self-contained.
    JobChanged(JobSummary),
    /// A job left the list.
    JobRemoved(JobId),
    /// A transcript became available for a job.
    TranscriptReady(JobId),
    /// The speaker list for a job changed (diarization finished, or a rename
    /// was accepted).
    SpeakersChanged(JobId),
    /// A speaker rename could not be persisted and has been rolled back.
    RenameRejected {
        /// The job whose speaker list is affected.
        job: JobId,
        /// Which speaker.
        speaker: usize,
        /// Why it failed, already in user-facing form.
        error: UserFacingError,
    },
    /// An export finished.
    ExportFinished {
        /// The job that was exported.
        job: JobId,
        /// Every file that was written.
        paths: Vec<PathBuf>,
    },
    /// An export failed.
    ExportFailed {
        /// The job that was being exported.
        job: JobId,
        /// Why it failed, already in user-facing form.
        error: UserFacingError,
    },
    /// The settings were changed, possibly by another window or by the
    /// backend discovering something.
    SettingsChanged(AppSettings),
    /// A one-off message for the notice area.
    Notice {
        /// How loudly to say it.
        level: Severity,
        /// The message. Must already be user-facing text.
        text: String,
    },
}

impl ControllerEvent {
    /// The job this event is about, when it is about exactly one.
    pub fn job(&self) -> Option<&JobId> {
        match self {
            Self::JobAdded(j) | Self::JobChanged(j) => Some(&j.id),
            Self::JobRemoved(id)
            | Self::TranscriptReady(id)
            | Self::SpeakersChanged(id)
            | Self::ExportFinished { job: id, .. }
            | Self::ExportFailed { job: id, .. } => Some(id),
            Self::RenameRejected { job, .. } => Some(job),
            Self::SettingsChanged(_) | Self::Notice { .. } => None,
        }
    }
}

/// The application backend, as the UI needs it.
///
/// See the module documentation for the rules an implementation must follow.
/// A production implementation is a thin adapter over `vaani-app`'s pipeline;
/// a test implementation is [`crate::fake::FakeController`].
pub trait JobController: Send + Sync + 'static {
    /// Name shown in the window's status bar, e.g. `vaani-app` or `demo`.
    fn backend_name(&self) -> String;

    /// Whether this is a demonstration backend producing labelled sample data.
    ///
    /// The UI shows a persistent banner when this is `true`, so a screenshot
    /// of a fake run can never be mistaken for the product.
    fn is_demo(&self) -> bool;

    /// Every job, in creation order (oldest first).
    fn jobs(&self) -> Vec<JobSummary>;

    /// One job, or `None` if it is not (or is no longer) known.
    fn job(&self, id: &JobId) -> Option<JobSummary>;

    /// Accept a new job for `input`. Returns as soon as the job is registered.
    fn enqueue(&self, input: &Path) -> Result<JobId, UserFacingError>;

    /// Request cancellation. Returns as soon as the request is recorded.
    fn cancel(&self, id: &JobId) -> Result<(), UserFacingError>;

    /// Forget a job. Whether that also deletes its files is the backend's
    /// decision; the UI only asks.
    fn remove(&self, id: &JobId) -> Result<(), UserFacingError>;

    /// The transcript for a job, if it has one.
    ///
    /// Returned behind an `Arc` so opening a 20,000-word transcript costs a
    /// refcount bump rather than a deep copy, and so the transcript view can
    /// keep rendering from it while a rename is applied elsewhere.
    fn transcript(&self, id: &JobId)
    -> Result<Option<std::sync::Arc<Transcript>>, UserFacingError>;

    /// The speakers for a job, with talk times.
    fn speakers(&self, id: &JobId) -> Result<Vec<SpeakerSummary>, UserFacingError>;

    /// Rename a speaker, or restore the generated name with `None`.
    ///
    /// Applies in memory at once; persistence is asynchronous and reports
    /// failure as [`ControllerEvent::RenameRejected`].
    fn rename_speaker(
        &self,
        job: &JobId,
        speaker: usize,
        name: Option<String>,
    ) -> Result<(), UserFacingError>;

    /// Start an export. Writes on a worker thread.
    fn export(&self, request: &ExportRequest) -> Result<(), UserFacingError>;

    /// The current settings.
    fn settings(&self) -> AppSettings;

    /// Apply a partial settings update and return the new settings.
    ///
    /// The backend may reject a change (a model that is not downloaded, a
    /// device that is not present); on rejection the settings are unchanged.
    fn update_settings(&self, patch: SettingsPatch) -> Result<AppSettings, UserFacingError>;

    /// Take an event subscription.
    ///
    /// Call this **once** per controller. Crossbeam receivers share the queue,
    /// so a second subscriber would steal events from the first and the UI
    /// would silently lose progress updates. Later calls receive only events
    /// sent after they were made.
    fn subscribe(&self) -> Receiver<ControllerEvent>;

    /// Selectable ASR models.
    fn models(&self) -> Vec<Choice>;

    /// Selectable compute devices.
    fn devices(&self) -> Vec<Choice>;

    /// Selectable source languages. An entry with the empty id means "detect".
    fn languages(&self) -> Vec<Choice>;
}

/// The handful of things the UI needs from the host process that are not the
/// job pipeline: dialogs and the clipboard.
///
/// Split out from [`JobController`] on purpose. `rfd` is a dependency of the
/// binary, not of this crate, and a trait object is how a dialog reaches the
/// UI without the UI depending on a dialog library. The architect wires the
/// real implementation in wave 2; [`crate::fake::RecordingHostServices`] is the
/// test double.
pub trait HostServices: Send + Sync + 'static {
    /// Open a multi-select media file picker. An empty result means cancelled.
    fn pick_media_files(&self) -> Vec<PathBuf>;

    /// Ask where to write an export. `suggested_name` is the default file
    /// name, extension included. `None` means cancelled.
    fn pick_export_target(&self, suggested_name: &str) -> Option<ExportTarget>;

    /// Put `text` on the system clipboard.
    fn set_clipboard_text(&self, text: &str);

    /// A label for the host, shown in the window footer.
    fn host_name(&self) -> String;
}

/// A [`HostServices`] that does nothing, for hosts that have not wired dialogs
/// up yet.
///
/// Every method is a documented no-op rather than a panic: a window that dies
/// because the clipboard is unavailable is a worse product than one whose Copy
/// button silently does nothing for a moment.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnavailableHostServices;

impl HostServices for UnavailableHostServices {
    fn pick_media_files(&self) -> Vec<PathBuf> {
        Vec::new()
    }

    fn pick_export_target(&self, _suggested_name: &str) -> Option<ExportTarget> {
        None
    }

    fn set_clipboard_text(&self, _text: &str) {}

    fn host_name(&self) -> String {
        "no host services".to_owned()
    }
}

/// The user-facing projection of a [`VaaniError`].
///
/// The UI contract speaks [`UserFacingError`], because that is the only part of
/// an error the brief allows on screen. A backend that already has a
/// [`VaaniError`] in hand converts through here rather than formatting it, so
/// there is no way to accidentally put a diagnostic in the `detail` field.
#[must_use]
pub fn user_facing(error: &VaaniError) -> UserFacingError {
    error.user.clone()
}

/// A [`JobController`] behind an `Arc`, forwarded.
///
/// The UI holds its controller as `Rc<UiModel<C>>` and a test wants to reach
/// the same controller to assert on it. Rather than have every implementation
/// write this forwarding impl, the blanket impl provides it, and `C` can be
/// `Arc<TheRealBackend>` in production exactly as it is `Arc<FakeController>` in
/// a test. The swap between the two is then a change of one type argument.
impl<T: JobController + ?Sized> JobController for std::sync::Arc<T> {
    fn backend_name(&self) -> String {
        (**self).backend_name()
    }

    fn is_demo(&self) -> bool {
        (**self).is_demo()
    }

    fn jobs(&self) -> Vec<JobSummary> {
        (**self).jobs()
    }

    fn job(&self, id: &JobId) -> Option<JobSummary> {
        (**self).job(id)
    }

    fn enqueue(&self, input: &Path) -> Result<JobId, UserFacingError> {
        (**self).enqueue(input)
    }

    fn cancel(&self, id: &JobId) -> Result<(), UserFacingError> {
        (**self).cancel(id)
    }

    fn remove(&self, id: &JobId) -> Result<(), UserFacingError> {
        (**self).remove(id)
    }

    fn transcript(
        &self,
        id: &JobId,
    ) -> Result<Option<std::sync::Arc<Transcript>>, UserFacingError> {
        (**self).transcript(id)
    }

    fn speakers(&self, id: &JobId) -> Result<Vec<SpeakerSummary>, UserFacingError> {
        (**self).speakers(id)
    }

    fn rename_speaker(
        &self,
        job: &JobId,
        speaker: usize,
        name: Option<String>,
    ) -> Result<(), UserFacingError> {
        (**self).rename_speaker(job, speaker, name)
    }

    fn export(&self, request: &ExportRequest) -> Result<(), UserFacingError> {
        (**self).export(request)
    }

    fn settings(&self) -> AppSettings {
        (**self).settings()
    }

    fn update_settings(&self, patch: SettingsPatch) -> Result<AppSettings, UserFacingError> {
        (**self).update_settings(patch)
    }

    fn subscribe(&self) -> Receiver<ControllerEvent> {
        (**self).subscribe()
    }

    fn models(&self) -> Vec<Choice> {
        (**self).models()
    }

    fn devices(&self) -> Vec<Choice> {
        (**self).devices()
    }

    fn languages(&self) -> Vec<Choice> {
        (**self).languages()
    }
}

/// Every phase, in lifecycle order, including the one `JobPhase::ALL` omits.
///
/// `JobPhase::ALL` deliberately excludes `Cancelled` because it is the forward
/// path; a UI that renders the state machine has to render all of it, so this
/// list exists and a test asserts the two stay in step.
pub const ALL_PHASES: [JobPhase; 14] = [
    JobPhase::Queued,
    JobPhase::Preparing,
    JobPhase::ExtractingAudio,
    JobPhase::NormalizingAudio,
    JobPhase::Vad,
    JobPhase::Asr,
    JobPhase::Diarization,
    JobPhase::Alignment,
    JobPhase::PostProcessing,
    JobPhase::Structuring,
    JobPhase::Exporting,
    JobPhase::Complete,
    JobPhase::Failed,
    JobPhase::Cancelled,
];

#[cfg(test)]
mod tests {
    use super::*;
    use vaani_core::err;

    #[test]
    fn all_phases_is_the_union_of_the_state_machine_and_cancelled() {
        // If the architect adds a phase, this test fails and the UI gets a
        // deliberate "I do not know how to draw that" instead of a wrong label.
        let mut from_all: Vec<JobPhase> = JobPhase::ALL.to_vec();
        from_all.push(JobPhase::Cancelled);
        assert_eq!(from_all.len(), ALL_PHASES.len());
        for p in ALL_PHASES {
            assert!(
                from_all.contains(&p),
                "{p} is in ALL_PHASES but not in JobPhase::ALL + Cancelled"
            );
        }
    }

    #[test]
    fn all_phases_is_in_lifecycle_order() {
        for pair in ALL_PHASES.windows(2) {
            assert!(
                pair[0] < pair[1],
                "{} came after {} in ALL_PHASES",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn job_ids_are_distinct_from_other_strings() {
        assert_eq!(JobId::from("a"), JobId::new("a".to_owned()));
        assert_eq!(JobId::new("a").as_str(), "a");
    }

    #[test]
    fn export_format_extensions_match_the_domain_registry() {
        // `vaani_core::EXPORT_EXTENSIONS` is the product's advertised list; the
        // UI's enum is a closed version of it. If they diverge, the file picker
        // and the export control start lying to each other.
        for (suffix, _) in vaani_core::EXPORT_EXTENSIONS {
            assert!(
                ExportFormat::ALL.iter().any(|f| f.extension() == *suffix),
                "{suffix} is exported by the domain but not by the UI"
            );
        }
        assert_eq!(ExportFormat::ALL.len(), vaani_core::EXPORT_EXTENSIONS.len());
    }

    #[test]
    fn speaker_display_name_falls_back_when_blank() {
        let blank = SpeakerSummary {
            index: 0,
            name: Some("   ".to_owned()),
            talk_time_ms: 0,
            share: None,
        };
        assert_eq!(blank.display_name(), "Speaker 1");
        let named = SpeakerSummary {
            name: Some("Priya".to_owned()),
            ..blank
        };
        assert_eq!(named.display_name(), "Priya");
    }

    #[test]
    fn event_job_accessor_finds_the_job_or_says_none() {
        let id = JobId::from("j1");
        assert_eq!(ControllerEvent::JobRemoved(id.clone()).job(), Some(&id));
        assert_eq!(
            ControllerEvent::Notice {
                level: Severity::Info,
                text: "hi".into()
            }
            .job(),
            None
        );
    }

    #[test]
    fn settings_patch_emptiness_is_correct() {
        assert!(SettingsPatch::default().is_empty());
        assert!(
            !SettingsPatch {
                diarize: Some(true),
                ..Default::default()
            }
            .is_empty()
        );
    }

    #[test]
    fn unavailable_host_services_is_inert_not_fatal() {
        let h = UnavailableHostServices;
        assert!(h.pick_media_files().is_empty());
        assert!(h.pick_export_target("x.txt").is_none());
        h.set_clipboard_text("irrelevant");
        assert!(!h.host_name().is_empty());
    }

    #[test]
    fn error_events_carry_user_facing_text() {
        // The banner renders `error.title` and `error.detail` verbatim, so the
        // fake must build real `UserFacingError`s rather than strings.
        let e = user_facing(&err::model_unavailable());
        assert!(!e.title.is_empty() && !e.detail.is_empty());
        assert!(e.recovery.is_none(), "and the constructor adds no recovery");
    }
}

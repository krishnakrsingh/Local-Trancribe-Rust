//! View state: everything the window shows, expressed without Slint.
//!
//! # Why this is a separate module
//!
//! Two reasons, and both of them are about being able to test.
//!
//! The first is that Slint's generated types are awkward to assert on. A test
//! that wants to check "cancelling shows Cancelling, not Transcribing" against
//! a `ModelRc<VecModel<JobRow>>` has to reach through a property and index a
//! model. Against a `JobListState` it calls `state.rows()[0].status_text`.
//! Same assertion, a fraction of the ceremony, and the fast tests do not have
//! to stand up a window to run.
//!
//! The second is that the interesting logic here is not drawing. Clamping
//! progress, coalescing a burst of updates, extending a selection with shift,
//! deciding whether a rename actually changed anything: all of it is arithmetic
//! and string handling, and all of it deserves a unit test per branch. Putting
//! it in `.slint` bindings would mean either untestable logic or logic that
//! only runs when a real window is on screen.
//!
//! Nothing in this file may name a `slint` type. That is the rule which keeps
//! the dependency pointing one way.

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use vaani_core::{JobPhase, MediaLength, Segment, Transcript};

use crate::backend::{
    Choice, ControllerEvent, JobId, JobSummary, SettingsPatch, Severity, SpeakerSummary, Theme,
};

/// Rows per second in the transcript's own scroll model is not a thing; the
/// transcript scrolls in pixels. These are the two numbers that are.
const MAX_DETAIL_CHARS: usize = 400;

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

/// `1m 05s`, `45s`, `2h 03m`, `0s`. Never empty: a duration of zero is a fact,
/// and blanking it makes a real value look missing.
#[must_use]
pub fn format_duration(d: Duration) -> String {
    format_millis(d.as_millis() as i64)
}

/// The same, from milliseconds, which is what `Time` carries.
#[must_use]
pub fn format_millis(ms: i64) -> String {
    let total = ms.max(0);
    let (h, m, s) = (
        total / 3_600_000,
        (total / 60_000) % 60,
        (total / 1000) % 60,
    );
    if h > 0 {
        format!("{h}h {m:02}m")
    } else if m > 0 {
        format!("{m}m {s:02}s")
    } else {
        format!("{s}s")
    }
}

/// A remaining-time estimate, phrased so that it is obviously an estimate.
#[must_use]
pub fn format_eta(eta: Option<Duration>) -> String {
    match eta {
        Some(d) if d.as_secs() < 2 => "finishing".to_owned(),
        Some(d) => format!("about {} left", format_duration(d)),
        None => String::new(),
    }
}

/// `43%`, from a fraction. `None` in, empty out: an indeterminate phase must
/// never be drawn as `0%`.
#[must_use]
pub fn format_percent(fraction: Option<f64>) -> String {
    match fraction {
        Some(f) => format!("{:.0}%", (f.clamp(0.0, 1.0) * 100.0).round()),
        None => String::new(),
    }
}

/// `1,204 words`, with a thousands separator.
#[must_use]
pub fn format_word_count(words: usize) -> String {
    let digits = words.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3 + 8);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out.push_str(if words == 1 { " word" } else { " words" });
    out
}

/// The file name of a path, or the whole path if it has none.
#[must_use]
pub fn file_name_of(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

/// A one-line phase description, taken from the domain so the wording cannot
/// drift between the CLI, the window and the database.
#[must_use]
pub fn phase_label(phase: JobPhase) -> &'static str {
    phase.label()
}

/// A glyph for a job's state, chosen so the state survives a monochrome
/// display, a colour vision deficiency and a screen reader.
///
/// ASCII only. A symbol that is missing from the system font renders as a tofu
/// box, and a tofu box conveys less than the word it was meant to replace.
#[must_use]
pub fn status_glyph(job: &JobSummary) -> String {
    if job.cancel_requested && !job.phase.is_terminal() {
        return "..".to_owned();
    }
    match job.phase {
        JobPhase::Complete => "OK".to_owned(),
        JobPhase::Failed => "ERR".to_owned(),
        JobPhase::Cancelled => "STOP".to_owned(),
        JobPhase::Queued => "WAIT".to_owned(),
        _ => "RUN".to_owned(),
    }
}

/// The stable phase code, for diagnostics and for `accessible-description`.
#[must_use]
pub fn phase_code(phase: JobPhase) -> &'static str {
    phase.as_str()
}

// ---------------------------------------------------------------------------
// Job list
// ---------------------------------------------------------------------------

/// One row of the job list, flattened for display.
#[derive(Debug, Clone, PartialEq)]
pub struct JobRow {
    /// Job id, used as the row's identity for selection and for the callback.
    pub id: String,
    /// File name only; the full path goes in `input_path`.
    pub name: String,
    /// The full path, for the tooltip and the accessible description.
    pub input_path: String,
    /// Human phase name.
    pub phase_label: String,
    /// Stable phase code.
    pub phase_code: String,
    /// Clamped to `[0, 1]`, or `None` when indeterminate.
    pub progress: Option<f64>,
    /// Whether the bar should *sweep* rather than fill.
    ///
    /// Separate from `progress` on purpose. A sweep is an animation and an
    /// animation costs CPU every frame, so only a genuinely running phase gets
    /// one. A cancelled job with no progress draws an empty bar and says
    /// "Cancelled" in words, which is both cheaper and more honest than a bar
    /// that pretends to still be working.
    pub progress_indeterminate: bool,
    /// `43%`, or empty when indeterminate.
    pub progress_text: String,
    /// `about 2m left`, or empty.
    pub eta_text: String,
    /// The one-line status under the name: detail, error title, or the terminal
    /// summary. Never empty for a row the user can see.
    pub status_text: String,
    /// A glyph for the same state as `status_text`, so the row is readable
    /// without relying on colour. ASCII on purpose: a symbol missing from the
    /// system font renders as a tofu box, which is worse than a word.
    pub status_glyph: String,
    /// Whether the job is running (not queued, not finished).
    pub is_running: bool,
    /// Whether the job is waiting for a worker.
    pub is_queued: bool,
    /// Whether the job has finished one way or another.
    pub is_terminal: bool,
    /// Whether it finished successfully.
    pub is_complete: bool,
    /// Whether it was cancelled.
    pub is_cancelled: bool,
    /// Whether it failed.
    pub is_failed: bool,
    /// Whether cancellation has been requested but not yet observed.
    pub is_cancelling: bool,
    /// Whether this row is the selected one.
    pub is_selected: bool,
    /// Whether a transcript can be shown.
    pub has_transcript: bool,
    /// The error title, empty when there is no error.
    pub error_title: String,
    /// The error's recovery action, empty when it has none.
    pub error_recovery: String,
    /// The stable error code, empty when there is no error.
    pub error_code: String,
}

/// The job list's state: the rows, and which one is selected.
///
/// Selection is held as a [`JobId`], not as an index, because indices move when
/// a job finishes and disappears. A list that selects "row 2" and then renders
/// a different job there is a bug users describe as "it clicked the wrong
/// thing".
#[derive(Debug, Clone, Default, PartialEq)]
pub struct JobListState {
    jobs: Vec<JobSummary>,
    selected: Option<JobId>,
}

impl JobListState {
    /// An empty list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from a backend snapshot.
    #[must_use]
    pub fn from_jobs(jobs: Vec<JobSummary>) -> Self {
        let mut state = Self {
            jobs,
            selected: None,
        };
        state.select_default();
        state
    }

    /// The jobs, in creation order.
    #[must_use]
    pub fn jobs(&self) -> &[JobSummary] {
        &self.jobs
    }

    /// One job by id, or `None` if it is not in the list.
    #[must_use]
    pub fn job(&self, id: &JobId) -> Option<&JobSummary> {
        self.jobs.iter().find(|j| j.id == *id)
    }

    /// The selected job id.
    #[must_use]
    pub fn selected(&self) -> Option<&JobId> {
        self.selected.as_ref()
    }

    /// The selected job's summary.
    #[must_use]
    pub fn selected_job(&self) -> Option<&JobSummary> {
        self.selected
            .as_ref()
            .and_then(|id| self.jobs.iter().find(|j| j.id == *id))
    }

    /// Select `id`, if it is in the list. Returns whether it was.
    pub fn select(&mut self, id: &JobId) -> bool {
        if self.jobs.iter().any(|j| j.id == *id) {
            self.selected = Some(id.clone());
            true
        } else {
            false
        }
    }

    /// Select by row index into [`Self::rows`], which is newest first.
    pub fn select_row(&mut self, index: usize) -> bool {
        match self.rows().get(index) {
            Some(row) => {
                let id = JobId::from(row.id.as_str());
                self.select(&id)
            }
            None => false,
        }
    }

    /// Select the first job if nothing is selected, so the window is never
    /// showing an empty right-hand pane with jobs in the list.
    pub fn select_default(&mut self) {
        if self.selected.is_some() {
            return;
        }
        if let Some(first) = self.jobs.last() {
            self.selected = Some(first.id.clone());
        }
    }

    /// Insert or replace a job, keeping the list ordered by creation.
    ///
    /// Replacement is by id, and an update to an *older* job does not move it
    /// to the top: a list that reorders itself while you are reading it is
    /// unusable for a batch tool.
    pub fn upsert(&mut self, job: JobSummary) -> bool {
        match self.jobs.iter_mut().find(|j| j.id == job.id) {
            Some(slot) => {
                let changed = *slot != job;
                *slot = job;
                changed
            }
            None => {
                self.jobs.push(job);
                true
            }
        }
    }

    /// Forget a job. Clears the selection if it was the selected one.
    pub fn remove(&mut self, id: &JobId) -> bool {
        let before = self.jobs.len();
        self.jobs.retain(|j| j.id != *id);
        let removed = self.jobs.len() != before;
        if removed && self.selected.as_ref() == Some(id) {
            self.selected = None;
            self.select_default();
        }
        removed
    }

    /// Number of jobs currently running.
    #[must_use]
    pub fn running_count(&self) -> usize {
        self.jobs.iter().filter(|j| j.phase.is_active()).count()
    }

    /// Number of jobs in any terminal phase.
    #[must_use]
    pub fn finished_count(&self) -> usize {
        self.jobs.iter().filter(|j| j.phase.is_terminal()).count()
    }

    /// The display rows, newest first.
    #[must_use]
    pub fn rows(&self) -> Vec<JobRow> {
        let selected = self.selected.as_ref();
        self.jobs
            .iter()
            .rev()
            .map(|job| Self::row_for(job, selected.is_some_and(|s| *s == job.id)))
            .collect()
    }

    /// Flatten one job for display.
    #[must_use]
    pub fn row_for(job: &JobSummary, is_selected: bool) -> JobRow {
        let progress = job.overall.map(|f| f.clamp(0.0, 1.0));
        let is_cancelling = job.cancel_requested && !job.phase.is_terminal();
        let is_terminal = job.phase.is_terminal();
        // Only a running phase sweeps. See `progress_indeterminate`.
        let progress_indeterminate = progress.is_none() && job.phase.is_active();

        let status_text = if is_cancelling {
            "Cancelling…".to_owned()
        } else if let Some(error) = &job.error {
            error.title.clone()
        } else if let Some(detail) = job
            .detail
            .as_deref()
            .map(str::trim)
            .filter(|d| !d.is_empty())
        {
            detail.to_owned()
        } else if job.phase == JobPhase::Queued {
            "Waiting to start".to_owned()
        } else if is_terminal {
            match job.phase {
                JobPhase::Complete => "Finished".to_owned(),
                JobPhase::Cancelled => "Cancelled".to_owned(),
                _ => phase_label(job.phase).to_owned(),
            }
        } else {
            phase_label(job.phase).to_owned()
        };

        JobRow {
            id: job.id.as_str().to_owned(),
            name: file_name_of(&job.input),
            input_path: job.input.display().to_string(),
            phase_label: phase_label(job.phase).to_owned(),
            phase_code: phase_code(job.phase).to_owned(),
            progress,
            progress_indeterminate,
            progress_text: format_percent(progress),
            eta_text: format_eta(job.eta),
            status_text,
            status_glyph: status_glyph(job),
            is_running: job.phase.is_active(),
            is_queued: job.phase == JobPhase::Queued,
            is_terminal,
            is_complete: job.phase == JobPhase::Complete,
            is_cancelled: job.phase == JobPhase::Cancelled,
            is_failed: job.phase == JobPhase::Failed,
            is_cancelling,
            is_selected,
            has_transcript: job.has_transcript,
            error_title: job
                .error
                .as_ref()
                .map_or_else(String::new, |e| e.title.clone()),
            error_recovery: job
                .error
                .as_ref()
                .and_then(|e| e.recovery.clone())
                .unwrap_or_default(),
            error_code: job
                .error
                .as_ref()
                .map_or_else(String::new, |e| e.code.as_str().to_owned()),
        }
    }
}

// ---------------------------------------------------------------------------
// Event coalescing
// ---------------------------------------------------------------------------

/// Collapse a burst of events so a fast producer cannot flood the event loop.
///
/// # The rule, and why it is sound
///
/// `JobAdded` and `JobChanged` are *full snapshots* of a job, so applying only
/// the last one for a given job produces exactly the same final state as
/// applying all of them. Everything else — removals, transcript availability,
/// export results, errors — is kept, in order, because those are the events a
/// user must not miss.
///
/// This is the difference between a progress bar that repaints 400 times a
/// second and one that repaints at the display's rate. It is not an
/// optimisation, it is the difference between a usable window and a hot CPU.
#[must_use]
pub fn coalesce(events: Vec<ControllerEvent>) -> Vec<ControllerEvent> {
    let mut out: Vec<ControllerEvent> = Vec::with_capacity(events.len());
    let mut last_snapshot: Vec<usize> = Vec::new();

    for event in events {
        match &event {
            ControllerEvent::JobAdded(job) | ControllerEvent::JobChanged(job) => {
                let id = job.id.clone();
                if let Some(&slot) = last_snapshot.iter().find(|&&s| {
                    matches!(
                        &out[s],
                        ControllerEvent::JobAdded(j) | ControllerEvent::JobChanged(j) if j.id == id
                    )
                }) {
                    out[slot] = event;
                } else {
                    last_snapshot.push(out.len());
                    out.push(event);
                }
            }
            _ => out.push(event),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Transcript
// ---------------------------------------------------------------------------

/// One transcript row, flattened for display.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptRow {
    /// Index into `Transcript::segments`, which is also the row's identity for
    /// selection and for the click callback.
    pub index: usize,
    /// `hh:mm:ss`, or `mm:ss` under an hour.
    pub timestamp: String,
    /// The speaker's display name, or empty for an unattributed segment.
    pub speaker: String,
    /// The segment's text.
    pub text: String,
    /// Whether this segment is inside the selection.
    pub is_selected: bool,
    /// Whether this segment is the first of a run by this speaker. Drives the
    /// visual grouping, so a speaker's turn is visibly one block.
    pub is_turn_start: bool,
    /// Whether this segment matches the current search.
    pub is_match: bool,
    /// Whether this segment has no speaker attribution.
    pub is_unattributed: bool,
}

/// A contiguous selection over segment indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Selection {
    /// Where the selection was started.
    pub anchor: Option<usize>,
    /// Where the moving end is.
    pub focus: Option<usize>,
}

impl Selection {
    /// Nothing selected.
    #[must_use]
    pub fn none() -> Self {
        Self::default()
    }

    /// A single segment.
    #[must_use]
    pub fn single(index: usize) -> Self {
        Self {
            anchor: Some(index),
            focus: Some(index),
        }
    }

    /// Whether anything is selected.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.anchor.is_none() || self.focus.is_none()
    }

    /// The inclusive range, normalised so `start <= end`.
    #[must_use]
    pub fn range(&self) -> Option<(usize, usize)> {
        match (self.anchor, self.focus) {
            (Some(a), Some(f)) => Some((a.min(f), a.max(f))),
            _ => None,
        }
    }

    /// Whether `index` is inside the selection.
    #[must_use]
    pub fn contains(&self, index: usize) -> bool {
        self.range()
            .is_some_and(|(lo, hi)| index >= lo && index <= hi)
    }

    /// Number of selected segments.
    #[must_use]
    pub fn len(&self) -> usize {
        self.range().map_or(0, |(lo, hi)| hi - lo + 1)
    }
}

/// The transcript pane's state.
///
/// Holds the transcript behind an `Arc` so selecting, scrolling and copying
/// never touch the pipeline's data, and so a rename applied by the backend
/// cannot mutate a transcript that is mid-render.
#[derive(Debug, Clone, Default)]
pub struct TranscriptView {
    transcript: Option<Arc<Transcript>>,
    selection: Selection,
    query: String,
    matches: Vec<usize>,
    match_cursor: Option<usize>,
}

impl TranscriptView {
    /// An empty pane.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Show `transcript`. Resets the selection and the search, because a
    /// selection index into the previous transcript means nothing now.
    pub fn set_transcript(&mut self, transcript: Option<Arc<Transcript>>) {
        self.transcript = transcript;
        self.selection = Selection::none();
        self.set_query(String::new());
    }

    /// The transcript, if any.
    #[must_use]
    pub fn transcript(&self) -> Option<&Arc<Transcript>> {
        self.transcript.as_ref()
    }

    /// Number of segments.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.transcript.as_ref().map_or(0, |t| t.segments.len())
    }

    /// Total words across every segment.
    #[must_use]
    pub fn word_count(&self) -> usize {
        self.transcript
            .as_ref()
            .map_or(0, |t| t.segments.iter().map(|s| s.words.len()).sum())
    }

    /// Total attributed and unattributed time, for the footer.
    ///
    /// A **length**, which is what the footer displays: how much media this
    /// covers. When the transcript declares none, the inferred length is the
    /// span of its own first and last segment — an inference, not a
    /// measurement, and deliberately not the last segment's end, which is a
    /// position and would be wrong for anything that does not start at zero.
    #[must_use]
    pub fn duration(&self) -> Option<MediaLength> {
        self.transcript
            .as_ref()
            .map(|t| t.duration.unwrap_or_else(|| t.inferred_length()))
    }

    /// The current selection.
    #[must_use]
    pub fn selection(&self) -> Selection {
        self.selection
    }

    /// The active search string.
    #[must_use]
    pub fn query(&self) -> &str {
        &self.query
    }

    /// Indices of matching segments, in order.
    #[must_use]
    pub fn matches(&self) -> &[usize] {
        &self.matches
    }

    fn segment(&self, index: usize) -> Option<&Segment> {
        self.transcript.as_ref().and_then(|t| t.segments.get(index))
    }

    /// The display name for a segment's speaker, or `""` when unattributed.
    #[must_use]
    pub fn speaker_name(&self, index: usize) -> String {
        let Some(transcript) = self.transcript.as_ref() else {
            return String::new();
        };
        let Some(segment) = transcript.segments.get(index) else {
            return String::new();
        };
        match segment.speaker {
            None => String::new(),
            Some(speaker) => transcript
                .speaker(speaker)
                .map_or_else(|| format!("Speaker {}", speaker + 1), |s| s.display_name()),
        }
    }

    /// Whether `index` starts a new run by one speaker.
    #[must_use]
    pub fn is_turn_start(&self, index: usize) -> bool {
        let Some(transcript) = self.transcript.as_ref() else {
            return false;
        };
        match (transcript.segments.get(index), index.checked_sub(1)) {
            (Some(current), Some(previous)) => {
                transcript.segments.get(previous).map(|p| p.speaker) != Some(current.speaker)
            }
            (Some(_), None) => true,
            _ => false,
        }
    }

    /// Build one row. Cheap enough to call for a single visible row.
    #[must_use]
    pub fn row(&self, index: usize) -> Option<TranscriptRow> {
        let transcript = self.transcript.as_ref()?;
        let segment = transcript.segments.get(index)?;
        Some(TranscriptRow {
            index,
            timestamp: segment.start.format_hms(),
            speaker: self.speaker_name(index),
            text: segment.text(),
            is_selected: self.selection.contains(index),
            is_turn_start: self.is_turn_start(index),
            is_match: self.matches.contains(&index),
            is_unattributed: segment.speaker.is_none(),
        })
    }

    /// Build rows for `range`, skipping indices that do not exist.
    #[must_use]
    pub fn rows(&self, range: std::ops::Range<usize>) -> Vec<TranscriptRow> {
        let last = self.row_count();
        range
            .filter(|&i| i < last)
            .filter_map(|i| self.row(i))
            .collect()
    }

    /// Select one segment, clearing any range.
    pub fn select(&mut self, index: usize) {
        if index < self.row_count() {
            self.selection = Selection::single(index);
        }
    }

    /// Select a range from the current anchor to `index`.
    pub fn extend_to(&mut self, index: usize) {
        if index >= self.row_count() {
            return;
        }
        let anchor = self.selection.anchor.unwrap_or(index);
        self.selection.anchor = Some(anchor);
        self.selection.focus = Some(index);
    }

    /// Select everything.
    pub fn select_all(&mut self) {
        if self.row_count() > 0 {
            self.selection = Selection {
                anchor: Some(0),
                focus: Some(self.row_count() - 1),
            };
        }
    }

    /// Clear the selection.
    pub fn clear_selection(&mut self) {
        self.selection = Selection::none();
    }

    /// Move the selection by `delta` segments, clamped to the ends.
    ///
    /// `extend` keeps the anchor where it is, which is what shift-arrow means.
    /// Movement from an empty selection starts at the first segment rather than
    /// doing nothing, because a keyboard user pressing Down on a fresh pane
    /// expects to land somewhere.
    pub fn move_selection(&mut self, delta: isize, extend: bool) {
        let count = self.row_count();
        if count == 0 {
            return;
        }
        let current = self.selection.focus.unwrap_or(0);
        let next = (current as isize + delta).clamp(0, count as isize - 1) as usize;
        if extend {
            let anchor = self.selection.anchor.unwrap_or(current);
            self.selection = Selection {
                anchor: Some(anchor),
                focus: Some(next),
            };
        } else {
            self.selection = Selection::single(next);
        }
        if !self.matches.is_empty() {
            self.match_cursor = self.matches.iter().position(|&m| m >= next);
        }
    }

    /// Set the search string and recompute matches.
    ///
    /// Case-insensitive substring match, deliberately: a transcript tool with
    /// no regex is still a transcript tool, and a user who types a literal
    /// bracket is searching for a bracket.
    pub fn set_query(&mut self, query: String) {
        self.query = query;
        self.matches.clear();
        self.match_cursor = None;
        let needle = self.query.trim().to_lowercase();
        if needle.is_empty() {
            return;
        }
        if let Some(transcript) = self.transcript.as_ref() {
            for (index, segment) in transcript.segments.iter().enumerate() {
                if segment.text().to_lowercase().contains(&needle) {
                    self.matches.push(index);
                }
            }
        }
    }

    /// The next match at or after the current focus, wrapping.
    pub fn next_match(&mut self) -> Option<usize> {
        if self.matches.is_empty() {
            return None;
        }
        let from = self.selection.focus.unwrap_or(0);
        let next = self.matches.iter().position(|&m| m >= from).unwrap_or(0);
        self.match_cursor = Some(next);
        let index = self.matches[next];
        self.select(index);
        Some(index)
    }

    /// The previous match before the current focus, wrapping.
    pub fn previous_match(&mut self) -> Option<usize> {
        if self.matches.is_empty() {
            return None;
        }
        let from = self.selection.focus.unwrap_or(usize::MAX);
        let prev = self
            .matches
            .iter()
            .rposition(|&m| m < from)
            .unwrap_or(self.matches.len() - 1);
        self.match_cursor = Some(prev);
        let index = self.matches[prev];
        self.select(index);
        Some(index)
    }

    /// The full text of the selection, or of the whole transcript when nothing
    /// is selected, with timestamps and speaker names.
    ///
    /// Timestamps are included because the overwhelmingly common reason to copy
    /// out of a meeting transcript is to quote it, and a quote without a time
    /// reference cannot be found again.
    #[must_use]
    pub fn copy_text(&self) -> String {
        let Some(transcript) = self.transcript.as_ref() else {
            return String::new();
        };
        let range = self.selection.range();
        let mut out = String::new();
        for (index, segment) in transcript.segments.iter().enumerate() {
            if let Some((lo, hi)) = range
                && !(index >= lo && index <= hi)
            {
                continue;
            }
            let text = segment.text();
            if text.trim().is_empty() {
                continue;
            }
            let speaker = self.speaker_name(index);
            if !speaker.is_empty() {
                let _ = write!(out, "{}  {}: ", segment.start.format_hms(), speaker);
            } else {
                let _ = write!(out, "{}  ", segment.start.format_hms());
            }
            out.push_str(&text);
            out.push('\n');
        }
        out.trim_end().to_owned()
    }

    /// The full text of one segment, unabridged, for the detail pane.
    #[must_use]
    pub fn detail_text(&self, index: usize) -> String {
        let Some(segment) = self.segment(index) else {
            return String::new();
        };
        let mut out = String::new();
        let _ = write!(
            out,
            "{}–{}",
            segment.start.format_hms(),
            segment.end.format_hms()
        );
        out.push('\n');
        let speaker = self.speaker_name(index);
        if !speaker.is_empty() {
            let _ = writeln!(out, "{speaker}");
        }
        out.push_str(&segment.text());
        out
    }

    /// A one-line summary for the transcript pane's header.
    #[must_use]
    pub fn summary(&self) -> String {
        let Some(transcript) = self.transcript.as_ref() else {
            return "No transcript yet".to_owned();
        };
        let segments = transcript.segments.len();
        let words = self.word_count();
        let mut out = format!(
            "{} · {} · {}",
            format_word_count(words),
            plural(segments, "segment", "segments"),
            plural(transcript.speakers.len(), "speaker", "speakers")
        );
        if !self.matches.is_empty() {
            let _ = write!(
                out,
                " · {} of {}",
                plural(self.matches.len(), "match", "matches"),
                self.query
            );
        }
        out
    }

    /// Whether the detail pane has anything to show.
    #[must_use]
    pub fn detail(&self) -> Option<String> {
        let index = self.selection.focus?;
        let text = self.detail_text(index);
        (!text.is_empty()).then_some(text)
    }
}

fn plural(n: usize, one: &'static str, many: &'static str) -> String {
    if n == 1 {
        format!("{n} {one}")
    } else {
        format!("{n} {many}")
    }
}

// ---------------------------------------------------------------------------
// Speakers
// ---------------------------------------------------------------------------

/// One row of the speaker panel.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerRow {
    /// Cluster index, 0-based. Displayed as `index + 1`.
    pub index: int_display::Index,
    /// The name to show: the user's, or `Speaker N`.
    pub name: String,
    /// `Speaker 2` style label, shown greyed while a custom name is set so the
    /// mapping back to the cluster index is never lost.
    pub generated_name: String,
    /// `4m 12s` of attributed speech.
    pub talk_time: String,
    /// `23%`, or empty when there is nothing to divide by.
    pub share_text: String,
    /// `share` as a fraction in `[0, 1]`, for the bar width.
    pub share: Option<f32>,
    /// Whether this row is selected.
    pub is_selected: bool,
    /// Whether the rename editor is open on this row.
    pub is_editing: bool,
    /// Whether a custom name is set.
    pub is_named: bool,
}

/// Namespace for the 0-based speaker index type, so a speaker index can never
/// be passed where a job index is expected.
mod int_display {
    /// A speaker's cluster index.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Index(pub usize);

    impl std::fmt::Display for Index {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0 + 1)
        }
    }
}

pub use int_display::Index as SpeakerIndex;

/// An in-progress rename.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeakerEdit {
    /// Which speaker is being renamed.
    pub index: usize,
    /// The text in the editor.
    pub draft: String,
    /// The name before the edit, so an unchanged commit can be detected.
    pub original: Option<String>,
}

/// The speaker panel's state.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpeakerPanel {
    speakers: Vec<SpeakerSummary>,
    selected: Option<usize>,
    edit: Option<SpeakerEdit>,
}

impl SpeakerPanel {
    /// An empty panel.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the speaker list, dropping an edit that no longer applies.
    pub fn set_speakers(&mut self, speakers: Vec<SpeakerSummary>) {
        if let Some(edit) = &self.edit
            && !speakers.iter().any(|s| s.index == edit.index)
        {
            self.edit = None;
        }
        self.speakers = speakers;
    }

    /// The speakers.
    #[must_use]
    pub fn speakers(&self) -> &[SpeakerSummary] {
        &self.speakers
    }

    /// The selected speaker index.
    #[must_use]
    pub fn selected(&self) -> Option<usize> {
        self.selected
    }

    /// Select a speaker. Out-of-range indices are ignored rather than clamped,
    /// so a stale click cannot silently select a different person.
    pub fn select(&mut self, index: usize) -> bool {
        if self.speakers.iter().any(|s| s.index == index) {
            self.selected = Some(index);
            true
        } else {
            false
        }
    }

    /// Select the first speaker if none is selected.
    pub fn select_default(&mut self) {
        if self.selected.is_none()
            && let Some(first) = self.speakers.first()
        {
            self.selected = Some(first.index);
        }
    }

    /// The open edit, if any.
    #[must_use]
    pub fn edit(&self) -> Option<&SpeakerEdit> {
        self.edit.as_ref()
    }

    /// Open the rename editor on `index`, seeded with the current name.
    ///
    /// Returns `false` if that speaker does not exist.
    pub fn begin_rename(&mut self, index: usize) -> bool {
        let Some(speaker) = self.speakers.iter().find(|s| s.index == index) else {
            return false;
        };
        self.selected = Some(index);
        self.edit = Some(SpeakerEdit {
            index,
            draft: speaker.name.clone().unwrap_or_default(),
            original: speaker.name.clone(),
        });
        true
    }

    /// Replace the editor's contents.
    pub fn set_draft(&mut self, draft: String) {
        if let Some(edit) = self.edit.as_mut() {
            edit.draft = draft;
        }
    }

    /// Abandon the edit.
    pub fn cancel_rename(&mut self) {
        self.edit = None;
    }

    /// Finish the edit.
    ///
    /// Returns the `(index, name)` pair to send to the backend, or `None` when
    /// there is no edit or when the name is unchanged. A blank draft means
    /// "restore the generated label", which the backend expresses as
    /// `name: None`.
    ///
    /// Returning `None` for an unchanged name is what stops a rename from
    /// being written to the database every time someone presses Enter.
    pub fn commit_rename(&mut self) -> Option<(usize, Option<String>)> {
        let edit = self.edit.take()?;
        let trimmed = edit.draft.trim();
        let new_name = (!trimmed.is_empty()).then(|| trimmed.to_owned());
        if new_name == edit.original {
            return None;
        }
        // Apply locally at once. The backend persists asynchronously and sends
        // `SpeakersChanged` or `RenameRejected`; showing the new name
        // immediately is what makes renaming feel like it worked.
        if let Some(speaker) = self.speakers.iter_mut().find(|s| s.index == edit.index) {
            speaker.name = new_name.clone();
        }
        Some((edit.index, new_name))
    }

    /// Roll a rename back after the backend rejected it.
    pub fn revert_rename(&mut self, index: usize, to: Option<String>) {
        if let Some(speaker) = self.speakers.iter_mut().find(|s| s.index == index) {
            speaker.name = to;
        }
    }

    /// The display rows.
    #[must_use]
    pub fn rows(&self) -> Vec<SpeakerRow> {
        let edit_index = self.edit.as_ref().map(|e| e.index);
        self.speakers
            .iter()
            .map(|speaker| SpeakerRow {
                index: SpeakerIndex(speaker.index),
                name: speaker.display_name(),
                generated_name: format!("Speaker {}", speaker.index + 1),
                talk_time: format_millis(speaker.talk_time_ms),
                share_text: format_percent(speaker.share.map(f64::from)),
                share: speaker.share.map(|s| s.clamp(0.0, 1.0)),
                is_selected: self.selected == Some(speaker.index),
                is_editing: edit_index == Some(speaker.index),
                is_named: speaker
                    .name
                    .as_deref()
                    .is_some_and(|n| !n.trim().is_empty()),
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Error surface
// ---------------------------------------------------------------------------

/// The error banner.
///
/// One banner, not a list. A stack of toasts is a way of hiding the one error
/// that matters, and this product's errors are almost always the reason
/// something did not work, which the user needs to read.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Banner {
    /// Whether to show it at all.
    pub visible: bool,
    /// How loud it is.
    pub level: Option<Severity>,
    /// Short title.
    pub title: String,
    /// The explanation.
    pub detail: String,
    /// What to do about it.
    pub recovery: String,
    /// The stable code, shown small so a bug report can quote it.
    pub code: String,
    /// Whether the banner can be dismissed.
    pub dismissible: bool,
}

impl Banner {
    /// Hidden.
    #[must_use]
    pub fn hidden() -> Self {
        Self::default()
    }

    /// Show a user-facing error.
    #[must_use]
    pub fn from_error(level: Severity, error: &vaani_core::UserFacingError) -> Self {
        Self {
            visible: true,
            level: Some(level),
            title: error.title.clone(),
            detail: error.detail.clone(),
            recovery: error.recovery.clone().unwrap_or_default(),
            code: error.code.as_str().to_owned(),
            dismissible: true,
        }
    }

    /// Show a plain notice.
    #[must_use]
    pub fn notice(level: Severity, text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            visible: true,
            level: Some(level),
            title: text.clone(),
            detail: String::new(),
            recovery: String::new(),
            code: String::new(),
            dismissible: true,
        }
    }

    /// Hide it.
    pub fn dismiss(&mut self) {
        *self = Self::hidden();
    }
}

// ---------------------------------------------------------------------------
// Whole-window state
// ---------------------------------------------------------------------------

/// Everything the window shows that is not a transcript or a job.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AppView {
    /// The path in the file field.
    pub input_path: String,
    /// Whether that path looks like media we accept.
    pub input_accepted: bool,
    /// The backend's name.
    pub backend_name: String,
    /// Whether the backend is a demo, so a banner says so.
    pub is_demo: bool,
    /// The host providing dialogs.
    pub host_name: String,
    /// Settings.
    pub settings: crate::backend::AppSettings,
    /// Whether the settings panel is expanded. Window-local, never persisted
    /// here: whether the panel is open is not a setting, it is a moment.
    pub settings_open: bool,
    /// Whether the history list is the main surface.
    ///
    /// One surface at a time, deliberately. The old layout kept the job list,
    /// the transcript and a right-hand rail on screen at once, and the effect
    /// on a first run was three panels of equal weight and one obvious action
    /// buried in them. History is a place you *go*, not a place you live.
    pub history_open: bool,
    /// Whether the performance panel is showing.
    pub performance_open: bool,
    /// The current error surface.
    pub banner: Banner,
    /// Selectable models, in the order the backend listed them.
    pub models: Vec<Choice>,
    /// Selectable devices.
    pub devices: Vec<Choice>,
    /// Selectable languages.
    pub languages: Vec<Choice>,
    /// How many events the UI has applied. Diagnostics only.
    pub events_applied: u64,
    /// How many events were dropped by coalescing. Diagnostics only.
    pub events_coalesced: u64,
    /// The last few things that happened, newest last, for the footer's log.
    pub recent: VecDeque<String>,
}

/// How many lines the footer's activity log keeps. Enough to be useful, few
/// enough that the footer is not a second window.
const RECENT_LIMIT: usize = 4;

impl AppView {
    /// A window with nothing in it yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record something that happened, for the footer's activity log.
    pub fn note(&mut self, text: impl Into<String>) {
        self.recent.push_back(text.into());
        while self.recent.len() > RECENT_LIMIT {
            self.recent.pop_front();
        }
    }

    /// The drop-down labels for a choice list.
    #[must_use]
    pub fn labels(choices: &[Choice]) -> Vec<String> {
        choices
            .iter()
            .map(|c| match &c.detail {
                Some(detail) if !detail.is_empty() => format!("{} ({detail})", c.label),
                _ => c.label.clone(),
            })
            .collect()
    }

    /// The index of the choice with this id, or `-1`.
    ///
    /// A setting the backend has not listed is `-1` rather than `0`: silently
    /// showing the first entry for a value that is not selected would be a lie
    /// about which model is about to run.
    #[must_use]
    pub fn choice_index(choices: &[Choice], id: &str) -> i32 {
        choices
            .iter()
            .position(|c| c.id == id)
            .map_or(-1, |i| i as i32)
    }

    /// Model drop-down labels.
    #[must_use]
    pub fn model_labels(&self) -> Vec<String> {
        Self::labels(&self.models)
    }

    /// Device drop-down labels.
    #[must_use]
    pub fn device_labels(&self) -> Vec<String> {
        Self::labels(&self.devices)
    }

    /// Language drop-down labels.
    #[must_use]
    pub fn language_labels(&self) -> Vec<String> {
        Self::labels(&self.languages)
    }

    /// The id behind a drop-down label, for the `selected` callback.
    ///
    /// Matches the *composed* label as well as the bare one, because that is
    /// what the widget reports. The comparison is per choice, not against the
    /// whole list: matching "is this string anywhere in the list" would make the
    /// first entry match every label, which is a bug this function had and this
    /// comment is the scar from.
    #[must_use]
    pub fn choice_id(choices: &[Choice], label: &str) -> Option<String> {
        choices
            .iter()
            .find(|c| {
                c.label == label
                    || c.detail
                        .as_deref()
                        .is_some_and(|d| !d.is_empty() && format!("{} ({d})", c.label) == label)
            })
            .map(|c| c.id.clone())
    }

    /// Whether a job can be enqueued from the current file field.
    #[must_use]
    pub fn can_enqueue(&self) -> bool {
        !self.input_path.trim().is_empty() && self.input_accepted
    }

    /// The status line above the progress bar.
    #[must_use]
    pub fn activity(&self, jobs: &JobListState) -> String {
        let running = jobs.running_count();
        let total = jobs.jobs().len();
        if total == 0 {
            return "No jobs yet. Choose a file to transcribe.".to_owned();
        }
        if running > 0 {
            let queued = jobs
                .jobs()
                .iter()
                .filter(|j| j.phase == JobPhase::Queued)
                .count();
            let mut out = jobs.selected_job().map_or_else(
                || format!("{running} job(s) running"),
                |job| format!("{} - {}", file_name_of(&job.input), phase_label(job.phase)),
            );
            if queued > 0 {
                let _ = write!(out, " - {queued} waiting");
            }
            let _ = write!(out, " - {running} of {total} active");
            return out;
        }
        format!("{total} job(s), none running")
    }

    /// The value and the label for the big progress bar.
    ///
    /// `None` means *indeterminate*, which the bar renders as a sweep. That is
    /// only ever correct while a job is genuinely running: a sweep is an
    /// animation, an animation ticks at the display's refresh rate, and a window
    /// that sweeps forever costs a measurable slice of a core doing nothing.
    /// A window with three finished jobs in it must be idle.
    #[must_use]
    pub fn overall_progress(&self, jobs: &JobListState) -> (Option<f64>, String) {
        let Some(job) = jobs.selected_job() else {
            return (Some(0.0), String::new());
        };
        let fraction = job.overall.map(|f| f.clamp(0.0, 1.0));
        let running = job.phase.is_active();
        let shown = if running && fraction.is_none() {
            None
        } else {
            Some(fraction.unwrap_or(0.0))
        };
        let text = if running && fraction.is_some() {
            format_percent(fraction)
        } else {
            phase_label(job.phase).to_owned()
        };
        (shown, text)
    }

    /// Set the file field, recomputing whether it looks like media.
    pub fn set_input_path(&mut self, path: String) {
        let trimmed = path.trim();
        self.input_accepted = !trimmed.is_empty()
            && vaani_core::has_supported_extension(std::path::Path::new(trimmed));
        self.input_path = path;
    }

    /// Set the theme locally and produce the patch to send.
    pub fn set_theme(&mut self, theme: Theme) -> SettingsPatch {
        self.settings.theme = theme;
        SettingsPatch {
            theme: Some(theme),
            ..Default::default()
        }
    }
}

/// Maximum characters of a segment shown before the detail pane is used.
pub const DETAIL_CHAR_BUDGET: usize = MAX_DETAIL_CHARS;

#[cfg(test)]
mod tests {
    use super::*;
    use vaani_core::{ErrorCode, UserFacingError};

    fn job(id: &str, phase: JobPhase) -> JobSummary {
        JobSummary {
            id: JobId::from(id),
            input: std::path::PathBuf::from(format!("C:/media/{id}.mp4")),
            phase,
            overall: None,
            detail: None,
            eta: None,
            error: None,
            has_transcript: false,
            cancel_requested: false,
            word_count: None,
            device: None,
            timings: None,
        }
    }

    fn err() -> UserFacingError {
        UserFacingError::new(
            ErrorCode::ModelUnavailable,
            "Model unavailable.",
            "Download it.",
        )
        .with_recovery("Open Settings and download the model.")
    }

    #[test]
    fn durations_are_human_readable_at_every_scale() {
        assert_eq!(format_millis(0), "0s");
        assert_eq!(format_millis(45_000), "45s");
        assert_eq!(format_millis(65_000), "1m 05s");
        assert_eq!(format_millis(3_600_000), "1h 00m");
        assert_eq!(format_millis(7_380_000), "2h 03m");
        assert_eq!(format_millis(-5), "0s");
    }

    #[test]
    fn word_counts_get_thousands_separators() {
        assert_eq!(format_word_count(0), "0 words");
        assert_eq!(format_word_count(1), "1 word");
        assert_eq!(format_word_count(999), "999 words");
        assert_eq!(format_word_count(1_000), "1,000 words");
        assert_eq!(format_word_count(20_000), "20,000 words");
        assert_eq!(format_word_count(123_456), "123,456 words");
    }

    #[test]
    fn progress_out_of_range_is_clamped_not_wrapped() {
        let mut j = job("a", JobPhase::Asr);
        j.overall = Some(1.7);
        let list = JobListState::from_jobs(vec![j]);
        let row = &list.rows()[0];
        assert_eq!(row.progress, Some(1.0));
        assert_eq!(row.progress_text, "100%");

        let mut j = job("a", JobPhase::Asr);
        j.overall = Some(-0.4);
        let list = JobListState::from_jobs(vec![j]);
        assert_eq!(list.rows()[0].progress, Some(0.0));
    }

    #[test]
    fn indeterminate_progress_renders_no_percentage() {
        // The whole point of `Progress::overall() -> Option<f64>`.
        let list = JobListState::from_jobs(vec![job("a", JobPhase::Asr)]);
        let row = &list.rows()[0];
        assert_eq!(row.progress, None);
        assert_eq!(row.progress_text, "");
        assert_eq!(row.phase_label, "Transcribing");
    }

    #[test]
    fn every_phase_renders_a_distinct_row() {
        // Covers the whole state machine, including the two the UI is most
        // likely to get wrong: Cancelled and Failed.
        let mut seen = std::collections::HashSet::new();
        for phase in ALL_PHASES_LOCAL {
            let list = JobListState::from_jobs(vec![job("a", phase)]);
            let row = list.rows().remove(0);
            assert!(!row.status_text.is_empty(), "{phase} has no status text");
            assert!(!row.phase_code.is_empty());
            assert!(
                seen.insert((row.phase_code.clone(), row.status_text.clone())),
                "{phase} renders identically to another phase"
            );
            if phase.is_terminal() {
                assert!(row.is_terminal, "{phase} should be terminal");
                assert!(!row.is_running, "{phase} should not be running");
            } else if phase == JobPhase::Queued {
                assert!(row.is_queued);
            } else {
                assert!(row.is_running, "{phase} should be running");
            }
        }
    }

    const ALL_PHASES_LOCAL: [JobPhase; 14] = crate::backend::ALL_PHASES;

    #[test]
    fn cancelled_and_failed_are_told_apart() {
        let cancelled = JobListState::from_jobs(vec![job("a", JobPhase::Cancelled)]);
        let row = &cancelled.rows()[0];
        assert!(row.is_cancelled && !row.is_failed && !row.is_complete);
        assert_eq!(row.status_text, "Cancelled");

        let mut failed_job = job("b", JobPhase::Failed);
        failed_job.error = Some(err());
        let failed = JobListState::from_jobs(vec![failed_job]);
        let row = &failed.rows()[0];
        assert!(row.is_failed && !row.is_cancelled);
        assert_eq!(row.status_text, "Model unavailable.");
        assert_eq!(row.error_code, "model_unavailable");
        assert!(!row.error_recovery.is_empty());
    }

    #[test]
    fn a_cancel_request_shows_as_cancelling_not_as_running() {
        let mut j = job("a", JobPhase::Asr);
        j.cancel_requested = true;
        let list = JobListState::from_jobs(vec![j]);
        let row = &list.rows()[0];
        assert!(row.is_cancelling);
        assert_eq!(row.status_text, "Cancelling…");
    }

    #[test]
    fn cancel_requested_on_a_finished_job_is_ignored() {
        let mut j = job("a", JobPhase::Complete);
        j.cancel_requested = true;
        let list = JobListState::from_jobs(vec![j]);
        assert!(!list.rows()[0].is_cancelling);
    }

    #[test]
    fn a_detail_string_does_not_displace_the_phase_name() {
        let mut j = job("a", JobPhase::Asr);
        j.detail = Some("chunk 12 / 48".to_owned());
        let list = JobListState::from_jobs(vec![j]);
        let row = &list.rows()[0];
        assert_eq!(row.status_text, "chunk 12 / 48");
        // The phase is still in the row, for anyone who needs the code.
        assert_eq!(row.phase_label, "Transcribing");
    }

    #[test]
    fn blank_detail_falls_back_to_the_phase_name() {
        let mut j = job("a", JobPhase::Asr);
        j.detail = Some("   ".to_owned());
        let list = JobListState::from_jobs(vec![j]);
        assert_eq!(list.rows()[0].status_text, "Transcribing");
    }

    #[test]
    fn the_list_is_newest_first_and_updates_do_not_reorder_it() {
        let mut list = JobListState::new();
        list.upsert(job("a", JobPhase::Queued));
        list.upsert(job("b", JobPhase::Queued));
        list.upsert(job("c", JobPhase::Queued));
        assert_eq!(
            list.rows()
                .iter()
                .map(|r| r.id.as_str())
                .collect::<Vec<_>>(),
            vec!["c", "b", "a"]
        );
        // An update to the oldest job must not float it to the top.
        list.upsert(job("a", JobPhase::Asr));
        assert_eq!(
            list.rows()
                .iter()
                .map(|r| r.id.as_str())
                .collect::<Vec<_>>(),
            vec!["c", "b", "a"]
        );
    }

    #[test]
    fn selection_survives_an_update_and_follows_a_removal() {
        let mut list =
            JobListState::from_jobs(vec![job("a", JobPhase::Queued), job("b", JobPhase::Queued)]);
        assert_eq!(list.selected().map(JobId::as_str), Some("b"));
        list.upsert(job("b", JobPhase::Asr));
        assert_eq!(list.selected().map(JobId::as_str), Some("b"));
        assert!(list.remove(&JobId::from("b")));
        assert_eq!(list.selected().map(JobId::as_str), Some("a"));
    }

    #[test]
    fn selecting_a_job_that_is_not_there_fails_loudly() {
        let mut list = JobListState::from_jobs(vec![job("a", JobPhase::Queued)]);
        assert!(!list.select(&JobId::from("nope")));
        assert_eq!(list.selected().map(JobId::as_str), Some("a"));
    }

    #[test]
    fn coalescing_keeps_only_the_last_snapshot_per_job() {
        let mut j = job("a", JobPhase::Asr);
        j.overall = Some(0.1);
        let first = ControllerEvent::JobChanged(j.clone());
        j.overall = Some(0.2);
        let second = ControllerEvent::JobChanged(j.clone());
        j.overall = Some(0.9);
        let third = ControllerEvent::JobChanged(j);
        let out = coalesce(vec![
            first,
            ControllerEvent::Notice {
                level: Severity::Info,
                text: "a".into(),
            },
            second,
            third,
        ]);
        assert_eq!(out.len(), 2, "one snapshot plus one notice");
        let ControllerEvent::JobChanged(last) = &out[0] else {
            panic!("expected a job change");
        };
        assert_eq!(last.overall, Some(0.9));
    }

    #[test]
    fn coalescing_never_drops_a_removal_or_an_error() {
        let mut failed = job("a", JobPhase::Failed);
        failed.error = Some(err());
        let out = coalesce(vec![
            ControllerEvent::JobChanged(job("a", JobPhase::Asr)),
            ControllerEvent::JobRemoved(JobId::from("b")),
            ControllerEvent::JobChanged(failed),
            ControllerEvent::ExportFailed {
                job: JobId::from("a"),
                error: err(),
            },
            ControllerEvent::TranscriptReady(JobId::from("a")),
        ]);
        assert_eq!(out.len(), 4);
        assert!(matches!(out[0], ControllerEvent::JobChanged(_)));
        assert!(matches!(out[1], ControllerEvent::JobRemoved(_)));
        assert!(matches!(out[2], ControllerEvent::ExportFailed { .. }));
        assert!(matches!(out[3], ControllerEvent::TranscriptReady(_)));
    }

    #[test]
    fn coalescing_an_empty_burst_is_empty() {
        assert!(coalesce(Vec::new()).is_empty());
    }

    fn sample_view() -> TranscriptView {
        let mut view = TranscriptView::new();
        view.set_transcript(Some(crate::sample::sample_transcript_arc(300, 2)));
        view
    }

    #[test]
    fn transcript_rows_carry_timestamps_and_speakers() {
        let view = sample_view();
        let row = view.row(0).unwrap();
        assert_eq!(row.index, 0);
        assert_eq!(row.timestamp, "00:00");
        assert_eq!(row.speaker, "Speaker 1");
        assert!(row.text.starts_with(crate::sample::SAMPLE_SEGMENT_PREFIX));
        assert!(row.is_turn_start);
    }

    #[test]
    fn transcript_rows_out_of_range_are_none_not_a_panic() {
        let view = sample_view();
        assert!(view.row(view.row_count()).is_none());
        assert!(view.row(usize::MAX).is_none());
        assert!(
            view.rows(view.row_count()..view.row_count() + 10)
                .is_empty()
        );
    }

    #[test]
    fn speaker_runs_are_grouped_visually() {
        // One speaker: everything is one run, so only the first row starts a
        // turn. This is the case that decides whether a reader sees a meeting as
        // paragraphs or as a wall of alternating one-line fragments.
        let mut single = TranscriptView::new();
        single.set_transcript(Some(Arc::new(crate::sample::sample_transcript(300, 1))));
        assert!(single.row(0).unwrap().is_turn_start);
        assert!(
            !single.row(1).unwrap().is_turn_start,
            "one speaker means one run"
        );
        assert!(!single.row(5).unwrap().is_turn_start);

        // Two speakers alternate, so every row is a turn start. Asserting the
        // rotation rather than assuming it is what caught the earlier version of
        // this test being wrong about its own fixture.
        let two = sample_view();
        assert!(two.row(0).unwrap().is_turn_start);
        assert!(two.row(1).unwrap().is_turn_start, "the speaker changed");
        assert_ne!(two.speaker_name(0), two.speaker_name(1));
    }

    #[test]
    fn selection_clicks_and_shift_clicks_behave() {
        let mut view = sample_view();
        view.select(3);
        assert!(view.row(3).unwrap().is_selected);
        assert!(!view.row(4).unwrap().is_selected);

        view.extend_to(6);
        assert_eq!(view.selection().range(), Some((3, 6)));
        for i in 3..=6 {
            assert!(view.row(i).unwrap().is_selected, "segment {i} not selected");
        }
        assert!(!view.row(7).unwrap().is_selected);

        // Extending backwards normalises rather than inverting.
        view.extend_to(1);
        assert_eq!(view.selection().range(), Some((1, 3)));
    }

    #[test]
    fn selection_arrows_clamp_at_both_ends() {
        let mut view = sample_view();
        view.move_selection(-1, false);
        assert_eq!(view.selection().focus, Some(0));
        view.move_selection(-5, false);
        assert_eq!(view.selection().focus, Some(0));
        view.move_selection(1_000, false);
        assert_eq!(view.selection().focus, Some(view.row_count() - 1));
    }

    #[test]
    fn extending_with_shift_keeps_the_anchor() {
        let mut view = sample_view();
        view.select(5);
        view.move_selection(2, true);
        assert_eq!(view.selection().range(), Some((5, 7)));
        view.move_selection(-4, true);
        assert_eq!(view.selection().range(), Some((3, 5)));
    }

    #[test]
    fn selecting_a_row_that_does_not_exist_is_ignored() {
        let mut view = sample_view();
        let before = view.selection();
        view.select(999_999);
        view.extend_to(999_999);
        assert_eq!(view.selection(), before);
    }

    #[test]
    fn copy_covers_the_selection_with_timestamps_and_speakers() {
        let mut view = sample_view();
        view.select(0);
        view.extend_to(1);
        let text = view.copy_text();
        assert!(text.contains("00:00  Speaker 1:"), "{text}");
        assert_eq!(text.lines().count(), 2);
    }

    #[test]
    fn copy_with_no_selection_covers_the_whole_transcript() {
        let view = sample_view();
        let text = view.copy_text();
        assert_eq!(text.lines().count(), view.row_count());
    }

    #[test]
    fn copy_of_an_empty_transcript_is_empty_not_a_panic() {
        let view = TranscriptView::new();
        assert!(view.copy_text().is_empty());
        assert_eq!(view.row_count(), 0);
        assert_eq!(view.summary(), "No transcript yet");
    }

    #[test]
    fn search_finds_matches_case_insensitively_and_wraps() {
        let mut view = sample_view();
        view.set_query("SEGMENT 3".to_lowercase());
        assert!(!view.matches().is_empty());
        let first = view.next_match().unwrap();
        assert!(view.matches().contains(&first));
        // Wrapping around the end must still find something.
        view.select(view.row_count() - 1);
        assert!(view.next_match().is_some());
        assert!(view.previous_match().is_some());
    }

    #[test]
    fn search_with_no_hits_reports_nothing_and_does_not_panic() {
        let mut view = sample_view();
        view.set_query("zzzz-not-present".to_owned());
        assert!(view.matches().is_empty());
        assert_eq!(view.next_match(), None);
        assert_eq!(view.previous_match(), None);
    }

    #[test]
    fn clearing_the_search_clears_the_matches() {
        let mut view = sample_view();
        view.set_query("alpha".to_owned());
        assert!(!view.matches().is_empty());
        view.set_query(String::new());
        assert!(view.matches().is_empty());
    }

    #[test]
    fn summary_counts_words_segments_and_speakers() {
        let view = sample_view();
        let summary = view.summary();
        assert!(summary.contains("segment"), "{summary}");
        assert!(summary.contains("speaker"), "{summary}");
    }

    fn speaker(index: usize, name: Option<&str>) -> SpeakerSummary {
        SpeakerSummary {
            index,
            name: name.map(str::to_owned),
            talk_time_ms: 60_000 * (index as i64 + 1),
            share: Some(0.25),
        }
    }

    #[test]
    fn speakers_show_talk_time_and_share() {
        let mut panel = SpeakerPanel::new();
        panel.set_speakers(vec![speaker(0, None), speaker(1, Some("Priya"))]);
        panel.select_default();
        let rows = panel.rows();
        assert_eq!(rows[0].name, "Speaker 1");
        assert_eq!(rows[0].talk_time, "1m 00s");
        assert_eq!(rows[0].share_text, "25%");
        assert!(rows[0].is_selected);
        assert!(!rows[0].is_named);
        assert_eq!(rows[1].name, "Priya");
        assert!(rows[1].is_named);
        assert_eq!(rows[1].generated_name, "Speaker 2");
    }

    #[test]
    fn renaming_a_speaker_emits_one_event_and_never_touches_text() {
        let mut panel = SpeakerPanel::new();
        panel.set_speakers(vec![speaker(0, None)]);
        assert!(panel.begin_rename(0));
        panel.set_draft("  Priya  ".to_owned());
        let emitted = panel.commit_rename();
        assert_eq!(emitted, Some((0, Some("Priya".to_owned()))), "trimmed");
        assert_eq!(panel.rows()[0].name, "Priya");
    }

    #[test]
    fn renaming_to_the_same_name_emits_nothing() {
        let mut panel = SpeakerPanel::new();
        panel.set_speakers(vec![speaker(0, Some("Priya"))]);
        panel.begin_rename(0);
        panel.set_draft("Priya".to_owned());
        assert_eq!(
            panel.commit_rename(),
            None,
            "an unchanged name is not a write"
        );
    }

    #[test]
    fn a_blank_rename_restores_the_generated_label() {
        let mut panel = SpeakerPanel::new();
        panel.set_speakers(vec![speaker(0, Some("Priya"))]);
        panel.begin_rename(0);
        panel.set_draft("   ".to_owned());
        assert_eq!(panel.commit_rename(), Some((0, None)));
        assert_eq!(panel.rows()[0].name, "Speaker 1");
    }

    #[test]
    fn cancelling_a_rename_changes_nothing() {
        let mut panel = SpeakerPanel::new();
        panel.set_speakers(vec![speaker(0, None)]);
        panel.begin_rename(0);
        panel.set_draft("Priya".to_owned());
        panel.cancel_rename();
        assert!(panel.edit().is_none());
        assert_eq!(panel.commit_rename(), None);
        assert_eq!(panel.rows()[0].name, "Speaker 1");
    }

    #[test]
    fn renaming_a_speaker_that_does_not_exist_fails() {
        let mut panel = SpeakerPanel::new();
        panel.set_speakers(vec![speaker(0, None)]);
        assert!(!panel.begin_rename(7));
        assert!(panel.edit().is_none());
    }

    #[test]
    fn a_rejected_rename_can_be_rolled_back() {
        let mut panel = SpeakerPanel::new();
        panel.set_speakers(vec![speaker(0, Some("Priya"))]);
        panel.revert_rename(0, None);
        assert_eq!(panel.rows()[0].name, "Speaker 1");
    }

    #[test]
    fn replacing_the_speaker_list_drops_an_edit_for_a_gone_speaker() {
        let mut panel = SpeakerPanel::new();
        panel.set_speakers(vec![speaker(0, None), speaker(1, None)]);
        panel.begin_rename(1);
        panel.set_draft("Priya".to_owned());
        panel.set_speakers(vec![speaker(0, None)]);
        assert!(
            panel.edit().is_none(),
            "an edit for a vanished speaker must not survive"
        );
    }

    #[test]
    fn banner_shows_the_whole_error_including_recovery() {
        let mut banner = Banner::from_error(Severity::Error, &err());
        assert!(banner.visible);
        assert_eq!(banner.title, "Model unavailable.");
        assert_eq!(banner.code, "model_unavailable");
        assert!(banner.recovery.contains("Settings"));
        banner.dismiss();
        assert!(!banner.visible);
    }

    #[test]
    fn the_file_field_validates_the_extension() {
        let mut app = AppView::new();
        app.set_input_path("C:/media/meeting.mkv".to_owned());
        assert!(app.can_enqueue());
        app.set_input_path("C:/media/notes.txt".to_owned());
        assert!(!app.can_enqueue(), "a text file is not media we accept");
        app.set_input_path(String::new());
        assert!(!app.can_enqueue());
    }

    #[test]
    fn the_activity_line_counts_running_and_waiting_jobs() {
        let app = AppView::new();
        let empty = JobListState::new();
        assert!(app.activity(&empty).contains("No jobs"));

        let list = JobListState::from_jobs(vec![
            job("a", JobPhase::Asr),
            job("b", JobPhase::Queued),
            job("c", JobPhase::Complete),
        ]);
        let line = app.activity(&list);
        assert!(line.contains("1 waiting"), "{line}");
        assert!(line.contains("1 of 3 active"), "{line}");
    }

    #[test]
    fn an_indeterminate_overall_progress_falls_back_to_the_phase_name() {
        let app = AppView::new();
        let list = JobListState::from_jobs(vec![job("a", JobPhase::Asr)]);
        let (fraction, text) = app.overall_progress(&list);
        assert_eq!(fraction, None);
        assert_eq!(text, "Transcribing");
    }

    #[test]
    fn a_finished_job_shows_its_terminal_name_not_a_percentage() {
        let app = AppView::new();
        let mut j = job("a", JobPhase::Complete);
        j.overall = Some(1.0);
        let list = JobListState::from_jobs(vec![j]);
        let (fraction, text) = app.overall_progress(&list);
        assert_eq!(fraction, Some(1.0));
        assert_eq!(text, "Complete");
    }

    fn choice(id: &str, label: &str, detail: Option<&str>) -> Choice {
        Choice {
            id: id.to_owned(),
            label: label.to_owned(),
            detail: detail.map(str::to_owned),
            available: true,
        }
    }

    #[test]
    fn drop_down_labels_carry_the_detail() {
        let choices = vec![
            choice("sravaani", "SraVaani", Some("1.0 GB")),
            choice("tiny", "Tiny", None),
        ];
        assert_eq!(
            AppView::labels(&choices),
            vec!["SraVaani (1.0 GB)".to_owned(), "Tiny".to_owned()]
        );
    }

    #[test]
    fn a_selected_id_maps_back_to_a_drop_down_index_and_label() {
        let choices = vec![choice("a", "Alpha", None), choice("b", "Beta", None)];
        assert_eq!(AppView::choice_index(&choices, "b"), 1);
        assert_eq!(AppView::choice_index(&choices, "missing"), -1);
        assert_eq!(AppView::choice_id(&choices, "Beta").as_deref(), Some("b"));
        assert_eq!(AppView::choice_id(&choices, "Nope"), None);
    }

    #[test]
    fn a_label_with_a_detail_still_resolves_to_its_id() {
        // The widget reports the *label*, detail included, so the lookup has
        // to accept the composed form as well as the bare one.
        let choices = vec![choice("big", "SraVaani", Some("1.0 GB"))];
        assert_eq!(
            AppView::choice_id(&choices, "SraVaani (1.0 GB)").as_deref(),
            Some("big")
        );
    }

    #[test]
    fn the_activity_log_keeps_only_the_last_few_entries() {
        let mut app = AppView::new();
        for i in 0..10 {
            app.note(format!("event {i}"));
        }
        assert_eq!(app.recent.len(), 4);
        assert_eq!(app.recent.back().unwrap(), "event 9");
        assert_eq!(app.recent.front().unwrap(), "event 6");
    }
}

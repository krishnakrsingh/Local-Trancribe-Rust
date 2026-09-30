//! The Rust/Slint boundary.
//!
//! This is the **only** module in the crate that may name a `slint` type.
//! Everything above it works in [`crate::view`]'s plain structs and everything
//! below it is [`crate::backend`]. That one-way rule is what keeps three
//! separate concerns from tangling:
//!
//! * **Threading.** [`UiModel`] records the thread it was constructed on and
//!   asserts, on every property write, that it is still that thread. A worker
//!   thread that tried to touch the window would panic loudly at the point of
//!   the bug instead of corrupting a property nobody was looking at. Progress
//!   arrives from a worker through a crossbeam channel and is applied by
//!   [`UiModel::pump`], which only ever runs on the event loop.
//! * **Backpressure.** A transcription job emits progress far faster than a
//!   display refreshes. [`UiModel::pump`] drains the whole channel, collapses
//!   the burst with [`crate::view::coalesce`] and applies one update per job.
//!   Without that, the window spends its life in `set_property` and the CPU
//!   meter says what a user would say: "it is heating up my laptop".
//! * **Windowing.** The transcript is never materialised. A 20,000-word meeting
//!   is a few thousand segments, and the pane keeps only the rows the viewport
//!   can show plus a small overscan live at a time. The arithmetic is done in
//!   whole text lines ([`TranscriptLayout`]) so this module never needs to know
//!   how tall a line is in pixels, which is the one number that changes with
//!   font, scale factor and platform.

use std::cell::{Cell, Ref, RefCell};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::thread::ThreadId;
use std::time::Duration;

use crossbeam_channel::Receiver;
use slint::language::ColorScheme;
use slint::{
    ComponentHandle, Global as SlintGlobal, Model, ModelRc, SharedString, Timer, TimerMode,
    VecModel, Weak,
};
use vaani_core::{JobPhase, Transcript, UserFacingError};

use crate::backend::{
    AppSettings, Choice, ControllerEvent, ExportFormat, ExportRequest, HostServices, JobController,
    JobId, SettingsPatch, Severity, Theme,
};
use crate::slint_api::{AppWindow, JobRow as JobRowS, SpeakerRow as SpeakerRowS, TranscriptRow};
use crate::view::{self, AppView, Banner, JobListState, SpeakerPanel, TranscriptView};

/// Height of one line of transcript text, in logical pixels.
///
/// Pushed to the window on every commit so the Rust-side arithmetic and the
/// Slint-side geometry can never disagree. The real value depends on the font
/// the user has; this is the number the layout is *built* around, and being
/// approximately right is enough, because rows are measured in lines rather
/// than pixels.
pub const TRANSCRIPT_LINE_HEIGHT_PX: f32 = 22.0;

/// Horizontal padding inside a transcript row, in logical pixels: the selection
/// bar plus the layout's left and right padding.
const ROW_HORIZONTAL_PADDING_PX: f32 = 30.0;

/// Average glyph advance of the transcript body font, in logical pixels per
/// character, used only to estimate how many lines a segment wraps to.
///
/// Getting this wrong costs a little whitespace or an ellipsis, never lost text
/// and never a crash, because the detail pane always holds the full segment.
const PIXELS_PER_CHAR: f32 = 6.9;

/// Never estimate fewer characters than this per line, whatever the width. A
/// pane narrower than this is being resized to nothing, and a one-character line
/// would make every segment look like a paragraph.
const MIN_CHARS_PER_LINE: usize = 20;

/// Never estimate more than this many lines for one segment. A pathological
/// input (a segment with no spaces at all) would otherwise claim the entire
/// transcript's scroll height on its own.
const MAX_LINES_PER_SEGMENT: u16 = 40;

/// Extra rows kept live above and below the viewport, so a flick does not show
/// blank space while the next window is built.
const OVERSCAN_ROWS: usize = 8;

/// How often [`UiModel::pump`] runs. Faster than any display's redraw and slow
/// enough to be invisible in a profile.
pub const PUMP_INTERVAL: Duration = Duration::from_millis(33);

/// How many events to apply in one pump before leaving the rest for the next
/// tick.
///
/// A ceiling, so a backend that emits in a tight loop cannot make the UI thread
/// permanently unavailable. Nothing is dropped: the overflow stays in the
/// channel and is picked up on the next tick.
const MAX_EVENTS_PER_PUMP: usize = 4_096;

/// Row-count hysteresis for relayout, in characters per line.
///
/// Dragging a window edge produces a `viewport-changed` per frame. Re-deriving
/// every segment's line count on each one is wasted work, so a small change in
/// width is ignored until it is big enough to change the answer.
const RELAYOUT_HYSTERESIS: i64 = 3;

// ---------------------------------------------------------------------------
// Line layout
// ---------------------------------------------------------------------------

/// Per-segment line counts and their prefix sums, for windowing a transcript.
///
/// The unit is *text lines*, not pixels, for one reason: Slint decides the
/// height of a line from the font, and Rust must not hold a second opinion. If
/// the two ever disagreed, the scrollbar would lie, and a scrollbar that lies
/// is the single most disorienting bug a long document can have.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TranscriptLayout {
    chars_per_line: usize,
    /// Lines occupied by each segment.
    lines: Vec<u16>,
    /// Lines occupied by all segments before each one. Same length as `lines`.
    prefix: Vec<u32>,
    total: u32,
}

impl TranscriptLayout {
    /// Derive a layout for `transcript` at `chars_per_line` characters per line.
    #[must_use]
    pub fn build(transcript: Option<&Transcript>, chars_per_line: usize) -> Self {
        let mut layout = Self {
            chars_per_line: chars_per_line.max(MIN_CHARS_PER_LINE),
            ..Default::default()
        };
        let chars = layout.chars_per_line;
        layout.relayout(transcript, chars);
        layout
    }

    /// Re-derive the layout, skipping a width change too small to matter.
    pub fn relayout(&mut self, transcript: Option<&Transcript>, chars_per_line: usize) {
        let chars_per_line = chars_per_line.max(MIN_CHARS_PER_LINE);
        let Some(transcript) = transcript else {
            self.chars_per_line = chars_per_line;
            self.lines.clear();
            self.prefix.clear();
            self.total = 0;
            return;
        };
        if !self.lines.is_empty()
            && (i64::try_from(chars_per_line).unwrap_or(i64::MAX)
                - i64::try_from(self.chars_per_line).unwrap_or(0))
            .abs()
                < RELAYOUT_HYSTERESIS
        {
            return;
        }
        self.chars_per_line = chars_per_line;
        self.lines.clear();
        self.prefix.clear();
        self.total = 0;
        self.lines.reserve(transcript.segments.len());
        self.prefix.reserve(transcript.segments.len());
        for segment in &transcript.segments {
            self.prefix.push(self.total);
            let chars = segment.text().chars().count();
            // Round up, and never report zero: an empty-looking row cannot be
            // clicked and a zero-height row cannot be scrolled to.
            //
            // Plus one for the header line. Every row renders timestamp and
            // speaker above the text, and that header occupies a full line
            // height whether or not the text wraps. Counting only text lines
            // made every row one line short, so each bled into the next and
            // the whole pane rendered as overlapping mush — most visibly on
            // real transcripts, where every segment has a speaker. A row with
            // no header (unattributed, no match) wastes one line height of
            // whitespace; that is the safe direction, and the Slint side
            // gives the text `line-count - 1` lines so the header cannot steal
            // one from it.
            let text_lines = chars
                .div_ceil(chars_per_line)
                .clamp(1, usize::from(MAX_LINES_PER_SEGMENT) - 1);
            let lines = (1 + text_lines).min(usize::from(MAX_LINES_PER_SEGMENT));
            self.lines
                .push(u16::try_from(lines).unwrap_or(MAX_LINES_PER_SEGMENT));
            self.total += u32::from(self.lines[self.lines.len() - 1]);
        }
    }

    /// The characters per line this layout was built for.
    #[must_use]
    pub fn chars_per_line(&self) -> usize {
        self.chars_per_line
    }

    /// Number of segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.lines.len()
    }

    /// Total lines in the whole transcript.
    #[must_use]
    pub fn total_lines(&self) -> u32 {
        self.total
    }

    /// Lines occupied by segment `index`, or `1` for an index that is not there.
    #[must_use]
    pub fn line_count(&self, index: usize) -> u16 {
        self.lines.get(index).copied().unwrap_or(1).max(1)
    }

    /// Lines occupied by every segment before `index`.
    #[must_use]
    pub fn lines_before(&self, index: usize) -> u32 {
        self.prefix.get(index).copied().unwrap_or(self.total)
    }

    /// The last segment index whose text starts at or before `line`.
    fn segment_at_or_before(&self, line: u32) -> usize {
        self.prefix
            .partition_point(|&p| p <= line)
            .saturating_sub(1)
    }

    /// The first segment index whose text starts strictly after `line`.
    fn segment_after(&self, line: u32) -> usize {
        self.prefix.partition_point(|&p| p <= line)
    }

    /// The half-open range of segments a viewport at `offset_px` showing
    /// `viewport_px` needs, plus overscan.
    ///
    /// Out-of-range values are clamped rather than rejected, because the
    /// viewport reports zero before the first layout pass and while hidden.
    #[must_use]
    pub fn visible_range(
        &self,
        offset_px: f32,
        viewport_px: f32,
        line_height: f32,
        overscan: usize,
    ) -> std::ops::Range<usize> {
        let count = self.lines.len();
        if count == 0 || line_height <= 0.0 {
            return 0..0;
        }
        let offset_px = offset_px.max(0.0);
        let viewport_px = viewport_px.max(0.0);
        let first_line = (offset_px / line_height).floor().max(0.0) as u32;
        let last_line = ((offset_px + viewport_px) / line_height).ceil().max(0.0) as u32;

        let first = self.segment_at_or_before(first_line);
        // `segment_after` is exclusive, so a viewport ending exactly on a
        // boundary does not keep a whole extra row alive.
        let last = self.segment_after(last_line).max(first + 1);
        let first = first.saturating_sub(overscan).min(count);
        let last = (last + overscan).min(count).max(first);
        first..last
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// A `VecModel` plus the `ModelRc` handed to the window, so the window and this
/// module share one allocation instead of copying a row list per commit.
struct SharedModel<T> {
    vec: Rc<VecModel<T>>,
    rc: ModelRc<T>,
}

impl<T: Clone + PartialEq + 'static> SharedModel<T> {
    fn new() -> Self {
        let vec = Rc::new(VecModel::<T>::default());
        let rc = ModelRc::from(Rc::clone(&vec));
        Self { vec, rc }
    }
}

/// The three settings drop-downs.
///
/// Cached, and rebuilt only when the labels actually change. Rebuilding a
/// `ComboBox`'s model resets its `current-index`, and because that property is
/// two-way bound to the window's, the reset writes straight back and silently
/// un-selects the user's choice. That is not a cosmetic bug: it means the
/// device the user just picked reverts on the next commit.
struct OptionModels {
    model: SharedModel<SharedString>,
    device: SharedModel<SharedString>,
    language: SharedModel<SharedString>,
    cached: [Vec<String>; 3],
}

impl OptionModels {
    fn new() -> Self {
        Self {
            model: SharedModel::new(),
            device: SharedModel::new(),
            language: SharedModel::new(),
            cached: [Vec::new(), Vec::new(), Vec::new()],
        }
    }

    /// Push `labels` into slot `index` if it differs from what is cached.
    fn sync(&mut self, index: usize, labels: &[String]) {
        if self.cached[index] == labels {
            return;
        }
        let shared: Vec<SharedString> = labels
            .iter()
            .map(|l| SharedString::from(l.as_str()))
            .collect();
        let target = match index {
            0 => &self.model,
            1 => &self.device,
            _ => &self.language,
        };
        sync_vec_model(&target.vec, &shared);
        self.cached[index] = labels.to_vec();
    }
}

/// Everything the UI holds between commits. One `RefCell` so the borrow ends
/// before any Slint call, which is what stops a property write that re-enters
/// the UI from tripping the borrow checker at runtime.
#[derive(Debug)]
struct ViewState {
    app: AppView,
    jobs: JobListState,
    transcript: TranscriptView,
    speakers: SpeakerPanel,
    layout: TranscriptLayout,
    /// The job whose transcript and speakers are loaded, so a selection change
    /// is the only thing that triggers a reload.
    loaded_job: Option<JobId>,
    viewport_offset_px: f32,
    viewport_height_px: f32,
    pending_scroll_px: Option<f32>,
    /// The last rows handed to Slint, for diffing.
    window_rows: Vec<TranscriptRow>,
    job_rows: Vec<JobRowS>,
    speaker_rows: Vec<SpeakerRowS>,
}

impl ViewState {
    fn new(app: AppView) -> Self {
        Self {
            app,
            jobs: JobListState::new(),
            transcript: TranscriptView::new(),
            speakers: SpeakerPanel::new(),
            layout: TranscriptLayout::default(),
            loaded_job: None,
            viewport_offset_px: 0.0,
            viewport_height_px: 0.0,
            pending_scroll_px: None,
            window_rows: Vec::new(),
            job_rows: Vec::new(),
            speaker_rows: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// The controller
// ---------------------------------------------------------------------------

/// The window's controller: owns the backend, the state, and the one handle to
/// the window.
///
/// Construct it on the event-loop thread. [`UiModel::new`] records that thread
/// and every later Slint call asserts it, so the mistake is caught immediately
/// rather than turning into a torn property three frames later.
pub struct UiModel<C: JobController> {
    controller: C,
    host: Arc<dyn HostServices>,
    window: Weak<AppWindow>,
    /// The colour-scheme global, as a `'static` weak handle.
    ///
    /// A global's strong handle borrows the component that instantiated it, so
    /// the only way to keep one past `new` is the `'static` weak form. That is
    /// not an obscure incantation: it is how Slint expects a per-window global
    /// to be stored.
    theme: Weak<crate::slint_api::VaaniColors<'static>>,
    loop_thread: ThreadId,
    events: Receiver<ControllerEvent>,
    state: RefCell<ViewState>,
    jobs: SharedModel<JobRowS>,
    speakers: SharedModel<SpeakerRowS>,
    window_rows_model: SharedModel<TranscriptRow>,
    options: RefCell<OptionModels>,
    timer: Timer,
    pumps: Cell<u64>,
    live_transcript_rows: Cell<usize>,
}

impl<C: JobController> std::fmt::Debug for UiModel<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UiModel")
            .field("backend", &self.controller.backend_name())
            .field("loop_thread", &self.loop_thread)
            .field("pumps", &self.pumps.get())
            .finish_non_exhaustive()
    }
}

impl<C: JobController> UiModel<C> {
    /// Build the model, wire every callback, and take an event subscription.
    ///
    /// Must be called on the thread that will run the event loop. Does not show
    /// the window; the caller decides when.
    #[must_use]
    pub fn new(controller: C, host: Arc<dyn HostServices>, window: &Rc<AppWindow>) -> Rc<Self> {
        let events = controller.subscribe();
        let settings = controller.settings();
        let app = AppView {
            backend_name: controller.backend_name(),
            is_demo: controller.is_demo(),
            host_name: host.host_name(),
            settings: settings.clone(),
            models: controller.models(),
            devices: controller.devices(),
            languages: controller.languages(),
            ..AppView::new()
        };

        // The global's strong handle borrows the window, so it is downgraded to
        // the `'static` weak form immediately. That is the only lifetime a
        // stored handle can have, and it is what Slint provides `as_weak` for.
        let theme_globals =
            <crate::slint_api::VaaniColors<'_> as SlintGlobal<AppWindow>>::get(window);
        let theme: Weak<crate::slint_api::VaaniColors<'static>> =
            SlintGlobal::as_weak(&theme_globals);
        drop(theme_globals);

        let model = Rc::new(Self {
            controller,
            host,
            window: window.as_weak(),
            theme,
            loop_thread: std::thread::current().id(),
            events,
            state: RefCell::new(ViewState::new(app)),
            jobs: SharedModel::new(),
            speakers: SharedModel::new(),
            window_rows_model: SharedModel::new(),
            options: RefCell::new(OptionModels::new()),
            timer: Timer::default(),
            pumps: Cell::new(0),
            live_transcript_rows: Cell::new(0),
        });

        // The models are handed to the window once. Every later change goes
        // through the `VecModel`, which notifies the repeaters that are already
        // bound to it.
        window.set_jobs(model.jobs.rc.clone());
        window.set_speakers(model.speakers.rc.clone());
        window.set_transcript_window(model.window_rows_model.rc.clone());
        {
            let options = model.options.borrow();
            window.set_model_options(options.model.rc.clone());
            window.set_device_options(options.device.rc.clone());
            window.set_language_options(options.language.rc.clone());
        }

        model.install_callbacks(window);
        model.apply_theme(settings.theme);
        model.reload_jobs();
        model.commit();
        model.start_pump(PUMP_INTERVAL);
        model
    }

    /// The backend, for tests and for a host that wants to query it directly.
    pub fn controller(&self) -> &C {
        &self.controller
    }

    /// A weak handle to the window.
    pub fn window(&self) -> Weak<AppWindow> {
        self.window.clone()
    }

    /// How many pumps have run. Diagnostics and tests.
    pub fn pumps(&self) -> u64 {
        self.pumps.get()
    }

    /// How many transcript delegates are live right now.
    ///
    /// This is the number the windowing claim rests on, and it is a test
    /// assertion rather than a hope: for a 20,000-word transcript it must stay
    /// in the dozens, not the thousands.
    pub fn live_transcript_rows(&self) -> usize {
        self.live_transcript_rows.get()
    }

    /// Start draining the backend's events on a timer.
    ///
    /// A timer rather than a push callback, on purpose. A push callback would
    /// have to capture something to call into, and what it captures either
    /// leaks (an `Rc` the window also holds) or has to be an upgradeable weak
    /// handle anyway. At that point a timer that asks "is there anything?" every
    /// 33 ms is simpler and provably on the loop thread.
    pub fn start_pump(self: &Rc<Self>, interval: Duration) {
        let weak = Rc::downgrade(self);
        self.timer.start(TimerMode::Repeated, interval, move || {
            if let Some(model) = weak.upgrade() {
                model.pump();
            }
        });
    }

    /// Stop draining. Only useful in tests; closing the window ends the loop.
    pub fn stop_pump(&self) {
        self.timer.stop();
    }

    /// Drain the backend's event channel and apply everything that arrived.
    ///
    /// The only place backend events become UI state, and it must only be called
    /// on the event-loop thread. Cheap by construction: it never blocks, never
    /// touches the filesystem, and never takes a lock the backend might be
    /// holding across a long operation.
    pub fn pump(&self) {
        self.assert_on_loop();
        self.pumps.set(self.pumps.get() + 1);

        let mut drained = Vec::new();
        while let Ok(event) = self.events.try_recv() {
            drained.push(event);
            if drained.len() >= MAX_EVENTS_PER_PUMP {
                break;
            }
        }
        if drained.is_empty() {
            return;
        }

        let before = drained.len();
        let coalesced = view::coalesce(drained);
        {
            let mut state = self.state.borrow_mut();
            state.app.events_coalesced += u64::try_from(before - coalesced.len()).unwrap_or(0);
            for event in &coalesced {
                state.app.events_applied += 1;
                self.apply_event(&mut state, event);
            }
        }
        self.commit();
    }

    /// Apply one event to the state. The caller holds the state borrow.
    fn apply_event(&self, state: &mut ViewState, event: &ControllerEvent) {
        match event {
            ControllerEvent::JobAdded(job) => {
                let id = job.id.clone();
                state.jobs.upsert(job.clone());
                // Select what the user just asked for.
                //
                // `select_default` deliberately leaves an existing selection
                // alone - correct for a history list, wrong here. Queue a second
                // file and the previous job stayed selected, so the row being
                // watched, the progress bar and the transcript pane were all
                // showing a file that was not being transcribed. Picking a file
                // through the dialog has always done this; the typed-path
                // button did not, which is why the two ways in disagreed.
                state.jobs.select(&id);
                state
                    .app
                    .note(format!("Queued {}", view::file_name_of(&job.input)));
                // It is now the selected job, so the rule allows it - and the
                // job has no transcript yet, which is the correct thing to show
                // for one that has just started.
                if may_replace_the_pane(state.jobs.selected(), &id) {
                    self.load_job(state, &id);
                }
            }
            ControllerEvent::JobChanged(job) => {
                let id = job.id.clone();
                let changed = state.jobs.upsert(job.clone());
                let selected = state.jobs.selected().cloned();
                if changed && selected.as_ref() == Some(&id) && job.has_transcript {
                    // Show a transcript as soon as there is one. Waiting for
                    // `Complete` would mean a two-hour job shows nothing for
                    // two hours.
                    self.load_job(state, &id);
                }
            }
            ControllerEvent::JobRemoved(id) => {
                if state.jobs.remove(id) {
                    state.app.note(format!("Removed job {id}"));
                }
                if state.loaded_job.as_ref() == Some(id) {
                    state.transcript.set_transcript(None);
                    state.speakers.set_speakers(Vec::new());
                    state.layout.relayout(None, state.layout.chars_per_line());
                    state.loaded_job = None;
                }
                state.jobs.select_default();
            }
            ControllerEvent::TranscriptReady(id) => {
                // Only if this is the job the user is looking at.
                //
                // Every finished job emits this, including background ones, and
                // `load_job` replaces the pane. Without the check, transcribing
                // two files shows whichever happens to finish first: you read
                // one transcript and the other replaces it mid-sentence, with
                // no indication the pane is now showing a different file. The
                // selected job is the only thing that may change what is on
                // screen; a background event is not that.
                if may_replace_the_pane(state.jobs.selected(), id) {
                    self.load_job(state, id);
                }
            }
            ControllerEvent::SpeakersChanged(id) => {
                if may_replace_the_pane(state.jobs.selected(), id) {
                    self.load_speakers(state, id);
                }
            }
            ControllerEvent::RenameRejected { speaker, error, .. } => {
                // The panel applied the name optimistically. Put the generated
                // label back and say why, rather than leaving the UI showing a
                // name the database does not have.
                state.speakers.revert_rename(*speaker, None);
                state.app.banner = Banner::from_error(Severity::Error, error);
                state.app.note("Speaker rename was rejected");
            }
            ControllerEvent::ExportFinished { job, paths } => {
                let destinations = paths
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                state.app.banner =
                    Banner::notice(Severity::Info, format!("Exported {job} to {destinations}"));
                state.app.note("Export finished");
            }
            ControllerEvent::ExportFailed { error, .. } => {
                state.app.banner = Banner::from_error(Severity::Error, error);
                state.app.note("Export failed");
            }
            ControllerEvent::SettingsChanged(settings) => {
                state.app.settings = settings.clone();
                self.apply_theme(settings.theme);
                state.app.note("Settings changed");
            }
            ControllerEvent::Notice { level, text } => {
                state.app.banner = Banner::notice(*level, text.clone());
                state.app.note(text.clone());
            }
        }
    }

    /// Load a job's transcript and speakers. The caller holds the state borrow.
    fn load_job(&self, state: &mut ViewState, id: &JobId) {
        match self.controller.transcript(id) {
            Ok(transcript) => {
                let chars = state.layout.chars_per_line();
                state.transcript.set_transcript(transcript.clone());
                state.layout.relayout(transcript.as_deref(), chars);
                state.loaded_job = Some(id.clone());
            }
            Err(error) => {
                // A backend that cannot hand over a transcript is a real
                // failure. Hiding it would leave an empty pane that looks like
                // a job which has not finished.
                state.app.banner = Banner::from_error(Severity::Error, &error);
            }
        }
        self.load_speakers(state, id);
    }

    /// Load a job's speakers. The caller holds the state borrow.
    fn load_speakers(&self, state: &mut ViewState, id: &JobId) {
        match self.controller.speakers(id) {
            Ok(speakers) => {
                state.speakers.set_speakers(speakers);
                state.speakers.select_default();
            }
            Err(error) => state.app.banner = Banner::from_error(Severity::Warning, &error),
        }
    }

    /// Re-read the job list from the backend. Used at construction and by tests.
    pub fn reload_jobs(&self) {
        self.assert_on_loop();
        let jobs = self.controller.jobs();
        let selected = {
            let mut state = self.state.borrow_mut();
            state.jobs = JobListState::from_jobs(jobs);
            state.jobs.selected().cloned()
        };
        if let Some(id) = selected {
            let mut state = self.state.borrow_mut();
            self.load_job(&mut state, &id);
        }
        self.commit();
    }

    /// Push state into the window.
    ///
    /// The only method that writes Slint properties, and the reason every other
    /// method can be written as if the UI were single-threaded.
    pub fn commit(&self) {
        self.assert_on_loop();

        let mut state = self.state.borrow_mut();

        // -- jobs ------------------------------------------------------------
        let job_rows: Vec<JobRowS> = state.jobs.rows().iter().map(to_slint_job_row).collect();
        if job_rows != state.job_rows {
            sync_vec_model(&self.jobs.vec, &job_rows);
            state.job_rows = job_rows;
        }

        // -- speakers --------------------------------------------------------
        let speaker_rows: Vec<SpeakerRowS> = state
            .speakers
            .rows()
            .iter()
            .map(to_slint_speaker_row)
            .collect();
        if speaker_rows != state.speaker_rows {
            sync_vec_model(&self.speakers.vec, &speaker_rows);
            state.speaker_rows = speaker_rows;
        }

        // -- transcript window ----------------------------------------------
        // The point of the whole exercise: only the rows the viewport needs are
        // ever built, however long the transcript is.
        let range = state.layout.visible_range(
            state.viewport_offset_px,
            state.viewport_height_px,
            TRANSCRIPT_LINE_HEIGHT_PX,
            OVERSCAN_ROWS,
        );
        let window_rows: Vec<TranscriptRow> = (range.start..range.end)
            .filter_map(|i| to_slint_transcript_row(&state.transcript, &state.layout, i))
            .collect();
        if window_rows != state.window_rows {
            sync_vec_model(&self.window_rows_model.vec, &window_rows);
            state.window_rows = window_rows;
        }
        self.live_transcript_rows
            .set(self.window_rows_model.vec.row_count());

        // -- scalars ---------------------------------------------------------
        let (overall, overall_text) = state.app.overall_progress(&state.jobs);
        let activity = state.app.activity(&state.jobs);
        // "Busy" means there is unfinished work, not that a worker is spinning
        // right now: a queued job is the app being busy, and a Cancel button
        // that disappears between phases is a Cancel button people stop using.
        let busy = state.jobs.jobs().iter().any(|j| !j.phase.is_terminal());
        let selected_job = state
            .jobs
            .selected()
            .and_then(|id| state.jobs.rows().iter().position(|r| r.id == id.as_str()))
            .map_or(-1, |i| i as i32);
        let job_count = i32::try_from(state.jobs.jobs().len()).unwrap_or(i32::MAX);
        let can_cancel = state
            .jobs
            .selected_job()
            .is_some_and(|j| j.phase.is_active() && !j.cancel_requested);
        let can_export = state.jobs.selected_job().is_some_and(|j| j.has_transcript);

        let transcript_summary = state.transcript.summary();
        let transcript_rows_total = i32::try_from(state.transcript.row_count()).unwrap_or(0);
        let transcript_selected = state
            .transcript
            .selection()
            .focus
            .and_then(|i| i32::try_from(i).ok())
            .unwrap_or(-1);
        let detail = state.transcript.detail().unwrap_or_default();
        let has_detail = !detail.is_empty();
        let selection_summary = match state.transcript.selection().range() {
            Some((lo, hi)) if hi > lo => format!(
                "Segments {} to {} of {}",
                lo + 1,
                hi + 1,
                state.transcript.row_count()
            ),
            Some((lo, _)) => format!("Segment {} of {}", lo + 1, state.transcript.row_count()),
            None => String::new(),
        };
        let can_copy = transcript_rows_total > 0;
        let match_count = if state.transcript.matches().is_empty() {
            String::new()
        } else {
            format!(
                "{} of {}",
                state.transcript.matches().len(),
                state.transcript.row_count()
            )
        };

        let selected_speaker = state
            .speakers
            .selected()
            .and_then(|i| i32::try_from(i).ok())
            .unwrap_or(-1);
        let edit = state.speakers.edit().cloned();
        let editing_speaker = edit
            .as_ref()
            .and_then(|e| i32::try_from(e.index).ok())
            .unwrap_or(-1);
        let editing_text = edit.as_ref().map(|e| e.draft.clone()).unwrap_or_default();
        let editing_label = edit
            .as_ref()
            .and_then(|e| {
                state
                    .speakers
                    .speakers()
                    .iter()
                    .find(|s| s.index == e.index)
                    .map(|s| format!("Speaker {}", s.index + 1))
            })
            .unwrap_or_default();

        let banner = state.app.banner.clone();
        let footer = self.footer_text(&state);
        let model_index = AppView::choice_index(&state.app.models, &state.app.settings.model);
        let device_index = AppView::choice_index(&state.app.devices, &state.app.settings.device);
        let language_index =
            AppView::choice_index(&state.app.languages, &state.app.settings.language);
        let model_labels = state.app.model_labels();
        let device_labels = state.app.device_labels();
        let language_labels = state.app.language_labels();
        {
            let mut options = self.options.borrow_mut();
            options.sync(0, &model_labels);
            options.sync(1, &device_labels);
            options.sync(2, &language_labels);
        }
        let input_path = state.app.input_path.clone();
        let input_accepted = state.app.input_accepted;
        let can_enqueue = state.app.can_enqueue();
        let is_demo = state.app.is_demo;
        let settings_open = state.app.settings_open;
        let diarize = state.app.settings.diarize;
        let theme = state.app.settings.theme;
        let total_lines = state.layout.total_lines();
        let transcript_empty = transcript_empty_message(&state);
        let search_text = state.transcript.query().to_owned();

        // A scroll the UI decided on. `None` leaves the user's scroll alone,
        // which is the difference between a pane that follows the selection and
        // a pane that fights the user.
        let pending_scroll = state.pending_scroll_px.take();

        drop(state);

        // -- write -----------------------------------------------------------
        let Some(window) = self.window.upgrade() else {
            return;
        };
        window.set_selected_job(selected_job);
        window.set_job_count(job_count);
        window.set_can_cancel(can_cancel);
        window.set_transcript_rows_total(transcript_rows_total);
        window.set_transcript_lines_total(i32::try_from(total_lines).unwrap_or(i32::MAX));
        window.set_transcript_line_height(TRANSCRIPT_LINE_HEIGHT_PX);
        window.set_transcript_content_height(total_lines as f32 * TRANSCRIPT_LINE_HEIGHT_PX);
        window.set_transcript_selected(transcript_selected);
        window.set_transcript_summary(transcript_summary.into());
        window.set_transcript_empty(transcript_empty.into());
        window.set_transcript_has_detail(has_detail);
        window.set_transcript_detail(detail.into());
        window.set_selection_summary(selection_summary.into());
        window.set_can_copy(can_copy);
        window.set_match_count(match_count.into());
        window.set_search_text(search_text.into());
        window.set_selected_speaker(selected_speaker);
        window.set_editing_speaker(editing_speaker);
        window.set_editing_speaker_text(editing_text.into());
        window.set_editing_speaker_label(editing_label.into());
        window.set_input_path(input_path.into());
        window.set_input_accepted(input_accepted);
        window.set_can_enqueue(can_enqueue);
        window.set_busy(busy);
        window.set_activity(activity.into());
        window.set_overall(overall.map_or(-1.0, |f| f as f32));
        window.set_overall_text(overall_text.into());
        window.set_model_index(model_index);
        window.set_device_index(device_index);
        window.set_language_index(language_index);
        window.set_diarize(diarize);
        window.set_theme_index(theme_index(theme));
        window.set_settings_open(settings_open);
        window.set_can_export(can_export);
        window.set_banner_visible(banner.visible);
        window.set_banner_level(banner.level.map_or(0, severity_level));
        window.set_banner_title(banner.title.into());
        window.set_banner_detail(banner.detail.into());
        window.set_banner_recovery(banner.recovery.into());
        window.set_banner_code(banner.code.into());
        window.set_demo_mode(is_demo);
        window.set_footer(footer.into());
        window.set_scale_factor(self.scale_factor());
        window.set_display_scheme(self.color_scheme_name().into());
        if let Some(offset) = pending_scroll {
            window.set_transcript_scroll_y(-offset);
        }
    }

    // -----------------------------------------------------------------------
    // Callback handlers
    // -----------------------------------------------------------------------

    fn install_callbacks(self: &Rc<Self>, window: &AppWindow) {
        // Every callback captures a `Weak`, never an `Rc`. An `Rc` here would
        // make a cycle: window -> handler -> model -> weak window. The `Weak`
        // is a cycle, so the window and the model are dropped when the host
        // drops them, which matters in a test binary that builds a hundred of
        // them.
        macro_rules! weak {
            () => {
                Rc::downgrade(self)
            };
        }

        window.on_browse_input_requested({
            let weak = weak!();
            move || {
                if let Some(m) = weak.upgrade() {
                    m.browse_input();
                }
            }
        });
        window.on_enqueue_requested({
            let weak = weak!();
            move || {
                if let Some(m) = weak.upgrade() {
                    m.enqueue();
                }
            }
        });
        window.on_cancel_requested({
            let weak = weak!();
            move || {
                if let Some(m) = weak.upgrade() {
                    m.cancel();
                }
            }
        });
        window.on_input_edited({
            let weak = weak!();
            move |path| {
                if let Some(m) = weak.upgrade() {
                    m.input_edited(path.to_string());
                }
            }
        });
        window.on_select_job_requested({
            let weak = weak!();
            move |index| {
                if let Some(m) = weak.upgrade() {
                    m.select_job(index);
                }
            }
        });
        window.on_remove_job_requested({
            let weak = weak!();
            move |index| {
                if let Some(m) = weak.upgrade() {
                    m.remove_job(index);
                }
            }
        });
        window.on_copy_requested({
            let weak = weak!();
            move || {
                if let Some(m) = weak.upgrade() {
                    m.copy_selection();
                }
            }
        });
        window.on_select_all_requested({
            let weak = weak!();
            move || {
                if let Some(m) = weak.upgrade() {
                    m.select_all();
                }
            }
        });
        window.on_find_next_requested({
            let weak = weak!();
            move || {
                if let Some(m) = weak.upgrade() {
                    m.find_next();
                }
            }
        });
        window.on_find_previous_requested({
            let weak = weak!();
            move || {
                if let Some(m) = weak.upgrade() {
                    m.find_previous();
                }
            }
        });
        window.on_search_edited({
            let weak = weak!();
            move |query| {
                if let Some(m) = weak.upgrade() {
                    m.search(query.to_string());
                }
            }
        });
        window.on_select_segment_requested({
            let weak = weak!();
            move |index, extend| {
                if let Some(m) = weak.upgrade() {
                    m.select_segment(index, extend);
                }
            }
        });
        window.on_move_segment_requested({
            let weak = weak!();
            move |delta, extend| {
                if let Some(m) = weak.upgrade() {
                    m.move_segment(delta, extend);
                }
            }
        });
        window.on_begin_rename_requested({
            let weak = weak!();
            move |index| {
                if let Some(m) = weak.upgrade() {
                    m.begin_rename(index);
                }
            }
        });
        window.on_commit_rename_requested({
            let weak = weak!();
            move |index, name| {
                if let Some(m) = weak.upgrade() {
                    m.commit_rename(index, name.to_string());
                }
            }
        });
        window.on_cancel_rename_requested({
            let weak = weak!();
            move || {
                if let Some(m) = weak.upgrade() {
                    m.cancel_rename();
                }
            }
        });
        window.on_export_requested({
            let weak = weak!();
            move || {
                if let Some(m) = weak.upgrade() {
                    m.export();
                }
            }
        });
        window.on_model_selected({
            let weak = weak!();
            move |value| {
                if let Some(m) = weak.upgrade() {
                    m.setting_label_changed(SettingKind::Model, value.to_string());
                }
            }
        });
        window.on_device_selected({
            let weak = weak!();
            move |value| {
                if let Some(m) = weak.upgrade() {
                    m.setting_label_changed(SettingKind::Device, value.to_string());
                }
            }
        });
        window.on_language_selected({
            let weak = weak!();
            move |value| {
                if let Some(m) = weak.upgrade() {
                    m.setting_label_changed(SettingKind::Language, value.to_string());
                }
            }
        });
        window.on_diarize_toggled({
            let weak = weak!();
            move |on| {
                if let Some(m) = weak.upgrade() {
                    m.set_diarize(on);
                }
            }
        });
        window.on_theme_selected({
            let weak = weak!();
            move |index| {
                if let Some(m) = weak.upgrade() {
                    m.set_theme_index(index);
                }
            }
        });
        window.on_settings_toggled({
            let weak = weak!();
            move || {
                if let Some(m) = weak.upgrade() {
                    m.toggle_settings();
                }
            }
        });
        window.on_dismiss_banner_requested({
            let weak = weak!();
            move || {
                if let Some(m) = weak.upgrade() {
                    m.dismiss_banner();
                }
            }
        });
        window.on_viewport_changed({
            let weak = weak!();
            move |offset, height, width| {
                if let Some(m) = weak.upgrade() {
                    m.viewport_changed(offset, height, width);
                }
            }
        });
    }

    /// Open the file picker and queue whatever comes back.
    ///
    /// Picking a file and then having to press Transcribe is a wasted step for
    /// the common case of "I know which file I want", so the picker enqueues.
    /// The path field is filled with the first pick so the user can see what
    /// happened and correct it.
    pub fn browse_input(&self) {
        self.assert_on_loop();
        let picked = self.host.pick_media_files();
        if picked.is_empty() {
            return;
        }
        self.state
            .borrow_mut()
            .app
            .set_input_path(picked[0].display().to_string());
        for path in picked {
            let outcome = self.controller.enqueue(&path);
            self.pump();
            match outcome {
                Ok(id) => {
                    self.state.borrow_mut().jobs.select(&id);
                }
                Err(error) => self.show_error(&error),
            }
        }
        self.pump();
        self.commit();
    }

    /// Queue the file in the path field.
    pub fn enqueue(&self) {
        self.assert_on_loop();
        let path = {
            let mut state = self.state.borrow_mut();
            if !state.app.can_enqueue() {
                state.app.banner = Banner::notice(
                    Severity::Warning,
                    "Choose a video or audio file first. The path field accepts a full path.",
                );
                drop(state);
                self.commit();
                return;
            }
            PathBuf::from(state.app.input_path.trim())
        };
        let outcome = self.controller.enqueue(&path);
        // Pump before selecting. `enqueue` publishes the job as an event, and a
        // selection can only name a job the list already holds, so selecting
        // first would silently fail and the next Cancel click would act on
        // whatever was selected before — which is how a user cancels the wrong
        // recording.
        self.pump();
        match outcome {
            Ok(id) => {
                let mut state = self.state.borrow_mut();
                state.app.set_input_path(String::new());
                state
                    .app
                    .note(format!("Queued {}", view::file_name_of(&path)));
                // The job just queued becomes the selected one, so the window
                // shows the thing the user just asked for.
                state.jobs.select(&id);
            }
            Err(error) => self.show_error(&error),
        }
        self.commit();
    }

    /// Cancel the selected job.
    ///
    /// Immediate in the only sense that matters to a user: the row says
    /// "Cancelling" on this frame, because the state is updated here and the
    /// commit happens before the handler returns. Whether the worker actually
    /// stops is the backend's business, and it says so when it has.
    pub fn cancel(&self) {
        self.assert_on_loop();
        let target = {
            let mut state = self.state.borrow_mut();
            let Some(job) = state.jobs.selected_job() else {
                return;
            };
            if !job.phase.is_active() {
                return;
            }
            let mut updated = job.clone();
            updated.cancel_requested = true;
            let id = updated.id.clone();
            state.jobs.upsert(updated);
            id
        };
        if let Err(error) = self.controller.cancel(&target) {
            let mut state = self.state.borrow_mut();
            if let Some(job) = state.jobs.job(&target) {
                let mut reverted = job.clone();
                reverted.cancel_requested = false;
                state.jobs.upsert(reverted);
            }
            state.app.banner = Banner::from_error(Severity::Error, &error);
        }
        self.commit();
    }

    /// Select a job by its row index, loading its transcript.
    pub fn select_job(&self, index: i32) {
        self.assert_on_loop();
        if index < 0 {
            return;
        }
        let selected = {
            let mut state = self.state.borrow_mut();
            if !state.jobs.select_row(index as usize) {
                return;
            }
            state.jobs.selected().cloned()
        };
        if let Some(id) = selected {
            let mut state = self.state.borrow_mut();
            self.load_job(&mut state, &id);
            state.pending_scroll_px = Some(0.0);
        }
        self.commit();
    }

    /// Forget a job.
    pub fn remove_job(&self, index: i32) {
        self.assert_on_loop();
        if index < 0 {
            return;
        }
        let target = {
            let state = self.state.borrow();
            state
                .jobs
                .rows()
                .get(index as usize)
                .map(|r| JobId::from(r.id.as_str()))
        };
        let Some(target) = target else { return };
        if let Err(error) = self.controller.remove(&target) {
            self.show_error(&error);
            return;
        }
        self.pump();
        self.commit();
    }

    /// Set the file field from typed input.
    pub fn input_edited(&self, path: String) {
        self.assert_on_loop();
        self.state.borrow_mut().app.set_input_path(path);
        self.commit();
    }

    /// Copy the selection, or the whole transcript, to the clipboard.
    pub fn copy_selection(&self) {
        self.assert_on_loop();
        let (text, selected, total) = {
            let state = self.state.borrow();
            (
                state.transcript.copy_text(),
                state.transcript.selection().len(),
                state.transcript.row_count(),
            )
        };
        if text.is_empty() {
            return;
        }
        let lines = text.lines().count();
        self.host.set_clipboard_text(&text);
        let mut state = self.state.borrow_mut();
        state.app.banner = Banner::notice(
            Severity::Info,
            if selected == 0 || selected == total {
                format!("Copied the whole transcript ({lines} lines)")
            } else {
                format!("Copied {selected} of {total} segments ({lines} lines)")
            },
        );
        state.app.note("Copied to the clipboard");
        drop(state);
        self.commit();
    }

    /// Select every segment.
    pub fn select_all(&self) {
        self.assert_on_loop();
        {
            let mut state = self.state.borrow_mut();
            state.transcript.select_all();
            self.reveal_selection(&mut state);
        }
        self.commit();
    }

    /// Set the search string.
    pub fn search(&self, query: String) {
        self.assert_on_loop();
        self.state.borrow_mut().transcript.set_query(query);
        self.commit();
    }

    /// Jump to the next match, revealing it.
    fn jump_to_match(&self, forward: bool) {
        self.assert_on_loop();
        let found = {
            let mut state = self.state.borrow_mut();
            let found = if forward {
                state.transcript.next_match()
            } else {
                state.transcript.previous_match()
            };
            if found.is_some() {
                self.reveal_selection(&mut state);
            }
            found
        };
        if found.is_some() {
            self.commit();
        }
    }

    /// Jump to the next match.
    pub fn find_next(&self) {
        self.jump_to_match(true);
    }

    /// Jump to the previous match.
    pub fn find_previous(&self) {
        self.jump_to_match(false);
    }

    /// Select a segment, or extend the selection to it.
    pub fn select_segment(&self, index: i32, extend: bool) {
        self.assert_on_loop();
        if index < 0 {
            return;
        }
        {
            let mut state = self.state.borrow_mut();
            if extend {
                state.transcript.extend_to(index as usize);
            } else {
                state.transcript.select(index as usize);
            }
            self.reveal_selection(&mut state);
        }
        self.commit();
    }

    /// Move the selection by `delta` segments.
    pub fn move_segment(&self, delta: i32, extend: bool) {
        self.assert_on_loop();
        {
            let mut state = self.state.borrow_mut();
            state.transcript.move_selection(delta as isize, extend);
            self.reveal_selection(&mut state);
        }
        self.commit();
    }

    /// Open the rename editor on a speaker.
    pub fn begin_rename(&self, index: i32) {
        self.assert_on_loop();
        if index < 0 {
            return;
        }
        self.state
            .borrow_mut()
            .speakers
            .begin_rename(index as usize);
        self.commit();
    }

    /// Commit a rename and tell the backend.
    ///
    /// `name` is what the text field held when the user committed. The panel
    /// keeps the authoritative draft, so it is updated from the argument first:
    /// the widget's value is the user's intent, and using the panel's copy alone
    /// would silently drop a keystroke if the two ever disagreed.
    ///
    /// This is the clean event the brief asks for. The UI applies the name
    /// locally so the interaction feels instant, and asks the backend to
    /// persist it; the backend answers with `SpeakersChanged` on success or
    /// `RenameRejected` on failure, and the UI rolls back on the latter.
    pub fn commit_rename(&self, index: i32, name: String) {
        self.assert_on_loop();
        let emitted = {
            let mut state = self.state.borrow_mut();
            // The panel owns which edit is open. The index arrives from the
            // widget, and is checked against it so a stale click cannot rename
            // the wrong person.
            let matches_open_edit = state
                .speakers
                .edit()
                .is_some_and(|e| index >= 0 && e.index == index as usize);
            if !matches_open_edit {
                return;
            }
            state.speakers.set_draft(name);
            state.speakers.commit_rename()
        };
        let Some((speaker, name)) = emitted else {
            self.commit();
            return;
        };
        let job = self.state.borrow().loaded_job.clone();
        if let Some(job) = job
            && let Err(error) = self.controller.rename_speaker(&job, speaker, name)
        {
            let mut state = self.state.borrow_mut();
            state.speakers.revert_rename(speaker, None);
            state.app.banner = Banner::from_error(Severity::Error, &error);
        }
        self.pump();
        self.commit();
    }

    /// Abandon a rename.
    pub fn cancel_rename(&self) {
        self.assert_on_loop();
        self.state.borrow_mut().speakers.cancel_rename();
        self.commit();
    }

    /// Ask where to write, then export.
    ///
    /// The format is read from the widget rather than from a copy the UI keeps,
    /// because the widget is where the user changed it and a second copy is a
    /// second thing that can be stale.
    pub fn export(&self) {
        self.assert_on_loop();
        let format = self.export_format();
        let (job, stem) = {
            let state = self.state.borrow();
            let Some(job) = state.jobs.selected_job() else {
                return;
            };
            if !job.has_transcript {
                return;
            }
            let name = view::file_name_of(&job.input);
            let stem = name
                .rsplit_once('.')
                .map_or_else(|| name.clone(), |(s, _)| s.to_owned());
            (job.id.clone(), stem)
        };
        let suggested = format!("{stem}.{}", format.extension());
        let Some(target) = self.host.pick_export_target(&suggested) else {
            return;
        };
        let request = ExportRequest {
            job,
            format,
            target,
        };
        if let Err(error) = self.controller.export(&request) {
            self.show_error(&error);
        }
        self.pump();
        self.commit();
    }

    /// Apply a drop-down choice by the label the widget reported.
    pub fn setting_label_changed(&self, kind: SettingKind, label: String) {
        self.assert_on_loop();
        let id = {
            let state = self.state.borrow();
            let choices: &[Choice] = match kind {
                SettingKind::Model => &state.app.models,
                SettingKind::Device => &state.app.devices,
                SettingKind::Language => &state.app.languages,
            };
            AppView::choice_id(choices, &label)
        };
        let Some(id) = id else {
            return;
        };
        let patch = match kind {
            SettingKind::Model => SettingsPatch {
                model: Some(id),
                ..Default::default()
            },
            SettingKind::Device => SettingsPatch {
                device: Some(id),
                ..Default::default()
            },
            SettingKind::Language => SettingsPatch {
                language: Some(id),
                ..Default::default()
            },
        };
        self.push_settings(patch);
    }

    /// Turn speaker separation on or off.
    pub fn set_diarize(&self, on: bool) {
        self.assert_on_loop();
        self.push_settings(SettingsPatch {
            diarize: Some(on),
            ..Default::default()
        });
    }

    /// Set the colour theme by drop-down index.
    pub fn set_theme_index(&self, index: i32) {
        self.assert_on_loop();
        let Some(theme) = theme_from_index(index) else {
            return;
        };
        self.push_settings(SettingsPatch {
            theme: Some(theme),
            ..Default::default()
        });
    }

    /// Show or hide the settings panel.
    pub fn toggle_settings(&self) {
        self.assert_on_loop();
        let mut state = self.state.borrow_mut();
        state.app.settings_open = !state.app.settings_open;
        drop(state);
        self.commit();
    }

    /// Hide the error banner.
    pub fn dismiss_banner(&self) {
        self.assert_on_loop();
        self.state.borrow_mut().app.banner.dismiss();
        self.commit();
    }

    /// The viewport moved or resized. Called from Slint, on the loop thread.
    pub fn viewport_changed(&self, offset: f32, height: f32, width: f32) {
        self.assert_on_loop();
        {
            let mut state = self.state.borrow_mut();
            state.viewport_offset_px = offset;
            state.viewport_height_px = height;
            let chars = chars_per_line(width);
            let relayout = (i64::try_from(chars).unwrap_or(i64::MAX)
                - i64::try_from(state.layout.chars_per_line()).unwrap_or(0))
            .abs()
                >= RELAYOUT_HYSTERESIS;
            if relayout {
                let borrowed = state.transcript.transcript().cloned();
                state.layout.relayout(borrowed.as_deref(), chars);
            }
        }
        self.commit();
    }

    fn push_settings(&self, patch: SettingsPatch) {
        if patch.is_empty() {
            return;
        }
        // Applied locally first so the control does not snap back, then sent. A
        // rejection puts the old value back and says why.
        {
            let mut state = self.state.borrow_mut();
            apply_patch(&mut state.app.settings, patch.clone());
        }
        // The colour scheme lives in Slint, not in our state, so it has to be
        // pushed explicitly. Forgetting this is a bug the settings drop-down
        // made invisible: the state said `Dark`, the window stayed `Light`, and
        // nothing anywhere disagreed loudly enough to notice.
        let theme = self.state.borrow().app.settings.theme;
        self.apply_theme(theme);
        self.commit();
        match self.controller.update_settings(patch) {
            Ok(settings) => {
                let theme = settings.theme;
                let mut state = self.state.borrow_mut();
                state.app.settings = settings;
                drop(state);
                self.apply_theme(theme);
            }
            Err(error) => {
                let theme = self.state.borrow().app.settings.theme;
                self.apply_theme(theme);
                self.show_error(&error);
            }
        }
        self.commit();
    }

    fn show_error(&self, error: &UserFacingError) {
        let mut state = self.state.borrow_mut();
        state.app.banner = Banner::from_error(Severity::Error, error);
        state.app.note(error.title.clone());
    }

    /// Ask the state to scroll so the selection is visible. Caller holds borrow.
    fn reveal_selection(&self, state: &mut ViewState) {
        let Some(index) = state.transcript.selection().focus else {
            return;
        };
        if index >= state.layout.segment_count() {
            return;
        }
        let line_height = TRANSCRIPT_LINE_HEIGHT_PX;
        let top = state.layout.lines_before(index) as f32 * line_height;
        let bottom = top + f32::from(state.layout.line_count(index)) * line_height;
        // A viewport of zero means the window has not been laid out yet. Assume
        // a screenful rather than refusing to scroll: a selection the user
        // cannot see is worse than a scroll that will be corrected next frame.
        let viewport = if state.viewport_height_px > 0.0 {
            state.viewport_height_px
        } else {
            400.0
        };
        let current = state.viewport_offset_px;
        let offset = if top < current {
            top
        } else if bottom > current + viewport {
            (bottom - viewport).max(0.0)
        } else {
            current
        };
        if (offset - current).abs() > 0.5 {
            state.pending_scroll_px = Some(offset);
        }
    }

    fn footer_text(&self, state: &ViewState) -> String {
        let mut parts = vec![
            format!("backend {}", state.app.backend_name),
            format!("host {}", state.app.host_name),
        ];
        if state.app.is_demo {
            parts.push("DEMO DATA".to_owned());
        }
        parts.push(format!(
            "{} running, {} finished",
            state.jobs.running_count(),
            state.jobs.finished_count()
        ));
        // Which device did the work, for the job on screen.
        //
        // Shown because the alternative is a machine with a GPU in it, a
        // transcript that took six times as long as it should have, and nothing
        // to explain it. Read from the *selected* job rather than "the newest
        // one that has a device", which would report a finished job's device
        // while you read a different file.
        if let Some(device) = state.jobs.selected_job().and_then(|j| j.device.as_deref()) {
            parts.push(device.to_owned());
        }
        if let Some(last) = state.app.recent.back() {
            parts.push(last.clone());
        }
        parts.join("  |  ")
    }

    fn scale_factor(&self) -> f32 {
        self.window
            .upgrade()
            .map_or(1.0, |w| w.window().scale_factor())
    }

    /// Apply the settings to the one thing Slint owns: the colour scheme.
    /// The colour scheme Slint is actually rendering with.
    ///
    /// Read from the global rather than from our own state, because those two
    /// disagreeing is exactly the bug this accessor exists to catch: a theme
    /// drop-down that updates the state and not the window looks perfectly
    /// correct in a state assertion and does nothing on screen.
    pub fn color_scheme(&self) -> ColorScheme {
        self.theme_globals()
            .map_or(ColorScheme::Unknown, |g| g.get_color_scheme())
    }

    /// The colour scheme as a name, for the window's status bar.
    fn color_scheme_name(&self) -> String {
        let name = match self.color_scheme() {
            ColorScheme::Light => "light",
            ColorScheme::Dark => "dark",
            _ => "system",
        };
        name.to_owned()
    }

    /// Push a colour scheme into Slint.
    ///
    /// Takes the theme as an argument rather than reading it, because the caller
    /// usually already holds the state borrow, and re-borrowing it here would be
    /// a runtime panic rather than a compile error.
    fn apply_theme(&self, theme: Theme) {
        if let Some(globals) = self.theme_globals() {
            globals.set_color_scheme(color_scheme_for(theme));
        }
    }

    /// The theme global, if the window still exists.
    ///
    /// A `Weak` upgrade can fail, and a window that has been dropped must not
    /// take the process down with it.
    fn theme_globals(&self) -> Option<crate::slint_api::VaaniColors<'static>> {
        self.theme.upgrade()
    }

    /// The current settings, for tests.
    pub fn settings(&self) -> AppSettings {
        self.state.borrow().app.settings.clone()
    }

    /// The export format the user has chosen, or the first one if the widget
    /// has not been touched.
    pub fn export_format(&self) -> ExportFormat {
        let index = self.window.upgrade().map_or(0, |w| w.get_export_format());
        usize::try_from(index)
            .ok()
            .and_then(|i| ExportFormat::ALL.get(i).copied())
            .unwrap_or_default()
    }

    /// The error banner, for tests.
    pub fn banner(&self) -> Banner {
        self.state.borrow().app.banner.clone()
    }

    /// The transcript view, for tests.
    pub fn transcript_view(&self) -> Ref<'_, TranscriptView> {
        Ref::map(self.state.borrow(), |s| &s.transcript)
    }

    /// The speaker panel, for tests.
    pub fn speaker_panel(&self) -> Ref<'_, SpeakerPanel> {
        Ref::map(self.state.borrow(), |s| &s.speakers)
    }

    /// The job list, for tests.
    pub fn job_list(&self) -> Ref<'_, JobListState> {
        Ref::map(self.state.borrow(), |s| &s.jobs)
    }

    /// The whole app view, for tests.
    pub fn app_view(&self) -> Ref<'_, AppView> {
        Ref::map(self.state.borrow(), |s| &s.app)
    }

    /// Force a commit. For tests, and for a host that changed something the
    /// window cannot see — the display's scale factor is the real case, since
    /// Slint reports it through a window event rather than a property.
    pub fn refresh(&self) {
        self.commit();
    }

    /// Re-read the display's scale factor and push it to the window.
    ///
    /// A window manager sends `ScaleFactorChanged` when the window moves to a
    /// screen with different scaling. The window's own state is already correct
    /// at that point; the number the status bar shows is not, because it is only
    /// read at commit time. This is the one place that needs saying out loud,
    /// because "the status bar says 100% on a 150% display" is a bug nobody
    /// else would find.
    pub fn refresh_scale_factor(&self) {
        self.assert_on_loop();
        self.commit();
    }

    /// Assert this is the event-loop thread.
    ///
    /// A hard assert, not a debug one. Writing a Slint property from the wrong
    /// thread usually appears to work and then corrupts a value nobody is
    /// watching; failing here, with a message that names the fix, is worth more
    /// than the rare false positive.
    pub fn assert_on_loop(&self) {
        assert_eq!(
            std::thread::current().id(),
            self.loop_thread,
            "a Slint property was written from the wrong thread; UI state changes must go \
             through UiModel::pump or a Slint callback, both of which run on the event loop"
        );
    }
}

/// Which settings drop-down a label came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingKind {
    /// The speech model.
    Model,
    /// The compute device.
    Device,
    /// The spoken language.
    Language,
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

/// The number of characters a transcript row of this width can hold.
#[must_use]
pub fn chars_per_line(width_px: f32) -> usize {
    let usable = (width_px - ROW_HORIZONTAL_PADDING_PX).max(60.0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let chars = (usable / PIXELS_PER_CHAR) as usize;
    chars.clamp(MIN_CHARS_PER_LINE, 400)
}

/// Reconcile a `VecModel` with a new list, touching only what changed.
///
/// A `for` repeat over a model redraws the rows whose data changed, so pushing
/// every row on every commit would repaint the whole window at 30 Hz for no
/// reason. The prefix/suffix diff below reduces a one-row progress update to a
/// single `set_row_data`.
fn sync_vec_model<T: Clone + PartialEq + 'static>(model: &Rc<VecModel<T>>, rows: &[T]) {
    let existing = model.row_count();
    let shared = (0..existing.min(rows.len()))
        .take_while(|&i| model.row_data(i).as_ref() == Some(&rows[i]))
        .count();
    if shared == existing && existing == rows.len() {
        return;
    }
    for _ in shared..existing {
        model.remove(0);
    }
    for row in rows.iter().skip(shared) {
        model.push(row.clone());
    }
}

fn to_slint_job_row(row: &view::JobRow) -> JobRowS {
    JobRowS {
        id: row.id.as_str().into(),
        name: row.name.as_str().into(),
        input_path: row.input_path.as_str().into(),
        phase_label: row.phase_label.as_str().into(),
        phase_code: row.phase_code.as_str().into(),
        progress: row.progress.map_or(-1.0, |f| f as f32),
        progress_indeterminate: row.progress_indeterminate,
        progress_text: row.progress_text.as_str().into(),
        eta_text: row.eta_text.as_str().into(),
        status_text: row.status_text.as_str().into(),
        status_glyph: row.status_glyph.as_str().into(),
        is_running: row.is_running,
        is_queued: row.is_queued,
        is_terminal: row.is_terminal,
        is_complete: row.is_complete,
        is_failed: row.is_failed,
        is_cancelled: row.is_cancelled,
        is_cancelling: row.is_cancelling,
        is_selected: row.is_selected,
        has_transcript: row.has_transcript,
        error_title: row.error_title.as_str().into(),
        error_recovery: row.error_recovery.as_str().into(),
        error_code: row.error_code.as_str().into(),
    }
}

fn to_slint_speaker_row(row: &view::SpeakerRow) -> SpeakerRowS {
    SpeakerRowS {
        index: i32::try_from(row.index.0).unwrap_or(i32::MAX),
        display_index: i32::try_from(row.index.0).unwrap_or(i32::MAX) + 1,
        name: row.name.as_str().into(),
        generated_name: row.generated_name.as_str().into(),
        talk_time: row.talk_time.as_str().into(),
        share_text: row.share_text.as_str().into(),
        share: row.share.unwrap_or(-1.0),
        is_selected: row.is_selected,
        is_editing: row.is_editing,
        is_named: row.is_named,
    }
}

fn to_slint_transcript_row(
    view: &TranscriptView,
    layout: &TranscriptLayout,
    index: usize,
) -> Option<TranscriptRow> {
    let row = view.row(index)?;
    Some(TranscriptRow {
        index: i32::try_from(row.index).ok()?,
        lines_before: i32::try_from(layout.lines_before(index)).ok()?,
        line_count: i32::from(layout.line_count(index)),
        timestamp: row.timestamp.into(),
        speaker: row.speaker.into(),
        text: row.text.into(),
        is_selected: row.is_selected,
        is_turn_start: row.is_turn_start,
        is_match: row.is_match,
        is_unattributed: row.is_unattributed,
    })
}

fn severity_level(level: Severity) -> i32 {
    match level {
        Severity::Info => 0,
        Severity::Warning => 1,
        Severity::Error => 2,
    }
}

fn theme_index(theme: Theme) -> i32 {
    match theme {
        Theme::System => 0,
        Theme::Light => 1,
        Theme::Dark => 2,
    }
}

fn theme_from_index(index: i32) -> Option<Theme> {
    match index {
        0 => Some(Theme::System),
        1 => Some(Theme::Light),
        2 => Some(Theme::Dark),
        _ => None,
    }
}

fn color_scheme_for(theme: Theme) -> ColorScheme {
    match theme {
        Theme::System => ColorScheme::Unknown,
        Theme::Light => ColorScheme::Light,
        Theme::Dark => ColorScheme::Dark,
    }
}

fn apply_patch(settings: &mut AppSettings, patch: SettingsPatch) {
    if let Some(v) = patch.model {
        settings.model = v;
    }
    if let Some(v) = patch.device {
        settings.device = v;
    }
    if let Some(v) = patch.language {
        settings.language = v;
    }
    if let Some(v) = patch.diarize {
        settings.diarize = v;
    }
    if let Some(v) = patch.theme {
        settings.theme = v;
    }
}

fn transcript_empty_message(state: &ViewState) -> String {
    if state.transcript.row_count() > 0 {
        return "No segments".to_owned();
    }
    match state.jobs.selected_job() {
        Some(job) if job.phase.is_active() => {
            format!("{} is still running", view::file_name_of(&job.input))
        }
        Some(job) if job.phase == JobPhase::Complete => "This job produced no speech".to_owned(),
        Some(_) => "No transcript for this job".to_owned(),
        None => "No transcript yet".to_owned(),
    }
}

/// Whether an event about `event` may replace what is on screen.
///
/// The transcript pane shows exactly one job, and only the *selected* job may
/// put its words there. This is the rule that says so, in one place, because
/// the alternative is a transcript that swaps in mid-sentence: every finished
/// job emits `TranscriptReady`, `load_job` replaces the pane, and a user
/// transcribing two files would be reading one transcript while the other
/// replaced it with no visible cause.
///
/// A newly queued job is included because when it is also the selected one -
/// the first job, or the only job - its empty pane is the correct thing to
/// show, and skipping it would leave the previous file's words on screen under
/// the new file's name.
fn may_replace_the_pane(selected: Option<&JobId>, event: &JobId) -> bool {
    selected == Some(event)
}

#[cfg(test)]
mod selection_rule_tests {
    use super::may_replace_the_pane;
    use crate::JobId;

    #[test]
    fn a_background_job_may_not_replace_the_pane() {
        let selected = JobId::from("reading-this");
        let background = JobId::from("finished-first");
        assert!(
            !may_replace_the_pane(Some(&selected), &background),
            "the job the user is reading is the only one that may put words on screen"
        );
    }

    #[test]
    fn the_selected_job_may_replace_the_pane() {
        let selected = JobId::from("reading-this");
        assert!(may_replace_the_pane(Some(&selected), &selected));
    }

    #[test]
    fn nothing_is_shown_when_nothing_is_selected() {
        let any = JobId::from("whatever");
        assert!(
            !may_replace_the_pane(None, &any),
            "with no selection there is no transcript to show, so no event may invent one"
        );
    }
}

#[cfg(test)]
mod tests {
    use vaani_core::Word;

    use super::*;
    use crate::view::{format_millis, format_word_count};

    #[test]
    fn chars_per_line_shrinks_as_the_pane_narrows() {
        assert!(chars_per_line(1200.0) > chars_per_line(400.0));
    }

    #[test]
    fn chars_per_line_never_collapses_to_nothing() {
        // A pane dragged to nothing must not produce one character per line,
        // which would make every segment look like a paragraph.
        assert_eq!(chars_per_line(0.0), MIN_CHARS_PER_LINE);
        assert_eq!(chars_per_line(-100.0), MIN_CHARS_PER_LINE);
        assert!(chars_per_line(10_000.0) <= 400);
    }

    #[test]
    fn layout_totals_add_up() {
        let t = crate::sample::sample_transcript(500, 2);
        let layout = TranscriptLayout::build(Some(&t), 80);
        assert_eq!(layout.segment_count(), t.segments.len());
        let sum: u32 = (0..layout.segment_count())
            .map(|i| u32::from(layout.line_count(i)))
            .sum();
        assert_eq!(sum, layout.total_lines());
    }

    #[test]
    fn layout_lines_before_is_a_prefix_sum() {
        let t = crate::sample::sample_transcript(500, 2);
        let layout = TranscriptLayout::build(Some(&t), 80);
        assert_eq!(layout.lines_before(0), 0);
        let mut running = 0;
        for i in 0..layout.segment_count() {
            assert_eq!(layout.lines_before(i), running);
            running += u32::from(layout.line_count(i));
        }
    }

    #[test]
    fn layout_of_an_empty_transcript_is_empty_not_a_panic() {
        let layout = TranscriptLayout::build(None, 80);
        assert_eq!(layout.segment_count(), 0);
        assert_eq!(layout.total_lines(), 0);
        assert_eq!(layout.visible_range(0.0, 500.0, 22.0, 8), 0..0);
    }

    #[test]
    fn visible_range_never_exceeds_the_viewport_plus_overscan() {
        // This is the windowing claim, as a number: scrolling a 20,000-word
        // transcript must never build more than a screenful plus overscan.
        let t = crate::sample::sample_transcript(20_000, 2);
        let layout = TranscriptLayout::build(Some(&t), 80);
        let count = layout.segment_count();
        assert!(
            count > 1_000,
            "need a long transcript for this to mean anything"
        );

        let line_height = TRANSCRIPT_LINE_HEIGHT_PX;
        let viewport = 700.0;
        let total_px = layout.total_lines() as f32 * line_height;
        let rows_per_screen = (viewport / line_height) as usize + 1;
        let mut offset = 0.0;
        let mut worst = 0;
        while offset < total_px {
            let range = layout.visible_range(offset, viewport, line_height, OVERSCAN_ROWS);
            assert!(!range.is_empty(), "empty window at offset {offset}");
            assert!(range.end <= count);
            worst = worst.max(range.end - range.start);
            offset += viewport;
        }
        assert!(
            worst <= rows_per_screen + 2 * OVERSCAN_ROWS,
            "window grew to {worst} rows for a {rows_per_screen}-row viewport"
        );
    }

    #[test]
    fn visible_range_at_the_very_end_still_produces_rows() {
        let t = crate::sample::sample_transcript(500, 2);
        let layout = TranscriptLayout::build(Some(&t), 80);
        let total_px = layout.total_lines() as f32 * TRANSCRIPT_LINE_HEIGHT_PX;
        let count = layout.segment_count();
        let range = layout.visible_range(total_px, 500.0, TRANSCRIPT_LINE_HEIGHT_PX, OVERSCAN_ROWS);
        assert!(range.end <= count);
        assert!(!range.is_empty(), "the last screen must have content");
    }

    #[test]
    fn a_zero_sized_viewport_builds_nothing_rather_than_everything() {
        // The viewport reports zero while the window is hidden. Building every
        // row then would be the exact bug the windowing exists to prevent.
        let t = crate::sample::sample_transcript(2_000, 2);
        let layout = TranscriptLayout::build(Some(&t), 80);
        let range = layout.visible_range(0.0, 0.0, TRANSCRIPT_LINE_HEIGHT_PX, OVERSCAN_ROWS);
        assert!(
            range.end - range.start <= 2 * OVERSCAN_ROWS + 1,
            "a zero-height viewport built {} rows",
            range.end - range.start
        );
    }

    #[test]
    fn relayout_ignores_a_change_too_small_to_matter() {
        let t = crate::sample::sample_transcript(500, 2);
        let mut layout = TranscriptLayout::build(Some(&t), 80);
        let before = layout.clone();
        layout.relayout(Some(&t), 82);
        assert_eq!(layout, before, "a two-character change should not relayout");
        layout.relayout(Some(&t), 120);
        assert_ne!(layout, before);
    }

    #[test]
    fn layout_clamps_a_pathologically_long_segment() {
        let mut t = crate::sample::sample_transcript(50, 1);
        t.segments[0].words = vec![Word::new(
            "x".repeat(200_000),
            vaani_core::Time(0),
            vaani_core::Time(1_000),
        )];
        let layout = TranscriptLayout::build(Some(&t), 40);
        assert_eq!(layout.line_count(0), MAX_LINES_PER_SEGMENT);
    }

    #[test]
    fn every_row_reserves_its_header_line() {
        // Every rendered row shows timestamp and speaker above the text, and
        // that header occupies a full line height. Counting only text lines
        // made each row one line short so all of them bled into each other —
        // the overlapping-transcript bug. A one-word segment needs two lines
        // (header plus its word), never one.
        let mut t = crate::sample::sample_transcript(10, 0);
        t.segments[0].words = vec![Word::new(
            "hello",
            vaani_core::Time(0),
            vaani_core::Time(500),
        )];
        let layout = TranscriptLayout::build(Some(&t), 80);
        assert_eq!(
            layout.line_count(0),
            2,
            "one header line plus one text line, not one line total"
        );
    }

    #[test]
    fn theme_indices_round_trip() {
        for theme in [Theme::System, Theme::Light, Theme::Dark] {
            assert_eq!(theme_from_index(theme_index(theme)), Some(theme));
        }
        assert_eq!(theme_from_index(3), None);
        assert_eq!(theme_from_index(-1), None);
    }

    #[test]
    fn settings_patch_applies_only_what_it_names() {
        let mut settings = AppSettings {
            model: "a".to_owned(),
            diarize: true,
            ..AppSettings::default()
        };
        apply_patch(
            &mut settings,
            SettingsPatch {
                theme: Some(Theme::Dark),
                ..Default::default()
            },
        );
        assert_eq!(settings.theme, Theme::Dark);
        assert_eq!(settings.model, "a");
        assert!(settings.diarize);
    }

    #[test]
    fn sync_vec_model_appends_removes_and_replaces() {
        let model = Rc::new(VecModel::<i32>::default());
        sync_vec_model(&model, &[1, 2, 3]);
        assert_eq!(model.row_count(), 3);
        sync_vec_model(&model, &[1, 2, 3]);
        assert_eq!(model.row_count(), 3, "no change is a no-op");
        sync_vec_model(&model, &[1, 9, 3]);
        assert_eq!(model.row_data(1), Some(9));
        sync_vec_model(&model, &[1, 9]);
        assert_eq!(model.row_count(), 2);
        sync_vec_model(&model, &[7, 8, 9, 10]);
        assert_eq!(model.row_count(), 4);
        assert_eq!(model.row_data(0), Some(7));
        assert_eq!(model.row_data(3), Some(10));
        sync_vec_model(&model, &[]);
        assert_eq!(model.row_count(), 0);
    }

    #[test]
    fn a_large_transcript_word_count_is_exact() {
        let t = crate::sample::sample_transcript(20_000, 2);
        let expected: usize = t.segments.iter().map(|s| s.words.len()).sum();
        let mut view = TranscriptView::new();
        view.set_transcript(Some(Arc::new(t)));
        assert_eq!(view.word_count(), expected);
        assert!(
            expected >= 20_000,
            "the fixture must really be a 20,000-word transcript, not {expected}"
        );
    }

    #[test]
    fn summarising_a_large_transcript_is_fast() {
        // The number quoted in the report depends on this: a 20,000-word
        // transcript must summarise without walking every word.
        let t = crate::sample::sample_transcript(20_000, 2);
        let mut view = TranscriptView::new();
        view.set_transcript(Some(Arc::new(t)));
        let start = std::time::Instant::now();
        let summary = view.summary();
        let elapsed = start.elapsed();
        assert!(summary.contains("words"));
        assert!(
            elapsed < Duration::from_millis(50),
            "summarising took {elapsed:?}"
        );
    }

    #[test]
    fn formatting_helpers_used_by_the_footer_are_covered() {
        assert_eq!(format_millis(0), "0s");
        assert_eq!(format_word_count(3), "3 words");
    }
}

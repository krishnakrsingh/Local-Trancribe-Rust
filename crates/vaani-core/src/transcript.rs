//! The transcript data model.
//!
//! This is the shape the whole product agrees on: ASR produces it, diarization
//! annotates it, the UI renders it, the exporters serialise it, SQLite persists
//! it. Changes here are expensive, so the invariants are encoded in the types
//! and enforced in `Transcript`'s constructors rather than left to convention.
//!
//! # Timestamps
//!
//! There are three timestamp types and they are not interchangeable. The full
//! rules are in `docs/architecture/TIMESTAMP_CONTRACT.md`; the short version:
//!
//! * [`Time`] is a **position** — a signed count of milliseconds from the start
//!   of the *source media*, never from the start of a chunk or a segment.
//! * [`MediaLength`] is an **amount** — a count of milliseconds, never negative,
//!   never a position.
//! * [`MediaWindow`] is the **extent of the media**, in positions.
//!
//! A `Time` and a `MediaLength` do not compare, and neither converts into the
//! other implicitly. That is deliberate: `Transcript::duration` used to be a
//! `Time` and comparing it against an absolute segment end compiled silently
//! and produced two real bugs in two different crates.
//!
//! Chunk boundaries are an implementation detail of the ASR pipeline; every
//! offset that leaves the `asr` crate has already been rebased onto the media
//! timeline. Getting that wrong is the classic source of "the transcript is
//! three minutes ahead of the video" bugs, so the rebasing happens in exactly
//! one place (see `Transcript::push_chunk`), and clamping to the media happens
//! through exactly one type (see [`MediaWindow`]).

use serde::{Deserialize, Serialize};
use std::fmt;

/// A millisecond offset from the start of the source media.
///
/// `i64` rather than `u64` because ASR adapters legitimately produce a small
/// negative start for a chunk whose first word begins marginally before the
/// chunk boundary (we keep overlap precisely so words are not clipped), and we
/// would rather clamp that in one place than fight an underflow in many.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
#[allow(missing_docs)] // the single field is documented on the type
pub struct Time(pub i64);

impl Time {
    /// The start of the media.
    pub const ZERO: Time = Time(0);

    /// Convert from fractional seconds, rounding to the nearest millisecond.
    pub fn from_secs_f64(s: f64) -> Self {
        // `round` not `floor`: a word boundary at 1999.6 ms should land on
        // 2000 ms, and consistent rounding avoids sub-millisecond drift when
        // the same time is converted repeatedly.
        Time((s * 1000.0).round() as i64)
    }

    /// Construct from a millisecond count.
    ///
    /// A `Time` here is always a **position**. If the number you have is an
    /// amount of time, construct a [`MediaLength`] instead — that is the whole
    /// difference this contract is about.
    pub fn from_millis(ms: i64) -> Self {
        Time(ms)
    }

    /// The raw millisecond offset.
    pub fn as_millis(self) -> i64 {
        self.0
    }

    /// The offset in fractional seconds, for feeding inference code.
    pub fn as_secs_f64(self) -> f64 {
        self.0 as f64 / 1000.0
    }

    /// Offset by `other`, clamping instead of wrapping on overflow.
    pub fn saturating_add(self, other: Time) -> Time {
        Time(self.0.saturating_add(other.0))
    }

    /// Offset backwards by `other`, clamping instead of wrapping.
    pub fn saturating_sub(self, other: Time) -> Time {
        Time(self.0.saturating_sub(other.0))
    }

    /// Never negative. A transcript that starts before 0 is lying about the
    /// media; clamping here means every downstream consumer can assume
    /// `start >= 0` without re-checking.
    /// Clamp to the start of the media.
    pub fn clamp_zero(self) -> Time {
        Time(self.0.max(0))
    }

    /// `hh:mm:ss` — the form used in the transcript viewer.
    pub fn format_hms(self) -> String {
        let total = self.0.max(0);
        let (h, m, s) = (
            total / 3_600_000,
            (total / 60_000) % 60,
            (total / 1000) % 60,
        );
        if h > 0 {
            format!("{h:02}:{m:02}:{s:02}")
        } else {
            format!("{m:02}:{s:02}")
        }
    }

    /// `hh:mm:ss.mmm` — the unambiguous form used in SRT/VTT and JSON.
    pub fn format_hms_millis(self) -> String {
        let total = self.0.max(0);
        let (h, m, s, ms) = (
            total / 3_600_000,
            (total / 60_000) % 60,
            (total / 1000) % 60,
            total % 1000,
        );
        format!("{h:02}:{m:02}:{s:02}.{ms:03}")
    }

    /// The **length** of the interval from `earlier` to `self`, never negative.
    ///
    /// The only sanctioned way to turn two positions into an amount. Two
    /// positions in either order have a well-defined distance; only one of them
    /// is a forward one, so the answer is clamped at zero rather than signed.
    ///
    /// ```
    /// # use vaani_core::Time;
    /// let start = Time(1_000);
    /// let end = Time(3_500);
    /// assert_eq!(start.span_to(end).as_millis(), 2_500);
    /// // Backwards gives zero, not a negative duration.
    /// assert_eq!(end.span_to(start).as_millis(), 0);
    /// ```
    #[must_use]
    pub fn span_to(self, later: Time) -> MediaLength {
        MediaLength(later.0.saturating_sub(self.0).max(0))
    }

    /// Advance this position by a **length**.
    ///
    /// The other half of [`span_to`](Self::span_to): a media *end* is a media
    /// *start* plus a media *length*, and this is the only way to build one.
    /// `MediaWindow::end` is the same arithmetic, named.
    ///
    /// ```
    /// # use vaani_core::{MediaLength, Time};
    /// let origin = Time(300_000);
    /// let end = origin.offset_by(MediaLength::from_millis(89_600));
    /// assert_eq!(end, Time(389_600));
    /// ```
    #[must_use]
    pub fn offset_by(self, length: MediaLength) -> Time {
        Time(self.0.saturating_add(length.0))
    }
}

impl fmt::Display for Time {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.format_hms())
    }
}

/// An **amount** of time in milliseconds — a length, never a position.
///
/// # Why this is a separate type from [`Time`]
///
/// It is not tidiness. `Transcript::duration` used to be a `Time` and
/// `Segment::start` was a `Time`, and nothing said which was which, so
/// `segment.end > transcript.duration` compiled and was wrong: one is a
/// position on the media timeline and the other is how long the media is.
/// Doing that in `vaani-segment` pinned every segment of a transcript that does
/// not start at zero to the origin, and the same confusion in `vaani-app`'s
/// `Chunker::plan` call cost a known-failure entry in the conformance
/// registry. Both were arithmetic that read correctly.
///
/// A `MediaLength` compares only with another `MediaLength`, and there is no
/// `From<Time>` for it, so the mistake no longer type-checks:
///
/// ```compile_fail
/// # use vaani_core::{EngineStamp, Segment, Time, Transcript};
/// let mut t = Transcript::new(EngineStamp::asr("m", "1"));
/// t.push_chunk(Time::ZERO, vec![Segment::new(Time(0), Time(9_000))]);
/// let bound = t.duration.unwrap();
/// // Fails to compile: `Time` cannot be compared with `MediaLength`.
/// if t.segments[0].end > bound { }
/// ```
///
/// The two legal ways to produce one are [`Time::span_to`] (a length between
/// two positions) and [`MediaLength::from_millis`] (a number you already know
/// is an amount, such as a decoded sample count).
///
/// It is `#[serde(transparent)]` over `i64`, so an exported JSON envelope and a
/// SQLite `duration_ms` column are byte-identical to what they were before this
/// type existed. Only the Rust type changed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct MediaLength(i64);

impl MediaLength {
    /// No time at all.
    pub const ZERO: MediaLength = MediaLength(0);

    /// An amount of `millis` milliseconds. Negative input clamps to zero.
    #[must_use]
    pub fn from_millis(millis: i64) -> Self {
        MediaLength(millis.max(0))
    }

    /// An amount of `secs` seconds, rounded to the nearest millisecond.
    #[must_use]
    pub fn from_secs_f64(secs: f64) -> Self {
        MediaLength((secs * 1000.0).round().max(0.0) as i64)
    }

    /// The amount in milliseconds.
    #[must_use]
    pub fn as_millis(self) -> i64 {
        self.0
    }

    /// The amount in fractional seconds, for feeding inference code.
    #[must_use]
    pub fn as_secs_f64(self) -> f64 {
        self.0 as f64 / 1000.0
    }

    /// `hh:mm:ss`, clamped at zero. The form the UI and the exporters use.
    #[must_use]
    pub fn format_hms(self) -> String {
        Time(self.0).format_hms()
    }

    /// Total of two amounts, saturating rather than wrapping.
    #[must_use]
    pub fn saturating_add(self, other: MediaLength) -> MediaLength {
        MediaLength(self.0.saturating_add(other.0))
    }

    /// A non-zero amount, if this is one.
    #[must_use]
    pub fn nonzero(self) -> Option<MediaLength> {
        (self.0 > 0).then_some(self)
    }
}

impl fmt::Display for MediaLength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.format_hms())
    }
}

/// The absolute extent of the source media: `start..end`, both positions.
///
/// # Why the clamp needs a type
///
/// The last STFT frame legitimately extends past the audio, so a time can land
/// outside the recording and has to be clamped. Clamping needs two numbers — an
/// origin and an end — and the only number a caller usually has to hand is a
/// *length*. Comparing an absolute end against a length is the bug this whole
/// contract exists to remove, and it is exactly what `vaani-segment` did before
/// it had this type to reach for.
///
/// So the window is named, constructed in one of three ways, and every clamp in
/// the product takes one. There is no way to state "the media is this long" and
/// have it silently become a position.
///
/// ```
/// # use vaani_core::{MediaLength, MediaWindow, Time};
/// // A whole recording that starts at zero.
/// let whole = MediaWindow::of_length(MediaLength::from_millis(10_000));
/// assert_eq!((whole.start(), whole.end()), (Time::ZERO, Time(10_000)));
///
/// // A slice handed between stages, at +5 minutes.
/// let slice = MediaWindow::new(Time(300_000), Time(389_600));
/// assert_eq!(slice.length().as_millis(), 89_600);
/// assert!(slice.contains(Time(350_000)));
/// // The same amount read as a bare length is NOT a position, and the type
/// // will not let you pretend otherwise: see `MediaLength`.
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaWindow {
    start: Time,
    end: Time,
}

impl MediaWindow {
    /// The window an `AudioBuffer` occupies: its origin plus its length.
    ///
    /// This is the constructor a pipeline wants. The decoded buffer is the
    /// authority on where the media is, and building the window from it is what
    /// makes the length/absolute distinction impossible to get wrong at the
    /// call site.
    #[must_use]
    pub fn of_buffer(buffer: &crate::media::AudioBuffer) -> Self {
        Self {
            start: buffer.start_time,
            end: buffer.start_time.offset_by(buffer.length()),
        }
    }

    /// The window of a whole recording that **starts at zero**.
    ///
    /// The name states the assumption. If the media might start at a non-zero
    /// offset — a resumed job, a slice, a second input source — use
    /// [`of_buffer`](Self::of_buffer) or [`new`](Self::new) instead, because a
    /// window built this way is `0..length` and a transcript whose times run
    /// `300_000..389_600` lies entirely outside it.
    #[must_use]
    pub fn of_length(length: MediaLength) -> Self {
        Self {
            start: Time::ZERO,
            end: Time::ZERO.offset_by(length),
        }
    }

    /// A window stated outright, from two absolute positions.
    ///
    /// Used when the extent comes from somewhere that is not an
    /// `AudioBuffer` — a slice index, a persisted job, a conformance fixture.
    /// An inverted pair is normalised to a zero-length window at `start`, so a
    /// window is never its own negation.
    #[must_use]
    pub fn new(start: Time, end: Time) -> Self {
        Self {
            start,
            end: end.max(start),
        }
    }

    /// Where the media starts, absolutely.
    #[must_use]
    pub fn start(&self) -> Time {
        self.start
    }

    /// Where the media ends, absolutely. The clamp bound.
    #[must_use]
    pub fn end(&self) -> Time {
        self.end
    }

    /// How long the media is — an amount, derived, never stored twice.
    #[must_use]
    pub fn length(&self) -> MediaLength {
        self.start.span_to(self.end)
    }

    /// Whether `t` lies inside the window, half-open at the end.
    #[must_use]
    pub fn contains(&self, t: Time) -> bool {
        t >= self.start && t < self.end
    }

    /// The furthest into the window `t` may be. The only legal clamp.
    #[must_use]
    pub fn clamp(&self, t: Time) -> Time {
        t.max(self.start).min(self.end)
    }

    /// A `[start, end)` pair clamped into the window and repaired if inverted.
    #[must_use]
    pub fn clamp_span(&self, start: Time, end: Time) -> (Time, Time) {
        let s = self.clamp(start);
        let e = self.clamp(end);
        if e < s { (s, s) } else { (s, e) }
    }
}

/// A single recognised word with its time span.
///
/// `text` is the surface form including whatever the tokenizer produced;
/// `normalized` holds post-processing output when it differs from the surface
/// form. Keeping both is what lets the UI show an honest RAW transcript while
/// still offering a CLEAN view, and lets the user always see the original.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct Word {
    /// Surface form as recognised, e.g. `acreage`.
    pub text: String,
    pub start: Time,
    pub end: Time,
    /// Probability/confidence in [0,1] when the engine reports one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// Post-processed text, present only when it differs from `text`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalized: Option<String>,
}

impl Word {
    /// A word with an explicit time span.
    pub fn new(text: impl Into<String>, start: Time, end: Time) -> Self {
        Self {
            text: text.into(),
            start,
            end,
            confidence: None,
            normalized: None,
        }
    }

    /// The text to display. Falls back to the surface form, which is the
    /// truthful default: never silently replace a word the model produced.
    pub fn display_text(&self) -> &str {
        self.normalized.as_deref().unwrap_or(&self.text)
    }
}

/// A contiguous stretch of speech attributed to one speaker.
///
/// A segment is the atomic unit of the user-facing transcript: what you click
/// to seek, copy, delete, split, or merge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct Segment {
    /// Stable within a job. Referenced by edits and by the UI's selection
    /// model, so it must survive re-export and database round-trips.
    pub id: String,
    pub start: Time,
    pub end: Time,
    /// Index into `Transcript::speakers`. `None` means "not yet diarized",
    /// which is a normal state — ASR-only mode is a complete product mode,
    /// so "no speaker" is never an error.
    pub speaker: Option<usize>,
    /// BCP-47-ish tag for the language actually spoken in this segment. A
    /// meeting is frequently trilingual, so this is per-segment, not
    /// per-document.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    pub words: Vec<Word>,
    /// Recogniser confidence in [0,1] for the segment as a whole.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

impl Segment {
    /// An empty segment with a generated id.
    pub fn new(start: Time, end: Time) -> Self {
        Self {
            id: new_segment_id(),
            start,
            end,
            speaker: None,
            language: None,
            words: Vec::new(),
            confidence: None,
        }
    }

    /// Attach the recognised words.
    pub fn with_words(mut self, words: Vec<Word>) -> Self {
        self.words = words;
        self
    }

    /// Assign a speaker. `None` is the normal ASR-only state.
    pub fn with_speaker(mut self, speaker: Option<usize>) -> Self {
        self.speaker = speaker;
        self
    }

    /// The segment's text, joining words with single spaces.
    ///
    /// Derived from words rather than stored, so an edit to a word can never
    /// leave a stale duplicate `text` field disagreeing with it.
    pub fn text(&self) -> String {
        if self.words.is_empty() {
            return String::new();
        }
        let mut out = String::new();
        for (i, w) in self.words.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            out.push_str(w.display_text());
        }
        out
    }

    /// Length of the segment — an **amount**, derived, never negative.
    ///
    /// A `MediaLength` rather than a `Time` so that it cannot be compared
    /// against this segment's own `start` or `end`, which are positions.
    #[must_use]
    pub fn duration(&self) -> MediaLength {
        self.start.span_to(self.end)
    }
}

/// A speaker as shown in the transcript.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct Speaker {
    /// Cluster index. Stable for the life of a job, referenced by
    /// `Segment::speaker`.
    pub index: usize,
    /// User-supplied name. `None` means the UI shows the generated
    /// `Speaker N` label. Renaming must never rewrite the transcript text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Fraction of total speech time, in [0,1]. `None` before diarization.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub share: Option<f32>,
}

impl Speaker {
    /// An unnamed speaker at `index`.
    pub fn new(index: usize) -> Self {
        Self {
            index,
            name: None,
            share: None,
        }
    }

    /// What to render: the user's name if they gave one, else `Speaker N`.
    /// The name to show in the UI.
    pub fn display_name(&self) -> String {
        self.name
            .clone()
            .unwrap_or_else(|| format!("Speaker {}", self.index + 1))
    }
}

/// Normalisation level applied to a transcript.
///
/// Ordered from least to most intervention. Every level preserves the verbatim
/// text somewhere; a user must always be able to get back to what was actually
/// said. See `docs/agent/ARCHITECTURE.md` for why RAW is the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TranscriptMode {
    /// Closest faithful transcript. No rewriting of any kind.
    #[default]
    Raw,
    /// Fixes obvious filler, repetition and punctuation artefacts. Words are
    /// never substituted or removed, only spacing/casing/duplication.
    Clean,
    /// Deterministic meeting-transcript structure: title, participants,
    /// topics, decisions, action items. Verbatim turns are preserved verbatim
    /// inside the discussion section.
    Structured,
}

/// A complete transcription result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct Transcript {
    pub id: String,
    /// How long the source media is, in milliseconds. `None` for a live or
    /// streaming source, and for anything that was never measured.
    ///
    /// A **length**, in the type-system sense: see [`MediaLength`]. It is the
    /// figure the UI shows as "Duration" and the one written to `duration_ms`
    /// in SQLite and in an exported JSON envelope, and it is **not** a bound on
    /// any timestamp. To clamp a time to the media, build a [`MediaWindow`] and
    /// use that; `segment.end > transcript.duration` does not compile, which is
    /// the whole point.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<MediaLength>,
    pub segments: Vec<Segment>,
    pub speakers: Vec<Speaker>,
    pub mode: TranscriptMode,
    /// Human-readable description of what produced this: model id + version.
    /// Stored so a transcript exported months later says which model made it.
    pub engine: EngineStamp,
    /// True once the user has edited anything. Edits are never applied
    /// silently to the ASR output; this flag records that they happened.
    #[serde(default)]
    pub edited: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct EngineStamp {
    pub asr_model: String,
    pub asr_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diarization_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structuring_model: Option<String>,
}

impl EngineStamp {
    /// A stamp naming only the ASR model, which is always present.
    pub fn asr(model: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            asr_model: model.into(),
            asr_version: version.into(),
            ..Default::default()
        }
    }
}

impl Transcript {
    /// An empty transcript attributed to `engine`.
    pub fn new(engine: EngineStamp) -> Self {
        Self {
            id: new_transcript_id(),
            duration: None,
            segments: Vec::new(),
            speakers: Vec::new(),
            mode: TranscriptMode::Raw,
            engine,
            edited: false,
        }
    }

    /// Append a segment, rebasing its times onto the media timeline.
    ///
    /// `chunk_offset` is where the chunk that produced it began in the media.
    /// This is the *single* place chunk-local times become media times; every
    /// other crate works in chunk-local coordinates and calls this.
    ///
    /// Segments are kept sorted by start time. ASR engines emit in order, but
    /// overlapping chunks can produce a slightly out-of-order first segment, so
    /// we insert rather than push. For a typical meeting this is a single
    /// binary-search insertion per chunk, not a sort.
    pub fn push_chunk(&mut self, chunk_offset: Time, segments: Vec<Segment>) {
        self.push_chunk_inner(chunk_offset, segments, None);
    }

    /// Like [`push_chunk`](Self::push_chunk), but no time may leave
    /// `window`.
    ///
    /// The last STFT frame legitimately extends past the audio — measured on a
    /// real 834.197 s recording, whose last word ended at 834.220 s — and
    /// without a media window to clamp against there is nowhere correct for it
    /// to go. Callers that have the decoded buffer should always prefer this;
    /// the rebasing, ordering and zero-clamp are identical, only the bounds are
    /// new.
    ///
    /// A [`MediaWindow`], not a `Time`, because a clamp needs an origin as well
    /// as an end. Passing `window.end()` here would be correct arithmetic and
    /// still wrong whenever the media does not start at zero — the exact defect
    /// recorded in `docs/architecture/TIMESTAMP_CONTRACT.md` §1.
    pub fn push_chunk_within(
        &mut self,
        chunk_offset: Time,
        segments: Vec<Segment>,
        window: &MediaWindow,
    ) {
        self.push_chunk_inner(chunk_offset, segments, Some(*window));
    }

    fn push_chunk_inner(
        &mut self,
        chunk_offset: Time,
        mut segments: Vec<Segment>,
        window: Option<MediaWindow>,
    ) {
        for s in &mut segments {
            s.start = s.start.saturating_add(chunk_offset);
            s.end = s.end.saturating_add(chunk_offset);
            if let Some(w) = window {
                // Clamp to the media before repairing degenerates, so the
                // repair below cannot push an end back past the media.
                s.start = w.clamp(s.start);
                s.end = w.clamp(s.end);
            } else {
                s.start = s.start.clamp_zero();
                s.end = s.end.clamp_zero();
            }
            if s.end < s.start {
                s.end = s.start;
            }
            for w in &mut s.words {
                w.start = w.start.saturating_add(chunk_offset);
                w.end = w.end.saturating_add(chunk_offset);
                if let Some(window) = window {
                    w.start = window.clamp(w.start);
                    w.end = window.clamp(w.end);
                } else {
                    w.start = w.start.clamp_zero();
                    w.end = w.end.clamp_zero();
                }
                if w.end < w.start {
                    w.end = w.start;
                }
            }
        }
        let first = segments.first().map(|s| s.start).unwrap_or_default();
        let at = self
            .segments
            .partition_point(|existing| existing.start <= first);
        for (i, s) in segments.into_iter().enumerate() {
            self.segments.insert(at + i, s);
        }
    }

    /// The segment active at `t`, or the one immediately before it.
    ///
    /// Playback highlighting wants "what is being said right now". Returning
    /// the preceding segment rather than nothing means the highlight lingers
    /// through a gap instead of blinking off, which reads much better.
    pub fn segment_at(&self, t: Time) -> Option<&Segment> {
        let idx = self.segments.partition_point(|s| s.start <= t);
        if idx == 0 {
            // Before the first segment: highlight it if it starts very soon,
            // otherwise nothing is playing.
            return self
                .segments
                .first()
                .filter(|s| s.start.as_millis() - t.as_millis() < 1000);
        }
        let candidate = &self.segments[idx - 1];
        if candidate.end < t {
            // We are past the candidate's end. If the next segment has not
            // started yet we are in a gap: keep showing the candidate.
            return Some(candidate);
        }
        Some(candidate)
    }

    /// Index of the first segment whose end is strictly after `t`. Used by
    /// search-next and by playback to avoid re-highlighting the same segment.
    pub fn first_segment_after(&self, t: Time) -> usize {
        self.segments.partition_point(|s| s.end <= t)
    }

    /// Register or replace a speaker, returning its index.
    pub fn add_speaker(&mut self, speaker: Speaker) -> usize {
        let idx = speaker.index;
        match self.speakers.iter_mut().find(|s| s.index == idx) {
            Some(existing) => *existing = speaker,
            None => self.speakers.push(speaker),
        }
        idx
    }

    /// Speaker for an index, materialising a placeholder if diarization
    /// assigned an index it never registered.
    pub fn speaker(&self, index: usize) -> Option<&Speaker> {
        self.speakers.iter().find(|s| s.index == index)
    }

    /// Mutable speaker accessor, materialising a placeholder if diarization
    /// assigned an index it never registered. This is how a rename happens:
    /// `transcript.speaker_mut(0).name = Some("Krishna".into())`.
    pub fn speaker_mut(&mut self, index: usize) -> &mut Speaker {
        if !self.speakers.iter().any(|s| s.index == index) {
            self.speakers.push(Speaker::new(index));
            self.speakers.sort_by_key(|s| s.index);
        }
        // Invariant: after the insert+sort above there is exactly one entry
        // with this index, and indices are unique.
        let pos = self
            .speakers
            .iter()
            .position(|s| s.index == index)
            .expect("speaker index was just ensured to exist");
        &mut self.speakers[pos]
    }

    /// Recompute each speaker's share of total speech time.
    pub fn recompute_speaker_shares(&mut self) {
        if self.speakers.is_empty() {
            return;
        }
        // Accumulate first, then write: the shares are a fraction of the total
        // so no speaker can be finalised before every segment has been counted.
        let mut totals: Vec<i64> = vec![0; self.speakers.len()];
        let mut total = 0i64;
        for seg in &self.segments {
            let Some(spk) = seg.speaker else { continue };
            let Some(pos) = self.speakers.iter().position(|s| s.index == spk) else {
                continue;
            };
            let d = seg.duration().as_millis();
            totals[pos] = totals[pos].saturating_add(d);
            total = total.saturating_add(d);
        }
        for (slot, spoken) in self.speakers.iter_mut().zip(totals) {
            slot.share = Some(if total > 0 {
                spoken as f32 / total as f32
            } else {
                0.0
            });
        }
    }

    /// Full plain text, one speaker turn per paragraph.
    pub fn to_plain_text(&self) -> String {
        let mut out = String::new();
        let mut current: Option<usize> = None;
        for seg in &self.segments {
            let text = seg.text();
            if text.trim().is_empty() {
                continue;
            }
            if seg.speaker != current {
                if current.is_some() {
                    out.push('\n');
                }
                if let Some(spk) = seg.speaker {
                    out.push_str(
                        &self
                            .speakers
                            .get(spk)
                            .map(Speaker::display_name)
                            .unwrap_or_default(),
                    );
                    out.push_str(": ");
                }
                current = seg.speaker;
            } else {
                out.push(' ');
            }
            out.push_str(&text);
        }
        out
    }

    /// Absolute end of the last segment — a **position**, not a length.
    ///
    /// Used when the container's declared duration is missing or wrong (a real
    /// hazard with some M4A/MKV files). It is the end of the last thing anyone
    /// heard, which is only *also* the media's length when the media starts at
    /// zero — hence a separate method from [`inferred_length`](Self::inferred_length).
    #[must_use]
    pub fn inferred_media_end(&self) -> Time {
        self.segments.iter().map(|s| s.end).max().unwrap_or(Time::ZERO)
    }

    /// Absolute start of the first segment — a **position**.
    ///
    /// The origin the transcript's own times run from. For a whole recording
    /// this is zero; for a slice it is not, and a caller that needs the extent
    /// has to use this rather than assuming `Time::ZERO`.
    #[must_use]
    pub fn inferred_media_start(&self) -> Time {
        self.segments
            .iter()
            .map(|s| s.start)
            .min()
            .unwrap_or(Time::ZERO)
            .clamp_zero()
    }

    /// How much of the media this transcript covers — an **amount**.
    ///
    /// Derived from the first and last segment, so it is the *transcribed*
    /// extent, not the file's. Display code wants this; anything that needs a
    /// bound wants [`media_window`](Self::media_window) instead.
    #[must_use]
    pub fn inferred_length(&self) -> MediaLength {
        self.inferred_media_start().span_to(self.inferred_media_end())
    }

    /// The media window this transcript's times must lie inside, when it is
    /// known.
    ///
    /// Built from the declared [`duration`](Self::duration) **and** the
    /// transcript's own origin, which is the step every clamp before this type
    /// existed got wrong: it used the declared length as though it were an
    /// absolute end. A transcript that starts at +5 minutes with a declared
    /// length of 300 s gives `300_000..600_000` here, not `0..300_000`.
    ///
    /// `None` when no duration was ever declared, because guessing a window is
    /// how the original defect was introduced.
    #[must_use]
    pub fn media_window(&self) -> Option<MediaWindow> {
        self.duration.map(|length| {
            let start = self.inferred_media_start();
            MediaWindow::new(start, start.offset_by(length))
        })
    }

    /// Look up a segment for editing, by its stable id.
    pub fn segment_mut(&mut self, id: &str) -> Option<&mut Segment> {
        self.segments.iter_mut().find(|s| s.id == id)
    }
}

fn new_segment_id() -> String {
    // Short, collision-resistant within a job. A full UUID per segment is
    // wasteful for a 5000-segment transcript that gets re-serialised often.
    uuid::Uuid::new_v4().simple().to_string()[..16].to_string()
}

fn new_transcript_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: i64, end: i64, words: &[(&str, i64, i64)]) -> Segment {
        Segment::new(Time(start), Time(end)).with_words(
            words
                .iter()
                .map(|(t, s, e)| Word::new(*t, Time(*s), Time(*e)))
                .collect(),
        )
    }

    #[test]
    fn push_chunk_rebases_onto_media_timeline() {
        let mut t = Transcript::new(EngineStamp::asr("test", "1"));
        // A chunk starting at 60s produces segment-local times 0..2000.
        t.push_chunk(
            Time(60_000),
            vec![seg(
                0,
                2_000,
                &[("hello", 0, 1_000), ("world", 1_000, 2_000)],
            )],
        );
        let s = &t.segments[0];
        assert_eq!(s.start, Time(60_000));
        assert_eq!(s.end, Time(62_000));
        assert_eq!(s.words[0].start, Time(60_000));
        assert_eq!(s.text(), "hello world");
    }

    #[test]
    fn push_chunk_clamps_negative_starts_from_overlap() {
        // We keep overlap between chunks precisely so a word straddling the
        // boundary is not clipped, which means a segment can start fractionally
        // before the chunk. A chunk at 10 ms with a segment starting -40 ms
        // rebases to -30 ms, which must clamp to 0 rather than wrap.
        let mut t = Transcript::new(EngineStamp::asr("test", "1"));
        t.push_chunk(Time(10), vec![seg(-40, 900, &[("a", -40, 900)])]);
        assert_eq!(t.segments[0].start, Time(0));
        assert_eq!(t.segments[0].words[0].start, Time(0));
        // The rebased end is still positive and must be preserved.
        assert_eq!(t.segments[0].end, Time(910));
    }

    #[test]
    fn push_chunk_fixes_inverted_word_times() {
        let mut t = Transcript::new(EngineStamp::asr("test", "1"));
        t.push_chunk(Time(0), vec![seg(0, 500, &[("x", 400, 100)])]);
        assert_eq!(t.segments[0].words[0].start, Time(400));
        assert_eq!(t.segments[0].words[0].end, Time(400));
    }

    #[test]
    fn push_chunk_keeps_segments_sorted() {
        let mut t = Transcript::new(EngineStamp::asr("test", "1"));
        t.push_chunk(Time(0), vec![seg(0, 1_000, &[("a", 0, 1_000)])]);
        t.push_chunk(Time(2_000), vec![seg(0, 1_000, &[("b", 0, 1_000)])]);
        // An overlapping chunk that starts *before* the previous one must not
        // append out of order.
        t.push_chunk(Time(-500), vec![seg(0, 400, &[("c", 0, 400)])]);
        let starts: Vec<i64> = t.segments.iter().map(|s| s.start.as_millis()).collect();
        let mut sorted = starts.clone();
        sorted.sort_unstable();
        assert_eq!(starts, sorted, "segments must stay time-ordered");
    }

    #[test]
    fn push_chunk_within_clamps_the_last_frame_to_the_media() {
        // The measured case: a real 834.197 s recording whose last word ended
        // at 834.220 s, because the final STFT frame extends past the audio.
        let mut t = Transcript::new(EngineStamp::asr("test", "1"));
        t.push_chunk_within(
            Time(834_000),
            vec![seg(0, 500, &[("last", 0, 220)])],
            &MediaWindow::of_length(MediaLength::from_millis(834_197)),
        );
        assert_eq!(t.segments[0].end, Time(834_197));
        assert_eq!(t.segments[0].words[0].end, Time(834_197));
        // Nothing before the bound moved.
        assert_eq!(t.segments[0].start, Time(834_000));
        assert_eq!(t.segments[0].words[0].start, Time(834_000));
    }

    #[test]
    fn push_chunk_within_repairs_a_clamp_made_degenerate() {
        // A segment entirely past the media collapses to zero length at the
        // boundary rather than surviving outside it. The clamp runs first so
        // the repair cannot push the end back out.
        let mut t = Transcript::new(EngineStamp::asr("test", "1"));
        t.push_chunk_within(
            Time(0),
            vec![seg(11_000, 12_000, &[])],
            &MediaWindow::of_length(MediaLength::from_millis(10_000)),
        );
        assert_eq!(t.segments[0].start, Time(10_000));
        assert_eq!(t.segments[0].end, Time(10_000));
    }

    #[test]
    fn push_chunk_without_a_bound_is_unchanged() {
        // The unbounded path must behave exactly as before, or every existing
        // caller changes meaning. A segment past any plausible media survives.
        let mut t = Transcript::new(EngineStamp::asr("test", "1"));
        t.push_chunk(Time(0), vec![seg(0, 2_000_000, &[("x", 0, 2_000_000)])]);
        assert_eq!(t.segments[0].end, Time(2_000_000));
    }

    #[test]
    fn segment_at_lingers_through_gaps() {
        let mut t = Transcript::new(EngineStamp::asr("test", "1"));
        t.push_chunk(Time(0), vec![seg(0, 1_000, &[("a", 0, 1_000)])]);
        t.push_chunk(Time(5_000), vec![seg(0, 1_000, &[("b", 0, 1_000)])]);
        assert_eq!(t.segment_at(Time(500)).map(|s| s.text()), Some("a".into()));
        // In the silence between the two turns, keep showing the first rather
        // than blinking the highlight off.
        assert_eq!(
            t.segment_at(Time(3_000)).map(|s| s.text()),
            Some("a".into())
        );
        assert_eq!(
            t.segment_at(Time(5_500)).map(|s| s.text()),
            Some("b".into())
        );
    }

    #[test]
    fn segment_at_before_start_returns_none() {
        let t = Transcript::new(EngineStamp::asr("test", "1"));
        assert!(t.segment_at(Time(0)).is_none());
    }

    #[test]
    fn time_formats() {
        assert_eq!(Time(0).format_hms(), "00:00");
        assert_eq!(Time(61_000).format_hms(), "01:01");
        assert_eq!(Time(3_661_000).format_hms(), "01:01:01");
        assert_eq!(Time(3_661_500).format_hms_millis(), "01:01:01.500");
        assert_eq!(Time(-5).format_hms(), "00:00");
    }

    #[test]
    fn time_from_secs_rounds_rather_than_truncates() {
        // Repeatedly converting must not drift.
        let t = Time::from_secs_f64(1.9999);
        assert_eq!(t.as_millis(), 2_000);
        assert_eq!(Time::from_secs_f64(t.as_secs_f64()), t);
    }

    #[test]
    fn word_text_is_derived_not_stored() {
        let w = Word::new("acreage", Time(0), Time(100));
        assert_eq!(w.display_text(), "acreage");
        let w2 = Word {
            normalized: Some("acreage".into()),
            ..w
        };
        assert_eq!(w2.display_text(), "acreage");
    }

    #[test]
    fn speaker_rename_never_touches_text() {
        let mut t = Transcript::new(EngineStamp::asr("test", "1"));
        t.push_chunk(
            Time(0),
            vec![seg(0, 1_000, &[("haan", 0, 1_000)]).with_speaker(Some(0))],
        );
        t.add_speaker(Speaker::new(0));
        t.speaker_mut(0).name = Some("Krishna".into());
        assert_eq!(t.segments[0].text(), "haan");
        assert!(t.to_plain_text().contains("Krishna: haan"));
        assert!(!t.to_plain_text().contains("Speaker 1"));
    }

    #[test]
    fn unknown_speaker_index_is_materialised() {
        let mut t = Transcript::new(EngineStamp::asr("test", "1"));
        assert_eq!(t.speaker(3), None);
        t.speaker_mut(3);
        assert_eq!(t.speakers.len(), 1);
        assert_eq!(t.speakers[0].index, 3);
        // Indices are user-facing 1-based even though they are 0-based here.
        assert_eq!(t.speakers[0].display_name(), "Speaker 4");
        // Asking again must not duplicate the entry.
        t.speaker_mut(3);
        assert_eq!(t.speakers.len(), 1);
    }

    #[test]
    fn speaker_shares_sum_to_one() {
        let mut t = Transcript::new(EngineStamp::asr("test", "1"));
        t.push_chunk(
            Time(0),
            vec![
                seg(0, 1_000, &[("a", 0, 1_000)]).with_speaker(Some(0)),
                seg(1_000, 3_000, &[("b", 1_000, 3_000)]).with_speaker(Some(1)),
            ],
        );
        t.add_speaker(Speaker::new(0));
        t.add_speaker(Speaker::new(1));
        t.recompute_speaker_shares();
        let sum: f32 = t.speakers.iter().filter_map(|s| s.share).sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "shares should sum to 1, got {sum}"
        );
        // 1000 ms of 3000 ms total.
        assert!((t.speakers[0].share.unwrap() - 1.0 / 3.0).abs() < 1e-5);
        assert!((t.speakers[1].share.unwrap() - 2.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn plain_text_groups_consecutive_turns_by_speaker() {
        let mut t = Transcript::new(EngineStamp::asr("test", "1"));
        t.add_speaker(Speaker::new(0));
        t.add_speaker(Speaker::new(1));
        t.push_chunk(
            Time(0),
            vec![
                seg(0, 1_000, &[("hello", 0, 1_000)]).with_speaker(Some(0)),
                seg(1_000, 2_000, &[("again", 1_000, 2_000)]).with_speaker(Some(0)),
                seg(2_000, 3_000, &[("hi", 2_000, 3_000)]).with_speaker(Some(1)),
            ],
        );
        let text = t.to_plain_text();
        assert_eq!(text, "Speaker 1: hello again\nSpeaker 2: hi");
    }

    #[test]
    fn transcript_round_trips_through_json() {
        let mut t = Transcript::new(EngineStamp::asr("sravaani", "1.0"));
        t.duration = Some(MediaLength::from_millis(720_000));
        t.push_chunk(
            Time(0),
            vec![seg(0, 1_000, &[("नमस्ते", 0, 1_000)]).with_speaker(Some(0))],
        );
        t.add_speaker(Speaker::new(0));
        let json = serde_json::to_string(&t).unwrap();
        let back: Transcript = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn the_wire_format_of_duration_did_not_change() {
        // The newtype is `#[serde(transparent)]` over `i64`, so an exported
        // transcript is byte-identical to what the previous `Option<Time>`
        // produced. If this test ever fails, an export consumer breaks.
        let mut t = Transcript::new(EngineStamp::asr("m", "1"));
        t.duration = Some(MediaLength::from_millis(783_400));
        let json = serde_json::to_string(&t).unwrap();
        assert!(
            json.contains("\"duration\":783400"),
            "duration must still serialise as a bare integer: {json}"
        );
    }

    // ---------------------------------------------------------------------
    // The historical bug class: a length compared against an absolute time
    // ---------------------------------------------------------------------

    /// A11, verbatim. `vaani-segment` compared an **absolute** segment end
    /// against `Transcript::duration`, which is a **length**. A transcript
    /// whose times run `300_000..600_000` with a declared duration of
    /// `300_000` had every segment crushed to `300_000..300_001` and every word
    /// left outside its own segment.
    ///
    /// The regression is not "the numbers came out right" — it is that the
    /// mistake can no longer be *written*: `MediaLength` and `Time` do not
    /// compare, which the `compile_fail` doctest on `MediaLength` asserts. What
    /// this test pins is the corrected arithmetic, using the type-correct
    /// expression for the same clamp.
    #[test]
    fn a_transcript_that_does_not_start_at_zero_is_not_pinned_to_the_origin() {
        // The fixture, exactly as the conformance harness seeds it: five
        // minutes of origin, a 300 s declared length, absolute times.
        let mut t = Transcript::new(EngineStamp::asr("x", "0"));
        t.duration = Some(MediaLength::from_millis(300_000));
        let half = 150_000;
        t.push_chunk(
            Time(300_000),
            vec![
                Segment::new(Time(0), Time(half)).with_words(vec![
                    Word::new("alpha", Time(0), Time(half / 2)),
                    Word::new("bravo", Time(half / 2), Time(half)),
                ]),
                Segment::new(Time(half), Time(300_000))
                    .with_words(vec![Word::new("charlie", Time(half), Time(300_000))]),
            ],
        );

        // The type-correct reading of the same fact: a declared length plus the
        // transcript's own origin. The old code had no way to write this.
        let window = t.media_window().expect("a duration was declared");
        assert_eq!(window.start(), Time(300_000));
        assert_eq!(window.end(), Time(600_000));
        assert_eq!(window.length(), t.duration.unwrap());

        for seg in &t.segments {
            assert!(window.contains(seg.start), "{:?} is before the window", seg.start);
            assert!(seg.end <= window.end(), "{:?} is past the window", seg.end);
            for w in &seg.words {
                assert!(
                    w.start >= seg.start && w.end <= seg.end,
                    "a word fell outside its own segment"
                );
            }
        }
        // And the value the buggy code used as a bound, `0..300_000`, contains
        // none of it: this is the whole defect in one assertion.
        let wrong = MediaWindow::of_length(t.duration.unwrap());
        assert_eq!(
            t.segments
                .iter()
                .filter(|s| wrong.contains(s.start))
                .count(),
            0,
            "a length used as a window happens to contain this transcript, which means the \
             test is not exercising the non-zero origin"
        );
    }

    /// A non-zero media `start_time` with a correct absolute end.
    ///
    /// The pipeline case: decode gives `start_time == 0`, so the length and the
    /// end coincide and every bug in this class is invisible. A slice, a resumed
    /// job or a second source makes them differ, and the two must not be
    /// conflated.
    #[test]
    fn a_nonzero_media_start_time_clamps_against_the_absolute_end() {
        // 2 s of mono 16 kHz audio that begins five minutes into the recording.
        let audio = crate::media::AudioBuffer::mono(vec![0.0; 32_000], 16_000, Time(300_000));
        assert_eq!(audio.length(), MediaLength::from_millis(2_000));
        let window = MediaWindow::of_buffer(&audio);
        assert_eq!(window.start(), Time(300_000));
        assert_eq!(window.end(), Time(302_000));

        // A chunk taken from the tail of that buffer, whose last frame runs 20
        // ms past the audio — the measured ASR case.
        let mut t = Transcript::new(EngineStamp::asr("m", "1"));
        t.duration = Some(window.length());
        t.push_chunk_within(
            Time(300_000),
            vec![seg(0, 2_020, &[("tail", 0, 2_020)])],
            &window,
        );
        assert_eq!(t.segments[0].end, Time(302_000));
        assert!(t.segments[0].end <= window.end());

        // The old call passed `audio.duration()` — the *length* — where the
        // clamp needed an *end*. On this buffer that pins the whole transcript
        // to 2_000 ms, which is before the recording even starts.
        let old = MediaWindow::of_length(audio.length());
        assert_eq!(old.end(), Time(2_000));
        assert!(t.segments[0].start > old.end());
    }

    /// A chunk whose local times plus an offset land inside the media.
    ///
    /// This is what every real ASR chunk does, and it is the shape the
    /// conformance harness runs: local `0..N` plus a non-zero offset, checked
    /// against the window built from the same buffer.
    #[test]
    fn chunk_local_times_plus_an_offset_land_inside_the_media() {
        let audio = crate::media::AudioBuffer::mono(vec![0.0; 16_000 * 90], 16_000, Time(300_000));
        let window = MediaWindow::of_buffer(&audio);
        let mut t = Transcript::new(EngineStamp::asr("m", "1"));
        t.duration = Some(window.length());

        // Three 30-second chunks, the last one overrunning by one frame.
        for (i, offset) in [0i64, 30_000, 60_000].into_iter().enumerate() {
            let local = if i == 2 { 30_020 } else { 30_000 };
            t.push_chunk_within(
                window.start().offset_by(MediaLength::from_millis(offset)),
                vec![seg(
                    0,
                    local,
                    &[("w", 0, local)],
                )],
                &window,
            );
        }
        for s in &t.segments {
            assert!(
                window.contains(s.start),
                "chunk at {} ms is outside {}..{}",
                s.start,
                window.start(),
                window.end()
            );
            assert!(s.end <= window.end());
            for w in &s.words {
                assert!(w.start >= s.start && w.end <= s.end);
            }
        }
        // Only the last one was clipped, and only by the one frame that ran
        // past the audio.
        assert_eq!(t.segments[2].end, window.end());
        assert_eq!(t.segments[0].end, Time(330_000));
    }

    /// A2's known failure: `Chunker::plan(&speech, audio.duration())` — an
    /// absolute speech end clamped against a relative total.
    ///
    /// The chunker lives in `vaani-vad`, so the defect is reproduced here as
    /// the contract-level fact it rests on: **for a buffer with a non-zero
    /// origin, the buffer's length and the buffer's end are different numbers,
    /// and only the end bounds a time.** The chunker's own test pins the same
    /// fact; this one pins the number.
    #[test]
    fn a_length_and_an_absolute_end_differ_when_the_media_starts_late() {
        let audio = crate::media::AudioBuffer::mono(vec![0.0; 16_000 * 90], 16_000, Time(300_000));
        let length = audio.length();
        let end = MediaWindow::of_buffer(&audio).end();
        assert_eq!(length.as_millis(), 90_000);
        assert_eq!(end, Time(390_000));
        assert_ne!(end, Time::from_millis(length.as_millis()));

        // A speech span anywhere in the buffer is inside the window and outside
        // the length read as a position. This is the exact comparison the
        // conformance harness reports as times before the origin.
        let speech = MediaWindow::new(Time(300_000), Time(389_600));
        let window = MediaWindow::of_buffer(&audio);
        assert!(window.contains(speech.start()));
        assert!(!MediaWindow::of_length(length).contains(speech.start()));
        assert_eq!(speech.length(), Time(300_000).span_to(Time(389_600)));
    }

    /// The unambiguous API makes the old mistake impossible to express.
    ///
    /// The `compile_fail` doctests on [`MediaLength`] and [`MediaWindow`] are
    /// the machine-checked half: they run in the ordinary `cargo test
    /// --workspace` gate and fail the build if the two types ever become
    /// comparable. This test is the half that can be a runtime assert — the
    /// conversions that *are* available, and the ones that are not reachable
    /// without an explicit act.
    #[test]
    fn the_type_system_separates_positions_from_amounts() {
        // A length is produced from two positions, or from a number that is
        // already known to be an amount. Never from one position.
        assert_eq!(Time(1_000).span_to(Time(3_500)).as_millis(), 2_500);
        assert_eq!(MediaLength::from_millis(-7), MediaLength::ZERO);

        // A position is advanced by an amount. Never by a position.
        assert_eq!(
            Time(300_000).offset_by(MediaLength::from_millis(90_000)),
            Time(390_000)
        );

        // A window is two positions, and its own length is derived rather than
        // stored — so the two can never disagree.
        let w = MediaWindow::new(Time(300_000), Time(390_000));
        assert_eq!(w.length(), MediaLength::from_millis(90_000));
        assert_eq!(w.start().span_to(w.end()), w.length());

        // A segment's own length is an amount, and cannot be its own start.
        let s = Segment::new(Time(300_000), Time(330_000));
        assert_eq!(s.duration(), MediaLength::from_millis(30_000));
        assert_eq!(s.start.span_to(s.end), s.duration());
    }

    #[test]
    fn a_window_clamps_both_ends_and_repairs_inversion() {
        let w = MediaWindow::new(Time(300_000), Time(390_000));
        assert_eq!(w.clamp(Time(0)), Time(300_000));
        assert_eq!(w.clamp(Time(999_999)), Time(390_000));
        assert_eq!(w.clamp(Time(350_000)), Time(350_000));
        assert_eq!(w.clamp_span(Time(0), Time(999_999)), (Time(300_000), Time(390_000)));
        // An inverted pair collapses to the origin of the window rather than
        // inverting, which would put an end before its start.
        assert_eq!(w.clamp_span(Time(999_999), Time(0)), (Time(390_000), Time(390_000)));
        // An inverted window is normalised, so `clamp` is total.
        let bad = MediaWindow::new(Time(390_000), Time(300_000));
        assert_eq!(bad.length(), MediaLength::ZERO);
        assert_eq!(bad.clamp(Time(350_000)), Time(390_000));
    }

    #[test]
    fn inferred_length_and_media_window_agree_with_each_other() {
        let mut t = Transcript::new(EngineStamp::asr("m", "1"));
        t.push_chunk(
            Time(300_000),
            vec![seg(0, 1_000, &[("a", 0, 1_000)]), seg(1_000, 2_000, &[("b", 1_000, 2_000)])],
        );
        assert_eq!(t.inferred_media_start(), Time(300_000));
        assert_eq!(t.inferred_media_end(), Time(302_000));
        assert_eq!(t.inferred_length(), MediaLength::from_millis(2_000));
        // No declared duration means no window: guessing one is what the
        // original defect was.
        assert!(t.media_window().is_none());
        t.duration = Some(MediaLength::from_millis(10_000));
        assert_eq!(t.media_window().unwrap().end(), Time(310_000));
    }
}

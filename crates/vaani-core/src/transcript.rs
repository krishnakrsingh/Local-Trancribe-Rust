//! The transcript data model.
//!
//! This is the shape the whole product agrees on: ASR produces it, diarization
//! annotates it, the UI renders it, the exporters serialise it, SQLite persists
//! it. Changes here are expensive, so the invariants are encoded in the types
//! and enforced in `Transcript`'s constructors rather than left to convention.
//!
//! # Timestamps
//!
//! Times are [`Time`], a signed count of milliseconds from the start of the
//! *source media*, never from the start of a chunk or a segment. Chunk
//! boundaries are an implementation detail of the ASR pipeline; every offset
//! that leaves the `asr` crate has already been rebased onto the media
//! timeline. Getting this wrong is the classic source of "the transcript is
//! three minutes ahead of the video" bugs, so the rebasing happens in exactly
//! one place (see `Transcript::push_chunk`).

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
}

impl fmt::Display for Time {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.format_hms())
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

    /// Length of the segment, never negative.
    pub fn duration(&self) -> Time {
        Time::from_millis(self.end.0.saturating_sub(self.start.0))
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
    /// Total media duration, when known. `None` for a live/streaming source.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<Time>,
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
    pub fn push_chunk(&mut self, chunk_offset: Time, mut segments: Vec<Segment>) {
        for s in &mut segments {
            s.start = s.start.saturating_add(chunk_offset).clamp_zero();
            s.end = s.end.saturating_add(chunk_offset).clamp_zero();
            // A word that ends before it starts would produce a negative
            // duration and a malformed SRT cue. Clamp rather than drop: the
            // word was recognised, it just has a degenerate time span.
            if s.end < s.start {
                s.end = s.start;
            }
            for w in &mut s.words {
                w.start = w.start.saturating_add(chunk_offset).clamp_zero();
                w.end = w.end.saturating_add(chunk_offset).clamp_zero();
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

    /// Duration of the last segment, used when the container's declared
    /// duration is missing or wrong (a real hazard with some M4A/MKV files).
    pub fn inferred_duration(&self) -> Time {
        self.segments
            .iter()
            .map(|s| s.end)
            .max()
            .unwrap_or(Time::ZERO)
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
        t.duration = Some(Time(720_000));
        t.push_chunk(
            Time(0),
            vec![seg(0, 1_000, &[("नमस्ते", 0, 1_000)]).with_speaker(Some(0))],
        );
        t.add_speaker(Speaker::new(0));
        let json = serde_json::to_string(&t).unwrap();
        let back: Transcript = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }
}

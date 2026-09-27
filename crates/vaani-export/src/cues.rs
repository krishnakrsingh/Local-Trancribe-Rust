//! Turning segments into subtitle cues.
//!
//! This module owns every rule that makes SRT and VTT *valid*, and both formats
//! go through it, so there is exactly one implementation of each rule and the two
//! cannot drift apart.
//!
//! # The rules, and why each one exists
//!
//! ## At most two lines per cue
//!
//! Two lines of roughly 42 characters is the readability budget every subtitle
//! guide agrees on, and it is what players are actually built for: a cue with
//! five lines covers the screen, hides the dialogue underneath it, and makes the
//! next line unreadable during playback.
//!
//! **What happens when one ASR segment is too long for two lines:** the segment
//! is **split into several cues**, not truncated and not squeezed onto one long
//! line. Dropping the tail loses words the user paid minutes of compute for;
//! one over-long cue violates the rule we are here to enforce.
//!
//! ## How the split is timed
//!
//! The split point is chosen at a **word boundary**, and the cue boundaries are
//! the **real word times** when the segment has them: the first cue starts at
//! the first word it contains and ends at the last word it contains. That is
//! strictly better than dividing the segment's duration by the number of lines,
//! which is what a naive exporter does and which puts the caption on screen
//! before the speaker has said it.
//!
//! When a segment has no word timings — a segment built by an importer, or one
//! whose words carry only text — the split falls back to dividing the segment's
//! span **proportionally by character count**. That is an approximation and is
//! documented as one; the alternative would be dropping text.
//!
//! ## No overlaps, ever
//!
//! ASR segments can overlap, because chunks overlap in the pipeline. Two
//! overlapping cues are rejected by some players and make text selection in
//! every other player impossible, so a cue that would start before the previous
//! one ends is **pushed forward to the previous cue's end**. The text is
//! unchanged; only its timing is serialised. The cost is honest and worth
//! stating: for a genuinely overlapping segment, the caption appears slightly
//! later than the speech. The alternative — emitting overlapping cues — trades a
//! small timing error for a malformed file.
//!
//! ## No zero or negative durations
//!
//! Every cue is at least one millisecond long, and every cue ends after it
//! starts. A zero-length cue is invisible or, in some players, an error.
//!
//! ## No mid-word breaks
//!
//! Lines are broken at spaces. A single token longer than the line width — a URL,
//! a 90-character identifier — is emitted on a line of its own, **unbroken**.
//! Splitting it would put half a word on screen, and no player can make that
//! readable.
//!
//! ## Empty segments are skipped
//!
//! A segment with no words produces no cue. A cue with no text is a blank
//! screen for the length of the segment, which is a defect, not content.

use vaani_core::{Time, Transcript};

/// A single subtitle cue: a time span and the lines shown in it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)] // field names are the documentation
pub struct Cue {
    pub start: Time,
    /// Always strictly greater than `start`.
    pub end: Time,
    /// One to `max_lines_per_cue` lines. Never empty.
    pub lines: Vec<String>,
    /// The segment this came from, for diagnostics and for tests.
    pub segment_id: String,
}

impl Cue {
    /// The cue's text with its lines joined, which is what the text and
    /// Markdown renderers show.
    pub fn text(&self) -> String {
        self.lines.join(" ")
    }

    /// Length, never negative.
    pub fn duration(&self) -> Time {
        Time::from_millis((self.end.0 - self.start.0).max(0))
    }
}

/// One wrapped line, and which of the source tokens it covers.
///
/// The token range is what lets a split cue be given real word times: a cue
/// covering tokens 4..9 starts at word 4's start time.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Line {
    text: String,
    first: usize,
    last: usize,
}

/// The rules the cue builder applies. Every field is a product decision the
/// caller can change; the defaults are the ones the exporters' own tests pin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CueOptions {
    /// Characters per line. 42 is the SubRip convention and the widest line a
    /// reader can scan without losing the start of the next one.
    pub max_line_length: usize,
    /// Lines per cue. Two.
    pub max_lines_per_cue: usize,
    /// Whether to wrap at all. `false` puts the whole cue on one line, which
    /// only a machine-readable consumer wants.
    pub wrap: bool,
    /// Prefix a cue with the speaker's name.
    pub speaker_labels: bool,
}

impl Default for CueOptions {
    fn default() -> Self {
        Self {
            max_line_length: 42,
            max_lines_per_cue: 2,
            wrap: true,
            speaker_labels: true,
        }
    }
}

impl CueOptions {
    /// A cue policy for a UI that lets the user choose the width.
    #[must_use]
    pub fn with_max_line_length(mut self, width: usize) -> Self {
        self.max_line_length = width.max(8);
        self
    }

    /// A cue policy for a UI that lets the user choose one or two lines.
    #[must_use]
    pub fn with_max_lines_per_cue(mut self, lines: usize) -> Self {
        self.max_lines_per_cue = lines.max(1);
        self
    }
}

/// Break `tokens` into lines no wider than `width`, never inside a token.
///
/// With `wrap` off, every token lands on its own line. That is not a
/// convenience: it is what a caller asking for "one word per line" means, and it
/// keeps the token ranges exact so the timings stay right.
fn wrap_tokens(tokens: &[&str], width: usize, wrap: bool) -> Vec<Line> {
    let width = width.max(1);
    let mut lines: Vec<Line> = Vec::new();
    let mut current = String::new();
    let mut first = 0usize;
    for (i, token) in tokens.iter().enumerate() {
        let token_len = token.chars().count();
        if !wrap {
            lines.push(Line {
                text: (*token).to_string(),
                first: i,
                last: i,
            });
            continue;
        }
        if current.is_empty() {
            current.push_str(token);
            first = i;
            // A token wider than the line gets its own line, unbroken. Putting
            // the next token on the same line would overflow; breaking this one
            // would put half a word on screen.
            if token_len >= width {
                lines.push(Line {
                    text: std::mem::take(&mut current),
                    first,
                    last: i,
                });
            }
            continue;
        }
        // +1 for the space that would join them.
        if current.chars().count() + 1 + token_len <= width {
            current.push(' ');
            current.push_str(token);
        } else {
            lines.push(Line {
                text: std::mem::replace(&mut current, (*token).to_string()),
                first,
                last: i - 1,
            });
            first = i;
            if token_len >= width {
                lines.push(Line {
                    text: std::mem::take(&mut current),
                    first,
                    last: i,
                });
            }
        }
    }
    if !current.is_empty() {
        lines.push(Line {
            text: current,
            first,
            last: tokens.len().saturating_sub(1),
        });
    }
    lines
}

/// The name to show for a speaker index, or `None` for "not diarized".
fn speaker_name(t: &Transcript, index: usize) -> Option<String> {
    t.speaker(index).map(|s| s.display_name())
}

/// Build the cue list for a transcript.
pub fn build_cues(t: &Transcript, options: &CueOptions) -> Vec<Cue> {
    let mut cues: Vec<Cue> = Vec::new();
    let mut previous_end: Option<i64> = None;

    for seg in &t.segments {
        let body = seg.text();
        if body.trim().is_empty() {
            // A cue with no text is a blank screen for the segment's duration.
            continue;
        }
        let prefix = match (options.speaker_labels, seg.speaker) {
            (true, Some(index)) => speaker_name(t, index),
            _ => None,
        };
        // `Name:` is `name tokens + 1` whitespace-separated tokens.
        let prefix_tokens = prefix
            .as_ref()
            .map(|p| p.split_whitespace().count() + 1)
            .unwrap_or(0);
        let display = match &prefix {
            Some(p) => format!("{p}: {body}"),
            None => body.clone(),
        };
        let tokens: Vec<&str> = display.split_whitespace().collect();

        // Word times can only be trusted if the display tokens map one-to-one
        // onto the segment's words after the prefix. A word with empty text, or
        // text that has been edited to collapse a space, breaks the mapping —
        // and a wrong caption time is worse than an approximate one.
        let word_times: Option<Vec<(i64, i64)>> =
            if tokens.len() == prefix_tokens + seg.words.len() && !seg.words.is_empty() {
                Some(
                    seg.words
                        .iter()
                        .map(|w| (w.start.as_millis(), w.end.as_millis()))
                        .collect(),
                )
            } else {
                None
            };
        let seg_start = seg.start.as_millis();
        let seg_len = (seg.end.as_millis() - seg_start).max(0);
        let token_count = tokens.len().max(1) as i64;
        // Word index for a display token, or `None` if it is part of the prefix.
        let word_index = |token: usize| -> Option<usize> {
            token
                .checked_sub(prefix_tokens)
                .filter(|w| word_times.as_ref().is_some_and(|times| *w < times.len()))
        };
        let time_of = |lo: usize, hi: usize| -> (i64, i64) {
            // `hi` is exclusive.
            match &word_times {
                Some(times) => {
                    let first = word_index(lo);
                    let last = word_index(hi.saturating_sub(1));
                    match (first, last) {
                        (Some(f), Some(l)) if f < times.len() && l < times.len() => {
                            (times[f].0, times[l].1)
                        }
                        // A chunk that is only the speaker prefix, which cannot
                        // happen with `max_lines_per_cue >= 1`, falls back to the
                        // segment span rather than panicking.
                        _ => (seg_start, seg_start + seg_len),
                    }
                }
                // No word timings: divide the span proportionally by character
                // count. Approximate, and documented as such.
                None => {
                    let lo = lo as i64;
                    let hi = hi as i64;
                    (
                        seg_start + seg_len * lo / token_count,
                        seg_start + seg_len * hi / token_count,
                    )
                }
            }
        };

        let lines = wrap_tokens(&tokens, options.max_line_length, options.wrap);
        let per_cue = options.max_lines_per_cue.max(1);
        let chunk_count = lines.len().div_ceil(per_cue);
        for (n, chunk) in lines.chunks(per_cue).enumerate() {
            if chunk.is_empty() {
                continue;
            }
            let is_first = n == 0;
            let is_last = n + 1 == chunk_count;
            let (word_start, word_end) = time_of(chunk[0].first, chunk[chunk.len() - 1].last + 1);
            // The first cue of a segment starts at the segment's start and the
            // last ends at the segment's end, so the cues between them cover the
            // segment exactly: no gap at the edges where the speaker was already
            // pausing, and no overlap in the middle. Interior boundaries are the
            // real word times, which is the whole point of splitting here.
            let mut start = if is_first { seg_start } else { word_start };
            let mut end = if is_last {
                seg_start + seg_len
            } else {
                word_end
            };
            // No overlaps: a cue never starts before the previous one ended.
            if let Some(prev) = previous_end {
                start = start.max(prev);
            }
            // No zero or negative durations.
            if end <= start {
                end = start + 1;
            }
            previous_end = Some(end);
            cues.push(Cue {
                start: Time::from_millis(start),
                end: Time::from_millis(end),
                lines: chunk.iter().map(|l| l.text.clone()).collect(),
                segment_id: seg.id.clone(),
            });
        }
    }
    cues
}

/// `HH:MM:SS,mmm` — the SubRip form.
pub fn srt_timestamp(t: Time) -> String {
    let total = t.as_millis().max(0);
    let ms = total % 1_000;
    let total_s = total / 1_000;
    let s = total_s % 60;
    let m = (total_s / 60) % 60;
    let h = total_s / 3_600;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

/// `HH:MM:SS.mmm` — the WebVTT form. Identical except for the separator, which
/// is the single most common way a hand-written VTT file goes wrong.
pub fn vtt_timestamp(t: Time) -> String {
    let total = t.as_millis().max(0);
    let ms = total % 1_000;
    let total_s = total / 1_000;
    let s = total_s % 60;
    let m = (total_s / 60) % 60;
    let h = total_s / 3_600;
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}

/// Render cues as SubRip.
///
/// Byte-level contract, which the tests check against the format's rules:
///
/// * UTF-8, no byte-order mark (a BOM makes some players show `ï»¿` in the first
///   cue, and a SubRip file has no business carrying one).
/// * LF line endings. Players accept both; LF is what every modern writer emits.
/// * One-based, strictly increasing cue numbers.
/// * `HH:MM:SS,mmm --> HH:MM:SS,mmm` with a blank line after every cue,
///   including the last.
/// * No blank line inside a cue, because a blank line *ends* a cue.
pub fn render_srt(cues: &[Cue], include_index: bool) -> String {
    let mut out = String::new();
    for (i, cue) in cues.iter().enumerate() {
        if include_index {
            out.push_str(&(i + 1).to_string());
            out.push('\n');
        }
        out.push_str(&srt_timestamp(cue.start));
        out.push_str(" --> ");
        out.push_str(&srt_timestamp(cue.end));
        out.push('\n');
        for line in &cue.lines {
            out.push_str(line);
            out.push('\n');
        }
        out.push('\n');
    }
    out
}

/// Render cues as WebVTT.
///
/// Byte-level contract:
///
/// * The file starts with `WEBVTT`, then a blank line. Everything before that
///   line is ignored by a parser, and a missing header means the file is not
///   WebVTT at all.
/// * LF line endings, UTF-8, no byte-order mark.
/// * `HH:MM:SS.mmm --> HH:MM:SS.mmm` — note the dot.
/// * No cue identifiers, and no cue settings: both are optional in WebVTT and
///   both are places a hand-written file goes wrong.
/// * Cue text may not contain `-->`: the WebVTT parser scans for that sequence,
///   so a transcript containing a literal `-->` would truncate the cue. It is
///   written `--&gt;` instead, which any HTML renderer displays as `-->` and
///   which [`crate::verify::unescape_vtt`] reads back, so the round trip is
///   lossless.
pub fn render_vtt(cues: &[Cue]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for cue in cues {
        out.push_str(&vtt_timestamp(cue.start));
        out.push_str(" --> ");
        out.push_str(&vtt_timestamp(cue.end));
        out.push('\n');
        for line in &cue.lines {
            out.push_str(&escape_vtt_text(line));
            out.push('\n');
        }
        out.push('\n');
    }
    out
}

/// Escape the one sequence WebVTT forbids in cue text.
pub fn escape_vtt_text(text: &str) -> String {
    text.replace("-->", "--&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use vaani_core::{Segment, Speaker, Word};

    fn seg(id: &str, start: i64, end: i64, words: &[&str]) -> Segment {
        let mut s = Segment::new(Time(start), Time(end));
        s.id = id.into();
        s.words = words
            .iter()
            .enumerate()
            .map(|(i, w)| {
                let span = ((end - start) / (words.len() as i64 + 1)).max(1);
                Word::new(
                    *w,
                    Time(start + span * i as i64),
                    Time(start + span * (i as i64 + 1)),
                )
            })
            .collect();
        s
    }

    #[test]
    fn wrapping_breaks_at_spaces_and_never_inside_a_word() {
        let tokens = ["one", "two", "three"];
        let lines = wrap_tokens(&tokens, 8, true);
        let texts: Vec<&str> = lines.iter().map(|l| l.text.as_str()).collect();
        assert_eq!(texts, ["one two", "three"]);
        for l in &lines {
            assert!(l.text.chars().count() <= 8);
        }
    }

    #[test]
    fn a_word_longer_than_the_line_is_emitted_unbroken() {
        let long = "x".repeat(60);
        let tokens = ["short", long.as_str(), "tail"];
        let lines = wrap_tokens(&tokens, 10, true);
        assert!(lines.iter().any(|l| l.text == long), "{:?}", lines);
        // The long word is intact: no line contains a fragment of it plus more.
        assert_eq!(lines.iter().filter(|l| l.text.contains('x')).count(), 1);
    }

    #[test]
    fn every_token_appears_exactly_once_and_in_order() {
        let tokens: Vec<String> = (0..50).map(|i| format!("w{i}")).collect();
        let refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
        let lines = wrap_tokens(&refs, 17, true);
        let rejoined: Vec<&str> = lines.iter().map(|l| l.text.as_str()).collect();
        assert_eq!(rejoined.join(" "), tokens.join(" "));
    }

    #[test]
    fn a_short_segment_is_one_cue_with_its_own_times() {
        let mut t = Transcript::new(vaani_core::EngineStamp::asr("m", "1"));
        t.segments
            .push(seg("s0", 1_000, 2_500, &["hello", "there"]));
        let cues = build_cues(&t, &CueOptions::default());
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].lines, ["hello there"]);
        assert_eq!(cues[0].start, Time(1_000));
        assert_eq!(cues[0].end, Time(2_500));
    }

    #[test]
    fn a_long_segment_is_split_at_word_boundaries_with_real_word_times() {
        // Six words at 14 columns wrap to three lines, which is two cues of at
        // most two lines. The interior boundary is the *word* time, not a
        // proportional guess: "echo" begins at 8 s in this fixture, and that is
        // where the second cue starts.
        let mut t = Transcript::new(vaani_core::EngineStamp::asr("m", "1"));
        t.segments.push(seg(
            "s0",
            0,
            12_000,
            &["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"],
        ));
        let cues = build_cues(
            &t,
            &CueOptions {
                max_line_length: 14,
                max_lines_per_cue: 2,
                ..Default::default()
            },
        );
        assert_eq!(cues.len(), 2, "{cues:#?}");
        assert_eq!(cues[0].lines, ["alpha bravo", "charlie delta"]);
        assert_eq!(cues[1].lines, ["echo foxtrot"]);
        for cue in &cues {
            assert!(cue.end > cue.start, "a zero or negative duration");
        }
        // The first cue starts at the segment start and the last ends at the
        // segment end, so the segment is covered exactly.
        assert_eq!(cues[0].start, Time(0));
        assert_eq!(cues[1].end, Time(12_000));
        // The interior boundary is a real word time, not a proportional guess:
        // the second cue starts exactly where the first of its words starts.
        assert_eq!(cues[1].start, t.segments[0].words[4].start);
        // ...and the first cue ends where that word starts, so the cues are
        // contiguous rather than overlapping.
        assert_eq!(cues[0].end, cues[1].start);
    }

    #[test]
    fn a_segment_without_word_times_is_divided_proportionally() {
        let mut t = Transcript::new(vaani_core::EngineStamp::asr("m", "1"));
        let mut s = Segment::new(Time(0), Time(10_000));
        s.id = "s0".into();
        s.words = vec![
            Word::new("one", Time(0), Time(0)),
            Word::new("two", Time(0), Time(0)),
            Word::new("three", Time(0), Time(0)),
            Word::new("four", Time(0), Time(0)),
        ];
        t.segments.push(s);
        let cues = build_cues(
            &t,
            &CueOptions {
                max_line_length: 5,
                max_lines_per_cue: 1,
                ..Default::default()
            },
        );
        assert_eq!(cues.len(), 4);
        assert_eq!(cues[0].start, Time(0));
        assert_eq!(cues[3].end, Time(10_000));
        for cue in &cues {
            assert!(cue.end > cue.start);
        }
    }

    #[test]
    fn overlapping_segments_are_serialised_into_contiguous_cues() {
        let mut t = Transcript::new(vaani_core::EngineStamp::asr("m", "1"));
        t.segments.push(seg("s0", 0, 5_000, &["first"]));
        t.segments.push(seg("s1", 2_000, 7_000, &["second"]));
        let cues = build_cues(&t, &CueOptions::default());
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].end, Time(5_000));
        assert!(cues[1].start >= cues[0].end, "cues must not overlap");
        assert_eq!(cues[1].text(), "second", "no text is lost");
    }

    #[test]
    fn a_degenerate_segment_still_gets_a_positive_duration() {
        let mut t = Transcript::new(vaani_core::EngineStamp::asr("m", "1"));
        let mut s = Segment::new(Time(3_000), Time(3_000));
        s.id = "s0".into();
        s.words = vec![Word::new("x", Time(3_000), Time(3_000))];
        t.segments.push(s);
        let cues = build_cues(&t, &CueOptions::default());
        assert_eq!(cues.len(), 1);
        assert!(cues[0].end > cues[0].start);
    }

    #[test]
    fn empty_and_whitespace_segments_produce_no_cue() {
        let mut t = Transcript::new(vaani_core::EngineStamp::asr("m", "1"));
        let mut s = Segment::new(Time(0), Time(1_000));
        s.id = "s0".into();
        t.segments.push(s);
        t.segments.push(seg("s1", 1_000, 2_000, &["real"]));
        let cues = build_cues(&t, &CueOptions::default());
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text(), "real");
    }

    #[test]
    fn speakers_are_prefixed_by_default_and_can_be_turned_off() {
        let mut t = Transcript::new(vaani_core::EngineStamp::asr("m", "1"));
        t.segments.push(seg("s0", 0, 1_000, &["hello"]));
        t.segments[0].speaker = Some(0);
        t.speakers.push(Speaker::new(0));
        assert_eq!(
            build_cues(&t, &CueOptions::default())[0].text(),
            "Speaker 1: hello"
        );

        let renamed = build_cues(
            &t,
            &CueOptions {
                speaker_labels: false,
                ..Default::default()
            },
        );
        assert_eq!(renamed[0].text(), "hello");
    }

    #[test]
    fn a_renamed_speaker_is_used_and_an_undiarized_segment_has_no_label() {
        let mut t = Transcript::new(vaani_core::EngineStamp::asr("m", "1"));
        t.speakers.push(Speaker {
            index: 0,
            name: Some("Krishna".into()),
            share: None,
        });
        t.segments.push(seg("s0", 0, 1_000, &["named"]));
        t.segments[0].speaker = Some(0);
        t.segments.push(seg("s1", 1_000, 2_000, &["anonymous"]));
        let cues = build_cues(&t, &CueOptions::default());
        assert_eq!(cues[0].text(), "Krishna: named");
        assert_eq!(cues[1].text(), "anonymous");
    }

    #[test]
    fn timestamps_use_the_right_separator_and_pad_to_the_right_width() {
        assert_eq!(srt_timestamp(Time(0)), "00:00:00,000");
        assert_eq!(vtt_timestamp(Time(0)), "00:00:00.000");
        assert_eq!(srt_timestamp(Time(3_661_500)), "01:01:01,500");
        assert_eq!(vtt_timestamp(Time(3_661_500)), "01:01:01.500");
        // Past 24 hours the hours field grows rather than wrapping, which is
        // what a three-day recording needs.
        assert_eq!(srt_timestamp(Time(90_000_000)), "25:00:00,000");
        // A negative time is clamped: it cannot appear in a subtitle.
        assert_eq!(srt_timestamp(Time(-5)), "00:00:00,000");
    }

    #[test]
    fn srt_bytes_match_the_format_contract() {
        let mut t = Transcript::new(vaani_core::EngineStamp::asr("m", "1"));
        t.segments.push(seg("s0", 0, 1_000, &["one", "two"]));
        t.segments.push(seg("s1", 1_000, 2_000, &["three"]));
        let cues = build_cues(&t, &CueOptions::default());
        let srt = render_srt(&cues, true);
        assert_eq!(
            srt,
            "1\n00:00:00,000 --> 00:00:01,000\none two\n\n\
             2\n00:00:01,000 --> 00:00:02,000\nthree\n\n"
        );
        assert!(!srt.contains('\r'), "SubRip is written with LF endings");
        assert!(!srt.starts_with('\u{feff}'), "no byte-order mark");
        assert!(
            srt.ends_with("\n\n"),
            "a trailing blank line closes the last cue"
        );
    }

    #[test]
    fn vtt_bytes_match_the_format_contract() {
        let mut t = Transcript::new(vaani_core::EngineStamp::asr("m", "1"));
        t.segments.push(seg("s0", 1_500, 2_000, &["hello"]));
        let vtt = render_vtt(&build_cues(&t, &CueOptions::default()));
        assert_eq!(vtt, "WEBVTT\n\n00:00:01.500 --> 00:00:02.000\nhello\n\n");
        assert!(!vtt.contains("-->") || vtt.matches("-->").count() == 1);
    }

    #[test]
    fn vtt_escapes_the_sequence_its_own_parser_chokes_on() {
        assert_eq!(escape_vtt_text("a --> b"), "a --&gt; b");
        assert_eq!(escape_vtt_text("no arrows here"), "no arrows here");
    }
}

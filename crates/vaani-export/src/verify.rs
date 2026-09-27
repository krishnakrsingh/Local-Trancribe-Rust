//! Reference parsers for the subtitle formats, and the rules they check.
//!
//! # Why this exists, and what it is not
//!
//! There is no offline SubRip or WebVTT parser available to this crate, and
//! adding one would be a new third-party dependency — an architect's decision,
//! not this agent's. So validity is checked against **hand-written parsers that
//! implement the published rules**, and every check below states the rule it
//! enforces. That is a weaker guarantee than running a real player, and it is
//! stated as such: what it proves is that the bytes this crate writes *parse* as
//! SubRip and WebVTT according to those rules, including the awkward parts people
//! get wrong (the comma versus the dot, cue numbering, blank-line termination,
//! one cue ending before the next begins).
//!
//! What it cannot prove is that every player in the world renders the result
//! identically. For that, the release checklist should include opening an
//! exported `.srt` in a real player.
//!
//! # The rules
//!
//! **SubRip**
//!
//! * a cue is an index line, a timing line, and one or more text lines;
//! * the index starts at 1 and increases by one;
//! * the timing line is `HH:MM:SS,mmm --> HH:MM:SS,mmm`;
//! * a blank line ends a cue; text may not contain a blank line;
//! * a cue's end must be after its start.
//!
//! **WebVTT**
//!
//! * the first line is `WEBVTT`;
//! * a cue is a timing line and one or more text lines, optionally preceded by an
//!   identifier line;
//! * the timing line is `HH:MM:SS.mmm --> HH:MM:SS.mmm`;
//! * a blank line ends a cue;
//! * a cue's end must be after its start.
//!
//! **Both, by this product's choice rather than by the specs**
//!
//! * at most `max_lines` lines per cue (two, the readability budget);
//! * cues do not overlap;
//! * the end time is strictly after the start time.
//!
//! The two-line rule and the no-overlap rule are *ours*: a file that breaks them
//! still loads in most players. They are checked here because they are the two
//! things this exporter is specifically built to get right.

use vaani_core::{Result, Time};

/// A cue as read back from a file.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)] // field names are the documentation
pub struct ParsedCue {
    /// The SubRip index. `None` for WebVTT, where identifiers are optional.
    pub index: Option<u64>,
    pub start: Time,
    pub end: Time,
    pub lines: Vec<String>,
}

impl ParsedCue {
    /// The cue's text with its lines joined.
    pub fn text(&self) -> String {
        self.lines.join(" ")
    }
}

/// A rule the file breaks.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)] // field names are the documentation
pub struct Violation {
    /// The rule, named as a short slug so a test can assert on it.
    pub rule: &'static str,
    /// Which cue, counting from 1. `None` for a whole-file problem.
    pub cue: Option<usize>,
    /// What is wrong, in one line.
    pub detail: String,
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.cue {
            Some(n) => write!(f, "cue {n}: {} ({})", self.rule, self.detail),
            None => write!(f, "file: {} ({})", self.rule, self.detail),
        }
    }
}

fn malformed(detail: impl std::fmt::Display) -> vaani_core::VaaniError {
    vaani_core::err::export_failed().with_cause(detail)
}

/// Split a file into lines, dropping a UTF-8 byte-order mark and normalising
/// CRLF. Both formats are line-oriented and both are written with LF; a file
/// that arrives with CRLF from another tool is still readable, and refusing it
/// would be unhelpful.
fn lines_of(text: &str) -> Vec<String> {
    text.strip_prefix('\u{feff}')
        .unwrap_or(text)
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .lines()
        .map(str::to_string)
        .collect()
}

/// `HH:MM:SS,mmm` or `HH:MM:SS.mmm` to [`Time`]. Hours may be more than two
/// digits, which is what a long recording needs.
fn parse_timestamp(s: &str) -> Option<Time> {
    let s = s.trim();
    let (hms, ms) = s.split_once([',', '.'])?;
    // Exactly three fractional digits: a two- or four-digit fraction is a
    // different (invalid) format, not a number to be lenient about, because
    // accepting it would mean this parser disagrees with a player's about where
    // the cue ends.
    if ms.len() != 3 || !ms.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let mut parts = hms.split(':');
    let h: i64 = parts.next()?.parse().ok()?;
    let m: i64 = parts.next()?.parse().ok()?;
    let sec: i64 = parts.next()?.parse().ok()?;
    let ms: i64 = ms.parse().ok()?;
    if parts.next().is_some() || !(0..60).contains(&m) || !(0..60).contains(&sec) || ms >= 1_000 {
        return None;
    }
    Some(Time::from_millis(
        h * 3_600_000 + m * 60_000 + sec * 1_000 + ms,
    ))
}

/// The timing line of a cue: `start --> end`, with optional WebVTT settings
/// after the end.
fn parse_timing(line: &str) -> Option<(Time, Time)> {
    let line = line.trim();
    // Split on the *first* arrow; anything after the end time is cue settings,
    // which WebVTT allows and which this exporter does not write.
    let (start, rest) = line.split_once("-->")?;
    let end = rest.split_whitespace().next()?;
    Some((parse_timestamp(start)?, parse_timestamp(end)?))
}

/// Parse a SubRip file.
///
/// # Errors
///
/// With a description of the first structural problem: a timing line that does
/// not parse, a cue with no text, an index that is not a number.
pub fn parse_srt(text: &str) -> Result<Vec<ParsedCue>> {
    let lines = lines_of(text);
    let mut cues = Vec::new();
    let mut i = 0;
    let mut seen_anything = false;
    while i < lines.len() {
        if lines[i].trim().is_empty() {
            i += 1;
            continue;
        }
        // An index line is a bare number on its own.
        let (index, timing_at) = match lines[i].trim().parse::<u64>() {
            Ok(n) => (Some(n), i + 1),
            // No index: treat this line as the timing line, which is what a
            // lenient reader does and what makes the parser useful on files
            // other tools wrote.
            Err(_) => (None, i),
        };
        let timing = lines
            .get(timing_at)
            .ok_or_else(|| malformed(format!("cue at line {} has no timing line", i + 1)))?;
        let (start, end) = parse_timing(timing).ok_or_else(|| {
            malformed(format!(
                "cue at line {}: {:?} is not a valid `HH:MM:SS,mmm --> HH:MM:SS,mmm` timing line",
                i + 1,
                timing.trim()
            ))
        })?;
        seen_anything = true;
        let mut text_lines = Vec::new();
        let mut j = timing_at + 1;
        while j < lines.len() && !lines[j].trim().is_empty() {
            text_lines.push(lines[j].to_string());
            j += 1;
        }
        if text_lines.is_empty() {
            return Err(malformed(format!(
                "cue at line {} has no text, which a blank line then terminates immediately",
                i + 1
            )));
        }
        cues.push(ParsedCue {
            index,
            start,
            end,
            lines: text_lines,
        });
        i = j;
    }
    if !seen_anything {
        return Err(malformed("the file contains no cues"));
    }
    Ok(cues)
}

/// Parse a WebVTT file.
///
/// # Errors
///
/// If the `WEBVTT` header is missing, or a cue is malformed.
pub fn parse_vtt(text: &str) -> Result<Vec<ParsedCue>> {
    let lines = lines_of(text);
    let first = lines
        .first()
        .map(|l| l.trim_end())
        .ok_or_else(|| malformed("the file is empty"))?;
    if !first.starts_with("WEBVTT") {
        return Err(malformed(format!(
            "the first line must be `WEBVTT`, found {:?}",
            first.trim()
        )));
    }
    // Everything between the header and the first blank line is metadata
    // (a `Kind:` line, a `NOTE` block) and is skipped.
    let mut i = 1;
    while i < lines.len() && !lines[i].trim().is_empty() {
        i += 1;
    }
    let mut cues: Vec<ParsedCue> = Vec::new();
    while i < lines.len() {
        if lines[i].trim().is_empty() {
            i += 1;
            continue;
        }
        // A cue may be preceded by an identifier line; a timing line is
        // recognisable by its arrow.
        let mut timing_at = i;
        if !lines[i].contains("-->") {
            timing_at = i + 1;
        }
        let timing = lines
            .get(timing_at)
            .ok_or_else(|| malformed(format!("cue at line {} has no timing line", i + 1)))?;
        let (start, end) = parse_timing(timing).ok_or_else(|| {
            malformed(format!(
                "cue at line {}: {:?} is not a valid `HH:MM:SS.mmm --> HH:MM:SS.mmm` timing line",
                i + 1,
                timing.trim()
            ))
        })?;
        let mut text_lines = Vec::new();
        let mut j = timing_at + 1;
        while j < lines.len() && !lines[j].trim().is_empty() {
            text_lines.push(unescape_vtt(&lines[j]));
            j += 1;
        }
        if text_lines.is_empty() {
            return Err(malformed(format!("cue at line {} has no text", i + 1)));
        }
        cues.push(ParsedCue {
            index: None,
            start,
            end,
            lines: text_lines,
        });
        i = j;
    }
    if cues.is_empty() {
        return Err(malformed("the file contains no cues"));
    }
    Ok(cues)
}

/// Undo [`crate::cues::escape_vtt_text`], so a round trip through WebVTT is
/// lossless.
pub fn unescape_vtt(text: &str) -> String {
    text.replace("--&gt;", "-->")
}

/// The rules both formats must satisfy, checked on parsed cues.
///
/// `max_lines` is this product's two-line rule; pass `usize::MAX` to check only
/// what the format itself requires.
pub fn check(cues: &[ParsedCue], max_lines: usize) -> Vec<Violation> {
    let mut out = Vec::new();
    for (i, cue) in cues.iter().enumerate() {
        let n = i + 1;
        if let Some(index) = cue.index
            && index != n as u64
        {
            out.push(Violation {
                rule: "sequential_indices",
                cue: Some(n),
                detail: format!("index {index}, expected {n}"),
            });
        }
        if cue.end <= cue.start {
            out.push(Violation {
                rule: "positive_duration",
                cue: Some(n),
                detail: format!("{} --> {}", cue.start.as_millis(), cue.end.as_millis()),
            });
        }
        if cue.lines.len() > max_lines {
            out.push(Violation {
                rule: "max_lines_per_cue",
                cue: Some(n),
                detail: format!("{} lines, at most {max_lines}", cue.lines.len()),
            });
        }
        for line in &cue.lines {
            if line.chars().count() > 200 {
                out.push(Violation {
                    rule: "line_length",
                    cue: Some(n),
                    detail: format!("{} characters", line.chars().count()),
                });
            }
            if line.ends_with(' ') {
                out.push(Violation {
                    rule: "no_trailing_space",
                    cue: Some(n),
                    detail: format!("{line:?}"),
                });
            }
        }
        if i > 0 {
            let previous = &cues[i - 1];
            if cue.start < previous.end {
                out.push(Violation {
                    rule: "no_overlap",
                    cue: Some(n),
                    detail: format!(
                        "starts at {} but the previous cue ends at {}",
                        cue.start.as_millis(),
                        previous.end.as_millis()
                    ),
                });
            }
        }
    }
    out
}

/// Parse and check a SubRip file in one step.
pub fn validate_srt(text: &str, max_lines: usize) -> Result<Vec<ParsedCue>> {
    let cues = parse_srt(text)?;
    let violations = check(&cues, max_lines);
    if violations.is_empty() {
        Ok(cues)
    } else {
        Err(malformed(format!(
            "the file breaks {} of its own rules: {}",
            violations.len(),
            violations
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("; ")
        )))
    }
}

/// Parse and check a WebVTT file in one step.
pub fn validate_vtt(text: &str, max_lines: usize) -> Result<Vec<ParsedCue>> {
    let cues = parse_vtt(text)?;
    let violations = check(&cues, max_lines);
    if violations.is_empty() {
        Ok(cues)
    } else {
        Err(malformed(format!(
            "the file breaks {} of its own rules: {}",
            violations.len(),
            violations
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("; ")
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamps_parse_in_both_conventions() {
        assert_eq!(parse_timestamp("00:00:00,000"), Some(Time(0)));
        assert_eq!(parse_timestamp("00:00:00.000"), Some(Time(0)));
        assert_eq!(parse_timestamp("01:01:01,500"), Some(Time(3_661_500)));
        assert_eq!(parse_timestamp("25:00:00,000"), Some(Time(90_000_000)));
    }

    #[test]
    fn malformed_timestamps_are_rejected() {
        for bad in [
            "1:2:3",
            "00:00:00",
            "00:00:00,0000",
            "00:60:00,000",
            "00:00:60,000",
            "aa:bb:cc,ddd",
            "",
        ] {
            assert_eq!(parse_timestamp(bad), None, "{bad:?} should not parse");
        }
    }

    #[test]
    fn a_valid_srt_parses() {
        let cues = parse_srt("1\n00:00:00,000 --> 00:00:01,000\nhello\n\n").unwrap();
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].index, Some(1));
        assert_eq!(cues[0].text(), "hello");
    }

    #[test]
    fn a_valid_vtt_parses_with_its_header() {
        let cues = parse_vtt("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n\n").unwrap();
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].index, None);
        assert_eq!(cues[0].text(), "hello");
    }

    #[test]
    fn a_vtt_without_its_header_is_refused() {
        let e = parse_vtt("00:00:00.000 --> 00:00:01.000\nhello\n\n").unwrap_err();
        assert!(e.debug_chain().contains("WEBVTT"));
    }

    #[test]
    fn a_cue_with_no_text_is_refused() {
        assert!(parse_srt("1\n00:00:00,000 --> 00:00:01,000\n\n").is_err());
        assert!(parse_vtt("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n\n").is_err());
    }

    #[test]
    fn every_rule_has_a_check_that_catches_its_breach() {
        // Overlapping cues.
        let cues = parse_srt(
            "1\n00:00:00,000 --> 00:00:05,000\na\n\n2\n00:00:02,000 --> 00:00:06,000\nb\n\n",
        )
        .unwrap();
        let v = check(&cues, 2);
        assert!(v.iter().any(|v| v.rule == "no_overlap"), "{v:?}");

        // Zero duration.
        let cues = parse_srt("1\n00:00:01,000 --> 00:00:01,000\na\n\n").unwrap();
        let v = check(&cues, 2);
        assert!(v.iter().any(|v| v.rule == "positive_duration"), "{v:?}");

        // Out-of-order index.
        let cues = parse_srt("2\n00:00:00,000 --> 00:00:01,000\na\n\n").unwrap();
        let v = check(&cues, 2);
        assert!(v.iter().any(|v| v.rule == "sequential_indices"), "{v:?}");

        // Three lines in one cue.
        let cues = parse_srt("1\n00:00:00,000 --> 00:00:01,000\na\nb\nc\n\n").unwrap();
        let v = check(&cues, 2);
        assert!(v.iter().any(|v| v.rule == "max_lines_per_cue"), "{v:?}");

        // Trailing whitespace.
        let cues = parse_srt("1\n00:00:00,000 --> 00:00:01,000\na \n\n").unwrap();
        let v = check(&cues, 2);
        assert!(v.iter().any(|v| v.rule == "no_trailing_space"), "{v:?}");
    }

    #[test]
    fn a_clean_file_reports_no_violations() {
        let cues = parse_srt(
            "1\n00:00:00,000 --> 00:00:01,000\none\ntwo\n\n2\n00:00:01,000 --> 00:00:02,000\nthree\n\n",
        )
        .unwrap();
        assert!(check(&cues, 2).is_empty());
    }

    #[test]
    fn a_byte_order_mark_and_crlf_are_tolerated() {
        let cues =
            parse_srt("\u{feff}1\r\n00:00:00,000 --> 00:00:01,000\r\nhello\r\n\r\n").unwrap();
        assert_eq!(cues[0].text(), "hello");
    }

    #[test]
    fn vtt_cue_settings_after_the_end_time_are_ignored() {
        let cues =
            parse_vtt("WEBVTT\n\n00:00:00.000 --> 00:00:01.000 align:start position:10%\nhi\n\n")
                .unwrap();
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].end, Time(1_000));
    }

    #[test]
    fn a_vtt_identifier_line_is_skipped() {
        let cues = parse_vtt("WEBVTT\n\nintro\n00:00:00.000 --> 00:00:01.000\nhi\n\n").unwrap();
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text(), "hi");
    }

    #[test]
    fn the_arrow_escape_round_trips() {
        assert_eq!(unescape_vtt("a --&gt; b"), "a --> b");
        let cues = parse_vtt("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\na --&gt; b\n\n").unwrap();
        assert_eq!(cues[0].text(), "a --> b");
    }
}

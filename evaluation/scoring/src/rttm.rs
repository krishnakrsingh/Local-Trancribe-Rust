//! NIST RTTM: the diarization reference and hypothesis format.
//!
//! One line, ten whitespace-separated fields:
//!
//! ```text
//! Type FileID ChanID StartTime Duration Ortho Stype Name NoSpeakers Confidence
//! SPEAKER meeting 1 12.34 3.45 <NA> <NA> spk1 <NA> <NA>
//! ```
//!
//! * `Type` is `SPEAKER`, `SPKR-INFO` (metadata, ignored for scoring) or
//!   `NOSPEAKER` (marked non-speech; **not** counted as a `SPEAKER` turn).
//! * `StartTime` and `Duration` are in **seconds**.
//! * `Name` is the speaker label. The harness never compares labels as
//!   strings: a permutation is resolved first, so `SPEAKER_00` and `spk1` may
//!   legitimately name the same person.
//!
//! Times are carried internally as integer **microseconds**. Floating-point
//! seconds lose the ability to represent a 1 ms boundary exactly, and a
//! diarization boundary that lands 1 ms off is a 1 ms DER error that no amount
//! of later arithmetic can recover. Parsing therefore multiplies by
//! `MICROS` once and never divides again except when writing.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Microseconds per second. The internal time unit.
pub const MICROS: i64 = 1_000_000;

/// Record type as written in field 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum RttmType {
    /// A speech turn by one speaker.
    Speaker,
    /// Per-file metadata. Ignored for scoring.
    SpkrInfo,
    /// A marked non-speech region.
    NoSpeaker,
}

impl RttmType {
    /// Wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Speaker => "SPEAKER",
            Self::SpkrInfo => "SPKR-INFO",
            Self::NoSpeaker => "NOSPEAKER",
        }
    }

    /// Parse a wire name, case-insensitively.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_uppercase().as_str() {
            "SPEAKER" => Some(Self::Speaker),
            "SPKR-INFO" | "SPKRINFO" => Some(Self::SpkrInfo),
            "NOSPEAKER" => Some(Self::NoSpeaker),
            _ => None,
        }
    }
}

/// One RTTM record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RttmRecord {
    /// Record type.
    pub r#type: RttmType,
    /// File identifier. Free text; never compared across files.
    pub file_id: String,
    /// Channel index, 1-based.
    pub channel: u32,
    /// Start time in microseconds from the start of the file.
    pub start_us: i64,
    /// Duration in microseconds.
    pub duration_us: i64,
    /// Speaker label. Free text.
    pub name: String,
    /// Number of speakers, if the record carried one.
    pub num_speakers: Option<u32>,
    /// Confidence, if the record carried one.
    pub confidence: Option<f64>,
    /// The source line, kept for error messages and for round-trip tests.
    pub raw: String,
}

/// Equality ignores [`RttmRecord::raw`], so a parsed record compares equal to
/// the record it was written back out as. `raw` is a diagnostic, not data;
/// letting it participate in equality would make every round-trip test fail
/// for a reason that has nothing to do with the parser.
impl PartialEq for RttmRecord {
    fn eq(&self, other: &Self) -> bool {
        self.r#type == other.r#type
            && self.file_id == other.file_id
            && self.channel == other.channel
            && self.start_us == other.start_us
            && self.duration_us == other.duration_us
            && self.name == other.name
            && self.num_speakers == other.num_speakers
            && self.confidence == other.confidence
    }
}

impl RttmRecord {
    /// A `SPEAKER` record, the only kind that carries a turn.
    pub fn speaker(name: &str, start_us: i64, duration_us: i64) -> Self {
        Self {
            r#type: RttmType::Speaker,
            file_id: "audio".to_string(),
            channel: 1,
            start_us,
            duration_us,
            name: name.to_string(),
            num_speakers: None,
            confidence: None,
            raw: String::new(),
        }
    }

    /// End time in microseconds, saturating.
    pub fn end_us(&self) -> i64 {
        self.start_us.saturating_add(self.duration_us).max(0)
    }

    /// True for a `SPEAKER` record with a non-empty label and positive
    /// duration. Everything else is metadata or an artefact.
    pub fn is_turn(&self) -> bool {
        self.r#type == RttmType::Speaker
            && !self.name.trim().is_empty()
            && self.duration_us > 0
            && self.start_us >= 0
    }
}

/// Why a file could not be parsed. Carries the line number, because "invalid
/// RTTM on line 1" is useless in a 400-line reference file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RttmError {
    /// 1-based line number.
    pub line: usize,
    /// What was wrong.
    pub message: String,
}

impl fmt::Display for RttmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RTTM line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for RttmError {}

/// Parse an RTTM document. Blank lines and `#` comments are skipped; every
/// other line must have ten fields, and field 1 must be a known type.
pub fn parse(text: &str) -> Result<Vec<RttmRecord>, RttmError> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let n = i + 1;
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') || t.starts_with(";;") {
            continue;
        }
        let f: Vec<&str> = t.split_whitespace().collect();
        if f.len() < 10 {
            return Err(RttmError {
                line: n,
                message: format!("expected 10 fields, found {}", f.len()),
            });
        }
        let r#type = RttmType::parse(f[0]).ok_or_else(|| RttmError {
            line: n,
            message: format!("unknown record type `{}`", f[0]),
        })?;
        let channel = f[2].parse::<u32>().map_err(|_| RttmError {
            line: n,
            message: format!("ChanID `{}` is not an integer", f[2]),
        })?;
        let start_us = parse_seconds(f[3]).map_err(|m| RttmError {
            line: n,
            message: format!("StartTime: {m}"),
        })?;
        let duration_us = parse_seconds(f[4]).map_err(|m| RttmError {
            line: n,
            message: format!("Duration: {m}"),
        })?;
        let num_speakers = parse_opt_u32(f[8]);
        let confidence = f[9].parse::<f64>().ok().filter(|c| c.is_finite());
        out.push(RttmRecord {
            r#type,
            file_id: f[1].to_string(),
            channel,
            start_us,
            duration_us,
            name: f[7].to_string(),
            num_speakers,
            confidence,
            raw: t.to_string(),
        });
    }
    Ok(out)
}

fn parse_opt_u32(s: &str) -> Option<u32> {
    if s.eq_ignore_ascii_case("<NA>") || s == "-" {
        None
    } else {
        s.parse().ok()
    }
}

/// Parse decimal seconds into integer microseconds, exactly.
fn parse_seconds(s: &str) -> Result<i64, String> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("<NA>") {
        return Err("<NA> is not a time".to_string());
    }
    let neg = s.starts_with('-');
    let body = s.trim_start_matches(['-', '+']);
    let (int_part, frac_part) = match body.split_once('.') {
        Some((a, b)) => (a, b),
        None => (body, ""),
    };
    if int_part.is_empty() && frac_part.is_empty() {
        return Err(format!("`{s}` is not a time"));
    }
    if !int_part.chars().all(|c| c.is_ascii_digit())
        || !frac_part.chars().all(|c| c.is_ascii_digit())
    {
        return Err(format!("`{s}` is not a decimal number"));
    }
    let int_v: i64 = if int_part.is_empty() {
        0
    } else {
        int_part
            .parse()
            .map_err(|_| format!("`{s}` does not fit in i64 seconds"))?
    };
    // Take six fractional digits; anything finer is sub-microsecond and below
    // the resolution of the pipeline that produced the audio.
    let mut micro_frac: i64 = 0;
    for i in 0..6 {
        let d = frac_part
            .as_bytes()
            .get(i)
            .map(|b| (b - b'0') as i64)
            .unwrap_or(0);
        micro_frac += d * 10i64.pow(5 - i as u32);
    }
    let total = int_v
        .saturating_mul(MICROS)
        .saturating_add(micro_frac)
        .saturating_mul(if neg { -1 } else { 1 });
    Ok(total)
}

/// Render microseconds as seconds with exactly six decimals, so a written file
/// round-trips through [`parse`] to the same integer.
fn format_seconds(us: i64) -> String {
    let neg = us < 0;
    let a = us.unsigned_abs();
    let s = a / MICROS as u64;
    let frac = a % MICROS as u64;
    format!("{}{}.{:06}", if neg { "-" } else { "" }, s, frac)
}

impl fmt::Display for RttmRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let na = "<NA>";
        write!(
            f,
            "{} {} {} {} {} {} {} {} {} {}",
            self.r#type.as_str(),
            self.file_id,
            self.channel,
            format_seconds(self.start_us),
            format_seconds(self.duration_us.max(0)),
            na,
            self.r#type.as_str(),
            self.name,
            self.num_speakers
                .map(|n| n.to_string())
                .unwrap_or_else(|| na.into()),
            self.confidence
                .map(|c| format!("{c:.2}"))
                .unwrap_or_else(|| na.into()),
        )
    }
}

/// Render records as an RTTM document, sorted by start time then speaker.
pub fn write(records: &[RttmRecord]) -> String {
    let mut rs = records.to_vec();
    rs.sort_by(|a, b| a.start_us.cmp(&b.start_us).then(a.name.cmp(&b.name)));
    let mut s = String::new();
    for r in &rs {
        s.push_str(&r.to_string());
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_canonical_line_round_trips_exactly() {
        let line = "SPEAKER meeting 1 12.34 3.45 <NA> <NA> spk1 <NA> <NA>";
        let rs = parse(line).unwrap();
        assert_eq!(rs.len(), 1);
        let r = &rs[0];
        assert_eq!(r.r#type, RttmType::Speaker);
        assert_eq!(r.file_id, "meeting");
        assert_eq!(r.channel, 1);
        assert_eq!(r.start_us, 12_340_000);
        assert_eq!(r.duration_us, 3_450_000);
        assert_eq!(r.name, "spk1");
        assert!(r.is_turn());
        let again = parse(&r.to_string()).unwrap();
        assert_eq!(again, rs);
    }

    #[test]
    fn a_millisecond_boundary_is_exact() {
        let rs = parse("SPEAKER f 1 0.001 0.001 <NA> <NA> a <NA> <NA>").unwrap();
        assert_eq!(rs[0].start_us, 1_000);
        assert_eq!(rs[0].end_us(), 2_000);
    }

    #[test]
    fn sub_microsecond_digits_are_truncated_not_rounded() {
        let rs = parse("SPEAKER f 1 1.0000009 0.5 <NA> <NA> a <NA> <NA>").unwrap();
        assert_eq!(rs[0].start_us, 1_000_000);
    }

    #[test]
    fn comments_and_blank_lines_are_skipped() {
        let text = "# a comment\n\nSPEAKER f 1 0 1 <NA> <NA> a <NA> <NA>\n\n";
        assert_eq!(parse(text).unwrap().len(), 1);
    }

    #[test]
    fn spkr_info_and_nospeaker_are_not_turns() {
        let text = "SPKR-INFO f 1 0 10 <NA> <NA> <NA> 2 <NA>\n\
                    NOSPEAKER f 1 1 1 <NA> <NA> <NA> <NA> <NA>\n\
                    SPEAKER f 1 2 1 <NA> <NA> a <NA> <NA>\n";
        let rs = parse(text).unwrap();
        assert_eq!(rs.len(), 3);
        assert_eq!(rs.iter().filter(|r| r.is_turn()).count(), 1);
    }

    #[test]
    fn a_short_line_names_its_line_number() {
        let e = parse("SPEAKER f 1 0\n").unwrap_err();
        assert_eq!(e.line, 1);
        assert!(e.to_string().contains("10 fields"));
        let e = parse("\n\nSPEAKER f 1 0 1 <NA> <NA> a <NA>\n").unwrap_err();
        assert_eq!(e.line, 3);
    }

    #[test]
    fn an_unknown_type_is_rejected_rather_than_guessed() {
        let e = parse("AUDIOMONO f 1 0 1 <NA> <NA> a <NA> <NA>").unwrap_err();
        assert!(e.message.contains("AUDIOMONO"));
    }

    #[test]
    fn a_non_numeric_time_is_rejected() {
        let e = parse("SPEAKER f 1 later 1 <NA> <NA> a <NA> <NA>").unwrap_err();
        assert!(e.message.contains("StartTime"));
    }

    #[test]
    fn the_writer_sorts_and_round_trips() {
        let rs = vec![
            RttmRecord::speaker("b", 2_000_000, 1_000_000),
            RttmRecord::speaker("a", 1_000_000, 500_000),
        ];
        let doc = write(&rs);
        let lines: Vec<&str> = doc.lines().collect();
        assert!(lines[0].contains(" a "), "{doc}");
        assert!(lines[1].contains(" b "), "{doc}");
        let parsed = parse(&doc).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].name, "a");
        assert_eq!(parsed[0].start_us, 1_000_000);
        assert_eq!(parsed[1].name, "b");
        assert_eq!(parsed[1].start_us, 2_000_000);
    }
}

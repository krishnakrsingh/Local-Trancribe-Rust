//! Rendering a transcript to text, Markdown or JSON.
//!
//! Each renderer is a pure function from `(&Transcript, &ExportOptions)` to a
//! `String`, with no filesystem and no clock, which is what makes "export twice,
//! get the same bytes" a testable property rather than a hope.
//!
//! # Text
//!
//! One line per segment: `[hh:mm:ss] Speaker: words`, or
//! `[hh:mm:ss] words` when the job has no speakers. **Not wrapped**, on purpose:
//! a plain-text transcript is something people paste into an editor, a chat box
//! or a mail, and a hard line break in the middle of a sentence is worse than a
//! long line, which every reader reflows. A consumer that wants wrapping has the
//! JSON.
//!
//! # Markdown
//!
//! A heading, a metadata list, then one section per speaker and one paragraph
//! per segment with a bold timestamp. **Wrapped**, because Markdown is read
//! rendered and a 900-character paragraph is unreadable in a viewer.
//!
//! ## Escaping
//!
//! Body text is escaped so that a transcript containing `#hashtag` or
//! `*emphasis*` renders as the words that were said rather than as markup:
//! `\`, `` ` ``, `*`, `_`, `[`, `]`, `<`, `>`, `|`, `~` are escaped everywhere,
//! and a leading `#`, `>`, `-`, `+`, `*` or an ordered-list marker is escaped
//! only at the start of a line, where it would otherwise begin a block. The
//! title and the metadata values are escaped the same way, so a file called
//! `*urgent*.m4a` cannot become a heading.
//!
//! # JSON
//!
//! Metadata **around** the transcript, never a parallel redefinition of it: the
//! `transcript` field is produced by `vaani_core`'s own `Serialize`, so the
//! exported shape cannot drift from the domain type, and reading it back gives
//! an equal `Transcript` — which is the single most important test in this
//! crate. `exported_at` appears only when the caller supplied a timestamp, so
//! two exports of the same transcript are byte-identical.
//!
//! Key order is fixed by the struct order, and `serde_json` preserves it.

use vaani_core::{APP_VERSION, DATA_MODEL_VERSION, MediaLength, Time, Transcript};

use crate::cues;
use crate::error;
use crate::format::ExportFormat;
use crate::options::ExportOptions;

/// Render a transcript in `format`.
///
/// # Errors
///
/// Only if the transcript cannot produce a valid document in that format, which
/// for these renderers means an internal inconsistency. The file-writing half of
/// an export is [`crate::export_to_file`].
pub fn render(
    transcript: &Transcript,
    format: ExportFormat,
    options: &ExportOptions,
) -> vaani_core::Result<String> {
    match format {
        ExportFormat::Text => Ok(render_text(transcript, options)),
        ExportFormat::Markdown => Ok(render_markdown(transcript, options)),
        ExportFormat::Srt => Ok(cues::render_srt(
            &cues::build_cues(transcript, &options.cue_options()),
            options.include_cue_numbers,
        )),
        ExportFormat::Vtt => Ok(cues::render_vtt(&cues::build_cues(
            transcript,
            &options.cue_options(),
        ))),
        ExportFormat::Json => render_json(transcript, options),
    }
}

/// Replace characters that would break a single line of text, in any format.
///
/// Control characters become a space rather than being dropped, so a stray
/// newline inside a word cannot silently join two words, and a trailing space
/// cannot appear at the end of a subtitle line (which some players render
/// visibly). `\t` and the non-breaking space are kept: they are legitimate text.
pub fn sanitise_line(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    for ch in line.chars() {
        match ch {
            '\n' | '\r' => out.push(' '),
            c if (c as u32) < 0x20 && c != '\t' => out.push(' '),
            c => out.push(c),
        }
    }
    out.trim_end().to_string()
}

/// Collapse runs of whitespace, for formats that treat whitespace as layout.
fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The turn's text with its speaker, if the options ask for one and the
/// transcript has one for this segment.
fn turn_text(
    transcript: &Transcript,
    index: usize,
    options: &ExportOptions,
    with_speaker: bool,
) -> String {
    let seg = &transcript.segments[index];
    let text = collapse(&seg.text());
    if text.is_empty() {
        // A segment with no words produces no line at all. Labelling an empty
        // turn would put "Speaker 1:" on its own line in the transcript, which
        // reads as a truncation bug rather than as silence.
        return String::new();
    }
    if !options.speaker_labels || !with_speaker {
        return text;
    }
    match seg.speaker.and_then(|i| transcript.speaker(i)) {
        Some(s) => format!("{}: {}", s.display_name(), text),
        None => text,
    }
}

fn timestamp_prefix(time: Time, options: &ExportOptions) -> String {
    if options.include_timestamps {
        format!("[{}] ", time.format_hms())
    } else {
        String::new()
    }
}

fn render_text(transcript: &Transcript, options: &ExportOptions) -> String {
    let mut out = String::new();
    for (i, seg) in transcript.segments.iter().enumerate() {
        let text = turn_text(transcript, i, options, true);
        if text.is_empty() {
            continue;
        }
        out.push_str(&timestamp_prefix(seg.start, options));
        out.push_str(&sanitise_line(&text));
        out.push('\n');
    }
    out
}

/// Escape the characters that mean something anywhere inside a Markdown line.
///
/// `\`, `` ` ``, `*`, `_`, `[`, `]`, `<`, `>`, `|`, `~` and `!` are escaped
/// wherever they appear. The remaining Markdown syntax (a leading `#`, a
/// leading `-`, a leading `1.`) only matters at the *start of a line*, so it is
/// handled by [`escape_markdown`], and this function is what a value in the
/// middle of a line needs: escaping a leading marker there would put a
/// backslash in the middle of a version number.
pub fn escape_inline(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    for ch in text.chars() {
        if matches!(
            ch,
            '\\' | '`' | '*' | '_' | '[' | ']' | '<' | '>' | '|' | '~' | '!'
        ) {
            out.push('\\');
        }
        out.push(if ch == '\n' { ' ' } else { ch });
    }
    out
}

/// Escape Markdown text that starts a line: a heading, a paragraph, a list item.
///
/// On top of [`escape_inline`], a leading `#`, `>`, `=`, `-` or `+`, and a
/// leading ordered-list marker, are escaped — those begin a block, and a
/// transcript that starts a line with `1.` would otherwise turn into a numbered
/// list. Mid-line the same characters are ordinary text and are left alone, so a
/// sentence does not become a thicket of backslashes.
pub fn escape_markdown(text: &str) -> String {
    let escaped = escape_inline(text);
    let mut out = String::with_capacity(escaped.len() + 2);
    let mut first = true;
    for (pos, ch) in escaped.chars().enumerate() {
        let leading = if first && pos <= 1 {
            matches!(ch, '#' | '>' | '-' | '+' | '=')
                || (ch.is_ascii_digit() && escaped.chars().nth(pos + 1) == Some('.'))
        } else {
            false
        };
        if leading {
            out.push('\\');
        }
        out.push(ch);
        if !ch.is_whitespace() {
            first = false;
        }
    }
    out
}

fn render_markdown(transcript: &Transcript, options: &ExportOptions) -> String {
    let mut out = String::new();
    out.push_str("# ");
    out.push_str(&escape_markdown(&options.document_title()));
    out.push_str("\n\n");

    if options.include_metadata {
        out.push_str(&markdown_metadata(transcript, options));
        out.push('\n');
    }

    // One section per speaker, in the order the transcript lists them, so a
    // renamed speaker gets a heading with their name. A turn with no speaker
    // does not start a section of its own: "not yet diarized" is not a person,
    // and inventing a "Speaker" heading for it would be a lie about the data.
    let mut current_speaker: Option<usize> = None;
    let mut open_section = false;
    for (i, seg) in transcript.segments.iter().enumerate() {
        // No speaker prefix on the turn itself: the section heading above it
        // already says whose turn this is, and saying it twice reads badly.
        let text = turn_text(transcript, i, options, false);
        if text.is_empty() {
            continue;
        }
        if let Some(who) = seg.speaker
            && Some(who) != current_speaker
        {
            if open_section {
                out.push('\n');
            }
            let name = transcript
                .speaker(who)
                .map(|s| s.display_name())
                .unwrap_or_else(|| "Speaker".to_string());
            out.push_str("## ");
            out.push_str(&escape_markdown(&name));
            out.push_str("\n\n");
            current_speaker = Some(who);
            open_section = true;
        }
        let prefix = if options.include_timestamps {
            format!("**[{}]** ", seg.start.format_hms())
        } else {
            String::new()
        };
        let escaped = escape_markdown(&sanitise_line(&text));
        // The timestamp is part of the line a reader sees, so it counts against
        // the width: the first line is wrapped narrower to leave room for it,
        // and the continuation lines use the full width.
        let first_width = options
            .max_line_length
            .saturating_sub(prefix.chars().count())
            .max(8);
        for (n, line) in wrap_lines(&escaped, first_width, options.max_line_length)
            .into_iter()
            .enumerate()
        {
            if n == 0 {
                out.push_str(&prefix);
            } else {
                out.push('\n');
            }
            out.push_str(&line);
        }
        out.push_str("\n\n");
    }
    out
}

/// Wrap text into lines, breaking at spaces only.
///
/// The first line gets `first_width` and the rest `width`, which is how a
/// paragraph with a bold timestamp in front of it is laid out: the marker is
/// part of the first line a reader sees, so it has to be counted.
fn wrap_lines(text: &str, first_width: usize, width: usize) -> Vec<String> {
    let width = width.max(8);
    let first_width = first_width.max(8);
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for token in text.split(' ') {
        if token.is_empty() {
            continue;
        }
        // While no line has been finished yet, the narrower first-line width
        // applies — including to the tokens that continue the first line.
        let limit = if lines.is_empty() { first_width } else { width };
        if current.is_empty() {
            current.push_str(token);
        } else if current.chars().count() + 1 + token.chars().count() <= limit {
            current.push(' ');
            current.push_str(token);
        } else {
            lines.push(std::mem::replace(&mut current, token.to_string()));
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

fn markdown_metadata(transcript: &Transcript, options: &ExportOptions) -> String {
    // Collected first and rendered second, rather than through a closure that
    // borrows the output while the output is being built.
    let mut rows: Vec<(String, String)> = Vec::new();
    if let Some(name) = &options.source_name {
        rows.push(("Source".into(), name.clone()));
    }
    if let Some(d) = transcript.duration {
        rows.push(("Duration".into(), d.format_hms()));
    }
    rows.push(("Engine".into(), transcript.engine.asr_model.clone()));
    // `asr_version` is not optional in the contract crate: every engine has one.
    rows.push((
        "Engine version".into(),
        transcript.engine.asr_version.clone(),
    ));
    if let Some(v) = &transcript.engine.diarization_model {
        rows.push(("Speakers by".into(), v.to_string()));
    }
    if !transcript.speakers.is_empty() {
        rows.push(("Speakers".into(), transcript.speakers.len().to_string()));
    }
    rows.push((
        "Mode".into(),
        format!("{:?}", transcript.mode).to_lowercase(),
    ));
    rows.push(("Segments".into(), transcript.segments.len().to_string()));
    rows.push((
        "Words".into(),
        transcript
            .segments
            .iter()
            .map(|s| s.words.len())
            .sum::<usize>()
            .to_string(),
    ));
    if transcript.edited {
        rows.push(("Edited".into(), "yes".into()));
    }
    if let Some(ms) = options.timestamp_ms
        && let Some(text) = vaani_storage::clock::rfc3339(ms)
    {
        rows.push(("Exported".into(), text));
    }
    let mut out = String::new();
    for (k, v) in rows {
        out.push_str("- **");
        out.push_str(&escape_markdown(&k));
        out.push_str(":** ");
        out.push_str(&escape_inline(&v));
        out.push('\n');
    }
    out
}

/// The JSON envelope. Field order here is the field order in the file.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[allow(missing_docs)] // field names are the documentation; the type docs explain each
pub struct JsonExport {
    /// Identifies the file as a VaaniDesk export. A consumer can check this
    /// before reading anything else.
    pub format: String,
    /// The envelope's own version, so a future field is a version bump rather
    /// than a silent change.
    pub format_version: u32,
    /// Which build wrote it.
    pub generator: String,
    pub app_version: String,
    /// The domain model's version, from `vaani_core`.
    pub data_model_version: u32,
    /// RFC 3339, present only when the caller supplied a timestamp. Its absence
    /// is what makes the export byte-for-byte reproducible.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exported_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<JsonSource>,
    pub options: JsonOptions,
    /// The transcript, serialised by `vaani_core`. Never a second definition of
    /// it: this is *the* domain type's own representation.
    pub transcript: Transcript,
}

/// What was transcribed, for a consumer that wants the context.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct JsonSource {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<i64>,
    pub speakers: usize,
}

/// The options in force, so an export is self-describing.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[allow(missing_docs)] // these mirror ExportOptions, field for field
pub struct JsonOptions {
    pub speaker_labels: bool,
    pub include_timestamps: bool,
    pub max_line_length: usize,
    pub max_lines_per_cue: usize,
    pub wrap: bool,
}

fn render_json(transcript: &Transcript, options: &ExportOptions) -> vaani_core::Result<String> {
    let export = JsonExport {
        format: "vaani.transcript".to_string(),
        format_version: 1,
        generator: format!("vaani-export {}", crate::VERSION),
        app_version: APP_VERSION.to_string(),
        data_model_version: DATA_MODEL_VERSION,
        exported_at: options.timestamp_ms.and_then(vaani_storage::clock::rfc3339),
        source: options.source_name.as_ref().map(|name| JsonSource {
            file: Some(name.clone()),
            duration_ms: transcript.duration.map(MediaLength::as_millis),
            speakers: transcript.speakers.len(),
        }),
        options: JsonOptions {
            speaker_labels: options.speaker_labels,
            include_timestamps: options.include_timestamps,
            max_line_length: options.max_line_length,
            max_lines_per_cue: options.max_lines_per_cue,
            wrap: options.wrap,
        },
        transcript: transcript.clone(),
    };
    // Pretty, because a user opens this file to look at it. Deterministic,
    // because serde_json preserves field order and there is no map in it.
    serde_json::to_string_pretty(&export)
        .map_err(|e| error::invalid_transcript(&format!("it could not be serialised as JSON: {e}")))
}

/// Read a transcript back out of a JSON export.
///
/// The counterpart to the JSON renderer, and the reason the envelope exists: a
/// consumer needs the `Transcript` and not the file's metadata, and it must be
/// the *same* transcript, so this uses `vaani_core`'s own `Deserialize`.
///
/// # Errors
///
/// If the file is not a VaaniDesk JSON export, or the transcript inside it does
/// not parse.
pub fn read_json_export(text: &str) -> vaani_core::Result<Transcript> {
    let export: JsonExport = serde_json::from_str(text).map_err(error::bad_json)?;
    if export.format != "vaani.transcript" {
        return Err(error::bad_json(format!(
            "expected a \"vaani.transcript\" file, found {:?}",
            export.format
        )));
    }
    Ok(export.transcript)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vaani_core::{EngineStamp, Segment, Speaker, Word};

    fn transcript() -> Transcript {
        let mut t = Transcript::new(EngineStamp::asr("sravaani", "1.0"));
        t.duration = Some(MediaLength::from_millis(65_000));
        t.speakers = vec![Speaker {
            index: 0,
            name: Some("Krishna".into()),
            share: Some(1.0),
        }];
        for (i, (who, text)) in [(Some(0usize), "hello everyone"), (None, "unlabelled")]
            .iter()
            .enumerate()
        {
            let mut seg = Segment::new(Time(i as i64 * 1_000), Time(i as i64 * 1_000 + 900));
            seg.id = format!("s{i}");
            seg.speaker = *who;
            seg.words = text
                .split(' ')
                .enumerate()
                .map(|(w, t)| Word::new(t, Time(w as i64), Time(w as i64 + 1)))
                .collect();
            t.segments.push(seg);
        }
        t
    }

    #[test]
    fn text_has_one_timestamped_line_per_segment() {
        let out = render_text(&transcript(), &ExportOptions::default());
        assert_eq!(out, "[00:00] Krishna: hello everyone\n[00:01] unlabelled\n");
    }

    #[test]
    fn text_timestamps_and_speakers_can_be_turned_off() {
        let o = ExportOptions::default()
            .without_speaker_labels()
            .clone_no_timestamps();
        let out = render_text(&transcript(), &o);
        assert_eq!(out, "hello everyone\nunlabelled\n");
    }

    impl ExportOptions {
        fn clone_no_timestamps(mut self) -> Self {
            self.include_timestamps = false;
            self
        }
    }

    #[test]
    fn markdown_has_a_heading_metadata_and_speaker_sections() {
        let o = ExportOptions::default().with_source_name("standup.m4a");
        let out = render_markdown(&transcript(), &o);
        assert!(out.starts_with("# standup.m4a\n\n"), "{out}");
        // `format_hms` is the transcript viewer's form: minutes and seconds
        // under an hour, and hours when there are any.
        assert!(out.contains("- **Duration:** 01:05"), "{out}");
        assert!(out.contains("## Krishna\n"), "{out}");
        assert!(out.contains("**[00:00]** hello everyone"), "{out}");
        // The unlabelled turn is in the same section, because it has no speaker
        // of its own rather than a second one.
        assert_eq!(out.matches("## ").count(), 1);
    }

    #[test]
    fn markdown_escapes_markup_in_the_body_and_the_title() {
        let mut t = Transcript::new(EngineStamp::asr("m", "1"));
        let mut seg = Segment::new(Time(0), Time(1_000));
        seg.id = "s0".into();
        seg.words = vec![
            Word::new("#hashtag", Time(0), Time(1)),
            Word::new("*bold*", Time(1), Time(2)),
            Word::new("_under_", Time(2), Time(3)),
            Word::new("[link](x)", Time(3), Time(4)),
            Word::new("`code`", Time(4), Time(5)),
            Word::new("a|b", Time(5), Time(6)),
            Word::new("> quote", Time(6), Time(7)),
        ];
        t.segments.push(seg);
        let o = ExportOptions::default()
            .with_source_name("*urgent*.m4a")
            .with_title("#not a heading");
        let out = render_markdown(&t, &o);
        assert!(out.starts_with("# \\#not a heading\n"), "{out}");
        assert!(out.contains("\\#hashtag"), "{out}");
        assert!(out.contains("\\*bold\\*"), "{out}");
        assert!(out.contains("\\_under\\_"), "{out}");
        assert!(out.contains("\\[link\\]"), "{out}");
        assert!(out.contains("\\`code\\`"), "{out}");
        assert!(out.contains("a\\|b"), "{out}");
        assert!(out.contains("\\> quote"), "{out}");
    }

    #[test]
    fn markdown_never_starts_a_line_with_a_list_marker() {
        assert_eq!(escape_markdown("- dash"), "\\- dash");
        assert_eq!(escape_markdown("1. one"), "\\1. one");
        assert_eq!(escape_markdown("> quote"), "\\> quote");
        // Mid-line the same characters are harmless and are not escaped, so the
        // text does not turn into a thicket of backslashes.
        assert_eq!(escape_markdown("a-b"), "a-b");
        assert_eq!(escape_markdown("3-2"), "3-2");
    }

    #[test]
    fn markdown_wraps_but_never_inside_a_word() {
        let mut t = Transcript::new(EngineStamp::asr("m", "1"));
        let mut seg = Segment::new(Time(0), Time(1_000));
        seg.id = "s0".into();
        seg.words = (0..12)
            .map(|i| Word::new(format!("word{i}"), Time(i), Time(i + 1)))
            .collect();
        t.segments.push(seg);
        let o = ExportOptions::default().clone_width(20);
        let out = render_markdown(&t, &o);
        let body: Vec<&str> = out.lines().filter(|l| l.contains("word")).collect();
        assert!(body.len() > 1, "the paragraph should have wrapped");
        for line in body {
            assert!(line.chars().count() <= 20, "{line:?} is too long");
            assert!(!line.contains("wor "), "a word was split");
        }
    }

    impl ExportOptions {
        fn clone_width(mut self, width: usize) -> Self {
            self.max_line_length = width;
            self
        }
    }

    #[test]
    fn json_round_trips_through_its_own_reader() {
        let t = transcript();
        let o = ExportOptions::default().with_source_name("standup.m4a");
        let json = render_json(&t, &o).unwrap();
        assert_eq!(read_json_export(&json).unwrap(), t);
    }

    #[test]
    fn json_is_byte_identical_without_a_timestamp_and_differs_with_one() {
        let t = transcript();
        let a = render_json(&t, &ExportOptions::default()).unwrap();
        let b = render_json(&t, &ExportOptions::default()).unwrap();
        assert_eq!(a, b, "two exports of the same transcript must match");
        assert!(!a.contains("exported_at"), "no timestamp was supplied");

        let stamped = ExportOptions::default().with_timestamp_ms(1_700_000_000_000);
        let c = render_json(&t, &stamped).unwrap();
        assert!(c.contains("\"exported_at\""), "{c}");
        assert!(c.contains("2023-11-14T22:13:20Z"), "{c}");
        assert_ne!(a, c);
    }

    #[test]
    fn json_carries_the_envelope_around_the_domain_type() {
        let t = transcript();
        let o = ExportOptions::default().with_source_name("standup.m4a");
        let json = render_json(&t, &o).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["format"], "vaani.transcript");
        assert_eq!(value["format_version"], 1);
        assert_eq!(value["data_model_version"], DATA_MODEL_VERSION);
        assert_eq!(value["app_version"], APP_VERSION);
        assert_eq!(value["source"]["file"], "standup.m4a");
        assert_eq!(value["source"]["speakers"], 1);
        // The transcript is the domain type's own shape, not a copy of it.
        assert_eq!(
            value["transcript"]["segments"][0]["words"][0]["text"],
            "hello"
        );
        assert_eq!(value["transcript"]["speakers"][0]["name"], "Krishna");
    }

    #[test]
    fn reading_someone_elses_json_is_a_clear_error() {
        assert!(read_json_export("not json at all").is_err());
        assert!(read_json_export("{}").is_err());
        // A complete envelope that is somebody else's format: the parse
        // succeeds and the *format check* is what refuses it, which is the path
        // that matters for a file from another tool.
        const OTHER_FORMAT_JSON: &str = r#"{
            "format": "something.else",
            "format_version": 1,
            "generator": "x",
            "app_version": "x",
            "data_model_version": 1,
            "options": {
                "speaker_labels": true,
                "include_timestamps": true,
                "max_line_length": 42,
                "max_lines_per_cue": 2,
                "wrap": true
            },
            "transcript": {
                "id": "t",
                "segments": [],
                "speakers": [],
                "mode": "raw",
                "engine": {"asr_model": "m", "asr_version": "1"},
                "edited": false
            }
        }"#;
        let e = read_json_export(OTHER_FORMAT_JSON).unwrap_err();
        assert!(
            e.debug_chain().contains("vaani.transcript"),
            "{}",
            e.debug_chain()
        );
        assert_eq!(e.code(), vaani_core::ErrorCode::ExportFailed);
        // A transcript that parses but is the wrong shape is also refused.
        let bad = r#"{"format":"vaani.transcript","format_version":1,"generator":"x",
            "app_version":"x","data_model_version":1,"options":{},"transcript":{"id":1}}"#;
        assert!(read_json_export(bad).is_err());
    }

    #[test]
    fn control_characters_cannot_reach_the_output() {
        assert_eq!(sanitise_line("a\r\nb"), "a  b");
        assert_eq!(sanitise_line("trailing   "), "trailing");
        assert_eq!(sanitise_line("keep\ttab"), "keep\ttab");
        assert_eq!(
            sanitise_line("bell\u{7}"),
            "bell",
            "a trailing control character leaves no trailing space"
        );
    }
}

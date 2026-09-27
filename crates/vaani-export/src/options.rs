//! What the caller wants out of an export.
//!
//! One options struct for every format, rather than one per format, because the
//! choices are mostly shared (speaker labels, wrapping) and a caller that has to
//! learn five option types has learned nothing. Format-specific switches default
//! to the right thing and are only touched deliberately.

use crate::cues::CueOptions;

/// Options for one export.
///
/// [`Default`] is the product default: speakers labelled where diarization ran,
/// timestamps included, metadata included, two lines of 42 characters per cue,
/// and **no overwriting** of an existing file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportOptions {
    /// Prefix each turn with the speaker's name, where diarization produced one.
    pub speaker_labels: bool,
    /// Include timestamps. Meaningless for JSON, which always carries them.
    pub include_timestamps: bool,
    /// Include the header block: source file, duration, engine, export time.
    /// Applies to Markdown and JSON.
    pub include_metadata: bool,
    /// Number the cues. SubRip requires it; the spec calls it optional.
    pub include_cue_numbers: bool,
    /// Characters per line for wrapped formats.
    pub max_line_length: usize,
    /// Lines per cue for the subtitle formats.
    pub max_lines_per_cue: usize,
    /// Whether to wrap long lines at all.
    pub wrap: bool,
    /// Replace an existing file. `false` by default: an export must never
    /// silently destroy the previous one.
    pub overwrite: bool,
    /// When the export happened, in epoch milliseconds, recorded in the JSON
    /// metadata. `None` omits the field, which is what makes an export
    /// **byte-for-byte reproducible** — the property the round-trip tests rely
    /// on. The CLI and the UI set it.
    pub timestamp_ms: Option<i64>,
    /// The source media's file name, used in the metadata block and to build a
    /// filename when the target is a directory.
    pub source_name: Option<String>,
    /// Overrides the document title in Markdown and JSON.
    pub title: Option<String>,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            speaker_labels: true,
            include_timestamps: true,
            include_metadata: true,
            include_cue_numbers: true,
            max_line_length: 42,
            max_lines_per_cue: 2,
            wrap: true,
            overwrite: false,
            timestamp_ms: None,
            source_name: None,
            title: None,
        }
    }
}

impl ExportOptions {
    /// The cue policy these options imply.
    pub fn cue_options(&self) -> CueOptions {
        CueOptions {
            max_line_length: self.max_line_length,
            max_lines_per_cue: self.max_lines_per_cue,
            wrap: self.wrap,
            speaker_labels: self.speaker_labels,
        }
    }

    /// Replace an existing file. Without this, an export that would overwrite
    /// something fails with a clear error.
    #[must_use]
    pub fn with_overwrite(mut self, overwrite: bool) -> Self {
        self.overwrite = overwrite;
        self
    }

    /// Name the source media, for the metadata block and the filename.
    #[must_use]
    pub fn with_source_name(mut self, name: impl Into<String>) -> Self {
        self.source_name = Some(name.into());
        self
    }

    /// Override the document title.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Record when the export happened, in epoch milliseconds.
    #[must_use]
    pub fn with_timestamp_ms(mut self, ms: i64) -> Self {
        self.timestamp_ms = Some(ms);
        self
    }

    /// Turn speaker labels off.
    #[must_use]
    pub fn without_speaker_labels(mut self) -> Self {
        self.speaker_labels = false;
        self
    }

    /// The document title: the caller's, or the source file's name, or a
    /// neutral default.
    pub fn document_title(&self) -> String {
        if let Some(t) = &self.title {
            return t.clone();
        }
        match &self.source_name {
            Some(name) => name.clone(),
            None => "Transcript".to_string(),
        }
    }

    /// The filename stem for a directory target.
    pub fn file_stem(&self) -> String {
        crate::format::default_stem(self.source_name.as_deref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_are_the_product_defaults() {
        let o = ExportOptions::default();
        assert!(o.speaker_labels, "diarized transcripts are labelled");
        assert!(o.include_timestamps);
        assert!(o.include_metadata);
        assert!(o.include_cue_numbers);
        assert_eq!(o.max_line_length, 42);
        assert_eq!(o.max_lines_per_cue, 2);
        assert!(o.wrap);
        assert!(!o.overwrite, "an export must not clobber by default");
        assert_eq!(o.timestamp_ms, None, "reproducible by default");
    }

    #[test]
    fn the_builders_compose() {
        let o = ExportOptions::default()
            .with_overwrite(true)
            .with_source_name("standup.m4a")
            .with_title("Monday standup")
            .with_timestamp_ms(1_700_000_000_000)
            .without_speaker_labels();
        assert!(o.overwrite);
        assert_eq!(o.document_title(), "Monday standup");
        assert_eq!(o.file_stem(), "standup");
        assert!(!o.speaker_labels);
        assert_eq!(o.timestamp_ms, Some(1_700_000_000_000));
    }

    #[test]
    fn the_title_falls_back_to_the_source_name_then_to_a_neutral_word() {
        assert_eq!(
            ExportOptions::default()
                .with_source_name("standup.m4a")
                .document_title(),
            "standup.m4a"
        );
        assert_eq!(ExportOptions::default().document_title(), "Transcript");
    }

    #[test]
    fn the_cue_policy_follows_the_options() {
        let o = ExportOptions::default()
            .without_speaker_labels()
            .clone_with_wrap(false)
            .clone_with_width(60);
        let c = o.cue_options();
        assert_eq!(c.max_line_length, 60);
        assert!(!c.wrap);
        assert!(!c.speaker_labels);
    }

    impl ExportOptions {
        fn clone_with_wrap(mut self, wrap: bool) -> Self {
            self.wrap = wrap;
            self
        }
        fn clone_with_width(mut self, width: usize) -> Self {
            self.max_line_length = width;
            self
        }
    }
}

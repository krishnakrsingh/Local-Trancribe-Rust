//! The export formats, and where a file goes.
//!
//! # The list is frozen
//!
//! These five variants — `Text`, `Markdown`, `Srt`, `Vtt`, `Json` — are the ones
//! the UI offers, and they are matched by name to `vaani_ui::ExportFormat`. The
//! UI owns that list; this crate owns the rendering. Adding a sixth format is a
//! deliberate product decision that has to be made in *both* places, so nothing
//! here invents one on its own.
//!
//! The extensions match `vaani_core::EXPORT_EXTENSIONS`, which is the third
//! place the same five appear (the file picker). One test asserts the agreement.

use std::path::{Path, PathBuf};

use vaani_core::Result;

use crate::error;

/// A transcript format.
///
/// Deliberately the same five variants, in the same order, as the UI's frozen
/// list, so `ExportFormat::ALL` can be zipped with it directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ExportFormat {
    /// Plain text, one timestamped line per segment. For pasting into a chat
    /// window or a document.
    Text,
    /// Markdown with headings, a metadata block and speaker sections. For a
    /// notes system, a wiki, or a pull-request comment.
    Markdown,
    /// SubRip subtitles. For a video player.
    Srt,
    /// WebVTT subtitles. For the web, and for anything that plays in a browser.
    Vtt,
    /// The whole transcript as JSON, with metadata around it. For another tool.
    Json,
}

impl ExportFormat {
    /// Every format, in the order the UI lists them.
    pub const ALL: [ExportFormat; 5] =
        [Self::Text, Self::Markdown, Self::Srt, Self::Vtt, Self::Json];

    /// The file extension, without the dot. Matches
    /// `vaani_core::EXPORT_EXTENSIONS` and the file picker's filter.
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Text => "txt",
            Self::Markdown => "md",
            Self::Srt => "srt",
            Self::Vtt => "vtt",
            Self::Json => "json",
        }
    }

    /// The name shown in the export dialog.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Text => "Plain text",
            Self::Markdown => "Markdown",
            Self::Srt => "SubRip subtitles",
            Self::Vtt => "WebVTT subtitles",
            Self::Json => "JSON (full structure)",
        }
    }

    /// A one-line description for a tooltip or a CLI `--help`.
    pub const fn description(self) -> &'static str {
        match self {
            Self::Text => "One timestamped line per segment",
            Self::Markdown => "Headings, a metadata block and speaker sections",
            Self::Srt => "SubRip subtitles for a video player",
            Self::Vtt => "WebVTT subtitles for the browser",
            Self::Json => "The whole transcript, with metadata",
        }
    }

    /// The IANA media type, for a file manager or an HTTP response.
    pub const fn media_type(self) -> &'static str {
        match self {
            Self::Text => "text/plain; charset=utf-8",
            Self::Markdown => "text/markdown; charset=utf-8",
            Self::Srt => "application/x-subrip; charset=utf-8",
            Self::Vtt => "text/vtt; charset=utf-8",
            Self::Json => "application/json; charset=utf-8",
        }
    }

    /// Whether this format carries timestamps. The UI uses it to decide whether
    /// to offer a "include timestamps" toggle at all.
    pub const fn has_timestamps(self) -> bool {
        matches!(
            self,
            Self::Text | Self::Markdown | Self::Srt | Self::Vtt | Self::Json
        )
    }

    /// Whether this format is line-oriented and therefore subject to the
    /// two-line-per-cue rule.
    pub const fn is_cue_based(self) -> bool {
        matches!(self, Self::Srt | Self::Vtt)
    }

    /// The wire name, used in the export log and in JSON metadata.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Markdown => "markdown",
            Self::Srt => "srt",
            Self::Vtt => "vtt",
            Self::Json => "json",
        }
    }

    /// Parse the wire name. A format this build does not know is `None` rather
    /// than a default, so a newer build's export log still reads correctly.
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|f| f.as_str() == s)
    }

    /// A filename for this format, given a stem.
    ///
    /// The stem is sanitised by [`crate::writer::safe_file_name`], so this is
    /// safe to call with a media file's name.
    pub fn file_name(self, stem: &str) -> String {
        format!(
            "{}.{}",
            crate::writer::safe_file_name(stem),
            self.extension()
        )
    }
}

impl std::fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Where the user wants the file.
///
/// Deliberately the same two variants as the UI's frozen `ExportTarget`, so the
/// UI can hand one straight through. `Directory` means "pick a name for me in
/// here"; `File` means "exactly here".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportTarget {
    /// A directory the user chose. The filename is derived from the source
    /// media and the format.
    Directory(PathBuf),
    /// An exact path, from a save dialog. The extension is the one the user
    /// typed, even if it disagrees with the format.
    File(PathBuf),
}

impl ExportTarget {
    /// The path the user chose, before the filename is resolved.
    pub fn path(&self) -> &Path {
        match self {
            Self::Directory(p) | Self::File(p) => p,
        }
    }

    /// The full path this target resolves to for `format`, given a filename stem.
    ///
    /// # Errors
    ///
    /// If the target is a directory and a file name cannot be built from `stem`
    /// — an empty name, or one that sanitises to nothing.
    pub fn resolve(&self, format: ExportFormat, stem: &str) -> Result<PathBuf> {
        match self {
            Self::Directory(dir) => {
                let name = format.file_name(stem);
                if name == format!("{}.", format.extension()) {
                    return Err(error::no_file_name());
                }
                Ok(dir.join(name))
            }
            Self::File(path) => Ok(path.clone()),
        }
    }
}

/// Build the default file name for a transcript.
///
/// The stem comes from the source media's name without its extension, so
/// `standup.m4a` exports as `standup.srt`. Falls back to `transcript` when there
/// is no usable name, which is what a transcript pasted from a chat window is.
pub fn default_stem(source_name: Option<&str>) -> String {
    let Some(name) = source_name else {
        return "transcript".to_string();
    };
    let stem = Path::new(name)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let safe = crate::writer::safe_file_name(&stem);
    if safe.is_empty() {
        "transcript".to_string()
    } else {
        safe
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_five_formats_are_the_frozen_ones() {
        // If A5's list changes, this test is where the disagreement shows up.
        let names: Vec<&str> = ExportFormat::ALL.iter().map(|f| f.as_str()).collect();
        assert_eq!(names, ["text", "markdown", "srt", "vtt", "json"]);
    }

    #[test]
    fn extensions_match_the_contract_crate_registry() {
        // `vaani_core::EXPORT_EXTENSIONS` is the file picker's list. Three lists
        // that must agree: this one, that one, and the UI's labels.
        for (ext, _) in vaani_core::EXPORT_EXTENSIONS {
            assert!(
                ExportFormat::ALL.iter().any(|f| f.extension() == *ext),
                "the contract crate advertises {ext}, which no format produces"
            );
        }
        assert_eq!(vaani_core::EXPORT_EXTENSIONS.len(), ExportFormat::ALL.len());
    }

    #[test]
    fn wire_names_round_trip() {
        for f in ExportFormat::ALL {
            assert_eq!(ExportFormat::parse(f.as_str()), Some(f));
        }
        assert_eq!(ExportFormat::parse("docx"), None);
    }

    #[test]
    fn every_format_has_a_label_a_description_and_a_media_type() {
        for f in ExportFormat::ALL {
            assert!(!f.label().is_empty());
            assert!(!f.description().is_empty());
            assert!(f.media_type().contains("utf-8"));
            assert!(!f.file_name("standup").is_empty());
        }
    }

    #[test]
    fn a_directory_target_builds_a_name_and_a_file_target_does_not() {
        let dir = ExportTarget::Directory(PathBuf::from("C:/out"));
        assert_eq!(
            dir.resolve(ExportFormat::Srt, "standup").unwrap(),
            PathBuf::from("C:/out/standup.srt")
        );
        let file = ExportTarget::File(PathBuf::from("C:/out/whatever.i_want.vtt"));
        assert_eq!(
            file.resolve(ExportFormat::Srt, "standup").unwrap(),
            PathBuf::from("C:/out/whatever.i_want.vtt"),
            "an explicit path is the user's decision, not ours"
        );
    }

    #[test]
    fn an_unusable_stem_still_produces_a_usable_name() {
        let dir = ExportTarget::Directory(PathBuf::from("out"));
        assert_eq!(
            dir.resolve(ExportFormat::Json, "...").unwrap(),
            PathBuf::from("out/transcript.json")
        );
        assert_eq!(default_stem(None), "transcript");
        assert_eq!(default_stem(Some("/media/standup.m4a")), "standup");
        assert_eq!(default_stem(Some("C:\\media\\Q3 review.mkv")), "Q3 review");
        assert_eq!(default_stem(Some("../../etc/passwd")), "passwd");
    }

    #[test]
    fn file_names_cannot_escape_their_directory() {
        let dir = ExportTarget::Directory(PathBuf::from("out"));
        let resolved = dir.resolve(ExportFormat::Text, "../../../evil").unwrap();
        assert_eq!(resolved.parent().unwrap(), Path::new("out"));
        assert!(!resolved.to_string_lossy().contains(".."));
    }
}

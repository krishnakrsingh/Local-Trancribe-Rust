//! # vaani-core
//!
//! The domain model that every other crate agrees on. Deliberately the only
//! crate with **no** heavyweight dependencies: no ONNX Runtime, no Slint, no
//! database, no async runtime. Everything else may depend on this; this
//! depends on serde, uuid, thiserror and `time`, and nothing else.
//!
//! That restriction is the point. It means a change to the transcript shape
//! can be type-checked in seconds, and it means the data model cannot quietly
//! acquire a dependency on a subsystem it is supposed to describe.
//!
//! ## Module map
//!
//! | module | owns | must not know about |
//! |---|---|---|
//! | [`error`] | the user-facing error taxonomy | anything; it is the leaf |
//! | [`transcript`] | `Transcript`/`Segment`/`Word`/`Speaker`/`Time` | ONNX, Slint, SQL |
//! | [`job`] | the job lifecycle, `Progress`, `CancelToken` | how any phase is implemented |
//! | [`media`] | `MediaInfo`, `AudioBuffer`, `Language` | which backend produced the info |
//! | [`pipeline`] | the stage traits every subsystem implements | anything concrete |
//!
//! ## Invariants that live here
//!
//! * Times are milliseconds from the start of the **source media**, never from
//!   a chunk. Rebasing happens in exactly one place: `Transcript::push_chunk`.
//! * `Segment` text is *derived* from its words, never stored, so an edit can
//!   never leave two disagreeing copies of the same string.
//! * A missing speaker is a normal state, not an error. ASR-only mode is a
//!   complete product mode.
//! * Renaming a speaker never alters transcript text.
//! * Every user-facing error carries a recovery action.
//! * Progress either has a real denominator or is explicitly indeterminate.
//!   It is never a fabricated percentage.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod job;
pub mod media;
pub mod pipeline;
pub mod transcript;

pub use error::{ErrorCode, Result, UserFacingError, VaaniError, err};
pub use job::{CancelToken, JobPhase, Progress, ProgressKind};
pub use media::{
    AudioBuffer, AudioStreamInfo, DEFAULT_ESTIMATE_RTF, Language, MediaInfo, VideoStreamInfo,
};
pub use transcript::{EngineStamp, Segment, Speaker, Time, Transcript, TranscriptMode, Word};

/// Product version. Kept in `core` so every crate, the CLI, the database and
/// the exported JSON agree without any of them hard-coding it.
///
/// Independent of model versions on purpose: an app 1.0.0 can run SraVaani
/// 2.0 with a different diarization engine, and exported transcripts record
/// the models, not the app.
pub const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Crate version of the domain model, recorded in exported JSON so a consumer
/// can tell whether a field they do not recognise is new or removed.
pub const DATA_MODEL_VERSION: u32 = 1;

/// File extensions the product advertises accepting.
///
/// Used by the file picker filters, the drag-and-drop acceptance test, and the
/// file-association registration. One list so the three can never disagree.
pub const SUPPORTED_EXTENSIONS: &[&str] = &[
    // video containers
    "mp4", "mkv", "mov", "webm", "avi", "m4v", "mpg", "mpeg", "wmv", "flv", "ts",
    // audio containers
    "wav", "mp3", "m4a", "aac", "flac", "ogg", "opus", "wma", "aiff", "amr",
];

/// Whether a path looks like media we accept, by extension.
///
/// A hint for the UI, never a hard gate: the media backend probes the actual
/// content, because a `.mp4` full of something else and a media file with an
/// unusual extension both happen in the wild.
pub fn has_supported_extension(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            SUPPORTED_EXTENSIONS
                .iter()
                .any(|s| s.eq_ignore_ascii_case(e))
        })
        .unwrap_or(false)
}

/// File extensions the product writes.
pub const EXPORT_EXTENSIONS: &[(&str, &str)] = &[
    ("txt", "Plain text"),
    ("md", "Markdown"),
    ("srt", "SubRip subtitles"),
    ("vtt", "WebVTT subtitles"),
    ("json", "JSON (full structure)"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extension_matching_is_case_insensitive() {
        assert!(has_supported_extension(std::path::Path::new("C:/a/B.MP4")));
        assert!(has_supported_extension(std::path::Path::new("meeting.mkv")));
        assert!(!has_supported_extension(std::path::Path::new("notes.txt")));
        assert!(!has_supported_extension(std::path::Path::new(
            "noextension"
        )));
    }

    #[test]
    fn export_extensions_have_unique_suffixes() {
        let mut seen: Vec<&str> = EXPORT_EXTENSIONS.iter().map(|(s, _)| *s).collect();
        seen.sort_unstable();
        let before = seen.len();
        seen.dedup();
        assert_eq!(
            seen.len(),
            before,
            "duplicate export suffix in the registry"
        );
    }
}

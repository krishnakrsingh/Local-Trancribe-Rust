//! # vaani-export
//!
//! Getting a finished transcript out of VaaniDesk and into whatever the user
//! needs next: a subtitle file for a video, a Markdown document for a wiki, a
//! JSON file for a script.
//!
//! Owned by agent **A6**. The `Cargo.toml` in this directory is frozen by the
//! architect and is part of the integration surface; the contents of the rest of
//! this crate are not.
//!
//! Read `docs/agent/PARALLEL_OWNERSHIP.md` before the first edit.
//!
//! ## The five formats
//!
//! [`ExportFormat::ALL`] is exactly `Text`, `Markdown`, `Srt`, `Vtt`, `Json` —
//! the same five, in the same order, as the UI's frozen `vaani_ui::ExportFormat`,
//! and the same extensions as `vaani_core::EXPORT_EXTENSIONS`. Three lists, one
//! set of names; a test asserts the agreement. A sixth format is a product
//! decision that has to be made in all three places, so this crate does not
//! invent one.
//!
//! ## The five properties
//!
//! 1. **Valid output.** SRT and VTT are produced by one cue builder that
//!    enforces the rules both formats share: at most two lines per cue, no
//!    overlaps, no zero-length cues, no mid-word breaks. See [`cues`] for the
//!    long-segment rule and why it is "split into more cues" rather than
//!    "truncate" or "one long line".
//! 2. **JSON round-trips.** The exported transcript is `vaani_core`'s own
//!    `Serialize`, with metadata *around* it, so reading the file back gives an
//!    equal `Transcript`. It cannot drift from the domain type because it *is*
//!    the domain type.
//! 3. **Atomic writes.** A temporary file in the same directory, then a rename.
//!    An interrupted export never leaves a truncated file where a complete one
//!    was.
//! 4. **Safe paths.** A media file called `../../evil.txt` cannot make an export
//!    write outside the directory the user chose. Two independent checks, because
//!    this is the one place where a bug is a security bug.
//! 5. **Nothing is overwritten silently.** An existing file is replaced only
//!    when the caller says so.
//!
//! ## Using it
//!
//! ```
//! use vaani_export::{ExportFormat, ExportOptions, ExportTarget, export_to_file};
//! use vaani_storage::{fixtures, testing::TempDir};
//!
//! // A temporary directory, because the doctest runs in the crate folder and an
//! // export that refuses to overwrite would fail on the second run.
//! let dir = TempDir::new("doctest");
//! let transcript = fixtures::single_word();
//! let options = ExportOptions::default().with_source_name("standup.m4a");
//! let target = ExportTarget::Directory(dir.path().to_path_buf());
//! let written = export_to_file(&transcript, ExportFormat::Srt, &target, &options)?;
//! println!("{} bytes at {}", written.bytes, written.path.display());
//! # Ok::<(), vaani_core::VaaniError>(())
//! ```
//!
//! [`render`] is the pure half, with no filesystem and no clock, which is what
//! makes "export twice, get the same bytes" a testable property.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod cues;
pub mod error;
pub mod format;
pub mod options;
pub mod render;
pub mod verify;
pub mod writer;

use std::path::PathBuf;

use vaani_core::{Result, Transcript};

pub use cues::{Cue, CueOptions};
pub use format::{ExportFormat, ExportTarget, default_stem};
pub use options::ExportOptions;
pub use render::{JsonExport, read_json_export, render};
pub use writer::{safe_file_name, write_atomic};

/// Crate version, from the workspace. Recorded in exported JSON so a support
/// conversation starts with a version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// What one export produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportedFile {
    /// Where it was written.
    pub path: PathBuf,
    /// How many bytes, as handed to the file system.
    pub bytes: u64,
    /// The format it was rendered in.
    pub format: ExportFormat,
    /// The cues it contains, for the subtitle formats; 0 for the others. The
    /// pipeline's log line wants this, and computing it again would mean
    /// building the cue list twice.
    pub cues: usize,
}

/// Export a transcript to a file.
///
/// The whole operation in one call, because the two halves cannot be made safe
/// separately: rendering then writing lets a caller forget the atomic write, and
/// writing then rendering lets a caller leave a truncated file behind.
///
/// # Errors
///
/// If the transcript cannot be rendered, if the target resolves to a path
/// outside the directory the user chose, if the file exists and
/// [`ExportOptions::overwrite`] is false, or if the write fails.
pub fn export_to_file(
    transcript: &Transcript,
    format: ExportFormat,
    target: &ExportTarget,
    options: &ExportOptions,
) -> Result<ExportedFile> {
    let path = target.resolve(format, &options.file_stem())?;
    if let ExportTarget::Directory(dir) = target {
        writer::assert_inside(dir, &path)?;
    }
    let contents = render(transcript, format, options)?;
    let cues = if format.is_cue_based() {
        cues::build_cues(transcript, &options.cue_options()).len()
    } else {
        0
    };
    let bytes = writer::write_atomic(&path, &contents, options.overwrite)?;
    Ok(ExportedFile {
        path,
        bytes,
        format,
        cues,
    })
}

/// Export several formats into one directory, in the order given.
///
/// For the "Export to…" button's "all formats" option. Stops at the first
/// failure and reports it, rather than leaving a half-finished set that the
/// user has to work out which files are missing.
///
/// # Errors
///
/// The first error any individual export returns.
pub fn export_all(
    transcript: &Transcript,
    formats: &[ExportFormat],
    directory: &std::path::Path,
    options: &ExportOptions,
) -> Result<Vec<ExportedFile>> {
    let mut out = Vec::with_capacity(formats.len());
    for f in formats {
        out.push(export_to_file(
            transcript,
            *f,
            &ExportTarget::Directory(directory.to_path_buf()),
            options,
        )?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vaani_storage::fixtures;
    use vaani_storage::testing::TempDir;

    #[test]
    fn exporting_to_a_directory_uses_the_source_name() {
        let dir = TempDir::new("export-dir");
        let t = fixtures::single_word();
        let o = ExportOptions::default().with_source_name("standup.m4a");
        for f in ExportFormat::ALL {
            let written = export_to_file(
                &t,
                f,
                &ExportTarget::Directory(dir.path().to_path_buf()),
                &o,
            )
            .unwrap();
            assert_eq!(
                written.path.file_name().unwrap().to_string_lossy(),
                format!("standup.{}", f.extension())
            );
            assert!(written.bytes > 0, "{f} wrote nothing");
            assert_eq!(written.format, f);
        }
        // Five files, one per format, and nothing else in the directory.
        let mut names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                "standup.json",
                "standup.md",
                "standup.srt",
                "standup.txt",
                "standup.vtt"
            ]
        );
    }

    #[test]
    fn exporting_twice_produces_the_same_bytes() {
        let dir = TempDir::new("export-idempotent");
        let t = fixtures::transcript(5, 12);
        let o = ExportOptions::default().with_source_name("meeting.wav");
        let mut first = std::collections::BTreeMap::new();
        for f in ExportFormat::ALL {
            let target = ExportTarget::Directory(dir.path().to_path_buf());
            let w = export_to_file(&t, f, &target, &o).unwrap();
            first.insert(f, std::fs::read(&w.path).unwrap());
        }
        for f in ExportFormat::ALL {
            let target = ExportTarget::Directory(dir.path().to_path_buf());
            let w = export_to_file(&t, f, &target, &o.clone().overwrite_or_default()).unwrap();
            assert_eq!(first[&f], std::fs::read(&w.path).unwrap(), "{f} differs");
        }
    }

    impl ExportOptions {
        fn overwrite_or_default(mut self) -> Self {
            if self.overwrite {
                self
            } else {
                self.overwrite = true;
                self
            }
        }
    }

    #[test]
    fn a_crafted_name_cannot_escape_the_target_directory() {
        let dir = TempDir::new("export-escape");
        let outside = TempDir::new("export-outside");
        let t = fixtures::single_word();
        // The name comes from the *media file*, which a user can call anything.
        let o = ExportOptions::default().with_source_name("../../evil.txt");
        let target = ExportTarget::Directory(dir.path().to_path_buf());
        let written = export_to_file(&t, ExportFormat::Text, &target, &o).unwrap();
        assert_eq!(written.path.parent().unwrap(), dir.path());
        assert!(!outside.path().join("evil.txt").exists());
    }

    #[test]
    fn an_existing_file_is_not_clobbered_by_default() {
        let dir = TempDir::new("export-clobber");
        let t = fixtures::single_word();
        let o = ExportOptions::default().with_source_name("a.wav");
        let target = ExportTarget::Directory(dir.path().to_path_buf());
        export_to_file(&t, ExportFormat::Text, &target, &o).unwrap();
        let path = dir.join("a.txt");
        std::fs::write(&path, "mine").unwrap();
        let e = export_to_file(&t, ExportFormat::Text, &target, &o).unwrap_err();
        assert_eq!(e.code(), vaani_core::ErrorCode::ExportFailed);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "mine");
    }

    #[test]
    fn export_all_writes_every_format_and_stops_at_the_first_failure() {
        let dir = TempDir::new("export-all");
        let t = fixtures::single_word();
        let o = ExportOptions::default().with_source_name("a.wav");
        let files = export_all(&t, &ExportFormat::ALL, dir.path(), &o).unwrap();
        assert_eq!(files.len(), 5);
        for f in &files {
            assert!(f.path.exists(), "{} is missing", f.path.display());
        }
        // A second run without overwrite stops at the first existing file and
        // says so, rather than replacing four of the five and then failing.
        let e = export_all(&t, &ExportFormat::ALL, dir.path(), &o).unwrap_err();
        assert!(e.user.detail.contains("already there"), "{e}");
    }

    #[test]
    fn subtitle_exports_report_their_cue_count() {
        let dir = TempDir::new("export-cues");
        let t = fixtures::transcript(3, 20);
        let o = ExportOptions::default().with_source_name("a.wav");
        let target = ExportTarget::Directory(dir.path().to_path_buf());
        let srt = export_to_file(&t, ExportFormat::Srt, &target, &o).unwrap();
        let vtt = export_to_file(&t, ExportFormat::Vtt, &target, &o).unwrap();
        let txt = export_to_file(&t, ExportFormat::Text, &target, &o).unwrap();
        assert_eq!(srt.cues, vtt.cues);
        assert!(srt.cues > 0);
        assert_eq!(txt.cues, 0, "a text export has no cues to count");
    }
}

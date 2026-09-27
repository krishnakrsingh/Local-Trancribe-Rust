//! Export failures, in the application's error taxonomy.
//!
//! Same rule as everywhere else in this codebase: no raw `std::io` text, no
//! `serde_json` message and no path the user did not choose reaches the UI. The
//! one thing an export failure must always say is *which* file it could not
//! write, because the user's next action is always about that file.

use std::fmt::Display;
use std::path::Path;

use vaani_core::{ErrorCode, UserFacingError, VaaniError, err};

/// The name we are willing to show the user.
fn display_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "that location".to_string())
}

/// A write failed. The path is named, the cause is kept for the log.
pub(crate) fn write_failed(path: &Path, cause: impl Display) -> VaaniError {
    VaaniError::new(
        UserFacingError::new(
            ErrorCode::ExportFailed,
            "Could not write the export.",
            format!("The file {} could not be written.", display_name(path)),
        )
        .with_recovery(
            "Check that the folder exists, that the file is not open in another \
             program, and that there is free disk space. Then try again.",
        ),
    )
    .with_cause(cause)
}

/// The target already exists and overwriting was not requested.
pub(crate) fn already_exists(path: &Path) -> VaaniError {
    VaaniError::new(
        UserFacingError::new(
            ErrorCode::ExportFailed,
            "That file already exists.",
            format!("{} is already there.", display_name(path)),
        )
        .with_recovery("Choose a different name, or confirm replacing the existing file."),
    )
}

/// The file name could not be turned into something safe to write.
pub(crate) fn no_file_name() -> VaaniError {
    VaaniError::new(
        UserFacingError::new(
            ErrorCode::ExportFailed,
            "No file name for the export.",
            "There is no file name to write this transcript to.",
        )
        .with_recovery("Choose where to save the export, then try again."),
    )
}

/// The target would have escaped the directory the user chose.
pub(crate) fn outside_target(name: &str) -> VaaniError {
    VaaniError::new(
        UserFacingError::new(
            ErrorCode::ExportFailed,
            "Unsafe file name.",
            format!(
                "The name \"{name}\" would have written the export outside the \
                 folder you chose, so it was not written."
            ),
        )
        .with_recovery("Choose a name without slashes or dots, then try again."),
    )
}

/// Reading back a JSON export that this crate wrote failed.
pub(crate) fn bad_json(cause: impl Display) -> VaaniError {
    err::export_failed()
        .with_cause(format!("reading an exported JSON transcript: {cause}"))
        .with_recovery(
            "The file is not a VaaniDesk JSON export, or it was edited by hand. \
             Export it again from VaaniDesk.",
        )
}

/// The transcript cannot produce a valid file in this format.
///
/// Only reachable for input the contract crate should never have produced, so
/// it is reported rather than papered over.
pub(crate) fn invalid_transcript(what: &str) -> VaaniError {
    err::export_failed()
        .with_cause(format!("the transcript cannot be exported: {what}"))
        .with_recovery("Please report this: it means the transcript is malformed.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_write_failure_names_the_file_and_offers_a_recovery() {
        let e = write_failed(
            Path::new("C:/secret/place/standup.srt"),
            std::io::Error::other("disk full"),
        );
        assert_eq!(e.code(), ErrorCode::ExportFailed);
        assert!(e.user.detail.contains("standup.srt"));
        assert!(!e.user.detail.contains("secret"), "leaked a directory");
        assert!(e.user.recovery.is_some());
        assert!(e.debug_chain().contains("disk full"));
    }

    #[test]
    fn an_existing_file_is_its_own_error_with_its_own_recovery() {
        let e = already_exists(Path::new("/home/k/standup.vtt"));
        assert_eq!(e.code(), ErrorCode::ExportFailed);
        assert!(e.user.detail.contains("standup.vtt"));
        assert!(e.user.recovery.unwrap().contains("different name"));
    }

    #[test]
    fn an_unsafe_name_says_what_it_refused() {
        let e = outside_target("../../evil.txt");
        assert!(e.user.detail.contains("outside the folder"));
        assert!(e.user.recovery.is_some());
    }

    #[test]
    fn a_bare_path_still_produces_a_readable_sentence() {
        let e = write_failed(Path::new("/"), "boom");
        assert!(e.user.detail.contains("that location"));
    }
}

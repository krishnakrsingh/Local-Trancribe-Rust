//! Mapping SQLite failures onto the application's error taxonomy.
//!
//! # Why this module exists
//!
//! `vaani_core` owns the rule that no raw Rust text ever reaches the user and
//! that every user-facing error carries a recovery action. `rusqlite`'s
//! `Error` is exactly the kind of thing that must not be rendered: "database
//! disk image is malformed" is a developer's sentence, not a user's. So every
//! `rusqlite::Error` that leaves this crate is translated here into a
//! [`VaaniError`] with a reviewed message, and the original is preserved as the
//! `source` chain for the log.
//!
//! Three failures get a *specific* message rather than the generic one,
//! because each needs a different action from the user:
//!
//! * **corrupt / not a database** — the file is not ours or is damaged. The
//!   user must restore or delete it. Continuing would silently lose data.
//! * **busy / locked** — another process (usually the CLI while the app is
//!   open, or two instances) holds the write lock. The user should close the
//!   other one; retrying is correct.
//! * **anything else** — generic, with the diagnostics in the cause chain.
//!
//! These constructors are public because [`crate::Store::with_read`] hands out a
//! `&rusqlite::Connection`: a caller writing their own SQL needs a way to turn
//! a `rusqlite::Error` into the application's taxonomy, and the alternative
//! would be for every such caller to invent its own user-facing text.
//!
//! # The detail/cause split
//!
//! Following `docs/agent/PARALLEL_OWNERSHIP.md` and the contract crate: the
//! user-facing `detail` names only the *file name* of the database, never the
//! full path, because a path the user did not choose must not appear in the UI.
//! Everything else — SQLite's own message, the SQL that failed — goes into the
//! cause.

use std::fmt::Display;
use std::path::Path;

use vaani_core::{ErrorCode, UserFacingError, VaaniError, err};

/// The name we are willing to show the user for a database file.
fn display_file(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "the data folder".to_string())
}

/// A database failure with no more specific classification.
pub fn generic(cause: impl Display) -> VaaniError {
    err::storage_failed().with_cause(cause).with_recovery(
        "Restart VaaniDesk and try again. If it keeps happening, check that \
             the VaaniDesk data folder is not read-only or out of disk space.",
    )
}

/// The file is not a usable database. Never retry silently: the user's data is
/// at stake and the app must not overwrite it on the next open.
pub fn corrupt(path: &Path, cause: impl Display) -> VaaniError {
    VaaniError::new(
        UserFacingError::new(
            ErrorCode::StorageFailed,
            "Local data is damaged.",
            format!(
                "The file {} could not be read as a VaaniDesk database. It is \
                 either damaged, or it belongs to a different program.",
                display_file(path)
            ),
        )
        .with_recovery(
            "Do not delete it yet. Copy it somewhere safe, then restore an \
             earlier copy of the VaaniDesk data folder. Transcripts you have \
             already exported are not affected.",
        ),
    )
    .with_cause(cause)
}

/// Another process holds the write lock. Retrying is the correct user action.
pub fn busy(path: &Path, cause: impl Display) -> VaaniError {
    VaaniError::new(
        UserFacingError::new(
            ErrorCode::StorageFailed,
            "Storage is busy.",
            format!(
                "{} is in use by another copy of VaaniDesk, so this change could \
                 not be saved yet.",
                display_file(path)
            ),
        )
        .with_recovery("Close any other VaaniDesk window (or command line tool) and try again."),
    )
    .with_cause(cause)
}

/// Classify a `rusqlite` error and wrap it.
///
/// `path` is only used to name the file in the user-facing text.
pub fn from_sqlite(e: rusqlite::Error, path: &Path) -> VaaniError {
    if let rusqlite::Error::SqliteFailure(ffi, msg) = &e {
        match ffi.code {
            // SQLITE_CORRUPT and SQLITE_NOTADB. NOTADB is the common one: the
            // user pointed the app at a folder that is not a database, or the
            // file is truncated to fewer than 100 bytes.
            rusqlite::ffi::ErrorCode::DatabaseCorrupt | rusqlite::ffi::ErrorCode::NotADatabase => {
                return corrupt(path, msg.as_deref().unwrap_or("SQLite reported corruption"));
            }
            // SQLITE_BUSY / SQLITE_LOCKED. With `busy_timeout` set these should
            // be rare; they mean a *second process* is writing.
            rusqlite::ffi::ErrorCode::DatabaseBusy | rusqlite::ffi::ErrorCode::DatabaseLocked => {
                return busy(path, msg.as_deref().unwrap_or("SQLite reported a lock"));
            }
            _ => {}
        }
    }
    generic(e)
}

/// Wrap a `std::io` error. Used for the file-level operations (hashing the
/// input media, sizing files on disk) that sit next to the database.
pub fn from_io(what: &str, e: std::io::Error) -> VaaniError {
    err::storage_failed()
        .with_cause(format!("{what}: {e}"))
        .with_recovery("Check that the file still exists and that VaaniDesk can read it.")
}

/// A storage failure that is a bug rather than a user situation: a transcript
/// for a job that was never saved, a row that does not parse, a setting holding
/// the wrong shape. The message is a diagnostic and belongs in the cause chain.
pub fn err_internal(message: impl Display) -> VaaniError {
    err::storage_failed()
        .with_cause(message)
        .with_recovery("Restart VaaniDesk. If it keeps happening, please report this.")
}

/// A `rusqlite::Error` that stands for "this row or value is not what the
/// schema says it must be", tagged `SQLITE_CORRUPT` so it is classified as data
/// damage rather than as a bug or a lock.
///
/// Used when a *value* is unreadable — an unknown job phase, a `mode` this build
/// does not recognise, malformed JSON in a column. The distinction matters:
/// these are the failures that happen when a user opens a database written by a
/// different build, and they deserve "Local data is damaged" rather than a
/// stack trace.
pub fn sqlite_corrupt(message: impl Into<String>) -> rusqlite::Error {
    rusqlite::Error::SqliteFailure(
        rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CORRUPT),
        Some(message.into()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vaani_core::ErrorCode as Code;

    #[test]
    fn corruption_is_reported_as_such_and_never_leaks_sql() {
        let e = rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_NOTADB),
            Some("file is not a database".to_string()),
        );
        let v = from_sqlite(e, Path::new("C:/secret/place/vaani.db"));
        assert_eq!(v.code(), Code::StorageFailed);
        assert_eq!(v.user.title, "Local data is damaged.");
        // The full path never reaches the UI, only the file name.
        assert!(!v.user.detail.contains("secret"));
        assert!(v.user.detail.contains("vaani.db"));
        // ...but the diagnostic survives for the log.
        assert!(v.debug_chain().contains("file is not a database"));
        assert!(v.user.recovery.is_some());
    }

    #[test]
    fn corruption_code_is_mapped_too() {
        let e = rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CORRUPT),
            None,
        );
        let v = from_sqlite(e, Path::new("vaani.db"));
        assert_eq!(v.user.title, "Local data is damaged.");
        assert!(v.debug_chain().contains("corrupt"));
    }

    #[test]
    fn busy_gets_its_own_recovery() {
        let e = rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY),
            Some("database is locked".to_string()),
        );
        let v = from_sqlite(e, Path::new("/var/lib/vaani/vaani.db"));
        assert_eq!(v.user.title, "Storage is busy.");
        assert!(v.user.recovery.unwrap().contains("Close any other"));
    }

    #[test]
    fn unknown_sqlite_errors_fall_back_to_the_generic_message() {
        let e = rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CONSTRAINT),
            Some("UNIQUE constraint failed: jobs.id".to_string()),
        );
        let v = from_sqlite(e, Path::new("vaani.db"));
        assert_eq!(v.user.title, "Could not access local storage.");
        assert!(v.debug_chain().contains("UNIQUE constraint"));
    }

    #[test]
    fn a_bare_path_falls_back_to_a_readable_phrase() {
        assert_eq!(display_file(Path::new("")), "the data folder");
        // A drive root has no file name either; a trailing separator does not
        // make one (`C:/x/y/` is the file `y`).
        assert_eq!(display_file(Path::new("C:/")), "the data folder");
        assert_eq!(display_file(Path::new("C:/x/y/")), "y");
    }
}

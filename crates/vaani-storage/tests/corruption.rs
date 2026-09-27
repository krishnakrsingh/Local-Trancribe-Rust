//! What happens when the database file is damaged.
//!
//! The rule this file enforces: a corrupt database produces a **clear error**.
//! Not a panic (a user sees a crash dialog and blames us), not an empty job list
//! (the user believes their transcripts are gone and starts re-transcribing
//! three hours of meetings), and not a silently recreated database (that is data
//! loss with a cheerful message).
//!
//! Each test damages a *real* database built by the real code, so it exercises
//! SQLite's actual failure modes rather than a mock.

use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};

use vaani_core::ErrorCode;
use vaani_storage::{JobRecord, Store, fixtures, testing::TempDir};

/// A healthy database with one completed job and one transcript in it.
fn healthy(dir: &TempDir) -> std::path::PathBuf {
    let path = dir.join("v.db");
    let store = Store::open(&path).unwrap();
    let mut job = JobRecord::new("j", dir.join("standup.m4a"));
    job.phase = vaani_core::JobPhase::Complete;
    store.upsert_job(&job).unwrap();
    store
        .replace_transcript("j", &fixtures::transcript(3, 20))
        .unwrap();
    store.close().unwrap();
    drop(store);
    // Fold the write-ahead log in, so the damage below hits the one file a user
    // would actually back up.
    let _ = std::fs::remove_file(format!("{}-wal", path.display()));
    path
}

/// Assert that opening failed with a *clear* error rather than a panic.
fn assert_clear_failure(path: &std::path::Path, what: &str) {
    match Store::open(path) {
        Ok(_) => panic!("{what}: opening a damaged database succeeded"),
        Err(e) => {
            assert_eq!(
                e.code(),
                ErrorCode::StorageFailed,
                "{what}: wrong error code"
            );
            // The user is told what happened and what to do, and is not shown
            // SQLite's own words.
            assert!(!e.user.title.is_empty(), "{what}: no title");
            assert!(!e.user.detail.is_empty(), "{what}: no detail");
            assert!(
                e.user.recovery.is_some(),
                "{what}: a storage error must offer a recovery action, got {:?}",
                e.user
            );
            assert!(
                !e.user.detail.contains("SQLITE"),
                "{what}: raw SQLite text leaked into the UI: {}",
                e.user.detail
            );
            // The diagnostic is still available for the log.
            assert!(!e.debug_chain().is_empty());
        }
    }
}

#[test]
fn a_truncated_database_is_reported_not_panicked_on() {
    let dir = TempDir::new("corrupt-truncate");
    let path = healthy(&dir);
    // Chop the file in half. The header still claims a plausible size, so this
    // is the case that produces SQLITE_CORRUPT rather than SQLITE_NOTADB.
    let len = std::fs::metadata(&path).unwrap().len();
    let file = OpenOptions::new().write(true).open(&path).unwrap();
    file.set_len(len / 2).unwrap();
    drop(file);
    assert_clear_failure(&path, "truncated to half");
}

#[test]
fn a_file_that_is_not_a_database_is_reported_clearly() {
    let dir = TempDir::new("corrupt-notadb");
    let path = dir.join("v.db");
    // 4 KB of text: long enough to have a plausible header size, and it is
    // exactly what a user gets when they point the app at a text file.
    std::fs::write(
        &path,
        "this is not a database, it is a shopping list\n".repeat(80),
    )
    .unwrap();
    assert_clear_failure(&path, "not a database");
}

#[test]
fn a_tiny_file_is_reported_clearly() {
    let dir = TempDir::new("corrupt-tiny");
    let path = dir.join("v.db");
    std::fs::write(&path, b"SQLite format 3").unwrap();
    assert_clear_failure(&path, "shorter than a database header");
}

#[test]
fn an_empty_file_is_treated_as_a_new_database() {
    // Zero bytes is *not* corruption: it is what `CREATE TABLE` leaves behind
    // after a crash on a brand-new install, and treating it as damage would
    // stop a first-time user from starting.
    let dir = TempDir::new("corrupt-empty");
    let path = dir.join("v.db");
    std::fs::write(&path, b"").unwrap();
    let store = Store::open(&path).unwrap();
    assert_eq!(
        store.schema_version().unwrap(),
        vaani_storage::migrations::latest_version()
    );
}

#[test]
fn scribbling_over_the_header_is_reported_clearly() {
    let dir = TempDir::new("corrupt-header");
    let path = healthy(&dir);
    // Overwrite the 100-byte header with garbage, keeping the length. This is
    // what a failing disk or a bad restore produces.
    let mut file = OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::Start(0)).unwrap();
    file.write_all(&[0xA5u8; 100]).unwrap();
    file.sync_all().unwrap();
    drop(file);
    assert_clear_failure(&path, "scribbled header");
}

#[test]
fn scribbling_over_the_middle_is_caught_by_the_integrity_check() {
    // Damage in the middle of the file can survive the open, so the guarantee
    // there is `integrity_check` reporting it rather than the open failing.
    let dir = TempDir::new("corrupt-middle");
    let path = healthy(&dir);
    let len = std::fs::metadata(&path).unwrap().len();
    let mut file = OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::Start(len / 2)).unwrap();
    file.write_all(&[0x5Au8; 4096]).unwrap();
    file.sync_all().unwrap();
    drop(file);

    match Store::open(&path) {
        Ok(store) => {
            // If it opened, the check must tell the truth about it — either it
            // fails with a clear error, or it says the file is not intact.
            // "Opens fine, check says ok" on a scribbled file would be the bug.
            match store.integrity_check() {
                Ok(report) => assert!(
                    !report.ok,
                    "a scribbled database reported a clean integrity check"
                ),
                Err(e) => {
                    assert_eq!(e.code(), ErrorCode::StorageFailed);
                    assert!(e.user.recovery.is_some());
                }
            }
        }
        Err(e) => {
            assert_eq!(e.code(), ErrorCode::StorageFailed);
            assert!(e.user.recovery.is_some());
        }
    }
}

#[test]
fn a_healthy_database_passes_its_own_integrity_check() {
    // The other half of the previous test: a check that always fails is as
    // useless as no check.
    let dir = TempDir::new("corrupt-healthy");
    let path = healthy(&dir);
    let store = Store::open(&path).unwrap();
    let report = store.integrity_check().unwrap();
    assert!(report.ok, "{:?}", report.messages);
    assert_eq!(report.messages, vec!["ok".to_string()]);
}

#[test]
fn a_damaged_database_is_never_silently_recreated() {
    // The worst possible behaviour: "could not read it" followed by a fresh
    // empty database, which looks to the user like their history vanished.
    let dir = TempDir::new("corrupt-no-clobber");
    let path = healthy(&dir);
    let before = std::fs::read(&path).unwrap();
    let len = before.len() as u64;
    {
        let file = OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(len / 3).unwrap();
    }
    assert!(Store::open(&path).is_err());
    let after = std::fs::read(&path).unwrap();
    assert_eq!(
        after.len() as u64,
        len / 3,
        "the damaged file must be left exactly as it was found"
    );
    assert_eq!(
        &after[..64.min(after.len())],
        &before[..64.min(before.len())]
    );
}

#[test]
fn a_directory_in_place_of_the_database_is_reported_clearly() {
    let dir = TempDir::new("corrupt-directory");
    let path = dir.join("v.db");
    std::fs::create_dir_all(&path).unwrap();
    let e = Store::open(&path).unwrap_err();
    assert_eq!(e.code(), ErrorCode::StorageFailed);
    assert!(e.user.recovery.is_some());
}

//! The upgrade path, tested against a database built by the *previous* schema.
//!
//! This is the test that matters most for a desktop app, because it is the only
//! one that runs against a file some real user has. A fresh-install test proves
//! nothing about upgrades: it creates the schema the *current* build wants.
//! Here the database is created at v1, filled with the rows a v1 build would
//! have written, and only then opened with the current build, so the v2
//! migration runs against real data.
//!
//! # Why the rows are written with SQL rather than with the store's own API
//!
//! [`Store::replace_transcript`] targets the *current* schema, which includes
//! the two columns migration 2 adds, so it cannot populate a v1 database. In
//! the field it never has to: a v1 database's contents were written by the v1
//! build. So this test writes them the way that build did — same tables, same
//! columns, same values — using the v1 DDL in
//! [`vaani_storage::migrations::V1_INITIAL`] as the specification. Anything less
//! would be testing a database this build created, which is the fresh-install
//! test again.

use rusqlite::{Connection, params};
use vaani_core::{EngineStamp, JobPhase, Transcript};
use vaani_storage::{JobRecord, Store, fixtures, migrations, testing::TempDir};

/// Build a database at schema `version` and write one job plus one transcript
/// into it exactly as a build of that schema would have.
fn seed(dir: &TempDir, version: u32) -> (std::path::PathBuf, JobRecord, Transcript) {
    let path = dir.join("v.db");
    let job = {
        let mut j = JobRecord::new("job-1", dir.join("standup.m4a"));
        j.phase = JobPhase::Complete;
        j.model = Some("sravaani-1.0".into());
        j.device = Some("cuda".into());
        j.language = Some("hi".into());
        j.duration_ms = Some(783_000);
        j = j.with_content_hash("a".repeat(64));
        j
    };
    let mut transcript = fixtures::transcript(0xC0FFEE, 40);
    transcript.engine = EngineStamp::asr("sravaani", "1.0");
    transcript.mode = vaani_core::TranscriptMode::Raw;
    // The v1 build's write below stores a rename for cluster 0, so the expected
    // transcript carries it too. `speaker_mut` is the contract crate's own
    // rename entry point, so this is the same value the UI would have written.
    if let Some(s) = transcript.speakers.iter_mut().find(|s| s.index == 0) {
        s.name = Some("Krishna".into());
    }

    let mut conn = Connection::open(&path).unwrap();
    migrations::migrate(&mut conn, &path, version).unwrap();
    write_v1_job(&conn, &job);
    if version == 1 {
        write_v1_transcript(&conn, "job-1", &transcript);
    }
    drop(conn);
    (path, job, transcript)
}

/// The v1 build's `INSERT INTO jobs`, column for column.
fn write_v1_job(conn: &Connection, job: &JobRecord) {
    conn.execute(
        "INSERT INTO jobs (id, input_path, input_name, container, media_json, duration_ms, phase,
                           created_at_ms, started_at_ms, finished_at_ms, error_code, error_message,
                           model, device, language, word_count, segment_count, speaker_count,
                           content_hash)
         VALUES (?1,?2,?3,?4,'{}',?5,?6,?7,?8,?9,NULL,NULL,?10,?11,?12,0,0,0,?13)",
        params![
            job.id,
            job.input_path,
            job.input_name,
            job.container,
            job.duration_ms,
            job.phase.as_str(),
            job.created_at_ms,
            job.started_at_ms,
            job.finished_at_ms,
            job.model,
            job.device,
            job.language,
            job.content_hash,
        ],
    )
    .unwrap();
}

/// The v1 build's transcript write: no `segment_count`, no `word_count`.
fn write_v1_transcript(conn: &Connection, job_id: &str, t: &Transcript) {
    let tx = conn
        .unchecked_transaction()
        .expect("the v1 build wrote the transcript in one transaction too");
    {
        let mut seg = tx
            .prepare(
                "INSERT INTO segments
                    (id, job_id, ordinal, start_ms, end_ms, speaker_index, text, language, confidence)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
            )
            .unwrap();
        for (i, s) in t.segments.iter().enumerate() {
            seg.execute(params![
                s.id,
                job_id,
                i as i64,
                s.start.as_millis(),
                s.end.as_millis(),
                s.speaker.map(|x| x as i64),
                s.text(),
                s.language,
                s.confidence.map(f64::from),
            ])
            .unwrap();
        }
    }
    {
        let mut w = tx
            .prepare(
                "INSERT INTO words
                    (job_id, segment_id, ordinal, text, normalized, start_ms, end_ms, confidence)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
            )
            .unwrap();
        for s in &t.segments {
            for (i, word) in s.words.iter().enumerate() {
                w.execute(params![
                    job_id,
                    s.id,
                    i as i64,
                    word.text,
                    word.normalized,
                    word.start.as_millis(),
                    word.end.as_millis(),
                    word.confidence.map(f64::from),
                ])
                .unwrap();
            }
        }
    }
    {
        let mut sp = tx
            .prepare(
                "INSERT INTO speakers
                    (job_id, cluster_index, ordinal, display_name, custom_name, talk_time_ms, share)
                 VALUES (?1,?2,?3,?4,?5,?6,?7)",
            )
            .unwrap();
        for (i, s) in t.speakers.iter().enumerate() {
            sp.execute(params![
                job_id,
                s.index as i64,
                i as i64,
                s.display_name(),
                // Whatever the caller renamed, which is what the v1 build stored.
                s.name,
                1_000i64,
                s.share.map(f64::from),
            ])
            .unwrap();
        }
    }
    {
        // The v1 build kept the search index in the same transaction as the
        // transcript, so an upgraded database is searchable. A fixture that
        // skipped this would make the test assert something the old build never
        // promised.
        let mut fts = tx
            .prepare("INSERT INTO segment_fts (job_id, segment_id, text) VALUES (?1, ?2, ?3)")
            .unwrap();
        for s in &t.segments {
            fts.execute(params![job_id, s.id, s.text()]).unwrap();
        }
    }
    tx.execute(
        "INSERT INTO transcripts
            (job_id, transcript_id, mode, language, engine_json, duration_ms, edited, created_at_ms)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
        params![
            job_id,
            t.id,
            "raw",
            "hi",
            serde_json::to_string(&t.engine).unwrap(),
            t.duration.map(vaani_core::MediaLength::as_millis),
            i64::from(t.edited),
            1_700_000_000_000i64,
        ],
    )
    .unwrap();
    tx.commit().unwrap();
}

fn words_of(t: &Transcript) -> usize {
    t.segments.iter().map(|s| s.words.len()).sum()
}

#[test]
fn a_v1_database_upgrades_without_losing_anything() {
    let dir = TempDir::new("upgrade");
    let (path, job, transcript) = seed(&dir, 1);
    assert!(
        migrations::latest_version() >= 2,
        "there must be a migration beyond v1 for this test to mean anything"
    );

    // --- Open with the current build: v2 must now run. ----------------------
    let store = Store::open(&path).unwrap();
    assert_eq!(
        store.schema_version().unwrap(),
        migrations::latest_version()
    );
    let log = store.applied_migrations().unwrap();
    assert_eq!(log.len() as u32, migrations::latest_version());
    assert_eq!(log[0].name, "initial");
    assert!(log[1].name.contains("transcript_counts"));
    assert!(log.iter().all(|m| m.applied_at_ms > 0));

    // Nothing was lost from the job.
    let after = store.load_job("job-1").unwrap().expect("the job survived");
    assert_eq!(after.input_path, job.input_path);
    assert_eq!(after.input_name, "standup.m4a");
    assert_eq!(after.phase, JobPhase::Complete);
    assert_eq!(after.model.as_deref(), Some("sravaani-1.0"));
    assert_eq!(after.device.as_deref(), Some("cuda"));
    assert_eq!(after.duration_ms, Some(783_000));
    assert_eq!(after.content_hash, "a".repeat(64));

    // Nothing was lost from the transcript, to the millisecond.
    let back = store.load_transcript("job-1").unwrap();
    fixtures::assert_transcript_eq(&transcript, &back);
    assert_eq!(words_of(&back), words_of(&transcript));

    // The rename survived, and recovery did not touch the transcript.
    assert_eq!(
        back.speakers
            .iter()
            .find(|s| s.index == 0)
            .unwrap()
            .name
            .as_deref(),
        Some("Krishna")
    );

    // **The point of the migration**: the new columns are populated from the
    // rows that already existed, not left at zero for the user to notice.
    let summary = store.transcript_summary("job-1").unwrap().unwrap();
    assert_eq!(summary.segment_count, transcript.segments.len() as i64);
    assert_eq!(summary.word_count, words_of(&transcript) as i64);
    assert_eq!(
        summary.segment_count,
        store.count_segments("job-1").unwrap()
    );
    assert_eq!(summary.word_count, store.count_words("job-1").unwrap());
    assert_eq!(summary.created_at_ms, 1_700_000_000_000);

    // The new table exists, is writable and cascades.
    store
        .record_export(&vaani_storage::ExportRecord::new(
            Some("job-1"),
            dir.join("standup.srt"),
            "srt",
            4096,
        ))
        .unwrap();
    assert_eq!(store.list_exports(Some("job-1")).unwrap().len(), 1);
    store.delete_job("job-1").unwrap();
    assert!(store.list_exports(None).unwrap().is_empty());
    assert_eq!(store.count_words("job-1").unwrap(), 0);
}

#[test]
fn the_v2_columns_exist_on_a_fresh_database_too() {
    let dir = TempDir::new("upgrade-columns");
    let store = Store::open(dir.join("v.db")).unwrap();
    let path = store.path().to_path_buf();
    let columns: Vec<String> = store
        .with_read(|conn| {
            let map = |e: rusqlite::Error| vaani_storage::error::from_sqlite(e, &path);
            let mut stmt = conn
                .prepare("PRAGMA table_info(transcripts)")
                .map_err(map)?;
            let rows = stmt.query_map([], |r| r.get::<_, String>(1)).map_err(map)?;
            let mut v = Vec::new();
            for r in rows {
                v.push(r.map_err(map)?);
            }
            Ok(v)
        })
        .unwrap();
    assert!(columns.contains(&"segment_count".to_string()));
    assert!(columns.contains(&"word_count".to_string()));
}

#[test]
fn upgrading_twice_changes_nothing_and_keeps_every_millisecond() {
    let dir = TempDir::new("upgrade-twice");
    let (path, _job, transcript) = seed(&dir, 1);

    let first = Store::open(&path).unwrap();
    let applied = first.applied_migrations().unwrap();
    let after_first = first.load_transcript("job-1").unwrap();
    fixtures::assert_transcript_eq(&transcript, &after_first);
    drop(first);

    let second = Store::open(&path).unwrap();
    assert_eq!(
        second.applied_migrations().unwrap().len(),
        applied.len(),
        "reopening must not re-run a migration"
    );
    fixtures::assert_transcript_eq(&transcript, &second.load_transcript("job-1").unwrap());
}

#[test]
fn a_migration_failure_leaves_the_previous_version_intact() {
    // The v2 migration is one transaction. If its body cannot complete, the
    // `user_version` bump must not have happened either, so the next open tries
    // again instead of believing in a half-applied schema.
    let dir = TempDir::new("upgrade-atomic");
    let (path, _job, _t) = seed(&dir, 1);
    {
        // Occupy the name migration 2 wants to create.
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch("CREATE TABLE exports (id INTEGER PRIMARY KEY, bogus TEXT)")
            .unwrap();
    }

    let err = Store::open(&path).unwrap_err();
    assert_eq!(err.code(), vaani_core::ErrorCode::StorageFailed);
    assert!(
        err.debug_chain().contains("exports"),
        "the failure should name the object that conflicted: {}",
        err.debug_chain()
    );

    let conn = Connection::open(&path).unwrap();
    let version: i64 = conn
        .query_row("PRAGMA user_version", [], |r| r.get(0))
        .unwrap();
    assert_eq!(version, 1, "a failed migration must not bump the version");
    let jobs: i64 = conn
        .query_row("SELECT COUNT(*) FROM jobs", [], |r| r.get(0))
        .unwrap();
    assert_eq!(jobs, 1, "the v1 data is untouched");
    let words: i64 = conn
        .query_row("SELECT COUNT(*) FROM words", [], |r| r.get(0))
        .unwrap();
    assert!(words > 0, "the v1 words are untouched");
}

#[test]
fn a_job_interrupted_before_the_upgrade_is_recovered_after_it() {
    // Crash recovery and migration in the same breath: a v1 database with a job
    // stuck in `ASR` must come out the other side terminal, with its data
    // intact. This is exactly the file a user who closed the app during an
    // update will have.
    let dir = TempDir::new("upgrade-recovery");
    let (path, _job, _t) = seed(&dir, 1);
    {
        let conn = Connection::open(&path).unwrap();
        conn.execute(
            "INSERT INTO jobs (id, input_path, input_name, phase, created_at_ms, content_hash)
             VALUES ('running', 'C:/x/running.wav', 'running.wav', 'ASR', 1, 'b')",
            [],
        )
        .unwrap();
    }
    let store = Store::open(&path).unwrap();
    assert_eq!(
        store.schema_version().unwrap(),
        migrations::latest_version()
    );
    assert_eq!(
        store.load_job("running").unwrap().unwrap().phase,
        JobPhase::Failed
    );
    assert_eq!(
        store.load_job("job-1").unwrap().unwrap().phase,
        JobPhase::Complete
    );
    assert!(store.load_transcript("job-1").is_ok());
}

#[test]
fn an_upgraded_database_can_be_written_again_immediately() {
    // The upgrade must leave a database the *current* build can write to, not
    // merely one it can read.
    let dir = TempDir::new("upgrade-rewrite");
    let (path, _job, _t) = seed(&dir, 1);
    let store = Store::open(&path).unwrap();
    let fresh = fixtures::transcript(99, 12);
    store.replace_transcript("job-1", &fresh).unwrap();

    // The user's rename of cluster 0 survives the rewrite: re-running
    // diarization must not make them name Speaker 1 again. That is a deliberate
    // behaviour of `replace_transcript`, so it is asserted rather than assumed.
    let mut expected = fresh.clone();
    for s in expected.speakers.iter_mut() {
        if s.index == 0 {
            s.name = Some("Krishna".into());
        }
    }
    let back = store.load_transcript("job-1").unwrap();
    fixtures::assert_transcript_eq(&expected, &back);
    let summary = store.transcript_summary("job-1").unwrap().unwrap();
    assert_eq!(summary.segment_count, fresh.segments.len() as i64);
    assert_eq!(summary.word_count, words_of(&fresh) as i64);
}

#[test]
fn the_upgraded_database_is_searchable() {
    // Migration 1 created the FTS index, so an upgraded database must return the
    // same search results as a fresh one.
    let dir = TempDir::new("upgrade-search");
    let (path, _job, transcript) = seed(&dir, 1);
    let store = Store::open(&path).unwrap();
    let needle = fixtures::any_word(&transcript);
    let results = store
        .search(&vaani_storage::SearchQuery::new(needle.clone()))
        .unwrap();
    assert!(!results.is_empty(), "{needle:?} should be findable");
    assert_eq!(results.hits[0].job_id, "job-1");
}

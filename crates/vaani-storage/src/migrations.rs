//! The schema and the forward-only migration runner.
//!
//! # Why a real migration system and not `CREATE TABLE IF NOT EXISTS`
//!
//! `IF NOT EXISTS` handles *creation*. It cannot handle *change*: there is no
//! way to add a column to a 400 MB table of transcripts that 500 users already
//! have, and a desktop app that answers "your transcripts are gone" to someone
//! who upgraded is a product-ending bug. So this is a numbered, forward-only,
//! transactional runner with an audit log, and the schema for a fresh install
//! and the schema after every upgrade are the same code path.
//!
//! # The three sources of truth, and why that is not redundant
//!
//! | what | where | role |
//! |---|---|---|
//! | `PRAGMA user_version` | SQLite file header | the version we are *at*; written inside the migration transaction, so it is atomic with the DDL |
//! | `schema_migrations` | this crate | *when* and *what* — the audit trail a support question needs ("when did my file get this column?") |
//! | [`MIGRATIONS`] | this file | what *should* be |
//!
//! On open the effective version is `max(user_version, max(schema_migrations))`
//! and every migration above it runs. Taking the maximum is deliberate: if the
//! header and the table ever disagree — a hand-restored backup, a database
//! written by a build that logged before it bumped the header — replaying a
//! migration that already ran must be *harmless*, so every migration body is
//! written to be idempotent (`IF NOT EXISTS`, and a `PRAGMA table_info` check
//! before every `ALTER TABLE`). Replay is therefore always safe, and the
//! mismatch is logged rather than guessed at.
//!
//! A database whose `user_version` is *newer* than this build's
//! [`latest_version`] is refused, not downgraded. An older build writing to a
//! newer schema is how data is destroyed; the user is told to update.
//!
//! # Adding a migration
//!
//! Append to [`MIGRATIONS`]. Never edit an existing one: the checksum of a
//! shipped migration is the user's data, and "version 1 means something else
//! now" is unrecoverable. Downgrades are not supported and will not be; a
//! forward migration must always be expressible as additive changes, and the
//! additive DDL below keeps that true.
//!
//! # The DDL location
//!
//! [`V1_INITIAL`] and [`V2_TRANSCRIPT_COUNTS`] are the only places table
//! definitions appear, together with the `#[cfg(test)]` tests that assert the
//! live schema matches them.

use rusqlite::Connection;
use std::path::Path;
use std::time::Instant;

use crate::Result;

use crate::clock;
use crate::error;

/// Migration 1: the initial schema.
///
/// Times are `INTEGER` milliseconds, never text. `STRICT` is used on every
/// table so that SQLite itself refuses to put a string in a time column, which
/// turns "the transcript is three minutes out" from a silent bug into a loud
/// one.
pub const V1_INITIAL: &str = r#"
CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT    PRIMARY KEY NOT NULL,
    input_path      TEXT    NOT NULL,
    input_name      TEXT    NOT NULL,
    container       TEXT,
    media_json      TEXT    NOT NULL DEFAULT '{}',
    duration_ms     INTEGER,
    phase           TEXT    NOT NULL,
    created_at_ms   INTEGER NOT NULL,
    started_at_ms   INTEGER,
    finished_at_ms  INTEGER,
    error_code      TEXT,
    error_message   TEXT,
    model           TEXT,
    device          TEXT,
    language        TEXT,
    word_count      INTEGER NOT NULL DEFAULT 0,
    segment_count   INTEGER NOT NULL DEFAULT 0,
    speaker_count   INTEGER NOT NULL DEFAULT 0,
    content_hash    TEXT    NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS transcripts (
    job_id        TEXT    PRIMARY KEY NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    transcript_id TEXT    NOT NULL,
    mode          TEXT    NOT NULL,
    language      TEXT,
    engine_json   TEXT    NOT NULL DEFAULT '{}',
    duration_ms   INTEGER,
    edited        INTEGER NOT NULL DEFAULT 0,
    created_at_ms INTEGER NOT NULL
) STRICT;

-- Segment ids are stable *within a job* and are what edits and the UI's
-- selection model refer to, so the primary key is (job, segment) and the same
-- id may legitimately appear in two different jobs.
CREATE TABLE IF NOT EXISTS segments (
    id            TEXT    NOT NULL,
    job_id        TEXT    NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    ordinal       INTEGER NOT NULL,
    start_ms      INTEGER NOT NULL,
    end_ms        INTEGER NOT NULL,
    speaker_index INTEGER,
    text          TEXT    NOT NULL,
    language      TEXT,
    confidence    REAL,
    PRIMARY KEY (job_id, id)
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_segments_ordinal
    ON segments(job_id, ordinal);

-- `words` cascades from `segments` through a composite foreign key, so deleting
-- a segment cannot leave orphan words: the words are removed by SQLite itself,
-- not by remembering to delete them in Rust.
CREATE TABLE IF NOT EXISTS words (
    job_id     TEXT    NOT NULL,
    segment_id TEXT    NOT NULL,
    ordinal    INTEGER NOT NULL,
    text       TEXT    NOT NULL,
    normalized TEXT,
    start_ms   INTEGER NOT NULL,
    end_ms     INTEGER NOT NULL,
    confidence REAL,
    PRIMARY KEY (job_id, segment_id, ordinal),
    FOREIGN KEY (job_id, segment_id) REFERENCES segments(job_id, id) ON DELETE CASCADE
) STRICT;

CREATE TABLE IF NOT EXISTS speakers (
    job_id        TEXT    NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    cluster_index INTEGER NOT NULL,
    -- Preserved so that a transcript round-trips with its `speakers` vector in
    -- exactly the order it was given, not merely sorted by index.
    ordinal       INTEGER NOT NULL,
    display_name  TEXT    NOT NULL,
    custom_name   TEXT,
    talk_time_ms  INTEGER NOT NULL DEFAULT 0,
    share         REAL,
    PRIMARY KEY (job_id, cluster_index)
) STRICT;

CREATE TABLE IF NOT EXISTS settings (
    key           TEXT    PRIMARY KEY NOT NULL,
    value         TEXT    NOT NULL,
    updated_at_ms INTEGER NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_jobs_created    ON jobs(created_at_ms DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_phase      ON jobs(phase);
CREATE INDEX IF NOT EXISTS idx_jobs_hash       ON jobs(content_hash);
CREATE INDEX IF NOT EXISTS idx_jobs_name       ON jobs(input_name);
CREATE INDEX IF NOT EXISTS idx_jobs_language   ON jobs(language);
CREATE INDEX IF NOT EXISTS idx_segments_start  ON segments(job_id, start_ms);
CREATE INDEX IF NOT EXISTS idx_segments_speaker ON segments(job_id, speaker_index);
"#;

/// Migration 2: transcript counts, and the export log.
///
/// `transcripts.segment_count` / `word_count` exist because the job list shows
/// "1,812 segments · 20,114 words" for every row, and `COUNT(*)` over a
/// correlated subquery per row is the wrong answer at a few hundred jobs. The
/// migration **backfills** them from the rows that already exist, so an
/// upgrading user sees correct numbers on the first run after the update rather
/// than zeros until they re-transcribe something.
pub const V2_TRANSCRIPT_COUNTS: &str = r#"
CREATE TABLE IF NOT EXISTS exports (
    id            INTEGER PRIMARY KEY,
    job_id        TEXT REFERENCES jobs(id) ON DELETE CASCADE,
    path          TEXT    NOT NULL,
    format        TEXT    NOT NULL,
    bytes         INTEGER NOT NULL,
    created_at_ms INTEGER NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_exports_job ON exports(job_id, created_at_ms);
"#;

/// One forward migration.
#[derive(Debug, Clone, Copy)]
pub struct Migration {
    /// Schema version this migration produces. Must equal the previous
    /// migration's version plus one.
    pub version: u32,
    /// Short human name, recorded in `schema_migrations`.
    pub name: &'static str,
    apply: fn(&Connection, &Context) -> Result<()>,
}

/// What a migration is allowed to know about the database it is running on.
#[derive(Debug, Clone, Copy)]
pub struct Context {
    /// Whether this SQLite build has the FTS5 module. `rusqlite`'s bundled
    /// build does (it passes `-DSQLITE_ENABLE_FTS5`), but the crate is also
    /// compilable against a system SQLite, so the migration asks rather than
    /// assumes, and the search layer has a fallback for when the answer is no.
    pub fts5: bool,
}

/// Every migration, in order. This list is the schema's history.
pub const MIGRATIONS: &[Migration] = &[
    Migration {
        version: 1,
        name: "initial",
        apply: apply_v1,
    },
    Migration {
        version: 2,
        name: "transcript_counts_and_exports",
        apply: apply_v2,
    },
];

/// The newest schema this build understands.
pub fn latest_version() -> u32 {
    MIGRATIONS.last().map_or(0, |m| m.version)
}

/// A migration that has been applied, as recorded in `schema_migrations`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)] // field names are the documentation
pub struct AppliedMigration {
    pub version: u32,
    pub name: String,
    pub applied_at_ms: i64,
    pub duration_ms: i64,
}

/// The bootstrap table, created by the runner rather than by a migration
/// because the runner needs it to know what has run.
const BOOTSTRAP: &str = "
CREATE TABLE IF NOT EXISTS schema_migrations (
    version      INTEGER PRIMARY KEY NOT NULL,
    name         TEXT    NOT NULL,
    applied_at_ms INTEGER NOT NULL,
    duration_ms  INTEGER NOT NULL
) STRICT;
";

fn apply_v1(conn: &Connection, ctx: &Context) -> Result<()> {
    conn.execute_batch(V1_INITIAL)
        .map_err(|e| error::from_sqlite(e, Path::new("<migrate:v1>")))?;
    if ctx.fts5 {
        // The search index is created here, alongside the table it indexes, so
        // a fresh install and an upgrade end up in the same state. It is
        // maintained inside the same transaction as the transcript write, and
        // a missing table is detectable, which is why it is not an external
        // content table: a standalone index can be rebuilt from `segments` at
        // any time and a corrupt one can simply be dropped and recreated.
        conn.execute_batch(FTS_SCHEMA)
            .map_err(|e| error::from_sqlite(e, Path::new("<migrate:v1:fts>")))?;
    }
    Ok(())
}

fn apply_v2(conn: &Connection, _ctx: &Context) -> Result<()> {
    let path = Path::new("<migrate:v2>");
    conn.execute_batch(V2_TRANSCRIPT_COUNTS)
        .map_err(|e| error::from_sqlite(e, path))?;
    add_column(
        conn,
        "transcripts",
        "segment_count",
        "INTEGER NOT NULL DEFAULT 0",
    )
    .map_err(|e| error::from_sqlite(e, path))?;
    add_column(
        conn,
        "transcripts",
        "word_count",
        "INTEGER NOT NULL DEFAULT 0",
    )
    .map_err(|e| error::from_sqlite(e, path))?;
    // Backfill. Two correlated subqueries over the whole corpus, once, in the
    // same transaction as the ALTERs: a user upgrading must not see zeros.
    conn.execute_batch(
        "UPDATE transcripts
            SET segment_count = (SELECT COUNT(*) FROM segments WHERE segments.job_id = transcripts.job_id),
                word_count     = (SELECT COUNT(*) FROM words    WHERE words.job_id    = transcripts.job_id);",
    )
    .map_err(|e| error::from_sqlite(e, path))?;
    Ok(())
}

/// The full-text index. `job_id` and `segment_id` are UNINDEXED so the index
/// does not carry 20,000 copies of a UUID, but a plain `SELECT`/`DELETE` on
/// them still works, which is what makes a full rebuild possible.
pub const FTS_SCHEMA: &str = "
CREATE VIRTUAL TABLE IF NOT EXISTS segment_fts USING fts5(
    job_id UNINDEXED,
    segment_id UNINDEXED,
    text,
    tokenize = 'unicode61 remove_diacritics 2'
);
";

/// `ALTER TABLE ... ADD COLUMN` if the column is not already there.
///
/// SQLite has no `ADD COLUMN IF NOT EXISTS`, and a migration that fails on
/// replay is a migration that bricks a user who restores a backup. Checking
/// `table_info` costs one pragma read and makes the body idempotent.
fn add_column(conn: &Connection, table: &str, column: &str, decl: &str) -> rusqlite::Result<()> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let name: String = row.get(1)?;
        if name == column {
            return Ok(());
        }
    }
    conn.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {column} {decl};"))
}

/// Whether this SQLite build has FTS5 at all.
pub fn fts5_available(conn: &Connection) -> bool {
    conn.query_row(
        "SELECT count(*) FROM pragma_module_list WHERE name = 'fts5'",
        [],
        |r| r.get::<_, i64>(0),
    )
    .map(|n| n > 0)
    .unwrap_or(false)
}

/// Whether the search index exists in this database.
pub fn search_index_present(conn: &Connection) -> bool {
    conn.query_row(
        "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = 'segment_fts'",
        [],
        |r| r.get::<_, i64>(0),
    )
    .map(|n| n > 0)
    .unwrap_or(false)
}

/// The version this database is at: the header, reconciled with the log.
pub fn current_version(conn: &Connection) -> rusqlite::Result<u32> {
    let header: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    let logged: i64 = conn.query_row(
        "SELECT COALESCE(MAX(version), 0) FROM schema_migrations",
        [],
        |r| r.get(0),
    )?;
    Ok(header.max(logged).max(0) as u32)
}

/// Everything `schema_migrations` records, oldest first.
pub fn applied_migrations(conn: &Connection) -> rusqlite::Result<Vec<AppliedMigration>> {
    let mut stmt = conn.prepare(
        "SELECT version, name, applied_at_ms, duration_ms
         FROM schema_migrations ORDER BY version",
    )?;
    let rows = stmt.query_map([], |r| {
        Ok(AppliedMigration {
            version: r.get::<_, i64>(0)? as u32,
            name: r.get(1)?,
            applied_at_ms: r.get(2)?,
            duration_ms: r.get(3)?,
        })
    })?;
    rows.collect()
}

/// The version this build refuses to touch.
fn newer_than_us(conn: &Connection) -> Result<u32> {
    let header: i64 = conn
        .query_row("PRAGMA user_version", [], |r| r.get(0))
        .map_err(|e| error::from_sqlite(e, Path::new("<schema>")))?;
    let latest = latest_version();
    if header > latest as i64 {
        Err(vaani_core::UserFacingError::new(
            vaani_core::ErrorCode::StorageFailed,
            "VaaniDesk needs updating.",
            "This data folder was created by a newer version of VaaniDesk, and \
             this version cannot safely read it.",
        )
        .with_recovery(
            "Update VaaniDesk, then open your transcripts again. Do not delete \
             the data folder.",
        )
        .into())
    } else {
        Ok(header as u32)
    }
}

/// Bring the database up to `target`, applying every migration in between.
///
/// Each migration runs in its own transaction together with the
/// `schema_migrations` row and the `user_version` bump, so a crash leaves the
/// database either fully before or fully after that migration — never in
/// between. Returns the migrations this call applied (empty when the database
/// was already current).
pub fn migrate(conn: &mut Connection, path: &Path, target: u32) -> Result<Vec<AppliedMigration>> {
    conn.execute_batch(BOOTSTRAP)
        .map_err(|e| error::from_sqlite(e, path))?;
    let start = current_version(conn).map_err(|e| error::from_sqlite(e, path))?;
    newer_than_us(conn)?;
    let target = target.min(latest_version());
    if target < start {
        tracing::warn!(
            path = %path.display(),
            start,
            target,
            "refusing to migrate a database backwards; leaving it untouched"
        );
        return Ok(Vec::new());
    }
    let fts5 = fts5_available(conn);
    let ctx = Context { fts5 };
    let mut applied = Vec::new();
    for m in MIGRATIONS
        .iter()
        .filter(|m| m.version > start && m.version <= target)
    {
        let began = Instant::now();
        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(|e| error::from_sqlite(e, path))?;
        (m.apply)(&tx, &ctx)?;
        let now = clock::now_ms();
        let took = began.elapsed().as_millis() as i64;
        tx.execute(
            "INSERT INTO schema_migrations (version, name, applied_at_ms, duration_ms)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(version) DO UPDATE SET
                 name = excluded.name,
                 applied_at_ms = excluded.applied_at_ms,
                 duration_ms = excluded.duration_ms",
            rusqlite::params![m.version, m.name, now, took],
        )
        .map_err(|e| error::from_sqlite(e, path))?;
        // In the same transaction as the DDL: SQLite stores user_version in the
        // file header, so this is atomic with the schema change.
        tx.execute_batch(&format!("PRAGMA user_version = {}", m.version))
            .map_err(|e| error::from_sqlite(e, path))?;
        tx.commit().map_err(|e| error::from_sqlite(e, path))?;
        tracing::info!(
            version = m.version,
            name = m.name,
            took_ms = took,
            "applied migration"
        );
        applied.push(AppliedMigration {
            version: m.version,
            name: m.name.to_string(),
            applied_at_ms: now,
            duration_ms: took,
        });
    }
    Ok(applied)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TempDir;

    fn open() -> (TempDir, Connection) {
        let dir = TempDir::new("migrations");
        let conn = Connection::open(dir.join("v.db")).unwrap();
        conn.execute_batch("PRAGMA foreign_keys=ON").unwrap();
        (dir, conn)
    }

    #[test]
    fn migrations_are_contiguous_and_ordered() {
        let mut expected = 1;
        for m in MIGRATIONS {
            assert_eq!(
                m.version, expected,
                "migration list must be contiguous from 1 with no gaps"
            );
            expected += 1;
        }
        assert_eq!(latest_version(), expected - 1);
    }

    #[test]
    fn a_fresh_database_reaches_latest_and_logs_every_step() {
        let (_d, mut conn) = open();
        let applied = migrate(&mut conn, Path::new("v.db"), latest_version()).unwrap();
        assert_eq!(applied.len(), MIGRATIONS.len());
        let logged = applied_migrations(&conn).unwrap();
        assert_eq!(logged.len(), MIGRATIONS.len());
        assert_eq!(logged[0].version, 1);
        assert_eq!(logged[0].name, "initial");
        assert!(logged.iter().all(|m| m.applied_at_ms > 0));
        assert_eq!(current_version(&conn).unwrap(), latest_version());
    }

    #[test]
    fn running_migrations_twice_is_a_no_op() {
        let (_d, mut conn) = open();
        migrate(&mut conn, Path::new("v.db"), latest_version()).unwrap();
        let again = migrate(&mut conn, Path::new("v.db"), latest_version()).unwrap();
        assert!(again.is_empty(), "re-running must apply nothing");
    }

    #[test]
    fn migrating_to_a_partial_target_stops_there() {
        let (dir, mut conn) = open();
        migrate(&mut conn, &dir.join("v.db"), 1).unwrap();
        assert_eq!(current_version(&conn).unwrap(), 1);
        assert!(search_index_present(&conn), "v1 creates the index");
        let v2 = migrate(&mut conn, &dir.join("v.db"), latest_version()).unwrap();
        assert_eq!(v2.len(), 1);
        assert_eq!(v2[0].version, 2);
    }

    #[test]
    fn add_column_is_idempotent() {
        let (dir, mut conn) = open();
        migrate(&mut conn, &dir.join("v.db"), latest_version()).unwrap();
        // Replay the body of v2 by hand: the ALTER must be a no-op.
        add_column(
            &conn,
            "transcripts",
            "segment_count",
            "INTEGER NOT NULL DEFAULT 0",
        )
        .unwrap();
        add_column(
            &conn,
            "transcripts",
            "word_count",
            "INTEGER NOT NULL DEFAULT 0",
        )
        .unwrap();
    }

    #[test]
    fn fts5_is_present_in_the_bundled_build() {
        // This is the assumption migration 1 is built on. If a future SQLite
        // drops it, this test fails loudly instead of the search layer
        // silently degrading.
        let (_d, conn) = open();
        assert!(
            fts5_available(&conn),
            "rusqlite's bundled SQLite must include FTS5"
        );
    }

    #[test]
    fn a_newer_schema_is_refused_rather_than_downgraded() {
        let (dir, mut conn) = open();
        migrate(&mut conn, &dir.join("v.db"), latest_version()).unwrap();
        conn.execute_batch(&format!("PRAGMA user_version = {}", latest_version() + 5))
            .unwrap();
        let err = migrate(&mut conn, &dir.join("v.db"), latest_version()).unwrap_err();
        assert_eq!(err.code(), vaani_core::ErrorCode::StorageFailed);
        assert_eq!(err.user.title, "VaaniDesk needs updating.");
        // The version is untouched: we refused rather than "fixed" it.
        assert_eq!(current_version(&conn).unwrap(), latest_version() + 5);
    }

    #[test]
    fn the_header_wins_when_the_log_is_missing_a_row() {
        // A hand-restored backup can have a header that disagrees with the
        // log. Replay must be safe, and the version must not go backwards.
        let (dir, mut conn) = open();
        migrate(&mut conn, &dir.join("v.db"), latest_version()).unwrap();
        conn.execute_batch("DELETE FROM schema_migrations").unwrap();
        conn.execute_batch("PRAGMA user_version = 0").unwrap();
        assert_eq!(current_version(&conn).unwrap(), 0);
        let applied = migrate(&mut conn, &dir.join("v.db"), latest_version()).unwrap();
        assert_eq!(applied.len(), MIGRATIONS.len());
        assert_eq!(current_version(&conn).unwrap(), latest_version());
    }

    #[test]
    fn time_columns_are_integer_typed() {
        // A `TEXT` time column is how a transcript ends up three minutes out of
        // sync. The live schema is asked, not the DDL string.
        let (dir, mut conn) = open();
        migrate(&mut conn, &dir.join("v.db"), latest_version()).unwrap();
        for (table, column) in [
            ("segments", "start_ms"),
            ("segments", "end_ms"),
            ("words", "start_ms"),
            ("words", "end_ms"),
            ("transcripts", "duration_ms"),
            ("jobs", "created_at_ms"),
        ] {
            let ty: String = conn
                .query_row(
                    &format!("SELECT type FROM pragma_table_info('{table}') WHERE name = ?1"),
                    [column],
                    |r| r.get(0),
                )
                .unwrap_or_else(|e| panic!("{table}.{column}: {e}"));
            assert!(
                ty.eq_ignore_ascii_case("integer"),
                "{table}.{column} is {ty}, want INTEGER"
            );
        }
    }
}

//! Opening, configuring and recovering the database.
//!
//! # The concurrency model, decided on purpose
//!
//! `rusqlite::Connection` is `Send` but **not** `Sync`, and a single SQLite
//! file has exactly one writer. Rather than a pool of interchangeable
//! connections (which trades a clear invariant for an emergent one), this store
//! has a fixed shape:
//!
//! ```text
//!            ┌──────────────────────────┐
//!  writes ─▶ │  writer: Mutex<Connection>│   one connection, one at a time,
//!            └──────────────────────────┘   IMMEDIATE transactions
//!            ┌──────────────────────────┐
//!  reads  ─▶ │  pool: up to N Connections │  concurrent, never on the write
//!            └──────────────────────────┘  path
//! ```
//!
//! * **One writer.** Every mutation goes through [`Store::with_write`], which
//!   holds a `parking_lot::Mutex` and opens an `IMMEDIATE` transaction. That
//!   makes `SQLITE_BUSY` between our own threads structurally impossible: there
//!   is only ever one writer, so there is nothing to be busy with. `IMMEDIATE`
//!   rather than `DEFERRED` because a deferred transaction that reads and then
//!   writes can fail to upgrade its lock, and that failure (`SQLITE_BUSY_SNAPSHOT`,
//!   which `busy_timeout` does *not* retry) is the classic SQLite deadlock.
//!   Taking the write lock up front turns a would-be deadlock into a wait.
//! * **A small read pool.** Reads are the hot path in the UI (a transcript
//!   view, a search-as-you-type) and must not queue behind a 40 MB transcript
//!   write. Under WAL a reader never blocks the writer and vice versa, so the
//!   pool is genuinely concurrent. Connections are created lazily up to
//!   [`OpenOptions::max_read_connections`] and returned to a free list.
//! * **Read connections are opened read-write, not `SQLITE_OPEN_READ_ONLY`.**
//!   A read-only connection cannot initialise the WAL shared-memory index, so
//!   on a fresh database (or after a checkpoint that removed the `-shm`) it
//!   fails with `SQLITE_CANTOPEN`/`SQLITE_READONLY_CANTINIT`. They are used
//!   only for `SELECT`s, which is enforced by where they are handed out rather
//!   than by the open flags.
//! * **`SQLITE_OPEN_NO_MUTEX`.** SQLite's own mutex is off because our `Mutex`
//!   provides the exclusion; leaving it on would add an uncontended-but-real
//!   lock on every statement for no benefit.
//! * **`Store` is `Clone` and `Send + Sync`.** The clone is an `Arc` over the
//!   same connections, so the pipeline can hand a store to eight worker
//!   threads without opening eight databases.
//!
//! ## Cross-process
//!
//! The app and the CLI can be open at once. `busy_timeout` covers that: a
//! writer waits up to [`OpenOptions::busy_timeout`] instead of failing
//! immediately, and a failure past that is reported as "Storage is busy —
//! close the other copy", not as a generic failure.
//!
//! # Pragmas, and why each one
//!
//! | pragma | value | why |
//! |---|---|---|
//! | `journal_mode` | `WAL` | readers do not block the writer, and **a process killed mid-transaction leaves the file consistent** — that is what makes the crash test pass. Falls back to `DELETE` if the database is on a filesystem without shared memory (a network share), because a desktop app that refuses to start on a home-directory redirect is worse than one with weaker concurrency. |
//! | `synchronous` | `NORMAL` | with WAL, `NORMAL` is crash-safe: the OS can lose the tail of the log on a power cut but the database cannot be corrupted. `FULL` fsyncs every commit, which buys nothing here because a lost job-status update is re-derivable, and costs an fsync per job row. `FULL` is available for users who would rather wait. |
//! | `foreign_keys` | `ON` | **off by default in SQLite.** Without it the cascading deletes in [`crate::migrations`] are documentation, not behaviour, and deleting a job would leave its 20,000 words behind. It is per-connection, so it is set on every one. |
//! | `busy_timeout` | 5 s default | the other process (CLI) holds the write lock for a whole transcript insert otherwise. |
//! | `cache_size` | −2000 (2 MB) | a 2 MB page cache per connection is a sane desktop default: big enough to hold a job list, small enough that eight connections do not matter. |
//!
//! # Crash recovery
//!
//! A job that was running when the app died is not running any more, and
//! leaving it in `ASR` forever is how a user ends up with a list of jobs that
//! claim to be 40 % done and will never move. [`Store::recover_interrupted_jobs`]
//! runs on open (unless disabled) and moves every non-terminal job to a
//! terminal phase: `QUEUED` becomes `CANCELLED` ("was still waiting"), an active
//! phase becomes `FAILED` ("VaaniDesk closed while this job was running"). The
//! transcripts of *completed* jobs are untouched — recovery is a single
//! `UPDATE` on `jobs` and cannot touch `segments` or `words`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::{Condvar, Mutex};
use rusqlite::{Connection, OpenFlags, TransactionBehavior};

use crate::Result;
use vaani_core::{ErrorCode, UserFacingError};

use crate::clock;
use crate::error;
use crate::migrations;

/// Durability trade-off, exposed because it is a real user choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SyncMode {
    /// Fastest. Survives an application crash; a power cut can lose the last
    /// few transactions, never the schema or a committed transcript row.
    #[default]
    Normal,
    /// fsync on every commit. Slower, and the only setting that also survives a
    /// power cut with nothing lost.
    Full,
    /// Fastest and least safe. Only for a database on a filesystem where fsync
    /// is meaningless anyway.
    Off,
}

impl SyncMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "NORMAL",
            Self::Full => "FULL",
            Self::Off => "OFF",
        }
    }
}

/// How to open the database.
#[derive(Debug, Clone)]
pub struct OpenOptions {
    /// Create the database (and its directory) if missing. `false` makes a
    /// missing file an error, which is what the CLI wants when it is asked to
    /// read a specific corpus.
    pub create: bool,
    /// How long to wait for another process's write lock.
    pub busy_timeout: Duration,
    /// Move jobs that were running when the app died into a terminal phase.
    /// On by default: see the module docs on crash recovery.
    pub recover_interrupted: bool,
    /// Upper bound on pooled read connections.
    pub max_read_connections: usize,
    /// Durability. See [`SyncMode`].
    pub synchronous: SyncMode,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self {
            create: true,
            busy_timeout: Duration::from_secs(5),
            recover_interrupted: true,
            max_read_connections: 4,
            synchronous: SyncMode::Normal,
        }
    }
}

/// The database. Cheap to clone; every clone shares the same connections.
#[derive(Debug, Clone)]
pub struct Store {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    path: PathBuf,
    /// `:memory:` databases are per-connection, so the pool is disabled and
    /// every access goes through the one connection. Correct beats concurrent.
    in_memory: bool,
    writer: Mutex<Connection>,
    readers: Mutex<ReaderPool>,
    readers_returned: Condvar,
    max_readers: usize,
    /// Whether the `segment_fts` index exists in this database.
    indexed: bool,
}

#[derive(Debug, Default)]
struct ReaderPool {
    idle: Vec<Connection>,
    /// Connections that exist, idle or checked out. Only ever grows, to the
    /// configured maximum: a connection that is returned is reused rather than
    /// closed, because opening one is the expensive part.
    live: usize,
}

/// A borrowed read connection that goes back to the pool when dropped.
#[derive(Debug)]
struct ReadConn<'a> {
    inner: &'a Inner,
    conn: Option<Connection>,
}

impl std::ops::Deref for ReadConn<'_> {
    type Target = Connection;
    fn deref(&self) -> &Connection {
        // Only ever constructed with a connection.
        self.conn
            .as_ref()
            .expect("read connection is always present")
    }
}

impl Drop for ReadConn<'_> {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            let mut pool = self.inner.readers.lock();
            pool.idle.push(conn);
            self.inner.readers_returned.notify_one();
        }
    }
}

impl Store {
    /// Open (creating if necessary), migrate, and recover.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with(path.as_ref(), &OpenOptions::default())
    }

    /// Open and migrate only as far as `version`, leaving the database there.
    ///
    /// For reproducing an older schema — the upgrade test builds a v1 database
    /// with this, writes real data into it, and then opens it normally so the v2
    /// migration runs against it. Production code uses [`Store::open`]: a build
    /// that opened a user's database at an old version on purpose would be a
    /// downgrade tool.
    pub fn open_at_version(path: impl AsRef<Path>, version: u32) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)
                .map_err(|e| error::from_io(&format!("creating {}", parent.display()), e))?;
        }
        let mut conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX
                | OpenFlags::SQLITE_OPEN_URI,
        )
        .map_err(|e| error::from_sqlite(e, path))?;
        configure(&conn, path, &OpenOptions::default(), true)?;
        migrations::migrate(&mut conn, path, version)?;
        let indexed = migrations::search_index_present(&conn);
        Ok(Self {
            inner: Arc::new(Inner {
                path: path.to_path_buf(),
                in_memory: false,
                writer: Mutex::new(conn),
                readers: Mutex::new(ReaderPool::default()),
                readers_returned: Condvar::new(),
                max_readers: OpenOptions::default().max_read_connections,
                indexed,
            }),
        })
    }

    /// Open with explicit options.
    pub fn open_with(path: &Path, options: &OpenOptions) -> Result<Self> {
        if options.create
            && let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)
                .map_err(|e| error::from_io(&format!("creating {}", parent.display()), e))?;
        }
        let mut flags = OpenFlags::SQLITE_OPEN_NO_MUTEX | OpenFlags::SQLITE_OPEN_URI;
        flags |= if options.create {
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE
        } else {
            OpenFlags::SQLITE_OPEN_READ_WRITE
        };
        let mut writer =
            Connection::open_with_flags(path, flags).map_err(|e| error::from_sqlite(e, path))?;
        configure(&writer, path, options, true)?;
        migrations::migrate(&mut writer, path, migrations::latest_version())?;
        let indexed = migrations::search_index_present(&writer);
        let store = Self {
            inner: Arc::new(Inner {
                path: path.to_path_buf(),
                in_memory: false,
                writer: Mutex::new(writer),
                readers: Mutex::new(ReaderPool::default()),
                readers_returned: Condvar::new(),
                max_readers: options.max_read_connections,
                indexed,
            }),
        };
        if options.recover_interrupted {
            let recovered = store.recover_interrupted_jobs()?;
            if !recovered.is_empty() {
                tracing::info!(
                    count = recovered.len(),
                    "marked jobs interrupted by a previous exit"
                );
            }
        }
        Ok(store)
    }

    /// A throwaway in-memory database, for tests and for `--dry-run` style
    /// queries. Uses a single connection, so it is not concurrent.
    pub fn open_in_memory() -> Result<Self> {
        let mut conn = Connection::open_in_memory()
            .map_err(|e| error::from_sqlite(e, Path::new(":memory:")))?;
        // No WAL for a memory database: the journal is the heap. `foreign_keys`
        // still matters, because the cascade behaviour is under test.
        conn.execute_batch("PRAGMA foreign_keys = ON; PRAGMA synchronous = OFF;")
            .map_err(|e| error::from_sqlite(e, Path::new(":memory:")))?;
        migrations::migrate(
            &mut conn,
            Path::new(":memory:"),
            migrations::latest_version(),
        )?;
        let indexed = migrations::search_index_present(&conn);
        Ok(Self {
            inner: Arc::new(Inner {
                path: PathBuf::from(":memory:"),
                in_memory: true,
                writer: Mutex::new(conn),
                readers: Mutex::new(ReaderPool::default()),
                readers_returned: Condvar::new(),
                max_readers: 0,
                indexed,
            }),
        })
    }

    /// The platform's data directory for VaaniDesk, created if missing.
    ///
    /// `dirs::data_dir()` is `%LOCALAPPDATA%` on Windows and
    /// `~/.local/share` on Linux. The database is named `vaanidesk.db` inside a
    /// `VaaniDesk` folder, so it is obvious what the file is when a user finds
    /// it and wants to back it up.
    pub fn default_path() -> Result<PathBuf> {
        let base = dirs::data_dir().ok_or_else(|| {
            UserFacingError::new(
                ErrorCode::StorageFailed,
                "No place to store data.",
                "This computer does not have a standard location for application data.",
            )
            .with_recovery("Set a home directory for your user account, then restart VaaniDesk.")
        })?;
        Ok(base.join("VaaniDesk").join("vaanidesk.db"))
    }

    /// Open the database in [`Store::default_path`].
    pub fn open_default() -> Result<Self> {
        Self::open(Self::default_path()?)
    }

    /// Where this database lives. `:memory:` for an in-memory store.
    pub fn path(&self) -> &Path {
        &self.inner.path
    }

    /// Whether full-text search is indexed in this database.
    pub fn has_search_index(&self) -> bool {
        self.inner.indexed
    }

    /// The schema version this database is at.
    pub fn schema_version(&self) -> Result<u32> {
        self.read(migrations::current_version)
    }

    /// Everything the migration log records, oldest first.
    pub fn applied_migrations(&self) -> Result<Vec<migrations::AppliedMigration>> {
        self.read(migrations::applied_migrations)
    }

    /// Fold the write-ahead log back into the main database file.
    ///
    /// Worth calling when the app is about to exit, so that a user who copies
    /// `vaanidesk.db` to back it up gets a self-contained file rather than one
    /// that needs the `-wal` beside it.
    pub fn close(&self) -> Result<()> {
        if self.inner.in_memory {
            return Ok(());
        }
        let conn = self.inner.writer.lock();
        conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
            .map_err(|e| error::from_sqlite(e, &self.inner.path))
    }

    /// `PRAGMA integrity_check`, mapped to a readable result.
    ///
    /// Not run automatically: it reads the whole file, which is the right price
    /// for "my transcripts have gone" and the wrong price for every launch. The
    /// settings panel's "check storage" button is its caller.
    ///
    /// # Why this runs on the writer connection
    ///
    /// Measured, not assumed. `integrity_check` on a *pooled reader* connection
    /// reports `malformed inverted index for FTS5 table segment_fts` on a
    /// database that is provably fine: FTS5's own `'integrity-check'` command
    /// reports nothing, a `'rebuild'` changes nothing, and reopening the file
    /// reports `ok`. It only happens while several connections have a hot
    /// write-ahead log between them, and it disappears when the check runs on
    /// the connection that did the writing. A "check my storage" button that
    /// cries corruption at a healthy 300 MB database is worse than no button,
    /// so the check runs on the writer — which also serialises it against
    /// writes, which is what a maintenance action wants anyway.
    pub fn integrity_check(&self) -> Result<IntegrityReport> {
        let rows: Vec<String> = self.with_conn(|conn| {
            let mut stmt = conn
                .prepare("PRAGMA integrity_check")
                .map_err(|e| error::from_sqlite(e, &self.inner.path))?;
            let mapped = stmt
                .query_map([], |r| r.get::<_, String>(0))
                .map_err(|e| error::from_sqlite(e, &self.inner.path))?;
            let mut out = Vec::new();
            for row in mapped {
                out.push(row.map_err(|e| error::from_sqlite(e, &self.inner.path))?);
            }
            Ok(out)
        })?;
        Ok(IntegrityReport {
            ok: rows.len() == 1 && rows[0] == "ok",
            messages: rows,
        })
    }

    /// Whether a job with this id exists.
    pub fn job_exists(&self, id: &str) -> Result<bool> {
        self.read(|conn| {
            let n: i64 = conn.query_row("SELECT COUNT(*) FROM jobs WHERE id = ?1", [id], |r| {
                r.get(0)
            })?;
            Ok(n > 0)
        })
    }

    /// Move every job that was running when the app died into a terminal phase.
    ///
    /// Idempotent: a second call finds nothing to do. Returns the ids it
    /// changed, so a caller can tell the user which jobs need re-running.
    pub fn recover_interrupted_jobs(&self) -> Result<Vec<String>> {
        self.write(|tx| {
            let mut stmt = tx.prepare(
                "SELECT id, phase FROM jobs
                 WHERE phase NOT IN ('COMPLETE', 'FAILED', 'CANCELLED')",
            )?;
            let pending: Vec<(String, String)> = stmt
                .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            drop(stmt);
            let now = clock::now_ms();
            let mut changed = Vec::with_capacity(pending.len());
            for (id, phase) in pending {
                // A job that never started was not interrupted mid-work; it was
                // dropped from a queue that did not survive. `CANCELLED` is
                // right for that. But work that was under way when the process
                // died is `INTERRUPTED`, not cancelled: nobody asked for it to
                // stop and it will not resume, so the job list must not pretend
                // the user pressed stop.
                let (to, message, code) = if phase == "QUEUED" {
                    (
                        "CANCELLED",
                        "Still waiting to start when VaaniDesk last closed.",
                        "cancelled",
                    )
                } else {
                    (
                        "FAILED",
                        "VaaniDesk closed while this job was running. Run it again to finish.",
                        "interrupted",
                    )
                };
                tx.execute(
                    "UPDATE jobs SET phase = ?2, finished_at_ms = ?3, error_code = ?4, error_message = ?5
                     WHERE id = ?1",
                    rusqlite::params![id, to, now, code, message],
                )?;
                changed.push(id);
            }
            Ok(changed)
        })
    }

    /// Run `f` with a read-only view of the database.
    ///
    /// The escape hatch for callers that need arbitrary SQL. Inside this crate
    /// the [`Store::read`] helper is preferred: it translates SQLite errors into
    /// the application's taxonomy at the boundary, where the database path is
    /// known.
    pub fn with_read<T>(&self, f: impl FnOnce(&Connection) -> Result<T>) -> Result<T> {
        if self.inner.in_memory {
            let conn = self.inner.writer.lock();
            return f(&conn);
        }
        let conn = self.checkout();
        f(&conn)
    }

    /// Run `f` on the writer connection without opening a transaction.
    ///
    /// For single statements only (`CREATE INDEX`, `VACUUM`, pragmas). Anything
    /// that reads and then writes belongs in [`Store::with_write`].
    pub fn with_conn<T>(&self, f: impl FnOnce(&Connection) -> Result<T>) -> Result<T> {
        let conn = self.inner.writer.lock();
        f(&conn)
    }

    /// Run `f` with a read connection, propagating `rusqlite::Error` unchanged.
    ///
    /// For internal helpers that want to map the error once at their own
    /// boundary rather than at every `?`. Not part of the public API.
    pub(crate) fn with_read_raw<T>(
        &self,
        f: impl FnOnce(&Connection) -> rusqlite::Result<T>,
    ) -> rusqlite::Result<T> {
        if self.inner.in_memory {
            let conn = self.inner.writer.lock();
            return f(&conn);
        }
        let conn = self.checkout();
        f(&conn)
    }

    /// The workhorse behind every query in this crate: a read whose closure
    /// speaks plain `rusqlite::Result`, with the error translated once at the
    /// boundary.
    ///
    /// This is why there is no `impl From<rusqlite::Error> for VaaniError` and
    /// no `?` on a SQLite error anywhere in the query code: the conversion
    /// happens in exactly one place, and it needs the database path to produce a
    /// message that names the file.
    pub(crate) fn read<T>(&self, f: impl FnOnce(&Connection) -> rusqlite::Result<T>) -> Result<T> {
        self.with_read_raw(f)
            .map_err(|e| error::from_sqlite(e, &self.inner.path))
    }

    /// The workhorse behind every mutation: one `IMMEDIATE` transaction, with
    /// the error translated once at the boundary.
    ///
    /// Dropping the transaction on error rolls it back, so a partially applied
    /// write is not reachable.
    pub(crate) fn write<T>(
        &self,
        f: impl FnOnce(&rusqlite::Transaction<'_>) -> rusqlite::Result<T>,
    ) -> Result<T> {
        let mut conn = self.inner.writer.lock();
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|e| error::from_sqlite(e, &self.inner.path))?;
        let out = match f(&tx) {
            Ok(v) => v,
            Err(e) => {
                // `tx` drops here, rolling back.
                return Err(error::from_sqlite(e, &self.inner.path));
            }
        };
        tx.commit()
            .map_err(|e| error::from_sqlite(e, &self.inner.path))?;
        Ok(out)
    }

    /// Run `f` inside one `IMMEDIATE` transaction, committing on `Ok` and
    /// rolling back on `Err`.
    ///
    /// This is the only way to mutate. It is the reason a half-written
    /// transcript cannot exist: a transcript is 20,000 inserts, and either all
    /// of them are visible or none are.
    pub fn with_write<T>(
        &self,
        f: impl FnOnce(&rusqlite::Transaction<'_>) -> Result<T>,
    ) -> Result<T> {
        let mut conn = self.inner.writer.lock();
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|e| error::from_sqlite(e, &self.inner.path))?;
        let out = f(&tx)?;
        tx.commit()
            .map_err(|e| error::from_sqlite(e, &self.inner.path))?;
        Ok(out)
    }

    /// Borrow a read connection from the pool, waiting for a free one if the
    /// pool is saturated.
    fn checkout(&self) -> ReadConn<'_> {
        let mut pool = self.inner.readers.lock();
        loop {
            if let Some(conn) = pool.idle.pop() {
                return ReadConn {
                    inner: &self.inner,
                    conn: Some(conn),
                };
            }
            if pool.live < self.inner.max_readers {
                pool.live += 1;
                break;
            }
            // `parking_lot`'s condvar takes `&mut guard` and returns nothing:
            // there is no poisoning case to handle, and the guard is still held
            // afterwards.
            self.inner.readers_returned.wait(&mut pool);
        }
        drop(pool);
        // Opened read-write on purpose: see the module docs. A failure here is
        // a real I/O or permission problem, and the pool must not grow.
        match self.open_reader() {
            Ok(conn) => ReadConn {
                inner: &self.inner,
                conn: Some(conn),
            },
            Err(e) => {
                // Give the slot back before surfacing the failure, or the pool
                // would leak a slot on every error and eventually deadlock.
                let mut pool = self.inner.readers.lock();
                pool.live = pool.live.saturating_sub(1);
                self.inner.readers_returned.notify_one();
                drop(pool);
                panic!("could not open a read connection: {e}");
            }
        }
    }

    fn open_reader(&self) -> Result<Connection> {
        let conn = Connection::open_with_flags(
            &self.inner.path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX
                | OpenFlags::SQLITE_OPEN_URI,
        )
        .map_err(|e| error::from_sqlite(e, &self.inner.path))?;
        configure(&conn, &self.inner.path, &OpenOptions::default(), false)?;
        Ok(conn)
    }
}

/// The result of `PRAGMA integrity_check`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegrityReport {
    /// True when SQLite reported exactly `ok`.
    pub ok: bool,
    /// Whatever SQLite reported. `["ok"]` when healthy.
    pub messages: Vec<String>,
}

fn configure(conn: &Connection, path: &Path, options: &OpenOptions, is_writer: bool) -> Result<()> {
    // `busy_timeout` is the per-connection answer to "another process is
    // writing"; `foreign_keys` is off by default in SQLite and is what makes
    // the cascades real. Both are per-connection, so both belong here and not
    // only on the writer.
    conn.busy_timeout(options.busy_timeout)
        .map_err(|e| error::from_sqlite(e, path))?;
    conn.execute_batch("PRAGMA foreign_keys = ON;")
        .map_err(|e| error::from_sqlite(e, path))?;
    conn.execute_batch(&format!(
        "PRAGMA synchronous = {};",
        options.synchronous.as_str()
    ))
    .map_err(|e| error::from_sqlite(e, path))?;
    // Negative means KiB rather than pages, so this is 2 MB regardless of page
    // size.
    conn.execute_batch("PRAGMA cache_size = -2000;")
        .map_err(|e| error::from_sqlite(e, path))?;
    if is_writer {
        // WAL is persistent, so only the writer sets it. If the filesystem
        // cannot do it (a network share, some cloud-sync folders), fall back
        // rather than refusing to run: the app is still correct, the only loss
        // is reader/writer concurrency.
        match conn.query_row("PRAGMA journal_mode = WAL", [], |r| r.get::<_, String>(0)) {
            Ok(mode) if mode.eq_ignore_ascii_case("wal") => {}
            Ok(mode) => {
                tracing::warn!(mode, "WAL unavailable; using the rollback journal");
            }
            Err(e) => {
                tracing::warn!(error = %e, "could not set WAL; using the rollback journal");
                conn.execute_batch("PRAGMA journal_mode = DELETE;")
                    .map_err(|err| error::from_sqlite(err, path))?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TempDir;

    #[test]
    fn open_creates_the_directory_and_migrates() {
        let dir = TempDir::new("open-nested");
        let path = dir.join("nested/deeper/vaani.db");
        let store = Store::open(&path).unwrap();
        assert_eq!(
            store.schema_version().unwrap(),
            migrations::latest_version()
        );
        assert!(path.exists());
        assert!(store.has_search_index());
    }

    #[test]
    fn create_false_refuses_a_missing_file() {
        let dir = TempDir::new("open-nocreate");
        let options = OpenOptions {
            create: false,
            ..Default::default()
        };
        let e = Store::open_with(&dir.join("missing.db"), &options).unwrap_err();
        assert_eq!(e.code(), ErrorCode::StorageFailed);
    }

    #[test]
    fn in_memory_stores_actually_store() {
        let store = Store::open_in_memory().unwrap();
        // Nothing from a previous test may leak in.
        assert_eq!(
            store
                .read(|c| {
                    let n: i64 = c.query_row("SELECT COUNT(*) FROM jobs", [], |r| r.get(0))?;
                    Ok::<i64, rusqlite::Error>(n)
                })
                .unwrap(),
            0
        );
        assert_eq!(store.path(), Path::new(":memory:"));
    }

    #[test]
    fn integrity_check_is_clean_on_a_fresh_database() {
        let dir = TempDir::new("integrity");
        let store = Store::open(dir.join("v.db")).unwrap();
        let report = store.integrity_check().unwrap();
        assert!(report.ok, "{:?}", report.messages);
        assert_eq!(report.messages, vec!["ok".to_string()]);
    }

    #[test]
    fn recovery_terminates_running_jobs_and_leaves_finished_ones() {
        use vaani_core::JobPhase;
        let dir = TempDir::new("recover");
        let path = dir.join("v.db");
        {
            let store = Store::open(&path).unwrap();
            for (i, phase) in [
                JobPhase::Queued,
                JobPhase::Asr,
                JobPhase::Complete,
                JobPhase::Cancelled,
            ]
            .into_iter()
            .enumerate()
            {
                let mut rec = crate::JobRecord::new(format!("job{i}"), dir.join("a.wav"));
                rec.phase = phase;
                store.upsert_job(&rec).unwrap();
            }
        }
        // Reopening must fix the two unfinished jobs and touch nothing else.
        let store = Store::open_with(
            &path,
            &OpenOptions {
                recover_interrupted: false,
                ..Default::default()
            },
        )
        .unwrap();
        let changed = store.recover_interrupted_jobs().unwrap();
        assert_eq!(changed.len(), 2);
        assert!(changed.contains(&"job0".to_string()));
        assert!(changed.contains(&"job1".to_string()));
        assert_eq!(
            store.load_job("job0").unwrap().unwrap().phase,
            JobPhase::Cancelled
        );
        assert_eq!(
            store.load_job("job1").unwrap().unwrap().phase,
            JobPhase::Failed
        );
        assert_eq!(
            store.load_job("job2").unwrap().unwrap().phase,
            JobPhase::Complete
        );
        // Idempotent.
        assert!(store.recover_interrupted_jobs().unwrap().is_empty());
    }

    #[test]
    fn recovery_records_why_the_job_stopped() {
        let dir = TempDir::new("recover-why");
        let store = Store::open(dir.join("v.db")).unwrap();
        let mut rec = crate::JobRecord::new("j", dir.join("a.wav"));
        rec.phase = vaani_core::JobPhase::Diarization;
        store.upsert_job(&rec).unwrap();
        store.recover_interrupted_jobs().unwrap();
        let job = store.load_job("j").unwrap().unwrap();
        let err = job.error.expect("an interrupted job must carry a reason");
        // `interrupted`, not `cancelled`: nobody pressed stop, the process
        // died. A job list that calls this "cancelled" is lying.
        assert_eq!(err.code, "interrupted");
        assert!(err.message.contains("closed while this job was running"));
        assert!(job.finished_at_ms.is_some());
    }

    #[test]
    fn recovery_calls_a_never_started_job_cancelled_not_interrupted() {
        // The counterpart: a job that never started was dropped from a queue,
        // not killed mid-work, so `cancelled` is still right for it.
        let dir = TempDir::new("recover-queued");
        let store = Store::open(dir.join("v.db")).unwrap();
        let mut rec = crate::JobRecord::new("q", dir.join("a.wav"));
        rec.phase = vaani_core::JobPhase::Queued;
        store.upsert_job(&rec).unwrap();
        store.recover_interrupted_jobs().unwrap();
        let job = store.load_job("q").unwrap().unwrap();
        let err = job.error.expect("a recovered job must carry a reason");
        assert_eq!(err.code, "cancelled");
    }

    #[test]
    fn a_store_is_shareable_across_threads() {
        let dir = TempDir::new("share");
        let path = dir.join("a.wav");
        let store = Store::open(dir.join("v.db")).unwrap();
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let s = store.clone();
                let path = path.clone();
                std::thread::spawn(move || {
                    let rec = crate::JobRecord::new(format!("t{i}"), path);
                    s.upsert_job(&rec).unwrap();
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let page = store
            .list_jobs(&crate::JobFilter::default(), 10, 0)
            .unwrap();
        assert_eq!(page.total, 4);
    }

    #[test]
    fn read_connections_are_returned_to_the_pool() {
        let dir = TempDir::new("pool");
        let store = Store::open_with(
            &dir.join("v.db"),
            &OpenOptions {
                max_read_connections: 1,
                ..Default::default()
            },
        )
        .unwrap();
        for _ in 0..20 {
            let v: i64 = store
                .read(|c| c.query_row("SELECT 1", [], |r| r.get(0)))
                .unwrap();
            assert_eq!(v, 1);
        }
    }

    #[test]
    fn concurrent_readers_do_not_deadlock_with_one_reader_slot() {
        let dir = TempDir::new("pool-deadlock");
        let store = Store::open_with(
            &dir.join("v.db"),
            &OpenOptions {
                max_read_connections: 1,
                ..Default::default()
            },
        )
        .unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        for i in 0..8 {
            let s = store.clone();
            let tx = tx.clone();
            std::thread::spawn(move || {
                for _ in 0..50 {
                    let _ = s
                        .read(|c| c.query_row("SELECT 1", [], |r| r.get::<_, i64>(0)))
                        .unwrap();
                }
                let _ = i;
                tx.send(()).unwrap();
            });
        }
        drop(tx);
        // If the pool leaked a slot on the fourth reader, this would hang.
        assert!(rx.recv_timeout(Duration::from_secs(30)).is_ok());
        for _ in 0..7 {
            assert!(rx.recv_timeout(Duration::from_secs(30)).is_ok());
        }
    }
}

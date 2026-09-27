//! # vaani-storage
//!
//! The job database: persistence, search, and crash recovery. SQLite via
//! `rusqlite`, bundled, in one file under the user's data directory. No server,
//! no installer step, no "where is my connection string" — a desktop app that
//! needs PostgreSQL installed is not a desktop app.
//!
//! Owned by agent **A6**. The `Cargo.toml` in this directory is frozen by the
//! architect and is part of the integration surface; the contents of the rest of
//! this crate are not.
//!
//! Read `docs/agent/PARALLEL_OWNERSHIP.md` before the first edit.
//!
//! ## What is in here
//!
//! | module | owns |
//! |---|---|
//! | [`migrations`] | the schema DDL, the migration list, and the runner |
//! | [`store`] | opening, pragmas, the connection model, crash recovery |
//! | [`jobs`] | `JobRecord` and everything about "what was asked for" |
//! | [`transcripts`] | `Transcript` persistence and the round-trip guarantee |
//! | [`search`] | FTS5 index queries, ranking, snippets, and the no-FTS5 fallback |
//! | [`settings`] | the settings key/value store and the export log |
//! | [`stats`] | space on disk, row counts, `VACUUM` |
//! | [`hash`] | SHA-256 for the duplicate-transcription check |
//! | [`clock`] | epoch milliseconds |
//! | [`testing`], [`fixtures`] | shared test scaffolding, `#[doc(hidden)]` |
//!
//! ## The five properties this crate is built around
//!
//! 1. **A transcript round-trips exactly.** Every `Time`, every confidence,
//!    every stable segment id, every speaker, in the same order. Verified by a
//!    property test over hundreds of generated transcripts, not one example.
//! 2. **A half-written transcript cannot exist.** A transcript is written in one
//!    `IMMEDIATE` transaction: the old rows, 20,000 new words, the search index
//!    and the counters are all visible together or none are.
//! 3. **Upgrading never loses a transcript.** Forward-only, numbered,
//!    transactional migrations with a `schema_migrations` audit log. A database
//!    from a newer build is refused rather than downgraded.
//! 4. **Being killed is survivable.** WAL plus one-transaction-per-transcript
//!    means a `TerminateProcess` mid-write leaves the previous transcript
//!    intact and the interrupted job marked, not lost. Tested by actually
//!    killing a child process mid-transaction.
//! 5. **Errors are for users.** Every failure leaves as a
//!    [`vaani_core::VaaniError`] with a reviewed message and a recovery action;
//!    a corrupt database says so instead of panicking or silently starting
//!    empty.
//!
//! ## Using it
//!
//! ```no_run
//! use vaani_storage::{JobRecord, Store, new_job_id};
//!
//! let store = Store::open_default()?;
//! let job = JobRecord::new(new_job_id(), "C:/Media/standup.m4a");
//! store.upsert_job(&job)?;
//! # Ok::<(), vaani_core::VaaniError>(())
//! ```
//!
//! `Store` is `Clone + Send + Sync`, and every clone shares one writer and a
//! small read pool, so handing it to eight worker threads is free. See
//! [`store`] for why that is the shape.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// The application's result type.
///
/// `vaani_core::Result<T, E = VaaniError>` declares a default type parameter,
/// and a type alias cannot supply one, so `vaani_core::Result<T>` does not
/// compile. This alias exists so that every signature in this crate reads
/// `Result<T>` and still returns exactly `vaani_core::Result<T, VaaniError>` —
/// the same type, so `?` and `.map_err` work interchangeably across the crate
/// boundary and callers never see two error types.
pub type Result<T> = vaani_core::Result<T, vaani_core::VaaniError>;

pub mod clock;
pub mod error;
pub mod fixtures;
pub mod hash;
pub mod jobs;
pub mod migrations;
pub mod search;
pub mod settings;
pub mod stats;
pub mod store;
pub mod testing;
pub mod transcripts;

pub use jobs::{JobFilter, JobOrder, JobPage, JobRecord, StoredError, new_job_id};
pub use search::{SearchEngine, SearchHit, SearchQuery, SearchResults};
pub use settings::ExportRecord;
pub use stats::StorageStats;
pub use store::{IntegrityReport, OpenOptions, Store, SyncMode};
pub use transcripts::{SpeakerRow, TranscriptStats, TranscriptSummary};

/// Crate version, from the workspace. Recorded in the migration log and in
/// exported transcripts so a support conversation starts with a version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

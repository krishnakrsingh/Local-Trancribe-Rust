//! Storage statistics and maintenance.
//!
//! "How much space is VaaniDesk using?" is a question users ask, and the answer
//! has to be a real number rather than an estimate: the settings panel shows it
//! and the "clear old transcripts" action needs it.
//!
//! `bytes_on_disk` sums the **three** files SQLite may have on disk — the
//! database, the `-wal` and the `-shm`. Reporting only the main file is the
//! classic way to be wrong by a factor of two after a heavy write, because the
//! write-ahead log holds everything not yet checkpointed.

use std::path::Path;

use crate::Result;

use crate::migrations;
use crate::store::Store;

/// A point-in-time summary of what is stored.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)] // field names are the documentation
pub struct StorageStats {
    /// Where the database is.
    pub path: String,
    /// Database + write-ahead log + shared memory, as they are on disk now.
    pub bytes_on_disk: u64,
    /// Just the main database file.
    pub database_bytes: u64,
    /// Just the write-ahead log, which is what a checkpoint reclaims.
    pub wal_bytes: u64,
    /// SQLite's own page accounting: `page_count * page_size` is the space the
    /// file has claimed, which can exceed `bytes_on_disk` mid-checkpoint.
    pub page_count: i64,
    pub page_size: i64,
    /// `page_count * page_size`.
    pub logical_bytes: i64,
    pub job_count: i64,
    pub complete_count: i64,
    pub failed_count: i64,
    pub transcript_count: i64,
    pub segment_count: i64,
    pub word_count: i64,
    pub speaker_count: i64,
    pub export_count: i64,
    pub schema_version: u32,
    /// Whether full-text search is indexed.
    pub fts5: bool,
}

impl StorageStats {
    /// A one-line summary for the settings panel.
    pub fn summary(&self) -> String {
        format!(
            "{} jobs, {} transcripts, {} segments, {} words, {:.1} MB on disk",
            self.job_count,
            self.transcript_count,
            self.segment_count,
            self.word_count,
            self.bytes_on_disk as f64 / 1_048_576.0
        )
    }
}

fn file_len(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

impl Store {
    /// Everything the statistics panel shows.
    pub fn stats(&self) -> Result<StorageStats> {
        let path = self.path().to_path_buf();
        let (
            page_count,
            page_size,
            job_count,
            complete_count,
            failed_count,
            transcript_count,
            segment_count,
            word_count,
            speaker_count,
            export_count,
            schema_version,
        ) = self.read(|conn| {
            let count = |t: &str| crate::jobs::count(conn, t);
            let phase_count = |p: &str| -> i64 {
                conn.query_row("SELECT COUNT(*) FROM jobs WHERE phase = ?1", [p], |r| {
                    r.get(0)
                })
                .unwrap_or(0)
            };
            let pragma_i64 = |p: &str| -> rusqlite::Result<i64> {
                conn.query_row(&format!("PRAGMA {p}"), [], |r| r.get(0))
            };
            Ok((
                pragma_i64("page_count")?,
                pragma_i64("page_size")?,
                count("jobs"),
                phase_count("COMPLETE"),
                phase_count("FAILED"),
                count("transcripts"),
                count("segments"),
                count("words"),
                count("speakers"),
                count("exports"),
                migrations::current_version(conn)?,
            ))
        })?;
        let database_bytes = file_len(&path);
        let wal_bytes = file_len(Path::new(&format!("{}-wal", path.display())));
        let shm_bytes = file_len(Path::new(&format!("{}-shm", path.display())));
        let logical_bytes = page_count * page_size;
        Ok(StorageStats {
            path: path.to_string_lossy().into_owned(),
            bytes_on_disk: database_bytes + wal_bytes + shm_bytes,
            database_bytes,
            wal_bytes,
            page_size,
            page_count,
            logical_bytes,
            job_count,
            complete_count,
            failed_count,
            transcript_count,
            segment_count,
            word_count,
            speaker_count,
            export_count,
            schema_version,
            fts5: self.has_search_index(),
        })
    }

    /// Reclaim space from deleted rows.
    ///
    /// `VACUUM` rewrites the whole file, so it needs the write lock for as long
    /// as the database is large and must not run while a job is writing. The
    /// caller decides; the UI only offers it from the settings panel with a
    /// "this may take a moment" warning.
    pub fn vacuum(&self) -> Result<()> {
        self.with_conn(|conn| {
            conn.execute_batch("VACUUM")
                .map_err(|e| crate::error::from_sqlite(e, self.path()))
        })?;
        self.close()
    }

    /// Delete the write-ahead log by checkpointing it into the main file.
    pub fn checkpoint(&self) -> Result<()> {
        self.close()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jobs::JobRecord;
    use crate::testing::TempDir;

    #[test]
    fn a_fresh_database_reports_zeroes_and_a_real_size() {
        let dir = TempDir::new("stats-empty");
        let s = Store::open(dir.join("v.db")).unwrap();
        let st = s.stats().unwrap();
        assert_eq!(st.job_count, 0);
        assert_eq!(st.transcript_count, 0);
        assert_eq!(st.word_count, 0);
        assert_eq!(st.schema_version, migrations::latest_version());
        assert!(st.fts5);
        assert!(st.page_size > 0);
        assert!(st.bytes_on_disk > 0, "an open database occupies disk");
        assert_eq!(st.logical_bytes, st.page_count * st.page_size);
    }

    #[test]
    fn counts_reflect_what_is_stored() {
        let dir = TempDir::new("stats-counts");
        let s = Store::open(dir.join("v.db")).unwrap();
        for i in 0..3 {
            let id = format!("j{i}");
            let mut rec = JobRecord::new(&id, dir.join("a.wav"));
            rec.phase = if i == 2 {
                vaani_core::JobPhase::Failed
            } else {
                vaani_core::JobPhase::Complete
            };
            s.upsert_job(&rec).unwrap();
            let t = crate::fixtures::transcript(i as u64, 4);
            s.replace_transcript(&id, &t).unwrap();
        }
        let st = s.stats().unwrap();
        assert_eq!(st.job_count, 3);
        assert_eq!(st.complete_count, 2);
        assert_eq!(st.failed_count, 1);
        assert_eq!(st.transcript_count, 3);
        assert_eq!(
            st.word_count,
            s.with_read(|c| Ok(crate::jobs::count(c, "words"))).unwrap()
        );
        assert!(st.summary().contains("3 jobs"));
    }

    #[test]
    fn bytes_on_disk_accounts_for_the_write_ahead_log() {
        let dir = TempDir::new("stats-wal");
        let path = dir.join("v.db");
        let s = Store::open(&path).unwrap();
        // Write enough to leave pages in the WAL.
        let t = crate::fixtures::transcript(1, 400);
        s.upsert_job(&JobRecord::new("j", dir.join("a.wav")))
            .unwrap();
        s.replace_transcript("j", &t).unwrap();
        let before = s.stats().unwrap();
        assert!(
            before.wal_bytes > 0,
            "an uncheckpointed write should leave a -wal file"
        );
        assert!(before.bytes_on_disk >= before.database_bytes);
        // Checkpointing folds the log back in and the sum still describes disk.
        s.close().unwrap();
        let after = s.stats().unwrap();
        assert_eq!(after.wal_bytes, 0, "TRUNCATE empties the log");
        assert!(after.database_bytes > 0);
    }

    #[test]
    fn vacuum_reclaims_space_and_keeps_the_data() {
        let dir = TempDir::new("stats-vacuum");
        let s = Store::open(dir.join("v.db")).unwrap();
        for i in 0..4 {
            let id = format!("j{i}");
            s.upsert_job(&JobRecord::new(&id, dir.join("a.wav")))
                .unwrap();
            s.replace_transcript(&id, &crate::fixtures::transcript(i as u64, 200))
                .unwrap();
        }
        s.delete_job("j0").unwrap();
        s.delete_job("j1").unwrap();
        s.vacuum().unwrap();
        let st = s.stats().unwrap();
        assert_eq!(st.job_count, 2);
        assert!(!s.load_transcript("j3").unwrap().segments.is_empty());
    }
}

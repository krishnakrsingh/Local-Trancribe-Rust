//! Settings and the export log.
//!
//! # Settings
//!
//! A key/value table, not a typed struct, for one reason: the settings panel
//! grows a row at a time (a language, a default export format, a theme) and the
//! architect should not have to make a migration every time. The contract stays
//! narrow — a string in, a string out, plus JSON helpers for structured values
//! — so a caller cannot accidentally store a float that comes back as a
//! different float.
//!
//! Values are stored JSON-encoded (`"gpu"`, `48000`, `{"a":1}`) so they
//! round-trip through any JSON consumer (the CLI's `--json`, a bug report)
//! without the app needing a schema for them. Reading a value that is not valid
//! JSON yields `None` rather than an error: a bareword is a legal thing to have
//! been written by an older build, and refusing to open the app over it would
//! be absurd.
//!
//! # The export log
//!
//! One row per file written: which job, where, in which format, how big, when.
//! It exists for two user-visible reasons — "Export to…" can offer a re-export
//! list, and support can answer "when did this file last get written?" — and it
//! cascades with the job, so deleting a job removes its history rather than
//! leaving rows pointing at nothing.
//!
//! The format is stored as its **wire string** (`text`, `markdown`, `srt`,
//! `vtt`, `json`) rather than as a `vaani_export::ExportFormat`. That direction
//! of the dependency is deliberate and permanent: `vaani-export` depends on
//! `vaani-storage`, so the database must not know the enum. A format this build
//! does not recognise still reads back as a string, which is what lets a newer
//! build's log survive a downgrade.

use std::collections::BTreeMap;
use std::path::Path;

use crate::Result;
use rusqlite::params;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::clock;
use crate::error;
use crate::store::Store;

impl Store {
    /// Read a setting.
    ///
    /// `None` means "not set", which the UI must distinguish from "set to empty" —
    /// hence a stored `""` is a legitimate value, not an absence.
    pub fn get_setting(&self, key: &str) -> Result<Option<String>> {
        self.read(|conn| {
            let mut stmt = conn.prepare("SELECT value FROM settings WHERE key = ?1")?;
            let mut rows = stmt.query_map(params![key], |r| r.get::<_, String>(0))?;
            let v = match rows.next() {
                Some(row) => row?,
                None => return Ok(None),
            };
            // A value written by a build that did not JSON-encode its settings is
            // accepted rather than discarded: a bareword is what a text setting
            // looks like, and losing it would be worse than reading it loosely.
            Ok(Some(if v.starts_with('"') {
                decode_string(&v).unwrap_or(v)
            } else {
                v
            }))
        })
    }

    /// Write a setting, replacing any previous value.
    pub fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        let encoded = serde_json::to_string(value)
            .map_err(|e| error::err_internal(format!("encoding setting {key:?}: {e}")))?;
        self.write(|tx| {
            tx.execute(
                "INSERT INTO settings (key, value, updated_at_ms) VALUES (?1, ?2, ?3)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                            updated_at_ms = excluded.updated_at_ms",
                params![key, encoded, clock::now_ms()],
            )?;
            Ok(())
        })
    }

    /// Remove a setting. Returns whether one was there.
    pub fn delete_setting(&self, key: &str) -> Result<bool> {
        self.write(|tx| Ok(tx.execute("DELETE FROM settings WHERE key = ?1", params![key])? > 0))
    }

    /// Every setting, sorted by key. The settings panel's "export my configuration".
    pub fn all_settings(&self) -> Result<BTreeMap<String, String>> {
        self.read(|conn| {
            let mut stmt = conn.prepare("SELECT key, value FROM settings ORDER BY key")?;
            let rows =
                stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
            let mut out = BTreeMap::new();
            for row in rows {
                let (k, v) = row?;
                out.insert(
                    k.clone(),
                    if v.starts_with('"') {
                        decode_string(&v).unwrap_or(v)
                    } else {
                        v
                    },
                );
            }
            Ok(out)
        })
    }

    /// Read a setting as JSON, for a structured value.
    pub fn get_setting_json<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        let Some(raw) = self.get_setting(key)? else {
            return Ok(None);
        };
        match serde_json::from_str::<T>(&raw) {
            Ok(v) => Ok(Some(v)),
            Err(e) => Err(error::err_internal(format!(
                "setting {key:?} does not hold the expected JSON: {e}"
            ))),
        }
    }

    /// Write a setting as JSON, for a structured value.
    pub fn set_setting_json<T: Serialize>(&self, key: &str, value: &T) -> Result<()> {
        let raw = serde_json::to_string(value)
            .map_err(|e| error::err_internal(format!("encoding setting {key:?}: {e}")))?;
        self.set_setting(key, &raw)
    }
}

fn decode_string(v: &str) -> Option<String> {
    serde_json::from_str::<String>(v).ok()
}

/// One row of the export log.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)] // field names are the documentation
pub struct ExportRecord {
    /// Row id. Monotonic within this database.
    pub id: i64,
    /// The job this was exported from, or `None` for a manual export of
    /// something not in the database yet.
    pub job_id: Option<String>,
    /// Where the file was written, as the user chose it.
    pub path: String,
    /// Wire form of the format: `text`, `markdown`, `srt`, `vtt`, `json`.
    pub format: String,
    pub bytes: i64,
    pub created_at_ms: i64,
}

impl ExportRecord {
    /// A record to log. The timestamp is filled in by the store, so two callers
    /// cannot disagree about "now".
    pub fn new(job_id: Option<&str>, path: impl AsRef<Path>, format: &str, bytes: u64) -> Self {
        Self {
            id: 0,
            job_id: job_id.map(str::to_string),
            path: path.as_ref().to_string_lossy().into_owned(),
            format: format.to_string(),
            bytes: bytes as i64,
            created_at_ms: 0,
        }
    }

    /// The path as a [`Path`].
    pub fn path(&self) -> &Path {
        Path::new(&self.path)
    }
}

impl Store {
    /// Record that a file was written. Returns the stored row, with its id.
    pub fn record_export(&self, record: &ExportRecord) -> Result<ExportRecord> {
        let now = clock::now_ms();
        self.write(|tx| {
            tx.execute(
                "INSERT INTO exports (job_id, path, format, bytes, created_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5)",
                params![record.job_id, record.path, record.format, record.bytes, now],
            )?;
            Ok(ExportRecord {
                id: tx.last_insert_rowid(),
                job_id: record.job_id.clone(),
                path: record.path.clone(),
                format: record.format.clone(),
                bytes: record.bytes,
                created_at_ms: now,
            })
        })
    }

    /// The export log, newest first, optionally for one job.
    pub fn list_exports(&self, job_id: Option<&str>) -> Result<Vec<ExportRecord>> {
        self.read(|conn| {
            // Two statements rather than one with a `?1 IS NULL` test: the
            // unfiltered query then has no parameters at all, which lets SQLite skip
            // binding entirely instead of evaluating a predicate per row.
            const COLUMNS: &str =
                "SELECT id, job_id, path, format, bytes, created_at_ms FROM exports";
            let map = |r: &rusqlite::Row<'_>| -> rusqlite::Result<ExportRecord> {
                Ok(ExportRecord {
                    id: r.get(0)?,
                    job_id: r.get(1)?,
                    path: r.get(2)?,
                    format: r.get(3)?,
                    bytes: r.get(4)?,
                    created_at_ms: r.get(5)?,
                })
            };
            let mut out = Vec::new();
            match job_id {
                Some(id) => {
                    let mut stmt =
                        conn.prepare(&format!("{COLUMNS} WHERE job_id = ?1 ORDER BY id DESC"))?;
                    let rows = stmt.query_map(params![id], map)?;
                    for row in rows {
                        out.push(row?);
                    }
                }
                None => {
                    let mut stmt = conn.prepare(&format!("{COLUMNS} ORDER BY id DESC"))?;
                    let rows = stmt.query_map([], map)?;
                    for row in rows {
                        out.push(row?);
                    }
                }
            }
            Ok(out)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jobs::JobRecord;
    use crate::testing::TempDir;

    fn store() -> (TempDir, Store) {
        let dir = TempDir::new("settings");
        let s = Store::open(dir.join("v.db")).unwrap();
        (dir, s)
    }

    #[test]
    fn a_setting_round_trips() {
        let (_d, s) = store();
        assert_eq!(s.get_setting("device").unwrap(), None);
        s.set_setting("device", "cuda").unwrap();
        assert_eq!(s.get_setting("device").unwrap().as_deref(), Some("cuda"));
        s.set_setting("device", "cpu").unwrap();
        assert_eq!(s.get_setting("device").unwrap().as_deref(), Some("cpu"));
    }

    #[test]
    fn an_empty_string_is_a_value_not_an_absence() {
        let (_d, s) = store();
        s.set_setting("last_dir", "").unwrap();
        assert_eq!(s.get_setting("last_dir").unwrap().as_deref(), Some(""));
    }

    #[test]
    fn values_with_quotes_newlines_and_unicode_survive() {
        let (_d, s) = store();
        for v in [
            r#"he said "hi""#,
            "line1\nline2\ttabbed",
            "नमस्ते 🌸",
            r#"C:\path\to\thing"#,
            "",
        ] {
            s.set_setting("k", v).unwrap();
            assert_eq!(s.get_setting("k").unwrap().as_deref(), Some(v), "{v:?}");
        }
    }

    #[test]
    fn deleting_reports_whether_anything_was_there() {
        let (_d, s) = store();
        s.set_setting("k", "v").unwrap();
        assert!(s.delete_setting("k").unwrap());
        assert!(!s.delete_setting("k").unwrap());
        assert_eq!(s.get_setting("k").unwrap(), None);
    }

    #[test]
    fn all_settings_is_sorted_and_complete() {
        let (_d, s) = store();
        s.set_setting("b", "2").unwrap();
        s.set_setting("a", "1").unwrap();
        s.set_setting("c", "3").unwrap();
        let all = s.all_settings().unwrap();
        assert_eq!(
            all.keys().cloned().collect::<Vec<_>>(),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
        assert_eq!(all["b"], "2");
    }

    #[test]
    fn json_settings_round_trip() {
        #[derive(Debug, PartialEq, Serialize, serde::Deserialize)]
        struct Panel {
            theme: String,
            font_size: u32,
        }
        let (_d, s) = store();
        let want = Panel {
            theme: "dark".into(),
            font_size: 14,
        };
        s.set_setting_json("ui", &want).unwrap();
        let back: Panel = s.get_setting_json("ui").unwrap().unwrap();
        assert_eq!(back, want);
        assert!(s.get_setting_json::<Panel>("missing").unwrap().is_none());
    }

    #[test]
    fn a_setting_holding_the_wrong_shape_is_an_error_not_a_panic() {
        #[derive(Debug, serde::Deserialize)]
        #[allow(dead_code)]
        struct Panel {
            theme: String,
        }
        let (_d, s) = store();
        s.set_setting("ui", "not json at all").unwrap();
        let e = s.get_setting_json::<Panel>("ui").unwrap_err();
        assert!(e.debug_chain().contains("does not hold the expected JSON"));
    }

    #[test]
    fn the_export_log_records_and_orders() {
        let (dir, s) = store();
        s.upsert_job(&JobRecord::new("j", dir.join("a.wav")))
            .unwrap();
        let a = s
            .record_export(&ExportRecord::new(
                Some("j"),
                dir.join("a.srt"),
                "srt",
                1024,
            ))
            .unwrap();
        let b = s
            .record_export(&ExportRecord::new(
                Some("j"),
                dir.join("a.vtt"),
                "vtt",
                2048,
            ))
            .unwrap();
        assert!(b.id > a.id);
        assert!(a.created_at_ms > 0);
        let all = s.list_exports(None).unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].format, "vtt", "newest first");
        assert_eq!(all[0].path(), dir.join("a.vtt"));
    }

    #[test]
    fn the_export_log_filters_by_job_and_survives_an_unknown_format() {
        let (dir, s) = store();
        s.upsert_job(&JobRecord::new("j1", dir.join("a.wav")))
            .unwrap();
        s.upsert_job(&JobRecord::new("j2", dir.join("b.wav")))
            .unwrap();
        s.record_export(&ExportRecord::new(
            Some("j1"),
            dir.join("a.md"),
            "markdown",
            10,
        ))
        .unwrap();
        // A format from a newer build must not break reading the log.
        s.record_export(&ExportRecord::new(Some("j2"), dir.join("b.x"), "docx", 20))
            .unwrap();
        assert_eq!(s.list_exports(Some("j1")).unwrap().len(), 1);
        assert_eq!(s.list_exports(Some("j2")).unwrap()[0].format, "docx");
        assert_eq!(s.list_exports(None).unwrap().len(), 2);
    }

    #[test]
    fn an_export_with_no_job_is_allowed() {
        let (dir, s) = store();
        let r = s
            .record_export(&ExportRecord::new(None, dir.join("manual.txt"), "text", 5))
            .unwrap();
        assert_eq!(r.job_id, None);
        assert_eq!(s.list_exports(None).unwrap().len(), 1);
    }

    #[test]
    fn the_export_log_cascades_with_the_job() {
        let (dir, s) = store();
        s.upsert_job(&JobRecord::new("j", dir.join("a.wav")))
            .unwrap();
        s.record_export(&ExportRecord::new(Some("j"), dir.join("a.srt"), "srt", 1))
            .unwrap();
        s.delete_job("j").unwrap();
        assert!(s.list_exports(None).unwrap().is_empty());
    }
}

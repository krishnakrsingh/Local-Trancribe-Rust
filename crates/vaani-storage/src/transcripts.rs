//! Persisting and reading back a [`Transcript`].
//!
//! # The one property that matters
//!
//! **What comes out of the database equals what went in.** Not "close enough":
//! every [`Time`], every confidence, every segment id, every speaker, in the same
//! order. Segment ids are documented in the contract crate as stable across
//! database round-trips because edits and the UI's selection model refer to
//! them; a store that renumbered them on save would break every edit a user had
//! made. So ids are stored, never regenerated, and the round trip is a property
//! test over hundreds of generated transcripts rather than one hand-written
//! case.
//!
//! # Times
//!
//! Every time is an `INTEGER` count of milliseconds in **source-media
//! coordinates**, exactly as [`vaani_core::Time`] holds it, including negative
//! values (`Time` is signed for a reason: ASR produces small negative starts
//! from chunk overlap). Nothing is reformatted, rounded, clamped or converted
//! on the way in or out. SQLite is not asked to interpret a time column; with
//! `STRICT` tables it *cannot* store a string in one, which turns the classic
//! "timestamps are strings now" bug into an error at write time.
//!
//! `duration_ms` is the one column that is **not** a position: it is a
//! **length** ([`vaani_core::MediaLength`]), so it is stored and read through
//! `as_millis` / `from_millis` and never compared against a segment or word
//! time. The distinction is a type, not a convention — see
//! `docs/architecture/TIMESTAMP_CONTRACT.md`.
//!
//! # Text
//!
//! `segments.text` stores the **derived** form, i.e. exactly what
//! [`vaani_core::Segment::text`] returns: the display text of its words joined
//! with single spaces. It is stored rather than computed on read for two
//! reasons: the search index needs one row per segment, and the job list needs a
//! preview without loading 20,000 words. It is never a second source of truth —
//! it is recomputed on every write from the words written in the same
//! transaction, and a test asserts that every loaded segment's derived text
//! still equals its stored text.
//!
//! # Atomicity
//!
//! [`Store::replace_transcript`] writes a transcript in a single `IMMEDIATE`
//! transaction: the old rows, the new segments, ~20,000 words, the speakers, the
//! search index and the job's counters either all become visible together or
//! none of them do. A half-written transcript is worse than no transcript,
//! because the user cannot tell which half they have.

use std::collections::HashMap;

use crate::Result;
use rusqlite::{Connection, params};
use vaani_core::{
    EngineStamp, MediaLength, Segment, Speaker, Time, Transcript, TranscriptMode, Word,
};

use crate::clock;
use crate::error;
use crate::store::Store;

/// What a write actually stored, for the pipeline's log line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(missing_docs)] // the three counts
pub struct TranscriptStats {
    pub segments: i64,
    pub words: i64,
    pub speakers: i64,
}

/// A transcript's metadata without its words, for list views.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)] // field names are the documentation
pub struct TranscriptSummary {
    pub transcript_id: String,
    pub mode: TranscriptMode,
    /// The dominant per-segment language, if the model identified one.
    /// Informational: a trilingual meeting has no single answer, which is why
    /// the per-segment tag in [`Transcript`] is the real data and this is a
    /// convenience for sorting and for the search facet.
    pub language: Option<String>,
    pub engine: EngineStamp,
    pub duration_ms: Option<i64>,
    pub edited: bool,
    pub created_at_ms: i64,
    pub segment_count: i64,
    pub word_count: i64,
}

/// A speaker as stored, including fields that are not in
/// [`vaani_core::Speaker`]: the generated label and the measured talk time.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)] // field names are the documentation
pub struct SpeakerRow {
    pub index: usize,
    pub display_name: String,
    pub custom_name: Option<String>,
    pub talk_time_ms: i64,
    pub share: Option<f32>,
}

impl SpeakerRow {
    /// What the UI shows: the rename if there is one, else the generated label.
    pub fn name(&self) -> &str {
        self.custom_name.as_deref().unwrap_or(&self.display_name)
    }

    /// The `vaani_core` view of this row.
    pub fn to_core(&self) -> Speaker {
        Speaker {
            index: self.index,
            name: self.custom_name.clone(),
            share: self.share,
        }
    }
}

fn no_such_job(job_id: &str) -> vaani_core::VaaniError {
    vaani_core::err::internal()
        .with_cause(format!(
            "cannot store a transcript for unknown job {job_id}"
        ))
        .with_recovery("This is a bug: a transcript was produced for a job that was never saved.")
}

/// A row that cannot be read is a corrupt-data error, not a generic one.
fn corrupt(message: String) -> rusqlite::Error {
    error::sqlite_corrupt(message)
}

/// The most frequent per-segment language, as the document's language.
///
/// Ties break on the language tag, ascending, so the result does not depend on
/// hash-map iteration order: a transcript that is half Hindi and half English
/// must store the same document language every time it is written.
fn dominant_language(t: &Transcript) -> Option<String> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for seg in &t.segments {
        if let Some(lang) = seg.language.as_deref() {
            *counts.entry(lang).or_default() += 1;
        }
    }
    counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(b.0)))
        .map(|(lang, _)| lang.to_string())
}

/// Total spoken time per speaker index, from the segments themselves.
fn talk_times(t: &Transcript) -> HashMap<usize, i64> {
    let mut out: HashMap<usize, i64> = HashMap::new();
    for seg in &t.segments {
        if let Some(s) = seg.speaker {
            *out.entry(s).or_default() += seg.duration().as_millis().max(0);
        }
    }
    out
}

/// The same text `vaani_core::Segment::text()` derives, recomputed locally so
/// the store never depends on a private method and so the invariant is visible
/// where it is enforced.
fn derive_text(words: &[Word]) -> String {
    let mut out = String::new();
    for (i, w) in words.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(w.display_text());
    }
    out
}

fn mode_wire(m: TranscriptMode) -> &'static str {
    match m {
        TranscriptMode::Raw => "raw",
        TranscriptMode::Clean => "clean",
        TranscriptMode::Structured => "structured",
    }
}

fn parse_mode(s: &str) -> rusqlite::Result<TranscriptMode> {
    match s {
        "raw" => Ok(TranscriptMode::Raw),
        "clean" => Ok(TranscriptMode::Clean),
        "structured" => Ok(TranscriptMode::Structured),
        other => Err(corrupt(format!(
            "transcripts.mode holds an unknown mode {other:?}"
        ))),
    }
}

impl Store {
    /// Replace the transcript for a job, atomically.
    ///
    /// The job must already exist: a transcript without a job has no input file,
    /// no duration and nothing to show in the list, so it is a programming error
    /// rather than something to paper over.
    ///
    /// Speaker renames survive. A user who has named "Speaker 2" Krishna and then
    /// re-runs diarization should not be asked again, so the previous
    /// `custom_name` for each cluster index is carried across.
    pub fn replace_transcript(&self, job_id: &str, t: &Transcript) -> Result<TranscriptStats> {
        let engine_json = serde_json::to_string(&t.engine)
            .map_err(|e| error::err_internal(format!("encoding EngineStamp: {e}")))?;
        let language = dominant_language(t);
        let talks = talk_times(t);
        let now = clock::now_ms();

        // Build every row *before* taking the write lock, so the lock is held for
        // the inserts and not for the string building.
        let mut segs: Vec<SegRow<'_>> = Vec::with_capacity(t.segments.len());
        let mut words: Vec<WordRow<'_>> =
            Vec::with_capacity(t.segments.iter().map(|s| s.words.len()).sum());
        for (ordinal, seg) in t.segments.iter().enumerate() {
            segs.push(SegRow {
                id: &seg.id,
                ordinal: ordinal as i64,
                start: seg.start.as_millis(),
                end: seg.end.as_millis(),
                speaker: seg.speaker.map(|s| s as i64),
                text: seg.text(),
                language: seg.language.as_deref(),
                confidence: seg.confidence.map(f64::from),
            });
            for (wi, w) in seg.words.iter().enumerate() {
                words.push(WordRow {
                    segment_id: &seg.id,
                    ordinal: wi,
                    text: &w.text,
                    normalized: w.normalized.as_deref(),
                    start: w.start.as_millis(),
                    end: w.end.as_millis(),
                    confidence: w.confidence.map(f64::from),
                });
            }
        }
        let speaker_rows: Vec<SpeakerWrite> = t
            .speakers
            .iter()
            .enumerate()
            .map(|(ordinal, spk)| SpeakerWrite {
                index: spk.index as i64,
                ordinal: ordinal as i64,
                display_name: spk.display_name(),
                name: spk.name.clone(),
                share: spk.share.map(f64::from),
                talk_time: talks.get(&spk.index).copied().unwrap_or(0),
            })
            .collect();
        let stats = TranscriptStats {
            segments: t.segments.len() as i64,
            words: words.len() as i64,
            speakers: speaker_rows.len() as i64,
        };

        // The job must exist: a transcript without a job has no input file, no
        // duration and nothing to show in the list, so it is a programming error
        // rather than something to paper over. Checked before the write so the
        // failure is a sentence rather than a constraint violation.
        if !self.job_exists(job_id)? {
            return Err(no_such_job(job_id));
        }

        self.write(|tx| {
            // Carry the user's renames across a re-transcription.
            let mut custom: HashMap<i64, String> = HashMap::new();
            {
                let mut stmt = tx.prepare(
                    "SELECT cluster_index, custom_name FROM speakers
                     WHERE job_id = ?1 AND custom_name IS NOT NULL",
                )?;
                let rows = stmt.query_map(params![job_id], |r| {
                    Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
                })?;
                for row in rows {
                    let (idx, name) = row?;
                    custom.insert(idx, name);
                }
            }

            tx.execute("DELETE FROM segment_fts WHERE job_id = ?1", params![job_id])?;
            // `words` goes with `segments` through the cascade; deleting the
            // parent explicitly is both correct and far faster than relying on
            // 20,000 row-by-row cascades. Speakers are replaced outright: a
            // re-transcription may find a different number of clusters, and the
            // renames read above are about to be re-applied.
            tx.execute("DELETE FROM segments WHERE job_id = ?1", params![job_id])?;
            tx.execute("DELETE FROM speakers WHERE job_id = ?1", params![job_id])?;

            {
                let mut stmt = tx.prepare(
                    "INSERT INTO segments
                        (id, job_id, ordinal, start_ms, end_ms, speaker_index, text, language, confidence)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                )?;
                for s in &segs {
                    stmt.execute(params![
                        s.id,
                        job_id,
                        s.ordinal,
                        s.start,
                        s.end,
                        s.speaker,
                        &s.text,
                        s.language,
                        s.confidence,
                    ])?;
                }
            }
            {
                let mut stmt = tx.prepare(
                    "INSERT INTO words
                        (job_id, segment_id, ordinal, text, normalized, start_ms, end_ms, confidence)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                )?;
                for w in &words {
                    stmt.execute(params![
                        job_id,
                        w.segment_id,
                        w.ordinal,
                        w.text,
                        w.normalized,
                        w.start,
                        w.end,
                        w.confidence,
                    ])?;
                }
            }
            {
                let mut stmt = tx.prepare(
                    "INSERT INTO speakers
                        (job_id, cluster_index, ordinal, display_name, custom_name, talk_time_ms, share)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                )?;
                for s in &speaker_rows {
                    stmt.execute(params![
                        job_id,
                        s.index,
                        s.ordinal,
                        &s.display_name,
                        // A name the user already gave this cluster wins: a
                        // re-transcription must not make them name Speaker 1
                        // again. With no stored name, the incoming transcript's
                        // own name is kept, so a transcript that arrives already
                        // named round-trips.
                        custom
                            .get(&s.index)
                            .cloned()
                            .or_else(|| s.name.clone()),
                        s.talk_time,
                        s.share,
                    ])?;
                }
            }
            {
                let mut stmt = tx.prepare(
                    "INSERT INTO segment_fts (job_id, segment_id, text) VALUES (?1, ?2, ?3)",
                )?;
                for s in &segs {
                    stmt.execute(params![job_id, s.id, &s.text])?;
                }
            }

            tx.execute(
                "INSERT INTO transcripts
                    (job_id, transcript_id, mode, language, engine_json, duration_ms, edited,
                     created_at_ms, segment_count, word_count)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)
                 ON CONFLICT(job_id) DO UPDATE SET
                     transcript_id = excluded.transcript_id,
                     mode = excluded.mode,
                     language = excluded.language,
                     engine_json = excluded.engine_json,
                     duration_ms = excluded.duration_ms,
                     edited = excluded.edited,
                     created_at_ms = excluded.created_at_ms,
                     segment_count = excluded.segment_count,
                     word_count = excluded.word_count",
                params![
                    job_id,
                    t.id,
                    mode_wire(t.mode),
                    language,
                    engine_json,
                    t.duration.map(MediaLength::as_millis),
                    i64::from(t.edited),
                    now,
                    stats.segments,
                    stats.words,
                ],
            )?;

            tx.execute(
                "UPDATE jobs SET segment_count = ?2, word_count = ?3, speaker_count = ?4
                 WHERE id = ?1",
                params![job_id, stats.segments, stats.words, stats.speakers],
            )?;
            if let Some(d) = t.duration.map(MediaLength::as_millis) {
                tx.execute(
                    "UPDATE jobs SET duration_ms = ?2 WHERE id = ?1 AND duration_ms IS NULL",
                    params![job_id, d],
                )?;
            }
            Ok(stats)
        })
    }

    /// Load a transcript, or `None` when the job has none yet.
    ///
    /// `None` is a normal state, not an error: a job that is still running has
    /// no transcript, and the UI shows a progress view rather than a failure.
    pub fn try_load_transcript(&self, job_id: &str) -> Result<Option<Transcript>> {
        self.read(|conn| load_transcript(conn, job_id))
    }

    /// Load a transcript, treating its absence as an error.
    ///
    /// For callers that have already established the job is complete; the UI's
    /// "open a finished job" path wants this, because a missing transcript for a
    /// completed job is a real fault.
    pub fn load_transcript(&self, job_id: &str) -> Result<Transcript> {
        self.try_load_transcript(job_id)?.ok_or_else(|| {
            vaani_core::err::internal()
                .with_cause(format!("no transcript is stored for job {job_id}"))
                .with_recovery("Re-run the job to produce a transcript for it.")
        })
    }

    /// A transcript's metadata, without loading its words.
    pub fn transcript_summary(&self, job_id: &str) -> Result<Option<TranscriptSummary>> {
        self.read(|conn| summary(conn, job_id))
    }

    /// A speaker's stored record, including the generated label and talk time.
    pub fn speaker(&self, job_id: &str, cluster_index: usize) -> Result<Option<SpeakerRow>> {
        self.read(|conn| {
            let mut stmt = conn.prepare(
                "SELECT cluster_index, display_name, custom_name, talk_time_ms, share
                 FROM speakers WHERE job_id = ?1 AND cluster_index = ?2",
            )?;
            let mut rows = stmt.query_map(params![job_id, cluster_index as i64], |r| {
                let share: Option<f64> = r.get(4)?;
                Ok(SpeakerRow {
                    index: r.get::<_, i64>(0)?.max(0) as usize,
                    display_name: r.get(1)?,
                    custom_name: r.get(2)?,
                    talk_time_ms: r.get(3)?,
                    share: share.map(|s| s as f32),
                })
            })?;
            match rows.next() {
                Some(row) => Ok(Some(row?)),
                None => Ok(None),
            }
        })
    }

    /// Rename a speaker, or clear the name by passing `None`.
    ///
    /// Only `custom_name` changes. The generated label ("Speaker 2") remains as
    /// the fallback and the transcript text is never touched: a rename is a
    /// label, not an edit, and the contract crate is explicit that the two must
    /// not be conflated.
    pub fn rename_speaker(
        &self,
        job_id: &str,
        cluster_index: usize,
        name: Option<&str>,
    ) -> Result<bool> {
        let trimmed = name.map(str::trim).filter(|n| !n.is_empty());
        self.write(|tx| {
            let changed = tx.execute(
                "UPDATE speakers SET custom_name = ?3 WHERE job_id = ?1 AND cluster_index = ?2",
                params![job_id, cluster_index as i64, trimmed],
            )?;
            // Naming a speaker the diarizer never produced is a no-op with a
            // clear signal, not a silent insert of a phantom speaker.
            Ok(changed > 0)
        })
    }

    /// Total stored words for a job, counted from the rows rather than read from
    /// the denormalised counter. The two must agree; a test asserts it.
    pub fn count_words(&self, job_id: &str) -> Result<i64> {
        self.read(|conn| {
            let n: i64 = conn.query_row(
                "SELECT COUNT(*) FROM words WHERE job_id = ?1",
                params![job_id],
                |r| r.get(0),
            )?;
            Ok(n)
        })
    }

    /// Total stored segments for a job.
    pub fn count_segments(&self, job_id: &str) -> Result<i64> {
        self.read(|conn| {
            let n: i64 = conn.query_row(
                "SELECT COUNT(*) FROM segments WHERE job_id = ?1",
                params![job_id],
                |r| r.get(0),
            )?;
            Ok(n)
        })
    }
}

struct SegRow<'a> {
    id: &'a str,
    ordinal: i64,
    start: i64,
    end: i64,
    speaker: Option<i64>,
    text: String,
    language: Option<&'a str>,
    confidence: Option<f64>,
}

struct WordRow<'a> {
    segment_id: &'a str,
    ordinal: usize,
    text: &'a str,
    normalized: Option<&'a str>,
    start: i64,
    end: i64,
    confidence: Option<f64>,
}

struct SpeakerWrite {
    index: i64,
    ordinal: i64,
    display_name: String,
    name: Option<String>,
    share: Option<f64>,
    talk_time: i64,
}

fn load_transcript(conn: &Connection, job_id: &str) -> rusqlite::Result<Option<Transcript>> {
    let header = {
        let mut stmt = conn.prepare(
            "SELECT transcript_id, mode, duration_ms, edited, engine_json
             FROM transcripts WHERE job_id = ?1",
        )?;
        let mut rows = stmt.query_map(params![job_id], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Option<i64>>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, String>(4)?,
            ))
        })?;
        match rows.next() {
            Some(row) => Some(row?),
            None => None,
        }
    };
    let Some((transcript_id, mode, duration_ms, edited, engine_json)) = header else {
        return Ok(None);
    };
    let mut t = Transcript {
        id: transcript_id,
        duration: duration_ms.map(MediaLength::from_millis),
        segments: Vec::new(),
        speakers: Vec::new(),
        mode: parse_mode(&mode)?,
        engine: serde_json::from_str(&engine_json)
            .map_err(|e| corrupt(format!("transcripts.engine_json is not valid: {e}")))?,
        edited: edited != 0,
    };

    // Words first, in one query in primary-key order: the (job, segment,
    // ordinal) primary key makes this an index scan with no sort, which is what
    // turns a 20,000-word load into one query instead of 1,800 round trips.
    let mut words_by_segment: HashMap<String, Vec<Word>> = HashMap::new();
    {
        let mut stmt = conn.prepare(
            "SELECT segment_id, text, normalized, start_ms, end_ms, confidence
             FROM words WHERE job_id = ?1 ORDER BY segment_id, ordinal",
        )?;
        let rows = stmt.query_map(params![job_id], |r| {
            let confidence: Option<f64> = r.get(5)?;
            Ok((
                r.get::<_, String>(0)?,
                Word {
                    text: r.get(1)?,
                    normalized: r.get(2)?,
                    start: Time::from_millis(r.get(3)?),
                    end: Time::from_millis(r.get(4)?),
                    confidence: confidence.map(|c| c as f32),
                },
            ))
        })?;
        for row in rows {
            let (segment_id, w) = row?;
            words_by_segment.entry(segment_id).or_default().push(w);
        }
    }

    {
        let mut stmt = conn.prepare(
            "SELECT id, start_ms, end_ms, speaker_index, text, language, confidence
             FROM segments WHERE job_id = ?1 ORDER BY ordinal",
        )?;
        let rows = stmt.query_map(params![job_id], |r| {
            let confidence: Option<f64> = r.get(6)?;
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, Option<i64>>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, Option<String>>(5)?,
                confidence,
            ))
        })?;
        for row in rows {
            let (id, start, end, speaker, stored_text, language, confidence) = row?;
            let words = words_by_segment.remove(&id).unwrap_or_default();
            // `segments.text` is a cache of `Segment::text()`. If the two ever
            // disagree the words are the truth — they are what the user reads
            // and edits — so say so loudly and keep the words.
            if derive_text(&words) != stored_text {
                tracing::warn!(
                    segment = %id,
                    "segments.text disagrees with its words; the words win"
                );
            }
            t.segments.push(Segment {
                id,
                start: Time::from_millis(start),
                end: Time::from_millis(end),
                speaker: speaker.and_then(|s| usize::try_from(s).ok()),
                language,
                words,
                confidence: confidence.map(|c| c as f32),
            });
        }
    }

    {
        let mut stmt = conn.prepare(
            "SELECT cluster_index, custom_name, share FROM speakers
             WHERE job_id = ?1 ORDER BY ordinal",
        )?;
        let rows = stmt.query_map(params![job_id], |r| {
            let share: Option<f64> = r.get(2)?;
            Ok(Speaker {
                index: r.get::<_, i64>(0)?.max(0) as usize,
                name: r.get(1)?,
                share: share.map(|s| s as f32),
            })
        })?;
        for row in rows {
            t.speakers.push(row?);
        }
    }
    Ok(Some(t))
}

fn summary(conn: &Connection, job_id: &str) -> rusqlite::Result<Option<TranscriptSummary>> {
    let mut stmt = conn.prepare(
        "SELECT transcript_id, mode, language, duration_ms, edited, created_at_ms,
                engine_json, segment_count, word_count
         FROM transcripts WHERE job_id = ?1",
    )?;
    let mut rows = stmt.query_map(params![job_id], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, Option<String>>(2)?,
            r.get::<_, Option<i64>>(3)?,
            r.get::<_, i64>(4)?,
            r.get::<_, i64>(5)?,
            r.get::<_, String>(6)?,
            r.get::<_, i64>(7)?,
            r.get::<_, i64>(8)?,
        ))
    })?;
    let Some(row) = rows.next() else {
        return Ok(None);
    };
    let row = row?;
    Ok(Some(TranscriptSummary {
        transcript_id: row.0,
        mode: parse_mode(&row.1)?,
        language: row.2,
        duration_ms: row.3,
        edited: row.4 != 0,
        created_at_ms: row.5,
        engine: serde_json::from_str(&row.6)
            .map_err(|e| corrupt(format!("transcripts.engine_json is not valid: {e}")))?,
        segment_count: row.7,
        word_count: row.8,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jobs::JobRecord;
    use crate::testing::TempDir;

    fn store_with_job(id: &str) -> (TempDir, Store) {
        let dir = TempDir::new("transcripts");
        let s = Store::open(dir.join("v.db")).unwrap();
        s.upsert_job(&JobRecord::new(id, dir.join("a.wav")))
            .unwrap();
        (dir, s)
    }

    fn simple() -> Transcript {
        let mut t = Transcript::new(EngineStamp::asr("sravaani", "1.0"));
        t.duration = Some(MediaLength::from_millis(5_000));
        t.segments.push(
            Segment::new(Time(0), Time(1_000))
                .with_words(vec![Word::new("hello", Time(0), Time(500))])
                .with_speaker(Some(0)),
        );
        t.speakers.push(Speaker::new(0));
        t
    }

    #[test]
    fn a_transcript_round_trips_exactly() {
        let (_d, s) = store_with_job("j");
        let t = simple();
        s.replace_transcript("j", &t).unwrap();
        assert_eq!(s.load_transcript("j").unwrap(), t);
    }

    #[test]
    fn an_unknown_job_is_refused_with_a_clear_cause() {
        let (_d, s) = store_with_job("j");
        let e = s.replace_transcript("ghost", &simple()).unwrap_err();
        assert!(e.debug_chain().contains("unknown job ghost"));
    }

    #[test]
    fn replacing_is_a_swap_not_an_append() {
        let (_d, s) = store_with_job("j");
        s.replace_transcript("j", &simple()).unwrap();
        let mut bigger = simple();
        bigger.segments.push(Segment::new(Time(1_000), Time(2_000)));
        s.replace_transcript("j", &bigger).unwrap();
        assert_eq!(s.load_transcript("j").unwrap().segments.len(), 2);
        assert_eq!(s.count_segments("j").unwrap(), 2);
    }

    #[test]
    fn a_failed_write_leaves_the_previous_transcript_intact() {
        let (_d, s) = store_with_job("j");
        s.replace_transcript("j", &simple()).unwrap();
        // Two segments sharing an id violate the primary key half way through the
        // write, so the transaction must roll back to the first transcript.
        let mut broken = Transcript::new(EngineStamp::asr("m", "1"));
        broken.segments = vec![
            Segment::new(Time(0), Time(1)),
            Segment::new(Time(1), Time(2)),
        ];
        broken.segments[1].id = broken.segments[0].id.clone();
        assert!(s.replace_transcript("j", &broken).is_err());
        let back = s.load_transcript("j").unwrap();
        assert_eq!(
            back.segments.len(),
            1,
            "the first transcript survived intact"
        );
    }

    #[test]
    fn speaker_renames_survive_a_re_transcription() {
        let (_d, s) = store_with_job("j");
        s.replace_transcript("j", &simple()).unwrap();
        assert!(s.rename_speaker("j", 0, Some("Krishna")).unwrap());
        let mut again = simple();
        again.id = "different-id".into();
        s.replace_transcript("j", &again).unwrap();
        assert_eq!(
            s.load_transcript("j").unwrap().speakers[0].name.as_deref(),
            Some("Krishna")
        );
        let row = s.speaker("j", 0).unwrap().unwrap();
        assert_eq!(row.display_name, "Speaker 1");
        assert_eq!(row.name(), "Krishna");
    }

    #[test]
    fn talk_time_is_measured_from_the_segments() {
        let (_d, s) = store_with_job("j");
        let t = crate::fixtures::transcript(3, 6);
        s.replace_transcript("j", &t).unwrap();
        let mut total = 0;
        for (i, spk) in t.speakers.iter().enumerate() {
            let row = s.speaker("j", spk.index).unwrap().unwrap();
            total += row.talk_time_ms;
            let _ = i;
        }
        let spoken: i64 = t
            .segments
            .iter()
            .filter(|s| s.speaker.is_some())
            .map(|s| s.duration().as_millis())
            .sum();
        assert_eq!(total, spoken);
    }

    #[test]
    fn renaming_a_speaker_who_does_not_exist_is_false() {
        let (_d, s) = store_with_job("j");
        s.replace_transcript("j", &simple()).unwrap();
        assert!(!s.rename_speaker("j", 9, Some("Nobody")).unwrap());
    }

    #[test]
    fn clearing_a_name_falls_back_to_the_generated_label() {
        let (_d, s) = store_with_job("j");
        s.replace_transcript("j", &simple()).unwrap();
        s.rename_speaker("j", 0, Some("Krishna")).unwrap();
        s.rename_speaker("j", 0, None).unwrap();
        assert_eq!(s.load_transcript("j").unwrap().speakers[0].name, None);
    }

    #[test]
    fn a_job_without_a_transcript_is_none_not_an_error() {
        let (_d, s) = store_with_job("j");
        assert!(s.try_load_transcript("j").unwrap().is_none());
        assert!(s.transcript_summary("j").unwrap().is_none());
        let e = s.load_transcript("j").unwrap_err();
        assert!(e.debug_chain().contains("no transcript"));
    }

    #[test]
    fn the_summary_matches_the_transcript_without_loading_words() {
        let (_d, s) = store_with_job("j");
        let mut t = simple();
        t.mode = TranscriptMode::Clean;
        t.edited = true;
        s.replace_transcript("j", &t).unwrap();
        let sum = s.transcript_summary("j").unwrap().unwrap();
        assert_eq!(sum.mode, TranscriptMode::Clean);
        assert!(sum.edited);
        assert_eq!(sum.segment_count, 1);
        assert_eq!(sum.word_count, 1);
        assert_eq!(sum.duration_ms, Some(5_000));
        assert_eq!(sum.engine, t.engine);
    }

    #[test]
    fn counters_agree_with_the_rows() {
        let (_d, s) = store_with_job("j");
        let t = crate::fixtures::transcript(7, 11);
        let stats = s.replace_transcript("j", &t).unwrap();
        assert_eq!(stats.words, s.count_words("j").unwrap());
        assert_eq!(stats.segments, s.count_segments("j").unwrap());
        let job = s.load_job("j").unwrap().unwrap();
        assert_eq!(job.word_count, stats.words);
        assert_eq!(job.segment_count, stats.segments);
        assert_eq!(job.speaker_count, stats.speakers);
    }

    #[test]
    fn stored_segment_text_always_equals_the_derived_text() {
        let (_d, s) = store_with_job("j");
        let t = crate::fixtures::transcript(5, 9);
        s.replace_transcript("j", &t).unwrap();
        s.read(|conn| {
            let mut stmt =
                conn.prepare("SELECT text FROM segments WHERE job_id = 'j' ORDER BY ordinal")?;
            let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
            let mut i = 0;
            for row in rows {
                assert_eq!(row?, t.segments[i].text(), "row {i} disagrees");
                i += 1;
            }
            assert_eq!(i, t.segments.len());
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn the_dominant_segment_language_is_stable() {
        let mut t = Transcript::new(EngineStamp::asr("m", "1"));
        t.segments = vec![
            Segment::new(Time(0), Time(1)).with_speaker(Some(0)),
            Segment::new(Time(1), Time(2)).with_speaker(Some(1)),
        ];
        t.segments[0].language = Some("hi".into());
        t.segments[1].language = Some("en".into());
        let (_d, s) = store_with_job("j");
        s.replace_transcript("j", &t).unwrap();
        // A tie must not depend on iteration order: the tag breaks it.
        for _ in 0..8 {
            assert_eq!(
                s.transcript_summary("j")
                    .unwrap()
                    .unwrap()
                    .language
                    .as_deref(),
                Some("hi")
            );
        }
    }
}

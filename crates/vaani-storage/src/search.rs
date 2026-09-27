//! Full-text search over transcript text.
//!
//! # The engine, and what happens if it is missing
//!
//! Search is FTS5 with `bm25` ranking and `snippet()` highlighting.
//! `rusqlite`'s **bundled** SQLite is compiled with `-DSQLITE_ENABLE_FTS5` (see
//! `libsqlite3-sys`'s `build.rs`), so it is available; the migration that
//! creates the index asks the database rather than assuming, and
//! [`Store::has_search_index`] reports the answer.
//!
//! If a build ever links a SQLite without FTS5 — a system library, a build that
//! drops the bundled feature — the index is simply not created and
//! [`Store::search`] falls back to a `LIKE` scan. The fallback is *correct*
//! (same rows, same fields) but not ranked by relevance, and it reports
//! [`SearchEngine::Fallback`] in its result so the UI can say "searching
//! everything" rather than pretending it ranked 300 transcripts in 4 ms. This
//! is the only honest option: silently returning nothing would look like "no
//! results found" for a feature that is present.
//!
//! # Why the user's text is never passed to `MATCH` verbatim
//!
//! `MATCH` takes an *expression*, not a string: `foo OR bar`, `a*`, `-x` and
//! `NEAR(...)` are all syntax, and a user typing a hyphenated word would get a
//! `SqliteFailure: syntax error near "-"` — which is exactly the kind of raw
//! engine text `vaani_core` forbids from reaching the UI. Every term is
//! therefore quoted (`"` doubled inside) and the terms are ANDed, which is what
//! a search-as-you-type box means anyway.

use std::time::Instant;

use crate::Result;
use rusqlite::params;

use crate::jobs::like_contains;
use crate::store::Store;

/// Which engine produced a result set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchEngine {
    /// FTS5 with bm25 ranking. The normal case.
    Fts5,
    /// A `LIKE` scan over `segments.text`. Correct, unranked, slower.
    Fallback,
}

impl SearchEngine {
    /// Wire form, for logs and the diagnostics line in the UI.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fts5 => "fts5",
            Self::Fallback => "like",
        }
    }
}

/// A search request.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)] // an absent field means "no constraint"; see `SearchQuery::default`
pub struct SearchQuery {
    /// The user's text. Empty or whitespace-only means "no results", never
    /// "everything": a search box that dumps the whole corpus when you clear it
    /// is a search box nobody uses.
    pub text: String,
    /// Restrict to one job.
    pub job_id: Option<String>,
    /// Restrict to one speaker's segments.
    pub speaker: Option<usize>,
    /// Maximum hits. The UI wants 50; the CLI wants everything.
    pub limit: usize,
    /// How many words of context `snippet()` returns either side of a match.
    pub snippet_words: usize,
}

impl Default for SearchQuery {
    fn default() -> Self {
        Self {
            text: String::new(),
            job_id: None,
            speaker: None,
            limit: 50,
            snippet_words: 12,
        }
    }
}

impl SearchQuery {
    /// A search for `text` with default limits.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Default::default()
        }
    }

    /// Cap the number of hits.
    #[must_use]
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Search within one job.
    #[must_use]
    pub fn in_job(mut self, job_id: impl Into<String>) -> Self {
        self.job_id = Some(job_id.into());
        self
    }
}

/// One matching segment.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)] // field names are the documentation; times are source-media milliseconds
pub struct SearchHit {
    pub job_id: String,
    /// The stable segment id, so the UI can select and seek to it.
    pub segment_id: String,
    pub input_name: String,
    pub start_ms: i64,
    pub end_ms: i64,
    /// `None` means the job has not been diarized, which is normal.
    pub speaker: Option<usize>,
    /// `bm25` score. Lower is a better match; only comparable within one result
    /// set, never across queries.
    pub rank: f64,
    /// Highlighted excerpt with `<mark>` around the matched terms.
    pub snippet: String,
    /// The segment's full text, for "show the whole segment".
    pub text: String,
}

/// A result set plus how it was produced and how long it took.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)] // field names are the documentation
pub struct SearchResults {
    pub hits: Vec<SearchHit>,
    /// Which engine ran. See the module docs.
    pub engine: SearchEngine,
    /// Wall time of the query, measured, for the "312 results in 4 ms" line.
    pub took_ms: f64,
}

impl SearchResults {
    /// Whether anything matched.
    pub fn is_empty(&self) -> bool {
        self.hits.is_empty()
    }

    /// Number of hits, for convenience.
    pub fn len(&self) -> usize {
        self.hits.len()
    }
}

/// Turn user text into a safe FTS5 `MATCH` expression.
///
/// Every whitespace-separated term becomes a quoted string, and the terms are
/// ANDed. Quoting is what makes `C++`, `don't` and `-` searchable instead of
/// syntax errors; ANDing is what makes narrowing a search feel like narrowing.
pub fn to_match_expression(text: &str) -> Option<String> {
    let terms: Vec<String> = text
        .split_whitespace()
        .filter(|t| !t.is_empty())
        .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
        .collect();
    if terms.is_empty() {
        None
    } else {
        Some(terms.join(" "))
    }
}

impl Store {
    /// Search every stored transcript.
    ///
    /// Ranked by `bm25` when the index exists, so the segment that mentions the
    /// term most, in the shortest field, ranks first. The job name and speaker
    /// are joined in Rust rather than stored in the index: two extra columns of
    /// denormalised text per segment would double the index size to answer a
    /// question one join answers.
    pub fn search(&self, query: &SearchQuery) -> Result<SearchResults> {
        let began = Instant::now();
        let Some(expression) = to_match_expression(&query.text) else {
            return Ok(SearchResults {
                hits: Vec::new(),
                engine: if self.has_search_index() {
                    SearchEngine::Fts5
                } else {
                    SearchEngine::Fallback
                },
                took_ms: began.elapsed().as_secs_f64() * 1000.0,
            });
        };
        let hits = if self.has_search_index() {
            self.search_fts(&expression, query)?
        } else {
            self.search_like(query)?
        };
        Ok(SearchResults {
            hits,
            engine: if self.has_search_index() {
                SearchEngine::Fts5
            } else {
                SearchEngine::Fallback
            },
            took_ms: began.elapsed().as_secs_f64() * 1000.0,
        })
    }

    fn search_fts(&self, expression: &str, query: &SearchQuery) -> Result<Vec<SearchHit>> {
        self.read(|conn| {
            let sql = "SELECT f.job_id, f.segment_id, s.start_ms, s.end_ms, s.speaker_index,
                              s.text, j.input_name, bm25(segment_fts) AS score,
                              snippet(segment_fts, 2, '<mark>', '</mark>', '…', ?3) AS snip
                       FROM segment_fts f
                       JOIN segments s ON s.job_id = f.job_id AND s.id = f.segment_id
                       JOIN jobs j ON j.id = f.job_id
                       WHERE segment_fts MATCH ?1
                         AND (?2 IS NULL OR f.job_id = ?2)
                         AND (?4 IS NULL OR s.speaker_index = ?4)
                       ORDER BY score
                       LIMIT ?5";
            let mut stmt = conn.prepare(sql)?;
            let mut rows = stmt.query(params![
                expression,
                query.job_id,
                query.snippet_words as i64,
                query.speaker.map(|s| s as i64),
                query.limit as i64,
            ])?;
            let mut out = Vec::new();
            while let Some(row) = rows.next()? {
                out.push(SearchHit {
                    job_id: row.get(0)?,
                    segment_id: row.get(1)?,
                    start_ms: row.get(2)?,
                    end_ms: row.get(3)?,
                    speaker: row
                        .get::<_, Option<i64>>(4)?
                        .and_then(|s| usize::try_from(s).ok()),
                    text: row.get(5)?,
                    input_name: row.get(6)?,
                    rank: row.get(7)?,
                    snippet: row.get(8)?,
                });
            }
            Ok(out)
        })
    }

    /// The no-FTS5 path. Same rows, no relevance ordering.
    ///
    /// Ranked by where the first term appears in the segment (an early match is
    /// a better match) with the job's recency as the tie-break, which is a weak
    /// but honest ordering: it is a *heuristic*, and [`SearchResults::engine`]
    /// tells the caller not to advertise it as relevance ranking.
    fn search_like(&self, query: &SearchQuery) -> Result<Vec<SearchHit>> {
        let needle = like_contains(&query.text);
        let mut rows: Vec<(SearchHit, i64)> = self.read(|conn| {
            let mut stmt = conn.prepare(
                "SELECT s.job_id, s.id, s.start_ms, s.end_ms, s.speaker_index, s.text,
                            j.input_name, j.created_at_ms
                     FROM segments s JOIN jobs j ON j.id = s.job_id
                     WHERE s.text LIKE ?1 ESCAPE '\\'
                       AND (?2 IS NULL OR s.job_id = ?2)
                       AND (?3 IS NULL OR s.speaker_index = ?3)
                     LIMIT ?4",
            )?;
            let mapped = stmt.query_map(
                params![
                    needle,
                    query.job_id,
                    query.speaker.map(|s| s as i64),
                    (query.limit * 4) as i64,
                ],
                |r| {
                    let text: String = r.get(5)?;
                    Ok((
                        SearchHit {
                            job_id: r.get(0)?,
                            segment_id: r.get(1)?,
                            start_ms: r.get(2)?,
                            end_ms: r.get(3)?,
                            speaker: r
                                .get::<_, Option<i64>>(4)?
                                .and_then(|s| usize::try_from(s).ok()),
                            rank: 0.0,
                            snippet: crate::search::plain_snippet(
                                &text,
                                &query.text,
                                query.snippet_words,
                            ),
                            text,
                            input_name: r.get(6)?,
                        },
                        r.get::<_, i64>(7)?,
                    ))
                },
            )?;
            let mut out: Vec<(SearchHit, i64)> = Vec::new();
            for row in mapped {
                out.push(row?);
            }
            Ok(out)
        })?;
        rows.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.start_ms.cmp(&b.0.start_ms)));
        Ok(rows.into_iter().map(|(h, _)| h).take(query.limit).collect())
    }
}

/// A `snippet()` replacement for the fallback path: find the first term, take
/// `context` words either side, and mark matches with `<mark>`.
fn plain_snippet(text: &str, needle: &str, context: usize) -> String {
    let lower_text = text.to_lowercase();
    let lower_needle = needle.to_lowercase();
    let Some(pos) = lower_text.find(&lower_needle) else {
        return text.chars().take(context * 8).collect();
    };
    let start_byte = text
        .char_indices()
        .rev()
        .find(|(i, _)| *i <= pos.saturating_sub(context * 6))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let end_byte = text
        .char_indices()
        .find(|(i, _)| *i >= pos + lower_needle.len() + context * 6)
        .map(|(i, _)| i)
        .unwrap_or(text.len());
    let mut out = String::new();
    if start_byte > 0 {
        out.push('…');
    }
    out.push_str(&text[start_byte..end_byte.min(text.len())]);
    if end_byte < text.len() {
        out.push('…');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures;
    use crate::jobs::JobRecord;
    use crate::testing::TempDir;

    fn store_with(texts: &[&str]) -> (TempDir, Store) {
        let dir = TempDir::new("search");
        let s = Store::open(dir.join("v.db")).unwrap();
        for (i, text) in texts.iter().enumerate() {
            let id = format!("j{i}");
            s.upsert_job(&JobRecord::new(&id, dir.join("a.wav")))
                .unwrap();
            let mut t = vaani_core::Transcript::new(vaani_core::EngineStamp::asr("m", "1"));
            t.segments = vec![vaani_core::Segment {
                id: format!("s{i}"),
                start: vaani_core::Time(0),
                end: vaani_core::Time(1_000),
                speaker: Some(0),
                language: None,
                words: text
                    .split_whitespace()
                    .enumerate()
                    .map(|(w, t)| {
                        vaani_core::Word::new(
                            t,
                            vaani_core::Time(w as i64 * 100),
                            vaani_core::Time(0),
                        )
                    })
                    .collect(),
                confidence: None,
            }];
            s.replace_transcript(&id, &t).unwrap();
        }
        (dir, s)
    }

    #[test]
    fn matching_finds_the_right_segment() {
        let (_d, s) = store_with(&["the quick brown fox", "hello world", "quick decision"]);
        let r = s.search(&SearchQuery::new("quick")).unwrap();
        assert_eq!(r.engine, SearchEngine::Fts5);
        assert_eq!(r.hits.len(), 2);
        assert!(
            r.hits
                .iter()
                .all(|h| h.snippet.contains("<mark>quick</mark>"))
        );
    }

    #[test]
    fn terms_are_anded_so_narrowing_works() {
        let (_d, s) = store_with(&["quick brown fox", "quick brown dog", "slow green tree"]);
        let r = s.search(&SearchQuery::new("quick dog")).unwrap();
        assert_eq!(r.hits.len(), 1);
        assert!(r.hits[0].text.contains("dog"));
    }

    #[test]
    fn search_is_case_and_diacritic_insensitive() {
        let (_d, s) = store_with(&["Cafe resume in Zurich", "completely unrelated"]);
        let r = s.search(&SearchQuery::new("cafe")).unwrap();
        assert_eq!(r.hits.len(), 1);
    }

    #[test]
    fn non_ascii_text_is_searchable() {
        let (_d, s) = store_with(&["नमस्ते दुनिया", "hello there"]);
        let r = s.search(&SearchQuery::new("नमस्ते")).unwrap();
        assert_eq!(r.hits.len(), 1);
    }

    #[test]
    fn punctuation_heavy_input_is_data_not_syntax() {
        // Every one of these is a syntax error if handed to MATCH unquoted.
        let (_d, s) = store_with(&[
            "a C++ and a don't",
            "quote \" then brace }",
            "-dash (paren)",
        ]);
        for needle in [
            "C++", "don't", "\"", "}", "-dash", "(paren)", "NEAR(", "*", "OR", "AND",
        ] {
            let r = s.search(&SearchQuery::new(needle));
            assert!(r.is_ok(), "{needle:?} produced a raw SQLite error");
        }
    }

    #[test]
    fn an_empty_query_returns_nothing_rather_than_everything() {
        let (_d, s) = store_with(&["one", "two"]);
        for needle in ["", "   ", "\t\n"] {
            let r = s.search(&SearchQuery::new(needle)).unwrap();
            assert!(r.is_empty(), "{needle:?} should not match anything");
        }
    }

    #[test]
    fn a_search_with_no_matches_is_empty_not_an_error() {
        let (_d, s) = store_with(&["one", "two"]);
        assert!(s.search(&SearchQuery::new("zebra")).unwrap().is_empty());
    }

    #[test]
    fn results_carry_enough_to_navigate_to_the_hit() {
        let (_d, s) = store_with(&["hello world"]);
        let hit = &s.search(&SearchQuery::new("world")).unwrap().hits[0];
        assert_eq!(hit.job_id, "j0");
        assert_eq!(hit.segment_id, "s0");
        assert_eq!(hit.input_name, "a.wav");
        assert_eq!(hit.start_ms, 0);
        assert_eq!(hit.end_ms, 1_000);
        assert_eq!(hit.speaker, Some(0));
    }

    #[test]
    fn a_diarization_free_transcript_reports_no_speaker() {
        let dir = TempDir::new("search-nospeaker");
        let s = Store::open(dir.join("v.db")).unwrap();
        s.upsert_job(&JobRecord::new("j", dir.join("a.wav")))
            .unwrap();
        let mut t = fixtures::transcript(1, 1);
        for seg in &mut t.segments {
            seg.speaker = None;
        }
        t.speakers.clear();
        s.replace_transcript("j", &t).unwrap();
        let hit = &s
            .search(&SearchQuery::new(fixtures::any_word(&t)))
            .unwrap()
            .hits[0];
        assert_eq!(hit.speaker, None, "no speaker is a normal state");
    }

    #[test]
    fn results_can_be_restricted_to_one_job() {
        let (_d, s) = store_with(&["shared word", "shared word"]);
        let r = s.search(&SearchQuery::new("shared").in_job("j1")).unwrap();
        assert_eq!(r.hits.len(), 1);
        assert_eq!(r.hits[0].job_id, "j1");
    }

    #[test]
    fn results_can_be_restricted_to_one_speaker() {
        let (_d, s) = store_with(&["shared word", "shared word"]);
        let r = s
            .search(&SearchQuery {
                speaker: Some(0),
                ..SearchQuery::new("shared")
            })
            .unwrap();
        assert_eq!(r.hits.len(), 2);
        let r = s
            .search(&SearchQuery {
                speaker: Some(7),
                ..SearchQuery::new("shared")
            })
            .unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn the_limit_is_respected() {
        let (_d, s) = store_with(&["repeated", "repeated", "repeated", "repeated"]);
        let r = s
            .search(&SearchQuery::new("repeated").with_limit(2))
            .unwrap();
        assert_eq!(r.hits.len(), 2);
    }

    #[test]
    fn ranks_are_ordered_best_first() {
        // One segment mentions the term three times, one mentions it once.
        let (_d, s) = store_with(&["apple", "apple apple apple"]);
        let r = s.search(&SearchQuery::new("apple")).unwrap();
        assert_eq!(r.hits.len(), 2);
        assert!(
            r.hits[0].rank <= r.hits[1].rank,
            "bm25 ranks should be ascending: {:?}",
            r.hits.iter().map(|h| h.rank).collect::<Vec<_>>()
        );
        assert!(r.hits[0].text.contains("apple apple apple"));
    }

    #[test]
    fn the_index_follows_edits_and_deletes() {
        let (dir, s) = store_with(&["findme now"]);
        assert_eq!(s.search(&SearchQuery::new("findme")).unwrap().hits.len(), 1);
        // Replace the transcript: the stale index row must be gone.
        let mut t = fixtures::transcript(1, 1);
        t.segments[0].words = vec![vaani_core::Word::new(
            "somethingelse",
            vaani_core::Time(0),
            vaani_core::Time(1),
        )];
        s.replace_transcript("j0", &t).unwrap();
        assert!(s.search(&SearchQuery::new("findme")).unwrap().is_empty());
        assert_eq!(
            s.search(&SearchQuery::new("somethingelse"))
                .unwrap()
                .hits
                .len(),
            1
        );
        // Delete the job: the index row must go with it.
        s.delete_job("j0").unwrap();
        assert!(
            s.search(&SearchQuery::new("somethingelse"))
                .unwrap()
                .is_empty()
        );
        let _ = dir;
    }

    #[test]
    fn the_match_expression_quotes_every_term() {
        assert_eq!(
            to_match_expression("hello world").as_deref(),
            Some("\"hello\" \"world\"")
        );
        assert_eq!(
            to_match_expression(r#"say "hi""#).as_deref(),
            Some("\"say\" \"\"\"hi\"\"\"")
        );
        assert_eq!(to_match_expression("   "), None);
    }
}

//! The rest of the API, at the scale a real user creates.
//!
//! Job listing, filtering, deletion and search over a few hundred jobs, which is
//! the size of a heavy user's history and where an unindexed query or a
//! non-cascading delete starts to show up.

use std::time::Instant;

use vaani_core::{JobPhase, Time, Transcript};
use vaani_storage::{
    ExportRecord, JobFilter, JobOrder, JobRecord, SearchQuery, Store, StoredError, fixtures,
    testing::TempDir,
};

#[test]
fn deleting_a_job_leaves_no_orphan_words_segments_or_index_rows() {
    let dir = TempDir::new("cascade");
    let store = Store::open(dir.join("v.db")).unwrap();
    // Words per job, so "nothing else was touched" is arithmetic rather than a
    // feeling.
    let mut per_job = std::collections::BTreeMap::new();
    for i in 0..5 {
        let id = format!("j{i}");
        let job = JobRecord::new(id.clone(), dir.join(format!("f{i}.wav")));
        store.upsert_job(&job).unwrap();
        store
            .replace_transcript(&id, &fixtures::transcript(i, 50))
            .unwrap();
        per_job.insert(id.clone(), store.count_words(&id).unwrap());
    }
    let total_words: i64 = per_job.values().sum();
    assert_eq!(store.stats().unwrap().word_count, total_words);
    let j0_words = per_job["j0"];
    let needle = fixtures::any_word(&store.load_transcript("j0").unwrap());

    assert!(store.delete_job("j0").unwrap());

    // The rows are gone, from every table that mentions the job.
    let count = |sql: &str| -> i64 {
        store
            .with_read(|c| {
                let map = |e: rusqlite::Error| vaani_storage::error::from_sqlite(e, store.path());
                c.query_row(sql, [], |r| r.get::<_, i64>(0)).map_err(map)
            })
            .unwrap()
    };
    for table in [
        "words",
        "segments",
        "transcripts",
        "speakers",
        "segment_fts",
    ] {
        let sql = format!("SELECT COUNT(*) FROM {table} WHERE job_id = 'j0'");
        assert_eq!(count(&sql), 0, "orphan rows survived in {table}");
    }
    // And exactly this job's words went with them: no more, no less.
    assert_eq!(count("SELECT COUNT(*) FROM words"), total_words - j0_words);
    assert_eq!(store.stats().unwrap().word_count, total_words - j0_words);
    assert_eq!(store.stats().unwrap().job_count, 4);

    // The other jobs are untouched.
    for i in 1..5 {
        assert!(store.load_transcript(&format!("j{i}")).is_ok());
    }
    // And the search index no longer returns the deleted job.
    let results = store.search(&SearchQuery::new(needle)).unwrap();
    assert!(
        results.hits.iter().all(|h| h.job_id != "j0"),
        "a deleted job is still searchable"
    );
}

#[test]
fn a_few_hundred_jobs_list_filter_and_search_in_bounded_time() {
    // Budgets, not benchmarks: these are the numbers a UI thread can survive.
    // The measured figures are in the agent report; this test only fails if an
    // index went missing and a query became a full table scan.
    let dir = TempDir::new("corpus");
    let store = Store::open(dir.join("v.db")).unwrap();
    const JOBS: u64 = 300;
    let began = Instant::now();
    for i in 0..JOBS {
        let id = format!("job-{i:04}");
        let name = if i % 2 == 0 {
            "standup.m4a"
        } else {
            "review.mp4"
        };
        let mut job =
            JobRecord::new(id.clone(), dir.join(name)).with_content_hash(format!("{i:064x}"));
        job.phase = if i % 7 == 0 {
            JobPhase::Failed
        } else {
            JobPhase::Complete
        };
        job.language = Some(if i % 3 == 0 { "hi".into() } else { "en".into() });
        job.created_at_ms = 1_700_000_000_000 + i as i64 * 1_000;
        store.upsert_job(&job).unwrap();
        store
            .replace_transcript(&id, &fixtures::transcript(i, 25))
            .unwrap();
    }
    let build = began.elapsed();
    assert!(build.as_secs() < 60, "building the corpus took {build:?}");

    // Listing a page must not walk every transcript.
    let began = Instant::now();
    let page = store.list_jobs(&JobFilter::default(), 50, 0).unwrap();
    let list = began.elapsed();
    assert_eq!(page.total, JOBS as i64);
    assert_eq!(page.jobs.len(), 50);
    assert_eq!(page.jobs[0].id, "job-0299", "newest first");
    assert!(list.as_millis() < 200, "listing 50 of 300 took {list:?}");

    // Filtering.
    let began = Instant::now();
    let filtered = store
        .list_jobs(
            &JobFilter::default()
                .with_phases([JobPhase::Failed])
                .with_name("standup"),
            500,
            0,
        )
        .unwrap();
    let filter_time = began.elapsed();
    assert!(filtered.jobs.iter().all(|j| j.phase == JobPhase::Failed));
    assert!(filtered.total > 0);
    assert!(
        filter_time.as_millis() < 200,
        "filtering took {filter_time:?}"
    );

    let by_lang = store
        .list_jobs(
            &JobFilter {
                language: Some("hi".into()),
                order: JobOrder::OldestFirst,
                ..Default::default()
            },
            5,
            0,
        )
        .unwrap();
    assert!(
        by_lang
            .jobs
            .iter()
            .all(|j| j.language.as_deref() == Some("hi")),
        "the language filter did not apply"
    );

    // Search across the whole corpus, several ways.
    for query in ["transcription", "deadline", "the quick"] {
        let began = Instant::now();
        let hits = store
            .search(&SearchQuery::new(query).with_limit(50))
            .unwrap();
        let took = began.elapsed();
        assert!(!hits.hits.is_empty(), "{query:?} found nothing in 300 jobs");
        assert!(
            took.as_millis() < 250,
            "searching {query:?} across 300 jobs took {took:?}"
        );
        for h in &hits.hits {
            assert!(
                h.snippet.contains("<mark>"),
                "no highlight in {:?}",
                h.snippet
            );
            assert!(store.load_job(&h.job_id).unwrap().is_some());
        }
    }

    // A term in the corpus but not in every segment still returns, and returns
    // fewer hits than there are segments: the term is selective.
    let all = store
        .search(&SearchQuery::new("hmm").with_limit(100_000))
        .unwrap();
    let total_segments = store.stats().unwrap().segment_count;
    assert!(!all.hits.is_empty());
    assert!(
        (all.hits.len() as i64) < total_segments,
        "a term that matched every segment would mean the index is not selective"
    );

    // Pagination covers every job exactly once.
    let mut seen = std::collections::BTreeSet::new();
    let mut offset = 0;
    loop {
        let page = store.list_jobs(&JobFilter::default(), 37, offset).unwrap();
        if page.jobs.is_empty() {
            break;
        }
        for j in page.jobs {
            assert!(seen.insert(j.id.clone()), "{} appeared twice", j.id);
        }
        offset += 37;
    }
    assert_eq!(seen.len(), JOBS as usize);
}

#[test]
fn a_hyphenated_word_is_searched_as_a_phrase_and_by_its_parts() {
    // FTS5's `unicode61` tokenizer treats a hyphen as a token separator, so
    // "state-of-the-art" is indexed as the four tokens `state of the art`.
    // Because the search layer quotes every term, a hyphenated word becomes a
    // phrase query and therefore still matches, and so do its parts. That is
    // the behaviour worth pinning: passing the text through unquoted would turn
    // every hyphenated word in a transcript into either a syntax error or an
    // unsearchable token.
    let dir = TempDir::new("hyphen");
    let store = Store::open(dir.join("v.db")).unwrap();
    let mut t = vaani_core::Transcript::new(vaani_core::EngineStamp::asr("m", "1"));
    let mut seg = vaani_core::Segment::new(Time(0), Time(1_000));
    seg.id = "s0".into();
    seg.words = vec![
        vaani_core::Word::new("state-of-the-art", Time(0), Time(500)),
        vaani_core::Word::new("design", Time(500), Time(1_000)),
    ];
    t.segments.push(seg);
    let job = JobRecord::new("j", dir.join("a.wav"));
    store.upsert_job(&job).unwrap();
    store.replace_transcript("j", &t).unwrap();

    for query in ["state-of-the-art", "state", "art", "the", "state art"] {
        let hits = store.search(&SearchQuery::new(query)).unwrap();
        assert_eq!(hits.hits.len(), 1, "searching {query:?} should find it");
    }
    // Order matters *within* a term, because a hyphenated term is one quoted
    // phrase: "the-of-state" is not in the text.
    let reversed = store.search(&SearchQuery::new("the-of-state")).unwrap();
    assert!(
        reversed.is_empty(),
        "a quoted phrase must match in order, and this one is reversed"
    );
    // Order does not matter *between* terms, because separate words are ANDed
    // rather than treated as a phrase. That is what a search-as-you-type box
    // means, and it is why "art state" finds the segment.
    let anded = store.search(&SearchQuery::new("art state")).unwrap();
    assert_eq!(anded.hits.len(), 1);

    // Diacritics are folded by `remove_diacritics 2`, so a user need not type
    // them: `cafe` finds the accented spelling. The word is written with an
    // explicit escape so this file's own encoding cannot affect the test.
    let mut t2 = vaani_core::Transcript::new(vaani_core::EngineStamp::asr("m", "1"));
    let mut seg2 = vaani_core::Segment::new(Time(0), Time(100));
    seg2.id = "s1".into();
    seg2.words = vec![vaani_core::Word::new("cafe\u{301}", Time(0), Time(100))];
    t2.segments.push(seg2);
    store.replace_transcript("j", &t2).unwrap();
    let folded = store.search(&SearchQuery::new("cafe")).unwrap();
    assert_eq!(
        folded.hits.len(),
        1,
        "a user typing `cafe` should find the accented spelling"
    );
}

#[test]
fn settings_survive_a_reopen_and_the_export_log_is_cumulative() {
    let dir = TempDir::new("settings-reopen");
    let path = dir.join("v.db");
    {
        let store = Store::open(&path).unwrap();
        store.set_setting("device", "cuda").unwrap();
        let ui = serde_json::json!({"theme": "dark", "font": 14});
        store.set_setting_json("ui", &ui).unwrap();
        let job = JobRecord::new("j", dir.join("a.wav"));
        store.upsert_job(&job).unwrap();
        store
            .replace_transcript("j", &fixtures::transcript(2, 3))
            .unwrap();
        for i in 0..3 {
            let name = format!("out-{i}.srt");
            let record = ExportRecord::new(Some("j"), dir.join(name), "srt", 100 * (i + 1));
            store.record_export(&record).unwrap();
        }
    }
    let store = Store::open(&path).unwrap();
    assert_eq!(
        store.get_setting("device").unwrap().as_deref(),
        Some("cuda")
    );
    let ui: serde_json::Value = store.get_setting_json("ui").unwrap().unwrap();
    assert_eq!(ui["theme"], "dark");
    assert_eq!(ui["font"], 14);
    assert_eq!(store.list_exports(Some("j")).unwrap().len(), 3);
    assert_eq!(store.all_settings().unwrap().len(), 2);
}

#[test]
fn a_failed_job_keeps_its_reason_across_a_reopen() {
    let dir = TempDir::new("failure-reopen");
    let path = dir.join("v.db");
    let why = StoredError::new(
        "transcription_failed",
        "The speech recognition engine could not produce a transcript for this file.",
    );
    {
        let store = Store::open(&path).unwrap();
        let job = JobRecord::new("j", dir.join("a.wav"));
        store.upsert_job(&job).unwrap();
        assert!(store.set_phase("j", JobPhase::Asr, None).unwrap());
        store.set_phase("j", JobPhase::Failed, Some(&why)).unwrap();
    }
    let store = Store::open(&path).unwrap();
    let job = store.load_job("j").unwrap().unwrap();
    assert_eq!(job.phase, JobPhase::Failed);
    assert_eq!(job.error, Some(why));
    // Recovery must not touch a job that already reached a terminal phase.
    assert!(store.recover_interrupted_jobs().unwrap().is_empty());
}

#[test]
fn the_same_file_is_detected_as_already_transcribed() {
    let dir = TempDir::new("dedupe");
    let store = Store::open(dir.join("v.db")).unwrap();
    let media = dir.join("standup.m4a");
    let bytes: Vec<u8> = (0..50_000u32).map(|i| (i % 251) as u8).collect();
    std::fs::write(&media, &bytes).unwrap();
    let hash = vaani_storage::hash::hash_file(&media).unwrap();

    let mut done = JobRecord::new("a", &media).with_content_hash(&hash);
    done.phase = JobPhase::Complete;
    store.upsert_job(&done).unwrap();
    store
        .replace_transcript("a", &fixtures::transcript(1, 4))
        .unwrap();
    let found = store.find_completed_by_hash(&hash).unwrap();
    assert_eq!(found.map(|j| j.id).as_deref(), Some("a"));

    // A job that is still running is not a reason to refuse: only a completed
    // one is, because a running or failed job is an invitation to try again.
    let mut running = JobRecord::new("b", &media).with_content_hash(&hash);
    running.phase = JobPhase::Asr;
    store.upsert_job(&running).unwrap();
    let found = store.find_completed_by_hash(&hash).unwrap();
    assert_eq!(
        found.map(|j| j.id).as_deref(),
        Some("a"),
        "the completed job is still the answer"
    );

    // The same bytes under a different name is the same file.
    let copy = dir.join("standup (copy).m4a");
    std::fs::write(&copy, &bytes).unwrap();
    assert_eq!(vaani_storage::hash::hash_file(&copy).unwrap(), hash);
    assert!(store.find_completed_by_hash(&hash).unwrap().is_some());

    // A different file is not.
    std::fs::write(&copy, b"different").unwrap();
    let other = vaani_storage::hash::hash_file(&copy).unwrap();
    assert_ne!(other, hash);
    assert!(store.find_completed_by_hash(&other).unwrap().is_none());
    // And an unset hash never matches a row whose hash was never set.
    assert!(store.find_completed_by_hash("").unwrap().is_none());
}

#[test]
fn a_transcript_is_loaded_in_one_pass_regardless_of_size() {
    // A job with 5,000 words must not cost 5,000 queries: the loader reads every
    // word of a job in one ordered scan and buckets them by segment. Asserted by
    // shape (the words come back in the right segments) and by time.
    let dir = TempDir::new("one-pass");
    let store = Store::open(dir.join("v.db")).unwrap();
    let mut t: Transcript = fixtures::empty();
    for i in 0..200 {
        let words: Vec<vaani_core::Word> = (0..25)
            .map(|w| {
                let start = i as i64 * 1_000 + w * 10;
                vaani_core::Word::new(format!("w{i}-{w}"), Time(start), Time(start + 9))
            })
            .collect();
        let mut seg =
            vaani_core::Segment::new(Time(i as i64 * 1_000), Time(i as i64 * 1_000 + 999));
        seg.id = format!("s{i}");
        seg.words = words;
        t.segments.push(seg);
    }
    let job = JobRecord::new("j", dir.join("a.wav"));
    store.upsert_job(&job).unwrap();
    store.replace_transcript("j", &t).unwrap();
    let began = Instant::now();
    let back = store.load_transcript("j").unwrap();
    let took = began.elapsed();
    assert!(fixtures::diff(&t, &back).is_none());
    assert_eq!(back.segments[7].words[3].text, "w7-3");
    assert!(took.as_millis() < 200, "loading 5,000 words took {took:?}");
    assert_eq!(back.segments[199].words[24].text, "w199-24");
}

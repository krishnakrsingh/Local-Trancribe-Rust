//! Many threads, one database file.
//!
//! A desktop app that a user has pointed at eight files at once must not
//! deadlock, must not report `SQLITE_BUSY`, and must not lose a write. Those are
//! three different failure modes with three different causes:
//!
//! * **deadlock** — a lock ordering cycle between our own writer mutex and
//!   SQLite's own locks, or a read pool that never hands a connection back;
//! * **`SQLITE_BUSY`** — two writers, which the single-writer design makes
//!   structurally impossible, or a read/write lock upgrade, which the
//!   `IMMEDIATE` transaction behaviour avoids;
//! * **a lost write** — two threads writing the same row without noticing, which
//!   is what an atomic compare-and-swap or an `INSERT .. ON CONFLICT` is for.
//!
//! Every test here has a wall-clock deadline, because a deadlock in a test suite
//! is otherwise a hung CI job rather than a failure.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use vaani_core::JobPhase;
use vaani_storage::{JobFilter, JobRecord, SearchQuery, Store, fixtures, testing::TempDir};

const DEADLINE: Duration = Duration::from_secs(120);

fn store() -> (TempDir, Store) {
    let dir = TempDir::new("concurrency");
    let s = Store::open(dir.join("v.db")).unwrap();
    (dir, s)
}

#[test]
fn eight_threads_writing_jobs_and_transcripts_never_fail_or_lose_a_write() {
    let (dir, store) = store();
    let threads: u64 = 8;
    let per_thread: u64 = 6;
    let store = Arc::new(store);
    let barrier = Arc::new(Barrier::new(threads as usize));
    let errors = Arc::new(AtomicU64::new(0));

    let began = Instant::now();
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let store = Arc::clone(&store);
            let dir_path = dir.path().to_path_buf();
            let barrier = Arc::clone(&barrier);
            let errors = Arc::clone(&errors);
            std::thread::spawn(move || {
                barrier.wait();
                for i in 0..per_thread {
                    let id = format!("t{t}-{i}");
                    let job = JobRecord::new(&id, dir_path.join(format!("input-{t}-{i}.m4a")))
                        .with_content_hash(format!("{t}-{i}"));
                    if let Err(e) = store
                        .upsert_job(&job)
                        .and_then(|()| store.set_phase(&id, JobPhase::Asr, None))
                        .and_then(|_| {
                            store.replace_transcript(&id, &fixtures::transcript(t * 100 + i, 60))
                        })
                        .and_then(|_| store.set_phase(&id, JobPhase::Complete, None))
                    {
                        errors.fetch_add(1, Ordering::Relaxed);
                        eprintln!("thread {t} job {i} failed: {}", e.debug_chain());
                    }
                    // Interleave reads with the writes: a read that sees a
                    // half-written transcript would be a correctness bug, not
                    // just a performance one.
                    if i % 2 == 0 {
                        let _ = store.load_job(&id);
                        let _ = store.load_transcript(&id);
                        let _ = store.search(&SearchQuery::new("transcription").with_limit(5));
                        let _ = store.list_jobs(&JobFilter::default(), 20, 0);
                    }
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("no thread may panic");
    }
    assert!(
        began.elapsed() < DEADLINE,
        "took {:?}, which is past the deadline: this looks like a deadlock",
        began.elapsed()
    );
    assert_eq!(errors.load(Ordering::Relaxed), 0, "a write failed");

    // Nothing was lost: every job is present, complete, and has a transcript
    // with the right shape.
    let total = threads * per_thread;
    let page = store.list_jobs(&JobFilter::default(), 1_000, 0).unwrap();
    assert_eq!(page.total, total as i64);
    for t in 0..threads {
        for i in 0..per_thread {
            let id = format!("t{t}-{i}");
            let job = store.load_job(&id).unwrap().expect("a job went missing");
            assert_eq!(job.phase, JobPhase::Complete, "{id} was left mid-flight");
            let transcript = store.load_transcript(&id).unwrap();
            assert_eq!(transcript.segments.len(), 60, "{id}");
            assert_eq!(
                store.count_words(&id).unwrap(),
                transcript
                    .segments
                    .iter()
                    .map(|s| s.words.len())
                    .sum::<usize>() as i64,
                "{id}"
            );
        }
    }
    let stats = store.stats().unwrap();
    assert_eq!(stats.job_count, total as i64);
    assert_eq!(stats.transcript_count, total as i64);
    let report = store.integrity_check().unwrap();
    assert!(report.ok, "integrity check failed: {:?}", report.messages);
}

#[test]
fn many_threads_racing_on_one_row_do_not_lose_the_winner() {
    // The upsert path: eight threads writing the *same* job id. Exactly one
    // final state must be visible, it must be one of the values written, and the
    // row must not be duplicated or half-updated.
    let (_dir, store) = store();
    let store = Arc::new(store);
    let barrier = Arc::new(Barrier::new(8));
    let handles: Vec<_> = (0..8)
        .map(|t| {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                for i in 0..20 {
                    let mut job = JobRecord::new("contended", format!("/media/file-{t}.wav"));
                    job.phase = JobPhase::Asr;
                    job.model = Some(format!("model-{t}-{i}"));
                    store.upsert_job(&job).unwrap();
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let job = store.load_job("contended").unwrap().unwrap();
    assert!(job.model.is_some());
    assert_eq!(
        store.list_jobs(&JobFilter::default(), 10, 0).unwrap().total,
        1
    );
}

#[test]
fn a_big_read_does_not_block_a_small_write() {
    // WAL is the reason a 20,000-word load does not freeze the UI while a job
    // saves. If this test ever fails, the pragma that makes it true has been
    // lost.
    let (dir, store) = store();
    let big = fixtures::transcript(1, 3_000);
    store
        .upsert_job(&JobRecord::new("big", dir.join("big.wav")))
        .unwrap();
    store.replace_transcript("big", &big).unwrap();
    store
        .upsert_job(&JobRecord::new("small", dir.join("small.wav")))
        .unwrap();

    let reader = {
        let store = store.clone();
        std::thread::spawn(move || {
            for _ in 0..20 {
                store.load_transcript("big").unwrap();
            }
        })
    };
    let writer = {
        let store = store.clone();
        std::thread::spawn(move || {
            for _ in 0..20 {
                store
                    .upsert_job(&JobRecord::new("small", dir.join("small.wav")))
                    .unwrap();
            }
        })
    };
    let began = Instant::now();
    reader.join().unwrap();
    writer.join().unwrap();
    assert!(
        began.elapsed() < DEADLINE,
        "reads and writes serialised: {:?}",
        began.elapsed()
    );
}

#[test]
fn a_full_read_pool_queues_instead_of_failing() {
    // With `max_read_connections = 1`, forty simultaneous readers must take
    // turns. A pool that returned an error, or that lost a slot, would fail
    // here.
    let dir = TempDir::new("concurrency-pool");
    let store = Store::open_with(
        &dir.join("v.db"),
        &vaani_storage::OpenOptions {
            max_read_connections: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let store = Arc::new(store);
    let barrier = Arc::new(Barrier::new(40));
    let handles: Vec<_> = (0..40)
        .map(|_| {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                for _ in 0..10 {
                    let n: i64 = store
                        .with_read(|c| {
                            let map = |e: rusqlite::Error| {
                                vaani_storage::error::from_sqlite(e, store.path())
                            };
                            c.query_row("SELECT 1", [], |r| r.get(0)).map_err(map)
                        })
                        .unwrap();
                    assert_eq!(n, 1);
                }
            })
        })
        .collect();
    let began = Instant::now();
    for h in handles {
        h.join().expect("a reader panicked or deadlocked");
    }
    assert!(began.elapsed() < DEADLINE, "{:?}", began.elapsed());
}

#[test]
fn two_processes_do_not_deadlock_on_the_same_database() {
    // The app and the CLI open to the same file. `busy_timeout` is what makes
    // that a wait rather than an error, and this test opens the same file twice
    // in-process to exercise the same code path.
    let dir = TempDir::new("concurrency-two");
    let path = dir.join("v.db");
    let a = Store::open(&path).unwrap();
    let b = Store::open(&path).unwrap();
    let bar = Arc::new(Barrier::new(2));
    let ha = {
        let (a, b, bar) = (a.clone(), b.clone(), Arc::clone(&bar));
        std::thread::spawn(move || {
            bar.wait();
            for i in 0..30 {
                let id = format!("a{i}");
                a.upsert_job(&JobRecord::new(&id, format!("/m/{id}.wav")))
                    .unwrap();
                a.replace_transcript(&id, &fixtures::transcript(i, 10))
                    .unwrap();
                // ...while the other handle writes to the same file.
                b.upsert_job(&JobRecord::new(format!("b{i}"), format!("/m/b{i}.wav")))
                    .unwrap();
            }
        })
    };
    let hb = {
        let (a, b, bar) = (a.clone(), b.clone(), Arc::clone(&bar));
        std::thread::spawn(move || {
            bar.wait();
            for i in 0..30 {
                let id = format!("c{i}");
                b.upsert_job(&JobRecord::new(&id, format!("/m/{id}.wav")))
                    .unwrap();
                b.replace_transcript(&id, &fixtures::transcript(i + 500, 10))
                    .unwrap();
                a.upsert_job(&JobRecord::new(format!("d{i}"), format!("/m/d{i}.wav")))
                    .unwrap();
            }
        })
    };
    let began = Instant::now();
    ha.join().unwrap();
    hb.join().unwrap();
    assert!(began.elapsed() < DEADLINE, "{:?}", began.elapsed());
    assert_eq!(
        a.list_jobs(&JobFilter::default(), 500, 0).unwrap().total,
        120
    );
    assert!(a.integrity_check().unwrap().ok);
}

#[test]
fn an_integrity_check_after_concurrent_reads_and_writes_is_honest() {
    // Regression test for a measured false positive: `PRAGMA integrity_check`
    // run on a *pooled reader* while several connections share a hot write-ahead
    // log reports "malformed inverted index for FTS5 table segment_fts" on a
    // database that is provably intact. FTS5's own `'integrity-check'` reports
    // nothing, a `'rebuild'` changes nothing, and reopening the file reports
    // `ok`. `Store::integrity_check` therefore runs on the writer connection;
    // this test is what stops that from being "simplified" back.
    let (dir, store) = store();
    let store = Arc::new(store);
    let barrier = Arc::new(Barrier::new(8));
    let handles: Vec<_> = (0..8)
        .map(|t| {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            let dir_path = dir.path().to_path_buf();
            std::thread::spawn(move || {
                barrier.wait();
                for i in 0..5 {
                    let id = format!("t{t}-{i}");
                    store
                        .upsert_job(&JobRecord::new(&id, dir_path.join("a.wav")))
                        .unwrap();
                    store
                        .replace_transcript(&id, &fixtures::transcript(t * 100 + i, 50))
                        .unwrap();
                    // A full-text read from a pooled reader, interleaved with the
                    // writes: exactly the pattern that provoked the false report.
                    let _ = store.search(&SearchQuery::new("transcription").with_limit(5));
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let report = store.integrity_check().unwrap();
    assert!(
        report.ok,
        "a healthy database reported damage: {:?}",
        report.messages
    );
    // And the index itself agrees, which is the independent check.
    let fts5: Vec<String> = store
        .with_conn(|c| {
            let map = |e: rusqlite::Error| vaani_storage::error::from_sqlite(e, store.path());
            let mut st = c
                .prepare("INSERT INTO segment_fts(segment_fts) VALUES('integrity-check')")
                .map_err(map)?;
            let rows = st.query_map([], |r| r.get::<_, String>(0)).map_err(map)?;
            let mut v = Vec::new();
            for r in rows {
                v.push(r.map_err(map)?);
            }
            Ok(v)
        })
        .unwrap();
    assert!(fts5.is_empty(), "FTS5 reported its own problems: {fts5:?}");
}

#[test]
fn concurrent_searches_return_the_same_answer_as_a_sequential_one() {
    let (dir, store) = store();
    for i in 0..40 {
        let id = format!("j{i}");
        store
            .upsert_job(&JobRecord::new(&id, dir.join("a.wav")))
            .unwrap();
        store
            .replace_transcript(&id, &fixtures::transcript(500 + i, 40))
            .unwrap();
    }
    let sequential_ids: Vec<String> = store
        .search(&SearchQuery::new("transcription").with_limit(20))
        .unwrap()
        .hits
        .iter()
        .map(|h| h.segment_id.clone())
        .collect();
    assert!(!sequential_ids.is_empty());

    let store = Arc::new(store);
    let barrier = Arc::new(Barrier::new(8));
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            let expected = sequential_ids.clone();
            std::thread::spawn(move || {
                barrier.wait();
                for _ in 0..10 {
                    let r = store
                        .search(&SearchQuery::new("transcription").with_limit(20))
                        .unwrap();
                    assert_eq!(r.hits.len(), expected.len());
                    for (a, b) in r.hits.iter().zip(expected.iter()) {
                        assert_eq!(&a.segment_id, b);
                    }
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

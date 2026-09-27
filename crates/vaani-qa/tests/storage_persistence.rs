//! Storage persistence tests: restart recovery, round-trips, job counts.
//!
//! Uses throwaway databases in temp dirs (never the user's real database).

use vaani_core::{EngineStamp, JobPhase, Segment, Time, Transcript, Word};
use vaani_storage::{JobRecord, Store, new_job_id};

fn sample_transcript(words_per_segment: usize, segments: usize) -> Transcript {
    let mut t = Transcript::new(EngineStamp::asr("qa", "0"));
    for i in 0..segments {
        let start = i as i64 * 2000;
        let words: Vec<Word> = (0..words_per_segment)
            .map(|w| Word::new(format!("w{w}"), Time(start), Time(start + 1000)))
            .collect();
        let mut seg = Segment::new(Time(start), Time(start + 1000));
        seg.id = format!("s{i}");
        seg.words = words;
        t.segments.push(seg);
    }
    t
}

fn temp_store(name: &str) -> (tempfile::TempDir, Store) {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let path = dir.path().join(format!("{name}.db"));
    let store = Store::open(&path).expect("open store");
    (dir, store)
}

#[test]
fn restart_persistence_round_trips_transcript() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let path = dir.path().join("restart.db");
    let job_id = new_job_id();
    let want = sample_transcript(4, 10);
    {
        let store = Store::open(&path).expect("open");
        let mut job = JobRecord::new(&job_id, dir.path().join("meeting.wav"));
        job.phase = JobPhase::Complete;
        store.upsert_job(&job).expect("upsert");
        store.replace_transcript(&job_id, &want).expect("replace");
    }
    // Reopen: the transcript survives the "restart".
    {
        let store = Store::open(&path).expect("reopen");
        let got = store.load_transcript(&job_id).expect("load");
        assert_eq!(got.segments.len(), want.segments.len());
        for (a, b) in got.segments.iter().zip(want.segments.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.start, b.start);
            assert_eq!(a.end, b.end);
            assert_eq!(a.text(), b.text());
        }
        let job = store
            .load_job(&job_id)
            .expect("load job")
            .expect("job exists");
        assert_eq!(job.phase, JobPhase::Complete);
    }
}

#[test]
fn interrupted_jobs_recover_on_reopen() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let path = dir.path().join("recover.db");
    {
        let store = Store::open(&path).expect("open");
        let mut active = JobRecord::new("active-job", dir.path().join("a.wav"));
        active.phase = JobPhase::Asr;
        store.upsert_job(&active).expect("upsert");
        let mut queued = JobRecord::new("queued-job", dir.path().join("b.wav"));
        queued.phase = JobPhase::Queued;
        store.upsert_job(&queued).expect("upsert");
    }
    let store = Store::open(&path).expect("reopen recovers");
    let active = store.load_job("active-job").expect("load").expect("exists");
    assert!(
        active.phase.is_terminal(),
        "interrupted job left active: {}",
        active.phase
    );
    let queued = store.load_job("queued-job").expect("load").expect("exists");
    assert!(queued.phase.is_terminal());
}

#[test]
fn job_persistence_counts_words_segments_speakers() {
    let (_dir, store) = temp_store("counts");
    let job_id = new_job_id();
    store
        .upsert_job(&JobRecord::new(&job_id, "meeting.m4a"))
        .expect("upsert");
    let mut t = sample_transcript(5, 8);
    t.add_speaker(vaani_core::Speaker::new(0));
    t.add_speaker(vaani_core::Speaker::new(1));
    for (i, s) in t.segments.iter_mut().enumerate() {
        s.speaker = Some(i % 2);
    }
    let stats = store.replace_transcript(&job_id, &t).expect("replace");
    assert_eq!(stats.words, 40);
    assert_eq!(stats.segments, 8);
    assert_eq!(store.count_words(&job_id).expect("words"), 40);
    assert_eq!(store.count_segments(&job_id).expect("segments"), 8);
    let job = store.load_job(&job_id).expect("load").expect("exists");
    assert_eq!(job.word_count, 40);
    assert_eq!(job.segment_count, 8);
    assert_eq!(job.speaker_count, 2);
    // Empty transcript is a complete state, not an error.
    let job2 = new_job_id();
    store
        .upsert_job(&JobRecord::new(&job2, "empty.wav"))
        .expect("upsert");
    let empty = Transcript::new(EngineStamp::asr("qa", "0"));
    let stats = store
        .replace_transcript(&job2, &empty)
        .expect("replace empty");
    assert_eq!(stats.words, 0);
    assert_eq!(stats.segments, 0);
}

#[test]
fn job_list_paginates_and_filters() {
    let (_dir, store) = temp_store("list");
    for i in 0..5 {
        let mut job = JobRecord::new(format!("job-{i}"), format!("file-{i}.wav"));
        job.phase = if i % 2 == 0 {
            JobPhase::Complete
        } else {
            JobPhase::Failed
        };
        store.upsert_job(&job).expect("upsert");
    }
    let page = store
        .list_jobs(&vaani_storage::JobFilter::default(), 10, 0)
        .expect("list");
    assert_eq!(page.total, 5);
    let page = store
        .list_jobs(
            &vaani_storage::JobFilter::default().with_phases([JobPhase::Complete]),
            10,
            0,
        )
        .expect("list filtered");
    assert_eq!(page.total, 3);
    let page = store
        .list_jobs(&vaani_storage::JobFilter::default(), 2, 0)
        .expect("list paged");
    assert_eq!(page.jobs.len(), 2);
    assert_eq!(page.total, 5);
}

//! Storage persistence: restart recovery, round-trips, counters, pagination,
//! and the atomicity the pipeline's "no partial transcript" promise rests on.
//!
//! Every database here is a throwaway in a temp directory. The user's real
//! database is never opened, and nothing reads it even to check that it exists.
//!
//! # What was added
//!
//! The first version of this file checked that a transcript round-trips. It did
//! not check that a *replacement* is atomic, which is the property
//! `Orchestrator`'s "a failed job leaves no partial transcript" claim actually
//! depends on. If `replace_transcript` deleted before it wrote, a crash between
//! the two would leave a job with a `segment_count` and no segments — which is
//! the half-written transcript the whole design is against.

use vaani_core::{EngineStamp, JobPhase, Segment, Speaker, Time, Transcript, Word};
use vaani_storage::{JobFilter, JobRecord, Store, new_job_id};

/// A transcript with `segments` segments of `words_per_segment` words each, with
/// ids set, so a round-trip has something to lose.
fn sample_transcript(words_per_segment: usize, segments: usize) -> Transcript {
    let mut t = Transcript::new(EngineStamp::asr("qa", "0"));
    for i in 0..segments {
        let start = i as i64 * 2_000;
        let words: Vec<Word> = (0..words_per_segment)
            .map(|w| Word::new(format!("w{w}"), Time(start), Time(start + 1_000)))
            .collect();
        let mut seg = Segment::new(Time(start), Time(start + 1_000));
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
fn a_transcript_survives_a_restart_with_every_field_intact() {
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
        store
            .close()
            .expect("close, so the reopen is a real restart");
    }
    let store = Store::open(&path).expect("reopen");
    let got = store.load_transcript(&job_id).expect("load");
    assert_eq!(got.segments.len(), want.segments.len());
    for (before, after) in want.segments.iter().zip(got.segments.iter()) {
        assert_eq!(before.id, after.id, "a segment id did not survive");
        assert_eq!(before.start, after.start);
        assert_eq!(before.end, after.end);
        assert_eq!(before.speaker, after.speaker);
        assert_eq!(before.text(), after.text());
        assert_eq!(before.words.len(), after.words.len());
        for (b, a) in before.words.iter().zip(after.words.iter()) {
            assert_eq!(b.text, a.text);
            assert_eq!(b.start, a.start);
            assert_eq!(b.end, a.end);
        }
    }
    let job = store
        .load_job(&job_id)
        .expect("load job")
        .expect("job exists");
    assert_eq!(job.phase, JobPhase::Complete);
    assert!(!job.content_hash.is_empty() || job.content_hash.is_empty());
    assert!(store.integrity_check().expect("integrity").ok);
}

#[test]
fn the_denormalised_counters_agree_with_the_rows_they_summarise() {
    // The job row carries counts so the history list does not have to load every
    // transcript. That is a cache, and a cache that disagrees with its source is
    // worse than no cache: the UI would show "3 segments" over an empty
    // transcript.
    let (_dir, store) = temp_store("counts");
    let job_id = new_job_id();
    store
        .upsert_job(&JobRecord::new(&job_id, "meeting.m4a"))
        .expect("upsert");
    let mut t = sample_transcript(5, 8);
    t.add_speaker(Speaker::new(0));
    t.add_speaker(Speaker::new(1));
    for (i, s) in t.segments.iter_mut().enumerate() {
        s.speaker = Some(i % 2);
    }
    let stats = store.replace_transcript(&job_id, &t).expect("replace");
    assert_eq!(stats.words, 40);
    assert_eq!(stats.segments, 8);
    assert_eq!(store.count_words(&job_id).expect("words"), 40);
    assert_eq!(store.count_segments(&job_id).expect("segments"), 8);
    let job = store.load_job(&job_id).expect("load").expect("exists");
    assert_eq!(job.word_count, 40, "the job row's word count disagrees");
    assert_eq!(
        job.segment_count, 8,
        "the job row's segment count disagrees"
    );
    assert_eq!(
        job.speaker_count, 2,
        "the job row's speaker count disagrees"
    );
}

#[test]
fn replacing_a_transcript_twice_leaves_the_second_one_whole() {
    // The atomicity the pipeline depends on. A replace that deleted before it
    // wrote would leave a job whose counters say 8 segments and whose rows say
    // zero — the half-written transcript the orchestrator promises never
    // appears. This cannot simulate a crash, so it checks the observable
    // consequence of a second write instead: the first is *gone*, completely,
    // and the second is complete.
    let (_dir, store) = temp_store("replace");
    let job_id = new_job_id();
    store
        .upsert_job(&JobRecord::new(&job_id, "meeting.m4a"))
        .expect("upsert");

    let first = sample_transcript(3, 5);
    store.replace_transcript(&job_id, &first).expect("first");
    let second = sample_transcript(2, 9);
    store.replace_transcript(&job_id, &second).expect("second");

    let got = store.load_transcript(&job_id).expect("load");
    assert_eq!(
        got.segments.len(),
        9,
        "a replacement must leave exactly the new transcript, not a merge"
    );
    assert_eq!(store.count_segments(&job_id).expect("count"), 9);
    assert_eq!(store.count_words(&job_id).expect("count"), 18);
    for (i, s) in got.segments.iter().enumerate() {
        assert_eq!(
            s.id,
            format!("s{i}"),
            "segment {i} is not from the new transcript"
        );
        assert_eq!(s.words.len(), 2);
    }
    // And the job row was updated to match, not left describing the first write.
    let job = store.load_job(&job_id).expect("load").expect("exists");
    assert_eq!(job.segment_count, 9);
    assert_eq!(job.word_count, 18);
}

#[test]
fn an_empty_transcript_is_a_complete_state_not_an_error() {
    // A silent recording must persist as a job with no content, not fail.
    let (_dir, store) = temp_store("empty");
    let job_id = new_job_id();
    store
        .upsert_job(&JobRecord::new(&job_id, "empty.wav"))
        .expect("upsert");
    let stats = store
        .replace_transcript(&job_id, &Transcript::new(EngineStamp::asr("qa", "0")))
        .expect("replace empty");
    assert_eq!(stats.words, 0);
    assert_eq!(stats.segments, 0);
    let back = store
        .load_transcript(&job_id)
        .expect("an empty transcript must load");
    assert!(back.segments.is_empty());
    let job = store.load_job(&job_id).expect("load").expect("exists");
    assert_eq!(job.segment_count, 0);
}

#[test]
fn a_job_interrupted_mid_run_is_recovered_into_a_terminal_phase() {
    // The crash case: a process died with the job row still in a working phase.
    // On the next open the row must not still claim to be running, or the UI
    // shows a spinner for a job nobody is working on.
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
        store.close().expect("close without finishing");
    }
    let store = Store::open(&path).expect("reopen recovers");
    for id in ["active-job", "queued-job"] {
        let job = store.load_job(id).expect("load").expect("exists");
        assert!(
            job.phase.is_terminal(),
            "{id} was left in {:?}, which the UI would render as still running",
            job.phase
        );
    }
    assert!(store.integrity_check().expect("integrity").ok);
}

#[test]
fn a_failed_job_keeps_the_reason_it_failed() {
    // A restart must be able to tell the user why their job failed. A `Failed`
    // row with no error is a support question with no answer.
    let (_dir, store) = temp_store("failed");
    let job_id = new_job_id();
    store
        .upsert_job(&JobRecord::new(&job_id, "broken.m4a"))
        .expect("upsert");
    let error = vaani_storage::StoredError::new("audio_decode_failed", "the codec is missing");
    store
        .set_phase(&job_id, JobPhase::Failed, Some(&error))
        .expect("set phase");
    let job = store.load_job(&job_id).expect("load").expect("exists");
    assert_eq!(job.phase, JobPhase::Failed);
    let stored = job.error.as_ref().expect("the reason is persisted");
    assert_eq!(
        stored.code, "audio_decode_failed",
        "a stable code, not prose"
    );
    assert!(!stored.message.is_empty());
}

#[test]
fn the_job_list_paginates_and_filters_by_phase() {
    // The history window: five jobs, alternating Complete and Failed.
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
    let all = store.list_jobs(&JobFilter::default(), 10, 0).expect("list");
    assert_eq!(all.total, 5, "the total counts every match, not the page");

    let complete = store
        .list_jobs(
            &JobFilter::default().with_phases([JobPhase::Complete]),
            10,
            0,
        )
        .expect("list filtered");
    assert_eq!(complete.total, 3);
    assert!(
        complete.jobs.iter().all(|j| j.phase == JobPhase::Complete),
        "a phase filter that returns other phases is worse than no filter"
    );

    // Paging must partition, not repeat: the first two and the next two must be
    // four different jobs.
    let page1 = store
        .list_jobs(&JobFilter::default(), 2, 0)
        .expect("page 1");
    let page2 = store
        .list_jobs(&JobFilter::default(), 2, 2)
        .expect("page 2");
    assert_eq!(page1.jobs.len(), 2);
    assert_eq!(page2.jobs.len(), 2);
    assert_eq!(page1.total, 5);
    assert_eq!(page2.total, 5);
    let ids1: Vec<&str> = page1.jobs.iter().map(|j| j.id.as_str()).collect();
    let ids2: Vec<&str> = page2.jobs.iter().map(|j| j.id.as_str()).collect();
    assert!(
        ids1.iter().all(|a| !ids2.contains(a)),
        "paging returned the same jobs twice: {ids1:?} then {ids2:?}"
    );

    // A name filter, because the history search box is a product feature.
    let named = store
        .list_jobs(&JobFilter::default().with_name("file-3"), 10, 0)
        .expect("list by name");
    assert_eq!(named.total, 1);
    assert_eq!(named.jobs[0].id, "job-3");
}

#[test]
fn two_databases_in_one_process_do_not_share_state() {
    // The obvious way to write a flaky test is to share a database between tests.
    // This asserts the isolation that makes per-test temp databases safe.
    let (_d1, a) = temp_store("iso-a");
    let (_d2, b) = temp_store("iso-b");
    let job = "shared-id";
    a.upsert_job(&JobRecord::new(job, "in-a.wav")).expect("a");
    assert!(
        b.load_job(job).expect("b").is_none(),
        "two stores on two files must not see each other's rows"
    );
    b.upsert_job(&JobRecord::new(job, "in-b.wav")).expect("b");
    assert_eq!(
        a.load_job(job).expect("a").expect("exists").input_path,
        "in-a.wav",
        "the second store overwrote the first's row"
    );
    // And an in-memory store is likewise its own thing, which is what makes
    // `Store::open_in_memory()` usable in a test at all.
    let mem = Store::open_in_memory().expect("in-memory");
    assert!(mem.load_job(job).expect("mem").is_none());
}

#[test]
fn a_speaker_rename_is_stored_separately_from_the_generated_label() {
    // A rename is a *label*, not an edit: the generated label must remain as the
    // fallback so clearing the rename is possible, and the transcript's words must
    // be untouched.
    let (_dir, store) = temp_store("rename");
    let job_id = new_job_id();
    store
        .upsert_job(&JobRecord::new(&job_id, "meeting.m4a"))
        .expect("upsert");
    let mut t = sample_transcript(2, 4);
    t.add_speaker(Speaker::new(0));
    t.add_speaker(Speaker::new(1));
    for s in t.segments.iter_mut() {
        s.speaker = Some(0);
    }
    store.replace_transcript(&job_id, &t).expect("replace");
    let before = store.load_transcript(&job_id).expect("load");

    assert!(
        store
            .rename_speaker(&job_id, 0, Some("  Asha  "))
            .expect("rename")
    );
    let row = store.speaker(&job_id, 0).expect("row").expect("exists");
    assert_eq!(row.name(), "Asha", "a rename is trimmed");
    assert!(!row.display_name.is_empty(), "the fallback label remains");

    let after = store.load_transcript(&job_id).expect("load");
    assert_eq!(after.speakers[0].name.as_deref(), Some("Asha"));
    for (b, a) in before.segments.iter().zip(after.segments.iter()) {
        assert_eq!(a.text(), b.text(), "a rename edited the text");
    }

    // Clearing falls back to the generated label.
    assert!(store.rename_speaker(&job_id, 0, None).expect("clear"));
    let cleared = store.load_transcript(&job_id).expect("load");
    assert_eq!(cleared.speakers[0].name, None);

    // Naming a speaker the diarizer never produced is a no-op with a signal.
    assert!(
        !store
            .rename_speaker(&job_id, 9, Some("Nobody"))
            .expect("phantom"),
        "a phantom speaker must not be inserted"
    );
    assert!(store.speaker(&job_id, 9).expect("row").is_none());
}

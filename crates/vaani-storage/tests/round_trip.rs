//! The core correctness property: a transcript that goes into the database and
//! comes back is the same transcript.
//!
//! Not "the same text" and "the same word count" — the same thing, field for
//! field, to the millisecond, with the same stable segment ids in the same
//! order. Segment ids are documented in the contract crate as surviving database
//! round-trips because edits and the UI's selection refer to them, and a store
//! that renumbered or reordered them on save would silently break every edit the
//! user had made.
//!
//! The inputs are generated from a seed, so a failure is reproducible from the
//! seed alone.

use vaani_core::{JobPhase, Time, Transcript};
use vaani_storage::{JobRecord, Store, fixtures, testing::TempDir};

/// How many transcripts to try, and how many segments in each. 200 x ~40 covers
/// the awkward shapes (empty segments, overlapping segments, inverted word
/// times, negative starts, every script) hundreds of times over without taking
/// minutes.
const CASES: u64 = 200;

#[test]
fn every_generated_transcript_round_trips_exactly() {
    let dir = TempDir::new("roundtrip");
    let store = Store::open(dir.join("v.db")).unwrap();

    for case in 0..CASES {
        let seed = case.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let segments = 1 + (case as usize % 40);
        let want = fixtures::transcript(seed, segments);
        let job_id = format!("case-{case}");

        let mut job = JobRecord::new(&job_id, dir.join("input.m4a"));
        job.phase = JobPhase::Complete;
        store.upsert_job(&job).unwrap();
        let stats = store.replace_transcript(&job_id, &want).unwrap();

        let got = store.load_transcript(&job_id).unwrap();
        if let Some(difference) = fixtures::diff(&want, &got) {
            panic!("seed {seed:#x} (case {case}, {segments} segments): {difference}");
        }

        // The counters agree with the rows, and the job knows the shape too.
        let words: usize = want.segments.iter().map(|s| s.words.len()).sum();
        assert_eq!(stats.words, words as i64, "seed {seed:#x}");
        assert_eq!(stats.segments, segments as i64, "seed {seed:#x}");
        assert_eq!(store.count_words(&job_id).unwrap(), words as i64);
        let job_back = store.load_job(&job_id).unwrap().unwrap();
        assert_eq!(job_back.word_count, words as i64);
        assert_eq!(job_back.segment_count, segments as i64);
        assert_eq!(job_back.speaker_count, want.speakers.len() as i64);
    }
}

#[test]
fn a_second_write_of_the_same_transcript_is_idempotent() {
    let dir = TempDir::new("roundtrip-idempotent");
    let store = Store::open(dir.join("v.db")).unwrap();
    let t = fixtures::transcript(7, 25);
    store
        .upsert_job(&JobRecord::new("j", dir.join("a.wav")))
        .unwrap();
    store.replace_transcript("j", &t).unwrap();
    let first = store.load_transcript("j").unwrap();
    store.replace_transcript("j", &t).unwrap();
    let second = store.load_transcript("j").unwrap();
    assert!(fixtures::diff(&first, &second).is_none());
    assert_eq!(store.count_segments("j").unwrap(), 25);
}

#[test]
fn times_are_stored_in_source_media_coordinates_and_never_rebased() {
    // A store that stored chunk-relative times, or that clamped them to zero,
    // would produce a transcript that is out of sync with the video. Negative
    // starts are legal in `Time` and must survive.
    let dir = TempDir::new("roundtrip-times");
    let store = Store::open(dir.join("v.db")).unwrap();
    let mut t = fixtures::empty();
    for (i, (start, end)) in [
        (-500i64, 0i64),
        (0, 1),
        (1, 1),
        (999_999_999, 1_000_000_000),
    ]
    .into_iter()
    .enumerate()
    {
        let mut seg = vaani_core::Segment::new(Time(start), Time(end));
        seg.id = format!("s{i}");
        seg.words = vec![vaani_core::Word::new("word", Time(start), Time(end))];
        t.segments.push(seg);
    }
    t.duration = Some(vaani_core::MediaLength::from_millis(1_000_000_000));
    store
        .upsert_job(&JobRecord::new("j", dir.join("a.wav")))
        .unwrap();
    store.replace_transcript("j", &t).unwrap();
    let back = store.load_transcript("j").unwrap();
    assert_eq!(
        back.segments[0].start,
        Time(-500),
        "a negative start is data"
    );
    assert_eq!(back.segments[1].end, Time(1));
    assert_eq!(
        back.segments[2].end,
        Time(1),
        "a zero-length segment survives"
    );
    assert_eq!(back.segments[3].end, Time(1_000_000_000));
    assert_eq!(
        back.duration,
        Some(vaani_core::MediaLength::from_millis(1_000_000_000))
    );
    // The raw column is an integer, not a formatted string.
    let raw: i64 = store
        .with_read(|conn| {
            let map = |e: rusqlite::Error| vaani_storage::error::from_sqlite(e, store.path());
            conn.query_row("SELECT start_ms FROM segments WHERE id = 's0'", [], |r| {
                r.get(0)
            })
            .map_err(map)
        })
        .unwrap();
    assert_eq!(raw, -500);
}

#[test]
fn confidences_keep_the_difference_between_zero_and_absent() {
    // "The model was sure" (Some(0.0)) and "the model said nothing"
    // (None) are different facts, and collapsing them is a silent data loss.
    let dir = TempDir::new("roundtrip-confidence");
    let store = Store::open(dir.join("v.db")).unwrap();
    let mut t = fixtures::empty();
    for (i, confidence) in [Some(0.0f32), None, Some(1.0), Some(0.123_456_79)]
        .into_iter()
        .enumerate()
    {
        let mut seg = vaani_core::Segment::new(Time(i as i64), Time(i as i64 + 1));
        seg.id = format!("s{i}");
        seg.confidence = confidence;
        seg.words = vec![vaani_core::Word {
            text: "w".into(),
            start: Time(i as i64),
            end: Time(i as i64 + 1),
            confidence,
            normalized: None,
        }];
        t.segments.push(seg);
    }
    store
        .upsert_job(&JobRecord::new("j", dir.join("a.wav")))
        .unwrap();
    store.replace_transcript("j", &t).unwrap();
    let back = store.load_transcript("j").unwrap();
    for (i, want) in [Some(0.0f32), None, Some(1.0), Some(0.123_456_79)]
        .into_iter()
        .enumerate()
    {
        assert_eq!(back.segments[i].confidence, want, "segment {i}");
        assert_eq!(back.segments[i].words[0].confidence, want, "word {i}");
    }
}

#[test]
fn segment_ids_survive_and_stay_unique_within_a_job() {
    let dir = TempDir::new("roundtrip-ids");
    let store = Store::open(dir.join("v.db")).unwrap();
    let t = fixtures::transcript(31, 30);
    store
        .upsert_job(&JobRecord::new("j", dir.join("a.wav")))
        .unwrap();
    store.replace_transcript("j", &t).unwrap();
    let back = store.load_transcript("j").unwrap();
    let before: Vec<&str> = t.segments.iter().map(|s| s.id.as_str()).collect();
    let after: Vec<&str> = back.segments.iter().map(|s| s.id.as_str()).collect();
    assert_eq!(before, after, "ids and their order are part of the data");

    // The same id in a *different* job is fine: ids are stable within a job.
    store
        .upsert_job(&JobRecord::new("k", dir.join("b.wav")))
        .unwrap();
    store.replace_transcript("k", &t).unwrap();
    assert_eq!(
        store.load_transcript("k").unwrap().segments[0].id,
        t.segments[0].id
    );
    assert_eq!(store.count_segments("j").unwrap(), 30);
    assert_eq!(store.count_segments("k").unwrap(), 30);
}

#[test]
fn speaker_order_is_preserved_rather_than_sorted() {
    // `Transcript::speakers` is a `Vec`, and the contract crate's own
    // `to_plain_text` indexes it positionally, so a store that sorted the
    // speakers on the way out would change what the UI prints.
    let dir = TempDir::new("roundtrip-speaker-order");
    let store = Store::open(dir.join("v.db")).unwrap();
    let mut t = fixtures::transcript(5, 10);
    // Deliberately not in index order.
    t.speakers = vec![
        vaani_core::Speaker {
            index: 3,
            name: Some("Third".into()),
            share: Some(0.5),
        },
        vaani_core::Speaker {
            index: 1,
            name: None,
            share: Some(0.25),
        },
        vaani_core::Speaker {
            index: 2,
            name: Some("Second".into()),
            share: None,
        },
    ];
    store
        .upsert_job(&JobRecord::new("j", dir.join("a.wav")))
        .unwrap();
    store.replace_transcript("j", &t).unwrap();
    let back = store.load_transcript("j").unwrap();
    assert_eq!(
        back.speakers.iter().map(|s| s.index).collect::<Vec<_>>(),
        vec![3, 1, 2],
        "the order the caller gave must come back unchanged"
    );
    assert!(fixtures::diff(&t, &back).is_none());
}

#[test]
fn a_transcript_with_no_speakers_round_trips() {
    // ASR-only mode is a complete product mode, not an error and not a special
    // case: it must store and load like any other transcript.
    let dir = TempDir::new("roundtrip-no-speakers");
    let store = Store::open(dir.join("v.db")).unwrap();
    let mut t = fixtures::transcript(9, 12);
    t.speakers.clear();
    for s in &mut t.segments {
        s.speaker = None;
    }
    store
        .upsert_job(&JobRecord::new("j", dir.join("a.wav")))
        .unwrap();
    store.replace_transcript("j", &t).unwrap();
    let back = store.load_transcript("j").unwrap();
    assert!(back.speakers.is_empty());
    assert!(back.segments.iter().all(|s| s.speaker.is_none()));
    assert!(fixtures::diff(&t, &back).is_none());
    let job = store.load_job("j").unwrap().unwrap();
    assert_eq!(job.speaker_count, 0);
}

#[test]
fn an_empty_transcript_round_trips() {
    let dir = TempDir::new("roundtrip-empty");
    let store = Store::open(dir.join("v.db")).unwrap();
    let t: Transcript = fixtures::empty();
    store
        .upsert_job(&JobRecord::new("j", dir.join("a.wav")))
        .unwrap();
    let stats = store.replace_transcript("j", &t).unwrap();
    assert_eq!(stats.segments, 0);
    assert_eq!(stats.words, 0);
    let back = store.load_transcript("j").unwrap();
    assert!(back.segments.is_empty());
    assert_eq!(back, t);
    // It is searchable without crashing, and finds nothing.
    assert!(
        store
            .search(&vaani_storage::SearchQuery::new("anything"))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn a_transcript_survives_being_loaded_a_thousand_times() {
    // Cheap insurance against a connection being left in a bad state by a read:
    // the pool hands the same connection to the next caller.
    let dir = TempDir::new("roundtrip-repeat");
    let store = Store::open(dir.join("v.db")).unwrap();
    let t = fixtures::transcript(17, 20);
    store
        .upsert_job(&JobRecord::new("j", dir.join("a.wav")))
        .unwrap();
    store.replace_transcript("j", &t).unwrap();
    for i in 0..1000 {
        let back = store.load_transcript("j").unwrap();
        if let Some(d) = fixtures::diff(&t, &back) {
            panic!("load {i}: {d}");
        }
    }
}

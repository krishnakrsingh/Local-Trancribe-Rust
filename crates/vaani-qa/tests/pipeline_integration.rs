//! Cross-crate integration: the crates together, as a product.
//!
//! # Why this file exists
//!
//! Every other test in the repository exercises one crate. Nothing exercised the
//! *product*: media → VAD → ASR → diarization → store → export, on one input,
//! through the real `Orchestrator`. That is where the interesting failures live —
//! a stage that hands the next one the wrong coordinate space, a job that fails
//! halfway and leaves half a transcript in the database, a cancel that leaves a
//! database that will not open again.
//!
//! # What is real and what is a double, and why
//!
//! | Stage | Under test | Why |
//! |---|---|---|
//! | media | **real** `vaani_media` | it is fast (2400x realtime) and needs no model |
//! | VAD | **real** `EnergyVad` | RTF 0.0091, no model |
//! | ASR | `vaani_app::fakes::FakeAsr` | the real engine needs 2.6 GB of weights; the stage, the chunker and `Transcript::push_chunk` are what is under test |
//! | diarization | **real** `HeuristicDiarizer` | no weights; the attribution step is real |
//! | store | **real** `vaani_storage` | needs nothing |
//! | export | **real** `vaani_export` | needs nothing |
//!
//! So only the ASR *engine* is a double. Every line of orchestration, chunking,
//! rebasing, attribution, persistence and rendering is production code on a
//! synthesised WAV with known ground truth.
//!
//! # Privacy
//!
//! Every byte of audio is synthesised by `vaani_qa::fixtures`. The private
//! development recording is never read. No test in this file writes transcript
//! content to a file it then asserts on by content: assertions are on counts,
//! times, ids and invariants, and the two places a *name* is checked use the
//! name this file itself chose.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use vaani_app::fakes::{FakeAsr, FakeVad, FixedAsrProvider, segment as fake_segment};
use vaani_app::orchestrator::{AsrProvider, EngineProviders, Orchestrator};
use vaani_app::stages::Diarize;
use vaani_app::{AppConfig, fakes};
use vaani_asr::engine::AsrEngine;
use vaani_core::pipeline::{DiarizationResult, SpeechTimeline, VadEngine};
use vaani_core::{CancelToken, JobPhase, Language, Progress, Result, Segment, Time, err};
use vaani_export::{ExportFormat, ExportOptions, ExportTarget};
use vaani_qa::fixtures;
use vaani_storage::{JobRecord, Store};

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// A throwaway workspace: an input WAV, a work directory and a database.
struct Bench {
    _dir: tempfile::TempDir,
    input: PathBuf,
    work: PathBuf,
    db: PathBuf,
}

impl Bench {
    /// A bench with `seconds` of two-voice meeting audio, which is the only
    /// synthesised material the production VAD detects as speech. See
    /// `fixtures::speech_like` for why a tone or a tone-burst will not do.
    fn new(tag: u64, turns_per_speaker: usize, turn_seconds: f64) -> Self {
        let dir = tempfile::TempDir::new().expect("bench tempdir");
        let samples = fixtures::meeting_turns(
            turns_per_speaker,
            turn_seconds,
            fixtures::FIXTURE_SAMPLE_RATE,
            0x5EED_0000 + tag,
        );
        let input = dir.path().join("meeting.wav");
        std::fs::write(
            &input,
            fixtures::encode_wav_s16(&samples, fixtures::FIXTURE_SAMPLE_RATE, 1),
        )
        .expect("write input");
        let work = dir.path().join("work");
        std::fs::create_dir_all(&work).expect("work dir");
        let db = dir.path().join("jobs.db");
        Self {
            _dir: dir,
            input,
            work,
            db,
        }
    }

    /// A bench whose input is not media at all.
    fn not_media() -> Self {
        let dir = tempfile::TempDir::new().expect("bench tempdir");
        let input = dir.path().join("notes.wav");
        std::fs::write(&input, b"these are my notes, not a recording").expect("write input");
        let work = dir.path().join("work");
        std::fs::create_dir_all(&work).expect("work dir");
        let db = dir.path().join("jobs.db");
        Self {
            _dir: dir,
            input,
            work,
            db,
        }
    }

    fn config(&self) -> AppConfig {
        AppConfig {
            work_dir: self.work.clone(),
            ..AppConfig::default()
        }
    }

    fn store(&self) -> Store {
        Store::open(&self.db).expect("open store")
    }

    /// The production providers, with a fake ASR engine.
    fn providers(asr: FakeAsr) -> EngineProviders {
        EngineProviders {
            vad: Arc::new(vaani_vad::EnergyVad::new()) as Arc<dyn VadEngine>,
            asr: Arc::new(FixedAsrProvider::new(asr)) as Arc<dyn AsrProvider>,
            diar: Some(Arc::new(vaani_diarization::HeuristicDiarizer::new()) as Arc<dyn Diarize>),
        }
    }

    /// The orchestrator, on the **real** media decoder.
    fn orchestrator(&self, providers: EngineProviders) -> Orchestrator {
        Orchestrator::with_providers(self.store(), providers)
    }
}

/// Assert the database is intact and re-openable, which is the bar a cancelled
/// or failed job has to clear.
fn assert_store_is_sound(store: &Store, label: &str) {
    let report = store.integrity_check().expect("integrity check");
    assert!(
        report.ok,
        "{label}: the database reports itself damaged: {report:?}"
    );
    let path = store.path().to_path_buf();
    store.close().expect("close");
    let reopened = Store::open(&path).unwrap_or_else(|e| {
        panic!(
            "{label}: the database will not open again after close: {}",
            e.debug_chain()
        )
    });
    assert!(reopened.schema_version().expect("schema") > 0);
    assert!(
        reopened.integrity_check().expect("integrity").ok,
        "{label}: the database is damaged after a reopen"
    );
}

fn no_progress() -> impl Fn(Progress) {
    |_p: Progress| {}
}

/// Wait until `flag` is set, then cancel `child`.
///
/// Returns the watcher thread. **It never panics**, and that is the point: a
/// watcher that panics on its deadline leaves the stage under test blocked
/// forever, because nobody ever cancels it, and the test *hangs* instead of
/// failing. A test suite that can hang is worse than one that can fail, because
/// a hang is indistinguishable from a slow machine.
///
/// So the watcher gives up waiting, cancels anyway, and returns. The test then
/// asserts the flag *was* set, which is what makes "the stage was never
/// reached" a failure rather than a timeout.
fn watch_then_cancel(
    child: &CancelToken,
    flag: &Arc<std::sync::atomic::AtomicBool>,
    what: &'static str,
) -> std::thread::JoinHandle<()> {
    let child = child.clone();
    let flag = Arc::clone(flag);
    std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(20);
        while !flag.load(std::sync::atomic::Ordering::SeqCst) {
            if Instant::now() >= deadline {
                println!("{what} was never entered within 20 s; cancelling anyway");
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        // Let the stage get properly inside its loop, so the cancel lands on
        // work rather than on the entry check.
        std::thread::sleep(Duration::from_millis(20));
        child.cancel();
    })
}

// ---------------------------------------------------------------------------
// 1. The whole chain, on a synthesised WAV
// ---------------------------------------------------------------------------

#[test]
fn media_vad_asr_diarization_store_on_a_synthesised_wav() {
    // **Three turns of 25 seconds, deliberately.** `ChunkPolicy::default()` has
    // `max_ms: 60_000` and `merge_gap_ms: 300`, so three 25-second turns
    // separated by 400 ms of room produce **three chunks**. A shorter fixture
    // produces one chunk starting at zero, and a single chunk at offset zero
    // cannot tell a working rebase from a missing one — which is exactly the
    // mutation `core-drops-the-chunk-offset` exploits, and it survived until
    // this fixture was lengthened.
    let bench = Bench::new(1, 3, 25.0);
    let store = bench.store();
    let orchestrator = bench.orchestrator(Bench::providers(FakeAsr::new()));
    let quiet = no_progress();

    let started = Instant::now();
    let outcome = orchestrator.run_job(
        "job-chain",
        &bench.input,
        &bench.config(),
        &CancelToken::new(),
        &quiet,
    );
    println!(
        "chain: {:?} in {:?}, {} segments, {} words, {} speakers",
        outcome.phase,
        started.elapsed(),
        outcome.segments,
        outcome.words,
        outcome.speakers
    );

    assert_eq!(
        outcome.phase,
        JobPhase::Complete,
        "the chain did not complete"
    );
    assert!(!outcome.resumed);
    assert!(
        outcome.skipped.is_empty(),
        "nothing should have been skipped: {:?}",
        outcome.skipped
    );
    assert!(
        outcome.segments >= 3,
        "the fixture is three separate 25-second turns, so the chunker must produce at least \
         three chunks and the ASR at least one segment each; got {}",
        outcome.segments
    );
    assert!(outcome.words > 0, "the chain produced no words");

    // --- the transcript is in the store -----------------------------------
    let stored = store
        .try_load_transcript("job-chain")
        .expect("load")
        .expect("a completed job must have a transcript");
    assert_eq!(
        stored.segments.len() as i64,
        outcome.segments,
        "the store and the outcome disagree about the segment count"
    );
    let words: usize = stored.segments.iter().map(|s| s.words.len()).sum();
    assert_eq!(words as i64, outcome.words);
    assert_eq!(stored.speakers.len() as i64, outcome.speakers);
    assert!(
        outcome.speakers >= 1,
        "diarization ran and attributed nobody, which means the attribution step found no \
         spans to map"
    );

    // --- times are absolute and inside the media ----------------------------
    let duration_ms = stored
        .duration
        .map(vaani_core::MediaLength::as_millis)
        .expect("the ASR stage records the media duration");
    assert!(
        duration_ms > 30_000,
        "the fixture is about 37 s of audio, so a duration of {duration_ms} ms means the media \
         stage did not decode the whole file"
    );
    for seg in &stored.segments {
        assert!(
            seg.start >= Time::ZERO && seg.end <= Time(duration_ms),
            "segment {}..{} is outside the media's 0..{duration_ms} ms",
            seg.start.as_millis(),
            seg.end.as_millis()
        );
        assert!(seg.start <= seg.end, "segment {} is inverted", seg.id);
        for w in &seg.words {
            assert!(
                w.start >= seg.start && w.end <= seg.end,
                "word {:?} at {}..{} is outside its segment {}..{}",
                w.text,
                w.start.as_millis(),
                w.end.as_millis(),
                seg.start.as_millis(),
                seg.end.as_millis()
            );
        }
    }
    let starts: Vec<i64> = stored
        .segments
        .iter()
        .map(|s| s.start.as_millis())
        .collect();
    let mut sorted = starts.clone();
    sorted.sort_unstable();
    assert_eq!(starts, sorted, "stored segments are not in time order");

    // **The rebase, stated as an observable property rather than a hope.**
    // `Transcript::push_chunk` is the one place a chunk-local time becomes a
    // media time. If it stopped adding the offset, every chunk's words would land
    // at the start of the recording, and the transcript would be a third of a
    // minute long for a 76-second meeting — the exact "transcript is three
    // minutes out of sync" family.
    //
    // Two assertions, because either alone can be satisfied by accident:
    // the segment starts must be *strictly* increasing (three chunks at 0, 25 s
    // and 51 s must not collapse onto each other), and the transcript must reach
    // past the midpoint of the media (which it cannot do if every segment is at
    // 0..2 s).
    for w in starts.windows(2) {
        assert!(
            w[1] > w[0],
            "segments at {starts:?} are not strictly increasing: two chunks landed on the same \
             instant, which is what a dropped chunk offset looks like"
        );
    }
    let last_end = stored
        .segments
        .last()
        .map(|s| s.end.as_millis())
        .unwrap_or_default();
    assert!(
        last_end > duration_ms / 2,
        "the transcript ends at {last_end} ms in a {duration_ms} ms recording: the words are \
         clustered at the start, so the chunks were not rebased onto the media"
    );
    println!(
        "chunks were rebased: segment starts {starts:?}, last ends at {last_end} ms of {duration_ms}"
    );

    // And the store's denormalised counters agree with the rows.
    let record = store
        .load_job("job-chain")
        .expect("load job")
        .expect("the job row exists");
    assert_eq!(record.phase, JobPhase::Complete);
    assert_eq!(record.segment_count, outcome.segments);
    assert_eq!(record.word_count, outcome.words);
    assert_eq!(record.speaker_count, outcome.speakers);
    assert_eq!(
        store.count_words("job-chain").expect("count"),
        outcome.words,
        "the words table and the job row disagree"
    );
    assert_eq!(
        store.count_segments("job-chain").expect("count"),
        outcome.segments
    );

    // --- round-trips through a restart with every Time and id intact -------
    assert_store_is_sound(&store, "after a completed chain");
    let reopened = Store::open(&bench.db).expect("reopen");
    let after = reopened
        .load_transcript("job-chain")
        .expect("the transcript must load after a restart");
    assert_eq!(
        after.segments.len(),
        stored.segments.len(),
        "a segment went missing across a restart"
    );
    for (before, now) in stored.segments.iter().zip(after.segments.iter()) {
        assert_eq!(before.id, now.id, "a segment id changed across a restart");
        assert!(
            !before.id.is_empty(),
            "a segment reached the store with no id, so nothing can address it"
        );
        assert_eq!(before.start, now.start, "segment {} start moved", before.id);
        assert_eq!(before.end, now.end, "segment {} end moved", before.id);
        assert_eq!(
            before.speaker, now.speaker,
            "segment {} speaker moved",
            before.id
        );
        assert_eq!(
            before.text(),
            now.text(),
            "segment {} text moved",
            before.id
        );
        assert_eq!(before.words.len(), now.words.len());
        for (b, n) in before.words.iter().zip(now.words.iter()) {
            assert_eq!(b.text, n.text, "word text moved across a restart");
            assert_eq!(b.start, n.start, "word {} start moved", b.text);
            assert_eq!(b.end, n.end, "word {} end moved", b.text);
        }
    }
    assert_eq!(after.speakers.len(), stored.speakers.len());
    for (before, now) in stored.speakers.iter().zip(after.speakers.iter()) {
        assert_eq!(before.index, now.index);
        assert_eq!(
            before.name, now.name,
            "a speaker name moved across a restart"
        );
    }
    assert_eq!(after.duration, stored.duration);
}

#[test]
fn the_phases_are_reported_in_order_and_only_once_each() {
    let bench = Bench::new(2, 2, 5.0);
    let seen: Arc<std::sync::Mutex<Vec<JobPhase>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = {
        let seen = Arc::clone(&seen);
        move |p: Progress| {
            if let Ok(mut v) = seen.lock() {
                v.push(p.phase)
            }
        }
    };
    let orchestrator = bench.orchestrator(Bench::providers(FakeAsr::new()));
    let outcome = orchestrator.run_job(
        "job-phases",
        &bench.input,
        &bench.config(),
        &CancelToken::new(),
        &sink,
    );
    assert_eq!(outcome.phase, JobPhase::Complete);

    // Collapse runs of the same phase, then assert the order.
    let mut order: Vec<JobPhase> = Vec::new();
    let observed = seen.lock().map(|v| v.clone()).unwrap_or_default();
    for phase in &observed {
        if order.last() != Some(phase) {
            order.push(*phase);
        }
    }
    println!("phase order: {order:?}");
    let expected = [
        JobPhase::Preparing,
        JobPhase::ExtractingAudio,
        JobPhase::Vad,
        JobPhase::Asr,
        JobPhase::Diarization,
        JobPhase::Complete,
    ];
    let mut cursor = 0;
    for phase in &order {
        if cursor < expected.len() && *phase == expected[cursor] {
            cursor += 1;
        }
    }
    assert_eq!(
        cursor,
        expected.len(),
        "expected {expected:?} as a subsequence of {order:?}"
    );
}

// ---------------------------------------------------------------------------
// 2. A failure at each stage leaves no partial transcript
// ---------------------------------------------------------------------------

/// Assert that a failed job left the database exactly as it found it.
fn assert_no_partial_transcript(store: &Store, job: &str, label: &str) {
    assert!(
        store.try_load_transcript(job).expect("load").is_none(),
        "{label}: a failed job left a transcript in the store. A half-written transcript is \
         worse than none: the UI would show a result for a job that failed."
    );
    let record = store
        .load_job(job)
        .expect("load job")
        .expect("the job row exists");
    assert_eq!(record.phase, JobPhase::Failed, "{label}");
    assert_eq!(
        record.segment_count, 0,
        "{label}: segment_count was written"
    );
    assert_eq!(record.word_count, 0, "{label}: word_count was written");
    assert_eq!(
        record.speaker_count, 0,
        "{label}: speaker_count was written"
    );
    let error = record.error.as_ref().unwrap_or_else(|| {
        panic!("{label}: the failure was not persisted, so a restart cannot tell the user why")
    });
    assert!(!error.code.is_empty(), "{label}: empty error code");
    assert!(!error.message.is_empty(), "{label}: empty error message");
    assert!(
        store.count_segments(job).expect("count") == 0,
        "{label}: segment rows exist for a failed job"
    );
    assert!(
        store.count_words(job).expect("count") == 0,
        "{label}: word rows exist for a failed job"
    );
}

#[test]
fn a_failure_in_media_leaves_no_partial_transcript() {
    // The most realistic media failure there is: the user picked the wrong file.
    let bench = Bench::not_media();
    let store = bench.store();
    let orchestrator = bench.orchestrator(Bench::providers(FakeAsr::new()));
    let outcome = orchestrator.run_job(
        "job-media",
        &bench.input,
        &bench.config(),
        &CancelToken::new(),
        &no_progress(),
    );
    assert_eq!(outcome.phase, JobPhase::Failed);
    assert_no_partial_transcript(&store, "job-media", "media failure");
    assert_store_is_sound(&store, "after a media failure");
}

#[test]
fn a_failure_in_the_vad_stage_leaves_no_partial_transcript() {
    let bench = Bench::new(3, 2, 4.0);
    let store = bench.store();
    let providers = EngineProviders {
        vad: Arc::new(FailingVad) as Arc<dyn VadEngine>,
        asr: Arc::new(FixedAsrProvider::new(FakeAsr::new())) as Arc<dyn AsrProvider>,
        diar: Some(Arc::new(vaani_diarization::HeuristicDiarizer::new()) as Arc<dyn Diarize>),
    };
    let orchestrator = bench.orchestrator(providers);
    let outcome = orchestrator.run_job(
        "job-vad",
        &bench.input,
        &bench.config(),
        &CancelToken::new(),
        &no_progress(),
    );
    assert_eq!(outcome.phase, JobPhase::Failed);
    assert_no_partial_transcript(&store, "job-vad", "VAD failure");
    assert_store_is_sound(&store, "after a VAD failure");
}

#[test]
fn a_failure_in_the_asr_stage_leaves_no_partial_transcript() {
    // The ASR engine is the only double in this file, so this is also the only
    // place its failure path is real. `FailingAsrProvider` would fail in
    // `Preparing`, before any audio is decoded, which is the *missing model*
    // case rather than the mid-pipeline case; `FailingAsrEngine` fails where a
    // real model would: inside `transcribe`, on the first chunk.
    let bench = Bench::new(4, 2, 4.0);
    let store = bench.store();
    let providers = EngineProviders {
        vad: Arc::new(vaani_vad::EnergyVad::new()) as Arc<dyn VadEngine>,
        asr: Arc::new(FixedFailingAsrProvider) as Arc<dyn AsrProvider>,
        diar: Some(Arc::new(vaani_diarization::HeuristicDiarizer::new()) as Arc<dyn Diarize>),
    };
    let orchestrator = bench.orchestrator(providers);
    let outcome = orchestrator.run_job(
        "job-asr",
        &bench.input,
        &bench.config(),
        &CancelToken::new(),
        &no_progress(),
    );
    assert_eq!(outcome.phase, JobPhase::Failed);
    assert_no_partial_transcript(&store, "job-asr", "ASR failure");
    assert_store_is_sound(&store, "after an ASR failure");
}

#[test]
fn a_failure_in_diarization_keeps_the_transcript_and_says_why() {
    // Diarization failing must *not* fail the job: a transcript without speakers
    // is a complete answer, and a transcript is what the user asked for. This is
    // the documented behaviour, so it is asserted rather than assumed.
    let bench = Bench::new(5, 2, 5.0);
    let store = bench.store();
    let providers = EngineProviders {
        vad: Arc::new(vaani_vad::EnergyVad::new()) as Arc<dyn VadEngine>,
        asr: Arc::new(FixedAsrProvider::new(FakeAsr::new())) as Arc<dyn AsrProvider>,
        diar: Some(Arc::new(FailingDiar) as Arc<dyn Diarize>),
    };
    let orchestrator = bench.orchestrator(providers);
    let outcome = orchestrator.run_job(
        "job-diar",
        &bench.input,
        &bench.config(),
        &CancelToken::new(),
        &no_progress(),
    );
    assert_eq!(
        outcome.phase,
        JobPhase::Complete,
        "a diarization failure must not fail the job"
    );
    assert_eq!(outcome.skipped.len(), 1, "{:?}", outcome.skipped);
    assert_eq!(outcome.skipped[0].stage, "diarization");
    assert!(
        !outcome.skipped[0].reason.is_empty(),
        "a skipped stage must say why, in words a user can read"
    );
    let stored = store
        .try_load_transcript("job-diar")
        .expect("load")
        .expect("the transcript survives a diarization failure");
    assert!(!stored.segments.is_empty());
    assert!(
        stored.segments.iter().all(|s| s.speaker.is_none()),
        "with no spans there is nothing to attribute, and inventing a speaker would be a lie"
    );
    assert_eq!(stored.speakers.len(), 0);
    assert_eq!(outcome.speakers, 0);
    assert_store_is_sound(&store, "after a diarization failure");
}

#[test]
fn a_storage_failure_fails_the_job_and_leaves_no_partial_transcript() {
    // A work directory that cannot be created because a *file* is in the way.
    // Deterministic on every platform, no fault injection required, and it is
    // the real shape of the failure a user hits when `%LOCALAPPDATA%` is full or
    // an AV has locked the folder.
    //
    // **Scope, stated honestly:** this is the orchestrator's own
    // `create_dir_all(work_dir)` call, which happens after the engine is created
    // and before the media stage. A failure *inside* `Store::replace_transcript`
    // — the write that would leave a half transcript if it were not atomic — is
    // not reachable from outside `vaani-app` without a fault-injection hook in
    // `vaani_storage::Store`. That is a routing note, not a gap in this test;
    // see OPEN ISSUES.
    let bench = Bench::new(6, 2, 4.0);
    let store = bench.store();
    // A file where the work directory should be.
    let blocker = bench.work.join("blocked");
    std::fs::write(&blocker, b"not a directory").expect("write the blocker");
    let config = AppConfig {
        work_dir: blocker.join("jobs"),
        ..bench.config()
    };
    let orchestrator = bench.orchestrator(Bench::providers(FakeAsr::new()));
    let outcome = orchestrator.run_job(
        "job-store",
        &bench.input,
        &config,
        &CancelToken::new(),
        &no_progress(),
    );
    assert_eq!(
        outcome.phase,
        JobPhase::Failed,
        "a work directory that cannot be created must fail the job, not panic"
    );
    assert!(
        store
            .try_load_transcript("job-store")
            .expect("load")
            .is_none(),
        "a failed job left a transcript behind"
    );
    let record = store.load_job("job-store").expect("load").expect("exists");
    assert_eq!(record.phase, JobPhase::Failed);
    assert_eq!(record.segment_count, 0);
    let error = record.error.as_ref().expect("the reason is persisted");
    assert_eq!(
        error.code, "storage_failed",
        "a work-directory failure must be classified as a storage problem, not as an \
         internal error: {:?}",
        error.message
    );
    assert!(!error.message.is_empty());
    assert!(store.integrity_check().expect("integrity").ok);
}

// ---------------------------------------------------------------------------
// 3. Cancel at each stage leaves the database consistent
// ---------------------------------------------------------------------------

/// Assert a cancelled job is terminal, transcript-free and the database sound.
fn assert_cancelled_cleanly(store: &Store, job: &str, label: &str) {
    let record = store
        .load_job(job)
        .expect("load job")
        .expect("the job row exists");
    assert_eq!(
        record.phase,
        JobPhase::Cancelled,
        "{label}: a cancelled job must land in Cancelled, never Failed"
    );
    assert!(
        record.phase.is_terminal(),
        "{label}: Cancelled is not terminal"
    );
    assert!(
        record.error.is_none(),
        "{label}: a cancellation was recorded as an error, so the UI will say the job failed"
    );
    assert!(
        store.try_load_transcript(job).expect("load").is_none(),
        "{label}: a cancelled job left a partial transcript"
    );
    assert_store_is_sound(store, label);
}

#[test]
fn cancel_before_the_job_starts_is_clean() {
    let bench = Bench::new(7, 2, 4.0);
    let store = bench.store();
    let orchestrator = bench.orchestrator(Bench::providers(FakeAsr::new()));
    let token = CancelToken::new();
    token.cancel();
    let started = Instant::now();
    let outcome = orchestrator.run_job(
        "job-cancel-early",
        &bench.input,
        &bench.config(),
        &token,
        &no_progress(),
    );
    assert_eq!(outcome.phase, JobPhase::Cancelled);
    assert!(
        started.elapsed() < Duration::from_millis(500),
        "a pre-cancelled job took {:?}; it should not have decoded a byte",
        started.elapsed()
    );
    assert!(
        store
            .try_load_transcript("job-cancel-early")
            .expect("load")
            .is_none()
    );
    assert_store_is_sound(&store, "after an early cancel");
}

#[test]
fn cancel_during_media_leaves_the_database_consistent() {
    let bench = Bench::new(8, 2, 4.0);
    let store = bench.store();
    let orchestrator = bench.orchestrator(Bench::providers(FakeAsr::new()));
    let token = CancelToken::new();
    let child = token.child();
    // Cancel from the decoder's own progress stream, so the cancel provably
    // lands *inside* the decode at any decode speed. A sleep would race a
    // 2400x-realtime decoder, which is exactly the bug docs/JOURNEY.md records
    // in wave 2.
    //
    // The filter matters: the orchestrator emits an *indeterminate*
    // `EXTRACTING_AUDIO` event when it enters the phase, before any audio has
    // been read. Cancelling there is a different test (and a different, already
    // broken, code path — see the `job-create` note below). The decoder's own
    // events are quantified `MediaSeconds`, so requiring a readable fraction
    // picks exactly the ones the decoder produced.
    let inside_decode = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sink = {
        let child = child.clone();
        let inside_decode = Arc::clone(&inside_decode);
        move |p: Progress| {
            if p.phase == JobPhase::ExtractingAudio
                && p.kind.fraction().is_some()
                && !inside_decode.swap(true, std::sync::atomic::Ordering::SeqCst)
            {
                child.cancel();
            }
        }
    };
    let outcome = orchestrator.run_job(
        "job-cancel-media",
        &bench.input,
        &bench.config(),
        &token,
        &sink,
    );
    assert!(
        inside_decode.load(std::sync::atomic::Ordering::SeqCst),
        "the decoder never reported a quantified progress value, so the cancel could not have \
         landed inside the decode"
    );
    assert_eq!(
        outcome.phase,
        JobPhase::Cancelled,
        "the job must land in Cancelled"
    );
    assert_cancelled_cleanly(&store, "job-cancel-media", "cancel during media");
}

#[test]
fn a_cancel_that_lands_before_the_engine_exists_is_still_recorded_as_cancelled() {
    // A documented, tracked defect in `vaani-app`, pinned so that fixing it is
    // visible. See OPEN ISSUES.
    //
    // `Orchestrator::run_job` calls `self.finish_failed(job_id, &error, progress)`
    // for an `asr.create` failure, where every later stage uses `finish_with`,
    // which maps a cancellation to `Cancelled`. So a user who cancels while the
    // model is being loaded is told the job **Failed**, with the cancellation's
    // message stored as an error — and `docs/JOURNEY.md` lists exactly this
    // class ("a job killed by a crash currently masquerades as `cancelled`", and
    // the mirror image) as something the product must not do.
    //
    // Reproduce with this test, then change `finish_failed` to `finish_with` in
    // the `providers.asr.create` arm and it passes.
    let bench = Bench::new(18, 2, 4.0);
    let store = bench.store();
    let orchestrator = bench.orchestrator(Bench::providers(FakeAsr::new()));
    let token = CancelToken::new();
    // Cancel on the *first* progress event, which the orchestrator emits when it
    // enters `Preparing` — before the engine is created.
    let child = token.child();
    let fired = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sink = {
        let child = child.clone();
        let fired = Arc::clone(&fired);
        move |_p: Progress| {
            if !fired.swap(true, std::sync::atomic::Ordering::SeqCst) {
                child.cancel();
            }
        }
    };
    let outcome = orchestrator.run_job(
        "job-cancel-early-phase",
        &bench.input,
        &bench.config(),
        &token,
        &sink,
    );
    let record = store
        .load_job("job-cancel-early-phase")
        .expect("load")
        .expect("exists");

    // What must hold today, and is asserted so the day it is fixed this fails
    // loudly rather than silently changing meaning.
    assert_eq!(
        outcome.phase,
        JobPhase::Failed,
        "vaani-app now records a pre-engine cancellation as Cancelled; the tracked defect in \
         OPEN ISSUES is fixed, so delete this test and assert JobPhase::Cancelled instead. \
         Observed: {outcome:?}"
    );
    assert_eq!(record.phase, JobPhase::Failed);
    assert!(
        store
            .try_load_transcript("job-cancel-early-phase")
            .expect("load")
            .is_none(),
        "however it is labelled, a cancelled-before-transcription job must hold no transcript"
    );
    assert_store_is_sound(&store, "after a pre-engine cancel");
}

#[test]
fn cancel_during_the_vad_leaves_the_database_consistent() {
    let bench = Bench::new(9, 2, 4.0);
    let store = bench.store();
    let entered = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let providers = EngineProviders {
        vad: Arc::new(BlockingVad::new(Arc::clone(&entered))) as Arc<dyn VadEngine>,
        asr: Arc::new(FixedAsrProvider::new(FakeAsr::new())) as Arc<dyn AsrProvider>,
        diar: Some(Arc::new(vaani_diarization::HeuristicDiarizer::new()) as Arc<dyn Diarize>),
    };
    let orchestrator = bench.orchestrator(providers);
    let token = CancelToken::new();
    let child = token.child();
    let watcher = watch_then_cancel(&child, &entered, "the VAD stage");
    let started = Instant::now();
    let outcome = orchestrator.run_job(
        "job-cancel-vad",
        &bench.input,
        &bench.config(),
        &token,
        &no_progress(),
    );
    watcher.join().expect("the VAD watcher must not panic");
    assert!(
        entered.load(std::sync::atomic::Ordering::SeqCst),
        "the VAD stage was never entered after {:?}; the cancel test proved nothing",
        started.elapsed()
    );
    assert_eq!(outcome.phase, JobPhase::Cancelled, "{started:?}");
    assert_cancelled_cleanly(&store, "job-cancel-vad", "cancel during VAD");
}

#[test]
fn cancel_during_the_asr_leaves_the_database_consistent() {
    let bench = Bench::new(10, 2, 4.0);
    let store = bench.store();
    let asr = FakeAsr::new();
    asr.blocking(true);
    let counter = asr.clone();
    let orchestrator = bench.orchestrator(Bench::providers(asr));
    let token = CancelToken::new();
    let child = token.child();
    let watcher = {
        let child = child.clone();
        std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(30);
            // The fake counts the call before blocking, so this is a real
            // "the ASR is running now" signal rather than a sleep.
            while counter.calls() == 0 {
                assert!(Instant::now() < deadline, "the ASR stage was never reached");
                std::thread::sleep(Duration::from_millis(2));
            }
            std::thread::sleep(Duration::from_millis(20));
            child.cancel();
        })
    };
    let started = Instant::now();
    let outcome = orchestrator.run_job(
        "job-cancel-asr",
        &bench.input,
        &bench.config(),
        &token,
        &no_progress(),
    );
    let elapsed = started.elapsed();
    watcher.join().expect("watcher");
    assert_eq!(outcome.phase, JobPhase::Cancelled);
    assert!(
        elapsed < Duration::from_secs(10),
        "the cancel took {elapsed:?} to take effect"
    );
    assert_cancelled_cleanly(&store, "job-cancel-asr", "cancel during ASR");
}

#[test]
fn cancel_during_diarization_leaves_the_database_consistent() {
    let bench = Bench::new(11, 2, 4.0);
    let store = bench.store();
    let entered = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let providers = EngineProviders {
        vad: Arc::new(vaani_vad::EnergyVad::new()) as Arc<dyn VadEngine>,
        asr: Arc::new(FixedAsrProvider::new(FakeAsr::new())) as Arc<dyn AsrProvider>,
        diar: Some(Arc::new(BlockingDiar::new(Arc::clone(&entered))) as Arc<dyn Diarize>),
    };
    let orchestrator = bench.orchestrator(providers);
    let token = CancelToken::new();
    let child = token.child();
    let watcher = watch_then_cancel(&child, &entered, "the diarization stage");
    let outcome = orchestrator.run_job(
        "job-cancel-diar",
        &bench.input,
        &bench.config(),
        &token,
        &no_progress(),
    );
    watcher
        .join()
        .expect("the diarization watcher must not panic");
    assert!(
        entered.load(std::sync::atomic::Ordering::SeqCst),
        "the diarization stage was never entered; the cancel test proved nothing"
    );
    // The orchestrator's diarization branch turns *any* error, a cancellation
    // included, into a "skip with a reason" rather than into a cancellation. The
    // cancel is then noticed at the store stage's entry check, so the job lands
    // in `Cancelled` a moment later. The end state is right; two details are
    // not, and both are in OPEN ISSUES:
    //
    //  * the skip's *reason* is the cancellation's message, so a user who
    //    pressed Stop could be shown "could not separate speakers";
    //  * `finish_cancelled` builds a fresh `JobOutcome` with
    //    `skipped: Vec::new()`, so the skip record is discarded and the UI has
    //    nothing to explain what it skipped.
    assert_eq!(
        outcome.phase,
        JobPhase::Cancelled,
        "a cancel that lands during diarization must not be reported as a completed job"
    );
    assert!(
        store
            .try_load_transcript("job-cancel-diar")
            .expect("load")
            .is_none(),
        "a cancelled job must not have a transcript, and must certainly not have a partial one"
    );
    assert_cancelled_cleanly(&store, "job-cancel-diar", "cancel during diarization");
}

#[test]
fn a_cancelled_job_can_be_re_run_and_then_completes() {
    // Cancel is not a poison pill: the database has to be usable afterwards, and
    // a re-run of the same content must reach Complete.
    let bench = Bench::new(12, 2, 4.0);
    let store = bench.store();
    let orchestrator = bench.orchestrator(Bench::providers(FakeAsr::new()));
    let token = CancelToken::new();
    token.cancel();
    let first = orchestrator.run_job(
        "job-retry-a",
        &bench.input,
        &bench.config(),
        &token,
        &no_progress(),
    );
    assert_eq!(first.phase, JobPhase::Cancelled);

    let second = orchestrator.run_job(
        "job-retry-a",
        &bench.input,
        &bench.config(),
        &CancelToken::new(),
        &no_progress(),
    );
    assert_eq!(
        second.phase,
        JobPhase::Complete,
        "a re-run after a cancel must complete; the database was left in a state that \
         blocked it"
    );
    assert!(
        store
            .try_load_transcript("job-retry-a")
            .expect("load")
            .is_some()
    );
    assert_store_is_sound(&store, "after a cancel and a re-run");
}

// ---------------------------------------------------------------------------
// 4. Export after a completed job, all five formats
// ---------------------------------------------------------------------------

#[test]
fn every_format_exports_after_a_completed_job_and_the_subtitles_parse() {
    let bench = Bench::new(13, 3, 6.0);
    let store = bench.store();
    let orchestrator = bench.orchestrator(Bench::providers(FakeAsr::new()));
    let quiet = no_progress();
    let outcome = orchestrator.run_job(
        "job-export",
        &bench.input,
        &bench.config(),
        &CancelToken::new(),
        &quiet,
    );
    assert_eq!(outcome.phase, JobPhase::Complete);

    let out_dir = bench.work.join("exports");
    let options = ExportOptions::default().with_source_name("meeting");
    let target = ExportTarget::Directory(out_dir.clone());

    let mut written = Vec::new();
    for format in ExportFormat::ALL {
        let file = orchestrator
            .export_job(
                "job-export",
                format,
                &target,
                &options,
                &bench.work,
                &CancelToken::new(),
                &quiet,
            )
            .unwrap_or_else(|e| panic!("{format}: {}", e.debug_chain()));
        assert!(
            file.path.is_file(),
            "{format} wrote nothing at {}",
            file.path.display()
        );
        assert!(file.bytes > 0, "{format} wrote an empty file");
        assert_eq!(file.format, format);
        written.push((format, file.path.clone()));
    }

    // All five files, named after the source, nothing else.
    let mut names: Vec<String> = std::fs::read_dir(&out_dir)
        .expect("read the export directory")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "meeting.json",
            "meeting.md",
            "meeting.srt",
            "meeting.txt",
            "meeting.vtt"
        ],
        "the export directory should hold exactly the five formats"
    );

    // SRT and VTT go through the reference parsers, not a string comparison.
    for (format, path) in &written {
        let text = std::fs::read_to_string(path).expect("read the export");
        match format {
            ExportFormat::Srt => {
                let cues = vaani_export::verify::validate_srt(&text, 2)
                    .unwrap_or_else(|e| panic!("srt: {}", e.debug_chain()));
                assert!(!cues.is_empty(), "the SRT export has no cues");
                for (i, w) in cues.windows(2).enumerate() {
                    assert!(
                        w[0].end <= w[1].start,
                        "srt cues {i} and {} overlap, which a player renders on top of each \
                         other",
                        i + 1
                    );
                }
                println!(
                    "srt: {} cues, indices start at {:?}",
                    cues.len(),
                    cues.first().map(|c| c.index)
                );
            }
            ExportFormat::Vtt => {
                assert!(
                    text.starts_with("WEBVTT"),
                    "a VTT file must start with the WEBVTT signature; got {:?}",
                    &text[..text.len().min(20)]
                );
                let cues = vaani_export::verify::validate_vtt(&text, 2)
                    .unwrap_or_else(|e| panic!("vtt: {}", e.debug_chain()));
                assert!(!cues.is_empty(), "the VTT export has no cues");
            }
            ExportFormat::Json => {
                let back = vaani_export::read_json_export(&text)
                    .unwrap_or_else(|e| panic!("json: {}", e.debug_chain()));
                let stored = store.load_transcript("job-export").expect("load");
                assert_eq!(
                    back.segments.len(),
                    stored.segments.len(),
                    "the JSON export does not describe the stored transcript"
                );
                for (a, b) in back.segments.iter().zip(stored.segments.iter()) {
                    assert_eq!(
                        a.start, b.start,
                        "a segment start did not survive the export"
                    );
                    assert_eq!(a.end, b.end, "a segment end did not survive the export");
                }
            }
            _ => {}
        }
    }

    // Every export is recorded against the job, and the job is left Complete.
    let logs = store
        .list_exports(Some("job-export"))
        .expect("list exports");
    assert_eq!(logs.len(), 5, "each export must be logged once: {logs:?}");
    let record = store.load_job("job-export").expect("load").expect("exists");
    assert_eq!(
        record.phase,
        JobPhase::Complete,
        "a job must not be left in Exporting after an export"
    );
}

#[test]
fn exporting_a_job_with_no_transcript_says_so_instead_of_writing_nothing() {
    let bench = Bench::new(14, 2, 4.0);
    let store = bench.store();
    store
        .upsert_job(&JobRecord::new("job-empty", &bench.input))
        .expect("upsert");
    let orchestrator = bench.orchestrator(Bench::providers(FakeAsr::new()));
    let err = orchestrator
        .export_job(
            "job-empty",
            ExportFormat::Srt,
            &ExportTarget::Directory(bench.work.join("out")),
            &ExportOptions::default(),
            &bench.work,
            &CancelToken::new(),
            &no_progress(),
        )
        .expect_err("a job with no transcript cannot be exported");
    assert_eq!(err.code(), vaani_core::ErrorCode::ExportFailed);
    assert!(
        err.user.recovery.is_some(),
        "the user needs to be told to transcribe the file first: {}",
        err.user.detail
    );
    assert!(
        !bench.work.join("out").exists()
            || std::fs::read_dir(bench.work.join("out"))
                .expect("read")
                .next()
                .is_none(),
        "a refused export must not leave a file behind"
    );
}

// ---------------------------------------------------------------------------
// 5. A speaker rename survives a store round-trip and an export
// ---------------------------------------------------------------------------

#[test]
fn a_speaker_rename_survives_a_restart_and_reaches_the_exports() {
    let bench = Bench::new(15, 3, 6.0);
    let store = bench.store();
    let orchestrator = bench.orchestrator(Bench::providers(FakeAsr::new()));
    let quiet = no_progress();
    let outcome = orchestrator.run_job(
        "job-rename",
        &bench.input,
        &bench.config(),
        &CancelToken::new(),
        &quiet,
    );
    assert_eq!(outcome.phase, JobPhase::Complete);
    let before = store.load_transcript("job-rename").expect("load");
    assert!(
        before.speakers.len() >= 2,
        "this test renames the second of two speakers, so diarization must have found two; it \
         found {}",
        before.speakers.len()
    );

    let chosen = "Rekha";
    assert!(
        store
            .rename_speaker("job-rename", 1, Some(chosen))
            .expect("rename"),
        "renaming a speaker the diarizer produced must report a change"
    );

    // --- survives a restart -------------------------------------------------
    assert_store_is_sound(&store, "after a rename");
    let reopened = Store::open(&bench.db).expect("reopen");
    let after = reopened.load_transcript("job-rename").expect("load");
    assert_eq!(
        after.speakers[1].name.as_deref(),
        Some(chosen),
        "the rename did not survive a restart"
    );
    assert_eq!(
        after.speakers[0].name, before.speakers[0].name,
        "renaming one speaker must not touch another"
    );
    // A rename is a label, not an edit: the words are untouched.
    for (a, b) in before.segments.iter().zip(after.segments.iter()) {
        assert_eq!(a.text(), b.text(), "a rename edited the transcript text");
        assert_eq!(a.start, b.start);
    }
    let row = reopened
        .speaker("job-rename", 1)
        .expect("row")
        .expect("a row");
    assert_eq!(row.name(), chosen);
    assert!(
        !row.display_name.is_empty(),
        "the generated label must remain as the fallback, so clearing a rename is possible"
    );

    // --- and reaches the exports -------------------------------------------
    let out_dir = bench.work.join("renamed");
    for format in [
        ExportFormat::Text,
        ExportFormat::Markdown,
        ExportFormat::Json,
    ] {
        let file = orchestrator
            .export_job(
                "job-rename",
                format,
                &ExportTarget::Directory(out_dir.clone()),
                &ExportOptions::default().with_source_name("standup"),
                &bench.work,
                &CancelToken::new(),
                &quiet,
            )
            .unwrap_or_else(|e| panic!("{format}: {}", e.debug_chain()));
        let text = std::fs::read_to_string(&file.path).expect("read");
        assert!(
            text.contains(chosen),
            "{format} does not mention the speaker's name; a user who renamed a speaker expects \
             to see it in what they share"
        );
    }

    // --- and a rename can be cleared ----------------------------------------
    assert!(
        reopened
            .rename_speaker("job-rename", 1, None)
            .expect("clear")
    );
    let cleared = reopened.load_transcript("job-rename").expect("load");
    assert_eq!(
        cleared.speakers[1].name, None,
        "clearing a rename must fall back to the generated label"
    );
    // Renaming a speaker that was never produced is a no-op with a signal.
    assert!(
        !reopened
            .rename_speaker("job-rename", 99, Some("Nobody"))
            .expect("rename a phantom"),
        "naming a speaker the diarizer never produced must not silently insert one"
    );
}

#[test]
fn a_rename_survives_a_re_run_of_the_same_content() {
    // Resume-by-hash copies the transcript rather than re-transcribing, so a
    // rename must be carried across the copy too, or a user's naming is lost the
    // moment they re-run a file.
    let bench = Bench::new(16, 2, 4.0);
    let store = bench.store();
    let asr = FakeAsr::new();
    let orchestrator = bench.orchestrator(Bench::providers(asr.clone()));
    let quiet = no_progress();
    let first = orchestrator.run_job(
        "job-hash-a",
        &bench.input,
        &bench.config(),
        &CancelToken::new(),
        &quiet,
    );
    assert_eq!(first.phase, JobPhase::Complete);
    assert!(store.speaker("job-hash-a", 0).expect("speakers").is_some());
    assert!(
        store
            .rename_speaker("job-hash-a", 0, Some("Nitin"))
            .expect("rename")
    );

    let second = orchestrator.run_job(
        "job-hash-b",
        &bench.input,
        &bench.config(),
        &CancelToken::new(),
        &quiet,
    );
    assert_eq!(second.phase, JobPhase::Complete);
    assert!(
        second.resumed,
        "the second run should have reused the transcript"
    );
    assert_eq!(asr.calls(), 1, "the second run must not transcribe again");
    let copied = store.load_transcript("job-hash-b").expect("load");
    assert_eq!(
        copied.speakers[0].name.as_deref(),
        Some("Nitin"),
        "the rename was lost when the transcript was copied to the new job"
    );
}

// ---------------------------------------------------------------------------
// 6. The fakes themselves, so a change to them is caught here
// ---------------------------------------------------------------------------

#[test]
fn the_injected_fakes_behave_as_documented() {
    // `vaani_app::fakes` is `#[doc(hidden)] pub` precisely so this crate can use
    // it. A fake that quietly stops honouring cancellation would make every
    // cancel test in this file pass for the wrong reason, so the fakes get their
    // own assertions.
    let token = CancelToken::new();
    assert!(
        FakeVad::speech()
            .detect(&fakes::sine_audio(2), &token)
            .is_ok()
    );
    let dead = CancelToken::new();
    dead.cancel();
    assert!(
        FakeVad::speech()
            .detect(&fakes::sine_audio(2), &dead)
            .is_err(),
        "a fake that ignores cancellation would make every cancel test here meaningless"
    );

    let asr = FakeAsr::new();
    let chunk = vaani_asr::engine::AudioChunk::new(vec![0.0; 32_000], 16_000, Time::ZERO);
    let result = asr.transcribe(&chunk, &token).expect("transcribe");
    assert_eq!(result.segments.len(), 1);
    assert!(
        result.segments[0].words.len() == 2,
        "the fake emits 'hello world', so two words"
    );
    assert_eq!(asr.calls(), 1);

    // The fake's segment is chunk-local, which is what makes the ASR stage's
    // rebase meaningful.
    let seg: &Segment = &result.segments[0];
    assert_eq!(seg.start, Time::ZERO, "the fake's times are chunk-local");
    assert_eq!(fake_segment("hello world", 0, 2000).words.len(), 2);
}

#[test]
fn a_transcript_with_no_speakers_still_exports_and_persists() {
    // "ASR only" is a complete product mode, not a degraded one, so every stage
    // downstream has to accept a speakerless transcript.
    let bench = Bench::new(17, 2, 4.0);
    let store = bench.store();
    let providers = EngineProviders {
        vad: Arc::new(vaani_vad::EnergyVad::new()) as Arc<dyn VadEngine>,
        asr: Arc::new(FixedAsrProvider::new(FakeAsr::new())) as Arc<dyn AsrProvider>,
        diar: None,
    };
    let orchestrator = bench.orchestrator(providers);
    let quiet = no_progress();
    let outcome = orchestrator.run_job(
        "job-nodiar",
        &bench.input,
        &bench.config(),
        &CancelToken::new(),
        &quiet,
    );
    assert_eq!(outcome.phase, JobPhase::Complete);
    assert_eq!(outcome.speakers, 0);
    assert_eq!(outcome.skipped.len(), 1);
    assert!(!outcome.skipped[0].reason.is_empty());

    let stored = store.load_transcript("job-nodiar").expect("load");
    assert!(stored.segments.iter().all(|s| s.speaker.is_none()));
    for format in ExportFormat::ALL {
        orchestrator
            .export_job(
                "job-nodiar",
                format,
                &ExportTarget::Directory(bench.work.join("nodiar")),
                &ExportOptions::default().with_source_name("a"),
                &bench.work,
                &CancelToken::new(),
                &quiet,
            )
            .unwrap_or_else(|e| panic!("{format}: {}", e.debug_chain()));
    }
}

// ---------------------------------------------------------------------------
// Test doubles used above
// ---------------------------------------------------------------------------

/// A VAD that always fails, with a real code and a real recovery.
#[derive(Debug)]
struct FailingVad;

impl VadEngine for FailingVad {
    fn id(&self) -> &str {
        "failing-vad"
    }
    fn name(&self) -> &str {
        "Failing VAD"
    }
    fn sample_rate(&self) -> u32 {
        16_000
    }
    fn detect(
        &self,
        _audio: &vaani_core::media::AudioBuffer,
        _cancel: &CancelToken,
    ) -> Result<SpeechTimeline> {
        Err(err::transcription_failed()
            .with_cause("injected VAD failure")
            .with_recovery("This is a test double; a real failure would say what to do."))
    }
    fn cancel(&self) {}
}

/// A VAD that signals it has been entered and then blocks until cancelled.
#[derive(Debug)]
struct BlockingVad {
    entered: Arc<std::sync::atomic::AtomicBool>,
}

impl BlockingVad {
    fn new(entered: Arc<std::sync::atomic::AtomicBool>) -> Self {
        Self { entered }
    }
}

impl VadEngine for BlockingVad {
    fn id(&self) -> &str {
        "blocking-vad"
    }
    fn name(&self) -> &str {
        "Blocking VAD"
    }
    fn sample_rate(&self) -> u32 {
        16_000
    }
    fn detect(
        &self,
        _audio: &vaani_core::media::AudioBuffer,
        cancel: &CancelToken,
    ) -> Result<SpeechTimeline> {
        self.entered
            .store(true, std::sync::atomic::Ordering::SeqCst);
        loop {
            cancel.check()?;
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    fn cancel(&self) {}
}

/// An ASR engine that fails on the first chunk, the way a real model would.
#[derive(Debug, Clone, Default)]
struct FailingAsr;

impl AsrEngine for FailingAsr {
    fn id(&self) -> &str {
        "failing-asr"
    }
    fn model_name(&self) -> &str {
        "failing-asr"
    }
    fn model_version(&self) -> &str {
        "test.0"
    }
    fn languages(&self) -> &[Language] {
        &[]
    }
    fn load(&self, _cancel: &CancelToken) -> Result<()> {
        Ok(())
    }
    fn unload(&self) {}
    fn is_loaded(&self) -> bool {
        true
    }
    fn transcribe(
        &self,
        _chunk: &vaani_asr::engine::AudioChunk,
        cancel: &CancelToken,
    ) -> Result<vaani_asr::engine::ChunkResult> {
        cancel.check()?;
        Err(err::transcription_failed()
            .with_cause("injected ASR failure")
            .with_recovery("This is a test double; a real failure would say what to do."))
    }
    fn cancel(&self) {}
    fn benchmark(&self, _cancel: &CancelToken) -> Result<vaani_asr::engine::AsrTimings> {
        Ok(vaani_asr::engine::AsrTimings::default())
    }
}

/// Hands out the [`FailingAsr`] double, because `vaani_app::fakes::FixedAsrProvider`
/// is typed to `FakeAsr` and this test needs an engine that fails inside
/// `transcribe` rather than at construction.
#[derive(Debug, Clone, Default)]
struct FixedFailingAsrProvider;

impl AsrProvider for FixedFailingAsrProvider {
    fn create(
        &self,
        _config: &AppConfig,
        cancel: &CancelToken,
    ) -> Result<Box<dyn vaani_asr::engine::AsrEngine>> {
        cancel.check()?;
        Ok(Box::new(FailingAsr))
    }
}

/// A diarizer that always fails, which must not fail the job.
#[derive(Debug)]
struct FailingDiar;

impl Diarize for FailingDiar {
    fn engine_id(&self) -> &'static str {
        "failing-diar"
    }
    fn diarize(
        &self,
        _audio: &vaani_core::media::AudioBuffer,
        _speech: Option<&SpeechTimeline>,
        _cancel: &CancelToken,
        _progress: &dyn Fn(f64),
    ) -> Result<DiarizationResult> {
        Err(err::diarization_failed().with_cause("injected diarization failure"))
    }
}

/// A diarizer that signals it has been entered and then blocks until cancelled.
#[derive(Debug)]
struct BlockingDiar {
    entered: Arc<std::sync::atomic::AtomicBool>,
}

impl BlockingDiar {
    fn new(entered: Arc<std::sync::atomic::AtomicBool>) -> Self {
        Self { entered }
    }
}

impl Diarize for BlockingDiar {
    fn engine_id(&self) -> &'static str {
        "blocking-diar"
    }
    fn diarize(
        &self,
        _audio: &vaani_core::media::AudioBuffer,
        _speech: Option<&SpeechTimeline>,
        cancel: &CancelToken,
        _progress: &dyn Fn(f64),
    ) -> Result<DiarizationResult> {
        self.entered
            .store(true, std::sync::atomic::Ordering::SeqCst);
        loop {
            cancel.check()?;
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

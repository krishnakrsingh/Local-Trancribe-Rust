//! Regression chain over committed public fixtures (`tests/fixtures/qa/`).
//!
//! The full cross-crate path on small, committed, non-sensitive audio:
//! decode → VAD → chunk → diarization (with VAD timeline) → attribution →
//! storage round-trip → all five export formats. The private 12-minute meeting
//! and `scratch/clip.wav` are never touched; if the committed fixtures are
//! absent the test skips loudly instead of passing vacuously.
//!
//! The speech fixture is onset-rich harmonic-comb voices over a room floor
//! (never digital silence: gaps of mathematical zero break the VAD's adaptive
//! floor in a way no real capture would), alternating two base pitches so two
//! speakers are separable.

use std::path::PathBuf;

use vaani_core::pipeline::{DiarizationEngine, VadEngine};
use vaani_core::{CancelToken, EngineStamp, JobPhase, Transcript};
use vaani_qa::conformance::check_export_stage;

fn fixture(name: &str) -> Option<PathBuf> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/qa")
        .join(name);
    if path.is_file() {
        Some(path)
    } else {
        println!("SKIP: regression_chain — committed fixture {name} absent");
        None
    }
}

#[test]
fn cross_crate_chain_on_committed_fixtures() {
    let Some(turns) = fixture("qa-turns-8s.wav") else {
        return;
    };
    let Some(silence) = fixture("qa-silence-1s.wav") else {
        return;
    };
    let cancel = CancelToken::new();

    // 1. Media: decode to mono 16 kHz with absolute times.
    let audio = vaani_media::decode_to_mono16k(&turns, &cancel).expect("decode turns");
    assert_eq!((audio.sample_rate, audio.channels), (16_000, 1));
    assert_eq!(audio.samples.len(), 121_600);

    // 2. VAD: speech found in turns, none in silence.
    let vad = vaani_vad::EnergyVad::new();
    let timeline = vad.detect(&audio, &cancel).expect("vad turns");
    assert!(!timeline.segments.is_empty(), "turns must contain speech");
    let quiet = vaani_media::decode_to_mono16k(&silence, &cancel).expect("decode silence");
    let quiet_timeline = vad.detect(&quiet, &cancel).expect("vad silence");
    assert!(quiet_timeline.segments.is_empty());

    // 3. Chunking: speech produces at least one chunk, silence none.
    let policy = vaani_vad::ChunkPolicy::default();
    let chunks = vaani_vad::Chunker::new(policy).plan(&timeline, audio.duration());
    assert!(!chunks.is_empty());
    let no_chunks = vaani_vad::Chunker::new(policy).plan(&quiet_timeline, quiet.duration());
    assert!(no_chunks.is_empty());

    // 4. Diarization reuses the VAD timeline (the real pipeline path).
    let diar = vaani_diarization::HeuristicDiarizer::new();
    diar.load(&cancel).expect("diar load");
    let result = diar
        .diarize_with_timeline(&audio, Some(&timeline), &cancel, &|_| {})
        .expect("diarize");
    assert!(result.num_speakers >= 1);

    // 5. Attribution onto a fake ASR transcript, then persistence.
    let mut transcript = Transcript::new(EngineStamp::asr("qa-regression", "0"));
    for chunk in &chunks {
        let seg = vaani_core::Segment::new(chunk.start, chunk.end).with_words(vec![
            vaani_core::Word::new("regression", chunk.start, chunk.end),
        ]);
        transcript.push_chunk(vaani_core::Time::ZERO, vec![seg]);
    }
    vaani_diarization::attribute_speakers(&mut transcript, &result);

    let dir = tempfile::TempDir::new().expect("tempdir");
    let store = vaani_storage::Store::open(dir.path().join("qa.db")).expect("open store");
    let job_id = vaani_storage::new_job_id();
    let mut job = vaani_storage::JobRecord::new(&job_id, &turns);
    job.phase = JobPhase::Complete;
    store.upsert_job(&job).expect("upsert");
    store
        .replace_transcript(&job_id, &transcript)
        .expect("store");
    let back = store.load_transcript(&job_id).expect("load");
    assert_eq!(back.segments.len(), transcript.segments.len());

    // 6. Export conformance on the persisted transcript.
    let stage = vaani_qa::conformance::VaaniExportStage;
    let report = check_export_stage(&stage, &back);
    assert!(report.ok(), "export conformance broke on the chain output");
    println!(
        "chain: {} chunks, {} speakers, {} segments, {}/{} export checks passed",
        chunks.len(),
        result.num_speakers,
        back.segments.len(),
        report.passed(),
        report.outcomes.len()
    );
}

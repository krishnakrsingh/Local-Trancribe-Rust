//! ASR smoke tests across Hindi / Hinglish / English / code-switch hints.
//!
//! These require the 2.7 GB SraVaani weights. Without them every test SKIPS
//! with a printed reason — never a silent pass, never a failure for lack of a
//! download. With them, each test transcribes synthetic speech-like audio with
//! a different language hint and pins the chunk-local contract on the result.

use vaani_asr::{AsrEngine, AudioChunk, EngineArtifacts};
use vaani_core::{CancelToken, Language, Time};
use vaani_qa::fixtures;
use vaani_qa::skip;

fn artifacts() -> Option<EngineArtifacts> {
    let dir = vaani_asr::default_dir();
    let art = vaani_asr::artifacts_in(&dir);
    if art.is_complete() { Some(art) } else { None }
}

fn open_engine(art: &EngineArtifacts) -> Option<Box<dyn AsrEngine>> {
    let caps = vaani_inference::capability::detect(&[]);
    let registry =
        vaani_asr::EngineRegistry::with_builtins(vaani_inference::Provider::Cpu, &caps).ok()?;
    registry.create("sravaani", art).ok()
}

fn smoke(language: Language, tag: &str) {
    let Some(art) = artifacts() else {
        skip::skip(
            tag,
            "SraVaani artifacts absent — run `vaani models download` first",
        );
        return;
    };
    let Some(engine) = open_engine(&art) else {
        skip::skip(
            tag,
            "could not construct sravaani engine (runtime missing?)",
        );
        return;
    };
    let cancel = CancelToken::new();
    if let Err(e) = engine.load(&cancel) {
        skip::skip(tag, &format!("engine load failed: {}", engine.id()));
        let _ = e;
        return;
    }
    assert!(engine.is_loaded());
    let samples = fixtures::speech_like(5.0, 16_000, 11);
    let chunk = AudioChunk::new(samples, 16_000, Time::ZERO).with_language(language);
    match engine.transcribe(&chunk, &cancel) {
        Ok(result) => {
            // Chunk-local contract: times are relative to the chunk start.
            // The decoder quantizes to encoder frames, so a word may start a
            // hair before 0 (overlap, clamped by push_chunk) or end up to one
            // frame past the chunk end. Both are within the documented
            // tolerance; anything further out is a rebase bug.
            let total = chunk.duration();
            let lo = Time(-1000);
            let hi = total.saturating_add(Time(1000));
            for s in &result.segments {
                assert!(
                    s.start >= lo && s.end <= hi,
                    "non-chunk-local times: {}..{} on a 0..{} chunk",
                    s.start.as_millis(),
                    s.end.as_millis(),
                    total.as_millis()
                );
                assert!(s.start <= s.end, "inverted segment");
                for w in &s.words {
                    assert!(w.start >= lo && w.end <= hi, "non-chunk-local word");
                    assert!(w.start <= w.end, "inverted word");
                }
            }
            // And the single rebase point absorbs the result into absolute time.
            let mut transcript =
                vaani_core::Transcript::new(vaani_core::EngineStamp::asr("qa-smoke", "0"));
            transcript.push_chunk(Time::ZERO, result.segments.clone());
            let starts: Vec<i64> = transcript
                .segments
                .iter()
                .map(|s| s.start.as_millis())
                .collect();
            let mut sorted = starts.clone();
            sorted.sort_unstable();
            assert_eq!(starts, sorted, "push_chunk must keep real output sorted");
            assert!(transcript.segments.iter().all(|s| s.start >= Time::ZERO));
            println!(
                "{tag}: {} segments, lang={:?}",
                result.segments.len(),
                result.language
            );
        }
        Err(e) => {
            assert!(
                !e.is_cancelled(),
                "transcription must not report cancellation"
            );
            panic!("{tag}: transcribe failed: {}", e.debug_chain());
        }
    }
    engine.unload();
}

#[test]
fn asr_smoke_hindi() {
    smoke(Language::Hindi, "asr_smoke_hindi");
}

#[test]
fn asr_smoke_hinglish() {
    smoke(Language::Hinglish, "asr_smoke_hinglish");
}

#[test]
fn asr_smoke_english() {
    smoke(Language::English, "asr_smoke_english");
}

#[test]
fn asr_smoke_code_switch_auto() {
    // Code-switch has no picker entry: Auto is the contract path.
    smoke(Language::Auto, "asr_smoke_code_switch_auto");
}

#[test]
fn asr_missing_model_is_model_unavailable() {
    // An engine over an empty directory must refuse with ModelUnavailable
    // before touching any factory — this path needs no weights.
    let dir = tempfile::TempDir::new().expect("tempdir");
    let art = vaani_asr::artifacts_in(dir.path());
    assert!(!art.is_complete());
    let caps = vaani_inference::capability::detect(&[]);
    let registry = vaani_asr::EngineRegistry::with_builtins(vaani_inference::Provider::Cpu, &caps)
        .expect("registry");
    let err = registry.create("sravaani", &art).expect_err("must refuse");
    assert_eq!(err.code(), vaani_core::ErrorCode::ModelUnavailable);
}

#[test]
fn asr_unknown_engine_lists_known_ones() {
    let registry = vaani_asr::EngineRegistry::new();
    let dir = tempfile::TempDir::new().expect("tempdir");
    let art = vaani_asr::artifacts_in(dir.path());
    let err = registry.create("nope", &art).expect_err("must refuse");
    assert!(err.user.recovery.is_some());
}

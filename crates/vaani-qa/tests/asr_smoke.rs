//! ASR smoke tests across the language hints, plus the model-absent paths.
//!
//! # What is real and what is skipped
//!
//! The four language tests need the SraVaani weights *and* the ONNX Runtime.
//! Neither is in a fresh checkout, so on such a machine they **skip loudly**,
//! through `vaani_qa::skip::require`, which can only return `Some` for one of
//! four closed [`Requirement`](vaani_qa::skip::Requirement) values and prints the
//! reason and the remedy.
//!
//! What they must *not* do — and what the first version of this file did — is
//! skip because the engine "could not be constructed". That string admits it
//! might not be a missing runtime at all: a corrupt `encoder.onnx`, a
//! mismatched ONNX build, a tokenizer that will not load. All of those produced
//! the same green tick as "we do not have the weights", which is the exact
//! failure mode a smoke test exists to catch. Here, present-but-broken is a
//! `panic!` with the error's chain, via
//! [`Environment::engine_problem`](vaani_qa::skip::Environment::engine_problem).
//!
//! The live decode is in `tests/asr_live.rs`, which is opt-in and says so.

use vaani_asr::engine::{AsrEngine, AudioChunk};
use vaani_asr::{EngineArtifacts, EngineRegistry};
use vaani_core::{CancelToken, Language, Time};
use vaani_inference::Provider;
use vaani_qa::conformance::check_errors_are_actionable;
use vaani_qa::fixtures;
use vaani_qa::skip::{self, Environment, Requirement};

/// The artefacts, or `None` with a printed reason.
fn artifacts(test: &str) -> Option<EngineArtifacts> {
    if skip::require(test, Requirement::ModelWeights).is_some() {
        return None;
    }
    if skip::require(test, Requirement::OnnxRuntime).is_some() {
        return None;
    }
    let dir = Environment::get().weights_dir().to_path_buf();
    Some(vaani_asr::artifacts_in(&dir))
}

/// Open the real engine, or **fail**.
///
/// There is no skip on this path. Once the weights and the runtime are both
/// present, an engine that will not open is a broken installation or a broken
/// release, and the test says so with the full error chain.
fn open_engine(test: &str, art: &EngineArtifacts) -> Box<dyn AsrEngine> {
    let env = Environment::get();
    let caps = vaani_inference::capability::detect(&[]);
    let registry = match EngineRegistry::with_builtins(Provider::Cpu, &caps) {
        Ok(r) => r,
        Err(e) => panic!("{test}: {}", env.engine_problem(test, &e)),
    };
    match registry.create("sravaani", art) {
        Ok(engine) => engine,
        Err(e) => panic!("{test}: {}", env.engine_problem(test, &e)),
    }
}

/// Transcribe one synthesised chunk and pin the chunk-local contract on it.
fn smoke(language: Language, tag: &str) {
    let Some(art) = artifacts(tag) else {
        return;
    };
    let engine = open_engine(tag, &art);
    if let Err(e) = engine.load(&CancelToken::new()) {
        panic!(
            "{tag}: the engine is installed but will not load, which is a failure and not a \
             skip: {}",
            Environment::get().engine_problem(tag, &e)
        );
    }
    assert!(engine.is_loaded());
    assert!(engine.supports_word_timestamps());
    assert!(engine.supports_timestamps());
    assert!(!engine.supports_streaming(), "SraVaani 1.0 is a file model");

    let cancel = CancelToken::new();
    let samples = fixtures::voice(
        6 * fixtures::FIXTURE_SAMPLE_RATE as usize,
        fixtures::FIXTURE_SAMPLE_RATE,
        0x11,
        140.0,
        0.9,
    );
    let chunk =
        AudioChunk::new(samples, fixtures::FIXTURE_SAMPLE_RATE, Time::ZERO).with_language(language);
    let result = engine
        .transcribe(&chunk, &cancel)
        .unwrap_or_else(|e| panic!("{tag}: transcribe failed: {}", e.debug_chain()));

    // The chunk-local contract. The decoder quantises to encoder frames, so a
    // word may start a frame early (overlap, clamped by `push_chunk`) or end one
    // frame past the chunk; both are inside the documented tolerance. Anything
    // further out is a rebase bug, and a *relative* time here is the bug this
    // whole file exists to prevent.
    let total = chunk.duration();
    let lo = Time(-1000);
    let hi = total.saturating_add(Time(1000));
    for s in &result.segments {
        assert!(
            s.start >= lo && s.end <= hi,
            "{tag}: non-chunk-local segment {}..{} on a 0..{} chunk",
            s.start.as_millis(),
            s.end.as_millis(),
            total.as_millis()
        );
        assert!(s.start <= s.end, "{tag}: inverted segment");
        for w in &s.words {
            assert!(
                w.start >= lo && w.end <= hi,
                "{tag}: non-chunk-local word at {}..{}",
                w.start.as_millis(),
                w.end.as_millis()
            );
            assert!(w.start <= w.end, "{tag}: inverted word");
        }
    }

    // And the single rebase point absorbs it into absolute time.
    let mut transcript = vaani_core::Transcript::new(vaani_core::EngineStamp::asr("qa", "0"));
    transcript.push_chunk(Time::ZERO, result.segments.clone());
    let starts: Vec<i64> = transcript
        .segments
        .iter()
        .map(|s| s.start.as_millis())
        .collect();
    let mut sorted = starts.clone();
    sorted.sort_unstable();
    assert_eq!(
        starts, sorted,
        "{tag}: push_chunk must keep real output sorted"
    );
    assert!(
        transcript.segments.iter().all(|s| s.start >= Time::ZERO),
        "{tag}: a negative time survived the rebase"
    );
    // And re-running the same chunk gives the same words. The model is fp16 and
    // deterministic, so any drift here is a real defect rather than noise.
    let again = engine
        .transcribe(&chunk, &cancel)
        .unwrap_or_else(|e| panic!("{tag}: second transcribe failed: {}", e.debug_chain()));
    assert_eq!(
        result.segments.len(),
        again.segments.len(),
        "{tag}: the same audio produced a different number of segments on a second run"
    );
    println!(
        "{tag}: {} segments, {} decode steps over {} encoder frames",
        result.segments.len(),
        result.timings.decode_steps,
        result.timings.encoder_frames
    );
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
    // Code-switching has no picker entry: `Auto` is the contract path.
    smoke(Language::Auto, "asr_smoke_code_switch_auto");
}

// ---------------------------------------------------------------------------
// The paths that must work with nothing installed
// ---------------------------------------------------------------------------

#[test]
fn an_incomplete_installation_is_model_unavailable_before_any_factory_runs() {
    // Needs no weights and no runtime: the registry must refuse an empty
    // directory, and it must say "not downloaded" rather than "could not load",
    // because the download screen keys off exactly that distinction.
    let dir = tempfile::TempDir::new().expect("tempdir");
    let art = vaani_asr::artifacts_in(dir.path());
    assert!(!art.is_complete());
    let caps = vaani_inference::capability::detect(&[]);
    let registry = EngineRegistry::with_builtins(Provider::Cpu, &caps).expect("registry");
    let err = registry
        .create("sravaani", &art)
        .expect_err("an empty directory must be refused");
    assert_eq!(err.code(), vaani_core::ErrorCode::ModelUnavailable);
    // The message names what is absent in the cause, which is what a support
    // conversation needs.
    assert!(
        err.debug_chain().contains("encoder") || err.debug_chain().contains("tokenizer"),
        "the cause must name the missing artefacts: {}",
        err.debug_chain()
    );

    // **Tracked defect, `vaani-asr` src/engine.rs**, registered as
    // `errors.actionable[asr_model_not_downloaded]` in
    // `vaani_qa::conformance::all_known_failures`. The check below still runs and
    // still measures; its verdict is `Known` rather than `Pass` because the
    // recovery is genuinely missing. `tests/known_failures.rs` fails the day it
    // is added.
    let report = check_errors_are_actionable(&[("asr_model_not_downloaded", &err)]);
    assert_eq!(
        report.known(),
        1,
        "the missing recovery should be a tracked known failure, not a pass and not a \
         unexplained failure: {}",
        report.render()
    );
    assert!(
        report.outcomes[0].detail.contains("still failing"),
        "the tracked entry now reads {:?}; the defect looks fixed, so delete it from \
         all_known_failures() and let this check pass",
        report.outcomes[0].detail
    );
    println!("{}", report.render());
}

#[test]
fn an_unknown_engine_id_names_the_ones_that_exist() {
    // The message is the whole feature: a typo in a settings file has to produce
    // a list the user can act on. Registered with the builtins on purpose — the
    // first version of this file used `EngineRegistry::new()`, which has no
    // engines at all, so the list it was checking was always empty and the
    // assertion could not fail.
    let caps = vaani_inference::capability::detect(&[]);
    let registry = EngineRegistry::with_builtins(Provider::Cpu, &caps).expect("registry");
    assert!(
        registry.ids().contains(&"sravaani"),
        "the built-in registry must offer sravaani, got {:?}",
        registry.ids()
    );
    let dir = tempfile::TempDir::new().expect("tempdir");
    let art = vaani_asr::artifacts_in(dir.path());
    let err = registry.create("nope", &art).expect_err("must refuse");
    assert!(err.user.recovery.is_some());
    let chain = err.debug_chain();
    assert!(
        chain.contains("sravaani"),
        "the error must name the engine that does exist, got: {chain}"
    );
}

#[test]
fn a_missing_artefacts_path_leaks_nothing_to_the_ui() {
    // `Display` is what the UI renders and what ends up in a bug report, so a
    // filesystem path in it is both a privacy problem and an unactionable
    // message.
    let artifacts = EngineArtifacts {
        frontend_dir: PathBuf::from("definitely/not/here"),
        encoder: PathBuf::from("definitely/not/here/encoder.onnx"),
        decoder_joint: PathBuf::from("definitely/not/here/dj.onnx"),
        tokenizer: PathBuf::from("definitely/not/here/tok.model"),
    };
    let e = SraVaaniEngine::new(&artifacts).expect_err("no artefacts, no engine");
    assert_eq!(
        e.code(),
        vaani_core::ErrorCode::ModelUnavailable,
        "preproc.json is read first, so the honest code is \"not downloaded\""
    );
    assert!(e.user.recovery.is_some());
    let shown = e.to_string();
    for leak in ["definitely", "not/here", "\\", "/"] {
        assert!(
            !shown.contains(leak),
            "the user-facing message leaked {leak:?}: {shown}"
        );
    }
    assert!(
        e.debug_chain().contains("definitely"),
        "the developer log must keep the path"
    );
}

#[test]
fn the_environment_summary_names_what_this_machine_cannot_run() {
    println!("{}", skip::environment_summary());
    let env = Environment::get();
    for req in skip::ALL_REQUIREMENTS {
        if !env.satisfies(req) {
            let reason = env.reason_unmet(req);
            assert!(!reason.is_empty());
            println!("  cannot run the {} gate: {reason}", req.slug());
        }
    }
}

use std::path::PathBuf;

use vaani_asr::SraVaaniEngine;

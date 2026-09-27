//! The conformance gate: every real stage, every guarantee.
//!
//! One test per stage, so a failure names the implementation. Each is one call
//! to [`assert_conforms`], which is the whole point: the checks are in the
//! harness, not here, so a new stage gets the entire contract for the price of
//! one line.
//!
//! Nine implementations are covered: three engines (`vaani_media`'s decoder,
//! `vaani_vad`'s `EnergyVad`, `vaani_diarization`'s `HeuristicDiarizer`) and all
//! six `vaani_app` `PipelineStage`s.

use std::sync::Arc;

use vaani_app::fakes::{FakeAsr, FixedAsrProvider};
use vaani_app::stages::{AsrStage, DiarStage, ExportStage, MediaStage, StoreStage, VadStage};
use vaani_core::JobPhase;
use vaani_export::{ExportFormat, ExportOptions, ExportTarget};
use vaani_qa::conformance::{
    AppStage, ConformanceStage, EnergyVadStage, HeuristicDiarStage, MediaStage as QaMediaStage,
    Upstream, assert_conforms, check_export_stage, check_progress_is_clamped, print_report,
    sample_transcript,
};
use vaani_storage::Store;
use vaani_vad::ChunkPolicy;

/// The ASR stage under test uses a deterministic engine, because the real one
/// needs 2.6 GB of weights and this gate must be runnable everywhere. The stage
/// is the code under test; the engine is a fixture.
fn asr_stage() -> AsrStage {
    AsrStage::new(
        Box::new(FakeAsr::new()),
        vaani_core::Language::Auto,
        ChunkPolicy::default(),
    )
}

fn store_stage() -> StoreStage {
    let dir = tempfile::TempDir::new().expect("tempdir");
    // The TempDir is leaked so the database outlives the stage: `StoreStage`
    // holds an open connection and a `JobRecord` pointing at a path.
    let dir = Box::leak(Box::new(dir));
    let store = Store::open(dir.path().join("conformance.db")).expect("store");
    StoreStage::new(
        store,
        "conformance-job".to_owned(),
        dir.path().join("in.wav"),
    )
    .with_engine("fake-asr", "cpu", "auto")
}

fn export_stage() -> ExportStage {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let dir = Box::leak(Box::new(dir));
    // `overwrite` because a conformance run executes the stage several times
    // over the same audio, and the export contract deliberately refuses to
    // clobber. Without it the second run fails and the determinism check reads
    // a deliberate policy as a bug.
    let options = ExportOptions {
        overwrite: true,
        ..ExportOptions::default().with_source_name("meeting.wav")
    };
    ExportStage::new(
        ExportFormat::Srt,
        ExportTarget::Directory(dir.path().join("out")),
        options,
    )
}

// ---------------------------------------------------------------------------
// The engines
// ---------------------------------------------------------------------------

#[test]
fn vaani_media_conforms() {
    assert_conforms(&QaMediaStage);
}

#[test]
fn energy_vad_conforms() {
    assert_conforms(&EnergyVadStage::new());
}

#[test]
fn heuristic_diarizer_conforms() {
    assert_conforms(&HeuristicDiarStage::new());
}

// ---------------------------------------------------------------------------
// The vaani_app stages
// ---------------------------------------------------------------------------

#[test]
fn app_media_stage_conforms() {
    assert_conforms(
        &AppStage::new(Box::new(MediaStage::new()), Upstream::Nothing)
            .with_work_audio(vaani_qa::conformance::long_work_audio),
    );
}

#[test]
fn app_vad_stage_conforms() {
    assert_conforms(&AppStage::new(
        Box::new(VadStage::new(Arc::new(vaani_vad::EnergyVad::new()))),
        Upstream::Audio,
    ));
}

#[test]
fn app_asr_stage_conforms() {
    assert_conforms(&AppStage::new(
        Box::new(asr_stage()),
        Upstream::AudioAndSpeech,
    ));
}

#[test]
fn app_diarization_stage_conforms() {
    // The *production* engine, not `FakeDiar`. `FakeDiar` hard-codes
    // `SpeakerSpan { start: Time::ZERO }` and never looks at the buffer's sample
    // rate, so a conformance run against it would be measuring the fake rather
    // than the stage. `FakeDiar` is the right tool for the orchestration tests
    // in `pipeline_integration.rs`, where the buffer starts at zero and the
    // point is the pipeline, not the contract.
    //
    // Note what this choice means: `DiarStage` itself performs no sample-rate
    // validation of its own, unlike `VadStage` and `AsrStage`. It relies on the
    // engine to check. That is defence in depth in the wrong order — a future
    // engine that resampled silently would not be caught at the stage boundary.
    // Tracked in OPEN ISSUES.
    assert_conforms(&AppStage::new(
        Box::new(DiarStage::new(Arc::new(
            vaani_diarization::HeuristicDiarizer::new(),
        ))),
        Upstream::AudioSpeechTranscript,
    ));
}

#[test]
fn app_store_stage_conforms() {
    assert_conforms(&AppStage::new(
        Box::new(store_stage()),
        Upstream::AudioSpeechTranscript,
    ));
}

#[test]
fn app_export_stage_conforms() {
    assert_conforms(&AppStage::new(
        Box::new(export_stage()),
        Upstream::TranscriptOnly,
    ));
}

// ---------------------------------------------------------------------------
// The contract guarantees that are not about a single stage
// ---------------------------------------------------------------------------

#[test]
fn progress_cannot_be_reported_out_of_range() {
    // A stage cannot report 1.5 or NaN, whatever it is asked to. This is
    // checked on `StageContext::report` because `ProgressKind::fraction`
    // clamps, so no stage's own output can reveal the raw value.
    let report = check_progress_is_clamped();
    print_report(&report);
    assert!(report.ok(), "{}", report.render());
}

#[test]
fn export_conforms() {
    let report = check_export_stage(&sample_transcript());
    print_report(&report);
    assert!(report.ok(), "{}", report.render());
}

#[test]
fn the_diarization_engine_reports_raw_fractions_it_can_actually_produce() {
    // `HeuristicDiarizer::diarize` takes a bare `f64`, so its progress is
    // asserted here rather than through `Progress`, which has no bare-fraction
    // form. The claim is narrow and falsifiable: every fraction it hands the
    // sink is finite and inside [0, 1].
    let engine = vaani_diarization::HeuristicDiarizer::new();
    use vaani_core::pipeline::DiarizationEngine;
    engine.load(&vaani_core::CancelToken::new()).expect("load");
    let audio = HeuristicDiarStage::new().work_audio();
    let seen = std::sync::Mutex::new(Vec::new());
    let sink = {
        let seen = &seen;
        move |f: f64| {
            if let Ok(mut s) = seen.lock() {
                s.push(f);
            }
        }
    };
    let result = engine.diarize(&audio, &vaani_core::CancelToken::new(), &sink);
    assert!(result.is_ok(), "diarize failed: {:?}", result.err());
    let fractions = seen.into_inner().expect("progress lock");
    for (i, f) in fractions.iter().enumerate() {
        assert!(
            f.is_finite() && (0.0..=1.0).contains(f),
            "progress fraction {i} was {f}: the engine must never report a value a progress \
             bar cannot draw"
        );
    }
    for w in fractions.windows(2) {
        assert!(w[1] >= w[0], "progress went backwards: {fractions:?}");
    }
    println!(
        "diarizer reported {} progress values, all in [0,1] and non-decreasing",
        fractions.len()
    );
    // And the ASR fake must exist for the stage tests; assert the provider shape
    // so a change to `vaani_app::fakes` is caught here rather than as a mystery.
    use vaani_app::orchestrator::AsrProvider;
    let provider = FixedAsrProvider::new(FakeAsr::new());
    let engine = provider
        .create(
            &vaani_app::AppConfig::default(),
            &vaani_core::CancelToken::new(),
        )
        .expect("fake asr provider");
    assert_eq!(engine.id(), "fake-asr");
    // The phases the orchestrator walks must include the two this crate's
    // integration tests inject failures into, or those tests are asserting
    // against a phase the pipeline does not have.
    for phase in [
        JobPhase::ExtractingAudio,
        JobPhase::Vad,
        JobPhase::Asr,
        JobPhase::Diarization,
        JobPhase::Exporting,
    ] {
        assert!(JobPhase::parse(phase.as_str()).is_some(), "{phase}");
    }
}

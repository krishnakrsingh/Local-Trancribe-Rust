//! The known-failure registry, audited.
//!
//! A known failure is a contract violation this harness found in a crate it does
//! not own. It is neither fixed here nor deleted from the check: the check still
//! runs, still measures, and its verdict is [`Verdict::Known`] rather than
//! [`Verdict::Pass`]. That is only defensible if the list is policed, and that
//! is this file's job.
//!
//! Four properties, each of which fails if the registry is used as a place to
//! hide things:
//!
//! 1. **Every entry names an owning crate.** An entry that says only "known
//!    problem" is indistinguishable from an excuse.
//! 2. **Every entry still reproduces.** A tracked defect that has been fixed
//!    upstream must fail here, so the entry is deleted rather than left to grey
//!    out a check forever.
//! 3. **The list is short.** A conformance suite with more known failures than
//!    guarantees is a suite that has stopped measuring anything.
//! 4. **Nothing is quietly added.** Raising this number is a deliberate act with
//!    a commit message attached, not a side effect of a failing test.
//!
//! Every entry below is a real defect with a real consequence for a user. They
//! are reproduced verbatim in `tests/known_failure_repros.rs`.

use std::sync::Arc;

use vaani_app::stages::{AsrStage, DiarStage, MediaStage, VadStage};
use vaani_qa::conformance::{
    AppStage, ConformanceStage, EnergyVadStage, HeuristicDiarStage, MediaStage as QaMediaStage,
    Upstream, Verdict, all_known_failures, asr_stage_known_failures, core_known_failures,
    run_conformance,
};
use vaani_vad::ChunkPolicy;

/// The ceiling. Five entries across the whole harness, for a codebase this size,
/// is already too many; the point of the number is that it has to be argued for.
const MAX_ENTRIES: usize = 6;

#[test]
fn the_registry_is_short_and_every_entry_names_an_owner() {
    let mut entries = all_known_failures();
    entries.extend(asr_stage_known_failures());

    assert!(
        entries.len() <= MAX_ENTRIES,
        "the known-failure registry has {} entries (ceiling {MAX_ENTRIES}). Each one is a \
         contract this product does not honour; a suite with more known failures than \
         guarantees is not measuring anything:\n{}",
        entries.len(),
        entries
            .iter()
            .map(|k| format!("  {} -> {}", k.check, k.owned_by))
            .collect::<Vec<_>>()
            .join("\n")
    );

    for k in &entries {
        assert!(
            k.check.starts_with("progress.")
                || k.check.starts_with("errors.actionable[")
                || k.check.starts_with("times."),
            "{}: known failures are only allowed for the universal guarantees, never for a \
             stage's bespoke behaviour",
            k.check
        );
        assert!(
            k.owned_by.starts_with("vaani-"),
            "{}: a known failure must name the crate that owns it, and this crate owns none of \
             them; got {:?}",
            k.check,
            k.owned_by
        );
        assert!(
            k.why.len() > 80,
            "{}: the reason must say what is wrong and what a user sees, not just name a \
             check",
            k.check
        );
    }

    // Duplicate check names would mean one entry was shadowing another.
    let mut names: Vec<&str> = entries.iter().map(|k| k.check.as_str()).collect();
    names.sort_unstable();
    let count = names.len();
    names.dedup();
    assert_eq!(
        names.len(),
        count,
        "two known failures share a check name: {names:?}"
    );
}

#[test]
fn every_tracked_failure_still_reproduces() {
    // Each entry, driven directly. If one of these starts passing, the fix
    // landed and the entry is stale; this test fails and says to delete it.
    let mut still_failing = 0usize;

    // --- 1. vaani-core: a NaN progress fraction reaches the UI ---------------
    {
        let report = vaani_qa::conformance::check_progress_is_clamped();
        let nan = report
            .outcomes
            .iter()
            .find(|o| o.name == "progress.clamped[nan]")
            .expect("the NaN case must be measured");
        assert_eq!(
            nan.verdict,
            Verdict::Known,
            "the NaN outcome should be a tracked known failure, not a pass and not a fail"
        );
        assert!(
            nan.detail.contains("still failing"),
            "the NaN case now reads {:?}; the defect looks fixed, so delete this entry from \
             core_known_failures() and let the check pass",
            nan.detail
        );
        assert!(
            nan.detail.contains("Some(NaN)"),
            "the repro should be re-run and described precisely; got: {}",
            nan.detail
        );
        still_failing += 1;
    }

    // --- 2 and 3. the two engines report a caller error as Internal ---------
    {
        let vad = EnergyVadStage::new();
        let samples = vad.error_samples();
        let wrong = samples
            .iter()
            .find(|(label, _)| label == "energy_vad_wrong_sample_rate")
            .expect("the VAD must still offer a wrong-rate sample");
        assert_eq!(
            wrong.1.code(),
            vaani_core::ErrorCode::Internal,
            "the VAD's wrong-rate error is no longer Internal; if it has been reclassified, \
             delete the entry and let the actionable-error check pass"
        );
        assert!(
            wrong.1.user.recovery.is_none(),
            "the VAD's wrong-rate error now carries a recovery; delete the entry"
        );
        still_failing += 1;
    }
    {
        let diar = HeuristicDiarStage::new();
        let samples = diar.error_samples();
        for label in ["diar_wrong_sample_rate", "diar_stereo_not_downed"] {
            let sample = samples
                .iter()
                .find(|(l, _)| l == label)
                .unwrap_or_else(|| panic!("the diarizer must still offer a {label} sample"));
            assert_eq!(
                sample.1.code(),
                vaani_core::ErrorCode::Internal,
                "{label} is no longer Internal; delete the entry"
            );
            assert!(
                sample.1.user.recovery.is_none(),
                "{label} now carries a recovery; delete the entry"
            );
            still_failing += 1;
        }
    }

    // --- 4. the ASR stage's coordinate-space violation: FIXED ----------------
    //
    // This block used to re-drive `times.no_time_precedes_the_origin` and assert
    // it was still failing. It no longer is, so the entry has been deleted from
    // `asr_stage_known_failures()` and the check passes outright. Rather than
    // silently drop the coverage, the check is now asserted as a **Pass**: the
    // ASR stage is run on the same +5-minute work audio that used to expose the
    // defect, and it is required to honour the origin. That is a stronger
    // statement than the old one, because a regression now fails the gate
    // instead of being absorbed into a KnownFailure.
    {
        let stage = AppStage::new(
            Box::new(AsrStage::new(
                Box::new(vaani_app::fakes::FakeAsr::new()),
                vaani_core::Language::Auto,
                ChunkPolicy::default(),
            )),
            Upstream::AudioAndSpeech,
        );
        let report = run_conformance(&stage);
        let outcome = report
            .outcomes
            .iter()
            .find(|o| o.name == "times.no_time_precedes_the_origin")
            .expect("the coordinate-space check must have run");
        assert_eq!(
            outcome.verdict,
            Verdict::Pass,
            "the ASR stage has regressed on a non-zero buffer origin. It builds a MediaWindow \
             from the decoded buffer and must plan and clamp inside it; see \
             docs/architecture/TIMESTAMP_CONTRACT.md and the entry removed from \
             asr_stage_known_failures(). Detail: {}",
            outcome.detail
        );
        assert!(
            outcome.detail.contains("0 of"),
            "the check reports pairs before the origin: {}",
            outcome.detail
        );
        assert!(
            asr_stage_known_failures().is_empty(),
            "a new known failure was registered for the ASR stage without a commit that says \
             why: {:?}",
            asr_stage_known_failures()
        );
    }

    // --- 5. the ASR engine's missing-model error carries no recovery --------
    {
        let caps = vaani_inference::capability::detect(&[]);
        let registry =
            vaani_asr::EngineRegistry::with_builtins(vaani_inference::Provider::Cpu, &caps)
                .expect("registry");
        let dir = tempfile::TempDir::new().expect("tempdir");
        let art = vaani_asr::artifacts_in(dir.path());
        let err = registry
            .create("sravaani", &art)
            .expect_err("an absent installation must be refused");
        assert_eq!(
            err.code(),
            vaani_core::ErrorCode::ModelUnavailable,
            "the code is wrong; the entry is about the missing recovery, not the code"
        );
        assert!(
            err.user.recovery.is_none(),
            "the missing-model error now carries a recovery; delete the entry from \
             all_known_failures() and let the actionable-error check pass"
        );
        let report = vaani_qa::conformance::check_errors_are_actionable(&[(
            "asr_model_not_downloaded",
            &err,
        )]);
        assert_eq!(report.known(), 1, "{}", report.render());
        assert!(
            report.outcomes[0].detail.contains("still failing"),
            "the entry now reads {:?}; delete it",
            report.outcomes[0].detail
        );
        still_failing += 1;
    }

    assert_eq!(
        still_failing,
        all_known_failures().len(),
        "every registered entry must be re-driven by this test; one of them is not"
    );
}

#[test]
fn a_stage_with_no_registered_failures_reports_none() {
    // The other half of the contract: if every stage declared known failures,
    // the mechanism would be worthless. The media stage and the VAD engine
    // between them are clean apart from the two shared error-construction
    // entries, and the store and export stages are clean outright.
    let clean = run_conformance(&AppStage::new(
        Box::new(vaani_app::stages::StoreStage::new(
            vaani_storage::Store::open_in_memory().expect("store"),
            "known-failure-audit".to_owned(),
            "in.wav".into(),
        )),
        Upstream::AudioSpeechTranscript,
    ));
    assert_eq!(
        clean.known(),
        0,
        "the store stage has no registered defects, so nothing may be Known: {}",
        clean.render()
    );
    assert!(clean.ok(), "{}", clean.render());

    // And the four that are clean must be clean for the right reason: the
    // checks actually ran.
    assert!(
        clean.passed() >= 8,
        "the store stage only ran {} checks; a suite that skips is not a suite",
        clean.passed()
    );
}

#[test]
fn every_stage_the_gate_covers_is_still_conformant() {
    // A belt-and-braces re-run of the whole matrix in one place, so a change to
    // any adapter shows up as a single named failure rather than eleven.
    let stages: Vec<(String, Box<dyn ConformanceStage>)> = vec![
        ("vaani-media".into(), Box::new(QaMediaStage)),
        ("energy-vad".into(), Box::new(EnergyVadStage::new())),
        ("heuristic-diar".into(), Box::new(HeuristicDiarStage::new())),
        (
            "app:media".into(),
            Box::new(
                AppStage::new(Box::new(MediaStage::new()), Upstream::Nothing)
                    .with_work_audio(vaani_qa::conformance::long_work_audio),
            ),
        ),
        (
            "app:vad".into(),
            Box::new(AppStage::new(
                Box::new(VadStage::new(Arc::new(vaani_vad::EnergyVad::new()))),
                Upstream::Audio,
            )),
        ),
        (
            "app:asr".into(),
            Box::new(AppStage::new(
                Box::new(AsrStage::new(
                    Box::new(vaani_app::fakes::FakeAsr::new()),
                    vaani_core::Language::Auto,
                    ChunkPolicy::default(),
                )),
                Upstream::AudioAndSpeech,
            )),
        ),
        (
            "app:diarization".into(),
            Box::new(AppStage::new(
                Box::new(DiarStage::new(Arc::new(
                    vaani_diarization::HeuristicDiarizer::new(),
                ))),
                Upstream::AudioSpeechTranscript,
            )),
        ),
    ];

    let mut total_checks = 0usize;
    for (name, stage) in &stages {
        let report = run_conformance(stage.as_ref());
        total_checks += report.outcomes.len();
        assert!(
            report.ok(),
            "{name} does not conform: {}\n{}",
            report.failures().join(", "),
            report.render()
        );
        println!(
            "{name}: {} passed, {} known, {} not applicable",
            report.passed(),
            report.known(),
            report.skipped.len()
        );
    }
    // The matrix must actually be checking something. This is the anti-regression
    // on the harness itself: if a refactor silently emptied `run_conformance`,
    // every stage would report zero and still be "ok".
    assert!(
        total_checks >= 100,
        "the whole matrix ran only {total_checks} checks across {} stages; the harness has \
         shrunk",
        stages.len()
    );
}

#[test]
fn core_known_failures_are_a_subset_of_the_full_registry() {
    let full = all_known_failures();
    for k in core_known_failures() {
        assert!(
            full.contains(&k),
            "core_known_failures() has {:?}, which all_known_failures() does not",
            k.check
        );
    }
}

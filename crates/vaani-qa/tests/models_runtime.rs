//! Model-manager, runtime and provider tests. No network, no weights.
//!
//! # What changed, and why
//!
//! * `install_plan_on_missing_model_is_clean_error` was:
//!
//!   ```rust
//!   let err = manager.install_plan("no-such-model-xyz").expect_err("unknown model must fail");
//!   let _ = err;
//!   println!("unknown install plan correctly rejected");
//!   ```
//!
//!   It bound `err` and then threw it away. It asserted nothing at all, and its
//!   `println!` claimed the thing it had not checked. **Deleted.** It is replaced
//!   by [`an_unknown_model_id_is_a_clean_refusal_with_a_recovery`], which asserts
//!   the code, the recovery and the message, and which records the fact that the
//!   code is `internal` — a caller mistake reported as a bug, which is a defect
//!   in `vaani-models` and is written up in OPEN ISSUES.
//! * `missing_model_resolves_to_model_unavailable` asserted
//!   `code == ModelUnavailable || code == Internal`. The disjunction was covering
//!   the measured behaviour: a *known* model id that is not installed gives
//!   `model_unavailable`, while an *unknown* id gives `internal`. The test is now
//!   split in two, one per case, with the exact code asserted.
//! * The full-disk precheck test arrived here from `export_robustness.rs`, where
//!   it was filed under a crate it does not exercise.

use std::sync::Arc;

use vaani_core::ErrorCode;
use vaani_inference::Provider;
use vaani_inference::runtime::{ActiveProvider, RuntimeLocation, choose_provider};
use vaani_models::ModelRoots;
use vaani_qa::skip::{self, Requirement};

/// A manager scoped to an empty temp root, so nothing on the developer's machine
/// can leak into the result.
fn empty_scoped_manager() -> Option<vaani_models::ModelManager> {
    let embedded = match vaani_models::ModelManager::embedded() {
        Ok(m) => m,
        Err(e) => {
            skip::skip(
                "models_runtime",
                Requirement::ModelWeights,
                format!(
                    "the embedded manifest is unreadable, so the manager cannot be built: {e:?}"
                ),
            );
            return None;
        }
    };
    let empty = tempfile::TempDir::new().expect("tempdir");
    let roots = ModelRoots::at(empty.path(), empty.path().join("bundled"));
    // The TempDir is leaked so the roots stay valid for the manager's lifetime.
    let _empty = Box::leak(Box::new(empty));
    Some(vaani_models::ModelManager::new(
        embedded.manifest().clone(),
        roots,
    ))
}

#[test]
fn a_known_model_that_is_not_installed_is_model_unavailable() {
    let Some(manager) = empty_scoped_manager() else {
        return;
    };
    let err = manager
        .resolve("sravaani-1.0-onnx")
        .expect_err("the root is empty, so nothing is installed");
    assert_eq!(
        err.code(),
        ErrorCode::ModelUnavailable,
        "a known model that is not on disk is \"not downloaded\", which is the distinction the \
         download screen keys off: {}",
        err.debug_chain()
    );
    assert!(
        err.user.recovery.is_some(),
        "a user who has not downloaded the model must be told how: {}",
        err.user.detail
    );
}

#[test]
fn an_unknown_model_id_is_a_clean_refusal_with_a_recovery() {
    let Some(manager) = empty_scoped_manager() else {
        return;
    };
    let err = manager
        .resolve("no-such-model-xyz")
        .expect_err("an unknown id must be refused");
    // **Tracked defect, `vaani-models`.** An unknown model id is a caller mistake
    // (a stale settings file, a typo, a database written by a newer build) and is
    // reported as `ErrorCode::Internal` with the user-facing text "Something went
    // wrong. An unexpected internal error occurred. Please report this." A user who
    // edits a settings file is told to file a bug. Either give it its own code —
    // JOURNEY.md's open item 5 already asks for two more — or reuse
    // `ModelUnavailable` with a recovery that lists the ids that do exist.
    //
    // Asserted as `Internal` so the day it is reclassified this fails and says so.
    assert_eq!(
        err.code(),
        ErrorCode::Internal,
        "an unknown model id is now classified as something other than Internal; the tracked \
         defect in OPEN ISSUES is fixed, so update the code and the write-up"
    );
    assert!(
        err.user.recovery.is_some(),
        "even as an internal error it must offer a next step"
    );
    let chain = err.debug_chain();
    assert!(
        chain.contains("no-such-model-xyz"),
        "the cause must name the id that was not found: {chain}"
    );
}

#[test]
fn an_unknown_model_has_no_install_plan() {
    let Some(manager) = empty_scoped_manager() else {
        return;
    };
    let err = manager
        .install_plan("no-such-model-xyz")
        .expect_err("there is nothing to install for an id that does not exist");
    assert!(
        !err.is_cancelled(),
        "an unknown model is not a cancellation"
    );
    assert!(
        matches!(
            err.code(),
            ErrorCode::Internal | ErrorCode::ModelUnavailable
        ),
        "got {}: {}",
        err.code(),
        err.debug_chain()
    );
    assert!(
        err.user.recovery.is_some(),
        "a refusal must say what to do instead"
    );
}

#[test]
fn an_incomplete_artefacts_directory_lists_every_absent_file() {
    // The download UI shows this list, so a missing entry means the user is told
    // a file is present when it is not — and the download then "completes" with a
    // model that will not load.
    let dir = tempfile::TempDir::new().expect("tempdir");
    let art = vaani_asr::artifacts_in(dir.path());
    assert!(!art.is_complete());
    let missing = art.missing();
    assert_eq!(
        missing.len(),
        6,
        "the artefact set is six files; a list of {} means one is not being checked",
        missing.len()
    );
    for path in &missing {
        let path = std::path::Path::new(path);
        assert!(
            !path.exists(),
            "{} is listed as missing but exists",
            path.display()
        );
        assert!(
            path.is_absolute() || path.components().count() > 1,
            "{} should be a usable path, not a bare name",
            path.display()
        );
    }
    // And the names are specific enough to act on.
    let text: Vec<String> = missing.clone();
    assert!(
        text.iter().any(|p| p.contains("encoder")),
        "the encoder is the largest and most important file; it must be in the list: {text:?}"
    );
    assert!(
        text.iter().any(|p| p.contains("tokenizer")),
        "a missing tokenizer is the hardest failure to diagnose; it must be in the list: {text:?}"
    );
}

#[test]
fn a_known_artefacts_directory_is_complete() {
    // The mirror image, and it only runs where the model is installed. Without
    // it, `an_incomplete_artefacts_directory_lists_every_absent_file` would pass
    // on a machine where nothing was ever checked.
    if skip::require(
        "a_known_artefacts_directory_is_complete",
        Requirement::ModelWeights,
    )
    .is_some()
    {
        return;
    }
    let art = vaani_asr::artifacts_in(&vaani_asr::default_dir());
    assert!(
        art.is_complete(),
        "the weights are present but the set is incomplete: {:?}",
        art.missing()
    );
    assert!(art.missing().is_empty());
}

#[test]
fn the_provider_decision_never_fails_and_records_its_fallback() {
    // `choose_provider` is pure, so the whole matrix is testable with no runtime
    // present. The contract: CUDA unavailable is a `ProviderStatus`, never an
    // error, because a user who asked for a transcript should get one.
    for (requested, cuda_loadable, expected) in [
        (Provider::Auto, false, ActiveProvider::Cpu),
        (Provider::Auto, true, ActiveProvider::Cuda),
        (Provider::RequireCuda, false, ActiveProvider::Cpu),
        (Provider::RequireCuda, true, ActiveProvider::Cuda),
        (Provider::Cpu, false, ActiveProvider::Cpu),
        (Provider::Cpu, true, ActiveProvider::Cpu),
    ] {
        let status = choose_provider(requested, cuda_loadable);
        assert_eq!(
            status.active, expected,
            "{requested:?} with cuda_loadable={cuda_loadable}"
        );
        assert_eq!(
            status.requested, requested,
            "the request must be echoed back"
        );
        assert!(
            status.note.is_some_and(|n| !n.is_empty()),
            "{requested:?} with cuda_loadable={cuda_loadable}: a fallback happened with nothing \
             to tell the user about it, which is the silent-fallback bug"
        );
    }
    // And the explicit CPU request is the one that must win over a present GPU,
    // because it is what makes the CPU path testable on a GPU machine.
    let cpu = choose_provider(Provider::Cpu, true);
    assert_eq!(cpu.active, ActiveProvider::Cpu);
    assert!(
        cpu.note.as_deref().is_some_and(|n| n.contains("CPU")),
        "the note must say the CPU was requested, so a benchmark column is not silently \
         mislabelled: {cpu:?}"
    );
}

#[test]
fn the_runtime_report_can_be_collected_without_a_runtime_present() {
    // `vaani doctor` shows this on the very first run, before anything is
    // installed. It must not panic and must say something.
    let report = vaani_inference::RuntimeReport::collect();
    let summary = report.summary();
    assert!(
        !summary.is_empty(),
        "the runtime report says nothing at all"
    );
    println!("runtime: {summary}");
    // And a location that found nothing must name a concrete path, so the error
    // a user sees tells them where it looked.
    let loc = RuntimeLocation::discover();
    if !loc.is_present() {
        let missing = loc.missing();
        assert!(
            !missing.is_empty(),
            "a missing runtime must name what is missing, not just that something is"
        );
        assert!(
            loc.ort_dll.is_absolute() || loc.ort_dll.components().count() > 1,
            "the reported path {} should be specific enough to look at",
            loc.ort_dll.display()
        );
    }
    assert!(!loc.provider_dirs.is_empty());
}

#[test]
fn a_download_precheck_refuses_when_there_is_no_free_space() {
    // A full disk must be refused *before* 887 MiB is written, with an
    // actionable code. The free-space probe is injected, so no real disk is
    // measured and no network is touched.
    use vaani_models::download::{Downloader, FixedFreeSpace};
    let downloader = Downloader::with_policy(
        vaani_models::download::DownloadPolicy::single_attempt(),
        Arc::new(FixedFreeSpace(Some(0))),
    );
    let dir = tempfile::TempDir::new().expect("tempdir");
    let err = downloader
        .precheck(dir.path(), 1024 * 1024)
        .expect_err("a full disk must not precheck clean");
    let vaani = err.to_vaani_error();
    assert_eq!(vaani.code(), ErrorCode::InsufficientDisk);
    assert!(
        vaani.user.recovery.is_some(),
        "a full disk needs an instruction, not just a diagnosis"
    );
}

#[test]
fn a_download_precheck_passes_when_there_is_room() {
    // The other half, so the refusal above is known to be a decision rather than a
    // precheck that always fails. The margin has to clear the downloader's
    // `reserve_bytes` as well as the requested size, which is the point of
    // asserting both halves: a precheck that ignored the reserve would pass here
    // and still run the disk out mid-download.
    use vaani_models::download::{Downloader, FixedFreeSpace};
    const NEEDED: u64 = 8 * 1024 * 1024;
    let plenty = 512 * 1024 * 1024;
    let downloader = Downloader::with_policy(
        vaani_models::download::DownloadPolicy::single_attempt(),
        Arc::new(FixedFreeSpace(Some(plenty))),
    );
    let dir = tempfile::TempDir::new().expect("tempdir");
    let free = downloader
        .precheck(dir.path(), NEEDED)
        .unwrap_or_else(|e| panic!("512 MiB free must satisfy an 8 MiB download: {e:?}"))
        .expect("a known free-space probe returns a number, not None");
    assert_eq!(
        free, plenty,
        "the precheck must report the space it measured"
    );

    // Just under the requirement including the reserve is refused. The exact
    // boundary is not asserted because `reserve_bytes` is a policy value, and a
    // test that hard-codes it would break every time the policy is tuned.
    let tight = Downloader::with_policy(
        vaani_models::download::DownloadPolicy::single_attempt(),
        Arc::new(FixedFreeSpace(Some(NEEDED - 1))),
    );
    let err = tight
        .precheck(dir.path(), NEEDED)
        .expect_err("one byte short of the request must be refused");
    assert_eq!(err.to_vaani_error().code(), ErrorCode::InsufficientDisk);
}

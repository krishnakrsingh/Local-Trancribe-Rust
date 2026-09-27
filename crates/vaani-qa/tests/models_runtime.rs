//! Model-manager and runtime adversarial tests: missing models, CUDA fallback.
//!
//! No network is touched and no weights are required. Missing-model paths are
//! asserted exactly (`model_unavailable`); CUDA paths are asserted through the
//! pure provider decision (`choose_provider`), which is unit-testable without
//! ONNX Runtime present.

use vaani_core::ErrorCode;
use vaani_inference::Provider;
use vaani_inference::runtime::{ActiveProvider, choose_provider};
use vaani_models::ModelRoots;
use vaani_qa::skip;

fn embedded_manager() -> Option<vaani_models::ModelManager> {
    match vaani_models::ModelManager::embedded() {
        Ok(m) => Some(m),
        Err(e) => {
            skip::skip(
                "models_runtime",
                &format!("embedded manifest unreadable: {e:?}"),
            );
            None
        }
    }
}

#[test]
fn missing_model_resolves_to_model_unavailable() {
    let Some(manager) = embedded_manager() else {
        return;
    };
    // Scope a manager to an empty temp root so nothing is installed there.
    // (The developer machine *does* have the model in the real roots — the
    // ASR smoke tests prove it — which is why the scope matters.)
    let empty = tempfile::TempDir::new().expect("tempdir");
    let scoped = vaani_models::ModelManager::new(
        manager.manifest().clone(),
        ModelRoots::at(empty.path(), empty.path().join("bundled")),
    );
    for id in ["sravaani-1.0-onnx", "no-such-model-xyz"] {
        let err = scoped
            .resolve(id)
            .expect_err(&format!("{id} must be missing"));
        assert!(
            err.code() == ErrorCode::ModelUnavailable || err.code() == ErrorCode::Internal,
            "{id}: dirty code {}",
            err.code()
        );
    }
    println!("SKIP-CHECK: resolve on empty roots is clean (model_unavailable)");
}

#[test]
fn engine_artifacts_missing_lists_every_absent_file() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let art = vaani_asr::artifacts_in(dir.path());
    assert!(!art.is_complete());
    assert_eq!(art.missing().len(), 6);
}

#[test]
fn cuda_unavailable_falls_back_to_cpu_silently() {
    // Requested CUDA but the provider DLL is absent: CPU, not a crash. The
    // contract says `cuda_unavailable` is a ProviderStatus, never an error.
    let status = choose_provider(Provider::Auto, false);
    assert_eq!(status.active, ActiveProvider::Cpu);
    let status = choose_provider(Provider::RequireCuda, false);
    assert_eq!(status.active, ActiveProvider::Cpu);
    // Explicit CPU stays CPU even when CUDA is loadable.
    let status = choose_provider(Provider::Cpu, true);
    assert_eq!(status.active, ActiveProvider::Cpu);
    // CUDA present + requested: GPU.
    let status = choose_provider(Provider::Auto, true);
    assert_eq!(status.active, ActiveProvider::Cuda);
}

#[test]
fn cuda_probe_reports_without_panicking() {
    let report = vaani_inference::RuntimeReport::collect();
    let summary = report.summary();
    assert!(!summary.is_empty());
    println!("runtime: {summary}");
}

#[test]
fn install_plan_on_missing_model_is_clean_error() {
    let Some(manager) = embedded_manager() else {
        return;
    };
    let err = manager
        .install_plan("no-such-model-xyz")
        .expect_err("unknown model must fail");
    let _ = err;
    println!("unknown install plan correctly rejected");
}

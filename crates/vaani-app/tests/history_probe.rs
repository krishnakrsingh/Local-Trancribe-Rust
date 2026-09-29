//! One-off probe: load history from the real user database.
//!
//! Run explicitly, never in CI:
//! ```text
//! cargo test -p vaani-app --release --test history_probe -- --ignored --nocapture
//! ```

use std::sync::Arc;

#[test]
#[ignore = "reads the real user database; run explicitly"]
fn history_loads_from_the_real_database() {
    let path = dirs::data_dir()
        .expect("data dir")
        .join("VaaniDesk")
        .join("vaanidesk.db");
    if !path.is_file() {
        println!("SKIP: no real database at {}", path.display());
        return;
    }
    let store = vaani_storage::Store::open(&path).expect("open real db");
    let providers = vaani_app::orchestrator::EngineProviders::production(
        vaani_models::ModelManager::embedded().expect("manifest"),
    );
    let orchestrator = Arc::new(vaani_app::orchestrator::Orchestrator::with_providers(
        store, providers,
    ));
    let backend = vaani_app::backend::AppBackend::with_orchestrator(orchestrator).expect("backend");
    let jobs = vaani_ui::backend::JobController::jobs(&*backend);
    println!("real database holds {} listed jobs", jobs.len());
    assert!(
        !jobs.is_empty(),
        "the real database has jobs; the list must not be empty"
    );
    let with_transcript = jobs.iter().filter(|j| j.has_transcript).count();
    println!("{with_transcript} of {} have transcripts", jobs.len());
    for j in jobs.iter().take(3) {
        println!(
            "  {:?} {} transcript={} words={:?}",
            j.phase,
            j.input
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            j.has_transcript,
            j.word_count,
        );
    }
}

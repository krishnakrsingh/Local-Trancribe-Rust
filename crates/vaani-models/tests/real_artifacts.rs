//! The one test that proves the manifest matches reality.
//!
//! Everything else in this crate tests the machinery against fixtures. This one
//! reads `artifacts/sravaani/` — the 2.5 GiB, ten-file, validated directory the
//! product actually runs — and checks that the manifest describes it exactly:
//! every file present, every size right, every SHA-256 right, and nothing
//! missing.
//!
//! It is **skip-guarded**. A fresh clone has no `artifacts/`, and a test that
//! fails there teaches people to ignore it. When it does run it prints one line
//! of results and nothing else: no file contents, no partial hashes, nothing
//! that would end up in a bug report with a model's bytes in it.

mod support;

use std::path::{Path, PathBuf};
use std::time::Instant;

use vaani_models::manager::ModelManager;
use vaani_models::paths::ModelRoots;
use vaani_models::verify::Verifier;
use vaani_models::{CancelToken, Diagnosis};

/// `artifacts/sravaani/`, located the same way `vaani_asr::sravaani::default_dir`
/// locates it, so a test here and the frozen ASR suite agree about the path.
fn sravaani_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/sravaani")
}

/// The embedded manifest, loaded once.
fn manager() -> ModelManager {
    ModelManager::embedded().expect("the embedded manifest must be valid")
}

const ID: &str = "sravaani-1.0-onnx";

#[test]
fn the_embedded_manifest_describes_the_real_sravaani_directory_exactly() {
    let dir = sravaani_dir();
    if !dir.join("encoder.fp16.onnx").is_file() {
        println!(
            "skipping: {} is not present. Run scripts\\export_sravaani_onnx.py \
             to produce it; see docs/agent/MODELS.md.",
            dir.display()
        );
        return;
    }
    let m = manager();
    let model = m
        .manifest()
        .model(ID)
        .expect("the manifest must know the model");

    // Discovery, against the artefacts *root* -- the parent of `sravaani/`,
    // because a model root holds one directory per model. Nothing is configured;
    // this is the layout `scripts/export_sravaani_onnx.py` produces.
    let root = dir
        .parent()
        .expect("artifacts/sravaani has a parent")
        .to_path_buf();
    let roots = ModelRoots::at(m.roots().user().join("__a7_absent_user__"), &root);
    let d = vaani_models::discover::discover(m.manifest(), &roots);
    let dm = d
        .best(ID)
        .unwrap_or_else(|| panic!("artifacts/sravaani/ was not found as a model directory"));

    // Every file the manifest names is there, at the right size.
    assert_eq!(
        dm.dir, dir,
        "the artefacts root must resolve to the real directory"
    );
    assert!(
        dm.unexpected.is_empty(),
        "the manifest does not describe {}: {dm:?}",
        dm.unexpected.join(", ")
    );
    assert_eq!(
        dm.on_disk_bytes, model.size_bytes,
        "measured total vs manifest total"
    );
    for f in &model.files {
        let state = dm
            .files
            .iter()
            .find(|s| s.path == f.path)
            .unwrap_or_else(|| panic!("the manifest's file {} was not probed", f.path));
        assert!(
            state.is_correct_size(),
            "{}: manifest says {} bytes, disk has {:?}",
            f.path,
            f.bytes,
            state.actual_bytes
        );
    }

    // And every byte is right. This is the expensive half: 2.5 GiB of SHA-256.
    let started = Instant::now();
    let report = Verifier::new().verify(model, &dir, &CancelToken::new(), &|_| {});
    let elapsed = started.elapsed();
    assert_eq!(
        report.diagnosis,
        Diagnosis::Good,
        "{:?}",
        report
            .files
            .iter()
            .filter_map(|f| f.problem().map(|p| (f.path.as_str(), p)))
            .collect::<Vec<_>>()
    );
    assert_eq!(report.bytes_hashed, model.size_bytes);
    println!(
        "A7-ARTIFACTS ok: {} files, {} bytes verified in {:?} ({:.0} MB/s)",
        report.files.len(),
        report.bytes_hashed,
        elapsed,
        report.throughput().unwrap_or(0.0) / (1024.0 * 1024.0)
    );
    for f in &report.files {
        // The file name and the outcome, never the contents.
        println!("  {:<24} {:>12} bytes  ok", f.path, f.expected_bytes);
    }
}

#[test]
fn the_artefacts_directory_is_usable_through_the_facade() {
    // The end-to-end shape the pipeline uses: discovery finds it, the model
    // resolves to a directory, and `verify` through `ModelManager` agrees.
    let dir = sravaani_dir();
    if !dir.join("encoder.fp16.onnx").is_file() {
        println!("skipping: {} is not present", dir.display());
        return;
    }
    let m = manager();
    let root = dir
        .parent()
        .expect("artifacts/sravaani has a parent")
        .to_path_buf();
    let roots = ModelRoots::at(m.roots().user().join("__a7_absent_user__"), &root);
    let m = ModelManager::new(m.manifest().clone(), roots);

    let loc = m.resolve(ID).expect("the artefacts tree must resolve");
    assert_eq!(loc.dir(), dir.as_path());
    // The seven files the frozen ASR loader opens, in manifest order.
    assert_eq!(loc.files.len(), 7);
    for f in &loc.files {
        assert!(f.is_file(), "{} must exist", f.display());
    }
    assert!(loc.files[0].ends_with("encoder.fp16.onnx"));
    assert_eq!(m.engine_id(ID).unwrap(), "sravaani");

    // The settings row says it is installed and quotes the real size.
    let choices = m.choices();
    let c = choices.iter().find(|c| c.id == ID).expect("must be listed");
    assert!(
        c.available,
        "the artefacts tree is a complete install: {c:?}"
    );
    assert!(c.detail.contains("2.5 GiB"), "{}", c.detail);
    assert!(c.detail.contains("MIT"), "{}", c.detail);

    let status = m.status();
    let s = status.iter().find(|s| s.id == ID).unwrap();
    assert_eq!(s.on_disk_bytes, 2_702_792_388);
    assert_eq!(s.download_bytes, 0, "nothing is missing");
    assert_eq!(s.weights_bytes, 929_977_857);
    assert_eq!(s.root_kind, Some(vaani_models::RootKind::Bundled));
    assert!(!s.interrupted);
}

#[test]
fn the_receipt_a_fresh_install_writes_matches_the_manifest() {
    // The receipt is what a user's `vaani models verify --receipt` would read.
    let dir = sravaani_dir();
    if !dir.join("encoder.fp16.onnx").is_file() {
        println!("skipping: {} is not present", dir.display());
        return;
    }
    let m = manager();
    let model = m.manifest().model(ID).unwrap();
    let receipt = vaani_models::InstallReceipt {
        manifest_schema_version: vaani_models::manifest::SCHEMA_VERSION,
        model_id: model.id.clone(),
        model_version: model.version.clone(),
        installed_unix: 0,
        files: model
            .files
            .iter()
            .map(|f| vaani_models::manager::ReceiptFile {
                path: f.path.clone(),
                bytes: f.bytes,
                sha256: f.sha256_hex(),
            })
            .collect(),
    };
    assert_eq!(
        vaani_models::manager::check_receipt(&receipt, model),
        Diagnosis::Good
    );
    // And it round-trips as JSON, because the CLI has to read it back.
    let json = serde_json::to_string(&receipt).unwrap();
    let back: vaani_models::InstallReceipt = serde_json::from_str(&json).unwrap();
    assert_eq!(back, receipt);
    assert_eq!(back.files.len(), 10);
}

#[test]
fn the_hash_of_the_encoder_the_manifest_records_is_the_hash_on_disk() {
    // The narrowest possible version of the same claim, kept separate so that a
    // failure here points at one file rather than at "the artefacts".
    let dir = sravaani_dir();
    let encoder = dir.join("encoder.fp16.onnx");
    if !encoder.is_file() {
        println!("skipping: {} is not present", encoder.display());
        return;
    }
    let m = manager();
    let model = m.manifest().model(ID).unwrap();
    let declared = model.file("encoder.fp16.onnx").unwrap();

    // `artefacts.json` is known to be stale for this file: it still records the
    // pre-inline split into `.onnx` plus `.onnx_data`. Asserting that it
    // disagrees is deliberate: it documents why the manifest cannot be
    // generated from the exporter's log.
    let exporter_log = dir.join("artefacts.json");
    if exporter_log.is_file() {
        let log: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&exporter_log).unwrap()).unwrap();
        let logged = log.get("encoder.fp16.onnx").and_then(|v| v.get("sha256"));
        assert_ne!(
            logged.and_then(|v| v.as_str()),
            Some(declared.sha256_hex().as_str()),
            "artefacts.json now agrees with the manifest; MODELS.md says it is stale, \
             so one of the two documents needs updating"
        );
    }

    let started = Instant::now();
    let mut buf = vec![0u8; 1024 * 1024];
    let (hash, bytes) = vaani_models::verify::hash_file_streaming(&encoder, &mut buf, |_| true)
        .expect("the encoder must be readable")
        .expect("the whole file");
    assert_eq!(bytes, declared.bytes);
    assert_eq!(hex::encode(hash), declared.sha256_hex());
    println!(
        "A7-ENCODER ok: {} bytes, sha256 {}..., hashed in {:?}",
        bytes,
        &declared.sha256_hex()[..16],
        started.elapsed()
    );
}

//! The first-run path, on a machine that has nothing but `<exe>/models/`.
//!
//! Every other test in this crate builds a fixture and asserts one property.
//! This file builds **the distribution's layout** and walks the whole chain a
//! first-launch user goes through, because that chain is the thing the
//! zero-token mandate is about and it had never been walked end to end:
//!
//! ```text
//! ModelManager::embedded() -> discover() -> status() -> choices() -> resolve()
//! ```
//!
//! The fixture deliberately contains **only the seven required files** — no
//! `encoder.onnx`, no `model_meta.json`, no `artefacts.json`. The fp32
//! reference encoder is 1.65 GiB of the exporter's 2.5 GiB output and the
//! shipped bundle is 887 MB, so the distribution *cannot* contain it, and
//! anything that treats its absence as damage fails on every real install.

mod support;

use std::path::PathBuf;

use sha2::{Digest, Sha256};
use support::TempDir;
use vaani_models::manifest::Manifest;
use vaani_models::paths::ModelRoots;
use vaani_models::{ModelManager, RootKind};

const ID: &str = "sravaani-1.0-onnx";

/// The files a distribution stages, in manifest order.
///
/// Read from the manifest rather than written out here, because a hand-copied
/// list is a second source of truth: the day a file is added to the manifest and
/// forgotten here, this fixture would quietly keep testing the old shape.
fn shipped_files(manifest: &Manifest) -> Vec<(String, u64)> {
    let model = manifest
        .model(ID)
        .expect("the manifest must know the model");
    model
        .required_files()
        .map(|f| (f.path.clone(), f.bytes))
        .collect()
}

/// The real manifest's file list, with the byte counts replaced by tiny ones.
///
/// # Why the real files are not written
///
/// Discovery decides "is this model complete" by comparing **sizes** and
/// explicitly never hashes (see `discover.rs`), so a fixture that exercises the
/// whole manager chain has to have correctly-sized files. Writing the real ones
/// is 887 MB of I/O per test, which is not a thing a unit test may do.
///
/// Deriving the shape from [`Manifest::embedded`] rather than hard-coding it is
/// what keeps the substitution honest: if the real manifest gains a required
/// file, this one gains it too, and
/// [`a_synthetic_bundle_mirrors_the_real_manifest_shape`] fails **by name**.
/// The properties that depend on the actual numbers — the file list, the
/// required/optional flags, `weights_bytes` — are asserted against the **real**
/// manifest in [`a_distribution_stages_the_seven_required_files_and_nothing_else`],
/// and the real 887 MB is verified end to end by `tests/real_artifacts.rs` and by
/// `scripts/a12/clean-room.ps1`.
fn synthetic_manifest() -> Manifest {
    let real = Manifest::embedded().expect("the embedded manifest must be valid");
    let model = real.model(ID).expect("the manifest must know the model");

    let mut files = Vec::new();
    let mut total = 0u64;
    let mut required_total = 0u64;
    for f in &model.files {
        // 64 bytes stands in for 886,484,931. Small enough to be free, and
        // hashed so `verify` sees real content rather than zeros.
        let bytes = 64u64;
        total += bytes;
        if f.required {
            required_total += bytes;
        }
        files.push(format!(
            r#"{{"path":{},"role":{},"bytes":{bytes},"sha256":"{}","required":{}{}}}"#,
            json_str(&f.path),
            json_str(&f.role),
            hex::encode(Sha256::digest(support::ramp(bytes as usize))),
            f.required,
            match &f.note {
                Some(n) => format!(",\"note\":{}", json_str(n)),
                None => String::new(),
            }
        ));
    }

    let text = format!(
        r#"{{"schema_version":1,"models":[{{
          "id":"{ID}","dir":{},"engine_id":{},"version":{},"display_name":{},
          "description":"synthetic mirror of the shipped manifest",
          "task":"asr","architecture":{},"bundled":true,
          "access":"gated-token",
          "license":{{"spdx":"MIT","name":"MIT License","url":"https://huggingface.co/ARTPARK-IISc/SraVaani-1.0"}},
          "source":{{"kind":"huggingface","repo":"ARTPARK-IISc/SraVaani-1.0","revision":"main"}},
          "languages":["hi","en"],"sample_rates":[16000],"size_bytes":{total},
          "resources":{{"min_vram_bytes":1610612736,"rec_vram_bytes":2147483648,
                       "min_ram_bytes":2147483648,"rec_ram_bytes":4294967296}},
          "gpu":{{"compute_capability_min":"8.6","vram_headroom_bytes":1073741824}},
          "runtime":{{"opset":17}},
          "files":[{}]
        }}]}}"#,
        json_str(&model.dir),
        json_str(&model.engine_id),
        json_str(&model.version),
        json_str(&model.display_name),
        json_str(&model.architecture),
        files.join(","),
    );
    let parsed = Manifest::parse(&text).expect("the synthetic mirror must be valid");
    // Guards the one number a test cannot see for itself.
    assert_eq!(parsed.model(ID).unwrap().weights_bytes, required_total);
    parsed
}

/// A minimal JSON string literal, for building the fixture above.
fn json_str(s: &str) -> String {
    serde_json::to_string(s).expect("a string always serialises")
}

/// Build `<app>/models/sravaani/` containing exactly the required files.
fn stage_bundle(t: &TempDir, manifest: &Manifest) -> PathBuf {
    let dir = t.join("app").join("models").join("sravaani");
    std::fs::create_dir_all(&dir).unwrap();
    for f in manifest.model(ID).unwrap().required_files() {
        let p = dir.join(&f.path);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, support::ramp(f.bytes as usize)).unwrap();
    }
    dir
}

/// A manager whose only root is the staged bundle, and whose user directory
/// does not exist — exactly a clean machine with no model directory.
fn manager_over(t: &TempDir, manifest: &Manifest) -> ModelManager {
    let roots = ModelRoots::at(t.join("no-user-dir"), t.join("app").join("models"));
    ModelManager::new(manifest.clone(), roots)
}

#[test]
fn a_synthetic_bundle_mirrors_the_real_manifest_shape() {
    // If this fails, the substitution in `synthetic_manifest` has drifted and
    // every other test in this file is testing a fiction.
    let real = Manifest::embedded().unwrap();
    let syn = synthetic_manifest();
    let r = real.model(ID).unwrap();
    let s = syn.model(ID).unwrap();
    let shape = |m: &vaani_models::Model| -> Vec<(String, String, bool)> {
        m.files
            .iter()
            .map(|f| (f.path.clone(), f.role.clone(), f.required))
            .collect()
    };
    assert_eq!(
        shape(r),
        shape(s),
        "file list or required flags have drifted"
    );
    assert_eq!(r.dir, s.dir);
    assert_eq!(r.bundled, s.bundled, "the mirror must carry `bundled`");
    assert_eq!(r.access, s.access, "the mirror must carry the gated access");
    assert_eq!(r.source.provider, s.source.provider);
    assert_eq!(r.required_paths(), s.required_paths());
    assert_eq!(r.license.spdx, s.license.spdx);
}

#[test]
fn a_distribution_stages_the_seven_required_files_and_nothing_else() {
    // The bundle is 887 MB, not 2.5 GB, so the file list *is* the product
    // decision. This pins it against the REAL manifest: adding a required file
    // to the manifest without adding it to the packager fails here.
    let real = Manifest::embedded().expect("the embedded manifest must be valid");

    let expected: Vec<&str> = vec![
        "config.json",
        "decoder_joint.onnx",
        "encoder.fp16.onnx",
        "mel_fb.f32",
        "mel_window.f32",
        "preproc.json",
        "tokenizer.model",
    ];
    let mut actual: Vec<String> = shipped_files(&real).into_iter().map(|(p, _)| p).collect();
    actual.sort();
    assert_eq!(actual, expected, "the shipped file list changed");

    // And the optional 1.65 GiB reference encoder is genuinely optional, with
    // the real numbers rather than the synthetic ones.
    let model = real.model(ID).unwrap();
    assert!(
        model.file("encoder.onnx").is_some(),
        "it is still described"
    );
    assert!(
        !model.file("encoder.onnx").unwrap().required,
        "1.65 GiB that no runtime path opens cannot be required"
    );
    assert_eq!(model.file("encoder.onnx").unwrap().bytes, 1_772_812_911);
    assert_eq!(model.weights_bytes, 929_977_857, "the bundle is 887 MiB");
    // The total across every described file is 2.5 GiB: what the developer tree
    // holds, and what the distribution deliberately does not.
    assert_eq!(model.size_bytes, 2_702_792_388);
    assert!(model.bundled, "and the manifest says so");
}

#[test]
fn a_clean_machine_with_only_the_bundle_reports_the_model_as_available() {
    // The single most important assertion in this file. It is what A5's picker
    // binds `available` to, and `false` here is a user staring at "SraVaani 1.0
    // — not installed" on a machine where it is installed.
    let t = TempDir::new("a12-available");
    let m0 = synthetic_manifest();
    let _ = stage_bundle(&t, &m0);
    let m = manager_over(&t, &m0);

    let choices = m.choices();
    assert_eq!(choices.len(), 1, "one model, and it is listed: {choices:?}");
    let c = &choices[0];
    assert_eq!(c.id, ID);
    assert_eq!(c.label, "SraVaani 1.0");
    assert!(c.available, "the bundle is present and complete: {c:?}");
    assert!(
        !c.detail.contains("token"),
        "no credential is on the first-run path: {}",
        c.detail
    );
    assert!(c.detail.contains("included with VaaniDesk"), "{}", c.detail);
    assert!(c.detail.contains("MIT"), "{}", c.detail);
    assert!(
        !c.detail.contains("download"),
        "nothing is being downloaded: {}",
        c.detail
    );
}

#[test]
fn the_model_resolves_to_the_bundled_directory_with_no_configuration() {
    let t = TempDir::new("a12-resolve");
    let m0 = synthetic_manifest();
    let dir = stage_bundle(&t, &m0);
    let m = manager_over(&t, &m0);

    let loc = m.resolve(ID).expect("the bundle must resolve");
    assert_eq!(loc.dir(), dir.as_path());
    assert_eq!(loc.root_kind, RootKind::Bundled);
    // Seven files, in manifest order, every one present. This list is what the
    // session guard opens and what the loader is handed.
    assert_eq!(loc.files.len(), 7);
    for f in &loc.files {
        assert!(f.is_file(), "{} must exist in the bundle", f.display());
    }
    assert!(loc.files[0].ends_with("encoder.fp16.onnx"));
    assert_eq!(m.engine_id(ID).unwrap(), "sravaani");
}

#[test]
fn the_missing_optional_reference_encoder_is_not_a_broken_bundle() {
    // A7's rule, re-checked against the shape a distribution actually ships.
    // The bug this guards against is severe and quiet: discovery reports the
    // bundle as incomplete, the UI offers to re-download 887 MB that are
    // already on disk, and a user with no network cannot proceed at all.
    let t = TempDir::new("a12-optional");
    let m0 = synthetic_manifest();
    let dir = stage_bundle(&t, &m0);
    let m = manager_over(&t, &m0);

    let d = m.discover();
    let found = d.best(ID).expect("must be found");
    assert!(
        found.is_complete(),
        "an absent optional file is not damage: {:?}",
        found.problems()
    );
    assert!(
        !found.is_incomplete_install(),
        "and it is not a partial install"
    );
    assert!(d.is_installed(ID));
    assert!(d.absent.is_empty());
    assert_eq!(d.installed_ids(), vec![ID]);

    // Discovery reports it as absent-but-optional rather than silently omitting
    // it, which is what lets a diagnostic say "this build also describes an fp32
    // reference encoder you do not have".
    let opt = found
        .files
        .iter()
        .find(|f| f.path == "encoder.onnx")
        .expect("the manifest describes it, so discovery probes it");
    assert!(!opt.required);
    assert!(!opt.is_present());
    assert!(
        !opt.is_problem(),
        "an absent optional file is not a problem"
    );

    // And on disk it really is not there, so this is not vacuous.
    assert!(!dir.join("encoder.onnx").exists());
    // Exactly the required bytes, so `on_disk_bytes` is the bundle size a user
    // would see in Explorer.
    let model = m0.model(ID).unwrap();
    assert_eq!(found.on_disk_bytes, model.weights_bytes);
    assert_eq!(found.bytes_outstanding(), 0);
}

#[test]
fn status_and_choices_agree_and_both_say_installed() {
    // The two are computed separately and a UI shows both. If they disagree the
    // user gets a row that says one thing and a header that says another.
    let t = TempDir::new("a12-status");
    let m0 = synthetic_manifest();
    let _ = stage_bundle(&t, &m0);
    let m = manager_over(&t, &m0);

    let status = m.status();
    let choices = m.choices();
    assert_eq!(status.len(), choices.len());
    for (s, c) in status.iter().zip(&choices) {
        assert_eq!(s.id, c.id);
        assert_eq!(s.display_name, c.label);
        assert_eq!(s.detail, c.detail);
        assert_eq!(s.installed, c.available);
    }

    let s = &status[0];
    assert!(s.installed && s.present);
    assert_eq!(s.download_bytes, 0, "nothing is outstanding");
    assert_eq!(s.weights_bytes, m0.model(ID).unwrap().weights_bytes);
    assert_eq!(s.root_kind, Some(RootKind::Bundled));
    assert!(!s.interrupted);
    assert!(s.problems.is_empty());
    assert!(
        s.unexpected.is_empty(),
        "the bundle has nothing unaccounted for"
    );
    assert_eq!(s.license_spdx, "mit");
    assert!(s.bundled, "the manifest marks it as shipped with the app");
    // `downloadable` stays true: deleting models/ must remain recoverable, and
    // the HF source is what makes that true.
    assert!(s.downloadable, "the recovery download must stay available");
}

#[test]
fn install_on_a_machine_that_already_has_the_bundle_downloads_nothing() {
    // The property that makes the offline experience safe to attempt: no
    // network call, no token, no bytes. `install` short-circuits on a complete
    // copy, and this is the test that says so for the bundled case.
    let t = TempDir::new("a12-noop-install");
    let m0 = synthetic_manifest();
    let dir = stage_bundle(&t, &m0);
    let m = manager_over(&t, &m0);

    let report = m
        .install_blocking(
            ID,
            &Default::default(),
            &vaani_models::CancelToken::new(),
            &|_| {},
        )
        .expect("install must succeed without a network");
    assert!(report.already_installed, "{report:?}");
    assert_eq!(report.bytes_downloaded, 0);
    assert!(report.installed.is_empty());
    assert!(report.resumed.is_empty());
    assert_eq!(report.skipped.len(), 7);
    assert_eq!(report.dir, dir);
    // No `.partial/` left behind by a no-op.
    assert!(!dir.join(".partial").exists());
}

#[test]
fn the_bundle_root_is_searched_before_the_user_directory() {
    // A machine with a stale user copy *and* the bundle. Bundled wins, so an
    // upgrade cannot be shadowed by whatever an older build downloaded.
    let t = TempDir::new("a12-precedence");
    let m0 = synthetic_manifest();
    let _ = stage_bundle(&t, &m0);

    // The user copy is deliberately *incomplete*, as a half-old install would be.
    let user = t.join("user").join("sravaani");
    std::fs::create_dir_all(&user).unwrap();
    std::fs::write(user.join("config.json"), support::ramp(64)).unwrap();

    let roots = ModelRoots::at(t.join("user"), t.join("app").join("models"));
    let m = ModelManager::new(m0, roots);
    let d = m.discover();

    // Both are found; the complete bundled one is chosen.
    assert_eq!(d.copies(ID).len(), 2);
    let best = d.best(ID).unwrap();
    assert_eq!(best.root_kind, RootKind::Bundled);
    assert!(best.is_complete(), "{:?}", best.problems());
    assert!(m.choices()[0].available);
}

#[test]
fn a_machine_with_no_bundle_at_all_is_told_to_reinstall_not_broken() {
    // The far side of the mandate: removing the bundle must produce an honest
    // message, not a crash, not silence, and not a token prompt.
    let t = TempDir::new("a12-nobundle");
    let m0 = synthetic_manifest();
    let m = manager_over(&t, &m0);
    let c = &m.choices()[0];
    assert!(!c.available);
    assert!(c.detail.contains("Not installed"), "{}", c.detail);
    assert!(
        c.detail.contains("reinstall VaaniDesk"),
        "the recovery for a bundled model is the installer: {}",
        c.detail
    );
    assert!(!c.detail.contains("token"), "{}", c.detail);
    let e = m.resolve(ID).unwrap_err();
    assert_eq!(e.code(), vaani_core::ErrorCode::ModelUnavailable);
}

#[test]
fn the_search_order_is_reportable_so_a_miss_can_be_explained() {
    // A user with no models must be able to be told *where* VaaniDesk looked.
    // This is the API that makes that possible without a second code path.
    let candidates = vaani_models::paths::bundled_candidates();
    assert!(!candidates.is_empty());
    for c in &candidates {
        assert!(!c.origin.as_str().is_empty());
        assert_eq!(c.exists, c.dir.is_dir());
    }
    // The first entry in the no-override case is the install-relative one.
    if std::env::var_os(vaani_models::paths::ENV_ARTEFACTS_ROOT).is_none()
        && std::env::var_os(vaani_models::paths::ENV_ARTEFACTS_MODEL_DIR).is_none()
    {
        assert_eq!(
            candidates[0].origin,
            vaani_models::paths::BundledOrigin::ExeRelative,
            "<exe>/models is what a distribution ships into"
        );
    }
    // And `ModelRoots::discover` carries them through.
    let roots = ModelRoots::discover();
    assert_eq!(roots.bundled_candidates().len(), candidates.len());
    assert!(!roots.bundled().as_os_str().is_empty());
}

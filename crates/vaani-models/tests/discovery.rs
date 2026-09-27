//! Discovery against a real directory tree.
//!
//! Every case is a tree a user can actually have: nothing installed, one model
//! complete, one missing a file, a file the manifest never mentioned, a nested
//! layout, and a file with the right size and the wrong contents. The last one
//! is the interesting one: discovery must **not** report it, and the reason why
//! is a design decision rather than an omission.

mod support;

use std::path::Path;

use support::TempDir;
use vaani_models::manifest::Manifest;
use vaani_models::paths::{ModelRoots, RootKind};
use vaani_models::{ModelManager, discover};

/// A one-file model, so a test can make exactly one thing wrong.
fn one_file_manifest() -> Manifest {
    Manifest::parse(
        r#"{"schema_version": 1, "models": [
        { "id": "m1", "engine_id": "e1", "version": "1", "display_name": "M1",
          "description": "d", "task": "asr", "architecture": "a", "access": "open",
          "license": {"spdx":"MIT","name":"MIT","url":"https://example.org/l"},
          "source": {"kind":"bundled"}, "languages": ["hi"], "sample_rates": [16000],
          "size_bytes": 24,
          "files": [
            {"path":"weights.bin","role":"weights","bytes":8,
             "sha256":"0000000000000000000000000000000000000000000000000000000000000000"},
            {"path":"fe/tables.bin","role":"frontend","bytes":16,
             "sha256":"1111111111111111111111111111111111111111111111111111111111111111"}
          ]},
        { "id": "m2", "engine_id": "e2", "version": "1", "display_name": "M2",
          "description": "d", "task": "vad", "architecture": "a", "access": "open",
          "license": {"spdx":"Apache-2.0","name":"Apache","url":"https://example.org/l"},
          "source": {"kind":"bundled"}, "languages": ["en"], "sample_rates": [16000],
          "size_bytes": 4,
          "files": [
            {"path":"vad.onnx","role":"weights","bytes":4,
             "sha256":"2222222222222222222222222222222222222222222222222222222222222222"}
          ]}
      ]}"#,
    )
    .expect("the fixture must be a valid manifest")
}

fn roots(base: &Path) -> ModelRoots {
    ModelRoots::at(base.join("user"), base.join("bundled"))
}

#[test]
fn nothing_installed_lists_every_model_as_absent() {
    let t = TempDir::new("disc-none");
    let m = one_file_manifest();
    let d = discover::discover(&m, &roots(t.path()));

    assert!(d.models.is_empty(), "{:?}", d.models);
    assert_eq!(d.absent, vec!["m1", "m2"]);
    assert!(d.unknown_dirs.is_empty());
    assert!(!d.is_installed("m1"));
    assert!(!d.is_present("m1"));
    assert!(d.installed_ids().is_empty());
    // Both roots were looked at and both were reported missing, rather than
    // silently skipped.
    assert_eq!(d.roots.len(), 2);
    assert!(d.roots.iter().all(|r| !r.exists));
    // And the models are still *offered*: absent is not the same as unknown.
    assert_eq!(
        d.availability(&m),
        vec![("m1".to_string(), false), ("m2".to_string(), false)]
    );
}

#[test]
fn one_complete_model_is_installed() {
    let t = TempDir::new("disc-full");
    t.file("user/m1/weights.bin", 8);
    t.file("user/m1/fe/tables.bin", 16);
    let m = one_file_manifest();
    let d = discover::discover(&m, &roots(t.path()));

    let found = d.best("m1").expect("must be found");
    assert_eq!(found.root_kind, RootKind::User);
    assert!(found.is_complete(), "{:?}", found.problems());
    assert_eq!(found.dir, t.join("user/m1"));
    assert_eq!(found.on_disk_bytes, 24);
    assert_eq!(found.bytes_outstanding(), 0);
    assert!(d.is_installed("m1"));
    assert!(!d.is_installed("m2"));
    assert_eq!(d.installed_ids(), vec!["m1"]);
    assert!(d.absent.contains(&"m2".to_string()));
}

#[test]
fn a_model_missing_one_file_is_present_but_not_installed() {
    let t = TempDir::new("disc-missing");
    t.file("user/m1/weights.bin", 8);
    // fe/tables.bin absent.
    let m = one_file_manifest();
    let d = discover::discover(&m, &roots(t.path()));

    let found = d.best("m1").unwrap();
    assert!(!found.is_complete());
    assert!(d.is_present("m1") && !d.is_installed("m1"));
    let problems = found.problems();
    assert_eq!(problems.len(), 1);
    assert_eq!(problems[0].path, "fe/tables.bin");
    assert!(!problems[0].is_present());
    assert_eq!(found.bytes_outstanding(), 16);
    assert!(found.is_incomplete_install());
}

#[test]
fn a_file_with_the_right_size_and_the_wrong_content_is_not_a_discovery_problem() {
    // The deliberate decision: discovery answers "is it here and is it the
    // right length", not "is it the right bytes". Hashing 2.5 GB on every
    // settings-panel refresh is a six-second freeze, so content is
    // `verify`'s job and the UI has a button for it. The file is still
    // *reachable* by verify, which is the part that matters.
    let t = TempDir::new("disc-samesize");
    t.bytes("user/m1/weights.bin", &support::other_ramp(8));
    t.bytes("user/m1/fe/tables.bin", &support::other_ramp(16));
    let m = one_file_manifest();
    let d = discover::discover(&m, &roots(t.path()));

    let found = d.best("m1").unwrap();
    assert!(
        found.is_complete(),
        "discovery must not hash: {:?}",
        found.problems()
    );
    let manager = ModelManager::new(m, roots(t.path()));
    let report = manager
        .verify("m1", &vaani_models::CancelToken::new(), &|_| {})
        .expect("verify must run");
    assert!(!report.is_ok());
    assert!(
        report.files.iter().any(|f| matches!(
            f.problem(),
            Some(vaani_models::FileProblem::ContentDiffers { .. })
        )),
        "{:?}",
        report.files
    );
}

#[test]
fn a_wrong_size_is_reported_by_discovery_without_hashing() {
    let t = TempDir::new("disc-wrongsize");
    t.file("user/m1/weights.bin", 7); // one byte short
    t.file("user/m1/fe/tables.bin", 16);
    let m = one_file_manifest();
    let d = discover::discover(&m, &roots(t.path()));

    let found = d.best("m1").unwrap();
    let w = found
        .files
        .iter()
        .find(|f| f.path == "weights.bin")
        .unwrap();
    assert!(w.is_present() && !w.is_correct_size());
    assert_eq!(w.actual_bytes, Some(7));
    assert_eq!(w.expected_bytes, 8);
    assert!(found.is_incomplete_install());
}

#[test]
fn an_unknown_extra_file_is_reported_and_never_fails_the_model() {
    let t = TempDir::new("disc-extra");
    t.file("user/m1/weights.bin", 8);
    t.file("user/m1/fe/tables.bin", 16);
    t.file("user/m1/notes-from-a-human.txt", 5);
    t.dir("user/m1/scratch");
    let m = one_file_manifest();
    let d = discover::discover(&m, &roots(t.path()));

    let found = d.best("m1").unwrap();
    assert_eq!(found.unexpected, vec!["notes-from-a-human.txt"]);
    assert!(found.is_complete(), "an extra file is not a broken model");
    // 8 + 16 + 5; the nested file and the directory are counted correctly.
    assert_eq!(found.on_disk_bytes, 29);
}

#[test]
fn our_own_bookkeeping_is_not_reported_as_a_stray_file() {
    let t = TempDir::new("disc-mine");
    t.file("user/m1/weights.bin", 8);
    t.file("user/m1/fe/tables.bin", 16);
    t.file("user/m1/.installed.json", 40);
    t.file("user/m1/.partial/weights.bin.123.part", 4);
    t.file("user/.vaani-locks/m1.lock", 60);
    let m = one_file_manifest();
    let d = discover::discover(&m, &roots(t.path()));

    let found = d.best("m1").unwrap();
    assert!(found.unexpected.is_empty(), "{:?}", found.unexpected);
    assert!(found.has_partial, "an interrupted install is visible");
    // The lock directory lives under the model *root*, and is ours, so it is
    // never reported as an unclaimed model directory.
    assert!(
        d.unknown_dirs.is_empty(),
        "{:?}",
        d.unknown_dirs.iter().map(|u| &u.name).collect::<Vec<_>>()
    );
    // The model is complete *and* flagged: the files are all there, but a
    // `.partial` directory means something was in flight, and the UI shows that
    // rather than declaring victory.
    assert!(found.is_complete());
    assert!(found.is_incomplete_install());
}

#[test]
fn an_unclaimed_directory_is_reported_with_its_size() {
    let t = TempDir::new("disc-unknown");
    t.file("user/whisper-large/model.bin", 4096);
    t.file("user/whisper-large/config.json", 16);
    t.file("bundled/m1/weights.bin", 8);
    t.file("bundled/m1/fe/tables.bin", 16);
    let m = one_file_manifest();
    let d = discover::discover(&m, &roots(t.path()));

    assert_eq!(d.unknown_dirs.len(), 1, "{:?}", d.unknown_dirs);
    let u = &d.unknown_dirs[0];
    assert_eq!(u.name, "whisper-large");
    assert_eq!(u.root_kind, RootKind::User);
    assert_eq!(u.bytes, 4112);
    // The claimed directory in the other root is not reported as unclaimed.
    assert!(d.is_installed("m1"));
}

#[test]
fn the_user_root_wins_over_the_bundled_one_and_both_are_listed() {
    let t = TempDir::new("disc-prec");
    t.file("user/m1/weights.bin", 8);
    t.file("user/m1/fe/tables.bin", 16);
    t.file("bundled/m1/weights.bin", 8);
    t.file("bundled/m1/fe/tables.bin", 16);
    let m = one_file_manifest();
    let d = discover::discover(&m, &roots(t.path()));

    assert_eq!(d.copies("m1").len(), 2);
    assert_eq!(d.best("m1").unwrap().root_kind, RootKind::User);
}

#[test]
fn a_bundled_model_resolves_from_the_artefacts_tree_with_no_configuration() {
    // This is the shipped SraVaani case: `artifacts/sravaani/`, found with no
    // environment variable and no setting, exactly as
    // `vaani_asr::sravaani::default_dir` finds it.
    let t = TempDir::new("disc-bundled");
    t.file("artifacts/m1/weights.bin", 8);
    t.file("artifacts/m1/fe/tables.bin", 16);
    let m = one_file_manifest();
    let r = ModelRoots::at(t.join("user"), t.join("artifacts"));
    let d = discover::discover(&m, &r);

    assert!(d.is_installed("m1"));
    assert_eq!(d.best("m1").unwrap().dir, t.join("artifacts/m1"));
    assert_eq!(d.best("m1").unwrap().root_kind, RootKind::Bundled);
}

#[test]
fn a_configured_root_is_searched_last() {
    let t = TempDir::new("disc-configured");
    t.file("extra/m1/weights.bin", 8);
    t.file("extra/m1/fe/tables.bin", 16);
    let m = one_file_manifest();
    let r = ModelRoots::at(t.join("user"), t.join("bundled")).with_configured(t.join("extra"));
    let d = discover::discover(&m, &r);

    assert_eq!(d.best("m1").unwrap().root_kind, RootKind::Configured);
    assert_eq!(d.best("m1").unwrap().dir, t.join("extra/m1"));
}

#[test]
fn status_and_choices_agree_with_discovery() {
    let t = TempDir::new("disc-status");
    t.file("user/m1/weights.bin", 8);
    t.file("user/m1/fe/tables.bin", 16);
    let m = one_file_manifest();
    let manager = ModelManager::new(m, roots(t.path()));

    let status = manager.status();
    assert_eq!(status.len(), 2);
    assert!(status[0].installed && !status[1].present);
    assert_eq!(status[0].license_spdx, "mit");
    assert_eq!(status[0].total_bytes, 24);
    assert_eq!(status[0].weights_bytes, 24);
    assert_eq!(status[0].on_disk_bytes, 24);
    assert_eq!(status[0].download_bytes, 0);
    assert_eq!(status[0].engine_id, "e1");
    assert_eq!(status[1].download_bytes, 4);

    // The exact shape A5's UI binds to.
    let choices = manager.choices();
    assert_eq!(choices.len(), 2);
    assert_eq!(choices[0].id, "m1");
    assert_eq!(choices[0].label, "M1");
    assert!(choices[0].available);
    assert!(
        choices[0].detail.contains("Installed"),
        "{}",
        choices[0].detail
    );
    assert!(!choices[1].available, "m2 is known but not installed");
    assert!(
        choices[1].detail.contains("Not installed"),
        "an unavailable row must say why: {}",
        choices[1].detail
    );
    assert_eq!(choices[1].detail, status[1].detail);
}

#[test]
fn a_half_installed_model_is_offered_with_its_reason_not_hidden() {
    let t = TempDir::new("disc-half");
    t.file("user/m1/weights.bin", 8);
    let m = one_file_manifest();
    let manager = ModelManager::new(m, roots(t.path()));
    let choices = manager.choices();
    let c = choices.iter().find(|c| c.id == "m1").unwrap();
    assert!(!c.available, "a model missing a file cannot be selected");
    assert!(c.detail.contains("Not ready"), "{}", c.detail);
    assert!(c.detail.contains("missing or damaged"), "{}", c.detail);
    assert_eq!(choices.len(), 2, "it is shown, not hidden");
}

#[test]
fn resolve_refuses_a_model_that_is_not_fully_installed() {
    let t = TempDir::new("disc-resolve");
    t.file("user/m1/weights.bin", 8);
    let m = one_file_manifest();
    let manager = ModelManager::new(m, roots(t.path()));

    let e = manager.resolve("m1").unwrap_err();
    assert_eq!(e.code(), vaani_core::ErrorCode::ModelUnavailable);
    assert!(e.user.recovery.is_some());

    // Once the file is there, resolve hands back the directory the ASR loader
    // wants, with the seven-style path list in manifest order.
    t.file("user/m1/fe/tables.bin", 16);
    let loc = manager.resolve("m1").expect("now installed");
    assert_eq!(loc.dir(), t.join("user/m1").as_path());
    assert_eq!(loc.files.len(), 2);
    assert!(loc.files[0].ends_with("weights.bin"));

    // An id nobody has.
    let e = manager.resolve("nope").unwrap_err();
    assert!(e.user.recovery.unwrap().contains("vaani models list"));
}

#[test]
fn discovery_is_fast_enough_to_run_at_startup() {
    // The requirement is "a user with a 2 GB model directory must not wait
    // seconds for a list". Measured, not asserted by construction: a tree with
    // one large model and 400 extra files, ten passes, and a budget.
    let t = TempDir::new("disc-perf");
    t.file("user/m1/weights.bin", 8);
    t.file("user/m1/fe/tables.bin", 16);
    for i in 0..400 {
        t.file(&format!("user/m1/extra{i}.bin"), 64);
    }
    let m = one_file_manifest();
    let r = roots(t.path());

    // Warm the filesystem cache so the number is not a cold-cache artefact.
    let _ = discover::discover(&m, &r);
    let start = std::time::Instant::now();
    let mut last = std::time::Duration::ZERO;
    for _ in 0..10 {
        last = discover::discover(&m, &r).elapsed;
    }
    let mean = start.elapsed() / 10;
    println!("discovery over 402 files in a model directory: mean {mean:?}, last pass {last:?}");
    // Generous, because a loaded machine is slow: the point is that it is
    // milliseconds, not that it is a particular number.
    assert!(
        mean < std::time::Duration::from_millis(250),
        "discovery took {mean:?} over 402 files, which is a startup freeze"
    );
}

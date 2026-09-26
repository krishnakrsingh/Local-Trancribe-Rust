//! Runtime smoke test: can this machine actually load and run an ONNX model?
//!
//! This is deliberately the *first* integration test in the repository,
//! because it fails for reasons no unit test can catch: a missing DLL, a
//! provider that refuses to register, a driver that is too old, a GPU with less
//! free VRAM than the model needs.
//!
//! It is skipped -- loudly, with a printed reason -- when the runtime or the
//! model is absent, so `cargo test` on a fresh checkout is still useful. It is
//! never silently passed: a skip prints the exact path it looked for.

use std::path::PathBuf;

use vaani_inference::{OnnxSession, Provider, RuntimeReport, runtime};

fn repo_root() -> PathBuf {
    // tests/ lives two levels under the workspace root.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crate is under <root>/crates/vaani-inference")
        .to_path_buf()
}

/// Locate an exported model artefact, or `None` with the path we looked for.
fn artefact(name: &str) -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("VAANI_ARTEFACTS") {
        let p = PathBuf::from(dir).join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    let p = repo_root().join("artifacts").join("sravaani").join(name);
    p.is_file().then_some(p)
}

#[test]
fn runtime_is_present_and_reports_itself() {
    let report = RuntimeReport::collect();
    println!("{}", report.summary());
    println!("  ort dll:  {}", report.ort_dll);
    println!("  missing:  {:?}", report.missing_dlls);
    println!("  adapters: {:?}", report.machine.adapters);
    println!("  cuda:     {:?}", report.machine.cuda);
    println!(
        "  cpu:      {} logical, {} usable",
        report.machine.logical_cpus, report.machine.usable_cpus
    );
    println!(
        "  ram:      {:.1} GB",
        report.machine.total_ram_bytes as f64 / 1e9
    );
    println!("  threads:  {}", report.machine.recommended_ort_threads());

    if !report.runtime_present {
        println!(
            "SKIP runtime smoke test: {} not found. Run scripts\\fetch_runtime.ps1.",
            report.ort_dll
        );
        return;
    }
    // If the runtime is present, the machine must have been describable. These
    // are the facts the settings panel shows a user, so a zero here is a bug
    // that would render as "unknown hardware" in the product.
    assert!(report.machine.logical_cpus >= 1);
    assert!(
        report.machine.total_ram_bytes > 0,
        "RAM query returned nothing"
    );
}

#[test]
fn cuda_encoder_runs_on_this_machine() {
    let Some(model) = artefact("encoder.fp16.onnx") else {
        println!("SKIP: artifacts/sravaani/encoder.fp16.onnx not present.");
        println!(
            "      Run: python scripts\\export_sravaani_onnx.py --src SraVaani-1.0 --out artifacts\\sravaani"
        );
        return;
    };
    if !runtime::location().is_present() {
        println!("SKIP: ONNX Runtime not present; see the runtime_is_present test.");
        return;
    }

    let report = RuntimeReport::collect();
    let caps = &report.machine;

    // Cap the arena at the *inference* adapter's free VRAM, minus a reserve for
    // the desktop compositor. Using the display adapter here would size the
    // arena against an integrated GPU's shared memory and defeat the cap.
    let vram_cap = caps
        .inference_adapter()
        .map(|a| a.free_vram_bytes().saturating_sub(256 * 1024 * 1024))
        .filter(|v| *v > 0);

    let started = std::time::Instant::now();
    let mut session = OnnxSession::load(&model, Provider::Auto, caps, vram_cap)
        .unwrap_or_else(|e| panic!("loading {} failed: {}", model.display(), e.debug_chain()));
    let load_ms = started.elapsed().as_millis();

    println!("loaded {} in {load_ms} ms", model.display());
    println!("  provider: {:?}", session.provider_status());
    println!("  cuda:     {:?}", caps.cuda);
    println!(
        "  inference adapter: {:?}",
        caps.inference_adapter()
            .map(|a| (&a.name, a.vram_display()))
    );
    println!("  inputs:   {:?}", session.input_names());
    println!("  outputs:  {:?}", session.output_names());
    println!("  threads:  {}", session.threads());

    // The encoder is exported in native fp16, so the graph boundary is fp16.
    // The front end computes log-mel in f32 and converts once, here, via
    // `F16Tensor::from_f32` -- which is why that conversion lives in the
    // inference layer and not in the ASR crate.
    //
    // T=1600 mel frames is 16 s of audio at hop 160; with 8x subsampling that
    // must yield exactly 200 encoder frames.
    let t_frames = 1600i64;
    let n_mels = 128i64;
    let feats_f32 = vaani_inference::F32Tensor::new(
        vec![1, n_mels, t_frames],
        (0..(n_mels * t_frames) as usize)
            .map(|i| (i % 97) as f32 / 97.0)
            .collect(),
    )
    .expect("built a valid test tensor");
    let feats = vaani_inference::F16Tensor::from_f32(&feats_f32);
    let lengths = vaani_inference::I64Tensor::scalar(t_frames);

    let run_once = |s: &mut vaani_inference::OnnxSession| {
        s.run(
            &[
                vaani_inference::Input::f16("feats", feats.clone()),
                vaani_inference::Input::i64("feature_lengths", lengths.clone()),
            ],
            &["enc_out", "enc_lengths"],
        )
        .unwrap_or_else(|e| panic!("encoder run failed: {}", e.debug_chain()))
    };

    // Warm-up: the first call pays cuDNN algorithm selection and CUDA module
    // loading. Timing it would put a 10x-worse RTF in the report.
    let _ = run_once(&mut session);

    let started = std::time::Instant::now();
    let out = run_once(&mut session);
    let elapsed = started.elapsed();

    let enc = out[0].as_f16().expect("enc_out is f16").to_f32();
    let enc_len = out[1].as_i64().expect("enc_lengths is i64");

    println!(
        "encoder: 16 s of audio in {} ms -> RTF {:.3}",
        elapsed.as_millis(),
        elapsed.as_secs_f64() / 16.0
    );
    println!(
        "  enc_out shape {:?}, enc_lengths {:?}",
        enc.shape, enc_len.data
    );

    assert_eq!(enc.shape[0], 1, "batch must be 1");
    assert_eq!(enc.shape[1], 1024, "SraVaani enc_d_model is 1024");
    assert!(
        enc.shape[2] >= 200,
        "expected at least 200 encoder frames for 1600 mel frames, got {}",
        enc.shape[2]
    );
    // ceil(1600 / 8) == 200. NeMo's subsampling is 8x, and the graph reports the
    // true length; assert it exactly so a change in the frontend is caught here
    // rather than as mysteriously bad transcripts.
    assert_eq!(
        enc_len.data[0], 200,
        "8x subsampling should give exactly 200 frames"
    );
    assert_eq!(
        enc.shape[2], 200,
        "output must be trimmed to the reported length"
    );
    assert!(
        enc.data.iter().all(|v| v.is_finite()),
        "encoder produced NaN or Inf; the model or the runtime is broken"
    );
    assert!(
        enc.data.iter().any(|v| *v != 0.0),
        "encoder produced an all-zero output for non-zero input"
    );
}

//! Diagnostic: which CUDA provider options can actually run the SraVaani encoder?
//!
//! Run with:
//!   cargo run -p vaani-inference --example conv_probe
//!
//! This exists because the CUDA execution provider offers several conv
//! strategies, ONNX Runtime's cuDNN path fails on this graph for at least one
//! of them, and the only honest way to find out which is to try them. The
//! winning combination is then hard-coded in `OnnxSession::load` with a
//! comment, and this file records why.
//!
//! It is an example rather than a test because it prints a table of
//! pass/fail-with-reason rather than asserting one answer, and because on a
//! machine without the runtime it is not meaningful.

use std::path::PathBuf;

use vaani_inference::session::Input;
use vaani_inference::{
    F16Tensor, F32Tensor, I64Tensor, MachineCapabilities, OnnxSession, Provider,
};

/// (label, which model, provider option overrides)
const TRIALS: &[(&str, &str)] = &[
    ("fp16 encoder, GPU", "encoder.fp16.onnx"),
    ("fp32 encoder, GPU", "encoder.onnx"),
];

fn artefact(name: &str) -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("under <root>/crates/")
        .to_path_buf();
    let p = root.join("artifacts").join("sravaani").join(name);
    p.is_file().then_some(p)
}

fn main() {
    let loc = vaani_inference::runtime::location();
    if !loc.is_present() {
        println!(
            "no ONNX Runtime at {}; run scripts\\fetch_runtime.ps1",
            loc.ort_dll.display()
        );
        return;
    }
    let caps: MachineCapabilities = vaani_inference::capability::detect(&loc.provider_dirs);
    println!(
        "machine: {}",
        vaani_inference::RuntimeReport::collect().summary()
    );
    println!("cuda:    {:?}", caps.cuda);
    println!();

    if artefact("encoder.fp16.onnx").is_none() {
        println!("no encoder.fp16.onnx; run the exporter first");
        return;
    }

    let t_frames = 1600i64;
    let n_mels = 128i64;
    let f32 = F32Tensor::new(
        vec![1, n_mels, t_frames],
        (0..(n_mels * t_frames) as usize)
            .map(|i| (i % 97) as f32 / 97.0)
            .collect(),
    )
    .expect("tensor");
    let f16t = F16Tensor::from_f32(&f32);
    let lengths = I64Tensor::scalar(t_frames);
    let vram = caps
        .inference_adapter()
        .map(|a| a.free_vram_bytes().saturating_sub(256 * 1024 * 1024))
        .filter(|v| *v > 0);

    println!("{:<24} {:<8} DETAIL", "TRIAL", "RESULT");
    println!("{}", "-".repeat(120));

    for (label, model) in TRIALS {
        let Some(path) = artefact(model) else {
            println!("{label:<44} {:<10} {model} not present", "SKIP");
            continue;
        };
        let is_fp16 = model.contains("fp16");
        match try_one(label, &path, &caps, vram, is_fp16, &f16t, &f32, &lengths) {
            Ok((ms, provider)) => println!(
                "{label:<24} {:<8} {provider}, 16 s audio in {ms:>6} ms  RTF {:.3}",
                "OK",
                ms as f64 / 1000.0 / 16.0
            ),
            Err(e) => {
                // One line, trimmed to the informative part: the ORT message is
                // 1500 characters of cuDNN JSON.
                let detail = e
                    .lines()
                    .map(str::trim)
                    .find(|l| l.contains("Failed") || l.contains("failure") || l.contains("error"))
                    .unwrap_or(e.lines().last().unwrap_or(""))
                    .to_string();
                let n = detail.len().min(88);
                println!("{label:<24} {:<8} {}", "FAIL", &detail[..n]);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn try_one(
    label: &str,
    model: &std::path::Path,
    caps: &MachineCapabilities,
    vram: Option<u64>,
    is_fp16: bool,
    f16t: &F16Tensor,
    f32: &F32Tensor,
    lengths: &I64Tensor,
) -> Result<(u128, String), String> {
    let mut session = OnnxSession::load_with_options(model, Provider::Auto, caps, vram, Vec::new())
        .map_err(|e| e.debug_chain())?;
    let provider = session.provider_status().active.as_str().to_string();

    // Warm up, then time, so cuDNN algorithm selection is not in the number.
    let run = |s: &mut OnnxSession| {
        let feats: Vec<Input> = if is_fp16 {
            vec![Input::f16("feats", f16t.clone())]
        } else {
            vec![Input::f32("feats", f32.clone())]
        };
        let mut v = feats;
        v.push(Input::i64("feature_lengths", lengths.clone()));
        s.run(&v, &["enc_out", "enc_lengths"])
            .map_err(|e| e.debug_chain())
    };
    run(&mut session).map_err(|e| format!("{label}: warm-up: {e}"))?;
    let t0 = std::time::Instant::now();
    let out = run(&mut session).map_err(|e| format!("{label}: {e}"))?;
    let ms = t0.elapsed().as_millis();
    let enc = if is_fp16 {
        out[0].as_f16().expect("f16").to_f32()
    } else {
        out[0].as_f32().expect("f32").clone()
    };
    if !enc.data.iter().all(|v| v.is_finite()) {
        return Err(format!("{label}: non-finite encoder output"));
    }
    Ok((ms, provider))
}

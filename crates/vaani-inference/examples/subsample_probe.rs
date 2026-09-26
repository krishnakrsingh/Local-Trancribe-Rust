//! Print the encoder's reported output length for a range of input lengths.
//!
//! The subsampling law is `ceil(T / 8)`, but that has to be *measured*, not
//! assumed: a wrong constant here drags every transcript out of sync with the
//! media, and the only way to be sure is to ask the graph.
//!
//!   cargo run -p vaani-inference --example subsample_probe

use vaani_inference::session::F32Tensor;
use vaani_inference::{F16Tensor, I64Tensor, Input, OnnxSession, Provider, capability, runtime};

fn main() {
    let loc = runtime::location();
    if !loc.is_present() {
        println!(
            "no runtime at {}; run scripts\\fetch_runtime.ps1",
            loc.ort_dll.display()
        );
        return;
    }
    let model = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("artifacts")
        .join("sravaani")
        .join("encoder.fp16.onnx");
    if !model.is_file() {
        println!("no encoder.fp16.onnx; run scripts\\export_sravaani_onnx.py");
        return;
    }
    let caps = capability::detect(&loc.provider_dirs);
    let vram = caps
        .inference_adapter()
        .map(|a| a.free_vram_bytes().saturating_sub(256 * 1024 * 1024))
        .filter(|v| *v > 0);
    let mut s = OnnxSession::load(&model, Provider::Auto, &caps, vram).expect("load");

    println!(
        "{:>8} {:>8} {:>8} {:>10}  shape/finite",
        "T", "T", "ceil(T/8)", "enc_lengths"
    );
    for t in [
        200i64, 400, 437, 800, 1600, 1601, 3200, 4000, 4800, 8000, 16000, 24000,
    ] {
        let ceil = (t + 7) / 8;
        let feats = F16Tensor::from_f32(
            &F32Tensor::new(
                vec![1, 128, t],
                (0..(128 * t) as usize)
                    .map(|i| (i % 97) as f32 / 97.0)
                    .collect(),
            )
            .expect("tensor"),
        );
        let out = s
            .run(
                &[
                    Input::f16("feats", feats),
                    Input::i64("feature_lengths", I64Tensor::scalar(t)),
                ],
                &["enc_out", "enc_lengths"],
            )
            .expect("run");
        let enc = out[0].as_f16().expect("f16").to_f32();
        let len = out[1].as_i64().expect("i64").data[0];
        let finite = enc.data.iter().all(|v| v.is_finite());
        println!(
            "{t:>8} {t:>8} {ceil:>8} {len:>10}  out2={} finite={}",
            enc.shape[2], finite
        );
    }
}

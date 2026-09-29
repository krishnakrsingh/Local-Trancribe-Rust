//! Micro-benchmark for SraVaani's decode joint, in isolation.
//!
//! The whole-pipeline profile says the decode loop costs a few hundred
//! microseconds per emission step. This says *why*, by calling the 21-node
//! graph in a tight loop with fixed inputs: whatever that costs per call is the
//! floor the real loop cannot go below, and it is the only number that moves
//! when a session option changes.
//!
//! It is also the place a custom kernel would have to beat. A transducer's
//! decode step is a prediction network plus a joint plus an argmax, all on one
//! frame, all sequentially dependent on the previous step. The arithmetic is
//! trivial; the cost is everything around it.
//!
//! Usage:
//!     cargo run --release --example decode-bench -- [--provider auto|cpu|cuda]
//!         [--iters 2000] [--cuda-graph] [--exhaustive] [--tf32] [--io-binding]

use std::time::{Duration, Instant};

use vaani_inference::{ConvSearch, CudaTuning, F32Tensor, I32Tensor, Input, OnnxSession, Provider};

/// The decode joint's shapes, from `decoder_joint.onnx`:
///
/// ```text
///   f       FLOAT [1, 1024, 1]
///   tgt     INT32 [1, 1]
///   tlen    INT32 [1]
///   h_in    FLOAT [1, 1, 640]
///   c_in    FLOAT [1, 1, 640]
///   -> logits FLOAT [1, 1, 1, 5006], target_length INT32 [1],
///      h_out FLOAT [1, 1, 640], c_out FLOAT [1, 1, 640]
/// ```
const D_MODEL: usize = 1024;
const STATE: usize = 640;
const LOGIT_WIDTH: usize = 5006;

struct Args {
    provider: Provider,
    iters: usize,
    tuning: CudaTuning,
    model_dir: String,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("decode-bench: {e}");
        std::process::exit(2);
    }
}

/// `vaani_core::Result` errors into a line of text.
///
/// `?` is used liberally below on tensor construction, and mapping the error
/// once here keeps every call site free of `.map_err`.
trait E<T> {
    fn e(self) -> Result<T, String>;
}

impl<T, E2: std::fmt::Display> E<T> for Result<T, E2> {
    fn e(self) -> Result<T, String> {
        self.map_err(|x| x.to_string())
    }
}

fn run() -> Result<(), String> {
    let mut provider = Provider::Auto;
    let mut iters = 2000usize;
    let mut cuda_graph = false;
    let mut exhaustive = false;
    let mut tf32 = true;
    let mut model_dir = "artifacts/sravaani".to_string();

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--provider" => {
                provider = Provider::parse(&it.next().ok_or("--provider needs a value")?)
                    .ok_or("unknown provider")?;
            }
            "--iters" => {
                iters = it
                    .next()
                    .ok_or("--iters needs a value")?
                    .parse()
                    .map_err(|e| format!("--iters: {e}"))?;
            }
            "--cuda-graph" => cuda_graph = true,
            "--exhaustive" => exhaustive = true,
            "--no-tf32" => tf32 = false,
            "--model-dir" => model_dir = it.next().ok_or("--model-dir needs a value")?,
            "-h" | "--help" => {
                println!(
                    "usage: decode-bench [--provider P] [--iters N] [--cuda-graph] \
                     [--exhaustive] [--no-tf32] [--model-dir DIR]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }

    let tuning = CudaTuning {
        cuda_graph,
        tf32,
        conv_search: if exhaustive {
            ConvSearch::Exhaustive
        } else {
            ConvSearch::Heuristic
        },
        conv_max_workspace: false,
    };
    let args = Args {
        provider,
        iters,
        tuning,
        model_dir,
    };

    let path = std::path::PathBuf::from(&args.model_dir).join("decoder_joint.onnx");
    if !path.is_file() {
        return Err(format!("{} not found", path.display()));
    }

    let caps = vaani_inference::capability::detect(&[]);
    let t0 = Instant::now();
    let mut session = OnnxSession::load_tuned(&path, args.provider, &caps, None, args.tuning)
        .map_err(|e| format!("load: {}", e.debug_chain()))?;
    let load = t0.elapsed();
    println!("model      {}", path.display());
    println!("provider   {:?}", session.provider_status().active.as_str());
    println!("tuning     {:?}", args.tuning);
    println!("load       {:.2} s", load.as_secs_f64());
    println!();

    let f = F32Tensor::new(vec![1, D_MODEL as i64, 1], vec![0.01f32; D_MODEL]).e()?;
    let tgt = I32Tensor::new(vec![1, 1], vec![0i32]).e()?;
    let tlen = I32Tensor::new(vec![1], vec![1i32]).e()?;
    let h = F32Tensor::new(vec![1, 1, STATE as i64], vec![0f32; STATE]).e()?;
    let c = F32Tensor::new(vec![1, 1, STATE as i64], vec![0f32; STATE]).e()?;

    // Warm-up: CUDA module loading, cuDNN handle creation, the first plan. A
    // benchmark that includes these reports a cold start as a steady-state cost.
    for _ in 0..20 {
        let _ = session
            .run(
                &[
                    Input::f32("f", f.clone()),
                    Input::i32("tgt", tgt.clone()),
                    Input::i32("tlen", tlen.clone()),
                    Input::f32("h_in", h.clone()),
                    Input::f32("c_in", c.clone()),
                ],
                &["logits", "h_out", "c_out"],
            )
            .e()?;
    }

    let mut best = Duration::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        for _ in 0..args.iters {
            let out = session
                .run(
                    &[
                        Input::f32("f", f.clone()),
                        Input::i32("tgt", tgt.clone()),
                        Input::i32("tlen", tlen.clone()),
                        Input::f32("h_in", h.clone()),
                        Input::f32("c_in", c.clone()),
                    ],
                    &["logits", "h_out", "c_out"],
                )
                .e()?;
            // Touch the logits so the work cannot be optimised away and the
            // argmax a real decoder would do is represented.
            std::hint::black_box(&out[0]);
        }
        let e = t.elapsed();
        if e < best {
            best = e;
        }
    }
    let per = best.as_secs_f64() / args.iters as f64;
    println!("iters      {}", args.iters);
    println!("best       {:.2} ms total", best.as_secs_f64() * 1000.0);
    println!("per call   {:.1} us", per * 1e6);
    println!(
        "logits     {LOGIT_WIDTH} floats returned per call ({} KiB)",
        LOGIT_WIDTH * 4 / 1024
    );
    println!();
    println!(
        "At 8.6 steps per second of speech, this is {:.1} s per hour of audio.",
        per * 8.6 * 3600.0
    );
    Ok(())
}

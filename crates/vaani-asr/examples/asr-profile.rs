//! SraVaani performance profile: where the time actually goes, on the device
//! that is actually present.
//!
//! The pipeline's end-to-end number is an average over four very different
//! costs, and an average hides the one that matters. This harness reports the
//! split the engine already measures - front end, encoder, decode - plus a
//! per-emission-step cost for the decode loop, because that loop runs a small
//! ONNX call once per emitted token and is the part most likely to be paying
//! launch overhead rather than arithmetic.
//!
//! Usage:
//!     cargo run --release --example asr-profile -- <wav> [--provider auto|cpu|cuda] [--repeat N]
//!
//! Deliberately an example rather than a bin: it is a development instrument,
//! it must not end up in a release, and it is not part of the benchmark
//! contract that `vaani-bench` publishes.

use std::path::PathBuf;
use std::time::Instant;

use vaani_asr::engine::AudioChunk;
use vaani_asr::{EngineRegistry, artifacts_in};
use vaani_core::{CancelToken, Time};
use vaani_inference::Provider;

struct Args {
    input: PathBuf,
    provider: Provider,
    repeat: usize,
    per_chunk: bool,
    fixed_chunk_s: Option<f64>,
    chunk_target_ms: Option<i64>,
    ab: bool,
    batch: bool,
}

fn parse() -> Result<Args, String> {
    let mut input = None;
    let mut provider = Provider::Auto;
    let mut repeat = 1usize;
    let mut per_chunk = false;
    let mut fixed_chunk_s: Option<f64> = None;
    let mut chunk_target_ms: Option<i64> = None;
    let mut ab = false;
    let mut batch = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--provider" => {
                let v = it.next().ok_or("--provider needs a value")?;
                provider = Provider::parse(&v).ok_or_else(|| format!("unknown provider {v}"))?;
            }
            "--per-chunk" => per_chunk = true,
            "--ab" => ab = true,
            // One `transcribe_batch` call for the whole job, which is what
            // production does. Without this flag the profiler measures the
            // serial per-chunk loop and cannot see the front-end change at all.
            "--batch" => batch = true,
            "--chunk-target-ms" => {
                chunk_target_ms = Some(
                    it.next()
                        .ok_or("--chunk-target-ms needs a value")?
                        .parse()
                        .map_err(|e| format!("--chunk-target-ms: {e}"))?,
                );
            }
            "--fixed-chunk-s" => {
                fixed_chunk_s = Some(
                    it.next()
                        .ok_or("--fixed-chunk-s needs a value")?
                        .parse()
                        .map_err(|e| format!("--fixed-chunk-s: {e}"))?,
                );
            }
            "--repeat" => {
                repeat = it
                    .next()
                    .ok_or("--repeat needs a value")?
                    .parse()
                    .map_err(|e| format!("--repeat: {e}"))?;
            }
            "-h" | "--help" => {
                println!("usage: asr-profile <wav> [--provider auto|cpu|cuda] [--repeat N]");
                std::process::exit(0);
            }
            other if other.starts_with('-') => return Err(format!("unknown flag {other}")),
            other => {
                if input.is_some() {
                    return Err("expected one input file".into());
                }
                input = Some(PathBuf::from(other));
            }
        }
    }
    Ok(Args {
        input: input.ok_or("missing input file")?,
        provider,
        repeat: repeat.max(1),
        per_chunk,
        fixed_chunk_s,
        chunk_target_ms,
        ab,
        batch,
    })
}

/// Read a 16-bit PCM WAV far enough to get mono f32 at 16 kHz.
///
/// Deliberately hand-rolled and deliberately narrow: this is a profiler, and it
/// must not be the thing that breaks. Anything it cannot read it says so.
fn read_wav16k_mono(path: &std::path::Path) -> Result<(Vec<f32>, u32), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let mut pos = 12usize;
    let (mut fmt, mut data) = (None::<(u16, u16, u32, u16)>, None::<(usize, usize)>);
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = pos + 8;
        if id == b"fmt " && body + 16 <= bytes.len() {
            fmt = Some((
                u16::from_le_bytes(bytes[body..body + 2].try_into().unwrap()),
                u16::from_le_bytes(bytes[body + 2..body + 4].try_into().unwrap()),
                u32::from_le_bytes(bytes[body + 4..body + 8].try_into().unwrap()),
                u16::from_le_bytes(bytes[body + 14..body + 16].try_into().unwrap()),
            ));
        } else if id == b"data" {
            data = Some((body, size.min(bytes.len().saturating_sub(body))));
        }
        pos = body + size + (size & 1);
    }
    let (tag, channels, rate, bits) = fmt.ok_or("no fmt chunk")?;
    let (dstart, dlen) = data.ok_or("no data chunk")?;
    if channels == 0 {
        return Err("a zero-channel WAV".into());
    }
    let raw = &bytes[dstart..dstart + dlen];
    let frame_bytes = (bits / 8) as usize * channels as usize;
    if frame_bytes == 0 {
        return Err(format!("unsupported bit depth {bits}"));
    }
    let frames = raw.len() / frame_bytes;
    let mut mono = Vec::with_capacity(frames);
    for i in 0..frames {
        let mut acc = 0f32;
        for c in 0..channels as usize {
            let o = i * frame_bytes + c * (bits / 8) as usize;
            let v = match (tag, bits) {
                // 16-bit int, the format every fixture in this repo used to be
                (1, 16) => {
                    f32::from(i16::from_le_bytes(raw[o..o + 2].try_into().unwrap())) / 32768.0
                }
                // 32-bit int
                (1, 32) => {
                    i32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) as f32 / 2_147_483_648.0
                }
                // 32-bit IEEE float, which is what the current fixtures are
                (3, 32) => f32::from_le_bytes(raw[o..o + 4].try_into().unwrap()),
                (t, b) => return Err(format!("unsupported PCM tag={t} bits={b}")),
            };
            acc += v;
        }
        mono.push(acc / channels as f32);
    }
    Ok((mono, rate))
}

/// Linear resample to 16 kHz, which is what the front end wants.
fn resample_to_16k(x: &[f32], from: u32) -> Vec<f32> {
    if from == 16_000 {
        return x.to_vec();
    }
    let ratio = f64::from(from) / 16_000.0;
    let out_len = (x.len() as f64 / ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let t = i as f64 * ratio;
        let i0 = t.floor() as usize;
        let frac = (t - i0 as f64) as f32;
        let a = x.get(i0).copied().unwrap_or(0.0);
        let b = x.get(i0 + 1).copied().unwrap_or(0.0);
        out.push(a + (b - a) * frac);
    }
    out
}

fn main() {
    if let Err(e) = run() {
        eprintln!("asr-profile: {e}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args = parse()?;

    let dir = std::env::var("VAANI_WEIGHTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("artifacts/sravaani"));
    if !dir.join("encoder.fp16.onnx").is_file() {
        return Err(format!(
            "weights not found in {}; set VAANI_WEIGHTS_DIR",
            dir.display()
        ));
    }

    let (raw, rate) = read_wav16k_mono(&args.input)?;
    let samples = resample_to_16k(&raw, rate);
    let audio_s = samples.len() as f64 / 16_000.0;
    println!("input      {}", args.input.display());
    println!(
        "audio      {:.2} s  ({} samples @ {} Hz -> 16 kHz mono)",
        audio_s,
        samples.len(),
        rate
    );

    let t_load = Instant::now();
    let caps = vaani_inference::capability::detect(&[]);
    let registry = EngineRegistry::with_builtins(args.provider, &caps)
        .map_err(|e| format!("registry: {}", e.debug_chain()))?;
    let art = artifacts_in(&dir);
    let engine = registry
        .create("sravaani", &art)
        .map_err(|e| format!("engine: {}", e.debug_chain()))?;
    let load_s = t_load.elapsed().as_secs_f64();
    println!("load       {load_s:.2} s");
    println!();

    let cancel = CancelToken::new();
    let mut best: Option<(f64, u64, u64, u64, usize, usize)> = None;

    // Production chunking, not one giant call. The encoder's activation memory
    // scales with the chunk's length, so an 834-second "chunk" is not a
    // faster way to measure the same work - it is a different workload that
    // runs out of VRAM. The chunker also decides how many decode steps happen,
    // which is the number the whole profile turns on.
    let buffer = vaani_core::media::AudioBuffer::mono(samples.clone(), 16_000, Time::ZERO);
    let window = vaani_core::MediaWindow::of_buffer(&buffer);
    let t_vad = Instant::now();
    // The energy detector, because that is what the product actually runs: the
    // neural VAD is only used when a model is present *and* the caller asks, and
    // no VAD model ships. A profile taken with a different chunker is a profile
    // of a different application.
    let vad = vaani_vad::neural::preferred_engine(
        None,
        false,
        args.provider,
        vaani_vad::EnergyVadConfig::default(),
    )
    .map_err(|e| format!("vad: {}", e.debug_chain()))?;
    let timeline = vad
        .detect(&buffer, &cancel)
        .map_err(|e| format!("vad: {}", e.debug_chain()))?;
    let mut policy = vaani_vad::ChunkPolicy::default();
    if let Some(ms) = args.chunk_target_ms {
        policy.target_ms = ms;
    }
    let chunks = vaani_vad::Chunker::new(policy).plan_within(&timeline, &window);
    println!(
        "vad        {:.2}s  speech {:.1}s of {:.1}s  -> {} chunks",
        t_vad.elapsed().as_secs_f64(),
        timeline.speech_duration_ms as f64 / 1000.0,
        audio_s,
        chunks.len()
    );

    // The experiment. Production chunk lengths are all slightly different, and
    // the encoder graph takes a *dynamic* time dimension, so every one of those
    // lengths is a shape ONNX Runtime has never planned before: a fresh
    // execution plan, a fresh cuDNN algorithm choice, and then a fraction of
    // that time actually spent on the audio. Feeding identical lengths instead
    // isolates that fixed cost, and the difference between the two runs is what
    // making the shape static is worth.
    let chunks: Vec<vaani_vad::Chunk> = match args.fixed_chunk_s {
        None => chunks,
        Some(secs) => {
            let step = vaani_core::MediaLength::from_secs_f64(secs);
            let mut v = Vec::new();
            let mut at = Time::ZERO;
            while at < window.end() {
                let end = if at.offset_by(step) < window.end() {
                    at.offset_by(step)
                } else {
                    window.end()
                };
                v.push(vaani_vad::Chunk { start: at, end });
                at = end;
            }
            println!(
                "fixed-chunk {secs}s  -> {} chunks, all the same length",
                v.len()
            );
            v
        }
    };
    println!();

    if args.ab {
        return ab_mode(&args, &chunks, &buffer);
    }

    for r in 1..=args.repeat {
        let (mut enc_total, mut dec_total, mut front_total) = (0u64, 0u64, 0u64);
        let (mut steps_total, mut words_total) = (0usize, 0usize);
        let mut fingerprint: u64 = 1469598103934665603;
        let t0 = Instant::now();

        // Build the chunk list once, outside the timing loop. Slicing the buffer
        // is memcpy, not speech recognition, and leaving it inside made the
        // front-end figure include work the engine does not do.
        let inputs: Vec<AudioChunk> = chunks
            .iter()
            .map(|c| {
                let slice = buffer.slice_mono(c.start, c.end);
                AudioChunk::new(slice, 16_000, c.start)
            })
            .filter(|c| !c.samples.is_empty())
            .collect();

        if args.batch {
            // The production path: the batched encoder with a per-chunk sink,
            // which is exactly what the ASR stage does.
            let mut results: Vec<std::result::Result<(), String>> =
                Vec::with_capacity(inputs.len());
            engine.transcribe_streaming(&inputs, &cancel, &mut |i, result| {
                let res = result;
                if args.per_chunk {
                    let c = &chunks[i];
                    println!(
                        "   chunk {:>7.1}s  frames {:6}  enc {:6}ms  dec {:6}ms  steps {:5}  words {:4}",
                        (c.end.as_millis() - c.start.as_millis()) as f64 / 1000.0,
                        ((c.end.as_millis() - c.start.as_millis()) as f64 / 10.0).round() as i64,
                        res.timings.encode_ms,
                        res.timings.decode_ms,
                        res.timings.decode_steps,
                        res.segments.iter().map(|s| s.words.len()).sum::<usize>(),
                    );
                }
                enc_total += res.timings.encode_ms;
                dec_total += res.timings.decode_ms;
                front_total += res.timings.frontend_ms;
                steps_total += res.timings.decode_steps;
                words_total += res.segments.iter().map(|s| s.words.len()).sum::<usize>();
                for w in res.segments.iter().flat_map(|s| s.words.iter()) {
                    for b in w.text.as_bytes() {
                        fingerprint ^= u64::from(*b);
                        fingerprint = fingerprint.wrapping_mul(1099511628211);
                    }
                }
                results.push(Ok(()));
                true
            });
        } else {
            for c in &chunks {
                let slice = buffer.slice_mono(c.start, c.end);
                if slice.is_empty() {
                    continue;
                }
                let chunk = AudioChunk::new(slice, 16_000, c.start);
                let res = engine
                    .transcribe(&chunk, &cancel)
                    .map_err(|e| format!("transcribe: {}", e.debug_chain()))?;
                if args.per_chunk {
                    println!(
                        "   chunk {:>7.1}s  frames {:6}  enc {:6}ms  dec {:6}ms  steps {:5}  words {:4}",
                        (c.end.as_millis() - c.start.as_millis()) as f64 / 1000.0,
                        ((c.end.as_millis() - c.start.as_millis()) as f64 / 10.0).round() as i64,
                        res.timings.encode_ms,
                        res.timings.decode_ms,
                        res.timings.decode_steps,
                        res.segments.iter().map(|s| s.words.len()).sum::<usize>(),
                    );
                }
                enc_total += res.timings.encode_ms;
                dec_total += res.timings.decode_ms;
                front_total += res.timings.frontend_ms;
                steps_total += res.timings.decode_steps;
                words_total += res.segments.iter().map(|s| s.words.len()).sum::<usize>();
                for w in res.segments.iter().flat_map(|s| s.words.iter()) {
                    for b in w.text.as_bytes() {
                        fingerprint ^= u64::from(*b);
                        fingerprint = fingerprint.wrapping_mul(1099511628211);
                    }
                }
            }
        }
        let wall = t0.elapsed().as_secs_f64();
        let rtf = wall / audio_s;
        let per_step = if steps_total > 0 {
            dec_total as f64 * 1000.0 / steps_total as f64
        } else {
            0.0
        };
        let prov = engine
            .active_provider()
            .map_or("unknown", vaani_inference::ActiveProvider::as_str);
        println!(
            "run {r:<3} wall {wall:7.2}s  rtf {rtf:6.4}  front {front_total:5}ms  enc {enc_total:5}ms  dec {dec_total:6}ms  steps {steps_total:5}  {words_total:5} words  {per_step:5.2}us/step  fp={fingerprint:016x}  on {prov}"
        );
        let cand = (
            rtf,
            front_total,
            enc_total,
            dec_total,
            steps_total,
            words_total,
        );
        if best.is_none_or(|b| cand.0 < b.0) {
            best = Some(cand);
        }
    }

    let (rtf, front, enc, dec, steps, words) = best.expect("at least one run");
    println!();
    println!("--- best of {} ---", args.repeat);
    println!(
        "rtf        {rtf:.4}  ({:.1}x realtime)",
        1.0 / rtf.max(f64::MIN_POSITIVE)
    );
    println!(
        "front end  {front} ms  ({:.1}%)",
        pct(front as f64, rtf * audio_s * 1000.0)
    );
    println!(
        "encoder    {enc} ms  ({:.1}%)",
        pct(enc as f64, rtf * audio_s * 1000.0)
    );
    println!(
        "decode     {dec} ms  ({:.1}%)",
        pct(dec as f64, rtf * audio_s * 1000.0)
    );
    if steps > 0 {
        println!(
            "decode     {steps} emission steps, {:.2} us each",
            dec as f64 * 1000.0 / steps as f64
        );
    }
    println!("words      {words}");
    Ok(())
}

fn pct(part: f64, whole: f64) -> f64 {
    if whole <= 0.0 {
        0.0
    } else {
        part / whole * 100.0
    }
}

/// Interleaved A/B of the encoder's frame bucket.
///
/// Two separate processes cannot be compared on this machine: the GPU is a
/// laptop part and drifts by around twenty percent between runs, which is the
/// same size as the effect being measured. So both settings run *here*, in one
/// process, alternating, and the medians are compared. Warm-up runs first for
/// each setting, because the first chunk pays cuDNN algorithm selection and
/// CUDA module loading, and that is not what is being compared.
fn ab_mode(
    args: &Args,
    chunks: &[vaani_vad::Chunk],
    buffer: &vaani_core::media::AudioBuffer,
) -> Result<(), String> {
    use vaani_asr::engine::AsrEngine as _;
    use vaani_asr::{SraVaaniEngine, artifacts_in, open_sravaani};

    let dir = std::env::var("VAANI_WEIGHTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("artifacts/sravaani"));
    let art = artifacts_in(&dir);
    let caps = vaani_inference::capability::detect(&[]);
    let engine: SraVaaniEngine = open_sravaani(&art, args.provider, &caps)
        .map_err(|e| format!("engine: {}", e.debug_chain()))?;
    let cancel = CancelToken::new();

    let mut enc: Vec<Vec<u64>> = vec![Vec::new(); BUCKETS.len()];
    let mut words: Vec<usize> = vec![0; BUCKETS.len()];

    let one_pass = |bucket: usize| -> Result<(u64, u64, u64, usize, usize), String> {
        engine.set_frame_bucket(bucket);
        let (mut e, mut d, mut f, mut st, mut w) = (0u64, 0u64, 0u64, 0usize, 0usize);
        for c in chunks {
            let slice = buffer.slice_mono(c.start, c.end);
            if slice.is_empty() {
                continue;
            }
            let chunk = AudioChunk::new(slice, 16_000, c.start);
            let res = engine
                .transcribe(&chunk, &cancel)
                .map_err(|x| format!("transcribe: {}", x.debug_chain()))?;
            e += res.timings.encode_ms;
            d += res.timings.decode_ms;
            f += res.timings.frontend_ms;
            st += res.timings.decode_steps;
            w += res.segments.iter().map(|s| s.words.len()).sum::<usize>();
        }
        Ok((e, d, f, st, w))
    };

    // Warm every path: the first pass through a shape pays plan construction,
    // and cuDNN algorithm selection is per shape too.
    for b in BUCKETS {
        one_pass(b)?;
    }

    println!(
        "alternating encoder-frame-bucket A/B, {} rounds, buckets {:?}",
        args.repeat, BUCKETS
    );
    for r in 1..=args.repeat {
        for (i, bucket) in BUCKETS.iter().enumerate() {
            let (e, _d, _f, _st, w) = one_pass(*bucket)?;
            enc[i].push(e);
            words[i] = w;
            println!("  round {r} bucket {bucket:>5}: enc {e:6} ms   words {w}");
        }
    }

    println!();
    println!(
        "{:<10} {:>10} {:>10} {:>10}  {:>8}  {:>7}",
        "bucket", "median", "min", "max", "words", "vs off"
    );
    let med = |v: &Vec<u64>| {
        let mut s = v.clone();
        s.sort_unstable();
        s[s.len() / 2] as f64
    };
    let base = med(&enc[0]);
    for (i, v) in enc.iter().enumerate() {
        let mut s = v.clone();
        s.sort_unstable();
        let m = s[s.len() / 2];
        let name = if BUCKETS[i] == 0 {
            "off".to_owned()
        } else {
            BUCKETS[i].to_string()
        };
        println!(
            "{:<10} {:>9} ms {:>9} ms {:>9} ms  {:>8}  {:>6.2}x",
            name,
            m,
            s[0],
            s[s.len() - 1],
            words[i],
            base / m as f64
        );
    }
    Ok(())
}

/// Bucket sizes compared, in frames. `0` is the unbucketed baseline.
///
/// Deliberately stops at 2048. At 4096 the encoder's padded activation asks the
/// CUDA arena for 134 MB, and on a 4 GB card - with the earlier shapes' plans
/// already resident in the same arena - only ~112 MB is left, so the run aborts
/// with a BFC arena allocation failure. That is the ceiling this knob has, and
/// it is a good reason for the product default to sit well below it.
const BUCKETS: [usize; 4] = [0, 512, 1024, 2048];

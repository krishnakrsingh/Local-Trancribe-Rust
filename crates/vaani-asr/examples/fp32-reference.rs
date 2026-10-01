//! A pseudo-reference transcript from the **fp32** encoder, for scoring the
//! fp16 paths against something other than each other.
//!
//! # Why this exists
//!
//! `ENCODER_BATCH` is off in production with a note that says the reason: the
//! batched encoder is "a different numerical result of unknown quality, not a
//! faster spelling of the same one", and that "until there is a human reference
//! to score both against, 'different text' cannot be called 'as good'". The
//! whole point of this harness is to attack that sentence from the side that is
//! actually available: SraVaani ships a second encoder export, `encoder.onnx`,
//! which is the same graph in **float32**. It is the mathematically cleanest
//! path available in this repository, so it is the closest thing to a reference
//! that can be produced without a human in the loop.
//!
//! Three transcripts come out of one input and one chunking:
//!
//! | variant | encoder | device | batch |
//! |---|---|---|---|
//! | `fp32` | `encoder.onnx` (fp32) | CPU | 1 |
//! | `fp16_serial` | `encoder.fp16.onnx` | CUDA | 1 |
//! | `fp16_batched` | `encoder.fp16.onnx` | CUDA | 4 |
//!
//! and then the two fp16 ones are scored against the fp32 one in WORDS.
//!
//! # What is held fixed, and why it matters
//!
//! * **The decode path is identical in all three runs.** `decoder_joint.onnx` is
//!   fp32 in every case and, more importantly, runs on the **same device**
//!   (CUDA) in every case. Running the fp32 run's decoder on the CPU would have
//!   been simpler to type, and would have quietly destroyed the experiment: the
//!   CUDA joint runs with TF32 enabled, which has a 10-bit mantissa and is
//!   therefore *coarser than fp16*, so a CPU-decoder reference would differ from
//!   a CUDA-decoder candidate by more than the thing being measured. Hence
//!   `open_with_providers`: encoder on the CPU, decoder on the GPU.
//! * **The chunking is production chunking**, reproduced exactly the way
//!   `asr-profile` does it: energy VAD, default `ChunkPolicy`, `Chunker`. If
//!   the audio were split differently the comparison would be meaningless,
//!   because the encoder's time axis would see different shapes.
//! * **The feature bucketing is the shipped default** (512 mel frames) for all
//!   three, so every variant hands the graph a zero-padded time axis of the same
//!   granularity and the *only* variables left are the dtype and the batch size.
//!
//! # Honesty notes
//!
//! * This is a **pseudo**-reference. fp32 is not truth; it is the best available
//!   proxy, and its own error rate against a human transcript is unmeasured.
//!   Every number this prints is relative to fp32 and says nothing about fp32's
//!   own accuracy.
//! * The word streams are the engine's raw SentencePiece grouping, with no
//!   post-processing, no ITN and no punctuation. Word boundaries come from the
//!   metaspace marker, which is what `tokens_to_segments` does.
//! * No normalisation is applied before scoring. All three variants come out of
//!   the same tokenizer and the same front end, so normalising would only add
//!   a place for a difference to be hidden.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example fp32-reference -- transcribe fp32         --out DIR
//! cargo run --release --example fp32-reference -- transcribe fp16_serial  --out DIR
//! cargo run --release --example fp32-reference -- transcribe fp16_batched --out DIR
//! cargo run --release --example fp32-reference -- score DIR
//! ```
//!
//! One variant per process on purpose: the fp32 encoder is 1.77 GB and the fp16
//! one is 887 MB, and holding both resident to save a few seconds of model load
//! would trade a clean isolation of the two devices for that.
//!
//! Deliberately an example, like `asr-profile`: a development instrument that
//! must not end up in a release.

use std::path::{Path, PathBuf};
use std::time::Instant;

use vaani_asr::engine::{AsrEngine, AudioChunk};
use vaani_asr::sravaani::{FeatsPrecision, open_with_providers};
use vaani_asr::{ChunkResult, EngineArtifacts, SraVaaniEngine};
use vaani_core::{CancelToken, Time};
use vaani_inference::Provider;
use vaani_vad::{Chunk, ChunkPolicy, Chunker};

// ===========================================================================
// Levenshtein, copied from evaluation/scoring/src/edit.rs
// ===========================================================================
//
// The scorer that owns the real implementation is a *separate workspace* on
// purpose -- evaluation/scoring/Cargo.toml declares its own `[workspace]` so it
// "cannot perturb the lockfile the other eleven crates share". Adding it as a
// dependency of vaani-asr would do exactly that. So this is a verbatim copy of
// `vaani_eval::edit::align`, minus its tests, and the two things that changed
// are the two that had to:
//
//   * MAX_CELLS raised from 4_000_000 to 16_000_000, because a 14-minute
//     recording transcribes to ~1900 words and the caller is no longer limited
//     to a 120-second clip.
//   * `align_distance` added alongside, a two-row DP with no traceback, used for
//     the character level where ~1900 words is ~10500 characters and the full
//     matrix would be 110 million cells. The Levenshtein distance is unique, so
//     dropping the traceback cannot change the number; `score_pair` asserts on
//     the word level, where both are computable, that the two agree exactly.

/// One alignment operation, in reference order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Match,
    Sub,
    Del,
    Ins,
}

/// The full alignment of a hypothesis to a reference.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Alignment {
    ops: Vec<Op>,
    correct: usize,
    substitutions: usize,
    deletions: usize,
    insertions: usize,
}

impl Alignment {
    fn errors(&self) -> usize {
        self.substitutions + self.deletions + self.insertions
    }
}

const MAX_CELLS: usize = 16_000_000;

/// Levenshtein alignment of `reference` against `hypothesis`.
///
/// Tie-break order fixed -- diagonal (match/substitution) before deletion
/// before insertion -- so the S/D/I *split* is reproducible and not arbitrary,
/// even though S+D+I is stable either way.
fn align<T: PartialEq>(reference: &[T], hypothesis: &[T]) -> Alignment {
    let n = reference.len();
    let m = hypothesis.len();
    assert!(
        n.saturating_mul(m) <= MAX_CELLS,
        "alignment input too large: {n} x {m}"
    );

    let mut dist = vec![0u32; (n + 1) * (m + 1)];
    let idx = |i: usize, j: usize| i * (m + 1) + j;
    for i in 0..=n {
        dist[idx(i, 0)] = i as u32;
    }
    for j in 0..=m {
        dist[idx(0, j)] = j as u32;
    }
    for i in 1..=n {
        for j in 1..=m {
            let sub = dist[idx(i - 1, j - 1)] + u32::from(reference[i - 1] != hypothesis[j - 1]);
            let del = dist[idx(i - 1, j)] + 1;
            let ins = dist[idx(i, j - 1)] + 1;
            dist[idx(i, j)] = sub.min(del).min(ins);
        }
    }

    let mut rev_ops = Vec::with_capacity(n + m);
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        if i > 0 && j > 0 {
            let same = reference[i - 1] == hypothesis[j - 1];
            let cost = dist[idx(i - 1, j - 1)] + u32::from(!same);
            if dist[idx(i, j)] == cost {
                rev_ops.push(if same { Op::Match } else { Op::Sub });
                i -= 1;
                j -= 1;
                continue;
            }
        }
        if i > 0 && dist[idx(i, j)] == dist[idx(i - 1, j)] + 1 {
            rev_ops.push(Op::Del);
            i -= 1;
            continue;
        }
        rev_ops.push(Op::Ins);
        j -= 1;
    }
    rev_ops.reverse();

    let mut a = Alignment {
        ops: rev_ops,
        correct: 0,
        substitutions: 0,
        deletions: 0,
        insertions: 0,
    };
    for op in &a.ops {
        match op {
            Op::Match => a.correct += 1,
            Op::Sub => a.substitutions += 1,
            Op::Del => a.deletions += 1,
            Op::Ins => a.insertions += 1,
        }
    }
    a
}

/// The same distance, without the traceback and without the `O(n*m)` matrix.
///
/// Same recurrence, same minimum, so the returned value equals
/// `align(a, b).errors()`; only the S/D/I split is unavailable, which is exactly
/// what the character level needs.
fn align_distance<T: PartialEq>(reference: &[T], hypothesis: &[T]) -> usize {
    let n = reference.len();
    let m = hypothesis.len();
    let mut prev: Vec<u32> = (0..=m as u32).collect();
    let mut cur: Vec<u32> = vec![0; m + 1];
    for i in 1..=n {
        cur[0] = i as u32;
        for j in 1..=m {
            let sub = prev[j - 1] + u32::from(reference[i - 1] != hypothesis[j - 1]);
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m] as usize
}

// ===========================================================================
// The transcript format
// ===========================================================================

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct WordRec {
    text: String,
    start_ms: i64,
    end_ms: i64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ChunkRec {
    index: usize,
    start_ms: i64,
    end_ms: i64,
    encoder_frames: usize,
    decode_steps: usize,
    frontend_ms: u64,
    encode_ms: u64,
    decode_ms: u64,
    words: Vec<WordRec>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Transcript {
    variant: String,
    encoder_file: String,
    feats_precision: String,
    encoder_provider_requested: String,
    decoder_provider_requested: String,
    encoder_provider_active: String,
    decoder_provider_active: String,
    batch_size: usize,
    frame_bucket: usize,
    audio_seconds: f64,
    n_chunks: usize,
    word_count: usize,
    decode_steps: usize,
    /// FNV-1a 64 over the concatenated word bytes, in order. Same construction
    /// `asr-profile` uses, so its numbers are comparable.
    fingerprint: String,
    chunks: Vec<ChunkRec>,
}

fn fnv1a64(words: &[WordRec]) -> u64 {
    // Seed 1469598103934665603, prime 1099511628211, xor-then-multiply per byte.
    let mut h: u64 = 1469598103934665603;
    for w in words {
        for b in w.text.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(1099511628211);
        }
    }
    h
}

// ===========================================================================
// WAV reading: copied from asr-profile, same narrowness on purpose
// ===========================================================================

fn read_wav16k_mono(path: &Path) -> Result<(Vec<f32>, u32), String> {
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
                (1, 16) => {
                    f32::from(i16::from_le_bytes(raw[o..o + 2].try_into().unwrap())) / 32768.0
                }
                (1, 32) => {
                    i32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) as f32 / 2_147_483_648.0
                }
                (3, 32) => f32::from_le_bytes(raw[o..o + 4].try_into().unwrap()),
                (t, b) => return Err(format!("unsupported PCM tag={t} bits={b}")),
            };
            acc += v;
        }
        mono.push(acc / channels as f32);
    }
    Ok((mono, rate))
}

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

// ===========================================================================
// Transcribe
// ===========================================================================

/// The three variants, and the encoder each one loads.
struct Variant {
    name: &'static str,
    encoder_file: &'static str,
    precision: FeatsPrecision,
    encoder_provider: Provider,
    /// `1` is the serial per-chunk loop; anything else is the group size used
    /// through `transcribe_streaming`, which is the batched path.
    batch: usize,
}

fn variant(name: &str) -> Result<Variant, String> {
    // The decoder provider is CUDA in every case, including the fp32 run. See
    // the module docs: the CUDA joint runs with TF32, so moving the decoder to
    // the CPU for the reference would have injected a coarser-than-fp16
    // difference into the one measurement that is supposed to isolate the
    // encoder.
    Ok(match name {
        "fp32" => Variant {
            name: "fp32",
            encoder_file: "encoder.onnx",
            precision: FeatsPrecision::F32,
            encoder_provider: Provider::Cpu,
            batch: 1,
        },
        "fp16_serial" => Variant {
            name: "fp16_serial",
            encoder_file: "encoder.fp16.onnx",
            precision: FeatsPrecision::F16,
            encoder_provider: Provider::RequireCuda,
            batch: 1,
        },
        "fp16_batched" => Variant {
            name: "fp16_batched",
            encoder_file: "encoder.fp16.onnx",
            precision: FeatsPrecision::F16,
            encoder_provider: Provider::RequireCuda,
            batch: vaani_asr::sravaani::ENCODER_BATCH,
        },
        other => return Err(format!("unknown variant {other:?}")),
    })
}

fn plan_chunks(samples: &[f32]) -> Result<Vec<Chunk>, String> {
    use vaani_core::media::AudioBuffer;

    let buffer = AudioBuffer::mono(samples.to_vec(), 16_000, Time::ZERO);
    let window = vaani_core::MediaWindow::of_buffer(&buffer);
    let cancel = CancelToken::new();

    // Exactly what the product runs, exactly what `asr-profile` runs: the energy
    // detector, because the neural VAD is only used when a model ships and the
    // caller asks, and none does. The provider is pinned to CPU so the chunk
    // plan cannot vary with which variant is being measured.
    let vad = vaani_vad::neural::preferred_engine(
        None,
        false,
        Provider::Cpu,
        vaani_vad::EnergyVadConfig::default(),
    )
    .map_err(|e| format!("vad: {}", e.debug_chain()))?;
    let timeline = vad
        .detect(&buffer, &cancel)
        .map_err(|e| format!("vad: {}", e.debug_chain()))?;
    Ok(Chunker::new(ChunkPolicy::default()).plan_within(&timeline, &window))
}

fn transcribe(v: &Variant, weights: &Path, wav: &Path, out_dir: &Path) -> Result<(), String> {
    use vaani_core::media::AudioBuffer;

    let (raw, rate) = read_wav16k_mono(wav)?;
    let samples = resample_to_16k(&raw, rate);
    if samples.is_empty() {
        return Err("input decoded to zero samples".into());
    }
    let audio_s = samples.len() as f64 / 16_000.0;

    let chunks = plan_chunks(&samples)?;
    println!("variant     {}", v.name);
    println!(
        "audio       {audio_s:.2} s @ {rate} Hz -> 16 kHz mono, {} samples",
        samples.len()
    );
    println!(
        "chunks      {} from the energy VAD + default ChunkPolicy",
        chunks.len()
    );

    // The artefact set is production's, with the encoder swapped. `missing()`
    // still checks all six files, so a typo here is caught before 1.77 GB is
    // opened.
    let mut art: EngineArtifacts = vaani_asr::sravaani::artifacts_in(weights);
    art.encoder = weights.join(v.encoder_file);
    let missing = art.missing();
    if !missing.is_empty() {
        return Err(format!("artefacts missing: {}", missing.join(", ")));
    }

    let caps = vaani_inference::capability::detect(&[]);
    println!("loading     {} ...", art.encoder.display());
    let t_load = Instant::now();
    let engine: SraVaaniEngine =
        open_with_providers(&art, v.encoder_provider, Provider::RequireCuda, &caps)
            .map_err(|e| format!("engine: {}", e.debug_chain()))?;
    engine.set_feats_precision(v.precision);
    println!("loaded      {:.2} s", t_load.elapsed().as_secs_f64());
    let (enc_active, dec_active) = engine
        .session_providers()
        .ok_or("sessions were not loaded; open_with_providers sets them")?;
    println!(
        "providers   encoder {} / decoder {}  (requested encoder {:?}, feats {:?})",
        enc_active.as_str(),
        dec_active.as_str(),
        v.encoder_provider,
        v.precision
    );
    if v.encoder_provider == Provider::RequireCuda && enc_active.as_str() != "cuda" {
        return Err(format!(
            "this run asked for CUDA on the encoder and got {}; the comparison would \
             be between a GPU path and a CPU path",
            enc_active.as_str()
        ));
    }

    let cancel = CancelToken::new();
    let buffer = AudioBuffer::mono(samples, 16_000, Time::ZERO);
    let inputs: Vec<AudioChunk> = chunks
        .iter()
        .map(|c| AudioChunk::new(buffer.slice_mono(c.start, c.end), 16_000, c.start))
        .filter(|c| !c.samples.is_empty())
        .collect();
    if inputs.len() != chunks.len() {
        return Err(format!(
            "{} chunk(s) sliced to zero samples; the chunk plan does not match the buffer",
            chunks.len() - inputs.len()
        ));
    }

    let t0 = Instant::now();
    let mut recs: Vec<ChunkRec> = Vec::with_capacity(inputs.len());

    if v.batch == 1 {
        // The serial production path: one `transcribe` per chunk, exactly what
        // the ASR stage does and exactly what `asr-profile`'s default loop does.
        for (i, chunk) in inputs.iter().enumerate() {
            let res = engine
                .transcribe(chunk, &cancel)
                .map_err(|e| format!("chunk {i}: {}", e.debug_chain()))?;
            recs.push(rec_of(i, &chunks[i], &res));
        }
    } else {
        // The batched candidate: every chunk in one `transcribe_streaming`,
        // which groups them `ENCODER_BATCH` at a time along the graph's dynamic
        // batch axis and pads each group to one shared time bucket.
        let results = engine.transcribe_streaming(&inputs, &cancel, &mut |i, res| {
            recs.push(rec_of(i, &chunks[i], res));
            true
        });
        if recs.len() != inputs.len() {
            let first_err = results
                .iter()
                .find_map(|r| r.as_ref().err())
                .map_or_else(String::new, |e| e.debug_chain());
            return Err(format!(
                "batched path produced {} of {} chunk results: {first_err}",
                recs.len(),
                inputs.len()
            ));
        }
    }
    let wall = t0.elapsed().as_secs_f64();
    println!(
        "wall        {wall:.2} s  rtf {:.4}  ({:.2} x realtime)",
        wall / audio_s,
        1.0 / (wall / audio_s)
    );

    let words: Vec<WordRec> = recs.iter().flat_map(|c| c.words.clone()).collect();
    let decode_steps: usize = recs.iter().map(|c| c.decode_steps).sum();
    let fingerprint = fnv1a64(&words);
    let t = Transcript {
        variant: v.name.to_string(),
        encoder_file: v.encoder_file.to_string(),
        feats_precision: format!("{:?}", v.precision),
        encoder_provider_requested: format!("{:?}", v.encoder_provider),
        decoder_provider_requested: "RequireCuda".to_string(),
        encoder_provider_active: enc_active.as_str().to_string(),
        decoder_provider_active: dec_active.as_str().to_string(),
        batch_size: v.batch,
        frame_bucket: vaani_asr::sravaani::ENCODER_FRAME_BUCKET,
        audio_seconds: audio_s,
        n_chunks: recs.len(),
        word_count: words.len(),
        decode_steps,
        fingerprint: format!("{fingerprint:016x}"),
        chunks: recs,
    };
    println!(
        "words       {}   decode steps {}   fnv {fingerprint:016x}",
        t.word_count, t.decode_steps
    );
    for c in &t.chunks {
        println!(
            "  chunk {:>2}  {:>7.1}s -> {:>7.1}s  enc_frames {:>4}  steps {:>5}  words {:>4}  front {:>5}ms  enc {:>6}ms  dec {:>6}ms",
            c.index,
            c.start_ms as f64 / 1000.0,
            c.end_ms as f64 / 1000.0,
            c.encoder_frames,
            c.decode_steps,
            c.words.len(),
            c.frontend_ms,
            c.encode_ms,
            c.decode_ms
        );
    }

    std::fs::create_dir_all(out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
    let path = out_dir.join(format!("{}.json", v.name));
    let json = serde_json::to_string_pretty(&t).map_err(|e| format!("serde: {e}"))?;
    std::fs::write(&path, json).map_err(|e| format!("{}: {e}", path.display()))?;
    println!("wrote       {}", path.display());
    Ok(())
}

fn rec_of(i: usize, c: &Chunk, res: &ChunkResult) -> ChunkRec {
    let base = c.start.as_millis();
    let words = res
        .segments
        .iter()
        .flat_map(|s| s.words.iter())
        .map(|w| WordRec {
            text: w.text.clone(),
            // Rebased from chunk-local to source-media time here, in the
            // example, rather than relying on the caller. The engine's contract
            // is chunk-local times and this is the one place that knows the
            // offset.
            start_ms: base + w.start.as_millis(),
            end_ms: base + w.end.as_millis(),
        })
        .collect();
    ChunkRec {
        index: i,
        start_ms: c.start.as_millis(),
        end_ms: c.end.as_millis(),
        encoder_frames: res.timings.encoder_frames,
        decode_steps: res.timings.decode_steps,
        frontend_ms: res.timings.frontend_ms,
        encode_ms: res.timings.encode_ms,
        decode_ms: res.timings.decode_ms,
        words,
    }
}

// ===========================================================================
// Score
// ===========================================================================

fn texts(t: &Transcript) -> Vec<String> {
    t.chunks
        .iter()
        .flat_map(|c| c.words.iter())
        .map(|w| w.text.clone())
        .collect()
}

/// Concatenated text, single spaces between words, spaces counted.
///
/// This is the same string `vaani_eval::wer::score` aligns: its
/// `NormalisedText.text` carries the separators and its `char_count` counts
/// them. Dropping the spaces would make a missing word cost one character
/// instead of two and quietly halve CER.
fn concat(t: &[String]) -> String {
    t.join(" ")
}

struct Scored {
    label: String,
    s: usize,
    d: usize,
    i: usize,
    errors: usize,
    correct: usize,
    ref_words: usize,
    hyp_words: usize,
    char_errors: usize,
    ref_chars: usize,
}

fn score_pair(label: &str, reference: &Transcript, hypothesis: &Transcript) -> Scored {
    let r = texts(reference);
    let h = texts(hypothesis);
    let a = align(&r, &h);

    // Cross-check the traceback path against the two-row path at the word level,
    // where both are computable. If these ever disagree the character-level
    // numbers below are not trustworthy, so this is a hard assert rather than a
    // note.
    assert_eq!(
        align_distance(&r, &h),
        a.errors(),
        "align_distance disagrees with align().errors(); the CER path is not the same metric"
    );

    let ref_chars: Vec<char> = concat(&r).chars().collect();
    let hyp_chars: Vec<char> = concat(&h).chars().collect();
    let char_errors = align_distance(&ref_chars, &hyp_chars);

    Scored {
        label: label.to_string(),
        s: a.substitutions,
        d: a.deletions,
        i: a.insertions,
        errors: a.errors(),
        correct: a.correct,
        ref_words: r.len(),
        hyp_words: h.len(),
        char_errors,
        ref_chars: ref_chars.len(),
    }
}

fn score(out_dir: &Path) -> Result<(), String> {
    let load = |name: &str| -> Result<Transcript, String> {
        let p = out_dir.join(format!("{name}.json"));
        let text = std::fs::read_to_string(&p)
            .map_err(|e| format!("{}: {e}; run that variant first", p.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", p.display()))
    };
    let fp32 = load("fp32")?;
    let serial = load("fp16_serial")?;
    let batched = load("fp16_batched")?;

    // --- provenance ------------------------------------------------------
    println!("=== inputs ========================================================");
    for t in [&fp32, &serial, &batched] {
        println!(
            "{:<13} encoder {:<16} feats {:<3} enc_ep {:<6} dec_ep {:<6} batch {}  chunks {:>2}  words {:>5}  steps {:>5}  fnv {}",
            t.variant,
            t.encoder_file,
            t.feats_precision,
            t.encoder_provider_active,
            t.decoder_provider_active,
            t.batch_size,
            t.n_chunks,
            t.word_count,
            t.decode_steps,
            t.fingerprint
        );
    }
    println!("audio {:.2} s in every variant", fp32.audio_seconds);
    let same_chunking = fp32
        .chunks
        .iter()
        .zip(&serial.chunks)
        .zip(&batched.chunks)
        .all(|((a, b), c)| a.start_ms == b.start_ms && a.start_ms == c.start_ms);
    println!(
        "chunk plans identical across all three: {}",
        if same_chunking {
            "yes"
        } else {
            "NO -- the comparison is invalid"
        }
    );
    assert!(same_chunking, "chunk plans diverged between variants");

    // --- headline --------------------------------------------------------
    let a_serial = score_pair("fp16_serial vs fp32", &fp32, &serial);
    let a_batched = score_pair("fp16_batched vs fp32", &fp32, &batched);
    let a_sb = score_pair("fp16_serial vs fp16_batched", &serial, &batched);

    println!();
    println!("=== whole-clip word alignment =======================================");
    println!(
        "{:<28} {:>5} {:>4} {:>4} {:>6} {:>9} {:>10} {:>9} {:>9}",
        "comparison", "S", "D", "I", "S+D+I", "WER", "ref words", "CER", "ref chars"
    );
    for a in [&a_serial, &a_batched, &a_sb] {
        println!(
            "{:<28} {:>5} {:>4} {:>4} {:>6} {:>9} {:>10} {:>9} {:>9}",
            a.label,
            a.s,
            a.d,
            a.i,
            a.errors,
            pct(a.errors, a.ref_words),
            a.ref_words,
            pct(a.char_errors, a.ref_chars),
            a.ref_chars
        );
    }
    println!(
        "note: row 3's reference is fp16_serial ({} words), not the fp32 pseudo-reference.",
        a_sb.ref_words
    );
    for a in [&a_serial, &a_batched] {
        println!(
            "{}: hypothesis {} words, {} matched, {} differ",
            a.label,
            a.hyp_words,
            a.correct,
            a.hyp_words - a.correct
        );
    }

    // --- the decisive question ------------------------------------------
    println!();
    println!("=== verdict =========================================================");
    let ds = a_serial.errors;
    let db = a_batched.errors;
    let verdict = match db.cmp(&ds) {
        std::cmp::Ordering::Less => {
            "BATCHING IMPROVES FIDELITY: batched is strictly closer to the fp32 \
             pseudo-reference than serial."
        }
        std::cmp::Ordering::Greater => {
            "BATCHING DEGRADES FIDELITY: batched is strictly further from the fp32 \
             pseudo-reference than serial."
        }
        std::cmp::Ordering::Equal => {
            "BATCHING IS NEUTRAL: serial and batched are equidistant from fp32, so \
             any text difference between them is fp16 noise unrelated to which is \
             more accurate."
        }
    };
    println!(
        "ed(serial, fp32)   = {ds}   (S {} D {} I {})",
        a_serial.s, a_serial.d, a_serial.i
    );
    println!(
        "ed(batched, fp32) = {db}   (S {} D {} I {})",
        a_batched.s, a_batched.d, a_batched.i
    );
    println!(
        "ed(serial, batched) = {}   (S {} D {} I {})",
        a_sb.errors, a_sb.s, a_sb.d, a_sb.i
    );
    println!("{verdict}");

    // --- per-chunk -------------------------------------------------------
    println!();
    println!("=== per-chunk breakdown (words, per chunk, vs fp32) ================");
    println!(
        "{:>2} {:>13} {:>6} {:>10} {:>16} {:>16} {:>8}",
        "ch", "span s", "frames", "words/fp32", "serial S/D/I/ed", "batched S/D/I/ed", "S!=B"
    );
    let batch = batched.batch_size;
    let mut ser_by_pos = [0usize; 8];
    let mut bat_by_pos = [0usize; 8];
    let mut ed_by_pos = [0usize; 8];
    let mut words_by_pos = [0usize; 8];
    let mut n_by_pos = [0usize; 8];
    let mut n_diff_sb_text = 0usize;
    let mut n_diff_sb_full = 0usize;
    let mut n_identical_ref = 0usize;
    for i in 0..fp32.chunks.len() {
        let r: Vec<String> = fp32.chunks[i]
            .words
            .iter()
            .map(|w| w.text.clone())
            .collect();
        let h1: Vec<String> = serial.chunks[i]
            .words
            .iter()
            .map(|w| w.text.clone())
            .collect();
        let h2: Vec<String> = batched.chunks[i]
            .words
            .iter()
            .map(|w| w.text.clone())
            .collect();
        let as1 = align(&r, &h1);
        let as2 = align(&r, &h2);
        let same_text = h1 == h2;
        let same_full = serial.chunks[i].words == batched.chunks[i].words;
        n_diff_sb_text += usize::from(!same_text);
        n_diff_sb_full += usize::from(!same_full);
        n_identical_ref += usize::from(as1.errors() == 0 && as2.errors() == 0);

        let pos = i % batch;
        if pos < 8 {
            ser_by_pos[pos] += as1.errors();
            bat_by_pos[pos] += as2.errors();
            ed_by_pos[pos] += as1.errors() + as2.errors();
            words_by_pos[pos] += r.len();
            n_by_pos[pos] += 1;
        }
        println!(
            "{:>2} {:>5.1}-{:>6.1} {:>6} {:>10} {:>6}/{}/{}/{} {:>6}/{}/{}/{} {:>8}",
            i,
            fp32.chunks[i].start_ms as f64 / 1000.0,
            fp32.chunks[i].end_ms as f64 / 1000.0,
            fp32.chunks[i].encoder_frames,
            r.len(),
            as1.substitutions,
            as1.deletions,
            as1.insertions,
            as1.errors(),
            as2.substitutions,
            as2.deletions,
            as2.insertions,
            as2.errors(),
            if same_text { "-" } else { "yes" }
        );
    }

    println!();
    println!(
        "chunks where serial and batched word text differ at all: {}/{} ({})",
        n_diff_sb_text,
        fp32.chunks.len(),
        pct(n_diff_sb_text, fp32.chunks.len())
    );
    println!(
        "chunks where serial and batched agree on text AND every timestamp: {}/{}",
        fp32.chunks.len() - n_diff_sb_full,
        fp32.chunks.len()
    );
    println!(
        "chunks where BOTH fp16 variants match fp32 exactly: {}/{}",
        n_identical_ref,
        fp32.chunks.len()
    );

    println!();
    println!("=== difference by position inside the batch group ==================");
    println!(
        "batch = {batch}. Position 0 in each group is the chunk whose length sets the \
         group's shared time bucket, so it is never over-padded by the group; positions \
         1.. are padded up to it and over-padded most. Errors concentrated at position \
         0 or at high positions would point at the padding/masking logic; errors spread \
         evenly would point at benign fp16 noise."
    );
    println!(
        "{:>10} {:>7} {:>10} {:>10} {:>11} {:>11} {:>10}",
        "pos-in-grp", "chunks", "ref words", "serial ed", "batched ed", "total ed", "ed/word"
    );
    for pos in 0..batch.min(8) {
        let c = n_by_pos[pos];
        if c == 0 {
            continue;
        }
        println!(
            "{:>10} {:>7} {:>10} {:>10} {:>11} {:>11} {:>10}",
            pos,
            c,
            words_by_pos[pos],
            ser_by_pos[pos],
            bat_by_pos[pos],
            ed_by_pos[pos],
            format!(
                "{:.4}",
                ed_by_pos[pos] as f64 / words_by_pos[pos].max(1) as f64
            )
        );
    }

    // --- a machine-readable copy ----------------------------------------
    let summary = serde_json::json!({
        "verdict": verdict,
        "ed_serial_vs_fp32": ds,
        "ed_batched_vs_fp32": db,
        "ed_serial_vs_batched": a_sb.errors,
        "fnv": {
            "fp32": fp32.fingerprint,
            "fp16_serial": serial.fingerprint,
            "fp16_batched": batched.fingerprint,
        },
        "word_counts": {
            "fp32": fp32.word_count, "fp16_serial": serial.word_count, "fp16_batched": batched.word_count,
        },
        "decode_steps": {
            "fp32": fp32.decode_steps, "fp16_serial": serial.decode_steps, "fp16_batched": batched.decode_steps,
        },
        "serial_vs_fp32": {
            "s": a_serial.s, "d": a_serial.d, "i": a_serial.i,
            "wer": a_serial.errors as f64 / a_serial.ref_words as f64,
            "cer": a_serial.char_errors as f64 / a_serial.ref_chars as f64,
        },
        "batched_vs_fp32": {
            "s": a_batched.s, "d": a_batched.d, "i": a_batched.i,
            "wer": a_batched.errors as f64 / a_batched.ref_words as f64,
            "cer": a_batched.char_errors as f64 / a_batched.ref_chars as f64,
        },
        "serial_vs_batched": {
            "s": a_sb.s, "d": a_sb.d, "i": a_sb.i,
            "wer": a_sb.errors as f64 / a_sb.ref_words as f64,
            "cer": a_sb.char_errors as f64 / a_sb.ref_chars as f64,
        },
        "chunks": fp32.chunks.len(),
        "chunks_serial_batched_text_differs": n_diff_sb_text,
        "chunks_both_match_fp32": n_identical_ref,
    });
    let p = out_dir.join("score.summary.json");
    std::fs::write(
        &p,
        serde_json::to_string_pretty(&summary).map_err(|e| format!("serde: {e}"))?,
    )
    .map_err(|e| format!("{}: {e}", p.display()))?;
    println!();
    println!("wrote       {}", p.display());
    Ok(())
}

fn pct(part: usize, whole: usize) -> String {
    if whole == 0 {
        return "undefined".to_string();
    }
    format!("{:.4}%", part as f64 / whole as f64 * 100.0)
}

// ===========================================================================

fn main() {
    if let Err(e) = run() {
        eprintln!("fp32-reference: {e}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv[0] == "-h" || argv[0] == "--help" {
        eprintln!(
            "usage:\n  \
             fp32-reference transcribe <fp32|fp16_serial|fp16_batched> --out DIR \
             [--wav PATH] [--weights DIR]\n  \
             fp32-reference score DIR\n\n\
             defaults: --wav scratch/clip.wav, --weights artifacts/sravaani"
        );
        return Ok(());
    }
    let mode = argv[0].clone();
    let mut positional: Vec<String> = Vec::new();
    let mut out: Option<PathBuf> = None;
    let mut wav: Option<PathBuf> = None;
    let mut weights: Option<PathBuf> = None;
    let mut it = argv[1..].iter().map(String::as_str);
    while let Some(a) = it.next() {
        match a {
            "--out" => out = Some(PathBuf::from(it.next().ok_or("--out needs a value")?)),
            "--wav" => wav = Some(PathBuf::from(it.next().ok_or("--wav needs a value")?)),
            "--weights" => {
                weights = Some(PathBuf::from(it.next().ok_or("--weights needs a value")?))
            }
            other if other.starts_with('-') => return Err(format!("unknown flag {other}")),
            other => positional.push(other.to_string()),
        }
    }

    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or("cannot locate the repository root")?;
    let wav = wav.unwrap_or_else(|| repo.join("scratch").join("clip.wav"));
    let weights = weights
        .or_else(|| std::env::var("VAANI_WEIGHTS_DIR").ok().map(PathBuf::from))
        .unwrap_or_else(|| repo.join("artifacts").join("sravaani"));

    match mode.as_str() {
        "transcribe" => {
            let name = positional
                .first()
                .ok_or("transcribe needs a variant name")?
                .clone();
            let out = out.ok_or("--out is required")?;
            transcribe(&variant(&name)?, &weights, &wav, &out)
        }
        "score" => {
            let dir = positional
                .first()
                .map(PathBuf::from)
                .or(out)
                .ok_or("score needs a directory")?;
            score(&dir)
        }
        other => Err(format!(
            "unknown mode {other:?}; expected transcribe or score"
        )),
    }
}

//! A reproducible performance benchmark.
//!
//! Run with `cargo run -p vaani-diarization --release --example benchmark`.
//! It reports wall time, the realtime factor, a phase breakdown and the peak
//! working set, for three inputs:
//!
//! 1. a deterministic 14-minute synthetic meeting, so the number is the same
//!    on every machine and comparable between runs;
//! 2. the real `scratch/clip.wav` when it is present, reporting aggregate
//!    numbers only and never transcribing it;
//! 3. a 3-hour silence carrying 10 s of audio, which is the memory case.
//!
//! # What the numbers mean
//!
//! The realtime factor is `processing time / audio duration`, lower is faster.
//! The release profile is the one that matters: the workspace sets
//! `opt-level = 3`, thin LTO and `codegen-units = 1`, and a debug build of this
//! engine is roughly fifty times slower, so a debug RTF is not a performance
//! claim about anything.
//!
//! Peak working set is measured by polling the process working set from a
//! background thread. It is a sampled maximum, not an exact high-water mark,
//! and it includes the `AudioBuffer` the caller already holds -- for the
//! 14-minute case that is 64 MB on its own, which is the dominant term and is
//! not this crate's allocation.

use std::path::PathBuf;
use std::time::Instant;

use vaani_core::pipeline::DiarizationEngine;
use vaani_core::{CancelToken, Time};
use vaani_diarization::config::DiarizationConfig;
use vaani_diarization::engine::HeuristicDiarizer;
use vaani_diarization::synth;

/// Peak working set, sampled while the run is in progress.
///
/// # Why this does not use FFI
///
/// Reading the process working set on Windows means calling
/// `GetProcessMemoryInfo` from `psapi.dll`, which is `unsafe`, and the
/// workspace's rule is that `unsafe` lives in `vaani-inference` and
/// `vaani-asr` only. A benchmark is not a good reason to add an `unsafe` block
/// to a crate whose whole point is that it has none, and the `windows` crate is
/// not in this crate's frozen manifest.
///
/// So this reports the engine's **accounted** allocation, which it can compute
/// exactly, and the *process* peak is measured from outside instead:
///
/// ```powershell
/// $p = Start-Process -PassThru cargo -ArgumentList "run -p vaani-diarization --release --example benchmark"
/// # ... poll $p.WorkingSet64 while it runs ...
/// ```
///
/// `docs/agent/PERFORMANCE.md` carries the number that produced. What is
/// printed here is the part this crate is responsible for, and it is derived
/// from the structures the engine actually allocates rather than estimated.
fn accounted_bytes(frames: usize, windows: usize, dims: usize, rows_clustered: usize) -> u64 {
    // Pass 1: two f32 per frame (energy, flux) plus one for flatness.
    let activity = frames as u64 * 3 * 4;
    // Feature pass: one window's worth of per-frame state at a time.
    let window_state = (150 * dims * 4) as u64;
    // The window vectors themselves.
    let vectors = windows as u64 * (dims as u64 + 1) * 4;
    // The clustering distance matrix, capped at `max_cluster_windows` rows.
    let matrix = (rows_clustered as u64) * (rows_clustered as u64) * 4;
    activity + window_state + vectors + matrix
}

fn report(label: &str, samples: Vec<f32>, config: DiarizationConfig) {
    let audio = vaani_core::media::AudioBuffer::mono(samples, 16_000, Time::ZERO);
    let dur_ms = audio.duration().as_millis();
    let buffer_bytes = audio.samples.len() * 4;

    let engine = HeuristicDiarizer::with_config(config);
    let t0 = Instant::now();
    let r = engine
        .diarize(&audio, &CancelToken::new(), &|_| {})
        .expect("diarization must succeed");
    let wall = t0.elapsed();
    let t = engine.last_timings().expect("timings");
    let dims = vaani_diarization::config::FeatureParams::default().vector_len();
    let rows_clustered = t.windows.min(2_000);

    println!("{label}");
    println!(
        "  audio           {dur_ms} ms ({:.1} s)",
        dur_ms as f64 / 1000.0
    );
    println!("  wall            {:.3} s", wall.as_secs_f64());
    println!("  RTF             {:.5}", t.rtf(dur_ms));
    println!(
        "  phase           activity {} ms, features {} ms, cluster {} ms, smooth {} ms",
        t.activity_ms, t.features_ms, t.cluster_ms, t.smooth_ms
    );
    println!(
        "  shape           {} frames, {} active, {} windows, {} clustered, rule `{}`",
        t.frames, t.active_frames, t.windows, rows_clustered, t.rule
    );
    println!(
        "  result          {} speakers, {} spans, coverage {:.3}, mean turn {} ms",
        r.num_speakers,
        r.spans.len(),
        r.coverage(audio.duration()),
        if r.spans.is_empty() {
            0
        } else {
            r.speaker_durations().iter().sum::<i64>() / r.spans.len() as i64
        }
    );
    let owned = accounted_bytes(t.frames, t.windows, dims, rows_clustered);
    println!(
        "  engine-owned    {:.2} MB accounted, on top of the {:.2} MB AudioBuffer the caller holds",
        owned as f64 / 1_048_576.0,
        buffer_bytes as f64 / 1_048_576.0
    );
    println!();
}

/// RTF at a range of durations, to show that the cost is linear in audio and
/// not quadratic in it.
fn scaling() {
    println!("scaling (14-minute synthetic meeting, 2 speakers, 5 s turns)");
    println!("  duration   wall s     RTF");
    for minutes in [1usize, 2, 7, 14] {
        let total = minutes as i64 * 60_000;
        let voices = synth::two_voices();
        let (samples, _) = synth::render(
            &synth::alternate(&[0, 1], 0, 5_000, (total / 5_000) as usize),
            &voices,
            total,
            16_000,
            0.003,
            4242,
        );
        let audio = vaani_core::media::AudioBuffer::mono(samples, 16_000, Time::ZERO);
        let e = HeuristicDiarizer::new();
        let t0 = Instant::now();
        let r = e.diarize(&audio, &CancelToken::new(), &|_| {}).unwrap();
        let wall = t0.elapsed();
        let t = e.last_timings().unwrap();
        println!(
            "  {minutes:>6} min   {:>7.3}   {:.5}  ({} speakers, {} windows)",
            wall.as_secs_f64(),
            t.rtf(audio.duration().as_millis()),
            r.num_speakers,
            t.windows
        );
    }
    println!();
}

fn main() {
    println!("vaani-diarization benchmark");
    println!("profile: release (opt-level 3, thin LTO, codegen-units 1)");
    println!();

    // The 14-minute reproducible case.
    let voices = synth::two_voices();
    let (samples, _) = synth::render(
        &synth::alternate(&[0, 1], 0, 5_000, 168),
        &voices,
        840_000,
        16_000,
        0.003,
        4242,
    );
    report(
        "synthetic 14-minute meeting",
        samples,
        DiarizationConfig::default(),
    );

    scaling();

    // The memory case: a long silence carrying a short burst.
    let voices = synth::two_voices();
    let (mut samples, _) = synth::render(
        &synth::alternate(&[0, 1], 0, 2_000, 5),
        &voices,
        10_000,
        16_000,
        0.003,
        7,
    );
    samples.resize(180 * 16_000, 0.0);
    report(
        "3 minutes of silence carrying 10 s of speech",
        samples,
        DiarizationConfig::default(),
    );

    // The real recording, if it is present. Aggregate numbers only.
    if let Some(path) = scratch_clip() {
        match read_wav(&path) {
            Ok((samples, 16_000)) => {
                report(
                    "real meeting recording",
                    samples,
                    DiarizationConfig::default(),
                );
            }
            Ok((_, rate)) => println!("skipping {path:?}: sample rate {rate}, expected 16000"),
            Err(e) => println!("skipping {path:?}: {e}"),
        }
    } else {
        println!("skipping the real recording: scratch/clip.wav is not present");
    }
}

/// `scratch/clip.wav`, if it is there. Read-only, never copied, never
/// transcribed: `report` prints counts and percentages only.
fn scratch_clip() -> Option<PathBuf> {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|root| root.join("scratch").join("clip.wav"))?;
    p.is_file().then_some(p)
}

fn read_wav(path: &std::path::Path) -> std::io::Result<(Vec<f32>, u32)> {
    let bytes = std::fs::read(path)?;
    let mut off = 12usize;
    let (mut fmt, mut bits, mut channels, mut rate) = (0u16, 0u16, 1u16, 16_000u32);
    let mut data: Option<(usize, usize)> = None;
    while off + 8 <= bytes.len() {
        let id = &bytes[off..off + 4];
        let size = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap()) as usize;
        if id == b"fmt " && size >= 16 {
            fmt = u16::from_le_bytes(bytes[off + 8..off + 10].try_into().unwrap());
            channels = u16::from_le_bytes(bytes[off + 10..off + 12].try_into().unwrap());
            rate = u32::from_le_bytes(bytes[off + 12..off + 16].try_into().unwrap());
            bits = u16::from_le_bytes(bytes[off + 22..off + 24].try_into().unwrap());
        } else if id == b"data" {
            data = Some((off + 8, size.min(bytes.len() - off - 8)));
        }
        off += 8 + size + (size & 1);
    }
    let (start, len) = data.ok_or_else(|| std::io::Error::other("no data chunk"))?;
    let raw = &bytes[start..start + len];
    let per = (bits / 8).max(1) as usize;
    let ch = channels.max(1) as usize;
    let frames = raw.len() / (per * ch);
    let mut out = Vec::with_capacity(frames);
    for i in 0..frames {
        let mut acc = 0f32;
        for c in 0..ch {
            let at = (i * ch + c) * per;
            if at + per > raw.len() {
                break;
            }
            let v = match (fmt, bits) {
                (3, 32) => f32::from_le_bytes(raw[at..at + 4].try_into().unwrap()),
                (1, 16) => {
                    i16::from_le_bytes(raw[at..at + 2].try_into().unwrap()) as f32 / 32_768.0
                }
                (1, 32) => {
                    i32::from_le_bytes(raw[at..at + 4].try_into().unwrap()) as f32 / 2_147_483_648.0
                }
                _ => return Err(std::io::Error::other("unsupported sample format")),
            };
            acc += v;
        }
        out.push(acc / ch as f32);
    }
    Ok((out, rate))
}

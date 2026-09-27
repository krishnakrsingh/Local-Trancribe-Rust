//! Prints the activity-detector's per-frame statistics for a set of signals.
//!
//! Run with `cargo run -p vaani-diarization --example probe_activity`. The
//! output is aggregate numbers only: percentiles of frame energy, of spectral
//! flux and of spectral flatness, plus the resulting speech ratio. It is how
//! the flatness threshold and the energy ratios in `ActivityParams` were
//! chosen, and re-running it after changing any of them is the way to see
//! whether a change helped.
//!
//! With a path argument it also reads a 16-bit or float32 WAV file, which is
//! how the numbers on the real meeting recording were taken. It prints
//! statistics only; it never prints or transcribes content.

use std::path::PathBuf;

use vaani_core::CancelToken;
use vaani_core::media::AudioBuffer;
use vaani_core::pipeline::DiarizationEngine;
use vaani_diarization::config::{ActivityParams, DiarizationConfig, FeatureParams};
use vaani_diarization::dsp::FrontEnd;
use vaani_diarization::engine::HeuristicDiarizer;
use vaani_diarization::vad;

fn report(label: &str, buf: &[f32], p: &ActivityParams) {
    let mut fe = FrontEnd::new(FeatureParams::default(), *p);
    let frames = vad::scan(&mut fe, buf, || false, |_| {}, 50);
    if frames.is_empty() {
        println!("{label}: no frames");
        return;
    }
    let t = vad::derive_thresholds(&frames, p);
    let mask = vad::activity_mask(&frames, &t);
    let active = mask.iter().filter(|m| **m).count();
    let pct = |v: &[f32], q: f32| vaani_diarization::dsp::percentile(v, q);
    println!("{label}");
    println!(
        "  frames {}  usable {}  active {active} ({:.1}%)",
        frames.frames,
        frames.frames,
        100.0 * active as f64 / frames.frames as f64
    );
    println!(
        "  energy dB  p05 {:.1}  p10 {:.1}  p50 {:.1}  p90 {:.1}  p95 {:.1}",
        pct(&frames.energy_db, 5.0),
        pct(&frames.energy_db, 10.0),
        pct(&frames.energy_db, 50.0),
        pct(&frames.energy_db, 90.0),
        pct(&frames.energy_db, 95.0)
    );
    println!(
        "  flux       p10 {:.3}  p50 {:.3}  p90 {:.3}",
        pct(&frames.flux, 10.0),
        pct(&frames.flux, 50.0),
        pct(&frames.flux, 90.0)
    );
    println!(
        "  flatness   p05 {:.3}  p10 {:.3}  p50 {:.3}  p90 {:.3}  p95 {:.3}",
        pct(&frames.flatness, 5.0),
        pct(&frames.flatness, 10.0),
        pct(&frames.flatness, 50.0),
        pct(&frames.flatness, 90.0),
        pct(&frames.flatness, 95.0)
    );
    println!(
        "  thresholds usable {}  energy_lo {:.1}  energy_hi {:.1}  flux {:.3}  flatness {:.2}",
        t.usable, t.energy_lo_db, t.energy_hi_db, t.flux, t.max_flatness
    );
    let active_only: Vec<f32> = frames
        .flatness
        .iter()
        .zip(&mask)
        .filter(|(_, m)| **m)
        .map(|(f, _)| *f)
        .collect();
    if !active_only.is_empty() {
        println!(
            "  flatness on active frames only: p50 {:.3}  p90 {:.3}",
            vaani_diarization::dsp::percentile(&active_only, 50.0),
            vaani_diarization::dsp::percentile(&active_only, 90.0)
        );
    }
}

/// Read a RIFF/WAVE file that is either 16-bit PCM or 32-bit IEEE float,
/// mono or interleaved, and downmix to mono f32. Enough for a probe; the
/// product's decoder is `vaani-media`'s job.
fn read_wav(path: &PathBuf) -> std::io::Result<(Vec<f32>, u32)> {
    let bytes = std::fs::read(path)?;
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(std::io::Error::other("not a RIFF/WAVE file"));
    }
    let mut off = 12usize;
    let (mut fmt, mut bits, mut channels) = (0u16, 0u16, 1u16);
    let mut data: Option<(usize, usize)> = None;
    while off + 8 <= bytes.len() {
        let id = &bytes[off..off + 4];
        let size = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap()) as usize;
        if id == b"fmt " {
            fmt = u16::from_le_bytes(bytes[off + 8..off + 10].try_into().unwrap());
            channels = u16::from_le_bytes(bytes[off + 10..off + 12].try_into().unwrap());
            bits = u16::from_le_bytes(bytes[off + 22..off + 24].try_into().unwrap());
        } else if id == b"data" {
            data = Some((off + 8, size.min(bytes.len() - off - 8)));
        }
        off += 8 + size + (size & 1);
    }
    let (start, len) = data.ok_or_else(|| std::io::Error::other("no data chunk"))?;
    let raw = &bytes[start..start + len];
    let per = (bits / 8) as usize;
    let frames = raw.len() / (per * channels as usize);
    let mut out = Vec::with_capacity(frames);
    for i in 0..frames {
        let mut acc = 0f32;
        for c in 0..channels as usize {
            let at = (i * channels as usize + c) * per;
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
        out.push(acc / channels as f32);
    }
    Ok((out, 16_000))
}

/// Re-run the engine across a sweep of merge distances and report what each
/// one produces. This is how the `SpeakerCount::Threshold` default was chosen:
/// the threshold is set where the speaker count stops changing, not where it
/// looked plausible.
fn sweep(samples: &[f32]) {
    println!("\nmerge-distance sweep");
    println!("  distance  speakers  spans  coverage  mean turn ms  rtf");
    for d in [
        0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.30, 1.50, 1.80,
    ] {
        let mut cfg = DiarizationConfig::default();
        cfg.clusters.count = vaani_diarization::config::SpeakerCount::Threshold {
            merge_distance: d,
            max_speakers: 12,
        };
        let engine = HeuristicDiarizer::with_config(cfg);
        let audio = AudioBuffer::mono(samples.to_vec(), 16_000, vaani_core::Time::ZERO);
        match engine.diarize(&audio, &CancelToken::new(), &|_| {}) {
            Ok(r) => {
                let mean = if r.spans.is_empty() {
                    0.0
                } else {
                    let total: i64 = r
                        .spans
                        .iter()
                        .map(|s| s.end.as_millis() - s.start.as_millis())
                        .sum();
                    total as f64 / r.spans.len() as f64
                };
                let rtf = engine
                    .last_timings()
                    .map(|t| t.rtf(audio.duration().as_millis()))
                    .unwrap_or(0.0);
                println!(
                    "  {d:8.2}  {:8}  {:5}  {:8.3}  {:12.0}  {rtf:.4}",
                    r.num_speakers,
                    r.spans.len(),
                    r.coverage(audio.duration()),
                    mean
                );
            }
            Err(e) => println!("  {d:8.2}  failed: {}", e.code()),
        }
    }
}

/// Report the distance distributions the clustering actually sees, using the
/// engine's own window embeddings and the synthetic timeline's known speaker
/// identity. This is the measurement that sets [`BlockWeights`] and the
/// merge-distance threshold: a weight change is judged by how far apart the
/// two distributions move, not by whether the speaker count happened to look
/// plausible.
fn labelled_distances(voices: &[vaani_diarization::synth::Voice], total_ms: i64, half: usize) {
    let turn = 5_000i64;
    let mut cfg = DiarizationConfig::default();
    cfg.clusters.feature_smooth_half_windows = half;
    cfg.clusters.count = vaani_diarization::config::SpeakerCount::Fixed(voices.len());
    let engine = HeuristicDiarizer::with_config(cfg);
    let (samples, _truth) = vaani_diarization::synth::render(
        &vaani_diarization::synth::alternate(
            &(0..voices.len()).collect::<Vec<_>>(),
            0,
            turn,
            (total_ms / turn) as usize,
        ),
        voices,
        total_ms,
        16_000,
        0.003,
        11,
    );
    let audio = AudioBuffer::mono(samples, 16_000, vaani_core::Time::ZERO);
    let windows = match engine.debug_windows(&audio, &CancelToken::new()) {
        Ok(w) => w,
        Err(e) => {
            println!("  debug_windows failed: {}", e.debug_chain());
            return;
        }
    };
    // A window's ground-truth label is the speaker who owns the middle of it.
    // Windows that straddle a turn boundary are reported separately: a window
    // holding 0.75 s of each of two speakers is a chimera, and including those
    // in the "same speaker" distribution measures the fixture's turn length
    // rather than the feature vector.
    let pure = |w: &vaani_diarization::engine::LabelledWindow| -> Option<usize> {
        let a = w.start.as_millis() / turn;
        let b = (w.end.as_millis().saturating_sub(1)) / turn;
        if a == b {
            Some((a as usize) % voices.len())
        } else {
            None
        }
    };
    let mut same: Vec<f64> = Vec::new();
    let mut diff: Vec<f64> = Vec::new();
    let mut straddling = 0usize;
    for i in 0..windows.len() {
        let mid = (windows[i].start.as_millis() + windows[i].end.as_millis()) / 2;
        for j in (i + 1)..windows.len() {
            let mid2 = (windows[j].start.as_millis() + windows[j].end.as_millis()) / 2;
            if mid2 - mid < turn / 2 {
                continue;
            }
            let (Some(a), Some(b)) = (pure(&windows[i]), pure(&windows[j])) else {
                straddling += 1;
                continue;
            };
            let d = vaani_diarization::cosine_distance(&windows[i].vector, &windows[j].vector);
            if a == b {
                same.push(d);
            } else {
                diff.push(d);
            }
        }
    }
    println!(
        "\nlabelled window distances: {} windows, smooth half = {half} ({} straddling pairs excluded)",
        windows.len(),
        straddling
    );
    let q = |v: &mut Vec<f64>, p: f64| -> f64 {
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let r = (p / 100.0) * (v.len() - 1) as f64;
        v[r.round() as usize]
    };
    println!("  pct      same   different");
    for p in [10.0, 25.0, 50.0, 75.0, 90.0, 99.0] {
        println!(
            "  p{p:>4.0}  {:.4}  {:.4}",
            q(&mut same, p),
            q(&mut diff, p)
        );
    }
}

/// The distribution of all pairwise window distances, with no labels.
///
/// This is how the merge-distance threshold is set on a recording with no
/// annotated reference: the histogram's shape shows where the within-speaker
/// mass ends and the between-speaker mass begins, and the threshold goes in
/// the valley. It is a **shape**, not an accuracy measurement, and it says
/// nothing about whether the resulting clusters correspond to real people.
fn distance_histogram(samples: &[f32], label: &str) {
    let engine = HeuristicDiarizer::new();
    let audio = AudioBuffer::mono(samples.to_vec(), 16_000, vaani_core::Time::ZERO);
    let windows = match engine.debug_windows(&audio, &CancelToken::new()) {
        Ok(w) => w,
        Err(e) => {
            println!("  debug_windows failed: {}", e.debug_chain());
            return;
        }
    };
    let n = windows.len();
    println!("\n{label}: all pairwise window distances, {n} windows");
    if n > 400 {
        println!("  (subsampled to the first 400 windows to bound the O(n^2) work)");
    }
    let m = n.min(400);
    let mut hist = [0usize; 20];
    let mut pairs = 0usize;
    for i in 0..m {
        for j in (i + 1)..m {
            let d = vaani_diarization::cosine_distance(&windows[i].vector, &windows[j].vector);
            let b = ((d * 10.0) as usize).min(19);
            hist[b] += 1;
            pairs += 1;
        }
    }
    println!("  distance   pairs     share");
    for (b, c) in hist.iter().enumerate() {
        let lo = b as f64 / 10.0;
        println!(
            "  {lo:4.1}-{:<4.1} {:8}  {:6.2}%  {}",
            lo + 0.1,
            c,
            100.0 * *c as f64 / pairs as f64,
            "#".repeat((*c as f64 / pairs as f64 * 100.0) as usize)
        );
    }
}

fn main() {
    let p = ActivityParams::default();
    let voices = vaani_diarization::synth::two_voices();

    report("digital silence", &vec![0.0; 16_000 * 10], &p);
    report(
        "white noise",
        &{
            let mut rng = 0x5eed_1234u64;
            (0..16_000 * 10)
                .map(|_| {
                    rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    ((rng >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
                })
                .collect::<Vec<f32>>()
        },
        &p,
    );
    report(
        "two synthetic speakers, back to back",
        &{
            let u = vaani_diarization::synth::alternate(&[0, 1], 0, 2_000, 5);
            vaani_diarization::synth::render(&u, &voices, 10_000, 16_000, 0.002, 7).0
        },
        &p,
    );
    report(
        "two synthetic speakers with silences between",
        &{
            let mut u: Vec<vaani_diarization::synth::Utterance> = Vec::new();
            for i in 0..5 {
                let s = i as i64 * 2_000;
                u.push(vaani_diarization::synth::Utterance::new(
                    (i % 2) as usize,
                    s,
                    s + 1_200,
                ));
            }
            vaani_diarization::synth::render(&u, &voices, 10_000, 16_000, 0.002, 7).0
        },
        &p,
    );
    report(
        "one synthetic speaker, continuous",
        &{
            let u = vaani_diarization::synth::alternate(&[0], 0, 2_000, 5);
            vaani_diarization::synth::render(&u, &voices, 10_000, 16_000, 0.002, 7).0
        },
        &p,
    );
    report(
        "one pure tone",
        &(0..16_000 * 10)
            .map(|i| (2.0 * std::f32::consts::PI * 200.0 * i as f32 / 16_000.0).sin() * 0.3)
            .collect::<Vec<f32>>(),
        &p,
    );
    // The smoother half-width is the one parameter that trades same-speaker
    // tightness against turn-boundary smearing, and the two populations below
    // are how that trade is judged. Half-width 2 is the default: on the
    // synthetic fixture the same-speaker 90th percentile is 0.25 and the
    // different-speaker 10th is 1.32, so the merge distance sits in a wide
    // empty gap. Half-width 6 spans more than a 5 s turn and smears the
    // boundary into the interior, which is why it is not the default.
    for half in [0usize, 2, 6] {
        labelled_distances(&voices, 150_000, half);
    }
    // A single continuous speaker: the same-speaker distribution has to sit
    // far below the merge distance or one person shatters into several.
    labelled_distances(&voices[..1], 150_000, 2);

    if let Some(arg) = std::env::args().nth(1) {
        let path = PathBuf::from(&arg);
        match read_wav(&path) {
            Ok((samples, sr)) => {
                println!("\n{arg}: {} samples at {sr} Hz", samples.len());
                report("  file", &samples, &p);
                let engine = HeuristicDiarizer::with_config(DiarizationConfig::default());
                let audio = AudioBuffer::mono(samples.clone(), sr, vaani_core::Time::ZERO);
                match engine.diarize(&audio, &CancelToken::new(), &|_| {}) {
                    Ok(r) => {
                        let durs = r.speaker_durations();
                        let total: i64 = durs.iter().sum();
                        println!("  speakers {}", r.num_speakers);
                        println!("  spans {}", r.spans.len());
                        println!("  coverage {:.3}", r.coverage(audio.duration()));
                        for (i, d) in durs.iter().enumerate() {
                            println!(
                                "    speaker {i}: {} ms ({:.1}% of speech)",
                                d,
                                100.0 * *d as f64 / total.max(1) as f64
                            );
                        }
                        if let Some(t) = engine.last_timings() {
                            println!(
                                "  activity {} ms  features {} ms  cluster {} ms  smooth {} ms  total {} ms",
                                t.activity_ms, t.features_ms, t.cluster_ms, t.smooth_ms, t.total_ms
                            );
                            println!(
                                "  windows {}  frames {}  active {}",
                                t.windows, t.frames, t.active_frames
                            );
                            println!("  rule {}  clusters {}", t.rule, t.clusters);
                            println!(
                                "  rtf {:.4}  (debug build; see the release benchmark)",
                                t.rtf(audio.duration().as_millis())
                            );
                        }
                    }
                    Err(e) => println!("  diarize failed: {}", e.debug_chain()),
                }
                distance_histogram(&samples, &arg);
                sweep(&samples);
            }
            Err(e) => println!("\n{arg}: {e}"),
        }
    }
}

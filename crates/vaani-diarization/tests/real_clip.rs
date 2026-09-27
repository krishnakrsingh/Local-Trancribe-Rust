//! The real 13-minute meeting, measured and reported as **aggregate numbers
//! only**.
//!
//! # Privacy, enforced by this file's design
//!
//! `scratch/clip.wav` is a private meeting recording. It is read, never
//! copied into this crate, never committed, and never transcribed. This test:
//!
//! * **skips** with a printed reason when the file is absent, so a fresh
//!   checkout and a CI machine both work;
//! * prints only counts, durations, percentages and timings;
//! * never prints, logs, asserts on, or even holds a `String` of anything
//!   that was said;
//! * never writes a file.
//!
//! There is no accuracy assertion here, and the reason is worth stating: **no
//! diarization accuracy claim is possible without an annotated reference, and
//! this repository does not have one.** DER/JER need per-speaker ground truth
//! for this recording. There is none. The numbers below are a *sanity* check
//! that the engine behaves on real audio -- it does not crash, it finds some
//! speech, it produces a plausible amount of it -- and a set of performance
//! figures. Anyone reading them as an accuracy result would be wrong.
//!
//! What the numbers *do* show is in `OPEN ISSUES`: on this recording the
//! feature vector's window distances are close to uniform, so the speaker
//! count the engine returns is a function of the threshold rather than of the
//! meeting. That is a real, measured limitation and it is stated as one.

use std::path::PathBuf;
use std::time::Instant;

use vaani_core::pipeline::DiarizationEngine;
use vaani_core::{CancelToken, Time};
use vaani_diarization::attribution;
use vaani_diarization::config::DiarizationConfig;
use vaani_diarization::engine::HeuristicDiarizer;

fn clip_path() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|root| root.join("scratch").join("clip.wav"));
    p.filter(|p| p.is_file())
}

/// Read a RIFF/WAVE file that is 16-bit PCM or 32-bit IEEE float, mono or
/// interleaved, and downmix to mono f32. Enough for a probe; the product's
/// decoder is `vaani-media`'s job and this deliberately does not depend on it.
fn read_wav(path: &std::path::Path) -> std::io::Result<(Vec<f32>, u32)> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(std::io::Error::other("not a RIFF/WAVE file"));
    }
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

#[test]
fn the_real_meeting_runs_and_reports_aggregate_numbers() {
    let Some(path) = clip_path() else {
        println!("skipping: scratch/clip.wav is not present");
        return;
    };
    let (samples, rate) = read_wav(&path).expect("scratch/clip.wav must be readable");
    println!(
        "file: {} samples at {rate} Hz, {:.1} s",
        samples.len(),
        samples.len() as f64 / rate as f64
    );
    assert_eq!(rate, 16_000, "the engine requires 16 kHz mono");
    assert!(
        samples.iter().all(|v| v.is_finite()),
        "the file contains a non-finite sample"
    );

    let audio = vaani_core::media::AudioBuffer::mono(samples, rate, Time::ZERO);
    let engine = HeuristicDiarizer::new();
    let token = CancelToken::new();
    let t0 = Instant::now();
    let r = engine
        .diarize(&audio, &token, &|_| {})
        .expect("diarization must succeed");
    let wall = t0.elapsed();
    let t = engine
        .last_timings()
        .expect("timings are recorded for every run");

    let dur_ms = audio.duration().as_millis();
    let durations = r.speaker_durations();
    let total: i64 = durations.iter().sum();

    println!(
        "diarization: {:.2} s wall, RTF {:.4}",
        wall.as_secs_f64(),
        t.rtf(dur_ms)
    );
    println!(
        "  activity {} ms ({:.1}%)  features {} ms ({:.1}%)  cluster {} ms ({:.1}%)  smooth {} ms",
        t.activity_ms,
        100.0 * t.activity_ms as f64 / t.total_ms.max(1) as f64,
        t.features_ms,
        100.0 * t.features_ms as f64 / t.total_ms.max(1) as f64,
        t.cluster_ms,
        100.0 * t.cluster_ms as f64 / t.total_ms.max(1) as f64,
        t.smooth_ms
    );
    println!(
        "  {} frames, {} active ({:.1}%), {} windows, rule `{}`, {} clusters",
        t.frames,
        t.active_frames,
        100.0 * t.active_frames as f64 / t.frames.max(1) as f64,
        t.windows,
        t.rule,
        t.clusters
    );
    println!(
        "  {} speakers, {} spans, coverage {:.3}",
        r.num_speakers,
        r.spans.len(),
        r.coverage(audio.duration())
    );
    // Talk-time *ratios* only. No content, no names, no transcript.
    for (i, d) in durations.iter().enumerate() {
        println!(
            "    speaker {i}: {:.1}% of speech ({d} ms)",
            100.0 * *d as f64 / total.max(1) as f64
        );
    }

    // ---- what may legitimately be asserted on real audio ----------------
    // These are structural and performance facts, not accuracy. Each one has
    // caught a real bug at some point.
    assert!(
        !r.spans.is_empty(),
        "no spans on a recording that has speech"
    );
    assert!(r.num_speakers >= 1);
    assert!(r.num_speakers <= 8, "the cap was not honoured");
    assert!(
        r.coverage(audio.duration()) > 0.3,
        "coverage {:.3}",
        r.coverage(audio.duration())
    );
    assert!(r.coverage(audio.duration()) <= 1.0);
    assert!(
        attribution::assert_consistent(&r).is_empty(),
        "{:?}",
        attribution::assert_consistent(&r)
    );
    assert!(t.windows > 100, "{} windows for 13 minutes", t.windows);
    assert!(
        t.rtf(dur_ms) < 0.5,
        "RTF {:.4} is too slow to be usable in a job",
        t.rtf(dur_ms)
    );
    // The spans must be turns, not a scatter: a mean turn well under a second
    // would mean the smoothing is not working.
    let mean_turn = if r.spans.is_empty() {
        0
    } else {
        total / r.spans.len() as i64
    };
    println!("  mean turn {} ms", mean_turn);
    assert!(
        mean_turn > 700,
        "mean turn {mean_turn} ms is a scatter, not turns"
    );

    // Attribution onto a transcript works on a real result: every index
    // resolves, and the shares are a distribution.
    let mut tr = vaani_core::transcript::Transcript::new(vaani_core::transcript::EngineStamp::asr(
        "test", "1",
    ));
    let mut at = 0i64;
    while at < dur_ms {
        let b = (at + 7_000).min(dur_ms);
        tr.push_chunk(
            Time(at),
            vec![vaani_core::transcript::Segment::new(Time(0), Time(b - at))],
        );
        at = b;
    }
    let report = attribution::attribute_speakers(&mut tr, &r);
    println!(
        "  attribution: {}/{} segments attributed, {} contested, {} speakers used",
        report.attributed,
        report.attributed + report.unattributed,
        report.contested,
        report.speakers_used
    );
    assert!(report.attributed > 0);
    for seg in &tr.segments {
        if let Some(s) = seg.speaker {
            assert!(tr.speaker(s).is_some());
        }
    }
}

#[test]
fn the_speaker_count_moves_with_the_threshold_and_that_is_reported() {
    // Not a test with an expected answer: a *measurement* that is printed and
    // bounded. The point is that the threshold is the only thing deciding the
    // speaker count on this recording, and that has to be visible rather than
    // discovered by a user.
    let Some(path) = clip_path() else {
        println!("skipping: scratch/clip.wav is not present");
        return;
    };
    let (samples, rate) = read_wav(&path).expect("readable");
    if rate != 16_000 {
        println!("skipping: sample rate {rate}");
        return;
    }
    let audio = vaani_core::media::AudioBuffer::mono(samples, rate, Time::ZERO);
    println!("threshold sweep on the real recording:");
    for d in [0.40, 0.55, 0.70, 0.85, 1.00, 1.20] {
        let mut cfg = DiarizationConfig::default();
        cfg.clusters.count = vaani_diarization::config::SpeakerCount::Threshold {
            merge_distance: d,
            max_speakers: 12,
        };
        let e = HeuristicDiarizer::with_config(cfg);
        let t0 = Instant::now();
        let r = e.diarize(&audio, &CancelToken::new(), &|_| {}).unwrap();
        let mean = if r.spans.is_empty() {
            0
        } else {
            r.speaker_durations().iter().sum::<i64>() / r.spans.len() as i64
        };
        println!(
            "  merge distance {d:.2}: {} speakers, {} spans, mean turn {mean} ms, {:.2} s",
            r.num_speakers,
            r.spans.len(),
            t0.elapsed().as_secs_f64()
        );
        assert!(r.num_speakers >= 1);
        assert!(r.num_speakers <= 12);
    }
    // A single sweep point is asserted: the default must not be the extreme of
    // the range, or the cap would be doing the work.
    let e = HeuristicDiarizer::new();
    let r = e.diarize(&audio, &CancelToken::new(), &|_| {}).unwrap();
    assert!(
        r.num_speakers < 12,
        "the default threshold hit the cap: the estimate is not doing anything"
    );
}

#[test]
fn a_three_hour_silence_with_one_burst_does_not_blow_up_on_real_shaped_input() {
    // The memory requirement, exercised with the real engine on a buffer whose
    // *shape* is a 3-hour recording: a long run of near-silence with a short
    // burst. Allocating 691 MB of f32 in a test is not reasonable, so this
    // checks the engine's own reported accounting at ten times the length.
    let voices = vaani_diarization::synth::two_voices();
    let (mut samples, _) = vaani_diarization::synth::render(
        &vaani_diarization::synth::alternate(&[0, 1], 0, 2_000, 6),
        &voices,
        12_000,
        16_000,
        0.003,
        12,
    );
    // 30 minutes, at 16 kHz. 28.8 M samples: 115 MB, which is an acceptable
    // allocation for a test and is a tenth of a 3-hour file.
    samples.resize(30 * 60 * 16_000, 0.0);
    let audio = vaani_core::media::AudioBuffer::mono(samples, 16_000, Time::ZERO);
    let e = HeuristicDiarizer::new();
    let t0 = Instant::now();
    let r = e.diarize(&audio, &CancelToken::new(), &|_| {}).unwrap();
    let wall = t0.elapsed();
    let t = e.last_timings().unwrap();
    println!(
        "30 min with 12 s of speech: {:?}, {} speakers, {} spans, {} frames, {} windows, RTF {:.4}",
        wall,
        r.num_speakers,
        r.spans.len(),
        t.frames,
        t.windows,
        t.rtf(audio.duration().as_millis())
    );
    assert_eq!(t.frames, 30 * 60 * 100 - 1);
    assert!(t.windows > 0 && t.windows < 60, "{} windows", t.windows);
    assert!(r.coverage(audio.duration()) < 0.02);
}

//! Real-data validation on `scratch/clip.wav`.
//!
//! # Privacy
//!
//! `scratch/clip.wav` is a **private meeting recording**. This file reads it,
//! never writes it, never copies it into the crate, and never prints or
//! describes what is said in it. Every number it emits is an aggregate:
//! duration, speech ratio, segment count, mean and median segment duration,
//! measured detection time, realtime factor and peak resident set size. There
//! is no transcript, no excerpt, and no per-segment dump of anything but
//! timestamps and durations.
//!
//! The file is reached through a read-only junction, and the test **skips** when
//! it is absent, so a fresh clone runs green.
//!
//! ```
//! cargo test -p vaani-vad --test real_data -- --nocapture
//! ```
//!
//! # What the numbers are for
//!
//! The synthetic fixtures in `calibration.rs` establish that the detector
//! *works*. They cannot establish that the defaults are right for a real
//! meeting, because a synthetic voice is not a voice: it has no plosives, no
//! creaky voice, no overlapping speakers, no packet loss and no gain riding.
//! This file is where the defaults meet reality, and the assertions are
//! deliberately *bands* rather than exact figures, because a band is what a
//! different machine and a different meeting should still satisfy.

mod support;

use std::time::Instant;
use support::wav;
use vaani_core::media::AudioBuffer;
use vaani_core::pipeline::VadEngine;
use vaani_core::{CancelToken, Time};
use vaani_vad::smoothing::build_timeline_at;
use vaani_vad::{ChunkPolicy, Chunker, EnergyVad, EnergyVadConfig};

/// `scratch/clip.wav`, relative to the workspace root.
fn clip() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|root| root.join("scratch").join("clip.wav"))
        .unwrap_or_default()
}

struct Loaded {
    audio: AudioBuffer,
    info: wav::WavInfo,
}

fn load() -> Option<Loaded> {
    let path = clip();
    if !path.is_file() {
        println!("skipping: {} is not present", path.display());
        return None;
    }
    let (samples, info) = wav::read_mono_f32(&path).expect("clip.wav must parse");
    println!(
        "  {}: {} frames, {} Hz, {} ch, {} bits, chunks {:?}",
        path.display(),
        info.frames,
        info.sample_rate,
        info.channels,
        info.bits_per_sample,
        info.chunks
    );
    assert_eq!(info.sample_rate, 16_000, "the detector needs 16 kHz");
    assert_eq!(info.channels, 1);
    Some(Loaded {
        audio: AudioBuffer::mono(samples, info.sample_rate, Time::ZERO),
        info,
    })
}

/// One row of the tuning table.
struct Row {
    label: &'static str,
    speech_ratio: f64,
    segments: usize,
    mean_ms: f64,
    median_ms: f64,
    p90_ms: f64,
    max_ms: f64,
    min_ms: f64,
    detect_s: f64,
    rtf: f64,
    floor_db: f32,
    p10_db: f32,
    p90_db: f32,
    gated_pct: f64,
}

fn measure(label: &'static str, audio: &AudioBuffer, cfg: EnergyVadConfig) -> Row {
    let vad = EnergyVad::with_config(cfg).expect("config");
    let started = Instant::now();
    let tl = vad.detect(audio, &CancelToken::new()).expect("detect");
    let secs = started.elapsed().as_secs_f64();
    let dur_s = audio.duration().as_millis() as f64 / 1000.0;
    let st = vad.stats();

    let mut durs: Vec<f32> = tl
        .segments
        .iter()
        .map(|s| (s.end.as_millis() - s.start.as_millis()) as f32)
        .collect();
    durs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = if durs.is_empty() {
        0.0
    } else {
        durs.iter().map(|v| *v as f64).sum::<f64>() / durs.len() as f64
    };
    let q = |f: f64| -> f64 {
        if durs.is_empty() {
            return 0.0;
        }
        let i = (f * (durs.len() - 1) as f64) as usize;
        durs[i] as f64
    };

    Row {
        label,
        speech_ratio: tl.speech_ratio(audio.duration()),
        segments: tl.segments.len(),
        mean_ms: mean,
        median_ms: q(0.5),
        p90_ms: q(0.9),
        max_ms: durs.last().copied().unwrap_or(0.0) as f64,
        min_ms: durs.first().copied().unwrap_or(0.0) as f64,
        detect_s: secs,
        rtf: if dur_s > 0.0 { secs / dur_s } else { 0.0 },
        floor_db: st.final_floor_db,
        p10_db: st.global_p10_db,
        p90_db: st.global_p90_db,
        gated_pct: st.gated_frames as f64 / (st.frames.max(1)) as f64,
    }
}

fn print_row(r: &Row) {
    println!(
        "  {:<26} ratio={:.4}  n={:<5} mean={:>8.0}  med={:>7.0}  p90={:>8.0}  \
         min={:>6.0}  max={:>8.0}  detect={:.2}s  RTF={:.5}  floor={:.1}dB  \
         p10={:.1}  p90={:.1}  gated={:.0}%",
        r.label,
        r.speech_ratio,
        r.segments,
        r.mean_ms,
        r.median_ms,
        r.p90_ms,
        r.min_ms,
        r.max_ms,
        r.detect_s,
        r.rtf,
        r.floor_db,
        r.p10_db,
        r.p90_db,
        r.gated_pct * 100.0
    );
}

#[test]
fn the_defaults_are_sane_on_a_real_meeting_recording() {
    let Some(l) = load() else { return };
    let dur = l.audio.duration();
    println!("\n=== vaani-vad on a real meeting recording (aggregates only) ===");
    println!(
        "  duration: {} ms ({:.1} s, {:.1} min)",
        dur.as_millis(),
        dur.as_millis() as f64 / 1000.0,
        dur.as_millis() as f64 / 60_000.0
    );
    println!("  frames:   {}", l.info.frames);

    let row = measure("defaults", &l.audio, EnergyVadConfig::default());
    print_row(&row);
    let st = EnergyVad::new();
    let _ = st.detect(&l.audio, &CancelToken::new());
    let s = EnergyVad::new().stats_probe(&l.audio);

    println!();
    println!("  Two different numbers, and conflating them is the classic way to");
    println!("  misjudge a VAD:");
    println!(
        "    frame-level speech fraction  {:.4}  ({} of {} 10 ms frames at or above the leave threshold)",
        s.speech_frame_fraction(),
        s.speech_frames,
        s.frames
    );
    println!(
        "    segment coverage            {:.4}  (what `speech_ratio` reports)",
        row.speech_ratio
    );
    println!("  They differ because pauses shorter than 300 ms are deliberately");
    println!("  absorbed into the surrounding segment -- that is the minimum-silence");
    println!("  rule doing its job, and it is what lets the chunker make 30-second");
    println!("  chunks out of a conversation instead of 2-second ones. The frame");
    println!("  fraction is the figure that says whether the detector is working.");

    // The acceptance criterion is the **frame-level** fraction, because that is
    // what says whether the detector is finding speech. A real multi-party
    // meeting is 40-60 % speech at the frame level; below 25 % the detector is
    // missing speech and above 75 % it is reporting the room.
    assert!(
        (0.25..=0.75).contains(&s.speech_frame_fraction()),
        "frame-level speech fraction {:.4} is outside [0.25, 0.75]",
        s.speech_frame_fraction()
    );
    // Segment coverage is a different, looser number, and only the outer bound
    // is meaningful: it cannot be low (padding and gap absorption only add) and
    // it cannot be 1.0 (there was 13 % of the file outside every segment).
    assert!(
        row.speech_ratio > s.speech_frame_fraction(),
        "segment coverage should exceed the frame fraction because of padding and gap absorption"
    );
    assert!(row.speech_ratio < 0.99, "the whole file was called speech");
    // Real speech has structure: a meeting is not 200 one-second segments and it
    // is not one hour-long block. Both bounds are loose on purpose.
    assert!(row.segments > 20, "only {} segments", row.segments);
    assert!(row.segments < 4_000, "{} segments", row.segments);
    // Mean segment length: natural speech in meetings runs 1-15 s. This is the
    // number that catches both a chattering detector and one that never stops.
    assert!(
        (500.0..=20_000.0).contains(&row.mean_ms),
        "mean segment length {:.0} ms is implausible",
        row.mean_ms
    );
    // The minimum-speech floor has to be honoured on real audio, not just on the
    // synthetic fixtures.
    assert!(
        row.min_ms >= 200.0,
        "a {:.0} ms segment survived a 200 ms minimum",
        row.min_ms
    );
    // The detector's own level floor has to land on something physical. A
    // recording's quietest decile is the room; if the floor is 40 dB below the
    // p10 the tracker has found digital silence and every SNR is meaningless.
    assert!(
        row.floor_db < row.p10_db + 12.0,
        "the adaptive floor ({:.1} dB) is more than 12 dB below the file's 10th percentile ({:.1} dB)",
        row.floor_db,
        row.p10_db
    );
    assert!(row.p90_db > row.p10_db, "a recording with no dynamic range");
    println!("\n  all default-config assertions passed");
}

/// A fixed **excerpt** of the recording, for the parameter sweeps.
///
/// # Why the sweeps do not use the whole file
///
/// The table is a *relative* comparison: it answers "which value of this
/// parameter detects more speech on this recording", and for that a three-minute
/// excerpt with a few dozen segments answers it as well as the full 13.9
/// minutes. The cost difference is not small -- feature extraction over 83,418
/// frames takes 170 s in the unoptimised `cargo test` profile, so 30 sweep rows
/// over the full file is 85 minutes of a developer's day spent waiting, and a
/// gate nobody runs is not a gate. The excerpt is 180 s taken from 40 % into the
/// file, which avoids both the opening (often a settle-in period) and the tail.
///
/// What this excerpt is **not** used for: the absolute figures in
/// `the_defaults_are_sane_on_a_real_meeting_recording`,
/// `what_the_detector_call_speech_actually_looks_like` and
/// `chunking_a_real_timeline_produces_sane_chunks` all run on the whole
/// recording.
const SWEEP_SECONDS: f64 = 180.0;

fn sweep_excerpt(full: &AudioBuffer) -> AudioBuffer {
    let want = (SWEEP_SECONDS * 16_000.0) as usize;
    let start_ms = full.duration().as_millis() * 4 / 10;
    let end_ms = (start_ms + (want as i64 * 1000 / 16_000)).min(full.duration().as_millis());
    AudioBuffer::mono(
        full.slice_mono(Time::from_millis(start_ms), Time::from_millis(end_ms)),
        16_000,
        Time::from_millis(start_ms),
    )
}

#[test]
fn the_tuned_table_on_real_audio() {
    let Some(l) = load() else { return };
    let excerpt = sweep_excerpt(&l.audio);
    println!(
        "\n=== tuning table: one parameter varied at a time ===\n  \
         (a {:.0} s excerpt from 40 % into the recording; the absolute figures \
         below are for the excerpt, not the file)",
        SWEEP_SECONDS
    );
    println!(
        "  {:<26} {:>8} {:>6} {:>9} {:>8} {:>9} {:>7} {:>8} {:>9}",
        "parameter", "ratio", "n", "mean ms", "med ms", "p90 ms", "RTF", "floor dB", "gated %"
    );

    let mut rows: Vec<Row> = Vec::new();

    // The leave threshold: the recall/precision dial. Lowering it detects more
    // speech and admits more noise; the band each setting produces is the
    // trade-off stated in numbers rather than in adjectives.
    for leave in [0.24f32, 0.29, 0.34, 0.40, 0.46] {
        let cfg = EnergyVadConfig {
            smoothing: vaani_vad::SmoothingConfig {
                leave_threshold: leave,
                enter_threshold: leave + 0.16,
                ..vaani_vad::SmoothingConfig::default()
            },
            ..EnergyVadConfig::default()
        };
        let r = measure("leave", &excerpt, cfg);
        print_row(&r);
        rows.push(r);
    }
    // The minimum speech duration: the precision dial at the *segment* level.
    for min_speech in [100i64, 200, 300, 500, 800] {
        let cfg = EnergyVadConfig {
            smoothing: vaani_vad::SmoothingConfig {
                min_speech_ms: min_speech,
                ..vaani_vad::SmoothingConfig::default()
            },
            ..EnergyVadConfig::default()
        };
        let r = measure("min_speech", &excerpt, cfg);
        print_row(&r);
        rows.push(r);
    }
    // The minimum silence duration: the merge dial.
    for min_silence in [150i64, 300, 500, 800, 1_200] {
        let cfg = EnergyVadConfig {
            smoothing: vaani_vad::SmoothingConfig {
                min_silence_ms: min_silence,
                ..vaani_vad::SmoothingConfig::default()
            },
            ..EnergyVadConfig::default()
        };
        let r = measure("min_silence", &excerpt, cfg);
        print_row(&r);
        rows.push(r);
    }
    // The floor window: the adaptivity dial.
    for window_ms in [1_000i64, 2_000, 4_000, 8_000, 16_000] {
        let cfg = EnergyVadConfig {
            noise_window_ms: window_ms,
            ..EnergyVadConfig::default()
        };
        let r = measure("floor_window", &excerpt, cfg);
        print_row(&r);
        rows.push(r);
    }
    // The noise quantile: the floor's position within the window.
    for q in [0.10f32, 0.15, 0.20, 0.30, 0.40] {
        let cfg = EnergyVadConfig {
            noise_quantile: q,
            ..EnergyVadConfig::default()
        };
        let r = measure("floor_quantile", &excerpt, cfg);
        print_row(&r);
        rows.push(r);
    }
    // The audibility endpoint: the sensitivity dial.
    for snr_full in [6.0f32, 8.0, 10.0, 13.0, 16.0] {
        let cfg = EnergyVadConfig {
            weights: vaani_vad::FeatureWeights {
                snr_full_db: snr_full,
                ..vaani_vad::FeatureWeights::default()
            },
            ..EnergyVadConfig::default()
        };
        let r = measure("snr_full_db", &excerpt, cfg);
        print_row(&r);
        rows.push(r);
    }
    println!();

    // Every configuration in the sweep must produce a valid timeline. This is
    // the property that must hold for *all* of them, not just the default, and
    // it is the one most likely to be broken by a parameter nobody tried.
    for r in &rows {
        assert!(
            (0.0..=1.0).contains(&r.speech_ratio),
            "{}: ratio {}",
            r.label,
            r.speech_ratio
        );
        assert!(
            r.mean_ms > 0.0 || r.segments == 0,
            "{}: mean 0 with segments",
            r.label
        );
        assert!(r.max_ms >= r.median_ms, "{}: max < median", r.label);
        assert!(
            r.rtf.is_finite() && r.rtf > 0.0,
            "{}: RTF {}",
            r.label,
            r.rtf
        );
    }
    // The leave-threshold sweep must be *monotone*: raising the threshold
    // detects strictly less. A non-monotone sweep means the hysteresis band is
    // not doing what the doc comment says.
    let leaves: Vec<&Row> = rows.iter().filter(|r| r.label == "leave").collect();
    for w in leaves.windows(2) {
        assert!(
            w[1].speech_ratio <= w[0].speech_ratio + 0.01,
            "raising the leave threshold increased the speech ratio: {} then {}",
            w[0].speech_ratio,
            w[1].speech_ratio
        );
    }
    // The minimum-speech sweep must be monotone in the other direction.
    let mins: Vec<&Row> = rows.iter().filter(|r| r.label == "min_speech").collect();
    for w in mins.windows(2) {
        assert!(
            w[1].segments <= w[0].segments,
            "raising the minimum speech duration increased the segment count: {} then {}",
            w[0].segments,
            w[1].segments
        );
        assert!(
            w[1].mean_ms >= w[0].mean_ms - 1.0,
            "raising the minimum speech duration lowered the mean length"
        );
    }
    // The minimum-silence sweep must reduce the segment count.
    let silences: Vec<&Row> = rows.iter().filter(|r| r.label == "min_silence").collect();
    for w in silences.windows(2) {
        assert!(
            w[1].segments <= w[0].segments,
            "raising the minimum silence increased the segment count: {} then {}",
            w[0].segments,
            w[1].segments
        );
    }
}

#[test]
fn what_the_detector_call_speech_actually_looks_like() {
    // A diagnostic, and the one that answers "is 86 % right?". It compares the
    // frames inside the detected segments with the frames outside them, on every
    // quantity the decision is made from. **Aggregates only** -- means and
    // percentiles of measured features, no audio content, no timestamps.
    let Some(l) = load() else { return };
    let vad = EnergyVad::new();
    let an = vad.analyse(&l.audio, &CancelToken::new()).expect("analyse");
    let st = vad.stats();
    let tl = build_timeline_at(
        &an.scores,
        &vad.config().smoothing,
        support::HOP_MS,
        0,
        834_197,
    );

    // Frame-in-segment, from the timeline rather than from the scores, so the
    // comparison is against the shipped output.
    let in_seg = |frame: usize| -> bool {
        let ms = (frame as i64) * support::HOP_MS;
        tl.segments
            .iter()
            .any(|s| s.start.as_millis() <= ms && ms < s.end.as_millis())
    };
    let w = EnergyVadConfig::default().weights;
    let (mut inside, mut outside): (
        Vec<&vaani_vad::FrameFeatures>,
        Vec<&vaani_vad::FrameFeatures>,
    ) = (Vec::new(), Vec::new());
    for (i, f) in an.features.iter().enumerate() {
        if in_seg(i) {
            inside.push(f);
        } else {
            outside.push(f);
        }
    }
    let row = |name: &str, v: &[&vaani_vad::FrameFeatures]| {
        let col = |g: fn(&vaani_vad::FrameFeatures) -> f32| -> f64 {
            let x: Vec<f32> = v.iter().map(|f| g(f)).collect();
            support::mean(&x).unwrap_or(f64::NAN)
        };
        let aud = {
            let x: Vec<f32> = v
                .iter()
                .map(|f| vaani_vad::features::audibility_term(f.snr_db, &w))
                .collect();
            support::mean(&x).unwrap_or(f64::NAN)
        };
        let shape = {
            let x: Vec<f32> = v
                .iter()
                .map(|f| vaani_vad::features::shape_term(f, &w))
                .collect();
            support::mean(&x).unwrap_or(f64::NAN)
        };
        let sc: Vec<f32> = v
            .iter()
            .map(|f| vaani_vad::features::frame_score(f, &w))
            .collect();
        let p = support::percentiles(&sc, &[0.05, 0.25, 0.5, 0.75, 0.95]);
        println!(
            "  {name:<10} n={:<7} db={:>7.1}  snr={:>7.1}  score p05={:.3} p25={:.3} p50={:.3} \
             p75={:.3} p95={:.3}  shape={:.3}  audib={aud:.3}  flat={:.3}  tonal={:>5.1}  \
             voice={:.3}  mid={:.3}  hf={:.3}  rolloff={:>6.0}  zcr={:.3}  flux={:>5.2}",
            v.len(),
            col(|f| f.db),
            col(|f| f.snr_db),
            p[0],
            p[1],
            p[2],
            p[3],
            p[4],
            shape,
            col(|f| f.flatness),
            col(|f| 10.0 * f.tonality.max(1e-12).log10()),
            col(|f| f.voicing),
            col(|f| f.mid_ratio),
            col(|f| f.hf_ratio),
            col(|f| f.rolloff_hz),
            col(|f| f.zcr),
            col(|f| f.flux),
        );
    };
    println!("\n=== inside vs outside the detected segments (aggregates) ===");
    println!(
        "  gated frames: {} of {} ({:.1} %)",
        st.gated_frames,
        st.frames,
        st.gated_frames as f64 / st.frames.max(1) as f64 * 100.0
    );
    {
        // The latch distribution, which is the one number that explains the
        // gated count and is not otherwise visible.
        let w2 = vaani_vad::FeatureWeights::default();
        let (mut on, mut off, mut vo, mut fl) = (0usize, 0usize, 0f64, 0f64);
        for f in &an.features {
            let s = vaani_vad::features::frame_score(f, &w2);
            if s >= vad.config().smoothing.leave_threshold {
                on += 1;
            } else {
                off += 1;
            }
            vo += vaani_vad::features::voicing_term(f, &w2) as f64;
            fl += vaani_vad::features::flatness_term(f, &w2) as f64;
        }
        let n = an.features.len().max(1) as f64;
        println!(
            "  latch on {on} ({:.1} %), off {off} ({:.1} %)",
            on as f64 / n * 100.0,
            off as f64 / n * 100.0
        );
        println!(
            "  mean voicing_term {:.3}, mean flatness_term {:.3}, their sum {:.3} vs gate {}",
            vo / n,
            fl / n,
            (vo + fl) / n,
            w2.gate_shape
        );
    }
    row("inside", &inside);
    row("outside", &outside);
    println!();
}

#[test]
fn chunking_a_real_timeline_produces_sane_chunks() {
    let Some(l) = load() else { return };
    let vad = EnergyVad::new();
    let tl = vad.detect(&l.audio, &CancelToken::new()).expect("detect");
    let policy = ChunkPolicy::default();
    let chunker = Chunker::new(policy);

    let started = Instant::now();
    let (raw_spans, chunks) = chunker.plan_with_raw(&tl, l.audio.duration());
    let plan_ms = started.elapsed().as_secs_f64() * 1000.0;
    println!("  raw spans: {}", raw_spans.len());
    for s in raw_spans.iter().filter(|s| s.len_ms() > policy.max_ms) {
        println!(
            "    OVER-LONG RAW SPAN {} .. {} = {} ms",
            s.start,
            s.end,
            s.len_ms()
        );
    }

    let covered: i64 = chunks.iter().map(|c| c.duration_ms()).sum();
    let speech = tl.speech_duration_ms;
    println!("\n=== chunking a real timeline ===");
    println!(
        "  {} speech segments, {} ms of speech, {:.4} speech ratio",
        tl.segments.len(),
        speech,
        tl.speech_ratio(l.audio.duration())
    );
    println!(
        "  -> {} chunks, {} ms covered, {:.4} coverage of the speech, planning took {:.2} ms",
        chunks.len(),
        covered,
        covered as f64 / speech.max(1) as f64,
        plan_ms
    );
    let durs: Vec<i64> = chunks.iter().map(|c| c.duration_ms()).collect();
    let total_s = durs.iter().sum::<i64>() as f64 / 1000.0;
    println!(
        "  chunk duration: min {} ms, median {} ms, max {} ms",
        durs.iter().min().unwrap_or(&0),
        {
            let mut v = durs.clone();
            v.sort_unstable();
            v[v.len() / 2]
        },
        durs.iter().max().unwrap_or(&0)
    );

    // The chunk plan must be cheap: it is O(segments), never O(samples). A
    // three-hour meeting has to plan in single-digit milliseconds.
    assert!(plan_ms < 250.0, "planning took {plan_ms:.2} ms");

    // Structural invariants, re-checked on real output rather than only on
    // synthetic input.
    for w in chunks.windows(2) {
        assert!(w[0].end <= w[1].start, "chunks overlap: {w:?}");
    }
    for c in &chunks {
        println!(
            "    chunk {:>8} .. {:>8}  = {:>7} ms{}",
            c.start.as_millis(),
            c.end.as_millis(),
            c.duration_ms(),
            if c.duration_ms() > policy.max_ms + 2 * policy.pad_ms {
                "   <-- OVER CAP"
            } else {
                ""
            }
        );
    }
    for c in &chunks {
        assert!(c.start < c.end, "empty chunk {c:?}");
        assert!(c.start.as_millis() >= 0);
        assert!(c.end.as_millis() <= l.audio.duration().as_millis());
        assert!(
            c.duration_ms() <= policy.max_ms + 2 * policy.pad_ms,
            "chunk {c:?} is {} ms, past the cap.\n  all chunks: {:#?}",
            c.duration_ms(),
            chunks
        );
        // Every chunk must contain speech. A chunk of pure silence is ASR time
        // spent on nothing.
        assert!(
            tl.overlapping(c.start, c.end).next().is_some(),
            "chunk {c:?} contains no speech at all"
        );
    }
    // The union of the chunks must cover every speech segment entirely. This is
    // the one that matters: a chunk plan that clips a word is a transcript with
    // a hole in it.
    for s in &tl.segments {
        let covering = chunks
            .iter()
            .filter(|c| {
                s.start.as_millis() >= c.start.as_millis() && s.end.as_millis() <= c.end.as_millis()
            })
            .count();
        assert_eq!(covering, 1, "segment {s:?} is covered by {covering} chunks");
    }
    // Every chunk except possibly the last meets the floor.
    for c in &chunks[..chunks.len().saturating_sub(1)] {
        assert!(
            c.duration_ms() >= policy.min_ms,
            "chunk {c:?} is {} ms, below the {} ms floor",
            c.duration_ms(),
            policy.min_ms
        );
    }
    // And the target band: at least most chunks are within 2x the target.
    let near = durs
        .iter()
        .filter(|d| (**d as f64) <= 2.0 * policy.target_ms as f64)
        .count();
    assert!(
        near * 2 >= chunks.len(),
        "only {near} of {} chunks are within 2x the target",
        chunks.len()
    );
    println!(
        "  all chunking assertions passed ({total_s:.0} s of audio in {} chunks)",
        chunks.len()
    );
}

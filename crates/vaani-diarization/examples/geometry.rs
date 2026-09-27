//! Ground-truth geometry and block statistics for named fixtures.
//!
//! Every number this prints comes from [`vaani_diarization::eval`], which is the
//! same code the synthetic end-to-end test scores with. A diagnostic and a test
//! that compute ground truth differently are two different tests wearing a
//! trenchcoat; that is why the labelling lives in the library.
//!
//! ```text
//! cargo run -p vaani-diarization --example geometry --            # all fixtures
//! cargo run -p vaani-diarization --example geometry -- tone       # one of them
//! ```

use vaani_core::media::AudioBuffer;
use vaani_core::{CancelToken, Time};
use vaani_diarization::config::DiarizationConfig;
use vaani_diarization::engine::HeuristicDiarizer;
use vaani_diarization::eval;
use vaani_diarization::synth::{self, Voice};

/// One named case: voices, which of them speak, and the timeline.
struct Fixture {
    name: &'static str,
    voices: Vec<Voice>,
    speakers: Vec<usize>,
    turn_ms: i64,
    turns: usize,
    noise: f32,
    /// Per-utterance gain override, to test level robustness.
    gains: Option<Vec<f32>>,
}

impl Fixture {
    fn utterances(&self) -> Vec<synth::Utterance> {
        let base = synth::alternate(&self.speakers, 0, self.turn_ms, self.turns);
        match &self.gains {
            None => base,
            Some(g) => base
                .into_iter()
                .enumerate()
                .map(|(i, u)| u.with_gain(g[i % g.len()]))
                .collect(),
        }
    }
}

/// A voice differing from `base` in one respect, for the parameter sweeps.
fn vary(base: &Voice, f0: f32, rolloff: f32) -> Voice {
    Voice {
        f0,
        harmonic_rolloff_db: rolloff,
        ..base.clone()
    }
}

fn fixtures() -> Vec<Fixture> {
    let two = synth::two_voices();
    let three = synth::three_voices();
    let low = two[0].clone();
    let high = two[1].clone();
    vec![
        Fixture {
            name: "one-speaker-long",
            voices: vec![low.clone()],
            speakers: vec![0],
            turn_ms: 4_000,
            turns: 30,
            noise: 0.003,
            gains: None,
        },
        Fixture {
            name: "two-speaker-4s-turns",
            voices: two.clone(),
            speakers: vec![0, 1],
            turn_ms: 4_000,
            turns: 30,
            noise: 0.003,
            gains: None,
        },
        Fixture {
            name: "two-speaker-2s-turns",
            voices: two.clone(),
            speakers: vec![0, 1],
            turn_ms: 2_000,
            turns: 60,
            noise: 0.003,
            gains: None,
        },
        Fixture {
            name: "three-speaker",
            voices: three,
            speakers: vec![0, 1, 2],
            turn_ms: 4_000,
            turns: 30,
            noise: 0.003,
            gains: None,
        },
        Fixture {
            // The pitch-only shortcut must not be the only defence.
            name: "same-pitch-different-tilt",
            voices: vec![vary(&low, 130.0, -11.0), vary(&high, 130.0, -5.0)],
            speakers: vec![0, 1],
            turn_ms: 4_000,
            turns: 30,
            noise: 0.003,
            gains: None,
        },
        Fixture {
            // Long turns, so that a long analysis window can still sit wholly
            // inside one speaker's turn. Without this the window-length sweep
            // measures the fixture rather than the window: at 4 s turns a 4 s
            // window always straddles a boundary, so there are no pure windows
            // and the cross-speaker population is empty.
            name: "two-speaker-12s-turns",
            voices: two.clone(),
            speakers: vec![0, 1],
            turn_ms: 12_000,
            turns: 12,
            noise: 0.003,
            gains: None,
        },
        Fixture {
            name: "level-20db",
            voices: two.clone(),
            speakers: vec![0, 1],
            turn_ms: 4_000,
            turns: 30,
            noise: 0.003,
            gains: Some(vec![1.0, 0.1]),
        },
        Fixture {
            // Code-switching: one person, two voices. There is no correct
            // answer for a speaker-diariser here beyond "do not crash", and
            // the point is to see what it actually does.
            name: "code-switching-one-person",
            voices: two.clone(),
            speakers: vec![0, 1],
            turn_ms: 4_000,
            turns: 20,
            noise: 0.003,
            gains: None,
        },
    ]
}

fn run(f: &Fixture, cfg: &DiarizationConfig) {
    let total_ms = f.turn_ms * f.turns as i64;
    let utts = f.utterances();
    let (samples, truth) = synth::render(&utts, &f.voices, total_ms, 16_000, f.noise, 2024);
    let audio = AudioBuffer::mono(samples, 16_000, Time::ZERO);
    let engine = HeuristicDiarizer::with_config(cfg.clone());
    let cancel = CancelToken::new();
    let windows = engine.debug_windows(&audio, &cancel).expect("windows");
    if windows.is_empty() {
        println!("{:<30} NO WINDOWS", f.name);
        return;
    }
    let labels = eval::label_windows(&windows, &truth, cfg.features.frame_ms(), 0);
    let g = eval::measure_geometry(&windows, &labels);
    let q = eval::measure_quality(&windows, &labels);

    println!(
        "{:<30} windows {:>3}  pure {:>3}  mixed {:>3}",
        f.name,
        windows.len(),
        g.pure_windows,
        g.mixed_windows
    );
    println!(
        "    same       n {:>5}  p10 {:>6.4}  p50 {:>6.4}  p90 {:>6.4}  max {:>6.4}",
        g.same.n, g.same.p10, g.same.p50, g.same.p90, g.same.max
    );
    println!(
        "    different  n {:>5}  min {:>6.4}  p10 {:>6.4}  p50 {:>6.4}  max {:>6.4}",
        g.different.n, g.different.min, g.different.p10, g.different.p50, g.different.max
    );
    println!(
        "    margin {:>7.4}   clusters {:>2}   purity {:.4}  recall {:.4}  complete {:.4}   (merge {:.3})",
        g.margin,
        windows.iter().map(|w| w.cluster).max().unwrap_or(0) + 1,
        q.purity,
        q.recall,
        q.complete_accuracy,
        cfg.effective_merge_distance(),
    );

    // Per-block, per-dimension across-window spread, so it is visible which
    // dimensions are live and which the MinSpread floor dropped.
    let layout = vaani_diarization::Layout::of(&cfg.features);
    let names = [
        ("cepstra", layout.cepstra, layout.delta),
        ("delta", layout.delta, layout.double_delta),
        ("ddelta", layout.double_delta, layout.pitch),
        ("pitch", layout.pitch, layout.voicedness),
        ("voicing", layout.voicedness, layout.shape),
        ("shape", layout.shape, layout.bands),
        ("bands", layout.bands, layout.ltas),
        ("ltas", layout.ltas, layout.total),
    ];
    let mut sds: Vec<f64> = Vec::new();
    for j in 0..layout.total {
        let col: Vec<f64> = windows.iter().map(|w| f64::from(w.raw[j])).collect();
        let mean = col.iter().sum::<f64>() / col.len() as f64;
        let var = col.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>()
            / (col.len() - 1).max(1) as f64;
        sds.push(var.sqrt());
    }
    let mut line = String::new();
    for (name, lo, hi) in names {
        let floor = (0..layout.total)
            .filter(|&j| j >= lo && j < hi)
            .map(|j| {
                f64::from(vaani_diarization::features::min_spread_for(
                    j,
                    &layout,
                    &cfg.min_spread,
                ))
            })
            .fold(f64::INFINITY, f64::min);
        let live = sds[lo..hi].iter().filter(|s| **s > floor).count();
        let med = {
            let mut v = sds[lo..hi].to_vec();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        line.push_str(&format!(" {name} {med:.5}({live}/{})", hi - lo));
    }
    println!("    spreads:{line}");

    // The raw value of the absolute dimensions, split by true speaker. For
    // pitch the *mean* is the signal, so a spread says nothing -- this is the
    // number that has to differ between two speakers.
    let mut by_speaker: std::collections::BTreeMap<i32, Vec<f64>> = Default::default();
    for (i, l) in labels.iter().enumerate() {
        if l.pure {
            by_speaker
                .entry(l.speaker)
                .or_default()
                .push(f64::from(windows[i].raw[layout.pitch]));
        }
    }
    let pitch_line: Vec<String> = by_speaker
        .iter()
        .map(|(sp, v)| {
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            let (lo, hi) = v
                .iter()
                .fold((f64::MAX, f64::MIN), |(a, b), x| (a.min(*x), b.max(*x)));
            format!(
                "spk {sp} n {:>3} mean {mean:>7.3} [{lo:.3},{hi:.3}]",
                v.len()
            )
        })
        .collect();
    println!("    raw log2 f0: {}", pitch_line.join("  |  "));
    if std::env::args().any(|a| a == "--pitch-dump") {
        let dump: Vec<String> = labels
            .iter()
            .enumerate()
            .filter(|(_, l)| l.pure)
            .map(|(i, l)| format!("{}:{:.2}", l.speaker, windows[i].raw[layout.pitch]))
            .collect();
        println!("    pitch by window: {}", dump.join(" "));
    }

    // Where the distance actually comes from.
    //
    // A cosine distance is the share of squared length that the two vectors do
    // *not* have in common, so it decomposes additively per dimension:
    // `1 - dot = sum_j (a_j - b_j)^2 / (2 * ||a|| * ||b||)`. Accumulating that
    // per block, separately for same-speaker and different-speaker pairs, says
    // which blocks are producing the within-speaker spread -- the thing that
    // has to come down before any merge distance can be chosen -- and which
    // blocks are producing the between-speaker signal.
    let pure: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter(|(_, l)| l.pure)
        .map(|(i, _)| i)
        .collect();
    let mut blocks: Vec<(&str, usize, usize)> = names.to_vec();
    blocks.push(("carrier", layout.total, layout.total + 1));
    let mut same_acc = vec![0f64; blocks.len()];
    let mut diff_acc = vec![0f64; blocks.len()];
    let mut same_n = 0f64;
    let mut diff_n = 0f64;
    for (x, &i) in pure.iter().enumerate() {
        for &j in &pure[x + 1..] {
            let a = &windows[i].vector;
            let b = &windows[j].vector;
            let na: f64 = a
                .iter()
                .map(|v| f64::from(*v) * f64::from(*v))
                .sum::<f64>()
                .sqrt();
            let nb: f64 = b
                .iter()
                .map(|v| f64::from(*v) * f64::from(*v))
                .sum::<f64>()
                .sqrt();
            if na <= 1e-12 || nb <= 1e-12 {
                continue;
            }
            let same = labels[i].speaker == labels[j].speaker;
            for (k, (_, lo, hi)) in blocks.iter().enumerate() {
                let mut sq = 0f64;
                for v in &a[*lo..*hi] {
                    sq += f64::from(*v) * f64::from(*v);
                }
                for v in &b[*lo..*hi] {
                    sq += f64::from(*v) * f64::from(*v);
                }
                let mut cross = 0f64;
                for u in 0..(hi - lo) {
                    cross += f64::from(a[lo + u]) * f64::from(b[lo + u]);
                }
                let share = (sq - 2.0 * cross) / (2.0 * na * nb);
                if same {
                    same_acc[k] += share;
                    same_n += 0.0;
                } else {
                    diff_acc[k] += share;
                }
            }
            if same {
                same_n += 1.0;
            } else {
                diff_n += 1.0;
            }
        }
    }
    println!(
        "    distance from, by block (mean share; same n {} diff n {})",
        same_n as usize, diff_n as usize
    );
    for (k, (name, lo, hi)) in blocks.iter().enumerate() {
        let s = if same_n > 0.0 {
            same_acc[k] / same_n
        } else {
            f64::NAN
        };
        let d = if diff_n > 0.0 {
            diff_acc[k] / diff_n
        } else {
            f64::NAN
        };
        let signal = d - s;
        let marker = if signal > 0.05 { "  <-" } else { "" };
        println!(
            "      {:<8} dims {:>2}  same {:>7.4}  different {:>7.4}  signal {:>7.4}{marker}",
            name,
            hi - lo,
            s,
            d,
            signal
        );
    }
}

/// The one-speaker within-speaker tail and the two-speaker cross-speaker floor,
/// which are the two numbers that decide whether a merge distance exists.
///
/// A merge distance is only meaningful where `1spk p90 < 2spk p10`. Reporting
/// the whole curve, rather than the best single point, is the point: it shows
/// whether a setting separates the two, or whether every setting leaves one of
/// the two numbers wrong.
struct Row {
    one_max: f64,
    one_p50: f64,
    one_p90: f64,
    two_p10: f64,
    one_clusters: usize,
    two_clusters: usize,
}

impl Row {
    /// The gap a merge distance has to fit into, if there is one.
    fn margin(&self) -> f64 {
        self.two_p10 - self.one_p90
    }
}

fn sweep_row(mutate: impl Fn(&mut DiarizationConfig)) -> Row {
    let mut one_p50 = f64::NAN;
    let mut one_max = f64::NAN;
    let mut one_p90 = f64::NAN;
    let mut two_p10 = f64::NAN;
    let mut one_clusters = 0usize;
    let mut two_clusters = 0usize;
    for (i, name) in ["one-speaker-long", "two-speaker-12s-turns"]
        .iter()
        .enumerate()
    {
        let f = fixtures()
            .into_iter()
            .find(|f| f.name == *name)
            .expect("fixture");
        let mut cfg = DiarizationConfig::default();
        mutate(&mut cfg);
        let total_ms = f.turn_ms * f.turns as i64;
        let (samples, truth) =
            synth::render(&f.utterances(), &f.voices, total_ms, 16_000, f.noise, 2024);
        let audio = AudioBuffer::mono(samples, 16_000, Time::ZERO);
        let engine = HeuristicDiarizer::with_config(cfg.clone());
        let windows = engine
            .debug_windows(&audio, &CancelToken::new())
            .expect("windows");
        if windows.is_empty() {
            continue;
        }
        let labels = eval::label_windows(&windows, &truth, cfg.features.frame_ms(), 0);
        let g = eval::measure_geometry(&windows, &labels);
        let clusters = windows.iter().map(|w| w.cluster).max().unwrap_or(0) + 1;
        if i == 0 {
            one_p50 = g.same.p50;
            one_p90 = g.same.p90;
            one_max = g.same.max;
            one_clusters = clusters;
        } else {
            two_p10 = g.different.p10;
            two_clusters = clusters;
        }
    }
    Row {
        one_max,
        one_p50,
        one_p90,
        two_p10,
        one_clusters,
        two_clusters,
    }
}

fn sweep_smoothing(halves: &[usize]) {
    println!(
        "{:>5}  {:>9} {:>9}  {:>9}  {:>8} {:>8}  {:>8}",
        "half", "1spk p50", "1spk p90", "2spk p10", "margin", "1 clus", "2 clus"
    );
    for &half in halves {
        let r = sweep_row(|cfg| cfg.clusters.feature_smooth_half_windows = half);
        println!(
            "{half:>5}  {:>9.4} {:>9.4}  {:>9.4}  {:>8.4} {:>8} {:>8}",
            r.one_p50,
            r.one_p90,
            r.two_p10,
            r.margin(),
            r.one_clusters,
            r.two_clusters
        );
    }
}

/// Sweep the minimum speech ratio a window must have to be clustered at all.
///
/// The within-speaker distance distribution is bimodal, with a median near zero
/// and a p90 at the maximum: that is a few outlier windows rather than broad
/// noise, and a quarter of the pairs are dragged by them. The hypothesis worth
/// testing is that the outliers are windows which are mostly silence, where the
/// median is taken over a handful of frames and describes almost nothing.
fn sweep_speech(ratios: &[f32]) {
    println!(
        "{:>6}  {:>9} {:>9} {:>9}  {:>9}  {:>8} {:>8} {:>8}",
        "ratio", "1spk p50", "1spk p90", "1spk max", "2spk p10", "margin", "1 clus", "2 clus"
    );
    for &ratio in ratios {
        let r = sweep_row(|cfg| cfg.clusters.min_window_speech_ratio = ratio);
        println!(
            "{ratio:>6.2}  {:>9.4} {:>9.4} {:>9.4}  {:>9.4}  {:>8.4} {:>8} {:>8}",
            r.one_p50,
            r.one_p90,
            r.one_max,
            r.two_p10,
            r.margin(),
            r.one_clusters,
            r.two_clusters
        );
    }
}

/// Sweep the carrier magnitude: the evidence requirement.
///
/// The one dial that changes the absolute distance scale rather than the
/// relative ordering of the two populations, because the carrier is the only
/// part of the vector that is not scaled with the features.
fn sweep_carrier(carriers: &[f32]) {
    println!(
        "{:>7}  {:>9} {:>9}  {:>9}  {:>8} {:>8} {:>8}",
        "carrier", "1spk p50", "1spk p90", "2spk p10", "margin", "1 clus", "2 clus"
    );
    for &c in carriers {
        let r = sweep_row(|cfg| cfg.features.weights.carrier = c);
        println!(
            "{c:>7.2}  {:>9.4} {:>9.4}  {:>9.4}  {:>8.4} {:>8} {:>8}",
            r.one_p50,
            r.one_p90,
            r.two_p10,
            r.margin(),
            r.one_clusters,
            r.two_clusters
        );
    }
}

/// Sweep the analysis window length, in frames. 150 frames is 1.5 s.
///
/// The lever that matters. A window is summarised by a median over its frames,
/// so a longer window averages over proportionally more phonemes: speaker
/// identity is constant across the whole window and survives, the vowel
/// sequence has a ~200 ms period and averages away. The within-speaker *tail*
/// is what has to come down -- it is the pairs of windows that happened to catch
/// different vowels -- and window length is the direct attack on it.
fn sweep_window(frames: &[usize]) {
    println!(
        "{:>6} {:>5}  {:>9} {:>9}  {:>9}  {:>8} {:>8} {:>8}",
        "frames", "secs", "1spk p50", "1spk p90", "2spk p10", "margin", "1 clus", "2 clus"
    );
    for &f in frames {
        let r = sweep_row(|cfg| {
            cfg.clusters.window_frames = f;
            cfg.clusters.window_hop_frames = (f / 2).max(1);
        });
        println!(
            "{f:>6} {:>5.2}  {:>9.4} {:>9.4}  {:>9.4}  {:>8.4} {:>8} {:>8}",
            f as f64 / 100.0,
            r.one_p50,
            r.one_p90,
            r.two_p10,
            r.margin(),
            r.one_clusters,
            r.two_clusters
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--drift") {
        // One speaker, intonation rising 110 -> 150 Hz inside every 4 s turn.
        // The question is which block spreads, because the answer decides
        // whether this is fixable.
        let base = synth::two_voices()[0].clone();
        let utterances = synth::alternate(&[0], 0, 4_000, 24);
        let total_ms = 4_000 * 24;
        let mut samples = vec![0f32; (total_ms * 16_000 / 1_000) as usize];
        let mut truth = vec![-1i32; (total_ms / 10) as usize];
        for (i, u) in utterances.iter().enumerate() {
            let mut rng = synth::Rng::new(2024 + i as u64);
            let lo = (u.start_ms * 16_000 / 1_000) as usize;
            let hi = (u.end_ms * 16_000 / 1_000) as usize;
            let rendered = synth::render_glide(&base, 110.0, 150.0, hi - lo, 16_000, &mut rng);
            for (k, v) in rendered.iter().enumerate() {
                samples[lo + k] = *v * u.gain * 0.5;
            }
            let range = (u.start_ms / 10) as usize..(u.end_ms / 10) as usize;
            for slot in &mut truth[range] {
                *slot = 0;
            }
        }
        let cfg = DiarizationConfig::default();
        let audio = AudioBuffer::mono(samples, 16_000, Time::ZERO);
        let engine = HeuristicDiarizer::with_config(cfg.clone());
        let windows = engine
            .debug_windows(&audio, &CancelToken::new())
            .expect("windows");
        let labels = eval::label_windows(&windows, &truth, cfg.features.frame_ms(), 0);
        let g = eval::measure_geometry(&windows, &labels);
        println!("drift 110->150 Hz, one speaker");
        println!(
            "    same n {} p50 {:.4} p90 {:.4} max {:.4}",
            g.same.n, g.same.p50, g.same.p90, g.same.max
        );
        let layout = vaani_diarization::Layout::of(&cfg.features);
        let names = [
            ("cepstra", layout.cepstra, layout.delta),
            ("delta", layout.delta, layout.double_delta),
            ("ddelta", layout.double_delta, layout.pitch),
            ("pitch", layout.pitch, layout.voicedness),
            ("voicing", layout.voicedness, layout.shape),
            ("shape", layout.shape, layout.bands),
            ("bands", layout.bands, layout.ltas),
            ("ltas", layout.ltas, layout.total),
        ];
        let pure: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, l)| l.pure)
            .map(|(i, _)| i)
            .collect();
        let mut acc = vec![0f64; names.len()];
        let mut n = 0f64;
        for (x, &i) in pure.iter().enumerate() {
            for &j in &pure[x + 1..] {
                let a = &windows[i].vector;
                let b = &windows[j].vector;
                let na: f64 = a
                    .iter()
                    .map(|v| f64::from(*v) * f64::from(*v))
                    .sum::<f64>()
                    .sqrt();
                let nb: f64 = b
                    .iter()
                    .map(|v| f64::from(*v) * f64::from(*v))
                    .sum::<f64>()
                    .sqrt();
                if na <= 1e-12 || nb <= 1e-12 {
                    continue;
                }
                for (k, (_, lo, hi)) in names.iter().enumerate() {
                    let mut sq = 0f64;
                    for v in &a[*lo..*hi] {
                        sq += f64::from(*v) * f64::from(*v);
                    }
                    for v in &b[*lo..*hi] {
                        sq += f64::from(*v) * f64::from(*v);
                    }
                    let mut cross = 0f64;
                    for u in 0..(hi - lo) {
                        cross += f64::from(a[lo + u]) * f64::from(b[lo + u]);
                    }
                    acc[k] += (sq - 2.0 * cross) / (2.0 * na * nb);
                }
                n += 1.0;
            }
        }
        println!("    where the within-speaker distance comes from (one speaker, so all noise):");
        for (k, (name, _, _)) in names.iter().enumerate() {
            let share = acc[k] / n.max(1.0);
            println!("      {name:<8} {:.4}", share);
        }
        return;
    }
    if args.iter().any(|a| a == "--sweep-smooth") {
        sweep_smoothing(&[0, 1, 2, 3, 4, 6, 8, 10, 12]);
        return;
    }
    if args.iter().any(|a| a == "--sweep-speech") {
        sweep_speech(&[0.4, 0.6, 0.75, 0.85, 0.9, 0.95]);
        return;
    }
    if args.iter().any(|a| a == "--sweep-carrier") {
        sweep_carrier(&[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]);
        return;
    }
    if args.iter().any(|a| a == "--sweep-window") {
        sweep_window(&[75, 100, 150, 200, 250, 300, 400, 500]);
        return;
    }
    let only = args.get(1).cloned();
    let cfg = DiarizationConfig::default();
    println!(
        "merge {:.3}  window {:.1}s  smooth half {}",
        cfg.effective_merge_distance(),
        cfg.clusters.window_frames as f64 / 100.0,
        cfg.clusters.feature_smooth_half_windows
    );
    for f in fixtures() {
        if only.as_deref().is_some_and(|o| !f.name.contains(o)) {
            continue;
        }
        run(&f, &cfg);
    }
}

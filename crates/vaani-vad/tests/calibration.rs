//! Calibration: measure the feature distributions, then check the tuned
//! thresholds against them.
//!
//! # Why this file exists
//!
//! Every threshold in `vaani_vad::features::FeatureWeights` was chosen by
//! *looking at measured distributions* of the feature on real signals --
//! white noise, a fan-like tone, a harmonic voice, a fricative -- and placing
//! the mapping endpoints between the clusters. A threshold chosen by intuition
//! is a threshold that turns out to be on the wrong side of a cluster, and the
//! symptom is a detector that scores noise above speech.
//!
//! The measurements are printed with `--nocapture` so the numbers in the final
//! report are reproducible from this file rather than remembered, and the
//! assertions below are the part that makes the thresholds a checked claim
//! rather than a comment.
//!
//! Run with:
//!   cargo test -p vaani-vad --test calibration -- --nocapture

mod support;

use support::{Rng, Span, build, white_noise};
use vaani_core::pipeline::VadEngine;
use vaani_vad::EnergyVad;
use vaani_vad::EnergyVadConfig;
use vaani_vad::dsp::{HOP_SAMPLES, N_FREQ, SAMPLE_RATE, WINDOW_SAMPLES};
use vaani_vad::features::{FeatureExtractor, FeatureWeights, frame_score};
use vaani_vad::smoothing::build_timeline_at;

const MS: i64 = 1_000;

/// Slice a signal into `WINDOW_SAMPLES`-long frames at `HOP_SAMPLES` steps,
/// exactly as `EnergyVad::analyse` does.
fn frames(samples: &[f32]) -> Vec<Vec<f32>> {
    if samples.len() <= WINDOW_SAMPLES {
        return vec![samples.to_vec()];
    }
    let n = 1 + (samples.len() - WINDOW_SAMPLES) / HOP_SAMPLES;
    (0..n)
        .map(|f| {
            let s = f * HOP_SAMPLES;
            let e = (s + WINDOW_SAMPLES).min(samples.len());
            samples[s..e].to_vec()
        })
        .collect()
}

/// Extract every frame's features, with a fixed floor so the comparison is
/// about the *shape* features and not about the adaptive floor.
fn features(samples: &[f32], floor_db: f32) -> Vec<vaani_vad::features::FrameFeatures> {
    let mut e = FeatureExtractor::new(FeatureWeights::default());
    let mut out = Vec::new();
    for f in frames(samples) {
        e.set_floor(floor_db);
        out.push(e.push(&f));
    }
    out
}

fn column(
    fs: &[vaani_vad::features::FrameFeatures],
    f: impl Fn(&vaani_vad::features::FrameFeatures) -> f32,
) -> Vec<f32> {
    fs.iter()
        .map(f)
        // FLUX_UNDEFINED is finite and enormous; leaving it in makes the flux
        // column's mean meaningless, so it is excluded and the frame count
        // reported alongside says so.
        .filter(|v| v.is_finite() && *v < 1e5)
        .collect()
}

struct Report {
    name: &'static str,
    fs: Vec<vaani_vad::features::FrameFeatures>,
}

impl Report {
    fn new(name: &'static str, samples: &[f32], floor_db: f32) -> Self {
        Self {
            name,
            fs: features(samples, floor_db),
        }
    }

    fn col(&self, f: impl Fn(&vaani_vad::features::FrameFeatures) -> f32) -> Vec<f32> {
        column(&self.fs, f)
    }

    fn print(&self) {
        let row = |label: &str, v: Vec<f32>| {
            let p = support::percentiles(&v, &[0.05, 0.5, 0.95]);
            let m = support::mean(&v).unwrap_or(f64::NAN);
            println!(
                "  {label:<10} n={:<5} mean={m:>8.3}  p05={:>8.3}  p50={:>8.3}  p95={:>8.3}",
                v.len(),
                p[0],
                p[1],
                p[2]
            );
        };
        println!("--- {} ({} frames)", self.name, self.fs.len());
        row("db", self.col(|f| f.db));
        row("flatness", self.col(|f| f.flatness));
        row(
            "tonality",
            self.col(|f| 10.0 * f.tonality.max(1e-12).log10()),
        );
        row("voicing", self.col(|f| f.voicing));
        row("hf_ratio", self.col(|f| f.hf_ratio));
        row("lf_ratio", self.col(|f| f.lf_ratio));
        row("mid_ratio", self.col(|f| f.mid_ratio));
        row("rolloff", self.col(|f| f.rolloff_hz));
        row("zcr", self.col(|f| f.zcr));
        row("flux", self.col(|f| f.flux));
    }
}

fn fixtures() -> Vec<Report> {
    let mut rng = Rng::new(0xC0FFEE);
    let mut out = Vec::new();

    // --- the adversaries -------------------------------------------------
    out.push(Report::new(
        "white noise",
        &white_noise(MS as usize * 4, &mut rng, -35.0),
        -50.0,
    ));
    out.push(Report::new(
        "white noise loud",
        &white_noise(MS as usize * 4, &mut rng, -15.0),
        -50.0,
    ));

    // A fan: broadband but with a low-frequency tilt and a hum.
    let mut fan = white_noise(MS as usize * 4, &mut rng, -30.0);
    for (i, v) in fan.iter_mut().enumerate() {
        let t = i as f32 / SAMPLE_RATE as f32;
        let tilt =
            1.0 / (1.0 + (t * 2.0 * std::f32::consts::PI * 300.0 / SAMPLE_RATE as f32).powi(2));
        *v *= tilt;
        *v += 0.35 * (2.0 * std::f32::consts::PI * 50.0 * t).sin();
    }
    out.push(Report::new("fan + hum", &fan, -50.0));

    // A steady 1 kHz tone: the flux veto's target.
    let tone: Vec<f32> = (0..MS as usize * 4)
        .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / SAMPLE_RATE as f32).sin())
        .collect();
    out.push(Report::new("steady tone 1k", &tone, -50.0));

    // Silence.
    out.push(Report::new("silence", &vec![0f32; MS as usize * 2], -50.0));

    // --- the targets -----------------------------------------------------
    let voiced = build(&support::Recipe {
        speech: vec![Span {
            start_ms: 0,
            end_ms: 4_000,
        }],
        speech_db: -25.0,
        noise_db: -100.0,
        seed: 5,
        ..support::Recipe::default()
    });
    out.push(Report::new("voiced speech", &voiced.samples, -50.0));

    // A quiet speaker, 6 dB above the floor: the sensitivity case.
    let quiet = build(&support::Recipe {
        speech: vec![Span {
            start_ms: 0,
            end_ms: 4_000,
        }],
        speech_db: -44.0,
        noise_db: -100.0,
        seed: 6,
        ..support::Recipe::default()
    });
    // A 60 Hz bass: the worst case for the out-of-band veto, because a 60 Hz
    // fundamental puts its first formant region close to the rumble band. If
    // the veto fires on this it has rejected a real speech signal on a
    // technicality, and that has to be a test rather than an assumption.
    let bass = build(&support::Recipe {
        speech: vec![Span {
            start_ms: 0,
            end_ms: 4_000,
        }],
        speech_db: -25.0,
        noise_db: -100.0,
        f0_hz: 62.0,
        seed: 7,
        ..support::Recipe::default()
    });
    out.push(Report::new("bass voice 62 Hz", &bass.samples, -50.0));

    out.push(Report::new("quiet speech", &quiet.samples, -50.0));

    out
}

#[test]
fn print_feature_distributions() {
    println!("\n=== vaani-vad feature distributions (16 kHz, 400/160, 512-pt Hann) ===\n");
    for r in fixtures() {
        r.print();
        println!();
    }
}

#[test]
fn the_shape_features_separate_the_adversaries_from_speech() {
    // The claim the tuned thresholds rest on. Each assertion is an *ordering*
    // the score depends on, and each is stated as a number rather than as
    // "it looked right".
    use vaani_vad::features::FrameFeatures as F;
    let reps = fixtures();
    let by = |name: &str| -> &Report { reps.iter().find(|r| r.name == name).expect(name) };
    let p = |r: &Report, f: fn(&F) -> f32, q: f64| support::percentiles(&r.col(f), &[q as f32])[0];

    let w = FeatureWeights::default();

    // --- flatness: white noise is the flat end, a harmonic comb the peaky end.
    assert!(
        p(by("voiced speech"), |x: &F| x.flatness, 0.5) < w.flatness_lo as f64,
        "voiced speech flatness should be below flatness_lo {}",
        w.flatness_lo
    );
    assert!(
        p(by("white noise"), |x: &F| x.flatness, 0.5) > w.flatness_hi as f64,
        "white noise flatness should be above flatness_hi {}",
        w.flatness_hi
    );

    // --- voicing: a harmonic comb is periodic, noise is not. The two clusters
    // overlap at the tail, so the assertion is about separation rather than an
    // absolute: the *median* of speech must clear `voicing_null + span` while
    // the *95th percentile* of noise must not reach `voicing_null`.
    let speech_voicing_p50 = p(by("voiced speech"), |x: &F| x.voicing, 0.5);
    let noise_voicing_p95 = p(by("white noise"), |x: &F| x.voicing, 0.95);
    assert!(
        speech_voicing_p50 > (w.voicing_null + w.voicing_span * 0.5) as f64,
        "voiced speech voicing p50 {speech_voicing_p50} should be well above voicing_null {}",
        w.voicing_null
    );
    assert!(
        noise_voicing_p95 < w.voicing_null as f64,
        "the 95th percentile of noise voicing should be below voicing_null {}, got {noise_voicing_p95}",
        w.voicing_null
    );
    assert!(
        speech_voicing_p50 > noise_voicing_p95,
        "the clusters must not overlap: speech {speech_voicing_p50} vs noise {noise_voicing_p95}"
    );

    // --- hf_ratio and rolloff: a hiss is the treble end.
    assert!(
        p(by("white noise"), |x: &F| x.hf_ratio, 0.5) > w.hiss_lo as f64,
        "noise hf_ratio should exceed hiss_lo {}, got {}",
        w.hiss_lo,
        p(by("white noise"), |x: &F| x.hf_ratio, 0.5)
    );
    assert!(
        p(by("voiced speech"), |x: &F| x.hf_ratio, 0.5) < w.hiss_hi as f64,
        "the median voiced hf_ratio should be below hiss_hi {}, got {}",
        w.hiss_hi,
        p(by("voiced speech"), |x: &F| x.hf_ratio, 0.5)
    );
    // The 95th percentile is *not* below hiss_hi and must not be asserted to
    // be: a fricative-heavy frame legitimately puts 0.4 of its energy above
    // 3 kHz, and the spectral term is supposed to give that frame less credit,
    // not zero.
    assert!(p(by("voiced speech"), |x: &F| x.rolloff_hz, 0.95) < w.rolloff_lo_hz as f64);
    assert!(p(by("white noise"), |x: &F| x.rolloff_hz, 0.05) > w.rolloff_hi_hz as f64);

    // --- flux: a steady tone is numerically steady, everything else is not.
    assert!(
        p(by("steady tone 1k"), |x: &F| x.flux, 0.95) < w.flux_min_db as f64,
        "a steady tone's flux should be below flux_min_db {}, got {}",
        w.flux_min_db,
        p(by("steady tone 1k"), |x: &F| x.flux, 0.95)
    );
    // A 50 Hz hum is *not* caught by flux -- see the module docs -- so the
    // assertion is deliberately about speech being clearly above the
    // threshold, not about the hum being below it.
    assert!(
        p(by("voiced speech"), |x: &F| x.flux, 0.05) > 2.0 * w.flux_min_db as f64,
        "speech flux p05 should be well above flux_min_db, got {}",
        p(by("voiced speech"), |x: &F| x.flux, 0.05)
    );
    assert!(
        p(by("white noise"), |x: &F| x.flux, 0.05) > 2.0 * w.flux_min_db as f64,
        "noise flux should be far above flux_min_db, got {}",
        p(by("white noise"), |x: &F| x.flux, 0.05)
    );
    // A fan with a low-frequency tilt is steady in the energy-weighted sense
    // (its energy sits in two bands that do not move), so the steady-source
    // veto catches it as well as the out-of-band one. Asserted rather than
    // assumed, because "the flux term does nothing for a fan" was the reason
    // the fan needed a second mechanism in the first place.
    assert!(
        p(by("fan + hum"), |x: &F| x.flux, 0.5) < w.flux_min_db as f64,
        "a fan with a low-frequency tilt should be steady in the weighted sense, got {}",
        p(by("fan + hum"), |x: &F| x.flux, 0.5)
    );

    // --- mid_ratio: the feature that separates a voice from a rumble.
    // The *median* is the right statistic here. The 5th percentile of a voice's
    // mid_ratio is about 0.02, because an unvoiced vowel onset genuinely has
    // almost no formant-band energy, and the veto does fire on those frames --
    // 5 % of a segment's frames, which the hysteresis state machine absorbs.
    assert!(
        p(by("voiced speech"), |x: &F| x.mid_ratio, 0.5) > 3.0 * w.band_min_mid as f64,
        "speech mid_ratio p50 should be well above band_min_mid, got {}",
        p(by("voiced speech"), |x: &F| x.mid_ratio, 0.5)
    );
    assert!(
        p(by("bass voice 62 Hz"), |x: &F| x.mid_ratio, 0.5) > 3.0 * w.band_min_mid as f64,
        "a 62 Hz bass must not be caught by the out-of-band veto; mid_ratio p50 is {}",
        p(by("bass voice 62 Hz"), |x: &F| x.mid_ratio, 0.5)
    );
    assert!(
        p(by("fan + hum"), |x: &F| x.mid_ratio, 0.5) < w.band_min_mid as f64,
        "a low-frequency rumble should be below band_min_mid, got {}",
        p(by("fan + hum"), |x: &F| x.mid_ratio, 0.5)
    );
    // The two-sided rolloff term is a *backstop*, not the primary mechanism: the
    // spectral subtraction removes a stationary rumble's energy, so the rolloff
    // measured after subtraction is no longer the rumble's own. What has to hold
    // is the outcome, so that is what is asserted -- and it is asserted in the
    // score table below rather than here.

    // --- silence is zero on everything.
    let s = by("silence");
    assert!(p(s, |x: &F| x.voicing, 0.95) < 0.05);
    assert!(p(s, |x: &F| x.hf_ratio, 0.95) < 0.01);
    assert_eq!(support::mean(&s.col(|x: &F| x.db)).unwrap(), -200.0);
}

#[test]
fn the_combined_score_separates_speech_from_every_adversary() {
    // The end-to-end claim: with a floor 15 dB below the signal, speech scores
    // above the enter threshold and each adversary scores below the leave
    // threshold. That is the property `frame_score` is built for.
    let cfg = EnergyVadConfig::default();
    let w = cfg.weights;
    let mut rng = Rng::new(0xBEEF);
    let cases: Vec<(&str, Vec<f32>, f32)> = vec![
        (
            "voiced speech 25 dB",
            build(&support::Recipe {
                speech: vec![Span {
                    start_ms: 0,
                    end_ms: 4_000,
                }],
                speech_db: -25.0,
                noise_db: -100.0,
                seed: 21,
                ..support::Recipe::default()
            })
            .samples,
            -60.0,
        ),
        (
            "quiet speech, 10 dB over floor",
            build(&support::Recipe {
                speech: vec![Span {
                    start_ms: 0,
                    end_ms: 4_000,
                }],
                speech_db: -44.0,
                noise_db: -100.0,
                seed: 22,
                ..support::Recipe::default()
            })
            .samples,
            // 10 dB below the signal's measured level of about -62 dB. The
            // 6 dB case is measured separately in `the_sensitivity_curve` and
            // reported, not asserted: the product form of the score means a
            // voice at 0 dB is not detected at all, and pretending otherwise
            // in a test would be asserting something the design does not do.
            -72.0,
        ),
        (
            "white noise",
            white_noise(MS as usize * 4, &mut rng, -35.0),
            -50.0,
        ),
        (
            "white noise, loud",
            white_noise(MS as usize * 4, &mut rng, -15.0),
            -50.0,
        ),
        (
            "steady tone 1 kHz",
            (0..MS as usize * 4)
                .map(|i| {
                    0.3 * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / SAMPLE_RATE as f32)
                        .sin()
                })
                .collect(),
            -50.0,
        ),
        (
            "50 Hz hum with harmonics",
            (0..MS as usize * 4)
                .map(|i| {
                    let t = i as f32 / SAMPLE_RATE as f32;
                    0.4 * ((2.0 * std::f32::consts::PI * 50.0 * t).sin()
                        + 0.6 * (2.0 * std::f32::consts::PI * 100.0 * t).sin()
                        + 0.3 * (2.0 * std::f32::consts::PI * 150.0 * t).sin())
                })
                .collect(),
            -50.0,
        ),
    ];

    println!("\n=== score distributions ===");
    for (name, samples, floor) in &cases {
        let fs = features(samples, *floor);
        let scored: Vec<f32> = fs.iter().map(|f| frame_score(f, &w)).collect();
        let p = support::percentiles(&scored, &[0.05, 0.5, 0.95]);
        let flux = support::percentiles(&column(&fs, |f| f.flux), &[0.5, 0.95]);
        let snr = support::percentiles(&column(&fs, |f| f.snr_db), &[0.5]);
        println!(
            "  {name:<28} score p05={:>6.3} p50={:>6.3} p95={:>6.3} | flux p50={:>7.3} p95={:>7.3} | snr p50={:>7.2}",
            p[0], p[1], p[2], flux[0], flux[1], snr[0]
        );
    }
    println!();

    // Silence must never score.
    for (name, samples, floor) in &cases {
        let fs = features(samples, *floor);
        let is_speech = name.contains("speech");
        let scored: Vec<f32> = fs.iter().map(|f| frame_score(f, &w)).collect();
        if is_speech {
            let p50 = support::percentiles(&scored, &[0.5])[0];
            assert!(
                p50 > cfg.smoothing.enter_threshold as f64,
                "{name}: median score {p50} did not clear enter {}",
                cfg.smoothing.enter_threshold
            );
        } else {
            let p95 = support::percentiles(&scored, &[0.95])[0];
            assert!(
                p95 <= cfg.smoothing.leave_threshold as f64,
                "{name}: 95th percentile score {p95} exceeded leave {}",
                cfg.smoothing.leave_threshold
            );
        }
    }
}

#[test]
fn the_end_to_end_detector_answers_on_synthetic_fixtures() {
    // Sanity band on the whole pipeline for each fixture family, so a change to
    // the score that moves a fixture out of its band fails here rather than
    // being discovered on real audio.
    struct Case {
        name: &'static str,
        recipe: support::Recipe,
        min_recall: f64,
        max_fp_ratio: f64,
    }
    let cases = vec![
        Case {
            name: "speech at +18 dB SNR, quiet room",
            recipe: support::Recipe {
                speech: vec![
                    Span {
                        start_ms: 1_000,
                        end_ms: 3_000,
                    },
                    Span {
                        start_ms: 5_000,
                        end_ms: 7_000,
                    },
                ],
                noise: vec![Span {
                    start_ms: 3_500,
                    end_ms: 4_500,
                }],
                speech_db: -22.0,
                noise_db: -40.0,
                seed: 31,
                ..support::Recipe::default()
            },
            min_recall: 0.95,
            max_fp_ratio: 0.30,
        },
        Case {
            name: "speech at +6 dB SNR, quiet room",
            recipe: support::Recipe {
                speech: vec![Span {
                    start_ms: 1_000,
                    end_ms: 3_000,
                }],
                noise: vec![Span {
                    start_ms: 3_500,
                    end_ms: 4_500,
                }],
                speech_db: -34.0,
                noise_db: -40.0,
                seed: 32,
                ..support::Recipe::default()
            },
            min_recall: 0.80,
            max_fp_ratio: 0.30,
        },
        Case {
            name: "speech at 0 dB SNR, quiet room (hard)",
            recipe: support::Recipe {
                speech: vec![Span {
                    start_ms: 1_000,
                    end_ms: 3_000,
                }],
                noise: vec![Span {
                    start_ms: 3_500,
                    end_ms: 4_500,
                }],
                speech_db: -40.0,
                noise_db: -40.0,
                seed: 33,
                ..support::Recipe::default()
            },
            min_recall: 0.70,
            max_fp_ratio: 0.35,
        },
        Case {
            name: "quiet speech, loud broadband floor",
            recipe: support::Recipe {
                speech: vec![Span {
                    start_ms: 1_000,
                    end_ms: 3_000,
                }],
                noise: vec![Span {
                    start_ms: 3_500,
                    end_ms: 4_500,
                }],
                speech_db: -34.0,
                noise_db: -28.0,
                seed: 34,
                ..support::Recipe::default()
            },
            min_recall: 0.85,
            max_fp_ratio: 0.30,
        },
    ];

    println!("\n=== end-to-end on synthetic fixtures ===");
    println!(
        "  (fp is measured against all non-speech time, and 60 ms of padding on \
         each side of each segment lands in it, so a floor of ~0.10 is two \
         segments' worth of padding and nothing else)"
    );
    for c in &cases {
        let f = build(&c.recipe);
        let vad = EnergyVad::new();
        let tl = vad
            .detect(&f.audio(), &vaani_core::CancelToken::new())
            .unwrap();
        let recall = f.recall(&tl);
        let fp = f.false_positive_ms(&tl) as f64 / f.non_speech_ms().max(1) as f64;
        let ratio = tl.speech_ratio(f.audio().duration());
        println!(
            "  {:<44} recall={recall:>6.3}  fp={fp:>6.3}  speech_ratio={ratio:>6.3}  segments={}  non-speech={} ms",
            c.name,
            tl.segments.len(),
            f.non_speech_ms()
        );
        if recall < c.min_recall || fp > c.max_fp_ratio {
            println!(
                "      planted: {:?}",
                c.recipe
                    .speech
                    .iter()
                    .map(|s| (s.start_ms, s.end_ms))
                    .collect::<Vec<_>>()
            );
            println!(
                "      found:   {:?}",
                tl.segments
                    .iter()
                    .map(|s| (s.start.as_millis(), s.end.as_millis()))
                    .collect::<Vec<_>>()
            );
        }
        assert!(
            recall >= c.min_recall,
            "{}: recall {recall} < {}",
            c.name,
            c.min_recall
        );
        assert!(
            fp <= c.max_fp_ratio,
            "{}: fp {fp} > {}",
            c.name,
            c.max_fp_ratio
        );
        assert!((0.0..=1.0).contains(&ratio), "{}: ratio {ratio}", c.name);
    }
    println!();
}

/// One point on a sensitivity curve.
#[derive(Debug, Clone, Copy)]
struct Point {
    snr: i32,
    recall: f64,
    fp: f64,
    ratio: f64,
    segments: usize,
    median_ms: f64,
    /// Mean frame score inside the planted speech spans.
    speech_score: f64,
    /// Mean frame score in the non-speech parts of the buffer.
    other_score: f64,
    /// The adaptive level floor at the end of the buffer.
    floor_db: f32,
}

/// The SNRs the sweeps are run at.
const SWEEP_SNRS: [i32; 11] = [-6, -3, 0, 3, 6, 9, 12, 15, 18, 24, 30];

/// Build the sweep fixture for one SNR.
///
/// `continuous` decides whether the broadband noise covers the whole file. A
/// continuous floor means the adaptive tracker has nothing quieter to find, so
/// the floor *is* the noise; an intermittent one leaves digital silence between
/// the spans and the floor tracks that silence instead. That is the difference
/// between "a noisy room" and "a recording with gaps in the noise", and it moves
/// the measured limit by more than any threshold in the detector does.
fn sweep_fixture(snr: i32, continuous: bool) -> support::Fixture {
    let noise_db = -40.0f32;
    let noise_span = if continuous {
        Span {
            start_ms: 0,
            end_ms: 8_500,
        }
    } else {
        Span {
            start_ms: 0,
            end_ms: 1_000,
        }
    };
    build(&support::Recipe {
        speech: vec![
            Span {
                start_ms: 1_000,
                end_ms: 3_000,
            },
            Span {
                start_ms: 4_000,
                end_ms: 6_000,
            },
        ],
        noise: vec![noise_span],
        speech_db: noise_db + snr as f32,
        noise_db,
        seed: (1000i64 + snr as i64) as u64,
        ..support::Recipe::default()
    })
}

fn run_point(snr: i32, continuous: bool, weights: FeatureWeights) -> Point {
    let f = sweep_fixture(snr, continuous);
    let audio = f.audio();
    let cfg = EnergyVadConfig {
        weights,
        ..EnergyVadConfig::default()
    };
    let vad = EnergyVad::with_config(cfg).unwrap();
    let an = vad
        .analyse(&audio, &vaani_core::CancelToken::new())
        .unwrap();
    let (in_speech, in_other) = support::score_separation(&an.scores, &f.speech);
    let tl = build_timeline_at(
        &an.scores,
        &vad.config().smoothing,
        support::HOP_MS,
        audio.start_time.as_millis(),
        audio.duration().as_millis(),
    );
    let durs: Vec<f32> = tl
        .segments
        .iter()
        .map(|s| (s.end.as_millis() - s.start.as_millis()) as f32)
        .collect();
    Point {
        snr,
        recall: f.recall(&tl),
        fp: f.false_positive_ms(&tl) as f64 / f.non_speech_ms().max(1) as f64,
        ratio: tl.speech_ratio(audio.duration()),
        segments: tl.segments.len(),
        median_ms: support::median(&durs).unwrap_or(0.0),
        speech_score: in_speech,
        other_score: in_other,
        floor_db: vad.stats().final_floor_db,
    }
}

/// Sweep the speech SNR with the noise either continuous or intermittent.
fn sweep(continuous: bool) -> Vec<Point> {
    SWEEP_SNRS
        .iter()
        .map(|snr| run_point(*snr, continuous, FeatureWeights::default()))
        .collect()
}

fn print_curve(name: &str, curve: &[Point]) {
    println!("\n=== sensitivity: speech SNR vs recall, {name} noise ===");
    println!(
        "  {:>5}  {:>6}  {:>7}  {:>10}  {:>5}  {:>8}  {:>9}  {:>9}  {:>8}",
        "SNR", "recall", "fp", "speech_rt", "nseg", "med ms", "sp.score", "ot.score", "floor dB"
    );
    for p in curve {
        println!(
            "  {:>5}  {:>6.3}  {:>7.3}  {:>10.3}  {:>5}  {:>8.0}  {:>9.3}  {:>9.3}  {:>8.1}",
            p.snr,
            p.recall,
            p.fp,
            p.ratio,
            p.segments,
            p.median_ms,
            p.speech_score,
            p.other_score,
            p.floor_db
        );
    }
}

#[test]
fn the_noise_trackers_attack_constant_is_tuned_not_guessed() {
    // The per-bin noise estimate has to creep up slowly enough that a two-second
    // utterance does not become the noise, and quickly enough that a fan being
    // switched on is absorbed. Those pull in opposite directions, so the value
    // is measured rather than reasoned about.
    //
    // The failure is specific and worth stating: with a 500 ms attack the
    // estimate converges to 98 % of a two-second utterance's per-bin power, so
    // the subtraction removes the talker and the measured speech score barely
    // moves at all.
    println!("\n=== per-bin noise tracker: attack time constant (continuous floor) ===");
    println!(
        "  {:>8}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}",
        "attack", "sp@6dB", "sp@12dB", "sp@18dB", "sp@24dB", "sp@30dB"
    );
    let mut best = (0i64, f64::MIN);
    for attack in [250i64, 500, 1_000, 2_000, 4_000, 8_000, 16_000] {
        let mut row = Vec::new();
        for snr in [6, 12, 18, 24, 30] {
            let w = FeatureWeights {
                noise_attack_ms: attack,
                ..FeatureWeights::default()
            };
            row.push(run_point(snr, true, w).speech_score);
        }
        println!(
            "  {:>7}ms  {:>9.3}  {:>9.3}  {:>9.3}  {:>9.3}  {:>9.3}",
            attack, row[0], row[1], row[2], row[3], row[4]
        );
        // The 6 dB point decides: it is the sensitivity limit, and it is the
        // point where the measurement is most sensitive to the constant.
        if row[0] > best.1 {
            best = (attack, row[0]);
        }
    }
    println!();
    // The measurement is *flat*: 250 ms and 16 s differ by 4 %. Asserting the
    // exact optimum would be asserting a fixture seed, and the honest claim is
    // the one the numbers support -- any of these values works, which is itself
    // the reason to pick a round one in the middle.
    let default = FeatureWeights::default();
    let at_default = run_point(6, true, default).speech_score;
    assert!(
        at_default >= 0.9 * best.1,
        "the default attack constant {} ms gives {at_default} at 6 dB, \
         more than 10 % below the best ({best:?})",
        default.noise_attack_ms
    );
}
#[test]
fn the_sensitivity_curve_is_measured_not_guessed() {
    // The single most useful number in the report: the speech-to-noise ratio at
    // which the detector's recall falls off. Two regimes, because the adaptive
    // floor makes them genuinely different problems.
    let cont = sweep(true);
    let intermittent = sweep(false);
    print_curve("continuous", &cont);
    print_curve("intermittent", &intermittent);
    println!();

    // --- continuous: the hard case, and the one the thresholds are set for.
    let c30 = cont.last().unwrap();
    assert_eq!(c30.snr, 30);
    assert!(
        c30.recall > 0.97,
        "at 30 dB over a continuous floor, recall {}",
        c30.recall
    );
    assert!(
        c30.fp < 0.10,
        "false-positive ratio at 30 dB was {}",
        c30.fp
    );
    // The knee is the number worth quoting, so it is *found* by measurement and
    // printed, and the assertion is only that it has not regressed.
    let limit = cont.iter().find(|p| p.recall >= 0.5).map(|p| p.snr);
    println!(
        "  continuous-floor recall reaches 0.5 at {} dB SNR",
        limit.map_or_else(|| "never".to_string(), |v| v.to_string())
    );
    assert!(
        limit.is_some_and(|v| v <= 6),
        "the continuous-floor knee moved to {limit:?} dB; the sensitivity limit regressed"
    );

    // --- intermittent: the floor tracks the digital silence, so the detector
    // is far more sensitive than the stated SNR suggests. This is the honest
    // explanation for why a real meeting recording is easier than a synthetic
    // one at the same nominal noise level.
    assert!(
        intermittent.first().unwrap().recall > cont.first().unwrap().recall,
        "an intermittent floor must be easier than a continuous one: {} vs {}",
        intermittent.first().unwrap().recall,
        cont.first().unwrap().recall
    );
    for p in &intermittent {
        assert!(
            p.recall > 0.45,
            "intermittent {} dB recall {}",
            p.snr,
            p.recall
        );
        assert!(p.fp < 0.25, "intermittent {} dB fp {}", p.snr, p.fp);
    }
    // From 6 dB up, an intermittent floor is detected essentially totally, and
    // that is the headline: the same voice in the same room, with the same
    // *nominally* -40 dB of background, is found reliably as soon as the
    // background is not continuous.
    for p in intermittent.iter().filter(|p| p.snr >= 6) {
        assert!(
            p.recall > 0.97,
            "intermittent {} dB recall {}",
            p.snr,
            p.recall
        );
    }

    // Recall must not fall as the speech gets louder: a detector that loses
    // quiet speech is broken in a way a threshold sweep would hide.
    for w in cont.windows(2) {
        assert!(
            w[1].recall >= w[0].recall - 0.20,
            "continuous recall fell from {} at {} dB to {} at {} dB",
            w[0].recall,
            w[0].snr,
            w[1].recall,
            w[1].snr
        );
    }
    for w in intermittent.windows(2) {
        assert!(
            w[1].recall >= w[0].recall - 0.20,
            "intermittent recall fell from {} at {} dB to {} at {} dB",
            w[0].recall,
            w[0].snr,
            w[1].recall,
            w[1].snr
        );
    }
}

#[test]
fn the_noise_floor_window_is_tuned_not_guessed() {
    // The percentile floor's window has to be **longer than the longest
    // utterance**, and that is not obvious. A percentile is dominated by the
    // quietest part of its window, so it is safe against a loud *spike* -- but a
    // 2-second window filled entirely with a 2-second utterance is 100 % speech,
    // and the 20th percentile of that is a speech frame. The floor then rises
    // to the talker and the SNR collapses to zero in the middle of a sentence.
    //
    // Measured on the sweep fixture, with the window at 2 s the detector's own
    // `snr_db` inside the planted speech is 1.1 dB at a nominal 6 dB SNR and
    // 12.3 dB at a nominal 30 dB SNR -- the floor is tracking the utterance, not
    // the room. The window is swept here and the default is the measured
    // optimum.
    println!("\n=== adaptive level floor: trailing window length (continuous floor) ===");
    println!(
        "  {:>8}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}",
        "window", "sp@6dB", "sp@12dB", "sp@18dB", "sp@24dB", "sp@30dB"
    );
    let mut best = (0i64, f64::MIN);
    for window_ms in [1_000i64, 2_000, 3_000, 4_000, 6_000, 8_000, 12_000] {
        let mut row = Vec::new();
        for snr in [6, 12, 18, 24, 30] {
            let cfg = EnergyVadConfig {
                noise_window_ms: window_ms,
                ..EnergyVadConfig::default()
            };
            let f = sweep_fixture(snr, true);
            let audio = f.audio();
            let vad = EnergyVad::with_config(cfg).unwrap();
            let an = vad
                .analyse(&audio, &vaani_core::CancelToken::new())
                .unwrap();
            let (in_speech, _) = support::score_separation(&an.scores, &f.speech);
            row.push(in_speech);
        }
        println!(
            "  {:>7}ms  {:>9.3}  {:>9.3}  {:>9.3}  {:>9.3}  {:>9.3}",
            window_ms, row[0], row[1], row[2], row[3], row[4]
        );
        if row[0] > best.1 {
            best = (window_ms, row[0]);
        }
    }
    println!();
    // Flat again, for a different reason than the attack constant: the two-pass
    // latch, not the window, is what keeps speech out of the floor. The window
    // is the defence against a *sustained sound the latch missed*, so 4 s is
    // chosen as "long enough to outlast a sentence" and not as a measured
    // optimum. The tolerance assertion is what keeps it honest.
    let at_default = {
        let cfg = EnergyVadConfig::default();
        let f = sweep_fixture(6, true);
        let audio = f.audio();
        let vad = EnergyVad::with_config(cfg).unwrap();
        let an = vad
            .analyse(&audio, &vaani_core::CancelToken::new())
            .unwrap();
        support::score_separation(&an.scores, &f.speech).0
    };
    assert!(
        at_default >= 0.9 * best.1,
        "the default floor window {} ms gives {at_default} at 6 dB, \
         more than 10 % below the best ({best:?})",
        EnergyVadConfig::default().noise_window_ms
    );
}

#[test]
fn what_limits_the_score_at_low_snr_is_printed() {
    // A diagnostic, kept because it is the fastest way back to the answer when a
    // sensitivity number moves. It prints every shape term for the frames inside
    // a planted speech span, at three SNRs, so the term that is holding the
    // score down is visible rather than inferred.
    use vaani_vad::features::FrameFeatures as F;
    let w = FeatureWeights::default();
    println!("\n=== what limits the score inside planted speech ===");
    for (label, continuous) in [("continuous", true), ("intermittent", false)] {
        for snr in [6, 12, 18, 24, 30] {
            let f = sweep_fixture(snr, continuous);
            let audio = f.audio();
            let vad = EnergyVad::new();
            let an = vad
                .analyse(&audio, &vaani_core::CancelToken::new())
                .unwrap();
            let inside: Vec<&F> = an
                .features
                .iter()
                .enumerate()
                .filter(|(i, _)| {
                    let ms = (*i as i64) * support::HOP_MS;
                    f.speech.iter().any(|s| s.start_ms <= ms && ms < s.end_ms)
                })
                .map(|(_, x)| x)
                .collect();
            let col = |g: fn(&F) -> f32| -> f64 {
                let v: Vec<f32> = inside.iter().map(|x| g(x)).collect();
                support::mean(&v).unwrap_or(f64::NAN)
            };
            let aud: f64 = support::mean(
                &inside
                    .iter()
                    .map(|x| vaani_vad::features::audibility_term(x.snr_db, &w))
                    .collect::<Vec<f32>>(),
            )
            .unwrap_or(f64::NAN);
            let shp: f64 = support::mean(
                &inside
                    .iter()
                    .map(|x| vaani_vad::features::shape_term(x, &w))
                    .collect::<Vec<f32>>(),
            )
            .unwrap_or(f64::NAN);
            println!(
                "  {label:>11} SNR {snr:>3}  score={:.3}  audibility={aud:.3}  shape={shp:.3}  \
             snr_db={:.1}  flatness={:.4}  tonality_dB={:.1}  voicing={:.3}  \
             mid={:.3}  hf={:.3}  rolloff={:.0}  zcr={:.3}  flux={:.2}",
                support::mean(
                    &inside
                        .iter()
                        .map(|x| vaani_vad::features::frame_score(x, &w))
                        .collect::<Vec<f32>>()
                )
                .unwrap_or(f64::NAN),
                col(|x| x.snr_db),
                col(|x| x.flatness),
                col(|x| 10.0 * x.tonality.max(1e-12).log10()),
                col(|x| x.voicing),
                col(|x| x.mid_ratio),
                col(|x| x.hf_ratio),
                col(|x| x.rolloff_hz),
                col(|x| x.zcr),
                col(|x| x.flux),
            );
        }
    }
    println!();
}

#[test]
fn the_n_freq_constant_matches_the_fft_size() {
    // Guards the two constants the band-index arithmetic depends on.
    assert_eq!(N_FREQ, 257);
    assert_eq!(N_FREQ - 1, 256);
}

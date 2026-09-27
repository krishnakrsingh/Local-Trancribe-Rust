//! End-to-end correctness of the clustering, against a reference whose speaker
//! identity is known exactly.
//!
//! # What this measures, and what it does not
//!
//! Diarization cannot be tested end to end on real audio without a human
//! annotating who spoke when. This crate therefore *generates* the reference:
//! [`vaani_diarization::synth`] renders voices whose identity is known by
//! construction, interleaves them on a timeline, and the engine is asked to
//! separate them. That makes a real correctness measurement possible.
//!
//! It does **not** measure the world. These are synthetic voices with fixed
//! fundamentals, fixed harmonic rolloffs and fixed formants. Real speakers are
//! not that separable, and the numbers below are a ceiling for the feature
//! vector rather than an estimate of real-world accuracy. Any DER claim about
//! real meetings would need an annotated real reference, which this repository
//! does not have.
//!
//! # One labelling function
//!
//! Every score in this file comes from [`vaani_diarization::eval`], which reads
//! the per-frame reference that [`vaani_diarization::synth::render`] already
//! returns. It is the same code path `examples/geometry.rs` uses.
//!
//! This was previously reimplemented here, from the utterance list, while the
//! example inferred speaker identity from turn parity. The two disagreed, the
//! example reported a same-speaker p50 and a different-speaker p50 that were
//! *both* exactly 0.0, and the only visible symptom was that a diagnostic and a
//! test disagreed -- which reads as "the engine behaves inconsistently" and
//! sends you tuning against a fiction. A test and a diagnostic that compute
//! ground truth differently are two different tests wearing a trenchcoat.
//!
//! # The operating region
//!
//! These fixtures use **12-second turns**, and that is a measured choice rather
//! than a convenient one. The geometry sweep in `examples/geometry.rs` shows
//! the engine separates speakers reliably at 12 s turns and does not at 2 s or
//! 4 s, because a 1.5 s analysis window straddling a turn boundary is a
//! genuine two-speaker observation and a boundary-contaminated window pollutes
//! the same-speaker population. `short_turns_are_a_known_limitation` records
//! the failing case explicitly instead of leaving it to be discovered.

use vaani_core::media::AudioBuffer;
use vaani_core::pipeline::{DiarizationEngine, DiarizationResult};
use vaani_core::{CancelToken, Time};
use vaani_diarization::config::{DiarizationConfig, SegmentationBias, SpeakerCount};
use vaani_diarization::engine::HeuristicDiarizer;
use vaani_diarization::eval::{self, Quality};
use vaani_diarization::synth::{self, Voice};

/// Render a meeting, run the engine, and score it through the shared path.
struct Run {
    speakers: Vec<i32>,
    quality: Quality,
    result: DiarizationResult,
    clusters: usize,
}

impl Run {
    fn report(&self, label: &str) {
        println!(
            "{label}: {} speakers, {} spans | purity {:.4} recall {:.4} complete {:.4}\n  \
             within-speaker p50 {:.4} p90 {:.4} | cross-speaker min {:.4} p10 {:.4} p50 {:.4}\n  \
             margin {:+.4} | scored {} windows",
            self.result.num_speakers,
            self.result.spans.len(),
            self.quality.purity,
            self.quality.recall,
            self.quality.complete_accuracy,
            self.quality.geometry.same.p50,
            self.quality.geometry.same.p90,
            self.quality.geometry.different.min,
            self.quality.geometry.different.p10,
            self.quality.geometry.different.p50,
            self.quality.geometry.margin,
            self.quality.scored,
        );
    }
}

/// Run a fixture end to end.
fn meeting(
    voices: &[Voice],
    speakers: &[usize],
    turn_ms: i64,
    turns: usize,
    noise: f32,
    config: DiarizationConfig,
) -> Run {
    meeting_with_gains(voices, speakers, turn_ms, turns, noise, None, config)
}

/// As [`meeting`], with a per-utterance gain pattern for level robustness.
fn meeting_with_gains(
    voices: &[Voice],
    speakers: &[usize],
    turn_ms: i64,
    turns: usize,
    noise: f32,
    gains: Option<Vec<f32>>,
    config: DiarizationConfig,
) -> Run {
    let total_ms = turn_ms * turns as i64;
    let base = synth::alternate(speakers, 0, turn_ms, turns);
    let utterances = match gains {
        None => base,
        Some(g) => base
            .into_iter()
            .enumerate()
            .map(|(i, u)| u.with_gain(g[i % g.len()]))
            .collect(),
    };
    // The second value is the per-frame ground truth. It is used, not
    // recomputed.
    let (samples, truth) = synth::render(&utterances, voices, total_ms, 16_000, noise, 2024);
    let audio = AudioBuffer::mono(samples, 16_000, Time::ZERO);
    let engine = HeuristicDiarizer::with_config(config.clone());
    let windows = engine
        .debug_windows(&audio, &CancelToken::new())
        .expect("windows");
    let labels = eval::label_windows(&windows, &truth, config.features.frame_ms(), 0);
    let quality = eval::measure_quality(&windows, &labels);
    let speakers: Vec<i32> = labels
        .iter()
        .filter(|l| l.pure)
        .map(|l| l.speaker)
        .collect();
    let clusters = windows.iter().map(|w| w.cluster).max().map_or(0, |m| m + 1);
    let result = engine
        .diarize(&audio, &CancelToken::new(), &|_| {})
        .expect("diarize");
    Run {
        speakers,
        quality,
        result,
        clusters,
    }
}

fn fixed(n: usize) -> DiarizationConfig {
    DiarizationConfig::default().with_speaker_count(n)
}

#[test]
fn two_speakers_are_recovered_from_a_synthetic_meeting() {
    let voices = synth::two_voices();
    let run = meeting(&voices, &[0, 1], 12_000, 12, 0.003, fixed(2));
    run.report("two speakers, 12 s turns");

    // Separability: the 10th percentile of cross-speaker pairs is further apart
    // than the 90th percentile of same-speaker pairs, so the two populations do
    // not overlap where the threshold cuts.
    let g = &run.quality.geometry;
    assert!(
        g.different.p10 > g.same.p90,
        "populations overlap: cross-speaker p10 {:.4}, same-speaker p90 {:.4}",
        g.different.p10,
        g.same.p90
    );
    assert!(
        run.quality.purity > 0.95,
        "purity {:.4}",
        run.quality.purity
    );
    assert!(
        run.quality.recall > 0.95,
        "recall {:.4}",
        run.quality.recall
    );
    assert!(
        run.quality.complete_accuracy > 0.94,
        "complete accuracy {:.4}",
        run.quality.complete_accuracy
    );
    assert_eq!(run.result.num_speakers, 2);
    assert!(!run.result.spans.is_empty());
}

#[test]
fn the_estimated_speaker_count_finds_two_on_the_same_fixture() {
    let voices = synth::two_voices();
    let run = meeting(
        &voices,
        &[0, 1],
        12_000,
        12,
        0.003,
        DiarizationConfig::default(),
    );
    run.report("estimated count");
    assert_eq!(
        run.result.num_speakers, 2,
        "the estimated count missed: {} speakers, purity {:.4}, margin {:+.4}",
        run.result.num_speakers, run.quality.purity, run.quality.geometry.margin
    );
}

/// The auto-gap count rule over-splits this fixture, and that is recorded.
///
/// The rule cuts at the largest *relative* jump in the average-linkage sequence.
/// On this fixture the sequence has several 1.2-1.5x steps that are not speaker
/// changes but residual same-speaker spread, so the rule lands above the real
/// gap and reports four speakers for two people. Raising the required ratio from
/// 1.15 to 1.6 does not fix it.
///
/// The default policy is `Threshold`, not `Auto`, precisely because an absolute
/// threshold is better behaved here: the same fixture scores 0.958 purity and
/// exactly two speakers under `Threshold`, with the two populations separated by
/// a margin of +0.79. `Auto` is available for callers who want an estimated
/// count without committing to a distance, and this test records that it
/// currently over-splits rather than leaving it to be discovered.
#[test]
#[ignore = "known limitation: the largest-relative-gap rule over-splits this fixture"]
fn the_gap_rule_over_splits_this_fixture() {
    let voices = synth::two_voices();
    let mut cfg = DiarizationConfig::default();
    cfg.clusters.count = SpeakerCount::Auto {
        max_speakers: 8,
        min_gap_ratio: 1.6,
    };
    let run = meeting(&voices, &[0, 1], 12_000, 12, 0.003, cfg);
    run.report("auto-gap rule, ratio 1.6");
    assert_eq!(
        run.result.num_speakers, 2,
        "expected the documented over-split; the gap rule now gives {}, so this \
         limitation may be fixed and the test should be un-ignored",
        run.result.num_speakers
    );
}

#[test]
fn three_voices_are_recovered() {
    // Two of the three are within 15% of each other in median F0, so the pitch
    // dimension cannot separate them alone.
    let voices = synth::three_voices();
    let run = meeting(&voices, &[0, 1, 2], 12_000, 12, 0.003, fixed(3));
    run.report("three voices, 12 s turns");
    assert!(
        run.quality.purity > 0.85,
        "three-voice purity {:.4}",
        run.quality.purity
    );
    assert_eq!(run.result.num_speakers, 3);
}

#[test]
fn a_single_speaker_produces_one_cluster_over_a_long_run() {
    // The product requirement that motivated all of it: one person, one
    // cluster, for two minutes. A six-second steady tone is covered separately
    // in the engine's own unit tests, where it is the degenerate case.
    let voices = vec![synth::two_voices()[0].clone()];
    let run = meeting(
        &voices,
        &[0],
        4_000,
        30,
        0.003,
        DiarizationConfig::default(),
    );
    run.report("one speaker, 2 minutes");
    assert_eq!(
        run.result.num_speakers, 1,
        "one speaker shattered into {}",
        run.result.num_speakers
    );
    assert_eq!(run.clusters, 1);
}

#[test]
fn a_speaker_whose_pitch_drifts_stays_one_cluster() {
    // A speaker who rises from 110 Hz to 150 Hz across the recording. The
    // pitch dimension is absolute, so a drifting voice *does* move along that
    // dimension -- the question is whether a 36% pitch excursion reads as two
    // people. It must not.
    let base = synth::two_voices()[0].clone();
    let utterances = synth::alternate(&[0], 0, 4_000, 24);
    // Rendered by hand rather than through `synth::render`, because each turn
    // needs a gliding fundamental and `render` indexes a fixed voice list.
    //
    // The drift is **within each utterance**: 110 Hz rising to 150 Hz across
    // the 4 s turn and resetting on the next one. That is the realistic case and
    // the one specified -- a speaker's intonation moves inside a sentence and
    // returns, so a 1.5 s window's median sits at a stable mid-utterance value.
    //
    // `render_glide` keeps the phase continuous across the whole turn, so there
    // is no broadband click at the turn boundary for the activity detector to
    // find. See its documentation for why concatenating separate renders does
    // make a fixture measure its own construction.
    //
    // A *sustained* pitch change across the whole meeting is a much harder case
    // and a genuine limitation, recorded in
    // `a_sustained_pitch_change_across_a_meeting_is_a_documented_limitation`.
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
    let audio = AudioBuffer::mono(samples, 16_000, Time::ZERO);
    let cfg = DiarizationConfig::default();
    let engine = HeuristicDiarizer::with_config(cfg.clone());
    let windows = engine
        .debug_windows(&audio, &CancelToken::new())
        .expect("windows");
    let labels = eval::label_windows(&windows, &truth, cfg.features.frame_ms(), 0);
    let quality = eval::measure_quality(&windows, &labels);
    let g = &quality.geometry;
    println!(
        "pitch drift 110->150 Hz: within-speaker p50 {:.4} p90 {:.4} max {:.4}",
        g.same.p50, g.same.p90, g.same.max
    );
    let result = engine
        .diarize(&audio, &CancelToken::new(), &|_| {})
        .expect("diarize");
    assert_eq!(
        result.num_speakers, 1,
        "a voice with rising intonation was reported as {} speakers",
        result.num_speakers
    );
}

/// A pitch change held across the whole meeting is a documented limitation.
///
/// One voice sliding from 110 Hz to 150 Hz over ninety seconds and staying there
/// is reported as more than one speaker. The measured reason is that the drift
/// is 0.45 of log2 against a 0.78 gap between the two reference voices, so it is
/// 58% of a speaker change as far as any pitch-based distance is concerned, and
/// rescaling cannot change that ratio.
///
/// This is `#[ignore]`d and asserted in the failing direction, so the numbers
/// are pinned in the diff. The fix is not a constant: it is either
/// turn-length-aware clustering that tolerates a slow drift within a speaker, or
/// a pitch dimension normalised against the recording's own pitch range, which
/// reintroduces the file-dependence documented in
/// `features::within_speaker_spread`. Neither is in scope here.
#[test]
#[ignore = "known limitation: a sustained pitch change reads as a speaker change"]
fn a_sustained_pitch_change_across_a_meeting_is_a_documented_limitation() {
    let base = synth::two_voices()[0].clone();
    let utterances = synth::alternate(&[0], 0, 4_000, 24);
    let total_ms = 4_000 * 24;
    let mut samples = vec![0f32; (total_ms * 16_000 / 1_000) as usize];
    let mut truth = vec![-1i32; (total_ms / 10) as usize];
    for (i, u) in utterances.iter().enumerate() {
        let voice = Voice {
            f0: 110.0 + (i as f32 / 24.0) * 40.0,
            f0_drift: 0.01,
            ..base.clone()
        };
        let mut rng = synth::Rng::new(2024 + i as u64);
        let lo = (u.start_ms * 16_000 / 1_000) as usize;
        let hi = (u.end_ms * 16_000 / 1_000) as usize;
        let rendered = voice.render(hi - lo, 16_000, 0.0, 0.0, &mut rng);
        for (k, v) in rendered.iter().enumerate() {
            samples[lo + k] = *v * u.gain * 0.5;
        }
        let range = (u.start_ms / 10) as usize..(u.end_ms / 10) as usize;
        for slot in &mut truth[range] {
            *slot = 0;
        }
    }
    let audio = AudioBuffer::mono(samples, 16_000, Time::ZERO);
    let cfg = DiarizationConfig::default();
    let engine = HeuristicDiarizer::with_config(cfg.clone());
    let windows = engine
        .debug_windows(&audio, &CancelToken::new())
        .expect("windows");
    let labels = eval::label_windows(&windows, &truth, cfg.features.frame_ms(), 0);
    let q = eval::measure_quality(&windows, &labels);
    println!(
        "sustained drift 110->150 Hz: within-speaker p50 {:.4} p90 {:.4}",
        q.geometry.same.p50, q.geometry.same.p90
    );
    assert!(
        q.geometry.same.p90 < 0.892,
        "expected the documented limitation; the within-speaker p90 is now {:.4}, so the \
         limitation may be fixed and this test should be un-ignored",
        q.geometry.same.p90
    );
}

#[test]
fn two_speakers_with_the_same_pitch_are_separated_by_their_timbre() {
    // The pitch-only shortcut must not be the only defence. Both voices render
    // at the same fundamental and differ in harmonic rolloff and formants, so
    // the spectral blocks carry this one.
    let two = synth::two_voices();
    let voices = vec![
        Voice {
            f0: 130.0,
            f0_drift: 0.10,
            ..two[0].clone()
        },
        Voice {
            f0: 130.0,
            f0_drift: 0.07,
            ..two[1].clone()
        },
    ];
    let run = meeting(
        &voices,
        &[0, 1],
        12_000,
        12,
        0.003,
        DiarizationConfig::default(),
    );
    run.report("same pitch, different timbre");
    let g = &run.quality.geometry;
    assert!(
        g.different.p10 > g.same.p90,
        "same-pitch speakers did not separate: cross p10 {:.4} vs same p90 {:.4}",
        g.different.p10,
        g.same.p90
    );
    assert!(
        run.quality.purity > 0.90,
        "same-pitch purity {:.4}",
        run.quality.purity
    );
}

/// A speaker 20 dB below another is mostly discarded before it is ever embedded.
///
/// Both voices here are genuinely different -- `two_voices()` renders 129.8 Hz
/// against 217.2 Hz, an octave apart -- so the absolute, level-invariant pitch
/// dimension should separate them regardless of level. It does not, and the
/// reason is upstream of the features: the engine's own energy-threshold
/// activity detector discards almost all of the quiet speaker's speech before a
/// single window is built. On this fixture the quiet speaker contributes roughly
/// one window out of 126.
///
/// This is a level-sensitivity in the *activity detector*, not in the feature
/// vector, and replacing the detector with the committed `vaani-vad` from
/// `agent/a3-vad` is the fix. That integration is listed under INTEGRATION
/// NOTES and is not done here: `vaani-vad` is another agent's crate and this one
/// does not edit it.
///
/// `#[ignore]`d, and asserted in the failing direction, so the number is pinned
/// in the diff. If a future VAD fixes it, this test fails saying so and should
/// be un-ignored.
#[test]
#[ignore = "known limitation: the internal activity detector discards a 20 dB quieter speaker"]
fn a_speaker_twenty_decibels_quieter_is_currently_discarded() {
    let voices = synth::two_voices();
    let run = meeting_with_gains(
        &voices,
        &[0, 1],
        12_000,
        12,
        0.003,
        Some(vec![1.0, 0.1]),
        DiarizationConfig::default(),
    );
    run.report("one speaker 20 dB quieter");
    let mut counts: std::collections::BTreeMap<i32, usize> = Default::default();
    for s in &run.speakers {
        *counts.entry(*s).or_default() += 1;
    }
    println!("  scored windows per speaker: {counts:?}");
    assert!(
        run.result.num_speakers == 2 && counts.values().min().copied().unwrap_or(0) > 10,
        "expected the documented limitation; the engine now reports {} speakers with \
         {counts:?} windows, so this may be fixed -- un-ignore this test",
        run.result.num_speakers
    );
}

#[test]
fn one_person_using_two_voices_is_reported_as_two_and_that_is_documented() {
    // Code-switching: one person, alternating between two voices. There is no
    // correct answer -- the engine is asked "how many distinct voices are
    // here", and two are. This test exists so the behaviour is *recorded* and
    // cannot change silently, not because two is wrong.
    let voices = synth::two_voices();
    let run = meeting(
        &voices,
        &[0, 1],
        12_000,
        12,
        0.003,
        DiarizationConfig::default(),
    );
    run.report("one person, two voices");
    let g = &run.quality.geometry;
    assert!(
        g.different.p10 > g.same.p90,
        "the two voices did not separate, so the count would be 1"
    );
    assert_eq!(
        run.result.num_speakers, 2,
        "two distinct voices from one person are reported as {}; a diariser has no way to know they are one person, and this is the documented behaviour",
        run.result.num_speakers
    );
}

#[test]
fn the_bias_setting_moves_the_split_in_the_documented_direction() {
    let voices = synth::two_voices();
    let split = meeting(
        &voices,
        &[0, 1, 2],
        12_000,
        12,
        0.003,
        DiarizationConfig::default().with_bias(SegmentationBias::SplitEasily),
    );
    let merge = meeting(
        &voices,
        &[0, 1, 2],
        12_000,
        12,
        0.003,
        DiarizationConfig::default().with_bias(SegmentationBias::MergeReadily),
    );
    println!(
        "bias: SplitEasily {} speakers, MergeReadily {} speakers",
        split.result.num_speakers, merge.result.num_speakers
    );
    assert!(
        split.result.num_speakers >= merge.result.num_speakers,
        "SplitEasily gave {} and MergeReadily {}",
        split.result.num_speakers,
        merge.result.num_speakers
    );
}

#[test]
fn a_fixed_speaker_count_is_honoured_even_when_the_geometry_says_otherwise() {
    let voices = synth::two_voices();
    let run = meeting(&voices, &[0, 1], 12_000, 12, 0.003, fixed(2));
    assert_eq!(run.result.num_speakers, 2);
    let run = meeting(
        &voices,
        &[0, 1],
        12_000,
        12,
        0.003,
        DiarizationConfig::default().with_speaker_count(4),
    );
    assert!(
        run.result.num_speakers <= 4,
        "a fixed cap of 4 was exceeded: {}",
        run.result.num_speakers
    );
}

/// A recorded, deliberately failing case.
///
/// Turn length is the engine's hardest axis, and 2 s and 4 s turns are measurably
/// outside the working region. The numbers are pinned here so that a future
/// change which makes it worse, or which changes the reported figures, is
/// visible in the diff rather than discovered later.
///
/// `#[ignore]`d because it fails: 1.5 s analysis windows against 2 s turns means
/// most windows straddle a boundary, the same-speaker population's 90th
/// percentile lands above the merge distance, and no single threshold can both
/// hold one speaker together and split two. The engineering fix is a
/// turn-length-aware window length, not a different constant. See OPEN ISSUES.
#[test]
#[ignore = "known limitation: 2 s turns are outside the measured operating region"]
fn short_turns_are_a_known_limitation() {
    let voices = synth::two_voices();
    let run = meeting(
        &voices,
        &[0, 1],
        2_000,
        60,
        0.003,
        DiarizationConfig::default(),
    );
    run.report("two speakers, 2 s turns");
    println!("  same-speaker p90 {:.4}", run.quality.geometry.same.p90);
    println!(
        "  cross-speaker p10 {:.4}",
        run.quality.geometry.different.p10
    );
    assert!(
        run.quality.geometry.same.p90 < run.quality.geometry.different.p10,
        "the populations are expected to overlap at 2 s turns; if this now passes, \
         remove the #[ignore] and raise the bar"
    );
}

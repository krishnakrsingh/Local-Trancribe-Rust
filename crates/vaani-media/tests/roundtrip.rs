//! The round trip the ASR golden tests depend on.
//!
//! `crates/vaani-asr` carries 12,928 reference log-mel features and 202 reference
//! token ids, computed from a 16 kHz mono input. If this crate's output for such
//! a file is not *bit identical* to the input, every one of those numbers becomes
//! unreproducible and the parity tests stop meaning anything. So the requirement
//! is not "close enough" — it is exact, and these tests say so.

mod common;

use common::{Fixture, SampleFormat, WavSpec, flac, tone, wav};
use vaani_core::{CancelToken, Time};

#[test]
fn a_16k_mono_wav_round_trips_sample_for_sample() {
    // Values chosen as exact multiples of 1/32768 so that "sample exact" is bit
    // exact rather than approximately exact, and so a mistake shows up as a
    // specific wrong bit rather than as a rounding wobble.
    let n = 64_000usize;
    let samples: Vec<f32> = (0..n)
        .map(|i| ((i as i64 * 7) % 40_001 - 20_000) as f32 / 32_768.0)
        .collect();

    let fx = Fixture::new(
        "wav",
        &wav(&samples, &WavSpec::pcm(16_000, 1, SampleFormat::S16)),
    );
    let out = vaani_media::decode_to_mono16k(fx.path(), &CancelToken::new())
        .expect("a 16 kHz mono WAV must decode");

    assert_eq!(out.sample_rate, 16_000);
    assert_eq!(out.channels, 1);
    assert_eq!(
        out.start_time,
        Time::ZERO,
        "the buffer must start at zero so sample n is time n / 16000"
    );
    assert_eq!(out.frames(), n, "the length must be exact, not rounded");
    assert_eq!(out.duration(), Time::from_millis(4000));

    for (i, (&got, &want)) in out.samples.iter().zip(&samples).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "sample {i} changed: {got} != {want}"
        );
    }
}

#[test]
fn a_16k_mono_float_wav_round_trips_bit_for_bit() {
    // Including the awkward values. A resampler that "cleans up" a denormal or a
    // negative zero would be doing something a lossless path must not do.
    //
    // Every value is inside `[-1, 1]` except the denormals, and that is
    // deliberate: the downmix's guard against a runaway gain clamps anything
    // beyond `MAX_INPUT_SAMPLE`, and that is a separate, asserted decision —
    // see `out_of_range_samples_are_clamped_and_counted` below. Mixing the two
    // here would test the clamp while claiming to test the bypass.
    let samples: Vec<f32> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        1e-30,
        -1e-30,
        f32::EPSILON,
        -f32::EPSILON,
    ];
    let fx = Fixture::new(
        "wav",
        &wav(&samples, &WavSpec::pcm(16_000, 1, SampleFormat::F32)),
    );
    let out = vaani_media::decode_to_mono16k(fx.path(), &CancelToken::new()).expect("decode");
    assert_eq!(out.samples.len(), samples.len());
    for (i, (&got, &want)) in out.samples.iter().zip(&samples).enumerate() {
        assert_eq!(got.to_bits(), want.to_bits(), "sample {i}: {got} != {want}");
    }
}

#[test]
fn out_of_range_samples_are_clamped_and_counted() {
    // A float WAV can hold anything, and a recorder with a broken gain stage
    // really does write 100.0 into it. Handing that to the ASR front end as-is
    // saturates every log-mel bin, so the downmix clamps at
    // `MAX_INPUT_SAMPLE` — and the report has to say it happened, because a
    // transcript of a clipped file should not look like a clean one.
    let samples: Vec<f32> = vec![0.25, 123.456_79, -987.654_3, 8.0, -8.0, 8.5, 0.0];
    let fx = Fixture::new(
        "wav",
        &wav(&samples, &WavSpec::pcm(16_000, 1, SampleFormat::F32)),
    );
    let decoded = vaani_media::decode(
        fx.path(),
        &CancelToken::new(),
        &vaani_media::DecodeOptions::default(),
    )
    .expect("decode");

    let cap = vaani_media::MAX_INPUT_SAMPLE;
    for (i, &got) in decoded.audio.samples.iter().enumerate() {
        assert_eq!(
            got,
            samples[i].clamp(-cap, cap),
            "sample {i} was not clamped as designed: {got}"
        );
        assert!(got.is_finite(), "sample {i} must be finite");
    }
    assert_eq!(
        decoded.report.over_unity_samples, 5,
        "every sample outside [-1, 1] is over-unity, whether or not it needed clamping: \
         123.46, -987.65, 8.0, -8.0 and 8.5"
    );
    assert_eq!(
        decoded.report.clamped_samples, 3,
        "only the three samples past the limit were altered: 123.46, -987.65 and 8.5"
    );
    assert!(
        decoded.report.had_to_repair_samples(),
        "a report that does not admit the repair is a silent lie"
    );
}

#[test]
fn a_16k_mono_flac_round_trips_bit_for_bit() {
    // The same guarantee through a real lossless *codec*, which is a stronger
    // statement: it proves the FLAC path as well as the resampler bypass.
    let n = 64_000usize;
    let pcm: Vec<i16> = (0..n)
        .map(|i| ((i as i64 * 13) % 30_000 - 15_000) as i16)
        .collect();
    let fx = Fixture::new("flac", &flac(&pcm, 16_000, 1));
    let out = vaani_media::decode_to_mono16k(fx.path(), &CancelToken::new()).expect("decode");
    assert_eq!(out.samples.len(), n);
    for (i, (&got, &want)) in out.samples.iter().zip(&pcm).enumerate() {
        let expect = f64::from(want) / 32768.0;
        let got = f64::from(got);
        assert!(
            (got - expect).abs() < 1e-12,
            "sample {i}: {got} != {expect}"
        );
    }
}

#[test]
fn a_16k_stereo_wav_downmixes_and_resamples_in_one_pass() {
    // 32 kHz stereo to 16 kHz mono: both the downmix and the rate change, and the
    // downmix must happen *before* the resample so the filter sees one channel.
    let n = 32_000usize;
    let l = tone(n, 32_000, 400.0, 0.6);
    let r: Vec<f32> = l.iter().map(|v| v * 0.5).collect();
    let samples = common::interleave(&l, &r, 2);
    let fx = Fixture::new(
        "wav",
        &wav(&samples, &WavSpec::pcm(32_000, 2, SampleFormat::F32)),
    );
    let out = vaani_media::decode_to_mono16k(fx.path(), &CancelToken::new()).expect("decode");
    assert_eq!(out.sample_rate, 16_000);
    assert_eq!(out.channels, 1);
    assert_eq!(out.samples.len(), 16_000);
    // (0.6 + 0.3) / 2 = 0.45, and 400 Hz is far below the cutoff.
    let mid = &out.samples[2000..14_000];
    let (amp, delay) = common::measure_tone(mid, 16_000, 400.0);
    assert!(
        (amp - 0.45).abs() < 0.01,
        "amplitude {amp}, want 0.45 (the average of 0.6 and 0.3)"
    );
    // The delay here is a *fixed offset*, not a time warp, and the bound is
    // loose on purpose. A phase read from a single correlation is only accurate
    // to about `16000 / (2*pi*f)` samples per radian, which at 400 Hz is six
    // samples — so a tight threshold would be asserting the estimator's noise,
    // not the resampler. What actually matters is pinned elsewhere: the time
    // contract in `every_output_sample_index_maps_to_the_right_millisecond`, the
    // absence of drift in `the_output_grid_does_not_drift_over_a_long_signal`,
    // and the impulse response in `a_step_settles_quickly_and_does_not_oscillate_for_long`.
    // 8 ms is well past anything a constant filter delay can be and well inside
    // what speech recognition tolerates.
    assert!(
        delay.abs() < 128.0,
        "group delay {delay} samples is too large to be a fixed filter delay"
    );
}

#[test]
fn a_16k_mono_file_needs_no_resampler_at_all() {
    // The bypass is not an optimisation detail, it is the reason the golden
    // features are reproducible, so assert that it is actually taken.
    let plan = vaani_media::DEFAULT_RESAMPLER.plan(16_000, 16_000);
    assert!(plan.passthrough);
    assert!(plan.exact_polyphase);
    let samples = tone(1000, 16_000, 300.0, 0.5);
    let out = vaani_media::resample_mono(&samples, 16_000).expect("bypass");
    assert_eq!(
        out, samples,
        "the bypass must be an identity, not an approximation"
    );
}

#[test]
fn every_output_sample_index_maps_to_the_right_millisecond() {
    // `AudioBuffer`'s time contract, checked on a real decoded buffer rather than
    // on a hand-built one, because that is where an off-by-one would hide.
    let fx = Fixture::new(
        "wav",
        &wav(
            &tone(16_000 * 3, 16_000, 250.0, 0.4),
            &WavSpec::pcm(16_000, 1, SampleFormat::S16),
        ),
    );
    let out = vaani_media::decode_to_mono16k(fx.path(), &CancelToken::new()).expect("decode");
    assert_eq!(out.duration(), Time::from_millis(3000));
    assert_eq!(out.time_of(0), Time::ZERO);
    assert_eq!(out.time_of(16_000), Time::from_millis(1000));
    assert_eq!(out.time_of(16_000 * 3 - 1), Time::from_millis(2999));
    assert_eq!(out.frames(), 48_000);
}

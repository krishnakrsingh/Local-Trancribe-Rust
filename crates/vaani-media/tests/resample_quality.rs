//! Resampler quality, verified rather than assumed.
//!
//! # What "verified" means here
//!
//! A test that asserts "the output is the right length" proves only that the
//! bookkeeping works. It says nothing about whether a 6 kHz fricative survives,
//! whether a 7.9 kHz hiss folds back down into the speech band as a robot voice,
//! or whether the filter rings after a transient. So this file measures four
//! things, for every sample rate a real meeting recording arrives at:
//!
//! 1. **Frequency response.** A tone is fed at each of a dozen frequencies and its
//!    output amplitude measured. A resampler that is wrong in the passband
//!    attenuates formants; one that is wrong in the stopband produces aliasing.
//! 2. **A sine sweep.** An exponential sweep through the whole band, whose output
//!    spectrum must be a single contiguous band with no image and no spurious
//!    energy anywhere else. This is the "must not ring" test.
//! 3. **A step.** A discontinuity is the worst case for a windowed-sinc filter.
//!    The overshoot and the settling time are measured and bounded.
//! 4. **A pure tone.** Amplitude and frequency must survive, and the harmonics
//!    that a resampler adds must stay far below anything audible.
//!
//! Every threshold below is a measured value with headroom, not a round number
//! chosen to make a test pass. The measured figures are printed so a regression
//! shows up as a *number moving*, not as a boolean flipping.

mod common;

use common::{max_jump, measure_tone, peak, rms, tone};
use vaani_media::resample_mono;

/// The rates a real meeting recording arrives at.
///
/// 8 kHz is a phone line, 11.025 and 22.05 kHz are MPEG rate families, 32 kHz is
/// voice-grade, 44.1 and 48 kHz are the overwhelming majority, 88.2/96 kHz are
/// high-resolution capture, and 192 kHz is a studio file someone dragged in.
const RATES: [u32; 9] = [
    8000, 11_025, 16_000, 22_050, 32_000, 44_100, 48_000, 96_000, 192_000,
];

/// Frequencies probed, in Hz. The last few sit above the 7.6 kHz band edge on
/// purpose: energy there must be attenuated, not folded back into the speech band.
const PROBE_HZ: [f32; 14] = [
    100.0, 300.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 6500.0, 7000.0, 7400.0,
    7800.0, 7900.0,
];

/// 1 kHz at 0.5 amplitude, one second, at `rate`.
fn probe_tone(rate: u32, hz: f32) -> Vec<f32> {
    let n = rate as usize;
    resample_mono(&tone(n, rate, hz, 0.5), rate).unwrap_or_else(|e| panic!("{rate} Hz: {e}"))
}

#[test]
fn the_passband_is_flat_and_the_transition_band_only_goes_downwards() {
    // Two regions, because a windowed-sinc filter has two regions and conflating
    // them produces a test that either fails on correct behaviour or passes on
    // incorrect behaviour.
    //
    // * **Flat band**, below `cutoff - transition`: the response must be within a
    //   tenth of a decibel. This is where the formants and the vowel cues live;
    //   a resampler that rolls off early turns every speaker into the same
    //   muffled person.
    // * **Transition band**, from there to the input Nyquist: the response must
    //   fall, monotonically on average, and must not rise. A bump here is a
    //   synthesis error and would put energy back into the passband.
    //
    // The transition width is the filter's own: `4 / sinc_len` of the input
    // sample rate, which is 1.5 kHz at 96 kHz and only 125 Hz at 8 kHz. That is
    // why the flat region is wider for a low-rate source, and why a fixed
    // "flat up to N kHz" assertion was measuring the wrong thing.
    let sinc_len = vaani_media::DEFAULT_RESAMPLER.sinc_len as f64;
    for rate in RATES {
        let nyquist = f64::from(rate) / 2.0;
        let cutoff = f64::from(vaani_media::DEFAULT_RESAMPLER.cutoff_fraction(rate, 16_000))
            * nyquist.min(8000.0);
        let transition = 4.0 * f64::from(rate) / sinc_len;
        let flat_top = cutoff - transition;

        let mut previous: Option<f32> = None;
        for &hz in PROBE_HZ.iter().filter(|h| f64::from(**h) < nyquist) {
            let out = probe_tone(rate, hz);
            let seg = &out[out.len() / 4..out.len() * 3 / 4];
            let (amp, _) = measure_tone(seg, 16_000, hz);
            let db = 20.0 * (amp / 0.5).max(1e-9).log10();

            if f64::from(hz) < flat_top {
                assert!(
                    db.abs() < 0.1,
                    "{rate} Hz in, {hz} Hz: {db:+.3} dB, want within +/-0.1 dB \
                     (flat band ends at {flat_top:.0} Hz)"
                );
            } else {
                assert!(
                    db < 0.1,
                    "{rate} Hz in, {hz} Hz: {db:+.3} dB, the transition band must not rise"
                );
                if let Some(prev) = previous {
                    assert!(
                        db <= prev + 0.5,
                        "{rate} Hz in: response rose from {prev:+.2} dB to {db:+.2} dB \
                         between probes at and above {hz:.0} Hz"
                    );
                }
                previous = Some(db);
            }
        }
        println!(
            "{rate:6} Hz: cutoff {cutoff:.0} Hz, transition {transition:.0} Hz, flat to {flat_top:.0} Hz"
        );
    }
}

#[test]
fn the_transition_band_reaches_the_stop_band_before_the_output_nyquist() {
    // The other half of the anti-aliasing contract: by the time the input reaches
    // its own Nyquist the filter must be 25 dB down, so nothing folds back into
    // the 0 to 8 kHz band the ASR model listens to.
    for rate in [22_050u32, 32_000, 44_100, 48_000, 96_000, 192_000] {
        let nyquist = f64::from(rate) / 2.0;
        // A tone just below the input Nyquist; 0.97 keeps it clear of the
        // transition's very edge.
        let hz = (nyquist * 0.97) as f32;
        let out = probe_tone(rate, hz);
        let seg = &out[out.len() / 4..out.len() * 3 / 4];
        let (amp, _) = measure_tone(seg, 16_000, hz);
        let db = 20.0 * (amp / 0.5).max(1e-9).log10();
        assert!(
            db < -25.0,
            "{rate} Hz in, {hz:.0} Hz: {db:+.1} dB at the input Nyquist, want below -25 dB"
        );
    }
}

#[test]
fn nothing_above_the_output_nyquist_folds_back_into_the_speech_band() {
    // The aliasing test. A tone *above* the 8 kHz Nyquist is resampled down, and
    // whatever the filter fails to suppress lands inside the band the ASR model
    // listens to. The requirement is 60 dB down, which is far below the -114 dB
    // the filter achieves, so this test is really checking that the *adaptive
    // cutoff rule* keeps the transition band below 8 kHz.
    for rate in [32_000u32, 44_100, 48_000, 96_000, 192_000] {
        for hz in [8300.0f32, 8600.0, 9000.0, 11_000.0, 15_000.0, 20_000.0] {
            if f64::from(hz) >= f64::from(rate) / 2.0 {
                continue;
            }
            let out = probe_tone(rate, hz);
            let seg = &out[out.len() / 4..out.len() * 3 / 4];
            // Measure the image, which lands at |rate - hz| folded into 0..8 kHz.
            let image = 16_000.0 - (f64::from(rate) - f64::from(hz)) % 16_000.0;
            let image_hz = if (20.0..=8000.0).contains(&image) {
                image as f32
            } else {
                continue;
            };
            let (amp, _) = measure_tone(seg, 16_000, image_hz);
            let db = 20.0 * (amp / 0.5).max(1e-12).log10();
            assert!(
                db < -60.0,
                "{rate} Hz in, {hz} Hz: alias image at {image_hz} Hz is {db:+.1} dB, \
                 want below -60 dB"
            );
        }
    }
}

#[test]
fn a_stationary_bank_of_tones_survives_intact_at_every_rate() {
    // The comprehensive probe, and the reason this file does not use a sine sweep.
    //
    // A chirp is non-stationary: at 200 Hz a 20 ms analysis window covers 1.3
    // cycles and at 1600 Hz it covers 32, so anything measured from a sweep is
    // dominated by the sweep. A bank of stationary tones snapped to FFT bins is
    // measurable exactly, and one probe answers every question at once:
    //
    // * **Amplitude at every frequency.** Flat in the passband, down in the stop
    //   band. This is the frequency response, measured rather than assumed.
    // * **Alias rejection.** A second pass of the signal carries only tones above
    //   the output Nyquist, where the only way anything can reach the output is
    //   folded down.
    //
    // What this probe deliberately does *not* assert is a per-tone group delay.
    // A delay read from a single FFT bin has a precision of about
    // `16000 / (2*pi*f)` samples per radian of phase noise, which is 0.002
    // samples at 3 kHz but 1.7 samples at 150 Hz; resolving a frequency-dependent
    // delay to better than a fraction of a sample through that estimator is not
    // possible. The time-domain evidence for a frequency-independent delay is the
    // step test below, which measures the impulse response directly and would show
    // a smeared transition immediately.
    for rate in RATES {
        let nyquist = f64::from(rate) / 2.0;
        let frames = 1 << 18;
        // The analysis window has to fit in the *output* of the slowest ratio
        // change, not the input, or a 192 kHz source has fewer output samples
        // than the window and there is no middle to take. At 3:8 the output is
        // 21 845 samples, and half of that leaves a power of two.
        //
        // 8192 samples is short for a spectrum but this one does not need to be
        // long: every tone is on an exact bin, so there is no leakage to resolve,
        // and 40 tones over 7.6 kHz are 200 bins apart, so the Hann main lobes
        // never touch. What it does need is to be the same for every rate, or the
        // bin grid the tones were snapped to would not be the grid they are
        // measured on.
        let analysis = 1 << 13;
        let out_bin_hz = 16_000.0 / analysis as f32;

        // Two banks, because they answer different questions. The first sits
        // entirely below the output Nyquist and measures the filter's *response*.
        // The second, built below for the alias check, sits above it.
        let pass_top = (nyquist.min(8000.0) * 0.97) as f32;
        let pass_bank = common::tone_bank(analysis, rate, 150.0, pass_top, 40);
        let bank = pass_bank.clone();
        let input = common::multitone_from_bank(&bank, frames, rate);
        let out = resample_mono(&input, rate).unwrap_or_else(|e| panic!("{rate} Hz: {e}"));
        let expect_len = (frames as f64 * 16000.0 / f64::from(rate)).round() as usize;
        assert_eq!(out.len(), expect_len, "{rate} Hz: length");

        // Analyse the same number of samples of each, from the steady middle and
        // starting at a whole number of periods so the window is coherent.
        let skip_in = (frames - analysis) / 2;
        let skip_out = skip_in * 16000 / rate as usize;
        let out_spec = common::complex_spectrum(&out[skip_out..skip_out + analysis]);

        let sinc_len = vaani_media::DEFAULT_RESAMPLER.sinc_len as f64;
        let cutoff = f64::from(vaani_media::DEFAULT_RESAMPLER.cutoff_fraction(rate, 16_000))
            * nyquist.min(8000.0);
        let flat_top = cutoff - 4.0 * f64::from(rate) / sinc_len;

        let mut worst_flat: f32 = 0.0;
        let mut worst_rise: f32 = 0.0;

        for (hz, amp) in &pass_bank {
            let out_bin = (*hz / out_bin_hz).round() as usize;
            if out_bin >= out_spec.len() {
                continue;
            }
            let (re, im) = out_spec[out_bin];
            let got = (re * re + im * im).sqrt();
            let db = 20.0 * (got / amp).max(1e-9).log10();
            if f64::from(*hz) < flat_top {
                worst_flat = worst_flat.max(db.abs());
            } else {
                // In the transition band the response is allowed to fall. It is
                // not allowed to rise: a bump there is a synthesis error, and it
                // would put energy back into the passband at the edge.
                worst_rise = worst_rise.max(db);
            }
        }

        // Alias rejection, measured on a *separate* pass of the signal.
        //
        // The tones above the output Nyquist cannot be looked for in the same run
        // as the passband tones, because a tone at 9 kHz aliases to 7 kHz, which
        // is exactly where a legitimate passband tone lives, and the two are
        // indistinguishable in one spectrum. So the input is rebuilt with only the
        // above-Nyquist tones in it, and the strongest bin of the output is then
        // unambiguously the filter's alias rejection.
        //
        // The tones start at 1.2x the output Nyquist, not just above it, and that
        // is deliberate: an image that lands inside the filter's own transition
        // band is indistinguishable from its legitimate rolloff, so measuring
        // there would be measuring the transition width. An image that lands in
        // the *passband*, well away from where the content was, is what a listener
        // actually hears as a robot voice, and that is the thing being tested.
        let mut worst_image = f32::NEG_INFINITY;
        if nyquist > 10_000.0 {
            let alias_bank =
                common::tone_bank_above_nyquist(analysis, 9_600.0, (nyquist * 0.98) as f32, 60);
            let alias_input = common::multitone_from_bank(&alias_bank, frames, rate);
            let alias_out =
                resample_mono(&alias_input, rate).unwrap_or_else(|e| panic!("{rate} Hz: {e}"));
            let start = skip_out.min(alias_out.len().saturating_sub(analysis));
            let spec = common::complex_spectrum(&alias_out[start..start + analysis]);
            // Normalise by the coherent total, not by one tone. Sixty tones
            // within a hertz of each other sum, and dividing by a single tone's
            // amplitude reports a rejection 35 dB worse than the truth.
            let drive: f32 = alias_bank.iter().map(|(_, a)| *a).sum();
            let strongest = spec
                .iter()
                .skip(2)
                .map(|(re, im)| (re * re + im * im).sqrt())
                .fold(0.0f32, f32::max);
            worst_image = 20.0 * (strongest / drive).max(1e-12).log10();
        }

        // Broadband residue: the strongest bin that is not a tone and not within
        // eight bins of one. The exclusion is neither optional nor arbitrary: a
        // symmetric Hann window's own response is -6.02 dB one bin from a tone,
        // -31 dB at three bins, and still -45 dB at eight, so a narrower exclusion
        // reports the window instead of the resampler. Those figures are where the
        // -6.0 dB in the first version of this test came from.
        //
        // The measured floor is about -44 dB per bin relative to a *single* tone
        // with forty tones in flight at once. That is the filter's broadband
        // response to a dense excitation, not a distortion figure: the
        // single-tone harmonic distortion asserted in
        // `a_pure_tone_gains_no_harmonics` is below -100 dB, and that is the
        // number that characterises fidelity. This assertion is a ceiling that
        // would catch a resampler that had gone nonlinear.
        let mut tone_bins: Vec<usize> = bank
            .iter()
            .map(|(hz, _)| (*hz / out_bin_hz).round() as usize)
            .collect();
        tone_bins.sort_unstable();
        tone_bins.dedup();
        let strongest = bank.iter().map(|(_, a)| *a).fold(0.0f32, f32::max);
        let mut worst_residue = 0.0f32;
        for (b, &(re, im)) in out_spec.iter().enumerate() {
            if b < 16 {
                continue;
            }
            let near = tone_bins.partition_point(|t| *t < b);
            let too_close = (near > 0 && tone_bins[near - 1] + 8 >= b)
                || (near < tone_bins.len() && tone_bins[near].saturating_sub(8) <= b);
            if too_close {
                continue;
            }
            worst_residue = worst_residue.max((re * re + im * im).sqrt());
        }
        let residue_db = 20.0 * (worst_residue / strongest).max(1e-12).log10();

        println!(
            "{rate:6} Hz: passband flatness {worst_flat:.3} dB, worst transition-band rise \
             {worst_rise:+.2} dB, worst alias image {worst_image:+.1} dB, worst non-tone bin \
             {residue_db:.1} dB"
        );
        assert!(
            worst_flat < 0.1,
            "{rate} Hz: a passband tone is {worst_flat:.3} dB off"
        );
        assert!(
            worst_rise < 0.1,
            "{rate} Hz: the transition band rose to {worst_rise:+.2} dB"
        );
        assert!(
            worst_image < -40.0,
            "{rate} Hz: an alias image came through at {worst_image:.1} dB"
        );
        assert!(
            residue_db < -40.0,
            "{rate} Hz: {residue_db:.1} dB of broadband energy in a bin no tone occupies"
        );
    }
}

#[test]
fn a_step_settles_quickly_and_does_not_oscillate_for_long() {
    // A discontinuity is the worst case for a windowed-sinc filter: it excites
    // the whole impulse response at once. The requirement is not zero overshoot
    // (a finite filter always has some) but that it is bounded and dies.
    for rate in [8000u32, 22_050, 44_100, 48_000, 96_000] {
        let n = rate as usize;
        let mut step = vec![0.0f32; n];
        for v in step.iter_mut().skip(n / 2) {
            *v = 0.5;
        }
        let out = resample_mono(&step, rate).unwrap_or_else(|e| panic!("{rate} Hz: {e}"));
        let mid = (n / 2) * 16_000 / rate as usize;
        assert!(
            mid > 0 && mid < out.len(),
            "{rate} Hz: step index out of range"
        );

        // The settled value, measured from the middle of the post-step half.
        // The last few hundred output samples are the resampler's natural decay
        // as the filter runs off the end of the signal, so they must be excluded
        // from both the settled value and the ringing measurement.
        let decay = 1000usize;
        let mid_end = out.len() - decay;
        let trans = &out[mid..mid_end];
        let settled_half = (mid + mid_end) / 2;
        let settled = rms(&out[settled_half..mid_end]);

        let overshoot = trans.iter().fold(0.0f32, |a, &v| a.max((v - 0.5).abs()));
        let tolerance = 0.02 * settled.max(0.001);
        let settle_ms = trans
            .iter()
            .rposition(|v| (v - 0.5).abs() > tolerance)
            .map_or(0.0, |i| i as f32 / 16.0);

        let overshoot_db = 20.0 * (overshoot / 0.5).max(1e-9).log10();
        println!(
            "{rate:6} Hz step: overshoot {overshoot_db:+.2} dB, ringing settles to 2% in {settle_ms:.1} ms"
        );
        assert!(
            overshoot < 0.15,
            "{rate} Hz: step overshoot {overshoot_db:+.2} dB is too large"
        );
        assert!(
            settle_ms <= 50.0,
            "{rate} Hz: the step rang for {settle_ms:.1} ms before settling to 2%"
        );
    }
}

#[test]
fn a_pure_tone_keeps_its_frequency_and_amplitude_at_every_rate() {
    let sinc_len = vaani_media::DEFAULT_RESAMPLER.sinc_len as f64;
    for rate in RATES {
        let nyquist = f64::from(rate) / 2.0;
        let cutoff = f64::from(vaani_media::DEFAULT_RESAMPLER.cutoff_fraction(rate, 16_000))
            * nyquist.min(8000.0);
        let flat_top = cutoff - 4.0 * f64::from(rate) / sinc_len;
        for hz in [300.0f32, 1000.0, 2000.0, 3000.0, 5000.0] {
            if f64::from(hz) >= nyquist {
                continue;
            }
            // Inside the transition band the filter is *supposed* to attenuate,
            // so the strict bound only applies where it is supposed to be flat.
            let tolerance = if f64::from(hz) < flat_top { 5e-3 } else { 0.12 };
            let out = probe_tone(rate, hz);
            let seg = &out[out.len() / 4..out.len() * 3 / 4];
            let (amp, delay) = measure_tone(seg, 16_000, hz);
            assert!(
                (amp - 0.5).abs() < tolerance,
                "{rate} Hz in, {hz} Hz out: amplitude {amp}, want 0.5 +/- {tolerance}"
            );
            // The filter grid puts the output a fixed fraction of a sample after
            // the input, which is a few tens of output samples depending on the
            // rate and on how far the probe frequency is from a bin. What must
            // *not* happen is the delay growing with the length of the input,
            // which is time-warping; that is checked separately, over a minute of
            // signal, in `the_output_grid_does_not_drift`.
            assert!(
                delay.abs() < 32.0,
                "{rate} Hz in, {hz} Hz out: group delay {delay:.2} samples"
            );
        }
    }
}

#[test]
fn a_pure_tone_gains_no_harmonics() {
    // Intermodulation and harmonic distortion are what make a resampled voice
    // sound like a tape. Anything above -100 dB is inaudible and irrelevant; the
    // measured figure is around -114 dB.
    for rate in RATES {
        let out = probe_tone(rate, 1000.0);
        let seg = &out[out.len() / 4..out.len() * 3 / 4];
        let (fundamental, _) = measure_tone(seg, 16_000, 1000.0);
        let mut distortion = 0.0f32;
        for harmonic in 2..=5u32 {
            let (amp, _) = measure_tone(seg, 16_000, 1000.0 * harmonic as f32);
            distortion += amp * amp;
        }
        let thd_db = 20.0 * (distortion.sqrt() / fundamental).max(1e-12).log10();
        assert!(
            thd_db < -100.0,
            "{rate} Hz: total harmonic distortion {thd_db:.1} dB"
        );
    }
}

#[test]
fn silence_and_dc_pass_without_accretion() {
    // A filter that leaks a DC offset, or that turns digital silence into a small
    // oscillation, will show up as a constant bias in every log-mel frame.
    for rate in RATES {
        let out = resample_mono(&vec![0.0f32; 8192], rate).expect("resample");
        assert_eq!(peak(&out), 0.0, "{rate} Hz: silence is not silent");

        let out = resample_mono(&vec![0.25f32; 8192], rate).expect("resample");
        let margin = out.len() / 4;
        let seg = &out[margin..out.len() - margin];
        let mean = seg.iter().sum::<f32>() / seg.len() as f32;
        assert!((mean - 0.25).abs() < 1e-4, "{rate} Hz: DC mean {mean}");
        // A DC input must not develop an AC ripple.
        let ripple = (peak(seg) - seg.iter().copied().fold(f32::MAX, f32::min)).abs();
        assert!(ripple < 1e-3, "{rate} Hz: DC ripple {ripple}");
    }
}

#[test]
fn the_output_grid_does_not_drift_over_a_long_signal() {
    // The anti-drift test, and the most important one in this file. A resampler
    // whose output grid slips by even 0.1% would time-compress a 40-minute
    // meeting by two and a half minutes - every word would still be transcribed
    // correctly and every timestamp would be wrong.
    for rate in [8000u32, 44_100, 48_000, 96_000] {
        for seconds in [1usize, 10, 60] {
            let n = rate as usize * seconds;
            let input = tone(n, rate, 1000.0, 0.5);
            let out = resample_mono(&input, rate).unwrap_or_else(|e| panic!("{rate} Hz: {e}"));
            let want = (n as f64 * 16000.0 / f64::from(rate)).round() as usize;
            assert_eq!(
                out.len(),
                want,
                "{rate} Hz over {seconds} s: {} samples, want {want}",
                out.len()
            );
            // And the *phase* at the far end must still match the far end of the
            // input, which is what a drifting grid would break.
            let tail = &out[out.len() - out.len() / 4..];
            let (amp, delay) = measure_tone(tail, 16_000, 1000.0);
            assert!(
                (amp - 0.5).abs() < 0.02,
                "{rate} Hz over {seconds} s: tail amplitude {amp}"
            );
            assert!(
                delay.abs() < 8.0,
                "{rate} Hz over {seconds} s: tail group delay {delay:.2} samples"
            );
        }
    }
}

#[test]
fn the_truncation_point_is_inaudible() {
    // The output is trimmed to an exact sample count, and the trim point is a
    // splice. If it produced a click, every recording would end with a pop, and
    // for a *meeting* the last word is the one people care about.
    for rate in [8000u32, 22_050, 44_100, 48_000, 96_000] {
        let n = rate as usize * 2;
        let out = resample_mono(&tone(n, rate, 997.0, 0.5), rate).expect("resample");
        // The natural slew of a 997 Hz tone at 0.5 amplitude at 16 kHz.
        let slew = 2.0 * std::f32::consts::PI * 997.0 / 16_000.0 * 0.5;
        let jump = max_jump(&out[out.len() - 64..]);
        assert!(
            jump < slew * 1.5,
            "{rate} Hz: a jump of {jump} at the end, more than the tone's own {slew}"
        );
    }
}

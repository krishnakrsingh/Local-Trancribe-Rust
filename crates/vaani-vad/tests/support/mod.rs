//! Deterministic synthetic audio fixtures for the VAD tests.
//!
//! Everything here is generated in-process. Nothing is read from disk, and
//! nothing is copied from the private meeting recording: the fixtures exist so
//! that a test can state *"I planted speech at these exact times and at this
//! SNR, and the detector found it"*, which is a claim that is only meaningful
//! against a signal whose ground truth is known by construction.
//!
//! # Why synthesised and not sampled
//!
//! A real recording has no ground truth. You can listen to it and decide
//! whether a segment is speech, but you cannot put that decision in a test
//! without a human in the loop, and a human in the loop is a flaky test. The
//! cost of synthesis is that the fixtures are not *exactly* speech, so the
//! tests can pass on a signal that is easier than real speech. The
//! countermeasure is to make the synthetic voice genuinely hard: a harmonic
//! comb with a moving fundamental, a moving formant, a 4-8 Hz amplitude
//! envelope, fricative bursts with the right spectral tilt, and jitter -- see
//! [`speech_like`].
//!
//! # The random source
//!
//! `Rng` is a xorshift64* with a fixed seed, not a dependency on a crate. Three
//! reasons: the manifest is frozen, a fixture that changes between runs cannot
//! be used to measure a tuning regression, and `f32::sin`-based hashes have
//! platform-dependent tails that turn a "sometimes" into a "sometimes on CI".

// `tests/calibration.rs` and `tests/real_data.rs` each compile this module
// separately and use different halves of it, so a helper that only one of them
// calls is dead code in the other binary. Deleting it would mean the two
// binaries stop sharing a fixture -- the whole point of this module -- so the
// lint is switched off here, once, with a note.
#![allow(dead_code)]

use vaani_vad::dsp::{HOP_SAMPLES, SAMPLE_RATE, WINDOW_SAMPLES};

pub mod wav;

/// The frame hop in milliseconds, i.e. 10 at 16 kHz with a 160-sample hop.
pub const HOP_MS: i64 = (HOP_SAMPLES as i64 * 1000) / SAMPLE_RATE as i64;

/// A deterministic xorshift64* generator.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Seeded. Any seed is valid; zero is remapped because xorshift is stuck
    /// at zero.
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    /// The next `u64`.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f32 {
        // 24 bits is the f32 mantissa, so this is exact and never rounds to 1.0.
        ((self.next_u64() >> 40) as f32) / (1u32 << 24) as f32
    }

    /// Uniform in `[-1, 1)`.
    pub fn bipolar(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }

    /// Approximately Gaussian, by the sum of four uniforms (Irwin-Hall).
    ///
    /// Not Box-Muller: it needs a logarithm and a rejection loop, and a sum of
    /// uniforms has a bounded support and a well-understood variance, which is
    /// what a fixture wants.
    pub fn gaussian(&mut self) -> f32 {
        (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 1.732_050_8
    }
}

/// Milliseconds to a sample count at 16 kHz.
pub fn ms_to_samples(ms: i64) -> usize {
    (ms * SAMPLE_RATE as i64 / 1000) as usize
}

/// Samples to milliseconds at 16 kHz, rounded to the nearest millisecond.
pub fn samples_to_ms(n: usize) -> i64 {
    (n as i64 * 1000 + SAMPLE_RATE as i64 / 2) / SAMPLE_RATE as i64
}

/// A stretch of audio described in milliseconds, so a fixture reads as a
/// timeline rather than as a sample count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// Start in milliseconds from the beginning of the buffer.
    pub start_ms: i64,
    /// End in milliseconds, exclusive.
    pub end_ms: i64,
}

impl Span {
    /// Length in milliseconds.
    pub fn len_ms(&self) -> i64 {
        self.end_ms - self.start_ms
    }
}

/// What a fixture is made of, so a test can state its ground truth.
#[derive(Debug, Clone)]
pub struct Recipe {
    /// Spans containing speech-like audio.
    pub speech: Vec<Span>,
    /// Spans containing broadband white noise.
    pub noise: Vec<Span>,
    /// Spans containing a pure tone. `u32` is the frequency in Hz.
    pub tones: Vec<(Span, u32)>,
    /// Root mean square level of the noise spans, in dB. `-40.0` is a quiet
    /// room, `-20.0` is a loud one.
    pub noise_db: f32,
    /// Root mean square level of the speech spans, in dB. Set this to
    /// `noise_db + snr` to state the SNR of the fixture directly, which is
    /// what every test wants to control.
    pub speech_db: f32,
    /// Seed for the noise and the voice's jitter.
    pub seed: u64,
    /// Whether tone spans also carry 50/100/150 Hz mains hum, modelling a
    /// badly balanced capture.
    pub hum: bool,
    /// Fundamental of the synthetic voice, in Hz. `0.0` or less picks one in
    /// 95-185 Hz per span. 60 is the worst case for the detector's
    /// out-of-band veto and is worth a fixture of its own.
    pub f0_hz: f32,
    /// Level of the room's own noise floor, in dB, laid down across the whole
    /// buffer.
    ///
    /// **Not zero, and that matters.** A fixture whose gaps are *digital*
    /// silence at -200 dB is not a recording, it is a construction, and it
    /// breaks the detector's adaptive floor in a way no real input would: the
    /// floor's 20th percentile over a window containing any digital silence is
    /// the digital silence, so the reported SNR inside an utterance swung
    /// between 1 dB and 46 dB depending on how much of the trailing window
    /// happened to be silent. A real capture's quietest moment is a room, at
    /// -60 to -80 dB for a home and -45 dB for a live venue.
    ///
    /// The default of -100 dB is 60 dB below the noise spans, so it does not
    /// change any SNR, but it is a *number* rather than a mathematical zero and
    /// the floor tracker therefore has a stable target.
    pub room_db: f32,
}

impl Default for Recipe {
    fn default() -> Self {
        Self {
            speech: Vec::new(),
            noise: Vec::new(),
            tones: Vec::new(),
            noise_db: -40.0,
            speech_db: -25.0,
            seed: 1,
            hum: false,
            f0_hz: 0.0,
            room_db: -100.0,
        }
    }
}

/// A buffer plus the ground truth that goes with it.
#[derive(Debug, Clone)]
pub struct Fixture {
    /// Mono f32 samples at 16 kHz.
    pub samples: Vec<f32>,
    /// The spans that contain speech, in milliseconds.
    pub speech: Vec<Span>,
    /// The spans that contain nothing but noise or a tone.
    pub noise: Vec<Span>,
    /// Root mean square of the whole buffer, in dB.
    pub rms_db: f32,
}

impl Fixture {
    /// Build an `AudioBuffer` at time zero.
    pub fn audio(&self) -> vaani_core::media::AudioBuffer {
        vaani_core::media::AudioBuffer::mono(
            self.samples.clone(),
            SAMPLE_RATE,
            vaani_core::Time::ZERO,
        )
    }

    /// Length in milliseconds.
    pub fn len_ms(&self) -> i64 {
        samples_to_ms(self.samples.len())
    }

    /// Whether every planted speech span is detected.
    ///
    /// **Recall**, in the sense that matters: what fraction of the speech I
    /// planted is covered by at least one detected segment, as a fraction of
    /// each span's own duration rather than a hit/miss per span. A detector
    /// that finds the first 90 % of a long utterance and none of a short one
    /// scores better on hit/miss-per-span than one that finds all of it, and
    /// worse on this.
    pub fn recall(&self, timeline: &vaani_core::pipeline::SpeechTimeline) -> f64 {
        let total: i64 = self.speech.iter().map(|s| s.len_ms()).sum();
        if total == 0 {
            return 1.0;
        }
        let mut covered = 0i64;
        for s in &self.speech {
            for seg in &timeline.segments {
                let lo = seg.start.as_millis().max(s.start_ms);
                let hi = seg.end.as_millis().min(s.end_ms);
                if hi > lo {
                    covered += hi - lo;
                }
            }
        }
        covered as f64 / total as f64
    }

    /// Total detected milliseconds in the parts of the buffer that contain **no
    /// planted speech**.
    ///
    /// This is the false-positive figure, and it is deliberately defined as
    /// "everything that is not speech" rather than "the spans I happened to fill
    /// with noise". With a *continuous* noise floor the two differ by the
    /// speech itself, and measuring false positives against a noise span that
    /// overlaps the speech reports the detector's true positives as its errors.
    ///
    /// It is a **lower** bound on the error, because the 60 ms of padding around
    /// each detected segment bleeds into the neighbouring non-speech time and
    /// is counted here. A fixture that leaves a wide margin between speech and
    /// the region being scored is used when the figure has to be tight.
    pub fn false_positive_ms(&self, timeline: &vaani_core::pipeline::SpeechTimeline) -> i64 {
        let mut total = 0i64;
        for seg in &timeline.segments {
            let lo = seg.start.as_millis();
            let hi = seg.end.as_millis();
            // Walk the non-speech gaps this segment overlaps.
            let mut cursor = lo;
            for s in &self.speech {
                if s.end_ms <= cursor {
                    continue;
                }
                if s.start_ms >= hi {
                    break;
                }
                if s.start_ms > cursor {
                    total += (s.start_ms.min(hi) - cursor).max(0);
                }
                cursor = cursor.max(s.end_ms);
            }
            total += (hi - cursor).max(0);
        }
        total
    }

    /// Total milliseconds of the buffer that contain no planted speech.
    pub fn non_speech_ms(&self) -> i64 {
        self.len_ms() - self.speech.iter().map(|s| s.len_ms()).sum::<i64>()
    }

    /// Total milliseconds of the noise spans, for expressing the false-positive
    /// figure as a ratio.
    pub fn noise_ms(&self) -> i64 {
        self.noise.iter().map(|s| s.len_ms()).sum()
    }
}

/// A speech-like signal: a harmonic comb with a moving fundamental, a moving
/// formant, an amplitude envelope, fricative bursts, and jitter.
///
/// The point of the detail is that each feature the detector uses has to be
/// *earned*. A plain sine would be rejected by the steady-source veto; a fixed
/// harmonic comb with no envelope would be too; a comb with a fixed formant and
/// a fixed level would give the wrong `rolloff`; and an unmodulated comb is not
/// what speech looks like in any of the statistics that matter here.
///
/// `f0_hz` is the fundamental. The formant and the envelope are still derived
/// from `rng`, so two voices at the same pitch are not identical signals, but
/// the pitch itself is a parameter because it is the variable the detector's
/// out-of-band veto is most sensitive to: a 60 Hz fundamental is the worst case
/// for "a voice has energy in the formant band".
pub fn speech_like(len: usize, rng: &mut Rng, level: f32, f0_hz: f32) -> Vec<f32> {
    let mut out = vec![0f32; len];
    let f0_base = f0_hz;
    let f1_base = 400.0 + rng.unit() * 500.0;
    let syl_rate = 3.0 + rng.unit() * 3.0;
    let env_rate = 4.0 + rng.unit() * 4.0;
    // A real voice's spectrum is roughly flat to 500 Hz and falls at 6-12
    // dB/octave above the first formant, out to 3-4 kHz. A `1/h` source
    // spectrum with only 8-16 harmonics is *not* that: for a 62 Hz fundamental
    // its top harmonic is 496 Hz and the whole signal sits below the formant
    // band, which is a property of the generator and not of any voice. The
    // exponent and the harmonic count are set so the synthetic voice has
    // energy in 500-3500 Hz at every pitch, because otherwise a test of the
    // out-of-band veto measures the generator instead of the detector.
    let tilt = 0.7f32;
    let n_harm = 16 + (rng.unit() * 16.0) as usize;

    for (i, slot) in out.iter_mut().enumerate() {
        let t = i as f32 / SAMPLE_RATE as f32;
        // Jitter: a real voice is never perfectly periodic, and a perfectly
        // periodic signal makes the autocorrelation trivially perfect.
        let jitter = 1.0 + 0.012 * (2.0 * std::f32::consts::PI * 4.7 * t).sin();
        let f0 =
            (f0_base * jitter) * (1.0 + 0.05 * (2.0 * std::f32::consts::PI * syl_rate * t).sin());
        let f1 = f1_base * (1.0 + 0.35 * (2.0 * std::f32::consts::PI * syl_rate * 0.6 * t).sin());

        let mut s = 0f32;
        for h in 1..=n_harm {
            let fh = f0 * h as f32;
            if fh > SAMPLE_RATE as f32 / 2.0 - 100.0 {
                break;
            }
            // A 1/h^0.7 tilt is the source spectrum; the formant is a
            // resonance on top of it.
            let src = 1.0 / h as f32 * (1.0 / h as f32).powf(tilt - 1.0);
            let formant = {
                let d = (fh - f1) / (f0 * 2.5 + 200.0);
                1.0 / (1.0 + d * d)
            };
            s += src * formant * (2.0 * std::f32::consts::PI * fh * t).sin();
        }
        // A fricative component: high-frequency noise, present on about a third
        // of the envelope cycle, which is roughly the real unvoiced fraction.
        let unvoiced = if (2.0 * std::f32::consts::PI * syl_rate * t).sin() > 0.35 {
            1.0
        } else {
            0.0
        };
        let fric = unvoiced
            * 0.25
            * (rng.bipolar() * 0.5 + (2.0 * std::f32::consts::PI * 4200.0 * t).sin() * 0.5);
        let env =
            0.35 + 0.65 * (0.5 + 0.5 * (2.0 * std::f32::consts::PI * env_rate * t).sin()).abs();
        *slot = level * env * (s * 0.35 + fric);
    }
    out
}

/// Broadband noise at a target RMS in dB.
pub fn white_noise(len: usize, rng: &mut Rng, level_db: f32) -> Vec<f32> {
    // No sqrt(2) here: level_db is the RMS, and a Gaussian with standard
    // deviation 1 already has RMS 1. The factor is a sine's, and putting it
    // here silently makes every noise fixture 3 dB louder than it claims.
    let amp = 10f32.powf(level_db / 20.0);
    (0..len).map(|_| rng.gaussian() * amp).collect()
}

/// A steady tone plus optional mains hum, at a target RMS in dB.
pub fn tone_with_hum(len: usize, hz: f32, level_db: f32, hum: bool) -> Vec<f32> {
    let amp = 10f32.powf(level_db / 20.0) * std::f32::consts::SQRT_2;
    let mut out: Vec<f32> = (0..len)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            amp * (2.0 * std::f32::consts::PI * hz * t).sin()
        })
        .collect();
    if hum {
        for (i, v) in out.iter_mut().enumerate() {
            let t = i as f32 / SAMPLE_RATE as f32;
            // 50 Hz plus its 2nd and 3rd harmonics, which is what a badly
            // balanced capture actually contains.
            *v += amp
                * 0.6
                * ((2.0 * std::f32::consts::PI * 50.0 * t).sin()
                    + 0.5 * (2.0 * std::f32::consts::PI * 100.0 * t).sin()
                    + 0.25 * (2.0 * std::f32::consts::PI * 150.0 * t).sin());
        }
    }
    out
}

/// Add `a` into `b` at `at`, scaled by `gain`.
pub fn mix_into(b: &mut [f32], at: usize, a: &[f32], gain: f32) {
    for (i, v) in a.iter().enumerate() {
        if at + i >= b.len() {
            break;
        }
        b[at + i] += v * gain;
    }
}

/// A speech-like voice at a controlled fundamental, in Hz.
pub fn voice_at(len: usize, rng: &mut Rng, level: f32, f0_hz: f32) -> Vec<f32> {
    speech_like(len, rng, level, f0_hz)
}

/// Root mean square of a buffer, in dB.
pub fn rms_db(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return -200.0;
    }
    let acc: f32 = samples.iter().map(|v| v * v).sum();
    10.0 * (acc / samples.len() as f32).max(1e-20).log10()
}

/// Build a fixture from a recipe.
///
/// Speech spans get [`speech_like`] at `speech_db`; noise spans get
/// [`white_noise`] at `noise_db`; tone spans get a steady tone. Everything else
/// is digital silence. Setting `speech_db = noise_db + snr` states the SNR of
/// the fixture exactly, which is what the tests need in order to claim
/// anything about sensitivity.
pub fn build(recipe: &Recipe) -> Fixture {
    let total_ms = recipe
        .speech
        .iter()
        .chain(recipe.noise.iter())
        .chain(recipe.tones.iter().map(|(s, _)| s))
        .map(|s| s.end_ms)
        .max()
        .unwrap_or(0);
    let n = ms_to_samples(total_ms);
    // The room first, then the noise spans, then the speech, so every fixture
    // has the same noise floor everywhere.
    let mut samples = white_noise(n, &mut Rng::new(recipe.seed ^ 0x5EED), recipe.room_db);
    let mut rng = Rng::new(recipe.seed);
    for s in &recipe.speech {
        let len = ms_to_samples(s.len_ms());
        // f0 <= 0 means "pick one in the 95-185 Hz range", and a different one
        // per span, so two spans in one fixture are not the same voice.
        let f0 = if recipe.f0_hz > 0.0 {
            recipe.f0_hz
        } else {
            95.0 + rng.unit() * 90.0
        };
        // Generated at unit level, then scaled so the *achieved* RMS is
        // `speech_db`. Without this the label is a generator parameter and not
        // a measurement: the harmonic comb's RMS depends on the harmonic count,
        // the tilt exponent, the pitch and the envelope, and came out about
        // 16 dB below the requested level. Every SNR number in the report would
        // then have been 16 dB wrong in the flattering direction, which is the
        // worst way to be wrong.
        let mut v = speech_like(len, &mut rng, 1.0, f0);
        let measured = rms_db(&v);
        let scale = 10f32.powf((recipe.speech_db - measured) / 20.0);
        for x in &mut v {
            *x *= scale;
        }
        mix_into(&mut samples, ms_to_samples(s.start_ms), &v, 1.0);
    }
    for s in &recipe.noise {
        let n = ms_to_samples(s.len_ms());
        let v = white_noise(n, &mut rng, recipe.noise_db);
        mix_into(&mut samples, ms_to_samples(s.start_ms), &v, 1.0);
    }
    for (s, hz) in &recipe.tones {
        let n = ms_to_samples(s.len_ms());
        let v = tone_with_hum(n, *hz as f32, recipe.noise_db + 6.0, recipe.hum);
        mix_into(&mut samples, ms_to_samples(s.start_ms), &v, 1.0);
    }

    Fixture {
        rms_db: rms_db(&samples),
        samples,
        speech: recipe.speech.clone(),
        noise: recipe.noise.clone(),
    }
}

/// Split a score sequence's total into speech and non-speech, for reporting.
///
/// A convenience for the tuning harness: given the planted spans, return the
/// mean score inside them and outside them, which is the two numbers a
/// threshold sweep actually needs.
pub fn score_separation(scores: &[f32], speech: &[Span]) -> (f64, f64) {
    let in_speech = |ms: i64| speech.iter().any(|s| s.start_ms <= ms && ms < s.end_ms);
    let (mut a, mut na) = (0f64, 0usize);
    let (mut b, mut nb) = (0f64, 0usize);
    for (i, &s) in scores.iter().enumerate() {
        if in_speech(i as i64 * HOP_MS) {
            a += s as f64;
            na += 1;
        } else {
            b += s as f64;
            nb += 1;
        }
    }
    (
        if na == 0 { 0.0 } else { a / na as f64 },
        if nb == 0 { 0.0 } else { b / nb as f64 },
    )
}

/// Percentiles of a slice, by linear interpolation. `q` in `[0, 1]`.
///
/// Hand-rolled rather than from a crate: the manifest is frozen, and a fixture
/// that needs a sort is not a fixture anybody should be writing.
pub fn percentiles(values: &[f32], qs: &[f32]) -> Vec<f64> {
    if values.is_empty() {
        return qs.iter().map(|_| f64::NAN).collect();
    }
    let mut v: Vec<f32> = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    qs.iter()
        .map(|&q| {
            let q = q.clamp(0.0, 1.0);
            let idx = q * (v.len() - 1) as f32;
            let lo = idx.floor() as usize;
            let hi = idx.ceil() as usize;
            let frac = idx - lo as f32;
            (v[lo] as f64) * (1.0 - frac as f64) + (v[hi] as f64) * frac as f64
        })
        .collect()
}

/// The mean of a slice, or `None` if empty.
pub fn mean(values: &[f32]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().map(|v| *v as f64).sum::<f64>() / values.len() as f64)
}

/// The median of a slice, via [`percentiles`].
pub fn median(values: &[f32]) -> Option<f64> {
    percentiles(values, &[0.5]).into_iter().next()
}

/// The number of analysis frames a buffer of `len` samples produces.
///
/// The same formula as `EnergyVad::analyse`, exposed so a test can build a
/// score array of the right length without duplicating the arithmetic.
pub fn frame_count(len: usize) -> usize {
    if len <= WINDOW_SAMPLES {
        1
    } else {
        1 + (len - WINDOW_SAMPLES) / HOP_SAMPLES
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_generator_is_reproducible() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut c = Rng::new(8);
        let mut a = Rng::new(7);
        assert_ne!(a.next_u64(), c.next_u64());
    }

    #[test]
    fn unit_is_in_range_and_never_exactly_one() {
        let mut r = Rng::new(3);
        for _ in 0..10_000 {
            let u = r.unit();
            assert!((0.0..1.0).contains(&u), "{u}");
        }
    }

    #[test]
    fn gaussian_has_the_intended_variance() {
        // Four uniforms have variance 4/12 = 1/3, so the scale factor is
        // sqrt(3) = 1.7320508.
        let mut r = Rng::new(11);
        let n = 200_000;
        let vals: Vec<f32> = (0..n).map(|_| r.gaussian()).collect();
        let mean = vals.iter().sum::<f32>() / n as f32;
        let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "variance {var}");
    }

    #[test]
    fn a_fixture_has_the_length_its_recipe_implies() {
        let f = build(&Recipe {
            speech: vec![Span {
                start_ms: 0,
                end_ms: 1_000,
            }],
            noise: vec![Span {
                start_ms: 1_000,
                end_ms: 2_000,
            }],
            noise_db: -40.0,
            seed: 1,
            ..Recipe::default()
        });
        assert_eq!(f.len_ms(), 2_000);
        assert_eq!(f.samples.len(), 32_000);
    }

    #[test]
    fn noise_only_fixtures_measure_at_their_requested_level() {
        let f = build(&Recipe {
            noise: vec![Span {
                start_ms: 0,
                end_ms: 4_000,
            }],
            noise_db: -40.0,
            seed: 2,
            ..Recipe::default()
        });
        assert!(
            (f.rms_db + 40.0).abs() < 0.5,
            "requested -40 dB, measured {}",
            f.rms_db
        );
    }

    #[test]
    fn speech_spans_are_audible_over_the_noise_floor() {
        let f = build(&Recipe {
            speech: vec![Span {
                start_ms: 0,
                end_ms: 4_000,
            }],
            noise: vec![Span {
                start_ms: 4_000,
                end_ms: 8_000,
            }],
            noise_db: -40.0,
            seed: 3,
            ..Recipe::default()
        });
        let speech_rms = rms_db(&f.samples[..ms_to_samples(4_000)]);
        let noise_rms = rms_db(&f.samples[ms_to_samples(4_000)..]);
        assert!(
            speech_rms - noise_rms > 9.0,
            "speech {speech_rms} vs noise {noise_rms}"
        );
    }
}

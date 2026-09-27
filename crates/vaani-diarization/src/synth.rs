//! Synthetic voices and turn timelines, for tests and for the benchmark.
//!
//! # Why this is a public module
//!
//! The end-to-end correctness test for a clustering engine is a recording
//! whose speaker identity is known exactly, and the only honest way to get one
//! without a human annotating a real meeting is to synthesise it. That makes
//! the generator part of the crate's test surface, not a test helper buried in
//! a `#[cfg(test)]` block, because the integration tests in `tests/` need it
//! and integration tests can only see the public API.
//!
//! It is also what the benchmark uses to produce a reproducible RTF number: a
//! deterministic 14-minute meeting, the same every run, on every machine.
//!
//! # What a synthetic "voice" is
//!
//! Not a pure tone. A pure tone is trivially separable and would make the
//! clustering look better than it is. Each [`Voice`] is:
//!
//! * a fundamental that **drifts** over an utterance, and carries vibrato —
//!   a real voice is never at one frequency, and a static F0 would make the
//!   pitch feature look far more stable than it is;
//! * a harmonic series with a per-voice rolloff, standing in for a vocal
//!   tract's spectral envelope;
//! * three formant boosts, so the spectral shape is not just a falling line;
//! * a breathiness term: shaped noise mixed in, which is what keeps the
//!   autocorrelation from being a perfect delta function and stops the
//!   features from being degenerate.
//!
//! Two voices are therefore separated by *four* things at once — median F0,
//! harmonic rolloff, formant placement, breathiness — which is what makes a
//! pass on this fixture evidence about the feature vector rather than about
//! one lucky dimension.

/// A deterministic xorshift generator.
///
/// A hand-rolled PRNG rather than a dependency: the manifest is frozen, and a
/// test fixture must be reproducible across machines and across Rust versions,
/// which a library's `rand` cannot promise.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// A generator from an explicit seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    /// The next value in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        // xorshift64*, the standard three-shift generator.
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((v >> 40) as f32) / 16_777_216.0
    }

    /// The next value in `[-1, 1)`.
    pub fn next_bipolar(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }
}

/// A synthetic speaker.
#[derive(Debug, Clone, PartialEq)]
pub struct Voice {
    /// Fundamental at the start of the utterance, Hz.
    pub f0: f32,
    /// How far the fundamental drifts across an utterance, as a fraction.
    pub f0_drift: f32,
    /// Number of harmonics to synthesise.
    pub harmonics: usize,
    /// Amplitude of harmonic `k` relative to the fundamental, in dB.
    pub harmonic_rolloff_db: f32,
    /// The three formant centre frequencies, Hz.
    pub formants_hz: [f32; 3],
    /// How much each formant boosts its neighbourhood, dB.
    pub formant_boost_db: f32,
    /// How much shaped noise is mixed in, in `[0, 1]`.
    pub breathiness: f32,
    /// Vibrato rate, Hz.
    pub vibrato_hz: f32,
    /// Vibrato depth, as a fraction of the fundamental.
    pub vibrato_depth: f32,
}

impl Voice {
    /// A voice with a straight harmonic rolloff and no formants or vibrato.
    ///
    /// This is the "simplest thing that is still a voice" and the right
    /// starting point for a test that wants to control exactly one thing.
    #[must_use]
    pub fn simple(f0: f32, harmonics: usize) -> Self {
        Self {
            f0,
            f0_drift: 0.06,
            harmonics,
            harmonic_rolloff_db: -9.0,
            formants_hz: [500.0, 1_500.0, 2_500.0],
            formant_boost_db: 0.0,
            breathiness: 0.0,
            vibrato_hz: 5.0,
            vibrato_depth: 0.01,
        }
    }

    /// The fundamental frequency actually rendered, in Hz.
    ///
    /// **Not** the `f0` field. `render` scales by `(1 + f0_drift)`, so a voice
    /// declaring 118 Hz with a drift of 0.10 renders at 129.8 Hz. Using the
    /// field as ground truth in a pitch test is wrong by two semitones before
    /// any measurement happens, and the drift is applied to `f0_drift` as well
    /// at the start of an utterance, so the first 300 ms sit higher still.
    ///
    /// This is the mid-utterance value, where the drift ramp has saturated.
    #[must_use]
    pub fn nominal_f0(&self) -> f32 {
        self.f0 * (1.0 + self.f0_drift)
    }

    /// Amplitude of harmonic `k`, `k = 1` being the fundamental.
    #[must_use]
    pub fn harmonic_amplitude(&self, k: usize) -> f32 {
        let k = k.max(1) as f32;
        let mut a = 10f32.powf(self.harmonic_rolloff_db * (k - 1.0) / 20.0);
        if self.formant_boost_db != 0.0 {
            let f = k * self.f0;
            let mut boost = 0f32;
            for centre in self.formants_hz {
                // A one-octave-wide bump, which is roughly a formant bandwidth.
                let octaves = (f / centre).log2();
                boost += (-4.0 * octaves * octaves).exp();
            }
            a *= 10f32.powf(self.formant_boost_db * boost / 20.0);
        }
        a
    }

    /// Synthesise `n` samples of this voice, starting at `phase` radians.
    ///
    /// `t0_seconds` is the offset into the utterance, used for the pitch
    /// drift and the vibrato, so a long utterance does not sound like the same
    /// note repeated.
    #[must_use]
    pub fn render(
        &self,
        n: usize,
        sample_rate: u32,
        t0_seconds: f64,
        phase: f32,
        rng: &mut Rng,
    ) -> Vec<f32> {
        let mut out = vec![0f32; n];
        let f0_nominal = self.f0 * (1.0 + self.f0_drift);
        let mut phase_acc = phase;
        for (i, sample) in out.iter_mut().enumerate() {
            let t = t0_seconds + i as f64 / sample_rate as f64;
            // The drift is a half-cycle ramp across the utterance, so the voice
            // starts high, ends low, and is stationary in the middle.
            let ramp = (2.0 * t / 0.6).clamp(-1.0, 1.0) as f32;
            let drift = self.f0_drift * 0.5 * (1.0 - ramp);
            let vibrato = self.vibrato_depth
                * (2.0 * std::f32::consts::PI * self.vibrato_hz * t as f32).sin();
            let f0 = f0_nominal * (1.0 + drift + vibrato).max(0.5);
            let w = 2.0 * std::f32::consts::PI * f0 / sample_rate as f32;
            phase_acc += w;
            if phase_acc > 2.0 * std::f32::consts::PI {
                phase_acc -= 2.0 * std::f32::consts::PI;
            }
            let mut v = 0f32;
            for k in 1..=self.harmonics {
                let f = k as f32 * f0;
                if f > sample_rate as f32 * 0.45 {
                    break;
                }
                v += self.harmonic_amplitude(k) * (phase_acc * k as f32).sin();
            }
            if self.breathiness > 0.0 {
                v = (1.0 - self.breathiness) * v + self.breathiness * rng.next_bipolar() * 0.3;
            }
            *sample = v;
        }
        // Normalise so the voice's loudness is set by the caller, not by how
        // many harmonics it happened to be given.
        let peak = out.iter().fold(0f32, |m, v| m.max(v.abs()));
        if peak > 1e-6 {
            out.iter_mut().for_each(|v| *v /= peak);
        }
        out
    }
}

/// One stretch of one voice on the timeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Utterance {
    /// Index into the voice list. Also the ground-truth speaker id.
    pub speaker: usize,
    /// Start, absolute, in milliseconds.
    pub start_ms: i64,
    /// End, absolute, in milliseconds.
    pub end_ms: i64,
    /// Linear gain for this stretch, so a speaker can be loud in one turn and
    /// quiet in the next without the features moving.
    pub gain: f32,
}

impl Utterance {
    /// An utterance at nominal gain.
    #[must_use]
    pub fn new(speaker: usize, start_ms: i64, end_ms: i64) -> Self {
        Self {
            speaker,
            start_ms,
            end_ms,
            gain: 1.0,
        }
    }

    /// Set the gain.
    #[must_use]
    pub fn with_gain(mut self, gain: f32) -> Self {
        self.gain = gain;
        self
    }
}

/// Render one continuous utterance whose fundamental glides from `f0_start` to
/// `f0_end`.
///
/// For a speaker's intonation: a question that rises through a clause and
/// settles. The glide is **continuous in phase** -- the fundamental's phase
/// accumulates across the whole utterance -- so there is no discontinuity at
/// any point and the result contains no clicks.
///
/// That detail is the entire reason this is not `render` called repeatedly in
/// pieces. Concatenating separate renders restarts the phase at each boundary,
/// and a phase step is a broadband discontinuity: the activity detector reads
/// it as flux and the cepstra jump, so a window spanning a boundary looks like a
/// different speaker. A fixture built that way measures its own construction,
/// not the engine, and it produced a within-speaker p90 of 1.95 for what should
/// have been a gentle rise.
#[must_use]
pub fn render_glide(
    voice: &Voice,
    f0_start: f32,
    f0_end: f32,
    n: usize,
    sample_rate: u32,
    rng: &mut Rng,
) -> Vec<f32> {
    let mut out = vec![0f32; n];
    let mut phase = 0f32;
    for (i, sample) in out.iter_mut().enumerate() {
        let t = if n <= 1 {
            0.0
        } else {
            i as f32 / (n - 1) as f32
        };
        let f0 = f0_start + (f0_end - f0_start) * t;
        let w = 2.0 * std::f32::consts::PI * f0 / sample_rate as f32;
        phase += w;
        if phase > 2.0 * std::f32::consts::PI {
            phase -= 2.0 * std::f32::consts::PI;
        }
        let mut v = 0f32;
        for k in 1..=voice.harmonics {
            let f = k as f32 * f0;
            if f > sample_rate as f32 * 0.45 {
                break;
            }
            v += voice.harmonic_amplitude(k) * (phase * k as f32).sin();
        }
        if voice.breathiness > 0.0 {
            v = (1.0 - voice.breathiness) * v + voice.breathiness * rng.next_bipolar() * 0.3;
        }
        *sample = v;
    }
    let peak = out.iter().fold(0f32, |m, v| m.max(v.abs()));
    if peak > 1e-6 {
        out.iter_mut().for_each(|v| *v /= peak);
    }
    out
}

/// Render a timeline of utterances into a mono buffer.
///
/// Samples outside every utterance are silence. The return value is
/// `(samples, per-frame ground truth)` where the ground truth is one label per
/// `frame_ms` of audio, `-1` for no speech. That ground truth is an **annotated
/// reference**, which is why the synthetic end-to-end test can report a
/// measured error rate and the real-recording run cannot.
#[must_use]
pub fn render(
    utterances: &[Utterance],
    voices: &[Voice],
    total_ms: i64,
    sample_rate: u32,
    noise_floor: f32,
    seed: u64,
) -> (Vec<f32>, Vec<i32>) {
    let n = (total_ms * sample_rate as i64 / 1_000).max(0) as usize;
    let mut out = vec![0f32; n];
    let mut rng = Rng::new(seed);
    let mut phase = 0f32;

    for u in utterances {
        let Some(voice) = voices.get(u.speaker) else {
            continue;
        };
        if u.end_ms <= u.start_ms {
            continue;
        }
        let lo = ((u.start_ms * sample_rate as i64 / 1_000).max(0) as usize).min(n);
        let hi = (((u.end_ms * sample_rate as i64 / 1_000).max(0) as usize).min(n)).max(lo);
        let rendered = voice.render(hi - lo, sample_rate, 0.0, phase, &mut rng);
        phase += 0.7;
        if phase > 2.0 * std::f32::consts::PI {
            phase -= 2.0 * std::f32::consts::PI;
        }
        for (i, v) in rendered.iter().enumerate() {
            // A short fade at each end, so a turn boundary is not a
            // discontinuity. A click is broadband and the activity detector
            // reads clicks as speech.
            let pos = i as f64 / rendered.len().max(1) as f64;
            let env = (pos * 40.0).min(1.0) * ((1.0 - pos) * 40.0).min(1.0);
            out[lo + i] += v * u.gain * env as f32;
        }
    }

    if noise_floor > 0.0 {
        for s in out.iter_mut() {
            *s += noise_floor * rng.next_bipolar();
        }
    }

    let frame_ms: i64 = 10;
    let frames = (n * 1000) / sample_rate as usize / frame_ms as usize;
    let mut truth = vec![-1i32; frames];
    for (fi, slot) in truth.iter_mut().enumerate() {
        let t = fi as i64 * frame_ms;
        for u in utterances {
            if u.end_ms <= u.start_ms {
                continue;
            }
            if t >= u.start_ms && t < u.end_ms {
                *slot = u.speaker as i32;
                break;
            }
        }
    }
    (out, truth)
}

/// Two clearly different voices: a low one with a steep harmonic rolloff and
/// strong formants, and a higher one with a shallow rolloff and no formants.
///
/// Four differences at once — median F0, rolloff, formant presence and
/// breathiness — so passing this fixture is evidence about the whole feature
/// vector and not about the pitch dimension alone.
#[must_use]
pub fn two_voices() -> Vec<Voice> {
    vec![
        Voice {
            f0: 118.0,
            f0_drift: 0.10,
            harmonics: 14,
            harmonic_rolloff_db: -11.0,
            formants_hz: [520.0, 1_180.0, 2_500.0],
            formant_boost_db: 9.0,
            breathiness: 0.10,
            vibrato_hz: 4.6,
            vibrato_depth: 0.014,
        },
        Voice {
            f0: 203.0,
            f0_drift: 0.07,
            harmonics: 20,
            harmonic_rolloff_db: -5.0,
            formants_hz: [800.0, 1_900.0, 3_100.0],
            formant_boost_db: 0.0,
            breathiness: 0.02,
            vibrato_hz: 6.1,
            vibrato_depth: 0.008,
        },
    ]
}

/// Three voices, for a harder instance: two of them deliberately close in
/// median F0 so the pitch dimension alone cannot separate them.
#[must_use]
pub fn three_voices() -> Vec<Voice> {
    let mut v = two_voices();
    v.push(Voice {
        f0: 132.0,
        f0_drift: 0.14,
        harmonics: 8,
        harmonic_rolloff_db: -16.0,
        formants_hz: [420.0, 2_100.0, 2_900.0],
        formant_boost_db: 6.0,
        breathiness: 0.18,
        vibrato_hz: 3.9,
        vibrato_depth: 0.02,
    });
    v
}

/// An alternating turn timeline: `speakers` laid end to end in `turn_ms`
/// chunks, starting at `start_ms`.
///
/// Deterministic, which is the point: the same call produces the same
/// timeline, so a test's measured numbers can be compared between runs.
#[must_use]
pub fn alternate(speakers: &[usize], start_ms: i64, turn_ms: i64, turns: usize) -> Vec<Utterance> {
    (0..turns)
        .map(|i| {
            let s = speakers[i % speakers.len()];
            // A deterministic per-turn gain in [0.55, 1.0] so a speaker is not
            // at constant loudness, which would make the level statistics
            // unrealistically stable.
            let gain = 0.55 + 0.45 * ((i * 37 % 17) as f32 / 17.0);
            Utterance::new(
                s,
                start_ms + i as i64 * turn_ms,
                start_ms + (i as i64 + 1) * turn_ms,
            )
            .with_gain(gain)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rng_is_deterministic_and_in_range() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..1_000 {
            let x = a.next_f32();
            assert_eq!(x, b.next_f32());
            assert!((0.0..1.0).contains(&x), "{x}");
        }
    }

    #[test]
    fn the_rng_does_not_get_stuck_at_zero() {
        let mut r = Rng::new(1);
        let values: Vec<f32> = (0..100).map(|_| r.next_f32()).collect();
        assert!(values.iter().any(|v| *v > 0.0));
        assert!(values.iter().all(|v| (0.0..1.0).contains(v)));
    }

    #[test]
    fn a_rendered_voice_is_bounded_and_normalised() {
        let v = Voice::simple(150.0, 12);
        let mut rng = Rng::new(1);
        let out = v.render(8_000, 16_000, 0.0, 0.0, &mut rng);
        assert_eq!(out.len(), 8_000);
        let peak = out.iter().fold(0f32, |m, x| m.max(x.abs()));
        assert!((peak - 1.0).abs() < 1e-3, "peak {peak}");
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn a_voice_with_harmonics_lands_energy_on_its_overtones() {
        let v = Voice {
            harmonics: 5,
            harmonic_rolloff_db: -6.0,
            ..Voice::simple(200.0, 5)
        };
        assert!(v.harmonic_amplitude(1) > v.harmonic_amplitude(3));
        assert!(v.harmonic_amplitude(3) > v.harmonic_amplitude(5));
    }

    #[test]
    fn a_steep_rolloff_really_is_steeper() {
        let steep = Voice {
            harmonic_rolloff_db: -14.0,
            ..Voice::simple(200.0, 8)
        };
        let shallow = Voice {
            harmonic_rolloff_db: -3.0,
            ..Voice::simple(200.0, 8)
        };
        let k = 6;
        assert!(steep.harmonic_amplitude(k) < shallow.harmonic_amplitude(k) * 0.5);
    }

    #[test]
    fn two_voices_differ_in_every_dimension_the_test_relies_on() {
        let v = two_voices();
        assert_eq!(v.len(), 2);
        assert!(v[1].f0 / v[0].f0 > 1.5, "median F0 must be well separated");
        assert!(v[1].harmonic_rolloff_db > v[0].harmonic_rolloff_db);
        assert!(v[0].formant_boost_db > v[1].formant_boost_db);
        assert!(v[0].breathiness > v[1].breathiness);
    }

    #[test]
    fn the_third_voice_is_close_in_pitch_to_the_first() {
        // The three-voice fixture is only hard if two of them are close in the
        // dimension that does most of the work.
        let v = three_voices();
        assert_eq!(v.len(), 3);
        let ratio = v[2].f0 / v[0].f0;
        assert!(ratio > 1.0 && ratio < 1.4, "ratio {ratio}");
    }

    #[test]
    fn rendering_an_empty_timeline_gives_silence() {
        let (buf, truth) = render(&[], &two_voices(), 1_000, 16_000, 0.0, 1);
        assert_eq!(buf.len(), 16_000);
        assert!(buf.iter().all(|v| v.abs() < 1e-9));
        assert!(truth.iter().all(|t| *t == -1));
    }

    #[test]
    fn the_ground_truth_labels_every_10ms_of_each_utterance() {
        let u = vec![Utterance::new(1, 500, 1_000)];
        let (buf, truth) = render(&u, &two_voices(), 2_000, 16_000, 0.0, 1);
        assert_eq!(truth.len(), 200);
        assert!(truth[..50].iter().all(|t| *t == -1));
        assert!(truth[50..100].iter().all(|t| *t == 1));
        assert!(truth[100..].iter().all(|t| *t == -1));
        assert_eq!(buf.len(), 32_000);
    }

    #[test]
    fn an_utterance_is_never_a_click() {
        // A turn boundary that is a discontinuity is broadband, and the
        // activity detector would read the click as speech. Check that the
        // fade actually happened.
        let u = vec![Utterance::new(0, 0, 400)];
        let (buf, _) = render(&u, &two_voices(), 400, 16_000, 0.0, 1);
        assert!(buf[0].abs() < 0.05, "starts at {}", buf[0]);
        let n = buf.len();
        assert!(buf[n - 1].abs() < 0.05, "ends at {}", buf[n - 1]);
    }

    #[test]
    fn the_noise_floor_is_additive_and_bounded() {
        let u = vec![Utterance::new(0, 0, 1_000)];
        let (quiet, _) = render(&u, &two_voices(), 1_000, 16_000, 0.0, 1);
        let (loud, _) = render(&u, &two_voices(), 1_000, 16_000, 0.01, 1);
        assert!(
            loud.iter()
                .any(|v| v.abs() > quiet.iter().fold(0f32, |m, x| m.max(x.abs())))
        );
        assert!(loud.iter().all(|v| v.abs() < 2.0));
    }

    #[test]
    fn a_turn_timeline_alternates_and_stays_in_bounds() {
        let u = alternate(&[0, 1], 0, 1_000, 6);
        assert_eq!(u.len(), 6);
        assert_eq!(
            u.iter().map(|x| x.speaker).collect::<Vec<_>>(),
            vec![0, 1, 0, 1, 0, 1]
        );
        for (i, x) in u.iter().enumerate() {
            assert_eq!(x.start_ms, i as i64 * 1_000);
            assert_eq!(x.end_ms, (i as i64 + 1) * 1_000);
            assert!(x.gain > 0.0 && x.gain <= 1.0);
        }
    }

    #[test]
    fn an_out_of_range_voice_index_is_skipped_not_panicked_on() {
        let u = vec![Utterance::new(9, 0, 1_000)];
        let (buf, _) = render(&u, &two_voices(), 1_000, 16_000, 0.0, 1);
        assert!(buf.iter().all(|v| v.abs() < 1e-9));
    }

    #[test]
    fn a_zero_length_or_inverted_utterance_is_ignored() {
        let u = vec![
            Utterance::new(0, 500, 500),
            Utterance::new(0, 900, 100),
            Utterance::new(0, 0, 400),
        ];
        let (buf, truth) = render(&u, &two_voices(), 1_000, 16_000, 0.0, 1);
        assert!(buf.iter().any(|v| v.abs() > 0.01));
        assert!(truth[..40].iter().all(|t| *t == 0));
        assert!(truth[40..].iter().all(|t| *t == -1));
    }

    #[test]
    fn rendering_is_reproducible() {
        let u = alternate(&[0, 1], 0, 800, 8);
        let (a, ta) = render(&u, &two_voices(), 8_000, 16_000, 0.005, 42);
        let (b, tb) = render(&u, &two_voices(), 8_000, 16_000, 0.005, 42);
        assert_eq!(a, b);
        assert_eq!(ta, tb);
    }

    #[test]
    fn utterance_clips_are_clamped_to_the_buffer() {
        let u = vec![Utterance::new(0, 900, 5_000)];
        let (buf, _) = render(&u, &two_voices(), 1_000, 16_000, 0.0, 1);
        assert_eq!(buf.len(), 16_000);
        assert!(buf.iter().all(|v| v.is_finite()));
    }
}

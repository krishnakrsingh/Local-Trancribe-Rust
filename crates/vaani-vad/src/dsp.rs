//! Framing and spectral analysis for the energy detector.
//!
//! # Why these constants are the ones they are
//!
//! This module deliberately reproduces the *conventions* of the frozen
//! front end in `crates/vaani-asr/src/frontend.rs` rather than inventing a
//! second set of its own:
//!
//! | | ASR front end (`asr::frontend`) | this module |
//! |---|---|---|
//! | sample rate | 16 000 | 16 000 |
//! | `n_fft` | 512 | 512 |
//! | `win_length` | 400 (25 ms) | 400 (25 ms) |
//! | `hop_length` | 160 (10 ms) | 160 (10 ms) |
//! | window | symmetric Hann | symmetric Hann |
//! | placement | **centred** in the 512 frame, offset 56 | **centred**, offset 56 |
//! | power spectrum | `z.re*z.re + z.im*z.im` read from bin `k` | identical |
//!
//! The reason is not numerical parity -- this module does not produce mel
//! features, so no golden test can compare it against anything. The reason is
//! **temporal**: the window and hop determine where a segment boundary can
//! possibly be placed. If the VAD ran on 20 ms/10 ms frames while the encoder
//! consumed 25 ms/10 ms features, every reported onset would be quantised to a
//! different grid than the features that describe it, and the 60 ms of padding
//! that hides quantisation error would have to grow to cover 10 ms of
//! systematic bias in *both* directions. Sharing the grid means the error is
//! bounded by the 25 ms window alone, which is what `pad_ms` is sized against.
//!
//! 25 ms / 10 ms is also the standard telephony and WebRTC frame geometry
//! (G.729, G.723.1, and WebRTC's `kVadDefaultFrameSizeMs`), chosen so a frame
//! holds at least two periods of the lowest pitch a VAD must track
//! (60 Hz -> 25 ms is 1.5 periods) while still being short enough that a
//! plosive is not smeared across a 50 ms span.
//!
//! # What is deliberately *not* shared
//!
//! The window and the mel filterbank are not read from `mel_window.f32` /
//! `mel_fb.f32`, and `vaani-asr` is not a dependency. Two reasons, in order of
//! importance:
//!
//! 1. **A VAD is not a mel-spectrogram front end.** This module needs
//!    spectral *flatness*, *rolloff* and *tonality* over the raw 257 bins.
//!    Projecting onto 128 overlapping mel triangles destroys exactly the
//!    information flatness is made of -- the triangles are deliberately
//!    overlapping and area-normalised, so `geometric_mean` of a mel spectrum
//!    is not the flatness of the signal.
//! 2. **`vaani-asr` is frozen and the dependency would point the wrong way.**
//!    Making the VAD stage depend on the recogniser means a change to the
//!    recogniser's pre-processing becomes a change to the VAD. See
//!    `INTEGRATION NOTES`: the helper that *should* be shared is the
//!    symmetric-Hann generator and the centred-window frame layout, not the
//!    mel projection.
//!
//! # The power-spectrum invariant
//!
//! `frontend.rs:297-301` carries a warning about a real bug: exploiting
//! Hermitian symmetry to read the imaginary part from the mirrored bin `n-k`
//! yields `re^2 + re^2`, not `re^2 + im^2`, because for a real input
//! `Re(X[n-k]) == Re(X[k])` and `Im(X[n-k]) == -Im(X[k])`. That is not a
//! performance micro-optimisation; it is a wrong-numbers bug that still
//! produces a plausible spectrum. This module reads `z.im` from bin `k`.

use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;

/// The sample rate both detectors require.
///
/// Fixed by the contract: the pipeline resamples to the model's rate before
/// calling, and `VadEngine::sample_rate()` reports this back so the pipeline
/// can check rather than discover the mismatch.
pub const SAMPLE_RATE: u32 = 16_000;

/// Analysis window, 25 ms at 16 kHz. See the module docs for why this is the
/// ASR front end's `win_length` and not a round 20 ms.
pub const WINDOW_SAMPLES: usize = 400;

/// Hop, 10 ms at 16 kHz. Matches the ASR front end's `hop_length` exactly.
pub const HOP_SAMPLES: usize = 160;

/// FFT size, 512.
pub const N_FFT: usize = 512;

/// Number of usable bins, `n_fft / 2 + 1`.
pub const N_FREQ: usize = N_FFT / 2 + 1;

/// Where the 400-sample window sits inside the 512-sample FFT frame.
///
/// `torch.stft` pads `win_length` on *both* sides to reach `n_fft`. The frozen
/// front end puts it at `[56, 456)`; so does this, for the reason given in the
/// module docs.
pub const WINDOW_OFFSET: usize = (N_FFT - WINDOW_SAMPLES) / 2;

/// Nominal analysis window in milliseconds, as `WINDOW_SAMPLES` implies.
pub const WINDOW_MS: i64 = (WINDOW_SAMPLES as i64 * 1000) / SAMPLE_RATE as i64;

/// Nominal hop in milliseconds, as `HOP_SAMPLES` implies.
pub const HOP_MS: i64 = (HOP_SAMPLES as i64 * 1000) / SAMPLE_RATE as i64;

/// `0.5 * (1 - cos(2*pi*i/(len-1)))`: **symmetric** Hann, the same definition
/// `torch.hann_window(len, periodic=False)` produces.
///
/// Symmetric rather than periodic, and that is not a stylistic choice. A
/// periodic window is `0.5 * (1 - cos(2*pi*i/len))`, which starts at exactly
/// 0 and never reaches 1; the symmetric form starts at 0 and *ends* at 0, so
/// consecutive windows tile without a step discontinuity at the seam. With a
/// 60 % overlap (400 window, 160 hop) a step at every seam injects a broadband
/// click at 100 Hz, which is precisely the band a broadband-noise discriminator
/// tries to measure. `frontend.rs:391-397` pins the same definition.
#[must_use]
pub fn symmetric_hann(len: usize) -> Vec<f32> {
    if len <= 1 {
        return vec![1.0; len];
    }
    let n = len as f32;
    (0..len)
        .map(|i| {
            let t = i as f32 / (n - 1.0);
            0.5 - 0.5 * (2.0 * std::f32::consts::PI * t).cos()
        })
        .collect()
}

/// Total power of a signal, ignoring the mean.
///
/// The mean is removed first because a near-constant offset -- a DC bias from
/// a bad capture, or the near-silent tail of a compressed stream -- otherwise
/// dominates the sum and makes a digital-silence frame look loud.
#[must_use]
pub fn ac_power(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    samples.iter().map(|v| (v - mean) * (v - mean)).sum()
}

/// Proportion of adjacent sample pairs whose sign differs, in [0, 1].
///
/// Exact zeros are skipped rather than counted as a crossing. Counting them
/// makes a hard-clipped signal (many consecutive zeros) report an arbitrarily
/// low ZCR, which is the opposite of what ZCR measures: the point of the
/// feature is that it is high for noise-like signals and low for tonal ones,
/// and a clipped signal is noise-like.
#[must_use]
pub fn zero_crossing_rate(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let mut crossings = 0usize;
    let mut counted = 0usize;
    let mut last_sign = 0i8;
    for v in samples {
        let s = if *v > 0.0 {
            1i8
        } else if *v < 0.0 {
            -1i8
        } else {
            continue;
        };
        if last_sign != 0 {
            counted += 1;
            if s != last_sign {
                crossings += 1;
            }
        }
        last_sign = s;
    }
    if counted == 0 {
        return 0.0;
    }
    crossings as f32 / counted as f32
}

/// A reusable 512-point real power-spectrum analyser.
///
/// Holds only scratch buffers and the FFT plan. It is deliberately **not**
/// `Sync`: `power()` mutates `buffer` and `power`, so a shared instance is a
/// data race. The detector constructs one per `detect()` call instead, which
/// costs a few microseconds of planning and removes the question entirely.
pub struct Spectrum {
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    window: Vec<f32>,
    buffer: Vec<Complex32>,
    power: Vec<f32>,
}

impl Spectrum {
    /// The analysis window, so a caller can normalise for its gain.
    #[must_use]
    pub fn window(&self) -> &[f32] {
        &self.window
    }

    /// `sum(w[i]^2)` over the window.
    ///
    /// This is the divisor that turns a windowed frame's total spectral power
    /// back into a signal mean square, and getting it wrong is a 26 dB error
    /// (the same mistake is a factor of `len`, i.e. 400, or 26 dB, and it is
    /// the reason this is a function rather than an inline expression).
    ///
    /// By Parseval, `sum_k |X[k]|^2 == n_fft * sum_n (w[n]x[n])^2`, and for a
    /// stationary signal `sum_n w[n]^2 x[n]^2 ~= (sum w^2) * mean(x^2)`. So
    /// `mean(x^2) ~= sum_k |X[k]|^2 / (n_fft * sum w^2)`.
    #[must_use]
    pub fn window_energy(&self) -> f32 {
        self.window.iter().map(|v| v * v).sum::<f32>().max(1e-9)
    }
}

impl std::fmt::Debug for Spectrum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Spectrum")
            .field("n_fft", &N_FFT)
            .field("n_freq", &N_FREQ)
            .field("window", &WINDOW_SAMPLES)
            .finish_non_exhaustive()
    }
}

impl Spectrum {
    /// Plan the transform and allocate the scratch buffers.
    #[must_use]
    pub fn new() -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        Self {
            fft,
            window: symmetric_hann(WINDOW_SAMPLES),
            buffer: vec![Complex32::ZERO; N_FFT],
            power: vec![0.0; N_FREQ],
        }
    }

    /// The power spectrum `|X[k]|^2` of one `WINDOW_SAMPLES`-long frame.
    ///
    /// `frame` is taken by value of its length only: it must be exactly
    /// [`WINDOW_SAMPLES`]. A shorter slice is zero-padded, which is the
    /// correct behaviour at the tail of a buffer -- a frame that hangs off the
    /// end of the audio genuinely contains only part-signal.
    ///
    /// The returned slice borrows `self` and is valid until the next call.
    pub fn power(&mut self, frame: &[f32]) -> &[f32] {
        for (i, slot) in self.buffer.iter_mut().enumerate() {
            let v = if (WINDOW_OFFSET..WINDOW_OFFSET + WINDOW_SAMPLES).contains(&i) {
                let idx = i - WINDOW_OFFSET;
                // Out-of-range samples contribute zero, which is what
                // `center = true` with constant padding means in the frozen
                // front end. No branch on the frame length is needed.
                frame.get(idx).copied().unwrap_or(0.0) * self.window[idx]
            } else {
                0.0
            };
            *slot = Complex32::new(v, 0.0);
        }
        self.fft.process(&mut self.buffer);

        for (k, p) in self.power.iter_mut().enumerate() {
            let z = self.buffer[k];
            // `im` is read from bin `k` itself, never from the mirror; see the
            // module docs.
            *p = z.re * z.re + z.im * z.im;
        }
        &self.power
    }
}

impl Default for Spectrum {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn symmetric_hann_has_the_torch_definition() {
        let w = symmetric_hann(5);
        // torch.hann_window(5, periodic=False) == [0, 0.5, 1, 0.5, 0]
        assert!(approx(w[0], 0.0, 1e-6), "w[0] = {}", w[0]);
        assert!(approx(w[2], 1.0, 1e-6), "w[2] = {}", w[2]);
        assert!(approx(w[4], 0.0, 1e-6), "w[4] = {}", w[4]);
        assert!(approx(w[1], 0.5, 1e-6), "w[1] = {}", w[1]);
    }

    #[test]
    fn symmetric_hann_is_symmetric_about_its_centre_and_is_not_the_periodic_form() {
        let w = symmetric_hann(400);
        for i in 0..400 {
            assert!(approx(w[i], w[399 - i], 1e-6), "asymmetric at {i}");
        }
        // The distinguishing property: the last sample is 0 for symmetric and
        // 0.5*(1-cos(2pi*399/400)) = 2.4e-5 for periodic. They differ at the
        // seam, which is where a 60 %-overlap tiling would click.
        assert!(approx(w[399], 0.0, 1e-6));
        let periodic_last = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * 399.0 / 400.0).cos();
        assert!(periodic_last > 1e-5, "the two forms must actually differ");
    }

    #[test]
    fn hann_of_length_one_does_not_divide_by_zero() {
        let w = symmetric_hann(1);
        assert_eq!(w, vec![1.0]);
        assert!(symmetric_hann(0).is_empty());
    }

    #[test]
    fn window_offset_puts_the_window_at_56_456() {
        // The frozen front end's invariant. If this ever changes, the VAD's
        // segment boundaries move by 56 samples (3.5 ms) relative to the
        // features the encoder sees.
        assert_eq!(WINDOW_OFFSET, 56);
        assert_eq!(WINDOW_OFFSET + WINDOW_SAMPLES, 456);
    }

    #[test]
    fn ac_power_ignores_a_dc_offset() {
        let with_offset = vec![1.0f32; 100];
        assert!(ac_power(&with_offset) < 1e-4, "DC is not speech energy");
        let alternating = (0..100)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect::<Vec<_>>();
        assert!(approx(ac_power(&alternating), 100.0, 1e-3));
        assert_eq!(ac_power(&[]), 0.0);
    }

    #[test]
    fn zcr_is_one_for_a_full_scale_alternating_signal_and_zero_for_a_constant() {
        let alt = (0..400)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect::<Vec<_>>();
        assert!(approx(zero_crossing_rate(&alt), 1.0, 1e-6));
        assert_eq!(zero_crossing_rate(&vec![0.7f32; 400]), 0.0);
        assert_eq!(zero_crossing_rate(&[1.0]), 0.0);
    }

    #[test]
    fn zcr_does_not_treat_exact_zeros_as_a_low_crossing_rate() {
        // Hard clipping: a 100 Hz sine with its zero-crossing neighbourhoods
        // flattened to exact 0, which is what a saturating input stage produces
        // and what a heavily quantised quiet passage looks like. The feature
        // *skips* exact zeros instead of counting them as samples that failed
        // to cross, so each half-cycle still contributes exactly one crossing
        // and the rate survives -- it is normalised by the samples that had a
        // sign, not by the buffer length.
        //
        // The old fixture for this was a zero/±1 square wave whose control was
        // the same wave with zeros replaced by +1e-9; that control *adds* a
        // crossing at every plateau edge, so the two versions differ by design
        // and the assertion could never hold. The control here is a
        // hand-computed naive rate instead.
        let n = WINDOW_SAMPLES;
        let sine: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 100.0 * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();
        let clipped: Vec<f32> = sine
            .iter()
            .map(|v| if v.abs() < 0.5 { 0.0 } else { *v })
            .collect();
        let zeros = clipped.iter().filter(|v| **v == 0.0).count();
        assert!(
            zeros > n / 4,
            "the fixture is supposed to be mostly zeros, got {zeros}/{n}"
        );

        let z = zero_crossing_rate(&clipped);
        // 100 Hz at 16 kHz is 200 crossings per second, i.e. 5 in 25 ms. The
        // feature normalises by the samples that *had* a sign, so a clipped
        // signal reads slightly *high* against that per-buffer reference -- the
        // opposite of the failure the skip is there to prevent.
        let truth = 2.0 * 100.0 / SAMPLE_RATE as f32;
        assert!(
            z > truth,
            "exact zeros dragged the rate below the true one: {z} vs {truth}"
        );
        assert!(
            z < 1.5 * truth,
            "clipped sine read {z}, the true rate is {truth}"
        );
        // And the naive rate -- crossings over the whole buffer, counting every
        // plateau sample as a non-crossing -- really is lower. Without this the
        // assertion above is satisfied by any rate at or above the truth,
        // including one that never looked at the samples at all.
        let naive = 5.0 / n as f32;
        assert!(
            z > 1.1 * naive,
            "clipping cost the rate a tenth or more: {z} vs naive {naive}"
        );
        // An unclipped reference agrees, and a low tone reads lower than a high
        // one, so the feature is not simply returning a constant.
        assert!((zero_crossing_rate(&sine) - truth).abs() < 0.2 * truth);
        let hi: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();
        assert!(zero_crossing_rate(&hi) > 4.0 * zero_crossing_rate(&sine));
    }

    #[test]
    fn the_power_spectrum_is_not_the_broken_symmetric_version() {
        // Guards the specific bug `frontend.rs:290-296` documents. The wrong
        // implementation reads `re` from the mirrored bin and computes `2*re^2`;
        // for a real cosine aligned with an FFT bin the imaginary part is ~0, so
        // the result has twice the energy and still looks like a spectrum.
        // Parseval is the cheapest detector: it holds exactly, for the correct
        // form and not for the broken one.
        let mut s = Spectrum::new();
        let frame: Vec<f32> = (0..WINDOW_SAMPLES)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / SAMPLE_RATE as f32).sin() * 0.5
            })
            .collect();
        let p = s.power(&frame).to_vec();
        let total: f32 = p.iter().sum();
        // Parseval for the windowed, zero-padded real sequence:
        //   sum over ALL n_fft bins of |X[k]|^2 == n_fft * sum_n (w[n] x[n])^2
        // and `power` returns only the 257 non-negative bins, which carry half
        // that for a real signal -- so the divisor is `n_fft / 2`.
        //
        //   sum_n (w[n] x[n])^2 ~= (sum w^2) * mean(x^2)
        let w: f32 = symmetric_hann(WINDOW_SAMPLES).iter().map(|v| v * v).sum();
        let mean_square = frame.iter().map(|v| v * v).sum::<f32>() / WINDOW_SAMPLES as f32;
        let expected = (N_FFT / 2) as f32 * w * mean_square;
        let ratio = total / expected;
        assert!(
            (ratio - 1.0).abs() < 0.02,
            "Parseval ratio {ratio} != 1; the mirrored-bin implementation gives ~2"
        );
    }

    #[test]
    fn a_short_frame_is_zero_padded_rather_than_panicking() {
        // The tail of a buffer produces frames that hang off the end.
        let mut s = Spectrum::new();
        let p = s.power(&[0.5f32; 100]);
        assert_eq!(p.len(), N_FREQ);
        assert!(p.iter().all(|v| v.is_finite()));
        // An empty frame is a legitimate degenerate input: silence.
        let p = s.power(&[]);
        assert!(p.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn silence_produces_an_all_zero_spectrum() {
        let mut s = Spectrum::new();
        let p = s.power(&vec![0.0f32; WINDOW_SAMPLES]);
        assert!(p.iter().all(|v| *v == 0.0));
    }
}

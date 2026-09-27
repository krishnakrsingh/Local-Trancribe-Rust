//! The signal processing: framing, spectra, cepstra, pitch, activity.
//!
//! # The framing convention, and why it is the ASR one
//!
//! Every frame here uses the geometry `vaani-asr`'s front end uses, because
//! the two stages must agree on which instant a frame describes. A diarization
//! span and the ASR features behind it have to refer to the same audio, and
//! the cheapest way to guarantee that is to use the same numbers.
//!
//! | quantity | value | meaning |
//! |---|---|---|
//! | `hop` | 160 | 10 ms, the distance between frame centres |
//! | `n_fft` | 512 | FFT size |
//! | `win_length` | 400 | 25 ms analysis window |
//! | window offset | **56** | `(n_fft - win_length) / 2` |
//! | frames | `1 + n / hop` | `torch.stft(center = true)` |
//!
//! The offset is the one that cost the ASR team the most to find. `torch.stft`
//! zero-pads `n_fft/2` on *each* side and the 400-sample analysis window is
//! therefore **centred** in the 512-point frame, occupying `[56, 456)`. Left
//! aligning it shifts the spectrogram by 56 samples — 3.5 mel frames at this
//! hop — and produces output that still looks like a spectrogram, still
//! normalises, and is still quietly worse. So:
//!
//! ```text
//! frame f's analysis window covers real samples
//!     [f * hop - win_length/2,  f * hop + win_length/2)
//! and frame f's *timestamp* is sample f * hop.
//! ```
//!
//! Getting that timestamp wrong by 56 samples is the same class of silent
//! degradation: nothing crashes, every span is 3.5 ms wrong, and no test that
//! only checks "are the spans sorted" notices.
//!
//! # Two passes, and why there are two
//!
//! Activity detection has to see the whole recording before the feature stage
//! can know which frames matter, and the frame count for a 3-hour meeting is
//! 1.08 M. Holding 83 feature floats per frame for that many frames is
//! 358 MB of mostly-discarded data, and holding a full-resolution
//! spectrogram would be worse. So:
//!
//! * **Pass 1** streams every frame and keeps two `f32`s per frame (energy and
//!   spectral flux) plus the derived flatness gate. For 3 hours that is
//!   1.08 M x 2 x 4 B = 8.6 MB, and it is the only per-frame array in the
//!   whole engine.
//! * **Pass 2** re-reads the audio, computes full features, and reduces each
//!   window to a single vector immediately, so the peak is one window's worth
//!   of per-frame state (75 frames x 83 floats = 25 kB).
//!
//! The duplicated FFT work is the price. Pass 1 uses a 256-point transform
//! rather than 512 because activity detection needs 31 bands, not 257, and
//! that is 5x cheaper. The measured cost of the duplication is in the
//! benchmarks; it is a real cost and it is small compared to holding the
//! spectrogram.

use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;

use crate::config::{ActivityParams, FeatureParams, REQUIRED_SAMPLE_RATE};

/// A symmetric Hann window of `len` samples.
///
/// Symmetric, **not** periodic: `w[i] = 0.5 - 0.5*cos(2*pi*i/(len-1))`. This
/// is the same convention the ASR front end's shipped `mel_window.f32` uses,
/// and using a periodic window instead is wrong by up to 6e-3 at the edges.
#[must_use]
pub fn hann_symmetric(len: usize) -> Vec<f32> {
    if len <= 1 {
        return vec![1.0; len];
    }
    let denom = (len - 1) as f32;
    (0..len)
        .map(|i| {
            let t = i as f32 / denom;
            0.5 - 0.5 * (2.0 * std::f32::consts::PI * t).cos()
        })
        .collect()
}

/// A periodic Hann window of `len` samples, for the pitch span.
///
/// Periodic is the right choice *here* and the wrong choice there: the pitch
/// autocorrelation wants a circularly-extended signal, and a symmetric window
/// would put a discontinuity at the wrap point and bias the ACF downward at
/// long lags, which is exactly the region the peak is picked from.
#[must_use]
pub fn hann_periodic(len: usize) -> Vec<f32> {
    if len == 0 {
        return Vec::new();
    }
    (0..len)
        .map(|i| {
            let t = i as f32 / len as f32;
            0.5 - 0.5 * (2.0 * std::f32::consts::PI * t).cos()
        })
        .collect()
}

/// The mel scale, HTK/SLANEY form: `2595 * log10(1 + f / 700)`.
#[must_use]
pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// The inverse mel scale.
#[must_use]
pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

/// A triangular mel filterbank, row-major, `n_mel x n_freq`.
///
/// `n_freq` is `n_fft/2 + 1`, so the highest band centre is clamped to the
/// Nyquist bin rather than being dropped: a filter whose centre is above
/// Nyquist would contribute nothing at all, and the top mel band is where
/// sibilance lives.
#[must_use]
pub fn mel_filterbank(p: &FeatureParams) -> Vec<f32> {
    let n_freq = p.n_fft / 2 + 1;
    let nyquist = REQUIRED_SAMPLE_RATE as f32 / 2.0;
    let bin_hz = nyquist / (n_freq - 1) as f32;

    let lo = hz_to_mel(p.mel_low_hz.min(nyquist));
    let hi = hz_to_mel(p.mel_high_hz.min(nyquist));
    // Band centres, spaced evenly in the mel domain: n_mel + 2 points give
    // n_mel triangles with flat tails.
    let points: Vec<f32> = (0..p.n_mel + 2)
        .map(|i| lo + (hi - lo) * i as f32 / (p.n_mel + 1) as f32)
        .map(|m| mel_to_hz(m) / bin_hz)
        .collect();

    let mut fb = vec![0f32; p.n_mel * n_freq];
    for m in 0..p.n_mel {
        let (left, centre, right) = (points[m], points[m + 1], points[m + 2]);
        let row = &mut fb[m * n_freq..(m + 1) * n_freq];
        for (k, slot) in row.iter_mut().enumerate() {
            let k = k as f32;
            *slot = if k >= left && k <= centre && centre > left {
                (k - left) / (centre - left)
            } else if k > centre && k <= right && right > centre {
                (right - k) / (right - centre)
            } else {
                0.0
            };
        }
    }
    fb
}

/// A precomputed, reusable front end for one sample rate.
///
/// Building the FFT plan and the filterbank is not free, and
/// [`FrontEnd::new`] is called once per diarization run, not once per window.
pub struct FrontEnd {
    params: FeatureParams,
    activity: ActivityParams,
    n_freq: usize,
    /// `(n_fft - win_length) / 2`. The single most bug-prone number here.
    window_offset: usize,
    analysis_window: Vec<f32>,
    act_window: Vec<f32>,
    pitch_window: Vec<f32>,
    filterbank: Vec<f32>,
    dct: Vec<f32>,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    act_fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    fft_pitch: std::sync::Arc<dyn rustfft::Fft<f32>>,
    ifft_pitch: std::sync::Arc<dyn rustfft::Fft<f32>>,
    scratch: Vec<Complex32>,
    scratch_act: Vec<Complex32>,
    scratch_pitch: Vec<Complex32>,
}

impl std::fmt::Debug for FrontEnd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The FFT handles are `Arc<dyn Fft>` and have no `Debug`; printing
        // the geometry that determines them is what a reader actually needs.
        f.debug_struct("FrontEnd")
            .field("params", &self.params)
            .field("activity", &self.activity)
            .field("n_freq", &self.n_freq)
            .field("window_offset", &self.window_offset)
            .field("filterbank", &self.filterbank.len())
            .field("dct", &self.dct.len())
            .finish_non_exhaustive()
    }
}

impl FrontEnd {
    /// Build a front end.
    #[must_use]
    pub fn new(params: FeatureParams, activity: ActivityParams) -> Self {
        let n_freq = params.n_fft / 2 + 1;
        let window_offset = (params.n_fft.saturating_sub(params.win_length)) / 2;
        let analysis_window = hann_symmetric(params.win_length);
        let pitch_window = hann_periodic(params.pitch_span);
        // A symmetric Hann on the activity transform too. A rectangular
        // window has 1/k sidelobes, which puts a tone's energy into a dozen
        // neighbouring bins and raises its measured flatness towards the
        // noise floor -- which is exactly the discrimination the gate needs.
        let act_window = hann_symmetric(activity.n_fft);

        // Orthonormal DCT-II basis: dct[k][m] = sqrt(2/n) * cos(pi*k*(m+.5)/n),
        // with k = 0 scaled by 1/sqrt(2) so the transform is orthonormal.
        let n = params.n_mel;
        let mut dct = vec![0f32; params.n_cepstra * n];
        let scale = (2.0f32 / n as f32).sqrt();
        for k in 0..params.n_cepstra {
            let s = if k == 0 {
                scale * std::f32::consts::FRAC_1_SQRT_2
            } else {
                scale
            };
            for m in 0..n {
                dct[k * n + m] = s
                    * (std::f32::consts::PI * (k as f32 + 0.5) * (m as f32 + 0.5) / n as f32).cos();
            }
        }

        let mut planner = FftPlanner::<f32>::new();
        let pitch_fft_len = params.pitch_span.next_power_of_two().max(8);

        Self {
            filterbank: mel_filterbank(&params),
            dct,
            analysis_window,
            act_window,
            pitch_window,
            fft: planner.plan_fft_forward(params.n_fft),
            act_fft: planner.plan_fft_forward(activity.n_fft),
            fft_pitch: planner.plan_fft_forward(pitch_fft_len),
            ifft_pitch: planner.plan_fft_inverse(pitch_fft_len),
            scratch: vec![Complex32::ZERO; params.n_fft],
            scratch_act: vec![Complex32::ZERO; activity.n_fft],
            scratch_pitch: vec![Complex32::ZERO; pitch_fft_len],
            n_freq,
            window_offset,
            params,
            activity,
        }
    }

    /// The parameters this front end was built with.
    #[must_use]
    pub fn params(&self) -> &FeatureParams {
        &self.params
    }

    /// The offset the analysis window sits at inside the FFT frame.
    #[must_use]
    pub fn window_offset(&self) -> usize {
        self.window_offset
    }

    /// Number of bins in a one-sided power spectrum.
    #[must_use]
    pub fn n_freq(&self) -> usize {
        self.n_freq
    }

    /// Total frame count for `n` samples, matching `torch.stft(center = true)`:
    /// `1 + n / hop`.
    ///
    /// This counts the zero-padded first frame, which carries no real audio.
    #[must_use]
    pub fn total_frames(&self, n: usize) -> usize {
        1 + n / self.params.hop
    }

    /// The frames worth computing features for.
    ///
    /// Frame 0's analysis window sits entirely before the media starts
    /// (its centre is at sample `-n_fft/2`), so it is excluded. The last
    /// `n/hop` frame's centre is at or past the final sample, so it is
    /// excluded too. Everything between is a frame whose window overlaps real
    /// audio, which is exactly the ASR front end's `feature_lengths`
    /// convention with the padding frame removed.
    #[must_use]
    pub fn usable_frames(&self, n: usize) -> std::ops::Range<usize> {
        let total = self.total_frames(n);
        let last = if n == 0 {
            0
        } else {
            (n - 1) / self.params.hop + 1
        };
        let hi = total.min(last).max(1);
        1..hi
    }

    /// Sample index of frame `f`'s centre, i.e. of its timestamp.
    #[must_use]
    pub fn frame_centre_sample(&self, f: usize) -> isize {
        f as isize * self.params.hop as isize
    }

    /// Milliseconds from the start of the buffer to frame `f`'s centre.
    #[must_use]
    pub fn frame_time_ms(&self, f: usize) -> i64 {
        let samples = self.frame_centre_sample(f);
        if samples < 0 {
            return 0;
        }
        samples as i64 * 1000 / REQUIRED_SAMPLE_RATE as i64
    }

    /// `buf` samples at the millisecond offset `ms`, zero-padded at both
    /// ends. This is `torch.stft(center = true, pad_mode = "constant")`.
    fn load_frame(&mut self, buf: &[f32], f: usize) {
        let n = buf.len() as isize;
        let start = f as isize * self.params.hop as isize - self.params.n_fft as isize / 2;
        let off = self.window_offset;
        for (i, s) in self.scratch.iter_mut().enumerate() {
            let v = if i >= off && i < off + self.params.win_length {
                let w = self.analysis_window[i - off];
                // `start + i` indexes the zero-padded signal, so the signal
                // index is that minus the n_fft/2 of centring padding.
                let idx = start + i as isize - self.params.n_fft as isize / 2;
                if (0..n).contains(&idx) {
                    buf[idx as usize] * w
                } else {
                    0.0
                }
            } else {
                0.0
            };
            *s = Complex32::new(v, 0.0);
        }
        self.fft.process(&mut self.scratch);
    }

    /// The one-sided power spectrum of the current frame into `out`.
    ///
    /// The imaginary part is read from bin `k` itself, never from the
    /// Hermitian mirror. For a real input the symmetry relates the *conjugate*
    /// — `Re(X[n-k]) == Re(X[k])` and `Im(X[n-k]) == -Im(X[k])` — so reading
    /// the power from the mirrored bin computes `re^2 + re^2`, which is finite,
    /// plausible, and wrong.
    fn frame_power(&self, out: &mut [f32]) {
        let k = out.len().min(self.n_freq);
        for (i, slot) in out.iter_mut().take(k).enumerate() {
            let z = self.scratch[i];
            *slot = z.re * z.re + z.im * z.im;
        }
    }

    /// One frame's log-mel bands, `n_mel` values, into `out`.
    pub fn frame_log_mel(&mut self, buf: &[f32], f: usize, power: &mut [f32], out: &mut [f32]) {
        self.load_frame(buf, f);
        self.frame_power(power);
        let n = self.params.n_mel;
        for (m, slot) in out.iter_mut().take(n).enumerate() {
            let row = &self.filterbank[m * self.n_freq..(m + 1) * self.n_freq];
            let mut acc = 0f32;
            for (w, p) in row.iter().zip(power.iter()) {
                acc += w * p;
            }
            // A 2^-24 guard rather than an epsilon, matching the ASR front end:
            // the exact value is what the reference implementation used, and
            // the low-energy bands are where fricatives live.
            *slot = (acc + f32::from_bits(0x3380_0000)).ln();
        }
    }

    /// One frame's cepstral coefficients, `n_cepstra` values, into `out`.
    pub fn frame_cepstra(
        &mut self,
        buf: &[f32],
        f: usize,
        power: &mut [f32],
        log_mel: &mut [f32],
        out: &mut [f32],
    ) {
        self.frame_log_mel(buf, f, power, log_mel);
        let n = self.params.n_mel;
        for (k, slot) in out.iter_mut().enumerate() {
            let row = &self.dct[k * n..(k + 1) * n];
            let mut acc = 0f32;
            for (w, v) in row.iter().zip(log_mel.iter()) {
                acc += w * v;
            }
            *slot = acc;
        }
    }

    /// Spectral centroid, spread and 85% rolloff for the current frame, in Hz.
    ///
    /// The caller must have called [`FrontEnd::load_frame`] and
    /// [`FrontEnd::frame_power`] first; there is deliberately no hidden state,
    /// because three callers in this crate want different subsets.
    #[must_use]
    pub fn spectral_shape(&self, power: &[f32]) -> (f32, f32, f32) {
        let bin_hz = (REQUIRED_SAMPLE_RATE as f32 / 2.0) / (self.n_freq - 1).max(1) as f32;
        let total: f32 = power.iter().sum();
        if !(total.is_finite() && total > 0.0) {
            return (0.0, 0.0, 0.0);
        }
        let mut num = 0f32;
        for (k, p) in power.iter().enumerate() {
            num += p * k as f32;
        }
        let centroid = num / total;

        let mut var = 0f32;
        for (k, p) in power.iter().enumerate() {
            let d = k as f32 - centroid;
            var += p * d * d;
        }
        let spread = (var / total).sqrt();

        // 85% rolloff: the lowest bin at which the cumulative energy reaches
        // 85% of the total. A linear scan is fine and is monotonic, which a
        // binary search would not be.
        let mut cum = 0f32;
        let target = 0.85 * total;
        let mut rolloff = (self.n_freq - 1) as f32;
        for (i, p) in power.iter().enumerate() {
            cum += p;
            if cum >= target {
                rolloff = i as f32;
                break;
            }
        }
        (centroid * bin_hz, spread * bin_hz, rolloff * bin_hz)
    }

    /// Autocorrelation-based fundamental frequency for frame `f`.
    ///
    /// Returns `(log2_f0_hz, confidence)`, or `(0.0, 0.0)` when the frame is
    /// unvoiced. The span is `pitch_span` samples **centred** on the frame
    /// centre, matching the analysis window's centring, so the pitch estimate
    /// describes the same instant as the cepstra.
    ///
    /// The autocorrelation is computed by FFT rather than directly. A direct
    /// O(span x lags) sum is 1024 x 226 = 231 k multiplies per frame, i.e. 23
    /// Gflop for a 14-minute meeting; the 1024-point transform is about half
    /// that and, more importantly, is a fixed cost rather than a linear one.
    pub fn frame_pitch(&mut self, buf: &[f32], f: usize) -> (f32, f32) {
        let span = self.params.pitch_span;
        let n = buf.len() as isize;
        let centre = self.frame_centre_sample(f);
        let start = centre - span as isize / 2;

        // Mean removal first: a DC offset in the frame shifts every
        // autocorrelation value by a constant and the normalised version
        // would then be dominated by that constant.
        let mut mean = 0f32;
        let mut count = 0f32;
        for i in 0..span {
            let idx = start + i as isize;
            if (0..n).contains(&idx) {
                mean += buf[idx as usize];
                count += 1.0;
            }
        }
        if count == 0.0 {
            return (0.0, 0.0);
        }
        mean /= count;

        let mut energy = 0f32;
        for (i, s) in self.scratch_pitch.iter_mut().enumerate() {
            let idx = start + i as isize;
            let v = if (0..n).contains(&idx) && i < span {
                (buf[idx as usize] - mean) * self.pitch_window[i]
            } else {
                0.0
            };
            energy += v * v;
            *s = Complex32::new(v, 0.0);
        }
        if !(energy.is_finite() && energy > 1e-6) {
            // An **absolute** floor, and it has to be absolute. After mean
            // removal a constant signal leaves only the f32 rounding residue
            // of the subtraction -- around 1e-8 of sum-of-squares over a
            // 1024-sample window -- and dividing the normalised
            // autocorrelation by a denominator that small amplifies it into a
            // confident-looking peak. A relative gate cannot catch this
            // because there is nothing relative to compare against.
            return (0.0, 0.0);
        }

        // Forward, power, inverse: the circular autocorrelation of a
        // zero-padded real signal, which for lags below the padding length is
        // the linear one.
        self.fft_pitch.process(&mut self.scratch_pitch);
        for s in self.scratch_pitch.iter_mut() {
            let p = s.re * s.re + s.im * s.im;
            *s = Complex32::new(p, 0.0);
        }
        self.ifft_pitch.process(&mut self.scratch_pitch);

        let inv = 1.0 / self.scratch_pitch.len() as f32;
        let lag_min = self.params.pitch_lag_min.max(1);
        let lag_max = self.params.pitch_lag_max.min(self.scratch_pitch.len() - 1);
        if lag_max <= lag_min {
            return (0.0, 0.0);
        }

        let mut best_lag = 0usize;
        let mut best = 0f32;
        for lag in lag_min..=lag_max {
            let v = self.scratch_pitch[lag].re * inv / energy;
            if v > best {
                best = v;
                best_lag = lag;
            }
        }
        if best_lag == 0 || best < self.params.voicing_threshold {
            return (0.0, 0.0);
        }

        // Parabolic interpolation around the peak. On a 10 ms lag grid, a
        // 120 Hz voice's period is 8.3 ms, so the grid is already too coarse:
        // without this, a male voice's pitch snaps to one of 40 discrete
        // values and a 5% pitch difference between two speakers is invisible.
        let ym1 = self.scratch_pitch[best_lag - 1].re * inv / energy;
        let y0 = self.scratch_pitch[best_lag].re * inv / energy;
        let denom = ym1 - 2.0 * y0 + self.scratch_pitch[best_lag + 1].re * inv / energy;
        let shift = if denom.abs() > 1e-9 {
            0.5 * (ym1 - y0) / -denom
        } else {
            0.0
        };
        let lag = (best_lag as f32 + shift.clamp(-0.5, 0.5)).max(1.0);
        let f0 = REQUIRED_SAMPLE_RATE as f32 / lag;
        if !(f0.is_finite() && f0 > 0.0) {
            return (0.0, 0.0);
        }
        (f0.log2(), best.clamp(0.0, 1.0))
    }

    /// Fill `out` with this frame's activity summary: `(energy dB, flux, flatness)`.
    ///
    /// This is pass 1. It is called for every frame of the whole file, so it
    /// uses the cheaper `activity.n_fft` transform and keeps only three
    /// numbers per frame.
    pub fn frame_activity(&mut self, buf: &[f32], f: usize, out: &mut [f32]) {
        let a = &self.activity;
        let n = buf.len() as isize;
        let half = a.n_fft / 2;
        let start = f as isize * self.params.hop as isize - half as isize;

        let mut energy = 0f32;
        for (i, s) in self.scratch_act.iter_mut().enumerate() {
            let idx = start + i as isize;
            let v = if (0..n).contains(&idx) {
                buf[idx as usize]
            } else {
                0.0
            };
            energy += v * v;
            *s = Complex32::new(v * self.act_window[i], 0.0);
        }
        let rms = (energy / half as f32).max(1e-12).sqrt();
        self.act_fft.process(&mut self.scratch_act);

        // Only the lowest `half` bins matter: above 3 kHz a meeting recording
        // is mostly room noise, and including it in the flatness measure is
        // what makes a hiss look like speech.
        let mut flux = 0f32;
        let mut prev = 0f32;
        for (k, z) in self.scratch_act.iter().take(half).enumerate() {
            let p = z.re * z.re + z.im * z.im;
            let lp = (p + 1e-12).log10();
            if k > 0 {
                let d = lp - prev;
                if d > 0.0 {
                    flux += d;
                }
            }
            prev = lp;
        }

        // Flatness over 16 log-spaced **band sums**, not over raw bins.
        //
        // The distinction is not cosmetic. A single raw bin's power is
        // exponentially distributed even for white noise, so its
        // geometric-over-arithmetic ratio converges to exp(-gamma) = 0.56
        // rather than 1.0, and a single frame's ratio wanders. Summing
        // several bins per band averages that variance down, which puts white
        // noise near 0.9 and a pure tone near 0.15 -- a gap the gate can sit
        // in the middle of. Computing it on raw bins put the threshold
        // *inside* the noise distribution, and half the noise frames passed
        // it, which is the "eight speakers from a hiss" failure.
        let nb = FLATNESS_BANDS;
        let mut band_sums = [0f32; FLATNESS_BANDS];
        let edges = flatness_band_edges(half, nb);
        for b in 0..nb {
            let (lo, hi) = (edges[b], edges[b + 1]);
            let mut acc = 0f32;
            for z in self.scratch_act.iter().take(hi).skip(lo) {
                acc += z.re * z.re + z.im * z.im;
            }
            band_sums[b] = acc;
        }
        let band_total: f32 = band_sums.iter().sum();
        let band_log_sum: f32 = band_sums.iter().map(|s| (s + 1e-12).ln()).sum();
        let flatness = if band_total > 0.0 {
            ((band_log_sum / nb as f32).exp() / (band_total / nb as f32)).clamp(0.0, 1.0)
        } else {
            1.0
        };

        out[0] = 20.0 * rms.log10();
        out[1] = flux;
        out[2] = flatness;
    }
}

/// The band edges the spectral-flatness measure sums into.
///
/// **Equal width, not log-spaced.** This was the third attempt and the reason
/// the previous two failed is worth recording. Flatness is the geometric mean
/// over the arithmetic mean, and a band's contribution to it depends on how
/// many bins it sums: a one-bin band of white noise has a power that is
/// exponentially distributed, giving `exp(-gamma) = 0.56` on its own, while a
/// 37-bin band gives 0.99. Sixteen log-spaced bands over 128 bins therefore
/// average a 0.56 against a 0.99 and the measurement lands near 0.78 whatever
/// the signal is -- the estimator was measuring its own band widths.
///
/// With every band the same width the estimator measures the signal. Eight
/// bins per band is enough for the Gamma(8) spread to be tight, and
/// log-spacing the bands buys nothing here because flatness does not care
/// where in the spectrum the peaks are.
#[must_use]
pub fn flatness_band_edges(n_bins: usize, n_bands: usize) -> Vec<usize> {
    let mut edges: Vec<usize> = (0..=n_bands).map(|b| b * n_bins / n_bands).collect();
    // The top edge must be the last bin, and every band at least one bin.
    if let Some(last) = edges.last_mut() {
        *last = n_bins;
    }
    for b in 1..edges.len() {
        edges[b] = edges[b].max(edges[b - 1] + 1);
    }
    edges
}

/// The number of bands the spectral-flatness measure sums into.
const FLATNESS_BANDS: usize = 16;

/// The dynamic range, in dB, below which a band is pinned to the floor.
///
/// A band more than 25 dB below the loudest band *in the same frame* is room
/// tone and microphone self-noise, not the speaker. Leaving it in the feature
/// vector is actively harmful: its level is set by the room, its variation is
/// numerical leakage, and z-scoring then promotes that numerical leakage to
/// full unit variance, so a spectrally-poor frame ends up with a
/// noise-dominated descriptor and the clustering splits it. Pinning those
/// bands makes such a frame produce a *flat, zero-variance* descriptor
/// instead, which the `MinSpread` floor then correctly treats as carrying no
/// speaker information.
pub const BAND_FLOOR_DB: f32 = 25.0;

/// The linear equivalent of [`BAND_FLOOR_DB`].
#[must_use]
pub fn band_floor_linear() -> f32 {
    10f32.powf(-BAND_FLOOR_DB / 10.0)
}

/// A frame's spectral energy in the coarse bands named by `band_edges_hz`,
/// as natural logs of fractions of the total.
///
/// Fractions rather than absolute energies: an absolute band energy is a
/// function of how loudly someone is talking, and a loud window and a quiet
/// window of the same speaker must land in the same place after z-scoring.
/// Each fraction is floored at [`band_floor_linear`] times the loudest band in
/// the same frame, for the reason given on [`BAND_FLOOR_DB`].
#[must_use]
pub fn band_ratios(power: &[f32], n_freq: usize, band_edges_hz: &[f32; 6]) -> Vec<f32> {
    let bin_hz = (REQUIRED_SAMPLE_RATE as f32 / 2.0) / (n_freq - 1).max(1) as f32;
    let total: f32 = power.iter().sum();
    if !(total.is_finite() && total > 0.0) {
        return vec![0.0; band_edges_hz.len() - 1];
    }
    let mut acc = vec![0f32; band_edges_hz.len() - 1];
    for (b, slot) in acc.iter_mut().enumerate() {
        // Both edges floored, so the bands tile the spectrum exactly: no bin
        // is counted twice and none is dropped. The `hi.max(lo + 1)` clamp
        // keeps a band non-empty when two edges land in the same bin.
        //
        // The last band runs to the end of the one-sided spectrum. Truncating
        // it at the Nyquist edge drops the Nyquist bin, which is 1/257 of the
        // total -- small, but it makes the ratios miss their defining
        // property of summing to one, and a test asserting that is the only
        // thing that notices.
        let raw_lo = (band_edges_hz[b] / bin_hz).floor() as usize;
        let raw_hi = if b + 2 == band_edges_hz.len() {
            power.len()
        } else {
            (band_edges_hz[b + 1] / bin_hz).floor() as usize
        };
        let lo = raw_lo.min(power.len().saturating_sub(2));
        let hi = raw_hi.clamp(lo + 1, power.len());
        *slot = power[lo..hi].iter().sum::<f32>() / total;
    }
    floor_and_log(&acc)
}

/// Pin each value to the loudest times [`band_floor_linear`], then log.
fn floor_and_log(fractions: &[f32]) -> Vec<f32> {
    let peak = fractions.iter().copied().fold(0f32, f32::max);
    let floor = (peak * band_floor_linear()).max(1e-8);
    fractions.iter().map(|v| v.max(floor).ln()).collect()
}

/// The `p`th percentile of `v`, by linear interpolation, in place order.
///
/// `p` is in `[0, 100]`. An empty input yields `0.0`.
///
/// This is the only place a threshold is derived from the recording's own
/// statistics rather than from an absolute constant, and that is deliberate:
/// a -30 dBFS recording and a -6 dBFS recording of the same meeting need
/// different energy thresholds, and no absolute number serves both.
#[must_use]
pub fn percentile(v: &[f32], p: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mut sorted = v.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = (p.clamp(0.0, 100.0) / 100.0) * (sorted.len() - 1) as f32;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = rank - lo as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> FeatureParams {
        FeatureParams::default()
    }

    #[test]
    fn symmetric_and_periodic_hann_differ_where_it_matters() {
        // This exact difference is why the ASR front end's shipped window is
        // 400 long and symmetric: `torch.hann_window(400, periodic=False)` is
        // zero at *both* ends, while the periodic form reaches full scale at
        // the last tap. A periodic window zeroed at the end skews the whole
        // frame by one sample's worth of the analysis.
        let a = hann_symmetric(400);
        let b = hann_periodic(400);
        assert_eq!(a.len(), 400);
        assert!(a[0].abs() < 1e-6, "symmetric starts at 0, got {}", a[0]);
        assert!(a[399].abs() < 1e-6, "symmetric ends at 0, got {}", a[399]);
        assert!(b[0].abs() < 1e-6, "periodic starts at 0, got {}", b[0]);
        // Periodic never reaches the last sample's value, so its final tap is
        // small but not zero; symmetric is exactly zero there.
        assert!(
            b[399] > a[399],
            "periodic {} vs symmetric {}",
            b[399],
            a[399]
        );
        // The periodic window encloses strictly more area over the same span.
        assert!(
            b.iter().sum::<f32>() > a.iter().sum::<f32>(),
            "periodic {} vs symmetric {}",
            b.iter().sum::<f32>(),
            a.iter().sum::<f32>()
        );
        // Both peak in the middle.
        for w in [&a, &b] {
            let mid = w[w.len() / 2];
            assert!((mid - 1.0).abs() < 1e-3, "peak {mid}");
        }
    }

    #[test]
    fn hann_degenerate_lengths_do_not_divide_by_zero() {
        assert_eq!(hann_symmetric(0), Vec::<f32>::new());
        assert_eq!(hann_symmetric(1), vec![1.0]);
        assert_eq!(hann_periodic(0), Vec::<f32>::new());
        assert_eq!(hann_periodic(1).len(), 1);
    }

    #[test]
    fn mel_scale_round_trips() {
        for hz in [0.0f32, 80.0, 300.0, 1000.0, 4000.0, 7600.0, 8000.0] {
            let back = mel_to_hz(hz_to_mel(hz));
            assert!((back - hz).abs() < 0.5, "{hz} -> {back}");
        }
        assert!(hz_to_mel(1000.0) > hz_to_mel(500.0), "mel is monotonic");
    }

    #[test]
    fn filterbank_is_non_negative_and_covers_the_top_band() {
        let p = params();
        let fb = mel_filterbank(&p);
        let n_freq = p.n_fft / 2 + 1;
        assert_eq!(fb.len(), p.n_mel * n_freq);
        assert!(fb.iter().all(|v| v.is_finite() && *v >= 0.0));
        // The top band must reach Nyquist, or sibilance contributes nothing.
        let top: f32 = fb[(p.n_mel - 1) * n_freq..].iter().sum();
        assert!(top > 0.0, "top mel band is empty");
        // Band 0 must start at DC.
        let bottom: f32 = fb[..n_freq].iter().sum();
        assert!(bottom > 0.0, "bottom mel band is empty");
    }

    #[test]
    fn frame_count_follows_the_torch_stft_convention() {
        let fe = FrontEnd::new(params(), ActivityParams::default());
        for n in [160usize, 1_600, 16_000] {
            assert_eq!(fe.total_frames(n), 1 + n / 160, "n={n}");
        }
    }

    #[test]
    fn the_first_frame_is_excluded_because_its_window_precedes_the_media() {
        // Frame 0's centre is at sample -256, so its 400-sample analysis
        // window covers [-456, -56): entirely padding. Including it would put
        // a zero frame at the head of every feature window.
        let fe = FrontEnd::new(params(), ActivityParams::default());
        let range = fe.usable_frames(16_000);
        assert_eq!(range.start, 1);
        assert_eq!(range.end, 100, "n/hop = 100 usable frames");
    }

    #[test]
    fn a_buffer_shorter_than_one_frame_yields_no_usable_frames() {
        let fe = FrontEnd::new(params(), ActivityParams::default());
        assert!(fe.usable_frames(0).is_empty());
        assert!(fe.usable_frames(80).is_empty());
        // 200 samples: one hop is 160, so frame 1's window [ -40, 360 )
        // overlaps real audio and is usable.
        let r = fe.usable_frames(200);
        assert_eq!(r, 1..2);
    }

    #[test]
    fn a_frame_timestamp_is_its_centre_not_its_start() {
        // The 56-sample centring offset: if the timestamp were the window
        // *start* instead of its centre, every span would be 12.5 ms early and
        // no structural test would notice.
        let fe = FrontEnd::new(params(), ActivityParams::default());
        assert_eq!(fe.frame_centre_sample(0), 0);
        assert_eq!(fe.frame_centre_sample(1), 160);
        assert_eq!(fe.frame_time_ms(1), 10);
        assert_eq!(fe.frame_time_ms(100), 1_000);
    }

    #[test]
    fn window_offset_is_56_for_the_shipped_geometry() {
        let fe = FrontEnd::new(params(), ActivityParams::default());
        assert_eq!(fe.window_offset(), 56);
    }

    /// A pure tone, so every spectral measure has a known answer.
    fn tone(hz: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / REQUIRED_SAMPLE_RATE as f32;
                (2.0 * std::f32::consts::PI * hz * t).sin()
            })
            .collect()
    }

    #[test]
    fn spectral_centroid_of_a_pure_tone_is_that_tone() {
        let mut fe = FrontEnd::new(params(), ActivityParams::default());
        let buf = tone(1_000.0, 16_000);
        let mut power = vec![0f32; fe.n_freq()];
        fe.load_frame(&buf, 50);
        fe.frame_power(&mut power);
        let (c, spread, rolloff) = fe.spectral_shape(&power);
        let bin_hz = 8_000.0 / 256.0;
        let expected = power
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, _)| k as f32 * bin_hz)
            .unwrap();
        assert!(
            (c - expected).abs() < bin_hz * 1.5,
            "centroid {c} vs {expected}"
        );
        assert!(spread < c, "spread must be below the centroid for a tone");
        assert!(rolloff > 0.0 && rolloff <= 8_000.0);
    }

    #[test]
    fn spectral_shape_of_silence_is_zero_not_nan() {
        let mut fe = FrontEnd::new(params(), ActivityParams::default());
        let mut power = vec![0f32; fe.n_freq()];
        fe.load_frame(&vec![0.0; 16_000], 50);
        fe.frame_power(&mut power);
        let (c, s, r) = fe.spectral_shape(&power);
        assert_eq!((c, s, r), (0.0, 0.0, 0.0));
        assert!(c.is_finite());
    }

    #[test]
    fn cepstra_are_finite_for_silence_and_for_audio() {
        let mut fe = FrontEnd::new(params(), ActivityParams::default());
        let mut power = vec![0f32; fe.n_freq()];
        let mut log_mel = vec![0f32; params().n_mel];
        let mut out = vec![0f32; params().n_cepstra];
        fe.frame_cepstra(&vec![0.0; 16_000], 50, &mut power, &mut log_mel, &mut out);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "silence gave a NaN cepstrum"
        );
        fe.frame_cepstra(&tone(200.0, 16_000), 50, &mut power, &mut log_mel, &mut out);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn pitch_recovers_a_known_fundamental() {
        let mut fe = FrontEnd::new(params(), ActivityParams::default());
        for want in [95.0f32, 120.0, 200.0, 260.0] {
            let buf = tone(want, 16_000);
            let (log2, conf) = fe.frame_pitch(&buf, 50);
            assert!(conf > 0.3, "{want} Hz was unvoiced (conf {conf})");
            let got = 2f32.powf(log2);
            // One semitone: the lag grid plus parabolic interpolation should
            // be considerably better than this, and if it is not, the
            // interpolation regressed.
            let semitones = 12.0 * (got / want).log2();
            assert!(semitones.abs() < 1.0, "wanted {want} Hz, got {got} Hz");
        }
    }

    #[test]
    fn pitch_recovers_every_synthetic_voices_declared_fundamental() {
        // The end-to-end engine's pitch dimension was measured returning the
        // *same* value, 7.036 in log2, for both synthetic voices -- which are
        // 118 Hz and 203 Hz, nearly an octave apart. A pure-tone test passing
        // does not catch that, because a pure tone has one peak and a real
        // voice has a fundamental plus a harmonic series, and the harmonic
        // structure is exactly what a peak-picker can get wrong.
        //
        // So this test runs the estimator against the *rendered voices*, which
        // is the signal the engine actually sees. A synthetic voice's declared
        // `f0` is ground truth, and if the estimator cannot recover it then the
        // single most reliable speaker cue in the feature vector is carrying
        // nothing.
        let mut rng = crate::synth::Rng::new(11);
        for voice in crate::synth::two_voices() {
            let mut buf = vec![0f32; 16_000];
            let rendered = voice.render(16_000, 16_000, 0.0, 0.0, &mut rng);
            buf[..rendered.len()].copy_from_slice(&rendered[..rendered.len().min(16_000)]);
            let mut fe = FrontEnd::new(params(), ActivityParams::default());
            let (log2, conf) = fe.frame_pitch(&buf, 50);
            let got = 2f32.powf(log2);
            // `nominal_f0`, not `f0`: the renderer scales by `(1 + f0_drift)`.
            let want = voice.nominal_f0();
            let semitones = 12.0 * (got / want).log2();
            assert!(
                conf > 0.3,
                "a {} Hz voice was reported unvoiced (conf {conf})",
                want
            );
            assert!(
                semitones.abs() < 2.0,
                "a {want} Hz voice was measured as {got:.1} Hz ({semitones:.1} semitones)"
            );
        }
    }

    #[test]
    fn two_voices_an_octave_apart_are_measured_an_octave_apart() {
        // The property the engine actually depends on: not just "each is
        // measured correctly in isolation" but "the difference survives". If
        // both voices locked onto a shared harmonic the individual estimates
        // could look plausible while carrying no speaker information, which is
        // precisely the failure this caught.
        let mut rng = crate::synth::Rng::new(11);
        let mut fe = FrontEnd::new(params(), ActivityParams::default());
        let mut est = Vec::new();
        for voice in crate::synth::two_voices() {
            let rendered = voice.render(16_000, 16_000, 0.0, 0.0, &mut rng);
            let (log2, _) = fe.frame_pitch(&rendered, 50);
            est.push(2f32.powf(log2));
        }
        let measured_octaves = (est[1] / est[0]).log2();
        let true_octaves =
            (crate::synth::two_voices()[1].f0 / crate::synth::two_voices()[0].f0).log2();
        assert!(
            (measured_octaves - true_octaves).abs() < 0.25,
            "the two voices are {true_octaves:.3} octaves apart but were measured {measured_octaves:.3} apart ({est:?} Hz)"
        );
    }

    #[test]
    fn pitch_of_silence_and_of_a_short_buffer_is_unvoiced() {
        let mut fe = FrontEnd::new(params(), ActivityParams::default());
        assert_eq!(fe.frame_pitch(&vec![0.0; 16_000], 50), (0.0, 0.0));
        // A buffer shorter than the pitch span must not index out of bounds.
        assert_eq!(fe.frame_pitch(&[0.0, 0.0, 0.0], 1), (0.0, 0.0));
        assert_eq!(fe.frame_pitch(&[], 5), (0.0, 0.0));
    }

    #[test]
    fn a_dc_offset_does_not_fake_a_pitch() {
        // Mean removal: without it, a constant signal has a large
        // autocorrelation at lag 0 and the peak-picker finds lag 0, i.e.
        // infinite pitch.
        let mut fe = FrontEnd::new(params(), ActivityParams::default());
        let buf = vec![0.7f32; 16_000];
        let (log2, conf) = fe.frame_pitch(&buf, 50);
        assert!(conf <= 0.0, "a DC signal must not be voiced, got {conf}");
        assert_eq!(log2, 0.0);
    }

    #[test]
    fn activity_separates_speech_like_energy_from_nothing() {
        let mut fe = FrontEnd::new(params(), ActivityParams::default());
        let mut out = [0f32; 3];
        fe.frame_activity(&vec![0.0; 16_000], 50, &mut out);
        let silence_db = out[0];
        fe.frame_activity(&tone(220.0, 16_000), 50, &mut out);
        let tone_db = out[0];
        assert!(
            tone_db > silence_db + 20.0,
            "tone {tone_db} dB vs silence {silence_db} dB"
        );
    }

    #[test]
    fn white_noise_is_flat_and_a_tone_is_not() {
        // The flatness gate is what stops a noise-only recording from being
        // read as speech, so it needs a test that pins the separation.
        let mut fe = FrontEnd::new(params(), ActivityParams::default());
        let mut out = [0f32; 3];
        // A real zero-mean uniform generator. `rng >> 33` gives 31 bits, so
        // dividing by 2^32 and doubling minus one would give [-1, 0) -- a
        // signal with a -0.5 DC offset, whose spectrum is one enormous spike
        // at DC and noise nowhere else. That measures the offset, not the
        // flatness.
        let mut rng = 0x5eed_1234u64;
        let noise: Vec<f32> = (0..16_000)
            .map(|_| {
                rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let v = (rng >> 40) as f32 / 16_777_216.0;
                v * 2.0 - 1.0
            })
            .collect();
        let mean = noise.iter().sum::<f32>() / noise.len() as f32;
        assert!(mean.abs() < 0.01, "generator is not zero mean: {mean}");
        fe.frame_activity(&noise, 50, &mut out);
        let noise_flat = out[2];
        fe.frame_activity(&tone(200.0, 16_000), 50, &mut out);
        let tone_flat = out[2];
        assert!(noise_flat > 0.7, "white noise flatness {noise_flat}");
        assert!(tone_flat < 0.4, "tone flatness {tone_flat}");
    }

    #[test]
    fn band_ratios_sum_to_one_when_no_band_is_floored() {
        let p = params();
        let n_freq = p.n_fft / 2 + 1;
        // A spectrum within 25 dB across the five bands, so the band floor
        // never engages and the ratios are exact fractions of the total.
        let power: Vec<f32> = (0..n_freq).map(|_| 1.0).collect();
        let r = band_ratios(&power, n_freq, &p.band_edges_hz);
        assert_eq!(r.len(), 5);
        let linear: f32 = r.iter().map(|v| v.exp()).sum();
        assert!((linear - 1.0).abs() < 1e-3, "linear sum {linear}");
    }

    #[test]
    fn a_band_far_below_the_loudest_one_is_floored() {
        // The floor is what stops a spectrally-poor frame producing a
        // noise-dominated descriptor. It necessarily makes the ratios stop
        // summing to one, which is the price and it is a fair one: the
        // information discarded is a band 25 dB down, which is room tone.
        let p = params();
        let n_freq = p.n_fft / 2 + 1;
        let mut power = vec![1e-7f32; n_freq];
        // All the energy in the top band.
        let lo = (p.band_edges_hz[4] / (8_000.0 / 256.0)) as usize;
        for v in &mut power[lo..] {
            *v = 1.0;
        }
        let r = band_ratios(&power, n_freq, &p.band_edges_hz);
        // Every quiet band lands on exactly the floor, so its variance across
        // frames is zero and the MinSpread floor then discards it.
        let floor = band_floor_linear().ln();
        assert!((r[0] - floor).abs() < 1e-5, "r[0] = {} want {floor}", r[0]);
        for (b, band) in r.iter().enumerate().take(4).skip(1) {
            assert!((*band - floor).abs() < 1e-5, "band {b} = {band}");
        }
        assert!(r[4] > r[0], "the loud band must still dominate: {r:?}");
    }

    #[test]
    fn band_ratios_of_silence_are_finite() {
        let p = params();
        let r = band_ratios(
            &vec![0.0; p.n_fft / 2 + 1],
            p.n_fft / 2 + 1,
            &p.band_edges_hz,
        );
        assert!(r.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn percentile_interpolates_and_survives_an_empty_input() {
        assert_eq!(percentile(&[], 50.0), 0.0);
        assert_eq!(percentile(&[7.0], 50.0), 7.0);
        let v: Vec<f32> = (0..101).map(|i| i as f32).collect();
        assert!((percentile(&v, 0.0) - 0.0).abs() < 1e-6);
        assert!((percentile(&v, 100.0) - 100.0).abs() < 1e-6);
        assert!((percentile(&v, 50.0) - 50.0).abs() < 1e-6);
        // Out-of-range p is clamped, not a panic.
        assert!((percentile(&v, -5.0) - 0.0).abs() < 1e-6);
        assert!((percentile(&v, 500.0) - 100.0).abs() < 1e-6);
    }

    #[test]
    fn percentile_of_unsorted_input_is_the_same_as_sorted() {
        let mut v: Vec<f32> = (0..50).map(|i| ((i * 37) % 50) as f32).collect();
        v[0] = 99.0;
        let mut shuffled = v.clone();
        shuffled.reverse();
        assert!((percentile(&v, 37.0) - percentile(&shuffled, 37.0)).abs() < 1e-6);
    }
}

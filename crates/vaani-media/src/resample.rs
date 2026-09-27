//! Rate conversion: any sample rate to 16 kHz mono, with [`rubato`].
//!
//! # The configuration, and why each number is what it is
//!
//! The resampler is `rubato::SincFixedIn<f32>` configured as follows. Every
//! value is a deliberate choice; the reasoning is here because a resampler
//! configured by accident is how a transcript ends up subtly wrong.
//!
//! | Parameter | Value | Why |
//! |---|---|---|
//! | `sinc_len` | **256** | Taps in the windowed-sinc prototype. The transition band is about `4 / sinc_len` of the input sample rate, so 256 taps gives roughly 690 Hz at 44.1 kHz. A longer filter narrows the transition but costs CPU linearly; 256 measured THD below -114 dB, which is far past anything a 16 kHz ASR front end can hear. |
//! | `window` | **`BlackmanHarris2`** | Squared Blackman-Harris: about -92 dB of stop-band leakage. The stop-band *depth* is what keeps aliasing out of the speech band, and that matters more here than the transition width. |
//! | `f_cutoff` | **adaptive, 0.5 to 0.95** | See below. |
//! | `oversampling_factor` | **`out / gcd(out, in)`, capped at 1024** | See below. This is the one that makes the resampler *synchronous*. |
//! | `interpolation` | **`Nearest` (exact polyphase), `Cubic` as fallback** | See below. |
//! | `chunk_size` | **1024** input frames | 64 ms at 16 kHz. Large enough that the per-call overhead is irrelevant, small enough that the memory held for latency is 4 KB. |
//! | `max_resample_ratio_relative` | **1.0** | The ratio never changes. Allowing it to would let a caller accidentally turn a fixed conversion into a drifting one. |
//!
//! ## Why `Nearest` and a ratio-derived `oversampling_factor`
//!
//! Both rates are integers, so `out / in` is a rational number `p / q` in lowest
//! terms, and output sample *k* sits at input position `k * q / p`. The
//! fractional part therefore takes exactly `p` distinct values. `rubato` can
//! precompute one filter per fractional position and pick between them with no
//! error at all — a *synchronous polyphase* resampler, which is what every
//! textbook design for integer-rate conversion actually is.
//!
//! `p` is `out / gcd(out, in)`, so `oversampling_factor = p` and
//! `SincInterpolationType::Nearest` makes the filter table exactly the `p`
//! phases. For 44.1 kHz to 16 kHz that is 160 phases of 256 taps, 164 KB, built
//! once per file in about a millisecond.
//!
//! The alternative — a large fixed `oversampling_factor` with cubic
//! interpolation between phases — introduces an interpolation error on top of
//! the filter's own error. Measured on a 1 kHz tone, that error showed up as a
//! step-response ripple 23 dB worse than the exact polyphase path. When `p`
//! exceeds the cap (only possible for a sample rate coprime with 16000, which no
//! real recorder produces) the code falls back to cubic interpolation over 256
//! phases, which is still comfortably better than linear and is documented as
//! such.
//!
//! ## Why the cutoff is adaptive
//!
//! `rubato` takes `f_cutoff` as a fraction of the *lower* of the two Nyquist
//! frequencies, and scales it by the resample ratio when downsampling, so
//! passing a single constant gives a single constant *bandwidth* — which is
//! wrong, because a filter's transition band is a fixed number of hertz only
//! relative to a fixed sample rate.
//!
//! The code computes the cutoff so the whole transition band fits *below* the
//! output Nyquist:
//!
//! ```text
//! transition_hz  ~= 2 * input_rate / sinc_len
//! f_cutoff        = clamp((output_rate / 2 - transition_hz) / (min(in, out) / 2), 0.5, 0.95)
//! ```
//!
//! The consequences, all measured in `tests/resample_quality.rs`:
//!
//! * 8 kHz to 16 kHz is an *upsample*, so nothing can alias and the cutoff
//!   goes to its 0.95 ceiling: 3.8 kHz of a 4 kHz band is kept.
//! * 44.1 kHz and 48 kHz land at 0.95: 7.6 kHz is kept and the stop band is
//!   complete before 8 kHz, so nothing folds back into the speech band.
//! * 96 kHz lands at 0.906 and 192 kHz at 0.8125, because their transition bands
//!   are 750 Hz and 1500 Hz wide and the output Nyquist does not have room for
//!   them at 0.95. Spending 0.8 kHz of inaudible air to avoid aliasing is the
//!   right trade for speech.
//!
//! The 0.95 ceiling is `rubato`'s own recommendation, and the 0.5 floor is the
//! widest cutoff that is still a low-pass.
//!
//! ## What the output looks like
//!
//! Measured over 8 kHz, 11.025 kHz, 22.05 kHz, 32 kHz, 44.1 kHz, 48 kHz, 96 kHz
//! and 192 kHz inputs on this machine:
//!
//! * **Length is exact.** `out.len() == round(input.len() * out / in)` for every
//!   rate tested, to the sample.
//! * **A pure tone stays a pure tone.** Amplitude error below 1.4e-5 relative,
//!   and second-plus-third-harmonic distortion below -114 dB.
//! * **Group delay** is a constant 3.0 to 4.5 output samples (0.19 to 0.28 ms),
//!   a property of the filter grid, not an accumulating shift.
//!
//! That last point is the one that would silently ruin a transcript if it were
//! wrong. A resampler whose output grid drifts would compress or stretch a
//! 40-minute meeting by a fraction of a percent — 40 seconds of drift by the
//! end — and nothing would fail, the words would just be timestamped against
//! the wrong audio.
//!
//! ## Passthrough
//!
//! When the input rate already equals the target, no filter is constructed at
//! all and samples are moved verbatim. That is what makes a 16 kHz mono WAV
//! round-trip sample-exact, which the ASR golden tests depend on.

use rubato::{
    Resampler, ResamplerConstructionError, SincFixedIn, SincInterpolationParameters,
    SincInterpolationType, WindowFunction,
};
use vaani_core::Result;

use crate::TARGET_SAMPLE_RATE;
use crate::errors;

/// The window function the prototype filter is built with.
///
/// Exposed as a name rather than a `rubato::WindowFunction` so that the public
/// API of this crate does not require a caller to depend on `rubato` in order
/// to read its configuration.
pub const WINDOW_FUNCTION: &str = "BlackmanHarris2";

/// How the resampler is configured, independently of the rates it will run at.
///
/// Exposed so the settings that matter can be asserted on, logged by the
/// benchmark harness, and overridden in a test without constructing a filter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResamplerConfig {
    /// Taps in the windowed-sinc prototype filter. Must be a multiple of 8;
    /// `rubato` rounds up otherwise.
    pub sinc_len: usize,
    /// Input frames consumed per inner `process` call.
    pub chunk_size: usize,
    /// Upper bound on the number of polyphase branches. Above it the code falls
    /// back to interpolating between a fixed number of phases.
    pub max_oversampling_factor: usize,
    /// Largest cutoff, as a fraction of the lower of the two Nyquist
    /// frequencies. This is `rubato`'s recommended starting point.
    pub max_cutoff: f32,
    /// Smallest cutoff, as a fraction of the lower of the two Nyquist
    /// frequencies. Anything below 0.5 stops being a low-pass filter.
    pub min_cutoff: f32,
}

/// The configuration this crate uses. See the module documentation for why.
pub const DEFAULT_RESAMPLER: ResamplerConfig = ResamplerConfig {
    sinc_len: 256,
    chunk_size: 1024,
    max_oversampling_factor: 1024,
    max_cutoff: 0.95,
    min_cutoff: 0.5,
};

/// The concrete settings derived from a [`ResamplerConfig`] and a rate pair.
///
/// A value rather than a side effect, so the choice can be inspected and tested.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResamplerPlan {
    /// Input sample rate in Hz.
    pub input_rate: u32,
    /// Output sample rate in Hz.
    pub output_rate: u32,
    /// Output frames per input frame, as an `f64` so no precision is lost.
    pub resample_ratio: f64,
    /// Cutoff as a fraction of the lower of the two Nyquist frequencies.
    pub cutoff_fraction: f32,
    /// Polyphase branches. Equal to `output_rate / gcd` when
    /// [`ResamplerPlan::exact_polyphase`] is true.
    pub oversampling_factor: usize,
    /// Taps in the prototype filter.
    pub sinc_len: usize,
    /// Input frames consumed per inner call.
    pub chunk_size: usize,
    /// True when there is one filter per output phase and nothing is
    /// interpolated; false when the fallback is in use.
    pub exact_polyphase: bool,
    /// True when no filtering happens at all because the rates are equal.
    pub passthrough: bool,
}

impl ResamplerConfig {
    /// Work out the settings for one rate pair.
    ///
    /// Total, and deterministic: the same pair always produces the same plan,
    /// which is what lets a test assert on it.
    #[must_use]
    pub fn plan(&self, input_rate: u32, output_rate: u32) -> ResamplerPlan {
        let ratio = f64::from(output_rate) / f64::from(input_rate);
        let passthrough = input_rate == output_rate;
        let branches = exact_branch_count(input_rate, output_rate);
        let exact = branches <= self.max_oversampling_factor;
        ResamplerPlan {
            input_rate,
            output_rate,
            resample_ratio: ratio,
            cutoff_fraction: self.cutoff_fraction(input_rate, output_rate),
            oversampling_factor: if exact {
                branches
            } else {
                self.max_oversampling_factor.min(256)
            },
            sinc_len: self.sinc_len,
            chunk_size: self.chunk_size,
            exact_polyphase: exact,
            passthrough,
        }
    }

    /// The cutoff this configuration chooses for a rate pair.
    ///
    /// Places the whole transition band below the output Nyquist; see the
    /// module documentation.
    #[must_use]
    pub fn cutoff_fraction(&self, input_rate: u32, output_rate: u32) -> f32 {
        let nyquist = f64::from(input_rate.min(output_rate)) / 2.0;
        if nyquist <= 0.0 {
            return self.max_cutoff;
        }
        // Half-width of the windowed-sinc transition band, in hertz. Two taps
        // per hertz of transition per sample of sinc length is the usual rule of
        // thumb for a Blackman-family window; measured settling times in
        // tests/resample_quality.rs agree with it.
        let transition_hz = 2.0 * f64::from(input_rate) / self.sinc_len.max(1) as f64;
        let headroom_hz = f64::from(output_rate) / 2.0 - transition_hz;
        let fraction = (headroom_hz / nyquist) as f32;
        fraction.clamp(self.min_cutoff, self.max_cutoff)
    }
}

/// Greatest common divisor, on `u32` rates.
fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.max(1)
}

/// `output_rate / gcd(output_rate, input_rate)`, floored at 1.
///
/// The number of distinct phases an integer-ratio conversion visits. See the
/// module documentation for why this is the number that matters.
fn exact_branch_count(input_rate: u32, output_rate: u32) -> usize {
    let g = gcd(input_rate, output_rate);
    (output_rate / g).max(1) as usize
}

/// The number of output frames a length of input is expected to produce.
///
/// `round`, not `floor`: a 3-hour recording must not lose a millisecond per
/// resampling decision, and `floor` biases the output grid low by half a sample
/// per call.
#[must_use]
pub fn expected_output_frames(input_frames: u64, input_rate: u32, output_rate: u32) -> u64 {
    if input_rate == 0 {
        return input_frames;
    }
    let scaled = input_frames as f64 * f64::from(output_rate) / f64::from(input_rate);
    // `f64::round` is exact enough for any plausible length. The guard is for a
    // nonsense rate pair, where the product can be `inf`: Rust's `as` cast
    // saturates, so the finite check is what makes zero the answer rather than
    // `u64::MAX`.
    if scaled.is_finite() && scaled > 0.0 {
        scaled.round() as u64
    } else {
        0
    }
}

/// Rate-converts a mono `f32` signal, appending the result to `out`.
///
/// The one-shot form of [`MonoResampler`], for tests, for a whole-file
/// conversion of audio that is already in memory, and for any caller that has
/// no reason to stream. `out` is cleared first.
pub fn resample_mono(input: &[f32], input_rate: u32) -> Result<Vec<f32>> {
    let mut rs = MonoResampler::new(input_rate)?;
    let mut out = Vec::with_capacity(input.len() + 1);
    rs.push(input, &mut out)?;
    rs.finish(&mut out)?;
    rs.trim_to_expected(&mut out);
    Ok(out)
}

/// A streaming mono resampler, configured once for one rate pair.
///
/// # Memory
///
/// Holds at most `chunk_size` staged input frames, one output scratch buffer of
/// `chunk_size * ratio + 10` frames, and the filter table
/// (`sinc_len * oversampling_factor` floats — 164 KB for 44.1 kHz to 16 kHz).
/// Nothing scales with the length of the signal, which is what lets a
/// 40-minute file be converted inside a constant footprint.
///
/// # Output length
///
/// [`MonoResampler::finish`] guarantees that the total number of frames
/// appended across all calls is `round(input_frames * ratio)`. Getting a filter
/// to land on an exact sample count needs two things: the tail is flushed with
/// silence so the last `sinc_len / 2` input samples get their turn, and then the
/// result is trimmed or zero-padded to the target. In practice the flush
/// overshoots and only the trim is used; the pad path exists because for some
/// rate pairs the filter's output grid ends one sample short, and a one-sample
/// (62.5 microsecond) zero is preferable to a buffer whose length depends on the
/// filter's internal bookkeeping.
pub struct MonoResampler {
    inner: Inner,
    plan: ResamplerPlan,
    input_frames: u64,
    finished: bool,
}

enum Inner {
    /// Rates are equal; samples move verbatim.
    Passthrough,
    Sinc(Sinc),
}

struct Sinc {
    resampler: Box<SincFixedIn<f32>>,
    /// Input frames waiting to be handed to the filter.
    stage: Vec<f32>,
    /// Reused output buffer, sized once.
    scratch: Vec<f32>,
}

impl std::fmt::Debug for MonoResampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MonoResampler")
            .field("plan", &self.plan)
            .field("input_frames", &self.input_frames)
            .field("finished", &self.finished)
            .finish()
    }
}

impl MonoResampler {
    /// A resampler from `input_rate` to [`TARGET_SAMPLE_RATE`].
    ///
    /// # Errors
    ///
    /// `UnsupportedMedia` if the rate pair is one no filter can be built for.
    pub fn new(input_rate: u32) -> Result<Self> {
        Self::with_target(input_rate, TARGET_SAMPLE_RATE)
    }

    /// A resampler between two arbitrary rates.
    ///
    /// # Errors
    ///
    /// `UnsupportedMedia` if either rate is zero, or if the pair is one no
    /// filter can be built for.
    pub fn with_target(input_rate: u32, output_rate: u32) -> Result<Self> {
        if input_rate == 0 || output_rate == 0 {
            return Err(errors::internal(format!(
                "resampler asked for the impossible rate pair {input_rate} -> {output_rate}"
            )));
        }
        let plan = DEFAULT_RESAMPLER.plan(input_rate, output_rate);
        if plan.passthrough {
            return Ok(Self {
                inner: Inner::Passthrough,
                plan,
                input_frames: 0,
                finished: false,
            });
        }
        let interpolation = if plan.exact_polyphase {
            SincInterpolationType::Nearest
        } else {
            SincInterpolationType::Cubic
        };
        let parameters = SincInterpolationParameters {
            sinc_len: plan.sinc_len,
            f_cutoff: plan.cutoff_fraction,
            oversampling_factor: plan.oversampling_factor,
            interpolation,
            window: WindowFunction::BlackmanHarris2,
        };
        let resampler =
            SincFixedIn::<f32>::new(plan.resample_ratio, 1.0, parameters, plan.chunk_size, 1)
                .map_err(|e: ResamplerConstructionError| {
                    errors::unsupported_sample_rate(input_rate, output_rate)
                        .with_cause(format!("rubato SincFixedIn: {e}"))
                })?;
        let scratch = vec![0.0f32; resampler.output_frames_max()];
        let stage = Vec::with_capacity(plan.chunk_size);
        Ok(Self {
            inner: Inner::Sinc(Sinc {
                resampler: Box::new(resampler),
                stage,
                scratch,
            }),
            plan,
            input_frames: 0,
            finished: false,
        })
    }

    /// The settings this resampler is running with.
    #[must_use]
    pub fn plan(&self) -> ResamplerPlan {
        self.plan
    }

    /// Whether samples pass through untouched because the rates are equal.
    #[must_use]
    pub fn is_passthrough(&self) -> bool {
        self.plan.passthrough
    }

    /// Input frames handed to [`MonoResampler::push`] so far.
    #[must_use]
    pub fn input_frames(&self) -> u64 {
        self.input_frames
    }

    /// The exact number of output frames the input seen so far must produce.
    #[must_use]
    pub fn expected_output_frames(&self) -> u64 {
        expected_output_frames(
            self.input_frames,
            self.plan.input_rate,
            self.plan.output_rate,
        )
    }

    /// Append input frames. Output is appended to `out`, which is not cleared.
    ///
    /// Any whole chunks the input completes are converted immediately; a partial
    /// chunk is staged until [`MonoResampler::push`] or
    /// [`MonoResampler::finish`] completes it. So the output length after a
    /// `push` is not final, and only the total after `finish` is.
    ///
    /// # Errors
    ///
    /// `Internal` if the underlying filter rejects its own buffers, which
    /// would be a bug in this crate rather than anything about the input.
    pub fn push(&mut self, input: &[f32], out: &mut Vec<f32>) -> Result<()> {
        if self.finished {
            return Err(errors::internal(
                "MonoResampler::push called after MonoResampler::finish",
            ));
        }
        self.input_frames += input.len() as u64;
        match &mut self.inner {
            Inner::Passthrough => {
                out.reserve(input.len());
                out.extend_from_slice(input);
                Ok(())
            }
            Inner::Sinc(s) => {
                s.stage.extend_from_slice(input);
                let chunk = self.plan.chunk_size;
                let mut offset = 0;
                while s.stage.len() - offset >= chunk {
                    Self::run_chunk(s, offset, chunk, out)?;
                    offset += chunk;
                }
                if offset > 0 {
                    s.stage.drain(..offset);
                }
                Ok(())
            }
        }
    }

    /// Flush the tail and make the total output length exact.
    ///
    /// Appends to `out` rather than replacing it, and expects `out` to be empty
    /// on entry: a caller that has already drained a batch into the same vector
    /// would otherwise get that batch twice.
    ///
    /// Must be called once, after the last [`MonoResampler::push`]. Calling it
    /// twice is a no-op; calling [`MonoResampler::push`] afterwards is an error,
    /// because the output grid has already been closed.
    ///
    /// # Errors
    ///
    /// `Internal` if the underlying filter rejects its own buffers.
    pub fn finish(&mut self, out: &mut Vec<f32>) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        if let Inner::Sinc(s) = &mut self.inner {
            if self.input_frames == 0 {
                return Ok(());
            }
            let chunk = self.plan.chunk_size;
            s.stage.resize(chunk, 0.0);
            // Two passes. The first emits everything the retained filter history
            // still owes, which is most of a chunk's worth of output; the second
            // guarantees the count is met for the rate pairs where the first
            // lands one sample short. Anything beyond the target length is
            // dropped below.
            for _ in 0..2 {
                Self::run_chunk(s, 0, chunk, out)?;
            }
            s.stage.clear();
        }
        Ok(())
    }

    /// Force the accumulated output length to the exact expected value.
    ///
    /// Truncates when the filter produced more than the target, which is the
    /// normal case, and zero-pads when it produced less. Returns `true` if it
    /// changed the length. Call this after [`MonoResampler::finish`]; the
    /// one-shot [`resample_mono`] uses it, and so should any caller that cares
    /// about an exact length.
    pub fn trim_to_expected(&self, out: &mut Vec<f32>) -> bool {
        let want = usize::try_from(self.expected_output_frames()).unwrap_or(usize::MAX);
        if out.len() == want {
            return false;
        }
        if out.len() > want {
            out.truncate(want);
        } else {
            out.resize(want, 0.0);
        }
        true
    }

    /// Discard all state and start again from the beginning of a stream.
    pub fn reset(&mut self) {
        self.input_frames = 0;
        self.finished = false;
        match &mut self.inner {
            Inner::Passthrough => {}
            Inner::Sinc(s) => {
                s.stage.clear();
                s.resampler.reset();
            }
        }
    }

    /// Hand one full chunk to the filter, appending what it produces.
    fn run_chunk(s: &mut Sinc, offset: usize, chunk: usize, out: &mut Vec<f32>) -> Result<usize> {
        let Sinc {
            resampler,
            stage,
            scratch,
        } = s;
        let (_, produced) = resampler
            .process_into_buffer(
                &[&stage[offset..offset + chunk]],
                &mut [&mut *scratch],
                None,
            )
            .map_err(|e| {
                errors::internal(format!("rubato SincFixedIn::process_into_buffer: {e}"))
            })?;
        out.reserve(produced);
        out.extend_from_slice(&scratch[..produced]);
        Ok(produced)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gcd_handles_the_degenerate_cases() {
        assert_eq!(gcd(0, 48000), 48000);
        assert_eq!(gcd(48000, 0), 48000);
        assert_eq!(gcd(1, 1), 1);
        assert_eq!(gcd(44100, 16000), 100);
        assert_eq!(gcd(48000, 16000), 16000);
    }

    #[test]
    fn branch_count_is_the_reduced_numerator_of_the_ratio() {
        // This is the number the whole polyphase design rests on.
        assert_eq!(exact_branch_count(48000, 16000), 1);
        assert_eq!(exact_branch_count(44100, 16000), 160);
        assert_eq!(exact_branch_count(8000, 16000), 2);
        assert_eq!(exact_branch_count(22050, 16000), 320);
        assert_eq!(exact_branch_count(11025, 16000), 640);
        assert_eq!(exact_branch_count(96000, 16000), 1);
        assert_eq!(exact_branch_count(16000, 16000), 1);
    }

    #[test]
    fn every_real_recording_rate_gets_an_exact_polyphase_plan() {
        // If this ever fails, a real meeting has fallen back to interpolation
        // and the step-response ripple gets 23 dB worse for no reason.
        for rate in [
            8000u32, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000, 176400, 192000,
        ] {
            let plan = DEFAULT_RESAMPLER.plan(rate, 16000);
            assert!(
                plan.exact_polyphase,
                "{rate} Hz fell back to interpolation ({} branches)",
                plan.oversampling_factor
            );
        }
    }

    #[test]
    fn a_rate_coprime_with_the_target_falls_back_and_says_so() {
        // 16001 Hz cannot happen from a recorder, but a truncated or corrupt
        // header can produce something like it, and the code must not try to
        // build a 16000-branch table.
        let plan = DEFAULT_RESAMPLER.plan(16001, 16000);
        assert!(!plan.exact_polyphase);
        assert!(plan.oversampling_factor <= 256);
        // The fallback still has to be a working cutoff.
        assert!(plan.cutoff_fraction > 0.5);
    }

    #[test]
    fn equal_rates_are_a_passthrough_plan() {
        let plan = DEFAULT_RESAMPLER.plan(16000, 16000);
        assert!(plan.passthrough);
        assert_eq!(plan.resample_ratio, 1.0);
        assert_eq!(plan.chunk_size, DEFAULT_RESAMPLER.chunk_size);
    }

    #[test]
    fn cutoff_grows_as_the_input_rate_falls_because_the_transition_narrows() {
        let c8 = DEFAULT_RESAMPLER.cutoff_fraction(8000, 16000);
        let c44 = DEFAULT_RESAMPLER.cutoff_fraction(44100, 16000);
        let c48 = DEFAULT_RESAMPLER.cutoff_fraction(48000, 16000);
        let c96 = DEFAULT_RESAMPLER.cutoff_fraction(96000, 16000);
        let c192 = DEFAULT_RESAMPLER.cutoff_fraction(192000, 16000);
        assert!(
            (c8 - 0.95).abs() < 1e-6,
            "8k should hit the ceiling, got {c8}"
        );
        assert!((c44 - 0.95).abs() < 1e-3, "44.1k got {c44}");
        assert!((c48 - 0.95).abs() < 1e-3, "48k got {c48}");
        assert!(c96 < c48, "96k must pull the cutoff down, got {c96}");
        assert!(c192 < c96, "192k must pull it down further, got {c192}");
        // The point of the rule: the transition band still ends below 8 kHz.
        for rate in [8000u32, 11025, 22050, 32000, 44100, 48000, 96000, 192000] {
            let c = DEFAULT_RESAMPLER.cutoff_fraction(rate, 16000) as f64;
            let nyq = f64::from(rate.min(16000)) / 2.0;
            let transition = 2.0 * f64::from(rate) / DEFAULT_RESAMPLER.sinc_len as f64;
            let band_top = c * nyq + transition;
            if rate > 16000 {
                assert!(
                    band_top <= 8000.0 + 1e-6,
                    "{rate} Hz transition reaches {band_top} Hz, past the 8 kHz Nyquist"
                );
            }
        }
    }

    #[test]
    fn cutoff_is_clamped_into_range_even_for_absurd_rates() {
        for rate in [1u32, 2, 100, 100_000_000, u32::MAX / 2] {
            let c = DEFAULT_RESAMPLER.cutoff_fraction(rate, 16000);
            assert!(
                (DEFAULT_RESAMPLER.min_cutoff..=DEFAULT_RESAMPLER.max_cutoff).contains(&c),
                "{rate} Hz produced cutoff {c}"
            );
            assert!(c.is_finite());
        }
        assert_eq!(DEFAULT_RESAMPLER.cutoff_fraction(0, 16000), 0.95);
    }

    #[test]
    fn expected_length_rounds_rather_than_truncates() {
        // 100 frames at 48 kHz to 16 kHz is 33.33 output frames.
        assert_eq!(expected_output_frames(100, 48000, 16000), 33);
        assert_eq!(expected_output_frames(48000, 48000, 16000), 16000);
        assert_eq!(expected_output_frames(8000, 8000, 16000), 16000);
        assert_eq!(expected_output_frames(0, 48000, 16000), 0);
        // Rounding, not truncation: 3 hours must not lose a millisecond per call.
        assert_eq!(expected_output_frames(3, 48000, 16000), 1); // exactly 1.0
        assert_eq!(expected_output_frames(5, 48000, 16000), 2); // 1.67 -> 2
    }

    #[test]
    fn expected_length_survives_a_zero_input_rate_without_dividing_by_zero() {
        assert_eq!(expected_output_frames(100, 0, 16000), 100);
    }

    #[test]
    fn equal_rates_move_samples_verbatim() {
        // The property the 16 kHz mono round-trip test depends on.
        let input: Vec<f32> = (0..1000).map(|i| i as f32 / 7.0).collect();
        let out = resample_mono(&input, 16000).expect("passthrough");
        assert_eq!(out, input);
    }

    #[test]
    fn passthrough_preserves_even_denormal_and_negative_zero() {
        let input = vec![0.0, -0.0, f32::MIN_POSITIVE, -1.0, 1.0e-40];
        let out = resample_mono(&input, 16000).expect("passthrough");
        assert_eq!(out.len(), input.len());
        for (a, b) in input.iter().zip(&out) {
            assert_eq!(a.to_bits(), b.to_bits(), "bit pattern changed");
        }
    }

    #[test]
    fn a_zero_rate_is_an_error_not_a_division_by_zero() {
        let e = MonoResampler::new(0).expect_err("zero rate must be rejected");
        assert_eq!(e.code(), vaani_core::ErrorCode::Internal);
        let e = MonoResampler::with_target(16000, 0).expect_err("zero target must be rejected");
        assert_eq!(e.code(), vaani_core::ErrorCode::Internal);
    }

    #[test]
    fn output_length_is_exact_for_every_real_rate() {
        // The anti-drift test. If the output grid drifted, this is where it shows
        // up, and a transcript would be quietly time-warped.
        for rate in [
            8000u32, 11025, 16000, 22050, 32000, 44100, 48000, 96000, 192000,
        ] {
            for frames in [1usize, 7, 63, 1000, 4095, 4096, 4097, 100_000] {
                let n_out = ((frames as f64) * 16000.0 / f64::from(rate)).round() as usize;
                let input: Vec<f32> = (0..frames).map(|i| (i as f32 * 0.001).sin()).collect();
                let out = resample_mono(&input, rate)
                    .unwrap_or_else(|e| panic!("{rate} Hz / {frames} frames: {e}"));
                assert_eq!(
                    out.len(),
                    n_out,
                    "{rate} Hz, {frames} frames in, expected {n_out}, got {}",
                    out.len()
                );
            }
        }
    }

    #[test]
    fn pushing_in_arbitrary_pieces_gives_the_same_answer_as_one_piece() {
        // Streaming must not change the numbers, or the streaming API and the
        // buffered one would disagree on the length of a meeting.
        let rate = 44100u32;
        let input: Vec<f32> = (0..50_000).map(|i| (i as f32 * 0.0007).sin()).collect();
        let whole = resample_mono(&input, rate).expect("whole");

        for piece in [1usize, 137, 1024, 4096, 9999] {
            let mut rs = MonoResampler::new(rate).expect("resampler");
            let mut out = Vec::new();
            for chunk in input.chunks(piece) {
                rs.push(chunk, &mut out).expect("push");
            }
            rs.finish(&mut out).expect("finish");
            rs.trim_to_expected(&mut out);
            assert_eq!(out.len(), whole.len(), "piece size {piece}");
            for (i, (a, b)) in out.iter().zip(&whole).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "piece size {piece}, sample {i}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn pushing_nothing_and_finishing_produces_nothing() {
        let mut rs = MonoResampler::new(48000).expect("resampler");
        let mut out = Vec::new();
        rs.finish(&mut out).expect("finish");
        assert!(out.is_empty());
        rs.trim_to_expected(&mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn a_single_sample_input_is_converted_not_dropped() {
        for rate in [8000u32, 44100, 48000, 96000] {
            let out = resample_mono(&[0.5], rate).unwrap_or_else(|e| panic!("{rate} Hz: {e}"));
            let want = expected_output_frames(1, rate, 16000);
            assert_eq!(out.len() as u64, want, "{rate} Hz");
        }
    }

    #[test]
    fn an_input_shorter_than_one_chunk_still_produces_the_right_length() {
        // The flush has to stand in for the filter history the loop never built.
        for rate in [8000u32, 44100, 48000, 96000] {
            for frames in [1usize, 5, 100, 1023] {
                let input = vec![0.25f32; frames];
                let out = resample_mono(&input, rate).expect("resample");
                let want = expected_output_frames(frames as u64, rate, 16000) as usize;
                assert_eq!(out.len(), want, "{rate} Hz, {frames} frames");
            }
        }
    }

    #[test]
    fn reset_makes_a_resampler_reusable_for_a_second_stream() {
        let mut rs = MonoResampler::new(48000).expect("resampler");
        let mut first = Vec::new();
        rs.push(&vec![0.5f32; 5000], &mut first).expect("push");
        rs.finish(&mut first).expect("finish");
        rs.trim_to_expected(&mut first);
        rs.reset();

        let mut second = Vec::new();
        rs.push(&vec![0.5f32; 5000], &mut second).expect("push");
        rs.finish(&mut second).expect("finish");
        rs.trim_to_expected(&mut second);
        assert_eq!(first, second, "reset did not clear the filter state");
    }

    #[test]
    fn pushing_after_finish_is_rejected_rather_than_silently_wrong() {
        let mut rs = MonoResampler::new(48000).expect("resampler");
        let mut out = Vec::new();
        rs.push(&[0.1; 100], &mut out).expect("push");
        rs.finish(&mut out).expect("finish");
        assert!(rs.push(&[0.1; 100], &mut out).is_err());
        // And finishing twice is harmless, because an error path may retry.
        assert!(rs.finish(&mut out).is_ok());
    }

    #[test]
    fn trim_reports_whether_it_changed_anything() {
        let rs = MonoResampler::new(48000).expect("resampler");
        let mut out = vec![0.0f32; 100];
        assert!(
            rs.trim_to_expected(&mut out),
            "100 frames present, 0 pushed: must trim"
        );
        assert!(out.is_empty());
        assert!(
            !rs.trim_to_expected(&mut out),
            "already at the expected length"
        );
    }

    #[test]
    fn a_pure_tone_keeps_its_frequency_and_amplitude() {
        // The in-crate half of the quality verification. The frequency-response,
        // sweep and step tests live in tests/resample_quality.rs because they
        // need an FFT and a spectrum.
        for rate in [8000u32, 22050, 32000, 44100, 48000, 96000] {
            let f = 1000.0f32;
            let amp = 0.5f32;
            let n = usize::try_from(rate).unwrap() * 2;
            let input: Vec<f32> = (0..n)
                .map(|i| amp * (2.0 * std::f32::consts::PI * f * i as f32 / rate as f32).sin())
                .collect();
            let out = resample_mono(&input, rate).unwrap_or_else(|e| panic!("{rate} Hz: {e}"));
            let seg = &out[4000..out.len() - 4000];
            let (measured, _delay) = goertzel(seg, 16_000, f);
            assert!(
                (measured - amp).abs() < 1e-4,
                "{rate} Hz: amplitude {measured}, want {amp}"
            );
            // Second and third harmonic: a resampler that added distortion would
            // show up here long before it showed up in a transcript.
            let (h2, _) = goertzel(seg, 16_000, 2.0 * f);
            let (h3, _) = goertzel(seg, 16_000, 3.0 * f);
            let thd = (h2 * h2 + h3 * h3).sqrt() / measured;
            assert!(thd < 1e-4, "{rate} Hz: THD {thd} is too high");
        }
    }

    #[test]
    fn silence_stays_silent() {
        for rate in [8000u32, 44100, 48000, 96000] {
            let out = resample_mono(&vec![0.0f32; 4000], rate).expect("resample");
            assert!(
                out.iter().all(|&v| v == 0.0),
                "{rate} Hz leaked noise into silence"
            );
        }
    }

    #[test]
    fn a_dc_offset_survives_a_rate_change_unchanged() {
        // A DC offset is the one signal a resampler must pass with unity gain;
        // if it does not, every recording with a microphone bias hum is wrong.
        for rate in [8000u32, 22050, 44100, 48000, 96000] {
            let out = resample_mono(&vec![0.3f32; 8000], rate).expect("resample");
            let margin = out.len() / 4;
            let seg = &out[margin..out.len() - margin];
            let mean = seg.iter().sum::<f32>() / seg.len() as f32;
            assert!((mean - 0.3).abs() < 1e-4, "{rate} Hz: DC mean {mean}");
        }
    }

    /// Amplitude and phase of one frequency in a signal, by direct correlation.
    ///
    /// Phase comes back as a delay in samples, which is how the group-delay
    /// measurement in the module documentation was taken.
    fn goertzel(x: &[f32], rate: u32, f: f32) -> (f32, f32) {
        let w = 2.0 * std::f64::consts::PI * f64::from(f) / f64::from(rate);
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (i, &v) in x.iter().enumerate() {
            let ph = w * i as f64;
            re += f64::from(v) * ph.cos();
            im -= f64::from(v) * ph.sin();
        }
        let n = x.len() as f64;
        (
            (2.0 * (re * re + im * im).sqrt() / n) as f32,
            (im.atan2(re) / w) as f32,
        )
    }
}

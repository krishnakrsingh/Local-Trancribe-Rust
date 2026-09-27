//! Per-frame features, and the rules that turn them into a speech score.
//!
//! # Why not a threshold on RMS
//!
//! The naive detector is `rms > threshold`, and it fails in five distinct ways
//! on real meeting audio, each of which cost somebody a bug report:
//!
//! 1. **The room is not known in advance.** A hard-coded dBFS threshold is
//!    correct in exactly one room. It loses a quiet speaker in a live room and
//!    reports the whole file as speech in a quiet one. Hence the *adaptive*
//!    floor in [`crate::noise`]: every threshold below is relative to the room,
//!    not to full scale.
//! 2. **A percentile, not a mean, is the floor.** If the floor is a running
//!    mean of frame energy, one loud speaker raises the floor for everyone and
//!    the quiet speaker disappears. A low percentile over a trailing window is
//!    by construction dominated by the quietest tenth of the recent past, so a
//!    loud passage cannot move it fast.
//! 3. **Loudness does not imply speech.** A cough, a slammed door and a
//!    shouting fan are all "above the floor". Energy alone cannot separate
//!    them, so [`FrameFeatures`] carries four shape features that are near-zero
//!    for noise and high for speech, and the score is a *product* of an
//!    audibility term and a shape term rather than a weighted sum. A sum lets
//!    the audibility term alone carry a broadband noise burst over the
//!    threshold; a product does not.
//! 4. **Broadband noise is not loud noise, it is flat noise.** A fan, an HVAC
//!    duct and a hiss all have a nearly flat spectrum, so
//!    [`FrameFeatures::flatness`] sits near 1.0, [`FrameFeatures::tonality`]
//!    sits at 1.0 (peak / mean of a flat spectrum *is* 1), and the voice
//!    energy is spread thin across 257 bins instead of concentrated in a
//!    harmonic comb. All three move the wrong way at once, which is why three
//!    of the five shape terms are the spectral ones.
//! 5. **A steady sound is not a talking sound.** A 50 Hz hum, a sustained
//!    whistle and a test tone have an almost unchanging spectrum frame to
//!    frame, while speech -- even a held vowel -- pulses at the pitch period
//!    (4-8 Hz amplitude modulation) and moves its formants.
//!    [`FrameFeatures::flux`] measures exactly that, and is used as an explicit
//!    veto rather than folded into the average.
//!
//! # The score is a product, and why
//!
//! ```text
//! score = audibility * shape
//! ```
//!
//! Read it as "the frame must be both audible above this room *and* shaped like
//! speech". A product is the honest combination of two independent pieces of
//! evidence; a sum lets one strong term outvote three weak ones, which is
//! exactly the failure in point 3. Both factors are in [0, 1] and both are
//! documented, so the score is inspectable frame by frame in a debug log.

use serde::{Deserialize, Serialize};

use crate::dsp::{HOP_MS, HOP_SAMPLES, N_FFT, N_FREQ, SAMPLE_RATE, Spectrum, zero_crossing_rate};

/// `flux` reported for the very first frame of a buffer.
///
/// The first frame has no predecessor, so there is nothing to difference it
/// against and a raw implementation reports `flux == 0` -- which is
/// indistinguishable from a perfectly steady signal, and would therefore fire
/// the steady-source veto on the opening frame of every file. Reporting a value
/// above any plausible [`FeatureWeights::flux_min_db`] is the honest encoding
/// of "unknown": the veto cannot fire, and the frame is decided on its other
/// four features alone, which is the correct behaviour for one frame out of
/// thousands.
pub const FLUX_UNDEFINED: f32 = 1.0e6;

/// Number of contiguous linear-frequency bands used for the flux measure.
///
/// The spectrum is bucketed into 24 bands rather than differenced bin by bin.
/// A single FFT bin of speech has tens of dB of frame-to-frame variance from
/// leakage alone, so per-bin flux measures the window, not the signal; summing
/// within a band first makes it measure the signal.
pub const N_FLUX_BANDS: usize = 24;

/// Lower edge of the "high" band, in Hz. Above this a frame is hissy.
pub const HF_EDGE_HZ: f32 = 3_500.0;

/// Lower edge of the "low" band, in Hz. Below this a frame is rumbling.
pub const LF_EDGE_HZ: f32 = 300.0;

/// Lower edge of the speech formant band, in Hz.
pub const MID_EDGE_LO_HZ: f32 = 500.0;

/// Upper edge of the speech formant band, in Hz.
pub const MID_EDGE_HI_HZ: f32 = 3_500.0;

/// The band edges used for the flux measure, in Hz.
///
/// Linear, because speech discriminative structure is linear-in-frequency
/// (formants, harmonics). A mel spacing would put three bands above 6 kHz,
/// which is exactly where a fricative is a single tone and a hiss is a wall.
const FLUX_BAND_HZ: [f32; N_FLUX_BANDS + 1] = [
    0.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1400.0, 1700.0, 2000.0, 2400.0, 2800.0,
    3200.0, 3600.0, 4000.0, 4500.0, 5000.0, 5500.0, 6000.0, 6500.0, 7000.0, 7400.0, 7700.0, 7900.0,
    8000.0,
];

/// Everything measurable about one 25 ms frame, before any decision is made.
///
/// Deliberately *all* of these are computed even though the default decision
/// rule does not use every one of them at full weight: they are the debug
/// surface, and a tuning run that can print them is a tuning run that converges.
/// `FrameFeatures` is `Serialize` so a frame dump can be diffed between two
/// parameter sets.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct FrameFeatures {
    /// Frame energy in dB, `10*log10(mean(x^2))` with the mean removed. Not
    /// dBFS in the mastering sense: full scale is 0 dB and there is no integer
    /// quantisation, so it is simply a level in dB relative to 1.0.
    pub db: f32,
    /// `db` minus the adaptive floor for this frame. The adaptive part.
    pub snr_db: f32,
    /// Geometric mean / arithmetic mean of the power spectrum over bins
    /// `1..n_freq`, in [0, 1]. 1.0 is white; ~0.01 is a strong harmonic comb.
    pub flatness: f32,
    /// Ratio of the strongest bin to the **median** bin, >= 1.
    ///
    /// The median, not the mean, and that is not a style choice. The mean is not
    /// a robust estimate of "a typical bin" when a minority of bins carry the
    /// signal: for a harmonic comb occupying 50 of 257 bins, the mean is 5x the
    /// median, and after noise subtraction -- which leaves a `floor * N`
    /// pedestal in every bin -- the mean is dominated by the other 206 while
    /// the median is exactly the pedestal. Measured on the fixtures, a clean
    /// voice reads `peak/mean` = 15.9 dB and the same voice under a 12 dB-SNR
    /// broadband floor reads 7.1 dB, which is *below the noise cluster*: one
    /// fixture with a noise floor and without it put the same voice on opposite
    /// sides of the tonality mapping. With the median, 206 of the 257 bins
    /// decide the denominator and the measurement is invariant to how many bins
    /// the signal occupies.
    ///
    /// For white noise, `peak/median` is the extreme-value ratio of 256
    /// exponential samples, about 6, i.e. 7.9 dB. For a harmonic comb it is
    /// three orders of magnitude. The clusters do not touch.
    pub tonality: f32,
    /// Peak of the normalised autocorrelation over the 60-400 Hz lag range,
    /// re-based so that noise reads ~0. See [`EnergyVadConfig`][crate::config::EnergyVadConfig].
    pub voicing: f32,
    /// Energy above [`HF_EDGE_HZ`] as a fraction of total, in [0, 1]. A
    /// broadband hiss measures about 0.64; voiced speech about 0.11.
    pub hf_ratio: f32,
    /// Energy below [`LF_EDGE_HZ`] as a fraction of total, in [0, 1]. A
    /// mains-hum-only frame measures 0.99; voiced speech about 0.23.
    pub lf_ratio: f32,
    /// Energy between [`MID_EDGE_LO_HZ`] and [`MID_EDGE_HI_HZ`] as a fraction
    /// of total, in [0, 1].
    ///
    /// **This is the feature that separates a voice from a rumble.** A voice
    /// puts most of its radiated energy in the formant band no matter how low
    /// its fundamental is -- a 60 Hz bass still has 58 harmonics below
    /// 3.5 kHz -- and it measures 0.4-0.8 here. A 50 Hz mains hum with its
    /// 2nd and 3rd harmonics measures 0.001. A broadband hiss measures 0.9.
    /// Nothing else in the feature set separates those two cases, and the
    /// reasoning is in [`out_of_band_penalty`].
    pub mid_ratio: f32,
    /// Frequency below which 85 % of the cumulative energy lies, in Hz.
    pub rolloff_hz: f32,
    /// Zero-crossing rate of the raw frame, in [0, 1].
    pub zcr: f32,
    /// Mean absolute change of per-band dB from the previous frame, in dB,
    /// **weighted by each band's share of the frame's energy**.
    ///
    /// Near 0 for a steady tone or a steady room tone; 2-8 dB for noise or
    /// speech. See [`FeatureExtractor::push`] for why the weighting is load
    /// bearing rather than cosmetic.
    pub flux: f32,
}

/// Maps raw features to the [0, 1] terms the decision rule consumes.
///
/// Every constant here is a tuning parameter with a stated meaning, and
/// `EnergyVadConfig` holds them all so a tuning run is a data change and not a
/// code change. `#[non_exhaustive]` is not used because the struct is ours and
/// adding a field must be a compile error at every construction site -- which
/// is the point.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct FeatureWeights {
    /// `snr_db` at which the audibility term saturates, in dB. Half this value
    /// gives 0.5.
    pub snr_full_db: f32,
    /// `tonality` (as `10*log10`) at which the structure term reaches 0.
    pub tonality_lo_db: f32,
    /// `tonality` (as `10*log10`) at which the structure term reaches 1.
    pub tonality_hi_db: f32,
    /// `flatness` at which the flatness term reaches 0 (very flat = noise).
    pub flatness_hi: f32,
    /// `flatness` at which the flatness term reaches 1 (very peaky = tonal).
    pub flatness_lo: f32,
    /// `hf_ratio` at which the spectral-shape term reaches 0 (hiss-like).
    pub hiss_lo: f32,
    /// `hf_ratio` at which the spectral-shape term reaches 1 (speech-like).
    pub hiss_hi: f32,
    /// `rolloff_hz` at which the rolloff half of the spectral term reaches 0.
    pub rolloff_lo_hz: f32,
    /// `rolloff_hz` at which the rolloff half reaches 1.
    pub rolloff_hi_hz: f32,
    /// `rolloff_hz` at which the *low* side of the two-sided rolloff term
    /// reaches 0, i.e. below this a frame is sub-bass rumble.
    pub rolloff_floor_lo_hz: f32,
    /// `rolloff_hz` at which the low side reaches 1.
    pub rolloff_floor_hi_hz: f32,
    /// Share of the spectral-shape term taken from `rolloff` rather than from
    /// `hf_ratio`. The two measure the same thing in a mirrored way; averaging
    /// them makes the term survive a band edge at 3 kHz moving to 3.5 kHz.
    pub rolloff_share: f32,
    /// Normalised-autocorrelation peak that noise is expected to reach by
    /// chance alone, from the extreme-value order statistics of ~230 lags over
    /// 400 samples. Below this the frame is treated as unvoiced.
    pub voicing_null: f32,
    /// Span above `voicing_null` that maps to a fully voiced frame.
    pub voicing_span: f32,
    /// ZCR at which the zero-crossing term peaks, in [0, 1].
    pub zcr_peak: f32,
    /// Distance from `zcr_peak` at which the zero-crossing term reaches 0.
    pub zcr_span: f32,
    /// `flux` below which a frame is considered *steady*, in dB.
    ///
    /// 0.3 dB, placed in the measured gap: a dead steady tone and a 50 Hz hum
    /// both measure 0.00-0.11, and the *5th percentile* of a synthetic voice
    /// measures 0.87. So the threshold has 2.7x of headroom below and 2.9x
    /// above. It was 0.5, which still vetoed 5 % of loud speech frames; the
    /// score drop was small but it is pure loss, and moving the threshold into
    /// the middle of the gap costs nothing.
    pub flux_min_db: f32,
    /// `snr_db` above which the steady-source veto applies, in dB.
    pub flux_veto_snr_db: f32,
    /// Multiplier applied to the score of a steady, clearly-audible frame.
    pub flux_penalty: f32,
    /// `mid_ratio` below which a frame is considered *out of band*, i.e. a
    /// rumble rather than a voice.
    ///
    /// Measured: a 50/100/150 Hz mains hum is 0.001, a fan's low-frequency
    /// tilt is 0.10, voiced speech is 0.4-0.6, a quiet voice 0.7, broadband
    /// noise 0.9, a 1 kHz tone 1.0. 0.15 sits in the gap.
    pub band_min_mid: f32,
    /// `snr_db` above which the out-of-band veto applies, in dB.
    pub band_veto_snr_db: f32,
    /// Multiplier applied to the score of an out-of-band frame.
    pub band_penalty: f32,
    /// Time constant of the per-bin noise estimate's **rise**, in ms.
    ///
    /// 500 ms. The estimate has to be slow enough that a talker does not become
    /// the noise, and fast enough that a fan being switched on is absorbed
    /// within a second. This is the single value that trades those two against
    /// each other, and it is a time constant rather than a per-frame
    /// coefficient so it does not silently change meaning if the hop does.
    pub noise_attack_ms: i64,
    /// Time constant of the per-bin noise estimate's **fall**, in ms.
    ///
    /// 50 ms, i.e. five frames. Fast on purpose: this is the "the room just got
    /// quieter, snap to it" direction, and a slow fall means a meeting that
    /// started in a noisy café stays calibrated to the café for the whole job.
    pub noise_release_ms: i64,
    /// Fraction of the noise estimate left in each bin after subtraction.
    ///
    /// 0.15. Without a floor, subtracting a matched white-noise estimate from a
    /// white-noise frame leaves a spiky residue whose round-off measures as a
    /// harmonic comb, so a noise frame reads as a textbook vowel. See
    /// [`NoiseSpectrum`].
    pub noise_floor: f32,
    /// `voicing` at or above which a frame is considered voice-shaped and is
    /// therefore **not** allowed to raise the adaptive noise floor.
    ///
    /// Not used on its own; see [`Self::gate_shape`]. Retained as the
    /// voiceless-but-structured second clause's companion, and exposed because
    /// `voicing_null` and `voicing_span` are the numbers that make it
    /// meaningful.
    pub gate_voicing: f32,
    /// `flatness` at or below which a frame counts as strongly structured.
    pub gate_flatness: f32,
    /// A frame is voice-shaped, and therefore excluded from the noise-floor
    /// update, when
    ///
    /// ```text
    /// voicing_term + flatness_term >= gate_shape
    /// ```
    ///
    /// where both terms are the same `[0, 1]` mappings the score uses.
    ///
    /// **Why a sum and not a conjunction.** A conjunction of "voiced *and*
    /// non-flat" is the obvious form and it is wrong: an unvoiced fricative is
    /// non-flat but not voiced, so a conjunction admits every /s/ to the floor.
    /// Measured on the sweep fixture, that made the floor jump to the level of
    /// the fricatives inside an utterance and swung the detector's own `snr_db`
    /// between 1.0 dB and 46 dB *within the same sentence*.
    ///
    /// **Why a sum and not an OR.** An OR of the same two terms admits a steady
    /// mains hum (voicing 1.0) and a test tone, which is a different problem --
    /// see [`steady_source_penalty`] -- but it also admits a broadband noise
    /// burst whose flatness happens to dip.
    ///
    /// 0.8 is the level where at least one of the two terms is most of the way
    /// to "speech": a vowel scores 1.8, a fricative 0.94, white noise 0.00, a
    /// fan 0.10. The gap between the fricative and the fan is where 0.8 sits.
    pub gate_shape: f32,
    /// Share of the shape term taken by the voicing feature.
    pub w_voicing: f32,
    /// Share of the shape term taken by the tonality feature.
    pub w_tonality: f32,
    /// Share of the shape term taken by the flatness feature.
    pub w_flatness: f32,
    /// Share of the shape term taken by the spectral shape feature.
    pub w_spectral: f32,
    /// Share of the shape term taken by the zero-crossing feature.
    pub w_zcr: f32,
}

impl Default for FeatureWeights {
    /// The tuned defaults. The reasoning for each value is in the module docs
    /// and the measured effect of each is in the tuning table in the final
    /// report; `energy.rs` documents the ones that are not self-evident.
    fn default() -> Self {
        Self {
            // 10 dB above the room is "clearly audible speech" to a listener in
            // that room. Measured on the fixtures: at this value a voice 6 dB
            // above the floor scores 0.60 and clears the enter threshold, and a
            // voice at 0 dB scores 0 and is not detected at all. The 0 dB limit
            // is a real property of the product form, not a tuning oversight --
            // see the module docs -- and it is the strongest argument for the
            // neural path.
            snr_full_db: 10.0,
            // tonality as 10*log10(peak/median). Measured with the definition in
            // `FrameFeatures::tonality`: white noise 13.1-13.9 dB, a harmonic
            // voice 24-27 dB (p05 15-18), a 62 Hz bass 24 dB, a fan with hum 47,
            // a steady tone 138. The mapping endpoints sit in the gap between
            // the noise cluster and the speech cluster's 5th percentile.
            tonality_lo_db: 16.0,
            tonality_hi_db: 22.0,
            // flatness (geometric mean / arithmetic mean). Measured: white
            // noise 0.56 (p05 0.52), a harmonic voice 0.008, a fricative-heavy
            // speech frame 0.24, a fan-with-hum 0.010, a tone 0.000. Note the
            // Wiener-entropy flatness of a *Gaussian* spectrum is about 0.37 by
            // Jensen's inequality, not 1.0, and forgetting that is how a
            // threshold gets placed on the wrong side of the noise cluster.
            // 0.28 -> 0 and 0.08 -> 1 puts the two clusters 0.24 apart.
            flatness_hi: 0.28,
            flatness_lo: 0.08,
            // hf_ratio: a broadband hiss puts 5/8 of its energy above 3 kHz and
            // measures 0.64; voiced speech measures 0.11 and a fricative-heavy
            // frame 0.41. Mapping 0.50 -> 0 and 0.15 -> 1 separates them.
            hiss_lo: 0.50,
            hiss_hi: 0.15,
            // rolloff at 85 %: hiss 6.9 kHz, a steady tone 1.0 kHz, a fan with
            // hum 94 Hz, a harmonic voice 2.5 kHz (p95 4.3 kHz). The endpoints
            // are placed so the hum and the voice both reach 1 while the hiss
            // reaches 0.
            rolloff_lo_hz: 5_000.0,
            rolloff_hi_hz: 2_800.0,
            // The low side. Measured: a 50/100/150 Hz hum has 85 % of its
            // energy below 94 Hz, so anything at or below 150 Hz is sub-bass
            // and the term is 0. A voice's rolloff is 1.4-4.3 kHz and a hiss's
            // 6.9 kHz, so 600 Hz is a safe floor for "clearly not a rumble".
            rolloff_floor_lo_hz: 150.0,
            rolloff_floor_hi_hz: 600.0,
            rolloff_share: 0.4,
            // From the measurement, not from theory: the 95th percentile of
            // `voicing` over 400-sample frames of white noise is 0.184 and its
            // median 0.159. 0.19 sits just above the observed noise tail so
            // noise is not credited with voicing, and a 0.26 span puts the voice
            // median (0.41) at 0.85 and the voice p95 (0.88) at 1.
            voicing_null: 0.19,
            voicing_span: 0.26,
            // Speech ZCR is 0.08-0.22 with a mode near 0.10 (voiced, low
            // pitches dominate real speech time); white noise measures 0.51
            // and a 50 Hz hum 0.04. A trapezoid peaked at 0.12 with a 0.32 span
            // gives speech ~0.9, noise 0, and the hum 0.75.
            zcr_peak: 0.12,
            zcr_span: 0.32,
            // Measured clusters with the energy-weighted flux: a steady 1 kHz
            // tone 0.00, a 50/100/150 Hz hum 0.10, a fan 0.10, a harmonic voice
            // p05 0.87 / p50 2.8, white noise p50 1.9. 0.3 is the middle of the
            // log gap.
            flux_min_db: 0.3,
            // Only a frame that is *also* clearly above the floor gets vetoed,
            // so a quiet but steady signal (a distant hum) does not have its
            // score cut twice.
            flux_veto_snr_db: 10.0,
            flux_penalty: 0.25,
            // 0.15 sits in the measured gap between a mains hum (0.001) and a
            // voice (0.4+). A 60 Hz bass voice is the worst case for this
            // feature and still measures above 0.4, because its harmonics reach
            // the formant band even when its fundamental does not.
            band_min_mid: 0.15,
            band_veto_snr_db: 8.0,
            band_penalty: 0.22,
            noise_attack_ms: 500,
            noise_release_ms: 50,
            noise_floor: 0.15,
            gate_voicing: 0.32,
            gate_flatness: 0.15,
            gate_shape: 0.80,
            // Voicing is the single most reliable single-frame speech cue, so
            // it takes the largest share. ZCR is the weakest (it is only
            // diagnostic for the hum case) and takes the smallest.
            w_voicing: 0.32,
            w_tonality: 0.20,
            w_flatness: 0.16,
            w_spectral: 0.20,
            w_zcr: 0.12,
        }
    }
}

/// Whether a frame looks like a voice, using **only floor-independent** features.
///
/// # This is the gate on the adaptive noise floor, and the reason it is not the score
///
/// A percentile floor is dominated by the quietest part of its window, so it is
/// immune to a loud *spike* -- one cough, one door. It is **not** immune to a
/// sustained sound: a 6-second window filled with a 6-second utterance is 100 %
/// speech, and its 20th percentile is a speech frame. The floor rises to the
/// talker, `snr_db` collapses to zero in the middle of a sentence, and the
/// detector switches itself off. Measured on the sweep fixture before this gate
/// existed, the detector's own `snr_db` inside planted speech was 1.1 dB at a
/// nominal 6 dB SNR and 8.1 dB at a nominal 24 dB SNR: it was tracking the
/// utterance, not the room. Widening the window from 2 s to 6 s recovered about
/// 9 % of the score and then saturated, because the problem is not the window
/// length -- it is that speech is *in* the window.
///
/// The obvious fix is to gate the floor update on the frame's *score*, and that
/// deadlocks. At a low SNR the score is low, so the frame is not gated out, so
/// it raises the floor, so the SNR is lower, so the score is lower. Measured, a
/// score-based gate moved the speech score at a 12 dB SNR from 0.069 to 0.069 --
/// exactly nothing, because every frame scored 0.069, which is below any
/// sensible gate.
///
/// So the gate has to be computed from features that **do not depend on the
/// floor at all**: `voicing` is an autocorrelation of the waveform and `flatness`
/// is a ratio within one frame. Neither reads the floor. A frame that is
/// voice-shaped by those tests is not evidence about the room, and is excluded
/// from the floor's window.
///
/// # The two clauses, and why there are two
///
/// `voicing` alone would admit every unvoiced fricative -- and fricatives are a
/// large share of real speech, so the floor would be dragged up by a /s/ and the
/// next /s/ would be missed. `flatness` alone would admit a steady hum, a test
/// tone and any narrowband sound. Either alone is wrong; the conjunction is what
/// a voice satisfies and a noise source does not.
///
/// # The failure mode, stated honestly
///
/// A **sustained tonal** sound -- a mains hum, a sustained whistle, a test tone
/// -- satisfies both clauses (voicing 1.0 because lags that are multiples of its
/// period correlate perfectly, flatness 0.0) and so is **never** allowed into
/// the floor. That is deliberate: the floor is a *room* estimate, and a tonal
/// sound is not the room. Those signals are rejected by
/// [`steady_source_penalty`] and [`out_of_band_penalty`] instead, which is why
/// both of those exist and why neither is optional.
#[must_use]
pub fn is_voice_shaped(f: &FrameFeatures, w: &FeatureWeights) -> bool {
    voicing_term(f, w) + flatness_term(f, w) >= w.gate_shape
}

/// The voicing half of the shape term, in [0, 1]. Factored out so the noise
/// floor's gate and the score use *the same mapping*; two copies of a threshold
/// is two numbers to keep in step.
#[must_use]
pub fn voicing_term(f: &FrameFeatures, w: &FeatureWeights) -> f32 {
    map_range(f.voicing, w.voicing_null, w.voicing_null + w.voicing_span)
}

/// The flatness half of the shape term, in [0, 1].
#[must_use]
pub fn flatness_term(f: &FrameFeatures, w: &FeatureWeights) -> f32 {
    map_range(f.flatness, w.flatness_hi, w.flatness_lo)
}

#[must_use]
fn clamp01(v: f32) -> f32 {
    if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) }
}

/// The median of a slice, by a 512-bucket counting histogram over its range.
///
/// `tonality` needs this 100 times per second of audio. A `select_nth_unstable`
/// on 256 elements is about 2000 comparisons, i.e. 200 000 per second of audio,
/// and the detector's whole per-frame budget is dominated by one 512-point FFT.
/// A histogram is 512 increments plus a short scan.
///
/// The bucket count is deliberately coarse relative to the dynamic range: the
/// answer only feeds a ratio through a `10*log10` and a threshold, and 512
/// buckets over the slice's own min..max is a resolution of about 0.2 % of the
/// range, which is far below anything the threshold cares about. A slice whose
/// values span zero (all bins at the subtraction floor) would make the range
/// degenerate, hence the `max`.
#[must_use]
pub fn median_of(values: &[f32]) -> f32 {
    const BUCKETS: usize = 512;
    if values.is_empty() {
        return 0.0;
    }
    if values.len() == 1 {
        return values[0];
    }
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for v in values {
        if v.is_finite() {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
    }
    if !lo.is_finite() || hi <= lo {
        return lo.max(0.0);
    }
    let scale = (BUCKETS - 1) as f32 / (hi - lo);
    let mut hist = [0u32; BUCKETS];
    let mut counted = 0usize;
    for v in values {
        if !v.is_finite() {
            continue;
        }
        let b = ((v - lo) * scale) as usize;
        hist[b.min(BUCKETS - 1)] += 1;
        counted += 1;
    }
    if counted == 0 {
        return 0.0;
    }
    // Lower median, matching the "floor(q*n)" convention used everywhere else
    // in this crate.
    let target = (counted / 2) as u32;
    let mut acc = 0u32;
    for (b, c) in hist.iter().enumerate() {
        acc += *c;
        if acc > target {
            return lo + b as f32 / scale;
        }
    }
    hi
}

#[must_use]
fn map_range(v: f32, lo: f32, hi: f32) -> f32 {
    if (hi - lo).abs() < f32::EPSILON {
        0.0
    } else {
        clamp01((v - lo) / (hi - lo))
    }
}

/// The audibility half of the score, in [0, 1].
///
/// `snr_db` is measured against the adaptive floor, so this is "how far above
/// *this room*", which is the only formulation that survives a fan being turned
/// on halfway through a meeting.
#[must_use]
pub fn audibility_term(snr_db: f32, w: &FeatureWeights) -> f32 {
    map_range(snr_db, 0.0, w.snr_full_db)
}

/// The shape half of the score, in [0, 1].
///
/// A weighted mean of five independent speechiness terms, each already in
/// [0, 1]. The weights are normalised at use time rather than assumed to sum
/// to 1, so editing one weight in a tuning run cannot silently rescale the
/// whole term.
#[must_use]
pub fn shape_term(f: &FrameFeatures, w: &FeatureWeights) -> f32 {
    let voicing = voicing_term(f, w);
    let tonality = map_range(
        10.0 * f.tonality.max(1e-12).log10(),
        w.tonality_lo_db,
        w.tonality_hi_db,
    );
    let flatness = flatness_term(f, w);
    let tilt = map_range(f.hf_ratio, w.hiss_lo, w.hiss_hi);
    // Two-sided: a frame whose 85 %-energy point is *below* the voice range is
    // a rumble just as surely as one whose point is above it is a hiss. A
    // single-sided ramp saturates at 1 for a 50 Hz hum, which is how a hum
    // ends up looking maximally speech-shaped.
    let rolloff_high = map_range(f.rolloff_hz, w.rolloff_lo_hz, w.rolloff_hi_hz);
    let rolloff_low = map_range(f.rolloff_hz, w.rolloff_floor_lo_hz, w.rolloff_floor_hi_hz);
    let rolloff = rolloff_high.min(rolloff_low);
    let spectral = w
        .rolloff_share
        .mul_add(rolloff, (1.0 - w.rolloff_share) * tilt);
    let zcr = clamp01(1.0 - (f.zcr - w.zcr_peak).abs() / w.zcr_span.max(1e-6));

    let total = w.w_voicing + w.w_tonality + w.w_flatness + w.w_spectral + w.w_zcr;
    if total <= 0.0 {
        return 0.0;
    }
    let acc = w.w_voicing * voicing
        + w.w_tonality * tonality
        + w.w_flatness * flatness
        + w.w_spectral * spectral
        + w.w_zcr * zcr;
    clamp01(acc / total)
}

/// The steady-source veto.
///
/// A frame that is far above the noise floor *and* whose per-band spectrum is
/// not changing is a steady sound: mains hum, a fan, a sustained whistle, a
/// test tone, a held synth note. None of those is speech, and all of them have
/// high level, which is precisely the case a level-based rule gets wrong.
///
/// The `snr_db` guard matters: without it, a *quiet* steady source would be
/// penalised twice (it is near the floor and it is steady), and the penalty
/// would push a genuinely quiet but moving speech frame under the threshold
/// whenever the room itself is very stable.
#[must_use]
pub fn steady_source_penalty(f: &FrameFeatures, w: &FeatureWeights) -> f32 {
    if f.flux < w.flux_min_db && f.snr_db > w.flux_veto_snr_db {
        w.flux_penalty
    } else {
        1.0
    }
}

/// The out-of-band veto: a frame with no energy in the formant band.
///
/// # Why this is a veto and not another weighted term
///
/// A 50 Hz mains hum with its 2nd and 3rd harmonics is **peaky** (tonality
/// 20 dB), **non-flat** (flatness 0.01), **periodic** (voicing 0.70, because
/// its 100 Hz component is inside the 60-400 Hz pitch search range and lags
/// that are multiples of its period correlate perfectly) and **low-ZCR**
/// (0.04). Four of the five shape features accept it. Measured on the
/// fixtures, its score is 0.97 -- a confident, confident wrong answer.
///
/// That is not a tuning problem, it is structural. The evidence that
/// distinguishes it is a single fact -- *where* its energy is -- and averaging
/// one rejecting fact with four accepting ones is guaranteed to lose. A frame
/// whose energy is 0.1 % between 500 Hz and 3.5 kHz is a rumble, full stop.
///
/// A voice does not have that property at any pitch. A 60 Hz bass has 58
/// harmonics below 3.5 kHz and a formant structure that puts most of its
/// *radiated* energy between 300 Hz and 3 kHz; measured on the fixtures,
/// `mid_ratio` is 0.4-0.8 for every synthetic voice from 95 Hz to 185 Hz and
/// 0.001 for the hum. A broadband hiss measures 0.9 and a 1 kHz tone 1.0, so
/// this veto does not touch either.
///
/// # The `snr_db` guard
///
/// Without it, a quiet rumble would be penalised twice: once for being near the
/// floor and once for being out of band. That matters in a genuinely quiet
/// room, where the second penalty would push a quiet *voice* under the
/// threshold whenever the room itself has a low-frequency rumble.
#[must_use]
pub fn out_of_band_penalty(f: &FrameFeatures, w: &FeatureWeights) -> f32 {
    if f.mid_ratio < w.band_min_mid && f.snr_db > w.band_veto_snr_db {
        w.band_penalty
    } else {
        1.0
    }
}

/// The combined decision score for one frame, in [0, 1].
///
/// `score = audibility * shape * steady_penalty * out_of_band_penalty`. Every
/// factor is reported separately by [`frame_score`], so a debug log can show
/// which one rejected a frame.
#[must_use]
pub fn frame_score(f: &FrameFeatures, w: &FeatureWeights) -> f32 {
    clamp01(
        audibility_term(f.snr_db, w)
            * shape_term(f, w)
            * steady_source_penalty(f, w)
            * out_of_band_penalty(f, w),
    )
}

/// [`frame_score`] with every intermediate, for a tuning run or a debug log.
///
/// The `Vec` is one allocation per call and this is not on the hot path --
/// [`frame_score`] is -- so the allocation is acceptable for the price of being
/// able to answer "why was this frame rejected" without re-deriving it.
#[must_use]
pub fn frame_score_parts(f: &FrameFeatures, w: &FeatureWeights) -> [f32; 4] {
    [
        audibility_term(f.snr_db, w),
        shape_term(f, w),
        steady_source_penalty(f, w),
        out_of_band_penalty(f, w),
    ]
}

/// Frame timing helpers, kept next to the frame rate they describe.
pub const fn frames_to_ms(frames: usize) -> i64 {
    (frames as i64 * HOP_SAMPLES as i64 * 1000) / SAMPLE_RATE as i64
}

/// Milliseconds to whole frames on the energy detector's 10 ms grid.
///
/// Rounding *up* is deliberate everywhere a duration is a floor (minimum
/// speech, minimum silence, minimum chunk): a 200 ms minimum implemented as
/// `200/10 = 20` frames is exact, but 250 ms as `25` frames is not, and
/// rounding down would make a stated minimum quietly smaller than advertised.
#[must_use]
pub fn ms_to_frames_ceil(ms: i64) -> usize {
    ms_to_frames_at_ceil(ms, HOP_MS)
}

/// As [`ms_to_frames_ceil`], on an explicit frame grid.
///
/// The energy detector runs at [`HOP_MS`] and the neural one at 16 ms, so
/// "minimum speech 200 ms" is 20 frames in one and 13 in the other. A duration
/// that is expressed in milliseconds has to be converted against the *caller's*
/// grid, or one of the two engines silently applies a different rule from the
/// one that is documented.
#[must_use]
pub fn ms_to_frames_at_ceil(ms: i64, hop_ms: i64) -> usize {
    if ms <= 0 || hop_ms <= 0 {
        return 0;
    }
    ((ms + hop_ms - 1) / hop_ms) as usize
}

/// Frame index to milliseconds, on an explicit grid.
#[must_use]
pub fn frames_to_ms_at(frames: usize, hop_ms: i64) -> i64 {
    frames as i64 * hop_ms
}

/// The start time of frame `i` within a buffer that begins at `buffer_start`.
#[must_use]
pub fn frame_start_ms(frame: usize, buffer_start: i64) -> i64 {
    buffer_start + frames_to_ms(frame)
}

/// The end time of frame `i` (exclusive), i.e. the start of the next frame.
///
/// A half-open interval means adjacent segments can share a boundary instant
/// without overlapping, which is what `SpeechSegment`'s consumers assume.
#[must_use]
pub fn frame_end_ms(frame: usize, buffer_start: i64) -> i64 {
    buffer_start + frames_to_ms(frame + 1)
}

/// Computes [`FrameFeatures`] frame by frame.
///
/// Owns the FFT plan and the previous frame's band energies, so it is `!Sync`
/// by construction and one is created per `detect()` call.
pub struct FeatureExtractor {
    spectrum: Spectrum,
    /// The per-bin noise estimate, and the subtracted spectrum it produces.
    noise: NoiseSpectrum,
    /// The subtracted spectrum of the previous frame, for the flux measure.
    prev_bands_db: [f32; N_FLUX_BANDS],
    have_prev: bool,
    /// Bin index at which each flux band starts, precomputed.
    band_start_bin: [usize; N_FLUX_BANDS],
    band_end_bin: [usize; N_FLUX_BANDS],
    /// Index of the first bin above `HF_EDGE_HZ`.
    hf_start_bin: usize,
    /// Index of the first bin above `LF_EDGE_HZ`.
    lf_end_bin: usize,
    /// Bin range of the speech formant band.
    mid_start_bin: usize,
    mid_end_bin: usize,
    /// Cumulative energy, for the rolloff, recomputed per frame.
    weights: FeatureWeights,
    floor_db: f32,
}

impl std::fmt::Debug for FeatureExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeatureExtractor")
            .field("have_prev", &self.have_prev)
            .field("floor_db", &self.floor_db)
            .finish_non_exhaustive()
    }
}

impl FeatureExtractor {
    /// The floor in dB used for the current frame's `snr_db`.
    ///
    /// Set by [`Self::set_floor`] before each frame; the extractor does not
    /// own the floor because the floor's window is a *policy* decision
    /// ([`crate::noise::SlidingQuantile`]) and the feature layer's job is to
    /// measure, not to decide.
    #[must_use]
    pub fn floor_db(&self) -> f32 {
        self.floor_db
    }

    /// Set the floor applied to the next frame.
    pub fn set_floor(&mut self, floor_db: f32) {
        self.floor_db = floor_db;
    }

    /// Analyse one frame of exactly [`WINDOW_SAMPLES`] samples.
    pub fn push(&mut self, frame: &[f32]) -> FrameFeatures {
        let window_energy = self.spectrum.window_energy();
        let power = self.spectrum.power(frame).to_vec();

        // The *level* is measured on the raw spectrum: it is a measurement of
        // what arrived, and subtracting the room from it would make the audibility
        // term circular.
        let mut total = 0f32;
        for p in power.iter() {
            total += *p;
        }
        // Frame mean square, via Parseval. Without the window-energy divisor
        // this is off by a factor of `WINDOW_SAMPLES`, i.e. 26 dB, which is
        // enough to move every SNR comparison by two orders of magnitude.
        //
        // The divisor is `n_fft / 2`, not `n_fft`, because the sum runs over the
        // 257 **non-negative** frequency bins. For a real signal the mirrored
        // half carries the same power, so summing only one half halves the
        // result -- exactly 3.01 dB, which is precisely the size of the Hann
        // window's own loss and therefore the easiest error in this file to make
        // and hardest to see. Caught by `db_is_corrected_for_the_window_gain`.
        let ms = total / ((N_FFT / 2) as f32 * window_energy);
        let db = 10.0 * ms.max(1e-20).log10();

        // Every *shape* feature is measured on the noise-subtracted spectrum.
        // This is the single most consequential line in the file: see
        // `NoiseSpectrum` for the measurement that justifies it.
        let sub = self.noise.process(&power);
        let mut sub_total = 0f32;
        for p in sub.iter() {
            sub_total += *p;
        }

        // --- flatness, tonality, band ratios, rolloff ----------------------
        // Bin 0 is excluded: it is the DC bin, whose value is the windowed
        // frame's residual mean and is not part of any spectral statistic.
        //
        // The median is computed with a 257-bucket counting histogram rather
        // than by sorting, because this is the hot loop -- 100 frames per
        // second of audio -- and a sort here would dominate the detector's cost.
        let mut log_sum = 0f32;
        let mut peak = 0f32;
        let mut lf = 0f32;
        let mut hf = 0f32;
        let mut mid = 0f32;
        let mut bins = 0f32;
        for (k, &p) in sub.iter().enumerate() {
            if k == 0 {
                // The DC bin is excluded from every ratio below.
                continue;
            }
            log_sum += p.max(1e-20).ln();
            if p > peak {
                peak = p;
            }
            if k < self.lf_end_bin {
                lf += p;
            }
            if k >= self.hf_start_bin {
                hf += p;
            }
            if k >= self.mid_start_bin && k < self.mid_end_bin {
                mid += p;
            }
            bins += 1.0;
        }
        let median = median_of(&sub[1..N_FREQ]);
        let geo = if bins > 0.0 {
            (log_sum / bins).exp()
        } else {
            0.0
        };
        let mean = sub_total / N_FREQ as f32;
        let flatness = if mean > 0.0 { geo / mean } else { 1.0 };
        // `peak / median`, with the median floored so a frame whose every bin is
        // at the numerical minimum does not produce an infinite ratio. The
        // *denominator* uses `N_FREQ` rather than `(N_FREQ - 1)`: it is a reference
        // level for a ratio, not a mean square, so the half-spectrum factor does
        // not apply and there is nothing to correct.
        let denom = median.max(1e-20);
        let tonality = if peak > 0.0 { peak / denom } else { 1.0 };

        let hf_ratio = if sub_total > 0.0 { hf / sub_total } else { 0.0 };
        let lf_ratio = if sub_total > 0.0 { lf / sub_total } else { 0.0 };
        let mid_ratio = if sub_total > 0.0 {
            mid / sub_total
        } else {
            0.0
        };

        let rolloff_hz = if sub_total > 0.0 {
            let threshold = 0.85 * sub_total;
            let mut acc = 0f32;
            let mut hz = 0f32;
            for (k, &p) in sub.iter().enumerate().skip(1) {
                acc += p;
                if acc >= threshold {
                    hz = k as f32 * SAMPLE_RATE as f32 / 2.0 / (N_FREQ - 1) as f32;
                    break;
                }
            }
            hz
        } else {
            0.0
        };

        // --- flux ------------------------------------------------------------
        // **Energy-weighted**, and taken on the subtracted spectrum so a band
        // the noise dominated cannot contribute dBs of "change" from round-off.
        let had_prev = self.have_prev;
        let mut flux = if had_prev { 0.0 } else { FLUX_UNDEFINED };
        let mut weight_total = 0f32;
        for b in 0..N_FLUX_BANDS {
            let e: f32 = sub[self.band_start_bin[b]..self.band_end_bin[b]]
                .iter()
                .sum();
            if had_prev {
                let band_db = 10.0 * e.max(1e-20).log10();
                // A floor of 1e-6 of the frame's total, so a band that is
                // *almost* empty still counts a little and the weights never sum
                // to zero for a non-silent frame.
                let w = e.max(sub_total * 1e-6);
                flux += w * (band_db - self.prev_bands_db[b]).abs();
                weight_total += w;
            }
            self.prev_bands_db[b] = 10.0 * e.max(1e-20).log10();
        }
        self.have_prev = true;
        if had_prev {
            flux = if weight_total > 0.0 {
                flux / weight_total
            } else {
                0.0
            };
        }

        FrameFeatures {
            db,
            snr_db: db - self.floor_db,
            flatness,
            tonality,
            voicing: normalised_autocorrelation_peak(frame, 60.0, 400.0),
            hf_ratio,
            lf_ratio,
            mid_ratio,
            rolloff_hz,
            zcr: zero_crossing_rate(frame),
            flux,
        }
    }

    /// The weights, for callers that want to re-score stored features.
    #[must_use]
    pub fn weights(&self) -> &FeatureWeights {
        &self.weights
    }
}

impl FeatureExtractor {
    /// Build an extractor.
    ///
    /// `weights` is held only so [`Self::weights`] can report it; scoring is
    /// done by [`frame_score`] with an explicit `FeatureWeights` so the same
    /// function serves the live path and a tuning sweep over stored features.
    #[must_use]
    pub fn new(weights: FeatureWeights) -> Self {
        let bin_hz = SAMPLE_RATE as f32 / 2.0 / (N_FREQ - 1) as f32;
        let mut band_start_bin = [0usize; N_FLUX_BANDS];
        let mut band_end_bin = [0usize; N_FLUX_BANDS];
        for b in 0..N_FLUX_BANDS {
            let lo = (FLUX_BAND_HZ[b] / bin_hz).floor() as usize;
            let hi = (FLUX_BAND_HZ[b + 1] / bin_hz).ceil() as usize;
            band_start_bin[b] = lo.clamp(1, N_FREQ);
            band_end_bin[b] = hi.clamp(band_start_bin[b] + 1, N_FREQ);
        }
        Self {
            spectrum: Spectrum::new(),
            noise: NoiseSpectrum::new(
                HOP_MS,
                weights.noise_attack_ms,
                weights.noise_release_ms,
                weights.noise_floor,
            ),
            prev_bands_db: [0.0; N_FLUX_BANDS],
            have_prev: false,
            band_start_bin,
            band_end_bin,
            hf_start_bin: (HF_EDGE_HZ / bin_hz).ceil() as usize,
            lf_end_bin: (LF_EDGE_HZ / bin_hz).floor() as usize,
            mid_start_bin: (MID_EDGE_LO_HZ / bin_hz).ceil() as usize,
            mid_end_bin: (MID_EDGE_HI_HZ / bin_hz).ceil() as usize,
            weights,
            floor_db: -120.0,
        }
    }
}

/// Peak of the normalised autocorrelation over the pitch lag range.
///
/// `min_hz`/`max_hz` bound the fundamental frequency the detector will credit
/// as voicing. 60 Hz is the lowest pitch a human voice reaches (a large
/// bass); 400 Hz is above a child's and above an adult's comfortable speaking
/// pitch, so anything faster is not being counted as a periodic signal -- which
/// is the right call, because a 1 kHz test tone has 227 candidate lags
/// available in a 400-sample frame and the largest normalised autocorrelation
/// over all of them is high by chance.
///
/// The signal is mean-removed first: autocorrelation is a *similarity* measure
/// and a DC offset correlates perfectly with itself at every lag, which would
/// make any signal with a bias look perfectly periodic.
///
/// # The result is in [-1, 1]
///
/// Cauchy-Schwarz bounds `|sum a_i b_i| <= sqrt(sum a^2 * sum b^2)`, and the
/// divisor is exactly the two sides' energies over the overlapping range, so
/// no frame can produce a value outside that range. A test asserts it, because
/// the obvious transcription of this formula does not and the failure is
/// invisible -- `voicing` just becomes a larger number and the score silently
/// saturates.
#[must_use]
pub fn normalised_autocorrelation_peak(samples: &[f32], min_hz: f32, max_hz: f32) -> f32 {
    let n = samples.len();
    if n < 8 || min_hz <= 0.0 || max_hz <= min_hz {
        return 0.0;
    }
    let lag_min = (SAMPLE_RATE as f32 / max_hz).ceil() as usize;
    let lag_max = (SAMPLE_RATE as f32 / min_hz).floor() as usize;
    let lag_min = lag_min.max(1);
    let lag_max = lag_max.min(n / 2);
    if lag_max <= lag_min {
        return 0.0;
    }

    let mean = samples.iter().sum::<f32>() / n as f32;
    let x: Vec<f32> = samples.iter().map(|v| v - mean).collect();
    // Prefix energy, so each lag's two sums are O(1).
    let mut prefix = vec![0f32; n + 1];
    for i in 0..n {
        prefix[i + 1] = prefix[i] + x[i] * x[i];
    }
    let total_energy = prefix[n];
    if total_energy <= 0.0 {
        return 0.0;
    }

    let mut best = 0f32;
    for lag in lag_min..=lag_max {
        let tail = n - lag;
        // The autocorrelation at `lag` pairs `x[0..tail]` with `x[lag..n]`, so
        // the two energies are the energy of *those two ranges* -- the head is
        // `tail` samples long, not `lag` samples long. Using `prefix[lag]` here
        // is a real bug: it divides by the wrong energy, the ratio is not
        // bounded by 1, and `voicing` comes out as 6.4 for a signal whose true
        // correlation is 0.98. Cauchy-Schwarz then does not apply and the whole
        // feature is meaningless.
        let e1 = prefix[tail];
        let e2 = prefix[n] - prefix[lag];
        if e1 <= 0.0 || e2 <= 0.0 {
            continue;
        }
        let mut acc = 0f32;
        for i in 0..tail {
            acc += x[i] * x[i + lag];
        }
        let r = acc / (e1 * e2).sqrt();
        if r > best {
            best = r;
        }
    }
    if best.is_nan() { 0.0 } else { best }
}

/// Per-bin noise-spectrum tracker, and the spectral subtraction built on it.
///
/// # Why this exists, and what it bought
///
/// The first version of this detector measured flatness, tonality and the band
/// ratios on the **raw** spectrum. On synthetic fixtures that looked fine, and
/// then the measured sensitivity curve on a *continuous* broadband floor said
/// otherwise: the mean frame score inside planted speech was 0.010 at a 6 dB
/// SNR, 0.069 at 12 dB and 0.447 at 24 dB. Recall was zero below 18 dB SNR.
///
/// The reason is not subtle and it is not fixable by moving a threshold. Every
/// shape feature is a *ratio* of spectral quantities, so adding a broadband
/// floor inflates the denominator and fills the empty bins: a harmonic comb
/// measured at flatness 0.008 becomes 0.19 the moment a 12 dB-SNR noise floor
/// is added under it, because ~200 of the 256 bins are no longer empty. The
/// audibility term is already saturated at 12 dB, so the score is entirely
/// limited by the shape term and there is nothing left to trade.
///
/// Subtracting the estimated noise spectrum before measuring the shape
/// restores the contrast, and the measurement after it is the same curve with
/// recall 1.0 from 6 dB.
///
/// # The tracker
///
/// One exponential estimate per bin, with an asymmetric response:
///
/// ```text
/// P < N  ->  N += release * (P - N)      // the room got quieter: follow it down
/// P >= N ->  N += attack  * (P - N)      // the room got louder: follow it up slowly
/// ```
///
/// This is minimum statistics in one line per bin, and the asymmetry is the
/// whole point. A symmetric tracker with a fast response follows a talker and
/// then subtracts the talker; a symmetric tracker with a slow response takes
/// minutes to notice a fan being switched on. Snapping down and creeping up
/// does neither.
///
/// `attack` is expressed as a **time constant in milliseconds** rather than a
/// coefficient, because the coefficient that means "half a second" depends on
/// the hop and the hop is a parameter.
///
/// # The subtraction floor
///
/// ```text
/// P'[k] = max(P[k] - N[k], floor * N[k])
/// ```
///
/// The floor is not cosmetic and getting it wrong inverts the result. Without
/// it, subtracting a *white noise* estimate from a white noise frame leaves
/// `max(P - N, 0)`, which is a **spiky** spectrum full of near-zero bins whose
/// values are FFT round-off. Its flatness is very low, its tonality is very
/// high, and the detector reports a noise frame as a textbook harmonic voice.
/// With the floor, the same frame leaves a flat residual at `floor * N` in
/// every bin, which measures as flat and unvoiced, which is what it is.
#[derive(Debug, Clone)]
pub struct NoiseSpectrum {
    est: Vec<f32>,
    /// Rise coefficient: `1 - exp(-hop / attack)`.
    up: f32,
    /// Fall coefficient: `1 - exp(-hop / release)`.
    down: f32,
    floor: f32,
    primed: bool,
}

impl NoiseSpectrum {
    /// `hop_ms` and the two time constants are in milliseconds.
    #[must_use]
    pub fn new(hop_ms: i64, attack_ms: i64, release_ms: i64, floor: f32) -> Self {
        let coeff = |tau_ms: i64| -> f32 {
            let tau = tau_ms.max(1) as f32;
            1.0 - (-hop_ms as f32 / tau).exp()
        };
        Self {
            est: vec![0.0; N_FREQ],
            up: coeff(attack_ms),
            down: coeff(release_ms),
            floor,
            primed: false,
        }
    }

    /// The current estimate, for tests and diagnostics.
    #[must_use]
    pub fn estimate(&self) -> &[f32] {
        &self.est
    }

    /// Update the estimate from a raw power spectrum and return the subtracted
    /// one.
    ///
    /// The update uses the *raw* spectrum, not the subtracted one: feeding the
    /// subtraction back into the tracker would make the estimate chase its own
    /// output and decay to zero over any quiet passage.
    pub fn process(&mut self, power: &[f32]) -> Vec<f32> {
        if power.len() < N_FREQ {
            return vec![0.0; N_FREQ];
        }
        if !self.primed {
            // Seed from the first frame rather than from zero. Starting at zero
            // makes the first second's subtraction a no-op, and the first
            // second of a file is exactly where a recording that opens with
            // speech is decided.
            self.est.copy_from_slice(&power[..N_FREQ]);
            self.primed = true;
        } else {
            // Zip rather than index: the estimate and the spectrum are the same
            // length by construction, and the two coefficients are disjoint field
            // borrows of `self`.
            for (&p, n) in power[..N_FREQ].iter().zip(self.est.iter_mut()) {
                let c = if p < *n { self.down } else { self.up };
                *n += c * (p - *n);
            }
        }
        let mut out = vec![0f32; N_FREQ];
        for k in 0..N_FREQ {
            let n = self.est[k];
            out[k] = (power[k] - n).max(self.floor * n).max(0.0);
        }
        out
    }
}

#[cfg(test)]
mod noise_tests {
    use super::*;
    use crate::dsp::WINDOW_SAMPLES;

    #[test]
    fn the_estimate_snaps_down_and_creeps_up() {
        let mut n = NoiseSpectrum::new(10, 500, 50, 0.15);
        let quiet: Vec<f32> = (0..N_FREQ).map(|_| 1e-4).collect();
        let loud: Vec<f32> = (0..N_FREQ).map(|_| 1.0).collect();
        n.process(&quiet);
        assert!((n.estimate()[10] - 1e-4).abs() < 1e-9);

        // 50 frames -- one second at the 20 ms hop -- of a 10 000x rise. The
        // attack constant is 50 frames, so one second of signal is exactly one
        // time constant: the estimate has moved, but by nowhere near as far as
        // the signal did.
        for _ in 0..50 {
            n.process(&loud);
        }
        let after_up = n.estimate()[10];
        assert!(
            after_up < 0.9,
            "50 frames of a 10000x rise moved the estimate all the way to {after_up}"
        );
        assert!(
            after_up > quiet[10] * 10.0,
            "the estimate did not move at all: {after_up}"
        );

        // The rate is the point, so it is checked against a tracker whose attack
        // time constant is ten times longer on the same input. After one second
        // the fast tracker is a third of the way to the signal; the slow one has
        // barely left where it started.
        let mut slow = NoiseSpectrum::new(10, 5_000, 50, 0.15);
        slow.process(&quiet);
        for _ in 0..50 {
            slow.process(&loud);
        }
        let slow_after = slow.estimate()[10];
        assert!(
            slow_after < after_up / 3.0,
            "the attack rate is not respected: {slow_after} (attack 5000 ms) vs {after_up} (attack 500 ms)"
        );

        // One quiet frame: the release is 6.7 frames, so it snaps most of the
        // way down -- a step down is information, a slow creep up is not.
        n.process(&quiet);
        let after_down = n.estimate()[10];
        assert!(
            after_down < 0.9 * after_up,
            "the estimate did not follow a quieter frame down: {after_up} -> {after_down}"
        );
    }

    #[test]
    fn the_first_frame_seeds_the_estimate() {
        let mut n = NoiseSpectrum::new(10, 500, 50, 0.15);
        let p: Vec<f32> = (0..N_FREQ).map(|_| 0.5).collect();
        let out = n.process(&p);
        // max(0.5 - 0.5, 0.15*0.5) = 0.075 in every bin: a flat residual.
        for v in &out {
            assert!((v - 0.075).abs() < 1e-6, "{v}");
        }
    }

    #[test]
    fn subtracting_noise_restores_the_flatness_of_a_harmonic_comb() {
        // The measurement that motivated the whole thing -- and the *sequence*
        // it needs. Subtraction cannot sharpen anything if the tracker is fed
        // the signal it is meant to remove: on a constant input the fixed point
        // is `est = p / 2`, the residual is a scaled copy of the input, and
        // every ratio feature is unchanged by construction. A noise *estimate*
        // has to be learned from noise-only frames, which is what the
        // production path does with its warm-up.
        let mut rng = 1234u64;
        let mut rand = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            ((rng >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        };
        let comb_amp = 0.2;
        // The noise floor is a tenth of the comb's amplitude, so it visibly
        // fills the inter-harmonic valleys without hiding the peaks.
        let noise_amp = 0.02;
        let comb: Vec<f32> = {
            let mut v = vec![0f32; WINDOW_SAMPLES];
            for (i, s) in v.iter_mut().enumerate() {
                let t = i as f32 / SAMPLE_RATE as f32;
                let mut x = 0f32;
                for h in 1..=10 {
                    x += (1.0 / h as f32)
                        * (2.0 * std::f32::consts::PI * 120.0 * h as f32 * t).sin();
                }
                *s = comb_amp * x;
            }
            v
        };
        let mut spec = crate::dsp::Spectrum::new();
        let noise_only: Vec<f32> = (0..WINDOW_SAMPLES).map(|_| noise_amp * rand()).collect();
        let noise_power = spec.power(&noise_only).to_vec();
        let mut noisy = comb.clone();
        for s in &mut noisy {
            *s += noise_amp * rand();
        }
        let dirty = spec.power(&noisy).to_vec();

        // **Why not spectral flatness here.** A comb's inter-harmonic valleys sit
        // around 1e-13 of its peaks, so Wiener's estimate -- a geometric mean over
        // bins clamped at 1e-20 -- is dominated by the clamp and barely moves
        // when a floor is added or removed: the measured improvement for this
        // fixture is 1.8x, and the assertion "subtraction recovers the contrast"
        // is not expressible in that unit. The floor is measured directly
        // instead: the median power in the valleys, over the median power on the
        // harmonics. That ratio is exactly what subtraction is supposed to move,
        // it is in dB-like units the caller can reason about, and it has no
        // numerical floor in it.
        let bin_hz = SAMPLE_RATE as f32 / N_FFT as f32;
        let mut on_harmonic = vec![false; N_FREQ];
        for h in 1..=10 {
            let b = ((120.0 * h as f32 / bin_hz).round() as i32).clamp(0, N_FREQ as i32 - 1);
            // Two bins either side: the Hann main lobe is wider than one bin.
            for d in -2..=2 {
                if b + d >= 0 {
                    on_harmonic[(b + d) as usize] = true;
                }
            }
        }
        let valley_floor = |p: &[f32]| -> f32 {
            let mut v: Vec<f32> = (1..N_FREQ)
                .filter(|k| !on_harmonic[*k])
                .map(|k| p[k])
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            v[v.len() / 2]
        };
        let peak_level = |p: &[f32]| -> f32 {
            let mut v: Vec<f32> = (1..N_FREQ)
                .filter(|k| on_harmonic[*k])
                .map(|k| p[k])
                .collect();
            v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            v[v.len() / 2]
        };
        let floor_ratio = |p: &[f32]| -> f32 { valley_floor(p) / peak_level(p) };

        let dirty_ratio = floor_ratio(&dirty);

        // Learn the floor from noise only, then let the comb arrive. 150 frames
        // is 1.5 s of pure noise: long enough for a 200 ms attack to converge on
        // the floor to within its per-bin sampling scatter, which is the limit on
        // how well *any* single-frame estimate can do here.
        let mut n = NoiseSpectrum::new(10, 200, 50, 0.15);
        for _ in 0..150 {
            n.process(&noise_power);
        }
        let mut last = Vec::new();
        for _ in 0..5 {
            last = n.process(&dirty);
        }
        let enhanced_ratio = floor_ratio(&last);

        // A tenth of the comb's amplitude is a -20 dB floor, so the valley level
        // starts around 0.3 % of the peak level and must end far below that.
        assert!(
            dirty_ratio > 0.002,
            "the fixture's floor is too low to measure: {dirty_ratio}"
        );
        assert!(
            enhanced_ratio < dirty_ratio / 2.5,
            "subtraction should remove most of the floor: {dirty_ratio} -> {enhanced_ratio}"
        );
        assert!(
            enhanced_ratio < 0.0015,
            "the residual still has a floor of {enhanced_ratio} of the peak level"
        );
    }

    #[test]
    fn subtracting_a_matched_noise_leaves_a_flat_residual_not_a_spiky_one() {
        // The failure the `floor` parameter exists to prevent. Without it, the
        // residual of a white-noise frame is `max(P - N, 0)`, which is spiky,
        // and the spiky residual measures as a harmonic voice.
        let mut rng = 99u32;
        let mut rand = || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            (rng as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let mut spec = crate::dsp::Spectrum::new();
        let mut last_floored = Vec::new();
        let mut last_bare = Vec::new();
        let mut n = NoiseSpectrum::new(10, 100, 50, 0.15);
        for _ in 0..200 {
            let frame: Vec<f32> = (0..WINDOW_SAMPLES).map(|_| 0.05 * rand()).collect();
            let p = spec.power(&frame).to_vec();
            last_floored = n.process(&p);
            // The same subtraction with no floor, computed here so the
            // comparison is in one place.
            last_bare = (0..N_FREQ)
                .map(|k| (p[k] - n.estimate()[k]).max(0.0))
                .collect();
        }
        let flatness = |p: &[f32]| -> f32 {
            let mut log_sum = 0f32;
            let mut total = 0f32;
            for v in p.iter() {
                total += *v;
            }
            for p in p.iter().skip(1) {
                log_sum += p.max(1e-20).ln();
            }
            (log_sum / (N_FREQ - 1) as f32).exp() / (total / N_FREQ as f32).max(1e-20)
        };
        let bare = flatness(&last_bare);
        let floored = flatness(&last_floored);
        assert!(
            floored > bare,
            "the floored residual should be *flatter* than the bare one: bare {bare}, floored {floored}"
        );
        assert!(
            floored > 0.3,
            "a noise frame's residual should still read as noise-like, got {floored}"
        );
    }

    #[test]
    fn a_short_or_empty_spectrum_is_handled() {
        let mut n = NoiseSpectrum::new(10, 500, 50, 0.15);
        assert_eq!(n.process(&[]).len(), N_FREQ);
        assert!(n.process(&[1.0; 4]).iter().all(|v| *v == 0.0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::WINDOW_SAMPLES;
    use std::f32::consts::PI;

    fn tone(buf: &mut [f32], hz: f32, amp: f32) {
        for (i, v) in buf.iter_mut().enumerate() {
            *v += amp * (2.0 * PI * hz * i as f32 / SAMPLE_RATE as f32).sin();
        }
    }
    /// A deterministic 64-bit xorshift64* white-noise frame.
    ///
    /// 64-bit, not 32-bit: a 32-bit xorshift's outputs from a small seed are
    /// visibly correlated with each other, and its *normalised autocorrelation*
    /// measures about 0.32 rather than the 0.16 the extreme-value distribution
    /// predicts for white noise. That is enough to make `voicing_null` look wrong
    /// in a test that is supposed to be checking something else entirely.
    fn frame_of(bits: u64, ms: i64) -> Vec<f32> {
        let n = (SAMPLE_RATE as i64 * ms / 1000) as usize;
        let mut s = bits | 1;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let v = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
            out.push(((v >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0);
        }
        out
    }

    /// A speech-like signal: a harmonic comb with a drifting fundamental, a
    /// drifting formant, and a 4-8 Hz amplitude envelope.
    ///
    /// Every feature the detector uses has to be *earned* by the test signal, or
    /// the test measures the generator rather than the detector: a plain sine is
    /// a *steady* source and is rejected by the steady-source veto, and an
    /// unmodulated comb has no formant structure and therefore the wrong
    /// rolloff. This one has both.
    fn voice_like(len: usize, seed: u64, f0: f32) -> Vec<f32> {
        let mut r = seed | 1;
        let mut next = || {
            r ^= r >> 12;
            r ^= r << 25;
            r ^= r >> 27;
            let v = r.wrapping_mul(0x2545_F491_4F6C_DD1D);
            ((v >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        };
        let f1 = 500.0 + next().abs() * 400.0;
        let mut out = vec![0f32; len];
        for (i, slot) in out.iter_mut().enumerate() {
            let t = i as f32 / SAMPLE_RATE as f32;
            let env = 0.4 + 0.6 * (0.5 + 0.5 * (2.0 * std::f32::consts::PI * 6.0 * t).sin());
            let mut s = 0f32;
            for h in 1..=12 {
                let fh = f0 * h as f32;
                if fh > 7_600.0 {
                    break;
                }
                let d = (fh - f1) / (f0 * 3.0 + 300.0);
                s += (1.0 / h as f32)
                    * (1.0 / (1.0 + d * d))
                    * (2.0 * std::f32::consts::PI * fh * t).sin();
            }
            *slot = 0.3 * env * s;
        }
        out
    }

    /// Slice a continuous signal into `WINDOW_SAMPLES` frames at the detector's
    /// own hop, exactly as `EnergyVad::analyse` does.
    fn slice_frames(signal: &[f32]) -> Vec<Vec<f32>> {
        if signal.len() <= WINDOW_SAMPLES {
            return vec![signal.to_vec()];
        }
        let n = 1 + (signal.len() - WINDOW_SAMPLES) / HOP_SAMPLES;
        (0..n)
            .map(|f| {
                let s = f * HOP_SAMPLES;
                signal[s..(s + WINDOW_SAMPLES).min(signal.len())].to_vec()
            })
            .collect()
    }

    /// Run a signal through a fresh extractor and return every frame's features.
    ///
    /// **Multi-frame is not optional here.** The extractor holds a per-bin noise
    /// estimate seeded from the *first* frame, so a one-frame test subtracts the
    /// frame from itself: the result is a scaled copy of the input, every shape
    /// feature is unchanged, and the test measures nothing. A steady tone
    /// additionally has `flux == FLUX_UNDEFINED` on its only frame, so the
    /// steady-source veto cannot fire. Feeding a real slice of a real signal
    /// fixes both, and it is what the production path does.
    fn extract(frames: &[Vec<f32>], floor_db: f32) -> Vec<FrameFeatures> {
        let mut e = FeatureExtractor::new(FeatureWeights::default());
        let mut out = Vec::new();
        for f in frames {
            e.set_floor(floor_db);
            out.push(e.push(f));
        }
        out
    }

    /// Extract from a continuous signal, discarding the first `warm` frames so
    /// the noise tracker and the flux measure have both converged.
    fn extract_signal(signal: &[f32], floor_db: f32, warm: usize) -> Vec<FrameFeatures> {
        let all = extract(&slice_frames(signal), floor_db);
        if all.len() <= warm {
            all
        } else {
            all[warm..].to_vec()
        }
    }
    /// The `q`-quantile of a slice, for assertions on a distribution.
    ///
    /// Nearest-rank rather than interpolated: the assertions only need "is the
    /// 95th percentile of this above or below that", and an interpolated value
    /// between two order statistics is a number no frame in the signal had.
    #[must_use]
    fn percentile_of(v: &[f32], q: f64) -> f64 {
        if v.is_empty() {
            return f64::NAN;
        }
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let i = ((q.clamp(0.0, 1.0) * (s.len() - 1) as f64).round()) as usize;
        s[i.min(s.len() - 1)] as f64
    }

    /// A continuous 1 kHz tone, `secs` long.
    /// A continuous 1 kHz tone, `secs` long.
    fn tone_signal(secs: f64, hz: f32, amp: f32) -> Vec<f32> {
        let n = (secs * SAMPLE_RATE as f64) as usize;
        (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * hz * i as f32 / SAMPLE_RATE as f32).sin())
            .collect()
    }

    /// The q-quantile of a slice, for assertions on a distribution.
    fn noise_signal(secs: f64, seed: u64, amp: f32) -> Vec<f32> {
        let n = (secs * SAMPLE_RATE as f64) as usize;
        let step = WINDOW_SAMPLES;
        let blocks: Vec<Vec<f32>> = (0..n / step + 2)
            .map(|i| frame_of(seed.wrapping_add(i as u64 * 0x9E37_79B9), 25))
            .collect();
        let mut out = Vec::with_capacity(n);
        let mut i = 0;
        while out.len() < n {
            out.extend_from_slice(&blocks[i / step]);
            i += step;
        }
        out.truncate(n);
        for v in &mut out {
            *v *= amp;
        }
        out
    }

    #[test]
    fn a_steady_tone_has_almost_no_flux_and_noise_has_a_lot() {
        let t = extract_signal(&tone_signal(0.6, 1000.0, 0.3), -60.0, 10);
        let nz = extract_signal(&noise_signal(0.6, 7, 0.05), -60.0, 10);
        let tone_flux = t[t.len() - 1].flux;
        let noise_flux = nz[nz.len() - 1].flux;
        assert!(
            tone_flux < 0.05,
            "a steady tone must be numerically steady, got {tone_flux} dB"
        );
        assert!(
            noise_flux > 1.5,
            "white noise must move, got {noise_flux} dB"
        );
    }

    #[test]
    fn white_noise_is_flat_low_tonality_and_unvoiced_while_speech_shaped_audio_is_not() {
        let nf = extract_signal(&noise_signal(0.6, 11, 0.05), -60.0, 10);
        let vf = extract_signal(&voice_like(9_600, 21, 120.0), -60.0, 10);
        let (nf, vf) = (nf[nf.len() - 1], vf[vf.len() - 1]);

        // Wiener's flatness of a *Gaussian* spectrum is about 0.37 by Jensen's
        // inequality, not 1.0, and the calibration table in
        // `tests/calibration.rs` measures 0.52-0.63 for the generator used here.
        assert!(nf.flatness > 0.3, "white noise flatness {}", nf.flatness);
        // `peak / median` over 257 bins. For *independent* exponentially
        // distributed bins the extreme-value theory puts this near 4-6, but a
        // windowed FFT's bins are strongly correlated, which drags the median
        // down relative to the peak: measured on this generator it is about 21.
        // The analytic number is not the target, so the assertion is a wide
        // band plus the contrast against a voice, which is what the feature is
        // for.
        assert!(
            (12.0..32.0).contains(&nf.tonality),
            "white noise tonality {} is outside the measured band",
            nf.tonality
        );
        assert!(
            nf.tonality * 5.0 < vf.tonality,
            "white noise tonality {} vs voice {}",
            nf.tonality,
            vf.tonality
        );
        assert!(nf.voicing < 0.25, "noise voicing {}", nf.voicing);
        // The 4-8 kHz band is half of a flat 0-8 kHz spectrum, and a little less
        // once the 0-300 Hz band is in the denominator, so white noise reads
        // 0.41-0.45 here rather than 0.5.
        assert!(nf.hf_ratio > 0.35, "noise hf_ratio {}", nf.hf_ratio);

        assert!(vf.flatness < 0.2, "voiced flatness {}", vf.flatness);
        assert!(vf.tonality > 15.0, "voiced tonality {}", vf.tonality);
        assert!(vf.voicing > 0.4, "voiced voicing {}", vf.voicing);
        assert!(vf.hf_ratio < 0.35, "voiced hf_ratio {}", vf.hf_ratio);
    }

    #[test]
    fn the_score_rejects_white_noise_that_is_loud_above_the_floor() {
        // The headline failure mode: `rms > threshold` calls this speech.
        let w = FeatureWeights::default();
        let f = extract_signal(&noise_signal(0.8, 3, 0.1), -40.0, 10);
        let scored: Vec<f32> = f.iter().map(|x| frame_score(x, &w)).collect();
        let p95 = percentile_of(&scored, 0.95);
        assert!(
            p95 < 0.2,
            "loud white noise scored {p95} at p95; the audibility term alone must not carry it"
        );
    }

    #[test]
    fn the_score_rejects_a_steady_tone_even_when_it_is_loud() {
        // The flux veto, end to end. Without it these frames score high: they
        // are peaky, non-flat, low-ZCR-in-band and well above the floor.
        let w = FeatureWeights::default();
        let f = extract_signal(&tone_signal(0.8, 1000.0, 0.3), -40.0, 10);
        let last = f[f.len() - 1];
        let ungated = audibility_term(last.snr_db, &w) * shape_term(&last, &w);
        assert!(
            ungated > 0.5,
            "the veto only means something if the ungated score is high; got {ungated}"
        );
        let gated = frame_score(&last, &w);
        assert!(
            gated < 0.3,
            "a steady 1 kHz tone scored {gated} after the veto"
        );
    }

    #[test]
    fn the_veto_does_not_fire_on_a_quiet_steady_frame() {
        // A distant hum sits near the floor. Penalising it twice would push a
        // genuinely quiet speech frame under the threshold in a very stable
        // room, so the veto requires a high SNR.
        let w = FeatureWeights::default();
        let mut hum = vec![0f32; 9_600];
        for (i, v) in hum.iter_mut().enumerate() {
            *v = 0.01 * (2.0 * std::f32::consts::PI * 50.0 * i as f32 / SAMPLE_RATE as f32).sin();
        }
        let f = extract_signal(&hum, -40.0, 10);
        assert_eq!(steady_source_penalty(&f[f.len() - 1], &w), 1.0);
    }

    #[test]
    fn the_veto_does_not_fire_on_loud_moving_speech() {
        let w = FeatureWeights::default();
        let f = extract_signal(&voice_like(9_600, 22, 130.0), -50.0, 10);
        let last = f[f.len() - 1];
        assert!(last.flux > w.flux_min_db, "flux was {}", last.flux);
        assert_eq!(steady_source_penalty(&last, &w), 1.0);
        let s = frame_score(&last, &w);
        assert!(s > 0.4, "score {s}");
    }

    #[test]
    fn silence_scores_zero_everywhere() {
        let w = FeatureWeights::default();
        let f = extract(&slice_frames(&vec![0f32; 9_600]), -120.0);
        assert!(f[0].db < -100.0, "silence db {}", f[0].db);
        assert_eq!(frame_score(&f[f.len() - 1], &w), 0.0);
    }

    #[test]
    fn db_is_corrected_for_the_window_gain() {
        // A 0.3-amplitude sine has mean square 0.09/2 = 0.045, i.e. -13.5 dB.
        // Two corrections are needed to get there and each is worth exactly
        // 3.01 dB if omitted: the Hann window's own loss, and the fact that the
        // power sum runs over the 257 *non-negative* bins, which is half the
        // spectrum for a real signal. Omitting either lands on -16.5.
        let f = extract_signal(&tone_signal(0.6, 1000.0, 0.3), -120.0, 10);
        let db = f[f.len() - 1].db;
        assert!((db + 13.5).abs() < 0.2, "db was {db}, expected about -13.5");
    }

    #[test]
    fn rms_amplitude_doubles_are_six_db() {
        let q = extract_signal(&tone_signal(0.6, 800.0, 0.1), -120.0, 10);
        let l = extract_signal(&tone_signal(0.6, 800.0, 0.2), -120.0, 10);
        let d = l[l.len() - 1].db - q[q.len() - 1].db;
        assert!((d - 6.0206).abs() < 0.05, "6 dB expected, got {d} dB");
    }
    #[test]
    fn lf_ratio_separates_a_hum_from_a_treble_hiss() {
        let mut hum = vec![0f32; WINDOW_SAMPLES];
        tone(&mut hum, 50.0, 0.2);
        tone(&mut hum, 100.0, 0.1);
        let hiss = frame_of(21, 25);
        let f = extract(&[hum, hiss], -120.0);
        // A 100 Hz tone is not all low-frequency: its 25 ms window holds 2.5 cycles, so`n        // Hann leakage puts real energy in the 200-500 Hz bins and lf_ratio reads`n        // 0.87, not 0.99. The separation is the point; the exact figure is not.`n        assert!(f[0].lf_ratio > 0.8, "hum lf_ratio {}", f[0].lf_ratio);
        assert!(f[0].hf_ratio < 0.01, "hum hf_ratio {}", f[0].hf_ratio);
        assert!(f[1].lf_ratio < 0.05, "hiss lf_ratio {}", f[1].lf_ratio);
        assert!(f[1].hf_ratio > 0.5, "hiss hf_ratio {}", f[1].hf_ratio);
    }

    #[test]
    fn rolloff_falls_as_energy_moves_up() {
        let mut low = vec![0f32; WINDOW_SAMPLES];
        tone(&mut low, 400.0, 0.3);
        let mut high = vec![0f32; WINDOW_SAMPLES];
        tone(&mut high, 5000.0, 0.3);
        let f = extract(&[low, high], -120.0);
        assert!(
            f[0].rolloff_hz < 800.0,
            "low tone rolloff {}",
            f[0].rolloff_hz
        );
        assert!(
            f[1].rolloff_hz > 4000.0,
            "high tone rolloff {}",
            f[1].rolloff_hz
        );
    }

    #[test]
    fn autocorrelation_peak_finds_the_period_of_a_150hz_tone() {
        let mut f = vec![0f32; WINDOW_SAMPLES];
        tone(&mut f, 150.0, 0.4);
        let r = normalised_autocorrelation_peak(&f, 60.0, 400.0);
        assert!(
            r > 0.9,
            "a clean periodic frame should correlate near 1, got {r}"
        );
    }

    #[test]
    fn autocorrelation_peak_is_bounded_by_one() {
        // Cauchy-Schwarz. The obvious transcription of the normalisation
        // divides by the wrong pair of energies, is not bounded at all, and the
        // symptom is a `voicing` of 6.4 for a signal that correlates at 0.98 --
        // which makes the voicing term saturate and quietly stop discriminating.
        let mut rng = 12345u32;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            (rng as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for hz in [60.0f32, 75.0, 120.0, 200.0, 333.0, 400.0] {
            let f: Vec<f32> = (0..WINDOW_SAMPLES)
                .map(|i| {
                    let t = i as f32 / SAMPLE_RATE as f32;
                    (2.0 * std::f32::consts::PI * hz * t).sin() + 0.4 * next()
                })
                .collect();
            let r = normalised_autocorrelation_peak(&f, 60.0, 400.0);
            assert!(
                (-1.0..=1.0).contains(&r),
                "{hz} Hz produced {r}, outside Cauchy-Schwarz"
            );
        }
    }

    #[test]
    fn autocorrelation_peak_ignores_a_dc_offset() {
        // A biased signal correlates perfectly with itself at every lag. If the
        // mean is not removed first, "is this periodic?" is answered "yes" for
        // anything with a bias. Noise plus a bias is the control, because a
        // periodic test signal would correlate anyway and the test would pass
        // for the wrong reason.
        let mut f = frame_of(1, 25);
        for v in &mut f {
            *v += 0.5;
        }
        let r = normalised_autocorrelation_peak(&f, 60.0, 400.0);
        assert!(r < 0.2, "a DC offset read as voicing: {r}");
    }

    #[test]
    fn autocorrelation_peak_is_zero_for_silence_and_degenerate_input() {
        assert_eq!(normalised_autocorrelation_peak(&[], 60.0, 400.0), 0.0);
        assert_eq!(
            normalised_autocorrelation_peak(&vec![0f32; WINDOW_SAMPLES], 60.0, 400.0),
            0.0
        );
        // A 4-sample input cannot have a lag in the 60-400 Hz range.
        assert_eq!(
            normalised_autocorrelation_peak(&[1.0, 2.0, 3.0, 4.0], 60.0, 400.0),
            0.0
        );
        // An inverted range is a caller bug, not a panic.
        assert_eq!(
            normalised_autocorrelation_peak(&frame_of(1, 25), 400.0, 60.0),
            0.0
        );
    }

    #[test]
    fn all_features_are_finite_for_every_fixture() {
        // NaN anywhere in a feature poisons the score for the rest of the run,
        // because the floor, the hysteresis state and the segment builder all
        // consume it. The one way to be sure is to assert finiteness on the
        // pathological inputs.
        let cases: Vec<Vec<f32>> = vec![
            vec![0.0; WINDOW_SAMPLES],
            vec![1.0; WINDOW_SAMPLES],
            vec![-1.0; WINDOW_SAMPLES],
            frame_of(5, 25),
            {
                let mut v = vec![0f32; WINDOW_SAMPLES];
                for (i, s) in v.iter_mut().enumerate() {
                    *s = if i % 3 == 0 { 1e-30 } else { 0.0 };
                }
                v
            },
        ];
        for c in cases {
            let f = extract(&[c], -120.0);
            let g = f[0];
            assert!(g.db.is_finite(), "db {:?}", g.db);
            assert!(g.flatness.is_finite(), "flatness {:?}", g.flatness);
            assert!(g.tonality.is_finite(), "tonality {:?}", g.tonality);
            assert!(g.voicing.is_finite(), "voicing {:?}", g.voicing);
            assert!(g.hf_ratio.is_finite(), "hf_ratio {:?}", g.hf_ratio);
            assert!(g.lf_ratio.is_finite(), "lf_ratio {:?}", g.lf_ratio);
            assert!(g.rolloff_hz.is_finite(), "rolloff {:?}", g.rolloff_hz);
            assert!(g.zcr.is_finite(), "zcr {:?}", g.zcr);
            assert!(g.flux.is_finite(), "flux {:?}", g.flux);
            assert!((0.0..=1.0).contains(&g.flatness.clamp(0.0, 1.0)));
        }
    }

    #[test]
    fn ms_to_frames_rounds_up_so_a_stated_minimum_is_never_smaller_than_advertised() {
        assert_eq!(ms_to_frames_ceil(0), 0);
        assert_eq!(ms_to_frames_ceil(-5), 0);
        assert_eq!(ms_to_frames_ceil(10), 1);
        assert_eq!(ms_to_frames_ceil(200), 20);
        // 25 ms at a 10 ms hop is 2.5 frames. Truncating would give a 20 ms
        // floor from a stated 25 ms.
        assert_eq!(ms_to_frames_ceil(25), 3);
        assert_eq!(ms_to_frames_ceil(1), 1);
    }

    #[test]
    fn frame_times_are_half_open_and_derived_from_the_hop() {
        assert_eq!(HOP_SAMPLES, 160);
        assert_eq!(frames_to_ms(1), 10);
        assert_eq!(frame_start_ms(0, 5_000), 5_000);
        assert_eq!(frame_end_ms(0, 5_000), 5_010);
        // Rebasing: a buffer that starts at 5 minutes must not produce times
        // near zero.
        assert_eq!(frame_start_ms(300, 300_000), 303_000);
    }
}

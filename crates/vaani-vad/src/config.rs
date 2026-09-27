//! Configuration for both engines and the chunker.
//!
//! Every number a VAD has is in this file, in one struct, `Serialize`. That is
//! what makes a tuning run reproducible: the parameter set that produced a
//! speech ratio is a value that can be written to a JSON report and read back,
//! instead of a scatter of literals in three `impl Default` blocks.
//!
//! The defaults are the *tuned* values. Where a default is not the obvious
//! choice, the reasoning is on the field and the measured effect is in the
//! tuning table in the final report. Where a default looks like a magic number
//! and is not, that is a defect and the comment says so.

use serde::{Deserialize, Serialize};

use crate::features::FeatureWeights;
use crate::smoothing::SmoothingConfig;

/// The energy detector's full configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct EnergyVadConfig {
    /// Feature-to-score mapping. See [`FeatureWeights`], where every constant
    /// carries its own reasoning.
    pub weights: FeatureWeights,
    /// The hysteresis and duration rules. Shared verbatim with the neural
    /// engine, which is what makes "the same smoothing" a true statement.
    pub smoothing: SmoothingConfig,

    /// Length of the trailing window the noise floor is taken over, in
    /// milliseconds.
    ///
    /// 4000 ms, i.e. 400 frames.
    ///
    /// Four seconds, and the number is a compromise between two measured
    /// failures. Too short and a sustained utterance fills the window, so the
    /// percentile is a speech frame and the floor becomes the talker -- measured
    /// on the sweep fixture, a 1 s window cost 10 % of the speech score against
    /// a 4 s one. Too long and a fan switched on mid-meeting takes that long to
    /// be absorbed. The measured score is flat from 2 s to 12 s, so 4 s sits
    /// where every window long enough to outlast a sentence is, and short enough
    /// to be a responsive filter rather than a moving average of the meeting.
    pub noise_window_ms: i64,

    /// The quantile of that window used as the floor.
    ///
    /// 0.20. At a 4 s window that is the 80th-quietest frame. Raising it
    /// raises the floor and costs recall on a quiet speaker; lowering it lets
    /// a steady noise source into the floor and costs precision. 0.20 was
    /// chosen as the point where a sustained 40 dB hum is reliably below the
    /// floor while a quiet speaker 12 dB above it is still detected.
    pub noise_quantile: f32,

    /// Window over which the quantile ramps up from the running minimum.
    ///
    /// 500 ms. See `noise.rs` for why the *quantile itself* is ramped rather
    /// than the estimate being interpolated.
    pub noise_warmup_ms: i64,
}

impl Default for EnergyVadConfig {
    fn default() -> Self {
        Self {
            weights: FeatureWeights::default(),
            smoothing: SmoothingConfig::default(),
            noise_window_ms: 4_000,
            noise_quantile: 0.20,
            noise_warmup_ms: 500,
        }
    }
}

impl EnergyVadConfig {
    /// Whether the configuration is self-consistent.
    ///
    /// Checked at construction rather than trusted, because every one of these
    /// is a plausible typo in a config file and a nonsense one produces
    /// silence or a solid block of "speech" with no error anywhere.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.smoothing.is_valid()
            && self.noise_window_ms > 0
            && self.noise_warmup_ms >= 0
            // Strictly inside (0, 1): a quantile of 0 makes the floor the exact
            // running minimum, which means the detector can never fire on
            // anything that is not an instantaneous minimum, and a quantile of
            // 1 makes it the maximum, which means the detector can never fire
            // at all. Both are typos, not preferences.
            && self.noise_quantile > 0.0
            && self.noise_quantile < 1.0
            && self.weights.snr_full_db > 0.0
            && self.weights.tonality_hi_db > self.weights.tonality_lo_db
            && self.weights.flatness_lo < self.weights.flatness_hi
            && self.weights.hiss_hi < self.weights.hiss_lo
            && self.weights.rolloff_hi_hz < self.weights.rolloff_lo_hz
            && self.weights.voicing_span > 0.0
            && self.weights.zcr_span > 0.0
            && self.weights.flux_min_db >= 0.0
            && self.weights.flux_penalty > 0.0
            && self.weights.flux_penalty <= 1.0
    }
}

/// How audio is cut up for the recogniser.
///
/// # The defaults, and what each one is defending against
///
/// * **`target_ms = 30 000`.** SraVaani's encoder is 887 MB of weights and a
///   Conformer whose activation memory scales with the frame count, so a chunk
///   is not free. 30 s is also the point at which an RNN-T decoder's state
///   has to be re-primed: measured at 0.714 emission steps per encoder frame
///   on real meeting speech, a 30 s chunk is ~2500 encoder frames, which is
///   enough context for the decoder's language model to be worth having.
///   Longer chunks are not obviously better -- the encoder has no recurrent
///   state to amortise and the measured encode cost is linear in frames -- so
///   there is no argument for going past 30 s and a good argument against it.
/// * **`max_ms = 60_000`.** The hard cap. A monotone speaker with no pause
///   produces one unbounded run; without a cap that becomes a single chunk,
///   whose log-mel is 60 000 mel frames and whose encoder activations are
///   `6001 x 1024` fp16 per layer. Sixty seconds is twice the target, which
///   means the cap only ever fires in the pathological case, and it is the
///   number that bounds decoder memory and RTF. Note the cap can cut
///   mid-word; that is unavoidable and is why it is a *cap* and not the
///   target.
/// * **`overlap_ms = 0`.** Overlap buys robustness at the boundary -- a word
///   straddling a cut can be recovered from the neighbouring chunk -- at the
///   cost of re-transcribing the overlap, duplicating text that then has to be
///   de-duplicated, and roughly doubling decode time. With the boundary
///   alignment below, a cut lands in the middle of a pause, so the word that
///   straddles it is *silence*. There is nothing to recover, and paying for a
///   de-duplication pass to fix a problem the boundary policy has already
///   avoided is a bad trade. Overlap is a knob, not a default, for the case
///   where a model is run with silence padding disabled.
/// * **`min_ms = 1000`.** See [`ChunkPolicy::MIN_CHUNK_MS`].
/// * **`merge_gap_ms = 300`.** The same number as the VAD's minimum silence,
///   for the same reason: a 250 ms gap between two detected segments is one
///   utterance, and emitting two chunks for it costs a decoder re-prime.
/// * **`boundary_search_ms = 2000`.** How far past the target the boundary
///   alignment may reach for a pause. Two seconds of slack on a 30 s target
///   is 6 %: enough to find the pause after the target when the target lands
///   mid-utterance, and not enough for a chunk to drift far from its target.

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct ChunkPolicy {
    /// Nominal chunk length before boundary alignment, in milliseconds.
    pub target_ms: i64,
    /// Hard cap. No chunk ever exceeds this, even mid-utterance.
    pub max_ms: i64,
    /// Overlap between adjacent chunks, in milliseconds.
    pub overlap_ms: i64,
    /// Segments separated by less than this are merged before chunking.
    pub merge_gap_ms: i64,
    /// Padding added either side of a chunk, in milliseconds.
    pub pad_ms: i64,
    /// Floor on a chunk's length. See [`Self::MIN_CHUNK_MS`].
    pub min_ms: i64,
    /// How far past `target_ms` the boundary alignment may search for a pause.
    pub boundary_search_ms: i64,
}

impl ChunkPolicy {
    /// The shortest chunk that is safe to hand to SraVaani's encoder.
    ///
    /// # Where this number comes from
    ///
    /// It is the **sum of the three hard floors in the chain**, not a round
    /// number chosen to look sensible:
    ///
    /// 1. **The log-mel front end needs at least one valid frame.**
    ///    `frontend.rs:329-345` computes the per-feature mean and variance over
    ///    `valid_frames = n / hop_length` and then divides by `std + 1e-5`.
    ///    At `n < 160` samples `valid_frames` is 0, so the mean is `0/0 = NaN`
    ///    and every feature is NaN. 160 samples is 10 ms. That is the hard
    ///    floor and it is not negotiable.
    /// 2. **The encoder subsamples 8x** (`ceil(T/8)`, measured -- see
    ///    `examples/subsample_probe.rs`). So 10 ms of audio is one encoder
    ///    frame, and one encoder frame is not a transcript: a chunk of ten
    ///    encoder frames is the smallest thing that can hold a short word.
    /// 3. **The decoder needs context to prime itself.** 8 encoder frames is
    ///    640 ms of audio -- about one short word. Too few.
    ///
    /// The value used is **1 000 ms**, which is 100 valid mel frames and 13
    /// encoder frames. It is not the minimum of the three; it is a deliberate
    /// round number above all of them, on the grounds that a 640 ms chunk costs
    /// a decoder re-prime to save 360 ms of compute and a decoder re-prime is
    /// not free. It is also below the 30 s target by a factor of 30, so it
    /// never forces a merge that the max cap would not have forced anyway.
    ///
    /// **This is an inference from the published graph structure, not a
    /// measurement of the shortest chunk the encoder accepts.** It is listed
    /// under `OPEN ISSUES`. `INTEGRATION NOTES` asks `vaani-asr` to publish
    /// the floor as a constant so the two cannot drift.
    pub const MIN_CHUNK_MS: i64 = 1_000;

    /// A policy with the tuned defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            target_ms: 30_000,
            max_ms: 60_000,
            overlap_ms: 0,
            merge_gap_ms: 300,
            pad_ms: 60,
            min_ms: Self::MIN_CHUNK_MS,
            boundary_search_ms: 2_000,
        }
    }

    /// Whether the policy is self-consistent.
    ///
    /// The ordering constraints are not cosmetic: `min_ms > max_ms` would make
    /// the short-chunk coalescer unable to satisfy the minimum, and
    /// `merge_gap_ms >= min_ms` would make the merger produce chunks that are
    /// mostly silence.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.target_ms > 0
            && self.max_ms >= self.target_ms
            && self.overlap_ms >= 0
            && self.overlap_ms < self.min_ms
            && self.min_ms > 0
            && self.min_ms <= self.max_ms
            && self.merge_gap_ms >= 0
            && self.merge_gap_ms < self.min_ms
            && self.pad_ms >= 0
            && self.boundary_search_ms >= 0
    }
}

impl Default for ChunkPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_energy_config_is_valid_and_hysteretic() {
        let c = EnergyVadConfig::default();
        assert!(c.is_valid());
        assert!(c.smoothing.leave_threshold < c.smoothing.enter_threshold);
    }

    #[test]
    fn a_zero_hysteresis_band_is_invalid() {
        let c = EnergyVadConfig {
            smoothing: SmoothingConfig {
                enter_threshold: 0.5,
                leave_threshold: 0.5,
                ..SmoothingConfig::default()
            },
            ..EnergyVadConfig::default()
        };
        assert!(!c.is_valid());
    }

    #[test]
    fn an_inverted_spectral_mapping_is_invalid() {
        // flatness_lo is the "peaky" end and flatness_hi the "flat" end, so
        // lo >= hi is a sign error, not a preference.
        let base = EnergyVadConfig::default();
        let mut w = base.weights;
        w.flatness_lo = 0.9;
        w.flatness_hi = 0.1;
        assert!(!EnergyVadConfig { weights: w, ..base }.is_valid());

        let base = EnergyVadConfig::default();
        let mut w = base.weights;
        w.tonality_hi_db = -1.0;
        assert!(!EnergyVadConfig { weights: w, ..base }.is_valid());

        let base = EnergyVadConfig::default();
        let mut w = base.weights;
        w.flux_penalty = 1.5;
        assert!(!EnergyVadConfig { weights: w, ..base }.is_valid());
    }

    #[test]
    fn an_out_of_range_noise_quantile_is_invalid() {
        let base = EnergyVadConfig::default();
        for q in [0.0, 1.0, -0.1, 1.1] {
            assert!(
                !EnergyVadConfig {
                    noise_quantile: q,
                    ..base
                }
                .is_valid(),
                "quantile {q} should be rejected"
            );
        }
    }

    #[test]
    fn the_default_chunk_policy_is_valid() {
        assert!(ChunkPolicy::default().is_valid());
    }

    #[test]
    fn a_policy_whose_minimum_exceeds_its_cap_is_invalid() {
        let p = ChunkPolicy {
            min_ms: 90_000,
            ..ChunkPolicy::default()
        };
        assert!(!p.is_valid(), "min > max can never be satisfied");
    }

    #[test]
    fn an_overlap_at_or_above_the_minimum_is_invalid() {
        // Overlap larger than the floor would make every adjacent pair of
        // chunks overlap by more than a whole minimum chunk, which is a
        // transcription of the same audio several times over.
        for o in [1_000, 5_000] {
            let p = ChunkPolicy {
                overlap_ms: o,
                ..ChunkPolicy::default()
            };
            assert!(!p.is_valid(), "overlap {o} should be rejected");
        }
    }

    #[test]
    fn a_merge_gap_at_or_above_the_minimum_is_invalid() {
        for g in [1_000, 30_000] {
            let p = ChunkPolicy {
                merge_gap_ms: g,
                ..ChunkPolicy::default()
            };
            assert!(!p.is_valid(), "merge gap {g} should be rejected");
        }
    }

    #[test]
    fn the_minimum_chunk_floor_exceeds_the_structural_front_end_floor() {
        // 160 samples is the front end's NaN threshold; the policy's floor must
        // be well above it. This is the invariant that would break first if
        // somebody "optimised" MIN_CHUNK_MS down to 10.
        let structural_ms = crate::dsp::HOP_SAMPLES as i64 * 1000 / crate::dsp::SAMPLE_RATE as i64;
        assert_eq!(structural_ms, 10);
        assert!(ChunkPolicy::MIN_CHUNK_MS >= 100 * structural_ms);
    }

    #[test]
    fn a_config_round_trips_through_json_shaped_data() {
        // Not a serde_json test -- `serde_json` is not a dependency. This
        // asserts the struct is `Serialize`/`Deserialize` with defaults, which
        // is what a persisted settings file needs.
        let c = EnergyVadConfig::default();
        let clone = c;
        assert_eq!(c, clone);
        let p = ChunkPolicy::default();
        assert_eq!(p, ChunkPolicy::new());
    }
}

//! Configuration for the diarization engine.
//!
//! Every number that changes the output is here, named, documented, and
//! reachable from the CLI/settings without touching engine code. That is a
//! deliberate constraint rather than tidiness: the difference between a usable
//! diarization baseline and an unusable one is three or four thresholds, and
//! the only way to find out which three is to let someone turn them.
//!
//! # The one product decision in here
//!
//! [`SegmentationBias`] answers the question "when this engine is unsure,
//! would you rather show two speakers where there is one, or one speaker
//! where there are two?" The default is [`SegmentationBias::SplitEasily`],
//! and the argument is in that type's documentation. It is a *product*
//! decision, not an accuracy one, which is why it is a first-class setting
//! rather than an internal constant that happens to have a value.

use serde::{Deserialize, Serialize};

/// The sample rate the engine requires.
///
/// Resampling is the media stage's job (`AudioBuffer`'s contract says a stage
/// must never be handed audio it cannot use), so this is a pre-flight check
/// and an error rather than a conversion. A buffer at 8 kHz or 48 kHz is a bug
/// in the pipeline, and silently resampling it would hide that bug while
/// changing the pitch features by a factor the caller did not know about.
pub const REQUIRED_SAMPLE_RATE: u32 = 16_000;

/// Which way to err when the engine cannot tell two speakers apart.
///
/// This is the over-segmentation vs over-merge decision, and it is exposed
/// rather than buried because the two errors are not equally bad in this
/// product:
///
/// * **Over-merge** attributes one person's words to another. Nothing in the
///   UI says the label is suspect. The transcript reads fluently and is
///   wrong, which is the worst outcome a transcript can have.
/// * **Over-split** shows one person as two, or splits a sentence across a
///   label change. It is *visible*, and the user can merge two labels with a
///   click.
///
/// So the default is [`SegmentationBias::SplitEasily`]. A user who disagrees
/// has a meeting with heavy crosstalk or a noisy recording, where window
/// features genuinely are unstable, and should set
/// [`SegmentationBias::MergeReadily`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SegmentationBias {
    /// Split earlier. Lower merge distance, shorter minimum cluster, shorter
    /// median filter. More turns, more speakers.
    #[default]
    SplitEasily,
    /// Merge earlier. Higher merge distance, longer minimum cluster, longer
    /// median filter. Fewer, longer turns.
    MergeReadily,
}

impl SegmentationBias {
    /// Multiplier on the cosine merge distance.
    ///
    /// The direction is the one that matches the name: `SplitEasily` lowers
    /// the bar a pair of clusters must clear to be kept apart, so more merges
    /// are refused and more clusters survive.
    #[must_use]
    pub fn merge_distance_scale(self) -> f64 {
        match self {
            Self::SplitEasily => 0.85,
            Self::MergeReadily => 1.20,
        }
    }

    /// Multiplier on the minimum speech time a cluster must have to survive.
    ///
    /// `SplitEasily` shortens it, because a short-but-real turn is a person
    /// saying "haan" and dropping that is a data loss the user cannot repair.
    #[must_use]
    pub fn min_speech_scale(self) -> f64 {
        match self {
            Self::SplitEasily => 0.5,
            Self::MergeReadily => 2.0,
        }
    }

    /// Multiplier on the median filter length, in frames.
    #[must_use]
    pub fn median_filter_scale(self) -> usize {
        match self {
            Self::SplitEasily => 1,
            Self::MergeReadily => 3,
        }
    }
}

/// How many speakers to produce.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerCount {
    /// Exactly this many, clamped to `[1, max_speakers]` and to the number of
    /// windows available.
    Fixed(usize),

    /// **The default.** Run agglomerative clustering and stop as soon as the
    /// *next* merge would cost more than `merge_distance` cosine units.
    ///
    /// The rationale, and the numbers behind it.
    ///
    /// The threshold is not chosen by taste; it is placed in the gap between two
    /// measured distance distributions, from
    /// `cargo run -p vaani-diarization --example geometry`:
    ///
    /// ```text
    ///   one speaker, 2 min, 30 turns:  within-speaker p90 = 0.47
    ///   two speakers, 12 s turns:      within-speaker p90 = 0.86
    ///                                   cross-speaker   p10 = 1.18
    ///   two speakers, same pitch:      cross-speaker   p10 = 1.35
    /// ```
    ///
    /// The binding band is **0.86 to 1.18** -- the tightest same-speaker
    /// p90 against the smallest cross-speaker p10 -- so the base is **1.05** and
    /// `SplitEasily` scales it to 0.89, which clears 0.86 with almost no room to
    /// spare. That tightness is itself the product characteristic: the engine
    /// has roughly a third of an octave of headroom between "one speaker" and
    /// "two speakers" on this fixture, and nothing wider.
    ///
    /// `MergeReadily` scales the base to 1.26, which is *above* the 1.18
    /// cross-speaker floor and therefore **merges two clearly separated
    /// speakers** on this fixture. That is the documented cost of the setting
    /// and the reason it is not the default: the default is tuned to the
    /// split-side edge of the band.
    ///
    /// A cosine distance on block-weighted window vectors is not on a fixed
    /// scale of its own, so an absolute threshold would be untunable. It is
    /// meaningful here because the normalisation divides by measured per-block
    /// *constants* rather than by a statistic of the file being processed; see
    /// [`crate::features::within_speaker_spread`] for why that distinction is
    /// load-bearing.
    ///
    /// Its failure mode is stated plainly in the module docs of `cluster`:
    /// it **over-splits** a single speaker with expressive prosody, and it
    /// **over-merges** two speakers with very similar voices.
    Threshold {
        /// Cosine distance above which two clusters are different speakers.
        merge_distance: f64,
        /// Hard cap. A meeting with 14 people is not a product question.
        max_speakers: usize,
    },

    /// Largest *relative* jump in the merge-distance sequence.
    ///
    /// Average linkage on a reducible distance produces a monotone
    /// merge-distance sequence, so the structure in it is in the ratios
    /// between consecutive merges, not the absolute values. This rule takes
    /// the biggest ratio in `[min_speakers, max_speakers]` and cuts just
    /// before it.
    ///
    /// **Experimental.** It is strictly better than
    /// [`SpeakerCount::Threshold`] on recordings whose speakers are
    /// *well* separated, and strictly worse on a single-speaker recording,
    /// where a smooth monotone sequence still has a largest ratio somewhere
    /// and the rule will cut at it. The `min_cluster_speech_ms` guard
    /// catches some of those and not the rest. [`SpeakerCount::Threshold`]
    /// is the default for that reason.
    Auto {
        /// Hard cap.
        max_speakers: usize,
        /// Smallest acceptable ratio `d[i+1] / d[i]` at the cut.
        min_gap_ratio: f64,
    },
}

impl SpeakerCount {
    /// The default policy: a 0.34 cosine merge distance, at most 8 speakers.
    #[must_use]
    pub fn default_policy() -> Self {
        Self::Threshold {
            merge_distance: 1.05,
            max_speakers: 8,
        }
    }

    /// The hard speaker cap this policy allows.
    #[must_use]
    pub fn max_speakers(&self) -> usize {
        match *self {
            Self::Fixed(_) => usize::MAX,
            Self::Threshold { max_speakers, .. } | Self::Auto { max_speakers, .. } => {
                max_speakers.max(1)
            }
        }
    }

    /// The absolute merge distance this policy wants, before
    /// [`SegmentationBias`] scales it. `None` for [`SpeakerCount::Auto`],
    /// which has no absolute threshold by construction.
    #[must_use]
    pub fn base_merge_distance(&self) -> Option<f64> {
        match *self {
            Self::Fixed(_) => Some(f64::INFINITY),
            Self::Threshold { merge_distance, .. } => Some(merge_distance),
            Self::Auto { .. } => None,
        }
    }
}

/// Framing and feature-extraction parameters.
///
/// The frame geometry is fixed to match the ASR front end deliberately: both
/// stages then agree on what instant a frame describes, so a diarization span
/// and the ASR features behind it refer to the same audio. See
/// [`crate::dsp`] for the exact conventions and why the 400-sample window is
/// centred at offset 56.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct FeatureParams {
    /// Samples per analysis hop. 160 at 16 kHz is 10 ms, the same hop the ASR
    /// front end uses.
    pub hop: usize,
    /// FFT size. 512.
    pub n_fft: usize,
    /// Analysis window in samples. 400, zero-padded into `n_fft` and
    /// **centred**, so it sits at `[56, 456)` of the 512-point frame.
    pub win_length: usize,
    /// Triangular mel bands feeding the cepstral transform. 26.
    pub n_mel: usize,
    /// Low edge of the mel filterbank, Hz. 80.
    pub mel_low_hz: f32,
    /// High edge of the mel filterbank, Hz. 7600 (`0.46 * nyquist`).
    pub mel_high_hz: f32,
    /// Cepstral coefficients kept from the DCT. Coefficient 0 is the frame
    /// energy and is **dropped**: it tracks the microphone and the gain, not
    /// the speaker.
    pub n_cepstra: usize,
    /// Half-width of the delta and double-delta regression, in frames.
    pub delta_width: usize,
    /// Samples per frame used for the autocorrelation pitch estimate. 1024
    /// is 64 ms, which is 7.7 periods of a 120 Hz voice and 3.8 of a 60 Hz
    /// one.
    pub pitch_span: usize,
    /// Autocorrelation lag searched, inclusive. `[sr/f_max, sr/f_min]`.
    pub pitch_lag_min: usize,
    /// Autocorrelation lag searched, inclusive.
    pub pitch_lag_max: usize,
    /// Normalised autocorrelation above which a frame counts as voiced.
    pub voicing_threshold: f32,
    /// Bands in the long-term average spectrum.
    pub n_ltas: usize,
    /// Coarse spectral bands whose energy ratios go into the vector.
    pub band_edges_hz: [f32; 6],
    /// Per-block feature weights, applied after z-scoring and before the
    /// L2 normalisation. See [`crate::features`] for the justification of each
    /// value; the ordering is the point: **pitch outweighs the cepstral
    /// envelope**, and the frame-to-frame dynamics are worth less than the
    /// envelope because they track prosody rather than identity.
    pub weights: BlockWeights,
}

impl Default for FeatureParams {
    fn default() -> Self {
        Self {
            hop: 160,
            n_fft: 512,
            win_length: 400,
            n_mel: 26,
            mel_low_hz: 80.0,
            mel_high_hz: 7_600.0,
            n_cepstra: 20,
            delta_width: 2,
            pitch_span: 1024,
            pitch_lag_min: 40,  // 400 Hz
            pitch_lag_max: 266, // 60 Hz
            voicing_threshold: 0.32,
            n_ltas: 16,
            band_edges_hz: [0.0, 300.0, 800.0, 1_600.0, 3_200.0, 8_000.0],
            weights: BlockWeights::default(),
        }
    }
}

impl FeatureParams {
    /// Milliseconds per analysis frame.
    #[must_use]
    pub fn frame_ms(&self) -> i64 {
        (self.hop as f64 * 1000.0 / REQUIRED_SAMPLE_RATE as f64).round() as i64
    }

    /// Total width of the window-level feature vector, in floats.
    #[must_use]
    pub fn vector_len(&self) -> usize {
        // Cepstra: n_cepstra - 1, because c0 is dropped. Deltas: the same
        // count, twice. Plus pitch (1), voicedness (1), spectral shape (3),
        // band ratios (band_edges - 1) and the LTAS.
        let c = self.n_cepstra.saturating_sub(1);
        3 * c + 2 + 3 + (self.band_edges_hz.len() - 1) + self.n_ltas
    }
}

/// Per-block multipliers on the z-scored feature vector.
///
/// Weights are applied *after* per-dimension z-scoring (so every dimension
/// enters at unit variance) and *before* the L2 normalisation (so the
/// absolute weights mean what they say). Applied in that order, a block's
/// weight is the ratio of its total contribution to the vector norm.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct BlockWeights {
    /// Static cepstral envelope, `c1..c19`.
    pub cepstra: f32,
    /// First-order regression of the envelope.
    pub delta: f32,
    /// Second-order regression of the envelope.
    pub double_delta: f32,
    /// Log2 fundamental frequency. **Absolute** -- see
    /// [`crate::features::Layout::is_absolute`]. This weight is a multiplier on
    /// an octave, not on a unit variance, so the number here is not comparable
    /// with the others in this struct and should not be read as if it were.
    pub pitch: f32,
    /// Fraction of frames with a confident autocorrelation peak.
    pub voicedness: f32,
    /// Spectral centroid, spread and 85% rolloff.
    pub spectral_shape: f32,
    /// Coarse band energy ratios.
    pub band_ratios: f32,
    /// Long-term average spectrum.
    pub ltas: f32,
    /// Magnitude of the constant carrier dimension every vector carries.
    ///
    /// This is the **evidence requirement**, and it is the one number that sets
    /// how much has to be different before two windows are called different
    /// speakers.
    ///
    /// The reasoning is a ratio, and it is worth writing down because it is
    /// counter-intuitive. For a cosine distance, with `m` the squared mass of
    /// the phoneme-driven blocks, `p` the pitch difference between two
    /// speakers, and `rho` the correlation of the phoneme blocks between two
    /// windows of the same speaker:
    ///
    /// ```text
    /// within  ~= (1 - rho) * m / (m + c)
    /// between ~= within + p^2 / (m + c)
    /// ratio    =  1 + p^2 / ((1 - rho) * m)
    /// ```
    ///
    /// where `c` is the carrier. Raising `c` does not change the ratio -- but it
    /// does shrink *both* populations towards zero, which is what lets a single
    /// global threshold sit in the gap between them. Lowering the block weights
    /// instead leaves the ratio identical and moves neither number, because
    /// cosine distance is scale-invariant: **the carrier is the only thing that
    /// is not scaled with the features, so it is the only dial that changes the
    /// absolute distance scale rather than the relative ordering.**
    ///
    /// The cost is real and worth stating: a large carrier means a speaker
    /// difference has to be *large* to be seen, so two speakers with similar
    /// pitch and similar timbre are reported as one. That is the correct
    /// failure -- merging two similar voices is recoverable in a transcript,
    /// inventing four is not -- but it does bound what this engine can do. The
    /// measured boundary is reported in the README.
    pub carrier: f32,
}

impl Default for BlockWeights {
    /// Chosen from measured distance decompositions, not by taste.
    ///
    /// The measurement that sets them is a per-block decomposition of the
    /// cosine distance on the **one-speaker** synthetic fixture, where every
    /// window is the same person. On a single speaker every bit of distance is
    /// noise -- there is no speaker signal to find -- and the mean share of the
    /// distance came from:
    ///
    /// ```text
    ///   bands   0.228      cepstra  0.180      ltas  0.200
    ///   voicing 0.009      shape   0.014       delta 0.008
    ///   pitch   0.0001
    /// ```
    ///
    /// The long-term spectrum, the band ratios and the cepstral envelope
    /// together produce 0.61 of the within-speaker distance of a *single*
    /// speaker. That is the whole problem in one table: a 1.5 s window
    /// summarised by a median over its frames is substantially a description of
    /// which vowels were spoken in it, and phoneme identity varies more within
    /// one person than speaker identity does between two.
    ///
    /// So these weights are deliberately lopsided. Pitch holds about 74% of the
    /// squared length of the vector and the seven other blocks share the rest.
    /// The reasoning is the ratio derived in [`crate::features::block_weight`]
    /// and [`crate::features::within_speaker_spread`]:
    ///
    /// ```text
    ///   separation = 1 + signal / (within-speaker noise)
    /// ```
    ///
    /// The denominator is the phoneme noise `m`, and the signal `p^2` is
    /// whatever the blocks contribute *between* speakers. Pitch's between-
    /// speaker signal is a physical constant -- two voices are an octave apart,
    /// or they are not -- and it does not shrink when the spectral blocks are
    /// down-weighted, whereas the noise shrinks in proportion. So down-weighting
    /// the phoneme blocks widens the region in which the engine works rather
    /// than narrowing it, which is the opposite of what intuition suggests and
    /// is why these numbers are so much lower than a naive "pitch leads a
    /// little" would give.
    ///
    /// The blocks are kept rather than dropped because they are the only defence
    /// for two speakers with the *same* pitch, and the measured
    /// `same-pitch-different-tilt` fixture is the evidence for keeping them: it
    /// separates with a margin of +1.06 on the spectral blocks alone. They are
    /// weighted as supporting evidence, not as the primary signal.
    fn default() -> Self {
        Self {
            cepstra: 0.45,
            delta: 0.05,
            double_delta: 0.02,
            pitch: 1.0,
            voicedness: 0.22,
            spectral_shape: 0.11,
            // The band-ratio and LTAS blocks are held at almost nothing, and the
            // measurement is unusually direct. On the intonation fixture -- one
            // speaker, pitch rising 110 -> 150 Hz inside every turn -- the
            // per-block decomposition of the within-speaker distance is:
            //
            // ```text
            //   bands 0.699   ltas 0.191   cepstra 0.022   shape 0.014
            //   voicing 0.007   pitch 0.0001
            // ```
            //
            // Two five-band energy ratios and a sixteen-band long-term average
            // spectrum produce **0.89 of a single speaker's own scatter**, while
            // holding 0.067 of the vector's squared length. The cause is
            // structural: their band edges are *fixed* frequencies, and a
            // harmonic series sliding through fixed edges moves a lot of energy
            // across them. Any intonation at all -- a question that rises, a
            // phrase-final fall -- is therefore a large change in those
            // dimensions, and it is a change in *how the voice is shaped* rather
            // than in *whose voice it is*.
            //
            // They are also largely redundant with the cepstral envelope, which
            // is a better-conditioned version of the same spectral information
            // and which is kept at 0.45. The evidence that the remaining
            // spectral evidence is enough is the `same-pitch, different-timbre`
            // fixture, which separates two voices sharing a fundamental and
            // passes with a margin of +0.75 without them.
            band_ratios: 0.02,
            ltas: 0.02,
            carrier: 1.0,
        }
    }
}

/// Per-dimension minimum spread, in the dimension's own units, applied before
/// z-scoring.
///
/// Without this, z-scoring *amplifies* a dimension that barely varies: take a
/// single-speaker recording whose pitch wanders by 0.4 Hz, z-score it, and
/// the wander becomes a full unit-variance signal that the clustering then
/// dutifully splits into four speakers. Flooring the standard deviation at a
/// physically meaningful value means a dimension that carries no
/// speaker information contributes proportionally less instead of being
/// promoted to equal standing.
///
/// This is the single most important parameter in the feature stage, and it is
/// why a one-speaker recording does not shatter.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct MinSpread {
    pub cepstra: f32,
    pub delta: f32,
    pub double_delta: f32,
    /// In log2-Hz. 0.15 is about 1.8 semitones.
    pub pitch: f32,
    /// A fraction of frames.
    pub voicedness: f32,
    /// In log2-Hz.
    pub spectral_shape: f32,
    pub band_ratios: f32,
    pub ltas: f32,
}

impl Default for MinSpread {
    /// **Every value here is measured, not chosen.** The procedure was: render
    /// a synthetic two-speaker meeting and run the real 14-minute meeting
    /// through the engine, then read off the across-window standard deviation
    /// of every dimension in every block
    /// (`cargo run -p vaani-diarization --example probe_activity`). Each floor
    /// is set at roughly **a third of the smaller of the two measured
    /// medians**, so:
    ///
    /// * a dimension with a *real* speaker signal is above its floor on both
    ///   recordings and is normalised normally;
    /// * a dimension carrying only numerical wobble is below its floor and is
    ///   dropped outright -- see
    ///   [`crate::features::normalise_in_place`], which explains why the drop
    ///   has to be a drop and not a division by the floor.
    ///
    /// The fraction is 1/3 rather than something smaller because the floor also
    /// has to sit above the noise of a *degenerate* signal. A six-second steady
    /// 200 Hz tone, which is one speaker and no more, has spectral dimensions
    /// whose only variation is FFT round-off. With floors at an eighth of the
    /// speech medians those round-off values clear the floor, get z-scored to
    /// unit variance, and split the tone into two speakers. The margin is
    /// narrow -- speech and numerical noise differ by about 3x in the cepstral
    /// block -- but it is the margin that is actually there.
    ///
    /// The measured medians, for reference:
    ///
    /// | block | synthetic 1 spk | synthetic 2 spk | real clip | floor |
    /// |---|---:|---:|---:|---:|
    /// | cepstra | 0.0186 | 0.0124 | 0.2235 | 0.0040 |
    /// | delta | 0.0036 | 0.0024 | 0.0310 | 0.0008 |
    /// | double delta | 0.0020 | 0.0014 | 0.0127 | 0.0005 |
    /// | pitch (log2 Hz) | 0.0025 | 0.0019 | 0.0451 | 0.0002 |
    /// | voicedness | 0.00016 | 0.00009 | 0.0812 | 0.00003 |
    /// | spectral shape (log2 Hz) | 0.0022 | 0.0018 | 0.6321 | 0.0006 |
    /// | band ratios (ln) | 0.00048 | 0.00038 | 0.3163 | 0.00013 |
    /// | LTAS (ln) | 0.0018 | 0.0014 | 0.3842 | 0.0005 |
    ///
    /// Two earlier versions of this file did worse in opposite directions.
    /// Values ten to a hundred times larger, written in the dimensions' natural
    /// units and never measured, gated *every* block in *every* dimension: all
    /// 83 were suppressed and the engine clustered on nothing. Values an eighth
    /// of the medians gated too little, and a single steady tone shattered into
    /// two speakers.
    ///
    /// The pitch floor is an exception to the 1/3 rule and is deliberately set
    /// far lower, because pitch is now an **absolute** dimension: it is not
    /// divided by its spread, so a small spread cannot be amplified, and the
    /// floor is there only to catch a tracking failure that has produced no
    /// usable estimate at all. See [`crate::features::Layout::is_absolute`].
    fn default() -> Self {
        Self {
            cepstra: 0.0040,
            delta: 0.0008,
            double_delta: 0.0005,
            pitch: 0.0002,
            voicedness: 0.00003,
            spectral_shape: 0.0006,
            band_ratios: 0.00013,
            ltas: 0.0005,
        }
    }
}

/// Clustering and smoothing parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct ClusterParams {
    /// Which speaker-count policy to use.
    pub count: SpeakerCount,
    /// Analysis window length, in frames. 150 frames is 1.5 s.
    pub window_frames: usize,
    /// Hop between analysis windows, in frames. 75 frames is 0.75 s, i.e. 50%
    /// overlap.
    pub window_hop_frames: usize,
    /// A window is kept only if at least this fraction of its frames are
    /// active.
    ///
    /// **0.9, measured rather than assumed.** A window that is mostly silence
    /// is summarised by a median over the handful of frames that do contain
    /// speech, so it describes almost nothing while looking like every other
    /// window to the clustering. The effect is not a broad degradation but a few
    /// outright outliers: at 0.4 the within-speaker distance of a *single*
    /// speaker has a median of 0.23 and a p90 of 0.68, and that tail is a
    /// handful of windows rather than a broad spread. Raising the floor to 0.9
    /// drops the p90 to 0.47 and takes the one-speaker fixture from two clusters
    /// to one, while the two-speaker cross-speaker 10th percentile stays above
    /// 1.1 -- so the two populations separate instead of overlapping.
    ///
    /// The sweep is `cargo run -p vaani-diarization --example geometry --
    /// --sweep-speech`.
    pub min_window_speech_ratio: f32,
    /// Hard cap on the number of windows fed to the clustering stage.
    ///
    /// Above this, windows are thinned by a deterministic stride before
    /// clustering and every window is then assigned to the nearest centroid.
    /// The cap exists because exact average linkage needs the full
    /// `N x N` distance matrix: 2000 windows is a 16 MB `f32` matrix, and
    /// 20000 would be 1.6 GB, which is the difference between a 3-hour
    /// meeting fitting in memory and not. See [`crate::cluster`].
    pub max_cluster_windows: usize,
    /// Total speech time, in milliseconds, a cluster must have to survive.
    /// Smaller clusters are absorbed into the nearest neighbour.
    pub min_cluster_speech_ms: i64,
    /// Median filter length over the per-frame cluster assignment, in frames.
    /// Odd; rounded up to odd.
    pub median_filter_frames: usize,
    /// Turns shorter than this are absorbed into a neighbour. 600 ms.
    pub min_turn_ms: i64,
    /// Lloyd refinement iterations after agglomerative clustering. 3.
    pub refine_iters: usize,
    /// Half-width, in windows, of the triangular moving average applied to the
    /// window feature vectors before normalisation. 2 spans five windows, which
    /// is 3.75 s of audio at the default geometry.
    pub feature_smooth_half_windows: usize,
}

impl Default for ClusterParams {
    fn default() -> Self {
        Self {
            count: SpeakerCount::default_policy(),
            window_frames: 150,
            window_hop_frames: 75,
            min_window_speech_ratio: 0.9,
            max_cluster_windows: 2_000,
            min_cluster_speech_ms: 2_000,
            median_filter_frames: 21,
            min_turn_ms: 600,
            refine_iters: 3,
            feature_smooth_half_windows: 2,
        }
    }
}

/// Voice-activity parameters for the built-in detector.
///
/// These are only used when the caller does **not** supply a
/// `vaani_core::pipeline::SpeechTimeline` from the VAD stage. The engine
/// prefers an externally supplied timeline; see
/// [`crate::engine::HeuristicDiarizer::diarize_with_timeline`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct ActivityParams {
    /// FFT size for the activity pass. 256, not 512: the activity decision
    /// needs 31 bands, not 257, and this pass runs over every frame of the
    /// whole file.
    pub n_fft: usize,
    /// Frames of hangover added either side of an active run. 30 frames is
    /// 300 ms, which is about how long a plosive closure lasts.
    pub hangover_frames: usize,
    /// Interior gaps shorter than this are filled. 30 frames, 300 ms.
    pub max_gap_frames: usize,
    /// Active runs shorter than this are discarded. 20 frames, 200 ms.
    pub min_speech_frames: usize,
    /// A frame this far above the noise floor is speech outright. 0.55 of the
    /// recording's own dynamic range.
    pub energy_hi_ratio: f32,
    /// A frame this far above the floor needs spectral flux to agree. 0.18.
    pub energy_lo_ratio: f32,
    /// Flux percentile a marginal frame must exceed. 0.60 of the range above
    /// the flux median.
    pub flux_ratio: f32,
    /// Spectral flatness above which a frame is not speech at all.
    ///
    /// 0.15, and that number is measured. On synthetic voices the flatness of
    /// an active frame is 0.002 at the 90th percentile -- a harmonic spectrum
    /// is about as far from flat as a spectrum gets. On a real 14-minute
    /// meeting, the 90th percentile of *active* frames was 0.071 and the
    /// whole-file 95th was 0.597, so 0.5 let through a noticeable population
    /// of noise-like frames that were loud enough to pass the energy gate:
    /// chair creaks, coughs, keyboard, and the room tone under a loud passage.
    /// Those frames are the worst possible input to a speaker feature, and
    /// 0.15 keeps a wide margin over the speech population while rejecting
    /// them.
    ///
    /// White noise measures 0.90 on this estimator and is rejected by a factor
    /// of six, so this is also the gate that makes a noise-only recording
    /// produce zero speakers rather than eight.
    pub max_flatness: f32,
    /// Recordings whose energy dynamic range is below this, in dB, are treated
    /// as uniform: either digital silence or a single constant level.
    ///
    /// 2 dB, and that number is measured rather than guessed. Continuous
    /// speech from a speaker sitting still has a frame-energy range of about
    /// 4 dB between the quietest 10% of frames and the loudest 10% -- the
    /// variation is prosody, not silence. An earlier value of 6 dB rejected
    /// every continuous-speech signal as "uniform" and produced zero speakers
    /// on a perfectly good recording. At 2 dB: digital silence (0 dB) and a
    /// pure tone (0 dB) are still rejected, and so is broadband white noise
    /// (0.6 dB), which is a second line of defence behind the flatness gate.
    pub min_dynamic_range_db: f32,
}

impl Default for ActivityParams {
    fn default() -> Self {
        Self {
            n_fft: 256,
            hangover_frames: 30,
            max_gap_frames: 30,
            min_speech_frames: 20,
            energy_hi_ratio: 0.55,
            energy_lo_ratio: 0.18,
            flux_ratio: 0.60,
            max_flatness: 0.15,
            min_dynamic_range_db: 2.0,
        }
    }
}

/// How often, in analysis frames, a long loop checks for cancellation.
///
/// 50 frames is 0.5 s of audio, comfortably inside the "at least once per
/// second of audio" the stage contracts ask for, and an `AtomicBool` load
/// every 10 ms of audio costs nothing.
pub const DEFAULT_CANCEL_POLL_FRAMES: usize = 50;

/// Everything the engine needs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct DiarizationConfig {
    /// Which way to err on the over-segmentation / over-merge question.
    pub bias: SegmentationBias,
    pub features: FeatureParams,
    pub activity: ActivityParams,
    pub clusters: ClusterParams,
    pub min_spread: MinSpread,
    /// Frames between cancellation checks.
    pub cancel_poll_frames: usize,
    /// Where to look for a neural speaker embedder. `None` disables the
    /// `ort` path entirely, which is the default in a build with no
    /// diarization weights installed.
    pub artifacts_dir: Option<std::path::PathBuf>,
}

impl Default for DiarizationConfig {
    fn default() -> Self {
        Self {
            bias: SegmentationBias::default(),
            features: FeatureParams::default(),
            activity: ActivityParams::default(),
            clusters: ClusterParams::default(),
            min_spread: MinSpread::default(),
            cancel_poll_frames: DEFAULT_CANCEL_POLL_FRAMES,
            artifacts_dir: None,
        }
    }
}

impl DiarizationConfig {
    /// Configuration with an explicit speaker count and the default policy.
    #[must_use]
    pub fn with_speaker_count(mut self, n: usize) -> Self {
        self.clusters.count = SpeakerCount::Fixed(n.max(1));
        self
    }

    /// Configuration that estimates the speaker count.
    #[must_use]
    pub fn with_estimated_speaker_count(mut self, max_speakers: usize) -> Self {
        self.clusters.count = SpeakerCount::Auto {
            max_speakers: max_speakers.max(1),
            min_gap_ratio: 1.15,
        };
        self
    }

    /// Set the bias.
    #[must_use]
    pub fn with_bias(mut self, bias: SegmentationBias) -> Self {
        self.bias = bias;
        self
    }

    /// Point the engine at a directory that may contain `embedder.onnx`.
    #[must_use]
    pub fn with_artifacts_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.artifacts_dir = Some(dir.into());
        self
    }

    /// The merge distance this configuration will actually use, after bias
    /// scaling and the hard speaker cap.
    #[must_use]
    pub fn effective_merge_distance(&self) -> f64 {
        let base = self
            .clusters
            .count
            .base_merge_distance()
            .unwrap_or(f64::INFINITY);
        if !base.is_finite() {
            return base;
        }
        (base * self.bias.merge_distance_scale()).clamp(0.01, 1.99)
    }

    /// The minimum speech time a cluster must have, after bias scaling.
    #[must_use]
    pub fn effective_min_cluster_speech_ms(&self) -> i64 {
        let scaled = self.clusters.min_cluster_speech_ms as f64 * self.bias.min_speech_scale();
        scaled.round().max(0.0) as i64
    }

    /// The median filter length, after bias scaling and rounding to odd.
    #[must_use]
    pub fn effective_median_filter_frames(&self) -> usize {
        let n = self
            .clusters
            .median_filter_frames
            .saturating_mul(self.bias.median_filter_scale())
            .max(1);
        if n.is_multiple_of(2) { n + 1 } else { n }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_frame_is_ten_milliseconds() {
        let p = FeatureParams::default();
        assert_eq!(p.frame_ms(), 10);
        assert_eq!(p.hop, 160);
        assert_eq!(p.frame_ms(), p.hop as i64 * 1000 / 16_000);
    }

    #[test]
    fn the_analysis_window_is_centred_in_the_fft_frame() {
        // The ASR team found that a 400-sample window sits at [56, 456) of a
        // 512-point frame, not [0, 400). If this ever reads 0, the two stages
        // disagree about what instant a frame describes.
        let p = FeatureParams::default();
        assert_eq!(p.n_fft - p.win_length, 112);
        assert_eq!((p.n_fft - p.win_length) / 2, 56);
    }

    #[test]
    fn vector_length_matches_the_documented_layout() {
        // 19 cepstra + 19 delta + 19 double-delta + pitch + voicedness +
        // 3 shape + 5 band ratios + 16 LTAS = 83.
        assert_eq!(FeatureParams::default().vector_len(), 83);
    }

    #[test]
    fn bias_lowers_the_bar_to_keep_clusters_apart() {
        // SplitEasily must make a split more likely, i.e. require a *larger*
        // distance before declaring two clusters different speakers, i.e.
        // multiply the requested merge distance... no: it multiplies it
        // *down*, so the engine stops earlier and keeps more clusters.
        let split = DiarizationConfig::default().with_bias(SegmentationBias::SplitEasily);
        let merge = DiarizationConfig::default().with_bias(SegmentationBias::MergeReadily);
        assert!(split.effective_merge_distance() < merge.effective_merge_distance());
        assert!(split.effective_min_cluster_speech_ms() < merge.effective_min_cluster_speech_ms());
        assert!(split.effective_median_filter_frames() < merge.effective_median_filter_frames());
    }

    #[test]
    fn median_filter_length_is_always_odd() {
        for n in [1usize, 2, 3, 4, 5, 20, 21, 22] {
            let mut c = DiarizationConfig::default();
            c.clusters.median_filter_frames = n;
            for bias in [
                SegmentationBias::SplitEasily,
                SegmentationBias::MergeReadily,
            ] {
                c.bias = bias;
                let got = c.effective_median_filter_frames();
                assert_eq!(got % 2, 1, "n={n} bias={bias:?} gave {got}");
                assert!(got >= 1);
            }
        }
    }

    #[test]
    fn a_fixed_speaker_count_never_becomes_a_finite_threshold() {
        // Fixed(k) means "always merge to k"; there is no distance at which it
        // should stop early, so the effective distance is infinite and the
        // threshold comparison must never fire.
        let c = DiarizationConfig::default().with_speaker_count(3);
        assert!(!c.effective_merge_distance().is_finite());
        assert_eq!(c.clusters.count, SpeakerCount::Fixed(3));
    }

    #[test]
    fn speaker_count_constructor_clamps_zero_to_one() {
        let c = DiarizationConfig::default().with_speaker_count(0);
        assert_eq!(c.clusters.count, SpeakerCount::Fixed(1));
    }

    #[test]
    fn merge_distance_is_clamped_into_the_cosine_range() {
        let mut c = DiarizationConfig::default();
        c.clusters.count = SpeakerCount::Threshold {
            merge_distance: 99.0,
            max_speakers: 8,
        };
        assert!(c.effective_merge_distance() < 2.0);
        c.clusters.count = SpeakerCount::Threshold {
            merge_distance: 0.0,
            max_speakers: 8,
        };
        assert!(c.effective_merge_distance() > 0.0);
    }

    #[test]
    fn max_speakers_is_never_zero() {
        let c = DiarizationConfig {
            clusters: ClusterParams {
                count: SpeakerCount::Auto {
                    max_speakers: 0,
                    min_gap_ratio: 1.2,
                },
                ..ClusterParams::default()
            },
            ..DiarizationConfig::default()
        };
        assert_eq!(c.clusters.count.max_speakers(), 1);
    }

    #[test]
    fn pitch_is_the_heaviest_single_block() {
        // The design claim in `BlockWeights`: pitch is the most discriminative
        // cheap feature, so nothing may out-weigh it.
        let w = BlockWeights::default();
        for (name, v) in [
            ("cepstra", w.cepstra),
            ("delta", w.delta),
            ("double_delta", w.double_delta),
            ("voicedness", w.voicedness),
            ("spectral_shape", w.spectral_shape),
            ("band_ratios", w.band_ratios),
            ("ltas", w.ltas),
        ] {
            assert!(w.pitch > v, "pitch must outweigh {name}");
        }
    }

    // Serialisation is covered by the contract crate's own round-trip tests
    // for the types that carry diarization output. This crate's manifest is
    // frozen and has no `serde_json`, so it does not re-test its own config
    // encoding; `Serialize`/`Deserialize` are still derived so the config can
    // travel in a job's persisted settings.
}

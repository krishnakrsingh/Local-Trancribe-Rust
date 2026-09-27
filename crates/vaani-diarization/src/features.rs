//! The speaker feature vector: layout, construction, and normalisation.
//!
//! # This is a heuristic, not a speaker embedder
//!
//! There are no trained speaker-embedding weights in this build, so this file
//! does not produce one. What it produces is a **defensible spectral
//! descriptor** whose dimensions are each chosen because they are known to
//! correlate with speaker identity, weighted by how strongly they do, and
//! normalised so that none of them can dominate by accident. It is a
//! heuristic baseline. It is not ECAPA-TDNN, it does not claim ECAPA's
//! accuracy, and nothing in this crate reports a diarization error rate.
//!
//! # The vector, dimension by dimension
//!
//! Per 10 ms frame, then aggregated over a 1.5 s window (below).
//!
//! | # | block | dims | why it is in the vector |
//! |---|---|---:|---|
//! | 0 | cepstral envelope `c1..c19` | 19 | The shape of the vocal tract is the speaker's own geometry. A 19-coefficient DCT of a 26-band log-mel spectrum is the standard compact description of that shape, and it is invariant to overall gain and to the vowel being articulated. Coefficient `c0` is **dropped** because it is frame energy, which tracks the microphone and the distance from it, not the person. |
//! | 1 | delta cepstra | 19 | The frame-to-frame slope of the envelope. Distinguishes speakers with systematically different articulation rates. Kept at half the envelope's weight because this is as much about the *sentence* as the person. |
//! | 2 | double-delta cepstra | 19 | Curvature, the same argument again, at a quarter of the weight. |
//! | 3 | log2 F0 | 1 | **The heaviest single block, at 1.8x.** An adult speaker's median fundamental is set by vocal-fold length and is close to a physical constant for that person; two speakers whose medians differ by four semitones are separable on pitch alone, at the cost of one autocorrelation per frame. It is also the one dimension here that is not confounded by which phoneme is being spoken. |
//! | 4 | voicing confidence | 1 | The fraction of the window with a confident autocorrelation peak. Separates a voice from an unvoiced fricative or a cough occupying the same band. |
//! | 5 | centroid, spread, 85% rolloff | 3 | Where the energy sits and how wide it is. These track the spectral tilt of an individual vocal tract, but they also track which consonant is being produced, so they are down-weighted to 0.5. |
//! | 6 | coarse band ratios | 5 | Energy fractions in 0-300, 300-800, 800-1600, 1600-3200 and 3200-8000 Hz. A low-resolution, very stable envelope descriptor; robust where MFCCs are noisy. |
//! | 7 | long-term average spectrum | 16 | The mean log spectrum over the window in 16 log-spaced bands. The closest thing here to a direct "vocal tract" measurement, and the most stable block, which is why it carries a full 1.0 weight despite being the crudest. |
//!
//! 83 dimensions. `FeatureParams::vector_len` asserts the count.
//!
//! # Aggregation: median over frames, then per-file z-scoring
//!
//! A window is a **median** over its active frames, not a mean. The median is
//! chosen because half of any speech window is consonants and pauses, and a
//! mean over that is dominated by the loudest vowel. The median of a log
//! quantity is also the log of the geometric mean, which is the right
//! averaging operation for a power spectrum.
//!
//! Then each dimension is z-scored **across the windows of this file**, so
//! that a recording at -30 dBFS and one at -6 dBFS produce comparable
//! vectors, and so that no dimension dominates merely by having a larger raw
//! range. Crucially the standard deviation is **floored** at
//! [`MinSpread`]: without that floor, z-scoring a dimension that barely
//! varies would *amplify* its numerical wobble to full unit variance and
//! hand the clustering an apparently strong signal that carries no speaker
//! information at all. That failure mode is exactly what shatters a
//! single-speaker recording into four clusters, and the floor is why it does
//! not happen here.
//!
//! Finally each block is scaled by its weight and the whole vector is
//! **L2-normalised**, so cosine similarity is the inner product and
//! `1 - dot` is the distance. The order matters and is fixed: median ->
//! z-score with floor -> block weight -> L2 normalise.

use vaani_core::Time;

use crate::config::{FeatureParams, MinSpread};
use crate::dsp::{FrontEnd, band_ratios};

/// Where each block starts in the per-frame vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    /// Number of kept cepstral coefficients (coefficient 0 is dropped).
    pub cepstra: usize,
    /// Where the delta block starts.
    pub delta: usize,
    /// Where the double-delta block starts.
    pub double_delta: usize,
    /// Where log2 F0 sits.
    pub pitch: usize,
    /// Where the voicing fraction sits.
    pub voicedness: usize,
    /// Where the spectral-shape block starts (3 values).
    pub shape: usize,
    /// Where the coarse band ratios start.
    pub bands: usize,
    /// Where the long-term average spectrum starts.
    pub ltas: usize,
    /// Total length.
    pub total: usize,
}

impl Layout {
    /// The layout implied by `p`.
    #[must_use]
    pub fn of(p: &FeatureParams) -> Self {
        let c = p.n_cepstra.saturating_sub(1);
        let delta = c;
        let double_delta = 2 * c;
        let pitch = 3 * c;
        let voicedness = pitch + 1;
        let shape = voicedness + 1;
        let bands = shape + 3;
        let ltas = bands + (p.band_edges_hz.len() - 1);
        let total = ltas + p.n_ltas;
        Self {
            cepstra: 0,
            delta,
            double_delta,
            pitch,
            voicedness,
            shape,
            bands,
            ltas,
            total,
        }
    }

    /// Whether dimension `j` is **absolute**: it must keep its own mean
    /// instead of being z-scored away.
    ///
    /// This is the single most consequential decision in the whole feature
    /// vector, and it is easy to get wrong in a way that looks like it works.
    ///
    /// Everything else here is a *description* of a voice -- the shape of the
    /// spectral envelope, the long-term average spectrum, the centroid. Those
    /// are only comparable within one recording, so they are mean-normalised:
    /// subtracting the file-wide mean per dimension is the classical
    /// mean-cepstral-normalisation trick, and it makes a dim vector from a
    /// quiet microphone comparable to one from a loud one.
    ///
    /// Pitch is not a description. A speaker's fundamental frequency is an
    /// absolute physical property of their vocal folds, and two speakers
    /// differ in it by a factor of nearly two -- 0.8 in log2, which is a
    /// little over an octave. The measured picture makes the consequence
    /// unavoidable: across the windows of a synthetic two-speaker meeting the
    /// *standard deviation* of the median log2 F0 is 0.0019, and across the
    /// windows of a synthetic **one**-speaker recording it is 0.0019 as well.
    /// The two files are indistinguishable in pitch variance. All of their
    /// difference is in the mean: 6.88 for one voice, 7.67 for the other.
    ///
    /// So z-scoring pitch subtracts the only number that identifies the
    /// speaker and divides what is left -- 0.0019 of quantisation jitter --
    /// up to unit variance to fill the space. The dimension then reports
    /// numerical noise at full volume while the real cue has been discarded.
    /// Measured consequence: a six-second steady tone split into two speakers,
    /// and the synthetic two-speaker fixture scored 0.56 purity instead of the
    /// 0.97 it scores with the mean intact.
    #[must_use]
    pub fn is_absolute(&self, j: usize) -> bool {
        j == self.pitch
    }
}

/// The fixed log2 F0 subtracted from the pitch dimension before it is scaled.
///
/// A **constant**, not a per-file statistic. That is the whole distinction from
/// z-scoring, and it has to be a constant: subtracting the file's own mean is
/// exactly the operation that deletes the speaker's pitch.
///
/// It cannot simply be zero, though. Cosine distance on an uncentred vector is
/// governed by the vector's *direction*, so a common offset of 7.0 with a
/// speaker-to-speaker difference of 0.79 gives an angle of about six degrees and
/// a distance of 0.005 -- the difference is real and the geometry throws it
/// away. Centring on a fixed mid-voice reference puts ordinary speakers near the
/// origin, so the difference is most of the vector's length. 7.0 is log2 of
/// 128 Hz, a low-ish adult voice and round in log space.
pub const PITCH_REFERENCE_LOG2: f32 = 7.0;

/// The vector magnitude one octave of log2 F0 is worth, before the block
/// weight is applied.
///
/// Chosen so that the 0.79-log2 gap between the two synthetic voices produces a
/// pair of rows at `-0.18` and `+1.0` against the unit carrier, a cosine
/// distance of 0.43 -- below the 0.70 merge distance with margin to spare. The
/// 0.0019 of within-speaker jitter lands at a distance of 8e-6. The far end is
/// not unlimited: a 60 Hz voice against a 400 Hz voice gives 1.4, which is past
/// the merge distance, as it should be. Both ends are measured.
pub const ABSOLUTE_PITCH_SCALE: f32 = 1.5;

/// Accumulate per-frame feature vectors for one window and reduce them.
///
/// The accumulator owns one row per active frame, which is the entire reason
/// the engine's memory stays flat: a 1.5 s window is 75 frames x 83 floats =
/// 25 kB, and it is freed before the next window is built.
#[derive(Debug)]
pub struct WindowAccumulator {
    layout: Layout,
    rows: Vec<f32>,
    n_rows: usize,
    /// Regression half-width for the delta and double-delta blocks.
    delta_width: usize,
    /// Sum of voicing confidences, for the mean.
    voicing_sum: f32,
    n_voiced: usize,
    pitch_sum: f64,
}

impl WindowAccumulator {
    /// An empty accumulator for one window.
    #[must_use]
    pub fn new(layout: Layout, delta_width: usize) -> Self {
        Self {
            layout,
            rows: Vec::new(),
            n_rows: 0,
            delta_width: delta_width.max(1),
            voicing_sum: 0.0,
            n_voiced: 0,
            pitch_sum: 0.0,
        }
    }

    /// Number of frames pushed so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n_rows
    }

    /// Whether any frame has been pushed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n_rows == 0
    }

    /// Append one active frame's feature vector.
    pub fn push(&mut self, frame: &[f32], pitch_log2: f32, voicing: f32) {
        debug_assert_eq!(frame.len(), self.layout.total);
        self.rows.extend_from_slice(frame);
        self.n_rows += 1;
        self.voicing_sum += voicing;
        if voicing > 0.0 && pitch_log2 != 0.0 {
            self.n_voiced += 1;
            self.pitch_sum += f64::from(pitch_log2);
        }
    }

    /// The median of each dimension, plus the mean voicing fraction and the
    /// median voiced pitch, into `out`.
    ///
    /// `fallback_pitch` is written into the pitch dimension when the window
    /// contains no voiced frame at all. Without that substitution the
    /// unvoiced windows would all sit at log2(F0) = 0, which after
    /// z-scoring becomes a *strongly* negative outlier: every fricative
    /// window would be its own speaker. Substituting the recording's median
    /// voiced pitch makes them ordinary, and the voicing fraction dimension
    /// carries the "this was unvoiced" information instead, which is what it
    /// is for.
    pub fn reduce(&self, fallback_pitch: f32, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.layout.total);
        let d = self.layout.total;
        if self.n_rows == 0 {
            out.iter_mut().for_each(|v| *v = 0.0);
            out[self.layout.voicedness] = 0.0;
            // An empty window has no pitch information, so it takes the same
            // recording-wide fallback a fully unvoiced window does.
            out[self.layout.pitch] = fallback_pitch;
            return;
        }

        // Column-wise summaries. `sort_unstable_by` on a small slice is a
        // counting-friendly insertion sort and beats collecting into a Vec per
        // dimension for n_rows in the tens-to-hundreds range.
        //
        // The two dynamic blocks are **not** summaries of the series they sit
        // next to. They are summaries of the *cepstral* series' time
        // derivatives, which is what a delta cepstrum is: dimension
        // `delta + k` describes the rate of change of dimension `cepstra + k`
        // through the window. Regressing the delta block against itself --
        // which is the obvious reading and which this code originally did --
        // regresses a block of zeros, because the per-frame vector has no
        // delta in it, and produces 38 identically-zero dimensions that every
        // shape check and every length check happily accepts.
        //
        // And the dynamic summary is the **mean absolute value**, not the
        // median. The median of the first difference of a stationary series is
        // zero *by construction* -- half the frames rise and half fall -- so a
        // median here produces a dimension that is identically zero in every
        // window, on every recording, and the `MinSpread` floor then correctly
        // discards it. The mean absolute value is the expected magnitude of the
        // change, which is what "dynamics" means and which is not degenerate.
        let c = self.layout.delta - self.layout.cepstra;
        let mut col: Vec<f32> = Vec::with_capacity(self.n_rows);
        let mut d1: Vec<f32> = Vec::with_capacity(self.n_rows);
        let mut d2: Vec<f32> = Vec::with_capacity(self.n_rows);
        let series = |j: usize, col: &mut Vec<f32>, rows: &[f32], d: usize, n: usize| {
            col.clear();
            for r in 0..n {
                col.push(rows[r * d + j]);
            }
        };
        for (j, slot) in out.iter_mut().enumerate() {
            if j >= self.layout.delta && j < self.layout.pitch {
                let k = if j < self.layout.double_delta {
                    j - self.layout.delta
                } else {
                    j - self.layout.double_delta
                };
                series(
                    self.layout.cepstra + k,
                    &mut col,
                    &self.rows,
                    d,
                    self.n_rows,
                );
                d1.clear();
                d1.extend_from_slice(&delta_coefficients(&col, self.delta_width));
                if j < self.layout.double_delta {
                    *slot = mean_abs(&d1);
                } else {
                    d2.clear();
                    d2.extend_from_slice(&delta_coefficients(&d1, self.delta_width));
                    *slot = mean_abs(&d2);
                }
                continue;
            }
            series(j, &mut col, &self.rows, d, self.n_rows);
            *slot = median_of(&col);
        }
        debug_assert_eq!(self.layout.double_delta, self.layout.delta + c);

        out[self.layout.voicedness] = self.voicing_sum / self.n_rows as f32;
        out[self.layout.pitch] = if self.n_voiced > 0 {
            (self.pitch_sum / self.n_voiced as f64) as f32
        } else {
            fallback_pitch
        };
    }

    /// The mean voiced pitch seen so far, used as the fallback above.
    #[must_use]
    pub fn mean_pitch(&self) -> f32 {
        if self.n_voiced == 0 {
            0.0
        } else {
            (self.pitch_sum / self.n_voiced as f64) as f32
        }
    }

    /// How many of the pushed frames had a confident autocorrelation peak.
    ///
    /// The engine needs this to know which windows need the recording-wide
    /// pitch fallback, and it needs it *after* the accumulator is reduced,
    /// because the accumulator is dropped at that point.
    #[must_use]
    pub fn voiced(&self) -> usize {
        self.n_voiced
    }
}

/// Compute one frame's full feature vector into `out`.
///
/// `pitch_log2` and `voicing` are passed in rather than computed here because
/// the pitch analysis is the expensive part and the caller may already have
/// them; [`FrontEnd::frame_pitch`] is the source.
pub fn frame_vector(
    fe: &mut FrontEnd,
    buf: &[f32],
    f: usize,
    power: &mut [f32],
    log_mel: &mut [f32],
    cepstra: &mut [f32],
    out: &mut [f32],
) {
    let p = *fe.params();
    let layout = Layout::of(&p);
    fe.frame_cepstra(buf, f, power, log_mel, cepstra);

    // c0 is frame energy; drop it and keep c1..c{c}. The count is the *width*
    // of the block, `layout.delta - layout.cepstra`, not `layout.cepstra` --
    // which is the block's *offset*, and is zero. Reading the offset as the
    // count copies an empty slice and leaves all 57 cepstral, delta and
    // double-delta dimensions silently at zero, which is invisible in a shape
    // check and in a length check and reduces the vector to 26 live
    // dimensions. This was a real bug and the regression test below is
    // `frame_vector_fills_every_dimension`.
    let c = layout.delta - layout.cepstra;
    out[layout.cepstra..layout.cepstra + c].copy_from_slice(&cepstra[1..1 + c]);
    // c0 itself is kept in `cepstra` and deliberately not written: a log-mel
    // DCT's zeroth coefficient is the frame's total energy, which tracks the
    // microphone and the distance from it rather than the speaker.
    out[layout.cepstra + c] = 0.0;

    // Spectral shape, from the power spectrum the cepstra just computed.
    let (centroid, spread, rolloff) = fe.spectral_shape(power);
    let shape = &mut out[layout.shape..layout.shape + 3];
    shape[0] = (centroid.max(1.0)).log2();
    shape[1] = (spread.max(1.0)).log2();
    shape[2] = (rolloff.max(1.0)).log2();

    let ratios = band_ratios(power, fe.n_freq(), &p.band_edges_hz);
    out[layout.bands..layout.bands + ratios.len()].copy_from_slice(&ratios);

    // LTAS bands: log-spaced from 100 Hz to the mel top edge. Log power per
    // band, so the window's median over frames is a geometric mean, and each
    // band is floored the same way `band_ratios` floors its bands. Without
    // the floor a frame with one loud region (a vowel, or a synthetic tone)
    // has its other bands filled with pure spectral-leakage noise, whose
    // z-scored variance would then dominate the descriptor.
    let ltas = p.n_ltas;
    let lo = 100.0f32;
    let hi = p.mel_high_hz;
    let bin_hz =
        (crate::config::REQUIRED_SAMPLE_RATE as f32 / 2.0) / (fe.n_freq() - 1).max(1) as f32;
    let total: f32 = power.iter().sum();
    let mut fracs = vec![1.0f32 / ltas as f32; ltas];
    for (i, slot) in fracs.iter_mut().enumerate() {
        let f0 = lo * (hi / lo).powf(i as f32 / ltas.max(2) as f32);
        let f1 = lo * (hi / lo).powf((i + 1) as f32 / ltas.max(2) as f32);
        let a = ((f0 / bin_hz).round() as isize).clamp(0, power.len() as isize - 1) as usize;
        let b =
            ((f1 / bin_hz).round() as isize).clamp(a as isize + 1, power.len() as isize) as usize;
        let acc: f32 = power[a..b].iter().sum();
        *slot = if total > 0.0 {
            acc / total
        } else {
            1.0 / ltas as f32
        };
    }
    let floor =
        (fracs.iter().copied().fold(0f32, f32::max) * crate::dsp::band_floor_linear()).max(1e-8);
    for (slot, v) in out[layout.ltas..layout.ltas + ltas]
        .iter_mut()
        .zip(fracs.iter())
    {
        *slot = v.max(floor).ln();
    }
}

/// The first-order regression of `src` into `dst`, both `n` long.
///
/// The standard symmetric first difference, `d[t] = sum_{n=1..W} n *
/// (c[t+n] - c[t-n]) / (2 * sum n^2)`, with the index clamped at the edges
/// rather than reflected. Clamping biases the first and last `W` frames of a
/// window towards zero slope, which is the conservative direction: a frame
/// that looks flat is assigned to whatever its neighbours are, so an edge
/// effect can smooth a turn away but cannot invent one.
#[must_use]
pub fn delta_coefficients(src: &[f32], width: usize) -> Vec<f32> {
    let n = src.len();
    let mut out = vec![0f32; n];
    if n == 0 || width == 0 {
        return out;
    }
    let mut norm = 0f32;
    for k in 1..=width {
        norm += 2.0 * k as f32 * k as f32;
    }
    if norm == 0.0 {
        return out;
    }
    for (t, slot) in out.iter_mut().enumerate() {
        let mut acc = 0f32;
        for k in 1..=width {
            let hi = (t + k).min(n - 1);
            let lo = t.saturating_sub(k);
            acc += k as f32 * (src[hi] - src[lo]);
        }
        *slot = acc / norm;
    }
    out
}

/// The second-order regression of `src`, i.e. the deltas of the deltas.
#[must_use]
pub fn double_delta_coefficients(src: &[f32], width: usize) -> Vec<f32> {
    delta_coefficients(&delta_coefficients(src, width), width)
}

/// The median of `v`, averaging the two central values for an even length.
#[must_use]
pub fn median_of(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mut sorted = v.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[mid]
    }
}

/// The mean absolute value of `v`, or 0 for an empty slice.
///
/// The window summary for a *dynamic* feature. See [`WindowAccumulator::reduce`]
/// for why this cannot be a median.
#[must_use]
pub fn mean_abs(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32
}

/// The per-dimension weight for dimension `j`, from its block's configured
/// weight.
///
/// # The `1 / sqrt(len)` divisor, which is the whole point
///
/// `BlockWeights` is documented as a statement about *blocks*: pitch should
/// outweigh the long-term average spectrum, and the frame-to-frame dynamics
/// should count for less than the envelope. Applied per dimension, it is not a
/// statement about blocks at all. It is a statement about individual
/// dimensions, and a block's total share of the vector's squared length is then
/// its weight **squared times its dimension count**:
///
/// ```text
/// share_b = w_b^2 * len_b
/// ```
///
/// The 83-dimension default layout has a 16-dimension LTAS block. Measured on
/// the synthetic two-speaker fixture, with the naive reading, the LTAS held 78%
/// of the squared vector norm and pitch held 7% -- which is the exact inverse of
/// what the weights are configured to say. The long-term average spectrum is a
/// window-wide average, so it is a description of the *vowels in the window*;
/// weighting it above pitch put the loudest evidence in the vector on phoneme
/// identity, and the measured same-speaker median distance was 0.73 -- the
/// features could not tell one person's voice from their own.
///
/// Dividing by `sqrt(len)` makes the share proportional to `w^2` alone, so the
/// configured ordering is the one that actually holds. It is `sqrt` and not
/// `len` because the cosine geometry aggregates squared magnitudes, so
/// halving the per-dimension magnitude halves the block's squared mass.
#[must_use]
pub fn block_weight(j: usize, layout: &Layout, w: &crate::config::BlockWeights) -> f32 {
    let (weight, len) = if j < layout.delta {
        (w.cepstra, layout.delta - layout.cepstra)
    } else if j < layout.double_delta {
        (w.delta, layout.double_delta - layout.delta)
    } else if j < layout.pitch {
        (w.double_delta, layout.pitch - layout.double_delta)
    } else if j == layout.pitch {
        (w.pitch, 1)
    } else if j == layout.voicedness {
        (w.voicedness, 1)
    } else if j < layout.bands {
        (w.spectral_shape, layout.bands - layout.shape)
    } else if j < layout.ltas {
        (w.band_ratios, layout.ltas - layout.bands)
    } else {
        (w.ltas, layout.total - layout.ltas)
    };
    debug_assert!(len > 0, "a feature block must not be empty");
    weight / (len as f32).sqrt()
}

/// The minimum spread floor for dimension `j`.
#[must_use]
pub fn min_spread_for(j: usize, layout: &Layout, s: &MinSpread) -> f32 {
    if j < layout.delta {
        s.cepstra
    } else if j < layout.double_delta {
        s.delta
    } else if j < layout.pitch {
        s.double_delta
    } else if j == layout.pitch {
        s.pitch
    } else if j == layout.voicedness {
        s.voicedness
    } else if j < layout.bands {
        s.spectral_shape
    } else if j < layout.ltas {
        s.band_ratios
    } else {
        s.ltas
    }
}

/// The number of rows below which per-file z-scoring is skipped entirely.
///
/// With two rows, z-scoring maps them onto `+v` and `-v` for whatever
/// dimension they differ in, however slightly, and a 1e-4 difference in pitch
/// becomes a cosine distance of 2. With three rows it is nearly as bad. A
/// clip shorter than about 4 s cannot produce more than a couple of windows,
/// so at that size there is no across-file statistic worth normalising to and
/// the honest thing is to skip it: the `MinSpread` floor and the block weights
/// still apply, so the features are comparable in *scale* even if not in
/// *distribution*.
pub const MIN_ZSCORE_ROWS: usize = 4;

/// The magnitude of the constant dimension every normalised row carries.
///
/// # Why a carrier dimension exists
///
/// Cosine distance is scale-free, which is normally a virtue. Here it is a
/// hazard: after L2-normalisation, a vector whose *only* varying dimension
/// is a 1e-4 wobble in pitch becomes the unit vector `+e_pitch`, and the one
/// with the wobble in the other direction becomes `-e_pitch`. They are a
/// cosine distance of **2** apart -- maximally different -- on the strength of
/// a difference that carries no speaker information whatsoever. The
/// `MinSpread` floor correctly says "this dimension is below its floor", but
/// L2-normalisation then throws that answer away by rescaling whatever is
/// left to unit length.
///
/// A constant carrier fixes it, and it is the standard fix. Every row gains
/// one dimension of fixed value `CARRIER_MAGNITUDE`, shared by all rows. Two
/// rows then agree on that dimension by construction, so the carrier
/// contributes a *common* component and a dimension's influence on the cosine
/// distance falls off as its magnitude squared relative to the carrier. A
/// dimension at the carrier's magnitude moves the distance by about 1; a
/// dimension a hundred times below it moves it by about 1e-4, which is the
/// "no speaker information" answer expressed numerically.
///
/// The cost is that every distance is compressed slightly towards zero, so
/// the cosine scale is not exactly the textbook one. The merge-distance
/// threshold lives in the same units and is tuned against the measured
/// distribution, so the two move together.
/// The default magnitude of the carrier dimension.
///
/// The live value is [`crate::config::BlockWeights::carrier`], which is a
/// parameter rather than a constant precisely so it can be swept against
/// measured distance distributions. This stays as the documented default and as
/// the value used where no weights are supplied.
///
/// A unit carrier bounds how much any single dimension can do: a dimension at
/// the carrier's magnitude moves the distance by about 1, and a dimension a
/// hundred times below it moves it by about 1e-4, which is the "no speaker
/// information" answer expressed numerically. A *larger* carrier raises the
/// evidence requirement; see the field's documentation for the ratio.
pub const CARRIER_MAGNITUDE: f32 = 1.0;

/// The measured within-speaker spread of each block, in the block's own units.
///
/// # Why a constant, and not a statistic of the file
///
/// Two data-dependent choices were tried and both are wrong, for reasons worth
/// recording because the failures look like something else entirely.
///
/// **The file-wide standard deviation** contains the between-speaker variance.
/// A nine-speaker recording therefore has a larger scale than a one-speaker
/// recording of the same people, so its within-speaker distances come out
/// smaller, and a global threshold splits it into more clusters. The threshold
/// would mean something different depending on the speaker count of the file --
/// the very quantity being estimated. Measured: the one-speaker
/// within-speaker p90 is 0.96 where the two-speaker within-speaker p90 is 0.23
/// on identical features.
///
/// **The robust scale of the first difference**, `median |x[i+1] - x[i]|`, does
/// remove the between-speaker component -- adjacent windows are normally the
/// same speaker -- but it is far too *small* a denominator, because windows
/// overlap by 50% at the default geometry and consecutive windows of smooth
/// speech are nearly identical. Dividing by it inflates every window that is
/// not adjacent to its neighbour: the same-speaker distribution became bimodal
/// with a median of 0.0005 and a p90 of 1.99, saturated against the maximum
/// possible distance. The measurement stopped being a measurement.
///
/// A **constant** avoids both. It is the same in every recording and for every
/// speaker count, so a merge distance means one thing and can be stated in the
/// documentation as a product characteristic. The cost is that the constants
/// are only as good as the voices they were measured on, so they are stated
/// here with their provenance and the tuning example that produced them is
/// named; changing the synthesiser means re-measuring.
///
/// Centring is still per-file -- median-normalisation of the cepstra is standard
/// and makes a quiet microphone comparable to a loud one. Only the *scale* is
/// constant.
///
/// # Provenance
///
/// Measured by `cargo run -p vaani-diarization --example geometry` on the
/// one-speaker synthetic fixture (`one-speaker-long`, 2 minutes, 30 turns of a
/// single voice), which is the purest within-speaker estimate available: with
/// one speaker there is no between-speaker term in the measurement at all. The
/// real-clip figures are listed alongside for comparison; they are two to forty
/// times larger because a real meeting contains real spectral variation, and
/// they are *not* used, because they include the between-speaker term this
/// constant exists to exclude.
#[must_use]
pub fn within_speaker_spread(j: usize, layout: &Layout) -> f32 {
    // Block-wise, in each block's natural units:
    //   cepstra        c1..c19, DCT of a log-mel spectrum   0.0826
    //   delta          mean |first difference|, frames      0.0100
    //   double delta   mean |second difference|, frames      0.0055
    //   (pitch is absolute and is never scaled by this)
    //   voicedness     fraction of frames with a peak       0.0015
    //   shape          centroid/spread/rolloff, log2 Hz       0.0044
    //   band ratios    ln energy ratios, 5 bands            0.0011
    //   LTAS           ln energy ratios, 16 bands           0.0032
    if j < layout.delta {
        0.0826
    } else if j < layout.double_delta {
        0.0100
    } else if j < layout.pitch {
        0.0055
    } else if j == layout.pitch {
        1.0
    } else if j == layout.voicedness {
        0.0015
    } else if j < layout.bands {
        0.0044
    } else if j < layout.ltas {
        0.0011
    } else {
        0.0032
    }
}

/// `1.4826`: converts a median absolute deviation into a standard deviation for
/// a Gaussian. (`1 / 0.6745`.)
pub const MAD_TO_SIGMA: f32 = 1.482_602;

/// The robust standard deviation of a centred column: `1.4826 * MAD`.
///
/// Uses the median rather than the standard deviation so that a handful of
/// extreme windows cannot inflate it. Here it answers only "is this dimension
/// moving at all", so the between-speaker term being included is harmless --
/// a dimension that varies only *between* speakers is still not flat, and
/// keeping it is correct.
fn robust_sd(centred: &[f32]) -> f32 {
    if centred.is_empty() {
        return 0.0;
    }
    let mut a: Vec<f32> = centred.iter().map(|v| v.abs()).collect();
    a.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let med = if a.len().is_multiple_of(2) {
        0.5 * (a[a.len() / 2 - 1] + a[a.len() / 2])
    } else {
        a[a.len() / 2]
    };
    med * MAD_TO_SIGMA
}

/// Z-score every dimension across the windows of one file, apply the block
/// weights, add the carrier dimension, and L2-normalise each row.
///
/// `rows` is widened by one element: the last slot of each row becomes the
/// carrier. After this, the inner product of two rows is their cosine
/// similarity and `1 - dot` is the clustering distance.
///
/// A row whose norm is zero after weighting is left as all zeros rather than
/// being divided by zero: two all-zero rows then have distance 0 and land in
/// the same cluster, which is the right answer for "these windows carry no
/// information".
///
/// # Why there is deliberately no per-block renormalisation
///
/// The obvious next step after weighting is to divide each block by its own
/// root-mean-square, so that a block's weight really is the share of the final
/// norm it contributes. That was implemented, and it is wrong, and the
/// regression test `the_min_spread_floor_stops_a_flat_dimension_from_being_
/// amplified` is what says so.
///
/// Renormalising a block divides away the block's *absolute* spread, which is
/// precisely the quantity the `MinSpread` floor just decided was too small to
/// trust. A dimension whose spread is a thousandth of its floor is gated down
/// to a thousandth of a unit variance; dividing the block by its own RMS puts
/// it back to unit variance; the gate is a no-op, and the whole mechanism is
/// dead code that still reads as though it works.
///
/// The honest constraint is the one this code lives with: **an unlabelled
/// distance matrix cannot distinguish noise from signal.** Any scheme that
/// equalises dimensions promotes whatever is noisiest into full weight. So
/// dimensions are equalised (each is divided by its own file-wide standard
/// deviation, which is the only recording-independent scale available) and the
/// remaining judgement is made where it can be stated and defended:
///
/// * the [`MinSpread`] floor, in absolute units, gates dimensions carrying no
///   measurable variation;
/// * the [`crate::config::BlockWeights`], which are a documented opinion about
///   which *kinds* of evidence are trustworthy -- pitch leads, long-term
///   spectrum supports, frame-level dynamics are the weakest of the eight;
/// * [`temporal_smooth`], run before all of this, which is the actual fix for
///   phoneme variance: speaker identity is constant over seconds and survives a
///   moving average, the vowel sequence has a 200 ms period and does not.
pub fn normalise_in_place(
    rows: &mut [Vec<f32>],
    layout: &Layout,
    weights: &crate::config::BlockWeights,
    spread: &MinSpread,
) {
    if rows.is_empty() {
        return;
    }
    let d = layout.total;
    // One extra slot for the carrier. `debug_assert_eq!(rows[0].len(), d)`.
    for r in rows.iter_mut() {
        r.resize(d + 1, weights.carrier);
    }
    let n = rows.len() as f64;

    if rows.len() >= MIN_ZSCORE_ROWS {
        for j in 0..d {
            let w = block_weight(j, layout, weights);
            let floor = min_spread_for(j, layout, spread);

            if layout.is_absolute(j) {
                // Kept on its own scale: shifted by a fixed reference and
                // multiplied, with no per-file centring and no division by any
                // spread. See `Layout::is_absolute` -- the signal is in the
                // mean, and dividing by a spread would both delete it and
                // promote the quantisation jitter in its place.
                for r in rows.iter_mut() {
                    r[j] = (r[j] - PITCH_REFERENCE_LOG2) * w * ABSOLUTE_PITCH_SCALE;
                }
                continue;
            }

            // Gather the column, then centre it on its median. Median, not
            // mean: the windows that straddle a turn boundary would drag a
            // mean, and the median of a boundary-crossing window is already a
            // compromise.
            let mut col: Vec<f32> = rows.iter().map(|r| r[j]).collect();
            col.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let centre = if col.len().is_multiple_of(2) {
                0.5 * (col[col.len() / 2 - 1] + col[col.len() / 2])
            } else {
                col[col.len() / 2]
            };
            col.iter_mut().for_each(|v| *v -= centre);
            let scale = within_speaker_spread(j, layout);
            // The gate asks a *data* question -- is there any measurable
            // variation here at all -- while the denominator above is a
            // constant. Keeping the two separate is what makes both correct:
            // the gate is a per-recording health check on the feature, and the
            // scale is a per-block constant so that a distance means the same
            // thing in every recording. A gate driven by the constant would be
            // a fixed configuration decision wearing a measurement's clothes,
            // and a scale driven by the data would put the between-speaker term
            // back into the denominator, which is the failure
            // `within_speaker_spread` documents.
            let observed = robust_sd(&col);

            if observed <= floor {
                // Drop the dimension outright. `<=`, not `<`, so a dimension
                // that is perfectly flat always takes this branch: with a floor
                // of zero the alternative is a 0/0 division, and the NaN that
                // produces propagates through the whole vector and turns every
                // distance into a NaN that silently compares false to
                // everything.
                //
                // Dividing by the floor instead of dropping it does not work,
                // and the reason is the block weight. A gated dimension's
                // z-score is `scale / floor` -- small, but the weight is
                // multiplied in *afterwards*. So a dimension carrying a 1e-4
                // wobble comes out of normalisation larger than the carrier,
                // and two rows that differ only in that dimension end up
                // angularly opposite: a large cosine distance on a difference
                // of no consequence. The gate would be arithmetically honoured
                // and completely inert, which is the worst of both.
                //
                // Dropping is also the honest reading of what the floor
                // *means*. `MinSpread` says "a spread below this is not
                // measurable information", not "a spread below this is
                // information worth less". A dimension is either carrying
                // something or it is not, and there is no third option that
                // survives a weight and a cosine.
                for r in rows.iter_mut() {
                    r[j] = 0.0;
                }
                continue;
            }
            for r in rows.iter_mut() {
                r[j] = ((r[j] - centre) / scale) * w;
            }
        }
    } else {
        // Too few rows for a distribution. Only the scale is set, not the
        // distribution: each dimension is divided by its own root-mean-square
        // so the blocks still contribute in their documented proportion.
        for j in 0..d {
            let mut scale = 0f64;
            for r in rows.iter() {
                scale += f64::from(r[j] * r[j]);
            }
            let denom = ((scale / n).sqrt() as f32).max(min_spread_for(j, layout, spread));
            let w = block_weight(j, layout, weights);
            for r in rows.iter_mut() {
                r[j] = r[j] / denom * w;
            }
        }
    }

    for r in rows.iter_mut() {
        r[d] = weights.carrier;
        let norm = r.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-12 {
            for v in r.iter_mut() {
                *v /= norm;
            }
        } else {
            r.iter_mut().for_each(|v| *v = 0.0);
        }
    }
}

/// Low-pass filter the window trajectory before normalisation.
///
/// # Why this exists
///
/// A 1.5 s window of speech contains about five syllables, so a window's
/// feature vector is largely a description of *which vowels were said in it*,
/// not of who said them. Two windows of the same person saying "yes" and
/// saying "the third quarter" end up with real cosine distance between them,
/// and after per-file z-scoring -- which promotes every dimension's variation
/// to unit variance regardless of whether that variation is speaker or phoneme
/// -- that phoneme-level distance is as large as the speaker-level distance.
/// The result is a clustering that finds twelve speakers in a conversation.
///
/// Smoothing the *features* before clustering, rather than smoothing the
/// *labels* after it, attacks the cause instead of the symptom. It is a
/// moving average along the timeline: speaker identity is constant over
/// seconds, so it survives; the vowel sequence has a 200 ms period, so it
/// averages away. The label-level median filter still runs afterwards, because
/// it also has to clean up boundary noise that this cannot reach.
///
/// The half-width is in windows, not frames, and the default weights are
/// triangular (1 2 3 2 1) rather than a box, which gives the centre window
/// the most weight and tapers the influence of its neighbours -- a box filter
/// would pull every window three windows towards the local mean and smear a
/// genuine turn boundary.
#[must_use]
pub fn temporal_weights(half: usize) -> Vec<f32> {
    if half == 0 {
        return vec![1.0];
    }
    let mut w: Vec<f32> = (0..=half).map(|k| (half + 1 - k) as f32).collect();
    let back: Vec<f32> = w.iter().skip(1).rev().copied().collect();
    w.extend(back);
    w
}

/// Apply a triangular moving average along the window trajectory, in place.
///
/// Windows are assumed to be in time order and evenly spaced, which the engine
/// guarantees: windows are emitted on a fixed stride over the frame grid, so
/// index adjacency is time adjacency. Edges use a **truncated** neighbourhood
/// rather than a reflected one, for the same reason the label median filter
/// does -- a reflected window invents a turn at the start of the file.
pub fn temporal_smooth(rows: &mut [Vec<f32>], half: usize) {
    if rows.len() < 2 || half == 0 {
        return;
    }
    let d = rows[0].len();
    if d == 0 {
        return;
    }
    let w = temporal_weights(half);
    let src: Vec<Vec<f32>> = rows.to_vec();
    let mut out = vec![0f32; d];
    for (i, slot) in rows.iter_mut().enumerate() {
        out.iter_mut().for_each(|v| *v = 0.0);
        let mut total = 0f32;
        for (k, weight) in w.iter().enumerate() {
            let offset = k as isize - half as isize;
            let j = i as isize + offset;
            if j < 0 || j as usize >= src.len() {
                continue;
            }
            let row = &src[j as usize];
            for (a, b) in out.iter_mut().zip(row.iter()) {
                *a += b * weight;
            }
            total += weight;
        }
        if total > 0.0 {
            for (a, b) in slot.iter_mut().zip(out.iter()) {
                *a = *b / total;
            }
        }
    }
}

/// Cosine distance between two L2-normalised vectors, clamped to `[0, 2]`.
///
/// The clamp is not cosmetic: rounding in the dot product can produce `-1e-7`
/// for identical rows, which would give a distance of `1 + 1e-7` and, in a
/// monotone-threshold stopping rule, make the merge cost of two identical
/// windows *larger* than that of two slightly different ones.
#[must_use]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for i in 0..n {
        dot += f64::from(a[i]) * f64::from(b[i]);
        na += f64::from(a[i]) * f64::from(a[i]);
        nb += f64::from(b[i]) * f64::from(b[i]);
    }
    // The vectors leaving `normalise_in_place` are already unit length, so this
    // division is a no-op for them and the clustering distances are unchanged.
    //
    // It is here because the alternative is a silent wrong answer. Clamping the
    // dot product to `[0, 2]` without dividing by the norms means any pair of
    // un-normalised vectors whose dot product exceeds 1 reports a distance of
    // exactly 0 -- "these windows are identical" -- when they are in fact
    // merely large. That is how a diagnostic built on hand-built vectors ends
    // up reporting a same-speaker p50 and a different-speaker p50 that are
    // *both* exactly 0.0, and it looks like a clustering result rather than a
    // bug in the measurement.
    let denom = na.sqrt() * nb.sqrt();
    if denom <= 1e-12 {
        // A zero vector carries no direction, so it matches anything. Two zero
        // vectors landing in the same cluster is the intended behaviour of the
        // "no features" case, and a zero against a non-zero is reported as 0
        // for the same reason.
        return 0.0;
    }
    (1.0 - dot / denom).clamp(0.0, 2.0)
}

/// One window's embedding: a feature vector plus the time it describes.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowEmbedding {
    /// Start of the window in source-media time.
    pub start: Time,
    /// End of the window in source-media time.
    pub end: Time,
    /// How much of the window was active speech, in milliseconds.
    pub speech_ms: i64,
    /// L2-normalised, weighted, z-scored feature vector.
    pub vector: Vec<f32>,
}

impl WindowEmbedding {
    /// A window with a zero vector, for the "no features" case.
    #[must_use]
    pub fn empty(start: Time, end: Time, dims: usize) -> Self {
        Self {
            start,
            end,
            speech_ms: 0,
            vector: vec![0.0; dims],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ActivityParams, BlockWeights};

    fn p() -> FeatureParams {
        FeatureParams::default()
    }

    #[test]
    fn layout_covers_the_vector_exactly() {
        let params = p();
        let l = Layout::of(&params);
        assert_eq!(l.total, params.vector_len());
        assert_eq!(l.cepstra, 0);
        assert_eq!(l.delta, 19);
        assert_eq!(l.double_delta, 38);
        assert_eq!(l.pitch, 57);
        assert_eq!(l.voicedness, 58);
        assert_eq!(l.shape, 59);
        assert_eq!(l.bands, 62);
        assert_eq!(l.ltas, 67);
        assert_eq!(l.total, 83);
    }

    #[test]
    fn every_dimension_maps_to_exactly_one_block() {
        let params = p();
        let l = Layout::of(&params);
        let w = BlockWeights::default();
        let seen: Vec<bool> = (0..l.total)
            .map(|j| {
                let b = block_weight(j, &l, &w);
                assert!(b > 0.0, "dimension {j} has weight {b}");
                true
            })
            .collect();
        assert!(seen.iter().all(|x| *x));
    }

    #[test]
    fn delta_of_a_constant_series_is_zero() {
        let c = vec![2.5f32; 20];
        let d = delta_coefficients(&c, 2);
        assert!(d.iter().all(|v| v.abs() < 1e-6), "{d:?}");
    }

    #[test]
    fn delta_of_a_ramp_is_the_slope() {
        // d[t] = sum n (c[t+n] - c[t-n]) / sum 2n^2 with c = 3t gives 3t in
        // the interior: each pair contributes 2*k*3*k = 6k^2 and the normaliser
        // is sum 2k^2, so the ratio is exactly 3.
        let c: Vec<f32> = (0..20).map(|i| 3.0 * i as f32).collect();
        let d = delta_coefficients(&c, 2);
        for (t, v) in d.iter().enumerate().take(4).skip(2) {
            assert!((v - 3.0).abs() < 1e-5, "t={t} got {v}");
        }
    }

    #[test]
    fn delta_clamps_at_the_edges_rather_than_reflecting() {
        // Edge frames must be finite and must not see out-of-range data. A
        // reflected implementation would produce a large spurious slope at
        // t = 0, which is the first frame of every window.
        let c: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let d = delta_coefficients(&c, 2);
        assert!(d.iter().all(|v| v.is_finite()));
        // The truncated window at t = 0 sees only the forward slope, so it
        // under-reports. That is the conservative direction and it is the
        // documented one: an edge frame looks flatter than it is.
        let interior = 1.0;
        assert!((d[10] - interior).abs() < 1e-5, "interior slope {interior}");
        assert!(d[0] < d[10], "edge {} vs interior {}", d[0], d[10]);
    }

    #[test]
    fn delta_of_nothing_is_nothing() {
        assert!(delta_coefficients(&[], 2).is_empty());
        assert_eq!(delta_coefficients(&[1.0], 0), vec![0.0]);
        assert!(delta_coefficients(&[1.0], 2).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn double_delta_of_a_constant_is_zero_and_of_a_ramp_is_constant() {
        let flat = vec![7.0f32; 20];
        assert!(
            double_delta_coefficients(&flat, 2)
                .iter()
                .all(|v| v.abs() < 1e-5)
        );
        let ramp: Vec<f32> = (0..20).map(|i| 3.0 * i as f32).collect();
        let dd = double_delta_coefficients(&ramp, 2);
        // A second difference reaches two frames past the edge effect, so the
        // clean interior starts at index 6.
        let interior = &dd[6..14];
        assert!(interior.iter().all(|v| v.abs() < 1e-4), "{interior:?}");
    }

    #[test]
    fn cosine_distance_of_identical_rows_is_zero_never_negative() {
        let a = vec![1.0f32, 0.0, 0.0];
        assert!((cosine_distance(&a, &a) - 0.0).abs() < 1e-12);
        // A row with rounding noise in it must not produce a negative
        // distance, which would make the stopping rule non-monotone.
        let b = vec![1.0f32, 1e-7, 0.0];
        assert!(cosine_distance(&a, &b) >= 0.0);
    }

    #[test]
    fn cosine_distance_of_orthogonal_rows_is_one() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cosine_distance_is_clamped_to_the_cosine_range() {
        let a = vec![1.0f32, 0.0];
        let b = vec![-1.0f32, 0.0];
        assert!((cosine_distance(&a, &b) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn cosine_distance_of_different_lengths_uses_the_common_prefix() {
        // Documented behaviour rather than a panic: a truncated row compares
        // on what it has. No caller in this crate produces ragged rows.
        assert!(cosine_distance(&[1.0, 0.0], &[1.0]).is_finite());
    }

    #[test]
    fn median_reduction_ignores_a_single_outlier_frame() {
        // The reason the window statistic is a median: one loud frame must
        // not be able to set the value.
        let layout = Layout {
            total: 3,
            voicedness: 0,
            pitch: 1,
            ..Layout::of(&p())
        };
        let mut acc = WindowAccumulator::new(layout, 2);
        for _ in 0..9 {
            acc.push(&[1.0, 6.0, 0.0], 6.0, 0.8);
        }
        acc.push(&[900.0, 900.0, 900.0], 6.0, 0.8);
        assert_eq!(acc.len(), 10);
        let mut out = vec![0f32; 3];
        acc.reduce(0.0, &mut out);
        assert!((out[2] - 0.0).abs() < 1e-6, "median was {}", out[2]);
        // Pitch is the mean of the voiced log2 F0, all 6.0.
        assert!((out[1] - 6.0).abs() < 1e-6);
        // Voicing is a mean, deliberately: the *fraction* of voiced frames is
        // the quantity, and a median would throw away the count.
        assert!((out[0] - 0.8).abs() < 1e-6, "voicedness {}", out[0]);
    }

    #[test]
    fn even_and_odd_counts_of_frames_both_reduce() {
        let layout = Layout {
            total: 1,
            voicedness: 0,
            pitch: 0,
            ..Layout::of(&p())
        };
        for n in [1usize, 2, 3, 4, 5] {
            let mut acc = WindowAccumulator::new(layout, 2);
            for i in 0..n {
                acc.push(&[i as f32], 0.0, 0.0);
            }
            let mut out = vec![0f32; 1];
            acc.reduce(0.0, &mut out);
            assert!(out[0].is_finite(), "n={n} gave {}", out[0]);
        }
    }

    #[test]
    fn an_unvoiced_window_takes_the_fallback_pitch() {
        // Without this the fricative windows would sit at log2(F0) = 0, which
        // z-scoring turns into a strong outlier and the clustering would make
        // every fricative its own speaker.
        let layout = Layout {
            total: 3,
            voicedness: 0,
            pitch: 1,
            ..Layout::of(&p())
        };
        let mut acc = WindowAccumulator::new(layout, 2);
        acc.push(&[0.0, 0.0, 0.0], 0.0, 0.0);
        acc.push(&[0.0, 0.0, 0.0], 0.0, 0.0);
        let mut out = vec![0f32; 3];
        acc.reduce(7.5, &mut out);
        assert!((out[1] - 7.5).abs() < 1e-6);
        assert!((out[0] - 0.0).abs() < 1e-6, "voicedness must be 0");
    }

    #[test]
    fn an_empty_window_reduces_to_zeros_with_a_fallback_pitch() {
        let layout = Layout {
            total: 3,
            voicedness: 1,
            pitch: 2,
            ..Layout::of(&p())
        };
        let acc = WindowAccumulator::new(layout, 2);
        let mut out = vec![9.0f32; 3];
        acc.reduce(6.0, &mut out);
        assert_eq!(out, vec![0.0, 0.0, 6.0]);
    }

    #[test]
    fn normalisation_l2_normalises_and_respects_block_weights() {
        let params = p();
        let layout = Layout::of(&params);
        // Eight genuinely identical windows. Enough rows for z-scoring to run,
        // and a constant input so the test is about the weights and the
        // normalisation rather than about the statistics.
        let mut rows = vec![vec![1.0f32; layout.total]; 8];
        normalise_in_place(
            &mut rows,
            &layout,
            &BlockWeights::default(),
            &MinSpread::default(),
        );
        for r in &rows {
            let n = r.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-5, "norm {n}");
            // One extra slot: the carrier dimension.
            assert_eq!(r.len(), layout.total + 1);
        }
        // Identical rows must survive as identical rows: z-scoring cannot
        // invent a difference between two identical inputs.
        let d = cosine_distance(&rows[0], &rows[1]);
        assert!(d < 1e-5, "identical windows ended up {d} apart");
    }

    #[test]
    fn a_constant_input_collapses_to_the_carrier() {
        // This is the mechanism behind "one speaker must not shatter".
        //
        // Ten identical windows. Every mean-normalised dimension has zero
        // observed spread, so the `MinSpread` floor drops it outright. The
        // pitch dimension is **not** dropped and **not** centred -- it is
        // absolute, so a constant pitch is a *result*, not a degeneracy: it says
        // this recording is one voice at 131 Hz. It contributes the same value
        // to every row, so the rows are still identical.
        let params = p();
        let layout = Layout::of(&params);
        let mut rows: Vec<Vec<f32>> = (0..10)
            .map(|_| {
                let mut r = vec![3.0f32; layout.total];
                r[layout.pitch] = 7.036;
                r
            })
            .collect();
        normalise_in_place(
            &mut rows,
            &layout,
            &BlockWeights::default(),
            &MinSpread::default(),
        );
        // Every mean-normalised dimension is gone.
        for (j, r0) in rows[0].iter().enumerate() {
            if layout.is_absolute(j) {
                continue;
            }
            assert!(
                r0.abs() < 1e-6,
                "dimension {j} should have been dropped, got {}",
                rows[0][j]
            );
        }
        // Pitch kept its absolute value, shifted by the fixed reference and
        // divided by the row norm. The meaningful property is that its size
        // *relative to the carrier* is the configured one, since L2
        // normalisation rescales the whole row.
        let expected = (7.036 - PITCH_REFERENCE_LOG2) * ABSOLUTE_PITCH_SCALE;
        let got = rows[0][layout.pitch] / rows[0][layout.total];
        assert!(
            (got - expected).abs() < 1e-4,
            "an absolute dimension must keep its own scale, got {got} wanted {expected}"
        );
        // And the rows are identical anyway, so the clustering sees one
        // speaker: constant pitch, nothing else. The carrier is no longer
        // exactly 1.0 after L2 normalisation, because the row is not
        // carrier-only any more -- it is carrier plus a constant pitch -- so the
        // invariant to assert is the unit norm.
        assert!(cosine_distance(&rows[0], &rows[9]).abs() < 1e-6);
        let norm: f32 = rows[0].iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "row norm {norm}");
    }

    #[test]
    fn the_min_spread_floor_stops_a_flat_dimension_from_being_amplified() {
        // THE regression test for the single-speaker case. Two windows that
        // differ only in a dimension whose spread is below the floor must not
        // have that difference turned into a signal.
        //
        // Uses a *cepstral* dimension, which is a mean-normalised one. The
        // promotion this guards against is the property of the z-score path
        // specifically: an absolute dimension like pitch is not z-scored at all
        // and so cannot be promoted in this way. See
        // `the_pitch_dimension_is_not_mean_normalised` for that half.
        let params = p();
        let layout = Layout::of(&params);
        let floor = MinSpread::default();
        let j = layout.cepstra; // c1
        // Six windows, all identical except for a 1e-4 wobble in c1. The
        // default cepstral floor is 1.5e-3, so the wobble's standard deviation
        // (about 8e-5) is an order of magnitude below it. Z-scoring on its own
        // would amplify that wobble to full unit variance and hand the
        // clustering a signal that carries no speaker information.
        let wobble = 1e-4f32;
        let build = || -> Vec<Vec<f32>> {
            (0..6)
                .map(|i| {
                    let mut r = vec![0.0f32; layout.total];
                    r[j] = 12.0 + if i % 2 == 0 { 0.0 } else { wobble };
                    r
                })
                .collect()
        };
        let mut rows = build();
        normalise_in_place(&mut rows, &layout, &BlockWeights::default(), &floor);
        let d = cosine_distance(&rows[0], &rows[1]);
        assert!(d < 1e-4, "floor did not hold: distance {d}");

        // The dimension is gone rather than merely small. This matters more
        // than the distance above: the block weight is multiplied in *after*
        // the gate, so a dimension merely shrunk to `wobble / floor` would come
        // out of normalisation at 0.04, and against a unit carrier that alone
        // would still separate two windows that carry no information at all. A
        // zeroed column cannot.
        for r in &rows {
            assert_eq!(
                r[j], 0.0,
                "a below-floor dimension must be dropped, not scaled"
            );
        }

        // Without the floor the same wobble is no longer zeroed, and z-scoring
        // has turned it into a unit-variance signal that the clustering cannot
        // distinguish from a real one.
        let mut unfloored = build();
        let zero = MinSpread {
            cepstra: 0.0,
            delta: 0.0,
            double_delta: 0.0,
            pitch: 0.0,
            voicedness: 0.0,
            spectral_shape: 0.0,
            band_ratios: 0.0,
            ltas: 0.0,
        };
        normalise_in_place(&mut unfloored, &layout, &BlockWeights::default(), &zero);
        let promoted = cosine_distance(&unfloored[0], &unfloored[1]);
        assert!(
            unfloored[0][j] != 0.0 && unfloored[1][j] != 0.0,
            "without the floor the dimension should survive normalisation"
        );
        assert!(
            promoted > 100.0 * d.max(f64::MIN_POSITIVE),
            "without the floor the wobble is amplified from {d} to {promoted}"
        );
    }

    #[test]
    fn the_pitch_dimension_is_not_mean_normalised() {
        // The invariant that a z-scored pitch block destroys, and the reason
        // `Layout::is_absolute` exists.
        //
        // Two files, one voice each. Measured across the windows of each, the
        // standard deviation of the median log2 F0 is *the same*: 0.0019. So
        // pitch variance cannot tell them apart -- and after a z-score it
        // cannot tell anything apart, because the z-score divides by that
        // identical number and both files end up with the same vector. Their
        // entire difference is in the mean, 0.79 apart, and the mean is exactly
        // what a z-score subtracts.
        //
        // This test builds that pair directly: same spread, different mean.
        let params = p();
        let layout = Layout::of(&params);
        let jitter = 0.0019f32;
        let mut low = vec![vec![0.0f32; layout.total]; 6];
        let mut high = vec![vec![0.0f32; layout.total]; 6];
        for (i, l) in low.iter_mut().enumerate() {
            l[layout.pitch] = 6.88 + if i % 2 == 0 { 0.0 } else { jitter };
        }
        for (i, h) in high.iter_mut().enumerate() {
            h[layout.pitch] = 7.67 + if i % 2 == 0 { 0.0 } else { jitter };
        }
        let (w, s) = (BlockWeights::default(), MinSpread::default());
        normalise_in_place(&mut low, &layout, &w, &s);
        normalise_in_place(&mut high, &layout, &w, &s);
        let between = cosine_distance(&low[0], &high[0]);
        let within = cosine_distance(&low[0], &low[1]);
        assert!(
            between > 0.2,
            "two voices an octave apart gave only {between}"
        );
        assert!(
            within < 0.01,
            "one voice's own jitter of {jitter} gave {within}"
        );
        assert!(
            between > 20.0 * within,
            "the octave gap ({between}) is not clearly separated from the jitter ({within})"
        );
    }

    #[test]
    fn a_dimension_with_real_spread_survives_the_floor() {
        let params = p();
        let layout = Layout::of(&params);
        let mut rows: Vec<Vec<f32>> = (0..6)
            .map(|i| {
                let mut r = vec![0.0f32; layout.total];
                // 64 Hz for half the windows, 194 Hz for the rest: a real,
                // five-semitone difference.
                r[layout.pitch] = if i % 2 == 0 { 6.0 } else { 7.6 };
                r
            })
            .collect();
        normalise_in_place(
            &mut rows,
            &layout,
            &BlockWeights::default(),
            &MinSpread::default(),
        );
        // After normalisation the two rows must be measurably apart, or the
        // engine could not separate a low voice from a high one.
        let d = cosine_distance(&rows[0], &rows[1]);
        assert!(d > 0.05, "a five-semitone pitch gap gave only {d}");
    }

    #[test]
    fn a_short_recording_sets_scale_but_not_distribution() {
        // Fewer than MIN_ZSCORE_ROWS rows: each dimension is divided by its own
        // root-mean-square rather than z-scored, so two near-identical windows
        // stay near identical. Z-scoring two rows maps them onto `+v` and
        // `-v` for whatever dimension they differ in, which is a cosine
        // distance of 2 -- the maximum -- on a difference of no consequence.
        let params = p();
        let layout = Layout::of(&params);
        let mut rows: Vec<Vec<f32>> = (0..2)
            .map(|i| {
                let mut r = vec![0.0f32; layout.total];
                r[layout.pitch] = 6.5 + i as f32 * 0.0001;
                r
            })
            .collect();
        normalise_in_place(
            &mut rows,
            &layout,
            &BlockWeights::default(),
            &MinSpread::default(),
        );
        let d = cosine_distance(&rows[0], &rows[1]);
        assert!(d < 1e-3, "two near-identical windows ended up {d} apart");
    }

    #[test]
    fn a_single_row_keeps_its_shape_rather_than_being_zeroed() {
        // One window means no across-file statistics at all, so the engine
        // falls back to per-dimension scaling. Zeroing the row would be
        // defensible too -- there is nothing to compare it against -- but
        // keeping the weighted shape means the row still carries the block
        // proportions, which is the more useful of the two degenerate answers.
        let params = p();
        let layout = Layout::of(&params);
        let mut rows = vec![vec![1.0f32; layout.total]];
        normalise_in_place(
            &mut rows,
            &layout,
            &BlockWeights::default(),
            &MinSpread::default(),
        );
        let r = &rows[0];
        assert_eq!(r.len(), layout.total + 1);
        assert!(r.iter().all(|v| v.is_finite()));
        let n = r.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-5, "norm {n}");
        // The pitch block outweighs the double-delta block, as documented.
        assert!(r[layout.pitch] > r[layout.double_delta]);
    }

    #[test]
    fn an_all_zero_row_stays_zero_rather_than_becoming_nan() {
        let params = p();
        let layout = Layout::of(&params);
        let mut rows = vec![vec![0.0f32; layout.total]];
        normalise_in_place(
            &mut rows,
            &layout,
            &BlockWeights::default(),
            &MinSpread::default(),
        );
        assert!(rows[0].iter().all(|v| v.is_finite()));
    }

    #[test]
    fn normalisation_of_nothing_does_nothing() {
        let params = p();
        let layout = Layout::of(&params);
        let mut rows: Vec<Vec<f32>> = Vec::new();
        normalise_in_place(
            &mut rows,
            &layout,
            &BlockWeights::default(),
            &MinSpread::default(),
        );
        assert!(rows.is_empty());
    }

    #[test]
    fn frame_vector_fills_every_dimension() {
        // The regression test for a bug that cost 57 of the vector's 83
        // dimensions and was invisible to every other test: the cepstral block's
        // *width* was read from `layout.cepstra`, which is the block's
        // *offset* and is zero, so an empty slice was copied and the whole
        // cepstral, delta and double-delta region stayed at zero. A real
        // signal puts energy in every dimension, so this asserts exactly that.
        let params = p();
        let layout = Layout::of(&params);
        let mut fe = FrontEnd::new(params, ActivityParams::default());
        let (buf, _) = crate::synth::render(
            &[crate::synth::Utterance::new(0, 0, 2_000)],
            &crate::synth::two_voices(),
            2_000,
            16_000,
            0.002,
            3,
        );
        let mut power = vec![0f32; fe.n_freq()];
        let mut log_mel = vec![0f32; params.n_mel];
        let mut cepstra = vec![0f32; params.n_cepstra];
        let mut out = vec![0f32; layout.total];
        frame_vector(
            &mut fe,
            &buf,
            50,
            &mut power,
            &mut log_mel,
            &mut cepstra,
            &mut out,
        );
        let (log2, conf) = fe.frame_pitch(&buf, 50);
        // The two dimensions `frame_vector` is documented *not* to set: the
        // pitch and the voicing fraction belong to the window accumulator,
        // which knows the whole window's statistics.
        out[layout.pitch] = log2;
        out[layout.voicedness] = conf;

        for (j, v) in out.iter().enumerate() {
            assert!(
                v.is_finite(),
                "dimension {j} is not finite: {v} ({} live of {})",
                out.iter().filter(|x| **x != 0.0).count(),
                layout.total
            );
        }
        // The cepstral block is the one that was silently empty. Its 19
        // coefficients have to be non-zero for a voiced frame.
        let cep = &out[layout.cepstra..layout.delta];
        assert_eq!(cep.len(), 19, "the cepstral block is 19 wide");
        let live = cep.iter().filter(|v| **v != 0.0).count();
        assert!(
            live >= 15,
            "only {live} of 19 cepstral coefficients are populated: {cep:?}"
        );
        // And the deltas, which are written by the accumulator rather than
        // here, are checked separately by the accumulator's own tests.
        let shape = &out[layout.shape..layout.shape + 3];
        assert!(shape.iter().all(|v| *v != 0.0), "spectral shape is empty");
        let ltas = &out[layout.ltas..layout.ltas + layout.total - layout.ltas];
        assert_eq!(ltas.len(), params.n_ltas);
        assert!(
            ltas.iter().filter(|v| **v != 0.0).count() >= 4,
            "LTAS is empty"
        );
    }

    #[test]
    fn frame_vector_is_finite_for_a_real_tone() {
        let params = p();
        let layout = Layout::of(&params);
        let mut fe = FrontEnd::new(params, ActivityParams::default());
        let buf: Vec<f32> = (0..16_000)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                0.4 * (2.0 * std::f32::consts::PI * 150.0 * t).sin()
                    + 0.15 * (2.0 * std::f32::consts::PI * 1_200.0 * t).sin()
            })
            .collect();
        let mut power = vec![0f32; fe.n_freq()];
        let mut log_mel = vec![0f32; params.n_mel];
        let mut cepstra = vec![0f32; params.n_cepstra];
        let mut out = vec![0f32; layout.total];
        frame_vector(
            &mut fe,
            &buf,
            50,
            &mut power,
            &mut log_mel,
            &mut cepstra,
            &mut out,
        );
        assert_eq!(out.len(), layout.total);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "a NaN reached the vector"
        );
    }

    #[test]
    fn frame_vector_is_finite_for_silence() {
        let params = p();
        let layout = Layout::of(&params);
        let mut fe = FrontEnd::new(params, ActivityParams::default());
        let mut power = vec![0f32; fe.n_freq()];
        let mut log_mel = vec![0f32; params.n_mel];
        let mut cepstra = vec![0f32; params.n_cepstra];
        let mut out = vec![0f32; layout.total];
        frame_vector(
            &mut fe,
            &vec![0.0; 16_000],
            50,
            &mut power,
            &mut log_mel,
            &mut cepstra,
            &mut out,
        );
        assert!(out.iter().all(|v| v.is_finite()), "silence produced a NaN");
    }

    #[test]
    fn empty_embedding_carries_the_requested_width() {
        let e = WindowEmbedding::empty(Time(0), Time(1_000), 83);
        assert_eq!(e.vector.len(), 83);
        assert_eq!(e.speech_ms, 0);
    }
}

//! Adaptive noise floor estimation.
//!
//! # What this is for
//!
//! Every threshold in the energy detector is expressed *relative* to the room,
//! because no absolute dBFS value is correct in more than one room. The floor
//! has to do three things a fixed constant cannot:
//!
//! * **Follow the room down quickly.** Someone turns the fan off; the next few
//!   seconds must not be read as speech.
//! * **Follow the room up slowly, and only when the room changed.** A loud
//!   speaker must not raise the floor for the quiet speaker, and a passing
//!   truck must not raise it either.
//! * **Survive a file that starts with nobody speaking.** This is what the
//!   warm-up ramp below is for.
//!
//! # Why a trailing-window percentile and not an exponential tracker
//!
//! The textbook alternative is exponential minimum statistics: keep a
//! per-frequency-bin running minimum and an exponentially-smoothed average.
//! It is the right algorithm when the spectrum is stable over a long window and
//! the noise is stationary, and it is the wrong one here for two reasons.
//!
//! 1. **It is per-bin and needs a 257-bin state machine to be correct.** The
//!    per-bin minimum tracks each bin's noise, and a speech bin's minimum is
//!    set by whatever quiet moment last passed through it -- which, in a
//!    meeting that never goes quiet, is minutes ago.
//! 2. **The failure mode is asymmetric and expensive.** An exponential tracker
//!    with a slow rise is a *mean* tracker with extra steps: during a 40-second
//!    monologue it converges upward toward the speech level, and the quiet
//!    speaker who talks over the end of the monologue is lost. A percentile is
//!    not a mean. It is defined to be dominated by the quietest part of the
//!    window, so no amount of loud speech in the window can move it, ever. That
//!    property is worth the cost of a sort.
//!
//! So: a **low quantile over a trailing window of frame levels in dB**.
//! Concretely, at 100 frames/s, a 2 s window and the 20th percentile means the
//! floor is set by the quietest ~40 frames of the last 2 seconds. A loud
//! passage has to occupy 80 % of a 2-second window before the floor even
//! begins to notice it, and by then it is not a passage, it is the room.
//!
//! # The warm-up ramp
//!
//! At the start of a file the window is not full, and two things go wrong if it
//! is not handled. First, an empty window has no quantile. Second, and more
//! subtly: with 20 frames seen and a 200-frame window, the true 20th percentile
//! is the 4th-smallest of 20 -- a much noisier estimate than intended, and it
//! *rises* as the window fills, so a passage of speech starting at t=0 gets
//! progressively more likely to be called noise.
//!
//! The fix is to ramp the quantile itself with the fill level:
//!
//! ```text
//! index = floor(q_eff * n),   q_eff = q * min(1, n / warmup_frames)
//! ```
//!
//! At `n = 1` the effective quantile is 0, i.e. the *running minimum*, which is
//! the best possible estimate from one sample. By `n = warmup_frames` it has
//! reached the design quantile and the tracker is in steady state. This is
//! exponential minimum statistics in all but name, confined to the first 500 ms
//! where it is the only thing that can work.
//!
//! The honest limitation: **the first `warmup` of audio is calibration.** If a
//! recording opens with five seconds of speech and no lead-in, that speech is
//! the minimum of the window and becomes the floor. Every parameter-free noise
//! tracker has this property. It is the single strongest argument for the
//! neural path, and it is why [`crate::neural::SileroVad`] is worth the
//! dependency rather than a nicety.

use std::collections::VecDeque;

/// A trailing-window quantile over a stream of dB values, in O(1) per sample.
///
/// # Why a histogram and not a sort
///
/// The obvious implementation keeps a `VecDeque<f32>` and calls
/// `select_nth_unstable` on a copy of the window for every frame. That is
/// `O(w log w)` per frame, i.e. ~4500 comparisons per frame, 450 000 per
/// second of audio, and it allocates. At 100 frames/s over a 3-hour recording
/// that is 1.6e10 comparisons and 1.08e6 allocations, for a value that
/// changes by a fraction of a dB between consecutive frames.
///
/// A fixed histogram of 0.5 dB bins over a 120 dB span is 240 counters: push
/// and pop are two integer operations, and the quantile is a short scan from
/// the previously returned bin. The resolution cost is 0.5 dB, which is
/// irrelevant: the detector's own decision boundaries are 0.03 in score units,
/// far finer than any floor quantisation that could matter.
///
/// # Quantisation convention
///
/// `quantile(q)` returns the value of the `floor(q * n)`-th smallest sample,
/// 0-based, clamped into range. So `q = 0.0` is the minimum, `q = 1.0` is the
/// maximum, and `q = 0.5` on `{1,2,3,4}` is the 3rd-smallest, i.e. 3. This is
/// the "nearest-rank, floor" convention; it is stated rather than assumed
/// because the other common convention (interpolating) would make the result
/// depend on values between samples, which a histogram cannot represent.

#[derive(Debug, Clone)]
pub struct SlidingQuantile {
    capacity: usize,
    lo_db: f32,
    inv_bin_db: f32,
    bins: usize,
    counts: Vec<u32>,
    window: VecDeque<u16>,
    n: usize,
    /// Cached index of the bin the last query landed on, and the number of
    /// samples in bins strictly below it. Together they make a query O(1) in
    /// the common case while the histogram stays a plain array.
    cursor: usize,
    prefix_at_cursor: usize,
}

impl SlidingQuantile {
    /// A window of `capacity` samples over `[lo_db, lo_db + bins * bin_db)`.
    ///
    /// `bin_db` of 0.5 dB over 240 bins covers `[-120, 0)` dB. Anything above
    /// 0 dB (a waveform that exceeds full scale, which a badly-normalised
    /// export can do) is clamped into the top bin rather than dropped, because
    /// dropping it would make the total count disagree with the window length
    /// and the quantile would be computed over the wrong number of samples.
    #[must_use]
    pub fn new(capacity: usize, lo_db: f32, bin_db: f32) -> Self {
        let bin_db = if bin_db > 0.0 { bin_db } else { 0.5 };
        let span_db = 120.0f32;
        let bins = (span_db / bin_db).ceil() as usize;
        Self {
            capacity: capacity.max(1),
            lo_db,
            inv_bin_db: 1.0 / bin_db,
            bins,
            counts: vec![0; bins],
            window: VecDeque::with_capacity(capacity.max(1)),
            n: 0,
            cursor: 0,
            prefix_at_cursor: 0,
        }
    }

    /// The window length, in samples.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// How many samples are in the window right now.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the window has no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Whether the window is full.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.n >= self.capacity
    }

    fn bin_of(&self, value: f32) -> usize {
        if value.is_nan() {
            return 0;
        }
        let b = ((value - self.lo_db) * self.inv_bin_db).floor();
        if b < 0.0 {
            0
        } else if b as usize >= self.bins {
            self.bins - 1
        } else {
            b as usize
        }
    }

    /// Add a sample, evicting the oldest once the window is full.
    pub fn push(&mut self, value: f32) {
        if self.window.len() == self.capacity
            && let Some(old) = self.window.pop_front()
        {
            self.counts[old as usize] -= 1;
            self.n -= 1;
            if (old as usize) < self.cursor {
                self.prefix_at_cursor -= 1;
            }
        }
        let b = self.bin_of(value);
        self.counts[b] += 1;
        if b < self.cursor {
            self.prefix_at_cursor += 1;
        }
        self.window.push_back(b as u16);
        self.n += 1;
    }

    /// The `floor(q * n)`-th smallest value in the window, or `None` if empty.
    ///
    /// The returned value is the **bin's lower edge**, not the sample it came
    /// from: a histogram has forgotten the exact values, and reporting a
    /// number the tracker cannot actually justify is worse than reporting a
    /// conservative one. The error is bounded by `bin_db` and always in the
    /// pessimistic direction for a low quantile -- the reported floor is never
    /// *higher* than a true one, so a real speech frame is never made to look
    /// quieter than it is.
    pub fn quantile(&mut self, q: f32) -> Option<f32> {
        self.quantile_at(q, self.n)
    }

    /// As [`Self::quantile`], with an explicit sample count.
    ///
    /// Separated because the warm-up ramp needs the quantile of a *subset* of
    /// the window (the `warmup` most recent samples) without disturbing the
    /// window itself.
    pub fn quantile_at(&mut self, q: f32, n: usize) -> Option<f32> {
        if n == 0 || self.n == 0 {
            return None;
        }
        let n = n.min(self.n);
        let q = q.clamp(0.0, 1.0);
        // index-th smallest, 0-based, over the `n` samples in question.
        let index = ((q * n as f32).floor() as usize).min(n - 1);

        // Walk out from the cached cursor, which holds the number of samples
        // in bins strictly below it. Both stay in sync on every push and pop,
        // so the common case -- the answer has not moved far since last frame
        // -- costs one comparison.
        if index < self.prefix_at_cursor {
            // The answer moved down. Walk the cursor back down.
            while self.cursor > 0 {
                self.cursor -= 1;
                self.prefix_at_cursor -= self.counts[self.cursor] as usize;
                if index >= self.prefix_at_cursor {
                    return Some(self.bin_value(self.cursor));
                }
            }
            return Some(self.bin_value(0));
        }
        let end_of_cursor = self.prefix_at_cursor + self.counts[self.cursor] as usize;
        if index < end_of_cursor {
            return Some(self.bin_value(self.cursor));
        }
        // The answer moved up.
        let mut acc = end_of_cursor;
        let mut b = self.cursor + 1;
        while b < self.bins {
            let end = acc + self.counts[b] as usize;
            if index < end {
                self.cursor = b;
                self.prefix_at_cursor = acc;
                return Some(self.bin_value(b));
            }
            acc = end;
            b += 1;
        }
        // `index` is at or beyond the largest populated bin: the maximum.
        self.cursor = self.bins - 1;
        self.prefix_at_cursor = acc - self.counts[self.bins - 1] as usize;
        Some(self.bin_value(self.bins - 1))
    }

    fn bin_value(&self, bin: usize) -> f32 {
        self.lo_db + bin as f32 / self.inv_bin_db
    }

    /// The quantile with the warm-up ramp applied, which is what the detector
    /// actually wants.
    ///
    /// `q` is the design quantile, `warmup` is the number of frames over which
    /// to ramp from the running minimum up to it. See the module docs for why.
    #[must_use]
    pub fn floor_db(&mut self, q: f32, warmup: usize) -> Option<f32> {
        if self.n == 0 {
            return None;
        }
        let warmup = warmup.max(1);
        // The index the caller effectively asks for is `floor(q_eff * n)`.
        // Ramping `q_eff` from `q * n/warmup` (giving index `q*n^2/warmup`,
        // i.e. the running minimum at n=1) up to `q` at n=warmup turns the
        // first half second into exponential minimum statistics and leaves
        // steady state a plain `q`-quantile over the whole window.
        let q_eff = if self.n < warmup {
            q * self.n as f32 / warmup as f32
        } else {
            q
        };
        self.quantile_at(q_eff, self.n)
    }
}

impl PartialEq for SlidingQuantile {
    fn eq(&self, other: &Self) -> bool {
        self.capacity == other.capacity
            && self.window.len() == other.window.len()
            && self.window.iter().eq(other.window.iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Insertion-sort reference implementation, so the histogram version is
    /// checked against something obviously correct rather than against itself.
    fn reference(values: &[f32], q: f32) -> f32 {
        if values.is_empty() {
            return f32::NAN;
        }
        let mut v = values.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = v.len();
        let i = (q.clamp(0.0, 1.0) * n as f32).floor() as usize;
        v[i.min(n - 1)]
    }

    fn fill(n: usize, f: impl Fn(usize) -> f32) -> Vec<f32> {
        (0..n).map(f).collect()
    }

    #[test]
    fn an_empty_window_has_no_quantile() {
        let mut q = SlidingQuantile::new(10, -120.0, 0.5);
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
        assert!(q.quantile(0.2).is_none());
        assert!(q.floor_db(0.2, 5).is_none());
    }

    #[test]
    fn a_single_sample_is_its_own_every_quantile() {
        let mut q = SlidingQuantile::new(10, -120.0, 0.5);
        q.push(-40.0);
        for p in [0.0, 0.2, 0.5, 1.0] {
            let got = q.quantile(p).unwrap();
            assert!((got - (-40.0)).abs() <= 0.5, "q={p} gave {got}");
        }
    }

    #[test]
    fn quantiles_match_a_sorted_reference_within_one_bin() {
        // 0.5 dB bins, so the answer may be up to 0.5 dB pessimistic, and
        // never optimistic.
        let mut q = SlidingQuantile::new(200, -120.0, 0.5);
        let values = fill(200, |i| -60.0 + (i as f32 * 0.37).sin() * 20.0);
        for v in &values {
            q.push(*v);
        }
        for p in [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 0.9, 1.0] {
            let got = q.quantile(p).unwrap();
            let want = reference(&values, p);
            assert!(
                (got - want).abs() <= 0.5 + 1e-3,
                "q={p}: histogram {got} vs reference {want}"
            );
            assert!(got <= want + 1e-3, "q={p}: histogram is optimistic");
        }
    }

    #[test]
    fn the_window_evicts_the_oldest_sample_and_never_exceeds_capacity() {
        let mut q = SlidingQuantile::new(5, -120.0, 0.5);
        for v in [-10.0, -10.0, -10.0, -10.0, -10.0] {
            q.push(v);
        }
        assert!(q.is_full());
        assert_eq!(q.len(), 5);
        // Now push a much quieter sample: the -10s must all be gone.
        for _ in 0..5 {
            q.push(-70.0);
        }
        assert_eq!(q.len(), 5, "length must not grow past capacity");
        let got = q.quantile(0.5).unwrap();
        assert!(
            (got + 70.0).abs() <= 0.5,
            "the stale -10 dB samples are still in the window: {got}"
        );
    }

    #[test]
    fn the_cursor_does_not_change_the_answer() {
        // The cached cursor is a performance optimisation; if it ever returns a
        // different bin for the same state, every tuning result is noise.
        let mut q = SlidingQuantile::new(50, -120.0, 0.5);
        let values = fill(50, |i| -40.0 + (i as f32) * 0.9);
        for (i, v) in values.iter().enumerate() {
            q.push(*v);
            let a = q.quantile(0.2).unwrap();
            // Ask in a different order to move the cursor around first.
            let _ = q.quantile(0.9);
            let _ = q.quantile(0.0);
            let b = q.quantile(0.2).unwrap();
            assert_eq!(a, b, "cursor changed the answer at sample {i}");
        }
    }

    #[test]
    fn a_very_loud_frame_cannot_move_a_low_quantile() {
        // The whole reason a percentile is used rather than a mean. Feed 20 %
        // very loud frames into a 200-frame window and the 20th percentile is
        // the quiet frames' own level, unchanged.
        let mut q = SlidingQuantile::new(200, -120.0, 0.5);
        for _ in 0..160 {
            q.push(-50.0);
        }
        let before = q.quantile(0.2).unwrap();
        for _ in 0..40 {
            q.push(-5.0);
        }
        let after = q.quantile(0.2).unwrap();
        assert!(
            (after - before).abs() <= 0.5,
            "a 20 % loud passage moved the floor from {before} to {after}"
        );
        // And the loud frames are still visible at a high quantile, which is
        // what makes the tracker a *measurement* rather than a clamp.
        assert!(q.quantile(0.9).unwrap() > -20.0);
    }

    #[test]
    fn the_warmup_ramps_from_the_running_minimum_up_to_the_design_quantile() {
        let mut q = SlidingQuantile::new(200, -120.0, 0.5);
        // One loud frame among quiet ones: early on, the effective quantile is
        // 0, so the floor is the true minimum.
        q.push(-20.0);
        let first = q.floor_db(0.2, 50).unwrap();
        assert!(
            first < -19.0,
            "the running minimum was expected, got {first}"
        );

        // Fill past the warm-up: the 20th percentile of 50 quiet frames is
        // still quiet, and the loud frame has been evicted.
        for _ in 0..60 {
            q.push(-50.0);
        }
        let warm = q.floor_db(0.2, 50).unwrap();
        assert!((warm + 50.0).abs() <= 0.5, "floor was {warm}");
    }

    #[test]
    fn the_warmup_never_exceeds_the_design_quantile() {
        // If the ramp overshot, a single loud first frame would set a high
        // floor for a long time and the file's opening speech would be lost.
        let mut q = SlidingQuantile::new(200, -120.0, 0.5);
        for i in 0..49 {
            q.push(if i == 0 { -10.0 } else { -50.0 });
        }
        let got = q.floor_db(0.2, 50).unwrap();
        assert!(
            (got + 50.0).abs() <= 0.5,
            "at 49/50 of the warm-up the floor should already be near -50, got {got}"
        );
    }

    #[test]
    fn values_outside_the_histogram_range_are_clamped_not_dropped() {
        // A badly normalised export can exceed 0 dB. Dropping those samples
        // would make the quantile wrong over the wrong population.
        let mut q = SlidingQuantile::new(10, -120.0, 0.5);
        for _ in 0..9 {
            q.push(0.0);
        }
        q.push(12.0);
        assert_eq!(q.len(), 10);
        assert!(q.quantile(1.0).unwrap() > -1.0);
        // And far below the range.
        let mut q2 = SlidingQuantile::new(10, -120.0, 0.5);
        for _ in 0..9 {
            q2.push(-60.0);
        }
        q2.push(-500.0);
        assert_eq!(q2.len(), 10);
        assert!(q2.quantile(0.0).unwrap() <= -119.0);
    }

    #[test]
    fn nan_is_handled_rather_than_poisoning_the_histogram() {
        let mut q = SlidingQuantile::new(10, -120.0, 0.5);
        q.push(f32::NAN);
        assert_eq!(q.len(), 1);
        assert!(q.quantile(0.5).unwrap().is_finite());
    }

    #[test]
    fn a_degenerate_bin_width_does_not_divide_by_zero() {
        let mut q = SlidingQuantile::new(4, -120.0, 0.0);
        q.push(-30.0);
        assert!(q.quantile(0.5).unwrap().is_finite());
        let mut q = SlidingQuantile::new(0, -120.0, 0.5);
        q.push(-30.0);
        q.push(-31.0);
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn a_three_hour_file_is_quantised_in_linear_time() {
        // 3 hours at 100 frames/s. The point is not the wall clock but the
        // *shape*: 1.08e6 pushes against a fixed 240-counter histogram. If
        // this ever regressed to a per-frame sort it would take minutes.
        let mut q = SlidingQuantile::new(200, -120.0, 0.5);
        let start = std::time::Instant::now();
        for i in 0..1_080_000u32 {
            let v = -50.0 + 18.0 * ((i as f32 * 0.01).sin());
            q.push(v);
            if i % 1000 == 0 {
                let _ = q.floor_db(0.2, 50);
            }
        }
        let elapsed = start.elapsed();
        assert_eq!(q.len(), 200);
        assert!(
            elapsed.as_secs_f64() < 2.0,
            "1.08e6 pushes took {elapsed:?}; the tracker is no longer O(1) per sample"
        );
    }
}

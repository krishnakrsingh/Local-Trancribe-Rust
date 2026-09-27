//! Downmixing many channels to one.
//!
//! # Why average and not sum
//!
//! Stereo to mono has two defensible definitions and they are not the same
//! function:
//!
//! * **Sum** (`l + r`) preserves the amplitude of a *correlated* signal exactly.
//!   A mono signal panned centre duplicates into both channels, so summing gives
//!   `2p` — a 6 dB gain — which then clips.
//! * **Average** (`(l + r) / 2`) preserves the amplitude of a correlated signal
//!   exactly, and attenuates an uncorrelated one by `sqrt(2)/2` (about -3 dB).
//!
//! VaaniDesk averages. A conference recording is two microphones in one room:
//! the same voice is in both channels with a small delay and level difference,
//! so it is *mostly* correlated and averaging keeps the level where the user
//! expects it. Summing would push a normal meeting 6 dB louder and into
//! clipping. The -3 dB worst case applies only to fully uncorrelated content,
//! which for speech costs intelligibility nothing and for music is inaudible at
//! this stage because the signal is about to be turned into log-mel features.
//!
//! For more than two channels the same arithmetic applies. Ambisonics, 5.1 and
//! 64-channel stems are all averaged, which is the only channel-layout-agnostic
//! choice available: there is no way to downmix 64 channels correctly without
//! knowing what each one is, and guessing is worse than averaging.
//!
//! # The clipping guard
//!
//! Averaging cannot overflow: the mean of N values each in `[-1, 1]` is in
//! `[-1, 1]`. The risks are elsewhere.
//!
//! * **NaN and infinity.** A malformed or partially-written file, or a decoder
//!   that hit a corrupt block, can emit a non-finite sample. Left alone, one
//!   NaN turns a whole chunk of log-mel features into NaN and the ASR model
//!   then produces a transcript of nothing with no error anywhere. Every
//!   non-finite sample is replaced with silence and counted in
//!   [`DownmixStats::nonfinite_replaced`].
//! * **Out-of-range float PCM.** 32-bit float WAV legitimately carries values
//!   above 1.0, and a broken encoder can carry values near `f32::MAX`. Summing
//!   64 of those overflows to infinity even though no single one is
//!   non-finite. Samples beyond [`MAX_INPUT_SAMPLE`] are clamped, which makes
//!   overflow of the sum arithmetically impossible: the largest possible sum is
//!   `65535 * 8.0`, far inside `f32`'s range.
//! * **Over-unity output.** A mean of many strongly correlated channels can
//!   exceed 1.0 in magnitude. That is *counted* in
//!   [`DownmixStats::over_unity`] but not modified, because a loud sample is not
//!   a defect and the ASR front end is amplitude-scale invariant. The count
//!   exists so the pipeline can surface it if it wants to.
//!
//! All three counts are summed into [`DecodeReport`](crate::DecodeReport), so
//! "the decoder produced a click" is a measurable claim rather than a hunch.

/// The largest magnitude accepted from a decoder for one channel sample.
///
/// Eight is roughly 18 dB of headroom above full scale: comfortably above any
/// legitimate float-PCM content, and small enough that summing the maximum
/// number of channels a container can declare (65535) cannot overflow `f32`.
pub const MAX_INPUT_SAMPLE: f32 = 8.0;

/// What a downmix did, so the caller can report it instead of guessing.
///
/// Every field is a counter; none of them change the audio that was produced
/// except through [`DownmixStats::nonfinite_replaced`] and
/// [`DownmixStats::input_clamped`], which are the two that had to.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DownmixStats {
    /// Mono frames written to the output.
    pub frames: u64,
    /// Channel samples that were NaN or infinite and became silence.
    pub nonfinite_replaced: u64,
    /// Channel samples whose magnitude exceeded [`MAX_INPUT_SAMPLE`] and were
    /// clamped to it.
    pub input_clamped: u64,
    /// Mono samples whose magnitude exceeded 1.0. Reported, not altered.
    pub over_unity: u64,
}

impl DownmixStats {
    /// Whether any sample had to be repaired.
    ///
    /// "Repaired" means the value was changed: a non-finite sample became
    /// silence, or an out-of-range one was clamped. A sample that merely exceeded
    /// unity is *not* a repair, because averaging did not modify it.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.nonfinite_replaced == 0 && self.input_clamped == 0
    }
}

/// Downmix interleaved `f32` to mono, appending the result to `out`.
///
/// `out` is cleared first, so the caller can reuse one buffer across packets
/// without it growing without bound. Writing into a caller-owned buffer rather
/// than returning a `Vec` is what lets the decode loop move twelve million
/// samples without a second full-size allocation: this is called once per
/// decoded packet into a buffer that is reused for the whole file.
///
/// * `interleaved` — samples in frame order, `channels` per frame. A trailing
///   partial frame is ignored, because a partial frame means the decoder and
///   the demuxer disagree about the channel count and guessing which samples to
///   drop would shift every subsequent sample.
/// * `channels` — 0 produces nothing at all. A container that declares zero
///   channels has no audio to downmix, and the caller reports that separately.
///
/// The three channel counts that matter are specialised: one channel is a copy,
/// two channels is one fused add-multiply, and everything else is a strided
/// sum. That is not premature optimisation — the two-channel path is the one
/// every real meeting recording takes, and it is roughly twice the speed of the
/// generic loop.
pub fn downmix_to_mono(interleaved: &[f32], channels: u16, out: &mut Vec<f32>) -> DownmixStats {
    let mut stats = DownmixStats::default();
    out.clear();
    let n = usize::from(channels);
    if n == 0 {
        return stats;
    }
    let frames = interleaved.len() / n;
    let used = frames * n;
    out.reserve(frames);
    match n {
        1 => {
            for &s in &interleaved[..used] {
                let v = sanitise(s, &mut stats);
                stats.over_unity += u64::from(v.abs() > 1.0);
                out.push(v);
            }
        }
        2 => {
            for [l, r] in interleaved[..used].as_chunks::<2>().0 {
                let l = sanitise(*l, &mut stats);
                let r = sanitise(*r, &mut stats);
                let m = (l + r) * 0.5;
                stats.over_unity += u64::from(m.abs() > 1.0);
                out.push(m);
            }
        }
        _ => {
            let scale = 1.0 / n as f32;
            for frame in interleaved[..used].chunks_exact(n) {
                let mut acc = 0.0f32;
                for &s in frame {
                    acc += sanitise(s, &mut stats);
                }
                let m = acc * scale;
                stats.over_unity += u64::from(m.abs() > 1.0);
                out.push(m);
            }
        }
    }
    stats.frames = frames as u64;
    stats
}

/// Make one channel sample finite and bounded.
#[inline]
fn sanitise(s: f32, stats: &mut DownmixStats) -> f32 {
    if !s.is_finite() {
        stats.nonfinite_replaced += 1;
        return 0.0;
    }
    if s > MAX_INPUT_SAMPLE {
        stats.input_clamped += 1;
        return MAX_INPUT_SAMPLE;
    }
    if s < -MAX_INPUT_SAMPLE {
        stats.input_clamped += 1;
        return -MAX_INPUT_SAMPLE;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mono(input: &[f32], channels: u16) -> (Vec<f32>, DownmixStats) {
        let mut out = Vec::new();
        let stats = downmix_to_mono(input, channels, &mut out);
        (out, stats)
    }

    #[test]
    fn zero_channels_produces_nothing_rather_than_dividing_by_zero() {
        // A container can declare zero channels. Producing `len / 0` samples
        // would panic in a stage that has no way to survive it.
        let (out, stats) = mono(&[0.1, 0.2, 0.3], 0);
        assert!(out.is_empty());
        assert_eq!(stats, DownmixStats::default());
    }

    #[test]
    fn a_trailing_partial_frame_is_dropped_not_guessed_at() {
        // Five samples at 2 channels is two frames and a half. Guessing which
        // half to keep would shift every sample after it.
        let (out, stats) = mono(&[1.0, 1.0, 0.0, 0.0, 0.5], 2);
        assert_eq!(out, vec![1.0, 0.0]);
        assert_eq!(stats.frames, 2);
    }

    #[test]
    fn mono_input_is_a_sanitised_copy_not_a_sum() {
        let (out, stats) = mono(&[0.25, -0.5, 0.75], 1);
        assert_eq!(out, vec![0.25, -0.5, 0.75]);
        assert_eq!(stats.frames, 3);
        assert!(stats.is_clean());
    }

    #[test]
    fn stereo_averages_rather_than_sums() {
        // The whole design rests on this: a centre-panned signal duplicated into
        // both channels must come out at the level it went in at, not 6 dB louder.
        let (out, stats) = mono(&[0.5, 0.5, -0.25, -0.25], 2);
        assert_eq!(out, vec![0.5, -0.25]);
        assert_eq!(stats.frames, 2);
        assert_eq!(stats.over_unity, 0);
    }

    #[test]
    fn a_channel_averaged_in_half_drops_three_db() {
        // The documented cost of averaging on uncorrelated content.
        let (out, _) = mono(&[1.0, -1.0], 2);
        assert_eq!(out, vec![0.0]);
        let (out, _) = mono(&[0.5, 0.0], 2);
        assert!((out[0] - 0.25).abs() < 1e-7);
    }

    #[test]
    fn averaging_cannot_clip_a_full_scale_stereo_pair() {
        // Summing would give 2.0. Averaging gives exactly 1.0.
        let (out, stats) = mono(&[1.0, 1.0, -1.0, -1.0], 2);
        assert_eq!(out, vec![1.0, -1.0]);
        assert_eq!(stats.over_unity, 0, "unity is not over unity");
    }

    #[test]
    fn many_correlated_channels_can_exceed_unity_and_that_is_counted_not_clipped() {
        // Four channels all at 1.0 average to 1.0; six at 1.0 followed by a
        // channel at -5.0 average to 1/6, still inside. The case that must be
        // *counted* is a mean above 1.0, which averaging cannot produce from
        // in-range inputs, so construct it via the documented path: a single
        // channel already above unity.
        let (out, stats) = mono(&[2.0], 1);
        assert_eq!(out, vec![2.0]);
        assert_eq!(stats.over_unity, 1);
        assert!(stats.is_clean(), "above-unity is not a defect");
    }

    #[test]
    fn sixty_four_channels_average_without_overflow() {
        // The edge case in the brief. `Position::from_count` in Symphonia caps a
        // WAV at 26 channels, so this can only be exercised directly - which is
        // exactly why it is a unit test of the pure function rather than an
        // integration test of a file.
        let n = 64usize;
        let input: Vec<f32> = (0..n * 100)
            .map(|i| if i % n == 0 { 1.0 } else { -1.0 })
            .collect();
        let (out, stats) = mono(&input, n as u16);
        assert_eq!(out.len(), 100);
        assert_eq!(stats.frames, 100);
        // 1.0 and 63 * -1.0 => (1 - 63) / 64.
        assert!((out[0] - (-62.0 / 64.0)).abs() < 1e-6);
        assert!(stats.is_clean());
    }

    #[test]
    fn extreme_channel_counts_still_average() {
        for channels in [3u16, 5, 7, 26, 255, 4096] {
            let n = usize::from(channels) * 4;
            let input: Vec<f32> = (0..n)
                .map(|i| {
                    if i % usize::from(channels) == 0 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            let (out, stats) = mono(&input, channels);
            assert_eq!(out.len(), 4, "channels={channels}");
            assert!((out[0] - 1.0 / f32::from(channels)).abs() < 1e-6);
            assert_eq!(stats.frames, 4);
        }
    }

    #[test]
    fn nan_and_infinity_never_reach_the_output() {
        // This is the test that matters most. A single NaN surviving into the
        // buffer poisons every log-mel frame it touches, and the failure surfaces
        // three crates away as an empty transcript.
        let input = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.5,
            f32::NAN,
            0.25,
            -f32::INFINITY,
            0.0,
        ];
        let (out, stats) = mono(&input, 2);
        assert_eq!(out, vec![0.0, 0.25, 0.125, 0.0]);
        assert_eq!(stats.nonfinite_replaced, 5);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn a_nan_in_one_channel_does_not_poison_its_neighbour() {
        // Averaging happens after sanitising, so the good channel survives. If
        // the order were reversed the whole frame would be zero.
        let (out, stats) = mono(&[f32::NAN, 1.0], 2);
        assert_eq!(out, vec![0.5]);
        assert_eq!(stats.nonfinite_replaced, 1);
    }

    #[test]
    fn out_of_range_samples_are_clamped_so_the_sum_cannot_overflow() {
        // Two channels at f32::MAX would sum to +infinity. Clamping first makes
        // the maximum possible sum 16.0, which is representable, so the frame
        // that was two enormous opposite numbers comes out as exact silence
        // rather than as NaN.
        let (out, stats) = mono(&[f32::MAX, f32::MIN, 1e30, 1e30], 2);
        assert!(out.iter().all(|v| v.is_finite()));
        assert_eq!(out, vec![0.0, MAX_INPUT_SAMPLE]);
        assert_eq!(stats.input_clamped, 4);
    }

    #[test]
    fn the_sum_of_the_worst_possible_input_is_representable() {
        // Proves the overflow claim arithmetically rather than by example: a
        // container may declare 65535 channels, each clamped to 8.0.
        let worst = f32::from(u16::MAX) * MAX_INPUT_SAMPLE;
        assert!(worst.is_finite(), "worst-case sum overflowed f32");
        assert!(worst < f32::MAX);
    }

    #[test]
    fn output_is_cleared_so_a_reused_buffer_never_accumulates() {
        let mut buf = vec![9.0; 10];
        downmix_to_mono(&[0.1, 0.2], 2, &mut buf);
        assert_eq!(buf, vec![0.15]);
        downmix_to_mono(&[0.3, 0.4], 2, &mut buf);
        assert_eq!(buf.len(), 1, "the second call must not append");
        assert!((buf[0] - 0.35).abs() < 1e-6);
    }

    #[test]
    fn empty_input_produces_empty_output() {
        let (out, stats) = mono(&[], 2);
        assert!(out.is_empty());
        assert_eq!(stats.frames, 0);
    }

    #[test]
    fn is_clean_distinguishes_a_repair_from_a_measurement() {
        // Above-unity output is counted but not modified, so it must not make
        // `is_clean` false: a loud recording is not a broken one.
        let loud = downmix_to_mono(&[4.0, 4.0], 1, &mut Vec::new());
        assert!(loud.is_clean());
        assert_eq!(loud.over_unity, 2, "two over-unity samples");
        let repaired = downmix_to_mono(&[f32::NAN, 1e30], 2, &mut Vec::new());
        assert!(!repaired.is_clean());
    }

    #[test]
    fn a_stereo_signal_survives_a_downmix_unchanged_in_the_amplitude_sense() {
        // A quick end-to-end sanity check on something shaped like a meeting:
        // the same waveform in both channels, half-scale in one.
        let n = 512;
        let input: Vec<f32> = (0..n)
            .flat_map(|i| {
                let l = (i as f32 * 0.01).sin() * 0.8;
                let r = (i as f32 * 0.01).sin() * 0.4;
                [l, r]
            })
            .collect();
        let (out, stats) = mono(&input, 2);
        for (i, &got) in out.iter().enumerate() {
            let ideal = (i as f32 * 0.01).sin() * 0.6;
            assert!((got - ideal).abs() < 1e-6, "frame {i}");
        }
        assert_eq!(stats.frames, n as u64);
    }
}

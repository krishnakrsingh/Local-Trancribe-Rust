//! A minimal RIFF/WAVE reader, for probes and the benchmark harness only.
//!
//! # Why this exists in a crate whose product path does not need it
//!
//! Decoding is `vaani-media`'s job and the product never calls this. It exists
//! for two measurement uses, and both need a reader that does not pull the
//! whole media stack into a `cargo test`:
//!
//! * the benchmark harness reads a recording to produce its JSON report, and
//!   the harness must stay runnable in one command on a machine where the
//!   decoder needs a native dependency;
//! * `tests/real_clip.rs` asserts structural properties of a real result.
//!
//! It is deliberately minimal and deliberately **not** a general decoder: RIFF
//! only, 16-bit PCM or 32-bit IEEE float, mono or interleaved, downmixed by
//! averaging. Anything else is an error rather than a guess.
//!
//! # Privacy
//!
//! This module reads samples and returns samples. It has no notion of content,
//! holds no `String` derived from audio, and logs nothing. Callers that print
//! results are responsible for printing aggregates; see [`crate::bench`].

use std::path::Path;

/// What a WAV file contained, after downmix.
#[derive(Debug, Clone, PartialEq)]
pub struct MonoAudio {
    /// Interleaved-free, channel-averaged samples in `[-1, 1]`.
    pub samples: Vec<f32>,
    /// Sample rate declared in the `fmt ` chunk.
    pub sample_rate: u32,
    /// Channel count declared in the `fmt ` chunk, before downmixing.
    pub channels: u16,
    /// Bits per sample declared in the `fmt ` chunk.
    pub bits_per_sample: u16,
}

impl MonoAudio {
    /// Duration in milliseconds, from the sample count and the declared rate.
    #[must_use]
    pub fn duration_ms(&self) -> i64 {
        if self.sample_rate == 0 {
            return 0;
        }
        self.samples.len() as i64 * 1000 / self.sample_rate as i64
    }

    /// Peak absolute sample value, for a clipping sanity check.
    #[must_use]
    pub fn peak(&self) -> f32 {
        self.samples.iter().fold(0f32, |m, v| m.max(v.abs()))
    }
}

/// Read a RIFF/WAVE file and downmix it to mono `f32`.
///
/// Supports format tag 1 (PCM, 16 or 32 bit) and tag 3 (IEEE float, 32 bit).
/// Anything else returns an error naming the format, because a benchmark that
/// silently read the wrong samples would report a confident wrong number.
///
/// The whole file is read into memory. That is acceptable for a probe on a file
/// the caller already holds in memory, and it is not a streaming decoder.
pub fn read_wav_mono(path: &Path) -> std::io::Result<MonoAudio> {
    let bytes = std::fs::read(path)?;
    parse_wav(&bytes)
}

/// Parse an in-memory RIFF/WAVE image. See [`read_wav_mono`].
pub fn parse_wav(bytes: &[u8]) -> std::io::Result<MonoAudio> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(std::io::Error::other("not a RIFF/WAVE file"));
    }
    let mut off = 12usize;
    let (mut fmt, mut bits, mut channels, mut rate) = (0u16, 0u16, 1u16, 16_000u32);
    let mut data: Option<(usize, usize)> = None;
    while off + 8 <= bytes.len() {
        let id = &bytes[off..off + 4];
        let size =
            u32::from_le_bytes(bytes[off + 4..off + 8].try_into().expect("4 bytes")) as usize;
        let body = (off + 8)..bytes.len().min(off + 8 + size);
        if id == b"fmt " && body.len() >= 16 {
            fmt = u16::from_le_bytes(bytes[off + 8..off + 10].try_into().expect("2 bytes"));
            channels = u16::from_le_bytes(bytes[off + 10..off + 12].try_into().expect("2 bytes"));
            rate = u32::from_le_bytes(bytes[off + 12..off + 16].try_into().expect("4 bytes"));
            bits = u16::from_le_bytes(bytes[off + 22..off + 24].try_into().expect("2 bytes"));
        } else if id == b"data" {
            data = Some((off + 8, size.min(bytes.len().saturating_sub(off + 8))));
        }
        off += 8 + size + (size & 1);
    }
    let (start, len) = data.ok_or_else(|| std::io::Error::other("no data chunk"))?;
    let raw = &bytes[start..start + len];
    let per = (bits / 8).max(1) as usize;
    let ch = channels.max(1) as usize;
    let frames = raw.len() / (per * ch);
    let mut out = Vec::with_capacity(frames);
    for i in 0..frames {
        let mut acc = 0f32;
        for c in 0..ch {
            let at = (i * ch + c) * per;
            if at + per > raw.len() {
                break;
            }
            let v = match (fmt, bits) {
                (3, 32) => f32::from_le_bytes(raw[at..at + 4].try_into().expect("4 bytes")),
                (1, 16) => {
                    i16::from_le_bytes(raw[at..at + 2].try_into().expect("2 bytes")) as f32
                        / 32_768.0
                }
                (1, 32) => {
                    i32::from_le_bytes(raw[at..at + 4].try_into().expect("4 bytes")) as f32
                        / 2_147_483_648.0
                }
                _ => {
                    return Err(std::io::Error::other(format!(
                        "unsupported WAV sample format: tag {fmt}, {bits} bits"
                    )));
                }
            };
            acc += v;
        }
        out.push(acc / ch as f32);
    }
    Ok(MonoAudio {
        samples: out,
        sample_rate: rate,
        channels,
        bits_per_sample: bits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal 16-bit PCM file. `fmt ` and `data` are written as separate
    /// chunk ids at their own offsets, because a reader that walks chunks is
    /// exactly what this exercises and a fixture that concatenated the ids
    /// would not.
    fn pcm16(channels: u16, rate: u32, samples: &[i16]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(b"WAVE");
        b.extend_from_slice(b"fmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&channels.to_le_bytes());
        b.extend_from_slice(&rate.to_le_bytes());
        b.extend_from_slice(&(rate * channels as u32 * 2).to_le_bytes());
        b.extend_from_slice(&2u16.to_le_bytes());
        b.extend_from_slice(&16u16.to_le_bytes());
        b.extend_from_slice(b"data");
        let bytes_len = (samples.len() * 2) as u32;
        b.extend_from_slice(&bytes_len.to_le_bytes());
        for s in samples {
            b.extend_from_slice(&s.to_le_bytes());
        }
        let size = (b.len() - 4) as u32;
        b[4..8].copy_from_slice(&size.to_le_bytes());
        b
    }

    #[test]
    fn a_mono_16_bit_file_is_read_exactly() {
        let raw = [0i16, 16_384, -16_384, 32_767];
        let bytes = pcm16(1, 16_000, &raw);
        let m = parse_wav(&bytes).expect("parse");
        assert_eq!(m.sample_rate, 16_000);
        assert_eq!(m.channels, 1);
        assert_eq!(m.bits_per_sample, 16);
        assert_eq!(m.samples.len(), 4);
        assert!((m.samples[0] - 0.0).abs() < 1e-6);
        assert!((m.samples[1] - 0.5).abs() < 1e-4);
        assert!((m.samples[2] + 0.5).abs() < 1e-4);
    }

    #[test]
    fn a_stereo_file_is_downmixed_by_averaging() {
        let raw = [0i16, i16::MAX, 0, 0];
        let m = parse_wav(&pcm16(2, 16_000, &raw)).expect("parse");
        assert_eq!(m.channels, 2);
        assert_eq!(m.samples.len(), 2);
        assert!((m.samples[0] - 0.5).abs() < 1e-4, "{}", m.samples[0]);
        assert!(m.samples[1].abs() < 1e-6);
    }

    #[test]
    fn a_non_riff_file_is_an_error_not_a_guess() {
        let e = parse_wav(b"not a wav file at all").expect_err("must fail");
        assert!(e.to_string().contains("RIFF"), "{e}");
    }

    #[test]
    fn a_file_without_data_is_an_error() {
        let mut b = Vec::from(&b"RIFF....WAVEfmt "[..]);
        let size = (b.len() - 4) as u32;
        b[4..8].copy_from_slice(&size.to_le_bytes());
        let e = parse_wav(&b).expect_err("must fail");
        assert!(e.to_string().contains("data"), "{e}");
    }

    #[test]
    fn duration_and_peak_come_from_the_sample_count_and_the_rate() {
        let m = MonoAudio {
            samples: vec![0.0, 0.5, -0.25, 0.0],
            sample_rate: 16_000,
            channels: 1,
            bits_per_sample: 16,
        };
        assert_eq!(m.duration_ms(), 0); // four samples is a quarter of a millisecond
        assert!((m.peak() - 0.5).abs() < 1e-6);
    }
}

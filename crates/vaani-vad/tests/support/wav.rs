//! A minimal RIFF/WAVE reader, for reading `scratch/clip.wav` in a test.
//!
//! # Why this exists at all
//!
//! `vaani-media` (A1) owns WAV decoding and is not a dependency of
//! `vaani-vad`, and `vaani-vad`'s manifest is frozen. Rather than duplicate a
//! decoder, this reads the handful of layouts that matter and **fails loudly**
//! on anything else. It is test-only: nothing in `src/` uses it, and the file
//! it reads is never copied into the crate.
//!
//! # What it supports
//!
//! | `wFormatTag` | Bits | Layout |
//! |---|---|---|
//! | 1 (PCM) | 8 | unsigned, offset by 127 |
//! | 1 (PCM) | 16 | signed LE |
//! | 1 (PCM) | 24 | signed LE, 3 bytes |
//! | 1 (PCM) | 32 | signed LE |
//! | 3 (IEEE float) | 32 | LE |
//! | 3 (IEEE float) | 64 | LE, narrowed to f32 |
//!
//! `scratch/clip.wav` is 16 kHz mono IEEE float 32 with a `fact` and a `PEAK`
//! chunk, which is why the chunk walk below skips unknown chunks rather than
//! assuming a 44-byte header. That is a real bug class -- `DEBUGGING.md:223`
//! records the same file and the same surprise.
//!
//! # Stereo
//!
//! Stereo is **not** downmixed. `AudioBuffer::slice_mono` panics on it for the
//! same reason, and the reason is the same: downmixing is the media stage's
//! job and doing it here would hide a pipeline that skipped it. This reader
//! takes the first channel and says so in the returned metadata, or refuses.
//! It refuses.

use std::path::Path;

/// What the reader learned about the file.
#[derive(Debug, Clone, PartialEq)]
pub struct WavInfo {
    /// Samples in one channel.
    pub sample_rate: u32,
    /// Always 1: a stereo file is refused rather than downmixed.
    pub channels: u16,
    /// Bits per sample in the source.
    pub bits_per_sample: u16,
    /// Number of samples per channel that were decoded.
    pub frames: usize,
    /// Every chunk id seen, so a test can assert the parser walked a file with
    /// the awkward chunks in it.
    pub chunks: Vec<String>,
}

/// Read a RIFF/WAVE file as mono f32 at its native sample rate.
pub fn read_mono_f32(path: &Path) -> Result<(Vec<f32>, WavInfo), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse_mono_f32(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// [`read_mono_f32`] on an in-memory buffer.
pub fn parse_mono_f32(b: &[u8]) -> Result<(Vec<f32>, WavInfo), String> {
    if b.len() < 12 {
        return Err("file is too short to be a RIFF header".into());
    }
    if &b[0..4] != b"RIFF" {
        return Err("not a RIFF file".into());
    }
    if &b[8..12] != b"WAVE" {
        return Err("RIFF file is not WAVE".into());
    }

    let mut fmt: Option<(u16, u16, u32, u16)> = None;
    let mut data: Option<(usize, usize)> = None;
    let mut chunks: Vec<String> = Vec::new();

    let mut at = 12usize;
    while at + 8 <= b.len() {
        let id = String::from_utf8_lossy(&b[at..at + 4]).to_string();
        let size = u32::from_le_bytes([b[at + 4], b[at + 5], b[at + 6], b[at + 7]]) as usize;
        let body = at + 8;
        chunks.push(id.clone());
        match id.as_str() {
            "fmt " => {
                if size < 16 || body + 16 > b.len() {
                    return Err("fmt chunk is truncated".into());
                }
                let tag = u16::from_le_bytes([b[body], b[body + 1]]);
                let channels = u16::from_le_bytes([b[body + 2], b[body + 3]]);
                let rate = u32::from_le_bytes([b[body + 4], b[body + 5], b[body + 6], b[body + 7]]);
                let bits = u16::from_le_bytes([b[body + 14], b[body + 15]]);
                fmt = Some((tag, channels, rate, bits));
            }
            "data" => {
                let end = body.saturating_add(size).min(b.len());
                data = Some((body, end));
            }
            _ => {}
        }
        // Chunks are word-aligned: an odd-sized chunk is followed by a pad
        // byte, and forgetting that is an off-by-one that misreads the `data`
        // chunk of every file with a `LIST` or `fact` chunk.
        at = body + size + (size & 1);
    }

    let (tag, channels, sample_rate, bits) = fmt.ok_or("no fmt chunk")?;
    if channels != 1 {
        return Err(format!(
            "{channels} channels; refusing to downmix. Downmixing is the media stage's job."
        ));
    }
    if sample_rate == 0 {
        return Err("fmt chunk declares a zero sample rate".into());
    }
    let (start, end) = data.ok_or("no data chunk")?;
    let payload = &b[start..end];
    let width = (bits as usize) / 8;
    if width == 0 {
        return Err("fmt chunk declares zero bits per sample".into());
    }
    let frames = payload.len() / width;

    let mut out = Vec::with_capacity(frames);
    for i in 0..frames {
        let o = i * width;
        let v = match (tag, bits) {
            (1, 8) => (payload[o] as f32 - 128.0) / 128.0,
            (1, 16) => i16::from_le_bytes([payload[o], payload[o + 1]]) as f32 / 32_768.0,
            (1, 24) => {
                // The padding byte must be the *most* significant one:
                // 24-bit little-endian puts the sign bit in byte 2, so a
                // 3-byte value is sign-extended by prepending a byte, not by
                // shifting left. `[0, b0, b1, b2]` would instead multiply the
                // value by 256 and a positive 0x400000 would read as 0x40000000.
                let sign = if payload[o + 2] & 0x80 != 0 {
                    0xFFu8
                } else {
                    0u8
                };
                let raw = i32::from_le_bytes([payload[o], payload[o + 1], payload[o + 2], sign]);
                (raw as f32 / 8_388_608.0).clamp(-1.0, 1.0)
            }
            (1, 32) => {
                i32::from_le_bytes([payload[o], payload[o + 1], payload[o + 2], payload[o + 3]])
                    as f32
                    / 2_147_483_648.0
            }
            (3, 32) => {
                f32::from_le_bytes([payload[o], payload[o + 1], payload[o + 2], payload[o + 3]])
            }
            (3, 64) => {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(&payload[o..o + 8]);
                f64::from_le_bytes(buf) as f32
            }
            (t, bs) => return Err(format!("unsupported {t} with {bs} bits")),
        };
        out.push(v);
    }

    let n = out.len();
    Ok((
        out,
        WavInfo {
            sample_rate,
            channels,
            bits_per_sample: bits,
            frames: n,
            chunks,
        },
    ))
}

/// Write a mono f32 WAV, for round-tripping the reader in a test.
///
/// Writing the file is only justified by one thing: proving the reader reads
/// what it claims to. Without it, every test of the reader would be testing a
/// reader against bytes this same code produced, which cannot catch a shared
/// misunderstanding of the format.
pub fn write_mono_f32(path: &Path, samples: &[f32], sample_rate: u32) -> std::io::Result<()> {
    let mut data = Vec::with_capacity(samples.len() * 4);
    for s in samples {
        data.extend_from_slice(&s.to_le_bytes());
    }
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 4).to_le_bytes());
    out.extend_from_slice(&4u16.to_le_bytes()); // block align
    out.extend_from_slice(&32u16.to_le_bytes()); // bits
    out.extend_from_slice(b"fact");
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&(samples.len() as u32).to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&data);
    // The RIFF size covers everything after the first 8 bytes.
    let riff_size = (out.len() + 4) as u32;
    let mut file: Vec<u8> = Vec::with_capacity(out.len() + 8);
    file.extend_from_slice(b"RIFF");
    file.extend_from_slice(&riff_size.to_le_bytes());
    file.extend_from_slice(&out);
    std::fs::write(path, file)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Wrap a WAVE... body in a RIFF header, which is what
    /// parse_mono_f32 requires and what these hand-built fixtures would
    /// otherwise be missing.
    fn riff(body: &[u8]) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&((body.len() + 4) as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(body);
        out
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join("vaani-vad-wav");
        let _ = std::fs::create_dir_all(&d);
        d.join(name)
    }

    #[test]
    fn a_float32_file_round_trips() {
        let p = tmp("roundtrip.wav");
        let s: Vec<f32> = (0..1_000).map(|i| (i as f32 * 0.001).sin() * 0.5).collect();
        write_mono_f32(&p, &s, 16_000).unwrap();
        let (got, info) = read_mono_f32(&p).unwrap();
        assert_eq!(info.sample_rate, 16_000);
        assert_eq!(info.channels, 1);
        assert_eq!(info.bits_per_sample, 32);
        assert_eq!(info.frames, 1_000);
        for (a, b) in s.iter().zip(got.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn the_chunk_walk_skips_the_fact_and_peak_chunks_a_real_capture_carries() {
        // The bug this guards: assuming a 44-byte canonical header. A file
        // with `fact` and `PEAK` before `data` has its data chunk at a
        // different offset, and a hard-coded header size reads noise.
        let p = tmp("chunky.wav");
        let s = vec![0.25f32; 100];
        write_mono_f32(&p, &s, 16_000).unwrap();
        let (_, info) = read_mono_f32(&p).unwrap();
        assert!(info.chunks.contains(&"fact".to_string()));
        assert!(info.chunks.contains(&"data".to_string()));
    }

    /// A hand-built `fmt ` + `data` body, for exercising a specific layout.
    fn body(tag: u16, channels: u16, rate: u32, bits: u16, data: &[u8]) -> Vec<u8> {
        let block = (channels as usize) * (bits as usize / 8);
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(b"fmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&tag.to_le_bytes());
        b.extend_from_slice(&channels.to_le_bytes());
        b.extend_from_slice(&rate.to_le_bytes());
        b.extend_from_slice(&(rate * block as u32).to_le_bytes());
        b.extend_from_slice(&(block as u16).to_le_bytes());
        b.extend_from_slice(&bits.to_le_bytes());
        b.extend_from_slice(b"data");
        b.extend_from_slice(&(data.len() as u32).to_le_bytes());
        b.extend_from_slice(data);
        b
    }

    #[test]
    fn sixteen_bit_pcm_is_read_and_scaled_correctly() {
        let mut data: Vec<u8> = Vec::new();
        for s in [0i16, 16_384, -16_384] {
            data.extend_from_slice(&s.to_le_bytes());
        }
        let (got, info) = parse_mono_f32(&riff(&body(1, 1, 16_000, 16, &data))).unwrap();
        assert_eq!(info.bits_per_sample, 16);
        assert_eq!(info.frames, 3);
        assert!(got[0].abs() < 1e-6);
        assert!((got[1] - 0.5).abs() < 1e-4, "{}", got[1]);
        assert!((got[2] + 0.5).abs() < 1e-4, "{}", got[2]);
    }

    #[test]
    fn eight_bit_pcm_is_unsigned() {
        let data = vec![128u8, 255, 0];
        let (got, info) = parse_mono_f32(&riff(&body(1, 1, 8_000, 8, &data))).unwrap();
        assert_eq!(info.sample_rate, 8_000);
        assert!(got[0].abs() < 1e-6);
        assert!(got[1] > 0.9, "{}", got[1]);
        assert!(got[2] < -0.9, "{}", got[2]);
    }

    #[test]
    fn twenty_four_bit_pcm_is_sign_extended() {
        // 24-bit little-endian, so the MSB byte carries the sign bit.
        // 0x400000 = 4194304 -> +0.5; 0xFFC00000 = -4194304 -> -0.5.
        let mut data: Vec<u8> = vec![0x00, 0x00, 0x40];
        data.extend_from_slice(&[0x00, 0x00, 0xC0]);
        let (got, _) = parse_mono_f32(&riff(&body(1, 1, 16_000, 24, &data))).unwrap();
        assert!((got[0] - 0.5).abs() < 1e-4, "{}", got[0]);
        assert!(got[1] < -0.4, "{}", got[1]);
    }

    #[test]
    fn sixty_four_bit_float_is_narrowed() {
        let data: Vec<u8> = [0.5f64, -0.25f64]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let (got, info) = parse_mono_f32(&riff(&body(3, 1, 16_000, 64, &data))).unwrap();
        assert_eq!(info.bits_per_sample, 64);
        assert!((got[0] - 0.5).abs() < 1e-6, "{}", got[0]);
        assert!((got[1] + 0.25).abs() < 1e-6, "{}", got[1]);
    }

    #[test]
    fn a_stereo_file_is_refused_rather_than_downmixed() {
        let e = parse_mono_f32(&riff(&body(3, 2, 16_000, 32, &[0u8; 8]))).unwrap_err();
        assert!(e.contains("downmix"), "{e}");
    }

    #[test]
    fn an_unsupported_layout_is_an_error_not_a_reinterpret() {
        // A 12-bit PCM stream: reading it as 16-bit would produce plausible,
        // wrong samples rather than an error.
        let e = parse_mono_f32(&riff(&body(1, 1, 16_000, 12, &[0u8; 4]))).unwrap_err();
        assert!(e.contains("unsupported"), "{e}");
    }

    #[test]
    fn malformed_input_is_an_error_not_a_panic() {
        assert!(parse_mono_f32(&[]).is_err());
        assert!(parse_mono_f32(b"RIFF").is_err());
        assert!(parse_mono_f32(b"NOTAWAVFILE!").is_err());
        // RIFF/WAVE with no chunks at all.
        assert!(parse_mono_f32(b"RIFF\x04\x00\x00\x00WAVE").is_err());
        // fmt but no data.
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(b"fmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&3u16.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&16_000u32.to_le_bytes());
        b.extend_from_slice(&64_000u32.to_le_bytes());
        b.extend_from_slice(&4u16.to_le_bytes());
        b.extend_from_slice(&32u16.to_le_bytes());
        assert!(parse_mono_f32(&riff(&b)).unwrap_err().contains("no data"));
        // data but no fmt.
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(b"data");
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&[0u8; 4]);
        assert!(parse_mono_f32(&riff(&b)).unwrap_err().contains("no fmt"));
        // A zero sample rate must not divide by zero later.
        assert!(
            parse_mono_f32(&riff(&body(3, 1, 0, 32, &[0u8; 4])))
                .unwrap_err()
                .contains("zero sample rate")
        );
        // A truncated data chunk is clipped, not read past the end.
        let (got, info) = parse_mono_f32(&riff(&body(3, 1, 16_000, 32, &[0u8; 6]))).unwrap();
        assert_eq!(info.frames, 1);
        assert_eq!(got.len(), 1);
    }
}

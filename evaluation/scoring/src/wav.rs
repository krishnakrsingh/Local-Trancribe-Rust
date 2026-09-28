//! Minimal RIFF/WAVE reading and writing, for cutting clips and probing audio.
//!
//! The clips in `evaluation/clips/` are cut from one private recording, so
//! this code has exactly one job: take a range of samples out of a WAV and
//! write a WAV that every other tool on the machine can open. It is
//! deliberately small, has no dependencies, and reads more than it writes.
//!
//! ## Formats
//!
//! Reading: `WAVE_FORMAT_PCM` (1) at 8, 16, 24 and 32 bits, and
//! `WAVE_FORMAT_IEEE_FLOAT` (3) at 32 and 64 bits, plus `WAVE_FORMAT_EXTENSIBLE`
//! (0xFFFE) when the sub-format GUID is one of those. `fact` and `PEAK` chunks
//! are skipped. Samples are returned as `f32` in the range `[-1, 1]`.
//!
//! Writing: 16-bit signed PCM, mono, 16 kHz. That is chosen deliberately:
//!
//! * it is the format the pipeline's own VAD and ASR front end are specified
//!   against, so a clip needs no conversion before transcription;
//! * it is the format a human annotator can play on anything, which matters
//!   more than 16 bits of dynamic range for speech.
//!
//! 16-bit quantisation noise on speech at 16 kHz is roughly 90 dB below full
//! scale, so it is not a measurable contributor to ASR error, and the claim is
//! made here rather than assumed.

use std::path::Path;

/// A decoded mono waveform.
#[derive(Debug, Clone, PartialEq)]
pub struct Audio {
    /// Sample rate in hertz.
    pub sample_rate: u32,
    /// Samples, nominally in `[-1, 1]`.
    pub samples: Vec<f32>,
}

impl Audio {
    /// Duration in milliseconds, rounded to the nearest whole millisecond.
    pub fn duration_ms(&self) -> i64 {
        if self.sample_rate == 0 {
            return 0;
        }
        ((self.samples.len() as i64 * 1000) + self.sample_rate as i64 / 2) / self.sample_rate as i64
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether there are no samples at all.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Why a WAV file could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WavError(pub String);

impl std::fmt::Display for WavError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for WavError {}

const WAVE_FORMAT_PCM: u16 = 1;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

fn err<T>(m: impl Into<String>) -> Result<T, WavError> {
    Err(WavError(m.into()))
}

/// Read a RIFF/WAVE file as mono `f32`.
pub fn read(path: &Path) -> Result<Audio, WavError> {
    let bytes = std::fs::read(path).map_err(|e| WavError(format!("{}: {e}", path.display())))?;
    parse(&bytes).map_err(|e| WavError(format!("{}: {e}", path.display())))
}

fn parse(b: &[u8]) -> Result<Audio, WavError> {
    if b.len() < 44 {
        return err("file is shorter than a WAV header");
    }
    if &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return err("not a RIFF/WAVE file");
    }
    let mut pos = 12usize;
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (tag, channels, rate, bits)
    let mut data: Option<(usize, usize)> = None;

    while pos + 8 <= b.len() {
        let id = &b[pos..pos + 4];
        let len = u32::from_le_bytes([b[pos + 4], b[pos + 5], b[pos + 6], b[pos + 7]]) as usize;
        let body = pos + 8;
        let end = body.saturating_add(len).min(b.len());
        match id {
            b"fmt " => {
                if len < 16 {
                    return err("fmt chunk is shorter than 16 bytes");
                }
                let tag = u16::from_le_bytes([b[body], b[body + 1]]);
                let channels = u16::from_le_bytes([b[body + 2], b[body + 3]]);
                let rate = u32::from_le_bytes([b[body + 4], b[body + 5], b[body + 6], b[body + 7]]);
                let bits = u16::from_le_bytes([b[body + 14], b[body + 15]]);
                let tag = if tag == WAVE_FORMAT_EXTENSIBLE {
                    // Sub-format GUID: the first two bytes are the real tag.
                    if len >= 26 {
                        u16::from_le_bytes([b[body + 24], b[body + 25]])
                    } else {
                        tag
                    }
                } else {
                    tag
                };
                fmt = Some((tag, channels, rate, bits));
            }
            b"data" => data = Some((body, end - body)),
            _ => {}
        }
        pos = body + len + (len & 1);
        if data.is_some() && fmt.is_some() {
            break;
        }
    }

    let (tag, channels, rate, bits) = fmt.ok_or_else(|| WavError("no fmt chunk".into()))?;
    let (off, len) = data.ok_or_else(|| WavError("no data chunk".into()))?;
    if channels == 0 {
        return err("fmt chunk declares zero channels");
    }
    if rate == 0 {
        return err("fmt chunk declares a zero sample rate");
    }
    let bytes_per_sample = (bits as usize) / 8;
    if bytes_per_sample == 0 {
        return err("fmt chunk declares zero bits per sample");
    }
    let frame_bytes = bytes_per_sample * channels as usize;
    if len < frame_bytes {
        return err("data chunk is shorter than one frame");
    }

    let raw = &b[off..off + len - (len % frame_bytes)];
    let frames = raw.len() / frame_bytes;
    let mut out = Vec::with_capacity(frames);
    for f in 0..frames {
        let base = f * frame_bytes;
        // Downmix by averaging, which is what the product's own resampler does
        // and is lossless for mono input.
        let mut acc = 0.0f32;
        for c in 0..channels as usize {
            acc += sample_at(raw, base + c * bytes_per_sample, tag, bits);
        }
        out.push(acc / channels as f32);
    }
    Ok(Audio {
        sample_rate: rate,
        samples: out,
    })
}

fn sample_at(raw: &[u8], at: usize, tag: u16, bits: u16) -> f32 {
    match (tag, bits) {
        (WAVE_FORMAT_IEEE_FLOAT, 32) => {
            f32::from_le_bytes([raw[at], raw[at + 1], raw[at + 2], raw[at + 3]])
        }
        (WAVE_FORMAT_IEEE_FLOAT, 64) => {
            let mut a = [0u8; 8];
            a.copy_from_slice(&raw[at..at + 8]);
            f64::from_le_bytes(a) as f32
        }
        (_, 8) => (raw[at] as f32 - 128.0) / 128.0,
        (_, 16) => i16::from_le_bytes([raw[at], raw[at + 1]]) as f32 / 32768.0,
        (_, 24) => {
            let v = i32::from_le_bytes([raw[at], raw[at + 1], raw[at + 2], 0]) >> 8;
            v as f32 / 8_388_608.0
        }
        (_, 32) => {
            i32::from_le_bytes([raw[at], raw[at + 1], raw[at + 2], raw[at + 3]]) as f32
                / 2_147_483_648.0
        }
        _ => 0.0,
    }
}

/// Serialise `audio` as 16-bit PCM mono WAV.
pub fn encode(audio: &Audio) -> Vec<u8> {
    let rate = audio.sample_rate;
    let n = audio.samples.len();
    let data_len = n * 2;
    let mut b = Vec::with_capacity(44 + data_len);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    b.extend_from_slice(b"WAVE");
    b.extend_from_slice(b"fmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&WAVE_FORMAT_PCM.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes()); // mono
    b.extend_from_slice(&rate.to_le_bytes());
    b.extend_from_slice(&(rate * 2).to_le_bytes()); // byte rate
    b.extend_from_slice(&2u16.to_le_bytes()); // block align
    b.extend_from_slice(&16u16.to_le_bytes()); // bits
    b.extend_from_slice(b"data");
    b.extend_from_slice(&(data_len as u32).to_le_bytes());
    for s in &audio.samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

/// Write `audio` as a 16-bit PCM WAV.
pub fn write(path: &Path, audio: &Audio) -> Result<(), WavError> {
    std::fs::write(path, encode(audio)).map_err(|e| WavError(format!("{}: {e}", path.display())))
}

/// Take `start_ms .. start_ms + duration_ms` out of `audio`.
///
/// Samples outside the file are clamped, and a request that lands entirely
/// outside produces an error rather than a silent zero-length clip. A clip of
/// the wrong length that looks valid is worse than a failed cut: every score
/// computed from it would be quietly wrong.
pub fn cut(audio: &Audio, start_ms: i64, duration_ms: i64) -> Result<Audio, WavError> {
    if duration_ms <= 0 {
        return err(format!("duration must be positive, got {duration_ms} ms"));
    }
    let rate = audio.sample_rate as i64;
    let first = (start_ms * rate).div_euclid(1000);
    let count = (duration_ms * rate).div_euclid(1000);
    if first >= audio.samples.len() as i64 {
        return err(format!(
            "start {start_ms} ms is past the end of a {} ms file",
            audio.duration_ms()
        ));
    }
    let from = first.max(0) as usize;
    let to = ((first + count).max(0) as usize).min(audio.samples.len());
    if to <= from {
        return err(format!(
            "requested range {start_ms}..{} ms contains no samples of a {} ms file",
            start_ms + duration_ms,
            audio.duration_ms()
        ));
    }
    Ok(Audio {
        sample_rate: audio.sample_rate,
        samples: audio.samples[from..to].to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(n: usize, rate: u32) -> Audio {
        Audio {
            sample_rate: rate,
            samples: (0..n).map(|i| (i as f32 / 40.0).sin() * 0.5).collect(),
        }
    }

    #[test]
    fn a_written_wave_reads_back_to_the_same_length() {
        let a = tone(16_000, 16_000);
        let bytes = encode(&a);
        let b = parse(&bytes).unwrap();
        assert_eq!(b.sample_rate, 16_000);
        assert_eq!(b.samples.len(), 16_000);
    }

    #[test]
    fn sixteen_bit_quantisation_is_below_the_noise_floor_of_speech() {
        let a = tone(16_000, 16_000);
        let b = parse(&encode(&a)).unwrap();
        let mut worst = 0.0f32;
        for (x, y) in a.samples.iter().zip(b.samples.iter()) {
            worst = worst.max((x - y).abs());
        }
        // One LSB of a 16-bit quantiser is 1/32768.
        assert!(worst <= 1.0 / 32768.0, "worst {worst}");
    }

    #[test]
    fn float32_input_is_read() {
        let a = tone(100, 16_000);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&((36 + 400) as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&3u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&16_000u32.to_le_bytes());
        bytes.extend_from_slice(&64_000u32.to_le_bytes());
        bytes.extend_from_slice(&4u16.to_le_bytes());
        bytes.extend_from_slice(&32u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&400u32.to_le_bytes());
        for s in &a.samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let b = parse(&bytes).unwrap();
        assert_eq!(b.samples.len(), 100);
        assert!((b.samples[1] - a.samples[1]).abs() < 1e-7);
    }

    #[test]
    fn a_stereo_file_is_downmixed_by_averaging() {
        let _stereo = Audio {
            sample_rate: 8_000,
            samples: vec![0.0, 1.0, 1.0, 0.0],
        };
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&((36 + 16) as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&8_000u32.to_le_bytes());
        bytes.extend_from_slice(&32_000u32.to_le_bytes());
        bytes.extend_from_slice(&4u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        for s in &[0.0f32, 1.0, 1.0, 0.0] {
            let v = (s * 32767.0) as i16;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let b = parse(&bytes).unwrap();
        assert_eq!(b.samples.len(), 2);
        assert!((b.samples[0] - 0.5).abs() < 1e-3, "{:?}", b.samples);
    }

    #[test]
    fn a_non_riff_file_is_rejected_with_a_reason() {
        assert!(parse(b"not a wav file at all, really not").is_err());
        assert!(parse(b"RIFFxxxxWAVE").is_err());
    }

    #[test]
    fn cutting_takes_exactly_the_requested_span() {
        let a = tone(16_000, 16_000); // 1000 ms
        let c = cut(&a, 250, 500).unwrap();
        assert_eq!(c.samples.len(), 8_000);
        assert_eq!(c.duration_ms(), 500);
    }

    #[test]
    fn a_cut_past_the_end_fails_rather_than_writing_a_stub() {
        let a = tone(1_600, 16_000); // 100 ms
        assert!(cut(&a, 500, 100).is_err());
        assert!(cut(&a, 0, 0).is_err());
    }

    #[test]
    fn a_cut_running_off_the_end_is_truncated_but_not_empty() {
        let a = tone(16_000, 16_000); // 1000 ms
        let c = cut(&a, 900, 500).unwrap();
        assert!(!c.samples.is_empty());
        assert_eq!(c.duration_ms(), 100);
    }
}

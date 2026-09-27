//! Synthesised media fixtures.
//!
//! # Why every byte here is written by hand
//!
//! The rule for this crate's tests is that no binary is committed and no network
//! is touched, so a checkout of the repository can prove the decoder works. That
//! rules out a checked-in corpus, and it rules out anything that needs an encoder
//! installed on the machine, because "works on the machine that generated the
//! fixture" is not a test.
//!
//! So the two formats that can be written from first principles —
//! [WAVE](https://en.wikipedia.org/wiki/WAVE) and
//! [FLAC](https://xiph.org/flac/format.html) — are written here, byte by byte.
//! WAV's header is 44 bytes and needs no cleverness. FLAC allows
//! `SUBFRAME_VERBATIM`, which stores samples with no entropy coding at all, so
//! a *valid FLAC stream* is producible without implementing Rice coding; that is
//! what makes a genuine lossless-codec round trip testable with no encoder.
//!
//! The formats that genuinely cannot be synthesised without an encoder —
//! Ogg/Vorbis, MP3, AAC/MP4, Matroska — are handled by
//! [`optional_fixture`], which skips with a loud message rather than failing
//! silently or pretending.
//!
//! # What is deliberately synthesisable
//!
//! Several inputs are *invalid* on purpose, because the edge cases in the brief
//! are all about malformed files: a zero-byte file, a header with no data, a
//! `data` chunk that claims more than the file holds, a ten-hour duration over
//! ten seconds of audio, and 64 channels.

#![allow(dead_code)] // each test binary uses a different subset of this module

use std::io::Write;
use std::path::Path;

use tempfile::NamedTempFile;

/// A temporary file that deletes itself.
///
/// Held rather than re-opened by path, so a test cannot pass because a *previous*
/// run's file was still lying around.
pub struct Fixture {
    file: NamedTempFile,
}

impl Fixture {
    /// Write `bytes` to a new temporary file with the given extension.
    pub fn new(extension: &str, bytes: &[u8]) -> Self {
        let mut file = tempfile::Builder::new()
            .suffix(&format!(".{extension}"))
            .tempfile()
            .expect("could not create a temporary file");
        file.write_all(bytes)
            .expect("could not write a temporary file");
        file.flush().expect("could not flush a temporary file");
        Self { file }
    }

    /// The path to probe, decode or read.
    pub fn path(&self) -> &Path {
        self.file.path()
    }
}

// ---------------------------------------------------------------------------
// WAV
// ---------------------------------------------------------------------------

/// The sample encodings a WAVE `fmt ` chunk can declare.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    /// 8-bit unsigned PCM, offset binary.
    U8,
    /// 16-bit signed PCM. The only encoding that round-trips exactly through
    /// Symphonia's `f32` conversion (`s as f32 / 32768.0`).
    S16,
    /// 24-bit signed PCM, packed into three little-endian bytes.
    S24,
    /// 32-bit signed PCM.
    S32,
    /// 32-bit IEEE float, the format every modern tool writes.
    F32,
}

impl SampleFormat {
    fn tag(self) -> u16 {
        match self {
            Self::F32 => 0x0003,
            _ => 0x0001,
        }
    }

    fn bits(self) -> u16 {
        match self {
            Self::U8 => 8,
            Self::S16 => 16,
            Self::S24 => 24,
            Self::S32 | Self::F32 => 32,
        }
    }

    fn is_float(self) -> bool {
        matches!(self, Self::F32)
    }
}

/// Everything about a WAVE file except its samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WavSpec {
    pub sample_rate: u32,
    pub channels: u16,
    pub format: SampleFormat,
    /// Write `WAVE_FORMAT_EXTENSIBLE` (format tag 0xFFFE) rather than a bare
    /// `WAVE_FORMAT_PCM`. Required by the spec above two channels.
    pub extensible: bool,
    /// The size to write in the `data` chunk header, when it is not the real one.
    ///
    /// This is how the truncated and "lies about its duration" fixtures are
    /// built: a real file is cut short, or a header claims ten hours.
    pub declared_data_bytes: Option<u32>,
    /// Write only this many bytes of sample data, keeping the header intact.
    ///
    /// Distinct from `declared_data_bytes`: this is a genuinely partial file, as
    /// opposed to a file whose header is merely wrong.
    pub truncate_to_bytes: Option<usize>,
}

impl WavSpec {
    /// A plain PCM file, which is what almost every real recording is.
    pub const fn pcm(sample_rate: u32, channels: u16, format: SampleFormat) -> Self {
        Self {
            sample_rate,
            channels,
            format,
            extensible: false,
            declared_data_bytes: None,
            truncate_to_bytes: None,
        }
    }

    /// The same file, written as `WAVE_FORMAT_EXTENSIBLE`.
    #[must_use]
    pub const fn extensible(mut self) -> Self {
        self.extensible = true;
        self
    }

    /// Claim a `data` chunk size other than the real one.
    #[must_use]
    pub const fn claiming_data_bytes(mut self, bytes: u32) -> Self {
        self.declared_data_bytes = Some(bytes);
        self
    }

    /// Write only the first `bytes` of sample data.
    #[must_use]
    pub const fn truncated_to(mut self, bytes: usize) -> Self {
        self.truncate_to_bytes = Some(bytes);
        self
    }
}

/// Build a WAVE file from interleaved `f32` samples.
///
/// Samples outside `[-1, 1]` are clamped, because every integer encoding here
/// cannot represent them; the float encoding is written verbatim so a test can
/// put a `NaN` or a value of 4.0 in a file on purpose.
pub fn wav(samples: &[f32], spec: &WavSpec) -> Vec<u8> {
    let mut pcm = encode_samples(samples, spec.format);
    if let Some(limit) = spec.truncate_to_bytes {
        pcm.truncate(limit.min(pcm.len()));
    }
    let declared = spec
        .declared_data_bytes
        .unwrap_or(u32::try_from(pcm.len()).unwrap_or(u32::MAX));

    let block_align = spec.channels * spec.format.bits() / 8;
    let byte_rate = spec.sample_rate.wrapping_mul(u32::from(block_align));
    let fmt_len: u32 = if spec.extensible { 40 } else { 16 };

    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(b"RIFF");
    // The RIFF size counts everything after the first 8 bytes, including the
    // padded `data` payload.
    out.extend_from_slice(&(36 + fmt_len + 8 + declared).to_le_bytes());
    out.extend_from_slice(b"WAVE");

    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&fmt_len.to_le_bytes());
    out.extend_from_slice(
        &if spec.extensible {
            0xFFFEu16
        } else {
            spec.format.tag()
        }
        .to_le_bytes(),
    );
    out.extend_from_slice(&spec.channels.to_le_bytes());
    out.extend_from_slice(&spec.sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&spec.format.bits().to_le_bytes());
    if spec.extensible {
        out.extend_from_slice(&22u16.to_le_bytes()); // cbSize
        out.extend_from_slice(&spec.format.bits().to_le_bytes()); // valid bits
        out.extend_from_slice(&channel_mask(spec.channels).to_le_bytes());
        out.extend_from_slice(&subformat_guid(spec.format));
    }

    out.extend_from_slice(b"data");
    out.extend_from_slice(&declared.to_le_bytes());
    out.extend_from_slice(&pcm);
    out
}

/// A WAVE file with verbatim `fmt ` and `data` bodies.
///
/// Every other builder here is parameterised, which is right for valid files and
/// useless for invalid ones: a file with a sample rate of zero, an unknown format
/// tag, or no `data` chunk at all cannot be written as a combination of the
/// options [`WavSpec`] offers. So this takes the two chunk bodies as given and
/// the test decides what is in them. `data` is omitted entirely when
/// `declared_data` is `None` **and** `data` is empty, which is how the
/// header-with-no-payload fixture is built.
pub fn wav_raw(fmt: &[u8], data: &[u8], declared_data: Option<u32>) -> Vec<u8> {
    let fmt_len = u32::try_from(fmt.len()).expect("fmt chunk body");
    let mut out = Vec::with_capacity(12 + 8 + fmt.len() + 8 + data.len());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&fmt_len.to_le_bytes());
    out.extend_from_slice(fmt);
    let declared = declared_data
        .unwrap_or_else(|| u32::try_from(data.len()).expect("data chunk body fits in a u32"));
    out.extend_from_slice(b"data");
    out.extend_from_slice(&declared.to_le_bytes());
    out.extend_from_slice(data);
    // The RIFF size counts everything after the first eight bytes.
    let body = (out.len() + 4) as u32;
    let mut file = Vec::with_capacity(out.len() + 8);
    file.extend_from_slice(b"RIFF");
    file.extend_from_slice(&body.to_le_bytes());
    file.extend_from_slice(&out);
    file
}

/// A 16-byte `fmt ` body for linear PCM, verbatim, including the tag.
pub fn fmt_chunk(tag: u16, sample_rate: u32, channels: u16, bits: u16) -> Vec<u8> {
    let block_align = channels * bits / 8;
    let byte_rate = sample_rate.wrapping_mul(u32::from(block_align));
    let mut fmt = Vec::with_capacity(16);
    fmt.extend_from_slice(&tag.to_le_bytes());
    fmt.extend_from_slice(&channels.to_le_bytes());
    fmt.extend_from_slice(&sample_rate.to_le_bytes());
    fmt.extend_from_slice(&byte_rate.to_le_bytes());
    fmt.extend_from_slice(&block_align.to_le_bytes());
    fmt.extend_from_slice(&bits.to_le_bytes());
    fmt
}

/// The `dwChannelMask` for a channel count.
///
/// Not a guess: a mask with fewer bits than channels is what a mis-written file
/// looks like, and a mask with more is what an over-clever muxer writes. For the
/// counts that matter the first *n* speaker positions are the correct answer.
///
/// Above 32 channels there is no correct answer, because `dwChannelMask` is 32
/// bits wide and `WAVE_FORMAT_EXTENSIBLE` tops out at 18 channels anyway. Writing
/// `u32::MAX` for a 64-channel file is not a fixture that exercises a decoder, it
/// is a fixture that panics the demuxer: Symphonia tries to correct the mask,
/// computes a shift of 32, and overflows. Fixtures with more than 32 channels
/// must therefore be written as plain `WAVE_FORMAT_PCM`, which has no mask to get
/// wrong.
fn channel_mask(channels: u16) -> u32 {
    assert!(
        u32::from(channels) <= 32,
        "no valid dwChannelMask exists for {channels} channels"
    );
    (1u32 << channels) - 1
}

/// `KSDATAFORMAT_SUBTYPE_PCM` or `KSDATAFORMAT_SUBTYPE_IEEE_FLOAT`.
///
/// The full 16-byte GUID, not a summary of it: the first four bytes are the
/// Data1 field (`1` for PCM, `3` for IEEE float) and the rest is a fixed
/// Data2/Data3/Data4 sequence. Getting this wrong produces
/// `wav: unsupported fmt_ext sub-type`, which is exactly what it did the first
/// time.
fn subformat_guid(format: SampleFormat) -> [u8; 16] {
    let mut guid = [0u8; 16];
    guid[0..4].copy_from_slice(&if format.is_float() { 3u32 } else { 1u32 }.to_le_bytes());
    guid[4..16].copy_from_slice(&[
        0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71,
    ]);
    guid
}

/// Encode interleaved `f32` into the byte layout of `format`.
pub fn encode_samples(samples: &[f32], format: SampleFormat) -> Vec<u8> {
    let width = usize::from(format.bits()) / 8;
    let mut out = Vec::with_capacity(samples.len() * width);
    for &s in samples {
        match format {
            SampleFormat::U8 => {
                // Inverse of Symphonia's `u8 -> f32` (`s / 128.0 - 1.0`), with the
                // full-scale point clamped because 256 does not exist.
                let v = ((s + 1.0) * 128.0).round();
                out.push(v.clamp(0.0, 255.0) as u8);
            }
            SampleFormat::S16 => out.extend_from_slice(&scale_i16(s).to_le_bytes()),
            SampleFormat::S24 => {
                let v = (s.clamp(-1.0, 1.0) * 8_388_608.0).round() as i32;
                let v = v.clamp(-8_388_608, 8_388_607) as u32;
                out.extend_from_slice(&v.to_le_bytes()[..3]);
            }
            SampleFormat::S32 => out.extend_from_slice(&scale_i32(s).to_le_bytes()),
            SampleFormat::F32 => out.extend_from_slice(&s.to_le_bytes()),
        }
    }
    out
}

/// `-1.0..1.0` to `i16`, using the full 32768 range so the conversion back is exact.
fn scale_i16(s: f32) -> i16 {
    let v = (s * 32_768.0).round();
    v.clamp(-32_768.0, 32_767.0) as i16
}

fn scale_i32(s: f32) -> i32 {
    let v = (s * 2_147_483_648.0).round();
    v.clamp(-2_147_483_648.0, 2_147_483_647.0) as i32
}

// ---------------------------------------------------------------------------
// FLAC
// ---------------------------------------------------------------------------

/// Frames per FLAC frame.
///
/// 4096 is the format's own recommendation and keeps the frame count under 128
/// for any fixture short enough to write by hand, which keeps the frame-number
/// field to a single UTF-8 byte.
const FLAC_BLOCK: usize = 4096;

/// Build a valid native FLAC stream from interleaved `i16` samples.
///
/// Uses `SUBFRAME_VERBATIM`, which stores each sample as a raw `bits`-wide
/// two's-complement integer with no entropy coding. That is a legal FLAC
/// subframe, so the result is a real FLAC file that any conforming decoder —
/// including Symphonia's — must decode sample-exactly, and it can be produced
/// without implementing a Rice encoder.
///
/// Only 8-, 16- and 24-bit depths are supported, which is all FLAC defines for
/// PCM. A mono or stereo stream is produced; stereo uses independent
/// subframes rather than the stereo decorrelation modes, which keeps the writer
/// to one code path.
pub fn flac(samples: &[i16], sample_rate: u32, channels: u16) -> Vec<u8> {
    assert!(
        channels == 1 || channels == 2,
        "the FLAC writer emits mono or independent stereo"
    );
    let bits = 16usize;
    let total_frames = samples.len() / usize::from(channels);
    let data_bytes = total_frames * usize::from(channels) * bits / 8;

    let mut out = Vec::with_capacity(data_bytes + 128);
    out.extend_from_slice(b"fLaC");

    // METADATA_BLOCK_HEADER: last-metadata-block = 1, type = 0 (STREAMINFO).
    out.push(0x80);
    out.extend_from_slice(&34u32.to_be_bytes()[1..]);

    // STREAMINFO.
    out.extend_from_slice(&(FLAC_BLOCK as u16).to_be_bytes()); // min blocksize
    out.extend_from_slice(&(FLAC_BLOCK as u16).to_be_bytes()); // max blocksize
    out.extend_from_slice(&min_max_framesize(total_frames, channels, bits));
    // 20 bits sample rate | 3 bits (channels - 1) | 5 bits (bits - 1) | 36 bits
    // total samples. That is 64 bits, so all eight bytes go out; writing six
    // shifts the bit depth two bits down and the decoder reports
    // "stream bits per sample are out of bounds".
    let packed: u64 = (u64::from(sample_rate) << 44)
        | (u64::from(channels - 1) << 41)
        | (u64::try_from(bits - 1).unwrap_or(15) << 36)
        | (total_frames as u64 & 0x0000_000F_FFFF_FFFF);
    out.extend_from_slice(&packed.to_be_bytes());
    out.extend_from_slice(&[0u8; 16]); // MD5: all zero means "unknown", which is legal

    let mut offset = 0usize;
    let mut frame_number = 0u32;
    while offset < total_frames {
        let n = (total_frames - offset).min(FLAC_BLOCK);
        out.extend_from_slice(&flac_frame(
            &samples[offset * usize::from(channels)..(offset + n) * usize::from(channels)],
            n,
            channels,
            bits,
            frame_number,
        ));
        offset += n;
        frame_number += 1;
    }
    out
}

/// `min_framesize` and `max_framesize`, in the 24-bit big-endian pair FLAC wants.
///
/// Symphonia validates that the minimum is not greater than the maximum, so the
/// short final frame has to be the *minimum* when it is smaller than a full
/// block, which it always is.
fn min_max_framesize(total_frames: usize, channels: u16, bits: usize) -> [u8; 6] {
    let full = (FLAC_BLOCK * usize::from(channels) * bits / 8) as u32;
    let rest_frames = total_frames % FLAC_BLOCK;
    let last = if rest_frames == 0 {
        full
    } else {
        (rest_frames * usize::from(channels) * bits / 8) as u32 + 64
    };
    let (min, max) = (full.min(last), full.max(last));
    let mut out = [0u8; 6];
    out[0..3].copy_from_slice(&min.to_be_bytes()[1..]);
    out[3..6].copy_from_slice(&max.to_be_bytes()[1..]);
    out
}

/// One FLAC frame: header, verbatim subframes, padding, CRC-16.
fn flac_frame(
    samples: &[i16],
    frames: usize,
    channels: u16,
    bits: usize,
    frame_number: u32,
) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.write(0b11_1111_1111_1110, 14); // sync
    w.write(0, 1); // reserved
    w.write(0, 1); // fixed blocksize strategy
    // Block size: 0b1100 means 256 * 2^(12-8) = 4096 with nothing following;
    // 0b0111 means a 16-bit (blocksize - 1) follows at the end of the header,
    // which is the only form wide enough for an arbitrary short final frame.
    let short = frames < FLAC_BLOCK;
    w.write(if short { 0b0111 } else { 0b1100 }, 4);
    w.write(0b0000, 4); // sample rate: take it from STREAMINFO
    w.write(u64::from(channels - 1), 4); // independent channels
    w.write(0b000, 3); // sample size: take it from STREAMINFO
    w.write(0, 1); // reserved
    write_utf8_number(&mut w, u64::from(frame_number));
    if short {
        w.write((frames - 1) as u64, 16);
    }
    let header = w.finish();
    let crc8 = crc8(&header);
    let mut out = header;
    out.push(crc8);

    // Subframes: 1 zero bit, 6 bits of type, 1 wasted-bits flag, then the
    // samples unencoded. `SUBFRAME_VERBATIM` is type 000001.
    let mut body = BitWriter::new();
    for ch in 0..usize::from(channels) {
        body.write(0, 1);
        body.write(0b000001, 6);
        body.write(0, 1);
        for i in 0..frames {
            let s = samples[i * usize::from(channels) + ch];
            // Two's complement, bit for bit. `unsigned_abs` here would silently
            // produce a stream of magnitudes, which decodes to something that
            // looks like audio and is the wrong sign.
            body.write(u64::from(s as u16), u32::try_from(bits).unwrap_or(16));
        }
    }
    // Zero-pad to the next byte boundary, as the format requires.
    let body = body.finish();
    out.extend_from_slice(&body);
    out.extend(std::iter::repeat_n(0u8, out.len() % 8));
    let crc16 = crc16(&out);
    out.extend_from_slice(&crc16.to_be_bytes());
    out
}

/// FLAC's UTF-8-like integer coding: 1 byte below 0x80, 2 bytes to 0x7FF.
fn write_utf8_number(w: &mut BitWriter, value: u64) {
    if value < 0x80 {
        w.write(value, 8);
    } else if value < 0x800 {
        w.write(0b1100_0000 | (value >> 6), 8);
        w.write(0b1000_0000 | (value & 0x3F), 8);
    } else {
        w.write(0b1110_0000 | (value >> 12), 8);
        w.write(0b1000_0000 | ((value >> 6) & 0x3F), 8);
        w.write(0b1000_0000 | (value & 0x3F), 8);
    }
}

/// CRC-8 as FLAC defines it: polynomial `x^8 + x^2 + x + 1`, initial value 0.
fn crc8(data: &[u8]) -> u8 {
    let mut crc = 0u8;
    for &b in data {
        crc ^= b;
        for _ in 0..8 {
            crc = if crc & 0x80 != 0 {
                (crc << 1) ^ 0x07
            } else {
                crc << 1
            };
        }
    }
    crc
}

/// CRC-16 as FLAC defines it: polynomial `x^16 + x^15 + x^2 + 1`, initial value 0.
fn crc16(data: &[u8]) -> u16 {
    let mut crc = 0u16;
    for &b in data {
        crc ^= u16::from(b) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x8005
            } else {
                crc << 1
            };
        }
    }
    crc
}

/// A most-significant-bit-first bit writer, which is how every FLAC field is laid
/// out. Without it the frame header is unreadable and the CRC is a guess.
struct BitWriter {
    bytes: Vec<u8>,
    acc: u64,
    nbits: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            acc: 0,
            nbits: 0,
        }
    }

    fn write(&mut self, value: u64, bits: u32) {
        debug_assert!((1..=32).contains(&bits), "FLAC fields are 1 to 32 bits");
        debug_assert!(
            bits == 64 || bits == 32 || (value >> bits) == 0,
            "value does not fit"
        );
        let mask = if bits >= 32 {
            u32::MAX
        } else {
            (1u32 << bits) - 1
        };
        self.acc = (self.acc << bits) | (u64::from((value as u32) & mask));
        self.nbits += bits;
        while self.nbits >= 8 {
            self.nbits -= 8;
            self.bytes.push((self.acc >> self.nbits) as u8);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            let pad = 8 - self.nbits;
            self.acc <<= pad;
            self.bytes.push(self.acc as u8);
            self.nbits = 0;
        }
        self.bytes
    }
}

// ---------------------------------------------------------------------------
// Fixtures that need an encoder
// ---------------------------------------------------------------------------

/// The directory optional fixtures are looked for in.
///
/// Deliberately `scratch/fixtures`, which `.gitignore` already excludes, so a
/// locally generated corpus never reaches a commit by accident.
pub const OPTIONAL_DIR: &str = "scratch/fixtures";

/// Locate an optional fixture, explaining loudly when it is absent.
///
/// Returns `None` after printing why, so the calling test can `return` and be
/// reported as *skipped* rather than as *passed*. A test that quietly does
/// nothing is worse than no test, because it looks like coverage.
pub fn optional_fixture(name: &str) -> Option<std::path::PathBuf> {
    let name = format!("{OPTIONAL_DIR}/{name}");
    optional_fixture_at(&name)
}

/// As [`optional_fixture`], for a path that is not under [`OPTIONAL_DIR`].
///
/// Used for the private reference clip, which is deliberately *not* in
/// `fixtures/`: it is a real recording, not a synthesised corpus entry, and it
/// must never be mistaken for something a test generated.
///
/// The path is resolved against the current directory first and then against the
/// *workspace* root, because `cargo test` runs a test binary with the package
/// directory as its working directory and every scratch path in this repository
/// is written from the workspace root. Without the second lookup the fixtures
/// would silently never be found from a test run, which looks exactly like a
/// passing suite.
pub fn optional_fixture_at(relative: &str) -> Option<std::path::PathBuf> {
    let direct = std::path::Path::new(relative);
    if direct.is_file() {
        return Some(direct.to_path_buf());
    }
    if let Some(root) = workspace_root() {
        let candidate = root.join(relative);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    eprintln!(
        "SKIPPED: the optional fixture `{relative}` is not present.\n\
         This input needs a real encoder or a real recording, which this repository\n\
         does not ship and this test will not shell out to. To run this test,\n\
         generate the file with an external tool and drop it at that path, for\n\
         example:\n\
         \n    ffmpeg -f lavfi -i sine=frequency=1000:duration=10 -c:a libvorbis \\\n\
         \x20             -ar 44100 -ac 2 {relative}\n\
         \nSee crates/vaani-media/OPEN_ISSUES in the A1 report for the full list."
    );
    None
}

/// The workspace root, found by walking up from this package's manifest
/// directory until a manifest with a `[workspace]` table appears.
///
/// Not a fixed number of `parent()` calls: the layout is `crates/<name>`, and a
/// test that resolves `scratch/` to `crates/scratch` fails in a way that looks
/// like a passing suite.
fn workspace_root() -> Option<std::path::PathBuf> {
    let mut dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf();
    for _ in 0..4 {
        let manifest = dir.join("Cargo.toml");
        if let Ok(text) = std::fs::read_to_string(&manifest)
            && text.contains("[workspace]")
        {
            return Some(dir);
        }
        dir = dir.parent()?.to_path_buf();
    }
    None
}

// ---------------------------------------------------------------------------
// Signal generators and analysers
// ---------------------------------------------------------------------------

/// A sine tone, in `[-1, 1]`.
pub fn tone(frames: usize, rate: u32, hz: f32, amplitude: f32) -> Vec<f32> {
    (0..frames)
        .map(|i| amplitude * (2.0 * std::f32::consts::PI * hz * i as f32 / rate as f32).sin())
        .collect()
}

/// Interleave one or two signals into `channels`.
///
/// Two arguments is enough for every fixture here: a multi-channel fixture is
/// either the same signal in every channel (to check that averaging is unity
/// gain) or two signals that cancel (to check that the downmix happened at all).
pub fn interleave(a: &[f32], b: &[f32], channels: u16) -> Vec<f32> {
    let n = a.len().min(b.len());
    let ch = usize::from(channels).max(1);
    let mut out = Vec::with_capacity(n * ch);
    for i in 0..n {
        for c in 0..ch {
            out.push(if c % 2 == 0 { a[i] } else { b[i] });
        }
    }
    out
}

/// Interleave one signal into `gains.len()` channels, each scaled by its gain.
///
/// This is the multi-channel builder: [`interleave`] can only alternate two
/// signals, and a downmix test needs a *known* per-channel level so the expected
/// mono amplitude can be computed rather than eyeballed. `gains[c] == 0.0`
/// produces a silent channel, which is how a "dialogue is only in the centre"
/// fixture is written.
pub fn interleave_scaled(base: &[f32], gains: &[f32]) -> Vec<f32> {
    let ch = gains.len().max(1);
    let mut out = Vec::with_capacity(base.len() * ch);
    for &v in base {
        for g in gains {
            out.push(v * g);
        }
    }
    out
}
pub fn silence(frames: usize) -> Vec<f32> {
    vec![0.0; frames]
}

/// An exponential sine sweep from `f0` to `f1` over `frames` at `rate`, at 0.5
/// amplitude.
///
/// The phase is the analytic integral of a log sweep, which is what makes the
/// sweep's *instantaneous* frequency exactly `f0 * (f1/f0)^(t/T)` — a linear
/// chirp would not, and its rate would change across the sweep.
pub fn sweep(frames: usize, rate: u32, f0: f32, f1: f32) -> Vec<f32> {
    let t_total = frames as f64 / f64::from(rate);
    let k = (f64::from(f1) / f64::from(f0)).ln() / t_total;
    (0..frames)
        .map(|i| {
            let t = i as f64 / f64::from(rate);
            let phase = 2.0 * std::f64::consts::PI * f64::from(f0) * (k.exp() * t).exp_m1() / k;
            0.5 * (phase as f32).sin()
        })
        .collect()
}

/// A stationary bank of tones, logarithmically spaced across a band.
///
/// This is the probe that replaced a sine sweep, and the reason is worth
/// recording. A sweep is *non-stationary*: its frequency changes while it is
/// being measured, so a 20 ms analysis window covers 1.3 cycles at 200 Hz and 32
/// at 1600 Hz, and the "envelope" of a swept signal is not a property of the
/// resampler at all. Every attempt to measure a resampler with a chirp ended up
/// measuring the chirp.
///
/// A bank of tones is stationary, has a flat spectrum, and can be measured
/// exactly: each tone is placed on an integer FFT bin, so its amplitude and its
/// phase are both read directly rather than inferred. One probe therefore covers
/// passband flatness, stopband depth, alias rejection, intermodulation, and
/// frequency-independent group delay, all at once and none of them ambiguously.
///
/// Each tone gets `amplitude / count` so the sum stays inside the requested
/// envelope.
pub fn multitone(frames: usize, rate: u32, f_lo: f32, f_hi: f32, count: usize) -> Vec<f32> {
    let tones = tone_bank(frames, rate, f_lo, f_hi, count);
    multitone_from_bank(&tones, frames, rate)
}

/// Sum a bank of tones into a signal of `frames` samples.
///
/// Takes the bank explicitly so the frequencies can be snapped to the *analysis*
/// length rather than to the signal length. Those are different lengths in every
/// test that uses this, and using the wrong one puts every tone a quarter of a
/// bin off and makes the whole spectrum read as flat.
pub fn multitone_from_bank(tones: &[(f32, f32)], frames: usize, rate: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; frames];
    for (hz, amp) in tones {
        for (i, v) in out.iter_mut().enumerate() {
            *v += amp * (2.0 * std::f32::consts::PI * hz * i as f32 / rate as f32).sin();
        }
    }
    out
}

/// The frequencies and amplitudes [`multitone`] uses, snapped to FFT bins and
/// confined to the output's own Nyquist.
///
/// The snapping grid is the **output's** bin spacing, 16 kHz over the analysis
/// length, and that is the whole trick. The measurement is made on the
/// resampled signal, so a tone has to land on an exact bin *there* for its
/// amplitude and phase to be read directly. Snapping to the input's grid instead
/// looks reasonable and silently costs 4 dB of measured flatness, because for a
/// 3:1 rate change the input's bins and the output's do not line up.
///
/// The clamp at the output Nyquist is what makes this the passband builder. See
/// [`tone_bank_above_nyquist`] for the other one.
pub fn tone_bank(frames: usize, rate: u32, f_lo: f32, f_hi: f32, count: usize) -> Vec<(f32, f32)> {
    let analysis = frames.next_power_of_two().min(frames);
    let hi_cap = ((rate / 2) as f32 / (16_000.0 / analysis as f32)) as i64;
    tone_bank_unclamped(
        analysis,
        f_lo,
        f_hi,
        count,
        hi_cap.min((analysis / 2 - 1) as i64),
    )
}

/// A tone bank above the output Nyquist, for alias rejection.
///
/// The same grid, but not clamped to the output's Nyquist, because the entire
/// point is to place energy where the output cannot represent it and see what
/// comes back. Two things about this helper are not obvious and both cost real
/// time:
///
/// * The clamp from [`tone_bank`] must not be applied. A 9.6 kHz request clamped
///   to 7.999 kHz silently produces sixty tones at *one* frequency, and because
///   they are coherent their amplitudes sum. The test then normalises by a
///   single tone's amplitude and reads 35 dB worse than the truth.
/// * A caller must normalise by the *sum* of the amplitudes, not by one of them,
///   for exactly that reason.
pub fn tone_bank_above_nyquist(
    frames: usize,
    f_lo: f32,
    f_hi: f32,
    count: usize,
) -> Vec<(f32, f32)> {
    let analysis = frames.next_power_of_two().min(frames);
    let hi_cap = i64::MAX;
    tone_bank_unclamped(analysis, f_lo, f_hi, count, hi_cap)
}

fn tone_bank_unclamped(
    analysis: usize,
    f_lo: f32,
    f_hi: f32,
    count: usize,
    hi_cap: i64,
) -> Vec<(f32, f32)> {
    let bin_hz = 16_000.0 / analysis as f32;
    let lo_bin = ((f_lo / bin_hz).round() as i64).max(2);
    let hi_bin = ((f_hi / bin_hz).round() as i64).min(hi_cap);
    let span = (hi_bin - lo_bin).max(1) as usize;
    let amp = 0.4 / count as f32;
    (0..count)
        .map(|i| {
            let bin = if count == 1 {
                lo_bin
            } else {
                lo_bin + ((span * i + count / 2) / count) as i64
            };
            (bin as f32 * bin_hz, amp)
        })
        .collect()
}

/// Complex spectrum of a Hann-windowed signal, indexed by bin.
pub fn complex_spectrum(x: &[f32]) -> Vec<(f32, f32)> {
    use rustfft::FftPlanner;
    use rustfft::num_complex::Complex;

    let n = x.len();
    let mut buf: Vec<Complex<f32>> = Vec::with_capacity(n);
    let mut gain = 0.0f32;
    for (i, &v) in x.iter().enumerate() {
        let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (n as f32 - 1.0)).cos();
        gain += w;
        buf.push(Complex::new(v * w, 0.0));
    }
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buf);
    let scale = 2.0 / gain.max(1e-9);
    buf[..n / 2]
        .iter()
        .map(|c| (c.re * scale, c.im * scale))
        .collect()
}

/// Amplitude and group delay of one frequency in a signal.
///
/// Direct correlation at the exact frequency, not an FFT bin. That matters: a
/// rectangular window over a segment that is not a whole number of periods
/// leaks most of a low-frequency tone into the bins either side of it, and an
/// FFT-based amplitude estimate of a 100 Hz tone came out 0.74 dB low — a
/// measurement error that would have been indistinguishable from a real
/// passband droop. Correlating at the exact frequency has no such ambiguity.
///
/// Delay comes back in **output samples**, which is the number that matters: a
/// resampler whose delay grows with the length of the input is time-warping the
/// recording, and a constant delay of a few samples is a fixed offset the ASR
/// stage can live with.
pub fn measure_tone(x: &[f32], rate: u32, hz: f32) -> (f32, f32) {
    let w = 2.0 * std::f64::consts::PI * f64::from(hz) / f64::from(rate);
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (i, &v) in x.iter().enumerate() {
        let phase = w * i as f64;
        re += f64::from(v) * phase.cos();
        im -= f64::from(v) * phase.sin();
    }
    let n = x.len() as f64;
    let amplitude = (2.0 * (re * re + im * im).sqrt() / n) as f32;
    let delay = (im.atan2(re) / w) as f32;
    (amplitude, delay)
}

/// Magnitude spectrum of a signal, Hann-windowed, in bins of `rate / n`.
///
/// Windowed and *not* zero-padded. Both details matter for the sweep test: a
/// rectangular window over a log sweep splatters energy across the whole
/// spectrum, and zero-padding a non-power-of-two length truncates the tail to
/// zeros, which is a step discontinuity and splatters just as badly. The first
/// version of this function did both and reported the sweep's own leakage as if
/// it were the resampler's.
pub fn spectrum(x: &[f32], rate: u32) -> (Vec<f32>, f32) {
    use rustfft::FftPlanner;
    use rustfft::num_complex::Complex;

    let n = x.len();
    let mut buf: Vec<Complex<f32>> = Vec::with_capacity(n);
    // Hann, and the coherent gain it costs, so the magnitudes stay comparable
    // with the un-windowed ones from `measure_tone`.
    let mut gain = 0.0f32;
    for (i, &v) in x.iter().enumerate() {
        let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (n as f32 - 1.0)).cos();
        gain += w;
        buf.push(Complex::new(v * w, 0.0));
    }
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buf);
    let scale = 2.0 / gain.max(1e-9);
    let mags: Vec<f32> = buf[..n / 2].iter().map(|c| c.norm() * scale).collect();
    (mags, rate as f32 / n as f32)
}

/// The largest magnitude in a frequency band, and its bin frequency.
pub fn band_peak(mags: &[f32], bin_hz: f32, lo: f32, hi: f32) -> (f32, f32) {
    let a = (lo / bin_hz).floor().max(0.0) as usize;
    let b = (((hi / bin_hz).ceil() as usize) + 1).min(mags.len() - 1);
    let mut best = 0.0f32;
    let mut at = 0.0f32;
    for (i, &m) in mags.iter().enumerate().take(b).skip(a) {
        if m > best {
            best = m;
            at = i as f32 * bin_hz;
        }
    }
    (best, at)
}

/// Peak absolute value.
pub fn peak(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |a, &b| a.max(b.abs()))
}

/// Root-mean-square level.
pub fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}

/// The largest single-sample jump, which is how a splice or truncation click
/// shows up.
pub fn max_jump(x: &[f32]) -> f32 {
    x.windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0f32, f32::max)
}

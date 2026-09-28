//! Synthetic audio fixtures, written byte by byte.
//!
//! # Why this exists
//!
//! The QA harness must never commit media blobs and must never depend on the
//! private 12-minute meeting or on `scratch/clip.wav`. Every fixture here is
//! synthesised in-process: PCM S16 WAV files (the one container writable from
//! first principles with a 44-byte header) plus short helpers for tones,
//! silence, deterministic noise, speech-like bursts and two-speaker alternation.
//!
//! The committed regression fixtures under `tests/fixtures/qa/` are produced by
//! this same writer (see `scripts/qa/make_fixtures.rs` in the module docs of
//! the integration tests), so "committed" and "generated" can never disagree.

use std::path::{Path, PathBuf};

/// Sample rate every fixture is written at: the contract rate.
pub const FIXTURE_SAMPLE_RATE: u32 = 16_000;

/// Deterministic PRNG (SplitMix64): no `rand` dependency, reproducible noise.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    /// A generator for `seed`.
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    /// The next 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A float in `[-1.0, 1.0)`.
    pub fn next_sample(&mut self) -> f32 {
        (self.next_u64() as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
    }
}

/// Silence: `seconds` of zeros at 16 kHz mono.
pub fn silence(seconds: f64) -> Vec<f32> {
    vec![0.0; (seconds * f64::from(FIXTURE_SAMPLE_RATE)) as usize]
}

/// A pure tone at `freq_hz`, amplitude `peak` in `[0, 1]`.
pub fn tone(frames: usize, sample_rate: u32, freq_hz: f64, peak: f32) -> Vec<f32> {
    (0..frames)
        .map(|i| {
            let t = i as f64 / f64::from(sample_rate);
            (2.0 * std::f64::consts::PI * freq_hz * t).sin() as f32 * peak
        })
        .collect()
}

/// Deterministic white noise at `peak` amplitude.
pub fn noise(frames: usize, seed: u64, peak: f32) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..frames).map(|_| rng.next_sample() * peak).collect()
}

/// Speech-like bursts: 120 ms tone pips separated by 180 ms of silence, with a
/// slow amplitude wobble so an energy detector sees onsets rather than a drone.
///
/// Note: this simple material does NOT trigger the production energy VAD (its
/// steady-source veto rejects pure tones by design). Tests needing VAD-positive
/// audio must use [`voice`]; tests needing diarization-positive audio can use
/// either, since the diarizer carries its own activity detector.
pub fn speech_like(seconds: f64, sample_rate: u32, seed: u64) -> Vec<f32> {
    let frames = (seconds * f64::from(sample_rate)) as usize;
    let mut rng = Rng::new(seed);
    let base_freq = 140.0 + (rng.next_u64() % 120) as f64;
    let mut out = vec![0.0f32; frames];
    let burst = (0.12 * f64::from(sample_rate)) as usize;
    let gap = (0.18 * f64::from(sample_rate)) as usize;
    let mut i = 0;
    let mut k = 0usize;
    while i < frames {
        let freq = base_freq * (1.0 + 0.15 * ((k % 5) as f64 - 2.0) / 2.0);
        for j in 0..burst {
            if i + j >= frames {
                break;
            }
            let t = (i + j) as f64 / f64::from(sample_rate);
            let env = ((j as f64 / burst as f64) * std::f64::consts::PI).sin();
            out[i + j] = (2.0 * std::f64::consts::PI * freq * t).sin() as f32 * 0.5 * env as f32;
        }
        i += burst + gap;
        k += 1;
    }
    out
}

/// Two-speaker alternation: speaker A (low tone) and speaker B (high tone) take
/// `turn_seconds` turns, so diarization has something separable to find.
///
/// Note: pure tones by design do not trigger the production energy VAD. This
/// material is for the diarizer (own activity detector) and for media-path
/// tests, not for VAD-positive chains — see [`voice`] for those.
pub fn two_speaker_turns(
    turns_per_speaker: usize,
    turn_seconds: f64,
    sample_rate: u32,
) -> Vec<f32> {
    let turn_frames = (turn_seconds * f64::from(sample_rate)) as usize;
    let mut out = Vec::with_capacity(turn_frames * turns_per_speaker * 2);
    for turn in 0..turns_per_speaker * 2 {
        let freq = if turn % 2 == 0 { 130.0 } else { 320.0 };
        let offset = out.len();
        for i in 0..turn_frames {
            let t = (offset + i) as f64 / f64::from(sample_rate);
            let env = ((i as f64 / turn_frames as f64) * std::f64::consts::PI).sin();
            out.push((2.0 * std::f64::consts::PI * freq * t).sin() as f32 * 0.45 * env as f32);
        }
    }
    out
}

/// A speech-shaped voice: harmonic comb with moving fundamental, slow
/// syllabic amplitude envelope, jitter and fricative bursts.
///
/// A plain sine is rejected by the VAD's steady-source veto *by design*, and
/// white noise has no speech shape, so neither may stand in for "speech" in a
/// VAD-positive test. This generator earns each feature the detector looks
/// for: harmonics with a spectral tilt for tonality, a wandering `f0` plus
/// jitter so autocorrelation is nontrivial, a 3–6 Hz syllabic envelope for
/// onsets, and unvoiced frication on part of the cycle.
pub fn voice(frames: usize, sample_rate: u32, seed: u64, f0_hz: f32, level: f32) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    // A second stream so the fricative noise decorrelates from the formant.
    let mut rng_noise = Rng::new(seed ^ 0x5EED_0123_4567_89AB);
    let formant = 400.0 + (rng_noise.next_u64() % 500) as f32;
    let syl_rate = 3.0 + (rng.next_u64() % 3000) as f32 / 1000.0;
    let env_rate = 4.0 + (rng.next_u64() % 4000) as f32 / 1000.0;
    let mut out = vec![0.0f32; frames];
    for (i, slot) in out.iter_mut().enumerate() {
        let t = i as f32 / sample_rate as f32;
        let jitter = 1.0 + 0.012 * (2.0 * std::f32::consts::PI * 4.7 * t).sin();
        let f0 = f0_hz * jitter * (1.0 + 0.05 * (2.0 * std::f32::consts::PI * syl_rate * t).sin());
        let f1 = formant * (1.0 + 0.35 * (2.0 * std::f32::consts::PI * syl_rate * 0.6 * t).sin());
        let mut s = 0.0f32;
        for h in 1..=24 {
            let fh = f0 * h as f32;
            if fh > sample_rate as f32 / 2.0 - 100.0 {
                break;
            }
            let src = 1.0 / (h as f32).powf(0.7);
            let d = (fh - f1) / (f0 * 2.5 + 200.0);
            s += src * (1.0 / (1.0 + d * d)) * (2.0 * std::f32::consts::PI * fh * t).sin();
        }
        let unvoiced = if (2.0 * std::f32::consts::PI * syl_rate * t).sin() > 0.35 {
            1.0
        } else {
            0.0
        };
        let fric = unvoiced * 0.25 * rng_noise.next_sample();
        let env =
            0.35 + 0.65 * (0.5 + 0.5 * (2.0 * std::f32::consts::PI * env_rate * t).sin()).abs();
        *slot = level * env * (s * 0.06 + fric);
    }
    for s in out.iter_mut() {
        *s = s.clamp(-1.0, 1.0);
    }
    out
}

/// Two speech-shaped voices in alternation, for VAD-positive chains that also
/// need separable speakers.
pub fn two_voices_turns(
    turns_per_speaker: usize,
    turn_seconds: f64,
    sample_rate: u32,
    seed: u64,
) -> Vec<f32> {
    let turn_frames = (turn_seconds * f64::from(sample_rate)) as usize;
    let mut out = Vec::with_capacity(turn_frames * turns_per_speaker * 2);
    for turn in 0..turns_per_speaker * 2 {
        let f0 = if turn % 2 == 0 { 120.0 } else { 200.0 };
        out.extend(voice(turn_frames, sample_rate, seed + turn as u64, f0, 0.9));
    }
    out
}

/// Root-mean-square level of `samples`, in dB.
pub fn rms_db(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return -200.0;
    }
    let acc: f64 = samples.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    (10.0 * (acc / samples.len() as f64).max(1e-20).log10()) as f32
}

/// Scale `samples` in place so the achieved RMS equals `target_db`.
///
/// The harmonic comb's RMS depends on pitch, tilt and envelope (about 16 dB
/// below unit level here), so without this any stated level is a generator
/// parameter, not a measurement.
pub fn scale_to_db(samples: &mut [f32], target_db: f32) {
    let measured = rms_db(samples);
    let scale = 10f32.powf((target_db - measured) / 20.0);
    for x in samples.iter_mut() {
        *x *= scale;
    }
}

/// A room floor at −100 dB: a real capture's quietest moment is a room, never
/// digital silence. Gaps of mathematical zero break the VAD's adaptive noise
/// floor (its trailing-window percentile collapses to zero), so VAD-positive
/// fixtures are always laid over this.
pub fn room_floor(frames: usize, seed: u64) -> Vec<f32> {
    // Uniform[−1,1] has RMS 1/√3; scale to 10^(−100/20).
    let scale = 10f32.powf(-100.0 / 20.0) / 0.577_350_3;
    let mut rng = Rng::new(seed);
    (0..frames).map(|_| rng.next_sample() * scale).collect()
}

/// A VAD-positive meeting bed: room floor plus a continuous voice at −25 dB.
pub fn meeting(frames: usize, sample_rate: u32, seed: u64, f0_hz: f32) -> Vec<f32> {
    let mut out = room_floor(frames, seed ^ 0x5EED);
    let mut v = voice(frames, sample_rate, seed, f0_hz, 1.0);
    scale_to_db(&mut v, -25.0);
    for (o, x) in out.iter_mut().zip(v) {
        *o += x;
    }
    out
}

/// A VAD-positive two-voice meeting bed: alternating voices at −25 dB over a
/// room floor, with 400 ms room-only gaps between turns.
pub fn meeting_turns(
    turns_per_speaker: usize,
    turn_seconds: f64,
    sample_rate: u32,
    seed: u64,
) -> Vec<f32> {
    let turn_frames = (turn_seconds * f64::from(sample_rate)) as usize;
    let gap_frames = (0.4 * f64::from(sample_rate)) as usize;
    let total = (turn_frames + gap_frames) * turns_per_speaker * 2;
    let mut out = room_floor(total, seed ^ 0x5EED);
    for turn in 0..turns_per_speaker * 2 {
        let f0 = if turn % 2 == 0 { 120.0 } else { 200.0 };
        let mut v = voice(turn_frames, sample_rate, seed + turn as u64, f0, 1.0);
        scale_to_db(&mut v, -25.0);
        let at = turn * (turn_frames + gap_frames);
        for (i, x) in v.into_iter().enumerate() {
            out[at + i] += x;
        }
    }
    out
}

/// A long, cheap, speech-*free* bed for stages that need wall time rather than
/// speech: a room floor under a slow tremolo on a two-tone drone.
///
/// `voice()` is deliberately expensive — 24 harmonics per sample — so building a
/// 25-minute buffer from it would cost minutes of test time for no extra
/// coverage. The media stage does not run a VAD, so it does not need
/// speech-shaped audio; it needs a file long enough that a cancel can land
/// inside the decode. This generator is integer-and-`sin`-light and gets there
/// in about a second.
pub fn long_bed(seconds: u32, sample_rate: u32, seed: u64) -> Vec<f32> {
    let frames = seconds as usize * sample_rate as usize;
    let mut rng = Rng::new(seed);
    // A per-sample amplitude ramp on a fixed pair of partials, so the decoded
    // buffer is not trivially compressible and the downmixer has real work.
    let a = 90.0 + (rng.next_u64() % 40) as f64;
    let b = 197.0 + (rng.next_u64() % 60) as f64;
    let tremolo_hz = 3.0 + (rng.next_u64() % 200) as f64 / 100.0;
    (0..frames)
        .map(|i| {
            let t = i as f64 / f64::from(sample_rate);
            let env = 0.25 + 0.35 * (2.0 * std::f64::consts::PI * tremolo_hz * t).sin();
            let drone = (2.0 * std::f64::consts::PI * a * t).sin() * 0.6
                + (2.0 * std::f64::consts::PI * b * t).sin() * 0.3;
            (drone * env) as f32
        })
        .collect()
}

/// Interleave `channels` copies of mono material into one frame-interleaved
/// buffer (channel 0 = `left`, channel 1 = `right`, extras cycle).
pub fn interleave(left: &[f32], right: &[f32], channels: u16) -> Vec<f32> {
    let frames = left.len().max(right.len());
    let mut out = Vec::with_capacity(frames * channels as usize);
    for i in 0..frames {
        for c in 0..channels {
            let s = if c % 2 == 0 {
                *left.get(i).unwrap_or(&0.0)
            } else {
                *right.get(i).unwrap_or(&0.0)
            };
            out.push(s);
        }
    }
    out
}

/// Encode mono or interleaved `samples` (in `[-1, 1]`) as a PCM S16 WAV.
pub fn encode_wav_s16(samples: &[f32], sample_rate: u32, channels: u16) -> Vec<u8> {
    let frames = samples.len() / channels.max(1) as usize;
    let data = &samples[..frames * channels as usize];
    // The declared rate is absurd on purpose in the adversarial cases (u32::MAX),
    // and this function is how those cases reach a stage at all, so the header
    // arithmetic saturates rather than overflowing. A wrapped byte rate would be
    // worse than a panic: it would look plausible to a decoder, and the one
    // thing this harness must not do is hand a stage a quietly wrong file.
    let byte_rate = sample_rate
        .saturating_mul(u32::from(channels))
        .saturating_mul(2);
    let data_bytes = u32::try_from(data.len() * 2).unwrap_or(u32::MAX);
    let mut out = Vec::with_capacity(44 + data.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&36u32.saturating_add(data_bytes).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&(channels.saturating_mul(2)).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    for &s in &samples[..frames * channels as usize] {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Encode mono `samples` as IEEE-float WAV (format tag 3).
pub fn encode_wav_f32(samples: &[f32], sample_rate: u32, channels: u16) -> Vec<u8> {
    let frames = samples.len() / channels.max(1) as usize;
    let data = &samples[..frames * channels as usize];
    // Saturating for the same reason as `encode_wav_s16`: an absurd declared
    // rate must reach the decoder as an absurd header, not as a panic here and
    // not as a wrapped value that looks plausible.
    let byte_rate = sample_rate
        .saturating_mul(u32::from(channels))
        .saturating_mul(4);
    let data_bytes = u32::try_from(data.len() * 4).unwrap_or(u32::MAX);
    let mut out = Vec::with_capacity(44 + data.len() * 4);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&36u32.saturating_add(data_bytes).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&3u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&(channels.saturating_mul(4)).to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    for &s in data {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// A temporary `.wav` file holding `samples`; deleted on drop.
#[derive(Debug)]
pub struct WavFixture {
    _dir: tempfile::TempDir,
    path: PathBuf,
}

impl WavFixture {
    /// Write mono `samples` at 16 kHz to a new temp file.
    pub fn mono16k(samples: &[f32]) -> Self {
        Self::with_rate_channels(samples, FIXTURE_SAMPLE_RATE, 1, false)
    }

    /// Write `samples` at an explicit rate/channel layout.
    pub fn with_rate_channels(
        samples: &[f32],
        sample_rate: u32,
        channels: u16,
        float32: bool,
    ) -> Self {
        let bytes = if float32 {
            encode_wav_f32(samples, sample_rate, channels)
        } else {
            encode_wav_s16(samples, sample_rate, channels)
        };
        Self::from_bytes(&bytes, "wav")
    }

    /// Write raw `bytes` with the given extension (for malformed inputs).
    pub fn from_bytes(bytes: &[u8], extension: &str) -> Self {
        let dir = tempfile::TempDir::new().expect("qa tempdir");
        let path = dir.path().join(format!("qa-fixture.{extension}"));
        std::fs::write(&path, bytes).expect("qa fixture write");
        Self { _dir: dir, path }
    }

    /// The path to hand to `probe`/`decode`.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Root-mean-square level of `samples`.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (sum / samples.len() as f64).sqrt() as f32
}

/// Peak absolute value of `samples`.
pub fn peak(samples: &[f32]) -> f32 {
    samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max)
}

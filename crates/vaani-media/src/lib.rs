//! Decoding, demuxing and resampling.
//!
//! Owned by agent **A1**. The `Cargo.toml` in this directory is frozen by
//! the architect and is part of the integration surface; the contents of this
//! file are yours.
//!
//! If you need a dependency that is not in the manifest, do not edit the
//! manifest. Report it under `INTEGRATION NOTES` and the architect will add it
//! in one commit that every agent rebases onto.
//!
//! Read `docs/agent/PARALLEL_OWNERSHIP.md` before the first edit.
//!
//! # What this crate is for
//!
//! Everything downstream of this crate — VAD, ASR, diarization, the front end —
//! consumes one shape of data: a mono 16 kHz `f32` buffer whose sample index *n*
//! is time *n* / 16000 seconds into the recording. This crate is the only place
//! that decides how a container, a codec, a channel layout and a sample rate
//! become that buffer, and it is therefore the only place where a mistake turns
//! into a transcript that is silently out of sync with the audio.
//!
//! Three rules follow from that, and they shape every design decision below.
//!
//! ## 1. The decoded sample count is the truth
//!
//! A container's declared duration is a *claim*. Matroska in particular writes a
//! duration that is routinely wrong, MP4's `mvhd` duration excludes encoder
//! delay, and a file that is still being copied declares whatever its header was
//! written with. The number of audio frames a decoder actually emits is a
//! measurement.
//!
//! So [`MediaInfo::declared_duration_ms`] carries the container's claim and
//! [`DecodeReport::decoded_duration_ms`] carries the measurement, and the
//! pipeline is expected to feed the measurement into
//! `MediaInfo::authoritative_duration`, which prefers it. See
//! [`probe`](crate::probe) for how each figure is obtained.
//!
//! ## 2. Decoding must be cancellable in bounded time
//!
//! A user who clicks Cancel on a 40-minute recording must not wait 40 minutes.
//! [`decode`](crate::decode) polls the [`CancelToken`] at least once per second
//! of *source* audio and additionally at least once every 250 ms of wall clock,
//! so cancellation latency is bounded by the slower of the two. The measured
//! latency on the reference machine is in the crate report.
//!
//! ## 3. Malformed input is a fact, not an exception
//!
//! A recording that is 40% copied is a normal thing for a user to try to
//! transcribe. By default the decoder returns the audio it managed to decode and
//! sets [`DecodeReport::truncated`]; a caller that would rather fail loudly sets
//! [`DecodeOptions::on_truncated`] to [`TruncatedPolicy::Fail`]. Zero decoded
//! audio is always an error, because an empty buffer downstream is a crash in
//! somebody else's stage.
//!
//! # Module map
//!
//! * [`probe`] — container and stream inspection without decoding.
//! * [`decode`] — the decoder, the streaming iterator, and the buffered entry
//!   point the pipeline should call.
//! * [`resample`] — the [`rubato`](https://docs.rs/rubato) wrapper, including
//!   the exact filter configuration and why it was chosen.
//! * [`downmix`] — many channels to one, with the clipping guard.
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//! use vaani_core::CancelToken;
//! use vaani_media::{decode_to_mono16k, probe};
//!
//! # fn main() -> vaani_core::Result<()> {
//! let path = Path::new("meeting.mp4");
//! let info = probe(path)?;
//! println!("{} {} Hz", info.container, info.declared_duration_ms.unwrap_or(0));
//!
//! let cancel = CancelToken::new();
//! let audio = decode_to_mono16k(path, &cancel)?;
//! assert_eq!(audio.sample_rate, 16_000);
//! assert_eq!(audio.channels, 1);
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
#![forbid(unsafe_code)]

pub mod decode;
pub mod downmix;
pub mod probe;
pub mod resample;

mod errors;

pub use decode::{
    AudioStream, DecodeOptions, DecodeReport, DecodedAudio, TruncatedPolicy, decode,
    decode_to_mono16k, decode_to_mono16k_with_progress,
};
pub use downmix::{DownmixStats, MAX_INPUT_SAMPLE, downmix_to_mono};
pub use probe::{probe, probe_bytes};
pub use resample::{
    DEFAULT_RESAMPLER, MonoResampler, ResamplerConfig, ResamplerPlan, WINDOW_FUNCTION,
    expected_output_frames, resample_mono,
};

/// The sample rate every model stage in VaaniDesk requires.
///
/// Not a preference: the ASR front end's log-mel filterbank, the VAD feature
/// extractor and the diarization embedding model were all trained at 16 kHz, and
/// `vaani-asr`'s golden features are only valid at this rate.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Crate version, from the workspace.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

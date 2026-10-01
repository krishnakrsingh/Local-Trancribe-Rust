//! Decoding a container to mono 16 kHz `f32`.
//!
//! # Which API the pipeline should call
//!
//! **Use [`decode_to_mono16k`]**, or [`decode`] if you also want the
//! [`DecodeReport`]. It returns the whole recording in one allocation, and for
//! the reference workload — a 14-minute meeting — that is 12.8 million `f32`,
//! about 51 MB, which is nothing next to the ~900 MB model the next stage loads.
//!
//! [`AudioStream`] exists for two cases and should be reached for only then:
//!
//! * **Memory-constrained decoding.** A 10-hour recording is 576 million samples,
//!   2.3 GB. A caller that cannot hold that can consume
//!   [`AudioStream::for_each_chunk`] and hand each chunk straight to a downstream
//!   stage.
//! * **Previewing.** The UI can play the first thirty seconds without waiting for
//!   a two-hour file to decode.
//!
//! Both paths run the *same* loop over the same state, so they cannot disagree
//! about the samples, the length, or the sample-exactness of a 16 kHz input.
//!
//! # Memory: one buffer, one pass, no second copy
//!
//! The invariant is that the output `Vec<f32>` is written exactly once, in order,
//! and is never copied. Concretely, per decoded packet:
//!
//! 1. Symphonia's decoded buffer is converted to interleaved `f32` into a scratch
//!    buffer sized to one packet. PCM decoders produce at most a few thousand
//!    frames per packet, so this is tens of kilobytes and is reused.
//! 2. That is downmixed to mono into a second scratch buffer, also reused.
//! 3. The mono samples are handed to the resampler, which appends output
//!    straight into the caller's `Vec`.
//!
//! The output vector is pre-reserved from the container's *declared* duration
//! before the first packet, which is the one place the untrustworthy claim is
//! useful: it avoids the repeated doubling of `Vec` growth that would otherwise
//! copy 51 MB once. If the claim is wrong the vector simply grows as usual, and
//! if it is wildly wrong the reservation is capped by
//! [`DecodeOptions::reserve_limit_bytes`] so a header claiming ten hours of a
//! ten-second file cannot make the process try to reserve 2.3 GB up front.
//!
//! The file itself is never held in memory: Symphonia's `MediaSourceStream` reads
//! through a 64 KB ring buffer, so the decode's resident set is the output buffer
//! plus a few hundred kilobytes of filter tables and scratch.
//!
//! # Cancellation
//!
//! The pipeline contract says a stage must poll `CancelToken` at least once per
//! second of work. This decoder polls on **either** of two conditions, whichever
//! comes first:
//!
//! * one second of *source* audio has been decoded, or
//! * 250 ms of wall clock has passed.
//!
//! The first is the contract. The second exists because the contract alone is not
//! a latency bound: it bounds *work*, not *elapsed time*, and a file that decodes
//! slowly would make Cancel take proportionally longer for no good reason. With
//! both conditions, cancellation latency is bounded by
//! [`DecodeOptions::cancel_check_wall_ms`] of wall clock regardless of content,
//! and in practice the first poll after `cancel()` arrives within a few
//! milliseconds because decoding runs far faster than realtime.
//!
//! # Truncated and partially written files
//!
//! **Decision: by default, return the audio decoded so far and say so.**
//!
//! A file that is 40% copied is a normal thing for a user to try. Failing the
//! whole job teaches them VaaniDesk cannot be trusted with files that are still
//! arriving; silently pretending the file was complete teaches them its
//! timestamps are lies. So [`DecodeOptions::on_truncated`] defaults to
//! [`TruncatedPolicy::AllowPartial`]: the decode stops at the first unrecoverable
//! error, [`DecodeReport::truncated`] is set, a human-readable reason lands in
//! [`DecodeReport::warnings`], and the pipeline decides what to do about it. A
//! caller that would rather be told no sets [`TruncatedPolicy::Fail`] and gets an
//! error whose recovery names the real cause.
//!
//! Two outcomes are *not* partial and never are:
//!
//! * **Zero audio frames decoded.** Always an error, under every policy. An empty
//!   `AudioBuffer` has `duration() == Time::ZERO` and every downstream chunker
//!   produces nothing, so returning one would hand the next stage a job that can
//!   never finish.
//! * **Cancellation.** Always an error, always `Cancelled`.
//!
//! A single *undecodable packet* is not treated as truncation at all. Symphonia's
//! own contract says such a packet should be discarded and decoding continued,
//! and one corrupt frame in a 40-minute recording should not cost the user the
//! other 39 minutes and 59 seconds. The skip is counted in
//! [`DecodeReport::warnings`].
//!
//! # Time base
//!
//! The returned `AudioBuffer::start_time` is always [`vaani_core::Time::ZERO`] and
//! `sample_rate` is always [`TARGET_SAMPLE_RATE`], so a sample index *n* is time
//! *n* / 16000 seconds into the recording. Encoder delay and padding are trimmed
//! by the decoder (`AudioDecoderOptions::gapless`, on by default), so a stream
//! that begins with 576 samples of MP3 padding contributes 0 samples, not 36 ms
//! of silence that would shift every word downstream.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use symphonia::core::codecs::CodecParameters;
use symphonia::core::codecs::audio::{
    AudioCodecId, AudioCodecParameters, AudioDecoder, AudioDecoderOptions,
};
use symphonia::core::codecs::registry::CodecRegistry;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatReader, Track, TrackType};
use symphonia::default::get_codecs;
use vaani_core::media::AudioBuffer;
use vaani_core::{CancelToken, MediaInfo, Result, Time};

use crate::TARGET_SAMPLE_RATE;
use crate::downmix;
use crate::errors;
use crate::probe;
use crate::resample::MonoResampler;

/// How a decode that hit an unrecoverable error part-way through should end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TruncatedPolicy {
    /// Return the audio decoded so far, with [`DecodeReport::truncated`] set.
    ///
    /// The default, and the right answer for a file that is still downloading or
    /// being copied. See the module documentation.
    #[default]
    AllowPartial,
    /// Return the error instead, with a recovery that names the real cause.
    Fail,
}

/// Knobs for a decode. The defaults are what the pipeline wants.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeOptions {
    /// The rate to produce. [`TARGET_SAMPLE_RATE`] in practice; the field exists
    /// so the choice is visible rather than buried in a constant, and so a test
    /// can convert to something else.
    pub target_sample_rate: u32,
    /// What to do about a stream that ends early.
    pub on_truncated: TruncatedPolicy,
    /// Upper bound on the output buffer's initial reservation, in bytes.
    ///
    /// A container can declare a ten-hour duration for a ten-second file. Without
    /// a cap, honouring that claim would try to reserve 2.3 GB before reading a
    /// byte. The cap costs nothing in the normal case: a four-hour meeting at
    /// 16 kHz mono is 921 MB and a real one is far shorter.
    pub reserve_limit_bytes: usize,
    /// Hard ceiling on the *decoded output*, in bytes, whatever the container
    /// claims.
    ///
    /// `reserve_limit_bytes` only bounds the initial reservation. A file whose
    /// header says ten seconds and whose packets go on for ten hours passes that
    /// cap easily -- the vector simply grows as decoding proceeds, which is
    /// exactly what the reservation cap was documented as permitting. At 16 kHz
    /// mono f32 that is 230 kB per second of audio, so an hour is 827 MB and a
    /// long enough file ends the process on a machine with 4.7 GB free.
    ///
    /// This is a different kind of bound and it is the one that actually holds:
    /// it is checked as samples are produced, so a lying header cannot buy
    /// unbounded memory. Eight hours at 16 kHz mono f32 is 1.8 GB, which is far
    /// beyond any meeting and comfortably inside what the machine can decode.
    ///
    /// Deliberately not configurable from the environment. A ceiling that an
    /// environment variable can raise is not a ceiling, and this one exists
    /// precisely for the case where nothing else is trusted.
    pub max_output_bytes: usize,
    /// How often to check for cancellation, in seconds of source audio. `1.0` is
    /// the pipeline contract's minimum.
    pub cancel_check_audio_seconds: f64,
    /// Wall-clock ceiling between cancellation checks, in milliseconds.
    ///
    /// The contract is expressed in audio seconds because that is what bounds
    /// *work*; this bounds *latency*. See the module documentation.
    pub cancel_check_wall_ms: u64,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            target_sample_rate: TARGET_SAMPLE_RATE,
            on_truncated: TruncatedPolicy::AllowPartial,
            // 512 MB. A four-hour meeting at 16 kHz mono is 921 MB, so this caps
            // the pathological cases without touching a realistic one.
            reserve_limit_bytes: 512 * 1024 * 1024,
            // 8 hours at 16 kHz mono f32. 1.8 GB, far beyond any real meeting.
            max_output_bytes: 8 * 3600 * 16_000 * 4,
            cancel_check_audio_seconds: 1.0,
            cancel_check_wall_ms: 250,
        }
    }
}

/// Everything measured while decoding, including the number the pipeline should
/// trust for the recording's duration.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeReport {
    /// The container's claim, as [`probe`] produced it.
    pub media_info: MediaInfo,
    /// Audio frames the decoder emitted, at the source rate. This is the
    /// measurement, and it is the one to believe.
    pub source_frames: u64,
    /// Sample rate the decoder actually produced, which is not always the rate
    /// the container declared.
    pub source_sample_rate: u32,
    /// Channel count the decoder actually produced.
    pub source_channels: u16,
    /// Frames in the returned buffer.
    pub output_frames: usize,
    /// Rate of the returned buffer.
    pub output_sample_rate: u32,
    /// `output_frames * 1000 / output_sample_rate`.
    ///
    /// The authoritative duration. Feed this to `MediaInfo::authoritative_duration`
    /// rather than using `declared_duration_ms`.
    pub decoded_duration_ms: i64,
    /// Whether the stream ended early, was damaged, or was incomplete.
    pub truncated: bool,
    /// Channel samples that were NaN or infinite and were replaced with silence.
    pub nonfinite_samples: u64,
    /// Channel samples clamped to [`downmix::MAX_INPUT_SAMPLE`].
    pub clamped_samples: u64,
    /// Mono samples whose magnitude exceeded 1.0. Reported, not altered.
    pub over_unity_samples: u64,
    /// Human-readable notes, one per anomaly. Safe to log; never shown verbatim
    /// in the UI, because each carries a container-level diagnostic.
    pub warnings: Vec<String>,
    /// Wall-clock time spent in this crate.
    pub elapsed: Duration,
}

impl DecodeReport {
    /// Progress in `[0, 1]` against the container's claim.
    ///
    /// `None` when there is no claim to measure against, because a fabricated
    /// percentage is exactly what `vaani-core`'s progress contract forbids.
    /// Clamped, because the claim can be short as well as long — a 10-hour
    /// declaration for 10 seconds of audio reaches 1.0 immediately, and a bar
    /// that sits at 100% for nine and a half hours is a lie of a different kind.
    #[must_use]
    pub fn fraction_of_claim(&self) -> Option<f64> {
        let declared = self.media_info.declared_duration_ms.filter(|d| *d > 0)?;
        if self.source_sample_rate == 0 {
            return None;
        }
        let done_ms = self.source_frames * 1000 / u64::from(self.source_sample_rate);
        Some((done_ms as f64 / declared as f64).clamp(0.0, 1.0))
    }

    /// Seconds of source audio decoded so far, for `Progress::media`.
    #[must_use]
    pub fn source_seconds(&self) -> f64 {
        if self.source_sample_rate == 0 {
            return 0.0;
        }
        self.source_frames as f64 / f64::from(self.source_sample_rate)
    }

    /// Whether any sample had to be repaired, which is worth a log line.
    #[must_use]
    pub fn had_to_repair_samples(&self) -> bool {
        self.nonfinite_samples > 0 || self.clamped_samples > 0
    }

    /// Fold one batch of downmix counters in.
    ///
    /// `source_frames` accumulates the *mono* frames the downmix produced, which
    /// is the decoder's frame count; a container that declares a different figure
    /// is exactly the discrepancy this crate exists to measure.
    fn accumulate(report: &mut Self, stats: &crate::downmix::DownmixStats) {
        report.source_frames += stats.frames;
        report.nonfinite_samples += stats.nonfinite_replaced;
        report.clamped_samples += stats.input_clamped;
        report.over_unity_samples += stats.over_unity;
    }
}

/// A decoded buffer and the measurements that came with it.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedAudio {
    /// Mono, at `DecodeOptions::target_sample_rate`, `start_time == Time::ZERO`.
    pub audio: AudioBuffer,
    /// What was measured. See [`DecodeReport::decoded_duration_ms`].
    pub report: DecodeReport,
}

impl DecodedAudio {
    /// The buffer, for callers that do not want the report.
    #[must_use]
    pub fn into_audio(self) -> AudioBuffer {
        self.audio
    }
}

/// Decode a file to mono 16 kHz `f32`.
///
/// The entry point the pipeline should call. See the module documentation for the
/// memory, cancellation and truncation contracts.
///
/// # Errors
///
/// * `UnsupportedMedia` if the file cannot be opened, is empty, or is not a
///   container this build demuxes.
/// * `NoAudioStream` if the container has no usable audio track.
/// * `AudioDecodeFailed` if the audio codec has no decoder, if the data is
///   malformed, if the stream ends early and [`DecodeOptions::on_truncated`] is
///   [`TruncatedPolicy::Fail`], or if the decode produced no audio at all.
/// * `Cancelled` if `cancel` is set.
/// * `UnsupportedMedia` if the declared sample rate is one no resampler can be
///   built for.
pub fn decode_to_mono16k(path: &Path, cancel: &CancelToken) -> Result<AudioBuffer> {
    decode(path, cancel, &DecodeOptions::default()).map(DecodedAudio::into_audio)
}

/// As [`decode_to_mono16k`], reporting progress as it goes.
///
/// `progress` is called with [`vaani_core::Progress::media`] over
/// [`vaani_core::JobPhase::ExtractingAudio`] at most once per half second of
/// source audio, which for the reference workload is about 1700 calls for a
/// 14-minute meeting. When the container declares no duration the total is zero,
/// which `Progress::media` reports as indeterminate rather than as a percentage.
///
/// This is the only function in the crate that knows about job phases. The
/// streaming API deliberately does not, so a caller driving chunks itself is not
/// forced to accept this crate's opinion about the job state machine — it can
/// build the same `Progress` from [`DecodeReport::source_seconds`].
pub fn decode_to_mono16k_with_progress(
    path: &Path,
    cancel: &CancelToken,
    progress: &vaani_core::pipeline::ProgressSink,
) -> Result<AudioBuffer> {
    let options = DecodeOptions::default();
    let started = Instant::now();
    let mut stream = AudioStream::open(path, &options)?;
    let total_s = stream
        .media_info()
        .declared_duration_ms
        .filter(|ms| *ms > 0)
        .map_or(0.0, |ms| ms as f64 / 1000.0);
    let mut samples = Vec::new();
    reserve_for(&mut samples, Some(total_s), &options);

    let mut chunk = Vec::new();
    let mut last_reported = f64::NEG_INFINITY;
    while stream.next_chunk_into(&mut chunk, cancel)? {
        samples.extend_from_slice(&chunk);
        let done = stream.report().source_seconds();
        if done - last_reported >= 0.5 {
            last_reported = done;
            progress(vaani_core::Progress::media(
                vaani_core::JobPhase::ExtractingAudio,
                done,
                total_s,
            ));
        }
    }
    let report = stream.finish()?;
    let out = collect(path, &stream, samples, report, started.elapsed())?;
    // The closing callback reports seconds, like every other one this function
    // makes, so that a consumer plotting "seconds of media processed" does not
    // see the value jump from 834.2 back to 1.0 at the end. `total_s` is zero
    // when the container declared no duration, and then a unit fraction is the
    // only thing there is to say.
    let (done, whole) = if total_s > 0.0 {
        (total_s, total_s)
    } else {
        (1.0, 1.0)
    };
    progress(vaani_core::Progress::media(
        vaani_core::JobPhase::ExtractingAudio,
        done,
        whole,
    ));
    Ok(out.audio)
}

/// Decode a file, returning both the buffer and what was measured.
///
/// # Errors
///
/// See [`decode_to_mono16k`]; the same set, filtered by `options`.
pub fn decode(path: &Path, cancel: &CancelToken, options: &DecodeOptions) -> Result<DecodedAudio> {
    let started = Instant::now();
    let mut stream = AudioStream::open(path, options)?;
    let declared = stream
        .media_info()
        .declared_duration_ms
        .filter(|ms| *ms > 0)
        .map(|ms| ms as f64 / 1000.0);
    let mut samples = Vec::new();
    reserve_for(&mut samples, declared, options);

    let mut chunk = Vec::new();
    while stream.next_chunk_into(&mut chunk, cancel)? {
        // The enforced half of the memory bound. `reserve_for` above only caps
        // the *initial* reservation from the container's claim; this caps what
        // actually accumulates, so a file that declares ten seconds and then
        // delivers ten hours of packets fails here instead of taking the
        // process with it.
        //
        // Checked before extending rather than after, so the process never
        // momentarily holds more than the limit.
        let next = samples
            .len()
            .saturating_add(chunk.len())
            .saturating_mul(std::mem::size_of::<f32>());
        if next > options.max_output_bytes {
            // Report how far it actually got, so the message can distinguish
            // "slightly too long" from "this file is lying about its length".
            // The *source* rate is the right divisor: `samples` holds source-rate
            // frames at this point, before the resample to 16 kHz.
            let rate = stream.report().source_sample_rate.max(1) as i64;
            let decoded_ms = (samples.len() as i64).saturating_mul(1000) / rate;
            let limit_ms =
                (options.max_output_bytes / std::mem::size_of::<f32>()) as i64 * 1000 / rate;
            return Err(errors::too_long(path, decoded_ms, limit_ms));
        }
        samples.extend_from_slice(&chunk);
    }
    let report = stream.finish()?;
    collect(path, &stream, samples, report, started.elapsed())
}

/// Finish a collected decode: exact length, a report that matches the buffer, and
/// a refusal to hand back an empty one.
///
/// The resampler's output grid can end a sample short of the target for some rate
/// pairs, so [`AudioStream::trim_collected`] closes that gap. Doing it here, once,
/// is why the buffered and streaming paths agree on length to the sample.
fn collect(
    path: &Path,
    stream: &AudioStream,
    mut samples: Vec<f32>,
    mut report: DecodeReport,
    elapsed: Duration,
) -> Result<DecodedAudio> {
    stream.trim_collected(&mut samples);
    if samples.is_empty() {
        return Err(errors::no_audio_decoded(path));
    }
    let rate = if report.output_sample_rate == 0 {
        TARGET_SAMPLE_RATE
    } else {
        report.output_sample_rate
    };
    report.output_frames = samples.len();
    report.decoded_duration_ms = (samples.len() as u64 * 1000 / u64::from(rate)) as i64;
    report.elapsed = elapsed;
    tracing::debug!(
        path = %path.display(),
        container = %report.media_info.container,
        frames = report.output_frames,
        rate,
        source_frames = report.source_frames,
        source_rate = report.source_sample_rate,
        declared_ms = report.media_info.declared_duration_ms.unwrap_or(-1),
        decoded_ms = report.decoded_duration_ms,
        elapsed_ms = elapsed.as_millis() as u64,
        truncated = report.truncated,
        warnings = report.warnings.len(),
        nonfinite = report.nonfinite_samples,
        "media decode finished"
    );
    Ok(DecodedAudio {
        audio: AudioBuffer::mono(samples, rate, Time::ZERO),
        report,
    })
}

/// Pre-size the output buffer from the container's claim, within the cap.
fn reserve_for(samples: &mut Vec<f32>, declared_s: Option<f64>, options: &DecodeOptions) {
    let Some(seconds) = declared_s.filter(|s| s.is_finite() && *s > 0.0) else {
        return;
    };
    let want = (seconds * f64::from(options.target_sample_rate)) as usize;
    let cap = options.reserve_limit_bytes / std::mem::size_of::<f32>();
    let want = want.min(cap);
    if want > samples.capacity() {
        samples.reserve_exact(want);
    }
}

/// A streaming decoder: mono 16 kHz chunks as they become available.
///
/// The iterator half of this crate's API. See the module documentation for when
/// to reach for it; for the pipeline, [`decode_to_mono16k`] is the answer.
pub struct AudioStream {
    path: PathBuf,
    options: DecodeOptions,
    reader: Box<dyn FormatReader>,
    decoder: Box<dyn AudioDecoder>,
    track_id: u32,
    resampler: MonoResampler,
    media_info: MediaInfo,
    /// Interleaved `f32` for one decoded packet.
    scratch_interleaved: Vec<f32>,
    /// Mono `f32` for one decoded packet.
    scratch_mono: Vec<f32>,
    /// Whatever the resampler produced since the last append.
    scratch_resampled: Vec<f32>,
    /// Output frames already handed to the caller.
    emitted: u64,
    /// Whether the end of the stream has been reached.
    eof: bool,
    /// Whether the resampler's tail has been flushed.
    flushed: bool,
    report: DecodeReport,
}

impl std::fmt::Debug for AudioStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioStream")
            .field("path", &self.path)
            .field("container", &self.media_info.container)
            .field("emitted", &self.emitted)
            .field("eof", &self.eof)
            .field("resampler", &self.resampler.plan())
            .finish_non_exhaustive()
    }
}

impl AudioStream {
    /// Open a file and prepare to stream its audio.
    ///
    /// Decodes nothing yet, so this is cheap and safe on a file that is still
    /// being written. A file with no audio track fails here, not on the first
    /// chunk, because the container has already told us.
    ///
    /// # Errors
    ///
    /// See [`decode_to_mono16k`].
    pub fn open(path: &Path, options: &DecodeOptions) -> Result<Self> {
        let metadata = std::fs::metadata(path).map_err(|e| errors::open_failed(path, &e))?;
        let size_bytes = metadata.len();
        let file = File::open(path).map_err(|e| errors::open_failed(path, &e))?;
        if size_bytes == 0 {
            return Err(errors::unrecognised_container(
                path,
                &SymphoniaError::DecodeError("the file is empty"),
            ));
        }
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        let reader = probe::open_reader(Box::new(file), extension.as_deref(), path)?;
        let media_info = probe::describe(reader.as_ref(), size_bytes);

        let track =
            pick_audio_track(reader.as_ref()).ok_or_else(|| errors::no_audio_track(path))?;
        let track_id = track.id;
        let params = track
            .codec_params
            .as_ref()
            .ok_or_else(|| errors::no_audio_track(path))?;
        let CodecParameters::Audio(audio_params) = params else {
            return Err(errors::no_audio_track(path));
        };
        let sample_rate = audio_params
            .sample_rate
            .filter(|r| *r > 0)
            .ok_or_else(|| errors::zero_sample_rate(path))?;
        let channels = audio_params.channels.as_ref().map_or(0, |c| c.count());
        if channels == 0 {
            return Err(errors::no_audio_track(path));
        }
        let decoder = build_decoder(audio_params.codec, audio_params, path)?;
        let resampler = MonoResampler::with_target(sample_rate, options.target_sample_rate)?;

        // Every codec in scope produces at most a few thousand frames per packet,
        // so one scratch allocation each, reused for the whole file, keeps the
        // inner loop off the allocator entirely.
        const SCRATCH_FRAMES: usize = 8192;
        let report = DecodeReport {
            media_info: media_info.clone(),
            source_frames: 0,
            source_sample_rate: sample_rate,
            source_channels: u16::try_from(channels).unwrap_or(u16::MAX),
            output_frames: 0,
            output_sample_rate: options.target_sample_rate,
            decoded_duration_ms: 0,
            truncated: false,
            nonfinite_samples: 0,
            clamped_samples: 0,
            over_unity_samples: 0,
            warnings: Vec::new(),
            elapsed: Duration::ZERO,
        };
        Ok(Self {
            path: path.to_path_buf(),
            options: options.clone(),
            reader,
            decoder,
            track_id,
            resampler,
            media_info,
            scratch_interleaved: Vec::with_capacity(SCRATCH_FRAMES * channels),
            scratch_mono: Vec::with_capacity(SCRATCH_FRAMES),
            scratch_resampled: Vec::with_capacity(SCRATCH_FRAMES * 2),
            emitted: 0,
            eof: false,
            flushed: false,
            report,
        })
    }

    /// What the container says, including its duration claim.
    #[must_use]
    pub fn media_info(&self) -> &MediaInfo {
        &self.media_info
    }

    /// What has been measured so far. Cheap; safe to call every chunk.
    #[must_use]
    pub fn report(&self) -> &DecodeReport {
        &self.report
    }

    /// The exact number of output frames the audio seen so far must produce.
    #[must_use]
    pub fn expected_output_frames(&self) -> u64 {
        self.resampler.expected_output_frames()
    }

    /// Decode until the next chunk of target-rate mono audio is ready.
    ///
    /// Appends to `out`, which is cleared first, and returns whether anything was
    /// appended. A `false` return means the stream is finished and
    /// [`AudioStream::finish`] should be called.
    ///
    /// Chunk sizes follow the container's packet boundaries and are therefore
    /// *nearly* uniform rather than exactly uniform — a WAVE file typically
    /// yields batches of 341 and 342 samples with an occasional 682, because that
    /// is how its packets fall. They are bounded, which is the property that
    /// matters: nothing here is proportional to the length of the file.
    ///
    /// Polls `cancel` at least once per second of source audio and at least once
    /// every [`DecodeOptions::cancel_check_wall_ms`] milliseconds of wall clock.
    ///
    /// # Errors
    ///
    /// * `Cancelled` if `cancel` is set.
    /// * `AudioDecodeFailed` if a packet will not decode and
    ///   [`DecodeOptions::on_truncated`] is [`TruncatedPolicy::Fail`], or if the
    ///   format changes in a way this decoder cannot follow.
    pub fn next_chunk_into(&mut self, out: &mut Vec<f32>, cancel: &CancelToken) -> Result<bool> {
        out.clear();
        if self.eof {
            return Ok(false);
        }
        cancel.check()?;

        let audio_deadline = (f64::from(self.resampler.plan().input_rate)
            * self.options.cancel_check_audio_seconds.max(0.001))
        .max(1.0) as u64;
        let wall_deadline = Duration::from_millis(self.options.cancel_check_wall_ms.max(1));
        let mut last_check = Instant::now();
        let mut frames_since_check = 0u64;
        let chunk_started = Instant::now();

        loop {
            let packet = match self.reader.next_packet() {
                Ok(Some(p)) => p,
                Ok(None) => {
                    self.eof = true;
                    break;
                }
                Err(SymphoniaError::ResetRequired) => {
                    self.handle_reset_required()?;
                    continue;
                }
                Err(e) => {
                    self.stop_or_fail(e)?;
                    break;
                }
            };

            if packet.track_id != self.track_id {
                // Video and subtitle packets are expected and carry no audio.
                continue;
            }
            // The decoded buffer borrows the decoder, so it is copied into this
            // stream's scratch and released inside the block. Copying here rather
            // than in a `&mut self` method is what keeps the borrow checker
            // satisfied: `self.decoder` and `self.scratch_interleaved` are
            // disjoint fields, and a method taking `&mut self` would not be.
            let (frames, rate, channels) = {
                let decoded = match self.decoder.decode(&packet) {
                    Ok(buf) => buf,
                    // Symphonia's own contract: a packet that fails to decode
                    // should be discarded and decoding continued. One corrupt
                    // frame must not cost the user the other forty minutes, so
                    // this is a counted warning rather than a failure.
                    Err(SymphoniaError::DecodeError(reason)) => {
                        self.report
                            .warnings
                            .push(format!("skipped one undecodable audio packet: {reason}"));
                        continue;
                    }
                    Err(SymphoniaError::ResetRequired) => {
                        self.handle_reset_required()?;
                        continue;
                    }
                    Err(e) => {
                        self.stop_or_fail(e)?;
                        break;
                    }
                };
                let frames = decoded.frames();
                let rate = decoded.spec().rate();
                let channels = decoded.spec().channels().count();
                if frames > 0 {
                    // `copy_to_slice_interleaved` copies into an existing slice of
                    // exactly the right length; it does not grow one. Sizing the
                    // scratch first is what keeps this a memset rather than a
                    // reallocation, because the capacity is reserved in `open`.
                    self.scratch_interleaved.resize(frames * channels, 0.0);
                    decoded.copy_to_slice_interleaved(&mut self.scratch_interleaved);
                }
                (frames, rate, channels)
            };
            if frames == 0 {
                continue;
            }
            self.resample_packet(frames, rate, channels, out)?;
            frames_since_check += frames as u64;

            if frames_since_check >= audio_deadline || last_check.elapsed() >= wall_deadline {
                cancel.check()?;
                last_check = Instant::now();
                frames_since_check = 0;
            }
            if !out.is_empty() {
                return Ok(true);
            }
            // A file can be enormous and produce no output for a long time — an
            // MP3 with a 10-minute Xing header, or a video track being skipped.
            // Without this the loop would look hung; with it, the failure names
            // itself.
            if chunk_started.elapsed() > Duration::from_secs(60) {
                return Err(errors::internal(format!(
                    "{}: decoded 60 s of packets without producing any audio",
                    self.path.display()
                )));
            }
        }

        if !self.flushed {
            self.flushed = true;
            // `MonoResampler::finish` *appends*, and `scratch_resampled` still
            // holds the last packet's output, which `resample_packet` already
            // handed to the caller. Without this clear the final batch would be
            // emitted twice: the transcript would be fine but the last second of
            // audio would be a duplicate, spliced mid-waveform.
            self.scratch_resampled.clear();
            self.resampler.finish(&mut self.scratch_resampled)?;
            self.append_capped(out);
        }
        Ok(!out.is_empty())
    }

    /// As [`AudioStream::next_chunk_into`], returning an owned chunk.
    ///
    /// Convenient, and one allocation per chunk. Prefer
    /// [`AudioStream::next_chunk_into`] in a loop that is collecting.
    ///
    /// # Errors
    ///
    /// See [`AudioStream::next_chunk_into`].
    pub fn next_chunk(&mut self, cancel: &CancelToken) -> Result<Option<Vec<f32>>> {
        let mut buf = Vec::new();
        if self.next_chunk_into(&mut buf, cancel)? {
            Ok(Some(buf))
        } else {
            Ok(None)
        }
    }

    /// Stream every chunk to `sink`, returning the final report.
    ///
    /// `sink` receives the chunk and the time its first sample falls at, in
    /// source-media coordinates. Returning `Err` from `sink` aborts the decode and
    /// that error is returned unchanged, so a downstream stage can stop the whole
    /// job by failing.
    ///
    /// # Errors
    ///
    /// See [`AudioStream::next_chunk_into`], plus whatever `sink` returned.
    pub fn for_each_chunk<F>(&mut self, cancel: &CancelToken, mut sink: F) -> Result<DecodeReport>
    where
        F: FnMut(&[f32], Time) -> Result<()>,
    {
        let started = Instant::now();
        let mut buf = Vec::new();
        loop {
            // The offset has to be taken *before* the read. `next_chunk_into`
            // advances `emitted` as it fills the buffer, so reading it afterwards
            // stamps every chunk with the time its *last* sample falls at — the
            // first chunk of a 16 kHz file came out stamped at 72 ms, and every
            // downstream timestamp built on it would have been one chunk late.
            let start = self.time_of(self.emitted);
            if !self.next_chunk_into(&mut buf, cancel)? {
                break;
            }
            sink(&buf, start)?;
        }
        let mut report = self.finish()?;
        report.elapsed = started.elapsed();
        Ok(report)
    }

    /// Close the stream and take the final report.
    ///
    /// Idempotent. After this the stream produces no more audio; the caller owns
    /// whatever it collected.
    ///
    /// # Errors
    ///
    /// `Internal` if the resampler fails to flush, which would be a bug here, and
    /// `AudioDecodeFailed` if the stream turned out to be short of the
    /// container's claim and [`DecodeOptions::on_truncated`] is
    /// [`TruncatedPolicy::Fail`].
    pub fn finish(&mut self) -> Result<DecodeReport> {
        if !self.flushed {
            self.flushed = true;
            self.eof = true;
            self.scratch_resampled.clear();
            self.resampler.finish(&mut self.scratch_resampled)?;
        }
        self.report.output_frames = self.emitted as usize;
        self.report.decoded_duration_ms = if self.report.output_sample_rate == 0 {
            0
        } else {
            (self.emitted * 1000 / u64::from(self.report.output_sample_rate)) as i64
        };
        self.note_short_stream()?;
        Ok(self.report.clone())
    }

    /// Force a collected buffer to the exact expected length.
    ///
    /// Public because a caller driving [`AudioStream`] itself has the same need as
    /// [`decode`]: the filter's output grid can land a sample short for some rate
    /// pairs, and the buffered and streaming paths must not disagree about length.
    pub fn trim_collected(&self, samples: &mut Vec<f32>) -> bool {
        let want = usize::try_from(self.expected_output_frames()).unwrap_or(usize::MAX);
        if samples.len() == want {
            return false;
        }
        if samples.len() > want {
            samples.truncate(want);
        } else {
            samples.resize(want, 0.0);
        }
        true
    }

    /// Source-media time of an output sample index.
    #[must_use]
    pub fn time_of(&self, frame: u64) -> Time {
        let rate = u64::from(self.report.output_sample_rate.max(1));
        Time::from_millis((frame * 1000 / rate) as i64)
    }

    /// Convert one decoded packet to mono `f32`, resample it, and append.
    /// Downmix and resample one already-copied packet, then append the result.
    fn resample_packet(
        &mut self,
        frames: usize,
        rate: u32,
        channels: usize,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        let ch = usize::from(self.report.source_channels);
        if channels == 0 || frames == 0 || ch == 0 {
            return Ok(());
        }
        // The decoder may change the rate or channel count mid-stream: an ADPCM
        // block boundary, or a sample-rate switch inside an MKV track. Rebuilding
        // is the only safe response, and guessing would make every sample after
        // the switch wrong by a factor the user cannot hear but the model can.
        if rate != self.report.source_sample_rate || channels != ch {
            let cause = format!(
                "audio format changed mid-stream at frame {}: {} Hz / {} ch became {rate} Hz / {channels} ch",
                self.report.source_frames,
                self.report.source_sample_rate,
                self.report.source_channels,
            );
            self.report.warnings.push(cause.clone());
            return Err(errors::malformed(
                &self.path,
                &SymphoniaError::DecodeError("mid-stream format change"),
            )
            .with_cause(cause));
        }

        // Defensive: whole frames only. A partial frame means the decoder and the
        // demuxer disagree about the layout, and dropping samples silently would
        // shift everything after it.
        self.scratch_interleaved
            .truncate(self.scratch_interleaved.len() / ch * ch);

        let stats = downmix::downmix_to_mono(
            &self.scratch_interleaved,
            self.report.source_channels,
            &mut self.scratch_mono,
        );
        DecodeReport::accumulate(&mut self.report, &stats);

        self.scratch_resampled.clear();
        self.resampler
            .push(&self.scratch_mono, &mut self.scratch_resampled)?;
        self.append_capped(out);
        Ok(())
    }

    /// Append the resampler's output, never exceeding the exact expected length.
    ///
    /// The cap is what makes a streaming consumer and a collecting one agree on
    /// the length: the filter's raw output grid overshoots by up to a chunk, and
    /// the excess is dropped here rather than at the end, where a streaming caller
    /// could no longer reach it.
    fn append_capped(&mut self, out: &mut Vec<f32>) {
        let want = self.resampler.expected_output_frames();
        let room = want.saturating_sub(self.emitted);
        let take = (self.scratch_resampled.len() as u64).min(room) as usize;
        if take == 0 {
            return;
        }
        out.reserve(take);
        out.extend_from_slice(&self.scratch_resampled[..take]);
        self.emitted += take as u64;
    }

    /// Flag a stream that ended *cleanly* and still fell short of the container.
    ///
    /// [`stop_or_fail`] covers the case where the demuxer raises an error
    /// part-way through. This covers the one it does not: a WAV whose `data`
    /// chunk claims ten hours and holds one second is a perfectly readable file
    /// that simply ends, so `next_packet` returns `Ok(None)`, nothing is flagged,
    /// and the caller gets a clean report and a transcript of 0.003% of the
    /// meeting. That is the "40% copied" case from the brief, arrived at through
    /// the one route that looks healthy, so it gets its own check.
    ///
    /// The tolerance is deliberately loose. Container durations are routinely a
    /// little optimistic — MP4's `mvhd` counts encoder delay, Matroska rounds to
    /// the last cluster — and a tight threshold would make
    /// [`TruncatedPolicy::Fail`] reject files that decode perfectly. Two percent
    /// plus half a second is far above any of that and far below any real
    /// truncation.
    fn note_short_stream(&mut self) -> Result<()> {
        if self.report.truncated {
            return Ok(());
        }
        let declared = self.report.media_info.declared_duration_ms.unwrap_or(0);
        if declared <= 0 {
            return Ok(());
        }
        let decoded = self.report.decoded_duration_ms.max(0);
        if declared <= decoded + declared / 50 + 500 {
            return Ok(());
        }

        self.report.truncated = true;
        self.report.warnings.push(format!(
            "the container declared {declared} ms of audio but only {decoded} ms was present; \
             the file looks like an incomplete copy"
        ));
        match self.options.on_truncated {
            TruncatedPolicy::Fail => Err(errors::short_of_claim(&self.path, declared, decoded)),
            TruncatedPolicy::AllowPartial => {
                tracing::warn!(
                    path = %self.path.display(),
                    declared_ms = declared,
                    decoded_ms = decoded,
                    "stream ended short of the container's declared duration"
                );
                Ok(())
            }
        }
    }

    /// The demuxer changed its track list. Re-read it and rebuild the decoder.
    fn handle_reset_required(&mut self) -> Result<()> {
        let track = pick_audio_track(self.reader.as_ref())
            .ok_or_else(|| errors::no_audio_track(&self.path))?;
        let params = track
            .codec_params
            .as_ref()
            .ok_or_else(|| errors::no_audio_track(&self.path))?;
        let CodecParameters::Audio(audio_params) = params else {
            return Err(errors::no_audio_track(&self.path));
        };
        self.decoder = build_decoder(audio_params.codec, audio_params, &self.path)?;
        self.decoder.reset();
        self.track_id = track.id;
        self.resampler.reset();
        self.report.warnings.push(
            "the container changed its track layout mid-stream; the decoder was rebuilt"
                .to_string(),
        );
        Ok(())
    }

    /// Record an unrecoverable stream error and decide whether to keep what we
    /// have or give up.
    ///
    /// `Ok(())` means "the stream is over, hand back what has been decoded", and
    /// the caller breaks out of its read loop. `Err` propagates.
    fn stop_or_fail(&mut self, e: SymphoniaError) -> Result<()> {
        let message = e.to_string();
        self.report.truncated = true;
        // The warning is shown in the UI's warnings list, so it has to be a
        // sentence a user can act on. The demuxer's own text is kept after it,
        // because that is what a developer needs and the report is the only
        // place it survives: pushing the bare error was how this crate ended up
        // telling a user their recording was "truncated" with no indication of
        // how much of it was there.
        let decoded_s = self.report.source_seconds();
        let warning = match self.report.media_info.declared_duration_ms {
            Some(declared) if declared > 0 => format!(
                "the recording stops early: {decoded_s:.1} s of audio was readable out of a \
                 declared {:.1} s, so the rest was not in the file ({message})",
                declared as f64 / 1000.0
            ),
            _ => format!(
                "the recording stops early: {decoded_s:.1} s of audio was readable before the \
                 file ended, so the rest was not in it ({message})"
            ),
        };
        self.report.warnings.push(warning);
        self.eof = true;
        match self.options.on_truncated {
            TruncatedPolicy::Fail => Err(errors::from_symphonia(&self.path, &e)),
            TruncatedPolicy::AllowPartial => {
                tracing::warn!(
                    path = %self.path.display(),
                    error = %message,
                    source_frames = self.report.source_frames,
                    "audio stream ended early; returning the audio decoded so far"
                );
                Ok(())
            }
        }
    }
}

/// The audio track to decode: the default one, else the first with a codec.
///
/// Cloned rather than borrowed, because the reader has to stay mutable for
/// `next_packet` while the track's parameters are still being consulted.
fn pick_audio_track(reader: &dyn FormatReader) -> Option<Track> {
    reader
        .default_track(TrackType::Audio)
        .or_else(|| {
            reader
                .tracks()
                .iter()
                .find(|t| t.track_type() == Some(TrackType::Audio))
        })
        .cloned()
}

/// Build a decoder for a track's codec, or say which codec we could not handle.
fn build_decoder(
    codec_id: AudioCodecId,
    params: &AudioCodecParameters,
    path: &Path,
) -> Result<Box<dyn AudioDecoder>> {
    let registry: &CodecRegistry = get_codecs();
    // Gapless is on: it trims the encoder delay an MP3 or AAC frame carries, so
    // the first sample of the buffer is the first sample of real audio and the
    // transcript's timestamps line up with what the user hears.
    let options = AudioDecoderOptions::default().gapless(true);
    registry
        .make_audio_decoder(params, &options)
        .map_err(|_| errors::unsupported_codec(path, &probe::codec_label(codec_id)))
}

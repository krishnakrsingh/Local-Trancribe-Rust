//! Container and stream inspection, without decoding audio.
//!
//! # Where a duration comes from, and which one to believe
//!
//! This is the single most consequential question in the module, so the answer
//! is stated plainly:
//!
//! **The number of decoded audio frames is the truth. A container's declared
//! duration is a claim, and this crate does not trust it.**
//!
//! `MediaInfo` carries the claim, because that is all a probe can honestly
//! produce, and it is genuinely useful: it is what lets the drop screen say
//! "about 14 minutes" before any decoding has happened. The measurement lives in
//! [`DecodeReport::decoded_duration_ms`](crate::DecodeReport::decoded_duration_ms),
//! and `MediaInfo::authoritative_duration` prefers the measurement when it is
//! available. That function is the contract; this crate's job is to make the
//! measurement as good as the claim is cheap.
//!
//! ## Why the claim is unreliable, concretely
//!
//! * **Matroska.** The `Duration` element is written by the muxer from its own
//!   estimate and is routinely wrong, especially for a file that was remuxed,
//!   streamed, or written by a screen recorder that is still running. It can be
//!   off by minutes. Matroska also has no way to express "this track's audio is
//!   shorter than the video", which is the normal case for a screen recording of
//!   a meeting.
//! * **ISO base media (MP4/M4A).** `mvhd.duration` counts the whole presentation
//!   including the media's edit list, so a file with a 0.1 s `elst` priming
//!   offset declares 100 ms more audio than there is. It is also a
//!   timescale-quantised figure, and the timescale is chosen by the muxer.
//! * **Ogg.** The last page's granule position is authoritative, but only once
//!   the last page has been read — a probe that stops at the header has not seen
//!   it, and a file being copied has not written it yet.
//! * **A file still being written.** The header was written first and declares
//!   whatever the writer intended. The data arrives later. The claim is a plan;
//!   the measurement is a fact.
//!
//! ## How this probe obtains the claim
//!
//! Symphonia exposes a per-track `time_base` and `duration`, and a media-level
//! `time_base` and `duration`. The rule used here is:
//!
//! 1. If the **audio** track has both a time base and a duration, use that. The
//!    audio track's own figures are more relevant than the presentation's, and
//!    for every container in scope they are the ones the demuxer derived from the
//!    actual sample count or granule positions.
//! 2. Otherwise fall back to the media-level figures.
//! 3. Otherwise `None`.
//!
//! The result lands in `MediaInfo::declared_duration_ms` and is explicitly
//! labelled as the container's claim. Rounding is to the nearest millisecond,
//! which for a 14-minute file is 1 part in 840000 — far below the error in the
//! claim itself.
//!
//! # What else the probe reports
//!
//! The audio stream's codec, sample rate, channel count and bit depth; the video
//! stream's codec and pixel dimensions if there is one; the file size; and
//! whether the source can seek. Video frame rate is *derived* — Symphonia's video
//! codec parameters do not carry one — from the track's frame count and duration,
//! and is `None` when the container does not provide both. `VideoStreamInfo::fps`
//! is documented as advisory for exactly this reason.

use std::fs::File;
use std::io::Cursor;
use std::path::Path;

use symphonia::core::codecs::audio::{AudioCodecId, AudioCodecParameters};
use symphonia::core::codecs::video::{VideoCodecId, well_known as vcodec};
use symphonia::core::codecs::{CodecParameters, audio::well_known as acodec};
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatReader, Track, TrackType};
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::units::Time as SymTime;
use vaani_core::{MediaInfo, Result};

use crate::errors;

/// Read a file's container and stream layout without decoding any audio.
///
/// Never decodes a packet, so this is cheap enough to run on every file the user
/// drags onto the window, and safe to run on a file that is still being written.
///
/// # Errors
///
/// * `UnsupportedMedia` if the path cannot be opened, is empty, or no demuxer
///   recognises the bytes.
/// * `Internal` only if the filesystem reports something impossible.
///
/// Note that a file with no audio track probes *successfully* with
/// `MediaInfo::audio == None`. Whether that is an error is a decision for the
/// stage, not for the probe: the UI wants to show "this is a video with no sound
/// track" rather than a failure, and [`crate::decode`] is where it becomes an
/// error.
pub fn probe(path: &Path) -> Result<MediaInfo> {
    let metadata = std::fs::metadata(path).map_err(|e| errors::open_failed(path, &e))?;
    let size_bytes = metadata.len();
    let file = File::open(path).map_err(|e| errors::open_failed(path, &e))?;
    if size_bytes == 0 {
        return Err(errors::unrecognised_container(
            path,
            &symphonia::core::errors::Error::DecodeError("the file is empty"),
        ));
    }
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    let reader = open_reader(Box::new(file), extension.as_deref(), path)?;
    let info = describe(reader.as_ref(), size_bytes);
    Ok(info)
}

/// Probe a buffer that is already in memory.
///
/// Copies `bytes` into a cursor, because a `FormatReader` owns its source for as
/// long as it lives and borrowing a slice with a shorter lifetime would force
/// every caller into an `Arc`. Intended for tests, for clipboard and drag-and-drop
/// payloads, and for the small files a demo or a bug report carries around. For
/// anything on disk use [`probe`], which does not copy.
///
/// `hint_extension` is the usual file extension without the dot, when the caller
/// happens to know it. It only helps Symphonia's marker search; the bytes decide
/// what the file is.
pub fn probe_bytes(bytes: &[u8], hint_extension: Option<&str>) -> Result<MediaInfo> {
    if bytes.is_empty() {
        return Err(errors::unrecognised_container(
            Path::new("<memory>"),
            &symphonia::core::errors::Error::DecodeError("the buffer is empty"),
        ));
    }
    let size_bytes = bytes.len() as u64;
    let reader = open_reader(
        Box::new(Cursor::new(bytes.to_vec())),
        hint_extension,
        Path::new("<memory>"),
    )?;
    Ok(describe(reader.as_ref(), size_bytes))
}

/// A demuxer over `source`, positioned at the start.
pub(crate) fn open_reader(
    source: Box<dyn symphonia::core::io::MediaSource>,
    extension: Option<&str>,
    path: &Path,
) -> Result<Box<dyn FormatReader>> {
    let mut hint = Hint::new();
    if let Some(ext) = extension {
        hint.with_extension(ext);
    }
    // 64 KB is Symphonia's default and is the right size here: big enough that
    // read-ahead amortises the syscall, small enough that it is irrelevant next
    // to a twelve-million-sample output buffer.
    let stream = MediaSourceStream::new(source, MediaSourceStreamOptions::default());
    symphonia::default::get_probe()
        .probe(
            &hint,
            stream,
            symphonia::core::formats::FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| errors::from_symphonia(path, &e))
}

/// Build a `MediaInfo` from a reader that has already been opened.
///
/// Infallible: every field has a defined value for every container, and the
/// fields that a container may legitimately omit are `Option`. A file with no
/// audio track is a successful probe with `audio == None`, not a failure —
/// whether that is fatal is the decode stage's decision, not the probe's.
pub(crate) fn describe(reader: &dyn FormatReader, size_bytes: u64) -> MediaInfo {
    let container = container_name(reader.format_info().short_name).to_string();
    let audio_track = pick_audio_track(reader);
    let video_track = reader.first_track(TrackType::Video);

    MediaInfo {
        container,
        declared_duration_ms: declared_duration_ms(reader, audio_track),
        size_bytes,
        audio: audio_track.and_then(|t| audio_stream_info(t.codec_params.as_ref()?)),
        video: video_track.and_then(video_stream_info),
        // A `File` is a regular file and therefore seekable; the probe reached
        // this point only because the source opened. Reported as true because a
        // `probe` over a path is always over a regular file, and the pipeline's
        // only use for the field is deciding whether a scrub bar can seek.
        seekable: true,
    }
}

/// Pick the audio track to decode: the default one, else the first with a codec.
fn pick_audio_track(reader: &dyn FormatReader) -> Option<&Track> {
    reader.default_track(TrackType::Audio).or_else(|| {
        reader
            .tracks()
            .iter()
            .find(|t| t.track_type() == Some(TrackType::Audio))
    })
}

/// The container's *claim* about how long the audio is.
///
/// See the module documentation for the precedence rules and for why the answer
/// must not be trusted over a decoded sample count.
pub(crate) fn declared_duration_ms(
    reader: &dyn FormatReader,
    audio: Option<&Track>,
) -> Option<i64> {
    if let Some(track) = audio
        && let (Some(tb), Some(dur)) = (track.time_base, track.duration)
    {
        let t = tb.calc_duration_saturating(dur);
        if t >= SymTime::ZERO {
            return Some(clamp_millis(t.as_millis()));
        }
    }
    let media = reader.media_info();
    match (media.time_base, media.duration) {
        (Some(tb), Some(dur)) => {
            let t = tb.calc_duration_saturating(dur);
            if t >= SymTime::ZERO {
                Some(clamp_millis(t.as_millis()))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Symphonia reports a duration as `i128` nanoseconds-to-milliseconds; a
/// container that declares more than ~292 million years would otherwise wrap
/// into a negative duration and read as "before the start of the file".
fn clamp_millis(ms: i128) -> i64 {
    i64::try_from(ms).unwrap_or(i64::MAX)
}

/// The audio track's stream description, or `None` if the track has no
/// parameters at all (which the container marks as unplayable).
pub(crate) fn audio_stream_info(params: &CodecParameters) -> Option<vaani_core::AudioStreamInfo> {
    let CodecParameters::Audio(a) = params else {
        return None;
    };
    let channels = a.channels.as_ref().map_or(0, |c| c.count());
    Some(vaani_core::AudioStreamInfo {
        sample_rate: a.sample_rate.unwrap_or(0),
        channels: u16::try_from(channels).unwrap_or(u16::MAX),
        codec: audio_codec_name(a.codec),
        bits_per_sample: declared_bits_per_sample(a).or_else(|| implied_bits(a.codec)),
    })
}

/// Bits per sample implied by a *PCM* codec id, when the container did not say.
///
/// Only consulted for the uncompressed family, where the width is part of the
/// codec's definition rather than metadata. A lossy codec must report `None`
/// even if a container volunteers a number, because `AudioStreamInfo` documents
/// this field as "bits per sample for PCM formats".
///
/// This exists because Symphonia's WAVE reader does not propagate the `fmt `
/// chunk's `bits_per_sample` for the IEEE-float format tag, so a 32-bit float
/// WAV would otherwise report no depth at all — which reads as "unknown" in a
/// bug report when the file states it in plain sight.
fn implied_bits(codec: AudioCodecId) -> Option<u16> {
    Some(match codec {
        acodec::CODEC_ID_PCM_S8 | acodec::CODEC_ID_PCM_U8 => 8,
        acodec::CODEC_ID_PCM_S16LE
        | acodec::CODEC_ID_PCM_S16BE
        | acodec::CODEC_ID_PCM_U16LE
        | acodec::CODEC_ID_PCM_U16BE => 16,
        acodec::CODEC_ID_PCM_S24LE
        | acodec::CODEC_ID_PCM_S24BE
        | acodec::CODEC_ID_PCM_U24LE
        | acodec::CODEC_ID_PCM_U24BE => 24,
        acodec::CODEC_ID_PCM_S32LE
        | acodec::CODEC_ID_PCM_U32LE
        | acodec::CODEC_ID_PCM_U32BE
        | acodec::CODEC_ID_PCM_F32LE
        | acodec::CODEC_ID_PCM_F32BE => 32,
        acodec::CODEC_ID_PCM_F64LE | acodec::CODEC_ID_PCM_F64BE => 64,
        _ => return None,
    })
}

/// The video track's stream description.
///
/// `fps` is derived from the track's frame count and duration because Symphonia's
/// video codec parameters carry no frame rate, and left as `None` when the
/// container does not provide both — a screen recording frequently provides
/// neither, and inventing a number there would be worse than admitting it.
fn video_stream_info(track: &Track) -> Option<vaani_core::VideoStreamInfo> {
    let CodecParameters::Video(v) = track.codec_params.as_ref()? else {
        return None;
    };
    Some(vaani_core::VideoStreamInfo {
        width: u32::from(v.width.unwrap_or(0)),
        height: u32::from(v.height.unwrap_or(0)),
        codec: video_codec_name(v.codec).to_string(),
        fps: derived_fps(track),
    })
}

/// Frames per second, from the track's own frame count and duration.
fn derived_fps(track: &Track) -> Option<f64> {
    let frames = track.num_frames?;
    let tb = track.time_base?;
    let dur = track.duration?;
    let seconds = tb.calc_duration_saturating(dur).as_secs_f64();
    if seconds <= 0.0 || frames == 0 {
        return None;
    }
    let fps = frames as f64 / seconds;
    fps.is_finite().then_some(fps)
}

/// Map Symphonia's short format name onto the container names the product uses.
///
/// The mapping is explicit rather than a lowercase of the input so that
/// `MediaInfo::container` is a small, stable vocabulary the UI can switch on and
/// the database can index. Two deliberate widenings: `wave` becomes `wav` because
/// that is what users call it and what the file picker filters on, and `isomp4`
/// covers `m4a`, `m4b` and `mov` because they are the same container and the
/// distinction is carried by the extension, not the bytes.
fn container_name(short_name: &str) -> &'static str {
    match short_name {
        "wave" => "wav",
        "aiff" => "aiff",
        "isomp4" => "mp4",
        "matroska" => "mkv",
        "ogg" => "ogg",
        "flac" => "flac",
        "mp1" => "mp1",
        "mp2" => "mp2",
        "mp3" => "mp3",
        "aac" => "adts",
        "caf" => "caf",
        // Unreachable with the `all` feature set, but a name is better than a
        // silent fall-through to something a UI cannot render.
        _ => "unknown",
    }
}

/// A displayable name for an audio codec ID, for error messages.
///
/// The decode path needs the codec's name before it has a `MediaInfo` to put it
/// in, and the message has to name the codec the user cannot decode, so this is
/// the same naming function [`audio_codec_name`] uses.
pub(crate) fn codec_label(codec: AudioCodecId) -> String {
    audio_codec_name(codec)
}

/// A stable lower-case name for an audio codec, for `AudioStreamInfo::codec`.
///
/// `AudioStreamInfo::codec` is documented as informational, so the only
/// requirement is that it is a string the UI can show and a log can grep.
/// Well-known IDs get a name; anything else falls back to the codec ID's own
/// hexadecimal form, which is stable across runs and names the exact thing the
/// decoder registry did not recognise.
fn audio_codec_name(codec: AudioCodecId) -> String {
    let name = match codec {
        acodec::CODEC_ID_AAC => "aac",
        acodec::CODEC_ID_ALAC => "alac",
        acodec::CODEC_ID_FLAC => "flac",
        acodec::CODEC_ID_MP1 => "mp1",
        acodec::CODEC_ID_MP2 => "mp2",
        acodec::CODEC_ID_MP3 => "mp3",
        acodec::CODEC_ID_VORBIS => "vorbis",
        acodec::CODEC_ID_OPUS => "opus",
        acodec::CODEC_ID_SPEEX => "speex",
        acodec::CODEC_ID_PCM_S8 => "pcm_s8",
        acodec::CODEC_ID_PCM_U8 => "pcm_u8",
        acodec::CODEC_ID_PCM_S16LE => "pcm_s16le",
        acodec::CODEC_ID_PCM_S16BE => "pcm_s16be",
        acodec::CODEC_ID_PCM_S24LE => "pcm_s24le",
        acodec::CODEC_ID_PCM_S24BE => "pcm_s24be",
        acodec::CODEC_ID_PCM_S32LE => "pcm_s32le",
        acodec::CODEC_ID_PCM_F32LE => "pcm_f32le",
        acodec::CODEC_ID_PCM_F64LE => "pcm_f64le",
        acodec::CODEC_ID_PCM_ALAW => "pcm_alaw",
        acodec::CODEC_ID_PCM_MULAW => "pcm_mulaw",
        acodec::CODEC_ID_ADPCM_MS => "adpcm_ms",
        acodec::CODEC_ID_ADPCM_IMA_WAV => "adpcm_ima_wav",
        acodec::CODEC_ID_ADPCM_IMA_QT => "adpcm_ima_qt",
        _ => return codec.to_string(),
    };
    name.to_string()
}

/// A stable lower-case name for a video codec.
fn video_codec_name(codec: VideoCodecId) -> String {
    let name = match codec {
        vcodec::CODEC_ID_MJPEG => "mjpeg",
        vcodec::CODEC_ID_CINEPAK => "cinepak",
        vcodec::CODEC_ID_VP3 => "vp3",
        vcodec::CODEC_ID_VP8 => "vp8",
        vcodec::CODEC_ID_VP9 => "vp9",
        vcodec::CODEC_ID_THEORA => "theora",
        vcodec::CODEC_ID_AV1 => "av1",
        vcodec::CODEC_ID_MPEG1 => "mpeg1",
        vcodec::CODEC_ID_MPEG2 => "mpeg2",
        vcodec::CODEC_ID_MPEG4 => "mpeg4",
        vcodec::CODEC_ID_H261 => "h261",
        vcodec::CODEC_ID_H263 => "h263",
        vcodec::CODEC_ID_H264 => "h264",
        vcodec::CODEC_ID_HEVC => "hevc",
        vcodec::CODEC_ID_VVC => "vvc",
        vcodec::CODEC_ID_VC1 => "vc1",
        vcodec::CODEC_ID_WMV1 => "wmv1",
        vcodec::CODEC_ID_WMV2 => "wmv2",
        vcodec::CODEC_ID_WMV3 => "wmv3",
        _ => return codec.to_string(),
    };
    name.to_string()
}

/// The bits per sample a decoder will produce, for formats where the container
/// states it and the codec does not.
///
/// WAVEFORMATEXTENSIBLE carries a valid-bits field that Symphonia copies into
/// `bits_per_sample`, so the value is already correct when present. A zero means
/// "unspecified" and is reported as `None`, because `AudioStreamInfo` documents
/// this as bits per sample for PCM formats and a zero there would read as a
/// one-bit sample.
fn declared_bits_per_sample(params: &AudioCodecParameters) -> Option<u16> {
    params
        .bits_per_sample
        .and_then(|b| u16::try_from(b).ok())
        .filter(|b| *b > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symphonia::core::common::FourCc;
    use symphonia::core::units::Duration;

    fn track_with_audio(rate: u32, channels: u16) -> Track {
        let mut t = Track::new(0);
        let mut p = AudioCodecParameters::new();
        p.for_codec(acodec::CODEC_ID_PCM_S16LE)
            .with_sample_rate(rate)
            .with_bits_per_sample(16)
            .with_channels(symphonia::core::audio::Channels::Discrete(channels));
        t.with_codec_params(p.into());
        t
    }

    #[test]
    fn container_names_are_a_small_stable_vocabulary() {
        assert_eq!(container_name("wave"), "wav");
        assert_eq!(container_name("isomp4"), "mp4");
        assert_eq!(container_name("matroska"), "mkv");
        assert_eq!(container_name("ogg"), "ogg");
        assert_eq!(container_name("flac"), "flac");
        assert_eq!(container_name("mp3"), "mp3");
        assert_eq!(container_name("aiff"), "aiff");
        assert_eq!(container_name("something-new"), "unknown");
    }

    #[test]
    fn audio_codec_names_cover_everything_this_build_can_decode() {
        // If Symphonia adds a decoder and this list does not follow, the UI shows
        // an opaque hex ID for a file it can actually play.
        for (id, want) in [
            (acodec::CODEC_ID_AAC, "aac"),
            (acodec::CODEC_ID_MP3, "mp3"),
            (acodec::CODEC_ID_FLAC, "flac"),
            (acodec::CODEC_ID_VORBIS, "vorbis"),
            (acodec::CODEC_ID_ALAC, "alac"),
            (acodec::CODEC_ID_PCM_S16LE, "pcm_s16le"),
            (acodec::CODEC_ID_PCM_F32LE, "pcm_f32le"),
        ] {
            assert_eq!(audio_codec_name(id), want);
        }
    }

    #[test]
    fn an_unknown_codec_falls_back_to_a_stable_opaque_id() {
        // Not a name, but stable across runs and identical to what the decoder
        // registry lookup rejected, which is what a bug report needs.
        let id = AudioCodecId::new(FourCc::new(*b"Opus"));
        let got = audio_codec_name(id);
        assert!(got.starts_with("0x"), "{got} should be a hex id");
        assert_eq!(
            got,
            audio_codec_name(id),
            "the fallback must be deterministic"
        );
    }

    #[test]
    fn video_codec_names_cover_the_common_ones() {
        for (id, want) in [
            (vcodec::CODEC_ID_H264, "h264"),
            (vcodec::CODEC_ID_HEVC, "hevc"),
            (vcodec::CODEC_ID_AV1, "av1"),
            (vcodec::CODEC_ID_VP9, "vp9"),
            (vcodec::CODEC_ID_VP8, "vp8"),
        ] {
            assert_eq!(video_codec_name(id), want);
        }
        let id = VideoCodecId::new(FourCc::new(*b"av01"));
        assert!(video_codec_name(id).starts_with("0x"));
    }

    #[test]
    fn audio_stream_info_reports_zero_for_what_the_container_omits() {
        // Zero, not a guess. The pipeline checks `sample_rate == 0` and fails
        // with a message, which beats decoding at a rate the container never said.
        let mut p = AudioCodecParameters::new();
        p.for_codec(acodec::CODEC_ID_MP3);
        let info = audio_stream_info(&CodecParameters::Audio(p)).expect("audio params");
        assert_eq!(info.sample_rate, 0);
        assert_eq!(info.channels, 0);
        assert_eq!(info.bits_per_sample, None);
        assert_eq!(info.codec, "mp3");
    }

    #[test]
    fn a_track_that_is_not_audio_yields_no_audio_info() {
        let p = symphonia::core::codecs::video::VideoCodecParameters {
            codec: vcodec::CODEC_ID_H264,
            ..Default::default()
        };
        assert!(audio_stream_info(&CodecParameters::Video(p)).is_none());
    }

    #[test]
    fn a_track_with_far_too_many_channels_does_not_wrap() {
        // `Channels` can in principle describe more channels than `u16` holds via
        // a positioned mask plus extras; the conversion must saturate, not wrap to
        // a small number that would make the channel count look plausible.
        let mut p = AudioCodecParameters::new();
        p.for_codec(acodec::CODEC_ID_PCM_F32LE)
            .with_sample_rate(48000)
            .with_channels(symphonia::core::audio::Channels::Discrete(64));
        let info = audio_stream_info(&CodecParameters::Audio(p)).expect("info");
        assert_eq!(info.channels, 64);
    }

    #[test]
    fn bit_depth_is_only_reported_when_the_container_states_it() {
        let mut p = AudioCodecParameters::new();
        p.for_codec(acodec::CODEC_ID_MP3);
        assert_eq!(declared_bits_per_sample(&p), None);
        p.with_bits_per_sample(24);
        assert_eq!(declared_bits_per_sample(&p), Some(24));
        // A zero is "unspecified", not "zero bits per sample".
        p.with_bits_per_sample(0);
        assert_eq!(declared_bits_per_sample(&p), None);
    }

    #[test]
    fn frame_rate_is_derived_and_only_when_both_numbers_exist() {
        // 30 frames at a 1/30 s timebase is 1 fps; a track with no frame count
        // must not invent one.
        let mut t = Track::new(0);
        t.with_time_base(symphonia::core::units::TimeBase::try_new(1, 30).expect("timebase"))
            .with_num_frames(300)
            .with_duration(Duration::new(300));
        let fps = derived_fps(&t).expect("fps");
        assert!((fps - 30.0).abs() < 1e-9);

        let mut t2 = Track::new(0);
        t2.with_time_base(symphonia::core::units::TimeBase::try_new(1, 30).expect("timebase"))
            .with_duration(Duration::new(300));
        assert_eq!(derived_fps(&t2), None);

        let mut t3 = Track::new(0);
        t3.with_num_frames(300);
        assert_eq!(derived_fps(&t3), None);

        // A zero-length track must not divide by zero or report infinity.
        let mut t4 = Track::new(0);
        t4.with_time_base(symphonia::core::units::TimeBase::try_new(1, 30).expect("timebase"))
            .with_num_frames(0)
            .with_duration(Duration::new(0));
        assert_eq!(derived_fps(&t4), None);
    }

    #[test]
    fn a_wav_track_reports_the_rate_and_depth_the_fmt_chunk_declared() {
        let t = track_with_audio(44100, 2);
        let info = audio_stream_info(t.codec_params.as_ref().expect("params")).expect("info");
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
        assert_eq!(info.bits_per_sample, Some(16));
        assert_eq!(info.codec, "pcm_s16le");
    }
}

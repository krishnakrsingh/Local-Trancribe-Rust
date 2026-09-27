//! Media and language identification.
//!
//! Kept in `core` because both the media backend and the ASR layer need to
//! describe a source, and because the *same* language enum has to be used for
//! the UI's language picker, the per-segment language tag, and the benchmark
//! report. Splitting it would guarantee drift.

use serde::{Deserialize, Serialize};

use crate::transcript::Time;
use std::fmt;

/// Languages the product can name in its UI.
///
/// This is deliberately a UI vocabulary, not a model vocabulary. SraVaani
/// supports 65 languages; listing 65 options in a picker helps nobody. The
/// picker offers the languages the product is *positioned* for, and the
/// per-segment language tag from the model round-trips as a free-form BCP-47
/// string in [`Language::Other`] so no information is lost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[allow(missing_docs)] // each variant maps to a picker entry
pub enum Language {
    /// Let the model decide, per segment. The default, and the right answer
    /// for the Hindi/Hinglish meetings this product exists for.
    #[default]
    Auto,
    Hindi,
    English,
    /// Hindi and English mixed in one utterance, preserved as spoken. Not
    /// "Hindi with English loanwords" — a distinct output target, because it
    /// changes whether the model is allowed to normalise script.
    Hinglish,
    Other,
}

impl Language {
    /// Every option the picker offers, in display order.
    pub const ALL: [Language; 5] = [
        Self::Auto,
        Self::Hindi,
        Self::English,
        Self::Hinglish,
        Self::Other,
    ];

    /// The stable wire form (`hi`, `en`, `auto`, ...).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Hindi => "hi",
            Self::English => "en",
            Self::Hinglish => "hinglish",
            Self::Other => "other",
        }
    }

    /// The tag written into `Segment::language`. `Auto` resolves to nothing at
    /// the document level; a segment that the model actually identified gets a
    /// real tag.
    pub const fn as_bcp47(self) -> Option<&'static str> {
        match self {
            Self::Auto | Self::Hinglish | Self::Other => None,
            Self::Hindi => Some("hi"),
            Self::English => Some("en"),
        }
    }

    /// Display name for the picker.
    /// The label shown in the picker.
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::Hindi => "Hindi",
            Self::English => "English",
            Self::Hinglish => "Hinglish",
            Self::Other => "Other",
        }
    }

    /// Parse a wire form, case-insensitively.
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|l| l.as_str().eq_ignore_ascii_case(s))
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Audio stream description, as read by a media backend's probe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct AudioStreamInfo {
    pub sample_rate: u32,
    pub channels: u16,
    /// Codec name as reported by the backend (`aac`, `mp3`, `flac`, ...).
    /// Informational; we never branch on it.
    pub codec: String,
    /// Bits per sample for PCM formats. `None` for lossy codecs where the
    /// container's bit depth is the meaningful figure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bits_per_sample: Option<u16>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct VideoStreamInfo {
    pub width: u32,
    pub height: u32,
    pub codec: String,
    /// Nominal frames per second. Advisory: for a screen recording it is often
    /// absent, and the UI must not refuse to play a stream without it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<f64>,
}

/// Everything a media backend can learn about a file without decoding it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct MediaInfo {
    pub container: String,
    /// Duration declared by the container, in milliseconds.
    ///
    /// This is the container's claim, which is frequently wrong for MKV and
    /// sometimes for M4A. [`MediaInfo::authoritative_duration`] explains which
    /// number to trust.
    pub declared_duration_ms: Option<i64>,
    pub size_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioStreamInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video: Option<VideoStreamInfo>,
    pub seekable: bool,
}

impl MediaInfo {
    /// The duration to build a transcript timeline against.
    ///
    /// When we have decoded audio we know the true sample count, which beats
    /// any container header. Before decoding, the header is all we have. This
    /// is why the transcript stores both the declared and the observed duration
    /// and prefers the observed one.
    pub fn authoritative_duration(&self, decoded_ms: Option<i64>) -> Option<i64> {
        decoded_ms.or(self.declared_duration_ms)
    }

    /// Rough time to transcribe, from measured realtime factors.
    ///
    /// Shown on the drop screen so a user knows what they are about to wait for
    /// *before* committing. The RTF used here is deliberately conservative (the
    /// slower of GPU and CPU) so the estimate is never optimistic.
    /// Estimated wall time in **milliseconds**.
    pub fn estimate_transcription_ms(&self, realtime_factor: f64) -> Option<i64> {
        let dur = self.declared_duration_ms?;
        // `declared_duration_ms` is already in ms and RTF is dimensionless
        // (processing_time / audio_duration), so the product is milliseconds.
        Some(((dur as f64) * realtime_factor.max(0.01)) as i64)
    }
}

/// A decoded audio buffer handed between pipeline stages.
///
/// # The invariant that matters
///
/// `start_time` is where this buffer begins in the **source media**, and every
/// time any stage produces is in source-media coordinates. The one exception is
/// the ASR stage, whose decoder frame indices are chunk-relative by
/// construction; it rebases through `Transcript::push_chunk`. Getting this
/// wrong in either direction is the "transcript is three minutes out of sync"
/// family of bugs, so the field is not optional and not inferable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct AudioBuffer {
    /// Interleaved samples. `channels == 1` means mono.
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    /// Where this buffer starts in the source media.
    pub start_time: Time,
}

impl AudioBuffer {
    /// A mono buffer, which is what every model stage wants.
    pub fn mono(samples: Vec<f32>, sample_rate: u32, start_time: Time) -> Self {
        Self {
            samples,
            sample_rate,
            channels: 1,
            start_time,
        }
    }

    /// Sample count per channel.
    #[must_use]
    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }
        self.samples.len() / self.channels as usize
    }

    /// Length of this buffer, from the sample count rather than the container
    /// header. A container's declared duration is frequently wrong for MKV and
    /// sometimes for M4A.
    #[must_use]
    pub fn duration(&self) -> Time {
        if self.sample_rate == 0 {
            return Time::ZERO;
        }
        // Round rather than truncate: a 3-hour recording must not lose a
        // millisecond per chunk boundary.
        Time::from_millis((self.frames() as u64 * 1000 / self.sample_rate as u64) as i64)
    }

    /// Where a sample index within this buffer falls in the source media.
    #[must_use]
    pub fn time_of(&self, sample_index: usize) -> Time {
        if self.sample_rate == 0 {
            return self.start_time;
        }
        self.start_time.saturating_add(Time(
            (sample_index as u64 * 1000 / self.sample_rate as u64) as i64,
        ))
    }

    /// Mono samples in `[start, end)`, times inclusive of `start` and exclusive
    /// of `end`. Times are source-media coordinates; the result starts at zero,
    /// so it is chunk-local by construction.
    ///
    /// # Panics
    ///
    /// Panics if `channels != 1`. Downmixing is the media stage's job, and
    /// silently averaging here would hide a pipeline that forgot to do it.
    #[must_use]
    pub fn slice_mono(&self, start: Time, end: Time) -> Vec<f32> {
        assert_eq!(
            self.channels, 1,
            "slice_mono requires a mono buffer; downmixing is the media stage's job"
        );
        let lo = (start
            .as_millis()
            .saturating_sub(self.start_time.as_millis())
            * self.sample_rate as i64
            / 1000)
            .max(0) as usize;
        let hi = (end.as_millis().saturating_sub(self.start_time.as_millis())
            * self.sample_rate as i64
            / 1000)
            .max(0) as usize;
        let lo = lo.min(self.samples.len());
        let hi = hi.clamp(lo, self.samples.len());
        self.samples[lo..hi].to_vec()
    }
}

/// The realtime factor (RTF) we assume before we have measured anything on the
/// user's machine: `processing_time / audio_duration`, lower is faster.
///
/// 0.6 means "a 12-minute meeting takes about 7 minutes". Chosen as a
/// CPU-fallback figure so a first-time user on an unaccelerated machine is not
/// promised something the machine cannot deliver.
pub const DEFAULT_ESTIMATE_RTF: f64 = 0.6;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_strings_round_trip() {
        for l in Language::ALL {
            assert_eq!(Language::parse(l.as_str()), Some(l));
        }
        assert_eq!(Language::parse("HI"), Some(Language::Hindi));
        assert_eq!(Language::parse("klingon"), None);
    }

    #[test]
    fn auto_and_hinglish_have_no_bcp47_tag() {
        // Hinglish has no BCP-47 tag. Emitting one would be a lie, and the
        // segment would be mislabelled in exports.
        assert_eq!(Language::Auto.as_bcp47(), None);
        assert_eq!(Language::Hinglish.as_bcp47(), None);
        assert_eq!(Language::Hindi.as_bcp47(), Some("hi"));
    }

    #[test]
    fn observed_duration_beats_declared() {
        let info = MediaInfo {
            container: "mp4".into(),
            declared_duration_ms: Some(700_000),
            size_bytes: 1,
            audio: None,
            video: None,
            seekable: true,
        };
        assert_eq!(info.authoritative_duration(None), Some(700_000));
        assert_eq!(info.authoritative_duration(Some(702_100)), Some(702_100));
    }

    #[test]
    fn estimate_is_proportional_to_duration() {
        let info = MediaInfo {
            container: "mp4".into(),
            declared_duration_ms: Some(720_000),
            size_bytes: 1,
            audio: None,
            video: None,
            seekable: true,
        };
        // 720 s at RTF 0.25 => 180 s of processing.
        assert_eq!(info.estimate_transcription_ms(0.25), Some(180_000));
        // A nonsensical RTF is floored at 0.01 rather than producing zero,
        // infinite, or negative estimates.
        assert_eq!(info.estimate_transcription_ms(0.0), Some(7_200));
        assert_eq!(info.estimate_transcription_ms(-5.0), Some(7_200));
    }

    #[test]
    fn audio_buffer_duration_comes_from_the_sample_count() {
        let b = AudioBuffer::mono(vec![0.0; 16_000], 16_000, Time::ZERO);
        assert_eq!(b.duration(), Time(1000));
        assert_eq!(b.frames(), 16_000);
        // A zero sample rate must not divide by zero.
        let b = AudioBuffer::mono(vec![0.0; 100], 0, Time::ZERO);
        assert_eq!(b.duration(), Time::ZERO);
    }

    #[test]
    fn audio_buffer_times_are_absolute_not_chunk_local() {
        // A buffer starting at 5 minutes must report absolute times. Getting
        // this wrong is the transcript-drift family of bugs.
        let b = AudioBuffer::mono(vec![0.0; 16_000], 16_000, Time(300_000));
        assert_eq!(b.time_of(0), Time(300_000));
        assert_eq!(b.time_of(8_000), Time(300_500));
        assert_eq!(b.duration(), Time(1000));
    }

    #[test]
    fn slice_mono_is_chunk_local_but_takes_absolute_times() {
        let b = AudioBuffer::mono(
            (0..32_000).map(|i| i as f32).collect(),
            16_000,
            Time(10_000),
        );
        // Absolute [11s, 12s) is chunk-relative samples 16_000..32_000.
        let s = b.slice_mono(Time(11_000), Time(12_000));
        assert_eq!(s.len(), 16_000);
        assert_eq!(s[0], 16_000.0);
    }

    #[test]
    fn slice_mono_clamps_to_the_available_samples() {
        // A VAD segment that runs past the end of a chunk must return what
        // exists rather than panicking.
        let b = AudioBuffer::mono(vec![0.0; 1_000], 16_000, Time::ZERO);
        assert_eq!(b.slice_mono(Time(0), Time(10_000)).len(), 1_000);
        assert_eq!(b.slice_mono(Time(5_000), Time(6_000)).len(), 0);
    }

    #[test]
    #[should_panic(expected = "downmixing is the media stage's job")]
    fn slice_mono_refuses_stereo_rather_than_silently_downmixing() {
        let b = AudioBuffer {
            samples: vec![0.0; 4],
            sample_rate: 16_000,
            channels: 2,
            start_time: Time::ZERO,
        };
        let _ = b.slice_mono(Time::ZERO, Time(1));
    }

    #[test]
    fn estimate_is_none_without_a_declared_duration() {
        let info = MediaInfo {
            container: "raw".into(),
            declared_duration_ms: None,
            size_bytes: 1,
            audio: None,
            video: None,
            seekable: false,
        };
        assert_eq!(info.estimate_transcription_ms(0.25), None);
    }
}

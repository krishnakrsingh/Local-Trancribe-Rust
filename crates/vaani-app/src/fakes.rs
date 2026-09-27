//! Test doubles: a fake VAD, ASR engine, diarizer and ASR provider.
//!
//! Compiled for tests only. The fakes honour [`CancelToken`] exactly like the
//! real engines (per-chunk checks, blocking-until-cancelled transcription) so
//! cancellation and progress tests exercise the real paths.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use vaani_asr::engine::{AsrEngine, AsrTimings, AudioChunk, ChunkResult};
use vaani_core::media::AudioBuffer;
use vaani_core::pipeline::{
    DiarizationResult, SpeakerSpan, SpeechSegment, SpeechTimeline, VadEngine,
};
use vaani_core::{err, CancelToken, Language, Result, Segment, Time, Word};

use crate::orchestrator::AsrProvider;
use crate::stages::Diarize;
use crate::AppConfig;

/// One word per 500 ms of chunk, chunk-local times.
pub fn segment(text: &str, start_ms: i64, end_ms: i64) -> Segment {
    let words: Vec<Word> = text
        .split_whitespace()
        .enumerate()
        .map(|(index, word)| {
            let span = (end_ms - start_ms) / text.split_whitespace().count().max(1) as i64;
            Word::new(
                word,
                Time(start_ms + index as i64 * span),
                Time(start_ms + (index as i64 + 1) * span),
            )
        })
        .collect();
    let (start, end) = words
        .iter()
        .fold((Time(end_ms), Time(start_ms)), |(lo, hi), word| {
            (lo.min(word.start), hi.max(word.end))
        });
    Segment::new(start, end).with_words(words)
}

/// A mono 16 kHz sine buffer of `seconds` seconds.
pub fn sine_audio(seconds: u64) -> AudioBuffer {
    let frames = 16_000_usize * seconds as usize;
    let samples: Vec<f32> = (0..frames)
        .map(|i| (i as f32 / 16_000.0 * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.5)
        .collect();
    AudioBuffer::mono(samples, 16_000, Time::ZERO)
}

/// Write `seconds` of mono 16 kHz 16-bit PCM WAV to `path`.
///
/// The pipeline probes the real input file before the injected decoder runs,
/// so tests need a container the media backend actually recognises. `tag`
/// varies the tone so different tests hash differently.
pub fn write_wav(path: &std::path::Path, seconds: u64, tag: u64) {
    let frames = 16_000_usize * seconds as usize;
    let data_len = frames * 2;
    let mut bytes = Vec::with_capacity(44 + data_len);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len as u32).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&16_000u32.to_le_bytes());
    bytes.extend_from_slice(&32_000u32.to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(data_len as u32).to_le_bytes());
    let frequency = 440.0 + (tag % 8) as f32 * 55.0;
    for i in 0..frames {
        let sample = (i as f32 / 16_000.0 * frequency * 2.0 * std::f32::consts::PI).sin() * 0.4;
        let quantized = (sample * i16::MAX as f32) as i16;
        bytes.extend_from_slice(&quantized.to_le_bytes());
    }
    std::fs::write(path, bytes).unwrap();
}

/// A fake [`VadEngine`]: the whole buffer is speech, or silence.
#[derive(Debug, Clone)]
pub struct FakeVad {
    speech: bool,
}

impl FakeVad {
    /// Detects the whole buffer as one speech segment.
    #[must_use]
    pub fn speech() -> Self {
        Self { speech: true }
    }

    /// Detects silence.
    #[must_use]
    pub fn silence() -> Self {
        Self { speech: false }
    }
}

impl VadEngine for FakeVad {
    fn id(&self) -> &str {
        "fake-vad"
    }

    fn name(&self) -> &str {
        "Fake VAD"
    }

    fn sample_rate(&self) -> u32 {
        16_000
    }

    fn detect(&self, audio: &AudioBuffer, cancel: &CancelToken) -> Result<SpeechTimeline> {
        cancel.check()?;
        if !self.speech || audio.samples.is_empty() {
            return Ok(SpeechTimeline::default());
        }
        let end = audio.duration();
        Ok(SpeechTimeline {
            segments: vec![SpeechSegment {
                start: Time::ZERO,
                end,
                confidence: None,
            }],
            speech_duration_ms: end.as_millis(),
        })
    }

    fn cancel(&self) {}
}

/// A fake [`AsrEngine`]: one segment per chunk with deterministic words.
///
/// * `blocking(true)` makes `transcribe` wait for cancellation, for the
///   cancel-mid-ASR test.
/// * Every call is counted in [`FakeAsr::calls`].
#[derive(Debug)]
pub struct FakeAsr {
    inner: Arc<FakeAsrInner>,
}

#[derive(Debug)]
struct FakeAsrInner {
    calls: AtomicUsize,
    blocking: AtomicUsize,
}

impl FakeAsr {
    /// A non-blocking fake.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(FakeAsrInner {
                calls: AtomicUsize::new(0),
                blocking: AtomicUsize::new(0),
            }),
        }
    }

    /// Make `transcribe` block until the token is cancelled.
    pub fn blocking(&self, blocking: bool) {
        self.inner
            .blocking
            .store(usize::from(blocking), Ordering::SeqCst);
    }

    /// How many times `transcribe` was entered.
    #[must_use]
    pub fn calls(&self) -> usize {
        self.inner.calls.load(Ordering::SeqCst)
    }
}

impl Default for FakeAsr {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for FakeAsr {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl AsrEngine for FakeAsr {
    fn id(&self) -> &str {
        "fake-asr"
    }

    fn model_name(&self) -> &str {
        "fake-asr"
    }

    fn model_version(&self) -> &str {
        "test.0"
    }

    fn languages(&self) -> &[Language] {
        &[]
    }

    fn load(&self, _cancel: &CancelToken) -> Result<()> {
        Ok(())
    }

    fn unload(&self) {}

    fn is_loaded(&self) -> bool {
        true
    }

    fn transcribe(&self, chunk: &AudioChunk, cancel: &CancelToken) -> Result<ChunkResult> {
        self.inner.calls.fetch_add(1, Ordering::SeqCst);
        if self.inner.blocking.load(Ordering::SeqCst) != 0 {
            loop {
                cancel.check()?;
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        cancel.check()?;
        let duration_ms = chunk.duration().as_millis().max(0);
        let end_ms = duration_ms.min(2000);
        if end_ms <= 0 {
            return Ok(ChunkResult::default());
        }
        Ok(ChunkResult {
            segments: vec![segment("hello world", 0, end_ms)],
            language: None,
            timings: AsrTimings::default(),
        })
    }

    fn cancel(&self) {}

    fn benchmark(&self, _cancel: &CancelToken) -> Result<AsrTimings> {
        Ok(AsrTimings::default())
    }
}

/// An [`AsrProvider`] that always hands out the same fake engine.
#[derive(Debug, Clone)]
pub struct FixedAsrProvider {
    engine: FakeAsr,
}

impl FixedAsrProvider {
    /// A provider cloning `engine` per job (call counts are shared).
    #[must_use]
    pub fn new(engine: FakeAsr) -> Self {
        Self { engine }
    }
}

impl AsrProvider for FixedAsrProvider {
    fn create(&self, _config: &AppConfig, cancel: &CancelToken) -> Result<Box<dyn AsrEngine>> {
        cancel.check()?;
        Ok(Box::new(self.engine.clone()))
    }
}

/// An [`AsrProvider`] that fails engine creation, for the model-missing path.
#[derive(Debug, Clone, Default)]
pub struct FailingAsrProvider;

impl AsrProvider for FailingAsrProvider {
    fn create(&self, _config: &AppConfig, cancel: &CancelToken) -> Result<Box<dyn AsrEngine>> {
        cancel.check()?;
        Err(err::model_unavailable().with_cause("fake model is not installed"))
    }
}

/// A fake [`Diarize`]: splits the audio into halves, one speaker each.
#[derive(Debug, Clone, Default)]
pub struct FakeDiar;

impl FakeDiar {
    /// A diarizer splitting at the midpoint.
    #[must_use]
    pub fn halves() -> Self {
        Self
    }
}

impl Diarize for FakeDiar {
    fn engine_id(&self) -> &'static str {
        "fake-diar"
    }

    fn diarize(
        &self,
        audio: &AudioBuffer,
        _speech: Option<&SpeechTimeline>,
        cancel: &CancelToken,
        progress: &dyn Fn(f64),
    ) -> Result<DiarizationResult> {
        cancel.check()?;
        let total = audio.duration().as_millis();
        progress(1.0);
        if total <= 0 {
            return Ok(DiarizationResult::default());
        }
        let mid = total / 2;
        Ok(DiarizationResult {
            spans: vec![
                SpeakerSpan {
                    speaker: 0,
                    start: Time::ZERO,
                    end: Time(mid),
                    score: None,
                },
                SpeakerSpan {
                    speaker: 1,
                    start: Time(mid),
                    end: Time(total),
                    score: None,
                },
            ],
            num_speakers: 2,
            elapsed_ms: 0,
        })
    }
}

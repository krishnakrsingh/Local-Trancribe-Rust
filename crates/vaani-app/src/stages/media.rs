//! The media stage: probe the container, decode to mono 16 kHz.
//!
//! The only stage that opens the source file. It validates with
//! [`probe`](vaani_media::probe) first so an unreadable file fails fast with
//! `UnsupportedMedia`, then decodes with progress through
//! [`decode_to_mono16k_with_progress`](vaani_media::decode_to_mono16k_with_progress).
//!
//! Transient decode failures (`AudioDecodeFailed`, e.g. a file that is still
//! being copied) are retried; anything else — including cancellation — returns
//! immediately.

use std::path::Path;
use std::sync::Arc;

use vaani_core::media::AudioBuffer;
use vaani_core::pipeline::{PipelineStage, ProgressSink, StageContext};
use vaani_core::{err, CancelToken, ErrorCode, JobPhase, Result, VaaniError};

/// An injectable decode function, so tests can simulate a flaky disk without
/// one.
pub type DecodeFn =
    Arc<dyn Fn(&Path, &CancelToken, &ProgressSink) -> Result<AudioBuffer> + Send + Sync>;

/// Probe first so a non-media file fails before any decode work starts.
fn default_decode(
    path: &Path,
    cancel: &CancelToken,
    progress: &ProgressSink,
) -> Result<AudioBuffer> {
    let _info = vaani_media::probe(path)?;
    vaani_media::decode_to_mono16k_with_progress(path, cancel, progress)
}

/// Only `AudioDecodeFailed` is worth a second attempt. A missing file, an
/// unknown container, a stream with no audio, or a cancellation will fail the
/// same way twice.
fn is_transient(error: &VaaniError) -> bool {
    error.code() == ErrorCode::AudioDecodeFailed
}

/// The `probe/decode` step of the pipeline.
pub struct MediaStage {
    decoder: DecodeFn,
    max_attempts: u32,
}

impl MediaStage {
    /// Probe plus the real decoder, with two total attempts.
    #[must_use]
    pub fn new() -> Self {
        Self {
            decoder: Arc::new(default_decode),
            max_attempts: 2,
        }
    }

    /// A stage with an explicit decoder and attempt budget, for tests.
    pub fn with_decoder(decoder: DecodeFn, max_attempts: u32) -> Self {
        Self {
            decoder,
            max_attempts: max_attempts.max(1),
        }
    }

    /// Total decode attempts.
    #[must_use]
    pub fn max_attempts(&self) -> u32 {
        self.max_attempts
    }
}

impl Default for MediaStage {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for MediaStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MediaStage")
            .field("max_attempts", &self.max_attempts)
            .finish_non_exhaustive()
    }
}

impl PipelineStage for MediaStage {
    fn name(&self) -> &'static str {
        "media"
    }

    fn phase(&self) -> JobPhase {
        JobPhase::ExtractingAudio
    }

    fn run(&self, ctx: &mut StageContext<'_>) -> Result<()> {
        ctx.check()?;
        let mut last_error: Option<VaaniError> = None;
        for attempt in 1..=self.max_attempts {
            ctx.check()?;
            match (self.decoder)(ctx.input, ctx.cancel, ctx.progress) {
                Ok(audio) => {
                    debug_assert_eq!(
                        audio.channels, 1,
                        "the decoder must hand downstream mono audio"
                    );
                    ctx.artifacts.audio = Some(audio);
                    return Ok(());
                }
                Err(error) if error.is_cancelled() => return Err(error),
                Err(error) if is_transient(&error) && attempt < self.max_attempts => {
                    tracing::warn!(
                        attempt,
                        total = self.max_attempts,
                        error = error.debug_chain(),
                        "transient decode failure, retrying"
                    );
                    last_error = Some(error);
                }
                Err(error) => return Err(error),
            }
        }
        Err(last_error.unwrap_or_else(err::audio_decode_failed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use vaani_core::pipeline::StageArtifacts;
    use vaani_core::Time;

    fn audio() -> AudioBuffer {
        AudioBuffer::mono(vec![0.0; 16_000], 16_000, Time::ZERO)
    }

    fn run_with(decoder: DecodeFn, attempts: u32) -> (Result<()>, StageArtifacts) {
        let stage = MediaStage::with_decoder(decoder, attempts);
        let mut artifacts = StageArtifacts::default();
        let cancel = CancelToken::new();
        let dir = std::env::temp_dir();
        let result = {
            let mut ctx = StageContext {
                input: Path::new("input.wav"),
                work_dir: &dir,
                cancel: &cancel,
                progress: &|_| {},
                artifacts: &mut artifacts,
            };
            stage.run(&mut ctx)
        };
        (result, artifacts)
    }

    #[test]
    fn transient_failure_is_retried_then_succeeds() {
        let calls = Arc::new(AtomicUsize::new(0));
        let decoder_calls = Arc::clone(&calls);
        let decoder: DecodeFn = Arc::new(move |_, _, _| {
            if decoder_calls.fetch_add(1, Ordering::SeqCst) == 0 {
                Err(err::audio_decode_failed())
            } else {
                Ok(audio())
            }
        });
        let (result, artifacts) = run_with(decoder, 2);
        assert!(result.is_ok());
        assert!(artifacts.audio.is_some());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn persistent_transient_failure_uses_the_whole_budget() {
        let calls = Arc::new(AtomicUsize::new(0));
        let decoder_calls = Arc::clone(&calls);
        let decoder: DecodeFn = Arc::new(move |_, _, _| {
            decoder_calls.fetch_add(1, Ordering::SeqCst);
            Err(err::audio_decode_failed())
        });
        let (result, _) = run_with(decoder, 3);
        assert!(result.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn non_transient_failure_is_not_retried() {
        let calls = Arc::new(AtomicUsize::new(0));
        let decoder_calls = Arc::clone(&calls);
        let decoder: DecodeFn = Arc::new(move |_, _, _| {
            decoder_calls.fetch_add(1, Ordering::SeqCst);
            Err(err::unsupported_media())
        });
        let (result, _) = run_with(decoder, 3);
        assert!(result.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn stage_reports_its_name_and_phase() {
        let stage = MediaStage::new();
        assert_eq!(stage.name(), "media");
        assert_eq!(stage.phase(), JobPhase::ExtractingAudio);
        assert!(stage.is_available().is_ok());
    }
}

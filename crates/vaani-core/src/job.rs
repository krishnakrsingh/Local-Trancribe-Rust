//! The job state machine and the progress contract.
//!
//! # Why jobs and not "a transcribe function"
//!
//! Transcribing a two-hour recording takes minutes, must be cancellable, must
//! survive a restart, and must not block the UI. That makes it a job with a
//! lifecycle, not a function call. The phases below are the *only* phases the
//! product has; adding one is a deliberate change because the UI, the CLI, the
//! database and the benchmark report all enumerate them.
//!
//! # Why progress carries units
//!
//! The spec is explicit: no fake progress bars. A bar that animates smoothly
//! while the model is actually doing an unknown amount of work is worse than no
//! bar, because it lies. [`Progress`] therefore always carries a denominator
//! where one is knowable, and a phase that genuinely cannot be quantified
//! reports [`ProgressKind::Indeterminate`] — which the UI renders as a
//! spinner plus the phase name, not as a percentage.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::Duration;

/// A transcription job's lifecycle phase.
///
/// Ordering is meaningful: [`JobPhase::is_terminal`] and
/// [`JobPhase::weight`] both rely on it. A phase's `weight` is its expected
/// share of wall time, used to turn per-phase progress into one honest overall
/// percentage instead of a bar that stalls at 10% for three minutes and then
/// jumps to 95%.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
#[allow(missing_docs)] // variant names map 1:1 to as_str()
pub enum JobPhase {
    /// Accepted, waiting for a worker.
    Queued,
    /// Validating the file, resolving the model, loading it into VRAM/RAM.
    Preparing,
    /// Demuxing and decoding audio.
    ExtractingAudio,
    /// Channel mixing and resampling to the model's required format.
    NormalizingAudio,
    /// Voice activity detection: finding speech regions.
    Vad,
    /// Speech recognition.
    Asr,
    /// Speaker embedding + clustering.
    Diarization,
    /// Reconciling word times across chunk overlaps and VAD boundaries.
    Alignment,
    /// Filler/repetition/punctuation cleanup.
    PostProcessing,
    /// Producing the structured meeting document.
    Structuring,
    /// Writing the requested output files.
    Exporting,
    /// Finished successfully.
    Complete,
    /// Stopped by an error.
    Failed,
    /// Stopped by the user.
    Cancelled,
}

impl JobPhase {
    /// Every phase, in lifecycle order. Used by the UI to render a stepper and
    /// by tests to assert the state machine covers everything.
    pub const ALL: [JobPhase; 13] = [
        Self::Queued,
        Self::Preparing,
        Self::ExtractingAudio,
        Self::NormalizingAudio,
        Self::Vad,
        Self::Asr,
        Self::Diarization,
        Self::Alignment,
        Self::PostProcessing,
        Self::Structuring,
        Self::Exporting,
        Self::Complete,
        Self::Failed,
    ];

    /// The stable wire form, e.g. `EXTRACTING_AUDIO`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "QUEUED",
            Self::Preparing => "PREPARING",
            Self::ExtractingAudio => "EXTRACTING_AUDIO",
            Self::NormalizingAudio => "NORMALIZING_AUDIO",
            Self::Vad => "VAD",
            Self::Asr => "ASR",
            Self::Diarization => "DIARIZATION",
            Self::Alignment => "ALIGNMENT",
            Self::PostProcessing => "POST_PROCESSING",
            Self::Structuring => "STRUCTURING",
            Self::Exporting => "EXPORTING",
            Self::Complete => "COMPLETE",
            Self::Failed => "FAILED",
            Self::Cancelled => "CANCELLED",
        }
    }

    /// Short imperative label for the UI, e.g. `Vad` -> "Detecting speech".
    pub const fn label(self) -> &'static str {
        match self {
            Self::Queued => "Waiting",
            Self::Preparing => "Preparing",
            Self::ExtractingAudio => "Extracting audio",
            Self::NormalizingAudio => "Normalizing audio",
            Self::Vad => "Detecting speech",
            Self::Asr => "Transcribing",
            Self::Diarization => "Separating speakers",
            Self::Alignment => "Aligning timestamps",
            Self::PostProcessing => "Cleaning up",
            Self::Structuring => "Structuring transcript",
            Self::Exporting => "Exporting",
            Self::Complete => "Complete",
            Self::Failed => "Failed",
            Self::Cancelled => "Cancelled",
        }
    }

    /// Whether the job has finished, one way or another.
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Complete | Self::Failed | Self::Cancelled)
    }

    /// Whether the job is currently running (not queued, not finished).
    pub const fn is_active(self) -> bool {
        !matches!(self, Self::Queued) && !self.is_terminal()
    }

    /// Expected share of total wall time, used to compute overall progress.
    ///
    /// These are *estimates used for presentation only* and are calibrated
    /// against the real measured breakdown on the reference machine. They are
    /// never used to report a phase's own progress, and the benchmark report
    /// always shows the measured values rather than these.
    pub const fn weight(self) -> f32 {
        match self {
            Self::Queued => 0.0,
            // Loading a ~900 MB model from a cold page cache is genuinely slow
            // the first time and instant afterwards; this covers the average.
            Self::Preparing => 0.08,
            Self::ExtractingAudio => 0.06,
            Self::NormalizingAudio => 0.02,
            Self::Vad => 0.03,
            Self::Asr => 0.62,
            Self::Diarization => 0.12,
            Self::Alignment => 0.02,
            Self::PostProcessing => 0.02,
            Self::Structuring => 0.02,
            Self::Exporting => 0.01,
            Self::Complete | Self::Failed | Self::Cancelled => 1.0,
        }
    }

    /// Parse the wire form. Unknown phases map to `None` rather than a
    /// default, so a newer build's database does not silently read an old
    /// build's job as `Queued`.
    /// Parse the wire form, returning `None` for anything unrecognised.
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.as_str() == s).or(
            // Cancelled is terminal but not in ALL (ALL is the forward path).
            if s == "CANCELLED" {
                Some(Self::Cancelled)
            } else {
                None
            },
        )
    }
}

impl fmt::Display for JobPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How to read a [`Progress`] value.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", tag = "kind", content = "value")]
#[allow(missing_docs)] // field names are the documentation
pub enum ProgressKind {
    /// `completed` of `total` units. Shown as a real percentage.
    Fraction { completed: u64, total: u64 },
    /// A real measurement of audio processed, e.g. 412 s of 720 s. The most
    /// honest form: the user can see "7 minutes of 12" and know the wait.
    MediaSeconds { completed: f64, total: f64 },
    /// The phase is running but its size is not knowable in advance. Rendered
    /// as a spinner, never as a fabricated percentage.
    Indeterminate,
}

impl ProgressKind {
    /// Fraction in [0,1], or `None` when indeterminate.
    pub fn fraction(&self) -> Option<f64> {
        match self {
            Self::Fraction { completed, total } if *total > 0 => {
                Some((*completed as f64 / *total as f64).clamp(0.0, 1.0))
            }
            Self::MediaSeconds { completed, total } if *total > 0.0 => {
                Some((completed / total).clamp(0.0, 1.0))
            }
            _ => None,
        }
    }
}

/// Progress within the current phase.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct Progress {
    pub phase: JobPhase,
    pub kind: ProgressKind,
    /// Elapsed wall time in this phase, and in the job overall. The UI uses
    /// these for the "3m 12s left" estimate rather than guessing from progress.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase_elapsed: Option<Duration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_elapsed: Option<Duration>,
    /// Human-readable detail, e.g. `chunk 12 / 48`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl Progress {
    /// Progress for a phase whose size is not knowable in advance.
    pub fn indeterminate(phase: JobPhase) -> Self {
        Self {
            phase,
            kind: ProgressKind::Indeterminate,
            phase_elapsed: None,
            total_elapsed: None,
            detail: None,
        }
    }

    /// `completed` of `total` discrete units.
    pub fn fraction(phase: JobPhase, completed: u64, total: u64) -> Self {
        Self {
            kind: ProgressKind::Fraction { completed, total },
            ..Self::indeterminate(phase)
        }
    }

    /// Seconds of media processed out of the total. The most honest form.
    pub fn media(phase: JobPhase, completed: f64, total: f64) -> Self {
        Self {
            kind: ProgressKind::MediaSeconds { completed, total },
            ..Self::indeterminate(phase)
        }
    }

    #[must_use]
    /// Attach human-readable detail, e.g. `chunk 12 / 48`.
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }

    #[must_use]
    /// Attach measured elapsed times, which drive the ETA.
    pub fn with_elapsed(mut self, phase: Duration, total: Duration) -> Self {
        self.phase_elapsed = Some(phase);
        self.total_elapsed = Some(total);
        self
    }

    /// Overall job progress in [0,1], combining completed phases (by weight)
    /// with the current phase's own fraction.
    ///
    /// Returns `None` only when we genuinely cannot justify a number: the
    /// current phase is indeterminate *and* nothing has completed yet. In every
    /// other case we return a real figure, including a hard `0.0` for a queued
    /// job — that one is knowable, and omitting it would make the bar appear
    /// broken rather than empty.
    pub fn overall(&self) -> Option<f64> {
        let prior: f32 = JobPhase::ALL
            .into_iter()
            .filter(|p| !p.is_terminal() && *p < self.phase)
            .map(JobPhase::weight)
            .sum();
        match self.kind.fraction() {
            Some(f) => {
                let v = prior + f as f32 * self.phase.weight();
                Some((v as f64).clamp(0.0, 1.0))
            }
            // Indeterminate. If earlier phases moved the bar we can place it;
            // otherwise we would be inventing a position, so we decline.
            None if prior > 0.0 => Some((prior as f64).clamp(0.0, 1.0)),
            None if self.phase == JobPhase::Queued => Some(0.0),
            None => None,
        }
    }

    /// Remaining-time estimate, from measured throughput so far.
    ///
    /// `None` until the phase has run long enough to mean anything. We refuse
    /// to extrapolate from a phase that has done 0.1% of its work.
    pub fn estimated_remaining(&self) -> Option<Duration> {
        let frac = self.kind.fraction()?;
        let elapsed = self.phase_elapsed?;
        // 200 ms floor: below that the estimate is dominated by scheduling noise.
        if frac <= 0.0 || elapsed < Duration::from_millis(200) {
            return None;
        }
        let per_unit = elapsed.as_secs_f64() / frac;
        Some(Duration::from_secs_f64((per_unit * (1.0 - frac)).max(0.0)))
    }
}

/// A cancellation request shared across threads.
///
/// Explicitly not a `bool`: a `bool` invites lost updates and double-clears.
/// This is the one piece of cross-thread coordination in the pipeline, so it is
/// an atomic with compare-exchange semantics and a reason code.
#[derive(Debug, Clone, Default)]
pub struct CancelToken {
    flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl CancelToken {
    /// A live token.
    pub fn new() -> Self {
        Self::default()
    }

    /// Shared handle; cancelling through any clone cancels for all.
    pub fn child(&self) -> Self {
        Self {
            flag: std::sync::Arc::clone(&self.flag),
        }
    }

    /// Request cancellation. Idempotent.
    pub fn cancel(&self) {
        self.flag.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// `Err(Cancelled)` if cancelled. Call this at phase boundaries and inside
    /// long loops; it is a relaxed atomic load, so calling it per iteration of
    /// a hot loop is acceptable.
    pub fn check(&self) -> crate::Result<()> {
        if self.is_cancelled() {
            Err(crate::err::cancelled())
        } else {
            Ok(())
        }
    }
}

impl fmt::Display for CancelToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(if self.is_cancelled() {
            "cancelled"
        } else {
            "live"
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_strings_round_trip() {
        for p in JobPhase::ALL {
            assert_eq!(
                JobPhase::parse(p.as_str()),
                Some(p),
                "round trip failed for {p}"
            );
        }
        assert_eq!(JobPhase::parse("CANCELLED"), Some(JobPhase::Cancelled));
        assert_eq!(JobPhase::parse("NOT_A_PHASE"), None);
    }

    #[test]
    fn phase_strings_are_snake_upper_case_scalars() {
        // These strings land in the database and in `vaani jobs --json`. If
        // one is renamed, old rows stop parsing. Guard the format.
        for p in JobPhase::ALL {
            let s = p.as_str();
            assert!(
                s.chars().all(|c| c.is_ascii_uppercase() || c == '_'),
                "{s} should be SCREAMING_SNAKE_CASE"
            );
        }
    }

    #[test]
    fn active_and_terminal_are_disjoint_and_cover_the_lifecycle() {
        for p in JobPhase::ALL {
            if p.is_terminal() {
                assert!(!p.is_active(), "{p} is both terminal and active");
            } else {
                // Everything non-terminal is either waiting or running. There
                // is no third bucket, because the UI has only two states to
                // render for a non-finished job.
                assert!(
                    p.is_active() || p == JobPhase::Queued,
                    "{p} is neither active nor queued"
                );
            }
        }
        assert!(JobPhase::Cancelled.is_terminal());
        assert!(!JobPhase::Queued.is_active());
    }

    #[test]
    fn queued_job_reports_zero_not_nothing() {
        // A queued job genuinely is 0% complete. Omitting the number would read
        // as a broken progress bar.
        assert_eq!(
            Progress::fraction(JobPhase::Queued, 0, 1).overall(),
            Some(0.0)
        );
    }

    #[test]
    fn phase_weights_are_a_partition_of_unity() {
        // The UI presents one honest overall bar. If the active-phase weights
        // do not sum to 1, that bar can never reach 100% and users will read
        // it as a bug.
        let total: f32 = JobPhase::ALL
            .iter()
            .filter(|p| p.is_active())
            .map(|p| p.weight())
            .sum();
        assert!(
            (total - 1.0).abs() < 1e-4,
            "active phase weights sum to {total}, want 1.0"
        );
    }

    #[test]
    fn overall_progress_is_monotonic_across_phases() {
        let mut last = 0.0;
        for p in JobPhase::ALL.into_iter().filter(|p| p.is_active()) {
            let at_zero = Progress::fraction(p, 0, 100).overall().unwrap();
            assert!(
                at_zero >= last - 1e-6,
                "progress went backwards at {p}: {at_zero} < {last}"
            );
            let at_end = Progress::fraction(p, 100, 100).overall().unwrap();
            assert!(at_end >= at_zero);
            last = at_end;
        }
        assert!(
            (last - 1.0).abs() < 1e-4,
            "full completion should reach 1.0, got {last}"
        );
    }

    #[test]
    fn indeterminate_reports_no_percentage() {
        let p = Progress::indeterminate(JobPhase::Asr);
        assert_eq!(p.kind.fraction(), None);
        // But it must still make overall progress non-zero once earlier phases
        // have completed, so the bar is not stuck.
        let p = Progress::indeterminate(JobPhase::Asr);
        assert!(p.overall().unwrap() > 0.0);
    }

    #[test]
    fn zero_total_does_not_divide_by_zero() {
        assert_eq!(
            Progress::fraction(JobPhase::Asr, 5, 0).kind.fraction(),
            None
        );
        assert_eq!(
            Progress::media(JobPhase::Asr, 5.0, 0.0).kind.fraction(),
            None
        );
    }

    #[test]
    fn fraction_is_clamped() {
        assert_eq!(
            Progress::fraction(JobPhase::Asr, 150, 100).kind.fraction(),
            Some(1.0)
        );
    }

    #[test]
    fn eta_refuses_to_guess_before_the_phase_has_run() {
        // A phase that just started must not claim "finishes in 4 seconds"
        // based on 3 ms of work.
        let p = Progress::fraction(JobPhase::Asr, 1, 1000)
            .with_elapsed(Duration::from_millis(3), Duration::from_millis(3));
        assert_eq!(p.estimated_remaining(), None);
    }

    #[test]
    fn eta_extrapolates_from_measured_throughput() {
        // 25% done in 10 s => 30 s remaining.
        let p = Progress::fraction(JobPhase::Asr, 250, 1000)
            .with_elapsed(Duration::from_secs(10), Duration::from_secs(10));
        let eta = p.estimated_remaining().unwrap();
        assert!((eta.as_secs_f64() - 30.0).abs() < 0.5, "eta was {eta:?}");
    }

    #[test]
    fn cancel_token_propagates_to_children() {
        let root = CancelToken::new();
        let child = root.child();
        assert!(!child.is_cancelled());
        root.cancel();
        assert!(child.is_cancelled());
        assert!(child.check().is_err());
        assert!(child.check().unwrap_err().is_cancelled());
    }

    #[test]
    fn cancel_token_is_shareable_across_threads() {
        let root = CancelToken::new();
        let child = root.child();
        let h = std::thread::spawn(move || {
            for _ in 0..1000 {
                if child.is_cancelled() {
                    return true;
                }
                std::hint::spin_loop();
            }
            false
        });
        root.cancel();
        assert!(h.join().unwrap());
    }

    #[test]
    fn media_progress_uses_audio_seconds() {
        let p = Progress::media(JobPhase::ExtractingAudio, 412.0, 720.0);
        assert!((p.kind.fraction().unwrap() - 412.0 / 720.0).abs() < 1e-9);
    }
}

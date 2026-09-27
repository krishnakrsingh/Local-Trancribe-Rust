//! Progress reporting for transfers, and how it maps onto `vaani_core`.
//!
//! # Why not `vaani_core::Progress` directly
//!
//! Because a model install is not a job phase. It happens inside
//! [`JobPhase::Preparing`] when the pipeline needs a model, and it happens in
//! the settings panel when there is no job at all. A model download is minutes
//! long and 2.5 GB wide, and the UI needs **bytes** to draw an ETA as well as a
//! fraction. So this module has its own small type, with
//! [`TransferProgress::as_core`] to hand it to anything that wants the job-level
//! contract.
//!
//! # The callback, not a channel
//!
//! A `&dyn Fn` is enough for both consumers: a Slint UI wraps it in a
//! `invoke_from_event_loop`, and the CLI prints a bar. A channel would force
//! every caller to own a thread and a drain loop for no gain, and it would make
//! "did the progress actually arrive" a question rather than a fact.

use serde::{Deserialize, Serialize};
use vaani_core::{JobPhase, Progress};

/// Which half of the manager is reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[allow(missing_docs)] // each variant maps to a label
pub enum Stage {
    /// Fetching bytes over the network.
    Downloading,
    /// Hashing bytes already on disk.
    Verifying,
    /// Moving a verified temporary file into place.
    Installing,
    /// Deleting a model.
    Removing,
}

impl Stage {
    /// A short imperative label for the UI, e.g. `Downloading` -> "Downloading".
    pub const fn label(self) -> &'static str {
        match self {
            Self::Downloading => "Downloading",
            Self::Verifying => "Checking",
            Self::Installing => "Installing",
            Self::Removing => "Removing",
        }
    }

    /// The stable wire form.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Downloading => "downloading",
            Self::Verifying => "verifying",
            Self::Installing => "installing",
            Self::Removing => "removing",
        }
    }
}

/// One progress tick.
///
/// Every field is a real measurement. `bytes_total` is zero only when the size
/// genuinely is not knowable yet, and in that case the fraction is `None` and
/// the UI is expected to show a spinner rather than invent a percentage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct TransferProgress {
    /// Which half of the manager is reporting.
    pub stage: Stage,
    /// The model being worked on.
    pub model_id: String,
    /// The file being worked on, when there is one.
    pub file: Option<String>,
    /// Files finished so far.
    pub files_done: usize,
    /// Files in the operation.
    pub files_total: usize,
    /// Bytes moved or hashed.
    pub bytes_done: u64,
    /// Bytes in the whole operation, or 0 when unknown.
    pub bytes_total: u64,
}

impl TransferProgress {
    /// A tick with no file in progress yet.
    #[must_use]
    pub fn new(stage: Stage, model_id: impl Into<String>, bytes_total: u64) -> Self {
        Self {
            stage,
            model_id: model_id.into(),
            file: None,
            files_done: 0,
            files_total: 0,
            bytes_done: 0,
            bytes_total,
        }
    }

    /// The byte fraction in `[0, 1]`, or `None` when the total is unknown.
    #[must_use]
    pub fn fraction(&self) -> Option<f64> {
        (self.bytes_total > 0)
            .then(|| (self.bytes_done as f64 / self.bytes_total as f64).clamp(0.0, 1.0))
    }

    /// A one-line description for a log or a CLI, e.g.
    /// `Downloading encoder.fp16.onnx, 412.0 of 845.4 MiB (49%)`.
    #[must_use]
    pub fn describe(&self) -> String {
        let file = self.file.as_deref().unwrap_or("(no file)");
        match self.fraction() {
            Some(f) => format!(
                "{} {file}, {:.1} of {:.1} MiB ({:.0}%)",
                self.stage.label(),
                mib(self.bytes_done),
                mib(self.bytes_total),
                f * 100.0
            ),
            None => format!(
                "{} {file}, {:.1} MiB",
                self.stage.label(),
                mib(self.bytes_done)
            ),
        }
    }

    /// Project onto the job-level progress contract.
    ///
    /// A model install happens during [`JobPhase::Preparing`], so that is the
    /// phase reported. Callers that want a different phase pass it in; the
    /// fraction is clamped, and a zero total yields
    /// [`ProgressKind::Indeterminate`] rather than a division by zero.
    #[must_use]
    pub fn as_core(&self, phase: JobPhase) -> Progress {
        match self.fraction() {
            Some(_) => Progress::fraction(phase, self.bytes_done, self.bytes_total.max(1))
                .with_detail(self.describe()),
            None => Progress::indeterminate(phase).with_detail(self.describe()),
        }
    }
}

fn mib(b: u64) -> f64 {
    b as f64 / (1024.0 * 1024.0)
}

/// A progress sink. `&dyn Fn` covers a UI callback and a CLI printer without
/// either of them owning a thread.
pub type ProgressFn<'a> = &'a (dyn Fn(TransferProgress) + Send + Sync);

/// A sink that discards everything, for callers that do not care.
pub fn ignore(_: TransferProgress) {}

/// Collects every tick, for tests and for a CLI that wants the history.
///
/// A test that asserts "progress was reported" is otherwise asserting on a mock;
/// this makes the real callback observable.
#[derive(Debug, Clone, Default)]
pub struct ProgressLog {
    ticks: std::sync::Arc<std::sync::Mutex<Vec<TransferProgress>>>,
}

impl ProgressLog {
    /// A new, empty log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A `&dyn Fn` that records into this log. Borrow it for as long as the log
    /// is used.
    pub fn sink(&self) -> impl Fn(TransferProgress) + Send + Sync + 'static {
        let ticks = std::sync::Arc::clone(&self.ticks);
        move |p| {
            if let Ok(mut v) = ticks.lock() {
                v.push(p);
            }
        }
    }

    /// A snapshot of every tick so far.
    #[must_use]
    pub fn ticks(&self) -> Vec<TransferProgress> {
        self.ticks.lock().map(|v| v.clone()).unwrap_or_default()
    }

    /// How many ticks arrived.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ticks.lock().map(|v| v.len()).unwrap_or(0)
    }

    /// Whether anything arrived.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the reported byte count ever reached the total.
    #[must_use]
    pub fn reached_total(&self) -> bool {
        self.ticks()
            .iter()
            .any(|t| t.bytes_total > 0 && t.bytes_done >= t.bytes_total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use vaani_core::ProgressKind;

    fn p() -> TransferProgress {
        TransferProgress {
            stage: Stage::Downloading,
            model_id: "m".into(),
            file: Some("encoder.fp16.onnx".into()),
            files_done: 1,
            files_total: 2,
            bytes_done: 500,
            bytes_total: 1000,
        }
    }

    #[test]
    fn a_fraction_is_real_or_absent_never_invented() {
        assert_eq!(p().fraction(), Some(0.5));
        let mut unknown = p();
        unknown.bytes_total = 0;
        assert_eq!(unknown.fraction(), None);
        // Over-reporting is clamped rather than showing 140%.
        let mut over = p();
        over.bytes_done = 1400;
        assert_eq!(over.fraction(), Some(1.0));
    }

    #[test]
    fn a_zero_total_projects_to_indeterminate_not_a_division_by_zero() {
        let mut unknown = p();
        unknown.bytes_total = 0;
        let core = unknown.as_core(JobPhase::Preparing);
        assert_eq!(core.kind, ProgressKind::Indeterminate);
        assert_eq!(core.kind.fraction(), None);
        // The detail is still there, so a spinner has something to say.
        assert!(core.detail.unwrap().contains("encoder.fp16.onnx"));
    }

    #[test]
    fn the_projection_uses_the_phase_the_caller_names() {
        let core = p().as_core(JobPhase::Preparing);
        assert_eq!(core.phase, JobPhase::Preparing);
        assert_eq!(core.kind.fraction(), Some(0.5));
        assert!(core.overall().is_some());
    }

    #[test]
    fn descriptions_carry_bytes_and_a_percentage() {
        let d = p().describe();
        assert!(d.contains("Downloading"));
        assert!(d.contains("encoder.fp16.onnx"));
        assert!(d.contains("50%"));
    }

    #[test]
    fn stages_have_stable_names_and_labels() {
        for s in [
            Stage::Downloading,
            Stage::Verifying,
            Stage::Installing,
            Stage::Removing,
        ] {
            assert!(!s.as_str().is_empty());
            assert!(!s.label().is_empty());
        }
    }

    #[test]
    fn a_progress_log_records_what_really_arrived() {
        let log = ProgressLog::new();
        let sink = log.sink();
        let mut t = p();
        sink(t.clone());
        t.bytes_done = 1000;
        sink(t);
        assert_eq!(log.len(), 2);
        assert!(log.reached_total());
        assert!(!log.is_empty());
    }

    #[test]
    fn elapsed_can_be_carried_for_an_eta() {
        // `as_core` deliberately does not invent elapsed times; the caller knows
        // them and a fabricated one would make `estimated_remaining` a lie.
        let core = p()
            .as_core(JobPhase::Preparing)
            .with_elapsed(Duration::from_secs(1), Duration::from_secs(2));
        assert!(core.estimated_remaining().is_some());
    }
}

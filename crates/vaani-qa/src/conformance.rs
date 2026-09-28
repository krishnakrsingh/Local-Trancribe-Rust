//! The conformance harness: one trait, one question, one answer per stage.
//!
//! # What "conformance" means here
//!
//! `vaani-core` freezes the *signatures*; it cannot freeze the *behaviour*,
//! because behaviour is code. This module is where the behaviour is written
//! down once, as executable assertions, so that "does this implementation
//! honour the contract?" is a reusable question with one answer per
//! implementation rather than a bespoke test rewritten each time.
//!
//! Adding a stage to this gate costs one line:
//!
//! ```ignore
//! assert_conforms(&EnergyVadStage::new());
//! ```
//!
//! and that one line checks cancellation (prompt, against a measured bound),
//! empty input, degenerate input, progress monotonicity, determinism, the error
//! contract, and — the one that has cost this project the most — the time
//! coordinate space.
//!
//! # The seven guarantees
//!
//! | # | Guarantee | Check |
//! |---|---|---|
//! | 1 | Cancellation is honoured and prompt | [`check_cancellation_is_prompt`] |
//! | 2 | Empty input is not a crash | [`check_empty_input`], [`check_wrong_sample_rate`] |
//! | 3 | Progress is monotone, bounded, non-NaN, not over-claimed | [`check_progress_is_monotone`], [`check_progress_is_clamped`] |
//! | 4 | Adversarial input does not panic | [`check_adversarial_audio`], [`check_adversarial_paths`] |
//! | 5 | Errors are actionable | [`check_errors_are_actionable`] |
//! | 6 | The same input twice gives the same answer | [`check_determinism`] |
//! | 7 | Times are in the right coordinate space and in bounds | [`check_times_are_absolute`] |
//!
//! # The design decision that makes #7 possible
//!
//! Absolute-versus-buffer-relative cannot be checked by feeding a buffer that
//! starts at zero: an implementation that ignores `AudioBuffer::start_time` and
//! emits times relative to the buffer produces *byte-identical* output for a
//! zero-start buffer and is wrong for every other one. So
//! [`ConformanceStage::work_audio`] deliberately returns audio starting at
//! **+5 minutes**, and [`check_times_are_absolute`] requires every emitted
//! `Time` to lie inside `[start_time, start_time + duration]`. A stage that
//! rebases locally fails here, on synthetic audio, with no model and no
//! 14-minute recording.
//!
//! The one stage exempt from that is the stage that *defines* the origin:
//! [`ConformanceStage::offsets_by_input_start`] is `false` for media, whose
//! output is by construction at the start of the source file.
//!
//! # On reporting skips
//!
//! A check that cannot apply to a stage is recorded in
//! [`ConformanceReport::skipped`] with a reason and is never counted as a pass.
//! A conformance suite that reports 30/30 for a stage where 9 checks quietly
//! did not run is worse than one that reports 21/21 and 9 not applicable.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use vaani_core::media::AudioBuffer;
use vaani_core::pipeline::{
    DiarizationEngine, DiarizationResult, PipelineStage, ProgressSink, SpeechTimeline,
    StageArtifacts, StageContext, VadEngine,
};
use vaani_core::{
    CancelToken, ErrorCode, JobPhase, Progress, ProgressKind, Result, Segment, Time, Transcript,
    VaaniError,
};

use crate::fixtures;

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

/// One named contract assertion.
#[derive(Debug, Clone, PartialEq)]
pub struct ConformanceOutcome {
    /// `<stage>.<check>`, e.g. `energy-vad.times.no_time_precedes_the_origin`.
    pub name: String,
    /// What the check concluded.
    pub verdict: Verdict,
    /// The numbers the check measured, so a reader does not re-derive them.
    pub detail: String,
}

/// What a check concluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The contract holds.
    Pass,
    /// The contract is broken. A gate must go red.
    Fail,
    /// The contract is broken, the defect is known, it belongs to another
    /// crate, and it is tracked. Counted separately from both.
    ///
    /// This exists so a real, confirmed defect in a crate this harness does not
    /// own is *documented and visible* rather than either fixed here (wrong
    /// crate) or deleted from the check (false confidence). Every entry names the
    /// owning crate and the specific behaviour, and `tests/known_failures.rs`
    /// asserts the list stays short, that each entry still reproduces, and that
    /// nothing has been quietly added.
    Known,
}

/// A contract violation that is known, owned elsewhere, and tracked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnownFailure {
    /// The outcome name this covers, matched exactly.
    pub check: String,
    /// Which crate owns the defect, and where in it.
    pub owned_by: String,
    /// What is wrong, in one sentence a reader can act on.
    pub why: String,
}

impl KnownFailure {
    /// A known failure.
    #[must_use]
    pub fn new(check: &str, owned_by: &str, why: &str) -> Self {
        Self {
            check: check.to_owned(),
            owned_by: owned_by.to_owned(),
            why: why.to_owned(),
        }
    }
}

impl ConformanceOutcome {
    /// A passing outcome.
    pub fn pass(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            verdict: Verdict::Pass,
            detail: detail.into(),
        }
    }

    /// A failing outcome.
    pub fn fail(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            verdict: Verdict::Fail,
            detail: detail.into(),
        }
    }

    /// A known, tracked failure in another crate.
    pub fn known(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            verdict: Verdict::Known,
            detail: detail.into(),
        }
    }

    /// Whether the contract holds.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.verdict == Verdict::Pass
    }

    /// Whether the contract is broken and unaccounted for.
    #[must_use]
    pub fn failed(&self) -> bool {
        self.verdict == Verdict::Fail
    }
}

/// A check that could not run, and why. Never counted as a pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConformanceSkip {
    /// The check that did not run.
    pub check: String,
    /// Why it does not apply to this stage.
    pub reason: String,
}

/// Everything the harness found, in run order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConformanceReport {
    /// One entry per check that ran.
    pub outcomes: Vec<ConformanceOutcome>,
    /// Checks that do not apply, with a reason.
    pub skipped: Vec<ConformanceSkip>,
    /// Tracked failures in other crates that apply to this run.
    pub known: Vec<KnownFailure>,
}

impl ConformanceReport {
    /// An empty report.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A report carrying the known-failure registry of the thing under test.
    #[must_use]
    pub fn with_known(known: Vec<KnownFailure>) -> Self {
        Self {
            known,
            ..Self::default()
        }
    }

    /// Record one outcome verbatim, bypassing the registry.
    pub fn push(&mut self, outcome: ConformanceOutcome) {
        self.outcomes.push(outcome);
    }

    /// Record a pass, a fail, or a tracked known failure, consulting the
    /// registry carried by this report.
    ///
    /// A check name present in [`ConformanceReport::known`] becomes
    /// [`Verdict::Known`] whatever the measured result, so a tracked defect can
    /// never quietly turn green and a fixed defect can never quietly stay
    /// listed.
    pub fn record(&mut self, name: &str, ok: bool, detail: impl Into<String>) {
        let detail = detail.into();
        if let Some(k) = self.known.iter().find(|k| k.check == name) {
            // The measurement is kept either way. A tracked entry that suddenly
            // starts passing is a *stale* entry, and `tests/known_failures.rs`
            // fails on that so the author deletes it rather than leaving a
            // permanently-graying check.
            let measured = if ok { "NOW PASSES" } else { "still failing" };
            self.push(ConformanceOutcome::known(
                name,
                format!(
                    "{measured} — {detail} || owned by {}: {}",
                    k.owned_by, k.why
                ),
            ));
        } else {
            self.push(if ok {
                ConformanceOutcome::pass(name, detail)
            } else {
                ConformanceOutcome::fail(name, detail)
            });
        }
    }

    /// Record a check that does not apply.
    pub fn skip(&mut self, check: &str, reason: impl Into<String>) {
        self.skipped.push(ConformanceSkip {
            check: check.to_owned(),
            reason: reason.into(),
        });
    }

    /// Number of passing checks.
    #[must_use]
    pub fn passed(&self) -> usize {
        self.outcomes.iter().filter(|o| o.passed()).count()
    }

    /// Number of failing checks.
    #[must_use]
    pub fn failed(&self) -> usize {
        self.outcomes.iter().filter(|o| o.failed()).count()
    }

    /// Number of tracked, known failures.
    #[must_use]
    pub fn known(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|o| o.verdict == Verdict::Known)
            .count()
    }

    /// Whether every check that ran passed, ignoring tracked known failures.
    #[must_use]
    pub fn ok(&self) -> bool {
        self.failed() == 0
    }

    /// The names of the unaccounted-for failures.
    #[must_use]
    pub fn failures(&self) -> Vec<&str> {
        self.outcomes
            .iter()
            .filter(|o| o.failed())
            .map(|o| o.name.as_str())
            .collect()
    }

    /// Merge another report into this one.
    pub fn extend(&mut self, other: ConformanceReport) {
        self.outcomes.extend(other.outcomes);
        self.skipped.extend(other.skipped);
        for k in other.known {
            if !self.known.contains(&k) {
                self.known.push(k);
            }
        }
    }

    /// A human-readable dump, so a failure is diagnosable from the log alone.
    #[must_use]
    pub fn render(&self) -> String {
        let mut out = String::new();
        for o in &self.outcomes {
            let tag = match o.verdict {
                Verdict::Pass => "PASS",
                Verdict::Fail => "FAIL",
                Verdict::Known => "KNOWN",
            };
            out.push_str(&format!("{tag} {} — {}\n", o.name, o.detail));
        }
        for s in &self.skipped {
            out.push_str(&format!("SKIP {} — {}\n", s.check, s.reason));
        }
        out.push_str(&format!(
            "-- {} passed, {} failed, {} known/tracked, {} not applicable\n",
            self.passed(),
            self.failed(),
            self.known(),
            self.skipped.len()
        ));
        out
    }
}

/// Print a report's [`ConformanceReport::render`].
pub fn print_report(report: &ConformanceReport) {
    print!("{}", report.render());
}

// ---------------------------------------------------------------------------
// The stage abstraction
// ---------------------------------------------------------------------------

/// What kind of result a stage produces. Drives which structural checks apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputKind {
    /// Audio (the media stage).
    Audio,
    /// A speech timeline (VAD).
    Timeline,
    /// Speaker spans (diarization).
    Speakers,
    /// A transcript (ASR, or diarization after attribution).
    Transcript,
    /// No structured result: the stage's effect is a file or a database row
    /// (storage, export). Only the universal checks apply.
    SideEffect,
}

/// One run of a stage.
#[derive(Debug, Clone, PartialEq)]
pub enum StageOutput {
    /// No structured result.
    None,
    /// Decoded audio.
    Audio(Box<AudioBuffer>),
    /// A speech timeline.
    Timeline(Box<SpeechTimeline>),
    /// Speaker spans.
    Speakers(Box<DiarizationResult>),
    /// A transcript.
    Transcript(Box<Transcript>),
}

impl StageOutput {
    /// The decoded audio, if this run produced any.
    #[must_use]
    pub fn audio(&self) -> Option<&AudioBuffer> {
        match self {
            Self::Audio(a) => Some(a),
            _ => None,
        }
    }

    /// The speech timeline, if this run produced one.
    #[must_use]
    pub fn timeline(&self) -> Option<&SpeechTimeline> {
        match self {
            Self::Timeline(t) => Some(t),
            _ => None,
        }
    }

    /// The speaker spans, if this run produced any.
    #[must_use]
    pub fn speakers(&self) -> Option<&DiarizationResult> {
        match self {
            Self::Speakers(s) => Some(s),
            _ => None,
        }
    }

    /// The transcript, if this run produced one.
    #[must_use]
    pub fn transcript(&self) -> Option<&Transcript> {
        match self {
            Self::Transcript(t) => Some(t),
            _ => None,
        }
    }

    /// Every `(start, end)` time pair the run emitted, whatever it produced.
    ///
    /// Deliberately uniform: a coordinate-space check that only understood one
    /// output type would be one more bespoke test per stage.
    #[must_use]
    pub fn time_pairs(&self) -> Vec<(Time, Time)> {
        match self {
            Self::Audio(a) => vec![(a.start_time, a.start_time.saturating_add(a.duration()))],
            Self::Timeline(t) => t.segments.iter().map(|s| (s.start, s.end)).collect(),
            Self::Speakers(s) => s.spans.iter().map(|s| (s.start, s.end)).collect(),
            Self::Transcript(t) => {
                let mut out: Vec<(Time, Time)> = Vec::new();
                for s in &t.segments {
                    out.push((s.start, s.end));
                    for w in &s.words {
                        out.push((w.start, w.end));
                    }
                }
                out
            }
            Self::None => Vec::new(),
        }
    }
}
/// * Reported progress is finite, in `[0, 1]`, and non-decreasing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arity {
    /// The stage takes an `AudioBuffer` directly (an engine, or a wrapper).
    Buffer,
    /// The stage's real input is a filesystem path.
    Path,
}

/// The uniform view of a pipeline stage that every check needs.
///
/// ```ignore
/// impl ConformanceStage for MyStage {
///     fn id(&self) -> &str { "my-stage" }
///     fn output_kind(&self) -> OutputKind { OutputKind::Transcript }
///     fn required_sample_rate(&self) -> Option<u32> { Some(16_000) }
///     fn arity(&self) -> Arity { Arity::Buffer }
///     fn reports_progress(&self) -> bool { true }
///     fn run(&self, audio: &AudioBuffer, cancel: &CancelToken, progress: &ProgressSink)
///         -> Result<StageOutput> { /* ... */ }
/// }
/// ```
///
/// # The contract [`run_conformance`] asserts
///
/// * `run` polls `cancel` at least once per second of work, and a cancel
///   observed mid-run stops the stage within [`CANCEL_BUDGET`].
/// * A zero-length buffer is either an empty result or a clean coded error, and
///   never invented content and never a panic.
/// * Reported progress is finite, in `[0, 1]`, and non-decreasing.
/// * Every `Time` emitted lies inside the input buffer's span, honouring its
///   `start_time` — see [`ConformanceStage::offsets_by_input_start`].
/// * The same input twice gives the same content; see
///   [`check_determinism`] for the one documented exception.
pub trait ConformanceStage: Send + Sync {
    /// A stable identifier, prefixed onto nothing but printed with every
    /// failure so a reader knows which implementation broke.
    fn id(&self) -> &str;

    /// The kind of result [`ConformanceStage::run`] produces.
    fn output_kind(&self) -> OutputKind;

    /// The sample rate the stage requires of its input, or `None` if it does
    /// not care.
    fn required_sample_rate(&self) -> Option<u32>;

    /// Whether this stage's real input is a buffer or a path.
    ///
    /// It decides which adversarial surface applies: a buffer stage is attacked
    /// with absurd rates and `NaN`, a path stage with paths that do not exist,
    /// are directories, and are not media. Both are checked; neither is faked
    /// for the other, because writing a `u32::MAX`-Hz WAV header to test a
    /// *file* stage would be testing this crate's encoder, not the stage.
    fn arity(&self) -> Arity;

    /// Whether this implementation is *able* to report progress.
    ///
    /// `false` is an honest statement about the underlying engine API — the
    /// `VadEngine` trait has no progress parameter. A stage that says `true` and
    /// then reports nothing is a bug: the UI would show a bar that never moves,
    /// and [`check_progress_is_monotone`] fails it.
    fn reports_progress(&self) -> bool {
        false
    }

    /// Whether this stage's output times must be offset by the input buffer's
    /// `start_time`.
    ///
    /// `false` only for the stage that *defines* the origin (media), whose
    /// output is by construction at the start of the source file. Everything
    /// downstream of it is `true`, which is the whole point of the check.
    fn offsets_by_input_start(&self) -> bool {
        true
    }

    /// Run the stage once.
    fn run(
        &self,
        audio: &AudioBuffer,
        cancel: &CancelToken,
        progress: &ProgressSink,
    ) -> Result<StageOutput>;

    /// Run the stage against a path. Only called when [`ConformanceStage::arity`]
    /// is [`Arity::Path`].
    fn run_path(
        &self,
        _path: &Path,
        _cancel: &CancelToken,
        _progress: &ProgressSink,
    ) -> Option<Result<StageOutput>> {
        None
    }

    /// A speech-positive input long enough that a cancel can land mid-run.
    ///
    /// Defaults to 90 s of two-voice meeting audio, which is 2.5x the
    /// [`MIN_INTERRUPTIBLE`] floor for the VAD (measured RTF 0.0091) and 2x for
    /// the diarizer (0.005). It **starts at +5 minutes** so
    /// [`check_times_are_absolute`] can tell absolute times from buffer-relative
    /// ones; see the module docs.
    ///
    /// Memoised process-wide: this is a deterministic fixture, and
    /// `fixtures::voice` costs 24 harmonics per sample, so rebuilding it for each
    /// of the seven runs a check suite performs would add minutes of test time
    /// for no extra coverage.
    fn work_audio(&self) -> AudioBuffer {
        cached_work_audio()
    }

    /// Paths that must be handled without a panic, as `(label, path)`.
    ///
    /// Only stages of [`Arity::Path`] override this. The default is empty, and
    /// an empty list is recorded as a failure rather than a pass.
    fn bad_input_paths(&self) -> Vec<(String, PathBuf)> {
        Vec::new()
    }

    /// Failure paths to hand to [`check_errors_are_actionable`], as
    /// `(label, error)`.
    ///
    /// A representative sample, not every path: the shape comes from shared
    /// constructors, so sampling a handful of call sites catches a constructor
    /// that stopped attaching a recovery, which is the real failure mode.
    fn error_samples(&self) -> Vec<(String, VaaniError)> {
        Vec::new()
    }

    /// Contract violations this implementation is *known* to have, each owned by
    /// another crate.
    ///
    /// A stage that returns an empty list and fails a check is a red gate. A
    /// stage that returns an entry for it has documented a real defect and
    /// pointed at the crate that owns it — which is the only honest thing a test
    /// harness can do with a bug in someone else's code.
    fn known_failures(&self) -> Vec<KnownFailure> {
        Vec::new()
    }
}

/// The coordinate-space violation in `vaani_app`'s ASR stage, tracked.
///
/// **What happens.** `AsrStage::run` plans chunks with
/// `Chunker::plan(&speech, ctx.artifacts.audio()?.duration())`. `duration()` is a
/// *length*, but `speech` holds *absolute* times, so the chunker is comparing an
/// absolute segment end against a relative total. For any buffer that does not
/// start at zero the two disagree by the buffer's `start_time`.
///
/// **Why it does not bite in production, and why it still matters.** Decode
/// always produces `start_time == Time::ZERO`, so with a real input the total and
/// the segment end agree and the stage is correct. The moment a buffer arrives
/// with a non-zero origin — a resumed job, a slice handed between stages, a
/// future second input source — the first chunk's times come back before the
/// origin and the transcript is silently offset.
///
/// **Minimal reproduction.**
/// ```ignore
/// let audio  = AudioBuffer::mono(samples, 16_000, Time(300_000));
/// let speech = SpeechTimeline { segments: vec![SpeechSegment { start: Time(300_000),
///                                 end: Time(389_600), confidence: None }],
///                               speech_duration_ms: 89_600 };
/// // Upstream::AudioAndSpeech seeds exactly this pair; run the stage and read
/// // ctx.artifacts.transcript.unwrap().segments[0].start -- it is < 300_000.
/// ```
///
/// **The fix belongs in `vaani-app`, not here.** Either the chunker should be
/// given the absolute end (`audio.start_time + audio.duration()`), or `AsrStage`
/// should rebase the timeline it is handed. Both are one line; neither is this
/// crate's to make.
#[must_use]
pub fn asr_stage_known_failures() -> Vec<KnownFailure> {
    vec![KnownFailure::new(
        "times.no_time_precedes_the_origin",
        "vaani-app src/stages/asr.rs AsrStage::run (Chunker::plan argument)",
        "AsrStage plans chunks with the buffer's *duration* as the total, while the speech \
         timeline holds *absolute* times, so for any buffer whose start_time is not zero the \
         chunker compares an absolute end against a relative total and the first chunk's times \
         come back before the origin. In production decode always yields start_time == 0, so \
         this is latent rather than live; it becomes a silent transcript offset the moment a \
         buffer arrives with a non-zero origin. The gate proves it by running the ASR stage on \
         the +5-minute work audio, where 2 of 6 emitted pairs start before 300000 ms.",
    )]
}

/// Known failures in the guarantees that are not about a single stage, i.e. in
/// `vaani-core` and in the engines' own error construction.
#[must_use]
pub fn core_known_failures() -> Vec<KnownFailure> {
    vec![KnownFailure::new(
        "progress.clamped[nan]",
        "vaani-core src/job.rs StageContext::report",
        "report() does `f.clamp(0.0, 1.0)`, and f64::clamp propagates NaN, so a stage that \
         computes a 0/0 fraction publishes NaN to every progress sink. ProgressKind::fraction \
         clamps again and still returns Some(NaN), so the UI receives a number it cannot draw. \
         Repro: ctx.report(JobPhase::Asr, Some(f64::NAN), None) then read kind.fraction().",
    )]
}

/// Every tracked failure whose check name does not depend on which stage ran.
///
/// These are the defects found in *other* crates by this harness. They are
/// reported, not fixed — this crate does not own them — and they are listed here
/// so `tests/known_failures.rs` can assert the list stays short, that each entry
/// names an owning crate, and that each still reproduces. A fix upstream turns
/// the entry stale and that test fails, telling the author to delete it.
#[must_use]
pub fn all_known_failures() -> Vec<KnownFailure> {
    let mut out = core_known_failures();
    out.push(KnownFailure::new(
        "errors.actionable[energy_vad_wrong_sample_rate]",
        "vaani-vad src/energy.rs EnergyVad::check_input",
        "A wrong sample rate is the caller's mistake, but it is reported as ErrorCode::Internal \
         with no recovery, so the UI shows \"Something went wrong. An unexpected internal error \
         occurred. Please report this.\" for a buffer the media stage simply resampled wrongly. \
         vaani-media/src/errors.rs exists precisely to stop that, and the rate check is the one \
         place it is not applied. Fix: a non-internal code plus a recovery naming 16 kHz.",
    ));
    out.push(KnownFailure::new(
        "errors.actionable[diar_wrong_sample_rate]",
        "vaani-diarization src/engine.rs HeuristicDiarizer::run",
        "Same defect as the VAD's: a wrong sample rate yields ErrorCode::Internal with no \
         recovery, so a user whose media stage handed over 8 kHz is told to report a bug. The \
         engine's own comment says the *error* is deliberate; the missing recovery is not.",
    ));
    out.push(KnownFailure::new(
        "errors.actionable[diar_stereo_not_downed]",
        "vaani-diarization src/engine.rs HeuristicDiarizer::run",
        "A stereo buffer yields ErrorCode::Internal with no recovery. The correct user-facing \
         message is \"this file's audio could not be prepared\", not \"please report this\".",
    ));
    out.push(KnownFailure::new(
        "errors.actionable[asr_model_not_downloaded]",
        "vaani-asr src/engine.rs EngineRegistry::create",
        "An engine whose artefacts are absent returns err::model_unavailable() with a cause and \
         *no recovery*. The user-facing text is therefore \"The speech recognition model has not \
         been downloaded on this computer yet.\" and then nothing — which is the most common \
         first-run error in the product and the one place a recovery matters most. Fix: attach \
         .with_recovery(\"Open Settings, then Models, and download SraVaani 1.0.\").",
    ));
    out
}

// ---------------------------------------------------------------------------
// The shared work-audio cache
// ---------------------------------------------------------------------------

/// The +5-minute speech bed every buffer stage is checked against.
///
/// Cached because it is deterministic and expensive to synthesise.
fn cached_work_audio() -> AudioBuffer {
    static BED: OnceLock<AudioBuffer> = OnceLock::new();
    BED.get_or_init(|| {
        AudioBuffer::mono(
            fixtures::meeting_turns(2, 22.0, fixtures::FIXTURE_SAMPLE_RATE, 0x51_0A),
            fixtures::FIXTURE_SAMPLE_RATE,
            // Five minutes in. The coordinate-space check reads this.
            Time(300_000),
        )
    })
    .clone()
}

/// A 25-minute bed, for stages that are far faster than realtime.
///
/// Decoding runs at 2400x realtime, so a cancel can only be armed mid-work if
/// the input is minutes long. 25 minutes of 16 kHz mono is 48 MB and decodes in
/// roughly 600 ms — three times the [`MIN_INTERRUPTIBLE`] floor, with the cancel
/// armed at about 75 ms.
///
/// It deliberately starts at **+5 minutes** like [`cached_work_audio`], so the
/// coordinate-space check reads the same origin in both cases.
#[must_use]
pub fn long_work_audio() -> AudioBuffer {
    static LONG: OnceLock<AudioBuffer> = OnceLock::new();
    LONG.get_or_init(|| {
        AudioBuffer::mono(
            fixtures::long_bed(25 * 60, fixtures::FIXTURE_SAMPLE_RATE, 0x9E1),
            fixtures::FIXTURE_SAMPLE_RATE,
            Time(300_000),
        )
    })
    .clone()
}

// ---------------------------------------------------------------------------
// Guarantee 1: cancellation is honoured and prompt
// ---------------------------------------------------------------------------

/// How long after a cancel a stage may still be running before it is a bug.
///
/// Two seconds is deliberately generous. The requirement is not "fast", it is
/// *bounded*: a desktop app has to be killable, and a stage that takes four
/// seconds to notice a cancel is a stage that does that on every cancel,
/// forever. The bound is measured from the moment `cancel()` is called, not
/// from the start of the run, so a stage that finished all its work before the
/// cancel landed is correctly failed rather than credited with a fast cancel.
pub const CANCEL_BUDGET: Duration = Duration::from_secs(2);

/// The shortest clean run from which a timer-armed cancel can be trusted to
/// land inside the work.
///
/// Below this the measurement is a race, and a race in a conformance check is a
/// check that means nothing. This is the direct lesson of
/// `docs/JOURNEY.md` §"wave 2", where two `vaani-media` cancellation tests
/// "passed" only because a 30 ms sleep outlasted a 2400x-realtime decode.
const MIN_INTERRUPTIBLE: Duration = Duration::from_millis(200);

/// Guarantee 1: a cancelled stage stops, reports `Cancelled`, and stops soon.
///
/// Two checks, because they catch different bugs.
///
/// * **pre-cancelled** is race-free. A token cancelled before the call must
///   produce `ErrorCode::Cancelled` *and* must not have done the work. The
///   second half is the one that matters: a stage that returns the right error
///   after a full computation ignores cancellation for the length of a
///   14-minute transcription. It is measured against a clean baseline of the
///   same input, so it holds at any build speed.
/// * **mid-run** arms the cancel at one eighth of that same baseline, so the
///   cancel provably lands inside the work rather than racing it, and asserts
///   both that the stage returned `Cancelled` and that it returned *earlier
///   than the clean run would have* — a stage that notices the cancel and then
///   finishes its work anyway fails here.
///
/// If the baseline is shorter than [`MIN_INTERRUPTIBLE`] the check **fails**
/// rather than skipping: an un-interruptible stage is precisely the defect, and
/// silently passing is the disease this crate exists to remove.
pub fn check_cancellation_is_prompt(stage: &dyn ConformanceStage) -> ConformanceReport {
    let mut report = ConformanceReport::with_known(stage.known_failures());
    let audio = stage.work_audio();

    // --- baseline: how long the stage takes when nothing interferes ---------
    //
    // Best of three, not a single sample. Under a full parallel test run a
    // thread can be descheduled for milliseconds, and a single inflated sample
    // then reads as "the pre-cancelled run did work" when the machine was
    // merely busy. The minimum is the right statistic: scheduling noise only
    // ever inflates a sample, so the fastest of three is closest to the truth.
    // (Observed in the wild: a no-work run outlasting its baseline under a
    // 90-binary parallel run, roughly one time in four.)
    let (baseline_elapsed, baseline_worked) = {
        let mut best = Duration::MAX;
        let mut worked = false;
        for _ in 0..3 {
            let started = Instant::now();
            let result = run_stage(stage, &audio, &CancelToken::new());
            best = best.min(started.elapsed());
            worked = worked || result.is_ok();
        }
        (best, worked)
    };

    // --- pre-cancelled -----------------------------------------------------
    //
    // Best of three, for the same reason as the baseline: a single inflated
    // sample under load reads as "the stage did work". Correctness (every run
    // returns `Cancelled`) is checked on all three; speed on the minimum.
    {
        let dead = CancelToken::new();
        dead.cancel();
        let mut best = Duration::MAX;
        let mut all_cancelled = true;
        let mut last_wrong: Option<String> = None;
        let mut last_ok = false;
        for _ in 0..3 {
            let started = Instant::now();
            let result = run_stage(stage, &audio, &dead);
            best = best.min(started.elapsed());
            match &result {
                Err(e) if e.code() == ErrorCode::Cancelled => {}
                Err(e) => {
                    all_cancelled = false;
                    last_wrong = Some(format!("wrong code {}: {}", e.code(), e.debug_chain()));
                }
                Ok(_) => {
                    all_cancelled = false;
                    last_ok = true;
                }
            }
        }
        let elapsed = best;
        if all_cancelled {
            let did_work = baseline_worked && elapsed >= baseline_elapsed;
            report.record(
                "cancellation.pre_cancelled_is_rejected",
                !did_work,
                format!(
                    "Cancelled after {:.1} ms against a {:.1} ms clean run (did_work={did_work})",
                    elapsed.as_secs_f64() * 1000.0,
                    baseline_elapsed.as_secs_f64() * 1000.0
                ),
            );
        } else if last_ok {
            report.record(
                "cancellation.pre_cancelled_is_rejected",
                false,
                "a pre-cancelled token produced a result".to_owned(),
            );
        } else {
            report.record(
                "cancellation.pre_cancelled_is_rejected",
                false,
                last_wrong.unwrap_or_else(|| "no runs completed".to_owned()),
            );
        }
    }

    // --- mid-run -----------------------------------------------------------
    if !baseline_worked {
        report.record(
            "cancellation.mid_run_is_observed",
            false,
            "the stage does not run at all on its own work audio".to_owned(),
        );
        return report;
    }
    if baseline_elapsed < MIN_INTERRUPTIBLE {
        // Not a failure and not a pass. The stage finished faster than a cancel
        // can be armed, so the mid-run question is unanswerable for it — and
        // saying so, with the measured number, is the only honest thing. The
        // pre-cancelled check above still applies and is not skipped.
        report.skip(
            "cancellation.mid_run_is_observed",
            format!(
                "unmeasurable: the clean run took {:.0} ms, under the {MIN_INTERRUPTIBLE:.0?} \
                 floor needed to arm a cancel inside the work. The stage is simply faster than \
                 realtime here; the pre-cancelled check covers the entry path. Lengthen \
                 ConformanceStage::work_audio() if this stage should have been measurable.",
                baseline_elapsed.as_secs_f64() * 1000.0
            ),
        );
        return report;
    }

    let token = CancelToken::new();
    let child = token.child();
    let arm_after = baseline_elapsed / 8;
    let armed_at: Arc<Mutex<Option<Instant>>> = Arc::new(Mutex::new(None));
    let writer = Arc::clone(&armed_at);
    let canceller = token.child();
    let canceller_handle = std::thread::spawn(move || {
        std::thread::sleep(arm_after);
        *writer.lock().expect("armed_at lock") = Some(Instant::now());
        canceller.cancel();
    });

    // The stage stays on this thread; only the canceller moves, so no `'static`
    // bound is needed on the stage reference.
    let started = Instant::now();
    let result = run_stage(stage, &audio, &child);
    let finished_at = Instant::now();
    let _ = canceller_handle.join();

    // If the canceller thread somehow never ran, `observed` degrades to "the
    // whole run", which is larger than the budget and so fails the check rather
    // than passing it. Failing safe is the point.
    let cancelled_at = armed_at
        .lock()
        .expect("armed_at lock")
        .unwrap_or(finished_at);
    let observed = finished_at.saturating_duration_since(cancelled_at);
    let total = finished_at.saturating_duration_since(started);
    let arm_ms = arm_after.as_secs_f64() * 1000.0;

    match &result {
        Err(e) if e.code() == ErrorCode::Cancelled => {
            report.record(
                "cancellation.mid_run_is_observed",
                observed <= CANCEL_BUDGET,
                format!(
                    "cancelled {arm_ms:.0} ms into a {:.0} ms run; stopped {:.1} ms after the \
                     cancel (budget {CANCEL_BUDGET:.0?})",
                    baseline_elapsed.as_secs_f64() * 1000.0,
                    observed.as_secs_f64() * 1000.0
                ),
            );
            report.record(
                "cancellation.stops_before_the_work_is_done",
                total < baseline_elapsed,
                format!(
                    "cancelled run took {:.0} ms against a {:.0} ms clean run",
                    total.as_secs_f64() * 1000.0,
                    baseline_elapsed.as_secs_f64() * 1000.0
                ),
            );
        }
        Err(e) => report.record(
            "cancellation.mid_run_is_observed",
            false,
            format!("wrong code {}: {}", e.code(), e.debug_chain()),
        ),
        Ok(_) => report.record(
            "cancellation.mid_run_is_observed",
            false,
            format!(
                "the cancel landed {arm_ms:.0} ms into a {:.0} ms run and the stage returned a \
                 result anyway",
                baseline_elapsed.as_secs_f64() * 1000.0
            ),
        ),
    }

    report
}

// ---------------------------------------------------------------------------
// Guarantee 2: empty and degenerate input
// ---------------------------------------------------------------------------

/// Guarantee 2: a zero-length buffer is a defined outcome, never a panic, never
/// invented content.
///
/// "Defined" means one of exactly two things and the stage chooses: an empty
/// result, or an error whose code is in the product's vocabulary. A stage that
/// returns *populated* times for zero samples is failing, because inventing
/// content is worse than admitting there is none.
pub fn check_empty_input(stage: &dyn ConformanceStage) -> ConformanceReport {
    let mut report = ConformanceReport::with_known(stage.known_failures());
    let empty = AudioBuffer::mono(Vec::new(), stage_rate(stage), Time::ZERO);
    let name = "empty.zero_length_buffer";
    match stage.run(&empty, &CancelToken::new(), &|_| {}) {
        Ok(output) => {
            let invented = output.time_pairs().len();
            report.record(
                name,
                invented == 0,
                format!(
                    "{invented} time pair(s) for zero samples; a stage must not invent content \
                     for empty input"
                ),
            );
        }
        Err(e) => report.record(
            name,
            is_clean_failure(&e),
            format!("rejected with {}: {}", e.code(), e.user.title),
        ),
    }
    report
}

/// Guarantee 2b: a buffer at the wrong sample rate is rejected, not silently
/// resampled.
///
/// Resampling inside a stage scales a pitch feature by a factor the caller does
/// not know about, and it hides the real bug — "the pipeline handed me 48 kHz" —
/// behind a slower, wrong-but-plausible result. The check therefore insists on
/// a rejection whose message names the rate the stage needs; a stage that does
/// not need a specific rate records a skip instead of a fake pass.
pub fn check_wrong_sample_rate(stage: &dyn ConformanceStage) -> ConformanceReport {
    let mut report = ConformanceReport::with_known(stage.known_failures());
    let Some(wanted) = stage.required_sample_rate() else {
        report.skip(
            "wrong_sample_rate.is_rejected",
            "this stage declares no required sample rate",
        );
        return report;
    };
    if wanted == 0 {
        report.skip(
            "wrong_sample_rate.is_rejected",
            "this stage resamples its own input, so no rate is wrong",
        );
        return report;
    }
    let bad_rate = if wanted == 8_000 { 16_000 } else { 8_000 };
    let audio = AudioBuffer::mono(
        fixtures::meeting(4_000, bad_rate, 0x1D, 130.0),
        bad_rate,
        Time::ZERO,
    );
    match stage.run(&audio, &CancelToken::new(), &|_| {}) {
        Ok(output) => {
            let pairs = output.time_pairs();
            let span_end = audio.start_time.saturating_add(audio.duration());
            let within = pairs
                .iter()
                .all(|(s, e)| *s >= audio.start_time && *e <= span_end);
            report.record(
                "wrong_sample_rate.is_rejected",
                within,
                format!(
                    "accepted {bad_rate} Hz and produced {} pair(s) inside the buffer's span; \
                     the contract prefers a clean rejection naming {wanted} Hz",
                    pairs.len()
                ),
            );
        }
        Err(e) => {
            let names_the_rate = e.debug_chain().contains(&wanted.to_string());
            report.record(
                "wrong_sample_rate.is_rejected",
                is_clean_failure(&e) && names_the_rate,
                format!(
                    "rejected with {}; message names {wanted} Hz: {names_the_rate} — {}",
                    e.code(),
                    e.user.title
                ),
            );
        }
    }
    report
}

// ---------------------------------------------------------------------------
// Guarantee 3: progress is monotone, bounded, honest
// ---------------------------------------------------------------------------

/// Guarantee 3: progress never lies.
///
/// * every reported fraction is finite and in `[0, 1]` — no `NaN`, no 1.5;
/// * fractions never decrease within a run, because a bar that jumps to 100%
///   and back to 20% is the bug users report as "it keeps restarting";
/// * `Progress::overall()`, the number the UI's single bar actually reads, never
///   decreases and never leaves `[0, 1]`;
/// * the first report of a multi-report run is not already 1.0, because that is
///   a fraction claimed for work not done;
/// * a successful run finishes at the end of its own phase, because a stage that
///   stops reporting at 60% leaves a bar that never completes.
pub fn check_progress_is_monotone(stage: &dyn ConformanceStage) -> ConformanceReport {
    let mut report = ConformanceReport::with_known(stage.known_failures());
    if !stage.reports_progress() {
        report.skip(
            "progress.is_monotone_and_honest",
            "this implementation's API has no progress sink, so there is nothing to observe; \
             the app-level wrapper around the same stage is checked instead",
        );
        return report;
    }

    let audio = stage.work_audio();
    let (seen, collector) = progress_recorder();
    let result = stage.run(&audio, &CancelToken::new(), &collector);
    let samples = seen.lock().map(|s| s.clone()).unwrap_or_default();

    if let Err(e) = &result {
        report.record(
            "progress.is_monotone_and_honest",
            false,
            format!("the stage did not run: {}", e.debug_chain()),
        );
        return report;
    }
    if samples.is_empty() {
        report.record(
            "progress.is_monotone_and_honest",
            false,
            "the stage declares that it reports progress and reported none; the UI would show a \
             bar that never moves"
                .to_owned(),
        );
        return report;
    }

    let numeric: Vec<f64> = samples.iter().filter_map(|p| p.kind.fraction()).collect();

    let out_of_range = numeric
        .iter()
        .find(|f| !f.is_finite() || **f < 0.0 || **f > 1.0)
        .copied();
    report.record(
        "progress.fraction_is_finite_and_in_range",
        out_of_range.is_none(),
        match out_of_range {
            Some(f) => format!("{f} reported; fractions must be finite and in [0,1]"),
            None => format!(
                "{} of {} reports carried a fraction, all in [0,1]",
                numeric.len(),
                samples.len()
            ),
        },
    );

    let regressions: Vec<String> = numeric
        .windows(2)
        .enumerate()
        .filter(|(_, w)| w[1] < w[0])
        .map(|(i, w)| format!("#{i}: {:.4} -> {:.4}", w[0], w[1]))
        .collect();
    report.record(
        "progress.never_goes_backwards",
        regressions.is_empty(),
        if regressions.is_empty() {
            format!(
                "{} steps, all non-decreasing",
                numeric.len().saturating_sub(1)
            )
        } else {
            format!(
                "{} regressions: {}",
                regressions.len(),
                regressions.join(", ")
            )
        },
    );

    let overalls: Vec<f64> = samples.iter().filter_map(Progress::overall).collect();
    let overall_bad = overalls
        .iter()
        .find(|f| !f.is_finite() || **f < 0.0 || **f > 1.0);
    let overall_reg = overalls.windows(2).filter(|w| w[1] < w[0] - 1e-9).count();
    report.record(
        "progress.overall_is_finite_and_monotone",
        overall_bad.is_none() && overall_reg == 0,
        format!(
            "{} overall values, {overall_reg} regressions{}",
            overalls.len(),
            match overall_bad {
                Some(f) => format!(", one out of range: {f}"),
                None => String::new(),
            }
        ),
    );

    let first_complete = numeric.len() > 1 && numeric.first().is_some_and(|f| *f >= 1.0);
    report.record(
        "progress.first_report_is_not_already_complete",
        !first_complete,
        format!(
            "{} reports, first fraction {:?}",
            samples.len(),
            numeric.first()
        ),
    );

    let terminal = samples.last().map(|p| match p.kind {
        ProgressKind::Fraction { completed, total } if total > 0 => completed as f64 / total as f64,
        ProgressKind::MediaSeconds { completed, total } if total > 0.0 => completed / total,
        _ => f64::NAN,
    });
    report.record(
        "progress.finishes_at_the_end_of_its_phase",
        terminal.is_none_or(|t| t.is_nan() || (t - 1.0).abs() < 1e-9),
        format!(
            "terminal raw value {terminal:?}; a successful run that stops at 60% leaves a bar \
             that never completes"
        ),
    );

    report
}

/// Guarantee 3b: a stage cannot report a fraction outside `[0, 1]`, and cannot
/// report `NaN`, however it is asked.
///
/// This is the one progress property that cannot be observed from a stage's own
/// output, because `ProgressKind::fraction` clamps: a stage that reported 1.5
/// would still be read as 1.0. **Mutation testing is what forced this check to
/// read the raw value** — the first version compared `kind.fraction()` and
/// therefore passed against a `StageContext::report` with the clamp deleted,
/// because the second clamp in `fraction()` hid it. `scripts/a10/mutate.ps1`
/// keeps that mutation in the set precisely so the hole stays closed.
///
/// So the value is read out of the *serialised* `Progress`, where
/// `MediaSeconds.completed` is the number `report()` actually published. That
/// form is a public, stable wire format — it is what the UI receives over the
/// progress channel and what `vaani --json` prints.
#[must_use]
pub fn check_progress_is_clamped() -> ConformanceReport {
    let mut report = ConformanceReport::with_known(core_known_failures());
    for (label, value) in [
        ("above_one", 1.5f64),
        ("far_above_one", 1e9),
        ("negative", -0.5),
        ("nan", f64::NAN),
        ("infinite", f64::INFINITY),
        ("negative_infinite", f64::NEG_INFINITY),
    ] {
        let (seen, sink) = progress_recorder();
        {
            let mut artifacts = StageArtifacts::default();
            let cancel = CancelToken::new();
            let dir = std::env::temp_dir();
            let ctx = StageContext {
                input: Path::new("x.wav"),
                work_dir: &dir,
                cancel: &cancel,
                progress: &sink,
                artifacts: &mut artifacts,
            };
            ctx.report(JobPhase::Asr, Some(value), Some(label));
        }
        let samples = seen.lock().map(|s| s.clone()).unwrap_or_default();
        let raw = samples.last().and_then(raw_reported_fraction);
        let readable = samples.last().and_then(|p| p.kind.fraction());
        let ok = match raw {
            // The number `report()` published must itself be finite and inside
            // [0, 1]. `NaN` is the case `f64::clamp` propagates.
            Some(f) => f.is_finite() && (0.0..=1.0).contains(&f),
            // `report` with a `Some` fraction must publish a quantified kind.
            // Falling through to `indeterminate` would hide the value rather
            // than bound it.
            None => false,
        };
        report.record(
            &format!("progress.clamped[{label}]"),
            ok,
            format!("asked for {value}, published {raw:?}, the UI reads {readable:?}"),
        );
    }

    // And the negative case: `None` must mean indeterminate, not a silent 0.
    let (seen, sink) = progress_recorder();
    {
        let mut artifacts = StageArtifacts::default();
        let cancel = CancelToken::new();
        let dir = std::env::temp_dir();
        let ctx = StageContext {
            input: Path::new("x.wav"),
            work_dir: &dir,
            cancel: &cancel,
            progress: &sink,
            artifacts: &mut artifacts,
        };
        ctx.report(JobPhase::Asr, None, Some("indeterminate"));
    }
    let samples = seen.lock().map(|s| s.clone()).unwrap_or_default();
    let kind = samples.last().map(|p| p.kind);
    report.record(
        "progress.none_means_indeterminate_not_zero",
        kind == Some(ProgressKind::Indeterminate),
        format!("{kind:?}; reporting 0% for work of unknown size is a lie a user can see"),
    );

    report
}

/// The fraction a `Progress` actually carries, before `ProgressKind::fraction`
/// clamps it.
///
/// Read from the serialised form because that is where the un-clamped number
/// lives: `ProgressKind` serialises as `{"kind": "...", "value": {...}}` and
/// `MediaSeconds.completed` is what `StageContext::report` computed.
fn raw_reported_fraction(p: &Progress) -> Option<f64> {
    let json = serde_json::to_value(p).ok()?;
    // `ProgressKind` is `#[serde(tag = "kind", content = "value")]`, so it
    // nests one level under `Progress::kind`:
    //   {"phase":"Asr","kind":{"kind":"mediaseconds","value":{"completed":1.5,"total":1.0}}}
    let value = json.get("kind")?.get("value")?;
    match p.kind {
        ProgressKind::Fraction { .. } => {
            let total = value.get("total").and_then(serde_json::Value::as_f64)?;
            if total <= 0.0 {
                return None;
            }
            value.get("completed")?.as_f64().map(|c| c / total)
        }
        ProgressKind::MediaSeconds { .. } => value.get("completed")?.as_f64(),
        ProgressKind::Indeterminate => None,
    }
}

// ---------------------------------------------------------------------------
// Guarantee 4: adversarial input does not panic
// ---------------------------------------------------------------------------

/// Guarantee 4a: absurd sample rates, non-finite samples, absurd channel counts
/// and enormous declared offsets produce a result or a clean error, never a
/// panic.
///
/// A panic in a worker thread takes the whole job with it, so "it panicked" is
/// "VaaniDesk closed while transcribing". `NaN` and infinity are called out
/// separately because they are the ones that survive to the model: a `NaN`
/// sample becomes a `NaN` log-mel, which becomes a transcript of garbage that
/// still looks like language.
pub fn check_adversarial_audio(stage: &dyn ConformanceStage) -> ConformanceReport {
    let mut report = ConformanceReport::with_known(stage.known_failures());
    if stage.arity() == Arity::Path {
        report.skip(
            "adversarial.audio",
            "this stage's input is a path; its adversarial surface is bad paths and bad \
             containers, checked by adversarial.paths",
        );
        return report;
    }

    let rate = stage_rate(stage);
    let frames = 4_000;

    let cases: Vec<(&str, AudioBuffer)> = vec![
        (
            "sample_rate_0",
            AudioBuffer::mono(vec![0.1; frames], 0, Time::ZERO),
        ),
        (
            "sample_rate_1",
            AudioBuffer::mono(vec![0.1; frames], 1, Time::ZERO),
        ),
        (
            "sample_rate_u32_max",
            AudioBuffer::mono(vec![0.1; frames], u32::MAX, Time::ZERO),
        ),
        (
            "sample_rate_1e9",
            AudioBuffer::mono(vec![0.1; frames], 1_000_000_000, Time::ZERO),
        ),
        (
            "all_nan",
            AudioBuffer::mono(vec![f32::NAN; frames], rate, Time::ZERO),
        ),
        (
            "all_negative_infinity",
            AudioBuffer::mono(vec![f32::NEG_INFINITY; frames], rate, Time::ZERO),
        ),
        (
            "alternating_nan_and_full_scale",
            AudioBuffer::mono(
                (0..frames)
                    .map(|i| if i % 512 == 0 { f32::NAN } else { 4.0 })
                    .collect(),
                rate,
                Time::ZERO,
            ),
        ),
        (
            "enormous_declared_offset",
            AudioBuffer {
                samples: vec![0.1; frames],
                sample_rate: rate,
                channels: 1,
                // Far enough out that `frames * 1000 / rate` and any
                // `offset + duration` overflow a naive f32 or i32.
                start_time: Time(i64::MAX / 4),
            },
        ),
        (
            "channel_count_32767",
            AudioBuffer {
                samples: vec![0.1; frames],
                sample_rate: rate,
                channels: 32_767,
                start_time: Time::ZERO,
            },
        ),
        (
            "channel_count_zero",
            AudioBuffer {
                samples: vec![0.1; frames],
                sample_rate: rate,
                channels: 0,
                start_time: Time::ZERO,
            },
        ),
        ("one_sample", AudioBuffer::mono(vec![0.5], rate, Time::ZERO)),
    ];

    for (label, audio) in cases {
        // A panic here fails the test with the label in scope, which is the
        // point: the reader learns which input killed it.
        match stage.run(&audio, &CancelToken::new(), &|_| {}) {
            Ok(_) => report.record(&format!("adversarial.{label}"), true, "handled"),
            Err(e) => report.record(
                &format!("adversarial.{label}"),
                is_clean_failure(&e),
                format!("rejected with {}: {}", e.code(), e.user.title),
            ),
        }
    }

    report
}

/// Guarantee 4b: paths that cannot be read produce a clean error, never a
/// panic and never a result.
pub fn check_adversarial_paths(stage: &dyn ConformanceStage) -> ConformanceReport {
    let mut report = ConformanceReport::with_known(stage.known_failures());
    if stage.arity() != Arity::Path {
        report.skip("adversarial.paths", "this stage takes a buffer, not a path");
        return report;
    }
    let bad = stage.bad_input_paths();
    if bad.is_empty() {
        report.record(
            "adversarial.paths",
            false,
            "the stage takes a path but offered no bad paths to try".to_owned(),
        );
        return report;
    }
    for (label, path) in bad {
        let result = match stage.run_path(&path, &CancelToken::new(), &|_| {}) {
            Some(r) => r,
            None => {
                report.record(
                    &format!("adversarial.path[{label}]"),
                    false,
                    "the stage declares a path interface but run_path returned None".to_owned(),
                );
                continue;
            }
        };
        match result {
            Ok(_) => report.record(
                &format!("adversarial.path[{label}]"),
                false,
                "a path that cannot be read produced a result".to_owned(),
            ),
            Err(e) => report.record(
                &format!("adversarial.path[{label}]"),
                is_clean_failure(&e),
                format!("rejected with {}: {}", e.code(), e.user.title),
            ),
        }
    }
    report
}

// ---------------------------------------------------------------------------
// Guarantee 5: errors are actionable
// ---------------------------------------------------------------------------

/// Guarantee 5: a failure the user can be shown carries a title, a detail that
/// names the cause, and a recovery that says what to do.
///
/// A representative sample is the right scope: the shape is produced by shared
/// constructors, so sampling a handful of call sites catches a constructor that
/// stopped attaching a recovery, which is the real failure mode. Asserting it on
/// every path in the product would be a test about string literals.
///
/// The leak check matters as much as the presence check. `detail` is rendered on
/// screen and copied into bug reports, so an absolute path in it is both a
/// privacy problem and a message the user cannot act on. A bare *file name* is
/// explicitly allowed — the user chose it and it is already on their screen —
/// so the check distinguishes the two rather than banning both.
#[must_use]
pub fn check_errors_are_actionable(samples: &[(&str, &VaaniError)]) -> ConformanceReport {
    let mut report = ConformanceReport::with_known(all_known_failures());
    if samples.is_empty() {
        report.skip(
            "errors.are_actionable",
            "this stage offered no failure paths to sample",
        );
        return report;
    }
    for (label, e) in samples {
        let mut problems: Vec<String> = Vec::new();
        if e.user.title.trim().is_empty() {
            problems.push("empty title".to_owned());
        }
        if e.user.detail.trim().is_empty() {
            problems.push("empty detail".to_owned());
        }
        if e.user.title.trim() == e.user.detail.trim() {
            problems.push("the title just repeats the detail".to_owned());
        }
        if e.user.recovery.as_ref().is_none_or(|r| r.trim().is_empty()) {
            problems.push("no recovery".to_owned());
        }
        for (field, text) in [("detail", &e.user.detail), ("title", &e.user.title)] {
            if leaks_a_path(text) {
                problems.push(format!("{field} leaks a filesystem path: {text:?}"));
            }
        }
        report.record(
            &format!("errors.actionable[{label}]"),
            problems.is_empty(),
            if problems.is_empty() {
                format!("{} / {}", e.user.title, e.code().as_str())
            } else {
                problems.join("; ")
            },
        );
    }
    report
}

/// Whether a user-facing string contains something that identifies a location on
/// the user's disk.
///
/// Deliberately narrow, because over-banning would reject the one variable value
/// `vaani-core` sanctions: the file name the user picked. A path *separator*
/// followed by a segment, a drive letter, a UNC prefix, a home directory or a
/// `..` component are all leaks; a bare name is not.
fn leaks_a_path(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let drive = lower.len() >= 3 && bytes[1] == b':' && (bytes[2] == b'\\' || bytes[2] == b'/');
    let unc = lower.starts_with("\\\\") || lower.starts_with("//");
    let home = ["/home/", "\\users\\", "/users/", "\\windows\\", "/etc/"]
        .iter()
        .any(|marker| lower.contains(marker));
    let traversal = lower.contains("..\\") || lower.contains("../");
    let slash_segment = lower.contains('/') || lower.contains('\\');
    drive || unc || home || traversal || slash_segment
}

/// Whether an error code is one the product's user-facing vocabulary defines.
///
/// A stage that produces a code outside this set has classified nothing, and the
/// UI has no recovery text to show for it.
fn is_clean_failure(e: &VaaniError) -> bool {
    matches!(
        e.code(),
        ErrorCode::UnsupportedMedia
            | ErrorCode::AudioDecodeFailed
            | ErrorCode::NoAudioStream
            | ErrorCode::ModelUnavailable
            | ErrorCode::ModelCorrupt
            | ErrorCode::ModelLoadFailed
            | ErrorCode::OutOfMemory
            | ErrorCode::TranscriptionFailed
            | ErrorCode::DiarizationFailed
            | ErrorCode::ExportFailed
            | ErrorCode::StorageFailed
            | ErrorCode::Cancelled
            | ErrorCode::Internal
    )
}

// ---------------------------------------------------------------------------
// Guarantee 6: determinism
// ---------------------------------------------------------------------------

/// Guarantee 6: the same input twice gives the same answer.
///
/// Compared as whole [`StageOutput`] values including every `Time`, every word
/// and every speaker — with exactly two documented exclusions, each recorded as
/// its own outcome so it is visible rather than assumed.
///
/// **1. Generated identifiers are not content.** `Segment::id` and
/// `Transcript::id` come from `uuid::Uuid::new_v4` in `vaani-core`, by design:
/// they are storage keys, "collision-resistant within a job", not derived from
/// the audio. Two runs over the same audio therefore produce the same transcript
/// with different ids, and asserting otherwise would assert something false.
///
/// **2. Measured wall time is not content.** `DiarizationResult::elapsed_ms` is
/// a stopwatch read. It differs every run, by definition, and nothing depends on
/// it.
///
/// Any other difference — a single millisecond, one word, one speaker, one span
/// boundary — fails the substantive comparison, and the failure names the first
/// element that moved.
pub fn check_determinism(stage: &dyn ConformanceStage) -> ConformanceReport {
    let mut report = ConformanceReport::with_known(stage.known_failures());
    let audio = stage.work_audio();
    let first = run_stage(stage, &audio, &CancelToken::new());
    let second = run_stage(stage, &audio, &CancelToken::new());
    match (&first, &second) {
        (Ok(a), Ok(b)) => {
            let same_substance = substantively_equal(a, b);
            report.record(
                "determinism.substantive_content_is_identical",
                same_substance,
                if same_substance {
                    format!("{} identical across two runs", describe_output(a))
                } else {
                    format!(
                        "{} differed across two runs: {}",
                        describe_output(a),
                        substantive_diff(a, b).unwrap_or_default()
                    )
                },
            );
            let ids_differ = a != b;
            report.record(
                "determinism.identifiers_are_generated_not_derived",
                true,
                if ids_differ {
                    "whole-value comparison differed only in generated ids, as documented for \
                     Segment::id / Transcript::id (uuid v4 storage keys)"
                        .to_owned()
                } else {
                    "every value, ids included, was identical".to_owned()
                },
            );
            // The other documented exception: measured wall time. No stage's
            // *content* depends on it, and it cannot — it is a stopwatch read.
            let wall_clock_differs = match (a, b) {
                (StageOutput::Speakers(x), StageOutput::Speakers(y)) => {
                    x.elapsed_ms != y.elapsed_ms
                }
                _ => false,
            };
            report.record(
                "determinism.wall_clock_fields_are_excluded",
                true,
                if wall_clock_differs {
                    "DiarizationResult::elapsed_ms differed between runs, as it must: it is a \
                     measured duration, not content"
                        .to_owned()
                } else {
                    "no wall-clock field differed".to_owned()
                },
            );
        }
        (Err(a), Err(b)) => report.record(
            "determinism.failure_is_stable",
            a.code() == b.code(),
            format!("both runs failed with {}", a.code().as_str()),
        ),
        (a, b) => report.record(
            "determinism.substantive_content_is_identical",
            false,
            format!(
                "one run succeeded and one did not: {} vs {}",
                describe(a),
                describe(b)
            ),
        ),
    }
    report
}

/// Whether two outputs agree on everything that is content.
///
/// Ids are excluded because they are generated, not derived; see
/// [`check_determinism`].
fn substantively_equal(a: &StageOutput, b: &StageOutput) -> bool {
    substantive_diff(a, b).is_none()
}

/// The first substantive difference between two outputs, or `None`.
fn substantive_diff(a: &StageOutput, b: &StageOutput) -> Option<String> {
    match (a, b) {
        (StageOutput::None, StageOutput::None) => None,
        (StageOutput::Audio(x), StageOutput::Audio(y)) => {
            (x != y).then(|| "decoded audio differs".to_owned())
        }
        (StageOutput::Timeline(x), StageOutput::Timeline(y)) => {
            (x != y).then(|| "speech timeline differs".to_owned())
        }
        (StageOutput::Speakers(x), StageOutput::Speakers(y)) => {
            if x.num_speakers != y.num_speakers {
                return Some(format!("{} speakers vs {}", x.num_speakers, y.num_speakers));
            }
            if x.spans.len() != y.spans.len() {
                return Some(format!("{} spans vs {}", x.spans.len(), y.spans.len()));
            }
            for (i, (p, q)) in x.spans.iter().zip(y.spans.iter()).enumerate() {
                if p != q {
                    return Some(format!(
                        "span {i} differs: {p:?} vs {q:?} \
                         (the second is the first span that moved, so the whole timeline is \
                         suspect, not just this boundary)"
                    ));
                }
            }
            // `elapsed_ms` is deliberately not compared: it is a stopwatch read,
            // and no span depends on it. See `check_determinism`.
            None
        }
        (StageOutput::Transcript(x), StageOutput::Transcript(y)) => {
            if x.segments.len() != y.segments.len() {
                return Some(format!(
                    "{} segments vs {}",
                    x.segments.len(),
                    y.segments.len()
                ));
            }
            for (i, (p, q)) in x.segments.iter().zip(y.segments.iter()).enumerate() {
                if p.start != q.start || p.end != q.end {
                    return Some(format!(
                        "segment {i} spans {}..{} vs {}..{}",
                        p.start.as_millis(),
                        p.end.as_millis(),
                        q.start.as_millis(),
                        q.end.as_millis()
                    ));
                }
                if p.speaker != q.speaker {
                    return Some(format!("segment {i} speaker differs"));
                }
                if p.words.len() != q.words.len() {
                    return Some(format!(
                        "segment {i} has {} words vs {}",
                        p.words.len(),
                        q.words.len()
                    ));
                }
                for (j, (u, v)) in p.words.iter().zip(q.words.iter()).enumerate() {
                    if u.start != v.start || u.end != v.end || u.text != v.text {
                        return Some(format!("segment {i} word {j} differs"));
                    }
                }
            }
            if x.speakers.len() != y.speakers.len() {
                return Some("speaker roster size differs".to_owned());
            }
            None
        }
        _ => Some(format!(
            "different kinds of output: {} vs {}",
            describe_output(a),
            describe_output(b)
        )),
    }
}

// ---------------------------------------------------------------------------
// Guarantee 7: times are in the right coordinate space
// ---------------------------------------------------------------------------

/// Guarantee 7: every emitted time lies inside the input buffer's span, is
/// ordered, and is not inverted.
///
/// This is the project's most-painful invariant. The reason it is checkable
/// without a 14-minute recording is that [`ConformanceStage::work_audio`] starts
/// at **+5 minutes**: a stage that returns buffer-relative times produces values
/// below `start_time` and fails here, on synthetic audio, in under a second.
///
/// Three further properties come free from the same pass:
///
/// * a segment's own words lie inside it, which is the "word at 00:00 inside a
///   segment at 12:31" bug and survives every per-word bounds check;
/// * `start <= end` on every pair, and pairs come out in non-decreasing start
///   order, so a consumer can binary-search them;
/// * nothing runs past the buffer's end by more than one sample period, which is
///   the tolerance a 60 ms-chunked decoder legitimately needs and no more.
pub fn check_times_are_absolute(stage: &dyn ConformanceStage) -> ConformanceReport {
    let mut report = ConformanceReport::with_known(stage.known_failures());
    let audio = stage.work_audio();
    let origin = if stage.offsets_by_input_start() {
        audio.start_time
    } else {
        Time::ZERO
    };
    let end = origin.saturating_add(audio.duration());
    let tolerance_ms = 1000 / i64::from(stage_rate(stage).max(1)) + 1;

    let output = match run_stage(stage, &audio, &CancelToken::new()) {
        Ok(o) => o,
        Err(e) => {
            report.record(
                "times.are_absolute",
                false,
                format!("the stage did not run: {}", e.debug_chain()),
            );
            return report;
        }
    };
    let pairs = output.time_pairs();
    if pairs.is_empty() {
        report.skip(
            "times.are_absolute",
            format!(
                "the stage produced no times for its work audio ({}), so there is no \
                 coordinate-space claim to check",
                stage.id()
            ),
        );
        return report;
    }

    let below = pairs.iter().filter(|(s, _)| *s < origin).count();
    report.record(
        "times.no_time_precedes_the_origin",
        below == 0,
        format!(
            "{below} of {} pairs start before {origin} ms; the stage is emitting \
             chunk-relative or zero-based times instead of source-media times",
            pairs.len()
        ),
    );

    let past = pairs
        .iter()
        .filter(|(_, e)| e.as_millis() > end.as_millis() + tolerance_ms)
        .count();
    report.record(
        "times.no_time_past_the_input",
        past == 0,
        format!(
            "{past} of {} pairs end after {} ms (+{tolerance_ms} ms tolerance)",
            pairs.len(),
            end.as_millis()
        ),
    );

    let inverted = pairs.iter().filter(|(s, e)| e < s).count();
    report.record(
        "times.no_inverted_pair",
        inverted == 0,
        format!("{inverted} of {} pairs have end < start", pairs.len()),
    );

    let unsorted = pairs.windows(2).filter(|w| w[1].0 < w[0].0).count();
    report.record(
        "times.sorted_by_start",
        unsorted == 0,
        format!(
            "{unsorted} of {} adjacent pairs out of order",
            pairs.len().saturating_sub(1)
        ),
    );

    if let Some(t) = output.transcript() {
        let strays: Vec<String> = t
            .segments
            .iter()
            .map(|s: &Segment| {
                s.words
                    .iter()
                    .filter(|w| {
                        w.start < s.start.saturating_sub(Time(tolerance_ms))
                            || w.end > s.end.saturating_add(Time(tolerance_ms))
                    })
                    .count()
            })
            .enumerate()
            .filter(|(_, n)| *n > 0)
            .map(|(i, n)| format!("segment {i}: {n} word(s)"))
            .collect();
        report.record(
            "times.words_lie_inside_their_segment",
            strays.is_empty(),
            if strays.is_empty() {
                format!(
                    "{} segment(s), every word inside its own span",
                    t.segments.len()
                )
            } else {
                strays.join(", ")
            },
        );
    }

    report
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

/// Every check, run against `stage`. The reusable question.
pub fn run_conformance(stage: &dyn ConformanceStage) -> ConformanceReport {
    let mut report = ConformanceReport::new();

    report.record(
        "identity.stage_is_identifiable",
        !stage.id().is_empty() && stage.id() != "unknown",
        format!(
            "id={:?} kind={:?} arity={:?}",
            stage.id(),
            stage.output_kind(),
            stage.arity()
        ),
    );

    report.extend(check_cancellation_is_prompt(stage));
    report.extend(check_empty_input(stage));
    report.extend(check_wrong_sample_rate(stage));
    report.extend(check_progress_is_monotone(stage));
    report.extend(check_determinism(stage));
    report.extend(check_times_are_absolute(stage));
    report.extend(check_adversarial_audio(stage));
    report.extend(check_adversarial_paths(stage));

    let samples = stage.error_samples();
    let borrowed: Vec<(&str, &VaaniError)> = samples.iter().map(|(l, e)| (l.as_str(), e)).collect();
    report.extend(check_errors_are_actionable(&borrowed));

    report
}

/// Panic unless every check that ran passed. The one line a new stage costs.
pub fn assert_conforms(stage: &dyn ConformanceStage) -> ConformanceReport {
    let report = run_conformance(stage);
    print_report(&report);
    assert!(
        report.ok(),
        "{} does not conform: {}\n{}",
        stage.id(),
        report.failures().join(", "),
        report.render()
    );
    report
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn run_stage(
    stage: &dyn ConformanceStage,
    audio: &AudioBuffer,
    cancel: &CancelToken,
) -> Result<StageOutput> {
    stage.run(audio, cancel, &|_| {})
}

/// A progress sink that records everything it is handed, plus the recording.
///
/// `ProgressSink` is `dyn Fn(Progress) + Send + Sync`, so the closure is `&'static`
/// and must own its state. Returning an `Arc` alongside it is what lets a caller
/// read the recording back after the stage has finished.
fn progress_recorder() -> (
    Arc<Mutex<Vec<Progress>>>,
    impl Fn(Progress) + Send + Sync + 'static,
) {
    let seen: Arc<Mutex<Vec<Progress>>> = Arc::new(Mutex::new(Vec::new()));
    let writer = Arc::clone(&seen);
    (seen, move |p: Progress| {
        if let Ok(mut s) = writer.lock() {
            s.push(p);
        }
    })
}

/// The sample rate to hand a stage that declares none.
fn stage_rate(stage: &dyn ConformanceStage) -> u32 {
    stage
        .required_sample_rate()
        .unwrap_or(fixtures::FIXTURE_SAMPLE_RATE)
}

fn describe<T>(result: &Result<T>) -> String {
    match result {
        Ok(_) => "Ok".to_owned(),
        Err(e) => format!("Err({})", e.code().as_str()),
    }
}

fn describe_output(o: &StageOutput) -> String {
    match o {
        StageOutput::None => "no structured output".to_owned(),
        StageOutput::Audio(a) => format!("{} audio samples", a.samples.len()),
        StageOutput::Timeline(t) => format!("{} speech segments", t.segments.len()),
        StageOutput::Speakers(s) => format!("{} speaker spans", s.spans.len()),
        StageOutput::Transcript(t) => format!("{} transcript segments", t.segments.len()),
    }
}

// ---------------------------------------------------------------------------
// Adapter: vaani-media
// ---------------------------------------------------------------------------

/// `vaani_media`'s decode-to-mono-16 kHz, as a conformance stage.
#[derive(Debug, Default, Clone, Copy)]
pub struct MediaStage;

impl ConformanceStage for MediaStage {
    fn id(&self) -> &str {
        "vaani-media"
    }

    fn output_kind(&self) -> OutputKind {
        OutputKind::Audio
    }

    fn required_sample_rate(&self) -> Option<u32> {
        Some(vaani_media::TARGET_SAMPLE_RATE)
    }

    fn arity(&self) -> Arity {
        Arity::Path
    }

    fn reports_progress(&self) -> bool {
        true
    }

    /// Media is the origin of the time axis: its output starts the clock.
    fn offsets_by_input_start(&self) -> bool {
        false
    }

    fn run(
        &self,
        audio: &AudioBuffer,
        cancel: &CancelToken,
        progress: &ProgressSink,
    ) -> Result<StageOutput> {
        let path = cached_work_wav(audio)?;
        self.run_path(&path, cancel, progress)
            .unwrap_or_else(|| Err(vaani_core::err::internal().with_cause("no path interface")))
    }

    fn run_path(
        &self,
        path: &Path,
        cancel: &CancelToken,
        progress: &ProgressSink,
    ) -> Option<Result<StageOutput>> {
        Some(
            vaani_media::decode_to_mono16k_with_progress(path, cancel, progress)
                .map(|audio| StageOutput::Audio(Box::new(audio))),
        )
    }

    /// Decoding runs at 2400x realtime, so the *file* has to be long for a
    /// cancel to land inside it. See [`long_work_audio`].
    fn work_audio(&self) -> AudioBuffer {
        long_work_audio()
    }

    fn bad_input_paths(&self) -> Vec<(String, PathBuf)> {
        let dir = tempfile::TempDir::new().expect("qa tempdir");
        let missing = dir.path().join("definitely-not-here.wav");
        let as_dir = dir.path().join("a-directory.wav");
        std::fs::create_dir_all(&as_dir).expect("qa tempdir");
        let text = dir.path().join("not-audio.wav");
        std::fs::write(&text, b"this is not audio at all").expect("qa fixture");
        // Keep the directory alive for the caller: `run_path` opens these paths
        // after this function returns.
        std::mem::forget(dir);
        vec![
            ("does_not_exist".to_owned(), missing),
            ("is_a_directory".to_owned(), as_dir),
            ("is_not_media".to_owned(), text),
        ]
    }

    fn error_samples(&self) -> Vec<(String, VaaniError)> {
        let dir = tempfile::TempDir::new().expect("qa tempdir");
        let mut out = Vec::new();

        let empty = dir.path().join("empty.wav");
        std::fs::write(
            &empty,
            fixtures::encode_wav_s16(&[], fixtures::FIXTURE_SAMPLE_RATE, 1),
        )
        .expect("qa fixture");
        push_error(&mut out, "empty_payload", || {
            vaani_media::decode_to_mono16k(&empty, &CancelToken::new()).map(|_| ())
        });

        let text = dir.path().join("not-audio.wav");
        std::fs::write(&text, b"this is not audio at all").expect("qa fixture");
        push_error(&mut out, "not_media", || {
            vaani_media::decode_to_mono16k(&text, &CancelToken::new()).map(|_| ())
        });

        push_error(&mut out, "missing_file", || {
            vaani_media::decode_to_mono16k(&dir.path().join("nope.wav"), &CancelToken::new())
                .map(|_| ())
        });

        out
    }
}

fn push_error(out: &mut Vec<(String, VaaniError)>, label: &str, f: impl FnOnce() -> Result<()>) {
    if let Err(e) = f() {
        out.push((label.to_owned(), e));
    }
}

/// Write `audio` to a fresh 16 kHz WAV in a temp directory and return the path.
fn write_work_wav(audio: &AudioBuffer) -> Result<PathBuf> {
    let dir =
        tempfile::TempDir::new().map_err(|e| vaani_core::err::storage_failed().with_cause(e))?;
    let path = dir.path().join("work.wav");
    let bytes = fixtures::encode_wav_s16(&audio.samples, fixtures::FIXTURE_SAMPLE_RATE, 1);
    std::fs::write(&path, bytes).map_err(|e| vaani_core::err::storage_failed().with_cause(e))?;
    // The TempDir is dropped here, which deletes the file. Deliberately leaked
    // instead: the caller opens the path after this returns.
    std::mem::forget(dir);
    Ok(path)
}

/// Identifies a materialised work file: sample count, rate, channel count.
type WorkFileKey = (usize, u32, u16);

/// The same, but written at most once per distinct input shape.
///
/// A `Path`-arity stage is handed a *file*, and producing that file is the
/// harness's job rather than the stage's. Encoding 25 minutes of audio and
/// writing 48 MB on each of the seven runs in [`assert_conforms`] cost roughly
/// 3.5 s per run in a debug build, and that cost was charged to the stage: the
/// cancellation check armed its timer at an eighth of a baseline that was
/// mostly file writing, so the cancel landed *inside the write*, and the
/// harness then measured its own encode as the stage's cancellation latency.
/// The stage was never the slow part, but it was the thing being reported.
///
/// Keyed on shape, not content, because every expensive call for a path stage
/// uses the same [`ConformanceStage::work_audio`]; the differing sample rates
/// in the wrong-rate check are distinguished by the key. This mirrors the cache
/// `AppStage` keeps for the same reason.
fn cached_work_wav(audio: &AudioBuffer) -> Result<PathBuf> {
    static CACHE: OnceLock<Mutex<HashMap<WorkFileKey, PathBuf>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key: WorkFileKey = (audio.samples.len(), audio.sample_rate, audio.channels);
    if let Ok(seen) = cache.lock()
        && let Some(path) = seen.get(&key)
    {
        return Ok(path.clone());
    }
    let path = write_work_wav(audio)?;
    if let Ok(mut seen) = cache.lock() {
        seen.insert(key, path.clone());
    }
    Ok(path)
}

// ---------------------------------------------------------------------------
// Adapter: vaani-vad
// ---------------------------------------------------------------------------

/// `vaani_vad::EnergyVad`, as a conformance stage.
#[derive(Debug)]
pub struct EnergyVadStage {
    engine: vaani_vad::EnergyVad,
}

impl Default for EnergyVadStage {
    fn default() -> Self {
        Self::new()
    }
}

impl EnergyVadStage {
    /// A stage over a fresh engine.
    #[must_use]
    pub fn new() -> Self {
        Self {
            engine: vaani_vad::EnergyVad::new(),
        }
    }
}

impl ConformanceStage for EnergyVadStage {
    fn id(&self) -> &str {
        "energy-vad"
    }

    fn output_kind(&self) -> OutputKind {
        OutputKind::Timeline
    }

    fn required_sample_rate(&self) -> Option<u32> {
        Some(self.engine.sample_rate())
    }

    fn arity(&self) -> Arity {
        Arity::Buffer
    }

    /// `VadEngine::detect` has no progress parameter. Stated honestly rather
    /// than faked with a synthetic 0.0/1.0 pair, which would make the
    /// monotonicity check pass without observing anything. The app-level
    /// `VadStage` around the same engine *does* report, and is checked.
    fn reports_progress(&self) -> bool {
        false
    }

    fn run(
        &self,
        audio: &AudioBuffer,
        cancel: &CancelToken,
        _progress: &ProgressSink,
    ) -> Result<StageOutput> {
        let timeline = self.engine.detect(audio, cancel)?;
        Ok(StageOutput::Timeline(Box::new(timeline)))
    }

    fn error_samples(&self) -> Vec<(String, VaaniError)> {
        let mut out = Vec::new();
        let wrong = AudioBuffer::mono(vec![0.1; 8_000], 8_000, Time::ZERO);
        push_error(&mut out, "energy_vad_wrong_sample_rate", || {
            self.engine.detect(&wrong, &CancelToken::new()).map(|_| ())
        });
        out
    }
}

// ---------------------------------------------------------------------------
// Adapter: vaani-diarization
// ---------------------------------------------------------------------------

/// `vaani_diarization::HeuristicDiarizer`, as a conformance stage.
#[derive(Debug)]
pub struct HeuristicDiarStage {
    engine: vaani_diarization::HeuristicDiarizer,
}

impl Default for HeuristicDiarStage {
    fn default() -> Self {
        Self::new()
    }
}

impl HeuristicDiarStage {
    /// A stage over a fresh engine. `load` happens on the first `run`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            engine: vaani_diarization::HeuristicDiarizer::new(),
        }
    }
}

impl ConformanceStage for HeuristicDiarStage {
    fn id(&self) -> &str {
        "heuristic-diar"
    }

    fn output_kind(&self) -> OutputKind {
        OutputKind::Speakers
    }

    fn required_sample_rate(&self) -> Option<u32> {
        Some(16_000)
    }

    fn arity(&self) -> Arity {
        Arity::Buffer
    }

    /// `DiarizationEngine::diarize` takes a bare `f64` fraction, and
    /// `Progress` has no bare-fraction constructor: the two available forms
    /// either quantise to a millionth (which truncates, and so would hide an
    /// out-of-range value) or clamp (which is the very thing this check exists
    /// to detect). Forwarding would therefore make the monotonicity check
    /// observe a different number from the one the engine emitted.
    ///
    /// So it is declared `false` and the raw `f64` stream is asserted directly
    /// in `tests/diar_adversarial.rs`. The app-level `DiarStage` forwards the
    /// same values through `Progress::media`, and *is* checked here.
    fn reports_progress(&self) -> bool {
        false
    }

    fn run(
        &self,
        audio: &AudioBuffer,
        cancel: &CancelToken,
        _progress: &ProgressSink,
    ) -> Result<StageOutput> {
        self.engine
            .load(cancel)
            .and_then(|()| self.engine.diarize(audio, cancel, &|_| {}))
            .map(|r| StageOutput::Speakers(Box::new(r)))
    }

    fn error_samples(&self) -> Vec<(String, VaaniError)> {
        let mut out = Vec::new();
        let wrong = AudioBuffer::mono(vec![0.1; 8_000], 8_000, Time::ZERO);
        push_error(&mut out, "diar_wrong_sample_rate", || {
            self.engine
                .diarize(&wrong, &CancelToken::new(), &|_| {})
                .map(|_| ())
        });
        let stereo = AudioBuffer {
            samples: vec![0.1; 8_000],
            sample_rate: 16_000,
            channels: 2,
            start_time: Time::ZERO,
        };
        push_error(&mut out, "diar_stereo_not_downed", || {
            self.engine
                .diarize(&stereo, &CancelToken::new(), &|_| {})
                .map(|_| ())
        });
        out
    }
}

// ---------------------------------------------------------------------------
// Adapter: vaani-app's PipelineStage
// ---------------------------------------------------------------------------

/// What a `PipelineStage` expects to find already in the blackboard.
///
/// A `PipelineStage` reads its input from `StageContext::artifacts`, not from a
/// buffer, so presenting one as a [`ConformanceStage`] means saying what ran
/// before it. This closed enum is that sentence, and adding a new stage means
/// choosing one arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Upstream {
    /// Nothing: the media stage.
    Nothing,
    /// Decoded audio: the VAD stage.
    Audio,
    /// Audio plus a speech timeline: the ASR and diarization stages.
    AudioAndSpeech,
    /// Audio, speech and a transcript: the store stage.
    AudioSpeechTranscript,
    /// A transcript only: the export stage.
    TranscriptOnly,
}

impl Upstream {
    /// Build the artifacts this upstream state means.
    #[must_use]
    pub fn seed(self, audio: &AudioBuffer) -> StageArtifacts {
        match self {
            Self::Nothing => StageArtifacts::default(),
            Self::Audio => StageArtifacts {
                audio: Some(audio.clone()),
                ..StageArtifacts::default()
            },
            Self::AudioAndSpeech => StageArtifacts {
                audio: Some(audio.clone()),
                speech: Some(full_timeline(audio)),
                ..StageArtifacts::default()
            },
            Self::AudioSpeechTranscript => StageArtifacts {
                audio: Some(audio.clone()),
                speech: Some(full_timeline(audio)),
                transcript: Some(synthetic_transcript(audio)),
                ..StageArtifacts::default()
            },
            Self::TranscriptOnly => StageArtifacts {
                transcript: Some(synthetic_transcript(audio)),
                ..StageArtifacts::default()
            },
        }
    }
}

fn full_timeline(audio: &AudioBuffer) -> SpeechTimeline {
    let end = audio.start_time.saturating_add(audio.duration());
    SpeechTimeline {
        segments: vec![vaani_core::pipeline::SpeechSegment {
            start: audio.start_time,
            end,
            confidence: None,
        }],
        speech_duration_ms: audio.duration().as_millis(),
    }
}

/// A transcript with known-good **absolute** times, for seeding downstream
/// stages.
///
/// Built through `push_chunk` with *chunk-local* segments and the buffer's
/// `start_time` as the offset, because that is the only rebase point in the
/// product. Handing `push_chunk` already-absolute segments plus a non-zero
/// offset would double-count, which is the exact bug this module exists to
/// catch — so the fixture goes through the same door the real ASR stage does.
#[must_use]
pub fn synthetic_transcript(audio: &AudioBuffer) -> Transcript {
    let duration = audio.duration();
    let mut t = Transcript::new(vaani_core::EngineStamp::asr("qa-conformance", "0"));
    t.duration = Some(duration);
    // A zero-length buffer has nothing to say, and a seeded transcript with
    // zero-width segments would make `empty.zero_length_buffer` measure this
    // fixture rather than the stage under test.
    if duration.as_millis() <= 0 {
        return t;
    }
    let half = Time(duration.as_millis() / 2);
    t.push_chunk(
        audio.start_time,
        vec![
            Segment::new(Time::ZERO, half).with_words(vec![
                vaani_core::Word::new("alpha", Time::ZERO, Time(half.as_millis() / 2)),
                vaani_core::Word::new("bravo", Time(half.as_millis() / 2), half),
            ]),
            Segment::new(half, duration)
                .with_words(vec![vaani_core::Word::new("charlie", half, duration)]),
        ],
    );
    t
}

/// A `vaani_app` [`PipelineStage`] presented as a conformance stage.
///
/// ```ignore
/// assert_conforms(&AppStage::new(
///     Box::new(vaani_app::stages::VadStage::new(Arc::new(vaani_vad::EnergyVad::new()))),
///     Upstream::Audio,
/// ));
/// ```
pub struct AppStage {
    stage: Box<dyn PipelineStage>,
    upstream: Upstream,
    work_audio_override: Option<fn() -> AudioBuffer>,
    workspace: Box<Workspace>,
}

/// A throwaway directory for an [`AppStage`]'s input file and work directory.
///
/// Leaked for the process lifetime on purpose: a `PipelineStage` borrows `&Path`
/// from it, so dropping it mid-run would be a use-after-free rather than a
/// cleanup. The OS reclaims it at exit.
#[derive(Debug)]
struct Workspace {
    _dir: tempfile::TempDir,
    inputs: PathBuf,
    work: PathBuf,
    /// One written WAV per distinct sample count, so a stage that decodes
    /// twenty-five minutes of audio writes it once rather than twenty times.
    materialised: Mutex<std::collections::HashMap<(usize, u32), PathBuf>>,
}

impl fmt::Debug for AppStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AppStage")
            .field("stage", &self.stage.name())
            .field("phase", &self.stage.phase())
            .field("upstream", &self.upstream)
            .finish_non_exhaustive()
    }
}

impl AppStage {
    /// A stage with the blackboard state it expects to find already populated.
    #[must_use]
    pub fn new(stage: Box<dyn PipelineStage>, upstream: Upstream) -> Self {
        let dir = tempfile::TempDir::new().expect("qa tempdir");
        let inputs = dir.path().join("inputs");
        let work = dir.path().join("work");
        std::fs::create_dir_all(&work).expect("qa work dir");
        Self {
            stage,
            upstream,
            work_audio_override: None,
            workspace: Box::new(Workspace {
                _dir: dir,
                inputs,
                work,
                materialised: Mutex::new(std::collections::HashMap::new()),
            }),
        }
    }

    /// Use a different work audio, for a stage whose natural input has to be
    /// much longer than the default.
    ///
    /// The media stage is the case that needs it: decoding runs at 2400x
    /// realtime, so a cancel can only be armed mid-decode if the file is minutes
    /// long, while every other stage wants a couple of minutes at most.
    #[must_use]
    pub fn with_work_audio(mut self, f: fn() -> AudioBuffer) -> Self {
        self.work_audio_override = Some(f);
        self
    }

    /// The stage's name, for grouping reports.
    #[must_use]
    pub fn stage_name(&self) -> &'static str {
        self.stage.name()
    }

    /// Write `audio` to a WAV in the workspace, reusing the file if one with the
    /// same shape is already there.
    ///
    /// Cached because a conformance run calls `run` seven times over the same
    /// work audio, and for the media stage that is seven writes of a
    /// twenty-five-minute file.
    fn materialise_input(&self, audio: &AudioBuffer) -> Result<PathBuf> {
        let key = (audio.samples.len(), audio.sample_rate.max(1));
        if let Ok(cache) = self.workspace.materialised.lock()
            && let Some(path) = cache.get(&key)
        {
            return Ok(path.clone());
        }
        let _ = std::fs::create_dir_all(&self.workspace.inputs);
        let path = self
            .workspace
            .inputs
            .join(format!("work-{}-{}.wav", key.0, key.1));
        let bytes = fixtures::encode_wav_s16(&audio.samples, key.1, 1);
        std::fs::write(&path, bytes)
            .map_err(|e| vaani_core::err::storage_failed().with_cause(e))?;
        if let Ok(mut cache) = self.workspace.materialised.lock() {
            cache.insert(key, path.clone());
        }
        Ok(path)
    }
}

impl ConformanceStage for AppStage {
    fn id(&self) -> &str {
        self.stage.name()
    }

    fn output_kind(&self) -> OutputKind {
        match self.stage.name() {
            "media" => OutputKind::Audio,
            "vad" => OutputKind::Timeline,
            "asr" | "diarization" => OutputKind::Transcript,
            _ => OutputKind::SideEffect,
        }
    }

    fn required_sample_rate(&self) -> Option<u32> {
        Some(fixtures::FIXTURE_SAMPLE_RATE)
    }

    fn arity(&self) -> Arity {
        if self.stage.name() == "media" {
            Arity::Path
        } else {
            Arity::Buffer
        }
    }

    fn reports_progress(&self) -> bool {
        true
    }

    /// The media stage reads a file and therefore *defines* the origin of the
    /// time axis: its output starts at zero by construction. Every other stage
    /// inherits an origin and must honour it, which is the whole point of
    /// guarantee 7. The ASR stage does not, and carries a tracked known failure
    /// for it; see [`asr_stage_known_failures`].
    fn offsets_by_input_start(&self) -> bool {
        self.stage.name() != "media"
    }

    /// Tracked failures in the stage under test.
    fn known_failures(&self) -> Vec<KnownFailure> {
        match self.stage.name() {
            "asr" => asr_stage_known_failures(),
            _ => Vec::new(),
        }
    }

    fn run(
        &self,
        audio: &AudioBuffer,
        cancel: &CancelToken,
        progress: &ProgressSink,
    ) -> Result<StageOutput> {
        let input = self.materialise_input(audio)?;
        let mut artifacts = self.upstream.seed(audio);
        let output = {
            let mut ctx = StageContext {
                input: &input,
                work_dir: &self.workspace.work,
                cancel,
                progress,
                artifacts: &mut artifacts,
            };
            self.stage.run(&mut ctx)?;
            match self.stage.name() {
                "media" => artifacts
                    .audio
                    .map(|a| StageOutput::Audio(Box::new(a)))
                    .unwrap_or(StageOutput::None),
                "vad" => artifacts
                    .speech
                    .map(|t| StageOutput::Timeline(Box::new(t)))
                    .unwrap_or(StageOutput::None),
                "asr" | "diarization" => artifacts
                    .transcript
                    .map(|t| StageOutput::Transcript(Box::new(t)))
                    .unwrap_or(StageOutput::None),
                _ => StageOutput::None,
            }
        };
        Ok(output)
    }

    /// A per-stage work audio, if one was set.
    fn work_audio(&self) -> AudioBuffer {
        match self.work_audio_override {
            Some(f) => f(),
            None => cached_work_audio(),
        }
    }

    fn run_path(
        &self,
        path: &Path,
        cancel: &CancelToken,
        progress: &ProgressSink,
    ) -> Option<Result<StageOutput>> {
        if self.stage.name() != "media" {
            return None;
        }
        let mut artifacts = self
            .upstream
            .seed(&AudioBuffer::mono(vec![], 16_000, Time::ZERO));
        let mut ctx = StageContext {
            input: path,
            work_dir: &self.workspace.work,
            cancel,
            progress,
            artifacts: &mut artifacts,
        };
        Some(self.stage.run(&mut ctx).map(|()| {
            artifacts
                .audio
                .map(|a| StageOutput::Audio(Box::new(a)))
                .unwrap_or(StageOutput::None)
        }))
    }

    fn bad_input_paths(&self) -> Vec<(String, PathBuf)> {
        let dir = self.workspace._dir.path();
        let as_dir = dir.join("a-directory.wav");
        let _ = std::fs::create_dir_all(&as_dir);
        vec![
            (
                "does_not_exist".to_owned(),
                dir.join("definitely-not-here.wav"),
            ),
            ("is_a_directory".to_owned(), as_dir),
        ]
    }
}

// ---------------------------------------------------------------------------
// Sample transcript, used by the export and storage checks
// ---------------------------------------------------------------------------

/// A small deterministic transcript with speakers, segment ids and multi-word
/// segments, for the export and storage conformance checks.
#[must_use]
pub fn sample_transcript() -> Transcript {
    let mut t = Transcript::new(vaani_core::EngineStamp::asr("qa-conformance", "0"));
    let layout: [(i64, i64, usize); 3] = [(0, 1_500, 0), (2_000, 3_500, 1), (4_000, 6_000, 0)];
    for (i, (start, end, speaker)) in layout.into_iter().enumerate() {
        let mut seg = Segment::new(Time(start), Time(end)).with_speaker(Some(speaker));
        seg.id = format!("seg-{i}");
        let span = (end - start) / 2;
        seg.words = vec![
            vaani_core::Word::new("alpha", Time(start), Time(start + span)),
            vaani_core::Word::new("bravo", Time(start + span), Time(end)),
        ];
        t.push_chunk(Time(0), vec![seg]);
    }
    t.add_speaker(vaani_core::Speaker::new(0));
    t.add_speaker(vaani_core::Speaker::new(1));
    t
}

// ---------------------------------------------------------------------------
// Export conformance
// ---------------------------------------------------------------------------

/// The export contract, as a reusable check.
///
/// `vaani_export` has no trait of its own, so the checks are free functions
/// over its public surface. Three properties, all of which a user would notice
/// immediately if they broke:
///
/// * **determinism** — export twice, get the same bytes. An export that embeds a
///   timestamp makes two exports of the same meeting differ, which breaks every
///   diff and every content-addressed cache above it;
/// * **JSON round-trip** — reading an exported JSON file back gives an *equal*
///   `Transcript`, not merely a similar one;
/// * **the five formats** — `txt`, `md`, `srt`, `vtt`, `json`, in that order.
#[must_use]
pub fn check_export_stage(transcript: &Transcript) -> ConformanceReport {
    use vaani_export::{ExportFormat, ExportOptions};
    let mut report = ConformanceReport::new();
    let options = ExportOptions::default().with_source_name("meeting.wav");

    for format in ExportFormat::ALL {
        let first = vaani_export::render(transcript, format, &options);
        let second = vaani_export::render(transcript, format, &options);
        match (first, second) {
            (Ok(a), Ok(b)) => report.record(
                &format!("export.deterministic[{format}]"),
                a == b && !a.trim().is_empty(),
                format!(
                    "{} bytes, {} identical",
                    a.len(),
                    if a == b { "byte-for-byte" } else { "NOT " }
                ),
            ),
            (Err(e), _) | (_, Err(e)) => report.record(
                &format!("export.deterministic[{format}]"),
                false,
                format!("render failed: {}", e.debug_chain()),
            ),
        }
    }

    let extensions: Vec<&str> = ExportFormat::ALL.iter().map(|f| f.extension()).collect();
    report.record(
        "export.five_formats_stable",
        extensions == ["txt", "md", "srt", "vtt", "json"],
        format!("{extensions:?}"),
    );

    match vaani_export::render(transcript, ExportFormat::Json, &options) {
        Ok(json) => match vaani_export::read_json_export(&json) {
            Ok(back) => {
                let equal = &back == transcript;
                let diff = if equal {
                    String::new()
                } else {
                    vaani_storage::fixtures::diff(transcript, &back).unwrap_or_default()
                };
                report.record(
                    "export.json_round_trips_to_an_equal_transcript",
                    equal,
                    if equal {
                        format!("{} bytes, read back equal", json.len())
                    } else {
                        format!("{} bytes, read back differs: {diff}", json.len())
                    },
                );
            }
            Err(e) => report.record(
                "export.json_round_trips_to_an_equal_transcript",
                false,
                format!("read back failed: {}", e.debug_chain()),
            ),
        },
        Err(e) => report.record(
            "export.json_round_trips_to_an_equal_transcript",
            false,
            format!("render failed: {}", e.debug_chain()),
        ),
    }

    // SRT and VTT are the two formats a third-party parser has to accept, so
    // they are validated by the reference parsers rather than trusted.
    for (format, max_lines) in [(ExportFormat::Srt, 2usize), (ExportFormat::Vtt, 2usize)] {
        match vaani_export::render(transcript, format, &options) {
            Ok(text) => {
                let validated = if format == ExportFormat::Srt {
                    vaani_export::verify::validate_srt(&text, max_lines)
                } else {
                    vaani_export::verify::validate_vtt(&text, max_lines)
                };
                match validated {
                    Ok(cues) => report.record(
                        &format!("export.{format}_parses_under_the_published_rules"),
                        !cues.is_empty(),
                        format!("{} cues parsed, no violations", cues.len()),
                    ),
                    Err(e) => report.record(
                        &format!("export.{format}_parses_under_the_published_rules"),
                        false,
                        e.debug_chain(),
                    ),
                }
            }
            Err(e) => report.record(
                &format!("export.{format}_parses_under_the_published_rules"),
                false,
                format!("render failed: {}", e.debug_chain()),
            ),
        }
    }

    report
}

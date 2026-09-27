//! Skip policy: the only four reasons a test may not run, and nothing else.
//!
//! # The disease this replaces
//!
//! The first version of this crate skipped with a free-form `bool` computed at
//! the call site. That is a green build with no information in it, because the
//! call site decides, and the call site is where a *real* failure looks exactly
//! like a missing file. The old `asr_smoke` is the worked example: it skipped
//! with `"could not construct sravaani engine (runtime missing?)"` — a string
//! that admits it might not be a missing runtime at all. A corrupt `encoder.onnx`,
//! a wrong CUDA build, a tokenizer that will not load: all of them produced the
//! same green tick as "we do not have the weights".
//!
//! # The rule
//!
//! A test may skip for exactly four reasons, all of which are facts about the
//! *machine* rather than about the *code*:
//!
//! * [`Requirement::ModelWeights`] — the 2.6 GB SraVaani export is absent.
//! * [`Requirement::OnnxRuntime`] — `onnxruntime.dll` is absent.
//! * [`Requirement::PrivateClip`] — `scratch/clip.wav` is absent.
//! * [`Requirement::RealContainers`] — no operator-supplied sample media dir.
//!
//! Everything else is a failure. In particular:
//!
//! * **"the artefacts directory exists but `is_complete()` is false" is a
//!   skip**, because that is a half-finished download, a machine fact.
//! * **"the artefacts are complete but the engine will not construct" is a
//!   FAILURE.** That is a broken model, a broken runtime, or a bug, and it is
//!   exactly the thing a smoke test exists to find. The distinction is drawn
//!   once, here, by [`Environment::engine_problem`], and it is a hard error,
//!   never a skip.
//!
//! # The audit questions, answered
//!
//! * *How does it decide to skip?* By asking [`Environment`] about one of four
//!   closed [`Requirement`] values. It cannot answer "yes" to anything else,
//!   because there is nothing else to ask.
//! * *Can it mask a real failure?* No. Every real failure is signalled by
//!   `assert`/`panic`/`expect`, none of which route through this module, and
//!   the one case that used to route through it — engine construction — now
//!   panics instead.
//! * *Does it ever skip when the thing under test should have run?* It can, in
//!   exactly one way: if a machine fact is mis-reported. [`SkipGuard`] closes
//!   that by recording every skip in a process-wide ledger, so
//!   `skipped_in_process()` can assert that the number of skips matches the
//!   number of tests that legitimately could not run, and so a test run can
//!   print exactly what did not run.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::OnceLock;

/// The only four conditions under which a test may not run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Requirement {
    /// The SraVaani model export (encoder, decoder, tokenizer, preproc).
    ModelWeights,
    /// The ONNX Runtime native library and its CUDA provider.
    OnnxRuntime,
    /// The private development recording, which is never in the repository.
    PrivateClip,
    /// Operator-supplied real container samples (`VAANI_QA_MEDIA_DIR`).
    RealContainers,
    /// The operator has not opted into the opt-in live ASR gate.
    ///
    /// The fifth value exists because `tests/asr_live.rs` is opt-in for a reason
    /// that is not a machine fact: a 40-second decode costs about 35 seconds on
    /// the CPU execution provider, which is a large fraction of this suite's
    /// runtime. That is a *policy* decision, so it belongs in the closed set
    /// where it is printed and recorded rather than in an `if` that returns
    /// quietly.
    ///
    /// The first version of that test did exactly the quiet thing -- an early
    /// `return` behind an unset environment variable -- and a test run reported
    /// "4 passed" for four tests that never ran. That is the exact disease this
    /// module exists to prevent, and it was reintroduced by the very crate that
    /// wrote the rules.
    OptInLiveAsr,
}

impl Requirement {
    /// A stable slug, used in the ledger and in test output.
    pub const fn slug(self) -> &'static str {
        match self {
            Self::ModelWeights => "model-weights",
            Self::OnnxRuntime => "onnx-runtime",
            Self::PrivateClip => "private-clip",
            Self::RealContainers => "real-containers",
            Self::OptInLiveAsr => "opt-in-live-asr",
        }
    }

    /// The instruction that makes the requirement satisfiable, printed with
    /// every skip. A skip nobody can act on is a skip nobody will fix.
    pub const fn remedy(self) -> &'static str {
        match self {
            Self::ModelWeights => "run `vaani models download` (needs a Hugging Face token)",
            Self::OnnxRuntime => "run scripts\\fetch_runtime.ps1",
            Self::PrivateClip => "place the development recording at scratch\\clip.wav",
            Self::RealContainers => "set VAANI_QA_MEDIA_DIR to a folder of sample media",
            Self::OptInLiveAsr => {
                "set $env:VAANI_QA_RUNTIME_DLL to the onnxruntime.dll and re-run with --nocapture"
            }
        }
    }
}

/// What the machine this process is running on actually has.
///
/// Computed once per process, because every one of these is a filesystem probe
/// and the answer cannot change under a test run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Environment {
    runtime_present: bool,
    runtime_detail: String,
    weights_dir: PathBuf,
    weights_complete: bool,
    weights_missing: Vec<String>,
    private_clip: Option<PathBuf>,
    real_containers: Option<PathBuf>,
    live_asr_opt_in: Option<PathBuf>,
}

impl Environment {
    /// Probe the machine.
    pub fn probe() -> Environment {
        let location = vaani_inference::runtime::location();
        let weights_dir = vaani_asr::default_dir();
        let artifacts = vaani_asr::artifacts_in(&weights_dir);
        Environment {
            runtime_present: location.is_present(),
            runtime_detail: location
                .missing()
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join(", "),
            weights_dir,
            weights_complete: artifacts.is_complete(),
            weights_missing: artifacts.missing().to_vec(),
            private_clip: private_clip(),
            real_containers: std::env::var_os("VAANI_QA_MEDIA_DIR").map(PathBuf::from),
            live_asr_opt_in: std::env::var_os("VAANI_QA_RUNTIME_DLL").map(PathBuf::from),
        }
    }

    /// The process-wide environment.
    pub fn get() -> &'static Environment {
        static ENV: OnceLock<Environment> = OnceLock::new();
        ENV.get_or_init(Environment::probe)
    }

    /// The SraVaani artefact directory.
    pub fn weights_dir(&self) -> &Path {
        &self.weights_dir
    }

    /// Whether the ONNX Runtime was found.
    pub fn runtime_present(&self) -> bool {
        self.runtime_present
    }

    /// Whether the SraVaani export is complete.
    pub fn weights_complete(&self) -> bool {
        self.weights_complete
    }

    /// The private development recording, if it is on this machine.
    pub fn private_clip(&self) -> Option<&Path> {
        self.private_clip.as_deref()
    }

    /// The `onnxruntime.dll` the operator pointed the live ASR gate at, if any.
    ///
    /// The variable names a **file**, not a directory, and that is deliberate: it
    /// is a test-only knob added because
    /// `vaani_inference::RuntimeLocation::discover` cannot find the runtime when
    /// the build's `CARGO_TARGET_DIR` lives outside the repository. See
    /// `vaani_qa::conformance::check_export_stage` for the same shape of
    /// "library and caller disagree about a convenience" and OPEN ISSUES for
    /// the defect itself.
    #[must_use]
    pub fn live_asr_opt_in(&self) -> Option<&Path> {
        self.live_asr_opt_in.as_deref()
    }

    /// The operator-supplied real-container directory, if set.
    pub fn real_containers(&self) -> Option<&Path> {
        self.real_containers.as_deref()
    }

    /// Whether `req` is satisfied on this machine.
    pub fn satisfies(&self, req: Requirement) -> bool {
        match req {
            Requirement::ModelWeights => self.weights_complete,
            Requirement::OnnxRuntime => self.runtime_present,
            Requirement::PrivateClip => self.private_clip.is_some(),
            Requirement::RealContainers => {
                self.real_containers.as_ref().is_some_and(|d| d.is_dir())
            }
            Requirement::OptInLiveAsr => self.live_asr_opt_in.is_some(),
        }
    }

    /// The one-line reason a requirement is unmet, naming what is missing.
    pub fn reason_unmet(&self, req: Requirement) -> String {
        match req {
            Requirement::ModelWeights => format!(
                "{}/{} missing: {}",
                self.weights_dir.display(),
                self.weights_missing.len(),
                if self.weights_missing.is_empty() {
                    "<unknown>".to_owned()
                } else {
                    self.weights_missing.join(", ")
                }
            ),
            Requirement::OnnxRuntime => {
                if self.runtime_detail.is_empty() {
                    format!(
                        "onnxruntime.dll not found at {}",
                        vaani_inference::runtime::location().ort_dll.display()
                    )
                } else {
                    format!("onnxruntime.dll unusable: {}", self.runtime_detail)
                }
            }
            Requirement::PrivateClip => {
                "scratch/clip.wav is absent; it is private and never committed".to_owned()
            }
            Requirement::RealContainers => {
                "VAANI_QA_MEDIA_DIR is unset or is not a directory".to_owned()
            }
            Requirement::OptInLiveAsr => {
                "VAANI_QA_RUNTIME_DLL is unset, so the opt-in live decode did not run".to_owned()
            }
        }
    }

    /// Classify "the artefacts are here but the engine will not open".
    ///
    /// This is deliberately **not** a [`Requirement`]. It is the bug class a
    /// smoke test exists to catch: a present model that will not load. Returning
    /// `Err` here means the caller panics.
    ///
    /// # Errors
    ///
    /// A message naming the stage that failed and the error's chain.
    pub fn engine_problem(&self, stage: &str, error: &vaani_core::VaaniError) -> String {
        assert!(
            self.weights_complete,
            "engine_problem() called with incomplete weights: that is a skip, not a bug"
        );
        format!(
            "{stage}: SraVaani weights are present at {} but the engine failed to \
             open. This is a broken model or a broken runtime, NOT a missing \
             download, and it must not be skipped. {}",
            self.weights_dir.display(),
            error.debug_chain()
        )
    }

    /// One line describing what this machine can and cannot run.
    pub fn summary(&self) -> String {
        let mut out = format!(
            "weights={} runtime={} private_clip={} real_containers={}",
            if self.weights_complete {
                "complete"
            } else {
                "absent"
            },
            if self.runtime_present {
                "present"
            } else {
                "absent"
            },
            self.private_clip.is_some(),
            self.real_containers.is_some()
        );
        for req in ALL_REQUIREMENTS {
            if !self.satisfies(req) {
                out.push_str(&format!(
                    "\n  skip {}: {}",
                    req.slug(),
                    self.reason_unmet(req)
                ));
            }
        }
        out
    }
}

/// Every [`Requirement`], for tests that want to iterate the skip policy itself.
pub const ALL_REQUIREMENTS: [Requirement; 5] = [
    Requirement::ModelWeights,
    Requirement::OnnxRuntime,
    Requirement::PrivateClip,
    Requirement::RealContainers,
    Requirement::OptInLiveAsr,
];

/// Where `scratch/clip.wav` is expected, relative to the repository root.
fn private_clip() -> Option<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)?;
    let p = root.join("scratch").join("clip.wav");
    p.is_file().then_some(p)
}

// ---------------------------------------------------------------------------
// The ledger
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
struct SkipRecord {
    test: String,
    requirement: Requirement,
    reason: String,
}

fn ledger() -> &'static Mutex<Vec<SkipRecord>> {
    static LEDGER: OnceLock<Mutex<Vec<SkipRecord>>> = OnceLock::new();
    LEDGER.get_or_init(|| Mutex::new(Vec::new()))
}

/// Every skip taken in this process, in order.
///
/// A test binary that skips and says nothing is indistinguishable from one that
/// passed. This is what makes the difference measurable from outside.
pub fn skipped_in_process() -> Vec<(String, &'static str, String)> {
    ledger()
        .lock()
        .map(|l| {
            l.iter()
                .map(|r| (r.test.clone(), r.requirement.slug(), r.reason.clone()))
                .collect()
        })
        .unwrap_or_default()
}

/// How many skips this process has taken.
pub fn skip_count_in_process() -> usize {
    skipped_in_process().len()
}

/// Print a loud skip and record it.
///
/// Prefer [`require`], which cannot be used in the "should have run" case by
/// construction. This is the primitive underneath it.
pub fn skip(test: &str, requirement: Requirement, reason: impl Into<String>) {
    let reason = reason.into();
    println!(
        "SKIP: {test} — {} unmet ({}) — {}",
        requirement.slug(),
        reason,
        requirement.remedy()
    );
    if let Ok(mut l) = ledger().lock() {
        l.push(SkipRecord {
            test: test.to_owned(),
            requirement,
            reason,
        });
    }
}

/// The result of asking for a requirement: `None` means *run the test*.
pub type MaybeSkip = Option<SkipGuard>;

/// Ask whether `req` is satisfied.
///
/// * `Some(guard)` — the requirement is unmet. The skip has been printed and
///   recorded; the caller must `return` immediately and must not assert
///   anything else.
/// * `None` — the requirement is satisfied. The test **must** run. There is no
///   third outcome, and in particular there is no outcome in which a broken
///   engine, a corrupt file or a wrong answer becomes a skip.
pub fn require(test: &str, req: Requirement) -> MaybeSkip {
    let env = Environment::get();
    if env.satisfies(req) {
        None
    } else {
        let reason = env.reason_unmet(req);
        skip(test, req, reason.clone());
        Some(SkipGuard { req })
    }
}

/// Proof that a skip was legitimate.
///
/// Deliberately not `Drop`-based: a guard that fails on drop is a guard that
/// fires after the test has already reported success, which is the worst time
/// to report anything. It exists so the caller has something to hold and the
/// intent is visible in the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkipGuard {
    req: Requirement,
}

impl SkipGuard {
    /// The requirement that was unmet.
    pub fn requirement(&self) -> Requirement {
        self.req
    }
}

/// A one-line description of this machine, for the test output.
pub fn environment_summary() -> String {
    Environment::get().summary()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_requirement_has_a_distinct_slug_and_a_remedy() {
        let mut slugs: Vec<&str> = ALL_REQUIREMENTS.iter().map(|r| r.slug()).collect();
        slugs.sort_unstable();
        let count = slugs.len();
        slugs.dedup();
        assert_eq!(slugs.len(), count, "two requirements share a slug");
        for r in ALL_REQUIREMENTS {
            assert!(!r.remedy().is_empty(), "{} has no remedy", r.slug());
        }
    }

    #[test]
    fn the_probe_is_self_consistent() {
        let env = Environment::probe();
        if env.weights_complete() {
            assert!(
                env.weights_missing.is_empty(),
                "complete weights must list nothing missing"
            );
        } else {
            assert!(
                !env.weights_missing.is_empty(),
                "incomplete weights must name what is absent, or the skip reason is useless"
            );
        }
        assert!(env.summary().contains("weights="));
        for r in ALL_REQUIREMENTS {
            let reason = env.reason_unmet(r);
            assert!(!reason.is_empty(), "{} has an empty reason", r.slug());
        }
    }

    #[test]
    fn a_satisfied_requirement_never_skips() {
        // The real clip is absent in CI and present on the developer machine;
        // whichever it is, `require` must agree with `satisfies` exactly.
        //
        // The ledger is read by *filtering on this test's own name*, never by
        // counting. The first version compared `skip_count_in_process()` before
        // and after, which is a race: the ledger is process-wide, and a test on
        // another thread taking its own skip between the two reads makes the
        // count differ. It failed roughly one run in six, which is the worst
        // possible failure rate — a flake that looks like a policy bug.
        let env = Environment::get();
        let mine = |entries: &[(String, &'static str, String)]| {
            entries
                .iter()
                .filter(|(t, _, _)| t == "skip_policy_selftest")
                .count()
        };
        for r in ALL_REQUIREMENTS {
            let before = mine(&skipped_in_process());
            let guard = require("skip_policy_selftest", r);
            let after = mine(&skipped_in_process());
            if env.satisfies(r) {
                assert!(guard.is_none(), "{} is satisfied but skipped", r.slug());
                assert_eq!(after, before, "a satisfied requirement recorded a skip");
            } else {
                let g = guard.expect("an unmet requirement must skip");
                assert_eq!(g.requirement(), r);
                assert!(
                    after > before,
                    "{} is unmet but nothing was recorded, so the run would not say what did \
                     not happen",
                    r.slug()
                );
            }
        }
    }

    #[test]
    fn the_ledger_records_what_did_not_run() {
        let r = Requirement::PrivateClip;
        let env = Environment::get();
        skip("ledger_selftest", r, "deliberate");
        let entries = skipped_in_process();
        let mine = entries
            .iter()
            .filter(|(t, _, _)| t == "ledger_selftest")
            .count();
        assert_eq!(mine, 1, "the ledger must hold the skip it was told about");
        assert!(entries.iter().any(|(_, slug, _)| *slug == r.slug()));
        if env.satisfies(r) {
            assert!(mine == 1, "only the deliberate skip should be present");
        }
    }
}

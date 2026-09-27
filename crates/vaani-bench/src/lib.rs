//! A9 performance lab: staged benchmark harness for VaaniDesk.
//!
//! This crate **measures**; it never modifies product code. Each pipeline
//! stage gets an independent high-resolution timer, and the result is a
//! [`report::BenchReport`] that is either a measurement or an explicit
//! unknown marker. See `docs/agent/PERFORMANCE.md` for the rules every number
//! here obeys.
//!
//! # Current coverage
//!
//! The live end-to-end path is not wired yet: there is no `vaani-app`
//! pipeline to call (A2 owns it) and loading real weights is the model
//! manager's job (A7). So [`run_benchmark`] honestly reports `status: "skip"`
//! with the reason in `note` whenever a stage cannot run, and the binary
//! exits 0. A missing input file is the one hard error (exit 2), because that
//! is a caller mistake, not a missing dependency.
//!
//! # `vaani benchmark <file> [--json]` contract (backed here for A8)
//!
//! - `vaani-bench <file>` prints the human table to stdout.
//! - `vaani-bench <file> --json` prints one JSON [`report::BenchReport`] to
//!   stdout.
//! - Missing/unreadable `<file>` prints `vaani-bench: error: ...` to stderr
//!   and exits 2 (both modes).
//! - The private meeting fixture travels ONLY as the `<file>` argument (or
//!   `VAANI_BENCH_FILE`, resolved to an argument by `scripts/bench/`). It is
//!   `*.mp4`-gitignored and is never read beyond file metadata by this crate.

pub mod report;
pub mod sys;

use std::path::{Path, PathBuf};

pub use report::{BenchReport, StageTimer, measure};

/// Pipeline stages, in product order
/// (`docs/architecture/INTEGRATION_CONTRACT.md` §14).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Startup,
    Media,
    Vad,
    Asr,
    Diar,
    Post,
    Export,
}

impl Stage {
    /// All stages in run order.
    pub fn all() -> [Stage; 7] {
        [
            Stage::Startup,
            Stage::Media,
            Stage::Vad,
            Stage::Asr,
            Stage::Diar,
            Stage::Post,
            Stage::Export,
        ]
    }

    /// CLI/table label.
    pub fn as_str(&self) -> &'static str {
        match self {
            Stage::Startup => "startup",
            Stage::Media => "media",
            Stage::Vad => "vad",
            Stage::Asr => "asr",
            Stage::Diar => "diar",
            Stage::Post => "post",
            Stage::Export => "export",
        }
    }
}

/// Clean, caller-actionable benchmark failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchError {
    /// The input path does not exist.
    MissingInput { path: String },
    /// The input path exists but is not a file (or has no metadata).
    UnreadableInput { path: String, reason: String },
}

impl std::fmt::Display for BenchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BenchError::MissingInput { path } => {
                write!(f, "input file not found: {path}")
            }
            BenchError::UnreadableInput { path, reason } => {
                write!(f, "cannot read input file {path}: {reason}")
            }
        }
    }
}

impl std::error::Error for BenchError {}

/// What the environment probe found (filesystem checks only; nothing is
/// loaded, so probing is cheap and side-effect free).
#[derive(Debug, Clone, Default)]
struct EnvProbe {
    runtime_hint: Option<PathBuf>,
    model_hint: Option<PathBuf>,
}

impl EnvProbe {
    fn probe() -> Self {
        Self {
            runtime_hint: find_runtime_dir(),
            model_hint: find_model_file(),
        }
    }

    /// Reasons the full run cannot happen yet. Empty means "all present".
    fn missing_reasons(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.runtime_hint.is_none() {
            out.push(
                "no ONNX runtime (set VAANI_RUNTIME_DIR to the ORT directory, \
                 or place runtime/ next to the binary; see scripts/fetch_runtime.ps1)"
                    .to_string(),
            );
        }
        if self.model_hint.is_none() {
            out.push(
                "no model weights (set VAANI_MODELS_DIR / VAANI_ARTIFACTS_ROOT / \
                 VAANI_MODEL_DIRS, or provide artifacts/sravaani/encoder.fp16.onnx)"
                    .to_string(),
            );
        }
        // Even with runtime + weights present, the staged pipeline lives in
        // vaani-app (A2, wave 2), which this crate must not implement itself:
        // services never call each other except through the app hub.
        out.push(
            "live staged pipeline wiring pending vaani-app (A2); \
             vaani-bench records the harness skeleton honestly until then"
                .to_string(),
        );
        out
    }
}

/// Candidate directories for the ONNX Runtime, in precedence order.
fn runtime_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(dir) = std::env::var_os("VAANI_RUNTIME_DIR") {
        out.push(PathBuf::from(dir));
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        out.push(dir.join("runtime"));
        // Workspace layout: <root>/target/a9/{debug,release}/vaani-bench(.exe)
        // -> walk up to <root>/runtime.
        if let Some(root) = exe
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
        {
            out.push(root.join("runtime"));
        }
    }
    out.push(PathBuf::from("runtime"));
    out
}

fn find_runtime_dir() -> Option<PathBuf> {
    runtime_candidates().into_iter().find(|dir| {
        dir.is_dir()
            && std::fs::read_dir(dir).is_ok_and(|mut entries| {
                entries.any(|e| {
                    e.is_ok_and(|e| e.file_name().to_string_lossy().starts_with("onnxruntime"))
                })
            })
    })
}

/// Candidate model files, in precedence order.
fn model_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for var in ["VAANI_MODELS_DIR", "VAANI_ARTIFACTS_ROOT"] {
        if let Some(dir) = std::env::var_os(var) {
            let dir = PathBuf::from(dir);
            out.push(dir.join("encoder.fp16.onnx"));
            out.push(dir.join("sravaani").join("encoder.fp16.onnx"));
        }
    }
    if let Some(dirs) = std::env::var_os("VAANI_MODEL_DIRS") {
        for dir in std::env::split_paths(&dirs) {
            out.push(dir.join("encoder.fp16.onnx"));
        }
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        out.push(
            dir.join("artifacts")
                .join("sravaani")
                .join("encoder.fp16.onnx"),
        );
    }
    out.push(PathBuf::from("artifacts/sravaani/encoder.fp16.onnx"));
    out.push(
        PathBuf::from("artifacts")
            .join("sravaani")
            .join("encoder.fp16.onnx"),
    );
    out
}

fn find_model_file() -> Option<PathBuf> {
    model_candidates().into_iter().find(|p| p.is_file())
}

/// Run the staged benchmark over `input`.
///
/// - Returns `Err` only for caller mistakes (missing/unreadable input).
/// - Returns `Ok` with `status == "skip"` and a `note` when a stage cannot
///   run (missing runtime/weights/pipeline): a skip is not a failure.
/// - Stages are timed independently; `wall_ms` is measured separately and is
///   not the sum of the stages.
pub fn run_benchmark(input: &Path) -> Result<BenchReport, BenchError> {
    let wall = StageTimer::start();

    // --- input validation (caller mistakes are hard errors) ---
    // Existence is checked first so a missing path reports MissingInput
    // rather than a stat error. The metadata stat itself is the media-stage
    // measurement (decoding lives in vaani-media, A1).
    if !input.exists() {
        return Err(BenchError::MissingInput {
            path: input.display().to_string(),
        });
    }

    // --- startup: arg validation + env probe (measured, not assumed) ---
    let (env, startup_ms) = measure(EnvProbe::probe);

    // --- media stage: file metadata stat only. This crate must not
    // --- reimplement decoding; that is vaani-media's job. ---
    let (metadata, media_ms) = measure(|| std::fs::metadata(input));
    let metadata = metadata.map_err(|e| BenchError::UnreadableInput {
        path: input.display().to_string(),
        reason: e.to_string(),
    })?;
    if !metadata.is_file() {
        return Err(BenchError::UnreadableInput {
            path: input.display().to_string(),
            reason: "not a regular file".to_string(),
        });
    }
    let input_line = format!("{} ({} bytes)", input.display(), metadata.len());

    // --- remaining stages: skipped until the vaani-app (A2) pipeline exists
    // --- to call. Each keeps its own timer so the skeleton already times
    // --- stages independently; today each measures ~0 ms of harness work. ---
    let vad_ms = StageTimer::start().elapsed_ms();
    let asr_ms = StageTimer::start().elapsed_ms();
    let diar_ms = StageTimer::start().elapsed_ms();
    let post_ms = StageTimer::start().elapsed_ms();
    let export_ms = StageTimer::start().elapsed_ms();

    let reasons = env.missing_reasons();
    let note = format!("SKIP: {}", reasons.join("; "));
    let wall_ms = wall.elapsed_ms();

    Ok(BenchReport {
        input: input_line,
        status: "skip".to_string(),
        note: Some(note),
        startup_ms,
        media_ms,
        vad_ms,
        asr_ms,
        diar_ms,
        post_ms,
        export_ms,
        wall_ms,
        audio_duration_ms: 0,
        peak_rss_mb: sys::peak_rss_mb(),
        peak_vram_mb: sys::peak_vram_mb(),
        rtf: 0.0,
        segments: 0,
        speakers: 0,
        failures: 0,
        git_sha: sys::git_sha(),
        device: sys::device_label(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_order_matches_the_product_pipeline() {
        let names: Vec<_> = Stage::all().iter().map(Stage::as_str).collect();
        assert_eq!(
            names,
            ["startup", "media", "vad", "asr", "diar", "post", "export"]
        );
    }

    #[test]
    fn missing_file_is_a_clean_error() {
        let missing = Path::new("definitely-not-here-12345.mp4");
        assert!(!missing.exists());
        let err = run_benchmark(missing).expect_err("missing file must fail");
        let text = err.to_string();
        // Clean: names the path, no panic payload, no backtrace hint.
        assert!(text.contains("definitely-not-here-12345.mp4"), "{text}");
        assert!(text.contains("not found"), "{text}");
    }

    #[test]
    fn directory_input_is_a_clean_error() {
        let dir = std::env::temp_dir();
        let err = run_benchmark(&dir).expect_err("a directory must fail");
        assert!(matches!(err, BenchError::UnreadableInput { .. }));
        assert!(err.to_string().contains("not a regular file"));
    }

    #[test]
    fn present_file_without_pipeline_skips_gracefully() {
        let path =
            std::env::temp_dir().join(format!("vaani-bench-dry-run-{}.bin", std::process::id()));
        std::fs::write(&path, b"not real media").expect("fixture must be writable");
        let report = run_benchmark(&path).expect("present file must not error");
        std::fs::remove_file(&path).ok();
        assert_eq!(report.status, "skip");
        let note = report.note.as_ref().expect("skip must carry a reason");
        assert!(!note.is_empty());
        // Skip honesty: nothing ran, so nothing is claimed.
        assert_eq!(report.failures, 0, "skips are not failures");
        assert_eq!(report.segments, 0);
        assert_eq!(report.speakers, 0);
        assert_eq!(report.audio_duration_ms, 0);
        assert!(!report.rtf_is_valid());
        assert!(report.wall_ms >= report.startup_ms);
    }

    #[test]
    fn bench_error_display_is_one_clean_line() {
        let e = BenchError::MissingInput {
            path: "x.mp4".to_string(),
        };
        assert_eq!(e.to_string(), "input file not found: x.mp4");
    }
}

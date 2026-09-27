//! Staged benchmark report: the machine-readable contract.
//!
//! [`BenchReport`] is what `vaani benchmark <file> [--json]` emits. Every
//! field is either a measurement taken with a high-resolution timer / OS
//! probe, or an explicit unknown marker (`0` with `audio_duration_ms == 0`,
//! `None`, `"unknown"`). See `docs/agent/PERFORMANCE.md`: a number with no
//! provenance is worse than no number.

use std::time::Instant;

/// One row of the `vaani benchmark` contract.
///
/// Field names are the contract: `vaani-cli` (A8) parses `--json` output by
/// these names, so renaming a field is a breaking change. Extra fields
/// (`input`, `status`, `note`, `audio_duration_ms`) are provenance, not
/// scope creep: without them `rtf` cannot be reproduced.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BenchReport {
    /// Input path as passed on the command line (never read beyond metadata).
    pub input: String,
    /// `"ok"` when every stage ran, `"skip"` when one or more stages were
    /// skipped. Numbers under a `"skip"` report are harness overhead and
    /// zeros, never stage measurements; see `note`.
    pub status: String,
    /// Human reason for `status == "skip"`, or `None` on a full run.
    pub note: Option<String>,
    /// Harness startup: process entry to first stage probe. This is the
    /// *benchmark harness* startup, not application startup; the 1.5 s product
    /// budget cannot be validated from this crate.
    pub startup_ms: u64,
    /// Media probe/decode stage.
    pub media_ms: u64,
    /// Voice-activity-detection stage.
    pub vad_ms: u64,
    /// ASR stage (front end + encoder + decode loop).
    pub asr_ms: u64,
    /// Diarization stage.
    pub diar_ms: u64,
    /// Post-processing stage.
    pub post_ms: u64,
    /// Export stage.
    pub export_ms: u64,
    /// Wall time for the whole benchmarked run, measured independently of the
    /// stage timers (not their sum: it includes harness overhead between
    /// stages).
    pub wall_ms: u64,
    /// Source audio duration in milliseconds. `0` means unknown / not
    /// measured; `rtf` is meaningless in that case by construction.
    pub audio_duration_ms: u64,
    /// Peak process working set in MiB, via `GetProcessMemoryInfo`
    /// (`PeakWorkingSetSize`). `0.0` only when the OS probe itself failed;
    /// never estimated.
    pub peak_rss_mb: f64,
    /// Peak VRAM budget in MiB, via the DXGI budget of the *inference*
    /// adapter (NVIDIA + usable CUDA), or `None` when there is no such
    /// adapter. Deliberately NOT the largest adapter: on a hybrid laptop the
    /// largest budget is the iGPU's shared system memory, which CUDA cannot
    /// use (`crates/vaani-inference/src/capability.rs`, `inference_adapter`
    /// vs `preferred_adapter`).
    pub peak_vram_mb: Option<f64>,
    /// `wall_ms / audio_duration_ms`. Lower is faster; `< 1` is faster than
    /// realtime. Valid only when `status == "ok"` and
    /// `audio_duration_ms > 0`; otherwise `0.0` as an explicit unknown
    /// marker (a measured RTF of exactly zero is impossible).
    pub rtf: f64,
    /// Transcript segments produced (0 when skipped).
    pub segments: u64,
    /// Speakers distinguished (0 when skipped).
    pub speakers: u64,
    /// Stages that failed (skips are not failures).
    pub failures: u64,
    /// Short git SHA of the measured tree, or `"unknown"` when it could not
    /// be determined. Never fabricated.
    pub git_sha: String,
    /// `"cuda (<adapter>)"` when an NVIDIA adapter with a loadable CUDA
    /// runtime was measured, else `"cpu"`.
    pub device: String,
}

impl BenchReport {
    /// Realtime factor: `processing_ms / audio_ms`.
    ///
    /// Returns `0.0` (the explicit unknown marker) when `audio_ms == 0`,
    /// because dividing by an unknown duration would invent a number.
    /// [`Self::rtf_is_valid`] tells callers whether the value is meaningful.
    pub fn rtf(wall_ms: u64, audio_ms: u64) -> f64 {
        if audio_ms == 0 {
            return 0.0;
        }
        wall_ms as f64 / audio_ms as f64
    }

    /// Whether `self.rtf` is a measurement rather than the unknown marker.
    pub fn rtf_is_valid(&self) -> bool {
        self.status == "ok" && self.audio_duration_ms > 0
    }

    /// Serialize for `vaani benchmark <file> --json` (single line, stdout).
    pub fn to_json(&self) -> String {
        // `serde_json::to_string` on a struct with only serializable fields
        // cannot fail; on failure report the debug form rather than panic, so
        // a `--json` consumer always gets one line of output.
        serde_json::to_string(self).unwrap_or_else(|e| {
            format!("{{\"status\":\"error\",\"reason\":\"report serialization failed: {e}\"}}")
        })
    }

    /// Parse a report back (round-trip tested).
    pub fn from_json(text: &str) -> serde_json::Result<Self> {
        serde_json::from_str(text)
    }

    /// Human table for the non-`--json` path (stdout).
    pub fn render_table(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("vaani-bench  status={}\n", self.status));
        out.push_str(&format!("input   {}\n", self.input));
        out.push_str("stage        ms\n");
        for (name, ms) in [
            ("startup", self.startup_ms),
            ("media", self.media_ms),
            ("vad", self.vad_ms),
            ("asr", self.asr_ms),
            ("diar", self.diar_ms),
            ("post", self.post_ms),
            ("export", self.export_ms),
        ] {
            out.push_str(&format!("{name:<12}{ms:>6}\n"));
        }
        out.push_str(&format!("{:<12}{:>6}\n", "wall", self.wall_ms));
        out.push_str(&format!(
            "audio_ms  {}\n",
            if self.audio_duration_ms > 0 {
                self.audio_duration_ms.to_string()
            } else {
                "n/a (not measured)".to_string()
            }
        ));
        out.push_str(&format!(
            "rtf       {}\n",
            if self.rtf_is_valid() {
                format!("{:.3}", self.rtf)
            } else {
                "n/a (no audio duration measured)".to_string()
            }
        ));
        out.push_str(&format!("peak_rss_mb  {:.1}\n", self.peak_rss_mb));
        out.push_str(&format!(
            "peak_vram_mb {}\n",
            self.peak_vram_mb
                .map(|v| format!("{v:.1}"))
                .unwrap_or_else(|| "n/a (no NVIDIA inference adapter)".to_string())
        ));
        out.push_str(&format!("segments  {}\n", self.segments));
        out.push_str(&format!("speakers  {}\n", self.speakers));
        out.push_str(&format!("failures  {}\n", self.failures));
        out.push_str(&format!("git_sha   {}\n", self.git_sha));
        out.push_str(&format!("device    {}\n", self.device));
        if let Some(note) = &self.note {
            out.push_str(&format!("note      {note}\n"));
        }
        out
    }
}

/// A high-resolution stage timer.
///
/// Each pipeline stage gets its own timer so stages are timed independently:
/// the timestamps never share state, and `wall_ms` is measured separately
/// (see [`BenchReport::wall_ms`]).
#[derive(Debug, Clone)]
pub struct StageTimer {
    start: Instant,
}

impl StageTimer {
    /// Start timing now (`Instant::now`, nanosecond resolution where the OS
    /// provides it).
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Milliseconds elapsed since [`Self::start`].
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }
}

/// Time a closure, returning `(value, elapsed_ms)`.
pub fn measure<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let timer = StageTimer::start();
    let value = f();
    (value, timer.elapsed_ms())
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn sample_ok() -> BenchReport {
        BenchReport {
            input: "meeting.mp4".to_string(),
            status: "ok".to_string(),
            note: None,
            startup_ms: 120,
            media_ms: 800,
            vad_ms: 150,
            asr_ms: 36_000,
            diar_ms: 900,
            post_ms: 40,
            export_ms: 60,
            wall_ms: 38_200,
            audio_duration_ms: 720_000,
            peak_rss_mb: 812.4,
            peak_vram_mb: Some(3531.0),
            rtf: BenchReport::rtf(38_200, 720_000),
            segments: 412,
            speakers: 4,
            failures: 0,
            git_sha: "abc1234".to_string(),
            device: "cuda (NVIDIA GeForce RTX 3050 Ti Laptop GPU)".to_string(),
        }
    }

    #[test]
    fn report_json_round_trip() {
        let report = sample_ok();
        let text = report.to_json();
        // Single line: safe to stream on stdout alongside logs.
        assert!(!text.contains('\n'), "JSON report must be one line");
        let back = BenchReport::from_json(&text).expect("round-trip must parse");
        assert_eq!(report, back);
    }

    #[test]
    fn report_json_round_trip_with_unknowns() {
        let mut report = sample_ok();
        report.status = "skip".to_string();
        report.note = Some("SKIP: no runtime".to_string());
        report.peak_vram_mb = None;
        report.rtf = 0.0;
        report.audio_duration_ms = 0;
        let back = BenchReport::from_json(&report.to_json()).expect("round-trip must parse");
        assert_eq!(report, back);
        assert!(!back.rtf_is_valid());
    }

    #[test]
    fn rtf_math() {
        // PERFORMANCE.md definition: RTF = processing_time / audio_duration.
        assert!((BenchReport::rtf(2_000, 40_000) - 0.05).abs() < 1e-12);
        assert!((BenchReport::rtf(38_200, 720_000) - 38_200.0 / 720_000.0).abs() < 1e-12);
        // Faster than realtime is below one.
        assert!(BenchReport::rtf(500, 16_000) < 1.0);
    }

    #[test]
    fn rtf_with_unknown_audio_is_the_unknown_marker() {
        assert_eq!(BenchReport::rtf(1_000, 0), 0.0);
        assert_eq!(BenchReport::rtf(0, 0), 0.0);
    }

    #[test]
    fn stage_timer_measures_elapsed_time() {
        let timer = StageTimer::start();
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(timer.elapsed_ms() >= 5, "timer must not run short");
    }

    #[test]
    fn measure_returns_value_and_elapsed() {
        let (value, ms) = measure(|| 40 + 2);
        assert_eq!(value, 42);
        assert_eq!(ms, 0, "a trivial closure should cost no whole ms");
    }
}

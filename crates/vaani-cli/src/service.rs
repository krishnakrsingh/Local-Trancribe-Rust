//! `doctor`, `models`, `jobs`, and `export` handlers.
//!
//! These talk to the service crates (`vaani-inference`, `vaani-models`,
//! `vaani-storage`, `vaani-export`) directly. They are outside the vaani-app
//! seam: the swap changes who *runs the pipeline*, not who reports runtime
//! health or lists jobs.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use vaani_core::{ErrorCode, Result, VaaniError, err};

/// Everything `vaani doctor` prints, in one serialisable value.
///
/// `RuntimeReport::collect()` never fails by design (it must run on a broken
/// machine and say what is broken), so `doctor` likewise returns `Ok` unless
/// its own output cannot be serialised.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorReport {
    /// [`vaani_core::APP_VERSION`].
    pub app_version: String,
    /// One-line verdict (same string the settings header will show).
    pub summary: String,
    /// Inference runtime presence + missing DLLs + machine capabilities.
    pub runtime: vaani_inference::RuntimeReport,
    /// One entry per manifest model: installed or not, and the headline.
    pub models: Vec<ModelLine>,
    /// Data-model/schema versions a support bundle needs.
    pub data_model_version: u32,
    pub storage_schema_version: Option<u32>,
}

/// One line of `vaani models list` / the doctor's model section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLine {
    /// Manifest id, e.g. `sravaani-1.0-onnx`.
    pub id: String,
    /// Display label from the manifest.
    pub label: String,
    /// Whether the model can be used right now.
    pub available: bool,
    /// Human detail; the reason when unavailable. Never asserted on.
    pub detail: String,
}

/// Collect the doctor report. Never fails on a broken machine.
pub fn doctor() -> DoctorReport {
    let runtime = vaani_inference::RuntimeReport::collect();
    let summary = runtime.summary();
    let models = model_lines();
    let storage_schema_version = vaani_storage::Store::open_default()
        .ok()
        .and_then(|s| s.schema_version().ok());
    DoctorReport {
        app_version: vaani_core::APP_VERSION.to_string(),
        summary,
        runtime,
        models,
        data_model_version: vaani_core::DATA_MODEL_VERSION,
        storage_schema_version,
    }
}

/// `true` when the machine is ready to transcribe (runtime present and at
/// least one model available). Scripts use the exit code; humans read `summary`.
pub fn doctor_is_ready(report: &DoctorReport) -> bool {
    report.runtime.runtime_present && report.models.iter().any(|m| m.available)
}

fn model_manager() -> Result<vaani_models::ModelManager> {
    vaani_models::ModelManager::embedded()
        .map_err(|e| err::internal().with_cause(format!("model manifest is unreadable: {e}")))
}

/// One [`ModelLine`] per manifest choice, with install state.
pub fn model_lines() -> Vec<ModelLine> {
    let Ok(mgr) = model_manager() else {
        return Vec::new();
    };
    mgr.choices()
        .into_iter()
        .map(|c| ModelLine {
            id: c.id,
            label: c.label,
            available: c.available,
            detail: c.detail,
        })
        .collect()
}

/// `models list` output rows: `(id, label, available, detail)`.
pub fn models_list() -> Result<Vec<ModelLine>> {
    model_manager().map(|mgr| {
        mgr.choices()
            .into_iter()
            .map(|c| ModelLine {
                id: c.id,
                label: c.label,
                available: c.available,
                detail: c.detail,
            })
            .collect()
    })
}

/// `models ensure <id>`: resolve, or print the install plan and explain what is
/// missing.
///
/// Downloads need a Hugging Face token the CLI does not hold (ADR-005: the
/// caller owns the keyring/DPAPI handoff; `InstallOptions.token` is `Redacted`
/// and is never a CLI flag). So `ensure` on a missing model reports the plan
/// (bytes, fit) and returns `model_unavailable` / `auth_required` with a
/// recovery action, rather than attempting an anonymous fetch that the gate
/// would reject.
pub fn models_ensure(id: &str) -> Result<String> {
    let mgr = model_manager()?;
    if let Ok(loc) = mgr.resolve(id) {
        return Ok(format!("{} is installed at {}", id, loc.dir().display()));
    }
    let plan = mgr.install_plan(id)?;
    Err(if plan.already_on_disk {
        err::model_corrupt()
            .with_cause(format!("{id} is on disk but failed verification"))
            .with_recovery("Run `vaani models verify`, then re-download.")
    } else {
        err::model_unavailable()
            .with_cause(format!(
                "{id} needs {} to download",
                vaani_models::human_bytes(plan.download_bytes)
            ))
            .with_recovery(
                "Add your Hugging Face token in Settings, then download the model there.",
            )
    })
}

/// `models verify <id>`: checksum every file, report the diagnosis.
pub fn models_verify(id: &str, cancel: &vaani_core::CancelToken) -> Result<String> {
    let mgr = model_manager()?;
    let report = mgr.verify(id, cancel, &vaani_models::progress::ignore)?;
    let mut out = format!(
        "{}: {:?} ({} files, {} hashed)",
        report.model_id,
        report.diagnosis,
        report.files.len(),
        vaani_models::human_bytes(report.bytes_hashed)
    );
    if !report.failed.is_empty() {
        out.push_str(&format!("\nfailed: {}", report.failed.join(", ")));
    }
    Ok(out)
}

/// `models remove <id>`: delete from disk, report bytes freed.
pub fn models_remove(id: &str) -> Result<String> {
    let mgr = model_manager()?;
    let report = mgr.remove(id)?;
    if report.was_installed {
        Ok(format!(
            "removed {} ({} freed)",
            report.dir.display(),
            vaani_models::human_bytes(report.bytes_freed)
        ))
    } else {
        Err(err::model_unavailable().with_cause(format!("{id} is not installed")))
    }
}

/// One row of `vaani jobs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobLine {
    /// Stable job id.
    pub id: String,
    /// Source file name.
    pub input: String,
    /// Lifecycle phase wire string (`JobPhase::as_str`, SCREAMING_SNAKE).
    pub phase: String,
    /// Word/segment/speaker counts when a transcript is stored.
    pub words: i64,
    pub segments: i64,
    pub speakers: i64,
    /// Stored error code wire string, when failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
}

/// `jobs`: list the most recent jobs, newest first.
pub fn jobs_list(limit: u32) -> Result<Vec<JobLine>> {
    let store = vaani_storage::Store::open_default().map_err(|e: VaaniError| {
        // `Store::open_default` failures are storage failures with a stable code.
        if e.code() == ErrorCode::StorageFailed {
            e
        } else {
            err::storage_failed()
                .with_cause(format!("could not open the job database: {}", e.user.title))
        }
    })?;
    let page = store.list_jobs(&vaani_storage::JobFilter::default(), limit.max(1), 0)?;
    Ok(page
        .jobs
        .into_iter()
        .map(|j| JobLine {
            id: j.id,
            input: j.input_name,
            phase: j.phase.as_str().to_string(),
            words: j.word_count,
            segments: j.segment_count,
            speakers: j.speaker_count,
            error_code: j.error.map(|e| e.code),
        })
        .collect())
}

/// `export <transcript.json> --format <fmt|all> --out <path>`: read a
/// `vaani.transcript` JSON envelope and write the requested formats.
///
/// Returns the files written.
pub fn export_file(
    input: &Path,
    format: &str,
    out: Option<&Path>,
    overwrite: bool,
) -> Result<Vec<PathBuf>> {
    use vaani_export::{ExportFormat, ExportOptions, ExportTarget};
    let text = std::fs::read_to_string(input).map_err(|e| {
        err::export_failed()
            .with_cause(format!("could not read {}: {e}", input.display()))
            .with_recovery("Check the path and try again.")
    })?;
    let transcript = vaani_export::read_json_export(&text).map_err(|e: VaaniError| {
        err::export_failed()
            .with_cause(format!("not a VaaniDesk transcript: {}", e.user.title))
            .with_recovery("Export a transcript with `vaani transcribe --out` first.")
    })?;

    let options = ExportOptions {
        overwrite,
        source_name: input.file_stem().map(|s| s.to_string_lossy().into_owned()),
        ..ExportOptions::default()
    };

    // Default `--out` is the transcript's own directory, named from the stem.
    let out = out.map(PathBuf::from).unwrap_or_else(|| {
        input
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    });
    // A directory target names files from the stem; an exact-file target is
    // honoured verbatim (the extension the user typed wins).
    let target = if out.is_dir() || out.extension().is_none() {
        ExportTarget::Directory(out)
    } else {
        ExportTarget::File(out)
    };

    if format.eq_ignore_ascii_case("all") {
        let dir = match &target {
            ExportTarget::Directory(d) => d.clone(),
            ExportTarget::File(f) => f
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from(".")),
        };
        return Ok(
            vaani_export::export_all(&transcript, &ExportFormat::ALL, &dir, &options)?
                .into_iter()
                .map(|f| f.path)
                .collect(),
        );
    }
    let fmt = ExportFormat::parse(&format.to_ascii_lowercase())
        .ok_or_else(|| err::export_failed().with_cause(format!("unknown format: {format}")))?;
    Ok(vec![
        vaani_export::export_to_file(&transcript, fmt, &target, &options)?.path,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doctor_report_is_collectable_without_runtime_or_models() {
        // Must run on a broken machine and describe it, not fail.
        let r = doctor();
        assert!(!r.summary.is_empty());
        assert_eq!(r.app_version, vaani_core::APP_VERSION);
        assert_eq!(r.data_model_version, vaani_core::DATA_MODEL_VERSION);
        // JSON shape is a script interface: freeze the top-level keys.
        let v: serde_json::Value = serde_json::to_value(&r).unwrap();
        for key in ["app_version", "summary", "runtime", "models"] {
            assert!(v.get(key).is_some(), "doctor JSON missing `{key}`");
        }
    }

    #[test]
    fn export_rejects_non_transcript_json_by_code() {
        let dir = std::env::temp_dir().join("vaani-cli-test-export");
        std::fs::create_dir_all(&dir).unwrap();
        let bad = dir.join("bad.json");
        std::fs::write(&bad, r#"{"format":"nope"}"#).unwrap();
        let e = export_file(&bad, "srt", None, true).unwrap_err();
        assert_eq!(e.code(), ErrorCode::ExportFailed);
        std::fs::remove_file(&bad).ok();
    }

    #[test]
    fn export_rejects_missing_input_by_code() {
        let e = export_file(
            Path::new("definitely-not-here-12345.json"),
            "srt",
            None,
            true,
        )
        .unwrap_err();
        assert_eq!(e.code(), ErrorCode::ExportFailed);
    }
}

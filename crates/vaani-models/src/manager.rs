//! The façade the UI and the CLI both call.
//!
//! # One manager, two front ends
//!
//! A5's settings panel and A8's `vaani models …` command need the same
//! questions answered the same way: what exists, what is installed, what does it
//! cost to install, install it, verify it, remove it, and tell me how it is
//! getting on. If the UI and the CLI each answered those from their own code
//! they would disagree within a week, so they both call this.
//!
//! Nothing here needs a UI, a window handle, or a running pipeline. Progress
//! arrives as a callback, so a CLI prints a bar and a Slint panel calls
//! `invoke_from_event_loop`, and neither of them has to know about the other.
//!
//! # `Choice` is the UI's contract
//!
//! [`Choice`] is exactly `{ id, label, detail, available }`. `available: false`
//! means *known but unusable* and `detail` then carries the reason, because a
//! model the user cannot see is a model they cannot ask about. Hiding
//! `SraVaani 1.0 — not installed` behind an empty list is how a user concludes
//! the product is broken.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use vaani_core::CancelToken;
use vaani_inference::MachineCapabilities;

use crate::discover::{self, DiscoveredModel, Discovery, MANAGER_ENTRIES};
use crate::download::{Downloader, Redacted};
use crate::lock::{self, FileGuard, InstallLock, RECEIPT_FILE};
use crate::manifest::{Access, Manifest, ManifestError, Model, Provider, Task};
use crate::paths::{ModelRoots, RootKind};
use crate::progress::{ProgressFn, Stage, TransferProgress};
use crate::verify::{Diagnosis, Verifier, VerifyReport};

/// One row of the model picker.
///
/// Field names and types are the UI's contract; see the module docs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct Choice {
    /// The manifest id, stable across releases.
    pub id: String,
    /// What the picker shows.
    pub label: String,
    /// One line of context. When `available` is `false` this is the **reason**,
    /// not a summary.
    pub detail: String,
    /// Whether the model can actually be selected right now.
    pub available: bool,
}

/// Everything the settings panel shows about one model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct ModelStatus {
    pub id: String,
    /// The id in `vaani_asr::EngineRegistry` that consumes this model.
    pub engine_id: String,
    pub display_name: String,
    pub description: String,
    pub task: Task,
    pub version: String,
    pub architecture: String,
    /// SPDX identifier, e.g. `mit`. Shown next to the model, always.
    pub license_spdx: String,
    pub license_name: String,
    /// The licence text, as a URL.
    pub license_url: String,
    pub access: Access,
    pub source_provider: Provider,
    /// `org/name` for a Hub model, empty otherwise.
    pub source_repo: String,
    /// Where to read the model card, when the manifest names it.
    pub homepage: Option<String>,
    pub languages: Vec<String>,
    pub sample_rates: Vec<u32>,
    /// Every byte the manifest describes, optional files included.
    pub total_bytes: u64,
    /// Bytes of the files the runtime actually loads.
    pub weights_bytes: u64,
    /// Bytes measured on disk in the best copy found.
    pub on_disk_bytes: u64,
    /// Bytes an install would still fetch.
    pub download_bytes: u64,
    /// Whether every required file is present at the right size.
    pub installed: bool,
    /// Whether any copy exists at all, complete or not.
    pub present: bool,
    /// Required files that are absent or the wrong size.
    pub problems: Vec<String>,
    /// Files in the directory the manifest does not mention.
    pub unexpected: Vec<String>,
    /// Where the loader should read from, once installed.
    pub location: Option<PathBuf>,
    /// Which root that is.
    pub root_kind: Option<RootKind>,
    /// A `.partial/` exists, so a previous install was interrupted.
    pub interrupted: bool,
    pub min_vram_bytes: u64,
    pub rec_vram_bytes: u64,
    pub min_ram_bytes: u64,
    pub rec_ram_bytes: u64,
    /// Whether an install could fetch it at all.
    pub downloadable: bool,
    /// One human line. The same text as [`Choice::detail`].
    pub detail: String,
}

/// What an install would cost before the user commits to it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct InstallPlan {
    pub id: String,
    pub display_name: String,
    /// Every byte the manifest describes.
    pub total_bytes: u64,
    /// Bytes that would actually be fetched, optional reference files excluded.
    pub download_bytes: u64,
    /// Bytes already present at the right size.
    pub already_on_disk: bool,
    /// Free space on the target volume, when the platform can say.
    pub free_bytes: Option<u64>,
    /// `download_bytes` plus the downloader's reserve.
    pub required_free_bytes: u64,
    /// Whether it fits. `None` when the platform cannot say.
    pub fits_on_disk: Option<bool>,
    /// Whether the weights plus headroom fit in this machine's free VRAM.
    pub fits_in_vram: Option<bool>,
    /// The machine's CUDA compute capability, when it could be read.
    pub compute_capability: Option<(u8, u8)>,
    /// Whether that satisfies the model's minimum.
    pub compute_capability_ok: bool,
    /// The base URL downloads come from.
    pub source: String,
    /// Whether a token is needed.
    pub access: Access,
    /// Whether the model can be fetched at all.
    pub downloadable: bool,
    /// The directory an install would write to.
    pub target_dir: PathBuf,
}

/// Where a model lives, for the loader.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelLocation {
    /// The manifest id.
    pub id: String,
    /// The directory. Pass this to `vaani_asr::sravaani::artifacts_in`.
    pub dir: PathBuf,
    /// Which root it came from.
    pub root_kind: RootKind,
    /// Absolute paths of the required files, in manifest order.
    pub files: Vec<PathBuf>,
}

impl ModelLocation {
    /// The directory to hand the ASR loader.
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

/// What an install did.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct InstallReport {
    pub model_id: String,
    /// Where the model now is.
    pub dir: PathBuf,
    /// Files moved into place by this call.
    pub installed: Vec<String>,
    /// Files that were already present and correct.
    pub skipped: Vec<String>,
    /// Files that were resumed rather than fetched from zero.
    pub resumed: Vec<String>,
    /// Bytes actually transferred over the network.
    pub bytes_downloaded: u64,
    /// Wall time.
    pub elapsed: Duration,
    /// True when a complete copy was already present and nothing was fetched.
    pub already_installed: bool,
}

/// What a removal did.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct RemoveReport {
    pub model_id: String,
    /// The directory that was removed.
    pub dir: PathBuf,
    /// Bytes freed, measured before deleting.
    pub bytes_freed: u64,
    /// Whether the directory existed at all.
    pub was_installed: bool,
}

/// What is written to `.installed.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct InstallReceipt {
    /// Mirrors the manifest schema version, so a receipt from a future build is
    /// recognisable rather than merely wrong.
    pub manifest_schema_version: u32,
    pub model_id: String,
    pub model_version: String,
    pub installed_unix: u64,
    pub files: Vec<ReceiptFile>,
}

/// One line of a receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct ReceiptFile {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

/// Options for one install.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InstallOptions {
    /// Also fetch the manifest's optional files.
    ///
    /// `false` by default, and this is a product decision rather than a
    /// technicality: `SraVaani`'s required files are 930 MB and its optional
    /// reference encoder is another 1.65 GiB that no runtime path opens. Making
    /// the reference encoder opt-in is the difference between a first run that
    /// fits on a hotel connection and one that does not.
    pub include_optional: bool,
    /// A bearer token, when the source needs one.
    pub token: Option<Redacted>,
}

/// Owns the manifest, the roots, and the operations on them.
#[derive(Debug)]
pub struct ModelManager {
    manifest: Manifest,
    roots: ModelRoots,
    downloader: Option<Downloader>,
}

impl ModelManager {
    /// A manager over an explicit manifest and roots.
    #[must_use]
    pub fn new(manifest: Manifest, roots: ModelRoots) -> ModelManager {
        ModelManager {
            manifest,
            roots,
            downloader: None,
        }
    }

    /// A manager over the manifest compiled into this build, with the roots
    /// this machine would use.
    ///
    /// # Errors
    ///
    /// [`ManifestError`] if the embedded manifest is invalid, which is a bug in
    /// this repository rather than anything a user did.
    pub fn embedded() -> Result<ModelManager, ManifestError> {
        Ok(ModelManager::new(
            Manifest::embedded()?,
            ModelRoots::discover(),
        ))
    }

    /// A manager over a manifest file the user supplied.
    ///
    /// This is how a model is added without a new build.
    ///
    /// # Errors
    ///
    /// [`ManifestError::Io`] if the file cannot be read, or any validation error
    /// if it is not a manifest this build understands.
    pub fn from_manifest_file(path: &Path) -> Result<ModelManager, ManifestError> {
        let text = std::fs::read_to_string(path).map_err(|e| ManifestError::Io {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        let manifest = Manifest::parse(&text)?;
        Ok(ModelManager::new(manifest, ModelRoots::discover()))
    }

    /// Use a specific downloader, e.g. one with a simulated free-space probe.
    #[must_use]
    pub fn with_downloader(mut self, downloader: Downloader) -> ModelManager {
        self.downloader = Some(downloader);
        self
    }

    /// Add a configured search root.
    #[must_use]
    pub fn with_extra_root(mut self, dir: impl Into<PathBuf>) -> ModelManager {
        self.roots = self.roots.with_configured(dir);
        self
    }

    /// The manifest.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// The search roots.
    pub fn roots(&self) -> &ModelRoots {
        &self.roots
    }

    /// Find every model in every root. Cheap: names and sizes only, no hashing.
    #[must_use]
    pub fn discover(&self) -> Discovery {
        discover::discover(&self.manifest, &self.roots)
    }

    /// Everything the settings panel shows, one entry per manifest model, in
    /// manifest order.
    #[must_use]
    pub fn status(&self) -> Vec<ModelStatus> {
        let d = self.discover();
        self.manifest
            .models
            .iter()
            .map(|m| self.status_of(m, &d))
            .collect()
    }

    /// The picker's rows: exactly what A5's `Vec<Choice>` binds to.
    ///
    /// A model that is not usable is still present, with `available: false` and
    /// a `detail` that says why.
    #[must_use]
    pub fn choices(&self) -> Vec<Choice> {
        self.status()
            .into_iter()
            .map(|s| Choice {
                id: s.id,
                label: s.display_name,
                detail: s.detail,
                available: s.installed,
            })
            .collect()
    }

    /// Where the loader should read a model from.
    ///
    /// # Errors
    ///
    /// `ModelUnavailable` when the model is not installed, or an unknown-model
    /// error when the id is not in the manifest.
    pub fn resolve(&self, id: &str) -> vaani_core::Result<ModelLocation> {
        let model = self.require_model(id)?;
        let d = self.discover();
        let found = d.best(id).filter(|m| m.is_complete()).ok_or_else(|| {
            vaani_core::err::model_unavailable()
                .with_cause(format!("{id} is not installed in any root"))
                .with_recovery("Download it from Settings first.")
        })?;
        Ok(ModelLocation {
            id: model.id.clone(),
            dir: found.dir.clone(),
            root_kind: found.root_kind,
            files: model
                .required_paths()
                .into_iter()
                .filter_map(|p| model.file(p))
                .map(|f| found.dir.join(&f.relative))
                .collect(),
        })
    }

    /// The engine id in `vaani_asr::EngineRegistry` for a model.
    ///
    /// # Errors
    ///
    /// When the id is not in the manifest.
    pub fn engine_id(&self, id: &str) -> vaani_core::Result<&str> {
        Ok(self.require_model(id)?.engine_id.as_str())
    }

    /// Verify an installed model against the manifest.
    ///
    /// Streams every file, reports progress, and is cancellable mid-file.
    /// Returns `Err(Cancelled)` if the caller cancelled; every other problem is
    /// a [`VerifyReport`] with a diagnosis, because the UI has to say *which*
    /// file.
    ///
    /// # Errors
    ///
    /// Unknown model, not installed, or cancelled.
    pub fn verify(
        &self,
        id: &str,
        cancel: &CancelToken,
        progress: ProgressFn<'_>,
    ) -> vaani_core::Result<VerifyReport> {
        let loc = self.resolve_installed(id)?;
        let model = self.require_model(id)?;
        let report = Verifier::new().verify(model, loc.dir(), cancel, progress);
        if report.cancelled {
            return Err(vaani_core::err::cancelled());
        }
        Ok(report)
    }

    /// Hold every required file of a model open for reading, which on Windows
    /// makes them undeletable until the guards are dropped.
    ///
    /// This is the session side of "a model file being read by an inference
    /// session must not be deletable": a concurrent `remove` gets a clear
    /// "in use" error instead of unlinking a file out from under a live session.
    ///
    /// # Errors
    ///
    /// Unknown model, not installed, or a file that cannot be opened.
    pub fn guard_files(&self, id: &str) -> vaani_core::Result<Vec<FileGuard>> {
        let loc = self.resolve_installed(id)?;
        let mut out = Vec::new();
        for f in &loc.files {
            out.push(FileGuard::open(f).map_err(|e| {
                vaani_core::err::model_load_failed()
                    .with_cause(format!("could not open {}: {e}", f.display()))
            })?);
        }
        Ok(out)
    }

    /// What an install would cost, and whether it can happen.
    ///
    /// VRAM is not consulted: the caller normally already holds a
    /// [`MachineCapabilities`] snapshot, and building one here would mean
    /// loading the CUDA provider from the settings panel. Use
    /// [`ModelManager::install_plan_with`] for the full answer.
    ///
    /// # Errors
    ///
    /// Unknown model.
    pub fn install_plan(&self, id: &str) -> vaani_core::Result<InstallPlan> {
        self.install_plan_with(id, None, None)
    }

    /// [`ModelManager::install_plan`] against a known machine, and against a
    /// free-space figure the caller has already measured.
    ///
    /// # Errors
    ///
    /// Unknown model.
    pub fn install_plan_with(
        &self,
        id: &str,
        free_bytes: Option<u64>,
        caps: Option<&MachineCapabilities>,
    ) -> vaani_core::Result<InstallPlan> {
        let model = self.require_model(id)?;
        let d = self.discover();
        let found = d.best(id);
        let on_disk = found.is_some_and(DiscoveredModel::is_complete);
        let download_bytes: u64 = model
            .required_files()
            .map(|f| {
                found
                    .and_then(|x| x.files.iter().find(|s| s.path == f.path))
                    .map_or(f.bytes, |s| s.bytes_outstanding())
            })
            .sum();
        let target_dir = self.roots.install_target().join(&model.dir);
        let downloader = self.downloader.clone_or_new();
        let required_free = download_bytes.saturating_add(downloader.policy().reserve_bytes);
        let free = free_bytes.or_else(|| crate::sys::free_bytes(self.roots.install_target()));
        Ok(InstallPlan {
            id: model.id.clone(),
            display_name: model.display_name.clone(),
            total_bytes: model.size_bytes,
            download_bytes,
            already_on_disk: on_disk,
            free_bytes: free,
            required_free_bytes: required_free,
            fits_on_disk: free.map(|f| f >= required_free),
            fits_in_vram: caps
                .map(|c| c.vram_fits(model.weights_bytes, model.gpu.vram_headroom_bytes as f64)),
            compute_capability: caps.and_then(|c| c.cuda.compute_capability),
            compute_capability_ok: model
                .compute_capability_ok(caps.and_then(|c| c.cuda.compute_capability)),
            source: model.source.url_template.clone(),
            access: model.access,
            downloadable: model.source.provider.is_downloadable(),
            target_dir,
        })
    }

    /// Fetch everything a model is missing, verify each file, and move it into
    /// place.
    ///
    /// Ordering, which is the whole design:
    ///
    /// 1. take the install lock, or report that another window has it;
    /// 2. refuse up front if the volume cannot hold the model;
    /// 3. for each file, download to `.partial/`, hash it, and only then
    ///    `fs::rename` it to its real name;
    /// 4. write `.installed.json` last.
    ///
    /// A cancelled or failed install therefore leaves only `.partial/` files and
    /// whatever was already complete. The loader never sees a half-written
    /// weights file, and the next attempt resumes rather than restarting.
    ///
    /// # Errors
    ///
    /// Any [`DownloadError`] projected into the application taxonomy, plus
    /// unknown-model and not-installed-as-usable.
    pub async fn install(
        &self,
        id: &str,
        options: &InstallOptions,
        cancel: &CancelToken,
        progress: ProgressFn<'_>,
    ) -> vaani_core::Result<InstallReport> {
        let model = self.require_model(id)?.clone();
        let start = Instant::now();
        let d = self.discover();

        // A complete copy anywhere wins, and nothing is fetched. This is what
        // makes `install` idempotent for the bundled SraVaani model.
        if let Some(found) = d.best(id).filter(|m| m.is_complete()) {
            return Ok(InstallReport {
                model_id: model.id.clone(),
                dir: found.dir.clone(),
                installed: Vec::new(),
                skipped: model
                    .required_paths()
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect(),
                resumed: Vec::new(),
                bytes_downloaded: 0,
                elapsed: start.elapsed(),
                already_installed: true,
            });
        }

        let target_dir = self.roots.install_target().join(&model.dir);
        std::fs::create_dir_all(target_dir.join(lock::PARTIAL_DIR)).map_err(|e| {
            vaani_core::err::storage_failed()
                .with_cause(format!("could not create {}: {e}", target_dir.display()))
        })?;
        // The lock lives under the model *root*, not inside `target_dir`, so
        // `remove` can delete the model directory while holding the lock.
        let _lock = InstallLock::acquire(self.roots.install_target(), &model.dir, id)
            .map_err(|e| e.to_vaani_error())?;

        let wanted: Vec<&crate::manifest::ModelFile> = model
            .files
            .iter()
            .filter(|f| f.required || options.include_optional)
            .collect();
        let outstanding: u64 = wanted
            .iter()
            .map(|f| {
                let p = target_dir.join(&f.relative);
                match std::fs::metadata(&p) {
                    Ok(m) if m.len() == f.bytes => 0,
                    _ => f.bytes,
                }
            })
            .sum();

        let mut downloader = self.downloader.clone_or_new();
        if let Some(token) = &options.token {
            downloader = downloader.with_bearer(token.clone());
        }
        downloader
            .precheck(&target_dir, outstanding)
            .map_err(|e| e.to_vaani_error())?;

        let mut report = InstallReport {
            model_id: model.id.clone(),
            dir: target_dir.clone(),
            installed: Vec::new(),
            skipped: Vec::new(),
            resumed: Vec::new(),
            bytes_downloaded: 0,
            elapsed: start.elapsed(),
            already_installed: false,
        };
        let total_bytes: u64 = wanted.iter().map(|f| f.bytes).sum();
        let mut done_bytes = 0u64;

        for f in &wanted {
            cancel.check()?;
            let final_path = target_dir.join(&f.relative);
            if std::fs::metadata(&final_path).is_ok_and(|m| m.len() == f.bytes) {
                report.skipped.push(f.path.clone());
                done_bytes += f.bytes;
                continue;
            }
            let Some(url) = model.source.url_for(&f.path) else {
                return Err(vaani_core::err::model_unavailable()
                    .with_cause(format!("{id} has no download URL for {}", f.path))
                    .with_recovery("Reinstall VaaniDesk to restore the files that ship with it."));
            };
            let part = Downloader::part_path(&target_dir, &f.path);
            let before = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
            let from = done_bytes;
            let dl = downloader
                .download_file(url.as_str(), &part, f.bytes, &f.sha256, cancel, &|p| {
                    let mut p = p;
                    p.model_id = id.to_string();
                    p.file = Some(f.path.clone());
                    p.bytes_done = from + p.bytes_done.min(f.bytes);
                    p.bytes_total = total_bytes;
                    progress(p);
                })
                .await
                .map_err(|e| e.to_vaani_error())?;

            if let Some(parent) = final_path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    vaani_core::err::storage_failed()
                        .with_cause(format!("could not create {}: {e}", parent.display()))
                })?;
            }
            progress(TransferProgress {
                stage: Stage::Installing,
                model_id: id.to_string(),
                file: Some(f.path.clone()),
                files_done: report.installed.len() + report.skipped.len(),
                files_total: wanted.len(),
                bytes_done: done_bytes + f.bytes,
                bytes_total: total_bytes,
            });
            // The atomic step. Everything before this was in `.partial/`.
            std::fs::rename(&dl.part, &final_path).map_err(|e| {
                vaani_core::err::storage_failed().with_cause(format!(
                    "could not move {} into place: {e}",
                    dl.part.display()
                ))
            })?;
            if dl.resumed_from > 0 {
                report.resumed.push(f.path.clone());
            }
            report.installed.push(f.path.clone());
            report.bytes_downloaded += dl.bytes.saturating_sub(before.min(dl.bytes));
            done_bytes += f.bytes;
        }

        self.write_receipt(&model, &target_dir)?;
        report.elapsed = start.elapsed();
        Ok(report)
    }

    /// [`ModelManager::install`] for a caller with no async runtime.
    ///
    /// # Panics
    ///
    /// If called from inside a Tokio runtime, which is where a nested runtime
    /// is not merely wasteful but a panic. Call [`ModelManager::install`] there.
    ///
    /// # Errors
    ///
    /// As [`ModelManager::install`].
    pub fn install_blocking(
        &self,
        id: &str,
        options: &InstallOptions,
        cancel: &CancelToken,
        progress: ProgressFn<'_>,
    ) -> vaani_core::Result<InstallReport> {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                vaani_core::err::internal().with_cause(format!("could not start a runtime: {e}"))
            })?
            .block_on(self.install(id, options, cancel, progress))
    }

    /// Delete a model from disk.
    ///
    /// Takes the install lock first, and refuses **before deleting anything** if
    /// a required file is open elsewhere, so a refusal never leaves a model half
    /// removed.
    ///
    /// # Errors
    ///
    /// Unknown model, not installed, or a file in use.
    pub fn remove(&self, id: &str) -> vaani_core::Result<RemoveReport> {
        let model = self.require_model(id)?;
        let d = self.discover();
        let Some(found) = d.best(id) else {
            return Err(vaani_core::err::model_unavailable()
                .with_cause(format!("{id} is not installed in any root"))
                .with_recovery("There is nothing to remove."));
        };
        // A bundled or configured copy is not ours to delete: the exporter owns
        // `artifacts/` and a user pointed at the configured root themselves.
        if found.root_kind != RootKind::User {
            return Err(vaani_core::err::storage_failed()
                .with_cause(format!(
                    "{id} lives in the {} root, which VaaniDesk does not own",
                    found.root_kind
                ))
                .with_recovery("Delete the folder yourself if you want it gone."));
        }

        let _lock = InstallLock::acquire(&found.root, &found.dir_name, id)
            .map_err(|e| e.to_vaani_error())?;
        for f in model.files.iter().filter(|f| f.required) {
            let p = found.dir.join(&f.relative);
            if lock::is_in_use(&p) {
                return Err(
                    vaani_core::VaaniError::new(vaani_core::UserFacingError::new(
                        vaani_core::ErrorCode::StorageFailed,
                        "The model is in use.",
                        "Another VaaniDesk window or process is reading the model, so it \
                         cannot be deleted right now.",
                    ))
                    .with_cause(format!("{} is open elsewhere", p.display()))
                    .with_recovery(
                        "Close other VaaniDesk windows, wait for any transcription to finish, \
                     then try again.",
                    ),
                );
            }
        }

        let bytes_freed = found.on_disk_bytes;
        let removed = std::fs::remove_dir_all(&found.dir);
        match removed {
            Ok(()) => Ok(RemoveReport {
                model_id: id.to_string(),
                dir: found.dir.clone(),
                bytes_freed,
                was_installed: true,
            }),
            Err(e) if crate::sys::is_sharing_violation(&e) => Err(vaani_core::VaaniError::new(
                vaani_core::UserFacingError::new(
                    vaani_core::ErrorCode::StorageFailed,
                    "The model is in use.",
                    "Another VaaniDesk window or process is reading the model, so it \
                         cannot be deleted right now.",
                ),
            )
            .with_cause(e)
            .with_recovery("Close other VaaniDesk windows, then try again.")),
            Err(e) => Err(vaani_core::err::storage_failed()
                .with_cause(format!("could not remove {}: {e}", found.dir.display()))),
        }
    }

    /// Status for one model against an existing discovery.
    fn status_of(&self, model: &Model, d: &Discovery) -> ModelStatus {
        let found = d.best(&model.id);
        let present = found.is_some();
        let installed = found.is_some_and(DiscoveredModel::is_complete);
        let problems: Vec<String> = found
            .map(|f| f.problems().iter().map(|s| s.path.clone()).collect())
            .unwrap_or_default();
        let download_bytes = found.map_or(model.weights_bytes, DiscoveredModel::bytes_outstanding);
        let detail = describe(model, found, installed, present, &problems);
        ModelStatus {
            id: model.id.clone(),
            engine_id: model.engine_id.clone(),
            display_name: model.display_name.clone(),
            description: model.description.clone(),
            task: model.task,
            version: model.version.clone(),
            architecture: model.architecture.clone(),
            license_spdx: model.license.spdx.clone(),
            license_name: model.license.name.clone(),
            license_url: model.license.url.to_string(),
            access: model.access,
            source_provider: model.source.provider,
            source_repo: model.source.repo.clone(),
            homepage: model.source.homepage.as_ref().map(ToString::to_string),
            languages: model.languages.clone(),
            sample_rates: model.sample_rates.clone(),
            total_bytes: model.size_bytes,
            weights_bytes: model.weights_bytes,
            on_disk_bytes: found.map_or(0, |f| f.on_disk_bytes),
            download_bytes,
            installed,
            present,
            problems,
            unexpected: found.map(|f| f.unexpected.clone()).unwrap_or_default(),
            location: found.map(|f| f.dir.clone()),
            root_kind: found.map(|f| f.root_kind),
            interrupted: found.is_some_and(DiscoveredModel::is_incomplete_install),
            min_vram_bytes: model.resources.min_vram_bytes,
            rec_vram_bytes: model.resources.rec_vram_bytes,
            min_ram_bytes: model.resources.min_ram_bytes,
            rec_ram_bytes: model.resources.rec_ram_bytes,
            downloadable: model.source.provider.is_downloadable(),
            detail,
        }
    }

    /// Resolve a model that is installed, for verify/remove/guard.
    fn resolve_installed(&self, id: &str) -> vaani_core::Result<ModelLocation> {
        let model = self.require_model(id)?;
        let d = self.discover();
        let found = d.best(id).filter(|m| m.is_complete()).ok_or_else(|| {
            let problems: Vec<String> = d
                .best(id)
                .map(|f| f.problems().iter().map(|s| s.path.clone()).collect())
                .unwrap_or_default();
            vaani_core::err::model_unavailable()
                .with_cause(format!("{id} is not fully installed: {problems:?}"))
                .with_recovery("Download it from Settings, or run the model check.")
        })?;
        Ok(ModelLocation {
            id: model.id.clone(),
            dir: found.dir.clone(),
            root_kind: found.root_kind,
            files: model
                .required_paths()
                .into_iter()
                .filter_map(|p| model.file(p))
                .map(|f| found.dir.join(&f.relative))
                .collect(),
        })
    }

    fn require_model(&self, id: &str) -> vaani_core::Result<&Model> {
        self.manifest.model(id).ok_or_else(|| {
            // `model_unavailable`, not `internal`. The id here is whatever the
            // user typed into `--model` or the settings picker, so an id that is
            // not in the manifest is something *they* did, not a bug in this
            // program -- and `internal` tells a user nothing and asks them to
            // file a bug. The first end-to-end run on a real meeting found this
            // by running `transcribe --model <a typo>`: the CLI printed
            // `{"code":"internal", ...}`. `ErrorCode` has no `UnknownModel`
            // variant (see OPEN ISSUES: it should), and `ModelUnavailable` is
            // the closest honest existing code; the title ("Model
            // unavailable.") and the recovery both read correctly for a name
            // that is not in the picker.
            vaani_core::err::model_unavailable()
                .with_cause(format!(
                    "unknown model {id:?}; known: {:?}",
                    self.manifest.ids()
                ))
                .with_recovery("Run `vaani models list` to see the available models.")
        })
    }

    /// Write `.installed.json` last, through a temp file and a rename, so its
    /// presence always means the model is complete.
    fn write_receipt(&self, model: &Model, dir: &Path) -> vaani_core::Result<()> {
        let receipt = InstallReceipt {
            manifest_schema_version: crate::manifest::SCHEMA_VERSION,
            model_id: model.id.clone(),
            model_version: model.version.clone(),
            installed_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            files: model
                .files
                .iter()
                .map(|f| ReceiptFile {
                    path: f.path.clone(),
                    bytes: f.bytes,
                    sha256: f.sha256_hex(),
                })
                .collect(),
        };
        let body = serde_json::to_vec_pretty(&receipt).map_err(|e| {
            vaani_core::err::internal().with_cause(format!("could not encode the receipt: {e}"))
        })?;
        let tmp = dir.join(format!("{RECEIPT_FILE}.{}.tmp", std::process::id()));
        std::fs::write(&tmp, &body).map_err(|e| {
            vaani_core::err::storage_failed()
                .with_cause(format!("could not write {}: {e}", tmp.display()))
        })?;
        std::fs::rename(&tmp, dir.join(RECEIPT_FILE)).map_err(|e| {
            vaani_core::err::storage_failed()
                .with_cause(format!("could not move the receipt into place: {e}"))
        })
    }
}

/// A `Downloader` is cheap to rebuild but not free (it owns a connection pool),
/// so a manager that has none makes one on demand rather than at construction.
trait DownloaderExt {
    fn clone_or_new(&self) -> Downloader;
}

impl DownloaderExt for Option<Downloader> {
    fn clone_or_new(&self) -> Downloader {
        match self {
            Some(d) => d.clone(),
            None => Downloader::new(),
        }
    }
}

/// The one line a picker row shows.
///
/// When a model cannot be used this is the **reason**, because a row with
/// `available: false` and no explanation is a support ticket.
fn describe(
    model: &Model,
    found: Option<&DiscoveredModel>,
    installed: bool,
    present: bool,
    problems: &[String],
) -> String {
    let licence = model.license.spdx.to_uppercase();
    let mut parts: Vec<String> = Vec::new();
    if installed {
        parts.push("Installed".to_string());
        parts.push(format!(
            "{} on disk",
            human_bytes(found.map_or(0, |f| f.on_disk_bytes))
        ));
    } else if present {
        let have = found.map_or(0, |f| f.on_disk_bytes);
        parts.push("Not ready".to_string());
        if found.is_some_and(|f| f.has_partial) {
            parts.push("an install was interrupted".to_string());
        }
        parts.push(format!(
            "{} of {} present",
            human_bytes(have),
            human_bytes(model.weights_bytes)
        ));
        if !problems.is_empty() {
            parts.push(format!("{} file(s) missing or damaged", problems.len()));
        }
    } else {
        parts.push("Not installed".to_string());
        parts.push(format!("{} to download", human_bytes(model.weights_bytes)));
    }
    parts.push(licence);
    if model.resources.min_vram_bytes > 0 {
        parts.push(format!(
            "needs {} VRAM",
            human_bytes(model.resources.min_vram_bytes)
        ));
    } else {
        parts.push("runs on CPU".to_string());
    }
    if model.access.needs_token() {
        parts.push("needs a Hugging Face token".to_string());
    }
    parts.join(" · ")
}

/// A human size, binary units, because that is what Windows reports and what a
/// user will compare against Explorer.
#[must_use]
pub fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} B")
    } else {
        format!("{v:.1} {}", UNITS[u])
    }
}

/// Verify a receipt against a manifest, for a `--check-receipt` style command.
///
/// # Errors
///
/// [`Diagnosis::Missing`] when the receipt names a file the manifest does not
/// have, or a hash mismatch.
pub fn check_receipt(receipt: &InstallReceipt, model: &Model) -> Diagnosis {
    for f in &receipt.files {
        match model.file(&f.path) {
            None => return Diagnosis::Missing,
            Some(m) if m.sha256_hex() != f.sha256 => return Diagnosis::ContentDiffers,
            Some(_) => {}
        }
    }
    Diagnosis::Good
}

/// The names this crate owns inside a model directory, re-exported so a caller
/// can exclude them from its own listings.
pub const MANAGER_FILE_NAMES: &[&str] = MANAGER_ENTRIES;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_sizes_use_the_units_windows_reports() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(999), "999 B");
        assert_eq!(human_bytes(1024), "1.0 KiB");
        assert_eq!(human_bytes(1_610_612_736), "1.5 GiB");
        assert_eq!(human_bytes(2_702_792_388), "2.5 GiB");
    }

    #[test]
    fn a_choice_is_exactly_the_four_fields_the_ui_binds() {
        // A structural guard: if a field is added, A5's Slint binding is what
        // breaks, so this test is the thing that notices first.
        let c = Choice {
            id: "m".into(),
            label: "M".into(),
            detail: "D".into(),
            available: true,
        };
        let json = serde_json::to_value(&c).unwrap();
        let keys: Vec<&String> = json.as_object().unwrap().keys().collect();
        assert_eq!(keys, vec!["available", "detail", "id", "label"]);
    }

    #[test]
    fn a_receipt_check_distinguishes_a_stale_manifest_from_a_good_one() {
        let m = Manifest::embedded().unwrap();
        let model = m.model("sravaani-1.0-onnx").unwrap();
        let good = InstallReceipt {
            manifest_schema_version: crate::manifest::SCHEMA_VERSION,
            model_id: model.id.clone(),
            model_version: model.version.clone(),
            installed_unix: 0,
            files: model
                .files
                .iter()
                .map(|f| ReceiptFile {
                    path: f.path.clone(),
                    bytes: f.bytes,
                    sha256: f.sha256_hex(),
                })
                .collect(),
        };
        assert_eq!(check_receipt(&good, model), Diagnosis::Good);

        let mut tampered = good.clone();
        tampered.files[0].sha256 = "0".repeat(64);
        assert_eq!(check_receipt(&tampered, model), Diagnosis::ContentDiffers);

        let mut extra = good;
        extra.files.push(ReceiptFile {
            path: "not-in-the-manifest.bin".into(),
            bytes: 1,
            sha256: "0".repeat(64),
        });
        assert_eq!(check_receipt(&extra, model), Diagnosis::Missing);
    }

    #[test]
    fn an_absent_model_is_shown_with_its_reason_not_hidden() {
        let m = Manifest::embedded().unwrap();
        let model = m.model("sravaani-1.0-onnx").unwrap();
        let d = describe(model, None, false, false, &[]);
        assert!(d.starts_with("Not installed"), "{d}");
        // 929,977,857 bytes of required files. The fp32 reference encoder is
        // excluded, which is the difference between 886.9 MiB and 2.5 GiB.
        assert_eq!(model.weights_bytes, 929_977_857);
        assert!(d.contains("886.9 MiB"), "{d}");
        assert!(d.contains("MIT"), "{d}");
        assert!(d.contains("Hugging Face token"), "{d}");
        assert!(d.contains("VRAM"), "{d}");
    }

    #[test]
    fn a_half_installed_model_says_so() {
        let m = Manifest::embedded().unwrap();
        let model = m.model("sravaani-1.0-onnx").unwrap();
        let d = Discovery {
            roots: Vec::new(),
            models: vec![DiscoveredModel {
                id: model.id.clone(),
                dir_name: model.dir.clone(),
                root_kind: RootKind::User,
                root: PathBuf::from("r"),
                dir: PathBuf::from("r/sravaani"),
                files: Vec::new(),
                unexpected: Vec::new(),
                has_partial: true,
                on_disk_bytes: 100,
            }],
            absent: Vec::new(),
            unknown_dirs: Vec::new(),
            elapsed: Duration::ZERO,
        };
        let found = d.best(&model.id);
        let problems = vec!["encoder.fp16.onnx".to_string()];
        let text = describe(model, found, false, true, &problems);
        assert!(text.starts_with("Not ready"), "{text}");
        assert!(text.contains("interrupted"), "{text}");
        assert!(text.contains("missing or damaged"), "{text}");
    }
}

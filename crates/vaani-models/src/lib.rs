//! Model discovery, download, verification and manifest.
//!
//! Owned by agent **A7**. The `Cargo.toml` in this directory is frozen by
//! the architect and is part of the integration surface; the contents of this
//! file are yours.
//!
//! If you need a dependency that is not in the manifest, do not edit the
//! manifest. Report it under `INTEGRATION NOTES` and the architect will add it
//! in one commit that every agent rebases onto.
//!
//! Read `docs/agent/PARALLEL_OWNERSHIP.md` before the first edit.
//!
//! ---
//!
//! # What this crate is for
//!
//! The product must be able to say, honestly and before the user waits, five
//! things about any model on the machine: **is it here, is it intact, how big
//! is it, where did it come from, and what may I do with it**. Those five
//! questions are the whole surface, and they are spread across five modules
//! because each has a different failure mode and a different cost:
//!
//! | module | owns | cost | must not |
//! |---|---|---|---|
//! | [`manifest`] | what a model *is*: files, hashes, licence, access | parsed once, ~50 µs | guess anything a field states |
//! | [`paths`] | where models live | pure | write outside a model directory |
//! | [`discover`] | what is *on disk* | one `read_dir` per model dir, no hashing | read a byte of a weights file |
//! | [`verify`] | whether the bytes are *right* | streams at 1 MiB, cancellable | buffer a whole file |
//! | [`download`] | getting the bytes here | resumable, verified, atomic | leave a partial file under a real name |
//! | [`lock`] | one installer at a time | one `create_new` | rely on a timestamp to decide staleness |
//! | [`manager`] | the façade the UI and CLI call | — | need either of them |
//!
//! # The four properties that are not obvious
//!
//! **Discovery never hashes.** It runs at startup and every time the settings
//! panel opens. Hashing 2.5 GB takes about six seconds; listing it takes
//! microseconds, because the only I/O is `read_dir` and `metadata`. Hashing is
//! [`verify`]'s job, and the user asks for it.
//!
//! **Verification streams.** The encoder is 845 MiB and a desktop app may already
//! be holding a CUDA context. Verification reads a fixed 1 MiB buffer; the test
//! suite measures peak working set so this stays a measurement.
//!
//! **A hash mismatch is a corrupt download, not a broken app.** The recovery is
//! "download it again", and the manifest hashes are what make that a safe thing
//! to say. `artefacts.json` in `artifacts/sravaani/` is *stale* for
//! `encoder.fp16.onnx`, which is exactly why [`manifest`] carries hashes
//! measured from the files that are actually on disk.
//!
//! **Installing is atomic and mutually exclusive.** `.partial/<name>` is written
//! and hashed, then `fs::rename`d into place; `.installed.json` is written last.
//! A `<model dir>/.vaani-install.lock` file created with `create_new` and held
//! with `share_mode = 0` makes two concurrent installs impossible, and lets the
//! OS — not a pid heuristic — decide whether a leftover lock is stale.
//!
//! # Entry points
//!
//! ```no_run
//! use vaani_models::{CancelToken, ModelManager, progress::ignore};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let manager = ModelManager::embedded()?;
//!
//! // What the settings panel shows. No UI, no runtime, no I/O beyond stat.
//! for choice in manager.choices() {
//!     println!("{} [{}] {}", choice.label, choice.id, choice.detail);
//! }
//!
//! // Where the ASR loader should read from. `dir()` is exactly what
//! // `vaani_asr::sravaani::artifacts_in` takes, so the pipeline does
//! // `artifacts_in(where_.dir())`. vaani-asr is frozen and is not a dependency
//! // here, so the example stops at the directory.
//! let where_ = manager.resolve("sravaani-1.0-onnx")?;
//! println!("reading SraVaani from {}", where_.dir().display());
//! println!("engine id: {}", manager.engine_id("sravaani-1.0-onnx")?);
//!
//! // Verify, streaming, cancellable.
//! let cancel = CancelToken::new();
//! let report = manager.verify("sravaani-1.0-onnx", &cancel, &ignore)?;
//! println!("{:?} in {:?}", report.diagnosis, report.elapsed);
//! # Ok(())
//! # }
//! ```
//!
//! # Deliberate gaps
//!
//! * **No `keyring`.** A Hugging Face token is fetched by the caller and handed
//!   in as [`download::Redacted`]. This crate must not be able to read the
//!   Windows Credential Manager by accident, and it cannot be zeroized without
//!   the `zeroize` dependency. See `INTEGRATION NOTES`.
//! * **No `hf-hub`.** ADR-005 names it as the intended downloader; it is not in
//!   `[workspace.dependencies]`, so the transfer is hand-rolled on `reqwest`.
//!   The Hugging Face URL layout, redirects and range behaviour are implemented
//!   directly, and tested against a local HTTP server. See `OPEN ISSUES`.
//! * **No `windows` crate.** [`sys`] declares two FFI calls by hand for free
//!   disk space and peak working set, because the crate's `Cargo.toml` is frozen
//!   without it. Both carry SAFETY notes and both have a one-line replacement
//!   once the dependency lands.

#![warn(missing_docs)]
#![warn(clippy::doc_markdown)]

/// Crate version, from the workspace.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod discover;
pub mod download;
pub mod lock;
pub mod manager;
pub mod manifest;
pub mod paths;
pub mod progress;
pub mod sys;
pub mod verify;

pub use discover::{DiscoveredModel, Discovery, FileState, RootPresence, UnknownDir};
pub use download::{
    DownloadError, DownloadPolicy, Downloader, FileDownload, FreeSpaceProbe, Redacted,
    SystemFreeSpace,
};
pub use lock::{FileGuard, InstallLock, LockError};
pub use manager::{
    Choice, InstallOptions, InstallPlan, InstallReceipt, InstallReport, ModelLocation,
    ModelManager, ModelStatus, RemoveReport, human_bytes,
};
pub use manifest::{
    Access, GpuRequirements, Licence, Manifest, ManifestError, Model, ModelFile, Provider,
    Resources, RuntimeRequirements, Source, Task, role,
};
pub use paths::{ModelRoots, RootKind};
pub use progress::{ProgressFn, ProgressLog, Stage, TransferProgress};
pub use verify::{Diagnosis, FileOutcome, FileProblem, FileReport, Verifier, VerifyReport};

// Re-exported so a caller does not have to name `vaani_core` just to cancel a
// download, which is the one thing every caller of `install` needs.
pub use vaani_core::CancelToken;

// `MachineCapabilities` is what `install_plan_with` takes, and A2 already holds
// one, so it is re-exported rather than made a second import.
pub use vaani_inference::MachineCapabilities;

//! Verifying an install: every file present, the right size, the right bytes.
//!
//! # Streaming, and why it is not optional
//!
//! The encoder is 845 MiB. Reading it into a `Vec<u8>` to hash it would need
//! 845 MiB of process memory, in a desktop app that may already be holding a
//! CUDA context and a 20 s chunk of activations, on a laptop with 8 GB of RAM.
//! So verification reads a fixed **1 MiB** buffer and nothing else, and the
//! test suite measures peak working set to prove it.
//!
//! # The five outcomes, and why they are five
//!
//! The recovery differs for each, which is the only reason to separate them:
//!
//! | outcome | what happened | what the user does |
//! |---|---|---|
//! | `Ok` | size and hash match | nothing |
//! | `Missing` | the file is not there | download it |
//! | `Truncated` | shorter than the manifest says | download it again — this is an interrupted download |
//! | `WrongSize` | longer than the manifest says | this is not the file the manifest describes |
//! | `ContentDiffers` | right length, wrong bytes | download it again, **or** the manifest was edited |
//! | `Locked` | another process has it open | close the other window and retry |
//!
//! The last row of the table is the one that cannot be resolved from one file.
//! `ContentDiffers` with every file at the right length is ambiguous between a
//! corrupted transfer and a hand-edited manifest, and this module says so in
//! the message rather than picking one and being wrong half the time. See
//! [`VerifyReport::content_disagreement`].

use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use crate::manifest::Model;
use crate::progress::{ProgressFn, TransferProgress};
use crate::sys;
use vaani_core::CancelToken;

/// Default hash buffer. 1 MiB: large enough that the syscall overhead is
/// irrelevant, small enough that peak memory is invisible next to a model.
pub const DEFAULT_BUFFER_BYTES: usize = 1024 * 1024;

/// How often progress is reported, in bytes hashed.
///
/// 4 MiB on an 845 MiB file is 211 updates: smooth enough for a progress bar,
/// cheap enough not to matter. Reporting every 1 MiB chunk would be 845 calls
/// per file and buys nothing a human can see.
pub const PROGRESS_EVERY_BYTES: u64 = 4 * 1024 * 1024;

/// What is wrong with one file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileProblem {
    /// Not on disk at all.
    Missing,
    /// Present, shorter than the manifest says. An interrupted download looks
    /// exactly like this.
    Truncated {
        /// Bytes the manifest says.
        expected: u64,
        /// Bytes on disk.
        actual: u64,
    },
    /// Present, longer than the manifest says. Not an interrupted download:
    /// this is a different file.
    WrongSize {
        /// Bytes the manifest says.
        expected: u64,
        /// Bytes on disk.
        actual: u64,
    },
    /// Exactly the right length, different bytes.
    ContentDiffers {
        /// The manifest's hash, hex.
        expected_sha256: String,
        /// What we measured, hex.
        actual_sha256: String,
    },
    /// Another process has the file open without sharing it. On Windows this is
    /// `ERROR_SHARING_VIOLATION`, and it means an inference session or a
    /// backup tool is reading the model right now.
    Locked,
    /// Could not be read for some other reason (permissions, a directory where a
    /// file should be, an I/O error).
    Unreadable {
        /// The operating system's message.
        message: String,
    },
}

impl FileProblem {
    /// Whether re-downloading the file is the right response.
    pub fn is_recoverable_by_download(&self) -> bool {
        matches!(
            self,
            Self::Missing | Self::Truncated { .. } | Self::ContentDiffers { .. }
        )
    }

    /// The stable wire form, for logs and `--json` output.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Missing => "missing",
            Self::Truncated { .. } => "truncated",
            Self::WrongSize { .. } => "wrong_size",
            Self::ContentDiffers { .. } => "content_differs",
            Self::Locked => "locked",
            Self::Unreadable { .. } => "unreadable",
        }
    }
}

/// The result for one file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileOutcome {
    /// Present, right size, right hash.
    Verified {
        /// Bytes hashed.
        bytes: u64,
        /// The measured hash, hex.
        sha256: String,
    },
    /// Something is wrong.
    Failed(FileProblem),
}

impl FileOutcome {
    /// Whether the file passed.
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Verified { .. })
    }
}

/// Per-file detail in a [`VerifyReport`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileReport {
    /// Manifest path.
    pub path: String,
    /// What the file is for.
    pub role: String,
    /// Whether the model is unusable without it.
    pub required: bool,
    /// Bytes the manifest says.
    pub expected_bytes: u64,
    /// Bytes on disk, when it could be determined.
    pub actual_bytes: Option<u64>,
    /// What happened.
    pub outcome: FileOutcome,
    /// Wall time spent on this file.
    pub elapsed: Duration,
}

impl FileReport {
    /// The problem, if there is one.
    pub fn problem(&self) -> Option<&FileProblem> {
        match &self.outcome {
            FileOutcome::Verified { .. } => None,
            FileOutcome::Failed(p) => Some(p),
        }
    }
}

/// The single most important thing a verification found.
///
/// Ordered by how much it matters, so a caller that wants one number can take
/// [`Diagnosis::as_str`] and a caller that wants to be precise can switch on
/// it. Several problems can be present at once; the report keeps the per-file
/// detail and this is the headline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Diagnosis {
    /// Every file verified.
    Good,
    /// Stopped part-way because the caller cancelled.
    Cancelled,
    /// A file is open in another process.
    Locked,
    /// A file could not be read.
    Unreadable,
    /// A required file is not on disk.
    Missing,
    /// A file is shorter than the manifest says.
    Truncated,
    /// A file is longer than the manifest says.
    WrongSize,
    /// Every file is the right length but at least one is the wrong content.
    ContentDiffers,
}

impl Diagnosis {
    /// The stable wire form.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Good => "good",
            Self::Cancelled => "cancelled",
            Self::Locked => "locked",
            Self::Unreadable => "unreadable",
            Self::Missing => "missing",
            Self::Truncated => "truncated",
            Self::WrongSize => "wrong_size",
            Self::ContentDiffers => "content_differs",
        }
    }

    /// Whether the model can be used as it stands.
    pub const fn is_good(&self) -> bool {
        matches!(self, Self::Good)
    }
}

/// The result of verifying one model directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyReport {
    /// The manifest id.
    pub model_id: String,
    /// The directory that was checked.
    pub dir: PathBuf,
    /// Per-file detail, in manifest order.
    pub files: Vec<FileReport>,
    /// The headline.
    pub diagnosis: Diagnosis,
    /// Manifest paths of the files that failed.
    pub failed: Vec<String>,
    /// Bytes actually hashed, for a throughput figure.
    pub bytes_hashed: u64,
    /// Wall time.
    pub elapsed: Duration,
    /// Whether the caller cancelled.
    pub cancelled: bool,
}

impl VerifyReport {
    /// Whether the model is intact and complete.
    pub fn is_ok(&self) -> bool {
        self.diagnosis.is_good()
    }

    /// Whether every file is present at the right length but at least one
    /// hashes differently.
    ///
    /// This is the state that cannot be diagnosed from the filesystem alone: a
    /// corrupted transfer and a hand-edited manifest produce identical bytes on
    /// disk. The message says so rather than guessing.
    pub fn content_disagreement(&self) -> bool {
        self.diagnosis == Diagnosis::ContentDiffers
    }

    /// Bytes per second, or `None` if nothing was hashed.
    pub fn throughput(&self) -> Option<f64> {
        let s = self.elapsed.as_secs_f64();
        (s > 0.0).then(|| self.bytes_hashed as f64 / s)
    }

    /// The application error for this report, if it is not good.
    ///
    /// Three problems the user can hit map to three different codes, which is
    /// the point: a missing model, a damaged model and a full disk are not the
    /// same message and not the same next step.
    pub fn to_error(&self) -> Option<vaani_core::VaaniError> {
        use vaani_core::{UserFacingError, err};
        let label = self.model_id.clone();
        let listed = |files: &[String]| {
            if files.is_empty() {
                String::new()
            } else {
                format!(" ({} file(s) affected)", files.len())
            }
        };
        match self.diagnosis {
            Diagnosis::Good => None,
            Diagnosis::Cancelled => Some(err::cancelled()),
            Diagnosis::Missing => Some(
                err::model_unavailable()
                    .with_cause(format!("{label}: missing {}", self.failed.join(", ")))
                    .with_recovery("Download the model again from Settings."),
            ),
            Diagnosis::Truncated => Some(
                err::model_corrupt()
                    .with_cause(format!("{label}: truncated {}", self.failed.join(", ")))
                    .with_recovery(
                        "Download the model again. A file shorter than expected is almost \
                         always a download that was interrupted.",
                    ),
            ),
            Diagnosis::WrongSize => Some(
                err::model_corrupt()
                    .with_cause(format!("{label}: wrong size {}", self.failed.join(", ")))
                    .with_recovery(
                        "These files are not the ones this model is made of. Download the \
                         model again, and check nothing else was put in this folder.",
                    ),
            ),
            Diagnosis::ContentDiffers => Some(
                err::model_corrupt()
                    .with_cause(format!(
                        "{label}: content differs {}",
                        self.failed.join(", ")
                    ))
                    .with_recovery(
                        "Every file is the right size but the contents do not match. Either \
                         the download was damaged, or the model list was edited. Download the \
                         model again; if it keeps happening, restore the model list.",
                    ),
            ),
            Diagnosis::Locked => Some(
                vaani_core::VaaniError::new(UserFacingError::new(
                    vaani_core::ErrorCode::StorageFailed,
                    "The model is in use.",
                    format!(
                        "Another VaaniDesk window or process is reading the model, so it \
                         cannot be checked or replaced right now.{}",
                        listed(&self.failed)
                    ),
                ))
                .with_cause(format!("{label}: locked {}", self.failed.join(", ")))
                .with_recovery(
                    "Close other VaaniDesk windows, wait for any transcription to finish, \
                     then try again.",
                ),
            ),
            Diagnosis::Unreadable => Some(
                vaani_core::VaaniError::new(UserFacingError::new(
                    vaani_core::ErrorCode::StorageFailed,
                    "The model could not be read.",
                    format!(
                        "Some model files could not be opened.{} This is usually a \
                         permissions problem or a drive that is not connected.",
                        listed(&self.failed)
                    ),
                ))
                .with_cause(format!("{label}: unreadable {}", self.failed.join(", ")))
                .with_recovery("Check that the model folder is readable, then try again."),
            ),
        }
    }
}

/// Streams a SHA-256 over a file with a fixed buffer.
///
/// The buffer is passed in rather than allocated per call so that verifying
/// ten files allocates once. Never reads the whole file: that is the property
/// the peak-RSS test exists to protect.
///
/// `on_chunk` returns `false` to stop early, which is how cancellation lands
/// within a megabyte rather than at the end of an 845 MiB file. `Ok(None)`
/// means it stopped; `Ok(Some(..))` means it reached the end.
pub fn hash_file_streaming(
    path: &Path,
    buffer: &mut [u8],
    mut on_chunk: impl FnMut(u64) -> bool,
) -> std::io::Result<Option<([u8; 32], u64)>> {
    let mut f = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut total = 0u64;
    loop {
        let n = f.read(buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
        total += n as u64;
        if !on_chunk(n as u64) {
            return Ok(None);
        }
    }
    Ok(Some((hasher.finalize().into(), total)))
}

/// Verifies model directories.
///
/// Holds the buffer *size* rather than the buffer, so one allocation is made
/// per model regardless of how many files it has, and so the hashing closure
/// can borrow it without fighting the verifier for `&mut self`.
#[derive(Debug)]
pub struct Verifier {
    buffer_bytes: usize,
    progress_every: u64,
}

impl Default for Verifier {
    fn default() -> Self {
        Self::new()
    }
}

impl Verifier {
    /// A verifier with the default 1 MiB buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::with_buffer(DEFAULT_BUFFER_BYTES)
    }

    /// A verifier with a caller-chosen buffer. A smaller buffer lowers peak
    /// memory and costs throughput; 1 MiB is a good trade on every machine
    /// this ships to. Silently floored at 4 KiB, because a zero-length read
    /// buffer is an infinite loop rather than a small allocation.
    #[must_use]
    pub fn with_buffer(bytes: usize) -> Self {
        Self {
            buffer_bytes: bytes.max(4096),
            progress_every: PROGRESS_EVERY_BYTES,
        }
    }

    /// The hash buffer size this verifier will use.
    pub fn buffer_bytes(&self) -> usize {
        self.buffer_bytes
    }

    /// How often to report progress, in bytes hashed.
    #[must_use]
    pub fn with_progress_every(mut self, bytes: u64) -> Self {
        self.progress_every = bytes.max(1);
        self
    }

    /// Verify every file of `model` inside `dir`.
    ///
    /// Infallible on purpose: a damaged file is a **result**, not an exception,
    /// because the caller has to show which file failed and a bare `Err` cannot
    /// carry that. Callers that want an error use [`VerifyReport::to_error`],
    /// and cancellation is reported in the report rather than thrown, so a
    /// half-finished bar can still be drawn.
    pub fn verify(
        &mut self,
        model: &Model,
        dir: &Path,
        cancel: &CancelToken,
        progress: ProgressFn<'_>,
    ) -> VerifyReport {
        let start = Instant::now();
        let total_bytes = model.files.iter().map(|f| f.bytes).sum::<u64>();
        let mut done_bytes = 0u64;
        let mut files = Vec::with_capacity(model.files.len());
        let mut cancelled = false;
        // One allocation for the whole model: this is the property that keeps
        // peak memory at 1 MiB while hashing an 845 MiB encoder.
        let mut buffer = vec![0u8; self.buffer_bytes];

        for f in &model.files {
            let path = dir.join(&f.relative);
            let file_start = Instant::now();
            // The closure takes every mutable value as an argument rather than
            // capturing it, so it never fights the loop for a borrow. It is
            // `Fn`, not `FnMut`, because nothing in it changes.
            let report = |files_done: usize, bytes_done: u64| {
                progress(TransferProgress {
                    stage: crate::progress::Stage::Verifying,
                    model_id: model.id.clone(),
                    file: Some(f.path.clone()),
                    files_done,
                    files_total: model.files.len(),
                    bytes_done,
                    bytes_total: total_bytes,
                });
            };

            let actual_bytes = fs_len(&path);
            let outcome = if cancelled {
                FileOutcome::Failed(FileProblem::Missing)
            } else {
                match actual_bytes {
                    None => FileOutcome::Failed(FileProblem::Missing),
                    // The size check is free and it is the one that catches an
                    // interrupted download, so it comes before 845 MiB of
                    // hashing.
                    Some(actual) if actual != f.bytes => FileOutcome::Failed(if actual < f.bytes {
                        FileProblem::Truncated {
                            expected: f.bytes,
                            actual,
                        }
                    } else {
                        FileProblem::WrongSize {
                            expected: f.bytes,
                            actual,
                        }
                    }),
                    Some(_) => {
                        let mut since_report = 0u64;
                        match self.hash_one(
                            &path,
                            cancel,
                            &mut buffer,
                            &mut done_bytes,
                            &mut since_report,
                            &report,
                        ) {
                            HashOutcome::Hashed(hash, bytes) => {
                                if hash == f.sha256 {
                                    FileOutcome::Verified {
                                        bytes,
                                        sha256: hex::encode(hash),
                                    }
                                } else {
                                    FileOutcome::Failed(FileProblem::ContentDiffers {
                                        expected_sha256: f.sha256_hex(),
                                        actual_sha256: hex::encode(hash),
                                    })
                                }
                            }
                            HashOutcome::Failed(p) => FileOutcome::Failed(p),
                            HashOutcome::Cancelled => {
                                cancelled = true;
                                FileOutcome::Failed(FileProblem::Missing)
                            }
                        }
                    }
                }
            };

            report(files.len() + 1, done_bytes);
            files.push(FileReport {
                path: f.path.clone(),
                role: f.role.clone(),
                required: f.required,
                expected_bytes: f.bytes,
                actual_bytes,
                outcome,
                elapsed: file_start.elapsed(),
            });
        }

        let failed: Vec<String> = files
            .iter()
            .filter(|f| f.problem().is_some() && f.required)
            .map(|f| f.path.clone())
            .collect();

        VerifyReport {
            model_id: model.id.clone(),
            dir: dir.to_path_buf(),
            diagnosis: diagnose(&files, cancelled),
            files,
            failed,
            bytes_hashed: done_bytes,
            elapsed: start.elapsed(),
            cancelled,
        }
    }

    /// Hash one file, reporting progress and honouring cancellation.
    fn hash_one(
        &self,
        path: &Path,
        cancel: &CancelToken,
        buffer: &mut [u8],
        done_bytes: &mut u64,
        since_report: &mut u64,
        report: &(dyn Fn(usize, u64) + Send + Sync),
    ) -> HashOutcome {
        let mut cancelled = false;
        let result = hash_file_streaming(path, buffer, |n| {
            *done_bytes += n;
            *since_report += n;
            // Two atomic loads per MiB. The chunk is 1 MiB of file I/O, so this
            // is not measurable, and it is what makes cancellation land within
            // a megabyte instead of at the end of an 845 MiB file.
            if cancel.is_cancelled() {
                cancelled = true;
                return false;
            }
            if *since_report >= self.progress_every {
                *since_report = 0;
                report(0, *done_bytes);
            }
            true
        });
        if cancelled {
            return HashOutcome::Cancelled;
        }
        match result {
            Ok(Some((hash, bytes))) => HashOutcome::Hashed(hash, bytes),
            // Stopped without the cancel token having been seen: only possible if
            // `on_chunk` returned false for another reason, which it does not.
            Ok(None) => HashOutcome::Cancelled,
            Err(e) => HashOutcome::Failed(classify_io(&e)),
        }
    }
}

/// What hashing one file produced.
#[derive(Debug, Clone, PartialEq, Eq)]
enum HashOutcome {
    Hashed([u8; 32], u64),
    Failed(FileProblem),
    Cancelled,
}

fn fs_len(p: &Path) -> Option<u64> {
    std::fs::metadata(p)
        .ok()
        .filter(|m| m.is_file())
        .map(|m| m.len())
}

/// Turn an I/O error into the problem the user needs to see.
fn classify_io(e: &std::io::Error) -> FileProblem {
    if sys::is_sharing_violation(e) {
        FileProblem::Locked
    } else {
        FileProblem::Unreadable {
            message: e.to_string(),
        }
    }
}

/// Rank of a diagnosis; higher wins. Locked is above Unreadable because
/// "another window has it open" is a fact the user can act on immediately and
/// "could not be read" is not.
fn rank(d: Diagnosis) -> u8 {
    match d {
        Diagnosis::Good => 0,
        Diagnosis::ContentDiffers => 1,
        Diagnosis::WrongSize => 2,
        Diagnosis::Truncated => 3,
        Diagnosis::Missing => 4,
        Diagnosis::Unreadable => 5,
        Diagnosis::Locked => 6,
        Diagnosis::Cancelled => 7,
    }
}

/// Pick the headline diagnosis, most serious first.
fn diagnose(files: &[FileReport], cancelled: bool) -> Diagnosis {
    if cancelled {
        return Diagnosis::Cancelled;
    }
    let mut worst = Diagnosis::Good;
    for f in files {
        // An optional file being wrong is a note, not a fault: the fp32
        // reference encoder may legitimately be absent.
        if !f.required {
            continue;
        }
        let d = match f.problem() {
            None => continue,
            Some(FileProblem::Missing) => Diagnosis::Missing,
            Some(FileProblem::Truncated { .. }) => Diagnosis::Truncated,
            Some(FileProblem::WrongSize { .. }) => Diagnosis::WrongSize,
            Some(FileProblem::ContentDiffers { .. }) => Diagnosis::ContentDiffers,
            Some(FileProblem::Locked) => Diagnosis::Locked,
            Some(FileProblem::Unreadable { .. }) => Diagnosis::Unreadable,
        };
        if rank(d) > rank(worst) {
            worst = d;
        }
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnoses_rank_the_worst_finding_first() {
        let mk = |p: Option<FileProblem>| FileReport {
            path: "x".into(),
            role: "weights".into(),
            required: true,
            expected_bytes: 1,
            actual_bytes: Some(1),
            outcome: match p {
                None => FileOutcome::Verified {
                    bytes: 1,
                    sha256: "ab".into(),
                },
                Some(p) => FileOutcome::Failed(p),
            },
            elapsed: Duration::ZERO,
        };
        let differs = |()| FileProblem::ContentDiffers {
            expected_sha256: "aa".into(),
            actual_sha256: "bb".into(),
        };
        // A missing file outranks a corrupt one: the file that is absent is the
        // one that has to be fetched.
        assert_eq!(
            diagnose(
                &[mk(Some(differs(()))), mk(Some(FileProblem::Missing))],
                false
            ),
            Diagnosis::Missing
        );
        // And a locked file outranks both, because retrying will not help until
        // the other window closes.
        assert_eq!(
            diagnose(
                &[
                    mk(Some(FileProblem::Missing)),
                    mk(Some(FileProblem::Locked))
                ],
                false
            ),
            Diagnosis::Locked
        );
        assert_eq!(diagnose(&[mk(None)], false), Diagnosis::Good);
        assert_eq!(diagnose(&[mk(None)], true), Diagnosis::Cancelled);
        // An optional file failing does not make the model bad.
        let mut optional = mk(Some(FileProblem::Missing));
        optional.required = false;
        assert_eq!(diagnose(&[optional], false), Diagnosis::Good);
    }

    #[test]
    fn diagnosis_names_are_stable_wire_forms() {
        for d in [
            Diagnosis::Good,
            Diagnosis::Cancelled,
            Diagnosis::Locked,
            Diagnosis::Unreadable,
            Diagnosis::Missing,
            Diagnosis::Truncated,
            Diagnosis::WrongSize,
            Diagnosis::ContentDiffers,
        ] {
            assert!(!d.as_str().is_empty());
            assert!(
                d.as_str()
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c == '_')
            );
        }
    }

    #[test]
    fn streaming_a_file_matches_the_known_hash_of_a_known_input() {
        // SHA-256("abc") is a published constant, so this pins the hashing path
        // itself rather than trusting round-tripping our own code against itself.
        let p = std::env::temp_dir().join(format!("vaani-a7-abc-{}.bin", std::process::id()));
        std::fs::write(&p, b"abc").unwrap();
        let mut buf = vec![0u8; 4]; // deliberately smaller than the file
        let (hash, n) = hash_file_streaming(&p, &mut buf, |_| true)
            .unwrap()
            .expect("the whole file");
        assert_eq!(n, 3);
        assert_eq!(
            hex::encode(hash),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // Stopping early returns `None` rather than a hash of a partial file,
        // which is the whole point: a half-read must never look like a result.
        let mut calls = 0;
        let stopped = hash_file_streaming(&p, &mut buf, |_| {
            calls += 1;
            false
        })
        .unwrap();
        assert!(stopped.is_none());
        assert_eq!(
            calls, 1,
            "it must stop on the first chunk, not read to the end"
        );
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn a_sharing_violation_is_reported_as_locked_not_unreadable() {
        let e = std::io::Error::from_raw_os_error(sys::ERROR_SHARING_VIOLATION);
        assert_eq!(classify_io(&e), FileProblem::Locked);
        let e = std::io::Error::other("permission denied");
        assert!(matches!(classify_io(&e), FileProblem::Unreadable { .. }));
    }

    #[test]
    fn the_three_problems_map_to_three_different_error_codes() {
        let mk = |d: Diagnosis, failed: &[&str]| VerifyReport {
            model_id: "m".into(),
            dir: PathBuf::from("d"),
            files: Vec::new(),
            diagnosis: d,
            failed: failed.iter().map(|s| (*s).to_string()).collect(),
            bytes_hashed: 0,
            elapsed: Duration::ZERO,
            cancelled: d == Diagnosis::Cancelled,
        };
        // A missing model, a damaged model and a cancelled check are three
        // different user-facing problems, and the taxonomy has three codes.
        assert_eq!(
            mk(Diagnosis::Missing, &["a"]).to_error().unwrap().code(),
            vaani_core::ErrorCode::ModelUnavailable
        );
        for d in [
            Diagnosis::Truncated,
            Diagnosis::WrongSize,
            Diagnosis::ContentDiffers,
        ] {
            assert_eq!(
                mk(d, &["a"]).to_error().unwrap().code(),
                vaani_core::ErrorCode::ModelCorrupt,
                "{d:?} must be a corrupt-model problem"
            );
        }
        assert_eq!(
            mk(Diagnosis::Locked, &["a"]).to_error().unwrap().code(),
            vaani_core::ErrorCode::StorageFailed
        );
        assert!(
            mk(Diagnosis::Cancelled, &[])
                .to_error()
                .unwrap()
                .is_cancelled()
        );
        assert!(mk(Diagnosis::Good, &[]).to_error().is_none());
    }

    #[test]
    fn every_error_message_carries_a_recovery() {
        for d in [
            Diagnosis::Missing,
            Diagnosis::Truncated,
            Diagnosis::WrongSize,
            Diagnosis::ContentDiffers,
            Diagnosis::Locked,
            Diagnosis::Unreadable,
        ] {
            let e = mk_err(d);
            assert!(
                e.user.recovery.is_some(),
                "{d:?} has no recovery action, which is the failure this product exists to avoid"
            );
        }
    }

    fn mk_err(d: Diagnosis) -> vaani_core::VaaniError {
        VerifyReport {
            model_id: "m".into(),
            dir: PathBuf::from("d"),
            files: Vec::new(),
            diagnosis: d,
            failed: vec!["encoder.fp16.onnx".into()],
            bytes_hashed: 0,
            elapsed: Duration::ZERO,
            cancelled: false,
        }
        .to_error()
        .unwrap()
    }

    #[test]
    fn the_content_disagreement_message_names_both_causes() {
        // It genuinely is ambiguous, and a message that picks one is wrong half
        // the time. Assert the honest version survives.
        let e = mk_err(Diagnosis::ContentDiffers);
        let r = e.user.recovery.unwrap();
        assert!(r.contains("edited"), "{r}");
        assert!(r.contains("Download"), "{r}");
    }

    #[test]
    fn the_verifier_chooses_one_buffer_size_however_many_files_it_checks() {
        assert_eq!(Verifier::with_buffer(64 * 1024).buffer_bytes(), 64 * 1024);
        // A silly request is floored rather than producing a zero-length read
        // loop that would spin forever on a non-empty file.
        assert_eq!(Verifier::with_buffer(0).buffer_bytes(), 4096);
    }
}

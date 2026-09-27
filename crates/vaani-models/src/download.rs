//! Resumable, verified, atomic downloads.
//!
//! # The five properties, and the code that provides each
//!
//! | requirement | how |
//! |---|---|
//! | stable URL per file | [`crate::manifest::Source::url_for`], derived from the manifest rather than stored, so it cannot go stale |
//! | resume after an interruption | `Range: bytes=<len>-` against the existing `.part` file, appended |
//! | retry with backoff | [`DownloadPolicy::delay_for`], exponential and capped, on connection errors and 5xx/408/429 |
//! | progress in bytes and fraction, cancellable | [`TransferProgress`], checked every chunk |
//! | only moved into place after the hash verifies | `.partial` -> hash -> `fs::rename` |
//! | free-space check before starting | [`Downloader::precheck`], against an injectable [`FreeSpaceProbe`] |
//! | clear error when the server has no ranges | [`DownloadError::NoRangeSupport`] |
//!
//! # The atomic ordering
//!
//! ```text
//! .partial/<name>.part   append, hash, and only then
//! <name>                 fs::rename -- atomic on NTFS within a volume
//! .installed.json        the receipt, written last, in the same manner
//! ```
//!
//! The temp file is created **in the model's own directory**, not in `%TEMP%`.
//! That is not tidiness: `fs::rename` is only atomic within a volume, and
//! `%TEMP%` is frequently on `C:` while a user may have pointed the model
//! directory at `D:`. A cross-volume "atomic" move is a copy followed by a
//! delete, and a crash in the middle of it leaves exactly the corrupt file this
//! design exists to prevent.
//!
//! Note also that no extra space is needed for the final move: `fs::rename` is a
//! directory-entry update, not a copy, so the free-space requirement is the sum
//! of the remaining bytes and nothing more. A naive "download to temp then copy"
//! design would need twice the model size, which for `SraVaani` is 5 GiB instead
//! of 2.5 GiB and would refuse to install on a laptop that has the room.
//!
//! # Hugging Face
//!
//! A Hub model is `org/name`; a file is at
//! `https://huggingface.co/org/name/resolve/main/<path>`, which 302s to a CDN
//! (`cas-bridge.xethub.hf.co` / the xet bridge). Three consequences, all
//! handled here:
//!
//! * redirects are followed (reqwest's default limit of 10 is kept, and a
//!   redirect to a *different* host is normal, not an attack);
//! * a CDN presigned URL answers a range request from a byte offset, so resume
//!   works across the redirect — but the offset is the one we asked for, and the
//!   `Content-Range` is checked to be sure;
//! * **a bearer token is sent only to `huggingface.co`**, never forwarded to
//!   the CDN host, because forwarding a credential to a host that did not ask
//!   for it is how tokens leak. See [`crate::manifest::Source::needs_bearer`].

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use reqwest::header::{HeaderMap, HeaderValue, RANGE};
use sha2::{Digest, Sha256};

use crate::lock::PARTIAL_DIR;
use crate::progress::{ProgressFn, Stage, TransferProgress};
use crate::sys;
use vaani_core::CancelToken;

/// Bytes of slack demanded on top of the model, so a download cannot succeed and
/// then fail while unpacking. 64 MiB is roughly one 4K video's worth and costs
/// the user nothing.
pub const DEFAULT_RESERVE_BYTES: u64 = 64 * 1024 * 1024;

/// How a string-sized secret is held so it cannot reach a log.
#[derive(Clone, PartialEq, Eq)]
pub struct Redacted(String);

impl Redacted {
    /// Wrap a secret.
    #[must_use]
    pub fn new(secret: impl Into<String>) -> Self {
        Self(secret.into())
    }

    /// Read the secret. The only way out, and the only reason the type exists is
    /// to make reaching it a deliberate, greppable act.
    #[must_use]
    pub fn expose(&self) -> &str {
        &self.0
    }

    /// Whether there is a secret at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::fmt::Debug for Redacted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Redacted(<redacted>)")
    }
}

impl std::fmt::Display for Redacted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<redacted>")
    }
}

/// A source of free-space figures.
///
/// Injectable so the disk-full path is testable. Filling a real volume to
/// discover what a full disk does is not a test anyone should write.
pub trait FreeSpaceProbe: Send + Sync + std::fmt::Debug {
    /// Free bytes available to this user on the volume holding `path`, or
    /// `None` when the platform cannot say.
    fn free_bytes(&self, path: &Path) -> Option<u64>;
}

/// The real thing: `GetDiskFreeSpaceExW` on Windows, unknown elsewhere.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemFreeSpace;

impl FreeSpaceProbe for SystemFreeSpace {
    fn free_bytes(&self, path: &Path) -> Option<u64> {
        sys::free_bytes(path)
    }
}

/// A probe that always says the same thing, for tests.
#[derive(Debug, Clone, Copy)]
pub struct FixedFreeSpace(pub Option<u64>);

impl FreeSpaceProbe for FixedFreeSpace {
    fn free_bytes(&self, _path: &Path) -> Option<u64> {
        self.0
    }
}

/// Retry, timeout and reserve settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadPolicy {
    /// Total attempts per file, including the first.
    pub max_attempts: u32,
    /// The first retry's delay. Doubles per attempt.
    pub base_delay: Duration,
    /// The ceiling on that delay.
    pub max_delay: Duration,
    /// Slack demanded on top of the model size.
    pub reserve_bytes: u64,
    /// Whether a server that ignores `Range` may be retried from zero.
    ///
    /// `false` by default. Silently restarting a 2.5 GB download from the
    /// beginning because a CDN ignored a range header looks to a user on a hotel
    /// connection like a bug that repeats forever, so it is an explicit,
    /// reported failure instead.
    pub allow_restart_without_range: bool,
    /// Connect timeout.
    pub connect_timeout: Duration,
    /// Read timeout for the whole body.
    pub read_timeout: Duration,
    /// Write buffer.
    pub chunk_bytes: usize,
}

impl Default for DownloadPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 4,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(20),
            reserve_bytes: DEFAULT_RESERVE_BYTES,
            allow_restart_without_range: false,
            connect_timeout: Duration::from_secs(30),
            read_timeout: Duration::from_secs(600),
            chunk_bytes: 256 * 1024,
        }
    }
}

impl DownloadPolicy {
    /// A policy that gives up immediately, for tests that assert a single
    /// failure mode without waiting through the backoff.
    #[must_use]
    pub fn single_attempt() -> Self {
        Self {
            max_attempts: 1,
            base_delay: Duration::ZERO,
            ..Self::default()
        }
    }

    /// The delay before retry number `attempt`, counting the first retry as 0.
    ///
    /// Exponential and capped. No jitter, deliberately: these retries are for a
    /// flaky home connection and a CDN, not a thundering herd, and a jitter term
    /// would make the backoff untestable without seeding an RNG. The cap is what
    /// stops a fourth attempt from waiting nine minutes.
    #[must_use]
    pub fn delay_for(&self, attempt: u32) -> Duration {
        let factor = 1u32.checked_shl(attempt).unwrap_or(u32::MAX);
        self.base_delay.saturating_mul(factor).min(self.max_delay)
    }
}

/// Why a download failed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum DownloadError {
    /// The connection failed.
    #[error("network error while fetching {url}: {message}")]
    Network {
        /// The URL, with any query string that could carry a token removed.
        url: String,
        /// The transport's message.
        message: String,
    },
    /// A response we cannot use.
    #[error("{url} returned HTTP {status}")]
    Http {
        /// The URL.
        url: String,
        /// The status code.
        status: u16,
    },
    /// The published file is not where the manifest says.
    #[error(
        "{url} was not found (HTTP 404); the model's published files have moved or the revision is wrong"
    )]
    NotFound {
        /// The URL.
        url: String,
    },
    /// A credential is needed and none was supplied.
    #[error("{url} requires authentication and no token was supplied")]
    AuthRequired {
        /// The URL.
        url: String,
    },
    /// A credential was supplied and refused.
    #[error(
        "{url} refused the supplied token; the account has not been granted access to this model"
    )]
    AccessNotGranted {
        /// The URL.
        url: String,
    },
    /// The server ignored a range request.
    #[error(
        "{url} does not support resuming: it ignored a range request for the bytes already downloaded"
    )]
    NoRangeSupport {
        /// The URL.
        url: String,
    },
    /// The server's `Content-Range` did not start where we asked.
    #[error("{url} answered a range request with the wrong byte range ({header:?})")]
    BadContentRange {
        /// The URL.
        url: String,
        /// What the server said.
        header: String,
    },
    /// The connection ended early.
    #[error("{path} stopped at {received} of {expected} bytes")]
    ShortRead {
        /// The temporary file.
        path: PathBuf,
        /// Bytes received.
        received: u64,
        /// Bytes expected.
        expected: u64,
    },
    /// The bytes arrived but are not the file.
    #[error("{path} does not match its expected checksum (expected {expected}, got {actual})")]
    HashMismatch {
        /// The temporary file.
        path: PathBuf,
        /// The manifest's hash.
        expected: String,
        /// What was measured.
        actual: String,
    },
    /// Not enough room for the model.
    #[error("not enough free disk space: {needed} bytes needed, {free} free")]
    InsufficientDisk {
        /// Bytes the model needs, including the reserve.
        needed: u64,
        /// Bytes free.
        free: u64,
    },
    /// The disk filled up part-way through a write.
    #[error("the disk filled up while writing {path}")]
    DiskFull {
        /// The file being written.
        path: PathBuf,
    },
    /// A local I/O failure.
    #[error("could not write {path}: {message}")]
    Io {
        /// The path.
        path: PathBuf,
        /// The operating system's message.
        message: String,
    },
    /// The caller cancelled.
    #[error("the download was cancelled")]
    Cancelled,
    /// Every attempt failed.
    #[error("giving up on {url} after {attempts} attempt(s): {last}")]
    RetriesExhausted {
        /// The URL.
        url: String,
        /// Attempts made.
        attempts: u32,
        /// The last failure.
        last: Box<DownloadError>,
    },
}

impl DownloadError {
    /// Project onto the application error taxonomy.
    ///
    /// The distinctions that matter to a user: 404 is a *manifest or upstream*
    /// problem, 401/403 are credential problems with two different actions, a
    /// hash mismatch is a corrupt download, a short disk is a disk, and a
    /// cancellation is not a failure at all.
    #[must_use]
    pub fn to_vaani_error(&self) -> vaani_core::VaaniError {
        use vaani_core::{ErrorCode, UserFacingError, VaaniError, err};
        match self {
            Self::Cancelled => err::cancelled(),
            Self::NotFound { .. } => VaaniError::new(UserFacingError::new(
                ErrorCode::ModelUnavailable,
                "This model is not available.",
                "The files for this model are not where the model list says they are. It \
                 may have been moved or withdrawn by its publisher.",
            ))
            .with_cause(self)
            .with_recovery("Update VaaniDesk, or choose a different model."),
            Self::AuthRequired { .. } => err::auth_required()
                .with_cause(self)
                .with_recovery("Add your Hugging Face token in Settings, then try again."),
            Self::AccessNotGranted { .. } => {
                err::access_not_granted().with_cause(self).with_recovery(
                    "Request access to this model on its Hugging Face page. Once it is \
                     approved, try again.",
                )
            }
            Self::HashMismatch { .. } => err::model_corrupt().with_cause(self).with_recovery(
                "The downloaded file was damaged in transit. Download it again; if it \
                     keeps happening, try a different network.",
            ),
            Self::InsufficientDisk { .. } | Self::DiskFull { .. } => err::insufficient_disk()
                .with_cause(self)
                .with_recovery("Free up some disk space, then try again."),
            Self::Io { .. } => err::storage_failed()
                .with_cause(self)
                .with_recovery("Check that the models folder is writable, then try again."),
            Self::Network { .. }
            | Self::Http { .. }
            | Self::NoRangeSupport { .. }
            | Self::BadContentRange { .. }
            | Self::ShortRead { .. } => {
                // `err::network_failed()` already tells the user to check the
                // connection in its detail, so the recovery must *add* something
                // rather than repeat it -- a doubled sentence reads as a bug.
                err::network_failed().with_cause(self).with_recovery(
                    "A hotel, campus or tethered network may block large downloads. \
                     Try a different network if this keeps happening.",
                )
            }
            // Follow the cause rather than the wrapper. `try_once` never returns
            // a `RetriesExhausted`, so this cannot recurse, and it is the
            // difference between "the connection dropped four times" and "the
            // file that arrived was not the file you asked for".
            Self::RetriesExhausted { last, .. } => last.to_vaani_error(),
        }
    }
}

/// What one file's download did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileDownload {
    /// The verified temporary file, ready to be renamed into place.
    pub part: PathBuf,
    /// Bytes written in total.
    pub bytes: u64,
    /// Bytes that were already present, i.e. the resume offset. Zero for a
    /// fresh download.
    pub resumed_from: u64,
    /// Attempts used.
    pub attempts: u32,
    /// Wall time.
    pub elapsed: Duration,
}

/// Fetches model files.
///
/// `Clone` is cheap and shares one connection pool: `reqwest::Client` is an
/// `Arc` inside, and a manager that is cloned per job must not open a new pool
/// each time.
#[derive(Debug, Clone)]
pub struct Downloader {
    client: reqwest::Client,
    policy: DownloadPolicy,
    free_space: Arc<dyn FreeSpaceProbe>,
    auth: Option<Redacted>,
}

impl Default for Downloader {
    fn default() -> Self {
        Downloader::with_policy(DownloadPolicy::default(), Arc::new(SystemFreeSpace))
    }
}

impl Downloader {
    /// A downloader with the default policy and the real free-space probe.
    #[must_use]
    pub fn new() -> Downloader {
        Self::default()
    }

    /// A downloader with a chosen policy and free-space probe.
    ///
    /// # Panics
    ///
    /// As [`Downloader::new`].
    #[must_use]
    pub fn with_policy(policy: DownloadPolicy, free_space: Arc<dyn FreeSpaceProbe>) -> Downloader {
        let client = reqwest::Client::builder()
            .connect_timeout(policy.connect_timeout)
            .timeout(policy.read_timeout)
            // Hugging Face answers `resolve/` with a 302 to a CDN. Following it
            // is required, and the default limit of 10 is right.
            .redirect(reqwest::redirect::Policy::limited(10))
            .user_agent(concat!("VaaniDesk/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("the default HTTP client must build");
        Downloader {
            client,
            policy,
            free_space,
            auth: None,
        }
    }

    /// Attach a bearer token.
    ///
    /// The secret is only ever sent to a host the manifest marks as needing one.
    /// `vaani-models` has no `keyring` dependency — see `INTEGRATION NOTES` —
    /// so the caller fetches it from Windows Credential Manager and hands it
    /// here.
    #[must_use]
    pub fn with_bearer(mut self, token: Redacted) -> Downloader {
        self.auth = Some(token);
        self
    }

    /// The policy in force.
    pub fn policy(&self) -> &DownloadPolicy {
        &self.policy
    }

    /// Check that `dir` has room for `needed` bytes plus the reserve.
    ///
    /// `Ok(None)` means the platform cannot say, which is deliberately not an
    /// error: refusing an install because a probe failed would be worse than
    /// attempting it and reporting a real error.
    ///
    /// # Errors
    ///
    /// [`DownloadError::InsufficientDisk`] when the figure is known and short.
    pub fn precheck(&self, dir: &Path, needed: u64) -> Result<Option<u64>, DownloadError> {
        let required = needed.saturating_add(self.policy.reserve_bytes);
        let Some(free) = self.free_space.free_bytes(dir) else {
            return Ok(None);
        };
        if free < required {
            return Err(DownloadError::InsufficientDisk {
                needed: required,
                free,
            });
        }
        Ok(Some(free))
    }

    /// The path a file's partial download lives at while it is in flight.
    ///
    /// Includes the pid so two processes that somehow got past the lock cannot
    /// write to the same bytes.
    #[must_use]
    pub fn part_path(model_dir: &Path, file: &str) -> PathBuf {
        let name = file.rsplit('/').next().unwrap_or(file);
        model_dir
            .join(PARTIAL_DIR)
            .join(format!("{name}.{}.part", std::process::id()))
    }

    /// Fetch one file, resuming if a partial download exists.
    ///
    /// Returns the verified temporary file. The caller renames it into place;
    /// this function never writes to the model's real filenames.
    ///
    /// # Errors
    ///
    /// Any [`DownloadError`]. The temporary file is left behind on a transport
    /// failure or a cancellation so the next attempt can resume, and removed on
    /// a hash mismatch because its bytes are known to be wrong.
    pub async fn download_file(
        &self,
        url: &str,
        part: &Path,
        expected_bytes: u64,
        expected_sha256: &[u8; 32],
        cancel: &CancelToken,
        progress: ProgressFn<'_>,
    ) -> Result<FileDownload, DownloadError> {
        if let Some(parent) = part.parent() {
            std::fs::create_dir_all(parent).map_err(|e| DownloadError::Io {
                path: parent.to_path_buf(),
                message: e.to_string(),
            })?;
        }
        let start = Instant::now();
        let mut last: Option<DownloadError> = None;

        for attempt in 0..self.policy.max_attempts.max(1) {
            if cancel.is_cancelled() {
                return Err(DownloadError::Cancelled);
            }
            match self
                .try_once(url, part, expected_bytes, expected_sha256, cancel, progress)
                .await
            {
                Ok(out) => {
                    return Ok(FileDownload {
                        part: part.to_path_buf(),
                        attempts: attempt + 1,
                        elapsed: start.elapsed(),
                        ..out
                    });
                }
                Err(DownloadError::Cancelled) => return Err(DownloadError::Cancelled),
                // Not worth retrying: the request itself is wrong.
                Err(
                    e @ (DownloadError::NotFound { .. }
                    | DownloadError::AuthRequired { .. }
                    | DownloadError::AccessNotGranted { .. }
                    | DownloadError::InsufficientDisk { .. }
                    | DownloadError::DiskFull { .. }
                    | DownloadError::Io { .. }),
                ) => return Err(e),
                Err(e) => {
                    let transient = e.is_transient();
                    // A definitive answer is reported as itself. "This server does
                    // not support resuming" is a fact about the server, and
                    // wrapping it in "gave up after 4 attempts" hides the one
                    // sentence that tells the user what to do about it.
                    if !transient {
                        return Err(e);
                    }
                    last = Some(e);
                    if attempt + 1 >= self.policy.max_attempts.max(1) {
                        break;
                    }
                    let delay = self.policy.delay_for(attempt);
                    tracing::warn!(
                        url,
                        attempt = attempt + 1,
                        ?delay,
                        "model download attempt failed, backing off"
                    );
                    tokio::time::sleep(delay).await;
                }
            }
        }

        Err(match last {
            // One attempt is not "exhausted", it is "it did not work": the
            // wrapper adds nothing and hides the real diagnosis.
            Some(e) if self.policy.max_attempts.max(1) <= 1 => e,
            // Otherwise this is a transient failure that survived every retry.
            // The classification follows the *cause*, not the wrapper: four
            // attempts that all failed the checksum is a corrupt download, and
            // telling the user to check their internet connection would send
            // them to the wrong place.
            Some(e) => {
                return Err(DownloadError::RetriesExhausted {
                    url: redact_url(url),
                    attempts: self.policy.max_attempts.max(1),
                    last: Box::new(e),
                });
            }
            None => DownloadError::Network {
                url: redact_url(url),
                message: "no attempt was made".into(),
            },
        })
    }

    /// One attempt.
    async fn try_once(
        &self,
        url: &str,
        part: &Path,
        expected_bytes: u64,
        expected_sha256: &[u8; 32],
        cancel: &CancelToken,
        progress: ProgressFn<'_>,
    ) -> Result<FileDownload, DownloadError> {
        let safe_url = redact_url(url);
        let have = part_len(part);
        let mut request = self.client.get(url);
        if have > 0 {
            request = request.header(RANGE, format!("bytes={have}-"));
        }
        if let Some(token) = &self.auth {
            request = request.bearer_auth(token.expose());
        }

        let response = request.send().await.map_err(|e| DownloadError::Network {
            url: safe_url.clone(),
            // The full source chain, not just the top line: `reqwest` collapses
            // a connect refusal, a TLS failure and a reset into the same
            // sentence, and the cause is what identifies which one it was.
            message: chain(&e),
        })?;
        let status = response.status().as_u16();
        match status {
            401 => return Err(DownloadError::AuthRequired { url: safe_url }),
            403 => return Err(DownloadError::AccessNotGranted { url: safe_url }),
            404 | 410 => return Err(DownloadError::NotFound { url: safe_url }),
            _ => {}
        }

        let mut append = have > 0;
        if have > 0 {
            match status {
                206 => {
                    // The server honoured the range. Make sure it honoured *our*
                    // range: a server that answers `bytes 0-` to a request for
                    // `bytes 500-` would silently corrupt the file.
                    if let Some(h) = header_str(response.headers(), "content-range")
                        && !h.starts_with(&format!("bytes {have}-"))
                    {
                        return Err(DownloadError::BadContentRange {
                            url: safe_url,
                            header: h,
                        });
                    }
                }
                200 => {
                    if !self.policy.allow_restart_without_range {
                        return Err(DownloadError::NoRangeSupport { url: safe_url });
                    }
                    append = false;
                }
                s if s >= 500 => {
                    return Err(DownloadError::Http {
                        url: safe_url,
                        status: s,
                    });
                }
                s => {
                    return Err(DownloadError::Http {
                        url: safe_url,
                        status: s,
                    });
                }
            }
        } else if !(200..300).contains(&status) {
            return Err(DownloadError::Http {
                url: safe_url,
                status,
            });
        }

        let resumed_from = if append { have } else { 0 };
        let mut file = open_for_append(part, append)?;
        let mut written = resumed_from;
        // No write buffer: `chunk()` hands back a `Bytes` the transport already
        // allocated, so copying it through a second buffer would double the
        // memory for nothing. It also means the body is never fully buffered --
        // a 2.5 GiB file streams through in transport-sized pieces.
        let mut stream = response;
        progress(TransferProgress {
            stage: Stage::Downloading,
            model_id: String::new(),
            file: None,
            files_done: 0,
            files_total: 0,
            bytes_done: written,
            bytes_total: expected_bytes,
        });

        loop {
            if cancel.is_cancelled() {
                return Err(DownloadError::Cancelled);
            }
            let chunk = match stream.chunk().await {
                Ok(c) => c,
                Err(e) => {
                    // A server that stops early is the single most common way a
                    // download fails on a real connection, and the HTTP client
                    // reports it as an opaque body-decoding error. Turning that
                    // into "the transfer stopped at 32 KB of 64 KB" is the
                    // difference between a message the user can act on and one
                    // that sends them to check their router.
                    if expected_bytes > 0 && written < expected_bytes {
                        return Err(DownloadError::ShortRead {
                            path: part.to_path_buf(),
                            received: written,
                            expected: expected_bytes,
                        });
                    }
                    return Err(DownloadError::Network {
                        url: safe_url.clone(),
                        message: e.to_string(),
                    });
                }
            };
            let Some(chunk) = chunk else { break };
            if expected_bytes > 0 && written + chunk.len() as u64 > expected_bytes {
                return Err(DownloadError::ShortRead {
                    path: part.to_path_buf(),
                    received: written,
                    expected: expected_bytes,
                });
            }
            std::io::Write::write_all(&mut file, &chunk).map_err(|e| io_err(part, e))?;
            written += chunk.len() as u64;
            progress(TransferProgress {
                stage: Stage::Downloading,
                model_id: String::new(),
                file: None,
                files_done: 0,
                files_total: 0,
                bytes_done: written,
                bytes_total: expected_bytes,
            });
        }
        drop(file);

        if expected_bytes > 0 && written != expected_bytes {
            return Err(DownloadError::ShortRead {
                path: part.to_path_buf(),
                received: written,
                expected: expected_bytes,
            });
        }

        let actual = hash_part(part)?;
        if actual != *expected_sha256 {
            // The bytes are known to be wrong, so keeping them would make every
            // later resume start from a corrupt offset.
            let _ = std::fs::remove_file(part);
            return Err(DownloadError::HashMismatch {
                path: part.to_path_buf(),
                expected: hex::encode(expected_sha256),
                actual: hex::encode(actual),
            });
        }

        Ok(FileDownload {
            part: part.to_path_buf(),
            bytes: written,
            resumed_from,
            attempts: 1,
            elapsed: Duration::ZERO,
        })
    }

    /// Verify an already-downloaded temporary file, e.g. one left by a previous
    /// run, without touching the network.
    ///
    /// # Errors
    ///
    /// [`DownloadError::HashMismatch`] or [`DownloadError::ShortRead`].
    pub fn verify_part(
        &self,
        part: &Path,
        expected_bytes: u64,
        expected_sha256: &[u8; 32],
    ) -> Result<u64, DownloadError> {
        let got = part_len(part);
        if expected_bytes > 0 && got != expected_bytes {
            return Err(DownloadError::ShortRead {
                path: part.to_path_buf(),
                received: got,
                expected: expected_bytes,
            });
        }
        let actual = hash_part(part)?;
        if actual != *expected_sha256 {
            return Err(DownloadError::HashMismatch {
                path: part.to_path_buf(),
                expected: hex::encode(expected_sha256),
                actual: hex::encode(actual),
            });
        }
        Ok(got)
    }
}

impl DownloadError {
    /// Whether retrying could plausibly work.
    #[must_use]
    pub fn is_transient(&self) -> bool {
        match self {
            Self::Network { .. } | Self::ShortRead { .. } | Self::HashMismatch { .. } => true,
            Self::Http { status, .. } => *status >= 500 || *status == 408 || *status == 429,
            _ => false,
        }
    }
}

/// An error and everything that caused it, on one line.
fn chain(e: &dyn std::error::Error) -> String {
    let mut out = e.to_string();
    let mut cur = e.source();
    while let Some(c) = cur {
        out.push_str(": ");
        out.push_str(&c.to_string());
        cur = c.source();
    }
    out
}

fn io_err(path: &Path, e: std::io::Error) -> DownloadError {
    if sys::is_disk_full(&e) {
        DownloadError::DiskFull {
            path: path.to_path_buf(),
        }
    } else {
        DownloadError::Io {
            path: path.to_path_buf(),
            message: e.to_string(),
        }
    }
}

fn part_len(p: &Path) -> u64 {
    std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)
}

fn open_for_append(p: &Path, append: bool) -> Result<std::fs::File, DownloadError> {
    let mut f = std::fs::OpenOptions::new();
    f.write(true).create(true);
    if append {
        f.append(true);
    } else {
        f.truncate(true);
    }
    f.open(p).map_err(|e| io_err(p, e))
}

fn hash_part(p: &Path) -> Result<[u8; 32], DownloadError> {
    use std::io::Read;
    let mut f = std::fs::File::open(p).map_err(|e| io_err(p, e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; crate::verify::DEFAULT_BUFFER_BYTES];
    loop {
        let n = f.read(&mut buf).map_err(|e| io_err(p, e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().into())
}

fn header_str(h: &HeaderMap, name: &str) -> Option<String> {
    h.get(name)
        .and_then(|v| HeaderValue::to_str(v).ok())
        .map(str::to_string)
}

/// A URL with any query string removed.
///
/// Hugging Face's CDN redirect carries a presigned signature in the query, and
/// a signature is a credential. Error messages get the path and nothing else.
#[must_use]
pub fn redact_url(url: &str) -> String {
    match url.split_once('?') {
        Some((base, _)) => format!("{base}?<redacted>"),
        None => url.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_secret_never_prints_itself() {
        let s = Redacted::new("hf_EXAMPLESECRET");
        assert!(!format!("{s:?}").contains("EXAMPLESECRET"));
        assert!(!format!("{s}").contains("EXAMPLESECRET"));
        assert!(!format!("{s:?} {s}").contains("EXAMPLESECRET"));
        assert_eq!(s.expose(), "hf_EXAMPLESECRET");
        assert!(!s.is_empty());
        assert!(Redacted::new("").is_empty());
    }

    #[test]
    fn a_presigned_query_string_is_redacted_from_messages() {
        // A CDN signature is a credential, and a log line is the worst place for
        // one.
        let redacted = redact_url("https://cdn.example.org/f.bin?X-Amz-Signature=deadbeef");
        assert!(!redacted.contains("deadbeef"), "{redacted}");
        assert!(redacted.contains("?<redacted>"), "{redacted}");
        assert_eq!(redacted, "https://cdn.example.org/f.bin?<redacted>");
        // A URL with no query is left alone: there is nothing to hide, and
        // appending a marker would make the two forms harder to compare.
        assert_eq!(
            redact_url("https://cdn.example.org/f.bin"),
            "https://cdn.example.org/f.bin"
        );
    }

    #[test]
    fn backoff_grows_and_is_capped() {
        let p = DownloadPolicy {
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_millis(800),
            ..DownloadPolicy::default()
        };
        let got: Vec<u64> = (0..6).map(|a| p.delay_for(a).as_millis() as u64).collect();
        assert_eq!(got, vec![100, 200, 400, 800, 800, 800]);
        // A huge attempt number must not overflow into a tiny delay.
        assert_eq!(p.delay_for(u32::MAX), p.max_delay);
    }

    #[test]
    fn only_transient_failures_are_retried() {
        for e in [
            DownloadError::Network {
                url: "u".into(),
                message: "m".into(),
            },
            DownloadError::ShortRead {
                path: PathBuf::from("p"),
                received: 1,
                expected: 2,
            },
            DownloadError::HashMismatch {
                path: PathBuf::from("p"),
                expected: "a".into(),
                actual: "b".into(),
            },
        ] {
            assert!(e.is_transient(), "{e:?}");
        }
        for s in [500u16, 502, 503, 408, 429] {
            assert!(
                DownloadError::Http {
                    url: "u".into(),
                    status: s
                }
                .is_transient(),
                "HTTP {s} should be retried"
            );
        }
        for s in [400u16, 401, 403, 404, 410, 416] {
            assert!(
                !DownloadError::Http {
                    url: "u".into(),
                    status: s
                }
                .is_transient(),
                "HTTP {s} should not be retried"
            );
        }
        assert!(!DownloadError::NoRangeSupport { url: "u".into() }.is_transient());
        assert!(!DownloadError::Cancelled.is_transient());
    }

    #[test]
    fn download_failures_map_to_the_right_user_facing_code() {
        use vaani_core::ErrorCode;
        let cases: Vec<(DownloadError, ErrorCode)> = vec![
            (DownloadError::Cancelled, ErrorCode::Cancelled),
            (
                DownloadError::NotFound { url: "u".into() },
                ErrorCode::ModelUnavailable,
            ),
            (
                DownloadError::AuthRequired { url: "u".into() },
                ErrorCode::AuthRequired,
            ),
            (
                DownloadError::AccessNotGranted { url: "u".into() },
                ErrorCode::AccessNotGranted,
            ),
            (
                DownloadError::HashMismatch {
                    path: PathBuf::from("p"),
                    expected: "a".into(),
                    actual: "b".into(),
                },
                ErrorCode::ModelCorrupt,
            ),
            (
                DownloadError::InsufficientDisk { needed: 1, free: 0 },
                ErrorCode::InsufficientDisk,
            ),
            (
                DownloadError::DiskFull {
                    path: PathBuf::from("p"),
                },
                ErrorCode::InsufficientDisk,
            ),
            (
                DownloadError::NoRangeSupport { url: "u".into() },
                ErrorCode::NetworkFailed,
            ),
        ];
        for (e, want) in cases {
            let got = e.to_vaani_error();
            assert_eq!(got.code(), want, "for {e:?}");
            if want != ErrorCode::Cancelled {
                assert!(got.user.recovery.is_some(), "{e:?} has no recovery");
            }
        }
    }

    #[test]
    fn the_precheck_refuses_when_the_disk_is_short_and_says_by_how_much() {
        let d = Downloader::with_policy(
            DownloadPolicy {
                reserve_bytes: 1000,
                ..DownloadPolicy::default()
            },
            Arc::new(FixedFreeSpace(Some(5_000))),
        );
        let e = d.precheck(Path::new("C:/models"), 10_000).unwrap_err();
        match e {
            DownloadError::InsufficientDisk { needed, free } => {
                assert_eq!(needed, 11_000, "reserve must be included");
                assert_eq!(free, 5_000);
            }
            other => panic!("expected InsufficientDisk, got {other:?}"),
        }
        // Just enough passes.
        assert_eq!(d.precheck(Path::new("C:/models"), 4_000), Ok(Some(5_000)));
    }

    #[test]
    fn an_unknown_free_space_figure_is_not_a_refusal() {
        // A platform that cannot answer must not block every install.
        let d = Downloader::with_policy(DownloadPolicy::default(), Arc::new(FixedFreeSpace(None)));
        assert_eq!(d.precheck(Path::new("C:/models"), u64::MAX / 2), Ok(None));
    }

    #[test]
    fn a_partial_path_carries_the_pid_so_two_processes_cannot_collide() {
        let a = Downloader::part_path(Path::new("C:/m/sravaani"), "encoder.fp16.onnx");
        assert!(a.starts_with("C:\\m\\sravaani\\.partial"), "{a:?}");
        assert!(a.ends_with(format!("encoder.fp16.onnx.{}.part", std::process::id())));
        // A nested manifest path is flattened: a `.part` name containing a
        // separator would create a directory that does not exist.
        let b = Downloader::part_path(Path::new("C:/m/x"), "fe/b.bin");
        assert_eq!(
            b.file_name().unwrap().to_string_lossy(),
            format!("b.bin.{}.part", std::process::id())
        );
    }

    #[test]
    fn verifying_an_existing_part_needs_no_network() {
        let p = std::env::temp_dir().join(format!("vaani-a7-part-{}.bin", std::process::id()));
        std::fs::write(&p, b"abc").unwrap();
        let d = Downloader::new();
        let want = crate::manifest::parse_sha256(
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )
        .unwrap();
        assert_eq!(d.verify_part(&p, 3, &want), Ok(3));
        // Wrong size is caught before the hash.
        assert!(matches!(
            d.verify_part(&p, 4, &want),
            Err(DownloadError::ShortRead { .. })
        ));
        // Right size, wrong bytes.
        assert!(matches!(
            d.verify_part(&p, 3, &[0u8; 32]),
            Err(DownloadError::HashMismatch { .. })
        ));
        let _ = std::fs::remove_file(&p);
    }
}

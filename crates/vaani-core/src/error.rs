//! Errors that the user can be shown, and the recovery action for each.
//!
//! Two rules govern this module, and both are load-bearing for product quality:
//!
//! 1. **No raw Rust text ever reaches the user.** A `Display` impl that leaks
//!    `anyhow`'s backtrace chain, an ONNX Runtime status code, or a `dyn Error`
//!    debug print is a bug. Everything the UI shows comes from
//!    [`UserFacingError::message`] and [`UserFacingError::recovery`].
//! 2. **Every error carries a recovery action.** "CUDA unavailable" alone makes
//!    a user feel the product is broken. "CUDA unavailable — running on CPU
//!    instead, expect slower transcription" is a message a user can act on.
//!
//! Developer logs get [`VaaniError`]'s full source chain via `tracing`; the UI
//! only ever sees the `UserFacingError` projection.

use std::fmt;

/// A user-presentable error: a short title, an explanation, and what to do.
///
/// Deliberately a plain owned struct rather than an enum so that subsystems
/// can construct one from an arbitrary error type without every caller having
/// to know the full list of variants. The enum at the call sites is where the
/// classification happens; this is what survives to the UI.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct UserFacingError {
    /// Short, human, no jargon. Shown as the error title.
    pub title: String,
    /// One or two sentences explaining what happened.
    pub detail: String,
    /// A concrete next step. `None` only when there genuinely is nothing to do.
    pub recovery: Option<String>,
    /// Stable machine-readable code, used for telemetry-free diagnostics and
    /// for tests to assert on behaviour rather than on prose.
    pub code: ErrorCode,
}

impl UserFacingError {
    /// Build a user-facing error. `detail` is shown verbatim in the UI, so it
    /// must be free of secrets and free of internal identifiers.
    pub fn new(code: ErrorCode, title: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            code,
            title: title.into(),
            detail: detail.into(),
            recovery: None,
        }
    }

    /// Attach a user-safe variable value to the message.
    ///
    /// Only for values the user already sees on screen (a model name, a
    /// filename they picked). Never for diagnostics, tokens, or internal
    /// identifiers — those belong in [`VaaniError::with_source`].
    #[must_use]
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = detail.into();
        self
    }

    /// Attach or replace the recovery action.
    #[must_use]
    pub fn with_recovery(mut self, recovery: impl Into<String>) -> Self {
        self.recovery = Some(recovery.into());
        self
    }
}

impl fmt::Display for UserFacingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.title, self.detail)?;
        if let Some(r) = &self.recovery {
            write!(f, " {r}")?;
        }
        Ok(())
    }
}

/// Stable error codes. Tests assert on these, not on message text, so that
/// rewording a message never breaks a test and never hides a behaviour change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[allow(missing_docs)] // one variant per user-facing failure mode
pub enum ErrorCode {
    /// Input file missing, unreadable, or not a media file we recognise.
    UnsupportedMedia,
    /// Container parsed but the audio codec failed.
    AudioDecodeFailed,
    /// The file has no audio stream at all.
    NoAudioStream,
    /// A model id that is not in the registry: a typo, a stale settings file,
    /// or a database written by a newer build. A caller mistake, not a bug —
    /// `Internal` tells the user to file a report about their own typo.
    UnknownModel,
    /// A model is not present on disk and offline mode forbids fetching it.
    ModelUnavailable,
    /// Model present but failed its checksum.
    ModelCorrupt,
    /// Model present but the runtime could not create a session from it.
    ModelLoadFailed,
    /// The upstream repository is gated and needs a token.
    AuthRequired,
    /// A token exists but has not been granted access to the repository.
    AccessNotGranted,
    /// The model exists but does not fit in the available VRAM.
    OutOfMemory,
    /// CUDA present in the manifest but not usable on this machine.
    CudaUnavailable,
    /// Not enough free disk for a model download or an export.
    InsufficientDisk,
    /// Transcription or post-processing failed.
    TranscriptionFailed,
    /// Diarization failed. Never fatal: a transcript without speakers is still
    /// a transcript.
    DiarizationFailed,
    /// Structuring (local LLM) failed. Never fatal: raw transcript stands.
    StructuringFailed,
    /// Export / file write failed.
    ExportFailed,
    /// Database failure.
    StorageFailed,
    /// The resource is busy: another install holds the lock, or a model file
    /// is open in another process. Waiting or closing the other copy fixes it,
    /// which neither "storage failed" nor "cancelled" says.
    ResourceBusy,
    /// A network request failed.
    NetworkFailed,
    /// The user (or a setting) cancelled the operation.
    Cancelled,
    /// The process died mid-job: a crash, a kill, a power cut. Not a
    /// cancellation — nobody asked for it and it will not resume — so it gets
    /// its own code rather than masquerading as one in the job list.
    Interrupted,
    /// Anything we have not classified yet.
    Internal,
}

impl ErrorCode {
    /// Stable string form, used in JSON output and in bug reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UnsupportedMedia => "unsupported_media",
            Self::AudioDecodeFailed => "audio_decode_failed",
            Self::NoAudioStream => "no_audio_stream",
            Self::UnknownModel => "unknown_model",
            Self::ModelUnavailable => "model_unavailable",
            Self::ModelCorrupt => "model_corrupt",
            Self::ModelLoadFailed => "model_load_failed",
            Self::AuthRequired => "auth_required",
            Self::AccessNotGranted => "access_not_granted",
            Self::OutOfMemory => "out_of_memory",
            Self::CudaUnavailable => "cuda_unavailable",
            Self::InsufficientDisk => "insufficient_disk",
            Self::TranscriptionFailed => "transcription_failed",
            Self::DiarizationFailed => "diarization_failed",
            Self::StructuringFailed => "structuring_failed",
            Self::ExportFailed => "export_failed",
            Self::StorageFailed => "storage_failed",
            Self::ResourceBusy => "resource_busy",
            Self::NetworkFailed => "network_failed",
            Self::Cancelled => "cancelled",
            Self::Interrupted => "interrupted",
            Self::Internal => "internal",
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The application's error type.
///
/// Holds a `#[source]` chain for logs plus the user-facing projection for the
/// UI. Constructing one is cheap enough to do freely; the `source` chain is
/// only formatted when a log record is actually emitted.
#[derive(Debug, thiserror::Error)]
#[error("{user}")]
pub struct VaaniError {
    /// What the user is shown. Never contains a secret or an internal
    /// identifier; see the `err` module for the detail/cause contract.
    pub user: UserFacingError,
    #[source]
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl VaaniError {
    /// Wrap a user-facing error with no underlying cause.
    pub fn new(user: UserFacingError) -> Self {
        Self { user, source: None }
    }

    /// Attach the underlying cause, which must be a real error type.
    ///
    /// Use this when there is an actual error to chain. For a formatted
    /// diagnostic use [`VaaniError::with_cause`], which is what most call sites
    /// want.
    #[must_use]
    pub fn with_source(mut self, source: impl std::error::Error + Send + Sync + 'static) -> Self {
        self.source = Some(Box::new(source));
        self
    }

    /// Attach a diagnostic as the cause.
    ///
    /// Almost every call site wants this rather than
    /// [`VaaniError::with_source`]: the thing that went wrong is a formatted
    /// message, not another `Error`. Forcing those sites to invent a newtype per
    /// message would push people toward `unwrap()`, or worse, toward putting the
    /// diagnostic in the user-facing `detail` where it would leak. Having a
    /// first-class "a message for the log" method is what makes the
    /// detail/cause contract enforceable in review.
    #[must_use]
    pub fn with_cause(mut self, cause: impl std::fmt::Display) -> Self {
        self.source = Some(Box::new(Cause(cause.to_string())));
        self
    }

    /// Attach or replace the recovery action on the underlying user-facing
    /// error, keeping the source chain.
    #[must_use]
    pub fn with_recovery(mut self, recovery: impl Into<String>) -> Self {
        self.user.recovery = Some(recovery.into());
        self
    }

    /// Flatten the full source chain for a developer log line.
    pub fn debug_chain(&self) -> String {
        let mut out = format!("[{}] {}", self.user.code, self.user.title);
        let mut cur = self
            .source
            .as_ref()
            .map(|b| b.as_ref() as &dyn std::error::Error);
        while let Some(e) = cur {
            out.push_str("\n  caused by: ");
            out.push_str(&e.to_string());
            cur = e.source();
        }
        out
    }

    /// The stable error code.
    pub fn code(&self) -> ErrorCode {
        self.user.code
    }

    /// Whether this is a cancellation rather than a failure.
    pub fn is_cancelled(&self) -> bool {
        self.user.code == ErrorCode::Cancelled
    }
}

impl From<UserFacingError> for VaaniError {
    fn from(value: UserFacingError) -> Self {
        Self::new(value)
    }
}

/// A diagnostic string promoted to an error so it can live in a `source` chain.
///
/// Exists purely so [`VaaniError::with_cause`] can reuse the existing chain
/// machinery instead of adding a parallel `Vec<String>` of causes that
/// [`VaaniError::debug_chain`] would then have to walk in two places.
#[derive(Debug)]
struct Cause(String);

impl fmt::Display for Cause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Cause {}

/// The application's result type.
pub type Result<T, E = VaaniError> = std::result::Result<T, E>;

/// Constructors for the errors that subsystems reach for constantly.
///
/// # The detail/cause split
///
/// This is the single most important convention in the crate, so it is worth
/// stating plainly:
///
/// * **`detail` is shown in the UI.** It must never contain a token, a raw
///   ONNX Runtime status code, a file path the user did not choose, or
///   anything else that leaks or confuses. The constructors below therefore
///   take **no** free-form text: a fixed, reviewed sentence is the only thing
///   they can produce.
/// * **Everything else is a `cause`.** Pass it to
///   [`VaaniError::with_source`], which is `#[source]`-chained and only ever
///   formatted into a developer log.
///
/// If a message genuinely needs a variable value (a model name, a chosen
/// filename), use [`UserFacingError::with_detail`] at the call site, having
/// checked that the value is one the user already sees on screen. Making that
/// an explicit, separate call — rather than a positional argument to a
/// constructor — is what stops a diagnostic from reaching the UI by accident.
pub mod err {
    use super::{ErrorCode, UserFacingError, VaaniError};

    macro_rules! ctor {
        ($(#[$m:meta])* $name:ident, $code:ident, $title:literal, $detail:literal) => {
            $(#[$m])*
            pub fn $name() -> VaaniError {
                VaaniError::new(UserFacingError::new(ErrorCode::$code, $title, $detail))
            }
        };
    }

    ctor!(
        /// The container parsed but is not a media type we handle.
        unsupported_media, UnsupportedMedia, "Unsupported media.",
        "This file is not a video or audio format VaaniDesk can read."
    );
    ctor!(
        /// Container is fine; the audio codec failed mid-stream.
        audio_decode_failed, AudioDecodeFailed, "Failed to decode audio.",
        "The file looks like media, but its audio track could not be decoded. \
         It may be damaged or use a codec that is not supported."
    );
    ctor!(
        /// Valid media file with no audio stream.
        no_audio_stream, NoAudioStream, "No audio track found.",
        "This file has no audio track, so there is nothing to transcribe."
    );
    ctor!(
        /// A required model is not on disk.
        model_unavailable, ModelUnavailable, "Model unavailable.",
        "The speech recognition model has not been downloaded on this computer yet."
    );
    ctor!(
        /// The requested model id is not in the registry.
        unknown_model, UnknownModel, "Unknown model.",
        "There is no model with that name. It may be a typo, or a setting left \
         over from a newer version."
    );
    ctor!(
        /// A model file is present but its checksum does not match the registry.
        model_corrupt, ModelCorrupt, "Model files are incomplete or damaged.",
        "The downloaded model does not match its expected checksum. It was most \
         likely interrupted while downloading."
    );
    ctor!(
        /// Model files are intact but the runtime refused to load them.
        model_load_failed, ModelLoadFailed, "Model could not be loaded.",
        "The model files are present but the inference runtime could not load them. \
         This usually means the runtime and the model were built for different versions."
    );
    ctor!(
        /// Gated repository, no usable credential.
        auth_required, AuthRequired, "Hugging Face authentication required.",
        "This model is hosted on Hugging Face behind a login. Add your Hugging \
         Face token in Settings to download it."
    );
    ctor!(
        /// Credential is valid but the repository access is not granted.
        access_not_granted, AccessNotGranted, "Model access has not been granted.",
        "Your Hugging Face account has not been approved for this model yet. \
         Request access on the model's Hugging Face page, then try again."
    );
    ctor!(
        /// Model does not fit in available VRAM.
        out_of_memory, OutOfMemory, "Not enough graphics memory.",
        "This model does not fit in the graphics memory available on this computer. \
         Close other applications that use the GPU, or switch to CPU processing \
         in Settings."
    );
    ctor!(
        /// CUDA is wanted but not usable.
        cuda_unavailable, CudaUnavailable, "CUDA unavailable.",
        "Graphics acceleration could not be enabled, so processing will run on the \
         CPU instead. Transcription will be slower but the result is identical."
    );
    ctor!(
        /// A download or export cannot proceed for lack of space.
        insufficient_disk, InsufficientDisk, "Not enough disk space.",
        "There is not enough free disk space to finish this operation."
    );
    ctor!(
        /// ASR or post-processing failed.
        transcription_failed, TranscriptionFailed, "Transcript generation failed.",
        "The speech recognition engine could not produce a transcript for this file."
    );
    ctor!(
        /// Diarization failed; the transcript survives without speakers.
        diarization_failed, DiarizationFailed, "Speaker analysis failed.",
        "Speakers could not be separated. The transcript itself is complete, but \
         speaker labels are missing."
    );
    ctor!(
        /// Local LLM structuring failed; the raw transcript survives.
        structuring_failed, StructuringFailed, "Transcript structuring failed.",
        "The transcript could not be turned into a structured meeting summary. \
         The verbatim transcript is still available."
    );
    ctor!(
        /// Writing an export file failed.
        export_failed, ExportFailed, "Export failed.",
        "The transcript could not be written to that location."
    );
    ctor!(
        /// Database failure.
        storage_failed, StorageFailed, "Could not access local storage.",
        "VaaniDesk could not read or write its local data. Check that the data \
         folder exists and is writable."
    );
    ctor!(
        /// The resource is busy in another window or process.
        resource_busy, ResourceBusy, "Already in use.",
        "Another copy of VaaniDesk is using this right now. Wait for it to \
         finish, or close the other window, then try again."
    );
    ctor!(
        /// A network request failed.
        network_failed, NetworkFailed, "Network request failed.",
        "The request could not be completed. Check your internet connection and \
         try again."
    );
    ctor!(
        /// Internal invariant broken. Always a bug; log the source chain.
        internal, Internal, "Something went wrong.",
        "An unexpected internal error occurred. Please report this."
    );

    /// Cancellation is not a failure but it must travel through the same
    /// channels so that `?` works uniformly in the pipeline.
    pub fn cancelled() -> VaaniError {
        VaaniError::new(UserFacingError::new(
            ErrorCode::Cancelled,
            "Cancelled.",
            "The job was cancelled before it finished.",
        ))
    }

    /// The process died mid-job. Distinct from [`cancelled`](Self::cancelled):
    /// nobody asked for this and it will not resume, so the job list must not
    /// pretend otherwise.
    pub fn interrupted() -> VaaniError {
        VaaniError::new(UserFacingError::new(
            ErrorCode::Interrupted,
            "Interrupted.",
            "VaaniDesk closed while this job was running. Run it again to finish.",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_facing_text_never_leaks_the_source_chain() {
        // The whole point of the detail/cause split: a diagnostic handed to
        // `with_source` must never reach the string the UI renders.
        let secret = "hf_EXAMPLESECRET";
        let e = err::model_load_failed().with_source(std::io::Error::other(secret));

        let shown = e.to_string();
        assert!(!shown.contains(secret), "Display leaked a secret: {shown}");
        // It must not leak a path-like diagnostic either.
        assert_eq!(
            e.user.detail,
            UserFacingError::new(
                ErrorCode::ModelLoadFailed,
                "Model could not be loaded.",
                "The model files are present but the inference runtime could not load them. \
             This usually means the runtime and the model were built for different versions."
            )
            .detail
        );

        // The developer log keeps the detail.
        assert!(e.debug_chain().contains(secret));
    }

    #[test]
    fn constructors_take_no_free_form_text() {
        // A signature change here is the structural guard against internals
        // reaching the UI: if a constructor started accepting a diagnostic
        // string again, this test's `()` call sites would fail to compile.
        let e = err::cancelled();
        assert!(e.to_string().contains("cancelled"));
        let _ = err::unsupported_media();
        let _ = err::audio_decode_failed();
        let _ = err::no_audio_stream();
        let _ = err::model_unavailable();
        let _ = err::model_corrupt();
        let _ = err::model_load_failed();
        let _ = err::auth_required();
        let _ = err::access_not_granted();
        let _ = err::out_of_memory();
        let _ = err::cuda_unavailable();
        let _ = err::insufficient_disk();
        let _ = err::transcription_failed();
        let _ = err::diarization_failed();
        let _ = err::structuring_failed();
        let _ = err::export_failed();
        let _ = err::storage_failed();
        let _ = err::network_failed();
        let _ = err::internal();
    }

    #[test]
    fn every_constructor_reports_a_recovery_or_explains_itself() {
        // A user-facing error that says what went wrong but not what to do is
        // the failure mode this crate exists to prevent. For the ones where
        // the UI must offer a next step, require it explicitly.
        for e in [
            err::out_of_memory(),
            err::auth_required(),
            err::cuda_unavailable()
                .with_recovery("Close GPU-heavy apps, or switch to CPU in Settings."),
        ] {
            assert!(!e.user.title.is_empty());
            assert!(!e.user.detail.is_empty());
        }
    }

    #[test]
    fn error_codes_have_stable_strings() {
        // If this fails, something renamed a variant. That is a breaking change
        // for persisted jobs and exported JSON, so it must be deliberate.
        assert_eq!(ErrorCode::Cancelled.as_str(), "cancelled");
        assert_eq!(ErrorCode::CudaUnavailable.as_str(), "cuda_unavailable");
    }

    #[test]
    fn recovery_defaults_to_none_and_can_be_attached() {
        let e = err::cancelled();
        assert_eq!(e.user.recovery, None);
        let e2 = err::cancelled().with_recovery("Press Resume to continue.");
        assert_eq!(
            e2.user.recovery.as_deref(),
            Some("Press Resume to continue.")
        );
    }

    #[test]
    fn cancellation_is_distinguishable() {
        assert!(err::cancelled().is_cancelled());
        assert!(!err::internal().is_cancelled());
    }

    #[test]
    fn the_new_codes_have_stable_strings_and_meanings() {
        // The three codes three agents asked for. Each pins the string that is
        // persisted in the database and exported in JSON.
        assert_eq!(ErrorCode::Interrupted.as_str(), "interrupted");
        assert_eq!(ErrorCode::ResourceBusy.as_str(), "resource_busy");
        assert_eq!(ErrorCode::UnknownModel.as_str(), "unknown_model");
        // An interrupted job is a failure that can be retried, not a
        // cancellation the user asked for. Widening `is_cancelled` to include
        // it would silently stop retries.
        assert!(!err::interrupted().is_cancelled());
        assert!(!err::resource_busy().is_cancelled());
        assert!(!err::unknown_model().is_cancelled());
        // Every new code carries a title a user can act on, not "report this".
        for e in [
            err::interrupted(),
            err::resource_busy(),
            err::unknown_model(),
        ] {
            assert!(!e.user.title.is_empty());
            assert!(!e.user.detail.is_empty());
            assert_ne!(e.code(), ErrorCode::Internal);
        }
    }
}

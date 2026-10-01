//! Translation from I/O and demuxer failures into `vaani_core::Error`.
//!
//! # Why this module exists
//!
//! `vaani_core`'s error taxonomy is the product's contract with the user: every
//! error carries a fixed, reviewed sentence for the UI plus a recovery action,
//! and every diagnostic goes into the `#[source]` chain where only a developer
//! log can see it. The temptation with a library like Symphonia is to write
//! `Err(err::internal().with_cause(e))` at every `?`, which produces a UI that
//! says "Something went wrong. Please report this." for a file the user simply
//! moved to another folder. That is the failure mode this module exists to
//! prevent, so the mapping is written down once, here, and every failure path in
//! [`crate::probe`] and [`crate::decode`] goes through it.
//!
//! # The mapping
//!
//! | What happened | Code | Why |
//! |---|---|---|
//! | Path does not exist | `UnsupportedMedia` | `ErrorCode::UnsupportedMedia` is documented as "input file missing, unreadable, or not a media file we recognise" |
//! | Path is a directory, or permission denied | `UnsupportedMedia` | same, plus a recovery that names the real cause |
//! | The file is zero bytes | `UnsupportedMedia` | there is nothing to recognise, and "unsupported media" is the taxonomy's closest bucket |
//! | No demuxer recognised the bytes | `UnsupportedMedia` | the file is not media we can read |
//! | Container parsed, no audio track | `NoAudioStream` | the container is fine; there is simply nothing to transcribe |
//! | Container parsed, audio codec has no decoder | `AudioDecodeFailed` | the file *is* media; only its audio codec is outside the set Symphonia ships. The recovery names a format that works, which is the only thing the user can do |
//! | Malformed data | `AudioDecodeFailed` | the same message, plus "it may still be downloading" |
//! | A read hit EOF early | `AudioDecodeFailed` | a distinct recovery: this one is usually "wait for the copy to finish" |
//! | The stream ended short of the container's claim | `AudioDecodeFailed` | the same recovery; see [`short_of_claim`] for why the demuxer raises no error of its own here |
//! | Any other read error | `AudioDecodeFailed` | recovery names the disk and the file |
//! | A stream claimed a sample rate of zero | `AudioDecodeFailed` | a damaged header, and "copy the file again" is a real action. `Internal` would tell a user with a half-copied download to report a bug |
//! | No resampler can be built for the rate pair | `UnsupportedMedia` | the recovery names a standard rate |
//! | Everything decoded, but zero audio frames | `AudioDecodeFailed` | an empty buffer would crash a downstream stage |
//! | A bug in this crate | `Internal` | carries only a diagnostic |
//! | Caller cancelled | `Cancelled` | produced by `CancelToken::check`, never constructed here |
//!
//! # The detail/cause discipline
//!
//! No function here puts a diagnostic into the user-facing `detail`.
//! `vaani-core` enforces that by giving its constructors no free-form parameters,
//! and this module keeps it: the only variable text that ever reaches `detail` is
//! the *file name*, which the user chose and can already see. Absolute paths,
//! codec IDs, byte offsets and error strings all go to the cause.

use std::io;
use std::path::Path;

use symphonia::core::errors::Error as SymphoniaError;
use vaani_core::{UserFacingError, VaaniError, err};

/// Attach a detail, a recovery and a cause to a `vaani_core` error constructor.
///
/// Written as a helper rather than repeated at each call site because
/// `VaaniError` has no `with_detail`: only `UserFacingError` does, and every one
/// of these functions needs both a rewritten detail and a recovery, so building
/// the user half and then wrapping it is the only way to keep the two halves
/// from drifting apart.
fn compose(base: VaaniError, detail: Option<String>, recovery: &str, cause: String) -> VaaniError {
    let user: UserFacingError = base.user;
    let user = match detail {
        Some(d) => user.with_detail(d),
        None => user,
    };
    VaaniError::new(user.with_recovery(recovery)).with_cause(cause)
}

/// The file could not be opened, read, or is empty.
///
/// `ErrorCode::UnsupportedMedia` is the taxonomy's bucket for "input file
/// missing, unreadable, or not a media file we recognise", which is exactly the
/// set of things that can go wrong here. So a missing file is a user-facing error
/// with a recovery action, and never an internal one.
pub(crate) fn open_failed(path: &Path, source: &io::Error) -> VaaniError {
    let file = display_path(path);
    let recovery = match source.kind() {
        io::ErrorKind::NotFound => {
            "Check that the file is still in the folder you picked, then try again."
        }
        io::ErrorKind::PermissionDenied => {
            "Check that VaaniDesk is allowed to read the file, and that it is not open \
             exclusively in another program."
        }
        _ => "Check that the file still exists and that the disk holding it is healthy.",
    };
    compose(
        err::unsupported_media(),
        Some(format!(
            "This file is not a video or audio format VaaniDesk can read ({file})."
        )),
        recovery,
        format!("open({file}) failed: {source}"),
    )
}

/// The bytes at `path` did not match any demuxer we support, or there were none.
pub(crate) fn unrecognised_container(path: &Path, source: &SymphoniaError) -> VaaniError {
    let file = display_path(path);
    compose(
        err::unsupported_media(),
        Some(format!(
            "This file is not a video or audio format VaaniDesk can read ({file})."
        )),
        "Re-export the recording as MP4, MKV, MP3, M4A, WAV, FLAC or OGG, then try again.",
        format!("no demuxer matched {file}: {source}"),
    )
}

/// The container has no audio track to decode.
pub(crate) fn no_audio_track(path: &Path) -> VaaniError {
    compose(
        err::no_audio_stream(),
        None,
        "Choose a file that has an audio track, for example a recording rather than a slide deck.",
        format!("{} contains no audio track", display_path(path)),
    )
}

/// The audio track uses a codec this build has no decoder for.
///
/// Classified as `AudioDecodeFailed` rather than `UnsupportedMedia` because the
/// file *is* media VaaniDesk recognises as media; only its audio codec is outside
/// the set Symphonia ships. The recovery names a format that works, which is the
/// only thing the user can actually do about it.
pub(crate) fn unsupported_codec(path: &Path, codec: &str) -> VaaniError {
    compose(
        err::audio_decode_failed(),
        Some(format!(
            "The audio track in this file uses the {codec} codec, which VaaniDesk cannot decode."
        )),
        "Re-export the recording with MP3, AAC or WAV audio, then try again.",
        format!(
            "{}: no decoder registered for audio codec {codec}",
            display_path(path)
        ),
    )
}

/// A demuxer or decoder failed on the data itself.
pub(crate) fn malformed(path: &Path, source: &SymphoniaError) -> VaaniError {
    compose(
        err::audio_decode_failed(),
        None,
        "The file may be damaged, or it may still be downloading or copying. \
         Wait for it to finish and try again.",
        format!("{}: {source}", display_path(path)),
    )
}

/// A read failed part-way through, most often because the file is shorter than
/// its header claims.
///
/// Kept distinct from [`malformed`] only in the message, because the user action
/// is different: this one is usually "wait for the copy to finish".
pub(crate) fn truncated(path: &Path, source: &SymphoniaError) -> VaaniError {
    compose(
        err::audio_decode_failed(),
        Some("This file ends before its audio track does.".to_string()),
        "It may still be downloading or copying. Wait for that to finish, then try again.",
        format!("{}: {source}", display_path(path)),
    )
}

/// Any other I/O failure while reading the source.
pub(crate) fn read_failed(path: &Path, source: &io::Error) -> VaaniError {
    compose(
        err::audio_decode_failed(),
        None,
        "Check that the file is still readable and that the disk holding it is healthy.",
        format!("read({}) failed: {source}", display_path(path)),
    )
}

/// A container declared a sample rate of zero.
///
/// A damaged header, and the user *can* act on that, so this is deliberately not
/// `Internal`. "Please report this" for a file that arrived half-copied is the
/// exact failure this module exists to prevent, and the fix is the same one as
/// for any other short read: copy it again.
pub(crate) fn zero_sample_rate(path: &Path) -> VaaniError {
    compose(
        err::audio_decode_failed(),
        Some("This file's audio header does not say what sample rate it uses.".to_string()),
        "The file looks damaged or incomplete. Copy it again, or re-export the recording.",
        format!(
            "{}: audio track declares a sample rate of 0 Hz",
            display_path(path)
        ),
    )
}

/// The stream ended cleanly, but well short of what the container promised.
///
/// Reached only when the demuxer stopped without an error, which is what a
/// partially copied file does when the container's own length fields are what is
/// wrong: a WAV whose `data` chunk claims ten hours and holds one second is a
/// valid, readable, *short* file as far as the demuxer is concerned, so nothing
/// upstream ever raises an error. Without this the caller would see a clean
/// report and a transcript that is missing 99.99% of the meeting with nothing to
/// say so.
pub(crate) fn short_of_claim(path: &Path, declared_ms: i64, decoded_ms: i64) -> VaaniError {
    compose(
        err::audio_decode_failed(),
        Some("This file's audio stops well before the recording does.".to_string()),
        "It looks like an incomplete copy. Copy the file again, or re-export the recording.",
        format!(
            "{}: container declared {declared_ms} ms of audio, {decoded_ms} ms was present",
            display_path(path)
        ),
    )
}

/// A sample rate pair outside anything a real recorder produces.
pub(crate) fn unsupported_sample_rate(rate: u32, target: u32) -> VaaniError {
    compose(
        err::unsupported_media(),
        Some(format!(
            "This file's audio has an unusual sample rate ({rate} Hz) that VaaniDesk cannot resample."
        )),
        "Re-export the recording at a standard rate: 8, 16, 22.05, 32, 44.1, 48 or 96 kHz.",
        format!("no resampler could be built for {rate} Hz -> {target} Hz"),
    )
}

/// Everything parsed and decoded without error, but produced no audio at all.
///
/// A zero-length `AudioBuffer` is not a legitimate hand-off: `AudioBuffer::duration`
/// returns `Time::ZERO`, every chunker produces zero chunks, and the job either
/// hangs or reports an empty transcript. Failing here, with a message that names
/// the file, is strictly better.
pub(crate) fn no_audio_decoded(path: &Path) -> VaaniError {
    compose(
        err::audio_decode_failed(),
        Some(
            "No audio could be read from this file, even though it has an audio track.".to_string(),
        ),
        "Try re-exporting the recording, or open it in another program to check it plays.",
        format!("{}: decoded zero audio frames", display_path(path)),
    )
}

/// The file decodes to more audio than this build will hold in memory.
///
/// Its own constructor because the user's next step is genuinely different from
/// every other decode failure. "The file is damaged" and "this recording is
/// longer than the app will hold" call for opposite reactions -- one means fix
/// the file, the other means split the file -- and reporting the first when the
/// second happened sends someone to re-export a perfectly good recording.
pub(crate) fn too_long(path: &Path, decoded_ms: i64, limit_ms: i64) -> VaaniError {
    compose(
        err::resource_exhausted(),
        Some(format!(
            "This recording is longer than this build will transcribe in one \
             piece (about {}).",
            humanise_hours(limit_ms)
        )),
        "Split the recording into shorter parts and add them one at a time.",
        format!(
            "{}: decoded {decoded_ms} ms of audio, limit is {limit_ms} ms",
            display_path(path)
        ),
    )
}

/// Rough hours, for a message a person can read. Rounded down and floored at one
/// hour, because "0 hours" is not a number anyone can act on.
fn humanise_hours(limit_ms: i64) -> String {
    let hours = limit_ms / 3_600_000;
    if hours <= 1 {
        "1 hour".to_string()
    } else {
        format!("{hours} hours")
    }
}

/// A bug in this crate rather than bad input.
///
/// Every internal error here is a violated invariant, which is why it carries
/// only a diagnostic: there is nothing the user can do and nothing they should be
/// shown beyond the fixed "please report this".
pub(crate) fn internal(cause: impl std::fmt::Display) -> VaaniError {
    err::internal().with_cause(cause)
}

/// Classify a Symphonia error raised while demuxing or decoding.
pub(crate) fn from_symphonia(path: &Path, source: &SymphoniaError) -> VaaniError {
    match source {
        SymphoniaError::IoError(io) if io.kind() == io::ErrorKind::UnexpectedEof => {
            truncated(path, source)
        }
        SymphoniaError::IoError(io) => read_failed(path, io),
        SymphoniaError::Unsupported(_) => unrecognised_container(path, source),
        _ => malformed(path, source),
    }
}

/// The file name a user would recognise, for the user-facing `detail`.
///
/// `vaani_core::error` allows a chosen filename in `detail`: the user picked it,
/// so it is already on their screen. Absolute paths are not, because they leak
/// the user's directory layout into a UI that may end up in a bug report.
fn display_path(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "the selected file".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vaani_core::ErrorCode;

    #[test]
    fn a_missing_file_is_user_facing_and_actionable_not_internal() {
        // The single most important assertion in this module: a user who deleted
        // the file must not be told to report a bug.
        let e = open_failed(
            Path::new("/tmp/meeting.mp4"),
            &io::Error::new(io::ErrorKind::NotFound, "no such file"),
        );
        assert_eq!(e.code(), ErrorCode::UnsupportedMedia);
        assert!(!e.is_cancelled());
        assert!(
            e.user.recovery.is_some(),
            "every error must offer a next step"
        );
        assert!(
            e.user.detail.contains("meeting.mp4"),
            "detail: {}",
            e.user.detail
        );
        assert!(
            !e.user.detail.contains('/') && !e.user.detail.contains('\\'),
            "an absolute path leaked into the UI: {}",
            e.user.detail
        );
        assert!(
            e.debug_chain().contains("no such file"),
            "the cause must survive for logs"
        );
    }

    #[test]
    fn permission_denied_gets_its_own_recovery() {
        let e = open_failed(
            Path::new("locked.mkv"),
            &io::Error::new(io::ErrorKind::PermissionDenied, "access denied"),
        );
        let recovery = e.user.recovery.expect("recovery");
        assert!(
            recovery.contains("permission") || recovery.contains("allowed"),
            "{recovery}"
        );
    }

    #[test]
    fn every_constructor_produces_a_recovery_and_a_cause() {
        let path = Path::new("x.wav");
        let s = SymphoniaError::DecodeError("bad");
        let all = [
            open_failed(path, &io::Error::other("io")),
            unrecognised_container(path, &s),
            no_audio_track(path),
            unsupported_codec(path, "opus"),
            malformed(path, &s),
            truncated(path, &s),
            read_failed(path, &io::Error::other("io")),
            zero_sample_rate(path),
            unsupported_sample_rate(12345, 16000),
            no_audio_decoded(path),
            internal("bug"),
        ];
        for e in all {
            assert!(!e.user.title.is_empty(), "{e}");
            assert!(!e.user.detail.is_empty(), "{e}");
            assert!(!e.debug_chain().is_empty(), "{e}");
        }
    }

    #[test]
    fn an_empty_or_anonymous_path_still_produces_a_readable_detail() {
        // `std::env::current_dir()` and a bare directory both land here, and
        // "This file is not a video or audio format VaaniDesk can read ()." with
        // empty parentheses is not a message anyone should ship.
        let e = open_failed(Path::new(""), &io::Error::other("io"));
        assert!(
            e.user.detail.contains("the selected file"),
            "{}",
            e.user.detail
        );
        let e = open_failed(Path::new("/"), &io::Error::other("io"));
        assert!(
            e.user.detail.contains("the selected file"),
            "{}",
            e.user.detail
        );
    }

    #[test]
    fn an_eof_read_is_reported_as_truncation_not_as_corruption() {
        // The distinction is the whole point: a user whose file is still copying
        // needs to be told to wait, not that their file is damaged.
        let e = from_symphonia(
            Path::new("partial.mp4"),
            &SymphoniaError::IoError(io::Error::new(io::ErrorKind::UnexpectedEof, "eof")),
        );
        assert!(e.user.detail.contains("ends before"), "{}", e.user.detail);
        let recovery = e.user.recovery.expect("recovery");
        assert!(
            recovery.contains("downloading") || recovery.contains("copying"),
            "{recovery}"
        );
    }

    #[test]
    fn an_unsupported_codec_names_the_codec_and_a_format_that_works() {
        let e = unsupported_codec(Path::new("call.m4a"), "opus");
        assert_eq!(e.code(), ErrorCode::AudioDecodeFailed);
        assert!(e.user.detail.contains("opus"), "{}", e.user.detail);
        let recovery = e.user.recovery.expect("recovery");
        assert!(recovery.contains("MP3"), "{recovery}");
    }

    #[test]
    fn no_audio_stream_keeps_cores_own_message_because_it_is_already_right() {
        // `err::no_audio_stream`'s detail is exactly the situation; rewriting it
        // would only risk making it worse.
        let e = no_audio_track(Path::new("slides.mp4"));
        assert_eq!(e.code(), ErrorCode::NoAudioStream);
        assert_eq!(
            e.user.detail,
            "This file has no audio track, so there is nothing to transcribe."
        );
    }
}

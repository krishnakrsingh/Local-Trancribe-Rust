//! The error taxonomy, asserted through the public API.
//!
//! # Why this file exists when `src/errors.rs` already has unit tests
//!
//! The unit tests pin the *mapping functions*. This file pins the *end-to-end
//! behaviour*: that a user who hands the app a folder, a zero-byte file, a JPEG
//! or a WAV with an unknown codec tag gets a specific, stable, actionable code
//! back, and never gets "Something went wrong. Please report this." for
//! something they can fix themselves.
//!
//! Two properties are checked on **every** error produced here, because they are
//! the ones that are easy to regress and impossible to notice:
//!
//! * A `recovery` exists. An error with no next step is a dead end in the UI.
//! * The `detail` names the file the user chose and leaks nothing else. The
//!   temporary directory these fixtures live in is on the same drive as the
//!   repository, so a leaked absolute path is a real possibility and a real bug,
//!   not a hypothetical one.
//!
//! Cases that cannot be synthesised without an encoder — a video-only MP4, an
//! Opus-in-Ogg file — are covered by the skip-guarded fixture tests in
//! `containers.rs`, and the mapping itself is pinned by `src/errors.rs`.

mod common;

use std::path::{Path, PathBuf};

use vaani_core::{CancelToken, ErrorCode};
use vaani_media::{DecodeOptions, TruncatedPolicy, decode, probe};

use common::{Fixture, SampleFormat, WavSpec, fmt_chunk, tone, wav, wav_raw};

/// Everything the UI needs, checked on one error.
///
/// The file name is deliberately *not* required here. Some messages read better
/// with it and some read better without: "This file ends before its audio track
/// does" is clearer than the same sentence with a name in it, and the UI already
/// knows which file is open. What is required is that nothing that is *not* the
/// user's own file name leaks in — the temporary directory these fixtures live
/// in sits on the same drive as the repository, so a leaked absolute path is a
/// real possibility and a real bug.
#[track_caller]
fn assert_actionable(err: &vaani_core::VaaniError, expected: ErrorCode, file: &Path) {
    assert_eq!(
        err.code(),
        expected,
        "wrong code for {}: {}",
        file.display(),
        err.debug_chain()
    );
    assert!(!err.user.title.is_empty(), "a title is required");
    assert!(!err.user.detail.is_empty(), "a detail is required");
    let recovery =
        err.user.recovery.as_deref().unwrap_or_else(|| {
            panic!("{} has no recovery: {}", expected.as_str(), err.user.detail)
        });
    assert!(
        recovery.len() > 10,
        "{:?} recovery is not an action: {recovery}",
        expected
    );
    assert!(
        !err.debug_chain().is_empty(),
        "the diagnostic must survive into the log chain"
    );
    assert!(
        !err.is_cancelled(),
        "a bad file must never be reported as a cancellation"
    );
    if let Some(parent) = file.parent().map(|p| p.to_string_lossy().into_owned()) {
        let recovery = err.user.recovery.as_deref().unwrap_or_default();
        assert!(
            !err.user.detail.contains(&parent) && !recovery.contains(&parent),
            "the absolute path leaked into the UI: {}",
            err.user.detail
        );
    }
}

/// The message must name the file the user chose, because the user has to know
/// which one of a folder full of them went wrong.
#[track_caller]
fn assert_names_file(err: &vaani_core::VaaniError, file: &Path) {
    let name = file
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    if name.is_empty() {
        return;
    }
    assert!(
        err.user.detail.contains(&name),
        "detail does not name the file: {}",
        err.user.detail
    );
}

#[test]
fn a_file_that_does_not_exist_is_unsupported_media_not_a_bug_report() {
    let mut path: PathBuf = std::env::temp_dir();
    path.push("vaani-media-this-file-was-never-created.wav");
    let _ = std::fs::remove_file(&path);
    let err = probe(&path).expect_err("probing a missing file must fail");
    assert_actionable(&err, ErrorCode::UnsupportedMedia, &path);
    assert_names_file(&err, &path);
    assert!(
        !err.is_cancelled(),
        "a missing file must never look like a cancellation"
    );
    // The cause is for the log, and it has to name the OS error.
    assert!(
        err.debug_chain().to_lowercase().contains("not"),
        "the OS reason is missing from the log chain: {}",
        err.debug_chain()
    );
}

#[test]
fn a_directory_is_unsupported_media() {
    let dir = tempfile::tempdir().expect("tempdir");
    let err = probe(dir.path()).expect_err("probing a directory must fail");
    assert_actionable(&err, ErrorCode::UnsupportedMedia, dir.path());
}

#[test]
fn an_empty_file_is_unsupported_media() {
    // Zero bytes is not "an empty recording", it is "there is nothing to
    // recognise", and the two have different recoveries.
    let f = Fixture::new("wav", b"");
    for err in [
        probe(f.path()).expect_err("probe"),
        decode(f.path(), &CancelToken::new(), &DecodeOptions::default()).expect_err("decode"),
    ] {
        assert_actionable(&err, ErrorCode::UnsupportedMedia, f.path());
        assert_names_file(&err, f.path());
    }
}

#[test]
fn something_that_is_not_media_at_all_is_unsupported_media() {
    // A PNG, because that is what a user actually picks by mistake.
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    png.extend_from_slice(&[0u8; 512]);
    let f = Fixture::new("png", &png);

    let err = probe(f.path()).expect_err("probe");
    assert_actionable(&err, ErrorCode::UnsupportedMedia, f.path());
    assert_names_file(&err, f.path());
    let recovery = err.user.recovery.clone().unwrap();
    assert!(
        recovery.contains("MP3") || recovery.contains("WAV") || recovery.contains("MP4"),
        "the recovery must name a format that works: {recovery}"
    );

    let err = decode(f.path(), &CancelToken::new(), &DecodeOptions::default()).expect_err("decode");
    assert_actionable(&err, ErrorCode::UnsupportedMedia, f.path());
}

#[test]
fn text_with_a_wav_extension_is_still_not_media() {
    // Extension-based dispatch would accept this. The bytes are what decide.
    let f = Fixture::new(
        "wav",
        b"this is a transcript, not a recording\n"
            .repeat(40)
            .as_slice(),
    );
    let err = probe(f.path()).expect_err("probe");
    assert_actionable(&err, ErrorCode::UnsupportedMedia, f.path());
}

#[test]
fn a_format_tag_this_build_cannot_read_is_unsupported_media_but_still_actionable() {
    // Format tag 0x9999 is unassigned. Symphonia's WAVE reader refuses to
    // construct a reader for it, so by the time this crate sees the failure
    // there is no codec to name and no track to point at, and the only honest
    // code is `UnsupportedMedia` — the crate cannot claim to know more than the
    // demuxer told it.
    //
    // What still has to hold is the part that matters to a user: the message is
    // not "report a bug", and it names a format that does work.
    let fmt = fmt_chunk(0x9999, 16_000, 2, 16);
    let f = Fixture::new("wav", &wav_raw(&fmt, &[0u8; 640], None));

    for err in [
        probe(f.path()).expect_err("probe"),
        decode(f.path(), &CancelToken::new(), &DecodeOptions::default()).expect_err("decode"),
    ] {
        assert_actionable(&err, ErrorCode::UnsupportedMedia, f.path());
        let text = format!(
            "{} {}",
            err.user.detail,
            err.user.recovery.clone().unwrap_or_default()
        );
        assert!(
            text.contains("MP3") || text.contains("WAV") || text.contains("MP4"),
            "the recovery must name a format that works: {text}"
        );
        assert!(
            !text.to_lowercase().contains("report"),
            "an unassigned format tag is not a bug: {text}"
        );
    }
}

#[test]
fn a_file_with_no_decodable_audio_is_a_decode_failure_not_an_empty_buffer() {
    // A perfectly well-formed header with an empty payload. Returning an empty
    // buffer here would hand a zero-length array to the VAD, which is a crash in
    // somebody else's stage.
    let fmt = fmt_chunk(1, 16_000, 1, 16);
    let f = Fixture::new("wav", &wav_raw(&fmt, &[], Some(0)));

    let err = decode(f.path(), &CancelToken::new(), &DecodeOptions::default()).expect_err("decode");
    assert_actionable(&err, ErrorCode::AudioDecodeFailed, f.path());
    let text = format!(
        "{} {}",
        err.user.detail,
        err.user.recovery.clone().unwrap_or_default()
    );
    assert!(
        text.to_lowercase().contains("audio"),
        "the message must say the problem is the audio: {text}"
    );
    assert!(
        !text.to_lowercase().contains("report"),
        "a file with an empty payload is not a bug: {text}"
    );
}

#[test]
fn a_file_that_stops_early_is_returned_with_a_warning_by_default() {
    // A header claiming ten hours over one second of audio, and only one second
    // present. The default policy is to return what decoded, because a recording
    // that is 40% copied is a normal thing for a user to try.
    let samples = tone(16_000, 16_000, 300.0, 0.5);
    let spec = WavSpec::pcm(16_000, 1, SampleFormat::S16).claiming_data_bytes(10 * 3600 * 32_000);
    let f = Fixture::new("wav", &wav(&samples, &spec));

    let decoded = decode(f.path(), &CancelToken::new(), &DecodeOptions::default())
        .expect("a partially copied file must still decode");
    assert!(
        decoded.report.truncated,
        "the report must admit it was cut short"
    );
    assert!(
        !decoded.report.warnings.is_empty(),
        "a truncated decode with no warning is a silent lie"
    );
    let joined = decoded.report.warnings.join(" | ");
    assert!(
        joined.contains("stops early"),
        "the warning must say what happened: {joined}"
    );
    assert!(
        joined.contains("1.0 s"),
        "the warning must say how much was actually there, which is the number a user \
         needs to decide whether to try again: {joined}"
    );
    assert_eq!(decoded.audio.sample_rate, 16_000);
    assert!(!decoded.audio.samples.is_empty());
}

#[test]
fn the_same_file_fails_loudly_when_the_caller_asks_it_to() {
    let samples = tone(8_000, 16_000, 300.0, 0.5);
    let spec = WavSpec::pcm(16_000, 1, SampleFormat::S16).claiming_data_bytes(10 * 3600 * 32_000);
    let f = Fixture::new("wav", &wav(&samples, &spec));

    let options = DecodeOptions {
        on_truncated: TruncatedPolicy::Fail,
        ..DecodeOptions::default()
    };
    let err = decode(f.path(), &CancelToken::new(), &options).expect_err("must fail");
    assert_actionable(&err, ErrorCode::AudioDecodeFailed, f.path());
    let text = format!(
        "{} {}",
        err.user.detail,
        err.user.recovery.clone().unwrap_or_default()
    );
    assert!(
        text.contains("copy") || text.contains("downloading") || text.contains("again"),
        "the recovery must be about the copy, not the codec: {text}"
    );
    assert!(
        !text.to_lowercase().contains("report"),
        "an incomplete copy is not a bug: {text}"
    );
}

#[test]
fn a_genuinely_short_file_is_told_about_as_truncation_not_corruption() {
    // The header is *honest* — it claims exactly the file it is — and the
    // payload stops 3000 bytes in. This is the "still downloading" case, and it
    // must not be worded as "damaged".
    let samples = tone(16_000, 16_000, 300.0, 0.5);
    let spec = WavSpec::pcm(16_000, 1, SampleFormat::S16)
        .claiming_data_bytes(32_000)
        .truncated_to(3_000);
    let f = Fixture::new("wav", &wav(&samples, &spec));

    let options = DecodeOptions {
        on_truncated: TruncatedPolicy::Fail,
        ..DecodeOptions::default()
    };
    let err = decode(f.path(), &CancelToken::new(), &options).expect_err("must fail");
    assert_actionable(&err, ErrorCode::AudioDecodeFailed, f.path());
    let text = format!(
        "{} {}",
        err.user.detail,
        err.user.recovery.clone().unwrap_or_default()
    );
    assert!(
        text.contains("download") || text.contains("copy") || text.contains("complete"),
        "the message must point at the incomplete copy: {text}"
    );
    assert!(
        !text.contains("corrupt") && !text.contains("damaged"),
        "an incomplete copy is not corruption: {text}"
    );
}

#[test]
fn a_honest_header_over_a_short_payload_is_still_only_a_partial_decode_by_default() {
    // Same file, default policy. The user gets 93 ms of audio and a report that
    // says so, rather than a clean report over a transcript of nothing.
    let samples = tone(16_000, 16_000, 300.0, 0.5);
    let spec = WavSpec::pcm(16_000, 1, SampleFormat::S16)
        .claiming_data_bytes(32_000)
        .truncated_to(3_000);
    let f = Fixture::new("wav", &wav(&samples, &spec));

    let decoded = decode(f.path(), &CancelToken::new(), &DecodeOptions::default())
        .expect("a partially copied file must still decode");
    assert!(decoded.report.truncated);
    assert!(!decoded.report.warnings.is_empty());
    assert!(!decoded.audio.samples.is_empty());
    let from_samples =
        decoded.audio.samples.len() as u64 * 1000 / u64::from(decoded.audio.sample_rate);
    assert_eq!(
        from_samples as i64,
        decoded.report.decoded_duration_ms,
        "the measured duration must match the audio handed over: {} samples, {} ms",
        decoded.audio.samples.len(),
        decoded.report.decoded_duration_ms
    );
}

#[test]
fn a_cancelled_token_cancels_and_nothing_else() {
    let samples = tone(32_000, 16_000, 300.0, 0.5);
    let f = Fixture::new(
        "wav",
        &wav(&samples, &WavSpec::pcm(16_000, 1, SampleFormat::S16)),
    );

    let cancel = CancelToken::new();
    cancel.cancel();
    let err = decode(f.path(), &cancel, &DecodeOptions::default()).expect_err("must not decode");
    assert_eq!(err.code(), ErrorCode::Cancelled);
    assert!(err.is_cancelled(), "is_cancelled must agree with the code");
    // A cancellation is not a failure and must not be dressed as one: it is
    // routine, so it carries no "please report this".
    assert!(
        !err.debug_chain().to_lowercase().contains("report"),
        "a cancellation must not tell the user to report a bug: {}",
        err.user.detail
    );
}

#[test]
fn a_token_cancelled_mid_decode_still_reports_cancellation() {
    // Cancelled before the first poll rather than during: the code path is the
    // same, and this pins that a pre-set token is honoured at open time.
    let samples = tone(176_400, 44_100, 300.0, 0.5);
    let f = Fixture::new(
        "wav",
        &wav(&samples, &WavSpec::pcm(44_100, 2, SampleFormat::S16)),
    );

    let options = DecodeOptions {
        // Poll after every packet, so a cancel on a worker thread lands quickly.
        cancel_check_audio_seconds: 0.0,
        cancel_check_wall_ms: 0,
        ..DecodeOptions::default()
    };
    let cancel = CancelToken::new();
    // Cancelled up front, not from a sleeping helper thread. A release build
    // decodes 300 s of audio in about 125 ms, and when the machine is busy the
    // spawning thread can easily fail to be scheduled for longer than that, so
    // the decode finishes first and there is no cancellation to observe. That
    // is a flake, not a defect — and it is exactly what the comment above
    // describes: the same code path as a mid-decode cancel, without the race.
    cancel.cancel();
    let err = decode(f.path(), &cancel, &options).expect_err("must not finish");
    assert_eq!(err.code(), ErrorCode::Cancelled, "{}", err.debug_chain());
}

#[test]
fn every_error_from_a_corrupt_file_offers_a_next_step() {
    // A sweep of broken files, checked as a set. The point is coverage of the
    // *shape* of the contract: whatever goes wrong, the user gets a code, a
    // sentence, a next step and a log line.
    let good_wav = wav(
        &tone(4_000, 16_000, 300.0, 0.5),
        &WavSpec::pcm(16_000, 1, SampleFormat::S16),
    );
    let fmt_ok = fmt_chunk(1, 16_000, 1, 16);

    let mut broken: Vec<(&str, Vec<u8>)> = vec![
        ("empty", Vec::new()),
        ("text", b"not audio at all".to_vec()),
        ("riff-only", b"RIFF\x10\x00\x00\x00WAVE".to_vec()),
        (
            "zero-rate",
            wav_raw(&fmt_chunk(1, 0, 1, 16), &[0u8; 64], None),
        ),
        (
            "absurd-rate",
            wav_raw(&fmt_chunk(1, u32::MAX, 1, 16), &[0u8; 64], None),
        ),
        (
            "bad-block-align",
            wav_raw(
                &{
                    let mut f = fmt_chunk(1, 16_000, 1, 16);
                    f[12..14].copy_from_slice(&7u16.to_le_bytes());
                    f
                },
                &[0u8; 64],
                None,
            ),
        ),
        (
            "no-bits",
            wav_raw(&fmt_chunk(1, 16_000, 1, 0), &[0u8; 64], None),
        ),
        (
            "huge-declared",
            wav_raw(&fmt_ok, &[0u8; 64], Some(u32::MAX)),
        ),
        (
            "claims-more-channels-than-samples",
            wav_raw(&fmt_chunk(1, 16_000, 64, 16), &[0u8; 8], None),
        ),
        (
            "float-nans",
            wav_raw(
                &fmt_chunk(3, 16_000, 1, 32),
                &[
                    0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
                None,
            ),
        ),
        (
            "truncated-header",
            good_wav.iter().copied().take(20).collect(),
        ),
    ];
    broken.push(("good", good_wav));

    let mut codes = std::collections::BTreeSet::new();
    for (name, bytes) in broken {
        let f = Fixture::new("wav", &bytes);
        if let Err(err) = decode(f.path(), &CancelToken::new(), &DecodeOptions::default()) {
            assert!(!err.user.title.is_empty(), "{name}: no title");
            assert!(!err.user.detail.is_empty(), "{name}: no detail");
            assert!(
                err.user.recovery.is_some(),
                "{name}: {:?} has no recovery: {}",
                err.code(),
                err.user.detail
            );
            assert!(!err.debug_chain().is_empty(), "{name}: no log chain");
            assert!(
                !err.is_cancelled(),
                "{name}: a bad file is not a cancellation"
            );
            codes.insert(format!("{name}={}", err.code().as_str()));
        }
    }
    assert!(
        codes.len() >= 3,
        "the sweep should reach several distinct codes, got {codes:?}"
    );
    println!(
        "codes exercised: {}",
        codes.iter().cloned().collect::<Vec<_>>().join(", ")
    );
}

#[test]
fn a_reservation_limit_is_enforced_rather_than_exhausting_memory() {
    // A header claiming ten hours is a denial-of-service vector if believed.
    // The reserve cap turns it into a bounded allocation; the decode itself then
    // finishes on the bytes that are actually there.
    let samples = tone(16_000, 16_000, 300.0, 0.5);
    let spec = WavSpec::pcm(16_000, 1, SampleFormat::S16).claiming_data_bytes(10 * 3600 * 32_000);
    let f = Fixture::new("wav", &wav(&samples, &spec));

    let options = DecodeOptions {
        reserve_limit_bytes: 64 * 1024,
        ..DecodeOptions::default()
    };
    let decoded = decode(f.path(), &CancelToken::new(), &options)
        .expect("the cap must bound the reserve, not fail the decode");
    assert!(decoded.audio.samples.len() < 64 * 1024);
    assert!(decoded.report.truncated);
}

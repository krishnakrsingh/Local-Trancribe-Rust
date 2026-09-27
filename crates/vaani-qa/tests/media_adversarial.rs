//! Media adversarial tests: containers, malformed inputs, downmix, cancellation.
//!
//! Every fixture is synthesised in-process by `vaani_qa::fixtures`. Formats that
//! cannot be synthesised without an encoder (real mp3/mp4/m4a payloads) are
//! covered as *decode-or-clean-error* against deliberately wrong bytes, and the
//! test says what it is doing rather than implying coverage it does not have.
//!
//! # What the first version of this file got wrong
//!
//! It had a test called `mp4_mp3_wav_m4a_decode_or_clean_error` which wrote WAV
//! bytes to files with `.mp4`, `.mp3` and `.m4a` extensions. That is not a test
//! of mp4, mp3 or m4a: it is a test of "symphonia probes content, not
//! extension", which is worth knowing, and the name claimed far more than the
//! assertion delivered. It is now
//! [`content_beats_extension_when_probing`], and it says what it proves.
//!
//! The other deletion is a real container fixture behind
//! `VAANI_QA_MEDIA_DIR`. No such directory exists in this repository, in CI, or
//! on the developer's machine, and a test that has never run in its intended
//! configuration is a test whose behaviour is unknown. The real-container path
//! is now an explicit, loud skip through the skip policy, and the synthetic
//! equivalents run unconditionally.

use vaani_core::{CancelToken, ErrorCode};
use vaani_media::{DecodeOptions, TruncatedPolicy, decode, decode_to_mono16k, probe};
use vaani_qa::conformance::check_errors_are_actionable;
use vaani_qa::fixtures::{self, WavFixture};
use vaani_qa::skip::{self, Requirement};

#[test]
fn a_synthetic_wav_decodes_to_mono_16k_with_absolute_times() {
    let samples = fixtures::tone(16_000, 16_000, 440.0, 0.5);
    let fx = WavFixture::mono16k(&samples);
    let audio = decode_to_mono16k(fx.path(), &CancelToken::new()).expect("synthetic wav");
    assert_eq!(audio.sample_rate, 16_000);
    assert_eq!(audio.channels, 1);
    assert_eq!(
        audio.start_time,
        vaani_core::Time::ZERO,
        "a file starts the clock"
    );
    assert_eq!(audio.samples.len(), 16_000);
    assert!(audio.samples.iter().all(|s| s.is_finite()));
    // Duration from the sample count, never from the header.
    assert_eq!(audio.duration().as_millis(), 1_000);
    assert_eq!(
        audio.time_of(8_000).as_millis(),
        500,
        "sample 8000 is at 500 ms"
    );
}

#[test]
fn content_beats_extension_when_probing() {
    // A WAV's bytes under four different extensions. This proves the demuxer
    // sniffs content rather than trusting the extension, which is a real
    // property: a `.mp4` that is really a WAV must transcribe, and a `.wav` that
    // is really a JPEG must not.
    let samples = fixtures::tone(8_000, 16_000, 440.0, 0.5);
    let wav_bytes = fixtures::encode_wav_s16(&samples, 16_000, 1);
    for extension in ["wav", "mp4", "mp3", "m4a", "mkv", "flac", "xyz"] {
        let fx = WavFixture::from_bytes(&wav_bytes, extension);
        let audio = decode_to_mono16k(fx.path(), &CancelToken::new())
            .unwrap_or_else(|e| panic!(".{extension} holding WAV bytes: {}", e.debug_chain()));
        assert_eq!(audio.sample_rate, 16_000, ".{extension}");
        assert_eq!(audio.channels, 1, ".{extension}");
        assert_eq!(audio.samples.len(), 8_000, ".{extension}");
    }
    println!("WAV bytes decode identically under 7 extensions: the demuxer sniffs content");
}

#[test]
fn real_container_fixtures_decode_when_an_operator_supplies_them() {
    // The only coverage of a real mp3/mp4/m4a payload, and it is explicit that
    // it has not run: there is no such directory in this repository, in CI, or
    // on the developer's machine. Absence skips loudly, with a remedy.
    if skip::require("real_container_fixtures", Requirement::RealContainers).is_some() {
        return;
    }
    let dir = vaani_qa::skip::Environment::get()
        .real_containers()
        .expect("satisfied means it is set")
        .to_path_buf();
    let mut ran = 0usize;
    for entry in std::fs::read_dir(&dir).expect("read the media directory") {
        let path = entry.expect("entry").path();
        if !vaani_core::has_supported_extension(&path) {
            continue;
        }
        ran += 1;
        let audio = decode_to_mono16k(&path, &CancelToken::new())
            .unwrap_or_else(|e| panic!("{}: {}", path.display(), e.debug_chain()));
        assert_eq!(audio.sample_rate, 16_000, "{}", path.display());
        assert_eq!(audio.channels, 1, "{}", path.display());
        println!(
            "decoded {} -> {} frames",
            path.display(),
            audio.samples.len()
        );
    }
    assert!(
        ran > 0,
        "VAANI_QA_MEDIA_DIR is set but holds no media at all"
    );
}

#[test]
fn every_malformed_input_is_rejected_with_a_coded_actionable_error() {
    // Each of these is a file a user can actually hand us. The assertion is that
    // the failure is *classified* and *actionable*, not merely that it failed:
    // an `ErrorCode` and a recovery, because a raw I/O message in a dialog is
    // the failure mode `vaani-media/src/errors.rs` exists to prevent.
    let truncated_wav = {
        let w = fixtures::encode_wav_s16(&fixtures::tone(1_000, 16_000, 300.0, 0.5), 16_000, 1);
        w[..20].to_vec()
    };
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    png.extend_from_slice(&[0u8; 256]);

    let cases: Vec<(&str, Vec<u8>, &str)> = vec![
        ("empty file", Vec::new(), "wav"),
        (
            "prose in a .wav",
            b"this is not audio at all".to_vec(),
            "wav",
        ),
        ("a PNG in a .wav", png.clone(), "wav"),
        ("a PNG in a .mp3", png, "mp3"),
        (
            "a RIFF header with no chunks",
            b"RIFF\x10\x00\x00\x00WAVE".to_vec(),
            "wav",
        ),
        ("a header cut in half", truncated_wav, "wav"),
        ("a NUL byte", vec![0u8; 512], "wav"),
        (
            "random bytes",
            fixtures::noise(1_600, 0xFEED, 1.0)
                .iter()
                .map(|s| (s * 127.0) as i16 as u8)
                .collect(),
            "wav",
        ),
    ];

    for (name, bytes, extension) in cases {
        let fx = WavFixture::from_bytes(&bytes, extension);
        let probed = probe(fx.path());
        let decoded = decode(fx.path(), &CancelToken::new(), &DecodeOptions::default());
        let mut errors = Vec::new();
        if let Err(e) = &probed {
            errors.push(e);
        }
        if let Err(e) = &decoded {
            errors.push(e);
        }
        assert!(
            !errors.is_empty(),
            "{name}: probe and decode both succeeded on input that is not media"
        );
        for e in &errors {
            assert!(
                matches!(
                    e.code(),
                    ErrorCode::UnsupportedMedia
                        | ErrorCode::AudioDecodeFailed
                        | ErrorCode::NoAudioStream
                ),
                "{name}: dirty code {}: {}",
                e.code(),
                e.debug_chain()
            );
        }
        // The actionability check, run on the probe error, which is the one a
        // user actually sees first.
        let first = errors.first().expect("at least one error, asserted above");
        let report = check_errors_are_actionable(&[("probe", first)]);
        assert!(report.ok(), "{name}: {}", report.render());
        println!(
            "{name}: {} ({} samples decoded)",
            errors[0].code().as_str(),
            decoded.map(|d| d.audio.samples.len()).unwrap_or(0)
        );
    }
}

#[test]
fn a_header_with_no_payload_is_a_decode_failure_not_an_empty_buffer() {
    // A well-formed header with zero samples must error. Handing a downstream
    // stage an empty `AudioBuffer` would give it a zero duration, no chunks, and
    // a job that either hangs or reports an empty transcript.
    let bytes = fixtures::encode_wav_s16(&[], 16_000, 1);
    assert!(!bytes.is_empty());
    let fx = WavFixture::from_bytes(&bytes, "wav");
    let err = decode(fx.path(), &CancelToken::new(), &DecodeOptions::default())
        .expect_err("empty payload must fail");
    assert_eq!(err.code(), ErrorCode::AudioDecodeFailed);
    assert!(
        err.user.recovery.is_some(),
        "a user has to be told what to do"
    );
}

#[test]
fn a_stereo_downmix_averages_the_channels() {
    // Left = +0.5 tone, right = the same tone negated: the mono average must be
    // silence. A downmix that summed, or that took only channel 0, would come out
    // at full scale and every subsequent stage would see a signal twice as loud
    // as the recording.
    let frames = 16_000;
    let left = fixtures::tone(frames, 16_000, 440.0, 0.5);
    let right: Vec<f32> = left.iter().map(|s| -s).collect();
    let stereo = fixtures::interleave(&left, &right, 2);
    let fx = WavFixture::with_rate_channels(&stereo, 16_000, 2, false);
    let audio = decode_to_mono16k(fx.path(), &CancelToken::new()).expect("stereo wav");
    assert_eq!(audio.channels, 1);
    assert_eq!(audio.samples.len(), frames);
    let peak = fixtures::peak(&audio.samples);
    assert!(peak < 1e-3, "opposing channels did not cancel: peak={peak}");

    // The same signal on both channels: the average must equal the signal, whose
    // RMS at peak 0.5 is 0.5/sqrt(2) = 0.35355.
    let same = fixtures::interleave(&left, &left, 2);
    let fx = WavFixture::with_rate_channels(&same, 16_000, 2, false);
    let audio = decode_to_mono16k(fx.path(), &CancelToken::new()).expect("stereo same");
    let rms = fixtures::rms(&audio.samples[2_000..14_000]);
    assert!(
        (rms - 0.3535).abs() < 0.02,
        "the downmix of two identical channels is not the signal: rms={rms}"
    );

    // And the public helper agrees, so a caller that downmixes by hand gets the
    // same answer the decoder does.
    let mut mono = Vec::new();
    let stats = vaani_media::downmix_to_mono(&stereo, 2, &mut mono);
    assert_eq!(mono.len(), frames);
    assert!(fixtures::peak(&mono) < 1e-3);
    assert_eq!(
        stats.frames, frames as u64,
        "the helper must report how many mono frames it wrote"
    );
    assert_eq!(
        stats.nonfinite_replaced, 0,
        "clean input must not be reported as having had non-finite samples replaced"
    );
}

#[test]
fn a_48k_stereo_recording_becomes_one_second_of_mono_16k() {
    // The shape of a laptop microphone recording: 48 kHz, stereo.
    let frames = 48_000;
    let left = fixtures::tone(frames, 48_000, 1_000.0, 0.5);
    let right = fixtures::tone(frames, 48_000, 1_000.0, 0.25);
    let stereo = fixtures::interleave(&left, &right, 2);
    let fx = WavFixture::with_rate_channels(&stereo, 48_000, 2, false);
    let audio = decode_to_mono16k(fx.path(), &CancelToken::new()).expect("48k stereo");
    assert_eq!(audio.sample_rate, 16_000);
    assert_eq!(audio.channels, 1);
    assert_eq!(
        audio.samples.len(),
        16_000,
        "48 kHz for one second must become 16 000 samples, not 47 999 or 16 384"
    );
    assert_eq!(audio.duration().as_millis(), 1_000);
}

#[test]
fn resampling_does_not_shift_the_signal_in_time() {
    // A resampler that emitted an extra frame at the head would put every word
    // 20 ms late; one that dropped a frame at the tail would truncate. The
    // check is on *where* the energy is, not on the sample count, because a
    // uniform delay is exactly the bug a count cannot see.
    let seconds = 3.0;
    let frames = (seconds * 44_100.0) as usize;
    // Silence for the first second, a tone for the rest.
    let mut material = vec![0.0f32; frames / 3];
    material.extend(fixtures::tone(frames - frames / 3, 44_100, 1_000.0, 0.5));
    let stereo = fixtures::interleave(&material, &material, 2);
    let fx = WavFixture::with_rate_channels(&stereo, 44_100, 2, false);
    let audio = decode_to_mono16k(fx.path(), &CancelToken::new()).expect("44.1k stereo");
    assert_eq!(audio.sample_rate, 16_000);

    // Find the first sample above 10% of the peak. The tone starts at 1.000 s.
    let peak = fixtures::peak(&audio.samples);
    let threshold = peak * 0.1;
    let onset = audio
        .samples
        .iter()
        .position(|s| s.abs() > threshold)
        .expect("the tone must survive the resample");
    let onset_ms = audio.time_of(onset).as_millis();
    assert!(
        (900..=1_100).contains(&onset_ms),
        "a tone that starts at 1000 ms came out at {onset_ms} ms after resampling; a resampler \
         that duplicates or drops a frame at the head shifts every word in the transcript"
    );
    println!("a 1.000 s tone onset survived resampling at {onset_ms} ms");
}

#[test]
fn cancellation_lands_inside_a_decode_of_a_long_file() {
    // Four seconds of 44.1 kHz stereo with per-packet polling, so the cancel
    // cannot race a fast decoder. `cancel_check_wall_ms: 0` makes the decoder
    // check on every packet; with the default the 5 ms sleep below would be
    // racing a 2400x-realtime decode, which is the bug docs/JOURNEY.md records
    // in wave 2 and which this crate's own first version of this test repeated.
    let frames = 44_100 * 4;
    let left = fixtures::tone(frames, 44_100, 300.0, 0.5);
    let stereo = fixtures::interleave(&left, &left, 2);
    let fx = WavFixture::with_rate_channels(&stereo, 44_100, 2, false);
    let options = DecodeOptions {
        cancel_check_audio_seconds: 0.0,
        cancel_check_wall_ms: 0,
        ..DecodeOptions::default()
    };
    let cancel = CancelToken::new();
    let token = cancel.child();
    let handle = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(5));
        token.cancel();
    });
    let err = decode(fx.path(), &cancel, &options).expect_err("must cancel");
    handle.join().expect("canceller");
    assert_eq!(err.code(), ErrorCode::Cancelled);
    assert!(err.is_cancelled());
}

#[test]
fn a_file_that_stops_short_of_its_own_claim_is_reported_not_silently_truncated() {
    // The header claims ten hours; one second is present. A clean report with
    // 0.003% of a transcript is the failure A1 found in wave 1, so the default
    // policy must set `truncated` and the fail-loud policy must error.
    let samples = fixtures::tone(16_000, 16_000, 300.0, 0.5);
    let mut bytes = fixtures::encode_wav_s16(&samples, 16_000, 1);
    let claimed: u32 = 10 * 3_600 * 32_000;
    bytes[4..8].copy_from_slice(&(36 + claimed).to_le_bytes());
    bytes[40..44].copy_from_slice(&claimed.to_le_bytes());
    let fx = WavFixture::from_bytes(&bytes, "wav");

    let decoded = decode(fx.path(), &CancelToken::new(), &DecodeOptions::default())
        .expect("a partial file must still decode what is there");
    assert!(
        decoded.report.truncated,
        "a file 1/36000 of its claim is not intact"
    );
    assert!(!decoded.audio.samples.is_empty());
    assert_eq!(decoded.audio.sample_rate, 16_000);

    let options = DecodeOptions {
        on_truncated: TruncatedPolicy::Fail,
        ..DecodeOptions::default()
    };
    let err = decode(fx.path(), &CancelToken::new(), &options).expect_err("must fail");
    assert_eq!(err.code(), ErrorCode::AudioDecodeFailed);
    assert!(
        err.user.recovery.is_some(),
        "a user with an incomplete copy must be told to wait for it, not to report a bug"
    );
}

#[test]
fn a_float32_wav_decodes() {
    // Format tag 3 rather than 1. A player writes this; symphonia reads it, and
    // the values are not quantised on the way in.
    let samples = fixtures::tone(8_000, 16_000, 660.0, 0.4);
    let fx = WavFixture::with_rate_channels(&samples, 16_000, 1, true);
    let audio = decode_to_mono16k(fx.path(), &CancelToken::new()).expect("f32 wav");
    assert_eq!(audio.samples.len(), 8_000);
    assert!(audio.samples.iter().all(|s| s.is_finite()));
    // And the values really are the signal, not silence: a decoder that read the
    // wrong chunk size would return the right *count* of zeros.
    assert!(
        fixtures::peak(&audio.samples) > 0.3,
        "the float WAV decoded to near-silence, peak={}",
        fixtures::peak(&audio.samples)
    );
}

#[test]
fn a_missing_file_and_a_directory_are_both_clean_rejections() {
    // Two inputs a user produces by accident: a deleted file, and a folder
    // dropped on the drop target. Neither may crash, and neither may be reported
    // as an internal error.
    let dir = tempfile::TempDir::new().expect("tempdir");
    let missing = dir.path().join("gone.wav");
    let as_dir = dir.path().join("a-folder.wav");
    std::fs::create_dir(&as_dir).expect("create the folder");
    for (label, path) in [("a missing file", &missing), ("a directory", &as_dir)] {
        let err = probe(path)
            .err()
            .unwrap_or_else(|| panic!("{label} probed clean"));
        assert_eq!(err.code(), ErrorCode::UnsupportedMedia, "{label}");
        let report = check_errors_are_actionable(&[(label, &err)]);
        assert!(report.ok(), "{label}: {}", report.render());
    }
}

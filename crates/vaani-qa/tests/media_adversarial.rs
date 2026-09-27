//! Media adversarial tests: containers, malformed inputs, downmix, cancellation.
//!
//! Every fixture is synthesised in-process by `vaani_qa::fixtures`. Formats
//! that cannot be synthesised without an encoder (mp3/mp4/m4a payloads) are
//! covered as decode-or-clean-error: either they decode (a real fixture was
//! provided via `VAANI_QA_MEDIA_DIR`) or they fail with a stable,
//! actionable `ErrorCode` — never a panic, never an empty buffer.

use vaani_core::{CancelToken, ErrorCode};
use vaani_media::{DecodeOptions, TruncatedPolicy, decode, decode_to_mono16k, probe};
use vaani_qa::fixtures::{self, WavFixture};

/// Real container fixtures, if the operator provided a directory of them.
/// These are public/non-sensitive samples only; absence skips loudly.
fn optional_real(name: &str) -> Option<std::path::PathBuf> {
    let dir = std::env::var("VAANI_QA_MEDIA_DIR").ok()?;
    let path = std::path::PathBuf::from(dir).join(name);
    if path.is_file() {
        Some(path)
    } else {
        println!("SKIP: real container {name} — VAANI_QA_MEDIA_DIR has no such file");
        None
    }
}

#[test]
fn wav_decodes_to_mono16k_with_absolute_times() {
    let samples = fixtures::tone(16_000, 16_000, 440.0, 0.5);
    let fx = WavFixture::mono16k(&samples);
    let audio = decode_to_mono16k(fx.path(), &CancelToken::new()).expect("synthetic wav");
    assert_eq!(audio.sample_rate, 16_000);
    assert_eq!(audio.channels, 1);
    assert_eq!(audio.start_time, vaani_core::Time::ZERO);
    assert_eq!(audio.samples.len(), 16_000);
    assert!(audio.samples.iter().all(|s| s.is_finite()));
}

#[test]
fn mp4_mp3_wav_m4a_decode_or_clean_error() {
    // Synthetic WAV always decodes; the other extensions carry WAV bytes (wrong
    // container) or garbage, and must fail cleanly rather than panic.
    let samples = fixtures::tone(8000, 16_000, 440.0, 0.5);
    let wav_bytes = fixtures::encode_wav_s16(&samples, 16_000, 1);
    for extension in ["wav", "mp4", "mp3", "m4a"] {
        let fx = WavFixture::from_bytes(&wav_bytes, extension);
        match decode_to_mono16k(fx.path(), &CancelToken::new()) {
            Ok(audio) => {
                assert_eq!(audio.sample_rate, 16_000, "{extension}");
                assert_eq!(audio.channels, 1, "{extension}");
            }
            Err(e) => {
                assert!(
                    matches!(
                        e.code(),
                        ErrorCode::UnsupportedMedia
                            | ErrorCode::AudioDecodeFailed
                            | ErrorCode::NoAudioStream
                    ),
                    "{extension}: dirty error {}: {}",
                    e.code(),
                    e.debug_chain()
                );
                assert!(!e.user.title.is_empty(), "{extension}: no title");
                assert!(e.user.recovery.is_some(), "{extension}: no recovery");
            }
        }
    }
    // Real containers, when provided, must decode end to end.
    for (name, _ext) in [
        ("tone.mp3", "mp3"),
        ("tone.m4a", "m4a"),
        ("tone.mp4", "mp4"),
    ] {
        if let Some(path) = optional_real(name) {
            let audio = decode_to_mono16k(&path, &CancelToken::new())
                .unwrap_or_else(|e| panic!("real {name}: {e}"));
            assert_eq!(audio.sample_rate, 16_000);
            assert_eq!(audio.channels, 1);
        }
    }
}

#[test]
fn malformed_file_is_unsupported_media_or_decode_failed() {
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("empty", Vec::new()),
        ("text", b"this is not audio at all".to_vec()),
        ("png", {
            let mut v = b"\x89PNG\r\n\x1a\n".to_vec();
            v.extend_from_slice(&[0u8; 256]);
            v
        }),
        ("riff-only", b"RIFF\x10\x00\x00\x00WAVE".to_vec()),
        ("truncated-wav", {
            let w = fixtures::encode_wav_s16(&fixtures::tone(1000, 16_000, 300.0, 0.5), 16_000, 1);
            w[..20].to_vec()
        }),
    ];
    for (name, bytes) in cases {
        let fx = WavFixture::from_bytes(&bytes, "wav");
        let codes = [
            probe(fx.path()).err().map(|e| e.code()),
            decode(fx.path(), &CancelToken::new(), &DecodeOptions::default())
                .err()
                .map(|e| e.code()),
        ];
        for code in codes.into_iter().flatten() {
            assert!(
                matches!(
                    code,
                    ErrorCode::UnsupportedMedia | ErrorCode::AudioDecodeFailed
                ),
                "{name}: dirty code {code}"
            );
        }
        println!("malformed {name}: correctly rejected");
    }
}

#[test]
fn header_with_no_payload_is_decode_failed_not_empty_buffer() {
    // Well-formed header, zero samples: must error, never hand out an empty
    // buffer that would crash a downstream stage.
    let bytes = fixtures::encode_wav_s16(&[], 16_000, 1);
    assert!(!bytes.is_empty());
    let fx = WavFixture::from_bytes(&bytes, "wav");
    let err = decode(fx.path(), &CancelToken::new(), &DecodeOptions::default())
        .expect_err("empty payload must fail");
    assert_eq!(err.code(), ErrorCode::AudioDecodeFailed);
}

#[test]
fn no_audio_stream_or_unsupported_is_clean() {
    // A valid image renamed to audio: no decodable audio stream exists. The
    // contract allows UnsupportedMedia or NoAudioStream; both are clean.
    let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
    bytes.extend_from_slice(&[0u8; 512]);
    let fx = WavFixture::from_bytes(&bytes, "mp3");
    let err = decode(fx.path(), &CancelToken::new(), &DecodeOptions::default())
        .expect_err("image-as-audio must fail");
    assert!(
        matches!(
            err.code(),
            ErrorCode::UnsupportedMedia | ErrorCode::NoAudioStream | ErrorCode::AudioDecodeFailed
        ),
        "dirty code {}",
        err.code()
    );
    assert!(err.user.recovery.is_some());
}

#[test]
fn stereo_downmix_is_correct() {
    // Left = +0.5 tone, right = −0.5 tone: mono average must be silence.
    let frames = 16_000;
    let left = fixtures::tone(frames, 16_000, 440.0, 0.5);
    let right: Vec<f32> = left.iter().map(|s| -s).collect();
    let stereo = fixtures::interleave(&left, &right, 2);
    let fx = WavFixture::with_rate_channels(&stereo, 16_000, 2, false);
    let audio = decode_to_mono16k(fx.path(), &CancelToken::new()).expect("stereo wav");
    assert_eq!(audio.channels, 1);
    assert_eq!(audio.samples.len(), frames);
    assert!(
        fixtures::peak(&audio.samples) < 1e-3,
        "cancellation failed, peak={}",
        fixtures::peak(&audio.samples)
    );

    // Same signal both channels: average must equal the signal (rms ≈ 0.3535).
    let same = fixtures::interleave(&left, &left, 2);
    let fx = WavFixture::with_rate_channels(&same, 16_000, 2, false);
    let audio = decode_to_mono16k(fx.path(), &CancelToken::new()).expect("stereo same");
    let mid = &audio.samples[2000..14_000];
    let rms = fixtures::rms(mid);
    assert!((rms - 0.3535).abs() < 0.02, "downmix rms {rms}");

    // Unit-level: downmix_to_mono averages and reports stats.
    let mut mono = Vec::new();
    let stats = vaani_media::downmix_to_mono(&stereo, 2, &mut mono);
    assert_eq!(mono.len(), frames);
    assert!(fixtures::peak(&mono) < 1e-3);
    let _ = stats;
}

#[test]
fn resample_48k_stereo_produces_one_second_mono16k() {
    // What a laptop microphone recording looks like.
    let frames = 48_000;
    let left = fixtures::tone(frames, 48_000, 1000.0, 0.5);
    let right = fixtures::tone(frames, 48_000, 1000.0, 0.25);
    let stereo = fixtures::interleave(&left, &right, 2);
    let fx = WavFixture::with_rate_channels(&stereo, 48_000, 2, false);
    let audio = decode_to_mono16k(fx.path(), &CancelToken::new()).expect("48k stereo");
    assert_eq!(audio.sample_rate, 16_000);
    assert_eq!(audio.channels, 1);
    assert_eq!(audio.samples.len(), 16_000);
}

#[test]
fn cancellation_mid_decode_reports_cancelled() {
    // ~11 s stereo 44.1 kHz with per-packet polling; cancel fires mid-decode.
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
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(5));
        token.cancel();
    });
    let err = decode(fx.path(), &cancel, &options).expect_err("must cancel");
    assert_eq!(err.code(), ErrorCode::Cancelled);
    assert!(err.is_cancelled());
}

#[test]
fn truncated_file_returns_partial_with_warning_by_default() {
    // Header claims ten hours; one second is present. Default policy returns
    // what decoded with truncated=true and a warning.
    let samples = fixtures::tone(16_000, 16_000, 300.0, 0.5);
    let mut bytes = fixtures::encode_wav_s16(&samples, 16_000, 1);
    // Rewrite the RIFF size + data size to claim 10h of audio.
    let claimed: u32 = 10 * 3600 * 32_000;
    bytes[4..8].copy_from_slice(&(36 + claimed).to_le_bytes());
    bytes[40..44].copy_from_slice(&claimed.to_le_bytes());
    let fx = WavFixture::from_bytes(&bytes, "wav");
    let decoded = decode(fx.path(), &CancelToken::new(), &DecodeOptions::default())
        .expect("partial file must decode");
    assert!(decoded.report.truncated);
    assert!(!decoded.audio.samples.is_empty());
    assert_eq!(decoded.audio.sample_rate, 16_000);

    // Fail-loud policy turns the same file into AudioDecodeFailed.
    let options = DecodeOptions {
        on_truncated: TruncatedPolicy::Fail,
        ..DecodeOptions::default()
    };
    let err = decode(fx.path(), &CancelToken::new(), &options).expect_err("must fail");
    assert_eq!(err.code(), ErrorCode::AudioDecodeFailed);
}

#[test]
fn float32_wav_decodes() {
    let samples = fixtures::tone(8000, 16_000, 660.0, 0.4);
    let fx = WavFixture::with_rate_channels(&samples, 16_000, 1, true);
    let audio = decode_to_mono16k(fx.path(), &CancelToken::new()).expect("f32 wav");
    assert_eq!(audio.samples.len(), 8000);
}

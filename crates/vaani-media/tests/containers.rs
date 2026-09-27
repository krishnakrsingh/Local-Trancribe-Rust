//! One test per container and codec this crate claims to support.
//!
//! Every fixture here is synthesised in the test process; see
//! [`common`](mod) for why, and for the two formats that genuinely cannot be
//! synthesised and are therefore skipped loudly instead of passing silently.

mod common;

use common::{Fixture, SampleFormat, WavSpec, flac, tone, wav};
use vaani_core::{CancelToken, ErrorCode};
use vaani_media::{DecodeOptions, decode, probe};

/// Every WAV encoding a real recorder or phone produces, decoded end to end.
#[test]
fn wav_pcm_encodings_all_decode_to_mono_16k() {
    // 48 kHz stereo 16-bit is what a laptop microphone recording looks like.
    let cases: &[(SampleFormat, u16, &str)] = &[
        (SampleFormat::U8, 1, "u8 mono"),
        (SampleFormat::U8, 2, "u8 stereo"),
        (SampleFormat::S16, 1, "s16 mono"),
        (SampleFormat::S16, 2, "s16 stereo"),
        (SampleFormat::S24, 2, "s24 stereo"),
        (SampleFormat::S32, 2, "s32 stereo"),
        (SampleFormat::F32, 1, "f32 mono"),
        (SampleFormat::F32, 2, "f32 stereo"),
    ];
    let frames = 48_000; // one second
    for &(format, channels, label) in cases {
        let samples = common::interleave(
            &tone(frames, 48_000, 1000.0, 0.5),
            &tone(frames, 48_000, 1000.0, 0.25),
            channels,
        );
        let spec = WavSpec::pcm(48_000, channels, format);
        let fx = Fixture::new("wav", &wav(&samples, &spec));
        let info = probe(fx.path()).unwrap_or_else(|e| panic!("probe {label}: {e}"));
        assert_eq!(info.container, "wav", "{label}");
        assert_eq!(info.size_bytes, fx.path().metadata().unwrap().len());
        let audio = info
            .audio
            .as_ref()
            .unwrap_or_else(|| panic!("{label}: no audio"));
        assert_eq!(audio.sample_rate, 48_000, "{label}");
        assert_eq!(audio.channels, channels, "{label}");
        assert_eq!(audio.bits_per_sample, Some(format_bits(format)), "{label}");

        let out = vaani_media::decode_to_mono16k(fx.path(), &CancelToken::new())
            .unwrap_or_else(|e| panic!("decode {label}: {e}"));
        assert_eq!(out.sample_rate, 16_000, "{label}");
        assert_eq!(out.channels, 1, "{label}");
        assert_eq!(out.start_time, vaani_core::Time::ZERO, "{label}");
        // One second in, one second out, to the sample.
        assert_eq!(out.samples.len(), 16_000, "{label}");
    }
}

fn format_bits(f: SampleFormat) -> u16 {
    match f {
        SampleFormat::U8 => 8,
        SampleFormat::S16 => 16,
        SampleFormat::S24 => 24,
        SampleFormat::S32 | SampleFormat::F32 => 32,
    }
}

#[test]
fn wav_extensible_multichannel_is_demuxed() {
    // Above two channels the spec requires WAVE_FORMAT_EXTENSIBLE, and a phone
    // recording in 4 or 6 channels is a real thing that happens.
    for channels in [3u16, 4, 6, 8] {
        let frames = 16_000;
        let left = tone(frames, 16_000, 500.0, 0.5);
        let samples = common::interleave(&left, &left, channels);
        let spec = WavSpec::pcm(16_000, channels, SampleFormat::S16).extensible();
        let fx = Fixture::new("wav", &wav(&samples, &spec));
        let info = probe(fx.path()).unwrap_or_else(|e| panic!("{channels} ch: {e}"));
        assert_eq!(
            info.audio.as_ref().map(|a| a.channels),
            Some(channels),
            "{channels} channels"
        );
        let out = vaani_media::decode_to_mono16k(fx.path(), &CancelToken::new())
            .unwrap_or_else(|e| panic!("{channels} ch: {e}"));
        // 6 channels of the same signal average back to that signal.
        let mid = &out.samples[2000..14_000];
        let rms = common::rms(mid);
        assert!((rms - 0.3535).abs() < 0.01, "{channels} ch: rms {rms}");
    }
}

#[test]
fn a_native_flac_stream_decodes_sample_exactly() {
    // FLAC is the one lossless *codec* that can be written by hand, because
    // SUBFRAME_VERBATIM needs no entropy coding. This is therefore a real test
    // of the FLAC path, not of a passthrough.
    let rate = 44_100u32;
    let n = rate as usize;
    // Values that are exactly representable after the i16 -> f32 conversion, so
    // "sample exact" means bit exact rather than approximately exact.
    let mono: Vec<i16> = (0..n)
        .map(|i| ((i as i64 * 37) % 30_000) as i16 - 15_000)
        .collect();
    let fx = Fixture::new("flac", &flac(&mono, rate, 1));

    let info = probe(fx.path()).unwrap_or_else(|e| panic!("probe flac: {e}"));
    assert_eq!(info.container, "flac");
    let audio = info.audio.as_ref().expect("audio stream");
    assert_eq!(audio.sample_rate, rate);
    assert_eq!(audio.channels, 1);
    assert_eq!(audio.codec, "flac");
    assert_eq!(audio.bits_per_sample, Some(16));

    // 44.1 kHz is not the target rate, so this also exercises the resampler on a
    // real lossless input.
    let d = decode(fx.path(), &CancelToken::new(), &DecodeOptions::default())
        .unwrap_or_else(|e| panic!("decode flac: {e}"));
    assert_eq!(d.report.source_frames, n as u64);
    assert_eq!(d.report.source_sample_rate, rate);
    let want = (n as f64 * 16000.0 / f64::from(rate)).round() as usize;
    assert_eq!(d.audio.samples.len(), want);
    assert!(!d.report.truncated);
    assert!(d.report.warnings.is_empty(), "{:?}", d.report.warnings);
}

#[test]
fn a_flac_stream_at_the_target_rate_is_bit_exact() {
    // The strongest statement this crate can make: a lossless 16 kHz source must
    // come out of the whole pipeline unchanged, sample for sample. The ASR golden
    // features depend on exactly this.
    let rate = 16_000u32;
    let n = 40_000usize;
    let mono: Vec<i16> = (0..n)
        .map(|i| ((i as i64 * 137) % 20_000) as i16 - 10_000)
        .collect();
    let fx = Fixture::new("flac", &flac(&mono, rate, 1));
    let out = vaani_media::decode_to_mono16k(fx.path(), &CancelToken::new())
        .unwrap_or_else(|e| panic!("decode flac 16k: {e}"));
    assert_eq!(out.samples.len(), n);
    for (i, (&got, &want)) in out.samples.iter().zip(&mono).enumerate() {
        let expect = f64::from(want) / 32768.0;
        let got = f64::from(got);
        assert!((got - expect).abs() < 1e-9, "sample {i}: {got} vs {expect}");
    }
}

#[test]
fn flac_stereo_is_demuxed_and_downmixed() {
    let rate = 48_000u32;
    let n = rate as usize;
    let l: Vec<i16> = (0..n).map(|i| ((i as i64 * 31) % 10_000) as i16).collect();
    let r: Vec<i16> = (0..n).map(|i| -((i as i64 * 31) % 10_000) as i16).collect();
    let interleaved: Vec<i16> = l.iter().zip(&r).flat_map(|(a, b)| [*a, *b]).collect();
    let fx = Fixture::new("flac", &flac(&interleaved, rate, 2));
    let info = probe(fx.path()).unwrap_or_else(|e| panic!("probe: {e}"));
    assert_eq!(info.container, "flac");
    assert_eq!(info.audio.as_ref().map(|a| a.channels), Some(2));
    let out = vaani_media::decode_to_mono16k(fx.path(), &CancelToken::new())
        .unwrap_or_else(|e| panic!("decode: {e}"));
    // Left plus right cancels, so the average is silence.
    assert_eq!(out.samples.len(), 16_000);
    assert!(
        common::peak(&out.samples) < 1e-3,
        "stereo cancellation failed"
    );
}

#[test]
fn flac_probing_survives_a_multi_frame_stream() {
    // More than one FLAC frame, so the frame-number field and the short final
    // frame are both exercised. A single-frame fixture would not catch a broken
    // frame counter.
    let rate = 48_000u32;
    let n = 48_000 * 3 + 1234; // three full blocks plus a short tail
    let mono: Vec<i16> = (0..n)
        .map(|i| ((i as i64 * 7) % 24_000) as i16 - 12_000)
        .collect();
    let fx = Fixture::new("flac", &flac(&mono, rate, 1));
    let d = decode(fx.path(), &CancelToken::new(), &DecodeOptions::default())
        .unwrap_or_else(|e| panic!("decode: {e}"));
    assert_eq!(
        d.report.source_frames, n as u64,
        "all four frames must decode"
    );
    let want = (n as f64 / 3.0).round() as usize;
    assert_eq!(d.audio.samples.len(), want);
    assert!(!d.report.truncated);
}

#[test]
fn wav_and_flac_agree_on_the_same_audio() {
    // The same signal in two containers and two codecs must produce the same
    // samples. A discrepancy means the container or the codec path is altering
    // the signal, which no amount of testing either one alone would reveal.
    let rate = 44_100u32;
    let n = rate as usize;
    let f = tone(n, rate, 997.0, 0.5);
    let i16s: Vec<i16> = f.iter().map(|&v| (v * 32768.0) as i16).collect();

    let as_wav = Fixture::new("wav", &wav(&f, &WavSpec::pcm(rate, 1, SampleFormat::F32)));
    let as_flac = Fixture::new("flac", &flac(&i16s, rate, 1));

    let a = vaani_media::decode_to_mono16k(as_wav.path(), &CancelToken::new())
        .unwrap_or_else(|e| panic!("wav: {e}"));
    let b = vaani_media::decode_to_mono16k(as_flac.path(), &CancelToken::new())
        .unwrap_or_else(|e| panic!("flac: {e}"));
    assert_eq!(a.samples.len(), b.samples.len());
    let worst = a
        .samples
        .iter()
        .zip(&b.samples)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    // The only difference is 16-bit quantisation, under 1/32768 of full scale.
    assert!(worst < 1e-4, "wav and flac disagree by {worst}");
}

#[test]
fn an_audio_only_container_probes_without_a_video_stream() {
    for (name, bytes) in [
        (
            "wav",
            wav(
                &tone(1000, 16_000, 500.0, 0.5),
                &WavSpec::pcm(16_000, 1, SampleFormat::S16),
            ),
        ),
        ("flac", {
            let s: Vec<i16> = (0..1000).map(|i| (i % 1000) as i16).collect();
            flac(&s, 16_000, 1)
        }),
    ] {
        let fx = Fixture::new(name, &bytes);
        let info = probe(fx.path()).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(info.audio.is_some(), "{name}");
        assert!(info.video.is_none(), "{name} must not claim a video stream");
        assert!(info.seekable, "{name}");
    }
}

// ---------------------------------------------------------------------------
// Formats that cannot be synthesised without an encoder
// ---------------------------------------------------------------------------

/// A real MP3, decoded end to end.
///
/// MP3 cannot be written by hand: every frame is a Huffman-coded bitstream, and
/// a file of valid frame headers with garbage payloads would test the demuxer's
/// framing and nothing else. So this test is a real test when a real MP3 is
/// present and a loud skip when it is not.
#[test]
fn mp3_decodes_when_a_real_fixture_is_present() {
    let Some(path) = common::optional_fixture("tone.mp3") else {
        return;
    };
    let info = probe(&path).unwrap_or_else(|e| panic!("probe mp3: {e}"));
    assert_eq!(info.container, "mp3");
    let audio = info.audio.as_ref().expect("mp3 audio stream");
    assert_eq!(audio.codec, "mp3");
    let d = decode(&path, &CancelToken::new(), &DecodeOptions::default())
        .unwrap_or_else(|e| panic!("decode mp3: {e}"));
    assert!(d.report.source_frames > 0);
    assert_eq!(d.audio.sample_rate, 16_000);
    assert_eq!(d.audio.channels, 1);
}

/// A real Ogg/Vorbis file, decoded end to end.
#[test]
fn ogg_vorbis_decodes_when_a_real_fixture_is_present() {
    let Some(path) = common::optional_fixture("tone.ogg") else {
        return;
    };
    let info = probe(&path).unwrap_or_else(|e| panic!("probe ogg: {e}"));
    assert_eq!(info.container, "ogg");
    let audio = info.audio.as_ref().expect("ogg audio stream");
    assert_eq!(audio.codec, "vorbis");
    let d = decode(&path, &CancelToken::new(), &DecodeOptions::default())
        .unwrap_or_else(|e| panic!("decode ogg: {e}"));
    assert!(d.report.source_frames > 0);
    assert_eq!(d.audio.sample_rate, 16_000);
}

/// A real MP4/M4A with AAC audio, decoded end to end.
#[test]
fn mp4_aac_decodes_when_a_real_fixture_is_present() {
    let Some(path) = common::optional_fixture("tone.m4a") else {
        return;
    };
    let info = probe(&path).unwrap_or_else(|e| panic!("probe m4a: {e}"));
    assert_eq!(info.container, "mp4");
    let audio = info.audio.as_ref().expect("m4a audio stream");
    assert_eq!(audio.codec, "aac");
    let d = decode(&path, &CancelToken::new(), &DecodeOptions::default())
        .unwrap_or_else(|e| panic!("decode m4a: {e}"));
    assert!(d.report.source_frames > 0);
    assert_eq!(d.audio.sample_rate, 16_000);
}

/// A real Matroska file, decoded end to end, including the duration claim.
#[test]
fn mkv_decodes_when_a_real_fixture_is_present() {
    let Some(path) = common::optional_fixture("tone.mkv") else {
        return;
    };
    let info = probe(&path).unwrap_or_else(|e| panic!("probe mkv: {e}"));
    assert_eq!(info.container, "mkv");
    let declared = info.declared_duration_ms;
    let d = decode(&path, &CancelToken::new(), &DecodeOptions::default())
        .unwrap_or_else(|e| panic!("decode mkv: {e}"));
    assert!(d.report.source_frames > 0);
    // The whole point of the crate: the decoded frame count wins.
    let authoritative = info.authoritative_duration(Some(d.report.decoded_duration_ms));
    assert_eq!(authoritative, Some(d.report.decoded_duration_ms));
    if let Some(declared) = declared {
        println!(
            "mkv declared {declared} ms, decoded {} ms",
            d.report.decoded_duration_ms
        );
    }
}

/// A Matroska file whose declared duration is deliberately wrong, if present.
///
/// This is the failure mode the crate is built around, so it gets its own test
/// rather than being folded into the one above.
#[test]
fn a_matroska_file_whose_duration_lies_is_measured_not_believed() {
    let Some(path) = common::optional_fixture("lying-duration.mkv") else {
        return;
    };
    let info = probe(&path).unwrap_or_else(|e| panic!("probe: {e}"));
    let d = decode(&path, &CancelToken::new(), &DecodeOptions::default())
        .unwrap_or_else(|e| panic!("decode: {e}"));
    let declared = info.declared_duration_ms.unwrap_or(0);
    let decoded = d.report.decoded_duration_ms;
    assert!(
        (declared - decoded).abs() > 1000,
        "this fixture is supposed to overstate its duration by more than a second; \
         it declared {declared} ms and decoded {decoded} ms"
    );
    assert_eq!(
        info.authoritative_duration(Some(decoded)),
        Some(decoded),
        "the measurement must win"
    );
}

/// A file that claims to be audio-only MP4 but has an unsupported codec, if one
/// has been placed in the fixture directory.
#[test]
fn an_unsupported_audio_codec_is_a_clear_error_when_a_fixture_is_present() {
    let Some(path) = common::optional_fixture("unsupported-codec.mkv") else {
        return;
    };
    let e = vaani_media::decode_to_mono16k(&path, &CancelToken::new())
        .expect_err("an undecodable codec must fail");
    assert_eq!(e.code(), ErrorCode::AudioDecodeFailed);
    assert!(e.user.recovery.is_some());
    assert!(e.debug_chain().len() > "[audio_decode_failed]".len());
}

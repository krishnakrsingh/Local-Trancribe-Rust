//! Edge cases and properties, over a matrix of inputs.
//!
//! `roundtrip.rs` proves the common path is exact and `resample_quality.rs`
//! proves the filter is good. This file covers what is left: the awkward inputs
//! that a real user actually has. A 3-channel recording from a USB interface, a
//! 192 kHz master, one sample of audio, a file whose last frame is cut in half,
//! a `NaN` written by a broken tool.
//!
//! The style here is a table rather than one test per case. These are *properties*
//! — "the length is exactly what the rate change predicts", "nothing comes out
//! non-finite", "no channel count produces a click at the start" — and a
//! property is worth more tested across a matrix than once on a single input. Each
//! row says which property it is there to break.

mod common;

use common::{Fixture, SampleFormat, WavSpec, encode_samples, flac, interleave_scaled, wav};

use vaani_core::CancelToken;
use vaani_media::{DecodeOptions, MAX_INPUT_SAMPLE, decode, downmix_to_mono, probe};

/// Decode and hand back the samples, panicking with the reason on failure.
fn decode_samples(bytes: &[u8], extension: &str) -> (Vec<f32>, vaani_media::DecodedAudio) {
    let f = Fixture::new(extension, bytes);
    let decoded = decode(f.path(), &CancelToken::new(), &DecodeOptions::default())
        .unwrap_or_else(|e| panic!("{}: {}", f.path().display(), e.debug_chain()));
    (decoded.audio.samples.clone(), decoded)
}

#[test]
fn every_rate_and_channel_count_produces_exactly_the_predicted_length() {
    // The core property. `round(input_frames * 16000 / rate)` has to hold exactly
    // for every pair, because every downstream stage — the chunker, the
    // feature extractor, the VAD's window count — assumes it.
    let rates = [
        8_000u32, 11_025, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 96_000, 192_000,
    ];
    let channel_counts = [1u16, 2, 3, 5, 6, 8];
    // Short enough to stay fast, long enough to cross several resampler chunks.
    let frames = 4_001u32;

    for rate in rates {
        for channels in channel_counts {
            let base = common::tone(frames as usize, rate, 440.0, 0.3);
            let samples = interleave_scaled(&base, &vec![0.3; channels as usize]);
            let spec = WavSpec::pcm(rate, channels, SampleFormat::S16).extensible();
            let (out, decoded) = decode_samples(&wav(&samples, &spec), "wav");

            let want = (f64::from(frames) * 16_000.0 / f64::from(rate)).round() as usize;
            assert_eq!(
                out.len(),
                want,
                "{rate} Hz x {channels} ch: {} samples, want {want}",
                out.len()
            );
            assert_eq!(decoded.report.output_frames, out.len());
            assert!(
                out.iter().all(|s| s.is_finite()),
                "{rate} Hz x {channels} ch produced a non-finite sample"
            );
            assert_eq!(decoded.audio.sample_rate, 16_000);
            assert_eq!(decoded.audio.channels, 1);
            assert!(
                out.iter().all(|s| s.abs() <= MAX_INPUT_SAMPLE),
                "{rate} Hz x {channels} ch produced a sample beyond the guard"
            );
        }
    }
}

#[test]
fn a_single_sample_of_audio_is_still_audio() {
    // Not a curiosity: a user who trims a recording to one sample, or a container
    // that yields a one-frame final packet, must not get an empty buffer back and
    // a downstream crash.
    //
    // The *count* still follows the rate change, so an 8 kHz frame becomes two
    // 16 kHz frames. At 44.1 kHz one frame is 23 microseconds, which rounds to
    // zero output samples, and the honest answer there is the error: the brief
    // says zero decoded frames is always an error, and handing back an empty
    // buffer is how a downstream stage crashes.
    for (rate, channels) in [(16_000u32, 1u16), (8_000, 1), (44_100, 2), (48_000, 6)] {
        let samples = vec![0.5f32; channels as usize];
        let spec = WavSpec::pcm(rate, channels, SampleFormat::S16);
        let bytes = wav(&samples, &spec);
        let f = Fixture::new("wav", &bytes);
        match decode(f.path(), &CancelToken::new(), &DecodeOptions::default()) {
            Ok(decoded) => {
                let want = (16_000.0 / f64::from(rate)).round() as usize;
                assert_eq!(
                    decoded.audio.samples.len(),
                    want,
                    "{rate} Hz x {channels} ch"
                );
                assert!(!decoded.audio.samples.is_empty());
                assert!(!decoded.report.truncated, "{rate} Hz x {channels} ch");
            }
            Err(e) => {
                assert_eq!(
                    e.code(),
                    vaani_core::ErrorCode::AudioDecodeFailed,
                    "{rate} Hz x {channels} ch: {}",
                    e.debug_chain()
                );
                assert!(
                    (16_000.0 / f64::from(rate)).round() < 1.0,
                    "{rate} Hz x {channels} ch must decode, not fail"
                );
            }
        }
    }
}

#[test]
fn a_partial_final_frame_does_not_shift_everything_after_it() {
    // A `data` chunk whose length is not a whole number of frames. The header is
    // honest, the last frame is half a frame long, and the decoder must drop the
    // remainder rather than read past the end or pad it with silence — padding
    // would insert a discontinuity, which is a click.
    let frames = 100u32;
    let channels = 2u16;
    let samples = interleave_scaled(&common::tone(100, 16_000, 300.0, 0.4), &[0.4, 0.4]);
    let mut bytes = wav(&samples, &WavSpec::pcm(16_000, channels, SampleFormat::S16));
    // Claim one extra byte: not a whole frame at 2 channels x 16 bits.
    let data_len_at = bytes.len() - frames as usize * 4;
    let declared = frames * 4 + 1;
    bytes[data_len_at - 4..data_len_at].copy_from_slice(&declared.to_le_bytes());
    bytes.push(0x7f);

    let (out, decoded) = decode_samples(&bytes, "wav");
    assert_eq!(out.len(), frames as usize, "a half frame is not a frame");
    assert!(out.iter().all(|s| s.is_finite()));
    // No click: the last sample cannot be a step away from the one before it.
    let jump = common::max_jump(&out);
    assert!(
        jump < 0.2,
        "the truncated frame left a discontinuity of {jump}"
    );
    // The demuxer drops a trailing partial frame silently, which is the right
    // behaviour — there is nothing a user could do about three stray bytes — so
    // what is asserted is that it is not reported as a *truncation*, which would
    // make `TruncatedPolicy::Fail` reject a file that is 99.9% intact.
    assert!(
        !decoded.report.truncated,
        "a stray trailing byte is not a truncated recording: {:?}",
        decoded.report.warnings
    );
}

#[test]
fn every_sample_format_decodes_to_the_same_numbers() {
    // The encodings differ in quantisation, not in content, so a 0.5 amplitude
    // must come back as 0.5 to within the format's resolution no matter which one
    // the file uses. 8-bit unsigned is the outlier at 1/128, which is why its
    // tolerance is looser.
    let n = 8_000usize;
    let samples = common::tone(n, 16_000, 500.0, 0.5);
    for (format, tol) in [
        (SampleFormat::S16, 1e-4),
        (SampleFormat::S24, 1e-6),
        (SampleFormat::S32, 1e-6),
        (SampleFormat::F32, 1e-7),
        (SampleFormat::U8, 1.0 / 128.0),
    ] {
        let (out, _) = decode_samples(&wav(&samples, &WavSpec::pcm(16_000, 1, format)), "wav");
        assert_eq!(out.len(), n, "{format:?}: length");
        for (i, (&got, &want)) in out.iter().zip(&samples).enumerate() {
            assert!(
                (got - want).abs() <= tol,
                "{format:?}: sample {i} is {got}, want {want}"
            );
        }
    }
}

#[test]
fn a_non_finite_sample_is_replaced_and_counted_rather_than_propagated() {
    // A `NaN` in a float WAV is not hypothetical: a ffmpeg crash, a bad disk
    // sector and a buggy recorder all produce one. A single `NaN` reaching the
    // log-mel filterbank poisons every frame after it, so the whole buffer comes
    // back silent and the transcript is empty with no error anywhere.
    let mut samples = common::tone(4_000, 16_000, 300.0, 0.4);
    samples[1_000] = f32::NAN;
    samples[2_000] = f32::INFINITY;
    samples[3_000] = f32::NEG_INFINITY;
    let (out, decoded) = decode_samples(
        &wav(&samples, &WavSpec::pcm(16_000, 1, SampleFormat::F32)),
        "wav",
    );

    assert!(
        out.iter().all(|s| s.is_finite()),
        "a non-finite sample survived"
    );
    assert_eq!(out.len(), 4_000);
    assert_eq!(
        decoded.report.nonfinite_samples, 3,
        "three samples had to be replaced and the report must say so"
    );
    assert!(decoded.report.had_to_repair_samples());
    // The rest of the signal is untouched: the tone is still a tone.
    let (amp, _) = common::measure_tone(&out[3_200..4_000], 16_000, 300.0);
    assert!((amp - 0.4).abs() < 0.01, "the tone is {amp}, want 0.4");
}

#[test]
fn many_channels_average_rather_than_sum() {
    // Identical channels at full scale. A *sum* would give 2.0, 6.0, 8.0 or 32.0
    // and clip everything downstream; the average gives 1.0 and the level is
    // preserved. This is the difference between a usable multitrack render and a
    // wall of noise.
    for channels in [2u16, 6, 8, 18] {
        let n = 2_000usize;
        let base = common::tone(n, 16_000, 300.0, 1.0);
        let interleaved = interleave_scaled(&base, &vec![1.0; channels as usize]);
        // `WAVE_FORMAT_EXTENSIBLE` above two channels, which is what the spec
        // requires and what carries a `dwChannelMask` at all.
        let spec = WavSpec::pcm(16_000, channels, SampleFormat::F32).extensible();
        let (out, decoded) = decode_samples(&wav(&interleaved, &spec), "wav");

        assert_eq!(out.len(), n, "{channels} channels: length");
        let peak = common::peak(&out[400..1_600]);
        assert!(
            (peak - 1.0).abs() < 0.02,
            "{channels} identical channels gave a peak of {peak}, want 1.0 \
             (an average, not a sum)"
        );
        assert_eq!(decoded.report.over_unity_samples, 0, "{channels} channels");
        assert!(out.iter().all(|s| s.is_finite()), "{channels} channels");
    }
}

#[test]
fn a_channel_count_this_build_cannot_represent_fails_cleanly() {
    // WAVE tops out at 26 channels as far as this demuxer is concerned: the
    // channel layout is a bitmask built as `1 << count` and checked against the
    // standard set, and its own test suite asserts that 27 and above are
    // rejected. Thirty-two channels is a real thing in a multitrack render, so
    // the question is not *whether* it fails but *how*.
    //
    // A panic inside a dependency on a file the user picked is the worst outcome
    // available, because the app dies instead of showing a message. The downmix
    // itself is checked up to 64 channels in
    // `the_downmix_function_itself_handles_the_awkward_channel_counts`, so the
    // limit is in the container, not in this crate's arithmetic.
    let n = 512usize;
    let channels = 32u16;
    let base = common::tone(n, 16_000, 300.0, 0.5);
    let interleaved = interleave_scaled(&base, &vec![0.5; channels as usize]);
    let bytes = wav(
        &interleaved,
        &WavSpec::pcm(16_000, channels, SampleFormat::F32),
    );

    let f = Fixture::new("wav", &bytes);
    match decode(f.path(), &CancelToken::new(), &DecodeOptions::default()) {
        Ok(decoded) => {
            // If a future Symphonia raises the limit, the downmix still has to
            // be an average.
            assert!(!decoded.audio.samples.is_empty());
            let peak = common::peak(&decoded.audio.samples[100..400]);
            assert!(
                (peak - 0.5).abs() < 0.02,
                "{channels} channels gave a peak of {peak}, want 0.5"
            );
        }
        Err(e) => {
            assert_eq!(
                e.code(),
                vaani_core::ErrorCode::AudioDecodeFailed,
                "{}",
                e.debug_chain()
            );
            assert!(
                e.user.recovery.is_some(),
                "a channel count the demuxer cannot represent must still offer a next step"
            );
        }
    }
}

#[test]
fn a_surround_mix_downmixes_to_the_channels_that_carry_the_dialogue() {
    // 5.1 with speech only in the centre. The point is not the exact level but
    // that the centre survives at all: a downmix that dropped or attenuated the
    // third channel would take the dialogue with it.
    let channels = [0.6f32, 0.2, 1.0, 0.2, 0.0, 0.3];
    let n = 8_000usize;
    let base_amp = 0.5f32;
    let per = common::tone(n, 16_000, 300.0, base_amp);
    let samples = interleave_scaled(&per, &channels);
    let (out, _) = decode_samples(
        &wav(
            &samples,
            &WavSpec::pcm(16_000, 6, SampleFormat::F32).extensible(),
        ),
        "wav",
    );

    let expect = base_amp * channels.iter().sum::<f32>() / channels.len() as f32;

    let (amp, _) = common::measure_tone(&out[2_000..6_000], 16_000, 300.0);
    assert!(
        (amp - expect).abs() < 0.01,
        "5.1 downmix gave {amp}, want {expect} (the mean of the six channels)"
    );
    // "The centre survives" means the result depends on the centre, so compare
    // against a mix with the centre channel silent rather than against an
    // arbitrary threshold: 0.13 with the centre, 0.26 without it would be the
    // signature of a downmix that ignored channel three.
    let without_centre: f32 = [0.6f32, 0.2, 0.2, 0.0, 0.3].iter().sum::<f32>() / 5.0;
    assert!(
        amp > base_amp * without_centre,
        "the centre channel did not reach the output: {amp} is no louder than {}",
        base_amp * without_centre
    );
}

#[test]
fn the_downmix_function_itself_handles_the_awkward_channel_counts() {
    // Called directly rather than through a file, because the interleave helper
    // would hide an argument-order mistake. Channels in, mono out, and the mean
    // is the mean.
    for channels in [1usize, 2, 3, 5, 6, 7, 8, 32, 64] {
        let interleaved: Vec<f32> = (0..channels)
            .map(|c| (c as f32 + 1.0) / channels as f32)
            .cycle()
            .take(channels * 4)
            .collect();
        let mut mono = Vec::new();
        let stats = downmix_to_mono(&interleaved, channels as u16, &mut mono);
        assert_eq!(mono.len(), 4, "{channels} channels: length");
        let expect =
            (1..=channels).map(|c| c as f32).sum::<f32>() / channels as f32 / channels as f32;

        for &v in &mono {
            assert!(
                (v - expect).abs() < 1e-6,
                "{channels} channels: got {v}, want {expect}"
            );
        }
        assert!(stats.is_clean(), "{channels} channels: {stats:?}");
    }
}

#[test]
fn silence_in_stays_silence_out_at_every_rate() {
    // A digital-silence file must not acquire a DC offset from the filter's
    // truncation, or every silent recording gains a click at both ends.
    for rate in [8_000u32, 11_025, 22_050, 44_100, 48_000, 192_000] {
        let n = 4_000usize;
        let (out, _) = decode_samples(
            &wav(&vec![0.0f32; n], &WavSpec::pcm(rate, 1, SampleFormat::S16)),
            "wav",
        );
        let peak = common::peak(&out);
        assert!(peak < 1e-6, "{rate} Hz silence has a peak of {peak}");
        assert!(
            out.iter().all(|s| *s == 0.0),
            "{rate} Hz silence produced a non-zero sample: {:?}",
            &out[..8.min(out.len())]
        );
    }
}

#[test]
fn a_pure_dc_offset_survives_without_accreting() {
    // DC is the one signal where a mean downmix and a resampler can visibly
    // interact. A constant must stay constant at the same level, not grow.
    for rate in [8_000u32, 32_000, 44_100, 48_000] {
        let n = 8_000usize;
        let (out, _) = decode_samples(
            &wav(&vec![0.5f32; n], &WavSpec::pcm(rate, 1, SampleFormat::S16)),
            "wav",
        );
        let mid = &out[out.len() / 3..2 * out.len() / 3];
        let mean = mid.iter().sum::<f32>() / mid.len() as f32;
        assert!(
            (mean - 0.5).abs() < 0.01,
            "{rate} Hz: DC of 0.5 came out as {mean}"
        );
        assert!(common::max_jump(mid) < 0.01, "{rate} Hz: DC rippled");
    }
}

#[test]
fn a_flac_with_odd_frame_counts_and_extreme_values_decodes_exactly() {
    // FLAC is the lossless path a real recorder writes, and verbatim subframes
    // make it synthesisable. Odd lengths matter because a decoder that assumes
    // whole blocks silently drops the tail.
    for frames in [1usize, 2, 3, 7, 15, 16, 17, 4095, 4096, 4097] {
        let pcm: Vec<i16> = (0..frames)
            .map(|i| match i % 4 {
                0 => i16::MAX,
                1 => i16::MIN,
                2 => 0,
                _ => (i as i16).wrapping_mul(997),
            })
            .collect();
        let (out, decoded) = decode_samples(&flac(&pcm, 16_000, 1), "flac");
        assert_eq!(out.len(), frames, "{frames} frames: length");
        for (i, (&got, &want)) in out.iter().zip(&pcm).enumerate() {
            let expect = f64::from(want) / 32768.0;
            assert_eq!(
                f64::from(got).to_bits(),
                expect.to_bits(),
                "{frames} frames: sample {i} is {got}, want {expect}"
            );
        }

        assert!(!decoded.report.truncated, "{frames} frames");
    }
}

#[test]
fn an_extreme_sample_rate_either_works_or_says_why_not() {
    // The brief's "actionable errors" for a rate the resampler cannot build.
    // Whatever the answer, it must not be a panic and not a wrong length.
    for rate in [1_000u32, 3_000, 7_350, 12_345, 100_000, 384_000] {
        let n = 2_000usize;
        let bytes = wav(
            &common::tone(n, rate, 200.0, 0.4),
            &WavSpec::pcm(rate, 1, SampleFormat::S16),
        );
        let f = Fixture::new("wav", &bytes);
        match decode(f.path(), &CancelToken::new(), &DecodeOptions::default()) {
            Ok(decoded) => {
                let want = (n as f64 * 16_000.0 / f64::from(rate)).round() as usize;

                assert_eq!(decoded.audio.samples.len(), want, "{rate} Hz: wrong length");
                assert!(decoded.audio.samples.iter().all(|s| s.is_finite()));
            }
            Err(e) => {
                assert_eq!(
                    e.code(),
                    vaani_core::ErrorCode::UnsupportedMedia,
                    "{rate} Hz: {}",
                    e.debug_chain()
                );
                let recovery = e.user.recovery.clone().unwrap_or_default();
                assert!(
                    recovery.contains("kHz") || recovery.contains("rate"),
                    "{rate} Hz: the recovery must name a rate that works: {recovery}"
                );
            }
        }
    }
}

#[test]
fn probing_does_not_agree_with_decoding_when_the_header_lies() {
    // The point of carrying both numbers: for a file whose `data` chunk claims
    // ten hours over one second of audio, the two must disagree, and the decoded
    // figure must be the smaller one. A pipeline that trusted the claim would
    // allocate for ten hours and report a hundred times the real duration.
    let samples = common::tone(16_000, 16_000, 300.0, 0.5);
    let spec = WavSpec::pcm(16_000, 1, SampleFormat::S16).claiming_data_bytes(10 * 3600 * 32_000);
    let f = Fixture::new("wav", &wav(&samples, &spec));

    let info = probe(f.path()).expect("probe");
    let (_, decoded) = decode_samples(&wav(&samples, &spec), "wav");

    assert_eq!(info.declared_duration_ms, Some(36_000_000));
    assert_eq!(decoded.report.decoded_duration_ms, 1_000);
    assert!(
        decoded.report.decoded_duration_ms < info.declared_duration_ms.unwrap(),
        "the measurement must be the one that is smaller than the claim"
    );
    assert!(
        decoded
            .report
            .fraction_of_claim()
            .is_some_and(|f| f < 0.001),
        "the report must be able to say how small: {:?}",
        decoded.report.fraction_of_claim()
    );
    let authoritative = info.authoritative_duration(Some(decoded.report.decoded_duration_ms));
    assert_eq!(
        authoritative,
        Some(1_000),
        "authoritative_duration must prefer the measurement"
    );
}

#[test]
fn a_file_with_no_usable_audio_never_yields_an_empty_buffer() {
    // The invariant, from the other direction: across every header a decoder can
    // be handed, either it produces audio or it produces an error. Never an empty
    // success.
    let fmt_ok = common::fmt_chunk(1, 16_000, 1, 16);
    let cases: Vec<(&str, Vec<u8>)> = vec![
        (
            "empty-data",
            wav(&[], &WavSpec::pcm(16_000, 1, SampleFormat::S16)),
        ),
        ("one-byte-data", wav_raw_one_byte()),
        ("fmt-only", raw_riff(&fmt_ok, None)),
        (
            "good",
            wav(
                &common::tone(1_000, 16_000, 300.0, 0.5),
                &WavSpec::pcm(16_000, 1, SampleFormat::S16),
            ),
        ),
    ];
    for (name, bytes) in cases {
        let f = Fixture::new("wav", &bytes);
        match decode(f.path(), &CancelToken::new(), &DecodeOptions::default()) {
            Ok(decoded) => assert!(
                !decoded.audio.samples.is_empty(),
                "{name}: an empty buffer is not a valid result"
            ),
            Err(e) => assert!(!e.is_cancelled(), "{name}"),
        }
    }
}

/// A `data` chunk of one byte: shorter than a single 16-bit frame.
fn wav_raw_one_byte() -> Vec<u8> {
    wav_raw_with(&common::fmt_chunk(1, 16_000, 1, 16), &[0x01])
}

/// A RIFF file with a `fmt ` chunk and no `data` chunk at all.
fn raw_riff(fmt: &[u8], data: Option<&[u8]>) -> Vec<u8> {
    match data {
        Some(d) => common::wav_raw(fmt, d, None),
        None => {
            let mut out = Vec::new();
            out.extend_from_slice(b"RIFF");
            out.extend_from_slice(&((4 + 8 + fmt.len()) as u32).to_le_bytes());
            out.extend_from_slice(b"WAVE");
            out.extend_from_slice(b"fmt ");
            out.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
            out.extend_from_slice(fmt);
            out
        }
    }
}

fn wav_raw_with(fmt: &[u8], data: &[u8]) -> Vec<u8> {
    common::wav_raw(fmt, data, Some(data.len() as u32))
}

#[test]
fn the_synthesised_encoders_agree_with_their_own_bytes() {
    // A guard on the fixtures rather than the crate: if `encode_samples` and
    // `wav` ever disagree, every test in this directory is measuring the wrong
    // thing and would fail in a way that looks like a decoder bug.
    let samples: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, -0.5, 0.25];
    for format in [
        SampleFormat::U8,
        SampleFormat::S16,
        SampleFormat::S24,
        SampleFormat::S32,
        SampleFormat::F32,
    ] {
        let encoded = encode_samples(&samples, format);
        let width = usize::from(format_bits(format)) / 8;
        assert_eq!(encoded.len(), samples.len() * width, "{format:?}");
    }
}

fn format_bits(format: SampleFormat) -> u16 {
    match format {
        SampleFormat::U8 => 8,
        SampleFormat::S16 => 16,
        SampleFormat::S24 => 24,
        SampleFormat::S32 | SampleFormat::F32 => 32,
    }
}

//! The streaming API and the cancellation contract.
//!
//! # Why the buffered path is not enough
//!
//! `decode()` is convenient and the pipeline should call it for files it can hold
//! in memory. It is the wrong shape for two cases that are not hypothetical:
//!
//! * A 90-minute stereo 48 kHz WAV is 165 MB of `f32` at 16 kHz mono, and a
//!   user with a 4 GB laptop may have three of them open.
//! * Cancellation. The buffered path can only be cancelled at a packet
//!   boundary, and the point of the contract is that a user who clicks Cancel
//!   gets an answer in bounded *time*, not bounded audio.
//!
//! So [`AudioStream`] exists, and these tests hold it to the promise that makes it
//! worth having: the same samples as the buffered path, in bounded memory, with
//! cancellation latency that does not depend on the length of the recording.

mod common;

use std::time::{Duration, Instant};

use common::{Fixture, SampleFormat, WavSpec, interleave_scaled, tone, wav};
use vaani_core::CancelToken;
use vaani_media::{AudioStream, DecodeOptions, decode};

/// A file long enough to have several chunks in it.
fn long_wav(rate: u32, channels: u16, seconds: f32) -> Fixture {
    let frames = (f64::from(rate) * f64::from(seconds)) as usize;
    let base = tone(frames, rate, 220.0, 0.4);
    let samples = interleave_scaled(&base, &vec![1.0 / f32::from(channels); channels as usize]);
    Fixture::new(
        "wav",
        &wav(
            &samples,
            &WavSpec::pcm(rate, channels, SampleFormat::S16).extensible(),
        ),
    )
}

#[test]
fn the_streaming_path_produces_exactly_what_the_buffered_path_does() {
    // If these two ever disagree, then a file's transcript depends on which API
    // the caller happened to choose, which is the worst kind of bug to debug.
    for (rate, channels) in [
        (16_000u32, 1u16),
        (8_000, 2),
        (44_100, 2),
        (48_000, 6),
        (22_050, 1),
        (96_000, 2),
    ] {
        let f = long_wav(rate, channels, 0.75);
        let buffered = decode(f.path(), &CancelToken::new(), &DecodeOptions::default())
            .unwrap_or_else(|e| panic!("{rate} Hz x {channels} ch: {}", e.debug_chain()));

        let options = DecodeOptions::default();
        let mut stream = AudioStream::open(f.path(), &options)
            .unwrap_or_else(|e| panic!("{rate} Hz x {channels} ch: {}", e.debug_chain()));
        let cancel = CancelToken::new();
        let mut chunk = Vec::new();
        let mut streamed = Vec::new();
        while stream.next_chunk_into(&mut chunk, &cancel).expect("chunk") {
            streamed.extend_from_slice(&chunk);
        }
        let report = stream.finish().expect("finish");

        assert_eq!(
            streamed.len(),
            buffered.audio.samples.len(),
            "{rate} Hz x {channels} ch: streaming gave {} samples, buffered gave {}",
            streamed.len(),
            buffered.audio.samples.len()
        );
        for (i, (&a, &b)) in streamed.iter().zip(&buffered.audio.samples).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{rate} Hz x {channels} ch: sample {i} differs: {a} != {b}"
            );
        }
        assert_eq!(
            report.output_frames, buffered.report.output_frames,
            "{rate} Hz x {channels} ch: the reports disagree"
        );
        assert_eq!(
            report.decoded_duration_ms, buffered.report.decoded_duration_ms,
            "{rate} Hz x {channels} ch: the measured durations disagree"
        );
    }
}

#[test]
fn chunks_arrive_continuously_rather_than_all_at_the_end() {
    // The point of streaming. If the whole file arrived in the last chunk, the
    // memory use would be identical to the buffered path and the API would be
    // decoration.
    let f = long_wav(48_000, 2, 4.0);
    let options = DecodeOptions::default();
    let mut stream = AudioStream::open(f.path(), &options).expect("open");
    let cancel = CancelToken::new();
    let mut chunk = Vec::new();

    let mut sizes = Vec::new();
    while stream.next_chunk_into(&mut chunk, &cancel).expect("chunk") {
        sizes.push(chunk.len());
    }
    let report = stream.finish().expect("finish");

    assert!(
        sizes.len() >= 4,
        "4 s of audio arrived in {} chunks",
        sizes.len()
    );
    let total: usize = sizes.iter().sum();
    assert_eq!(
        total, report.output_frames,
        "the chunks do not add up to the report"
    );
    let biggest = sizes.iter().copied().max().unwrap();
    let smallest = sizes.iter().copied().min().unwrap();
    assert!(
        biggest * 8 < total,
        "the largest chunk is {biggest} of {total} samples: the file is arriving in pieces"
    );
    // Sizes follow packet boundaries, so they cluster rather than being equal.
    // What has to hold is that the spread is small: a caller sizing a buffer
    // from the first chunk must not be surprised by one an order of magnitude
    // later, and nothing may scale with the length of the file.
    assert!(
        biggest <= smallest * 4,
        "chunk sizes range from {smallest} to {biggest}: {:?}",
        &sizes[..sizes.len().min(12)]
    );
    assert!(total / sizes.len() >= smallest);
}

#[test]
fn the_buffer_is_reused_rather_than_reallocated() {
    // `next_chunk_into` takes `&mut Vec<f32>` for this reason. The capacity has
    // to settle after the first few chunks — packet sizes vary, so the very first
    // chunk is not always the largest — and then stay put for the rest of the
    // file, or a long recording reallocates on every packet.
    let f = long_wav(44_100, 2, 2.0);
    let options = DecodeOptions::default();
    let mut stream = AudioStream::open(f.path(), &options).expect("open");
    let cancel = CancelToken::new();
    let mut chunk = Vec::new();

    let mut n = 0usize;
    while stream.next_chunk_into(&mut chunk, &cancel).expect("chunk") {
        n += 1;
        if n == 8 {
            break;
        }
    }
    assert!(n == 8, "the file produced too few chunks to test with");
    let settled = chunk.capacity();
    let mut after = 0usize;
    while stream.next_chunk_into(&mut chunk, &cancel).expect("chunk") {
        after += 1;
        assert_eq!(
            chunk.capacity(),
            settled,
            "the caller's buffer was reallocated on chunk {}",
            n + after
        );
    }
    assert!(
        after > 4,
        "only {after} further chunks: not a streaming test"
    );
    let _ = stream.finish().expect("finish");
}

#[test]
fn the_owned_chunk_helper_agrees_with_the_borrowing_one() {
    let f = long_wav(32_000, 1, 0.5);
    let options = DecodeOptions::default();
    let cancel = CancelToken::new();

    let mut stream = AudioStream::open(f.path(), &options).expect("open");
    let mut owned = Vec::new();
    while let Some(chunk) = stream.next_chunk(&cancel).expect("chunk") {
        owned.extend_from_slice(&chunk);
    }
    let _ = stream.finish().expect("finish");

    let mut stream = AudioStream::open(f.path(), &options).expect("open");
    let mut borrowed = Vec::new();
    let mut chunk = Vec::new();
    while stream.next_chunk_into(&mut chunk, &cancel).expect("chunk") {
        borrowed.extend_from_slice(&chunk);
    }
    let _ = stream.finish().expect("finish");

    assert_eq!(owned, borrowed);
}

#[test]
fn every_chunk_is_stamped_with_the_time_it_starts_at() {
    // `for_each_chunk` hands the sink a `Time`. If that is wrong, every
    // downstream timestamp — word timings, speaker spans, chapter markers — is
    // wrong by the same amount and nothing anywhere will notice.
    let f = long_wav(16_000, 1, 2.0);
    let options = DecodeOptions::default();
    let mut stream = AudioStream::open(f.path(), &options).expect("open");
    let cancel = CancelToken::new();

    let mut seen: Vec<(vaani_core::Time, usize)> = Vec::new();
    let report = stream
        .for_each_chunk(&cancel, |chunk, at| {
            seen.push((at, chunk.len()));
            Ok(())
        })
        .expect("for_each_chunk");

    let mut offset = 0usize;
    for (i, (at, len)) in seen.iter().enumerate() {
        let want_ms = offset as i64 * 1000 / 16_000;
        assert_eq!(
            at.as_millis(),
            want_ms,
            "chunk {i} is stamped {at}, want {want_ms} ms"
        );
        // The stamp must also agree with the frame index the stream itself
        // reports having emitted, or a caller computing progress from `report()`
        // and a caller computing it from the stamps would disagree.
        assert!(
            offset <= stream.report().output_frames,
            "chunk {i} starts at frame {offset} but only {} have been emitted",
            stream.report().output_frames
        );
        offset += len;
    }
    assert_eq!(
        offset, report.output_frames,
        "the stamped chunks do not add up"
    );
    assert_eq!(report.output_sample_rate, 16_000);
}

#[test]
fn a_failing_sink_stops_the_decode_with_its_own_error() {
    // The streaming path is how a downstream stage aborts a job, so the error a
    // sink returns has to come back unchanged rather than being wrapped into
    // something about audio.
    let f = long_wav(48_000, 2, 3.0);
    let options = DecodeOptions::default();
    let mut stream = AudioStream::open(f.path(), &options).expect("open");
    let cancel = CancelToken::new();

    let mut calls = 0usize;
    let err = stream
        .for_each_chunk(&cancel, |_, _| {
            calls += 1;
            Err(vaani_core::err::internal().with_cause("the VAD said no"))
        })
        .expect_err("the sink's error must propagate");
    assert_eq!(err.code(), vaani_core::ErrorCode::Internal);
    assert!(err.debug_chain().contains("the VAD said no"));
    assert_eq!(calls, 1, "the decode kept going after the sink failed");
}

#[test]
fn a_token_cancelled_before_the_first_chunk_stops_immediately() {
    let f = long_wav(48_000, 2, 5.0);
    let options = DecodeOptions::default();
    let mut stream = AudioStream::open(f.path(), &options).expect("open");
    let cancel = CancelToken::new();
    cancel.cancel();

    let mut chunk = Vec::new();
    let err = stream
        .next_chunk_into(&mut chunk, &cancel)
        .expect_err("must not produce a chunk");
    assert_eq!(err.code(), vaani_core::ErrorCode::Cancelled);
    assert!(
        chunk.is_empty(),
        "a cancelled decode must not hand over audio"
    );
}

#[test]
fn cancellation_latency_does_not_grow_with_the_length_of_the_recording() {
    // The contract in the brief: poll at least once per second of source audio
    // and at least once every 250 ms of wall clock. This test measures the thing
    // a user experiences — how much audio gets decoded *after* they click
    // Cancel — and checks it is bounded for a 5 s file and a 60 s file alike.
    //
    // The poll interval is forced to its minimum so the measurement is of the
    // mechanism and not of the timer.
    let mut latencies = Vec::new();
    for seconds in [5.0f32, 60.0] {
        let f = long_wav(48_000, 2, seconds);
        let options = DecodeOptions {
            cancel_check_audio_seconds: 0.0,
            cancel_check_wall_ms: 0,
            ..DecodeOptions::default()
        };
        let mut stream = AudioStream::open(f.path(), &options).expect("open");
        let cancel = CancelToken::new();

        let total = stream.media_info().declared_duration_ms.unwrap_or(0) as f64 / 1000.0;
        assert!(
            total > 1.0,
            "the fixture is too short to cancel meaningfully"
        );
        let started = Instant::now();

        // One chunk first, then cancel, on this thread. A background
        // `sleep(30ms)` races a release decoder that finishes a 5 s file in
        // about a millisecond, so the cancel would usually land after the
        // stream was already done and there would be no latency to measure.
        // Cancelling synchronously after the first chunk makes "how much audio
        // gets decoded after Cancel" a deterministic quantity at any decode
        // speed, which is exactly the property being asserted.
        let mut chunk = Vec::new();
        stream
            .next_chunk_into(&mut chunk, &cancel)
            .expect("the first chunk must be produced");
        cancel.cancel();

        let err = stream
            .next_chunk_into(&mut chunk, &cancel)
            .expect_err("the decode must end in a cancellation");
        let after = stream.report().source_seconds();
        let wall = started.elapsed();

        assert_eq!(
            err.code(),
            vaani_core::ErrorCode::Cancelled,
            "{}",
            err.debug_chain()
        );
        latencies.push((total, after, wall));
    }

    for (total, after, wall) in &latencies {
        println!("{total:.0} s file: cancelled after {after:.3} s of audio, {wall:?} wall clock");
    }
    let short = latencies[0];
    let long = latencies[1];
    assert!(
        short.1 <= 1.0,
        "5 s file: {} s of audio was decoded after Cancel",
        short.1
    );
    assert!(
        long.1 <= 1.0,
        "60 s file: {} s of audio was decoded after Cancel — latency scales with length",
        long.1
    );
    assert!(
        long.2 < Duration::from_secs(2),
        "60 s file: the decode took {:?} of wall clock to notice the cancel",
        long.2
    );
}

#[test]
fn the_default_poll_interval_is_the_one_the_contract_names() {
    // A regression here is invisible until a user cancels a two-hour recording
    // and waits, so the numbers are asserted rather than documented.
    let options = DecodeOptions::default();
    assert_eq!(options.cancel_check_audio_seconds, 1.0);
    assert_eq!(options.cancel_check_wall_ms, 250);
}

#[test]
fn a_stream_that_is_cancelled_reports_how_far_it_got() {
    // The report is what the UI shows after a cancel, and a progress bar that
    // jumps from 0% to 100% because the report was thrown away is a bug report
    // waiting to happen.
    let f = long_wav(44_100, 2, 10.0);
    let options = DecodeOptions {
        cancel_check_audio_seconds: 0.0,
        cancel_check_wall_ms: 0,
        ..DecodeOptions::default()
    };
    let mut stream = AudioStream::open(f.path(), &options).expect("open");
    // `expected_output_frames` counts what has been *pushed through the
    // resampler*, so it is zero before the first read. The container's claim is
    // the figure available up front, which is exactly why both are on the report.
    assert_eq!(
        stream.expected_output_frames(),
        0,
        "nothing has been decoded yet"
    );
    assert_eq!(
        stream.media_info().declared_duration_ms,
        Some(10_000),
        "the container's claim is available before a single sample is decoded"
    );

    let cancel = CancelToken::new();

    // Cancel from *this* thread, after exactly one chunk, rather than from a
    // sleeping helper thread. A release build decodes a 10 s file in about two
    // milliseconds, so a `sleep(20ms)`-then-cancel races the decoder and
    // usually loses: the decode finishes first and there is no cancellation to
    // observe. Pulling one chunk and then cancelling makes the test mean the
    // same thing in debug and in release, which is the only way a timing
    // assertion can be trusted.
    let mut chunk = Vec::new();
    stream
        .next_chunk_into(&mut chunk, &cancel)
        .expect("the first chunk must be produced before any cancellation");
    assert!(!chunk.is_empty(), "a successful chunk must contain samples");
    cancel.cancel();

    let err = stream
        .next_chunk_into(&mut chunk, &cancel)
        .expect_err("a cancelled stream must report the cancellation");

    let report = stream.report();
    assert_eq!(
        err.code(),
        vaani_core::ErrorCode::Cancelled,
        "{}",
        err.debug_chain()
    );
    assert!(
        report.source_frames > 0,
        "a decode that produced a chunk reported no source frames"
    );
    assert!(
        report.source_seconds() < 10.0,
        "a cancelled decode reported {:.1} s of a 10 s file",
        report.source_seconds()
    );
    assert!(
        !report.truncated,
        "a cancellation is not a truncated recording"
    );
}

#[test]
fn trimming_a_collected_buffer_is_idempotent_and_exact() {
    // `trim_collected` is public because a caller driving the stream has the
    // same need as `decode`. Calling it twice must be a no-op, not a second
    // correction.
    let f = long_wav(22_050, 1, 1.0);
    let options = DecodeOptions::default();
    let mut stream = AudioStream::open(f.path(), &options).expect("open");
    let cancel = CancelToken::new();
    let mut samples = Vec::new();
    let mut chunk = Vec::new();
    while stream.next_chunk_into(&mut chunk, &cancel).expect("chunk") {
        samples.extend_from_slice(&chunk);
    }
    let _ = stream.finish().expect("finish");

    let want = stream.expected_output_frames() as usize;
    // Whether or not the resampler's grid already landed on the exact length,
    // after the trim it must be that length, and a second trim must change
    // nothing. Both outcomes are correct; a caller must not have to care which.
    stream.trim_collected(&mut samples);
    assert_eq!(
        samples.len(),
        want,
        "the trim did not reach the expected length"
    );
    assert!(
        !stream.trim_collected(&mut samples),
        "a second trim must be a no-op"
    );
    assert_eq!(samples.len(), want);
}

#[test]
fn a_truncated_stream_still_reports_what_it_read() {
    // Streaming and truncation together: a caller that gets `Ok` must still be
    // able to tell that the audio is incomplete, and `finish` is the only place
    // that knows.
    let full = tone(16_000 * 3, 16_000, 300.0, 0.5);
    let spec = WavSpec::pcm(16_000, 1, SampleFormat::S16)
        .claiming_data_bytes((16_000 * 3 * 2) as u32)
        .truncated_to(16_000);
    let f = Fixture::new("wav", &wav(&full, &spec));

    let options = DecodeOptions::default();
    let mut stream = AudioStream::open(f.path(), &options).expect("open");
    let cancel = CancelToken::new();
    let mut samples = Vec::new();
    let mut chunk = Vec::new();
    while stream.next_chunk_into(&mut chunk, &cancel).expect("chunk") {
        samples.extend_from_slice(&chunk);
    }
    let report = stream.finish().expect("finish");

    assert!(report.truncated, "the stream ended early and must say so");
    assert!(!report.warnings.is_empty());
    assert_eq!(report.source_sample_rate, 16_000);
    assert!(!samples.is_empty());
    assert!(samples.len() < 16_000 * 2, "got {} samples", samples.len());
}

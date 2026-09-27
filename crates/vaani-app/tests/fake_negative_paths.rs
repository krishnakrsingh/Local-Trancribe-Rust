//! The fakes must be able to say "no" as well as "yes".
//!
//! Every fake in this crate is biased towards making the pipeline *run* — a VAD
//! that always finds speech, an ASR that always returns words, a diarizer that
//! always finds two speakers. That bias is convenient and it is dangerous: a
//! stage that can only ever succeed cannot be observed failing, and the bugs
//! that hides are the ones that only appear on a recording of silence, an
//! empty directory, or a user who cancels.
//!
//! These tests pin the negative paths. Most are one line each, which is the
//! point: a negative case that is cheap to write is a negative case that gets
//! written.

use vaani_app::fakes::{sine_audio, FakeVad};
use vaani_core::pipeline::VadEngine;
use vaani_core::{CancelToken, Time};

#[test]
fn a_silent_detector_finds_nothing_in_a_signal() {
    // The audio is a real, non-empty sine wave. Only the detector's answer
    // differs, which isolates the behaviour under test from the input.
    let audio = sine_audio(3);
    assert!(!audio.samples.is_empty(), "the fixture must not be empty");

    let tl = FakeVad::silence()
        .detect(&audio, &CancelToken::new())
        .expect("detection itself should not fail");

    assert!(
        tl.segments.is_empty(),
        "a silent detector must not report segments, got {:?}",
        tl.segments
    );
    assert_eq!(tl.speech_duration_ms, 0);
    assert_eq!(tl.speech_ratio(audio.duration()), 0.0);
}

#[test]
fn a_speaking_detector_covers_the_whole_buffer() {
    // The counterpart, so `silence` cannot pass by always returning empty.
    let audio = sine_audio(3);
    let tl = FakeVad::speech()
        .detect(&audio, &CancelToken::new())
        .expect("detection itself should not fail");

    assert_eq!(tl.segments.len(), 1);
    assert_eq!(tl.segments[0].start, Time::ZERO);
    assert_eq!(tl.segments[0].end, audio.duration());
    assert!((tl.speech_ratio(audio.duration()) - 1.0).abs() < 1e-9);
}

#[test]
fn no_detector_reports_speech_in_an_empty_buffer() {
    // Empty input must not produce a zero-length segment. A segment where
    // `start == end` is a division by zero waiting for someone downstream.
    let empty = vaani_core::AudioBuffer::mono(Vec::new(), 16_000, Time::ZERO);

    for vad in [FakeVad::speech(), FakeVad::silence()] {
        let tl = vad
            .detect(&empty, &CancelToken::new())
            .expect("detection itself should not fail");
        assert!(
            tl.segments.is_empty(),
            "{} invented a segment in an empty buffer",
            vad.id()
        );
        for s in &tl.segments {
            assert!(s.start < s.end, "zero-length segment: {s:?}");
        }
    }
}

#[test]
fn a_cancelled_job_stops_before_any_work() {
    // Cancellation has to win over the answer, or a two-hour job keeps
    // computing after the user pressed stop.
    let token = CancelToken::new();
    token.cancel();

    let err = FakeVad::speech()
        .detect(&sine_audio(60), &token)
        .expect_err("a cancelled job must not return a timeline");

    assert_eq!(err.code(), vaani_core::ErrorCode::Cancelled);
}

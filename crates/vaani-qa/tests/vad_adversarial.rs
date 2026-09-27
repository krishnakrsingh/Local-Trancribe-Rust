//! VAD adversarial tests: silence, noise, speech-like audio, cancellation.
//!
//! All audio is synthesised by `vaani_qa::fixtures`. Assertions pin the
//! contract: 16 kHz required, absolute sorted timelines, `speech_ratio` in
//! `[0, 1]`, silence ratio ≈ 0, no panic on noise.

use vaani_core::media::AudioBuffer;
use vaani_core::pipeline::VadEngine;
use vaani_core::{CancelToken, Time};
use vaani_qa::fixtures::{self, FIXTURE_SAMPLE_RATE};
use vaani_vad::{ChunkPolicy, Chunker, EnergyVad};

fn speech_buffer() -> AudioBuffer {
    AudioBuffer::mono(
        fixtures::speech_like(6.0, FIXTURE_SAMPLE_RATE, 7),
        FIXTURE_SAMPLE_RATE,
        Time::ZERO,
    )
}

#[test]
fn engine_requires_16khz() {
    let vad = EnergyVad::new();
    assert_eq!(vad.sample_rate(), 16_000);
    assert_eq!(vaani_vad::SAMPLE_RATE, 16_000);
    assert!(!vad.id().is_empty());
    assert!(!vad.name().is_empty());
}

#[test]
fn long_silence_has_speech_ratio_near_zero() {
    let vad = EnergyVad::new();
    // 60 s of digital silence.
    let audio = AudioBuffer::mono(fixtures::silence(60.0), FIXTURE_SAMPLE_RATE, Time::ZERO);
    let timeline = vad
        .detect(&audio, &CancelToken::new())
        .expect("detect silence");
    assert!(
        timeline.segments.is_empty(),
        "silence produced {} segments",
        timeline.segments.len()
    );
    assert!(timeline.speech_ratio(audio.duration()) < 0.01);
    // Silence must not produce ASR chunks.
    let chunks = Chunker::new(ChunkPolicy::default()).plan(&timeline, audio.duration());
    assert!(chunks.is_empty(), "silence must not produce ASR chunks");
}

#[test]
fn noisy_audio_never_panics_and_stays_in_range() {
    let vad = EnergyVad::new();
    for (seed, peak) in [(1u64, 0.1f32), (2, 0.5), (3, 0.9)] {
        let audio = AudioBuffer::mono(
            fixtures::noise(16_000 * 5, seed, peak),
            FIXTURE_SAMPLE_RATE,
            Time::ZERO,
        );
        let timeline = vad
            .detect(&audio, &CancelToken::new())
            .expect("detect noise");
        let ratio = timeline.speech_ratio(audio.duration());
        assert!(
            (0.0..=1.0).contains(&ratio),
            "seed {seed}: ratio {ratio} out of range"
        );
        for w in timeline.segments.windows(2) {
            assert!(w[0].end <= w[1].start, "seed {seed}: overlapping segments");
        }
    }
}

#[test]
fn speech_like_audio_yields_absolute_sorted_timeline() {
    let vad = EnergyVad::new();
    let audio = speech_buffer();
    let timeline = vad
        .detect(&audio, &CancelToken::new())
        .expect("detect speech");
    let ratio = timeline.speech_ratio(audio.duration());
    assert!((0.0..=1.0).contains(&ratio), "ratio {ratio}");
    for w in timeline.segments.windows(2) {
        assert!(w[0].start <= w[0].end);
        assert!(w[0].end <= w[1].start, "segments must be sorted and merged");
    }
    let end = audio.start_time.saturating_add(audio.duration());
    for s in &timeline.segments {
        assert!(s.start >= audio.start_time && s.end <= end);
    }
    println!(
        "speech-like 6 s: {} segments, ratio={ratio:.3}",
        timeline.segments.len()
    );
}

#[test]
fn speech_shaped_voice_over_room_is_detected() {
    // Positive control: a harmonic-comb voice at −25 dB over a −100 dB room
    // floor must produce speech. (Pure tones and digital-silence gaps are
    // rejected by design — see fixtures::voice — so they are not used here.)
    let vad = EnergyVad::new();
    let audio = AudioBuffer::mono(
        fixtures::meeting(
            6 * FIXTURE_SAMPLE_RATE as usize,
            FIXTURE_SAMPLE_RATE,
            7,
            130.0,
        ),
        FIXTURE_SAMPLE_RATE,
        Time::ZERO,
    );
    let timeline = vad
        .detect(&audio, &CancelToken::new())
        .expect("detect voice");
    assert!(
        !timeline.segments.is_empty(),
        "speech-shaped voice produced no segments"
    );
    let ratio = timeline.speech_ratio(audio.duration());
    assert!(ratio > 0.05 && ratio <= 1.0, "implausible ratio {ratio}");
    let chunks = Chunker::new(ChunkPolicy::default()).plan(&timeline, audio.duration());
    assert!(
        !chunks.is_empty(),
        "detected speech must produce ASR chunks"
    );
    println!(
        "voice 6 s: {} segments, ratio={ratio:.3}",
        timeline.segments.len()
    );
}

#[test]
fn chunker_bounds_chunks_and_covers_speech() {
    use vaani_core::pipeline::{SpeechSegment, SpeechTimeline};
    let policy = ChunkPolicy::default();
    // One 90 s speech block: the chunker must split at the 60 s cap.
    let timeline = SpeechTimeline {
        segments: vec![SpeechSegment {
            start: Time(0),
            end: Time(90_000),
            confidence: None,
        }],
        speech_duration_ms: 90_000,
    };
    let chunks = Chunker::new(policy).plan(&timeline, Time(90_000));
    assert!(
        chunks.len() >= 2,
        "a 90 s block must be split, got {}",
        chunks.len()
    );
    // The cap applies pre-pad; padding (pad_ms each side) legitimately extends
    // a chunk, so the bound is max + 2 * pad.
    let bound = policy.max_ms + 2 * policy.pad_ms;
    for c in &chunks {
        assert!(
            (c.end.as_millis() - c.start.as_millis()) <= bound,
            "chunk {}..{} exceeds max+pad {bound}",
            c.start.as_millis(),
            c.end.as_millis()
        );
    }
    // Chunks cover the speech block contiguously in source-media coordinates.
    assert_eq!(chunks.first().map(|c| c.start), Some(Time(0)));
    assert_eq!(chunks.last().map(|c| c.end), Some(Time(90_000)));
}

#[test]
fn cancelled_vad_reports_cancelled() {
    let vad = EnergyVad::new();
    let audio = speech_buffer();
    let cancel = CancelToken::new();
    cancel.cancel();
    let err = vad.detect(&audio, &cancel).expect_err("must cancel");
    assert!(err.is_cancelled());
}

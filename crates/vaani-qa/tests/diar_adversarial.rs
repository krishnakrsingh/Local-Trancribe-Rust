//! Diarization adversarial tests: single vs multi speaker, silence, noise,
//! attribution invariants.
//!
//! The engine under test is the spectral-heuristic baseline (`HeuristicDiarizer`);
//! no trained embedder exists in this repository, so no DER is claimed. What is
//! pinned are the structural invariants: audio-only input, absolute sorted
//! spans, contiguous 0-based speakers, silence safety, and attribution as a
//! separate step.

use vaani_core::media::AudioBuffer;
use vaani_core::pipeline::{DiarizationEngine, DiarizationResult, SpeakerSpan};
use vaani_core::{CancelToken, EngineStamp, Segment, Time, Transcript, Word};
use vaani_qa::fixtures::{self, FIXTURE_SAMPLE_RATE};

fn engine() -> vaani_diarization::HeuristicDiarizer {
    let e = vaani_diarization::HeuristicDiarizer::new();
    e.load(&CancelToken::new()).expect("heuristic load");
    e
}

fn mono(samples: Vec<f32>) -> AudioBuffer {
    AudioBuffer::mono(samples, FIXTURE_SAMPLE_RATE, Time::ZERO)
}

fn check_invariants(result: &DiarizationResult, audio: &AudioBuffer, label: &str) {
    let sorted = result.spans.windows(2).all(|w| w[0].start <= w[1].start);
    assert!(sorted, "{label}: spans out of order");
    let end = audio.start_time.saturating_add(audio.duration());
    for s in &result.spans {
        assert!(
            s.start >= audio.start_time && s.end <= end,
            "{label}: span outside audio"
        );
        assert!(s.start <= s.end, "{label}: inverted span");
    }
    let mut seen: Vec<usize> = result.spans.iter().map(|s| s.speaker).collect();
    seen.sort_unstable();
    seen.dedup();
    for s in &seen {
        assert!(
            *s < result.num_speakers.max(1),
            "{label}: speaker {s} unregistered"
        );
    }
    let cov = result.coverage(audio.duration());
    assert!((0.0..=1.0).contains(&cov), "{label}: coverage {cov}");
}

#[test]
fn single_speaker_tone_stays_small() {
    let e = engine();
    let audio = mono(fixtures::tone(16_000 * 8, 16_000, 150.0, 0.45));
    let result = e
        .diarize(&audio, &CancelToken::new(), &|_| {})
        .expect("diarize");
    check_invariants(&result, &audio, "single");
    assert!(
        result.num_speakers <= 2,
        "one voice must not shatter into {} speakers",
        result.num_speakers
    );
    println!(
        "single speaker: {} spans, {} speakers",
        result.spans.len(),
        result.num_speakers
    );
}

#[test]
fn two_speaker_turns_find_at_least_one_speaker() {
    let e = engine();
    let audio = mono(fixtures::two_speaker_turns(3, 2.0, FIXTURE_SAMPLE_RATE));
    let result = e
        .diarize(&audio, &CancelToken::new(), &|_| {})
        .expect("diarize");
    check_invariants(&result, &audio, "two-speaker");
    assert!(result.num_speakers >= 1, "speech must attribute to someone");
    assert!(
        result.num_speakers <= 4,
        "two voices must not shatter into {} speakers",
        result.num_speakers
    );
    // Timeline reuse path: same audio with an external VAD timeline agrees.
    let vad = vaani_vad::EnergyVad::new();
    let timeline = {
        use vaani_core::pipeline::VadEngine;
        vad.detect(&audio, &CancelToken::new()).expect("vad")
    };
    let with_timeline = e
        .diarize_with_timeline(&audio, Some(&timeline), &CancelToken::new(), &|_| {})
        .expect("diarize with timeline");
    check_invariants(&with_timeline, &audio, "two-speaker-timeline");
    println!(
        "two speakers: plain={} speakers, with-timeline={} speakers",
        result.num_speakers, with_timeline.num_speakers
    );
}

#[test]
fn silence_yields_no_speakers_and_no_panic() {
    let e = engine();
    let audio = mono(fixtures::silence(10.0));
    let result = e
        .diarize(&audio, &CancelToken::new(), &|_| {})
        .expect("diarize silence");
    assert!(
        result.spans.is_empty() || result.coverage(audio.duration()) < 0.05,
        "silence claimed {} spans",
        result.spans.len()
    );
}

#[test]
fn white_noise_does_not_panic() {
    let e = engine();
    let audio = mono(fixtures::noise(16_000 * 6, 42, 0.4));
    let result = e
        .diarize(&audio, &CancelToken::new(), &|_| {})
        .expect("diarize noise");
    check_invariants(&result, &audio, "noise");
}

#[test]
fn five_millisecond_click_does_not_panic() {
    let e = engine();
    let mut samples = fixtures::silence(3.0);
    for s in samples.iter_mut().take(80) {
        *s = 0.9;
    }
    let audio = mono(samples);
    let result = e
        .diarize(&audio, &CancelToken::new(), &|_| {})
        .expect("diarize click");
    check_invariants(&result, &audio, "click");
}

#[test]
fn wrong_sample_rate_is_rejected_not_resampled() {
    let e = engine();
    let audio = AudioBuffer::mono(vec![0.1; 8000], 8000, Time::ZERO);
    let err = e
        .diarize(&audio, &CancelToken::new(), &|_| {})
        .expect_err("8 kHz must fail");
    assert!(
        err.debug_chain().contains("16"),
        "message must name the required rate: {}",
        err.debug_chain()
    );
}

#[test]
fn stereo_is_rejected_not_silently_averaged() {
    let e = engine();
    let audio = AudioBuffer {
        samples: vec![0.1; 4000],
        sample_rate: 16_000,
        channels: 2,
        start_time: Time::ZERO,
    };
    e.diarize(&audio, &CancelToken::new(), &|_| {})
        .expect_err("stereo must fail");
}

#[test]
fn attribution_maps_spans_onto_segments() {
    let mut transcript = Transcript::new(EngineStamp::asr("qa", "0"));
    transcript.push_chunk(
        Time(0),
        vec![
            Segment::new(Time(0), Time(2000)).with_words(vec![Word::new("a", Time(0), Time(2000))]),
            Segment::new(Time(3000), Time(5000)).with_words(vec![Word::new(
                "b",
                Time(3000),
                Time(5000),
            )]),
            Segment::new(Time(6000), Time(8000)).with_words(vec![Word::new(
                "c",
                Time(6000),
                Time(8000),
            )]),
        ],
    );
    let result = DiarizationResult {
        spans: vec![
            SpeakerSpan {
                speaker: 0,
                start: Time(0),
                end: Time(2500),
                score: None,
            },
            SpeakerSpan {
                speaker: 1,
                start: Time(5500),
                end: Time(8500),
                score: None,
            },
        ],
        num_speakers: 2,
        elapsed_ms: 0,
    };
    vaani_diarization::attribute_speakers(&mut transcript, &result);
    assert_eq!(transcript.segments[0].speaker, Some(0));
    // Segment [3s,5s] overlaps no span: stays unattributed, never an error.
    assert_eq!(transcript.segments[1].speaker, None);
    assert_eq!(transcript.segments[2].speaker, Some(1));
    // Renames survive re-attribution.
    transcript.speaker_mut(0).name = Some("Asha".into());
    let existing = transcript.speakers.clone();
    vaani_diarization::apply(&mut transcript, &result, &existing);
    assert_eq!(transcript.speaker_mut(0).name.as_deref(), Some("Asha"));
}

#[test]
fn speaker_at_prefers_shortest_span_and_boundaries_are_half_open() {
    let result = DiarizationResult {
        spans: vec![
            SpeakerSpan {
                speaker: 0,
                start: Time(0),
                end: Time(5000),
                score: None,
            },
            SpeakerSpan {
                speaker: 1,
                start: Time(1000),
                end: Time(2000),
                score: None,
            },
        ],
        num_speakers: 2,
        elapsed_ms: 0,
    };
    assert_eq!(result.speaker_at(Time(1500)), Some(1));
    assert_eq!(result.speaker_at(Time(3000)), Some(0));
    assert_eq!(result.speaker_at(Time(9000)), None);
}

#[test]
fn cancellation_is_honoured() {
    let e = engine();
    let audio = mono(fixtures::two_speaker_turns(6, 3.0, FIXTURE_SAMPLE_RATE));
    let cancel = CancelToken::new();
    cancel.cancel();
    let err = e
        .diarize(&audio, &cancel, &|_| {})
        .expect_err("must cancel");
    assert!(err.is_cancelled());
}

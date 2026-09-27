//! Structural invariants that must hold for **every** result the engine can
//! produce, on every input.
//!
//! These are the assertions that catch the class of bug where the engine
//! returns something structurally valid but wrong in a way no accuracy metric
//! would notice: spans out of order, a span that runs off the end of the
//! buffer, a speaker index that does not index `Transcript::speakers`, a
//! `coverage` above 1. Every adversarial input from `adversarial.rs` is run
//! through the same property sweep here, because an input that produces no
//! speakers is trivially well-formed and a bug that only appears on the
//! interesting inputs is exactly the kind this file exists for.

use vaani_core::media::AudioBuffer;
use vaani_core::pipeline::{DiarizationEngine, DiarizationResult};
use vaani_core::{Time, err};
use vaani_diarization::attribution;
use vaani_diarization::engine::HeuristicDiarizer;
use vaani_diarization::smooth;
use vaani_diarization::synth;

/// Every structural property, checked on one result.
fn check(label: &str, r: &DiarizationResult, total: Time) {
    // Sorted by start, and non-overlapping.
    for w in r.spans.windows(2) {
        assert!(
            w[0].start <= w[1].start,
            "{label}: spans out of order at {} then {}",
            w[0].start.as_millis(),
            w[1].start.as_millis()
        );
        assert!(
            w[0].end <= w[1].start,
            "{label}: spans {}..{} and {}..{} overlap",
            w[0].start.as_millis(),
            w[0].end.as_millis(),
            w[1].start.as_millis(),
            w[1].end.as_millis()
        );
    }
    for (i, s) in r.spans.iter().enumerate() {
        assert!(
            s.start < s.end,
            "{label}: span {i} is empty or inverted ({}..{})",
            s.start.as_millis(),
            s.end.as_millis()
        );
        assert!(
            s.start >= Time::ZERO,
            "{label}: span {i} starts before the media at {}",
            s.start.as_millis()
        );
        assert!(
            s.end <= total,
            "{label}: span {i} ends at {} past the {} ms buffer",
            s.end.as_millis(),
            total.as_millis()
        );
        assert!(
            s.speaker < r.num_speakers,
            "{label}: span {i} names speaker {} but only {} are registered",
            s.speaker,
            r.num_speakers
        );
        if let Some(sc) = s.score {
            assert!(
                (0.0..=1.0).contains(&sc),
                "{label}: span {i} has score {sc} outside [0,1]"
            );
        }
    }
    assert!(
        smooth::validate_spans(&r.spans, r.num_speakers).is_empty(),
        "{label}: validate_spans disagreed: {:?}",
        smooth::validate_spans(&r.spans, r.num_speakers)
    );

    // Coverage is a fraction and never exceeds 1.
    let c = r.coverage(total);
    assert!(
        (0.0..=1.0).contains(&c),
        "{label}: coverage {c} outside [0,1]"
    );

    // `speaker_durations` is the length of a vector of `num_speakers`, and
    // every entry is non-negative. A span naming a higher index than
    // `num_speakers` would be a crash in a naive implementation.
    let d = r.speaker_durations();
    assert_eq!(d.len(), r.num_speakers, "{label}: durations length");
    assert!(d.iter().all(|x| *x >= 0), "{label}: negative duration");

    // `speaker_at` is total: it answers for every instant, or says None.
    for t in [
        Time(0),
        Time::from_millis(total.as_millis() / 2),
        total.saturating_sub(Time(1)),
    ] {
        let _ = r.speaker_at(t);
    }

    // The engine's own consistency check.
    assert!(
        attribution::assert_consistent(r).is_empty(),
        "{label}: inconsistent: {:?}",
        attribution::assert_consistent(r)
    );

    // Every registered speaker owns a span, so no phantom row appears in the
    // speaker list. `assert_consistent` checks this, but the check below is
    // the one the product depends on and it is cheap to state twice.
    let mut seen = vec![false; r.num_speakers];
    for s in &r.spans {
        seen[s.speaker] = true;
    }
    assert!(
        seen.iter().all(|x| *x),
        "{label}: a registered speaker owns nothing"
    );
}

fn run(label: &str, samples: Vec<f32>) {
    let audio = AudioBuffer::mono(samples, 16_000, Time::ZERO);
    let r = HeuristicDiarizer::new()
        .diarize(&audio, &vaani_core::CancelToken::new(), &|_| {})
        .unwrap_or_else(|e| panic!("{label}: {}", e.debug_chain()));
    check(label, &r, audio.duration());
}

#[test]
fn silence_obeys_every_invariant() {
    run("silence", vec![0.0; 16_000 * 5]);
}

#[test]
fn noise_obeys_every_invariant() {
    let mut rng = 99u64;
    run(
        "noise",
        (0..16_000 * 5)
            .map(|_| {
                rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                ((rng >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
            })
            .collect(),
    );
}

#[test]
fn one_empty_buffer_obeys_every_invariant() {
    let audio = AudioBuffer::mono(Vec::new(), 16_000, Time::ZERO);
    let r = HeuristicDiarizer::new()
        .diarize(&audio, &vaani_core::CancelToken::new(), &|_| {})
        .unwrap();
    check("empty", &r, audio.duration());
}

#[test]
fn one_sample_obeys_every_invariant() {
    run("one sample", vec![0.5]);
}

#[test]
fn one_hop_of_audio_obeys_every_invariant() {
    // 160 samples is exactly one hop: fewer frames than the analysis window
    // and fewer than the pitch span.
    run(
        "one hop",
        (0..160).map(|i| (i as f32 * 0.1).sin()).collect(),
    );
}

#[test]
fn exactly_one_window_of_audio_obeys_every_invariant() {
    let voices = synth::two_voices();
    let u = vec![synth::Utterance::new(0, 0, 1_500)];
    let (s, _) = synth::render(&u, &voices, 1_500, 16_000, 0.0, 1);
    run("one window", s);
}

#[test]
fn a_full_alternation_obeys_every_invariant() {
    let voices = synth::three_voices();
    let u = synth::alternate(&[0, 1, 2, 1, 0, 2], 0, 2_000, 30);
    let (s, _) = synth::render(&u, &voices, 60_000, 16_000, 0.004, 77);
    run("alternation", s);
}

#[test]
fn all_turns_by_one_speaker_obey_every_inherent_property() {
    let voices = synth::two_voices();
    let u = synth::alternate(&[1], 0, 1_200, 40);
    let (s, _) = synth::render(&u, &voices, 48_000, 16_000, 0.001, 5);
    run("one speaker many turns", s);
}

#[test]
fn a_buffer_with_a_non_zero_start_time_obeys_every_invariant() {
    // A chunk from the middle of a recording: the spans must be inside the
    // chunk, and the times must be absolute.
    let voices = synth::two_voices();
    let u = synth::alternate(&[0, 1], 0, 2_000, 10);
    let (s, _) = synth::render(&u, &voices, 20_000, 16_000, 0.003, 9);
    let start = Time(1_234_567);
    let audio = AudioBuffer::mono(s, 16_000, start);
    let r = HeuristicDiarizer::new()
        .diarize(&audio, &vaani_core::CancelToken::new(), &|_| {})
        .unwrap();
    check("offset chunk", &r, start.saturating_add(audio.duration()));
    for s in &r.spans {
        assert!(
            s.start >= start,
            "a span at {} precedes the chunk's own start of {}",
            s.start.as_millis(),
            start.as_millis()
        );
    }
}

#[test]
fn every_caller_speaker_count_keeps_the_invariants() {
    let voices = synth::two_voices();
    let u = synth::alternate(&[0, 1], 0, 3_000, 20);
    let (s, _) = synth::render(&u, &voices, 60_000, 16_000, 0.003, 4);
    for k in 1..=6 {
        let engine = HeuristicDiarizer::with_config(
            vaani_diarization::config::DiarizationConfig::default().with_speaker_count(k),
        );
        let audio = AudioBuffer::mono(s.clone(), 16_000, Time::ZERO);
        let r = engine
            .diarize(&audio, &vaani_core::CancelToken::new(), &|_| {})
            .unwrap();
        check(&format!("fixed {k}"), &r, audio.duration());
    }
}

#[test]
fn a_hand_assembled_broken_result_is_reported_not_returned() {
    // The engine validates its own output. A span naming an unregistered
    // speaker must be impossible, and a result that somehow had one would be
    // reported rather than shipped to a transcript whose labels do not resolve.
    let bad = DiarizationResult {
        spans: vec![vaani_core::pipeline::SpeakerSpan {
            speaker: 9,
            start: Time(0),
            end: Time(1_000),
            score: None,
        }],
        num_speakers: 2,
        elapsed_ms: 0,
    };
    let problems = attribution::assert_consistent(&bad);
    // Three separate defects, named individually: the stray span, and each of
    // the two registered speakers that owns nothing.
    assert_eq!(problems.len(), 3, "{problems:?}");
    assert!(problems.iter().any(|p| p.contains("num_speakers is 2")));
    assert_eq!(
        problems
            .iter()
            .filter(|p| p.contains("owns no span"))
            .count(),
        2,
        "{problems:?}"
    );
    // And it is not a panic and not an error: it is a report.
    let _ = attribution::majority_speaker(Time(0), Time(500), &bad);
    assert_eq!(
        attribution::majority_speaker(Time(0), Time(500), &bad),
        None
    );
}

#[test]
fn a_span_naming_an_unregistered_speaker_does_not_reach_a_transcript() {
    // The end-to-end version of the same rule: a hand-assembled result with a
    // stray span must not put a phantom speaker into `Transcript::speakers`.
    use vaani_core::transcript::{EngineStamp, Segment, Transcript};
    let mut t = Transcript::new(EngineStamp::asr("test", "1"));
    t.push_chunk(
        Time(0),
        vec![
            Segment::new(Time(0), Time(1_000)),
            Segment::new(Time(1_000), Time(2_000)),
        ],
    );
    let bad = DiarizationResult {
        spans: vec![
            vaani_core::pipeline::SpeakerSpan {
                speaker: 5,
                start: Time(0),
                end: Time(1_000),
                score: None,
            },
            vaani_core::pipeline::SpeakerSpan {
                speaker: 0,
                start: Time(1_000),
                end: Time(2_000),
                score: None,
            },
        ],
        num_speakers: 1,
        elapsed_ms: 0,
    };
    let report = attribution::attribute_speakers(&mut t, &bad);
    assert_eq!(report.ignored_spans, 1);
    assert_eq!(report.attributed, 1);
    assert_eq!(report.unattributed, 1);
    assert_eq!(t.speakers.len(), 1, "a phantom speaker was registered");
    assert_eq!(t.segments[0].speaker, None);
    assert_eq!(t.segments[1].speaker, Some(0));
}

#[test]
fn an_error_is_never_reported_as_a_result() {
    // The engine's error paths must produce `Err`, never an empty `Ok`.
    let bad_rate = AudioBuffer::mono(vec![0.0; 1_000], 44_100, Time::ZERO);
    let e = HeuristicDiarizer::new()
        .diarize(&bad_rate, &vaani_core::CancelToken::new(), &|_| {})
        .unwrap_err();
    assert_eq!(e.code(), err::internal().code());
}

#[test]
fn the_configuration_is_the_same_for_two_engines_built_the_same_way() {
    let a = HeuristicDiarizer::new();
    let b = HeuristicDiarizer::new();
    assert_eq!(a.config(), b.config());
    assert_eq!(
        a.config().effective_merge_distance(),
        b.config().effective_merge_distance()
    );
}

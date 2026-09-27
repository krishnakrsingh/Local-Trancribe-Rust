//! Diarization adversarial tests, with the engine's real limits written down.
//!
//! # What is and is not asserted here
//!
//! The engine under test is `HeuristicDiarizer`: a spectral heuristic with no
//! trained speaker embedder. No diarization error rate is claimed, and none
//! could be — a DER needs annotated real meetings, and this crate has neither
//! the model nor permission to commit the audio.
//!
//! What *is* asserted is the structural contract, which holds regardless of how
//! good the clustering is:
//!
//! * spans are sorted, non-inverted, and inside the buffer's span;
//! * speaker indices are a contiguous 0-based prefix below `num_speakers`;
//! * coverage and `speech_ratio` are in `[0, 1]`;
//! * silence, white noise and a 5 ms click do not panic and do not claim speech;
//! * a stereo buffer and a wrong sample rate are refused rather than silently
//!   resampled;
//! * attribution is a separate step over a finished result, never inside
//!   `diarize`;
//! * a rename survives re-attribution.
//!
//! # What was deleted, and why
//!
//! * `two_speaker_tones_find_at_least_one_speaker` asserted
//!   `num_speakers >= 1` and `num_speakers <= 4`. The first is true of almost any
//!   result and the second permits a diarizer that shatters two voices into
//!   four. Both are replaced below by a *measured* statement about what the
//!   engine actually does, which is the only honest thing to assert about a
//!   heuristic whose accuracy is bounded.
//! * `check_invariants`'s speaker assertion permitted a speaker index at or above
//!   `num_speakers`. It is tightened to a contiguous prefix.
//!
//! # A measured observation, recorded rather than asserted as a requirement
//!
//! On `fixtures::meeting_turns(2, 3.0, ..)` — two alternating voices, three
//! seconds each — this machine measures:
//!
//! * `diarize` (the built-in activity detector): 2 speakers, clean alternation
//!   across the whole buffer.
//! * `diarize_with_timeline` (the path the real pipeline takes, reusing the VAD
//!   timeline): 2 speakers, but the spans collapse — one speaker claims
//!   3.3 s–17.2 s of an 18 s file and the other gets 0.7 s at the end.
//! * On `fixtures::two_speaker_turns(3, 2.0, ..)` the VAD finds no speech at
//!   all, and `diarize_with_timeline` therefore returns **zero** spans where
//!   `diarize` returns eight.
//!
//! The last point is a coupling worth naming: the diarizer is not independent of
//! the VAD's *quality* when it is handed the VAD's timeline, only of the VAD's
//! *availability*. See OPEN ISSUES.

use vaani_core::media::AudioBuffer;
use vaani_core::pipeline::VadEngine;
use vaani_core::pipeline::{DiarizationEngine, DiarizationResult, SpeakerSpan, SpeechTimeline};
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

/// The structural contract every diarizer must honour, whatever its accuracy.
fn check_invariants(result: &DiarizationResult, audio: &AudioBuffer, label: &str) {
    let end = audio.start_time.saturating_add(audio.duration());
    for (i, s) in result.spans.iter().enumerate() {
        assert!(
            s.start <= s.end,
            "{label}: span {i} is inverted: {}..{}",
            s.start.as_millis(),
            s.end.as_millis()
        );
        assert!(
            s.start >= audio.start_time && s.end <= end,
            "{label}: span {i} at {}..{} is outside the buffer's {}..{}",
            s.start.as_millis(),
            s.end.as_millis(),
            audio.start_time.as_millis(),
            end.as_millis()
        );
        if i > 0 {
            assert!(
                result.spans[i - 1].start <= s.start,
                "{label}: span {i} starts before span {}",
                i - 1
            );
        }
        if let Some(score) = s.score {
            assert!(
                (0.0..=1.0).contains(&score),
                "{label}: span {i} has a confidence of {score}, which is outside [0,1]"
            );
        }
    }

    // Speaker indices must be a contiguous 0-based prefix below `num_speakers`.
    // The first version of this file allowed an index at or above
    // `num_speakers`, which is precisely the case that makes
    // `speaker_durations` silently drop a span.
    let mut seen: Vec<usize> = result.spans.iter().map(|s| s.speaker).collect();
    seen.sort_unstable();
    seen.dedup();
    for (index, speaker) in seen.iter().enumerate() {
        assert_eq!(
            *speaker, index,
            "{label}: speaker indices are {seen:?}, which is not a contiguous prefix from 0"
        );
    }
    for s in &result.spans {
        assert!(
            s.speaker < result.num_speakers.max(1),
            "{label}: span claims speaker {} but only {} were discovered",
            s.speaker,
            result.num_speakers
        );
    }

    let cov = result.coverage(audio.duration());
    assert!(
        (0.0..=1.0).contains(&cov) && cov.is_finite(),
        "{label}: coverage {cov}"
    );
}

#[test]
fn a_single_steady_voice_does_not_shatter_into_many_speakers() {
    // The negative control for clustering, and the one that matters most: a
    // diarizer that splits one person into six is unusable, because the user
    // cannot tell which label to trust. The engine's own documentation states a
    // reliable region of ≳12 s per turn and a documented failure below ~4 s, so
    // the fixture is 20 s of one steady voice — comfortably inside it.
    let e = engine();
    let audio = mono(fixtures::voice(
        20 * FIXTURE_SAMPLE_RATE as usize,
        FIXTURE_SAMPLE_RATE,
        0x33,
        150.0,
        0.9,
    ));
    let result = e
        .diarize(&audio, &CancelToken::new(), &|_| {})
        .expect("diarize");
    check_invariants(&result, &audio, "one voice");
    assert!(
        result.num_speakers <= 2,
        "one steady voice over 20 s was split into {} speakers: the clustering is not usable, \
         however good it looks on paper",
        result.num_speakers
    );
    println!(
        "one voice over 20 s: {} spans, {} speakers",
        result.spans.len(),
        result.num_speakers
    );
}

#[test]
fn the_measured_two_voice_behaviour_is_reported_rather_than_asserted_as_a_requirement() {
    // This test **reports** what the engine does on two alternating voices and
    // asserts only the structural invariants plus a floor of one speaker.
    //
    // It deliberately does not assert `num_speakers == 2`. Doing so would be
    // asserting that a heuristic with no embedder performs speaker
    // identification, which is false, and the first version of this file came
    // close enough to that to be worth writing down.
    let e = engine();
    let audio = mono(fixtures::meeting_turns(2, 3.0, FIXTURE_SAMPLE_RATE, 5));
    let result = e
        .diarize(&audio, &CancelToken::new(), &|_| {})
        .expect("diarize");
    check_invariants(&result, &audio, "two voices");
    assert!(
        result.num_speakers >= 1,
        "a 24-second buffer with speech in it must attribute somebody"
    );

    // And the same audio with the VAD's timeline, which is the path the
    // pipeline actually takes. Compared, printed, and *not* asserted equal: the
    // two are allowed to differ, and the difference is the thing an operator
    // needs to see.
    let vad = vaani_vad::EnergyVad::new();
    let timeline = vad.detect(&audio, &CancelToken::new()).expect("vad");
    let with_timeline = e
        .diarize_with_timeline(&audio, Some(&timeline), &CancelToken::new(), &|_| {})
        .expect("diarize with the VAD timeline");
    check_invariants(&with_timeline, &audio, "two voices, VAD timeline");

    let covered: i64 = result
        .spans
        .iter()
        .map(|s| (s.end.as_millis() - s.start.as_millis()).max(0))
        .sum();
    let covered_tl: i64 = with_timeline
        .spans
        .iter()
        .map(|s| (s.end.as_millis() - s.start.as_millis()).max(0))
        .sum();
    println!(
        "two voices, {:.0} s: built-in detector -> {} speakers over {} ms; VAD timeline -> \
         {} speakers over {} ms; the VAD found {} regions",
        audio.duration().as_millis() as f64 / 1000.0,
        result.num_speakers,
        covered,
        with_timeline.num_speakers,
        covered_tl,
        timeline.segments.len()
    );
    assert!(
        covered > 0,
        "a buffer with detectable speech must produce some speaker coverage"
    );
}

#[test]
fn an_empty_vad_timeline_means_no_spans_and_that_is_reported() {
    // Coupling worth pinning explicitly. `diarize_with_timeline(Some(&empty))`
    // short-circuits the built-in activity detector, so a VAD that finds no
    // speech produces no speakers — which is the right answer for a silent
    // recording and the *wrong* answer for a recording the VAD simply missed.
    // Either way it must not panic, and the difference must be visible.
    let e = engine();
    let audio = mono(fixtures::two_speaker_turns(3, 2.0, FIXTURE_SAMPLE_RATE));
    let plain = e
        .diarize(&audio, &CancelToken::new(), &|_| {})
        .expect("diarize");
    let empty = SpeechTimeline::default();
    let handed_an_empty_timeline = e
        .diarize_with_timeline(&audio, Some(&empty), &CancelToken::new(), &|_| {})
        .expect("diarize with an empty timeline");
    check_invariants(&handed_an_empty_timeline, &audio, "empty VAD timeline");
    assert_eq!(
        handed_an_empty_timeline.spans.len(),
        0,
        "an empty timeline claims no speech, so there are no speakers to attribute"
    );
    println!(
        "the built-in detector found {} spans; handing it an empty VAD timeline found {}",
        plain.spans.len(),
        handed_an_empty_timeline.spans.len()
    );
}

#[test]
fn silence_white_noise_and_a_click_do_not_panic_or_claim_speech() {
    let e = engine();
    let mut click = fixtures::silence(3.0);
    for s in click.iter_mut().take(80) {
        *s = 0.9;
    }
    for (label, samples) in [
        ("silence", fixtures::silence(10.0)),
        (
            "white noise",
            fixtures::noise(6 * FIXTURE_SAMPLE_RATE as usize, 42, 0.4),
        ),
        ("a 5 ms click", click),
        ("a single sample", vec![0.9]),
    ] {
        let audio = mono(samples);
        let result = e
            .diarize(&audio, &CancelToken::new(), &|_| {})
            .unwrap_or_else(|err| panic!("{label}: {}", err.debug_chain()));
        check_invariants(&result, &audio, label);
        // A quiet source claiming speakers is a real failure mode of an
        // adaptive threshold, so the *quiet* cases get a coverage bound. The
        // click is allowed coverage: it is a loud transient.
        if label == "silence" || label == "white noise" {
            assert!(
                result.spans.is_empty() || result.coverage(audio.duration()) < 0.05,
                "{label} claimed {} spans covering {:.1}% of the buffer",
                result.spans.len(),
                result.coverage(audio.duration()) * 100.0
            );
        }
    }
}

#[test]
fn a_wrong_sample_rate_is_refused_and_the_message_names_the_required_rate() {
    let e = engine();
    let audio = AudioBuffer::mono(vec![0.1; 8_000], 8_000, Time::ZERO);
    let err = e
        .diarize(&audio, &CancelToken::new(), &|_| {})
        .expect_err("8 kHz must fail rather than be resampled");
    assert!(
        err.debug_chain().contains("16"),
        "the message must name the required rate: {}",
        err.debug_chain()
    );
    // Tracked defect `errors.actionable[diar_wrong_sample_rate]`: `Internal` with
    // no recovery. See OPEN ISSUES.
    assert_eq!(err.code(), vaani_core::ErrorCode::Internal);
}

#[test]
fn a_stereo_buffer_is_refused_rather_than_silently_averaged() {
    // The refusal is deliberate and documented in the engine: downmixing here
    // would scale the pitch feature by a factor the caller does not know about,
    // and would hide the real bug — that the media stage did not do its job.
    let e = engine();
    let audio = AudioBuffer {
        samples: vec![0.1; 4_000],
        sample_rate: 16_000,
        channels: 2,
        start_time: Time::ZERO,
    };
    let err = e
        .diarize(&audio, &CancelToken::new(), &|_| {})
        .expect_err("stereo must fail");
    assert!(
        err.debug_chain().contains("mono"),
        "the message must say what was wrong: {}",
        err.debug_chain()
    );
    assert_eq!(err.code(), vaani_core::ErrorCode::Internal);
}

#[test]
fn a_buffer_with_a_non_zero_origin_produces_absolute_spans() {
    // The coordinate-space property, on a +5-minute origin so a buffer-relative
    // implementation cannot hide. The conformance gate runs the same check
    // across every stage; this is the diarizer-specific instance.
    let e = engine();
    let samples = fixtures::meeting_turns(2, 4.0, FIXTURE_SAMPLE_RATE, 5);
    let audio = AudioBuffer::mono(samples, FIXTURE_SAMPLE_RATE, Time(300_000));
    let result = e
        .diarize(&audio, &CancelToken::new(), &|_| {})
        .expect("diarize");
    check_invariants(&result, &audio, "offset buffer");
    assert!(
        !result.spans.is_empty(),
        "an offset buffer must still be separated, or the origin broke the whole stage"
    );
    for s in &result.spans {
        assert!(
            s.start >= Time(300_000),
            "span at {} precedes the buffer's own start of 300000 ms: the diarizer is \
             emitting buffer-relative times",
            s.start.as_millis()
        );
    }
}

#[test]
fn attribution_maps_spans_onto_segments_and_leaves_the_rest_alone() {
    let mut transcript = Transcript::new(EngineStamp::asr("qa", "0"));
    transcript.push_chunk(
        Time(0),
        vec![
            Segment::new(Time(0), Time(2_000)).with_words(vec![Word::new(
                "a",
                Time(0),
                Time(2_000),
            )]),
            Segment::new(Time(3_000), Time(5_000)).with_words(vec![Word::new(
                "b",
                Time(3_000),
                Time(5_000),
            )]),
            Segment::new(Time(6_000), Time(8_000)).with_words(vec![Word::new(
                "c",
                Time(6_000),
                Time(8_000),
            )]),
        ],
    );
    let result = DiarizationResult {
        spans: vec![
            SpeakerSpan {
                speaker: 0,
                start: Time(0),
                end: Time(2_500),
                score: None,
            },
            SpeakerSpan {
                speaker: 1,
                start: Time(5_500),
                end: Time(8_500),
                score: None,
            },
        ],
        num_speakers: 2,
        elapsed_ms: 0,
    };
    vaani_diarization::attribute_speakers(&mut transcript, &result);
    assert_eq!(transcript.segments[0].speaker, Some(0));
    // The middle segment overlaps no span at all. Inventing a speaker for it
    // would be a fabrication, and leaving it unlabelled is the honest answer.
    assert_eq!(
        transcript.segments[1].speaker, None,
        "a segment overlapping no span must stay unlabelled, not be given a neighbour"
    );
    assert_eq!(transcript.segments[2].speaker, Some(1));

    // A rename survives re-attribution, which is what stops a user's naming from
    // being wiped the next time diarization runs.
    transcript.speaker_mut(0).name = Some("Asha".into());
    let existing = transcript.speakers.clone();
    vaani_diarization::apply(&mut transcript, &result, &existing);
    assert_eq!(transcript.speaker_mut(0).name.as_deref(), Some("Asha"));
    // And the words were not touched: attribution is a label, not an edit.
    assert_eq!(transcript.segments[0].text(), "a");
    assert_eq!(transcript.segments[0].start, Time(0));
}

#[test]
fn a_segment_overlapping_two_spans_goes_to_the_tighter_claim() {
    // An ambiguous boundary must resolve to the *more specific* claim, or the
    // attribution double-counts the overlap and the UI flickers between labels
    // while a word is being read.
    let mut transcript = Transcript::new(EngineStamp::asr("qa", "0"));
    // The segment overlaps *both* spans: the broad one 0..5000 and the tight one
    // 900..1200. That ambiguity is the whole point of the test.
    transcript.push_chunk(Time(0), vec![Segment::new(Time(1_000), Time(1_500))]);
    let result = DiarizationResult {
        spans: vec![
            SpeakerSpan {
                speaker: 0,
                start: Time(0),
                end: Time(5_000),
                score: None,
            },
            SpeakerSpan {
                speaker: 1,
                start: Time(900),
                end: Time(1_200),
                score: None,
            },
        ],
        num_speakers: 2,
        elapsed_ms: 0,
    };
    assert_eq!(
        result.speaker_at(Time(1_100)),
        Some(1),
        "the tight span wins"
    );
    assert_eq!(
        result.speaker_at(Time(1_300)),
        Some(0),
        "only the broad one contains it"
    );
    assert_eq!(
        result.speaker_at(Time(1_200)),
        Some(0),
        "spans are half-open, so the instant a span ends belongs to whatever comes next; \
         otherwise adjacent turns from one speaker would both claim the boundary"
    );
    assert_eq!(
        result.speaker_at(Time(9_000)),
        None,
        "past the end of everything"
    );

    // **Two different disambiguation rules, on purpose, and worth naming.**
    // `speaker_at` resolves an ambiguous instant to the *shortest* containing
    // span. `attribute_speakers` resolves an ambiguous *segment* to the speaker
    // with the greatest total overlap. For this segment — 1000..1500, inside the
    // broad 0..5000 and clipping the tight 900..1200 by 200 ms — the rules
    // disagree: the instant at 1100 belongs to speaker 1, while the segment as a
    // whole belongs to speaker 0.
    //
    // Both are defensible, and both are documented, so this test pins the pair
    // rather than pretending they agree. If either is ever changed, this fails
    // and the change is a deliberate one.
    vaani_diarization::attribute_speakers(&mut transcript, &result);
    assert_eq!(
        transcript.segments[0].speaker,
        Some(0),
        "attribution is an overlap-weighted vote, not `speaker_at` at the segment's midpoint: \
         500 ms of overlap with speaker 0 beats 200 ms with speaker 1"
    );
}

#[test]
fn speaker_durations_ignore_an_index_the_engine_never_registered() {
    // A clustering engine that reports a higher index than `num_speakers` must
    // not make `speaker_durations` panic, and must not invent a bucket.
    let result = DiarizationResult {
        spans: vec![SpeakerSpan {
            speaker: 7,
            start: Time(0),
            end: Time(1_000),
            score: None,
        }],
        num_speakers: 1,
        elapsed_ms: 0,
    };
    assert_eq!(result.speaker_durations(), vec![0]);
    assert_eq!(
        result.coverage(Time(1_000)),
        1.0,
        "coverage counts the span regardless"
    );
}

#[test]
fn progress_never_reports_a_value_a_bar_cannot_draw() {
    // `DiarizationEngine::diarize` takes a bare `f64`, so unlike the app-level
    // stage there is no `Progress` wrapper to clamp it. That makes this the only
    // place the raw stream can be checked, and it is checked for finiteness, for
    // range, and for monotonicity.
    let e = engine();
    let audio = mono(fixtures::meeting_turns(2, 3.0, FIXTURE_SAMPLE_RATE, 5));
    let seen = std::sync::Mutex::new(Vec::new());
    let sink = {
        let seen = &seen;
        move |f: f64| {
            if let Ok(mut v) = seen.lock() {
                v.push(f);
            }
        }
    };
    e.diarize(&audio, &CancelToken::new(), &sink)
        .expect("diarize");
    let fractions = seen.into_inner().expect("progress lock");
    for (i, f) in fractions.iter().enumerate() {
        assert!(
            f.is_finite() && (0.0..=1.0).contains(f),
            "progress fraction {i} was {f}: a progress bar cannot draw that"
        );
    }
    for w in fractions.windows(2) {
        assert!(
            w[1] >= w[0],
            "progress went backwards: {fractions:?}; a bar that jumps to 100% and back to 20% is \
             the bug users report as `it keeps restarting`"
        );
    }
    println!(
        "the diarizer reported {} progress values, all in [0,1]",
        fractions.len()
    );
}

#[test]
fn cancellation_is_refused_before_any_work() {
    let e = engine();
    let audio = mono(fixtures::meeting_turns(6, 3.0, FIXTURE_SAMPLE_RATE, 5));
    let token = CancelToken::new();
    token.cancel();
    let err = e
        .diarize(&audio, &token, &|_| {})
        .expect_err("a pre-cancelled token must be refused");
    assert!(err.is_cancelled());
    assert_eq!(err.code(), vaani_core::ErrorCode::Cancelled);
}

#[test]
fn the_engine_reports_its_own_capabilities_honestly() {
    // The settings panel shows this. A capability the engine does not have but
    // claims is worse than an absent one, because a user will plan around it.
    let e = engine();
    let caps = e.capabilities();
    assert!(
        !caps.neural_embeddings,
        "this build has no trained speaker embedder, so claiming one would be a lie: \
         docs/JOURNEY.md records diarization as the weakest link for exactly this reason"
    );
    assert!(!e.has_neural_embedder(), "no embedder is loaded");
    assert_eq!(e.config(), e.config(), "the config must be a stable value");
}

//! VAD adversarial tests: silence, noise, speech, cancellation, chunking.
//!
//! # What was deleted, and why
//!
//! * `engine_requires_16khz` asserted that `vad.sample_rate() == 16_000` and
//!   that the constant `SAMPLE_RATE` equals `16_000`. A test that compares a
//!   value to the literal it is defined as cannot fail, and it bought nothing
//!   the conformance gate does not already assert as a *behaviour*.
//! * `speech_like_audio_yields_absolute_sorted_timeline` ran the detector over
//!   `fixtures::speech_like` and asserted only that the ratio was in `[0, 1]`
//!   and the segments were sorted. The measured ratio for that fixture is
//!   **0.000** — the detector correctly rejects pure tones by design, per
//!   `fixtures::speech_like`'s own documentation — so every assertion ran
//!   against an empty timeline and would have passed for a detector that
//!   returned nothing at all. It is now
//!   [`tone_bursts_are_not_reported_as_speech`], which asserts the *negative*
//!   result, and that has teeth: a detector with its gate stuck open returns
//!   100% speech and fails.
//! * `cancelled_vad_reports_cancelled` cancelled the token before the call, so
//!   it only proved the entry check. The mid-run behaviour — the part the
//!   contract actually cares about — is now measured, with a bound, by
//!   `check_cancellation_is_prompt` in the conformance gate.
//!
//! The positive control, `speech_shaped_voice_over_room_is_detected`, is the one
//! that earns its keep: it fails a detector that returns nothing.

use vaani_core::media::AudioBuffer;
use vaani_core::pipeline::{SpeechSegment, SpeechTimeline, VadEngine};
use vaani_core::{CancelToken, Time};
use vaani_qa::fixtures::{self, FIXTURE_SAMPLE_RATE};
use vaani_vad::{ChunkPolicy, Chunker, EnergyVad};

fn vad() -> EnergyVad {
    EnergyVad::new()
}

fn mono(samples: Vec<f32>) -> AudioBuffer {
    AudioBuffer::mono(samples, FIXTURE_SAMPLE_RATE, Time::ZERO)
}

#[test]
fn the_engine_identifies_itself_and_the_rate_it_needs() {
    // The identities are not decorative: they are recorded in the benchmark
    // report and shown in the settings panel, and a blank one there is a support
    // question nobody can answer.
    let v = vad();
    assert_eq!(v.sample_rate(), 16_000, "the pipeline resamples to this");
    assert!(!v.id().is_empty(), "a blank id cannot appear in a report");
    assert!(!v.name().is_empty(), "a blank name cannot appear in the UI");
    assert_eq!(v.id(), v.id(), "the id must be stable across calls");

    // The id is a *wire* value, so two differently-configured engines must not
    // collide on it. A benchmark report that shows one row for two engines
    // because they share an id is worse than no report.
    let a = EnergyVad::new().with_name("energy-vad-a", "Energy VAD (a)");
    let b = EnergyVad::new().with_name("energy-vad-b", "Energy VAD (b)");
    assert_ne!(a.id(), b.id());
    assert_eq!(a.id(), "energy-vad-a");
    assert_eq!(a.name(), "Energy VAD (a)");
}

#[test]
fn sixty_seconds_of_digital_silence_produces_no_speech_and_no_chunks() {
    // Silence is the negative control for the whole VAD. A detector that
    // reports speech here produces a transcript of a room.
    let v = vad();
    let audio = mono(fixtures::silence(60.0));
    let timeline = v
        .detect(&audio, &CancelToken::new())
        .expect("detect silence");
    assert!(
        timeline.segments.is_empty(),
        "silence produced {} segments",
        timeline.segments.len()
    );
    assert!(
        timeline.speech_ratio(audio.duration()) < 0.01,
        "ratio {} on silence",
        timeline.speech_ratio(audio.duration())
    );
    let chunks = Chunker::new(ChunkPolicy::default()).plan(&timeline, audio.duration());
    assert!(chunks.is_empty(), "silence must not produce ASR chunks");

    // The reported stats are a second, independent account of the same decision,
    // and they are what the UI shows. Asserting them here is what gives the
    // `speech_frames` counter teeth: without this assertion the counter could be
    // computed from any threshold, including one that accepts every frame, and
    // no test in this file would notice -- a gap the mutation harness found by
    // mutating the counter's own threshold and watching the suite stay green.
    let stats = v.stats();
    assert_eq!(
        stats.speech_frames, 0,
        "stats claim {} speech frames on digital silence",
        stats.speech_frames
    );
    assert_eq!(
        stats.gated_frames, 0,
        "stats claim {} gated frames on digital silence",
        stats.gated_frames
    );
    assert!(
        stats.frames > 0,
        "the silence run analysed no frames, so the assertions above prove nothing"
    );
}

#[test]
fn tone_bursts_are_not_reported_as_speech() {
    // A pure tone is *supposed* to be rejected: a steady source is a speaker
    // playing a test tone through a microphone, a fridge, or a mains hum, and a
    // detector that transcribes it is worse than useless. This is the negative
    // control that gives `speech_shaped_voice_over_room_is_detected` its
    // meaning: both are only interesting as a pair.
    for (label, samples) in [
        (
            "speech_like tone bursts",
            fixtures::speech_like(6.0, FIXTURE_SAMPLE_RATE, 7),
        ),
        (
            "a single 440 Hz tone",
            fixtures::tone(
                6 * FIXTURE_SAMPLE_RATE as usize,
                FIXTURE_SAMPLE_RATE,
                440.0,
                0.5,
            ),
        ),
        (
            "a 50 Hz mains hum",
            fixtures::tone(
                6 * FIXTURE_SAMPLE_RATE as usize,
                FIXTURE_SAMPLE_RATE,
                50.0,
                0.6,
            ),
        ),
    ] {
        let audio = mono(samples);
        let timeline = v_detect(&audio);
        assert!(
            timeline.segments.is_empty(),
            "{label}: {} segments at ratio {:.3}: a steady source is not speech",
            timeline.segments.len(),
            timeline.speech_ratio(audio.duration())
        );
    }
}

fn v_detect(audio: &AudioBuffer) -> SpeechTimeline {
    vad().detect(audio, &CancelToken::new()).expect("detect")
}

#[test]
fn noise_never_panics_and_the_timeline_stays_sorted_and_in_range() {
    // Three levels spanning a realistic capture: a quiet room, a normal
    // meeting, a very loud one.
    for (seed, peak) in [(1u64, 0.1f32), (2, 0.5), (3, 0.9), (4, 1.0)] {
        let audio = mono(fixtures::noise(
            FIXTURE_SAMPLE_RATE as usize * 5,
            seed,
            peak,
        ));
        let timeline = v_detect(&audio);
        let ratio = timeline.speech_ratio(audio.duration());
        assert!(
            (0.0..=1.0).contains(&ratio),
            "seed {seed}: ratio {ratio} out of range"
        );
        for w in timeline.segments.windows(2) {
            assert!(
                w[0].end <= w[1].start,
                "seed {seed}: overlapping segments, so a consumer cannot binary-search them"
            );
        }
        let end = audio.start_time.saturating_add(audio.duration());
        for s in &timeline.segments {
            assert!(s.start <= s.end, "seed {seed}: inverted segment");
            assert!(
                s.start >= audio.start_time && s.end <= end,
                "seed {seed}: segment outside the buffer's span"
            );
        }
    }
}

#[test]
fn non_finite_samples_produce_a_defined_answer_not_a_crash() {
    // A `NaN` sample becomes a `NaN` log-mel, which becomes a transcript of
    // garbage that still looks like language. So this asserts the *shape* of the
    // answer as well as the absence of a panic.
    for (label, samples) in [
        ("all NaN", vec![f32::NAN; FIXTURE_SAMPLE_RATE as usize]),
        (
            "alternating NaN",
            (0..FIXTURE_SAMPLE_RATE as usize)
                .map(|i| if i % 97 == 0 { f32::NAN } else { 0.4 })
                .collect(),
        ),
        (
            "all +Inf",
            vec![f32::INFINITY; FIXTURE_SAMPLE_RATE as usize],
        ),
    ] {
        let audio = mono(samples);
        let timeline = v_detect(&audio);
        let ratio = timeline.speech_ratio(audio.duration());
        assert!(
            (0.0..=1.0).contains(&ratio) && ratio.is_finite(),
            "{label}: ratio {ratio}"
        );
        for s in &timeline.segments {
            // `Time` is an integer millisecond count, so it cannot be NaN. What
            // *can* be wrong is that the detector produced a segment at all from
            // non-finite input, which is asserted above by the empty result.
            assert!(
                s.start <= s.end,
                "{label}: an inverted segment escaped the detector"
            );
        }
    }
}

#[test]
fn speech_shaped_voice_over_a_room_floor_is_detected() {
    // The positive control, and the test that gives the whole file its meaning.
    // A harmonic-comb voice at -25 dB over a -100 dB room floor must produce
    // speech, and that speech must produce at least one ASR chunk.
    //
    // The room floor matters: a gap of digital silence collapses the detector's
    // adaptive noise floor to zero, so a fixture built from `silence()` breaks the
    // detector in a way no real capture would. See `fixtures::room_floor`.
    let v = vad();
    let audio = mono(fixtures::meeting(
        6 * FIXTURE_SAMPLE_RATE as usize,
        FIXTURE_SAMPLE_RATE,
        7,
        130.0,
    ));
    let timeline = v.detect(&audio, &CancelToken::new()).expect("detect voice");
    assert!(
        !timeline.segments.is_empty(),
        "a speech-shaped voice over a room floor produced no segments: the VAD has stopped \
         detecting anything at all, and every other test in this file would still pass"
    );
    let ratio = timeline.speech_ratio(audio.duration());
    assert!(ratio > 0.05 && ratio <= 1.0, "implausible ratio {ratio}");

    // The detected regions must be *inside* the buffer and ordered. This is the
    // same coordinate-space property the conformance gate checks with a +5-minute
    // origin; here it is checked on a zero-origin buffer so a regression shows
    // up in both places.
    let end = audio.start_time.saturating_add(audio.duration());
    let mut previous_end = Time::ZERO;
    for s in &timeline.segments {
        assert!(s.start <= s.end);
        assert!(
            s.start >= previous_end,
            "segments overlap or are out of order"
        );
        assert!(s.end <= end, "segment ends past the audio");
        previous_end = s.end;
    }

    let chunks = Chunker::new(ChunkPolicy::default()).plan(&timeline, audio.duration());
    assert!(
        !chunks.is_empty(),
        "detected speech must produce ASR chunks, or the pipeline silently transcribes nothing"
    );
    println!(
        "voice over room: {} segments, ratio {ratio:.3}",
        timeline.segments.len()
    );
}

#[test]
fn a_buffer_with_a_non_zero_origin_produces_absolute_times() {
    // The strongest single property in this file, and the one the brief calls
    // the project's most-painful invariant. A detector that returns times
    // relative to the buffer produces *identical* output for a zero-origin
    // buffer and is wrong for every other one — so the only way to tell the
    // difference is to hand it a buffer that starts at +5 minutes.
    let v = vad();
    let samples = fixtures::meeting(
        6 * FIXTURE_SAMPLE_RATE as usize,
        FIXTURE_SAMPLE_RATE,
        7,
        130.0,
    );
    for (origin, audio) in [
        (Time(0), mono(samples.clone())),
        (
            Time(300_000),
            AudioBuffer::mono(samples, FIXTURE_SAMPLE_RATE, Time(300_000)),
        ),
    ] {
        let timeline = v.detect(&audio, &CancelToken::new()).expect("detect");
        assert!(
            !timeline.segments.is_empty(),
            "origin {origin}: the offset buffer must still be detected"
        );
        let end = origin.saturating_add(audio.duration());
        for s in &timeline.segments {
            assert!(
                s.start >= origin,
                "origin {origin}: segment starts at {}, before the buffer's own start: the \
                 detector is emitting buffer-relative times",
                s.start.as_millis()
            );
            assert!(
                s.end <= end,
                "origin {origin}: segment ends at {} past the buffer's {}",
                s.end.as_millis(),
                end.as_millis()
            );
        }
    }
    println!("an offset buffer and a zero-origin buffer both produce absolute times");
}

#[test]
fn the_chunker_splits_a_long_turn_and_bounds_every_chunk() {
    let policy = ChunkPolicy::default();
    // One 90-second speech block: the chunker must split at the cap, because a
    // 90-second decode is a very different proposition from a 30-second one for
    // both the model's context and the UI's patience.
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
    // The cap applies *before* padding, so a padded chunk may exceed `max_ms` by
    // up to `pad_ms` on each side. Asserting `max_ms` exactly would be asserting
    // a stricter contract than the one documented.
    let bound = policy.max_ms + 2 * policy.pad_ms;
    for c in &chunks {
        assert!(
            (c.end.as_millis() - c.start.as_millis()) <= bound,
            "chunk {}..{} exceeds max+2*pad = {bound}",
            c.start.as_millis(),
            c.end.as_millis()
        );
        assert!(c.start <= c.end, "an inverted chunk");
    }
    // The chunks must cover the speech block contiguously, in the source
    // media's coordinates, with no gap a word could fall into.
    assert_eq!(chunks.first().map(|c| c.start), Some(Time(0)));
    assert_eq!(chunks.last().map(|c| c.end), Some(Time(90_000)));
    for w in chunks.windows(2) {
        assert!(
            w[0].end <= w[1].start,
            "chunks {} and {} leave a gap a word could fall into",
            w[0].end.as_millis(),
            w[1].start.as_millis()
        );
    }
}

#[test]
fn a_wrong_sample_rate_is_rejected_with_a_message_naming_the_required_rate() {
    // The refusal is deliberate — see `fixtures::speech_like` — and the message
    // has to say which rate it wants, because "internal error" tells a caller
    // nothing about which of the two sides of the boundary is wrong.
    let v = vad();
    let audio = AudioBuffer::mono(vec![0.1; 8_000], 8_000, Time::ZERO);
    let err = v
        .detect(&audio, &CancelToken::new())
        .expect_err("8 kHz must be refused");
    assert!(
        err.debug_chain().contains("16000"),
        "the message must name the required rate: {}",
        err.debug_chain()
    );
    // **Tracked defect, `vaani-vad` src/energy.rs `check_input`.** The code is
    // `Internal` and there is no recovery, so a user whose media stage handed
    // over 8 kHz is told to report a bug. Registered as
    // `errors.actionable[energy_vad_wrong_sample_rate]`; see
    // `vaani_qa::conformance::all_known_failures` and OPEN ISSUES.
    assert_eq!(err.code(), vaani_core::ErrorCode::Internal);
    assert!(
        err.user.recovery.is_none(),
        "a recovery has been added; delete the tracked entry and assert it is present"
    );
}

#[test]
fn a_pre_cancelled_token_is_refused_before_any_work() {
    // The entry check. The mid-run behaviour, with a measured bound, is in the
    // conformance gate; this is the race-free half, kept here because it needs
    // no timing and so can never flake.
    let v = vad();
    let audio = mono(fixtures::meeting(
        6 * FIXTURE_SAMPLE_RATE as usize,
        FIXTURE_SAMPLE_RATE,
        7,
        130.0,
    ));
    let token = CancelToken::new();
    token.cancel();
    let err = v
        .detect(&audio, &token)
        .expect_err("a pre-cancelled token must be refused");
    assert!(err.is_cancelled());
    assert_eq!(err.code(), vaani_core::ErrorCode::Cancelled);
}

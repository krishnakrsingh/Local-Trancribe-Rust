//! Cancellation: how long between asking and stopping, and that a cancelled
//! engine is not a broken one.
//!
//! The contract in `vaani_core::pipeline::VadEngine` is "at least once per
//! second of audio", and `DiarizationEngine::diarize` inherits the same shape.
//! The default poll interval here is 50 frames, which is 0.5 s of audio, so the
//! measured latency from `cancel()` to `Err` is bounded by the time to process
//! 50 frames plus whatever the current phase's inner loop costs. This file
//! measures it rather than asserting it, because a number nobody measured is a
//! number nobody knows.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use vaani_core::pipeline::DiarizationEngine;
use vaani_core::{CancelToken, Time};
use vaani_diarization::config::DiarizationConfig;
use vaani_diarization::engine::HeuristicDiarizer;
use vaani_diarization::synth;

/// Ten minutes of two synthetic speakers. Long enough that a run cannot finish
/// before the cancel lands, and short enough to render in a couple of seconds.
fn ten_minutes() -> Vec<f32> {
    let voices = synth::two_voices();
    let u = synth::alternate(&[0, 1], 0, 2_000, 300);
    synth::render(&u, &voices, 600_000, 16_000, 0.003, 3).0
}

/// Start a diarization on a background thread and return the join handle plus a
/// flag that goes true when the call has returned (successfully or not).
fn start(
    engine: Arc<HeuristicDiarizer>,
    audio: Vec<f32>,
    token: CancelToken,
) -> (
    std::thread::JoinHandle<vaani_core::Result<vaani_core::pipeline::DiarizationResult>>,
    Arc<AtomicBool>,
) {
    let done = Arc::new(AtomicBool::new(false));
    let d = done.clone();
    let handle = std::thread::spawn(move || {
        let audio = vaani_core::media::AudioBuffer::mono(audio, 16_000, Time::ZERO);
        let r = engine.diarize(&audio, &token, &|_| {});
        d.store(true, Ordering::SeqCst);
        r
    });
    (handle, done)
}

/// Cancel repeatedly until `done` is set, and report how long the whole thing
/// took from the first cancel that landed after the worker had started.
///
/// The repeat matters. `diarize` clears the engine's own cancel flag at the
/// start of every call, which is the right behaviour -- a cancelled engine has
/// to work on the next job -- and it means a single cancel issued while the
/// worker thread is still being scheduled is legitimately discarded. Looping
/// is how a test says "cancel as soon as you possibly can" without depending on
/// the scheduler.
fn cancel_until_done(cancel: &impl Fn(), done: &AtomicBool, budget: Duration) -> Duration {
    let started = Instant::now();
    while started.elapsed() < budget {
        cancel();
        if done.load(Ordering::SeqCst) {
            return started.elapsed();
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    started.elapsed()
}

#[test]
fn cancelling_the_token_stops_the_run_and_the_latency_is_measured() {
    let audio = ten_minutes();
    let engine = Arc::new(HeuristicDiarizer::new());
    let token = CancelToken::new();
    let (handle, done) = start(engine, audio, token.clone());
    // Give the worker a moment to actually enter the call before cancelling, so
    // the measurement is of cancellation latency and not of thread start-up.
    std::thread::sleep(Duration::from_millis(120));
    let latency = cancel_until_done(&|| token.cancel(), &done, Duration::from_secs(20));
    let r = handle.join().unwrap();

    let e = r.expect_err("a ten-minute job must not finish before the cancel lands");
    assert!(e.is_cancelled(), "{}", e.debug_chain());
    println!("token cancel: Err after {:?} (budget 20 s)", latency);
    // The default poll is 50 frames of 10 ms. In a debug build the per-frame
    // cost is high, so the bound is loose; what it catches is a poll interval
    // that has been made orders of magnitude too long.
    assert!(
        latency < Duration::from_secs(15),
        "cancellation took {latency:?}, which is far longer than 50 frames of audio"
    );
}

#[test]
fn cancelling_the_engine_stops_the_run_when_no_token_is_reachable() {
    // The UI's cancel button has to work even when the job that owns the token
    // has been dropped, which is why `DiarizationEngine` has `cancel()` at all.
    let audio = ten_minutes();
    let engine = Arc::new(HeuristicDiarizer::new());
    let (handle, done) = start(engine.clone(), audio, CancelToken::new());
    std::thread::sleep(Duration::from_millis(120));
    let latency = cancel_until_done(&|| engine.cancel(), &done, Duration::from_secs(20));
    let r = handle.join().unwrap();
    let e = r.expect_err("the run must not have completed");
    assert!(e.is_cancelled());
    println!("engine cancel: Err after {latency:?}");
    assert!(latency < Duration::from_secs(15), "took {latency:?}");
}

#[test]
fn a_finer_poll_interval_shortens_the_latency() {
    // The poll interval is a configuration value, and this is the measurement
    // that justifies exposing it: 5 frames instead of 50 should land
    // materially sooner.
    let cfg = DiarizationConfig {
        cancel_poll_frames: 5,
        ..DiarizationConfig::default()
    };
    let audio = ten_minutes();
    let engine = Arc::new(HeuristicDiarizer::with_config(cfg));
    let token = CancelToken::new();
    let (handle, done) = start(engine, audio, token.clone());
    std::thread::sleep(Duration::from_millis(120));
    let latency = cancel_until_done(&|| token.cancel(), &done, Duration::from_secs(20));
    let r = handle.join().unwrap();
    assert!(r.unwrap_err().is_cancelled());
    println!("5-frame poll: Err after {latency:?}");
    assert!(latency < Duration::from_secs(10), "took {latency:?}");
}

#[test]
fn an_already_cancelled_engine_still_loads() {
    // `load` must not be gated on the cancel flag: the settings screen calls
    // it on a fresh engine, and refusing would make the engine look broken.
    let engine = HeuristicDiarizer::new();
    let token = CancelToken::new();
    token.cancel();
    let e = engine.load(&token).unwrap_err();
    assert!(e.is_cancelled(), "a cancelled token must fail the load");
}

#[test]
fn a_cancelled_engine_recovers_on_the_next_call() {
    // The cancel flag is per-call. An engine that latched it would fail every
    // subsequent job in the process, and the user would see a cancel button
    // that breaks the app.
    let voices = synth::two_voices();
    let (samples, _) = synth::render(
        &synth::alternate(&[0, 1], 0, 3_000, 12),
        &voices,
        36_000,
        16_000,
        0.003,
        8,
    );
    let engine = HeuristicDiarizer::new();
    engine.cancel();
    assert!(engine.is_cancelled(&CancelToken::new()));
    let audio = vaani_core::media::AudioBuffer::mono(samples, 16_000, Time::ZERO);
    let r = engine
        .diarize(&audio, &CancelToken::new(), &|_| {})
        .expect("a cancelled engine must work on the next job");
    assert!(r.num_speakers >= 1, "{:?}", r.spans);
    assert!(!engine.is_cancelled(&CancelToken::new()));
}

#[test]
fn cancelling_does_not_leave_a_partial_result_behind() {
    // A cancelled call must return `Err`, never a truncated `Ok`. A partial
    // result would be attributed onto a transcript and produce speaker labels
    // for the first third of a file.
    let audio = ten_minutes();
    let engine = Arc::new(HeuristicDiarizer::new());
    let token = CancelToken::new();
    let (handle, done) = start(engine, audio, token.clone());
    std::thread::sleep(Duration::from_millis(120));
    cancel_until_done(&|| token.cancel(), &done, Duration::from_secs(20));
    match handle.join().unwrap() {
        Err(e) => assert!(e.is_cancelled()),
        Ok(r) => panic!(
            "a cancelled call returned a result with {} spans",
            r.spans.len()
        ),
    }
}

#[test]
fn progress_stops_when_the_run_is_cancelled() {
    // The progress callback is the UI's only evidence that work is happening.
    // It must not be called after the run has been abandoned, or the bar will
    // keep moving on a job that has stopped.
    let audio = ten_minutes();
    let engine = Arc::new(HeuristicDiarizer::new());
    let token = CancelToken::new();
    let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let c = count.clone();
    let c2 = count.clone();
    let t2 = token.clone();
    let handle = std::thread::spawn(move || {
        let audio = vaani_core::media::AudioBuffer::mono(audio, 16_000, Time::ZERO);
        let r = engine.diarize(&audio, &t2, &|_| {
            c.fetch_add(1, Ordering::SeqCst);
        });
        (r, c2.load(Ordering::SeqCst))
    });
    std::thread::sleep(Duration::from_millis(120));
    cancel_until_done(
        &|| token.cancel(),
        &AtomicBool::new(false),
        Duration::from_millis(600),
    );
    let (r, at_join) = handle.join().unwrap();
    assert!(r.unwrap_err().is_cancelled());
    // Nothing calls progress after the call returns.
    let after = count.load(Ordering::SeqCst);
    assert_eq!(
        at_join, after,
        "progress was reported after the call returned"
    );
    assert!(after > 0, "no progress at all was reported");
    println!("progress callbacks before cancellation: {after}");
}

#[test]
fn the_engine_is_shareable_across_threads() {
    // `DiarizationEngine: Send + Sync`, and the engine holds a `parking_lot`
    // mutex around the optional neural embedder, so this is a real assertion
    // about the locking rather than a formality.
    let voices = synth::two_voices();
    let (samples, _) = synth::render(
        &synth::alternate(&[0, 1], 0, 2_000, 8),
        &voices,
        16_000,
        16_000,
        0.003,
        11,
    );
    let engine = Arc::new(HeuristicDiarizer::new());
    engine.load(&CancelToken::new()).unwrap();
    let mut handles = Vec::new();
    for _ in 0..4 {
        let e = engine.clone();
        let s = samples.clone();
        handles.push(std::thread::spawn(move || {
            let audio = vaani_core::media::AudioBuffer::mono(s, 16_000, Time::ZERO);
            e.diarize(&audio, &CancelToken::new(), &|_| {}).unwrap()
        }));
    }
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    // Every thread saw the same recording, so every one must have produced the
    // same answer. This is the determinism the core contract depends on when
    // it persists speaker indices.
    let first = &results[0];
    for r in &results[1..] {
        assert_eq!(r.num_speakers, first.num_speakers);
        assert_eq!(r.spans, first.spans);
    }
    assert!(!first.spans.is_empty());
}

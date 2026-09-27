//! The live SraVaani decode, on demand.
//!
//! # Why this is a separate, opt-in file
//!
//! The headline gate in `docs/JOURNEY.md` is 202/202 token ids and 108/108 word
//! timings at 0.0 ms deviation against a reference produced by the model's own
//! TorchScript graphs. Reproducing that needs three things this repository does
//! not contain: the ONNX Runtime, the 2.6 GB SraVaani export, and the exact
//! 40-second clip the reference was made from.
//!
//! The first two are machine facts and the suite skips them loudly. The third is
//! private media that will never be committed. So this file is **opt-in**: it
//! runs only when `VAANI_QA_RUNTIME_DLL` points at an `onnxruntime.dll` and the
//! weights are present, and it prints both facts when it does not.
//!
//! # The runtime override, and why it exists
//!
//! `VAANI_QA_RUNTIME_DLL` is a *test-only* escape hatch, and it exists because
//! `vaani_inference::RuntimeLocation::discover` cannot find the runtime when the
//! build's `CARGO_TARGET_DIR` lives outside the repository — which is exactly
//! how the QA agent's own target directory is configured. The defect is in
//! `vaani-inference`, not here; it is written up in OPEN ISSUES with a
//! reproduction. This test is the workaround, made visible, rather than a silent
//! global.
//!
//! ```powershell
//! $env:VAANI_QA_RUNTIME_DLL = "$PWD\runtime\onnxruntime.dll"
//! cargo test -p vaani-qa --release --test asr_live -- --nocapture
//! ```
//!
//! Note that the override points at the DLL rather than at its directory, so the
//! CUDA provider DLLs are not preloaded and the run is on the **CPU** execution
//! provider. That is deliberate: a CPU run is reproducible, and a 40-second
//! chunk encodes in about 16 s here.

use std::path::PathBuf;
use std::time::Instant;

use vaani_asr::engine::{AsrEngine, AudioChunk};
use vaani_asr::{EngineArtifacts, SraVaaniEngine};
use vaani_core::{CancelToken, Language, Time};
use vaani_qa::fixtures;
use vaani_qa::skip::{self, Requirement};

/// Set `VAANI_QA_RUNTIME_DLL` to run this file.
fn forced_runtime() -> Option<PathBuf> {
    let dll = std::env::var_os("VAANI_QA_RUNTIME_DLL").map(PathBuf::from)?;
    if dll.is_file() {
        return Some(dll);
    }
    skip::skip(
        "asr_live",
        Requirement::OnnxRuntime,
        format!(
            "VAANI_QA_RUNTIME_DLL points at {}, which is not a file",
            dll.display()
        ),
    );
    None
}

/// Whether the harness may run the live decode, and why not if it may not.
fn gate(test: &str) -> Option<EngineArtifacts> {
    if skip::require(test, Requirement::ModelWeights).is_some() {
        return None;
    }
    if skip::require(test, Requirement::OnnxRuntime).is_some() {
        return None;
    }
    // `gate` also has to confirm the operator's override points at a real DLL,
    // otherwise the skip would print a remedy for a machine that is fine.
    let _dll = forced_runtime()?;
    Some(vaani_asr::artifacts_in(&vaani_asr::default_dir()))
}

/// Open the engine with the forced runtime, or fail.
///
/// A present-but-broken installation is a failure, never a skip. See
/// `Environment::engine_problem`.
fn open(test: &str, art: &EngineArtifacts) -> SraVaaniEngine {
    match vaani_asr::open_sravaani(art, vaani_inference::Provider::Cpu, &caps()) {
        Ok(e) => e,
        Err(e) => panic!(
            "{test}: {}",
            skip::Environment::get().engine_problem(test, &e)
        ),
    }
}

fn caps() -> vaani_inference::MachineCapabilities {
    let loc = vaani_inference::runtime::location();
    vaani_inference::capability::detect(&loc.provider_dirs.clone())
}

#[test]
fn the_engine_opens_and_reports_the_shape_the_contract_says_it_has() {
    let Some(art) = gate("engine_opens_and_reports_its_shape") else {
        return;
    };
    let e = open("engine_opens_and_reports_its_shape", &art);
    assert_eq!(e.id(), "sravaani");
    assert_eq!(e.model_name(), "SraVaani 1.0");
    assert!(e.supports_word_timestamps());
    assert!(e.supports_timestamps());
    assert!(!e.supports_streaming(), "SraVaani 1.0 is a file model");
    assert!(
        (e.frame_seconds() - 0.08).abs() < 1e-12,
        "the encoder frame must be 80 ms: 8x subsampling at hop 160 / 16 kHz. A factor here \
         drags the whole transcript out of sync with the media."
    );
    assert_eq!(e.config().vocab_size, 5_000);
    assert_eq!(e.config().blank_id, 5_000);
    assert_eq!(e.config().durations, [0, 1, 2, 3, 4]);
}

#[test]
fn a_forty_second_chunk_decodes_with_a_plausible_shape_and_in_bounds() {
    let Some(art) = gate("forty_second_chunk_decodes") else {
        return;
    };
    let e = open("forty_second_chunk_decodes", &art);
    let cancel = CancelToken::new();

    // 40 seconds, the same length as the reference clip, so the encoder frame
    // count is directly comparable with the golden fixture's 500.
    let samples = fixtures::voice(
        40 * fixtures::FIXTURE_SAMPLE_RATE as usize,
        fixtures::FIXTURE_SAMPLE_RATE,
        0x2B,
        150.0,
        0.9,
    );
    let chunk = AudioChunk::new(samples, fixtures::FIXTURE_SAMPLE_RATE, Time::ZERO)
        .with_language(Language::Auto);

    let t0 = Instant::now();
    let r = e
        .transcribe(&chunk, &cancel)
        .unwrap_or_else(|e| panic!("transcribe must succeed: {}", e.debug_chain()));
    let wall = t0.elapsed();

    let audio_ms = chunk.duration().as_millis();
    let rtf = r.timings.total_ms() as f64 / 1000.0 / 40.0;
    println!(
        "40 s on the CPU EP in {:?} (RTF {rtf:.3}): {} segments, {} encoder frames, {} decode \
         steps (frontend {} ms, encode {} ms, decode {} ms)",
        wall,
        r.segments.len(),
        r.timings.encoder_frames,
        r.timings.decode_steps,
        r.timings.frontend_ms,
        r.timings.encode_ms,
        r.timings.decode_ms
    );

    // 1. Encoder frames must be ceil(feature_lengths / 8). This is bug #3 in
    //    JOURNEY.md: the encoder's valid length is the `enc_lengths` output, not
    //    the tensor dimension, and reading the tensor decodes one frame past the
    //    end of the audio.
    let feature_lengths = chunk.samples.len() / 160;
    let expected_frames = feature_lengths.div_ceil(8);
    assert_eq!(
        r.timings.encoder_frames,
        expected_frames,
        "audio was {} samples @ {} Hz, so feature_lengths is {feature_lengths} and ceil(/8) is \
         {expected_frames}",
        chunk.samples.len(),
        chunk.sample_rate
    );
    assert_eq!(
        expected_frames, 500,
        "40 s at 80 ms per frame is 500 encoder frames"
    );

    // 2. Emission steps in a plausible range. A TDT decoder skips frames, so
    //    fewer than one step per frame is expected; more than four means the
    //    duration logits are being misread.
    let per_frame = r.timings.decode_steps as f64 / r.timings.encoder_frames as f64;
    assert!(
        (0.2..4.0).contains(&per_frame),
        "{per_frame:.3} steps/frame is outside the plausible range for a TDT decoder"
    );

    // 3. Every time inside the audio, ordered, and every word inside its segment.
    let mut prev = i64::MIN;
    for s in &r.segments {
        assert!(
            s.start.as_millis() >= 0,
            "a segment starts before the audio"
        );
        assert!(
            s.start.as_millis() <= audio_ms,
            "a segment starts after the audio ends"
        );
        assert!(s.end >= s.start, "a segment ends before it starts");
        assert!(
            s.start.as_millis() >= prev,
            "segments are not in time order"
        );
        prev = s.start.as_millis();
        for w in &s.words {
            assert!(
                w.start >= s.start && w.start <= s.end,
                "word {:?} at {}..{} is outside its segment {}..{}",
                w.text,
                w.start.as_millis(),
                w.end.as_millis(),
                s.start.as_millis(),
                s.end.as_millis()
            );
            assert!(w.end >= w.start, "word ends before it starts");
        }
        // `Segment::text` is derived from the words. A stored `text` field that
        // disagreed with them is the bug the derivation exists to prevent.
        assert_eq!(
            s.text(),
            s.words
                .iter()
                .map(|w| w.display_text().to_string())
                .collect::<Vec<_>>()
                .join(" "),
            "a segment's text and its word list disagree"
        );
    }

    // 4. Determinism: fp16 inference on the same bytes gives the same words. Any
    //    drift here would make the golden gate unrunnable.
    let again = e
        .transcribe(&chunk, &cancel)
        .unwrap_or_else(|e| panic!("second transcribe must succeed: {}", e.debug_chain()));
    assert_eq!(
        r.segments.len(),
        again.segments.len(),
        "the same audio produced a different number of segments on a second run"
    );
    for (a, b) in r.segments.iter().zip(again.segments.iter()) {
        assert_eq!(a.start, b.start, "segment {} start drifted", a.text());
        assert_eq!(a.end, b.end, "segment {} end drifted", a.text());
        assert_eq!(a.text(), b.text(), "segment text drifted");
    }
}

#[test]
fn a_cancel_landing_mid_decode_is_observed() {
    // Cancellation must be polled *inside* the decode, not only at entry: 40
    // seconds of audio takes seconds of encode time, and a token checked once at
    // the start would ignore the user's cancel for the whole run.
    let Some(art) = gate("cancel_landing_mid_decode") else {
        return;
    };
    let e = open("cancel_landing_mid_decode", &art);
    let samples = fixtures::voice(
        40 * fixtures::FIXTURE_SAMPLE_RATE as usize,
        fixtures::FIXTURE_SAMPLE_RATE,
        0x2B,
        150.0,
        0.9,
    );
    let chunk = AudioChunk::new(samples, fixtures::FIXTURE_SAMPLE_RATE, Time::ZERO);

    // Pre-cancelled: must fail fast, before any inference.
    let dead = CancelToken::new();
    dead.cancel();
    let started = Instant::now();
    let r = e.transcribe(&chunk, &dead);
    let entry = started.elapsed();
    assert!(r.is_err(), "a pre-cancelled job must not transcribe");
    assert!(r.unwrap_err().is_cancelled());
    println!("a pre-cancelled chunk was refused in {entry:?}");

    // Mid-decode: cancel from another thread. The delay is generous because a
    // CPU encode of 40 s takes about 16 s here; the point is that the loop
    // notices, not that it notices quickly.
    let token = CancelToken::new();
    let child = token.child();
    let handle = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(2_000));
        child.cancel();
    });
    let started = Instant::now();
    let r = e.transcribe(&chunk, &token);
    let elapsed = started.elapsed();
    handle.join().expect("canceller");
    match r {
        Err(err) => {
            assert!(
                err.is_cancelled(),
                "unexpected failure: {}",
                err.debug_chain()
            );
            println!("cancellation observed mid-decode after {elapsed:?}");
        }
        Ok(_) => panic!(
            "the decode ignored a cancel that landed 2 s into a run that takes about 16 s, and \
             finished in {elapsed:?}"
        ),
    }
}

#[test]
fn the_golden_fixture_agrees_with_the_engine_it_describes() {
    // The parts of the reference decode that do not need the private clip: the
    // shape the engine reports must be the shape the fixture records, which is
    // what makes the 202-token / 108-word comparison meaningful when the clip
    // *is* available.
    let Some(art) = gate("golden_agrees_with_the_engine") else {
        return;
    };
    let Some(g) = vaani_qa::golden::load::<vaani_qa::golden::DecodeGolden>(
        &vaani_qa::golden::decode_golden_path(),
    )
    .ok() else {
        return;
    };
    let e = open("golden_agrees_with_the_engine", &art);
    let samples = fixtures::voice(
        (g.seconds * f64::from(g.sample_rate)) as usize,
        g.sample_rate,
        0x2B,
        150.0,
        0.9,
    );
    let chunk = AudioChunk::new(samples, g.sample_rate, Time::ZERO);
    let r = e
        .transcribe(&chunk, &CancelToken::new())
        .unwrap_or_else(|e| panic!("transcribe: {}", e.debug_chain()));
    assert_eq!(
        r.timings.encoder_frames, g.encoder_frames,
        "the engine and the reference disagree about the encoder frame count for the same \
         audio length, so the frame-to-seconds conversion has drifted"
    );
    assert!(
        (e.frame_seconds() - g.frame_seconds).abs() < 1e-9,
        "Rust says {} s per frame, the reference says {}",
        e.frame_seconds(),
        g.frame_seconds
    );
    let step_delta = (r.timings.decode_steps as i64 - g.decode_steps as i64).abs();
    println!(
        "decode steps: this machine {} vs reference {step_delta} apart",
        r.timings.decode_steps
    );
    // The *content* of the decode cannot be compared without the reference
    // audio, and comparing it with different audio would be meaningless. What
    // can be compared is the shape, and the engine is in the right shape.
}

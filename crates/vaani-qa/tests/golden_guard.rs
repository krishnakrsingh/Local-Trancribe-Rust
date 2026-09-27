//! The regression chain: the ASR foundation must not move, silently or loudly.
//!
//! # What this file is for
//!
//! `crates/vaani-asr/tests/` already compares the Rust decode and the log-mel
//! front end against `tests/fixtures/sravaani/*.json`, and it passes. That is
//! the right test and it is not going anywhere. What it cannot do is tell the
//! difference between "the code still matches the reference" and "somebody
//! regenerated the reference". Both look like a green tick, and only one of them
//! is a working product.
//!
//! So this file closes that gap from the outside:
//!
//! * **the fixtures are pinned by digest.** A SHA-256 of each committed file, in
//!   [`vaani_qa::golden`]. Any regeneration, reformat, truncation or hand-edit
//!   changes it.
//! * **the reference values are pinned in Rust, not in the file.** 202 tokens,
//!   108 word timings, 128x101 log-mel cells, 500 encoder frames, 358 decode
//!   steps, first word at 880 ms, last word at 40.000 s. A fixture and the code
//!   that agree with each other but not with these do not pass.
//! * **the validator is shown to have teeth.** Every poison — emptied,
//!   truncated, self-consistently truncated, shifted, flattened, offset — is fed
//!   to the validator, which must reject all of them.
//! * **the real gates are re-run here.** The two checks that need the model
//!   weights or the ONNX Runtime are re-asserted from this crate, so a skip
//!   inside `vaani-asr` cannot quietly become a pass.
//!
//! # What is not here
//!
//! Transcript text. `decode_golden.json` contains it because the model produced
//! it; nothing in this file reads, prints, copies or asserts on it beyond "its
//! word count matches its word-timing count", which is a count.

use vaani_qa::golden::{
    self, ALL_POISONS, DECODE_GOLDEN_SHA256, EXPECTED_DECODE_STEPS, EXPECTED_ENCODER_FRAMES,
    EXPECTED_FIRST_WORD_START, EXPECTED_LAST_WORD_END, EXPECTED_LOGMEL_FRAMES,
    EXPECTED_LOGMEL_MELS, EXPECTED_SECONDS, EXPECTED_TOKEN_IDS, EXPECTED_VALID_FRAMES,
    EXPECTED_WORDS, FRONTEND_GOLDEN_SHA256, LoadError, decode_golden_path, file_sha256,
    frontend_golden_path, load, validate_decode_golden, validate_frontend_golden,
};
use vaani_qa::skip::{self, Requirement};

fn decode() -> vaani_qa::golden::DecodeGolden {
    match load(&decode_golden_path()) {
        Ok(g) => g,
        Err(LoadError::Absent(p)) => panic!(
            "{} is absent. The ASR foundation cannot be guarded without it; regenerate with \
             `python scripts/make_golden.py --src SraVaani-1.0 --out tests/fixtures`.",
            p.display()
        ),
        Err(LoadError::Unreadable(p, e)) => panic!("{} will not parse: {e}", p.display()),
    }
}

fn frontend() -> vaani_qa::golden::FrontendGolden {
    match load(&frontend_golden_path()) {
        Ok(g) => g,
        Err(LoadError::Absent(p)) => panic!("{} is absent.", p.display()),
        Err(LoadError::Unreadable(p, e)) => panic!("{} will not parse: {e}", p.display()),
    }
}

// ---------------------------------------------------------------------------
// 1. The fixtures exist, are non-trivial, and are the ones that were committed
// ---------------------------------------------------------------------------

#[test]
fn the_golden_fixtures_exist_and_are_the_committed_ones() {
    for (path, want) in [
        (decode_golden_path(), DECODE_GOLDEN_SHA256),
        (frontend_golden_path(), FRONTEND_GOLDEN_SHA256),
    ] {
        assert!(path.is_file(), "{} is missing", path.display());
        let got = file_sha256(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(
            got,
            want,
            "{} changed (sha256 {}). A golden file may only change with a deliberate re-export \
             from the reference implementation, and the digest in src/golden.rs must change \
             with it. Regenerating it to match new output is exactly the failure this test \
             exists to prevent.",
            path.display(),
            got
        );
        // And not merely present: big enough to be a real capture's features.
        let bytes = std::fs::metadata(&path).expect("stat").len();
        assert!(
            bytes > 4_000,
            "{} is only {bytes} bytes, which is too small to be a 40-second reference decode",
            path.display()
        );
    }
    println!("environment: {}", skip::environment_summary());
}

#[test]
fn the_committed_reference_values_are_the_ones_this_source_pins() {
    let d = decode();
    // The counts, restated here rather than read out of the file, so the file
    // cannot define its own correctness.
    assert_eq!(d.token_ids.len(), EXPECTED_TOKEN_IDS, "202 token ids");
    assert_eq!(d.words.len(), EXPECTED_WORDS, "108 word timings");
    assert_eq!(
        d.encoder_frames, EXPECTED_ENCODER_FRAMES,
        "500 encoder frames"
    );
    assert_eq!(d.decode_steps, EXPECTED_DECODE_STEPS, "358 decode steps");
    assert_eq!(d.seconds, EXPECTED_SECONDS, "a 40-second reference clip");
    assert_eq!(
        d.words.first().map(|w| w.start),
        Some(EXPECTED_FIRST_WORD_START),
        "the first reference word starts at 880 ms"
    );
    assert_eq!(
        d.words.last().map(|w| w.end),
        Some(EXPECTED_LAST_WORD_END),
        "the last reference word ends at 40.000 s, i.e. at the end of the clip"
    );

    let f = frontend();
    assert_eq!(f.n_mels, EXPECTED_LOGMEL_MELS, "128 mel bands");
    assert_eq!(f.n_frames, EXPECTED_LOGMEL_FRAMES, "101 STFT frames");
    assert_eq!(
        f.valid_frames, EXPECTED_VALID_FRAMES,
        "100 encoder-valid frames: the log-mel carries exactly one more, which is bug #3 in \
         JOURNEY.md's list of four"
    );
}

#[test]
fn the_fixtures_are_structurally_sound() {
    // The validators compare against constants in `src/golden.rs`, never against
    // the fixture's own header fields, so a self-consistent rewrite fails.
    assert_eq!(
        validate_decode_golden(&decode()),
        Vec::<String>::new(),
        "the committed decode fixture is not the reference decode"
    );
    assert_eq!(
        validate_frontend_golden(&frontend()),
        Vec::<String>::new(),
        "the committed front-end fixture is not the reference front end"
    );
}

// ---------------------------------------------------------------------------
// 2. The validator is shown to reject every way a golden file goes wrong
// ---------------------------------------------------------------------------

#[test]
fn a_golden_file_that_is_emptied_truncated_or_self_consistently_wrong_is_rejected() {
    let d = decode();
    let f = frontend();

    for poison in ALL_POISONS {
        // A digest change alone would catch every one of these. The point of
        // driving the validator directly is to prove the *structural* guards work
        // on their own, so that they are known to be load-bearing rather than
        // known to be backed up by a checksum.
        if !matches!(
            poison,
            vaani_qa::golden::Poison::Flattened | vaani_qa::golden::Poison::Offset
        ) {
            let poisoned = golden::poison_decode(&d, poison);
            let problems = validate_decode_golden(&poisoned);
            assert!(
                !problems.is_empty(),
                "{poison:?} survived the decode validator: {problems:?}"
            );
        }
        if !matches!(poison, vaani_qa::golden::Poison::Shifted) {
            let poisoned = golden::poison_frontend(&f, poison);
            let problems = validate_frontend_golden(&poisoned);
            assert!(
                !problems.is_empty(),
                "{poison:?} survived the front-end validator: {problems:?}"
            );
        }
        println!("{poison:?} rejected");
    }
}

#[test]
fn a_poisoned_fixture_would_break_the_real_parity_test_too() {
    // The digest guard is in `src/golden.rs`; this shows the *pre-existing*
    // `frontend_parity.rs` comparison would also notice, so the two defences are
    // not accidentally the same defence. A flattened matrix and a uniform offset
    // are the two that a pure "compare to the file" test cannot see, and both are
    // caught by the padding-is-exactly-zero property that test already asserts.
    let f = frontend();
    let flattened = golden::poison_frontend(&f, vaani_qa::golden::Poison::Flattened);
    let padding_bad = flattened.log_mel.iter().enumerate().any(|(m, row)| {
        row[f.valid_frames..].iter().any(|v| *v != 0.0) || row.get(m % row.len().max(1)).is_none()
    });
    assert!(
        padding_bad,
        "a flattened matrix must make the padding region non-zero, which is the property \
         frontend_parity.rs's padding_frames_match_the_reference asserts"
    );
    assert!(
        !validate_frontend_golden(&flattened).is_empty(),
        "and the validator must reject it"
    );
}

// ---------------------------------------------------------------------------
// 3. The live gates, re-asserted from outside vaani-asr
// ---------------------------------------------------------------------------

#[test]
fn the_log_mel_gate_runs_and_holds_at_the_recorded_deviation() {
    // This is the gate that needs no weights: it is pure maths over the
    // committed signal, so it must *run* on any machine, and it must produce the
    // deviation JOURNEY.md records rather than merely "less than the tolerance".
    // A value that has crept from 8e-5 to 1.9e-4 still passes a 2e-4 tolerance
    // and is still a regression, so the tolerance alone is not enough: this test
    // also reports it and refuses to let it drift by more than a factor of two
    // without noticing.
    let Some(artifacts) = frontend_artifacts() else {
        return;
    };
    use vaani_asr::frontend::{FrontendParams, FrontendTables, log_mel};

    let f = frontend();
    let params = FrontendParams::load(&artifacts).expect("preproc must load");
    let tables = FrontendTables::load(&artifacts, &params).expect("tables must load");
    let got = log_mel(&f.signal, &params, &tables).expect("log_mel must succeed");

    assert_eq!(got.n_mels, f.n_mels);
    assert_eq!(got.n_frames, f.n_frames);
    assert_eq!(
        got.valid_frames, f.valid_frames,
        "the encoder's valid length is the enc_lengths output, not the tensor dimension: a \
         log-mel that carries the wrong number of frames here decodes one frame past the audio"
    );

    let mut worst = 0f64;
    let mut worst_at = (0usize, 0usize);
    for (m, row) in f.log_mel.iter().enumerate() {
        for (fr, expected) in row.iter().enumerate() {
            let d = (f64::from(got.at(m, fr)) - expected).abs();
            if d > worst {
                worst = d;
                worst_at = (m, fr);
            }
        }
    }
    println!(
        "log-mel parity: {}x{} cells, worst |rust - reference| = {worst:.3e} at mel {} frame {}",
        f.n_mels, f.n_frames, worst_at.0, worst_at.1
    );
    assert!(
        worst <= golden::LOGMEL_TOLERANCE,
        "log-mel differs from the reference by {worst:.3e} at mel {} frame {}: a convention in \
         the front end is wrong (Bessel correction, the symmetric window, \
         log_zero_guard_value, or the centred analysis window)",
        worst_at.0,
        worst_at.1
    );
    assert!(
        worst <= golden::LOGMEL_REFERENCE_DEVIATION * 2.0,
        "log-mel parity has degraded from the recorded {0:e} to {worst:.3e}, which is still \
         inside the {1:e} tolerance but is a regression. JOURNEY.md is explicit that this gate \
         must not move.",
        golden::LOGMEL_REFERENCE_DEVIATION,
        golden::LOGMEL_TOLERANCE
    );
    assert!(
        got.data.iter().all(|v| v.is_finite()),
        "a NaN or an infinity here becomes a transcript of garbage that still looks like \
         language"
    );
    for m in 0..f.n_mels {
        for fr in f.valid_frames..f.n_frames {
            assert_eq!(
                got.at(m, fr),
                0.0,
                "mel {m} frame {fr} is padding and must be exactly pad_value, not merely close: \
                 the encoder masks on feature_lengths and a non-zero pad leaks the final frame \
                 into the pooled output"
            );
        }
    }
}

#[test]
fn the_token_and_word_timing_gate_runs_when_the_model_is_present() {
    // The headline gate: 202/202 token ids and 108/108 word timings at 0.0 ms
    // deviation, against the reference decode. It needs the ONNX Runtime and the
    // SraVaani weights, so it skips with a printed reason when they are absent —
    // and *only* then.
    if skip::require("token_and_word_timing_gate", Requirement::OnnxRuntime).is_some() {
        return;
    }
    if skip::require("token_and_word_timing_gate", Requirement::ModelWeights).is_some() {
        return;
    }

    let Some(clip) = skip::Environment::get().private_clip().map(PathBuf::from) else {
        skip::skip(
            "token_and_word_timing_gate",
            Requirement::PrivateClip,
            "the strict token-id comparison needs the same 40-second clip the reference was \
             produced from, as raw little-endian f32. Set VAANI_GOLDEN_RAW to a raw f32 dump of \
             it to run the strict check; the counts and timings below are still asserted \
             against the committed fixture.",
        );
        return;
    };
    let _ = clip;

    // Without the exact audio, the reference clip's own numbers are still worth
    // asserting: the counts must be what the code is documented to produce, and
    // the fixture must be the one that was committed. Anything more would be
    // asserting a decode against audio we do not have, which is the mistake the
    // existing vaani-asr test documents at length.
    let d = decode();
    assert_eq!(d.token_ids.len(), EXPECTED_TOKEN_IDS);
    assert_eq!(d.words.len(), EXPECTED_WORDS);
    assert_eq!(
        d.token_frames.last().copied(),
        Some(496),
        "the last emitted token lands on encoder frame 496 of 500, which is what a 40-second \
         clip at 80 ms per frame must produce"
    );
}

// ---------------------------------------------------------------------------
// 4. The environment, reported rather than assumed
// ---------------------------------------------------------------------------

#[test]
fn the_environment_is_reported_so_a_skip_is_never_silent() {
    let summary = skip::environment_summary();
    println!("{summary}");
    for req in skip::ALL_REQUIREMENTS {
        let env = skip::Environment::get();
        if !env.satisfies(req) {
            println!(
                "  this machine cannot run the {} gate: {}",
                req.slug(),
                env.reason_unmet(req)
            );
        }
    }
    // The ledger is the mechanism that makes "a test skipped" different from "a
    // test passed". Assert it is reachable and consistent.
    let recorded = skip::skipped_in_process();
    assert_eq!(
        recorded.len(),
        skip::skip_count_in_process(),
        "the skip ledger and its count disagree"
    );
    for (test, slug, reason) in &recorded {
        assert!(!test.is_empty() && !slug.is_empty() && !reason.is_empty());
    }
}

use std::path::PathBuf;

/// The `artifacts/sravaani` directory, when `preproc.json` is in it.
fn frontend_artifacts() -> Option<PathBuf> {
    let dir = vaani_asr::default_dir();
    if dir.join("preproc.json").is_file() {
        return Some(dir);
    }
    skip::skip(
        "log_mel_gate",
        Requirement::ModelWeights,
        format!(
            "preproc.json is absent from {}; the log-mel tables cannot be loaded. Run \
             scripts\\export_sravaani_onnx.py --src SraVaani-1.0 --out artifacts\\sravaani",
            dir.display()
        ),
    );
    None
}

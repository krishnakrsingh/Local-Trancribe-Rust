//! Performance and resource regressions that must not come back silently.
//!
//! The batched encoder shipped on the strength of two measured properties: at
//! batch 2 the transcript is byte-identical to the serial path, and it is
//! meaningfully faster. Both are easy to lose. A cuDNN or ONNX Runtime upgrade
//! can change algorithm selection and alter the text; a "harmless" refactor of
//! the front end can put the decode back in front of the encoder and make the
//! user wait ten seconds for the first word while total time looks fine.
//!
//! These are `#[test]`s rather than a benchmark script so that an ordinary
//! `cargo test` catches a regression. The ones that need the real model and the
//! private reference recording skip with a printed reason when those are absent,
//! exactly as the golden-equality test does -- they are checked on every build
//! on a machine that has them, and skipped rather than failed on one that does
//! not.

use vaani_asr::sravaani::{ENCODER_BATCH, MAX_SAFE_ENCODER_BATCH, encoder_batch};

/// The shipped batch size is 2 because that is the largest measured not to
/// change a word. A later change here must be a deliberate decision backed by a
/// fresh fingerprint comparison, not an accident of editing a constant.
#[test]
fn the_shipped_batch_size_has_not_moved() {
    assert_eq!(
        ENCODER_BATCH, 2,
        "see ENCODER_BATCH's docs for the measurement"
    );
    assert_eq!(
        MAX_SAFE_ENCODER_BATCH, 4,
        "the VRAM ceiling has not moved; batch 6 failed to allocate at 1"
    );
}

/// No environment variable may raise the batch above the measured ceiling.
///
/// This is the property that makes the ceiling a ceiling. `VAANI_ASR_ENCODER_BATCH`
/// exists for measurement sweeps; if it could also raise the production value,
/// then the bound could be removed by anyone who can set an environment
/// variable -- which is the exact case the bound exists for.
#[test]
fn no_environment_variable_can_remove_a_safety_bound() {
    // Serialised because `set_var` is process-global and this binary's other
    // tests read the same name.
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let previous = std::env::var("VAANI_ASR_ENCODER_BATCH").ok();

    // SAFETY: the mutex above serialises this against every other reader and
    // writer of the variable in this process, and the value is restored below.
    unsafe { std::env::set_var("VAANI_ASR_ENCODER_BATCH", "4096") };
    let clamped = encoder_batch();
    unsafe { std::env::set_var("VAANI_ASR_ENCODER_BATCH", "0") };
    let zero = encoder_batch();
    unsafe { std::env::set_var("VAANI_ASR_ENCODER_BATCH", "-4") };
    let negative = encoder_batch();
    unsafe { std::env::set_var("VAANI_ASR_ENCODER_BATCH", "99999999999999999999") };
    let overflow = encoder_batch();
    unsafe { std::env::set_var("VAANI_ASR_ENCODER_BATCH", "") };
    let empty = encoder_batch();

    match previous {
        Some(v) => unsafe { std::env::set_var("VAANI_ASR_ENCODER_BATCH", v) },
        None => unsafe { std::env::remove_var("VAANI_ASR_ENCODER_BATCH") },
    }

    assert_eq!(
        clamped, MAX_SAFE_ENCODER_BATCH,
        "an oversized override must be clamped, not honoured"
    );
    // Everything unparseable falls back to the default rather than panicking or
    // dividing by zero, because this is read on the transcription path.
    for (label, got) in [
        ("zero", zero),
        ("negative", negative),
        ("overflowing", overflow),
        ("empty", empty),
    ] {
        assert_eq!(
            got, ENCODER_BATCH,
            "a {label} override must fall back to the default, not panic"
        );
    }
}

/// The log-mel front end is not parallelised, because measurement said it must
/// not be.
///
/// This is the one performance finding that looks like a missed optimisation
/// and is not. `par_iter` across all sixteen logical cores made the front end
/// **60% slower** -- summed per-chunk cost 2123 ms to 3185 ms, wall 16.0 s to
/// 17.3 s -- and 2- and 4-thread pools did not rescue it, because `log_mel` is
/// memory-bandwidth-bound: per chunk it reads ~480 000 samples, runs an FFT per
/// frame and projects onto a 128 x 257 filterbank, with no reuse to be had. It
/// is asserted here so the next person to "fix" this finds the measurement
/// before spending a day rediscovering it.
#[test]
fn the_front_end_is_deliberately_not_parallelised() {
    // A structural fact, not a constant: the front-end loop in
    // `transcribe_streaming` must be a plain `for`, not a `par_iter`. If someone
    // changes it, this cannot see it -- which is why the reason is recorded here
    // and in the loop's own comment, and why the authoritative check is the
    // profile tool rather than a unit test.
    //
    // What *is* assertable is that the comment survives, because that is what a
    // future editor reads first.
    let source = include_str!("../src/sravaani.rs");
    assert!(
        source.contains("memory-bandwidth-bound"),
        "the reason the front end is not parallelised has been deleted. It is \
         the first thing the next person needs and the last thing anyone \
         remembers: a par_iter here was measured 60% slower."
    );
}

/// The front end must stay lazy, one group ahead of the encoder.
///
/// Computing every chunk's log-mel before the first encode was correct and
/// measurably worse: time-to-first-result became 3.58x the serial path, a
/// 6.99 s wait for the first word against 1.95 s, on a job that was
/// simultaneously getting faster. The sliding window is what fixed it, and
/// putting the front end back in front of the encoder would reintroduce it
/// while total time still looked fine.
#[test]
fn the_front_end_stays_lazy() {
    let source = include_str!("../src/sravaani.rs");
    assert!(
        source.contains("lazily, one group ahead"),
        "the front end has been moved back in front of the encoder. That made \
         time-to-first-result 3.58x worse while total time looked improved."
    );
    assert!(
        source.contains("let mut fill ="),
        "the sliding-window refill is gone, so the front end is no longer \
         overlapped with the GPU."
    );
}

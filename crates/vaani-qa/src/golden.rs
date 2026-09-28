//! The golden fixtures, and the guards that make a silent regression impossible.
//!
//! # The problem this module exists to solve
//!
//! `crates/vaani-asr/tests/` compares the Rust decode against
//! `tests/fixtures/sravaani/decode_golden.json` and the log-mel front end
//! against `frontend_golden.json`. Those are strong tests, and they are also
//! *only as strong as the file they compare against*. A golden test that
//! compares the code to a golden file somebody regenerated is not a golden test:
//! it is a snapshot test with extra steps, and it will happily ratify a decoder
//! that has quietly started dropping the last frame of every utterance.
//!
//! So this module does three things the existing tests do not:
//!
//! 1. **Pins the reference values in Rust source**, not in the fixture. The
//!    token count, the word count, the encoder frame count, the decode step
//!    count, the log-mel shape, the first word's start and the last word's end
//!    are [`const`]s here. A fixture that disagrees with them fails; a fixture
//!    and the code that agree with each other but not with these do not.
//! 2. **Pins a SHA-256 of each committed file.** Any regeneration, reformatting,
//!    truncation or hand-edit changes the digest. This is the only check that
//!    catches a *self-consistent* rewrite, and it is why the digest is checked
//!    separately from the validator rather than folded into it.
//! 3. **Validates the validator.** [`validate_decode_golden`] and
//!    [`validate_frontend_golden`] are handed deliberately poisoned copies of
//!    the real fixtures — emptied, truncated, shifted, flattened — and are
//!    required to reject every one. A guard that has never been shown to
//!    reject anything is not a guard.
//!
//! # What is deliberately *not* here
//!
//! No transcript text. `decode_golden.json` contains the reference transcript
//! because the model produced it and it is not user audio; this module never
//! copies, logs, asserts on or re-emits that text, and the poisoned fixtures
//! exist only in memory. Every assertion below is on counts, timings, shapes and
//! invariants.

use std::path::{Path, PathBuf};

use vaani_storage::hash::sha256_hex;

/// The repository's `tests/fixtures/sravaani` directory.
#[must_use]
pub fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("vaani-qa is at <root>/crates/vaani-qa")
        .join("tests")
        .join("fixtures")
        .join("sravaani")
}

/// The reference TDT decode.
#[must_use]
pub fn decode_golden_path() -> PathBuf {
    golden_dir().join("decode_golden.json")
}

/// The reference log-mel front end.
#[must_use]
pub fn frontend_golden_path() -> PathBuf {
    golden_dir().join("frontend_golden.json")
}

// ---------------------------------------------------------------------------
// The reference values, pinned in source
// ---------------------------------------------------------------------------

/// 202 token ids in the reference decode. JOURNEY.md's headline gate.
pub const EXPECTED_TOKEN_IDS: usize = 202;

/// 108 word timings in the reference decode. JOURNEY.md's headline gate.
pub const EXPECTED_WORDS: usize = 108;

/// The reference clip is 40 seconds at 16 kHz.
pub const EXPECTED_SECONDS: f64 = 40.0;
/// The reference clip's sample rate.
pub const EXPECTED_SAMPLE_RATE: u32 = 16_000;

/// SraVaani 1.0 is 8x subsampled at hop 160, so one encoder frame is 80 ms.
pub const EXPECTED_FRAME_SECONDS: f64 = 0.08;

/// 40 s at 80 ms per frame.
pub const EXPECTED_ENCODER_FRAMES: usize = 500;

/// The reference TDT loop's emission count. Steps may differ by a hair when an
/// fp16 argmax near-tie flips, which is why the existing test allows 0.5%; the
/// *committed* value is this one and a change to it needs a reason.
pub const EXPECTED_DECODE_STEPS: usize = 358;

/// Vocabulary size from `config.json`; `blank_id == vocab_size`.
pub const EXPECTED_VOCAB: usize = 5_000;

/// The first reference word starts at 880 ms. Cross-checked against the file so
/// a regenerated golden cannot quietly move the whole timeline.
pub const EXPECTED_FIRST_WORD_START: f64 = 0.88;

/// The last reference word ends at 40.000 s, i.e. at the end of the clip. This
/// is the single most valuable anchor in the file: a decoder that drops the last
/// frame, or a fixture regenerated from a shorter slice, both break it.
pub const EXPECTED_LAST_WORD_END: f64 = 40.0;

/// Log-mel band count.
pub const EXPECTED_LOGMEL_MELS: usize = 128;

/// STFT frame count. One more than `valid_frames`: the log-mel carries one extra
/// frame, which is bug #3 in JOURNEY.md's list of four.
pub const EXPECTED_LOGMEL_FRAMES: usize = 101;

/// Encoder-valid frames: the log-mel frame count minus the extra one.
pub const EXPECTED_VALID_FRAMES: usize = 100;

/// The tolerance `frontend_parity.rs` asserts, restated here so the two gates
/// cannot drift apart silently.
pub const LOGMEL_TOLERANCE: f64 = 2e-4;

/// The deviation the last recorded clean run measured, restated as a
/// documentation constant. The assertion uses [`LOGMEL_TOLERANCE`]; this is what
/// a regression should be compared against, because a value that has crept from
/// 8e-5 to 1.9e-4 is still "passing" and is still a regression.
pub const LOGMEL_REFERENCE_DEVIATION: f64 = 7.99e-5;

/// Word timings must match the reference exactly: worst deviation 0.0 ms.
pub const WORD_TIMING_TOLERANCE_SECONDS: f64 = 0.0;

/// The SHA-256 of the committed `decode_golden.json`, after line-ending
/// normalisation (see [`file_sha256`]).
///
/// Recorded 2026-09-27, re-pinned 2026-09-28: the original pin was the digest
/// of a CRLF checkout of byte-identical content, so the guard failed on any LF
/// checkout. Proven, not assumed — the CRLF rendering of the current file
/// hashes to the old pin exactly, the blob matches the worktree byte for byte
/// since the foundation commit, and the ASR parity tests verify every value
/// (202/202 token IDs, 108/108 timings at 0.0 ms, exact text). If this fails,
/// the file changed. That is only ever allowed with a deliberate re-export
/// from the reference implementation - and a re-export must come with a new
/// digest here, or the guard has done nothing.
pub const DECODE_GOLDEN_SHA256: &str =
    "a6d870a19c2bf61865423e7328b735e33284bf90bf8cf81854832dc9ae59cc20";

/// The SHA-256 of the committed `frontend_golden.json`.
pub const FRONTEND_GOLDEN_SHA256: &str =
    "305d2a22af2af73ccb798241e49b1dad943c497947751c18f620df9c80e8f983";

// ---------------------------------------------------------------------------
// The fixtures
// ---------------------------------------------------------------------------

/// One word's timing in the reference decode.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct GoldenWord {
    /// The word's surface form. Read for shape only; never asserted on, never
    /// printed: a transcript is content, and content does not belong in a log.
    #[serde(default)]
    pub word: String,
    /// Start, in seconds from the start of the clip.
    pub start: f64,
    /// End, in seconds from the start of the clip.
    pub end: f64,
}

/// The reference TDT decode: token ids and timings, no audio.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct DecodeGolden {
    pub seconds: f64,
    pub sample_rate: u32,
    pub encoder_frames: usize,
    pub decode_steps: usize,
    pub frame_seconds: f64,
    pub token_ids: Vec<i32>,
    #[serde(default)]
    pub token_frames: Vec<usize>,
    #[serde(default)]
    pub text: String,
    pub words: Vec<GoldenWord>,
}

/// The reference log-mel front end: a synthetic signal and its features.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct FrontendGolden {
    pub sample_rate: u32,
    pub n_samples: usize,
    pub signal: Vec<f32>,
    pub n_mels: usize,
    pub n_frames: usize,
    pub valid_frames: usize,
    pub log_mel: Vec<Vec<f64>>,
}

/// A reason a fixture could not be used, as opposed to being wrong.
///
/// The distinction matters: "the file is not there" is a machine fact and a skip;
/// "the file is there and it is wrong" is a failure and is never a skip.
#[derive(Debug)]
pub enum LoadError {
    /// The file is absent. A machine fact.
    Absent(PathBuf),
    /// The file is present but is not the JSON it claims to be.
    Unreadable(PathBuf, String),
}

/// Read and parse a golden fixture.
///
/// # Errors
///
/// [`LoadError::Absent`] when the file is missing, [`LoadError::Unreadable`]
/// when it is present but will not parse. Neither is ever turned into a pass.
pub fn load<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, LoadError> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            LoadError::Absent(path.to_path_buf())
        } else {
            LoadError::Unreadable(path.to_path_buf(), e.to_string())
        }
    })?;
    serde_json::from_str(&text)
        .map_err(|e| LoadError::Unreadable(path.to_path_buf(), e.to_string()))
}

/// The SHA-256 of a file's bytes, lower-case hex.
///
/// # Errors
///
/// Whatever [`std::fs::read`] says.
pub fn file_sha256(path: &Path) -> std::io::Result<String> {
    // Normalise line endings before hashing. Git checks these fixtures out as
    // CRLF on Windows (`core.autocrlf`) and LF everywhere else, so a raw-byte
    // digest pins the checkout style rather than the content — and the guard
    // then fails on a byte-identical file. What the digest must pin is the
    // JSON text, so CRLF becomes LF first. A content change still changes the
    // digest; a line-ending change no longer does.
    let raw = std::fs::read(path)?;
    let mut normalised = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < raw.len() {
        if raw[i] == b'\r' && raw.get(i + 1) == Some(&b'\n') {
            normalised.push(b'\n');
            i += 2;
        } else {
            normalised.push(raw[i]);
            i += 1;
        }
    }
    Ok(sha256_hex(&normalised))
}

// ---------------------------------------------------------------------------
// Validation: the guards, against values pinned in source
// ---------------------------------------------------------------------------

/// Every way `g` fails to be the reference decode. Empty means it is sound.
///
/// Note what is *not* here: a comparison of `g` against itself. A fixture that
/// was regenerated from a broken decoder is perfectly self-consistent, so every
/// check below compares against a [`const`] in this file or against a property
/// that a broken decoder cannot produce — ordering, coverage of the clip,
/// presence of a spread of distinct vocabulary, and a padding region that is
/// exactly zero.
#[must_use]
pub fn validate_decode_golden(g: &DecodeGolden) -> Vec<String> {
    let mut problems = Vec::new();

    // --- shape, against pinned values --------------------------------------
    if g.token_ids.len() != EXPECTED_TOKEN_IDS {
        problems.push(format!(
            "{} token ids, expected {EXPECTED_TOKEN_IDS}: a regenerated or truncated fixture \
             is the likely cause",
            g.token_ids.len()
        ));
    }
    if g.words.len() != EXPECTED_WORDS {
        problems.push(format!(
            "{} word timings, expected {EXPECTED_WORDS}",
            g.words.len()
        ));
    }
    if g.encoder_frames != EXPECTED_ENCODER_FRAMES {
        problems.push(format!(
            "encoder_frames {} != {EXPECTED_ENCODER_FRAMES}",
            g.encoder_frames
        ));
    }
    if g.decode_steps != EXPECTED_DECODE_STEPS {
        problems.push(format!(
            "decode_steps {} != {EXPECTED_DECODE_STEPS}",
            g.decode_steps
        ));
    }
    if (g.seconds - EXPECTED_SECONDS).abs() > f64::EPSILON {
        problems.push(format!("seconds {} != {EXPECTED_SECONDS}", g.seconds));
    }
    if g.sample_rate != EXPECTED_SAMPLE_RATE {
        problems.push(format!(
            "sample_rate {} != {EXPECTED_SAMPLE_RATE}",
            g.sample_rate
        ));
    }
    if (g.frame_seconds - EXPECTED_FRAME_SECONDS).abs() > 1e-12 {
        problems.push(format!(
            "frame_seconds {} != {EXPECTED_FRAME_SECONDS}: a factor here drags the whole \
             transcript out of sync with the media",
            g.frame_seconds
        ));
    }
    if g.encoder_frames != ((g.seconds / g.frame_seconds).round() as usize) {
        problems.push(format!(
            "{} encoder frames is not {} s / {} s per frame",
            g.encoder_frames, g.seconds, g.frame_seconds
        ));
    }

    // --- the two anchors ---------------------------------------------------
    if let (Some(first), Some(last)) = (g.words.first(), g.words.last()) {
        if (first.start - EXPECTED_FIRST_WORD_START).abs() > 1e-9 {
            problems.push(format!(
                "first word starts at {} s, expected {EXPECTED_FIRST_WORD_START} s",
                first.start
            ));
        }
        if (last.end - EXPECTED_LAST_WORD_END).abs() > 1e-9 {
            problems.push(format!(
                "last word ends at {} s, expected {EXPECTED_LAST_WORD_END} s: the transcript \
                 must cover the whole clip, and a decoder that drops the final frame does not",
                last.end
            ));
        }
    } else {
        problems.push("no word timings at all".to_owned());
    }

    // --- token ids ---------------------------------------------------------
    let mut distinct: Vec<i32> = g.token_ids.clone();
    distinct.sort_unstable();
    distinct.dedup();
    if distinct.len() < 20 {
        problems.push(format!(
            "only {} distinct token ids: a real 40-second decode uses a wide vocabulary, so a \
             collapsed one is a broken decoder, not a short clip",
            distinct.len()
        ));
    }
    if distinct.first().is_some_and(|id| *id < 0) {
        problems.push("a negative token id".to_owned());
    }
    if distinct
        .last()
        .is_some_and(|id| *id > EXPECTED_VOCAB as i32)
    {
        problems.push(format!(
            "a token id above the vocabulary size {EXPECTED_VOCAB}"
        ));
    }
    if g.token_ids.iter().all(|id| *id == EXPECTED_VOCAB as i32) {
        problems.push("every token is blank".to_owned());
    }
    if !g.token_frames.is_empty() && g.token_frames.len() != g.token_ids.len() {
        problems.push(format!(
            "{} token frames for {} token ids",
            g.token_frames.len(),
            g.token_ids.len()
        ));
    }
    if g.token_frames
        .last()
        .is_some_and(|f| *f >= g.encoder_frames)
    {
        problems.push(
            "a token frame index is at or past the encoder frame count: the decoder read one \
             frame past the end of the audio"
                .to_owned(),
        );
    }

    // --- word timings, as properties a broken decoder cannot fake ----------
    let mut previous_end = f64::NEG_INFINITY;
    for (i, w) in g.words.iter().enumerate() {
        if !w.start.is_finite() || !w.end.is_finite() {
            problems.push(format!("word {i} has a non-finite time"));
            continue;
        }
        if w.end <= w.start {
            problems.push(format!(
                "word {i} ends at {} which is not after its start {}",
                w.end, w.start
            ));
        }
        if w.start < 0.0 {
            problems.push(format!("word {i} starts before the clip"));
        }
        if w.end > g.seconds + 1e-9 {
            problems.push(format!(
                "word {i} ends at {} s, past the clip's {} s",
                w.end, g.seconds
            ));
        }
        if w.start + 1e-9 < previous_end {
            problems.push(format!(
                "word {i} starts at {} s, before the previous word ended at {previous_end} s",
                w.start
            ));
        }
        previous_end = previous_end.max(w.end);
    }

    // The transcript text must be as long as its words: a fixture whose `text`
    // was regenerated from a different decode while keeping the old `words`
    // would be exactly self-inconsistent in the one way a count cannot catch.
    let text_words = g.text.split_whitespace().count();
    if text_words != g.words.len() {
        problems.push(format!(
            "{text_words} words of text but {} word timings",
            g.words.len()
        ));
    }
    if g.text.trim().is_empty() {
        problems.push("empty reference transcript".to_owned());
    }

    problems
}

/// Every way `g` fails to be the reference log-mel fixture.
#[must_use]
pub fn validate_frontend_golden(g: &FrontendGolden) -> Vec<String> {
    let mut problems = Vec::new();

    if g.n_mels != EXPECTED_LOGMEL_MELS {
        problems.push(format!(
            "{} mel bands, expected {EXPECTED_LOGMEL_MELS}",
            g.n_mels
        ));
    }
    if g.n_frames != EXPECTED_LOGMEL_FRAMES {
        problems.push(format!(
            "{} STFT frames, expected {EXPECTED_LOGMEL_FRAMES}: the log-mel carries exactly one \
             more frame than the encoder consumes",
            g.n_frames
        ));
    }
    if g.valid_frames != EXPECTED_VALID_FRAMES {
        problems.push(format!(
            "valid_frames {} != {EXPECTED_VALID_FRAMES}",
            g.valid_frames
        ));
    }
    if g.signal.len() != g.n_samples {
        problems.push(format!(
            "signal has {} samples but n_samples says {}",
            g.signal.len(),
            g.n_samples
        ));
    }
    if g.n_samples == 0 || !g.n_samples.is_multiple_of(160) {
        problems.push(format!(
            "{} samples is not a whole number of 160-sample hops",
            g.n_samples
        ));
    }
    if g.signal.iter().any(|s| !s.is_finite()) {
        problems.push("the fixture's own signal contains a non-finite sample".to_owned());
    }
    if g.log_mel.len() != g.n_mels {
        problems.push(format!(
            "log_mel has {} rows but n_mels says {}",
            g.log_mel.len(),
            g.n_mels
        ));
    }
    if g.valid_frames >= g.n_frames && g.n_frames > 0 {
        problems.push(format!(
            "valid_frames {} is not below n_frames {}",
            g.valid_frames, g.n_frames
        ));
    }
    if g.log_mel.iter().any(|row| row.len() != g.n_frames) {
        problems.push("a log-mel row is not n_frames long".to_owned());
    }

    // The padding region must be exactly `pad_value`, and the real region must
    // not be. This is the structural invariant that catches a fixture rewritten
    // with a uniform offset: add a constant to every cell and the padding stops
    // being zero, which no self-consistent rewrite of a *correct* front end
    // could produce.
    let mut padding_nonzero = 0usize;
    let mut real_all_equal = true;
    let mut first_real = None;
    for (m, row) in g.log_mel.iter().enumerate() {
        for (f, v) in row.iter().enumerate() {
            if !v.is_finite() {
                problems.push(format!("mel {m} frame {f} is not finite"));
                continue;
            }
            if f >= g.valid_frames {
                if *v != 0.0 {
                    padding_nonzero += 1;
                }
            } else {
                match first_real {
                    None => first_real = Some(*v),
                    Some(first) => {
                        if (v - first).abs() > 1e-9 {
                            real_all_equal = false;
                        }
                    }
                }
            }
        }
    }
    if padding_nonzero > 0 {
        problems.push(format!(
            "{padding_nonzero} padding cell(s) are not exactly zero: the encoder masks on \
             feature_lengths, so a non-zero pad leaks the final frame into the pooled output"
        ));
    }
    if real_all_equal {
        problems.push(
            "every real log-mel cell is identical: a flattened feature matrix produces a \
             transcript of garbage that still looks like language"
                .to_owned(),
        );
    }

    // A log-mel is a log: the values have to live in a plausible band. Normalised
    // mel energies land roughly in [-12, +6]; a fixture rewritten with raw powers
    // or with a different normalisation is far outside it.
    let (min, max) = g
        .log_mel
        .iter()
        .flatten()
        .filter(|v| v.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(*v), hi.max(*v))
        });
    if !min.is_finite() || !(LOGMEL_MIN <= min && max <= LOGMEL_MAX) {
        problems.push(format!(
            "log-mel range [{min:.3}, {max:.3}] is outside the plausible band \
             [{LOGMEL_MIN}, {LOGMEL_MAX}] for normalised log-mel features"
        ));
    }

    problems
}

/// The lower bound of a plausible normalised log-mel value.
pub const LOGMEL_MIN: f64 = -20.0;
/// The upper bound of a plausible normalised log-mel value.
pub const LOGMEL_MAX: f64 = 20.0;

// ---------------------------------------------------------------------------
// Poisoning: proving the validator rejects something
// ---------------------------------------------------------------------------

/// A deliberately broken copy of a golden fixture, and why it is broken.
///
/// Used only in tests, and only in memory. The list is the specification of
/// "emptied, truncated, or replaced with something self-consistent but wrong",
/// which is the failure the brief for this crate names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Poison {
    /// Every array emptied. A truncated pipeline run writes this.
    Emptied,
    /// The arrays halved, with the counts in the header left alone.
    Truncated,
    /// The arrays halved *and* the header counts adjusted, so the file is
    /// internally consistent and still wrong. This is the one a
    /// count-against-the-file check cannot catch.
    SelfConsistentTruncation,
    /// Every time shifted two seconds later, header included. Internally
    /// consistent, ordered, covering nothing.
    Shifted,
    /// Every real cell set to the same value, header included.
    Flattened,
    /// Every cell, header included, offset by a constant. This is the classic
    /// "regenerated with a slightly different normalisation" rewrite.
    Offset,
}

/// Apply a [`Poison`] to a copy of the decode fixture.
#[must_use]
pub fn poison_decode(g: &DecodeGolden, poison: Poison) -> DecodeGolden {
    let mut out = g.clone();
    match poison {
        Poison::Emptied => {
            out.token_ids.clear();
            out.words.clear();
            out.token_frames.clear();
        }
        Poison::Truncated => {
            out.token_ids.truncate(EXPECTED_TOKEN_IDS / 2);
            out.words.truncate(EXPECTED_WORDS / 2);
            out.token_frames.truncate(EXPECTED_TOKEN_IDS / 2);
        }
        Poison::SelfConsistentTruncation => {
            out.token_ids.truncate(EXPECTED_TOKEN_IDS / 2);
            out.words.truncate(EXPECTED_WORDS / 2);
            out.token_frames.truncate(EXPECTED_TOKEN_IDS / 2);
            out.seconds = EXPECTED_SECONDS / 2.0;
            out.encoder_frames = EXPECTED_ENCODER_FRAMES / 2;
        }
        Poison::Shifted => {
            for w in &mut out.words {
                w.start += 2.0;
                w.end += 2.0;
            }
        }
        // A decode has no notion of a flattened feature matrix or a constant
        // offset, so those two poisons only mean something to the front end.
        Poison::Flattened | Poison::Offset => {}
    }
    out
}

/// Apply a [`Poison`] to a copy of the front-end fixture.
#[must_use]
pub fn poison_frontend(g: &FrontendGolden, poison: Poison) -> FrontendGolden {
    let mut out = g.clone();
    match poison {
        Poison::Emptied => {
            out.log_mel.clear();
            out.signal.clear();
            out.n_samples = 0;
        }
        Poison::Truncated => {
            for row in &mut out.log_mel {
                row.truncate(EXPECTED_LOGMEL_FRAMES / 2);
            }
        }
        Poison::SelfConsistentTruncation => {
            for row in &mut out.log_mel {
                row.truncate(EXPECTED_LOGMEL_FRAMES / 2);
            }
            out.n_frames = EXPECTED_LOGMEL_FRAMES / 2;
            out.valid_frames = EXPECTED_VALID_FRAMES / 2;
        }
        Poison::Flattened => {
            let first = out
                .log_mel
                .first()
                .and_then(|r| r.first().copied())
                .unwrap_or(0.0);
            for row in &mut out.log_mel {
                for v in row.iter_mut() {
                    *v = first;
                }
            }
        }
        Poison::Offset => {
            for row in &mut out.log_mel {
                for v in row.iter_mut() {
                    *v += 0.5;
                }
            }
        }
        Poison::Shifted => {}
    }
    out
}

/// Every [`Poison`], for tests that iterate.
pub const ALL_POISONS: [Poison; 6] = [
    Poison::Emptied,
    Poison::Truncated,
    Poison::SelfConsistentTruncation,
    Poison::Shifted,
    Poison::Flattened,
    Poison::Offset,
];

#[cfg(test)]
mod tests {
    use super::*;

    fn decode() -> Option<DecodeGolden> {
        load(&decode_golden_path()).ok()
    }

    fn frontend() -> Option<FrontendGolden> {
        load(&frontend_golden_path()).ok()
    }

    #[test]
    fn the_committed_decode_fixture_is_the_reference_decode() {
        let Some(g) = decode() else {
            panic!("{} is absent", decode_golden_path().display());
        };
        assert_eq!(
            validate_decode_golden(&g),
            Vec::<String>::new(),
            "the committed fixture no longer matches the values pinned in src/golden.rs"
        );
    }

    #[test]
    fn the_committed_frontend_fixture_is_the_reference_frontend() {
        let Some(g) = frontend() else {
            panic!("{} is absent", frontend_golden_path().display());
        };
        assert_eq!(
            validate_frontend_golden(&g),
            Vec::<String>::new(),
            "the committed fixture no longer matches the values pinned in src/golden.rs"
        );
    }

    #[test]
    fn every_poison_is_rejected_by_the_validator_alone() {
        // The digest would catch all of these too. This test exists to prove the
        // *validator* has teeth on its own, so that the structural guards are
        // known to work rather than known to be backed up by a checksum.
        let Some(d) = decode() else {
            panic!("decode_golden.json is absent");
        };
        let Some(f) = frontend() else {
            panic!("frontend_golden.json is absent");
        };
        for poison in ALL_POISONS {
            let poisoned = poison_decode(&d, poison);
            let problems = validate_decode_golden(&poisoned);
            // `Flattened` and `Offset` are front-end concepts: a decode has no
            // feature matrix to flatten and no normalisation to shift, so applying
            // them to the decode fixture would be testing nothing.
            if !matches!(poison, Poison::Flattened | Poison::Offset) {
                assert!(
                    !problems.is_empty(),
                    "{poison:?} survived the decode validator unchanged"
                );
            }
            let poisoned = poison_frontend(&f, poison);
            let problems = validate_frontend_golden(&poisoned);
            // `Shifted` moves word timings, which the front-end fixture does not
            // have, so it is a decode-only poison.
            if !matches!(poison, Poison::Shifted) {
                assert!(
                    !problems.is_empty(),
                    "{poison:?} survived the front-end validator unchanged"
                );
            }
        }
    }

    #[test]
    fn the_pinned_digests_match_the_committed_files() {
        for (path, want) in [
            (decode_golden_path(), DECODE_GOLDEN_SHA256),
            (frontend_golden_path(), FRONTEND_GOLDEN_SHA256),
        ] {
            let got = file_sha256(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            assert_eq!(
                got,
                want,
                "{} changed. A golden file may only change with a deliberate re-export from \
                 the reference implementation, and the digest in src/golden.rs must change \
                 with it.",
                path.display()
            );
        }
    }

    #[test]
    fn the_log_mel_tolerance_and_the_recorded_deviation_agree() {
        // These are relationships between two constants, so they are checked at
        // compile time rather than at run time. A tolerance that is 100x the
        // real deviation, or within 50% of it, is a *gate that measures nothing*
        // or a gate that fails on another machine, and neither is worth
        // discovering in CI.
        //
        // * 2e-4 is 2.5x the recorded 8e-5, which is the right shape: tight
        //   enough that every convention error found while writing the front end
        //   (a left-aligned window, a missing Bessel correction) misses it by
        //   three to four orders of magnitude, loose enough to absorb a
        //   different summation order in rustfft.
        // * Below 1.5x, a different FFT's summation order would report a
        //   spurious failure on a machine that is not this one.
        const { assert!(LOGMEL_REFERENCE_DEVIATION <= LOGMEL_TOLERANCE) };
        const { assert!(LOGMEL_TOLERANCE <= LOGMEL_REFERENCE_DEVIATION * 100.0) };
        const { assert!(LOGMEL_TOLERANCE >= LOGMEL_REFERENCE_DEVIATION * 1.5) };
        println!(
            "log-mel tolerance {LOGMEL_TOLERANCE:e} is {:.1}x the recorded deviation \
             {LOGMEL_REFERENCE_DEVIATION:e}",
            LOGMEL_TOLERANCE / LOGMEL_REFERENCE_DEVIATION
        );
    }
}

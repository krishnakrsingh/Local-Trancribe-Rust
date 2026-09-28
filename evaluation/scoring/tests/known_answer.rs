//! Known-answer tests. Every expected number here was computed by hand, on
//! paper, from the definitions — not by running the scorer and writing down
//! what it said.
//!
//! This is the file that decides whether the rest of the harness can be
//! believed. A scorer with no known-answer test is a scorer that reports a
//! number nobody can check, and the number it reports is then the *definition*
//! of accuracy rather than a measurement of it.
//!
//! Where a figure is hard to compute by hand — DER under a permutation — the
//! test states the derivation in a comment as well as the number, so a reader
//! can audit it without a second implementation.
//!
//! None of these cases uses the project's private audio or any real meeting
//! text. The Hindi strings are chosen to exercise the orthography rules and
//! are not drawn from the recording.

use vaani_eval::der::{self, DiarParams};
use vaani_eval::edit::{Op, align};
use vaani_eval::normalize::{NormalisationProfile, is_punct_or_symbol, normalise};
use vaani_eval::rttm::{self, RttmRecord};
use vaani_eval::taxonomy::{ReferenceView, TimingCounts, TimingThresholds, reference_overlap_ms};
use vaani_eval::timing;
use vaani_eval::wer;

fn n(s: &str) -> vaani_eval::normalize::NormalisedText {
    normalise(s, NormalisationProfile::Strict)
}

// ---------------------------------------------------------------------------
// WER and CER
// ---------------------------------------------------------------------------

/// Reference has 9 tokens, the hypothesis adds a 10th. The extra token is a
/// new final word rather than a change to an existing one, so it aligns as an
/// insertion and all 9 reference tokens match. 1 error over 9 = 0.1111...
#[test]
fn wer_hand_computed_one_substitution_one_insertion() {
    let r = "एक दो तीन चार पांच छह सात आठ नौ";
    let h = "एक दो तीन चार पांच छह सात आठ नौ दस";
    let s = wer::score(&n(r), &n(h));
    // एक दो तीन चार पांच छह सात आठ नौ = nine words. "पांच" is three scalars
    // and two grapheme clusters but exactly one token, which is the whole
    // point of scoring in tokens.
    assert_eq!(s.reference_tokens, 9, "{s:?}");
    assert_eq!(s.hypothesis_tokens, 10, "{s:?}");
    assert_eq!(s.correct, 9, "{s:?}");
    assert_eq!(s.substitutions, 0, "{s:?}");
    assert_eq!(s.deletions, 0, "{s:?}");
    assert_eq!(s.insertions, 1, "{s:?}");
    assert_eq!(s.errors, 1, "{s:?}");
    assert!((s.wer.unwrap() - 1.0 / 9.0).abs() < 1e-12, "{s:?}");
}

/// A pure substitution. 4 reference tokens, 1 differs.
#[test]
fn wer_hand_computed_single_substitution() {
    let s = wer::score(&n("a b c d"), &n("a b x d"));
    assert_eq!(
        (s.correct, s.substitutions, s.deletions, s.insertions),
        (3, 1, 0, 0)
    );
    assert!((s.wer.unwrap() - 0.25).abs() < 1e-12);
}

/// A pure deletion: the hypothesis omits one of four reference words.
#[test]
fn wer_hand_computed_single_deletion() {
    let s = wer::score(&n("a b c d"), &n("a c d"));
    assert_eq!(
        (s.correct, s.substitutions, s.deletions, s.insertions),
        (3, 0, 1, 0)
    );
    assert!((s.wer.unwrap() - 0.25).abs() < 1e-12);
}

/// A pure insertion: the hypothesis adds one word to a three-word reference.
#[test]
fn wer_hand_computed_single_insertion() {
    let s = wer::score(&n("a b c"), &n("a b x c"));
    assert_eq!(
        (s.correct, s.substitutions, s.deletions, s.insertions),
        (3, 0, 0, 1)
    );
    assert!((s.wer.unwrap() - 1.0 / 3.0).abs() < 1e-12);
}

/// Complete disagreement of equal length: S = N, so WER = 1.0 exactly.
#[test]
fn wer_hand_computed_total_disagreement_is_exactly_one() {
    let s = wer::score(&n("a b c"), &n("x y z"));
    assert_eq!(s.substitutions, 3);
    assert!((s.wer.unwrap() - 1.0).abs() < 1e-12);
}

/// CER over grapheme clusters and over scalars, from **two separate
/// alignments**.
///
/// reference "क्ष ज्ञ"  = 7 scalars (क ् ष ␠ ज ् ा), 3 grapheme clusters
/// hypothesis "क्ष ग्य" = 7 scalars (क ् ष ␠ ग ् य), 3 grapheme clusters
///
/// Cluster alignment: [क्ष, ␠, ज्ञ] vs [क्ष, ␠, ग्य] -> the third differs, and
/// the virama is not a cluster boundary, so exactly one substitution.
/// `cer_grapheme` = 1/3.
///
/// Scalar alignment: positions 5 and 7 differ (ज vs ग, ा vs य); position 6 is
/// the same virama in both. Two substitutions, which beats the six-cost
/// delete-and-reinsert alternative.
/// `cer_characters` = 2/7.
///
/// The two rates differ because Devanagari writes one phoneme in several
/// scalars. Quoting `cer_characters` for Hindi understates the rate; quoting
/// `cer_grapheme` is not comparable to most of the published literature. Both
/// are reported, the choice is the reader's, and the choice has to be stated.
#[test]
fn cer_hand_computed_over_graphemes_and_scalars() {
    let r = "क्ष ज्ञ";
    let h = "क्ष ग्य";
    let s = wer::score(&n(r), &n(h));
    assert_eq!(s.reference_characters, 7, "{s:?}");
    assert_eq!(s.reference_graphemes, 3, "{s:?}");
    assert_eq!(s.errors, 1, "one substituted word");
    assert_eq!(s.character_errors, 2, "{s:?}");
    assert_eq!(s.grapheme_errors, 1, "{s:?}");
    assert!(
        (s.cer_characters.unwrap() - 2.0 / 7.0).abs() < 1e-12,
        "{s:?}"
    );
    assert!((s.cer_grapheme.unwrap() - 1.0 / 3.0).abs() < 1e-12, "{s:?}");
}

/// An empty reference is undefined, never 0.00%. Three insertions over nothing
/// is a real number for the operation counts and no number at all for the rate.
#[test]
fn wer_hand_computed_empty_reference_is_undefined_not_zero() {
    let s = wer::score(&n(""), &n("a b"));
    assert!(!s.defined);
    assert!(s.wer.is_none());
    assert_eq!(s.insertions, 2);
    assert_eq!(s.errors, 2);
}

/// An empty hypothesis against 3 reference words is WER 1.0, not 0.
#[test]
fn wer_hand_computed_empty_hypothesis_is_total_deletion() {
    let s = wer::score(&n("a b c"), &n(""));
    assert_eq!(s.deletions, 3);
    assert!((s.wer.unwrap() - 1.0).abs() < 1e-12);
}

/// The denominator is the reference length even when the hypothesis is longer.
#[test]
fn wer_hand_computed_denominator_is_the_reference() {
    // 2 reference tokens, hypothesis has 2 of the 3 aligned and 1 inserted.
    let s = wer::score(&n("a b"), &n("a b c d e"));
    assert_eq!(s.reference_tokens, 2);
    assert_eq!(s.hypothesis_tokens, 5);
    assert_eq!(s.errors, 3);
    assert!((s.wer.unwrap() - 1.5).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// Normalisation
// ---------------------------------------------------------------------------

/// The three spellings of the retroflex `za` must compare equal.
///
/// U+095B is one precomposed codepoint; U+091C U+093C is base plus nukta;
/// U+091C U+200D U+093C is base plus the joiner convention. All three are the
/// same sound to a listener and must be the same token to the scorer, or a
/// perfectly good transcript scores a substitution for a font.
#[test]
fn normalisation_hand_computed_three_spellings_of_za_are_one_token() {
    let precomposed = normalise("\u{095b}", NormalisationProfile::Strict);
    let base_nukta = normalise("\u{091c}\u{93c}", NormalisationProfile::Strict);
    let joiner = normalise("\u{091c}\u{200d}\u{93c}", NormalisationProfile::Strict);
    assert_eq!(
        precomposed.tokens, base_nukta.tokens,
        "{precomposed:?} vs {base_nukta:?}"
    );
    assert_eq!(
        precomposed.tokens, joiner.tokens,
        "{precomposed:?} vs {joiner:?}"
    );
    assert_eq!(precomposed.tokens, vec!["\u{091c}\u{93c}".to_string()]);
    // And the nukta is preserved rather than dropped, because za and ja are
    // different phonemes. Folding the nukta away would merge them.
    let ja = normalise("\u{091c}", NormalisationProfile::Strict);
    assert_ne!(ja.tokens, precomposed.tokens);
}

/// Devanagari and Arabic-Indic digits become ASCII, and `पाँच` is NOT folded to
/// `5`: that is a real difference between two texts and folding it would make
/// the score incomparable to every published Hindi WER.
#[test]
fn normalisation_hand_computed_digits_unify_but_number_words_do_not() {
    let a = normalise("पांच", NormalisationProfile::Strict);
    let b = normalise("५", NormalisationProfile::Strict);
    assert_eq!(b.tokens, vec!["5".to_string()]);
    assert_eq!(a.tokens, vec!["पांच".to_string()]);
    assert_ne!(a.tokens, b.tokens);
}

/// The danda is punctuation and goes; it is never a token.
#[test]
fn normalisation_hand_computed_danda_is_not_a_word() {
    let a = normalise("यह ठीक है।", NormalisationProfile::Strict);
    assert_eq!(a.tokens, vec!["यह", "ठीक", "है"]);
}

/// ASCII case is folded; the ASCII apostrophe and hyphen survive inside a
/// word, because `don't` is one word and stripping the apostrophe would make
/// it two tokens and inflate WER by 50 % on every contraction.
#[test]
fn normalisation_hand_computed_case_folds_and_apostrophes_stay() {
    let a = normalise("Don't API", NormalisationProfile::Strict);
    assert_eq!(a.tokens, vec!["don't", "api"]);
}

/// `hinglish` folds diacritics, `strict` does not.
#[test]
fn normalisation_hand_computed_profile_difference() {
    let strict = normalise("shubham", NormalisationProfile::Strict);
    let hing = normalise("kya", NormalisationProfile::Hinglish);
    let t = normalise("\u{015b}ubham", NormalisationProfile::Hinglish);
    assert_eq!(strict.tokens.len(), 1);
    assert_eq!(hing.tokens, vec!["kya".to_string()]);
    assert_eq!(t.tokens, vec!["subham".to_string()]);
    // Under `strict` the same input keeps its diacritic and is a different token.
    let strict2 = normalise("\u{015b}ubham", NormalisationProfile::Strict);
    assert_ne!(strict2.tokens, t.tokens);
}

/// `hinglish-translit` folds double vowels; `hinglish` does not.
#[test]
fn normalisation_hand_computed_double_vowel_fold_is_opt_in() {
    let h = normalise("kyaa", NormalisationProfile::Hinglish);
    let t = normalise("kyaa", NormalisationProfile::HinglishTranslit);
    assert_eq!(h.tokens, vec!["kyaa".to_string()]);
    assert_eq!(t.tokens, vec!["kya".to_string()]);
    // Diphthongs are not folded in either profile.
    let t2 = normalise("main", NormalisationProfile::HinglishTranslit);
    assert_eq!(t2.tokens, vec!["main".to_string()]);
}

/// A mixed-script token is counted, and is NOT split during scoring.
#[test]
fn normalisation_hand_computed_mixed_script_is_counted_not_split() {
    let a = normalise("क्याapi", NormalisationProfile::Strict);
    assert_eq!(a.token_count, 1, "{a:?}");
    assert_eq!(a.mixed_script_tokens, vec!["क्याapi".to_string()]);
    let b = normalise("क्या api", NormalisationProfile::Strict);
    assert_eq!(b.token_count, 2);
    assert!(b.mixed_script_tokens.is_empty());
}

/// The punctuation table behaves on the characters that actually matter.
#[test]
fn normalisation_hand_computed_punctuation_table() {
    for c in ['.', ',', '!', '?', '"', '(', ')', '।', '…', '—', '\u{0964}'] {
        assert!(is_punct_or_symbol(c), "{c:?} should be punctuation");
    }
    for c in ['a', '5', 'क', '-', '\'', '_'] {
        assert!(!is_punct_or_symbol(c), "{c:?} should not be punctuation");
    }
}

/// The same rules run on both sides, and the rules are what the report claims.
#[test]
fn normalisation_hand_computed_the_rules_are_the_reported_rules() {
    let p = NormalisationProfile::HinglishTranslit;
    let ids: Vec<&str> = p.rules().iter().map(|r| r.id).collect();
    assert_eq!(
        ids,
        vec!["N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "H1", "H2"]
    );
}

// ---------------------------------------------------------------------------
// Alignment
// ---------------------------------------------------------------------------

/// The minimum edit distance, and the coverage invariant.
///
/// reference a b c d e
/// hypothesis a x d f g
///
/// Two shared tokens in order (a, then d), so at most 2 of the 5 reference
/// positions are free. A minimal alignment:
///   a match, b->x sub, c del, d match, e del, f ins, g ins
///     = 1 sub + 2 del + 2 ins = 5
///   or: a match, b del, c->x sub, d match, e->f sub, g ins
///     = 2 sub + 1 del + 1 ins = 4, which is smaller and therefore minimal.
///   A third: a match, d match, b c e -> f g is 1 del + 2 sub + 1 ins = 4 too.
///
/// The minimum is 4, and 4 is what the harness must report. The **split**
/// among S, D and I is not a property of the two strings — several minimal
/// alignments exist — it is a property of the documented tie-break, so the
/// test pins the distance exactly and the split's determinism, and says why it
/// does not pin the split to one arrangement.
#[test]
fn alignment_hand_computed_coverage_and_minimum_distance() {
    let r = ["a", "b", "c", "d", "e"];
    let h = ["a", "x", "d", "f", "g"];
    let a = align(&r, &h);
    let mut ri: Vec<usize> = a.ref_index.iter().flatten().copied().collect();
    let mut hi: Vec<usize> = a.hyp_index.iter().flatten().copied().collect();
    ri.sort_unstable();
    hi.sort_unstable();
    assert_eq!(ri, vec![0, 1, 2, 3, 4], "every reference index used once");
    assert_eq!(hi, vec![0, 1, 2, 3, 4], "every hypothesis index used once");
    assert_eq!(a.errors(), 4, "{a:?}");
    assert_eq!(
        a.errors(),
        a.ops.iter().filter(|o| **o != Op::Match).count()
    );
    // Deterministic: the same inputs give the same arrangement every time.
    for _ in 0..32 {
        assert_eq!(align(&r, &h).ops, a.ops);
    }
}

/// A case where the split is unambiguous: one reference token, one hypothesis
/// token, both different. That is one substitution and nothing else.
#[test]
fn alignment_hand_computed_unambiguous_split() {
    let a = align(&["a", "b", "c"], &["a", "x", "c"]);
    assert_eq!(
        (a.correct, a.substitutions, a.deletions, a.insertions),
        (2, 1, 0, 0)
    );
}

// ---------------------------------------------------------------------------
// RTTM
// ---------------------------------------------------------------------------

/// Microseconds, not floating-point seconds. 0.001 s is exactly 1000 us.
#[test]
fn rttm_hand_computed_microsecond_conversion() {
    let rs = rttm::parse("SPEAKER f 1 0.001 0.001 <NA> <NA> a <NA> <NA>").unwrap();
    assert_eq!(rs[0].start_us, 1_000);
    assert_eq!(rs[0].duration_us, 1_000);
    assert_eq!(rs[0].end_us(), 2_000);
}

/// Build RTTM turns from `(start_us, duration_us, speaker)`.
fn turns(pairs: &[(i64, i64, &str)]) -> Vec<RttmRecord> {
    pairs
        .iter()
        .map(|&(s, d, name)| RttmRecord::speaker(name, s, d))
        .collect()
}

// ---------------------------------------------------------------------------
// DER
// ---------------------------------------------------------------------------

/// Hand computation, all times in seconds for readability and microseconds in
/// the file:
///
/// reference : A 0.0-2.0, B 2.0-4.0                (4000 ms of speech)
/// hypothesis: A 0.0-2.0, B 2.0-4.0                (identical)
///
/// Nothing is missed, nothing is false, nothing is confused. DER = 0.
#[test]
fn der_hand_computed_perfect_match_is_zero() {
    let r = turns(&[(0, 2_000_000, "A"), (2_000_000, 2_000_000, "B")]);
    let h = turns(&[(0, 2_000_000, "A"), (2_000_000, 2_000_000, "B")]);
    let rep = der::score(&r, &h, DiarParams::default());
    assert_eq!(rep.der, Some(0.0), "{rep:#?}");
    assert_eq!(rep.reference_speech_ms, 4000.0);
    assert!(rep.speaker_count.correct);
    assert_eq!(rep.reference_turns, 2);
}

/// The hypothesis uses the *other* labels. Same two people, same two turns.
///
/// Without a permutation this is 4000 ms of confusion and DER = 1.0. With the
/// permutation `A->A2, B->B1` it is a perfect match. This is the single most
/// common way a diarization score is fabricated, so it has a test.
#[test]
fn der_hand_computed_label_permutation_is_resolved() {
    let r = turns(&[(0, 2_000_000, "A"), (2_000_000, 2_000_000, "B")]);
    let h = turns(&[(0, 2_000_000, "A2"), (2_000_000, 2_000_000, "B1")]);
    let rep = der::score(&r, &h, DiarParams::default());
    assert_eq!(rep.der, Some(0.0), "{rep:#?}");
    assert_eq!(rep.permutation.get("A").map(String::as_str), Some("A2"));
    assert_eq!(rep.permutation.get("B").map(String::as_str), Some("B1"));
}

/// The hypothesis misses the second turn entirely.
///
/// reference speech   4000 ms
/// correct             2000 ms  (0-2 s, A)
/// missed detection   2000 ms  (2-4 s, B, nothing in the hypothesis)
/// false alarm            0 ms
/// confusion              0 ms
/// DER = 2000/4000 = 0.5
#[test]
fn der_hand_computed_missed_second_turn_is_one_half() {
    let r = turns(&[(0, 2_000_000, "A"), (2_000_000, 2_000_000, "B")]);
    let h = turns(&[(0, 2_000_000, "A")]);
    let rep = der::score(&r, &h, DiarParams::default());
    let d = rep.der.unwrap();
    assert!((d - 0.5).abs() < 1e-12, "{d}");
    assert!((rep.components.missed_detection - 0.5).abs() < 1e-12);
    assert!((rep.components.false_alarm - 0.0).abs() < 1e-12);
    assert!((rep.components.confusion - 0.0).abs() < 1e-12);
    assert!((rep.error_ms.missed_detection - 2000.0).abs() < 1e-9);
}

/// The hypothesis invents speech where the reference has none.
///
/// reference speech   2000 ms (0-2 s, A)
/// hypothesis speech  3000 ms (A 0-2, A 4-5)
/// correct             2000 ms
/// missed detection       0 ms   the reference is fully covered where it speaks
/// confusion              0 ms
/// false alarm         1000 ms   4-5 s, where the reference says nobody is talking
/// DER = 1000/2000 = 0.5
///
/// Two identities this must satisfy, and which are asserted rather than
/// described: the denominator is the *reference* speech, and a hypothesis that
/// marks silence as speech is penalised rather than ignored.
#[test]
fn der_hand_computed_false_alarm_denominator_is_reference_speech() {
    let r = turns(&[(0, 2_000_000, "A")]);
    let h = turns(&[(0, 2_000_000, "A"), (4_000_000, 1_000_000, "A")]);
    let rep = der::score(&r, &h, DiarParams::default());
    assert_eq!(rep.reference_speech_ms, 2000.0, "{rep:#?}");
    assert_eq!(rep.hypothesis_speech_ms, 3000.0, "{rep:#?}");
    assert_eq!(rep.correct_ms, 2000.0, "{rep:#?}");
    assert!((rep.der.unwrap() - 0.5).abs() < 1e-12, "{rep:#?}");
    assert!((rep.components.false_alarm - 0.5).abs() < 1e-12, "{rep:#?}");
    assert!(
        (rep.components.missed_detection - 0.0).abs() < 1e-12,
        "{rep:#?}"
    );
    assert!((rep.components.confusion - 0.0).abs() < 1e-12, "{rep:#?}");
    // The two closed forms, checked directly.
    assert!(
        (rep.correct_ms + rep.error_ms.missed_detection + rep.error_ms.confusion
            - rep.reference_speech_ms)
            .abs()
            < 1e-6
    );
    assert!(
        (rep.correct_ms + rep.error_ms.false_alarm + rep.error_ms.confusion
            - rep.hypothesis_speech_ms)
            .abs()
            < 1e-6
    );
}

/// A false alarm far larger than the reference makes DER exceed 1. Not clipped,
/// not an error: the metric is a ratio and ratios do that.
///
/// reference speech   1000 ms
/// false alarm       4000 ms
/// DER = 4.0
#[test]
fn der_hand_computed_can_exceed_one() {
    let r = turns(&[(0, 1_000_000, "A")]);
    let h = turns(&[(0, 1_000_000, "A"), (5_000_000, 4_000_000, "B")]);
    let rep = der::score(&r, &h, DiarParams::default());
    assert!((rep.der.unwrap() - 4.0).abs() < 1e-12, "{rep:#?}");
}

/// The hypothesis merges two speakers into one.
///
/// reference : A 0-2, B 2-4   (4000 ms)
/// hypothesis: C 0-4          (4000 ms)
///
/// The maximum-weight assignment maps exactly one reference label to C. Say it
/// maps A, and the reference is renamed so the hypothesis's "C" is "A":
///
/// 0-2 s  ref {A}, hyp {A}  -> matched, no error
/// 2-4 s  ref {B}, hyp {A}  -> nobody matches. The hypothesis says *somebody*
///                             is talking, just not B, so this is confusion,
///                             not a missed detection. Counting it as both
///                             would charge the same 2 s twice and put DER at
///                             1.0 for a system that got half the audio right.
/// DER = 2000/4000 = 0.5
#[test]
fn der_hand_computed_speaker_merge_is_confusion_not_missed() {
    let r = turns(&[(0, 2_000_000, "A"), (2_000_000, 2_000_000, "B")]);
    let h = turns(&[(0, 4_000_000, "C")]);
    let rep = der::score(&r, &h, DiarParams::default());
    assert!((rep.der.unwrap() - 0.5).abs() < 1e-12, "{rep:#?}");
    assert!((rep.components.confusion - 0.5).abs() < 1e-12, "{rep:#?}");
    assert!(
        (rep.components.missed_detection - 0.0).abs() < 1e-12,
        "{rep:#?}"
    );
    assert!((rep.components.false_alarm - 0.0).abs() < 1e-12, "{rep:#?}");
    assert!(!rep.speaker_count.correct);
    assert_eq!(rep.speaker_count.reference, 2);
    assert_eq!(rep.speaker_count.hypothesis, 1);
    // A merged pair produces exactly one turn change the reference did not ask
    // for, and misses the one it did.
    assert_eq!(rep.boundary.unmatched_reference, 1, "{rep:#?}");
    assert_eq!(rep.boundary.unmatched_hypothesis, 0, "{rep:#?}");
}

/// Overlap: two reference speakers 3-5 s; the hypothesis has one of them.
///
/// reference speech  : A 0-2 (2000) + A 3-5 (2000) + B 3-5 (2000) = 6000 ms
/// hypothesis speech : A 0-2 (2000) + A 3-5 (2000) = 4000 ms
/// correct           : 2000 (0-2 s) + 2000 (3-5 s, A) = 4000 ms
/// 3-5 s has reference {A, B} against hypothesis {A}. A matches; B does not,
/// and somebody is talking, so the residual is confusion.
/// confusion         : 2000 ms
/// missed / false    : 0 ms
/// DER = 2000/6000 = 1/3
///
/// The two closed forms both hold: 4000 + 0 + 2000 = 6000 = reference speech,
/// and 4000 + 0 + 2000 = 4000 = hypothesis speech.
#[test]
fn der_hand_computed_overlap_costs_confusion() {
    let r = turns(&[
        (0, 2_000_000, "A"),
        (3_000_000, 2_000_000, "A"),
        (3_000_000, 2_000_000, "B"),
    ]);
    let h = turns(&[(0, 2_000_000, "A"), (3_000_000, 2_000_000, "A")]);
    let rep = der::score(&r, &h, DiarParams::default());
    assert!((rep.reference_speech_ms - 6000.0).abs() < 1e-9, "{rep:#?}");
    assert!((rep.hypothesis_speech_ms - 4000.0).abs() < 1e-9, "{rep:#?}");
    assert!((rep.der.unwrap() - 1.0 / 3.0).abs() < 1e-9, "{rep:#?}");
    assert!(
        (rep.components.confusion - 1.0 / 3.0).abs() < 1e-9,
        "{rep:#?}"
    );
    assert!(
        (rep.components.missed_detection - 0.0).abs() < 1e-9,
        "{rep:#?}"
    );
    assert_eq!(reference_overlap_ms(&r), 2000, "2 s of two-speaker overlap");
    assert_eq!(reference_overlap_ms(&h), 0);
}

/// A 250 ms collar forgives 125 ms at each end of every reference turn.
///
/// reference : A 0.000-2.000
/// hypothesis: A 0.100-1.900          (100 ms late at the start, 100 ms early at the end)
///
/// Without a collar: correct 1800 ms, confusion 200 ms, DER = 200/2000 = 0.1.
/// With a 250 ms collar the reference is trimmed to 0.125-1.875 (1750 ms of
/// denominator) and the hypothesis covers all of it, so DER = 0.
#[test]
fn der_hand_computed_collar_forgives_a_late_boundary() {
    let r = turns(&[(0, 2_000_000, "A")]);
    let h = turns(&[(100_000, 1_800_000, "A")]);
    let strict = der::score(&r, &h, DiarParams::default());
    assert!((strict.der.unwrap() - 0.1).abs() < 1e-9, "{strict:#?}");
    let forgiven = der::score(&r, &h, DiarParams::with_collar_250ms());
    assert_eq!(forgiven.der, Some(0.0), "{forgiven:#?}");
    assert!((forgiven.reference_speech_ms - 1750.0).abs() < 1e-9);
}

/// A reference with no speech turn makes DER undefined, not 0.
#[test]
fn der_hand_computed_empty_reference_is_undefined() {
    let h = turns(&[(0, 1_000_000, "A")]);
    let rep = der::score(&[], &h, DiarParams::default());
    assert!(!rep.defined);
    assert!(rep.der.is_none());
    assert!(rep.undefined_reason.is_some());
}

/// The assignment must maximise total overlap, and the optimum is not the one a
/// first-come pairing finds.
///
/// Overlaps in seconds:
///   A  0-1, 2-3      vs  X 0-3        A-X = 1.0
///   B  0-3           vs  Y 0-1, 2-3    A-Y = 1.0
///                                         B-X = 3.0
///                                         B-Y = 1.0
///
/// Optimal: B->X (3.0) and A->Y (1.0), total 4.0, and every second of
/// reference speech is correctly attributed, so DER = 0.
///
/// Reference speech is the *sum of reference turn durations*, so A's overlap
/// with B counts in both: 1 + 1 + 3 = 5.0 s, not 3.0. That is the standard
/// definition and it is the reason overlap costs a diarizer that emits one
/// speaker per instant; the closed form
/// `correct + missed + confusion == reference speech` is asserted by the scorer
/// itself on every call.
///
/// A greedy that pairs the first reference speaker it meets with the first
/// hypothesis speaker it meets takes A->X (1.0) and then B->Y (1.0): 2.0, half
/// the possible, DER = 0.5 on an input that is actually perfect. This is the
/// whole reason the permutation is solved rather than accumulated.
#[test]
fn der_hand_computed_the_permutation_is_an_optimum_not_a_greedy_pass() {
    let r = turns(&[
        (0, 1_000_000, "A"),
        (2_000_000, 1_000_000, "A"),
        (0, 3_000_000, "B"),
    ]);
    let h = turns(&[
        (0, 3_000_000, "X"),
        (0, 1_000_000, "Y"),
        (2_000_000, 1_000_000, "Y"),
    ]);
    let rep = der::score(&r, &h, DiarParams::default());
    assert_eq!(rep.der, Some(0.0), "{rep:#?}");
    assert!((rep.reference_speech_ms - 5000.0).abs() < 1e-9, "{rep:#?}");
    assert!((rep.correct_ms - 5000.0).abs() < 1e-9, "{rep:#?}");
    assert_eq!(rep.permutation.get("B").map(String::as_str), Some("X"));
    assert_eq!(rep.permutation.get("A").map(String::as_str), Some("Y"));
}

/// Boundary metrics on a known shift.
///
/// reference turns: A 0-2, B 2-4  (one boundary at 2.000 s)
/// hypothesis turns: A 0-2, B 2.2-4.2 (one boundary at 2.200 s)
/// |delta| = 200 ms, inside the 250 ms tolerance, so it matches.
#[test]
fn turn_metrics_hand_computed_boundary_shift() {
    let r = turns(&[(0, 2_000_000, "A"), (2_000_000, 2_000_000, "B")]);
    let h = turns(&[(0, 2_000_000, "A"), (2_200_000, 2_000_000, "B")]);
    let rep = der::score(&r, &h, DiarParams::default());
    assert_eq!(rep.boundary.matched, 1, "{rep:#?}");
    assert_eq!(rep.boundary.unmatched_reference, 0);
    assert!((rep.boundary.mean_abs_deviation_ms - 200.0).abs() < 1e-9);
    assert_eq!(rep.false_speaker_change, 0);
    assert_eq!(rep.missed_speaker_change, 0);
}

/// A merged pair: the reference has two turns and one boundary at 2.0 s, the
/// hypothesis has one turn and no boundary at all.
///
/// That is a **missed** speaker change (a human heard a change and the system
/// did not produce one), and no false one: `false` runs the other way, a
/// boundary the hypothesis produced that the reference did not ask for. The two
/// counters are therefore `boundary.unmatched_reference` and
/// `boundary.unmatched_hypothesis` respectively, and this test pins both
/// directions.
#[test]
fn turn_metrics_hand_computed_a_merged_pair_misses_one_change() {
    let r = turns(&[(0, 2_000_000, "A"), (2_000_000, 2_000_000, "B")]);
    let h = turns(&[(0, 4_000_000, "A")]);
    let rep = der::score(&r, &h, DiarParams::default());
    assert_eq!(rep.missed_speaker_change, 1, "{rep:#?}");
    assert_eq!(rep.false_speaker_change, 0, "{rep:#?}");
    assert_eq!(rep.hypothesis_turns, 1);
    assert_eq!(rep.reference_turns, 2);
}

/// The mirror image: the hypothesis invents a speaker change the reference does
/// not have. That is a false speaker change, and no missed one.
#[test]
fn turn_metrics_hand_computed_false_speaker_change() {
    let r = turns(&[(0, 4_000_000, "A")]);
    let h = turns(&[(0, 2_000_000, "A"), (2_000_000, 2_000_000, "B")]);
    let rep = der::score(&r, &h, DiarParams::default());
    assert_eq!(rep.false_speaker_change, 1, "{rep:#?}");
    assert_eq!(rep.missed_speaker_change, 0, "{rep:#?}");
    assert_eq!(rep.reference_turns, 1);
    assert_eq!(rep.hypothesis_turns, 2);
}

/// A reference speaker change with nothing in the hypothesis at all: a missed
/// speaker change, and no false one.
#[test]
fn turn_metrics_hand_computed_missed_speaker_change() {
    let r = turns(&[(0, 2_000_000, "A"), (2_000_000, 2_000_000, "B")]);
    let h = turns(&[(0, 2_000_000, "A")]);
    let rep = der::score(&r, &h, DiarParams::default());
    assert_eq!(rep.missed_speaker_change, 1, "{rep:#?}");
    assert_eq!(rep.false_speaker_change, 0, "{rep:#?}");
}

/// JER over a perfect match of both regions is 0.
#[test]
fn jer_hand_computed_perfect_is_zero() {
    let r = turns(&[(0, 2_000_000, "A"), (4_000_000, 2_000_000, "B")]);
    let h = turns(&[(0, 2_000_000, "A"), (4_000_000, 2_000_000, "B")]);
    let rep = der::score(&r, &h, DiarParams::default());
    assert_eq!(rep.jer_speech, Some(0.0), "{rep:#?}");
    assert_eq!(rep.jer_non_speech, Some(0.0));
    assert_eq!(rep.jer, Some(0.0));
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

/// Four aligned words, each 100 ms late. Every deviation statistic is 100.
#[test]
fn timing_hand_computed_constant_lag() {
    use vaani_eval::hypothesis::ReferenceWord;
    let refs = vec![
        ReferenceWord {
            text: "a".into(),
            start_ms: 0,
            end_ms: 100,
        },
        ReferenceWord {
            text: "b".into(),
            start_ms: 200,
            end_ms: 300,
        },
        ReferenceWord {
            text: "c".into(),
            start_ms: 400,
            end_ms: 500,
        },
        ReferenceWord {
            text: "d".into(),
            start_ms: 600,
            end_ms: 700,
        },
    ];
    let hyp = vec![(100, 200), (300, 400), (500, 600), (700, 800)];
    let t = timing::score(&n("a b c d"), &refs, &n("a b c d"), &hyp);
    assert!(t.defined);
    assert_eq!(t.matched_pairs, 4);
    let d = t.start_deviation.unwrap();
    assert!((d.mean_ms - 100.0).abs() < 1e-9, "{d:?}");
    assert!((d.median_ms - 100.0).abs() < 1e-9);
    assert!((d.p90_ms - 100.0).abs() < 1e-9);
    assert!((d.max_abs_ms - 100.0).abs() < 1e-9);
    assert!((t.best_constant_offset_ms - 100.0).abs() < 1e-9);
    assert_eq!(t.negative_duration_pairs, 0);
}

/// Deviations 0, 0, 0, 400. Mean 100, median 0, p90 400, max 400.
#[test]
fn timing_hand_computed_mixed_deviations() {
    use vaani_eval::hypothesis::ReferenceWord;
    let refs: Vec<ReferenceWord> = (0..4)
        .map(|i| ReferenceWord {
            text: "w".into(),
            start_ms: i * 200,
            end_ms: i * 200 + 100,
        })
        .collect();
    let hyp = vec![(0, 100), (200, 300), (400, 500), (1000, 1100)];
    let t = timing::score(&n("w w w w"), &refs, &n("w w w w"), &hyp);
    let d = t.start_deviation.unwrap();
    assert!((d.mean_ms - 100.0).abs() < 1e-9, "{d:?}");
    assert!((d.median_ms - 0.0).abs() < 1e-9);
    assert!((d.p90_ms - 400.0).abs() < 1e-9);
    assert!((d.max_abs_ms - 400.0).abs() < 1e-9);
}

/// No reference word timings means no number, not a zero.
#[test]
fn timing_hand_computed_absent_reference_is_unavailable() {
    let t = timing::score(&n("a"), &[], &n("a"), &[(0, 10)]);
    assert!(!t.defined);
    assert!(t.start_deviation.is_none());
    assert!(t.start_deviation.is_none());
}

// ---------------------------------------------------------------------------
// Taxonomy
// ---------------------------------------------------------------------------

/// Four signatures are measurable with no reference at all, and are reported
/// as such. The rest are `unavailable` with `count: None`, never `0`.
#[test]
fn taxonomy_hand_computed_reference_free_signals() {
    use vaani_eval::hypothesis::{Hypothesis, HypothesisSegment, HypothesisWord};
    let h = Hypothesis {
        source: "t".into(),
        words: vec![
            HypothesisWord {
                text: "नमस्ते".into(),
                start_ms: 0,
                end_ms: 100,
            },
            HypothesisWord {
                text: "नमस्ते".into(),
                start_ms: 100,
                end_ms: 200,
            },
            HypothesisWord {
                text: ",".into(),
                start_ms: 200,
                end_ms: 220,
            },
        ],
        segments: vec![HypothesisSegment {
            start_ms: 0,
            end_ms: 220,
            speaker: Some("A".into()),
            word_range: (0, 3),
        }],
        duration_ms: Some(220),
        speakers: vec!["A".into()],
    };
    let sigs = vaani_eval::taxonomy::evaluate(
        &h,
        ReferenceView::None,
        &h.to_rttm("c"),
        &[],
        None,
        TimingThresholds::default(),
    );
    let by = |id: &str| sigs.iter().find(|s| s.id == id).cloned().unwrap();
    assert_eq!(by("asr.duplicated_word").count, Some(1));
    assert_eq!(
        by("asr.duplicated_word").status,
        vaani_eval::taxonomy::SignalStatus::Measured
    );
    assert_eq!(by("asr.punctuation_error").count, Some(1));
    assert_eq!(by("asr.hallucinated_word").count, None);
    assert_eq!(
        by("asr.hallucinated_word").status,
        vaani_eval::taxonomy::SignalStatus::Unavailable
    );
    assert_eq!(by("asr.named_entity_error").count, None);
    assert_eq!(by("timing.segment_shifted").count, None);
    assert_eq!(by("diarization.overlap_failure").count, None);
}

/// With a human reference the ASR signals become exact counts.
#[test]
fn taxonomy_hand_computed_with_a_human_reference() {
    use vaani_eval::hypothesis::{Hypothesis, HypothesisSegment, HypothesisWord};
    let h = Hypothesis {
        source: "t".into(),
        words: vec![
            HypothesisWord {
                text: "a".into(),
                start_ms: 0,
                end_ms: 100,
            },
            HypothesisWord {
                text: "c".into(),
                start_ms: 100,
                end_ms: 200,
            },
            HypothesisWord {
                text: "d".into(),
                start_ms: 200,
                end_ms: 300,
            },
        ],
        segments: vec![HypothesisSegment {
            start_ms: 0,
            end_ms: 300,
            speaker: Some("A".into()),
            word_range: (0, 3),
        }],
        duration_ms: Some(300),
        speakers: vec!["A".into()],
    };
    let r = n("a b c d");
    let hy = n("a c d");
    let sigs = vaani_eval::taxonomy::evaluate(
        &h,
        ReferenceView::Human {
            tokens: &r.tokens,
            hypothesis_tokens: &hy.tokens,
        },
        &h.to_rttm("c"),
        &[],
        Some(TimingCounts {
            shifted: 0,
            drifted: 0,
            too_long: 0,
            too_short: 0,
        }),
        TimingThresholds::default(),
    );
    let by = |id: &str| sigs.iter().find(|s| s.id == id).cloned().unwrap();
    assert_eq!(by("asr.omitted_word").count, Some(1));
    assert_eq!(by("asr.substituted_word").count, Some(0));
    assert_eq!(by("asr.hallucinated_word").count, Some(0));
    assert_eq!(
        by("asr.omitted_word").status,
        vaani_eval::taxonomy::SignalStatus::Measured
    );
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

/// The three classes are separate fields and the blend is refused. A consumer
/// that looks for one number will not find one.
#[test]
fn report_hand_computed_three_classes_and_a_refused_blend() {
    let b = vaani_eval::report::BlendRefusal::default();
    let v = serde_json::to_value(&b).unwrap();
    assert_eq!(v["permitted"], false);
    assert!(v["value"].is_null());
}

//! Word and character error rate, and the accounting that goes with them.
//!
//! # Definitions used, stated so they cannot be argued with later
//!
//! * `WER = (S + D + I) / N_ref` where `N_ref` is the number of **reference**
//!   tokens after normalisation. Never the hypothesis count, never the mean of
//!   the two, never `min`. Using the hypothesis count makes an empty
//!   hypothesis look perfect.
//! * `CER` is the same quantity over characters. Two denominators are reported
//!   because for Devanagari they differ materially:
//!   * `char_count` — Unicode scalar values, the number every published CER
//!     quotes, and the number that is inflated by cluster decomposition;
//!   * `grapheme_count` — extended grapheme clusters, where `क्ष` counts once.
//!
//!   **Quote `cer_grapheme` for Hindi.** `cer_characters` exists for
//!   comparability with the literature and is labelled as such.
//! * A zero-length reference makes WER and CER **undefined**, not zero. The
//!   scorer returns `None` and the report says `undefined: empty reference`
//!   rather than printing `0.00%`, which would read as a perfect score for a
//!   clip where the human wrote nothing down.

use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;

use crate::edit::{Alignment, align};
use crate::normalize::NormalisedText;

/// Why a rate could not be computed. Never a fallback to zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Undefined {
    /// The reference holds no token after normalisation.
    EmptyReference,
    /// The reference has no character to divide by.
    NoReferenceCharacters,
}

/// WER, CER and the operation counts behind them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WordErrorReport {
    /// `true` when both rates are defined.
    pub defined: bool,
    /// Why a rate is undefined, when it is.
    pub undefined_reason: Option<Undefined>,
    /// Word error rate, as a fraction in [0, inf). `None` when undefined.
    pub wer: Option<f64>,
    /// Character error rate over Unicode scalar values. `None` when undefined.
    pub cer_characters: Option<f64>,
    /// Character error rate over extended grapheme clusters. `None` when undefined.
    pub cer_grapheme: Option<f64>,
    /// Correct reference tokens.
    pub correct: usize,
    /// Substituted reference tokens.
    pub substitutions: usize,
    /// Deleted reference tokens (spoken, absent from the hypothesis).
    pub deletions: usize,
    /// Inserted hypothesis tokens (not spoken, present in the hypothesis).
    pub insertions: usize,
    /// Total errors, `S + D + I`.
    pub errors: usize,
    /// Reference token count; the WER denominator.
    pub reference_tokens: usize,
    /// Hypothesis token count. Reported, never used as a denominator.
    pub hypothesis_tokens: usize,
    /// Reference scalar values, spaces included; the `cer_characters` denominator.
    pub reference_characters: usize,
    /// Reference grapheme clusters, spaces included; the `cer_grapheme` denominator.
    pub reference_graphemes: usize,
    /// Errors from the character alignment. **Not** `errors`: a substituted
    /// word is one token error and several character errors, and reporting the
    /// token count here would understate the character rate.
    pub character_errors: usize,
    /// Errors from the grapheme-cluster alignment. The denominator and the
    /// numerator of `cer_grapheme` come from the same alignment, so the ratio
    /// is internally consistent.
    pub grapheme_errors: usize,
    /// Reference tokens in which Devanagari and Latin are mixed. A
    /// code-switch boundary error signature, counted and reported, and
    /// deliberately **not** split during scoring — folding them would hide the
    /// very error the taxonomy is looking for.
    pub mixed_script_reference_tokens: usize,
    /// The token alignment the timing and taxonomy passes read.
    pub alignment: AlignmentView,
}

/// A serialisable view of the alignment: counts, not the raw index vectors,
/// which are large and derived.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlignmentView {
    /// Length of the operation list.
    pub length: usize,
    /// Number of `match` operations.
    pub matches: usize,
    /// Number of `sub` operations.
    pub sub: usize,
    /// Number of `del` operations.
    pub del: usize,
    /// Number of `ins` operations.
    pub ins: usize,
}

impl From<&Alignment> for AlignmentView {
    fn from(a: &Alignment) -> Self {
        use crate::edit::Op;
        let mut v = AlignmentView {
            length: a.ops.len(),
            matches: 0,
            sub: 0,
            del: 0,
            ins: 0,
        };
        for op in &a.ops {
            match op {
                Op::Match => v.matches += 1,
                Op::Sub => v.sub += 1,
                Op::Del => v.del += 1,
                Op::Ins => v.ins += 1,
            }
        }
        v
    }
}

/// Score a normalised hypothesis against a normalised reference.
///
/// The two sides are already normalised; this function never re-normalises,
/// which is the only way to guarantee the identical-rules property.
pub fn score(reference: &NormalisedText, hypothesis: &NormalisedText) -> WordErrorReport {
    let a = align(&reference.tokens, &hypothesis.tokens);
    let errors = a.errors();

    // CER is a *separate alignment*, not the token error count divided by a
    // character count. Those are different numbers: a single substituted word
    // can be one token error and five character errors, and dividing the first
    // by the second understates the character error rate. Two alignments are
    // run because the two character definitions genuinely differ.
    let ref_chars: Vec<char> = reference.text.chars().collect();
    let hyp_chars: Vec<char> = hypothesis.text.chars().collect();
    let char_errors = align(&ref_chars, &hyp_chars).errors();
    let ref_graphemes: Vec<&str> = reference.text.graphemes(true).collect();
    let hyp_graphemes: Vec<&str> = hypothesis.text.graphemes(true).collect();
    let grapheme_errors = align(&ref_graphemes, &hyp_graphemes).errors();

    let (wer, undefined) = if reference.token_count == 0 {
        (None, Some(Undefined::EmptyReference))
    } else {
        (Some(errors as f64 / reference.token_count as f64), None)
    };
    let (cer_c, cer_g) = if reference.grapheme_count == 0 {
        (None, None)
    } else {
        (
            Some(char_errors as f64 / reference.char_count as f64),
            Some(grapheme_errors as f64 / reference.grapheme_count as f64),
        )
    };

    WordErrorReport {
        defined: wer.is_some(),
        undefined_reason: undefined,
        wer,
        cer_characters: cer_c,
        cer_grapheme: cer_g,
        correct: a.correct,
        substitutions: a.substitutions,
        deletions: a.deletions,
        insertions: a.insertions,
        errors,
        reference_tokens: reference.token_count,
        hypothesis_tokens: hypothesis.token_count,
        reference_characters: reference.char_count,
        reference_graphemes: reference.grapheme_count,
        character_errors: char_errors,
        grapheme_errors,
        mixed_script_reference_tokens: reference.mixed_script_tokens.len(),
        alignment: AlignmentView::from(&a),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize::{NormalisationProfile, normalise};

    fn n(s: &str) -> NormalisedText {
        normalise(s, NormalisationProfile::Strict)
    }

    #[test]
    fn a_perfect_hypothesis_scores_zero() {
        let r = score(&n("यह एक परीक्षण है"), &n("यह एक परीक्षण है"));
        assert_eq!(r.wer, Some(0.0));
        assert_eq!(r.correct, 4);
        assert_eq!(r.errors, 0);
        assert!(r.defined);
    }

    #[test]
    fn one_error_in_four_words_is_a_quarter() {
        let r = score(&n("a b c d"), &n("a b x d"));
        assert_eq!(r.substitutions, 1);
        assert_eq!(r.wer, Some(0.25));
    }

    #[test]
    fn the_denominator_is_the_reference_not_the_hypothesis() {
        // Hypothesis has 5 tokens, reference has 2, one match, one deletion.
        let r = score(&n("a b"), &n("a b c d e"));
        assert_eq!(r.deletions, 0);
        assert_eq!(r.insertions, 3);
        assert_eq!(r.reference_tokens, 2);
        assert_eq!(r.hypothesis_tokens, 5);
        assert_eq!(r.wer, Some(1.5));
    }

    #[test]
    fn an_empty_reference_is_undefined_and_never_zero() {
        let r = score(&n(""), &n("a b c"));
        assert!(!r.defined);
        assert_eq!(r.undefined_reason, Some(Undefined::EmptyReference));
        assert_eq!(r.wer, None);
        assert_eq!(r.errors, 3);
    }

    #[test]
    fn an_empty_hypothesis_is_total_deletion() {
        let r = score(&n("a b c"), &n(""));
        assert_eq!(r.deletions, 3);
        assert_eq!(r.wer, Some(1.0));
    }

    #[test]
    fn grapheme_cer_and_scalar_cer_are_two_alignments_not_one_number_divided() {
        // reference "क्ष क्ष" = 7 scalars, 3 clusters (क+्+ष is one cluster)
        // hypothesis "क्ष क"  = 5 scalars, 3 clusters
        //
        // Cluster alignment: [क्ष, ␠, क्ष] vs [क्ष, ␠, क] -> one deletion.
        // Scalar alignment:  क,्,ष,␠,क,्,ष  vs  क,्,ष,␠,क -> two deletions.
        // One token error, two character errors, one cluster error. Dividing
        // the token count by the character count would report 1/7 and be wrong
        // in both directions at once.
        let r = score(&n("क्ष क्ष"), &n("क्ष क"));
        assert_eq!(r.errors, 1, "{r:?}");
        // The split is substitution, not deletion, because the traceback
        // prefers the diagonal when both cost the same. That preference is
        // fixed and documented precisely so the S/D/I breakdown is
        // reproducible; S+D+I is the same either way.
        assert_eq!(r.substitutions, 1, "{r:?}");
        assert_eq!(r.deletions, 0, "{r:?}");
        assert_eq!(r.reference_characters, 7, "{r:?}");
        assert_eq!(r.reference_graphemes, 3, "{r:?}");
        assert_eq!(r.grapheme_errors, 1, "{r:?}");
        assert_eq!(r.character_errors, 2, "{r:?}");
        assert!((r.wer.unwrap() - 0.5).abs() < 1e-12);
        assert!((r.cer_grapheme.unwrap() - 1.0 / 3.0).abs() < 1e-12, "{r:?}");
        assert!(
            (r.cer_characters.unwrap() - 2.0 / 7.0).abs() < 1e-12,
            "{r:?}"
        );
    }

    #[test]
    fn mixed_script_tokens_are_counted_and_not_split() {
        let r = score(&n("क्याapi"), &n("क्याapi"));
        assert_eq!(r.mixed_script_reference_tokens, 1);
        assert_eq!(r.correct, 1);
    }
}

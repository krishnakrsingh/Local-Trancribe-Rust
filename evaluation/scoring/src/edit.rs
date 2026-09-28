//! Levenshtein alignment with an operation traceback.
//!
//! One implementation, used for WER, CER and for the alignment that the
//! taxonomy signals and the timing metrics are read off. There is no second,
//! "fast" path: a scorer whose fast path can disagree with its correct path is
//! a scorer that reports a number nobody can reproduce.
//!
//! Complexity is `O(n*m)` time and `O(n*m)` memory for the traceback. For a
//! 120-second meeting clip that is a few hundred tokens, so the memory is
//! irrelevant; the cap in [`align`] exists so a pathological input is a clear
//! error rather than an out-of-memory abort.

/// One alignment operation, in reference order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// Reference and hypothesis agree.
    Match,
    /// Reference token present, hypothesis token different.
    Sub,
    /// Reference token absent from the hypothesis.
    Del,
    /// Hypothesis token absent from the reference.
    Ins,
}

/// The full alignment of a hypothesis to a reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Alignment {
    /// Operations in order. A [`Op::Sub`] names one reference index and one
    /// hypothesis index; the rest name one.
    pub ops: Vec<Op>,
    /// Reference index for each op, `None` for an insertion.
    pub ref_index: Vec<Option<usize>>,
    /// Hypothesis index for each op, `None` for a deletion.
    pub hyp_index: Vec<Option<usize>>,
    /// Correctly matched tokens.
    pub correct: usize,
    /// Substituted tokens.
    pub substitutions: usize,
    /// Deleted reference tokens.
    pub deletions: usize,
    /// Inserted hypothesis tokens.
    pub insertions: usize,
}

impl Alignment {
    /// Total errors: S + D + I. The numerator of both WER and CER.
    pub fn errors(&self) -> usize {
        self.substitutions + self.deletions + self.insertions
    }
}

/// Guard against a pathological input. 4 million cells is roughly
/// `n = m = 2000`; no 120-second clip is anywhere near it, and a report that
/// silently truncates would be worse than a refusal.
const MAX_CELLS: usize = 4_000_000;

/// Levenshtein alignment of `reference` against `hypothesis`.
///
/// The tie-break order is fixed — diagonal (match/substitution) before
/// deletion before insertion — because an unfixed tie-break makes the
/// S/D/I *split* arbitrary even though S+D+I is stable. The split is reported
/// separately, so it has to be deterministic.
pub fn align<T: PartialEq>(reference: &[T], hypothesis: &[T]) -> Alignment {
    let n = reference.len();
    let m = hypothesis.len();
    assert!(
        n.saturating_mul(m) <= MAX_CELLS,
        "alignment input too large: {n} x {m}"
    );

    // dist[i][j] = cost of aligning reference[..i] with hypothesis[..j].
    let mut dist = vec![0u32; (n + 1) * (m + 1)];
    let idx = |i: usize, j: usize| i * (m + 1) + j;
    for i in 0..=n {
        dist[idx(i, 0)] = i as u32;
    }
    for j in 0..=m {
        dist[idx(0, j)] = j as u32;
    }
    for i in 1..=n {
        for j in 1..=m {
            let sub = dist[idx(i - 1, j - 1)] + u32::from(reference[i - 1] != hypothesis[j - 1]);
            let del = dist[idx(i - 1, j)] + 1;
            let ins = dist[idx(i, j - 1)] + 1;
            dist[idx(i, j)] = sub.min(del).min(ins);
        }
    }

    // Walk back, collecting the alignment. Ties resolve to the diagonal, then
    // to the deletion, then to the insertion.
    let mut rev_ops = Vec::with_capacity(n + m);
    let mut rev_r = Vec::with_capacity(n + m);
    let mut rev_h = Vec::with_capacity(n + m);
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        if i > 0 && j > 0 {
            let same = reference[i - 1] == hypothesis[j - 1];
            let cost = dist[idx(i - 1, j - 1)] + u32::from(!same);
            if dist[idx(i, j)] == cost {
                rev_ops.push(if same { Op::Match } else { Op::Sub });
                rev_r.push(Some(i - 1));
                rev_h.push(Some(j - 1));
                i -= 1;
                j -= 1;
                continue;
            }
        }
        if i > 0 && dist[idx(i, j)] == dist[idx(i - 1, j)] + 1 {
            rev_ops.push(Op::Del);
            rev_r.push(Some(i - 1));
            rev_h.push(None);
            i -= 1;
            continue;
        }
        rev_ops.push(Op::Ins);
        rev_r.push(None);
        rev_h.push(Some(j - 1));
        j -= 1;
    }
    rev_ops.reverse();
    rev_r.reverse();
    rev_h.reverse();

    let mut a = Alignment {
        ops: rev_ops,
        ref_index: rev_r,
        hyp_index: rev_h,
        correct: 0,
        substitutions: 0,
        deletions: 0,
        insertions: 0,
    };
    for op in &a.ops {
        match op {
            Op::Match => a.correct += 1,
            Op::Sub => a.substitutions += 1,
            Op::Del => a.deletions += 1,
            Op::Ins => a.insertions += 1,
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count(a: &Alignment, op: Op) -> usize {
        a.ops.iter().filter(|o| **o == op).count()
    }

    #[test]
    fn identical_sequences_align_to_matches_only() {
        let r = ["a", "b", "c"];
        let a = align(&r, &r);
        assert_eq!(a.correct, 3);
        assert_eq!(a.errors(), 0);
        assert!(a.ops.iter().all(|o| *o == Op::Match));
    }

    #[test]
    fn a_replacement_is_one_substitution_not_one_plus_one() {
        let a = align(&["a", "b", "c"], &["a", "x", "c"]);
        assert_eq!(count(&a, Op::Sub), 1);
        assert_eq!(count(&a, Op::Del), 0);
        assert_eq!(count(&a, Op::Ins), 0);
        assert_eq!(a.correct, 2);
    }

    #[test]
    fn a_missing_token_is_a_deletion() {
        let a = align(&["a", "b", "c"], &["a", "c"]);
        assert_eq!(count(&a, Op::Del), 1);
        assert_eq!(count(&a, Op::Sub), 0);
        assert_eq!(count(&a, Op::Ins), 0);
    }

    #[test]
    fn an_extra_token_is_an_insertion() {
        let a = align(&["a", "c"], &["a", "b", "c"]);
        assert_eq!(count(&a, Op::Ins), 1);
        assert_eq!(count(&a, Op::Del), 0);
    }

    #[test]
    fn the_substitution_deletion_insertion_split_is_deterministic() {
        // One reference token, one hypothesis token, both different: 1 S.
        let a = align(&["x"], &["y"]);
        assert_eq!((a.substitutions, a.deletions, a.insertions), (1, 0, 0));
        // Repeat: the tie-break must not wobble between runs.
        for _ in 0..32 {
            let b = align(&["x"], &["y"]);
            assert_eq!(a.ops, b.ops);
        }
    }

    #[test]
    fn an_empty_reference_is_all_insertions() {
        let a = align::<String>(&[], &["a".to_string(), "b".to_string()]);
        assert_eq!(a.insertions, 2);
        assert_eq!(a.deletions, 0);
        assert_eq!(a.errors(), 2);
    }

    #[test]
    fn an_empty_hypothesis_is_all_deletions() {
        let a = align(&["a".to_string(), "b".to_string()], &[]);
        assert_eq!(a.deletions, 2);
        assert_eq!(a.insertions, 0);
    }

    #[test]
    fn two_empty_sequences_have_no_errors() {
        let a = align::<String>(&[], &[]);
        assert_eq!(a.errors(), 0);
        assert_eq!(a.correct, 0);
    }

    #[test]
    fn the_alignment_covers_every_token_on_both_sides_exactly_once() {
        let r = ["a", "b", "c", "d", "e"];
        let h = ["a", "x", "d", "f", "g"];
        let a = align(&r, &h);
        let mut ri: Vec<usize> = a.ref_index.iter().flatten().copied().collect();
        let mut hi: Vec<usize> = a.hyp_index.iter().flatten().copied().collect();
        ri.sort_unstable();
        hi.sort_unstable();
        assert_eq!(ri, (0..5).collect::<Vec<_>>());
        assert_eq!(hi, (0..5).collect::<Vec<_>>());
    }

    #[test]
    fn the_distance_equals_the_operation_count() {
        let r = ["a", "b", "c", "d"];
        let h = ["a", "c", "x", "d", "e"];
        let a = align(&r, &h);
        assert_eq!(
            a.errors(),
            a.ops.iter().filter(|o| **o != Op::Match).count()
        );
    }
}

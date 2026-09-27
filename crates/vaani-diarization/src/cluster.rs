//! Agglomerative clustering, speaker-count estimation, and refinement.
//!
//! # What runs, in order
//!
//! 1. **Agglomerative clustering, average linkage, cosine distance**, run all
//!    the way down to a single cluster so the whole dendrogram is available.
//!    Exact average linkage needs the full `N x N` distance matrix, so the
//!    window count is capped (see [`crate::config::ClusterParams::max_cluster_windows`])
//!    and every window outside the cap is assigned by its nearest centroid.
//! 2. **The stopping rule** picks how many of those clusters to keep.
//! 3. **The min-speech guard** absorbs any cluster that is too small to be a
//!    person into its nearest neighbour.
//! 4. **Lloyd refinement** re-assigns *every* window to its nearest centroid
//!    for a few iterations. This is the second stage, and it is there for a
//!    specific reason: average linkage makes a cluster's membership depend on
//!    the order in which its members arrived, so boundary windows end up in
//!    whichever cluster reached them first. One pass of nearest-centroid
//!    assignment fixes those, costs `N x k x d`, and is deterministic.
//!
//! # Why average linkage and not single or complete
//!
//! Single linkage is what everybody implements first and it is wrong for
//! diarization: it chains. A speaker whose pitch rises steadily through a
//! meeting forms a chain of windows each 0.1 away from the last, and
//! single-linkage glues the whole meeting into one cluster. Complete linkage
//! has the opposite failure: a speaker with one unusual window will not merge
//! with the rest at all, and the meeting shatters. Average linkage sits
//! between them and, importantly, its merge distances are **monotonically
//! non-decreasing**, which is what makes the whole-dendrogram stopping rules
//! in (2) meaningful.
//!
//! # The stopping rules, and how each one fails
//!
//! **Threshold** (the default). Keep merging while the next merge costs less
//! than `merge_distance` cosine units. Fails in both directions, and both
//! failures are inherent to a hand-built feature vector:
//!
//! * It **over-splits a single expressive speaker.** Median F0 is supposed to
//!   be a speaker constant, but a storyteller moves it by a third of an
//!   octave, and a [`MinSpread`](crate::config::MinSpread) floor calibrated
//!   for conversation does not stop that. This is the most likely wrong
//!   answer from this engine on a monologue, and it is the one the
//!   min-speech guard and the median filter only partly fix.
//! * It **over-merges two similar speakers.** Two men in the same octave
//!   with similar-sized vocal tracts are close to genuinely inseparable from
//!   spectral statistics alone; a trained embedder would separate them on
//!   detail this feature vector does not measure.
//!
//! **Auto (largest relative gap).** Take the biggest ratio between
//! consecutive merge distances and cut just before it. Strictly better on a
//! recording whose speakers are well separated, and it will happily cut a
//! single-speaker recording at a random point in a smooth monotone sequence.
//! It is not the default for that reason.
//!
//! **Fixed(k).** No stopping rule; merge to `k`. Used by the tests, which need
//! to check that the *linkage* works independently of the *rule*.
//!
//! # Determinism
//!
//! Every tie in this file is broken toward the lower index, and the
//! agglomeration itself breaks merge-distance ties by lowest index pair. The
//! engine's output for a given recording is therefore bit-identical across
//! runs, which matters because speaker indices are persisted in the
//! transcript and referenced by `Segment::speaker`.

use std::cmp::Ordering;

use ndarray::Array2;

use crate::config::{ClusterParams, SpeakerCount};
use crate::features::cosine_distance;

/// What agglomerative clustering produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Dendrogram {
    /// The row index at which each merge happened, i.e. `merges[i] = (a, b)`
    /// formed the cluster that contains rows `a` and `b`.
    pub merges: Vec<(usize, usize)>,
    /// The cosine cost of each merge. Monotonically non-decreasing for
    /// average linkage; asserted by the tests.
    pub costs: Vec<f64>,
    /// Number of rows clustered.
    pub rows: usize,
}

impl Dendrogram {
    /// Number of merges, which is `rows - 1` for a complete run.
    #[must_use]
    pub fn len(&self) -> usize {
        self.merges.len()
    }

    /// Whether anything was clustered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// The cost of the merge that reduced the cluster count from `k + 1` to
    /// `k`, or `None` when `k` is outside the range this run produced.
    #[must_use]
    pub fn cost_at_count(&self, k: usize) -> Option<f64> {
        // `k == rows` means no merge has happened yet, which has no cost; and
        // `k > rows` is a request the caller should have clamped.
        if k == 0 || k >= self.rows {
            return None;
        }
        // After `i` merges there are `rows - i` clusters, so reaching `k`
        // takes `rows - k` merges, and the last of those is index `rows - k - 1`.
        self.costs.get(self.rows - k - 1).copied()
    }
}

/// Agglomerative clustering with average linkage over a full cosine matrix.
///
/// `rows` must be square. Ties are broken by the lower row index, so the
/// result depends only on `rows` and not on iteration order.
#[must_use]
pub fn agglomerate(rows: usize, distances: &[f32]) -> Dendrogram {
    let mut dendro = Dendrogram {
        merges: Vec::new(),
        costs: Vec::new(),
        rows,
    };
    if rows <= 1 {
        return dendro;
    }
    debug_assert_eq!(distances.len(), rows * rows);

    // `parent` collapses the merge tree. `active` lists the surviving roots
    // by row index; `size` is the member count average linkage needs.
    let mut parent: Vec<usize> = (0..rows).collect();
    let mut active: Vec<usize> = (0..rows).collect();
    let mut size = vec![1usize; rows];

    let d = |a: usize, b: usize| -> f64 {
        if a == b {
            0.0
        } else {
            f64::from(distances[a * rows + b])
        }
    };

    while active.len() > 1 {
        let mut best = f64::INFINITY;
        let mut pair = (0usize, 0usize);
        for (ii, &i) in active.iter().enumerate() {
            for &j in active.iter().skip(ii + 1) {
                let cost = d(i, j);
                // Strict `<` plus the ascending scan order means the first
                // pair found at the minimum wins, which is the lowest index
                // pair. Deterministic.
                if cost < best {
                    best = cost;
                    pair = (i, j);
                }
            }
        }
        let (a, b) = pair;
        parent[b] = a;
        size[a] += size[b];
        active.retain(|x| *x != b);
        dendro.merges.push((a, b));
        dendro.costs.push(best);
    }
    // `parent` is only needed to document the tree; the replay in
    // `partition_at` is driven by `merges` alone, which keeps the merge log
    // the single source of truth.
    debug_assert_eq!(parent.len(), rows);
    dendro
}

/// Replay a merge log to recover the labels that `k` clusters produce.
///
/// This is what makes the stopping rules cheap: the clustering runs once, all
/// the way to one cluster, and every candidate `k` is then a linear replay
/// rather than a second clustering run.
///
/// Rows are relabelled to a contiguous `0..k` in order of first appearance, so
/// a cut always produces well-formed cluster indices.
#[must_use]
pub fn partition_at(dendro: &Dendrogram, k: usize) -> Vec<usize> {
    let rows = dendro.rows;
    if rows == 0 || k == 0 {
        return Vec::new();
    }
    let k = k.min(rows);
    let mut parent: Vec<usize> = (0..rows).collect();
    for &(a, b) in dendro.merges.iter().take(rows - k) {
        parent[b] = a;
    }
    let mut root_to_label: Vec<usize> = vec![usize::MAX; rows];
    let mut next = 0usize;
    let mut labels = vec![0usize; rows];
    for (i, slot) in labels.iter_mut().enumerate() {
        let mut r = i;
        while parent[r] != r {
            r = parent[r];
        }
        if root_to_label[r] == usize::MAX {
            root_to_label[r] = next;
            next += 1;
        }
        *slot = root_to_label[r];
    }
    labels
}

/// The full distance matrix for `rows` L2-normalised vectors.
///
/// A `f32` matrix of `rows^2` entries: 2000 rows is 16 MB, which is the whole
/// reason the caller caps the row count.
#[must_use]
pub fn distance_matrix(rows: &[Vec<f32>]) -> Array2<f32> {
    let n = rows.len();
    let mut m = Array2::<f32>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let v = cosine_distance(&rows[i], &rows[j]) as f32;
            m[[i, j]] = v;
            m[[j, i]] = v;
        }
    }
    m
}

/// L2-normalise a cluster centroid so the dot product stays a cosine.
fn normalise(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    } else {
        v.iter_mut().for_each(|x| *x = 0.0);
    }
}

/// Recompute centroids from an assignment.
fn centroids_of(rows: &[Vec<f32>], labels: &[usize], k: usize) -> Vec<Vec<f32>> {
    let d = rows.first().map(|r| r.len()).unwrap_or(0);
    let mut sums = vec![vec![0f32; d]; k];
    let mut counts = vec![0usize; k];
    for (r, l) in rows.iter().zip(labels) {
        if let Some(slot) = sums.get_mut(*l) {
            for (a, b) in slot.iter_mut().zip(r) {
                *a += b;
            }
        }
        if let Some(c) = counts.get_mut(*l) {
            *c += 1;
        }
    }
    for (i, n) in counts.iter().enumerate() {
        if *n > 0 {
            let inv = 1.0 / *n as f32;
            sums[i].iter_mut().for_each(|x| *x *= inv);
        } else {
            sums[i].iter_mut().for_each(|x| *x = 0.0);
        }
    }
    sums.iter_mut().for_each(|v| normalise(v));
    sums
}

/// One pass of Lloyd's algorithm: assign every row to its nearest centroid.
///
/// Returns `(labels, moved)` where `moved` is the number of rows whose label
/// changed. Ties go to the lower centroid index, so the pass is a pure
/// function of its input.
fn lloyd_pass(
    rows: &[Vec<f32>],
    centroids: &[Vec<f32>],
    previous: &[usize],
) -> (Vec<usize>, usize) {
    let mut labels = vec![0usize; rows.len()];
    let mut moved = 0usize;
    for (i, r) in rows.iter().enumerate() {
        let mut best = 0usize;
        let mut best_cost = f64::INFINITY;
        for (c, cen) in centroids.iter().enumerate() {
            let cost = cosine_distance(r, cen);
            if cost < best_cost {
                best_cost = cost;
                best = c;
            }
        }
        labels[i] = best;
        if previous.get(i).copied() != Some(best) {
            moved += 1;
        }
    }
    (labels, moved)
}

/// The outcome of clustering.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterOutcome {
    /// One cluster index per input row, contiguous from 0.
    pub labels: Vec<usize>,
    /// How many clusters survived.
    pub num_clusters: usize,
    /// L2-normalised cluster centroids.
    pub centroids: Vec<Vec<f32>>,
    /// The cost of the last merge that was performed, for diagnostics.
    pub final_cost: f64,
    /// Which stopping rule fired, for the benchmark report.
    pub rule: &'static str,
}

/// Cluster `rows`, which must already be z-scored, weighted and L2-normalised.
///
/// `speech_ms` gives each row its speech weight; it is used only by the
/// min-speech guard and by the `Auto` rule's tie-breaking, never by the
/// distances themselves.
#[must_use]
pub fn cluster(
    rows: &[Vec<f32>],
    speech_ms: &[i64],
    params: &ClusterParams,
    min_cluster_speech_ms: i64,
    merge_distance: f64,
) -> ClusterOutcome {
    let empty = ClusterOutcome {
        labels: Vec::new(),
        num_clusters: 0,
        centroids: Vec::new(),
        final_cost: 0.0,
        rule: "none",
    };
    if rows.is_empty() {
        return empty;
    }
    if rows.len() == 1 {
        return ClusterOutcome {
            labels: vec![0],
            num_clusters: 1,
            centroids: vec![rows[0].clone()],
            final_cost: 0.0,
            rule: "single-window",
        };
    }

    // --- 1. the dendrogram ------------------------------------------------
    // The `N x N` matrix is 16 MB at the default cap and is the reason the cap
    // exists. It is dropped as soon as the merge log is complete, because the
    // replay in `partition_at` needs only the log.
    let n = rows.len();
    let flat: Vec<f32> = distance_matrix(rows).iter().copied().collect();
    let dendro = agglomerate(n, &flat);
    drop(flat);

    // --- 2. the stopping rule ---------------------------------------------
    let (k, rule) = choose_count(&dendro, &params.count, merge_distance);

    // --- 3. the min-speech guard ------------------------------------------
    let mut labels = partition_at(&dendro, k);
    absorb_small_clusters(&mut labels, rows, speech_ms, min_cluster_speech_ms);
    let num_clusters = relabel_contiguous(&mut labels);

    // --- 4. Lloyd refinement ----------------------------------------------
    let mut centroids = centroids_of(rows, &labels, num_clusters.max(1));
    for _ in 0..params.refine_iters {
        if centroids.len() != num_clusters {
            break;
        }
        let (next, moved) = lloyd_pass(rows, &centroids, &labels);
        if moved == 0 {
            break;
        }
        let mut candidate = next;
        let nc = relabel_contiguous(&mut candidate);
        if nc != num_clusters {
            // A refinement step that empties a cluster has found a degenerate
            // configuration. Keeping the previous labels is the conservative
            // answer; the alternative is inventing a speaker.
            break;
        }
        labels = candidate;
        centroids = centroids_of(rows, &labels, num_clusters);
    }

    let final_cost = dendro.cost_at_count(num_clusters).unwrap_or(0.0);
    ClusterOutcome {
        labels,
        num_clusters,
        centroids,
        final_cost,
        rule,
    }
}

/// Apply the configured stopping rule to a completed dendrogram.
fn choose_count(
    dendro: &Dendrogram,
    count: &SpeakerCount,
    merge_distance: f64,
) -> (usize, &'static str) {
    let rows = dendro.rows;
    match *count {
        SpeakerCount::Fixed(k) => (k.clamp(1, rows), "fixed"),
        SpeakerCount::Threshold { max_speakers, .. } => {
            let cap = max_speakers.clamp(1, rows);
            // Keep merging while the next merge is affordable. Because
            // average linkage's costs are non-decreasing, "the smallest k whose
            // producing merge still fits under the threshold" is exactly the
            // answer: everything below it was affordable too.
            //
            // `cost_at_count(k)` is the cost of the merge that reduced the
            // count from k+1 to k, so scanning k upwards and taking the first
            // that fits walks the dendrogram from the bottom.
            let mut best = cap;
            for k in 1..=cap {
                match dendro.cost_at_count(k) {
                    Some(c) if c <= merge_distance => {
                        best = k;
                        break;
                    }
                    Some(_) => continue,
                    None => break,
                }
            }
            // If not even the final merge fits, no k qualifies and `best` stays
            // at the cap. That is deliberate: the threshold is then read as
            // "this recording has more than N speakers" rather than collapsing
            // the whole meeting into one label.
            (best, "threshold")
        }
        SpeakerCount::Auto {
            max_speakers,
            min_gap_ratio,
        } => {
            let cap = max_speakers.clamp(1, rows);
            let mut best = 1usize;
            let mut best_ratio = min_gap_ratio;
            // Cut just below the largest relative jump. If the jump is between
            // `costs[i]` and `costs[i+1]`, then the merges up to and including
            // `i` are the cheap ones and `costs[i+1]` is the first expensive
            // one, so the answer is the cluster count after performing merges
            // `0..=i`, which is `rows - i - 1`.
            //
            // The `- 1` is not cosmetic. With two speakers the jump sits on
            // the last merge and the formula gives 2; omitting it gives 1,
            // i.e. the whole meeting collapses to one label. With three
            // speakers the jump sits on the first of the two cross-speaker
            // merges and it gives 3 where the other form gives 4.
            for i in 0..dendro.costs.len().saturating_sub(1) {
                let lo = dendro.costs[i];
                if lo <= 1e-9 {
                    continue;
                }
                let ratio = dendro.costs[i + 1] / lo;
                if ratio > best_ratio {
                    best_ratio = ratio;
                    best = dendro.rows.saturating_sub(i + 1);
                }
            }
            (best.clamp(1, cap), "auto-gap")
        }
    }
}

/// Merge every cluster with less than `min_cluster_speech_ms` of speech into
/// whichever surviving cluster its members are closest to.
///
/// The alternative — dropping the cluster — would delete a person's turns
/// from the transcript's speaker annotation, which is worse than labelling
/// them as somebody else, because the user can see a wrong label and a
/// missing one looks like the engine simply did not run.
fn absorb_small_clusters(
    labels: &mut [usize],
    rows: &[Vec<f32>],
    speech_ms: &[i64],
    min_speech_ms: i64,
) {
    if min_speech_ms <= 0 {
        return;
    }
    let k = labels.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    if k <= 1 {
        return;
    }
    let mut total = vec![0i64; k];
    for (l, s) in labels.iter().zip(speech_ms) {
        if let Some(slot) = total.get_mut(*l) {
            *slot += s;
        }
    }
    let small: Vec<usize> = (0..k).filter(|i| total[*i] < min_speech_ms).collect();
    if small.is_empty() {
        return;
    }
    let big: Vec<usize> = (0..k).filter(|i| total[*i] >= min_speech_ms).collect();
    if big.is_empty() {
        return;
    }

    // Centroid of each small cluster, then the nearest surviving centroid.
    let mut small_centroid: Vec<(usize, Vec<f32>)> = Vec::new();
    for s in &small {
        let members: Vec<&Vec<f32>> = rows
            .iter()
            .zip(labels.iter())
            .filter(|(_, l)| *l == s)
            .map(|(r, _)| r)
            .collect();
        if members.is_empty() {
            continue;
        }
        let d = members[0].len();
        let mut c = vec![0f32; d];
        for m in &members {
            for (a, b) in c.iter_mut().zip(m.iter()) {
                *a += b;
            }
        }
        let inv = 1.0 / members.len() as f32;
        c.iter_mut().for_each(|x| *x *= inv);
        normalise(&mut c);
        small_centroid.push((*s, c));
    }
    let big_centroid: Vec<Vec<f32>> = big
        .iter()
        .map(|b| {
            let members: Vec<&Vec<f32>> = rows
                .iter()
                .zip(labels.iter())
                .filter(|(_, l)| *l == b)
                .map(|(r, _)| r)
                .collect();
            let d = members.first().map(|m| m.len()).unwrap_or(0);
            let mut c = vec![0f32; d];
            for m in &members {
                for (a, x) in c.iter_mut().zip(m.iter()) {
                    *a += x;
                }
            }
            if !members.is_empty() {
                c.iter_mut().for_each(|x| *x *= 1.0 / members.len() as f32);
            }
            normalise(&mut c);
            c
        })
        .collect();

    for (s, c) in small_centroid {
        let mut best = 0usize;
        let mut best_cost = f64::INFINITY;
        for (bi, bc) in big_centroid.iter().enumerate() {
            let cost = cosine_distance(&c, bc);
            if cost < best_cost {
                best_cost = cost;
                best = bi;
            }
        }
        let target = big[best];
        for l in labels.iter_mut() {
            if *l == s {
                *l = target;
            }
        }
    }
}

/// Renumber labels to `0..n` in order of first appearance, in place.
fn relabel_contiguous(labels: &mut [usize]) -> usize {
    let mut map: Vec<usize> = Vec::new();
    for l in labels.iter_mut() {
        match map.iter().position(|m| *m == *l) {
            Some(i) => *l = i,
            None => {
                map.push(*l);
                *l = map.len() - 1;
            }
        }
    }
    map.len()
}

/// Complete-linkage distance between two sets of rows, and the mean of it.
///
/// Complete linkage is `max_{i in A, j in B} d(i, j)`, so the reported figure
/// is the worst-case separation between two groups. It is the right
/// diagnostic for "are these two speakers actually separable?", because it
/// is the pair that a nearest-centroid assignment would get wrong.
#[must_use]
pub fn complete_link_separation(a: &[usize], b: &[usize], rows: &[Vec<f32>]) -> Option<(f64, f64)> {
    let max = complete_link_distance(a, b, rows);
    let mean = complete_link_mean(a, b, rows);
    max.zip(mean)
}

/// The complete-linkage (maximum) distance between two groups of rows.
#[must_use]
pub fn complete_link_distance(a: &[usize], b: &[usize], rows: &[Vec<f32>]) -> Option<f64> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let mut worst: f64 = 0.0;
    for &i in a {
        for &j in b {
            let d = cosine_distance(&rows[i], &rows[j]);
            if d > worst {
                worst = d;
            }
        }
    }
    Some(worst)
}

/// The mean of the complete-linkage distances between two groups of rows.
#[must_use]
pub fn complete_link_mean(a: &[usize], b: &[usize], rows: &[Vec<f32>]) -> Option<f64> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let mut acc = 0f64;
    let mut n = 0u64;
    for &i in a {
        for &j in b {
            acc += cosine_distance(&rows[i], &rows[j]);
            n += 1;
        }
    }
    Some(acc / n as f64)
}

/// Deterministic comparison used by the stopping rules.
#[must_use]
pub fn cheaper(a: (f64, usize, usize), b: (f64, usize, usize)) -> Ordering {
    a.0.partial_cmp(&b.0)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.1.cmp(&b.1))
        .then_with(|| a.2.cmp(&b.2))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Twelve rows in **three tight groups** of four, 120 degrees apart, with a
    /// deterministic wobble inside each group.
    ///
    /// Deliberately not twelve points spread evenly round a circle: that is a
    /// chain, and average linkage cutting a chain into three arcs is a
    /// different (and much harder) test than separating three blobs. This
    /// fixture answers "does the linkage find the obvious structure", so the
    /// structure has to be obvious.
    fn three_groups() -> Vec<Vec<f32>> {
        (0..12)
            .map(|i| {
                let group = i / 4;
                let a = group as f32 * 2.0 * std::f32::consts::PI / 3.0 + (i % 4) as f32 * 0.012;
                vec![a.cos(), a.sin()]
            })
            .collect()
    }

    fn norm(mut v: Vec<f32>) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= n);
        v
    }

    fn n_rows(rows: &[Vec<f32>]) -> Vec<f32> {
        distance_matrix(rows).iter().copied().collect()
    }

    #[test]
    fn an_empty_or_single_row_input_agglomerates_to_nothing() {
        assert!(agglomerate(0, &[]).is_empty());
        let d = agglomerate(1, &[0.0]);
        assert_eq!(d.len(), 0, "a single row needs no merges");
        assert_eq!(d.rows, 1);
    }

    #[test]
    fn agglomeration_produces_exactly_rows_minus_one_merges() {
        let rows = three_groups();
        let d = agglomerate(rows.len(), &n_rows(&rows));
        assert_eq!(d.len(), rows.len() - 1);
    }

    #[test]
    fn average_linkage_costs_are_monotonically_non_decreasing() {
        // The reducibility property the whole stopping-rule design rests on.
        // If this fails, every "largest gap" rule is meaningless.
        let rows = three_groups();
        let d = agglomerate(rows.len(), &n_rows(&rows));
        for w in d.costs.windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "costs went down: {w:?}");
        }
    }

    #[test]
    fn a_cut_at_three_reproduces_the_three_groups() {
        let rows = three_groups();
        let d = agglomerate(rows.len(), &n_rows(&rows));
        let labels = partition_at(&d, 3);
        assert_eq!(labels.len(), 12);
        assert_eq!(labels.iter().copied().max().unwrap() + 1, 3);
        for g in 0..3 {
            let members: Vec<usize> = (0..12).filter(|i| labels[*i] == g).collect();
            assert_eq!(members.len(), 4, "group {g} got {members:?}");
        }
    }

    #[test]
    fn a_cut_at_one_gives_every_row_the_same_label() {
        let rows = three_groups();
        let d = agglomerate(rows.len(), &n_rows(&rows));
        let labels = partition_at(&d, 1);
        assert!(labels.iter().all(|l| *l == 0));
    }

    #[test]
    fn a_cut_above_the_row_count_is_clamped_not_a_panic() {
        let rows = three_groups();
        let d = agglomerate(rows.len(), &n_rows(&rows));
        let labels = partition_at(&d, 99);
        assert_eq!(labels.len(), 12);
        // 99 clamps to the row count, i.e. every row is its own cluster.
        assert_eq!(labels, (0..12).collect::<Vec<_>>());
        assert!(partition_at(&d, 0).is_empty());
    }

    #[test]
    fn labels_are_contiguous_from_zero_at_every_cut() {
        let rows = three_groups();
        let d = agglomerate(rows.len(), &n_rows(&rows));
        for k in 1..=12 {
            let labels = partition_at(&d, k);
            let mut seen: Vec<usize> = labels.clone();
            seen.sort_unstable();
            seen.dedup();
            let want: Vec<usize> = (0..seen.len()).collect();
            assert_eq!(seen, want, "cut at {k} gave {seen:?}");
        }
    }

    #[test]
    fn cost_at_count_is_the_cost_of_the_merge_that_made_that_count() {
        let rows = three_groups();
        let d = agglomerate(rows.len(), &n_rows(&rows));
        // 12 rows, 11 merges. Reaching 12 costs nothing; reaching 1 is the
        // last merge.
        assert_eq!(d.cost_at_count(12), None);
        assert_eq!(d.cost_at_count(0), None);
        assert_eq!(d.cost_at_count(1), Some(d.costs[10]));
        assert_eq!(d.cost_at_count(2), Some(d.costs[9]));
        assert_eq!(d.cost_at_count(3), Some(d.costs[8]));
    }

    #[test]
    fn agglomeration_is_deterministic() {
        let rows = three_groups();
        let flat = n_rows(&rows);
        let a = agglomerate(rows.len(), &flat);
        let b = agglomerate(rows.len(), &flat);
        assert_eq!(a, b, "two runs of the same input disagreed");
    }

    #[test]
    fn the_threshold_rule_stops_at_the_widest_gap() {
        let rows = three_groups();
        let params = ClusterParams {
            count: SpeakerCount::Threshold {
                merge_distance: 0.34,
                max_speakers: 8,
            },
            refine_iters: 0,
            ..ClusterParams::default()
        };
        let out = cluster(&rows, &[1_000; 12], &params, 0, 0.34);
        assert_eq!(out.rule, "threshold");
        assert_eq!(
            out.num_clusters, 3,
            "labels {:?} cost {}",
            out.labels, out.final_cost
        );
    }

    #[test]
    fn the_threshold_rule_keeps_one_speaker_as_one() {
        // Rows that are all within 0.05 of each other: no gap anywhere, so the
        // rule must never split them.
        let rows: Vec<Vec<f32>> = (0..20)
            .map(|i| {
                let a = i as f32 * 0.01;
                norm(vec![a.cos(), a.sin()])
            })
            .collect();
        let params = ClusterParams {
            count: SpeakerCount::Threshold {
                merge_distance: 0.34,
                max_speakers: 8,
            },
            refine_iters: 0,
            ..ClusterParams::default()
        };
        let out = cluster(&rows, &[1_000; 20], &params, 0, 0.34);
        assert_eq!(out.num_clusters, 1, "labels {:?}", out.labels);
    }

    #[test]
    fn a_fixed_count_is_honoured_exactly() {
        let rows = three_groups();
        for k in [1usize, 2, 3, 5] {
            let params = ClusterParams {
                count: SpeakerCount::Fixed(k),
                refine_iters: 0,
                ..ClusterParams::default()
            };
            // Infinite merge distance, which is what `Fixed` produces.
            let out = cluster(&rows, &[1_000; 12], &params, 0, f64::INFINITY);
            assert_eq!(out.num_clusters, k.min(12));
        }
    }

    #[test]
    fn the_auto_rule_finds_the_gap_in_three_clean_groups() {
        let rows = three_groups();
        let params = ClusterParams {
            count: SpeakerCount::Auto {
                max_speakers: 8,
                min_gap_ratio: 1.2,
            },
            refine_iters: 0,
            ..ClusterParams::default()
        };
        let out = cluster(&rows, &[1_000; 12], &params, 0, f64::INFINITY);
        assert_eq!(out.rule, "auto-gap");
        assert_eq!(out.num_clusters, 3, "labels {:?}", out.labels);
    }

    #[test]
    fn the_max_speakers_cap_is_never_exceeded_by_any_rule() {
        let rows = three_groups();
        for cap in [1usize, 2, 3, 10] {
            let params = ClusterParams {
                count: SpeakerCount::Threshold {
                    merge_distance: 0.34,
                    max_speakers: cap,
                },
                refine_iters: 0,
                ..ClusterParams::default()
            };
            let out = cluster(&rows, &[1_000; 12], &params, 0, 0.34);
            assert!(
                out.num_clusters <= cap,
                "cap {cap} exceeded: {}",
                out.num_clusters
            );
        }
    }

    #[test]
    fn the_min_speech_guard_absorbs_a_tiny_cluster() {
        // Two tight groups, but the second has almost no speech. The guard
        // must fold it into the first rather than reporting a speaker who
        // said one word.
        let mut rows: Vec<Vec<f32>> = (0..10).map(|i| norm(vec![1.0, i as f32 * 0.001])).collect();
        for i in 0..3 {
            rows.push(norm(vec![-1.0, i as f32 * 0.001]));
        }
        let mut speech: Vec<i64> = vec![10_000; 10];
        speech.extend(vec![200; 3]);
        let params = ClusterParams {
            count: SpeakerCount::Fixed(2),
            refine_iters: 0,
            ..ClusterParams::default()
        };
        let out = cluster(&rows, &speech, &params, 2_000, f64::INFINITY);
        assert_eq!(out.num_clusters, 1, "the tiny cluster survived");
    }

    #[test]
    fn the_min_speech_guard_keeps_two_real_speakers() {
        let rows: Vec<Vec<f32>> = (0..6)
            .map(|i| norm(vec![1.0, i as f32 * 0.001]))
            .chain((0..6).map(|i| norm(vec![-1.0, i as f32 * 0.001])))
            .collect();
        let params = ClusterParams {
            count: SpeakerCount::Fixed(2),
            refine_iters: 0,
            ..ClusterParams::default()
        };
        let out = cluster(&rows, &[5_000; 12], &params, 2_000, f64::INFINITY);
        assert_eq!(out.num_clusters, 2);
    }

    #[test]
    fn the_min_speech_guard_never_merges_everything_into_nothing() {
        // Every cluster is tiny: the guard must decline to act rather than
        // erasing the whole result.
        let rows: Vec<Vec<f32>> = (0..6)
            .map(|i| norm(vec![1.0, i as f32 * 0.001]))
            .chain((0..6).map(|i| norm(vec![-1.0, i as f32 * 0.001])))
            .collect();
        let speech = vec![100i64; 12];
        let params = ClusterParams {
            count: SpeakerCount::Fixed(2),
            refine_iters: 0,
            ..ClusterParams::default()
        };
        let out = cluster(&rows, &speech, &params, 2_000, f64::INFINITY);
        assert_eq!(out.num_clusters, 2, "the guard ate both speakers");
    }

    #[test]
    fn zero_and_one_row_inputs_are_handled() {
        let params = ClusterParams::default();
        assert_eq!(cluster(&[], &[], &params, 0, 0.34).num_clusters, 0);
        let one = vec![vec![1.0f32, 0.0]];
        let out = cluster(&one, &[100], &params, 0, 0.34);
        assert_eq!(out.num_clusters, 1);
        assert_eq!(out.labels, vec![0]);
        assert_eq!(out.rule, "single-window");
    }

    #[test]
    fn lloyd_refinement_reassigns_a_straddling_row() {
        // Row 6 sits between the two groups. Agglomerative linkage decides by
        // history; the refinement step must move it to whichever centroid it is
        // actually closest to, and there the -1 group is twice as dense.
        let mut rows: Vec<Vec<f32>> = (0..5).map(|_| norm(vec![1.0, 0.0])).collect();
        rows.push(norm(vec![0.0, 1.0]));
        for _ in 0..5 {
            rows.push(norm(vec![-1.0, 0.0]));
        }
        let params = ClusterParams {
            count: SpeakerCount::Fixed(2),
            refine_iters: 5,
            ..ClusterParams::default()
        };
        let out = cluster(&rows, &[1_000; 11], &params, 0, f64::INFINITY);
        assert_eq!(out.num_clusters, 2);
        // Whichever cluster the straddle row ends up in, it must not be alone.
        let mut counts = vec![0usize; 2];
        for l in &out.labels {
            counts[*l] += 1;
        }
        assert!(counts.iter().all(|c| *c >= 2), "counts {counts:?}");
    }

    #[test]
    fn refinement_never_invents_a_speaker() {
        let rows: Vec<Vec<f32>> = (0..8).map(|i| norm(vec![1.0, i as f32 * 0.0001])).collect();
        let params = ClusterParams {
            count: SpeakerCount::Fixed(1),
            refine_iters: 10,
            ..ClusterParams::default()
        };
        let out = cluster(&rows, &[1_000; 8], &params, 0, f64::INFINITY);
        assert_eq!(out.num_clusters, 1);
    }

    #[test]
    fn centroids_are_l2_normalised() {
        let rows = three_groups();
        let params = ClusterParams {
            count: SpeakerCount::Fixed(3),
            refine_iters: 1,
            ..ClusterParams::default()
        };
        let out = cluster(&rows, &[1_000; 12], &params, 0, f64::INFINITY);
        for c in &out.centroids {
            let n = c.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-4, "centroid norm {n}");
        }
    }

    #[test]
    fn relabelling_is_contiguous_and_order_preserving() {
        let mut labels = vec![7usize, 3, 7, 9, 3];
        assert_eq!(relabel_contiguous(&mut labels), 3);
        assert_eq!(labels, vec![0, 1, 0, 2, 1]);
        let mut empty: Vec<usize> = Vec::new();
        assert_eq!(relabel_contiguous(&mut empty), 0);
    }

    #[test]
    fn complete_link_separation_is_monotone_in_the_separation() {
        // The diagnostic has to be monotone in the thing it measures, or it is
        // not a diagnostic. Group A is three near-identical rows; group B is
        // three near-identical rows close to A, group C three rows far from A.
        let a: Vec<usize> = vec![0, 1, 2];
        let near: Vec<Vec<f32>> = vec![
            norm(vec![1.0, 0.0]),
            norm(vec![1.0, 0.1]),
            norm(vec![1.0, -0.1]),
        ];
        let far: Vec<Vec<f32>> = vec![
            norm(vec![0.0, 1.0]),
            norm(vec![0.0, 0.9]),
            norm(vec![0.0, 1.1]),
        ];
        let b: Vec<usize> = vec![3, 4, 5];
        let c: Vec<usize> = vec![6, 7, 8];
        let rows: Vec<Vec<f32>> = [near.clone(), near.clone(), far.clone()]
            .concat()
            .into_iter()
            .chain(far)
            .collect();
        // A vs B (near) then A vs C (far). C's rows are appended twice so the
        // second group has its own copies, which does not change any distance
        // that involves A.
        let near_d = complete_link_distance(&a, &b, &rows).unwrap();
        let far_d = complete_link_distance(&a, &c, &rows).unwrap();
        assert!(far_d > near_d, "far {far_d} vs near {near_d}");
        let (max, mean) = complete_link_separation(&a, &b, &rows).unwrap();
        assert!(max >= mean);
        assert!(mean > 0.0);
    }

    #[test]
    fn complete_link_of_an_empty_group_is_none() {
        assert!(complete_link_distance(&[], &[0], &three_groups()).is_none());
        assert!(complete_link_separation(&[0], &[], &three_groups()).is_none());
    }

    #[test]
    fn the_cheaper_comparator_breaks_ties_by_index() {
        assert_eq!(cheaper((1.0, 0, 1), (1.0, 0, 2)), Ordering::Less);
        assert_eq!(cheaper((1.0, 1, 0), (1.0, 0, 9)), Ordering::Greater);
        assert_eq!(cheaper((0.5, 0, 0), (1.0, 9, 9)), Ordering::Less);
        // NaN must not panic; it is treated as equal and the indices decide.
        assert_eq!(cheaper((f64::NAN, 0, 0), (f64::NAN, 1, 0)), Ordering::Less);
    }
}

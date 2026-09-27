//! Turning a [`SpeechTimeline`] into the chunks the recogniser is handed.
//!
//! # Why a chunker, and not `for seg in segments`
//!
//! Because the recogniser's cost is not linear in the audio it is given and
//! not linear in the number of calls either. A chunk boundary is:
//!
//! * a **decoder re-prime** -- the RNN-T state is reset and has to re-learn the
//!   acoustic context;
//! * a possible **mid-word cut**, which costs a real word;
//! * a possible **memory bound** -- SraVaani's encoder activations scale with
//!   the frame count, so an unbounded chunk from a monotone speaker is an
//!   out-of-memory error 40 minutes into a job.
//!
//! All three are handled here, and the ordering between them is the design:
//!
//! ```text
//! segments -> merge -> split over-long -> greedy accumulate -> cut in a pause
//!          -> pad + clip -> coalesce anything under the floor
//! ```
//!
//! # The stage order, and why
//!
//! 1. **Merge** segments closer than `merge_gap_ms`. First, because it changes
//!    what the boundary search can find: a 250 ms gap is a pause *inside* an
//!    utterance and must not be a cut candidate.
//! 2. **Split** any span longer than `max_ms` into cap-sized pieces. This is
//!    what makes the greedy loop total: after it, every span is at most the
//!    cap, so the loop can never be in a state where it cannot make progress.
//!    Splitting here rather than special-casing the cap inside the loop is what
//!    turns "does a 10-minute monologue get cut?" from an argument about loop
//!    termination into one line of arithmetic (`ceil(len / max)` chunks).
//! 3. **Greedily accumulate** until the accumulated speech reaches
//!    `target_ms`.
//! 4. **Cut at the midpoint of the following pause.** Not at the target and
//!    not at the end of the last segment. The midpoint is the only point in the
//!    pause that gives the *next* chunk the same amount of slack, and cutting
//!    at either edge of a pause yields the same audio -- so the cut is by
//!    construction never mid-word. When the pause that follows the target is
//!    further away than `boundary_search_ms`, the cut is made at the *start* of
//!    that pause instead: waiting for a two-minute silence to reach its
//!    midpoint would produce a 150-second chunk for a 30-second target.
//! 5. **Pad** by `pad_ms`, then clip against the neighbour and the buffer so
//!    chunks can never overlap unless `overlap_ms > 0`. Interior padding is
//!    deliberately *not* counted twice: two adjacent chunks share the silence
//!    between them, and giving each the full 60 ms would transcribe it twice.
//! 6. **Coalesce** any chunk shorter than `min_ms` with a neighbour, as long
//!    as the result fits in `max_ms`. This is the enforcement of the front
//!    end's floor, and coalescing is the right mechanism: *extending* a short
//!    chunk forward would either eat the next chunk's leading silence or its
//!    first word, and eating a first word is a correctness bug.
//!
//! # The one case the floor cannot satisfy
//!
//! When the remaining media after a short chunk is itself shorter than
//! `min_ms`, the floor is clamped to the buffer and the chunk stays short. The
//! alternative -- dropping the audio -- is worse. This is asserted in a test
//! and is the documented behaviour, not an oversight.
//!
//! # What this never does
//!
//! **It never reads audio.** `plan` takes a `SpeechTimeline` (a few dozen bytes
//! per segment) and a total duration, and returns `Vec<Chunk>` of two `Time`s. A
//! three-hour recording's plan is a few hundred chunks and a few kilobytes; the
//! samples are sliced by the caller with `AudioBuffer::slice_mono`, one chunk at
//! a time. That is what keeps peak memory flat in the length of the recording.

use serde::{Deserialize, Serialize};
use vaani_core::Time;
use vaani_core::pipeline::{SpeechSegment, SpeechTimeline};

use crate::config::ChunkPolicy;

/// One chunk of audio to recognise, in **source-media** coordinates.
///
/// Deliberately not samples. Holding samples here would make `plan` a function
/// of the audio length, which is the property the whole design exists to
/// avoid. The caller slices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chunk {
    /// Inclusive start, in source-media coordinates.
    pub start: Time,
    /// Exclusive end, in source-media coordinates.
    pub end: Time,
}

impl Chunk {
    /// Length in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> i64 {
        self.end.as_millis() - self.start.as_millis()
    }

    /// Whether `t` falls inside the chunk, half-open.
    #[must_use]
    pub fn contains(&self, t: Time) -> bool {
        t >= self.start && t < self.end
    }

    /// Whether two chunks share any audio.
    #[must_use]
    pub fn overlaps(&self, other: &Chunk) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// The overlapping span of two chunks, or `None` if they are disjoint.
    #[must_use]
    pub fn intersect(&self, other: &Chunk) -> Option<Chunk> {
        let start = self.start.max(other.start);
        let end = self.end.min(other.end);
        if start < end {
            Some(Chunk { start, end })
        } else {
            None
        }
    }
}

/// A half-open span of the timeline, in milliseconds, with no confidence
/// attached. The chunker works on spans because a merge across a short gap has
/// no meaningful single confidence, and carrying one would be a lie.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// Inclusive start in milliseconds.
    pub start: i64,
    /// Exclusive end in milliseconds.
    pub end: i64,
}

impl Span {
    /// Length in milliseconds.
    #[must_use]
    pub fn len_ms(&self) -> i64 {
        self.end - self.start
    }

    /// Whether the span is non-empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

/// Split spans longer than `max_ms` into cap-sized pieces.
///
/// The invariant this establishes -- **no span is longer than the cap** -- is
/// what makes [`Chunker::boundaries`] provably terminating.
#[must_use]
pub fn split_at_cap(spans: &[Span], max_ms: i64) -> Vec<Span> {
    if max_ms <= 0 {
        return spans.to_vec();
    }
    let mut out = Vec::with_capacity(spans.len());
    for s in spans {
        let mut at = s.start;
        while at < s.end {
            let stop = at.saturating_add(max_ms).min(s.end);
            if stop <= at {
                break;
            }
            out.push(Span {
                start: at,
                end: stop,
            });
            at = stop;
        }
    }
    out
}

/// Merge segments whose gap is smaller than `gap_ms`.
///
/// Overlapping segments are merged too: a detector that emits two segments
/// sharing an instant is not wrong, but the chunker must not be the thing that
/// has to cope with it. Input must be sorted; output is sorted, non-overlapping,
/// and separated by at least `gap_ms` wherever the input allowed it.
#[must_use]
pub fn merge_segments(segments: &[SpeechSegment], gap_ms: i64) -> Vec<Span> {
    let mut out: Vec<Span> = Vec::with_capacity(segments.len());
    for s in segments {
        let start = s.start.as_millis();
        let end = s.end.as_millis();
        if end <= start {
            // A zero-length segment carries no audio; carrying it forward would
            // make a chunk boundary land on a point that bisects nothing.
            continue;
        }
        match out.last_mut() {
            Some(last) if start - last.end < gap_ms => {
                last.end = last.end.max(end);
            }
            _ => out.push(Span { start, end }),
        }
    }
    out
}

/// A chunk planner.
///
/// Holds only a `ChunkPolicy`, which is `Copy`, so one `Chunker` can be shared
/// across threads and planning is entirely a function of its inputs.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Chunker {
    policy: ChunkPolicy,
}

impl Chunker {
    /// A chunker with an explicit policy.
    #[must_use]
    pub fn new(policy: ChunkPolicy) -> Self {
        Self { policy }
    }

    /// The policy in force.
    #[must_use]
    pub fn policy(&self) -> &ChunkPolicy {
        &self.policy
    }

    /// A lazy iterator over the planned chunks.
    ///
    /// `plan` is `self.iter(..).collect()`. The boundaries are computed once,
    /// inside the call, and then handed out one at a time: a caller that wants
    /// to stream (slice, transcribe, drop) can `take(n)` without materialising
    /// a `Vec`, and the iterator carries no audio at any point.
    #[must_use]
    pub fn iter(&self, timeline: &SpeechTimeline, total: Time) -> ChunkIter {
        let merged = merge_segments(&timeline.segments, self.policy.merge_gap_ms);
        let capped = split_at_cap(&merged, self.policy.max_ms);
        let total_ms = total.as_millis().max(capped.last().map_or(0, |s| s.end));
        let raw = self.boundaries(&capped, total_ms);
        let chunks = self.pad_and_coalesce(raw, total_ms);
        ChunkIter {
            chunks: chunks.into_iter(),
        }
    }

    /// Every chunk, sorted, non-overlapping unless `overlap_ms > 0`, and
    /// within `[0, total]`.
    #[must_use]
    pub fn plan(&self, timeline: &SpeechTimeline, total: Time) -> Vec<Chunk> {
        self.iter(timeline, total).collect()
    }

    /// The greedy core: capped spans in, un-padded chunk ranges out.
    ///
    /// The cap is enforced **on the emitted cut**, not on the spans. This is not
    /// a redundancy with `split_at_cap`; it is the only place the cap can be
    /// broken. `split_at_cap` bounds each span's *length*, and a span's length
    /// plus the gap in front of it is what becomes a chunk's length, so a long
    /// silence followed by a long utterance produces a chunk that is
    /// `max_ms + gap` long. Measured on the 13.9-minute reference recording: a
    /// 24-second pause in front of a 37-second utterance produced a 61,300 ms
    /// chunk against a 60,000 ms cap, and a 2,000-case randomised sweep over
    /// span lengths and gaps did not reproduce it because the sweep's gaps
    /// topped out at 3 seconds.
    ///
    /// Termination: every iteration either advances `i` past at least one span
    /// or pushes a chunk and strictly increases `anchor`. Since `i` is bounded
    /// by `spans.len()` and `anchor` by `anchor + max_ms`, the loop is
    /// O(spans + chunks) and cannot spin.
    fn boundaries(&self, spans: &[Span], total_ms: i64) -> Vec<Span> {
        if spans.is_empty() {
            return Vec::new();
        }
        let target = self.policy.target_ms;
        let search = self.policy.boundary_search_ms;
        let max = self.policy.max_ms;

        let mut raw: Vec<Span> = Vec::new();
        let mut anchor = spans[0].start;
        let mut i = 0usize;

        while i < spans.len() {
            let is_last = i + 1 == spans.len();
            let want = if is_last {
                // The final span always closes the chunk; there is nothing after
                // it to align to.
                spans[i].end
            } else if spans[i].end - anchor >= target {
                self.choose_cut(spans, i, anchor, target, search, total_ms)
            } else {
                i += 1;
                continue;
            };
            // The cap, on the cut. `want` can exceed `anchor + max` because
            // `spans[i].start` may be well after `anchor` -- see the note above.
            let end = want.min(anchor.saturating_add(max)).max(anchor);
            raw.push(Span { start: anchor, end });
            anchor = end;
            // Advance past every span this chunk has *finished*, and stop at the
            // first one it has only partly consumed -- that remainder belongs to
            // the next chunk, and skipping it would drop audio.
            while i < spans.len() && spans[i].end <= anchor {
                i += 1;
            }
        }
        raw
    }

    /// The cut point after including spans `0..=i`.
    ///
    /// The midpoint of the pause that follows span `i`, or the *start* of that
    /// pause when its midpoint lies more than `search` past the target. The
    /// first qualifying gap is chosen rather than the widest: the first is the
    /// closest to `target`, and the widest can be arbitrarily far past it.
    fn choose_cut(
        &self,
        spans: &[Span],
        i: usize,
        anchor: i64,
        target: i64,
        search: i64,
        total_ms: i64,
    ) -> i64 {
        let gap_start = spans[i].end;
        let Some(next) = spans.get(i + 1) else {
            return gap_start.min(total_ms);
        };
        let gap_end = next.start.max(gap_start);
        if gap_end <= gap_start {
            // Adjacent spans, which only happens with `merge_gap_ms == 0`.
            // There is no pause to align to, so cut on the shared boundary.
            return gap_start.min(total_ms);
        }
        let mid = gap_start + (gap_end - gap_start) / 2;
        if mid - anchor <= target + search {
            mid.min(total_ms)
        } else {
            gap_start.min(total_ms)
        }
    }

    /// Pad, clip against the neighbour and the buffer, then enforce the floor.
    fn pad_and_coalesce(&self, raw: Vec<Span>, total_ms: i64) -> Vec<Chunk> {
        let pad = self.policy.pad_ms;
        let overlap = self.policy.overlap_ms;
        let mut chunks: Vec<Chunk> = Vec::with_capacity(raw.len());
        let mut cursor = 0i64;
        for s in raw {
            let start = (s.start - pad).max(0).max(cursor.saturating_sub(overlap));
            let end = (s.end + pad).min(total_ms).max(start);
            if end <= start {
                continue;
            }
            cursor = end;
            chunks.push(Chunk {
                start: Time(start),
                end: Time(end),
            });
        }
        coalesce_short(chunks, &self.policy)
    }

    /// Like [`Self::plan`], but also returns the un-padded, un-coalesced spans.
    ///
    /// Exists because the cap invariant is the one whose violation is a *memory*
    /// bug rather than a quality bug, and reproducing it needs the intermediate
    /// stage: the 13.9-minute reference recording produced a 61,300 ms chunk
    /// against a 60,000 ms cap that no hand-written case and no 2,000-case
    /// randomised sweep reproduced.
    #[must_use]
    pub fn plan_with_raw(&self, timeline: &SpeechTimeline, total: Time) -> (Vec<Span>, Vec<Chunk>) {
        let merged = merge_segments(&timeline.segments, self.policy.merge_gap_ms);
        let capped = split_at_cap(&merged, self.policy.max_ms);
        let total_ms = total.as_millis().max(capped.last().map_or(0, |s| s.end));
        let raw = self.boundaries(&capped, total_ms);
        let chunks = self.pad_and_coalesce(raw.clone(), total_ms);
        (raw, chunks)
    }
}

/// Merge any chunk below `min_ms` into a neighbour, as long as the result fits
/// in `max_ms`.
///
/// Two passes, and the second exists because the first cannot see the end of the
/// list:
///
/// * **Forward.** Accumulate, and keep accumulating while the current chunk is
///   still under the floor. A run of four 300 ms utterances therefore becomes
///   one chunk rather than four unusable ones.
/// * **Backward, once.** A short chunk at the *end* cannot absorb a following
///   chunk because there is none, so fold it back into its predecessor where
///   the cap allows.
///
/// A short chunk is only ever left when the material itself is short: the floor
/// cannot be met past the end of the media. That is the documented behaviour.
#[must_use]
pub fn coalesce_short(mut chunks: Vec<Chunk>, policy: &ChunkPolicy) -> Vec<Chunk> {
    let min = policy.min_ms;
    let max = policy.max_ms;
    if min <= 0 {
        return chunks;
    }

    let mut out: Vec<Chunk> = Vec::with_capacity(chunks.len());
    for c in chunks.drain(..) {
        match out.last_mut() {
            Some(prev) if prev.duration_ms() < min => {
                if c.end.as_millis() - prev.start.as_millis() <= max {
                    prev.end = c.end;
                    continue;
                }
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    chunks = out;

    while chunks.len() >= 2 {
        let n = chunks.len();
        if chunks[n - 1].duration_ms() >= min {
            break;
        }
        if chunks[n - 1].end.as_millis() - chunks[n - 2].start.as_millis() > max {
            break;
        }
        chunks[n - 2].end = chunks[n - 1].end;
        chunks.pop();
    }
    chunks
}

/// The lazy chunk iterator. See [`Chunker::iter`].
#[derive(Debug, Clone)]
pub struct ChunkIter {
    chunks: std::vec::IntoIter<Chunk>,
}

impl Iterator for ChunkIter {
    type Item = Chunk;

    fn next(&mut self) -> Option<Chunk> {
        self.chunks.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.chunks.size_hint()
    }
}

impl ExactSizeIterator for ChunkIter {}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: i64, end: i64) -> SpeechSegment {
        SpeechSegment {
            start: Time(start),
            end: Time(end),
            confidence: None,
        }
    }

    fn timeline(segments: Vec<SpeechSegment>) -> SpeechTimeline {
        let speech_duration_ms = segments
            .iter()
            .map(|s| s.end.as_millis() - s.start.as_millis())
            .sum();
        SpeechTimeline {
            segments,
            speech_duration_ms,
        }
    }

    fn policy() -> ChunkPolicy {
        ChunkPolicy::default()
    }

    #[test]
    fn merging_bridges_short_gaps_and_keeps_long_ones() {
        let s = vec![seg(0, 1_000), seg(1_200, 2_000), seg(5_000, 6_000)];
        let m = merge_segments(&s, 300);
        assert_eq!(m.len(), 2, "{m:?}");
        assert_eq!(
            m[0],
            Span {
                start: 0,
                end: 2_000
            }
        );
        assert_eq!(
            m[1],
            Span {
                start: 5_000,
                end: 6_000
            }
        );
    }

    #[test]
    fn merging_bridges_overlapping_segments_too() {
        let m = merge_segments(&[seg(0, 1_500), seg(1_000, 2_000)], 300);
        assert_eq!(
            m,
            vec![Span {
                start: 0,
                end: 2_000
            }]
        );
    }

    #[test]
    fn a_zero_length_segment_is_dropped_not_merged() {
        let m = merge_segments(&[seg(100, 100), seg(200, 300)], 300);
        assert_eq!(
            m,
            vec![Span {
                start: 200,
                end: 300
            }]
        );
    }

    #[test]
    fn splitting_at_the_cap_is_exact_arithmetic() {
        assert_eq!(
            split_at_cap(
                &[Span {
                    start: 0,
                    end: 600_000
                }],
                60_000
            ),
            (0..10)
                .map(|i| Span {
                    start: i * 60_000,
                    end: (i + 1) * 60_000
                })
                .collect::<Vec<_>>()
        );
        // A span already inside the cap is untouched.
        assert_eq!(
            split_at_cap(&[Span { start: 5, end: 100 }], 60_000),
            vec![Span { start: 5, end: 100 }]
        );
        // A remainder produces a short final piece rather than being lost.
        let s = split_at_cap(
            &[Span {
                start: 0,
                end: 130_000,
            }],
            60_000,
        );
        assert_eq!(s.len(), 3);
        assert_eq!(
            s[2],
            Span {
                start: 120_000,
                end: 130_000
            }
        );
        let total: i64 = s.iter().map(|x| x.len_ms()).sum();
        assert_eq!(total, 130_000, "splitting must not lose or duplicate audio");
        // A non-positive cap is a no-op rather than an infinite loop.
        assert_eq!(
            split_at_cap(&[Span { start: 0, end: 10 }], 0),
            vec![Span { start: 0, end: 10 }]
        );
    }

    #[test]
    fn an_empty_timeline_plans_nothing() {
        let c = Chunker::new(policy());
        assert!(c.plan(&timeline(vec![]), Time(600_000)).is_empty());
        assert!(c.plan(&timeline(vec![]), Time::ZERO).is_empty());
    }

    #[test]
    fn a_monologue_is_cut_at_the_cap_and_the_count_is_arithmetic() {
        // 10 minutes of unbroken speech with no pause anywhere. This is the
        // unbounded-chunk out-of-memory bug, and the count is exactly
        // ceil(600000 / 60000).
        let c = Chunker::new(policy());
        let tl = timeline(vec![seg(0, 600_000)]);
        let chunks = c.plan(&tl, Time(600_000));
        assert_eq!(chunks.len(), 10, "{chunks:?}");
        for w in chunks.windows(2) {
            assert!(w[0].end <= w[1].start, "chunks overlap: {w:?}");
        }
        // Every chunk but the first is exactly the cap, since the raw spans are
        // adjacent and the neighbour clamp eats the interior padding. The
        // first is the cap plus its trailing pad; its leading pad is clipped by
        // the buffer start.
        assert_eq!(chunks[0].duration_ms(), 60_060);
        for w in &chunks[1..chunks.len() - 1] {
            assert_eq!(w.duration_ms(), 60_000, "{w:?}");
        }
        // And the union covers the whole monologue.
        let covered: i64 = chunks.iter().map(|x| x.duration_ms()).sum();
        assert!(covered >= 600_000, "covered {covered}");
    }

    #[test]
    fn a_seven_hour_monologue_does_not_produce_a_single_chunk() {
        let c = Chunker::new(policy());
        let tl = timeline(vec![seg(0, 7_200_000)]);
        let chunks = c.plan(&tl, Time(7_200_000));
        assert_eq!(
            chunks.len(),
            120,
            "7.2e6 / 60000 = 120, got {}",
            chunks.len()
        );
        for ch in &chunks {
            assert!(ch.start < ch.end);
            assert!(ch.end.as_millis() <= 7_200_000);
        }
    }

    #[test]
    fn the_target_cuts_at_the_midpoint_of_the_following_pause() {
        let c = Chunker::new(policy());
        // Speech 0-30 s, pause 30-32 s, speech 32-70 s. The target is 30 s, so
        // the cut must land inside that pause.
        let tl = timeline(vec![seg(0, 30_000), seg(32_000, 70_000)]);
        let chunks = c.plan(&tl, Time(70_000));
        assert_eq!(chunks.len(), 2, "{chunks:?}");
        let cut = chunks[0].end.as_millis();
        assert!(
            (30_000..=32_000).contains(&cut),
            "the cut was {cut}, which is outside the pause"
        );
        // Strictly inside, so the next chunk gets the same slack: this is what
        // "never cut mid-word" reduces to arithmetically.
        assert!(cut > 30_000 + 60, "cut {cut} should be inside the pause");
        assert!(cut < 32_000, "cut {cut} should be inside the pause");
    }

    #[test]
    fn a_pause_far_past_the_target_is_not_waited_for() {
        // A two-minute pause immediately after the target. Cutting at its
        // midpoint would produce a 90-second chunk for a 30-second target.
        let c = Chunker::new(policy());
        let tl = timeline(vec![seg(0, 30_500), seg(150_000, 160_000)]);
        let chunks = c.plan(&tl, Time(160_000));
        assert!(chunks.len() >= 2);
        assert!(
            chunks[0].end.as_millis() <= 33_000,
            "chunk 0 ran to {}",
            chunks[0].end
        );
    }

    #[test]
    fn padding_is_applied_but_chunks_never_overlap_or_leave_the_buffer() {
        let c = Chunker::new(policy());
        let tl = timeline(vec![seg(10_000, 20_000), seg(100_000, 110_000)]);
        let chunks = c.plan(&tl, Time(120_000));
        assert_eq!(chunks[0].start.as_millis(), 10_000 - 60);
        for w in chunks.windows(2) {
            assert!(w[0].end <= w[1].start, "{w:?}");
        }
        for c in &chunks {
            assert!(c.start.as_millis() >= 0);
            assert!(c.end.as_millis() <= 120_000);
        }
    }

    #[test]
    fn total_covered_duration_is_exact_arithmetic() {
        let c = Chunker::new(policy());
        // The unambiguous case: one chunk, both pads inside the buffer.
        let tl = timeline(vec![seg(10_000, 20_000)]);
        let chunks = c.plan(&tl, Time(20_100));
        assert_eq!(
            chunks,
            vec![Chunk {
                start: Time(9_940),
                end: Time(20_060)
            }]
        );
        assert_eq!(chunks[0].duration_ms(), 10_000 + 2 * 60);

        // Three 40-second utterances with 2-second pauses. The interior pads
        // are *shared* between adjacent chunks -- giving each the full 60 ms
        // would transcribe the same silence twice -- and the leading pad is
        // clipped by the buffer start, so the covered total is the speech plus
        // the leading pad only.
        let tl = timeline(vec![
            seg(0, 40_000),
            seg(42_000, 82_000),
            seg(84_000, 120_000),
        ]);
        let chunks = c.plan(&tl, Time(120_200));
        assert_eq!(chunks.len(), 3, "{chunks:?}");
        let sum: i64 = chunks.iter().map(|x| x.duration_ms()).sum();
        assert_eq!(sum, 120_000 + 60, "{chunks:?}");
        // Every chunk is inside a target-sized band.
        for ch in &chunks {
            assert!(
                (30_000..=45_000).contains(&ch.duration_ms()),
                "{ch:?} is {} ms",
                ch.duration_ms()
            );
        }
    }

    #[test]
    fn a_chunk_is_never_shorter_than_the_receptive_field_when_material_exists() {
        // Four 300 ms utterances 400 ms apart. Each is under the 1 s floor, so
        // they must be coalesced rather than emitted as four unusable chunks.
        let c = Chunker::new(policy());
        let tl = timeline(vec![
            seg(0, 300),
            seg(700, 1_000),
            seg(1_400, 1_700),
            seg(2_100, 2_400),
        ]);
        let chunks = c.plan(&tl, Time(2_400));
        assert!(!chunks.is_empty());
        for ch in &chunks {
            assert!(
                ch.duration_ms() >= ChunkPolicy::MIN_CHUNK_MS,
                "chunk {ch:?} is {} ms, below the floor",
                ch.duration_ms()
            );
        }
        assert_segments_fully_covered(&tl, &chunks);
    }

    #[test]
    fn a_short_trailing_chunk_is_folded_back_into_its_predecessor() {
        // A 40-second utterance then a 50 ms blip. The second raw span is
        // 500 ms, which with padding is still under the 1 s floor, and it has
        // no successor to absorb -- so it must merge backwards rather than be
        // emitted at 500 ms.
        let c = Chunker::new(policy());
        let tl = timeline(vec![seg(0, 40_000), seg(40_500, 40_550)]);
        let chunks = c.plan(&tl, Time(40_600));
        assert_eq!(chunks.len(), 1, "{chunks:?}");
        assert!(chunks[0].duration_ms() >= ChunkPolicy::MIN_CHUNK_MS);
        assert_segments_fully_covered(&tl, &chunks);
    }

    #[test]
    fn the_final_chunk_may_be_short_when_the_media_is_short() {
        // The floor cannot be met because the material does not exist. The
        // alternative -- dropping the audio -- is worse, so the floor is clamped
        // to the buffer. Documented behaviour, asserted here.
        let c = Chunker::new(policy());
        let tl = timeline(vec![seg(0, 400)]);
        let chunks = c.plan(&tl, Time(500));
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].duration_ms() <= 500);
    }

    #[test]
    fn planning_is_a_pure_function_of_its_inputs() {
        let c = Chunker::new(policy());
        let tl = timeline(vec![
            seg(0, 25_000),
            seg(26_000, 55_000),
            seg(56_000, 90_000),
        ]);
        assert_eq!(c.plan(&tl, Time(90_000)), c.plan(&tl, Time(90_000)));
        assert_eq!(c.policy(), &policy());
    }

    #[test]
    fn the_iterator_matches_the_plan_and_can_be_taken_partially() {
        let c = Chunker::new(policy());
        let tl = timeline(vec![
            seg(0, 40_000),
            seg(42_000, 82_000),
            seg(84_000, 124_000),
        ]);
        let plan = c.plan(&tl, Time(124_000));
        let collected: Vec<Chunk> = c.iter(&tl, Time(124_000)).collect();
        assert_eq!(plan, collected);
        let first_two: Vec<Chunk> = c.iter(&tl, Time(124_000)).take(2).collect();
        assert_eq!(plan[..2].to_vec(), first_two);
        assert_eq!(c.iter(&tl, Time(124_000)).len(), plan.len());
    }

    #[test]
    fn the_output_is_always_sorted_non_overlapping_and_in_bounds() {
        // Adversarial input: overlapping segments, zero-length ones, a segment
        // far past the declared total, and one enormous span.
        let cases: Vec<(Vec<SpeechSegment>, i64)> = vec![
            (vec![], 1_000),
            (vec![seg(500, 500)], 1_000),
            (vec![seg(0, 1_000), seg(400, 900), seg(2_000, 3_000)], 4_000),
            (vec![seg(0, 900_000)], 900_000),
            (vec![seg(0, 10)], 10),
            (vec![seg(0, 5_000_000)], 5_000_000),
            (
                (0..200)
                    .map(|i| {
                        let s = i * 137 % 50_000;
                        seg(s, s + 900)
                    })
                    .collect(),
                60_000,
            ),
        ];
        let c = Chunker::new(policy());
        for (segments, total) in cases {
            let tl = timeline(segments);
            let chunks = c.plan(&tl, Time(total));
            for w in chunks.windows(2) {
                assert!(
                    w[0].end <= w[1].start,
                    "chunks overlap or are unsorted: {:?} then {:?}",
                    w[0],
                    w[1]
                );
            }
            for ch in &chunks {
                assert!(ch.start < ch.end, "empty chunk {ch:?}");
                assert!(ch.start.as_millis() >= 0, "{ch:?}");
                assert!(ch.end.as_millis() <= total, "{ch:?} vs total {total}");
            }
        }
    }

    #[test]
    fn chunks_agree_with_the_core_range_query() {
        // The cross-check the contract asks for: every speech segment is
        // covered *entirely* by exactly one chunk, and the query
        // `SpeechTimeline::overlapping` agrees with the plan.
        let mut segments = Vec::new();
        let mut at = 0i64;
        for i in 0..200 {
            let dur = 1_000 + (i as i64 % 4) * 900;
            let gap = 350 + (i as i64 % 7) * 400;
            segments.push(seg(at, at + dur));
            at += dur + gap;
        }
        let tl = timeline(segments);
        let total = at + 5_000;
        let c = Chunker::new(policy());
        let chunks = c.plan(&tl, Time(total));

        assert!(
            chunks.len() > 3,
            "expected several chunks, got {}",
            chunks.len()
        );
        for s in &tl.segments {
            let covering: Vec<&Chunk> = chunks
                .iter()
                .filter(|ch| {
                    s.start.as_millis() >= ch.start.as_millis()
                        && s.end.as_millis() <= ch.end.as_millis()
                })
                .collect();
            assert_eq!(
                covering.len(),
                1,
                "segment {s:?} is covered by {} chunks, not 1",
                covering.len()
            );
            // And the core query agrees.
            assert_eq!(tl.overlapping(s.start, s.end).count(), 1);
        }
        // The query never returns a segment that does not actually overlap.
        for ch in &chunks {
            for s in tl.overlapping(ch.start, ch.end) {
                assert!(
                    s.start.as_millis() < ch.end.as_millis()
                        && s.end.as_millis() > ch.start.as_millis(),
                    "overlapping() returned {s:?} for {ch:?}, which it does not overlap"
                );
            }
        }
        // Every chunk contains at least one segment: no chunk of pure silence.
        for ch in &chunks {
            assert!(
                tl.overlapping(ch.start, ch.end).next().is_some(),
                "chunk {ch:?} contains no speech at all"
            );
        }
    }

    #[test]
    fn a_nonzero_overlap_is_honoured_and_bounded() {
        let p = ChunkPolicy {
            overlap_ms: 500,
            ..ChunkPolicy::default()
        };
        let c = Chunker::new(p);
        let tl = timeline(vec![
            seg(0, 40_000),
            seg(41_000, 80_000),
            seg(81_000, 120_000),
        ]);
        let chunks = c.plan(&tl, Time(120_000));
        assert!(chunks.len() >= 2);
        for w in chunks.windows(2) {
            assert!(
                w[0].end > w[1].start,
                "a 500 ms overlap should make consecutive chunks overlap: {w:?}"
            );
            assert!(w[0].end.as_millis() - w[1].start.as_millis() <= 500 + 2 * 60);
        }
    }

    #[test]
    fn coalescing_is_a_no_op_on_already_valid_chunks() {
        let ok = vec![
            Chunk {
                start: Time(0),
                end: Time(30_000),
            },
            Chunk {
                start: Time(30_000),
                end: Time(60_000),
            },
        ];
        assert_eq!(coalesce_short(ok.clone(), &policy()), ok);
        // A zero minimum disables it entirely.
        let p = ChunkPolicy {
            min_ms: 0,
            ..ChunkPolicy::default()
        };
        assert_eq!(coalesce_short(ok.clone(), &p), ok);
    }

    #[test]
    fn coalescing_never_exceeds_the_cap() {
        let p = policy();
        // 40 short, non-overlapping, adjacent chunks: every one is under the
        // 1 s floor, so coalescing fires repeatedly and must still respect the
        // 60 s cap and produce a sorted, non-overlapping result.
        let short: Vec<Chunk> = (0..40)
            .map(|i| Chunk {
                start: Time(i * 300),
                end: Time(i * 300 + 300),
            })
            .collect();
        let out = coalesce_short(short, &p);
        assert!(out.len() < 40, "coalescing did nothing");
        for ch in &out {
            assert!(ch.duration_ms() <= p.max_ms, "{ch:?}");
        }
        for w in out.windows(2) {
            assert!(w[0].end <= w[1].start, "{out:?}");
        }
    }

    #[test]
    fn span_helpers_are_total() {
        assert!(Span { start: 5, end: 5 }.is_empty());
        assert!(!Span { start: 5, end: 6 }.is_empty());
        assert_eq!(Span { start: 5, end: 5 }.len_ms(), 0);
        assert_eq!(Span { start: 0, end: 100 }.len_ms(), 100);
    }

    #[test]
    fn a_long_silence_before_a_long_utterance_still_respects_the_cap() {
        // The bug the reference recording found. `split_at_cap` bounds each
        // span's *length*; a chunk's length is a span's length plus the gap in
        // front of it, so a 24-second pause in front of a 37-second utterance
        // produced a 61,300 ms chunk against a 60,000 ms cap. The randomised
        // sweep in `no_plan_ever_exceeds_the_cap` did not find it because its
        // gaps topped out at 3 seconds.
        let c = Chunker::new(policy());
        let tl = timeline(vec![seg(0, 20_000), seg(44_020, 81_320)]);
        let chunks = c.plan(&tl, Time(85_000));
        for ch in &chunks {
            assert!(
                ch.duration_ms() <= policy().max_ms + 2 * policy().pad_ms,
                "chunk {ch:?} is {} ms",
                ch.duration_ms()
            );
        }
        // And nothing is lost: not one sample of the 37-second utterance falls in
        // a hole, even though the cap had to cut it into two chunks, each of
        // which carries its part whole.
        assert_segments_fully_covered(&tl, &chunks);
    }

    #[test]
    fn no_plan_ever_exceeds_the_cap() {
        // The cap is the one bound whose violation is a *memory* bug rather than
        // a quality bug, so it is checked over a randomised sweep rather than on
        // a handful of hand-written cases.
        //
        // A deterministic xorshift, so a failure is reproducible and the sweep
        // does not flake.
        let mut rng: u64 = 0x5EED_1234;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let c = Chunker::new(policy());
        for case in 0..2_000u32 {
            let n = 1 + (next() % 60) as usize;
            let mut segments = Vec::with_capacity(n);
            let mut at = (next() % 5_000) as i64;
            for _ in 0..n {
                let dur = 1 + (next() % 8_000) as i64;
                // Gaps from zero to 40 s: below, around and far above the merge
                // threshold, because a gap larger than the cap is the case that
                // a span-length bound does not cover.
                let gap = (next() % 40_000) as i64;
                segments.push(seg(at, at + dur));
                at += dur + gap;
            }
            let total = at + 5_000;
            let tl = timeline(segments);
            let chunks = c.plan(&tl, Time(total));
            for ch in &chunks {
                assert!(
                    ch.duration_ms() <= policy().max_ms + 2 * policy().pad_ms,
                    "case {case}: chunk {ch:?} is {} ms against a {} ms cap \
                     (+{} ms padding each side)",
                    ch.duration_ms(),
                    policy().max_ms,
                    policy().pad_ms
                );
            }
            // A randomised plan must also never lose, hole, or duplicate audio.
            assert_segments_fully_covered(&tl, &chunks);
        }
    }

    #[test]
    fn chunk_geometry_helpers_are_half_open() {
        let a = Chunk {
            start: Time(0),
            end: Time(100),
        };
        let b = Chunk {
            start: Time(50),
            end: Time(150),
        };
        let c = Chunk {
            start: Time(100),
            end: Time(200),
        };
        assert_eq!(a.duration_ms(), 100);
        assert!(a.contains(Time(0)));
        assert!(a.contains(Time(99)));
        assert!(!a.contains(Time(100)));
        assert!(a.overlaps(&b));
        // Half-open: [0,100) and [100,200) share no audio.
        assert!(!a.overlaps(&c));
        assert_eq!(
            a.intersect(&b),
            Some(Chunk {
                start: Time(50),
                end: Time(100)
            })
        );
        assert_eq!(a.intersect(&c), None);
    }

    /// Every sample of every speech segment is inside some chunk.
    ///
    /// The invariant is *union* coverage, not "each segment is inside a single
    /// chunk", and the difference is not a weakening: `boundaries` clamps the
    /// emitted cut to `anchor + max_ms`, so a span that begins later than that
    /// -- or one long enough to be divided by `split_at_cap` -- is deliberately
    /// carried across two chunks, and each piece is still wholly inside one of
    /// them. What must never happen is a *hole*: audio that the plan skips over
    /// while reporting a chunk for it. So this walks each segment and requires
    /// the chunks to pick up exactly where the previous one stopped, with no
    /// gap anywhere inside the segment's own span.
    fn assert_segments_fully_covered(tl: &SpeechTimeline, chunks: &[Chunk]) {
        for s in &tl.segments {
            let (a, b) = (s.start.as_millis(), s.end.as_millis());
            let mut at = a;
            for ch in chunks {
                let (cs, ce) = (ch.start.as_millis(), ch.end.as_millis());
                if ce <= at || cs >= b {
                    continue;
                }
                assert!(
                    cs <= at,
                    "segment {s:?} has a hole: covered to {at} but the next chunk starts at {cs}"
                );
                at = ce;
            }
            assert!(
                at >= b,
                "segment {s:?} is not covered to its end ({b}); the last chunk touching it ends at {at}"
            );
        }
    }
}

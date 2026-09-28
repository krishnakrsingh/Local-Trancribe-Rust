//! The grouping algorithm.
//!
//! # What it does, in one paragraph
//!
//! Takes a transcript whose segments are one-per-word, walks the words in time
//! order, and cuts them into segments at a pause, at a speaker turn, or when a
//! length cap is reached — whichever comes first. It changes **which segment a
//! word belongs to** and nothing else: no word's text, start, end, confidence or
//! normalisation is read for anything but grouping, and none of them is
//! written.
//!
//! # The shape of the walk
//!
//! ```text
//!   flatten ──▶ stable sort by (start, end) ──▶ label each word's speaker
//!                                                  │
//!                                                  ▼
//!                                   for each word after the first:
//!                                      hard gap?  pause?  turn?  cap?
//!                                                  │
//!                                             break or not
//!                                                  │
//!                                                  ▼
//!                                   build segments, repair spans, assign ids
//! ```
//!
//! Flattening first is what makes the "every word exactly once" property
//! checkable: there is exactly one place words are enumerated, so a duplicate
//! or a drop has to be a bug in that one loop rather than a mistake repeated
//! across several.
//!
//! # Why the sort
//!
//! Segments arrive sorted by start, but a chunk that overlaps its predecessor
//! can produce a segment starting before the previous one ended, and a word
//! inside a segment can be out of order. The output must be sorted and
//! non-overlapping, so the flat word list is **stably** sorted by
//! `(start, end)`. Stability matters: two words with identical times must keep
//! their relative order or the text of a segment changes. On real ASR output
//! the sort is a no-op, and [`GroupingReport::reordered_words`] reports how
//! many positions it actually moved so a non-zero count is visible rather than
//! silent.
//!
//! # Why grouping happens *before* attribution
//!
//! This is the load-bearing ordering decision in the whole crate.
//!
//! Speaker attribution is an **overlap-weighted majority vote** over a
//! segment's time span, and `vaani_diarization::attribution` is where that
//! lives. Applied to one-word segments it is 1,913 independent votes, and a
//! single word landing on the wrong side of a clustering boundary can outvote
//! its neighbours — which is a real failure mode, not a theoretical one, and
//! it is why the shipped transcript attributed 98 % of segments and still read
//! as noise.
//!
//! Grouping by turn first makes the vote a **per-turn** decision. A turn is
//! several seconds of one voice; the vote is then over an interval that is
//! almost entirely inside one diarization span, so it lands on the right
//! speaker almost by construction. The number of decisions falls by an order of
//! magnitude and each one is easier to get right.
//!
//! The pipeline order is therefore: **group, then attribute.** The two steps
//! are separable on purpose — [`plan_groups`] takes an optional
//! `DiarizationResult` for the *break* decision, and the attribution vote stays
//! in `vaani-diarization` where it is already tested — so this crate does not
//! contain a second implementation of the vote that could drift from the first.
//!
//! # Punctuation
//!
//! The policy does not use punctuation, because there is none to use. SraVaani
//! 1.0 is a non-punctuation ASR model: `tokens_to_segments` concatenates
//! SentencePiece pieces with no separator and inserts no marks, and
//! `vaani_core::Word::text` documents the field as the "surface form as
//! recognised". A rule keyed on `.` or `?` would therefore never fire while
//! looking like it worked, which is worse than not having it. See
//! [`word_text_contains_no_punctuation`] for the check that keeps this honest,
//! and the crate tests for the assertion on real output.

use vaani_core::pipeline::DiarizationResult;
use vaani_core::transcript::{Segment, Speaker, Time, Transcript, Word};

use crate::ids;
use crate::policy::GroupingPolicy;
use crate::report::{BreakReason, GroupingReport};

/// The shortest span a grouped segment is allowed to have, in milliseconds.
///
/// A segment must be selectable, seekable and exportable as a caption, and all
/// three want a non-zero span. One millisecond is the smallest value that
/// satisfies them and the smallest value that is honestly describable as a
/// duration at all. It is a floor on the *segment's* span; it never moves a
/// word's own times.
pub const MIN_SEGMENT_MS: i64 = 1;

/// The absolute media window a transcript's times must lie inside.
///
/// # Why this is a parameter and not `Transcript::duration`
///
/// `Transcript::duration` is a **length** — "total media duration" — while
/// The absolute media window a transcript's times must lie inside.
///
/// # This is now a re-export, and that is the point
///
/// `MediaBounds` used to be defined here, as a **workaround**.
///
/// `Transcript::duration` is a **length** — "total media duration" — while
/// `Segment::start` and `Word::start` are **absolute** offsets from the start of
/// the media. Comparing one against the other is a category error, and it is one
/// this crate made: clamping a segment's absolute end to a length crushes every
/// segment in a transcript that does not start at zero.
///
/// That is not hypothetical. `vaani-qa`'s conformance harness runs every stage
/// against audio that starts at **+5 minutes**, precisely so a stage that emits
/// chunk-relative times is caught. Its seeded transcript carries
/// `duration = 300_000` (a length) and absolute times running to 600 000, and
/// clamping against the length produced three segments all pinned to
/// `300000..300001` with their words lying outside them. Three conformance
/// checks failed at once. The arithmetic was plausible; it was wrong.
///
/// The workaround existed because the contract crate had no honest way to say
/// "here is the media", so each feature crate invented one. `vaani-core` now
/// owns the type — `MediaWindow`, in `docs/architecture/TIMESTAMP_CONTRACT.md` —
/// and this alias exists so that every existing call site keeps compiling while
/// there is exactly one such type in the workspace.
///
/// The stronger fix is in the contract: `Transcript::duration` is an
/// `Option<MediaLength>`, which does not compare with a `Time` at all, so the
/// mistake no longer compiles rather than merely being discouraged.
///
/// Constructors come from the contract type: `MediaWindow::of_buffer` (the
/// pipelines), `MediaWindow::new` (an extent stated outright), and
/// `MediaWindow::of_length` (assumes a zero origin, and says so in its name).
pub use vaani_core::transcript::MediaWindow as MediaBounds;

/// One word, flattened out of its segment, plus where it came from.
#[derive(Debug, Clone)]
struct FlatWord {
    word: Word,
    /// Index into `Transcript::segments` of the segment this word came from.
    source: usize,
    /// Index of this word **within that source segment**.
    ///
    /// Recorded here rather than derived later by scanning backwards, because
    /// the stable sort can interleave the words of two source segments: a
    /// segment that starts earlier but has a word with a later time will have
    /// that word sorted past its neighbour's. Counting backwards over the flat
    /// list then stops at the wrong place and two groups can be handed the same
    /// offset, which is a duplicate segment id. The identity is known here and
    /// nowhere else, so it is captured here.
    index_in_source: usize,
    /// The attributed speaker for this word, or `None` when no diarization span
    /// covers it. `None` is the normal ASR-only state and is never an error.
    label: Option<usize>,
    /// `display_text().chars().count()`, cached because the char cap consults
    /// it on every candidate word.
    chars: usize,
}

/// One planned output segment, before it is written into a transcript.
///
/// Exposed so that a settings panel can answer "what would this policy do?"
/// without mutating anything, and so the property tests can assert on the plan
/// rather than on a transcript that has already been rebuilt around it.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedGroup {
    /// The group's id. See [`crate::ids`] for the mapping.
    pub id: String,
    /// Segment start, derived from the first member word's start.
    pub start: Time,
    /// Segment end, derived from the last member word's end.
    pub end: Time,
    /// The group's speaker, or `None` in ASR-only mode.
    pub speaker: Option<usize>,
    /// Mean of the member words' confidences, or `None` when none reported one.
    pub confidence: Option<f32>,
    /// The first non-`None` language among the source segments.
    pub language: Option<String>,
    /// The words, in order, with their own times untouched.
    pub words: Vec<Word>,
    /// Whether this finished group violates an enabled length cap. Counted in
    /// [`GroupingReport::segments_over_cap`], and the reason the runt guard
    /// exists.
    pub over_cap: bool,
}

impl PlannedGroup {
    /// The group's text, joining words with single spaces — the same string
    /// [`Segment::text`] will produce once this becomes a segment.
    #[must_use]
    pub fn text(&self) -> String {
        if self.words.is_empty() {
            return String::new();
        }
        let mut out = String::new();
        for (i, w) in self.words.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            out.push_str(w.display_text());
        }
        out
    }

    /// The span in milliseconds, never negative.
    #[must_use]
    pub fn duration_ms(&self) -> i64 {
        (self.end.as_millis() - self.start.as_millis()).max(0)
    }
}

/// The full result of planning: the groups, and what planning counted.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupingPlan {
    /// The groups, in time order.
    pub groups: Vec<PlannedGroup>,
    /// Counts of every break reason, every repair, and the before/after totals.
    pub report: GroupingReport,
}

impl GroupingPlan {
    /// Write the plan into `transcript`, replacing its segments.
    ///
    /// Everything else on the transcript — its id, duration, mode, engine stamp
    /// and `edited` flag — is left exactly as it was, because grouping changes
    /// the segmentation and nothing else.
    pub fn apply(self, transcript: &mut Transcript) {
        transcript.segments = self
            .groups
            .into_iter()
            .map(|g| Segment {
                id: g.id,
                start: g.start,
                end: g.end,
                speaker: g.speaker,
                language: g.language,
                words: g.words,
                confidence: g.confidence,
            })
            .collect();
    }
}

/// Group `transcript` in place, without speaker information.
///
/// No media bounds are applied, because this entry point does not know where
/// the media starts. Use [`group_within`] when the caller has an `AudioBuffer`;
/// see [`MediaBounds`] for why the distinction matters.///
/// Existing `Segment::speaker` values are **honoured**: a group's speaker is the
/// speaker owning the most of its words, so calling this on a transcript that
/// has already been attributed does not throw the attribution away. Use
/// [`group_with_diarization`] when a `DiarizationResult` is available — it also
/// *breaks* on speaker turns, which is the point.
///
/// # Errors
///
/// `Err` when `policy` is not [`GroupingPolicy::is_valid`]. An invalid policy is
/// a settings error, and silently producing a transcript from it is exactly the
/// class of defect this crate exists to remove; there is no sensible fallback
/// for a pause threshold set above the hard-gap ceiling.
pub fn group(
    transcript: &mut Transcript,
    policy: &GroupingPolicy,
) -> vaani_core::Result<GroupingReport> {
    let plan = plan_groups(transcript, None, None, policy)?;
    let report = plan.report;
    plan.apply(transcript);
    Ok(report)
}

/// [`group`], with the absolute media window the segments must lie inside.
///
/// This is the entry point a pipeline should call: it has the `AudioBuffer`, and
/// [`MediaWindow::of_buffer`](vaani_core::transcript::MediaWindow::of_buffer)
/// turns it into a window without the caller having to think about lengths
/// versus absolute offsets.
pub fn group_within(
    transcript: &mut Transcript,
    bounds: MediaBounds,
    policy: &GroupingPolicy,
) -> vaani_core::Result<GroupingReport> {
    let plan = plan_groups(transcript, None, Some(bounds), policy)?;
    let report = plan.report;
    plan.apply(transcript);
    Ok(report)
}

/// Group `transcript` in place **and** set each group's speaker, from `d`.
///
/// No media bounds are applied; see [`group_within`].
///
/// Grouping happens first and the speaker is a per-group decision; see the
/// module documentation for why that ordering is the whole point. A caller that
/// intends to run `vaani_diarization::attribute_speakers` afterwards — which
/// applies the careful overlap-weighted vote over each grouped span — should do
/// so, and that call wins; the labels written here are for a caller that has the
/// diarization result but not the diarization crate.
///
/// Speaker names already on the transcript are preserved, and `0..num_speakers`
/// is registered so every index this function writes resolves against
/// `Transcript::speakers`.
///
/// # Errors
///
/// As [`group`].
pub fn group_with_diarization(
    transcript: &mut Transcript,
    d: &DiarizationResult,
    policy: &GroupingPolicy,
) -> vaani_core::Result<GroupingReport> {
    group_with_diarization_within(transcript, d, None, policy)
}

/// [`group_with_diarization`], with the absolute media window the segments must
/// lie inside.
///
/// This is the entry point a pipeline should call, and it pairs with
/// `vaani_diarization::attribute_speakers` running immediately afterwards.
pub fn group_with_diarization_within(
    transcript: &mut Transcript,
    d: &DiarizationResult,
    bounds: Option<MediaBounds>,
    policy: &GroupingPolicy,
) -> vaani_core::Result<GroupingReport> {
    let plan = plan_groups(transcript, Some(d), bounds, policy)?;
    let report = plan.report;
    let num_speakers = d.num_speakers;
    // Register before applying, so an index a label produces always resolves.
    // `add_speaker` replaces the whole record by index, so a name the user set
    // has to be carried across explicitly rather than re-registered blank.
    let existing: Vec<Speaker> = transcript
        .speakers
        .iter()
        .filter(|s| s.index < num_speakers)
        .cloned()
        .collect();
    for s in existing {
        transcript.add_speaker(s);
    }
    for i in 0..num_speakers {
        if !transcript.speakers.iter().any(|s| s.index == i) {
            transcript.add_speaker(Speaker::new(i));
        }
    }
    plan.apply(transcript);
    transcript.recompute_speaker_shares();
    Ok(report)
}

/// Plan the grouping of `transcript` without changing it.
///
/// Pure: the same transcript, diarization result and policy always produce the
/// same plan, which is what makes the two pipelines comparable and the
/// property tests meaningful. `d` is optional; without it no break is ever a
/// speaker turn, and every group is attributed from the source segments'
/// existing speakers.
///
/// # Errors
///
/// As [`group`].
pub fn plan_groups(
    transcript: &Transcript,
    d: Option<&DiarizationResult>,
    bounds: Option<MediaBounds>,
    policy: &GroupingPolicy,
) -> vaani_core::Result<GroupingPlan> {
    plan_inner(transcript, d, bounds, policy, true)
}

/// [`plan_groups`], with the reporting log line suppressed.
///
/// The suppression is not an optimisation. [`threshold_sweep`] re-plans once per
/// candidate threshold, and the planner reports what it did; without this the
/// sweep would log a summary line per threshold *and* recurse into another sweep
/// from each of them, which is an unbounded recursion and a stack overflow. One
/// flag, at the one place that can break the cycle.
fn plan_inner(
    transcript: &Transcript,
    d: Option<&DiarizationResult>,
    bounds: Option<MediaBounds>,
    policy: &GroupingPolicy,
    report_outcome: bool,
) -> vaani_core::Result<GroupingPlan> {
    if !policy.is_valid() {
        return Err(vaani_core::err::internal().with_cause(format!(
            "the grouping policy is not self-consistent: {policy:?}"
        )));
    }
    let started = std::time::Instant::now();

    let mut report = GroupingReport::default();
    let words = flatten(transcript, d, &mut report);
    report.words_in = words.len();
    report.words_out = words.len();
    if words.is_empty() {
        report.segments_out = 0;
        return Ok(GroupingPlan {
            groups: Vec::new(),
            report,
        });
    }

    // The break decisions only need, per word: the gap to its predecessor, its
    // speaker label, and whether the label differs. Precomputing the *natural*
    // breaks lets the runt guard look ahead in O(1) instead of rescanning.
    let natural = natural_breaks(&words, policy);
    let cuts = cut_points(&words, &natural, policy, &mut report);
    let groups = build_groups(transcript, &words, &cuts, policy, bounds, &mut report);

    debug_assert_eq!(
        groups.iter().map(|g| g.words.len()).sum::<usize>(),
        report.words_in,
        "grouping must preserve every word exactly once"
    );
    report.segments_out = groups.len();
    report.words_out = groups.iter().map(|g| g.words.len()).sum();
    if report_outcome {
        log_outcome(transcript, &groups, &report, policy, started.elapsed());
    }
    // A wordless source segment is *dropped*, not merged, so the two do not add
    // up when any exist. Count the segments that actually contributed a word and
    // subtract the output, which is the honest definition either way.
    let with_words = report.segments_in - report.source_segments_without_words;
    report.source_segments_merged = with_words.saturating_sub(report.segments_out);
    Ok(GroupingPlan { groups, report })
}

#[derive(Debug, Clone, Copy)]
struct Cut {
    at: usize,
    reason: BreakReason,
}

/// Flatten every word out of every segment, in time order, with a speaker
/// label per word.
fn flatten(
    transcript: &Transcript,
    d: Option<&DiarizationResult>,
    report: &mut GroupingReport,
) -> Vec<FlatWord> {
    report.segments_in = transcript.segments.len();
    report.source_segments_without_words = transcript
        .segments
        .iter()
        .filter(|s| s.words.is_empty())
        .count();

    let mut out: Vec<FlatWord> = Vec::with_capacity(
        transcript
            .segments
            .iter()
            .map(|s| s.words.len())
            .sum::<usize>(),
    );
    for (i, seg) in transcript.segments.iter().enumerate() {
        for (k, word) in seg.words.iter().enumerate() {
            // Labelling happens whenever a diarization result is available, not
            // only when the turn rule is on: with the rule off the group still
            // wants a speaker, and a policy that was edited between runs must not
            // silently change what "no diarization" means.
            let label = d.map(|result| label_for(result, word)).unwrap_or(None);
            out.push(FlatWord {
                chars: word.display_text().chars().count(),
                word: word.clone(),
                source: i,
                index_in_source: k,
                label,
            });
        }
    }

    // Stable, so equal times keep their order and the text is unaffected.
    let before: Vec<(i64, i64)> = out
        .iter()
        .map(|w| (w.word.start.as_millis(), w.word.end.as_millis()))
        .collect();
    out.sort_by_key(|w| (w.word.start, w.word.end));
    report.reordered_words = before
        .iter()
        .enumerate()
        .filter(|(i, k)| {
            let now = (out[*i].word.start.as_millis(), out[*i].word.end.as_millis());
            now != **k
        })
        .count();
    out
}

/// The speaker covering the middle of `word`, or `None`.
///
/// The midpoint rather than the start: a word that straddles a diarization
/// boundary is claimed by the voice that holds more of it, and a start-time
/// lookup would claim it for whoever happened to own the first millisecond.
/// This is the *break* decision only — a local, cheap question about whether
/// the voice changed. The *attribution* decision is a separate, careful
/// overlap-weighted vote that lives in `vaani_diarization::attribution`.
fn label_for(d: &DiarizationResult, word: &Word) -> Option<usize> {
    let lo = word.start.as_millis();
    let hi = word.end.as_millis();
    let mid = if hi > lo { lo + (hi - lo) / 2 } else { lo };
    // `speaker_at` resolves an overlap to the shortest containing span and hands
    // back whatever index that span names. `DiarizationResult` documents that a
    // span may name a speaker the result never registered, and
    // `DiarizationResult::speaker_durations` ignores such a span rather than
    // panicking. Grouping follows the same rule, so a stray span cannot
    // manufacture a `Speaker 8` in the UI.
    d.speaker_at(Time::from_millis(mid))
        .filter(|speaker| *speaker < d.num_speakers)
}

/// Precompute, for every word index, whether a *natural* break (hard gap, pause
/// or speaker turn) falls before it.
fn natural_breaks(words: &[FlatWord], policy: &GroupingPolicy) -> Vec<Option<BreakReason>> {
    let mut out = vec![None; words.len()];
    for i in 1..words.len() {
        let gap = words[i].word.start.as_millis() - words[i - 1].word.end.as_millis();
        out[i] = if gap >= policy.max_gap_ms {
            Some(BreakReason::HardGap)
        } else if gap >= policy.pause_break_ms {
            Some(BreakReason::Pause)
        } else if policy.break_on_speaker_turn
            && words[i].label.is_some()
            && words[i].label != words[i - 1].label
        {
            Some(BreakReason::SpeakerTurn)
        } else {
            None
        };
    }
    out
}

/// The first natural break strictly after `from`, if any.
fn next_natural_break(natural: &[Option<BreakReason>], from: usize) -> Option<usize> {
    natural
        .iter()
        .enumerate()
        .skip(from + 1)
        .find(|(_, r)| r.is_some())
        .map(|(i, _)| i)
}

/// Which cap, if any, adding word `i` to the group starting at `group_start`
/// would exceed. `None` means the word fits.
///
/// A cap is never reported for the *first* word of a group: a single word longer
/// than `max_chars` must still form a segment of its own rather than being
/// split, because half a word on screen is unreadable.
fn cap_break(
    words: &[FlatWord],
    group_start: usize,
    i: usize,
    chars: usize,
    policy: &GroupingPolicy,
) -> Option<BreakReason> {
    if i == group_start {
        return None;
    }
    if policy.max_words > 0 && i - group_start + 1 > policy.max_words {
        return Some(BreakReason::WordCap);
    }
    if policy.max_chars > 0 && chars + 1 + words[i].chars > policy.max_chars {
        return Some(BreakReason::CharCap);
    }
    if policy.max_duration_ms > 0 {
        let span = words[i].word.end.as_millis() - words[group_start].word.start.as_millis();
        if span > policy.max_duration_ms {
            return Some(BreakReason::DurationCap);
        }
    }
    None
}

/// Walk the words once and decide every cut.
fn cut_points(
    words: &[FlatWord],
    natural: &[Option<BreakReason>],
    policy: &GroupingPolicy,
    report: &mut GroupingReport,
) -> Vec<Cut> {
    let mut cuts: Vec<Cut> = Vec::new();
    let mut group_start = 0usize;
    // The first word's own length is part of the group's budget. Starting at zero
    // would let a group exceed the cap by exactly one word, which is an
    // off-by-one in the one rule the caption exporter also depends on.
    let mut chars = words[0].chars;
    for i in 1..words.len() {
        let cut = if let Some(reason) = natural[i] {
            Some(Cut { at: i, reason })
        } else {
            cap_break(words, group_start, i, chars, policy).and_then(|reason| {
                // A cap firing this close to a natural boundary would
                // manufacture a runt segment — the exact one-word output this
                // crate exists to remove. Defer to the boundary instead: the cut
                // is *not* taken here, it is taken at the natural boundary, and
                // it is counted so the deferral is visible rather than silent.
                // With no natural boundary ahead there is nothing to defer to,
                // so the cap fires.
                if next_natural_break(natural, i)
                    .is_some_and(|next| next - i < policy.min_segment_words)
                {
                    report.deferred_by_reason(reason);
                    None
                } else {
                    Some(Cut { at: i, reason })
                }
            })
        };
        match cut {
            Some(c) => {
                cuts.push(c);
                group_start = i;
                chars = words[i].chars;
            }
            None => chars += 1 + words[i].chars,
        }
    }
    // A cut can land between two *overlapping* words, which would make the two
    // groups' spans overlap — and the only way to repair that by moving a
    // segment boundary is to orphan a word outside its own segment. So the cuts
    // are made disjoint here, before any segment is built, and the break counters
    // are incremented here rather than in the loop above: a break the policy
    // asked for but `disjoint_cuts` had to drop is not a break, and a report that
    // counted it as one would be lying in the one place a caller looks to find
    // out what grouping did. See `disjoint_cuts` for what the cost is.
    disjoint_cuts(words, cuts, report)
}

/// Move cut points so that no group's last word ends after the next group's
/// first word starts.
///
/// # Why this is not optional
///
/// Words can overlap: a chunk boundary can put a word partly before the
/// previous one ended, and `Transcript::push_chunk` cannot clip it without
/// losing a recognised word. A cut between two such words produces two groups
/// whose *spans* overlap, and the two obvious repairs are both wrong. Extending
/// the earlier group to the later one's start is a no-op; pushing the later
/// group's start forward to the earlier one's end leaves that group's own first
/// word **outside its own segment** — which is precisely what A10's conformance
/// suite caught, and precisely what a click-to-seek depends on.
///
/// So the cut is moved back over the offending word, putting it in the later
/// group. That makes each group's word-time range disjoint from the next, and
/// then every group's segment span, derived from its own words, is disjoint
/// from the next with no repair at all.
///
/// The cost: a group can lose its last word to the following one, and a cut
/// between the very first and second word cannot be made disjoint at all and is
/// **dropped** — those two words must share a segment, which is the only
/// non-overlapping way to represent them.
fn disjoint_cuts(words: &[FlatWord], cuts: Vec<Cut>, report: &mut GroupingReport) -> Vec<Cut> {
    let overlaps =
        |at: usize| words[at - 1].word.end.as_millis() > words[at].word.start.as_millis();
    let mut out: Vec<Cut> = Vec::with_capacity(cuts.len());
    for cut in cuts {
        // The group's start is the previous surviving cut, or 0. A cut cannot be
        // moved before that without emptying the group, and an empty group is
        // the one thing this crate must never produce.
        let group_start = out.last().map_or(0, |c: &Cut| c.at);
        let mut at = cut.at;
        while at > group_start + 1 && overlaps(at) {
            at -= 1;
        }
        if at <= group_start || overlaps(at) {
            // Either there is no room left to move the cut, or the group's own
            // first word already overlaps this one so no position between them
            // can be made disjoint. Both mean the same thing: these words have
            // to share a segment, which is the only representation of
            // overlapping words that is also non-overlapping. The break the
            // policy asked for is lost, and that is the honest cost.
            //
            // Counted as *dropped*, not as a break: the report's arithmetic
            // (`breaks == segments - 1`) is checked by the property tests, and a
            // counter that claimed a break that did not happen would be a lie
            // in the one place a caller looks to find out what grouping did.
            report.breaks_dropped_for_overlap += 1;
            tracing::debug!(
                requested_at = cut.at,
                group_start,
                reason = BreakReason::as_str(cut.reason),
                "dropped a segment boundary: its words overlap and cannot be split"
            );
            continue;
        }
        if at != cut.at {
            report.breaks_moved_for_overlap += 1;
            tracing::debug!(
                requested_at = cut.at,
                moved_to = at,
                reason = BreakReason::as_str(cut.reason),
                "moved a segment boundary back over an overlapping word"
            );
        }
        out.push(Cut {
            at,
            reason: cut.reason,
        });
    }
    // Counted here, not in the caller: `disjoint_cuts` decides which breaks
    // survived, so it is the only place that knows.
    for c in &out {
        report.record_break(c.reason);
    }
    out
}

/// Build the output groups from the flat word list and the cut positions.
fn build_groups(
    transcript: &Transcript,
    words: &[FlatWord],
    cuts: &[Cut],
    policy: &GroupingPolicy,
    bounds: Option<MediaBounds>,
    report: &mut GroupingReport,
) -> Vec<PlannedGroup> {
    let mut groups: Vec<PlannedGroup> = Vec::with_capacity(cuts.len() + 1);
    let mut boundaries: Vec<usize> = cuts.iter().map(|c| c.at).collect();
    boundaries.push(words.len());

    let mut from = 0usize;
    let mut previous_end: Option<Time> = None;
    for &to in &boundaries {
        if to <= from {
            continue;
        }
        let group = build_one(
            transcript,
            words,
            from,
            to,
            policy,
            bounds,
            previous_end,
            report,
        );
        previous_end = Some(group.end);
        groups.push(group);
        from = to;
    }
    groups
}

/// Build one output segment from `words[from..to]`.
fn build_one(
    transcript: &Transcript,
    words: &[FlatWord],
    from: usize,
    to: usize,
    policy: &GroupingPolicy,
    bounds: Option<MediaBounds>,
    previous_end: Option<Time>,
    report: &mut GroupingReport,
) -> PlannedGroup {
    let members = &words[from..to];
    let first = &members[0];
    let last = &members[members.len() - 1];

    // The span is derived from the words, never from the source segments' own
    // spans. `push_chunk` clamps a chunk's segments to that chunk's frame
    // count, so a source segment's end can sit up to a frame past the audio; a
    // span rebuilt from word times does not inherit that.
    let mut start = first.word.start.clamp_zero();
    let mut end = last.word.end.clamp_zero().max(start);

    // Non-overlap. The cut points were already made disjoint at the word level
    // (`disjoint_cuts`), so this is a belt-and-braces assertion rather than a
    // repair: if it ever has to move a boundary, the word-level pass missed
    // something, and moving the boundary here would orphan a word, so the
    // counter exists to make that visible.
    if let Some(prev) = previous_end
        && start < prev
    {
        start = prev;
        report.segments_pushed_for_overlap += 1;
    }

    // Media bounds, when the caller stated them. `push_chunk` has no upper
    // clamp, so a word can land past the end of the recording, and the buffer
    // the stage decoded is the authority on where the media ends.
    if let Some(bounds) = bounds {
        // `MediaWindow::clamp_span` is the only legal clamp, and it is
        // expressed in the contract crate so that there is exactly one
        // implementation of "inside the media" in the workspace.
        let (clamped_start, clamped_end) = bounds.clamp_span(start, end);
        if clamped_start != start {
            start = clamped_start;
            report.segments_clamped_to_media += 1;
        }
        if clamped_end != end {
            end = clamped_end;
            report.segments_clamped_to_media += 1;
        }
    }

    // Non-empty, enforced last so nothing above can produce a zero-length
    // segment. This can only push `end` past the media duration in the
    // contradictory case of a word reported at or after the declared end of the
    // media, which is already an upstream inconsistency; the report counts it.
    if end < start {
        end = start;
    }
    if end.as_millis() - start.as_millis() < MIN_SEGMENT_MS {
        end = Time::from_millis(start.as_millis() + MIN_SEGMENT_MS);
        report.segments_extended_to_minimum += 1;
    }

    let source_segment = &transcript.segments[first.source];

    let speaker = if members.iter().any(|w| w.label.is_some()) {
        majority(members.iter().filter_map(|w| w.label))
    } else {
        majority(
            members
                .iter()
                .filter_map(|w| transcript.segments[w.source].speaker),
        )
    };

    let mut confidence_count = 0usize;
    let mut confidence_sum = 0f32;
    for w in members {
        if let Some(c) = w.word.confidence {
            confidence_sum += c;
            confidence_count += 1;
        }
    }
    let confidence = (confidence_count > 0).then(|| confidence_sum / confidence_count as f32);

    // Language is the **most common** non-`None` language among the group's
    // words, not the first one.
    //
    // The difference matters and the property test caught it. "First non-None"
    // is a function of how the words happened to be segmented before grouping,
    // so grouping a transcript twice could change a group's language: the first
    // pass leaves it `None` when the group opens on a word with no language, and
    // the second pass then reads the first pass's `None`-then-`hinglish` group
    // as `hinglish`. A majority over the same word set is the same answer every
    // time, which is what makes grouping idempotent in its metadata and not only
    // in its boundaries.
    let language = majority_language(members, transcript);

    let text_chars: usize = members.iter().map(|w| w.chars).sum::<usize>() + members.len() - 1;
    let span = end.as_millis() - start.as_millis();
    let over_cap = (policy.max_words > 0 && members.len() > policy.max_words)
        || (policy.max_chars > 0 && text_chars > policy.max_chars)
        || (policy.max_duration_ms > 0 && span > policy.max_duration_ms);
    if over_cap {
        report.segments_over_cap += 1;
    }

    if speaker.is_some() {
        report.attributed += 1;
    } else {
        report.unattributed += 1;
    }

    PlannedGroup {
        id: ids::group_id(&source_segment.id, first.index_in_source),
        start,
        end,
        speaker,
        confidence,
        language,
        words: members.iter().map(|w| w.word.clone()).collect(),
        over_cap,
    }
}

/// What one threshold in a [`threshold_sweep`] produced.
///
/// Purely aggregate. There is no transcript content anywhere in this type, which
/// is what makes it safe to print for a real private recording.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThresholdOutcome {
    /// The `pause_break_ms` that was tried.
    pub threshold_ms: i64,
    /// Segments that threshold produced.
    pub segments: usize,
    /// Words per segment at that threshold.
    pub words_per_segment: f64,
    /// Breaks by reason at that threshold, excluding the pause rule itself, so
    /// the row says *why* the other rules fired.
    pub breaks: [(BreakReason, usize); 6],
    /// Mean segment duration in milliseconds.
    pub mean_duration_ms: f64,
    /// Longest segment in milliseconds.
    pub max_duration_ms: i64,
}

/// Re-plan the grouping at every threshold in [`THRESHOLD_SWEEP`] and report
/// what each produced, without changing anything.
///
/// This is the measurement the pause default is chosen from, and it exists as a
/// function rather than as a test for a reason that matters: the transcript it
/// runs on is a real private meeting, so the numbers have to come out of the
/// process that holds it. A test would need a fixture, and a fixture of a real
/// meeting is the meeting in the repository.
///
/// Pure and side-effect free — it plans, it never applies — so a caller can sweep
/// a stored transcript without risking it. Note that it does **not** log: the
/// sweep is a tuning operation, and the caller decides what to print.
#[must_use]
pub fn threshold_sweep(
    transcript: &Transcript,
    d: Option<&DiarizationResult>,
    bounds: Option<MediaBounds>,
    policy: &GroupingPolicy,
) -> Vec<ThresholdOutcome> {
    THRESHOLD_SWEEP
        .iter()
        .filter_map(|threshold| {
            let candidate = GroupingPolicy {
                pause_break_ms: *threshold,
                ..*policy
            };
            let plan = plan_inner(transcript, d, bounds, &candidate, false).ok()?;
            let s = group_stats(&plan.groups);
            Some(ThresholdOutcome {
                threshold_ms: *threshold,
                segments: plan.report.segments_out,
                words_per_segment: plan.report.words_per_segment(),
                breaks: plan.report.breaks(),
                mean_duration_ms: s.mean_duration_ms,
                max_duration_ms: s.max_duration_ms,
            })
        })
        .collect()
}

/// Report what grouping did, in aggregates, and — on request — the gap
/// distribution and the threshold sweep.
///
/// # Why the report is a log line and not a test
///
/// The transcripts this crate processes are real private meetings, and the
/// numbers behind the policy defaults can only come out of the process that
/// holds them. A test would need a fixture, and a fixture of a real meeting is
/// the meeting in the repository. So the measurement lives here, emits nothing
/// but counts, durations and policy values, and `scripts/a11/measure_grouping.ps1`
/// reads it out of a real run. The exported files it inspects are counted and
/// never opened.
///
/// `info` rather than `debug` because "this recording produced 227 segments, 188
/// of them at a speaker turn" is the first question anyone asks when a
/// transcript looks wrong, and `tracing` is already this project's answer to
/// that. One summary line per job.
///
/// The gap distribution and the sweep are gated on `VAANI_SEGMENT_MEASURE` and
/// off by default: they are a tuning cost — 28 numbers and seventeen extra
/// re-plans — not a per-job one.
fn log_outcome(
    transcript: &Transcript,
    groups: &[PlannedGroup],
    report: &GroupingReport,
    policy: &GroupingPolicy,
    elapsed: std::time::Duration,
) {
    if report.segments_out == 0 {
        return;
    }
    let s = group_stats(groups);
    let gaps = gap_distribution(transcript);
    let measuring = std::env::var_os("VAANI_SEGMENT_MEASURE").is_some();

    if measuring {
        tracing::info!(
            gaps = gaps.gaps,
            gap_negative = gaps.negative,
            gap_zero = gaps.zero,
            gap_percentiles_ms = ?gaps.percentiles,
            gap_at_or_above = ?gaps.at_or_above,
            "inter-word gap distribution"
        );
        for row in threshold_sweep(transcript, None, None, policy) {
            tracing::info!(
                threshold_ms = row.threshold_ms,
                segments = row.segments,
                words_per_segment = row.words_per_segment,
                mean_duration_ms = row.mean_duration_ms,
                max_duration_ms = row.max_duration_ms,
                breaks = ?row.breaks,
                "segmentation sweep (pause rule alone, no diarization result)"
            );
        }
    }

    tracing::info!(
        words = report.words_in,
        segments_in = report.segments_in,
        segments_out = report.segments_out,
        words_per_segment = report.words_per_segment(),
        breaks = ?report.breaks(),
        cap_breaks_deferred = report.cap_breaks_deferred,
        segments_over_cap = report.segments_over_cap,
        reordered_words = report.reordered_words,
        attributed = report.attributed,
        unattributed = report.unattributed,
        mean_duration_ms = s.mean_duration_ms,
        median_duration_ms = s.median_duration_ms,
        p90_duration_ms = s.p90_duration_ms,
        max_duration_ms = s.max_duration_ms,
        segments_over_7s = s.segments_over_7s,
        mean_chars_per_segment = s.mean_chars_per_segment,
        gap_p50_ms = gaps.percentile(0.50),
        gap_p90_ms = gaps.percentile(0.90),
        // The cost of this step, so a regression here is visible next to the
        // benefit. Grouping is a linear pass over the words plus a sort, so this
        // is microseconds on a 14-minute meeting and will stay that way.
        elapsed_us = elapsed.as_micros() as u64,
        policy = ?policy,
        "grouped the transcript into readable segments"
    );
}

/// The distribution of the silence between consecutive words, in milliseconds.
///
/// This is the measurement the pause threshold is chosen from, so it is part of
/// the public API rather than something a test computes privately: a settings
/// panel showing "at 500 ms you get 214 segments, at 700 ms you get 168" needs
/// the same numbers the tuning used, and the threshold is a value a user is
/// going to want to change.
///
/// A gap is `next word's start − this word's end`, over the whole transcript in
/// time order. Negative gaps are possible and are counted: they are overlapping
/// words, which chunk overlap produces, and hiding them would make the
/// distribution look better than the data is.
///
/// Contains no transcript content. Counts and durations only.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GapDistribution {
    /// Number of inter-word gaps measured. One fewer than the word count.
    pub gaps: usize,
    /// Gaps below zero: overlapping words.
    pub negative: usize,
    /// Gaps of exactly zero: abutting words.
    pub zero: usize,
    /// Percentiles in milliseconds, in the order given by [`PERCENTILES`].
    pub percentiles: Vec<i64>,
    /// Fraction of gaps at or above each threshold in [`THRESHOLD_SWEEP`].
    /// Parallel to that array.
    pub at_or_above: Vec<f64>,
    /// The sorted gaps, kept so an arbitrary percentile can be asked for after
    /// the fact without re-measuring. Public because it is a measurement, and a
    /// measurement you cannot interrogate is a number you have to trust.
    sorted: Vec<i64>,
}

impl GapDistribution {
    /// The `p` percentile of the gap distribution, in milliseconds, by
    /// nearest rank. `0.0` for a distribution with no gaps.
    #[must_use]
    pub fn percentile(&self, p: f64) -> i64 {
        crate::stats::percentile(&self.sorted, p)
    }

    /// The fraction of gaps at or above `threshold_ms`, in `[0, 1]`. This is the
    /// number a threshold decision is actually made on: at 500 ms it is the
    /// fraction of words that would start a new segment.
    #[must_use]
    pub fn fraction_at_or_above(&self, threshold_ms: i64) -> f64 {
        if self.sorted.is_empty() {
            return 0.0;
        }
        self.sorted.iter().filter(|g| **g >= threshold_ms).count() as f64 / self.sorted.len() as f64
    }

    /// Every gap, in ascending order. A copy, so a caller cannot disturb the
    /// measurement.
    #[must_use]
    pub fn sorted_gaps(&self) -> Vec<i64> {
        self.sorted.clone()
    }
}

/// The percentiles [`GapDistribution::percentiles`] reports, in order.
pub const PERCENTILES: [f64; 11] = [
    0.10, 0.25, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.98, 0.99,
];

/// The thresholds [`GapDistribution::at_or_above`] sweeps, in milliseconds.
///
/// Chosen to straddle the whole plausible range for a clause boundary: 100 ms is
/// shorter than a typical inter-word gap inside a phrase, 1 500 ms is longer
/// than almost any pause inside a turn, and 5 000 ms is a topic change. If the
/// distribution is a cliff rather than a plateau somewhere in this list, that is
/// the finding, and it is exactly the kind of thing that makes a default fragile.
pub const THRESHOLD_SWEEP: [i64; 17] = [
    100, 200, 300, 350, 400, 450, 500, 550, 600, 650, 700, 800, 900, 1_000, 1_200, 1_500, 2_000,
];

/// Measure the gap distribution of `transcript`'s words.
///
/// Pure, O(n log n) for the percentiles, and never panics on an empty or
/// single-word transcript — both return `gaps: 0`.
#[must_use]
pub fn gap_distribution(transcript: &Transcript) -> GapDistribution {
    let mut times: Vec<(i64, i64)> = transcript
        .segments
        .iter()
        .flat_map(|s| s.words.iter())
        .map(|w| (w.start.as_millis(), w.end.as_millis()))
        .collect();
    times.sort_unstable();

    let gaps: Vec<i64> = times.windows(2).map(|w| w[1].0 - w[0].1).collect();
    if gaps.is_empty() {
        return GapDistribution::default();
    }
    let negative = gaps.iter().filter(|g| **g < 0).count();
    let zero = gaps.iter().filter(|g| **g == 0).count();

    let mut sorted = gaps.clone();
    sorted.sort_unstable();
    let percentiles = PERCENTILES
        .iter()
        .map(|p| crate::stats::percentile(&sorted, *p))
        .collect();
    let total = sorted.len() as f64;
    let at_or_above = THRESHOLD_SWEEP
        .iter()
        .map(|t| sorted.iter().filter(|g| **g >= *t).count() as f64 / total)
        .collect();
    let _ = total;

    GapDistribution {
        gaps: gaps.len(),
        negative,
        zero,
        percentiles,
        at_or_above,
        sorted,
    }
}

/// The same numbers [`crate::stats::stats`] gives for a transcript, computed
/// from the groups before they are written into one.
///
/// Exists because the log line below is emitted from `plan_groups`, which has
/// not yet built a transcript, and building a throwaway one to measure it would
/// be the wrong trade for a function on the hot path of every job.
fn group_stats(groups: &[PlannedGroup]) -> crate::stats::TranscriptStats {
    let segments = groups.len();
    let words: usize = groups.iter().map(|g| g.words.len()).sum();
    let empty_segments = groups.iter().filter(|g| g.words.is_empty()).count();
    let attributed_segments = groups.iter().filter(|g| g.speaker.is_some()).count();
    let mut speakers: Vec<usize> = groups.iter().filter_map(|g| g.speaker).collect();
    speakers.sort_unstable();
    speakers.dedup();

    let mut durations: Vec<i64> = groups.iter().map(|g| g.duration_ms()).collect();
    let total_chars: usize = groups
        .iter()
        .map(|g| {
            g.words
                .iter()
                .map(|w| w.display_text().chars().count())
                .sum::<usize>()
                + g.words.len().saturating_sub(1)
        })
        .sum();
    let total_duration: i64 = durations.iter().sum();
    durations.sort_unstable();

    crate::stats::TranscriptStats {
        segments,
        words,
        empty_segments,
        attributed_segments,
        speakers_used: speakers.len(),
        mean_words_per_segment: words as f64 / segments.max(1) as f64,
        mean_chars_per_segment: total_chars as f64 / segments.max(1) as f64,
        mean_duration_ms: total_duration as f64 / segments.max(1) as f64,
        median_duration_ms: crate::stats::percentile(&durations, 0.50),
        p90_duration_ms: crate::stats::percentile(&durations, 0.90),
        max_duration_ms: durations.last().copied().unwrap_or(0),
        min_duration_ms: durations.first().copied().unwrap_or(0),
        segments_over_7s: durations.iter().filter(|d| **d > 7_000).count(),
    }
}

/// The most common non-`None` language among `members`, ties to the earliest
/// tag in word order.
///
/// A language tag is a `String` and not a small integer, so this counts
/// occurrences of each distinct tag rather than reusing [`majority`]. The number
/// of distinct tags inside one segment is one or two in practice — a meeting is
/// trilingual, not quadrilingual, inside a single clause — so a linear scan over
/// the tags seen so far is both correct and cheaper than allocating a map per
/// segment.
fn majority_language(members: &[FlatWord], transcript: &Transcript) -> Option<String> {
    let mut tags: Vec<(&str, usize)> = Vec::new();
    for w in members {
        let Some(lang) = transcript.segments[w.source].language.as_deref() else {
            continue;
        };
        match tags.iter_mut().find(|(t, _)| *t == lang) {
            Some(slot) => slot.1 += 1,
            None => tags.push((lang, 1)),
        }
    }
    // Highest count wins; a tie keeps the tag seen first, which is the earliest
    // in word order because the scan ran in that order.
    let mut best: Option<(&str, usize)> = None;
    for (tag, count) in &tags {
        if best.is_none_or(|(_, b)| *count > b) {
            best = Some((tag, *count));
        }
    }
    best.map(|(tag, _)| tag.to_owned())
}

/// The most common value in `values`, ties to the lower one. `None` for an
/// empty iterator, which is the ASR-only case rather than an error.
fn majority(values: impl Iterator<Item = usize>) -> Option<usize> {
    let mut counts: Vec<(usize, usize)> = Vec::new();
    for v in values {
        match counts.iter_mut().find(|(k, _)| *k == v) {
            Some(slot) => slot.1 += 1,
            None => counts.push((v, 1)),
        }
    }
    counts.sort_by_key(|(value, count)| (std::cmp::Reverse(*count), *value));
    counts.first().map(|(value, _)| *value)
}

/// Whether two transcripts are the same grouping: same segment count, same
/// Whether two transcripts have the same segmentation: same segment count, same
/// spans, same words with the same times, same speakers, same confidence — and
/// ignoring both segment ids and [`Segment::language`].
///
/// The language exclusion is a documented limitation, not a convenience.
/// `vaani_core::Word` carries no language, so a grouped segment's language is
/// derived from the segments its words came from rather than from the words
/// themselves, and re-grouping an already-grouped transcript cannot always
/// reproduce it even though every boundary, word, time, speaker and confidence
/// is identical. The property test found exactly that case. It cannot arise in
/// production, because the pipeline groups once and the per-segment language is
/// resolved by the ASR stage. The fix is a `language` field on `Word`, which is
/// a `vaani-core` change.
#[must_use]
pub fn same_segmentation(a: &Transcript, b: &Transcript) -> bool {
    if a.segments.len() != b.segments.len() {
        return false;
    }
    a.segments.iter().zip(b.segments.iter()).all(|(x, y)| {
        x.start == y.start
            && x.end == y.end
            && x.speaker == y.speaker
            && x.words.len() == y.words.len()
            && x.words.iter().zip(y.words.iter()).all(|(p, q)| {
                p.text == q.text
                    && p.start == q.start
                    && p.end == q.end
                    && p.confidence == q.confidence
                    && p.normalized == q.normalized
            })
    })
}

/// Whether two transcripts are the same grouping: same segment count, same
/// spans, same words with the same times, same speakers, same confidence and
/// language — and **ignoring segment ids**.
///
/// Exists because regrouping is idempotent, and an equality assertion that
/// included ids would be asserting the id format rather than the grouping. The
/// property test uses it both ways: that a second pass moves no boundary, and —
/// with ids included — that a second pass over an already-grouped transcript
/// reproduces the transcript exactly.
#[must_use]
pub fn same_grouping(a: &Transcript, b: &Transcript) -> bool {
    if a.segments.len() != b.segments.len() {
        return false;
    }
    a.segments.iter().zip(b.segments.iter()).all(|(x, y)| {
        x.start == y.start
            && x.end == y.end
            && x.speaker == y.speaker
            && x.language == y.language
            && x.words.len() == y.words.len()
            && x.words.iter().zip(y.words.iter()).all(|(p, q)| {
                p.text == q.text
                    && p.start == q.start
                    && p.end == q.end
                    && p.confidence == q.confidence
                    && p.normalized == q.normalized
            })
    })
}

/// Whether every word's surface form in `transcript` is free of punctuation.
///
/// The check that keeps the "no punctuation" claim honest. SraVaani emits none,
/// so a grouping policy keyed on sentence-final marks would never fire; this
/// makes that a *tested* statement about real transcripts rather than an
/// assumption in a doc comment.
///
/// Covers ASCII punctuation plus the general-punctuation and script-specific
/// ranges that actually occur in speech transcripts — including the Devanagari
/// danda (`।`) and CJK full stops — so it is meaningful outside Latin text. A
/// letter, a digit, a combining mark and an emoji are not punctuation and do
/// not fire it.
#[must_use]
pub fn word_text_contains_no_punctuation(transcript: &Transcript) -> bool {
    transcript
        .segments
        .iter()
        .flat_map(|s| s.words.iter())
        .all(|w| !w.display_text().chars().any(is_punctuation))
}

/// Ranges Unicode assigns to punctuation (`P*`) and the symbols that behave as
/// punctuation in running text, spelled out so the coverage is auditable on one
/// screen rather than hidden in a dependency.
const PUNCTUATION_RANGES: &[(u32, u32)] = &[
    (0x0021, 0x002F), // ASCII punctuation
    (0x003A, 0x0040), // ASCII punctuation
    (0x005B, 0x0060), // ASCII punctuation
    (0x007B, 0x007E), // ASCII punctuation
    (0x00A1, 0x00BF), // Latin-1 punctuation and symbols
    (0x060C, 0x060F), // Arabic comma, semicolon, question mark, full stop
    (0x061F, 0x061F), // Arabic question mark
    (0x06D4, 0x06D4), // Arabic full stop
    (0x0964, 0x0965), // Devanagari danda, double danda
    (0x2010, 0x2027), // general punctuation: hyphens, dashes, quotes, ellipsis
    (0x2030, 0x205E), // the rest of general punctuation
    (0x3001, 0x303F), // CJK punctuation
    (0xFF01, 0xFF65), // full-width forms
];

fn is_punctuation(c: char) -> bool {
    let cp = c as u32;
    PUNCTUATION_RANGES
        .iter()
        .any(|(lo, hi)| cp >= *lo && cp <= *hi)
}

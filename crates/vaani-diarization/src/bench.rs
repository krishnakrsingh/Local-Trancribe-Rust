//! A machine-readable diarization benchmark, and the label-free diagnostics
//! that go with it.
//!
//! # What this measures, and the one thing it will never measure
//!
//! This harness reports **structure and timing**: how many speakers the engine
//! claims, the spans it produced, how many speaker changes those spans contain,
//! the wall time and realtime factor, and the per-stage breakdown. Every field
//! is an output of the engine or a count derived from one.
//!
//! It reports **no accuracy whatsoever on any real recording.** Diarization
//! accuracy is defined against per-speaker ground truth (DER, JER), and this
//! repository has no annotated reference for `scratch/clip.wav` or for any
//! other real audio. A harness that printed a DER against no reference would be
//! worse than no harness, because the number would look like a measurement.
//! [`BenchReport::notes`] carries that statement in the machine-readable output
//! itself, so a JSON file cannot be read without it.
//!
//! The one place an accuracy-shaped number *does* appear is a
//! [`CaseKind::SyntheticReference`] case, where the reference is produced by
//! [`crate::synth::render`] -- the same call that produced the audio -- and
//! [`crate::eval`] scores against it. That measures the engine against a
//! generator, not against the world, and is labelled as such in the report.
//!
//! # What the label-free geometry block is for
//!
//! [`GeometryReport`] exists to answer a question that needs no labels: **is
//! there any structure in this recording's feature space at all?**
//!
//! * [`GeometryReport::pairwise`] is the distribution of every window-to-window
//!   cosine distance. A single blob means the features see one thing.
//! * [`MergeReport`] is the average-linkage merge-cost sequence. Its *largest
//!   relative gap* is the best available statement about where a speaker count
//!   would be justified, and comparing it with [`MergeReport::threshold_cut_k`]
//!   says whether the shipped threshold is sitting in that gap or somewhere else
//!   entirely.
//! * [`PitchReport`] summarises the one *absolute* dimension in the vector --
//!   median log2 F0 per window -- as an occupancy histogram over semitones,
//!   with the largest empty gap between occupied bins. A multi-party meeting
//!   with an octave between two voices shows a large internal gap; one
//!   expressive speaker does not.
//! * [`BlockShare`] is a per-block decomposition of the pairwise distance, so it
//!   is visible which blocks are producing the spread and which carry speaker
//!   signal.
//!
//! None of these are accuracy numbers. They are statements about the shape of
//! the input to the clustering, and they are the evidence that decides whether
//! the next thing to change is the features, the threshold or the segmentation.
//!
//! # Privacy
//!
//! Nothing in this module or in the binary that drives it can see transcript
//! text: it holds an [`vaani_core::media::AudioBuffer`] and a
//! [`vaani_core::pipeline::DiarizationResult`], and the only strings in the
//! report are case names, configuration numbers and engine ids. Span start and
//! end times are emitted because a benchmark whose output cannot be checked is
//! not a benchmark; they are turn *timing*, not content.

use std::path::PathBuf;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use vaani_core::media::AudioBuffer;
use vaani_core::pipeline::{DiarizationResult, SpeechTimeline};
use vaani_core::{CancelToken, Time};

use crate::config::DiarizationConfig;
use crate::engine::{HeuristicDiarizer, LabelledWindow};
use crate::eval::{self, Percentiles};
use crate::features::Layout;
use crate::{cluster, synth, vad};

/// The JSON schema identifier. Bumped on any breaking change to the shape.
pub const SCHEMA: &str = "vaani.diarization.bench/1";

/// The threshold sweep the harness runs when asked, in cosine units.
///
/// 0.20 to 1.30, chosen to bracket the shipped effective merge distance (0.89)
/// on both sides and to reach the cap, so a reader sees the whole step function
/// rather than three points of it. Two points are deliberately close together
/// (0.85, 0.89, 0.90) because that is where the default sits and where the
/// count is steepest.
pub const MERGE_SWEEP: [f64; 21] = [
    0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.85, 0.89, 0.90, 0.95, 1.00,
    1.05, 1.10, 1.15, 1.20, 1.30,
];

/// A five-number summary plus a mean, serialisable.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Stat {
    /// Number of samples.
    pub n: usize,
    /// Smallest sample. `None` for an empty population.
    pub min: Option<f64>,
    /// 10th percentile.
    pub p10: Option<f64>,
    /// Median.
    pub p50: Option<f64>,
    /// 90th percentile.
    pub p90: Option<f64>,
    /// Largest sample.
    pub max: Option<f64>,
    /// Arithmetic mean.
    pub mean: Option<f64>,
}

impl Stat {
    /// Summarise an unsorted population. An empty population is all `None`,
    /// which is what a missing measurement must look like in JSON: `null`, not
    /// `0.0`, because a zero would be a claim.
    #[must_use]
    pub fn of(mut values: Vec<f64>) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let at = |p: f64| values[(p * (values.len() - 1) as f64).round() as usize];
        Self {
            n: values.len(),
            min: Some(values[0]),
            p10: Some(at(0.10)),
            p50: Some(at(0.50)),
            p90: Some(at(0.90)),
            max: Some(values[values.len() - 1]),
            mean: Some(values.iter().sum::<f64>() / values.len() as f64),
        }
    }
}

/// What the engine is, so two reports can be compared honestly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineInfo {
    /// `DiarizationEngine::id`.
    pub id: String,
    /// Crate version.
    pub version: String,
    /// [`crate::capability_statement`], verbatim.
    pub capability: String,
    /// Whether a trained embedder is loaded. Always false in this build.
    pub neural_embeddings: bool,
    /// Whether the speaker count was estimated rather than supplied.
    pub speaker_count_estimated: bool,
    /// The merge distance the default configuration actually uses, after bias
    /// scaling. Reported because it is the number every "why two speakers"
    /// question turns on, and it is *not* `SpeakerCount::default_policy()`'s
    /// 1.05.
    pub effective_merge_distance: f64,
    /// The unscaled base the policy asked for, `None` for `Auto`.
    pub base_merge_distance: Option<f64>,
    /// The bias in force.
    pub bias: String,
    /// The stopping rule the default policy names.
    pub default_rule: String,
}

/// What kind of input a case ran on, and therefore what may be claimed from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseKind {
    /// A real recording. No accuracy is reported and none can be.
    RealClip,
    /// Generated audio, not scored.
    Synthetic,
    /// Generated audio scored against the generator's own per-frame reference.
    /// The **only** case kind that may carry a [`CaseReport::reference`].
    SyntheticReference,
    /// A re-run of the anchor input under a different configuration. The same
    /// accuracy rules as the anchor apply: none.
    Variant,
}

/// One configured run of the engine over one input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseReport {
    /// Stable name, used in the report and by anything comparing two reports.
    pub name: String,
    /// The primary case this one re-ran, for sweep and variant rows. `None` for
    /// a primary case. Present because two fixtures can share a duration and
    /// matching a sweep row to its input by anything else silently pairs the
    /// wrong numbers.
    pub anchor: Option<String>,
    /// What kind of input this is.
    pub kind: CaseKind,
    /// The input, in aggregates.
    pub input: InputInfo,
    /// The configuration that produced this case.
    pub config: ConfigInfo,
    /// Where the activity mask came from and how much it kept.
    pub activity: ActivityReport,
    /// The structural output.
    pub result: ResultReport,
    /// Timing and shape counts.
    pub timing: TimingReport,
    /// Label-free feature-space structure. `None` when the window dump was
    /// skipped to keep a sweep affordable.
    pub geometry: Option<GeometryReport>,
    /// Scored against a generated reference. `None` for every real case, and
    /// that absence is the whole point.
    pub reference: Option<ReferenceReport>,
    /// The spans of a **primary** case. Empty for [`CaseKind::Variant`], whose
    /// purpose is a count and not a span list: a full run produces 42 sweep
    /// points and 38 variants, and carrying ~350 spans in each of them turned a
    /// 2.2 MB report into a 90 kB one for no information. A variant that needs
    /// its spans is one `cargo run` away.
    pub segments: Vec<SpanInfo>,
}

/// The input, described without any content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputInfo {
    /// `"real_clip"`, `"synthetic"` or `"anchor"`.
    pub kind: String,
    /// The file's base name only, never a full path: a directory can carry a
    /// person's name.
    pub file: Option<String>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Channels in the file, before downmix.
    pub channels: u16,
    /// Bits per sample in the file, `0` for generated audio.
    pub bits_per_sample: u16,
    /// Sample count after downmix.
    pub samples: usize,
    /// Duration in milliseconds.
    pub duration_ms: i64,
    /// Peak absolute sample value, for a clipping sanity check.
    pub peak_sample: f32,
}

/// The configuration a case ran under, resolved to numbers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfigInfo {
    /// Bias in force.
    pub bias: String,
    /// The stopping-rule policy, as configured.
    pub policy: String,
    /// The merge distance actually used, after bias scaling.
    pub effective_merge_distance: f64,
    /// The unscaled base the policy asked for. `None` for `Auto`, infinite for
    /// `Fixed`.
    pub base_merge_distance: Option<f64>,
    /// Speaker cap, `None` when the policy has none.
    pub max_speakers: Option<usize>,
    /// Analysis window in frames.
    pub window_frames: usize,
    /// Hop between windows, in frames.
    pub window_hop_frames: usize,
    /// Fraction of a window that must be active for it to be embedded.
    pub min_window_speech_ratio: f32,
    /// Median filter length after bias scaling.
    pub median_filter_frames: usize,
    /// Turns shorter than this are absorbed.
    pub min_turn_ms: i64,
    /// Speech a cluster needs to survive, after bias scaling.
    pub effective_min_cluster_speech_ms: i64,
    /// Half-width of the feature moving average, in windows.
    pub feature_smooth_half_windows: usize,
    /// Constant carrier dimension magnitude.
    pub carrier: f32,
    /// Multiplier on the absolute pitch dimension.
    pub absolute_pitch_scale: f32,
    /// Reference subtracted from log2 F0 before scaling.
    pub pitch_reference_log2: f32,
    /// Minimum-spread floors, so the report records the gate configuration too.
    pub min_spread: crate::config::MinSpread,
    /// Block weights.
    pub block_weights: crate::config::BlockWeights,
    /// Activity-detector parameters, which change the mask and therefore the
    /// window set.
    pub activity: crate::config::ActivityParams,
}

impl ConfigInfo {
    /// Resolve a configuration into the numbers a reader needs.
    #[must_use]
    pub fn of(cfg: &DiarizationConfig) -> Self {
        use crate::config::SpeakerCount;
        let policy = match cfg.clusters.count {
            SpeakerCount::Fixed(k) => format!("fixed({k})"),
            SpeakerCount::Threshold {
                merge_distance,
                max_speakers,
            } => format!("threshold({merge_distance}, cap {max_speakers})"),
            SpeakerCount::Auto {
                max_speakers,
                min_gap_ratio,
            } => format!("auto-gap(cap {max_speakers}, ratio {min_gap_ratio})"),
        };
        Self {
            bias: format!("{:?}", cfg.bias),
            policy,
            effective_merge_distance: cfg.effective_merge_distance(),
            base_merge_distance: cfg.clusters.count.base_merge_distance(),
            max_speakers: match cfg.clusters.count {
                SpeakerCount::Fixed(_) => None,
                other => Some(other.max_speakers()),
            },
            window_frames: cfg.clusters.window_frames,
            window_hop_frames: cfg.clusters.window_hop_frames,
            min_window_speech_ratio: cfg.clusters.min_window_speech_ratio,
            median_filter_frames: cfg.effective_median_filter_frames(),
            min_turn_ms: cfg.clusters.min_turn_ms,
            effective_min_cluster_speech_ms: cfg.effective_min_cluster_speech_ms(),
            feature_smooth_half_windows: cfg.clusters.feature_smooth_half_windows,
            carrier: cfg.features.weights.carrier,
            absolute_pitch_scale: crate::features::ABSOLUTE_PITCH_SCALE,
            pitch_reference_log2: crate::features::PITCH_REFERENCE_LOG2,
            min_spread: cfg.min_spread,
            block_weights: cfg.features.weights,
            activity: cfg.activity,
        }
    }
}

/// Where the activity mask came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivitySource {
    /// This crate's own detector, because no timeline was supplied.
    Internal,
    /// A timeline supplied by `vaani-vad`, which is what `vaani-app` does.
    External,
    /// The generator's own reference intervals, padded. **Synthetic cases
    /// only.** A perfect VAD, used to separate "the activity gate lost the
    /// speaker" from "the features cannot see the speaker".
    Oracle,
}

/// What the activity stage kept and dropped.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActivityReport {
    /// Where the mask came from.
    pub source: ActivitySource,
    /// Analysis frames the engine visited.
    pub frames: usize,
    /// Frames inside the mask.
    pub active_frames: usize,
    /// `active_frames / frames`. The number that says whether a gate is
    /// working, and distinctly *not* the coverage below: hangover and
    /// gap-bridging inflate coverage by construction.
    pub active_frame_ratio: f64,
    /// Segments in the timeline handed to the feature stage.
    pub timeline_segments: usize,
    /// Total timeline coverage in milliseconds.
    pub timeline_speech_ms: i64,
    /// `timeline_speech_ms / duration_ms`.
    pub timeline_coverage: f64,
    /// Duration distribution of the timeline segments, before the engine's own
    /// window and turn rules.
    pub segment_durations: Stat,
    /// Fraction of windows that were kept for clustering, after the
    /// `min_window_speech_ratio` gate.
    pub window_keep_ratio: Option<f64>,
    /// The per-frame mask, one entry per analysis frame.
    ///
    /// Not serialised: it is 83,000 entries for a 14-minute recording and
    /// every aggregate that matters is already above. It is here so
    /// [`ReferenceReport`] can count, per generator speaker, how many of its
    /// frames reached the feature stage.
    #[serde(skip, default)]
    pub frame_mask: Vec<bool>,
}

/// The structural result. Counts and ratios only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResultReport {
    /// `DiarizationResult::num_speakers`.
    pub speaker_count: usize,
    /// Number of spans.
    pub span_count: usize,
    /// Adjacent spans naming different speakers. This is the number the
    /// transcript's structure leans on: the grouping policy breaks a segment on
    /// a speaker change, so this is most of the segment count.
    pub speaker_change_count: usize,
    /// Speaker changes per minute of audio, `None` for a zero-length input.
    pub speaker_changes_per_minute: Option<f64>,
    /// Fraction of the recording covered by spans, clamped to `[0, 1]`.
    pub coverage: f64,
    /// Talk time per speaker, in milliseconds, in `0..speaker_count`.
    pub talk_ms: Vec<i64>,
    /// Talk-time share per speaker. Sums to 1 when there is any talk time.
    pub talk_share: Vec<f64>,
    /// Span duration distribution.
    pub turn_durations: Stat,
    /// Span confidence: the engine's own unsmoothed-label agreement. The only
    /// per-span number here that is not timing, and the engine's honest
    /// uncertainty signal.
    pub turn_scores: Stat,
}

impl ResultReport {
    /// Derive the structural report from a result and the audio it describes.
    #[must_use]
    pub fn of(d: &DiarizationResult, audio: &AudioBuffer) -> Self {
        let changes = d
            .spans
            .windows(2)
            .filter(|p| p[0].speaker != p[1].speaker)
            .count();
        let durations: Vec<f64> = d
            .spans
            .iter()
            .map(|s| (s.end.as_millis() - s.start.as_millis()).max(0) as f64)
            .collect();
        let scores: Vec<f64> = d.spans.iter().filter_map(|s| s.score).collect();
        let talk = d.speaker_durations();
        let total: i64 = talk.iter().sum();
        let dur_ms = audio.duration().as_millis();
        Self {
            speaker_count: d.num_speakers,
            span_count: d.spans.len(),
            speaker_change_count: changes,
            speaker_changes_per_minute: (dur_ms > 0)
                .then(|| changes as f64 * 60_000.0 / dur_ms as f64),
            coverage: d.coverage(audio.duration()),
            talk_ms: talk.to_vec(),
            talk_share: talk
                .iter()
                .map(|d| {
                    if total > 0 {
                        *d as f64 / total as f64
                    } else {
                        0.0
                    }
                })
                .collect(),
            turn_durations: Stat::of(durations),
            turn_scores: Stat::of(scores),
        }
    }
}

/// Timing and the shape counts that explain it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimingReport {
    /// Wall time around the whole call.
    pub wall_ms: u64,
    /// The engine's own `total_ms`.
    pub total_ms: u64,
    /// Pass 1, the activity scan. Near zero when an external timeline was
    /// supplied, because the engine then does no scan at all.
    pub activity_ms: u64,
    /// Pass 2, features.
    pub features_ms: u64,
    /// Agglomeration, stopping rule, guard, Lloyd.
    pub cluster_ms: u64,
    /// Median filter and turn assembly.
    pub smooth_ms: u64,
    /// `wall_ms / duration_ms`. Lower is faster.
    pub rtf: f64,
    /// Analysis frames visited.
    pub frames: usize,
    /// Windows embedded.
    pub windows: usize,
    /// Rows actually fed to the clustering, after the stride cap.
    pub rows_clustered: usize,
    /// Clusters before the min-speech guard.
    pub clusters: usize,
    /// Which stopping rule fired.
    pub rule: String,
    /// Bytes the engine itself accounts for: the activity arrays, one window of
    /// per-frame state, the window vectors and the clustering matrix. The
    /// `AudioBuffer` the caller already holds is **not** in this number, and is
    /// reported separately so the two are not confused.
    pub engine_accounted_bytes: u64,
    /// The `AudioBuffer` the caller holds, for context.
    pub audio_buffer_bytes: u64,
}

impl TimingReport {
    /// Build the timing report from the engine's own instrumentation.
    #[must_use]
    pub fn of(
        t: &crate::engine::DiarizationTimings,
        wall: std::time::Duration,
        audio: &AudioBuffer,
        cfg: &DiarizationConfig,
    ) -> Self {
        let rows_clustered = t.windows.min(cfg.clusters.max_cluster_windows);
        let dims = cfg.features.vector_len();
        let accounted = (t.frames as u64 * 3 * 4)
            + (cfg.clusters.window_frames as u64 * dims as u64 * 4)
            + (t.windows as u64 * (dims as u64 + 1) * 4)
            + (rows_clustered as u64 * rows_clustered as u64 * 4);
        Self {
            wall_ms: wall.as_millis() as u64,
            total_ms: t.total_ms,
            activity_ms: t.activity_ms,
            features_ms: t.features_ms,
            cluster_ms: t.cluster_ms,
            smooth_ms: t.smooth_ms,
            rtf: t.rtf(audio.duration().as_millis()),
            frames: t.frames,
            windows: t.windows,
            rows_clustered,
            clusters: t.clusters,
            rule: t.rule.to_owned(),
            engine_accounted_bytes: accounted,
            audio_buffer_bytes: (audio.samples.len() * 4) as u64,
        }
    }
}

/// One span, in timing only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpanInfo {
    /// Index into [`ResultReport::talk_ms`].
    pub speaker: usize,
    /// Absolute start, in milliseconds from the start of the media.
    pub start_ms: i64,
    /// Absolute end, in milliseconds.
    pub end_ms: i64,
    /// `end_ms - start_ms`.
    pub duration_ms: i64,
    /// The engine's own confidence for the turn, if it reported one.
    pub score: Option<f64>,
}

/// Label-free structure in the feature space.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeometryReport {
    /// Windows dumped.
    pub windows: usize,
    /// All window-to-window cosine distances, after temporal smoothing,
    /// weighting and L2 normalisation -- i.e. the population the clustering
    /// sees.
    pub pairwise: Stat,
    /// The average-linkage merge sequence, and what the threshold did with it.
    pub merge: MergeReport,
    /// The absolute pitch dimension, as an occupancy histogram.
    pub pitch: PitchReport,
    /// Per-block share of the pairwise distance.
    pub blocks: Vec<BlockShare>,
}

/// The agglomeration's merge-cost sequence, and what a stopping rule did with it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MergeReport {
    /// Rows clustered.
    pub rows: usize,
    /// Merges performed; `rows - 1` for a full run.
    pub merges: usize,
    /// Cost of the merge that took the count from `k + 1` to `k`, for
    /// `k = 1..=min(12, rows - 1)`. This is the sequence a threshold cuts.
    pub cost_at_count: Vec<CountCost>,
    /// The largest *relative* jump in the sequence: the strongest label-free
    /// statement available about where a speaker count is justified.
    pub largest_relative_gap: Option<RelativeGap>,
    /// The five largest relative jumps, descending.
    pub top_relative_gaps: Vec<RelativeGap>,
    /// The count the configured threshold selects, computed from the sequence
    /// rather than from the result, so it can be compared directly with
    /// `largest_relative_gap`.
    pub threshold_cut_k: Option<usize>,
    /// 64 evenly spaced costs across the whole sequence, so the shape is
    /// visible without a 780-element array in the file.
    pub costs_sampled: Vec<f64>,
}

/// The cost of one merge.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CountCost {
    /// Cluster count after this merge.
    pub k: usize,
    /// Cosine cost of the merge.
    pub cost: f64,
}

/// A relative jump in the merge sequence.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RelativeGap {
    /// The cluster count the jump sits above: the sequence goes from this many
    /// clusters to one fewer across the jump.
    pub after_k: usize,
    /// The cost below the jump.
    pub from: f64,
    /// The cost above the jump.
    pub to: f64,
    /// `to / from`.
    pub ratio: f64,
}

/// The absolute pitch dimension, summarised without labels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PitchReport {
    /// Windows contributing a pitch value.
    pub windows: usize,
    /// Window pitch, in semitones above 1 Hz. `min` is the lowest voice and
    /// `max` the highest, so the span is the observable pitch range of the
    /// recording.
    pub semitones: Stat,
    /// Bin width in semitones.
    pub bin_semitones: f64,
    /// The lowest occupied bin's centre, in semitones above 1 Hz.
    pub bin_origin_semitones: Option<f64>,
    /// Bins holding at least 10% of the peak bin's count. A meeting with two
    /// clearly separated voices has few widely spaced bins; one expressive
    /// speaker has many contiguous ones.
    pub occupied_bins_above_10pct: usize,
    /// Occupancy per bin, as `[bin_index, count]`, so the shape is visible
    /// without a plot.
    pub histogram: Vec<(u16, usize)>,
    /// The largest run of unoccupied bins *between* two occupied ones, in
    /// semitones. A large value is pitch structure; zero is a continuum.
    ///
    /// **This is the number to look at for the "two speakers on a multi-party
    /// meeting" question.** A fourth of an octave or more of empty spectrum
    /// inside the occupied range is evidence that the vector is separating
    /// something, whatever the clustering then does with it.
    pub largest_internal_gap_semitones: f64,
    /// The largest empty run strictly *above* the highest occupied bin and
    /// below a voice-like ceiling, in semitones. Reports whether there is
    /// *room* for another voice the recording does not exercise.
    pub headroom_above_semitones: f64,
}

/// One block's share of a cosine distance, and the scale it is normalised by.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlockShare {
    /// Block name.
    pub block: String,
    /// Dimensions in the block, including the carrier's one.
    pub dims: usize,
    /// Mean share of the pairwise distance. The shares sum to 1 because a
    /// cosine distance on unit vectors decomposes additively per dimension.
    pub mean_share: f64,
    /// Mean share over same-speaker pairs, when a reference exists.
    pub same_speaker_share: Option<f64>,
    /// Mean share over different-speaker pairs, when a reference exists.
    pub different_speaker_share: Option<f64>,
    /// `different - same`: the speaker signal this block actually carries.
    pub speaker_signal: Option<f64>,
    /// Median, over the block's dimensions, of the **measured** spread of that
    /// dimension across this recording's windows, in the dimension's own units.
    /// This is the number `features::within_speaker_spread` and `MinSpread` are
    /// both compared against, so it is the only way to tell whether those
    /// constants still describe the audio.
    pub measured_median_spread: Option<f64>,
    /// The constant `normalise_in_place` divides this block by.
    pub scale_constant: Option<f64>,
    /// `measured_median_spread / scale_constant`. **1.0 means the block arrives
    /// at unit variance; 200 means it arrives amplified two hundred-fold** and
    /// will then take two hundred times its configured share of the vector once
    /// the row is L2-normalised. This is the whole normalisation story in one
    /// number, and it is why a heavily weighted block whose data is flat can
    /// contribute nothing to the distance while an almost-unweighted block whose
    /// data is noisy can contribute a third of it.
    pub spread_ratio: Option<f64>,
    /// The `MinSpread` floor for this block.
    pub min_spread_floor: Option<f64>,
    /// `measured_median_spread / min_spread_floor`. Far above 1 means the gate
    /// lets the block through at full strength; near 1 means the gate is what
    /// keeps it.
    pub floor_ratio: Option<f64>,
}

/// A score against a *generated* reference. Never present for real audio.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceReport {
    /// What the reference is, so it cannot be mistaken for a human annotation.
    pub provenance: String,
    /// Speakers the generator put in the audio.
    pub true_speakers: usize,
    /// Windows scored (pure windows only).
    pub scored: usize,
    /// Windows dropped for straddling a turn boundary.
    pub mixed_windows: usize,
    /// Fraction of scored windows in their speaker's dominant cluster.
    pub purity: Option<f64>,
    /// Fraction of scored windows whose speaker appears in any cluster.
    pub recall: Option<f64>,
    /// Strictest figure: a window counts only if its cluster is uniquely its
    /// speaker's best cluster.
    pub complete_accuracy: Option<f64>,
    /// `different.p10 - same.p90`. Positive means the two populations are
    /// separated and a threshold has somewhere to sit.
    pub population_margin: Option<f64>,
    /// Same-speaker pairwise distance summary.
    pub same_speaker: Stat,
    /// Different-speaker pairwise distance summary.
    pub different_speaker: Stat,
    /// Scored windows per true speaker, keyed by the generator's speaker index.
    /// This is what shows a speaker being *discarded* rather than merely merged.
    pub windows_per_true_speaker: Vec<(i32, usize)>,
    /// True frames per speaker, before the activity gate.
    pub true_frames_per_true_speaker: Vec<(i32, usize)>,
    /// Frames per speaker that survived the activity gate and reached the
    /// feature stage. A speaker at zero here was discarded **before** any
    /// feature was computed, which is a different defect from the features
    /// being unable to separate it.
    pub active_frames_per_true_speaker: Vec<(i32, usize)>,
}

/// One point of the merge-distance sweep.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepPoint {
    /// The case this point re-ran, so the internal and external-timeline
    /// sweeps are distinguishable in one flat list.
    pub case: String,
    /// The activity source that case used.
    pub activity: ActivitySource,
    /// The merge distance passed to `SpeakerCount::Threshold`, before bias
    /// scaling.
    pub requested: f64,
    /// The distance the engine actually used.
    pub effective: f64,
    /// Speakers reported.
    pub speaker_count: usize,
    /// Spans reported.
    pub span_count: usize,
    /// Speaker changes in the span sequence.
    pub speaker_change_count: usize,
    /// Mean turn in milliseconds.
    pub mean_turn_ms: f64,
    /// Rule that fired.
    pub rule: String,
    /// Wall time, so a sweep point is not free.
    pub wall_ms: u64,
    /// Scored against a generated reference, when the case has one. This is
    /// how "is there a threshold that would have been right?" is answered
    /// without a human annotation: **only** on generated audio.
    pub reference: Option<ReferenceReport>,
}

/// The whole report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchReport {
    /// [`SCHEMA`].
    pub schema: String,
    /// What the engine is.
    pub engine: EngineInfo,
    /// Statements about what this report does and does not claim. Present so
    /// that a JSON file read on its own cannot be mistaken for an accuracy
    /// measurement.
    pub notes: Vec<String>,
    /// The cases, in run order.
    pub cases: Vec<CaseReport>,
    /// The merge-distance sweep, run over **every** case that is a real
    /// recording -- so the internal and the external-timeline sweeps both
    /// appear -- and over the first synthetic fixture when there is no real
    /// input. This is the step function the speaker count is.
    pub merge_sweep: Vec<SweepPoint>,
    /// Configuration variants, run over the same set of cases.
    pub variants: Vec<CaseReport>,
}

impl BenchReport {
    /// A report with nothing in it but the engine description and the notes.
    #[must_use]
    pub fn empty() -> Self {
        let cfg = DiarizationConfig::default();
        Self {
            schema: SCHEMA.to_owned(),
            engine: EngineInfo {
                id: HeuristicDiarizer::ID.to_owned(),
                version: crate::VERSION.to_owned(),
                capability: crate::capability_statement(),
                neural_embeddings: false,
                speaker_count_estimated: true,
                effective_merge_distance: cfg.effective_merge_distance(),
                base_merge_distance: cfg.clusters.count.base_merge_distance(),
                bias: format!("{:?}", cfg.bias),
                default_rule: "threshold".to_owned(),
            },
            notes: notes(),
            cases: Vec::new(),
            merge_sweep: Vec::new(),
            variants: Vec::new(),
        }
    }

    /// Serialise to pretty JSON.
    ///
    /// # Errors
    /// Returns the serializer's error. NaN cannot appear: every optional
    /// statistic is `Option<f64>` precisely so a missing measurement is `null`
    /// rather than an unserialisable `NaN` that some readers silently coerce
    /// to zero.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// The case with this name, if there is one.
    #[must_use]
    pub fn case(&self, name: &str) -> Option<&CaseReport> {
        self.cases.iter().find(|c| c.name == name)
    }
}

/// The standing caveats, carried inside the machine-readable output.
fn notes() -> Vec<String> {
    let mut v = vec![
        "STRUCTURE AND TIMING ONLY for every real recording. No diarization error rate \
         (DER/JER), purity, recall or any other accuracy figure is reported for real audio, \
         because this repository has no annotated per-speaker reference for it. A number here \
         is an output of the engine or a count derived from one."
            .to_owned(),
        "Accuracy-shaped figures appear only under cases with kind = \"synthetic_reference\". \
         Their reference is the per-frame ground truth returned by synth::render, the same call \
         that generated the audio. That measures the engine against a generator with fixed \
         fundamentals, fixed harmonic rolloffs and fixed formants. It is a ceiling for the \
         feature vector, not an estimate of real-world accuracy."
            .to_owned(),
        "geometry.* is label-free structure in the feature space (pairwise distance \
         distribution, merge-cost sequence, pitch occupancy, per-block shares). It says whether \
         there is any structure to cluster. It is not a correctness measure."
            .to_owned(),
        "No transcript text, and nothing derived from what was said, is read, derived, printed or \
         written by this harness. Span start and end times are turn timing."
            .to_owned(),
        "Timing figures are from a release build. The workspace sets opt-level 3, thin LTO and \
         codegen-units 1; a debug build of this engine is roughly fifty times slower and a debug \
         number is not a performance claim."
            .to_owned(),
    ];
    v.push(vad_probe_note());
    v
}

/// What the harness should do.
#[derive(Debug, Clone, Default)]
pub struct BenchOptions {
    /// A real WAV file. When absent, no real case runs and a note says so.
    pub input: Option<PathBuf>,
    /// Also run the input with `vaani-vad`'s timeline, the path `vaani-app`
    /// takes. Requires the `vad-probe` feature; without it the request is
    /// recorded as a note rather than silently ignored.
    pub external_timeline: bool,
    /// Run the merge-distance sweep over the anchor input.
    pub sweep: bool,
    /// Run the configuration variants over the anchor input.
    pub variants: bool,
    /// Run the synthetic fixtures, including the oracle-timeline pair.
    pub synthetic: bool,
    /// How many times to run the primary case. The **last** run's timing is
    /// reported; repeats exist to see whether a number is stable, not to pick
    /// the best one.
    pub repeats: usize,
    /// Cap on how many anchors the sweep and the variants run over. `None`, the
    /// default, is every anchor -- which is the exhaustive table and costs
    /// `(21 + 19) * anchors` engine runs. `Some(1)` keeps one, which is what a
    /// test that only wants to check the *form* of a sweep row needs.
    pub max_anchors: Option<usize>,
}

impl BenchOptions {
    /// Options for one real clip, with a sweep and the variants.
    #[must_use]
    pub fn for_clip(path: impl Into<PathBuf>) -> Self {
        Self {
            input: Some(path.into()),
            sweep: true,
            variants: true,
            synthetic: true,
            external_timeline: true,
            repeats: 1,
            max_anchors: None,
        }
    }
}

/// Run the harness and produce a report.
///
/// Never fails for a missing input: a real clip that is not there produces a
/// report whose real cases are simply absent, and a note saying so. A benchmark
/// that cannot run on a fresh checkout is a benchmark nobody runs.
#[must_use]
pub fn run(opts: &BenchOptions) -> BenchReport {
    let mut report = BenchReport::empty();
    let mut anchors: Vec<Anchor> = Vec::new();

    // --- the real clip ---------------------------------------------------
    if let Some(path) = &opts.input {
        match crate::wav::read_wav_mono(path) {
            Ok(audio) => {
                let input = InputInfo {
                    kind: "real_clip".to_owned(),
                    file: Some(stem(path)),
                    sample_rate: audio.sample_rate,
                    channels: audio.channels,
                    bits_per_sample: audio.bits_per_sample,
                    samples: audio.samples.len(),
                    duration_ms: audio.duration_ms(),
                    peak_sample: audio.peak(),
                };
                let samples = audio.samples;
                for plan in timelines_to_run(opts) {
                    let name = match plan {
                        TimelinePlan::Internal => "real-clip",
                        TimelinePlan::External => "real-clip#external-timeline",
                        TimelinePlan::Oracle => "real-clip#oracle",
                    };
                    let repeats = if plan == TimelinePlan::Internal {
                        opts.repeats.max(1)
                    } else {
                        1
                    };
                    let (case, _) = run_case(CaseRun {
                        name: name.to_owned(),
                        anchor: None,
                        kind: CaseKind::RealClip,
                        input: input.clone(),
                        samples: &samples,
                        cfg: DiarizationConfig::default(),
                        plan,
                        truth: None,
                        repeats,
                        dump_geometry: true,
                    });
                    anchors.push(Anchor {
                        label: name.to_owned(),
                        plan,
                        input: input.clone(),
                        samples: samples.clone(),
                        truth: None,
                    });
                    report.cases.push(case);
                }
            }
            Err(e) => report.notes.push(format!(
                "The requested input could not be read, so no real-clip case was run: {e}"
            )),
        }
    } else {
        report.notes.push(
            "No --input was given, so no real recording was run. Real cases are the only ones \
             that exercise the product path."
                .to_owned(),
        );
    }

    // --- synthetic -------------------------------------------------------
    if opts.synthetic {
        for f in fixtures() {
            let (reference, samples) = render_fixture(&f);
            let input = InputInfo {
                kind: "synthetic".to_owned(),
                file: None,
                sample_rate: 16_000,
                channels: 1,
                bits_per_sample: 0,
                samples: samples.len(),
                duration_ms: samples.len() as i64 / 16,
                peak_sample: samples.iter().fold(0f32, |m, v| m.max(v.abs())),
            };
            // The level-varied fixture is run twice: once on this crate's own
            // activity detector and once on the generator's intervals. The pair
            // is the experiment that separates "the gate lost the speaker" from
            // "the features cannot see the speaker".
            let mut plans = vec![TimelinePlan::Internal];
            if f.oracle_twin {
                plans.insert(0, TimelinePlan::Oracle);
            }
            for plan in plans {
                let (case, _) = run_case(CaseRun {
                    name: match plan {
                        TimelinePlan::Oracle => format!("{}#oracle", f.name),
                        _ => f.name.to_owned(),
                    },
                    anchor: None,
                    kind: f.kind,
                    input: input.clone(),
                    samples: &samples,
                    cfg: DiarizationConfig::default(),
                    plan,
                    truth: Some(reference.clone()),
                    repeats: 1,
                    dump_geometry: true,
                });
                report.cases.push(case);
            }
            // When there is no real recording, every synthetic fixture anchors
            // the sweep and the variants, so the scored tables cover the hard
            // cases and not only the easy one. A scored table over a single
            // clean fixture is how a tuning number gets chosen.
            if opts.input.is_none() {
                anchors.push(Anchor {
                    label: f.name.to_owned(),
                    plan: TimelinePlan::Internal,
                    input,
                    samples,
                    truth: Some(reference),
                });
            }
        }
    }

    // --- sweep and variants, over every anchor ---------------------------
    //
    // Both the internal and the external-timeline sweep run, because they are
    // different experiments: the external one is the product path, and the
    // comparison between the two is the measurement that says whether a VAD
    // change would move the speaker count.
    if let Some(cap) = opts.max_anchors {
        anchors.truncate(cap.max(1));
    }
    for a in &anchors {
        if opts.sweep {
            for d in MERGE_SWEEP {
                let mut c = DiarizationConfig::default();
                c.clusters.count = crate::config::SpeakerCount::Threshold {
                    merge_distance: d,
                    max_speakers: 8,
                };
                // A reference means the window dump is needed, and the dump is a
                // second feature pass. On a 144 s fixture that is about 200 ms
                // per point, which is worth paying: a sweep that cannot be
                // scored is a sweep that cannot answer "was any threshold
                // right".
                let (case, _) = run_case(CaseRun {
                    name: format!("sweep-merge-{d:.2}"),
                    anchor: Some(a.label.clone()),
                    kind: CaseKind::Variant,
                    input: a.input.clone(),
                    samples: &a.samples,
                    cfg: c,
                    plan: a.plan,
                    truth: a.truth.clone(),
                    repeats: 1,
                    dump_geometry: a.truth.is_some(),
                });
                report.merge_sweep.push(SweepPoint {
                    case: a.label.clone(),
                    activity: source_of(a.plan),
                    requested: d,
                    effective: case.config.effective_merge_distance,
                    speaker_count: case.result.speaker_count,
                    span_count: case.result.span_count,
                    speaker_change_count: case.result.speaker_change_count,
                    mean_turn_ms: case.result.turn_durations.mean.unwrap_or(0.0),
                    rule: case.timing.rule.clone(),
                    wall_ms: case.timing.wall_ms,
                    reference: case.reference,
                });
            }
        }
        if opts.variants {
            for (name, mutate) in variants() {
                let mut c = DiarizationConfig::default();
                mutate(&mut c);
                // The geometry is not reported for a variant, but the *score* is
                // whenever a reference exists, and the score is the only part of
                // a variant that can say whether the change helped.
                let scored = a.truth.is_some();
                let (case, _) = run_case(CaseRun {
                    name: format!("variant-{name}"),
                    anchor: Some(a.label.clone()),
                    kind: CaseKind::Variant,
                    input: a.input.clone(),
                    samples: &a.samples,
                    cfg: c,
                    plan: a.plan,
                    truth: a.truth.clone(),
                    repeats: 1,
                    dump_geometry: scored,
                });
                report.variants.push(case);
            }
        }
    }

    report
}

/// Which activity sources the primary real-clip cases run under.
fn timelines_to_run(opts: &BenchOptions) -> Vec<TimelinePlan> {
    let mut plans = vec![TimelinePlan::Internal];
    if opts.external_timeline && external_timeline_available() {
        plans.push(TimelinePlan::External);
    }
    plans
}

/// An input the sweep and variants re-run.
struct Anchor {
    label: String,
    plan: TimelinePlan,
    input: InputInfo,
    samples: Vec<f32>,
    truth: Option<Vec<i32>>,
}

fn source_of(plan: TimelinePlan) -> ActivitySource {
    match plan {
        TimelinePlan::Internal => ActivitySource::Internal,
        TimelinePlan::External => ActivitySource::External,
        TimelinePlan::Oracle => ActivitySource::Oracle,
    }
}

/// Where a case's activity mask comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TimelinePlan {
    Internal,
    External,
    Oracle,
}

/// One case, as handed to [`run_case`].
struct CaseRun<'a> {
    name: String,
    anchor: Option<String>,
    kind: CaseKind,
    input: InputInfo,
    samples: &'a [f32],
    cfg: DiarizationConfig,
    plan: TimelinePlan,
    truth: Option<Vec<i32>>,
    repeats: usize,
    dump_geometry: bool,
}

fn run_case(run: CaseRun<'_>) -> (CaseReport, DiarizationConfig) {
    let CaseRun {
        name,
        anchor,
        kind,
        input,
        samples,
        cfg,
        plan,
        truth,
        repeats,
        dump_geometry,
    } = run;
    let audio = AudioBuffer::mono(samples.to_vec(), 16_000, Time::ZERO);
    let engine = HeuristicDiarizer::with_config(cfg.clone());
    let token = CancelToken::new();

    let timeline = match plan {
        TimelinePlan::Internal => None,
        TimelinePlan::External => external_timeline(&audio, &token),
        TimelinePlan::Oracle => truth
            .as_ref()
            .map(|t| oracle_timeline(t, cfg.features.frame_ms())),
    };

    let mut result = DiarizationResult::default();
    let mut wall = std::time::Duration::ZERO;
    let mut timings = crate::engine::DiarizationTimings::default();
    for _ in 0..repeats.max(1) {
        let t0 = Instant::now();
        result = engine
            .diarize_with_timeline(&audio, timeline.as_ref(), &token, &|_| {})
            .expect("diarization must succeed");
        wall = t0.elapsed();
        timings = engine.last_timings().unwrap_or_default();
    }

    let mut activity = activity_report(&audio, &cfg, plan, timeline.as_ref());
    // The engine walks the frame grid in `window_hop_frames` strides and keeps
    // a window only if `min_window_speech_ratio` of it is active, so this is the
    // fraction of candidate windows that reached the clustering.
    let candidate_windows = timings
        .frames
        .div_ceil(cfg.clusters.window_hop_frames.max(1))
        .max(1);
    activity.window_keep_ratio = Some(timings.windows as f64 / candidate_windows as f64);
    let (geometry, reference) = if dump_geometry {
        let windows = engine
            .debug_windows_with_timeline(&audio, timeline.as_ref(), &token)
            .expect("window dump");
        let labels = truth
            .as_ref()
            .map(|t| eval::label_windows(&windows, t, cfg.features.frame_ms(), 0));
        let geometry = label_free_geometry(&windows, &cfg, labels.as_deref());
        let reference = match (labels.as_deref(), truth.as_ref()) {
            (Some(l), Some(t)) => Some(reference_report(&windows, l, t, &activity)),
            _ => None,
        };
        (geometry, reference)
    } else {
        (None, None)
    };

    let case = CaseReport {
        name,
        anchor,
        kind,
        input,
        config: ConfigInfo::of(&cfg),
        activity,
        result: ResultReport::of(&result, &audio),
        timing: TimingReport::of(&timings, wall, &audio, &cfg),
        geometry,
        reference,
        // A variant exists to be counted. See `CaseReport::segments`.
        segments: if kind == CaseKind::Variant {
            Vec::new()
        } else {
            result
                .spans
                .iter()
                .map(|s| SpanInfo {
                    speaker: s.speaker,
                    start_ms: s.start.as_millis(),
                    end_ms: s.end.as_millis(),
                    duration_ms: (s.end.as_millis() - s.start.as_millis()).max(0),
                    score: s.score,
                })
                .collect()
        },
    };
    (case, cfg)
}

/// Reconstruct the mask the engine used, so the report can describe it.
///
/// This mirrors `HeuristicDiarizer::run`'s loop over usable frames and the
/// timeline it derived. It is a *reproduction*, not a re-measurement: the
/// authoritative counts are the engine's own `DiarizationTimings`, and the only
/// thing recomputed here is the per-frame boolean, which is cheap and which no
/// public API exposes.
fn active_mask(
    audio: &AudioBuffer,
    cfg: &DiarizationConfig,
    timeline: &SpeechTimeline,
) -> Vec<bool> {
    let fe = crate::dsp::FrontEnd::new(cfg.features, cfg.activity);
    let usable = fe.usable_frames(audio.samples.len());
    let frame_ms = cfg.features.frame_ms();
    let start_ms = audio.start_time.as_millis();
    let mut mask = vec![false; usable.len()];
    let mut seg = 0usize;
    for (i, f) in usable.enumerate() {
        let at = start_ms + fe.frame_time_ms(f);
        while seg < timeline.segments.len() && timeline.segments[seg].end.as_millis() <= at {
            seg += 1;
        }
        if seg < timeline.segments.len() && timeline.segments[seg].start.as_millis() <= at {
            mask[i] = true;
        }
    }
    let _ = frame_ms;
    mask
}

/// Describe the activity stage.
fn activity_report(
    audio: &AudioBuffer,
    cfg: &DiarizationConfig,
    plan: TimelinePlan,
    timeline: Option<&SpeechTimeline>,
) -> ActivityReport {
    let internal_timeline;
    let tl: &SpeechTimeline = match timeline {
        Some(t) => t,
        None => {
            let mut fe = crate::dsp::FrontEnd::new(cfg.features, cfg.activity);
            let a = vad::scan(&mut fe, &audio.samples, || false, |_| {}, usize::MAX);
            let th = vad::derive_thresholds(&a, &cfg.activity);
            let mask = vad::activity_mask(&a, &th);
            internal_timeline = vad::mask_to_timeline(
                &mask,
                cfg.features.frame_ms(),
                audio.start_time.as_millis(),
                &cfg.activity,
            );
            &internal_timeline
        }
    };
    let frame_mask = active_mask(audio, cfg, tl);
    let frames = frame_mask.len();
    let active_frames = frame_mask.iter().filter(|b| **b).count();
    let durations: Vec<f64> = tl
        .segments
        .iter()
        .map(|s| (s.end.as_millis() - s.start.as_millis()).max(0) as f64)
        .collect();
    let dur_ms = audio.duration().as_millis();
    ActivityReport {
        source: match plan {
            TimelinePlan::Internal => ActivitySource::Internal,
            TimelinePlan::External => ActivitySource::External,
            TimelinePlan::Oracle => ActivitySource::Oracle,
        },
        frames,
        active_frames,
        active_frame_ratio: if frames > 0 {
            active_frames as f64 / frames as f64
        } else {
            0.0
        },
        timeline_segments: tl.segments.len(),
        timeline_speech_ms: tl.speech_duration_ms,
        timeline_coverage: if dur_ms > 0 {
            tl.speech_duration_ms as f64 / dur_ms as f64
        } else {
            0.0
        },
        segment_durations: Stat::of(durations),
        window_keep_ratio: None,
        frame_mask,
    }
}

/// A perfect activity mask, built from the generator's own per-frame labels.
///
/// This is the experiment that separates "the activity gate discarded a
/// speaker" from "the features cannot see a speaker". It exists only where a
/// generated reference exists, and it is labelled [`ActivitySource::Oracle`] in
/// the output so it can never be mistaken for something the product can do.
fn oracle_timeline(truth: &[i32], frame_ms: i64) -> SpeechTimeline {
    let pad = 30i64 * frame_ms; // the same 300 ms hangover the internal detector uses
    let mut runs: Vec<(i64, i64)> = Vec::new();
    let mut i = 0usize;
    while i < truth.len() {
        if truth[i] == eval::NO_SPEAKER {
            i += 1;
            continue;
        }
        let start = i;
        while i < truth.len() && truth[i] != eval::NO_SPEAKER {
            i += 1;
        }
        runs.push((start as i64 * frame_ms - pad, i as i64 * frame_ms + pad));
    }
    // Bridge gaps under the same 300 ms the internal detector bridges, so the
    // two masks differ in *which* frames are kept and not in how they are
    // smoothed afterwards.
    let mut merged: Vec<(i64, i64)> = Vec::new();
    for s in runs {
        match merged.last_mut() {
            Some(prev) if s.0 - prev.1 <= pad => prev.1 = s.1,
            _ => merged.push(s),
        }
    }
    let segments: Vec<vaani_core::pipeline::SpeechSegment> = merged
        .into_iter()
        .map(|(a, b)| vaani_core::pipeline::SpeechSegment {
            start: Time::from_millis(a.max(0)),
            end: Time::from_millis(b.max(0)),
            confidence: Some(1.0),
        })
        .collect();
    let speech_duration_ms = segments
        .iter()
        .map(|s| s.end.as_millis() - s.start.as_millis())
        .sum();
    SpeechTimeline {
        segments,
        speech_duration_ms,
    }
}

/// Whether this build can produce an external timeline at all.
#[must_use]
pub fn external_timeline_available() -> bool {
    cfg!(feature = "vad-probe")
}

/// `vaani-vad`'s timeline for `audio`, when the `vad-probe` feature is on.
///
/// Without the feature this returns `None` and [`notes`] says so, rather than
/// silently substituting the internal detector for the pipeline's real one.
#[cfg(feature = "vad-probe")]
fn external_timeline(audio: &AudioBuffer, cancel: &CancelToken) -> Option<SpeechTimeline> {
    use vaani_core::pipeline::VadEngine;
    let vad = vaani_vad::EnergyVad::new();
    vad.detect(audio, cancel).ok()
}

/// See the `vad-probe` build of this function.
#[cfg(not(feature = "vad-probe"))]
fn external_timeline(_audio: &AudioBuffer, _cancel: &CancelToken) -> Option<SpeechTimeline> {
    None
}

/// The `vad-probe` note.
#[must_use]
pub fn vad_probe_note() -> String {
    if cfg!(feature = "vad-probe") {
        "Cases with source = \"external\" were produced by vaani-vad's EnergyVad, the engine \
         vaani-app uses. That is another crate's output being measured, not this crate's."
            .to_owned()
    } else {
        "The `vad-probe` cargo feature is OFF, so no case ran with an external SpeechTimeline. \
         That is the path vaani-app takes, so the product's own VAD is not represented in this \
         report. Build with `--features vad-probe` to include it."
            .to_owned()
    }
}

/// Score a case against a generated reference.
fn reference_report(
    windows: &[LabelledWindow],
    labels: &[eval::WindowLabel],
    truth: &[i32],
    activity: &ActivityReport,
) -> ReferenceReport {
    let q = eval::measure_quality(windows, labels);
    let mut per_speaker: std::collections::BTreeMap<i32, usize> = Default::default();
    for l in labels.iter().filter(|l| l.pure) {
        *per_speaker.entry(l.speaker).or_default() += 1;
    }
    let max_speaker = truth.iter().copied().max().unwrap_or(0);
    let mut true_frames: std::collections::BTreeMap<i32, usize> = Default::default();
    let mut active_frames: std::collections::BTreeMap<i32, usize> = Default::default();
    for (i, t) in truth.iter().enumerate() {
        if *t == eval::NO_SPEAKER {
            continue;
        }
        *true_frames.entry(*t).or_default() += 1;
        if activity.frame_mask.get(i).copied().unwrap_or(false) {
            *active_frames.entry(*t).or_default() += 1;
        }
    }
    let pick = |m: &std::collections::BTreeMap<i32, usize>| {
        (0..=max_speaker)
            .map(|s| (s, m.get(&s).copied().unwrap_or(0)))
            .collect::<Vec<_>>()
    };
    ReferenceReport {
        provenance: "synth::render per-frame ground truth, produced by the same call that \
                     produced the audio. This is a generator, not a human annotation."
            .to_owned(),
        true_speakers: (max_speaker + 1).max(0) as usize,
        scored: q.scored,
        mixed_windows: q.geometry.mixed_windows,
        purity: finite(q.purity),
        recall: finite(q.recall),
        complete_accuracy: finite(q.complete_accuracy),
        population_margin: finite(q.geometry.margin),
        same_speaker: stat_of(&q.geometry.same),
        different_speaker: stat_of(&q.geometry.different),
        windows_per_true_speaker: pick(&per_speaker),
        true_frames_per_true_speaker: pick(&true_frames),
        active_frames_per_true_speaker: pick(&active_frames),
    }
}

fn finite(v: f64) -> Option<f64> {
    if v.is_finite() { Some(v) } else { None }
}

fn stat_of(p: &Percentiles) -> Stat {
    Stat {
        n: p.n,
        min: finite(p.min),
        p10: finite(p.p10),
        p50: finite(p.p50),
        p90: finite(p.p90),
        max: finite(p.max),
        mean: finite(p.mean),
    }
}

/// All window-to-window distances, the merge sequence and the pitch histogram.
fn label_free_geometry(
    windows: &[LabelledWindow],
    cfg: &DiarizationConfig,
    labels: Option<&[eval::WindowLabel]>,
) -> Option<GeometryReport> {
    if windows.len() < 2 {
        return None;
    }
    let rows: Vec<Vec<f32>> = windows.iter().map(|w| w.vector.clone()).collect();
    let n = rows.len();
    let matrix = cluster::distance_matrix(&rows);
    let mut pairwise: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            pairwise.push(f64::from(matrix[[i, j]]));
        }
    }
    let flat: Vec<f32> = matrix.iter().copied().collect();
    let dendro = cluster::agglomerate(n, &flat);
    drop(flat);
    let layout = Layout::of(&cfg.features);
    let raw: Vec<Vec<f32>> = windows.iter().map(|w| w.raw.clone()).collect();
    Some(GeometryReport {
        windows: n,
        pairwise: Stat::of(pairwise),
        merge: merge_report(&dendro, cfg),
        pitch: pitch_report(windows, layout.pitch),
        blocks: block_shares(&rows, &raw, layout, labels, cfg),
    })
}

fn merge_report(dendro: &cluster::Dendrogram, cfg: &DiarizationConfig) -> MergeReport {
    let rows = dendro.rows;
    let top = 12.min(rows.saturating_sub(1));
    let mut cost_at_count: Vec<CountCost> = Vec::new();
    for k in 1..=top {
        if let Some(cost) = dendro.cost_at_count(k) {
            cost_at_count.push(CountCost { k, cost });
        }
    }
    let top_relative_gaps = relative_gaps(dendro, 5);
    let costs_sampled: Vec<f64> = if dendro.costs.is_empty() {
        Vec::new()
    } else {
        let m = dendro.costs.len();
        (0..64)
            .map(|i| {
                let idx = ((i as f64 / 63.0) * (m - 1) as f64).round() as usize;
                dendro.costs[idx.min(m - 1)]
            })
            .collect()
    };
    MergeReport {
        rows,
        merges: dendro.merges.len(),
        cost_at_count,
        largest_relative_gap: top_relative_gaps.first().copied(),
        threshold_cut_k: threshold_cut(dendro, cfg.effective_merge_distance(), rows),
        top_relative_gaps,
        costs_sampled,
    }
}

fn relative_gaps(dendro: &cluster::Dendrogram, keep: usize) -> Vec<RelativeGap> {
    let mut out: Vec<RelativeGap> = Vec::new();
    for (i, &from) in dendro.costs.iter().enumerate() {
        if from <= 1e-9 {
            continue;
        }
        let Some(&to) = dendro.costs.get(i + 1) else {
            break;
        };
        out.push(RelativeGap {
            after_k: dendro.rows - i - 1,
            from,
            to,
            ratio: to / from,
        });
    }
    out.sort_by(|a, b| {
        b.ratio
            .partial_cmp(&a.ratio)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.truncate(keep);
    out
}

/// The cluster count the configured threshold selects, computed from the
/// sequence. Mirrors `cluster::choose_count`'s `Threshold` arm.
fn threshold_cut(dendro: &cluster::Dendrogram, merge_distance: f64, cap: usize) -> Option<usize> {
    let mut best = cap.max(1);
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
    (best > 0).then_some(best)
}

/// The absolute pitch dimension, as an occupancy histogram over semitones.
fn pitch_report(windows: &[LabelledWindow], pitch_index: usize) -> PitchReport {
    let bin = 1.0f64;
    let semitones: Vec<f64> = windows
        .iter()
        .filter_map(|w| {
            let hz = 2f64.powf(f64::from(w.raw[pitch_index]));
            (hz > 1.0).then(|| 12.0 * (hz / 16_000.0).log2())
        })
        .collect();
    if semitones.is_empty() {
        return PitchReport {
            windows: 0,
            semitones: Stat::default(),
            bin_semitones: bin,
            bin_origin_semitones: None,
            occupied_bins_above_10pct: 0,
            histogram: Vec::new(),
            largest_internal_gap_semitones: 0.0,
            headroom_above_semitones: 0.0,
        };
    }
    let stat = stat_of(&Percentiles::of(semitones.clone()));
    let lo = semitones.iter().copied().fold(f64::MAX, f64::min).floor();
    let hi = semitones.iter().copied().fold(f64::MIN, f64::max).ceil();
    let bins = (((hi - lo) / bin).ceil() as usize + 1).max(1);
    let mut counts = vec![0usize; bins];
    for s in &semitones {
        let idx = (((s - lo) / bin).round() as isize).clamp(0, bins as isize - 1);
        counts[idx as usize] += 1;
    }
    let peak = counts.iter().copied().max().unwrap_or(0);
    let cut = ((peak as f64) * 0.10).ceil() as usize;
    let cut = cut.max(1);
    let occupied: Vec<usize> = (0..bins).filter(|&b| counts[b] >= cut).collect();
    let mut largest_gap = 0.0f64;
    let mut headroom = 0.0f64;
    if let (Some(&first), Some(&last)) = (occupied.first(), occupied.last()) {
        let mut run_len = 0usize;
        for occupied_here in counts
            .iter()
            .enumerate()
            .take(last + 1)
            .skip(first + 1)
            .map(|(_, c)| *c >= cut)
        {
            if occupied_here {
                largest_gap = largest_gap.max(run_len as f64 * bin);
                run_len = 0;
            } else {
                run_len += 1;
            }
        }
        largest_gap = largest_gap.max(run_len as f64 * bin);
        headroom = (bins.saturating_sub(1 + last)) as f64 * bin;
    }
    PitchReport {
        windows: semitones.len(),
        semitones: stat,
        bin_semitones: bin,
        bin_origin_semitones: Some(lo),
        occupied_bins_above_10pct: occupied.len(),
        histogram: counts
            .iter()
            .enumerate()
            .map(|(i, c)| (i as u16, *c))
            .collect(),
        largest_internal_gap_semitones: largest_gap,
        headroom_above_semitones: headroom,
    }
}

/// The per-block share of the pairwise distance.
///
/// A cosine distance decomposes additively per dimension as
/// `1 - dot = sum_j (a_j - b_j)^2 / (2 * ||a|| * ||b||)`, and accumulating that
/// per block says which blocks produce the within-speaker spread and which carry
/// the speaker signal. Without labels only the overall share exists, and that is
/// still the answer to "what is this distance made of".
///
/// The identity is exact only for equal-length `a` and `b`, so the rows are
/// L2-normalised first. In the product path they are already unit vectors --
/// `features::normalise_in_place` ends with an L2 normalisation -- and this makes
/// the function correct for any input rather than only for that one, which is
/// what makes [`Self::shares_sum_to_the_mean_distance`] a real test rather than
/// a statement about the caller.
fn block_shares(
    rows: &[Vec<f32>],
    raw: &[Vec<f32>],
    layout: Layout,
    labels: Option<&[eval::WindowLabel]>,
    cfg: &DiarizationConfig,
) -> Vec<BlockShare> {
    let blocks = block_ranges(layout);
    let unit: Vec<Vec<f32>> = rows
        .iter()
        .map(|r| {
            let n = r
                .iter()
                .map(|v| f64::from(*v) * f64::from(*v))
                .sum::<f64>()
                .sqrt();
            if n > 1e-12 {
                r.iter().map(|v| (f64::from(*v) / n) as f32).collect()
            } else {
                r.clone()
            }
        })
        .collect();
    // Three accumulators per block, with three denominators. Sharing one
    // denominator between "all pairs" and "same-speaker pairs" is the kind of
    // slip that reports a block's share as 1.14 when no cosine distance can
    // exceed 2, and it is invisible unless the shares are checked against the
    // mean pairwise distance.
    let mut all = vec![0f64; blocks.len()];
    let mut same = vec![0f64; blocks.len()];
    let mut diff = vec![0f64; blocks.len()];
    let (mut n_all, mut n_same, mut n_diff) = (0usize, 0usize, 0usize);
    for i in 0..unit.len() {
        for j in (i + 1)..unit.len() {
            let a = &unit[i];
            let b = &unit[j];
            let na: f64 = a
                .iter()
                .map(|v| f64::from(*v) * f64::from(*v))
                .sum::<f64>()
                .sqrt();
            let nb: f64 = b
                .iter()
                .map(|v| f64::from(*v) * f64::from(*v))
                .sum::<f64>()
                .sqrt();
            if na <= 1e-12 || nb <= 1e-12 {
                continue;
            }
            // A window that straddles a turn boundary has no single correct
            // label, so it is excluded rather than scored against one it does
            // not have. Without labels there is nothing to exclude.
            let usable = labels.map(|l| l[i].pure && l[j].pure).unwrap_or(true);
            if !usable {
                continue;
            }
            let is_same = match labels {
                Some(l) => l[i].speaker == l[j].speaker,
                None => true,
            };
            for (k, &(_, lo, hi, _)) in blocks.iter().enumerate() {
                let share = block_distance_share(a, b, lo, hi, na, nb);
                all[k] += share;
                if is_same {
                    same[k] += share
                } else {
                    diff[k] += share
                }
            }
            n_all += 1;
            if is_same { n_same += 1 } else { n_diff += 1 }
        }
    }
    blocks
        .iter()
        .enumerate()
        .map(|(k, (name, lo, hi, divided))| {
            let same_share = (labels.is_some() && n_same > 0).then(|| same[k] / n_same as f64);
            let diff_share = (labels.is_some() && n_diff > 0).then(|| diff[k] / n_diff as f64);
            let measured = median_measured_spread(raw, *lo, *hi);
            // The carrier is a constant and the pitch dimension is never divided
            // by a spread at all, so for those two blocks "how many unit
            // variances did this block arrive at" has no answer. Reporting the
            // ltas constant for the carrier, because it happens to be the last
            // branch of the same `if`, would be a number about a different block.
            let constant = divided
                .then(|| within_speaker_spread_of(*lo, layout))
                .flatten();
            let floor = (*divided || *name == "pitch")
                .then(|| min_spread_of(*lo, layout, &cfg.min_spread))
                .flatten();
            BlockShare {
                block: (*name).to_owned(),
                dims: hi - lo,
                mean_share: mean(all[k], n_all),
                speaker_signal: match (same_share, diff_share) {
                    (Some(s), Some(d)) => Some(d - s),
                    _ => None,
                },
                same_speaker_share: same_share,
                different_speaker_share: diff_share,
                spread_ratio: ratio(measured, constant),
                measured_median_spread: measured,
                scale_constant: constant,
                min_spread_floor: floor,
                floor_ratio: ratio(measured, floor),
            }
        })
        .collect()
}

fn ratio(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(x), Some(y)) if y.abs() > 1e-15 => Some(x / y),
        _ => None,
    }
}

/// The median, over a dimension range, of the robust spread of each dimension
/// across the recording's windows.
///
/// `MAD_TO_SIGMA` times the median **absolute** deviation of the median-centred
/// column, which is the same estimator `features::robust_sd` uses. It answers
/// "how much does this dimension actually move in this recording", which is the
/// question both `within_speaker_spread` and `MinSpread` are constants for.
///
/// The absolute value is taken **before** the median. A median of *signed*
/// deviations is zero for every input, which is the kind of bug that reports
/// "this dimension does not vary" for all 83 of them and looks like a
/// measurement until someone checks that a number that cannot be non-zero came
/// back as zero.
fn median_measured_spread(raw: &[Vec<f32>], lo: usize, hi: usize) -> Option<f64> {
    let mut out = Vec::new();
    for j in lo..hi.min(raw.first().map(|r| r.len()).unwrap_or(0)) {
        let mut col: Vec<f64> = raw.iter().map(|r| f64::from(r[j])).collect();
        if col.len() < 2 {
            continue;
        }
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let centre = col[col.len() / 2];
        for v in &mut col {
            *v = (*v - centre).abs();
        }
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        out.push(col[col.len() / 2] * 1.482_602);
    }
    if out.is_empty() {
        return None;
    }
    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(out[out.len() / 2])
}

/// `features::within_speaker_spread` for the block starting at `lo`.
fn within_speaker_spread_of(lo: usize, layout: Layout) -> Option<f64> {
    let ranges = block_ranges(layout);
    let (_, start, _, _) = ranges.iter().find(|(_, s, _, _)| *s == lo)?;
    // The constants are per block and every dimension in a block shares one, so
    // the first dimension of the block answers for all of them.
    Some(f64::from(crate::features::within_speaker_spread(
        *start, &layout,
    )))
}

/// `config::MinSpread` for the block starting at `lo`.
fn min_spread_of(lo: usize, layout: Layout, spread: &crate::config::MinSpread) -> Option<f64> {
    let ranges = block_ranges(layout);
    let (_, start, _, _) = ranges.iter().find(|(_, s, _, _)| *s == lo)?;
    Some(f64::from(crate::features::min_spread_for(
        *start, &layout, spread,
    )))
}

fn mean(total: f64, n: usize) -> f64 {
    if n > 0 { total / n as f64 } else { 0.0 }
}

fn block_distance_share(a: &[f32], b: &[f32], lo: usize, hi: usize, na: f64, nb: f64) -> f64 {
    let mut sq = 0f64;
    for v in &a[lo..hi] {
        sq += f64::from(*v) * f64::from(*v);
    }
    for v in &b[lo..hi] {
        sq += f64::from(*v) * f64::from(*v);
    }
    let mut cross = 0f64;
    for u in 0..(hi - lo) {
        cross += f64::from(a[lo + u]) * f64::from(b[lo + u]);
    }
    (sq - 2.0 * cross) / (2.0 * na * nb)
}

/// One feature block: its name, its dimension range, and whether
/// `normalise_in_place` divides it by a per-block constant scale.
///
/// `pitch` is `false` because it is the *absolute* dimension -- it is shifted by
/// a fixed reference and multiplied by a fixed scale, never divided by anything
/// measured -- and `carrier` is `false` because it is a constant by definition.
/// For both, "how many unit variances did this block arrive at" has no answer,
/// and reporting the neighbouring block's constant would be a number about a
/// different block.
fn block_ranges(layout: Layout) -> Vec<(&'static str, usize, usize, bool)> {
    vec![
        ("cepstra", layout.cepstra, layout.delta, true),
        ("delta", layout.delta, layout.double_delta, true),
        ("double_delta", layout.double_delta, layout.pitch, true),
        ("pitch", layout.pitch, layout.voicedness, false),
        ("voicedness", layout.voicedness, layout.shape, true),
        ("spectral_shape", layout.shape, layout.bands, true),
        ("band_ratios", layout.bands, layout.ltas, true),
        ("ltas", layout.ltas, layout.total, true),
        ("carrier", layout.total, layout.total + 1, false),
    ]
}

/// A named synthetic fixture.
struct Fixture {
    name: &'static str,
    kind: CaseKind,
    total_ms: i64,
    utterances: Vec<synth::Utterance>,
    voices: Vec<synth::Voice>,
    noise: f32,
    /// Per-speaker gain, multiplied into each utterance's own gain.
    speaker_gain: Vec<f32>,
    /// Whether to run this fixture a second time on the oracle timeline.
    oracle_twin: bool,
}

/// Render a fixture, returning `(per-frame reference, samples)`.
fn render_fixture(f: &Fixture) -> (Vec<i32>, Vec<f32>) {
    let utterances: Vec<synth::Utterance> = f
        .utterances
        .iter()
        .map(|u| {
            let g = f.speaker_gain.get(u.speaker).copied().unwrap_or(1.0);
            u.with_gain(u.gain * g)
        })
        .collect();
    let (samples, truth) = synth::render(&utterances, &f.voices, f.total_ms, 16_000, f.noise, 2024);
    (truth, samples)
}

/// The synthetic fixtures the harness runs.
fn fixtures() -> Vec<Fixture> {
    let two = synth::two_voices();
    let three = synth::three_voices();
    vec![
        // The documented operating region: 12 s turns, two clearly different
        // voices.
        Fixture {
            name: "synthetic-2spk-12s-turns",
            kind: CaseKind::SyntheticReference,
            total_ms: 12_000 * 12,
            utterances: synth::alternate(&[0, 1], 0, 12_000, 12),
            voices: two.clone(),
            noise: 0.003,
            speaker_gain: vec![1.0, 1.0],
            oracle_twin: false,
        },
        // A real meeting's turn length, three people.
        Fixture {
            name: "synthetic-3spk-3s-turns",
            kind: CaseKind::SyntheticReference,
            total_ms: 3_000 * 40,
            utterances: synth::alternate(&[0, 1, 2], 0, 3_000, 40),
            voices: three.clone(),
            noise: 0.003,
            speaker_gain: vec![1.0, 1.0, 1.0],
            oracle_twin: false,
        },
        // Two of the three share a fundamental to within 15%, so pitch alone
        // cannot separate them. This is the case that decides whether the
        // spectral blocks earn their weight.
        Fixture {
            name: "synthetic-3spk-similar-pitch-12s-turns",
            kind: CaseKind::SyntheticReference,
            total_ms: 12_000 * 12,
            utterances: synth::alternate(&[0, 1, 2], 0, 12_000, 12),
            voices: three.clone(),
            noise: 0.003,
            speaker_gain: vec![1.0, 1.0, 1.0],
            oracle_twin: false,
        },
        // One speaker 20 dB below the other two. The interesting part is the
        // pair of runs: this crate's own activity detector against the
        // generator's intervals.
        Fixture {
            name: "synthetic-3spk-one-20db-down",
            kind: CaseKind::SyntheticReference,
            total_ms: 12_000 * 12,
            utterances: synth::alternate(&[0, 1, 2], 0, 12_000, 12),
            voices: three.clone(),
            noise: 0.003,
            speaker_gain: vec![1.0, 1.0, 0.1],
            oracle_twin: true,
        },
        // One person, two minutes: the over-split direction.
        Fixture {
            name: "synthetic-1spk-2min",
            kind: CaseKind::SyntheticReference,
            total_ms: 4_000 * 30,
            utterances: synth::alternate(&[0], 0, 4_000, 30),
            voices: vec![two[0].clone()],
            noise: 0.003,
            speaker_gain: vec![1.0],
            oracle_twin: false,
        },
    ]
}

/// Configuration variants, each isolating one documented lever.
///
/// Every one of these is a *measurement*, not a candidate change. The point of
/// listing them is that "which knob is responsible" is a question with a table
/// for an answer, and the table has to be produced by running the same input
/// through the same harness with one number changed.
/// A configuration variant: a name and the one change it makes.
type Variant = (&'static str, fn(&mut DiarizationConfig));

fn variants() -> Vec<Variant> {
    use crate::config::SegmentationBias;
    vec![
        ("default", |_| {}),
        ("window-3.0s", |c| {
            c.clusters.window_frames = 300;
            c.clusters.window_hop_frames = 150;
        }),
        ("window-0.75s", |c| {
            c.clusters.window_frames = 75;
            c.clusters.window_hop_frames = 38;
        }),
        ("feature-smooth-0", |c| {
            c.clusters.feature_smooth_half_windows = 0
        }),
        ("feature-smooth-6", |c| {
            c.clusters.feature_smooth_half_windows = 6
        }),
        ("min-window-speech-0.5", |c| {
            c.clusters.min_window_speech_ratio = 0.5
        }),
        ("min-window-speech-0.0", |c| {
            c.clusters.min_window_speech_ratio = 0.0
        }),
        ("merge-readily", |c| {
            c.bias = SegmentationBias::MergeReadily;
        }),
        ("auto-gap", |c| {
            c.clusters.count = crate::config::SpeakerCount::Auto {
                max_speakers: 8,
                min_gap_ratio: 1.15,
            };
        }),
        ("carrier-4.0", |c| c.features.weights.carrier = 4.0),
        ("carrier-0.25", |c| c.features.weights.carrier = 0.25),
        ("pitch-weight-0", |c| c.features.weights.pitch = 0.0),
        ("voicedness-weight-0", |c| {
            c.features.weights.voicedness = 0.0
        }),
        ("cepstra-weight-0", |c| c.features.weights.cepstra = 0.0),
        ("spectral-weights-x4", |c| {
            c.features.weights.cepstra *= 4.0;
            c.features.weights.spectral_shape *= 4.0;
            c.features.weights.band_ratios *= 4.0;
            c.features.weights.ltas *= 4.0;
            c.features.weights.voicedness *= 4.0;
        }),
        ("median-filter-1", |c| c.clusters.median_filter_frames = 1),
        ("median-filter-61", |c| c.clusters.median_filter_frames = 61),
        ("min-turn-300ms", |c| c.clusters.min_turn_ms = 300),
        ("min-cluster-speech-10s", |c| {
            c.clusters.min_cluster_speech_ms = 10_000
        }),
    ]
}

fn stem(path: &std::path::Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "clip".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_population_is_null_and_not_zero() {
        let s = Stat::of(Vec::new());
        assert_eq!(s.n, 0);
        assert_eq!(s.min, None);
        assert_eq!(s.mean, None);
        let json = serde_json::to_string(&s).expect("serialize");
        assert!(json.contains("\"min\":null"), "{json}");
    }

    #[test]
    fn a_stat_reports_its_population_size_and_quantiles() {
        let s = Stat::of((0..101).map(|i| i as f64).collect());
        assert_eq!(s.n, 101);
        assert_eq!(s.min, Some(0.0));
        assert_eq!(s.max, Some(100.0));
        assert_eq!(s.p50, Some(50.0));
        assert!((s.mean.unwrap() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn an_empty_report_serialises_and_carries_the_caveats() {
        let r = BenchReport::empty();
        let json = r.to_json_pretty().expect("serialize");
        assert!(json.contains("STRUCTURE AND TIMING ONLY"), "{json}");
        assert!(
            !json.contains("NaN"),
            "NaN is not serialisable and must not appear"
        );
        let back: BenchReport = serde_json::from_str(&json).expect("round trip");
        assert_eq!(back.schema, SCHEMA);
        assert_eq!(back.notes, r.notes);
        assert!(back.cases.is_empty());
    }

    #[test]
    fn the_oracle_timeline_covers_exactly_the_labelled_frames_plus_hangover() {
        // Four frames of speaker 0, three of speaker 1, separated by 1.5 s of
        // silence. 300 ms of hangover either side of each run leaves the gap
        // visible, which is the case worth asserting.
        let mut truth = vec![0i32; 400];
        truth.extend(std::iter::repeat_n(-1, 150));
        truth.extend(std::iter::repeat_n(1, 300));
        let tl = oracle_timeline(&truth, 10);
        assert_eq!(tl.segments.len(), 2, "{tl:?}");
        assert_eq!(tl.segments[0].start.as_millis(), 0);
        assert_eq!(tl.segments[0].end.as_millis(), 4_000 + 300);
        assert_eq!(tl.segments[1].start.as_millis(), 5_500 - 300);
        assert_eq!(tl.segments[1].end.as_millis(), 8_500 + 300);
        assert!(
            tl.segments[0].end < tl.segments[1].start,
            "segments must not overlap"
        );
        // 4,000 ms of speech padded by 300 either side is 4,300; the second run
        // starts at 5,200 (its 300 ms of left hangover is not clipped, because
        // it lands inside the recording) and ends at 8,800, so 3,600.
        assert_eq!(tl.speech_duration_ms, 4_300 + 3_600);
    }

    #[test]
    fn a_short_gap_is_bridged_by_the_oracle_exactly_as_the_detector_bridges_it() {
        // 100 ms of silence between two runs, and 300 ms of hangover on each
        // side of it. The detector's own order is hangover, then bridging, so
        // this collapses into one segment -- and the oracle must agree, or the
        // oracle/internol comparison would be measuring two different
        // smoothers rather than two different masks.
        let truth = vec![0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1];
        let tl = oracle_timeline(&truth, 10);
        assert_eq!(tl.segments.len(), 1, "{tl:?}");
    }

    #[test]
    fn a_fully_silent_reference_produces_no_timeline() {
        assert!(oracle_timeline(&[-1, -1, -1], 10).segments.is_empty());
    }

    #[test]
    fn speaker_changes_count_boundaries_not_spans() {
        let d = DiarizationResult {
            spans: vec![
                vaani_core::pipeline::SpeakerSpan {
                    speaker: 0,
                    start: Time(0),
                    end: Time(1000),
                    score: None,
                },
                vaani_core::pipeline::SpeakerSpan {
                    speaker: 0,
                    start: Time(1000),
                    end: Time(2000),
                    score: None,
                },
                vaani_core::pipeline::SpeakerSpan {
                    speaker: 1,
                    start: Time(2000),
                    end: Time(3000),
                    score: None,
                },
                vaani_core::pipeline::SpeakerSpan {
                    speaker: 0,
                    start: Time(3000),
                    end: Time(4000),
                    score: None,
                },
            ],
            num_speakers: 2,
            elapsed_ms: 0,
        };
        let audio = AudioBuffer::mono(vec![0.0; 640_000], 16_000, Time::ZERO);
        let r = ResultReport::of(&d, &audio);
        assert_eq!(r.span_count, 4);
        assert_eq!(r.speaker_change_count, 2);
        assert_eq!(r.speaker_count, 2);
        assert!((r.talk_share[0] - 0.75).abs() < 1e-9);
        assert!((r.talk_share[1] - 0.25).abs() < 1e-9);
        // 4 s of spans inside a 40 s buffer.
        assert!((r.coverage - 0.1).abs() < 1e-9, "coverage {}", r.coverage);
    }

    #[test]
    fn a_result_report_on_an_empty_result_claims_nothing() {
        let audio = AudioBuffer::mono(Vec::new(), 16_000, Time::ZERO);
        let r = ResultReport::of(&DiarizationResult::default(), &audio);
        assert_eq!(r.speaker_count, 0);
        assert_eq!(r.span_count, 0);
        assert_eq!(r.speaker_change_count, 0);
        assert_eq!(r.speaker_changes_per_minute, None);
        assert_eq!(r.turn_scores.n, 0);
        assert_eq!(r.turn_durations.n, 0);
    }

    #[test]
    fn the_pitch_histogram_separates_an_octave() {
        // Two clusters an octave apart, a tenth of a semitone of jitter each.
        let mut vals = Vec::new();
        for i in 0..40 {
            vals.push((100.0 * (1.0 + i as f64 * 0.0006)).log2() as f32);
        }
        for i in 0..40 {
            vals.push((200.0 * (1.0 + i as f64 * 0.0006)).log2() as f32);
        }
        let p = pitch_report(&fake_windows(&vals), 57);
        assert_eq!(p.windows, 80);
        // 100 Hz to 200 Hz is 12 semitones; the occupied bins must leave a
        // clear gap in the middle of the histogram.
        assert!(
            p.largest_internal_gap_semitones >= 6.0,
            "expected a clear pitch gap, got {}",
            p.largest_internal_gap_semitones
        );
        assert!(p.occupied_bins_above_10pct <= 8, "{:?}", p.histogram);
    }

    #[test]
    fn the_pitch_histogram_finds_no_gap_in_a_continuum() {
        // A third of an octave of continuous variation, the kind one expressive
        // speaker produces.
        let v: Vec<f32> = (0..200)
            .map(|i| (100.0 * 2f32.powf(i as f32 / 600.0)).log2())
            .collect();
        let p = pitch_report(&fake_windows(&v), 57);
        assert_eq!(p.windows, 200);
        assert_eq!(
            p.largest_internal_gap_semitones, 0.0,
            "histogram {:?}",
            p.histogram
        );
    }

    #[test]
    fn an_unvoiced_window_population_reports_nothing_rather_than_zero() {
        let v: Vec<f32> = vec![-40.0, -40.0, -40.0];
        let p = pitch_report(&fake_windows(&v), 57);
        assert_eq!(p.windows, 0);
        assert_eq!(p.largest_internal_gap_semitones, 0.0);
        assert_eq!(p.bin_origin_semitones, None);
        assert!(p.histogram.is_empty());
    }

    /// `LabelledWindow`s whose only populated dimensions are `pitch` and the
    /// carrier.
    fn fake_windows(pitch: &[f32]) -> Vec<LabelledWindow> {
        let layout = Layout::of(&crate::config::FeatureParams::default());
        pitch
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mut vector = vec![0f32; layout.total + 1];
                vector[layout.pitch] = *p;
                vector[layout.total] = 1.0;
                LabelledWindow {
                    start: Time((i * 750) as i64),
                    end: Time((i * 750 + 1500) as i64),
                    speech_ms: 1500,
                    raw: vector.clone(),
                    vector,
                    cluster: 0,
                }
            })
            .collect()
    }

    #[test]
    fn a_config_info_records_the_effective_distance_not_the_base() {
        let cfg = DiarizationConfig::default();
        let info = ConfigInfo::of(&cfg);
        // 1.05 base x 0.85 SplitEasily, which is the number that actually cuts.
        assert!((info.effective_merge_distance - 0.8925).abs() < 1e-9);
        assert_eq!(info.base_merge_distance, Some(1.05));
        assert!(info.policy.contains("threshold"));
        assert_eq!(info.max_speakers, Some(8));
        assert_eq!(info.window_frames, 150);
    }

    #[test]
    fn a_fixed_policy_records_no_cap_and_an_infinite_effective_distance() {
        let info = ConfigInfo::of(&DiarizationConfig::default().with_speaker_count(3));
        assert!(info.effective_merge_distance.is_infinite());
        assert_eq!(info.max_speakers, None);
        assert!(info.policy.contains("fixed(3)"));
    }

    #[test]
    fn the_merge_sweep_brackets_the_shipped_distance_on_both_sides() {
        let d = DiarizationConfig::default().effective_merge_distance();
        assert!(
            MERGE_SWEEP.iter().any(|&v| v < d),
            "no point below the default"
        );
        assert!(
            MERGE_SWEEP.iter().any(|&v| v > d),
            "no point above the default"
        );
        let mut sorted = MERGE_SWEEP.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(sorted, MERGE_SWEEP.to_vec(), "the sweep must be ordered");
    }

    #[test]
    fn threshold_cut_walks_the_sequence_from_the_bottom() {
        // Costs 0.1, 0.2, 0.3, 0.9 for four merges over five rows: reaching k
        // takes `rows - k` merges, so the merge that produces k=1 costs 0.9,
        // k=2 costs 0.3, k=3 costs 0.2 and k=4 costs 0.1. The rule keeps the
        // *largest* k whose producing merge is affordable, which is the first k
        // scanning upwards whose cost fits.
        let dendro = cluster::Dendrogram {
            merges: vec![(0, 1), (0, 2), (0, 3), (0, 4)],
            costs: vec![0.1, 0.2, 0.3, 0.9],
            rows: 5,
        };
        assert_eq!(threshold_cut(&dendro, 0.15, 8), Some(4));
        assert_eq!(threshold_cut(&dendro, 0.25, 8), Some(3));
        assert_eq!(threshold_cut(&dendro, 0.35, 8), Some(2));
        assert_eq!(threshold_cut(&dendro, 0.95, 8), Some(1));
        // Nothing fits, so the cap is returned and read as "more than N".
        assert_eq!(threshold_cut(&dendro, 0.05, 8), Some(8));
    }

    #[test]
    fn relative_gaps_report_the_jump_and_its_count() {
        let dendro = cluster::Dendrogram {
            merges: vec![(0, 1), (0, 2), (0, 3), (0, 4)],
            costs: vec![0.1, 0.2, 0.3, 0.9],
            rows: 5,
        };
        let gaps = relative_gaps(&dendro, 5);
        // The 0.2 -> 0.3 step is 1.5x, the 0.3 -> 0.9 step is 3x, the 0.1 -> 0.2
        // step is 2x. The largest is 3x, sitting above 2 clusters.
        assert_eq!(gaps.len(), 3);
        assert_eq!(gaps[0].after_k, 2);
        assert!((gaps[0].ratio - 3.0).abs() < 1e-9);
    }

    /// The per-block shares must add up to the mean pairwise distance.
    ///
    /// A cosine distance on unit vectors decomposes additively per dimension,
    /// so `sum_j share_j` is the distance of each pair and the mean of that sum
    /// is the mean distance. The decomposition once divided *every* block by the
    /// same-speaker pair count while summing over all pairs, which reported a
    /// `spectral_shape` share of 1.14 on a fixture whose mean distance was 0.9 --
    /// a number no cosine distance can produce, and one that only this identity
    /// catches.
    #[test]
    fn block_shares_sum_to_the_mean_pairwise_distance() {
        let layout = Layout::of(&crate::config::FeatureParams::default());
        let rows: Vec<Vec<f32>> = (0..40)
            .map(|i| {
                let mut v = vec![0f32; layout.total + 1];
                // Two tight blobs, so the population is neither degenerate nor
                // uniformly spread.
                let centre = if i < 20 { 0.2 } else { -0.3 };
                for (j, slot) in v.iter_mut().enumerate() {
                    *slot = ((j as f32 * 0.37).sin() + centre) * 0.1;
                }
                v[layout.total] = 1.0;
                v
            })
            .collect();
        let shares = block_shares(&rows, &rows, layout, None, &DiarizationConfig::default());
        let total: f64 = shares.iter().map(|s| s.mean_share).sum();

        let mut distances = Vec::new();
        for i in 0..rows.len() {
            for j in (i + 1)..rows.len() {
                distances.push(crate::cosine_distance(&rows[i], &rows[j]));
            }
        }
        let mean_distance = distances.iter().sum::<f64>() / distances.len() as f64;
        assert!(
            (total - mean_distance).abs() < 1e-6,
            "block shares summed to {total}, mean distance is {mean_distance}"
        );
        assert!(
            total <= 2.0,
            "a cosine distance cannot exceed 2, got {total}"
        );
    }

    #[test]
    fn with_labels_the_shares_split_into_same_and_different_and_still_sum_to_the_distance() {
        let layout = Layout::of(&crate::config::FeatureParams::default());
        let rows: Vec<Vec<f32>> = (0..30)
            .map(|i| {
                let mut v = vec![0f32; layout.total + 1];
                let centre = if i < 15 { 0.15 } else { -0.35 };
                for (j, slot) in v.iter_mut().enumerate() {
                    *slot = ((j as f32 * 0.41).cos() + centre) * 0.1;
                }
                v[layout.total] = 1.0;
                v
            })
            .collect();
        let labels: Vec<eval::WindowLabel> = (0..30)
            .map(|i| eval::WindowLabel {
                speaker: if i < 15 { 0 } else { 1 },
                labelled_frames: 10,
                agreeing_frames: 10,
                pure: true,
            })
            .collect();
        let shares = block_shares(
            &rows,
            &rows,
            layout,
            Some(&labels),
            &DiarizationConfig::default(),
        );
        for s in &shares {
            assert!(s.same_speaker_share.is_some(), "{}", s.block);
            assert!(s.different_speaker_share.is_some(), "{}", s.block);
            let signal = s.speaker_signal.expect("signal");
            assert!(signal.abs() <= 2.0, "{} signal {signal}", s.block);
        }
        // Same-speaker pairs must be closer, and every block's share must be a
        // number a cosine distance could contain.
        let same: f64 = shares.iter().filter_map(|s| s.same_speaker_share).sum();
        let different: f64 = shares
            .iter()
            .filter_map(|s| s.different_speaker_share)
            .sum();
        assert!(
            different > same,
            "two separated blobs should separate: same {same}, different {different}"
        );
        assert!((different - same).abs() <= 2.0);
    }

    #[test]
    fn every_variant_changes_exactly_the_field_it_names() {
        for (name, mutate) in variants() {
            let mut c = DiarizationConfig::default();
            let before = ConfigInfo::of(&c);
            mutate(&mut c);
            let after = ConfigInfo::of(&c);
            let changed = [
                ("policy", before.policy != after.policy),
                ("bias", before.bias != after.bias),
                ("window", before.window_frames != after.window_frames),
                (
                    "smooth",
                    before.feature_smooth_half_windows != after.feature_smooth_half_windows,
                ),
                (
                    "min_window_speech",
                    before.min_window_speech_ratio != after.min_window_speech_ratio,
                ),
                ("carrier", before.carrier != after.carrier),
                ("weights", before.block_weights != after.block_weights),
                (
                    "median",
                    before.median_filter_frames != after.median_filter_frames,
                ),
                ("min_turn", before.min_turn_ms != after.min_turn_ms),
                (
                    "min_cluster_speech",
                    before.effective_min_cluster_speech_ms != after.effective_min_cluster_speech_ms,
                ),
            ];
            let n = changed.iter().filter(|(_, hit)| *hit).count();
            if name == "default" {
                assert_eq!(n, 0, "the default variant must change nothing");
            } else {
                assert!(n > 0, "variant {name} changed no configuration field");
            }
        }
    }
}

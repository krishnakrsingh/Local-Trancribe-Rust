# Diarization investigation — Track B

**Status:** investigation complete. **No production behaviour was changed.**
**Date:** 2026-09-29.
**Engine measured:** `vaani-diarization-heuristic 0.1.0`, `PITCH`-free spectral
heuristic, agglomerative average linkage, cosine distance. No trained speaker
embedder is installed; none is available on this machine (§ CANDIDATE
APPROACHES).

**Evidence:** two JSON reports produced by the harness this investigation built,
committed alongside this file:

| file | what it is | how to reproduce |
|---|---|---|
| `diarization-bench-2026-09-29.json` | the real 13.9-minute meeting, both activity sources, the merge sweep, 19 configuration variants, and the five scored synthetic fixtures | `cargo run -p vaani-diarization --release --features vad-probe --bin vaani-diar-bench -- --input scratch/clip.wav --external-timeline --sweep --variants --synthetic --repeats 3` |
| `diarization-bench-synthetic-2026-09-29.json` | the same five fixtures, each anchoring its own scored sweep and scored variant table. No private file involved. | `cargo run -p vaani-diarization --release --features vad-probe --bin vaani-diar-bench -- --synthetic --sweep --variants` |

**Privacy.** The harness reads `scratch/clip.wav` and never writes a sample, a
transcript, or anything derived from what was said. The only strings in the
report are case names, configuration numbers and engine ids. Span start and end
times are committed because the brief requires the segment list and because turn
timing is not content; the report says so in its own `notes` array. The
recording itself is untouched and untracked.

**What this document is not.** It is not a claim that the engine is accurate on
real audio. There is no annotated per-speaker reference for any real recording
in this repository, so **no accuracy figure for real audio appears anywhere in
this document or in either JSON file**, and the harness refuses to compute one:
a real case carries `reference: null` and a contract test fails if one ever
appears.

---

## 1. CURRENT PIPELINE

Precise, stage by stage, with `file:line`. Every statement here is from the
source, not from the crate's own documentation, and where the two disagree the
disagreement is called out.

### 1.1 Segmentation method

There is no segmentation model and no segmentation stage. A turn boundary is
produced by run-length encoding a per-frame cluster label sequence, breaking at
every inactive frame, and then applying three postprocessing rules. In order:

| step | code | what it does |
|---|---|---|
| frame labels | `engine.rs:822` `project_frames` | each active frame takes the label of the **latest** window containing it, by nearest centroid |
| median filter | `engine.rs:398` → `smooth.rs:102` | median over `median_filter_frames` (21, biased ×1 = 21) of per-frame labels; truncated neighbourhood at the edges |
| turn encoding | `engine.rs:404` → `smooth.rs:175` | run-length encode; **any inactive frame breaks a turn** |
| short-turn absorption | `engine.rs:413` → `smooth.rs:295` | a turn shorter than `min_turn_ms` (600) is merged into whichever neighbour is **longer**; the *short* turn's speaker wins |
| adjacent merge | `smooth.rs:265` | consecutive same-speaker turns touching in time merge; score is frame-count weighted |
| sanitise | `engine.rs:414` → `smooth.rs:351` | drop turns with `end <= start` |
| validate | `engine.rs:424` → `smooth.rs:364` | sorted, non-overlapping, `start < end`, every speaker `< num_speakers`, score in `[0,1]`; a violation is a hard `Err`, not a warning |

Frame geometry is deliberately identical to the ASR front end: 10 ms hop, 512-point
FFT, 400-sample window **centred** at offset 56 (`dsp.rs:280` `usable_frames`,
`dsp.rs` framing convention). Span times are absolute; `frame_time_ms(f) = f*10`.

### 1.2 VAD dependency, and how the mask is obtained

**Two sources, and which one runs is decided by the caller, not by the crate.**

* The product path: `vaani-app`'s `DiarStage::run`
  (`crates/vaani-app/src/stages/diar.rs:114`) passes `ctx.artifacts.speech` —
  the `SpeechTimeline` produced by `vaani-vad`'s `EnergyVad` — into
  `diarize_with_timeline`. When it is `Some`, `engine.rs:294-311` clones it and
  runs **no activity scan at all** (`activity_ms` is 0, measured).
* The fallback: `crate::vad` (`vad.rs`), used when no timeline is supplied.
  `vad::scan` (`vad.rs:153`) computes three per-frame quantities from one
  256-point transform: energy in dB, spectral flux over the lowest 128 bins, and
  spectral flatness over 16 **equal-width** bands of 8 bins
  (`dsp.rs:538`, `dsp.rs:640`).
  `derive_thresholds` (`vad.rs:116`) takes the 10th percentile of energy as the
  floor and the **99.9th** (`vad.rs:108`) as the loud reference — the 99.9 is a
  documented fix for a recording that is mostly silence.
  `activity_mask` (`vad.rs:133`) calls a frame speech if it clears
  `energy_hi_ratio` (0.55 of the file's own dynamic range) **or** clears
  `energy_lo_ratio` (0.18) with flux above its own 60th-percentile band, and
  never if flatness exceeds 0.15.
  `mask_to_timeline` (`vad.rs:204`) then adds 300 ms hangover either side, bridges
  300 ms gaps, and drops runs under 200 ms.

The per-frame mask is then derived identically in both paths
(`engine.rs:324-334`): a frame is active if its centre instant falls inside a
timeline segment.

### 1.3 The feature vector, dimension by dimension

There is no embedding model. `features.rs:345` builds an **83-dimension
hand-built spectral descriptor** per 10 ms frame, and `features.rs:238` reduces
one window's frames to one vector.

| # | block | dims | index | weight | how it is made | what it contributes |
|---|---|---:|---|---:|---|---|
| 0 | cepstral envelope `c1..c19` | 19 | 0–18 | 0.45 | orthonormal DCT-II of a 26-band log-mel spectrum, 80–7600 Hz (`dsp.rs:348`, `dsp.rs:366`); `c0` dropped as frame energy | vocal-tract shape; the standard compact timbre descriptor |
| 1 | delta cepstra | 19 | 19–37 | 0.05 | mean absolute value of the first difference of block 0 across the window | articulation-rate cue; mostly the sentence, not the person |
| 2 | double-delta | 19 | 38–56 | 0.02 | mean absolute value of the second difference | curvature, same argument |
| 3 | **log2 F0** | 1 | 57 | 1.00 | FFT autocorrelation over a 1024-sample periodic-Hann span, lags 40–266, parabolic peak interpolation, mean of the voiced frames (`dsp.rs:438`) | intended to be the primary speaker signal |
| 4 | voicedness | 1 | 58 | 0.22 | mean autocorrelation confidence over the window's frames | "was this window voiced" — an activity/loudness statistic |
| 5 | centroid, spread, 85 % rolloff | 3 | 59–61 | 0.11 | from the same 257-bin power spectrum (`dsp.rs:392`), log2 Hz | spectral tilt, but also which consonant |
| 6 | band ratios | 5 | 62–66 | 0.02 | energy fractions in 0–300/300–800/800–1600/1600–3200/3200–8000 Hz, floored 25 dB below the loudest band in the frame, then ln (`dsp.rs:653`, `dsp.rs:670`) | coarse stable envelope |
| 7 | LTAS | 16 | 67–82 | 0.02 | log-spaced 100 Hz–7600 Hz band energy fractions, floored, then ln (`features.rs:383-416`) | long-term average spectrum |
| — | carrier | 1 | 83 | 1.00 | a constant appended by `normalise_in_place` (`features.rs:769`) | evidence requirement: sets the absolute cosine scale |
| | **total** | **84** | | | | |

Per-dimension weight is `block_weight / sqrt(dims)` (`features.rs:517`), so a
block's share of the squared norm is `w²`.

**Three postprocessing steps sit between the frame vector and the clustering**, in
this order, and the order is load-bearing:

1. **median over frames** per dimension (`features.rs:281-308`), except the voicedness
   and pitch dimensions which are a mean;
2. **temporal smoothing** — a triangular moving average, half-width
   `feature_smooth_half_windows` = 2, i.e. 5 windows = 3.75 s, applied along the
   window timeline *before* normalisation (`engine.rs:365` → `features.rs:921`);
3. **normalisation** (`features.rs:757`), which is where the design lives:
   * every dimension except pitch is median-centred, divided by a **constant**
     `within_speaker_spread` (`features.rs:661`), multiplied by its block weight,
     and **dropped to zero** if its observed robust spread is at or below the
     `MinSpread` floor (`config.rs:476`);
   * pitch is **absolute** (`features.rs:139`): shifted by the fixed
     `PITCH_REFERENCE_LOG2 = 7.0` and multiplied by `ABSOLUTE_PITCH_SCALE = 1.5`,
     with no per-file centring and no division by any spread. It therefore
     bypasses the `MinSpread` gate entirely;
   * the carrier is set to `weights.carrier` = 1.0 and the row is L2-normalised;
   * the whole operation is skipped for fewer than `MIN_ZSCORE_ROWS` = 4 rows.

### 1.4 Clustering algorithm

`cluster::cluster` (`cluster.rs:317`), four steps:

1. **Agglomerative, average linkage, cosine distance** (`cluster.rs:119`), run
   **all the way down to one cluster** so the whole dendrogram exists. Exact
   average linkage needs the full `N×N` matrix (`cluster.rs:214`), which is why
   windows are thinned by a deterministic stride to `max_cluster_windows` = 2000.
   Ties break to the lowest index pair; the output is bit-deterministic
   (`cluster.rs:144-165`).
2. **Stopping rule** `choose_count` (`cluster.rs:394`):
   * `Threshold { merge_distance, max_speakers }` — **the default** — scans `k`
     upwards and keeps the largest `k` whose producing merge costs at most
     `merge_distance`; if nothing fits, it returns the cap.
   * `Auto { max_speakers, min_gap_ratio }` — cuts before the largest
     *relative* jump in the cost sequence.
   * `Fixed(k)` — merge to `k`.
3. **Min-speech guard** `absorb_small_clusters` (`cluster.rs:470`) — a cluster with
   less than `min_cluster_speech_ms` of speech is merged into the nearest
   surviving cluster, never dropped.
4. **Lloyd refinement**, `refine_iters` = 3 (`cluster.rs:363`): reassign every row
   to its nearest centroid, rejecting a step that would empty a cluster.

Then `relabel_contiguous` renumbers to `0..k` in order of first appearance.

### 1.5 Threshold selection, and who tunes it

`SpeakerCount::default_policy()` (`config.rs:174`) returns
`Threshold { merge_distance: 1.05, max_speakers: 8 }`. `SplitEasily` scales that
by 0.85 (`config.rs:65-70`), so **the distance the engine actually cuts at is
0.8925** (`config.rs:759`), and it is clamped to `[0.01, 1.99]`.

A4 tuned it by hand from a `geometry` sweep on synthetic fixtures, and the
provenance is written down (`config.rs:105-140`): the binding band was
within-speaker p90 0.86 against cross-speaker p10 1.18. **Nobody has ever
re-tuned it against real audio, because there is no reference for real audio.**
The same is true of `within_speaker_spread` (`features.rs:650-668`), which the
code states was measured on the `one-speaker-long` synthetic fixture and
deliberately *not* on the real clip "because they include the between-speaker
term this constant exists to exclude", and of `MinSpread` (`config.rs:477-537`),
whose table lists both synthetic medians and a real-clip column, with the floors
set at roughly a third of the smaller synthetic median.

### 1.6 Minimum speaker and segment duration

| quantity | value | code |
|---|---|---|
| minimum cluster speech | 2000 ms, ×0.5 bias = **1000 ms** | `config.rs:579`, `config.rs:773` |
| minimum turn | **600 ms** | `config.rs:584` |
| minimum activity run | 200 ms | `config.rs:629` |
| minimum window speech fraction | **0.9** | `config.rs:567` |
| minimum analysis window | 150 frames = **1.5 s**, hop 75 = 0.75 s | `config.rs:547-550` |
| median filter | 21 frames = 210 ms | `config.rs:582` |

### 1.7 Merge and split rules

There is no explicit split rule. The only mechanism that increases the speaker
count is the `Threshold`/`Auto` stopping rule keeping a finer cut. The only
mechanism that decreases it is the min-speech guard. Both are listed above.

### 1.8 Overlap handling

**None, and none is possible.** `smooth.rs:28-36` states the intent: spans are
non-overlapping and tight, and `validate_spans` enforces it. `SpeakerSpan` in the
core type permits overlap, but the engine never produces any. The consequence
is that a genuine cross-talk region is labelled as whichever of the two nearest
turns covers it, and the `score` field on the span
(`smooth.rs:43`, computed as `agreement` at `smooth.rs:227`) is the only signal
that the boundary was uncertain. **Measured on the real clip, that signal is
worthless: span score p50 = 1.000, p10 = 1.000, mean = 0.9995 over 310 spans.**
The engine reports total confidence on every turn of a real meeting, which means
it is not the least bit uncertain about any of them.

### 1.9 Speaker re-identification

None. There is no cross-chunk state, no second pass, no consistency check
between chunks, and no way to merge two labels the user has decided are one
person. `Transcript::add_speaker` preserving a user-set name is the only
continuity mechanism, and it is in `attribution.rs:186`.

### 1.10 Timestamp alignment

No alignment stage. The diarizer emits absolute-time spans; the transcript's
words are attributed to them by a duration-weighted overlap vote
(`attribution.rs:78` `overlap_vote`, `attribution.rs:186` `attribute_speakers`),
and the grouping policy then uses those spans to decide where a turn falls
(`crates/vaani-app/src/stages/diar.rs:139`). Attribution is a separate call and
never touches audio.

### 1.11 Every postprocessing heuristic, in order

1. hangover ±300 ms, gap bridge 300 ms, drop runs < 200 ms (`vad.rs:204`)
2. window kept only if ≥ 90 % of its frames are active (`engine.rs:501-573`)
3. recording-wide pitch fallback for a window with no voiced frame
   (`engine.rs:599-608`) — the **mean** of every voiced pitch in the recording
4. triangular temporal smoothing over 5 windows (`engine.rs:365`)
5. median-centring, constant-scale division, `MinSpread` gate, block weights,
   carrier, L2 normalisation (`features.rs:757`)
6. stride thinning to 2000 rows before clustering (`engine.rs:371-374`)
7. min-speech guard (`cluster.rs:470`)
8. Lloyd refinement, 3 iterations (`cluster.rs:363`)
9. "latest window wins" frame labelling (`engine.rs:822`)
10. median filter, 21 frames (`engine.rs:398`)
11. adjacent same-speaker merge (`smooth.rs:265`)
12. short-turn absorption into the **longer** neighbour (`smooth.rs:295`)
13. `score` = agreement of the *unsmoothed* labels (`smooth.rs:227`)

### 1.12 Dead or unexercised code

| item | state |
|---|---|
| `embed.rs` — the entire neural-embedder adapter | **written, compiles, never executed a forward pass.** No weights exist (`artifacts/` holds SraVaani's ASR graphs only). `NeuralEmbedder::open` and `::embed` have never been called with a real file. |
| `embed::runtime_available` | never called from anywhere in the workspace |
| `cluster::complete_link_separation` | used by `eval`, not by the engine |
| `SpeakerCount::Auto` | reachable only through `DiarizationConfig::with_estimated_speaker_count`; not the default and not exercised by the product path |
| `smooth::merge_adjacent_with_gap` with `max_gap_ms > 0` | the engine always calls `merge_adjacent`, i.e. gap 0; the gap-bridging form has no production caller |
| `dsp::frame_log_mel` | used by `frame_cepstra`; the public wrapper is otherwise unused |
| `eval::measure_quality` | not re-exported at the crate root and not used by the engine; only by tests and examples |

---

## 2. WHAT THE HARNESS MEASURES

`crates/vaani-diarization/src/bench.rs` and
`crates/vaani-diarization/src/bin/vaani-diar-bench.rs`. One command, one JSON
document, `docs/performance/diarization-bench-2026-09-29.json`.

It emits, per case: `speaker_count`, `span_count`, **`speaker_change_count`**
(adjacent spans naming different speakers — the number the transcript's grouping
policy leans on), `talk_share`, `turn_durations` and `turn_scores` as
percentiles, `wall_ms`, `rtf`, and the per-stage breakdown
(`activity_ms` / `features_ms` / `cluster_ms` / `smooth_ms`) with the frame and
window counts. It also emits the configuration *resolved to numbers* (including
the effective merge distance, the floors and the weights), the activity report
(mask source, frame ratio, coverage, timeline segment durations, window keep
ratio), and the full span list.

Three extra blocks exist because the investigation needed them and they are the
reason several of § 3's findings are findings rather than hunches:

* **`geometry.pairwise`** — every window-to-window cosine distance (304,590 pairs
  on the real clip), as percentiles.
* **`geometry.merge`** — the average-linkage cost sequence: the cost at each
  `k = 1..12`, the five largest *relative* jumps, the count the configured
  threshold selects, and 64 sampled costs across the whole 780-merge sequence.
* **`geometry.pitch`** and **`geometry.blocks`** — the absolute pitch dimension as
  a semitone occupancy histogram, and a per-block additive decomposition of the
  distance that also reports, per block, the **measured** spread, the constant it
  is divided by, and their ratio.

A decomposition identity test pins the last of these: the shares must sum to the
mean pairwise distance (`bench.rs` `block_shares_sum_to_the_mean_pairwise_distance`).
It caught a real bug in the harness itself — the shares were once divided by the
same-speaker pair count while summed over all pairs, which reported a
`spectral_shape` share of **1.14** on a fixture whose mean distance was 0.9.

### 2.1 A privacy note on the committed reports

The reports contain the full span list for every primary case: speaker index,
start ms, end ms, duration, score. That is turn timing and nothing else, it is
required for the report to be checkable, and it is the only per-case data that
is recording-specific. Nothing derived from speech content exists in the
harness, the binary, or the reports, and the report says so in its own `notes`.

---

## 3. OBSERVED FAILURE MODES

All measured on the 13.9-minute / 834,197 ms recording, 13,347,157 samples at
16 kHz, 83,419 analysis frames, 781 windows (internal) / 629 windows (external
VAD timeline). **No accuracy is claimed anywhere.**

### 3.1 The speaker count is 3 or 2 depending on which VAD runs, and that is the reported 2

| activity source | speakers | spans | **speaker changes** | coverage | talk share | windows | window keep ratio |
|---|---:|---:|---:|---:|---|---:|---:|
| internal `vad.rs` | **3** | 310 | **230** | 0.881 | 0.286 / 0.585 / 0.129 | 781 | 0.702 |
| `vaani-vad` `EnergyVad` (the product path) | **2** | 300 | **141** | 0.895 | 0.232 / 0.768 | 629 | 0.565 |

Same audio, same configuration, same engine. **The count moves, and so does the
number the transcript's structure depends on: 230 speaker changes becomes 141, a
39 % swing.** The product path is the second row, and 2 speakers is what
`docs/JOURNEY.md` records from the end-to-end run.

The mechanism is visible in the activity report. `EnergyVad` produces a
**2.3× more fragmented** mask: 240 timeline segments with a median duration of
2,090 ms, against 106 segments with a median of 4,910 ms. A 1.5 s analysis
window sitting on a 2,090 ms median segment cannot be 90 % speech, so
`min_window_speech_ratio = 0.9` rejects far more of them: **window keep ratio
0.565 against 0.702, i.e. 44 % of candidate windows versus 30 %.** The window
population the clustering sees is different, so the distance distribution is
different, so the fixed threshold lands somewhere else.

### 3.2 The absolute pitch dimension — the one the whole design rests on — is inert on real audio

| measurement | internal | external (product) | `synthetic-2spk-12s-turns` |
|---|---:|---:|---:|
| share of the mean pairwise distance | 0.00047 | **0.00028** | 0.0035 |
| configured block weight | 1.00 | 1.00 | 1.00 |
| pitch distribution p10 → p90 | 107 → 165 Hz | 108 → 175 Hz | 125 → 198 Hz |
| occupied 1-semitone bins | 12 | 12 | 2 |
| largest empty gap inside the occupied range | **0 semitones** | **0 semitones** | **8 semitones** |
| measured per-window spread (log2 Hz) | 0.248 | 0.239 | 0.526 |

The two synthetic voices are 8.9 semitones apart and the histogram has a clean
8-semitone hole between them. **The real meeting's per-window median F0 is a
continuum** — 12 occupied bins with no internal gap at all, spanning a fifth of
an octave from p10 to p90. There is no pitch structure to find.

The consequence is measurable to the span: zeroing the pitch weight
(`pitch-weight-0`) leaves the product-path result **unchanged** — 2 speakers,
300 spans, 141 changes, coverage 0.895, the same as the default — and leaves all
five scored synthetic fixtures bit-identical too. The heaviest block in the
vector, the one `PITCH_REFERENCE_LOG2` and `ABSOLUTE_PITCH_SCALE` exist for,
changes nothing. On the internal-activity path the same variant gives 3 speakers
and 309 spans against the default's 3 and 310: a difference of one span in 310.

### 3.3 The distance is not made of speaker information

Additive per-block decomposition of the pairwise distance, product path:

| block | dims | configured weight | **share of the distance** | measured spread | the constant it is divided by | ratio |
|---|---:|---:|---:|---:|---:|---:|
| voicedness | 1 | 0.22 | **0.395** | 0.198 | 0.0015 | **132×** |
| spectral_shape | 3 | 0.11 | **0.246** | 0.582 | 0.0044 | **132×** |
| band_ratios | 5 | **0.02** | **0.242** | 0.541 | 0.0011 | **492×** |
| cepstra | 19 | 0.45 | 0.081 | 1.176 | 0.0826 | 14× |
| ltas | 16 | **0.02** | 0.026 | 0.571 | 0.0032 | 178× |
| delta + double delta | 38 | 0.05 / 0.02 | 0.0004 | | | |
| **pitch** | 1 | **1.00** | **0.00028** | 0.239 | — (absolute) | — |
| carrier | 1 | 1.00 | 0.0011 | — | — | — |

The `ratio` column is the mechanism. `normalise_in_place` divides every
non-absolute block by a **constant** `within_speaker_spread` that was measured on
synthetic voices. On this recording the measured spread of `band_ratios` is
**492×** that constant, so after the division the block arrives at roughly 492
unit variances, and the L2 normalisation at the end hands it a quarter of the
vector. The same happens to `voicedness` at 132×. Meanwhile the *absolute* pitch
block is pinned by a fixed physical scale and cannot be amplified at all, so a
dimension with 0.24 of internal scatter contributes 0.03 % of the distance.

For contrast, on the one-speaker synthetic fixture every ratio is 0.0–1.0 — the
constants describe *those* voices — and the mean pairwise distance collapses to
0.158 instead of 0.99. **The normalisation constants are calibrated for a
generator whose per-window spectral scatter is two to three orders of magnitude
smaller than a real meeting's.**

And what the distance is actually made of is not identity: `voicedness` is "what
fraction of this window had a confident autocorrelation peak", which is a
loudness/SNR statistic, and `band_ratios` is five fixed-frequency energy
fractions, which move when a harmonic series slides across fixed edges.

The scored synthetic decomposition is the clean version of the same story. On
`synthetic-2spk-12s-turns`, where a reference exists, the **speaker signal**
(`different − same` share) is:

| block | speaker signal |
|---|---:|
| spectral_shape | 0.599 |
| band_ratios | 0.467 |
| cepstra | 0.346 |
| ltas | 0.176 |
| voicedness | 0.066 |
| **pitch** | **0.0013** |

The engine works on the synthetic fixtures **because of the spectral blocks**,
and the configuration sets those blocks to almost nothing (0.02 for `band_ratios`
and `ltas`, 0.11 for `spectral_shape`) precisely because `config.rs:414-443`
down-weighted them after measuring that they produced 0.89 of a single speaker's
own scatter on an intonation fixture. That trade was made on the wrong
measurement, and § 5 says what to do about it.

### 3.4 The threshold is not sitting in a gap

Average-linkage cost sequence on the real clip, internal / external:

```
k            1       2       3       4       5       6       7       8
internal   0.970   0.901   0.856   0.797   0.698   0.692   0.651   0.622
external   1.070   0.847   0.782   0.772   0.700   0.681   0.623   0.596
```

There is no discontinuity. The largest *relative* jump on the internal path is
1.35×, and it is at `after_k = 780` — the very last merge, at a cost of 0.0025,
where a 0.0025 → 0.0033 step is a 1.35× ratio and means nothing. The largest
*meaningful* jump on the external path is 1.26× at `after_k = 2`, i.e. the 2 → 1
merge. The shipped 0.8925 sits below cost@2 (0.847) and below cost@1 (1.070) on
the external path, so it cuts at `k = 2`; on the internal path it sits between
cost@3 (0.856) and cost@2 (0.901) and cuts at `k = 3`. **A fixed threshold on a
smooth sequence cuts wherever the sequence happens to be.**

### 3.5 The threshold's own answer is a plateau, which is the one piece of good news

The scored sweep on `synthetic-2spk-12s-turns` (true = 2 speakers, populations
separated by margin +0.788):

| effective merge distance | speakers | purity | recall | complete |
|---|---:|---:|---:|---:|
| 0.17 – 0.595 | 4 | 0.939 | 0.625 | 0.939 |
| 0.595 | 3 | 0.958 | 0.667 | 0.957 |
| **0.7225 – 1.105** | **2** | **0.958** | **1.000** | **0.957** |
| above 1.105 | — | — | — | — |

The correct answer, with full recall, holds across a **1.53× range** and the
shipped default sits comfortably inside it. On the *real* clip the same shipped
value sits in a 1.0 → 1.30 plateau that also reports 2 (with 3 at 0.95 and 5 at
0.90). **Where the geometry is clean, the shipped threshold is correct and
comfortably placed. The "2 speakers" result is not a mis-tuned threshold; it is
the absence of geometry.**

### 3.6 The internal activity detector discards speech, badly, and it is not the product path

`active_frames_per_true_speaker`, counted from the generator's own per-frame
reference:

| fixture | true frames | active frames | ratio | quietest speaker |
|---|---:|---:|---:|---:|
| `synthetic-2spk-12s-turns` | 14,400 | 10,339 | 0.718 | |
| `synthetic-3spk-3s-turns` | 12,000 | 7,669 | **0.639** | |
| `synthetic-3spk-similar-pitch-12s-turns` | 14,400 | 7,574 | 0.526 | |
| `synthetic-3spk-one-20db-down` | 14,400 | 9,727 | 0.676 | **127 / 4,800 = 2.6 %** |
| same, on the oracle timeline | 14,400 | 14,399 | 1.000 | **4,799 / 4,800** |
| `synthetic-1spk-2min` | 12,000 | 8,421 | 0.702 | |

Two things here. First, the existing `#[ignore]`d test
`a_speaker_twenty_decibels_quieter_is_currently_discarded` is confirmed and
sharpened: a speaker 20 dB below the others keeps **127 of 4,800 frames and zero
of 126 windows**. Second, and not previously recorded: `synthetic-3spk-3s-turns`
is **100 % labelled, back-to-back synthetic speech with no silence in it at all**
— every turn abuts the next, levels differ only by the generator's 0.55–1.0 gain
pattern, a 5 dB range — and the detector calls **64 %** of it speech. A third of
clean speech never reaches the feature stage.

**But the product does not use this detector.** `DiarStage` always passes
`EnergyVad`'s timeline. So the level-discard defect has no product-path impact,
and `EnergyVad`'s own frame-level speech fraction on this recording (0.52, per
`vaani-vad`'s own `VadStats` documentation) is *lower* than the internal
detector's 0.865, not higher.

### 3.7 The oracle experiment: a perfect VAD does not fix the under-count

`synthetic-3spk-one-20db-down` has three speakers, one 20 dB down, scored
against the generator. Run twice: once on the internal detector, once on an
**oracle** timeline built from the generator's own intervals with the same 300 ms
hangover and the same 300 ms gap bridge.

| | speakers reported | scored windows per true speaker | purity | recall | complete | margin |
|---|---:|---|---:|---:|---:|---:|
| internal detector | 2 | 56 / 56 / **0** | 0.973 | 0.750 | 0.973 | +0.793 |
| **oracle timeline** | **2** | 56 / 56 / **57** | 0.959 | 0.833 | 0.645 | +0.667 |

Restoring the quiet speaker's evidence — from **zero windows to 57, a third of
the corpus** — does **not** change the reported count, and it *lowers* the strict
complete accuracy from 0.973 to 0.645. **The activity gate is the cause of the
discard and not the cause of the under-count.** It is a real defect with no
product-path consequence, and it is emphatically not the explanation for the
2-speaker result.

### 3.8 No configuration change is the fix

19 single-knob variants, scored against the generator on all five fixtures.
Selected rows (spk / purity / recall / complete):

| variant | 2spk (T=2) | 3spk-3s (T=3) | 3spk-sim (T=3) | 3spk-20 dB (T=3) | 1spk (T=1) |
|---|---|---|---|---|---|
| default | 2 / 0.96 / 1.00 / 0.96 | 2 / 0.86 / 0.67 / 0.71 | 2 / 0.94 / 0.83 / 0.62 | 2 / 0.97 / 0.75 / 0.97 | 1 / 1.00 / 1.00 / 1.00 |
| `window-3.0s` | 2 / 0.92 / 1.00 / 0.92 | **3** / 1.00 / 0.33 / 1.00 | 2 / 0.90 / 0.67 / 0.65 | 2 / 0.94 / 0.75 / 0.94 | 1 / 1.00 / 1.00 / 1.00 |
| `window-0.75s` | 2 / 0.99 / 0.75 / 0.99 | 2 / 0.94 / 0.67 / 0.70 | 2 / 0.99 / 0.67 / 0.99 | 2 / 1.00 / 0.50 / 1.00 | **2** / 0.53 / 1.00 / 0.53 |
| `feature-smooth-0` | 2 / 1.00 / 0.50 / 1.00 | **3** / 1.00 / 0.33 / 1.00 | 2 / 1.00 / 0.50 / 1.00 | 2 / 1.00 / 0.50 / 1.00 | 1 / 1.00 / 1.00 / 1.00 |
| `feature-smooth-6` | 2 / 0.90 / 1.00 / 0.90 | 2 / 0.73 / 1.00 / 0.55 | **3** / 0.81 / 0.67 / 0.56 | 2 / 0.91 / 1.00 / 0.91 | 1 / 1.00 / 1.00 / 1.00 |
| `min-window-speech-0.5` | 2 / 0.98 / 0.75 / 0.98 | 2 / 0.83 / 0.67 / 0.72 | 2 / 0.97 / 0.83 / 0.66 | 2 / 1.00 / 0.50 / 1.00 | 1 / 1.00 / 1.00 / 1.00 |
| `min-window-speech-0.0` | 2 / 0.96 / 1.00 / 0.96 | 2 / 0.82 / 0.83 / 0.65 | 2 / 0.94 / 0.83 / 0.65 | **3** / 1.00 / 0.33 / 0.97 | 1 / 1.00 / 1.00 / 1.00 |
| `merge-readily` | 2 / 0.96 / 1.00 / 0.96 | **1** / 1.00 / 1.00 / 0.41 | 2 / 0.94 / 0.83 / 0.62 | 2 / 0.97 / 0.75 / 0.97 | 1 / 1.00 / 1.00 / 1.00 |
| `carrier-4.0` | 2 / 0.96 / 1.00 / 0.96 | **1** / 1.00 / 1.00 / 0.41 | 2 / 0.94 / 0.83 / 0.62 | 2 / 0.97 / 0.75 / 0.97 | 1 / 1.00 / 1.00 / 1.00 |
| `carrier-0.25` | 2 / 0.96 / 1.00 / 0.96 | 2 / 0.86 / 0.67 / 0.71 | 2 / 0.94 / 0.83 / 0.62 | 2 / 0.97 / 0.75 / 0.97 | **2** / 0.53 / 1.00 / 0.53 |
| `spectral-weights-x4` | 2 / 0.96 / 1.00 / 0.96 | 2 / 0.86 / 0.67 / 0.71 | 2 / 0.94 / 0.83 / 0.62 | 2 / 0.97 / 0.75 / 0.97 | **2** / 0.53 / 1.00 / 0.53 |
| `auto-gap` | **4** / 0.94 / 0.63 / 0.94 | 2 / 0.86 / 0.67 / 0.71 | **5** / 0.94 / 0.33 / 0.94 | 2 / 0.97 / 0.75 / 0.97 | **2** / 0.54 / 1.00 / 0.54 |
| `pitch-weight-0` | identical to default on all five | | | | |
| `voicedness-weight-0` | identical to default on all five | | | | |
| `cepstra-weight-0` | identical, except 3spk-3s → 1 spk | | | | |
| `median-filter-1`, `median-filter-61`, `min-turn-300ms`, `min-cluster-speech-10s` | identical to default on all five | | | | |

Three things to read out of that table.

* **Every knob that reaches the right count on one fixture breaks another.**
  `window-3.0s` and `feature-smooth-0` give 3 on `3spk-3s` and shatter nothing;
  `feature-smooth-6` gives 3 on `3spk-sim` and costs 0.16 of complete accuracy on
  `3spk-3s`; `min-window-speech-0.0` gives 3 on `3spk-20 dB` and nothing
  anywhere else. `merge-readily` and `carrier-4.0` "reach" recall 1.00 on
  `3spk-3s` by reporting **one** speaker, which is the vacuous case — `recall` is
  "fraction of generator speakers present in any cluster", so one cluster makes it
  1.00 by definition. The strict `complete_accuracy` column is the one that
  refuses to be flattered: 0.41.
* **The three largest contributors to the distance are the ones whose weights do
  nothing.** On the product path `pitch-weight-0` and `spectral-weights-x4` leave
  the real clip at exactly 2 speakers / 300 spans / 141 changes, and
  `voicedness-weight-0` and `cepstra-weight-0` at 299 / 139 and 301 / 143 — a
  swing of one or two spans in 300. What moves the answer is the **global scale**
  (`carrier` 0.25 → 3 speakers, 4.0 → 1), the **window length**, the
  **smoothing**, and the **threshold**. The dials that change *what the distance is
  made of* are inert; the dials that change its *scale* are decisive. That is the
  signature of a distance space with no speaker structure in it: you can only
  change the answer by changing the ruler.
* **`SpeakerCount::Auto` degenerates to the cap on real audio.** On the
  internal-activity path it returns **8** — the hard cap — because its largest
  relative gap is the meaningless 0.0025 → 0.0033 step at the bottom of the
  dendrogram. The rule compares unbounded ratios on a sequence whose absolute
  scale reaches 0.002, so it is always won by numerical dust. (On the external
  path it happens to land on 2, which is luck rather than merit: the same rule
  returns 4 for two speakers and 5 for three on the fixtures in the table above,
  and shatters the one-speaker fixture into 2 at purity 0.54.) This is not the
  `#[ignore]`d over-split the existing test records; it is a different and more
  fundamental failure, and the one-speaker shattering is recorded nowhere in the
  repository.

### 3.9 Smaller things, measured

* A span ends **3 ms** past the end of the recording (max `end_ms` 834,200 vs
  duration 834,197). One frame's worth, from `smooth.rs:209` adding `frame_ms` to
  the last frame's centre. The same class as the 23 ms ASR overshoot already fixed
  by `Transcript::push_chunk_within`.
* `turn_scores` p50 = 1.000 and p10 = 1.000 across all 310 real spans (mean
  0.9995), and p50 = 1.000 across the external path's 300 (mean 0.9978). The
  engine's only uncertainty signal is saturated and therefore useless.
* `engine_accounted_bytes` is 2.7–3.6 MB against a 50.9 MB `AudioBuffer` the
  caller already holds. The flat-memory claim in `engine.rs:33-44` holds.
* Diarization is not the bottleneck and never was: **RTF 0.0020–0.0022**,
  1.68 s for 834 s of audio on the product path and 1.82 s on the internal one,
  of which `features_ms` is 96 % and 88 % respectively.

---

## 4. LIKELY CAUSES

Ranked by how much of the observed error each accounts for, with the evidence
that ranks it.

### Cause 1 — the normalisation constants, not the features, decide the metric (dominant)

**Claim.** `within_speaker_spread` is a per-block constant measured on synthetic
voices whose per-window spectral scatter is 100–500× smaller than a real
meeting's. Dividing by it amplifies the real recording's spectral blocks by two
orders of magnitude, the L2 normalisation then hands them the vector, and the
blocks that were deliberately down-weighted end up carrying two thirds of the
distance. The absolute pitch block, the only one pinned by a physical constant,
is squeezed to 0.03 %.

**Evidence.** The `ratio` column of § 3.3, which is measured and not inferred:
`band_ratios` 492×, `voicedness` 132×, `spectral_shape` 132×, `ltas` 178×,
`cepstra` 14× on the product path; every ratio between 0.0 and 1.0 on the
one-speaker fixture, whose mean distance is then 0.158 rather than 0.99. The
consequence is the `pitch-weight-0` and `spectral-weights-x4` variants returning
the real clip's exact span count.

**Falsifiable prediction.** Dividing each block by its *measured* spread instead
of the constant should put the shares back near their configured `w²` values
(0.74 pitch, 0.20 cepstra, 0.048 voicedness, 0.0004 band_ratios) and should
change the real clip's speaker count. It should also make the merge-cost sequence
on a one-speaker recording look like a one-speaker recording's, instead of like
0.99 mean distance.

### Cause 2 — the feature vector has no speaker-discriminative dimension on real audio (dominant, and independent of Cause 1)

**Claim.** Whatever the scaling, a cosine distance between L2-normalised rows
contributes `(a_j − b_j)²/2` per dimension. A dimension that barely differs
between windows contributes nothing, however heavily it is weighted. On this
recording, median per-window F0 spans 107–175 Hz with **no internal gap in the
semitone histogram**, and `pitch`'s whole share of the distance is 0.00028.
Removing it changes 1 span out of 300.

**Evidence.** § 3.2 in full. The control is the synthetic fixture, where the
same code, the same weights and the same threshold produce 8.9 semitones of
separation, two occupied bins and an 8-semitone hole, and a scored sweep whose
correct answer is right.

**This is the real answer to "why 2 speakers", and it is not a bug in the
clustering, the threshold or the VAD.** It is that the descriptor does not
contain speaker identity on real audio, so no amount of cluster-count tuning can
recover it. Note also that on the synthetic fixtures the *speaker signal* lives
in `spectral_shape` (0.599) and `band_ratios` (0.467) and not in pitch (0.0013) —
so even where the geometry is clean, the engine is doing the work with the blocks
the configuration treats as supporting evidence.

### Cause 3 — the VAD choice changes the window population, and the threshold amplifies that into a different answer

**Claim.** `min_window_speech_ratio = 0.9` interacts with VAD fragmentation. A
2,090 ms median segment cannot support a 90 %-speech 1.5 s window, so 44 % of
candidate windows are discarded instead of 30 %, the window population changes,
and the fixed threshold cuts the resulting (smooth) cost sequence at a different
place.

**Evidence.** § 3.1: 3 speakers / 230 changes versus 2 / 141, from one input
and one configuration; keep ratio 0.702 versus 0.565; and the whole 0.9 → 0.5 →
0.0 dial moves the real clip's count 3 → 2 → 4 on the internal path. § 3.4 shows
the sequence has no gap for a threshold to be displaced *from*. § 3.8 shows
`auto-gap` returning the cap.

**Ranked third because it is not the root cause.** It decides *which* number you
get, not *whether* the number is right. Fixing it would turn 2 into 3, and there
is no evidence that 3 is the right answer — the geometry says the descriptor
cannot see a third speaker on this audio at all.

### Cause 4 — the threshold is a constant on a continuum (contributory, not causal)

**Claim.** Any single global cosine threshold applied to a smooth merge-cost
sequence lands wherever the sequence is. On this recording the largest
meaningful relative jump is 1.26× and the largest overall is 1.35× at a cost of
0.0025.

**Evidence.** § 3.4, and the counter-evidence that matters: § 3.5 shows the
shipped threshold is correct and inside a 1.53× plateau on the one fixture where
a reference exists. **The threshold is not mis-tuned. It is a scalar where the
problem is a shape.** A better stopping rule cannot invent structure that the
features do not contain, which is why `auto-gap` returns 8 and 5.

### Ruled out

* **Clustering strategy.** Not the limiter. The agglomeration itself separates
  the synthetic fixtures at purity 0.96 with a +0.79 population margin, and the
  real clip's cost sequence is monotone and well-formed (`cluster.rs:29-30`
  asserts monotonicity). AHC is a reasonable choice for this input. What fails is
  the *stopping rule applied to a feature space with no structure*, which is
  Causes 3 and 4, not the linkage.
* **The min-speech guard, the median filter, the short-turn absorption and the
  min-turn rule.** All four leave speaker *identity* identical to the default on
  all five scored fixtures — same cluster, same purity, same recall, same complete
  accuracy. They change turn structure and nothing else, which is what they are
  for: `min-turn-300ms` takes the real clip from 300 to 382 spans while the
  speaker count does not move, and `median-filter-61` takes it from 141 to 145
  changes.
* **The internal activity detector's level discard, as an explanation of the
  2-speaker result.** Measured and excluded by the oracle experiment, § 3.7.
  It is a real defect with no product-path impact.

---

## 5. CANDIDATE APPROACHES

### 5.1 `sherpa-onnx`, and what a real embedder would change

**The offline constraint, stated as a finding.** This machine has no network.

* `sherpa-onnx` is **not in the local cargo registry cache**. The cached crate
  set contains `ort 2.0.0-rc.13`, `ort-sys`, `ndarray 0.16/0.17`, `rustfft`,
  `rubato`, `webrtc-vad`, `half`, `tracing` and about 400 others; no `sherpa-onnx`,
  no `ort-sys` variant we do not already have, no Python, no `pyannote`.
* **No diarization or speaker-embedding model exists anywhere on this machine.**
  `artifacts/` contains `sravaani/` only: `encoder.fp16.onnx` (886 MB),
  `encoder.onnx` (1.77 GB), `decoder_joint.onnx` (43 MB), `tokenizer.model`,
  two filterbank matrices and four JSON configs. `runtime/` contains
  `onnxruntime.dll` (16.9 MB) plus the CUDA providers. `SraVaani-1.0/` is the ASR
  reference implementation. `models/` is empty.
* Therefore I **could not add `sherpa-onnx`, could not download it, could not
  download a segmentation or embedding model, and did not benchmark any of
  them.** There is no number in this document for a neural diarizer, and there
  will not be one until there is network access. **This section is a
  specification, not a measurement, and is labelled as such.**

**What `sherpa-onnx` offers.** It is a Rust binding to k2-fsa's ONNX inference
layer, wrapping speaker-embedding and segmentation models. For VaaniDesk it
would be two things: an x-vector/ECAPA-style embedding extractor over a fixed
window, and a pyannote-style segmentation model that outputs per-frame per-class
activations including overlap. The models `docs/decisions/ADR-006` names —
pyannote `segmentation-3.0` int8 and `nemo_en_titanet_small` — are distributed on
GitHub Releases, not Hugging Face, specifically so that no token is required,
which fits this product's zero-token mandate.

**What it would change, and what it would cost.**

* *What it would change:* Cause 2 directly. An ECAPA-TDNN embedding is trained to
  maximise speaker verification accuracy, and its EER on standard corpora is
  single-digit percent; a heuristic 83-dimension spectral descriptor with the
  measured properties in § 3.3 has no published EER because nobody has one. It
  also brings *vocal-tract* information this vector does not have at all: the
  cepstral envelope here is a within-file median-normalised 19-coefficient DCT of
  a 26-band log-mel, which is a description of the recording as much as the
  person.
* *Licences.* This is the part that needs counsel, and it is exactly parallel to
  the SraVaani question already open in `docs/JOURNEY.md` ("The SraVaani licence
  needs written confirmation before release"). ECAPA-TDNN as published by
  SpeechBrain is Apache-2.0. `nemo_en_titanet_small` is NVIDIA's, under the
  NVIDIA Source Code License — **which is not MIT and not Apache-2.0, and
  §2.5-style redistribution questions are exactly what A12 hit with CUDA.** The
  pyannote segmentation models carry their own CC-BY-4.0 weights licence and a
  model card with acceptable-use conditions. I have **not** verified any of these
  from the local machine; they come from knowledge and must be confirmed against
  the actual model cards before any bundling decision. In particular, shipping a
  diarizer would put a *second* unconfirmed weight licence into a distribution
  that is already waiting on one.
* *Integration cost into this native app.* Low on the runtime side, high on the
  contract side, and the contract side is the problem.
  - **The plumbing already exists and is unusually good for this.** `ort 2.0.0-rc.13`
    and `vaani-inference` are already the session layer, `runtime/onnxruntime.dll`
    is already shipped, `DiarizationEngine` already takes an `AudioBuffer` and
    never ASR output, and `embed.rs` is a **written, compiling, never-executed**
    `ort` adapter for exactly this: `EmbedderSpec` (`embed.rs:62`) with input
    name, rank, window samples, hop, embedding dim and normalisation; `open`
    (`embed.rs:236`) which checks the graph's real input and output names before
    running; `embed` (`embed.rs:304`) which refuses a mis-sized window, returns
    `None` for a silent window rather than a zero vector, L2-normalises, and
    rejects a non-finite output. `HeuristicDiarizer::load` already probes for it
    (`engine.rs:881-921`), and falls back to the heuristic with a `tracing::warn!`
    if the model is present but unloadable. **The integration seam was built and
    has never been exercised, which is the single best fact in this section.**
  - **The clustering is not reusable as-is.** `features::Layout` is 83 dimensions
    with named blocks and `normalise_in_place` is written around
    `within_speaker_spread` per block. An ECAPA embedding is 192 dimensions, one
    block, already L2-normalised, and must *not* be per-file z-scored (the same
    argument `features.rs:108-142` makes for absolute pitch, and for the same
    reason). Feeding one through the current normalisation would delete the
    speaker signal in exactly the way z-scoring deletes the pitch mean. A neural
    path needs its own normalisation branch, not a reuse.
  - **The `vad-probe` feature pattern is the right precedent** and is the
    template for how this should land: a feature, off by default, so the shipped
    product is unchanged and the alternative is opt-in. The `embed` path should
    have had one from the start and does not — that is a defect in *my* crate and
    is reported in § INTEGRATION NOTES.
  - **VRAM.** ADR-006 already computes the budget: 887 MB ASR encoder plus
    ~400 MiB activation headroom against 3.3 GB, with the segmentation session
    required to be dropped before the embedding session is created and both
    before ASR resumes. `MachineCapabilities::vram_fits` exists to answer that
    before load. Nothing in this investigation measures it, and a 14-minute
    meeting at RTF 0.002 would be a rounding error next to ASR's 0.04.

### 5.2 Alternative local embedding models, and what each would need

Nothing downloadable is present, so this is a shopping list with the
prerequisites named, not a comparison.

| model | shape | would need | note |
|---|---|---|---|
| ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`, or NVIDIA's reimplementation) | 192-d, 3.2 s @ 16 kHz, expects **80-band log-mel, 25 ms / 10 ms** | an ONNX export whose front end matches, or a graph that takes raw audio; `embed.rs:38-48` documents exactly this trap and checks it loudly at load | the reference choice; Apache-2.0 to verify |
| `nemo_en_titanet_large` | 192-d (large is 192) | same | NVIDIA licence, not MIT |
| `wespeaker` / `3D-Speaker` ResNet34 / ECAPA variants | 256–512-d | same; the ResNet ones are usually *more* robust on far-field audio, which is a meeting recording | worth evaluating precisely because our input is far-field |
| WavLM / HuBERT layer-6 x-vector head | 512–768-d | a much larger download and a 50× larger activation footprint | disproportionate for a desktop app on 3.3 GB of VRAM |
| A cheap MFCC + i-vector / GMM-UBM | ~200-d | classical, no NN, and the failure mode is the one this crate already has | would be an incremental move along an axis that is already the problem |
| **A learned linear projection trained on our own fixtures** | 83 → k | a training set, which is exactly what we do not have | see § WHAT WE NEED TO MEASURE |

**The one thing none of these is blocked on is a correctness question, and it is
the interesting one:** `embed.rs:22-40` is right that a wrong embedding is worse
than no embedding, because it produces plausible clusters that are confidently
wrong. The `EmbedderSpec` shape checks catch a mis-shaped graph. They cannot
catch a graph whose front end is wrong in a way that still type-checks — a
log-mel with the wrong window, or a hop of 25 ms — and the module says so.

### 5.3 Clustering strategy alternatives, and whether agglomerative is the limiter

| alternative | would it help here? | evidence |
|---|---|---|
| **Spectral clustering / normalised cuts** on the same vectors | No. It is a better way to find the structure the vectors have. The real clip's structure is a continuum (§ 3.4), so a better structure-finder returns a better answer to the wrong question. | § 3.4, § 3.5 |
| **k-means++ with k swept** | No, and worse: it needs `k` up front, which is the quantity in question. | § 3.5 — the scored sweep shows the answer is a threshold problem even where it is solvable |
| **GMM / Bayesian GMM with BIC** | Marginally better at finding a gap than a raw threshold, and it can *report "one cluster"*, which is the right answer for this clip rather than the wrong answer for a 3-speaker meeting. Worth building as a *second opinion the UI can show*, not as a replacement. | § 3.4 — the only defect it addresses is the threshold's arbitrariness |
| **Over-segment then agglomerate (Kaldi-style)** | This is the one alternative with a real argument. 3.8 and 3.1 both show the window population changing with the VAD, and over-segmentation-and-merge is more robust to a mask whose granularity you do not control. It does not touch Cause 1 or 2. | § 3.1 |
| **Turn-length-aware windowing** | Already named by A4 in the `#[ignore]`d `short_turns_are_a_known_limitation` test and still the right answer: 1.5 s windows against a real meeting's ~2.3 s mean turn means most windows straddle a boundary. Measured here: the 1.5 s default against a 2,090 ms median VAD segment. | § 3.1 |
| **Replace the linkage with something else** | No. `cluster.rs:1-65` picks average linkage for stated, correct reasons and the monotone cost property the stopping rules rely on is real (verified on the real clip: 0.970 down to 0.529, monotone). | § 3.4 |

**The honest summary: the linkage is not the problem and no stopping rule can
manufacture structure the features do not contain.** § 3.8's scored variant table
is the proof: every alternative dial fixes one fixture and breaks another.

### 5.4 Whether better VAD / segmentation materially helps

**Measured, and the answer is "it helps a real number of windows and does not
help the answer."**

* *Level sensitivity.* Confirmed and sharpened: the internal detector keeps
  127 of 4,800 frames (2.6 %) of a speaker 20 dB down, and 64 % of 100 %-labelled
  uniform synthetic speech. Both are real defects. **Neither is the product
  path**, which always uses `EnergyVad`, and `EnergyVad`'s own frame-level
  fraction on this recording (0.52) is *lower* than the internal detector's
  0.865, so a "better" VAD in the sense of lower coverage is not obviously
  better for this stage.
* *The oracle experiment is the direct answer.* Handing the engine a **perfect**
  mask — the generator's own intervals, every labelled frame active — restored the
  quiet speaker from 0 windows to 57 and **did not change the reported count**
  (2 → 2), while lowering complete accuracy from 0.973 to 0.645. A VAD cannot fix
  a feature-vector problem. **Do not spend effort here expecting a speaker-count
  win.**
* *The VAD does change the answer, and that is the interesting part.* Same audio,
  same config: 3 speakers / 230 changes with the internal mask, 2 / 141 with
  `EnergyVad`'s. A third of the difference in the number the transcript's
  structure depends on is a VAD-geometry artefact, mediated by a 0.9 window gate
  that was tuned on a synthetic timeline.
* *The "86.5 % of this meeting is speech" item from `docs/JOURNEY.md` is
  answered.* It is **coverage, not a frame fraction.** `vaani-vad`'s own
  `VadStats` documentation records 0.52 of frames at or above the leave threshold
  against 0.86 of coverage, a factor of 1.65 from the 60 ms of padding either
  side of every segment and the minimum-silence rule absorbing every pause under
  300 ms. The internal detector measures 0.845 frame-active on this recording.
  86.5 % is not a defect; it is coverage, and reading it as a speech fraction is
  the mistake. That closes an open item in `docs/JOURNEY.md` with a measurement
  rather than a guess.

### 5.5 Threshold selection: what it should be calibrated on, and how, without an annotated corpus

**What it is calibrated on now, and why that is wrong.** A hand-read gap between
two synthetic distance distributions, with the constants then re-used against
real audio where the *scale of the distribution itself* is different (§ 3.3).
The threshold is a number in a space whose units move with the recording.

**What it should be calibrated on: a per-recording statistic, not a constant.**
The score is the `Auto` rule's own insight — average linkage's cost sequence is
monotone, so the information is in the *shape* — and the measurement says the
current implementation of that insight is broken in a specific, fixable way
(§ 3.8: unbounded ratio, so numerical dust wins). A defensible formulation, which
**this investigation does not implement**, is:

> cut below the largest jump in the cost sequence, where a jump must clear both
> `min_gap_ratio` **and** an absolute floor — either a configured cosine value, or
> self-calibrating: the median of the cost sequence. The latter needs no new
> configuration and is scale-free, and it would have rejected the 0.0025 → 0.0033
> step that makes `Auto` return the cap.

**How that is obtained without an annotated corpus.** Three routes, in the order
I would take them, and only the first two are free:

1. **Self-consistency, which needs no labels at all.** Re-run the clustering with
   the window population perturbed in ways that should not change the answer
   (mask fragmentation, ±1 window of hop, ±1 of smoothing) and keep the counts
   that are stable. The harness already produces the perturbation axis — 42 sweep
   points and 38 variants per run, every one of them a legitimate perturbation.
   § 3.1 is a demonstration: the count is *not* stable, so the honest report is
   "not stable", which is information a user can be shown.
2. **A held-out synthetic corpus built to be harder and more varied than the
   current one.** `synth::Voice` currently has a fundamental, a rolloff, three
   formants, breathiness and vibrato. Real meeting audio has none of that in
   fixed form: speakers overlap, the microphone is far, the room has a noise
   floor with structure, two people can share a fundamental. Extending the
   generator costs nothing and is inside this crate. **It would still measure the
   engine against a generator**, and § 3.5 already shows that generator-measured
   optimality and real-audio behaviour diverge.
3. **A small hand-labelled corpus. 30 minutes, three or four speakers, labelled by
   ear for who-spoke-when, is enough to calibrate one scalar and to detect a
   regression.** This is the only route that produces a real accuracy number, and
   it is the only thing in this document that requires something this repository
   does not have. It is cheap in human time and it is the single highest-value
   missing asset in the whole project — see § WHAT WE NEED TO MEASURE.

---

## 6. EXPERIMENT PLAN

Ordered so that each step is worth doing whether or not the next one is
approved. Every step states what would falsify it.

**E1 — Re-normalise the blocks by their measured spread. Highest value, one flag.**
Change `normalise_in_place` to divide each non-absolute block by the block's
measured robust spread instead of `within_speaker_spread`, behind a config flag
defaulting to the current behaviour. *Prediction:* block shares move toward the
configured `w²` values; the real clip's count moves; a one-speaker recording's
mean distance collapses. *Falsified by:* shares unchanged, or one-speaker fixtures
shattering. *Cost:* one branch in one function, plus a flag. Note the known trap,
documented in `features.rs:727-756` for the *other* direction: dividing by a
measured statistic re-admits the between-speaker term, and this measurement would
be done on the *same* file the clusters come from. That is the whole tension and
it is why this is a flag and not a change. See section 8 for why it is not done
here.

**E2 — A `within_speaker_spread` calibration from real audio, per block.**
Fifteen minutes of work given E1's instrumentation. The measured ratios in § 3.3
(14×, 132×, 132×, 178×, 492×) are the input. *Falsified by:* the ratios differing
by more than ~3× between the two real-clip activity sources, which would mean the
constant cannot be a constant.

**E3 — Does the pitch dimension earn its place?** Set
`BlockWeights::pitch = 0` permanently if E1 lands, since it is measurably inert
on real audio and measurably harmless on synthetic (all five fixtures unchanged).
*Falsified by:* a fixture where pitch-weight-0 hurts. The three-speaker
similar-pitch fixture is the one to watch.

**E4 — Separate the activity source from the diarizer's configuration.**
`min_window_speech_ratio = 0.9` is the single highest-leverage unmeasured dial
(0.0 → 0.5 → 0.9 moves the real clip 4 → 2 → 3 on the internal path and 3 → 5 →
2 on the external path). Sweep it against the *VAD's own segment-duration
distribution*, i.e. derive the window gate from the observed granularity of the
mask rather than from a constant. *Falsified by:* the count not moving with the
gate once E1 has landed — which would mean the gate is not the mediator and
Cause 3 is wrong.

**E5 — A stopwatch on the geometry, not the engine.**
Turn-length-aware window length: choose `window_frames` from the observed
distribution of inter-turn gaps in the mask, the way the grouping policy chooses
its pause threshold. A4 named this as the fix for the `#[ignore]`d short-turn
limitation and it is still right. *Falsified by:* the `3spk-3s-turns` fixture not
improving, which would mean the failure is the features and not the window.

**E6 — Saturating confidence.** `turn_scores` p50 = 1.000 on a real recording
means the field is dead. Either make `agreement` a real margin (distance to the
runner-up centroid) or drop it. *Cost:* one line in `smooth.rs:227`. *This is the
only item in this plan with no evidence dispute: a confidence field that always
says 1.0 is worse than none, because a UI will render it.*

**E7 — Fix `SpeakerCount::Auto`'s unbounded ratio** with the median-cost floor
from § 5.5. *Falsified by:* the rule still returning the cap on the real clip. The
existing `#[ignore]`d `the_gap_rule_over_splits_this_fixture` test is the check,
and per the rules it is kept either way.

**E8 — With network access: the neural path.** Enable a `neural` feature
parallel to `vad-probe`, drop an ECAPA-TDNN ONNX into `artifacts/diarization/`,
and run the same harness over it. The `embed.rs` seam is already written. The
measurement that decides it is not a DER — it is the **scored synthetic table**,
where a real embedder should score materially above 0.96 purity on
`3spk-similar-pitch` and `3spk-3s-turns`, plus a **speaker-signal** column in
`geometry.blocks` where pitch's 0.0013 and `spectral_shape`'s 0.599 should
collapse towards each other.

**E9 — With a labelled corpus: everything above, for real.** 30 minutes, three or
four speakers. Then the harness gains a `reference` on a real case, and every
number in this document that says "no accuracy is claimed" becomes a measurement.
The harness is already structured for it: `CaseKind::RealClip` with
`reference: Some(..)` is the only change, plus the contract test that currently
*fails* if a real case carries a reference.

---

## 7. WHAT WE NEED TO MEASURE

In priority order, with what each unlocks.

1. **A labelled corpus. 30 minutes, 3–4 speakers, who-spoke-when.** Nothing else
   in this list matters as much. Every "no accuracy is claimed" in this document
   is a consequence of its absence, and the harness is already built to consume
   it: `CaseKind::RealClip` with a `reference` is the only change needed, plus a
   30-line loader. Until it exists, DER/JER are undefined here and any number
   claiming to be one is fiction.
2. **A second real recording, to see whether § 3 is a fact about the audio or
   about this meeting.** Every § 3 finding is from one file. The one that would
   generalise badly if it is file-specific is the `spread_ratio` column; the one
   that would generalise badly if it is general is the pitch continuum. One more
   recording separates them.
3. **An overlap measurement.** No overlapped speech exists in any fixture and the
   engine has no overlap handling at all (§ 1.8). `docs/JOURNEY.md` calls this a
   real meeting property. Nothing can be said about it until there is audio with
   it.
4. **The far-field question.** Every fixture is a synthetic voice at 0 dBFS with a
   white noise floor. A meeting is a voice at 0.5 m from a laptop microphone in a
   reverberant room, and 20 dB down from another voice. The one measurement that
   exists in this direction is the 20 dB fixture, and it is the one where the
   answer is "the features are fine, the gate ate it".
5. **What `EnergyVad`'s mask costs *specifically*.** Measured so far only as its
   effect on the count. The window-level distribution of *why* 44 % of candidate
   windows are rejected — silence fraction, segment length, boundary proximity —
   would turn E4 from a sweep into a derivation.

---

## 8. WHAT WE SHOULD NOT CHANGE YET

The section that matters. Each of these looks attractive, has support, and
should not be done.

**Do not retune the merge-distance threshold.** It is tempting because the
shipped 0.8925 was chosen from synthetic distance distributions and the real
recording disagrees with them. But § 3.5 measures the shipped value sitting
inside a **1.53× plateau** on the only fixture where a reference exists, with
purity 0.958 and recall 1.000 across the whole plateau, and § 3.4 shows the real
recording's merge sequence has no gap to move into. Changing a number that is
already correct where it can be measured, in order to change it where it cannot
be measured, is a coin flip dressed as a tuning pass.

**Do not switch the stopping rule to a gap/spectral method.** `Auto` already
exists, already over-splits the fixtures, and already degenerates to the cap on
real audio for a reason that is a bug in the rule rather than a fact about the
audio (§ 3.8). Fix `Auto` if you like (E7), but do not promote it: a better
structure-finder returns a better answer to a question whose answer is that there
is no structure.

**Do not swap the linkage for something modern.** § 3.4 shows the cost sequence is
monotone and well-formed on real audio, and `cluster.rs:1-65`'s reasons for average
linkage are correct. The problem is upstream of the linkage and downstream of it,
and neither spectral clustering nor k-means reaches it.

**Do not rewrite the feature vector from the decomposition in § 3.3.** This is the
most tempting item in the document and the worst. The decomposition is measured
and it does say the pitch block is inert on real audio. But "pitch is inert" and
"replace the vector" are different claims, and the second one is unsupported:
there is no evidence here that *any* 83-dimension spectral descriptor would do
better, and strong evidence that the synthetic fixtures would get worse (the
spectral blocks are currently doing the work at purity 0.96). A rewrite needs a
reference. It does not have one.

**Do not treat the VAD as the fix.** § 3.7 is the experiment: a perfect mask
restores a third of the evidence and does not move the count, while lowering
complete accuracy. Effort spent on VAD recall will not buy a speaker count. The
VAD *does* change the number (§ 3.1) and that deserves attention (E4), but the
change is a window-gate interaction, not a VAD-quality problem.

**Do not act on `docs/JOURNEY.md`'s "the VAD calls 86.5 % of this meeting
speech" as an open defect.** § 5.4 answers it: 86.5 % is segment coverage, the
frame-level fraction is 0.52, and the factor of 1.65 is documented hangover and
gap-bridging doing exactly what they are for. This item should be closed, not
worked.

**Do not raise `max_speakers` or make the engine more willing to over-split.**
`SegmentationBias::SplitEasily` is the default and the sweep already reaches the
cap of 8 at effective distances below 0.6. A diarizer that reports 8 speakers for
a 14-minute meeting because nobody wants to over-merge has traded a visible,
repairable error for a confident, wrong one — the exact trade `SegmentationBias`'s
own documentation refuses to make.

**Do not ship the neural path on a licence assumption.** § 5.1. Two unconfirmed
weight licences in one distribution is a worse position than one, and the
question needs the same written confirmation `docs/JOURNEY.md` already asks for
on SraVaani.

**Do not add `embedder.onnx` discovery to the shipped build yet.** The probe is
already wired and correct, but the feature that would let the benchmark *use* a
neural path is `vad-probe`-shaped and `embed` does not have one. That is a defect
in this crate; it is reported as a fix to make, not as a behaviour to change now.

**Do not act on `turn_scores`.** Listed under E6 rather than here because it is
genuinely broken rather than genuinely contested: a confidence of 1.000 on 310
real spans is a false statement. But "make it honest" and "make it useful" are
different changes, and only the first is justified today.

---

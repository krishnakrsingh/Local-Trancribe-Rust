# The timestamp contract

**Status:** frozen. This file, together with the types it names in `vaani-core`,
is the authority on what every number that looks like a time in VaaniDesk means.
If a crate and this file disagree, the crate is wrong.

**Scope:** `crates/vaani-core/src/transcript.rs` and every call site of
`Time`, `MediaLength` and `MediaWindow` anywhere in the workspace. It does not
govern the ONNX Runtime's or the log-mel's internal frame indices.

---

## 1. The problem this file exists to end

`Transcript::duration` used to be a `Time`, and so were `Segment::start`,
`Segment::end`, `Word::start` and `Word::end`. One type, two incompatible
meanings:

* a **length** — "how long the media is", a quantity;
* an **absolute offset** — "when in the media this happened", a position.

Nothing in the type distinguished them, so every call site had to *know* which
one it was holding, and a plausible-looking comparison between the two compiled
silently. That is not a hypothetical. It produced two real defects in two
different crates:

1. **A11 (`vaani-segment`).** A media clamp compared an **absolute** segment
   end against `Transcript::duration`, which is a **length**. For a transcript
   whose times ran `300 000..600 000` with a declared duration of `300 000`,
   that crushed every segment to `300 000..300 001` and left every word outside
   its own segment. Three conformance checks failed at once. The fix was a
   downstream workaround type, `vaani_segment::MediaBounds`, because the
   contract crate had no honest way to say "here is the media".
2. **A2 (`vaani-app`, `vaani-cli`).** `AsrStage::run` planned chunks with
   `Chunker::new(policy).plan(&speech, audio.duration())` while `speech` holds
   **absolute** times and `audio.duration()` is a **length**. For any buffer
   that does not start at zero, the chunker clamped absolute ends against a
   relative total and the last STFT frame's clamp did the same to the
   accumulated transcript. It was recorded as
   `times.no_time_precedes_the_origin` in
   `vaani_qa::conformance::asr_stage_known_failures` and it is **now fixed and
   the entry deleted** — see §9.

Both were arithmetic that read correctly. That is the whole problem: the mistake
is not gross, it is *readable*.

## 2. The rule

> **A `Time` is a position on the media timeline. A `MediaLength` is an
> amount of time. A `MediaWindow` is the extent of the media in positions.
> The three are different types and the compiler enforces it.**

### Units and epoch

Every `Time`, `MediaLength` and `MediaWindow` in this product is an **integral
number of milliseconds** (`i64`) measured from the **start of the source
media** — the instant at which the first sample of the recording was captured.
There is no other epoch, no wall clock, and no frame index in this contract.

`Time` is signed so that a chunk-local ASR time may legitimately be slightly
negative (we keep overlap between chunks precisely so a straddling word is not
clipped) and be clamped to zero in one place. Every `Time` that has left a
stage is `>= 0`.

### What each type means

| Type | Meaning | Where it appears |
|---|---|---|
| `Time` | an **absolute** offset, ms from the start of the source media | `Segment::start`/`end`, `Word::start`/`end`, `AudioBuffer::start_time`, `SpeechSegment::start`/`end`, `SpeakerSpan::start`/`end`, `Chunk::start`/`end`, `MediaWindow::start`/`end` |
| `MediaLength` | a **length**, ms, never negative, never a position | `Transcript::duration`, `Segment::duration()`, `Cue::duration()`, `MediaWindow::length()` |
| `MediaWindow` | the **extent** of the media: `start..end`, both absolute | every clamp in the product |

`Time` is `#[serde(transparent)]` over `i64` and `MediaLength` is too, so the
JSON and SQLite representations are **byte-identical** to what they were
before this change. Only the Rust type changed. `duration_ms` in an export, in
`transcripts.duration_ms`, and in `jobs.duration_ms` are still plain integers
and still mean *length*.

## 3. The legal operations

| Operation | Legal? | How |
|---|---|---|
| `Time + Time -> Time` | **yes** | `Time::saturating_add`. This is a chunk offset: both operands are positions. |
| `Time - Time -> MediaLength` | **yes** | `Time::span_to(other)`. The **only** way to get a length out of two positions. |
| `MediaLength + MediaLength -> MediaLength` | **yes** | `MediaLength::saturating_add`. |
| `Time + MediaLength -> Time` | **yes** | `Time::offset_by(length)`. A position advanced by an amount: this is how a media *end* is built from a media *start* and a media *length*. |
| `Time == Time`, `Time < Time` | **yes** | ordering positions against positions. |
| `MediaLength == MediaLength`, `MediaLength < MediaLength` | **yes** | ordering amounts against amounts. |
| `MediaLength == Time`, `Time < MediaLength`, … | **NO — does not compile** | no `PartialOrd`/`PartialEq` crosses the two. This is the defect, made into a type error. |
| `MediaLength -> Time` | **NO implicit conversion** | there is no `From`. Build a `MediaWindow` and ask it for its `end()`. |

`Duration` in this document always means `vaani_core::MediaLength`. It is
deliberately **not** called `Duration`, because `vaani-ui` imports
`std::time::Duration` and a contract type that collides with the standard
library's is a contract type people will shadow by accident.

## 4. `Transcript::duration` specifically

```rust
pub struct Transcript {
    /// Length of the source media in milliseconds, when known. `None` for a
    /// live/streaming source.
    pub duration: Option<MediaLength>,
    // ...
}
```

* It is a **length**. It is the number the UI shows as "Duration", the number
  written to `duration_ms` in SQLite and in an exported JSON envelope, and the
  number a realtime factor divides by.
* It is **not** a bound on any time. `segment.end <= transcript.duration` does
  not compile, and that is the point.
* It is `None` when the source was live or when nothing measured it. Every
  consumer must have an answer for `None`; `Transcript::inferred_length()`
  exists for the display cases and is documented as an inference, not a
  measurement.

## 5. How a caller clamps to the media

There is exactly one way, and it is a `MediaWindow`:

```rust
// Where the media is, in absolute terms. The AudioBuffer knows.
let window = vaani_core::MediaWindow::of_buffer(&audio);

// Accumulate an ASR chunk. The rebasing and the clamp are one call.
transcript.push_chunk_within(chunk.start, result.segments, &window);

// Group words into readable segments, inside that window.
vaani_segment::group_within(&mut transcript, window, &policy)?;

// Plan ASR chunks inside that window. `plan_within`, not `plan`.
let chunks = Chunker::new(policy).plan_within(&speech, &window);
```

`MediaWindow` has three constructors, and the choice between them is the whole
decision:

| Constructor | Use when | Window |
|---|---|---|
| `MediaWindow::of_buffer(&AudioBuffer)` | **the normal case** — a pipeline holds the decoded audio | `buffer.start_time .. buffer.start_time + buffer.length()` |
| `MediaWindow::new(start, end)` | a slice whose extent is genuinely known from elsewhere | exactly what you pass, normalised so `end >= start` |
| `MediaWindow::of_length(length)` | the media is known to start at zero — a whole file, a test fixture | `0 .. length` |

`of_length` is the only constructor that assumes an origin, and it is named for
the assumption it makes. **Never use it for a buffer with a non-zero
`start_time`**; that is the mistake this contract exists to prevent, and the
constructor exists only so that the zero-origin case is also spelled honestly
rather than written inline.

`MediaBounds`, which lived in `vaani-segment` as a downstream workaround, is
gone. `vaani_segment::MediaWindow` is a re-export of the contract type, so
there is exactly one such type in the workspace and it lives in the crate that
owns the meaning.

`vaani_vad::Chunker` follows the same rule. Its parameter is an **absolute
end**, so it is named `media_end` (it was `total`, which is what invited the
mistake) and there is a `plan_within` / `iter_within` that takes a
`MediaWindow` and therefore cannot be handed a length at all. Both production
pipelines use the window form.

## 6. What is legal per site

The surveyed sites, and what each one is allowed to do with a time. Anything not
listed here is a bug.

| Site | May treat `Transcript::duration` as | Correct? |
|---|---|---|
| `vaani-app/stages/asr.rs` writing `duration` | a length | yes — it writes `window.length()` |
| `vaani-cli/pipeline.rs` writing `duration` | a length | yes |
| `vaani-storage/transcripts.rs` → `duration_ms` | a length | yes |
| `vaani-export/render.rs` → `Duration` row, `source.duration_ms` | a length | yes |
| `vaani-ui/view.rs` footer | a length | yes — it now falls back to `inferred_length()`, a length |
| `vaani-segment/tests/properties.rs` bounds check | **nothing** | it was comparing an absolute end to a length; it now compares against a `MediaWindow` |
| `vaani-cli/tests/e2e_acceptance.rs` reach check | **nothing** | same; it now compares against the buffer's window |
| `vaani-ui/sample.rs` bounds check | **nothing** | same |
| `vaani-segment` `MediaBounds` | a window | was a workaround for the contract's gap; now the contract's own type, re-exported |
| `vaani-vad` `Chunker::plan(timeline, media_end)` | an absolute end | the parameter is renamed from `total`; `plan_within` takes a `MediaWindow` and is what both pipelines use |
| `vaani-qa` `synthetic_transcript` | a length in `duration`, absolute in the segments | this fixture *was* the A11 trap — a 89 600 ms length with 300 000..389 600 absolute times |

## 7. Known residual ambiguity (deliberate, not forgotten)

Five things are still not perfectly typed. Each is recorded here because each is
blocked by a crate boundary this change was not permitted to cross, and each is
a compile-clean, documented hole rather than an accident.

1. **`AudioBuffer::duration() -> Time`** (`vaani-core/src/media.rs`). It is a
   length and it is typed as a position. Changing it to `MediaLength` breaks
   `vaani-diarization`, which Track B owns and this track may not edit. The
   correct door is `MediaWindow::of_buffer(&audio)`, which both live and diar
   call sites now use. **Remaining step:** change the return type and cascade,
   in one commit, after Track B lands.
2. **`SpeechTimeline::speech_ratio(total: Time)`** and
   **`DiarizationResult::coverage(total: Time)`** (`vaani-core/src/pipeline.rs`).
   Both take a *length*. Both are called by `vaani-diarization`, which may not
   be edited here, and `pipeline.rs` is not in this track's ownership. **Both
   are named `total` or `coverage(total)`, which is at least honest about being
   a quantity.** **Remaining step:** change both to `MediaLength` and cascade
   into `vaani-diarization` and `vaani-vad`, in a commit that can move those
   crates' call sites together.
3. **`AsrEngine::transcribe` → `AudioChunk::duration() -> Time`**
   (`vaani-asr/src/engine.rs`). `vaani-asr` is frozen and numerically validated.
   `AudioChunk` has no `start_time` at all, which is precisely why its
   `duration()` is only meaningful as a length. **Remaining step:** frozen. Do
   not touch. Any future `AudioChunk` change is an ASR-frozen change.

4. **`Chunker::plan(&timeline, media_end: Time)`** keeps a `Time` parameter
   because it *is* an absolute end, and 31 `plan` call sites (mostly
   `vaani-vad`'s own tests) plus 2 `plan_with_raw` and 3 `iter` sites would have
   to move to the window form. The parameter has been renamed from `total` to
   `media_end` and documented, and both production call sites (`vaani-app`,
   `vaani-cli`) now use `plan_within`. The old entry point remains so the other
   crates compile, and a caller reading the signature now sees `media_end` rather
   than a name that invited the length. **Remaining step:** delete
   `plan`/`plan_with_raw` and cascade, in a commit that can also delete the
   `vaani-vad` and `vaani-qa` test call sites.

5. **`Time::saturating_sub(Time) -> Time`** returns a *span*, not a position.
   It is used by `AudioBuffer::slice_mono` to produce chunk-local offsets, which
   is a third meaning. It is documented as such on the method. **Remaining
   step:** `span_to` is the correct API; `saturating_sub` is retained only for
   the frozen-adjacent call sites.

## 8. How a mistake is now caught

Three layers, in order of strength.

1. **The compiler.** `MediaLength` and `Time` are unrelated types. The A11
   expression — `if seg.end > tr.duration.unwrap()` — does not compile. This is
   asserted by `compile_fail` doc tests on `MediaLength`, which run in the
   ordinary `cargo test --workspace` gate and need no dependency.
2. **The conformance harness.** `vaani-qa`'s work audio starts at **+5
   minutes** for exactly this reason: a stage that returns buffer-relative
   times, or clamps against a length, is caught on synthetic audio in under a
   second.
3. **The regression tests** in `crates/vaani-core/src/transcript.rs`, which
   reproduce the A11 and A2 cases verbatim and assert the fixed behaviour.

## 9. The A2 entry is gone

`vaani_qa::conformance::asr_stage_known_failures()` now returns an empty list,
and the entry it carried has been deleted. The registry is policed by
`vaani-qa/tests/known_failures.rs`, which re-drives every entry and fails when
one starts passing — that test is what forced the deletion, so this is the
mechanism working rather than an oversight.

The check it used to excuse now **passes**, and
`every_tracked_failure_still_reproduces` asserts that as a `Verdict::Pass`
rather than deleting the coverage. The measured result:

```
PASS times.no_time_precedes_the_origin — 0 of 6 pairs start before 05:00 ms
```

against the same +5-minute work audio that reported *2 of 6* before.

The fix has three parts, and only the first is a behaviour change:

1. `AsrStage::run` builds `MediaWindow::of_buffer(audio)` once and passes it to
   `Chunker::plan_within` and `Transcript::push_chunk_within`. The previous
   `push_chunk_within(..., audio.duration())` was passing a **length** as a
   bound, which on a +5-minute buffer pinned the whole transcript to
   `0..89 600` — before the recording even started.
2. `Chunker::pad_and_coalesce` clamps its leading pad to the window's start
   rather than to a hard-coded `0`. This is the one genuine behavioural change
   in `vaani-vad`, and it is a no-op for any media that begins at zero, which
   `a_zero_origin_buffer_is_unchanged_by_the_window` asserts by comparing
   `plan_within` against `plan` output directly.
3. `Transcript::duration` is a `MediaLength` and `Chunker`'s parameter is
   `media_end`, so neither mistake can be written again.

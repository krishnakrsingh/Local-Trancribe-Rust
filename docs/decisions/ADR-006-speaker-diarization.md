# ADR-006 — Speaker diarization architecture

**Status:** Accepted (not yet built)
**Date:** 2026-09-27

## Context

A meeting transcript is much more useful with `Speaker 1 / Speaker 2` than without, and
Hindi/English code-switched meetings are where attribution matters most. But diarization
must never be able to destroy a transcript.

The domain model already encodes that: `Segment::speaker` is `Option<usize>` and `None` is
a **normal state, not an error** — ASR-only mode is a complete product mode
(`crates/vaani-core/src/transcript.rs:149-152`). `ErrorCode::DiarizationFailed` is defined
as never fatal (`error.rs:104-106`) and its fixed message says the transcript is complete
and only the labels are missing.

Diarization runs on the same 3.3 GB GPU as an 887 MB encoder, on the same process, and
must be able to abandon the GPU cleanly.

## Decision

Two models, in sequence, using a **sherpa-onnx** Rust crate as the host, with both weights
downloadable from **GitHub Releases and no Hugging Face token required**.

| Stage | Model | Purpose |
|---|---|---|
| 1 | pyannote `segmentation-3.0`, **int8** | frame-level speaker-change detection: who is speaking when |
| 2 | `nemo_en_titanet_small` | speaker embeddings for the detected regions, then clustering |

Rationale for the split: segmentation and embedding are different problems with different
costs. int8 segmentation is cheap and its output is *structure* (boundaries and overlap),
which survives quantisation. Embeddings are compared by cosine similarity, where int8
quantisation noise would corrupt the distances. Clustering then runs on a few hundred
embeddings rather than on 16 kHz audio, so its cost is negligible either way.

Choosing sherpa-onnx rather than `pyannote.audio` or a Python worker means diarization is
in-process, in Rust, on the same ORT session machinery as ASR — and needs no Python at
runtime, consistent with the rest of the product.

## Consequences

Both models come from GitHub Releases or pre-bundled assets, with `access` in the model manifest set to `open` and no token required. Consistent with the core ASR architecture (ADR-005), the product operates 100% offline with zero token gating.

The pipeline is `Vad → Asr → Diarization → Alignment` in `JobPhase` order
(`crates/vaani-core/src/job.rs:44-50`), and the phases are not independent: diarization must
align its frame-level decisions to the ASR word time spans, which is what `Alignment` exists
for. `JobPhase::Diarization` is weighted 0.12 and `Asr` 0.62 of wall time (`job.rs:149-150`)
— presentation-only estimates, but they are what makes the overall bar move.

The total VRAM budget is now the binding constraint: 887 MB encoder + ~400 MiB activation
headroom + both diarization models, against 3.3 GB. `MachineCapabilities::vram_fits` exists
to answer that *before* load rather than as an OOM at chunk 40 (`capability.rs:189-201`).
Sequential model release is therefore required, not optional: the segmentation session must
be dropped before the embedding session is created, and both before ASR resumes.

Clustering must be deterministic. Speaker indices are persisted and referenced by
`Segment::speaker`, and `Transcript::speaker_mut` materialises a placeholder for an index it
has not seen (`transcript.rs:398-411`), so a non-deterministic clustering produces
transcripts whose speaker labels move between runs. There is no test for this yet because
there is no implementation.

`Speaker::name` is user-supplied and `display_name()` falls back to `Speaker N`
(`transcript.rs:238-240`). Renaming must never touch transcript text; this is tested at
`transcript.rs:589-600`.

## Alternatives rejected

| Alternative | Why not |
|---|---|
| pyannote 3.1 speaker-diarization pipeline | Python at runtime; a second process competing for VRAM |
| `nemo_en_titanet_small` alone, no segmentation | an embedding model cannot tell two overlapping speakers apart, and diarization meetings overlap |
| A single segmentation model with int8 embeddings | quantisation noise lands directly in the cosine distances that clustering depends on |
| Hugging Face-hosted diarization weights | pulls a token requirement into a capability that should not need one |
| Forced alignment instead of clustering | requires ASR internals; and the goal is *who*, not *when* |
| Making diarization required | it would turn an optional, fallible stage into a single point of failure for the whole product |

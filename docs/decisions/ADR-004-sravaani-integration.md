# ADR-004 — SraVaani model integration

**Status:** Accepted (built)
**Date:** 2026-09-27

## Context

**ARTPARK-IISc/SraVaani-1.0**: ~430M parameters, a FastConformer encoder (d_model 1024,
128 mel in, 8× subsampling; the layer count is not declared in `configuration_sravaani.py` and
is relied on nowhere here) and a hybrid TDT/RNNT decoder, pretrained on 31,255 h of the Vaani
dataset and fine-tuned on ~31,270 h across 65 Indian languages and dialects. MIT, but **gated
on Hugging Face**. Hindi average WER 12.4. It ships as a 908,884,628-byte TorchScript archive
exposing `.encoder` and `.decoder_joint`, and must run through ONNX Runtime on a 4 GB laptop
GPU, from a Rust binary, with the product still working on CPU when acceleration is
unavailable. Config: vocab_size 5000, `blank_id` 5000, durations [0,1,2,3,4], `num_durations`
5, `max_symbols` 10, `pred_hidden` 640, `pred_rnn_layers` 1, `feat_in` 128, sample_rate 16000,
`subsampling_factor` 8. Tokeniser: SentencePiece **unigram**, 5,000 pieces, `unk_id` 0, 2,190
with the U+2581 metaspace prefix; blank is **outside** the SP vocabulary.

## Decision

Export graphs from the same TorchScript at **different precisions**, and feed them different
dtypes. Three export decisions, each forced by something that cost real debugging time.

| Graph | Precision | Size | Role |
|---|---|---|---|
| `encoder.fp16.onnx` | **native fp16 I/O** | 886,484,931 B | production |
| `encoder.onnx` | fp32 | 1,772,812,911 B | reference / CPU fallback |
| `decoder_joint.onnx` | **fp32** | 43,033,439 B | TDT emission step |

1. **The encoder is exported natively from an fp16 module** — `encoder.half()` then
   `torch.onnx.export(..., opset_version=17, dynamo=False, external_data=False)`. **Not** via
   `onnxconverter_common.float16`, which rewrites SraVaani's integer length prologue
   `/pre_encode/Cast -> Add -> Div -> Floor` to fp16 and produces a graph ORT rejects.
2. **`dynamo=False`** — the legacy TorchScript exporter. The dynamo exporter cannot reproduce
   NeMo's `annotate(number, ...)` dynamic-shape hints, so the attention mask would freeze and
   the time axis would break.
3. **`preproc.pt` is unpacked once** into `mel_window.f32`, `mel_fb.f32` and `preproc.json`; it
   is a zip+pickle, unreadable from Rust without a pickle parser.

## Consequences

**The fp16-encoder / fp32-decoder split is the substance of this ADR.** The FLOPs are in the
encoder; the precision that matters is in the decoder. The decoder is only 43 MB, but its LSTM
state is threaded through **every emission step of a whole utterance** — minutes of audio,
thousands of sequential steps — so fp16 error would compound into wrong tokens. The encoder is
one feed-forward pass; fp16 there costs accuracy once and buys 2.0× on disk and 1.6× in wall
time (67–72 ms vs 110–112 ms for 16 s; RTF 0.004 vs 0.007). The fp16 boundary means the Rust
side speaks f16 at the edge, centralised in `F16Tensor::from_f32` (`session.rs:83`) so the
media and ASR crates never learn the encoder's precision. The front end still computes log-mel
in **f32**: the per-feature normalisation divides by a standard deviation, and losing mantissa
bits there is a real accuracy cost. `I32Tensor` and `I64Tensor` are separate types for the same
reason — the encoder takes int64 `feature_lengths`, the joint takes int32 targets, and swapping
them is a silent-wrong-answer bug. The export is verified, not assumed: 887 MB
single self-contained file (under the 2 GB protobuf limit, so no sidecar to lose and ORT can
mmap it), **99.609% argmax agreement** with the fp32 reference (`max|d| = 0.046`), aborting
below 99.0%; `decoder_joint` parity logits `max|d| = 2.29e-05`, `h` `4.76e-07`.

Two export limits are permanent. **The `decoder_joint` dead `target_length` output cannot be
dropped**: a TorchScript submodule cannot be called from a plain `nn.Module` or a traced
closure — *"not part of the active trace"* — so the Rust side requests
`["logits", "h_out", "c_out"]` **by name**. And **the log-mel front end must be reimplemented
in Rust** (`vaani-asr`, not yet written); its exact order is in
[MODELS.md](../agent/MODELS.md#the-log-mel-front-end-to-be-reimplemented-in-vaani-asr), where
two values look wrong and are not — `log_zero_guard_value` is `2⁻²⁴`, not NeMo's usual `1e-5`,
and the normalisation divides by `N−1` (Bessel) then **adds** `1e-5` to the standard deviation
rather than flooring it. `sentencepiece-rs` 0.2 is the intended Rust tokeniser — zero
dependencies, no C++ toolchain, Windows-safe — but it is young and needs a **byte-for-byte
decode parity test against Python `sentencepiece` 0.2.2**.

## Alternatives rejected

| Alternative | Why not |
|---|---|
| fp16 everywhere, including the decoder | LSTM state compounds over thousands of sequential emission steps |
| fp32 everywhere | 2.0× disk, ~1.6× time, and the fp32 encoder barely fits 3.3 GB with activations |
| `onnxconverter_common.float16` / the dynamo exporter | a graph ORT rejects; and frozen dynamic axes |
| The official SraVaani ONNX release (Google Drive) | unversioned blob, no per-file checksums, no pinned opset |
| A different ASR model | 65 Indian languages at ~12% Hindi WER; nothing else in reach matches |
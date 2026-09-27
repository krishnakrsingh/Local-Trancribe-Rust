# ADR-004 — SraVaani model integration

**Status:** Accepted (built)
**Date:** 2026-09-27

## Context

**ARTPARK-IISc/SraVaani-1.0**: ~430M parameters, a FastConformer encoder (d_model 1024, 128 mel
in, 8× subsampling) and a hybrid TDT/RNNT decoder, released under the **MIT License** (gated on Hugging Face upstream, but bundled directly into VaaniDesk distributions for 100% offline out-of-the-box operation), Hindi
average WER 12.4, 65 Indian languages, pretrained and fine-tuned on ~62,500 h of the Vaani
dataset. It ships as a 908,884,628-byte TorchScript archive exposing `.encoder` and
`.decoder_joint`, and must run on a 4 GB laptop GPU through ONNX Runtime, from a Rust binary,
with the product still working on CPU when acceleration is unavailable. Config: vocab 5000,
`blank_id` 5000, durations [0,1,2,3,4], `max_symbols` 10, `pred_hidden` 640,
`pred_rnn_layers` 1, `feat_in` 128, 16 kHz, `subsampling_factor` 8. Tokeniser: SentencePiece
**unigram**, 5,000 pieces, `unk_id` 0, 2,190 with the U+2581 metaspace prefix; blank is
**outside** the SP vocabulary.

## Decision

Export graphs from the same TorchScript at **different precisions**, and feed them different
dtypes. Four decisions, each forced by something that cost real debugging time.

| Graph | Precision | Size | Role |
|---|---|---|---|
| `encoder.fp16.onnx` | **native fp16 I/O** | 886,484,931 B | production |
| `encoder.onnx` | fp32 | 1,772,812,911 B | reference / CPU fallback |
| `decoder_joint.onnx` | **fp32** | 43,033,439 B | TDT emission step |

1. **The encoder is exported natively from an fp16 module** — `encoder.half()` then `torch.onnx.export(..., opset_version=17, dynamo=False, external_data=False)`. **Not** via `onnxconverter_common.float16`, which rewrites SraVaani's integer length prologue `/pre_encode/Cast -> Add -> Div -> Floor` to fp16 and produces a graph ORT rejects.
2. **`dynamo=False`** — the legacy exporter; the dynamo one cannot reproduce NeMo's `annotate(number, ...)` dynamic-shape hints, so the attention mask would freeze.
3. **`preproc.pt` is unpacked once** by the exporter into `mel_window.f32`, `mel_fb.f32` and `preproc.json` — a zip+pickle, unreadable from Rust without a pickle parser — and `tokenizer.model` is copied into the same directory ([COMMON_FAILURES §12](../agent/COMMON_FAILURES.md)).
4. **The log-mel front end is a hand-written Rust port** (`vaani-asr::frontend`) whose every constant comes from `preproc.json`. Five conventions look wrong and are not:

| Finding | What it is | What "correcting" it does |
|---|---|---|
| Window is **centred** | `torch.stft` pads a short window on **both** sides: `window_offset = (n_fft − win_length)/2 = 56`, not 0 | shifts the spectrogram 3.5 mel frames; output still looks right, parity off by 3–6 |
| Window is **symmetric** | `hann_window(400, periodic=False)`, not periodic | up to 6e-3 error in the highest-energy bins |
| **Bessel** correction | variance divides by `N − 1` | a bias that grows with clip length: 5 s fine, 20 s not |
| `log_zero_guard_value` | exactly `2⁻²⁴`, not NeMo's `1e-5` | changes the normalisation of the low-energy bins |
| **One extra STFT frame** | `center = true` → `1 + valid_frames` columns, the last is padding | a dropped or duplicated frame at the end of every chunk |

Each was a real bug, found by comparing **every one of the 128 × 101 fixture cells** against PyTorch rather than sampling; worst cell now 7.99e-05 against a 2e-4 tolerance. Step order in [MODELS.md](../agent/MODELS.md#the-log-mel-front-end-in-vaani-asr).

## Consequences

**The fp16-encoder / fp32-decoder split is the substance of this ADR, and the arithmetic decides it.** The joint is **43 MB**, so fp32 there costs nothing measurable; the encoder is **887 MB** and holds nearly all the arithmetic. But the joint's LSTM state (`h`, `c`) is threaded through **every emission step of a whole utterance** — minutes of audio, thousands of sequential steps — so fp16 error would compound into wrong tokens. The encoder is one feed-forward pass: fp16 there costs accuracy once and buys 2.0× on disk and ~1.6× in wall time (67 ms vs 111 ms for 16 s, RTF 0.004 vs 0.007). Parity is verified, not assumed: 99.609% argmax agreement with the fp32 reference (`max|d| = 0.046`), `decoder_joint` logits `max|d| = 2.29e-05`, and the decode itself — **202/202 token ids** and **108/108 word timings** identical to the reference decode of a 40 s clip, exact text.

**Two rules in the TDT decode loop are invariants, not style** (`sravaani.rs:30-39, 282-345`): **(1) state advances only on a non-blank** — on blank, `h`, `c` and `last` are untouched and only the frame index moves, and the obvious "simplify the loop" edit that treats a blank as a state update makes the model emit garbage that still reads as language; **(2) the `max_symbols` guard adds an *extra* frame on top of the skips** — a safety valve for a blank predicting duration 0, which would otherwise spin forever. NeMo's greedy TDT has the identical guard and it changes the output on pathological frames.

Two export limits are permanent: the `decoder_joint` dead `target_length` output cannot be dropped (a TorchScript submodule cannot be called from a plain `nn.Module`), so the Rust side requests `["logits", "h_out", "c_out"]` **by name**; and the decode must use `enc_lengths` rather than the tensor dimension, because `center = true` makes the tensor one column longer than `feature_lengths` ([COMMON_FAILURES §11](../agent/COMMON_FAILURES.md)).

## Alternatives rejected

| Alternative | Why not |
|---|---|
| fp16 everywhere, including the decoder | LSTM state compounds over thousands of sequential emission steps |
| fp32 everywhere | 2.0× disk, ~1.6× time, and the fp32 encoder barely fits 3.3 GB with activations |
| `onnxconverter_common.float16` / the dynamo exporter | a graph ORT rejects; and frozen dynamic axes |
| A general-purpose Rust audio crate for the front end | each computes log-mel under different conventions; see the five findings above |
| The official SraVaani ONNX release (Google Drive) | unversioned blob, no per-file checksums, no pinned opset |
| A different ASR model | 65 Indian languages at ~12% Hindi WER; nothing else in reach matches |

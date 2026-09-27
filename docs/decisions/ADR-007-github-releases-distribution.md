# ADR-007 — GitHub Releases distribution

**Status:** Accepted (updated to include bundled offline models)
**Date:** 2026-09-27 (Updated 2026-09-28)

## Context

VaaniDesk provides a complete, 100% offline desktop voice transcription tool. The distribution must provide an immediate, zero-token, zero-setup experience.

Requirements:
- An install that does not need a Rust toolchain, Python, the CUDA Toolkit, or a Hugging Face token.
- Immediate 100% offline functionality upon extraction or installation.
- A download that is verifiable before it is executed.
- A channel the existing Windows tooling and user base already trust.

## Decision

Publish release artefacts on **GitHub Releases**, containing pre-bundled runtime DLLs and the active production model weights (`models/sravaani/`).

| Artefact | Contents |
|---|---|
| Portable ZIP | `vaani.exe`, `vaani-cli.exe`, `runtime/` (**flat**), `models/sravaani/` (FP16 encoder + joint decoder + tokenizer), licence, `runtime-manifest.json` |
| Installer | Full offline Inno Setup installer packaging app binaries, `runtime/`, and `models/sravaani/` (see ADR-008) |

The ZIP's `runtime/` must be flat: all DLLs in one directory beside the executable (cuDNN 9 sibling loader rule).

Model weights for active production (`encoder.fp16.onnx`, `decoder_joint.onnx`, `tokenizer.model`, etc. ~930 MB) **are bundled directly** in both the ZIP and Installer so users never need tokens or network access.

## Consequences

GitHub Releases provides verified release assets with per-asset SHA-256 digests.
Because the production FP16 model (~930 MB) and runtime (~1.25 GB) are packaged together, the compressed release ZIP / installer is ~1.2–1.4 GB via LZMA2 compression. This enables instant out-of-the-box offline transcription on first launch.

## Alternatives rejected

| Alternative | Why not |
|---|---|
| Microsoft Store / MSIX | sandboxing conflicts with shipping native DLLs and a 1.25 GiB `runtime/` beside the executable |
| S3 + CloudFront | needs extra bucket/domain operational overhead |
| Excluding weights and requiring first-run HF login | Breaks 100% offline zero-setup requirement and forces users to create Hugging Face accounts |

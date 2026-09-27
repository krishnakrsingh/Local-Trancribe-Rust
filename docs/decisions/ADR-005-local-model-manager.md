# ADR-005 — Bundled offline models & local model manager

**Status:** Accepted (superseded to enforce pre-bundled offline models)
**Date:** 2026-09-27 (Updated 2026-09-28)

## Context

VaaniDesk must run a 430M-parameter model (SraVaani 1.0) under the **MIT License**.
The user requirement is **100% offline immediately with zero setup, zero first-launch downloads, and zero Hugging Face tokens or credentials required from the user.**

Constraints & Requirements:
- The user must never be prompted for a Hugging Face token or API key.
- The app must work out-of-the-box in air-gapped, offline, and enterprise environments.
- The active production model assets (`encoder.fp16.onnx`, `decoder_joint.onnx`, `tokenizer.model`, `mel_window.f32`, `mel_fb.f32`, `preproc.json`, `config.json`) totaling ~930 MB must be permanently packaged and locked into the application distribution (Installer and Portable ZIP).
- Source repository remains clean: weights are staged during release packaging and not checked into git.

## Decision

The application and installer **permanently bundle the production model assets** inside the installation directory (`<app>/models/sravaani/` or `<localappdata>/VaaniDesk/models/sravaani/`).

| Element | Choice |
|---|---|
| Model Distribution | **Bundled permanently** in the installer (`vaani.iss`) and portable package (`make-release.ps1`) |
| Access Level | **Open / Pre-installed** (`access: "open"`, no token/gating required) |
| Offline Guarantee | **100% offline immediately** on first launch with zero network calls |
| Storage Location | `<exe>/models/sravaani/` (portable) and `{localappdata}\VaaniDesk\models\sravaani\` (installed) |
| Integrity | Checked on boot via SHA-256 against `registry.json` |

## Consequences

1. **Zero User Setup:** End users do not need an internet connection, a Hugging Face account, or a Hugging Face token to start transcribing immediately.
2. **Package Size:** The installer size increases from ~400 MB to ~1.2–1.4 GB (LZMA2 compressed), and the extracted disk footprint includes the ~930 MB active FP16 model alongside the ~1.25 GB CUDA/cuDNN runtime. This size tradeoff is explicitly accepted to guarantee immediate 100% offline readiness.
3. **No Credential Manager Requirement for Base ASR:** The app does not force credential storage for core transcription, simplifying deployment on restricted Windows images.

## Alternatives rejected

| Alternative | Why not |
|---|---|
| On-demand gated download requiring HF token | Requires user tokens, internet on first run, and fails in offline/air-gapped setups |
| Unpackaged / user-provided model paths only | Fragile user experience; risks missing tokenizer/filterbank files |
| Bundling unneeded 1.77 GB FP32 reference model | Wastes 1.77 GB disk; the production FP16 model (`~930 MB`) provides 99.6% parity and 2.0× memory efficiency |

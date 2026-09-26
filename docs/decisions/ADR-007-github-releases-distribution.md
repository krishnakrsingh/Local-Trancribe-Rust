# ADR-007 — GitHub Releases distribution

**Status:** Accepted (not yet built)
**Date:** 2026-09-27

## Context

VaaniDesk is a ~5 GB product: 1.25 GiB of runtime DLLs, ~2.7 GB of model artefacts that
are downloaded rather than bundled, and a portable executable. The user base is Windows
desktop users who will not build from source and who may or may not have a Hugging Face
token.

Requirements:

- An install that does not need a Rust toolchain, Python, or the CUDA Toolkit.
- A download that is verifiable before it is executed.
- A channel the existing Windows tooling and the existing user base already trust.
- An update path that does not require the user to find a download page.

## Decision

Publish release artefacts on **GitHub Releases**, and make that URL the single source of
truth for version discovery. `vaani-updater` checks it and compares against
`APP_VERSION`.

| Artefact | Contents |
|---|---|
| Portable ZIP | `vaani.exe`, `vaani-cli.exe`, `runtime/` (**flat**), licence, a `runtime-manifest.json` |
| Installer | produced by a separate channel — see ADR-008 |

The ZIP's `runtime/` must be flat: all DLLs in one directory beside the executable. This is
not a preference. cuDNN 9's `cudnn64_9.dll` is a dispatcher that `LoadLibrary`s its siblings
by bare name, and the Windows search order starts at the executable's own directory
(`fetch_runtime.ps1:110-116`, `runtime.rs:200-219`).

Model weights are **not** in the ZIP. SraVaani-1.0 is gated, so the Model Manager fetches
them on the user's machine with their own token (ADR-005).

## Consequences

GitHub Releases is the right channel because it already provides assets, per-asset
SHA-256 in the API `digest` field, immutable tags, and a download counter — and because it
needs no new account relationship and no distribution contract negotiation. It is also
where this project's own pinned downloads already come from: the ONNX Runtime asset is
fetched from `github.com/microsoft/onnxruntime/releases/download/v1.30.0/` and its hash
`8fa4b08359af682cd605892cb59077049700b640128bb93fd2c7776cf9f55bdc` is a **first-party
value from the GitHub releases API**, not one we computed from a file we had already
downloaded (`fetch_runtime.ps1:60-64`). The same discipline must apply to our own
releases.

The 5 GB download is the real problem this ADR does not solve. Portable ZIP plus separate
model download is the right shape, but the updater must be resumable and must not restart
a 1.25 GiB transfer on a flaky connection — the same class of problem as an interrupted
886 MB model download, and for the same reason.

Publishing to Releases is also the only CI-less path that works today, given there is no
CI ([RELEASE.md](../agent/RELEASE.md)). It is a release process with no automation, which
is acceptable at 0.1.0 and will not be at 1.0.0.

Assets must be uploaded with `-cl` (or the equivalent) so `runtime/` is not mangled, and
the ZIP must be built with the same flat layout the script produces, or the first CUDA
inference on a user's machine fails with
`Could not locate cudnn_engines_runtime_compiled64_9.dll`.

## Alternatives rejected

| Alternative | Why not |
|---|---|
| Microsoft Store / MSIX | sandboxing conflicts with shipping native DLLs and a 1.25 GiB `runtime/` beside the executable; a store review cycle per release |
| S3 + CloudFront | needs a bucket, a domain, TLS and a CDN configuration, none of which exist |
| Self-hosted | same operational cost as S3 plus no distribution trust |
| Bundling the model weights in the ZIP | 2.7 GB more, and it redistributes gated artefacts outside the gate |
| GitHub Pages | a hosting site, not an artefact store; no content-addressed assets |

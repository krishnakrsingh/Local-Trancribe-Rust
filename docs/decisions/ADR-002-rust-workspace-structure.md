# ADR-002 — Rust workspace structure

**Status:** Accepted (2 of 13 members built)
**Date:** 2026-09-27

## Context

VaaniDesk spans decode, VAD, ASR, diarization, post-processing, structuring, export,
storage, model management, updating, a UI and a CLI. The failure mode at this size is a
single crate where any module can reach any other, so the only reviewable question becomes
"is this change acceptable". Two facts constrain the layout: the data model must
type-check in seconds with no native runtime and no GUI toolchain, and `unsafe` must be
confined to a directory. It currently lives in three files, all in `vaani-inference`.

## Decision

A Cargo workspace with **one crate per subsystem**, and two rules that make the split
worth having.

**Rule 1 — `vaani-core` has no heavy dependencies.** Its entire dependency set is `serde`,
`serde_json`, `thiserror`, `uuid`, `time` (`crates/vaani-core/Cargo.toml:10-18`). No ONNX
Runtime, no Slint, no database, no async runtime. Every other crate may depend on it; it
depends on nothing local.

**Rule 2 — members are added one at a time** (`Cargo.toml:7-9`). So a `cargo check`
failure always points at a single new crate rather than at a dozen half-written ones.

Members:

| Crate | Responsibility | State |
|---|---|---|
| `vaani-core` | domain model, job lifecycle, error taxonomy | **built** |
| `vaani-inference` | ORT session management, EP selection, machine capability | **built** |
| `vaani-media` | symphonia 0.6 pure-Rust probe/decode to mono 16 kHz f32, ffmpeg-sidecar fallback, Media Foundation via windows-rs for video | planned |
| `vaani-asr` | `AsrEngine` trait, SraVaani TDT adapter, log-mel front end | planned |
| `vaani-vad` | voice activity detection | planned |
| `vaani-diarization` | speaker embedding + clustering | planned |
| `vaani-postprocess` | filler/repetition/punctuation cleanup | planned |
| `vaani-structuring` | optional local LLM, must degrade gracefully | planned |
| `vaani-export` | TXT/MD/SRT/VTT/JSON | planned |
| `vaani-storage` | rusqlite, bundled SQLite | planned |
| `vaani-models` | model manager, manifests, keyring, downloads | planned |
| `vaani-updater` | self-update | planned |
| `vaani-app` | app state, job orchestration, undo | planned |
| `vaani-cli` | `vaani.exe` (GUI, `windows_subsystem = "windows"`) + `vaani-cli.exe` | planned |
| `vaani-ui` | Slint markup and presentation | planned |

Enforcement: `unsafe_op_in_unsafe_fn = "deny"` and a clippy set in `[workspace.lints]`
(`Cargo.toml:84-102`); `vaani-core` adds `#![forbid(unsafe_code)] #![warn(missing_docs)]`
(`crates/vaani-core/src/lib.rs:34-35`).

## Consequences

Thirteen crates means thirteen `Cargo.toml` files and a graph to hold in your head. The
payoff is that a change to the transcript shape touches one crate and compiles in
seconds, and that "which crate is allowed to do `unsafe`" has a one-word answer.

`[workspace.dependencies]` is the single registry for third-party versions
(`Cargo.toml:11-12`). A version is declared once, with a note. Only crates that are actual
members appear in `Cargo.lock`, so the ~30 declared-but-unused pins in
`Cargo.toml:28-82` are statements of intent, not resolved versions — do not read
`Cargo.lock` as evidence that, say, Slint is installed.

`vaani-inference` must stay ignorant of ASR. No mention of Conformer, TDT, mel or
vocabulary appears anywhere in it; it can load a sentence-embedding model without knowing
what a sentence is. The moment it needs a model name, the split has failed.

Splitting media decode out of ASR is what makes the ffmpeg fallback possible without
touching the inference path, and splitting structuring out is what makes "optional local
LLM, degrade gracefully" implementable at all.

## Alternatives rejected

| Alternative | Why not |
|---|---|
| One crate with modules | modules are not a boundary; a `pub(crate)` visibility change can cross any module line silently |
| A core crate plus one big `vaani-app` | the app would accumulate decode, ASR and storage; this is the layout that is being rejected, in miniature |
| Build each subsystem as a separate repository | 13 repos means 13 lockfiles, 13 CI configs and 13 versions of the domain model |
| Feature flags instead of crates | a feature is off by default in someone else's build; a crate is off by construction |

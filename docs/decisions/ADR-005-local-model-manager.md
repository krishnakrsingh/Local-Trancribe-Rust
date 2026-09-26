# ADR-005 — Local model manager

**Status:** Accepted (not yet built)
**Date:** 2026-09-27

## Context

VaaniDesk must run a 430M-parameter model that is MIT-licensed but **gated on Hugging
Face** and therefore requires a user token. The exported artefacts are ~2.7 GB. The
runtime itself is ~1.25 GiB of DLLs. Weights are never committed (`.gitignore:1-16`).

Constraints:

- A fresh clone must be small enough to clone. 908 MB of TorchScript plus 2.66 GB of
  artefacts is not a git repository.
- A user must be able to see, verify, repair and remove what is on their disk.
- A token is a credential. It must not appear in a log, a command line, or a crash dump.
- The app must work offline once a model is present, and must say so honestly when a model
  is missing rather than failing at load.

## Decision

A `vaani-models` crate owns a local model registry. Models are **fetched on the user's
machine, verified by SHA-256, and described by a manifest that records provenance**.

| Element | Choice |
|---|---|
| Storage | `keyring` 4 (`features = ["v1"]`, `Cargo.toml:72`) → **Windows Credential Manager**, with a **DPAPI-sealed file** fallback |
| Downloads | `hf-hub` 1.0 — **stated intent; not in `Cargo.toml`, not in `Cargo.lock`** |
| Manifests | `models/registry/*.json` and `models/manifests/*.json`, the only files under `models/` that are not gitignored (`.gitignore:13-14`) |
| Integrity | per-file SHA-256; mismatch is `ErrorCode::ModelCorrupt`, not a load failure |
| Offline | a present-and-verified model is used with no network call at all |

Required manifest fields: `id`, `version`, `provider`, `architecture`, `size`, `sha256`,
`license`, `access` (`open` / `gated-token` / `request-access`), `local_path`,
`download_url`, `runtime_requirements`, `gpu_requirements`. See
[MODELS.md](../agent/MODELS.md#the-model-manifest).

Token handling: the credential goes to Windows Credential Manager. It is never logged, never
an argument, never in an environment variable the app re-exports, never in a crash dump.
`.gitignore:77-80` ignores `*.token`, `*token*.json`, `.env`, `.env.*` as a backstop, not
as the mechanism.

## Consequences

Shipping an installer that contains no weights is the whole point. It also means the
first run is not fully functional, and the app must be honest about that rather than
degrading silently — which is why `ErrorCode::ModelUnavailable` and `ErrorCode::AuthRequired`
are separate codes and separate reviewed messages
(`crates/vaani-core/src/error.rs:310-337`), asserted in tests rather than inferred from
prose. A valid token that lacks repository access is `AccessNotGranted`, again a distinct
code: the user's next action is different in each case.

Gating forces the token question to be answered up front, which is a gift. A model that
were ungated would let a token get smuggled into a log line during a download
regression; a gated model makes that visible in the first hour of testing.

`keyring` 4.2 is a real choice, not a default. Its predecessors leaked credentials on
some Windows configurations, and the DPAPI fallback exists for locked-down enterprise
images where the Credential Manager API is unavailable.

**Nothing here is built.** `vaani-models` does not exist. `hf-hub` is not in
`[workspace.dependencies]` at all, so the "1.0" pin is a plan, not a resolved version.
The manifest schema above is a specification, not an implementation.

Download resumption and partial-file handling must be designed before the first release:
an interrupted 886 MB download is the most likely thing to go wrong on a real user's
connection, and `ErrorCode::ModelCorrupt` is documented as covering exactly that
("*It was most likely interrupted while downloading*", `error.rs:316-319`).

## Alternatives rejected

| Alternative | Why not |
|---|---|
| Bundle the weights in the installer | 2.7 GB installer, and it redistributes gated artefacts outside the gate |
| Let the user point at a model directory | no verification, no provenance, and no way to state the model in `EngineStamp` |
| Hugging Face's own CLI at runtime | adds Python to the shipped product's install footprint |
| `reqwest` + a hand-rolled downloader | we need gated-repo auth, resumable transfers and a cache; `hf-hub` already does all three |
| Windows Data Protection API only | DPAPI is the fallback, not the primary; Credential Manager is the platform's actual credential store |

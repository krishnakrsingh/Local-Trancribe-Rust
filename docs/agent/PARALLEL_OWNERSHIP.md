# Parallel ownership contract

Eight subsystems are being implemented concurrently by different agents. This
file is the *only* authority on who may edit what. It exists because a merge
conflict is cheap to prevent and expensive to diagnose, and because an agent
that cannot see the boundary will eventually step over it.

Read this before editing anything. If your task needs a file you do not own,
stop and report it under `OPEN ISSUES` instead of editing it.

## The rule

**Owners edit only their own paths. Nobody edits `Cargo.toml`, `Cargo.lock`, or
`crates/vaani-core/**` except the architect.**

`crates/vaani-core` is the contract crate. It holds every cross-boundary type
(`Transcript`, `Segment`, `Word`, `Speaker`, `JobPhase`, `Progress`,
`CancelToken`, `AudioBuffer`, `MediaInfo`, `SpeechTimeline`, `SpeakerSpan`,
`DiarizationResult`, the stage traits). It has **no** third-party dependencies,
which is what lets it be type-checked in seconds and lets every agent compile
against real signatures instead of guesses.

If a contract is wrong, the architect changes it, in one commit, and every
agent rebases. That is the intended flow — not eight agents each inventing a
private definition of "a buffer of audio".

## Ownership map

| Agent | Owns | Must not touch |
|---|---|---|
| **A1 media** | `crates/vaani-media/**` | anything else |
| **A3 vad** | `crates/vaani-vad/**` | anything else |
| **A4 diarization** | `crates/vaani-diarization/**` | anything else — **especially `crates/vaani-asr/**`** |
| **A5 ui** | `crates/vaani-ui/**`, `crates/vaani-ui/ui/**` (slint files) | anything else |
| **A6 storage+export** | `crates/vaani-storage/**`, `crates/vaani-export/**` | anything else |
| **A7 model manager** | `crates/vaani-models/**`, `models/**` | `crates/vaani-inference/**`, `crates/vaani-asr/**` |
| **A8 cli+dist** | `crates/vaani-cli/**`, `.github/**`, `installer/**`, `scripts/release/**` | the five existing files in `scripts/` (frozen, below) |
| **A10 qa** | `tests/**` | anything under `crates/` |
| **A2 app** | `crates/vaani-app/**` | anything else — especially frozen `vaani-asr` math |
| **A9 bench** | `crates/vaani-bench/**`, `scripts/bench/**` | product code (measure, don't rewrite) |
| **Architect** | `crates/vaani-core/**`, `Cargo.toml`, `Cargo.lock`, `docs/**`, `scripts/{build_doctor,fetch_runtime,export_sravaani_onnx,make_golden}.*`, `scripts/cudnn.sha256`, `.gitignore`, `rust-toolchain.toml` | — |

Wave 2 is now active (2026-09-27): **A2 job pipeline** (`crates/vaani-app/**`)
and **A9 performance lab** (`crates/vaani-bench/**`) are spawned against the
real module interfaces in `docs/architecture/INTEGRATION_CONTRACT.md`, which is
now the authoritative contract. Wave-1 crates remain owned as above; the
integration hub rule still holds — services never call each other, phases
advance only in `vaani-app`.

## Naming: the brief said `crates/media/`, the repo says `crates/vaani-media/`

The brief listed ownership as `crates/media/`, `crates/core/`, `crates/ui/`,
`crates/inference/` and so on. The repository already used a `vaani-` prefix
for the three crates that exist and are validated. Both cannot be true, and
adopting the brief's literal paths would create `crates/core/` alongside
`crates/vaani-core/` — two crates both claiming to be the domain model, which
is exactly the ambiguity Phase B is supposed to catch.

**Decision: the `vaani-` prefix wins, everywhere, for consistency.** The mapping
is:

| Brief | Actual |
|---|---|
| `crates/core/` | `crates/vaani-core/` (exists, validated) |
| `crates/inference/` | `crates/vaani-inference/` (exists, validated) |
| `crates/media/` | `crates/vaani-media/` |
| `crates/vad/` | `crates/vaani-vad/` |
| `crates/diarization/` | `crates/vaani-diarization/` |
| `crates/ui/` | `crates/vaani-ui/` |
| `crates/storage/` | `crates/vaani-storage/` |
| `crates/export/` | `crates/vaani-export/` |
| `crates/model_manager/` | `crates/vaani-models/` |
| `crates/app/` | `crates/vaani-app/` (wave 2) |
| `crates/cli/` | `crates/vaani-cli/` |

## Frozen: the validated ASR pipeline

These are verified against the model's own reference implementation and must
survive integration **byte for byte**:

- `crates/vaani-asr/src/frontend.rs` — the log-mel front end
- `crates/vaani-asr/src/sravaani.rs` — the TDT decode loop
- `crates/vaani-inference/src/session.rs` — the typed ONNX session wrapper
- `crates/vaani-inference/src/runtime.rs` — runtime discovery and `SetDllDirectoryW`
- `tests/fixtures/sravaani/decode_golden.json` — 202 reference token ids
- `tests/fixtures/sravaani/frontend_golden.json` — 12,928 reference features
- `crates/vaani-asr/tests/` — the parity tests

The invariants they encode, each of which cost a real bug to find:

1. `torch.stft` **centres** the analysis window: a 400-sample window sits at
   `[56, 456)` in a 512 frame. Left-aligning shifts the spectrogram 56 samples
   and degrades every transcript silently.
2. For a real input `Re(X[n-k]) == Re(X[k])`, so the power spectrum must read
   `Im` from bin `k`. Reading it from the mirrored bin computes `re² + re²`.
3. The encoder's valid length is the `enc_lengths` output, **not** the tensor
   dimension: the log-mel carries one extra STFT frame, so the tensor-derived
   figure decodes one frame past the end of the audio.
4. Bessel correction (`N-1`), a symmetric (not periodic) Hann window, and
   `log_zero_guard_value == 2^-24` exactly.

If integration changes the transcript, the front end, or the decode, **stop**.
That is a bug, not a refactor.

## Frozen scripts

`scripts/build_doctor.ps1`, `scripts/fetch_runtime.ps1`,
`scripts/export_sravaani_onnx.py`, `scripts/make_golden.py` and
`scripts/cudnn.sha256` exist and work. New scripts go under
`scripts/<agent>/`.

## Adding a dependency

`Cargo.toml`'s `[workspace.dependencies]` is architect-owned, so a shared
version is added once. If you need a crate that is not there, add it to **your
own** `Cargo.toml` with an explicit version and list it under `INTEGRATION
NOTES`. Do not edit the workspace manifest.

Do not run `cargo update`, and do not commit a regenerated `Cargo.lock` from a
feature branch — the architect regenerates it once at merge time, so eight
independent lockfile edits do not become eight conflicts.

## Verify your own work, not the workspace

`cargo test --workspace` will fail while other agents are mid-edit, and its
output is not yours to fix. Verify with:

```
cargo test -p <your-crate>
cargo clippy -p <your-crate> --all-targets -- -D warnings
cargo fmt -p <your-crate>
```

Set `CARGO_TARGET_DIR` to your own directory (you were told yours in your task)
so eight agents are not serialised on one cargo build lock.

## Never commit

Model weights, the ONNX Runtime, the CUDA redistributables, and the private
meeting recording. `.gitignore` covers `artifacts/`, `runtime/`, `scratch/`,
`SraVaani-1.0/` and every media extension. If something large appears in
`git status`, do not stage it.

## Reporting

Every agent ends with the eight-section report in its task brief. Machine
readable headings, no prose padding. The architect gates on it: a report that
claims a gate passed is re-verified by running the gate.

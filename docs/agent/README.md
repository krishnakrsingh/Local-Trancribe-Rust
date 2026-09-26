# VaaniDesk — agent documentation index

Audience: a coding agent about to change this repository. Not an end user.

Every document here answers five questions about its topic: **what exists**, **why it
exists**, **what must not change**, **how to test it**, **what it assumes**. Sections
that could be pasted into any Rust project have been deleted.

## State of the repository

Verified 2026-09-27 against `C:\Users\krish\Downloads\localvoicetool` by running the suite
in release.

| | |
|---|---|
| Workspace members | `crates/vaani-core`, `crates/vaani-inference`, `crates/vaani-asr` (3 of a planned 13) |
| Rust | 1.98.1, `x86_64-pc-windows-msvc`, pinned in `rust-toolchain.toml` |
| Tests | **111 passing** — 41 `vaani-core` + 31 `vaani-inference` unit + 2 `vaani-inference` integration + 28 `vaani-asr` unit + 3 `frontend_parity` + 6 `sravaani_transcribe` |
| Lints | `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --all --check` are both clean |
| Source lines | `vaani-core` ~2000, `vaani-inference` ~2150, `vaani-asr` ~1855 |
| `unsafe` | 3 files, all in `vaani-inference` (`capability.rs`, `runtime.rs`, `session.rs`) |
| Native runtime | present and working in `runtime/` (19 DLLs, 1279.2 MB) |
| Model artefacts | present and working in `artifacts/sravaani/` (10 files, 2.70 GB) |
| Binaries | **none**. No CLI, no GUI, no installer. **There is nothing to run.** |

## What exists, and what does not

**Exists and is verified end to end: the ASR path.** `crates/vaani-asr` computes the
log-mel front end, runs the fp16 encoder and the fp32 TDT decode loop on the GPU, groups
SentencePiece pieces into words, and produces a transcript with word-level times. It is
verified three ways, not asserted:

| Claim | Verification | Result |
|---|---|---|
| The front end is a faithful port | every one of 128 × 101 cells against PyTorch, tolerance 2e-4 | max \|d\| = **7.99e-05** |
| The decode reproduces the reference | token ids and word timings from the model's own TorchScript decode | **202/202** ids, **108/108** words within **0.0 ms**, exact text |
| It is fast enough to be interesting | `cargo test --release`, 40 s chunk | front end 203 ms, encoder 1498 ms, decode 299 ms, ASR RTF **0.032–0.051** over three runs |

**Does not exist. None of it is stubbed, and nothing in these documents should imply
otherwise:**

| Missing | Consequence |
|---|---|
| media ingestion (`vaani-media`) | nothing can open a file. The ASR path is fed samples by a test |
| VAD (`vaani-vad`) | every chunk is transcribed in full |
| diarization (`vaani-diarization`) | `Speaker` exists in the data model and nothing populates it |
| post-processing and structuring | raw model text is the only output |
| storage (`vaani-storage`) and export (`vaani-export`) | nothing persists; nothing writes a transcript out |
| the model manager (`vaani-models`) | artefacts are read from the repo checkout; `VAANI_ARTEFACTS` is the only override |
| the updater (`vaani-updater`) | — |
| the UI (`vaani-ui`), the app state machine (`vaani-app`), the CLI (`vaani-cli`) | **there is no application.** No `vaani.exe`, no `vaani doctor`, no `vaani --debug` |

The product-level targets in the requirements — under 1.5 s startup, faster than realtime —
have **not** been validated, because the pipeline is not wired together. See
[PERFORMANCE.md](PERFORMANCE.md) for what has and has not been measured.

## Document map

| Document | Read it when you need to |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | know which crate owns what, and what the dependency direction is |
| [MODELS.md](MODELS.md) | touch model weights, manifests, checksums, or the HF token |
| [BUILD.md](BUILD.md) | set up a machine from scratch, or change the toolchain or pins |
| [DEBUGGING.md](DEBUGGING.md) | something is slow, on the CPU, or failing at load |
| [PERFORMANCE.md](PERFORMANCE.md) | quote a speed number, or change anything in the hot path |
| [COMMON_FAILURES.md](COMMON_FAILURES.md) | you hit an error string that looks like the code's fault and is not |
| [RELEASE.md](RELEASE.md) | cut a build, sign it, or publish it |

## Decisions

Architecture decisions with their reasons. If you are about to reverse one, read the
ADR first and then change the ADR — do not just change the code.

| ADR | Decision | Status |
|---|---|---|
| [ADR-001](../decisions/ADR-001-native-ui-with-slint.md) | Native UI with Slint 1.18.1 | Accepted (planned) |
| [ADR-002](../decisions/ADR-002-rust-workspace-structure.md) | Rust workspace with one crate per subsystem | Accepted (partially built) |
| [ADR-003](../decisions/ADR-003-onnx-runtime.md) | ONNX Runtime 1.30.0 via `ort` 2.0.0-rc.13, `load-dynamic` | Accepted (built) |
| [ADR-004](../decisions/ADR-004-sravaani-integration.md) | SraVaani 1.0: native-fp16 encoder, fp32 decoder | Accepted (built) |
| [ADR-005](../decisions/ADR-005-local-model-manager.md) | Models fetched locally, never committed | Accepted (planned) |
| [ADR-006](../decisions/ADR-006-speaker-diarization.md) | Two-model diarization, pyannote + TitaNet | Accepted (planned) |
| [ADR-007](../decisions/ADR-007-github-releases-distribution.md) | GitHub Releases as the update channel | Accepted (planned) |
| [ADR-008](../decisions/ADR-008-winget-distribution.md) | WinGet as the install channel | Accepted (planned) |

## The five invariants that outrank local convenience

These are the rules a reviewer will hold you to. Each is enforced by a test today, not by
convention.

1. **Chunk-local times become media times in exactly one function.**
   `Transcript::push_chunk` at `crates/vaani-core/src/transcript.rs:351`. An `AsrEngine`
   returns chunk-local times and knows nothing of its offset in the media
   (`AudioChunk::source_offset` is diagnostic only). No other crate may add a chunk offset
   to a `Time`. If you add a second path, the transcript drifts from the video silently
   and no test fails.

2. **`UserFacingError::detail` is a reviewed constant, never a formatted diagnostic.**
   Every constructor in `vaani_core::err` (`crates/vaani-core/src/error.rs:281`) takes
   **zero arguments**. Diagnostics go to `VaaniError::with_cause`, which is
   `#[source]`-chained and only ever reaches a developer log. The constructors' arity is
   the guard; see `error.rs:433`.

3. **Progress has a denominator or it is `Indeterminate`.** Never a fabricated
   percentage. `JobPhase::weight()` must sum to 1.0 across active phases
   (`job.rs:399`) or the overall bar can never reach 100%.

4. **`ensure_initialised()` is the only thing allowed to touch `ort`.** It is a
   process-once `OnceLock` at `crates/vaani-inference/src/runtime.rs:209`, and it must
   also call `SetDllDirectoryW` before anything else, because cuDNN 9 is a split library
   that resolves its siblings lazily during the first inference run. Reordering it is how
   a "works on my machine" build gets shipped.

5. **A GPU claim requires `inference_adapter()`, never `preferred_adapter()`.**
   On this exact development machine `preferred_adapter()` returns the AMD integrated
   GPU, which CUDA cannot use. See
   [COMMON_FAILURES.md §4](COMMON_FAILURES.md) and
   [`hybrid_laptop_picks_nvidia_not_the_shared_memory_igpu`](../../crates/vaani-inference/src/capability.rs#L619).

## Ground rules for editing this repository

| Rule | Why |
|---|---|
| Add workspace members **one at a time** | `Cargo.toml:6-10` — so a `cargo check` failure points at one new crate. See [ADR-002](../decisions/ADR-002-rust-workspace-structure.md) |
| Declare a third-party version **once**, in `[workspace.dependencies]` | `Cargo.toml:14` — a version in two manifests is a version that will drift |
| No `unsafe` outside `vaani-inference` | every other crate is `#![forbid(unsafe_code)]`; `vaani-core` sets it at `lib.rs:34` |
| Never commit weights, runtime DLLs, or media | `.gitignore:9-51`; see [MODELS.md](MODELS.md) |
| Never commit the audio a golden fixture came from | the fixture is token ids and timings; the WAV stays in `scratch/` |
| Never put the HF token in a log, a command line, or a crash dump | see [MODELS.md §Token handling](MODELS.md#token-handling) |
| Assert on `ErrorCode`, never on message prose | `error.rs:74-75` — rewording a message must not break a test |
| Do not "improve" a front-end constant | every value comes from `preproc.json`; the deviations are documented in [COMMON_FAILURES §9-10](COMMON_FAILURES.md) |

## Commands you will actually run

```powershell
# cargo is NOT on PATH after a `--no-modify-path` rustup install
$env:Path += ";$env:USERPROFILE\.cargo\bin"

# The whole gate, cheapest stage first. This is the one to run first.
powershell -ExecutionPolicy Bypass -File scripts\build_doctor.ps1
powershell -ExecutionPolicy Bypass -File scripts\build_doctor.ps1 -Release   # release profile

cargo test --release --workspace                       # 111 tests
cargo clippy --workspace --all-targets -- -D warnings  # must be clean
cargo fmt --all --check                                # must be clean

# Diagnostics
cargo run -p vaani-inference --example conv_probe        # fp16 vs fp32 on the GPU
cargo run -p vaani-inference --example subsample_probe   # the ceil(T/8) law, measured
cargo test -p vaani-inference --test runtime_smoke -- --nocapture   # hardware + timings

# Parity. VAANI_GOLDEN_RAW must be an ABSOLUTE path to a raw f32 clip.
cargo test -p vaani-asr --test frontend_parity -- --nocapture
$env:VAANI_GOLDEN_RAW = "C:\...\scratch\golden_clip.f32"
cargo test -p vaani-asr --test sravaani_transcribe -- --nocapture --test-threads=1
```

`build_doctor.ps1` runs all of the above except the two examples, and reports a missing
prerequisite as `SKIP` with the command that would install it. A green `cargo test` on a
machine without `runtime/` or `artifacts/` proves nothing — read the printed reason.

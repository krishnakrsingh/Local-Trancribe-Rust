# Architecture

## What exists

Two library crates. `Cargo.toml:3-6` lists them; nothing else is a member yet.

```
Cargo.toml                        workspace root: [workspace.dependencies] is the single version registry
rust-toolchain.toml               Rust 1.98.1, x86_64-pc-windows-msvc, profile=minimal
crates/vaani-core/                domain model. zero heavy dependencies. no unsafe
  src/error.rs                    UserFacingError, ErrorCode (19), VaaniError, err module
  src/transcript.rs               Time, Word, Segment, Speaker, Transcript, TranscriptMode
  src/job.rs                      JobPhase (13), Progress, CancelToken
  src/media.rs                    Language, MediaInfo, AudioStreamInfo, VideoStreamInfo
  src/lib.rs                      re-exports, APP_VERSION, DATA_MODEL_VERSION, extension registries
crates/vaani-inference/           the ONNX Runtime boundary. the only crate with unsafe
  src/capability.rs               DXGI adapter enumeration, cudart probe, CPU/RAM
  src/runtime.rs                  RuntimeLocation::discover, ensure_initialised, choose_provider
  src/session.rs                  OnnxSession, F32/F16/I32/I64Tensor, Input, OutputTensor
  src/lib.rs                      RuntimeReport
  examples/conv_probe.rs          provider-option regression probe
  tests/runtime_smoke.rs          2 integration tests, one runs the real GPU encoder
```

Module count: 8 source files, 2 Cargo.toml, 1 example, 1 integration test.

## Why it exists

`vaani-core` exists so that the data model can be type-checked in seconds and so that it
cannot acquire a dependency on a subsystem it is supposed to describe. Its five
dependencies are `serde`, `serde_json`, `thiserror`, `uuid`, `time`
(`crates/vaani-core/Cargo.toml:10-18`) and nothing else — no ONNX Runtime, no Slint, no
database, no async runtime. Every other crate may depend on it; it depends on nothing
local.

`vaani-inference` exists to hold the *only* `unsafe` and the only third-party runtime
coupling, so the FFI blast radius is a directory rather than the program. It also exists
to answer one question honestly: **is GPU acceleration actually going to work, and on
what hardware** (`crates/vaani-inference/src/lib.rs:6-14`).

## Dependency direction

```
  vaani-ui         Slint markup + presentation   (planned)
      |  reads state, sends intents
      v
  vaani-app        app state machine, job orchestration, undo   (planned)
      |
      v
  services         vaani-asr  vaani-vad  vaani-diarization
                   vaani-postprocess  vaani-structuring  vaani-export
                   vaani-media  vaani-models  vaani-updater       (all planned)
      |
      +-----------+-----------+
      v                       v
  vaani-inference         vaani-storage        (built / planned)
  (ort, windows)          (rusqlite)
      |                       |
      +-----------+-----------+
                  v
              vaani-core            <-- the only crate everything agrees on
```

Rules that follow from the direction, not from preference:

- **`vaani-core` never depends upward.** It has no `#[cfg(feature)]` escape hatch and no
  dev-dependency on a heavier crate that would let a test import one.
- **`vaani-inference` knows nothing about ASR.** No mention of Conformer, TDT, mel,
  tokeniser or vocabulary appears anywhere in the crate. It can load a sentence
  embedding model without knowing what a sentence is
  (`crates/vaani-inference/src/lib.rs:26-28`).
- **Presentation lives only in Slint.** No drawing, no layout maths, no HTML, and no
  second GUI toolkit in `vaani-app` or `vaani-ui`. State that needs to be tested is Rust;
  state that only the UI needs is a Slint property.
- **`vaani-app` owns the job lifecycle.** No service may call another service. Phases
  advance in `vaani-app`; services are handed work and return results.

## Module boundary table

| Module | Owns | Must not know about | Enforced by |
|---|---|---|---|
| `core::error` | `UserFacingError`, `ErrorCode`, `VaaniError`, `err::*` | anything — it is the leaf | zero-argument constructors, `error.rs:433` |
| `core::transcript` | `Time`, `Word`, `Segment`, `Speaker`, `Transcript`, `TranscriptMode` | ONNX, Slint, SQL | `#![forbid(unsafe_code)]`, no deps |
| `core::job` | `JobPhase`, `Progress`, `CancelToken` | how any phase is implemented | weight-sum test `job.rs:399` |
| `core::media` | `MediaInfo`, `Language`, `DEFAULT_ESTIMATE_RTF` | which backend produced the info | no backend crates as deps |
| `inference::capability` | DXGI adapters, cudart probe, CPU/RAM, `inference_adapter()` vs `preferred_adapter()` | model names, session policy | `capability.rs:602` |
| `inference::runtime` | locating + one-time init of `onnxruntime.dll`, provider decision | what a model is | `choose_provider` is pure, `runtime.rs:332` |
| `inference::session` | typed wrapper over one `ort::session::Session` | model semantics, f16 policy rationale is *commented* here but the type is generic | `tensor_to_output` refuses a dtype mismatch rather than reinterpreting |
| `scripts/export_sravaani_onnx.py` | TorchScript → ONNX, `preproc.pt` → flat `.f32` | Rust, the UI, downloads | asserts `enc_lengths == ceil(T/8)` and >= 99% argmax |
| `scripts/fetch_runtime.ps1` | pinned download + SHA-256 verification into `runtime/` | Rust | `Assert-Hash` throws on mismatch or on a missing pin |

## What must not change

| Do not | Why | Where the rule lives |
|---|---|---|
| Move chunk-time rebasing out of `Transcript::push_chunk` | it is the only thing standing between chunk-local and media-local time; a second path drifts silently | `transcript.rs:315-350` |
| Add a parameter to any `err::*` constructor | arity is the structural guard that stops a diagnostic reaching the UI | `error.rs:262-280`, test at `error.rs:433` |
| Store `Segment` text as a field | an edit to a word would leave a stale duplicate disagreeing with it; `text()` is derived from `words` | `transcript.rs:194-206` |
| Change an `ErrorCode::as_str()` or a `JobPhase::as_str()` | both are persisted in the database and in exported JSON; a rename breaks old rows | `error.rs:475`, `job.rs:354` |
| Change `JobPhase::weight()` without re-checking the sum | the overall bar would never reach 100% | `job.rs:399` || Put `cuda_unavailable` on the *error* path | a silent CPU fallback is a normal outcome; it is a `ProviderStatus`, not a `VaaniError` | `lib.rs:30-37` |
| Use `preferred_adapter()` to decide anything about inference | it returns the AMD iGPU on this machine | `capability.rs:146` vs `:164` |
| Create a CUDA context at startup | costs hundreds of MB of VRAM and hundreds of ms; a user browsing history must not pay | `lib.rs:17-22` |
| Make `vaani-core` depend on anything new without writing a reason | the crate's entire value is its dependency emptiness | `lib.rs:3-10` |
| Feed the decoder the fp16 boundary tensor | the joint's LSTM state is threaded through every emission step; fp16 compounds over minutes | [ADR-004](../decisions/ADR-004-sravaani-integration.md) |

## How to test it

```powershell
$env:Path += ";$env:USERPROFILE\.cargo\bin"
cargo test --workspace          # 41 + 31 + 2, ~14 s (the integration test loads an 887 MB model)
cargo clippy --workspace --all-targets
cargo test -p vaani-inference --test runtime_smoke -- --nocapture   # prints adapters, cuda, threads, RTF
```

| Layer | What it proves | Command |
|---|---|---|
| `vaani-core` unit | the five invariants, plus `Time` formatting, phase round-trips, JSON round-trip | `cargo test -p vaani-core` |
| `vaani-inference` unit | `CudaDevicePropPrefix` byte offsets, provider precedence, tensor shape validation, f16 round-trip | `cargo test -p vaani-inference --lib` |
| `vaani-inference` integration | this machine can actually load the real encoder and it produces `enc_lengths == 200` for T=1600 | `cargo test -p vaani-inference --test runtime_smoke` |
| exporter self-checks | `enc_lengths == ceil(T/8)`, `d_model == 1024`, output finite, fp16 argmax agreement >= 99% | `python scripts/export_sravaani_onnx.py` |

There are no tests that skip silently. `runtime_smoke.rs` returns early with a printed
reason (`tests/runtime_smoke.rs:49,65,70`) rather than passing, and
`capability.rs:766` prints `skipping: no usable CUDA device` and returns.

## What it assumes

- **Windows 11, x64, MSVC.** `windows-rs` 0.62 and `SetDllDirectoryW` are load-bearing.
  There is no Linux or macOS path and no abstraction for one.
- **A WDDM display stack.** `capability.rs:240-246` reads `IDXGIAdapter3::Budget` rather
  than `DedicatedVideoMemory`, because the budget is what a process can actually
  allocate. This is correct on WDDM and is *not* the same number as board size — on this
  machine the NVIDIA card reports 3.3 GB of a 4 GB board.
- **`cudart64_13.dll` is loadable at probe time.** `probe_cuda` is passed
  `runtime::location().provider_dirs`; if the DLL is absent the probe reports
  `error = Some(...)` and `inference_adapter()` returns `None`. That is the correct
  answer, not a failure.
- **The runtime is process-global.** `ensure_initialised` can only run once; the `OnceLock`
  turns a second call into a no-op that returns the first result
  (`runtime.rs:189-238`).
- **Slint is the only UI toolkit and `rfd` is its only file picker.** Both are
  [planned](ARCHITECTURE.md) and neither is in `Cargo.toml` yet.
- **Model weights live outside the repository** and are checksummed by
  `artefacts.json`; see [MODELS.md](MODELS.md).

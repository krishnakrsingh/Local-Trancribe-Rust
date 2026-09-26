# Architecture

## What exists

Three library crates. `Cargo.toml:3-7` lists them; nothing else is a member yet.

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
  examples/subsample_probe.rs     asks the graph for its output length across 12 input lengths
  tests/runtime_smoke.rs          2 integration tests, one runs the real GPU encoder
crates/vaani-asr/                 the ASR path: front end, engine trait, SraVaani adapter
  src/frontend.rs                 waveform -> log-mel, a faithful port of processing_sravaani.py
  src/engine.rs                   AsrEngine trait, EngineRegistry, AudioChunk, EngineArtifacts
  src/sravaani.rs                 the two graphs, the greedy TDT decode loop, word grouping
  src/lib.rs                      module map, the two contracts, re-exports
  examples/dump_logmel.rs         writes the log-mel as raw f32 for numpy diffing
  tests/frontend_parity.rs        128x101 cell-by-cell parity against the PyTorch reference
  tests/sravaani_transcribe.rs    end-to-end decode; 6 tests, the strict one gated on a local clip
```

Module count: 12 source files, 3 Cargo.toml, 3 examples, 3 integration tests. Line counts:
`vaani-core` ~2000, `vaani-inference` ~2150, `vaani-asr` ~1855 of library source plus 39 in
its example and 621 in its two integration tests.

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
what hardware** (`crates/vaani-inference/src/lib.rs:6-13`).

`vaani-asr` exists so that the model is behind an interface and the front end is behind a
port. `AsrEngine` (`src/engine.rs:123`) has exactly the methods the spec names and no more,
because a capability only one engine has belongs on that engine, not on the shared trait.
`frontend.rs` is a hand-written port of `SraVaani-1.0/processing_sravaani.py` rather than a
library call, because every general-purpose audio crate computes a log-mel spectrogram under
different conventions. It is the only crate that knows what a SentencePiece piece is, and it
depends on `vaani-inference` for tensors and sessions, never the other way round.

## The two ASR contracts `vaani-asr` enforces

Both are stated in the crate's own module docs (`src/lib.rs:15-28`) and both are load-bearing.

**1. Engines receive a chunk and return chunk-local times.** `AsrEngine::transcribe` takes
an `AudioChunk` (`samples`, `sample_rate`, `source_offset`, `language`) and returns a
`ChunkResult` whose segment times are relative to the start of that chunk. The engine is
never given a file path and never learns where the chunk sat in the source media;
`AudioChunk::source_offset` is documented as diagnostic only. Rebasing happens in exactly
one function, `vaani_core::Transcript::push_chunk`
(`crates/vaani-core/src/transcript.rs:351`). This is what makes chunking an implementation
detail of the pipeline rather than something every engine has to get right — and a second
rebasing path would drift the transcript from the video silently, with no test failing.

**2. The front end is a faithful port, not an improvement.** Every constant comes from the
model's own `artifacts/sravaani/preproc.json`, which the exporter derives from `preproc.pt`;
nothing is hard-coded twice. Four conventions look wrong and are not, and each one costs
accuracy if it is "corrected":

| Convention | Value | Why it looks wrong |
|---|---|---|
| Bessel correction | variance divides by `N − 1`, not `N` | `N` is the textbook form; the bias scales with clip length |
| Symmetric analysis window | `torch.hann_window(400, periodic=False)`, zero-padded to 512 | a periodic window is the STFT default and is wrong by up to 6e-3 |
| Window placement | `window_offset = (n_fft − win_length) / 2 = 56` | left-aligning it looks equivalent and is not; see [COMMON_FAILURES §9](COMMON_FAILURES.md) |
| `log_zero_guard_value` | exactly `2⁻²⁴` | NeMo's usual `1e-5` is "obviously" the right value |

Plus two structural ones: `center = true` means the tensor has `1 + valid_frames` frames
and the last one is padding; and the `N == 1` variance is `0/0`, whose `NaN` is replaced with
`0` before the `+ 1e-5` is added to the standard deviation. The port is verified cell by
cell against PyTorch: max |rust − reference| = 7.99e-05 across all 128 × 101 cells
(`tests/frontend_parity.rs`). See [MODELS.md](MODELS.md#the-log-mel-front-end-in-vaani-asr)
for the exact step order.

## Crates that do not exist

Of the planned members, three exist: `vaani-core`, `vaani-inference`, `vaani-asr`. The rest
are declared as path dependencies in `[workspace.dependencies]` (`Cargo.toml:16-27`) and
appear in [ADR-002](../decisions/ADR-002-rust-workspace-structure.md), and that is all —
no source file references them:

| Crate | What it will own | Consequence of its absence |
|---|---|---|
| `vaani-media` | probe/decode to mono 16 kHz f32, ffmpeg fallback | there is no way to get audio out of a file; the ASR path is fed samples by a test |
| `vaani-vad` | voice activity detection | every chunk is transcribed in full |
| `vaani-diarization` | speaker embedding + clustering | `Speaker` exists in `vaani-core` and nothing populates it |
| `vaani-postprocess` | filler/repetition/punctuation cleanup | raw model text is the only output |
| `vaani-structuring` | optional local LLM | no summaries, no degraded mode to design |
| `vaani-export` | TXT/MD/SRT/VTT/JSON | nothing writes a transcript out |
| `vaani-storage` | rusqlite, bundled SQLite | nothing persists; `Transcript` lives in memory only |
| `vaani-models` | model manager, manifests, keyring, downloads | the artefacts are read from the repo checkout; `VAANI_ARTEFACTS` is the only override |
| `vaani-updater` | self-update | — |
| `vaani-app` | app state, job orchestration, undo | no job is ever started, so `JobPhase` has no driver |
| `vaani-cli` | `vaani.exe` + `vaani-cli.exe` | **there is no binary of any kind to run** |
| `vaani-ui` | Slint markup and presentation | — |

Concretely: `scripts/build_doctor.ps1` runs the ASR parity tests by calling the library
directly, and [RELEASE.md](RELEASE.md) has nothing to cut. Do not write documentation, error
messages, or UI copy that implies any of these exist.

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
                   vaani-media  vaani-models  vaani-updater
                   (only vaani-asr is built)
      |
      +-----------+-----------+
      v                       v
  vaani-inference         vaani-storage        (built / planned)
  (ort, windows)          (rusqlite)
      |                       |
      +-----------+-----------+
                  v
              vaani-core            <-- the only crate everything agrees on

  built today:  vaani-core  <-  vaani-inference  <-  vaani-asr
```

Rules that follow from the direction, not from preference:

- **`vaani-core` never depends upward.** It has no `#[cfg(feature)]` escape hatch and no
  dev-dependency on a heavier crate that would let a test import one.
- **`vaani-inference` knows nothing about ASR.** No mention of Conformer, TDT, mel,
  tokeniser or vocabulary appears anywhere in the crate. It can load a sentence
  embedding model without knowing what a sentence is
  (`crates/vaani-inference/src/lib.rs:26-28`).
- **`vaani-asr` owns the model semantics and nothing else.** It knows about mel bands,
  durations and SentencePiece; it does not know about files, VAD segments, speakers or
  storage. `frontend.rs` must not mention a token id, and `sravaani.rs` must not mention a
  transcript id.
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
| `inference::capability` | DXGI adapters, cudart probe, CPU/RAM, `inference_adapter()` vs `preferred_adapter()` | model names, session policy | `capability.rs:619` |
| `inference::runtime` | locating + one-time init of `onnxruntime.dll`, provider decision | what a model is | `choose_provider` is pure, `runtime.rs:364` |
| `inference::session` | typed wrapper over one `ort::session::Session` | model semantics, f16 policy rationale is *commented* here but the type is generic | `tensor_to_output` refuses a dtype mismatch rather than reinterpreting, `session.rs:586` |
| `asr::frontend` | pre-emphasis, STFT, mel, log, per-feature normalisation, frame accounting | ONNX, tokens, time bases, chunking | every constant from `preproc.json`; 128×101 parity test |
| `asr::engine` | the `AsrEngine` trait, `EngineRegistry`, `AudioChunk`, `EngineArtifacts` | any specific model | registry tests, `engine.rs:328-388` |
| `asr::sravaani` | the two graphs, the greedy TDT loop, piece→word grouping | chunking, VAD, diarization, storage | decode-parity fixture: 202/202 token ids |
| `scripts/export_sravaani_onnx.py` | TorchScript → ONNX, `preproc.pt` → flat `.f32`, `tokenizer.model` → artefacts dir | Rust, the UI, downloads | asserts `enc_lengths == ceil(T/8)` and >= 99% argmax |
| `scripts/make_golden.py` | both golden fixtures, from PyTorch and the reference TDT loop | Rust | refuses to write a fixture when its longhand front end disagrees with `extract_features` by > 1e-4 |
| `scripts/fetch_runtime.ps1` | pinned download + SHA-256 verification into `runtime/` | Rust | `Assert-Hash` throws on mismatch or on a missing pin |
| `scripts/build_doctor.ps1` | the eight-stage health gate, cheapest first | anything it does not check | exits with the number of failed stages |

## What must not change

| Do not | Why | Where the rule lives |
|---|---|---|
| Move chunk-time rebasing out of `Transcript::push_chunk` | it is the only thing standing between chunk-local and media-local time; a second path drifts silently | `transcript.rs:341-376` |
| Let an `AsrEngine` return media-local times | the engine is given a chunk and knows nothing of its offset; `source_offset` is diagnostic only | `engine.rs:50-52`, `sravaani.rs:668-674` |
| "Improve" a front-end constant | every value comes from `preproc.json` and each deviation is an accuracy regression that still looks like a transcript | `frontend.rs:41-43`, [COMMON_FAILURES §9-10](COMMON_FAILURES.md) |
| Add a parameter to any `err::*` constructor | arity is the structural guard that stops a diagnostic reaching the UI | `error.rs:262-280`, test at `error.rs:433` |
| Store `Segment` text as a field | an edit to a word would leave a stale duplicate disagreeing with it; `text()` is derived from `words` | `transcript.rs:194-206` |
| Change an `ErrorCode::as_str()` or a `JobPhase::as_str()` | both are persisted in the database and in exported JSON; a rename breaks old rows | `error.rs:475`, `job.rs:354` |
| Change `JobPhase::weight()` without re-checking the sum | the overall bar would never reach 100% | `job.rs:399` |
| Put `cuda_unavailable` on the *error* path | a silent CPU fallback is a normal outcome; it is a `ProviderStatus`, not a `VaaniError` | `lib.rs:30-37` |
| Use `preferred_adapter()` to decide anything about inference | it returns the AMD iGPU on this machine | `capability.rs:152` vs `:170` |
| Create a CUDA context at startup | costs hundreds of MB of VRAM and hundreds of ms; a user browsing history must not pay | `lib.rs:17-22` |
| Make `vaani-core` depend on anything new without writing a reason | the crate's entire value is its dependency emptiness | `lib.rs:3-10` |
| Feed the decoder the fp16 boundary tensor | the joint's LSTM state is threaded through every emission step; fp16 compounds over minutes | [ADR-004](../decisions/ADR-004-sravaani-integration.md) |
| Store one `Word` per SentencePiece piece | "ऑटोमैटिकली" is seven pieces; the metaspace `▁` is what says a word begins. The token ids are identical either way, so only the text shows the bug | `sravaani.rs:381-399` |
| Derive the decode length from the tensor dimension | the tensor has `1 + valid_frames` columns; `enc_lengths` is the truth | `sravaani.rs:459-514`, [COMMON_FAILURES §11](COMMON_FAILURES.md) |

## How to test it

```powershell
$env:Path += ";$env:USERPROFILE\.cargo\bin"
powershell -ExecutionPolicy Bypass -File scripts\build_doctor.ps1 -Release   # everything
cargo test --release --workspace          # 111 tests, ~25 s with the model present
cargo clippy --workspace --all-targets    # warning-free
```

Measured 2026-09-27, `cargo test --release --workspace`, 111 passed, 0 failed:
41 `vaani-core` + 31 `vaani-inference` unit + 2 `vaani-inference` integration + 28
`vaani-asr` unit + 3 `frontend_parity` + 6 `sravaani_transcribe`.

| Layer | What it proves | Command |
|---|---|---|
| `vaani-core` unit | the five invariants, plus `Time` formatting, phase round-trips, JSON round-trip | `cargo test -p vaani-core` |
| `vaani-inference` unit | `CudaDevicePropPrefix` byte offsets, provider precedence, tensor shape validation, f16 round-trip | `cargo test -p vaani-inference --lib` |
| `vaani-inference` integration | this machine can load the real encoder and it produces `enc_lengths == 200` for T=1600 | `cargo test -p vaani-inference --test runtime_smoke` |
| `vaani-asr` unit | the frame-count contract, the NaN guards, mel-major layout, `argmax_pair` on hand-built logits, `config.json` validation, engine opens without paging in 887 MB | `cargo test -p vaani-asr --lib` |
| front-end parity | every one of 128 × 101 cells matches PyTorch to 2e-4; padding frames are exactly `pad_value` | `cargo test -p vaani-asr --test frontend_parity -- --nocapture` |
| end-to-end decode | the real model on the GPU: frame count, step ratio, ordering, cancellation mid-decode, and — with a local clip — 202/202 token ids and an exact text match | `cargo test -p vaani-asr --test sravaani_transcribe -- --nocapture` |
| exporter self-checks | `enc_lengths == ceil(T/8)`, `d_model == 1024`, output finite, fp16 argmax agreement >= 99% | `python scripts/export_sravaani_onnx.py` |
| fixture generator | the longhand front end must agree with the model's own `extract_features` to 1e-4, or no fixture is written | `python scripts/make_golden.py` |

There are no tests that skip silently. `runtime_smoke.rs` and both `vaani-asr` integration
tests return early with a printed reason rather than passing, and `capability.rs` prints
`skipping: no usable CUDA device` and returns. **Read the printed reason**: a green
`cargo test` on a bare machine proves nothing, and `build_doctor.ps1` reports a missing
prerequisite as `SKIP`, never as `PASS`.

## What it assumes

- **Windows 11, x64, MSVC.** `windows-rs` 0.62 and `SetDllDirectoryW` are load-bearing.
  There is no Linux or macOS path and no abstraction for one.
- **A WDDM display stack.** `capability.rs:261-276` reads `IDXGIAdapter3::Budget` rather
  than `DedicatedVideoMemory`, because the budget is what a process can actually
  allocate. This is correct on WDDM and is *not* the same number as board size — on this
  machine the NVIDIA card reports 3.3 GB of a 4 GB board.
- **`cudart64_13.dll` is loadable at probe time.** `probe_cuda` is passed
  `runtime::location().provider_dirs`; if the DLL is absent the probe reports
  `error = Some(...)` and `inference_adapter()` returns `None`. That is the correct
  answer, not a failure.
- **The runtime is process-global.** `ensure_initialised` can only run once; the `OnceLock`
  turns a second call into a no-op that returns the first result
  (`runtime.rs:189-266`).
- **The ASR path is a library, not a program.** `sravaani_transcribe.rs` supplies synthetic
  samples, and the strict decode-parity test needs a local raw clip via
  `VAANI_GOLDEN_RAW`. Nothing in the repository opens a media file, so "the ASR path works"
  always means "the ASR path works on samples someone handed it".
- **Slint is the only UI toolkit and `rfd` is its only file picker.** Both are
  [planned](ARCHITECTURE.md) and neither is in `Cargo.toml` yet.
- **Model weights live outside the repository** and are checksummed by
  `artefacts.json`; see [MODELS.md](MODELS.md).

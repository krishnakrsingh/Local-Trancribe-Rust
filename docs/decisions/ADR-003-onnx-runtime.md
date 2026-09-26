# ADR-003 — ONNX Runtime for ASR

**Status:** Accepted (built)
**Date:** 2026-09-27

## Context

VaaniDesk must run a 430M-parameter FastConformer encoder plus an autoregressive TDT
decoder on Windows x64, on a 4 GB laptop GPU (3.3 GB of usable DXGI budget), with a CPU
fallback that produces an identical transcript. The reference machine is a hybrid laptop
that also reports an 8 GiB shared-memory iGPU, a trap of its own (`capability.rs:153-171`).
The encoder alone is 887 MB in fp16, so weights, runtime and artefacts total roughly 4 GB,
which rules out bundling them. Requirements: no build-time network access for anyone who
clones the repo; no `x86-64-v3` baseline that would refuse to run on older laptops; control
over which CUDA kernels are present; and a CPU path that always works.

## Decision

ONNX Runtime **1.30.0** (Microsoft's official `onnxruntime-win-x64-gpu_cuda13-1.30.0.zip`),
loaded at runtime through the **`ort` 2.0.0-rc.13** crate with the **`load-dynamic`** feature,
from DLLs we ship ourselves. `Cargo.toml:52-63` declares `ort` with `default-features = false`
and features `std`, `ndarray`, `half`, `cuda`, `load-dynamic`, `tracing`; `ort-sys` with
`default-features = false` and `std`, `load-dynamic` — so there is no C++ build step and no
`cmake` requirement.

Pins (`scripts/fetch_runtime.ps1:50-52`, `:63-64`):

| | Version | SHA-256 / note |
|---|---|---|
| ONNX Runtime | 1.30.0 | `8fa4b08359af682cd605892cb59077049700b640128bb93fd2c7776cf9f55bdc`, 294,386,823 bytes, from the GitHub releases API `digest` field |
| CUDA redist | 13.0.2 | cudart 13.0.96, cublas 13.1.0.3, nvrtc 13.0.88; hashes read from NVIDIA's `redistrib_13.0.2.json` |
| cuDNN | 9.14.0.64 `_cuda13` | `9b98b51bcead704e32640eca1770cadecec1bdf67c6db4093ff4fbcc4a206bd8`, in `scripts/cudnn.sha256` |

**Microsoft's DLL, not the `ort` crate's CDN build.** Microsoft's CI compiles for
**`86-real`** — native SASS for compute capability 8.6, the RTX 3050 Ti family. The pyke CDN
build ships `75;80;90;120`, leaving us dependent on PTX JIT and therefore on "no kernel image
is available for execution on the device" (CUDA 209) the first time an sm_86 kernel runs.

## Consequences

Shipping the DLLs ourselves removes the `ort` default's build-time network dependency, keeps
the `x86-64-v3` baseline out of our executable, and gives us a fixed ORT build. In exchange,
**we own the DLL search path**, and that is where the cost landed.

`ort` requires `ensure_initialised()` before any other call and its environment is
process-global, hence a `OnceLock` (`runtime.rs:169-238`). It must also call
`SetDllDirectoryW` for each provider directory and then preload every DLL, because **cuDNN 9
is a split library**: `cudnn64_9.dll` is a 265 KB dispatcher that `LoadLibrary`s its siblings
by bare name. Get this wrong and the failure is
`Could not locate cudnn_engines_runtime_compiled64_9.dll` from deep inside the provider
([COMMON_FAILURES §1](../agent/COMMON_FAILURES.md)).

CUDA is probed by dynamically `LoadLibraryW`-ing `cudart64_13.dll` / `cudart64_12.dll` /
`cudart64.dll` and resolving four symbols (`capability.rs:264-268`). **cudart is never
linked**, so the CUDA Toolkit is not a build or runtime dependency and the probe degrades to
"unknown" when the DLL is absent. Reading the compute capability required reproducing CUDA
13's `cudaDeviceProp` field order, because CUDA 13 inserted `uuid`/`luid` after `name` and
moved `major` from offset 328 to 360 ([COMMON_FAILURES §3](../agent/COMMON_FAILURES.md)).

`ort` 2.0.0-rc.13 **cannot set arbitrary CUDA provider options** — `CUDA::options` is
private and `ExecutionProviderDispatch::new` is crate-private, so `do_conv_in_fp16` is
unreachable. The four reachable ones are set through the typed builder
(`session.rs:316-333`): `gpu_mem_limit`, `arena_extend_strategy = kSameAsRequested`,
`cudnn_conv_use_max_workspace = 0`, `cudnn_conv_algo_search = HEURISTIC`. Note
`OnnxSession::load_with_options`'s `cuda_option_overrides` argument is **ignored**
(`session.rs:341`); the signature is misleading and should be fixed. The fallback contract is
`.fail_silently()` on the EP build (`session.rs:346`): if the provider refuses to register,
every node runs on the CPU EP and the session still produces correct output. Requesting GPU
and not getting it is a **normal outcome**, not an error. The cost is that ORT reports the
failure **only through `tracing`**, and no `tracing_subscriber` is installed yet — so a silent
CPU fallback currently looks like a mysteriously slow run.

## Alternatives rejected

| Alternative | Why not |
|---|---|
| `ort` default (statically linked) | build-time network dependency for every cloner; `x86-64-v3` baseline; pins to whatever a CDN serves that day |
| The `ort` crate's CDN ORT build | ships `75;80;90;120`; forces PTX JIT and CUDA 209 on sm_86 |
| TensorRT EP | proprietary, stronger driver requirement, and it adds nothing for a Conformer graph |
| Direct CUDA C++ via `cudarc`, or Python at runtime | hand-written host code and cuDNN graph builds; or a GIL, a second process competing for VRAM, and ~10 s interpreter start per launch |

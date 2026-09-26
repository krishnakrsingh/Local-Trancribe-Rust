# Performance

## RTF

```
RTF = processing_time / audio_duration
```

Lower is faster. `RTF = 0.004` means 16 seconds of audio transcribed in 64 ms. `RTF < 1`
is real time; `RTF > 1` is slower than real time.

This is the same definition used everywhere else in the product:

- `MediaInfo::estimate_transcription_ms(rtf)` — `declared_duration_ms * max(rtf, 0.01)`
  (`crates/vaani-core/src/media.rs:149-154`)
- `DEFAULT_ESTIMATE_RTF = 0.6` — the pre-measurement assumption shown on the drop screen
  (`media.rs:163`). Deliberately the *CPU* figure, so a first-time user on an
  unaccelerated machine is not promised something the machine cannot deliver.
- `conv_probe` prints `RTF = ms / 1000 / 16.0` for a fixed 16 s probe.

## What the measurements cover

**Encoder only. Nothing else.**

Every number below is a single call to `encoder.fp16.onnx` or `encoder.onnx` on
synthetic log-mel input. They measure:

- ONNX Runtime session load
- one CUDA-graph execution of the FastConformer encoder

They do **not** include, and must not be extrapolated to:

| Missing | Where it will live |
|---|---|
| audio demux and decode | `vaani-media` (planned) |
| resample / channel mix to 16 kHz mono f32 | `vaani-media` (planned) |
| the log-mel front end (pre-emphasis, STFT, mel, log, per-feature normalisation) | `vaani-asr` (planned) |
| the **TDT greedy decode loop** — one joint call per emission, autoregressive, so its cost is proportional to token count, not audio duration | `vaani-asr` (planned) |
| VAD | `vaani-vad` (planned) |
| speaker embedding + clustering | `vaani-diarization` (planned) |
| post-processing and structuring | `vaani-postprocess`, `vaani-structuring` (planned) |

The autoregressive decoder is the reason this caveat is not pedantic: its wall time is a
function of how many tokens the model emits, and that is not a linear function of audio
length. A 16 s clip might emit 40 tokens or 200. **Do not estimate end-to-end RTF from
these numbers.** Measure it when the pipeline exists.

## Measured results

Reference machine, verified 2026-09-27:

| | |
|---|---|
| OS | Windows 11 Home, build 26200, x64 |
| CPU | AMD Ryzen 9 5900HS, 8C/16T, 16 logical / 16 usable, 16.5 GB RAM |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU — DXGI budget 3,531,289,396 B (3.3 GB) |
| Also present | AMD Radeon(TM) Graphics, DXGI budget 8,082,601,165 B (**7.5 GiB shared**), vendor 0x1002 |
| Also present | Microsoft Basic Render Driver, 7,588,745,626 B, `is_software = true` |
| Driver | `cudaDriverGetVersion` = 13030 → CUDA 13.3, driver branch 610.88 |
| Compute capability | 8.6 (sm_86) |
| ORT | 1.30.0, CUDA 13 EP active |
| cuDNN | 9.14.0.64 |
| ORT intra-op threads | 15 (`recommended_ort_threads()` = usable − 1) |
| Build profile | `dev` (unoptimised Rust) — the GPU dominates, but these are not release numbers |

### Encoder throughput — 16 s of audio (T = 1600 mel frames), CUDA EP

| Model | Bytes on disk | Warm | RTF |
|---|---:|---:|---:|
| `encoder.fp16.onnx` (production) | 886,484,931 | **67–72 ms** | **0.004** |
| `encoder.onnx` (fp32 reference) | 1,772,812,911 | **110–112 ms** | **0.007** |

Source: `cargo run -p vaani-inference --example conv_probe`, two consecutive runs,
2026-09-27. fp16 is ~1.6× faster than fp32 on this GPU and 2.0× smaller on disk.

The **first** `conv_probe` run in a cold session measured 109 ms (fp16) and 113 ms (fp32).
The fp16 figure is ~40% worse than warm; the fp32 figure is indistinguishable from warm.
Treat the warm column as the number and the cold one as a reminder that page-cache state
matters.

### Session load

| Measurement | Value | Source |
|---|---:|---|
| fp16 encoder → first successful run, cold | 3,600 ms (recorded earlier) | brief |
| fp16 encoder → `OnnxSession::load` returns, 2026-09-27 | 7,449 ms | `cargo test ... -- --nocapture` |
| `recommended_ort_threads()` | 15 | `capability.rs:185` |

The 3.6 s and 7.4 s figures are both real measurements of the same operation on the same
machine and are not reconciled here; the difference is not explained by anything in the
code. Treat load time as **3.5–7.5 s cold** and do not quote a single number. Load time
includes cuDNN initialisation and ORT's CUDA module loading, neither of which is cached
between processes.

### Verified output shapes

| Input T (mel frames) | `enc_out` shape | `enc_lengths` | `ceil(T/8)` |
|---:|---|---:|---:|
| 1600 | `[1, 1024, 200]` | `[200]` | 200 |
| 200, 437, 4800, 24000 | — | matches `ceil(T/8)` | — |

`enc.shape[1] == 1024` is `enc_d_model` and is asserted at
`tests/runtime_smoke.rs:146`. Asserted by the exporter for T ∈ {200, 1600, 4800} at
`export_sravaani_onnx.py:432-447`, and by the integration test for T=1600 at
`runtime_smoke.rs:155-156`.

### Memory

| Quantity | Bytes | Note |
|---|---:|---|
| fp16 encoder weights on disk | 886,484,931 | single self-contained file, no external data |
| fp32 encoder weights on disk | 1,772,812,911 | just under the 2 GB protobuf limit, so still inline |
| `decoder_joint.onnx` | 43,033,439 | fp32, deliberately |
| Conformer activation headroom assumed by `vram_fits` | ~400 MiB | the value used in `capability.rs:621,672,676` |
| CUDA arena cap used in practice | free VRAM − 256 MiB | `runtime_smoke.rs:83`, `conv_probe.rs:64` |
| `runtime/` DLLs | 1,341,315,792 (1279.2 MB), 19 files | |
| `artifacts/sravaani/` | ~2.66 GB, 8 files | |
| `runtime/.cache/` | ~2,511.5 MB, 6 zips | includes a cuDNN 9.26.0.51 archive that is **not** installed |

## The tuning decisions, and what each one bought

Every one of these is in `OnnxSession::load_with_options` (`session.rs:316-333`).

| Setting | Value | Why | Documented cost avoided |
|---|---|---|---|
| `arena_extend_strategy` | `kSameAsRequested` | stops the BFC arena rounding up to the next power of two, which on a 3.3 GB budget is the difference between fitting and not | an OOM at load or mid-chunk |
| `cudnn_conv_use_max_workspace` | `0` | cuDNN's max-workspace search can allocate several GB. SraVaani's conv modules are depthwise-separable with small channel counts, so tensor-core conv algorithms would not be chosen anyway | several GB of waste |
| `cudnn_conv_algo_search` | `HEURISTIC` | the default `EXHAUSTIVE` search benchmarks algorithms on first use | a multi-second stall on the user's first job |
| `gpu_mem_limit` | caller's `max_vram_bytes` | caps the CUDA allocator arena | reserving memory the desktop compositor needs |
| graph optimisation | `Level3` | ORT's default; fuses the MatMul/Conv pairs the Conformer is full of | — |
| intra-op threads | `recommended_ort_threads()` = `usable_cpus − 1`, floor 1 | the encoder is already memory-bandwidth bound; oversubscribing adds contention without throughput | a "mysterious 2× slowdown" on a pinned process |
| thread count source | `GetProcessAffinityMask` | a process pinned to 4 of 16 cores must size its pool from 4 | oversubscription with no obvious cause |

Two things are deliberately **not** tuned:

- **`do_conv_in_fp16`.** It looks like the lever for "run convolutions in fp32" and it is
  not: `ort` 2.0.0-rc.13 exposes no key/value registration path for CUDA provider options
  (`CUDA::options` is private, `ExecutionProviderDispatch::new` is crate-private), so
  `do_conv_in_fp16 = 0` is unreachable. It was also never the actual problem — the real
  cause of the cuDNN graph-build failure was the DLL search path, and with that fixed
  both encoders run on the GPU. `session.rs:260-269` and `conv_probe.rs` are the record.
- **`OnnxSession::load_with_options`'s `cuda_option_overrides` parameter is ignored.**
  `session.rs:341` is `let _ = cuda_option_overrides;`. The function is called with
  `Vec::new()` by both the test and `conv_probe`, so this is currently harmless, but the
  signature is misleading: it implies overrides are applied and they are not. Fix the
  signature or apply the options; do not build on the assumption that they work.

## What must not change

| Do not | Why |
|---|---|
| Quote a whole-pipeline RTF from these numbers | the decode loop is autoregressive and its cost is not a function of audio length alone |
| Benchmark in `dev` profile and publish | use `--release`; note the profile in any published number |
| Benchmark the fp32 encoder and call it the product | `encoder.fp16.onnx` is the production path; the fp32 graph is the reference and the CPU fallback |
| Time the first run in a process | `conv_probe` and `runtime_smoke` both warm up first, precisely so cuDNN algorithm selection is not in the number |
| Drop the 256 MiB VRAM reserve | that margin is the desktop compositor's |
| Change `JobPhase::weight()` on the strength of a measurement | the weights are *presentation* estimates only, explicitly documented as such at `job.rs:136-139`, and the benchmark report is required to show measured values instead |
| Use `preferred_adapter()` to size anything | it returns the AMD iGPU's 7.5 GiB of shared memory on this machine |

## How to reproduce every number on this page

```powershell
$env:Path += ";$env:USERPROFILE\.cargo\bin"

# Full hardware + adapter + CUDA report, and a timed encoder run:
cargo test -p vaani-inference --test runtime_smoke -- --nocapture

# Both encoders, warm:
cargo run -p vaani-inference --example conv_probe

# Run conv_probe twice: the first process after a reboot is the cold column.
```

`runtime_smoke.rs:138-143` prints the RTF directly:

```
encoder: 16 s of audio in 68 ms -> RTF 0.004
  enc_out shape [1, 1024, 200], enc_lengths [200]
```

Any number added to this document must come from one of those two commands, must state
the machine, and must state the build profile. A number with no provenance is worse than
no number.

## What it assumes

- **A fixed 16 s probe (T = 1600 mel frames).** RTF is not constant in T: fixed setup
  costs (kernel selection, memory) amortise better over longer inputs, and attention cost
  grows with T. Do not compare a T=200 measurement to a T=24000 one as if the ratio were
  meaningful.
- **Batch size 1.** Every measurement is `batch = 1`. The graphs declare a dynamic batch
  axis, but no batched measurement has been taken.
- **cuDNN caches are warm within a process and cold across processes.** The 3.5–7.5 s
  load-time spread is real and unexplained; do not quote a single value.
- **`conv_probe` runs the fp16 trial first.** It pays CUDA-context and cuDNN-algo costs
  that the fp32 trial then does not, which biases fp16 *upward*. A fair comparison
  reorders the trials; that has not been done, so treat the 1.6× ratio as indicative.
- **The reference machine is the only machine measured.** A 3.3 GB DXGI budget with an
  8 GiB-shared-memory iGPU alongside it is a specific and awkward configuration. Numbers
  from a discrete-desktop or an eGPU machine are not comparable.

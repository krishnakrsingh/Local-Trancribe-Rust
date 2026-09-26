# Performance

Every figure on this page is a **measurement**, not an estimate, and each one names the
command that produced it. The only estimates here are labelled *estimate*. Where a number
could not be measured it says so rather than carrying a plausible-looking value.

## RTF

```
RTF = processing_time / audio_duration
```

Lower is faster. `RTF = 0.004` means 16 seconds of audio transcribed in 64 ms. `RTF < 1` is
real time; `RTF > 1` is slower than real time. The same definition is used everywhere else
in the product:

- `MediaInfo::estimate_transcription_ms(rtf)` — `declared_duration_ms * max(rtf, 0.01)`
  (`crates/vaani-core/src/media.rs:156`)
- `DEFAULT_ESTIMATE_RTF = 0.6` (`media.rs:170`) — the pre-measurement assumption shown on
  the drop screen, deliberately the *CPU* figure so a first-time user on an unaccelerated
  machine is not promised something the machine cannot deliver. It is still an **estimate**:
  no CPU-only run has been taken, so nothing here replaces it yet.
- `examples/conv_probe.rs:88` prints `RTF = ms / 1000 / 16.0` for a fixed 16 s probe.
- `crates/vaani-asr/tests/sravaani_transcribe.rs:177` prints the end-to-end figure for a
  40 s chunk.

## What the measurements cover

**ASR only: log-mel front end, encoder, TDT decode loop, SentencePiece detokenise and word
grouping.** That is the whole of `crates/vaani-asr`, and it is what every number below
measures. The 40 s figures are one `transcribe` call on one 40 s chunk; the 16 s figures are
one encoder run on synthetic log-mel input.

They do **not** include, and must not be extrapolated to:

| Missing | Where it will live | Measured? |
|---|---|---|
| audio demux and decode | `vaani-media` (planned) | **not measured** |
| resample / channel mix to 16 kHz mono f32 | `vaani-media` (planned) | **not measured** |
| VAD | `vaani-vad` (planned) | **not measured** |
| speaker embedding + clustering | `vaani-diarization` (planned) | **not measured** |
| post-processing and structuring | `vaani-postprocess`, `vaani-structuring` (planned) | **not measured** |
| transcript export, storage, the UI, the CLI | `vaani-export`, `vaani-storage`, `vaani-app`, `vaani-cli`, `vaani-ui` (all planned) | **not measured** |
| the full 834 s meeting recording | not built; no media pipeline exists to feed it | **not measured** |
| CPU-only RTF, i.e. the `DEFAULT_ESTIMATE_RTF = 0.6` figure | `Provider::Cpu` path works; it has never been timed | **not measured** |

### The product-level target has NOT been validated

The requirements ask for **under 1.5 s startup** and **faster than realtime** end to end.
Neither has been demonstrated, and neither can be, because media ingestion, VAD,
diarization and the UI do not exist. The end-to-end path is not wired together; it is a
library called from a test. The only ASR-related cold cost that has been measured is the
encoder session load, and on this machine that is **3.6–7.6 s** (below). If the 1.5 s
startup budget includes opening the ASR weights, today's measured load time does not meet
it. Do not report an application startup time, an end-to-end RTF, or a "faster than
realtime" claim from this page.

## Reference machine

Verified 2026-09-27 against the development machine:

| | |
|---|---|
| OS | Windows 11 Home, build 26200, x64 |
| CPU | AMD Ryzen 9 5900HS, 8C/16T, 16 logical / 16 usable, 16.5 GB RAM |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB board) — DXGI budget 3,531,289,396 B (3.3 GB) |
| Also present | AMD Radeon(TM) Graphics, DXGI budget 8,082,601,165 B (**7.5 GiB shared**), vendor 0x1002 |
| Also present | Microsoft Basic Render Driver, 7,588,745,626 B, `is_software = true` |
| Driver | `cudaDriverGetVersion` = 13030 → CUDA 13.3, driver branch 610.88 |
| Compute capability | 8.6 (sm_86) |
| ORT | 1.30.0, CUDA 13 EP active (`active: Cuda`) |
| cuDNN | 9.14.0.64 |
| ORT intra-op threads | 15 (`recommended_ort_threads()` = usable − 1, `capability.rs:194`) |
| Build profile | **`release`** (`lto = "thin"`, `codegen-units = 1`, `Cargo.toml:105-110`) |
| Rust | 1.98.1, `x86_64-pc-windows-msvc` |

## Measured: the ASR path, 40 s chunk

`cargo test --release -p vaani-asr --test sravaani_transcribe -- --nocapture --test-threads=1`,
which prints the three phases and the RTF. The chunk in that test is **synthetic** (a
two-tone signal of the golden fixture's length, 640,000 samples at 16 kHz), so these are
ASR-path costs, not accuracy results.

| Phase | Wall time | What it covers |
|---|---:|---|
| Log-mel front end | **203 ms** | pre-emphasis, 512-point STFT, 128 mel bands, log, per-feature Bessel normalisation, 4000 valid + 1 padding frame |
| Encoder session run | **1498 ms** | the whole `session.run` call **including per-call dispatch**; see the isolated figure below |
| TDT decode loop | **299 ms** | 131 emission steps over 500 encoder frames = **0.262 steps/frame** |
| End-to-end ASR RTF for 40 s | **0.050** and **0.032** in two runs | `total_ms / 1000 / 40`; quote the range, never one end of it |

Two further executions of the same command on the same machine measured **205 / 1481 /
345 ms** and RTF **0.051**, and RTF **0.032** respectively. So the phases are stable to
about ±2% across runs, and the **end-to-end RTF should be quoted as 0.032–0.051 with the
number of runs stated** — three runs, not one, and not the best one. The 1498 ms encoder
figure is the outlier in its group and is the one carrying per-call dispatch; the isolated
warm number below is an order of magnitude smaller and measures something else.

### The same 40 s on real speech

`matches_reference_token_ids_on_real_audio`, same profile, with
`VAANI_GOLDEN_RAW` set: 40 s of real meeting speech, **357 emission steps**, and **897 ms**
for the whole `transcribe` call. That is *faster* than the synthetic run above, and the
reason is the harness, not the audio: `debug_tokens` runs immediately before it in the same
process, so the encoder and the joint are both warm. Read it as "the ASR path costs about
0.9 s for 40 s of real speech once warm", and do not use it to revise the table above, which
is the cold, single-call measurement.

## Measured: isolated warm encoder run, 16 s of audio (T = 1600 mel frames)

`cargo run -p vaani-inference --example conv_probe` and
`cargo test --release -p vaani-inference --test runtime_smoke -- --nocapture`. Both warm the
session before timing, so cuDNN algorithm selection is not in the number.

| Model | Bytes on disk | Warm | RTF |
|---|---:|---:|---:|
| `encoder.fp16.onnx` (production) | 886,484,931 | **67 ms** | **0.004** |
| `encoder.onnx` (fp32 reference) | 1,772,812,911 | **111 ms** | **0.007** |

fp16 is ~1.6× faster than fp32 on this GPU and 2.0× smaller on disk. A repeat run of
`runtime_smoke` measured 68 ms / RTF 0.004 for the fp16 graph, which is the same number to
the resolution of a millisecond timer.

## Measured: correctness, not just speed

| Check | Result | Command |
|---|---|---|
| Front-end parity, all 128 × 101 cells | max \|rust − reference\| = **7.99e-05** (worst at mel 11, frame 67) | `cargo test --release -p vaani-asr --test frontend_parity -- --nocapture` |
| Decode parity, token ids | **202 / 202 identical** to the reference | `cargo test --release -p vaani-asr --test sravaani_transcribe matches_reference_token_ids_on_real_audio -- --nocapture` with `VAANI_GOLDEN_RAW` set |
| Decode parity, word timings | **108 / 108 words**, worst deviation **0.0 ms** | same |
| Transcript text | **exact match** with the reference | same |
| Decode step count | Rust 357 vs reference 358 (delta 1) | same |

These are the numbers that make the speed numbers meaningful: an ASR path that is fast and
wrong is not a result. Tolerance on the front end is 2e-4 on O(1) values
(`tests/frontend_parity.rs:113`); every convention error found while writing the port missed
it by three to four orders of magnitude.

## Measured: session load

| Measurement | Value | Source |
|---|---:|---|
| fp16 encoder, `OnnxSession::load` returns, cold | **3,600 ms** | an earlier session on this machine; profile not recorded |
| fp16 encoder, cold | **7,449 ms** | `cargo test ... -- --nocapture` (earlier run) |
| fp16 encoder, cold | **7,580 ms** | `cargo test --release -p vaani-inference --test runtime_smoke -- --nocapture`, 2026-09-27 |

**These are not reconciled.** Three measurements of the same operation on the same machine
span 3.6–7.6 s and nothing in the code explains the spread. Treat load time as
**3.6–7.6 s cold**, quote it as a range, and do not reduce it to a single value. Load time
includes cuDNN initialisation and ORT's CUDA module loading, neither of which is cached
between processes. Until it is explained, no startup budget may be built on it.

## Verified output shapes

`enc_lengths == ceil(T / 8)` exactly, where `T` is `feature_lengths`. Measured with
`cargo run --release -p vaani-inference --example subsample_probe`, which asks the graph
rather than assuming:

| `feature_lengths` T | `ceil(T/8)` | `enc_lengths` reported | `enc_out.shape[2]` |
|---:|---:|---:|---:|
| 200 | 25 | 25 | 25 |
| 437 | 55 | 55 | 55 |
| 1600 | 200 | 200 | 200 |
| 1601 | 201 | 201 | 201 |
| 4000 | 500 | 500 | 500 |
| 4800 | 600 | 600 | 600 |
| 24000 | 3000 | 3000 | 3000 |

`enc.shape[1] == 1024` is `enc_d_model`, asserted at `tests/runtime_smoke.rs:146`. The probe
sweeps twelve lengths; the seven rows above are the ones quoted anywhere in these docs.

**One trap, and it is a real one.** In the ASR path the log-mel *tensor* has
`1 + valid_frames` time columns, because `torch.stft` with `center = true` emits one more
frame than `feature_lengths` describes. A 40 s chunk is 4000 valid frames but a 4001-column
tensor, and the encoder subsamples the **tensor**: the tensor-derived length is
`ceil(4001/8) = 501` and the graph's `enc_lengths` reports **500**. PyTorch's reference
decodes exactly `enc_len` frames, so using the tensor dimension decodes one spurious frame
past the end of the audio. `sravaani.rs:503` uses `enc_lengths`; see
[COMMON_FAILURES §11](COMMON_FAILURES.md).

## Memory

| Quantity | Bytes | Note |
|---|---:|---|
| fp16 encoder weights on disk | 886,484,931 | single self-contained file, no external data |
| fp32 encoder weights on disk | 1,772,812,911 | just under the 2 GB protobuf limit, so still inline |
| `decoder_joint.onnx` | 43,033,439 | fp32, deliberately ([ADR-004](../decisions/ADR-004-sravaani-integration.md)) |
| Conformer activation headroom assumed by `vram_fits` | ~400 MiB | **an assumption**, the value used in the tests at `capability.rs:647,701`; it has not been measured against a real allocation |
| CUDA arena cap used in practice | free VRAM − 256 MiB | `tests/runtime_smoke.rs:94`, `examples/conv_probe.rs:74` |
| `runtime/` DLLs | 1,341,315,792 (1279.2 MB), 19 files | |
| `artifacts/sravaani/` | 2,702,792,388 (2577.6 MiB), 10 files | includes `tokenizer.model`, 325,287 B |
| `runtime/.cache/` | ~2,511.5 MB, 6 zips | includes a cuDNN 9.26.0.51 archive that is **not** installed |

## The tuning decisions, and what each one bought

All of these are in `OnnxSession::load_with_options` (`session.rs:326-353`).

| Setting | Value | Why | Documented cost avoided |
|---|---|---|---|
| `arena_extend_strategy` | `kSameAsRequested` | stops the BFC arena rounding up to the next power of two, which on a 3.3 GB budget is the difference between fitting and not | an OOM at load or mid-chunk |
| `cudnn_conv_use_max_workspace` | `0` | cuDNN's max-workspace search can allocate several GB. SraVaani's conv modules are depthwise-separable with small channel counts, so tensor-core conv algorithms would not be chosen anyway | several GB of waste |
| `cudnn_conv_algo_search` | `HEURISTIC` | the default `EXHAUSTIVE` search benchmarks algorithms on first use | a multi-second stall on the user's first job |
| `gpu_mem_limit` | caller's `max_vram_bytes` | caps the CUDA allocator arena | reserving memory the desktop compositor needs |
| graph optimisation | `Level3` | ORT's default; fuses the MatMul/Conv pairs the Conformer is full of | — |
| intra-op threads | `recommended_ort_threads()` = `usable_cpus − 1`, floor 1 | the encoder is already memory-bandwidth bound; oversubscribing adds contention without throughput | a "mysterious 2× slowdown" on a pinned process |
| thread count source | `GetProcessAffinityMask` | a process pinned to 4 of 16 cores must size its pool from 4 | oversubscription with no obvious cause |

Two things are deliberately **not** done:

- **`do_conv_in_fp16` is not set.** It looks like the lever for "run convolutions in fp32"
  and it is not reachable: `ort` 2.0.0-rc.13 exposes no key/value registration path for CUDA
  provider options (`CUDA::options` is private, `ExecutionProviderDispatch::new` is
  crate-private). It was also never the actual problem — the real cause of the cuDNN
  graph-build failure was the DLL search path, and with that fixed both encoders run on the
  GPU. `session.rs:277-286` and `conv_probe` are the record.
- **`load_with_options`'s `cuda_option_overrides` cannot be honoured, and says so loudly.**
  `session.rs:353-364` returns an internal error naming the keys rather than ignoring them,
  because a silently-ignored option in a benchmark means publishing numbers for a
  configuration nobody ran. Every current caller passes `Vec::new()`.

## What must not change

| Do not | Why |
|---|---|
| Quote a whole-pipeline or product RTF from these numbers | media decode, VAD, diarization, post-processing, structuring and the UI do not exist; the decode loop is autoregressive and its cost is a function of how many tokens the model emits, not of audio length |
| Present the 0.032–0.051 ASR RTF as the product's speed | it is one library call on one 40 s chunk. Quote the range and the phase breakdown, or quote nothing |
| Reduce the 3.5–7.5 s cold load to one number | three measurements, unexplained spread |
| Benchmark in `dev` profile and publish | every number on this page is `--release`; note the profile in any published number |
| Benchmark the fp32 encoder and call it the product | `encoder.fp16.onnx` is the production path; the fp32 graph is the reference and the CPU fallback |
| Time the first run in a process | `conv_probe` and `runtime_smoke` both warm up first, precisely so cuDNN algorithm selection is not in the number |
| Quote 1498 ms as "the encoder takes 1.5 s" | that is one dispatch-inclusive call inside a 40 s `transcribe`; the warm isolated encoder run is 67 ms. Both are real and they measure different things |
| Drop the 256 MiB VRAM reserve | that margin is the desktop compositor's |
| Change `JobPhase::weight()` on the strength of a measurement | the weights are *presentation* estimates only, documented as such at `job.rs:136-139` |
| Use `preferred_adapter()` to size anything | it returns the AMD iGPU's 7.5 GiB of shared memory on this machine |

## How to reproduce every number on this page

```powershell
$env:Path += ";$env:USERPROFILE\.cargo\bin"

# Everything: toolchain, fmt, check, clippy, tests, runtime, model, parity.
powershell -ExecutionPolicy Bypass -File scripts\build_doctor.ps1 -Release

# 40 s phase breakdown + ASR RTF, and the encoder's subsampling law:
cargo test --release -p vaani-asr --test sravaani_transcribe -- --nocapture --test-threads=1
cargo run  --release -p vaani-inference --example subsample_probe

# Front-end parity, 128x101 cells:
cargo test --release -p vaani-asr --test frontend_parity -- --nocapture

# Decode parity, 202 token ids / 108 word timings / exact text. The path must be
# absolute: cargo runs integration tests with the package directory as cwd.
$env:VAANI_GOLDEN_RAW = "C:\Users\krish\Downloads\localvoicetool\scratch\golden_clip.f32"
cargo test --release -p vaani-asr --test sravaani_transcribe `
  matches_reference_token_ids_on_real_audio -- --nocapture --test-threads=1

# Both encoders, warm, plus the cold session-load figure:
cargo test --release -p vaani-inference --test runtime_smoke -- --nocapture
cargo run    --release -p vaani-inference --example conv_probe
```

`runtime_smoke.rs` prints the RTF and the load time directly:

```
loaded ...\artifacts\sravaani\encoder.fp16.onnx in 7580 ms
encoder: 16 s of audio in 68 ms -> RTF 0.004
  enc_out shape [1, 1024, 200], enc_lengths [200]
```

Any number added to this document must come from one of those commands, must state the
machine, and must state the build profile. A number with no provenance is worse than no
number.

## What it assumes

- **Batch size 1, one chunk at a time.** Every measurement is `batch = 1`. The graphs
  declare a dynamic batch axis; no batched measurement has been taken.
- **cuDNN caches are warm within a process and cold across processes.** The 3.5–7.6 s
  load-time spread is real and unexplained.
- **`conv_probe` runs the fp16 trial first.** It pays CUDA-context and cuDNN-algo costs
  that the fp32 trial then does not, which biases fp16 *upward*. A fair comparison reorders
  the trials; that has not been done, so treat the 1.6× ratio as indicative.
- **The decode-step ratio is input-dependent, and the source comments' range is wrong.**
  `engine.rs:95` and `sravaani.rs:19` say "roughly 0.3–0.5 steps per frame on speech". Measured:
  **0.262** on a 40 s synthetic chunk and **0.714** on 40 s of real speech. Neither is inside
  the documented range, and the synthetic figure is not speech at all. Treat the comment as
  unverified, and the ratio as a property of the audio. The test's real assertion is the
  plausibility band 0.2–4.0 at `tests/sravaani_transcribe.rs:205`.
- **The reference machine is the only machine measured.** A 3.3 GB DXGI budget with an
  8 GiB-shared-memory iGPU alongside it is a specific and awkward configuration. Numbers
  from a discrete-desktop or an eGPU machine are not comparable, and no CPU-only number
  exists at all.

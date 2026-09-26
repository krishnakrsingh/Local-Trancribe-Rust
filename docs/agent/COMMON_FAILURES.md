# Common failures

Twelve failures that present as code bugs and are not. Each is: **symptom / cause / fix**.
Every error string here is quoted from a source file in this repository or from a tool
chain the repository pins. Failures 9–12 were found and fixed while writing the Rust ASR
path, and each is now covered by a regression test.

---

## 1. `Could not locate cudnn_engines_runtime_compiled64_9.dll`

**Symptom.** Session load fails, or the first inference run fails deep inside ONNX
Runtime's CUDA provider with this message. Everything still works on the CPU EP, so it
presents as "GPU acceleration mysteriously unavailable".

**Cause.** cuDNN 9 is a **split library**. `cudnn64_9.dll` is only a 265,328-byte
dispatcher; it `LoadLibrary`s its siblings *by bare name* at first use. The default
Windows DLL search order starts at the directory of the executable. In development the
executable lives in `target/<profile>/`, several directories away from `runtime/`, so
none of the siblings resolve.

The siblings in `runtime/`, all of which are required:

| File | Bytes |
|---|---:|
| `cudnn64_9.dll` (dispatcher) | 265,328 |
| `cudnn_adv64_9.dll` | 104,365,680 |
| `cudnn_cnn64_9.dll` | 1,500,784 |
| `cudnn_engines_precompiled64_9.dll` | 196,693,104 |
| `cudnn_engines_runtime_compiled64_9.dll` | 23,116,912 |
| `cudnn_graph64_9.dll` | 2,344,560 |
| `cudnn_heuristic64_9.dll` | 54,122,608 |
| `cudnn_ops64_9.dll` | 34,962,032 |

**Fix.** Already handled, in `runtime::ensure_initialised()`
(`crates/vaani-inference/src/runtime.rs:209-266`). Do not remove either half:

1. `SetDllDirectoryW` for each provider dir (`runtime.rs:235-239`, helper at `:273-287`) —
   appends the directory to the search order for every `LoadLibrary` made *after* the call.
2. `preload_dir` for each provider dir (`runtime.rs:243-245`, helper at `:294-322`) —
   resolves every DLL now so an incomplete install fails with *our* message naming the
   missing file. A `debug`-level `could not preload runtime DLL` (`runtime.rs:317`) names
   the culprit.

**The preloads alone are not enough, and that is the part worth remembering.** cuDNN's
dispatcher does not resolve all of its siblings when it loads; it resolves them **lazily,
during the first inference run**, by bare name, through the ordinary search order. A process
that preloads every DLL in `runtime/` and then runs a graph can still fail with this exact
message, because the preloads prove the files exist and are loadable, not that the *next*
`LoadLibrary` will find them. `SetDllDirectoryW` is the only half that covers the lazy case,
which is why it runs first and why the two are not interchangeable. If you ever see this
message with a complete `runtime/`, check the DLL **search order**, not the file list.

If you see this in a **shipped** build, the installer put the DLLs in a subdirectory.
`Expand-To` at `fetch_runtime.ps1:110-116` copies DLLs **flat** for exactly this reason,
and `RuntimeLocation::portable_layout` records which layout was found.

---

## 2. `Type parameter (T) of Optype (Add) bound to different types (tensor(float) and tensor(float16) in node (/pre_encode/Add))`

**Symptom.** ONNX Runtime refuses to load the fp16 encoder. Often seen with a
`CUDNN_BACKEND_API_FAILED` prefix, which is a red herring — the graph is invalid before
cuDNN is involved.

**Cause.** `onnxconverter_common.float16` is **wrong for this graph.** SraVaani's
`ConvSubsampling` computes the subsampling *length* with integer nodes operating on
`feature_lengths`:

```
/pre_encode/Cast  ->  /pre_encode/Add  ->  /pre_encode/Div  ->  /pre_encode/Floor
```

The post-hoc converter sees only "these are float tensor types" and rewrites the whole
`pre_encode` prologue to fp16, producing a graph where an fp16 activation is added to an
int32 constant. The export succeeds, PyTorch-side validation passes, and the graph is
rejected at load.

**Fix.** Export **natively from an fp16 module** so PyTorch inserts the boundary Casts
and the integer arithmetic stays integer (`export_sravaani_onnx.py:301-334`):

```python
enc_half = encoder.half()
torch.onnx.export(
    enc_half,
    (torch.randn(1, n_mels, t, dtype=torch.float16), torch.tensor([t], dtype=torch.long)),
    str(target),
    input_names=["feats", "feature_lengths"],
    output_names=["enc_out", "enc_lengths"],
    dynamic_axes={...},
    opset_version=17,
    dynamo=False,
    external_data=False,
    do_constant_folding=True,
)
```

Result: 887 MB, single self-contained file, **native fp16 graph I/O**, 99.609% argmax
agreement with the fp32 reference. See [ADR-004](../decisions/ADR-004-sravaani-integration.md).

---

## 3. Compute capability reported as `78.73` (or any two-ASCII-character value)

**Symptom.** `CudaProbe::compute_capability` reads back something absurd — on an RTX 3050
Ti, `78.73`. 78 and 73 are `'N'` and `'I'`, the first two bytes of the device *name*.

**Cause.** Reading `major`/`minor` from offset 0 of `cudaDeviceProp`. The field order is
not intuitive, and **CUDA 13 changed it**: `uuid[16]`, `luid[8]` and `luidDeviceNodeMask`
were inserted after `name[256]`, moving `major` from offset 328 to **360** and `minor` to
**364**. A layout copied from a CUDA 12 header silently returns garbage on a current
driver.

**Fix.** `CudaDevicePropPrefix` (`capability.rs:290-312`) reproduces CUDA 13's declaration
order exactly. Verified field offsets:

| Field | Offset |
|---|---:|
| `name` | 0 |
| `uuid` | 256 |
| `luid` | 272 |
| `luidDevice_node_mask` | 280 |
| `total_global_mem` | 288 |
| `regs_per_block` | 304 |
| `mem_pitch` | 312 |
| `max_threads_per_block` | 320 |
| `total_const_mem` | 352 |
| **`major`** | **360** |
| **`minor`** | **364** |
| struct size | 1032, align 8 |

Pinned by `device_prop_field_offsets_match_the_c_header` (`capability.rs:717-741`) and
proved against the real device by `compute_capability_read_from_the_real_device_is_plausible`
(`capability.rs:766-796`), which asserts the value is *not* the device name's first two
bytes.

The layout is only safe because `fetch_runtime.ps1:50-52` pins `cudart64_13.dll` exactly.
The DLL, not the header, is the contract.

---

## 4. The product claims GPU acceleration while running on the CPU

**Symptom.** Settings shows a GPU, everything runs, and it is 20–40× slower than expected.
On the reference machine the *wrong* adapter is the one that looks better.

Measured adapter list on the dev box:

| Adapter | DXGI budget | `is_software` | vendor | Usable by CUDA |
|---|---:|---|---|---|
| NVIDIA GeForce RTX 3050 Ti Laptop GPU | 3,531,289,396 B (3.3 GB) | false | 4318 (0x10DE) | **yes** |
| AMD Radeon(TM) Graphics | 8,082,601,165 B (**7.5 GiB**) | false | 4098 (0x1002) | no |
| Microsoft Basic Render Driver | 7,588,745,626 B | **true** | 5140 | no |

**Cause.** Ranking adapters by memory. `MachineCapabilities::preferred_adapter()`
(`capability.rs:146-151`) is `max_by_key(dedicated_vram_bytes)` filtered to
non-software — so it returns the **AMD integrated GPU**, which reports 7.5 GiB of
*shared system memory* through `IDXGIAdapter3::Budget`. That is a bigger number than the
discrete card's real budget. WARP is even larger and is only excluded by the
`is_software` flag.

Note also that `dedicated_vram_bytes` is the DXGI **Budget**, not
`DedicatedVideoMemory` (`capability.rs:240-246`) — deliberately, because on WDDM the
budget is what a process can actually allocate.

**Fix.** Use `inference_adapter()` (`capability.rs:164-171`) for anything about
inference:

```rust
if self.cuda.is_usable()
    && let Some(nv) = self.adapters.iter().find(|a| a.is_nvidia() && !a.is_software)
{
    return Some(nv);
}
None
```

It requires a working CUDA device *and* NVIDIA's PCI vendor id `0x10DE` — never a name
substring match. If CUDA is not usable it returns `None`, which is the honest answer.

`preferred_adapter()` is for **display** only. `RuntimeReport::summary()` already gets this
right and names the inference adapter (`lib.rs:83-95`). Pinned by
`hybrid_laptop_picks_nvidia_not_the_shared_memory_igpu` (`capability.rs:618-648`).

Related: `CudaProbe::is_usable()` requires `device_count > 0` **and** `error.is_none()`
(`capability.rs:89-91`). A probe that found a device but then failed must not be treated
as usable.

---

## 5. `Unexpected input data type. Actual: (tensor(float)), expected: (tensor(float16))`

**Symptom.** The encoder run fails immediately with a dtype mismatch, naming `feats`.

**Cause.** Feeding `Input::f32` to `encoder.fp16.onnx`. That graph is **natively fp16**;
its `feats` input and `enc_out` output are `FLOAT16`, not float. Verified from the proto:

| Graph | `feats` | `enc_out` |
|---|---|---|
| `encoder.fp16.onnx` | **FLOAT16** [batch,128,time] | **FLOAT16** [batch,1024,enc_time] |
| `encoder.onnx` | FLOAT [batch,128,time] | FLOAT [batch,1024,enc_time] |

**Fix.** Convert at the boundary, in one place:
`F16Tensor::from_f32(&F32Tensor)` then `Input::f16("feats", ...)`.

```rust
let feats_f32 = F32Tensor::new(vec![1, 128, t_frames], data)?;
let feats = F16Tensor::from_f32(&feats_f32);
session.run(&[Input::f16("feats", feats),
              Input::i64("feature_lengths", I64Tensor::scalar(t_frames))],
            &["enc_out", "enc_lengths"])
```

The conversion lives in the inference layer, not the ASR crate, so the media and front-end
crates never learn the encoder's precision (`session.rs:63-69`). The **front end still
computes log-mel in f32** — the per-feature normalisation divides by a standard deviation
and losing mantissa bits there is a real accuracy cost. The f16 round-trip is verified
lossless enough for log-mel values in [-8, 8] at `session.rs:606-617`.

The reverse error (`expected: (tensor(float))`) means you fed f16 to `encoder.onnx`.

---

## 6. `onnx.checker` `ShapeInferenceError` on an fp16 graph

**Symptom.** `onnx.checker.check_model(path, full_check=True)` raises
`ShapeInferenceError` (or a shape/type mismatch deep in a subgraph) on a converted fp16
model.

**Cause.** Stale `value_info` left behind by a post-hoc float16 conversion. The converter
rewrites initialiser and node types but does not consistently rewrite the
`value_info` entries that shape inference reads, so inference reports types that no
longer exist in the graph.

**Fix.** **It is harmless and the right answer is to ignore it.** The current exporter
does exactly that (`export_sravaani_onnx.py:470-482`): it runs the checker and *reports*
rather than aborts.

```python
try:
    onnx.checker.check_model(str(p), full_check=True)
    log(f"onnx.checker OK: {name}")
except Exception as exc:
    log(f"onnx.checker WARN {name}: {exc}")   # reported, not fatal
```

The durable fix, and what the production path uses, is to **export natively in fp16** so
there is no stale `value_info` to begin with — see failure 2. The real acceptance test is
not `onnx.checker`; it is that ONNX Runtime **loads and runs** the graph and that
`enc_lengths == ceil(T/8)` with a finite output
(`export_sravaani_onnx.py:431-447`). The generic float16 converter produced a graph that
failed exactly there, and no amount of PyTorch-side validation would have caught it.

---

## 7. `no 'link.exe'` / MSVC not found / `linker link.exe not found`

**Symptom.** `cargo build` compiles and then fails at the link step with
`error: linker 'link.exe' not found`, or `rustc` reports it cannot find the MSVC linker
or the Windows SDK headers/libs.

**Cause.** Visual Studio Build Tools is not installed, or is installed without the C++
workload. Note this surfaces at **link** time, not compile time, which is why it looks
like a toolchain problem rather than a build-tools problem.

**Fix.** Install **Visual Studio Build Tools 2022** with workload
**MSVC v143 — VS 2022 C++ x64/x86 build tools** (`VCTools`), plus a Windows SDK
(reference machine: 10.0.26100). Reference: MSVC 14.44.35207, VS Build Tools 2022
17.14.37710.

`rustc` auto-discovers MSVC through the registry. **Do not** prepend
`VC\Auxiliary\Build\vcvars64.bat` to `PATH`; the toolchain finds it, and doing it anyway
tends to pick up the wrong SDK.

Separately: if `cargo` itself is not found after installing rustup, it is at
`$env:USERPROFILE\.cargo\bin` and a `--no-modify-path` install does not add it to `PATH`.
See [BUILD.md §2](BUILD.md).

---

## 8. cuDNN too new (or too old) for the ONNX Runtime build

**Symptom.** `CUDNN_BACKEND_API_FAILED`, cuDNN version assertions in ORT's logs, or a
cuDNN 9.14 graph-build failure where a 9.x that "should be newer" is present instead.

**Cause.** ONNX Runtime's CUDA EP is compiled against a specific cuDNN major/minor and its
dispatcher validates at load. VaaniDesk pins **cuDNN 9.14.0.64 `_cuda13`** because that
is the version ORT 1.30.0 was built and CI'd against. Dropping in a different cuDNN — even
a newer one — is unsupported, and the cuDNN 9 dispatcher's `LoadLibrary`-by-bare-name
behaviour (failure 1) means the wrong version can be picked up from anywhere on the
search path.

Note the machine is currently a live example of this hazard:
`runtime/.cache/` contains `cudnn-windows-x86_64-9.26.0.51_cuda13-archive.zip` (1.28 GB),
downloaded but **not** expanded. The installed `cudnn64_9.dll` is 9.14.0.64, confirmed by
its `FileVersion`.

**Fix.** Pin. `fetch_runtime.ps1:52` sets `$CudnnVersion = '9.14.0.64'`, and the hash is in
`scripts/cudnn.sha256`:

```
9b98b51bcead704e32640eca1770cadecec1bdf67c6db4093ff4fbcc4a206bd8
```

`Assert-Hash` (`fetch_runtime.ps1:73-81`) throws on a mismatch and also on a *missing*
pin. Bumping it is a deliberate change that requires re-running the benchmark suite,
because a different ORT/cuDNN pairing means a different set of CUDA kernels and therefore
different timings — see [PERFORMANCE.md](PERFORMANCE.md).

The same applies to ONNX Runtime itself: `onnxruntime.dll` reports
`FileVersion 1.30.0.20260909.4.f2c39fe`, pinned at `fetch_runtime.ps1:50` with SHA-256
`8fa4b08359af682cd605892cb59077049700b640128bb93fd2c7776cf9f55bdc`. It is Microsoft's
release asset rather than the `ort` crate's CDN build because Microsoft's CI compiles for
`86-real` — native SASS for the RTX 3050 Ti's compute capability 8.6 — while the CDN build
ships `75;80;90;120` and would leave us dependent on PTX JIT
([ADR-003](../decisions/ADR-003-onnx-runtime.md)).

---

## 9. Transcripts are consistently, quietly worse than the model's

**Symptom.** Nothing crashes. Transcripts are intelligible, in the right language, with
roughly plausible WER — and *consistently* worse than the model card's numbers, in a way
that does not move when you change anything else. The tell is in the parity test:
`frontend_parity.rs` reports a max |d| of **3–6 on values that are O(1)** after
normalisation, with ~99% of cells over `1e-4`. Broken per-frame, flat per-band.

**Cause.** The STFT analysis window is **left-aligned** in the frame instead of centred.
`torch.stft` pads a window shorter than `n_fft` on **both** sides, so a 400-sample window
occupies `[56, 456)` of a 512-point frame, not `[0, 400)`. Left-aligning it shifts the
entire spectrogram by 56 samples — 3.5 mel frames at hop 160.

**Why it is nasty, and why the diagnosis above is the way round:** because the
normalisation is per-band, a *time* shift barely moves a band's *mean* or *standard
deviation*. So the error signature is **per-band flat, per-frame scattered**. If you had
been looking for a per-band offset you would have concluded the front end was fine. The
output is still a plausible log-mel spectrogram and still normalises to mean ~0, so there
is no warning anywhere: no NaN, no shape error, no ORT complaint.

**Fix.** Centre the window, using the same arithmetic PyTorch uses:

```rust
// frontend.rs:252
let window_offset = (params.n_fft.saturating_sub(params.win_length)) / 2;  // (512-400)/2 = 56
```

Regression-tested by `tests/frontend_parity.rs`, which compares **every one of the
128 × 101 cells** against values computed by PyTorch from the same `preproc` tensors
rather than sampling a few: the worst cell is currently 7.99e-05 against a 2e-4 tolerance.
Three to four orders of magnitude is what separates a convention error from float32
rounding, which is why the check has to be exhaustive to mean anything.

---

## 10. Power spectrum off by ~5.7 with a per-band mean of exactly 0

**Symptom.** Log-mel parity is off by about **5.7** — an order of magnitude worse than
failure 9 and much closer to a structural mistake. The diagnostic that identifies it: the
**per-band mean offset is exactly 0** while the **per-frame standard deviation is ~1.2**.

**Cause.** Hermitian symmetry misused when reading the power spectrum. For a real input the
symmetry relates the **conjugate**: `Re(X[n−k]) == Re(X[k])` and `Im(X[n−k]) == −Im(X[k])`.
It is therefore tempting to read the imaginary part from the mirrored bin `n_fft − k` rather
than from bin `k`. That computes `re² + re²` instead of `re² + im²`, which is not the power
spectrum at all. The symmetric part of the spectrum is *reproduced exactly*, which is why
the per-band mean survives untouched while the variance and the whole frame-to-frame
structure are wrong.

**Fix.** Read `Im` from bin `k` itself (`frontend.rs:297-300`):

```rust
for k in 0..params.n_freq {
    let z = spectrum[k];
    power[k] = z.re * z.re + z.im * z.im;
}
```

With `mag_power == 2.0` the reference's `sqrt(...).pow(2.0)` is an exact no-op detour, so
skipping it is bit-comparable, not an approximation. Same failure shape as 9: finite,
normalised, plausible, wrong.

---

## 11. The decode loop runs one frame too far, into the padding

**Symptom.** The encoder's output tensor has one more frame than `enc_lengths` reports, and
the last one is padding. In transcripts: a **trailing empty or hallucinated word**, or a
segment that ends slightly past the end of the audio. Note that the *reported* length is
correct — `frames == ceil(feature_lengths / 8)` still holds, and
`tests/sravaani_transcribe.rs:188-198` asserts exactly that and passes. The bug is
invisible to every length assertion and shows up only as a word at the end.

**Cause.** The log-mel tensor has **`1 + valid_frames`** time columns, because
`torch.stft` with `center = true` emits one more frame than `feature_lengths` describes —
and the encoder subsamples the **tensor**, not `feature_lengths`. So its output has
`ceil((1 + valid_frames) / 8)` columns. For a 40 s chunk: 4000 valid frames, a 4001-column
tensor, a tensor-derived length of `ceil(4001/8) = 501` — and `enc_lengths` reporting the
true **500**. PyTorch's reference decodes exactly `enc_len` frames, so a loop bounded by
the tensor width decodes one spurious frame past the end of the audio.

**Fix.** Use the `enc_lengths` **output**, never the tensor dimension
(`sravaani.rs:503-514`). It is requested for this reason and must not be "optimised away as
redundant with the shape" — it is not redundant, it is the only correct answer:

```rust
let reported = out[1].as_i64()?.data[0];
// ... bounds-check it against enc.shape[2], then decode `reported` frames
```

The law itself is measured, not assumed:
`cargo run --release -p vaani-inference --example subsample_probe` prints
`enc_lengths == ceil(T/8)` for twelve input lengths (200, 400, 437, 800, 1600, 1601, 3200,
4000, 4800, 8000, 16000, 24000).

---

## 12. Works in development, "the speech recognition model was not found" in the app

**Symptom.** Every Rust test, example and probe passes — they read `artifacts/sravaani/`
or whatever `VAANI_ARTEFACTS` points at, and the Hugging Face snapshot is right there in
`SraVaani-1.0/` — and a build that reads its model from somewhere else reports it missing.
Concretely: `EngineRegistry::create` refuses with `model_unavailable` naming
`tokenizer.model (...)` (`engine.rs:299-303`), and `SraVaaniEngine::new` on its own fails at
`model_load_failed` before a single weight is paged in. Nothing about the ONNX graphs is
wrong; 887 MB of encoder sit unused next to a directory with no tokeniser in it.

**Cause.** `tokenizer.model` was never copied into the artefacts directory. It exists in
the Hugging Face snapshot (`SraVaani-1.0/tokenizer.model`), so the exporter can read it and
the developer never notices, but `EngineArtifacts::missing()` requires it **in the artefacts
directory** alongside `preproc.json`, `mel_window.f32`, `mel_fb.f32`, the encoder and the
joint (`engine.rs:197-214`), and `SraVaaniEngine::new` opens it with
`SentencePieceProcessor::open` (`sravaani.rs:209-211`).

**Fix.** The exporter copies it, and must keep doing so
(`export_sravaani_onnx.py:497-501`):

```python
tok = src / "tokenizer.model"
if not tok.is_file():
    raise SystemExit(f"tokenizer.model not found at {tok}")
(out / "tokenizer.model").write_bytes(tok.read_bytes())
```

Treat the artefacts directory as a **unit**: graphs, front-end tables, front-end JSON and
tokeniser travel together, and `EngineArtifacts::missing()` is the check that says so. A
partial copy is not a smaller install, it is a broken one. `scripts/build_doctor.ps1:152-156`
tests for all four of the files the ASR path needs and reports the exporter command when one
is absent.

---

## Also seen, briefly

| Symptom | Cause | Where |
|---|---|---|
| `onnxruntime.dll not found at <path>` | `RuntimeLocation::discover` found nothing; the message names a concrete path, not a bare filename | `runtime.rs:212-217`, wrapped at `:262-265` |
| Output shape is `tensor(float)` when you expected f16, or vice versa | `tensor_to_output` tries f16 **first**, then f32, then i32, then i64. A wrong guess would reinterpret 2-byte samples as 4-byte floats and produce plausible garbage, so it errors instead | `session.rs:586-620` |
| "model has no input named `audio`" | `ensure_input` validates every name against the graph *before* allocating anything | `session.rs:507` |
| Everything is slow and no error is printed | silent CPU fallback — see [DEBUGGING.md](DEBUGGING.md). `ort` reports EP registration failure **only** through `tracing`, and no `tracing_subscriber` is installed yet | `session.rs:365-370` |
| `RequireCuda` ran on the CPU instead of failing | it does not hard-fail, by design and by test | `runtime.rs:380-387`, test `:404-412` |
| `cmake: command not found` from `ort-sys` | should not happen. The `ort` dependency is built with `load-dynamic` and `ort-sys` with `default-features = false`, so there is no C++ build | `Cargo.toml:53-64` |
| "works in dev, model not found in the app" | the artefacts directory is missing a file the exporter used to copy — most often `tokenizer.model` | [§12](#12-works-in-development-the-speech-recognition-model-was-not-found-in-the-app) |

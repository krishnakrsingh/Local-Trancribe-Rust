# Common failures

Eight failures that present as code bugs and are not. Each is: **symptom / cause / fix**.
Every error string here is quoted from a source file in this repository or from a tool
chain the repository pins.

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
(`crates/vaani-inference/src/runtime.rs:189-238`). Do not remove either half:

1. `SetDllDirectoryW` for each provider dir (`runtime.rs:215-219`) — appends the
   directory to the search order for every `LoadLibrary` made *after* the call, which
   also covers libraries loaded lazily during the first inference run.
2. `preload_dir` for each provider dir (`runtime.rs:223-225`) — resolves every DLL now so
   an incomplete install fails with *our* message naming the missing file. A
   `debug`-level `could not preload runtime DLL` (`runtime.rs:285`) names the culprit.

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
`hybrid_laptop_picks_nvidia_not_the_shared_memory_igpu` (`capability.rs:602-623`).

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

## Also seen, briefly

| Symptom | Cause | Where |
|---|---|---|
| `onnxruntime.dll not found at <path>` | `RuntimeLocation::discover` found nothing; the message names a concrete path, not a bare filename | `runtime.rs:192-197` |
| Output shape is `tensor(float)` when you expected f16, or vice versa | `tensor_to_output` tries f16 **first**, then f32, then i32, then i64. A wrong guess would reinterpret 2-byte samples as 4-byte floats and produce plausible garbage, so it errors instead | `session.rs:555-590` |
| "model has no input named `audio`" | `ensure_input` validates every name against the graph *before* allocating anything | `session.rs:476-484` |
| Everything is slow and no error is printed | silent CPU fallback — see [DEBUGGING.md](DEBUGGING.md). `ort` reports EP registration failure **only** through `tracing`, and no `tracing_subscriber` is installed yet | `session.rs:343-346` |
| `RequireCuda` ran on the CPU instead of failing | it does not hard-fail, by design and by test | `runtime.rs:348-355`, test `:373` |
| `cmake: command not found` from `ort-sys` | should not happen. The `ort` dependency is built with `load-dynamic` and `ort-sys` with `default-features = false`, so there is no C++ build | `Cargo.toml:52-63` |

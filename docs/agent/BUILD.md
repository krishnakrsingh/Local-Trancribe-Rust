# Build

## What exists

| Artefact | Path | Produced by | In git |
|---|---|---|---|
| Workspace manifest | `Cargo.toml` | hand-written | yes |
| Toolchain pin | `rust-toolchain.toml` | hand-written | yes |
| Native runtime | `runtime/` — 19 DLLs, 1,279.2 MB | `scripts/fetch_runtime.ps1` | **no** (`.gitignore:19`) |
| Download cache | `runtime/.cache/` — 6 zips, 2,511.5 MB | same script | **no** |
| Model artefacts | `artifacts/sravaani/` — 8 files, 2.66 GB | `scripts/export_sravaani_onnx.py` | **no** (`.gitignore:12`) |
| TorchScript source | `SraVaani-1.0/` — 908 MB `model-asr.fp16.ts` | Hugging Face, gated | **no** (`.gitignore:10`) |
| Checksum pins | `scripts/cudnn.sha256` | hand-written | yes |
| Build outputs | `target/` | cargo | **no** (`.gitignore:55`) |

Scripts that exist: `scripts/fetch_runtime.ps1`, `scripts/export_sravaani_onnx.py`,
`scripts/cudnn.sha256`. That is all of them.

Verified build inputs:

| Pin | Value | Where |
|---|---|---|
| Rust | 1.98.1 (797e8a9bc 2026-08-05) | `rust-toolchain.toml:6` |
| Host triple | `x86_64-pc-windows-msvc` | `rust-toolchain.toml:7` |
| MSVC | 14.44.35207, VS Build Tools 2022 17.14.37710 | dev box |
| Windows SDK | 10.0.26100 | dev box |
| `ort` / `ort-sys` | 2.0.0-rc.13 | `Cargo.toml:52,60`; `Cargo.lock` confirms |
| `windows` | 0.62.2 | `Cargo.lock` |
| `thiserror` | 2.0.21 | `Cargo.lock` |
| `serde` / `serde_json` | 1.0.229 / 1.0.151 | `Cargo.lock` |
| `time` | 0.3.55 | `Cargo.lock` |
| `uuid` | 1.26.1 | `Cargo.lock` |
| `half` | 2.7.1 | `Cargo.lock` |
| `tracing` | 0.1.44 | `Cargo.lock` |
| ONNX Runtime | 1.30.0 (`FileVersion 1.30.0.20260909.4.f2c39fe`) | `fetch_runtime.ps1:50` |
| cuDNN | 9.14.0.64 `_cuda13` (`cudnn64_9.dll FileVersion 9.14.0.64`) | `fetch_runtime.ps1:52` |
| CUDA redist | 13.0.2 → cudart 13.0.96, cublas 13.1.0.3, nvrtc 13.0.88 | `fetch_runtime.ps1:51,140` |
| onnx | 1.23.0 | installed on dev box |
| sentencepiece (Python) | 0.2.2 | installed on dev box |

`rustc` auto-discovers MSVC through the registry. **Do not** prepend
`VC\Auxiliary\Build\vcvars64.bat` to `PATH`; the toolchain finds it.

## Why it exists

The build is deliberately split so that a *shipping* user needs none of it. End users get
prebuilt binaries with the native runtime beside the executable. The Python exporter is a
development-time tool: `export_sravaani_onnx.py:4-8` states it explicitly, and the product
never invokes it. Nothing in `crates/` shells out to Python.

`[workspace.dependencies]` is the single registry for third-party versions
(`Cargo.toml:11-12`). Versions are declared once, with a note, and members reference them
with `{ workspace = true }`. Only the crates that are actually members appear in
`Cargo.lock` — the ~30 planned dependencies pinned at `Cargo.toml:28-82` are declarations
of intent, not resolved versions, and are not a source of truth until a member uses one.

## Cold setup, exact order

Order matters: `cargo test --workspace` needs the runtime *and* the artefacts, and the
artefacts need the TorchScript checkpoint.

### 1. Visual Studio Build Tools 2022

Install workload **MSVC v143 — VS 2022 C++ x64/x86 build tools** (`VCTools`), plus a
Windows 10/11 SDK (10.0.26100). rustc discovers it via the registry. Without the VCTools
workload you get a link error, not a compile error — see
[COMMON_FAILURES §7](COMMON_FAILURES.md).

### 2. rustup

```powershell
winget install Rustlang.Rustup
```

`cargo` lands at `$env:USERPROFILE\.cargo\bin`. **After a `--no-modify-path` install it is
not on `PATH`.** For the current session:

```powershell
$env:Path += ";$env:USERPROFILE\.cargo\bin"
cargo --version    # expect: cargo 1.98.1 (797e8a9bc 2026-08-05)
```

To persist it: `[Environment]::SetEnvironmentVariable('Path', "$env:USERPROFILE\.cargo\bin;$([Environment]::GetEnvironmentVariable('Path','User'))", 'User')`

`rust-toolchain.toml` pins 1.98.1, so `cargo` in this repo will not use your default
toolchain. Do not edit the pin to make a build pass.

### 3. Native runtime

```powershell
powershell -ExecutionPolicy Bypass -File scripts\fetch_runtime.ps1
```

Downloads to `runtime/` in a **flat layout** (all DLLs beside each other), verifies
SHA-256 on every archive, and writes `runtime/runtime-manifest.json`. ~1.25 GiB of DLLs
plus ~2.5 GiB of cached zips. Takes several minutes.

The flat layout is required, not cosmetic: the ORT CUDA provider resolves cudart/cublas/
cudnn through the standard Windows search order, which starts at the executable's own
directory. `Expand-To` at `fetch_runtime.ps1:110-116` strips the archive's top directory
and copies DLLs flat for exactly this reason. The installer and the release archive both
assume it.

### 4. Python exporter dependencies

Development-time only. Python 3.13.15 on the dev box.

```powershell
python -m pip install onnx onnxruntime onnxconverter-common sentencepiece
```

Plus `torch` (2.13.0, per the graphs' `producer_version`). `numpy` arrives with `onnx`.

### 5. Export the ONNX artefacts

```powershell
python scripts\export_sravaani_onnx.py --src SraVaani-1.0 --out artifacts\sravaani
```

Needs the gated `SraVaani-1.0/` directory present. `--skip-fp16` exists to debug the fp32
path but produces an install with **no production encoder**. The run needs roughly 3 GB of
free RAM to hold the fp32 and fp16 encoder graphs simultaneously.

Writes 8 files and `artefacts.json`. See [MODELS.md](MODELS.md) for what each one is.

### 6. Build and test

```powershell
cargo test --workspace
cargo clippy --workspace --all-targets
```

Expected: `41 passed` (vaani-core), `31 passed` (vaani-inference lib), `2 passed`
(vaani-inference integration, ~14 s — the second one loads the 887 MB model onto the GPU).

## What must not change

| Do not | Why |
|---|---|
| Un-pin a version in `fetch_runtime.ps1` | `$OrtVersion`, `$CudaRedistVersion`, `$CudnnVersion` are hard pins at `:50-52`; a different ORT build means a different set of CUDA kernels and therefore different timings |
| Bypass `Assert-Hash` or add a download without a pinned SHA-256 | it throws on a missing pin as well as a mismatch (`:75`). A floating version in a fetch script is how a build silently changes under you |
| Use a different cuDNN than 9.14.0.64 | 9.14.0.64 is the version ORT 1.30 was built and CI'd against. See [COMMON_FAILURES §8](COMMON_FAILURES.md) |
| Edit `rust-toolchain.toml` to fix a build | the pin is what makes the dev box and CI agree |
| Add a version to a member `Cargo.toml` directly | it goes in `[workspace.dependencies]` with a note |
| Add two crates to `[workspace] members` at once | `Cargo.toml:7-9`: members are added one at a time so a `cargo check` failure points at a single new crate |
| `cargo vendor` or check `onnxruntime.dll` into the tree | it is a 16.9 MB binary with its own licence (`runtime/LICENSE`) |
| Put `external_data=True` on the fp16 export | 887 MB is under the 2 GB protobuf limit, so a single self-contained file is possible and preferred: no sidecar to lose, and ORT can mmap it (`export_sravaani_onnx.py:359-361`) |

## How to test it

| Check | Command | Pass condition |
|---|---|---|
| Toolchain resolves | `cargo --version` | `1.98.1` |
| Runtime present | `cargo test -p vaani-inference --test runtime_smoke -- --nocapture` | prints `missing: []` |
| Runtime found where expected | `Test-Path runtime\onnxruntime.dll` | `True` |
| GPU path live | same command | `active: Cuda` and `enc_lengths [200]` |
| Both encoders run | `cargo run -p vaani-inference --example conv_probe` | two `OK` lines, no `FAIL` |
| Lints | `cargo clippy --workspace --all-targets` | no output |
| Release profile | `cargo build --release -p vaani-inference` | `lto="thin"`, `codegen-units=1` (`Cargo.toml:106-107`) |

## What it assumes

- **A gated Hugging Face token is available** before step 5. The repository cannot produce
  `SraVaani-1.0/` by itself; see [MODELS.md](MODELS.md).
- **The CUDA Toolkit is NOT a build or runtime dependency.** CUDA 13.3 is installed on the
  dev box and is irrelevant to the product. `capability.rs` never links cudart — it
  `LoadLibraryW`s `cudart64_13.dll` / `cudart64_12.dll` / `cudart64.dll` and degrades to
  "unknown" if absent. `fetch_runtime.ps1:15-19` states this: the redistributables ship
  runtime libraries only, no `nvcc`.
- **`scripts/build_doctor.ps1` does not exist**, despite the comment at `Cargo.toml:9`
  referring to it. `vaani doctor` and `vaani --debug` are also planned, not built — see
  [DEBUGGING.md](DEBUGGING.md).
- **`runtime/runtime-manifest.json` is not present on this machine.** The script writes it
  at `fetch_runtime.ps1:170`, so the last run did not reach the final step. Do not assume
  its existence; `Assert-Hash` output during the fetch is the current record.
- **A 4 GB laptop GPU is the design target.** The fp16 encoder is 887 MB and the measured
  free DXGI budget on the reference machine is 3,531,289,396 bytes. The 256 MiB reserve
  subtracted from free VRAM (`tests/runtime_smoke.rs:83`, `examples/conv_probe.rs:64`) is
  the desktop compositor's margin.

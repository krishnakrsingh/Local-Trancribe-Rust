# Build

## What exists

| Artefact | Path | Produced by | In git |
|---|---|---|---|
| Workspace manifest | `Cargo.toml` | hand-written | yes |
| Toolchain pin | `rust-toolchain.toml` | hand-written | yes |
| Native runtime | `runtime/` — 19 DLLs, 1,279.2 MB | `scripts/fetch_runtime.ps1` | **no** (`.gitignore:19`) |
| Download cache | `runtime/.cache/` — 6 zips, 2,511.5 MB | same script | **no** |
| Model artefacts | `artifacts/sravaani/` — 10 files, 2,702,792,388 B | `scripts/export_sravaani_onnx.py` | **no** (`.gitignore:12`) |
| TorchScript source | `SraVaani-1.0/` — 908 MB `model-asr.fp16.ts` | Hugging Face, gated | **no** (`.gitignore:10`) |
| Test fixtures | `tests/fixtures/sravaani/` — 2 JSON files | `scripts/make_golden.py` | yes |
| Checksum pins | `scripts/cudnn.sha256` | hand-written | yes |
| Build outputs | `target/` | cargo | **no** (`.gitignore:56`) |

Scripts that exist, all of them:

| Script | What it does | Needs |
|---|---|---|
| `scripts/fetch_runtime.ps1` | pinned download + SHA-256 verification of ONNX Runtime, CUDA redist and cuDNN into `runtime/` | network |
| `scripts/export_sravaani_onnx.py` | TorchScript → the three ONNX graphs + `preproc.json`, `mel_window.f32`, `mel_fb.f32`, `config.json`, `tokenizer.model`, `model_meta.json`, `artefacts.json` | `SraVaani-1.0/`, Python, ~3 GB RAM, ~10 min |
| `scripts/make_golden.py` | the two golden fixtures the Rust parity tests are gated on | `SraVaani-1.0/`, Python, a real WAV for the decode fixture |
| `scripts/build_doctor.ps1` | the eight-stage health gate; the recommended single command | nothing but cargo |
| `scripts/cudnn.sha256` | the one pinned hash, for cuDNN | — |

Verified build inputs:

| Pin | Value | Where |
|---|---|---|
| Rust | 1.98.1 (797e8a9bc 2026-08-05) | `rust-toolchain.toml:6` |
| Host triple | `x86_64-pc-windows-msvc` | `rust-toolchain.toml:7` |
| MSVC | 14.44.35207, VS Build Tools 2022 17.14.37710 | dev box |
| Windows SDK | 10.0.26100 | dev box |
| `ort` / `ort-sys` | 2.0.0-rc.13 | `Cargo.toml:53,61`; `Cargo.lock` confirms |
| `windows` | 0.62.2 | `Cargo.lock` |
| `thiserror` | 2.0.21 | `Cargo.lock` |
| `serde` / `serde_json` | 1.0.229 / 1.0.151 | `Cargo.lock` |
| `time` | 0.3.55 | `Cargo.lock` |
| `uuid` | 1.26.1 | `Cargo.lock` |
| `half` | 2.7.1 | `Cargo.lock` |
| `tracing` | 0.1.44 | `Cargo.lock` |
| `rustfft` | 6.4.1 | `Cargo.lock`, used by `vaani-asr::frontend` |
| `sentencepiece-rs` | 0.2.2 | `Cargo.lock`, used by `vaani-asr::sravaani` |
| ONNX Runtime | 1.30.0 (`FileVersion 1.30.0.20260909.4.f2c39fe`) | `fetch_runtime.ps1:50` |
| cuDNN | 9.14.0.64 `_cuda13` (`cudnn64_9.dll FileVersion 9.14.0.64`) | `fetch_runtime.ps1:52` |
| CUDA redist | 13.0.2 → cudart 13.0.96, cublas 13.1.0.3, nvrtc 13.0.88 | `fetch_runtime.ps1:51,140` |
| onnx | 1.23.0 | installed on dev box |
| torch | 2.13.0 (`+cu130`), the graphs' `producer_version` | installed on dev box |
| transformers | 5.15.0 | `make_golden.py` loads the HF wrapper with `trust_remote_code` |
| soundfile | 0.14.0 | `make_golden.py` reads the decode WAV |
| sentencepiece (Python) | 0.2.2 | reference decode for the fixture |
| av (PyAV) | 18.1.0 | extracting the 16 kHz mono WAV for the decode fixture |
| numpy | 2.5.2 | arrives with `onnx`; used by both Python scripts |

`rustc` auto-discovers MSVC through the registry. **Do not** prepend
`VC\Auxiliary\Build\vcvars64.bat` to `PATH`; the toolchain finds it.

## The one command

```powershell
$env:Path += ";$env:USERPROFILE\.cargo\bin"
powershell -ExecutionPolicy Bypass -File scripts\build_doctor.ps1
```

`scripts/build_doctor.ps1` is the recommended way to answer "does this tree work?". It runs
the whole gate cheapest-stage-first and prints one line per stage; `-Json` emits it
machine-readably and the exit code is the number of failed stages (capped at 125).

| # | Stage | What it runs | Fails when |
|---:|---|---|---|
| 1 | `toolchain` | `cargo --version` against the pin in `rust-toolchain.toml` | cargo is absent, or the pin cannot be read |
| 1b | `link` | builds and runs a throwaway `fn main() { println!("ok") }` crate | MSVC or the Windows SDK is missing — `cargo check` never calls the linker, so this is the only stage that catches it |
| 2 | `fmt` | `cargo fmt --all --check` | formatting drift |
| 3 | `check` | `cargo check --workspace --all-targets` | anything does not compile, examples and tests included |
| 4 | `clippy` | `cargo clippy --workspace --all-targets -- -D warnings` | any lint, including in tests and examples |
| 5 | `test` | `cargo test --workspace` | any test fails; a *skipped* test still passes, which is why stages 6–8 exist |
| 6 | `runtime` | `cargo test -p vaani-inference --test runtime_smoke` | the ONNX Runtime smoke test fails; `SKIP` if `runtime/` is absent, naming `fetch_runtime.ps1` |
| 7 | `model` | presence of `encoder.fp16.onnx`, `decoder_joint.onnx`, `preproc.json`, `tokenizer.model` | `SKIP` if any is absent, naming the exporter command |
| 8 | `golden` | `cargo test -p vaani-asr --test frontend_parity --test sravaani_transcribe -- --nocapture --test-threads=1`, with `VAANI_GOLDEN_RAW` pointed at `scratch\golden_clip.f32` | front-end or decode parity fails; `SKIP` if a fixture or the raw clip is missing |

Flags: `-Quick` stops after stage 5 (for CI, where 6–8 have their own jobs); `-Release` adds
`--release`, and is **required** for any stage whose timing you intend to quote — the same
rule as [PERFORMANCE.md](PERFORMANCE.md). Stages 6–8 never silently pass: each reports `SKIP`
with the command that would satisfy it. Stage 8 is also the only thing in the repository that
sets `VAANI_GOLDEN_RAW` for you, which is why the strict decode-parity test runs in the
doctor and not by default in `cargo test`.

## Why it exists

The build is deliberately split so that a *shipping* user needs none of it. End users get
prebuilt binaries with the native runtime beside the executable. The Python scripts are
development-time tools: `export_sravaani_onnx.py:4-8` and `make_golden.py:3-5` both state it
explicitly, and the product never invokes either. Nothing in `crates/` shells out to Python.

The fixtures exist because the ASR path has two places where a mistake produces a
*plausible* transcript rather than an error — the log-mel conventions and the TDT decode
semantics. A unit test cannot see either, because both produce finite, correctly-shaped,
language-looking output. A fixture computed by PyTorch and by the reference decode loop can.
See [ARCHITECTURE.md](ARCHITECTURE.md) and [COMMON_FAILURES §9-11](COMMON_FAILURES.md).

`[workspace.dependencies]` is the single registry for third-party versions
(`Cargo.toml:14`). Versions are declared once, with a note, and members reference them
with `{ workspace = true }`. Only the crates that are actually members appear in
`Cargo.lock` — the ~30 planned dependencies pinned at `Cargo.toml:29-83` are declarations
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
python -m pip install onnx onnxruntime onnxconverter-common sentencepiece transformers soundfile av
```

Plus `torch` (2.13.0, per the graphs' `producer_version`). `numpy` arrives with `onnx`.
`transformers`, `soundfile` and `av` are needed by `make_golden.py` in step 6, not by the
exporter; install them together so the fixture generator is not discovered to be broken
halfway through a setup.

### 5. Export the ONNX artefacts

```powershell
python scripts\export_sravaani_onnx.py --src SraVaani-1.0 --out artifacts\sravaani
```

Needs the gated `SraVaani-1.0/` directory present. `--skip-fp16` exists to debug the fp32
path but produces an install with **no production encoder**. The run needs roughly 3 GB of
free RAM to hold the fp32 and fp16 encoder graphs simultaneously.

Writes 9 files plus `artefacts.json`, including `tokenizer.model` — the tokeniser is part of
the install, not an optional extra; see [COMMON_FAILURES §12](COMMON_FAILURES.md). See
[MODELS.md](MODELS.md) for what each one is.

### 6. Generate the golden fixtures

The two fixtures in `tests/fixtures/sravaani/` are what the Rust parity tests are gated on.
They come from PyTorch and from the reference TDT loop, so they must be generated after the
model is present — and the **decode** fixture needs real speech.

The front-end fixture does not need audio: it uses a deterministic synthetic signal.

```powershell
python scripts\make_golden.py --src SraVaani-1.0 --out tests\fixtures
```

The decode fixture does. Extract a **16 kHz mono** WAV from a real recording first — with
PyAV, so the resample and downmix happen once, in one place:

```powershell
python -c "import av, numpy as np, soundfile as sf; c=av.open(r'<recording>.mp4'); \
r=av.audio.resampler.AudioResampler(format='s16', layout='mono', rate=16000); \
ch=[f.to_ndarray().reshape(-1) for fr in c.decode(audio=0) for f in r.resample(fr)]; \
x=(np.concatenate(ch)[:640000].astype(np.float32)/32768.0); \
sf.write('scratch/clip.wav', x, 16000, subtype='PCM_16'); print(len(x))"
```

40 s is 640,000 samples at 16 kHz and is what the committed fixture was made from. Then:

```powershell
python scripts\make_golden.py --src SraVaani-1.0 --out tests\fixtures `
  --decode-audio scratch\clip.wav --decode-seconds 40
```

That writes `decode_golden.json` **and** `scratch\golden_clip.f32`, a raw little-endian f32
copy of the same 40 s. The raw file is real meeting audio: it is written to `scratch/`, which
is gitignored (`.gitignore:48`), and must never be committed. It exists because
`VAANI_GOLDEN_RAW` points at it and that is what turns on the strict decode-parity test — see
[DEBUGGING.md](DEBUGGING.md).

**The committed fixture contains token ids and timings only, never audio.** That is
deliberate and is the whole privacy position of the test suite: `decode_golden.json` records
`token_ids`, `token_frames`, `text` and per-word `start`/`end`, and the WAV stays on the
machine that made it. Anyone can regenerate the fixture from their own clip and get the same
kind of check. `frontend_golden.json` additionally contains a synthetic signal, so it is safe
to commit by construction.

`make_golden.py` refuses to write the front-end fixture if its own longhand transcription of
the front end disagrees with the model's `extract_features` by more than 1e-4
(`make_golden.py:164-172`) — a disagreement means one of the two is wrong and the fixture must
not be baked.

### 7. Build and test

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_doctor.ps1
```

or, spelled out, the same stages by hand:

```powershell
cargo test --workspace
cargo clippy --workspace --all-targets
```

Expected from `cargo test --release --workspace`, measured 2026-09-27: **111 passed, 0
failed** — 41 `vaani-core`, 31 `vaani-inference` lib, 2 `vaani-inference` integration,
28 `vaani-asr` lib, 3 `frontend_parity`, 6 `sravaani_transcribe`. The integration tests are
the slow ones (they load the 887 MB encoder onto the GPU); everything else is instant. On a
machine without `runtime/` or `artifacts/` the last three report `skipping:` and still pass —
read the printed reason.

## What must not change

| Do not | Why |
|---|---|
| Un-pin a version in `fetch_runtime.ps1` | `$OrtVersion`, `$CudaRedistVersion`, `$CudnnVersion` are hard pins at `:50-52`; a different ORT build means a different set of CUDA kernels and therefore different timings |
| Bypass `Assert-Hash` or add a download without a pinned SHA-256 | it throws on a missing pin as well as a mismatch (`:75`). A floating version in a fetch script is how a build silently changes under you |
| Use a different cuDNN than 9.14.0.64 | 9.14.0.64 is the version ORT 1.30 was built and CI'd against. See [COMMON_FAILURES §8](COMMON_FAILURES.md) |
| Edit `rust-toolchain.toml` to fix a build | the pin is what makes the dev box and CI agree |
| Add a version to a member `Cargo.toml` directly | it goes in `[workspace.dependencies]` with a note |
| Add two crates to `[workspace] members` at once | `Cargo.toml:6-10`: members are added one at a time so a `cargo check` failure points at a single new crate. See [ADR-002](../decisions/ADR-002-rust-workspace-structure.md) |
| `cargo vendor` or check `onnxruntime.dll` into the tree | it is a 16.9 MB binary with its own licence (`runtime/LICENSE`) |
| Put `external_data=True` on the fp16 export | 887 MB is under the 2 GB protobuf limit, so a single self-contained file is possible and preferred: no sidecar to lose, and ORT can mmap it (`export_sravaani_onnx.py:359-361`) |
| Commit `scratch/`, or any `.wav`/`.mp4`, to satisfy a fixture | the decode fixture is ids and timings; the audio stays local. `.gitignore:22-51` |
| Regenerate `decode_golden.json` from synthetic audio | it exists to catch decode-semantics drift, and only real speech emits tokens |
| Skip stage 8 of `build_doctor.ps1` because stages 1–5 were green | stages 1–5 say the code compiles; only stage 8 says the model still produces the reference transcript |

## How to test it

| Check | Command | Pass condition |
|---|---|---|
| Everything at once | `powershell -ExecutionPolicy Bypass -File scripts\build_doctor.ps1` | `healthy`, or `N skipped` with each skip naming its command |
| Toolchain resolves | `cargo --version` | `1.98.1` |
| Runtime present | `cargo test -p vaani-inference --test runtime_smoke -- --nocapture` | prints `missing: []` |
| Runtime found where expected | `Test-Path runtime\onnxruntime.dll` | `True` |
| GPU path live | same command | `active: Cuda` and `enc_lengths [200]` |
| Both encoders run | `cargo run -p vaani-inference --example conv_probe` | two `OK` lines, no `FAIL` |
| Subsampling law | `cargo run -p vaani-inference --example subsample_probe` | `enc_lengths == ceil(T/8)` on every row |
| Front end matches PyTorch | `cargo test -p vaani-asr --test frontend_parity -- --nocapture` | `max |rust - reference| = 7.987e-5`, limit 2e-4 |
| Decode matches the reference | `build_doctor.ps1` stage 8, or `VAANI_GOLDEN_RAW` + `matches_reference_token_ids_on_real_audio` | `202/202` token ids, `108/108` words, `EXACT MATCH` |
| Lints | `cargo clippy --workspace --all-targets -- -D warnings` | no output |
| Formatting | `cargo fmt --all --check` | no output |
| Release profile | `Select-String -Path Cargo.toml -Pattern 'lto','codegen-units'` | `lto="thin"`, `codegen-units=1` (`Cargo.toml:107-108`) |

## What it assumes

- **A gated Hugging Face token is available** before step 5. The repository cannot produce
  `SraVaani-1.0/` by itself; see [MODELS.md](MODELS.md).
- **The CUDA Toolkit is NOT a build or runtime dependency.** CUDA 13.3 is installed on the
  dev box and is irrelevant to the product. `capability.rs` never links cudart — it
  `LoadLibraryW`s `cudart64_13.dll` / `cudart64_12.dll` / `cudart64.dll` and degrades to
  "unknown" if absent. `fetch_runtime.ps1:15-19` states this: the redistributables ship
  runtime libraries only, no `nvcc`.
- **`vaani doctor` and `vaani --debug` are still planned, not built.** No CLI crate exists.
  `scripts/build_doctor.ps1` is a developer script and is *not* the `vaani doctor` command
  the requirements describe; it exists because that command cannot be written yet. See
  [DEBUGGING.md](DEBUGGING.md) and [RELEASE.md](RELEASE.md).
- **`runtime/runtime-manifest.json` is not present on this machine.** The script writes it
  at `fetch_runtime.ps1:170`, so the last run did not reach the final step. Do not assume
  its existence; `Assert-Hash` output during the fetch is the current record.
- **The committed golden fixtures are only as good as the clip they came from.** They were
  generated from one 40 s Hindi meeting recording on the dev machine. Nothing about the code
  depends on that recording; a different clip produces a different fixture and the same kind
  of check.
- **A 4 GB laptop GPU is the design target.** The fp16 encoder is 887 MB and the measured
  free DXGI budget on the reference machine is 3,531,289,396 bytes. The 256 MiB reserve
  subtracted from free VRAM (`tests/runtime_smoke.rs:94`, `examples/conv_probe.rs:74`) is
  the desktop compositor's margin.

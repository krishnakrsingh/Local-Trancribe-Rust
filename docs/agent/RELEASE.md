# Release

## What exists

| Thing | State |
|---|---|
| A binary to ship | **none.** No `vaani.exe`, no `vaani-cli.exe`, no `vaani` anything |
| Version stamp | `APP_VERSION` = `env!("CARGO_PKG_VERSION")` = `0.1.0`, from `crates/vaani-core/Cargo.toml:3` |
| Data model version | `DATA_MODEL_VERSION: u32 = 1` (`crates/vaani-core/src/lib.rs:57`) |
| Release profile | `[profile.release]` in `Cargo.toml:104-109` |
| CI | **none.** No `.github/`, no pipeline, no test job |
| Installer | **none.** No WiX, no MSIX, no NSIS |
| Signing | **none.** No certificate, no `signtool` invocation |
| Update mechanism | **planned** — `vaani-updater` does not exist |
| Publish target | **planned** — GitHub Releases ([ADR-007](../decisions/ADR-007-github-releases-distribution.md)) and WinGet ([ADR-008](../decisions/ADR-008-winget-distribution.md)) |
| `scripts/build_doctor.ps1` | **exists** — the developer health gate, eight stages, `-Quick` / `-Release` / `-Json`. It is **not** the `vaani doctor` command, which still needs a CLI crate |
| `runtime/runtime-manifest.json` | **not present** on this machine; written by `fetch_runtime.ps1:170` |

In other words: release is entirely unbuilt. This document records the constraints that
already exist in code, because those are what a release process will have to satisfy.

## Why it exists

The constraints below are not bureaucracy; each one encodes a decision made earlier.

| Constraint | Reason | Where |
|---|---|---|
| Runtime DLLs ship **flat**, beside the executable | cuDNN 9's dispatcher `LoadLibrary`s its siblings by bare name, and the Windows search order starts at the executable's directory | `fetch_runtime.ps1:110-116`, failure 1 in [COMMON_FAILURES](COMMON_FAILURES.md) |
| `VAANI_RUNTIME_DIR` is an override, not the mechanism | it is for the test suite and for running from a build directory. The shipped product uses `<exe>/runtime` | `runtime.rs:107-144` |
| `RuntimeReport::collect()` never returns `Err` | a diagnostics command must be able to say what is broken without itself crashing | `lib.rs:61-75` |
| `Provider::Auto` silently falls back to CPU | an unavailable GPU is a normal outcome, not an error; the resulting `ProviderStatus` is what the UI and the benchmark report display | `lib.rs:30-37` |
| `Provider::RequireCuda` also falls back | a transcript is worth more than a benchmark purity guarantee; the status records the mismatch for the benchmark to flag | `runtime.rs:348-355` |
| `APP_VERSION` lives in `vaani-core` | so the CLI, the database and the exported JSON all agree without hard-coding it | `crates/vaani-core/src/lib.rs:47-53` |
| `DATA_MODEL_VERSION` is separate from `APP_VERSION` | an app 1.0.0 can run SraVaani 2.0 with a different diarization engine; exported transcripts record the *models*, not the app | `crates/vaani-core/src/lib.rs:54-57` |
| `EngineStamp` is persisted on every transcript | a transcript exported months later must say which model produced it | `transcript.rs:274-276, 283-292` |
| Weights are never in the installer | SraVaani-1.0 is MIT **but gated** on Hugging Face; the Model Manager fetches on the user's machine | `.gitignore:1-16`, [MODELS.md](MODELS.md) |

## Release profile, verbatim

`Cargo.toml:105-116`:

```toml
[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
panic = "unwind"
strip = "debuginfo"

[profile.bench]
opt-level = 3
lto = "thin"
debug = 1
```

| Setting | Note |
|---|---|
| `lto = "thin"`, `codegen-units = 1` | whole-program optimisation; what `codegen-units = 1` costs in build time is what it buys in the inference hot path |
| `panic = "unwind"` | **deliberately not `abort`**. Cancellation and job-resume rely on unwinding; `abort` would turn every `?` in a worker into a process kill |
| `strip = "debuginfo"` | drops symbols but keeps PDBs. Do not set `strip = "symbols"` — crash reports need the symbols |
| `[profile.bench]` keeps `debug = 1` | benchmarks must be reproducible, which needs the debug info |

Two binaries are planned, and they need different `[[bin]]` `windows_subsystem` settings
in the same package: a GUI `vaani.exe` with `windows_subsystem = "windows"` and a console
`vaani-cli.exe` with the default. Neither exists.

## What must not change

| Do not | Why |
|---|---|
| Ship without `SetDllDirectoryW` + preloads in `ensure_initialised` | acceleration breaks on every machine. See [COMMON_FAILURES §1](COMMON_FAILURES.md) |
| Ship a cuDNN other than 9.14.0.64, or an ONNX Runtime other than 1.30.0 | unsupported pairing; see [COMMON_FAILURES §8](COMMON_FAILURES.md) |
| Ship model weights in the installer | licence gate; and 2.70 GB of artefacts that a 4 GB laptop may not have disk for |
| Ship the CUDA Toolkit | the redistributables are runtime-only, no `nvcc`. `fetch_runtime.ps1:15-19` |
| Ship the `ort` crate's CDN ORT build | it ships `75;80;90;120` and forces PTX JIT; Microsoft's `86-real` build has native SASS for sm_86 ([ADR-003](../decisions/ADR-003-onnx-runtime.md)) |
| Set `panic = "abort"` | breaks cancellation |
| Rename an `ErrorCode` or a `JobPhase` wire string | both are persisted; a rename breaks old rows and old exports |
| Bump `DATA_MODEL_VERSION` without migrating | exported JSON consumers use it to tell a new field from a removed one |
| Use the `ort` crate's default (static) build | it makes a build-time network dependency and imposes an `x86-64-v3` baseline that would refuse to run on older end-user laptops (`runtime.rs:3-18`) |

## How to test a release

There is no release to test yet. The gate a candidate must pass before it can be called
one:

```powershell
$env:Path += ";$env:USERPROFILE\.cargo\bin"

# 1. Clean rebuild in release mode, no warnings.
cargo build --release --workspace
cargo clippy --release --workspace --all-targets

# 2. The full suite. The integration tests must actually load the model,
#    not skip. A green run that printed "SKIP" proves nothing.
cargo test --release --workspace -- --nocapture
#    111 tests. The load-bearing ones are `frontend_parity` (log-mel vs PyTorch)
#    and `sravaani_transcribe` (decode vs the reference fixture).

# 3. Confirm the runtime resolves from the *executable's* directory, not
#    from the repo. This is the check that matters for a shipped build.
.\target\release\<binary>.exe doctor     # planned; today, use:
Test-Path target\release\runtime\onnxruntime.dll

# 4. Confirm the flat layout survived packaging.
(Get-ChildItem target\release\runtime -Filter *.dll).Count   # expect 19
```

Step 2 must print, not skip:

```
provider: ProviderStatus { requested: Auto, active: Cuda, note: Some("Using the NVIDIA GPU.") }
enc_out shape [1, 1024, 200], enc_lengths [200]
```

If it prints `SKIP: artifacts/sravaani/encoder.fp16.onnx not present` the test passed
without testing anything. `runtime_smoke.rs` skips loudly and by design
(`tests/runtime_smoke.rs:49,65,70`).

## What a release process will have to do

Not implemented. Listed so the gaps are explicit, not so they can be skipped.

1. **Create `vaani-cli` and `vaani-app`.** Until then there is no `--version`, no
   `doctor`, and no `--debug`. `RuntimeReport` is the payload for all three
   ([DEBUGGING.md](DEBUGGING.md)). `scripts/build_doctor.ps1` is the developer-time
   substitute and covers everything except the binary's own arguments.
2. **Install a `tracing_subscriber`.** Declared in `[workspace.dependencies]`
   (`Cargo.toml:35`) but not a dependency of any member. Without it ORT's EP registration
   failures are invisible, and a shipped build can fail to accelerate with no diagnostic.
3. **Add CI.** At minimum: `cargo clippy --workspace --all-targets` warning-free,
   `cargo test --workspace`, and a check that `runtime/` and `artifacts/` are still
   gitignored. `scripts/build_doctor.ps1 -Quick` is the CI-shaped gate; stages 6–8 need a
   machine with the runtime and the model, and `cargo test` on a machine without them
   degrades to a no-op that still reports green. The parity stage must not be allowed to
   report `SKIP` in CI — that is the stage that says the model still transcribes correctly.
4. **Produce `runtime/runtime-manifest.json`.** `fetch_runtime.ps1:167-170` writes a
   per-DLL `{name, bytes, sha256}` manifest; the script's own header comment at `:23`
   incorrectly says the hashes are "pinned in `.\runtime\manifest.json`" — they are
   written there, not pinned there, and the only *pinned* hash file is
   `scripts/cudnn.sha256`. Fix that comment.
5. **Clean `runtime/.cache/`.** It currently holds 2,511.5 MB of zips, including a
   cuDNN **9.26.0.51** archive (1.28 GB) that is downloaded but not installed. It must
   never be shipped.
6. **Regenerate `artifacts/sravaani/artefacts.json`.** The current one is stale — see the
   report. Shipping against stale checksums means the app verifies a file it does not
   have. Note the directory now holds **10** files including `tokenizer.model`; the
   manifest must cover the tokeniser too, or the install is broken in a way that only
   shows up at the first decode ([COMMON_FAILURES §12](COMMON_FAILURES.md)).
7. **Publish.** GitHub Releases for the portable ZIP, WinGet for the installer
   ([ADR-007](../decisions/ADR-007-github-releases-distribution.md),
   [ADR-008](../decisions/ADR-008-winget-distribution.md)).
8. **Verify the download hash after publishing.** The ORT zip's SHA-256
   `8fa4b08359af682cd605892cb59077049700b640128bb93fd2c7776cf9f55bdc` is a first-party
   value from the GitHub releases API, not one we computed from a file we already had.

## What it assumes

- **Every benchmark number in [PERFORMANCE.md](PERFORMANCE.md) was taken with `--release`**,
  on the one machine described there. A number from a `dev` build, or from another
  machine, is not comparable and must not be reused.
- **cuDNN 9.14.0.64 and ONNX Runtime 1.30.0 stay the pinned pairing.** If either moves,
  re-measure everything and update [PERFORMANCE.md](PERFORMANCE.md) in the same change.
- **`runtime/` is gitignored and therefore not in any clone.** A fresh clone has no
  runtime, no artefacts and no models, so `cargo test` on a fresh clone will report
  `SKIP` for both integration tests. That is correct behaviour, not a regression — read
  the printed reason.
- **The one machine measured is a hybrid laptop with a 3.3 GB DXGI budget.** A different
  VRAM budget changes the `gpu_mem_limit` (`free VRAM − 256 MiB`) and therefore the
  OOM-vs-fits decision, which `vram_fits` tries to answer before load.
- **`SraVaani-1.0` access can still be granted.** If the upstream gate or licence changes,
  the model manifest's `access` and `license` fields must be re-verified before anything
  derived from the weights is redistributed.

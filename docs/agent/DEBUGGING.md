# Debugging

## What exists

Three real diagnostic surfaces, all in `vaani-inference`:

| Surface | What it tells you | Command |
|---|---|---|
| `RuntimeReport` | which `onnxruntime.dll` was resolved, which DLLs are missing, every DXGI adapter, the CUDA probe, CPU counts, RAM, the ORT thread count | `cargo test -p vaani-inference --test runtime_smoke -- --nocapture` |
| `conv_probe` example | whether the fp16 and fp32 encoders both run on the CUDA EP, and how fast | `cargo run -p vaani-inference --example conv_probe` |
| `tracing` events | session load, ORT init, preload failures | `RUST_LOG` — see below |

`RuntimeReport::collect()` **never fails** (`crates/vaani-inference/src/lib.rs:65`). A
missing runtime is a fact to report, not an error to propagate, because the diagnostics
command has to be able to say what is wrong without itself crashing. That is tested at
`lib.rs:110`.

## Why it exists

The dominant failure mode of this product is **silent CPU fallback**: the session loads,
produces correct output, and is 20–40× slower. Nothing errors. A capability panel that
guesses is worse than one that says it does not know, so the guiding rule in
`capability.rs:19-21` is *report what we measured, `Unknown` when we did not*.

That rule is why `CudaProbe` carries an `error: Option<String>` and why
`ProviderStatus` carries a `note`. Every silent path in this codebase has a field that
records why it was taken.

## `tracing` and the silent-fallback trap

**`ort` reports execution-provider registration failures only through `tracing`.**

`session.rs:346` builds the CUDA EP with `.fail_silently()`:

```rust
builder = builder
    .with_execution_providers([ep.build().fail_silently()])
```

If the provider refuses to register — cuDNN missing, DLL search path wrong, driver too
old, arena cap too small — every node runs on the built-in CPU EP, the session still
loads, the output is still correct, and the run is mysteriously slow. ORT says so through
`tracing` and nowhere else.

**Without a `tracing_subscriber` installed, none of that reaches a terminal.** The
`tracing` crate has no default sink. `tracing-subscriber` is declared in
`[workspace.dependencies]` (`Cargo.toml:34`) but is **not a dependency of any member
crate yet**, so today the EP registration failure is genuinely invisible in a normal
`cargo test` run. This is the single most important thing to know when a run is slower
than it should be.

### Installing a subscriber to see what ORT says

```powershell
# Everything vaani-inference emits, plus ORT's own tracing events.
$env:RUST_LOG = "vaani_inference=trace,ort=trace"
cargo test -p vaani-inference --test runtime_smoke -- --nocapture

# Narrow it to the EP registration decision only.
$env:RUST_LOG = "ort=trace"
```

`ort` 2.0.0-rc.13 is built with the `tracing` feature (`Cargo.toml:58`), so its events
carry the `ort` target. `vaani-inference`'s own events are:

| Level | Event | Emitted at |
|---|---|---|
| `info` | `"ONNX Runtime initialised"` with `ort`, `dirs`, `portable` | `runtime.rs:231` |
| `info` | `"session loaded"` with `model`, `provider`, `threads`, `inputs`, `outputs` | `session.rs:369` |
| `error` | `"ORT could not load {model}: {e}"` | `session.rs:360` |
| `error` | `"inference failed: {e}"` | `session.rs:461` |
| `error` | `"ONNX Runtime error: {e}"` | `session.rs:507` |
| `debug` | `"could not preload runtime DLL"` with `path` | `runtime.rs:285` |

A `debug`-level `could not preload runtime DLL` line naming a cuDNN sibling is the first
thing to read when acceleration misbehaves.

## `vaani --debug` and `RUST_LOG`

**`vaani --debug` does not exist.** No CLI crate has been created. It is planned for
`vaani-cli`, and when it lands it is specified to do two things: install a
`tracing_subscriber` with an `EnvFilter` honouring `RUST_LOG`, and turn on the
`RuntimeReport`-derived diagnostics panel. Neither exists today.

What you can use **right now**, which is equivalent and needs no code change:

```powershell
$env:RUST_LOG = "info"            # or trace, or a target filter as above
$env:Path += ";$env:USERPROFILE\.cargo\bin"

# Hardware + runtime + a real encoder run, all printed:
cargo test -p vaani-inference --test runtime_smoke -- --nocapture

# Provider options, fp16 vs fp32:
cargo run -p vaani-inference --example conv_probe
```

`RuntimeReport::summary()` is the one-line verdict you want in a bug report
(`lib.rs:78-102`). On the reference machine it prints:

```
GPU acceleration available - NVIDIA GeForce RTX 3050 Ti Laptop GPU, 3.3 GB, driver 13.3, compute 8.6
```

Critically, it names the **inference** adapter, not the display one
(`lib.rs:83-85`): using the display adapter here would print the AMD integrated GPU and
tell the user the wrong thing about their own machine.

## `conv_probe` — the provider-option regression probe

```powershell
cargo run -p vaani-inference --example conv_probe
```

It exists because ONNX Runtime's cuDNN path failed on this graph for at least one conv
strategy and the only honest way to find out which is to try them
(`examples/conv_probe.rs:8-10`). It loads `artifacts/sravaani/encoder.fp16.onnx` and
`artifacts/sravaani/encoder.onnx`, runs each with a warm-up pass so cuDNN algorithm
selection is not in the number, checks the output is finite, and prints a table.

Reference output on the dev box:

```
TRIAL                    RESULT   DETAIL
------------------------------------------------------
fp16 encoder, GPU        OK       cuda, 16 s audio in    67 ms  RTF 0.004
fp32 encoder, GPU        OK       cuda, 16 s audio in   110 ms  RTF 0.007
```

Run it after any of these:

- an ONNX Runtime version bump;
- a cuDNN version bump;
- a change to `OnnxSession::cuda_provider_options()`;
- any "it used to be faster" report;
- any unexplained CPU fallback.

It is an example, not a test, because it prints a table of pass/fail-with-reason rather
than asserting one answer, and because on a machine without the runtime it is not
meaningful (`conv_probe.rs:12-14`).

## Provider selection is pure and testable without ORT

`runtime::choose_provider(requested, cuda_loadable)` (`runtime.rs:332`) is deliberately
separated from session construction. Table of actual behaviour:

| `requested` | `cuda_loadable` | `active` | `note` |
|---|---|---|---|
| `Cpu` | `true` | `Cpu` | "Using the CPU, as requested." |
| `Cpu` | `false` | `Cpu` | same |
| `Auto` | `true` | `Cuda` | "Using the NVIDIA GPU." |
| `Auto` | `false` | `Cpu` | "Graphics acceleration is unavailable, so processing runs on the CPU. Transcription will be slower but the result is identical." |
| `RequireCuda` | `true` | `Cuda` | "Using the NVIDIA GPU." |
| `RequireCuda` | `false` | `Cpu` | "Graphics acceleration was requested but is unavailable, so the CPU was used instead." |

Match order is load-bearing: `Provider::Cpu` is matched **before** the CUDA-available arm
(`runtime.rs:337-338`). Otherwise a user who asked for the CPU would silently get the GPU,
which is both surprising and makes the CPU path untestable on a GPU machine.

**`RequireCuda` does not hard-fail.** It returns `active: Cpu` with a note. This is
tested at `runtime.rs:373-380` with the reasoning: *a transcript is worth more than a
benchmark purity guarantee, and the status is recorded so the benchmark can flag the
mismatch.* The doc comments on `Provider::RequireCuda` (`runtime.rs:53-56`) and on
`lib.rs:35-37` still claim it fails the session; those comments are wrong. Trust
`runtime.rs:332-366` and the test.

## Debugging the front end

The Rust log-mel front end does not exist yet. When you write it, the things that will
bite, all verified against `SraVaani-1.0/processing_sravaani.py`:

| Symptom | Cause |
|---|---|
| Transcript is systematically shifted | you forgot that `center=true` yields `1 + wav_len/160` frames, one more than `seq_len` |
| WER barely worse but output is garbage | you used `log(mel + 1e-5)` instead of `log(mel + 2⁻²⁴)` |
| Normalisation is subtly off | you used `Σ(x-mean)²/N` instead of `/(N-1)`; Bessel correction is at `modeling_sravaani.py:68` |
| Everything after a silence is shifted | you normalised using padded frames; the reference masks them out at `:66-68` |
| Last frame always wrong | you did not zero it; `pad_value = 0.0` is applied *after* normalisation at `:87-88` |
| Utterances shorter than 160 ms produce NaN | `denom - 1.0 == 0`; the reference's `masked_fill(isnan → 0)` at `:69` is the guard |
| Words are spaced wrong | the window is **symmetric** Hann (`periodic=False`), 400 samples, zero-padded to 512. A periodic window is wrong by up to 6e-3 |

## What must not change

| Do not | Why |
|---|---|
| Remove `.fail_silently()` from the CUDA EP build | it is the entire CPU-fallback story: the session keeps working when acceleration is unavailable, which is a normal outcome (`session.rs:343-344`) |
| Remove the `SetDllDirectoryW` loop from `ensure_initialised` | cuDNN 9 is a split library; this is the fix for `Could not locate cudnn_engines_runtime_compiled64_9.dll` (`runtime.rs:200-219`) |
| Remove the `preload_dir` loop | it is what turns ORT's error into *our* error naming the missing file |
| Call anything in `ort` before `ensure_initialised()` | the environment is process-global and `ort` requires init first |
| Treat `CudaProbe::error` as diagnostic-only | `is_usable()` requires `error.is_none()`; setting it makes `inference_adapter()` return `None` (`capability.rs:89-91`) |
| Assume `dedicated_vram_bytes` is board size | it is the DXGI *budget*. On this machine 3,531,289,396 bytes for a 4 GB board |
| Assume a `None` from `inference_adapter()` is a bug | it is the honest answer on an AMD-only machine or with a broken NVIDIA driver (`capability.rs:626-635`) |

## How to test it

```powershell
$env:Path += ";$env:USERPROFILE\.cargo\bin"

# Unit level — no runtime, no GPU required, and it says so rather than failing:
cargo test -p vaani-inference --lib
cargo test -p vaani-inference --lib compute_capability_read_from_the_real_device_is_plausible -- --nocapture

# Hardware + a real GPU run:
cargo test -p vaani-inference --test runtime_smoke -- --nocapture

# Provider options:
cargo run -p vaani-inference --example conv_probe

# With ORT's own tracing turned up:
$env:RUST_LOG = "ort=trace,vaani_inference=trace"
cargo test -p vaani-inference --test runtime_smoke -- --nocapture
```

`capability.rs:766-796` is the test that catches the `cudaDeviceProp` layout drift: it
reads the capability from the *real* device and asserts it is not the first two bytes of
the device name. If the offsets ever go wrong, that test fails with a message that says
exactly what happened.

## What it assumes

- **`tracing_subscriber` will be wired in by `vaani-cli`/`vaani-app`.** Until then the
  `ort` EP diagnostics are unreachable and you must reason from `ProviderStatus` and
  timing. This is a known, temporary gap, not a design choice.
- **A `debug`-profile build is what all measurements come from** unless you explicitly
  pass `--release`. The GPU dominates, so the wrapper overhead is small, but the numbers
  are not release-profile numbers and must not be quoted as such.
- **The tests print and skip rather than fail** when the runtime or the model is absent
  (`runtime_smoke.rs:49,65,70`). A green `cargo test` on a bare machine proves nothing
  about acceleration — read the printed `summary()`.
- **`CudaProbe::MIN_DRIVER_FOR_CUDA_13 = 13_000`** (`capability.rs:100`) records the
  requirement but is *not enforced anywhere*. ORT's CUDA-13 build needs a 13.x driver
  (>= 580). Nothing checks it; a too-old driver shows up as an EP registration failure and
  therefore as silent CPU fallback.

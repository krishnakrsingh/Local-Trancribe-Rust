# Debugging

## What exists

Seven diagnostic surfaces: two examples and a test in `vaani-inference`, two tests in
`vaani-asr`, one script, and `tracing` itself.

| Surface | What it tells you | Command |
|---|---|---|
| `scripts/build_doctor.ps1` | whether the tree is healthy, cheapest stage first: toolchain, link, fmt, check, clippy, tests, runtime, model, parity | `powershell -ExecutionPolicy Bypass -File scripts\build_doctor.ps1` |
| `RuntimeReport` | which `onnxruntime.dll` was resolved, which DLLs are missing, every DXGI adapter, the CUDA probe, CPU counts, RAM, the ORT thread count | `cargo test -p vaani-inference --test runtime_smoke -- --nocapture` |
| `conv_probe` example | whether the fp16 and fp32 encoders both run on the CUDA EP, and how fast | `cargo run -p vaani-inference --example conv_probe` |
| `subsample_probe` example | the encoder's **reported** output length for twelve input lengths — how the `ceil(T/8)` law was established rather than assumed | `cargo run -p vaani-inference --example subsample_probe` |
| `frontend_parity` test | the worst log-mel cell against PyTorch, and where it is | `cargo test -p vaani-asr --test frontend_parity -- --nocapture` |
| `sravaani_transcribe` test | the phase breakdown and RTF for a 40 s chunk; with `VAANI_GOLDEN_RAW` set, the strict token-id and word-timing comparison | `cargo test -p vaani-asr --test sravaani_transcribe -- --nocapture` |
| `tracing` events | session load, ORT init, preload failures | `RUST_LOG` — see below |

Plus `dump_logmel` (`crates/vaani-asr/examples/dump_logmel.rs`), which writes the log-mel as
raw little-endian f32 so the front end can be diffed against numpy offline — far faster to
iterate in than a Rust test with no visualisation:

```powershell
cargo run -p vaani-asr --example dump_logmel -- `
  tests\fixtures\sravaani\frontend_golden.json artifacts\sravaani scratch\rust_logmel.bin
```

`RuntimeReport::collect()` **never fails** (`crates/vaani-inference/src/lib.rs:65`). A
missing runtime is a fact to report, not an error to propagate, because the diagnostics
command has to be able to say what is wrong without itself crashing. That is tested at
`lib.rs:126`.

## Why it exists

The dominant failure mode of this product is **silent CPU fallback**: the session loads,
produces correct output, and is dramatically slower. Nothing errors. A capability panel that
guesses is worse than one that says it does not know, so the guiding rule in
`capability.rs:18-20` is *report what we measured, `Unknown` when we did not*.

("20–40×" is the figure older revisions of this document gave, and it is **an unverified rule
of thumb**: the CUDA path is 20–40× faster than the CPU path *for inference workloads
generally*. No CPU-only SraVaani run has ever been timed on this machine, so the real factor
for this model is unknown. See [PERFORMANCE.md](PERFORMANCE.md).)

That rule is why `CudaProbe` carries an `error: Option<String>` and why
`ProviderStatus` carries a `note`. Every silent path in this codebase has a field that
records why it was taken.

## `tracing` and the silent-fallback trap

**`ort` reports execution-provider registration failures only through `tracing`.**

`session.rs:365-370` builds the CUDA EP with `.fail_silently()`:

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
| `info` | `"ONNX Runtime initialised"` with `ort`, `dirs`, `portable` | `runtime.rs:251` |
| `info` | `"session loaded"` with `model`, `provider`, `threads`, `inputs`, `outputs` | `session.rs:398` |
| `error` | `"ORT could not load {model}: {e}"` | `session.rs:383` |
| `error` | `"inference failed: {e}"` | `session.rs:492` |
| `error` | `"ONNX Runtime error: {e}"` | `session.rs:538` |
| `debug` | `"could not preload runtime DLL"` with `path` | `runtime.rs:317` |

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

# The whole gate, with every stage's output on failure:
powershell -ExecutionPolicy Bypass -File scripts\build_doctor.ps1

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

Reference output on the dev box, `--release`, 2026-09-27:

```
TRIAL                    RESULT   DETAIL
------------------------------------------------------
fp16 encoder, GPU        OK       cuda, 16 s audio in    67 ms  RTF 0.004
fp32 encoder, GPU        OK       cuda, 16 s audio in   111 ms  RTF 0.007
```

Note what this probe is **not** for: the 16 s figure it prints is a warm, isolated encoder
run, and it is not the encoder cost inside a real transcription. The encoder phase of a
40 s `transcribe` call measures ~1.5 s, because that call includes per-call dispatch, four
times the work, and no warm-up loop. Both numbers are real; they measure different things.
See [PERFORMANCE.md](PERFORMANCE.md).

Run it after any of these:

- an ONNX Runtime version bump;
- a cuDNN version bump;
- a change to `OnnxSession::cuda_provider_options()`;
- any "it used to be faster" report;
- any unexplained CPU fallback.

It is an example, not a test, because it prints a table of pass/fail-with-reason rather
than asserting one answer, and because on a machine without the runtime it is not
meaningful (`conv_probe.rs:12-14`).

## `subsample_probe` — establishing the subsampling law

```powershell
cargo run -p vaani-inference --example subsample_probe
```

`enc_lengths == ceil(T / 8)` is the kind of constant that is *almost* always derived rather
than measured, and being 8× out drags every transcript out of sync with the media while
still reading as plausible text. So the probe asks the graph instead: twelve input lengths,
including the awkward ones (437, 1601), and it prints the reported `enc_lengths`, the
tensor's own `shape[2]`, and whether the output is finite.

Measured on the dev box, 2026-09-27:

```
     T        T ceil(T/8) enc_lengths  shape/finite
   200      200       25         25  out2=25 finite=true
   437      437       55         55  out2=55 finite=true
  1600     1600      200        200  out2=200 finite=true
  1601     1601      201        201  out2=201 finite=true
  4000     4000      500        500  out2=500 finite=true
  4800     4800      600        600  out2=600 finite=true
 24000    24000     3000       3000  out2=3000 finite=true
```

(the probe runs twelve lengths; seven are shown here.)

Note the two columns agreeing. In the **ASR path** they do not, and that is the point of
[COMMON_FAILURES §11](COMMON_FAILURES.md): the probe passes `feature_lengths == T`, while
`vaani-asr` passes a tensor with `1 + valid_frames` columns and `feature_lengths` one
smaller, which is why the decode must read `enc_lengths` and never the tensor dimension.

## `VAANI_GOLDEN_RAW` — turning on the strict decode comparison

`matches_reference_token_ids_on_real_audio` is the only test that compares the Rust decode
against the reference **token for token**, and it is skipped unless an environment variable
points at the audio the fixture was generated from:

```powershell
$env:VAANI_GOLDEN_RAW = "C:\Users\krish\Downloads\localvoicetool\scratch\golden_clip.f32"
cargo test --release -p vaani-asr --test sravaani_transcribe `
  matches_reference_token_ids_on_real_audio -- --nocapture --test-threads=1
```

| Fact | Value |
|---|---|
| Format | **raw little-endian f32**, mono, 16 kHz. Headerless |
| Produced by | `scripts/make_golden.py`, into `scratch/` (gitignored) |
| Length used | the fixture's `seconds × sample_rate` = 40 × 16000 = 640,000 samples; a longer file is truncated, a shorter one skips |
| Path | **absolute.** Cargo runs integration tests with the *package* directory as cwd, so a relative path resolves to `crates/vaani-asr/...` and fails to open |
| Output | `token id agreement: 202/202`, `word timings: 108 words, worst deviation 0.0 ms`, `EXACT MATCH` |

**It must be a raw f32 file, not a WAV.** This is not a style preference. soundfile's
float WAV header is longer than the canonical 44 bytes (it carries a `fact` chunk and a
format code), so a reader that assumes 44 reads **misaligned samples**: the transcript comes
out as confident nonsense. That presents as a broken *model* — wrong words, plausible
language, no error anywhere — and sends you looking at the decoder, the tokeniser and the
weights instead of at four bytes of offset. The generator writes raw for exactly this
reason (`make_golden.py:315-323`), and the comment in the test says so at
`tests/sravaani_transcribe.rs:289-293`.

If the token ids disagree, print the token stream before anything else: the test already
does, and the first 16 pieces tell you immediately whether the front end is wrong (garbage
from the first token) or the decode loop is wrong (right prefix, diverging later).

## Provider selection is pure and testable without ORT

`runtime::choose_provider(requested, cuda_loadable)` (`runtime.rs:364`) is deliberately
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
(`runtime.rs:369-379`). Otherwise a user who asked for the CPU would silently get the GPU,
which is both surprising and makes the CPU path untestable on a GPU machine.

**`RequireCuda` does not hard-fail.** It returns `active: Cpu` with a note. This is tested
at `runtime.rs:404-412` with the reasoning: *a transcript is worth more than a benchmark
purity guarantee, and the status is recorded so the benchmark can flag the mismatch.* The
doc comments on `Provider::RequireCuda` and on `lib.rs:30-37` still describe a stricter
contract than the code implements. Trust `runtime.rs:364-398` and the test.

## Debugging the front end

The Rust log-mel front end **exists**: `crates/vaani-asr/src/frontend.rs`, verified cell by
cell against PyTorch. What follows is therefore not a list of things to remember while
writing it — every row has already bitten, been fixed, and is regression-tested. They are
listed because they are the ways it will break again the moment someone "simplifies" it.
The long-form version of rows 1, 2 and 3 is
[COMMON_FAILURES §9-11](COMMON_FAILURES.md).

| Symptom | Cause | Test that catches it |
|---|---|---|
| Transcripts are consistently, quietly worse; parity max \|d\| is 3–6 on O(1) values, **flat per band, scattered per frame** | the window is left-aligned in the frame. `torch.stft` centres it: `window_offset = (n_fft − win_length)/2 = 56` | `frontend_parity.rs`, 128 × 101 cells |
| Parity off by ~5.7, per-band mean offset exactly 0, per-frame std ~1.2 | Hermitian symmetry misused: the imaginary part read from bin `n_fft − k`. The symmetry relates the *conjugate*, so that computes `re² + re²` | `frontend_parity.rs` |
| WER barely worse but output is garbage | `log(mel + 1e-5)` instead of `log(mel + 2⁻²⁴)` | `frontend_parity.rs` |
| Normalisation is subtly off | `Σ(x-mean)²/N` instead of `/(N-1)`; Bessel correction is at `modeling_sravaani.py:68` | `frontend_parity.rs` |
| Everything after a silence is shifted | normalisation used the padded frames; the reference masks them out at `:66-68` | `frontend_parity.rs` |
| Last frame always wrong | the padding frame was not zeroed; `pad_value = 0.0` is applied *after* normalisation at `:87-88` | `padding_frames_match_the_reference` |
| Utterances shorter than 160 ms produce NaN | `denom - 1.0 == 0`; the reference's `masked_fill(isnan → 0)` at `:69` is the guard | `single_frame_clip_does_not_produce_nan` |
| Words are spaced wrong | one `Word` per SentencePiece piece. "ऑटोमैटिकली" is seven pieces and `▁` is what says a word begins | decode fixture, 108/108 words |
| A trailing empty or hallucinated word | the decode ran past `enc_lengths` and consumed the padding frame | decode fixture, token ids |
| Everything is 8× out of sync with the video | `frame_seconds` derived from mel frames instead of encoder frames (8× subsampling) | `frame_seconds_is_eighty_milliseconds` |

Two habits make this cheap to debug. First, `dump_logmel` writes the features as raw f32
so the diff happens in numpy, not in a Rust test. Second, the parity fixture checks **every**
cell, because a sampled check passes a whole-wrong-mel-band implementation.

## What must not change

| Do not | Why |
|---|---|
| Remove `.fail_silently()` from the CUDA EP build | it is the entire CPU-fallback story: the session keeps working when acceleration is unavailable, which is a normal outcome (`session.rs:365-369`) |
| Remove the `SetDllDirectoryW` loop from `ensure_initialised` | cuDNN 9 is a split library whose siblings resolve **lazily, during the first inference run**; the preloads alone do not cover that (`runtime.rs:231-239`) |
| Remove the `preload_dir` loop | it is what turns ORT's error into *our* error naming the missing file (`runtime.rs:243-245`) |
| Call anything in `ort` before `ensure_initialised()` | the environment is process-global and `ort` requires init first |
| Treat `CudaProbe::error` as diagnostic-only | `is_usable()` requires `error.is_none()`; setting it makes `inference_adapter()` return `None` (`capability.rs:91-93`) |
| Assume `dedicated_vram_bytes` is board size | it is the DXGI *budget*. On this machine 3,531,289,396 bytes for a 4 GB board |
| Assume a `None` from `inference_adapter()` is a bug | it is the honest answer on an AMD-only machine or with a broken NVIDIA driver (`capability.rs:170-176`) |
| Feed a WAV to `VAANI_GOLDEN_RAW` | soundfile's float header is longer than 44 bytes; assuming 44 reads misaligned samples and looks like a broken model |
| Skip the parity tests because the change was "only" in the decode loop | the front end and the decode loop are separately gated, and both fail silently in the same way: plausible text |

## How to test it

```powershell
$env:Path += ";$env:USERPROFILE\.cargo\bin"

# The whole gate, cheapest first:
powershell -ExecutionPolicy Bypass -File scripts\build_doctor.ps1

# Unit level — no runtime, no GPU required, and it says so rather than failing:
cargo test -p vaani-inference --lib
cargo test -p vaani-inference --lib compute_capability_read_from_the_real_device_is_plausible -- --nocapture
cargo test -p vaani-asr --lib

# Hardware + a real GPU run:
cargo test -p vaani-inference --test runtime_smoke -- --nocapture

# Provider options, and the subsampling law:
cargo run -p vaani-inference --example conv_probe
cargo run -p vaani-inference --example subsample_probe

# Parity, front end and decode:
cargo test -p vaani-asr --test frontend_parity -- --nocapture
$env:VAANI_GOLDEN_RAW = "C:\...\scratch\golden_clip.f32"   # absolute path
cargo test -p vaani-asr --test sravaani_transcribe -- --nocapture --test-threads=1

# With ORT's own tracing turned up:
$env:RUST_LOG = "ort=trace,vaani_inference=trace"
cargo test -p vaani-inference --test runtime_smoke -- --nocapture
```

`capability.rs:815` is the test that catches the `cudaDeviceProp` layout drift: it reads the
capability from the *real* device and asserts it is not the first two bytes of the device
name. If the offsets ever go wrong, that test fails with a message that says exactly what
happened.

## What it assumes

- **`tracing_subscriber` will be wired in by `vaani-cli`/`vaani-app`.** Until then the
  `ort` EP diagnostics are unreachable and you must reason from `ProviderStatus` and
  timing. This is a known, temporary gap, not a design choice. `RUST_LOG` alone does
  nothing until a subscriber exists; `build_doctor.ps1` does not install one either.
- **Published measurements come from `--release`.** Every number in
  [PERFORMANCE.md](PERFORMANCE.md) was taken with `--release`; a `dev` build is fine for
  diagnosis and must never be quoted.
- **The tests print and skip rather than fail** when the runtime, the artefacts or the
  fixture is absent. A green `cargo test` on a bare machine proves nothing about
  acceleration and nothing about accuracy — read the printed reason, or run
  `build_doctor.ps1`, which reports a missing prerequisite as `SKIP`.
- **The strict decode comparison needs the original audio.** It cannot run on a fresh
  clone, because the audio is deliberately not in the repository. `build_doctor.ps1` looks
  for `scratch\golden_clip.f32` and reports `SKIP` if it is absent; it never fabricates a
  pass.
- **`CudaProbe::MIN_DRIVER_FOR_CUDA_13 = 13_000`** (`capability.rs:102`) records the
  requirement but is *not enforced anywhere*. ORT's CUDA-13 build needs a 13.x driver
  (>= 580). Nothing checks it; a too-old driver shows up as an EP registration failure and
  therefore as silent CPU fallback.

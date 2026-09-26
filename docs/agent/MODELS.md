# Models

## What exists

### On disk right now

`artifacts/sravaani/` — 2.66 GB, produced by `scripts/export_sravaani_onnx.py`. Verified
2026-09-27 against the actual files:

| File | Bytes | Graph I/O | Role |
|---|---:|---|---|
| `encoder.fp16.onnx` | 886,484,931 | `feats` **FLOAT16**, `enc_out` **FLOAT16** | **production encoder** |
| `encoder.onnx` | 1,772,812,911 | `feats` FLOAT, `enc_out` FLOAT | reference / CPU fallback |
| `decoder_joint.onnx` | 43,033,439 | all FLOAT except `tgt`/`tlen` INT32 | TDT emission step |
| `mel_window.f32` | 1,600 | 400 × f32 | `torch.hann_window(400, periodic=False)`, max 0.99998 |
| `mel_fb.f32` | 131,584 | 32,896 × f32 = 128 × 257 | mel filterbank, row-major |
| `preproc.json` | 426 | — | front-end parameters |
| `config.json` | 590 | — | model hyper-parameters, mirrored from the HF config |
| `artefacts.json` | 1,363 | — | sha256 + byte count of every output — **currently stale, see below** |

Graph signatures confirmed by loading the protos with `onnx` 1.23.0:

| Graph | opset | producer | inputs | outputs |
|---|---|---|---|---|
| `encoder.fp16.onnx` | 17 | pytorch 2.13.0 | `feats` FLOAT16 [batch,128,time], `feature_lengths` INT64 [batch] | `enc_out` FLOAT16 [batch,1024,enc_time], `enc_lengths` INT64 [batch] |
| `encoder.onnx` | 17 | pytorch 2.13.0 | same, FLOAT | same, FLOAT |
| `decoder_joint.onnx` | 17 | pytorch 2.13.0 | `f` [1,1024,1], `tgt` INT32 [1,1], `tlen` INT32 [1], `h_in` [1,1,640], `c_in` [1,1,640] | `logits` [1,1,1,5006], `target_length` INT32 [1] *(dead)*, `h_out` [1,1,640], `c_out` [1,1,640] |

`SraVaani-1.0/` — 908 MB TorchScript, gitignored. Contains `model-asr.fp16.ts`,
`tokenizer.model`, `preproc.pt`, `config.json`, `model_meta.json`, and the NeMo-derived
Python that produced them.

### Model facts, verified

| Fact | Value | Source |
|---|---|---|
| Parameters | ~430M FastConformer + RNNT prednet | `SraVaani-1.0/README.md:26` |
| Encoder layer count | **not verifiable in this repo.** `configuration_sravaani.py` declares no layer count; "17" comes from the model card's description and NeMo's default. Do not hard-code it | — |
| TorchScript archive | 908,884,628 B (908.9 MB) | measured |
| Upstream repo | `ARTPARK-IISc/SraVaani-1.0` | `SraVaani-1.0/README.md:44` |
| Licence | MIT (model card) | `SraVaani-1.0/README.md:2` |
| Languages | 65 Indian languages + dialects; 105 in pretraining | `README.md:32,35` |
| Datasets | `ARTPARK-IISc/Vaani`, `ARTPARK-IISc/Vaani-transcription-part` | `README.md:20-21` |
| Access | **gated — requires a Hugging Face token** | `README.md:44,155` |
| Hindi WER | 12.4 avg | `README.md:78` |
| Tokeniser | SentencePiece **unigram**, 5,000 pieces | measured |
| `unk_id` | 0 | measured |
| `bos_id` / `eos_id` / `pad_id` | −1 / −1 / −1 (none) | measured |
| Metaspace-prefixed pieces | 2,190 have U+2581 | measured |
| `blank_id` | 5,000 — **outside** the SP vocab | `preproc.json:16`, `config.json:12` |
| Sentencepiece Python ref | 0.2.2 | measured |

### Front-end parameters (`artifacts/sravaani/preproc.json`)

| Key | Value |
|---|---|
| `sample_rate` | 16000 |
| `n_fft` | 512 |
| `hop_length` | 160 |
| `win_length` | 400 |
| `n_mels` | 128 |
| `n_freq` | 257 |
| `preemph` | 0.97 |
| `mag_power` | 2.0 |
| `log` | true |
| `log_zero_guard_type` | `add` |
| `log_zero_guard_value` | 5.960464477539063e-08 (**exactly 2⁻²⁴**) |
| `normalize` | `per_feature` |
| `constant` | 1e-05 |
| `pad_value` | 0.0 |

`config.json`: `vocab_size` 5000, `blank_id` 5000, `durations` [0,1,2,3,4],
`num_durations` 5, `pred_hidden` 640, `pred_rnn_layers` 1, `max_symbols` 10,
`enc_d_model` 1024, `feat_in` 128, `sample_rate` 16000, `subsampling_factor` 8.

## Why it exists

Weights are never committed. The repository ships source and lightweight metadata only;
`artifacts/`, `runtime/`, `SraVaani-1.0/`, `models/` and every media extension are in
`.gitignore:1-50`. Two reasons, both load-bearing:

1. **Licence and access.** SraVaani-1.0 is MIT *but gated* on Hugging Face. Committing the
   weights would redistribute them outside the gate and outside the licence's terms.
2. **Size.** 908 MB of TorchScript plus 2.66 GB of artefacts is not a git repository.

The exported artefacts are checksummed by `artefacts.json` and, at release time, by the
model manifest described below, so a downloaded copy can be verified without re-running
the exporter.

The native-fp16-encoder / fp32-decoder split is the central technical decision here; see
[ADR-004](../decisions/ADR-004-sravaani-integration.md).

## What must not change

| Do not | Why |
|---|---|
| Commit any file under `artifacts/`, `runtime/`, `SraVaani-1.0/`, or `models/` | `.gitignore:10-19`; licence gate and size |
| "Fix" `log_zero_guard_value` to NeMo's usual `1e-5` | it is `2⁻²⁴` in `preproc.pt` (`preproc.json:12`). Changing it changes the normalisation. Verified against `SraVaani-1.0/processing_sravaani.py:84` |
| Use `onnxconverter_common.float16` on this graph | it produces a graph ONNX Runtime rejects. See [COMMON_FAILURES §2](COMMON_FAILURES.md) |
| Make `decoder_joint.onnx` fp16 | its LSTM state is threaded through every emission step; error compounds over minutes. `export_sravaani_onnx.py:328-333` |
| Feed f32 to `encoder.fp16.onnx` | native fp16 I/O. `F16Tensor::from_f32` is the only sanctioned conversion, `session.rs:83` |
| Pad or shorten `mel_window.f32` | it is exactly 400 samples and is symmetric Hann, not periodic. Zero-padding to 512 happens in the front end at load time, not in the artefact |
| Reorder the TDT decode loop's request for `decoder_joint` outputs | request `["logits", "h_out", "c_out"]` **by name**; `target_length` is dead but cannot be removed from the graph |
| Treat `preproc.pt` as readable from Rust | it is a zip+pickle. It is unpacked once by the exporter into `mel_window.f32` + `mel_fb.f32` + `preproc.json`. `export_sravaani_onnx.py:54-59` |

### The log-mel front end, to be reimplemented in `vaani-asr`

Not yet written. This is the exact order, read out of
`SraVaani-1.0/processing_sravaani.py:68-89` and `_normalize_per_feature`
(`SraVaani-1.0/modeling_sravaani.py:61-70`):

| # | Step | Exact definition |
|---|---|---|
| 1 | `seq_len` | `floor(wav_len / hop_length)` — `processing_sravaani.py:76` |
| 2 | pre-emphasis | `x[0] = w[0]`; `x[n] = w[n] - 0.97 * w[n-1]` — `:78` |
| 3 | STFT | `n_fft=512`, `hop=160`, `win_length=400`, `center=true`, `pad_mode="constant"` — `:79-80` |
| 4 | window | the 400-sample window is zero-padded by `torch.stft` to `n_fft=512` |
| 5 | power spectrogram | `sqrt(re² + im²).pow(2.0)`; because `mag_power == 2.0` the `sqrt` is a no-op and the result is `|X|²` — `:82` |
| 6 | mel | `mel = fb @ |X|²`, `fb` is `[128, 257]` — `:83` |
| 7 | log | `log(mel + 2⁻²⁴)` — `:84` |
| 8 | normalise | per feature: `mean` over valid frames, `var = Σ(x-mean)²/(N-1)` (**Bessel, N−1**), `std = sqrt(var)` with `NaN → 0`, then `std = std + 1e-5` (**added to the std**, not a floor), output `(x-mean)/std` — `modeling_sravaani.py:61-70` |
| 9 | pad | frames with index `>= seq_len` set to `pad_value = 0.0` — `:87-88` |

Consequences to carry into the Rust implementation:

- `center=true` means `torch.stft` zero-pads `n_fft/2` at each end, producing
  `1 + wav_len/hop` frames — **exactly one more than `seq_len`**. The last frame is
  always padding.
- `denom - 1.0` at `modeling_sravaani.py:68` divides by zero when `seq_len == 1`; the
  `masked_fill(isnan → 0)` at `:69` is what makes that survivable. Preserve it.
- The window is applied centred within the 512-point FFT buffer, so indices 200..311 of
  the buffer are the second half of the window followed by 112 zeros.
- All numerics stay `f32`. The single `f32 → f16` conversion happens at the graph edge in
  `F16Tensor::from_f32` (`session.rs:83`).

### Output shape invariant

`enc_lengths == ceil(T_mel / 8)` exactly. Asserted by the exporter at
`export_sravaani_onnx.py:437-444` for T ∈ {200, 1600, 4800}, and by
`tests/runtime_smoke.rs:155` for T=1600 → 200. Verified for T = 200, 437, 1600, 4800,
24000.

## How to test it

```powershell
# Model presence, hardware, and a real GPU run of the fp16 encoder:
$env:Path += ";$env:USERPROFILE\.cargo\bin"
cargo test -p vaani-inference --test runtime_smoke -- --nocapture

# Both encoders, fp16 vs fp32, on this machine:
cargo run -p vaani-inference --example conv_probe

# Re-derive the artefacts from the TorchScript (needs Python, ~10 min, 3 GB RAM):
python scripts/export_sravaani_onnx.py --src SraVaani-1.0 --out artifacts/sravaani
```

The exporter is its own test suite. It aborts on any of:

| Check | Line | Threshold |
|---|---|---|
| `window` shape == `(win_length,)` | `:73` | exact |
| `fb` is 2-D with `n_freq` columns | `:77` | exact |
| `cfg["feat_in"] == preproc["n_mels"]` | `:507` | exact |
| `cfg["blank_id"] == preproc["blank_id"]` | `:509` | exact |
| `logits` shape == `[1,1,1,5006]` | `:273` | exact |
| `h_out`/`c_out` shape == `[1,1,640]` | `:275` | exact |
| `decoder_joint` vs PyTorch parity | `:289` | `max|d| <= 1e-4` |
| `enc_lengths == ceil(T/8)` | `:440` | exact |
| `enc.shape[1] == 1024` | `:445` | exact |
| fp16 output finite | `:438` | exact |
| fp16 vs fp32 argmax agreement | `:464` | **>= 99.0%** |

Measured parity (last recorded run): `decoder_joint` logits `max|d| = 2.29e-05`, `h`
`4.76e-07`. `encoder.fp16` vs fp32: `max|d| = 0.046`, `mean|d| = 0.00082`, argmax
agreement **99.609%**.

### The model manifest

**Planned, not built** — `vaani-models` does not exist. When it does, every entry must
record all of these. Anything omitted is a field the app will have to guess at later.

| Field | Type | Example / note |
|---|---|---|
| `id` | string | `sravaani-1.0-onnx` |
| `version` | string | `1.0` — independent of `APP_VERSION` (`crates/vaani-core/src/lib.rs:47-52`) |
| `provider` | enum | `huggingface` \| `github-releases` \| `bundled` |
| `architecture` | string | `fastconformer-tdt-rnnt`, enc_d_model 1024, feat_in 128, subsampling_factor 8, pred_hidden 640, pred_rnn_layers 1 |
| `size` | bytes | 886,484,931 for the fp16 encoder |
| `sha256` | string | per file, from `artefacts.json` |
| `license` | string | `mit` |
| `access` | enum | `open` \| `gated-token` \| `request-access` |
| `local_path` | path | resolved after download; must be under the app data dir, never the repo |
| `download_url` | string | Hugging Face resolve URL for gated repos |
| `runtime_requirements` | table | `ort >= 1.30.0`, `cudnn 9.14.0.64`, `cuda redist 13.0.2`, `opset 17` |
| `gpu_requirements` | table | `compute_capability >= 8.6` for the fp16 path; `vram_mb` for the fit check |

`MachineCapabilities::vram_fits(weights_bytes, headroom)` (`capability.rs:196`) is the
pre-flight check that consumes `size` and `gpu_requirements`. It reasons about
`inference_adapter()`, so on this machine it sizes against the 3.3 GB NVIDIA budget and
not the AMD iGPU's 8 GB of shared memory.

## Token handling

The end user's Hugging Face token:

- lives in **Windows Credential Manager** via `keyring` 4 (`features = ["v1"]`,
  `Cargo.toml:72`) with a **DPAPI-sealed file** fallback;
- is **never** written to a log, a `tracing` field, a command-line argument, an
  environment variable the app re-exports, or a crash dump;
- is never in the repository. `.gitignore:77-80` ignores `*.token`, `*token*.json`, `.env`,
  `.env.*` as a backstop, not as the mechanism.

The end-user-facing error for a missing token is fixed and reviewed:
`err::auth_required()` — *"Hugging Face authentication required. The speech recognition
model has not been downloaded on this computer yet."* is `model_unavailable`;
`auth_required` says *"This model is hosted on Hugging Face behind a login. Add your
Hugging Face token in Settings to download it."* Both take **zero arguments**
(`error.rs:327-337`). A 401-with-a-real-body from `hf-hub` goes to `with_cause`, never
into `detail`. The distinction between a valid token and an ungranted one is
`ErrorCode::AuthRequired` vs `ErrorCode::AccessNotGranted`, asserted in tests, not
inferred from prose.

## What it assumes

- **The gate stays where it is.** If SraVaani-1.0 becomes ungated, `access` becomes
  `open` and the token path becomes optional — but the manifest field stays, so old
  manifests remain readable.
- **`artefacts.json` is regenerated by the same exporter run that produced the files.**
  It is currently stale — see the report; do not treat it as authoritative until
  re-run.
- **The 5,000-piece SentencePiece unigram model is the only tokeniser.** 2,190 pieces
  carry the U+2581 metaspace prefix; any Rust implementation must reproduce byte-for-byte
  including that prefix.
- **`sentencepiece-rs` 0.2 is young** (0.2.x, ~5.8k downloads at time of writing). It has
  zero dependencies, no C++ toolchain, and is Windows-safe, but it is unproven here. It
  needs a byte-for-byte decode parity test against Python `sentencepiece` 0.2.2 — the
  reference implementation, measured on this machine — **before** any transcript is
  considered trustworthy. There is currently no such test, because no Rust tokeniser
  exists.
- **The licence is MIT and the weights are gated.** Any redistribution of the artefacts,
  as opposed to the source, needs both facts re-checked first.

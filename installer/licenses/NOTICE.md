# Third-party notices — VaaniDesk

This file accompanies the VaaniDesk distribution. It names every third-party
component the application ships, what it is used for, and where its licence
text is in this distribution.

The licences themselves are in this directory. They are copied there by
`scripts/release/make-release.ps1`, which fails the build if a component is
listed here and its licence file is missing — an attribution that is written but
not shipped is the kind of mistake that is only discovered in a takedown.

---

## 1. SraVaani 1.0 — speech recognition weights

| | |
|---|---|
| **Component** | `models/sravaani/` — 887 MiB: the fp16 Conformer encoder, the fp32 TDT decoder, the SentencePiece tokenizer, the mel filterbank and analysis window, and the model config |
| **Licence** | MIT |
| **Attribution file** | `SraVaani-MIT-LICENSE.txt` (beside this file) |
| **Author** | SPIRE Lab, ARTPARK, and the Indian Institute of Science, Bangalore |
| **Source** | <https://huggingface.co/ARTPARK-IISc/SraVaani-1.0> |
| **What was redistributed** | ONNX graphs exported by VaaniDesk's own `scripts/export_sravaani_onnx.py` from the upstream TorchScript checkpoint `model-asr.fp16.ts` at revision `f5dd5358`. No upstream file is redistributed byte-for-byte. |
| **Why it is bundled** | The product decision is zero-token, 100 %-offline out of the box: a first-run user must be able to transcribe without an account, a network connection or a Hugging Face login. |

**Please cite the model if you publish results produced with VaaniDesk:**

> Sujith Pulikodan, Agneedh Basu, Pavan Kumar J, Pranav D Bhat, Suryansh
> Shukla, Nihar Desai, Prasanta Kumar Ghosh. *SraVaani 1.0: Scaling Inclusive
> Speech Recognition for Indic Languages.* arXiv:2608.08235, 2026.

The weights are licensed to you under MIT. MIT grants no warranty of any kind.
The publisher reports word error rates per language in the model card; the
figures most likely to matter to a user are Hindi 12.4 and English 14.8 average
WER, with low-resource tribal languages far higher.

**The upstream repository is access-gated.** Downloading from Hugging Face
requires accepting a click-through that shares your contact information with
the publisher. That is an *access* condition, not a redistribution restriction,
and the licence itself imposes no field-of-use, non-commercial or
no-redistribution clause — which is why these weights are redistributed here at
all. The publisher has not published a `LICENSE` file; see the provenance note
in `SraVaani-MIT-LICENSE.txt`.

---

## 2. ONNX Runtime — the inference engine

| | |
|---|---|
| **Component** | `runtime/onnxruntime.dll` and `runtime/onnxruntime_providers_shared.dll` |
| **Version** | 1.30.0, official Microsoft Windows x64 release build |
| **Licence** | MIT (Microsoft Corporation) |
| **Attribution file** | `ONNX-Runtime-LICENSE.txt` (beside this file) |
| **Source** | <https://github.com/microsoft/onnxruntime/releases/tag/v1.30.0> |
| **What was redistributed** | The unmodified `lib/` DLLs from the published zip, verified against the SHA-256 the GitHub Releases API publishes for that asset. |
| **Why it is bundled** | ONNX Runtime is the process that actually executes the model. A distribution that ships weights but not the runtime transcribes nothing. |

Microsoft's build is used rather than the `ort` crate's CDN build because
Microsoft's CUDA release compiles for `86-real` — native SASS for compute
capability 8.6 — while the CDN build ships `75;80;90;120` and would leave the
product dependent on PTX JIT at first run.

---

## 3. NVIDIA CUDA and cuDNN — GPU acceleration, **optional**

| | |
|---|---|
| **Component** | `runtime/cublas64_13.dll`, `runtime/cublasLt64_13.dll`, `runtime/cudart64_13.dll`, `runtime/cudnn*.dll`, `runtime/nvrtc*.dll`, `runtime/onnxruntime_providers_cuda.dll` |
| **Licence** | NVIDIA CUDA Toolkit EULA and the NVIDIA cuDNN terms — **not** MIT, and **not** open source |
| **Attribution file** | `NVIDIA-CUDA-LICENSE.txt` (beside this file) |
| **Source** | NVIDIA redistributable archives, fetched and hash-verified by the frozen `scripts/fetch_runtime.ps1` |
| **What was redistributed** | Unmodified runtime redistributables. No CUDA Toolkit, compiler or development tooling is included. |

**These are not in the default distribution.** VaaniDesk's default package ships
the CPU execution provider only, for two reasons that are both about the user's
machine rather than about elegance:

1. **It is 1.16 GiB of extra download for everyone** to accelerate the
   transcriptions of the users who own the hardware it accelerates. See the
   project's BENCHMARKS for the measured split.
2. **The licence is not an open-source licence.** NVIDIA's SDK agreement
   §2.5 states that the SDK may not be used "in any manner that would cause it
   to become subject to an open source software license", giving as examples
   licences that require, among other things, that software be "redistributable
   at no charge". VaaniDesk is MIT/Apache-licensed and distributed at no charge.
   Bundling 1.16 GiB of NVIDIA redistributables inside it is a question for
   counsel, not for a build script.

Adding GPU support is documented in `OFFLINE-INSTALL.md` and is a copy of files
into `runtime/`. The application detects them at startup and uses them
automatically; nothing else changes.

---

## 4. VaaniDesk itself

VaaniDesk is distributed under `MIT OR Apache-2.0`. The repository does not yet
carry a top-level `LICENSE` file, which is a known gap and is on the release
checklist. It does not affect the notices above, which are about third-party
components.

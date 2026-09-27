# VaaniDesk — offline installation

**You do not need an account, a network connection, a Hugging Face token, or
Python.** The speech recognition model is inside this installation, under
`models\`. If VaaniDesk is running, it can already transcribe.

If you are reading this because something is not working, skip to
[If something is wrong](#if-something-is-wrong) rather than reinstalling.

---

## What is in the box

```
vaani.exe              the application
vaani-cli.exe          the command line, and part of the self-test
models\sravaani\       the speech recognition model, 887 MiB
runtime\               ONNX Runtime, the process that runs the model
licenses\              who wrote what, and under what terms
OFFLINE-INSTALL.md     this file
verify-install.ps1     check that the above is all present and intact
vaani-dist.sha256      a checksum of every file in this installation
vaani-dist.json        what this distribution contains, in machine-readable form
vaani-model-registry.json  the expected size and checksum of every model file
```

## Check that it works

Open PowerShell in this folder and run:

```powershell
.\verify-install.ps1
```

It answers three questions and prints a verdict:

| check | what it means |
|---|---|
| **present** | every file this distribution should contain is here, at the right size |
| **intact** | every file hashes to what `vaani-model-registry.json` recorded — nothing has been corrupted or half-copied |
| **loadable** | ONNX Runtime starts and the model's encoder graph opens |

The third one is the point. A file can be the right size and the wrong bytes,
and the symptom a user sees is transcription that is mysteriously slow. This
check opens the model rather than trusting the file size, so it catches that.

It hashes 887 MiB, so it takes about **five seconds on an SSD** and a few
minutes on a slow or network drive. For a script:

```powershell
.\verify-install.ps1 -Json
```

exits `0` when everything passed and `1` when something did not, so it can gate
a deployment.

You do not need to run this to use VaaniDesk. It exists so that "it is slow" can
be answered with "here is which file" instead of a guess.

## Transcribe something

Double-click `vaani.exe`, drop in an audio or video file, and press start. Or
from PowerShell:

```powershell
.\vaani-cli.exe transcribe "C:\path\to\meeting.m4a" --out transcript
```

## Recording a screen

VaaniDesk is a native Windows desktop application: no browser, no localhost
server, no web view, and no screen recording of the room. Screen capture is not
something the application does.

## The model

`models\sravaani\` holds a multilingual speech recognition model covering 65
Indian languages and dialects, developed by SPIRE Lab at ARTPARK and the Indian
Institute of Science, Bangalore, and released under the MIT licence. It is
roughly 430 million parameters, stored as fp16 ONNX graphs.

`licenses\SraVaani-MIT-LICENSE.txt` has the full notice and the provenance.
`licenses\NOTICE.md` lists everything else in the box and who wrote it. If you
publish results produced with VaaniDesk, please cite the model; the citation is
in the notice.

## Optional: GPU acceleration

The default package uses the CPU, and works on any machine with no extra setup.
If you have an NVIDIA graphics card and want faster transcription, copy these
files into `runtime\` and restart — nothing else changes, and VaaniDesk will
find and use them automatically:

```
onnxruntime_providers_cuda.dll    the CUDA execution provider
onnxruntime_providers_tensorrt.dll
cublas64_13.dll                   CUDA maths library
cublasLt64_13.dll
cudart64_13.dll                   the CUDA runtime
nvrtc64_130_0.dll                 runtime compilation
nvrtc-builtins64_130.dll
cudnn64_9.dll                     cuDNN, required by the CUDA provider
cudnn_adv64_9.dll
cudnn_cnn64_9.dll
cudnn_ops64_9.dll
cudnn_graph64_9.dll
cudnn_heuristic64_9.dll
cudnn_engines_precompiled64_9.dll
cudnn_engines_runtime_compiled64_9.dll
```

**All of these must sit directly in `runtime\`, not in a subfolder.** cuDNN 9's
`cudnn64_9.dll` is a dispatcher that loads its siblings by bare name, and
Windows only looks beside the executable. If you nest them, VaaniDesk falls back
to the CPU with no error message at all — it looks like "GPU acceleration does
not work" rather than "you put the files in the wrong folder".

Where to get them: `scripts\fetch_runtime.ps1` in the source repository, which
downloads NVIDIA's own redistributable archives and verifies each against the
SHA-256 in NVIDIA's published manifest. **They are not in the default download
because they are 1.16 GiB, only NVIDIA users can use them, and they are
distributed by NVIDIA under its own SDK licence rather than an open-source
one.** `licenses\NOTICE.md` explains that in full; read it before redistributing
a build that includes them.

You also need a recent NVIDIA driver. VaaniDesk works without the driver — it
just uses the CPU.

## Portable use on a USB stick

Unpack the ZIP anywhere and run it. To keep the 887 MiB of weights off the
stick — or on a separate volume — create a file called `portable.txt` beside
`vaani.exe` containing one line: the full path to the directory that holds
`models\`.

```
D:\VaaniDesk-models
```

VaaniDesk will look there. It searches that location last, after the installed
location, so it never overrides an install.

---

## If something is wrong

Run the self-test first, and read what it says:

```powershell
.\verify-install.ps1
```

It names the specific file. Then:

| what it says | what to do |
|---|---|
| `models\sravaani` **MISSING** | Reinstall VaaniDesk. Nothing here needs a download or an account. |
| a model file is the **wrong size** | The copy was interrupted. Delete `models\sravaani` and reinstall. |
| a model file is a **checksum mismatch** | The file is the right size and the wrong bytes — usually a partial copy or a disk problem. Reinstall, and check your drive. |
| `onnxruntime.dll` **MISSING** | The distribution is incomplete. Reinstall. |
| the CUDA provider is **MISSING** | Expected. The default package is CPU-only. See *Optional: GPU acceleration*. |

For a fuller report:

```powershell
.\vaani-cli.exe doctor --json
```

For a report with the model's own bytes checked:

```powershell
.\vaani-cli.exe models verify sravaani-1.0-onnx
```

**Reinstalling is always the fix.** The model ships with the application, so
there is nothing to download and no credential to supply, whatever any message
suggests.

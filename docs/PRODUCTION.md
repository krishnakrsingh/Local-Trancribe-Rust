# Production operations

What this document states is what the code does. Where a number appears it was
measured on the machine named in [Verified behaviour](#verified-behaviour), and
where a limit appears it is enforced rather than intended.

- [What the system is](#what-the-system-is)
- [Installation](#installation)
- [Hardware requirements](#hardware-requirements)
- [Configuration](#configuration)
- [Resource limits](#resource-limits)
- [Security model](#security-model)
- [Performance characteristics](#performance-characteristics)
- [Monitoring and logs](#monitoring-and-logs)
- [Failure recovery](#failure-recovery)
- [Known limitations](#known-limitations)
- [Rollback](#rollback)
- [Verifying a build](#verifying-a-build)

---

## What the system is

A native Windows desktop application that transcribes audio and video files
locally. No audio leaves the machine, and there is no server component: the only
network traffic the product ever initiates is a model download from the
distribution it was configured with.

The pipeline for one file is six stages, each writing into a per-job work
directory and each able to fail independently:

```
media → vad → asr → diarization → grouping → store
```

A failure in diarization is never fatal: a transcript without speaker labels is
still a transcript, and the job records why the labels are missing. The same is
true of grouping.

## Installation

The product is a **portable directory**, not an installer. `ISCC.exe` is not
present on the build machine, so no `.mss` is produced; the portable ZIP is the
artefact to test on a clean machine.

Unpack anywhere the user has write access. Nothing is installed into the system,
no registry keys are written, and no service is registered. To remove the
product, delete the directory.

There are two flavours, and **they are not interchangeable**:

| | CPU | GPU |
|---|---|---|
| `runtime/` | 2 DLLs | 19 DLLs, 1.25 GiB |
| ZIP size | 0.81 GiB | 1.70 GiB |
| Unpacked | ~1.0 GiB | 2.16 GiB |
| Requires | nothing | an NVIDIA GPU with the CUDA provider |

Running the CPU build on a machine with a working NVIDIA GPU is a supported
configuration, not a broken one: it transcribes correctly and roughly six times
more slowly. The status bar says `CPU` when that is what happened, and the
settings panel says why, rather than the device drop-down showing "GPU (CUDA)"
as though it were live.

## Hardware requirements

**Minimum.** Any x86-64 CPU, 4 GB free RAM, ~1.1 GB disk for the CPU build.
Transcription is single-threaded per job with a shared engine, so extra cores
help throughput rather than latency.

**GPU path.** An NVIDIA GPU whose driver is visible to Windows, and whose free
VRAM exceeds the weights plus activation headroom. Measured on an RTX 3050 Ti
laptop with 3.3 GB usable:

| | peak VRAM |
|---|---|
| model weights resident | ~887 MB |
| serial transcription | 2408 MiB total |
| batch 2 (**shipped**) | 2817 MiB total, 31% free |
| batch 4 | 3530 MiB total, 14% free |
| batch 6 | allocation failure |

The model needs **1.5 GB of VRAM** to be offered at all. A smaller card is not
offered the GPU device rather than failing at load.

**Disk.** 887 MB of model weights, plus room for the audio you point it at. A job's
work directory holds decoded audio and partial exports and is removed when the
job finishes; see [Failure recovery](#failure-recovery) for the crash case.

## Configuration

There is no configuration file. Settings are chosen in the app's Settings panel
and persisted by the store. Every environment variable below is read by the
development or measurement tooling, and none of them can remove a safety limit.

| Variable | Read by | Effect |
|---|---|---|
| `VAANI_LOG` | desktop | Log filter. Defaults to `info`. |
| `VAANI_ASR_ENCODER_BATCH` | vaani-asr | Chunks per encoder call. **Clamped to 4**; 0, negative, empty and unparseable values fall back to 2. |
| `VAANI_ENCODER_FRAME_BUCKET` | vaani-asr | Mel-frame granularity of the encoder's time axis. `0` disables bucketing. Measured not to change the transcript. |
| `VAANI_ARTEFACTS` | vaani-asr, vaani-inference | Model directory override, for running against a local export. |
| `VAANI_RUNTIME_DIR` | vaani-inference, vaani-bench | ONNX Runtime directory override. |
| `VAANI_GOLDEN_RAW` | vaani-asr tests | Enables the strict token-id equality test. |

`VAANI_ASR_ENCODER_BATCH` is the one that matters operationally: setting it to
`1` restores the pre-optimisation serial encoder path, which produces a
byte-identical transcript anyway at batch 2 but is a useful thing to be able to
do. Setting it above 4 does not raise the ceiling; it is clamped.

## Resource limits

Every limit below is enforced in code with a test, and none is a default that an
environment variable can remove.

| Resource | Limit | Behaviour when hit |
|---|---|---|
| Concurrent jobs | 4 | `enqueue` refused with `ResourceExhausted`, before any row, directory or thread exists. |
| Jobs held | 68 (4 running + 64 queued) | Same refusal, naming the fix: remove one from History. |
| Decoded audio per file | 8 hours (1.8 GB at 16 kHz mono f32) | Decode refused with a message to split the file. |
| Initial buffer reservation | 512 MB | Capped from the container's declared duration, which is untrusted. |
| Encoder batch | 4 | Clamped; and a failed batched group is retried one chunk at a time rather than failing the job. |
| Decoded audio per job | ~51 MB for a 14-minute file | Multiplied by the concurrency limit, so at most ~204 MB of audio buffers at once. |

**Why a refused job is not a dropped job.** A limit refusal happens before the
work starts, so there is nothing to clean up: no job row, no work directory, no
thread. A job the user has cancelled does not hold its concurrency slot, because
cancellation is cooperative and the worker is unwinding rather than holding the
buffer that makes concurrency expensive.

**Why the decode ceiling is not configurable.** A ceiling an environment
variable can raise is not a ceiling. That one exists for the case where nothing
else is trusted, including the file itself.

## Security model

**The threat model is a hostile input file.** The application runs as the logged-in
user, reads files the user points it at, and writes only where that user can
write. It has no authentication, no authorisation and no privilege, so none of
those are attack surface.

What is enforced:

- **Path traversal on export.** Every output name goes through `safe_file_name`,
  which takes the last component, rejects `.` and `..`, and renames a Windows
  device name. Every write is then checked with `assert_inside`, which
  canonicalises both sides so a symlink cannot escape either.
- **Subprocess execution.** None in production code. The only `Command::new`
  calls are in tests, spawning the test binary itself.
- **`unsafe`.** Confined to `vaani-inference` (`session.rs`, `capability.rs`,
  `runtime.rs`), `vaani-desktop/src/host.rs` (COM file dialogs), `vaani-cli`
  (console attach) and `vaani-bench` (DXGI queries). Each block carries a comment
  justifying it. No `unsafe` outside those.
- **Downloads.** SHA-256 verified before the file is moved into place; the bytes
  land in a `.partial` directory and only become a model after the hash matches.
  Redirects are followed to a different host, which is normal for a CDN and is
  why that is allowed rather than blocked.
- **Deserialisation.** `serde_json` on model configuration and fixtures. No
  `bincode`, no `unsafe` deserialisation, no dynamic code.
- **Secrets.** None are stored, required or logged. The log records paths,
  durations, error codes and the active provider; it records no audio content and
  no transcript text.

**Not implemented, and therefore not claimed:** drag-and-drop of files
(Slint 1.18 exposes no winit file-drop callback), audio playback, and any network
feature other than the model download.

## Performance characteristics

Measured on the machine in [Verified behaviour](#verified-behaviour), CUDA,
interleaved A/B, medians of three. `scratch/clip.wav` is a 834-second Hindi
meeting, 1908 words, 23 chunks.

| | serial | shipped (batch 2) |
|---|---|---|
| total | 33.22 s | 27.70 s |
| time to first result | 1729 ms | 2252 ms (1.30x) |
| longest gap between results | 2413 ms | 2620 ms (1.09x) |
| transcript fingerprint | `c140c2e235e3a375` | `c140c2e235e3a375` |
| words | 1908 | 1908 |

**The transcript is byte-identical.** At batch 2 the batched encoder produces
the same words at the same timestamps as the serial path. At batch 4 it does
not — one word differs (`कंप्लीट` against `कम्पलीट`, a transliteration spelling
variant of the same English word, at identical timestamps) — which is why 2 is
the default and 4 the ceiling.

**Time to first result got 30% worse and stayed within budget.** Batching the
encoder means waiting for a second chunk before the first can be encoded. The
front end was then made lazy, one group ahead, so the first encode starts after
one front end rather than all 23. Before that fix it was 3.58x worse — a 6.99 s
wait — on a job that was simultaneously getting faster.

**What was measured and rejected**, recorded so it is not retried blind:

- *Parallelising the log-mel front end across all 16 cores made it 60% slower*
  (2123 ms to 3185 ms summed; 16.0 s to 17.3 s wall). It is memory-bandwidth-
  bound, not compute-bound, and 2- and 4-thread pools did not help.
- *Batching at 4 or more is faster still* (13.0 s) but changes the transcript.
- *CUDA graphs* crash in `ort 2.0.0-rc.13` with `expected 'typeinfo_ptr' to not
  be null`, on both the normal and the IO-binding path.
- *Input-buffer reuse in the decode loop* bought 915 µs to 899 µs per call and
  was removed.

**Accuracy is not measured.** There is no human reference transcript in this
repository and no WER, CER, DER or JER number is claimed. The evaluation harness
in `evaluation/` is built and has 8 clips cut; all 8 lack a human reference, and
the harness is hard-wired to refuse scoring a model-derived reference as ground
truth. That is a deliberate refusal and has not been worked around.

The fp32 encoder in `artifacts/sravaani/encoder.onnx` is used as a *pseudo*-reference
to compare two execution paths against each other. That measures the distance
between executions, not accuracy against speech.

## Monitoring and logs

**One log file, truncated on every run** so it cannot grow without bound somewhere
the user cannot find it:

```
%APPDATA%\VaaniDesk\vaanidesk.log
```

It sits beside the database in the same `VaaniDesk` directory, so there is one
folder to look in. Verified on this machine: `dirs::data_dir()` resolves to
`%APPDATA%` (Roaming) on Windows, *not* `%LOCALAPPDATA%`, and the file was
confirmed at that path.

Startup writes one line that answers the question a user actually asks — *why is
it slow*:

```
INFO VaaniDesk: runtime ready runtime=GPU acceleration available -
  NVIDIA GeForce RTX 3050 Ti Laptop GPU, 3.3 GB, driver 13.3, compute 8.6
```

The same information is on screen in the status bar before any job runs. If it
says `CPU only` while an NVIDIA GPU is installed, the CUDA provider DLLs are not
next to `onnxruntime.dll`: install the GPU flavour.

Per job, the log records phase transitions, the active provider, the measured
front-end/encode/decode split, and the failure chain on error. It records no audio
content and no transcript text.

## Failure recovery

| Failure | What happens | What survives |
|---|---|---|
| Unsupported or corrupt file | Job fails with a specific reason and a next step. | Everything; nothing was written. |
| File longer than 8 hours | Refused, told to split it. | Everything. |
| Model missing | Refused with the download instruction. | Everything. |
| Model corrupt | Refused; the hash check is what notices. | The `.partial` file, for a resumed download. |
| No GPU, or no CUDA DLL | Falls back to CPU and says so. | Everything, ~6x slower. |
| Job cancelled | Partial transcript discarded, so no half transcript is shown as complete. | The file; nothing else. |
| Process killed mid-job | Work directory survives; swept at next launch. | The file. |
| Export interrupted | Temporary file is dot-prefixed and `.tmp`; the previous export is untouched. | The previous export, complete. |
| Disk full | Job fails. | The previous export, complete. |

**Exports are atomic.** An export is written to a temporary file *in the same
directory* and renamed over the target, because a rename within a filesystem is
atomic and a cross-filesystem copy is not. A reader sees either the complete
previous file or the complete new one, never a half-written transcript. The
previous file is the user's only copy of twenty minutes of work.

**Model downloads are atomic in the same way**: `.partial` directory, hash, then
rename, with the receipt written last.

## Known limitations

Stated plainly, because each has been measured rather than assumed.

- **No accuracy measurement.** See above. This is the largest gap.
- **The 60-second chunk cap can cut mid-word.** `max_ms` in the chunk policy is
  60 s, and a monotone speaker with no pause reaches it. Overlap is 0, which is
  safe only because boundary alignment puts cuts in pauses — and the cap is
  exactly the case where that assumption does not hold.
- **Speaker separation is heuristic and can be wrong in both directions.** It
  over-splits a single expressive speaker and over-merges two similar voices. On
  the reference recording the feature vector's window distances are close to
  uniform, so the speaker count the engine returns is a function of the
  threshold rather than of the meeting.
- **Batch 2's transcript equivalence is verified on one recording.** Not a corpus.
- **Concurrency is capped at 4, not tuned.** It is bounded, not benchmarked
  against a queue-depth sweep.
- **Drag-and-drop, audio playback, and an installer do not exist.**
- **Long-run behaviour is characterised by design, not by a multi-day soak.** The
  leaks that a soak finds — worker handles and work directories — were found by
  audit and fixed, with tests. A 24-hour run has not been performed.

## Rollback

Every commit is independent and the tree is the release. To roll back:

```
git log --oneline -20          # find the last good commit
git revert <bad>..<good>       # or: git revert <bad>
```

The database schema is additive; a newer build's rows are ignored by an older
build rather than misread. There is no migration to undo.

To roll back the *artefact* rather than the source, keep the previous ZIP. The
portable directory has no installer state, so replacing the folder is the whole
rollback.

## Verifying a build

```
.\scripts\production-readiness.ps1
```

Returns non-zero on any release-blocking failure. Stages: clean tree, no private
media in history, formatting, clippy, type check, the full workspace suite, the
resource-limit / resource-leak / performance suites named separately, the
nine-stage build doctor, the real-media ship gate, golden token equality,
packaging with self-test, and verification of the staged artefact rather than
the build directory.

`-SkipRealMedia` is for a machine without the private reference recording. It is
not the default, and it prints exactly which real-media assertions were
therefore **unverified** rather than implying coverage that does not exist.

---

## Verified behaviour

Every measurement in this document was taken on:

- Windows 11, PowerShell 5.1
- 16 logical cores, 15.4 GB RAM, 4.7 GB free at the time of the measurements
- NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GB VRAM, driver 13.3, compute capability 8.6
- ONNX Runtime with the CUDA and cuDNN 9 providers
- `ort 2.0.0-rc.13`
- SraVaani 1.0, 886.9 MiB of weights, 887 MB of encoder

The GPU throttles severely under sustained load: the same configuration measured
13.0 s cold and 38.0 s after repeated runs. Every performance figure above is an
**interleaved** A/B with medians, and a single cold run should not be read as a
baseline.

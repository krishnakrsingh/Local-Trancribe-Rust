# The journey so far

This is the running log of how VaaniDesk got to its current state, written for
the next person (or the next session) who has to pick it up without re-reading
six agents' reports. It records decisions and their reasons, not just outcomes.

The authoritative record of *what is true right now* is the gate, not this file.
If the two disagree, the gate wins and this file is stale.

---

## The product

A native Windows desktop application that transcribes meeting recordings
end to end: **media → VAD → ASR → diarization → postprocess → structure →
storage → UI → export**. Real models, real audio, no ffmpeg, no web view, no
localhost server.

The test case is a private ~14-minute meeting recording that has to come out
the other end with a correct transcript, correct word timings and correct
speaker attribution.

## How the work was organised

Eleven subsystems, built by parallel agents, against **one frozen contract**.

The contract lives in `vaani-core`, which has no third-party dependencies. Every
cross-boundary type is defined there once: `Transcript`, `Segment`, `Word`,
`Speaker`, `JobPhase`, `Progress`, `CancelToken`, `AudioBuffer`, `MediaInfo`,
`SpeechTimeline`, `SpeakerSpan`, `DiarizationResult`, and the stage traits
`VadEngine`, `DiarizationEngine`, `PipelineStage`. That is what let eight agents
work at once without negotiating types at merge time.

Ownership is in `docs/agent/PARALLEL_OWNERSHIP.md`. Each agent got a git
worktree, a branch and its own `CARGO_TARGET_DIR`. Every crate's `Cargo.toml` was
written by the architect *before* the agent started, so a missing dependency is
discovered in the first minute rather than at integration.

### Why the `vaani-` prefix

The brief said `crates/media/`, `crates/core/`, `crates/ui/`. The repo already
used `vaani-` for its three validated crates. Adopting the brief's literal paths
would have produced `crates/core/` sitting next to `crates/vaani-core/` — two
crates both claiming to be the domain model, which is precisely the ambiguity
the contract was meant to eliminate. The prefix won, everywhere.

---

## Phase 1 — the validated ASR foundation

Before any of the parallel work, the SraVaani 1.0 model was exported to ONNX and
made to reproduce the reference implementation **exactly**. This is the part of
the project that was hardest to get right and the part that must not regress.

Four bugs were found, each of which degraded every transcript silently:

1. **`torch.stft` centres its analysis window.** A 400-sample window sits at
   `[56, 456)` in a 512 frame. Left-aligning shifts the spectrogram 56 samples
   and the decoder never says so.
2. **Power spectrum must read the imaginary part from bin `k`.** For a real
   input `Re(X[n-k]) == Re(X[k])`, so reading the mirrored bin computes
   `re² + re²` — a 3 dB error on every bin, forever.
3. **The encoder's valid length is the `enc_lengths` output**, not the tensor
   dimension. The log-mel carries one extra STFT frame, so the tensor-derived
   figure decodes one frame past the end of the audio.
4. **Tokenizer pieces concatenate into whole SentencePiece words.** Emitting
   them individually splits words.

These are locked in by golden fixtures:

| Gate | Value |
|---|---|
| Log-mel features | max deviation **7.99e-05** across all 128×101 cells |
| Token IDs | **202 / 202** exact |
| Word timings | **108 / 108** exact, worst deviation **0.0 ms** |
| Transcript text | exact match |

`scripts/build_doctor.ps1` runs nine gates: toolchain, link, fmt, check, clippy,
test, runtime, model, golden. **If the golden gate ever regresses, stop.** That
is a bug, not a refactor.

---

## Phase 2 — the contract freeze

`crates/vaani-core/src/pipeline.rs` and the `AudioBuffer` type, then eleven
manifest-only crate skeletons so that `cargo check` was a real gate from the very
first edit by any agent.

`AudioBuffer` carries a non-optional `start_time`, and its documentation says
why: times are absolute once they leave a stage, and the ASR decoder is the one
exception because its frame indices are chunk-relative by construction.
`Transcript::push_chunk` is the single rebasing point. Getting this wrong in
either direction is the "transcript is three minutes out of sync" family of
bugs, and the type now refuses to be ambiguous about it.

### One design decision worth recording

`DiarizationEngine` takes an `AudioBuffer`, not a transcript. Diarization is
therefore independent of ASR: it can be swapped, tuned, or turned off without
touching a line of ASR code, and "ASR only" is a complete product mode rather
than a degraded one.

---

## Phase 3 — wave 1, eight agents

| Agent | Crate | Outcome |
|---|---|---|
| A1 | `vaani-media` | 137 tests. Decode RTF **0.0004** (2400× real time) on the real clip. |
| A3 | `vaani-vad` | 170 tests. RTF **0.0091**. Neural path present but unexercised — no Silero weights. |
| A4 | `vaani-diarization` | 278 tests, 4 ignored. RTF **0.005**. Heuristic, and honestly labelled as such. |
| A5 | `vaani-ui` | 159 tests. Real native Slint window, **launched and screenshotted**. |
| A6 | `vaani-storage`, `vaani-export` | 248 tests. Migrations, FTS5, atomic writes, five export formats. |
| A7 | `vaani-models` | 149 tests. Manifest, streaming verification at **727 MB/s**, atomic installs. |
| A8 | CLI + CI + installer | *delivered in wave 2* |
| A10 | QA / conformance | *delivered in wave 2* |

All six wave-1 branches merged with **zero conflicts**, which is the whole point
of the ownership contract.

### The bugs agents found in their own code

Every one of these was found by a test written to fail, not by inspection:

- **A1:** the resampler's `finish` duplicated the last chunk of every file; the
  final ~70 ms of every decode was emitted twice. Also: a clean-ending short
  file reported "not truncated" with no warning, so a user would have got a
  clean report and 0.003% of a transcript.
- **A3:** the power spectrum summed over the 257 non-negative bins, which carry
  half the spectrum — a 3.01 dB error on every frame. And a gated-floor deadlock
  where the noise floor could only move on ungated frames, so once everything
  was gated it could never move again.
- **A4:** `embed_windows` read the wrong audio for **every window after the
  first** — a missing `lo` meant every turn in the file was a copy of turn zero.
  Invisible in a speaker count; visible as a pitch dimension reporting 131 Hz on
  every window of a file whose second voice is 217 Hz.
- **A5:** an indeterminate progress bar is an animation, so a window full of
  finished jobs swept forever at **23.7% of a core doing nothing**.
- **A6:** `PRAGMA integrity_check` reported a malformed FTS5 index on a provably
  intact database; and the atomic temp name used only the process id, so two
  exports in one process collided.
- **A7:** a lock file *inside* the model directory could not be deleted on
  Windows while held, because a file cannot be deleted without
  `FILE_SHARE_DELETE`.

### A4, honestly

A4 stopped mid-task with a real fork: the test suite and its own diagnostic
disagreed, and the honest purity number was 0.57 rather than the 0.97 it had
been reporting. The 0.97 had been achieved *while the feature floors were
suppressing every dimension* — the tests were passing on a near-degenerate
feature vector.

The decision was to keep tuning rather than re-baseline the assertions down to
the measured 0.57, on the grounds that a diarizer which splits one steady voice
into two speakers is a bug, not a documented limitation. It then found the
window-indexing bug above and reached **0.96 purity**, with a stated operating
region: reliable for turns of ≳12 s, documented as failing below ~4 s, and an
explicit list of five known failure modes.

It is still a heuristic baseline with no trained speaker embedder, and its
report says so in those words.

---

## Phase 4 — wave 2, recovered

`vaani-app` (the orchestrator), `vaani-bench`, the CLI binary, the QA suite and
the CI workflow were found **untracked in the main checkout** rather than in
their worktrees. `vaani-bench` did not compile.

They were recovered onto `recovery/wave2-wip` before anything was touched, so
none of it could be lost, then merged and verified. Five real defects came out
of that, all of which had been invisible while the code was merely compiling:

1. **`vaani-bench` did not compile** — a `Result` treated as an `Iterator`, and a
   `SIZE_T` stored as `u64` without a cast.
2. **A completed job reported no transcript.** The store stage announces
   `Complete` through the progress channel *before* the worker's own terminal
   event, and that path published a summary with stale transcript fields. A
   consumer acting on the first terminal event it sees — the correct thing to
   do — read the job as having no result at all. The UI would have flickered
   from "complete, empty" to "complete, with transcript".
3. **Export into a new folder failed** with a bare OS error. A save dialog hands
   back new folders routinely. Directory targets are now created; a typed *File*
   path in a missing folder still errors, because silently building a directory
   tree from a typo is worse than a clear failure.
4. **`AudioBuffer` was never re-exported from the `vaani-core` root**, and
   `vaani-app`'s fakes were `#[cfg(test)]`, so no cross-crate test could reach
   the pipeline's negative paths.
5. **Two `vaani-media` cancellation tests passed in debug and failed in
   release.** They slept 30 ms and then cancelled, racing a decoder that runs at
   2400× real time — so the cancel landed *after* the stream had already
   finished. They now cancel synchronously after one chunk, which means the
   same thing at any decode speed. **A1 had reported both as passing; they only
   ever passed in an unoptimized build.**

That last one is the general lesson, and it is why the gate runs release.

---

## Phase 6 — the product gaps, closed

Two parallel agents, working against the merged tree, fixed the two largest
things the end-to-end run exposed and then went looking for the rest.

**A11 built the grouping step.** The policy is worth stating in plain terms,
because it is a product decision rather than an implementation detail: start a
new segment when the speaker changes, when the segment would pass 84 characters
(two lines of 42, the caption budget), when a word ends more than 12 s after the
segment's first word, or after a long enough silence. One rule is deliberately
*not* a break: if a length cap would fire within three words of a pause, the
break waits for the natural boundary, because a cap firing one word before a
pause manufactures a one-word segment — the exact thing this crate exists to
remove.

The measurement contradicted the obvious assumption. 1,889 of 1,911 inter-word
gaps are **exactly zero**, because the ASR adapter gives each word the frame
immediately after the previous one, so the recogniser reports no inter-word
silence at all. The pause threshold is therefore nearly inert here and fired 6
times out of 226 breaks; the speaker turn carries 83%. That is not a reason to
remove it — an engine that reports true silence would need it — but it is the
reason the default is documented as a *plateau* (a 15× range of thresholds
moves the segment count by 3) rather than a tuned value. Reading a plateau as a
peak would have been the mistake.

A11 also found a real bug in its own first version: `embed_windows` read the
wrong audio for every window after the first, because a missing `lo` meant every
turn in the file was a copy of turn zero. Invisible in a speaker count, visible
as a pitch dimension reporting 131 Hz on every window of a file whose second
voice renders at 217 Hz.

**A12 shipped the offline bundle.** Discovery now checks `<app>/models/` first,
the ~887 MiB of MIT weights ship with the installer, and a user on a clean
machine with no network and no token transcribes. Verified for real, not
simulated: the distribution was extracted outside every checkout with
`LOCALAPPDATA` redirected, every `VAANI_*` variable cleared, Python removed from
`PATH`, and the HTTP proxies pointed at a closed port. The decision to ship
**CPU-only by default** is the interesting one — 1.25 GiB of the difference is
NVIDIA redistributables, and NVIDIA's SDK agreement §2.5 makes bundling them in
an MIT product a question for counsel rather than a build-script detail. GPU
remains one documented copy of files.

**Phase 7 fixed what the agents were forbidden to touch.** `vaani-core` and
`vaani-inference` are architect-owned, so six agents reported defects in them
and none could act. Those are the `VAANI_RUNTIME_DIR` bug, the `doctor` summary
that claimed GPU acceleration on a CPU-only install, three missing error codes,
the missing media clamp, and the build doctor itself reporting three false
failures while discarding the reason.

The build doctor is the one worth dwelling on. It had been reporting
`check`, `clippy` and `test` as FAIL while the underlying cargo commands were
green, because `Invoke-Stage` took a scriptblock whose body piped to
`Out-Null` — discarding the very output that would have explained the failure.
Underneath that were two PowerShell problems: `$Profile` collides with the
automatic profile-path variable, and splatting an array from inside that
scriptblock reached cargo as a bare `-`. A gate that lies is worse than no gate,
because it trains everyone to ignore it. It now prints its own failure reason.

---

## Where it stands

`cargo fmt --all --check`, `cargo clippy --workspace --all-targets -D warnings`
and `cargo test --workspace` are all clean, and `scripts/build_doctor.ps1`
reports all nine stages healthy. Every agent's claims above were re-verified by
running them, not taken on trust — which is how the A1 cancellation tests, the
three `vaani-app` failures, the golden digest's line-ending fragility and the
doctor's own three false failures were caught.

**1,602 tests pass across 91 test binaries**, 0 failures.

The ASR foundation has not moved through any of it: 202/202 token IDs, 108/108
word timings at 0.0 ms worst deviation, exact text, on the real 40-second
reference clip.

---

## Phase 5 — the first end-to-end run

Everything above was eleven subsystems measured one at a time. The full chain
had never been run on the real meeting. It has now, in release, through the
production path (`vaani transcribe`, the real `ModelManager`, the real
`EngineRegistry`, the real CUDA provider), and the gate is
`scripts\e2e\acceptance.ps1`.

On a 13.9-minute / 834.197 s recording, an RTX 3050 Ti with 3.3 GB of VRAM:

| stage | wall | share |
|---|---|---|
| model open + load (once) | 9.5 s | 18 % |
| ASR, 23 chunks | 33.0 s | 62 % |
| VAD | 6.7 s | 13 % |
| diarization | 3.8 s | 7 % |
| decode to mono 16 kHz | 0.12 s | 0.2 % |
| storage + five exports | 0.12 s | 0.2 % |
| **total** | **53.3 s** | |

**A 14-minute meeting takes 53 seconds, and ASR is 80 % of it.** Within ASR the
encoder is 21 s of the 33 s, the decoder 7.6 s and the log-mel front end 4.4 s.
Peak working set 1.49 GiB. Whole-job RTF 15.0; ASR alone 25.3.

**Diarization does not dominate.** At 3.8 s it is 7 % of the job. The brief's
guess that it might was wrong, and the thing that would make it dominate —
a trained speaker embedder — does not exist yet.

The result is *correct*, not merely produced: word timings are monotonic and in
bounds, every one of the 1,913 chunk-local segments is present in the transcript
at exactly `chunk_offset + chunk_local`, the transcript touches 230 of the 240
detected speech spans, speaker shares match measured talk time, the SQLite round
trip is field-for-field equal, and both subtitle files parse and check clean.
Cancellation takes 7 ms to take effect. Those are now asserts in
`crates/vaani-cli/tests/e2e_acceptance.rs`, not observations.

### What the run found

1. **Nothing groups words into segments.** `SraVaaniEngine::tokens_to_segments`
   emits one `Segment` per `Word` and says in its own doc comment that "segments
   are later merged into phrases and attributed to speakers by diarization". The
   attribution half happens. The merging half exists nowhere in the repository.
   The shipped transcript is 1,913 words in 1,913 segments, and the `.srt` is
   1,913 cues of one word each for a 14-minute meeting. This is the largest
   product gap in the project and no per-stage measurement could have found it,
   because every stage is behaving exactly as it was specified to.
2. **`RuntimeLocation::discover` ignores `VAANI_RUNTIME_DIR`.** It returns the
   variable's value as `ort_dll` instead of joining `onnxruntime.dll` onto it,
   so setting the variable makes the runtime report as *absent*. `vaani-inference`
   is frozen; reported, not fixed.
3. **`--out <a directory that does not exist>` failed** with a bare
   `os error 3` *after* the whole 14-minute job, on the CLI. The wave-2 fix for
   this ("directory targets are now created") went into `vaani-app`'s export
   stage, which the CLI does not use.
4. **An unknown `--model` id returned `ErrorCode::Internal`.** The id is
   something the user typed, and `internal` tells them nothing and asks them to
   file a bug. Fixed to `model_unavailable`.
5. **A segment ends 23 ms past the end of the recording.** `push_chunk` clamps
   chunk-local times at zero and has no media duration to clamp against.
   `vaani-core` is frozen; reported, not fixed.
6. **The VAD calls 86.5 % of this meeting speech.** Plausible for a recorded
   meeting, high enough to be worth a look.
7. **Two speakers for a 14-minute meeting.** Consistent with A4's stated
   operating region and its own honest labelling, but worth saying plainly: the
   diarizer is very likely under-counting here.

## What is still open

Ordered by how much it matters to shipping, not by convenience.

**Closed since the first end-to-end run** — kept here so the record shows what
was true when, not only what is true now.

- ~~Nothing groups words into segments.~~ **Closed by A11.** 1,912 one-word
  segments became 227 readable ones; the SRT went from 1,913 single-word cues to
  265. The policy was measured rather than guessed: 83% of the segmentation is
  the speaker turn, and the pause threshold turned out to be a plateau rather
  than a cliff (a 15× range of thresholds moves the segment count by 3).
- ~~`RuntimeLocation::discover` does not join `ORT_DLL` onto
  `VAANI_RUNTIME_DIR`.~~ **Closed.** Fixed, with a test that asserts the DLL
  path the override must produce.
- **`RuntimeReport::summary()` claimed GPU acceleration on a CPU-only
  install.** Not previously listed. Fixed; the summary now distinguishes usable
  GPU from present-but-unusable, and the self-test reports the real provider.
- ~~`ErrorCode` is missing variants.~~ **Closed.** `Interrupted`,
  `ResourceBusy` and `UnknownModel` all exist and are wired through, and a
  crash-killed job is no longer recorded as "cancelled".
- ~~A segment ends 23 ms past the end of the recording.~~
  **Closed by `Transcript::push_chunk_within`**, used by the app and both CLI
  transcribe paths.
- ~~The build doctor reports FAIL.~~ **Closed.** It had been reporting three
  false failures while discarding the reason; the cause was a PowerShell
  splatting bug in the script itself, not in the code.

**Still open:**

1. **Diarization is a heuristic, not a model.** No embedding weights exist. It
   is usable and honestly bounded, and it is the weakest link in the pipeline.
   A4 has named the exact operating region, and the first real run confirmed it:
   2 speakers for a 14-minute multi-party meeting is very likely an
   under-count. **This is now the single largest quality risk in the output.**
   The grouping policy leans on speaker turns (83% of breaks), so diarization
   errors now propagate into the transcript's structure, not just its labels.
2. **No SRT or VTT parser has validated the export.** A6 wrote reference
   parsers implementing the published rules; no third-party parser and no real
   player has checked the bytes. Phase 5 checked them with
   `vaani_export::verify` on the real transcript: both parse, both pass the
   two-line and no-overlap rules, and the cue count matches. Still no real
   player. On the release checklist: open one `.srt` and one `.vtt`.
3. **Downloads are unverified against the network.** There is no network on the
   build machine. A7 tested the entire transfer path over real TCP against a
   local HTTP/1.1 server, including `Range`, resume, redirect-to-CDN, auth and
   truncation — but the actual Hugging Face endpoint and its xet bridge have
   never been contacted. A7 names this as the most likely first real-world
   failure. The offline bundle is unaffected: it ships in the installer and was
   verified on a clean machine with no network at all.
4. **The SraVaani licence needs written confirmation before release.** A12
   verified the grant three ways and MIT permits redistributing the weights, but
   the upstream repository publishes no LICENSE file and states no copyright
   holder in machine-readable form. The distribution ships the MIT text with a
   provenance note that says so plainly. Get written confirmation before any
   public or commercial release.
5. **The GPU distribution is not bundled by default, deliberately.** NVIDIA's
   SDK agreement §2.5 is a question for counsel, not a build script, so the
   default distribution is CPU-only (925 MiB, 2.5–4× slower) and the GPU
   flavour is a documented copy of DLLs. Decide this before release.
6. **Paint performance is unmeasured.** A5 measured all the state work; nobody
   has measured frame time with a real transcript on screen.
7. **`Transcript::duration` is a length while segment times are absolute.**
   Nothing in the type distinguishes them, and it has already caused two real
   bugs in two different crates. A11 works around it with `MediaBounds`; the
   contract itself still permits the mistake. This is the highest-value
   remaining change to `vaani-core`, and it is a change to ten crates' call
   sites rather than a bug fix.
8. **The VAD calls 86.5% of this meeting speech.** Plausible for a recorded
   meeting, high enough to be worth a look.
9. **The CLI has no way to cancel a job.** Cancellation is instant once
   requested (5.5 ms) and the desktop app has a real cancel button, but the
   shipped CLI has no trigger for it.

Every defect in this project was a thing that was *plausible* and not *checked*:
a power spectrum missing a factor of two, a window index missing an offset, a
cancel that raced a fast decoder, a summary that reported a phase without the
fact that phase implied.

Phase 5 supplied the sharpest example yet. Every stage in this pipeline is
correct, the timings are good, and the product is still not shippable, because
the step *between* two stages — grouping words into segments — was assumed by a
doc comment and never built. A green per-stage benchmark is not a green product,
and the only thing that found this was running the whole chain once on real
audio.

The defences that actually caught them were golden files, tests written to fail,
and measuring the claim instead of repeating it. Keep those three.

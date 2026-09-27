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

## Where it stands

`cargo fmt --all --check`, `cargo clippy --workspace --all-targets -D warnings`
and `cargo test --workspace` are all clean. Every agent's claims above were
re-verified by running them, not taken on trust — which is how the A1
cancellation tests and the three `vaani-app` failures were caught.

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

1. **Nothing groups words into segments.** The single largest gap in the
   project, found only by the first end-to-end run. Every stage does exactly
   what it was specified to do; the specification is what is missing. A
   transcript is a word-per-line list and an `.srt` is one cue per word.
   Deciding the grouping policy is a product decision (by pause? by speaker
   turn? by maximum length?) that changes every downstream contract, so it is
   not something an acceptance pass should have invented on its own.
2. **Diarization is a heuristic, not a model.** No embedding weights exist. It
   is usable and honestly bounded, and it is the weakest link in the pipeline.
   A4 has named the exact operating region.
3. **`RuntimeLocation::discover` does not join `ORT_DLL` onto
   `VAANI_RUNTIME_DIR`**, so the one documented way to point the app at a
   runtime makes that runtime report as missing. `vaani-inference` is frozen.
4. **No SRT or VTT parser has validated the export.** A6 wrote reference
   parsers implementing the published rules; no third-party parser and no real
   player has checked the bytes. Phase 5 checked them with
   `vaani_export::verify` on the real transcript: both parse, both pass the
   two-line and no-overlap rules, and the cue count matches. Still no real
   player. On the release checklist: open one `.srt` and one `.vtt`.
5. **Downloads are unverified against the network.** There is no network on the
   build machine. A7 tested the entire transfer path over real TCP against a
   local HTTP/1.1 server, including `Range`, resume, redirect-to-CDN, auth and
   truncation — but the actual Hugging Face endpoint and its xet bridge have
   never been contacted. A7 names this as the most likely first real-world
   failure.
6. **`ErrorCode` is missing two variants** that three agents had to work around:
   `Interrupted` (a job killed by a crash currently masquerades as `cancelled`)
   and `ResourceBusy` (a lost install race and a locked model both surface as a
   storage error). A third is now visible: there is no `UnknownModel`, so a
   user who mistypes `--model` gets `model_unavailable`, whose title is close
   but not right.
7. **100% offline, pre-bundled models (architect mandate).** SraVaani is
   MIT-licensed, so the production weights — `encoder.fp16.onnx`,
   `decoder_joint.onnx`, `tokenizer.model`, the filterbank matrices and the
   configs, ~930 MB — are bundled permanently under `models/sravaani/`. No
   first-launch token prompt, no download gate on the baseline engine, and
   discovery checks `<app>/models/` first. This supersedes the earlier
   `gated-token` constraint: the Hugging Face gate no longer stands between a
   user and a working transcript.
8. **Paint performance is unmeasured.** A5 measured all the state work; nobody
   has measured frame time with a real transcript on screen.
9. **`push_chunk` has no upper clamp.** A decoded frame can land past the end of
   the recording — 23 ms did, on the real clip. `vaani-core` is frozen.

## The rule that keeps producing the bugs

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

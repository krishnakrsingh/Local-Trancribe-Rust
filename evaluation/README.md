# The VaaniDesk accuracy evaluation

**Status: the harness is built, the clips are cut, the system has been run on
every clip, and not one clip has a human reference.** Nothing in `results/` is
an accuracy measurement, and this file says so before anything else does.

---

## What this is for

The project has been validated *structurally*: word timings are monotone,
chunk-local segments are correctly rebased, the transcript round-trips through
SQLite, the exports parse. None of that says the words are right. This
directory is the part that would say so — given something true to measure
against.

## The one rule

**Ground truth is what a person wrote down after listening to the audio.**

It is not the model's transcript. It is not the transcript with the model's
mistakes corrected. It is not a reference implementation's output, including
this project's own SraVaani reference implementation, whose 202 golden token
ids are a system output and therefore not ground truth either.

A reference derived from any model, in whole or in part, is a
**self-consistency check**. It is worth something — it catches a decoder that
has started drifting — and it is not accuracy. The harness cannot be talked
into accepting one: [`hypothesis::Hypothesis::is_reference_grade`] is
hard-wired to `false`, and `evaluation/scoring/src/set.rs` additionally
requires an `ANNOTATED.md` provenance note before it will score a reference at
all. A reference file that appears from nowhere is exactly how a fabricated
ground truth gets scored as a real one.

## Current state, precisely

| | |
|---|---|
| Clips cut and verified | **8**, each exactly 90.000 s, 16 kHz mono PCM16 |
| Clips with a **human** ASR reference (`reference.txt`) | **0** |
| Clips with a **human** diarization reference (`reference.rttm`) | **0** |
| Clips with a **human** word-timing reference (`reference.words.json`) | **0** |
| Clips the system was run on | **8** |
| Clips with a real WER / CER | **0 — `unscored`** |
| Clips with a real DER / JER | **0 — `unscored`** |
| Clips with a real timestamp deviation | **0 — `unscored`** |

**Zero. Not "pending review" — zero.** No human has listened to these clips.
Every per-clip block in `results/` reports `status: "unscored"` with a machine
reason, and every rate field is `null`. A `null` is not a zero: zero means
"measured, found none", and `unscored` means "not measured".

## Layout

```
evaluation/
  .gitignore          what may never be committed, and why
  set.json            the manifest: which clips, cut from where, with what structure
  SCHEMA.md           every file format, verbatim
  ANNOTATION.md       the protocol for the human who writes the references
  README.md           this file
  clips/              <id>.wav                     [GITIGNORED - private audio]
  references/         <id>/reference.txt           [GITIGNORED - what was said]
                      <id>/reference.rttm          [GITIGNORED]
                      <id>/reference.words.json    [GITIGNORED]
                      <id>/ANNOTATED.md            [GITIGNORED - provenance]
  hypotheses/         <id>/hypothesis.json         [GITIGNORED - the system's output]
                      <id>/hypothesis.txt          [GITIGNORED]
                      <id>/hypothesis.rttm         [GITIGNORED]
  metadata/           <id>.meta.json               [committed - structure only]
  results/            summary.json, <id>.report.json [committed - aggregates only]
  templates/          empty reference files to copy and fill in
  tools/              run_evaluation.ps1
  scoring/            the harness: a standalone Rust crate
```

## The three error classes, and why they are not combined

| class | means | metric |
|---|---|---|
| **ASR error** | wrong words | WER, CER, and the substitution/deletion/insertion split |
| **diarization error** | right words, wrong speaker | DER, JER, speaker count, turn errors |
| **timestamp error** | right words and speaker, wrong time | deviation statistics |

The report contains a field `blend` that is permanently
`{ "permitted": false, "value": null, "why": "..." }`. This is not decoration.
A system with 30 % WER and 5 % DER has a word problem. A system with 8 % WER
and 60 % DER has a speaker problem. A system with 8 %, 5 % and 400 ms of drift
has a timing problem. Any arithmetic that collapses those three into one number
is a weighted average of guesses, and the weights are always chosen *after*
seeing the result. So the harness refuses, in the output, every time.

## Running it

Everything below is one command, and it is reproducible from a clean checkout
plus a built CLI:

```powershell
powershell -ExecutionPolicy Bypass -File evaluation\tools\run_evaluation.ps1
```

It does, in order: cut the clips, probe each one, run `vaani-cli transcribe`
on each, convert every export into the harness's hypothesis form, and score the
set. Re-running it overwrites `clips/`, `hypotheses/` and `results/` and
changes nothing else.

The harness itself:

```powershell
$env:CARGO_TARGET_DIR = "$env:TEMP\target-eval"
cargo build  --release --offline --manifest-path evaluation\scoring\Cargo.toml
cargo test   --release --offline --manifest-path evaluation\scoring\Cargo.toml
cargo clippy --offline --manifest-path evaluation\scoring\Cargo.toml --all-targets -- -D warnings

& "$env:CARGO_TARGET_DIR\release\vaani-eval.exe" schema      # the report schema
& "$env:CARGO_TARGET_DIR\release\vaani-eval.exe" rules       # the normalisation rules
& "$env:CARGO_TARGET_DIR\release\vaani-eval.exe" score --set evaluation
```

`score` exits `1` when no clip could be scored, so "the harness ran" can never
be misread as "the system is accurate". `--allow-unscored` accepts a
zero-scored set and says so on stderr.

## Why `evaluation/scoring` is its own workspace

The root `Cargo.toml` is architect-owned and this track may not edit it, so
`evaluation/scoring` declares its own `[workspace]` table and keeps its own
`Cargo.lock`. It builds and tests with `--manifest-path` and does not perturb
the lockfile the other sixteen crates share. Registering it as a member is a
one-line architect change; see INTEGRATION NOTES.

## No Python, and no network

The whole path is Rust, offline, on this machine. The brief permits Python for
clip preparation and it turned out not to be needed: `vaani-eval cut` and
`vaani-eval probe` are Rust, so the clips are reproducible with no interpreter
in the loop. There is no `jiwer`, no `pyannote`, no Whisper-style reference,
and no model in the scoring path. The four dependencies — `serde`,
`serde_json`, `unicode-segmentation`, `sha2` — were all already in the local
cargo registry cache and resolve with `--offline`.

`unicode-segmentation` is not decoration. Without it, CER over Devanagari counts
one phoneme as two or three Unicode scalars and inflates the rate by a factor
that has nothing to do with recognition quality. The harness reports both CER
over scalars and CER over grapheme clusters and tells you which to quote.

## The next step

One person, a set of headphones, and `evaluation/ANNOTATION.md`. A competent
Hindi-and-English annotator can produce a usable `reference.txt` for a 90-second
clip in roughly ten minutes. Eight clips is one afternoon. That single afternoon
turns `results/` from eight rows of `unscored` into the first real accuracy
measurement this project has ever had.

See `INTEGRATION NOTES` in the track report for the full list of what an
annotator needs in order to do it properly.

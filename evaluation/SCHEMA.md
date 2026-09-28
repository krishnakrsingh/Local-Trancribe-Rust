# The evaluation file formats, verbatim

Every field, every default, every unit. A format that is described in prose is
a format that two annotators will interpret differently, and two references that
were built differently cannot be compared.

All times are **integer milliseconds** unless the field name says otherwise.
`reference.rttm` is the one exception and is in **decimal seconds**, because
that is what the NIST RTTM specification says.

---

## 1. `set.json` — the manifest

The single source of truth for the set. `evaluation/scoring/src/set.rs` reads
it and refuses a version it does not know or a set with no clips.

```jsonc
{
  "version": 1,                       // must equal SET_VERSION (1)
  "id": "vaanidesk-meeting-001",
  "description": "…",
  "source": {
    "path": "scratch/clip.wav",       // repository-relative, gitignored
    "sha256": "…",                    // of the SOURCE file
    "sample_rate": 16000,
    "duration_ms": 834197,
    "description": "A real private meeting recording…"
  },
  "normalisation_profile": "strict",  // strict | hinglish | hinglish-translit
  "clips": [ { …see below… } ]
}
```

Each entry in `clips`:

```jsonc
{
  "id": "clip01",                     // matches clips/<id>.wav
  "start_ms": 30000,                  // offset into the source recording
  "duration_ms": 90000,               // 60 000..120 000 per the brief
  "topic": "pause-dense-opening",     // short human label
  "rationale": "…",                   // why this clip, in one sentence
  "covers": ["natural-pauses", "…"],  // required characteristics, auditable
  "speakers": null,                   // null = nobody has listened yet
  "speakers_provenance": "unverified",
  "language_mix": ["hinglish"],
  "language_mix_provenance": "documented_context",
  "difficulty": "moderate",
  "difficulty_provenance": "unverified",
  "clip_sha256": "…",                 // of clips/<id>.wav
  "structure": { …measured acoustic summary… }
}
```

### The four provenance values

Every judgement that only a listener can make carries one, so a reader can
tell a measurement from an assumption without opening the file.

| value | meaning |
|---|---|
| `human_verified` | a person confirmed it by listening |
| `measured` | computed from the audio, by `vaani-eval probe` |
| `documented_context` | stated in this project's own documentation, not re-measured here |
| `unverified` | asserted with no evidence. **Every listener-only property starts here.** |

`speakers: null` with `speakers_provenance: "unverified"` is the correct value
for an unannotated clip. `speakers: 0` would be a claim that nobody spoke, and
`speakers: 2` would be a guess wearing a number.

### The `structure` object

Produced by `vaani-eval probe`. Contains no content.

```jsonc
{
  "sample_rate": 16000,
  "duration_ms": 90000,
  "speech_fraction": 0.5774,          // fraction of 20 ms frames above the floor
  "speech_runs": 45,                  // runs of >= 200 ms
  "silence_spans": 37,                // gaps of >= 300 ms
  "longest_silence_ms": 2990,
  "longest_speech_ms": 3840,
  "notable_pauses": [ { "start_ms": …, "end_ms": … } ],   // gaps >= 700 ms
  "rms_p05_dbfs": -101.7,
  "rms_p50_dbfs": -59.6,
  "rms_p95_dbfs": -14.3,
  "speech_zcr_median": 1000.0,         // crossings/s; a speaking-rate PROXY
  "noise_floor_dbfs": -90.4,          // the 20th-percentile frame level
  "frame_count": 9000
}
```

`notable_pauses` positions are structural (when nobody spoke) rather than
content (what was said), which is why they are committable while a transcript
is not.

---

## 2. `clips/<id>.wav`

16 kHz, mono, signed 16-bit PCM. Chosen so that no conversion is needed before
transcription and so that an annotator can play it on anything.

Written by `vaani-eval cut`, which reads the file back and **fails** if the
result is not exactly the requested length:

```
clip01.wav  1440000 samples  16000 Hz  requested 90000 ms  actual 90000 ms  sha256 …
```

---

## 3. `references/<id>/reference.txt` — the ASR reference

One utterance per line. Nothing else. No timestamps, no speaker labels, no
markdown, no blank-line sectioning.

```
यह एक परीक्षण है
अब API integrate करना है
```

### What the annotator writes

* **Every word that was said**, including fillers, in the script the speaker
  used. If a Hindi speaker said an English word, write the English word; do not
  transliterate it into Devanagari and do not translate it into Hindi.
* **Names, product names and API names as they were pronounced**, not as they
  are spelled in the marketing material. Write what was said.
* **Inaudible words**: write `[inaudible]`, once per word you cannot recover.
  Do not guess, and do not leave a gap — a gap is indistinguishable from a
  deleted word and turns into a false WER penalty.
* **Overlapping speech**: transcribe what you can hear, one line per speaker in
  time order, and mark the overlap in `ANNOTATED.md`. A diarization reference
  that pretends overlap did not happen produces a DER that rewards the system
  for being wrong.
* **Repeated words**: write them. `बहुत बहुत अच्छा` is three words, and the
  harness's `asr.duplicated_word` signature exists to catch the *hypothesis*
  repeating itself — it cannot do that if the reference is tidied up first.

### What the annotator does not write

No punctuation. No case for Devanagari (it has none) and no effort to "correct"
Latin capitalisation, because the normaliser lowercases both sides identically
and a capital in the reference is invisible. No sentence segmentation beyond one
utterance per line.

Punctuation is *not* scored at all: rule N4 of the normalisation deletes
punctuation from both sides, which is why `asr.punctuation_error` in the
taxonomy counts punctuation tokens in the hypothesis directly rather than
relying on WER. If you care about punctuation, that is a separate measurement
and this harness does not make it.

### The normalisation, stated completely

The full rule list is in `evaluation/scoring/src/normalize.rs` and is printed
by `vaani-eval rules`. In order:

| id | rule | why it exists in a Hindi meeting |
|---|---|---|
| N1 | strip a leading UTF-8 BOM | a Windows editor adds it; it is not a word |
| N2 | Devanagari (U+0966–U+096F) and Arabic-Indic (U+06F0–U+06F9) digits → ASCII `0`–`9` | a human writes `५`, a recogniser writes `5` |
| N3 | precomposed nukta letters U+0958–U+095F → base + U+093C; `ZWJ + U+093C` → `U+093C`; U+0929→U+0928, U+0931→U+0930, U+0934→U+0935; drop bare U+200C/U+200D | `ज़` is one phoneme written three ways; the U+0929 family are spelling variants, not distinct sounds |
| N4 | delete Unicode punctuation, symbol and non-space separator characters | the danda and the comma are sentence marks, not words |
| N5 | collapse every run of Unicode whitespace to one U+0020 | |
| N6 | delete U+0964 and U+0965 (danda, double danda) | restated explicitly: the danda is the commonest stray token in Hindi exports |
| N7 | lowercase every character outside the Devanagari block, single-character mappings only | Devanagari has no case; single-char-only keeps U+0130 from expanding inside a token |
| N8 | split on U+0020, drop empty tokens | |

`hinglish` adds **H1**: fold the documented precomposed Latin letters to base
ASCII (`ā→a`, `ṣ→s`, `ṇ→n`, `ṭ→t`, `ñ→n`, `ç→c`, `æ→ae`, `ß→ss`, …), then
lowercase. Use it when the reference and the hypothesis are expected to differ
in transliteration convention.

`hinglish-translit` adds **H2**: fold `aa→a`, `ii→i`, `uu→u`, `ee→i`, `oo→u`
inside a Latin token. **Lossy**: it merges `be`/`bee`, `to`/`too`, `me`/`mee`.
Opt in only when you know the two sides use different transliteration
conventions, and never quote it next to a `strict` number. `ai`, `au`, `oi` and
`ou` are deliberately not folded: they are diphthongs, and folding them merges
distinct words.

What the normalisation does **not** do, and this matters more than the list
above:

* **No transliteration between Devanagari and Latin.** `kya` and `क्या` stay
  different tokens. A code-switched meeting is scored in the script the speaker
  used; converting would hide exactly the code-switch boundary errors the
  taxonomy needs to see.
* **No stemming or lemmatisation.** WER here is lexical.
* **No filler removal.** `अरे`, `तो`, `uh`, `um` are words.
* **No number-word folding.** `पाँच` against `5` is one substitution plus one
  deletion. That is a true statement about the two texts, and hiding it would
  make the score incomparable to every published Hindi WER.

The rules are applied **identically** to the reference and the hypothesis.
There is no code path that normalises one side harder than the other, and the
rule list is embedded in every report so a number can never be separated from
the rules that produced it.

---

## 4. `references/<id>/reference.rttm` — the diarization reference

NIST Rich Transcription Time-marked. Ten whitespace-separated fields, in
decimal **seconds**:

```
Type FileID ChanID StartTime Duration Ortho Stype Name NoSpeakers Confidence
SPEAKER clip01 1 12.340000 3.450000 <NA> <NA> A <NA> <NA>
```

| field | value |
|---|---|
| `Type` | `SPEAKER`, `SPKR-INFO` (metadata, ignored) or `NOSPEAKER` (marked non-speech, **not** a turn) |
| `FileID` | the clip id. Never compared across files. |
| `ChanID` | `1` |
| `StartTime`, `Duration` | decimal seconds. The parser multiplies by 10^6 once and never divides again, so a 1 ms boundary is exactly 1000 µs and not `0.0009999…`. |
| `Name` | your speaker label. Any consistent label. The harness never compares labels as strings: a permutation is resolved first, so `A` and `SPEAKER_00` may name the same person. |
| `NoSpeakers`, `Confidence` | `<NA>` unless you have them |

Blank lines and lines starting with `#` or `;;` are skipped. A line with fewer
than ten fields, an unknown type, or a non-numeric time is an **error naming the
line number**, not a warning.

### Annotating diarization

* One record per contiguous stretch of one person talking. Adjacent records with
  the same speaker should be merged — a boundary between two `A` records is a
  change the human did not hear, and it will be counted against you.
* **Overlap needs two records with the same start and end.** A system that
  emits at most one speaker at a time cannot represent overlap, and the DER it
  can reach is bounded below by however much overlap the reference contains.
  The report prints `diarization.overlap_failure` for exactly this reason: it is
  a property of the reference, reported before any hypothesis exists.
* Cover the **whole clip**, including the silences — a `SPEAKER` record that
  stops at a pause is a false alarm waiting to happen.
* Assign labels consistently across clips. Consistency is not required for
  correctness (the permutation is per clip) but it makes a human reading eight
  reports able to tell whether the same person was called the same thing.

---

## 5. `references/<id>/reference.words.json` — word timings

Optional, and the only way the timestamp block becomes measurable.

```json
[
  { "text": "यह",  "start_ms": 0,    "end_ms": 240 },
  { "text": "एक",  "start_ms": 260,  "end_ms": 480 }
]
```

* `text` is **already-normalised** under the set's `normalisation_profile`.
  Run `vaani-eval norm` on your `reference.txt` and copy the tokens, or the
  alignment will pair the wrong things and every deviation will be noise.
* One entry per token, in order, matching `reference.txt` **line for line and
  token for token**.
* Times to the nearest 10 ms. Claim the precision you actually used in
  `ANNOTATED.md`; the harness reports deviation in milliseconds and a reference
  timed to the nearest 20 ms will show a 10 ms median for a perfect system.
* Word start and end are the span over which **that person is producing that
  word**, not the span of the pause after it.

Without this file the timing block reports
`status: "unscored", reason: "no_word_timing_reference"` and the rates are
`null`. It does not fall back to the hypothesis's own timings, because that
would measure the system against itself and call it a timestamp accuracy.

---

## 6. `references/<id>/ANNOTATED.md` — provenance

**Required.** A reference without it is not scored.

```markdown
---
clip: clip01
annotator: <name or handle>
date: 2026-09-29
listens: 2
rttm: yes
word_timings: no
timing_precision_ms: 10
overlap_annotated: yes
language_mix_confirmed: [hinglish]
speakers_confirmed: 3
notes: >
  One stretch of overlap at about 00:47 where the second voice cuts in over the
  first. The second speaker is quieter and further away.
---

Anything else the next person needs to know.
```

The point of this file is not bureaucracy. It is that a reference which claims
to be human and has no evidence of being human is indistinguishable, to the
harness and to a reader, from a reference someone fabricated. The fields above
are the minimum for a claim to be checkable.

---

## 7. `hypotheses/<id>/hypothesis.*` — the system's output

Produced by `vaani-eval hyp`, never by hand. All four files are gitignored.

| file | what it is |
|---|---|
| `hypothesis.json` | the `vaani.transcript` export, **byte-for-byte as `vaani-cli` wrote it**. The scorer reads this file, so what is scored is what the pipeline produced and not a re-serialisation of the harness's view of it. |
| `hypothesis.txt` | every word, in order, single-spaced. The input the normaliser reads. |
| `hypothesis.rttm` | one `SPEAKER` record per **attributed segment** — that is, the diarization a user actually reads, not the diarizer's internal spans. It is coarser than the raw spans because the grouping policy runs before attribution, so a DER measured this way includes the grouping policy's effect. Documented rather than hidden. |
| `hypothesis.summary.json` | counts and intrinsic facts only. |

Segments with no speaker are dropped from the RTTM. In ASR-only mode there is no
diarization hypothesis to score, and a `SPEAKER` record with an empty label
would be scored as a false alarm against nothing.

---

## 8. `results/` — the reports

Committed, aggregate-only. `summary.json` holds the set counts and
normalisation rules; `<id>.report.json` holds one clip. The schema is printed
by `vaani-eval schema` and the shape is:

```jsonc
{
  "version": 1,
  "generated_by": "vaani-eval 0.1.0",
  "normalisation": { "profile": "strict", "rules": [ … ], "does_not": [ … ] },
  "set": {
    "clips": 8,
    "human_asr_reference": 0,
    "human_diarization_reference": 0,
    "human_word_timing_reference": 0,
    "awaiting_annotation": 8,
    "with_hypothesis": 8
  },
  "clips": [ {
    "clip_id": "clip01",
    "scored": false,
    "asr":          { "status": "unscored", "reason": "no_human_asr_reference", "wer": null, … },
    "diarization":  { "status": "unscored", "reason": "no_human_diarization_reference", "der": null, … },
    "timing":       { "status": "unscored", "reason": "no_word_timing_reference", … },
    "intrinsic":    { …counts, not accuracy… },
    "taxonomy":     [ { "id": "asr.duplicated_word", "class": "asr", "status": "measured", "count": 2, … } ]
  } ],
  "blend": { "permitted": false, "value": null, "why": "…" }
}
```

### The three `status` values

| value | meaning |
|---|---|
| `scored` | a human reference exists and the numbers are measurements |
| `unscored` | no human reference. `reason` says what is missing. Rates are `null`. |
| `undefined` | a reference exists but the metric has no denominator — an empty reference, a clip with no speech turns. **Distinct from `unscored` and never rendered as `0`.** |

A zero-length reference makes WER and CER undefined, not 0.00 %. Zero would read
as "a perfect score for a clip where the human wrote nothing down".

### Metric definitions, as implemented

* `WER = (S + D + I) / N_ref`, `N_ref` = reference token count after
  normalisation. Never the hypothesis count, never the mean of the two, never
  `min`. Using the hypothesis count makes an empty hypothesis look perfect.
* `cer_characters` and `cer_grapheme` come from **two separate alignments**, not
  from the token error count divided by a character count. One substituted word
  is one token error and two or three character errors; dividing the first by
  the second understates the character rate. **Quote `cer_grapheme` for Hindi.**
  `cer_characters` exists for comparability with the literature and is labelled
  as such.
* `DER = (missed detection + false alarm + confusion) / total reference speech`,
  accumulated per elementary interval. Reference speech is the **sum of
  reference turn durations**, so overlap counts in both speakers.
  * The components partition the error without double counting, and the scorer
    asserts two closed forms on every call:
    `correct + missed + confusion == reference speech` and
    `correct + false_alarm + unattributed == hypothesis speech`.
  * A reference-side residual is *missed detection* when the hypothesis says
    nobody is talking there, and *confusion* when the hypothesis says somebody
    is — just not this person. Counting it as both is a convention in some
    published implementations; it is not this one, because it charges the same
    2 s twice and puts DER at 1.0 for a system that got half the audio right.
  * Speaker labels are resolved by an **exact maximum-weight assignment**
    (Hungarian) on a padded matrix that contains an explicit "unmatched" slot on
    both sides. Not `n!` enumeration, and not a greedy pass: the optimal
    assignment is a real constraint and a test asserts that a greedy one gets it
    wrong.
  * The collar is `0` by default. With `--collar-ms 250` it forgives 125 ms at
    each end of every reference turn, and it forgives it on **both** sides —
    trimming only the reference would charge a hypothesis that runs 100 ms long
    for a boundary the collar was supposed to forgive. The strict number is the
    one that gets quoted; the forgiving number is reported beside it.
* `JER = 1 - |A ∩ B| / |A ∪ B|`, computed separately over the speech region and
  the non-speech region, then averaged. It answers a different question from DER
  — how good is the segmentation of the *conversation*, ignoring who is talking.
* Timing deviation is computed on the **same token alignment** as WER, so a
  substitution is still a timing comparison: the system said the right word at
  the wrong time, and that is a timestamp error, not a word error. `p90` is
  nearest-rank, never interpolated, because interpolating between two measured
  millisecond differences invents a value that was never measured.
  `best_constant_offset_ms` separates a systematic lag (a different bug) from
  jitter (the same bug).

### The taxonomy block

Every row of `docs/evaluation/ERROR_TAXONOMY.md` appears as a signal with an
`id`, a `class`, a `status` and a `count`:

| `status` | meaning |
|---|---|
| `measured` | counted exactly, from a reference and a hypothesis, or from the hypothesis alone when the signature is a claim about the hypothesis |
| `proxy` | counted without a reference and known to be an approximation. The `note` names the blind spot. |
| `unavailable` | not evaluated. `count` is `null`, never `0`, because "none found" and "not looked" are different claims. |

Four signals are `measured` today with no reference at all, because they are
claims about the hypothesis rather than about its correctness:
`asr.duplicated_word`, `asr.punctuation_error`,
`asr.code_switch_boundary_error` and `asr.hallucinated_word_proxy`.
Everything else is `unavailable`, with the reason attached.

# Error taxonomy for Hindi and Hinglish meetings

For regression testing, and for telling the three error classes apart when
something goes wrong.

**A taxonomy whose categories cannot be detected is a list of intentions.** Every
row below therefore has a **detectable signature**: an id, what it means in a
Hindi or Hinglish meeting, how you would identify it automatically from a
hypothesis and a reference, and — where the signature is a proxy or does not yet
exist — an honest statement of that.

The signatures are implemented, not described. Run
`vaani-eval score --set evaluation` and read `taxonomy` in
`results/<id>.report.json`; each entry has `id`, `class`, `status` (`measured`,
`proxy` or `unavailable`), `count`, `unit`, `threshold` and `note`. A `count` of
`null` means **not evaluated**, which is not the same as zero.

The three classes are never combined. See `evaluation/README.md` for why.

---

## How to read the `status` column

| status | meaning |
|---|---|
| `measured` | counted exactly. Either from a reference and a hypothesis, or — for four signatures that are claims about the hypothesis alone — from the hypothesis with no reference at all. |
| `proxy` | counted without a reference, and known to be an approximation. The `note` names the blind spot. **Never report a proxy count as an error rate.** |
| `unavailable` | not evaluated. `count` is `null`. Either the input is missing (no human reference) or the signature has not been built. |

As of this commit, with eight cut clips and **zero** human references:

* 4 signatures are `measured` (all reference-free, all ASR);
* 1 is `proxy`;
* 14 are `unavailable`.

---

## Class 1 — ASR error: the words are wrong

Levenshtein distance on normalised tokens, so the counts below partition the
token error exactly: `S + D + I == errors` and `errors == WER * N_ref`.

### `asr.substituted_word` — a wrong word was produced

| | |
|---|---|
| **Hindi** | a Hindi word decoded as a different Hindi word. Commonest pairs: retroflex/inter-dental confusion (क/ख, ट/त, ड/द), nasalisation dropped or invented (ं), matra errors that change the vowel, and schwa deletion — `कल` for `कर`. |
| **Hinglish** | a Hindi word decoded as an English word, or the reverse. `मैं` → `main`, `है` → `hi`/`hay`, `यह` → `yeah`, `नहीं` → `nain`. This is the single largest error mass in a code-switched meeting and it is not a "Hindi problem" — it is a vocabulary problem, and it is fixable with a language-tagged lexicon. |
| **Signature** | `substitutions` in the `asr` block, and the token-level alignment (`asr.alignment.sub`). |
| **Refinement available** | partition substitutions by script pair (Devanagari→Devanagari, Devanagari→Latin, Latin→Devanagari, Latin→Latin). Not implemented; it is a five-line change to the scorer and it is the split that decides whether the fix is a lexicon or a decoder. |
| **Status** | `unavailable` — no human reference. |

### `asr.hallucinated_word` — a word that was not said

| | |
|---|---|
| **Means** | an insertion: a hypothesis token with no aligned reference token. |
| **Hindi** | the decoder inventing plausible continuations during a pause or a crosstalk burst; a repetition loop (`अच्छा अच्छा अच्छा`); a plausible-sounding completion of a truncated word. |
| **Hinglish** | the same, plus the model "translating" a Hindi utterance into a fluent English sentence, which is a distinctive failure mode of a code-switched model and looks entirely fluent. |
| **Signature** | `insertions` in the `asr` block, plus `asr.hallucinated_word` in the taxonomy. |
| **Reference-free proxy** | `asr.hallucinated_word_proxy` counts contiguous runs of two or more adjacent identical normalised words. A decoding loop is the most visible form of a hallucination and needs no reference to see. |
| **The proxy's blind spot, stated** | a person who genuinely repeats a word produces exactly the same signal. The proxy therefore has a non-zero false-positive rate on any conversational audio, and its count is **not** an insertion count. A second proxy worth building: words whose start lands inside an acoustically silent frame, from `vaani-eval probe`. |
| **Status** | `unavailable`; the proxy is `measured`. |

### `asr.omitted_word` — a word that was said is missing

| | |
|---|---|
| **Means** | a deletion. |
| **Hindi** | schwa deletion collapsing two words into one; a light aspirate lost; a short vowel word skipped during a fast passage. |
| **Hinglish** | an English function word lost inside a Hindi sentence — *a*, *the*, *is*, *to* — which the ASR model's language prior discards. |
| **Signature** | `deletions` in the `asr` block, and `asr.omitted_word`. |
| **Status** | `unavailable` — no human reference. |

### `asr.duplicated_word` — a word or phrase emitted more than once

| | |
|---|---|
| **Means** | a contiguous run of two or more identical normalised words in the hypothesis. |
| **Why it is separate** | it is the only ASR signature measurable with **no reference at all**, because it is a claim about the hypothesis rather than about its correctness. It is also the cheapest high-value regression signal in this list: a change in this count between two builds is a change in decoder behaviour, whatever it did to WER. |
| **Signature** | `asr.duplicated_word` in the taxonomy; `duplicate_runs` and `adjacent_duplicate_words` in `intrinsic`. **Runs**, not pairs, because a four-word loop is one defect and `3` adjacent pairs. |
| **Known false positives** | a real repetition (`बहुत बहुत`), a genuine disfluency, a spelled-out abbreviation (`A P I`), and a reference that was tidied up. The reference must keep its repetitions, or this count becomes meaningless. |
| **Status** | `measured`. |

### `asr.code_switch_boundary_error` — a script boundary in the wrong place

| | |
|---|---|
| **Means** | the point at which the speaker changed language, or the shape of a token at that boundary, was mis-detected. |
| **Signature (hypothesis side)** | a hypothesis token that contains both Devanagari and Latin in one word — `क्याapi`, `करोapi`. Exactly measurable, no reference required. |
| **Signature (reference side)** | a reference token mixing both scripts. Reported separately so the two sides can be compared; a high reference count with a high hypothesis count is a reference-style problem, a high reference count with a zero hypothesis count is a real boundary error. |
| **The deliberate non-fix** | the scorer **does not split** a mixed-script token into its script runs. Splitting would convert the error into a match and hide precisely the thing this signature exists to count. This is the single most tempting wrong turn in the whole harness and it is documented at the call site. |
| **Status** | hypothesis side `measured`; reference side `unavailable`. |

### `asr.punctuation_error` — punctuation is wrong or missing

| | |
|---|---|
| **Means** | punctuation that is present and wrong, or absent. |
| **Signature** | a hypothesis word consisting **entirely** of punctuation characters. The normaliser deletes punctuation from both sides (rule N4), so **WER is structurally blind to punctuation** — this counter is the only thing that sees it. |
| **Why it is a separate category and not part of WER** | SraVaani emits no punctuation at all; its `tokens_to_segments` produces one `Segment` per `Word`. So today this count is 0 and always will be until a punctuation or capitalisation model exists. It is listed because a regression suite for a Hindi meeting transcript needs the row, and a row that reports 0 for a reason nobody wrote down is a row nobody trusts. |
| **Status** | `measured`, currently 0, and the honest reading is "the system does not attempt punctuation". |

### `asr.named_entity_error` — a name, a product or an API is wrong

| | |
|---|---|
| **Means** | a proper noun decoded as a different proper noun, or as a common noun. In a meeting this is the error that actually costs something: a wrong product name in a document is worse than a wrong filler word. |
| **Signature** | would be: aligned token pairs that differ, restricted to tokens that appear in `evaluation/metadata/entities.txt` — a list of the names, products, API names and acronyms that matter for this corpus. |
| **Why it is not implemented** | it needs **two** things that do not exist: a human reference, and the entity list. Guessing at the list from the meeting's subject matter would be inventing the ground truth for this category specifically, which is worse than having no category. |
| **Status** | `unavailable`, with both missing inputs named. |

---

## Class 2 — Diarization error: the words are right and the speaker is wrong

Every signature here is measurable from `reference.rttm` and `hypothesis.rttm`.
**All are `unavailable` today: no human diarization reference exists.**

The three components below are not re-counted in the taxonomy, deliberately.
They are `components.missed_detection`, `components.false_alarm` and
`components.confusion` of the `diarization` block, and duplicating the
arithmetic in a second place would create a second number free to drift away
from the first.

### `diarization.wrong_speaker` — the right words, the wrong person

| | |
|---|---|
| **Signature** | `diarization.components.confusion`, in milliseconds and as a fraction of reference speech. |
| **Not a signature of its own** | confusion and missed detection are not independent defects; they are two views of the same residual. Which one you get depends on whether the hypothesis says *somebody* is talking there. The scorer asserts the two closed forms on every call so the components cannot silently stop summing correctly. |
| **Status** | `unavailable`. |

### `diarization.missed_turn` — speech attributed to nobody

| | |
|---|---|
| **Signature** | `diarization.components.missed_detection`, plus `boundary.unmatched_reference` for the turn-level view. |
| **Status** | `unavailable`. |

### `diarization.false_turn` — speech attributed to somebody who is not talking

| | |
|---|---|
| **Signature** | `diarization.components.false_alarm` and `unattributed_hypothesis_ms`. |
| **Status** | `unavailable`. |

### `diarization.speaker_merge` — two people reported as one

| | |
|---|---|
| **Signature** | the number of distinct reference speakers whose best-matching hypothesis label is the same speaker's, i.e. `reference_speakers − distinct_matched_labels`. Already computed: `speaker_count.hypothesis < speaker_count.reference` is the coarse form, and the taxonomy gives the exact one. |
| **Diagnostic value** | a merge means DER is bounded below by the merged fraction, and no amount of boundary tuning will fix it. It is a **clustering** failure, not a **boundary** failure, and the two have completely different fixes. Telling them apart is most of the value of the taxonomy. |
| **Status** | `unavailable`. |

### `diarization.speaker_split` — one person reported as several

| | |
|---|---|
| **Signature** | `hypothesis_speakers − reference_speakers`, floored at 0. |
| **The operating point this exposes** | JOURNEY.md records that a 14-minute meeting yields 2 speakers, and A4's own documented region is "reliable for turns ≳12 s, documented as failing below ~4 s". A split is the expected failure there, and the count is the number that says whether the heuristic is usable on this corpus at all. |
| **Status** | `unavailable`. |

### `diarization.overlap_failure` — simultaneous speech collapsed to one voice

| | |
|---|---|
| **Signature** | `reference_overlap_ms`: the milliseconds in which **two distinct reference speakers** were active at once. Swept from the reference boundaries, so it is a property of the reference and can be reported **before any hypothesis exists**. |
| **Why it is the most important row in this class** | VaaniDesk's diarizer emits at most one speaker at an instant. If the reference contains *N* ms of overlap, then **DER ≥ N / reference_speech, for every system, including a perfect one.** An evaluation that omits overlap from its reference reports a flattering number and does not know it. The annotator protocol therefore makes recording overlap mandatory, and this row is what makes the cost of not doing it visible. |
| **Note on the implementation** | it counts distinct *speakers*, not distinct turns: one speaker pausing and resuming is not overlap. |
| **Status** | `unavailable`. |

---

## Class 3 — Timestamp error: the words and the speaker are right and the time is wrong

Measured on the **same token alignment** as WER, so a substitution is still a
timing comparison: the system said the right word at the wrong time, and that
is a timestamp error, not a word error. Needs `reference.words.json`.

**All are `unavailable` today: no human word-timing reference exists.**

Thresholds, all in the `threshold` field of each signal, all configurable:

| threshold | value | why that value |
|---|---|---|
| `drift_ms` | 200 | below this, a word is within a fifth of a second of where it belongs, which no consumer of a transcript can act on |
| `shifted_ms` | 500 | above this, the word is attached to the wrong moment in the conversation, which a human reading a transcript can act on |
| `long_ratio` | 1.5 | a word held 50 % longer than the reference is being smeared across its neighbour |
| `short_ratio` | 0.667 | a word held under two thirds as long is being clipped |

### `timing.segment_shifted` — the whole span is in the wrong place

| | |
|---|---|
| **Signature** | aligned pairs with `abs(hyp_start − ref_start) > shifted_ms`, counted over `start_deviation.max_abs_ms` for the magnitude. |
| **Distinguishing signal** | a **large** `best_constant_offset_ms` with a **small** spread. A systematic lag is a rebasing bug; jitter is a different bug. The scorer reports the median and the best constant offset precisely so these two do not have to be told apart by eye. |
| **Status** | `unavailable`. |

### `timing.word_timestamp_drift` — the span is right, the edges are soft

| | |
|---|---|
| **Signature** | aligned pairs with `abs(hyp_start − ref_start) > drift_ms`, with `start_deviation.p90_ms` for the magnitude. |
| **Why p90 and not the mean** | nearest-rank, never interpolated, because interpolating between two measured millisecond differences invents a value that was never measured. |
| **A known and unfixable-without-work cause** | JOURNEY.md records that 1,889 of 1,911 inter-word gaps are **exactly zero**, because the ASR adapter gives each word the frame immediately after the previous one. A decoder that reports no inter-word silence **cannot** place word boundaries better than its own frame resolution, and this signature is the number that says so. |
| **Status** | `unavailable`. |

### `timing.segment_too_long` — a word smeared across its neighbour

| | |
|---|---|
| **Signature** | aligned pairs with `hyp_duration > long_ratio * ref_duration`. |
| **Status** | `unavailable`. |

### `timing.segment_too_short` — a word clipped

| | |
|---|---|
| **Signature** | aligned pairs with `hyp_duration < short_ratio * ref_duration`. |
| **Companion integrity check** | `timing.negative_duration_pairs` counts aligned pairs whose hypothesis span is non-positive, and `intrinsic.negative_duration_words` counts the same over the whole hypothesis. A non-zero value is a structural defect rather than an accuracy result, and `JOURNEY.md` already records that word timings are monotone and in bounds — so these two counters are the regression gate for a property that currently holds. |
| **Status** | `unavailable`. |

---

## Summary table

| id | class | needs a human reference? | status today |
|---|---|---|---|
| `asr.duplicated_word` | asr | no — hypothesis only | **measured** |
| `asr.punctuation_error` | asr | no — hypothesis only | **measured** |
| `asr.code_switch_boundary_error` | asr | no — hypothesis only | **measured** |
| `asr.hallucinated_word_proxy` | asr | no — hypothesis only, approximate | **proxy** |
| `asr.substituted_word` | asr | yes | unavailable |
| `asr.hallucinated_word` | asr | yes | unavailable |
| `asr.omitted_word` | asr | yes | unavailable |
| `asr.mixed_script_reference_token` | asr | yes | unavailable |
| `asr.named_entity_error` | asr | yes **and** an entity list | unavailable |
| `diarization.wrong_speaker` | diarization | yes | unavailable |
| `diarization.missed_turn` | diarization | yes | unavailable |
| `diarization.false_turn` | diarization | yes | unavailable |
| `diarization.speaker_merge` | diarization | yes | unavailable |
| `diarization.speaker_split` | diarization | yes | unavailable |
| `diarization.overlap_failure` | diarization | yes (reference only) | unavailable |
| `timing.segment_shifted` | timing | yes + word timings | unavailable |
| `timing.word_timestamp_drift` | timing | yes + word timings | unavailable |
| `timing.segment_too_long` | timing | yes + word timings | unavailable |
| `timing.segment_too_short` | timing | yes + word timings | unavailable |

Every category in the brief is present. Three of them are **not measurable
today**, and each says which input is missing rather than reporting a number it
cannot support. That is the whole point of the table.

---

## Using this for regression testing

1. **Annotate.** Without a reference, this document is a specification and the
   harness is a self-consistency checker. Follow `evaluation/ANNOTATION.md`.
2. **Freeze the reference.** `set.json` carries a SHA-256 for the source and for
   each clip. Record the reference hashes in your release notes, or a changed
   reference will look exactly like a changed system.
3. **Fix the profile.** Quote the `normalisation.profile` and the parameter ids
   from every report. A WER without its normalisation is not a number.
4. **Read the three blocks separately.** They are three defects with three
   fixes, and the whole point of keeping them apart is that a blended score
   cannot tell you which one to work on.
5. **Watch the reference-free signals first.** `asr.duplicated_word`,
   `asr.punctuation_error`, `intrinsic.non_monotonic_word_pairs`,
   `intrinsic.negative_duration_words` and `intrinsic.unattributed_segments` are
   available on every build, today, with no annotation at all. They will catch a
   decoder regression before a single word has been transcribed by hand.
6. **Check the closed forms.** The scorer asserts
   `correct + missed + confusion == reference speech` and
   `correct + false_alarm + unattributed == hypothesis speech` in a
   `debug_assert` on every call, and two of the known-answer tests assert the
   same identities from the outside. If a future change breaks the
   decomposition, a test fails rather than a published number moving.

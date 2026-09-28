# How to annotate an evaluation clip

This is written for the person doing the listening, not for an engineer. It
assumes you speak Hindi and English and that you have never seen a NIST file.

**Everything you write stays on this machine.** `evaluation/references/` is
gitignored: your transcript of a private meeting is not a thing this repository
is allowed to hold. Read that again before you start — the whole reason the
clip cannot be scored right now is that this file does not exist yet, and the
whole reason it cannot be *committed* even once you write it is that it is
somebody's private conversation.

---

## Before you start: what you need

| | |
|---|---|
| **Audio** | `evaluation/clips/<id>.wav` — 16 kHz mono WAV, 90 seconds. Plays in anything. |
| **A way to scrub** | Any player with a millisecond readout is ideal (VLC's status bar, Audacity, ffplay). You need to place boundaries by ear, not by eye. |
| **A text editor** | UTF-8. Turn on "show invisibles" so you can see the spaces, because whitespace is the one thing that will silently corrupt a reference. |
| **This file** | open beside the player. |
| **Roughly ten minutes per clip** | for `reference.txt`. The RTTM is slower: 30–40 minutes for 90 seconds of audio, because you replay it repeatedly. |

**A second listener is worth ten times its cost** for the RTTM. Speaker
boundaries are where human annotation is most inconsistent, and a boundary
reference is the difference between measuring a diarizer and measuring an
annotator. If you get a second opinion, record it in `ANNOTATED.md` under
`second_listener` and say whether the two agreed. Where they disagreed, use the
one you believe and note the disagreement — a known 200 ms boundary
disagreement is a fact about the reference; a silently averaged one is a lie.

**Listen at normal speed first, all the way through, before writing anything.**
You will be wrong about the first ten seconds otherwise. Then write.

---

## Step 1 — `reference.txt`: the words

Copy `evaluation/templates/reference.txt` to `evaluation/references/<id>/`.

One utterance per line. Only the words. No timestamps, no speaker labels, no
punctuation, no blank lines.

### The rules that actually change your score

**Write the script the speaker used.** A Hindi speaker who says "API integrate
करना है" gets `API integrate करना है`, not `एपीआई इंटीग्रेट करना है`. The harness
deliberately does not transliterate, and a code-switched meeting is the case
that rule exists for.

**Write the words as they were pronounced, not as they are spelled.** If
somebody says the product name the way the marketing spells it, write it that
way. If they say it wrong, write it the way they said it — a recogniser that
transcribes the correct spelling has made an error, and hiding that in the
reference makes the system look better than it is.

**Keep the fillers.** `अरे`, `तो`, `हाँ`, `uh`, `basically`, `I mean` are
words. A reference with the fillers removed measures a different system than
the one that ships.

**Keep the repetitions.** If somebody says `बहुत बहुत अच्छा`, that is three
words. One of the taxonomy's error signatures is a decoder repeating itself,
and it is undetectable if the reference has been tidied.

**Numbers: write what was said.** If somebody says "पाँच", write `पाँच`. If
they say "five", write `five`. The harness does not convert between them, and
`पाँच` against `5` correctly scores as a substitution plus a deletion.

**Inaudible: write `[inaudible]`.** Once per word you cannot recover. Do not
guess a word from context — a guessed word that happens to be right inflates the
score, and one that happens to be wrong penalises the system for your hearing.
Do not leave a gap either: a blank where a word was is indistinguishable from a
deleted word and becomes a false WER penalty.

**Overlap: transcribe what you can hear.** One line per speaker, in time order.
Then mark the overlap in `ANNOTATED.md` and, if you can, encode it in the RTTM
(Step 2). Do not quietly pick one voice and pretend the other was not there —
that produces a diarization reference that rewards the system for failing to
notice the overlap.

**Non-speech sounds.** Ignore them. A cough is not a word. If somebody says
something unintelligible *and* coughs, that is `[inaudible]`, not two tokens.

### Then normalise it yourself, once

```
vaani-eval norm --in references\<id>\reference.txt --out references\<id>\reference.normalised.txt
```

Read the normalised file. It is the text that will actually be scored, so if
the normaliser did something surprising — a character it treated as
punctuation, a nukta it rewrote — you want to find out now, while you still
remember what the audio sounded like. The normaliser is
`evaluation/scoring/src/normalize.rs`; the rules are in `evaluation/SCHEMA.md`.

Keep both files. The normalised one is only for inspection; the scorer
normalises the original itself, and the point of doing this is to *see* what
that produces.

---

## Step 2 — `reference.rttm`: who is speaking

Copy `evaluation/templates/reference.rttm` to `evaluation/references/<id>/`.

One line per contiguous stretch of one person talking, in decimal seconds:

```
SPEAKER clip01 1 0.000000 8.400000 <NA> <NA> A <NA> <NA>
SPEAKER clip01 1 8.520000 11.900000 <NA> <NA> B <NA> <NA>
```

### The rules that actually change your score

**Use your own labels.** `A`, `B`, `SPK1`, whatever. The harness resolves a
permutation before comparing, so labels are never compared as strings. Use
whatever is fastest for you and be consistent within the clip.

**Merge adjacent stretches of the same person.** Two consecutive `A` records
with a 20 ms gap is a speaker change the human did not hear, and the harness
will count it as a missed speaker change — against the reference, which is the
wrong way round.

**Cover the whole clip.** A record that stops at a pause is a false alarm
waiting to happen: the hypothesis will keep talking through the pause and the
harness will charge it for the extra seconds.

**Record overlap as two records with identical start and end.**

```
SPEAKER clip01 1 47.000000 3.100000 <NA> <NA> A <NA> <NA>
SPEAKER clip01 1 47.000000 3.100000 <NA> <NA> B <NA> <NA>
```

This is the single most consequential thing in this file. VaaniDesk's
diarizer emits at most one speaker at a time, so **every millisecond of overlap
you record is a millisecond of DER it cannot avoid.** If you omit the overlap,
the reference claims the meeting never interrupted itself and the score is
flattering for the wrong reason. If you record it, the score is right and
unflattering, and the report tells you the bound.

**Boundaries: within about 100 ms.** Boundary annotation is the least precise
thing a human does with a stopwatch-free ear. The harness matches boundaries
inside a 250 ms tolerance and reports the mean and the maximum deviation, so
your imprecision shows up as a number rather than as a mystery. Say how
imprecise you were in `ANNOTATED.md`.

**Non-speech regions: leave them out.** A `NOSPEAKER` record is accepted and
parsed but is not scored as a turn. Omit gaps entirely; do not write them as
`SPEAKER` and do not bother with `NOSPEAKER`.

---

## Step 3 — `reference.words.json`: word timings (optional)

Only if you want the timestamp block to be measurable. It roughly triples the
time, and it is the most valuable and most error-prone part of the reference.

Copy `evaluation/templates/reference.words.json`.

* `text` must be the **normalised** token, copied from the `vaani-eval norm`
  output. Not the surface form, not your original. The alignment pairs tokens,
  and a mismatch here turns every deviation figure into noise.
* One entry per token, in the same order, matching `reference.txt` exactly.
* `start_ms` and `end_ms` are when **that person is producing that word** — not
  the gap after it, and not the span of the phrase. Round to the nearest 10 ms
  and put your actual precision in `ANNOTATED.md`; the harness will report a
  10 ms median for a perfect system if you timed to 20 ms, and you should be
  able to explain why.
* When in doubt, time a word's **centre** and set `end_ms = start_ms + 200`. A
  consistent bias is reported as `best_constant_offset_ms` and can be subtracted;
  random boundaries cannot.

---

## Step 4 — `ANNOTATED.md`: say who you are

Copy `evaluation/templates/ANNOTATED.md` and fill it in.

This file is why your work counts. The harness refuses to score a reference
without it, because a file that claims to be a human transcript and has no
evidence of being one is indistinguishable — to the harness and to whoever
reads the report in six months — from a transcript somebody made up. The
metadata is the claim; this file is the evidence.

Fill in:

* `annotator`, `date` — who and when;
* `listens` — how many times you played it before writing. One listen and a
  reference is a first draft, and saying so is worth more than a claim of
  certainty;
* `second_listener` and `agreement` — if you had one;
* `speakers_confirmed`, `language_mix_confirmed` — these are the values that
  currently say `unverified` in `metadata/<id>.meta.json`, and confirming them
  is the cheapest part of the whole job;
* `overlap_annotated` — yes or no. If no, the DER is not comparable to one
  where it is;
* `timing_precision_ms` if you did Step 3;
* `notes` — anything the next person needs. Where the audio is bad, where you
  are unsure, where somebody was off-mic, where a word is a coin flip. **A
  reference that flags its own uncertainty is worth more than one that hides
  it**, because the harness can discount a flagged word and cannot discount an
  unflagged wrong one.

---

## Step 5 — score it

```
vaani-eval score --set evaluation
```

Your clip moves from `unscored` to `scored` and the numbers appear. If it
does not, the `reason` field in `results/<id>.report.json` says which of the
three files is missing or unreadable. The most common cause is a missing
`ANNOTATED.md`.

Look at `taxonomy` as well as the rates. `asr.duplicated_word` and
`asr.punctuation_error` are counted with no reference at all; the rest become
exact counts the moment your `reference.txt` exists.

---

## What to do when you disagree with the system

If the transcript is wrong and your reference says otherwise, the system is
wrong. Write the reference the way it was said.

The one case where you should hesitate is if the recogniser heard a word you
did not, on a first listen. Play it again. Both of you are fallible; the
reference is the one with a human in the loop, and the harness is designed so
that its mistakes are visible — an unusually high WER concentrated on one
proper noun, or a whole clip much worse than the rest, is a signal to go back
and re-listen before the number gets published.

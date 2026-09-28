---
clip: <clip id, e.g. clip01>
annotator: <name or handle>
date: <YYYY-MM-DD>

# How many times you played the clip all the way through before writing.
# One listen and a reference is a first draft, and saying so is worth more
# than a claim of certainty.
listens: 1

# A second listener on the RTTM is worth ten times its cost: speaker
# boundaries are where human annotation is most inconsistent. Record the
# disagreement rather than averaging it away.
second_listener: <name, or "none">
agreement: <full | partial | n/a>

# These are the values metadata/<id>.meta.json currently reports as
# unverified. Confirming them is the cheapest part of the whole job.
speakers_confirmed: <int>
speaker_labels_used: <A,B / SPK1,SPK2 / ...>
language_mix_confirmed: [<hinglish, hindi, english, ...>]

# Did you record overlapping speech in the RTTM? If no, the DER for this clip
# is not comparable to a clip where you did, because every millisecond of
# overlap you omit is a millisecond of error the system is never charged for.
overlap_annotated: <yes | no | no overlap present>

# Only if you wrote reference.words.json.
word_timings: <yes | no>
timing_precision_ms: <10>

# Anything the next person needs to know: where the audio is bad, where you
# are unsure, where somebody was off-mic, where a word is a coin flip.
# A reference that flags its own uncertainty is worth more than one that
# hides it.
notes: >
  <free text>
---

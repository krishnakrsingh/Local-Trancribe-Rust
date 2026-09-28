//! Reading the system's own output as a hypothesis.
//!
//! The input is the JSON that `vaani transcribe --format json` writes, i.e.
//! the `vaani.transcript` envelope from `vaani-export`. It is read as a
//! **hypothesis** and never as ground truth. The distinction is enforced
//! structurally: this module has no path that can produce a
//! [`crate::normalize::NormalisedText`] it is willing to score as a reference,
//! and [`Hypothesis::is_reference_grade`] returns `false` for every transcript
//! that came out of a model, whatever produced it.
//!
//! Two artefacts come out of a hypothesis:
//!
//! * the **word list**, for WER/CER and for timestamp deviation;
//! * an **RTTM** view, for DER. The RTTM is built from the shipped segments
//!   and their attributed speakers — that is, from the attribution a user
//!   actually reads, not from the diarizer's internal spans. It is coarser than
//!   the raw spans, and the report says so, because the grouping policy in
//!   `vaani_segment` runs before attribution and a DER measured after it
//!   includes the grouping's effect.

use serde::{Deserialize, Serialize};

use crate::rttm::{self, MICROS, RttmRecord};

/// One recognised word with its time span, in milliseconds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HypothesisWord {
    /// Surface form as the engine produced it, before normalisation.
    pub text: String,
    /// Start, milliseconds.
    pub start_ms: i64,
    /// End, milliseconds.
    pub end_ms: i64,
}

/// One segment with its attributed speaker, if any.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HypothesisSegment {
    /// Segment start, milliseconds.
    pub start_ms: i64,
    /// Segment end, milliseconds.
    pub end_ms: i64,
    /// Speaker label from the transcript, or `None` when diarization did not
    /// run (ASR-only mode).
    pub speaker: Option<String>,
    /// Indices into [`Hypothesis::words`] for this segment.
    pub word_range: (usize, usize),
}

/// A reference word with its time span, for the timing pass. The reference RTTM
/// carries turns, not words, so word timings come from a sidecar
/// `reference.words.json` when the annotator supplies one; without it only
/// segment-level timing is measurable, and the report says which it did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceWord {
    /// Reference token, already normalised by the caller.
    pub text: String,
    /// Start, milliseconds.
    pub start_ms: i64,
    /// End, milliseconds.
    pub end_ms: i64,
}

/// The system's output for one clip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hypothesis {
    /// Which artefact this came from, for the report.
    pub source: String,
    /// Every word, in transcript order.
    pub words: Vec<HypothesisWord>,
    /// Every segment, in transcript order.
    pub segments: Vec<HypothesisSegment>,
    /// Media duration, milliseconds, when the transcript reported it.
    pub duration_ms: Option<i64>,
    /// Distinct speaker labels, in first-appearance order.
    pub speakers: Vec<String>,
}

impl Hypothesis {
    /// Always `false`. There is no such thing as a model output that is a
    /// ground truth, and this exists so a future code path that tries to score
    /// one has to delete a function that says no.
    pub fn is_reference_grade(&self) -> bool {
        false
    }

    /// Flatten to whitespace-joined surface text, one word per space, in
    /// order. This is the file the normaliser reads. It is **not** a
    /// human-readable transcript and is never committed.
    pub fn to_text(&self) -> String {
        self.words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// The RTTM view of this hypothesis: one record per attributed segment.
    ///
    /// Segments with no speaker are dropped, because in ASR-only mode there is
    /// no diarization hypothesis to score and a `SPEAKER` record with an empty
    /// label would be scored as a false alarm against nothing.
    pub fn to_rttm(&self, file_id: &str) -> Vec<RttmRecord> {
        self.segments
            .iter()
            .filter_map(|s| {
                let speaker = s.speaker.clone()?;
                let start_us = s.start_ms * 1000;
                let duration_us = (s.end_ms - s.start_ms).max(0) * 1000;
                if duration_us <= 0 {
                    return None;
                }
                let mut r = RttmRecord::speaker(&speaker, start_us, duration_us);
                r.file_id = file_id.to_string();
                Some(r)
            })
            .collect()
    }

    /// Reference-free integrity facts. These are **not** accuracy: they say
    /// the output is well formed, not that it is right, and the report labels
    /// them `intrinsic` so nobody can mistake them for a score.
    pub fn intrinsic(&self) -> Intrinsic {
        let mut non_monotonic = 0usize;
        let mut negative_duration = 0usize;
        for w in &self.words {
            if w.end_ms < w.start_ms {
                negative_duration += 1;
            }
        }
        for w in self.words.windows(2) {
            if w[1].start_ms < w[0].start_ms {
                non_monotonic += 1;
            }
        }
        let mut adjacent_duplicates = 0usize;
        let mut repeated_runs = 0usize;
        let mut in_run = false;
        for pair in self.words.windows(2) {
            let same = crate::normalize::normalise(
                &pair[0].text,
                crate::normalize::NormalisationProfile::Strict,
            )
            .tokens
            .first()
            .cloned()
                == crate::normalize::normalise(
                    &pair[1].text,
                    crate::normalize::NormalisationProfile::Strict,
                )
                .tokens
                .first()
                .cloned();
            if same {
                adjacent_duplicates += 1;
                if !in_run {
                    repeated_runs += 1;
                    in_run = true;
                }
            } else {
                in_run = false;
            }
        }
        let unattributed = self.segments.iter().filter(|s| s.speaker.is_none()).count();
        Intrinsic {
            word_count: self.words.len(),
            segment_count: self.segments.len(),
            speaker_count: self.speakers.len(),
            unattributed_segments: unattributed,
            non_monotonic_word_pairs: non_monotonic,
            negative_duration_words: negative_duration,
            adjacent_duplicate_words: adjacent_duplicates,
            duplicate_runs: repeated_runs,
            duration_ms: self.duration_ms,
        }
    }
}

/// Facts about a hypothesis that need no reference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Intrinsic {
    /// Words in the transcript.
    pub word_count: usize,
    /// Segments in the transcript.
    pub segment_count: usize,
    /// Distinct speakers attributed.
    pub speaker_count: usize,
    /// Segments with no speaker, i.e. ASR-only mode or a failed diarization.
    pub unattributed_segments: usize,
    /// Word pairs whose start time went backwards. Must be `0`; a non-zero
    /// value is a structural defect, not an accuracy result.
    pub non_monotonic_word_pairs: usize,
    /// Words whose end precedes their start.
    pub negative_duration_words: usize,
    /// Adjacent word pairs with identical normalised text. A duplication
    /// signature; counted, not judged.
    pub adjacent_duplicate_words: usize,
    /// Contiguous runs of the above, i.e. actual repeated spans.
    pub duplicate_runs: usize,
    /// Reported media duration, milliseconds.
    pub duration_ms: Option<i64>,
}

/// Why a transcript export could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HypothesisError {
    /// The file is not valid JSON.
    Json(String),
    /// The JSON is valid but is not a `vaani.transcript` envelope.
    NotATranscript(String),
    /// A field is present with the wrong type.
    Field(String),
}

impl std::fmt::Display for HypothesisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(m) => write!(f, "not valid JSON: {m}"),
            Self::NotATranscript(m) => write!(f, "not a vaani.transcript export: {m}"),
            Self::Field(m) => write!(f, "unexpected field: {m}"),
        }
    }
}

impl std::error::Error for HypothesisError {}

/// Read a `vaani.transcript` JSON export into a [`Hypothesis`].
///
/// Deliberately hand-walked through `serde_json::Value` rather than derived
/// from a copy of `vaani_core::Transcript`: the harness must not depend on
/// the domain type, because a change to that type must not silently change what
/// the evaluation scores. The two field names it reads are pinned by the tests
/// in this module, which is the real contract.
pub fn from_export_json(text: &str) -> Result<Hypothesis, HypothesisError> {
    let v: serde_json::Value =
        serde_json::from_str(text).map_err(|e| HypothesisError::Json(e.to_string()))?;
    if v.get("format").and_then(|f| f.as_str()) != Some("vaani.transcript") {
        return Err(HypothesisError::NotATranscript(format!(
            "format = {:?}, expected \"vaani.transcript\"",
            v.get("format")
        )));
    }
    let tr = v
        .get("transcript")
        .ok_or_else(|| HypothesisError::NotATranscript("no `transcript` key".into()))?;

    let speaker_names: Vec<(u64, String)> = tr
        .get("speakers")
        .and_then(|s| s.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|sp| {
                    let i = sp.get("index")?.as_u64()?;
                    let name = sp
                        .get("name")
                        .and_then(|n| n.as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| format!("spk{i}"));
                    Some((i, name))
                })
                .collect()
        })
        .unwrap_or_default();
    let name_of = |idx: u64| -> String {
        speaker_names
            .iter()
            .find(|(i, _)| *i == idx)
            .map(|(_, n)| n.clone())
            .unwrap_or_else(|| format!("spk{idx}"))
    };

    let mut words = Vec::new();
    let mut segments = Vec::new();
    let mut order: Vec<String> = Vec::new();

    for seg in tr
        .get("segments")
        .and_then(|s| s.as_array())
        .ok_or_else(|| HypothesisError::Field("transcript.segments is not an array".into()))?
    {
        let start_ms = seg
            .get("start")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| HypothesisError::Field("segment.start is not an integer".into()))?;
        let end_ms = seg
            .get("end")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| HypothesisError::Field("segment.end is not an integer".into()))?;
        let speaker = seg.get("speaker").and_then(|v| v.as_u64()).map(name_of);
        if let Some(s) = &speaker
            && !order.contains(s)
        {
            order.push(s.clone());
        }
        let first = words.len();
        for w in seg
            .get("words")
            .and_then(|w| w.as_array())
            .ok_or_else(|| HypothesisError::Field("segment.words is not an array".into()))?
        {
            let text = w
                .get("text")
                .and_then(|v| v.as_str())
                .ok_or_else(|| HypothesisError::Field("word.text is not a string".into()))?;
            let ws = w
                .get("start")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| HypothesisError::Field("word.start is not an integer".into()))?;
            let we = w
                .get("end")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| HypothesisError::Field("word.end is not an integer".into()))?;
            words.push(HypothesisWord {
                text: text.to_string(),
                start_ms: ws,
                end_ms: we,
            });
        }
        segments.push(HypothesisSegment {
            start_ms,
            end_ms,
            speaker,
            word_range: (first, words.len()),
        });
    }

    Ok(Hypothesis {
        source: v
            .get("source")
            .and_then(|s| s.get("file"))
            .and_then(|f| f.as_str())
            .unwrap_or("unknown")
            .to_string(),
        words,
        segments,
        duration_ms: tr.get("duration").and_then(|d| d.as_i64()),
        speakers: order,
    })
}

/// Build an RTTM document for a hypothesis, ready to hand to
/// [`crate::der::score`].
pub fn hypothesis_rttm(h: &Hypothesis, file_id: &str) -> String {
    rttm::write(&h.to_rttm(file_id))
}

/// Microseconds per millisecond, re-exported so callers converting a
/// millisecond hypothesis span into an RTTM record do not write `* 1000`
/// themselves and get it subtly wrong.
pub const US_PER_MS: i64 = MICROS / 1000;

#[cfg(test)]
mod tests {
    use super::*;

    const EXPORT: &str = r#"{
      "format": "vaani.transcript",
      "format_version": 1,
      "generator": "vaani-export 0.1.0",
      "source": { "file": "clip01", "duration_ms": 90000, "speakers": 2 },
      "transcript": {
        "duration": 90000,
        "segments": [
          { "id": "a", "start": 0, "end": 2000, "speaker": 0,
            "words": [ {"text": "नमस्ते", "start": 0, "end": 500},
                      {"text": "dosto", "start": 500, "end": 900} ] },
          { "id": "b", "start": 2000, "end": 4000, "speaker": 1,
            "words": [ {"text": "API", "start": 2000, "end": 2400} ] }
        ],
        "speakers": [ {"index": 0, "name": "A"}, {"index": 1, "name": "B"} ]
      }
    }"#;

    #[test]
    fn it_reads_words_segments_and_speakers() {
        let h = from_export_json(EXPORT).unwrap();
        assert_eq!(h.words.len(), 3);
        assert_eq!(h.words[0].text, "नमस्ते");
        assert_eq!(h.words[0].start_ms, 0);
        assert_eq!(h.segments.len(), 2);
        assert_eq!(h.segments[1].speaker.as_deref(), Some("B"));
        assert_eq!(h.speakers, vec!["A", "B"]);
        assert_eq!(h.duration_ms, Some(90_000));
        assert_eq!(h.segments[0].word_range, (0, 2));
        assert_eq!(h.segments[1].word_range, (2, 3));
    }

    #[test]
    fn a_model_output_is_never_reference_grade() {
        let h = from_export_json(EXPORT).unwrap();
        assert!(!h.is_reference_grade());
    }

    #[test]
    fn it_rejects_something_that_is_not_a_transcript_envelope() {
        let e = from_export_json(r#"{"format":"something.else"}"#).unwrap_err();
        assert!(matches!(e, HypothesisError::NotATranscript(_)));
        let e = from_export_json("not json").unwrap_err();
        assert!(matches!(e, HypothesisError::Json(_)));
    }

    #[test]
    fn a_missing_required_field_names_itself() {
        let bad = EXPORT.replace("\"start\": 0, \"end\": 2000", "\"end\": 2000");
        let e = from_export_json(&bad).unwrap_err();
        assert!(e.to_string().contains("segment.start"), "{e}");
    }

    #[test]
    fn asr_only_output_has_no_speaker_and_no_rttm_record() {
        let asr_only = EXPORT.replace("\"speaker\": 0,", "\"speaker\": null,");
        let h = from_export_json(&asr_only).unwrap();
        assert_eq!(h.segments[0].speaker, None);
        assert_eq!(h.to_rttm("clip01").len(), 1, "only the attributed segment");
        assert_eq!(h.intrinsic().unattributed_segments, 1);
    }

    #[test]
    fn the_rttm_view_uses_milliseconds_converted_exactly() {
        let h = from_export_json(EXPORT).unwrap();
        let doc = hypothesis_rttm(&h, "clip01");
        assert!(doc.contains("SPEAKER clip01 1 0.000000 2.000000"), "{doc}");
        assert!(doc.contains("SPEAKER clip01 1 2.000000 2.000000"), "{doc}");
        assert_eq!(US_PER_MS, 1000);
    }

    #[test]
    fn intrinsic_counts_duplicates_and_never_mislabels_them_as_accuracy() {
        let dup = EXPORT.to_string();
        let mut h = from_export_json(&dup).unwrap();
        h.words[2].text = "dosto".to_string();
        let i = h.intrinsic();
        assert_eq!(i.adjacent_duplicate_words, 1);
        assert_eq!(i.duplicate_runs, 1);
        assert_eq!(i.non_monotonic_word_pairs, 0);
    }

    #[test]
    fn a_backwards_word_start_is_counted_not_ignored() {
        let mut h = from_export_json(EXPORT).unwrap();
        h.words[2].start_ms = 10;
        assert_eq!(h.intrinsic().non_monotonic_word_pairs, 1);
    }
}

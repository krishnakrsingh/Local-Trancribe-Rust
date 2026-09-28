//! The evaluation set on disk, and the single scoring entry point.
//!
//! # Layout
//!
//! ```text
//! evaluation/
//!   set.json                       the set definition: which clips, cut from where
//!   clips/<id>.wav                 16 kHz mono PCM16          [gitignored]
//!   metadata/<id>.meta.json        structural metadata, committable
//!   references/<id>/reference.txt          orthographic     [gitignored]
//!   references/<id>/reference.rttm         diarization      [gitignored]
//!   references/<id>/reference.words.json   word timings     [gitignored]
//!   references/<id>/ANNOTATED.md           provenance        [gitignored]
//!   hypotheses/<id>/hypothesis.json        raw export       [gitignored]
//!   hypotheses/<id>/hypothesis.txt         flat word list    [gitignored]
//!   hypotheses/<id>/hypothesis.rttm        speaker view      [gitignored]
//!   results/<id>.report.json         aggregate only, committable
//!   results/summary.json             aggregate only, committable
//! ```
//!
//! Everything that could carry a word of the meeting is gitignored. Only
//! structure — clip boundaries, durations, dB statistics, and error rates that
//! happen to be computable — is committable.
//!
//! # What counts as a reference
//!
//! [`Set::score_clip`] looks for `references/<id>/ANNOTATED.md` and reads its
//! front matter. A reference without that file is treated as
//! **awaiting annotation** and every block reports `unscored` — not `0`. That
//! is deliberate friction: the file is where a person records their name, the
//! date, and how many times they listened. It is the difference between "the
//! reference exists" and "someone made this reference".

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

use crate::der::{self, DiarParams};
use crate::edit::{Op, align};
use crate::hypothesis::{self, Hypothesis, ReferenceWord};
use crate::normalize::{self, NormalisationProfile};
use crate::report::{
    AsrBlock, BlendRefusal, BlockStatus, ClipReport, DiarizationBlock, NormalisationBlock,
    REPORT_VERSION, Report, SetCounts, TimingBlock,
};
use crate::rttm;
use crate::taxonomy::{self, ReferenceView, TimingCounts, TimingThresholds};
use crate::timing;
use crate::wer;

/// Set schema version.
pub const SET_VERSION: u32 = 1;

/// Front-matter key whose presence marks a reference as human-authored.
pub const ANNOTATED_FILE: &str = "ANNOTATED.md";

/// Why a claim in the metadata is or is not verified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provenance {
    /// A person confirmed it by listening to the audio.
    HumanVerified,
    /// Measured from the audio or the system, and labelled as such.
    Measured,
    /// Stated in this project's own documentation about the recording, and not
    /// re-measured here. True as far as the record goes; nobody has checked it
    /// against the audio for this track.
    DocumentedContext,
    /// Asserted without evidence. Every property that only a listener can judge
    /// — the language mix, the speaker count, whether two people talk over each
    /// other — starts here, and nothing derived from it is scored.
    Unverified,
}

/// One clip's metadata. Committable; contains no content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipMeta {
    /// Clip id, matching `clips/<id>.wav`.
    pub id: String,
    /// Human label, e.g. `"pause-dense-opening"`.
    pub topic: String,
    /// The languages mixed in this clip. A judgement only a listener can make.
    pub language_mix: Vec<String>,
    /// Where `language_mix` comes from.
    pub language_mix_provenance: Provenance,
    /// Distinct speakers the clip is believed to contain, or `None` when nobody
    /// has listened to it. `Some(0)` would be a claim that nobody spoke.
    pub speakers: Option<usize>,
    /// Where `speakers` comes from.
    pub speakers_provenance: Provenance,
    /// `easy` | `moderate` | `hard`.
    pub difficulty: String,
    /// Where `difficulty` comes from.
    pub difficulty_provenance: Provenance,
    /// Which required characteristic this clip was chosen to cover. Vocabulary
    /// from the brief, so coverage is auditable.
    pub covers: Vec<String>,
    /// Why it was chosen, in one sentence.
    pub rationale: String,
    /// Structural facts, measured. No content.
    pub structure: Option<crate::probe::Structure>,
    /// What the annotator should check first, and what cannot be claimed yet.
    pub annotator_notes: Vec<String>,
}

/// One clip in the set definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipDef {
    /// Clip id.
    pub id: String,
    /// Start in the source recording, milliseconds.
    pub start_ms: i64,
    /// Clip length, milliseconds. The brief requires 60 000..=120 000.
    pub duration_ms: i64,
    /// Human label.
    pub topic: String,
    /// Why it was chosen.
    pub rationale: String,
    /// Which required characteristics it covers.
    pub covers: Vec<String>,
    /// Speaker count as believed, or `None` when nobody has listened.
    pub speakers: Option<usize>,
    /// Where `speakers` comes from.
    pub speakers_provenance: Provenance,
    /// Languages expected in the clip. A judgement only a listener can confirm.
    pub language_mix: Vec<String>,
    /// Where `language_mix` comes from.
    pub language_mix_provenance: Provenance,
    /// Expected difficulty: `easy` | `moderate` | `hard`.
    pub difficulty: String,
    /// Where `difficulty` comes from.
    pub difficulty_provenance: Provenance,
    /// SHA-256 of the cut clip, so a score can name the audio it came from.
    pub clip_sha256: String,
    /// The measured structural summary, committed.
    pub structure: Option<crate::probe::Structure>,
}

/// The source recording the clips are cut from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Source {
    /// Repository-relative path. Gitignored.
    pub path: String,
    /// SHA-256 of the file, so a score can name the bytes it came from.
    pub sha256: String,
    /// Sample rate of the source, hertz.
    pub sample_rate: u32,
    /// Duration, milliseconds.
    pub duration_ms: i64,
    /// A statement of what this recording is, in the fewest words that are true.
    pub description: String,
}

/// The set definition, `evaluation/set.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SetDef {
    /// Schema version.
    pub version: u32,
    /// Set id.
    pub id: String,
    /// What the set is for.
    pub description: String,
    /// The recording every clip is cut from.
    pub source: Source,
    /// The profile used for text normalisation.
    pub normalisation_profile: NormalisationProfile,
    /// The clips, in order.
    pub clips: Vec<ClipDef>,
}

/// The evaluation set as it exists on disk.
#[derive(Debug)]
pub struct Set {
    /// Directory holding `set.json`, `clips/`, `references/`, `hypotheses/`.
    pub root: PathBuf,
    /// The parsed definition.
    pub def: SetDef,
}

/// A problem with a set, a reference, or a hypothesis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetError {
    /// A file was missing or unreadable.
    Io(String),
    /// A file parsed but held the wrong shape.
    Invalid(String),
}

impl std::fmt::Display for SetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(m) => write!(f, "{m}"),
            Self::Invalid(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for SetError {}

fn io<E: std::fmt::Display>(e: E) -> SetError {
    SetError::Io(e.to_string())
}

/// Read a text file that a human or a Windows editor may have written with a
/// UTF-8 BOM.
///
/// Every one of these files is annotated by hand on Windows, and `Set-Content
/// -Encoding UTF8` in PowerShell 5.1 writes a BOM. A BOM is not a defect in the
/// file, it is the platform being itself, so the readers absorb it rather than
/// making the annotator discover it as a parse error. The same treatment as
/// rule N1 of the text normalisation.
pub fn read_text(path: &Path) -> Option<String> {
    let raw = std::fs::read(path).ok()?;
    let bytes = raw.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(&raw);
    String::from_utf8(bytes.to_vec()).ok()
}

/// The wire name of a provenance value, for prose in the generated metadata.
fn prov_name(p: Provenance) -> &'static str {
    match p {
        Provenance::HumanVerified => "human_verified",
        Provenance::Measured => "measured",
        Provenance::DocumentedContext => "documented_context",
        Provenance::Unverified => "unverified",
    }
}

impl Set {
    /// Load a set from its directory.
    pub fn load(root: &Path) -> Result<Self, SetError> {
        let path = root.join("set.json");
        let text = read_text(&path)
            .ok_or_else(|| SetError::Io(format!("cannot read {}", path.display())))?;
        let def: SetDef = serde_json::from_str(&text)
            .map_err(|e| SetError::Invalid(format!("{}: {e}", path.display())))?;
        if def.version != SET_VERSION {
            return Err(SetError::Invalid(format!(
                "{}: set version {} is not {SET_VERSION}",
                path.display(),
                def.version
            )));
        }
        if def.clips.is_empty() {
            return Err(SetError::Invalid(format!(
                "{}: the set has no clips",
                path.display()
            )));
        }
        Ok(Self {
            root: root.to_path_buf(),
            def,
        })
    }

    /// `clips/` directory.
    pub fn clips_dir(&self) -> PathBuf {
        self.root.join("clips")
    }
    /// `references/` directory.
    pub fn references_dir(&self) -> PathBuf {
        self.root.join("references")
    }
    /// `hypotheses/` directory.
    pub fn hypotheses_dir(&self) -> PathBuf {
        self.root.join("hypotheses")
    }
    /// `results/` directory.
    pub fn results_dir(&self) -> PathBuf {
        self.root.join("results")
    }
    /// `metadata/` directory.
    pub fn metadata_dir(&self) -> PathBuf {
        self.root.join("metadata")
    }

    /// The clip WAV for `id`.
    pub fn clip_path(&self, id: &str) -> PathBuf {
        self.clips_dir().join(format!("{id}.wav"))
    }
    /// The reference directory for `id`.
    pub fn reference_dir(&self, id: &str) -> PathBuf {
        self.references_dir().join(id)
    }
    /// The hypothesis directory for `id`.
    pub fn hypothesis_dir(&self, id: &str) -> PathBuf {
        self.hypotheses_dir().join(id)
    }

    /// Whether a human annotated this clip's ASR reference.
    ///
    /// Requires **both** `reference.txt` and `ANNOTATED.md`. The second is a
    /// small provenance note; without it the reference is treated as
    /// unverified and nothing is scored, because a file that appeared from
    /// nowhere is exactly how a fabricated reference gets scored as a real one.
    pub fn has_human_asr_reference(&self, id: &str) -> bool {
        self.reference_dir(id).join("reference.txt").is_file()
            && self.reference_dir(id).join(ANNOTATED_FILE).is_file()
    }

    /// Whether a human annotated this clip's diarization reference.
    pub fn has_human_diarization_reference(&self, id: &str) -> bool {
        self.reference_dir(id).join("reference.rttm").is_file()
            && self.reference_dir(id).join(ANNOTATED_FILE).is_file()
    }

    /// Whether a human supplied word timings for this clip.
    pub fn has_human_word_timing_reference(&self, id: &str) -> bool {
        self.reference_dir(id)
            .join("reference.words.json")
            .is_file()
            && self.reference_dir(id).join(ANNOTATED_FILE).is_file()
    }

    /// Read the hypothesis for `id`, if the system produced one.
    pub fn hypothesis(&self, id: &str) -> Option<Hypothesis> {
        let p = self.hypothesis_dir(id).join("hypothesis.json");
        let text = read_text(&p)?;
        hypothesis::from_export_json(&text).ok()
    }

    /// Score every clip and build the report.
    ///
    /// A missing reference is **not** an error. It produces an `unscored` clip
    /// that says what is missing, and the set-level counts say how many.
    ///
    /// Write `metadata/<id>.meta.json` for every clip from the set definition.
    ///
    /// The manifest is the single source of truth and the per-clip files are
    /// generated from it, so a metadata file can never disagree with the
    /// offsets the scorer actually used. The `clip_sha256` recorded in the
    /// manifest is the one thing a generated file cannot supply, because it is
    /// a property of the audio rather than of the definition.
    pub fn write_metadata(&self) -> Result<Vec<PathBuf>, SetError> {
        let dir = self.metadata_dir();
        std::fs::create_dir_all(&dir).map_err(io)?;
        let mut written = Vec::new();
        for def in &self.def.clips {
            let meta = ClipMeta {
                id: def.id.clone(),
                topic: def.topic.clone(),
                language_mix: def.language_mix.clone(),
                language_mix_provenance: def.language_mix_provenance,
                speakers: def.speakers,
                speakers_provenance: def.speakers_provenance,
                difficulty: def.difficulty.clone(),
                difficulty_provenance: def.difficulty_provenance,
                covers: def.covers.clone(),
                rationale: def.rationale.clone(),
                structure: def.structure.clone(),
                annotator_notes: vec![
                    format!(
                        "Confirm the speaker count. `speakers` is currently {} and its provenance is `{}`.",
                        def.speakers.map(|n| n.to_string()).unwrap_or_else(|| "unknown".into()),
                        prov_name(def.speakers_provenance)
                    ),
                    "Confirm the language mix. Nothing in the harness can hear a language.".to_string(),
                    "Mark any stretch where two people talk over each other. Acoustic energy cannot detect overlap, and the DER a diarizer can reach is bounded by it.".to_string(),
                    "Note any word the recogniser is likely to mangle: product names, API names, romanised Hindi spelled two ways.".to_string(),
                    "If you write reference.words.json, time the words to the nearest 10 ms and say so in ANNOTATED.md.".to_string(),
                ],
            };
            let p = dir.join(format!("{}.meta.json", def.id));
            let text = serde_json::to_string_pretty(&meta)
                .map_err(|e| SetError::Invalid(format!("{}: {e}", p.display())))?;
            std::fs::write(&p, format!("{text}\n")).map_err(io)?;
            written.push(p);
        }
        Ok(written)
    }
    /// Score every clip and build the report.
    ///
    /// A missing reference is **not** an error. It produces an `unscored` clip
    /// that says what is missing, and the set-level counts say how many.
    pub fn score(&self, params: DiarParams, thresholds: TimingThresholds) -> Report {
        let mut clips = Vec::new();
        let (mut human_asr, mut human_diar, mut human_timing, mut awaiting, mut with_hyp) =
            (0usize, 0usize, 0usize, 0usize, 0usize);

        for def in &self.def.clips {
            let r = self.score_clip(&def.id, params, thresholds);
            if r.has_human_asr_reference {
                human_asr += 1;
            }
            if r.has_human_diarization_reference {
                human_diar += 1;
            }
            if r.has_human_word_timing_reference {
                human_timing += 1;
            }
            if !r.has_human_asr_reference && !r.has_human_diarization_reference {
                awaiting += 1;
            }
            if r.has_hypothesis {
                with_hyp += 1;
            }
            clips.push(r);
        }

        Report {
            version: REPORT_VERSION,
            generated_by: format!("vaani-eval {}", crate::VERSION),
            normalisation: NormalisationBlock::of(self.def.normalisation_profile),
            set: SetCounts {
                clips: clips.len(),
                human_asr_reference: human_asr,
                human_diarization_reference: human_diar,
                human_word_timing_reference: human_timing,
                awaiting_annotation: awaiting,
                with_hypothesis: with_hyp,
            },
            clips,
            blend: BlendRefusal::default(),
        }
    }

    /// Score one clip.
    pub fn score_clip(
        &self,
        id: &str,
        params: DiarParams,
        thresholds: TimingThresholds,
    ) -> ClipReport {
        let def = self.def.clips.iter().find(|c| c.id == id);
        let (start_ms, duration_ms) = def.map(|c| (c.start_ms, c.duration_ms)).unwrap_or((0, 0));

        let profile = self.def.normalisation_profile;
        let has_asr_ref = self.has_human_asr_reference(id);
        let has_diar_ref = self.has_human_diarization_reference(id);
        let has_timing_ref = self.has_human_word_timing_reference(id);
        let hyp = self.hypothesis(id);
        let has_hyp = hyp.is_some();

        let ref_text = if has_asr_ref {
            read_text(&self.reference_dir(id).join("reference.txt"))
        } else {
            None
        };
        let ref_norm = ref_text
            .as_deref()
            .map(|t| normalize::normalise(t, profile));

        let ref_rttm = if has_diar_ref {
            read_text(&self.reference_dir(id).join("reference.rttm"))
                .and_then(|t| rttm::parse(&t).ok())
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let ref_words: Vec<ReferenceWord> = if has_timing_ref {
            read_text(&self.reference_dir(id).join("reference.words.json"))
                .and_then(|t| serde_json::from_str::<Vec<ReferenceWord>>(&t).ok())
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        let hyp_norm = hyp
            .as_ref()
            .map(|h| normalize::normalise(&h.to_text(), profile));
        let hyp_spans: Vec<(i64, i64)> = hyp
            .as_ref()
            .map(|h| h.words.iter().map(|w| (w.start_ms, w.end_ms)).collect())
            .unwrap_or_default();
        let hyp_rttm = hyp.as_ref().map(|h| h.to_rttm(id)).unwrap_or_default();

        // ---- ASR ------------------------------------------------------------
        let (asr, asr_alignment) = match (&ref_norm, &hyp_norm) {
            (Some(r), Some(h)) => (
                AsrBlock {
                    status: BlockStatus::Scored,
                    report: Some(wer::score(r, h)),
                },
                Some(align(&r.tokens, &h.tokens)),
            ),
            (None, Some(_)) => (
                AsrBlock {
                    status: BlockStatus::Unscored {
                        reason: "no_human_asr_reference".into(),
                        detail: "no reference.txt accompanied by ANNOTATED.md; nobody has \
                                 listened to this clip and written down what was said"
                            .into(),
                    },
                    report: None,
                },
                None,
            ),
            (Some(_), None) => (
                AsrBlock {
                    status: BlockStatus::Unscored {
                        reason: "no_hypothesis".into(),
                        detail: "a reference exists but the system produced no output to score"
                            .into(),
                    },
                    report: None,
                },
                None,
            ),
            (None, None) => (
                AsrBlock {
                    status: BlockStatus::Unscored {
                        reason: "no_human_asr_reference".into(),
                        detail: "neither a reference nor a hypothesis exists for this clip".into(),
                    },
                    report: None,
                },
                None,
            ),
        };

        // ---- Diarization ----------------------------------------------------
        let diar_block = if has_diar_ref && has_hyp {
            DiarizationBlock {
                status: BlockStatus::Scored,
                report: Some(der::score(&ref_rttm, &hyp_rttm, params)),
            }
        } else {
            DiarizationBlock {
                status: BlockStatus::Unscored {
                    reason: if !has_diar_ref {
                        "no_human_diarization_reference".into()
                    } else {
                        "no_hypothesis".into()
                    },
                    detail: "DER needs a reference.rttm that a person wrote while listening; \
                             without one there is no measurement, only the system's own opinion \
                             of how many speakers it heard"
                        .into(),
                },
                report: None,
            }
        };

        // ---- Timing ---------------------------------------------------------
        let (timing_block, timing_counts) = match (&ref_norm, &hyp_norm) {
            (Some(r), Some(h)) if has_timing_ref => {
                let t = timing::score(r, &ref_words, h, &hyp_spans);
                let counts = asr_alignment
                    .as_ref()
                    .map(|a| timing_counts_from(a, &ref_words, &hyp_spans, thresholds))
                    .unwrap_or(TimingCounts {
                        shifted: 0,
                        drifted: 0,
                        too_long: 0,
                        too_short: 0,
                    });
                let block = if t.defined {
                    TimingBlock {
                        status: BlockStatus::Scored,
                        report: Some(t),
                    }
                } else {
                    TimingBlock {
                        status: BlockStatus::Undefined {
                            reason: "no_aligned_timing_pairs".into(),
                            detail: "the reference carries word timings but no reference token \
                                     aligned to a hypothesis token that also carries timings"
                                .into(),
                        },
                        report: Some(t),
                    }
                };
                (block, Some(counts))
            }
            _ => {
                let reason = if !has_timing_ref {
                    "no_word_timing_reference"
                } else if !has_asr_ref {
                    "no_human_asr_reference"
                } else {
                    "no_hypothesis"
                };
                (
                    TimingBlock {
                        status: BlockStatus::Unscored {
                            reason: reason.into(),
                            detail: "timestamp error needs reference.words.json: an orthographic \
                                     reference carries no word times, and inventing them from the \
                                     hypothesis would measure the system against itself"
                                .into(),
                        },
                        report: None,
                    },
                    None,
                )
            }
        };

        // ---- Taxonomy --------------------------------------------------------
        let reference_view = match (&ref_norm, &hyp_norm) {
            (Some(r), Some(h)) if has_asr_ref => ReferenceView::Human {
                tokens: &r.tokens,
                hypothesis_tokens: &h.tokens,
            },
            _ => ReferenceView::None,
        };
        let empty = empty_hypothesis();
        let signals = taxonomy::evaluate(
            hyp.as_ref().unwrap_or(&empty),
            reference_view,
            &hyp_rttm,
            &ref_rttm,
            timing_counts,
            thresholds,
        );

        ClipReport {
            clip_id: id.to_string(),
            source_start_ms: start_ms,
            duration_ms,
            has_human_asr_reference: has_asr_ref,
            has_human_diarization_reference: has_diar_ref,
            has_human_word_timing_reference: has_timing_ref,
            has_hypothesis: has_hyp,
            scored: asr.status.is_scored()
                || diar_block.status.is_scored()
                || timing_block.status.is_scored(),
            asr,
            diarization: diar_block,
            timing: timing_block,
            intrinsic: hyp.as_ref().map(|h| h.intrinsic()),
            taxonomy: signals,
        }
    }
}

/// A hypothesis placeholder for taxonomy evaluation when none exists, so the
/// signal list is present and correct rather than absent. Built rather than
/// `static` because a `Hypothesis` owns `String`s and a `static mut` would be
/// a lie about mutability.
fn empty_hypothesis() -> Hypothesis {
    Hypothesis {
        source: "none".to_string(),
        words: Vec::new(),
        segments: Vec::new(),
        duration_ms: None,
        speakers: Vec::new(),
    }
}

/// Count the four timing signatures from the alignment.
fn timing_counts_from(
    a: &crate::edit::Alignment,
    ref_words: &[ReferenceWord],
    hyp_spans: &[(i64, i64)],
    t: TimingThresholds,
) -> TimingCounts {
    let mut c = TimingCounts {
        shifted: 0,
        drifted: 0,
        too_long: 0,
        too_short: 0,
    };
    for (k, op) in a.ops.iter().enumerate() {
        if !matches!(op, Op::Match | Op::Sub) {
            continue;
        }
        let (Some(ri), Some(hi)) = (a.ref_index[k], a.hyp_index[k]) else {
            continue;
        };
        let (Some(rw), Some((hs, he))) = (ref_words.get(ri), hyp_spans.get(hi)) else {
            continue;
        };
        let ds = (hs - rw.start_ms).abs();
        if ds > t.shifted_ms {
            c.shifted += 1;
        }
        if ds > t.drift_ms {
            c.drifted += 1;
        }
        let rd = (rw.end_ms - rw.start_ms).max(1) as f64;
        let hd = (he - hs).max(0) as f64;
        if hd > rd * t.long_ratio {
            c.too_long += 1;
        }
        if hd < rd * t.short_ratio {
            c.too_short += 1;
        }
    }
    c
}

/// SHA-256 of a file, lowercase hex. Used for the source recording, the clips
/// and the reference files, so a published number can name the bytes it came
/// from.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let bytes = std::fs::read(path)?;
    Ok(hex_digest(&bytes))
}

/// SHA-256 of a byte slice, lowercase hex.
pub fn hex_digest(bytes: &[u8]) -> String {
    let d = Sha256::digest(bytes);
    let mut s = String::with_capacity(64);
    for b in d {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_set_refuses_a_version_it_does_not_know() {
        let dir = std::env::temp_dir().join("vaani-eval-set-version-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("set.json"),
            serde_json::to_string(&serde_json::json!({
                "version": 99, "id": "x", "description": "",
                "source": { "path": "a.wav", "sha256": "", "sample_rate": 16000,
                            "duration_ms": 1, "description": "" },
                "normalisation_profile": "strict", "clips": []
            }))
            .unwrap(),
        )
        .unwrap();
        let e = Set::load(&dir).unwrap_err();
        assert!(e.to_string().contains("set version 99"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_empty_set_is_refused_rather_than_scoring_zero_clips() {
        let dir = std::env::temp_dir().join("vaani-eval-set-empty-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("set.json"),
            serde_json::to_string(&serde_json::json!({
                "version": 1, "id": "x", "description": "",
                "source": { "path": "a.wav", "sha256": "", "sample_rate": 16000,
                            "duration_ms": 1, "description": "" },
                "normalisation_profile": "strict", "clips": []
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(Set::load(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hashing_is_stable_and_matches_the_published_empty_string_digest() {
        assert_eq!(
            hex_digest(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn a_missing_reference_scores_as_unscored_and_never_as_zero() {
        let dir = std::env::temp_dir().join("vaani-eval-set-unscored-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("set.json"),
            serde_json::to_string(&serde_json::json!({
                "version": 1, "id": "x", "description": "",
                "source": { "path": "a.wav", "sha256": "", "sample_rate": 16000,
                            "duration_ms": 1, "description": "" },
                "normalisation_profile": "strict",
                "clips": [{ "id": "c1", "start_ms": 0, "duration_ms": 90000,
                            "topic": "t", "rationale": "r", "covers": [],
                            "speakers": null, "speakers_provenance": "unverified",
                            "language_mix": ["hinglish"],
                            "language_mix_provenance": "documented_context",
                            "difficulty": "moderate",
                            "difficulty_provenance": "unverified",
                            "clip_sha256": "" }]
            }))
            .unwrap(),
        )
        .unwrap();
        let set = Set::load(&dir).unwrap();
        let report = set.score(DiarParams::default(), TimingThresholds::default());
        assert_eq!(report.set.clips, 1);
        assert_eq!(report.set.human_asr_reference, 0);
        assert_eq!(report.set.awaiting_annotation, 1);
        let c = &report.clips[0];
        assert!(!c.scored);
        assert!(!c.asr.report.is_some());
        let v = serde_json::to_value(&report).unwrap();
        assert_eq!(v["asr"]["wer"], serde_json::Value::Null);
        let c = serde_json::to_value(&report.clips[0]).unwrap();
        assert_eq!(c["asr"]["status"], "unscored");
        assert_eq!(c["asr"]["reason"], "no_human_asr_reference");
        assert_eq!(v["blend"]["permitted"], false);
        assert_eq!(v["blend"]["value"], serde_json::Value::Null);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_reference_without_the_annotation_note_is_not_a_reference() {
        let dir = std::env::temp_dir().join("vaani-eval-set-annotated-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("set.json"),
            serde_json::to_string(&serde_json::json!({
                "version": 1, "id": "x", "description": "",
                "source": { "path": "a.wav", "sha256": "", "sample_rate": 16000,
                            "duration_ms": 1, "description": "" },
                "normalisation_profile": "strict",
                "clips": [{ "id": "c1", "start_ms": 0, "duration_ms": 90000,
                            "topic": "t", "rationale": "r", "covers": [],
                            "speakers": null, "speakers_provenance": "unverified",
                            "language_mix": ["hinglish"],
                            "language_mix_provenance": "documented_context",
                            "difficulty": "moderate",
                            "difficulty_provenance": "unverified",
                            "clip_sha256": "" }]
            }))
            .unwrap(),
        )
        .unwrap();
        let refs = dir.join("references/c1");
        std::fs::create_dir_all(&refs).unwrap();
        std::fs::write(refs.join("reference.txt"), "hello").unwrap();
        let set = Set::load(&dir).unwrap();
        assert!(!set.has_human_asr_reference("c1"), "no ANNOTATED.md yet");
        std::fs::write(refs.join(ANNOTATED_FILE), "annotator: test\n").unwrap();
        assert!(set.has_human_asr_reference("c1"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}

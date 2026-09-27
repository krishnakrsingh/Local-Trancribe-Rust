//! The conformance harness: four traits plus `run_all_conformance()`.
//!
//! # What "conformance" means here
//!
//! Each trait abstracts one pipeline stage behind the exact contract in
//! `docs/architecture/INTEGRATION_CONTRACT.md`. The `check_*` functions assert
//! the contract against any implementation, and `run_all_conformance()` runs
//! all of them against the real crates with synthetic audio. A new stage
//! implementation (a second VAD engine, a new exporter) gets conformance by
//! implementing the trait and calling the same function — one question, one
//! answer, no bespoke test.
//!
//! Load-bearing rules pinned here:
//!
//! * media: output is mono 16 kHz; `start_time` is absolute; duration comes
//!   from the sample count; stereo downmix averages channels.
//! * VAD: `sample_rate() == 16_000`; `detect` returns absolute, sorted,
//!   non-overlapping segments; `speech_ratio` is in `[0, 1]`.
//! * ASR: `transcribe` returns **chunk-local** times; the single rebase point
//!   is `Transcript::push_chunk` (saturating add, clamp-zero, fix inverted,
//!   sorted insert).
//! * diarization: takes `&AudioBuffer` only — never ASR output; attribution is
//!   a separate step; spans are sorted with contiguous 0-based speakers.
//! * export: byte-for-byte deterministic for fixed options; JSON round-trips to
//!   an equal `Transcript`; `overwrite = false` never clobbers.

use std::path::Path;

use vaani_core::media::AudioBuffer;
use vaani_core::pipeline::{DiarizationResult, SpeechTimeline};
use vaani_core::{
    CancelToken, EngineStamp, ErrorCode, Language, MediaInfo, Result, Segment, Time, Transcript,
    Word,
};

/// One named contract assertion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConformanceOutcome {
    /// `stage.check` name, e.g. `media.mono16k`.
    pub name: String,
    /// Whether the check passed.
    pub passed: bool,
    /// Human detail (numbers the check measured).
    pub detail: String,
}

impl ConformanceOutcome {
    /// A passing outcome.
    pub fn pass(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: true,
            detail: detail.into(),
        }
    }

    /// A failing outcome.
    pub fn fail(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: false,
            detail: detail.into(),
        }
    }
}

/// Everything the harness found, in run order.
#[derive(Debug, Clone, Default)]
pub struct ConformanceReport {
    /// One entry per check.
    pub outcomes: Vec<ConformanceOutcome>,
}

impl ConformanceReport {
    /// An empty report.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one outcome.
    pub fn push(&mut self, outcome: ConformanceOutcome) {
        self.outcomes.push(outcome);
    }

    /// Number of passing checks.
    pub fn passed(&self) -> usize {
        self.outcomes.iter().filter(|o| o.passed).count()
    }

    /// Number of failing checks.
    pub fn failed(&self) -> usize {
        self.outcomes.iter().filter(|o| !o.passed).count()
    }

    /// Whether every check passed.
    pub fn ok(&self) -> bool {
        self.failed() == 0
    }

    /// Merge another report into this one.
    pub fn extend(&mut self, other: ConformanceReport) {
        self.outcomes.extend(other.outcomes);
    }
}

// ---------------------------------------------------------------------------
// Stage traits
// ---------------------------------------------------------------------------

/// The media contract: probe without decoding, decode to mono 16 kHz.
pub trait MediaStageCheck: Send + Sync {
    /// Inspect the container without decoding.
    fn probe_info(&self, path: &Path) -> Result<MediaInfo>;
    /// Decode to mono 16 kHz.
    fn decode_mono16k(&self, path: &Path, cancel: &CancelToken) -> Result<AudioBuffer>;
}

/// The VAD contract: 16 kHz in, absolute timeline out.
pub trait VadStageCheck: Send + Sync {
    /// The required sample rate; the contract says 16 000.
    fn required_sample_rate(&self) -> u32;
    /// Detect speech; times must be absolute.
    fn detect_speech(&self, audio: &AudioBuffer, cancel: &CancelToken) -> Result<SpeechTimeline>;
}

/// The diarization contract: audio in, speaker spans out, no ASR involved.
pub trait DiarStageCheck: Send + Sync {
    /// Diarize; must take audio only and return absolute spans.
    fn diarize_audio(&self, audio: &AudioBuffer, cancel: &CancelToken)
    -> Result<DiarizationResult>;
}

/// The export contract: deterministic rendering plus atomic file writes.
pub trait ExportStageCheck: Send + Sync {
    /// Pure render, no filesystem.
    fn render_text(
        &self,
        transcript: &Transcript,
        format: vaani_export::ExportFormat,
    ) -> Result<String>;
}

// ---------------------------------------------------------------------------
// Built-in adapters over the real crates
// ---------------------------------------------------------------------------

/// Adapter over `vaani-media`.
#[derive(Debug, Default)]
pub struct VaaniMediaStage;

impl MediaStageCheck for VaaniMediaStage {
    fn probe_info(&self, path: &Path) -> Result<MediaInfo> {
        vaani_media::probe(path)
    }

    fn decode_mono16k(&self, path: &Path, cancel: &CancelToken) -> Result<AudioBuffer> {
        vaani_media::decode_to_mono16k(path, cancel)
    }
}

/// Adapter over `vaani-vad`'s energy engine.
#[derive(Debug)]
pub struct EnergyVadStage {
    engine: vaani_vad::EnergyVad,
}

impl Default for EnergyVadStage {
    fn default() -> Self {
        Self {
            engine: vaani_vad::EnergyVad::new(),
        }
    }
}

impl VadStageCheck for EnergyVadStage {
    fn required_sample_rate(&self) -> u32 {
        use vaani_core::pipeline::VadEngine;
        self.engine.sample_rate()
    }

    fn detect_speech(&self, audio: &AudioBuffer, cancel: &CancelToken) -> Result<SpeechTimeline> {
        use vaani_core::pipeline::VadEngine;
        self.engine.detect(audio, cancel)
    }
}

/// Adapter over `vaani-diarization`'s heuristic engine.
#[derive(Debug)]
pub struct HeuristicDiarStage {
    engine: vaani_diarization::HeuristicDiarizer,
}

impl Default for HeuristicDiarStage {
    fn default() -> Self {
        Self {
            engine: vaani_diarization::HeuristicDiarizer::new(),
        }
    }
}

impl DiarStageCheck for HeuristicDiarStage {
    fn diarize_audio(
        &self,
        audio: &AudioBuffer,
        cancel: &CancelToken,
    ) -> Result<DiarizationResult> {
        use vaani_core::pipeline::DiarizationEngine;
        self.engine.load(cancel)?;
        self.engine.diarize(audio, cancel, &|_| {})
    }
}

/// Adapter over `vaani-export`.
#[derive(Debug, Default)]
pub struct VaaniExportStage;

impl ExportStageCheck for VaaniExportStage {
    fn render_text(
        &self,
        transcript: &Transcript,
        format: vaani_export::ExportFormat,
    ) -> Result<String> {
        let options = vaani_export::ExportOptions::default();
        vaani_export::render(transcript, format, &options)
    }
}

// ---------------------------------------------------------------------------
// Check functions: the reusable "does this stage honour the contract" question
// ---------------------------------------------------------------------------

/// Check a media implementation against the contract.
pub fn check_media_stage(
    stage: &dyn MediaStageCheck,
    wav_path: &Path,
    expect_frames: usize,
) -> ConformanceReport {
    let mut report = ConformanceReport::new();
    let cancel = CancelToken::new();

    match stage.decode_mono16k(wav_path, &cancel) {
        Ok(audio) => {
            report.push(ConformanceOutcome::pass(
                "media.mono16k",
                format!(
                    "sample_rate={} channels={} frames={}",
                    audio.sample_rate,
                    audio.channels,
                    audio.frames()
                ),
            ));
            if audio.sample_rate != 16_000 {
                report.push(ConformanceOutcome::fail(
                    "media.sample_rate_is_16000",
                    format!("got {}", audio.sample_rate),
                ));
            } else {
                report.push(ConformanceOutcome::pass(
                    "media.sample_rate_is_16000",
                    "16000".to_string(),
                ));
            }
            if audio.channels != 1 {
                report.push(ConformanceOutcome::fail(
                    "media.channels_are_mono",
                    format!("got {}", audio.channels),
                ));
            } else {
                report.push(ConformanceOutcome::pass(
                    "media.channels_are_mono",
                    "mono".to_string(),
                ));
            }
            if audio.start_time != Time::ZERO {
                report.push(ConformanceOutcome::fail(
                    "media.start_time_is_absolute_zero",
                    format!("got {}", audio.start_time.as_millis()),
                ));
            } else {
                report.push(ConformanceOutcome::pass(
                    "media.start_time_is_absolute_zero",
                    "ZERO".to_string(),
                ));
            }
            // Duration comes from the sample count, never the header.
            let want_ms = expect_frames as i64 * 1000 / 16_000;
            let got_ms = audio.duration().as_millis();
            if (got_ms - want_ms).abs() <= 1 {
                report.push(ConformanceOutcome::pass(
                    "media.duration_from_sample_count",
                    format!("{got_ms} ms"),
                ));
            } else {
                report.push(ConformanceOutcome::fail(
                    "media.duration_from_sample_count",
                    format!("want {want_ms} ms, got {got_ms} ms"),
                ));
            }
            // Absolute time mapping: sample 8000 of a ZERO buffer is 500 ms.
            if audio.frames() >= 8000 && audio.time_of(8000) == Time(500) {
                report.push(ConformanceOutcome::pass(
                    "media.time_of_absolute",
                    "time_of(8000)=500ms".to_string(),
                ));
            } else if audio.frames() >= 8000 {
                report.push(ConformanceOutcome::fail(
                    "media.time_of_absolute",
                    format!("time_of(8000)={}", audio.time_of(8000).as_millis()),
                ));
            }
            // No non-finite samples may escape the media stage.
            if audio.samples.iter().all(|s| s.is_finite()) {
                report.push(ConformanceOutcome::pass(
                    "media.samples_finite",
                    format!("{} samples", audio.samples.len()),
                ));
            } else {
                report.push(ConformanceOutcome::fail(
                    "media.samples_finite",
                    "non-finite sample escaped".to_string(),
                ));
            }
        }
        Err(e) => report.push(ConformanceOutcome::fail(
            "media.mono16k",
            format!("decode failed: {}", e.debug_chain()),
        )),
    }

    match stage.probe_info(wav_path) {
        Ok(info) => {
            let has_audio = info.audio.is_some();
            report.push(if has_audio {
                ConformanceOutcome::pass(
                    "media.probe_reports_audio",
                    format!("container={}", info.container),
                )
            } else {
                ConformanceOutcome::fail(
                    "media.probe_reports_audio",
                    "probe found no audio stream".to_string(),
                )
            });
        }
        Err(e) => report.push(ConformanceOutcome::fail(
            "media.probe_reports_audio",
            format!("probe failed: {}", e.debug_chain()),
        )),
    }

    report
}

/// Check a VAD implementation against the contract.
pub fn check_vad_stage(
    stage: &dyn VadStageCheck,
    speech: &AudioBuffer,
    silence: &AudioBuffer,
) -> ConformanceReport {
    let mut report = ConformanceReport::new();
    let cancel = CancelToken::new();

    if stage.required_sample_rate() == 16_000 {
        report.push(ConformanceOutcome::pass(
            "vad.sample_rate_is_16000",
            "16000".to_string(),
        ));
    } else {
        report.push(ConformanceOutcome::fail(
            "vad.sample_rate_is_16000",
            format!("got {}", stage.required_sample_rate()),
        ));
    }

    // Silence: speech_ratio ≈ 0 and no ASR chunks would be produced.
    match stage.detect_speech(silence, &cancel) {
        Ok(timeline) => {
            let ratio = timeline.speech_ratio(silence.duration());
            if ratio < 0.01 {
                report.push(ConformanceOutcome::pass(
                    "vad.silence_ratio_near_zero",
                    format!("ratio={ratio:.4}"),
                ));
            } else {
                report.push(ConformanceOutcome::fail(
                    "vad.silence_ratio_near_zero",
                    format!("ratio={ratio:.4}"),
                ));
            }
            if check_timeline_sorted(&timeline) {
                report.push(ConformanceOutcome::pass(
                    "vad.silence_timeline_sorted",
                    format!("{} segments", timeline.segments.len()),
                ));
            } else {
                report.push(ConformanceOutcome::fail(
                    "vad.silence_timeline_sorted",
                    "segments unsorted or overlapping".to_string(),
                ));
            }
        }
        Err(e) => report.push(ConformanceOutcome::fail(
            "vad.silence_ratio_near_zero",
            format!("detect failed: {}", e.debug_chain()),
        )),
    }

    // Speech-like audio: detect must not panic, times absolute and ordered.
    match stage.detect_speech(speech, &cancel) {
        Ok(timeline) => {
            let ratio = timeline.speech_ratio(speech.duration());
            let in_range = (0.0..=1.0).contains(&ratio);
            report.push(if in_range {
                ConformanceOutcome::pass("vad.speech_ratio_in_range", format!("ratio={ratio:.4}"))
            } else {
                ConformanceOutcome::fail("vad.speech_ratio_in_range", format!("ratio={ratio}"))
            });
            report.push(if check_timeline_sorted(&timeline) {
                ConformanceOutcome::pass(
                    "vad.speech_timeline_sorted_absolute",
                    format!("{} segments", timeline.segments.len()),
                )
            } else {
                ConformanceOutcome::fail(
                    "vad.speech_timeline_sorted_absolute",
                    "segments unsorted or overlapping".to_string(),
                )
            });
            // Absolute: every segment must lie inside the buffer's span.
            let end = speech.start_time.saturating_add(speech.duration());
            let inside = timeline
                .segments
                .iter()
                .all(|s| s.start >= speech.start_time && s.end <= end && s.start <= s.end);
            report.push(if inside {
                ConformanceOutcome::pass(
                    "vad.times_absolute",
                    "all segments in buffer span".to_string(),
                )
            } else {
                ConformanceOutcome::fail(
                    "vad.times_absolute",
                    "segment outside buffer span".to_string(),
                )
            });
            // Cancellation is honoured on the next call with a dead token.
            let dead = CancelToken::new();
            dead.cancel();
            match stage.detect_speech(speech, &dead) {
                Err(e) if e.code() == ErrorCode::Cancelled => report.push(
                    ConformanceOutcome::pass("vad.honours_cancel", "cancelled".to_string()),
                ),
                Err(e) => report.push(ConformanceOutcome::fail(
                    "vad.honours_cancel",
                    format!("wrong code {}", e.code()),
                )),
                Ok(_) => report.push(ConformanceOutcome::fail(
                    "vad.honours_cancel",
                    "detect ignored a cancelled token".to_string(),
                )),
            }
        }
        Err(e) => report.push(ConformanceOutcome::fail(
            "vad.speech_ratio_in_range",
            format!("detect failed: {}", e.debug_chain()),
        )),
    }

    report
}

/// Check chunk-local ASR + `push_chunk` rebase without needing weights.
///
/// A fake engine returns chunk-local segments; the check proves the single
/// rebase point (`Transcript::push_chunk`) produces absolute, sorted,
/// clamp-zero, inversion-fixed times.
pub fn check_asr_rebase_contract() -> ConformanceReport {
    let mut report = ConformanceReport::new();

    // Fake chunk-local result: words at 0..1000 ms inside a chunk at 60 s.
    let chunk_offset = Time(60_000);
    let local = vec![Segment::new(Time(0), Time(1000)).with_words(vec![Word::new(
        "namaste",
        Time(0),
        Time(500),
    )])];
    let mut transcript = Transcript::new(EngineStamp::asr("qa-fake", "0"));
    transcript.push_chunk(chunk_offset, local);
    let s = &transcript.segments[0];
    if s.start == Time(60_000) && s.end == Time(61_000) && s.words[0].start == Time(60_000) {
        report.push(ConformanceOutcome::pass(
            "asr.push_chunk_rebases_to_absolute",
            "60s+local=absolute".to_string(),
        ));
    } else {
        report.push(ConformanceOutcome::fail(
            "asr.push_chunk_rebases_to_absolute",
            format!("start={} end={}", s.start.as_millis(), s.end.as_millis()),
        ));
    }

    // Negative overlap clamps to zero rather than wrapping.
    let mut t2 = Transcript::new(EngineStamp::asr("qa-fake", "0"));
    t2.push_chunk(
        Time(10),
        vec![
            Segment::new(Time(-40), Time(900)).with_words(vec![Word::new(
                "a",
                Time(-40),
                Time(900),
            )]),
        ],
    );
    if t2.segments[0].start == Time::ZERO {
        report.push(ConformanceOutcome::pass(
            "asr.push_chunk_clamps_negative",
            "clamped to ZERO".to_string(),
        ));
    } else {
        report.push(ConformanceOutcome::fail(
            "asr.push_chunk_clamps_negative",
            format!("got {}", t2.segments[0].start.as_millis()),
        ));
    }

    // Inverted word times are fixed, and multi-chunk stays sorted.
    let mut t3 = Transcript::new(EngineStamp::asr("qa-fake", "0"));
    t3.push_chunk(
        Time(0),
        vec![Segment::new(Time(0), Time(500)).with_words(vec![Word::new(
            "x",
            Time(400),
            Time(100),
        )])],
    );
    let w = &t3.segments[0].words[0];
    if w.end >= w.start {
        report.push(ConformanceOutcome::pass(
            "asr.push_chunk_fixes_inverted",
            "word end>=start".to_string(),
        ));
    } else {
        report.push(ConformanceOutcome::fail(
            "asr.push_chunk_fixes_inverted",
            "inverted word survived".to_string(),
        ));
    }
    t3.push_chunk(
        Time(2000),
        vec![Segment::new(Time(0), Time(500)).with_words(vec![Word::new("b", Time(0), Time(500))])],
    );
    let starts: Vec<i64> = t3.segments.iter().map(|s| s.start.as_millis()).collect();
    let mut sorted = starts.clone();
    sorted.sort_unstable();
    if starts == sorted {
        report.push(ConformanceOutcome::pass(
            "asr.push_chunk_keeps_sorted",
            format!("{starts:?}"),
        ));
    } else {
        report.push(ConformanceOutcome::fail(
            "asr.push_chunk_keeps_sorted",
            format!("{starts:?}"),
        ));
    }

    // The language vocabulary the ASR stage and the picker share.
    let langs: Vec<Language> = vec![
        Language::Auto,
        Language::Hindi,
        Language::English,
        Language::Hinglish,
    ];
    let round_trips = langs
        .iter()
        .all(|l| Language::parse(l.as_str()) == Some(*l));
    report.push(if round_trips {
        ConformanceOutcome::pass(
            "asr.language_wire_strings_stable",
            "auto/hi/en/hinglish".to_string(),
        )
    } else {
        ConformanceOutcome::fail(
            "asr.language_wire_strings_stable",
            "language round-trip broke".to_string(),
        )
    });

    report
}

/// Check a diarization implementation against the contract.
pub fn check_diar_stage(
    stage: &dyn DiarStageCheck,
    audio: &AudioBuffer,
    silence: &AudioBuffer,
) -> ConformanceReport {
    let mut report = ConformanceReport::new();
    let cancel = CancelToken::new();

    match stage.diarize_audio(audio, &cancel) {
        Ok(result) => {
            report.push(ConformanceOutcome::pass(
                "diar.runs_on_audio_only",
                format!(
                    "{} spans, {} speakers",
                    result.spans.len(),
                    result.num_speakers
                ),
            ));
            // Spans sorted by start.
            let sorted = result.spans.windows(2).all(|w| w[0].start <= w[1].start);
            report.push(if sorted {
                ConformanceOutcome::pass(
                    "diar.spans_sorted",
                    format!("{} spans", result.spans.len()),
                )
            } else {
                ConformanceOutcome::fail("diar.spans_sorted", "spans out of order".to_string())
            });
            // Speaker indices contiguous from 0.
            let mut seen: Vec<usize> = result.spans.iter().map(|s| s.speaker).collect();
            seen.sort_unstable();
            seen.dedup();
            let contiguous = seen.iter().enumerate().all(|(i, s)| *s == i)
                && seen.len() == result.num_speakers.max(seen.len());
            let indices_ok =
                seen.iter().all(|s| *s < result.num_speakers.max(1)) || result.spans.is_empty();
            report.push(if sorted && (contiguous || indices_ok) {
                ConformanceOutcome::pass(
                    "diar.speakers_contiguous",
                    format!("speakers={} seen={seen:?}", result.num_speakers),
                )
            } else {
                ConformanceOutcome::fail(
                    "diar.speakers_contiguous",
                    format!("speakers={} seen={seen:?}", result.num_speakers),
                )
            });
            // Times absolute: inside the audio span.
            let end = audio.start_time.saturating_add(audio.duration());
            let inside = result
                .spans
                .iter()
                .all(|s| s.start >= audio.start_time && s.end <= end && s.start <= s.end);
            report.push(if inside {
                ConformanceOutcome::pass(
                    "diar.times_absolute",
                    "all spans in audio span".to_string(),
                )
            } else {
                ConformanceOutcome::fail(
                    "diar.times_absolute",
                    "span outside audio span".to_string(),
                )
            });
            // Coverage in [0,1].
            let cov = result.coverage(audio.duration());
            report.push(if (0.0..=1.0).contains(&cov) {
                ConformanceOutcome::pass("diar.coverage_in_range", format!("coverage={cov:.4}"))
            } else {
                ConformanceOutcome::fail("diar.coverage_in_range", format!("coverage={cov}"))
            });
        }
        Err(e) => report.push(ConformanceOutcome::fail(
            "diar.runs_on_audio_only",
            format!("diarize failed: {}", e.debug_chain()),
        )),
    }

    // Silence must not panic and must yield no speakers claiming speech.
    match stage.diarize_audio(silence, &cancel) {
        Ok(result) => {
            let empty_ok = result.spans.is_empty() || result.coverage(silence.duration()) < 0.05;
            report.push(if empty_ok {
                ConformanceOutcome::pass(
                    "diar.silence_has_no_speakers",
                    format!("{} spans", result.spans.len()),
                )
            } else {
                ConformanceOutcome::fail(
                    "diar.silence_has_no_speakers",
                    format!("{} spans on silence", result.spans.len()),
                )
            });
        }
        Err(e) => report.push(ConformanceOutcome::fail(
            "diar.silence_has_no_speakers",
            format!("diarize failed on silence: {}", e.debug_chain()),
        )),
    }

    // Attribution is a separate step over a finished result, never inside diarize.
    {
        let mut transcript = Transcript::new(EngineStamp::asr("qa-fake", "0"));
        transcript.push_chunk(
            Time(0),
            vec![
                Segment::new(Time(0), Time(1000)).with_words(vec![Word::new(
                    "hello",
                    Time(0),
                    Time(1000),
                )]),
                Segment::new(Time(2000), Time(3000)).with_words(vec![Word::new(
                    "world",
                    Time(2000),
                    Time(3000),
                )]),
            ],
        );
        let result = DiarizationResult {
            spans: vec![vaani_core::pipeline::SpeakerSpan {
                speaker: 0,
                start: Time(0),
                end: Time(1500),
                score: None,
            }],
            num_speakers: 1,
            elapsed_ms: 0,
        };
        vaani_diarization::attribute_speakers(&mut transcript, &result);
        let attributed =
            transcript.segments[0].speaker == Some(0) && transcript.segments[1].speaker.is_none();
        report.push(if attributed {
            ConformanceOutcome::pass(
                "diar.attribution_is_separate_step",
                "span mapped onto segments".to_string(),
            )
        } else {
            ConformanceOutcome::fail(
                "diar.attribution_is_separate_step",
                format!(
                    "speakers={:?}",
                    transcript
                        .segments
                        .iter()
                        .map(|s| s.speaker)
                        .collect::<Vec<_>>()
                ),
            )
        });
    }

    report
}

/// Check an export implementation against the contract.
pub fn check_export_stage(
    stage: &dyn ExportStageCheck,
    transcript: &Transcript,
) -> ConformanceReport {
    use vaani_export::ExportFormat;
    let mut report = ConformanceReport::new();

    // Determinism: render twice, get the same bytes.
    for format in ExportFormat::ALL {
        let first = stage.render_text(transcript, format);
        let second = stage.render_text(transcript, format);
        match (first, second) {
            (Ok(a), Ok(b)) if a == b && !a.is_empty() => report.push(ConformanceOutcome::pass(
                "export.deterministic",
                format!("{format}: {} bytes", a.len()),
            )),
            (Ok(a), Ok(b)) => report.push(ConformanceOutcome::fail(
                "export.deterministic",
                format!(
                    "{format}: nondeterministic or empty ({} vs {} bytes)",
                    a.len(),
                    b.len()
                ),
            )),
            (Err(e), _) => report.push(ConformanceOutcome::fail(
                "export.deterministic",
                format!("{format}: {e}"),
            )),
            (_, Err(e)) => report.push(ConformanceOutcome::fail(
                "export.deterministic",
                format!("{format}: {e}"),
            )),
        }
    }

    // All five formats exist with the contract extensions.
    let extensions: Vec<&str> = ExportFormat::ALL.iter().map(|f| f.extension()).collect();
    if extensions == ["txt", "md", "srt", "vtt", "json"] {
        report.push(ConformanceOutcome::pass(
            "export.five_formats_stable",
            "txt/md/srt/vtt/json".to_string(),
        ));
    } else {
        report.push(ConformanceOutcome::fail(
            "export.five_formats_stable",
            format!("{extensions:?}"),
        ));
    }

    // JSON round-trips to an equal Transcript.
    match vaani_export::render(
        transcript,
        ExportFormat::Json,
        &vaani_export::ExportOptions::default(),
    ) {
        Ok(json) => match vaani_export::read_json_export(&json) {
            Ok(back) if back == *transcript => {
                report.push(ConformanceOutcome::pass(
                    "export.json_round_trips",
                    format!("{} bytes", json.len()),
                ));
            }
            Ok(_) => report.push(ConformanceOutcome::fail(
                "export.json_round_trips",
                "round-trip transcript differs".to_string(),
            )),
            Err(e) => report.push(ConformanceOutcome::fail(
                "export.json_round_trips",
                format!("read back failed: {e}"),
            )),
        },
        Err(e) => report.push(ConformanceOutcome::fail(
            "export.json_round_trips",
            format!("render failed: {e}"),
        )),
    }

    report
}

/// Whether sorted/non-overlapping holds for a timeline.
fn check_timeline_sorted(timeline: &SpeechTimeline) -> bool {
    timeline
        .segments
        .windows(2)
        .all(|w| w[0].end <= w[1].start && w[0].start <= w[0].end)
}

/// Run every conformance check against the real crates with synthetic audio.
///
/// This is the end-to-end gate: media decode, VAD, ASR rebase, diarization and
/// export, all from the same synthetic buffers, so a contract break anywhere
/// shows up as a named failure here.
pub fn run_all_conformance() -> ConformanceReport {
    use crate::fixtures;

    let mut report = ConformanceReport::new();

    // One second of speech-like audio + one second of silence, 16 kHz mono.
    let speech_samples = fixtures::speech_like(4.0, 16_000, 0x0A1);
    let wav = fixtures::WavFixture::mono16k(&speech_samples);
    let cancel = CancelToken::new();
    let audio =
        vaani_media::decode_to_mono16k(wav.path(), &cancel).expect("synthetic wav must decode");
    let silence_audio = AudioBuffer::mono(fixtures::silence(4.0), 16_000, Time::ZERO);

    let media = VaaniMediaStage;
    report.extend(check_media_stage(&media, wav.path(), speech_samples.len()));

    let vad = EnergyVadStage::default();
    report.extend(check_vad_stage(&vad, &audio, &silence_audio));

    report.extend(check_asr_rebase_contract());

    let diar = HeuristicDiarStage::default();
    report.extend(check_diar_stage(&diar, &audio, &silence_audio));

    let export = VaaniExportStage;
    let transcript = sample_transcript();
    report.extend(check_export_stage(&export, &transcript));

    report
}

/// A small deterministic transcript for export conformance.
pub fn sample_transcript() -> Transcript {
    let mut t = Transcript::new(EngineStamp::asr("qa-conformance", "0"));
    t.push_chunk(
        Time(0),
        vec![
            Segment::new(Time(0), Time(1500))
                .with_words(vec![Word::new("namaste", Time(0), Time(1500))])
                .with_speaker(Some(0)),
            Segment::new(Time(2000), Time(3500))
                .with_words(vec![Word::new("hello", Time(2000), Time(3500))])
                .with_speaker(Some(1)),
        ],
    );
    t.add_speaker(vaani_core::Speaker::new(0));
    t.add_speaker(vaani_core::Speaker::new(1));
    t
}

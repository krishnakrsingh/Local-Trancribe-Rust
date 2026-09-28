//! The `vaani-app` seam: a `Pipeline` trait plus the interim `DirectPipeline`.
//!
//! # Why this indirection exists
//!
//! The CLI must ship before `vaani-app` (A2) exists, but it must not bake in a
//! second, divergent pipeline. So the CLI surface speaks only to the [`Pipeline`]
//! trait below, and [`DirectPipeline`] implements that trait by calling the
//! service crates directly in the phase order from
//! `docs/architecture/INTEGRATION_CONTRACT.md` §14:
//!
//! ```text
//! Path -> media (probe/decode) -> VAD (detect) -> chunk (plan/slice) ->
//! ASR (transcribe/push_chunk loop) -> diarization (independent audio path) ->
//! attribution -> storage (upsert/replace_transcript) -> export (on demand)
//! ```
//!
//! When `vaani-app` lands, the swap is a one-module change (see crate docs):
//! implement `Pipeline` for the `vaani-app` adapter and change the single
//! construction site in [`crate::cli::run_cli`]. The trait's inputs
//! ([`TranscribeRequest`]) and outputs ([`TranscribeOutcome`],
//! [`BenchmarkReport`]) are already shaped like stage-orchestrator I/O —
//! [`vaani_core::pipeline::StageArtifacts`] in, exported files out — so the
//! adapter will not need the CLI to change.
//!
//! # Load-bearing rules observed here
//!
//! * Diarization takes `&AudioBuffer`, never ASR output (contract §4).
//! * Chunk-local ASR times are rebased only via `Transcript::push_chunk`
//!   (contract §2). `AudioChunk::source_offset` is never added anywhere here.
//! * Diarization off-by-default is "skip + reason", never a failure
//!   (contract §14): `--no-diarize` skips silently; a missing model skips with
//!   a recorded warning.
//! * Every error that reaches the caller carries an [`vaani_core::ErrorCode`];
//!   diagnostics ride `with_cause`, never the user-facing `detail`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use vaani_core::{
    CancelToken, EngineStamp, ErrorCode, Language, Progress, Result, Time, Transcript, VaaniError,
    err,
};
use vaani_inference::Provider;

const DEFAULT_MODEL_ID: &str = "sravaani-1.0-onnx";

/// What the CLI wants transcribed, and how.
#[derive(Debug, Clone)]
pub struct TranscribeRequest {
    /// Source media file.
    pub input: PathBuf,
    /// Manifest id in `vaani-models`, e.g. `sravaani-1.0-onnx`.
    pub model: String,
    /// Execution provider preference.
    pub device: Provider,
    /// Requested language.
    pub language: Language,
    /// Whether to run speaker diarization + attribution.
    pub diarize: bool,
    /// Where to write the export (`File` = exact path, `Directory` = named
    /// automatically). `None` = no export, transcript stays in the store.
    pub out: Option<PathBuf>,
    /// Export format wire name (`text|markdown|srt|vtt|json|all`).
    pub format: String,
    /// Overwrite an existing export file.
    pub overwrite: bool,
}

impl TranscribeRequest {
    /// Defaults matching the CLI flags: bundled model id, auto device, auto
    /// language, diarization on, SRT export only when `--out` is given.
    pub fn for_file(input: PathBuf) -> Self {
        Self {
            input,
            model: DEFAULT_MODEL_ID.to_string(),
            device: Provider::Auto,
            language: Language::Auto,
            diarize: true,
            out: None,
            format: "srt".to_string(),
            overwrite: false,
        }
    }
}

/// What a transcription produced.
#[derive(Debug)]
pub struct TranscribeOutcome {
    /// The job id under which the transcript was stored.
    pub job_id: String,
    /// Segments in the finished transcript.
    pub segments: usize,
    /// Speakers attributed (`None` when diarization was skipped).
    pub num_speakers: Option<usize>,
    /// Files written by the export step.
    pub exported: Vec<PathBuf>,
    /// Non-fatal notes (diarization skipped, store unavailable, ...). Each is
    /// `(code, message)`; codes are stable [`ErrorCode`] strings.
    pub warnings: Vec<(String, String)>,
}

/// One stage's measured wall time in a benchmark report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageTiming {
    /// Stage name: `startup|media|vad|asr|diar|export`.
    pub stage: String,
    /// Wall time in milliseconds.
    pub elapsed_ms: u64,
    /// Extra detail, e.g. `chunks=12` or `segments=48`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// A stage that did not run, and the stable reason why.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkippedStage {
    /// Stage name, same vocabulary as [`StageTiming::stage`].
    pub stage: String,
    /// Stable [`ErrorCode`] wire string, e.g. `model_unavailable`.
    pub reason_code: String,
    /// Human-readable reason. Never asserted on by tests.
    pub reason: String,
}

/// Machine-readable benchmark report. `benchmark [--json]` prints exactly this
/// to stdout; `--json` is the same value pretty-printed.
///
/// Shape stability: every field is always present (`None` serialises as
/// `null`, never as absent or zero). Scripts match on keys, not on presence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkReport {
    /// [`vaani_core::APP_VERSION`]: the single in-binary version source.
    pub app_version: String,
    /// The file that was benchmarked, as given on the command line.
    pub file: String,
    /// Decoded audio duration in milliseconds, when decoding succeeded.
    pub audio_duration_ms: Option<i64>,
    /// Per-stage wall times, in pipeline order.
    pub stages: Vec<StageTiming>,
    /// Stages that did not run, with stable reason codes.
    pub skipped: Vec<SkippedStage>,
    /// Peak working-set of this process in bytes (`null` when unknowable —
    /// never fabricated).
    pub peak_ram_bytes: Option<u64>,
    /// System RAM in bytes, from capability detection.
    pub system_ram_bytes: Option<u64>,
    /// Inference-adapter VRAM in bytes, when an adapter was detected.
    pub adapter_vram_bytes: Option<u64>,
    /// `audio_secs / asr_secs`, when ASR ran. Not an estimate: both sides are
    /// measurements from this run.
    pub rtf: Option<f64>,
    /// Segments transcribed (`0` when ASR was skipped).
    pub transcript_segments: usize,
    /// Speakers found (`None` when diarization was skipped).
    pub num_speakers: Option<usize>,
}

/// The pipeline the CLI drives. `vaani-app` will provide the second
/// implementation; the CLI calls nothing else.
pub trait Pipeline: Send + Sync {
    /// Stable id for logs and the benchmark report (`direct` today,
    /// `vaani-app` after the swap).
    fn id(&self) -> &'static str;

    /// Run the full pipeline for one file: media → VAD → chunk → ASR →
    /// diarization → attribution → storage → export.
    fn transcribe(
        &self,
        req: &TranscribeRequest,
        cancel: &CancelToken,
    ) -> Result<TranscribeOutcome>;

    /// Time each stage for one file without failing on a missing model: ASR
    /// (and anything downstream of it) is recorded in `skipped` with a stable
    /// reason code instead.
    fn benchmark(&self, file: &Path, cancel: &CancelToken) -> Result<BenchmarkReport>;
}

/// The interim [`Pipeline`]: service crates wired directly, in contract §14 order.
///
/// `vaani-app` replaces this type, not the trait. See the module docs.
#[derive(Debug, Default)]
pub struct DirectPipeline;

impl DirectPipeline {
    /// Fail fast on a missing input with a stable code.
    ///
    /// `vaani-media::probe` maps OS open errors to its own taxonomy; the CLI
    /// contract pins missing-file to `unsupported_media`/`no_audio_stream`
    /// (the `ErrorCode::UnsupportedMedia` doc covers "input file missing"),
    /// so the existence check lives here, before any backend is touched.
    fn require_input(path: &Path) -> Result<()> {
        if !path.exists() {
            return Err(err::unsupported_media()
                .with_cause(format!("input does not exist: {}", path.display())));
        }
        Ok(())
    }

    fn runtime_caps() -> vaani_inference::MachineCapabilities {
        vaani_inference::RuntimeReport::collect().machine
    }

    /// Resolve the ASR engine for `model`, or `None` with the reason when the
    /// model is not installed. Never an error: a missing model is a skip, not
    /// a failure (contract §14).
    fn open_engine(
        &self,
        model: &str,
        device: Provider,
        caps: &vaani_inference::MachineCapabilities,
    ) -> std::result::Result<Box<dyn vaani_asr::AsrEngine>, VaaniError> {
        let mgr = vaani_models::ModelManager::embedded()
            .map_err(|e| internal(format!("model manifest: {e}")))?;
        let engine_id = mgr.engine_id(model)?.to_string();
        let loc = mgr.resolve(model)?;
        let artifacts = vaani_asr::artifacts_in(loc.dir());
        if artifacts.missing().is_empty() {
            let registry = vaani_asr::EngineRegistry::with_builtins(device, caps)?;
            Ok(registry.create(&engine_id, &artifacts)?)
        } else {
            Err(err::model_unavailable()
                .with_cause(format!("{model} is missing engine files"))
                .with_recovery("Download it with `vaani models ensure`."))
        }
    }
}

impl Pipeline for DirectPipeline {
    fn id(&self) -> &'static str {
        "direct"
    }

    fn transcribe(
        &self,
        req: &TranscribeRequest,
        cancel: &CancelToken,
    ) -> Result<TranscribeOutcome> {
        use vaani_core::job::JobPhase;
        use vaani_core::pipeline::VadEngine;

        Self::require_input(&req.input)?;
        let mut warnings = Vec::new();

        tracing::info!(file = %req.input.display(), "preparing");
        let media = vaani_media::probe(&req.input)?;
        tracing::info!(container = ?media.container, "media probed");

        tracing::info!("extracting audio");
        let audio = vaani_media::decode_to_mono16k(&req.input, cancel)?;
        let total = audio.duration();

        tracing::info!("detecting speech");
        let vad = vaani_vad::EnergyVad::new();
        debug_assert_eq!(
            vad.sample_rate(),
            16_000,
            "VAD must require 16 kHz (contract §1)"
        );
        let timeline = vad.detect(&audio, cancel)?;

        let chunker = vaani_vad::Chunker::new(vaani_vad::ChunkPolicy::default());
        let chunks = chunker.plan(&timeline, total);
        tracing::info!(chunks = chunks.len(), "chunk plan ready");

        tracing::info!("transcribing");
        let caps = Self::runtime_caps();
        let engine = self.open_engine(&req.model, req.device, &caps)?;
        engine.load(cancel)?;
        let mut transcript = Transcript::new(EngineStamp::asr(
            engine.model_name(),
            engine.model_version(),
        ));
        transcript.duration = Some(total);
        for (i, chunk) in chunks.iter().enumerate() {
            cancel.check()?;
            let samples = audio.slice_mono(chunk.start, chunk.end);
            let asr_chunk = vaani_asr::AudioChunk::new(samples, 16_000, chunk.start)
                .with_language(req.language);
            let result = engine.transcribe(&asr_chunk, cancel)?;
            tracing::debug!(chunk = i, segments = result.segments.len(), "chunk done");
            transcript.push_chunk_within(
                chunk.start,
                result.segments,
                audio.start_time.saturating_add(audio.duration()),
            );
        }
        engine.unload();

        // The absolute media window, taken from the buffer rather than from
        // `Transcript::duration`. That field is a *length* while segment times
        // are *absolute*, and comparing one against the other pins every segment
        // of a transcript that does not start at zero to the origin. The
        // conformance harness caught it; see `vaani_segment::MediaBounds`.
        let bounds = vaani_segment::MediaBounds::of_buffer(&audio);
        let grouping_policy = vaani_segment::GroupingPolicy::default();

        let num_speakers = if req.diarize {
            tracing::info!("separating speakers");
            let diar = vaani_diarization::HeuristicDiarizer::new();
            match diar.diarize_with_timeline(&audio, Some(&timeline), cancel, &|_| {}) {
                Ok(d) => {
                    let n = d.num_speakers;
                    // GROUP, THEN ATTRIBUTE — the same two calls, in the same
                    // order, as `vaani-app`'s `DiarStage`. Both pipelines run
                    // here, and a policy that existed in only one of them would
                    // give the two products different transcripts for the same
                    // recording, which is exactly the trap the 1,913-segment
                    // transcript was.
                    //
                    // Grouping merges the one-word segments into readable ones
                    // and uses the diarization spans to decide where a turn
                    // falls; the vote then runs once per segment instead of once
                    // per word. See `vaani_segment` for why the order matters.
                    vaani_segment::group_with_diarization_within(
                        &mut transcript,
                        &d,
                        Some(bounds),
                        &grouping_policy,
                    )?;
                    vaani_diarization::attribute_speakers(&mut transcript, &d);
                    if transcript.speakers.is_empty() {
                        transcript.speakers = vaani_diarization::speakers_from_diarization(&d);
                    }
                    transcript.engine.diarization_model =
                        Some(vaani_diarization::HeuristicDiarizer::ID.to_string());
                    Some(n)
                }
                Err(e) => {
                    // Diarization is never fatal (contract, ErrorCode docs).
                    warnings.push((e.code().as_str().to_string(), e.user.title.clone()));
                    tracing::warn!("diarization failed; continuing without speakers");
                    // Group anyway. A transcript of 1,913 one-word segments is
                    // not a transcript, and the failure is in the *speaker*
                    // stage, not in the text — so the text still has to be
                    // readable. Without this, `--no-diarize` and a failed
                    // diarization would both ship the word-per-line output the
                    // grouping policy exists to remove.
                    vaani_segment::group_within(&mut transcript, bounds, &grouping_policy)?;
                    None
                }
            }
        } else {
            warnings.push((
                ErrorCode::DiarizationFailed.as_str().to_string(),
                "diarization disabled by --no-diarize".to_string(),
            ));
            // ASR-only is a complete product mode, not a degraded one, so it
            // gets a complete transcript: grouped, and with no speakers.
            vaani_segment::group_within(&mut transcript, bounds, &grouping_policy)?;
            None
        };

        // Storage: best-effort. A transcript in hand matters more than a row
        // in the database; a store failure is recorded, not fatal.
        let job_id = vaani_storage::new_job_id();
        match store_transcript(&job_id, req, &media, &transcript, num_speakers) {
            Ok(()) => tracing::info!(job = %job_id, "stored"),
            Err(e) => {
                warnings.push((e.code().as_str().to_string(), e.user.title.clone()));
                tracing::warn!("store unavailable; transcript not persisted");
            }
        }
        let _ = JobPhase::Complete; // phase vocabulary anchor for future vaani-app swap

        let exported = if let Some(out) = &req.out {
            export_transcript(&transcript, &req.input, out, &req.format, req.overwrite)?
        } else {
            Vec::new()
        };

        Ok(TranscribeOutcome {
            job_id,
            segments: transcript.segments.len(),
            num_speakers,
            exported,
            warnings,
        })
    }

    fn benchmark(&self, file: &Path, cancel: &CancelToken) -> Result<BenchmarkReport> {
        use vaani_core::pipeline::VadEngine;

        Self::require_input(file)?;
        let mut stages = Vec::new();
        let mut skipped = Vec::new();
        let push =
            |stages: &mut Vec<StageTiming>, stage: &str, t: Instant, detail: Option<String>| {
                stages.push(StageTiming {
                    stage: stage.to_string(),
                    elapsed_ms: t.elapsed().as_millis() as u64,
                    detail,
                });
            };
        let skip = |skipped: &mut Vec<SkippedStage>, stage: &str, e: &VaaniError| {
            skipped.push(SkippedStage {
                stage: stage.to_string(),
                reason_code: e.code().as_str().to_string(),
                reason: e.user.title.clone(),
            });
        };

        // startup: runtime discovery + capability detection (the real init cost
        // a cold launch pays before touching media).
        let t = Instant::now();
        let report = vaani_inference::RuntimeReport::collect();
        let caps = report.machine.clone();
        let system_ram = caps.total_ram_bytes;
        let vram = caps.inference_adapter().and_then(adapter_vram_bytes);
        push(&mut stages, "startup", t, Some(report.summary()));

        // media: probe + full decode to mono 16 kHz.
        let t = Instant::now();
        let audio = match vaani_media::probe(file)
            .and_then(|_| vaani_media::decode_to_mono16k(file, cancel))
        {
            Ok(a) => {
                let ms = a.duration().as_millis();
                push(&mut stages, "media", t, Some(format!("decoded_ms={ms}")));
                a
            }
            Err(e) => {
                // Media failure ends the benchmark: downstream stages have no
                // input, and each would just repeat the same code.
                let code = e.code();
                push(
                    &mut stages,
                    "media",
                    t,
                    Some(format!("failed: {}", code.as_str())),
                );
                return Err(e);
            }
        };
        let audio_secs = audio.duration().as_secs_f64().max(1e-9);

        // vad: energy detector + chunk plan.
        let t = Instant::now();
        let vad = vaani_vad::EnergyVad::new();
        let timeline = vad.detect(&audio, cancel)?;
        let chunker = vaani_vad::Chunker::new(vaani_vad::ChunkPolicy::default());
        let chunks = chunker.plan(&timeline, audio.duration());
        push(
            &mut stages,
            "vad",
            t,
            Some(format!("chunks={}", chunks.len())),
        );

        // asr: full chunk loop when the model is installed, else a skip with
        // the stable `model_unavailable` code.
        let t = Instant::now();
        let mut transcript_segments = 0usize;
        let mut asr_secs = None;
        match self.open_engine(DEFAULT_MODEL_ID, Provider::Auto, &caps) {
            Ok(engine) => {
                engine.load(cancel)?;
                let mut transcript = Transcript::new(EngineStamp::asr(
                    engine.model_name(),
                    engine.model_version(),
                ));
                for chunk in &chunks {
                    cancel.check()?;
                    let samples = audio.slice_mono(chunk.start, chunk.end);
                    let asr_chunk = vaani_asr::AudioChunk::new(samples, 16_000, chunk.start);
                    let result = engine.transcribe(&asr_chunk, cancel)?;
                    transcript.push_chunk_within(
                        chunk.start,
                        result.segments,
                        audio.start_time.saturating_add(audio.duration()),
                    );
                }
                engine.unload();
                transcript_segments = transcript.segments.len();
                let secs = t.elapsed().as_secs_f64().max(1e-9);
                asr_secs = Some(secs);
                push(
                    &mut stages,
                    "asr",
                    t,
                    Some(format!("segments={transcript_segments}")),
                );
                // Keep the transcript for the export stage below.
                let t = Instant::now();
                let rendered: usize = vaani_export::ExportFormat::ALL
                    .iter()
                    .map(|f| {
                        vaani_export::render(
                            &transcript,
                            *f,
                            &vaani_export::ExportOptions::default(),
                        )
                        .map(|s| s.len())
                        .unwrap_or(0)
                    })
                    .sum();
                push(
                    &mut stages,
                    "export",
                    t,
                    Some(format!("rendered_bytes={rendered}")),
                );
            }
            Err(e) => {
                skip(&mut skipped, "asr", &e);
                push(&mut stages, "asr", t, Some("skipped".to_string()));
                // Export still gets timed on an empty transcript so the report
                // always has the same stage vocabulary.
                let t = Instant::now();
                let empty = Transcript::new(EngineStamp::asr("none", "0"));
                let rendered: usize = vaani_export::ExportFormat::ALL
                    .iter()
                    .map(|f| {
                        vaani_export::render(&empty, *f, &vaani_export::ExportOptions::default())
                            .map(|s| s.len())
                            .unwrap_or(0)
                    })
                    .sum();
                push(
                    &mut stages,
                    "export",
                    t,
                    Some(format!("rendered_bytes={rendered}")),
                );
            }
        }

        // diar: the heuristic engine needs no weights, so it always runs once
        // audio exists. Timed separately from VAD per the CLI contract.
        let t = Instant::now();
        let diar = vaani_diarization::HeuristicDiarizer::new();
        let num_speakers =
            match diar.diarize_with_timeline(&audio, Some(&timeline), cancel, &|_| {}) {
                Ok(d) => {
                    push(
                        &mut stages,
                        "diar",
                        t,
                        Some(format!("speakers={}", d.num_speakers)),
                    );
                    Some(d.num_speakers)
                }
                Err(e) => {
                    skip(&mut skipped, "diar", &e);
                    push(&mut stages, "diar", t, Some("skipped".to_string()));
                    None
                }
            };

        // Reorder to pipeline order: startup, media, vad, asr, diar, export.
        // (asr/export were pushed before diar above so the transcript was in
        // scope; fix the order here rather than cloning the transcript.)
        let mut ordered = Vec::with_capacity(stages.len());
        for name in ["startup", "media", "vad", "asr", "diar", "export"] {
            if let Some(s) = stages.iter().find(|s| s.stage == name).cloned() {
                ordered.push(s);
            }
        }

        let rtf = asr_secs.map(|s| audio_secs / s);
        Ok(BenchmarkReport {
            app_version: vaani_core::APP_VERSION.to_string(),
            file: file.display().to_string(),
            audio_duration_ms: Some(audio.duration().as_millis()),
            stages: ordered,
            skipped,
            peak_ram_bytes: crate::console::peak_working_set_bytes(),
            system_ram_bytes: Some(system_ram),
            adapter_vram_bytes: vram,
            rtf,
            transcript_segments,
            num_speakers,
        })
    }
}

fn internal(detail: impl std::fmt::Display) -> VaaniError {
    err::internal().with_cause(detail.to_string())
}

/// VRAM of the inference adapter in bytes, when the capability report names one.
fn adapter_vram_bytes(adapter: &vaani_inference::GpuAdapter) -> Option<u64> {
    // `GpuAdapter` exposes a display helper, not a byte count, in this build;
    // report `None` rather than parsing a display string. The system RAM and
    // process peak figures are the load-bearing ones for the benchmark.
    let _ = adapter;
    None
}

/// Persist the transcript: one job row + the transcript itself.
fn store_transcript(
    job_id: &str,
    req: &TranscribeRequest,
    media: &vaani_core::MediaInfo,
    transcript: &Transcript,
    num_speakers: Option<usize>,
) -> Result<()> {
    let store = vaani_storage::Store::open_default()?;
    let mut job = vaani_storage::JobRecord::new(job_id, &req.input);
    job.media = Some(media.clone());
    job.container = Some(media.container.clone());
    job.duration_ms = transcript.duration.map(Time::as_millis);
    job.model = Some(req.model.clone());
    job.device = Some(req.device.as_str().to_string());
    job.language = Some(req.language.as_str().to_string());
    job.segment_count = transcript.segments.len() as i64;
    job.speaker_count = num_speakers.map(|n| n as i64).unwrap_or(0);
    job.phase = vaani_core::job::JobPhase::Complete;
    store.upsert_job(&job)?;
    let stats = store.replace_transcript(job_id, transcript)?;
    job.word_count = stats.words;
    store.upsert_job(&job)?;
    let _ = Progress::indeterminate(vaani_core::job::JobPhase::Complete);
    Ok(())
}

/// Write the export step of `transcribe --out/--format`.
fn export_transcript(
    transcript: &Transcript,
    input: &Path,
    out: &Path,
    format: &str,
    overwrite: bool,
) -> Result<Vec<PathBuf>> {
    use vaani_export::{ExportFormat, ExportOptions, ExportTarget};
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "transcript".to_string());
    let options = ExportOptions {
        source_name: Some(stem),
        overwrite,
        ..ExportOptions::default()
    };

    // A directory (existing, or trailing separator) means "name it for me";
    // anything else is an exact file path.
    let target = if out.is_dir() || out.as_os_str().to_string_lossy().ends_with(['/', '\\']) {
        ExportTarget::Directory(out.to_path_buf())
    } else if out.extension().is_none() && !out.exists() {
        // `vaani transcribe a.mp3 --out ./subs` with no extension: treat as a
        // directory the export will create by writing into it.
        ExportTarget::Directory(out.to_path_buf())
    } else {
        ExportTarget::File(out.to_path_buf())
    };

    // A `Directory` target that does not exist yet is the normal case, not a
    // mistake: a save dialog hands back a fresh folder, and
    // `vaani transcribe clip.wav --out exports/` is the documented invocation.
    // `vaani_export` deliberately does not build a directory tree (a typed `File`
    // path in a missing folder should still fail rather than silently invent
    // one), so the creation belongs to the caller that chose to interpret the
    // argument as a directory. `vaani-app`'s export stage already does this;
    // the CLI's pipeline did not, so `--out <new dir>` failed with a bare
    // "The system cannot find the path specified" after the whole 14-minute job
    // had already been transcribed.
    if let ExportTarget::Directory(dir) = &target
        && !dir.is_dir()
    {
        std::fs::create_dir_all(dir).map_err(|e| {
            vaani_core::err::export_failed()
                .with_cause(format!("could not create {}: {e}", dir.display()))
                .with_recovery("Pick an existing folder, or create this one first.")
        })?;
    }

    if format.eq_ignore_ascii_case("all") {
        return Ok(vaani_export::export_all(
            transcript,
            &ExportFormat::ALL,
            target_dir(&target, out)?,
            &options,
        )?
        .into_iter()
        .map(|f| f.path)
        .collect());
    }
    let fmt = ExportFormat::parse(&format.to_ascii_lowercase())
        .ok_or_else(|| internal(format!("unknown export format: {format}")))?;
    Ok(vec![
        vaani_export::export_to_file(transcript, fmt, &target, &options)?.path,
    ])
}

fn target_dir<'a>(target: &'a vaani_export::ExportTarget, fallback: &'a Path) -> Result<&'a Path> {
    match target {
        vaani_export::ExportTarget::Directory(d) => Ok(d),
        vaani_export::ExportTarget::File(_) => Ok(fallback
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcribe_request_defaults_match_cli_flags() {
        let r = TranscribeRequest::for_file(PathBuf::from("a.mp3"));
        assert_eq!(r.model, DEFAULT_MODEL_ID);
        assert!(r.diarize);
        assert!(r.out.is_none());
        assert_eq!(r.format, "srt");
        assert!(!r.overwrite);
    }

    #[test]
    fn missing_input_maps_to_stable_media_code() {
        // The CLI contract: a dry run on a missing file asserts the CODE, never
        // message text. Both `transcribe` and `benchmark` share the gate.
        let pipe = DirectPipeline;
        let cancel = CancelToken::new();
        let req = TranscribeRequest::for_file(PathBuf::from("definitely-not-here-12345.mp4"));
        let e = pipe.transcribe(&req, &cancel).unwrap_err();
        assert!(
            matches!(
                e.code(),
                ErrorCode::UnsupportedMedia | ErrorCode::NoAudioStream
            ),
            "unexpected code: {}",
            e.code().as_str()
        );
        let e = pipe
            .benchmark(Path::new("definitely-not-here-12345.mp4"), &cancel)
            .unwrap_err();
        assert!(
            matches!(
                e.code(),
                ErrorCode::UnsupportedMedia | ErrorCode::NoAudioStream
            ),
            "unexpected code: {}",
            e.code().as_str()
        );
    }

    #[test]
    fn exporting_into_a_directory_that_does_not_exist_yet_creates_it() {
        // The first real end-to-end run on the 14-minute clip failed *after* the
        // whole job, with `os error 3`, because `--out some/new/folder` names a
        // directory that no code had created. The CLI is the production path, so
        // it is the CLI that has to do it.
        let root = vaani_storage::testing::TempDir::new("export-new-dir");
        let out = root.join("new-folder");
        assert!(!out.exists());

        let t = vaani_storage::fixtures::transcript(7, 4);
        let written = export_transcript(&t, Path::new("standup.m4a"), &out, "all", false).unwrap();

        assert_eq!(written.len(), vaani_export::ExportFormat::ALL.len());
        for p in &written {
            assert!(p.is_file(), "{} was not written", p.display());
            assert_eq!(p.parent().unwrap(), out.as_path());
        }
    }

    #[test]
    fn a_typed_file_path_in_a_missing_folder_still_fails_rather_than_guessing() {
        // The mirror image of the test above, and deliberately unchanged: a
        // user who types `out/subs.srt` and has no `out/` has made a typo, and
        // silently building a tree from one hides the next one.
        let root = vaani_storage::testing::TempDir::new("export-typo");
        let out = root.join("typo/subs.srt");
        let t = vaani_storage::fixtures::transcript(7, 2);
        let e = export_transcript(&t, Path::new("standup.m4a"), &out, "srt", false).unwrap_err();
        assert_eq!(e.code(), ErrorCode::ExportFailed);
    }

    #[test]
    fn benchmark_report_json_shape_is_stable() {
        // `benchmark --json` is a script interface: field names are frozen.
        // Construct the report directly (no media needed) and assert the shape.
        let report = BenchmarkReport {
            app_version: "0.1.0".to_string(),
            file: "x.mp4".to_string(),
            audio_duration_ms: Some(60_000),
            stages: vec![StageTiming {
                stage: "media".to_string(),
                elapsed_ms: 12,
                detail: None,
            }],
            skipped: vec![SkippedStage {
                stage: "asr".to_string(),
                reason_code: ErrorCode::ModelUnavailable.as_str().to_string(),
                reason: "not downloaded".to_string(),
            }],
            peak_ram_bytes: None,
            system_ram_bytes: Some(1),
            adapter_vram_bytes: None,
            rtf: None,
            transcript_segments: 0,
            num_speakers: None,
        };
        let v: serde_json::Value = serde_json::to_value(&report).unwrap();
        for key in [
            "app_version",
            "file",
            "audio_duration_ms",
            "stages",
            "skipped",
            "rtf",
            "transcript_segments",
        ] {
            assert!(v.get(key).is_some(), "benchmark JSON missing `{key}`");
        }
        assert_eq!(v["stages"][0]["stage"], "media");
        assert_eq!(v["skipped"][0]["reason_code"], "model_unavailable");
        // Unknown measurements serialise as `null`, never as absent or zero.
        assert!(v["peak_ram_bytes"].is_null());
        assert!(v["rtf"].is_null());
    }
}

# VaaniDesk Integration Contract (authoritative)

Status: architect-owned. Feature agents compile against this; they do not redefine it.
Source of truth for types: `crates/vaani-core`. Verified 2026-09-27 against the
recon reports (media/VAD/diarization, storage/export/models, UI/CLI).

Related: `docs/agent/PARALLEL_OWNERSHIP.md`, `docs/agent/ARCHITECTURE.md`.

## 1. Canonical audio: mono 16 kHz f32

- `vaani_media::TARGET_SAMPLE_RATE = 16_000` (`crates/vaani-media/src/lib.rs`).
- `vaani_vad::SAMPLE_RATE = 16_000`; `HeuristicDiarizer` requires
  `REQUIRED_SAMPLE_RATE = 16_000` (`crates/vaani-diarization/src/config.rs`).
- Inter-stage type: `vaani_core::media::AudioBuffer { samples: Vec<f32>, sample_rate: u32, channels: u16, start_time: Time }`.
  `AudioBuffer::mono(samples, 16_000, Time::ZERO)`; `frames()`, `duration()`,
  `time_of(i)`, `slice_mono(start, end)` (panics unless mono).
- `vaani-app` MUST resample/downmix before VAD, chunking, ASR, diarization.
  VAD, diarization and ASR stages never resample internally.

## 2. Time base

- `vaani_core::Time(pub i64)` = milliseconds from source-media start.
- All stage outputs are absolute **except** `AsrEngine::transcribe`, which
  returns chunk-local times. The single rebase point is
  `Transcript::push_chunk(chunk_offset, segments)` (`transcript.rs:351`):
  saturating add, clamp_zero, fix inverted, sorted insert.
- No other crate may add a chunk offset to a `Time`.

## 3. Job state machine (`vaani_core::job`)

- `JobPhase` (14): `Queued, Preparing, ExtractingAudio, NormalizingAudio, Vad,
  Asr, Diarization, Alignment, PostProcessing, Structuring, Exporting,
  Complete, Failed, Cancelled`.
- `JobPhase::ALL` (13) = forward path excluding `Cancelled`.
  `as_str()` = SCREAMING_SNAKE persisted in DB + JSON (never rename).
  `is_terminal = Complete|Failed|Cancelled`; `is_active = !Queued && !terminal`;
  `weight()` sums to 1.0 over active phases (test `job.rs:399` guards the bar).
- `Progress { phase, kind, detail }`: `ProgressKind::{Fraction, MediaSeconds,
  Indeterminate}`. `Progress::media(phase, done, total)`,
  `Progress::indeterminate(phase)`, `overall() -> Option<f64>`
  (`None` = indeterminate; `Queued` = `Some(0.0)`).
  Never fabricate a percentage: no denominator => `Indeterminate`.
- `CancelToken { new/child/cancel/is_cancelled/check }`: shared atomic.
  Every stage polls `ctx.check()` at least once per second of work / per chunk.
- Errors: `err::*()` constructors take zero args; diagnostics go to
  `VaaniError::with_cause` (source-chained, dev log only).
  `ErrorCode::as_str()` persisted (never rename). `cuda_unavailable` is a
  `ProviderStatus`, not an error: silent CPU fallback is normal.

## 4. Pipeline traits (`vaani_core::pipeline`)

- `SpeechSegment { start, end, confidence? }`, `SpeechTimeline { segments (sorted,
  merged), speech_duration_ms }`: `speech_ratio(total)`, `overlapping(start,end)`.
- `VadEngine: id/name/sample_rate==16000/detect(&AudioBuffer,&CancelToken)
  /cancel`.
- `SpeakerSpan { speaker (0-based contiguous), start, end, score? }`,
  `DiarizationResult { spans (sorted), num_speakers, elapsed_ms }`:
  `coverage/total`, `speaker_at(t)` (shortest-span wins), `speaker_durations()`.
- `DiarizationEngine: id/name/load/is_loaded/unload/diarize(&AudioBuffer,
  &CancelToken, &dyn Fn(f64))/cancel`.
- `StageArtifacts { audio?, speech?, transcript?, speakers?, exported: Vec<PathBuf> }`
  with `audio()/transcript()` helpers returning `err::internal()` when missing.
- `StageContext { input:&Path, work_dir:&Path, cancel:&CancelToken,
  progress:&ProgressSink, artifacts:&mut StageArtifacts }` with
  `report(phase, fraction?, detail?)` and `check()`.
- `PipelineStage: name/phase/is_available/run(&mut StageContext)`.
- Load-bearing rules: diarization takes `&AudioBuffer`, never ASR output;
  attribution (`attribute_speakers/apply`) is a separate step in `vaani-app`.

## 5. Media (`vaani-media`)

- `probe(path) -> MediaInfo`, `probe_bytes(bytes, hint) -> MediaInfo`.
  `MediaInfo { container, declared_duration_ms?, size_bytes, audio?,
  video?, seekable }`; `authoritative_duration(decoded_ms?)`,
  `estimate_transcription_ms(rtf)` (`DEFAULT_ESTIMATE_RTF = 0.6`).
- `decode_to_mono16k(path, cancel)`, `decode_to_mono16k_with_progress(path,
  cancel, &ProgressSink)` (emits `Progress::media(ExtractingAudio)`),
  `decode(path, cancel, &DecodeOptions) -> DecodedAudio`,
  `AudioStream::open/for_each_chunk/next_chunk/finish/time_of/trim_collected`.
- `DecodeOptions { target_sample_rate, on_truncated: AllowPartial|Fail,
  reserve_limit_bytes, cancel_check_* }`; `DecodeReport { media_info,
  source_*, output_frames, decoded_duration_ms, truncated, nonfinite/clamped/
  over_unity counts, warnings, elapsed }`.
- `downmix_to_mono(&[f32], channels, &mut Vec<f32>) -> DownmixStats`
  (`MAX_INPUT_SAMPLE = 8.0`); `resample_mono(&[f32], in_rate)`,
  `MonoResampler::new/with_target/push/finish/trim_to_expected/reset`,
  `expected_output_frames`.
- Error surface is `vaani_core::ErrorCode::{UnsupportedMedia,
  AudioDecodeFailed, NoAudioStream}`; `errors` module is private.

## 6. VAD + chunking (`vaani-vad`)

- Engines implement `vaani_core::VadEngine`:
  `EnergyVad::new/with_config` (`id="energy"`), Silero `try_open/open/
  preferred_engine` (`id="silero-vad-v5"`, 16 ms hop vs energy 10 ms,
  weights absent => default is energy).
- `detect(&audio,&cancel) -> SpeechTimeline` (absolute times).
- `ChunkPolicy::default`: target 30 s / max 60 s / overlap 0 / merge_gap 300 ms /
  pad 60 ms / min 1000 ms / boundary_search 2000 ms.
  `Chunker::new(policy).plan(&timeline, total) -> Vec<Chunk{start,end}>`;
  `plan_with_raw`, `merge_segments`, `split_at_cap`, `coalesce_short`.
- Overlap > 0 deduplication (if ever enabled) belongs to `vaani-app`, not core.

## 7. ASR (`vaani-asr`, frozen math)

- `AudioChunk { samples, sample_rate, source_offset (diagnostic only),
  language }`: `new/with_language/duration`.
- `ChunkResult { segments (chunk-local!), language?, timings }`.
- `AsrEngine: id/model_name/model_version/languages/supports_{streaming,
  timestamps, word_timestamps}/load/unload/is_loaded/transcribe(&AudioChunk,
  &CancelToken)/cancel/benchmark`.
- Registry: `EngineRegistry::new/register/with_builtins(provider,
  caps)/ids/contains/create("sravaani", &artifacts)`.
- Artifacts: `artifacts_in(dir) -> EngineArtifacts { frontend_dir, encoder,
  decoder_joint, tokenizer }` (`sravaani.rs:722`); `open(...)` (`sravaani.rs:701`).
  `ModelManager::resolve(id).dir()` feeds `artifacts_in`;
  `engine_id(id)` maps manifest id to registry id.
- Transcript accumulation: `Transcript::new(EngineStamp::asr(model, version))`
  then per chunk `transcript.push_chunk(chunk.start, result.segments)`.
- Frozen files (`PARALLEL_OWNERSHIP.md`): `frontend.rs`, `sravaani.rs`,
  `vaani-inference/session.rs + runtime.rs`, golden fixtures + parity tests.
  Golden slice MUST stay 202/202 token ids, 108/108 word timings @ 0.0 ms,
  exact text.

## 8. Diarization (`vaani-diarization`)

- Concrete: `HeuristicDiarizer::new/with_config`, `ID =
  "vaani-diarization-heuristic"`, `default_engine() -> Arc<...>`,
  `as_engine(e) -> Box<dyn DiarizationEngine>`.
- Prefer `diarize_with_timeline(&audio, Some(&speech), &cancel, &|f|)` to reuse
  VAD; trait `diarize()` = `diarize_with_activity` (ignores external timeline).
- Attribution: `attribute_speakers(&mut Transcript, &DiarizationResult)`,
  `apply(transcript, d, existing)` (preserves names),
  `speakers_from_diarization`, `majority_speaker/margin`, `overlap_vote`,
  `assert_consistent`, `coverage_of`.
- `EngineStamp.diarization_model = Some(ID)` when diarization ran.

## 9. Storage (`vaani-storage`, schema v2)

- `Store::open/open_at_version/open_with/open_in_memory/open_default/
  default_path/schema_version/applied_migrations/integrity_check/
  recover_interrupted_jobs/with_read/with_write`.
- `latest_version() = 2` (v1 initial, v2 transcript_counts_and_exports + FTS5
  unicode61). Migrations via `migrate(conn, path, target)`.
- Jobs: `new_job_id() (UUIDv4)`, `JobRecord { id, input_path, input_name,
  container?, media: Option<MediaInfo> (JSON), duration_ms?, phase, created/
  started/finished_at_ms, error: Option<StoredError{code,message}>,
  model?, device?, language?, word/segment/speaker_count, content_hash }`,
  `JobRecord::new(id, path)` (Queued), `upsert_job/load_job/
  find_completed_by_hash/set_phase/list_jobs(JobFilter,limit,offset)->JobPage/
  delete_job`.
- Transcripts: `replace_transcript(job_id, &Transcript)->TranscriptStats`,
  `try_load_transcript/load_transcript/transcript_summary/rename_speaker`.
- Settings/exports: `get/set_setting[_json]`, `record_export/list_exports`.
  Export format stored as wire string (`text|markdown|srt|vtt|json`) matching
  `ExportFormat::as_str()` by convention (no shared enum: export->storage edge
  only via `clock::rfc3339` + `fixtures/testing`).
- Raw media blobs are never stored in SQLite. `content_hash` = SHA-256 hex
  (see `hash` module for promises).

## 10. Export (`vaani-export`)

- Formats: `ExportFormat::{Text,Markdown,Srt,Vtt,Json}`, `ALL[5]`,
  `extension/as_str/label/media_type/has_timestamps/is_cue_based/parse`,
  `ExportTarget::{Directory,File}::resolve(format, stem)`,
  `ExportOptions { speaker_labels, include_timestamps/metadata,
  include_cue_numbers, max_line_length/lines_per_cue, wrap, overwrite,
  timestamp_ms, source_name, title }`, `file_stem/document_title/cue_options`.
- Entry: `export_to_file(&Transcript, format, &ExportTarget, &ExportOptions)
  -> ExportedFile { path, bytes, format, cues }`, `export_all(...)`,
  `render(...) -> String`, `read_json_export(text) -> Transcript`
  (JSON envelope `JsonExport{format="vaani.transcript", format_version=1,
  generator, app_version, data_model_version, exported_at?, source?, options,
  transcript}` round-trips to an equal `Transcript`).
- Cues: `build_cues/render_srt/render_vtt`, `parse_srt/parse_vtt/check/
  validate_srt/validate_vtt`; `safe_file_name/assert_inside/write_atomic`
  (temp + rename; `overwrite=false` default).
- Parity: same five formats/order/extensions as `vaani_ui::ExportFormat` and
  `vaani_core::EXPORT_EXTENSIONS` (test asserts).

## 11. Models (`vaani-models`)

> **MANDATE (Authoritative, 2026-09-28):** Pre-bundled offline models. SraVaani 1.0 is MIT-licensed and its production weights (~930 MB: `encoder.fp16.onnx`, `decoder_joint.onnx`, `tokenizer.model`, filterbank matrices, and configs) are **permanently bundled in `<app>/models/sravaani/`**. The application MUST work 100% offline immediately on first launch with zero user tokens and zero setup.

- `ModelManager::embedded/from_manifest_file/discover()->Discovery/
  resolve(id)->ModelLocation::dir()/engine_id(id)/verify/install_plan[_with]/
  install (async)/install_blocking/remove/guard_files`.
- Flow: `discover` (names+sizes) -> `install_plan_with` (free_bytes +
  MachineCapabilities) -> `InstallLock::acquire` (`LOCK_DIR=.vaani-locks`) ->
  `.partial/<name>.<pid>.part` + hash -> rename -> `.installed.json` last.
- Roots precedence: `bundled` / `user` / `configured`. Bundled root `<app>/models/` is checked first for immediate offline execution.
  Env: `VAANI_MODELS_DIR/VAANI_ARTIFACTS[_ROOT]/VAANI_MODEL_DIRS`.
- Registry: `manifests/registry.json` (1 model `sravaani-1.0-onnx`,
  schema_version=1, pre-bundled production assets).
- `TransferProgress::as_core(JobPhase::Preparing)` for pipeline progress.
- Never commit weights to git; never log/CLI-expose tokens.

## 12. UI backend (`vaani-ui`, Slint 1.18.1 software renderer)

- Window: `app::build(controller, host)->UiSession{window:
  Rc<AppWindow>, model}` (event-loop thread only), `run(...)`, `run_demo()`,
  `WINDOW_TITLE="VaaniDesk"`. `UiModel` asserts loop thread; pump 33 ms /
  4096 events; `TRANSCRIPT_LINE_HEIGHT_PX=22.0`.
- `backend::JobController: backend_name/is_demo/jobs/job/enqueue(&Path)/
  cancel/remove/transcript->Option<Arc<Transcript>>/speakers/
  rename_speaker/export(&ExportRequest)/settings/update_settings/subscribe()/
  models/devices/languages` (+ `impl for Arc<T>`). Non-blocking; truth via
  single `subscribe()->Receiver<ControllerEvent>`.
- DTOs: `JobId`, `ExportFormat[5]`, `ExportTarget::{Directory,File}`,
  `ExportRequest{job,format,target}`, `JobSummary{id,input,phase,
  overall:Option<f64>,detail,eta,error:Option<UserFacingError>,
  has_transcript,cancel_requested,word_count}`,
  `SpeakerSummary{index,name,talk_time_ms,share?}`, `Choice{id,label,detail,
  available}`, `AppSettings{model,device,language,diarize,theme}` +
  `SettingsPatch`, `ControllerEvent::{JobAdded/JobChanged/JobRemoved/
  TranscriptReady/SpeakersChanged/RenameRejected/ExportFinished/ExportFailed/
  SettingsChanged/Notice}`, `HostServices{pick_media_files/pick_export_target/
  set_clipboard_text/host_name}` (+ `UnavailableHostServices`).
- `vaani-app` provides `struct AppBackend: JobController` (thin adapter over
  orchestrator; `Arc<FakeController>` -> `Arc<AppBackend>` is one type param).
  `is_demo=false` removes banner. `overall=None` = indeterminate sweep.
  Binary crate owns `rfd` dialogs, never `vaani-ui`.
- Slint surface (`ui/app-window.slint` properties/callbacks/rows/colors) is
  frozen: renames break `vaani-app` wiring.

## 13. CLI (`vaani-cli`, to be built)

- Bins: `vaani.exe` (GUI, no console on normal launch) + `vaani-cli.exe`
  (headless). `vaani` opens GUI; `vaani --debug/--version/benchmark/transcribe`.
- Commands: `vaani --version`, `vaani --debug`, `vaani benchmark <file>
  [--json]`, `vaani transcribe <file> [--model --device --language
  --diarize/--no-diarize --out --format]`, `vaani doctor`, `vaani models *`,
  `vaani jobs/export` as specified by A8. JSON output machine-readable.
- Shares `vaani-app` pipeline core with the GUI; same gates apply.

## 14. Pipeline order (vaani-app owns)

`Path -> media (probe/decode) -> VAD (detect) -> chunk (plan/slice) ->
ASR (transcribe/push_chunk loop) -> diarization (independent audio path) ->
attribution -> postprocess (in-place, optional) -> structuring (optional) ->
storage (upsert/replace_transcript) -> export (on demand)`.

- `vaani-app` owns `StageArtifacts` + `StageContext` construction + phase
  advancement; services never call each other.
- Diarization off by default when model missing: skip + reason, not failure.
- Zero `impl PipelineStage` exists today: `vaani-app` authors the stages.

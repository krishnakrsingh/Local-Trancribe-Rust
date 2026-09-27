//! The end-to-end acceptance gate: the real clip, the real model, no fakes.
//!
//! # Why this file exists
//!
//! Every stage in this project had been measured on the real meeting
//! recording *individually*. Nobody had ever run the whole chain on it, and a
//! chain that has never run together fails in the seams: a chunk plan that
//! hands the encoder 40 ms of audio, a rebasing point that is fed chunk-local
//! times twice, a speaker index that resolves to nothing. None of those show up
//! in a per-stage RTF, and every one of them produces output that *looks*
//! fine.
//!
//! So this test asserts the properties that only a full run can produce, and it
//! asserts them against the real `ModelManager`, the real `EngineRegistry` and
//! the real CUDA provider. It is deliberately an `#[ignore]`d integration test
//! rather than a unit test: it needs ~2.6 GB of weights and a multi-minute
//! recording, so it cannot live in the `cargo test --workspace` gate, but it
//! must not be a PowerShell script either, because the interesting assertions
//! need the library types.
//!
//! Run it with `scripts/e2e/acceptance.ps1`, which is the actual gate: it maps
//! a skip to a distinct exit code and never lets a skip look like a pass.
//!
//! # What it asserts
//!
//! | check | what a failure means |
//! |---|---|
//! | `word_timings_are_monotonic_and_in_bounds` | the classic chunk-rebasing bug: a word ends before it starts, or lands outside the media |
//! | `chunks_are_rebased_onto_the_media_timeline` | the transcript is a constant or chunk-length offset from the video |
//! | `segments_land_inside_detected_speech` | times are plausible-looking but not anchored to the audio |
//! | `speaker_attribution_is_consistent` | diarization is not actually running, or a segment names a speaker that does not exist |
//! | `the_stored_transcript_round_trips` | persistence loses a `Time` or a segment id |
//! | `every_export_format_is_valid` | the bytes on disk are not a subtitle file |
//!
//! # Privacy
//!
//! `scratch/clip.wav` is a real private meeting. Nothing in this file, and
//! nothing this test writes, may contain what was said in it. Every assertion
//! is on counts, durations, timings and structure; the report is a table of
//! aggregates. A failure message names a segment *index*, never a word.
//!
//! # Skipping
//!
//! Absent weights or absent clip, this prints one loud `SKIP` line naming what
//! is missing and returns. It never fails silently and it never passes
//! silently: the caller (`scripts/e2e/acceptance.ps1`) treats the skip marker
//! as its own exit code.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use vaani_cli::pipeline::Pipeline;
use vaani_core::pipeline::VadEngine;
use vaani_core::{CancelToken, EngineStamp, Language, Time, Transcript};
use vaani_inference::Provider;

/// Where the clip is, unless `VAANI_E2E_CLIP` says otherwise.
const DEFAULT_CLIP: &str = "../scratch/clip.wav";

/// The manifest id the CLI defaults to.
const MODEL_ID: &str = "sravaani-1.0-onnx";

/// The one long test. Everything below is a helper.
#[test]
#[ignore = "needs the real SraVaani weights and a real recording; run scripts/e2e/acceptance.ps1"]
fn the_real_meeting_survives_the_whole_chain() {
    // ---------------------------------------------------------------- preflight
    let Some(clip) = locate_clip() else {
        skip("no recording: set VAANI_E2E_CLIP or put a file at ../scratch/clip.wav");
    };
    if let Some(why) = runtime_missing() {
        skip(&why);
    }
    if let Some(why) = model_missing() {
        skip(&why);
    }

    let out_dir = output_dir();
    std::fs::create_dir_all(&out_dir).expect("create the acceptance output directory");
    let report_path = out_dir.join("acceptance.json");

    // ------------------------------------------------------------------ decode
    let cancel = CancelToken::new();
    let t0 = Instant::now();
    let media = vaani_media::probe(&clip).expect("probe the recording");
    let audio = vaani_media::decode_to_mono16k(&clip, &cancel).expect("decode the recording");
    let media_ms = t0.elapsed().as_millis() as u64;
    let total = audio.duration();
    let total_ms = total.as_millis();
    assert!(
        total_ms > 60_000,
        "the acceptance recording should be minutes long, got {total_ms} ms"
    );

    // --------------------------------------------------------------------- VAD
    let t0 = Instant::now();
    let vad = vaani_vad::EnergyVad::new();
    let timeline = vad.detect(&audio, &cancel).expect("run the VAD");
    let vad_ms = t0.elapsed().as_millis() as u64;

    let chunker = vaani_vad::Chunker::new(vaani_vad::ChunkPolicy::default());
    let chunks = chunker.plan(&timeline, total);
    assert!(!chunks.is_empty(), "the chunk plan is empty");
    for c in &chunks {
        assert!(
            c.end > c.start,
            "a planned chunk is not a range: {}..{}",
            c.start.as_millis(),
            c.end.as_millis()
        );
    }

    // --------------------------------------------------------------------- ASR
    let caps = vaani_inference::RuntimeReport::collect().machine;
    let provider = Provider::Auto;
    let t0 = Instant::now();
    let engine = open_engine(&caps, provider).expect("open the real SraVaani engine");
    engine.load(&cancel).expect("load the real weights");
    let asr_ms = t0.elapsed().as_millis() as u64;

    let mut transcript = Transcript::new(EngineStamp::asr(
        engine.model_name(),
        engine.model_version(),
    ));
    transcript.duration = Some(total);

    // The ground truth for the rebasing check: every chunk-local segment the
    // engine produced, together with the offset it *should* land at. This is
    // read off the engine's raw output, never off the transcript, so asserting
    // against it is not circular.
    let mut expected: Vec<(Time, Time)> = Vec::new();
    let mut timings = vaani_asr::AsrTimings::default();
    let mut asr_loop_ms = 0u64;

    for chunk in &chunks {
        cancel.check().expect("not cancelled");
        let samples = audio.slice_mono(chunk.start, chunk.end);
        let asr_chunk =
            vaani_asr::AudioChunk::new(samples, 16_000, chunk.start).with_language(Language::Auto);
        let t0 = Instant::now();
        let result = engine
            .transcribe(&asr_chunk, &cancel)
            .expect("transcribe a chunk");
        asr_loop_ms += t0.elapsed().as_millis() as u64;
        timings.frontend_ms += result.timings.frontend_ms;
        timings.encode_ms += result.timings.encode_ms;
        timings.decode_ms += result.timings.decode_ms;
        timings.encoder_frames += result.timings.encoder_frames;
        timings.decode_steps += result.timings.decode_steps;
        for s in &result.segments {
            expected.push((
                s.start.saturating_add(chunk.start).clamp_zero(),
                s.end.saturating_add(chunk.start).clamp_zero(),
            ));
        }
        transcript.push_chunk(chunk.start, result.segments);
    }
    engine.unload();

    // -------------------------------------------------------------- diarization
    let t0 = Instant::now();
    let diarizer = vaani_diarization::HeuristicDiarizer::new();
    let diar = diarizer
        .diarize_with_timeline(&audio, Some(&timeline), &cancel, &|_| {})
        .expect("diarize the recording");
    let diar_ms = t0.elapsed().as_millis() as u64;
    let attribution = vaani_diarization::attribute_speakers(&mut transcript, &diar);
    transcript.engine.diarization_model =
        Some(vaani_diarization::HeuristicDiarizer::ID.to_string());

    // ----------------------------------------------------------------- storage
    let t0 = Instant::now();
    let store = vaani_storage::Store::open_in_memory().expect("open a store");
    let job_id = vaani_storage::new_job_id();
    let mut job = vaani_storage::JobRecord::new(&job_id, &clip);
    job.duration_ms = Some(total_ms);
    job.segment_count = transcript.segments.len() as i64;
    job.speaker_count = diar.num_speakers as i64;
    job.phase = vaani_core::job::JobPhase::Complete;
    store.upsert_job(&job).expect("record the job");
    let stats = store
        .replace_transcript(&job_id, &transcript)
        .expect("persist the transcript");
    let store_ms = t0.elapsed().as_millis() as u64;
    let read_back = store
        .load_transcript(&job_id)
        .expect("read the transcript back");

    // ------------------------------------------------------------------ export
    let t0 = Instant::now();
    // Overwrite: the gate owns its output directory, and a gate that cannot be
    // re-run without deleting files by hand is a gate people stop re-running.
    let export_options = vaani_export::ExportOptions::default()
        .with_source_name(media_basename(&clip))
        .with_overwrite(true);
    let written = vaani_export::export_all(
        &transcript,
        &vaani_export::ExportFormat::ALL,
        &out_dir,
        &export_options,
    )
    .expect("export every format");
    let export_ms = t0.elapsed().as_millis() as u64;

    // ------------------------------------------------------------------ assert
    let words: usize = transcript.segments.iter().map(|s| s.words.len()).sum();

    let (past_segments, past_worst) = word_timings_are_monotonic_and_in_bounds(&transcript, total);
    chunks_are_rebased_onto_the_media_timeline(&transcript, &expected, &chunks);
    segments_land_inside_detected_speech(&transcript, &timeline);
    speaker_attribution_is_consistent(&transcript, &attribution, &diar);
    the_stored_transcript_round_trips(&transcript, &read_back);
    every_export_format_is_valid(&written, &transcript);

    // ------------------------------------------------------------------ report
    let peak_rss = vaani_cli::console::peak_working_set_bytes();
    let total_ms_measured =
        media_ms + vad_ms + asr_ms + asr_loop_ms + diar_ms + store_ms + export_ms;
    let secs = total_ms as f64 / 1000.0;
    let rtf = |ms: u64| {
        if ms == 0 {
            0.0
        } else {
            secs / (ms as f64 / 1000.0)
        }
    };

    println!("\n=== VaaniDesk end-to-end acceptance ===");
    println!("  recording            {} s", total_ms as f64 / 1000.0);
    println!("  container            {:?}", media.container);
    println!("  vad spans            {}", timeline.segments.len());
    println!(
        "  vad speech           {:.1} % of the recording",
        timeline.speech_ratio(total) * 100.0
    );
    println!("  chunks               {}", chunks.len());
    println!("  segments             {}", transcript.segments.len());
    println!("  words                {words}");
    println!("  speakers             {}", diar.num_speakers);
    println!(
        "  attributed           {:.1} % of segments",
        attribution.attributed_ratio() * 100.0
    );
    println!(
        "  store words / segments  {} / {}",
        stats.words, stats.segments
    );
    println!("\n  stage            wall ms   RTF");
    println!("  media            {media_ms:>8}  {:>7.4}", rtf(media_ms));
    println!("  vad              {vad_ms:>8}  {:>7.4}", rtf(vad_ms));
    println!("  asr (open+load)  {asr_ms:>8}");
    println!(
        "  asr (transcribe) {asr_loop_ms:>8}  {:>7.4}",
        rtf(asr_loop_ms)
    );
    println!("  diarization      {diar_ms:>8}  {:>7.4}", rtf(diar_ms));
    println!("  storage          {store_ms:>8}  {:>7.4}", rtf(store_ms));
    println!("  export (5 files) {export_ms:>8}  {:>7.4}", rtf(export_ms));
    println!("  ---------------------------");
    println!(
        "  total            {total_ms_measured:>8}  {:>7.4}",
        rtf(total_ms_measured)
    );
    println!(
        "\n  asr frontend/encode/decode ms  {}/{}/{}",
        timings.frontend_ms, timings.encode_ms, timings.decode_ms
    );
    println!(
        "  encoder frames / decode steps {} / {}",
        timings.encoder_frames, timings.decode_steps
    );
    match peak_rss {
        Some(b) => println!(
            "  peak working set  {} bytes ({:.2} GiB)",
            b,
            b as f64 / 1073741824.0
        ),
        None => println!("  peak working set  not measurable on this platform"),
    }
    println!("  exports:");
    for f in &written {
        println!("    {:<6} {:>10} bytes", f.format.extension(), f.bytes);
    }

    let report = serde_json::json!({
        "audio_ms": total_ms,
        "vad_spans": timeline.segments.len(),
        "vad_speech_ratio": timeline.speech_ratio(total),
        "chunks": chunks.len(),
        "segments": transcript.segments.len(),
        "words": words,
        "speakers": diar.num_speakers,
        "attributed_ratio": attribution.attributed_ratio(),
        "stored_words": stats.words,
        "stored_segments": stats.segments,
        "stage_ms": {
            "media": media_ms, "vad": vad_ms, "asr_open_load": asr_ms,
            "asr_transcribe": asr_loop_ms, "diar": diar_ms,
            "store": store_ms, "export": export_ms, "total": total_ms_measured,
        },
        "rtf": {
            "media": rtf(media_ms), "vad": rtf(vad_ms),
            "asr_transcribe": rtf(asr_loop_ms), "diar": rtf(diar_ms),
            "store": rtf(store_ms), "export": rtf(export_ms),
            "total": rtf(total_ms_measured),
        },
        "peak_rss_bytes": peak_rss,
        "asr_split_ms": {
            "frontend": timings.frontend_ms, "encode": timings.encode_ms,
            "decode": timings.decode_ms,
        },
        "encoder_frames": timings.encoder_frames,
        "decode_steps": timings.decode_steps,
        "words_per_segment": words as f64 / transcript.segments.len().max(1) as f64,
        "srt_cues": written
            .iter()
            .find(|f| f.format.extension() == "srt")
            .map(|f| f.cues)
            .unwrap_or(0),
        "past_the_end_segments": past_segments,
        "past_the_end_worst_ms": past_worst,
        "exports": written.iter().map(|f| serde_json::json!({
            "format": f.format.extension(), "bytes": f.bytes, "cues": f.cues,
        })).collect::<Vec<_>>(),
    });
    std::fs::write(
        &report_path,
        serde_json::to_string_pretty(&report).expect("serialise the report"),
    )
    .expect("write the report");
    println!("\n  report           {}", report_path.display());
    println!("ACCEPTANCE OK\n");
}

// ---------------------------------------------------------------------------
// The checks
// ---------------------------------------------------------------------------

/// No word ends before it starts; nothing lands outside the media.
fn word_timings_are_monotonic_and_in_bounds(t: &Transcript, total: Time) -> (usize, i64) {
    let total_ms = total.as_millis();
    let mut word_count = 0usize;

    for (i, seg) in t.segments.iter().enumerate() {
        assert!(
            seg.start.as_millis() >= 0,
            "segment {i} starts before the media at {} ms",
            seg.start.as_millis()
        );
        assert!(
            seg.end.as_millis() >= seg.start.as_millis(),
            "segment {i} has a negative duration: {}..{}",
            seg.start.as_millis(),
            seg.end.as_millis()
        );

        let mut prev_end: Option<Time> = None;
        for (j, w) in seg.words.iter().enumerate() {
            word_count += 1;
            assert!(
                w.end.as_millis() >= w.start.as_millis(),
                "segment {i} word {j} ends before it starts: {}..{}",
                w.start.as_millis(),
                w.end.as_millis()
            );
            assert!(
                w.start.as_millis() >= 0,
                "segment {i} word {j} starts before the media at {} ms",
                w.start.as_millis()
            );
            if let Some(p) = prev_end {
                assert!(
                    w.start.as_millis() >= p.as_millis(),
                    "segment {i} word {j} starts at {} ms, before the previous word ended at {} ms",
                    w.start.as_millis(),
                    p.as_millis()
                );
            }
            prev_end = Some(w.end);
        }
    }

    // Segments are kept sorted by `push_chunk`; a violation means the insertion
    // point is wrong, which is the same family of bug as an unrebased chunk.
    for w in t.segments.windows(2) {
        assert!(
            w[0].start.as_millis() <= w[1].start.as_millis(),
            "segments are out of order: {} ms then {} ms",
            w[0].start.as_millis(),
            w[1].start.as_millis()
        );
    }
    assert!(word_count > 0, "the transcript has no words at all");
    // The tail. `Transcript::push_chunk` clamps chunk-local times at zero but
    // has no media duration to clamp against, so an ASR frame that runs past
    // the end of the recording lands past the end of the recording. The front
    // end's frame grid is 20 ms, so a handful of frames' worth on the last
    // segment is arithmetic, not a rebasing bug; a *systematic* overshoot
    // would be one, and the two are told apart by the ratio below.
    let mut over_segments = 0usize;
    let mut over_words = 0usize;
    let mut worst = 0i64;
    for s in &t.segments {
        if s.end.as_millis() > total_ms {
            over_segments += 1;
            worst = worst.max(s.end.as_millis() - total_ms);
        }
        for w in &s.words {
            if w.end.as_millis() > total_ms {
                over_words += 1;
                worst = worst.max(w.end.as_millis() - total_ms);
            }
        }
    }
    let seg_ratio = over_segments as f64 / t.segments.len() as f64;
    println!(
        "  ..  past-the-end: {over_segments} segments, {over_words} words, worst {worst} ms \
         ({:.3} % of segments)",
        seg_ratio * 100.0
    );
    assert!(
        worst <= MAX_TAIL_OVERSHOOT_MS,
        "a time lands {worst} ms past the end of a {total_ms} ms recording; the tolerance is \
         {MAX_TAIL_OVERSHOOT_MS} ms (two front-end frames)"
    );
    assert!(
        seg_ratio <= 0.02,
        "{over_segments} of {} segments ({:.1} %) end after the media; that is a systematic \
         overshoot, not a frame-boundary tail",
        t.segments.len(),
        seg_ratio * 100.0
    );
    println!(
        "  ok  monotonic, in-bounds: {word_count} words, {} segments",
        t.segments.len()
    );
    (over_segments, worst)
}

/// How far past the end of the media a single time may land before the gate
/// calls it a defect rather than a frame boundary.
///
/// The log-mel front end's hop is 20 ms at 16 kHz and the encoder subsamples
/// 8x, so one encoder frame is 160 ms of audio; a decoded word's end time can
/// legitimately land one or two frames past the last sample. 40 ms is two
/// front-end frames, which is the smallest bound that is not itself an
/// estimate.
const MAX_TAIL_OVERSHOOT_MS: i64 = 40;

/// Words per segment a transcript should reach once the ASR adapter's
/// one-segment-per-word output is grouped into phrases. Reported, not asserted:
/// see `every_export_format_is_valid` for why.
const MIN_WORDS_PER_SEGMENT: f64 = 3.0;

/// The engine's chunk-local times were rebased exactly once, by the chunk offset.
fn chunks_are_rebased_onto_the_media_timeline(
    t: &Transcript,
    expected: &[(Time, Time)],
    chunks: &[vaani_vad::Chunk],
) {
    // The decisive test: for every segment the engine produced, a segment
    // exists at `chunk_offset + chunk_local`. If `push_chunk` were skipped the
    // transcript would hold the local times; if it double-added, or added a
    // chunk length instead of an offset, these pairs would not be there either.
    let present: BTreeSet<(i64, i64)> = t
        .segments
        .iter()
        .map(|s| (s.start.as_millis(), s.end.as_millis()))
        .collect();
    let mut missing = 0usize;
    let mut first_missing = None;
    for (i, e) in expected.iter().enumerate() {
        if !present.contains(&(e.0.as_millis(), e.1.as_millis())) {
            missing += 1;
            first_missing.get_or_insert(i);
        }
    }
    assert_eq!(
        missing,
        0,
        "{missing} of {} chunk-local segments are not at their expected rebased position \
         (first at expected index {:?}); the rebasing point is wrong",
        expected.len(),
        first_missing
    );

    // The transpose of that test: the transcript must span the whole recording,
    // not just the first chunk's worth of it. A transcript that never gets past
    // ~30 s is the signature of feeding chunk-local times straight in.
    let span_ms = t
        .segments
        .last()
        .map(|s| s.end)
        .unwrap_or(Time::ZERO)
        .as_millis();
    let media_ms = t.duration.map(|d| d.as_millis()).unwrap_or(0);
    assert!(
        span_ms * 10 > media_ms * 9,
        "the transcript only reaches {span_ms} ms of a {media_ms} ms recording; \
         the last chunks were not rebased"
    );

    // And it must not be shifted by a *constant* either: chunk k's segments
    // must sit inside chunk k's own window (plus the policy's padding). A
    // constant offset would push a whole chunk's worth of segments into a
    // neighbouring window.
    let pad = vaani_vad::ChunkPolicy::default().pad_ms;
    let mut placed = 0usize;
    for c in chunks {
        let inside = t
            .segments
            .iter()
            .filter(|s| {
                s.start.as_millis() >= c.start.as_millis() - pad
                    && s.start.as_millis() < c.end.as_millis()
            })
            .count();
        assert!(
            inside > 0,
            "no segment landed in the chunk at {}..{} ms; the transcript has drifted off the timeline",
            c.start.as_millis(),
            c.end.as_millis()
        );
        placed += inside;
    }
    assert_eq!(
        placed,
        t.segments.len(),
        "{} of {} segments are outside every chunk's window",
        t.segments.len() - placed,
        t.segments.len()
    );
    println!(
        "  ok  rebasing: {} chunk-local segments all present at offset positions, \
         spanning 0..{span_ms} ms",
        expected.len()
    );
}

/// Segment times are anchored to detected speech, not merely plausible.
fn segments_land_inside_detected_speech(
    t: &Transcript,
    timeline: &vaani_core::pipeline::SpeechTimeline,
) {
    let touching = t
        .segments
        .iter()
        .filter(|s| timeline.overlapping(s.start, s.end).next().is_some())
        .count();
    let ratio = touching as f64 / t.segments.len() as f64;
    // A real transcript has a few segments that straddle a VAD boundary or sit
    // in a gap the detector closed late. Anything below 80% means the timeline
    // and the transcript are describing different things.
    assert!(
        ratio >= 0.80,
        "only {touching} of {} segments ({:.1} %) overlap detected speech; the times are not \
         anchored to the audio",
        t.segments.len(),
        ratio * 100.0
    );

    // The stronger version: how much of the recording's *length* of speech did
    // the transcript actually cover? A transcript stuck in the first chunk
    // touches one or two VAD spans; a rebased one touches nearly all of them.
    let spans = timeline.segments.len().max(1);
    let touched = timeline
        .segments
        .iter()
        .filter(|sp| {
            t.segments
                .iter()
                .any(|s| s.start < sp.end && s.end > sp.start)
        })
        .count();
    let ratio = touched as f64 / spans as f64;
    assert!(
        ratio >= 0.50,
        "the transcript only touches {touched} of {spans} speech spans ({:.1} %); \
         that is what an unrebased transcript looks like",
        ratio * 100.0
    );
    println!("  ok  in sync: {touching} segments inside speech, {touched}/{spans} spans covered");
}

/// Speaker labels are internally consistent and actually present.
fn speaker_attribution_is_consistent(
    t: &Transcript,
    attribution: &vaani_diarization::attribution::AttributionReport,
    diar: &vaani_core::pipeline::DiarizationResult,
) {
    let mut per_speaker: std::collections::BTreeMap<usize, i64> = Default::default();
    let mut attributed = 0usize;
    for (i, s) in t.segments.iter().enumerate() {
        match s.speaker {
            None => {}
            Some(spk) => {
                attributed += 1;
                assert!(
                    t.speakers.iter().any(|k| k.index == spk),
                    "segment {i} claims speaker {spk}, which is not registered"
                );
                assert!(
                    spk < diar.num_speakers.max(1),
                    "segment {i} claims speaker {spk} but diarization found {}",
                    diar.num_speakers
                );
                *per_speaker.entry(spk).or_default() += s.duration().as_millis().max(0);
            }
        }
    }
    assert_eq!(
        attributed, attribution.attributed,
        "the attribution report says {} attributed, the transcript has {attributed}",
        attribution.attributed
    );
    let total_attributed_segments = t.segments.len();
    let unattributed = total_attributed_segments - attributed;
    assert!(
        attributed * 2 > unattributed,
        "only {attributed} of {total_attributed_segments} segments got a speaker; \
         diarization is not really running in this configuration"
    );
    assert_eq!(
        per_speaker.len(),
        diar.num_speakers,
        "{} speakers registered but {} own a segment",
        diar.num_speakers,
        per_speaker.len()
    );

    // Talk-time shares: recompute and compare against the measured per-speaker
    // totals, then check they sum to one.
    let sum_ms: i64 = per_speaker.values().sum();
    assert!(sum_ms > 0, "no attributed talk time at all");
    let mut t2 = t.clone();
    t2.recompute_speaker_shares();
    let shares: f32 = t2.speakers.iter().filter_map(|s| s.share).sum();
    assert!(
        (shares - 1.0).abs() < 1e-4,
        "speaker shares sum to {shares}, not 1"
    );
    for s in &t2.speakers {
        let Some(share) = s.share else { continue };
        let measured = per_speaker.get(&s.index).copied().unwrap_or(0) as f32 / sum_ms as f32;
        assert!(
            (share - measured).abs() < 1e-4,
            "speaker {} has share {share} but {} of {} ms of talk time ({measured})",
            s.index,
            per_speaker.get(&s.index).copied().unwrap_or(0),
            sum_ms
        );
    }
    println!(
        "  ok  speakers: {attributed}/{total_attributed_segments} attributed, \
         {} speakers, {sum_ms} ms attributed talk time",
        diar.num_speakers
    );
}

/// The stored transcript is equal to the in-memory one, field for field.
fn the_stored_transcript_round_trips(memory: &Transcript, stored: &Transcript) {
    if let Some(diff) = vaani_storage::fixtures::diff(memory, stored) {
        panic!("the stored transcript differs from the in-memory one: {diff}");
    }
    // `diff` compares the fields it knows; assert the counts too so a future
    // field that is not in `diff` cannot hide behind a passing comparison.
    assert_eq!(memory.segments.len(), stored.segments.len());
    assert_eq!(memory.speakers.len(), stored.speakers.len());
    let words: usize = memory.segments.iter().map(|s| s.words.len()).sum();
    let stored_words: usize = stored.segments.iter().map(|s| s.words.len()).sum();
    assert_eq!(words, stored_words);
    println!(
        "  ok  round trip: {} segments / {words} words equal",
        stored.segments.len()
    );
}

/// Every export format written, and the two subtitle formats parse and check clean.
fn every_export_format_is_valid(written: &[vaani_export::ExportedFile], original: &Transcript) {
    // The shape of the transcript, before the bytes. `SraVaaniEngine::
    // tokens_to_segments` emits one `Segment` per `Word` and says in its own
    // doc comment that "segments are later merged into phrases and attributed
    // to speakers by diarization". The attribution half happens. **The merging
    // half does not exist anywhere in this repository.** So the shipped
    // transcript is a word-per-line list, an SRT is one cue per word, and
    // speaker attribution is 1,900 independent per-word majority votes rather
    // than a few turn-level decisions. Measured on the 14-minute recording:
    // 1,913 segments holding 1,913 words, 1.00 words per segment, 1,913 cues.
    //
    // This is a *reported metric with a named threshold*, not an assert. The
    // threshold is not met today and cannot be met by a change to this crate:
    // grouping is a product decision (by pause? by speaker turn? by maximum
    // length?) that changes every downstream contract. Making the gate red on
    // it would make the gate a red herring; hiding it would be exactly the
    // "plausible and not checked" failure this project keeps producing. So it
    // is measured, printed, and asserted nowhere, and it is the first entry
    // under OPEN ISSUES.
    let words: usize = original.segments.iter().map(|s| s.words.len()).sum();
    let segs = original.segments.len().max(1);
    let per_segment = words as f64 / segs as f64;
    let cues = written
        .iter()
        .find(|f| f.format.extension() == "srt")
        .map(|f| f.cues)
        .unwrap_or(0);
    println!(
        "  ..  segment shape: {words} words in {segs} segments ({per_segment:.2} words/segment), \
         {cues} srt cues; a grouped transcript would be several words per segment"
    );
    if per_segment < MIN_WORDS_PER_SEGMENT {
        println!(
            "  !!  KNOWN GAP: {per_segment:.2} words per segment is below the {MIN_WORDS_PER_SEGMENT:.1} \
             a merged transcript should reach. Nothing merges the ASR adapter's one-segment-per-word \
             output, so the transcript is word-per-line and the SRT is one cue per word."
        );
    }

    assert_eq!(
        written.len(),
        vaani_export::ExportFormat::ALL.len(),
        "expected all five formats"
    );
    let mut by_ext = std::collections::BTreeMap::new();
    for f in written {
        assert!(f.path.is_file(), "{} was not written", f.path.display());
        assert_eq!(
            std::fs::metadata(&f.path).map(|m| m.len()).unwrap_or(0),
            f.bytes,
            "{} is {} bytes on disk but the export reported {}",
            f.path.display(),
            std::fs::metadata(&f.path).map(|m| m.len()).unwrap_or(0),
            f.bytes
        );
        by_ext.insert(f.format.extension().to_string(), f.path.clone());
    }
    for ext in ["txt", "md", "srt", "vtt", "json"] {
        assert!(by_ext.contains_key(ext), "no {ext} export was written");
    }

    let srt = std::fs::read_to_string(&by_ext["srt"]).expect("read the srt");
    let cues = vaani_export::verify::validate_srt(&srt, 2)
        .unwrap_or_else(|e| panic!("the .srt does not parse: {e}"));
    let violations = vaani_export::verify::check(&cues, 2);
    assert!(
        violations.is_empty(),
        "the .srt breaks {} rule(s): {}",
        violations.len(),
        violations
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("; ")
    );
    assert!(!cues.is_empty(), "the .srt has no cues");
    assert_eq!(
        cues.len(),
        by_ext_len(written, "srt"),
        "the .srt cue count disagrees with what the exporter reported"
    );

    let vtt = std::fs::read_to_string(&by_ext["vtt"]).expect("read the vtt");
    let vcues = vaani_export::verify::validate_vtt(&vtt, 2)
        .unwrap_or_else(|e| panic!("the .vtt does not parse: {e}"));
    let violations = vaani_export::verify::check(&vcues, 2);
    assert!(
        violations.is_empty(),
        "the .vtt breaks {} rule(s): {}",
        violations.len(),
        violations
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("; ")
    );
    assert!(vtt.starts_with("WEBVTT"), "the .vtt has no WEBVTT header");
    assert_eq!(cues.len(), vcues.len(), "the two subtitle files disagree");

    // JSON must be the domain type, so reading it back is equality, not
    // similarity. This is the strongest export check there is.
    let json = std::fs::read_to_string(&by_ext["json"]).expect("read the json");
    let back = vaani_export::read_json_export(&json).expect("the .json is not a transcript");
    if let Some(diff) = vaani_storage::fixtures::diff(original, &back) {
        panic!("the exported JSON does not read back to the same transcript: {diff}");
    }

    // Text and Markdown are not machine-checked for meaning, but an empty one
    // means the renderer silently produced nothing.
    let txt = std::fs::read_to_string(&by_ext["txt"]).expect("read the txt");
    assert!(!txt.trim().is_empty(), "the .txt export is empty");
    let md = std::fs::read_to_string(&by_ext["md"]).expect("read the md");
    assert!(!md.trim().is_empty(), "the .md export is empty");
    println!(
        "  ok  exports: 5 files, {} cues, srt and vtt parse and check clean, json round-trips",
        cues.len()
    );
}

fn by_ext_len(written: &[vaani_export::ExportedFile], ext: &str) -> usize {
    written
        .iter()
        .find(|f| f.format.extension() == ext)
        .map(|f| f.cues)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Cancellation and failure, on the real path
// ---------------------------------------------------------------------------

/// A real job, cancelled mid-transcription, timed from the request to the stop.
///
/// This is not a synthetic token test: it runs
/// [`vaani_cli::pipeline::DirectPipeline::transcribe`] -- the same entry point
/// `vaani transcribe` calls -- against the real clip and the real weights, and
/// cancels it from another thread while a chunk is in flight. The number that
/// matters to a user is the gap between *pressing cancel* and *the job
/// actually stopping*, so that is what is measured: `cancel.requested_at` to
/// the moment the worker returns.
///
/// The upper bound is the honest one. `DirectPipeline` polls the token at the
/// top of every chunk (`cancel.check()?` before `engine.transcribe`) and the
/// engine polls it inside its own decode, so the worst case is "the rest of the
/// chunk in flight". A 40 s chunk costs about 1.6 s of ASR, so a bound of one
/// chunk's worth of work is the contract, not a wish.
#[test]
#[ignore = "needs the real weights and a real recording; run scripts/e2e/acceptance.ps1"]
fn a_cancelled_job_stops_promptly_and_says_so() {
    let Some(clip) = locate_clip() else {
        skip("no recording: set VAANI_E2E_CLIP or put a file at ../scratch/clip.wav");
    };
    if let Some(why) = runtime_missing() {
        skip(&why);
    }
    if let Some(why) = model_missing() {
        skip(&why);
    }

    let token = CancelToken::new();
    let worker_token = token.child();
    let clip_for_worker = clip.clone();
    let requested = std::sync::Arc::new(std::sync::Mutex::new(None::<Instant>));

    let outcome: Option<vaani_core::VaaniError> = std::thread::scope(|s| {
        let canceller = s.spawn({
            let token = token.child();
            let requested = std::sync::Arc::clone(&requested);
            move || {
                // Long enough to be past the VAD and the model load, and inside
                // the ASR chunk loop: 6 s of an ~8 s model open plus a 1.5 s
                // first chunk.
                std::thread::sleep(std::time::Duration::from_millis(6_000));
                let at = Instant::now();
                *requested.lock().expect("lock") = Some(at);
                token.cancel();
            }
        });

        let pipeline = vaani_cli::pipeline::DirectPipeline;
        let worker = s.spawn(move || {
            let cancel = worker_token;
            let req = vaani_cli::pipeline::TranscribeRequest::for_file(clip_for_worker);
            let t = Instant::now();
            match pipeline.transcribe(&req, &cancel) {
                Ok(_) => None,
                Err(e) => {
                    eprintln!(
                        "  (the job ran for {} ms before it stopped)",
                        t.elapsed().as_millis()
                    );
                    Some(e)
                }
            }
        });
        canceller.join().expect("canceller");
        worker.join().expect("worker")
    });

    let e = outcome.expect("a cancelled job must not report success");
    let at = (*requested.lock().expect("lock")).expect("the canceller ran");
    let stopped_at = Instant::now();
    assert!(
        e.is_cancelled(),
        "a cancelled job failed with {} rather than a cancellation: {}",
        e.code().as_str(),
        e.user.title
    );
    let latency = stopped_at.duration_since(at);
    println!(
        "  ok  cancel: stopped {:?} after the request, code {}",
        latency,
        e.code().as_str()
    );
    // One chunk of ASR, generously. Measured 1.6 s per 40 s chunk, so 5 s is
    // three chunks' worth; a regression past that means a stage stopped
    // polling the token.
    assert!(
        latency < std::time::Duration::from_secs(5),
        "cancellation took {latency:?} to take effect"
    );
}

/// The two failures a user hits first, checked on the real path.
#[test]
#[ignore = "needs a real recording; run scripts/e2e/acceptance.ps1"]
fn real_failures_name_the_cause_and_a_recovery() {
    let Some(clip) = locate_clip() else {
        skip("no recording: set VAANI_E2E_CLIP or put a file at ../scratch/clip.wav");
    };
    let dir = output_dir();
    std::fs::create_dir_all(&dir).expect("create the acceptance output directory");
    let pipeline = vaani_cli::pipeline::DirectPipeline;

    // 1. A file that is not audio at all. A user drags a PDF onto the window.
    let not_media = dir.join("not-media.bin");
    std::fs::write(
        &not_media,
        b"this is not a media file, it is a note to self",
    )
    .expect("write");
    let e = pipeline
        .transcribe(
            &vaani_cli::pipeline::TranscribeRequest::for_file(not_media),
            &CancelToken::new(),
        )
        .expect_err("a text file must not transcribe");
    assert_ne!(
        e.code(),
        vaani_core::ErrorCode::Internal,
        "a non-media file reports `internal`, which tells a user nothing: {}",
        e.user.title
    );
    assert!(
        !e.user.recovery.as_deref().unwrap_or("").is_empty(),
        "[{}] has no recovery action",
        e.code().as_str()
    );
    assert!(
        !e.user.detail.is_empty() && !e.user.title.is_empty(),
        "[{}] has no user-facing text",
        e.code().as_str()
    );
    println!(
        "  ok  not-media: [{}] {} / recovery: {}",
        e.code().as_str(),
        e.user.title,
        e.user.recovery.as_deref().unwrap_or("-")
    );

    // 2. A model id that is not installed. The message has to say which model
    //    and what to do, because "model_unavailable" on its own is a shrug.
    let mut req = vaani_cli::pipeline::TranscribeRequest::for_file(clip);
    req.model = "sravaani-9.9-not-a-real-model".to_string();
    let e = pipeline
        .transcribe(&req, &CancelToken::new())
        .expect_err("an unknown model must not transcribe");
    assert_ne!(
        e.code(),
        vaani_core::ErrorCode::Internal,
        "an unknown model reports `internal`: {}",
        e.user.title
    );
    assert!(
        e.debug_chain().contains("sravaani-9.9-not-a-real-model"),
        "the error does not name the model the user asked for: {}",
        e.debug_chain()
    );
    assert!(
        !e.user.recovery.as_deref().unwrap_or("").is_empty(),
        "[{}] has no recovery action",
        e.code().as_str()
    );
    println!(
        "  ok  unknown-model: [{}] {} / recovery: {}",
        e.code().as_str(),
        e.user.title,
        e.user.recovery.as_deref().unwrap_or("-")
    );
}

/// The two checks above that need neither weights nor a recording, so they are
/// in the ordinary gate rather than in the `#[ignore]`d acceptance run.
///
/// The bar is not "it returns an error". The bar is what a user reads: a
/// stable code, a sentence that names the cause, and something to do about it.
/// An app that says "internal error" has failed regardless of its RTF.
#[test]
fn errors_carry_a_code_a_cause_and_a_recovery() {
    // A missing model id, before any ONNX session is touched.
    let mgr = vaani_models::ModelManager::embedded().expect("the embedded manifest parses");
    let e = mgr
        .resolve("sravaani-9.9-not-a-real-model")
        .expect_err("an unknown model id must not resolve");
    assert_ne!(
        e.code(),
        vaani_core::ErrorCode::Internal,
        "unknown model -> internal"
    );
    assert!(
        e.debug_chain().contains("sravaani-9.9-not-a-real-model"),
        "the error does not name the model: {}",
        e.debug_chain()
    );
    assert!(
        !e.user.recovery.as_deref().unwrap_or("").is_empty(),
        "no recovery action on an unknown model id"
    );

    // A file that is not media, at the media layer rather than the pipeline.
    let dir = std::env::temp_dir().join("vaani-e2e-errors");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("not-media.bin");
    std::fs::write(&path, b"this is not a media file").expect("write");
    let e = vaani_media::probe(&path).expect_err("a text file must not probe");
    assert_ne!(
        e.code(),
        vaani_core::ErrorCode::Internal,
        "non-media -> internal"
    );
    assert!(
        !e.user.recovery.as_deref().unwrap_or("").is_empty(),
        "no recovery action on a non-media file"
    );
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// Preflight
// ---------------------------------------------------------------------------

fn locate_clip() -> Option<PathBuf> {
    let p = std::env::var_os("VAANI_E2E_CLIP")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_CLIP));
    p.is_file().then_some(p)
}

fn runtime_missing() -> Option<String> {
    let loc = vaani_inference::runtime::location();
    (!loc.is_present()).then(|| {
        format!(
            "no ONNX Runtime: {} does not exist. Run scripts\\fetch_runtime.ps1, \
             or set VAANI_RUNTIME_DIR to a directory holding onnxruntime.dll \
             (looked in {})",
            loc.ort_dll.display(),
            loc.provider_dirs
                .iter()
                .map(|d| d.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    })
}

fn model_missing() -> Option<String> {
    match vaani_models::ModelManager::embedded() {
        Err(e) => Some(format!("the embedded model manifest is unusable: {e}")),
        Ok(mgr) => match mgr.resolve(MODEL_ID) {
            Ok(loc) => {
                let missing = vaani_asr::artifacts_in(loc.dir()).missing();
                (!missing.is_empty())
                    .then(|| format!("{MODEL_ID} is present but missing {}", missing.join(", ")))
            }
            Err(e) => Some(format!("{MODEL_ID} is not installed: {}", e.user.detail)),
        },
    }
}

fn output_dir() -> PathBuf {
    std::env::var_os("VAANI_E2E_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("vaani-e2e-acceptance"))
}

fn media_basename(p: &Path) -> String {
    p.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "recording".to_string())
}

/// A skip, loudly. The caller greps for this.
fn skip(why: &str) -> ! {
    println!("ACCEPTANCE SKIPPED: {why}");
    std::process::exit(0);
}

fn open_engine(
    caps: &vaani_inference::MachineCapabilities,
    provider: Provider,
) -> vaani_core::Result<Box<dyn vaani_asr::AsrEngine>> {
    let mgr = vaani_models::ModelManager::embedded()
        .map_err(|e| vaani_core::err::internal().with_cause(format!("model manifest: {e}")))?;
    let engine_id = mgr.engine_id(MODEL_ID)?.to_string();
    let loc = mgr.resolve(MODEL_ID)?;
    let registry = vaani_asr::EngineRegistry::with_builtins(provider, caps)?;
    registry.create(&engine_id, &vaani_asr::artifacts_in(loc.dir()))
}

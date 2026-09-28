//! Contract tests for the diarization benchmark harness.
//!
//! These exist because the harness's most dangerous failure mode is a report
//! that looks like a measurement and is not. Every test here is about one of
//! those:
//!
//! * a real recording must carry **no** accuracy figure, and must say so in the
//!   report itself;
//! * a missing statistic must be `null`, never `0.0`, because a zero is a
//!   claim and a `NaN` is worse than either;
//! * the thresholds the sweep reports must be the ones the engine actually
//!   used, after bias scaling -- reporting the configured 1.05 when the engine
//!   cut at 0.8925 would make the whole step function unreadable;
//! * a case with no speech must report nothing rather than guessing.
//!
//! Everything asserted here is a *structural* property. No test in this file
//! claims the diarizer is accurate, because this repository has no annotated
//! reference against which such a claim could be checked.

use std::path::PathBuf;

use vaani_diarization::bench::{self, BenchOptions, CaseKind};

fn clip_path() -> Option<PathBuf> {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|root| root.join("scratch").join("clip.wav"))?;
    p.is_file().then_some(p)
}

/// Read a WAV and downmix, without depending on the whole media stack.
fn clip() -> Option<(Vec<f32>, u32)> {
    let path = clip_path()?;
    let m = vaani_diarization::wav::read_wav_mono(&path).ok()?;
    Some((m.samples, m.sample_rate))
}

#[test]
fn a_report_with_no_input_still_serialises_and_explains_itself() {
    let report = bench::run(&BenchOptions::default());
    let json = report.to_json_pretty().expect("serialise");
    assert!(report.cases.is_empty());
    assert!(
        report
            .notes
            .iter()
            .any(|n| n.contains("No --input was given")),
        "notes: {:?}",
        report.notes
    );
    assert!(!json.contains("NaN"), "NaN is not a measurement");
    let back: bench::BenchReport = serde_json::from_str(&json).expect("round trip");
    assert_eq!(back, report);
}

#[test]
fn every_report_carries_the_caveats_in_the_machine_readable_output() {
    let report = bench::run(&BenchOptions::default());
    let joined = report.notes.join(" ");
    // A JSON file read on its own must not be mistakable for an accuracy
    // measurement, so the statement has to be *in* the file.
    assert!(joined.contains("STRUCTURE AND TIMING ONLY"), "{joined}");
    assert!(
        joined.contains("no annotated per-speaker reference"),
        "{joined}"
    );
    assert!(
        joined.contains("nothing derived from what was said"),
        "{joined}"
    );
}

#[test]
fn silence_produces_a_report_that_claims_nothing() {
    // 30 s of digital silence: a complete and correct answer, and a good test of
    // whether the harness turns "no speakers" into an error or a zero.
    let samples = vec![0.0f32; 16_000 * 30];
    let report = bench::run(&BenchOptions {
        input: None,
        synthetic: false,
        ..BenchOptions::default()
    });
    assert!(report.cases.is_empty());
    // And the zero-length case, through the public path.
    let r = vaani_core::pipeline::DiarizationEngine::diarize(
        &vaani_diarization::HeuristicDiarizer::new(),
        &vaani_core::media::AudioBuffer::mono(samples, 16_000, vaani_core::Time::ZERO),
        &vaani_core::CancelToken::new(),
        &|_| {},
    )
    .expect("silence is a result, not an error");
    assert_eq!(r.num_speakers, 0);
    assert!(r.spans.is_empty());
}

#[test]
fn a_real_recording_carries_no_reference_and_no_accuracy_figure() {
    let Some((_, rate)) = clip() else {
        println!("skipping: scratch/clip.wav is not present");
        return;
    };
    assert_eq!(rate, 16_000, "the engine requires 16 kHz mono");
    let report = bench::run(&BenchOptions {
        input: clip_path(),
        synthetic: false,
        sweep: false,
        variants: false,
        external_timeline: false,
        repeats: 1,
        max_anchors: None,
    });
    let case = report.case("real-clip").expect("the real case must exist");
    assert_eq!(case.kind, CaseKind::RealClip);
    assert!(
        case.reference.is_none(),
        "there is no annotated reference for a real recording, so there is
         nothing to score against and a report that produced one would be
         fabricating its ground truth"
    );
    // And nothing anywhere in the document that looks like an accuracy figure.
    let json = report.to_json_pretty().expect("serialise");
    assert!(
        !json.contains("\"purity\""),
        "no purity for a real recording"
    );
    assert!(
        !json.contains("\"recall\""),
        "no recall for a real recording"
    );
    assert!(!json.contains("\"der\""), "no DER is computed anywhere");
    assert!(!json.contains("\"jer\""), "no JER is computed anywhere");
    assert!(!json.contains("NaN"));
    // The structural figures are present and sane.
    assert!(case.result.speaker_count >= 1);
    assert!(case.result.speaker_count <= 8, "the cap must hold");
    assert!(!case.segments.is_empty(), "a meeting has turns");
    assert_eq!(case.segments.len(), case.result.span_count);
    assert!(case.result.speaker_change_count <= case.result.span_count);
    assert!(case.timing.rtf > 0.0 && case.timing.rtf < 0.5, "RTF");
    assert!(case.activity.active_frame_ratio > 0.0);
    assert!(case.activity.active_frame_ratio <= 1.0);
    assert!(case.activity.window_keep_ratio.is_some());
}

#[test]
fn a_real_recording_spans_are_sorted_tight_and_in_bounds() {
    let Some(_) = clip() else {
        println!("skipping: scratch/clip.wav is not present");
        return;
    };
    let report = bench::run(&BenchOptions {
        input: clip_path(),
        synthetic: false,
        sweep: false,
        variants: false,
        external_timeline: false,
        repeats: 1,
        max_anchors: None,
    });
    let case = report.case("real-clip").expect("case");
    let dur = case.input.duration_ms;
    let mut prev_end = 0i64;
    for (i, s) in case.segments.iter().enumerate() {
        assert!(s.start_ms < s.end_ms, "span {i} is empty or inverted");
        assert!(s.start_ms >= prev_end, "span {i} overlaps its predecessor");
        assert!(s.start_ms >= 0, "span {i} starts before the media");
        assert!(
            s.end_ms <= dur + 1_000,
            "span {i} runs past the media by a lot"
        );
        assert!(s.speaker < case.result.speaker_count);
        assert_eq!(s.duration_ms, s.end_ms - s.start_ms);
        if let Some(sc) = s.score {
            assert!((0.0..=1.0).contains(&sc), "score {sc} out of range");
        }
        prev_end = s.end_ms;
    }
    // The span list and the count cannot disagree, and the coverage figure has
    // to agree with the spans it came from.
    let total: i64 = case.segments.iter().map(|s| s.duration_ms).sum();
    let implied = total as f64 / dur as f64;
    assert!(
        (implied - case.result.coverage).abs() < 0.01,
        "coverage {} but the spans sum to {implied}",
        case.result.coverage
    );
}

#[test]
fn the_sweep_reports_the_distances_the_engine_actually_used() {
    // This is the invariant that makes the whole step function readable: the
    // `requested` column is the configuration and the `effective` column is the
    // number the stopping rule compared against. Reporting 1.05 where the
    // engine used 0.8925 would make the count-versus-threshold curve
    // uninterpretable, and the bug would be invisible because both columns
    // would be plausible.
    let report = bench::run(&BenchOptions {
        synthetic: true,
        sweep: true,
        max_anchors: Some(1),
        ..BenchOptions::default()
    });
    assert!(!report.merge_sweep.is_empty());
    for p in &report.merge_sweep {
        // SplitEasily scales the base by 0.85.
        let expected = p.requested * 0.85;
        assert!(
            (p.effective - expected).abs() < 1e-9,
            "requested {} became effective {}, expected {expected}",
            p.requested,
            p.effective
        );
        assert!(p.speaker_count >= 1);
        assert!(p.speaker_change_count <= p.span_count);
        assert!(p.mean_turn_ms >= 0.0);
        assert_eq!(p.rule, "threshold");
    }
    // The shipped default must appear in the sweep, or the curve does not cover
    // the operating point.
    let default = vaani_diarization::DiarizationConfig::default().effective_merge_distance();
    assert!(
        report
            .merge_sweep
            .iter()
            .any(|p| (p.effective - default).abs() < 1e-9),
        "the sweep does not include the shipped distance {default}"
    );
}

#[test]
fn every_synthetic_fixture_is_scored_and_its_score_is_a_generated_one() {
    let report = bench::run(&BenchOptions {
        synthetic: true,
        ..BenchOptions::default()
    });
    let scored: Vec<_> = report
        .cases
        .iter()
        .filter(|c| c.reference.is_some())
        .collect();
    assert!(scored.len() >= 5, "only {} scored cases", scored.len());
    for c in &scored {
        let r = c.reference.as_ref().expect("scored");
        assert!(
            r.provenance.contains("synth::render"),
            "the provenance must say where the reference came from"
        );
        assert!(r.true_speakers >= 1);
        assert!(r.purity.is_some_and(|p| (0.0..=1.0).contains(&p)));
        assert!(r.recall.is_some_and(|p| (0.0..=1.0).contains(&p)));
        assert!(
            r.complete_accuracy
                .is_some_and(|p| (0.0..=1.0).contains(&p))
        );
        assert_eq!(
            r.windows_per_true_speaker.len() as i32,
            r.true_speakers as i32,
            "every generator speaker must be accounted for"
        );
    }
    // The oracle case exists precisely so the activity gate can be separated
    // from the features, and it must be present and must keep every frame.
    let oracle = report
        .case("synthetic-3spk-one-20db-down#oracle")
        .expect("the oracle case must run");
    assert!(
        oracle.activity.active_frame_ratio > 0.99,
        "{:?}",
        oracle.activity
    );
    let internal = report
        .case("synthetic-3spk-one-20db-down")
        .expect("the internal case must run");
    assert!(internal.activity.active_frame_ratio < 0.8);
}

#[test]
fn the_block_shares_are_a_partition_of_the_distance_not_a_ranking() {
    // The decomposition is additive per dimension, so the shares must sum to the
    // mean pairwise distance. When they do not, the report is describing a
    // distance nobody computed -- which is how a block ends up "contributing
    // 1.14" of a quantity whose maximum is 2.
    for synthetic in [true, false] {
        let mut opts = BenchOptions {
            synthetic,
            repeats: 1,
            ..BenchOptions::default()
        };
        if !synthetic {
            if clip_path().is_none() {
                println!("skipping the real case: scratch/clip.wav is not present");
                continue;
            }
            opts.input = clip_path();
        }
        let report = bench::run(&opts);
        for c in report.cases.iter().filter(|c| c.geometry.is_some()) {
            let g = c.geometry.as_ref().expect("geometry");
            let sum: f64 = g.blocks.iter().map(|b| b.mean_share).sum();
            let mean = g.pairwise.mean.expect("a pairwise population");
            assert!(
                (sum - mean).abs() < 0.02 * mean + 1e-6,
                "{}: block shares sum to {sum}, mean pairwise distance is {mean}",
                c.name
            );
            // And the pitch block is always exactly one dimension, which is the
            // thing the whole absolute-pitch design rests on.
            let pitch = g.blocks.iter().find(|b| b.block == "pitch").expect("pitch");
            assert_eq!(pitch.dims, 1);
        }
    }
}

#[test]
fn a_run_is_deterministic() {
    // Speaker indices are persisted in the transcript, so two runs over the same
    // input must produce identical output or a re-run silently rewrites every
    // label in the file.
    let a = bench::run(&BenchOptions {
        synthetic: true,
        repeats: 1,
        ..BenchOptions::default()
    });
    let b = bench::run(&BenchOptions {
        synthetic: true,
        repeats: 1,
        ..BenchOptions::default()
    });
    let strip = |r: &bench::BenchReport| {
        r.cases
            .iter()
            .map(|c| {
                (
                    c.name.clone(),
                    c.result.speaker_count,
                    c.result.span_count,
                    c.result.talk_ms.clone(),
                    c.segments.clone(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(strip(&a), strip(&b), "two runs disagreed");
}

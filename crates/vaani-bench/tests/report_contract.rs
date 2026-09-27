//! Contract tests for the `--json` report shape.
//!
//! A8 parses this output, so field names and the round-trip are pinned here,
//! not just in unit tests. The literal construction below is deliberate: any
//! added/renamed field breaks compilation here first.

use vaani_bench::report::BenchReport;

fn sample() -> BenchReport {
    BenchReport {
        input: "meeting.mp4 (1234 bytes)".to_string(),
        status: "skip".to_string(),
        note: Some("SKIP: live staged pipeline wiring pending vaani-app (A2)".to_string()),
        startup_ms: 3,
        media_ms: 0,
        vad_ms: 0,
        asr_ms: 0,
        diar_ms: 0,
        post_ms: 0,
        export_ms: 0,
        wall_ms: 12,
        audio_duration_ms: 0,
        peak_rss_mb: 8.0,
        peak_vram_mb: None,
        rtf: 0.0,
        segments: 0,
        speakers: 0,
        failures: 0,
        git_sha: "unknown".to_string(),
        device: "cpu".to_string(),
    }
}

#[test]
fn json_report_has_every_contract_field() {
    let value: serde_json::Value =
        serde_json::from_str(&sample().to_json()).expect("report must be valid JSON");
    for field in [
        "input",
        "status",
        "note",
        "startup_ms",
        "media_ms",
        "vad_ms",
        "asr_ms",
        "diar_ms",
        "post_ms",
        "export_ms",
        "wall_ms",
        "audio_duration_ms",
        "peak_rss_mb",
        "peak_vram_mb",
        "rtf",
        "segments",
        "speakers",
        "failures",
        "git_sha",
        "device",
    ] {
        assert!(
            value.get(field).is_some(),
            "contract field missing: {field}"
        );
    }
}

#[test]
fn json_round_trip_through_the_public_api() {
    let report = sample();
    let back = BenchReport::from_json(&report.to_json()).expect("must parse");
    assert_eq!(report, back);
}

#[test]
fn rtf_definition_matches_performance_md() {
    // RTF = processing_time / audio_duration (docs/agent/PERFORMANCE.md).
    assert!((BenchReport::rtf(2_000, 40_000) - 0.05).abs() < 1e-12);
    assert_eq!(BenchReport::rtf(1_000, 0), 0.0);
}

#[test]
fn human_table_names_every_stage() {
    let table = sample().render_table();
    for stage in [
        "startup", "media", "vad", "asr", "diar", "post", "export", "wall",
    ] {
        assert!(table.contains(stage), "table must name stage {stage}");
    }
}

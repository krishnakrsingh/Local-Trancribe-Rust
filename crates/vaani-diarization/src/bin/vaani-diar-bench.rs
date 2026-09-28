//! `vaani-diar-bench` — the diarization benchmark harness.
//!
//! One command, one JSON document:
//!
//! ```powershell
//! cargo run -p vaani-diarization --release --bin vaani-diar-bench -- `
//!   --input scratch/clip.wav --out docs/performance/diarization-bench.json
//! ```
//!
//! # What it writes
//!
//! A [`BenchReport`](vaani_diarization::bench::BenchReport) as pretty JSON: the
//! engine's identity, the standing caveats, one case per input and
//! configuration, the merge-distance sweep, and the configuration variants.
//!
//! # What it will not write
//!
//! No transcript text, and nothing derived from what was said. The harness
//! holds samples and spans; the only strings in the output are case names,
//! configuration numbers and engine ids. Span start and end times are emitted
//! because a benchmark whose output cannot be checked is not a benchmark.
//!
//! The report contains **no accuracy figure for any real recording**, and says
//! so in its own `notes` array, because this repository has no annotated
//! reference for real audio. The only scored cases are generated ones, whose
//! reference is the generator's own per-frame labels.
//!
//! # Exit codes
//!
//! * `0` — the report was written. A missing input is *not* a failure: the
//!   report records the reason and the run still produced a document.
//! * `2` — the arguments were not understood, or `--out` could not be written.
//!
//! Nothing here can fail a job, abort a meeting, or leak content; it is a
//! measurement tool and it is written as one.

use std::path::PathBuf;
use std::process::ExitCode;

use vaani_diarization::bench::{BenchOptions, run};

const USAGE: &str = "\
vaani-diar-bench — diarization structure and timing benchmark

USAGE:
    vaani-diar-bench [OPTIONS]

OPTIONS:
    --input <PATH>        A WAV file. 16-bit PCM or 32-bit float, any channel
                          count, downmixed to mono. Optional: without it no real
                          recording is run and the report says so.
    --out <PATH>          Write the JSON here. Default: stdout.
    --external-timeline   Also run the input with vaani-vad's SpeechTimeline,
                          which is the path vaani-app takes. Needs the
                          `vad-probe` cargo feature.
    --sweep               Run the merge-distance sweep.
    --variants            Run the configuration variants.
    --synthetic           Run the synthetic fixtures, including the
                          oracle-timeline pair.
    --repeats <N>         Run the primary case N times. Default 1.
    --all                 The same as --sweep --variants --synthetic
                          --external-timeline.
    --pretty              Force pretty JSON. It is the only mode.
    -h, --help            This text.

EXAMPLES:
    # The real meeting, the sweep, the variants and the scored synthetic set.
    cargo run -p vaani-diarization --release --bin vaani-diar-bench -- \\
        --input scratch/clip.wav --all --out docs/performance/bench.json

    # Reproducible, no private file involved.
    cargo run -p vaani-diarization --release --bin vaani-diar-bench -- --all

    # With the pipeline's real VAD in the report.
    cargo run -p vaani-diarization --release --features vad-probe --bin \\
        vaani-diar-bench -- --input scratch/clip.wav --external-timeline
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let (opts, out) = match parse(&args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("vaani-diar-bench: {e}");
            eprint!("\n{USAGE}");
            return ExitCode::from(2);
        }
    };

    let report = run(&opts);
    let json = match report.to_json_pretty() {
        Ok(j) => j,
        Err(e) => {
            eprintln!("vaani-diar-bench: could not serialise the report: {e}");
            return ExitCode::from(2);
        }
    };

    match &out {
        Some(path) => {
            if let Some(parent) = path.parent()
                && !parent.as_os_str().is_empty()
                && let Err(e) = std::fs::create_dir_all(parent)
            {
                eprintln!(
                    "vaani-diar-bench: could not create {}: {e}",
                    parent.display()
                );
                return ExitCode::from(2);
            }
            if let Err(e) = std::fs::write(path, format!("{json}\n")) {
                eprintln!("vaani-diar-bench: could not write {}: {e}", path.display());
                return ExitCode::from(2);
            }
            summarise(&report);
            eprintln!("wrote {}", path.display());
        }
        None => println!("{json}"),
    }
    ExitCode::SUCCESS
}

/// Parse the arguments.
///
/// `# Errors`
/// Returns a message naming the offending flag. An unknown flag is an error
/// rather than a warning, because a benchmark that silently ignores the
/// configuration you asked for produces a report about a different experiment.
fn parse(args: &[String]) -> Result<(BenchOptions, Option<PathBuf>), String> {
    let mut opts = BenchOptions::default();
    let mut out: Option<PathBuf> = None;
    let mut i = 0usize;
    while i < args.len() {
        let a = args[i].as_str();
        let mut value = |flag: &str| -> Result<String, String> {
            i += 1;
            args.get(i)
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match a {
            "--input" => opts.input = Some(PathBuf::from(value("--input")?)),
            "--out" => out = Some(PathBuf::from(value("--out")?)),
            "--repeats" => {
                let v = value("--repeats")?;
                opts.repeats = v
                    .parse::<usize>()
                    .map_err(|_| format!("--repeats needs a positive integer, got {v:?}"))?;
                if opts.repeats == 0 {
                    return Err("--repeats needs at least 1".to_owned());
                }
            }
            "--external-timeline" => opts.external_timeline = true,
            "--sweep" => opts.sweep = true,
            "--variants" => opts.variants = true,
            "--synthetic" => opts.synthetic = true,
            "--all" => {
                opts.sweep = true;
                opts.variants = true;
                opts.synthetic = true;
                opts.external_timeline = true;
            }
            "--pretty" => {}
            other => return Err(format!("unknown argument {other:?}")),
        }
        i += 1;
    }
    Ok((opts, out))
}

/// A short human summary on stderr, so `--out` is still readable without
/// opening the file. Counts only.
fn summarise(report: &vaani_diarization::bench::BenchReport) {
    eprintln!(
        "vaani-diarization bench: {} case(s), {} sweep point(s), {} variant(s)",
        report.cases.len(),
        report.merge_sweep.len(),
        report.variants.len()
    );
    for c in &report.cases {
        eprintln!(
            "  {:<40} {:>2} speakers, {:>4} spans, {:>4} changes, RTF {:.4} ({:.0} ms)",
            c.name,
            c.result.speaker_count,
            c.result.span_count,
            c.result.speaker_change_count,
            c.timing.rtf,
            c.timing.wall_ms,
        );
    }
}

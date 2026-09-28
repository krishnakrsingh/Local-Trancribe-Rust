//! `vaani-eval` — the VaaniDesk accuracy evaluation harness.
//!
//! Hand-rolled argument parsing on purpose. This is a CI tool whose output is
//! read by scripts, so the flag surface is a contract that should be visible
//! in one file, and a dependency that can change a flag name between minor
//! versions is a dependency that can change what a benchmark means.
//!
//! ```text
//! vaani-eval <command> [flags]
//!
//!   selftest                       run the hand-computed known-answer suite
//!   probe   --in F --out F         structural summary of an audio file
//!   cut     --in F --out F --start MS --duration MS
//!                                  cut a clip and verify its length
//!   norm    --in F --out F [--profile P]
//!                                  apply the normalisation and print it
//!   hyp     --in F --out-dir D     read a vaani.transcript export, write
//!                                  hypothesis.{json,txt,rttm}
//!   score   --set D [--out F] [--profile P] [--collar-ms N] [--min-dur-ms N]
//!                                  score every clip in a set
//!   schema  [--out F]              print the report JSON schema
//!   rules   [--profile P]          print the normalisation rules
//! ```
//!
//! Exit codes: `0` success, `1` usage or runtime failure, `2` the set contained
//! a clip that could not be read at all. **Never** `0` for a set with no
//! scored clips, because "I ran the harness" must not be readable as "the
//! system is accurate"; use `--allow-unscored` for that, which prints a
//! warning to stderr and still exits `0`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use vaani_eval::der::DiarParams;
use vaani_eval::hypothesis::{self, Hypothesis};
use vaani_eval::normalize::{self, NormalisationProfile};
use vaani_eval::report::REPORT_VERSION;
use vaani_eval::rttm;
use vaani_eval::set::{self, Set};
use vaani_eval::taxonomy::TimingThresholds;
use vaani_eval::wav;

const USAGE: &str = "\
vaani-eval — VaaniDesk accuracy evaluation harness

USAGE
  vaani-eval <command> [flags]

COMMANDS
  selftest                              run the hand-computed known-answer suite
  probe     --in <file> --out <file>    structural summary of an audio file
  cut       --in <file> --out <file> --start <ms> --duration <ms>
                                         cut a clip and verify its length
  norm      --in <file> --out <file> [--profile strict|hinglish|hinglish-translit]
                                         apply the normalisation, print the rules
  hyp       --in <file> --out-dir <dir> read a vaani.transcript export and write
                                         hypothesis.json / .txt / .rttm
  score     --set <dir> [--out <file>] [--profile <p>] [--collar-ms <n>]
            [--min-dur-ms <n>] [--turn-tol-ms <n>] [--allow-unscored]
                                         score every clip in a set
  schema                                print the report JSON schema
  rules     [--profile <p>]            print the normalisation rules

EXIT CODES
  0 success
  1 usage or runtime failure
  2 a clip could not be read
  A set with no scored clips exits 1 unless --allow-unscored is given, so
  \"the harness ran\" can never be misread as \"the system is accurate\".
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("vaani-eval: {e}");
            ExitCode::from(1)
        }
    }
}

type R = Result<ExitCode, String>;

fn run(args: &[String]) -> R {
    let Some(cmd) = args.first().map(String::as_str) else {
        print!("{USAGE}");
        return Ok(ExitCode::from(2));
    };
    let flags = Flags::parse(&args[1..])?;
    match cmd {
        "help" | "--help" | "-h" => {
            print!("{USAGE}");
            Ok(ExitCode::SUCCESS)
        }
        "selftest" => selftest(),
        "metadata" => metadata(&flags),
        "probe" => probe(&flags),
        "cut" => cut(&flags),
        "norm" => norm(&flags),
        "hyp" => hyp(&flags),
        "score" => score(&flags),
        "schema" => schema(),
        "rules" => rules(&flags),
        "version" | "--version" => {
            println!("vaani-eval {}", vaani_eval::VERSION);
            Ok(ExitCode::SUCCESS)
        }
        other => Err(format!("unknown command `{other}`\n\n{USAGE}")),
    }
}

/// A minimal flag parser: `--name value` and `--name=value`, and bare booleans
/// for the switch flags. Unknown flags are an error, because a typo that
/// silently drops `--out` writes a benchmark to the wrong place.
#[derive(Debug, Default)]
struct Flags {
    values: Vec<(String, String)>,
    switches: Vec<String>,
}

const SWITCHES: &[&str] = &["--allow-unscored", "--emit-frames", "--no-timestamp"];

impl Flags {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut f = Self::default();
        let mut i = 0usize;
        while i < args.len() {
            let a = &args[i];
            if !a.starts_with("--") {
                return Err(format!("expected a flag, found `{a}`"));
            }
            if SWITCHES.contains(&a.as_str()) {
                f.switches.push(a.clone());
                i += 1;
                continue;
            }
            if let Some((k, v)) = a.split_once('=') {
                f.values.push((k.to_string(), v.to_string()));
                i += 1;
                continue;
            }
            let v = args
                .get(i + 1)
                .ok_or_else(|| format!("`{a}` needs a value"))?;
            f.values.push((a.clone(), v.clone()));
            i += 2;
        }
        Ok(f)
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.values
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    fn req(&self, name: &str) -> Result<&str, String> {
        self.get(name)
            .ok_or_else(|| format!("missing required flag `{name}`"))
    }

    fn num(&self, name: &str) -> Result<Option<i64>, String> {
        match self.get(name) {
            None => Ok(None),
            Some(v) => v
                .parse()
                .map(Some)
                .map_err(|_| format!("`{name}` must be an integer, got `{v}`")),
        }
    }

    fn has(&self, name: &str) -> bool {
        self.switches.iter().any(|s| s == name)
    }

    /// Reject anything the command does not understand.
    fn check_known(&self, allowed: &[&str]) -> Result<(), String> {
        for (k, _) in &self.values {
            let base = k.split('=').next().unwrap_or(k);
            if !allowed.contains(&base) {
                return Err(format!("unknown flag `{base}` for this command"));
            }
        }
        for s in &self.switches {
            if !allowed.contains(&s.as_str()) {
                return Err(format!("unknown switch `{s}` for this command"));
            }
        }
        Ok(())
    }
}

fn selftest() -> R {
    // The known-answer suite lives in `tests/known_answer.rs` and is run by
    // `cargo test`. This subcommand exists so a CI step can verify the
    // *binary* rather than the library, and it says so plainly rather than
    // pretending to be a second implementation.
    println!("vaani-eval selftest");
    println!("  the hand-computed known-answer suite is a cargo test target:");
    println!("    cargo test --manifest-path evaluation/scoring/Cargo.toml --test known_answer");
    println!("  it is not re-implemented here, because a second implementation of");
    println!("  the scorer is a second thing that can be wrong.");
    Ok(ExitCode::SUCCESS)
}

fn metadata(f: &Flags) -> R {
    f.check_known(&["--set"])?;
    let root = PathBuf::from(f.req("--set")?);
    let set = Set::load(&root).map_err(|e| e.to_string())?;
    let written = set.write_metadata().map_err(|e| e.to_string())?;
    for p in &written {
        println!("wrote {}", p.display());
    }
    println!(
        "{} per-clip metadata files, generated from {}. A generated file cannot \
         disagree with the manifest, which is the point.",
        written.len(),
        root.join("set.json").display()
    );
    Ok(ExitCode::SUCCESS)
}

fn probe(f: &Flags) -> R {
    f.check_known(&["--in", "--out", "--emit-frames"])?;
    let src = PathBuf::from(f.req("--in")?);
    let audio = wav::read(&src).map_err(|e| e.to_string())?;
    let a = vaani_eval::probe::analyse(&audio);
    let out = PathBuf::from(f.req("--out")?);
    let value = if f.has("--emit-frames") {
        serde_json::to_value(&a).map_err(|e| e.to_string())?
    } else {
        serde_json::to_value(&a.structure).map_err(|e| e.to_string())?
    };
    write_json(&out, &value)?;
    let s = &a.structure;
    println!(
        "{}",
        serde_json::to_string_pretty(&s).map_err(|e| e.to_string())?
    );
    Ok(ExitCode::SUCCESS)
}

fn cut(f: &Flags) -> R {
    f.check_known(&["--in", "--out", "--start", "--duration"])?;
    let src = PathBuf::from(f.req("--in")?);
    let dst = PathBuf::from(f.req("--out")?);
    let start = f.num("--start")?.ok_or("`--start` is required")?;
    let duration = f.num("--duration")?.ok_or("`--duration` is required")?;
    let audio = wav::read(&src).map_err(|e| e.to_string())?;
    let clip = wav::cut(&audio, start, duration).map_err(|e| e.to_string())?;
    if let Some(p) = dst.parent() {
        std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    wav::write(&dst, &clip).map_err(|e| e.to_string())?;
    // Verify what was written, by reading it back. A cut that silently
    // produced the wrong length would be scored as if it were the right one.
    let back = wav::read(&dst).map_err(|e| format!("the written clip does not read back: {e}"))?;
    println!(
        "{}  {} samples  {} Hz  requested {} ms  actual {} ms  sha256 {}",
        dst.display(),
        back.len(),
        back.sample_rate,
        duration,
        back.duration_ms(),
        set::hex_digest(&wav::encode(&back))
    );
    if back.duration_ms() != duration {
        return Err(format!(
            "the cut is {} ms but {} ms was requested",
            back.duration_ms(),
            duration
        ));
    }
    Ok(ExitCode::SUCCESS)
}

fn norm(f: &Flags) -> R {
    f.check_known(&["--in", "--out", "--profile"])?;
    let src = PathBuf::from(f.req("--in")?);
    let dst = PathBuf::from(f.req("--out")?);
    let profile = profile_of(f)?;
    let text = std::fs::read_to_string(&src).map_err(|e| e.to_string())?;
    let n = normalize::normalise(&text, profile);
    if let Some(p) = dst.parent() {
        std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    std::fs::write(&dst, n.tokens.join(" ")).map_err(|e| e.to_string())?;
    print_rules(profile);
    let v = serde_json::to_value(&n).map_err(|e| e.to_string())?;
    println!(
        "{}",
        serde_json::to_string_pretty(&v).map_err(|e| e.to_string())?
    );
    Ok(ExitCode::SUCCESS)
}

fn hyp(f: &Flags) -> R {
    f.check_known(&["--in", "--out-dir"])?;
    let src = PathBuf::from(f.req("--in")?);
    let dir = PathBuf::from(f.req("--out-dir")?);
    let text = std::fs::read_to_string(&src).map_err(|e| e.to_string())?;
    let h: Hypothesis = hypothesis::from_export_json(&text).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    // `hypothesis.json` is the system's own export, kept verbatim. The scorer
    // reads *that* file, so what is scored is byte-for-byte what the pipeline
    // produced and not a re-serialisation of this harness's view of it.
    std::fs::copy(&src, dir.join("hypothesis.json")).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("hypothesis.txt"), h.to_text()).map_err(|e| e.to_string())?;
    let id = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "hypothesis".to_string());
    std::fs::write(dir.join("hypothesis.rttm"), rttm::write(&h.to_rttm(&id)))
        .map_err(|e| e.to_string())?;
    // Counts only, so this one file is safe to read without opening anything
    // that could contain a word of the meeting.
    write_json(
        &dir.join("hypothesis.summary.json"),
        &serde_json::to_value(serde_json::json!({
            "source": h.source,
            "intrinsic": h.intrinsic(),
            "word_count": h.words.len(),
            "segment_count": h.segments.len(),
            "rttm_records": h.to_rttm(&id).len(),
        }))
        .map_err(|e| e.to_string())?,
    )?;
    let i = h.intrinsic();
    // Counts only. The word list is on disk, gitignored, and never printed.
    println!(
        "{}  words={} segments={} speakers={} unattributed={} duplicate_runs={} non_monotonic={}",
        dir.display(),
        i.word_count,
        i.segment_count,
        i.speaker_count,
        i.unattributed_segments,
        i.duplicate_runs,
        i.non_monotonic_word_pairs
    );
    Ok(ExitCode::SUCCESS)
}

fn score(f: &Flags) -> R {
    f.check_known(&[
        "--set",
        "--out",
        "--profile",
        "--collar-ms",
        "--min-dur-ms",
        "--turn-tol-ms",
        "--allow-unscored",
    ])?;
    let root = PathBuf::from(f.req("--set")?);
    let mut s = Set::load(&root).map_err(|e| e.to_string())?;
    if f.get("--profile").is_some() {
        s.def.normalisation_profile = profile_of(f)?;
    }
    let params = DiarParams {
        collar_us: f.num("--collar-ms")?.unwrap_or(0) * 1000,
        min_duration_us: f.num("--min-dur-ms")?.unwrap_or(0) * 1000,
        turn_tolerance_us: f.num("--turn-tol-ms")?.unwrap_or(250) * 1000,
    };
    let report = s.score(params, TimingThresholds::default());

    let json = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    let out = f
        .get("--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| s.results_dir().join("summary.json"));
    if let Some(p) = out.parent() {
        std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    std::fs::write(&out, format!("{json}\n")).map_err(|e| e.to_string())?;
    for c in &report.clips {
        let p = s.results_dir().join(format!("{}.report.json", c.clip_id));
        let j = serde_json::to_string_pretty(c).map_err(|e| e.to_string())?;
        std::fs::write(&p, format!("{j}\n")).map_err(|e| e.to_string())?;
    }

    print_summary(&report);
    println!("wrote {}", out.display());

    let scored = report.clips.iter().filter(|c| c.scored).count();
    if scored == 0 && !f.has("--allow-unscored") {
        eprintln!(
            "vaani-eval: 0 of {} clips could be scored, so no accuracy number exists. \
             Pass --allow-unscored to accept that as a successful run.",
            report.clips.len()
        );
        return Ok(ExitCode::from(1));
    }
    Ok(ExitCode::SUCCESS)
}

fn print_summary(r: &vaani_eval::report::Report) {
    println!("report version {REPORT_VERSION}, {}", r.generated_by);
    println!("normalisation profile: {}", r.normalisation.profile);
    println!(
        "clips {}  human ASR reference {}  human diarization reference {}  \
         human word-timing reference {}  awaiting annotation {}  with hypothesis {}",
        r.set.clips,
        r.set.human_asr_reference,
        r.set.human_diarization_reference,
        r.set.human_word_timing_reference,
        r.set.awaiting_annotation,
        r.set.with_hypothesis
    );
    println!();
    println!(
        "{:<12} {:<10} {:<14} {:<14} {:<12}",
        "clip", "scored", "asr(wer)", "diar(der)", "timing"
    );
    for c in &r.clips {
        println!(
            "{:<12} {:<10} {:<14} {:<14} {:<12}",
            c.clip_id,
            if c.scored { "yes" } else { "no" },
            fmt(c.asr.report.as_ref().and_then(|w| w.wer), &c.asr.status),
            fmt(
                c.diarization.report.as_ref().and_then(|d| d.der),
                &c.diarization.status
            ),
            fmt(
                c.timing
                    .report
                    .as_ref()
                    .and_then(|t| t.start_deviation.as_ref().map(|d| d.mean_ms)),
                &c.timing.status
            )
        );
    }
    println!();
    println!("blend: refused — {}", r.blend.why);
}

fn fmt(v: Option<f64>, status: &vaani_eval::report::BlockStatus) -> String {
    use vaani_eval::report::BlockStatus;
    match (v, status) {
        (Some(x), BlockStatus::Scored) => format!("{:.4}", x),
        (None, BlockStatus::Unscored { reason, .. }) => format!("unscored:{reason}"),
        (None, BlockStatus::Undefined { reason, .. }) => format!("undefined:{reason}"),
        (None, BlockStatus::Scored) => "n/a".to_string(),
        (Some(x), _) => format!("{x:.4}"),
    }
}

fn schema() -> R {
    println!("{}", SCHEMA);
    Ok(ExitCode::SUCCESS)
}

fn rules(f: &Flags) -> R {
    f.check_known(&["--profile"])?;
    print_rules(profile_of(f)?);
    Ok(ExitCode::SUCCESS)
}

fn print_rules(p: NormalisationProfile) {
    println!("normalisation profile: {}", p.id());
    for r in p.rules() {
        println!("  {:<4} {}", r.id, r.rule);
        if !r.why.is_empty() {
            println!("       why: {}", r.why);
        }
    }
    for d in vaani_eval::report::NormalisationBlock::of(p).does_not {
        println!("  does not: {d}");
    }
}

fn profile_of(f: &Flags) -> Result<NormalisationProfile, String> {
    match f.get("--profile") {
        None => Ok(NormalisationProfile::Strict),
        Some(s) => NormalisationProfile::parse(s).ok_or_else(|| {
            format!("unknown profile `{s}`; want strict|hinglish|hinglish-translit")
        }),
    }
}

fn write_json(p: &Path, v: &serde_json::Value) -> Result<(), String> {
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    std::fs::write(
        p,
        serde_json::to_string_pretty(v).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

/// The report schema, printed by `vaani-eval schema`. Descriptive rather than a
/// machine-readable JSON Schema, because the wire contract is the field names
/// and the three-class separation, both of which a JSON Schema cannot express.
const SCHEMA: &str = r#"{
  "version": 1,                          // report schema version; refuse unknown
  "generated_by": "vaani-eval <version>",
  "normalisation": {
    "profile": "strict | hinglish | hinglish-translit",
    "rules":     [ { "id": "N1", "rule": "...", "why": "..." } ],
    "does_not":  [ "..." ]
  },
  "set": {
    "clips": <int>,
    "human_asr_reference": <int>,
    "human_diarization_reference": <int>,
    "human_word_timing_reference": <int>,
    "awaiting_annotation": <int>,
    "with_hypothesis": <int>
  },
  "clips": [ {
    "clip_id": "<id>",
    "source_start_ms": <int>,
    "duration_ms": <int>,
    "has_human_asr_reference": <bool>,
    "has_human_diarization_reference": <bool>,
    "has_human_word_timing_reference": <bool>,
    "has_hypothesis": <bool>,
    "scored": <bool>,                     // true iff any block is scored
    "asr": {
      "status": "scored" | "unscored" | "undefined",
      "reason": "<code>",                 // present when not scored
      "detail": "<prose>",
      "wer": <number|null>,               // null means undefined, NOT zero
      "cer_characters": <number|null>,    // over Unicode scalar values
      "cer_grapheme": <number|null>,      // over grapheme clusters; quote this for Hindi
      "correct": <int>, "substitutions": <int>,
      "deletions": <int>, "insertions": <int>, "errors": <int>,
      "reference_tokens": <int>, "hypothesis_tokens": <int>,
      "reference_characters": <int>, "reference_graphemes": <int>,
      "mixed_script_reference_tokens": <int>,
      "alignment": { "length": <int>, "matches": <int>,
                     "sub": <int>, "del": <int>, "ins": <int> }
    },
    "diarization": {
      "status": "scored" | "unscored" | "undefined",
      "reason": "<code>", "detail": "<prose>",
      "der": <number|null>,
      "components": { "missed_detection": <f>, "false_alarm": <f>, "confusion": <f> },
      "error_ms":   { "missed_detection": <f>, "false_alarm": <f>, "confusion": <f> },
      "correct_ms": <f>,
      "jer": <number|null>, "jer_speech": <number|null>, "jer_non_speech": <number|null>,
      "speaker_count": { "reference": <int>, "hypothesis": <int>, "correct": <bool> },
      "permutation": { "<ref label>": "<hyp label>" },
      "boundary": { "matched": <int>, "unmatched_reference": <int>,
                    "unmatched_hypothesis": <int>,
                    "mean_abs_deviation_ms": <f>, "max_abs_deviation_ms": <f>,
                    "tolerance_ms": <f> },
      "false_speaker_change": <int>, "missed_speaker_change": <int>,
      "reference_turns": <int>, "hypothesis_turns": <int>,
      "params": { "collar_us": <int>, "min_duration_us": <int>, "turn_tolerance_us": <int> }
    },
    "timing": {
      "status": "scored" | "unscored" | "undefined",
      "reason": "<code>", "detail": "<prose>",
      "start_deviation": { "count": <int>, "mean_ms": <f>, "median_ms": <f>,
                           "p90_ms": <f>, "max_abs_ms": <f> },
      "end_deviation":   { "count": <int>, "mean_ms": <f>, "median_ms": <f>,
                           "p90_ms": <f>, "max_abs_ms": <f> },
      "matched_pairs": <int>,
      "negative_duration_pairs": <int>,
      "poorly_aligned_pairs": <int>,
      "best_constant_offset_ms": <f>,
      "unavailable": { "reason": "no_reference_word_timings" | "no_aligned_pairs",
                       "detail": "<...>" }
    },
    "intrinsic": {                        // reference-free; NOT accuracy
      "word_count": <int>, "segment_count": <int>, "speaker_count": <int>,
      "unattributed_segments": <int>,
      "non_monotonic_word_pairs": <int>, "negative_duration_words": <int>,
      "adjacent_duplicate_words": <int>, "duplicate_runs": <int>,
      "duration_ms": <int|null>
    },
    "taxonomy": [ {
      "id": "asr.duplicated_word", "class": "asr|diarization|timing",
      "status": "measured|proxy|unavailable",
      "count": <int|null>,               // null when unavailable; never 0 for unknown
      "unit": "...", "threshold": "<...>|null", "note": "..."
    } ]
  } ],
  "blend": { "permitted": false, "value": null, "why": "..." }
}"#;

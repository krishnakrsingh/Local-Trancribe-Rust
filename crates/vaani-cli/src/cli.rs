//! The clap CLI surface shared by both binaries.
//!
//! ```text
//! vaani [--debug] [--version] [benchmark <file> [--json]
//!                            | transcribe <file> [flags]
//!                            | doctor [--json]
//!                            | models {list [--json], ensure <id>, verify <id>, remove <id>}
//!                            | jobs [--json] [--limit N]
//!                            | export <transcript.json> [--format F] [--out P] [--overwrite]]
//! ```
//!
//! `--version` is clap's built-in (reports `APP_VERSION`). `--debug` enables
//! debug logging and, on the `vaani.exe` bin only, allocates a console.
//!
//! Exit codes: `0` success, `2` usage error, `3` a `VaaniError` reached the top
//! (serialised as `{"code": ...}` on stderr so scripts match on the code).

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use vaani_core::Language;
use vaani_inference::Provider;

use crate::pipeline::{DirectPipeline, Pipeline, TranscribeRequest};

/// VaaniDesk: local, private meeting transcription.
#[derive(Debug, Parser)]
#[command(name = "vaani", version = crate::APP_VERSION, about = "Local meeting transcription")]
pub struct Cli {
    /// Verbose logging. On vaani.exe, also ensures a console is attached.
    #[arg(long, global = true)]
    pub debug: bool,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

/// Parse CLI helpers shared with tests: device wire names.
pub fn parse_provider(s: &str) -> Option<Provider> {
    match s.to_ascii_lowercase().as_str() {
        "auto" => Some(Provider::Auto),
        "cpu" => Some(Provider::Cpu),
        "cuda" | "require_cuda" | "gpu" => Some(Provider::RequireCuda),
        _ => None,
    }
}

/// Parse CLI helpers shared with tests: language names (wire forms plus the
/// display names users actually type).
pub fn parse_language(s: &str) -> Option<Language> {
    if let Some(l) = Language::parse(s) {
        return Some(l);
    }
    match s.to_ascii_lowercase().as_str() {
        "hindi" | "hin" => Some(Language::Hindi),
        "english" | "eng" => Some(Language::English),
        _ => None,
    }
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Time each pipeline stage for one file. ASR is skipped (with a stable
    /// reason code) when the model is not installed.
    Benchmark {
        /// Media file to measure against.
        file: PathBuf,
        /// Print the machine-readable report (same value, pretty-printed).
        #[arg(long)]
        json: bool,
    },
    /// Transcribe one file end to end.
    Transcribe {
        /// Media file to transcribe.
        file: PathBuf,
        /// Manifest model id (default: sravaani-1.0-onnx).
        #[arg(long)]
        model: Option<String>,
        /// Compute device: auto|cpu|cuda (default: auto).
        #[arg(long, value_name = "DEVICE")]
        device: Option<String>,
        /// Language: auto|hindi|english|hinglish|other (default: auto).
        #[arg(long, value_name = "LANG")]
        language: Option<String>,
        /// Run speaker diarization (default: on).
        #[arg(long, overrides_with = "no_diarize")]
        diarize: bool,
        /// Disable speaker diarization.
        #[arg(long = "no-diarize")]
        no_diarize: bool,
        /// Where to write the transcript (file or directory).
        #[arg(long, value_name = "PATH")]
        out: Option<PathBuf>,
        /// Export format: text|markdown|srt|vtt|json|all (default: srt).
        #[arg(long, value_name = "FORMAT", default_value = "srt")]
        format: String,
        /// Overwrite an existing export file.
        #[arg(long)]
        overwrite: bool,
    },
    /// Runtime, model, and capability report. Exit code 0 when ready to
    /// transcribe, 3 otherwise.
    Doctor {
        /// Machine-readable output.
        #[arg(long)]
        json: bool,
    },
    /// Model management (download state, verification, removal).
    Models {
        #[command(subcommand)]
        command: ModelsCommands,
    },
    /// List recent jobs from the local database.
    Jobs {
        /// Machine-readable output.
        #[arg(long)]
        json: bool,
        /// Maximum rows (default: 20).
        #[arg(long, default_value_t = 20)]
        limit: u32,
    },
    /// Render a stored transcript (`vaani.transcript` JSON) to subtitle /
    /// document formats.
    Export {
        /// Transcript JSON file (as written by `transcribe --format json`).
        file: PathBuf,
        /// Format: text|markdown|srt|vtt|json|all (default: srt).
        #[arg(long, value_name = "FORMAT", default_value = "srt")]
        format: String,
        /// Output file or directory (default: beside the input).
        #[arg(long, value_name = "PATH")]
        out: Option<PathBuf>,
        /// Overwrite existing files.
        #[arg(long)]
        overwrite: bool,
    },
}

#[derive(Debug, Subcommand)]
pub enum ModelsCommands {
    /// List manifest models with install state.
    List {
        /// Machine-readable output.
        #[arg(long)]
        json: bool,
    },
    /// Ensure a model is installed (reports the install plan when missing).
    Ensure {
        /// Manifest id, e.g. sravaani-1.0-onnx.
        id: String,
    },
    /// Verify an installed model against the manifest checksums.
    Verify {
        /// Manifest id.
        id: String,
    },
    /// Remove an installed model from disk.
    Remove {
        /// Manifest id.
        id: String,
    },
}

/// Run the CLI with explicit args (bins pass `std::env::args()`).
/// Returns the process exit code: `0` ok, `2` usage, `3` runtime error.
pub fn run_cli<I, T>(args: I) -> i32
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let cli = match Cli::try_parse_from(args) {
        Ok(c) => c,
        Err(e) => {
            let _ = e.print();
            return if e.use_stderr() { 2 } else { 0 };
        }
    };
    init_logging(cli.debug);

    // THE vaani-app seam: this is the one construction site to change.
    // `DirectPipeline` -> the `vaani-app` adapter; everything below only
    // touches `&dyn Pipeline`.
    let pipeline = DirectPipeline;
    let result = dispatch(&cli, &pipeline);
    match result {
        Ok(code) => code,
        Err(e) => {
            let payload = serde_json::json!({
                "code": e.code().as_str(),
                "title": e.user.title,
                "detail": e.user.detail,
                "recovery": e.user.recovery,
            });
            eprintln!("{payload}");
            tracing::error!("{}", e.debug_chain());
            3
        }
    }
}

fn init_logging(debug: bool) {
    use tracing_subscriber::{EnvFilter, fmt};
    let level = if debug { "debug" } else { "warn" };
    let filter = EnvFilter::try_new(format!("vaani_cli={level},vaani={level}")).unwrap_or_default();
    let _ = fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

fn dispatch(cli: &Cli, pipeline: &dyn Pipeline) -> vaani_core::Result<i32> {
    let cancel = vaani_core::CancelToken::new();
    match &cli.command {
        None => {
            // `vaani` with no subcommand: the GUI bin opens the window; the
            // headless bin prints help. Disambiguated by the caller via
            // `run_cli_no_command_gui`.
            println!("VaaniDesk {}", crate::APP_VERSION);
            println!("Run `vaani doctor` to check this machine, or `vaani --help` for commands.");
            Ok(0)
        }
        Some(Commands::Benchmark { file, json }) => {
            let report = pipeline.benchmark(file, &cancel)?;
            if *json {
                println!("{}", serde_json::to_string_pretty(&report).unwrap());
            } else {
                print_benchmark_human(&report);
            }
            Ok(0)
        }
        Some(Commands::Transcribe {
            file,
            model,
            device,
            language,
            diarize: _diarize,
            no_diarize,
            out,
            format,
            overwrite,
        }) => {
            let mut req = TranscribeRequest::for_file(file.clone());
            if let Some(m) = model {
                req.model = m.clone();
            }
            if let Some(d) = device {
                req.device = parse_provider(d).ok_or_else(|| {
                    vaani_core::err::internal()
                        .with_cause(format!("unknown --device: {d} (want auto|cpu|cuda)"))
                })?;
            }
            if let Some(l) = language {
                req.language = parse_language(l).ok_or_else(|| {
                    vaani_core::err::internal().with_cause(format!(
                        "unknown --language: {l} (want auto|hindi|english|hinglish|other)"
                    ))
                })?;
            }
            // `--diarize`/`--no-diarize` override each other (last wins, via
            // `overrides_with`); absent both, diarization stays on.
            req.diarize = !*no_diarize;
            req.out = out.clone();
            req.format = format.clone();
            req.overwrite = *overwrite;
            let outcome = pipeline.transcribe(&req, &cancel)?;
            println!("job {}", outcome.job_id);
            println!("segments: {}", outcome.segments);
            match outcome.num_speakers {
                Some(n) => println!("speakers: {n}"),
                None => println!("speakers: skipped"),
            }
            for path in &outcome.exported {
                println!("exported: {}", path.display());
            }
            for (code, msg) in &outcome.warnings {
                println!("warning [{code}]: {msg}");
            }
            Ok(0)
        }
        Some(Commands::Doctor { json }) => {
            let report = crate::service::doctor();
            if *json {
                println!("{}", serde_json::to_string_pretty(&report).unwrap());
            } else {
                println!("VaaniDesk {}", report.app_version);
                println!("{}", report.summary);
                if !report.runtime.runtime_present {
                    println!("missing: {}", report.runtime.missing_dlls.join(", "));
                }
                for m in &report.models {
                    println!(
                        "model {}: {}",
                        m.id,
                        if m.available { "installed" } else { "missing" }
                    );
                }
            }
            Ok(if crate::service::doctor_is_ready(&report) {
                0
            } else {
                3
            })
        }
        Some(Commands::Models { command }) => match command {
            ModelsCommands::List { json } => {
                let rows = crate::service::models_list()?;
                if *json {
                    println!("{}", serde_json::to_string_pretty(&rows).unwrap());
                } else {
                    for m in &rows {
                        println!(
                            "{}: {}",
                            m.id,
                            if m.available { "installed" } else { &m.detail }
                        );
                    }
                }
                Ok(0)
            }
            ModelsCommands::Ensure { id } => {
                println!("{}", crate::service::models_ensure(id)?);
                Ok(0)
            }
            ModelsCommands::Verify { id } => {
                println!("{}", crate::service::models_verify(id, &cancel)?);
                Ok(0)
            }
            ModelsCommands::Remove { id } => {
                println!("{}", crate::service::models_remove(id)?);
                Ok(0)
            }
        },
        Some(Commands::Jobs { json, limit }) => {
            let rows = crate::service::jobs_list(*limit)?;
            if *json {
                println!("{}", serde_json::to_string_pretty(&rows).unwrap());
            } else {
                for j in &rows {
                    println!("{}  {}  {}  ({} words)", j.id, j.phase, j.input, j.words);
                }
            }
            Ok(0)
        }
        Some(Commands::Export {
            file,
            format,
            out,
            overwrite,
        }) => {
            let written = crate::service::export_file(file, format, out.as_deref(), *overwrite)?;
            for p in &written {
                println!("exported: {}", p.display());
            }
            Ok(0)
        }
    }
}

fn print_benchmark_human(report: &crate::pipeline::BenchmarkReport) {
    println!(
        "VaaniDesk {}  benchmark: {}",
        report.app_version, report.file
    );
    for s in &report.stages {
        match &s.detail {
            Some(d) => println!("  {:<8} {:>8} ms  {d}", s.stage, s.elapsed_ms),
            None => println!("  {:<8} {:>8} ms", s.stage, s.elapsed_ms),
        }
    }
    for s in &report.skipped {
        println!("  {:<8} skipped [{}]: {}", s.stage, s.reason_code, s.reason);
    }
    match report.rtf {
        Some(r) => println!("  RTF: {r:.3}"),
        None => println!("  RTF: n/a (ASR did not run)"),
    }
    if let Some(ram) = report.peak_ram_bytes {
        println!("  peak RAM: {} bytes", ram);
    }
}

/// Whether `run_cli` saw no subcommand. The `vaani.exe` bin treats that as
/// "open the GUI"; `vaani-cli.exe` prints the summary line instead. Exported
/// so the bins share the rule without duplicating it.
pub fn has_subcommand(args: &[String]) -> bool {
    // Any positional-or-flag beyond the program name counts as CLI mode. A
    // leading `--debug`/`--version` is CLI mode too (clap handles `--version`).
    args.len() > 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(argv: &[&str]) -> Cli {
        Cli::try_parse_from(argv).unwrap()
    }

    #[test]
    fn benchmark_parses_with_optional_json() {
        let c = parse(&["vaani", "benchmark", "a.mp4"]);
        assert!(matches!(
            c.command,
            Some(Commands::Benchmark { json: false, .. })
        ));
        let c = parse(&["vaani", "benchmark", "a.mp4", "--json"]);
        assert!(matches!(
            c.command,
            Some(Commands::Benchmark { json: true, .. })
        ));
    }

    #[test]
    fn transcribe_parses_all_flags() {
        let c = parse(&[
            "vaani",
            "transcribe",
            "a.mp4",
            "--model",
            "m",
            "--device",
            "cuda",
            "--language",
            "hindi",
            "--no-diarize",
            "--out",
            "o.srt",
            "--format",
            "srt",
            "--overwrite",
        ]);
        match c.command {
            Some(Commands::Transcribe {
                model,
                device,
                language,
                no_diarize,
                out,
                format,
                overwrite,
                ..
            }) => {
                assert_eq!(model.as_deref(), Some("m"));
                assert_eq!(device.as_deref(), Some("cuda"));
                assert_eq!(language.as_deref(), Some("hindi"));
                assert!(no_diarize);
                assert_eq!(out, Some(PathBuf::from("o.srt")));
                assert_eq!(format, "srt");
                assert!(overwrite);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn models_jobs_export_doctor_parse() {
        assert!(matches!(
            parse(&["vaani", "models", "list"]).command,
            Some(Commands::Models { .. })
        ));
        assert!(matches!(
            parse(&["vaani", "models", "ensure", "x"]).command,
            Some(Commands::Models { .. })
        ));
        assert!(matches!(
            parse(&["vaani", "jobs", "--limit", "5"]).command,
            Some(Commands::Jobs { limit: 5, .. })
        ));
        assert!(matches!(
            parse(&["vaani", "export", "t.json"]).command,
            Some(Commands::Export { .. })
        ));
        assert!(matches!(
            parse(&["vaani", "doctor", "--json"]).command,
            Some(Commands::Doctor { json: true })
        ));
    }

    #[test]
    fn device_and_language_parsing_reject_unknown() {
        assert!(parse_provider("auto").is_some());
        assert!(parse_provider("cpu").is_some());
        assert!(parse_provider("cuda").is_some());
        assert!(parse_provider("tpu").is_none());
        assert_eq!(parse_language("hindi"), Some(Language::Hindi));
        assert_eq!(parse_language("en"), Some(Language::English));
        assert_eq!(parse_language("hinglish"), Some(Language::Hinglish));
        assert!(parse_language("klingon").is_none());
    }

    #[test]
    fn global_debug_flag_parses() {
        let c = parse(&["vaani", "--debug", "doctor"]);
        assert!(c.debug);
    }

    #[test]
    fn transcribe_rejects_bad_device_and_language_by_code() {
        // Unknown --device/--language surface as internal errors with the
        // cause on the log chain (never matched by message text).
        let pipe = DirectPipeline;
        let cli = parse(&["vaani", "transcribe", "a.mp4", "--device", "tpu"]);
        let e = dispatch(&cli, &pipe).unwrap_err();
        assert!(e.debug_chain().contains("unknown --device"));
        let cli = parse(&["vaani", "transcribe", "a.mp4", "--language", "klingon"]);
        let e = dispatch(&cli, &pipe).unwrap_err();
        assert!(e.debug_chain().contains("unknown --language"));
    }

    #[test]
    fn version_flag_is_clap_builtin_not_a_subcommand() {
        // `--version` must exit 0 via clap, not fall into dispatch.
        let e = Cli::try_parse_from(["vaani", "--version"]).unwrap_err();
        assert!(!e.use_stderr());
    }
}

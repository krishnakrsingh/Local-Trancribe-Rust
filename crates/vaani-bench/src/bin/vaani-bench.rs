//! `vaani-bench <file> [--json]`: staged benchmark entry point.
//!
//! This binary backs the `vaani benchmark <file> [--json]` contract
//! (`docs/architecture/INTEGRATION_CONTRACT.md` §13) until `vaani-cli` (A8)
//! wires the shared pipeline core; the stage order, report shape and exit
//! codes are the contract, not the process name.
//!
//! Exit codes: `0` = report printed (including a graceful `skip`);
//! `2` = caller error (missing/unreadable input, bad usage). A skip is not a
//! failure, so CI stays green while the pipeline is being built.

use std::path::PathBuf;

const USAGE: &str = "\
usage: vaani-bench <file> [--json]

  <file>    meeting recording to benchmark (private fixture travels ONLY as
            this argument; it is never committed, see scripts/bench/)
  --json    print one machine-readable BenchReport JSON line to stdout
  -h/--help print this help

exit codes: 0 = report printed (ok or graceful skip); 2 = caller error.";

fn main() {
    std::process::exit(run());
}

fn run() -> i32 {
    let mut input: Option<PathBuf> = None;
    let mut json = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--json" => json = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return 0;
            }
            other if other.starts_with('-') => {
                eprintln!("vaani-bench: error: unknown flag: {other}\n{USAGE}");
                return 2;
            }
            other => {
                if input.is_some() {
                    eprintln!("vaani-bench: error: expected one <file>, got two\n{USAGE}");
                    return 2;
                }
                input = Some(PathBuf::from(other));
            }
        }
    }
    let Some(path) = input else {
        eprintln!("vaani-bench: error: missing <file> argument\n{USAGE}");
        return 2;
    };

    match vaani_bench::run_benchmark(&path) {
        Ok(report) => {
            if json {
                println!("{}", report.to_json());
            } else {
                println!("{}", report.render_table());
            }
            0
        }
        Err(e) => {
            eprintln!("vaani-bench: error: {e}");
            2
        }
    }
}

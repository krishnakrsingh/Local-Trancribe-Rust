//! `vaani-cli.exe`: the headless binary.
//!
//! Console-subsystem build: always has a console, suitable for scripts and
//! for `benchmark --json` pipelines. All behaviour lives in
//! `vaani_cli::cli::run_cli` so it is unit-testable without spawning a process.

fn main() -> ! {
    std::process::exit(vaani_cli::cli::run_cli(std::env::args()));
}

//! `vaani.exe`: the GUI launcher.
//!
//! Normal launch (no arguments) opens the Slint window with **no console** —
//! this bin builds with `#![windows_subsystem = "windows"]`, so Windows never
//! creates one. Any CLI invocation (`--version`, `--debug`, `benchmark`,
//! `transcribe`, `doctor`, `models`, `jobs`, `export`) reattaches to the
//! parent console first (`AttachConsole`), allocating a fresh console only
//! under `--debug` when there is no parent to join.
//!
//! GUI wiring belongs to `vaani-app` (A2): when it lands, the `launch_gui`
//! stub below becomes `vaani_app::run()` (one function, same seam as
//! `Pipeline` in `vaani_cli::pipeline`). Until then it reports the seam
//! instead of fabricating a window.

#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

fn main() -> ! {
    let args: Vec<String> = std::env::args().collect();
    if !vaani_cli::cli::has_subcommand(&args) {
        launch_gui();
    }
    // CLI mode: rejoin a console before any output.
    let debug = args.iter().any(|a| a == "--debug");
    if !vaani_cli::console::attach_parent() && debug {
        vaani_cli::console::alloc_console();
    }
    std::process::exit(vaani_cli::cli::run_cli(args));
}

/// Open the GUI. Interim stub: `vaani-app` owns the orchestrator + backend.
///
/// The stub attaches a console and says so, rather than exiting silently (a
/// silent no-window exit reads as "the installer is broken" in every support
/// channel). Exit code 0: the launcher itself worked.
fn launch_gui() -> ! {
    vaani_cli::console::attach_parent();
    println!("VaaniDesk {}", vaani_cli::APP_VERSION);
    println!("The desktop window is wired by vaani-app (not yet landed).");
    println!(
        "Meanwhile: `vaani doctor` checks this machine, `vaani transcribe <file>` transcribes it."
    );
    std::process::exit(0);
}

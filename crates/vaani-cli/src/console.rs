//! Console attachment for the GUI bin.
//!
//! `vaani.exe` builds with `windows_subsystem = "windows"`, so Windows starts
//! it with no console at all. When it is invoked as a CLI (`vaani transcribe
//! ...`, `vaani --version`, ...) it must rejoin a console before writing any
//! output:
//!
//! 1. `AttachConsole(ATTACH_PARENT_PROCESS)` — the normal case when run from
//!    `cmd.exe`, PowerShell, or Windows Terminal. Output lands in the shell
//!    the user already has; no window flashes.
//! 2. `AllocConsole` — only when `--debug` is present and there is no parent
//!    console (e.g. launched from Explorer with arguments). A fresh console
//!    window appears for the debug log.
//!
//! `vaani-cli.exe` is a console-subsystem binary and never calls this.
//!
//! ```text
//! vaani-app seam: none. This module is A8-owned glue with no pipeline role.
//! ```

/// Rejoin the parent console if there is one.
///
/// Returns `true` when a console is available for output afterwards (either it
/// was already ours, as in `vaani-cli.exe`, or the attach succeeded).
pub fn attach_parent() -> bool {
    #[cfg(windows)]
    {
        use windows::Win32::System::Console::{ATTACH_PARENT_PROCESS, AttachConsole};
        // `unsafe` justification: calling a Win32 console API with a documented
        // constant argument; no pointers, no retained state. This is the only
        // `unsafe` in the crate, and it cannot exist anywhere else — the GUI
        // bin must attach without pulling in an inference-sized dependency.
        unsafe { AttachConsole(ATTACH_PARENT_PROCESS).is_ok() }
    }
    #[cfg(not(windows))]
    {
        true
    }
}

/// Allocate a fresh console window. Only called under `--debug` when
/// [`attach_parent`] failed (no parent console to join).
pub fn alloc_console() {
    #[cfg(windows)]
    {
        use windows::Win32::System::Console::AllocConsole;
        // `unsafe` justification: same as above — argument-free Win32 call.
        let _ = unsafe { AllocConsole() };
    }
}

/// Peak working-set size of this process in bytes, if the platform can say.
///
/// Used by `benchmark` for `peak_ram_bytes`. `None` when the query fails, in
/// which case the benchmark report carries `null` rather than a fabricated
/// number (the no-fake-numbers rule in `docs/architecture/INTEGRATION_CONTRACT.md` §3).
pub fn peak_working_set_bytes() -> Option<u64> {
    #[cfg(windows)]
    {
        use windows::Win32::System::ProcessStatus::{
            GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
        };
        use windows::Win32::System::Threading::GetCurrentProcess;
        // `unsafe` justification: raw-handle Win32 query filling a
        // caller-owned struct; the struct is zero-initialised and `cb` is set
        // from its size, per the API contract.
        unsafe {
            let handle = GetCurrentProcess();
            let mut counters = std::mem::zeroed::<PROCESS_MEMORY_COUNTERS>();
            counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
            GetProcessMemoryInfo(handle, &mut counters, counters.cb)
                .is_ok()
                .then_some(counters.PeakWorkingSetSize as u64)
        }
    }
    #[cfg(not(windows))]
    {
        None
    }
}

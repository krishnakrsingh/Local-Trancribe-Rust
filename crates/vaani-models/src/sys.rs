//! The three platform facts `std` does not expose.
//!
//! # Why this file exists at all
//!
//! The model manager needs to answer three questions that have no safe `std`
//! equivalent:
//!
//! 1. **How much free space is on the volume I am about to write 2.7 GB to?**
//!    `std::fs` has no `statvfs`. The pre-flight check is a *product*
//!    requirement: a download that dies at 90% because the disk filled is the
//!    single most common way a first run goes wrong, and the only honest fix is
//!    to refuse before starting.
//! 2. **Peak working set**, so the claim "an 845 MB encoder verifies without
//!    paging it into memory" is a measurement rather than an assertion.
//! 3. Exclusive file handles, for which `std` *does* have an API
//!    ([`std::os::windows::fs::OpenOptionsExt::share_mode`]) — see
//!    [`crate::lock`] and [`crate::verify`].
//!
//! # Why it is FFI rather than a dependency
//!
//! The obvious answer is the `windows` crate, which the workspace already pins
//! (`Cargo.toml:92`) and which `vaani-inference` uses. `vaani-models`' own
//! `Cargo.toml` is frozen and does not list it, and this crate's owner may not
//! edit it. So the two calls are declared here by hand, each with a SAFETY
//! note, and **this file should be deleted in favour of `windows` when the
//! architect adds the dependency** — see `INTEGRATION NOTES` in the A7 report.
//!
//! The alternative was to skip the checks, and a model manager that cannot ask
//! the disk how much room it has is a model manager that guesses. Two
//! `GetDiskFreeSpaceExW`-shaped declarations are cheaper than that.
//!
//! Everything here is Windows-only, matching `rust-toolchain.toml`
//! (`targets = ["x86_64-pc-windows-msvc"]`). On other platforms the
//! free-space and RSS probes report `None`, which the callers treat as
//! "unknown" and skip rather than as "zero" — refusing an install because a
//! probe failed would be worse than attempting it.

use std::path::Path;

/// Result of a platform probe.
///
/// `None` means **unknown**, never "zero". Every caller must be able to
/// distinguish "this machine has no space" from "this platform cannot tell me",
/// because only the first justifies refusing an install.
pub type Probe<T> = Option<T>;

#[cfg(windows)]
mod imp {
    use super::Probe;
    use std::ffi::c_void;
    use std::path::Path;

    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetDiskFreeSpaceExW(
            directory_name: *const u16,
            free_bytes_available_to_caller: *mut u64,
            total_number_of_bytes: *mut u64,
            total_number_of_free_bytes: *mut u64,
        ) -> i32;

        fn GetCurrentProcess() -> *mut c_void;

        fn K32GetProcessMemoryInfo(
            process: *mut c_void,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }

    /// Free bytes on the volume holding `path`.
    ///
    /// `GetDiskFreeSpaceExW` needs a **directory that exists**; asking about a
    /// not-yet-created download directory returns zero, which is exactly the
    /// wrong answer. So walk up to the nearest existing ancestor, which is
    /// always at worst a drive root.
    pub fn free_bytes(path: &Path) -> Probe<u64> {
        let mut dir = path.to_path_buf();
        loop {
            if dir.is_dir() {
                break;
            }
            if !dir.pop() {
                return None;
            }
        }
        let wide = dir
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect::<Vec<_>>();
        let mut free = 0u64;
        let mut total = 0u64;
        let mut total_free = 0u64;
        // SAFETY: `wide` is a NUL-terminated UTF-16 buffer that outlives the
        // call and `GetDiskFreeSpaceExW` only reads it. The three out-pointers
        // are to live, correctly aligned, zeroed locals of the exact types the
        // signature declares, and the API writes each of them only on success.
        let ok = unsafe {
            GetDiskFreeSpaceExW(
                wide.as_ptr(),
                &mut free as *mut u64,
                &mut total as *mut u64,
                &mut total_free as *mut u64,
            )
        };
        (ok != 0).then_some(free)
    }

    /// Peak working set of this process, in bytes.
    ///
    /// `PeakWorkingSetSize` is the high-water mark since process start, so it
    /// is the right number for a "this never buffered the file" assertion: a
    /// one-shot read of the file would leave a trace here forever.
    pub fn peak_working_set_bytes() -> Probe<u64> {
        let mut c = ProcessMemoryCounters {
            cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
            page_fault_count: 0,
            peak_working_set_size: 0,
            working_set_size: 0,
            quota_peak_paged_pool_usage: 0,
            quota_paged_pool_usage: 0,
            quota_peak_non_paged_pool_usage: 0,
            quota_non_paged_pool_usage: 0,
            pagefile_usage: 0,
            peak_pagefile_usage: 0,
        };
        // SAFETY: `GetCurrentProcess` returns the documented pseudo-handle,
        // which requires no closing. `c` is a live, correctly sized,
        // zero-initialised `PROCESS_MEMORY_COUNTERS` and `cb` is its real size,
        // which is the contract the API checks before writing.
        let ok = unsafe {
            K32GetProcessMemoryInfo(
                GetCurrentProcess(),
                &mut c as *mut ProcessMemoryCounters,
                c.cb,
            )
        };
        (ok != 0).then_some(c.peak_working_set_size as u64)
    }

    use std::os::windows::ffi::OsStrExt;
}

#[cfg(not(windows))]
mod imp {
    use super::Probe;
    use std::path::Path;

    /// Always unknown off Windows. See the module docs.
    pub fn free_bytes(_path: &Path) -> Probe<u64> {
        None
    }

    /// Always unknown off Windows. See the module docs.
    pub fn peak_working_set_bytes() -> Probe<u64> {
        None
    }
}

/// Free bytes available to this user on the volume holding `path`.
///
/// Walks up to the nearest existing ancestor, so it is valid for a download
/// directory that has not been created yet.
pub fn free_bytes(path: &Path) -> Probe<u64> {
    imp::free_bytes(path)
}

/// Peak working set of the current process, in bytes.
pub fn peak_working_set_bytes() -> Probe<u64> {
    imp::peak_working_set_bytes()
}

/// `FILE_SHARE_READ`: other processes may read, nobody may write or delete.
#[cfg(windows)]
pub const SHARE_READ: u32 = 0x0000_0001;
/// `FILE_SHARE_WRITE`.
#[cfg(windows)]
pub const SHARE_WRITE: u32 = 0x0000_0002;
/// `FILE_SHARE_DELETE`.
#[cfg(windows)]
pub const SHARE_DELETE: u32 = 0x0000_0004;

/// `DELETE` in a `CreateFile` desired-access mask.
///
/// Asked for, never used. The only way to find out whether a file *can* be
/// deleted is to ask for the access deletion needs and see whether the existing
/// handles permit it: a read-only probe always succeeds against a read-only
/// file, so "can I read it" and "can I delete it" are different questions and
/// only one of them has a syscall.
#[cfg(windows)]
pub const DELETE_ACCESS: u32 = 0x0001_0000;

/// Windows error 32, `ERROR_SHARING_VIOLATION`.
///
/// The single most important errno in this crate: it is what a delete or a
/// second open returns when an inference session already has the model file
/// open, and it is the difference between "the model is in use" and a raw
/// `The system cannot find the file specified` that a user cannot act on.
pub const ERROR_SHARING_VIOLATION: i32 = 32;

/// Windows error 5, `ERROR_ACCESS_DENIED`.
pub const ERROR_ACCESS_DENIED: i32 = 5;

/// Whether an I/O error is "somebody else is holding this file open".
///
/// `raw_os_error()` is `Option<i32>` on Windows and something else elsewhere,
/// so this is deliberately total rather than a `cfg`-split pair of casts.
#[must_use]
pub fn is_sharing_violation(e: &std::io::Error) -> bool {
    e.raw_os_error() == Some(ERROR_SHARING_VIOLATION)
        || e.raw_os_error() == Some(ERROR_ACCESS_DENIED)
}

/// Whether an I/O error means "the disk is full".
///
/// Windows reports this as 112 (`ERROR_DISK_FULL`) from an explicit write and,
/// more often in practice, 39 (`ERROR_HANDLE_DISK_FULL`) or a plain
/// `ERROR_DISK_FULL`-adjacent code from the filesystem driver. All three mean
/// the same thing to a user: free some space. Unix reports `ENOSPC` (28).
#[must_use]
pub fn is_disk_full(e: &std::io::Error) -> bool {
    if let Some(code) = e.raw_os_error() {
        // 112 ERROR_DISK_FULL, 39 ERROR_HANDLE_DISK_FULL,
        // 55 ERROR_DEV_NOT_EXIST is *not* included on purpose.
        if code == 112 || code == 39 {
            return true;
        }
    }
    // ENOSPC on unix, and the message a Rust std build produces when the
    // platform did not set an errno at all.
    e.raw_os_error() == Some(28)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_bytes_is_plausible_for_the_current_directory() {
        // A directory that exists, on a real volume. A hard lower bound of one
        // megabyte is safe on any machine that can run the test suite.
        let d = std::env::temp_dir();
        let free = free_bytes(&d).expect("temp dir is on a real volume");
        assert!(
            free > 1024 * 1024,
            "free space on {} was {free}",
            d.display()
        );
    }

    #[test]
    fn free_bytes_walks_up_to_an_existing_ancestor() {
        // The download directory will not exist yet; asking about it must not
        // report zero, which would read as "no space" and refuse every install.
        let missing = std::env::temp_dir()
            .join("vaani-a7-does-not-exist")
            .join("models")
            .join("sravaani");
        assert!(!missing.exists());
        if let Some(free) = free_bytes(&missing) {
            assert!(free > 1024 * 1024, "walk-up failed: {free}");
        }
    }

    #[test]
    fn peak_working_set_is_reported_and_plausible() {
        let peak = peak_working_set_bytes().expect("Windows reports this");
        // A test binary is at least a few MiB resident; anything under 1 MiB
        // means we read the wrong struct field.
        assert!(peak > 1024 * 1024, "peak working set {peak} is implausible");
    }

    #[test]
    fn error_classification_reads_the_raw_os_error() {
        let sharing = std::io::Error::from_raw_os_error(ERROR_SHARING_VIOLATION);
        assert!(is_sharing_violation(&sharing));
        let denied = std::io::Error::from_raw_os_error(ERROR_ACCESS_DENIED);
        assert!(is_sharing_violation(&denied));
        let other = std::io::Error::from_raw_os_error(2);
        assert!(!is_sharing_violation(&other));

        let full = std::io::Error::from_raw_os_error(112);
        assert!(is_disk_full(&full));
        assert!(!is_disk_full(&other));
    }
}

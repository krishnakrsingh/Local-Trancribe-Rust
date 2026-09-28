//! Two processes must not install the same model at the same time, and a model
//! file in use must not be deletable.
//!
//! # The concurrency scheme
//!
//! An install is: download N files, verify each, move it into place. Two
//! windows doing that to the same directory produce two `.part` files writing to
//! the same bytes, and whichever renames last wins — leaving a model that is
//! half one download and half another, with a receipt saying it is fine.
//!
//! So an install holds a **lock file** for its whole duration:
//!
//! 1. `create_new` on `<model dir>/.vaani-install.lock`. `O_CREAT|O_EXCL` is
//!    atomic on NTFS *and* on every POSIX filesystem, so exactly one of two
//!    racing callers creates it. This is the whole mutual-exclusion argument;
//!    there is no window between "check" and "take".
//! 2. The creating handle is opened with `share_mode = 0`, so it is held
//!    **exclusively**: any other open of that file, from any process, fails with
//!    `ERROR_SHARING_VIOLATION`.
//! 3. The loser, on `AlreadyExists`, probes the file with a normal
//!    `File::open`. If the probe *fails* with a sharing violation, a live holder
//!    has it and the loser reports [`LockError::Held`]. If the probe *succeeds*,
//!    no process holds the file, so the lock is a crash leftover: it is removed
//!    and the acquisition is retried exactly once.
//!
//! Step 3 is the part worth arguing for. The usual alternative is a pid in the
//! lock file plus a "is that process alive" check, which needs a platform
//! liveness API, races against pid reuse, and is wrong on a machine where the
//! checker itself is not permitted to open the process. Here **the OS is the
//! arbiter**: a handle dies with its process, so "nobody can open it" is a
//! sound proof of "nobody holds it" with no heuristic and no timestamp.
//!
//! Off Windows the same `create_new` gives the mutual exclusion; staleness falls
//! back to an age threshold, which is weaker and documented as such. The product
//! ships Windows only (`rust-toolchain.toml`).
//!
//! # Atomic install ordering
//!
//! The lock stops two installs colliding. It does not stop a *crash* leaving a
//! half-written file where the loader will find it, so the ordering is separate
//! and deliberate:
//!
//! ```text
//! <model dir>/.partial/<name>.<pid>.part   written, then hashed
//! <model dir>/<name>                      fs::rename, only after the hash matches
//! <model dir>/.installed.json             the receipt, written last
//! ```
//!
//! `fs::rename` is atomic on NTFS within a volume, and the temp file is in the
//! same directory so it is always the same volume. The loader therefore never
//! sees a partially written `encoder.fp16.onnx`: it sees either the previous
//! file or the new one, and the model is only "installed" once the receipt
//! exists. A cancelled install leaves a `.part` file, which the next attempt
//! resumes from.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::sys;
use serde::{Deserialize, Serialize};

/// The directory lock files live in, under a model **root**.
///
/// Deliberately *not* inside the model directory. `remove` holds the lock while
/// it deletes the model directory, and on Windows a file cannot be deleted while
/// a handle to it is held without `FILE_SHARE_DELETE` — so a lock inside the
/// directory makes the directory undeletable, which a test found the hard way.
/// One level up, the lock describes the *operation* rather than the directory's
/// contents, and survives the directory it was protecting.
pub const LOCK_DIR: &str = ".vaani-locks";

/// A lock file's name, which is the model directory name plus this suffix.
pub const LOCK_SUFFIX: &str = ".lock";

/// The directory partial downloads live in.
pub const PARTIAL_DIR: &str = ".partial";

/// The receipt written once a model is complete.
pub const RECEIPT_FILE: &str = ".installed.json";

/// A lock left behind by a process that died is considered stale after this.
///
/// Only used off Windows; see the module docs.
pub const STALE_AFTER: Duration = Duration::from_secs(30 * 60);

/// Who holds a lock. Written into the lock file for a developer's benefit; it is
/// never read back for correctness, because the OS handle is the authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HolderInfo {
    /// The process that took the lock.
    pub pid: u32,
    /// The model it is installing.
    pub model_id: String,
    /// Unix seconds when the lock was taken.
    pub started_unix: u64,
}

impl HolderInfo {
    /// This process, now.
    #[must_use]
    pub fn here(model_id: impl Into<String>) -> Self {
        Self {
            pid: std::process::id(),
            model_id: model_id.into(),
            started_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
        }
    }
}

/// Why a lock could not be taken.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LockError {
    /// Another process or thread holds it. Not a failure of anything: the loser
    /// of a race has to be told so clearly and then stop.
    #[error(
        "{model_id} is already being installed by another VaaniDesk window or process (lock: {path})"
    )]
    Held {
        /// The model.
        model_id: String,
        /// The lock file.
        path: PathBuf,
    },
    /// The lock file could not be created for another reason.
    #[error("could not create the install lock at {path}: {message}")]
    Io {
        /// The lock file.
        path: PathBuf,
        /// The operating system's message.
        message: String,
    },
}

impl LockError {
    /// Project onto the application error taxonomy.
    ///
    /// A lost install race is `ResourceBusy`, not a storage failure and not a
    /// cancellation: another window or process holds the model, and waiting or
    /// closing the other copy fixes it, which neither of the old codes says.
    #[must_use]
    pub fn to_vaani_error(&self) -> vaani_core::VaaniError {
        match self {
            Self::Held { model_id, .. } => {
                vaani_core::VaaniError::new(vaani_core::UserFacingError::new(
                    vaani_core::ErrorCode::ResourceBusy,
                    "Already being installed.",
                    format!("{model_id} is already being installed in another window or process."),
                ))
                .with_cause(self)
                .with_recovery(
                    "Wait for the other installation to finish, or close the other window, \
                 then try again.",
                )
            }
            Self::Io { .. } => vaani_core::err::storage_failed()
                .with_cause(self)
                .with_recovery("Check that the models folder is writable, then try again."),
        }
    }
}

/// An exclusive claim on one model's install.
///
/// Dropping it releases the claim and removes the lock file. There is no
/// `unlock`: a lock that had to be released explicitly is a lock that can be
/// leaked.
#[derive(Debug)]
pub struct InstallLock {
    path: PathBuf,
    // `Option` so `Drop` can close the handle *before* removing the file. The
    // handle is held with `share_mode = 0`, which means deleting the file while
    // it is open fails with `ERROR_SHARING_VIOLATION` — so the order is not
    // tidiness, it is the only order that works.
    file: Option<File>,
    info: HolderInfo,
}

impl InstallLock {
    /// Take the lock for the model directory `dir_name` under `root`.
    ///
    /// Retries **once** against a stale lock. A second `AlreadyExists` after the
    /// retry means somebody is genuinely installing, because a live holder would
    /// have blocked the probe.
    ///
    /// # Errors
    ///
    /// [`LockError::Held`] when another process or thread has it, and
    /// [`LockError::Io`] when the lock directory cannot be created.
    pub fn acquire(root: &Path, dir_name: &str, model_id: &str) -> Result<InstallLock, LockError> {
        let lock_dir = root.join(LOCK_DIR);
        std::fs::create_dir_all(&lock_dir).map_err(|e| LockError::Io {
            path: lock_dir.clone(),
            message: e.to_string(),
        })?;
        let path = lock_dir.join(format!("{dir_name}{LOCK_SUFFIX}"));
        let info = HolderInfo::here(model_id);
        for attempt in 0..2 {
            match create_exclusive(&path) {
                Ok(mut file) => {
                    // Best effort: a lock that cannot record its holder is still
                    // a working lock, and refusing to take one over a diagnostic
                    // field would be absurd.
                    let _ = writeln!(file, "{}", serde_json::to_string(&info).unwrap_or_default());
                    return Ok(InstallLock {
                        path: path.clone(),
                        file: Some(file),
                        info: info.clone(),
                    });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    if attempt == 0 && lock_is_stale(&path) {
                        let _ = std::fs::remove_file(&path);
                        continue;
                    }
                    return Err(LockError::Held {
                        model_id: model_id.to_string(),
                        path,
                    });
                }
                Err(e) => {
                    return Err(LockError::Io {
                        path,
                        message: e.to_string(),
                    });
                }
            }
        }
        Err(LockError::Io {
            path,
            message: "lock acquisition did not converge".into(),
        })
    }

    /// The lock file's path, for a message that has to name it.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Who holds it, which is always this process.
    pub fn holder(&self) -> &HolderInfo {
        &self.info
    }
}

impl Drop for InstallLock {
    fn drop(&mut self) {
        // Close the exclusive handle first: with `share_mode = 0` the file
        // cannot be deleted while we hold it, so removing before closing would
        // fail and leave a lock that only a reboot clears.
        drop(self.file.take());
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Create the lock file, exclusively.
fn create_exclusive(path: &Path) -> std::io::Result<File> {
    let mut opts = OpenOptions::new();
    opts.write(true).create_new(true);
    exclusive(&mut opts);
    opts.open(path)
}

/// Apply `share_mode = 0` on Windows; a no-op elsewhere.
#[cfg(windows)]
fn exclusive(opts: &mut OpenOptions) {
    use std::os::windows::fs::OpenOptionsExt;
    // 0 = FILE_SHARE_NONE. We hold it exclusively so that a probe from any
    // other process or thread fails, which is how liveness is detected.
    opts.share_mode(0);
}

#[cfg(not(windows))]
fn exclusive(_opts: &mut OpenOptions) {}

/// Whether a lock file has no live holder.
fn lock_is_stale(path: &Path) -> bool {
    match File::open(path) {
        // Opened: nothing holds it exclusively, so nobody is installing.
        Ok(_) => true,
        Err(e) if sys::is_sharing_violation(&e) => false,
        Err(_) => age(path).is_none_or(|a| a > STALE_AFTER),
    }
}

fn age(path: &Path) -> Option<Duration> {
    let m = std::fs::metadata(path).ok()?;
    let t = m.modified().ok()?;
    SystemTime::now().duration_since(t).ok()
}

/// A handle that stops a model file being deleted or overwritten while it is
/// open.
///
/// On Windows the handle is taken with `share_mode = FILE_SHARE_READ`: other
/// processes may read the weights (a second window may want to load the same
/// model) but nobody may delete or write them. That is exactly the guarantee the
/// requirement asks for, and it is enforced by the kernel rather than by a
/// check-then-act race in our code.
///
/// Elsewhere this is a plain read handle and the guarantee does not hold,
/// because POSIX unlink semantics allow deleting an open file. The product
/// ships Windows only.
#[derive(Debug)]
pub struct FileGuard {
    path: PathBuf,
    _file: File,
}

impl FileGuard {
    /// Take a guard on `path`, which must exist, allowing other readers.
    pub fn open(path: &Path) -> std::io::Result<FileGuard> {
        let mut opts = OpenOptions::new();
        opts.read(true);
        guard_sharing(&mut opts);
        let file = opts.open(path)?;
        Ok(FileGuard {
            path: path.to_path_buf(),
            _file: file,
        })
    }

    /// Take a guard that permits **no** other open at all.
    ///
    /// This models the case that actually breaks verification: another process
    /// — an ONNX Runtime load, a backup tool, a virus scanner, an installer
    /// writing the file — holding the file with no sharing. A read-shared guard
    /// ([`FileGuard::open`]) deliberately does *not* stop a second reader,
    /// because two windows reading the same model is normal and must work.
    pub fn open_exclusive(path: &Path) -> std::io::Result<FileGuard> {
        let mut opts = OpenOptions::new();
        opts.read(true);
        exclusive(&mut opts);
        let file = opts.open(path)?;
        Ok(FileGuard {
            path: path.to_path_buf(),
            _file: file,
        })
    }

    /// The guarded path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(windows)]
fn guard_sharing(opts: &mut OpenOptions) {
    use std::os::windows::fs::OpenOptionsExt;
    // FILE_SHARE_READ only: readable by anyone, deletable by nobody.
    opts.share_mode(sys::SHARE_READ);
}

#[cfg(not(windows))]
fn guard_sharing(_opts: &mut OpenOptions) {}

/// Whether a path can be deleted right now, without deleting it.
///
/// A `metadata` is not enough: on Windows a file with an open handle is still
/// there and still reports its size. Nor is a read-open, which succeeds against
/// a file another process is holding open *for reading* — which is exactly the
/// case this has to detect. Deletion needs the `DELETE` access, and `DELETE` is
/// granted only when **every** existing handle on the file shares delete, so
/// asking for it and immediately closing is the honest test. Nothing is ever
/// marked for deletion here: the handle is opened and dropped.
///
/// Off Windows this always answers "not in use", because POSIX unlink succeeds
/// on an open file and the question has no equivalent.
#[must_use]
pub fn is_in_use(path: &Path) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        let mut opts = OpenOptions::new();
        opts.access_mode(sys::DELETE_ACCESS)
            .share_mode(sys::SHARE_READ | sys::SHARE_WRITE | sys::SHARE_DELETE);
        match opts.open(path) {
            Ok(handle) => {
                drop(handle);
                false
            }
            Err(e) => sys::is_sharing_violation(&e),
        }
    }
    #[cfg(not(windows))]
    {
        let _ = path;
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("vaani-a7-lock-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn lock_path(root: &Path) -> PathBuf {
        root.join(LOCK_DIR).join("m1.lock")
    }

    #[test]
    fn a_lock_is_exclusive_within_one_process() {
        let d = dir("excl");
        let a = InstallLock::acquire(&d, "m1", "m1").expect("first acquire");
        let b = InstallLock::acquire(&d, "m1", "m1");
        assert!(matches!(b, Err(LockError::Held { .. })), "{b:?}");
        assert_eq!(a.path(), lock_path(&d));
        assert_eq!(a.holder().pid, std::process::id());
    }

    #[test]
    fn the_lock_is_released_on_drop() {
        let d = dir("drop");
        {
            let _a = InstallLock::acquire(&d, "m1", "m1").unwrap();
            assert!(lock_path(&d).exists());
            assert!(InstallLock::acquire(&d, "m1", "m1").is_err());
        }
        assert!(!lock_path(&d).exists(), "drop must remove the file");
        let _b = InstallLock::acquire(&d, "m1", "m1").expect("re-acquirable after drop");
    }

    #[test]
    fn a_stale_lock_from_a_dead_holder_is_reclaimed() {
        // Simulates a process that died mid-install: the file is there and
        // nothing holds it, which is exactly the state the probe detects.
        let d = dir("stale");
        std::fs::create_dir_all(d.join(LOCK_DIR)).unwrap();
        std::fs::write(lock_path(&d), "{}").unwrap();
        let a = InstallLock::acquire(&d, "m1", "m1").expect("a stale lock must not deadlock");
        assert!(a.path().exists());
    }

    #[test]
    fn a_held_lock_names_the_model_and_the_path() {
        let d = dir("msg");
        let _a = InstallLock::acquire(&d, "sravaani", "sravaani").unwrap();
        let e = InstallLock::acquire(&d, "sravaani", "sravaani").unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains("sravaani"), "{msg}");
        assert!(msg.contains("already being installed"), "{msg}");
        let v = e.to_vaani_error();
        assert_eq!(v.code(), vaani_core::ErrorCode::ResourceBusy);
        assert!(v.user.recovery.is_some());
        assert!(v.to_string().contains("Already being installed"));
    }

    #[test]
    fn a_guard_makes_a_file_undeletable_and_the_removal_error_sayable() {
        let d = dir("guard");
        let f = d.join("encoder.fp16.onnx");
        std::fs::write(&f, b"weights").unwrap();
        assert!(!is_in_use(&f));

        let g = FileGuard::open(&f).expect("guard");
        assert!(is_in_use(&f), "a guarded file must report as in use");
        let err = std::fs::remove_file(&f).unwrap_err();
        assert!(
            sys::is_sharing_violation(&err),
            "expected a sharing violation, got {err:?}"
        );
        assert!(f.exists(), "the file must survive a blocked delete");
        drop(g);
        assert!(!is_in_use(&f));
        std::fs::remove_file(&f).unwrap();
    }

    #[test]
    fn the_lock_lives_beside_the_model_directory_not_inside_it() {
        // The regression this shape exists to prevent: a lock inside the model
        // directory makes the directory undeletable, because the handle held on
        // the lock file forbids deleting it.
        let d = dir("beside");
        let model = d.join("m1");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(model.join("weights.bin"), b"w").unwrap();
        let lock = InstallLock::acquire(&d, "m1", "m1").unwrap();
        assert!(
            !lock.path().starts_with(&model),
            "the lock is inside the model dir"
        );
        std::fs::remove_dir_all(&model).expect("the model dir must be deletable while locked");
        assert!(lock.path().exists());
    }

    #[test]
    fn a_read_shared_guard_still_lets_a_second_reader_in() {
        // Two windows reading the same model is normal and must work. The guard
        // denies writes and deletes, not reads; a guard that denied reads would
        // make a second VaaniDesk window unusable for no benefit.
        let d = dir("shared");
        let f = d.join("weights.bin");
        std::fs::write(&f, b"weights").unwrap();
        let a = FileGuard::open(&f).unwrap();
        let b = FileGuard::open(&f).expect("a second reader must be allowed");
        drop(b);
        drop(a);
    }

    #[test]
    fn an_exclusive_guard_stops_every_other_open_including_a_read() {
        // This is the case that breaks verification, and it is why the two
        // guards are different types of call rather than one flag.
        let d = dir("excl2");
        let f = d.join("weights.bin");
        std::fs::write(&f, b"weights").unwrap();
        let a = FileGuard::open_exclusive(&f).unwrap();
        let err = std::fs::File::open(&f).unwrap_err();
        assert!(sys::is_sharing_violation(&err), "{err:?}");
        assert!(is_in_use(&f));
        drop(a);
        assert!(std::fs::File::open(&f).is_ok());
    }

    #[test]
    fn two_threads_racing_for_one_model_produce_exactly_one_winner() {
        let d = dir("race");
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
        let winners = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let losers = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let d = d.clone();
            let barrier = std::sync::Arc::clone(&barrier);
            let winners = std::sync::Arc::clone(&winners);
            let losers = std::sync::Arc::clone(&losers);
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                match InstallLock::acquire(&d, "m1", "m1") {
                    Ok(lock) => {
                        // Hold it long enough that a real overlap is certain.
                        std::thread::sleep(Duration::from_millis(50));
                        winners.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        drop(lock);
                    }
                    Err(LockError::Held { .. }) => {
                        losers.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    }
                    Err(e) => panic!("unexpected {e:?}"),
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(winners.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(losers.load(std::sync::atomic::Ordering::SeqCst), 7);
        assert!(!lock_path(&d).exists());
    }

    #[test]
    fn an_unwritable_location_reports_an_io_error_not_a_contention_error() {
        let e = InstallLock::acquire(Path::new("Z:/definitely/not/here"), "m1", "m1").unwrap_err();
        assert!(matches!(e, LockError::Io { .. }), "{e:?}");
        assert_eq!(
            e.to_vaani_error().code(),
            vaani_core::ErrorCode::StorageFailed
        );
    }

    #[test]
    fn holder_info_round_trips_as_json_for_a_bug_report() {
        let h = HolderInfo::here("m1");
        let j = serde_json::to_string(&h).unwrap();
        let back: HolderInfo = serde_json::from_str(&j).unwrap();
        assert_eq!(back, h);
        assert!(h.started_unix > 1_600_000_000, "a plausible timestamp");
    }
}

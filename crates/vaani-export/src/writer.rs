//! Getting the bytes to disk, safely.
//!
//! # Atomic
//!
//! An export is written to a temporary file in the **same directory** and then
//! renamed over the target. `rename` is atomic within a filesystem, so a reader
//! of the target path sees either the complete previous file or the complete new
//! one — never a half-written transcript. That matters more than it sounds: the
//! previous file is the user's only copy of work that took twenty minutes, and a
//! crash during the write must not destroy it.
//!
//! Writing into the same directory (rather than the system temp directory) is
//! what makes the rename atomic: a rename across filesystems is a copy, and a
//! copy is not atomic. The temporary name starts with a dot and ends in `.tmp`,
//! so a user who lists the folder mid-export sees a clearly transient file, and
//! a crash leaves at most one such file behind rather than a plausible-looking
//! half-transcript.
//!
//! # Never outside the chosen directory
//!
//! A transcript's *name* comes from a media file's name, and a media file can be
//! called `../../evil.txt`. Every name is sanitised ([`safe_file_name`]) and the
//! resolved path is then checked against the directory the user chose
//! ([`assert_inside`]) before a single byte is written. Two independent checks,
//! because this is the one place in the product where a bug is a security bug.
//!
//! # Never silently
//!
//! An existing file is never replaced unless the caller passed
//! [`ExportOptions::overwrite`]. There is an unavoidable window between that
//! check and the rename, in which another process could create the file; the
//! window is microseconds and the consequence is replacing a file the user
//! created in the same moment, which is acceptable, and is documented here
//! rather than pretended away.

use std::io::Write;
use std::path::{Component, Path, PathBuf};

use vaani_core::Result;

use crate::error;

/// Windows device names. A file called `CON.txt` cannot be created on Windows,
/// and the failure mode is a confusing one, so they are renamed rather than
/// attempted.
const RESERVED: &[&str] = &[
    "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
    "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
];

/// The longest stem we will produce. A file name longer than this fails on some
/// filesystems, and no media file has a 200-character name.
const MAX_STEM: usize = 120;

/// Turn any string into a single, safe path component.
///
/// The rules, in order:
///
/// 1. take the **last** component, so `../../evil` and `C:\x\y` both become
///    their last segment rather than escaping;
/// 2. reject `.` and `..` outright;
/// 3. replace anything Windows forbids (`<>:"|?*`, control characters) with `_`;
/// 4. rename a Windows device name (`CON`, `NUL`, `COM1`, ...);
/// 5. trim trailing dots and spaces, which Windows silently strips, so the file
///    that appears on disk is the file that was asked for;
/// 6. truncate to [`MAX_STEM`] characters.
///
/// Returns `"transcript"` when nothing usable is left, so a caller always gets
/// something writeable.
pub fn safe_file_name(name: &str) -> String {
    // 1. Last component, for both separators, whatever the platform.
    let last = name.rsplit(['/', '\\']).next().unwrap_or(name).trim();
    if last.is_empty() || last == "." || last == ".." {
        // Fall back to the first usable component of the original, in case the
        // name was `../..`-shaped all the way down.
        let first = name
            .split(['/', '\\'])
            .map(str::trim)
            .find(|s| !s.is_empty() && *s != "." && *s != "..");
        return match first {
            Some(f) => sanitise(f),
            None => "transcript".to_string(),
        };
    }
    sanitise(last)
}

fn sanitise(component: &str) -> String {
    let mut out = String::with_capacity(component.len());
    for ch in component.chars() {
        let bad = matches!(ch, '<' | '>' | ':' | '"' | '|' | '?' | '*')
            || (ch as u32) < 0x20
            || ch == '\u{7f}';
        out.push(if bad { '_' } else { ch });
    }
    // 5. Trailing dots and spaces are invisible and unrepresentable on Windows.
    let trimmed = out.trim_end_matches(['.', ' ']).to_string();
    let mut out = if trimmed.is_empty() {
        "transcript".to_string()
    } else {
        trimmed
    };
    // 4. A device name, with or without an extension.
    let stem = out
        .split('.')
        .next()
        .unwrap_or(out.as_str())
        .to_ascii_uppercase();
    if RESERVED.contains(&stem.as_str()) {
        out = format!("_{out}");
    }
    // 6. Length.
    if out.chars().count() > MAX_STEM {
        out = out.chars().take(MAX_STEM).collect();
    }
    out
}

/// Fail unless `path` is directly inside `dir`.
///
/// Both are canonicalised first, so `..`, symlinks and `.` all resolve to the
/// same answer they would resolve to for the user. The parent is compared, not
/// the whole path, because the file itself need not exist yet.
pub fn assert_inside(dir: &Path, path: &Path) -> Result<()> {
    let wanted = canonical(dir);
    let parent = match path.parent().filter(|p| !p.as_os_str().is_empty()) {
        Some(p) => p,
        None => return Err(error::outside_target(&display(path))),
    };
    let got = canonical(parent);
    if got == wanted {
        Ok(())
    } else {
        Err(error::outside_target(&display(path)))
    }
}

/// Canonicalise, falling back to a lexically cleaned path when the target does
/// not exist yet.
///
/// `canonicalize` fails on a path that does not exist, and a directory that does
/// not exist is a different failure that the write itself reports with a better
/// message. Cleaning the path lexically still resolves `..` and `.`, which is
/// what this check needs, so the check is not silently skipped just because a
/// component is missing.
fn canonical(path: &Path) -> PathBuf {
    match path.canonicalize() {
        Ok(p) => p,
        Err(_) => {
            let mut out = PathBuf::new();
            for component in path.components() {
                match component {
                    Component::CurDir => {}
                    Component::ParentDir => {
                        out.pop();
                    }
                    other => out.push(other.as_os_str()),
                }
            }
            out
        }
    }
}

fn display(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string_lossy().into_owned())
}

/// Whether the name would be refused as unsafe, for tests and for a UI that
/// wants to warn before the user commits.
pub fn is_safe_component(name: &str) -> bool {
    let safe = safe_file_name(name);
    Path::new(&safe)
        .components()
        .all(|c| matches!(c, Component::Normal(_)))
}

/// Write `contents` to `path`, atomically.
///
/// Returns the number of bytes written. Refuses to replace an existing file
/// unless `overwrite`.
///
/// # Errors
///
/// If the file exists and `overwrite` is false, if the write or the rename
/// fails, or if the temporary file cannot be cleaned up afterwards (which is
/// logged, because failing the export over a stray temp file would be worse than
/// leaving it).
pub fn write_atomic(path: &Path, contents: &str, overwrite: bool) -> Result<u64> {
    if path.exists() && !overwrite {
        return Err(error::already_exists(path));
    }
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "export".to_string());
    let temp = dir.join(temp_name(&stem));

    let written = {
        let mut file = match std::fs::File::create(&temp) {
            Ok(f) => f,
            Err(e) => {
                let _ = std::fs::remove_file(&temp);
                return Err(error::write_failed(path, e));
            }
        };
        let bytes = contents.len() as u64;
        let result = file
            .write_all(contents.as_bytes())
            .and_then(|()| file.flush())
            // Sync before the rename: without it, a power cut can leave the
            // renamed file empty on some filesystems even though the write
            // "succeeded".
            .and_then(|()| file.sync_all());
        match result {
            Ok(()) => bytes,
            Err(e) => {
                drop(file);
                let _ = std::fs::remove_file(&temp);
                return Err(error::write_failed(path, e));
            }
        }
    };

    if let Err(e) = std::fs::rename(&temp, path) {
        let _ = std::fs::remove_file(&temp);
        return Err(error::write_failed(path, e));
    }
    Ok(written)
}

/// Distinguishes concurrent writes **within** one process.
///
/// The process id alone is not enough: two exports in the same process — the
/// "export all five formats" button, or two windows — would pick the same
/// temporary name, and the loser's rename would fail with a file that has
/// already been moved. Caught by the concurrent-write test, and it would have
/// been a rare, confusing "the system cannot find the file specified" in
/// production.
static WRITE_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The temporary name for one write: hidden, `.tmp`, and unique per call.
fn temp_name(stem: &str) -> String {
    let n = WRITE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!(".{stem}.{}-{n}.tmp", std::process::id())
}

/// The temporary file an atomic write to `path` would use.
///
/// Exposed so a test can prove the rename happened rather than a copy, and so the
/// uniqueness rule is stated in one place.
pub fn temp_path_for(path: &Path) -> PathBuf {
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "export".to_string());
    dir.join(temp_name(&stem))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vaani_storage::testing::TempDir;

    #[test]
    fn traversal_attempts_are_reduced_to_the_last_component() {
        // Whatever the input, what comes out is a single component with no
        // directory syntax in it.
        for hostile in [
            "../../evil",
            "..\\..\\evil",
            "/etc/passwd",
            "C:\\Windows\\System32\\evil",
            "....//....//evil",
        ] {
            let safe = safe_file_name(hostile);
            assert!(!safe.contains(".."), "{hostile} kept a dot-dot");
            assert!(!safe.contains('/'), "{hostile} kept a slash");
            assert!(!safe.contains('\\'), "{hostile} kept a backslash");
            assert!(is_safe_component(&safe), "{hostile} produced {safe}");
        }
        // The last component is what survives, whatever the path was.
        assert_eq!(safe_file_name("../../evil"), "evil");
        assert_eq!(safe_file_name("..\\..\\evil"), "evil");
        assert_eq!(safe_file_name("/etc/passwd"), "passwd");
        assert_eq!(safe_file_name("C:\\Windows\\System32\\evil"), "evil");
    }

    #[test]
    fn illegal_characters_are_replaced() {
        assert_eq!(safe_file_name("a<b>c:d\"e|f?g*h"), "a_b_c_d_e_f_g_h");
        assert_eq!(safe_file_name("bell\u{7}.txt"), "bell_.txt");
        assert_eq!(safe_file_name("ok name.m4a"), "ok name.m4a");
    }

    #[test]
    fn windows_device_names_are_renamed() {
        for reserved in ["CON", "nul", "COM1", "lpt9", "aux.txt"] {
            let safe = safe_file_name(reserved);
            assert!(safe.starts_with('_'), "{reserved} became {safe}");
        }
    }

    #[test]
    fn trailing_dots_and_spaces_are_trimmed() {
        assert_eq!(safe_file_name("name..."), "name");
        assert_eq!(safe_file_name("name   "), "name");
    }

    #[test]
    fn a_long_name_is_truncated() {
        let long = "x".repeat(400);
        assert_eq!(safe_file_name(&long).chars().count(), MAX_STEM);
    }

    #[test]
    fn an_unusable_name_still_produces_something() {
        for empty in ["", ".", "..", "///", "../.."] {
            let safe = safe_file_name(empty);
            assert!(!safe.is_empty(), "{empty:?} produced nothing");
            assert!(is_safe_component(&safe), "{empty:?} produced {safe}");
        }
    }

    #[test]
    fn safe_components_contain_no_path_syntax() {
        for hostile in ["../x", "a/b", "a\\b", "C:x", ".hidden/../x"] {
            assert!(is_safe_component(&safe_file_name(hostile)), "{hostile}");
        }
    }

    #[test]
    fn assert_inside_accepts_a_child_and_refuses_an_escape() {
        let dir = TempDir::new("inside");
        let inside = dir.join("out.srt");
        assert_inside(dir.path(), &inside).unwrap();
        for escape in ["../out.srt", "sub/../../out.srt", "..\\out.srt"] {
            let candidate = dir.join(escape);
            assert!(assert_inside(dir.path(), &candidate).is_err(), "{escape}");
        }
    }

    #[test]
    fn a_symlink_out_of_the_directory_is_refused() {
        // Only meaningful where symlinks can be created; skipped elsewhere
        // rather than failing, because the assertion is the same either way.
        let dir = TempDir::new("symlink");
        let outside = TempDir::new("symlink-target");
        let link = dir.join("link");
        if std::os::windows::fs::symlink_dir(outside.path(), &link).is_err() {
            return;
        }
        let candidate = link.join("out.srt");
        assert!(
            assert_inside(dir.path(), &candidate).is_err(),
            "a symlink must not be a way out"
        );
    }

    #[test]
    fn an_atomic_write_creates_the_file_and_leaves_no_temp_behind() {
        let dir = TempDir::new("atomic");
        let path = dir.join("out.txt");
        let n = write_atomic(&path, "hello", false).unwrap();
        assert_eq!(n, 5);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello");
        let leftovers: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n != "out.txt")
            .collect();
        assert!(leftovers.is_empty(), "left behind {leftovers:?}");
    }

    #[test]
    fn an_existing_file_is_not_replaced_without_permission() {
        let dir = TempDir::new("no-clobber");
        let path = dir.join("out.txt");
        write_atomic(&path, "first", false).unwrap();
        let e = write_atomic(&path, "second", false).unwrap_err();
        assert_eq!(e.code(), vaani_core::ErrorCode::ExportFailed);
        assert!(e.user.recovery.unwrap().contains("different name"));
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "first",
            "the previous file must survive"
        );
        // With permission it is replaced, atomically.
        write_atomic(&path, "second", true).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "second");
    }

    #[test]
    fn the_temporary_file_lives_beside_the_target() {
        // Beside, not in %TEMP%: a rename across filesystems is a copy, and a
        // copy is not atomic.
        let dir = TempDir::new("beside");
        let path = dir.join("out.txt");
        let temp = temp_path_for(&path);
        assert_eq!(temp.parent().unwrap(), dir.path());
        assert!(temp.to_string_lossy().ends_with(".tmp"));
    }

    #[test]
    fn writing_into_a_missing_directory_is_a_clear_error() {
        let dir = TempDir::new("missing-dir");
        let path = dir.join("nope/deeper/out.txt");
        let e = write_atomic(&path, "x", false).unwrap_err();
        assert_eq!(e.code(), vaani_core::ErrorCode::ExportFailed);
        assert!(e.user.detail.contains("out.txt"));
    }

    #[test]
    fn a_failed_write_leaves_the_previous_file_intact() {
        // A directory where the file should be: the rename fails, the temp file
        // is cleaned up, and the error names the target.
        let dir = TempDir::new("failed-write");
        let path = dir.join("out.txt");
        std::fs::create_dir(&path).unwrap();
        assert!(write_atomic(&path, "x", true).is_err());
        let leftovers: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "left behind {leftovers:?}");
    }
}

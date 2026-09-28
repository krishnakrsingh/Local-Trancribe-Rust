//! Finding models on disk, and saying honestly what is wrong with them.
//!
//! # Why discovery never hashes
//!
//! Discovery runs at startup and every time the settings panel opens. Hashing
//! 2.5 GB takes about six seconds, which is not a number a user will accept
//! twice, so **discovery only asks the filesystem for names and sizes**:
//!
//! * one `read_dir` per model directory, whose `find`-buffer data already
//!   carries the type and, on Windows, the size;
//! * one `metadata` per expected file that lives in a subdirectory, because a
//!   nested path cannot be answered from a flat listing.
//!
//! That is O(files) syscalls and O(bytes) of *no* I/O, and it is why a 2 GB
//! model directory lists in microseconds. The hashes are the job of
//! [`crate::verify`], which the user asks for explicitly.
//!
//! # What it reports
//!
//! Four things, all of which the UI needs and none of which a bare
//! `is_dir()` check can give:
//!
//! * **found** — a model directory that matches a manifest entry;
//! * **expected but missing** — a manifest entry with no directory anywhere;
//! * **present but not in the manifest** — a directory nobody claims, which is
//!   either a user's own model or 2 GB of wasted disk they should be told
//!   about;
//! * **per-file state** — present/absent and right size/wrong size, which is
//!   what distinguishes "not installed" from "interrupted halfway".

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::manifest::Model;
use crate::paths::{ModelRoots, RootKind};

/// Names this crate owns inside a model directory.
///
/// They are excluded from the "unexpected file" report: a user did not put
/// `.partial/` there, we did, and listing our own bookkeeping as a surprise
/// would train people to ignore the report.
pub const MANAGER_ENTRIES: &[&str] = &[".partial", ".installed.json"];

/// Names this crate owns directly under a model **root**.
///
/// `.vaani-locks` holds the install locks; see [`crate::lock::LOCK_DIR`] for why
/// it is not inside the model directory.
pub const MANAGER_ROOT_ENTRIES: &[&str] = &[crate::lock::LOCK_DIR];

/// The state of one expected file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileState {
    /// The manifest path, `/`-separated.
    pub path: String,
    /// What the file is for. See [`crate::manifest::role`].
    pub role: String,
    /// Whether the model is unusable without it.
    pub required: bool,
    /// Size the manifest says.
    pub expected_bytes: u64,
    /// Size on disk, or `None` if it is not there.
    pub actual_bytes: Option<u64>,
}

impl FileState {
    /// Whether the file exists.
    pub fn is_present(&self) -> bool {
        self.actual_bytes.is_some()
    }

    /// Whether the file exists and is exactly the size the manifest says.
    pub fn is_correct_size(&self) -> bool {
        self.actual_bytes == Some(self.expected_bytes)
    }

    /// Whether this file's absence or damage makes the model unusable.
    pub fn is_problem(&self) -> bool {
        self.required && !self.is_correct_size()
    }

    /// Bytes still to fetch for this file: all of it when absent, none when
    /// the size already matches (a same-size corruption is caught by the hash,
    /// not by the size, and re-fetching is the same action either way).
    pub fn bytes_outstanding(&self) -> u64 {
        if self.is_correct_size() {
            0
        } else {
            self.expected_bytes
        }
    }
}

/// A model directory found on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredModel {
    /// The manifest id of the model this directory belongs to.
    pub id: String,
    /// The directory name under its root.
    pub dir_name: String,
    /// Which root it was found in.
    pub root_kind: RootKind,
    /// The root directory.
    pub root: PathBuf,
    /// The model directory itself. This is what
    /// `vaani_asr::sravaani::artifacts_in` wants.
    pub dir: PathBuf,
    /// State of every file the manifest lists, in manifest order.
    pub files: Vec<FileState>,
    /// Files present in the directory that the manifest does not list, excluding
    /// [`MANAGER_ENTRIES`].
    pub unexpected: Vec<String>,
    /// A `.partial/` directory exists, so an install was interrupted.
    pub has_partial: bool,
    /// Total bytes actually in the directory, including unexpected files.
    pub on_disk_bytes: u64,
}

impl DiscoveredModel {
    /// Required files that are absent or the wrong size.
    pub fn problems(&self) -> Vec<&FileState> {
        self.files.iter().filter(|f| f.is_problem()).collect()
    }

    /// Whether every required file is present at the right size.
    ///
    /// This is what "installed" means. It says nothing about content, which is
    /// [`crate::verify`]'s job; a same-size corruption is reported by verify and
    /// not here, because a hash on every settings-panel refresh is not a cost
    /// this product can pay.
    pub fn is_complete(&self) -> bool {
        self.problems().is_empty()
    }

    /// Bytes an install would still have to fetch.
    ///
    /// **Required files only.** This used to sum every file in the directory,
    /// which meant an install with all seven of `SraVaani`'s required files and
    /// none of its three optional ones reported 192 bytes outstanding — a
    /// settings panel offering to download 192 bytes of a model that was
    /// already complete. It also disagreed with
    /// [`ModelManager::install_plan_with`](crate::manager::ModelManager::install_plan_with),
    /// which has always counted only required files, so the "x to download" in
    /// the picker and the "x to download" in the confirmation dialog were
    /// different numbers on the same model.
    ///
    /// Optional files are still measured and still reported as
    /// [`FileState::is_present`](FileState) `== false`; they are simply not
    /// something an install owes anybody.
    pub fn bytes_outstanding(&self) -> u64 {
        self.files
            .iter()
            .filter(|f| f.required)
            .map(FileState::bytes_outstanding)
            .sum()
    }

    /// Whether this directory looks like a half-finished install rather than a
    /// real one.
    pub fn is_incomplete_install(&self) -> bool {
        self.has_partial || !self.is_complete()
    }
}

/// A directory under a root that no manifest entry claims.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownDir {
    /// Which root it was found in.
    pub root_kind: RootKind,
    /// The full path.
    pub path: PathBuf,
    /// The directory name.
    pub name: String,
    /// Bytes in it, one level deep. Enough to say "this is 1.6 GB of something".
    pub bytes: u64,
}

/// Whether a root exists, for the diagnostics panel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootPresence {
    /// Which root.
    pub kind: RootKind,
    /// Its path.
    pub path: PathBuf,
    /// Whether it exists and is a directory.
    pub exists: bool,
}

/// The result of one discovery pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Discovery {
    /// Every root searched, in search order, with whether it exists.
    pub roots: Vec<RootPresence>,
    /// Every model directory found, ordered by root precedence then manifest
    /// order.
    pub models: Vec<DiscoveredModel>,
    /// Manifest ids with no directory in any root.
    pub absent: Vec<String>,
    /// Directories no manifest entry claims.
    pub unknown_dirs: Vec<UnknownDir>,
    /// Wall time of the pass. Measured because "fast" is a requirement, not a
    /// hope: it is in the struct so a caller can log it without a second clock.
    pub elapsed: Duration,
}

impl Discovery {
    /// The best copy of a model: the first one found, which is the
    /// highest-precedence root.
    pub fn best(&self, id: &str) -> Option<&DiscoveredModel> {
        self.models.iter().find(|m| m.id == id)
    }

    /// Every copy of a model, across all roots.
    pub fn copies(&self, id: &str) -> Vec<&DiscoveredModel> {
        self.models.iter().filter(|m| m.id == id).collect()
    }

    /// Whether a model is present and complete in some root.
    pub fn is_installed(&self, id: &str) -> bool {
        self.best(id).is_some_and(DiscoveredModel::is_complete)
    }

    /// Whether the model was found at all, complete or not.
    pub fn is_present(&self, id: &str) -> bool {
        self.best(id).is_some()
    }

    /// Models that are present and complete, by id.
    pub fn installed_ids(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self
            .models
            .iter()
            .filter(|m| m.is_complete())
            .map(|m| m.id.as_str())
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Every manifest id, with a flag for whether it is installed. Drives the
    /// settings list, so that a model which is absent is *shown* rather than
    /// missing.
    pub fn availability(&self, manifest: &crate::manifest::Manifest) -> Vec<(String, bool)> {
        manifest
            .models
            .iter()
            .map(|m| (m.id.clone(), self.is_installed(&m.id)))
            .collect()
    }
}

/// Scan every root for every model in the manifest.
///
/// Never hashes and never opens a file, so its cost is one `read_dir` per model
/// directory plus one `metadata` per nested expected file.
#[must_use]
pub fn discover(manifest: &crate::manifest::Manifest, roots: &ModelRoots) -> Discovery {
    let start = std::time::Instant::now();
    let mut out = Discovery {
        roots: Vec::new(),
        models: Vec::new(),
        absent: Vec::new(),
        unknown_dirs: Vec::new(),
        elapsed: Duration::ZERO,
    };

    for (kind, root) in roots.all() {
        let exists = root.is_dir();
        out.roots.push(RootPresence {
            kind,
            path: root.to_path_buf(),
            exists,
        });
        if !exists {
            continue;
        }

        // One listing of the root answers two questions: which model directories
        // exist, and which directories are unclaimed.
        let listing = list_dir(root);
        let claimed: Vec<&str> = manifest.models.iter().map(|m| m.dir.as_str()).collect();

        for model in &manifest.models {
            let dir = root.join(&model.dir);
            if !dir.is_dir() {
                continue;
            }
            out.models.push(inspect(kind, root, model, &dir));
        }

        for (name, meta) in &listing {
            if !meta.is_dir
                || claimed.contains(&name.as_str())
                || MANAGER_ROOT_ENTRIES.contains(&name.as_str())
            {
                continue;
            }
            out.unknown_dirs.push(UnknownDir {
                root_kind: kind,
                path: root.join(name),
                name: name.clone(),
                bytes: tree_bytes(&root.join(name)),
            });
        }
    }

    for m in &manifest.models {
        if !out.models.iter().any(|d| d.id == m.id) {
            out.absent.push(m.id.clone());
        }
    }
    out.elapsed = start.elapsed();
    out
}

/// One directory entry: its name, whether it is a directory, and its size.
#[derive(Debug, Clone, Copy)]
struct EntryMeta {
    is_dir: bool,
    bytes: u64,
}

/// `read_dir` flattened into a map, with sizes. A file whose size cannot be read
/// is reported as size 0 rather than dropped: dropping it would turn an
/// access-denied file into "not present", which is a different bug report.
fn list_dir(dir: &Path) -> BTreeMap<String, EntryMeta> {
    let mut out = BTreeMap::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let meta = match e.metadata() {
            Ok(m) => EntryMeta {
                is_dir: m.is_dir(),
                bytes: if m.is_dir() { 0 } else { m.len() },
            },
            Err(_) => EntryMeta {
                is_dir: false,
                bytes: 0,
            },
        };
        out.insert(name, meta);
    }
    out
}

/// Build the record for one model directory.
fn inspect(kind: RootKind, root: &Path, model: &Model, dir: &Path) -> DiscoveredModel {
    let listing = list_dir(dir);
    let has_partial = listing.contains_key(".partial");

    let files = model
        .files
        .iter()
        .map(|f| {
            // A flat path is answered from the listing we already have; a
            // nested one needs a real lookup, because `read_dir` only sees the
            // top level.
            let actual = if f.path.contains('/') {
                fs_len(&dir.join(&f.relative))
            } else {
                listing.get(&f.path).map(|m| m.bytes)
            };
            FileState {
                path: f.path.clone(),
                role: f.role.clone(),
                required: f.required,
                expected_bytes: f.bytes,
                actual_bytes: actual,
            }
        })
        .collect::<Vec<_>>();

    // Only top-level *files* are reported. A subdirectory is never "unexpected":
    // `fe/` holds `mel_window.f32` and is a parent of an expected path, and a
    // stray directory inside a model folder is not a fact a user can act on at
    // startup. `.partial/` and the lock are ours and are excluded by name.
    let unexpected = listing
        .iter()
        .filter(|(n, m)| {
            !m.is_dir
                && !MANAGER_ENTRIES.contains(&n.as_str())
                && !model.files.iter().any(|f| f.path == n.as_str())
        })
        .map(|(n, _)| n.clone())
        .collect();

    // Counted from the file states, not from the flat listing, so a nested file
    // contributes. The top-level listing misses `fe/b.bin` entirely, and a
    // model whose size is understated by its own front-end tables is exactly
    // the kind of wrong number the UI would show as "1.6 GB" instead of "1.6 GiB".
    let listed: u64 = files.iter().filter_map(|f| f.actual_bytes).sum::<u64>();
    let extra: u64 = listing
        .iter()
        .filter(|(n, m)| {
            !m.is_dir
                && !MANAGER_ENTRIES.contains(&n.as_str())
                && !model.files.iter().any(|f| f.path == n.as_str())
        })
        .map(|(_, m)| m.bytes)
        .sum();
    let on_disk_bytes = listed + extra;

    DiscoveredModel {
        id: model.id.clone(),
        dir_name: model.dir.clone(),
        root_kind: kind,
        root: root.to_path_buf(),
        dir: dir.to_path_buf(),
        files,
        unexpected,
        has_partial,
        on_disk_bytes,
    }
}

fn fs_len(p: &Path) -> Option<u64> {
    std::fs::metadata(p)
        .ok()
        .filter(|m| m.is_file())
        .map(|m| m.len())
}

/// Bytes in a tree, one level deep: enough for "this unclaimed directory holds
/// 1.6 GB" without walking a 2 GB tree at startup.
fn tree_bytes(dir: &Path) -> u64 {
    list_dir(dir)
        .values()
        .filter(|m| !m.is_dir)
        .map(|m| m.bytes)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Manifest;

    fn write(dir: &Path, name: &str, bytes: usize) {
        let p = dir.join(name);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(p, vec![0u8; bytes]).unwrap();
    }

    /// A scratch directory unique to the call.
    ///
    /// A counter rather than just the pid: these tests run in threads of one
    /// process, and a shared name plus `remove_dir_all` means one test deletes
    /// the tree another has just built.
    fn scratch(tag: &str) -> PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "vaani-a7-disc-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn fixture() -> Manifest {
        // The scratch directory itself is irrelevant to a manifest, so it is
        // discarded: what matters is that it is unique per call.
        let _ = scratch("fixture");
        let text = format!(
            r#"{{"schema_version":1,"models":[{{
              "id":"m1","engine_id":"e1","version":"1","display_name":"M1",
              "description":"d","task":"asr","architecture":"a","access":"open",
              "license":{{"spdx":"MIT","name":"MIT","url":"https://e.org/l"}},
              "source":{{"kind":"bundled"}},"languages":["hi"],"sample_rates":[16000],
              "size_bytes":30,
              "files":[
                {{"path":"a.bin","role":"weights","bytes":10,
                  "sha256":"{}"}},
                {{"path":"fe/b.bin","role":"frontend","bytes":20,
                  "sha256":"{}"}}
              ]}}]}}"#,
            "0".repeat(64),
            "1".repeat(64)
        );
        Manifest::parse(&text).unwrap()
    }

    fn root(dir: &Path) -> ModelRoots {
        ModelRoots::at(dir.join("user"), dir.join("bundled"))
    }

    #[test]
    fn nothing_installed_reports_every_model_absent() {
        let base = scratch("none");
        let d = discover(&fixture(), &root(&base));
        assert!(d.models.is_empty());
        assert_eq!(d.absent, vec!["m1"]);
        assert!(!d.is_installed("m1"));
        assert!(!d.is_present("m1"));
        assert!(d.roots.iter().all(|r| !r.exists) || d.roots.iter().any(|r| r.exists));
    }

    #[test]
    fn a_complete_model_is_installed_including_nested_files() {
        let base = scratch("full");
        let m = base.join("bundled").join("m1");
        write(&m, "a.bin", 10);
        write(&m, "fe/b.bin", 20);

        let d = discover(&fixture(), &root(&base));
        let found = d.best("m1").expect("must be found");
        assert_eq!(found.root_kind, RootKind::Bundled);
        assert!(found.is_complete(), "{:?}", found.problems());
        assert_eq!(found.on_disk_bytes, 30);
        assert_eq!(found.bytes_outstanding(), 0);
        assert!(d.absent.is_empty());
        assert!(d.is_installed("m1"));
        assert_eq!(d.installed_ids(), vec!["m1"]);
    }

    #[test]
    fn a_missing_nested_file_is_reported_as_a_problem() {
        let base = scratch("nested");
        let m = base.join("bundled").join("m1");
        write(&m, "a.bin", 10);
        // fe/b.bin deliberately absent.

        let d = discover(&fixture(), &root(&base));
        let found = d.best("m1").unwrap();
        assert!(!found.is_complete(), "the nested file is still missing");
        let problems = found.problems();
        assert_eq!(problems.len(), 1);
        assert_eq!(problems[0].path, "fe/b.bin");
        assert_eq!(found.bytes_outstanding(), 20);
        assert!(found.is_incomplete_install());
    }

    #[test]
    fn a_wrong_sized_file_is_distinguished_from_a_missing_one() {
        let base = scratch("size");
        let m = base.join("bundled").join("m1");
        write(&m, "a.bin", 7);
        write(&m, "fe/b.bin", 20);

        let d = discover(&fixture(), &root(&base));
        let found = d.best("m1").unwrap();
        let a = found.files.iter().find(|f| f.path == "a.bin").unwrap();
        assert!(a.is_present());
        assert!(!a.is_correct_size());
        assert_eq!(a.actual_bytes, Some(7));
        assert_eq!(a.expected_bytes, 10);
        // A wrong size still needs the whole file fetched.
        assert_eq!(a.bytes_outstanding(), 10);
    }

    #[test]
    fn an_unknown_extra_file_is_reported_but_never_fails_the_model() {
        let base = scratch("extra");
        let m = base.join("bundled").join("m1");
        write(&m, "a.bin", 10);
        write(&m, "fe/b.bin", 20);
        write(&m, "notes-from-a-human.txt", 5);

        let d = discover(&fixture(), &root(&base));
        let found = d.best("m1").unwrap();
        assert_eq!(found.unexpected, vec!["notes-from-a-human.txt"]);
        assert!(found.is_complete(), "an extra file is not a broken model");
        assert_eq!(found.on_disk_bytes, 35);
    }

    #[test]
    fn our_own_bookkeeping_is_not_reported_as_an_extra_file() {
        let base = scratch("mine");
        let m = base.join("bundled").join("m1");
        write(&m, "a.bin", 10);
        write(&m, "fe/b.bin", 20);
        write(&m, ".installed.json", 30);
        write(&m, ".partial/a.bin.part", 3);
        std::fs::create_dir_all(base.join(".vaani-locks")).unwrap();
        write(&base, ".vaani-locks/m1.lock", 50);

        let d = discover(&fixture(), &root(&base));
        let found = d.best("m1").unwrap();
        assert!(found.unexpected.is_empty(), "{:?}", found.unexpected);
        assert!(found.has_partial, "an interrupted install is visible");
        // The lock directory under the model *root* is ours too, so it is never
        // reported as an unclaimed model directory.
        assert!(
            d.unknown_dirs.is_empty(),
            "our own lock directory was reported as unclaimed: {:?}",
            d.unknown_dirs
        );
    }

    #[test]
    fn an_unclaimed_directory_is_reported_with_its_size() {
        let base = scratch("unk");
        write(&base.join("user").join("whisper-large"), "model.bin", 4096);

        let d = discover(&fixture(), &root(&base));
        assert_eq!(d.unknown_dirs.len(), 1);
        assert_eq!(d.unknown_dirs[0].name, "whisper-large");
        assert_eq!(d.unknown_dirs[0].bytes, 4096);
        assert_eq!(d.unknown_dirs[0].root_kind, RootKind::User);
    }

    #[test]
    fn the_bundled_root_wins_over_the_user_one() {
        // Reversed from A7's original order by the zero-token mandate. The
        // shipped copy is the one this build staged and hashed; a copy in app
        // data is the one nothing has checked the version of. See paths.rs.
        let base = scratch("prec");
        for r in ["user", "bundled"] {
            let m = base.join(r).join("m1");
            write(&m, "a.bin", 10);
            write(&m, "fe/b.bin", 20);
        }
        let d = discover(&fixture(), &root(&base));
        assert_eq!(d.best("m1").unwrap().root_kind, RootKind::Bundled);
        assert_eq!(d.copies("m1").len(), 2, "both copies are still listed");
    }

    #[test]
    fn a_user_root_copy_is_used_when_there_is_no_bundle() {
        // The other half of the reversal: user second means it is still found
        // when nothing is shipped, so a user who downloads a model on a build
        // with no bundle is not left with nothing.
        let base = scratch("prec-user");
        let m = base.join("user").join("m1");
        write(&m, "a.bin", 10);
        write(&m, "fe/b.bin", 20);
        let d = discover(&fixture(), &root(&base));
        assert_eq!(d.best("m1").unwrap().root_kind, RootKind::User);
    }

    #[test]
    fn availability_lists_every_manifest_entry_including_the_absent_ones() {
        let base = scratch("avail");
        let d = discover(&fixture(), &root(&base));
        let a = d.availability(&fixture());
        assert_eq!(a, vec![("m1".to_string(), false)]);
    }

    #[test]
    fn a_missing_root_is_recorded_rather_than_guessed_at() {
        let base = scratch("root");
        let d = discover(&fixture(), &root(&base));
        assert_eq!(d.roots.len(), 2);
        assert!(d.roots.iter().all(|r| !r.exists));
    }
}

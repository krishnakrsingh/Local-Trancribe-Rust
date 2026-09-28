//! Where models live, and how a manifest file path becomes a real one.
//!
//! # Three roots, searched in a fixed order
//!
//! | root | what it is | why it exists |
//! |---|---|---|
//! | [`RootKind::Bundled`] | `<app>/models/`, i.e. beside the installed executable | the distribution ships the weights. This is the zero-token, zero-download path and it is searched **first**, so a first-run user is never offered a download for something already on their disk. |
//! | [`RootKind::User`] | `%LOCALAPPDATA%\VaaniDesk\models` on Windows | where a *download* goes. Under app data, never the repo, never next to the executable: an installed app's directory is not writable by a standard user. |
//! | [`RootKind::Configured`] | whatever the user or `VAANI_MODEL_DIRS` names | a portable install on a slow drive, a network share, a second library. |
//!
//! # Why bundled is searched first (this reverses A7's original order)
//!
//! A7 put the user root first, on the reasoning that "a user who has deliberately
//! installed a model wants *that* one". That reasoning is sound for a
//! *download*, and it is still where downloads go — [`ModelRoots::install_target`]
//! is unchanged and still returns the user directory.
//!
//! It is wrong for a *shipped bundle*, for two measured reasons:
//!
//! 1. **A stale user copy silently outranks a verified shipped one.** The
//!    manifest's `version` field is a string; nothing checks that a copy on
//!    disk was installed by the build that is asking. A user who ran an earlier
//!    `VaaniDesk`, downloaded `SraVaani`, then installed the offline build gets
//!    the old copy, and the only symptom is a worse transcript.
//! 2. **It is a dead end on a clean machine.** A first-run user with no user
//!    directory at all would be told to download 887 MB of weights that are
//!    already sitting in the next folder, on a first launch that is supposed to
//!    require no network.
//!
//! A user who genuinely wants a *different* or *newer* model puts it in a
//! [`RootKind::Configured`] root, which is still searched last and still wins
//! over nothing.
//!
//! # The bundled search order
//!
//! [`bundled_search_order`] returns this list, in this order, and it is a
//! **tested function** rather than a chain of `if`s so that a change to the
//! order is a change to a list and shows up in a diff. See that function for the
//! reasoning per entry; the summary is:
//!
//! | # | location | when it exists |
//! |---|---|---|
//! | 1 | `VAANI_ARTIFACTS_ROOT` | always wins, exists or not — an explicit override is a statement, not a guess |
//! | 2 | parent of `VAANI_ARTEFACTS` | ditto; the model directory itself, as `vaani-asr` spells it |
//! | 3 | `<exe dir>/models/` | the installed distribution |
//! | 4 | `<exe dir>/../models/` | a portable ZIP whose executable lives in `bin/` |
//! | 5 | the path in `<exe dir>/portable.txt` | a portable install pointed at models on another volume |
//! | 6 | `<repo>/artifacts/` | the developer checkout, found by walking up from the executable and then from `CARGO_MANIFEST_DIR` |
//!
//! # Environment overrides
//!
//! | variable | effect |
//! |---|---|
//! | `VAANI_MODELS_DIR` | replaces the user root |
//! | `VAANI_ARTIFACTS` | the **model** directory used by `vaani_asr::sravaani::default_dir()`. This crate needs the *parent* of it, so that is what it takes. Honoured so a dev pointing the ASR crate somewhere else gets the same answer here. |
//! | `VAANI_ARTIFACTS_ROOT` | replaces the bundled root outright; wins over `VAANI_ARTEFACTS` |
//! | `VAANI_MODEL_DIRS` | `;`-separated list of extra [`RootKind::Configured`] roots, searched last |
//!
//! # Why this duplicates `vaani_asr::sravaani::default_dir()`
//!
//! Because `vaani-asr` is frozen and `vaani-models` may not depend on it (and
//! the reverse dependency would be worse: the ASR crate should not need the
//! model manager to find its own weights). The two functions are tested against
//! the same fixture tree, and the duplication is called out in
//! `INTEGRATION NOTES` so the architect can move the constant into
//! `vaani-core` when convenient.
//!
//! **This is now a known divergence, not just a duplication.**
//! `vaani_asr::sravaani::default_dir()` (`sravaani.rs:732`) resolves only
//! `VAANI_ARTEFACTS` and then the compile-time `<repo>/artifacts/sravaani`, and
//! knows nothing about `<exe dir>/models`. That is harmless in production
//! because the pipeline never calls it: `vaani_cli::pipeline` resolves through
//! [`crate::manager::ModelManager::resolve`] and passes
//! `loc.dir()` to `artifacts_in`. It is *not* harmless for the frozen ASR
//! tests, which do call it — see `INTEGRATION NOTES`.

use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Environment variable that replaces the user model root.
pub const ENV_USER_DIR: &str = "VAANI_MODELS_DIR";
/// Environment variable naming a model directory, matching `vaani-asr`.
pub const ENV_ARTEFACTS_MODEL_DIR: &str = "VAANI_ARTEFACTS";
/// Environment variable that replaces the bundled artefact root.
pub const ENV_ARTEFACTS_ROOT: &str = "VAANI_ARTIFACTS_ROOT";
/// Environment variable holding extra search roots, `;`-separated.
pub const ENV_EXTRA_DIRS: &str = "VAANI_MODEL_DIRS";

/// The application directory name under the per-user data directory.
pub const APP_DIR: &str = "VaaniDesk";
/// The model directory name under [`APP_DIR`], and also the directory name a
/// distribution ships beside the executable.
///
/// One constant for both because they are the same word on purpose: a portable
/// copy and an installed copy should not differ by a path component that nobody
/// documented.
pub const MODELS_DIR: &str = "models";

/// The file that marks a portable install and names its model root.
///
/// A user who unpacks the portable ZIP onto a USB stick and keeps the 887 MB of
/// weights on an external drive writes one line into `portable.txt` beside the
/// executable:
///
/// ```text
/// D:\VaaniDesk-models
/// ```
///
/// This is a real layout, not a hypothetical one: it is the only way a user can
/// get the weights off the stick without either re-downloading them or copying
/// 887 MB onto internal storage they do not have. It is last in the search order
/// for the same reason: it is a deliberate, manual, documented act, so it loses
/// to anything the installer did.
pub const PORTABLE_MARKER: &str = "portable.txt";

/// Which search root a model was found in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RootKind {
    /// A model directory that ships with the application, under `<app>/models/`.
    /// Highest precedence: see the module documentation.
    Bundled,
    /// The per-user model directory. Where a download goes.
    User,
    /// An explicitly configured directory. Lowest precedence.
    Configured,
}

impl RootKind {
    /// The stable wire form, e.g. `user`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bundled => "bundled",
            Self::User => "user",
            Self::Configured => "configured",
        }
    }
}

impl std::fmt::Display for RootKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Why a manifest file path was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathProblem {
    /// Empty, or only separators.
    Empty,
    /// Rooted (`/x`, `C:\x`), so it does not name something inside the model.
    Absolute,
    /// Contains a `..` component, so it can escape the model directory.
    ParentEscape,
    /// Contains a `\` or a drive specifier; manifests are portable and use `/`.
    NotPortable,
    /// Empty component, e.g. `a//b` or a trailing `/`.
    EmptyComponent,
    /// A `.` component, which is never what a human meant.
    CurrentDir,
}

impl PathProblem {
    /// A short explanation for a `ManifestError` message.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Empty => "is empty",
            Self::Absolute => "is an absolute path",
            Self::ParentEscape => "contains a '..' component",
            Self::NotPortable => "uses a backslash; manifest paths use '/'",
            Self::EmptyComponent => "has an empty path component",
            Self::CurrentDir => "contains a '.' component",
        }
    }
}

impl std::fmt::Display for PathProblem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Turn a manifest-relative path into a real one, rejecting anything that could
/// escape the model directory.
///
/// The check is on the **string**, before it becomes a [`PathBuf`], because
/// that is where the ambiguity is: `..` is a path component on every platform,
/// and a manifest is data that a human (or a compromised registry) supplies.
pub fn relative_path(p: &str) -> Result<PathBuf, PathProblem> {
    if p.trim().is_empty() {
        return Err(PathProblem::Empty);
    }
    if p.contains('\\') {
        return Err(PathProblem::NotPortable);
    }
    // A component check via `Path` catches `/x`, `C:/x` and `//server/share`.
    let as_path = Path::new(p);
    if as_path.is_absolute() || as_path.has_root() {
        return Err(PathProblem::Absolute);
    }
    // Checked on the string, before `Path` normalises it away: `Path::components`
    // collapses `a//b` to `a`/`b` and drops a trailing separator, so an empty
    // component written by a human would be silently accepted.
    if p.contains("//") || p.ends_with('/') {
        return Err(PathProblem::EmptyComponent);
    }
    let mut out = PathBuf::new();
    let mut components = 0usize;
    for c in Path::new(p).components() {
        match c {
            Component::Normal(part) => {
                let Some(part) = part.to_str() else {
                    return Err(PathProblem::NotPortable);
                };
                if part.is_empty() {
                    return Err(PathProblem::EmptyComponent);
                }
                out.push(part);
                components += 1;
            }
            Component::CurDir => return Err(PathProblem::CurrentDir),
            Component::ParentDir => return Err(PathProblem::ParentEscape),
            Component::RootDir | Component::Prefix(_) => return Err(PathProblem::Absolute),
        }
    }
    if components == 0 {
        return Err(PathProblem::EmptyComponent);
    }
    Ok(out)
}

/// The default per-user model directory: `%LOCALAPPDATA%\VaaniDesk\models`.
///
/// Falls back to `<temp>/VaaniDesk/models` only if `dirs` cannot find a data
/// directory at all, which on Windows means a broken or over-quota profile. A
/// wrong-but-existing path beats a panic at startup.
pub fn default_user_dir() -> PathBuf {
    if let Some(d) = std::env::var_os(ENV_USER_DIR) {
        return PathBuf::from(d);
    }
    dirs::data_local_dir()
        .map(|d| d.join(APP_DIR).join(MODELS_DIR))
        .unwrap_or_else(|| std::env::temp_dir().join(APP_DIR).join(MODELS_DIR))
}

/// Why a directory is in the bundled search order.
///
/// The reason travels with the path so that a diagnostic can say *why* a
/// directory was considered, which is the difference between "models not found"
/// and "models not found; you looked in `C:\Program Files\VaaniDesk\models` and
/// in `D:\models` because `portable.txt` says so".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[allow(missing_docs)] // each variant is its own documentation
pub enum BundledOrigin {
    /// `VAANI_ARTIFACTS_ROOT`. Wins over everything, exists or not.
    EnvArtifactsRoot,
    /// The parent of `VAANI_ARTEFACTS`, the variable `vaani-asr` reads.
    EnvArtefactsModelDir,
    /// `<exe dir>/models/`. The installed distribution.
    ExeRelative,
    /// `<exe dir>/../models/`. A portable ZIP whose executables sit in `bin/`.
    ExeParent,
    /// The path named in `<exe dir>/portable.txt`.
    PortableMarker,
    /// The developer checkout's `artifacts/`, or the compile-time repository.
    DeveloperTree,
}

impl BundledOrigin {
    /// Every origin, in search order.
    pub const ALL: [BundledOrigin; 6] = [
        Self::EnvArtifactsRoot,
        Self::EnvArtefactsModelDir,
        Self::ExeRelative,
        Self::ExeParent,
        Self::PortableMarker,
        Self::DeveloperTree,
    ];

    /// A short stable label for a log line or a doctor table.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EnvArtifactsRoot => "VAANI_ARTIFACTS_ROOT",
            Self::EnvArtefactsModelDir => "VAANI_ARTIFACTS",
            Self::ExeRelative => "<exe>/models",
            Self::ExeParent => "<exe>/../models",
            Self::PortableMarker => "portable.txt",
            Self::DeveloperTree => "developer tree",
        }
    }
}

impl std::fmt::Display for BundledOrigin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One directory in the bundled search order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundledCandidate {
    /// Why this directory is being considered.
    pub origin: BundledOrigin,
    /// The directory itself.
    pub dir: PathBuf,
    /// Whether it is a directory right now.
    ///
    /// A `false` here is not an error. It is the normal state of four of the six
    /// candidates on any given machine, and reporting it is what lets
    /// `vaani doctor` explain a miss instead of just declaring one.
    pub exists: bool,
}

/// The bundled search order, in full, whether or not the directories exist.
///
/// This is the single place the order is written down. It is a function rather
/// than a chain of `if`s so that:
///
/// * the order is *observable* — [`ModelRoots`] exposes it, so `vaani doctor`
///   and the settings panel can print the list rather than guess;
/// * the order is *testable* — a test can assert the whole sequence without
///   creating six directories on disk, which is the only way to test the
///   precedence between two candidates that cannot both exist.
///
/// Duplicates are removed, keeping the **first** occurrence, because a portable
/// install whose `portable.txt` names `<exe>/../models` should not list the same
/// directory twice in a diagnostic.
///
/// # Why the executable's directory is consulted and the CWD is not
///
/// A user who launches `vaani` from `C:\` with a double-click on the shortcut,
/// or from a file manager whose working directory is the Documents folder, must
/// still find its models. `std::env::current_dir` is a property of how the
/// process was started and carries no information about where the program lives;
/// `current_exe` does. Every exe-relative rule in this function uses the latter.
///
/// # The two rules that are not "look for a directory"
///
/// `VAANI_ARTIFACTS_ROOT` and `VAANI_ARTEFACTS` are **overrides, not
/// candidates**: they win even when the directory does not exist, and they stop
/// the search. A user who sets a variable to point at a model directory is
/// debugging a path problem, and answering them with a *different* directory
/// that happens to exist would make the bug undebuggable. They are still
/// reported in the candidate list — with `exists: false` — so a diagnostic can
/// show what was asked for.
#[must_use]
pub fn bundled_candidates() -> Vec<BundledCandidate> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf));
    bundled_candidates_from(exe_dir.as_deref(), &dev_tree())
}

/// The search order, with the executable directory and developer tree injected.
///
/// Split out from [`bundled_candidates`] so a test can drive it with a
/// synthetic `exe_dir` — `current_exe` in a test binary points at
/// `target/<profile>/deps/`, which is a real path that is not the path under
/// test, and asserting against it would test the build layout rather than the
/// rule.
pub fn bundled_candidates_from(exe_dir: Option<&Path>, dev_tree: &Path) -> Vec<BundledCandidate> {
    let mut out: Vec<BundledCandidate> = Vec::new();
    let mut push = |origin: BundledOrigin, dir: PathBuf| {
        // First occurrence wins, so precedence is decided by the order written
        // below and not by a later duplicate overwriting an earlier one.
        if out.iter().any(|c| c.dir == dir) {
            return;
        }
        let exists = dir.is_dir();
        out.push(BundledCandidate {
            origin,
            dir,
            exists,
        });
    };

    if let Some(d) = std::env::var_os(ENV_ARTEFACTS_ROOT) {
        push(BundledOrigin::EnvArtifactsRoot, PathBuf::from(d));
        // An override stops the search. See the doc comment.
        return out;
    }
    if let Some(d) = std::env::var_os(ENV_ARTEFACTS_MODEL_DIR) {
        // `vaani-asr` treats this as the model directory itself
        // (`sravaani.rs:733`), so the root is its parent.
        let p = PathBuf::from(d);
        let parent = p.parent().map_or_else(|| p.clone(), Path::to_path_buf);
        push(BundledOrigin::EnvArtefactsModelDir, parent);
        return out;
    }

    if let Some(exe) = exe_dir {
        // The installed distribution. This is the entry the whole zero-token
        // mandate rests on, so it is first among the filesystem candidates.
        push(BundledOrigin::ExeRelative, exe.join(MODELS_DIR));
        // The portable ZIP layout: `VaaniDesk/bin/vaani.exe` with `models/`
        // beside `bin/`, which is what a hand-unzipped archive looks like.
        if let Some(parent) = exe.parent() {
            push(BundledOrigin::ExeParent, parent.join(MODELS_DIR));
        }
        // A deliberate, manual redirect for a portable install. Read rather than
        // probed, because a one-line file the user wrote is a statement about
        // where their weights are and it deserves the same weight as an env var.
        if let Some(named) = read_portable_marker(&exe.join(PORTABLE_MARKER)) {
            push(BundledOrigin::PortableMarker, named);
        }
    }

    // `cargo run`/`cargo test` put binaries in `target/<profile>/deps`, so a
    // repository's `artifacts/` is three levels up. Same rule as
    // `RuntimeLocation::discover`, and the reason the dev tree is reachable from
    // a plain `cargo test` with no environment setup.
    if let Some(exe) = exe_dir {
        for up in [3usize, 4] {
            let mut base = exe.to_path_buf();
            for _ in 0..up {
                match base.parent() {
                    Some(p) => base = p.to_path_buf(),
                    None => break,
                }
            }
            let candidate = base.join("artifacts");
            if candidate.is_dir() {
                push(BundledOrigin::DeveloperTree, candidate);
                break;
            }
        }
    }
    push(BundledOrigin::DeveloperTree, dev_tree.join("artifacts"));

    out
}

/// The repository's `artifacts/` directory, from the compile-time manifest dir.
///
/// `CARGO_MANIFEST_DIR` is `<repo>/crates/vaani-models`. A relative
/// `artifacts` is the last resort, and it is a *worse* last resort than an
/// exe-relative path: a user who installed the app and deleted the models
/// deserves to be told about `%LOCALAPPDATA%\Program Files\...\models`, not about
/// a path relative to a working directory they never chose.
fn dev_tree() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map_or_else(|| PathBuf::from("."), |root| root.to_path_buf())
}

/// The first non-blank line of a `portable.txt`, as a path.
///
/// Returns `None` for a missing file, an unreadable one, an empty one, or one
/// whose first meaningful line is a comment. A malformed marker is *ignored*
/// rather than treated as an override: the whole point of this entry is that it
/// sits near the end of the order, so a typo in it degrades to "not found" and
/// the user gets the standard message naming every directory that was searched —
/// which includes the marker's own path, because it was pushed as a candidate.
fn read_portable_marker(path: &Path) -> Option<PathBuf> {
    let text = std::fs::read_to_string(path).ok()?;
    text.lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))
        .map(PathBuf::from)
}

/// The bundled root that ships with the application, or the closest thing to it.
///
/// The resolution rule, in one sentence: **an override wins outright; otherwise
/// the first candidate that exists; otherwise the most likely place for *this*
/// binary, so the error message names a path the user can actually go and look
/// at.**
///
/// That last clause is the one worth arguing for. When nothing is found, the
/// alternatives are to return a compile-time repository path (wrong: the user
/// has no repository, and the message would send them looking for a directory
/// that does not exist on their machine) or the exe-relative `models/`
/// (right: it is where an install *would* have put them, so "the installer did
/// not stage the models" is the conclusion they can act on).
///
/// # Environment
///
/// * `VAANI_ARTIFACTS_ROOT` — the bundled root, outright. Returned even if it
///   does not exist.
/// * `VAANI_ARTIFACTS` — the model *directory*, as `vaani-asr` spells it; its
///   parent is the root. Also returned even if it does not exist.
pub fn default_bundled_dir() -> PathBuf {
    let candidates = bundled_candidates();
    if let Some(c) = candidates
        .iter()
        .find(|c| matches!(c.origin, BundledOrigin::EnvArtifactsRoot))
    {
        return c.dir.clone();
    }
    if let Some(c) = candidates
        .iter()
        .find(|c| matches!(c.origin, BundledOrigin::EnvArtefactsModelDir))
    {
        return c.dir.clone();
    }
    if let Some(c) = candidates.iter().find(|c| c.exists) {
        return c.dir.clone();
    }
    // Nothing exists. Prefer the install-relative path the installer would have
    // used over the developer's `artifacts/`.
    if let Some(c) = candidates
        .iter()
        .find(|c| matches!(c.origin, BundledOrigin::ExeRelative))
    {
        return c.dir.clone();
    }
    candidates
        .iter()
        .find(|c| matches!(c.origin, BundledOrigin::DeveloperTree))
        .map_or_else(|| PathBuf::from(MODELS_DIR), |c| c.dir.clone())
}

/// Extra roots from `VAANI_MODEL_DIRS`.
fn env_extra_dirs() -> Vec<PathBuf> {
    std::env::var_os(ENV_EXTRA_DIRS)
        .map(|v| {
            std::env::split_paths(&v)
                .filter(|p| !p.as_os_str().is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// The ordered set of directories discovery searches.
///
/// Search order is **bundled, user, configured**. The reasoning is in the module
/// documentation; the short version is that the distribution ships the weights,
/// so a shipped copy that the build verified outranks a copy in app data that
/// nothing has checked is the version this build expects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRoots {
    user: PathBuf,
    bundled: PathBuf,
    configured: Vec<PathBuf>,
    /// The full bundled candidate list, kept so a diagnostic can show the order
    /// and say which entries were missing rather than only naming the one that
    /// was chosen. Populated by [`ModelRoots::discover`]; empty for
    /// [`ModelRoots::at`], which is a test constructor with nothing to report.
    bundled_candidates: Vec<BundledCandidate>,
}

impl ModelRoots {
    /// The roots this machine would use, honouring every environment override.
    pub fn discover() -> ModelRoots {
        let candidates = bundled_candidates();
        ModelRoots {
            user: default_user_dir(),
            bundled: default_bundled_dir(),
            configured: env_extra_dirs(),
            bundled_candidates: candidates,
        }
    }

    /// Roots with nothing overridden: user data dir, the bundled root, and no
    /// extras.
    ///
    /// Used by tests, which must not be at the mercy of whatever the developer's
    /// shell happens to export.
    #[must_use]
    pub fn at(user: impl Into<PathBuf>, bundled: impl Into<PathBuf>) -> ModelRoots {
        ModelRoots {
            user: user.into(),
            bundled: bundled.into(),
            configured: Vec::new(),
            bundled_candidates: Vec::new(),
        }
    }

    /// Add a configured root. Later additions sort after earlier ones.
    #[must_use]
    pub fn with_configured(mut self, dir: impl Into<PathBuf>) -> ModelRoots {
        self.configured.push(dir.into());
        self
    }

    /// The per-user download directory.
    pub fn user(&self) -> &Path {
        &self.user
    }

    /// The bundled model directory — the one the distribution ships into.
    pub fn bundled(&self) -> &Path {
        &self.bundled
    }

    /// The full bundled search order this machine resolved, including the
    /// candidates that were considered and rejected.
    ///
    /// Empty when the roots came from [`ModelRoots::at`], which has no search to
    /// report.
    #[must_use]
    pub fn bundled_candidates(&self) -> &[BundledCandidate] {
        &self.bundled_candidates
    }

    /// Explicitly configured directories, in search order.
    pub fn configured(&self) -> &[PathBuf] {
        &self.configured
    }

    /// Every root, in search order: bundled, user, then configured.
    ///
    /// Bundled first is the change from A7's original user-first order; see the
    /// module documentation for why, and for why [`ModelRoots::install_target`]
    /// is unaffected.
    pub fn all(&self) -> Vec<(RootKind, &Path)> {
        let mut v: Vec<(RootKind, &Path)> = vec![
            (RootKind::Bundled, self.bundled.as_path()),
            (RootKind::User, self.user.as_path()),
        ];
        v.extend(
            self.configured
                .iter()
                .map(|p| (RootKind::Configured, p.as_path())),
        );
        v
    }

    /// Where a model with this on-disk directory name should be searched for,
    /// in precedence order. Roots that do not exist are still returned;
    /// discovery does the existence check and reports a missing root once.
    pub fn candidates(&self, dir_name: &str) -> Vec<(RootKind, PathBuf)> {
        self.all()
            .into_iter()
            .map(|(k, p)| (k, p.join(dir_name)))
            .collect()
    }

    /// The root a *new* install should go into: the user directory, always.
    ///
    /// Unchanged by the search-order reversal, and deliberately so. Downloads
    /// never land in the bundled tree: that tree belongs to the installer and is
    /// under `Program Files` in a machine-wide install, which a standard user
    /// cannot write to. Writing there would also make `git status` in a checkout
    /// dirty in a way nobody could explain.
    pub fn install_target(&self) -> &Path {
        &self.user
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_paths_are_accepted_in_the_manifest_form() {
        assert_eq!(
            relative_path("encoder.fp16.onnx").unwrap(),
            PathBuf::from("encoder.fp16.onnx")
        );
        assert_eq!(
            relative_path("frontend/mel_window.f32").unwrap(),
            PathBuf::from("frontend").join("mel_window.f32")
        );
    }

    #[test]
    fn relative_paths_that_could_escape_the_model_directory_are_refused() {
        // Every one of these is a way to write outside the model directory, and
        // the manifest is data, so the check is on the string before it becomes
        // a path.
        for (input, why) in [
            ("", PathProblem::Empty),
            ("   ", PathProblem::Empty),
            ("/", PathProblem::Absolute),
            ("C:/models/x.onnx", PathProblem::Absolute),
            ("//server/share/x.onnx", PathProblem::Absolute),
            ("../x.onnx", PathProblem::ParentEscape),
            ("a/../../x.onnx", PathProblem::ParentEscape),
            ("..", PathProblem::ParentEscape),
            ("a\\b.onnx", PathProblem::NotPortable),
            ("a//b.onnx", PathProblem::EmptyComponent),
            ("a/", PathProblem::EmptyComponent),
            ("./a.onnx", PathProblem::CurrentDir),
            (".", PathProblem::CurrentDir),
        ] {
            assert_eq!(
                relative_path(input),
                Err(why),
                "expected {input:?} to be refused as {why:?}"
            );
        }
    }

    #[test]
    fn every_path_problem_has_a_reason() {
        for p in [
            PathProblem::Empty,
            PathProblem::Absolute,
            PathProblem::ParentEscape,
            PathProblem::NotPortable,
            PathProblem::EmptyComponent,
            PathProblem::CurrentDir,
        ] {
            assert!(!p.as_str().is_empty());
        }
    }

    #[test]
    fn search_order_puts_the_bundled_directory_first() {
        // The whole zero-token mandate rests on this line. A shipped, verified
        // copy outranks a copy in app data that nothing has checked the version
        // of.
        let r = ModelRoots::at("u", "b").with_configured("c");
        let kinds: Vec<RootKind> = r.all().into_iter().map(|(k, _)| k).collect();
        assert_eq!(
            kinds,
            vec![RootKind::Bundled, RootKind::User, RootKind::Configured]
        );
        assert_eq!(r.candidates("sravaani")[0].0, RootKind::Bundled);
    }

    #[test]
    fn the_install_target_is_still_the_user_directory() {
        // Reversing the *search* order must not reverse where downloads land. A
        // bundled tree is under Program Files on a machine-wide install, which a
        // standard user cannot write to.
        let r = ModelRoots::at("u", "b");
        assert_eq!(r.install_target(), Path::new("u"));
    }

    #[test]
    fn the_default_user_directory_is_under_the_per_user_data_directory() {
        // `dirs::data_local_dir()` is `%LOCALAPPDATA%` on Windows. Asserted by
        // suffix rather than absolutely, so the test is meaningful off Windows
        // too, where it is `~/.local/share`.
        let d = std::env::var_os(ENV_USER_DIR).map_or_else(default_user_dir, PathBuf::from);
        assert!(d.ends_with("VaaniDesk/models") || d.ends_with("VaaniDesk\\models"));
    }

    // -----------------------------------------------------------------------
    // The bundled search order
    //
    // These use `bundled_candidates_from` with a synthetic `exe_dir`, because
    // `current_exe` in a test binary points at `target/<profile>/deps/` and
    // asserting against that would test the build layout, not the rule.
    // -----------------------------------------------------------------------

    /// A scratch directory that exists, for an "exe lives here" fixture.
    fn scratch(tag: &str) -> PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let p = std::env::temp_dir().join(format!(
            "vaani-a12-order-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// The order of origins, which is the part that must not drift.
    fn origins(exe: Option<&Path>, dev: &Path) -> Vec<BundledOrigin> {
        bundled_candidates_from(exe, dev)
            .into_iter()
            .map(|c| c.origin)
            .collect()
    }

    /// The directories in the order the pure function produced them, for the
    /// precedence assertions. `bundled_candidates_from` reads the process
    /// environment, so a developer with `VAANI_ARTIFACTS_ROOT` exported would
    /// otherwise change what these tests mean.
    #[test]
    fn the_bundled_search_order_is_fixed_and_documented() {
        if std::env::var_os(ENV_ARTEFACTS_ROOT).is_some()
            || std::env::var_os(ENV_ARTEFACTS_MODEL_DIR).is_some()
        {
            // The override arms are tested separately; here they would mask the
            // very ordering being asserted.
            return;
        }
        let base = scratch("order");
        let exe = base.join("VaaniDesk");
        std::fs::create_dir_all(&exe).unwrap();
        let dev = base.join("repo");

        assert_eq!(
            origins(Some(&exe), &dev),
            vec![
                BundledOrigin::ExeRelative,
                BundledOrigin::ExeParent,
                BundledOrigin::DeveloperTree,
            ],
            "an installed layout must try <exe>/models, then <exe>/../models, \
             then the developer tree"
        );
    }

    #[test]
    fn exe_relative_models_is_the_first_candidate_under_an_installed_exe() {
        if std::env::var_os(ENV_ARTEFACTS_ROOT).is_some()
            || std::env::var_os(ENV_ARTEFACTS_MODEL_DIR).is_some()
        {
            return;
        }
        let base = scratch("exerel");
        let exe = base.join("Program Files").join("VaaniDesk");
        std::fs::create_dir_all(&exe).unwrap();
        let c = bundled_candidates_from(Some(&exe), &base.join("repo"));
        assert_eq!(c[0].origin, BundledOrigin::ExeRelative);
        assert_eq!(c[0].dir, exe.join("models"));
    }

    #[test]
    fn a_portable_zip_whose_exe_is_in_bin_finds_models_one_level_up() {
        if std::env::var_os(ENV_ARTEFACTS_ROOT).is_some()
            || std::env::var_os(ENV_ARTEFACTS_MODEL_DIR).is_some()
        {
            return;
        }
        let base = scratch("portable");
        let exe = base.join("VaaniDesk").join("bin");
        std::fs::create_dir_all(&exe).unwrap();
        std::fs::create_dir_all(base.join("VaaniDesk").join("models")).unwrap();
        let c = bundled_candidates_from(Some(&exe), &base.join("repo"));

        let parent = c
            .iter()
            .find(|c| c.origin == BundledOrigin::ExeParent)
            .expect("the <exe>/../models candidate must be offered");
        assert_eq!(parent.dir, base.join("VaaniDesk").join("models"));
        assert!(parent.exists, "and it exists in this fixture");
    }

    #[test]
    fn a_portable_marker_names_the_model_root() {
        if std::env::var_os(ENV_ARTEFACTS_ROOT).is_some()
            || std::env::var_os(ENV_ARTEFACTS_MODEL_DIR).is_some()
        {
            return;
        }
        let base = scratch("marker");
        let exe = base.join("stick").join("VaaniDesk");
        std::fs::create_dir_all(&exe).unwrap();
        // Comments, a blank line, and padding around the path: a user writing
        // this in Notepad must not get a path with a trailing space on it.
        let named = base.join("elsewhere");
        std::fs::create_dir_all(&named).unwrap();
        let body = format!("# VaaniDesk portable models\n\n  {}  \n", named.display());
        std::fs::write(exe.join(PORTABLE_MARKER), body).unwrap();

        let c = bundled_candidates_from(Some(&exe), &base.join("repo"));
        let m = c
            .iter()
            .find(|c| c.origin == BundledOrigin::PortableMarker)
            .expect("the marker must produce a candidate");
        assert_eq!(m.dir, named);
        assert!(m.exists);
    }

    #[test]
    fn a_malformed_portable_marker_is_ignored_rather_than_fatal() {
        if std::env::var_os(ENV_ARTEFACTS_ROOT).is_some()
            || std::env::var_os(ENV_ARTEFACTS_MODEL_DIR).is_some()
        {
            return;
        }
        // A marker of only comments and blank lines names nothing. The search
        // must continue, because this entry sits near the end precisely so a
        // typo in it degrades to "not found" with a full explanation rather than
        // to a wrong answer.
        let base = scratch("badmarker");
        let exe = base.join("app");
        std::fs::create_dir_all(&exe).unwrap();
        std::fs::write(exe.join(PORTABLE_MARKER), "# only a comment\n\n").unwrap();
        let c = bundled_candidates_from(Some(&exe), &base.join("repo"));
        assert!(
            !c.iter().any(|c| c.origin == BundledOrigin::PortableMarker),
            "an empty marker must not produce a candidate: {:?}",
            c.iter().map(|c| (&c.origin, &c.dir)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_repeated_directory_appears_once_at_its_highest_priority() {
        if std::env::var_os(ENV_ARTEFACTS_ROOT).is_some()
            || std::env::var_os(ENV_ARTEFACTS_MODEL_DIR).is_some()
        {
            return;
        }
        // A marker naming `<exe>/../models` collides with the `ExeParent`
        // candidate. A diagnostic that shows the same path twice under two
        // reasons reads as a bug in the diagnostic, and `ExeParent` is the
        // higher-priority reason, so that is the one that must survive.
        let base = scratch("dedup");
        let exe = base.join("root").join("bin");
        std::fs::create_dir_all(&exe).unwrap();
        let dup = base.join("root").join(MODELS_DIR);
        std::fs::create_dir_all(&dup).unwrap();
        std::fs::write(exe.join(PORTABLE_MARKER), dup.display().to_string()).unwrap();

        let c = bundled_candidates_from(Some(&exe), &base.join("repo"));
        let hits: Vec<&BundledOrigin> = c
            .iter()
            .filter(|c| c.dir == dup)
            .map(|c| &c.origin)
            .collect();
        assert_eq!(
            hits,
            vec![&BundledOrigin::ExeParent],
            "listed once, under the reason that outranks the marker"
        );
    }

    #[test]
    fn every_origin_has_a_label_for_a_diagnostic() {
        for o in BundledOrigin::ALL {
            assert!(!o.as_str().is_empty(), "{o:?} has no label");
        }
    }

    #[test]
    fn the_candidate_list_reports_which_directories_are_missing() {
        // The property that makes a diagnostic possible: a root that does not
        // exist is *listed* as not existing rather than dropped.
        let base = scratch("missing");
        let exe = base.join("nowhere");
        std::fs::create_dir_all(&exe).unwrap();
        let c = bundled_candidates_from(Some(&exe), &base.join("repo"));
        assert!(!c.is_empty());
        for cand in &c {
            assert_eq!(cand.exists, cand.dir.is_dir(), "{cand:?}");
        }
        assert!(
            c.iter().all(|c| !c.exists),
            "nothing in this fixture exists, so every entry must say so: {c:?}"
        );
    }
}

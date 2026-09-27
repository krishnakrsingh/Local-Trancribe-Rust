//! Where models live, and how a manifest file path becomes a real one.
//!
//! # Three roots, searched in a fixed order
//!
//! | root | what it is | why it exists |
//! |---|---|---|
//! | [`RootKind::User`] | `%LOCALAPPDATA%\VaaniDesk\models` on Windows | where a download goes. Under app data, never the repo, never next to the executable: an installed app's directory is not writable by a standard user. |
//! | [`RootKind::Bundled`] | the dev/export `artifacts/` tree | where `scripts/export_sravaani_onnx.py` puts the artefacts, and where the frozen ASR tests read them from. |
//! | [`RootKind::Configured`] | whatever the user or `VAANI_MODEL_DIRS` names | a portable install on a slow drive, a network share, a second library. |
//!
//! Order matters and is **user first**: a user who has deliberately installed a
//! model wants *that* one, not whatever a developer checkout happens to contain
//! beside the binary.
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
/// The model directory name under [`APP_DIR`].
pub const MODELS_DIR: &str = "models";

/// Which search root a model was found in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RootKind {
    /// The per-user model directory. Highest precedence.
    User,
    /// The bundled/exported `artifacts/` tree.
    Bundled,
    /// An explicitly configured directory. Lowest precedence.
    Configured,
}

impl RootKind {
    /// The stable wire form, e.g. `user`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Bundled => "bundled",
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

/// The bundled artefact root that ships with the checkout.
///
/// Search order, mirroring `vaani_inference::runtime::RuntimeLocation::discover`
/// in spirit — an explicit override, then a location relative to the running
/// binary, then a location relative to this crate's source tree so that
/// `cargo test` from the repository root works with no environment setup. The
/// current working directory is deliberately **not** consulted: a user
/// launching `vaani` from `C:\` must still find its models.
pub fn default_bundled_dir() -> PathBuf {
    if let Some(d) = std::env::var_os(ENV_ARTEFACTS_ROOT) {
        return PathBuf::from(d);
    }
    if let Some(d) = std::env::var_os(ENV_ARTEFACTS_MODEL_DIR) {
        // `vaani-asr` treats this as the model directory itself
        // (`sravaani.rs:733`), so the root is its parent.
        let p = PathBuf::from(d);
        return match p.parent() {
            Some(parent) => parent.to_path_buf(),
            None => p,
        };
    }

    // Beside the executable: `models/` and `artifacts/` in a portable ZIP.
    if let Some(exe) = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
    {
        let candidate = exe.join("artifacts");
        if candidate.is_dir() {
            return candidate;
        }
        // `cargo run`/`cargo test` place binaries in `target/<profile>/deps`, so
        // the repository's `artifacts/` is three levels up. Same rule as
        // `RuntimeLocation::discover`.
        for up in [3usize, 4] {
            let mut base = exe.clone();
            for _ in 0..up {
                match base.parent() {
                    Some(p) => base = p.to_path_buf(),
                    None => break,
                }
            }
            let candidate = base.join("artifacts");
            if candidate.is_dir() {
                return candidate;
            }
        }
    }

    // `CARGO_MANIFEST_DIR` is `<repo>/crates/vaani-models`.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map_or_else(|| PathBuf::from("artifacts"), |root| root.join("artifacts"))
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRoots {
    user: PathBuf,
    bundled: PathBuf,
    configured: Vec<PathBuf>,
}

impl ModelRoots {
    /// The roots this machine would use, honouring every environment override.
    pub fn discover() -> ModelRoots {
        ModelRoots {
            user: default_user_dir(),
            bundled: default_bundled_dir(),
            configured: env_extra_dirs(),
        }
    }

    /// Roots with nothing overridden: user data dir, the repository's
    /// `artifacts/`, and no extras.
    ///
    /// Used by tests, which must not be at the mercy of whatever the developer's
    /// shell happens to export.
    #[must_use]
    pub fn at(user: impl Into<PathBuf>, bundled: impl Into<PathBuf>) -> ModelRoots {
        ModelRoots {
            user: user.into(),
            bundled: bundled.into(),
            configured: Vec::new(),
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

    /// The bundled/exported artefact directory.
    pub fn bundled(&self) -> &Path {
        &self.bundled
    }

    /// Explicitly configured directories, in search order.
    pub fn configured(&self) -> &[PathBuf] {
        &self.configured
    }

    /// Every root, in search order: user, bundled, then configured.
    pub fn all(&self) -> Vec<(RootKind, &Path)> {
        let mut v: Vec<(RootKind, &Path)> = vec![
            (RootKind::User, self.user.as_path()),
            (RootKind::Bundled, self.bundled.as_path()),
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
    /// Downloads never land in the bundled tree. That tree belongs to the
    /// exporter and is read-only in an installed app, and writing to it would
    /// make `git status` in a checkout dirty in a way nobody could explain.
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
    fn search_order_puts_the_user_directory_first() {
        let r = ModelRoots::at("u", "b").with_configured("c");
        let kinds: Vec<RootKind> = r.all().into_iter().map(|(k, _)| k).collect();
        assert_eq!(
            kinds,
            vec![RootKind::User, RootKind::Bundled, RootKind::Configured]
        );
        assert_eq!(r.candidates("sravaani")[0].0, RootKind::User);
    }

    #[test]
    fn new_installs_always_go_to_the_user_directory() {
        // Never into `artifacts/`: that tree belongs to the exporter and is
        // read-only in an installed app.
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
}

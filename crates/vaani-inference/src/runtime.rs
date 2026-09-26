//! Locating and initialising the ONNX Runtime native library.
//!
//! # Why `load-dynamic` and not a linked import library
//!
//! The `ort` crate's default build downloads a *statically linked* ONNX
//! Runtime at compile time and links it into the binary. We do not want that:
//!
//! * It makes a **build-time network dependency** for anyone who clones the
//!   repository, and pins them to whatever prebuilt binary pyke's CDN serves
//!   that day.
//! * It puts an `x86-64-v3` baseline (Haswell / Zen+) into our executable,
//!   which would refuse to run on older end-user laptops.
//! * It gives us no control over which CUDA build we ship. Microsoft's own
//!   1.30.0 CUDA-13 release compiles for `86-real`, i.e. native SASS for the
//!   compute capability 8.6 in the RTX 3050 Ti family, so no PTX JIT and no
//!   "no kernel image is available" (CUDA 209) on first run. The CDN build
//!   ships `75;80;90;120` and would make us depend on JIT.
//!
//! So we ship the DLLs ourselves, next to the executable, and load them at
//! runtime. Windows resolves `onnxruntime.dll` from the executable's own
//! directory first, and we additionally call `ort::init_from` with an explicit
//! path so an incomplete install fails loudly at startup instead of silently
//! falling back to something else.
//!
//! See `docs/decisions/ADR-003-onnx-runtime.md`.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use vaani_core::{Result, err};

/// The file name of the core runtime library.
pub const ORT_DLL: &str = "onnxruntime.dll";

/// The CUDA execution provider, loaded as a separate DLL by ORT 1.19+.
pub const ORT_CUDA_DLL: &str = "onnxruntime_providers_cuda.dll";

/// Which execution provider a session should use.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
#[allow(missing_docs)] // each variant maps to a settings option
pub enum Provider {
    /// Use the GPU if a working CUDA runtime is present; otherwise CPU.
    ///
    /// This is the default because it is what a user wants and what the
    /// hardware almost always supports. It never fails: if CUDA registration
    /// is refused, ORT runs on the CPU EP and the session still works.
    #[default]
    Auto,
    /// Force the CPU. Exists so a user can reproduce a slow baseline, and so
    /// the CPU path gets exercised by the test suite on GPU machines.
    Cpu,
    /// Prefer the GPU and *record* that we had to fall back.
    ///
    /// Note this is **not** a hard failure, despite the name: if CUDA is
    /// unavailable the session runs on the CPU and `ProviderStatus::active`
    /// says so. That is deliberate -- a user who asks for a transcript should
    /// get one, slowly. What `RequireCuda` buys is that the fallback is
    /// *visible* rather than silent, so the benchmark harness can refuse to
    /// publish CPU timings in a GPU column.
    RequireCuda,
}

impl Provider {
    /// The stable wire form.
    /// The stable wire form.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::RequireCuda => "require_cuda",
        }
    }

    /// Parse a wire form, case-insensitively.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "cpu" => Some(Self::Cpu),
            "cuda" | "gpu" | "require_cuda" => Some(Self::RequireCuda),
            _ => None,
        }
    }
}

/// Where the native runtime lives.
///
/// Resolved once at startup. Exposed so the settings panel and `vaani doctor`
/// can show the user exactly which DLLs were loaded, which is the first thing
/// anyone asks when acceleration misbehaves.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct RuntimeLocation {
    pub ort_dll: PathBuf,
    /// Directories searched for the CUDA/cuDNN redistributable DLLs.
    pub provider_dirs: Vec<PathBuf>,
    /// True when the DLLs came from a `runtime/` subdirectory next to the
    /// executable rather than sitting beside it. Both layouts are supported so
    /// a portable ZIP and an installed copy can both work.
    pub portable_layout: bool,
}

impl RuntimeLocation {
    /// Search order, most specific first:
    ///
    /// 1. `VAANI_RUNTIME_DIR` -- an explicit override, used by the test suite
    ///    and by anyone running from a build directory.
    /// 2. `<exe dir>/runtime`
    /// 3. `<exe dir>`
    /// 4. walking 3 and then 4 directories up from the executable and looking
    ///    for `runtime/`. Cargo places test and example binaries in
    ///    `target/<profile>/{deps,examples}`, which is three levels below the
    ///    repository root, so this is what makes `cargo run` and `cargo test`
    ///    work from the root with no environment setup. The current working
    ///    directory is deliberately not consulted: a user launching
    ///    `vaani` from an arbitrary directory must still find the runtime.
    pub fn discover() -> RuntimeLocation {
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(Path::to_path_buf))
            .unwrap_or_else(|| PathBuf::from("."));

        if let Some(dir) = std::env::var_os("VAANI_RUNTIME_DIR") {
            let d = PathBuf::from(dir);
            return Self::at(d.clone(), vec![d], false);
        }

        let portable = exe_dir.join("runtime");
        if portable.join(ORT_DLL).is_file() {
            return Self::at(portable.join(ORT_DLL), vec![portable], true);
        }
        let flat = exe_dir.join(ORT_DLL);
        if flat.is_file() {
            return Self::at(flat, vec![exe_dir.clone()], false);
        }
        // `cargo run`/`cargo test` put binaries in target/<profile>/deps, so the
        // repository's runtime/ is three levels up.
        for up in [3usize, 4] {
            let mut base = exe_dir.clone();
            for _ in 0..up {
                match base.parent() {
                    Some(p) => base = p.to_path_buf(),
                    None => break,
                }
            }
            let candidate = base.join("runtime");
            if candidate.join(ORT_DLL).is_file() {
                return Self::at(candidate.join(ORT_DLL), vec![candidate], false);
            }
        }
        // Nothing found. Return the most likely location so the error message
        // names a concrete path the user can look at.
        Self::at(flat, vec![exe_dir], false)
    }

    fn at(ort_dll: PathBuf, provider_dirs: Vec<PathBuf>, portable_layout: bool) -> Self {
        Self {
            ort_dll,
            provider_dirs,
            portable_layout,
        }
    }

    /// Whether the core runtime DLL was found.
    pub fn is_present(&self) -> bool {
        self.ort_dll.is_file()
    }

    /// Missing DLLs, in the order the user must fix them.
    /// Which expected DLLs are absent, for an actionable error message.
    pub fn missing(&self) -> Vec<String> {
        let mut missing = Vec::new();
        if !self.ort_dll.is_file() {
            missing.push(ORT_DLL.to_string());
        }
        if !self
            .provider_dirs
            .iter()
            .any(|d| d.join(ORT_CUDA_DLL).is_file())
        {
            missing.push(ORT_CUDA_DLL.to_string());
        }
        missing
    }
}

static RUNTIME: OnceLock<RuntimeLocation> = OnceLock::new();
static INIT_RESULT: OnceLock<std::result::Result<(), String>> = OnceLock::new();

/// The resolved runtime location, initialising on first use.
pub fn location() -> &'static RuntimeLocation {
    RUNTIME.get_or_init(RuntimeLocation::discover)
}

/// Initialise ONNX Runtime exactly once per process.
///
/// `ort` requires this before any other call, and the environment it builds is
/// process-global, so a second call is an error. The `OnceLock` makes the
/// second call a no-op returning the first result, which is what every caller
/// actually wants.
///
/// **Nothing else in the process may touch `ort` before this runs.** In
/// particular the CUDA runtime is not created here: that happens lazily on the
/// first session, because CUDA context creation costs several hundred
/// megabytes of VRAM and a few hundred milliseconds, and a user who opens the
/// app to browse history should not pay for it.
pub fn ensure_initialised() -> Result<()> {
    let loc = location();
    let res = INIT_RESULT.get_or_init(|| {
        if !loc.is_present() {
            return Err(format!(
                "{} not found at {}. Run scripts\\fetch_runtime.ps1, or reinstall VaaniDesk.",
                ORT_DLL,
                loc.ort_dll.display()
            ));
        }

        // cuDNN 9 is a *split* library: `cudnn64_9.dll` is a dispatcher that
        // `LoadLibrary`s its ten siblings by bare name at first use. The default
        // Windows search order only looks next to the executable, and in
        // development the executable lives in `target/<profile>/`, several
        // directories away from `runtime/`. Without this the failure surfaces as
        //
        //     Could not locate cudnn_engines_runtime_compiled64_9.dll
        //
        // from deep inside ONNX Runtime's provider, long after the useful
        // context has scrolled away.
        //
        // `SetDllDirectoryW` is the documented mechanism: it appends a directory
        // to the search order for every `LoadLibrary` made *after* the call, and
        // unlike the preloads below it also covers libraries loaded lazily
        // during the first inference run.
        for dir in &loc.provider_dirs {
            if dir.is_dir() {
                set_dll_directory(dir);
            }
        }

        // Belt and braces: resolve each DLL now, so an incomplete install fails
        // with *our* message naming the missing file rather than ORT's.
        for dir in &loc.provider_dirs {
            preload_dir(dir);
        }

        ort::init_from(&loc.ort_dll)
            .map_err(|e| format!("failed to load {}: {e}", loc.ort_dll.display()))?
            .commit();

        tracing::info!(
            ort = %loc.ort_dll.display(),
            dirs = ?loc.provider_dirs,
            portable = loc.portable_layout,
            "ONNX Runtime initialised"
        );
        Ok(())
    });

    match res {
        Ok(()) => Ok(()),
        Err(e) => Err(err::internal().with_cause(e).with_recovery(
            "Reinstall VaaniDesk, or run scripts\\fetch_runtime.ps1 from a source checkout.",
        )),
    }
}

/// Append `dir` to the process DLL search order.
///
/// A no-op on failure: a locked-down environment can refuse it, and in that case
/// the preloads below plus the exe-adjacent layout are still worth trying. This
/// is an optimisation of the *error path*, not the success path.
fn set_dll_directory(dir: &Path) {
    let wide = dir
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    // SAFETY: `wide` is a NUL-terminated UTF-16 buffer that outlives the call,
    // and `SetDllDirectoryW` only reads it. The returned string is allocated by
    // the OS and intentionally leaked, per the documented contract.
    unsafe {
        let _ = windows::Win32::System::LibraryLoader::SetDllDirectoryW(windows::core::PCWSTR(
            wide.as_ptr(),
        ));
    }
}

/// Resolve every DLL in `dir` now, so an incomplete install fails with our
/// message naming the missing file rather than ORT's.
///
/// The handles are intentionally leaked: that is how a process keeps a runtime
/// DLL resident for its whole lifetime.
fn preload_dir(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let path = e.path();
        if path
            .extension()
            .is_some_and(|x| x.eq_ignore_ascii_case("dll"))
        {
            let wide = path
                .as_os_str()
                .encode_wide()
                .chain(std::iter::once(0))
                .collect::<Vec<_>>();
            // SAFETY: `wide` is NUL-terminated and outlives the call;
            // `LoadLibraryW` only reads it.
            unsafe {
                if windows::Win32::System::LibraryLoader::LoadLibraryW(windows::core::PCWSTR(
                    wide.as_ptr(),
                ))
                .is_err()
                {
                    tracing::debug!(path = %path.display(), "could not preload runtime DLL");
                }
            }
        }
    }
}

use std::os::windows::ffi::OsStrExt;

/// The result of a provider decision, for display in the UI and the benchmark
/// report.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct ProviderStatus {
    pub requested: Provider,
    /// The provider that is actually in use. `Cpu` when CUDA was requested but
    /// is unavailable, which is a normal outcome and never a crash.
    pub active: ActiveProvider,
    /// Why, in a form safe to show a user.
    pub note: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
#[allow(missing_docs)] // the two providers that exist
pub enum ActiveProvider {
    Cuda,
    Cpu,
}

impl ActiveProvider {
    /// The stable wire form, e.g. cuda or cpu.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
        }
    }
}

/// Decide the execution provider for a session.
///
/// Pure decision logic, separated from session construction so it is unit
/// testable without ONNX Runtime present. `cuda_loadable` is the answer to
/// "is the CUDA provider DLL actually here", which is the question that decides
/// this in practice: on Windows the provider is a separate DLL, and a machine
/// can have a perfect NVIDIA driver and still lack the provider.
pub fn choose_provider(requested: Provider, cuda_loadable: bool) -> ProviderStatus {
    // Order matters. The explicit `Cpu` request has to be matched *before* the
    // "CUDA is available" arm, or a user who asked for the CPU silently gets
    // the GPU -- which is both surprising and makes the CPU path untestable on
    // a machine that has a GPU.
    match requested {
        Provider::Cpu => ProviderStatus {
            requested,
            active: ActiveProvider::Cpu,
            note: Some("Using the CPU, as requested.".into()),
        },
        Provider::Auto | Provider::RequireCuda if cuda_loadable => ProviderStatus {
            requested,
            active: ActiveProvider::Cuda,
            note: Some("Using the NVIDIA GPU.".into()),
        },
        Provider::RequireCuda => ProviderStatus {
            requested,
            active: ActiveProvider::Cpu,
            note: Some(
                "Graphics acceleration was requested but is unavailable, so the CPU was used instead."
                    .into(),
            ),
        },
        Provider::Auto => ProviderStatus {
            requested,
            active: ActiveProvider::Cpu,
            note: Some(
                "Graphics acceleration is unavailable, so processing runs on the CPU. \
                 Transcription will be slower but the result is identical."
                    .into(),
            ),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_cuda_request_falls_back_without_failing() {
        // `RequireCuda` still runs on the CPU rather than erroring: a
        // transcript is worth more than a benchmark purity guarantee, and the
        // status is recorded so the benchmark can flag the mismatch.
        let s = choose_provider(Provider::RequireCuda, false);
        assert_eq!(s.active, ActiveProvider::Cpu);
        assert!(s.note.unwrap().contains("unavailable"));
    }

    #[test]
    fn auto_prefers_cuda_when_available() {
        assert_eq!(
            choose_provider(Provider::Auto, true).active,
            ActiveProvider::Cuda
        );
    }

    #[test]
    fn cpu_request_is_honoured_even_when_cuda_exists() {
        // This is the path that lets the suite exercise the CPU backend on a
        // GPU machine, so it must not be optimised away.
        let s = choose_provider(Provider::Cpu, true);
        assert_eq!(s.active, ActiveProvider::Cpu);
        assert_eq!(s.requested, Provider::Cpu);
    }

    #[test]
    fn provider_strings_round_trip() {
        for p in [Provider::Auto, Provider::Cpu, Provider::RequireCuda] {
            assert_eq!(Provider::parse(p.as_str()), Some(p));
        }
        assert_eq!(Provider::parse("GPU"), Some(Provider::RequireCuda));
        assert_eq!(Provider::parse("nonsense"), None);
    }

    #[test]
    fn discover_finds_a_runtime_or_names_a_concrete_missing_path() {
        // Never returns a bare filename: the error has to tell the user where
        // it looked.
        let loc = RuntimeLocation::discover();
        assert!(loc.ort_dll.is_absolute() || loc.ort_dll.starts_with("runtime"));
        assert!(!loc.provider_dirs.is_empty());
        if !loc.is_present() {
            assert!(!loc.missing().is_empty());
        }
    }

    #[test]
    fn missing_lists_the_provider_when_only_the_core_is_absent() {
        let loc = RuntimeLocation::at(
            PathBuf::from("definitely/not/here/onnxruntime.dll"),
            vec![PathBuf::from("definitely/not/here")],
            false,
        );
        assert!(!loc.is_present());
        let m = loc.missing();
        assert!(m.contains(&ORT_DLL.to_string()));
        assert!(m.contains(&ORT_CUDA_DLL.to_string()));
    }
}

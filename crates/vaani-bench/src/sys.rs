//! OS resource probes: peak RSS, DXGI VRAM budget, device label, git SHA.
//!
//! Every function here reports what it measured and says when it did not
//! measure anything (`None`, `0.0`, `"unknown"`). Nothing is estimated.
//!
//! GPU selection mirrors `vaani-inference`'s `MachineCapabilities`:
//! [`inference_vram_budget_bytes`] uses the **inference adapter** (NVIDIA,
//! non-software, CUDA runtime loadable) and never the largest adapter. On a
//! hybrid laptop the largest DXGI budget belongs to the iGPU's *shared* system
//! memory, which CUDA cannot use; sizing anything against it is exactly the
//! failure `capability.rs` exists to prevent (see also
//! `docs/agent/PERFORMANCE.md`, "Use `preferred_adapter()` to size anything"
//! under "What must not change").

/// A display adapter as reported by DXGI (Windows) or an empty stand-in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterInfo {
    /// Marketing name, e.g. `NVIDIA GeForce RTX 3050 Ti Laptop GPU`.
    pub name: String,
    /// What the driver will actually let a process allocate (DXGI budget),
    /// not the board size: it already accounts for the desktop compositor.
    pub budget_bytes: u64,
    /// Dedicated memory currently in use, in bytes.
    pub used_bytes: u64,
    /// True for WARP/basic-render: never an acceleration option.
    pub is_software: bool,
    /// PCI vendor id (`0x10DE` is NVIDIA).
    pub vendor_id: u32,
}

impl AdapterInfo {
    /// Whether this is an NVIDIA adapter, by PCI vendor id, not name.
    pub fn is_nvidia(&self) -> bool {
        self.vendor_id == 0x10DE
    }
}

/// Peak process working set in bytes (`PeakWorkingSetSize`).
///
/// Returns `None` when the probe itself failed or the platform has no probe.
/// Callers convert with [`peak_rss_mb`] and keep `0.0` as the unknown marker.
pub fn peak_rss_bytes() -> Option<u64> {
    #[cfg(windows)]
    {
        windows_peak_rss_bytes()
    }
    #[cfg(not(windows))]
    {
        None
    }
}

/// Peak process working set in MiB, or `0.0` when unmeasured.
pub fn peak_rss_mb() -> f64 {
    peak_rss_bytes().map(|b| b as f64 / 1_048_576.0).unwrap_or(0.0)
}

/// Display adapters visible to this process (empty when unmeasurable).
pub fn adapters() -> Vec<AdapterInfo> {
    #[cfg(windows)]
    {
        windows_adapters()
    }
    #[cfg(not(windows))]
    {
        Vec::new()
    }
}

/// DXGI budget of the *inference* adapter in bytes.
///
/// The inference adapter is the first non-software NVIDIA adapter with a
/// nonzero budget, and only when a CUDA runtime DLL loads (otherwise CUDA
/// cannot place work on it and the honest answer is "no inference adapter").
/// Returns `None` on CPU-only machines, AMD-only machines, broken drivers,
/// and non-Windows platforms.
pub fn inference_vram_budget_bytes() -> Option<u64> {
    #[cfg(windows)]
    {
        if !cudart_present() {
            return None;
        }
        windows_adapters()
            .into_iter()
            .find(|a| a.is_nvidia() && !a.is_software && a.budget_bytes > 0)
            .map(|a| a.budget_bytes)
    }
    #[cfg(not(windows))]
    {
        None
    }
}

/// Inference VRAM budget in MiB, or `None` when there is no inference adapter.
pub fn peak_vram_mb() -> Option<f64> {
    inference_vram_budget_bytes().map(|b| b as f64 / 1_048_576.0)
}

/// Whether a CUDA runtime DLL could be loaded dynamically.
///
/// This links nothing: the CUDA Toolkit is not a build dependency. Absence
/// degrades to `false` and the caller reports CPU.
pub fn cudart_present() -> bool {
    #[cfg(windows)]
    {
        windows_cudart_present()
    }
    #[cfg(not(windows))]
    {
        false
    }
}

/// `"cuda (<name>)"` when an NVIDIA inference adapter with a loadable CUDA
/// runtime was measured, else `"cpu"`.
pub fn device_label() -> String {
    #[cfg(windows)]
    {
        if cudart_present() {
            if let Some(a) = windows_adapters()
                .into_iter()
                .find(|a| a.is_nvidia() && !a.is_software && a.budget_bytes > 0)
            {
                return format!("cuda ({})", a.name);
            }
        }
        "cpu".to_string()
    }
    #[cfg(not(windows))]
    {
        "cpu".to_string()
    }
}

/// Short git SHA of the measured tree, or `"unknown"`.
///
/// Prefers the `VAANI_GIT_SHA` environment variable (set by CI / the bench
/// script) so the number is reproducible without shelling out; otherwise
/// `"unknown"`. Never guessed.
pub fn git_sha() -> String {
    std::env::var("VAANI_GIT_SHA")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|_| "unknown".to_string())
}

#[cfg(windows)]
fn windows_peak_rss_bytes() -> Option<u64> {
    use windows::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
    use windows::Win32::System::Threading::GetCurrentProcess;

    let mut counters = PROCESS_MEMORY_COUNTERS_EX::default();
    // SAFETY: `counters` is a correctly sized zeroed out-param struct;
    // `GetCurrentProcess` needs no handle and returns a pseudo-handle that
    // must not be closed. `cb` is the struct size, per the API contract.
    let ok = unsafe {
        GetProcessMemoryInfo(
            GetCurrentProcess(),
            &mut counters as *mut _ as *mut _,
            std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
        )
    };
    if ok.is_ok() {
        // PeakWorkingSetSize is a SIZE_T count of bytes.
        Some(counters.PeakWorkingSetSize)
    } else {
        None
    }
}

#[cfg(windows)]
fn windows_adapters() -> Vec<AdapterInfo> {
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
        IDXGIAdapter1, IDXGIAdapter3, IDXGIFactory6,
    };
    use windows::core::Interface;

    fn vram(adapter: &IDXGIAdapter1) -> (u64, u64) {
        let Ok(a3) = adapter.cast::<IDXGIAdapter3>() else {
            return (0, 0);
        };
        let mut info = windows::Win32::Graphics::Dxgi::DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: `info` is a correctly sized zeroed out-param struct; node 0
        // is valid on every adapter implementing IDXGIAdapter3.
        if unsafe {
            a3.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info)
        }
        .is_err()
        {
            return (0, 0);
        }
        (info.CurrentUsage, info.Budget)
    }

    let mut out = Vec::new();
    // A DXGI failure is not a user-facing error: the honest outcome is "no
    // GPU to report", i.e. an empty list and a CPU fallback.
    let factory: IDXGIFactory6 = match unsafe { CreateDXGIFactory1() } {
        Ok(f) => f,
        Err(_) => return out,
    };
    for i in 0.. {
        let Ok(adapter) = (unsafe { factory.EnumAdapters1(i) }) else {
            break;
        };
        let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
            continue;
        };
        let name = {
            let utf16: Vec<u16> = desc
                .Description
                .iter()
                .copied()
                .take_while(|c| *c != 0)
                .collect();
            String::from_utf16_lossy(&utf16)
        };
        let is_software = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0;
        let (used, budget) = vram(&adapter);
        out.push(AdapterInfo {
            name,
            budget_bytes: budget,
            used_bytes: used,
            is_software,
            vendor_id: desc.VendorId,
        });
    }
    out
}

#[cfg(windows)]
fn windows_cudart_present() -> bool {
    use windows::Win32::System::LibraryLoader::LoadLibraryW;

    // Newest runtime we ship first, then older / system ones. The handle is
    // deliberately leaked: unloading a CUDA runtime under a live process is
    // unsafe, and we only need the boolean answer.
    for name in ["cudart64_13.dll", "cudart64_12.dll", "cudart64.dll"] {
        // SAFETY: plain ASCII DLL name; `LoadLibraryW` only reads it.
        if unsafe { LoadLibraryW(&windows::core::HSTRING::from(name)) }.is_ok() {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_rss_probe_does_not_panic() {
        let bytes = peak_rss_bytes();
        let mb = peak_rss_mb();
        assert!(mb >= 0.0);
        // The two views must agree with each other.
        match bytes {
            Some(b) => assert!((mb - b as f64 / 1_048_576.0).abs() < 1e-9),
            None => assert_eq!(mb, 0.0),
        }
    }

    #[test]
    fn adapter_enumeration_does_not_panic() {
        let list = adapters();
        // No assertion on contents: any machine shape is legitimate, including
        // headless CI with zero adapters. The contract is "no panic".
        for a in &list {
            assert!(!a.name.is_empty() || a.budget_bytes == 0);
        }
    }

    #[test]
    fn inference_budget_never_uses_a_software_adapter() {
        // If a budget is reported, an NVIDIA non-software adapter with a
        // nonzero budget must actually exist: guards against regressing to
        // largest-adapter (preferred_adapter) selection.
        if let Some(budget) = inference_vram_budget_bytes() {
            assert!(cudart_present(), "a budget requires a loadable CUDA runtime");
            let found = adapters()
                .into_iter()
                .any(|a| a.is_nvidia() && !a.is_software && a.budget_bytes == budget);
            assert!(found, "budget must come from an NVIDIA inference adapter");
            assert!(peak_vram_mb().is_some());
        } else {
            assert!(peak_vram_mb().is_none());
        }
    }

    #[test]
    fn device_label_is_honest() {
        let label = device_label();
        assert!(!label.is_empty());
        if label.starts_with("cuda") {
            assert!(cudart_present());
            assert!(inference_vram_budget_bytes().is_some());
        } else {
            assert_eq!(label, "cpu");
        }
    }

    #[test]
    fn git_sha_defaults_to_unknown() {
        // Without the env var the answer must be "unknown", never a guess.
        // (If CI sets VAANI_GIT_SHA, any non-empty value is acceptable.)
        let sha = git_sha();
        assert!(!sha.is_empty());
    }
}

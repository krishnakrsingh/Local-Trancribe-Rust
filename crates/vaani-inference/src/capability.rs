//! Minimal Windows capability probes.
//!
//! Everything here answers a question the product needs to answer honestly:
//! "is GPU acceleration actually going to work, and on what hardware?" It does
//! that with two sources and no new dependencies:
//!
//! * **DXGI** for adapter enumeration, names and dedicated VRAM. Available on
//!   every Windows machine, needs no vendor runtime, and reports the memory
//!   the display driver actually reserved -- which is the number that matters
//!   for a 4 GB laptop GPU shared with the desktop compositor.
//! * **`cudart64_*.dll`, loaded dynamically** for the driver version and the
//!   compute capability. We deliberately do *not* link against the CUDA
//!   runtime: linking would make the CUDA Toolkit a build requirement, which
//!   the product forbids. `LoadLibrary`/`GetProcAddress` gives us the same
//!   numbers with no link-time coupling at all, and degrades to "unknown" if
//!   the DLL is absent.
//!
//! The guiding rule: **we report what we measured, and `Unknown` when we did
//! not measure it.** A capability panel that guesses is worse than one that
//! says it does not know, because a user will make a decision based on it.

use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
    DXGI_QUERY_VIDEO_MEMORY_INFO, IDXGIAdapter1, IDXGIAdapter3, IDXGIFactory6,
};
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
use windows::Win32::System::Threading::{
    ALL_PROCESSOR_GROUPS, GetActiveProcessorCount, GetCurrentProcess, GetProcessAffinityMask,
};
use windows::core::Interface;

/// A display adapter as reported by DXGI.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct GpuAdapter {
    /// Marketing name, e.g. `NVIDIA GeForce RTX 3050 Ti Laptop GPU`.
    pub name: String,
    /// Dedicated video memory in bytes, as the driver reports it.
    pub dedicated_vram_bytes: u64,
    /// Dedicated memory currently used by other processes, in bytes.
    pub dedicated_vram_used_bytes: u64,
    /// True for a WARP/basic-render adapter, which must never be reported as
    /// an acceleration option.
    pub is_software: bool,
    /// PCI vendor id, used to recognise NVIDIA without string matching.
    pub vendor_id: u32,
}

impl GpuAdapter {
    /// Free dedicated VRAM, saturating at zero.
    pub fn free_vram_bytes(&self) -> u64 {
        self.dedicated_vram_bytes
            .saturating_sub(self.dedicated_vram_used_bytes)
    }

    /// Whether this is an NVIDIA adapter, by PCI vendor id rather than by name.
    pub fn is_nvidia(&self) -> bool {
        // 0x10DE is NVIDIA's PCI vendor id.
        self.vendor_id == 0x10DE
    }

    /// Human-readable size, e.g. `4.0 GB`.
    pub fn vram_display(&self) -> String {
        format!(
            "{:.1} GB",
            self.dedicated_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        )
    }
}

/// What the CUDA runtime reported, if it could be loaded.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct CudaProbe {
    /// `cudaDriverGetVersion`, e.g. 13030 for CUDA 13.3.
    pub driver_version: Option<u32>,
    /// Number of CUDA-capable devices, from `cudaGetDeviceCount`.
    pub device_count: Option<i32>,
    /// Compute capability major/minor of device 0, e.g. `8.6`.
    pub compute_capability: Option<(u8, u8)>,
    /// Which runtime DLL we managed to load, if any.
    pub runtime_dll: Option<String>,
    /// Why the probe failed, for the developer log and the diagnostics panel.
    pub error: Option<String>,
}

impl CudaProbe {
    /// Whether a working CUDA device was actually found.
    pub fn is_usable(&self) -> bool {
        self.device_count.unwrap_or(0) > 0 && self.error.is_none()
    }

    /// Minimum NVIDIA driver for the CUDA version our runtime is built against.
    ///
    /// ONNX Runtime's CUDA 13 builds require a 13.x driver (>= 580); CUDA 12
    /// builds work on >= 525. We ship CUDA 13 and record the requirement here
    /// so the settings panel can say *why* acceleration is unavailable instead
    /// of only that it is.
    /// `cudaDriverGetVersion` value corresponding to CUDA 13.0.
    pub const MIN_DRIVER_FOR_CUDA_13: u32 = 13_000;

    /// The driver version as `major.minor`, or `unknown`.
    pub fn driver_display(&self) -> String {
        match self.driver_version {
            // cudaDriverGetVersion returns 1000 * major + 10 * minor.
            Some(v) if v >= 1000 => {
                let (maj, min) = (v / 1000, (v % 1000) / 10);
                format!("{maj}.{min}")
            }
            Some(v) => format!("{v}"),
            None => "unknown".into(),
        }
    }
}

/// Machine-level facts, gathered once and shown in Settings.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct MachineCapabilities {
    pub adapters: Vec<GpuAdapter>,
    pub cuda: CudaProbe,
    pub logical_cpus: usize,
    pub physical_cpus: Option<usize>,
    /// Physical RAM in bytes.
    pub total_ram_bytes: u64,
    pub available_ram_bytes: u64,
    /// CPUs this process is allowed to run on, from its affinity mask.
    ///
    /// ONNX Runtime's thread pool is sized from this. Sizing from the machine
    /// count instead oversubscribes badly on a box where the app is pinned to
    /// half the cores, which shows up as a mysterious 2x slowdown.
    pub usable_cpus: usize,
}

impl MachineCapabilities {
    /// The adapter we would actually use for inference.
    ///
    /// Requires a hardware adapter with dedicated memory; software adapters
    /// (WARP) are excluded because ORT cannot use them and pretending
    /// otherwise would put a fake "GPU" row in the settings panel.
    /// The largest *hardware* adapter. **Not** an inference choice.
    ///
    /// Only for display purposes: naming the biggest card in the settings
    /// panel. Inference must use [`Self::inference_adapter`], because on a
    /// hybrid laptop this method returns the integrated GPU whose "dedicated"
    /// memory is shared system memory that CUDA cannot use.
    ///
    /// Excludes software adapters (WARP), which ONNX Runtime cannot use and
    /// which would otherwise show up as a fake GPU row.
    pub fn preferred_adapter(&self) -> Option<&GpuAdapter> {
        self.adapters
            .iter()
            .filter(|a| !a.is_software && a.dedicated_vram_bytes > 0)
            .max_by_key(|a| a.dedicated_vram_bytes)
    }

    /// The adapter ONNX Runtime will actually place work on, or `None` for CPU.
    ///
    /// This **prefers the NVIDIA adapter**, and that is not a cosmetic choice.
    /// On a hybrid laptop the integrated GPU reports its *shared* system memory
    /// through `IDXGIAdapter3::Budget`, so an AMD integrated part can appear to
    /// have 8 GB of "dedicated" VRAM while the discrete NVIDIA part reports a
    /// true 4 GB. Ranking purely by memory therefore picks the adapter we
    /// cannot use, and the product ends up claiming GPU acceleration while
    /// running on the CPU -- the exact failure this whole module exists to
    /// prevent. We only pick NVIDIA when the CUDA runtime actually reports a
    /// usable device; otherwise the honest answer is CPU, and we say so.
    pub fn inference_adapter(&self) -> Option<&GpuAdapter> {
        if self.cuda.is_usable()
            && let Some(nv) = self
                .adapters
                .iter()
                .find(|a| a.is_nvidia() && !a.is_software)
        {
            return Some(nv);
        }
        None
    }

    /// Whether ONNX Runtime will place work on a GPU on this machine.
    pub fn gpu_acceleration_available(&self) -> bool {
        self.inference_adapter().is_some()
    }

    /// A sensible ONNX Runtime intra-op thread count.
    ///
    /// We leave one core for the UI thread and the audio pipeline. Going all
    /// the way to `usable_cpus` is measurably slower on this class of machine:
    /// the encoder is already memory-bandwidth bound, and oversubscribing adds
    /// contention without adding throughput.
    /// The ONNX Runtime intra-op thread count to use on this machine.
    pub fn recommended_ort_threads(&self) -> usize {
        self.usable_cpus.saturating_sub(1).max(1)
    }

    /// Whether a model of `weights_bytes` plausibly fits in VRAM.
    ///
    /// Deliberately conservative: it reserves a fraction for the desktop
    /// compositor and for the encoder's activation memory, which grows with
    /// chunk length. A model that "fits" on paper and then OOMs at chunk 40 is
    /// a far worse experience than being told up front.
    /// Whether a model plausibly fits in currently-free VRAM.
    pub fn vram_fits(&self, weights_bytes: u64, headroom: f64) -> bool {
        let Some(a) = self.inference_adapter() else {
            return false;
        };
        // Activations for a Conformer encoder at a 20 s chunk run to several
        // hundred MB; callers pass that in as `headroom` bytes.
        (weights_bytes + headroom as u64) <= a.free_vram_bytes()
    }
}

// ---------------------------------------------------------------------------
// DXGI
// ---------------------------------------------------------------------------

/// Enumerate display adapters.
///
/// A failure here is not an error the user needs to see: a machine with no DXGI
/// at all is vanishingly rare, and the correct outcome is "no GPU to report",
/// not a failure. We return an empty list and let the caller fall back to CPU.
pub fn enumerate_adapters() -> Vec<GpuAdapter> {
    let mut out = Vec::new();
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
        let (used, total) = query_vram(&adapter);
        out.push(GpuAdapter {
            name,
            dedicated_vram_bytes: total,
            dedicated_vram_used_bytes: used,
            is_software,
            vendor_id: desc.VendorId,
        });
    }
    out
}

/// Dedicated memory for an adapter, via `IDXGIAdapter3`.
///
/// We report the *budget* rather than `DedicatedVideoMemory`: on WDDM the
/// latter is the physical board size while the budget is what the driver will
/// actually let a process allocate, and it already accounts for the desktop
/// compositor. Sizing a model against the budget is what prevents a 4 GB card
/// from failing to load a model that "fits".
fn query_vram(adapter: &IDXGIAdapter1) -> (u64, u64) {
    let Ok(a3) = adapter.cast::<IDXGIAdapter3>() else {
        return (0, 0);
    };
    let mut vram = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
    // SAFETY: `vram` is a correctly sized zeroed out-param struct, and node
    // 0 is a valid node on every adapter that implements IDXGIAdapter3.
    if unsafe { a3.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut vram) }.is_err() {
        return (0, 0);
    }
    (vram.CurrentUsage, vram.Budget)
}

// ---------------------------------------------------------------------------
// CUDA
// ---------------------------------------------------------------------------

type CudaDriverGetVersion = unsafe extern "system" fn(*mut i32) -> i32;
type CudaGetDeviceCount = unsafe extern "system" fn(*mut i32) -> i32;
type CudaGetDeviceProperties = unsafe extern "system" fn(*mut CudaDevicePropPrefix, i32) -> i32;
type CudaRuntimeGetVersion = unsafe extern "system" fn(*mut i32) -> i32;

/// `cudaDeviceProp` up to and including `minor`, in the exact declaration order
/// from CUDA 13's `driver_types.h`, so the field offsets match the C struct.
///
/// Two things make this worth getting exactly right, and both are traps:
///
/// 1. **The field order is not intuitive.** `major` and `minor` sit after the
///    name and a dozen size fields, not at the front. Reading them from offset
///    0 yields the first two *characters* of the device name, so an RTX 3050 Ti
///    reports compute capability "78.73" ('N' = 78, 'I' = 73).
/// 2. **CUDA 13 inserted `uuid`, `luid` and `luidDeviceNodeMask` after
///    `name`.** That is what moved `major` from offset 328 (CUDA 12) to 360
///    (CUDA 13), and it is why a layout copied from an older header silently
///    returns garbage on a current driver.
///
/// Because we pin the exact `cudart64_13.dll` in `scripts/fetch_runtime.ps1`,
/// the layout is pinned too. `device_prop_field_offsets_match_the_c_header`
/// pins the offsets, and
/// `compute_capability_read_from_the_real_device_is_plausible` proves the
/// running runtime actually filled in what we think it did -- including a
/// check that the value is not the device name's first two bytes.
#[repr(C)]
#[derive(Clone, Copy)]
struct CudaDevicePropPrefix {
    name: [i8; 256],
    uuid: [u8; 16],
    luid: [i8; 8],
    luid_device_node_mask: u32,
    _pad0: u32,
    total_global_mem: u64,
    shared_mem_per_block: u64,
    regs_per_block: i32,
    warp_size: i32,
    mem_pitch: u64,
    max_threads_per_block: i32,
    max_threads_dim: [i32; 3],
    max_grid_size: [i32; 3],
    _pad1: u32,
    total_const_mem: u64,
    major: i32,
    minor: i32,
    /// The rest of CUDA 13's 1032-byte struct, never read.
    _rest: [u8; 1032 - 368],
}

impl CudaDevicePropPrefix {
    /// A zeroed property struct, as `cudaGetDeviceProperties` expects.
    fn zeroed() -> Self {
        // SAFETY: the struct is a plain `repr(C)` aggregate of integers and
        // arrays with no padding invariants, so an all-zero bit pattern is a
        // valid value for every field.
        unsafe { std::mem::zeroed() }
    }

    fn name_str(&self) -> String {
        let bytes: Vec<u8> = self
            .name
            .iter()
            .map(|c| if *c > 0 { *c as u8 } else { 0 })
            .collect();
        let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
        String::from_utf8_lossy(&bytes[..end]).trim().to_string()
    }
}

/// Probe the CUDA runtime by loading it dynamically.
///
/// Never links `cudart`, so the CUDA Toolkit is not a build or runtime
/// dependency. If the DLL cannot be found the probe reports `error` and the
/// caller falls back to the CPU execution provider.
pub fn probe_cuda(dll_search_paths: &[std::path::PathBuf]) -> CudaProbe {
    let mut probe = CudaProbe::default();
    let Ok(module) = (unsafe { load_cudart(dll_search_paths) }) else {
        probe.error = Some("CUDA runtime DLL not found next to the application".into());
        return probe;
    };
    probe.runtime_dll = Some(module.0.clone());

    // SAFETY: `module.0` is a live HMODULE from LoadLibrary; we only read
    // function pointers out of it and never unload it (see the Drop impl).
    unsafe {
        let Some(get_driver) = sym(module.1, "cudaDriverGetVersion") else {
            probe.error = Some("cudaDriverGetVersion missing from the CUDA runtime".into());
            return probe;
        };
        let driver_fn: CudaDriverGetVersion = std::mem::transmute(get_driver);
        let mut v = 0i32;
        if driver_fn(&mut v) != 0 {
            probe.error = Some("cudaDriverGetVersion failed; no NVIDIA driver present".into());
            return probe;
        }
        probe.driver_version = Some(v as u32);

        if let Some(p) = sym(module.1, "cudaRuntimeGetVersion") {
            let f: CudaRuntimeGetVersion = std::mem::transmute(p);
            let _ = f(&mut v);
        }

        if let Some(p) = sym(module.1, "cudaGetDeviceCount") {
            let f: CudaGetDeviceCount = std::mem::transmute(p);
            let mut n = -1i32;
            if f(&mut n) == 0 {
                probe.device_count = Some(n);
            }
        }

        if let (Some(p), Some(n)) = (sym(module.1, "cudaGetDeviceProperties"), probe.device_count)
            && n > 0
        {
            let f: CudaGetDeviceProperties = std::mem::transmute(p);
            let mut props = CudaDevicePropPrefix::zeroed();
            if f(&mut props, 0) == 0 {
                probe.compute_capability = Some((props.major as u8, props.minor as u8));
            }
        }
    }
    probe
}

/// Try each candidate directory, then the system search path.
unsafe fn load_cudart(paths: &[std::path::PathBuf]) -> Result<(String, HMODULE), String> {
    // Prefer the newest runtime we ship, then fall back to whatever the driver
    // or the system provides. ORT links a specific major version name, so we
    // only need *a* runtime here to answer "is there a working driver".
    const CANDIDATES: &[&str] = &["cudart64_13.dll", "cudart64_12.dll", "cudart64.dll"];
    for name in CANDIDATES {
        for dir in paths {
            let p = dir.join(name);
            // SAFETY: p is a path we just constructed; LoadLibraryW only
            // reads it. The returned handle is deliberately leaked below.
            if let Ok(h) = unsafe { LoadLibraryW(&windows::core::HSTRING::from(p.as_os_str())) } {
                return Ok((p.display().to_string(), h));
            }
        }
    }
    // Fall back to the standard search order.
    for name in CANDIDATES {
        // SAFETY: plain ASCII DLL name, NUL-terminated by HSTRING; only read.
        if let Ok(h) = unsafe { LoadLibraryW(&windows::core::HSTRING::from(*name)) } {
            return Ok(((*name).to_string(), h));
        }
    }
    Err("no cudart64_*.dll in the application directory or on PATH".into())
}

/// Resolve an export by name.
///
/// `GetProcAddress` takes an ANSI string, not a wide one, so the name has to be
/// NUL-terminated bytes. Every name we look up is a plain ASCII identifier, so
/// the conversion cannot fail and does not need to.
unsafe fn sym(module: HMODULE, name: &str) -> Option<unsafe extern "system" fn() -> isize> {
    let mut c = name.as_bytes().to_vec();
    c.push(0);
    // SAFETY: c is a NUL-terminated ASCII buffer that outlives the call, and
    // module is a live handle from LoadLibraryW.
    unsafe { GetProcAddress(module, windows::core::PCSTR(c.as_ptr())) }
}

/// Name of CUDA device 0, if the runtime is present.
///
/// DXGI already gives us an adapter name, so this exists only for the settings
/// panel's "as CUDA sees it" line, which is genuinely more accurate when
/// multiple adapters are installed.
pub fn cuda_device_name(dll_search_paths: &[std::path::PathBuf]) -> Option<String> {
    let Ok(module) = (unsafe { load_cudart(dll_search_paths) }) else {
        return None;
    };
    let p = unsafe { sym(module.1, "cudaGetDeviceProperties") }?;
    let f: CudaGetDeviceProperties = unsafe { std::mem::transmute(p) };
    let mut props = CudaDevicePropPrefix::zeroed();
    if unsafe { f(&mut props, 0) } != 0 {
        return None;
    }
    let n = props.name_str();
    (!n.is_empty()).then_some(n)
}

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

fn memory() -> (u64, u64) {
    let mut s = MEMORYSTATUSEX {
        dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
        ..Default::default()
    };
    // SAFETY: `s` is a correctly sized, zeroed struct with dwLength set, which
    // is the documented calling contract.
    if unsafe { GlobalMemoryStatusEx(&mut s) }.is_ok() {
        (s.ullTotalPhys as u64, s.ullAvailPhys as u64)
    } else {
        (0, 0)
    }
}

fn processor_counts() -> (usize, Option<usize>) {
    // SAFETY: a pure query with no preconditions.
    let logical = unsafe { GetActiveProcessorCount(ALL_PROCESSOR_GROUPS) }.max(1) as usize;
    (logical, read_physical_core_count())
}

/// Physical cores, read from the CPU's `Identifier` registry value.
///
/// `GetLogicalProcessorInformation` reports topology *records* rather than a
/// core count, and interpreting them correctly across processor groups, hybrid
/// CPUs and nested caches is a lot of code for a number that only ever labels a
/// row in the settings panel. The registry value is a single DWORD written by
/// Windows itself. A missing or nonsensical value is reported as `None` rather
/// than guessed at.
fn read_physical_core_count() -> Option<usize> {
    use windows::Win32::System::Registry::{HKEY_LOCAL_MACHINE, RRF_RT_REG_DWORD, RegGetValueW};

    const SUBKEY: &str = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0";
    const VALUE: &str = "Identifier";

    // PCWSTR is a pointer to a NUL-terminated UTF-16 buffer, so both strings
    // need their own owned buffer that outlives the call.
    let subkey: Vec<u16> = SUBKEY.encode_utf16().chain(std::iter::once(0)).collect();
    let value: Vec<u16> = VALUE.encode_utf16().chain(std::iter::once(0)).collect();

    let mut size: u32 = std::mem::size_of::<u32>() as u32;
    let mut out: u32 = 0;
    // SAFETY: `out` and `size` are live and correctly sized; both strings are
    // NUL-terminated and outlive the call; the key handle is a constant.
    let status = unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            windows::core::PCWSTR(subkey.as_ptr()),
            windows::core::PCWSTR(value.as_ptr()),
            RRF_RT_REG_DWORD,
            None,
            Some(&mut out as *mut u32 as *mut std::ffi::c_void),
            Some(&mut size),
        )
    };
    // RegGetValueW returns a WIN32_ERROR directly rather than a Result.
    if status != windows::Win32::Foundation::ERROR_SUCCESS {
        return None;
    }
    (out > 0).then_some(out as usize)
}
/// CPUs this process may run on.
///
/// A process pinned to a subset must not size a thread pool from the machine
/// count: the oversubscription shows up as a slowdown with no obvious cause.
fn usable_cpus() -> usize {
    let mut process_mask = 0usize;
    let mut system_mask = 0usize;
    // SAFETY: both out-params are live `usize`s, which is the width the API
    // expects on a 64-bit process, and `GetCurrentProcess` needs no handle.
    let ok = unsafe {
        GetProcessAffinityMask(
            GetCurrentProcess(),
            &mut process_mask as *mut usize,
            &mut system_mask as *mut usize,
        )
    }
    .is_ok();
    if !ok {
        return unsafe { GetActiveProcessorCount(ALL_PROCESSOR_GROUPS) }.max(1) as usize;
    }
    (process_mask.count_ones() as usize).max(1)
}
/// Gather everything. Cheap enough to call at startup and after a GPU change.
/// Probe everything. Safe to call at startup and after a GPU change.
pub fn detect(dll_search_paths: &[std::path::PathBuf]) -> MachineCapabilities {
    let (logical, physical) = processor_counts();
    let (total_ram, avail_ram) = memory();
    MachineCapabilities {
        adapters: enumerate_adapters(),
        cuda: probe_cuda(dll_search_paths),
        logical_cpus: logical,
        physical_cpus: physical,
        total_ram_bytes: total_ram,
        available_ram_bytes: avail_ram,
        usable_cpus: usable_cpus(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(name: &str, vram_mb: u64, used_mb: u64, software: bool) -> GpuAdapter {
        GpuAdapter {
            name: name.into(),
            dedicated_vram_bytes: vram_mb * 1024 * 1024,
            dedicated_vram_used_bytes: used_mb * 1024 * 1024,
            is_software: software,
            vendor_id: 0x10DE,
        }
    }

    fn machine(adapters: Vec<GpuAdapter>, usable: usize) -> MachineCapabilities {
        MachineCapabilities {
            adapters,
            cuda: CudaProbe::default(),
            logical_cpus: usable * 2,
            physical_cpus: Some(usable),
            total_ram_bytes: 16 * 1024 * 1024 * 1024,
            available_ram_bytes: 8 * 1024 * 1024 * 1024,
            usable_cpus: usable,
        }
    }

    #[test]
    fn software_adapters_are_never_preferred() {
        // WARP reports a large "dedicated" memory figure. Offering it as a GPU
        // option would be a lie and would be slower than the CPU anyway.
        let m = machine(vec![gpu("Microsoft Basic Render Driver", 0, 0, true)], 8);
        assert!(m.preferred_adapter().is_none());
    }

    #[test]
    fn adapter_with_zero_vram_is_never_preferred() {
        let m = machine(vec![gpu("Odd", 0, 0, false)], 8);
        assert!(m.preferred_adapter().is_none());
    }

    #[test]
    fn largest_hardware_adapter_wins() {
        let m = machine(
            vec![
                gpu("Integrated", 512, 100, false),
                gpu("RTX 3050 Ti", 4096, 500, false),
            ],
            8,
        );
        assert_eq!(m.preferred_adapter().unwrap().name, "RTX 3050 Ti");
    }

    #[test]
    fn hybrid_laptop_picks_nvidia_not_the_shared_memory_igpu() {
        // The real shape of the development machine: an AMD integrated part
        // reporting 8 GB of *shared* system memory, and a discrete RTX 3050 Ti
        // reporting a true 3.5 GB. Ranking by memory alone picks the AMD part,
        // which CUDA cannot use.
        let mut igpu = gpu("AMD Radeon(TM) Graphics", 8082, 0, false);
        igpu.vendor_id = 0x1002; // AMD
        let mut dgpu = gpu("NVIDIA GeForce RTX 3050 Ti Laptop GPU", 3531, 72, false);
        dgpu.vendor_id = 0x10DE; // NVIDIA
        let mut m = machine(vec![igpu, dgpu], 16);
        m.cuda = CudaProbe {
            device_count: Some(1),
            ..Default::default()
        };

        // Display order still reports the largest.
        assert_eq!(
            m.preferred_adapter().unwrap().name,
            "AMD Radeon(TM) Graphics"
        );
        // But inference goes to the card CUDA can actually use.
        let inf = m
            .inference_adapter()
            .expect("CUDA is usable, so a GPU must be chosen");
        assert_eq!(inf.name, "NVIDIA GeForce RTX 3050 Ti Laptop GPU");
        assert!(m.gpu_acceleration_available());
        // And the 858 MB encoder + 400 MB activations fits on the 3.5 GB card.
        let mib = 1024u64 * 1024;
        assert!(m.vram_fits(858 * mib, 400.0 * mib as f64));
    }

    #[test]
    fn no_cuda_means_no_inference_adapter_even_with_a_gpu() {
        // An AMD-only machine, or an NVIDIA machine with a broken driver, must
        // report CPU rather than optimistically claiming acceleration.
        let mut nv = gpu("NVIDIA RTX", 4096, 0, false);
        nv.vendor_id = 0x10DE;
        let m = machine(vec![nv], 8);
        assert!(m.preferred_adapter().is_some());
        assert!(m.inference_adapter().is_none());
        assert!(!m.gpu_acceleration_available());
    }

    #[test]
    fn free_vram_saturates_at_zero() {
        // A driver can report usage above budget transiently during a resize.
        let a = gpu("x", 1024, 2048, false);
        assert_eq!(a.free_vram_bytes(), 0);
    }

    #[test]
    fn ort_threads_leave_a_core_for_the_ui() {
        // On a 16-thread box this must be 15, not 16: the UI thread and the
        // audio decoder are not idle while ORT is running.
        let m = machine(vec![], 16);
        assert_eq!(m.recommended_ort_threads(), 15);
        let m = machine(vec![], 1);
        assert_eq!(m.recommended_ort_threads(), 1);
        let m = machine(vec![], 0);
        assert_eq!(m.recommended_ort_threads(), 1);
    }

    #[test]
    fn ort_threads_respect_process_affinity() {
        // A process pinned to 4 of 16 cores must size its pool from 4.
        let m = machine(vec![], 4);
        assert_eq!(m.recommended_ort_threads(), 3);
    }

    #[test]
    fn vram_fit_reserves_headroom() {
        // 4096 MiB board, 512 MiB already used by the desktop => ~3584 MiB free.
        let mut m = machine(vec![gpu("RTX 3050 Ti", 4096, 512, false)], 8);
        // `vram_fits` reasons about the *inference* adapter, which requires a
        // usable CUDA device. Without one the honest answer is "does not fit
        // on a GPU", which the next test covers.
        m.cuda = CudaProbe {
            device_count: Some(1),
            ..Default::default()
        };
        let mib = 1024u64 * 1024;
        // A 900 MB (858 MiB) fp16 encoder plus 400 MiB of activations fits.
        assert!(m.vram_fits(858 * mib, 400.0 * mib as f64));
        // 3.2 GiB of weights plus activations does not, and the failure has to
        // be visible *before* the model is loaded, not as an OOM at chunk 40.
        assert!(!m.vram_fits(3200 * mib, 400.0 * mib as f64));
        // The boundary is inclusive: exactly filling the free budget is a fit.
        assert!(m.vram_fits(3184 * mib, 400.0 * mib as f64));
    }

    #[test]
    fn vram_fit_is_false_without_a_gpu() {
        let m = machine(vec![], 8);
        assert!(!m.vram_fits(1, 0.0));
    }

    #[test]
    fn driver_version_formatting() {
        // `cudaDriverGetVersion` encodes a version as `major * 1000 + minor * 10`.
        let p = CudaProbe {
            driver_version: Some(13_030),
            ..Default::default()
        };
        assert_eq!(p.driver_display(), "13.3");
        let p = CudaProbe {
            driver_version: Some(12_080),
            ..Default::default()
        };
        assert_eq!(p.driver_display(), "12.8");
        // The development machine's actual driver, 610.88 -> CUDA 13.3.
        let p = CudaProbe {
            driver_version: Some(13_030),
            ..Default::default()
        };
        assert_eq!(p.driver_display(), "13.3");
        let p = CudaProbe::default();
        assert_eq!(p.driver_display(), "unknown");
    }

    #[test]
    fn cuda_probe_usability_requires_a_device_and_no_error() {
        assert!(!CudaProbe::default().is_usable());
        assert!(
            !CudaProbe {
                device_count: Some(0),
                ..Default::default()
            }
            .is_usable()
        );
        assert!(
            !CudaProbe {
                device_count: Some(1),
                error: Some("boom".into()),
                ..Default::default()
            }
            .is_usable()
        );
        assert!(
            CudaProbe {
                device_count: Some(1),
                ..Default::default()
            }
            .is_usable()
        );
    }

    #[test]
    fn device_prop_field_offsets_match_the_c_header() {
        use std::mem::{align_of, size_of};
        // Offsets from CUDA 13.3's driver_types.h on x86-64. `major` at 360 is
        // the give-away: CUDA 12 had it at 328, before `uuid`/`luid` were
        // inserted after `name`.
        assert_eq!(size_of::<CudaDevicePropPrefix>(), 1032);
        assert_eq!(align_of::<CudaDevicePropPrefix>(), 8);

        let p = CudaDevicePropPrefix::zeroed();
        let base = &p as *const _ as usize;
        fn off<T>(f: *const T, base: usize) -> usize {
            f as *const u8 as usize - base
        }
        assert_eq!(off(std::ptr::addr_of!(p.name), base), 0);
        assert_eq!(off(std::ptr::addr_of!(p.uuid), base), 256);
        assert_eq!(off(std::ptr::addr_of!(p.luid), base), 272);
        assert_eq!(off(std::ptr::addr_of!(p.luid_device_node_mask), base), 280);
        assert_eq!(off(std::ptr::addr_of!(p.total_global_mem), base), 288);
        assert_eq!(off(std::ptr::addr_of!(p.regs_per_block), base), 304);
        assert_eq!(off(std::ptr::addr_of!(p.mem_pitch), base), 312);
        assert_eq!(off(std::ptr::addr_of!(p.max_threads_per_block), base), 320);
        assert_eq!(off(std::ptr::addr_of!(p.total_const_mem), base), 352);
        assert_eq!(off(std::ptr::addr_of!(p.major), base), 360);
        assert_eq!(off(std::ptr::addr_of!(p.minor), base), 364);
    }

    #[test]
    fn device_prop_name_extraction() {
        let mut p = CudaDevicePropPrefix::zeroed();
        for (i, c) in b"NVIDIA GeForce RTX 3050 Ti".iter().enumerate() {
            p.name[i] = *c as i8;
        }
        assert_eq!(p.name_str(), "NVIDIA GeForce RTX 3050 Ti");
    }

    #[test]
    fn detect_does_not_panic_and_reports_something() {
        // The real probe runs against this machine. Whatever it finds, it must
        // return a coherent struct rather than panicking, because this runs on
        // the UI startup path.
        let m = detect(&[]);
        assert!(m.logical_cpus >= 1);
        assert!(m.usable_cpus >= 1);
        assert!(m.usable_cpus <= m.logical_cpus);
        assert!(m.total_ram_bytes > 0, "RAM query failed");
        assert!(m.available_ram_bytes <= m.total_ram_bytes);
    }

    #[test]
    fn compute_capability_read_from_the_real_device_is_plausible() {
        // The offset test above pins the layout, but only a real device proves
        // the runtime filled in what we think it did. A value outside 2.0..=12.9
        // means we are reading the wrong bytes, and a name whose first two
        // bytes match the reported major/minor means exactly that.
        let p = CudaProbe::default();
        let probe = probe_cuda(&[]);
        if !probe.is_usable() {
            println!("skipping: no usable CUDA device on this machine");
            return;
        }
        let (major, minor) = probe
            .compute_capability
            .expect("usable CUDA implies a capability");
        assert!(
            (2..=12).contains(&major),
            "implausible compute capability {major}.{minor}; \
             the cudaDeviceProp layout is probably wrong"
        );
        assert!(
            minor <= 9,
            "implausible compute capability minor version {minor}"
        );
        if let Some(name) = cuda_device_name(&[]) {
            let b = name.as_bytes();
            // 'N' == 78, 'I' == 73: the failure mode of reading major/minor out
            // of the name field. If the reported numbers look like ASCII, the
            // offsets are wrong.
            assert!(
                !(b.len() >= 2 && major == b[0] && minor == b[1]),
                "compute capability {major}.{minor} is the first two bytes of {name:?}; \
                 the cudaDeviceProp layout is wrong"
            );
        }
        let _ = p;
    }

    #[test]
    fn vram_display_is_human_readable() {
        assert_eq!(gpu("x", 4096, 0, false).vram_display(), "4.0 GB");
    }
}

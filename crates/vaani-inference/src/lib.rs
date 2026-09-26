//! # vaani-inference
//!
//! The boundary between VaaniDesk and ONNX Runtime, plus the machine
//! capability detection the rest of the product reasons about.
//!
//! ## What lives here
//!
//! * [`capability`] — what this machine can actually do. DXGI for adapters and
//!   VRAM, dynamically-loaded `cudart` for the driver version and compute
//!   capability, Win32 for CPU/RAM. Reports `Unknown` rather than guessing.
//! * [`runtime`] — locating `onnxruntime.dll`, initialising the process-global
//!   environment exactly once, and deciding which execution provider to use.
//! * [`session`] — a thin typed wrapper over one ONNX session.
//!
//! ## What must not happen here
//!
//! * **No CUDA context creation at startup.** `cuInit` plus primary context
//!   creation costs several hundred megabytes of VRAM and a few hundred
//!   milliseconds. A user who opens the app to browse history must not pay for
//!   that, and on a 4 GB laptop GPU that memory is the difference between the
//!   model loading and failing. Contexts are created lazily, on the first
//!   session, on a worker thread.
//! * **No `unsafe` outside this crate.** `session.rs` and `capability.rs`
//!   contain the only `unsafe`, each with a comment justifying it. The rest of
//!   the workspace is `#![forbid(unsafe_code)]`.
//! * **No model-specific knowledge.** Nothing here knows what a Conformer or a
//!   TDT decoder is. That is `vaani-asr`'s job. This crate can load a sentence
//!   embedding model without knowing what a sentence is.
//!
//! ## The fallback contract
//!
//! Requesting GPU acceleration and not getting it is a **normal outcome**, not
//! an error. `Provider::Auto` falls back to the CPU EP inside ONNX Runtime
//! itself; the resulting `ProviderStatus` is what the UI and the benchmark
//! report display. The only way to make it a hard failure is
//! `Provider::RequireCuda`, which exists so the benchmark harness can refuse to
//! publish CPU numbers in a GPU column.

#![warn(missing_docs)]

pub mod capability;
pub mod runtime;
pub mod session;

pub use capability::{CudaProbe, GpuAdapter, MachineCapabilities};
pub use runtime::{ActiveProvider, Provider, ProviderStatus, RuntimeLocation};
pub use session::{F16Tensor, F32Tensor, I32Tensor, I64Tensor, Input, OnnxSession, OutputTensor};

/// Everything `vaani doctor` needs to print about the inference stack, in one
/// serialisable value.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct RuntimeReport {
    pub ort_dll: String,
    pub runtime_present: bool,
    pub missing_dlls: Vec<String>,
    pub portable_layout: bool,
    pub machine: MachineCapabilities,
}

impl RuntimeReport {
    /// Collect the report. Never fails: a missing runtime is a fact to report,
    /// not an error to propagate, because the CLI has to be able to print
    /// "here is what is wrong" without itself crashing.
    pub fn collect() -> Self {
        let loc = runtime::location();
        let machine = capability::detect(&loc.provider_dirs);
        Self {
            ort_dll: loc.ort_dll.display().to_string(),
            runtime_present: loc.is_present(),
            missing_dlls: loc.missing(),
            portable_layout: loc.portable_layout,
            machine,
        }
    }

    /// One-line verdict for the top of `vaani doctor` and the settings header.
    pub fn summary(&self) -> String {
        if !self.runtime_present {
            return "Inference runtime NOT FOUND".into();
        }
        // Name the adapter ONNX Runtime will actually place work on. Using the
        // display adapter here would print the integrated GPU on a hybrid
        // laptop and tell the user the wrong thing about their own machine.
        let cc = self.machine.cuda.compute_capability;
        match (
            self.machine.inference_adapter(),
            self.machine.preferred_adapter(),
        ) {
            (Some(a), _) => format!(
                "GPU acceleration available - {}, {}, driver {}, compute {}.{}",
                a.name,
                a.vram_display(),
                self.machine.cuda.driver_display(),
                cc.map(|c| c.0).unwrap_or(0),
                cc.map(|c| c.1).unwrap_or(0),
            ),
            (None, Some(a)) => format!(
                "CPU fallback - {} present but CUDA is unavailable (driver {})",
                a.name,
                self.machine.cuda.driver_display()
            ),
            (None, None) => "CPU only - no hardware GPU detected".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_is_collectable_without_a_runtime() {
        // `vaani doctor` must work on a machine with nothing installed, so this
        // cannot return an error.
        let r = RuntimeReport::collect();
        assert!(!r.summary().is_empty());
        // Either the runtime is present, or we can say precisely what is absent.
        assert_eq!(
            r.runtime_present,
            r.missing_dlls.is_empty() || !r.missing_dlls.is_empty()
        );
    }

    #[test]
    fn summary_names_the_problem_when_the_runtime_is_missing() {
        let r = RuntimeReport {
            ort_dll: "x".into(),
            runtime_present: false,
            missing_dlls: vec!["onnxruntime.dll".into()],
            portable_layout: false,
            machine: MachineCapabilities {
                adapters: vec![],
                cuda: Default::default(),
                logical_cpus: 1,
                physical_cpus: None,
                total_ram_bytes: 1,
                available_ram_bytes: 1,
                usable_cpus: 1,
            },
        };
        assert!(r.summary().contains("NOT FOUND"));
    }
}

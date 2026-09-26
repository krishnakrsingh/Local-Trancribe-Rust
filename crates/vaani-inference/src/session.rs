//! A thin, typed wrapper over an ONNX Runtime session.
//!
//! # Scope
//!
//! This exists to do three things and no more:
//!
//! 1. Build a session with an explicit execution provider and thread count.
//! 2. Run it with named `f32`/`i32`/`i64` tensors without every caller having
//!    to remember which `inputs!` form and which output index.
//! 3. Turn ORT errors into [`vaani_core::VaaniError`] with a user-facing
//!    message, because `OrtError`'s `Display` is a status string that no user
//!    should ever read.
//!
//! It is deliberately *not* an abstraction layer. There is one inference
//! runtime, one tensor layout, and two hot call sites. Wrapping it further
//! would add indirection to the one code path where instruction count and
//! allocation behaviour actually matter. See ADR-003.

use std::path::Path;

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{Tensor, Value};
use vaani_core::{Result, err};

use crate::capability::MachineCapabilities;
use crate::runtime::{
    ActiveProvider, Provider, ProviderStatus, choose_provider, ensure_initialised, location,
};

/// A named, owned f32 tensor with an arbitrary rank.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)] // field names are the documentation
pub struct F32Tensor {
    pub shape: Vec<i64>,
    pub data: Vec<f32>,
}

impl F32Tensor {
    /// Build a tensor, checking that the shape matches the data.
    pub fn new(shape: impl Into<Vec<i64>>, data: Vec<f32>) -> Result<Self> {
        let shape = shape.into();
        let expected: i64 = shape.iter().product();
        if expected as usize != data.len() {
            return Err(err::internal().with_cause(format!(
                "tensor shape {shape:?} needs {expected} elements, got {}",
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }

    /// Reinterpret as a 2-D tensor of the given shape.
    pub fn dims2(&self, rows: usize, cols: usize) -> Result<Self> {
        Self::new(vec![rows as i64, cols as i64], self.data.clone())
    }

    /// The underlying samples.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

/// A named, owned f16 tensor with an arbitrary rank.
///
/// SraVaani's encoder is exported in native fp16, so the graph boundary is
/// fp16 and we have to speak it. The *front end* still computes log-mel in f32,
/// because the per-feature normalisation divides by a standard deviation and
/// losing mantissa bits there is a real accuracy cost; the single conversion to
/// f16 happens here, once, on its way to the device.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)] // field names are the documentation
pub struct F16Tensor {
    pub shape: Vec<i64>,
    pub data: Vec<half::f16>,
}

impl F16Tensor {
    /// Convert an f32 tensor, checking the shape first.
    ///
    /// This is the one place the f32 -> f16 boundary lives. Keeping it here
    /// means the media and ASR crates never have to know the encoder's
    /// precision, and a future fp8 or int8 encoder only changes this file.
    pub fn from_f32(t: &F32Tensor) -> Self {
        Self {
            shape: t.shape.clone(),
            data: t.data.iter().copied().map(half::f16::from_f32).collect(),
        }
    }

    /// Build directly from f16 samples.
    pub fn new(shape: impl Into<Vec<i64>>, data: Vec<half::f16>) -> Result<Self> {
        let shape = shape.into();
        let expected: i64 = shape.iter().product();
        if expected as usize != data.len() {
            return Err(err::internal().with_cause(format!(
                "f16 tensor shape {shape:?} needs {expected} elements, got {}",
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }

    /// Widen back to f32, for the decode loop and for comparison in tests.
    pub fn to_f32(&self) -> F32Tensor {
        F32Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|v| v.to_f32()).collect(),
        }
    }
}

/// A named i32 tensor.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)] // field names are the documentation
pub struct I32Tensor {
    pub shape: Vec<i64>,
    pub data: Vec<i32>,
}

impl I32Tensor {
    /// Build a tensor, checking that the shape matches the data.
    pub fn new(shape: impl Into<Vec<i64>>, data: Vec<i32>) -> Result<Self> {
        let shape = shape.into();
        let expected: i64 = shape.iter().product();
        if expected as usize != data.len() {
            return Err(err::internal().with_cause(format!(
                "i32 tensor shape {shape:?} needs {expected} elements, got {}",
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }

    /// A one-element tensor.
    pub fn scalar(v: i32) -> Self {
        Self {
            shape: vec![1],
            data: vec![v],
        }
    }
}

/// A named i64 tensor. ONNX Runtime's Conformer graphs take `feature_lengths`
/// as int64, which differs from the decoder's int32 targets; getting this
/// backwards is a silent-wrong-answer bug, so the two are distinct types.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)] // field names are the documentation
pub struct I64Tensor {
    pub shape: Vec<i64>,
    pub data: Vec<i64>,
}

impl I64Tensor {
    /// Build a tensor, checking that the shape matches the data.
    pub fn new(shape: impl Into<Vec<i64>>, data: Vec<i64>) -> Result<Self> {
        let shape = shape.into();
        let expected: i64 = shape.iter().product();
        if expected as usize != data.len() {
            return Err(err::internal().with_cause(format!(
                "i64 tensor shape {shape:?} needs {expected} elements, got {}",
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }

    /// A one-element tensor.
    pub fn scalar(v: i64) -> Self {
        Self {
            shape: vec![1],
            data: vec![v],
        }
    }
}

/// One named input to a run.
#[derive(Debug, Clone)]
#[allow(missing_docs)] // variant names are the documentation
pub enum Input {
    F32 { name: String, t: F32Tensor },
    F16 { name: String, t: F16Tensor },
    I32 { name: String, t: I32Tensor },
    I64 { name: String, t: I64Tensor },
}

impl Input {
    /// A named f32 input.
    pub fn f32(name: &str, t: F32Tensor) -> Self {
        Self::F32 {
            name: name.to_string(),
            t,
        }
    }
    /// A named f16 input.
    pub fn f16(name: &str, t: F16Tensor) -> Self {
        Self::F16 {
            name: name.to_string(),
            t,
        }
    }
    /// A named i32 input.
    pub fn i32(name: &str, t: I32Tensor) -> Self {
        Self::I32 {
            name: name.to_string(),
            t,
        }
    }
    /// A named i64 input.
    pub fn i64(name: &str, t: I64Tensor) -> Self {
        Self::I64 {
            name: name.to_string(),
            t,
        }
    }
}

/// A loaded ONNX model, ready to run.
pub struct OnnxSession {
    session: Session,
    status: ProviderStatus,
    /// Input names in graph order, for validation and diagnostics.
    input_names: Vec<String>,
    output_names: Vec<String>,
    threads: usize,
}

impl std::fmt::Debug for OnnxSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxSession")
            .field("provider", &self.status.active)
            .field("inputs", &self.input_names)
            .field("outputs", &self.output_names)
            .field("threads", &self.threads)
            .finish_non_exhaustive()
    }
}

impl OnnxSession {
    /// Load `model_path`, choosing an execution provider.
    ///
    /// `max_vram_bytes` caps the CUDA allocator arena. On a 4 GB laptop GPU an
    /// unbounded BFC arena will happily reserve memory the desktop compositor
    /// needs, and the failure then surfaces as an out-of-memory error in an
    /// unrelated place. Capping it makes the failure happen here, where we can
    /// explain it.
    /// Load a model, choosing an execution provider.
    pub fn load(
        model_path: &Path,
        provider: Provider,
        caps: &MachineCapabilities,
        max_vram_bytes: Option<u64>,
    ) -> Result<Self> {
        Self::load_with_options(model_path, provider, caps, max_vram_bytes, Vec::new())
    }

    /// The CUDA provider options VaaniDesk applies, and why each one is set.
    ///
    /// These are the options `ort` 2.0.0-rc.13 exposes on its typed `ep::CUDA`
    /// builder. They are applied in `load_with_options` and their rationale
    /// lives here, next to the code that uses them.
    ///
    /// * `gpu_mem_limit` - the inference adapter's free VRAM minus a reserve.
    ///   An unbounded BFC arena on a 4 GB laptop GPU reserves memory the desktop
    ///   compositor needs, and the failure then surfaces somewhere unrelated.
    /// * `arena_extend_strategy = kSameAsRequested` - stops the arena rounding
    ///   up to the next power of two, which on a 4 GB card is the difference
    ///   between fitting and not.
    /// * `cudnn_conv_use_max_workspace = 0` - cuDNN's max-workspace search can
    ///   allocate several GB. SraVaani's convolutions are depthwise-separable
    ///   with small channel counts, so tensor-core conv algorithms would not be
    ///   chosen anyway, and the workspace is pure waste.
    /// * `cudnn_conv_algo_search = HEURISTIC` - the default EXHAUSTIVE search
    ///   benchmarks algorithms on first use, which the user sees as a
    ///   multi-second stall on their first job.
    ///
    /// **A note on what is *not* here.** `ort` 2.0.0-rc.13 cannot set arbitrary
    /// CUDA provider options: `CUDA::options` is private and
    /// `ExecutionProviderDispatch::new` is crate-private, so there is no
    /// key/value registration path. That matters because `do_conv_in_fp16` is
    /// the documented lever for "run convolutions in fp32", and it looks like
    /// the fix for a cuDNN graph-build failure we hit while bringing this up.
    /// It was not: the real cause was the DLL search path (see
    /// `runtime::ensure_initialised`), and with that fixed both the fp16 and
    /// fp32 encoders run on the GPU. `examples/conv_probe.rs` measures both and
    /// is the regression test for it.
    pub fn cuda_provider_options() -> Vec<(&'static str, &'static str)> {
        vec![
            ("arena_extend_strategy", "kSameAsRequested"),
            ("cudnn_conv_use_max_workspace", "0"),
            ("cudnn_conv_algo_search", "HEURISTIC"),
        ]
    }

    /// Load with a recorded set of CUDA provider options.
    ///
    /// The argument is **not applied**: `ort` 2.0.0-rc.13 gives no way to set
    /// arbitrary CUDA provider options (see [`Self::cuda_provider_options`]),
    /// so there is nothing to override. It is kept as a parameter and, if a
    /// caller supplies anything, we *refuse* rather than quietly ignore it --
    /// a silently-ignored option in a benchmark is how you end up publishing
    /// numbers for a configuration nobody ran.
    ///
    /// Product code calls [`Self::load`].
    pub fn load_with_options(
        model_path: &Path,
        provider: Provider,
        caps: &MachineCapabilities,
        max_vram_bytes: Option<u64>,
        cuda_option_overrides: Vec<(String, String)>,
    ) -> Result<Self> {
        ensure_initialised()?;

        if !model_path.is_file() {
            return Err(err::model_unavailable()
                .with_cause(format!("model file not found: {}", model_path.display())));
        }

        let cuda_loadable = location()
            .provider_dirs
            .iter()
            .any(|d| d.join(crate::runtime::ORT_CUDA_DLL).is_file());
        let status = choose_provider(provider, cuda_loadable);

        let threads = caps.recommended_ort_threads();
        let mut builder = Session::builder()
            .map_err(ort_err)?
            // Level 3 is ORT's default and is what fuses the MatMul/Conv pairs
            // the Conformer is full of. There is no reason to lower it.
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_err)?
            .with_intra_threads(threads)
            .map_err(ort_err)?;

        if status.active == ActiveProvider::Cuda {
            let mut ep = ort::ep::CUDA::default()
                .with_device_id(0)
                // SameAsRequested keeps the arena from rounding up to the next
                // power of two, which on a 4 GB card is the difference between
                // fitting and not.
                .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested)
                // cuDNN's max-workspace search can allocate *several GB*. The
                // Conformer conv modules are depthwise-separable with small
                // channel counts, so tensor-core conv algorithms would not be
                // chosen anyway and the workspace is pure waste.
                .with_conv_max_workspace(false)
                // The default EXHAUSTIVE search benchmarks algorithms on first
                // use, which shows up as a multi-second stall on the first job.
                .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic);
            if let Some(limit) = max_vram_bytes {
                ep = ep.with_memory_limit(limit as usize);
            }
            if !cuda_option_overrides.is_empty() {
                // Better a loud failure than a benchmark that measured
                // something other than what it says it measured.
                let keys: Vec<&str> = cuda_option_overrides
                    .iter()
                    .map(|(k, _)| k.as_str())
                    .collect();
                return Err(err::internal().with_cause(format!(
                    "ort 2.0.0-rc.13 cannot apply CUDA provider options {keys:?}; \
                     use the typed ep::CUDA builder instead"
                )));
            }
            // `fail_silently` is the whole fallback story: if the provider
            // refuses to register, every node runs on the built-in CPU EP and
            // the session still produces correct output.
            builder = builder
                .with_execution_providers([ep.build().fail_silently()])
                .map_err(ort_err)?;
        } else {
            builder = builder
                .with_execution_providers([ort::ep::CPU::default().build()])
                .map_err(ort_err)?;
        }

        let session = builder.commit_from_file(model_path).map_err(|e| {
            // A model that will not load is almost always a version mismatch
            // between the exporter and the runtime, or a missing external-data
            // sidecar. Both are worth saying out loud in the log, and neither is
            // the user's fault, so the message stays neutral.
            let detail = format!("ORT could not load {}: {e}", model_path.display());
            tracing::error!(model = %model_path.display(), "{detail}");
            err::model_load_failed().with_cause(detail)
        })?;

        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();

        tracing::info!(
            model = %model_path.display(),
            provider = status.active.as_str(),
            threads,
            inputs = ?input_names,
            outputs = ?output_names,
            "session loaded"
        );

        Ok(Self {
            session,
            status,
            input_names,
            output_names,
            threads,
        })
    }

    /// Which provider actually ended up in use.
    pub fn provider_status(&self) -> &ProviderStatus {
        &self.status
    }

    /// Graph input names, in graph order.
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Graph output names, in graph order.
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// The intra-op thread count this session was built with.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Run with inputs given by name.
    ///
    /// Outputs are returned in the order requested, not graph order, because
    /// the ASR engine asks for exactly the tensors it wants and index arithmetic
    /// into `SessionOutputs` is the classic source of an off-by-one that
    /// produces plausible-looking garbage.
    /// Run the session.
    pub fn run(&mut self, inputs: &[Input], outputs: &[&str]) -> Result<Vec<OutputTensor>> {
        // `ort`'s `inputs!` macro needs a homogeneous argument list and we have
        // three tensor types, so we build the name/value map explicitly. Names
        // are checked against the graph *before* anything is allocated, so a
        // typo costs nothing and reports the real input list.
        let mut map: Vec<(&str, Value)> = Vec::with_capacity(inputs.len());
        for i in inputs {
            let value = match i {
                Input::F32 { name, t } => {
                    ensure_input(&self.input_names, name)?;
                    // Safe: F32Tensor validated data.len() == shape product at
                    // construction, and the Vec is moved in so ORT owns it
                    // exclusively.
                    let v = Tensor::from_array((t.shape.clone(), t.data.clone()))
                        .map_err(|e| err::internal().with_cause(format!("input {name}: {e}")))?;
                    Value::from(v)
                }
                Input::F16 { name, t } => {
                    ensure_input(&self.input_names, name)?;
                    let v = Tensor::from_array((t.shape.clone(), t.data.clone()))
                        .map_err(|e| err::internal().with_cause(format!("input {name}: {e}")))?;
                    Value::from(v)
                }
                Input::I32 { name, t } => {
                    ensure_input(&self.input_names, name)?;
                    let v = Tensor::from_array((t.shape.clone(), t.data.clone()))
                        .map_err(|e| err::internal().with_cause(format!("input {name}: {e}")))?;
                    Value::from(v)
                }
                Input::I64 { name, t } => {
                    ensure_input(&self.input_names, name)?;
                    let v = Tensor::from_array((t.shape.clone(), t.data.clone()))
                        .map_err(|e| err::internal().with_cause(format!("input {name}: {e}")))?;
                    Value::from(v)
                }
            };
            let name = match i {
                Input::F32 { name, .. }
                | Input::F16 { name, .. }
                | Input::I32 { name, .. }
                | Input::I64 { name, .. } => name.as_str(),
            };
            map.push((name, value));
        }
        for o in outputs {
            ensure_output(&self.output_names, o)?;
        }

        let result = self.session.run(map).map_err(|e| {
            tracing::error!("inference failed: {e}");
            err::transcription_failed().with_cause(format!("ORT run failed: {e}"))
        })?;

        let mut out = Vec::with_capacity(outputs.len());
        for name in outputs {
            // Borrow, don't move: SessionOutputs is an arena we still need
            // for the remaining requested outputs.
            let v = &result[*name];
            out.push(tensor_to_output(v, name)?);
        }
        Ok(out)
    }
}

fn ensure_input(names: &[String], name: &str) -> Result<()> {
    if names.is_empty() || names.iter().any(|n| n == name) {
        Ok(())
    } else {
        Err(err::internal().with_cause(format!(
            "model has no input named {name:?}; it has {names:?}"
        )))
    }
}

fn ensure_output(names: &[String], name: &str) -> Result<()> {
    if names.is_empty() || names.iter().any(|n| n == name) {
        Ok(())
    } else {
        Err(err::internal().with_cause(format!(
            "model has no output named {name:?}; it has {names:?}"
        )))
    }
}

/// Convert an ONNX Runtime error into ours.
///
/// `ort::Error<E>` is generic in a "rollback" type: builder methods yield
/// `Error<SessionBuilder>` because a failure hands the builder back, while
/// `run`/`commit` yield `Error<()>`. Rather than write two nearly identical
/// converters we take the type as a parameter, since neither variant affects
/// the message.
fn ort_err<E>(e: ort::Error<E>) -> vaani_core::VaaniError
where
    ort::Error<E>: std::fmt::Display,
{
    tracing::error!("ONNX Runtime error: {e}");
    err::internal().with_cause(format!("ORT: {e}"))
}

/// A tensor returned from a run.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)] // variant names are the documentation
pub enum OutputTensor {
    F32(F32Tensor),
    F16(F16Tensor),
    I32(I32Tensor),
    I64(I64Tensor),
}

impl OutputTensor {
    /// Borrow as f32, or fail if the dtype differs.
    pub fn as_f32(&self) -> Result<&F32Tensor> {
        match self {
            Self::F32(t) => Ok(t),
            other => Err(err::internal().with_cause(format!("expected f32 output, got {other:?}"))),
        }
    }

    /// Borrow as f16, or fail if the dtype differs.
    pub fn as_f16(&self) -> Result<&F16Tensor> {
        match self {
            Self::F16(t) => Ok(t),
            other => Err(err::internal().with_cause(format!("expected f16 output, got {other:?}"))),
        }
    }

    /// Borrow as i32, or fail if the dtype differs.
    pub fn as_i32(&self) -> Result<&I32Tensor> {
        match self {
            Self::I32(t) => Ok(t),
            other => Err(err::internal().with_cause(format!("expected i32 output, got {other:?}"))),
        }
    }

    /// Borrow as i64, or fail if the dtype differs.
    pub fn as_i64(&self) -> Result<&I64Tensor> {
        match self {
            Self::I64(t) => Ok(t),
            other => Err(err::internal().with_cause(format!("expected i64 output, got {other:?}"))),
        }
    }
}

fn tensor_to_output(v: &Value, name: &str) -> Result<OutputTensor> {
    // `extract_tensor` gives us the shape and a borrowed slice without needing
    // the `ndarray` feature, and errors rather than silently reinterpreting if
    // the dtype is not what we asked for. The order matters: f16 must be tried
    // before the wider types, and every attempt is explicit, because a wrong
    // guess would reinterpret 2-byte samples as 4-byte floats and produce
    // plausible-looking garbage rather than an error.
    if let Ok((shape, data)) = v.try_extract_tensor::<half::f16>() {
        return Ok(OutputTensor::F16(F16Tensor {
            shape: shape.iter().copied().collect(),
            data: data.to_vec(),
        }));
    }
    let dyno = v
        .try_extract_tensor::<f32>()
        .map_err(|_| err::internal().with_cause(format!("output {name} is not f32")));
    if let Ok((shape, data)) = dyno {
        return Ok(OutputTensor::F32(F32Tensor {
            shape: shape.iter().copied().collect(),
            data: data.to_vec(),
        }));
    }
    if let Ok((shape, data)) = v.try_extract_tensor::<i32>() {
        return Ok(OutputTensor::I32(I32Tensor {
            shape: shape.iter().copied().collect(),
            data: data.to_vec(),
        }));
    }
    if let Ok((shape, data)) = v.try_extract_tensor::<i64>() {
        return Ok(OutputTensor::I64(I64Tensor {
            shape: shape.iter().copied().collect(),
            data: data.to_vec(),
        }));
    }
    Err(err::internal().with_cause(format!("output {name} had an unexpected dtype")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_shape_must_match_its_data() {
        assert!(F32Tensor::new(vec![2, 3], vec![0.0; 6]).is_ok());
        // A shape that does not match is a bug in the caller; catching it here
        // turns a confusing ORT error into a clear one.
        let e = F32Tensor::new(vec![2, 3], vec![0.0; 5]).unwrap_err();
        assert!(e.debug_chain().contains("needs 6 elements"));
    }

    #[test]
    fn f16_boundary_conversion_is_lossless_enough_for_features() {
        // The log-mel values the encoder sees are roughly in [-8, 8] with a
        // per-feature standard deviation of 1. f16 has 11 bits of mantissa, so
        // relative error stays around 5e-4 -- far below the fp16 encoder's own
        // noise floor, and confirmed here rather than assumed.
        let t = F32Tensor::new(vec![1, 4], vec![-8.0, -1.0, 0.0, 7.5]).unwrap();
        let h = F16Tensor::from_f32(&t);
        let back = h.to_f32();
        for (a, b) in t.as_slice().iter().zip(back.as_slice()) {
            assert!((a - b).abs() <= a.abs() * 5e-4 + 1e-6, "{a} -> {b}");
        }
    }

    #[test]
    fn f16_tensor_validates_its_shape() {
        assert!(F16Tensor::new(vec![2, 2], vec![half::f16::ZERO; 4]).is_ok());
        assert!(F16Tensor::new(vec![2, 2], vec![half::f16::ZERO; 3]).is_err());
    }

    #[test]
    fn i64_and_i32_are_distinct_types() {
        // The Conformer encoder wants int64 lengths and the TDT joint wants
        // int32 targets. Keeping them as separate types is what prevents the
        // silent wrong-answer bug.
        let a: I64Tensor = I64Tensor::scalar(1);
        let b: I32Tensor = I32Tensor::scalar(1);
        assert_eq!(a.data[0], i64::from(b.data[0]));
        assert_ne!(std::mem::size_of::<I64Tensor>(), 0);
    }

    #[test]
    fn named_input_lookup_rejects_unknown_names() {
        let names = vec!["feats".to_string(), "feature_lengths".to_string()];
        assert!(ensure_input(&names, "feats").is_ok());
        let e = ensure_input(&names, "audio").unwrap_err();
        assert!(e.debug_chain().contains("no input named"));
    }

    #[test]
    fn output_type_mismatch_is_an_error_not_a_reinterpret() {
        let t = OutputTensor::I32(I32Tensor::scalar(1));
        assert!(t.as_f32().is_err());
        let t = OutputTensor::F32(F32Tensor::new(vec![1], vec![1.0]).unwrap());
        assert!(t.as_f32().is_ok());
        assert!(t.as_i32().is_err());
    }
}

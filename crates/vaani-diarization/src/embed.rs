//! The neural speaker-embedding path: an `ort` adapter for when the weights
//! exist.
//!
//! # This code is unexercised
//!
//! There are **no diarization model weights in this build**. `artifacts/`
//! contains SraVaani's ASR graphs and nothing else, so this module is written,
//! compiles, and has never run a real forward pass. It is documented here as
//! an interface contract, not as a working feature, and nothing in this crate
//! reports a number derived from it.
//!
//! Keeping it is worth the dead code because:
//!
//! * the interface is the part that is expensive to get wrong later — input
//!   rank, sample rate, window length, whether the output is L2-normalised,
//!   and what to do with an all-zero input are five separate ways to get
//!   plausible, wrong embeddings;
//! * a wrong embedding is *worse* than no embedding, because it produces
//!   plausible clusters that are confidently wrong, which is the failure mode
//!   this product cares about most.
//!
//! # The interface it expects
//!
//! An ECAPA-TDNN-family x-vector model, exported to ONNX with this contract:
//!
//! | property | required value |
//! |---|---|
//! | input name | `input` (configurable) |
//! | input dtype | `float32` |
//! | input shape | `[1, T]` or `[1, 1, T]`, `T = window_samples` |
//! | sample rate | 16 000 Hz, mono, **no resampling inside the graph** |
//! | window | `window_samples`, default 200 frames of 16 ms = 3.2 s |
//! | output name | `embedding` (configurable) |
//! | output dtype | `float32` |
//! | output shape | `[1, D]`, default `D = 192` |
//! | normalisation | if `normalize` is set, this module L2-normalises before returning |
//!
//! ECAPA-TDNN in its reference implementation expects 80-band log-mel with a
//! 25 ms window and 10 ms hop. This module feeds it **raw audio** and
//! documents that, because a model exported for a different front end is a
//! much more likely mistake than the one it is trying to catch, and it is
//! caught loudly at load time by the shape check rather than silently at
//! runtime.
//!
//! Anything that does not match — a missing file, a different input name, a
//! different window length, a rank the spec does not allow — makes
//! [`NeuralEmbedder::open`] fail with a `Caused by` diagnostic. It never
//! falls back to a partially-configured session.

use std::path::{Path, PathBuf};

use vaani_core::{Result, err};
use vaani_inference::{F32Tensor, Input, MachineCapabilities, OnnxSession, OutputTensor, Provider};

use crate::config::REQUIRED_SAMPLE_RATE;

/// The conventional file name, inside the configured artifacts directory.
pub const EMBEDDER_FILE: &str = "embedder.onnx";

/// What a speaker embedder is expected to look like.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbedderSpec {
    /// Graph input name.
    pub input_name: String,
    /// Graph output name.
    pub output_name: String,
    /// Input tensor rank: 2 for `[1, T]`, 3 for `[1, 1, T]`.
    pub input_rank: usize,
    /// Samples fed to the graph.
    pub window_samples: usize,
    /// Samples between successive windows.
    pub hop_samples: usize,
    /// Embedding width, checked against the graph's output.
    pub embedding_dim: usize,
    /// Whether the graph's output still needs L2 normalisation.
    pub normalize: bool,
}

impl Default for EmbedderSpec {
    /// The ECAPA-TDNN defaults: 16 kHz, 3.2 s windows at 50% overlap, a
    /// 192-dimensional embedding, and a graph that emits unnormalised output.
    fn default() -> Self {
        Self {
            input_name: "input".into(),
            output_name: "embedding".into(),
            input_rank: 3,
            window_samples: 200 * 16 * REQUIRED_SAMPLE_RATE as usize / 1_000,
            hop_samples: 100 * 16 * REQUIRED_SAMPLE_RATE as usize / 1_000,
            embedding_dim: 192,
            normalize: true,
        }
    }
}

impl EmbedderSpec {
    /// The shape the input tensor must have, for one window.
    #[must_use]
    pub fn input_shape(&self) -> Vec<i64> {
        match self.input_rank {
            2 => vec![1, self.window_samples as i64],
            3 => vec![1, 1, self.window_samples as i64],
            _ => Vec::new(),
        }
    }

    /// The sample rate the spec requires, which is the pipeline's rate.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        REQUIRED_SAMPLE_RATE
    }

    /// Whether `spec` is internally coherent.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.input_rank != 2 && self.input_rank != 3 {
            return Err(format!(
                "input_rank must be 2 or 3, got {}",
                self.input_rank
            ));
        }
        if self.window_samples == 0 {
            return Err("window_samples is zero".into());
        }
        if self.hop_samples == 0 || self.hop_samples > self.window_samples {
            return Err(format!(
                "hop_samples {} must be in 1..={}",
                self.hop_samples, self.window_samples
            ));
        }
        if self.embedding_dim == 0 {
            return Err("embedding_dim is zero".into());
        }
        if self.input_name.is_empty() || self.output_name.is_empty() {
            return Err("input_name and output_name must be non-empty".into());
        }
        Ok(())
    }
}

/// Where a neural embedder would live, and whether it is there.
#[derive(Debug, Clone, PartialEq)]
pub struct Availability {
    /// The path that was probed.
    pub path: PathBuf,
    /// Whether a file exists there.
    pub file_present: bool,
    /// Why the path is unusable, when it is. `None` when it is usable.
    pub problem: Option<String>,
}

impl Availability {
    /// Whether the embedder can be loaded.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.file_present && self.problem.is_none()
    }
}

/// The default embedder path for a configured artifacts directory.
#[must_use]
pub fn default_path(dir: &Path) -> PathBuf {
    dir.join(EMBEDDER_FILE)
}

/// Probe for a neural embedder without loading it.
///
/// Loading a session allocates accelerator memory and, on a machine with CUDA,
/// creates a context. A settings screen that lists the installed diarization
/// models must be able to answer "is it there?" without paying that, so the
/// probe is filesystem-only.
#[must_use]
pub fn probe(dir: &Path) -> Availability {
    let path = default_path(dir);
    if !path.is_file() {
        return Availability {
            problem: Some(format!("{} does not exist", path.display())),
            path,
            file_present: false,
        };
    }
    Availability {
        path,
        file_present: true,
        problem: None,
    }
}

/// Whether the ONNX Runtime shared library can be loaded from `dir`.
///
/// A genuine availability check that does not create a session: it asks
/// `onnxruntime.dll` to load and immediately gives it back, so no CUDA context
/// and no VRAM is spent.
pub fn runtime_available(dir: &Path) -> std::result::Result<(), String> {
    let path = dir.join("onnxruntime.dll");
    if !path.is_file() {
        return Err(format!("{} does not exist", path.display()));
    }
    // `init_from` is the only way to ask ORT whether it can load, and its
    // `commit` is the point at which the DLL is really opened. Nothing beyond
    // that happens here: no session, no CUDA context, no VRAM.
    let builder = ort::init_from(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    if !builder.commit() {
        return Err(format!("{} did not initialise", path.display()));
    }
    Ok(())
}

/// A loaded neural speaker embedder.
///
/// `load` opens a session through `vaani-inference`, so it uses the same
/// execution-provider choice, the same thread count and the same VRAM cap as
/// ASR. A diarization stage that quietly created its own CUDA context would
/// break the "no CUDA context at startup" rule `vaani-inference` documents.
pub struct NeuralEmbedder {
    session: OnnxSession,
    spec: EmbedderSpec,
    path: PathBuf,
}

impl std::fmt::Debug for NeuralEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeuralEmbedder")
            .field("path", &self.path)
            .field("spec", &self.spec)
            .field("session", &self.session)
            .finish()
    }
}

impl NeuralEmbedder {
    /// Open the embedder at `path`.
    ///
    /// The graph's own input and output names are checked against the spec
    /// **before** any run, so a model with the wrong signature fails here with
    /// a diagnostic rather than producing embeddings that are quietly
    /// mis-shaped.
    pub fn open(
        path: &Path,
        spec: EmbedderSpec,
        provider: Provider,
        caps: &MachineCapabilities,
        max_vram_bytes: Option<u64>,
    ) -> Result<Self> {
        spec.validate()
            .map_err(|e| err::model_corrupt().with_cause(format!("embedder spec: {e}")))?;
        if !path.is_file() {
            return Err(
                err::model_unavailable().with_cause(format!("no embedder at {}", path.display()))
            );
        }
        let session = OnnxSession::load(path, provider, caps, max_vram_bytes)?;
        if !session.input_names().iter().any(|n| n == &spec.input_name) {
            return Err(err::model_corrupt().with_cause(format!(
                "{} has inputs {:?}, not {:?}",
                path.display(),
                session.input_names(),
                spec.input_name
            )));
        }
        if !session.output_names().is_empty()
            && !session
                .output_names()
                .iter()
                .any(|n| n == &spec.output_name)
        {
            return Err(err::model_corrupt().with_cause(format!(
                "{} has outputs {:?}, not {:?}",
                path.display(),
                session.output_names(),
                spec.output_name
            )));
        }
        Ok(Self {
            session,
            spec,
            path: path.to_path_buf(),
        })
    }

    /// The spec this embedder was opened with.
    #[must_use]
    pub fn spec(&self) -> &EmbedderSpec {
        &self.spec
    }

    /// The model file path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Embed one window of audio.
    ///
    /// `window` must be exactly `spec.window_samples`; a shorter or longer
    /// slice is an error, because a graph with a fixed time dimension will
    /// either reject it or, worse, accept a broadcast shape and return
    /// something meaningless.
    ///
    /// An all-silent window returns `None` rather than a zero vector. A zero
    /// embedding has a cosine similarity of 0 with everything including
    /// itself, so every silent window would be maximally distant from every
    /// other window and the clustering would put them all in their own
    /// clusters. Returning `None` lets the caller leave the window
    /// unattributed, which is the truthful answer.
    pub fn embed(&mut self, window: &[f32]) -> Result<Option<Vec<f32>>> {
        if window.len() != self.spec.window_samples {
            return Err(err::internal().with_cause(format!(
                "embedder wants {} samples, got {}",
                self.spec.window_samples,
                window.len()
            )));
        }
        let energy: f32 = window.iter().map(|v| v * v).sum();
        if !(energy.is_finite() && energy > 1e-9) {
            return Ok(None);
        }
        let shape = self.spec.input_shape();
        let tensor = F32Tensor::new(shape, window.to_vec())?;
        let name = self.spec.input_name.clone();
        let out_name = self.spec.output_name.clone();
        let outputs = self
            .session
            .run(&[Input::f32(&name, tensor)], &[&out_name])?;
        let first = outputs.first().ok_or_else(|| {
            err::diarization_failed().with_cause("the embedder returned no output")
        })?;
        let t: &F32Tensor = match first {
            OutputTensor::F32(t) => t,
            other => {
                return Err(err::model_corrupt()
                    .with_cause(format!("embedder output is {other:?}, expected f32")));
            }
        };
        if t.data.len() != self.spec.embedding_dim {
            return Err(err::model_corrupt().with_cause(format!(
                "embedder returned {} values, spec says {}",
                t.data.len(),
                self.spec.embedding_dim
            )));
        }
        let mut v = t.data.clone();
        if self.spec.normalize {
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n > 1e-9 {
                v.iter_mut().for_each(|x| *x /= n);
            }
        }
        if v.iter().all(|x| !x.is_finite()) {
            return Err(
                err::diarization_failed().with_cause("the embedder produced a non-finite vector")
            );
        }
        Ok(Some(v))
    }

    /// Close the session, releasing any accelerator memory.
    pub fn close(self) {
        drop(self.session);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_spec_is_the_ecapa_geometry() {
        let s = EmbedderSpec::default();
        assert_eq!(s.sample_rate(), 16_000);
        assert_eq!(s.window_samples, 51_200, "200 frames of 16 ms");
        assert_eq!(s.hop_samples, 25_600, "50% overlap");
        assert_eq!(s.embedding_dim, 192);
        assert_eq!(s.input_shape(), vec![1, 1, 51_200]);
    }

    #[test]
    fn a_rank_two_spec_produces_a_two_dimensional_shape() {
        let s = EmbedderSpec {
            input_rank: 2,
            ..EmbedderSpec::default()
        };
        assert_eq!(s.input_shape(), vec![1, 51_200]);
    }

    #[test]
    fn spec_validation_catches_every_degenerate_geometry() {
        let ok = EmbedderSpec::default();
        assert!(ok.validate().is_ok());

        let bad = EmbedderSpec {
            input_rank: 4,
            ..EmbedderSpec::default()
        };
        assert!(bad.validate().unwrap_err().contains("input_rank"));

        let bad = EmbedderSpec {
            window_samples: 0,
            ..EmbedderSpec::default()
        };
        assert!(
            bad.validate()
                .unwrap_err()
                .contains("window_samples is zero")
        );

        let bad = EmbedderSpec {
            hop_samples: 0,
            ..EmbedderSpec::default()
        };
        assert!(bad.validate().unwrap_err().contains("hop_samples"));

        let bad = EmbedderSpec {
            hop_samples: 60_000,
            ..EmbedderSpec::default()
        };
        assert!(bad.validate().unwrap_err().contains("hop_samples"));

        let bad = EmbedderSpec {
            embedding_dim: 0,
            ..EmbedderSpec::default()
        };
        assert!(bad.validate().unwrap_err().contains("embedding_dim"));

        let bad = EmbedderSpec {
            input_name: String::new(),
            ..EmbedderSpec::default()
        };
        assert!(bad.validate().unwrap_err().contains("non-empty"));
    }

    #[test]
    fn probing_a_directory_with_no_weights_reports_the_path_it_looked_at() {
        let a = probe(Path::new("/definitely/not/here"));
        assert!(!a.is_available());
        assert!(!a.file_present);
        assert!(a.problem.unwrap().contains("embedder.onnx"));
        assert_eq!(a.path, PathBuf::from("/definitely/not/here/embedder.onnx"));
    }

    #[test]
    fn probing_a_directory_with_a_file_looks_available() {
        // The probe is filesystem-only by design, so this is as far as it
        // goes without loading a session.
        let dir = std::env::temp_dir().join("vaani-a4-probe-test");
        std::fs::create_dir_all(&dir).unwrap();
        let file = default_path(&dir);
        std::fs::write(&file, b"not really an onnx file").unwrap();
        let a = probe(&dir);
        assert!(a.file_present);
        assert!(
            a.is_available(),
            "the probe is a presence check, not a load"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn opening_a_missing_file_is_model_unavailable_not_a_panic() {
        let e = NeuralEmbedder::open(
            Path::new("/definitely/not/here/embedder.onnx"),
            EmbedderSpec::default(),
            Provider::Cpu,
            &vaani_inference::capability::detect(&[]),
            None,
        )
        .unwrap_err();
        assert_eq!(e.code(), vaani_core::ErrorCode::ModelUnavailable);
    }

    #[test]
    fn an_invalid_spec_is_rejected_before_the_filesystem_is_touched() {
        let e = NeuralEmbedder::open(
            Path::new("/definitely/not/here/embedder.onnx"),
            EmbedderSpec {
                input_rank: 9,
                ..EmbedderSpec::default()
            },
            Provider::Cpu,
            &vaani_inference::capability::detect(&[]),
            None,
        )
        .unwrap_err();
        assert_eq!(e.code(), vaani_core::ErrorCode::ModelCorrupt);
        assert!(e.debug_chain().contains("input_rank"));
    }

    #[test]
    fn a_missing_runtime_is_reported_not_swallowed() {
        let e = runtime_available(Path::new("/definitely/not/here")).unwrap_err();
        assert!(e.contains("onnxruntime.dll"), "{e}");
    }
}

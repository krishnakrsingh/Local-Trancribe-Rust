//! The `AsrEngine` abstraction and the engine registry.
//!
//! # Why a trait at all
//!
//! The product must be able to run more than one recogniser, and it must be
//! able to *compare* them. §35 of the requirements asks for a benchmark that
//! runs every installed engine on the same recording, and §36 for
//! `vaani benchmark --model <id>`. Both are impossible without an interface.
//!
//! The trait has exactly the methods the spec names, and no more:
//!
//! ```text
//! model_name, model_version, languages, supports_streaming,
//! supports_timestamps, supports_word_timestamps,
//! load, unload, transcribe, cancel, benchmark
//! ```
//!
//! That restraint matters. §45 of the requirements forbids premature
//! abstraction, and the failure mode of an ASR trait is particularly seductive:
//! every new Whisper-family model tempts you to add `supports_offsets`,
//! `supports_lm`, `supports_itn`, `supports_speaker_embedded`. Each one is a
//! `bool` on the trait and a `match` in every adapter, paid for by every engine
//! forever, to serve a model that does not exist yet.
//!
//! A capability that only one engine has belongs on that engine, reached
//! through a downcast or a dedicated registry entry, not on the shared
//! interface.
//!
//! # What `transcribe` is given and returns
//!
//! The engine is given a **chunk** of audio and returns segments in
//! **chunk-local** time. It is never given a file path and it never learns the
//! chunk's offset in the source media. Rebasing happens in exactly one place,
//! `vaani_core::Transcript::push_chunk`, which is what makes chunking an
//! implementation detail rather than something every engine has to get right.

use std::path::PathBuf;
use std::sync::Arc;

use vaani_core::{CancelToken, Language, Result, Segment, Time};
use vaani_inference::{MachineCapabilities, Provider};

/// A chunk of audio to recognise, in chunk-local coordinates.
#[derive(Debug, Clone)]
#[allow(missing_docs)] // field names are the documentation
pub struct AudioChunk {
    /// Mono, at the engine's required sample rate, nominally in [-1, 1].
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// Where this chunk began in the source media. Diagnostic only: engines
    /// must return chunk-local times and let the caller rebase.
    pub source_offset: Time,
    /// Language hint from the user. `Auto` means "you decide".
    pub language: Language,
}

impl AudioChunk {
    /// A chunk with the default `Auto` language hint.
    pub fn new(samples: Vec<f32>, sample_rate: u32, source_offset: Time) -> Self {
        Self {
            samples,
            sample_rate,
            source_offset,
            language: Language::Auto,
        }
    }

    /// Attach a language hint from the user's picker.
    pub fn with_language(mut self, l: Language) -> Self {
        self.language = l;
        self
    }

    /// Chunk length in milliseconds, from the sample count.
    pub fn duration(&self) -> Time {
        if self.sample_rate == 0 {
            return Time::ZERO;
        }
        Time::from_millis((self.samples.len() as u64 * 1000 / self.sample_rate as u64) as i64)
    }
}

/// What a single engine run cost. Real measurements, never estimates.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct AsrTimings {
    /// Wall time for feature extraction.
    pub frontend_ms: u64,
    /// Wall time inside the encoder session.
    pub encode_ms: u64,
    /// Wall time in the TDT / beam decode loop.
    pub decode_ms: u64,
    /// Number of encoder frames processed.
    pub encoder_frames: usize,
    /// Number of emission steps taken. Roughly 0.3-0.5 per encoder frame.
    pub decode_steps: usize,
}

impl AsrTimings {
    /// Sum of the three measured phases.
    /// Sum of the three measured phases.
    pub fn total_ms(&self) -> u64 {
        self.frontend_ms + self.encode_ms + self.decode_ms
    }
}

/// A recognised chunk, in chunk-local time.
#[derive(Debug, Clone, PartialEq, Default)]
#[allow(missing_docs)] // field names are the documentation
pub struct ChunkResult {
    /// Segments in time order, times relative to the start of the chunk.
    pub segments: Vec<Segment>,
    /// Identified language tag, when the engine reports one.
    pub language: Option<String>,
    pub timings: AsrTimings,
}

/// A speech recognition engine.
///
/// Implementations must be usable from a worker thread. `load` is expected to be
/// slow (hundreds of milliseconds to seconds for a real model) and is therefore
/// never called on the UI thread.
pub trait AsrEngine: Send + Sync + std::fmt::Debug {
    /// Stable id used by the CLI (`--model sravaani`) and the registry.
    fn id(&self) -> &str;

    /// Human name for the picker, e.g. `SraVaani 1.0`.
    fn model_name(&self) -> &str;

    /// Version of the *weights*, independent of the application version. An app
    /// 1.0.0 running SraVaani 2.0 is a normal, supported state.
    fn model_version(&self) -> &str;

    /// Languages the engine can be asked for. Advisory: `Auto` is always
    /// accepted even if this list is empty.
    fn languages(&self) -> &[Language];

    /// Whether the engine can recognise a live stream rather than a complete
    /// file. False for every engine we ship today; the trait carries it because
    /// the product's roadmap names it and a future engine will need it, and
    /// removing a trait method later is a breaking change for every adapter.
    fn supports_streaming(&self) -> bool {
        false
    }

    /// Whether the engine emits segment-level times. True for every engine that
    /// can produce a usable transcript at all.
    fn supports_timestamps(&self) -> bool {
        true
    }

    /// Whether the engine emits word-level times. This one genuinely varies:
    /// CTC models without a forced-alignment stage do not.
    fn supports_word_timestamps(&self) -> bool {
        false
    }

    /// Load the weights. Idempotent: calling it on a loaded engine must not
    /// reload. Must respect `cancel`.
    fn load(&self, cancel: &CancelToken) -> Result<()>;

    /// Release the weights and any accelerator memory. Must be callable when
    /// partially loaded, and must not fail.
    fn unload(&self);

    /// Whether the weights are currently loaded.
    fn is_loaded(&self) -> bool;

    /// Recognise one chunk. Times in the result are chunk-local.
    fn transcribe(&self, chunk: &AudioChunk, cancel: &CancelToken) -> Result<ChunkResult>;

    /// Equivalent to cancelling the token handed to [`AsrEngine::transcribe`].
    ///
    /// Kept as a separate method because the UI has a cancel button and a
    /// `CancelToken` may be owned by a job that has already been dropped.
    fn cancel(&self);

    /// Run a warm-up pass and report timings, without producing a transcript.
    ///
    /// Separate from `transcribe` because the first call on a real model pays
    /// page faults, cuDNN algorithm selection and CUDA module loading, and
    /// folding that into a reported speed number makes it a lie.
    fn benchmark(&self, cancel: &CancelToken) -> Result<AsrTimings>;
}

/// Files an engine needs on disk to load.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[allow(missing_docs)] // field names are the documentation
pub struct EngineArtifacts {
    /// Directory holding `preproc.json`, `mel_window.f32`, `mel_fb.f32`.
    pub frontend_dir: PathBuf,
    pub encoder: PathBuf,
    pub decoder_joint: PathBuf,
    pub tokenizer: PathBuf,
}

impl EngineArtifacts {
    /// Which artefacts are missing, for a user-presentable message.
    pub fn missing(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (label, p) in [
            ("preproc.json", self.frontend_dir.join("preproc.json")),
            ("mel_window.f32", self.frontend_dir.join("mel_window.f32")),
            ("mel_fb.f32", self.frontend_dir.join("mel_fb.f32")),
            ("encoder", self.encoder.clone()),
            ("decoder_joint", self.decoder_joint.clone()),
            ("tokenizer", self.tokenizer.clone()),
        ] {
            if !p.is_file() {
                out.push(format!("{label} ({})", p.display()));
            }
        }
        out
    }

    /// Whether every artefact is present.
    pub fn is_complete(&self) -> bool {
        self.missing().is_empty()
    }
}

/// A constructor for one engine kind, held in the registry.
pub type EngineFactory = Arc<dyn Fn(&EngineArtifacts) -> Result<Box<dyn AsrEngine>> + Send + Sync>;

/// Every engine this build can construct.
#[derive(Clone, Default)]
pub struct EngineRegistry {
    factories: Vec<(String, EngineFactory)>,
}

impl std::fmt::Debug for EngineRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineRegistry")
            .field(
                "engines",
                &self.factories.iter().map(|(id, _)| id).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl EngineRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an engine constructor under `id`.
    ///
    /// Returns nothing rather than `&mut Self`: a registry is built in one
    /// place and by one line per engine, and a chainable builder here invites
    /// `registry = registry.register(..)`, which is easier to get wrong than a
    /// statement.
    pub fn register(&mut self, id: impl Into<String>, factory: EngineFactory) {
        self.factories.push((id.into(), factory));
    }

    /// A registry with every engine compiled into this build.
    ///
    /// Construction opens each engine's *metadata* -- config, front-end tables,
    /// tokenizer -- but not the ONNX sessions, so listing installed engines in
    /// the settings screen does not page in 887 MB of weights. The sessions are
    /// opened when a job starts.
    /// Register every engine compiled into this build.
    pub fn with_builtins(provider: Provider, caps: &MachineCapabilities) -> Result<Self> {
        // The factory is a `'static` closure, so it owns a clone of the
        // capability snapshot. That is deliberate rather than a lifetime
        // workaround: a job must not have its VRAM budget silently change if
        // the user opens a different app on the GPU while it runs.
        let caps = caps.clone();
        let mut reg = Self::new();
        reg.register(
            "sravaani",
            Arc::new(move |art: &EngineArtifacts| {
                Ok(Box::new(crate::sravaani::open(art, provider, &caps)?) as Box<dyn AsrEngine>)
            }),
        );
        Ok(reg)
    }

    /// Ids of every engine this registry can construct.
    pub fn ids(&self) -> Vec<&str> {
        self.factories.iter().map(|(id, _)| id.as_str()).collect()
    }

    /// Whether an engine with this id is registered.
    pub fn contains(&self, id: &str) -> bool {
        self.factories.iter().any(|(k, _)| k == id)
    }

    /// Construct an engine by id.
    /// Construct an engine by id, refusing if its files are incomplete.
    pub fn create(&self, id: &str, artifacts: &EngineArtifacts) -> Result<Box<dyn AsrEngine>> {
        let (_, f) = self
            .factories
            .iter()
            .find(|(k, _)| k == id)
            .ok_or_else(|| err_unknown_engine(id, &self.ids()))?;
        let missing = artifacts.missing();
        if !missing.is_empty() {
            return Err(vaani_core::err::model_unavailable()
                .with_cause(format!("engine {id} is missing {}", missing.join(", "))));
        }
        f(artifacts)
    }
}

fn err_unknown_engine(id: &str, known: &[&str]) -> vaani_core::VaaniError {
    vaani_core::err::internal()
        .with_cause(format!("unknown ASR engine {id:?}; known: {known:?}"))
        .with_recovery("Run `vaani models list` to see the installed engines.")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn artifacts(dir: &Path) -> EngineArtifacts {
        EngineArtifacts {
            frontend_dir: dir.join("fe"),
            encoder: dir.join("enc.onnx"),
            decoder_joint: dir.join("dj.onnx"),
            tokenizer: dir.join("tok.model"),
        }
    }

    #[test]
    fn missing_lists_every_absent_artefact() {
        let a = artifacts(Path::new("/nonexistent"));
        let m = a.missing();
        assert_eq!(m.len(), 6, "expected all six to be reported missing: {m:?}");
        assert!(!a.is_complete());
        // The message must name concrete paths so a user can act on it.
        assert!(m.iter().all(|s| s.contains('/') || s.contains('\\')));
    }

    #[test]
    fn unknown_engine_reports_the_known_ones() {
        let reg = EngineRegistry::new();
        let e = reg
            .create("nope", &artifacts(Path::new("/nonexistent")))
            .unwrap_err();
        let chain = e.debug_chain();
        assert!(chain.contains("nope"), "{chain}");
        assert!(
            e.user.recovery.is_some(),
            "an unknown model needs a next step"
        );
    }

    #[test]
    fn registry_reports_its_ids() {
        let mut reg = EngineRegistry::new();
        assert!(reg.ids().is_empty());
        assert!(!reg.contains("x"));
        reg.register(
            "x",
            Arc::new(|_a: &EngineArtifacts| Err(vaani_core::err::internal())),
        );
        assert_eq!(reg.ids(), vec!["x"]);
        assert!(reg.contains("x"));
    }

    #[test]
    fn create_refuses_an_incomplete_installation_before_calling_the_factory() {
        // The factory would panic if it were reached with missing files, so
        // this also proves the check happens first.
        let mut reg = EngineRegistry::new();
        reg.register(
            "boom",
            Arc::new(|_a: &EngineArtifacts| panic!("factory must not be reached")),
        );
        let e = reg
            .create("boom", &artifacts(Path::new("/nonexistent")))
            .unwrap_err();
        assert_eq!(e.code(), vaani_core::ErrorCode::ModelUnavailable);
    }

    #[test]
    fn chunk_duration_is_derived_from_the_sample_count() {
        let c = AudioChunk::new(vec![0.0; 16_000], 16_000, Time::ZERO);
        assert_eq!(c.duration(), Time(1000));
        // A zero sample rate must not divide by zero.
        let c = AudioChunk::new(vec![0.0; 100], 0, Time::ZERO);
        assert_eq!(c.duration(), Time::ZERO);
    }
}

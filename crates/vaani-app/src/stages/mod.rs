//! The pipeline stages, one per step of `INTEGRATION_CONTRACT.md` §14.
//!
//! Every stage implements [`PipelineStage`](vaani_core::pipeline::PipelineStage)
//! and nothing else: stages never call each other, and only the orchestrator
//! constructs their contexts and advances phases.

pub mod asr;
pub mod diar;
pub mod export;
pub mod media;
pub mod store;
pub mod vad;

pub use asr::AsrStage;
pub use diar::{DiarStage, Diarize};
pub use export::ExportStage;
pub use media::{DecodeFn, MediaStage};
pub use store::StoreStage;
pub use vad::VadStage;

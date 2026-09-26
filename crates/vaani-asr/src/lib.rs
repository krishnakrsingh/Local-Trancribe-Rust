//! # vaani-asr
//!
//! The speech-recognition subsystem: the [`AsrEngine`](engine::AsrEngine)
//! abstraction, the engine registry, the log-mel front end, and the SraVaani
//! adapter.
//!
//! ## Module map
//!
//! | module | owns | must not know about |
//! |---|---|---|
//! | [`frontend`] | waveform -> log-mel, exactly as the model was trained | ONNX, tokens, time bases |
//! | [`engine`] | the `AsrEngine` trait, the registry, `AudioChunk` | any specific model |
//! | [`sravaani`] | the SraVaani graphs and the TDT decode loop | chunking, VAD, diarization |
//!
//! ## The two contracts that matter
//!
//! 1. **Times are chunk-local.** An engine returns times relative to the start
//!    of the chunk it was given, and learns nothing about the chunk's offset in
//!    the source media. `AudioChunk::source_offset` exists for logging only.
//!    `vaani_core::Transcript::push_chunk` does the rebasing, in one place, so
//!    chunking stays an implementation detail of the pipeline rather than
//!    something every engine has to get right.
//!
//! 2. **The front end is a faithful port, not an improvement.** Every constant
//!    comes from the model's own `preproc.json`. Bessel correction, a symmetric
//!    (not periodic) Hann window, `log_zero_guard_value == 2^-24`, and the
//!    single extra STFT frame are all deliberate and all cost accuracy if
//!    "corrected". See the module docs on [`frontend`] for each one.
//!
//! ## Verification
//!
//! * `frontend`'s unit tests cover the degenerate cases that produce NaN
//!   (silence, a one-frame clip) and the frame-count contract.
//! * `tests/frontend_parity.rs` compares the log-mel against values computed by
//!   PyTorch from the same `preproc` tensors.
//! * `tests/sravaani_transcribe.rs` runs the real model end to end and checks
//!   the decode invariants that only show up with real weights.
//!
//! Both integration tests skip with a printed reason when
//! `artifacts/sravaani/` is absent, so a fresh checkout is still testable.

#![warn(missing_docs)]

pub mod engine;
pub mod frontend;
pub mod sravaani;

pub use engine::{
    AsrEngine, AsrTimings, AudioChunk, ChunkResult, EngineArtifacts, EngineFactory, EngineRegistry,
};
pub use frontend::{FrontendParams, FrontendTables, LogMel, log_mel};
pub use sravaani::{
    SraVaaniConfig, SraVaaniEngine, artifacts_in, default_dir, open as open_sravaani,
};

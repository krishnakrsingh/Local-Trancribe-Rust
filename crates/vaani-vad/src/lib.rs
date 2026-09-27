//! Voice activity detection and speech chunking for VaaniDesk.
//!
//! Owned by agent **A3**. The `Cargo.toml` in this directory is frozen by
//! the architect and is part of the integration surface; the contents of this
//! file are yours.
//!
//! If you need a dependency that is not in the manifest, do not edit the
//! manifest. Report it under `INTEGRATION NOTES` and the architect will add it
//! in one commit that every agent rebases onto.
//!
//! Read `docs/agent/PARALLEL_OWNERSHIP.md` before the first edit.
//!
//! # What is here
//!
//! | Module | What it is |
//! |---|---|
//! | [`dsp`] | framing and a 512-point power spectrum, on the ASR front end's grid |
//! | [`features`] | per-frame measurements and the rule that scores them |
//! | [`noise`] | the adaptive noise floor: a trailing-window quantile |
//! | [`smoothing`] | hysteresis and the two duration rules, as pure functions |
//! | [`energy`] | the energy-based [`VadEngine`] |
//! | [`neural`] | a Silero VAD ONNX [`VadEngine`], **unvalidated -- no weights exist** |
//! | [`chunking`] | the chunk policy, which is what actually matters downstream |
//! | [`config`] | every tuned constant, in one `Serialize` struct |
//!
//! # The one-paragraph version
//!
//! A threshold on RMS fails on real meetings in five distinct ways -- it does
//! not know the room, one loud speaker raises the floor for everyone, loudness
//! is not speech, broadband noise is not distinguishable from speech by level,
//! and a steady sound is not a talking sound. So [`energy::EnergyVad`] combines
//! five shape features with an adaptive floor under a **product** rule (a frame
//! must be both audible above this room *and* shaped like speech), vetoes
//! steady sources with a spectral-flux test, and then applies hysteresis plus
//! minimum-speech and minimum-silence rules. The parameters and what each one
//! buys are in [`config`].
//!
//! # The part that matters downstream is not the detector
//!
//! It is [`chunking`]. A detector that finds every word and a chunker that cuts
//! every other one produce the same transcript. The chunker's job is to decide
//! where a decoder re-prime is worth paying for, to never cut a word, and to
//! bound the chunk length so a monotone speaker cannot produce an
//! out-of-memory error forty minutes into a job. See that module.
//!
//! # Two engines, one policy
//!
//! [`energy::EnergyVad`] and [`neural::SileroVad`] are two implementations of
//! the frozen [`vaani_core::VadEngine`] trait, and they share their
//! post-processing *value* -- one [`smoothing::SmoothingConfig`] -- rather than
//! their code path duplicating it. `neural` is honest about being unvalidated.

#![warn(missing_docs)]

pub mod chunking;
pub mod config;
pub mod dsp;
pub mod energy;
pub mod features;
pub mod neural;
pub mod noise;
pub mod smoothing;

pub use chunking::{Chunk, ChunkIter, Chunker, Span};
pub use config::{ChunkPolicy, EnergyVadConfig};
pub use energy::{EnergyVad, FrameAnalysis, VadStats};
pub use features::{FeatureWeights, FrameFeatures};
pub use neural::{SileroVad, preferred_engine};
pub use smoothing::{Run, SmoothingConfig};

/// The sample rate both engines require. The pipeline resamples before calling.
pub const SAMPLE_RATE: u32 = dsp::SAMPLE_RATE;

/// Crate version, from the workspace.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;
    use vaani_core::media::AudioBuffer;
    use vaani_core::pipeline::VadEngine;
    use vaani_core::{CancelToken, Time};

    #[test]
    fn the_two_exported_engines_both_satisfy_the_frozen_trait() {
        // The contract is a trait object in the pipeline, so both engines have
        // to be usable as one. This is a compile-time assertion dressed as a
        // test so the failure is readable.
        //
        // Only the energy engine appears in the list. `SileroVad` satisfies the
        // trait too, but constructing one requires ONNX weights that this
        // repository does not contain, so including it here would make this test
        // assert something it cannot check; `neural.rs` carries the equivalent
        // compile-time assertion for that path.
        let engines: Vec<Box<dyn VadEngine>> = vec![
            Box::new(EnergyVad::new()),
            Box::new(EnergyVad::new().with_name("energy-tuned", "Tuned")),
        ];
        for e in &engines {
            assert_eq!(e.sample_rate(), 16_000);
            assert!(!e.id().is_empty());
            assert!(!e.name().is_empty());
        }
    }

    #[test]
    fn the_crate_reports_its_sample_rate_once() {
        assert_eq!(SAMPLE_RATE, dsp::SAMPLE_RATE);
        assert_eq!(SAMPLE_RATE, 16_000);
    }

    #[test]
    fn the_default_detector_and_chunker_compose_on_silence() {
        let vad = EnergyVad::new();
        let a = AudioBuffer::mono(vec![0.0; 16_000 * 60], 16_000, Time::ZERO);
        let tl = vad.detect(&a, &CancelToken::new()).unwrap();
        let chunks = Chunker::new(ChunkPolicy::default()).plan(&tl, a.duration());
        assert!(tl.segments.is_empty());
        assert!(chunks.is_empty(), "silence must not produce ASR chunks");
    }
}

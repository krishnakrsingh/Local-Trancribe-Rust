//! SraVaani 1.0: a FastConformer encoder with a hybrid TDT transducer decoder.
//!
//! # Architecture as exported
//!
//! Two ONNX graphs plus a SentencePiece tokenizer, joined by a decode loop of
//! about forty lines of Rust. That is the whole inference stack:
//!
//! ```text
//!   log-mel [1,128,T] fp16 Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€“Â¶ encoder Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€“Â¶ enc_out [1,1024,ceil(T/8)] fp16
//!                                                       Ã¢â€â€š
//!   per emission step, 0.26-0.72 per frame:             Ã¢â€“Â¼
//!     decoder_joint(f[1,1024,1], last_id, h[1,1,640], c[1,1,640])
//!       Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€“Â¶ logits[1,1,1,5006] Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€“Â¶ argmax Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€“Â¶ (token, duration)
//! ```
//!
//! The joint runs once per *emission step*, not once per frame. TDT predicts
//! how many frames each emitted token owns, so a decoder that advanced a frame
//! at a time would spend most of its time on blanks. The resulting ratio is
//! audio dependent -- measured at 0.262 steps/frame on a synthetic two-tone
//! signal and 0.714 on 40 s of real meeting speech -- and
//! `transcribes_without_producing_nonsense` asserts it stays in a plausible
//! band, because reading the duration logits at the wrong offset still produces
//! a finite, plausible-looking transcript rather than an error.
//!
//! # fp16 encoder, fp32 joint
//!
//! `h` and `c` are the prediction network's LSTM state and they are threaded
//! through every emission step of an entire utterance. In fp16 that state
//! accumulates rounding error over minutes of audio and the decode quietly
//! degrades. The joint is 43 MB so fp32 there costs nothing measurable; the
//! encoder is 887 MB and holds essentially all of the arithmetic, so fp16 there
//! buys roughly 2x the speed. See `docs/decisions/ADR-004-sravaani-integration.md`.
//!
//! # Two rules in the decode loop that must not be "cleaned up"
//!
//! * **State advances only on a non-blank.** On blank, `h`, `c` and `last` are
//!   untouched and only the frame index moves. Standard RNN-T semantics, and
//!   what NeMo's `tdt_label_looping` does; treating a blank as a state update
//!   makes the model emit garbage.
//! * **The `max_symbols` guard adds an *extra* frame on top of the skips.** It
//!   is a safety valve for a blank that predicts duration 0, which would
//!   otherwise spin forever. NeMo's greedy TDT has the identical guard and it
//!   changes the output on pathological frames.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use sentencepiece_rs::SentencePieceProcessor;
use vaani_core::{CancelToken, Language, Result, Segment, Time, Word, err};
use vaani_inference::{
    CudaTuning, F16Tensor, F32Tensor, I32Tensor, I64Tensor, Input, MachineCapabilities,
    OnnxSession, OutputTensor, Provider,
};

use crate::engine::{AsrEngine, AsrTimings, AudioChunk, ChunkResult, EngineArtifacts};
use crate::frontend::{FrontendParams, FrontendTables, log_mel};

/// Architecture constants from the model's `config.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)] // field names are the documentation
pub struct SraVaaniConfig {
    pub vocab_size: usize,
    pub blank_id: i32,
    /// Frames each duration index represents: `durations[i]` frames.
    pub durations: [i32; 5],
    pub num_durations: usize,
    pub pred_hidden: usize,
    pub pred_rnn_layers: usize,
    pub max_symbols: usize,
    pub enc_d_model: usize,
    pub feat_in: usize,
    pub subsampling_factor: usize,
}

impl SraVaaniConfig {
    /// Parse the model's `config.json`, validating the invariants the decode
    /// loop relies on rather than trusting the file.
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| err::model_unavailable().with_cause(format!("{}: {e}", path.display())))?;
        #[derive(serde::Deserialize)]
        struct Raw {
            vocab_size: usize,
            blank_id: i32,
            durations: Vec<i32>,
            num_durations: usize,
            pred_hidden: usize,
            pred_rnn_layers: usize,
            max_symbols: usize,
            enc_d_model: usize,
            feat_in: usize,
            subsampling_factor: usize,
        }
        let r: Raw = serde_json::from_str(&text)
            .map_err(|e| err::model_corrupt().with_cause(format!("{}: {e}", path.display())))?;

        if r.num_durations != 5 || r.durations.len() != r.num_durations {
            return Err(err::model_corrupt().with_cause(format!(
                "durations {:?} disagrees with num_durations {}",
                r.durations, r.num_durations
            )));
        }
        if r.blank_id as usize != r.vocab_size {
            return Err(err::model_corrupt().with_cause(format!(
                "blank_id {} must equal vocab_size {}: the prediction network uses \
                 blank-as-padding, so the two cannot differ",
                r.blank_id, r.vocab_size
            )));
        }
        let mut durations = [0i32; 5];
        durations.copy_from_slice(&r.durations);
        Ok(Self {
            vocab_size: r.vocab_size,
            blank_id: r.blank_id,
            durations,
            num_durations: r.num_durations,
            pred_hidden: r.pred_hidden,
            pred_rnn_layers: r.pred_rnn_layers,
            max_symbols: r.max_symbols,
            enc_d_model: r.enc_d_model,
            feat_in: r.feat_in,
            subsampling_factor: r.subsampling_factor,
        })
    }

    /// Width of the joint's fused logit vector: `vocab` symbols, then the blank
    /// (which sits at index `vocab_size`), then the duration logits.
    #[must_use]
    pub fn logit_width(&self) -> usize {
        self.vocab_size + 1 + self.num_durations
    }

    /// Seconds of audio per encoder output frame.
    ///
    /// Exact: `subsampling * hop / sample_rate`. Being a factor of 8 out is
    /// the classic "the transcript is three minutes behind the video" bug, so
    /// this is computed once and nowhere else.
    #[must_use]
    pub fn frame_seconds(&self, frontend: &FrontendParams) -> f64 {
        self.subsampling_factor as f64 * frontend.hop_length as f64 / frontend.sample_rate as f64
    }
}

/// One token emitted by the decoder, with the encoder frame it was emitted on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)] // field names are the documentation
pub struct EmittedToken {
    pub id: i32,
    /// Encoder frame index, already in chunk-local terms.
    pub frame: i32,
}

/// Reused buffers for the decode loop.
///
/// The loop runs ~40 times per second of audio and every buffer is the same
/// size each time, so allocating per step is measurable. These live here and
/// are reset per call.
struct Scratch {
    /// `[enc_d_model]` the current encoder frame, for the joint's `f` input.
    frame: Vec<f32>,
    /// The previous step's fused logits.
    logits: Vec<f32>,
    /// `[layers * hidden]`, flattened from `[layers, 1, hidden]`.
    h: Vec<f32>,
    c: Vec<f32>,
}

/// A loaded SraVaani engine.
pub struct SraVaaniEngine {
    config: SraVaaniConfig,
    frontend: FrontendParams,
    tables: FrontendTables,
    tokenizer: SentencePieceProcessor,
    model_version: String,
    scratch: Mutex<Scratch>,
    /// Granularity of the encoder's time dimension. `0` disables bucketing.
    ///
    /// Atomic rather than a plain field so the performance lab can flip it
    /// between runs without contending with a transcribing worker.
    frame_bucket_slot: std::sync::atomic::AtomicUsize,
    /// `None` until `load` succeeds. An ONNX session cannot be "half loaded",
    /// so the options are presence of a session.
    sessions: Mutex<Option<Sessions>>,
}

struct Sessions {
    encoder: OnnxSession,
    decoder: OnnxSession,
}

impl SraVaaniEngine {
    /// Build from artefacts without touching the model weights.
    ///
    /// Cheap enough (a few hundred KB of text and floats) to run on the UI
    /// thread when the settings screen lists installed engines, so a user can
    /// see what is available without committing to a multi-hundred-megabyte
    /// load.
    pub fn new(artifacts: &EngineArtifacts) -> Result<Self> {
        // "Not downloaded yet" is the commonest first-run state, so it is the
        // error that must always arrive with a next step. Attaching it here, at
        // the point the files are read, is what guarantees it: attaching it in
        // the caller leaves the constructor's own error bare.
        let config = SraVaaniConfig::load(&artifacts.frontend_dir).map_err(with_download_hint)?;
        let frontend = FrontendParams::load(&artifacts.frontend_dir).map_err(with_download_hint)?;
        let tables = FrontendTables::load(&artifacts.frontend_dir, &frontend)?;

        // The front end and the encoder graph must agree on the mel count, or
        // the first `run` fails deep inside ORT with an opaque shape mismatch.
        if frontend.n_mels != config.feat_in {
            return Err(err::model_corrupt().with_cause(format!(
                "preproc has {} mel bands but the model expects {}",
                frontend.n_mels, config.feat_in
            )));
        }

        let tokenizer = SentencePieceProcessor::open(&artifacts.tokenizer).map_err(|e| {
            err::model_load_failed().with_cause(format!("{}: {e}", artifacts.tokenizer.display()))
        })?;

        let state = config.pred_rnn_layers * config.pred_hidden;
        Ok(Self {
            config,
            frontend,
            tables,
            tokenizer,
            model_version: "1.0".to_string(),
            frame_bucket_slot: std::sync::atomic::AtomicUsize::new(
                frame_bucket_override().unwrap_or(ENCODER_FRAME_BUCKET),
            ),
            scratch: Mutex::new(Scratch {
                frame: vec![0f32; 1024],
                logits: vec![0f32; 5006],
                h: vec![0f32; state],
                c: vec![0f32; state],
            }),
            sessions: Mutex::new(None),
        })
    }

    /// Seconds of audio per encoder frame, e.g. 0.080 for SraVaani at 16 kHz.
    #[must_use]
    pub fn frame_seconds(&self) -> f64 {
        self.config.frame_seconds(&self.frontend)
    }

    #[must_use]
    /// The model's architecture constants.
    pub fn config(&self) -> &SraVaaniConfig {
        &self.config
    }

    /// The greedy TDT loop.
    ///
    /// `enc` is `[enc_d_model][frames]`, exactly the layout `enc_out` comes
    /// back in, so frame `t` is a stride-`frames` read and no transpose is
    /// needed anywhere.
    fn greedy_decode(
        &self,
        enc: &F32Tensor,
        valid_frames: usize,
        sessions: &mut Sessions,
        cancel: &CancelToken,
    ) -> Result<(Vec<EmittedToken>, usize)> {
        let cfg = &self.config;
        let d = cfg.enc_d_model;
        let frames = enc.shape[2] as usize;
        let width = cfg.logit_width();
        if enc.shape[1] as usize != d {
            return Err(err::model_load_failed().with_cause(format!(
                "enc_out has {} channels, expected {d}",
                enc.shape[1]
            )));
        }
        let mut sc = self.scratch.lock().map_err(|_| poisoned())?;

        // Reset the transducer state. Blank doubles as the start token: the
        // prediction network's embedding has vocab_size + 1 rows, so `blank_id`
        // is a real in-range lookup, not a sentinel.
        sc.h.fill(0.0);
        sc.c.fill(0.0);
        let mut last = cfg.blank_id;

        let shape3 = vec![1, d as i64, 1];
        let state_shape = vec![cfg.pred_rnn_layers as i64, 1, cfg.pred_hidden as i64];
        let tlen_t = I32Tensor::scalar(1);

        let mut tokens: Vec<EmittedToken> = Vec::new();
        let mut steps = 0usize;
        let mut t: i64 = 0;
        let t_max = valid_frames.min(frames) as i64;

        while t < t_max {
            cancel.check()?;

            // Cache the frame as a tensor once per outer iteration. With
            // duration 0 the inner loop re-runs the joint on the SAME frame
            // with advanced state, and this avoids re-slicing it each time.
            let ti = t as usize;
            for k in 0..d {
                sc.frame[k] = enc.data[k * frames + ti];
            }
            let f = F32Tensor::new(shape3.clone(), sc.frame.clone())?;

            let mut added = 0usize;
            let mut need = true;
            while need && added < cfg.max_symbols {
                cancel.check()?;
                steps += 1;

                let out = sessions.decoder.run(
                    &[
                        Input::f32("f", f.clone()),
                        Input::i32("tgt", I32Tensor::new(vec![1, 1], vec![last])?),
                        Input::i32("tlen", tlen_t.clone()),
                        Input::f32("h_in", F32Tensor::new(state_shape.clone(), sc.h.clone())?),
                        Input::f32("c_in", F32Tensor::new(state_shape.clone(), sc.c.clone())?),
                    ],
                    &["logits", "h_out", "c_out"],
                )?;

                let logits = out[0].as_f32()?;
                if logits.data.len() != width {
                    return Err(err::model_load_failed().with_cause(format!(
                        "decoder_joint produced {} logits, expected {width}",
                        logits.data.len()
                    )));
                }
                sc.logits.copy_from_slice(&logits.data);

                // No log-softmax: argmax is invariant under it, and applying it
                // would cost a 5006-element exp on every emission step.
                let (k, skip) = argmax_pair(&sc.logits, cfg);

                if k != cfg.blank_id {
                    tokens.push(EmittedToken {
                        id: k,
                        frame: t as i32,
                    });
                    // State advances ONLY here. A blank is a pure time advance.
                    sc.h.copy_from_slice(&out[1].as_f32()?.data);
                    sc.c.copy_from_slice(&out[2].as_f32()?.data);
                    last = k;
                }
                added += 1;
                t += skip as i64;
                // Duration 0 = "this token consumes no acoustic time", so run
                // the joint again on the same frame with the advanced state.
                need = skip == 0;
            }
            if added == cfg.max_symbols {
                // The guard, faithfully: an extra frame *on top of* whatever the
                // skips already advanced.
                t += 1;
            }
        }
        Ok((tokens, steps))
    }

    /// The half of [`AsrEngine::transcribe`] that needs the session lock:
    /// encoder, then decoder.
    ///
    /// Split out so [`AsrEngine::transcribe_batch`] can compute the front end
    /// for many chunks at once and then walk these one at a time. There is
    /// exactly one place in this file that holds the session lock across real
    /// work, and it is here -- which is what makes the batch path's "the
    /// GPU half stays serial" a structural fact rather than a convention that a
    /// later edit can quietly break.
    #[allow(clippy::type_complexity)]
    fn encode_and_decode(
        &self,
        mel: &crate::frontend::LogMel,
        cancel: &CancelToken,
    ) -> Result<(Vec<EmittedToken>, u64, u64, usize, usize)> {
        let mut guard = self.sessions.lock().map_err(|_| poisoned())?;
        let sessions = guard
            .as_mut()
            .ok_or_else(|| err::model_unavailable().with_cause("engine not loaded"))?;

        let t1 = Instant::now();
        let (enc, valid_frames) = self.encode(sessions, mel, cancel)?;
        let encode_ms = t1.elapsed().as_millis() as u64;
        cancel.check()?;

        let t2 = Instant::now();
        let (tokens, steps) = self.greedy_decode(&enc, valid_frames, sessions, cancel)?;
        let decode_ms = t2.elapsed().as_millis() as u64;

        Ok((tokens, encode_ms, decode_ms, valid_frames, steps))
    }

    /// Encode several chunks in **one** call, along the graph's dynamic batch
    /// axis.
    ///
    /// The grouping rules are what make this a win rather than a risk:
    ///
    /// * All chunks in a group are padded to the **same** time bucket, so the
    ///   graph sees one shape. The largest chunk sets the bucket, so a group
    ///   pads no more than its own worst member would have padded alone -- and
    ///   usually less, because it replaces several distinct shapes with one.
    /// * `feature_lengths` carries each chunk's **true** valid frame count, so
    ///   the graph masks every padded tail exactly as it already does for the
    ///   frames it pads itself. That is why the frozen goldens are expected to
    ///   pass exactly, and they are checked on every build.
    ///
    /// One result per input chunk, in input order, each carrying its own
    /// reported length, because that length differs per chunk even inside one
    /// batched call.
    fn encode_group(
        &self,
        group: &[crate::frontend::LogMel],
        cancel: &CancelToken,
    ) -> Result<Vec<OneEncoded>> {
        cancel.check()?;
        if group.is_empty() {
            return Ok(Vec::new());
        }

        // One bucket for the whole group, from the longest member.
        let longest = group.iter().map(|m| m.n_frames).max().unwrap_or(0);
        let padded = self.bucketed_frames(longest);
        let batch = i64::try_from(group.len()).map_err(|_| {
            err::internal().with_cause("encoder group is larger than i64 can count")
        })?;

        // Mel-major within each chunk, chunks stacked along the new leading
        // axis. `pad_frames` zero-fills the time axis; the stack places each
        // chunk's rows in turn, so the tensor is exactly `[N, mels, padded]`.
        let mut data: Vec<f32> = Vec::with_capacity(group.len() * self.frontend.n_mels * padded);
        for mel in group {
            let rows = pad_frames(mel, padded);
            debug_assert_eq!(rows.len(), self.frontend.n_mels * padded);
            data.extend_from_slice(&rows);
        }

        let feats = F16Tensor::from_f32(&F32Tensor::new(
            vec![batch, self.frontend.n_mels as i64, padded as i64],
            data,
        )?);
        // One true length per batch item. This is what the graph masks with, so
        // a short chunk in a long-chunk's group is not decoded as padding.
        let lengths = I64Tensor::new(
            vec![batch],
            group.iter().map(|m| m.valid_frames as i64).collect(),
        )?;

        let t = Instant::now();
        let out = {
            let mut guard = self.sessions.lock().map_err(|_| poisoned())?;
            let sessions = guard
                .as_mut()
                .ok_or_else(|| err::model_unavailable().with_cause("engine not loaded"))?;
            sessions.encoder.run(
                &[
                    Input::f16("feats", feats),
                    Input::i64("feature_lengths", lengths),
                ],
                &["enc_out", "enc_lengths"],
            )?
        };
        let encode_ms = t.elapsed().as_millis() as u64;
        cancel.check()?;

        let enc = match &out[0] {
            OutputTensor::F16(t) => t.to_f32(),
            OutputTensor::F32(t) => t.clone(),
            other => {
                return Err(err::model_load_failed()
                    .with_cause(format!("enc_out had an unexpected type: {other:?}")));
            }
        };
        if enc.shape.len() != 3 {
            return Err(err::model_load_failed()
                .with_cause(format!("enc_out was not rank 3: {:?}", enc.shape)));
        }
        if enc.shape[0] != batch {
            return Err(err::model_load_failed().with_cause(format!(
                "asked for a batch of {batch} and got {}",
                enc.shape[0]
            )));
        }
        let reported = out[1].as_i64()?.data.clone();
        if reported.len() != group.len() {
            return Err(err::model_load_failed().with_cause(format!(
                "encoder reported {} lengths for a batch of {}",
                reported.len(),
                group.len()
            )));
        }

        let stride = enc.shape[2] as usize;
        let plane = enc.shape[1] as usize;
        let mut per_chunk = Vec::with_capacity(group.len());
        for (i, &frames) in reported.iter().enumerate() {
            if frames <= 0 {
                return Err(err::model_load_failed()
                    .with_cause(format!("encoder reported a length of {frames}")));
            }
            if frames as usize > stride {
                return Err(err::model_load_failed().with_cause(format!(
                    "encoder reported {frames} valid frames but only produced {stride}"
                )));
            }
            // One item's [plane, stride] block, copied so the next batch's
            // output buffer can be reused without invalidating it.
            let start = i * plane * stride;
            let slice = &enc.data[start..start + plane * stride];
            per_chunk.push(OneEncoded {
                data: slice.to_vec(),
                plane,
                stride,
                valid_frames: frames as usize,
                encode_ms,
            });
        }
        Ok(per_chunk)
    }

    /// Decode one chunk's encoder output and group its tokens into segments.
    ///
    /// The decode half is strictly per chunk: it carries LSTM state between
    /// emission steps and its graph is fixed at `batch=1`, so batching it would
    /// be a different model rather than a faster one.
    fn finish_one(
        &self,
        one: &OneEncoded,
        frontend_ms: u64,
        cancel: &CancelToken,
    ) -> Result<ChunkResult> {
        let enc = F32Tensor::new(
            vec![1, one.plane as i64, one.stride as i64],
            one.data.clone(),
        )?;
        let t2 = Instant::now();
        let (tokens, steps) = {
            let mut guard = self.sessions.lock().map_err(|_| poisoned())?;
            let sessions = guard
                .as_mut()
                .ok_or_else(|| err::model_unavailable().with_cause("engine not loaded"))?;
            self.greedy_decode(&enc, one.valid_frames, sessions, cancel)?
        };
        let decode_ms = t2.elapsed().as_millis() as u64;
        let segments = self.tokens_to_segments(&tokens, one.valid_frames)?;
        Ok(ChunkResult {
            segments,
            language: None,
            timings: AsrTimings {
                frontend_ms,
                encode_ms: one.encode_ms,
                decode_ms,
                encoder_frames: one.valid_frames,
                decode_steps: steps,
            },
        })
    }

    /// The raw decode output: one `(token id, surface piece)` per emitted
    /// token, in emission order.
    ///
    /// Exists because "the transcript has spaces in the wrong places" is
    /// impossible to diagnose from the transcript alone -- every sub-word piece
    /// looks like its own word. `vaani --debug dump-tokens` prints this, and
    /// it is what the word-grouping test asserts against.
    pub fn debug_tokens(
        &self,
        chunk: &AudioChunk,
        cancel: &CancelToken,
    ) -> Result<Vec<(i32, String)>> {
        let mel = log_mel(&chunk.samples, &self.frontend, &self.tables)?;
        let mut guard = self.sessions.lock().map_err(|_| poisoned())?;
        let sessions = guard
            .as_mut()
            .ok_or_else(|| err::model_unavailable().with_cause("engine not loaded"))?;
        let (enc, valid) = self.encode(sessions, &mel, cancel)?;
        let (tokens, _) = self.greedy_decode(&enc, valid, sessions, cancel)?;
        Ok(tokens
            .into_iter()
            .map(|t| {
                let piece = usize::try_from(t.id)
                    .ok()
                    .and_then(|i| self.tokenizer.id_to_piece(i).ok())
                    .unwrap_or_default()
                    .to_string();
                (t.id, piece)
            })
            .collect())
    }

    /// Group emitted pieces into words, and words into segments.
    ///
    /// A SentencePiece *piece* is not a word. "Ã Â¤â€˜Ã Â¤Å¸Ã Â¥â€¹Ã Â¤Â®Ã Â¥Ë†Ã Â¤Å¸Ã Â¤Â¿Ã Â¤â€¢Ã Â¤Â²Ã Â¥â‚¬" is seven pieces --
    /// `Ã¢â€“ÂÃ Â¤â€˜`, `Ã Â¤Å¸Ã Â¥â€¹`, `Ã Â¤Â®`, `Ã¢â€”Å’`, `Ã Â¤Å¸`, `Ã Â¤Â¿Ã Â¤â€¢`, `Ã Â¤Â²Ã Â¥â‚¬` -- and the metaspace marker `Ã¢â€“Â`
    /// is what says a new word begins. So:
    ///
    /// * pieces are concatenated with **no separator** into one word;
    /// * the word spans from its first piece's frame to its last piece's end,
    ///   which is what makes a click-to-seek land on the whole word;
    /// * one word is one segment.
    ///
    /// Storing one `Word` per piece is the mistake that makes a transcript look
    /// like it has a space after every syllable. It is invisible in a shape
    /// check and in a token-id comparison -- the ids are identical either way --
    /// and it is only visible in the text.
    ///
    /// Segments are later merged into phrases and attributed to speakers by
    /// diarization; the grouping here is deliberately the finest unit that is
    /// still a real word, so both of those steps have something to work with.
    fn tokens_to_segments(
        &self,
        tokens: &[EmittedToken],
        total_frames: usize,
    ) -> Result<Vec<Segment>> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let fd = self.frame_seconds();
        let end_all = Time::from_secs_f64(total_frames as f64 * fd);

        let mut segs: Vec<Segment> = Vec::new();
        // The word being accumulated, and the frame it started on.
        let mut text = String::new();
        let mut start_frame: Option<i32> = None;

        for tok in tokens {
            let piece = usize::try_from(tok.id)
                .ok()
                .and_then(|id| self.tokenizer.id_to_piece(id).ok())
                .unwrap_or_default();
            if piece.is_empty() {
                continue;
            }
            // A metaspace prefix means this piece opens a new word.
            if piece.starts_with('\u{2581}') && start_frame.is_some() {
                let start = Time::from_secs_f64(start_frame.expect("is_some") as f64 * fd);
                let end = Time::from_secs_f64(tok.frame as f64 * fd).max(start);
                segs.push(Segment::new(start, end).with_words(vec![Word::new(
                    std::mem::take(&mut text),
                    start,
                    end,
                )]));
                start_frame = None;
            }
            // `decode` renders the marker as a space; the first piece of a word
            // therefore decodes with a leading space we drop, and the rest
            // concatenate directly.
            let surface = self.tokenizer.decode(&[piece]).map_err(sp_error)?;
            text.push_str(surface.trim());
            if start_frame.is_none() {
                start_frame = Some(tok.frame);
            }
        }

        if let Some(sf) = start_frame {
            let start = Time::from_secs_f64(sf as f64 * fd);
            let end = Time::from_secs_f64(total_frames as f64 * fd).max(start);
            segs.push(Segment::new(start, end).with_words(vec![Word::new(text, start, end)]));
        }

        for s in &mut segs {
            s.end = s.end.min(end_all).max(s.start);
        }
        Ok(segs)
    }

    /// Run the encoder over a chunk's log-mel.
    ///
    /// Returns the encoder states **and the number of valid frames**, and the
    /// distinction is not pedantry.
    ///
    /// The log-mel tensor has `1 + valid_frames` time columns, because
    /// `torch.stft` with `center = true` emits one more frame than
    /// `feature_lengths` describes. The encoder subsamples the *tensor*, so its
    /// output has `ceil((1 + valid_frames) / 8)` columns -- 501 for a 40 s
    /// chunk, not 500. The extra column is padding, and the graph reports the
    /// true length in `enc_lengths`.
    ///
    /// PyTorch's reference decodes exactly `enc_len` frames, so using the
    /// tensor dimension instead decodes one spurious frame past the end of the
    /// audio. `enc_lengths` is requested for that reason and must not be
    /// optimised away as redundant with the shape.
    fn encode(
        &self,
        sessions: &mut Sessions,
        mel: &crate::frontend::LogMel,
        cancel: &CancelToken,
    ) -> Result<(F32Tensor, usize)> {
        cancel.check()?;
        // Round the time dimension up to a bucket before the graph sees it.
        //
        // The encoder takes a *dynamic* time dimension, and every distinct
        // value is a shape ONNX Runtime has never planned: a new execution
        // plan, a new cuDNN algorithm choice, and then only a fraction of the
        // time actually spent on the audio. Production chunk lengths are all
        // slightly different - 3123, 3052, 3170 frames - so a meeting paid
        // that fixed cost once per chunk. Measured on a 4 GB laptop GPU, a
        // 4.5-second chunk cost the same 490 ms to encode as a 60-second one,
        // which is the clearest possible statement that the cost was not in the
        // audio.
        //
        // Bucketing trades a little wasted arithmetic for far fewer plans. The
        // pad is zeros on the time axis and `feature_lengths` still carries the
        // true length, so the graph masks the tail exactly as it already does
        // for the frames it pads itself to reach a whole number of encoder
        // steps. The frozen goldens are what prove that, and they are checked
        // on every build.
        let padded = self.bucketed_frames(mel.n_frames);
        // The encoder graph's I/O is fp16, so the log-mel is converted once,
        // here. The front end itself stays f32 because the per-feature
        // normalisation divides by a standard deviation.
        let feats = F16Tensor::from_f32(&F32Tensor::new(
            vec![1, self.frontend.n_mels as i64, padded as i64],
            pad_frames(mel, padded),
        )?);
        let lengths = I64Tensor::scalar(mel.valid_frames as i64);
        let out = sessions.encoder.run(
            &[
                Input::f16("feats", feats),
                Input::i64("feature_lengths", lengths),
            ],
            &["enc_out", "enc_lengths"],
        )?;
        let enc = match &out[0] {
            OutputTensor::F16(t) => t.to_f32(),
            OutputTensor::F32(t) => t.clone(),
            other => {
                return Err(err::model_load_failed()
                    .with_cause(format!("enc_out had an unexpected type: {other:?}")));
            }
        };
        let reported = out[1].as_i64()?.data[0];
        if reported <= 0 {
            return Err(err::model_load_failed()
                .with_cause(format!("encoder reported a length of {reported}")));
        }
        if reported as usize > enc.shape[2] as usize {
            return Err(err::model_load_failed().with_cause(format!(
                "encoder reported {reported} valid frames but only produced {}",
                enc.shape[2]
            )));
        }
        Ok((enc, reported as usize))
    }

    /// The time-dimension length actually handed to the encoder graph.
    ///
    /// Rounds up to a multiple of [`ENCODER_FRAME_BUCKET`]. `0` disables
    /// bucketing, which is only useful for a benchmark that needs to reproduce
    /// the unbucketed cost.
    fn bucketed_frames(&self, n_frames: usize) -> usize {
        let bucket = self
            .frame_bucket_slot
            .load(std::sync::atomic::Ordering::Relaxed);
        if bucket == 0 {
            return n_frames;
        }
        n_frames.div_ceil(bucket) * bucket
    }

    /// Change the encoder's frame bucket. `0` disables bucketing.
    ///
    /// Exists for the performance lab's A/B mode, which needs both settings live
    /// in one process: this laptop's GPU drifts by twenty percent between runs,
    /// so the only trustworthy comparison interleaves them and takes medians.
    /// Two engines in one process also share the loaded weights on disk and the
    /// warm allocator, which is where most of the run-to-run variance lives.
    pub fn set_frame_bucket(&self, frames: usize) {
        self.frame_bucket_slot
            .store(frames, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Read the encoder's frame-bucket override from the environment.
///
/// Present so the performance lab can sweep the value without a rebuild for
/// each point. Absent, the engine uses [`ENCODER_FRAME_BUCKET`]. `0` means "no
/// bucketing", which is how the unbucketed cost is reproduced for comparison.
fn frame_bucket_override() -> Option<usize> {
    std::env::var("VAANI_ENCODER_FRAME_BUCKET")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
}

/// How many chunks go through the encoder in one call, when batching is enabled.
///
/// The encoder graph declares a **dynamic batch axis**, so a group of N chunks
/// is a batch of N rather than N separate executions, and measured on the
/// 14-minute reference recording that is worth 20.5 s to 13.0 s -- a 1.57x
/// end-to-end win, and it also cut run-to-run spread from 17.4-21.7 s down to
/// 12.6-13.1 s.
///
/// # It changes the transcript, so it is off by default
///
/// This is not a mathematically-equivalent reformulation. Batching feeds a
/// different tensor shape to the same graph, cuDNN selects different algorithms
/// for different shapes, and a different algorithm is a different
/// floating-point reduction order. On the reference recording the batched path
/// emits 6180 decode steps where the serial path emits 6183 and produces a
/// **different word stream** -- the same 1908 words in count, but not the same
/// text, as an FNV fingerprint over every word's bytes makes plain
/// (`c140c2e235e3a375` serial, `30d707e91dda3481` batched).
///
/// A four-chunk equivalence test *did* pass, on identical inputs, which is the
/// instructive part: a small case cannot distinguish this. The safe conclusion
/// is that batched encoding is a different numerical result of unknown quality,
/// not a faster spelling of the same one, and until there is a human reference
/// to score both against, "different text" cannot be called "as good".
///
/// So the ASR stage transcribes one chunk at a time, and this is reachable
/// only through [`AsrEngine::transcribe_streaming`] and the profiler's
/// `--batch` flag, which exist so the comparison can be re-run whenever a
/// reference exists.
pub const ENCODER_BATCH: usize = 4;

/// One chunk's slice of a batched encoder output, kept for the decode half.
///
/// `data` is a private copy of that item's `[plane, stride]` block rather than a
/// borrow, so the next batch's output buffer can be reused without invalidating
/// a decode that is still reading the previous one.
#[derive(Debug)]
pub struct OneEncoded {
    /// That item's `[plane, stride]` block, in row-major order.
    pub data: Vec<f32>,
    /// Encoder output channels (1024 for SraVaani 1.0).
    pub plane: usize,
    /// Padded encoder time steps.
    pub stride: usize,
    /// True length, from `enc_lengths`, not `stride`.
    pub valid_frames: usize,
    /// Wall time of the *whole* group, attributed to each chunk in it. Dividing
    /// it by the group size would be tidier and wrong: the cost is genuinely
    /// shared, and a per-chunk figure that added up to more than the group took
    /// would be a measurement artefact dressed up as a fact.
    pub encode_ms: u64,
}

/// Granularity, in mel frames, of the encoder's time dimension.
///
/// 512 frames is 5.12 s at a 10 ms hop. Chosen by measurement rather than by
/// taste: it turns a meeting's 23 differently-shaped chunks into a handful of
/// shapes while never padding a chunk by more than five seconds, and it cut the
/// encoder's wall time by roughly two thirds on a 14-minute recording. Halving
/// it buys back some of the wasted arithmetic and gives back some of the plan
/// cost; the curve is shallow on both sides, which is what makes this a safe
/// default rather than a tuned magic number.
pub const ENCODER_FRAME_BUCKET: usize = 512;

/// Copy `mel` into a `padded`-frame buffer, zero-filling the tail.
///
/// Mel-major, so each mel band's row is copied whole and the padding lands at
/// the end of every row. `padded <= mel.n_frames` returns the input untouched,
/// so a bucket smaller than the frame count can never lose data.
fn pad_frames(mel: &crate::frontend::LogMel, padded: usize) -> Vec<f32> {
    if padded <= mel.n_frames {
        return mel.data.clone();
    }
    let mut out = vec![0f32; mel.n_mels * padded];
    for m in 0..mel.n_mels {
        let (src, dst) = (m * mel.n_frames, m * padded);
        out[dst..dst + mel.n_frames].copy_from_slice(&mel.data[src..src + mel.n_frames]);
    }
    out
}

/// `argmax` over the symbol logits, and over the duration logits.///
/// Split out so it can be unit tested against hand-built logits, including the
/// boundary case where blank wins.
fn argmax_pair(logits: &[f32], cfg: &SraVaaniConfig) -> (i32, i32) {
    let (mut best, mut bi) = (f32::NEG_INFINITY, 0usize);
    // Symbols occupy 0..=vocab_size, where index `vocab_size` is blank. The
    // winning index is the absolute token id, so no offset subtraction.
    for (i, x) in logits.iter().take(cfg.vocab_size + 1).enumerate() {
        if *x > best {
            best = *x;
            bi = i;
        }
    }
    let (mut bskip, mut di) = (f32::NEG_INFINITY, 0usize);
    for j in 0..cfg.num_durations {
        let x = logits[cfg.vocab_size + 1 + j];
        if x > bskip {
            bskip = x;
            di = j;
        }
    }
    (bi as i32, cfg.durations[di])
}

/// Map a tokenizer failure into ours. A bad id is a model/artefact mismatch,
/// not a user error, so it is an internal failure with a concrete cause.
/// Attach the download hint to a "model not present" error.
fn with_download_hint(e: vaani_core::VaaniError) -> vaani_core::VaaniError {
    if e.code() == vaani_core::ErrorCode::ModelUnavailable {
        e.with_recovery(
            "Download SraVaani 1.0 from Settings, or from the MODEL REQUIRED panel \
             on the home screen. It is about 930 MB.",
        )
    } else {
        e
    }
}

fn sp_error(e: sentencepiece_rs::Error) -> vaani_core::VaaniError {
    err::internal().with_cause(format!("tokenizer: {e}"))
}

fn poisoned() -> vaani_core::VaaniError {
    err::internal().with_cause("engine mutex poisoned by a panicking worker thread")
}

impl std::fmt::Debug for SraVaaniEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SraVaaniEngine")
            .field("model_version", &self.model_version)
            .field(
                "loaded",
                &self.sessions.lock().map(|s| s.is_some()).unwrap_or(false),
            )
            .field("vocab", &self.config.vocab_size)
            .field("frame_seconds", &self.frame_seconds())
            .finish_non_exhaustive()
    }
}

impl AsrEngine for SraVaaniEngine {
    fn id(&self) -> &str {
        "sravaani"
    }

    fn model_name(&self) -> &str {
        "SraVaani 1.0"
    }

    fn model_version(&self) -> &str {
        &self.model_version
    }

    fn languages(&self) -> &[Language] {
        // The model card lists 65 languages; the picker offers the ones the
        // product is positioned for, and `Auto` covers the rest.
        &[
            Language::Auto,
            Language::Hindi,
            Language::Hinglish,
            Language::English,
        ]
    }

    fn supports_word_timestamps(&self) -> bool {
        true
    }

    fn load(&self, _cancel: &CancelToken) -> Result<()> {
        let mut guard = self.sessions.lock().map_err(|_| poisoned())?;
        if guard.is_some() {
            return Ok(()); // idempotent
        }
        *guard = None;
        Ok(())
    }

    fn unload(&self) {
        if let Ok(mut g) = self.sessions.lock() {
            *g = None;
        }
    }

    fn is_loaded(&self) -> bool {
        self.sessions.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    fn active_provider(&self) -> Option<vaani_inference::ActiveProvider> {
        // The encoder is the session that decides it, and both sessions are
        // created from the same provider, so either answers the question. Read
        // through the lock and ignore a poisoned one: "which device" must not
        // panic a UI that is only trying to draw a line of text.
        self.sessions
            .lock()
            .ok()
            .and_then(|g| g.as_ref().map(|s| s.encoder.provider_status().active))
    }

    fn cancel(&self) {
        // Cancellation travels through the `CancelToken` handed to
        // `transcribe`, which the job system owns and can cancel from the UI
        // thread. An engine-local token here would be a second source of truth
        // that nothing sets.
    }

    fn transcribe(&self, chunk: &AudioChunk, cancel: &CancelToken) -> Result<ChunkResult> {
        cancel.check()?;
        if chunk.sample_rate != self.frontend.sample_rate {
            return Err(err::internal().with_cause(format!(
                "chunk is {} Hz but the front end needs {} Hz; resampling is the media layer's job",
                chunk.sample_rate, self.frontend.sample_rate
            )));
        }

        let t0 = Instant::now();
        let mel = log_mel(&chunk.samples, &self.frontend, &self.tables)?;
        let frontend_ms = t0.elapsed().as_millis() as u64;
        cancel.check()?;

        let (tokens, encode_ms, decode_ms, valid_frames, steps) =
            self.encode_and_decode(&mel, cancel)?;
        cancel.check()?;

        let segments = self.tokens_to_segments(&tokens, valid_frames)?;

        Ok(ChunkResult {
            segments,
            language: None,
            timings: AsrTimings {
                frontend_ms,
                encode_ms,
                decode_ms,
                encoder_frames: valid_frames,
                decode_steps: steps,
            },
        })
    }

    fn transcribe_streaming(
        &self,
        chunks: &[AudioChunk],
        cancel: &CancelToken,
        sink: &mut dyn FnMut(usize, &ChunkResult) -> bool,
    ) -> Vec<Result<ChunkResult>> {
        let mut out: Vec<Result<ChunkResult>> = Vec::with_capacity(chunks.len());
        if chunks.is_empty() {
            return out;
        }

        // -- front ends, one chunk at a time -------------------------------
        //
        // Sequential, deliberately. This started as a `par_iter` across all
        // sixteen logical cores and it was **60% slower**: summed per-chunk
        // front-end cost went from 2123 ms to 3185 ms on the same 14-minute
        // recording, and wall time from 16.0 s to 17.3 s. Throttling the pool
        // to 2 and 4 threads did not rescue it.
        //
        // `log_mel` is memory-bandwidth-bound, not compute-bound. Per chunk it
        // reads ~480 000 samples, runs an FFT per frame and projects onto a
        // 128 x 257 filterbank -- a few megabytes of streaming traffic with no
        // reuse to be had. Sixteen threads doing that saturate memory bandwidth
        // and evict each other from L2/L3, so the parallel version pays
        // contention for the cores it was trying to use. Per-chunk front-end
        // time is not headroom that scales with cores.
        let mut mels: Vec<crate::frontend::LogMel> = Vec::with_capacity(chunks.len());
        let mut frontend_ms: Vec<u64> = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let t = Instant::now();
            match cancel
                .check()
                .and_then(|()| log_mel(&chunk.samples, &self.frontend, &self.tables))
            {
                Ok(mel) => {
                    frontend_ms.push(t.elapsed().as_millis() as u64);
                    mels.push(mel);
                }
                // A front-end failure short-circuits. Errors are reported in the
                // returned `Vec` at this chunk's index, not to the sink, which
                // has no way to surface one.
                Err(error) => {
                    out.push(Err(error));
                    return out;
                }
            }
        }

        // -- encoder, several chunks per call, drained per chunk ----------
        //
        // The encoder is ~57% of the job and its graph declares a **dynamic
        // batch axis**, so N chunks go through one `run()` as a batch of N
        // rather than N runs of one. `feature_lengths` is `[batch]` and carries
        // each chunk's true valid frame count, and the time axis is padded to a
        // single bucket for the whole group, which pads less than each chunk
        // would have padded alone.
        //
        // The sink fires the moment each chunk's decode finishes, so the UI sees
        // the meeting arrive while later chunks are still encoding. That is the
        // whole reason this is a callback and not a returned `Vec`.
        let mut halt = false;
        for (start, group) in mels.chunks(ENCODER_BATCH).enumerate() {
            if cancel.check().is_err() {
                break;
            }
            let offset = start * ENCODER_BATCH;
            let outs = match self.encode_group(group, cancel) {
                Ok(outs) => outs,
                Err(error) => {
                    // A group that cannot be encoded fails the job. There is no
                    // partial answer for a chunk that was never encoded, and
                    // inventing one would put silence in the transcript.
                    tracing::error!(
                        chunks = group.len(),
                        first = offset,
                        %error,
                        "encoder group failed; failing the job rather than leaving \
                         these chunks unrecognised"
                    );
                    for _ in 0..group.len() {
                        out.push(Err(err::model_load_failed().with_cause(format!(
                            "encoder group of {} chunks starting at {offset} failed",
                            group.len()
                        ))));
                    }
                    return out;
                }
            };
            for (i, one) in outs.into_iter().enumerate() {
                let index = offset + i;
                let result = self.finish_one(&one, frontend_ms[index], cancel);
                let failed = result.is_err();
                if let Ok(ok) = &result
                    && !sink(index, ok)
                {
                    halt = true;
                }
                out.push(result);
                if failed || halt {
                    return out;
                }
            }
        }
        // One result per input, so the caller can index by chunk without a
        // length check whether it ran out, was cancelled, or stopped early.
        while out.len() < chunks.len() {
            out.push(Err(err::cancelled()));
        }
        out
    }

    fn benchmark(&self, cancel: &CancelToken) -> Result<AsrTimings> {
        // A synthetic ten-second chunk is enough to warm cuDNN and page in the
        // weights without depending on a media file being present.
        let n = 10 * self.frontend.sample_rate as usize;
        let wav: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / self.frontend.sample_rate as f32;
                0.2 * (2.0 * std::f32::consts::PI * 180.0 * t).sin()
            })
            .collect();
        Ok(self
            .transcribe(
                &AudioChunk::new(wav, self.frontend.sample_rate, Time::ZERO),
                cancel,
            )?
            .timings)
    }
}

/// Construct a SraVaani engine from an artefacts directory.
pub fn open(
    artifacts: &EngineArtifacts,
    provider: Provider,
    caps: &MachineCapabilities,
) -> Result<SraVaaniEngine> {
    let engine = SraVaaniEngine::new(artifacts)?;
    // Reserve 256 MiB for the desktop compositor and other processes, which
    // DXGI's budget already counts but the BFC arena would happily take.
    let vram = caps
        .inference_adapter()
        .map(|a| a.free_vram_bytes().saturating_sub(256 * 1024 * 1024))
        .filter(|v| *v > 0);
    let sessions = Sessions {
        // The two graphs want opposite things, and giving them the same settings
        // is leaving most of the speed on the table.
        //
        // The encoder runs once per chunk on a *dynamic* time dimension, so a
        // CUDA graph would be re-captured for every new shape and cost more than
        // it saves. What it wants instead is not to see new shapes at all,
        // which is handled where the features are built.
        //
        // The decode joint is the opposite case: 21 nodes, every dimension
        // static, and it runs once per emission step - thousands of times per
        // chunk. At that ratio the cost is kernel-launch overhead, not
        // arithmetic, and capturing the graph is the right answer.
        //
        // Not switched on yet. `enable_cuda_graph=1` in this ONNX Runtime build
        // captures the run but then cannot hand the outputs back through the
        // ordinary path: the first emission step aborts inside `ort`'s tensor
        // extraction with "expected `typeinfo_ptr` to not be null". CUDA graph
        // in ORT is documented as pairing with IO binding, where outputs are
        // read from device buffers we own, so that is the shape this has to
        // take. Until then the loop pays per-step launch overhead, and the
        // honest numbers are the ones measured with it off.
        encoder: OnnxSession::load_tuned(
            &artifacts.encoder,
            provider,
            caps,
            vram,
            CudaTuning::default(),
        )?,
        decoder: OnnxSession::load_tuned(
            &artifacts.decoder_joint,
            provider,
            caps,
            vram,
            CudaTuning::default(),
        )?,
    };
    *engine.sessions.lock().map_err(|_| poisoned())? = Some(sessions);
    Ok(engine)
}

/// Default artefact layout inside a model directory.
pub fn artifacts_in(dir: &Path) -> EngineArtifacts {
    EngineArtifacts {
        frontend_dir: dir.to_path_buf(),
        encoder: dir.join("encoder.fp16.onnx"),
        decoder_joint: dir.join("decoder_joint.onnx"),
        tokenizer: dir.join("tokenizer.model"),
    }
}

/// Where the model directory lives, honouring `VAANI_ARTEFACTS`.
pub fn default_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("VAANI_ARTEFACTS") {
        return PathBuf::from(d);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|root| root.join("artifacts").join("sravaani"))
        .unwrap_or_else(|| PathBuf::from("artifacts/sravaani"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SraVaaniConfig {
        SraVaaniConfig {
            vocab_size: 5000,
            blank_id: 5000,
            durations: [0, 1, 2, 3, 4],
            num_durations: 5,
            pred_hidden: 640,
            pred_rnn_layers: 1,
            max_symbols: 10,
            enc_d_model: 1024,
            feat_in: 128,
            subsampling_factor: 8,
        }
    }

    fn frontend() -> FrontendParams {
        FrontendParams {
            sample_rate: 16_000,
            n_fft: 512,
            hop_length: 160,
            win_length: 400,
            n_mels: 128,
            n_freq: 257,
            preemph: 0.97,
            mag_power: 2.0,
            log: true,
            log_zero_guard_value: 5.960_464_5e-08,
            normalize: "per_feature".into(),
            constant: 1e-5,
            pad_value: 0.0,
            blank_id: 5000,
            window_file: String::new(),
            fb_file: String::new(),
        }
    }

    #[test]
    fn logit_width_is_vocab_plus_blank_plus_durations() {
        // 5000 symbols, index 5000 is blank, 5001..5005 are durations.
        assert_eq!(cfg().logit_width(), 5006);
    }

    #[test]
    fn frame_seconds_is_eighty_milliseconds() {
        // 8x subsampling at 10 ms per mel frame. Being 8x out here is the
        // classic transcript-drift bug.
        assert!((cfg().frame_seconds(&frontend()) - 0.08).abs() < 1e-12);
    }

    #[test]
    fn argmax_selects_symbols_and_durations_from_one_vector() {
        let c = cfg();
        let mut v = vec![-10.0f32; 5006];
        v[42] = 5.0; // symbol 42 wins
        v[5004] = 9.0; // duration index 3 wins -> 3 frames
        let (k, skip) = argmax_pair(&v, &c);
        assert_eq!(k, 42);
        assert_eq!(skip, 3);
    }

    #[test]
    fn blank_is_selected_by_its_absolute_index() {
        // blank_id == vocab_size == 5000, and the argmax is that absolute
        // index. Getting this wrong by an off-by-one emits a bogus token every
        // time the model is silent.
        let c = cfg();
        let mut v = vec![-10.0f32; 5006];
        v[5000] = 5.0;
        v[5002] = 1.0; // duration index 1
        let (k, skip) = argmax_pair(&v, &c);
        assert_eq!(k, 5000);
        assert_eq!(skip, 1);
    }

    #[test]
    fn duration_zero_is_selectable_and_means_stay_on_this_frame() {
        let c = cfg();
        let mut v = vec![-10.0f32; 5006];
        v[7] = 5.0;
        v[5001] = 9.0; // duration index 0 == durations[0] == 0
        let (k, skip) = argmax_pair(&v, &c);
        assert_eq!(k, 7);
        assert_eq!(
            skip, 0,
            "duration 0 must be selectable; the inner loop depends on it"
        );
    }

    #[test]
    fn all_negative_logits_still_produce_a_valid_token() {
        // Better an argmax of index 0 than a panic or a NaN.
        let c = cfg();
        let v = vec![-1e9f32; 5006];
        let (k, skip) = argmax_pair(&v, &c);
        assert_eq!(k, 0);
        assert_eq!(skip, 0);
    }

    #[test]
    fn config_rejects_a_blank_id_that_is_not_vocab_size() {
        // The prediction network uses blank-as-padding; a config where the two
        // differ would make blank an out-of-range embedding lookup.
        let dir = tempdir();
        std::fs::write(
            dir.join("config.json"),
            r#"{"vocab_size":5000,"blank_id":4999,"durations":[0,1,2,3,4],
                "num_durations":5,"pred_hidden":640,"pred_rnn_layers":1,
                "max_symbols":10,"enc_d_model":1024,"feat_in":128,
                "subsampling_factor":8}"#,
        )
        .unwrap();
        let e = SraVaaniConfig::load(&dir).unwrap_err();
        assert_eq!(e.code(), vaani_core::ErrorCode::ModelCorrupt);
    }

    #[test]
    fn config_rejects_a_duration_list_of_the_wrong_length() {
        let dir = tempdir();
        std::fs::write(
            dir.join("config.json"),
            r#"{"vocab_size":5000,"blank_id":5000,"durations":[0,1,2],
                "num_durations":5,"pred_hidden":640,"pred_rnn_layers":1,
                "max_symbols":10,"enc_d_model":1024,"feat_in":128,
                "subsampling_factor":8}"#,
        )
        .unwrap();
        assert!(SraVaaniConfig::load(&dir).is_err());
    }

    #[test]
    fn config_loads_the_shipped_artefacts() {
        let dir = default_dir();
        if !dir.join("config.json").is_file() {
            println!("skipping: artifacts/sravaani/config.json not present");
            return;
        }
        let c = SraVaaniConfig::load(&dir).expect("config must load");
        assert_eq!(c.vocab_size, 5000);
        assert_eq!(c.blank_id, 5000);
        assert_eq!(c.durations, [0, 1, 2, 3, 4]);
        assert_eq!(c.enc_d_model, 1024);
        assert_eq!(c.feat_in, 128);
        assert_eq!(c.subsampling_factor, 8);
        assert_eq!(c.max_symbols, 10);
        assert_eq!(c.pred_hidden, 640);
        assert_eq!(c.pred_rnn_layers, 1);
    }

    #[test]
    fn engine_opens_without_loading_weights() {
        let dir = default_dir();
        if !dir.join("tokenizer.model").is_file() {
            println!("skipping: tokenizer.model not present");
            return;
        }
        let a = artifacts_in(&dir);
        let e = SraVaaniEngine::new(&a).expect("engine must open");
        assert!(
            !e.is_loaded(),
            "construction must not load 887 MB of weights"
        );
        assert!((e.frame_seconds() - 0.08).abs() < 1e-12);
        assert_eq!(e.model_name(), "SraVaani 1.0");
        assert!(e.supports_word_timestamps());
        assert!(e.supports_timestamps());
        assert!(!e.supports_streaming());
    }

    #[test]
    fn tokenizer_pieces_carry_the_metaspace_marker() {
        // Word grouping depends on this. If `id_to_piece` did not return the
        // U+2581 prefix, every sub-word piece would become its own word and the
        // transcript would be shredded -- visibly, but only as "too many
        // spaces", which is easy to mistake for a cosmetic problem.
        let dir = default_dir();
        if !dir.join("tokenizer.model").is_file() {
            println!("skipping: tokenizer.model not present");
            return;
        }
        let sp = SentencePieceProcessor::open(dir.join("tokenizer.model")).expect("open");
        let mut with_marker = 0;
        let mut sampled: Vec<(usize, String)> = Vec::new();
        for id in 0..2000usize {
            let piece = sp.id_to_piece(id).unwrap_or_default().to_string();
            if piece.starts_with('\u{2581}') {
                with_marker += 1;
            }
            if id < 12 {
                sampled.push((id, piece));
            }
        }
        for (id, piece) in &sampled {
            println!("  {id:>4} {:?}", piece);
        }
        println!("  pieces with the metaspace marker in 0..2000: {with_marker}");
        assert!(
            with_marker > 100,
            "only {with_marker} of 2000 pieces start with U+2581; the tokenizer is \
             not the expected unigram model, or id_to_piece is decoding the marker"
        );
    }

    fn tempdir() -> PathBuf {
        let d = std::env::temp_dir().join(format!("vaani-asr-{}", std::process::id()));
        std::fs::create_dir_all(&d).expect("temp dir");
        d
    }
}

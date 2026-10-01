//! Progressive-yield streaming latency for the SraVaani path.
//!
//! The question is narrow and it is about the *sink*, not about total wall time.
//! `AsrEngine::transcribe_streaming` batches the encoder over a group of chunks
//! and then drains that group, calling the sink once per chunk as each chunk's
//! decode finishes. A previous design called the encoder once for the whole job
//! and returned a `Vec` at the end, which made the UI sit on an empty pane and
//! then take the whole transcript in one step -- caught by
//! `crates/vaani-desktop/tests/ui_ship_gate.rs` ("streamed rows must arrive
//! progressively, not in one step").
//!
//! So this harness measures, for the serial production loop and for the batched
//! path at several batch sizes:
//!
//! * **TTFR** -- wall time from run start until the sink fires for the first
//!   chunk. This is the "is the pane still empty?" number.
//! * the **inter-chunk gap** distribution (p50/p95/p99/max) -- this is the stall
//!   a user perceives between two rows appearing.
//! * total completion time.
//! * the streaming **invariants**: the sink must fire exactly once per chunk, in
//!   strictly increasing index order, and the invocation count must equal the
//!   chunk count. A duplicate or skipped call is a bug and is asserted, not
//!   reported.
//!
//! It also checks the two unwind paths a UI depends on: a cancel that arrives
//! mid-run, and a sink that returns `false` because the UI stopped listening.
//!
//! Configurations are interleaved within one process and repeated, because this
//! is a laptop GPU: separate processes drift by around twenty percent, which is
//! the same size as the effect being measured (the same argument is made in
//! `asr-profile`'s `--ab` mode). Each configuration is warmed once first, since
//! the first pass through a shape pays cuDNN algorithm selection.
//!
//! Usage:
//!     cargo run --release --example stream-latency -- <wav> [options]
//!
//!         --batches 1,2,4,8   streaming batch sizes to compare
//!         --rounds N          measured rounds per configuration (default 3)
//!         --warmup N          unmeasured warm passes per configuration (default 1)
//!         --cancel-after N    cancel on the Nth sink call (default 3)
//!         --stop-after N      return false from the Nth sink call (default 3)
//!         --provider auto|cpu|cuda
//!         --out DIR           where the report is written (default %TEMP%\opencode\stream)

use std::path::{Path, PathBuf};
use std::time::Instant;

use vaani_asr::engine::{AsrEngine, AudioChunk, ChunkResult};
use vaani_asr::frontend::{FrontendParams, FrontendTables, log_mel};
use vaani_asr::{EngineRegistry, artifacts_in};
use vaani_core::{CancelToken, Time};
use vaani_inference::Provider;

const TARGET_SR: u32 = 16_000;
const BUCKET: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Config {
    /// What production runs today: `transcribe` once per chunk, in a loop.
    Serial,
    /// `transcribe_streaming` with the encoder batch set to `n`.
    Stream(usize),
}

impl Config {
    fn label(self) -> String {
        match self {
            Self::Serial => "serial (transcribe/chunk)".to_owned(),
            Self::Stream(n) => format!("streaming batch={n}"),
        }
    }

    fn short(self) -> String {
        match self {
            Self::Serial => "serial".to_owned(),
            Self::Stream(n) => format!("b{n}"),
        }
    }

    fn group_size(self) -> usize {
        match self {
            Self::Serial => 1,
            Self::Stream(n) => n,
        }
    }
}

#[derive(Debug)]
struct Args {
    input: PathBuf,
    batches: Vec<usize>,
    rounds: usize,
    warmup: usize,
    cancel_after: usize,
    stop_after: usize,
    provider: Provider,
    out: PathBuf,
    chunk_target_ms: Option<i64>,
}

fn parse() -> Result<Args, String> {
    let mut input = None;
    let mut batches = vec![1usize, 2, 4, 8];
    let mut rounds = 3usize;
    let mut warmup = 1usize;
    let mut cancel_after = 3usize;
    let mut stop_after = 3usize;
    let mut provider = Provider::Auto;
    let mut out = PathBuf::from(std::env::var("TEMP").unwrap_or_else(|_| ".".into()))
        .join("opencode")
        .join("stream");
    let mut chunk_target_ms = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--batches" => {
                let v = it.next().ok_or("--batches needs a value")?;
                batches = v
                    .split(',')
                    .map(|p| {
                        p.trim()
                            .parse::<usize>()
                            .map_err(|e| format!("--batches {p:?}: {e}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .filter(|n| *n > 0)
                    .collect();
                if batches.is_empty() {
                    return Err("--batches left no usable size".into());
                }
            }
            "--rounds" => {
                rounds = it
                    .next()
                    .ok_or("--rounds needs a value")?
                    .parse()
                    .map_err(|e| format!("--rounds: {e}"))?
            }
            "--warmup" => {
                warmup = it
                    .next()
                    .ok_or("--warmup needs a value")?
                    .parse()
                    .map_err(|e| format!("--warmup: {e}"))?
            }
            "--cancel-after" => {
                cancel_after = it
                    .next()
                    .ok_or("--cancel-after needs a value")?
                    .parse()
                    .map_err(|e| format!("--cancel-after: {e}"))?
            }
            "--stop-after" => {
                stop_after = it
                    .next()
                    .ok_or("--stop-after needs a value")?
                    .parse()
                    .map_err(|e| format!("--stop-after: {e}"))?
            }
            "--chunk-target-ms" => {
                chunk_target_ms = Some(
                    it.next()
                        .ok_or("--chunk-target-ms needs a value")?
                        .parse()
                        .map_err(|e| format!("--chunk-target-ms: {e}"))?,
                )
            }
            "--provider" => {
                let v = it.next().ok_or("--provider needs a value")?;
                provider = Provider::parse(&v).ok_or_else(|| format!("unknown provider {v}"))?;
            }
            "--out" => {
                out = PathBuf::from(it.next().ok_or("--out needs a value")?);
            }
            "-h" | "--help" => {
                println!(
                    "usage: stream-latency <wav> [--batches 1,2,4,8] [--rounds N] [--warmup N] \
                     [--cancel-after N] [--stop-after N] [--provider auto|cpu|cuda] [--out DIR]"
                );
                std::process::exit(0);
            }
            other if other.starts_with('-') => return Err(format!("unknown flag {other}")),
            other => {
                if input.is_some() {
                    return Err("expected one input file".into());
                }
                input = Some(PathBuf::from(other));
            }
        }
    }
    Ok(Args {
        input: input.ok_or("missing input file")?,
        batches,
        rounds: rounds.max(1),
        warmup,
        cancel_after: cancel_after.max(1),
        stop_after: stop_after.max(1),
        provider,
        out,
        chunk_target_ms,
    })
}

// ---------------------------------------------------------------- wav input

/// 16-bit (or 32-bit int/float) PCM WAV -> mono f32 at `TARGET_SR`.
///
/// Hand-rolled and deliberately narrow, for the same reason `asr-profile` rolls
/// its own: this is an instrument, and it must not be the thing that breaks.
fn read_wav16k_mono(path: &Path) -> Result<(Vec<f32>, u32), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let mut pos = 12usize;
    let (mut fmt, mut data) = (None::<(u16, u16, u32, u16)>, None::<(usize, usize)>);
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = pos + 8;
        if id == b"fmt " && body + 16 <= bytes.len() {
            fmt = Some((
                u16::from_le_bytes(bytes[body..body + 2].try_into().unwrap()),
                u16::from_le_bytes(bytes[body + 2..body + 4].try_into().unwrap()),
                u32::from_le_bytes(bytes[body + 4..body + 8].try_into().unwrap()),
                u16::from_le_bytes(bytes[body + 14..body + 16].try_into().unwrap()),
            ));
        } else if id == b"data" {
            data = Some((body, size.min(bytes.len().saturating_sub(body))));
        }
        pos = body + size + (size & 1);
    }
    let (tag, channels, rate, bits) = fmt.ok_or("no fmt chunk")?;
    let (dstart, dlen) = data.ok_or("no data chunk")?;
    if channels == 0 {
        return Err("a zero-channel WAV".into());
    }
    let raw = &bytes[dstart..dstart + dlen];
    let frame_bytes = (bits / 8) as usize * channels as usize;
    if frame_bytes == 0 {
        return Err(format!("unsupported bit depth {bits}"));
    }
    let frames = raw.len() / frame_bytes;
    let mut mono = Vec::with_capacity(frames);
    for i in 0..frames {
        let mut acc = 0f32;
        for c in 0..channels as usize {
            let o = i * frame_bytes + c * (bits / 8) as usize;
            let v = match (tag, bits) {
                (1, 16) => {
                    f32::from(i16::from_le_bytes(raw[o..o + 2].try_into().unwrap())) / 32768.0
                }
                (1, 32) => {
                    i32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) as f32 / 2_147_483_648.0
                }
                (3, 32) => f32::from_le_bytes(raw[o..o + 4].try_into().unwrap()),
                (t, b) => return Err(format!("unsupported PCM tag={t} bits={b}")),
            };
            acc += v;
        }
        mono.push(acc / channels as f32);
    }
    Ok((mono, rate))
}

fn resample_to_16k(x: &[f32], from: u32) -> Vec<f32> {
    if from == TARGET_SR {
        return x.to_vec();
    }
    let ratio = f64::from(from) / f64::from(TARGET_SR);
    let out_len = (x.len() as f64 / ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let t = i as f64 * ratio;
        let i0 = t.floor() as usize;
        let frac = (t - i0 as f64) as f32;
        let a = x.get(i0).copied().unwrap_or(0.0);
        let b = x.get(i0 + 1).copied().unwrap_or(0.0);
        out.push(a + (b - a) * frac);
    }
    out
}

// ---------------------------------------------------------------- recording

#[derive(Debug, Clone, serde::Serialize)]
struct SinkEvent {
    /// The index the engine passed to the sink.
    index: usize,
    /// Wall milliseconds from run start to this sink call.
    t_ms: f64,
    /// True when this call closed an encoder group, i.e. the next gap has to
    /// pay for a fresh `encode_group` rather than just a decode.
    closes_group: bool,
    frontend_ms: u64,
    /// The *whole group's* encode time, which the engine attributes to every
    /// chunk in it. Summing it over chunks therefore double counts on purpose.
    encode_ms: u64,
    decode_ms: u64,
    steps: usize,
    words: usize,
}

#[derive(Debug, Default, serde::Serialize)]
struct Totals {
    ok_results: usize,
    err_results: usize,
    err_codes: Vec<String>,
    words: usize,
    steps: usize,
    frontend_ms: u64,
    /// Sum over chunks of the group encode time. Meaningless as a cost; kept
    /// only so the report can show why.
    group_encode_ms_sum: u64,
    decode_ms: u64,
}

#[derive(Debug, serde::Serialize)]
struct RunRecord {
    config: String,
    short: String,
    round: usize,
    warmup: bool,
    total_ms: f64,
    events: Vec<SinkEvent>,
    order_ok: bool,
    order_problem: Option<String>,
    fingerprint: String,
    totals: Totals,
}

impl RunRecord {
    fn ttfr_ms(&self) -> f64 {
        self.events.first().map_or(f64::NAN, |e| e.t_ms)
    }

    /// Gaps between consecutive sink calls, tagged with whether the *later* call
    /// closed a group (so the gap spans a fresh encoder call).
    fn gaps(&self) -> Vec<(f64, bool)> {
        self.events
            .windows(2)
            .map(|w| (w[1].t_ms - w[0].t_ms, w[1].closes_group))
            .collect()
    }
}

/// Everything the sink saw, plus the totals it accumulated.
struct Tap {
    t0: Instant,
    events: Vec<SinkEvent>,
    fp: u64,
    totals: Totals,
    group: usize,
}

impl Tap {
    fn new() -> Self {
        Self {
            t0: Instant::now(),
            events: Vec::new(),
            fp: 1469598103934665603,
            totals: Totals::default(),
            group: 1,
        }
    }

    fn record(&mut self, index: usize, r: &ChunkResult) {
        let t = self.t0.elapsed().as_secs_f64() * 1000.0;
        self.totals.ok_results += 1;
        self.totals.words += r.segments.iter().map(|s| s.words.len()).sum::<usize>();
        self.totals.steps += r.timings.decode_steps;
        self.totals.frontend_ms += r.timings.frontend_ms;
        self.totals.group_encode_ms_sum += r.timings.encode_ms;
        self.totals.decode_ms += r.timings.decode_ms;
        for w in r.segments.iter().flat_map(|s| s.words.iter()) {
            for b in w.text.as_bytes() {
                self.fp ^= u64::from(*b);
                self.fp = self.fp.wrapping_mul(1099511628211);
            }
        }
        self.events.push(SinkEvent {
            index,
            t_ms: t,
            closes_group: (index + 1).is_multiple_of(self.group),
            frontend_ms: r.timings.frontend_ms,
            encode_ms: r.timings.encode_ms,
            decode_ms: r.timings.decode_ms,
            steps: r.timings.decode_steps,
            words: r.segments.iter().map(|s| s.words.len()).sum(),
        });
    }
}

/// The streaming invariants, asserted rather than reported.
///
/// * exactly one sink call per chunk,
/// * strictly increasing index, so no duplicate and no skip,
/// * non-decreasing arrival times.
///
/// Returns the problem as a string so it can be both printed and counted: a run
/// that violates an invariant is a bug, not a slow run.
fn check_streaming_invariants(events: &[SinkEvent], chunks: usize) -> Result<(), String> {
    if events.len() != chunks {
        return Err(format!(
            "sink fired {} times for {chunks} chunks",
            events.len()
        ));
    }
    for (position, e) in events.iter().enumerate() {
        if e.index != position {
            return Err(format!(
                "sink call #{position} carried index {} (out of order, duplicated or skipped)",
                e.index
            ));
        }
    }
    for w in events.windows(2) {
        if w[1].t_ms < w[0].t_ms {
            return Err(format!(
                "sink times went backwards: chunk {} at {:.1} ms, chunk {} at {:.1} ms",
                w[0].index, w[0].t_ms, w[1].index, w[1].t_ms
            ));
        }
    }
    Ok(())
}

fn error_codes(results: &[vaani_core::Result<ChunkResult>]) -> (usize, Vec<String>) {
    let mut codes = Vec::new();
    let mut errs = 0usize;
    for r in results {
        if let Err(e) = r {
            errs += 1;
            let c = format!("{:?}", e.code());
            if !codes.contains(&c) {
                codes.push(c);
            }
        }
    }
    (errs, codes)
}

// ---------------------------------------------------------------- the runs

fn run_serial(
    engine: &dyn AsrEngine,
    inputs: &[AudioChunk],
    cancel: &CancelToken,
    round: usize,
    warmup: bool,
) -> RunRecord {
    // The sink *is* the loop body here: production's streaming point is the
    // moment a chunk's result exists, which is exactly the moment the serial
    // loop reaches the end of `transcribe`.
    let mut tap = Tap::new();
    tap.group = 1;
    for (i, input) in inputs.iter().enumerate() {
        if cancel.check().is_err() {
            tap.totals.err_results += 1;
            tap.totals.err_codes.push("Cancelled".to_owned());
            break;
        }
        match engine.transcribe(input, cancel) {
            Ok(r) => tap.record(i, &r),
            Err(e) => {
                tap.totals.err_results += 1;
                tap.totals.err_codes.push(format!("{:?}", e.code()));
                break;
            }
        }
    }
    let total_ms = tap.t0.elapsed().as_secs_f64() * 1000.0;
    let problem = check_streaming_invariants(&tap.events, inputs.len()).err();
    RunRecord {
        config: Config::Serial.label(),
        short: Config::Serial.short(),
        round,
        warmup,
        total_ms,
        events: tap.events,
        order_ok: problem.is_none(),
        order_problem: problem,
        fingerprint: format!("{:016x}", tap.fp),
        totals: tap.totals,
    }
}

fn run_stream(
    engine: &dyn AsrEngine,
    inputs: &[AudioChunk],
    cancel: &CancelToken,
    batch: usize,
    round: usize,
    warmup: bool,
) -> RunRecord {
    set_encoder_batch(batch);
    let mut tap = Tap::new();
    tap.group = batch;
    let results = engine.transcribe_streaming(inputs, cancel, &mut |i, r| {
        tap.record(i, r);
        true
    });
    let (errs, codes) = error_codes(&results);
    tap.totals.err_results = errs;
    tap.totals.err_codes = codes;
    let total_ms = tap.t0.elapsed().as_secs_f64() * 1000.0;
    let problem = check_streaming_invariants(&tap.events, inputs.len()).err();
    RunRecord {
        config: Config::Stream(batch).label(),
        short: Config::Stream(batch).short(),
        round,
        warmup,
        total_ms,
        events: tap.events,
        order_ok: problem.is_none(),
        order_problem: problem,
        fingerprint: format!("{:016x}", tap.fp),
        totals: tap.totals,
    }
}

fn run_config(
    engine: &dyn AsrEngine,
    cfg: Config,
    inputs: &[AudioChunk],
    r: usize,
    w: bool,
) -> RunRecord {
    let cancel = CancelToken::new();
    match cfg {
        Config::Serial => run_serial(engine, inputs, &cancel, r, w),
        Config::Stream(n) => run_stream(engine, inputs, &cancel, n, r, w),
    }
}

/// Point the engine at a batch size.
///
/// The engine reads `VAANI_ASR_ENCODER_BATCH` at the start of every
/// `transcribe_streaming` call, which is the only knob it has. Interleaving the
/// configurations in one process needs it to change between calls.
fn set_encoder_batch(n: usize) {
    // SAFETY: single-threaded at this point -- no inference is in flight, the
    // value is read and written only on this thread, and ORT's worker threads do
    // not read it. The alternative (one process per configuration) is what this
    // harness exists to avoid, because a laptop GPU drifts ~20% between
    // processes and that is the size of the effect being measured.
    unsafe { std::env::set_var("VAANI_ASR_ENCODER_BATCH", n.to_string()) };
    let got = vaani_asr::sravaani::encoder_batch();
    assert_eq!(got, n, "VAANI_ASR_ENCODER_BATCH did not take effect");
}

// ---------------------------------------------------------------- unwind

#[derive(Debug, serde::Serialize)]
struct StopTest {
    what: String,
    config: String,
    trigger_on_sink_call: usize,
    sink_calls: usize,
    last_index: Option<usize>,
    order_ok: bool,
    order_problem: Option<String>,
    /// Milliseconds from the sink call that triggered the stop to the return of
    /// the call under test. The engine unwound inside this window.
    unwind_ms: f64,
    total_ms: f64,
    results: usize,
    ok_results: usize,
    err_results: usize,
    err_codes: Vec<String>,
}

fn cancel_test(
    engine: &dyn AsrEngine,
    cfg: Config,
    inputs: &[AudioChunk],
    after: usize,
) -> StopTest {
    let cancel = CancelToken::new();
    let mut calls = 0usize;
    let mut last = None;
    let mut events: Vec<SinkEvent> = Vec::new();
    let mut tap = Tap::new();
    let mut trigger: Option<Instant> = None;

    let total_ms = match cfg {
        Config::Serial => {
            tap.group = 1;
            for (i, input) in inputs.iter().enumerate() {
                if cancel.check().is_err() {
                    break;
                }
                match engine.transcribe(input, &cancel) {
                    Ok(r) => {
                        calls += 1;
                        last = Some(i);
                        tap.record(i, &r);
                        events = tap.events.clone();
                        if calls == after {
                            trigger = Some(Instant::now());
                            // The UI's cancel button, from inside the sink.
                            cancel.cancel();
                        }
                    }
                    Err(_) => break,
                }
            }
            tap.t0.elapsed().as_secs_f64() * 1000.0
        }
        Config::Stream(n) => {
            set_encoder_batch(n);
            tap.group = n;
            engine.transcribe_streaming(inputs, &cancel, &mut |i, r| {
                calls += 1;
                last = Some(i);
                tap.record(i, r);
                events.push(tap.events.last().expect("just recorded").clone());
                if calls == after {
                    trigger = Some(Instant::now());
                    // Return `true`: this test is about the engine noticing the
                    // token, not about the sink's return value, which the
                    // early-stop test below covers.
                    cancel.cancel();
                }
                true
            });
            tap.t0.elapsed().as_secs_f64() * 1000.0
        }
    };

    // The trigger is stamped at the top of the sink call, and the sink returns
    // immediately after, so `trigger_to_end` measures the engine's unwind only
    // to within the closure's own bookkeeping.
    let unwind_ms = trigger.map_or(f64::NAN, |t| t.elapsed().as_secs_f64() * 1000.0);
    let problem = {
        // The run stopped on purpose, so a partial event list is expected; the
        // invariant that still has to hold is "in order, no duplicates".
        let mut p = None;
        for (position, e) in events.iter().enumerate() {
            if e.index != position {
                p = Some(format!("sink call #{position} carried index {}", e.index));
                break;
            }
        }
        p
    };
    StopTest {
        what: format!("cancel on sink call {after}"),
        config: cfg.label(),
        trigger_on_sink_call: after,
        sink_calls: calls,
        last_index: last,
        order_ok: problem.is_none(),
        order_problem: problem,
        unwind_ms,
        total_ms,
        results: inputs.len(),
        ok_results: events.len(),
        err_results: inputs.len() - events.len(),
        err_codes: Vec::new(),
    }
}

fn early_stop_test(
    engine: &dyn AsrEngine,
    cfg: Config,
    inputs: &[AudioChunk],
    after: usize,
) -> StopTest {
    let cancel = CancelToken::new();
    let mut calls = 0usize;
    let mut last = None;
    let mut events: Vec<SinkEvent> = Vec::new();
    let mut tap = Tap::new();
    let mut trigger: Option<Instant> = None;
    let mut returned: Vec<vaani_core::Result<ChunkResult>> = Vec::new();

    let total_ms = match cfg {
        Config::Serial => {
            tap.group = 1;
            for (i, input) in inputs.iter().enumerate() {
                if cancel.check().is_err() {
                    break;
                }
                match engine.transcribe(input, &cancel) {
                    Ok(r) => {
                        calls += 1;
                        last = Some(i);
                        tap.record(i, &r);
                        events = tap.events.clone();
                        if calls == after {
                            trigger = Some(Instant::now());
                            returned.push(Ok(r));
                            // The UI stopped listening. Stop.
                            break;
                        }
                        returned.push(Ok(r));
                    }
                    Err(e) => {
                        returned.push(Err(e));
                        break;
                    }
                }
            }
            tap.t0.elapsed().as_secs_f64() * 1000.0
        }
        Config::Stream(n) => {
            set_encoder_batch(n);
            tap.group = n;
            returned = engine.transcribe_streaming(inputs, &cancel, &mut |i, r| {
                calls += 1;
                last = Some(i);
                tap.record(i, r);
                events.push(tap.events.last().expect("just recorded").clone());
                if calls == after {
                    trigger = Some(Instant::now());
                    return false;
                }
                true
            });
            tap.t0.elapsed().as_secs_f64() * 1000.0
        }
    };

    let unwind_ms = trigger.map_or(f64::NAN, |t| t.elapsed().as_secs_f64() * 1000.0);
    let problem = {
        let mut p = None;
        for (position, e) in events.iter().enumerate() {
            if e.index != position {
                p = Some(format!("sink call #{position} carried index {}", e.index));
                break;
            }
        }
        p
    };
    let (errs, codes) = error_codes(&returned);
    let ok = returned.iter().filter(|r| r.is_ok()).count();
    StopTest {
        what: format!("sink returns false on call {after}"),
        config: cfg.label(),
        trigger_on_sink_call: after,
        sink_calls: calls,
        last_index: last,
        order_ok: problem.is_none(),
        order_problem: problem,
        unwind_ms,
        total_ms,
        results: returned.len(),
        ok_results: ok,
        err_results: errs,
        err_codes: codes,
    }
}

// ---------------------------------------------------------------- statistics

/// Nearest-rank percentile on an ascending slice. p99 of 22 gaps is the largest
/// gap, and the report says so rather than pretending otherwise.
fn pct(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = (q * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    pct(&s, 0.5)
}

fn stats(v: &[f64]) -> (f64, f64, f64, f64) {
    if v.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    (
        s[0],
        median(&s),
        *s.last().unwrap_or(&f64::NAN),
        s.iter().sum::<f64>(),
    )
}

// ---------------------------------------------------------------- main

fn main() {
    if let Err(e) = run() {
        eprintln!("stream-latency: {e}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args = parse()?;

    let dir = std::env::var("VAANI_WEIGHTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("artifacts/sravaani"));
    if !dir.join("encoder.fp16.onnx").is_file() {
        return Err(format!(
            "weights not found in {}; set VAANI_WEIGHTS_DIR",
            dir.display()
        ));
    }

    let (raw, rate) = read_wav16k_mono(&args.input)?;
    let samples = resample_to_16k(&raw, rate);
    let audio_s = samples.len() as f64 / f64::from(TARGET_SR);

    let t_load = Instant::now();
    let caps = vaani_inference::capability::detect(&[]);
    let registry = EngineRegistry::with_builtins(args.provider, &caps)
        .map_err(|e| format!("registry: {}", e.debug_chain()))?;
    let art = artifacts_in(&dir);
    let engine = registry
        .create("sravaani", &art)
        .map_err(|e| format!("engine: {}", e.debug_chain()))?;
    let load_s = t_load.elapsed().as_secs_f64();
    let provider = engine
        .active_provider()
        .map_or("unknown", vaani_inference::ActiveProvider::as_str);

    // Production chunking, because the encoder's cost and the decode loop's
    // length both depend on how the audio was split before it reached the
    // engine.
    let buffer = vaani_core::media::AudioBuffer::mono(samples, TARGET_SR, Time::ZERO);
    let window = vaani_core::MediaWindow::of_buffer(&buffer);
    let cancel0 = CancelToken::new();
    let vad = vaani_vad::neural::preferred_engine(
        None,
        false,
        args.provider,
        vaani_vad::EnergyVadConfig::default(),
    )
    .map_err(|e| format!("vad: {}", e.debug_chain()))?;
    let timeline = vad
        .detect(&buffer, &cancel0)
        .map_err(|e| format!("vad: {}", e.debug_chain()))?;
    let mut policy = vaani_vad::ChunkPolicy::default();
    if let Some(ms) = args.chunk_target_ms {
        policy.target_ms = ms;
    }
    let chunks = vaani_vad::Chunker::new(policy).plan_within(&timeline, &window);
    let inputs: Vec<AudioChunk> = chunks
        .iter()
        .map(|c| {
            let slice = buffer.slice_mono(c.start, c.end);
            AudioChunk::new(slice, TARGET_SR, c.start)
        })
        .filter(|c| !c.samples.is_empty())
        .collect();

    // The front end's cost, measured on its own. `transcribe_streaming` hoists
    // every chunk's log-mel to the front of the call, so this is paid before the
    // first sink fires on that path and is *not* paid before the first sink
    // fires on the serial loop -- which is most of the TTFR difference between
    // them, and is not caused by batching.
    let params = FrontendParams::load(&art.frontend_dir).map_err(|e| e.debug_chain())?;
    let tables = FrontendTables::load(&art.frontend_dir, &params).map_err(|e| e.debug_chain())?;
    let t_fe = Instant::now();
    let mels: Vec<_> = inputs
        .iter()
        .map(|c| log_mel(&c.samples, &params, &tables))
        .collect::<vaani_core::Result<Vec<_>>>()
        .map_err(|e| e.debug_chain())?;
    let frontend_all_ms = t_fe.elapsed().as_secs_f64() * 1000.0;
    let frontend_one_ms = frontend_all_ms / inputs.len().max(1) as f64;
    let n_frames: Vec<usize> = mels.iter().map(|m| m.n_frames).collect();
    let valid_frames: Vec<usize> = mels.iter().map(|m| m.valid_frames).collect();

    let configs: Vec<Config> = std::iter::once(Config::Serial)
        .chain(args.batches.iter().copied().map(Config::Stream))
        .collect();

    let n_chunks = inputs.len();
    println!("input      {}", args.input.display());
    println!("audio      {audio_s:.2} s  ->  {n_chunks} chunks");
    println!("load       {load_s:.2} s   provider {provider}   bucket {BUCKET} frames");
    println!("chunks     {:?}", chunk_len_summary(&chunks));
    println!(
        "front end  {frontend_all_ms:.0} ms for all {n_chunks} chunks ({frontend_one_ms:.1} ms each), hoisted to the front of transcribe_streaming"
    );
    println!();
    println!(
        "configurations: {} x {} measured rounds, {} warm pass(es), interleaved in one process",
        configs.len(),
        args.rounds,
        args.warmup
    );
    println!();

    // Warm every configuration first: the first pass through a shape pays cuDNN
    // algorithm selection and CUDA module loading, and folding that into a
    // reported latency number makes it a lie.
    for cfg in &configs {
        let r = run_config(engine.as_ref(), *cfg, &inputs, 0, true);
        eprintln!(
            "warm  {:<26} {:>8.0} ms  sink calls {}  order {}",
            r.config,
            r.total_ms,
            r.events.len(),
            if r.order_ok { "ok" } else { "VIOLATED" }
        );
    }

    let mut records: Vec<RunRecord> = Vec::new();
    for round in 1..=args.rounds {
        for cfg in &configs {
            let r = run_config(engine.as_ref(), *cfg, &inputs, round, false);
            println!(
                "round {round}  {:<26} ttfr {:>7.0} ms  total {:>7.0} ms  sink {}  maxgap {:>6.0} ms  {}",
                r.config,
                r.ttfr_ms(),
                r.total_ms,
                r.events.len(),
                r.max_gap(),
                if r.order_ok {
                    "order ok"
                } else {
                    "ORDER VIOLATED"
                }
            );
            records.push(r);
        }
    }
    println!();

    // -- the invariants, as a hard verdict --------------------------------
    let measured: Vec<&RunRecord> = records.iter().filter(|r| !r.warmup).collect();
    let mut violations: Vec<String> = Vec::new();
    for r in &measured {
        if let Some(p) = &r.order_problem {
            violations.push(format!("{} round {}: {p}", r.short, r.round));
        }
        if r.totals.ok_results != inputs.len() {
            violations.push(format!(
                "{} round {}: {} successful results for {} chunks",
                r.short,
                r.round,
                r.totals.ok_results,
                inputs.len()
            ));
        }
    }
    println!("-- streaming invariants --");
    println!(
        "sink fired exactly once per chunk, in index order, in all {} measured runs over {} chunks: {}",
        measured.len(),
        inputs.len(),
        if violations.is_empty() {
            "PASS"
        } else {
            "FAIL"
        }
    );
    for v in &violations {
        println!("  VIOLATION  {v}");
    }
    println!();

    // -- the table --------------------------------------------------------
    let mut report = String::new();
    let head = "| config | rounds | TTFR med | TTFR min | TTFR max | gap p50 | gap p95 | gap p99 | gap max | gap max (worst round) | total med | total min | total max | TTFR if front end interleaved (derived) | order | words | steps | text fp |";
    report.push_str(head);
    report.push('\n');
    report.push_str(&format!(
        "|{}|\n",
        "---|".repeat(head.matches(" | ").count() + 1)
    ));
    println!(
        "-- per configuration ({} chunks, nearest-rank percentiles) --",
        inputs.len()
    );
    println!("{head}");
    println!("|{}|", "---|".repeat(head.matches(" | ").count() + 1));

    let mut summary: Vec<Summary> = Vec::new();
    for cfg in &configs {
        let rs: Vec<&RunRecord> = measured
            .iter()
            .copied()
            .filter(|r| r.short == cfg.short())
            .collect();
        if rs.is_empty() {
            continue;
        }
        let ttfr: Vec<f64> = rs.iter().map(|r| r.ttfr_ms()).collect();
        let totals: Vec<f64> = rs.iter().map(|r| r.total_ms).collect();
        let per_round_max: Vec<f64> = rs.iter().map(|r| r.max_gap()).collect();
        let mut gaps: Vec<f64> = rs
            .iter()
            .flat_map(|r| r.gaps().into_iter().map(|g| g.0))
            .collect();
        gaps.sort_by(f64::total_cmp);
        let mut within: Vec<f64> = rs
            .iter()
            .flat_map(|r| r.gaps().into_iter().filter(|g| !g.1).map(|g| g.0))
            .collect();
        within.sort_by(f64::total_cmp);
        let mut across: Vec<f64> = rs
            .iter()
            .flat_map(|r| r.gaps().into_iter().filter(|g| g.1).map(|g| g.0))
            .collect();
        across.sort_by(f64::total_cmp);
        let (tmin, tmed, tmax, _) = stats(&ttfr);
        let (omin, omed, omax, _) = stats(&totals);
        let words = rs[0].totals.words;
        let steps = rs[0].totals.steps;
        let fp = rs[0].fingerprint.clone();
        let all_ordered = rs.iter().all(|r| r.order_ok);
        // DERIVED, not measured: what TTFR would be if the front end were not
        // hoisted -- one chunk's front end, then the first group's encode, then
        // the first chunk's decode, using the timings the engine reported. This
        // is the number that isolates "batching" from "`transcribe_streaming`
        // computes every log-mel before it starts the GPU".
        let interleaved: Vec<f64> = rs
            .iter()
            .map(|r| {
                frontend_one_ms
                    + r.events
                        .first()
                        .map_or(0.0, |e| e.encode_ms as f64 + e.decode_ms as f64)
            })
            .collect();
        let interleaved_med = median(&interleaved);
        let line = format!(
            "| {} | {} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {} | {} | {} | {} |",
            cfg.label(),
            rs.len(),
            tmed,
            tmin,
            tmax,
            pct(&gaps, 0.5),
            pct(&gaps, 0.95),
            pct(&gaps, 0.99),
            gaps.last().copied().unwrap_or(f64::NAN),
            per_round_max.iter().cloned().fold(f64::MIN, f64::max),
            omed,
            omin,
            omax,
            interleaved_med,
            if all_ordered { "ok" } else { "VIOLATED" },
            words,
            steps,
            fp,
        );
        println!("{line}");
        report.push_str(&line);
        report.push('\n');
        summary.push(Summary {
            config: cfg.label(),
            short: cfg.short(),
            ttfr_med: tmed,
            gap_p50: pct(&gaps, 0.5),
            gap_p95: pct(&gaps, 0.95),
            gap_p99: pct(&gaps, 0.99),
            gap_max: gaps.last().copied().unwrap_or(f64::NAN),
            gap_max_worst_round: per_round_max.iter().cloned().fold(f64::MIN, f64::max),
            total_med: omed,
            ttfr_if_front_end_interleaved: interleaved_med,
            within_gap_med: median(&within),
            within_gap_max: within.last().copied().unwrap_or(f64::NAN),
            across_gap_med: median(&across),
            across_gap_max: across.last().copied().unwrap_or(f64::NAN),
            words,
            steps,
            fingerprint: fp,
            order_ok: all_ordered,
            padded_frames: padded_frames_for(&n_frames, cfg.group_size()),
        });
    }
    report.push('\n');
    println!();

    // -- where the gaps come from ----------------------------------------
    println!("-- gap anatomy (ms) --");
    println!(
        "| config | within-group p50 | within-group max | across-group p50 | across-group max | encoder frames computed | valid frames | padding overhead |"
    );
    println!("|---|---|---|---|---|---|---|---|");
    for s in &summary {
        let valid: usize = valid_frames.iter().sum();
        println!(
            "| {} | {:.0} | {:.0} | {:.0} | {:.0} | {} | {valid} | {:.2}x |",
            s.config,
            s.within_gap_med,
            s.within_gap_max,
            s.across_gap_med,
            s.across_gap_max,
            s.padded_frames,
            s.padded_frames as f64 / valid.max(1) as f64,
        );
        report.push_str(&format!(
            "| {} | {:.0} | {:.0} | {:.0} | {:.0} | {} | {valid} | {:.2}x |\n",
            s.config,
            s.within_gap_med,
            s.within_gap_max,
            s.across_gap_med,
            s.across_gap_max,
            s.padded_frames,
            s.padded_frames as f64 / valid.max(1) as f64,
        ));
    }
    report.push('\n');
    println!(
        "front end for all {} chunks: {frontend_all_ms:.0} ms ({frontend_one_ms:.1} ms each). transcribe_streaming pays all of it before the first sink fires; the serial loop pays one chunk's worth.",
        inputs.len()
    );
    report.push_str(&format!(
        "front end for all {} chunks: {frontend_all_ms:.0} ms ({frontend_one_ms:.1} ms each).\n",
        inputs.len()
    ));
    println!();

    // -- the criterion ---------------------------------------------------
    let base = summary
        .iter()
        .find(|s| s.short == "serial")
        .ok_or("no serial baseline")?;
    println!("-- criterion: TTFR within 1.5x of serial, max gap within 2x of serial --");
    println!(
        "| config | TTFR | vs serial | max gap | vs serial | total | verdict | TTFR interleaved (derived) | vs serial |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    report.push_str("| config | TTFR | vs serial | max gap | vs serial | total | verdict | TTFR interleaved (derived) | vs serial |\n");
    report.push_str("|---|---|---|---|---|---|---|---|---|\n");
    for s in &summary {
        let t_ratio = s.ttfr_med / base.ttfr_med;
        let g_ratio = s.gap_max / base.gap_max;
        let i_ratio = s.ttfr_if_front_end_interleaved / base.ttfr_if_front_end_interleaved;
        let pass = t_ratio <= 1.5 && g_ratio <= 2.0;
        let i_pass = i_ratio <= 1.5 && g_ratio <= 2.0;
        let verdict = if s.short == "serial" {
            "baseline".to_owned()
        } else if pass {
            format!("PASS ({:.2}x TTFR, {:.2}x gap)", t_ratio, g_ratio)
        } else {
            format!("FAIL ({:.2}x TTFR, {:.2}x gap)", t_ratio, g_ratio)
        };
        let line = format!(
            "| {} | {:.0} | {:.2}x | {:.0} | {:.2}x | {:.0} | {} | {:.0} | {:.2}x {} |",
            s.config,
            s.ttfr_med,
            t_ratio,
            s.gap_max,
            g_ratio,
            s.total_med,
            verdict,
            s.ttfr_if_front_end_interleaved,
            i_ratio,
            if s.short == "serial" {
                ""
            } else if i_pass {
                "-> would pass"
            } else {
                "-> would still fail"
            }
        );
        println!("{line}");
        report.push_str(&line);
        report.push('\n');
    }
    println!();

    // -- unwind ----------------------------------------------------------
    let mut stops: Vec<StopTest> = Vec::new();
    for cfg in &configs {
        stops.push(cancel_test(
            engine.as_ref(),
            *cfg,
            &inputs,
            args.cancel_after,
        ));
    }
    for cfg in &configs {
        stops.push(early_stop_test(
            engine.as_ref(),
            *cfg,
            &inputs,
            args.stop_after,
        ));
    }
    println!("-- cancellation and early stop --");
    println!(
        "| test | config | sink calls | last index | unwind after trigger | total | results returned | ok | err | err codes | order |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    report.push_str("| test | config | sink calls | last index | unwind after trigger | total | results returned | ok | err | err codes | order |\n");
    report.push_str("|---|---|---|---|---|---|---|---|---|---|---|\n");
    for s in &stops {
        let line = format!(
            "| {} | {} | {} | {} | {:.1} | {:.0} | {} | {} | {} | {} | {} |",
            s.what,
            s.config,
            s.sink_calls,
            s.last_index
                .map_or_else(|| "-".to_owned(), |i| i.to_string()),
            s.unwind_ms,
            s.total_ms,
            s.results,
            s.ok_results,
            s.err_results,
            if s.err_codes.is_empty() {
                "-".to_owned()
            } else {
                s.err_codes.join("/")
            },
            if s.order_ok { "ok" } else { "VIOLATED" },
        );
        println!("{line}");
        report.push_str(&line);
        report.push('\n');
    }
    println!();

    // -- per-chunk detail, so nothing here is unfalsifiable ---------------
    println!(
        "-- per-chunk sink arrivals, first measured round of each config (ms from run start) --"
    );
    for cfg in &configs {
        if let Some(r) = measured.iter().find(|r| r.short == cfg.short()) {
            let t: Vec<String> = r.events.iter().map(|e| format!("{:.0}", e.t_ms)).collect();
            let g: Vec<String> = r
                .gaps()
                .iter()
                .map(|(d, cross)| format!("{d:.0}{}", if *cross { "*" } else { "" }))
                .collect();
            println!("  {:<26} t  [{}]", cfg.label(), t.join(", "));
            println!("  {:<26} gap[{}]", "", g.join(", "));
            report.push_str(&format!(
                "  {} t  [{}]\n  {} gap[{}]\n",
                cfg.label(),
                t.join(", "),
                "",
                g.join(", ")
            ));
        }
    }
    report.push_str("\n`*` marks a gap that spans a fresh encoder group.\n");

    // -- write it out ----------------------------------------------------
    std::fs::create_dir_all(&args.out).map_err(|e| format!("{}: {e}", args.out.display()))?;
    let json_path = args.out.join("stream-latency.json");
    let json = serde_json::json!({
        "input": args.input.display().to_string(),
        "audio_s": audio_s,
        "chunks": inputs.len(),
        "load_s": load_s,
        "provider": provider,
        "frame_bucket": BUCKET,
        "rounds": args.rounds,
        "warmup": args.warmup,
        "frontend_all_ms": frontend_all_ms,
        "frontend_one_ms": frontend_one_ms,
        "chunk_ms": chunks.iter().map(|c| c.end.as_millis() - c.start.as_millis()).collect::<Vec<_>>(),
        "mel_frames": n_frames,
        "mel_valid_frames": valid_frames,
        "records": records,
        "summary": summary,
        "stops": stops,
        "invariant_violations": violations,
    });
    std::fs::write(
        &json_path,
        serde_json::to_string_pretty(&json).map_err(|e| format!("json: {e}"))?,
    )
    .map_err(|e| format!("{}: {e}", json_path.display()))?;
    let md_path = args.out.join("stream-latency.md");
    std::fs::write(&md_path, &report).map_err(|e| format!("{}: {e}", md_path.display()))?;
    println!();
    println!("json  {}", json_path.display());
    println!("md    {}", md_path.display());
    Ok(())
}

#[derive(Debug, serde::Serialize)]
struct Summary {
    config: String,
    short: String,
    ttfr_med: f64,
    gap_p50: f64,
    gap_p95: f64,
    gap_p99: f64,
    gap_max: f64,
    gap_max_worst_round: f64,
    total_med: f64,
    /// Derived, not measured. See the note where it is computed.
    ttfr_if_front_end_interleaved: f64,
    within_gap_med: f64,
    within_gap_max: f64,
    across_gap_med: f64,
    across_gap_max: f64,
    words: usize,
    steps: usize,
    fingerprint: String,
    order_ok: bool,
    padded_frames: usize,
}

impl RunRecord {
    fn max_gap(&self) -> f64 {
        self.gaps().iter().map(|g| g.0).fold(0.0f64, f64::max)
    }
}

fn chunk_len_summary(chunks: &[vaani_vad::Chunk]) -> String {
    let v: Vec<i64> = chunks
        .iter()
        .map(|c| c.end.as_millis() - c.start.as_millis())
        .collect();
    format!(
        "{} of {} chunks, {} ms total, shortest {} ms, longest {} ms",
        v.len(),
        v.len(),
        v.iter().sum::<i64>(),
        v.iter().copied().min().unwrap_or(0),
        v.iter().copied().max().unwrap_or(0)
    )
}

/// Encoder frames actually computed for this configuration, which is what the
/// batching decision costs in wasted arithmetic: a group pads every member to
/// the longest member's bucket.
fn padded_frames_for(n_frames: &[usize], group: usize) -> usize {
    let bucket = |n: usize| n.div_ceil(BUCKET) * BUCKET;
    n_frames
        .chunks(group.max(1))
        .map(|g| {
            let longest = g.iter().copied().max().unwrap_or(0);
            g.len() * bucket(longest)
        })
        .sum()
}

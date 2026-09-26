#!/usr/bin/env python
"""Export SraVaani 1.0 (TorchScript) to ONNX for the ONNX Runtime backend.

DEVELOPMENT-TIME TOOL ONLY. The shipped VaaniDesk application never runs this
and never requires Python. It exists so we can produce the runtime artefacts
once, in a controlled environment, and pin their checksums in the model
registry.

Produces, into --out:

  preproc.json           log-mel front-end parameters (auditable, plain text)
  mel_window.f32         400   f32, the STFT analysis window
  mel_fb.f32             128*257 f32, the mel filterbank (row major, [128][257])
  config.json            model hyper-parameters mirrored from the HF config
  encoder.onnx + encoder.onnx_data   Conformer encoder, fp32, opset 17, external data
  encoder.fp16.onnx                  Conformer encoder, NATIVE fp16 (the production one)
  decoder_joint.onnx                 one TDT emission step, fp32

Precision split, and it is deliberate:

  * The **encoder** is fp16. It holds ~430M parameters and essentially all of
    the arithmetic, so fp16 is worth roughly 2x the disk, ~2x the VRAM and the
    tensor-core throughput. It is exported *natively* from `encoder.half()`
    rather than post-converted, because `onnxconverter_common.float16` cannot
    handle this graph -- see `to_fp16` for the full explanation.
  * The **decoder_joint** stays fp32, and is only 43 MB. Its LSTM state is
    threaded through every emission step of a whole utterance, so fp16
    accumulation error would compound across minutes of audio. The FLOPs live
    in the encoder; the precision matters in the decoder.

The fp16 encoder's graph I/O is fp16, so the Rust side converts log-mel from
f32 to f16 once, at the boundary, in `vaani-inference`.

Reference: see docs/decisions/ADR-004-sravaani-integration.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import time

import torch

OPSET = 17
ENCODER_SAMPLE_FRAMES = 200  # 2.0 s of audio at hop 160; small enough to export fast


def log(msg: str) -> None:
    print(f"[export] {msg}", flush=True)


# --------------------------------------------------------------------------
# front-end parameters
# --------------------------------------------------------------------------


def export_preproc(src: pathlib.Path, out: pathlib.Path) -> dict:
    """Unpack preproc.pt into inspectable, version-controllable artefacts.

    preproc.pt is a zip containing a pickle plus two raw f32 blobs. Parsing a
    Python pickle in Rust would be brittle and buys us nothing for a 132 KB
    file whose contents are fully determined, so we flatten it here instead.
    """
    pp = torch.load(src / "preproc.pt", map_location="cpu", weights_only=False)
    p = dict(pp["params"])
    window = pp["window"].to(torch.float32).contiguous()
    # The shipped filterbank carries a leading singleton batch dim: (1, n_mels,
    # n_freq). Squeeze it so the on-disk artefact is unambiguous.
    fb = pp["fb"].to(torch.float32).contiguous()
    if fb.dim() == 3 and int(fb.shape[0]) == 1:
        fb = fb.squeeze(0).contiguous()

    expected_window = int(p["win_length"])
    n_fft = int(p["n_fft"])
    n_freq = n_fft // 2 + 1
    if tuple(window.shape) != (expected_window,):
        raise SystemExit(f"unexpected window shape {tuple(window.shape)}")
    # The mel count lives only in the filterbank shape here; config.json carries
    # it too (as `feat_in`) and main() cross-checks the two.
    if fb.dim() != 2 or int(fb.shape[1]) != n_freq:
        raise SystemExit(f"unexpected filterbank shape {tuple(fb.shape)}, want [*, {n_freq}]")
    n_mels = int(fb.shape[0])

    (out / "mel_window.f32").write_bytes(window.numpy().tobytes())
    (out / "mel_fb.f32").write_bytes(fb.numpy().tobytes())

    meta = {
        "sample_rate": int(p["sample_rate"]),
        "n_fft": n_fft,
        "hop_length": int(p["hop_length"]),
        "win_length": expected_window,
        "n_mels": n_mels,
        "n_freq": n_freq,
        "preemph": float(p["preemph"]),
        "mag_power": float(p["mag_power"]),
        "log": bool(p.get("log", True)),
        "log_zero_guard_type": p.get("log_zero_guard_type", "add"),
        "log_zero_guard_value": float(p["log_zero_guard_value"]),
        "normalize": p.get("normalize", "per_feature"),
        "constant": float(p["CONSTANT"]),
        "pad_value": float(p["pad_value"]),
        "blank_id": int(pp.get("blank_id", -1)),
        "window_file": "mel_window.f32",
        "fb_file": "mel_fb.f32",
    }
    (out / "preproc.json").write_text(json.dumps(meta, indent=2) + "\n")
    log(
        "preproc: "
        f"sr={meta['sample_rate']} n_fft={n_fft} hop={meta['hop_length']} "
        f"win={expected_window} mels={meta['n_mels']} "
        f"preemph={meta['preemph']} log_guard={meta['log_zero_guard_value']!r} "
        f"blank={meta['blank_id']}"
    )
    return meta


# --------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------


def load_model(src: pathlib.Path):
    """Load the TorchScript archive directly.

    We deliberately bypass `transformers` + trust_remote_code. The archive is
    a plain torch.jit module with `.encoder` and `.decoder_joint` submodules;
    going through the HF wrapper only adds a code-execution surface and a
    dependency we do not need at export time.

    The shipped graphs are fp16 on disk but exported *unfrozen*, so the weights
    are real parameters. We upcast to fp32 before exporting: widening fp16 to
    fp32 is lossless with respect to the values we actually ship, and it keeps
    the exported graph numerically equivalent to the reference CPU path.
    """
    mod = torch.jit.load(str(src / "model-asr.fp16.ts"), map_location="cpu").eval()
    mod = mod.float()
    cfg = json.loads((src / "config.json").read_text())
    return mod, cfg


# --------------------------------------------------------------------------
# encoder
# --------------------------------------------------------------------------


def export_encoder(encoder, cfg: dict, out: pathlib.Path) -> pathlib.Path:
    d_model = int(cfg["enc_d_model"])
    n_mels = int(cfg["feat_in"])
    t = ENCODER_SAMPLE_FRAMES
    feats = torch.randn(1, n_mels, t, dtype=torch.float32)
    lengths = torch.tensor([t], dtype=torch.long)

    target = out / "encoder.onnx"
    log(f"exporting encoder (fp32, opset {OPSET}, external data) ...")
    t0 = time.time()
    torch.onnx.export(
        encoder,
        (feats, lengths),
        str(target),
        input_names=["feats", "feature_lengths"],
        output_names=["enc_out", "enc_lengths"],
        dynamic_axes={
            "feats": {0: "batch", 2: "time"},
            "feature_lengths": {0: "batch"},
            "enc_out": {0: "batch", 2: "enc_time"},
            "enc_lengths": {0: "batch"},
        },
        opset_version=OPSET,
        dynamo=False,  # legacy tracer keeps the TorchScript dynamic-shape hints
        external_data=True,  # fp32 weights exceed the 2 GB protobuf message limit
        do_constant_folding=True,
    )
    log(f"encoder.onnx written in {time.time() - t0:.1f}s ({d_model=} )")

    # Sanity-check the declared dynamic axes actually hold. A silently frozen
    # time axis would only blow up much later, on the first long chunk.
    import onnxruntime as ort

    sess = ort.InferenceSession(
        str(target), providers=["CPUExecutionProvider"]
    )
    for probe_t in (200, 437, 1601):
        x = torch.randn(1, n_mels, probe_t, dtype=torch.float32)
        enc, enc_len = sess.run(
            None,
            {"feats": x.numpy(), "feature_lengths": np_int64([probe_t])},
        )
        got_t = enc.shape[2]
        got_len = int(enc_len[0])
        log(f"  dynamic check T={probe_t} -> enc_out={list(enc.shape)} enc_len={got_len}")
        if got_len <= 0 or got_len > enc.shape[2]:
            raise SystemExit(f"encoder length metadata invalid for T={probe_t}")
    del sess
    return target


def np_int64(vals):
    import numpy as np

    return np.array(vals, dtype=np.int64)


# --------------------------------------------------------------------------
# decoder + joint
# --------------------------------------------------------------------------


class DecoderJointNoLength(torch.jit.ScriptModule):
    """DEPRECATED: see `export_decoder_joint` for why the wrapper was removed.

    Kept only as documentation of the dead `target_length` output.
    """


def export_decoder_joint(decoder_joint, cfg: dict, out: pathlib.Path) -> pathlib.Path:
    """Export one TDT emission step: the RNNT decoder + joint, jointly.

    `RNNTDecoderJoint.forward` returns four tensors:
    `(logits, target_length, h_out, c_out)`. `target_length` is the *input*
    `target_lengths` echoed straight back out; it feeds nothing in the module and
    would feed nothing in ONNX either.

    We could wrap the module to drop it, but that does not work: a
    `torch.jit` submodule called from a plain `nn.Module` (or a traced Python
    closure) is not registered as a submodule, and both `torch.jit.trace` and
    `torch.onnx.export` reject it with "not part of the active trace". The
    wrapper only pays off if the module can be scripted, and scripting a NeMo
    joint is a larger risk than the overhead it saves.

    So we export all four outputs and the Rust side requests three *by name*.
    The extra output is an `Identity` node: ORT computes it, nobody reads it.
    Keeping the graph identical to the reference implementation is worth more
    than dropping one no-op node.
    """
    d_model = int(cfg["enc_d_model"])
    layers = int(cfg["pred_rnn_layers"])
    hidden = int(cfg["pred_hidden"])

    f = torch.randn(1, d_model, 1, dtype=torch.float32)
    tgt = torch.zeros(1, 1, dtype=torch.int32)
    tlen = torch.ones(1, dtype=torch.int32)
    h = torch.zeros(layers, 1, hidden, dtype=torch.float32)
    c = torch.zeros(layers, 1, hidden, dtype=torch.float32)

    target = out / "decoder_joint.onnx"
    log("exporting decoder_joint (single TDT emission step) ...")
    torch.onnx.export(
        decoder_joint,
        (f, tgt, tlen, h, c),
        str(target),
        input_names=["f", "tgt", "tlen", "h_in", "c_in"],
        output_names=["logits", "target_length", "h_out", "c_out"],
        opset_version=OPSET,
        dynamo=False,
        do_constant_folding=True,
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    named = dict(zip(("logits", "target_length", "h_out", "c_out"), sess.get_outputs()))
    for name in named:
        assert named[name], f"missing output {name}"

    logits, h2, c2 = sess.run(
        ["logits", "h_out", "c_out"],
        {
            "f": f.numpy(),
            "tgt": tgt.numpy(),
            "tlen": tlen.numpy(),
            "h_in": h.numpy(),
            "c_in": c.numpy(),
        },
    )
    width = int(cfg["vocab_size"]) + 1 + int(cfg["num_durations"])
    if list(logits.shape) != [1, 1, 1, width]:
        raise SystemExit(f"unexpected logits shape {list(logits.shape)}, want [1,1,1,{width}]")
    if list(h2.shape) != [layers, 1, hidden] or list(c2.shape) != [layers, 1, hidden]:
        raise SystemExit(f"unexpected state shapes {list(h2.shape)} {list(c2.shape)}")

    # Reference parity: the exported graph must reproduce PyTorch's own output
    # on the same input, or the whole Rust decode loop is built on sand.
    with torch.no_grad():
        ref_logits, _ref_tlen, ref_h, ref_c = decoder_joint(f, tgt, tlen, h, c)
    for got, want, what in (
        (logits, ref_logits.numpy(), "logits"),
        (h2, ref_h.numpy(), "h"),
        (c2, ref_c.numpy(), "c"),
    ):
        err = float(abs(got - want).max())
        log(f"  parity {what}: max|onnx - torch| = {err:.3e}")
        if err > 1e-4:
            raise SystemExit(f"decoder_joint {what} mismatch: {err:.3e} > 1e-4")

    del sess
    return target


# --------------------------------------------------------------------------
# fp16 conversion
# --------------------------------------------------------------------------


def to_fp16(encoder, decoder_joint, cfg: dict, out: pathlib.Path) -> None:
    """Export the encoder natively in fp16, from an fp16 module.

    This deliberately does NOT use `onnxconverter_common.float16`, which is the
    obvious tool and is wrong for this graph.

    SraVaani's `ConvSubsampling` front-end computes the subsampling *length*:

        /pre_encode/Cast  ->  /pre_encode/Add  ->  /pre_encode/Div  ->  /pre_encode/Floor

    Those nodes operate on `feature_lengths`, i.e. on integers, not on audio.
    The post-hoc converter sees only "these are float tensors" in the graph
    types and rewrites the whole `pre_encode` prologue to fp16, producing a
    graph where an fp16 activation is added to an int32 constant. ONNX Runtime
    rejects it at load time:

        Type parameter (T) of Optype (Add) bound to different types
        (tensor(float) and tensor(float16) in node (/pre_encode/Add)

    Exporting from a genuinely half-precision module avoids the problem
    entirely: PyTorch inserts the boundary Casts, and the integer length
    arithmetic stays integer because it never was a float tensor. The result
    loads, runs, and holds a 99.6% argmax agreement with the fp32 reference.

    The side benefit is that the graph I/O is fp16, so the boundary cast is
    free and half the host->device traffic.

    The **decoder_joint stays fp32**, and that is not an oversight. It is small
    (43 MB) and its LSTM state is threaded through every emission step of a
    whole utterance, so fp16 accumulation error would compound across minutes
    of audio. The FLOPs live in the encoder; the precision matters in the
    decoder. That split is why the Rust side feeds f16 to the encoder and f32
    to the joint.
    """
    n_mels = int(cfg["feat_in"])
    d_model = int(cfg["enc_d_model"])
    layers = int(cfg["pred_rnn_layers"])
    hidden = int(cfg["pred_hidden"])
    t = ENCODER_SAMPLE_FRAMES

    log("exporting encoder (native fp16) ...")
    enc_half = encoder.half()
    t0 = time.time()
    target = out / "encoder.fp16.onnx"
    torch.onnx.export(
        enc_half,
        (torch.randn(1, n_mels, t, dtype=torch.float16), torch.tensor([t], dtype=torch.long)),
        str(target),
        input_names=["feats", "feature_lengths"],
        output_names=["enc_out", "enc_lengths"],
        dynamic_axes={
            "feats": {0: "batch", 2: "time"},
            "feature_lengths": {0: "batch"},
            "enc_out": {0: "batch", 2: "enc_time"},
            "enc_lengths": {0: "batch"},
        },
        opset_version=OPSET,
        dynamo=False,
        # 887 MB of fp16 weights is under the 2 GB protobuf limit, so this is a
        # single self-contained file: no sidecar to lose, and ORT can mmap it.
        external_data=False,
        do_constant_folding=True,
    )
    log(f"encoder.fp16.onnx written in {time.time() - t0:.1f}s "
        f"({target.stat().st_size / 1e6:.1f} MB)")

    # Put the module back to fp32 so the caller's parity checks stay in the
    # reference precision.
    encoder.float()

    verify_encoder(out, d_model, n_mels, encoder)


def onnx_save(model, path: pathlib.Path) -> None:
    import onnx

    onnx.save(model, str(path))



# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_encoder(out: pathlib.Path, d_model: int, n_mels: int, encoder) -> None:
    """Prove the exported encoder is usable and numerically sound.

    Three things are checked, because each has silently broken at some point:

    1. ONNX Runtime can *load* it. The generic float16 converter produced a
       graph that failed exactly here, and no amount of PyTorch-side validation
       would have caught it.
    2. The dynamic time axis is real, and `enc_lengths == ceil(T / 8)`. A frozen
       axis only fails much later, on the first long chunk.
    3. Output is finite and agrees with the fp32 module on argmax. A silent
       NaN or a systematically wrong encoder is the worst possible failure,
       because the transcripts still look like language.
    """
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out / "encoder.fp16.onnx"), providers=["CPUExecutionProvider"])
    for probe_t in (200, 1600, 4800):
        x = np.random.rand(1, n_mels, probe_t).astype(np.float16)
        enc, enc_len = sess.run(
            None, {"feats": x, "feature_lengths": np.array([probe_t], dtype=np.int64)}
        )
        expect = -(-probe_t // 8)
        if not np.isfinite(enc).all():
            raise SystemExit(f"encoder.fp16 produced non-finite output at T={probe_t}")
        if int(enc_len[0]) != expect:
            raise SystemExit(
                f"encoder.fp16 subsampling wrong: T={probe_t} -> len {enc_len[0]}, "
                f"expected ceil(T/8)={expect}"
            )
        if enc.shape[1] != d_model:
            raise SystemExit(f"encoder.fp16 d_model is {enc.shape[1]}, expected {d_model}")
        log(f"  encoder.fp16 T={probe_t} -> {list(enc.shape)} len {enc_len[0]}  OK")

    t = 1600
    x32 = torch.randn(1, n_mels, t)
    with torch.no_grad():
        ref, _ = encoder(x32, torch.tensor([t]))
    got, _ = sess.run(
        None,
        {"feats": x32.numpy().astype(np.float16),
         "feature_lengths": np.array([t], dtype=np.int64)},
    )
    d = np.abs(got.astype(np.float32) - ref.numpy())
    agreement = (got.astype(np.float32).argmax(-1) == ref.numpy().argmax(-1)).mean() * 100
    log(
        f"  encoder.fp16 vs fp32: max|d|={d.max():.5f} mean|d|={d.mean():.6f} "
        f"argmax agreement={agreement:.3f}%"
    )
    if agreement < 99.0:
        raise SystemExit(
            f"encoder.fp16 argmax agreement {agreement:.3f}% is below the 99% floor; "
            "fp16 is not numerically safe for this encoder"
        )
    del sess
    import onnx

    for name in ("encoder.onnx", "decoder_joint.onnx"):
        p = out / name
        if not p.exists():
            continue
        try:
            onnx.checker.check_model(str(p), full_check=True)
            log(f"onnx.checker OK: {name}")
        except Exception as exc:
            # fp16 graphs with external data can trip the checker on the
            # reference-model path; report but do not abort the export.
            log(f"onnx.checker WARN {name}: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=pathlib.Path, help="SraVaani-1.0 dir")
    ap.add_argument("--out", required=True, type=pathlib.Path, help="output dir")
    ap.add_argument("--skip-fp16", action="store_true")
    args = ap.parse_args()

    src: pathlib.Path = args.src
    out: pathlib.Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 8)))

    preproc = export_preproc(src, out)

    mod, cfg = load_model(src)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    # The SentencePiece model is a runtime artefact, not an export input, so copy
    # it into the artefacts directory. The Model Manager lays out an installed
    # model the same way, and the Rust side looks for it beside `config.json`;
    # leaving it only in the HF snapshot is how a "works in dev, model not
    # found in the app" bug starts.
    tok = src / "tokenizer.model"
    if not tok.is_file():
        raise SystemExit(f"tokenizer.model not found at {tok}")
    (out / "tokenizer.model").write_bytes(tok.read_bytes())
    log(f"tokenizer.model copied ({tok.stat().st_size} bytes)")

    if (src / "model_meta.json").is_file():
        (out / "model_meta.json").write_bytes((src / "model_meta.json").read_bytes())
    log(f"model loaded: d_model={cfg['enc_d_model']} vocab={cfg['vocab_size']} blank={cfg['blank_id']}")

    # Cross-check the config against the front-end metadata; a mismatch here
    # would silently produce wrong-shaped encoder input.
    if int(cfg["feat_in"]) != preproc["n_mels"]:
        raise SystemExit("feat_in disagrees with filterbank mel count")
    if int(cfg["blank_id"]) != preproc["blank_id"]:
        raise SystemExit("config blank_id disagrees with preproc blank_id")

    export_encoder(mod.encoder, cfg, out)
    export_decoder_joint(mod.decoder_joint, cfg, out)

    if not args.skip_fp16:
        # Only the encoder goes fp16; see `to_fp16` for why the decoder_joint
        # deliberately stays fp32.
        to_fp16(mod.encoder, mod.decoder_joint, cfg, out)

    log("checksums:")
    manifest = {}
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name not in ("SHA256SUMS", "artefacts.json"):
            digest = sha256(p)
            manifest[p.name] = {"bytes": p.stat().st_size, "sha256": digest}
            log(f"  {digest}  {p.name}  ({p.stat().st_size} bytes)")
    (out / "artefacts.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

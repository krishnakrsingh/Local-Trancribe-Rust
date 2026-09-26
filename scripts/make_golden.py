#!/usr/bin/env python
"""Produce the golden fixtures the Rust tests are gated on.

DEVELOPMENT-TIME TOOL. The shipped application never runs this and never needs
Python.

It writes, into the path given by --out (default `tests/fixtures`):

  sravaani/frontend_golden.json
      Log-mel features computed by PyTorch from the model's own preproc
      tensors, for a deterministic synthetic signal. `frontend_parity.rs`
      asserts the Rust front end reproduces these to 1e-4.

  sravaani/decode_golden.json
      Token ids and word timings produced by the *reference* implementation
      (`modeling_sravaani.py`'s TDT loop) for a slice of real speech, plus the
      text. `sravaani_transcribe.rs` asserts the Rust decode loop produces the
      same ids. This is the test that would catch a subtle change in the decode
      semantics -- a state update on blank, a missing max_symbols guard -- which
      no unit test can see.

The synthetic signal is generated rather than sampled from the user's private
meeting recording so the committed fixture contains no user audio. The decode
fixture is token ids and timings only, never the audio itself.

Usage:
    python scripts/make_golden.py --src SraVaani-1.0 --out tests/fixtures
    python scripts/make_golden.py --decode-audio scratch/clip.wav
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# The Windows console default is cp1252, which cannot print Devanagari. Every
# script here prints reference transcript text, so this has to be fixed before
# the first print, not caught per call.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:  # pragma: no cover - very old Python
    pass

import numpy as np
import torch

OPSET = 17


def log(msg: str) -> None:
    print(f"[golden] {msg}", flush=True)


# ---------------------------------------------------------------------------
# front-end fixture
# ---------------------------------------------------------------------------


def synthetic_signal(n: int, seed: int = 12345) -> np.ndarray:
    """A deterministic, speech-shaped signal.

    Three components so that the mel projection sees real structure: a 120 Hz
    buzz for the low bands, a 2 kHz "formant", and a 3.1 kHz second formant,
    plus low-amplitude noise so no band is exactly silent. Deterministic
    because the seed is fixed and the generator is a closed form.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / 16000.0
    x = (
        0.30 * np.sin(2 * np.pi * 120.0 * t)
        + 0.20 * np.sin(2 * np.pi * 2000.0 * t + 0.4)
        + 0.12 * np.sin(2 * np.pi * 3100.0 * t + 1.1)
        + 0.02 * rng.standard_normal(n)
    )
    # A short fade-in and fade-out so the first and last frames are not
    # discontinuous, which would put a click into the top mel bands.
    ramp = np.ones(n)
    edge = min(400, n // 8)
    ramp[:edge] = np.linspace(0.0, 1.0, edge)
    ramp[-edge:] = np.linspace(1.0, 0.0, edge)
    return (x * ramp).astype(np.float32)


def longhand_log_mel(wav: np.ndarray, pp: dict) -> np.ndarray:
    """An independent transcription of the front end, used as a cross-check.

    Written out longhand from `processing_sravaani.py` so that if the model's
    own `extract_features` and this file's reading of it ever disagree, the
    disagreement is reported rather than silently baked into a fixture.
    """
    p = dict(pp["params"])
    window = pp["window"].to(torch.float32)
    fb = pp["fb"].to(torch.float32)
    if fb.dim() == 3:
        fb = fb.squeeze(0)

    n_fft, hop, win = int(p["n_fft"]), int(p["hop_length"]), int(p["win_length"])
    w = torch.from_numpy(np.ascontiguousarray(wav)).float().unsqueeze(0)  # [1, T]
    seq_len = w.shape[1] // hop

    x = torch.cat((w[:, :1], w[:, 1:] - p["preemph"] * w[:, :-1]), dim=1)
    spec = torch.stft(
        x, n_fft=n_fft, hop_length=hop, win_length=win, window=window,
        center=True, pad_mode="constant", return_complex=True,
    )
    x = torch.view_as_real(spec)
    x = torch.sqrt(x.pow(2).sum(-1)).pow(p["mag_power"])
    x = torch.matmul(fb, x)                                    # [1, n_mels, T]
    x = torch.log(x + p["log_zero_guard_value"])

    # Per-feature normalisation over the VALID frames, Bessel corrected.
    steps = torch.arange(x.size(-1)).unsqueeze(0)
    valid = steps < seq_len
    n_valid = float(seq_len)
    mean = (x * valid.unsqueeze(1)).sum(dim=2) / n_valid          # [1, n_mels]
    var = ((x - mean.unsqueeze(2)) ** 2 * valid.unsqueeze(1)).sum(dim=2) / (n_valid - 1.0)
    sd = torch.sqrt(var)
    sd = torch.where(sd.isnan(), torch.zeros_like(sd), sd)
    x = (x - mean.unsqueeze(2)) / (sd + p["CONSTANT"]).unsqueeze(2)

    m = torch.arange(x.size(-1)).unsqueeze(0) >= seq_len
    x = x.masked_fill(m.unsqueeze(1), p["pad_value"])
    return x.squeeze(0).numpy()                                  # [n_mels, n_frames]


def load_hf_model(src: pathlib.Path):
    """The reference HF wrapper, which is where `extract_features` lives.

    The TorchScript archive only holds `.encoder` and `.decoder_joint`; the
    front end is Python, in `modeling_sravaani.py`. So the authority for the
    feature fixture has to come through the wrapper, loaded from the local
    directory with `trust_remote_code` -- which is fine here because this is a
    development-time script operating on a checkout we are shipping, and the
    shipped application never runs any of it.
    """
    from transformers import AutoModel

    return AutoModel.from_pretrained(str(src), trust_remote_code=True).eval().float()


def model_log_mel(model, wav: np.ndarray):
    """The model's own `extract_features` -- the authority for the fixture."""
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(wav)).float().unsqueeze(0)
        f, seq_len = model.extract_features(x, torch.tensor([x.shape[1]]))
    return f.squeeze(0).numpy(), int(seq_len[0])


def make_frontend_fixture(src: pathlib.Path, out: pathlib.Path) -> None:
    pp = torch.load(src / "preproc.pt", map_location="cpu", weights_only=False)
    model = load_hf_model(src)

    n = 16000  # exactly 1.0 s -> 100 valid frames, 101 STFT frames
    wav = synthetic_signal(n)

    feats, valid = model_log_mel(model, wav)
    # Cross-check: an independent transcription of the same maths must agree
    # with the model to float32 rounding. A mismatch here means this file's
    # reading of the front end is wrong, and the fixture would encode the error.
    mine = longhand_log_mel(wav, pp)
    if mine.shape != feats.shape:
        raise SystemExit(f"cross-check shape mismatch: {mine.shape} vs {feats.shape}")
    d = np.abs(mine - feats)
    log(f"  cross-check vs model.extract_features: max|d|={d.max():.3e} mean|d|={d.mean():.3e}")
    if d.max() > 1e-4:
        raise SystemExit(
            f"longhand front end disagrees with the model by {d.max():.3e}; "
            "one of the two is wrong and the fixture must not be generated"
        )

    p = dict(pp["params"])
    fixture = {
        "_comment": (
            "Generated by scripts/make_golden.py. Log-mel for a deterministic "
            "synthetic signal, computed by the model's own extract_features. "
            "Contains no user audio."
        ),
        "sample_rate": int(p["sample_rate"]),
        "n_samples": n,
        "signal": [float(v) for v in wav],
        "n_mels": int(feats.shape[0]),
        "n_frames": int(feats.shape[1]),
        "valid_frames": valid,
        # Every cell, not a sample: 128 x 101 f32 is a small fixture, and it
        # means one wrong mel band cannot hide in an unsampled region.
        "log_mel": [[float(v) for v in row] for row in feats],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "frontend_golden.json").write_text(json.dumps(fixture), encoding="utf-8")
    log(f"frontend_golden.json: {feats.shape[0]} mels x {feats.shape[1]} frames")


# ---------------------------------------------------------------------------
# decode fixture
# ---------------------------------------------------------------------------


def make_decode_fixture(
    src: pathlib.Path, out: pathlib.Path, wav_path: pathlib.Path, seconds: float,
    raw_out: pathlib.Path | None = None,
) -> None:
    """Reference TDT decode of a real-speech clip, from the TorchScript model.

    This is the fixture that pins the decode *semantics*. The loop below is
    transcribed from `modeling_sravaani.py::_greedy_one`; if the Rust loop
    diverges, this catches it.
    """
    import sys as _sys

    _sys.path.insert(0, str(src))
    pp = torch.load(src / "preproc.pt", map_location="cpu", weights_only=False)
    cfg = json.loads((src / "config.json").read_text())
    model = load_hf_model(src)
    pp["params"]["blank_id"] = cfg["blank_id"]

    import soundfile  # noqa: F401  (only to confirm availability)

    wav, sr = soundfile.read(str(wav_path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != cfg["sample_rate"]:
        # Fourier resample, matching the reference `_resample`.
        n_new = max(int(round(len(wav) * cfg["sample_rate"] / sr)), 1)
        X = np.fft.rfft(wav)
        Y = np.zeros(n_new // 2 + 1, dtype=X.dtype)
        k = min(len(X), len(Y))
        Y[:k] = X[:k]
        wav = (np.fft.irfft(Y, n=n_new) * (n_new / len(wav))).astype(np.float32)

    n = int(seconds * cfg["sample_rate"])
    wav = wav[:n]
    with torch.no_grad():
        xin = torch.from_numpy(wav).unsqueeze(0)
        f, flen = model.extract_features(xin, torch.tensor([xin.shape[1]]))
        enc, enc_len = model.encoder(f, flen)
    T = int(enc_len[0])

    # --- the reference greedy TDT loop, transcribed literally ---------------
    blank = cfg["blank_id"]
    durations = cfg["durations"]
    nd = cfg["num_durations"]
    L, H = cfg["pred_rnn_layers"], cfg["pred_hidden"]
    h = torch.zeros(L, 1, H)
    c = torch.zeros(L, 1, H)
    last, toks, frames = blank, [], []
    t = 0
    steps = 0
    while t < T:
        fr = enc.narrow(2, t, 1)
        added, need = 0, True
        while need and added < cfg["max_symbols"]:
            tgt = torch.tensor([[last]], dtype=torch.int32)
            logits, _, h2, c2 = model.decoder_joint(
                fr, tgt, torch.ones(1, dtype=torch.int32), h, c
            )
            v = logits[0, 0, 0]
            k = int(v[:-nd].argmax().item())
            skip = durations[int(v[-nd:].argmax().item())]
            if k != blank:
                toks.append(k)
                frames.append(t)
                h, c, last = h2, c2, k
            added += 1
            steps += 1
            t += skip
            need = skip == 0
        if added == cfg["max_symbols"]:
            t += 1

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(src / "tokenizer.model"))
    text = sp.decode(toks)

    fd = cfg["subsampling_factor"] * pp["params"]["hop_length"] / pp["params"]["sample_rate"]
    words = []
    groups = []
    for tok, fr in zip(toks, frames):
        if sp.id_to_piece(tok).startswith("▁") or not groups:
            groups.append([fr, [tok]])
        else:
            groups[-1][1].append(tok)
    for gi, (sfr, gtoks) in enumerate(groups):
        end_fr = groups[gi + 1][0] if gi + 1 < len(groups) else -(-int(flen[0]) // cfg["subsampling_factor"])
        words.append(
            {
                "word": sp.decode(gtoks),
                "start": round(sfr * fd, 3),
                "end": round(end_fr * fd, 3),
            }
        )

    fixture = {
        "_comment": (
            "Generated by scripts/make_golden.py. Reference TDT decode from the "
            "model's own TorchScript graphs, via modeling_sravaani.py's loop. "
            "Token ids and timings only; contains no user audio."
        ),
        "source": wav_path.name,
        "seconds": seconds,
        "sample_rate": cfg["sample_rate"],
        "encoder_frames": T,
        "decode_steps": steps,
        "frame_seconds": fd,
        "token_ids": toks,
        "token_frames": frames,
        "text": text,
        "words": words,
    }
    (out / "decode_golden.json").write_text(
        json.dumps(fixture, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    # A raw little-endian f32 sidecar next to the fixture, in the scratch
    # directory and NOT in tests/fixtures: it is real meeting audio and must
    # never be committed. Writing raw rather than WAV because soundfile's float
    # WAV header is longer than the canonical 44 bytes, and a Rust test that
    # assumes 44 silently reads misaligned garbage -- which looks like a broken
    # model rather than a broken reader.
    raw_out = raw_out or (out.parent.parent.parent / "scratch" / "golden_clip.f32")
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    raw_out.write_bytes(np.ascontiguousarray(wav, dtype=np.float32).tobytes())
    log(f"raw clip written to {raw_out} ({len(wav)} samples) -- NOT for committing")
    log(f"decode_golden.json: {len(toks)} tokens, {steps} steps, {T} encoder frames")
    log(f"  text: {text[:160]}")


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=pathlib.Path, default=pathlib.Path("SraVaani-1.0"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("tests/fixtures"))
    ap.add_argument("--decode-audio", type=pathlib.Path, help="wav for the decode fixture")
    ap.add_argument("--decode-seconds", type=float, default=30.0)
    ap.add_argument("--decode-raw-out", type=pathlib.Path, help="where to write the raw f32 clip")
    args = ap.parse_args()

    out = args.out / "sravaani"
    make_frontend_fixture(args.src, out)

    if args.decode_audio:
        if not args.decode_audio.is_file():
            log(f"decode audio not found: {args.decode_audio}; skipping that fixture")
            return 1
        make_decode_fixture(args.src, out, args.decode_audio, args.decode_seconds, args.decode_raw_out)
    else:
        log("no --decode-audio given; skipped the decode fixture")
    return 0


if __name__ == "__main__":
    sys.exit(main())

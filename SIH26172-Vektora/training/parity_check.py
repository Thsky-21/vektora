"""
parity_check.py — compile the firmware's feature_extract.c on the laptop and
compare it, value by value, with features.py (CLAUDE.md §6 risk 1).

Run (from training, after export_c.py):
    ../.venv/Scripts/python.exe parity_check.py

Needs a host C compiler (gcc on PATH; set CC to override). The same C file
builds into the ESP32 firmware, so a pass here means the device computes the
picture the model was trained on. A float mismatch here is a bug; a pass here
does not cover I2S capture levels, which must be checked on the board.

What is compared, per clip:
  raw log-mel dB     |C - Python|, max          (tolerance 0.01 dB)
  normalised         |C - Python|, max          (tolerance 0.001)
  int8 model input   number of differing values (tolerance: 0.1% of values,
                     and never more than 1 step apart)
Then the int8 model is run on BOTH int8 inputs and P(keyword) is compared.

The float comparisons skip values more than DYN_RANGE_DB below the clip's
loudest value. There, float32 (the ESP32 FPU is single precision) can't
resolve the cancellation inside the FFT sum. Measured on a full-scale 1 kHz
tone: 0.06 dB error in bands 116-120 dB below the peak, zero int8 effect. The
INMP441's dynamic range is ~87 dB, so a real signal never gets there. The
int8 check still covers every value.
"""

import csv
import json
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import features as F  # noqa: E402

FW_MAIN = os.path.join(C.PROTO, "firmware", "main")
HOST = os.path.join(C.PROTO, "firmware", "host_test")
N_PER_SOURCE = 8
TOL_DB, TOL_NORM, TOL_Q_FRAC = 0.01, 0.001, 0.001
DYN_RANGE_DB = 90.0


def test_clips(rng):
    """Real clips from every source (val/test/train) plus synthetic edge cases."""
    with open(C.MANIFEST) as f:
        rows = [r for r in csv.DictReader(f) if r["offset"] == "0"]
    clips, names = [], []
    for src in sorted({r["source"] for r in rows}):
        pool = [r for r in rows if r["source"] == src]
        for r in rng.choice(pool, size=min(N_PER_SOURCE, len(pool)), replace=False):
            x, sr = sf.read(C.resolve(r["path"]), dtype="int16")
            assert sr == F.SR and x.ndim == 1, r["path"]
            # same centre pad/crop as features.fix_length, done in int16
            clips.append(np.round(F.fix_length(x.astype(np.float32))).astype(np.int16))
            names.append(f"{src}:{os.path.basename(r['path'])}")
    t = np.arange(F.CLIP) / F.SR
    synth = {
        "digital silence": np.zeros(F.CLIP),
        "full-scale 1 kHz": 32767 * np.sin(2 * np.pi * 1000 * t),
        "clipped square": 32767 * np.sign(np.sin(2 * np.pi * 300 * t)),
        "negative full scale": -32768 * np.ones(F.CLIP),
        "white noise -40 dBFS": rng.normal(0, 328, F.CLIP),
        "7.5 kHz tone": 10000 * np.sin(2 * np.pi * 7500 * t),
    }
    for k, v in synth.items():
        clips.append(np.clip(np.round(v), -32768, 32767).astype(np.int16))
        names.append(f"synthetic:{k}")
    return np.stack(clips), names


def build(exe):
    cc = os.environ.get("CC", "gcc")
    cmd = [cc, "-std=gnu99", "-O2", "-Wall", "-Wextra", "-I", FW_MAIN,
           os.path.join(FW_MAIN, "feature_extract.c"), os.path.join(HOST, "parity_main.c"),
           "-o", exe, "-lm"]
    subprocess.run(cmd, check=True)


def main():
    rng = np.random.default_rng(0)
    norm = F.Normalizer.load(os.path.join(C.MODEL_DIR, "norm.json"))
    with open(os.path.join(C.MODEL_DIR, "quant_report.json")) as f:
        qr = json.load(f)
    in_s, in_zp = qr["input"]["scale"], qr["input"]["zero_point"]
    pcm, names = test_clips(rng)
    n = F.N_FRAMES * F.N_MELS

    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, "parity_main.exe")
        build(exe)
        pcm.astype("<i2").tofile(os.path.join(tmp, "in.i16"))
        subprocess.run([exe, os.path.join(tmp, "in.i16"), os.path.join(tmp, "out.bin")], check=True)
        out = np.fromfile(os.path.join(tmp, "out.bin"), dtype=np.uint8)

    rec = 4 * n + 4 * n + n
    assert out.size == rec * len(pcm), (out.size, rec * len(pcm))
    out = out.reshape(len(pcm), rec)
    c_raw = out[:, :4 * n].copy().view("<f4").reshape(-1, F.N_FRAMES, F.N_MELS)
    c_nrm = out[:, 4 * n:8 * n].copy().view("<f4").reshape(-1, F.N_FRAMES, F.N_MELS)
    c_q = out[:, 8 * n:].copy().view(np.int8).reshape(-1, F.N_FRAMES, F.N_MELS)

    x = pcm.astype(np.float32) / 32768.0
    p_raw = np.stack([F.logmel(c) for c in x])
    p_nrm = norm(p_raw)
    p_q = np.clip(np.round(p_nrm / in_s) + in_zp, -128, 127).astype(np.int8)

    ok = True
    print(f"{'clip':<52}{'max dB diff':>12}{'max norm diff':>15}{'int8 diffs':>12}")
    for i, name in enumerate(names):
        m = p_raw[i] >= p_raw[i].max() - DYN_RANGE_DB
        d_db = float(np.abs(c_raw[i] - p_raw[i])[m].max())
        d_n = float(np.abs(c_nrm[i] - p_nrm[i])[m].max())
        dq = np.abs(c_q[i].astype(int) - p_q[i].astype(int))
        bad = d_db > TOL_DB or d_n > TOL_NORM or dq.max() > 1 or (dq > 0).mean() > TOL_Q_FRAC
        ok &= not bad
        print(f"{name[:51]:<52}{d_db:>12.5f}{d_n:>15.6f}{int((dq > 0).sum()):>12}" + ("   FAIL" if bad else ""))

    # Same int8 model on both inputs.
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=os.path.join(C.MODEL_DIR, "model_int8.tflite"),
                             experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    ii, oo = it.get_input_details()[0]["index"], it.get_output_details()[0]
    os_, ozp = oo["quantization"]

    def run(q):
        it.set_tensor(ii, q[None, ..., None])
        it.invoke()
        return os_ * (it.get_tensor(oo["index"])[0].astype(np.float32) - ozp)

    kw = C.CLASSES.index("keyword")
    dp = [abs(run(c_q[i])[kw] - run(p_q[i])[kw]) for i in range(len(pcm))]
    print(f"\nP(keyword) from C features vs Python features: max diff {max(dp):.4f}")
    ok &= max(dp) <= 2 * os_
    print("PARITY PASS" if ok else "PARITY FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""
app.py — local Streamlit page for testing the prototype model by hand.

Run from the repo root:
    .venv/Scripts/python.exe -m streamlit run prototype/ui/app.py

What it does with a recording or an uploaded file:
  audio -> mono float, resampled to 16 kHz (NO peak normalisation — training
  doesn't use it either, see features.py)
  -> 1 s windows every 0.1 s (a clip shorter than 1 s is centre-padded)
  -> features.logmel -> Normalizer(norm.json) -> 3-class CNN
  -> P(keyword) per window; verdict = best window vs the threshold in report.json

No smoothing, no energy gate: the post-processing is still an open decision
(CLAUDE.md §3 item 5), so this page shows the model's raw per-window output.
"""

import io
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import streamlit as st  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.join(os.path.dirname(HERE), "training")
sys.path.insert(0, TRAINING)
import features as F  # noqa: E402
from config import CLASSES, MODEL_DIR, PROTO  # noqa: E402

HOP_S = 0.1
CAPTURE_DIR = os.path.join(PROTO, "data", "ui_captures")    # git-ignored (prototype/data/)

st.set_page_config(page_title="Vektora prototype tester", page_icon="🎙️", layout="wide")


@st.cache_resource
def load_model(stamp: float):
    """`stamp` = model file mtime; part of the cache key so a retrain reloads."""
    import tensorflow as tf
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "model.keras"))
    norm = F.Normalizer.load(os.path.join(MODEL_DIR, "norm.json"))
    with open(os.path.join(MODEL_DIR, "report.json")) as f:
        report = json.load(f)
    return model, norm, report


def read_audio(data: bytes) -> np.ndarray:
    """Any wav/flac/ogg/mp3 bytes -> mono float32 at 16 kHz."""
    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    except Exception:
        x, sr = librosa.load(io.BytesIO(data), sr=None, mono=True)
    if sr != F.SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=F.SR)
    return x.astype(np.float32)


def analyse(x, model, norm):
    padded = len(x) < F.CLIP
    if padded:
        x = F.fix_length(x)
    starts = list(range(0, len(x) - F.CLIP + 1, int(HOP_S * F.SR)))
    feats = np.stack([norm(F.logmel(x[s:s + F.CLIP])) for s in starts])[..., None]
    probs = model.predict(feats, batch_size=64, verbose=0)          # (n, 3)
    best = int(np.argmax(probs[:, 0]))
    return {
        "x": x, "padded": padded, "starts": np.array(starts) / F.SR,
        "probs": probs, "best": best, "best_feat": feats[best, ..., 0],
    }


def show(r, threshold, dur_in):
    p = r["probs"]
    best, bs = r["best"], r["starts"][r["best"]]
    kw = float(p[best, 0])

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    if kw >= threshold:
        c1.success(f"### ✅ Vektora detected\nbest window {bs:.1f}–{bs + 1:.1f} s")
    else:
        c1.error(f"### ❌ No Vektora\nbest window {bs:.1f}–{bs + 1:.1f} s")
    c2.metric("Best P(keyword)", f"{kw:.3f}")
    c3.metric("Threshold", f"{threshold:.2f}")
    c4.metric("Windows scored", len(p))

    st.caption(f"Input {dur_in:.2f} s · peak {np.max(np.abs(r['x'])):.3f} "
               f"({20 * np.log10(np.max(np.abs(r['x'])) + 1e-12):.1f} dBFS)"
               + (" · **shorter than 1 s: zero-padded** — the model has learned that exact "
                  "zeros are suspicious (CLAUDE.md §10.6), so treat this score with care"
                  if r["padded"] else ""))

    st.markdown("**Class probabilities per window** (x = window start, s)")
    df = pd.DataFrame(p, columns=CLASSES, index=np.round(r["starts"], 2))
    st.line_chart(df, height=260)

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 2.6))
        t = np.arange(len(r["x"])) / F.SR
        ax.plot(t, r["x"], lw=0.5, color="#4a6fa5")
        ax.axvspan(bs, bs + 1, color="orange", alpha=0.25)
        ax.set(xlabel="s", title="Waveform (shaded = best window)", xlim=(0, t[-1]))
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(6, 2.6))
        im = ax.imshow(r["best_feat"].T, origin="lower", aspect="auto", cmap="magma")
        ax.set(xlabel="frame (10 ms)", ylabel="mel band", title="Model input, best window (normalised)")
        fig.colorbar(im, ax=ax); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with st.expander("Per-window table"):
        st.dataframe(df.style.format("{:.3f}").highlight_max(axis=0), height=300)


# ---------------------------------------------------------------------------
st.title("🎙️ Vektora prototype — model tester")
model_path = os.path.join(MODEL_DIR, "model.keras")
try:
    model, norm, report = load_model(os.path.getmtime(model_path))
except Exception as e:
    st.error(f"Could not load the model from `{MODEL_DIR}` — run train.py first. ({e})")
    st.stop()

default_thr = float(report["threshold"])
with st.sidebar:
    st.subheader("Model")
    st.write(f"`training/model/model.keras` · {report['parameters']:,} params")
    st.write(f"Best epoch {report['best_epoch']} · val acc {report['best_val_acc']:.3f}")
    st.write(f"Test 3-class acc {report['test_accuracy_3class']:.3f}")
    threshold = st.slider("Threshold (P keyword)", 0.05, 0.99, default_thr, 0.01,
                          help=f"Default {default_thr:.2f} = value chosen on validation by train.py.")
    save = st.checkbox("Save recordings to prototype/data/ui_captures/", value=True)

tab_rec, tab_up = st.tabs(["🎙️ Record", "⬆️ Upload"])
with tab_rec:
    st.write("Press the mic, speak, press stop. Say **Vektora**, or try look-alikes.")
    rec = st.audio_input("Record", sample_rate=16000, label_visibility="collapsed")
    if rec is not None:
        data = rec.getvalue()
        x = read_audio(data)
        if save:
            os.makedirs(CAPTURE_DIR, exist_ok=True)
            sf.write(os.path.join(CAPTURE_DIR, f"cap_{int(time.time())}.wav"), x, F.SR)
        show(analyse(x, model, norm), threshold, len(x) / F.SR)

with tab_up:
    ups = st.file_uploader("WAV / FLAC / OGG / MP3 (any length, any sample rate)",
                           type=["wav", "flac", "ogg", "mp3"], accept_multiple_files=True)
    for up in ups or []:
        st.divider()
        st.subheader(up.name)
        data = up.getvalue()
        st.audio(data)
        try:
            x = read_audio(data)
        except Exception as e:
            st.error(f"Couldn't read this file ({e}).")
            continue
        show(analyse(x, model, norm), threshold, len(x) / F.SR)

"""
app.py — the Streamlit website.

Run locally from the repo root:
    .venv/Scripts/python.exe -m streamlit run demo/app.py

What happens when someone records (demo.md §4, step 4):
  browser mic -> WAV bytes (16 kHz) -> peak-normalise the whole recording
  -> slide a 1 s window along it every 0.1 s
  -> skip windows that are near-silent (a tiny "voice activity detector")
  -> log-mel picture -> normalise with the SAME numbers as training
  -> CNN -> one score per window -> best score vs threshold -> verdict
"""

import glob
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import soundfile as sf
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audio_utils as au  # noqa: E402
import plots as P  # noqa: E402
from plots import plt  # noqa: E402

st.set_page_config(page_title="Vektora wake-word demo", page_icon="🎙️", layout="wide")


# ---------------------------------------------------------------------------
# Load the trained model once and keep it in memory (cache_resource), instead
# of reloading it every time someone clicks something.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(stamp: float = 0.0):
    """`stamp` is the model file's timestamp. It isn't used inside, but it is
    part of the cache key, so retraining automatically reloads the model.
    Without it a running server keeps serving the OLD model for ever — which
    once looked exactly like "the new training didn't work"."""
    import tensorflow as tf
    model = tf.keras.models.load_model(os.path.join(HERE, "model", "vektora.keras"))
    norm = au.Normalizer.load(os.path.join(HERE, "model", "norm.json"))
    metrics = json.load(open(os.path.join(HERE, "model", "metrics.json")))
    return model, norm, metrics


def smooth(scores, n=7):
    """Average each score with its neighbours. A real wake word must stay
    confident for several windows in a row; a single spike as some other word
    drifts past is what causes false alarms. This is the cheapest and most
    effective anti-false-alarm trick there is (demo.md §5.6).

    Widened from 3 windows (0.3 s) to 7 (0.7 s) on 2026-09-12. "Vektora" takes
    about 0.7 s to say, so a genuine detection holds high confidence across
    that whole span, while a look-alike word produces a brief peak. The page
    then reports the BEST score, i.e. the maximum over ~30 sliding windows —
    thirty chances for one unlucky window to cross the line — so demanding a
    sustained peak rather than an instantaneous one is what makes that maximum
    trustworthy."""
    if len(scores) < n:
        return scores
    pad = np.pad(scores, (n // 2, n // 2), mode="edge")
    return np.convolve(pad, np.ones(n) / n, mode="valid")


def analyse(x_raw, model, norm, threshold):
    """Score a recording of any length. Returns everything the page draws."""
    raw_peak = float(np.max(np.abs(x_raw))) if len(x_raw) else 0.0
    x = au.peak_normalize(x_raw)
    starts, windows = zip(*au.sliding_windows(x, hop_s=0.1))
    loud = np.array([au.rms_dbfs(w) >= au.SILENCE_DBFS for w in windows])
    raw_scores = np.zeros(len(windows), dtype=np.float32)
    feats = [norm(au.logmel(w)) for w in windows]
    if loud.any():
        batch = np.stack([f for f, ok in zip(feats, loud) if ok])[..., None]
        raw_scores[loud] = model(batch, training=False).numpy().ravel()
    scores = smooth(raw_scores)
    best = int(np.argmax(scores))
    return {
        "x": x, "raw_peak": raw_peak, "duration": len(x) / au.SR,
        "times": np.array(starts) + 0.5, "scores": scores, "raw_scores": raw_scores, "best": best,
        "best_start": starts[best], "best_score": float(scores[best]),
        "best_feat": au.logmel(windows[best]), "any_loud": bool(loud.any()),
        "detected": float(scores[best]) >= threshold,
    }


def show_result(r, threshold):
    if r["raw_peak"] < 0.003:
        st.warning("🔇 That was almost completely silent. Check that the browser is allowed "
                   "to use the microphone, and speak a little closer.")
        return
    if not r["any_loud"]:
        st.info("🤫 Only near-silence was heard, so nothing was sent to the model.")
        return

    c1, c2, c3 = st.columns([2.2, 1, 1])
    if r["detected"]:
        c1.success(f"### ✅ Vektora detected\nat about {r['best_start'] + 0.5:.1f} s into the recording")
    else:
        c1.error("### ❌ No Vektora heard")
    c2.metric("Highest confidence", f"{r['best_score']:.0%}")
    c3.metric("Threshold", f"{threshold:.0%}", help="Chosen on validation data during training.")

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 2.6))
        hl = (r["best_start"], r["best_start"] + 1.0) if r["duration"] > 1.0 else None
        P.draw_waveform(ax, r["x"], au.SR, title="Your recording (shaded = best 1 s window)", highlight=hl)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(6, 2.6))
        if len(r["scores"]) > 1:
            P.draw_confidence(ax, r["times"], r["scores"], threshold, raw=r["raw_scores"])
        else:
            P.draw_gauge(ax, r["best_score"], threshold)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 3.0))
        P.draw_spectrogram(ax, r["best_feat"], title="What the model saw (best window)")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with right:
        st.markdown(
            "**How to read this**\n\n"
            "- **Spectrogram:** time runs left→right, pitch bottom→top, darker = louder. "
            "In *ve-**k-t**-ora* look for a short pale gap followed by a dark vertical "
            "burst: that's the *k-t* sound, the model's favourite clue.\n"
            "- **Confidence over time:** the model looks at a 1-second window, slides it "
            "0.1 s, looks again. A peak above the dashed line = detected.\n"
            "- Windows that are nearly silent are skipped (score 0), like the low-power "
            "*voice activity detector* stage of a real wake-word device.")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("🎙️ Vektora — wake-word detector")
st.caption("A ~24,000-parameter neural network, trained from scratch on our own recordings, "
           "listens for one word: **Vektora**.")

try:
    model, norm, metrics = load_model(os.path.getmtime(os.path.join(HERE, "model", "vektora.keras")))
except Exception as e:                       # model not trained yet
    st.error(f"Model files not found — run `demo/train.py` first. ({e})")
    st.stop()
threshold = float(metrics["threshold"])

COLLECT_DIR = os.path.join(HERE, "data")
LOCAL = os.path.isdir(COLLECT_DIR)      # only true on the dev machine, never on the Space
tab_try, tab_train, tab_how = st.tabs(["🎙️ Try it", "📊 How it was trained", "🧠 How it works"])

with tab_try:
    sens = st.select_slider(
        "Sensitivity", ["Strict", "Balanced", "Sensitive"], value="Balanced",
        help="Strict = fewer false alarms but you may have to repeat yourself. "
             "Sensitive = catches every Vektora but may fire at other words.")
    threshold = {"Strict": min(0.97, threshold + 0.15),
                 "Balanced": threshold,
                 "Sensitive": max(0.30, threshold - 0.15)}[sens]
    mode = st.radio("Input", ["🎙️ Record", "▶️ Example clips", "⬆️ Upload a file"],
                    horizontal=True, label_visibility="collapsed")
    audio_bytes = None
    if mode == "🎙️ Record":
        st.write("Press the mic, say **“Vektora”** once (or a few other words, to try to fool it), "
                 "then press stop. 2–4 seconds is ideal.")
        rec = st.audio_input("Record", label_visibility="collapsed")
        if rec is not None:
            audio_bytes = rec.getvalue()
    elif mode == "▶️ Example clips":
        examples = sorted(glob.glob(os.path.join(HERE, "examples", "*.wav")))
        if not examples:
            st.info("No example clips bundled.")
        else:
            by_name = {os.path.basename(p)[:-4].replace("_", " "): p for p in examples}
            choice = st.selectbox("These clips were held out: the model never saw them in training.",
                                  list(by_name))
            audio_bytes = open(by_name[choice], "rb").read()
            st.audio(audio_bytes, format="audio/wav")
    else:
        up = st.file_uploader("A short recording (wav, mp3, ogg, flac)", type=["wav", "mp3", "ogg", "flac"])
        if up is not None:
            audio_bytes = up.getvalue()
            st.audio(audio_bytes)

    if audio_bytes:
        try:
            x = au.load_audio(audio_bytes)
        except Exception as e:
            st.error(f"Couldn't read that audio file ({e}).")
        else:
            if LOCAL and mode == "🎙️ Record":      # keep live recordings for debugging
                d = os.path.join(COLLECT_DIR, "live_captures")
                os.makedirs(d, exist_ok=True)
                sf.write(os.path.join(d, f"cap_{int(time.time())}.wav"), x, au.SR)
            # Never let one failure blank the whole page: Streamlit runs every
            # tab in a single pass, so an exception here would also wipe out the
            # graphs and the Teach-it tab below.
            try:
                show_result(analyse(x, model, norm, threshold), threshold)
            except Exception as e:
                st.exception(e)

with tab_train:
    t = metrics["test"]
    n_pos, n_neg = t["hits"] + t["misses"], t["false_accepts"] + t["correct_rejects"]
    cs = metrics.get("continuous_speech")
    c = st.columns(4)
    c[0].metric("Vektoras caught (test)", f"{t['hits']} / {n_pos}")
    c[1].metric("False accepts (test)", f"{t['false_accepts']} / {n_neg}")
    if cs:
        c[2].metric("False fires per minute of talking", f"{cs['per_minute']:.1f}",
                    help="Measured by sliding the detector across held-out recordings of ordinary "
                         "conversation that the model never trained on.")
    else:
        c[2].metric("Training clips (after augmentation)",
                    f"{metrics.get('train_clips_after_augmentation', 0):,}",
                    help="Each recording is copied many times with shifts, volume changes, "
                         "different microphone tones and background noise, so a small dataset "
                         "teaches a lot more.")
    c[3].metric("Model size", f"{metrics['params']:,} params")
    st.caption("Test clips were locked away during training and scored once at the end. "
               "The test set is small, so treat these as rough numbers.")
    figs = [
        ("class_counts.png", "**The data.** Every clip is 1 second long. Blue = Vektora, orange = everything else. "
                             "*Soft negatives* are look-alike words (vector, victor) that teach the fine details."),
        ("examples.png", "**What the model sees.** Top: the raw waveform. Bottom: the log-mel spectrogram, the "
                         "'picture of the sound' that is the network's actual input."),
        ("learning_curves.png", "**Learning.** Loss (error) falling and accuracy rising, on training data (blue) and on "
                                "validation data the model doesn't learn from (orange). The dashed line is the epoch we kept."),
        ("confusion_matrix.png", "**Final exam.** Test clips the model never saw: hits, misses, false accepts, correct rejects."),
        ("score_distribution.png", "**Scores.** Each dot is one test clip. The dashed threshold sits in the gap "
                                   "between the two groups."),
        ("continuous_speech.png", "**The real test for a wake word.** The detector slid across minutes of ordinary "
                                  "conversation (Hindi, Kannada, English) that it never trained on. Every peak that "
                                  "stays below the dashed line is a false alarm avoided."),
    ]
    st.caption("This model is trained on the team's own recordings: 63 clips of “Vektora” from "
               "5–6 speakers, 27 deliberately confusing look-alike words (vector, victor), and "
               "noise generated in code so that silence isn't guessed at.")
    for name, caption in figs:
        path = os.path.join(HERE, "plots", name)
        if os.path.exists(path):
            st.markdown(caption)
            st.image(path, width="stretch")

with tab_how:
    st.markdown("""
#### The pipeline
1. **Record** — the browser captures 16,000 numbers per second from your microphone.
2. **Picture** — each 1-second window becomes a *log-mel spectrogram*: 98 time steps × 40 pitch bands,
   loudness in decibels, spaced the way human hearing is.
3. **Recognise** — a small *convolutional neural network* (the same family used for photos) slides
   3×3 pattern detectors over the picture: edges → sound pieces → the word.
4. **Decide** — the network outputs one number from 0 to 1. Above the threshold = "Vektora".

#### Why "Vektora"?
Three syllables give the model more to go on than a short word; the *k-t* sound is a sharp,
visible landmark; and it's not a word in any language, so nobody says it by accident.

#### The trade-off every wake word faces
- **False accept**: fires when nobody said it → annoying and a privacy problem.
- **Miss**: you said it and it didn't respond → you repeat yourself.

Raising the threshold trades misses for fewer false accepts. Wake-word products lean towards
fewer false accepts.

#### Honest limits
Trained on ~60 recordings of the word from 5–6 people, plus 27 look-alike words, so it works best
for voices like theirs and through a similar microphone. A product would use thousands of speakers
and measure *false accepts per hour* of real conversation — that is our next step.
""")

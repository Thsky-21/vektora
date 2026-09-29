"""
train.py — teach a small CNN to tell "Vektora" from everything else.

Run from the repo root (after prepare_data.py):
    .venv/Scripts/python.exe demo/train.py

Steps (demo.md §5.4–5.6):
  1. Load the train/val/test clips made by prepare_data.py.
  2. AUGMENT the training clips: make altered copies (shifted, louder/quieter,
     clipped, with background noise) so 45 recordings look like ~1000.
  3. Turn every clip into its log-mel picture and normalise it.
  4. Build the CNN and train it, watching validation loss (early stopping).
  5. Choose the detection THRESHOLD on the validation set.
  6. Score the untouched TEST set once: confusion matrix, precision, recall.
  7. Save the model + settings for the app, and the training graphs.
"""

import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")     # hide TensorFlow's info chatter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio_utils as au  # noqa: E402
import plots as P  # noqa: E402
from plots import plt  # noqa: E402

import tensorflow as tf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
PLOT_DIR = os.path.join(HERE, "plots")
SEED = 42
POS_COPIES, NEG_COPIES = 32, 6   # augmented copies per training clip (positives are rarer)
# Soft negatives ("vector", "victor") get as many copies as the positives.
# Without this the negative class is ~85% synthetic noise, so almost all of the
# model's effort goes into "word vs hiss" — which it finds easy — and hardly any
# into "Vektora vs vector", which is the ONLY distinction the demo is judged on.
# Oversampling the hard negatives puts the training effort where the boundary is.
SOFT_COPIES = 40
ROOM_AGC = False                 # see the comment in augment() — measured, not assumed
BROWSER_COPIES = 60              # browser-mic positives: few clips, but they match the live demo
GITHUB_ONLY = True               # must match prepare_data.py — see the comment there

rng = np.random.default_rng(SEED)
tf.keras.utils.set_random_seed(SEED)


# ---------------------------------------------------------------------------
# 1. Data augmentation (training clips only — never val/test)
# ---------------------------------------------------------------------------
def random_eq(x):
    """Pretend the clip was recorded on a different microphone.

    Every mic colours sound differently: phones cut the deep bass and roll off
    the treble, laptops sound thin, the project mic sounds like itself. Our
    "Vektora" clips come from the project mic while the everyday speech comes
    from a phone, so without this the network could cheat by recognising the
    MICROPHONE instead of the word (demo.md §0.5) — and then fail on the
    laptop mic used at the demo. Randomising the tone of every clip makes that
    shortcut useless."""
    n = len(x)
    spec = np.fft.rfft(x)
    f = np.maximum(np.fft.rfftfreq(n, 1 / au.SR), 1e-6)
    tilt_db = rng.uniform(-6, 6)                       # darker or brighter overall
    gain_db = tilt_db * np.log2(f / 20) / np.log2(8000 / 20)
    hp, lp = rng.uniform(50, 250), rng.uniform(5500, 8000)
    gain_db -= 12 * np.log2(np.maximum(hp / f, 1.0))   # roll off below hp
    gain_db -= 12 * np.log2(np.maximum(f / lp, 1.0))   # roll off above lp
    return np.fft.irfft(spec * 10 ** (gain_db / 20), n).astype(np.float32)


def random_speed(x):
    """Say the word faster or slower.

    Added 2026-09-12 after a live test: the detector caught "Vektora" when it
    was spoken LOUDLY and SLOWLY, and missed it when spoken at a normal pace.
    That is a straight gap in the training data — all 63 recordings are
    deliberate, careful pronunciations, and none of the other augmentations
    change the *rate* of speech, only where the word sits and how loud it is.

    This is "speed perturbation": resample the clip by a random factor, which
    shortens or lengthens the word AND shifts its pitch a little (a taller or
    shorter voice). Doing both at once is deliberate — it is the standard,
    well-tested trick in speech recognition, and it is one line of maths
    instead of a phase vocoder."""
    # Narrowed 2026-09-12 from 0.82-1.28: warping the word that far made the
    # model tolerant enough of duration and pitch that "vector" started to look
    # like a fast "Vektora". Enough pace tolerance for normal speech, not enough
    # to blur the one distinction that matters.
    rate = rng.uniform(0.9, 1.15)              # 0.9 = slower/deeper, 1.15 = faster/higher
    n = max(1, int(round(len(x) / rate)))
    y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return au.fix_length(y)                    # back to exactly 1 s, word centred


def random_room(x):
    """Make the clip sound like it crossed a room to reach a laptop.

    Every training recording is close-mic: the speaker's mouth is inches from a
    good microphone. At the demo the word travels a metre or two and arrives
    with reflections off the desk and walls smeared behind it. Convolving with
    a short burst of exponentially-decaying noise is the cheap standard
    approximation of that, and it costs one FFT."""
    t60 = rng.uniform(0.05, 0.35)                      # how long the echoes last
    n = int(t60 * au.SR)
    ir = rng.standard_normal(n).astype(np.float32) * np.exp(-np.arange(n) / (t60 * au.SR / 4))
    ir[0] = 1.0                                        # the direct sound
    wet = np.convolve(x, ir)[:len(x)]
    mix = rng.uniform(0.15, 0.6)                       # how far away the speaker is
    y = (1 - mix) * x + mix * wet / (np.max(np.abs(wet)) + 1e-9) * (np.max(np.abs(x)) + 1e-9)
    return y.astype(np.float32)


def random_agc(x):
    """Imitate the browser's automatic gain control.

    Chrome does not hand the page the raw microphone signal: it compresses it,
    lifting quiet parts and holding loud ones down, and leaves a noise floor
    behind. Our recordings are the opposite — deliberately loud and clipped.
    Without this the live audio is a kind of sound the model has never met."""
    env = np.abs(x)
    k = 400                                            # 25 ms smoothing, like a real AGC
    env = np.convolve(env, np.ones(k) / k, mode="same") + 1e-4
    strength = rng.uniform(0.3, 0.9)                   # 0 = untouched, 1 = fully levelled
    y = x / env ** strength
    y *= (np.max(np.abs(x)) + 1e-9) / (np.max(np.abs(y)) + 1e-9)
    y += rng.uniform(0.0005, 0.006) * rng.standard_normal(len(y)).astype(np.float32)
    return y.astype(np.float32)


def augment(x, noise_bank):
    """Return a randomly altered copy of a 1 s clip."""
    x = x.copy()
    # Domain augmentation, added 2026-09-12 after live false activations on
    # "vector"/"victor": on FILES the model separates them perfectly (0/27
    # fire), but through a browser microphone it does not. The word was never
    # the problem — the recording chain was. Both classes get this treatment,
    # so "sounds like a laptop mic" can never become a clue for either one.
    # Moderated 2026-09-12: at p=0.6/0.5 with full strength the model became so
    # conservative that live "Vektora" fell to 0.21-0.52. These milder settings
    # keep the domain robustness that rejects look-alikes through a laptop mic
    # without burying the keyword itself.
    # MEASURED 2026-09-12, demo day: room/AGC simulation at p=0.6/0.5 and at
    # p=0.35/0.3 both made live "Vektora" markedly WEAKER (1/6 and 2/6 of the
    # recorded live words fired, versus 4/6 without it) while only marginally
    # helping look-alike rejection, which was already clean on files. Guessing
    # at the live recording chain costs more than it buys; the fix is real
    # browser-mic recordings, not a better simulation. Disabled, kept for when
    # those recordings exist.
    if ROOM_AGC and rng.random() < 0.35:
        x = random_room(x)
    if ROOM_AGC and rng.random() < 0.3:
        x = random_agc(x)
    if rng.random() < 0.8:
        x = random_speed(x)
    if rng.random() < 0.7:
        x = random_eq(x)
    # (a) time shift up to ±0.15 s: live, the word won't sit where it did in
    #     the recordings (the app's sliding window can land anywhere)
    s = int(rng.integers(-2400, 2401))
    x = np.roll(x, s)
    if s > 0:
        x[:s] = 0
    elif s < 0:
        x[s:] = 0
    # (b) volume change: -20 dB (much quieter) to +6 dB (louder)
    x *= 10 ** (rng.uniform(-28, 6) / 20)      # widened 2026-09-12: quiet live speech was missed
    # (c) sometimes overdrive it so it clips, for EVERY class, so that
    #     "clipped" never becomes a clue for "Vektora" (demo.md §2, problem 2)
    if rng.random() < 0.3:
        x *= 10 ** (rng.uniform(3, 12) / 20)
    # (d) mix in background noise at a random signal-to-noise ratio
    if rng.random() < 0.8 and len(noise_bank):
        n = noise_bank[rng.integers(len(noise_bank))]
        snr_db = rng.uniform(5, 30)
        sig_rms = np.sqrt(np.mean(x ** 2)) + 1e-9
        n_rms = np.sqrt(np.mean(n ** 2)) + 1e-9
        x = x + n * (sig_rms / n_rms) * 10 ** (-snr_db / 20)
    return np.clip(x, -1, 1).astype(np.float32)


def spec_augment(f):
    """SpecAugment: blank one random strip of time and one of pitch, so the
    model can't depend on any single tiny detail. f is already normalised,
    so 0 = 'average'."""
    f = f.copy()
    t = int(rng.integers(0, 11))
    t0 = int(rng.integers(0, au.N_FRAMES - t + 1))
    f[t0:t0 + t, :] = 0
    b = int(rng.integers(0, 6))
    b0 = int(rng.integers(0, au.N_MELS - b + 1))
    f[:, b0:b0 + b] = 0
    return f


# ---------------------------------------------------------------------------
# 2. The model (demo.md §5.4) — ~24k parameters
# ---------------------------------------------------------------------------
def build_model():
    L = tf.keras.layers
    return tf.keras.Sequential([
        L.Input(shape=(au.N_FRAMES, au.N_MELS, 1)),
        # Block 1: 16 small 3x3 pattern detectors -> edges, stripes
        L.Conv2D(16, 3, padding="same", use_bias=False), L.BatchNormalization(), L.ReLU(),
        L.MaxPooling2D(2),                     # halve the picture: 98x40 -> 49x20
        # Block 2: 32 detectors -> pieces of sounds (the k-t burst, vowels)
        L.Conv2D(32, 3, padding="same", use_bias=False), L.BatchNormalization(), L.ReLU(),
        L.MaxPooling2D(2),                     # 49x20 -> 24x10
        # Block 3: 64 detectors -> word-level combinations
        L.Conv2D(64, 3, padding="same", use_bias=False), L.BatchNormalization(), L.ReLU(),
        # "Was each pattern present ANYWHERE?" -> 64 numbers, position-free
        L.GlobalAveragePooling2D(),
        L.Dropout(0.3),                        # switch off 30% at random while training
        L.Dense(1, activation="sigmoid"),      # one score, 0..1 = P(Vektora)
    ], name="vektora_cnn")


def choose_threshold(y, p):
    """Pick the detection cutoff on the VALIDATION set (never the test set).

    Wake words prefer few false accepts, so first try: just above the highest-
    scoring negative (zero false accepts on validation). If that would miss
    more than 20% of real Vektoras, fall back to the cutoff with the best
    balance of catching Vektoras and rejecting everything else."""
    neg_max = float(p[y == 0].max()) if (y == 0).any() else 0.0
    t = float(np.clip(neg_max + 0.02, 0.5, 0.95))
    recall = float((p[y == 1] >= t).mean())
    if recall >= 0.8:
        return t, f"just above the highest-scoring validation negative ({neg_max:.2f}); catches {recall:.0%} of validation Vektoras"
    grid = np.linspace(0.5, 0.95, 46)
    bal = [((p[y == 1] >= g).mean() + (p[y == 0] < g).mean()) / 2 for g in grid]
    t = float(grid[int(np.argmax(bal))])
    return t, f"zero-false-accept cutoff ({neg_max + 0.02:.2f}) would miss too many, so used best balance instead"


def false_fires_per_minute(model, norm, threshold):
    """The number a wake word is really judged by: how often does it fire at
    speech that ISN'T the keyword?

    Runs the app's sliding window over the held-out last 15% of each long
    talking recording — the part prepare_data.py kept out of training — and
    counts separate firings (a run of consecutive windows above the threshold
    is one firing, not twenty)."""
    import glob
    events, minutes, peak = 0, 0.0, 0.0
    trace_t, trace_s = [], []                          # for the graph
    for f in (glob.glob(os.path.join(HERE, "data", "other_speech", "*"))
              + glob.glob(os.path.join(HERE, "data", "live_other", "*"))):
        x = au.peak_normalize(au.load_audio(f))
        block = x[int(len(x) * 0.85):]                 # same split as prepare_data.py
        if len(block) < au.CLIP:
            continue
        windows = [w for _, w in au.sliding_windows(block, hop_s=0.1)]
        feats = np.stack([norm(au.logmel(w)) for w in windows])[..., None]
        scores = model.predict(feats, verbose=0).ravel()
        fires = scores >= threshold
        events += int(fires[0]) + int(np.sum(fires[1:] & ~fires[:-1]))
        trace_t.append(np.arange(len(scores)) * 0.1 + minutes * 60 + 0.5)
        trace_s.append(scores)
        minutes += len(block) / au.SR / 60
        peak = max(peak, float(scores.max()))
    if minutes == 0:
        return None
    return {"per_minute": events / minutes, "events": events,
            "minutes": minutes, "highest_score": peak,
            "trace_t": np.concatenate(trace_t), "trace_s": np.concatenate(trace_s)}


def main():
    d = np.load(os.path.join(HERE, "data", "prepared.npz"))
    X, y, src = ({s: d[f"{k}_{s}"] for s in ("train", "val", "test")} for k in ("X", "y", "src"))
    print(f"loaded: train {len(y['train'])}, val {len(y['val'])}, test {len(y['test'])} clips")

    # ---- augmentation ------------------------------------------------------
    # Noise bank: what gets mixed into clips as background. TRAINING clips
    # only, so nothing leaks from val/test.
    #
    # TRIED AND REVERTED (2026-09-12): also mixing browser-mic speech in here,
    # at SNR 3-25 dB, to make a noisy background normal for positives. It made
    # live detection WORSE (0/6 browser-mic Vektoras fired, versus 2/6 before):
    # burying an already-scarce positive under babble destroys more evidence
    # than the extra realism buys. More real live recordings is the fix.
    is_noise = np.array(["noise" in s for s in src["train"]])
    noise_bank = X["train"][is_noise]
    print(f"noise bank: {len(noise_bank)} clips mixed in as background")
    waves, labels = [], []
    for x, lab, s in zip(X["train"], y["train"], src["train"]):
        waves.append(x)
        labels.append(lab)
        # Browser-mic clips are few but they are the closest match to the live
        # demo (Chrome applies its own noise suppression and gain), so each one
        # gets many more augmented variations than a project-mic clip.
        if lab == 1:
            copies = BROWSER_COPIES if "browser mic" in s else POS_COPIES
        else:
            copies = SOFT_COPIES if "soft negative" in s else NEG_COPIES
        for _ in range(copies):
            waves.append(augment(x, noise_bank))
            labels.append(lab)
    labels = np.array(labels, dtype=np.float32)
    print(f"after augmentation: {len(labels)} training clips "
          f"({int(labels.sum())} Vektora, {int((labels == 0).sum())} not)")

    # ---- features ----------------------------------------------------------
    print("computing log-mel pictures...")
    F_train = np.stack([au.logmel(w) for w in waves])
    norm = au.Normalizer.fit(F_train)          # statistics from TRAINING data only
    F_train = np.stack([spec_augment(f) for f in norm(F_train)])
    F_val = norm(np.stack([au.logmel(w) for w in X["val"]]))
    F_test = norm(np.stack([au.logmel(w) for w in X["test"]]))

    # Class weights: make a mistake on the rarer class count proportionally more
    n_pos, n_neg = labels.sum(), (labels == 0).sum()
    class_weight = {0: len(labels) / (2 * n_neg), 1: len(labels) / (2 * n_pos)}

    # ---- train -------------------------------------------------------------
    model = build_model()
    model.summary(print_fn=lambda s: print("  " + s))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    # Early stopping: stop when validation loss hasn't improved for 12 epochs,
    # and roll back to the best epoch (protects against overfitting).
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12,
                                             restore_best_weights=True)
    # Halve the learning rate (smaller nudges) whenever validation loss stalls
    # for 3 epochs: big steps early to learn fast, small steps later to settle.
    calm = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                patience=3, min_lr=1e-5)
    hist = model.fit(F_train[..., None], labels, epochs=60, batch_size=32,
                     validation_data=(F_val[..., None], y["val"].astype(np.float32)),
                     class_weight=class_weight, callbacks=[early, calm], verbose=2)
    h = {k: v for k, v in hist.history.items() if k != "lr"}
    best_epoch = int(np.argmin(h["val_loss"])) + 1

    # ---- threshold (validation) and final score (test) ---------------------
    p_val = model.predict(F_val[..., None], verbose=0).ravel()
    threshold, why = choose_threshold(y["val"], p_val)
    p_test = model.predict(F_test[..., None], verbose=0).ravel()
    pred = (p_test >= threshold).astype(int)
    yt = y["test"].astype(int)
    tp = int(((pred == 1) & (yt == 1)).sum()); fn = int(((pred == 0) & (yt == 1)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum()); tn = int(((pred == 0) & (yt == 0)).sum())
    per_source = {s: {"n": int((src["test"] == s).sum()),
                      "correct": int(((pred == yt) & (src["test"] == s)).sum())}
                  for s in sorted(set(src["test"]))}
    metrics = {
        "threshold": threshold, "threshold_reason": why,
        "test": {"hits": tp, "misses": fn, "false_accepts": fp, "correct_rejects": tn,
                 "accuracy": (tp + tn) / len(yt),
                 "precision": tp / (tp + fp) if tp + fp else None,
                 "recall": tp / (tp + fn) if tp + fn else None,
                 "per_source": per_source},
        "continuous_speech": None,          # filled in below (arrays stripped out)
        "best_epoch": best_epoch, "epochs_run": len(h["loss"]),
        "params": int(model.count_params()),
        "train_clips_after_augmentation": int(len(labels)),
    }

    # ROLLBACK 2026-09-12: the model is trained on the GitHub clips only, so
    # there is no held-out talking recording that belongs to this dataset. The
    # "false fires per minute" measurement is only honest when the talking is
    # part of the same prepared split, so it is skipped (see prepare_data.py's
    # GITHUB_ONLY). Flip both flags together to bring it back.
    cs_full = None if GITHUB_ONLY else false_fires_per_minute(model, norm, threshold)
    metrics["continuous_speech"] = ({k: v for k, v in cs_full.items() if not k.startswith("trace")}
                                    if cs_full else None)

    print(f"\nthreshold = {threshold:.2f}  ({why})")
    print(f"TEST: {tp} hits, {fn} misses, {fp} false accepts, {tn} correct rejects "
          f"-> accuracy {metrics['test']['accuracy']:.1%}")
    for s, v in per_source.items():
        print(f"   {s:<24} {v['correct']}/{v['n']} correct")
    cs = metrics["continuous_speech"]
    if cs:
        print(f"held-out continuous talking: {cs['events']} false fires in {cs['minutes'] * 60:.0f} s "
              f"= {cs['per_minute']:.1f}/minute (highest score {cs['highest_score']:.2f})")

    # ---- save everything the app needs ------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "vektora.keras"))
    norm.save(os.path.join(MODEL_DIR, "norm.json"))
    json.dump(metrics, open(os.path.join(MODEL_DIR, "metrics.json"), "w"), indent=2)
    json.dump({k: [float(v) for v in vals] for k, vals in h.items()},
              open(os.path.join(MODEL_DIR, "history.json"), "w"))

    # ---- graphs ------------------------------------------------------------
    os.makedirs(PLOT_DIR, exist_ok=True)
    ep = np.arange(1, len(h["loss"]) + 1)

    # Learning curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, name in ((axes[0], "loss", "Loss (lower = better)"),
                          (axes[1], "accuracy", "Accuracy (higher = better)")):
        ax.plot(ep, h[key], color=P.BLUE, label="training")
        ax.plot(ep, h["val_" + key], color=P.ORANGE, label="validation")
        ax.axvline(best_epoch, color=P.INK2, linestyle="--", linewidth=1)
        ax.text(best_epoch, ax.get_ylim()[1], " kept this epoch", color=P.INK2, fontsize=8.5, va="top")
        ax.set_xlabel("epoch (one pass over the training data)")
        ax.set_title(name)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "learning_curves.png"), dpi=150)
    plt.close(fig)

    # Confusion matrix
    cm = np.array([[tp, fn], [fp, tn]])
    names = np.array([["hit", "miss"], ["false accept", "correct reject"]])
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    ax.imshow(cm, cmap=P.ENERGY, vmin=0, vmax=max(cm.max(), 1))
    for i in range(2):
        for j in range(2):
            dark = cm[i, j] > cm.max() * 0.55
            ax.text(j, i, f"{cm[i, j]}\n{names[i, j]}", ha="center", va="center",
                    color="white" if dark else P.INK, fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1], ["said Vektora", "said no"])
    ax.set_yticks([0, 1], ["was Vektora", "was not"])
    ax.set_xlabel(f"model's answer (threshold {threshold:.2f})"); ax.set_ylabel("truth")
    ax.grid(False)
    ax.set_title(f"Test set: {len(yt)} unseen clips")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # Score distribution: every test clip is one dot
    fig, ax = plt.subplots(figsize=(8, 3.0))
    for row, (lab, colour, name) in enumerate(((0, P.ORANGE, "not Vektora"), (1, P.BLUE, "Vektora"))):
        s = p_test[yt == lab]
        jitter = rng.uniform(-0.18, 0.18, len(s))
        ax.scatter(s, row + jitter, s=60, color=colour, edgecolor=P.SURFACE, linewidth=2, zorder=3)
        ax.text(-0.02, row, f"{name} ({len(s)})", ha="right", va="center", color=P.INK)
    ax.axvline(threshold, color=P.INK2, linestyle="--", linewidth=1.2)
    ax.text(threshold, 1.55, f" threshold {threshold:.2f}", color=P.INK2, fontsize=8.5, va="top")
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, 1.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("model's score  P(Vektora)")
    ax.set_title("Every test clip's score: right of the line = detected")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "score_distribution.png"), dpi=150)
    plt.close(fig)

    # Score across held-out continuous talking: the wake-word test that counts
    if cs_full:
        fig, ax = plt.subplots(figsize=(10, 3.0))
        ax.fill_between(cs_full["trace_t"], cs_full["trace_s"], color=P.ORANGE, alpha=0.15, linewidth=0)
        ax.plot(cs_full["trace_t"], cs_full["trace_s"], color=P.ORANGE, linewidth=1.5)
        ax.axhline(threshold, color=P.INK2, linestyle="--", linewidth=1.2)
        ax.text(0, threshold + 0.03, f"threshold {threshold:.2f}", color=P.INK2, fontsize=8.5)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("time (s) through held-out conversation the model never trained on")
        ax.set_ylabel("P(Vektora)")
        ax.set_title(f"Ordinary talking, {cs_full['minutes'] * 60:.0f} s: "
                     f"{cs_full['events']} false fires ({cs_full['per_minute']:.1f}/minute)")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "continuous_speech.png"), dpi=150)
        plt.close(fig)

    print("\nsaved: model/vektora.keras, model/norm.json, model/metrics.json, "
          "plots/learning_curves.png, plots/confusion_matrix.png, plots/score_distribution.png")


if __name__ == "__main__":
    main()

"""
config.py — every path and knob for the prototype training pipeline, in one place.

Nothing here is a feature-extraction constant: those live in features.py,
because they form the PARITY CONTRACT with the ESP32 C code and must never be
changed casually. Everything in THIS file can be changed and retrained freely.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)                              # repo root

# ---------------------------------------------------------------------------
# Data locations
# ---------------------------------------------------------------------------
# The team dataset ships in dataset/ (a copy of the Thsky-21/Data repo).
# Override with the environment variable VEKTORA_DATA.
DATA_ROOT = os.environ.get("VEKTORA_DATA", os.path.join(PROTO, "dataset"))

# Folder in DATA_ROOT -> (model class, source tag).
# The source tag is kept separately from the class so that evaluation can
# report "how many HARD negatives fired" on its own, even though hard and
# generic negatives are the same class to the model.
FOLDERS = {
    "positive":            ("keyword", "positive"),
    "hard_negative":       ("other",   "hard"),
    "generic_negative":    ("other",   "generic"),
    "background_negative": ("noise",   "noise"),
    # 2026-09-28: Akash's own recording session. Folder membership IS the
    # label (user's instruction) — positive_akash is all "Vektora" (incl.
    # deliberately similar-sounding pronunciations of it), negative_akash is
    # look-alike/acoustically-similar-but-different words (victor, vector,
    # victoria, ...). Kept as separate source tags, not merged into
    # "positive"/"hard", so SOURCE_MIX below can give Akash's own voice a
    # deliberate boost without disturbing the existing keyword/other/noise
    # class balance.
    "positive_akash":      ("keyword", "akash_pos"),
    "negative_akash":      ("other",   "akash_neg"),
}

# Long continuous recordings (minutes each), cut into 1 s windows by
# manifest.py. One sub-folder per class. Drop WAVs here; any sample rate works.
#   dataset/long/noise/   <- INMP441 background recordings (fan, lab, ...)
#   dataset/long/other/   <- INMP441 conversation with no "Vektora" in it
LONG_DIR = os.path.join(PROTO, "dataset", "long")

# External (non-INMP441) noise, e.g. ESC-50 / DEMAND / MUSAN. Any WAVs, any
# sample rate, any length, nested folders are fine. Used ONLY as background to
# mix into training clips, plus a small capped share of the noise class —
# see augment.py for why.
SR_HOP_NOISE = 16000             # long noise: back-to-back 1 s windows
SR_HOP_OTHER = 8000              # long conversation: 0.5 s hop (more variety)

EXTERNAL_NOISE_DIR = os.path.join(PROTO, "dataset", "external_noise")

# Outputs
BUILD_DIR = os.path.join(HERE, "runs")                     # manifest, audit, per-run archives
MANIFEST = os.path.join(BUILD_DIR, "manifest.csv")


# The manifest stores paths relative to the repo root ("dataset/positive/x.wav")
# so a fresh clone can retrain without editing it. Absolute paths still work.
def rel(path: str) -> str:
    try:
        r = os.path.relpath(path, PROTO)
    except ValueError:                                     # other drive (Windows)
        return path
    return path if r.startswith("..") else r.replace(os.sep, "/")


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROTO, path)
MODEL_DIR = os.path.join(HERE, "model")                    # trained model + norm + report

# ---------------------------------------------------------------------------
# Classes — order is the model's output order. LOCKED (CLAUDE.md §2).
# ---------------------------------------------------------------------------
CLASSES = ["keyword", "other", "noise"]

# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
SEED = 42
SPLIT = (0.70, 0.15, 0.15)       # train / val / test, by GROUP (see manifest.py)
TAKE_BLOCK = 10                  # consecutive take numbers grouped together

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
EPOCH_SIZE = 4000                # freshly augmented clips drawn per epoch
BATCH = 32
EPOCHS = 60
PATIENCE = 10                    # early stopping on validation loss
LR = 1e-3

# Share of each epoch drawn from each SOURCE. Deliberately not proportional to
# how many files exist: demo.md §15.4 measured that putting the effort on the
# look-alike boundary ("Vektora" vs "vector") is what makes them stop firing.
#
# Sources that don't exist yet (e.g. no long recordings) are skipped and the
# rest re-scaled. Noise sources share one budget ("noise" + "noise_long").
# 2026-09-28: reallocated WITHIN each class's existing budget to give Akash's
# own recordings (positive_akash / negative_akash) a real boost — this is
# the "works on my voice instantly" requirement — without changing the
# overall keyword=0.35 / other=0.50 / noise=0.15 class balance the model,
# threshold rule and prior evaluations were tuned against. keyword budget:
# 0.20 (positive, ~6 other speakers) + 0.15 (akash_pos) = 0.35 unchanged.
# other budget: 0.15 (hard) + 0.15 (akash_neg) + 0.08 (generic) + 0.07 (gsc)
# + 0.05 (conversation) = 0.50 unchanged. noise budget: 0.15 unchanged.
#
# 2026-09-28, three retrains on this axis (hard/akash_neg split), all else
# (incl. SEED, so each is fully reproducible, not noise) fixed. Result —
# akash_pos recall / akash_neg false-accept rate / positive-only recall:
#   hard=0.20 akash_neg=0.10 (v1): recall 80% (16/20)  FA 31.6% (6/19)  positive 84.2% (16/19)
#   hard=0.15 akash_neg=0.15 (v2): recall 50% (10/20)  FA  5.3% (1/19)  positive 84.2% (16/19)
#   hard=0.18 akash_neg=0.12 (v3): recall 65% (13/20)  FA 31.6% (6/19)  positive 63.2% (12/19)
# v3 is dominated by v1 (worse on every axis) — small-mix-change effects on
# this dataset are NOT a smooth interpolation between v1 and v2's endpoints,
# they reshape the whole training trajectory (v3 early-stopped at epoch 20
# vs v1/v2's ~55). v1 vs v2 is a real recall-vs-false-accept trade-off; v1
# was kept because the user's stated priority is live-demo recall on his own
# voice, and v1 matches the pre-Akash baseline's positive-only recall
# exactly (84.2%, same as Run 3). SETTLED at v1's weights — do not keep
# iterating this knob without new akash_neg recordings; three points is
# enough to see it isn't a simple gradient, and going further burns time
# chasing an effect that a 19-sample test bucket can't resolve precisely
# anyway. See docs/current_state.md for the full writeup.
SOURCE_MIX = {
    "positive":     0.20,   # keyword — the original ~6-speaker set
    "akash_pos":    0.15,   # keyword — Akash's own "Vektora" (demo voice)
    "hard":         0.20,   # other — vector, victor, ... (the boundary that matters)
    "akash_neg":    0.10,   # other — Akash's own look-alike words (known false-accept risk — see docs/current_state.md)
    "generic":      0.08,   # other — team's own INMP441 everyday words
    "gsc":          0.07,   # other — Google Speech Commands (NOT INMP441)
    "conversation": 0.05,   # other — long INMP441 talking, if recorded
    "noise":        0.15,   # noise — INMP441 clips + long recordings + capped external
}

# Cap on external (non-INMP441) clips inside the noise class. See augment.py.
EXTERNAL_NOISE_CLASS_SHARE = 0.30

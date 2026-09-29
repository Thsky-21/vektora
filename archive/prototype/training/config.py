"""
config.py — every path and knob for the prototype training pipeline, in one place.

Nothing here is a feature-extraction constant: those live in features.py,
because they form the PARITY CONTRACT with the ESP32 C code and must never be
changed casually. Everything in THIS file can be changed and retrained freely.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)                              # prototype/

# ---------------------------------------------------------------------------
# Data locations
# ---------------------------------------------------------------------------
# The team dataset lives OUTSIDE this repo (its own git repo, Thsky-21/Data).
# Override with the environment variable VEKTORA_DATA if it moves.
DATA_ROOT = os.environ.get("VEKTORA_DATA", r"C:\Users\User\documents\data")

# Folder in DATA_ROOT -> (model class, source tag).
# The source tag is kept separately from the class so that evaluation can
# report "how many HARD negatives fired" on its own, even though hard and
# generic negatives are the same class to the model.
FOLDERS = {
    "positive":            ("keyword", "positive"),
    "hard_negative":       ("other",   "hard"),
    "generic_negative":    ("other",   "generic"),
    "background_negative": ("noise",   "noise"),
}

# Long continuous recordings (minutes each), cut into 1 s windows by
# manifest.py. One sub-folder per class. Drop WAVs here; any sample rate works.
#   prototype/data/long/noise/   <- INMP441 background recordings (fan, lab, ...)
#   prototype/data/long/other/   <- INMP441 conversation with no "Vektora" in it
LONG_DIR = os.path.join(PROTO, "data", "long")

# External (non-INMP441) noise, e.g. ESC-50 / DEMAND / MUSAN. Any WAVs, any
# sample rate, any length, nested folders are fine. Used ONLY as background to
# mix into training clips, plus a small capped share of the noise class —
# see augment.py for why.
SR_HOP_NOISE = 16000             # long noise: back-to-back 1 s windows
SR_HOP_OTHER = 8000              # long conversation: 0.5 s hop (more variety)

EXTERNAL_NOISE_DIR =os.path.join(PROTO, "data", "external_noise")

# Outputs
BUILD_DIR = os.path.join(PROTO, "data", "build")           # manifest, caches (git-ignored)
MANIFEST = os.path.join(BUILD_DIR, "manifest.csv")
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
SOURCE_MIX = {
    "positive":     0.35,   # keyword
    "hard":         0.30,   # other — vector, victor, ... (the boundary that matters)
    "generic":      0.08,   # other — team's own INMP441 everyday words
    "gsc":          0.07,   # other — Google Speech Commands (NOT INMP441)
    "conversation": 0.05,   # other — long INMP441 talking, if recorded
    "noise":        0.15,   # noise — INMP441 clips + long recordings + capped external
}

# Cap on external (non-INMP441) clips inside the noise class. See augment.py.
EXTERNAL_NOISE_CLASS_SHARE = 0.30

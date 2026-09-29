"""
train.py — train the 3-class keyword model and score it honestly.

Run (from prototype/training, after manifest.py):
    ../../.venv/Scripts/python.exe train.py

Steps:
  1. Load the clips listed in manifest.csv (originals only).
  2. Build the noise bank from TRAINING noise only (+ external noise if present).
  3. Fit the normaliser on augmented training audio (the levels the model
     will really see), save norm.json.
  4. Train on freshly augmented clips every epoch, balanced by source.
  5. Choose the keyword threshold on VALIDATION.
  6. Score TEST once. Report per source, so hard-negative firings are visible.
  7. Save model.keras, norm.json, report.json, test_scores.csv to model/.
"""

import csv
import json
import os
import sys
from collections import Counter, defaultdict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import augment as A  # noqa: E402
import config as C  # noqa: E402
import features as F  # noqa: E402
import model as M  # noqa: E402

import tensorflow as tf  # noqa: E402

rng = np.random.default_rng(C.SEED)
tf.keras.utils.set_random_seed(C.SEED)
KW = C.CLASSES.index("keyword")
NOISE = C.CLASSES.index("noise")
NOISE_SOURCES = {"noise", "noise_long"}


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_rows():
    with open(C.MANIFEST) as f:
        rows = list(csv.DictReader(f))
    cache = {}
    for r in rows:
        if r["path"] not in cache:
            cache[r["path"]] = F.load_wav(r["path"])
        x = cache[r["path"]]
        off = int(r["offset"])
        if off or len(x) > F.CLIP * 1.5:                   # a window of a long recording
            r["audio"] = x[off:off + F.CLIP]
        else:                                              # a 1 s clip (some are short)
            r["audio"] = F.fix_length(x)                   # val/test + noise use this
            r["raw"] = x                                   # speech training keeps true length:
                                                           # augment pads with room tone, not zeros
        r["y"] = C.CLASSES.index(r["cls"])
    return rows


# ---------------------------------------------------------------------------
# 4. The training stream: a new random epoch every time
# ---------------------------------------------------------------------------
class TrainStream(tf.keras.utils.Sequence):
    def __init__(self, rows, aug, norm, ext_share):
        self.by_src = defaultdict(list)
        for r in rows:
            self.by_src["noise" if r["source"] in NOISE_SOURCES else r["source"]].append(r)
        mix = {s: w for s, w in C.SOURCE_MIX.items() if self.by_src.get(s)}
        tot = sum(mix.values())
        self.sources = list(mix)
        self.p = np.array([mix[s] / tot for s in self.sources])
        self.aug, self.norm, self.ext_share = aug, norm, ext_share
        print("  epoch mix:", {s: f"{p:.0%}" for s, p in zip(self.sources, self.p)})
        self.on_epoch_end()

    def __len__(self):
        return C.EPOCH_SIZE // C.BATCH

    def on_epoch_end(self):
        self.plan = rng.choice(len(self.sources), size=C.EPOCH_SIZE, p=self.p)

    def sample(self, src):
        if src == "noise":
            if self.ext_share and rng.random() < self.ext_share:
                return self.aug.noise(self.aug.nb.external_noise_example()), NOISE
            r = self.by_src["noise"][rng.integers(len(self.by_src["noise"]))]
            return self.aug.noise(r["audio"]), r["y"]
        pool = self.by_src[src]
        r = pool[rng.integers(len(pool))]
        return self.aug.speech(r.get("raw", r["audio"])), r["y"]

    def __getitem__(self, i):
        xs, ys = [], []
        for k in self.plan[i * C.BATCH:(i + 1) * C.BATCH]:
            x, y = self.sample(self.sources[k])
            xs.append(self.aug.spec(self.norm(F.logmel(x))))
            ys.append(y)
        return np.stack(xs)[..., None], np.array(ys)


def featurize(rows, norm):
    return np.stack([norm(F.logmel(r["audio"])) for r in rows])[..., None]


# ---------------------------------------------------------------------------
# 5. Threshold on validation
# ---------------------------------------------------------------------------
def choose_threshold(p_kw, is_kw):
    """Just above the highest-scoring validation non-keyword (zero false
    accepts on validation), floor 0.5 — unless that loses more than 20% of
    validation keywords, then the best balanced accuracy on a 0.5-0.95 grid."""
    neg_max = float(p_kw[~is_kw].max())
    t = float(np.clip(neg_max + 0.02, 0.5, 0.95))
    if (p_kw[is_kw] >= t).mean() >= 0.8:
        return t, f"just above highest validation non-keyword ({neg_max:.3f})"
    grid = np.linspace(0.5, 0.95, 46)
    bal = [((p_kw[is_kw] >= g).mean() + (p_kw[~is_kw] < g).mean()) / 2 for g in grid]
    return float(grid[int(np.argmax(bal))]), f"best balance (zero-FA cutoff {neg_max + 0.02:.3f} lost >20% keywords)"


def per_source(rows, p_kw, thr):
    out = {}
    for s in sorted({r["source"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["source"] == s]
        sc = p_kw[idx]
        fired = int((sc >= thr).sum())
        out[s] = dict(n=len(idx), fired=fired, max_score=round(float(sc.max()), 3),
                      median_score=round(float(np.median(sc)), 3))
    return out


def main():
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    rows = load_rows()
    split = defaultdict(list)
    for r in rows:
        split[r["split"]].append(r)
    print("clips per split:", {k: len(v) for k, v in split.items()})

    # 2. noise bank — training-split INMP441 noise only, never val/test
    nb = A.NoiseBank([r["audio"] for r in split["train"] if r["cls"] == "noise"],
                     external_dir=C.EXTERNAL_NOISE_DIR, rng=rng)
    print(f"noise bank: {len(nb.mic)} INMP441 snippets, {len(nb.ext)} external snippets")
    aug = A.Augmenter(nb, rng)
    ext_share = C.EXTERNAL_NOISE_CLASS_SHARE if nb.ext else 0.0

    # 3. normaliser from augmented training audio
    tmp = TrainStream(split["train"], aug, lambda f: f, ext_share)
    raw = []
    for _ in range(1500):
        x, _ = tmp.sample(tmp.sources[rng.choice(len(tmp.sources), p=tmp.p)])
        raw.append(F.logmel(x))
    norm = F.Normalizer.fit(np.stack(raw))
    norm.save(os.path.join(C.MODEL_DIR, "norm.json"))

    stream = TrainStream(split["train"], aug, norm, ext_share)
    Xv, yv = featurize(split["val"], norm), np.array([r["y"] for r in split["val"]])
    Xt, yt = featurize(split["test"], norm), np.array([r["y"] for r in split["test"]])

    # 4. train
    model = M.build(len(C.CLASSES))
    model.compile(optimizer=tf.keras.optimizers.Adam(C.LR),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print(f"model: {model.count_params():,} parameters")
    hist = model.fit(
        stream, validation_data=(Xv, yv), epochs=C.EPOCHS, verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=C.PATIENCE, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-5),
        ])
    model.save(os.path.join(C.MODEL_DIR, "model.keras"))

    # 5. threshold on validation
    pv = model.predict(Xv, verbose=0)
    thr, why = choose_threshold(pv[:, KW], yv == KW)

    # 6. test, once
    pt = model.predict(Xt, verbose=0)
    pred = pt.argmax(1)
    cm = np.zeros((3, 3), int)
    for a, b in zip(yt, pred):
        cm[a, b] += 1
    fire = pt[:, KW] >= thr
    is_kw = yt == KW
    tp, fn = int((fire & is_kw).sum()), int((~fire & is_kw).sum())
    fa, tn = int((fire & ~is_kw).sum()), int((~fire & ~is_kw).sum())
    best = int(np.argmin(hist.history["val_loss"]))
    report = dict(
        parameters=model.count_params(),
        epochs_run=len(hist.history["loss"]), best_epoch=best + 1,
        best_val_loss=round(float(hist.history["val_loss"][best]), 4),
        best_val_acc=round(float(hist.history["val_accuracy"][best]), 4),
        threshold=round(thr, 3), threshold_rule=why,
        test_accuracy_3class=round(float((pred == yt).mean()), 4),
        test_confusion_rows_true_cols_pred=dict(classes=C.CLASSES, matrix=cm.tolist()),
        test_keyword_at_threshold=dict(hits=tp, misses=fn, false_accepts=fa, correct_rejects=tn,
                                       recall=round(tp / max(tp + fn, 1), 4),
                                       precision=round(tp / max(tp + fa, 1), 4)),
        test_by_source=per_source(split["test"], pt[:, KW], thr),
        val_by_source=per_source(split["val"], pv[:, KW], thr),
        external_noise_snippets=len(nb.ext),
        history={k: [round(float(v), 4) for v in vs] for k, vs in hist.history.items()},
    )
    with open(os.path.join(C.MODEL_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(C.MODEL_DIR, "test_scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "source", "true", "pred", "p_keyword", "p_other", "p_noise"])
        for r, p, q in zip(split["test"], pt, pred):
            w.writerow([os.path.basename(r["path"]), r["source"], r["cls"], C.CLASSES[q], *np.round(p, 4)])

    print("\n=== RESULT (test set, never used for any decision) ===")
    print(f"best epoch {best + 1}/{len(hist.history['loss'])}, threshold {thr:.3f} ({why})")
    print(f"3-class accuracy {report['test_accuracy_3class']:.1%}")
    print("confusion (rows = true, cols = predicted):", C.CLASSES)
    for c, row in zip(C.CLASSES, cm):
        print(f"  {c:<8}{row}")
    print("keyword @ threshold:", report["test_keyword_at_threshold"])
    for s, d in report["test_by_source"].items():
        print(f"  {s:<10} n={d['n']:<4} fired={d['fired']:<3} max={d['max_score']:.3f} median={d['median_score']:.3f}")


if __name__ == "__main__":
    main()

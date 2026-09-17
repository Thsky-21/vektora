# Snapshot: run3-baseline

A frozen copy of the Run 3 model (clean-dataset baseline, 2026-09-17).
Nothing in this folder is ever overwritten by `train.py`, which only writes to
`training/model/`. Git tag: **`run3-baseline`**.

## What it is

| | |
|---|---|
| Model | 3-class CNN (keyword / other / noise), 23,827 params |
| Threshold | 0.50 on P(keyword), chosen on validation |
| Best epoch | 40 / 50 · val acc 0.869 · val loss 0.2957 |
| Clean test (113 clips) | 3-class acc 93.8% · keyword recall 17/19 · false accepts 5/94 |
| Data | `C:\Users\User\documents\data`, data-repo commit `115e37b`, 14 exclusions (`exclusions.csv`) |
| Split | `manifest.csv` (the exact train/val/test assignment used) |
| Environment | Python 3.10.7, TensorFlow 2.15.1, Keras 2.15.0, librosa 0.11.0, numpy 1.26.4 |

`model.keras` sha256: `a6edf07e0d78a6b077ea66d8fc8001a139fd352e88dc4e9c4a7da07d2eb4c125`
`norm.json`   sha256: `dc3b158b253094ef91d62ffec2e1f22dec34dc1c17e84422373794413eae8db3`

## Files

- `model.keras` — trained weights
- `norm.json` — the 40 + 40 per-band normalisation constants. **The model is useless
  without them.**
- `report.json`, `test_scores.csv`, `train_log.txt` — metrics, per-clip scores, training log
- `manifest.csv`, `exclusions.csv` — the exact data split and exclusions

The feature code the model expects is `training/features.py` **at tag
`run3-baseline`**. If `features.py` changes later, use the tagged version with this
model.

## Restoring it

Make it the active model again (the tester and scripts read `training/model/`):

```powershell
# from the repo root
Copy-Item prototype/training/snapshots/run3-baseline/model.keras, `
          prototype/training/snapshots/run3-baseline/norm.json, `
          prototype/training/snapshots/run3-baseline/report.json, `
          prototype/training/snapshots/run3-baseline/test_scores.csv `
          prototype/training/model/ -Force
```

Or restore the entire repo exactly as it was (code + model):

```powershell
git checkout run3-baseline      # look around (read-only "detached HEAD")
git checkout main               # come back
git switch -c from-run3 run3-baseline   # start new work from this point
```

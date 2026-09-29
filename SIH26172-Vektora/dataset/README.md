# Dataset: "Vektora" keyword, 16 kHz mono PCM16

A copy of the team dataset repo (github.com/Thsky-21/Data), placed here so a
fresh clone can retrain without editing a path. `training/config.py` reads
this folder by default (override with `VEKTORA_DATA`).
**Folder membership is the label.**

| Folder | Class | WAVs | Originals | Pre-made augmented copies (`aug_<k>_<orig>.wav`) |
|---|---|---|---|---|
| `positive/` | keyword | 765 | 153 | 612 (4 per original) |
| `positive_akash/` | keyword | 77 | 77 | 0 (the presenter's own voice, 2026-09-28) |
| `hard_negative/` | other | 448 | 112 | 336 (vector, victor, Vectra, spectra, sector, doctor, …) |
| `negative_akash/` | other | 101 | 101 | 0 (the presenter's own look-alike words) |
| `generic_negative/` | other | 1,020 | 340 | 680 (≈240 of the originals are Google Speech Commands, `gsc_*`, CC-BY-4.0) |
| `background_negative/` | noise | 101 | 101 | 0 |

Only **~884 unique recordings** exist; the rest are augmented copies.
`training/manifest.py` **drops the `aug_*` files** and augments online
instead (`training/augment.py`). It also splits by recording group, never by
file, so a copy can't sit in train and test at the same time (leakage). The
`aug_*` files are kept here for completeness and for anyone who wants the
original offline augmentation.

## Also here

| Path | What |
|---|---|
| `ui_captures/` | 47 clips saved by the Streamlit tester (`extras/streamlit_tester`) while testing by hand. Not used in training. |
| `extra/Humanspeech.m4a` | A long human-speech recording from the repo root (`Human speech/`). Candidate negative audio for the M4 false-accept soak. Not used in training. |
| `speakers.csv` | Speaker log from the guided recorder (`extras/recorder`). |
| `long/`, `external_noise/` | Don't exist yet. `manifest.py` picks up long INMP441 recordings from `long/noise/` and `long/other/`; `augment.py` uses external noise (ESC-50/MUSAN) from `external_noise/`. |

## Known data issues (see `challenges.md`, `docs/current_state.md`)

- 14 originals with dropped-sample gaps ("chopped audio") are **excluded via
  `training/exclusions.csv`**, not deleted.
- 27 augmented files are byte-identical to their originals.
- About 150 positives and 166 hard negatives have some clipping.
- The repo root used to hold loose copies (`vektora_*.wav`, `positive_audio/`,
  `soft_sample/`, `soft-negative/`, `background_noise/`). Every one is
  **byte-identical** to a file here (checked by MD5 on 2026-09-29), so they
  are not duplicated.

## Consent

These are voice recordings of team members and volunteers, plus the
CC-BY-4.0 Google Speech Commands clips. The source repo (Thsky-21/Data) is
already public.

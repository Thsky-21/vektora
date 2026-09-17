# prototype/training — the keyword model

Always run from this folder with the project venv.

```bash
../../.venv/Scripts/python.exe manifest.py   # list clips, fix the train/val/test split
../../.venv/Scripts/python.exe train.py      # train, pick threshold, score test -> model/
../../.venv/Scripts/python.exe model.py      # print the architecture
```

Re-run `manifest.py` whenever data is added.

| File | Role |
|---|---|
| `config.py` | Paths, classes, split, training knobs. Safe to change. |
| `features.py` | **Parity contract** with the ESP32 C code. Don't change casually. |
| `manifest.py` | Lists originals and splits them by take block / speaker. |
| `augment.py` | Fresh random alterations every epoch; external-noise policy. |
| `model.py` | 3-class CNN (~24k parameters). |
| `train.py` | Training, threshold, honest test report. |

## Where data goes

| What | Where | Notes |
|---|---|---|
| Team dataset | `C:\Users\User\documents\data` (or `VEKTORA_DATA` env var) | Only originals are used; `aug_*` files are ignored. |
| Long INMP441 background recordings | `prototype/data/long/noise/*.wav` | 16 kHz WAV, any length. Cut into 1 s windows automatically. |
| Long INMP441 talking (no "Vektora" in it) | `prototype/data/long/other/*.wav` | Same. Quiet windows are dropped automatically. |
| External noise (ESC-50, DEMAND, MUSAN...) | `prototype/data/external_noise/**` | Any rate, any length, subfolders fine. |

All of `prototype/data/` is git-ignored.

## Recording background noise on the INMP441

- **Record long files, not 1 s clips.** Record 2–10 minutes per condition, one
  file per condition, named after it (`fan_speed3_lab.wav`, `quiet_room_night.wav`,
  `corridor.wav`). Long files give hundreds of windows, and a time-based split
  inside each file keeps test windows away from train windows.
- **Use the same firmware and gain settings as the speech recordings.** If the
  firmware shift or gain is different, the noise has a different level than the
  speech it will be mixed with.
- **Cover several conditions:** silent room (the mic's own hiss), ceiling fan at
  each speed, laptop/projector fans, a chattering crowd far away, keyboard,
  chairs, doors, footsteps, rain/traffic through a window, and the board powered
  from the battery (TP4056) as well as from USB.
- **No clear nearby speech in `noise/`.** If words are recognisable, the file
  belongs in `other/`.

## External noise: where to get it

| Dataset | What | Size | License | Get it |
|---|---|---|---|---|
| **ESC-50** | 2,000 × 5 s clips, 50 classes (rain, dog, clock, keyboard, door knock, crowd...) | ~600 MB | CC BY-NC 3.0 | `github.com/karolpiczak/ESC-50` → "Download .zip" → put `audio/` here |
| **DEMAND** | Long multichannel recordings of 18 real places (kitchen, office, cafe, bus, park) | ~few hundred MB for the 16 kHz set | CC BY-SA 3.0 | `zenodo.org/record/1227121` → download the `*_16k.zip` files; one channel per environment is enough |
| **MUSAN** (noise part) | ~6 h of assorted noises | 11 GB for all of MUSAN; you only need `musan/noise` | CC BY 4.0 (mostly) | `openslr.org/17` |
| Speech Commands `_background_noise_` | 6 files (white/pink noise, dishes, bike, tap) | tiny | CC BY 4.0 | inside the Speech Commands v2 archive |

Start with **ESC-50 + DEMAND** because they're the best value for the download
size. Avoid the ESC-50 human-speech-like categories if you want to be strict
(`laughing`, `crying_baby`, `coughing`, `sneezing`, `breathing`, `snoring`):
they're not speech, but they're closer to it than a fan. Put them in the noise
folder only if you want the model to reject them as noise.

**Licensing:** ESC-50 is non-commercial. That's fine for SIH, but credit it in
the report.

## Is training on external (non-INMP441) noise a good idea?

Yes, as long as it's controlled. `augment.py` enforces the rules below.

- **It helps:** it adds hundreds of noise types the team can't record, and it
  teaches the model to find "Vektora" *inside* noise.
- **The risk:** if most of the noise class sounded "not like the INMP441", the
  model could learn "different microphone → noise" instead of "not speech →
  noise". That would score well on a mixed test set and fail on the device,
  where everything is INMP441.
- **What the code does about it:**
  1. External noise is mixed as background under **all** classes equally, so
     it's never a clue for one class.
  2. At most 30% of noise-class examples are external, and each gets real
     INMP441 background mixed underneath.
  3. Validation and test contain **only real recordings**. They are INMP441
     except the Speech Commands words (`gsc`), which come from contributors'
     own mics.
- **The best of both:** play external noise through a speaker and record it on
  the INMP441. That turns it into real-mic data.

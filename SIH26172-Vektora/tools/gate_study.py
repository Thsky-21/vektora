"""gate_study.py — pick the Stage-1 voice-gate thresholds from real clips.

Offline, on the training data, with the training feature code (features.py),
so the thresholds are chosen from measurements rather than guessed. The
firmware computes the same two numbers from the same log-mel row (gate.c):

  band_ratio = sum(mel energy, bands centred 300-3400 Hz) / sum(all bands)
  flatness   = geometric mean / arithmetic mean of the 40 linear mel energies

Per clip, only frames that would pass Stage 0 (energy >= floor + 9 dB, floor =
10th-percentile frame energy of the clip, min -70 dBFS) are scored, because
Stage 1 only ever runs on those. A clip "opens the gate" if >= GATE1_FRAMES
consecutive Stage-0 frames also pass Stage 1.

    .venv/Scripts/python.exe tools/gate_study.py [--max 200]

This is an OFFLINE study on 1 s clips, not a device measurement (Rule 0.1).
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "training"))
import features as F  # noqa: E402

DATA = os.environ.get("VEKTORA_DATA", os.path.join(HERE, "..", "dataset"))
SETS = {  # name -> (glob, should the gate open?)
    "positive_akash": (os.path.join(DATA, "positive_akash", "**", "*.wav"), True),
    "positive": (os.path.join(DATA, "positive", "**", "*.wav"), True),
    "hard_negative": (os.path.join(DATA, "hard_negative", "**", "*.wav"), True),  # speech: gate SHOULD open
    "background_negative": (os.path.join(DATA, "background_negative", "**", "*.wav"), False),
}
GATE0_DB = 9.0


def mel_centres():
    m = lambda f: 2595 * np.log10(1 + f / 700)  # HTK
    im = lambda x: 700 * (10 ** (x / 2595) - 1)
    pts = np.linspace(m(F.FMIN), m(F.FMAX), F.N_MELS + 2)
    return im(pts[1:-1])


def frame_stats(x):
    db = F.logmel(x)                        # (98, 40) dB
    lin = 10 ** (db / 10)
    c = mel_centres()
    band = (c >= 300) & (c <= 3400)
    ratio = lin[:, band].sum(1) / lin.sum(1)
    flat = np.exp(np.log(lin).mean(1)) / lin.mean(1)
    fr = np.lib.stride_tricks.sliding_window_view(x.astype(np.float64), F.WIN)[::F.HOP][:len(db)]
    e_db = 10 * np.log10((fr ** 2).mean(1) + 1e-12)
    return e_db, ratio, flat


def clip_opens(e_db, ratio, flat, r_min, f_max, n_frames):
    floor = max(np.percentile(e_db, 10), -70.0)
    s0 = e_db >= floor + GATE0_DB
    s1 = s0 & (ratio >= r_min) & (flat <= f_max)
    run = best = 0
    for v in s1:
        run = run + 1 if v else 0
        best = max(best, run)
    return best >= n_frames, s0.any()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=200, help="clips per set")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    stats = {}
    for name, (pat, _) in SETS.items():
        files = sorted(f for f in glob.glob(pat, recursive=True) if not os.path.basename(f).startswith("aug_"))
        if len(files) > a.max:
            files = list(rng.choice(files, a.max, replace=False))
        stats[name] = [frame_stats(F.fix_length(F.load_wav(f))) for f in files]
        print(f"{name:20} {len(files)} clips")

    # distributions on Stage-0 frames
    for name, st in stats.items():
        r, fl = [], []
        for e, ra, f in st:
            s0 = e >= max(np.percentile(e, 10), -70.0) + GATE0_DB
            r += list(ra[s0]); fl += list(f[s0])
        if r:
            print(f"{name:20} S0 frames {len(r):5}  band_ratio p10/50/90 "
                  f"{np.percentile(r, 10):.2f}/{np.percentile(r, 50):.2f}/{np.percentile(r, 90):.2f}"
                  f"  flatness p10/50/90 {np.percentile(fl, 10):.3f}/{np.percentile(fl, 50):.3f}/{np.percentile(fl, 90):.3f}")

    print("\nclip open rate (speech sets should be ~1, noise sets low):")
    print(f"{'frames':>6} {'ratio>=':>8} {'flat<=':>7} " + " ".join(f"{n[:14]:>14}" for n in SETS))
    for n in (2, 3, 5, 8):
        for r_min in (0.0, 0.4, 0.5, 0.6):
            for f_max in (1.0, 0.5, 0.4, 0.3):
                row = []
                for name in SETS:
                    o = [clip_opens(*s, r_min, f_max, n)[0] for s in stats[name]]
                    row.append(np.mean(o) if o else float("nan"))
                print(f"{n:6} {r_min:8.2f} {f_max:7.2f} " + " ".join(f"{v:14.2f}" for v in row))


if __name__ == "__main__":
    main()

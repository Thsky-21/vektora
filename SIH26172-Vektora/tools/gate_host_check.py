"""gate_host_check.py — run the FIRMWARE's gate.c (compiled for the laptop) on real clips.

    gcc -O2 -std=gnu11 -Ifirmware/main -o <exe> firmware/host_test/gate_main.c \
        firmware/main/gate.c firmware/main/feature_extract.c -lm
    .venv/Scripts/python.exe tools/gate_host_check.py <exe>

Two streams, each through the exact C gate (adaptive floor included):
  keyword  each Vektora clip after 6 s of quiet ambience (a background clip
           at -26 dB): does the gate open during the word, and how soon?
  noise    all background_negative clips back to back, at full level: what
           share of the time is the gate open (= model running)?
OFFLINE sanity check of the C code on dataset audio, not a device measurement.
"""

import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "training"))
import features as F  # noqa: E402

DATA = os.environ.get("VEKTORA_DATA", os.path.join(HERE, "..", "dataset"))
SR, HOP = 16000, 160


def i16(x):
    return np.clip(np.round(x * 32767), -32768, 32767).astype("<i2")


def run(exe, pcm):
    with tempfile.TemporaryDirectory() as d:
        fi, fo = os.path.join(d, "in.i16"), os.path.join(d, "out.csv")
        i16(pcm).tofile(fi)
        r = subprocess.run([exe, fi, fo], capture_output=True, text=True, check=True)
        csv = np.loadtxt(fo, delimiter=",", skiprows=1)
    return csv, r.stderr.strip()


def clips(folder, n, rng):
    fs = sorted(f for f in glob.glob(os.path.join(DATA, folder, "**", "*.wav"), recursive=True)
                if not os.path.basename(f).startswith("aug_"))
    return [fs[i] for i in rng.choice(len(fs), min(n, len(fs)), replace=False)]


def main():
    exe = sys.argv[1]
    rng = np.random.default_rng(0)
    bg = [F.load_wav(f) for f in clips("background_negative", 200, rng)]
    amb = np.concatenate(bg)

    # --- keyword clips in quiet ambience ---
    res = []
    for f in clips("positive_akash", 40, rng) + clips("positive", 40, rng):
        kw = F.fix_length(F.load_wav(f))
        on, off = F.active_span(kw)          # sample span of the spoken word
        # 6 s so the adaptive floor has settled (it rises 2 dB/s) before the word
        pre = amb[rng.integers(0, len(amb) - 7 * SR):][:6 * SR] * 0.05
        post = amb[rng.integers(0, len(amb) - 3 * SR):][:SR] * 0.05
        pcm = np.concatenate([pre, kw + amb[:len(kw)] * 0.05, post])
        csv, _ = run(exe, pcm)
        open_ = csv[:, 4].astype(bool)
        # frame f covers samples [160 f, 160 f + 400): index by frame END sample
        end_sample = csv[:, 0] * HOP + 400
        w_on, w_off = len(pre) + on, len(pre) + off
        in_word = (end_sample >= w_on) & (end_sample <= w_off + 0.4 * SR)
        first = np.argmax(open_ & (end_sample >= w_on)) if (open_ & (end_sample >= w_on)).any() else None
        settled = (end_sample >= len(pre) - 2 * SR) & (end_sample < len(pre) - 0.2 * SR)
        pre_open = open_[settled].mean()
        res.append((open_[in_word].any(),
                    (end_sample[first] - w_on) / SR * 1000 if first is not None else np.nan, pre_open))
    ok = np.array([r[0] for r in res])
    delay = np.array([r[1] for r in res])
    print(f"keyword: gate opened on {ok.sum()}/{len(ok)} words; open delay after word onset "
          f"median {np.nanmedian(delay):.0f} ms, p90 {np.nanpercentile(delay, 90):.0f} ms; "
          f"open during the quiet 2 s before (floor settled): {np.mean([r[2] for r in res]) * 100:.1f}% of frames")

    # --- noise only, full level ---
    csv, summary = run(exe, amb)
    open_ = csv[:, 4].astype(bool)
    print(f"noise:   {len(amb) / SR:.0f} s of background_negative at full level -> gate open "
          f"{open_.mean() * 100:.1f}% of the time ({summary})")


if __name__ == "__main__":
    main()

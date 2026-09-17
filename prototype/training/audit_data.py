"""
audit_data.py — measure every ORIGINAL recording for signs of corruption and
for signs that it did not come straight from the INMP441 (e.g. over a call).

Run:  ../../.venv/Scripts/python.exe audit_data.py
Writes prototype/data/build/audit.csv (one row per file, all measurements).
It removes nothing: exclusions are decided by a human in exclusions.csv.

Measurements, and why each one:

  rms_db, peak          Overall level. Outliers vs the folder's median are
                        suspicious (wrong gain, wrong file).
  clip_frac             Share of samples at full scale (|x| >= 0.999).
                        Heavy clipping destroys the spectrum's shape.
  gaps / gap_max_ms     INTERIOR runs of exact zeros >= 10 ms. Real acoustic
                        audio never contains exact zeros; a zero run in the
                        middle of a clip means dropped samples. Leading /
                        trailing zeros are ignored — that is just padding
                        (Speech Commands clips shorter than 1 s).
  abrupt_gaps           Gaps whose edges are loud (|x| > 0.02 right next to
                        the gap): the signal was cut, not faded. This is the
                        "chopped" signature confirmed by ear on noise_066/067
                        and vektora_019.
  floor_db              Quietest 10% of 25 ms blocks, ignoring exact zeros:
                        the recording's noise floor. A live INMP441 always
                        has one; a call's noise suppression pushes it down.
  hf_db                 Speech-frame energy at 4-7.6 kHz minus 0.3-3.4 kHz.
                        Narrow-band call codecs remove most of the top band.
  cutoff_hz             Highest frequency whose long-term level is within
                        45 dB of the spectrum's peak: the effective bandwidth.
  dur_s                 Length. Very short = truncated / empty.
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import features as F  # noqa: E402

OUT = os.path.join(C.BUILD_DIR, "audit.csv")


def zero_runs(x, min_len=160):
    z = np.concatenate([[0], (x == 0).astype(np.int8), [0]])
    d = np.diff(z)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    keep = (ends - starts) >= min_len
    return starts[keep], ends[keep]


def measure(x):
    n = len(x)
    s, e = zero_runs(x)
    interior = [(a, b) for a, b in zip(s, e) if a > 0 and b < n]
    abrupt = 0
    for a, b in interior:
        before = np.abs(x[max(a - 8, 0):a]).max()
        after = np.abs(x[b:b + 8]).max()
        if max(before, after) > 0.02:
            abrupt += 1
    blocks = x[: n // 400 * 400].reshape(-1, 400)
    brms = np.sqrt(np.mean(blocks ** 2, axis=1))
    nz = brms[brms > 0]
    floor_db = 20 * np.log10(np.percentile(nz, 10) + 1e-9) if len(nz) else -180.0
    # spectrum over the loud frames only (within 20 dB of the loudest)
    spec = np.abs(np.fft.rfft(blocks * np.hanning(400), axis=1)) ** 2
    loud = brms >= brms.max() * 0.1
    ps = spec[loud].mean(axis=0) + 1e-20
    f = np.fft.rfftfreq(400, 1 / F.SR)
    band = lambda lo, hi: ps[(f >= lo) & (f < hi)].sum()
    hf_db = 10 * np.log10(band(4000, 7600) / band(300, 3400))
    ps_db = 10 * np.log10(ps)
    above = np.where(ps_db >= ps_db.max() - 45)[0]
    cutoff = float(f[above.max()])
    return dict(
        dur_s=round(n / F.SR, 3),
        rms_db=round(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9), 1),
        peak=round(float(np.abs(x).max()), 3),
        clip_frac=round(float((np.abs(x) >= 0.999).mean()), 4),
        gaps=len(interior),
        abrupt_gaps=abrupt,
        gap_max_ms=round(max([(b - a) / 16 for a, b in interior], default=0), 1),
        floor_db=round(float(floor_db), 1),
        hf_db=round(float(hf_db), 1),
        cutoff_hz=int(cutoff),
    )


def main():
    rows = []
    for folder in C.FOLDERS:
        d = os.path.join(C.DATA_ROOT, folder)
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(".wav") and not fn.startswith("aug_"):
                x = F.load_wav(os.path.join(d, fn))
                rows.append(dict(folder=folder, file=fn, **measure(x)))
    os.makedirs(C.BUILD_DIR, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()

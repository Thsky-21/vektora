"""
manifest.py — list every usable clip, and decide train / val / test ONCE.

Run:  ../../.venv/Scripts/python.exe manifest.py      (from prototype/training)
Writes prototype/data/build/manifest.csv. Every later step reads that file,
so the split is fixed and identical across all experiments.

-----------------------------------------------------------------------------
Why the split is done by GROUP, not by file
-----------------------------------------------------------------------------
A model is only as good as its score on audio it has never heard. Two traps
would make that score lie:

1. Augmented copies. `aug_2_vektora_031.wav` is still `vektora_031.wav` —
   same person, same take, slightly altered. If one sits in train and the
   other in test, the test is partly a memory test. So: the pre-made `aug_*`
   files are DROPPED entirely (augmentation now happens fresh during
   training, augment.py), and only the 706 originals are split.

2. Same speaker, same sitting. Speakers are not labelled. But takes were
   recorded in runs — `vektora_031` and `vektora_032` are very likely the same
   person, same position, same minute. Splitting takes one by one would put
   near-identical neighbours on both sides. So consecutive take numbers are
   bundled into blocks of TAKE_BLOCK (10), and whole blocks are assigned.
   That approximates "hold out a session" without speaker labels.
   Google Speech Commands clips carry a real speaker ID in their filename
   (`gsc_cat_5db0e146_nohash_0` -> speaker 5db0e146), so those are grouped
   by true speaker.

Honest caveat: with ~6 speakers and no labels, some speakers WILL appear in
both train and test. The test score measures "new takes", not "new people".
The real new-person test is recording someone who is not in the dataset.

Exclusions (exclusions.csv): files judged unusable are NOT deleted. The split
is computed on ALL originals first, and excluded files are then marked
split="excluded" (or "domain_shift" for usable-but-unrepresentative audio,
e.g. recorded over a call). Computing the split before excluding means
removing a file never reshuffles anything else: every remaining file keeps
the split it had in earlier runs, so runs stay comparable. train.py only
uses train / val / test.

Long recordings (prototype/data/long/<class>/*.wav) are cut into 1 s windows
and split by TIME inside each file: first 70% train, next 15% val, last 15%
test, with a 1 s gap at each boundary so no window straddles two splits.
-----------------------------------------------------------------------------
"""

import csv
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

FIELDS = ["path", "offset", "cls", "source", "group", "split", "external", "note"]
EXCLUSIONS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exclusions.csv")


def load_exclusions():
    """(folder, file) -> (split label, reason)."""
    out = {}
    if os.path.exists(EXCLUSIONS):
        with open(EXCLUSIONS, newline="") as f:
            for r in csv.DictReader(f):
                label = {"exclude": "excluded", "domain_shift": "domain_shift"}[r["action"]]
                out[(r["folder"], r["file"])] = (label, f'{r["decided_by"]}: {r["reason"]}')
    return out


def take_group(source: str, fname: str) -> str:
    """Group key: which files must stay together in one split."""
    stem = os.path.splitext(fname)[0]
    m = re.match(r"gsc_[a-z]+_([0-9a-f]+)_nohash", stem)
    if m:                                         # Speech Commands: real speaker ID
        return f"gsc:{m.group(1)}"
    nums = re.findall(r"\d+", stem)
    if not nums:
        return f"{source}:{stem}"
    return f"{source}:block{int(nums[-1]) // C.TAKE_BLOCK}"


def assign_splits(groups: dict, rng) -> dict:
    """groups: key -> n_files. Shuffle the groups, then fill train/val/test
    until each holds its share of FILES. Returns key -> split."""
    keys = sorted(groups)
    rng.shuffle(keys)
    total = sum(groups.values())
    targets = [C.SPLIT[1] * total, C.SPLIT[2] * total]   # val, test
    out, filled = {}, [0, 0]
    for k in keys:
        if filled[0] < targets[0]:
            out[k], filled[0] = "val", filled[0] + groups[k]
        elif filled[1] < targets[1]:
            out[k], filled[1] = "test", filled[1] + groups[k]
        else:
            out[k] = "train"
    return out


def team_dataset(rng):
    rows, skipped = [], Counter()
    excl = load_exclusions()
    for folder, (cls, source) in C.FOLDERS.items():
        d = os.path.join(C.DATA_ROOT, folder)
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(".wav"))
        originals = [f for f in files if not f.startswith("aug_")]
        skipped[folder] = len(files) - len(originals)
        # GSC and the team's own clips are split separately so each gets 70/15/15
        by_group = defaultdict(list)
        for f in originals:
            by_group[take_group(source, f)].append(f)
        for family in ({k for k in by_group if k.startswith("gsc:")},
                       {k for k in by_group if not k.startswith("gsc:")}):
            if not family:
                continue
            split = assign_splits({k: len(by_group[k]) for k in family}, rng)
            for k in family:
                for f in by_group[k]:
                    sp, note = split[k], ""
                    if (folder, f) in excl:
                        label, why = excl.pop((folder, f))
                        sp, note = label, f"was {split[k]}; {why}"
                    rows.append(dict(path=os.path.join(d, f), offset=0, cls=cls,
                                     source="gsc" if k.startswith("gsc:") else source,
                                     group=k, split=sp, external=0, note=note))
    assert not excl, f"exclusions.csv names files that don't exist: {sorted(excl)}"
    return rows, skipped


def long_recordings():
    """Cut prototype/data/long/<cls>/*.wav into 1 s windows, split by time."""
    rows = []
    if not os.path.isdir(C.LONG_DIR):
        return rows
    for cls in os.listdir(C.LONG_DIR):
        if cls not in ("noise", "other"):
            print(f"  ! ignoring prototype/data/long/{cls}/ (use noise/ or other/)")
            continue
        source = "noise_long" if cls == "noise" else "conversation"
        hop = C.SR_HOP_NOISE if cls == "noise" else C.SR_HOP_OTHER
        for root, _, files in os.walk(os.path.join(C.LONG_DIR, cls)):
            for f in sorted(files):
                if not f.lower().endswith(".wav"):
                    continue
                p = os.path.join(root, f)
                info = sf.info(p)
                if info.samplerate != 16000:
                    print(f"  ! {p}: {info.samplerate} Hz — long recordings must be 16 kHz, skipped")
                    continue
                n = info.frames
                cut1, cut2 = int(n * 0.70), int(n * 0.85)
                x = sf.read(p, dtype="float32", always_2d=True)[0].mean(axis=1)
                for off in range(0, n - 16000 + 1, hop):
                    end = off + 16000
                    if end <= cut1 - 16000:
                        split = "train"
                    elif cut1 <= off and end <= cut2 - 16000:
                        split = "val"
                    elif cut2 <= off:
                        split = "test"
                    else:
                        continue                          # the 1 s safety gaps
                    w = x[off:end]
                    # A silent pause inside a conversation is not "other
                    # speech" — labelling it so would teach the model that
                    # silence is speech. Drop quiet windows from `other`.
                    if cls == "other" and 20 * np.log10(np.sqrt(np.mean(w ** 2)) + 1e-9) < -45:
                        continue
                    rows.append(dict(path=p, offset=off, cls=cls, source=source,
                                     group=f"long:{f}", split=split, external=0, note=""))
    return rows


def main():
    rng = np.random.default_rng(C.SEED)
    rows, skipped = team_dataset(rng)
    rows += long_recordings()
    os.makedirs(C.BUILD_DIR, exist_ok=True)
    with open(C.MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"dataset: {C.DATA_ROOT}")
    print("pre-made augmented files dropped:", dict(skipped))
    table = Counter((r["source"], r["split"]) for r in rows)
    sources = sorted({r["source"] for r in rows})
    cols = ("train", "val", "test", "excluded", "domain_shift")
    print(f"\n{'source':<14}" + "".join(f"{c:>13}" for c in cols) + f"{'groups':>8}")
    for s in sources:
        g = len({r['group'] for r in rows if r['source'] == s})
        print(f"{s:<14}" + "".join(f"{table[(s, sp)]:>13}" for sp in cols) + f"{g:>8}")
    print(f"\nwrote {len(rows)} rows -> {C.MANIFEST}")

    # Guard 1: no group may appear in two of train/val/test.
    seen = defaultdict(set)
    for r in rows:
        if not r["group"].startswith("long:") and r["split"] in ("train", "val", "test"):
            seen[r["group"]].add(r["split"])
    leaks = [g for g, s in seen.items() if len(s) > 1]
    assert not leaks, f"groups in more than one split: {leaks}"
    # Guard 2: no identical audio (under any file name) in two splits.
    import hashlib
    where = defaultdict(set)
    for r in rows:
        if r["split"] in ("train", "val", "test") and not r["group"].startswith("long:"):
            x, _ = sf.read(r["path"], dtype="int16")
            where[hashlib.md5(x.tobytes()).hexdigest()].add(r["split"])
    dup = sum(len(s) > 1 for s in where.values())
    assert dup == 0, f"{dup} identical recordings sit in more than one split"
    print("leakage checks passed: no shared groups, no identical audio across splits")


if __name__ == "__main__":
    main()

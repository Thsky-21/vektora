"""
prepare_data.py — collect every clip, give it a label, split it three ways.

Run from the repo root:
    .venv/Scripts/python.exe demo/prepare_data.py

What it does (demo.md §5.3):
  1. Loads the GitHub clips: 63 "Vektora" (label 1) + 27 soft negatives (label 0).
  2. Loads your phone recordings, if present:
       demo/data/phone_vektora/  -> cut into single words automatically  (label 1)
       demo/data/other_speech/   -> sliced into 1 s pieces              (label 0)
       demo/data/noise/          -> sliced into 1 s pieces              (label 0)
  3. Generates synthetic noise clips (white/pink/brown noise, mains hum) (label 0).
  4. Splits everything into train (70%) / validation (15%) / test (15%).
  5. Saves demo/data/prepared.npz, example clips for the app, and two graphs.

Nothing here trains a model — that's train.py.
"""

import glob
import os
import sys
from collections import Counter

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio_utils as au  # noqa: E402
import plots as P  # noqa: E402
from plots import plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
SEED = 42                      # fixed randomness -> the same split every run
SPLITS = ("train", "val", "test")
FRACTIONS = (0.70, 0.15, 0.15)
AUDIO_EXT = ("*.wav", "*.m4a", "*.mp3", "*.aac", "*.ogg", "*.flac", "*.3gp", "*.opus")

rng = np.random.default_rng(SEED)

# ROLLBACK 2026-09-12 (user's request, 30 min before the presentation).
# Train on the GitHub data ONLY: 63 positives + 27 soft negatives, plus the
# synthetic noise generated in code (which the original v0 model also had, and
# which is what stops digital silence from being guessed at).
# Everything recorded later — phone_vektora/, live_vektora/, live_other/,
# other_speech/, noise/ — is skipped. The folders are left on disk untouched,
# so setting this back to False restores the fuller dataset in one edit.
GITHUB_ONLY = True


def files_in(folder):
    out = []
    for ext in AUDIO_EXT:
        out += glob.glob(os.path.join(folder, ext))
    return sorted(out)


def split_list(items):
    """Shuffle and cut a list of clips 70/15/15. Used for sources where every
    clip is an independent recording, so any clip can go in any split."""
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    n_val = max(1, round(n * FRACTIONS[1])) if n >= 3 else 0
    n_test = max(1, round(n * FRACTIONS[2])) if n >= 3 else 0
    n_train = n - n_val - n_test
    return {"train": items[:n_train],
            "val": items[n_train:n_train + n_val],
            "test": items[n_train + n_val:]}


def slice_long(x, hop_s=0.5):
    """Cut a long recording into 1 s pieces every hop_s seconds."""
    hop = int(hop_s * au.SR)
    return [x[s:s + au.CLIP] for s in range(0, len(x) - au.CLIP + 1, hop)]


def split_long_recording(x):
    """For long recordings (talking, room noise) we split by TIME first —
    first 70% of the recording -> train, next 15% -> val, last 15% -> test —
    and only THEN slice into 1 s pieces. Slicing first would put overlapping
    pieces of the same sentence into both train and test ("data leakage",
    demo.md §5.3), and the test score would be a lie."""
    n = len(x)
    a, b = int(n * FRACTIONS[0]), int(n * (FRACTIONS[0] + FRACTIONS[1]))
    return {"train": slice_long(x[:a]), "val": slice_long(x[a:b]), "test": slice_long(x[b:])}


def segment_words(x, min_s=0.25, max_s=1.3, merge_gap_s=0.25):
    """Find each spoken "Vektora" in a recording of ~20 repetitions.

    librosa.effects.split finds stretches louder than (peak - 28 dB). Stretches
    separated by < 0.25 s are merged, because the silent closure inside
    "ve-K-Tora" would otherwise cut every word in two. Each word is then
    centred in a 1 s window taken from the real recording (so it keeps its
    real background noise rather than digital silence)."""
    import librosa
    iv = librosa.effects.split(x, top_db=28, frame_length=400, hop_length=160)
    merged = []
    for s, e in iv:
        if merged and (s - merged[-1][1]) < merge_gap_s * au.SR:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    words, too_long = [], 0
    for s, e in merged:
        dur = (e - s) / au.SR
        if dur < min_s:
            continue                    # a click or breath, not a word
        if dur > max_s:
            too_long += 1               # probably two words run together
            continue
        c = (s + e) // 2
        lo = c - au.CLIP // 2
        seg = x[max(0, lo):lo + au.CLIP]
        if lo < 0:
            seg = np.pad(seg, (-lo, 0))
        words.append(au.fix_length(seg))
    return words, too_long


def synthetic_noise(n):
    """Noise the model should learn to ignore: white (hiss), pink (rain /
    fans), brown (rumble), mains hum, and near-silence, at random volumes."""
    clips = []
    kinds = ["white", "pink", "brown", "hum", "silence"]
    for i in range(n):
        kind = kinds[i % len(kinds)]
        w = rng.standard_normal(au.CLIP).astype(np.float32)
        if kind in ("pink", "brown"):
            spec = np.fft.rfft(w)
            f = np.arange(len(spec)) + 1.0
            spec /= np.sqrt(f) if kind == "pink" else f
            w = np.fft.irfft(spec, n=au.CLIP).astype(np.float32)
        elif kind == "hum":
            t = np.arange(au.CLIP) / au.SR
            base = rng.choice([50.0, 60.0])
            w = sum(np.sin(2 * np.pi * base * k * t) / k for k in (1, 2, 3, 5)).astype(np.float32)
            w += 0.1 * rng.standard_normal(au.CLIP).astype(np.float32)
        level_db = rng.uniform(-70, -55) if kind == "silence" else rng.uniform(-50, -15)
        w *= 10 ** (level_db / 20) / (np.sqrt(np.mean(w ** 2)) + 1e-9)
        clips.append(np.clip(w, -1, 1).astype(np.float32))
    return clips


def main():
    buckets = {s: [] for s in SPLITS}          # split -> list of (wave, label, source)

    def add(split_dict, label, source):
        for s in SPLITS:
            buckets[s] += [(w, label, source) for w in split_dict[s]]

    # 1. GitHub clips: already 1 s each. Peak-normalise each one (each file is
    #    its own recording).
    pos = [au.peak_normalize(au.fix_length(au.load_audio(f))) for f in sorted(glob.glob(os.path.join(ROOT, "vektora_*.wav")))]
    soft = [au.peak_normalize(au.fix_length(au.load_audio(f))) for f in sorted(glob.glob(os.path.join(ROOT, "soft-negative", "*.wav")))]
    if not pos or not soft:
        sys.exit("Couldn't find vektora_*.wav / soft-negative/*.wav in the repo root. Did `git pull` run?")
    add(split_list(pos), 1, "Vektora (project mic)")
    add(split_list(soft), 0, "soft negative")

    # 2. Everything recorded after the GitHub set — skipped in rollback mode.
    seg_dir = os.path.join(DATA, "_vektora_word_segments")
    if GITHUB_ONLY:
        print("  GITHUB_ONLY: skipping phone / browser-mic / talking / room-noise recordings")
    else:
        for folder, source in (("phone_vektora", "Vektora (phone)"),
                               ("live_vektora", "Vektora (browser mic)")):
            words = []
            for f in files_in(os.path.join(DATA, folder)):
                x = au.peak_normalize(au.load_audio(f))
                found, too_long = segment_words(x)
                print(f"  {folder}/{os.path.basename(f)}: found {len(found)} words"
                      + (f" ({too_long} stretches skipped as too long)" if too_long else ""))
                words += found
            if words:
                os.makedirs(seg_dir, exist_ok=True)
                for i, w in enumerate(words):   # saved so you can LISTEN and check the cuts
                    sf.write(os.path.join(seg_dir, f"{folder}_{i:02d}.wav"), w, au.SR)
                add(split_list(words), 1, source)

        for folder, source in (("other_speech", "other speech (laptop)"),
                               ("live_other", "other speech (browser mic)"),
                               ("noise", "room noise")):
            for f in files_in(os.path.join(DATA, folder)):
                x = au.peak_normalize(au.load_audio(f))
                print(f"  {folder}/{os.path.basename(f)}: {len(x) / au.SR:.0f} s")
                if len(x) >= 8 * au.SR:          # long enough to split by time
                    add(split_long_recording(x), 0, source)
                else:                            # a short batch: keep it whole
                    add(split_list(slice_long(x, hop_s=0.25) or [au.fix_length(x)]), 0, source)

    # 3. Synthetic noise.
    add(split_list(synthetic_noise(150)), 0, "synthetic noise")

    # 4. Shuffle within each split and save.
    out = {}
    for s in SPLITS:
        order = rng.permutation(len(buckets[s]))
        items = [buckets[s][i] for i in order]
        out[f"X_{s}"] = np.stack([w for w, _, _ in items]).astype(np.float32)
        out[f"y_{s}"] = np.array([lab for _, lab, _ in items], dtype=np.int8)
        out[f"src_{s}"] = np.array([src for _, _, src in items])
    os.makedirs(DATA, exist_ok=True)
    np.savez_compressed(os.path.join(DATA, "prepared.npz"), **out)

    # ---- Report --------------------------------------------------------
    sources = sorted({src for s in SPLITS for src in out[f"src_{s}"]},
                     key=lambda k: (not k.startswith("Vektora"), k))
    print("\n  clips per source and split")
    print(f"  {'source':<24}{'train':>7}{'val':>6}{'test':>6}")
    for src in sources:
        c = [int((out[f'src_{s}'] == src).sum()) for s in SPLITS]
        print(f"  {src:<24}{c[0]:>7}{c[1]:>6}{c[2]:>6}")
    for s in SPLITS:
        y = out[f"y_{s}"]
        print(f"  {s:<5} total {len(y):>4}   Vektora {int(y.sum()):>3}   not Vektora {int((y == 0).sum()):>4}")
    if not GITHUB_ONLY and not files_in(os.path.join(DATA, "other_speech")) and not files_in(os.path.join(DATA, "live_other")):
        print("\n  WARNING: no phone recordings yet. The model will not know what ordinary"
              "\n  talking looks like, and will only have seen Vektora from the project mic"
              "\n  (demo.md section 0.5). Fine for a first test run; record before the real training.")

    # ---- Example clips for the app (TEST split only: never seen in training)
    # These become PUBLIC on the website. "other speech" is deliberately left
    # out: it's a slice of you talking about anything, which may be private.
    ex_dir = os.path.join(HERE, "examples")
    os.makedirs(ex_dir, exist_ok=True)
    for old in glob.glob(os.path.join(ex_dir, "*.wav")):
        os.remove(old)
    wanted = ({"Vektora (project mic)": 3, "soft negative": 3, "synthetic noise": 1}
              if GITHUB_ONLY else
              {"Vektora (project mic)": 2, "Vektora (browser mic)": 1, "Vektora (phone)": 1,
               "soft negative": 2, "room noise": 1})
    for src, k in wanted.items():
        idx = np.where(out["src_test"] == src)[0][:k]
        for j, i in enumerate(idx):
            name = src.replace(" (", "_").replace(")", "").replace(" ", "_").lower()
            sf.write(os.path.join(ex_dir, f"{name}_{j + 1}.wav"), out["X_test"][i], au.SR)

    # ---- Graphs --------------------------------------------------------
    plot_dir = os.path.join(HERE, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Graph 1: how many clips of each kind
    counts = Counter(src for s in SPLITS for src in out[f"src_{s}"])
    fig, ax = plt.subplots(figsize=(7.5, 1.3 + 0.45 * len(sources)))
    for i, src in enumerate(sources[::-1]):
        colour = P.BLUE if src.startswith("Vektora") else P.ORANGE
        ax.barh(i, counts[src], color=colour, height=0.62)
        ax.text(counts[src] + max(counts.values()) * 0.01, i, f"{counts[src]}", va="center", color=P.INK)
    ax.set_xlim(0, max(counts.values()) * 1.1)
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels(sources[::-1])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("number of 1-second clips")
    ax.set_title("What the model learns from", pad=26)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=P.BLUE), plt.Rectangle((0, 0), 1, 1, color=P.ORANGE)],
              labels=["Vektora (label 1)", "not Vektora (label 0)"],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "class_counts.png"), dpi=150)
    plt.close(fig)

    # Graph 2: one example of each kind: waveform on top, spectrogram below
    show = [s for s in sources if (out["src_train"] == s).any()][:5]
    fig, axes = plt.subplots(2, len(show), figsize=(3.1 * len(show), 5.2), squeeze=False)
    for j, src in enumerate(show):
        x = out["X_train"][np.where(out["src_train"] == src)[0][0]]
        P.draw_waveform(axes[0, j], x, au.SR, title=src)
        P.draw_spectrogram(axes[1, j], au.logmel(x), title="")
        if j:
            axes[1, j].set_ylabel("")
    fig.suptitle("Same second of audio, two ways: waveform (top) and log-mel spectrogram (bottom)",
                 x=0.01, ha="left", fontweight="bold", color=P.INK)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "examples.png"), dpi=150)
    plt.close(fig)

    print(f"\n  saved: data/prepared.npz, examples/*.wav, plots/class_counts.png, plots/examples.png")


if __name__ == "__main__":
    main()

"""
features.py — 1 second of audio -> the 98 x 40 picture the model sees.

=============================================================================
PARITY CONTRACT  (CLAUDE.md §6 risk 1)
=============================================================================
This exact computation will be re-written in C for the ESP32
(firmware/.../feature_extract.c). If the two disagree in ANY detail, the
model is shown a picture it never trained on and accuracy collapses with NO
error message. So this file is written in plain NumPy, one step per line, each
step something a C programmer can copy directly. Anything that is awkward to
re-derive in C (the mel filterbank, the Hann window, the normalisation
constants) will be EXPORTED as a table by export_tables(), not re-computed.

  input      int16 PCM, 16 kHz mono, exactly 16000 samples
  scale      x = sample / 32768.0                 (float, -1..1)
  framing    frame i = x[i*160 : i*160 + 400]      i = 0..97   (no padding)
  window     periodic Hann, w[n] = 0.5 - 0.5*cos(2*pi*n/400)
  spectrum   power = |FFT_400(frame * w)|^2        bins 0..200 (201 bins)
  mel        mel[b] = sum_k  M[b][k] * power[k]    M = 40 x 201, HTK, no norm,
                                                   20..7600 Hz (exported table)
  log        L = 10 * log10(max(mel, 1e-10))
  normalise  F[t][b] = (L[t][b] - MEAN[b]) / STD[b]   (40 + 40 constants,
                                                   from training data, norm.json)
  output     F: 98 rows (time) x 40 columns (mel band)

Why these choices (full reasoning: CLAUDE.md §8.6):
  * no centre padding — a live stream cannot pad with audio from the future,
    and (16000-400)//160 + 1 = 98 frames exactly.
  * HTK mel, un-normalised filters — one line of maths each; and since the
    table is exported anyway, C never has to re-derive it.
  * fixed per-band normalisation, NOT per-clip — per-clip statistics would
    make the same word look different depending on what else shares its
    second of audio.
  * NO peak normalisation (the demo had it). The demo normalised a whole
    recording before slicing it; a device listening to an endless stream
    has no "whole recording". The model must therefore cope with real
    INMP441 levels, which is what gain augmentation (augment.py) teaches.
=============================================================================
"""

import json

import librosa
import numpy as np
import soundfile as sf

SR = 16000
CLIP = 16000
WIN = 400                 # 25 ms
HOP = 160                 # 10 ms
N_FFT = 400               # = WIN, so no zero-padding inside a frame
N_MELS = 40
FMIN, FMAX = 20.0, 7600.0
N_FRAMES = (CLIP - WIN) // HOP + 1          # 98
LOG_FLOOR = 1e-10

# Periodic Hann (the FFT-friendly variant; librosa/scipy use this too).
WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)).astype(np.float32)

# 40 x 201 mel weights. librosa builds it once here; C gets it as a table.
MEL_FB = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN,
                             fmax=FMAX, htk=True, norm=None).astype(np.float32)

# Self-check: a mel band with no FFT bin under it would be a permanently dead
# input to the model — silently. Fail loudly instead.
assert N_FRAMES == 98, N_FRAMES
assert MEL_FB.shape == (N_MELS, N_FFT // 2 + 1)
assert (MEL_FB.max(axis=1) > 0).all(), "a mel band receives no FFT bin"


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------
def load_wav(path) -> np.ndarray:
    """Read a WAV as mono float32 at 16 kHz (resampling only if needed)."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x.astype(np.float32)


def fix_length(x: np.ndarray, n: int = CLIP) -> np.ndarray:
    """Centre-pad with zeros, or centre-crop, to exactly n samples."""
    if len(x) >= n:
        s = (len(x) - n) // 2
        return x[s:s + n]
    pad = n - len(x)
    return np.pad(x, (pad // 2, pad - pad // 2))


# ---------------------------------------------------------------------------
# The feature itself
# ---------------------------------------------------------------------------
def logmel(x: np.ndarray) -> np.ndarray:
    """16000 samples -> (98, 40) log-mel in dB. Un-normalised."""
    assert len(x) == CLIP, len(x)
    idx = np.arange(WIN)[None, :] + HOP * np.arange(N_FRAMES)[:, None]   # (98, 400)
    frames = x[idx] * WINDOW
    power = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) ** 2            # (98, 201)
    mel = power @ MEL_FB.T                                               # (98, 40)
    return (10.0 * np.log10(np.maximum(mel, LOG_FLOOR))).astype(np.float32)


class Normalizer:
    """Per-mel-band mean/std, computed once from TRAINING originals only and
    saved to norm.json. The C code gets the same 80 numbers."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, feats):                       # feats: (N, 98, 40)
        return cls(feats.mean(axis=(0, 1)), feats.std(axis=(0, 1)) + 1e-6)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["mean"], d["std"])

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"mean": self.mean.tolist(), "std": self.std.tolist()}, f, indent=1)

    def __call__(self, feats):
        return ((feats - self.mean) / self.std).astype(np.float32)


# ---------------------------------------------------------------------------
# Where is the word inside its second?  (used to shift WITHOUT truncating)
# ---------------------------------------------------------------------------
def active_span(x: np.ndarray, below_peak_db: float = 20.0):
    """(first, last) sample index of the loud part of a clip.

    Per-frame RMS, keep frames within `below_peak_db` of the loudest frame.
    -20 dB is the threshold CLAUDE.md §9.5/§9.7 measured as defensible on this
    team's recordings (-30 dB measures the room, not the word)."""
    n_frames = (len(x) - WIN) // HOP + 1          # works for any length >= WIN
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n_frames)[:, None]
    rms_db = 10 * np.log10(np.mean(x[idx] ** 2, axis=1) + 1e-12)
    on = np.nonzero(rms_db >= rms_db.max() - below_peak_db)[0]
    return int(on[0] * HOP), int(min(on[-1] * HOP + WIN, len(x)))

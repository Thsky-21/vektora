"""
audio_utils.py — the ONE place where sound becomes a picture.

Both train.py and app.py import from this file. That is deliberate: if the
website computed the spectrogram even slightly differently from training
(another hop size, another normalisation), the model would be shown pictures
it never learned from and would give confident nonsense WITHOUT any error.
This is called "training/serving skew" (demo.md §5.2). One shared file makes
it impossible.

Pipeline for one clip:
    audio file -> mono 16 kHz samples -> peak-normalise -> exactly 1 s
      -> log-mel spectrogram (98 time steps x 40 mel bands) -> normalise
      -> model
"""

import io
import json
import os
import subprocess
import tempfile

import librosa
import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Constants. Change these and you MUST retrain: the model only understands
# pictures made with exactly these settings.
# ---------------------------------------------------------------------------
SR = 16000          # sample rate: 16,000 numbers per second of audio
CLIP = SR           # model input length: exactly 1 second = 16,000 samples
N_FFT = 400         # FFT size = window size. (librosa slices the audio by
                    # N_FFT, so 512 here would give 97 time steps, not 98.
                    # Checked: all 40 mel bands still get >= 3 FFT bins.)
WIN = 400           # window: 25 ms slices (400 samples)
HOP = 160           # hop: a new slice every 10 ms (160 samples)
N_MELS = 40         # 40 mel bands (the picture's height)
FMIN, FMAX = 20, 7600   # frequency range covered; 8 kHz is the maximum at 16 kHz
N_FRAMES = (CLIP - WIN) // HOP + 1   # = 98, the picture's width

# Silence gate, a tiny "voice activity detector" (VAD). A 1 s window quieter
# than this (in dB relative to full scale) is not even shown to the model:
# there is nothing there to recognise.
SILENCE_DBFS = -45.0


# ---------------------------------------------------------------------------
# Loading audio
# ---------------------------------------------------------------------------
def _decode_with_ffmpeg(data: bytes) -> np.ndarray:
    """Fallback for formats soundfile can't read — phone recordings (.m4a),
    .aac, .opus. Asks ffmpeg to convert to raw 16 kHz mono 32-bit floats.

    The bytes go to a temporary FILE, not to ffmpeg's standard input: .m4a is
    an MP4 container whose index can sit at the end, so the decoder has to
    jump around the file, which a pipe can't do ("Format not recognised")."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        out = subprocess.run(
            # No -hide_banner: the ffmpeg on this machine is from 2013 and
            # rejects that flag ("Unrecognized option").
            ["ffmpeg", "-loglevel", "error", "-i", path,
             "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
            capture_output=True, check=True)
    finally:
        os.unlink(path)
    return np.frombuffer(out.stdout, dtype=np.float32).copy()


def load_audio(src) -> np.ndarray:
    """Load a file path OR raw bytes (what the browser sends) as mono 16 kHz
    float32 samples in the range -1..1."""
    if isinstance(src, (bytes, bytearray)):
        data = bytes(src)
    else:                                         # a path (str or Path)
        with open(src, "rb") as f:
            data = f.read()
    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        x = x.mean(axis=1)                        # stereo -> mono by averaging
        if sr != SR:
            x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    except Exception:
        x = _decode_with_ffmpeg(data)             # .m4a / .aac / etc.
    return x.astype(np.float32)


def peak_normalize(x: np.ndarray, target: float = 0.9) -> np.ndarray:
    """Scale so the loudest sample sits at `target`. Makes a quiet laptop mic
    and a loud project mic look alike to the model. Applied to a WHOLE
    recording, never per 1 s window, so silence stays quiet instead of being
    blown up into loud noise."""
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    return x if peak < 1e-5 else (x * (target / peak)).astype(np.float32)


def fix_length(x: np.ndarray, n: int = CLIP) -> np.ndarray:
    """Centre-pad with zeros or centre-crop to exactly n samples."""
    if len(x) >= n:
        start = (len(x) - n) // 2
        return x[start:start + n]
    pad = n - len(x)
    return np.pad(x, (pad // 2, pad - pad // 2))


def rms_dbfs(x: np.ndarray) -> float:
    """Loudness of a clip in dBFS (0 = maximum possible; -60 = very quiet)."""
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9))


# ---------------------------------------------------------------------------
# Features: the picture the model sees
# ---------------------------------------------------------------------------
def logmel(x: np.ndarray) -> np.ndarray:
    """1 s of audio -> log-mel spectrogram, shape (98, 40).

    Steps (demo.md §5.2): cut into 25 ms slices every 10 ms -> FFT each slice
    (which pitches are present) -> group the pitches into 40 mel bands (fine at
    low pitches, coarse at high, like human hearing) -> convert to decibels
    (log scale, also like human hearing).

    center=False means no padding is invented at the edges, which is why we
    get exactly 98 slices: (16000 - 400) // 160 + 1 = 98.
    """
    x = fix_length(x)
    mel = librosa.feature.melspectrogram(
        y=x, sr=SR, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
        window="hann", center=False, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        power=2.0)
    db = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)
    assert db.shape == (N_MELS, N_FRAMES), f"feature shape {db.shape} != {(N_MELS, N_FRAMES)}"
    return db.T.astype(np.float32)          # (40, 98) -> (98, 40): time first


class Normalizer:
    """Shifts/scales each mel band to mean 0, spread 1, using statistics
    computed ONCE from the training set and saved to norm.json. The app loads
    the same file, so training and website normalise identically."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, feats: np.ndarray):          # feats: (N, 98, 40)
        return cls(feats.mean(axis=(0, 1)), feats.std(axis=(0, 1)) + 1e-6)

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        return cls(d["mean"], d["std"])

    def save(self, path):
        json.dump({"mean": self.mean.tolist(), "std": self.std.tolist()},
                  open(path, "w"), indent=1)

    def __call__(self, feats):
        return (feats - self.mean) / self.std


# ---------------------------------------------------------------------------
# Sliding window: scoring recordings longer than 1 s
# ---------------------------------------------------------------------------
def sliding_windows(x: np.ndarray, hop_s: float = 0.1):
    """Yield (start_time_seconds, 1 s window) every hop_s seconds.
    A recording shorter than 1 s becomes a single zero-padded window."""
    if len(x) <= CLIP:
        yield 0.0, fix_length(x)
        return
    hop = int(hop_s * SR)
    for start in range(0, len(x) - CLIP + 1, hop):
        yield start / SR, x[start:start + CLIP]

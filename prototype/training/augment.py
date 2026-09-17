"""
augment.py — make a fresh, randomly altered copy of a training clip.

Called on every training clip, every epoch, so the model never sees exactly
the same picture twice. Never applied to validation or test.

Settings are inherited from demo.md §15.4, where each was MEASURED, with one
bug fixed (see safe_shift) and one addition (external noise, see NoiseBank).

-----------------------------------------------------------------------------
External noise (ESC-50, DEMAND, MUSAN...) — how it is used, and why so
-----------------------------------------------------------------------------
Every speech clip in this dataset came through the INMP441 (except the Speech
Commands ones). External noise did not. If external clips made up a large
part of the NOISE class, the model could learn a shortcut:
"doesn't sound like the INMP441 -> noise", instead of "isn't speech -> noise".
That shortcut is invisible on the test set and fails on the real device,
where EVERYTHING sounds like the INMP441.

So external noise is used in two controlled ways:
  1. As BACKGROUND mixed under clips of ALL classes (keyword, other, noise).
     Because every class receives it equally, "external-sounding" is never a
     clue for any one class. This is where it helps most: the model learns to
     find "Vektora" inside a fan, traffic, a crowd.
  2. As a capped share (config.EXTERNAL_NOISE_CLASS_SHARE) of noise-class
     examples — and each such clip first gets a real INMP441 background clip
     mixed under it, so it carries the mic's own hiss and colouring.
It is never used for validation or test: those stay 100% INMP441, so the
score reflects the real device.
-----------------------------------------------------------------------------
"""

import os

import numpy as np

import features as F


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-9)


class NoiseBank:
    """Holds 1 s noise snippets: INMP441 ones (training split only) and
    external ones (loaded from a folder of any-length, any-rate WAVs)."""

    def __init__(self, inmp441_clips, external_dir=None, rng=None, max_external=3000):
        self.rng = rng or np.random.default_rng(0)
        self.mic = [c for c in inmp441_clips if rms(c) > 1e-5]
        self.ext = []
        if external_dir and os.path.isdir(external_dir):
            paths = [os.path.join(r, f) for r, _, fs in os.walk(external_dir)
                     for f in fs if f.lower().endswith((".wav", ".flac"))]
            self.rng.shuffle(paths)
            for p in paths:
                try:
                    x = F.load_wav(p)
                except Exception as e:                       # corrupt / odd file
                    print(f"  ! skipped {p}: {e}")
                    continue
                for s in range(0, len(x) - F.CLIP + 1, F.CLIP):   # cut into 1 s pieces
                    w = x[s:s + F.CLIP]
                    if rms(w) > 1e-4:                        # skip digital silence
                        self.ext.append(w)
                if len(self.ext) >= max_external:
                    break

    def background(self):
        """A random snippet to mix UNDER a clip: 50/50 mic vs external."""
        pool = self.ext if (self.ext and self.rng.random() < 0.5) else self.mic
        return pool[self.rng.integers(len(pool))]

    def external_noise_example(self):
        """An external clip dressed up with real INMP441 floor underneath."""
        x = self.ext[self.rng.integers(len(self.ext))].copy()
        m = self.mic[self.rng.integers(len(self.mic))]
        return (x + m * (rms(x) / rms(m)) * 10 ** (-self.rng.uniform(0, 20) / 20)).astype(np.float32)


class Augmenter:
    def __init__(self, noise_bank: NoiseBank, rng):
        self.nb = noise_bank
        self.rng = rng

    # --- individual effects -------------------------------------------------
    def speed(self, x):
        """Speed perturbation 0.9-1.15x (demo.md §15.4: 0.82-1.28 made
        "vector" look like a fast "Vektora"; this range was the fix)."""
        rate = self.rng.uniform(0.9, 1.15)
        n = max(1, int(round(len(x) / rate)))
        return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)

    def eq(self, x):
        """Random tone: tilt +-6 dB, bass roll-off 50-250 Hz, treble roll-off
        5.5-8 kHz. Covers unit-to-unit INMP441 variation, enclosure and
        distance colouring, and the Speech Commands phone mics."""
        spec = np.fft.rfft(x)
        f = np.maximum(np.fft.rfftfreq(len(x), 1 / F.SR), 1e-6)
        g = self.rng.uniform(-6, 6) * np.log2(f / 20) / np.log2(8000 / 20)
        hp, lp = self.rng.uniform(50, 250), self.rng.uniform(5500, 8000)
        g -= 12 * np.log2(np.maximum(hp / f, 1.0))
        g -= 12 * np.log2(np.maximum(f / lp, 1.0))
        return np.fft.irfft(spec * 10 ** (g / 20), len(x)).astype(np.float32)

    def safe_shift(self, x, span):
        """Move the word within its 1 s window WITHOUT cutting any of it off.

        THE BUG THIS FIXES (CLAUDE.md §9.5): the demo shifted every clip by up
        to +-0.15 s regardless of where the word was. 17 of the 63 demo clips
        have under 150 ms of free space, so the shift pushed part of the word
        out of the window — training on half-words labelled as whole ones.
        Here the shift is limited to the actual slack on each side.

        `span` is (first, last) sample of the word in x, measured AFTER any
        speed change. The clip is then placed into a fresh 1 s window."""
        a, b = span
        word = x[a:b]
        if len(word) >= F.CLIP:                  # word fills the window: centre it
            return F.fix_length(word)
        # Choose where the word starts in the output window. Any position
        # keeps the whole word; +-0.15 s around its original place, clamped.
        orig = a - (len(x) - F.CLIP) // 2        # original start, in output coordinates
        lo, hi = 0, F.CLIP - len(word)
        start = int(np.clip(orig + self.rng.integers(-2400, 2401), lo, hi))
        # Keep the real surrounding audio (room tone, breath): take the 1 s of
        # x that puts the word at `start`. Where x runs out, fill with quiet
        # INMP441 room tone — NOT zeros.
        #
        # MEASURED BUG, 2026-09-17 first run: filling with zeros taught the
        # model "exact digital silence = speech", because only speech clips
        # got the padding. The 2 real noise clips that happen to contain
        # zero stretches (noise_066/067, edited recordings) then scored 0.99
        # as "Vektora". A live INMP441 never outputs exact zeros anyway.
        src0 = a - start
        out = self.room_tone(x)
        s0, s1 = max(src0, 0), min(src0 + F.CLIP, len(x))
        out[s0 - src0:s1 - src0] = x[s0:s1]
        return out

    def room_tone(self, x):
        """1 s of real INMP441 background, scaled to the quiet level of x
        (its 10th-percentile 25 ms block, ignoring exact-zero blocks)."""
        n = len(x) // 400 * 400
        blocks = np.sqrt(np.mean(x[:n].reshape(-1, 400) ** 2, axis=1))
        blocks = blocks[blocks > 0]
        level = float(np.percentile(blocks, 10)) if len(blocks) else 1e-4
        t = self.nb.mic[self.rng.integers(len(self.nb.mic))]
        t = np.roll(t, int(self.rng.integers(F.CLIP)))
        return (t * (level / rms(t))).astype(np.float32)

    def mix(self, x, noise, snr_db):
        return x + noise * (rms(x) / rms(noise)) * 10 ** (-snr_db / 20)

    # --- the recipe -----------------------------------------------------------
    def speech(self, x):
        """keyword / other-speech clips."""
        if self.rng.random() < 0.8:
            x = self.speed(x)                     # length is now 0.87-1.11 s
        x = self.safe_shift(x, F.active_span(x))  # back to exactly 1 s, word intact
        return self._finish(x)

    def noise(self, x):
        """noise-class clips: any shift is fine, no word to protect."""
        x = np.roll(x, int(self.rng.integers(0, F.CLIP)))
        # Occasionally a dropout of exact zeros (as in edited recordings), so
        # "digital silence" is never a clue that points only at speech.
        if self.rng.random() < 0.15:
            s = int(self.rng.integers(0, F.CLIP - 400))
            x = x.copy()
            x[s:s + int(self.rng.integers(400, 6400))] = 0
        return self._finish(x)

    def _finish(self, x):
        if self.rng.random() < 0.7:
            x = self.eq(x)
        # Level. Device audio is NOT peak-normalised (features.py), so the
        # model must accept the whole range a real INMP441 produces: a quiet
        # voice across the room through to a shout at 20 cm.
        x = x * 10 ** (self.rng.uniform(-28, 6) / 20)
        # Clip every class sometimes, so "clipped" is never a keyword clue
        # (a large share of the team's positive and hard-negative recordings
        # are clipped — audited 2026-09-17).
        if self.rng.random() < 0.3:
            x = x * 10 ** (self.rng.uniform(3, 12) / 20)
        if self.rng.random() < 0.8 and self.nb.mic:
            x = self.mix(x, self.nb.background(), self.rng.uniform(5, 30))
        return np.clip(x, -1, 1).astype(np.float32)

    def spec(self, f):
        """SpecAugment on the normalised picture: blank <=10 frames, <=5 bands."""
        f = f.copy()
        t = int(self.rng.integers(0, 11)); t0 = int(self.rng.integers(0, F.N_FRAMES - t + 1))
        b = int(self.rng.integers(0, 6)); b0 = int(self.rng.integers(0, F.N_MELS - b + 1))
        f[t0:t0 + t, :] = 0
        f[:, b0:b0 + b] = 0
        return f

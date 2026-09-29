"""
plots.py — one consistent look for every graph (training scripts AND the app).

Colours are a colour-blind-safe pair: blue = "Vektora", orange = "not Vektora"
(also used as blue = training data, orange = validation data). Text is always
dark ink, never the series colour, and every coloured mark also has a label,
so no information is carried by colour alone.
"""

import matplotlib

matplotlib.use("Agg")                       # draw to images, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"      # Vektora / training
ORANGE = "#eb6834"    # not Vektora / validation
AQUA = "#1baf7a"

# Sequential colour ramp for spectrograms and the confusion matrix:
# one hue, pale = little energy, dark = lots of energy.
ENERGY = LinearSegmentedColormap.from_list("energy", [SURFACE, "#8fb9ec", BLUE, "#0a2340"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.titlelocation": "left", "font.size": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "lines.linewidth": 2, "legend.frameon": False,
})


def draw_waveform(ax, x, sr, title="Waveform", highlight=None):
    """Sound as air-pressure over time. `highlight` = (start_s, end_s) shades
    the 1 s window the model was most confident about."""
    t = np.arange(len(x)) / sr
    ax.plot(t, x, color=INK2, linewidth=0.6)
    if highlight is not None:
        ax.axvspan(*highlight, color=BLUE, alpha=0.15, linewidth=0)
    ax.set_xlim(0, max(t[-1], 1e-3) if len(t) else 1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("time (s)")
    ax.set_yticks([-1, 0, 1])
    ax.grid(False)
    ax.set_title(title)


def draw_spectrogram(ax, feat, title="Log-mel spectrogram (what the model sees)"):
    """feat: (98 time steps, 40 mel bands). Drawn time left->right, low
    pitch at the bottom, darker = more energy."""
    im = ax.imshow(feat.T, origin="lower", aspect="auto", cmap=ENERGY,
                   extent=[0, 1.0, 0, feat.shape[1]])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mel band (low → high pitch)")
    ax.grid(False)
    ax.set_title(title)
    return im


def draw_gauge(ax, score, threshold):
    """For a single 1 s clip: one bar from 0 to the score, with the threshold."""
    ax.barh(0, 1, color=GRID, height=0.5)
    ax.barh(0, score, color=BLUE if score >= threshold else ORANGE, height=0.5)
    ax.axvline(threshold, color=INK, linestyle="--", linewidth=1.2)
    ax.text(threshold, 0.42, f"threshold {threshold:.2f}", color=INK2, fontsize=8.5, ha="center")
    ax.text(min(score, 0.9) + 0.01, 0, f"{score:.0%}", va="center", color=INK, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.4, 0.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("P(Vektora)")
    ax.set_title("Confidence")


def draw_confidence(ax, times, scores, threshold, raw=None):
    """Sliding-window confidence: one score per 1 s window, plotted at the
    window's centre. Crossing the dashed line = detection. `raw` (optional) is
    the unsmoothed score, drawn faintly so the smoothing is visible."""
    if raw is not None:
        ax.plot(times, raw, color=INK2, linewidth=1, alpha=0.35, label="raw")
    ax.fill_between(times, scores, color=BLUE, alpha=0.12, linewidth=0)
    ax.plot(times, scores, color=BLUE, marker="o", markersize=4,
            label="smoothed (0.3 s)" if raw is not None else None)
    if raw is not None:
        ax.legend(loc="upper right", fontsize=8)
    ax.axhline(threshold, color=INK2, linestyle="--", linewidth=1.2)
    ax.text(times[0] if len(times) else 0, threshold + 0.03,
            f"threshold {threshold:.2f}", color=INK2, fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("time (s) — centre of each 1 s window")
    ax.set_ylabel("P(Vektora)")
    ax.set_title("Confidence over time")

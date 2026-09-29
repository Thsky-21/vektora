"""
model.py — the keyword-spotting network. 3 outputs: keyword / other / noise.

Architecture carried over from the demo (demo.md §15.5), where it was trained
and measured on this team's data; only the last layer changes (1 sigmoid ->
3 softmax, CLAUDE.md §2).

    98x40x1 log-mel
      Conv 16 3x3 -> BN -> ReLU -> MaxPool 2     98x40 -> 49x20   edges, stripes
      Conv 32 3x3 -> BN -> ReLU -> MaxPool 2     49x20 -> 24x10   pieces of sounds (/kt/ burst, vowels)
      Conv 64 3x3 -> BN -> ReLU                  24x10            word-level combinations
      GlobalAveragePooling                        -> 64            "was each pattern present anywhere?"
      Dropout 0.3 -> Dense 3 -> Softmax           -> P(keyword), P(other), P(noise)

Why these pieces:
  * BatchNorm: keeps each layer's inputs in a steady range so training is
    stable. At conversion time it is folded INTO the convolution weights, so
    it costs nothing on the ESP32.
  * Global average pooling instead of Flatten: averages each detector over
    the whole picture, so the answer barely depends on WHERE in the second
    the word sits — which is exactly what a sliding window needs. It also
    saves ~15,000 parameters versus Flatten + Dense.
  * Softmax over 3 classes: the three probabilities sum to 1. The device will
    fire on P(keyword) alone; the other two exist to give the network
    separate "places" to put speech and non-speech, which makes false accepts
    easier to diagnose.
  * Every op here (Conv2D, MaxPool, Mean, FullyConnected, Softmax) is
    supported by TensorFlow Lite Micro with int8 kernels.

Why not the DS-CNN from CLAUDE.md §4 (the deleted sih26172 file)? Depthwise-
separable convolutions cut computation further, but this network is already
~24k parameters (~24 KB int8, budget 40 KB) and has a measured record on this
data. Swap it in later if on-device latency (§5: plain ESP32, no vector
instructions) turns out too slow — that is a measured decision, not a guess.
"""

import tensorflow as tf

import features as F


def build(n_classes: int = 3) -> tf.keras.Model:
    L = tf.keras.layers
    return tf.keras.Sequential([
        L.Input(shape=(F.N_FRAMES, F.N_MELS, 1)),
        L.Conv2D(16, 3, padding="same", use_bias=False), L.BatchNormalization(), L.ReLU(),
        L.MaxPooling2D(2),
        L.Conv2D(32, 3, padding="same", use_bias=False), L.BatchNormalization(), L.ReLU(),
        L.MaxPooling2D(2),
        L.Conv2D(64, 3, padding="same", use_bias=False), L.BatchNormalization(), L.ReLU(),
        L.GlobalAveragePooling2D(),
        L.Dropout(0.3),
        L.Dense(n_classes),
        L.Softmax(),
    ], name="vektora_kws")


if __name__ == "__main__":
    m = build()
    m.summary()
    print(f"parameters: {m.count_params():,}  (~{m.count_params() / 1024:.0f} KB as int8)")

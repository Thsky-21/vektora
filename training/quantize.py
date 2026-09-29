"""
quantize.py — model.keras (float32) -> model_int8.tflite (int8), then PROVE the
int8 model still makes the same decisions.

Run (from training, after train.py):
    ../.venv/Scripts/python.exe quantize.py

What int8 quantization is
-------------------------
The trained model stores every weight as a 32-bit float. The ESP32 has no
vector instructions and only ~300 KB of free RAM, so we store and compute
everything as 8-bit integers instead:

    real_value  ~=  scale * (int8_value - zero_point)

Each tensor gets its own (scale, zero_point). Weights are converted directly.
Activations (the numbers flowing BETWEEN layers) depend on the input, so the
converter must SEE typical inputs to learn their ranges. That is the
"representative dataset" below. If it is unrepresentative (e.g. only loud
clips), real inputs fall outside the learned range, get clipped, and accuracy
drops. That is why it uses the same augmented audio the model trained on.

The result is ~4x smaller and runs with integer maths only. The price is a
small rounding error. This script measures it rather than assuming it is
harmless.

The DEVICE therefore has to do one extra step, which is part of the parity
contract (features.py):
    q = clamp(round(F / input_scale) + input_zero_point, -128, 127)
and read the keyword probability back as
    p = output_scale * (q_out - output_zero_point)
Both (scale, zero_point) pairs are saved in quant_report.json and exported to C.

Outputs (in model/):
  model_int8.tflite   the model that goes on the ESP32
  quant_report.json   quantization params + float-vs-int8 comparison
"""

import json
import os
import sys
from collections import defaultdict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import augment as A  # noqa: E402
import config as C  # noqa: E402
import features as F  # noqa: E402
import train as T  # noqa: E402  (reuses its data loading + augmentation stream)

import tensorflow as tf  # noqa: E402

N_AUGMENTED = 400          # augmented training examples shown to the converter
KW = C.CLASSES.index("keyword")


def representative_features(train_rows, norm, rng):
    """Normalised features the model really sees: every clean training
    original plus N_AUGMENTED freshly augmented ones."""
    nb = A.NoiseBank([r["audio"] for r in train_rows if r["cls"] == "noise"],
                     external_dir=C.EXTERNAL_NOISE_DIR, rng=rng)
    aug = A.Augmenter(nb, rng)
    stream = T.TrainStream(train_rows, aug, norm, C.EXTERNAL_NOISE_CLASS_SHARE if nb.ext else 0.0)
    feats = [norm(F.logmel(r["audio"])) for r in train_rows]
    for _ in range(N_AUGMENTED):
        x, _ = stream.sample(stream.sources[rng.choice(len(stream.sources), p=stream.p)])
        feats.append(norm(F.logmel(x)))
    feats = np.stack(feats)[..., None].astype(np.float32)
    return feats[rng.permutation(len(feats))]


def convert(model, rep):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([f[None]] for f in rep)
    # Integer-only: fail loudly if any op would need a float kernel, because
    # the ESP32 build will only include int8 kernels.
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


class Int8Model:
    def __init__(self, blob):
        # Reference kernels, no XNNPACK delegate: closer to TFLite Micro.
        self.it = tf.lite.Interpreter(
            model_content=blob,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        self.in_scale, self.in_zp = self.inp["quantization"]
        self.out_scale, self.out_zp = self.out["quantization"]

    def predict(self, X):
        """Exactly what the firmware does: quantize -> run -> dequantize."""
        q = np.clip(np.round(X / self.in_scale) + self.in_zp, -128, 127).astype(np.int8)
        out = []
        for x in q:
            self.it.set_tensor(self.inp["index"], x[None])
            self.it.invoke()
            out.append(self.it.get_tensor(self.out["index"])[0])
        return self.out_scale * (np.array(out, np.float32) - self.out_zp)

    def ops(self):
        return sorted({d["op_name"] for d in self.it._get_ops_details()})

    def arena_estimate(self):
        """Rough upper bound on activation memory: the sum of the two largest
        non-constant tensors. The real TFLM arena is measured on the device."""
        sizes = sorted((int(np.prod(t["shape"])) for t in self.it.get_tensor_details()
                        if t["shape"].size and "const" not in t["name"].lower()), reverse=True)
        return sum(sizes[:2])


def keyword_stats(p_kw, y, thr):
    fire, is_kw = p_kw >= thr, y == KW
    return dict(hits=int((fire & is_kw).sum()), misses=int((~fire & is_kw).sum()),
                false_accepts=int((fire & ~is_kw).sum()), correct_rejects=int((~fire & ~is_kw).sum()))


def main():
    rng = np.random.default_rng(C.SEED)
    model = tf.keras.models.load_model(os.path.join(C.MODEL_DIR, "model.keras"))
    norm = F.Normalizer.load(os.path.join(C.MODEL_DIR, "norm.json"))
    with open(os.path.join(C.MODEL_DIR, "report.json")) as f:
        thr = float(json.load(f)["threshold"])

    rows = T.load_rows()
    split = defaultdict(list)
    for r in rows:
        split[r["split"]].append(r)

    rep = representative_features(split["train"], norm, rng)
    print(f"representative set: {len(rep)} examples, feature range {rep.min():.2f} .. {rep.max():.2f}")

    blob = convert(model, rep)
    out_path = os.path.join(C.MODEL_DIR, "model_int8.tflite")
    with open(out_path, "wb") as f:
        f.write(blob)
    q = Int8Model(blob)
    print(f"int8 model: {len(blob):,} bytes   ops: {', '.join(q.ops())}")
    print(f"input  scale {q.in_scale:.6f} zero_point {q.in_zp}   "
          f"(covers {q.in_scale * (-128 - q.in_zp):.2f} .. {q.in_scale * (127 - q.in_zp):.2f})")
    print(f"output scale {q.out_scale:.6f} zero_point {q.out_zp}")

    report = dict(
        tflite_bytes=len(blob), ops=q.ops(), threshold=thr,
        input=dict(scale=float(q.in_scale), zero_point=int(q.in_zp),
                   shape=[int(s) for s in q.inp["shape"]]),
        output=dict(scale=float(q.out_scale), zero_point=int(q.out_zp),
                    shape=[int(s) for s in q.out["shape"]], classes=C.CLASSES),
        representative_examples=len(rep),
        activation_bytes_rough=q.arena_estimate(),
        # Share of features the int8 input cannot represent (would be clipped).
        input_clip_fraction={}, compare={},
    )

    for name in ("val", "test"):
        X = T.featurize(split[name], norm)
        y = np.array([r["y"] for r in split[name]])
        pf = model.predict(X, verbose=0)
        pq = q.predict(X)
        lo, hi = q.in_scale * (-128 - q.in_zp), q.in_scale * (127 - q.in_zp)
        report["input_clip_fraction"][name] = round(float(((X < lo) | (X > hi)).mean()), 5)
        d = np.abs(pf[:, KW] - pq[:, KW])
        report["compare"][name] = dict(
            n=len(y),
            p_keyword_abs_diff_max=round(float(d.max()), 4),
            p_keyword_abs_diff_mean=round(float(d.mean()), 4),
            argmax_agreement=round(float((pf.argmax(1) == pq.argmax(1)).mean()), 4),
            fire_decision_flips=int(((pf[:, KW] >= thr) != (pq[:, KW] >= thr)).sum()),
            accuracy_3class_float=round(float((pf.argmax(1) == y).mean()), 4),
            accuracy_3class_int8=round(float((pq.argmax(1) == y).mean()), 4),
            keyword_float=keyword_stats(pf[:, KW], y, thr),
            keyword_int8=keyword_stats(pq[:, KW], y, thr),
        )

    with open(os.path.join(C.MODEL_DIR, "quant_report.json"), "w") as f:
        json.dump(report, f, indent=1)

    print(f"\n=== float32 vs int8 (threshold {thr}) ===")
    for name, c in report["compare"].items():
        print(f"{name}: n={c['n']}  |dP(keyword)| max {c['p_keyword_abs_diff_max']:.4f} "
              f"mean {c['p_keyword_abs_diff_mean']:.4f}  argmax agree {c['argmax_agreement']:.1%}  "
              f"fire flips {c['fire_decision_flips']}  input clipped {report['input_clip_fraction'][name]:.3%}")
        print(f"   3-class acc  float {c['accuracy_3class_float']:.1%}  int8 {c['accuracy_3class_int8']:.1%}")
        print(f"   keyword      float {c['keyword_float']}")
        print(f"                int8  {c['keyword_int8']}")


if __name__ == "__main__":
    main()

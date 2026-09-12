# CLAUDE.md — Vektora / SIH26172

Guidance for Claude Code sessions in this repo. See `CONTEXT.md` for what the
project actually is; this file is operational.

> **RESUMING? Read the "⏵ RESUME HERE" section at the top of `demo.md`.** It
> holds the current state, the one open problem (weak detection through the
> live browser mic — needs more "Teach it" recordings), the next actions, the
> dead ends already tried, and the diagnostics to re-run. Presentation: today.

> **2026-09-11 PIVOT — read `demo.md` first.** The user needs a demo for a
> presentation on 2026-09-12: a binary "Vektora / not Vektora" CNN trained on
> the team's own recordings (GitHub `Thsky-21/vektora`), served by a Streamlit
> app on a Hugging Face Docker Space. `demo.md` is the active plan and
> supersedes §2/§3/§6/§8 *for the demo*. The old ESP32 plan below is paused,
> not cancelled. **`sih26172/` was DELETED 2026-09-11 at the user's request**
> (so §4's file list is historical; `features.py`/`model_dscnn.py` no longer
> exist). Demo code lives in `demo/`. §1 (venv rules) and §7 (working style)
> still apply in full.
>
> **Version trap:** `pip install streamlit` unpinned pulls protobuf 7 and
> breaks TF 2.15 (needs protobuf < 5). Always pin `protobuf==4.25.9`; with it,
> streamlit resolves to 1.60.0 (installed in `.venv/`, verified).

> **Resuming the ESP32 plan? Read §8 first.** It is the full log of the 2026-09-10 session —
> every decision made, with the reasoning behind it, what was actually tested,
> and the one question still waiting on an answer (§3.1). §2 gives the *what*;
> §8 gives the *why*, so nothing has to be re-argued from scratch.

---

## 1. Environment — read this first

**Always use the project venv. Never install into global Python.**

```bash
.venv/Scripts/python.exe scripts/foo.py     # correct
python scripts/foo.py                       # WRONG - global interpreter
```

The global Python 3.10.7 is shared with unrelated work and hosts **jax 0.6.2,
torch 2.11.0, and mediapipe 0.10.21** with mutually conflicting version pins.
A global `pip install tensorflow` in this session downgraded `ml-dtypes`
0.5.x -> 0.3.2 and broke jax; it had to be manually repaired. Do not repeat this.

Known pre-existing global conflict, unsatisfiable, **not ours to fix**:
`mediapipe` requires protobuf<5 while `grpcio-status` requires >=5.26.1.

### The jax/TFLite trap (specific, will recur)

`tensorflow/lite/python/util.py` optionally imports jax:

```python
try:
    from jax import xla_computation as _xla_computation
except ImportError:
    _xla_computation = None
```

The guard catches only `ImportError`. A *present but broken* jax raises
`AttributeError`, which escapes and takes down `tf.lite` — the exact module
needed for quantization. If `import tensorflow` fails with
`ml_dtypes has no attribute float8_e3m4`, this is the cause.

### Versions

TensorFlow is pinned to **2.15.x** deliberately. Do not upgrade to 2.16+ without
asking: those pull in Keras 3, which changes the `tf.lite` conversion path used
in the quantization step, and diverges from the project guide's examples.

---

## 2. Decisions locked (do not silently revisit)

| Decision | Value | Notes |
|---|---|---|
| Model output classes | **3** — keyword / other speech / noise | User-chosen. Separating "speech that isn't the keyword" from "silence" is what protects the false-accept target |
| Python environment | **Isolated venv** at `.venv/` | User-chosen, after the global install broke jax |
| Feature shape | 98 x 40 x 1 | 1 s audio, 10 ms hop, 40 mel bands. Standard KWS convention |
| **Keyword** | **"Vektora"** | User-chosen 2026-09-10. 3 syllables fills the 98-frame window; the /kt/ stop-burst is a strong spectrogram landmark; not a real word in any language, so near-zero collision with ordinary speech |
| Data strategy | **Bootstrap first** on Google Speech Commands, stand-in word **"visual"** | User-chosen 2026-09-10. Validates the full chain before anyone records. "visual" shares the /v/ voiced-fricative onset with "Vektora", so the low-energy-onset problem surfaces during bootstrap. Real clips swap in later |
| Canonical scaffold | **`sih26172/`** | User-chosen 2026-09-10. `voice_activator/` deleted (all 39 files were 0 bytes). `ring_buffer.h` and `server/asr_whisper.py` were carried over as stubs |

## 3. Decisions still OPEN — ask, do not assume

All five *founding* decisions are closed and live in §2. What remains open:

1. **UNANSWERED QUESTION, asked at the end of the 2026-09-10 session.** Should
   `build_dataset.py` download Speech Commands v2 (~2.3 GB) itself, or expect
   the user to have unpacked it somewhere and just take a path? **Ask this
   before writing `build_dataset.py`.**
2. **`FMAX = 7600` is provisional** — a real trade-off, deliberately deferred.
   Revisit once there is a validation accuracy number to compare against. Full
   reasoning in §8.6.
3. **`FEATURE_MEAN` / `FEATURE_SCALE` in `features.py` are placeholders**
   (−40.0 / 30.0). They must be computed by `compute_norm_stats()` once the
   dataset exists, pasted back into `features.py`, **and** copied into
   `feature_extract.c`. Until then, normalised features will not be centred.
4. **When to record real "Vektora" clips.** The bootstrap decision defers this,
   it does not cancel it. The keyword is locked, so recording is safe to start
   at any time.
5. **The post-processing triple** — smoothing window, confidence threshold,
   refractory period. Not needed until firmware, but this is where the
   false-accept target is actually won (§6 risk 2).

---

## 4. Repo state

`sih26172/` is the single canonical scaffold; `voice_activator/` was deleted
2026-09-10. **Almost every source file is still 0 bytes** — do not assume any
file has content; check with `wc -l` before reading. Files with real content:

- `CONTEXT.md` — project context
- `CLAUDE.md` — this file
- `.gitignore` (root) — blocks `.venv/`, the 2.3 GB dataset, `*.npy`, `*.tflite`
- `sih26172/training/scripts/model_dscnn.py` — DS-CNN definition. **Tested**:
  23,747 params, ~23 KB int8, stem downsamples 98x40 -> 49x20
- `sih26172/training/scripts/features.py` — log-Mel extraction. **Tested**:
  self-checks pass, and a (3,98,40,1) batch feeds model_dscnn cleanly.
  Holds the PARITY CONTRACT block that feature_extract.c must mirror exactly

Everything else remains an empty stub.

---

## 5. Hardware reality

The board is an **ESP32-WROOM-32 DevKit V1** (plain ESP32), not an ESP32-S3.

`sih26172/README.md` said `idf.py set-target esp32s3`; corrected to `esp32`
on 2026-09-10. The difference is not cosmetic: the S3 has vector
instructions that accelerate neural-network inference and the plain ESP32 does
not. Any latency or CPU figure quoted from S3 sources will be optimistic for
this board. Expect roughly 2-3x slower inference than S3 benchmarks suggest.

---

## 6. Build order and where the risk actually is

```
features.py  ->  build_dataset.py  ->  train.py  ->  quantize.py  ->  firmware
  (DONE)            (next)            (needs
                                       dataset)
```

`build_dataset.py` is unblocked: the data decision is made (bootstrap on Google
Speech Commands, stand-in word "visual"). It must download/expect Speech
Commands v2, map the 35 words onto our 3 classes, and include "vector" /
"victor" / "vectra" as hard negatives in the *other speech* class once real
"Vektora" clips exist. Two placeholders in features.py (`FEATURE_MEAN`,
`FEATURE_SCALE`) must be filled from `compute_norm_stats()` after the dataset
is built, and then copied into the C code.

The neural network is **not** the hard part — it is ~40 lines. The two genuine
risks, both frequently underestimated:

1. **Feature-extraction parity.** The log-Mel spectrogram is computed twice:
   Python/librosa for training, hand-written C on the device for inference. If
   they disagree (fmin/fmax, normalisation, window, padding), the model sees a
   different picture than it trained on and accuracy collapses **silently, with
   no error**. Validate C output against Python output element-by-element on the
   same WAV before trusting any on-device result.

2. **Post-processing, not the model, sets the false-accept rate.** The
   smoothing window, confidence threshold and refractory period are where the
   `< 1 false accept/hour` target is won or lost.

### Keyword-specific traps ("Vektora")

- **"vector" is a near-collision.** It differs from "vektora" only in the final
  vowel, and this is an ML project — the word will be spoken near a live device
  constantly. "vector", "victor" and "vectra" MUST go into the *other speech*
  class as deliberate hard negatives. Without them the model has no reason to
  learn that the final vowel carries the distinction.
- **`/v/` is a low-energy onset.** A voiced fricative may fail to trip the
  stage-1 VAD energy gate, clipping the start of the word before detection
  runs. The 0.5 s pre-roll protects the *streamed* audio, not the *detection*
  window. Set the VAD threshold with this in mind.
- **Verify duration at the first recording session.** Three syllables spoken
  slowly can exceed the 980 ms window. If the whole word does not fit, the
  model only ever trains on partial words.

---

## 7. Working style for this project

The user is **learning while building** and has explicitly asked to make every
design decision themselves. Therefore:

- Write the code, but surface each real decision **before** committing to it,
  with the trade-offs explained plainly — not just a recommendation.
- Explain *why* an architectural choice exists (e.g. why depthwise-separable
  convolutions, why global average pooling instead of flatten), not just what
  the code does. Comment source files more heavily than normal.
- Do not assume prior ML knowledge. Do not skip the reasoning.
- Flag it honestly when a guide, README, or Claude's own earlier action is
  wrong, rather than quietly working around it.

---

## 8. Session log — 2026-09-10 (resume point)

This section exists because the reasoning behind the session's decisions was
delivered in chat and **the user did not read it**. It is recorded here in full
so no decision below has to be re-derived or re-argued. Read it before touching
`features.py` or writing `build_dataset.py`.

### 8.1 Environment — verified, not assumed

Checked directly rather than trusted from §1:

- `.venv/Scripts/python.exe` is Python 3.10.7 with **tensorflow 2.15.1**,
  keras 2.15.0, librosa 0.11.0, numpy 1.26.4, ml-dtypes 0.3.2.
- `import tensorflow` succeeds and `tf.lite.TFLiteConverter` resolves. The
  jax/TFLite trap in §1 is **not** active inside the venv, because no jax is
  installed there — so the broken optional import cannot fire. The trap remains
  real for the global interpreter. Do not install jax into `.venv/`.
- The global interpreter was not touched at any point this session.

### 8.2 Keyword: "Vektora" — why, and why not the others

The candidate list in earlier revisions was Netra / Vyom / Daksh. The user
chose **"Vektora"** (the project's own name) instead. It is the strongest of the
four, for three reasons that all trace back to the `< 1 false accept/hour`
target:

- **Three syllables is the biggest win.** The model sees a 1 s window as 98
  frames. A word occupying 60–70 of them gives the detector far more evidence
  than one occupying 25. This matters concretely because the post-processing
  smoothing window (§6 risk 2) averages confidence over consecutive frames: a
  long keyword can hold high confidence across many frames, while a short one
  produces a brief spike that is statistically indistinguishable from the
  spikes ordinary speech throws off constantly. **This is why `Vyom` was the
  weakest candidate** — one short vowel-dominated syllable — and why `Daksh`
  was mid: a distinctive `/kʃ/` cluster, but only one syllable of evidence.
- **The `/kt/` cluster is a gift to a convolutional network.** In "ve**kt**ora"
  there is a stop closure (brief near-silence) followed by a broadband burst.
  On a log-Mel spectrogram that is an unmistakable landmark: a dark vertical
  stripe followed by a bright one. Convolutional filters latch onto edges like
  that very efficiently — far better than a smooth vowel glide.
- **It is not a real word in any language, and that beats `Netra`.** This was
  the decisive argument. "Netra" is an actual Sanskrit/Hindi word (eye) that
  occurs in ordinary speech and inside compound names. Every real-word keyword
  is a standing false-accept source, because the device will genuinely hear it
  and be *correct* to fire — which still counts against the target. "Vektora"
  occurs in no language, so essentially the only way to hear it is on purpose.

The two costs of this choice are recorded as traps at the end of §6 ("vector"
as a near-collision requiring hard negatives; the low-energy `/v/` onset vs the
stage-1 energy gate). Neither is a reason to reconsider the keyword — both are
reasons to write specific things down, which has been done.

### 8.3 Data strategy: bootstrap first, stand-in word "visual"

Chosen over recording first. The reasoning: recording is cheap to *redo* only
if the pipeline is known-good. If the team records first and a
feature-extraction or labelling bug is found later, the session is wasted.
Bootstrapping proves the whole chain (features → dataset → train → quantize →
C parity) before anyone opens a microphone, so no recording session can be
invalidated by a pipeline bug.

**"visual" was chosen from the 35 Speech Commands words specifically because it
shares the `/v/` voiced-fricative onset with "Vektora".** That is not cosmetic:
it means the low-energy-onset problem (trap 2 in §6) surfaces during bootstrap,
against free public data, rather than being discovered on real recordings.

Note that Speech Commands is needed **regardless** of this decision — it is the
source for the *other speech* class and for the "vector"/"victor" hard
negatives. The decision was only ever about ordering.

### 8.4 Canonical scaffold: `sih26172/`

`voice_activator/` was deleted after verifying, programmatically, that **all 39
of its files were 0 bytes** — so nothing was lost. `sih26172/` won because it
held the only non-empty source file, a written README, its own `.gitignore`, a
`docs/measurements/` folder for the INA219 power data, and a
`firmware/` + `training/` + `server/` split that maps exactly onto the three
deliverables in `CONTEXT.md`.

Two filenames existed **only** in `voice_activator/` and were carried over as
empty stubs rather than lost: `firmware/main/ring_buffer.h` and
`server/asr_whisper.py`. `ring_buffer.h` matters — it is the 0.5 s pre-roll
buffer that `CONTEXT.md` requires, and `sih26172/` had no equivalent file.

### 8.5 `sih26172/README.md` — three errors fixed

- `idf.py set-target esp32s3` → `esp32`. This was the known inconsistency in
  §5. A hardware note about the 2–3× inference penalty was added alongside it.
- The `xxd` command's paths were relative to `sih26172/`, but the code block
  it sits in runs from `firmware/`. Corrected to `../training/...`.
- Every command used bare `pip` / `python`, i.e. the global interpreter that §1
  exists to forbid. All now use `../../.venv/Scripts/python.exe`.

### 8.6 `features.py` — written, and the three parity choices in it

This is the §6 risk-1 file. It deliberately hand-writes the spectrogram in
plain NumPy instead of calling `librosa.feature.melspectrogram()`, so each step
maps one-to-one onto the C that must mirror it. A convenience call would have
been three lines shorter and would have hidden exactly the details that break
parity. The constants live in one block marked **PARITY CONTRACT**; treat it as
a header shared with `firmware/main/feature_extract.c`.

Three choices were made for parity reasons. All are reversible, but each must
change on **both** sides in the same commit:

- **`center=False`.** librosa defaults to `center=True`, which reflect-pads 200
  samples onto both ends before framing. That is wrong here twice over: it is
  unimplementable on a live stream (the device cannot reflect-pad the future),
  and it changes the frame count from 98 to 101, breaking the locked shape.
  Useful confirmation: `(16000-400)//160 + 1 = 98` **exactly** matches the
  locked 98 × 40 × 1, which is strong evidence the constants are mutually
  consistent. **If a future session ever sees 101 frames, `center` has been
  flipped back to `True`.**
- **HTK mel formula (`htk=True`), un-normalised filters (`norm=None`).** There
  are two mel curves in common use and they are not the same: HTK is
  `2595*log10(1+f/700)`, one line of C; Slaney's is piecewise (linear below
  1 kHz, log above) plus per-filter area normalisation. librosa defaults to
  Slaney. Neither is more "correct", but using Slaney in Python and HTK in C
  would shift every band's centre frequency — precisely the silent failure this
  file exists to prevent. HTK was chosen because it is trivial to reproduce.
- **Fixed affine scaling, not per-clip CMVN.** Per-clip normalisation
  (subtract the clip's own mean, divide by its own std) gives free loudness
  invariance and is common in ASR. Rejected because on a streaming device it
  breaks an invariant we need: with per-clip stats, **the same word produces
  different features depending on whatever else shares its 1 s window.** That
  makes behaviour context-dependent in a way that is miserable to reason about
  when chasing false accepts, and it destabilises the input range that int8
  quantization depends on. The cost is that loudness variation must instead be
  handled by gain-jitter augmentation in `augment.py`.

**The `FMAX = 7600` trade-off (still open — see §3.2).** "Vektora" has two
discriminative cues at opposite ends of the spectrum: the `/kt/` burst is
broadband with much of its energy above 4 kHz, while the final `/-a/` vowel is
low-formant and is *the only thing* separating "vektora" from "vector".
`FMAX = 4000` would spend all 40 bands below 4 kHz — excellent formant
resolution, burst signature discarded. `FMAX = 7600` (just under the 8 kHz
Nyquist) keeps both, and because the mel scale is logarithmic the bands stay
bunched at the low end anyway, so widening the range costs less low-frequency
resolution than it appears to. Hence 7600, provisionally. Revisit with a
validation number in hand, not before.

Also note `fix_length()` centre-pads and centre-crops rather than appending,
so a word in a short clip lands mid-window where words in longer clips tend to
sit. Position invariance is then supplied by the model's global average pooling
plus deliberate time-shift augmentation.

### 8.7 What was actually tested (not merely written)

- `model_dscnn.py` had been written in a previous session but **never run** —
  TF was still installing. It now runs: 23,747 params, 92.8 KB float32,
  ~23.2 KB int8 (budget is 40 KB), output `(None, 3)`, stem downsampling
  98 × 40 → 49 × 20 as designed.
- `features.py` self-checks pass: derived frame count is 98; all 40 mel bands
  receive energy from at least one FFT bin (a band falling between bins would
  be a silently dead input feature); and a 440 Hz test tone lands in mel band 7
  (centre ≈ 417 Hz), which proves the filterbank is not transposed or reversed.
- **End-to-end**: a `(3, 98, 40, 1)` batch built by `features.py` feeds
  `model_dscnn.py` and returns `(3, 3)` probability rows summing to 1.0. Values
  are ≈0.333 each, which is correct and expected — the model is untrained.

### 8.8 Where to start next session

1. **Ask the unanswered question in §3.1** (dataset download vs. path).
2. Write `build_dataset.py`: expect/download Speech Commands v2, map the 35
   words onto the 3 locked classes, hold out "visual" as the stand-in keyword,
   and reserve "vector"/"victor"/"vectra" as hard negatives for the *other
   speech* class.
3. Run `compute_norm_stats()` and fill the two placeholders (§3.3).
4. Nothing is committed yet. The repo is one commit (`9aa2c58 Initial commit`)
   plus untracked work. Committing has not been asked for — ask first.

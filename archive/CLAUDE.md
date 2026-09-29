# CLAUDE.md — Vektora / SIH26172

Guidance for Claude Code sessions in this repo. See `CONTEXT.md` for what the
project actually is; this file is operational.

> **▶ START HERE (updated 2026-09-17, end of session): read §10 first.**
> §10 is the complete log of the 2026-09-17 session, which started the real
> prototype from scratch in `prototype/`. It covers the team dataset at
> `C:\Users\User\documents\data`, the training pipeline that was built, three
> training runs, the data cleanup, the open questions waiting on the user and
> the exact next steps (§10.12). §9 (the data plan) is still the background
> reference. §2–§8 are older history; where they conflict with §10, §10 wins.
>
> **Standing user instruction (2026-09-17), still in force:** do NOT change
> the model architecture, learning rate, optimizer, threshold rule, batch
> size, feature extraction or number of classes, and do NOT add dropout
> augmentation, until the user explicitly lifts it. Data cleanup and
> evaluation come first. Do NOT start a training run without being asked.

> **2026-09-17 — `prototype/` IS THE CANONICAL FOLDER.** Created at the
> user's request for the real working device. Layout: `firmware/` (ESP-IDF C),
> `training/`, `server/`, `data/` (git-ignored — voices, public repo),
> `docs/measurements/`. See `prototype/README.md`. It replaces the deleted
> `sih26172/`; wherever §2–§8 say `sih26172/`, read `prototype/`.

> **2026-09-15 — BACK TO THE FULL PROTOTYPE.** The 2026-09-12 presentation is
> done; the Streamlit demo shipped (`demo/`, see `demo.md` §15 for the full
> build record of the model that is actually committed). Work has resumed on
> the real SIH26172 deliverable: the keyword detector running on the
> **ESP32-WROOM-32**, with the cascade, the firmware and the server. §2–§8
> below are live again. `demo/` is not thrown away — its `audio_utils.py`
> holds the same 98×40 log-mel contract, and its CNN is the starting point
> for the on-device model.
>
> **START HERE TOMORROW: §9** — the full data plan for the two-model cascade,
> and the measured answer on the 1-second recording window. It ends with two
> next actions (§9.6) that have not been chosen yet.

> **RESUMING? Read the "⏵ CURRENT VERSION" section at the top of `demo.md`.**
> It describes the model that is actually trained and committed, with every
> number measured. `demo.md` §15.10 lists the experiments already tried and
> rejected — read it before proposing augmentation changes.

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

## 0. Team and scope — who does what

**The user owns the software. Teammates own the hardware.**

| Area | Owner | Covers |
|---|---|---|
| ML models | **User** | Dataset design, recording protocol, training, augmentation, quantization, thresholds, evaluation |
| Backend / server / cloud | **User** | Streaming receiver, ASR, dashboard, hosting, deployment, APIs |
| Firmware C | **Shared — assume the user unless told otherwise** | Feature-extraction parity (§6 risk 1), TFLite Micro integration, the cascade logic, I2S capture |
| Board bring-up and wiring | **Teammates** | INMP441, INA219, TP4056, 18650, LEDs, soldering, the power-measurement rig |

**Default assumption for every session in this repo: the work is software** —
ML models, training pipelines, the server, the cloud side. Lead with that, and
frame explanations for someone building the software half.

**But the user does hardware too whenever the team asks them to.** If they
bring an I2S register, an ESP-IDF menuconfig, an INA219 read or a wiring
question, it is fully in scope — answer it properly. Never redirect a hardware
question back to the teammates.

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
| Canonical scaffold | ~~`sih26172/`~~ → **`prototype/`** | `sih26172/` was deleted 2026-09-11. `prototype/` was created 2026-09-17 and is canonical (§10) |
| Training data | **Team dataset** at `C:\Users\User\documents\data` | 2026-09-17: replaces the "bootstrap on Speech Commands / visual" plan below. The team already has real "Vektora" recordings (§10.2) |
| Augmentation | **Online, from originals only** (user option "b") | 2026-09-17. Pre-made `aug_*` files are ignored |

## 3. Decisions still OPEN — ask, do not assume

> **2026-09-17: the current open questions are in §10.12.** Items 1–4 below
> are obsolete. There is no `build_dataset.py`, the team dataset exists, and
> the features live in `prototype/training/features.py` with a per-band
> normaliser instead of the two scalar placeholders. `FMAX = 7600` was kept.
> Item 5 (post-processing) is still open.

All five *founding* decisions are closed and live in §2. What remained open on 2026-09-10:

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

> **HISTORICAL. For the current repo state see §10.3.** Everything below
> describes `sih26172/`, which no longer exists.

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


---

## 9. Session log — 2026-09-15 — the full-prototype data plan

**Status: this is the plan to start from. Nothing here has been built yet.**
The user asked for this analysis to be logged verbatim so they can read it cold
the next morning and begin work. It is the answer to two questions: *"for the
two-model cascade to work efficiently, list all the types of data we want, be
specific"* and *"we record 'Vektora' and similar-sounding words for 1 second
each — is that good, or should we do it differently?"*

Scope reminder: the 2026-09-12 presentation is done. This section is about the
**real SIH26172 deliverable** — the detector running on the ESP32-WROOM-32.
§0 governs who does what.

---

### 9.1 A correction to the two-model picture

The user described it as *"one expensive DS-CNN, one to filter out noise or
human speech."* Close, but the split that actually works on a plain ESP32 has
**three** stages, and **only one of them must be a neural network**:

| Stage | What it asks | Cost | How often it runs | Learned? |
|---|---|---|---|---|
| **0 — Energy gate** | "Is there any sound at all?" | ~50 ops per 10 ms frame | Always. This is 90%+ of the device's life | **No.** Adaptive noise floor + RMS |
| **1 — Speech gate** | "Is that sound *speech*, or is it a fan / door / traffic?" | ~1–5% CPU | Only when stage 0 opens | **Optional** — classical or a tiny NN |
| **2 — KWS (DS-CNN)** | "Is that speech *'Vektora'*?" | 30–80 ms per inference | Only when stage 1 opens | **Yes.** ~24k params, int8 |

Two consequences that change the data plan:

**Stage 1 does not have to be a trained model, and there is a real argument for
it not being one.** Every trained model on the device must be quantized,
converted, and — the expensive part — have its feature extraction
**reimplemented in C and validated element-by-element against Python**. That is
§6 risk 1, the silent-failure risk. Doing it once is a day of careful work.
Doing it twice is two days plus twice the surface area for a parity bug, and a
parity bug in the *gate* looks exactly like "the wake word randomly doesn't
work."

**Recommendation: build stage 1 classically first** — band energies +
zero-crossing rate + spectral flatness against an adaptive noise floor, roughly
80 lines of C, no training, no parity contract. Measure what fraction of real
ambient audio it lets through. Only if it passes too much — say more than
15–20% of a normal room hour — train a tiny NN gate. The classical one may well
be enough, saving the entire second parity exercise.

The data list below covers **both** options, because the user may choose the
learned gate, and because *tuning* the classical gate needs almost the same
recordings that *training* the learned one would.

**The cascade's job is CPU duty cycle, not accuracy.** The gate exists so the
DS-CNN runs 2% of the time instead of 100%.

---

### 9.2 Data for Stage 1 — the speech/noise gate

The governing principle, and it is the opposite of normal ML instinct:

> **This gate must be tuned for near-100% recall on speech, not for accuracy.**
> Letting noise through costs a few milliseconds of wasted CPU — invisible.
> Rejecting a frame of real speech costs a missed wake word — the user sees the
> device ignore them. Target ~99.5% speech recall at whatever false-alarm rate
> that forces. A gate that is "95% accurate" by being even-handed is a worse
> product than one that is 99.5% / 60%.

#### 9.2a Speech (the positive class) — nearly free

| Type | Source | Amount | Why this specifically |
|---|---|---|---|
| Clean read speech | LibriSpeech, Common Voice | 3–5 h | The easy baseline case |
| **Indian-accented English** | Common Voice (filter `accent=india`), NPTEL lecture audio | 2–3 h | The device will be judged by Indian speakers. Rhythm and vowel quality differ enough to matter for a *frame-level* speech detector |
| **Regional languages** — Kannada, Hindi, Telugu, Tamil | Common Voice, IndicTTS, or record teammates | 1–2 h | People code-switch mid-sentence. If the gate has only heard English, Kannada conversation is out-of-distribution and it may close mid-word |
| Isolated short words | Speech Commands v2 (needed anyway) | Free | Short utterances with abrupt onsets — the hardest case for a gate with a hangover timer |
| **Quiet / far-field speech** | Recorded by us at 2–3 m | 1 h | **The single most important category.** A gate tuned on close-mic speech will close on someone talking from across the room, which is precisely the demo scenario |
| Whispered and soft speech | Recorded by us | 15 min | The `/v/` onset problem generalises: quiet speech is where gates fail |

#### 9.2b Non-speech (the negative class) — must be recorded by us

Public noise corpora (MUSAN, ESC-50, DEMAND, WHAM!) give 100+ hours free and
should absolutely be used. But they cannot substitute for the following,
because the gate operates on raw energy and band shape, where **the
microphone's own characteristics dominate**:

| Type | Amount | Why it can't be downloaded |
|---|---|---|
| **INMP441 noise floor, in a silent room** | 10 min | This mic has its own self-noise and a DC offset requiring a high-pass. Digital silence and laptop-mic silence are *nothing like it*. The energy threshold is set against this number |
| **The actual demo venue** — classroom, lab, corridor | 2–3 h | Reverberation and HVAC signature are room-specific |
| **Ceiling fan** at every speed | 20 min | Near-universal in Indian rooms, broadband, and the blade-pass thump is a periodic low-frequency signal that naive VADs read as voiced speech |
| Laptop fans, projector fan | 20 min | Present at every demo, right next to the device |
| **50 Hz mains hum** and switching-supply whine | 10 min | India is 50 Hz. Most public noise corpora are recorded in 60 Hz countries. Also: the TP4056 and the ESP32's own regulator inject this, so record it *with the real power arrangement* |
| Impulsive: door slams, chairs scraping, dropped objects, footsteps, keyboard typing, paper | 30 min | Sharp onsets are what trigger energy gates falsely |
| Handling noise, cable bump, table knock | 10 min | Conducted through the enclosure, huge amplitude, not acoustic |
| **Non-speech human sounds**: coughs, throat clearing, laughter, sighs, breathing on the mic | 20 min | Genuinely ambiguous. Decide deliberately which side they go on — put them in *noise* for the gate but in *other speech* for the KWS |
| Music, instrumental **and vocal** | 30 min | Vocal music is a serious false-accept source at both stages |
| **Replayed audio**: TV, YouTube, phone speaker playing into the INMP441 | 1–2 h | Extremely cheap to generate — leave it running unattended. Gives far-field speech-through-a-speaker, a realistic and nasty case |

#### 9.2c What the gate's data must be *labelled* with

Not clip labels — **frame-level boundaries**. For 30–60 minutes of continuous
mixed audio, marks saying "speech from 12.34 s to 14.90 s." This is the tedious
part. Two shortcuts:

- Use a strong offline VAD (Silero VAD, or WebRTC VAD at its most aggressive)
  to auto-label, then hand-correct. Roughly 10x faster than labelling from
  scratch.
- Or construct it synthetically: take clean speech, mix it into our recorded
  room noise at known SNRs, and the boundaries are known exactly because we
  placed them. **Do both** — the synthetic set for training volume, the
  hand-corrected real set for honest evaluation.

**Critical SNR point:** mix speech into noise across **0 to 30 dB SNR**, not
just clean. A gate trained only on clean speech has never had to find a voice
inside fan noise, which is the only condition that matters.

---

### 9.3 Data for Stage 2 — the DS-CNN keyword spotter

This is where the effort should go. Current state: **63 positives from 5–6
speakers, 27 soft negatives from 2–3 speakers.** That is roughly 5% of what a
robust product needs, and every limitation `demo.md` §15 records traces back to
it.

#### 9.3a Positives — "Vektora"

**Target: 15–25 speakers x ~40 utterances = 600–1000 clips.** At least 300 of
them recorded **through the INMP441 on the actual board**. For a college team
this is a week of asking people in the corridor, not a research project.

Why 15+ speakers matters more than 1000 clips from 6 people: the model's job is
to generalise to a judge it has never heard. Speaker count is the axis that
buys that; more utterances from a speaker already in the set buys much less.

Per speaker, vary these **eleven axes**. Do not attempt a full factorial — use
the ~8-minute script below:

| # | Axis | Levels to cover | Why it matters here |
|---|---|---|---|
| 1 | Speaker | 15–25 people | Pitch range is the dominant spectral variable |
| 2 | Sex / voice pitch | Deliberately balanced | The existing 63 clips may be skewed; a high-pitched voice shifts every formant |
| 3 | **Speaking rate** | slow, normal, fast/casual | **Already burned us once.** `demo.md` §15.4 records that the detector caught "Vektora" spoken loudly and slowly and missed it at normal pace, because all 63 recordings are careful deliberate pronunciations. Speed perturbation was a patch; real fast speech is the fix |
| 4 | **Distance** | 0.3 m, 1 m, 2 m, 3 m | The gap `demo.md` §15 calls "the one honest limitation". Level, reverberation and SNR all change together |
| 5 | Loudness | soft, normal, called loudly | Independent of distance — someone can murmur from 3 m |
| 6 | **Carrier context** | isolated; `"Vektora, what's the weather?"`; mid-sentence; repeated `"Vektora... Vektora"`; hesitant `"um, Vektora"` | See §9.5. Real use is *never* an isolated word, and coarticulation changes the final `/-a/` — the exact phoneme separating it from "vector" |
| 7 | Off-axis angle | facing the mic, 45°, 90°, facing away | MEMS mics roll off high frequencies off-axis, and the `/kt/` burst is high-frequency |
| 8 | Background condition | quiet, fan on, music playing, another person talking | Recorded-in noise beats mixed-in noise |
| 9 | Accent / first language | as many as campus offers | Direct robustness to judges |
| 10 | Vocal state | normal, tired, slightly hoarse, after exertion | Cheap to collect at the end of a session |
| 11 | **Recording chain** | INMP441 primarily; some laptop/phone | Prevents the model learning the *recording chain* instead of the word — `demo.md` §15.4's random-EQ augmentation exists to fake this; real variety is better |

**Concrete ~8-minute per-speaker script (~44 positives + ~20 negatives):**

```
At 1 m, quiet room:      6 x normal pace
                         4 x fast / casual
                         4 x slow / deliberate
                         4 x soft, almost muttered
At 3 m:                  4 x called across the room
                         4 x normal volume (hard case: quiet AND far)
At 0.3 m:                4 x normal
Off-axis / facing away:  4 x
In a carrier phrase:     6 x "Vektora, <a different command each time>"
Mid-sentence:            4 x "...so I said Vektora and it woke up"
```

#### 9.3b Hard negatives — the highest-value-per-clip data in the project

§6 already flags "vector" as a near-collision. **Broaden it. Target ~400–600
clips.** Same recording sessions, ~20 per speaker — two extra minutes per
person.

| Group | Words | Why |
|---|---|---|
| **The near-collisions** | vector, victor, Vectra, Vektor, spectra, sector | Differ only in the final vowel. Without these the model has no reason to learn that the final `/-a/` carries the distinction |
| **`/kt/` cluster words** | doctor, actor, factor, director, detector, projector, inspector, protector, character | The `/kt/` stop-burst is the landmark the CNN latches onto (§8.2). These words contain it and must not fire |
| **`/v/` onset words** | victory, vehicle, very, value, voice, video | Shares the quiet voiced-fricative onset |
| **Same 3-syllable rhythm** | camera, banana, tomorrow, Sephora, algebra, Ultra, opera, extra | Matches the temporal envelope without matching the phonemes — tests whether the model learned the *word* or the *shape* |
| **WARNING — partial keyword** | "vek", "tora", "vekto", "ektora" | **Frequently forgotten, and it will bite on device.** A sliding window sees the word half-in-frame on *every single detection*, twice. If partial words are not explicit negatives, the model fires early on "vek—" and the refractory period then blocks the real detection |
| **This-room-specific** | "vector", "detector", "training", "model", "dataset", "neural network" | This is an ML project. The demo room will be saturated with exactly these words, spoken near a live device (§6) |

#### 9.3c General speech — the bulk "other speech" class

| Source | Amount | Note |
|---|---|---|
| Speech Commands v2 | 105k clips, free | Already 16 kHz / 1 s / mono. Map the 35 words onto "other speech". Still the right backbone (§8.3) |
| Continuous conversation, chopped to 1 s windows | 10–20 h | Common Voice, LibriSpeech, plus **teammates talking for an hour** |
| **Indian-accented + regional-language speech** | 5–10 h | Same argument as the gate, more important here — the false-accept target is measured against whatever people actually say in the room |
| **Replayed TV/YouTube through the INMP441** | 5+ h unattended | The cheapest negative hours obtainable. Leave it recording overnight |
| **Hard-negative mining** | grows over time | Run each model version over the soak recording, collect every window scoring above ~0.3, add to "other speech", retrain. **The highest-yield loop in the entire project, and it costs no new recording** |

#### 9.3d Noise class, and room impulse responses

Same list as §9.2b — reuse it. Two additions specific to the KWS:

**Record our own room impulse responses.** `demo.md` §15.10 concluded
"simulating a recording chain you have not measured costs more than it buys,"
and every attempt to *guess* at far-field audio made the keyword weaker. That
conclusion is sound — but the operative word is **guess**. An RIR that is
*measured* is not a guess. Play a sine sweep from a speaker at 1/2/3 m and
various angles, record it on the INMP441, deconvolve, and we have the actual
room's response. Convolving clean positives with measured RIRs is legitimate
far-field augmentation. Free sets (OpenSLR SLR28, BUT ReverbDB, MIT IR Survey)
are a reasonable supplement, not a replacement.

**Record the device's own self-noise while running.** The ESP32's WiFi radio,
the switching regulator and the I2S clock all inject interference into the
INMP441 that is present *only* when the real firmware is running. Record 20
minutes with WiFi idle and 20 with WiFi transmitting. This noise is in every
on-device inference and in none of the training data right now.

---

### 9.4 The dataset nobody budgets for — and it decides the headline number

`CONTEXT.md` commits us to **< 1 false activation per hour**, measured, not
estimated. That number cannot be computed from any of the data above. It needs
one more thing:

> **8–24 hours of continuous ambient audio, recorded through the finished
> ESP32 + INMP441 in a realistic room, containing zero instances of
> "Vektora."**

Leave the board recording in the lab for a day. Talk normally, play music, run
a fan, hold meetings. Then replay it through the detector offline and count
firings. That is the false-accept rate, and it is the only way to have one.

It pays for itself three times: it is the evaluation set, it is the
hard-negative mining pool (§9.3c), and it is what to show judges — *"here is
eight hours of our lab, and it fired twice."*

Pair it with a **held-out device-in-the-loop verification set**: ~100 keyword
utterances and 1 hour of ambient, recorded through the exact final firmware,
never trained on, used only to confirm the C feature extraction matches Python
and the end-to-end system works.

---

### 9.5 Is 1 second right? — the measured answer

All 63 recordings were measured rather than guessed, since §6 flags this as
explicitly unverified ("Verify duration at the first recording session").

**A first pass at a `-30 dB` endpoint threshold suggested 39/63 words filled
the entire second. That was wrong** — it was catching room tone, not speech.
The clips have a **median dynamic range of only 33 dB** (quietest frames
≈ −37 dBFS, peaks at 0 dBFS and clipped), so a loose threshold measures the
room. At defensible thresholds:

| Endpoint threshold | Median word duration | p90 | Max |
|---|---|---|---|
| −15 dB below peak | 0.395 s | 0.759 s | 0.985 s |
| **−20 dB below peak** | **0.565 s** | **0.947 s** | 0.995 s |
| −25 dB below peak | 0.825 s | 0.995 s | 0.995 s |

Supporting measurements, all 63 files, at the −20 dB criterion:

- onset within 50 ms of window start: **24/63** (median onset 0.140 s)
- offset within 50 ms of window end: **14/63** (median offset 0.745 s)
- total slack (1.0 s − duration): median **0.435 s**, min **0.005 s**
- clips with **under 150 ms total slack: 17/63**

Absolute levels across the 63 files: quietest-5% frame median **−37.2 dBFS**
(range −120.0 .. −23.6); loudest frame median **−2.2 dBFS** (range −15.3 ..
−0.5). All 63 are exactly 1.000 s.

#### The verdict: keep the 1-second window. Change how we record.

**Keep 1 s / 98x40 as the model input.** The word is typically 0.40–0.65 s of
core energy — it fits, with room. The shape is locked, the DS-CNN is built for
it, Speech Commands is 1 s, and the on-device cost is real: 1 s of int16 audio
is 32 KB of the 520 KB SRAM, the feature buffer another ~4 KB, and doubling the
window roughly doubles both *plus* the convolution cost on a chip with no
vector instructions (§5). There is no accuracy case worth that price.

**But three things in the current recordings are actively hurting us, and none
of them are the window length:**

**(1) The words are jammed against the window edges.** At the −20 dB criterion,
**24 of 63 clips begin within 50 ms of t=0**, and **17 of 63 have under 150 ms
of total slack** across the whole second. `demo/train.py` applies time-shift
augmentation of **±0.15 s**. On those 17 clips that augmentation is *shifting
the word off the edge and truncating it* — training on partial words labelled
as complete ones. **This is a live bug in the current pipeline, not a future
concern.**

**(2) Isolated words teach the wrong cue.** Every positive is a word alone in a
second with silence either side. Real usage is `"Vektora, what's the weather"`
— the keyword is immediately followed by more speech, with no silence, and with
the final `/-a/` coarticulated into the next word. That final vowel is *the
only thing* distinguishing "vektora" from "vector" (§8.6). A model that has
only ever seen it followed by silence has learned it in a context that will
never occur on device.

**(3) The VAD cannot be tuned from these files.** Clipped, close-mic, high
noise floor. The stage-0 and stage-1 thresholds must come from INMP441
recordings.

#### What to do instead: record continuous sessions, segment in software

Stop recording isolated 1-second clips. Record **2–5 minute continuous
sessions** where the speaker works through a prompt list, then cut the windows
programmatically:

```
Record: one long WAV per speaker per condition, through the INMP441
Label:  a timestamp per keyword utterance (a keypress during recording,
        or forced alignment afterward - either is fine)
Cut:    for each labelled utterance, emit 5-10 one-second windows at
        random offsets that keep the whole word inside the frame
```

This gives five things at once:

1. **Natural coarticulation** — the keyword in real sentence context.
2. **Free position variety** — offset jitter at cut time replaces time-shift
   augmentation, and unlike the augmentation it *cannot truncate the word*,
   because we know where the word is.
3. **10x the positive clips** from the same speaking effort.
4. **Free hard negatives** — every window of the session that does *not*
   contain the keyword is a labelled negative in exactly the right acoustic
   condition, including the partial-word windows at the boundaries.
5. **Realistic silence and room tone** between utterances, for the gate.

One thing to verify once the new recordings exist: whether any speaker's slow
deliberate "Vektora" genuinely exceeds ~0.95 s. If a real speaker does, **do
not lengthen the window** — widen the speed-perturbation range slightly.
`demo.md` §15.4 records that 0.82–1.28x was too wide and 0.9–1.15x is the tuned
value, so move it carefully and re-measure the look-alikes each time.

---

### 9.6 The one thing to build before any of this

**Firmware that streams raw I2S audio from the INMP441 to a laptop over WiFi or
USB serial, and a Python script that writes it to WAV.**

Every high-value item above — the INMP441 noise floor, positives through the
real chain, the room IRs, the device self-noise, the 8-hour soak — is gated
behind being able to record through the actual hardware. It is maybe 150 lines
of ESP-IDF plus 40 lines of Python, it is squarely the user's half of the work
(§0), and it converts the entire data plan from blocked to merely tedious.

It also settles something that would otherwise be discovered late: whether the
I2S capture path, the DC offset handling and the gain staging produce audio
that resembles what the model was trained on at all.

**Two options were offered as the next action, and the user has not yet chosen:**

1. Write the capture firmware + receiver script.
2. Start with the recording-protocol tooling on the software side — the
   prompt-driven session recorder and the segmenter (§9.5).

---

### 9.7 Reproducing the §9.5 measurements

The endpointing script used above is not committed — it was scratch work. To
redo it: frame the audio at win=400 / hop=160, take per-frame RMS in dB,
subtract the clip's peak frame, and count frames above a threshold of −20 dB.
Do **not** use −30 dB on this dataset: the noise floor sits within 33 dB of the
peak, so −30 dB measures room tone and reports every word as ~0.995 s.

---

## 10. Session log — 2026-09-17 — the real prototype, from scratch (RESUME HERE)

Everything below was done in one session. Numbers are measured, not
estimated. The session ended with a full report to the user and **no training
running**. The user is reviewing the report, and some work waits on answers
from the user or the team (§10.12).

### 10.0 One-paragraph status

The team dataset (`C:\Users\User\documents\data`, 706 unique recordings) was
audited and a training pipeline was built in `prototype/training/`. Three runs
of a 3-class CNN (keyword / other / noise) followed.

- **Run 1** exposed an augmentation bug: exact-zero padding added only to
  speech clips.
- **Run 2** fixed that bug. It then exposed **chopped recordings** (exact-zero
  dropouts) in the data, which the model had learned to read as "Vektora".
- **Cleanup:** the user listened to the flagged files and confirmed 14 as
  bad. They are excluded through `exclusions.csv`.
- **Run 3 (clean dataset baseline)** scores best on paper: test recall 17/19,
  5/94 false accepts. **However, the gain cannot be credited to the cleanup.**
  Run 3's training set differs from Run 2's by only 2 files, so the swing is
  mostly run-to-run randomness.
- **Dataset not yet clean:**
  - 8 more chopped candidates await the user's decision.
  - Call-recorded positives are not identified.
  - The test set covers only one recording session.

**Next: user decisions (§10.12), then a one-time re-split, then multi-run
evaluation. No tuning yet.**

### 10.1 What the user decided this session

| Question | Answer |
|---|---|
| Where to build | New folder **`prototype/`**. Start **from scratch**; `demo/` is reference only |
| Dataset | `C:\Users\User\documents\data`: 4 folders (`positive`, `hard_negative`, `generic_negative`, `background_negative`), all 16 kHz / 1 s / WAV. It is its own git repo, `Thsky-21/Data` |
| Recording source | User: "all recorded on the INMP441, augmented later". **Not fully true:** 240 of the 340 generic originals are Google Speech Commands (`gsc_*`). Later the user added: "some Vektora samples were recorded over a call" |
| Speakers | Unknown. "Most samples are from about 6 people." User asked Claude to choose the best split |
| Pre-made augmented files | **(b)** Ignore them. Train on originals and augment online |
| Classes | **3**: keyword / other / noise. Hard and generic negatives merge into `other` |
| Background noise | **Both:** the user is recording INMP441 background noise *and* will add external noise. Neither is in the pipeline yet; the folders are empty |
| First build step | "Go with the build plan": manifest → split → features → train → evaluate → (int8 → firmware later) |
| After Run 2 | Cleanup first. Standing constraints on changes are in the top banner |

### 10.2 Dataset audit — `C:\Users\User\documents\data`

The user reported 2,446 samples. **Actual: 2,334 WAVs, and only 706 are
unique recordings.**

| Folder | WAVs | Originals | Pre-made augmented (ignored) |
|---|---|---|---|
| `positive` | 765 | **153** (`vektora_NNN.wav`) | 612 (`aug_0..3_vektora_NNN.wav`, 4 each) |
| `hard_negative` | **448** (user said 560) | **112** (`hard_neg_ (N).wav`, `hard_neg_(N).wav`) | 336 |
| `generic_negative` | 1020 | **340** = 100 team (`generic_neg_NNN`) + 240 Speech Commands (`gsc_<word>_<speakerid>_nohash_N`) | 680 (2 each) |
| `background_negative` | 101 | **101** (`noise_NNN`) | 0 |

- **All files are 16 kHz mono PCM_16.** 51 generic files are shorter than
  1 s; `fix_length` centre-pads them.
- **Each folder also holds a 2-byte extensionless file** (`positive_vektora`,
  `hard`, `generic`, `background`). These are GitHub folder placeholders and
  harmless.
- **Augmented files are named `aug_<k>_<original name>`**, so each copy
  traces back to its source. 27 augmented files are byte-identical to their
  original: the old augmentation did nothing to them.
- **The WAV headers carry no metadata.** All 706 originals have an identical
  minimal `fmt`+`data` header, so some tool re-wrote them all.
- **Upload history** (`git log` in the data repo): originals `004`–`081`
  existed by 2026-09-10 (vektora repo commit `3d22c8a`). `083`–`190` were
  uploaded to the data repo on 2026-09-16 in numeric batches. The history
  says nothing about how any file was recorded.
- **Overlap with the vektora repo:** the 63 root `vektora_*.wav` and 27
  `soft-negative/*.wav` files are byte-identical copies of files in the data
  repo.
- **Only 101 s of noise in total.** That is thin for the noise class.

### 10.3 What now exists in `prototype/`

```
prototype/
├── README.md               layout + contracts (features: training<->firmware; stream: firmware<->server)
├── firmware/               empty
├── server/                 empty
├── docs/measurements/      empty
├── data/                   GIT-IGNORED (.gitignore line added: prototype/data/)
│   ├── build/
│   │   ├── manifest.csv            current split (706 rows, incl. excluded)
│   │   ├── audit.csv               per-file measurements (audit_data.py)
│   │   ├── audit_categories.csv    A/B/C category + reasons per file
│   │   ├── train_log.txt           log of the latest run (= Run 3)
│   │   └── run1/ run2/ run3/       archived report.json, test_scores.csv,
│   │                               train_log.txt, manifest.csv (+ norm.json for 2, 3)
│   ├── long/noise/, long/other/    (not created yet) long INMP441 recordings go here
│   └── external_noise/             (not created yet) ESC-50 / DEMAND / MUSAN go here
└── training/
    ├── README.md           run order, data folders, noise download + policy
    ├── config.py           paths, CLASSES, split, training knobs, SOURCE_MIX
    ├── features.py         PARITY CONTRACT with the ESP32 C code
    ├── manifest.py         lists originals, splits by group, applies exclusions, leakage guards
    ├── exclusions.csv      human-decided exclusions (14 rows, all user-confirmed)
    ├── augment.py          online augmentation + NoiseBank + external-noise policy
    ├── model.py            3-class CNN
    ├── train.py            train, threshold on val, test once, reports
    ├── audit_data.py       corruption / recording-source measurements
    ├── compare_runs.py     side-by-side metrics of archived runs
    └── model/              latest model = Run 3 (model.keras is git-ignored via *.keras;
                            norm.json, report.json, test_scores.csv)
```

**Run order**, always from `prototype/training/` with the venv:

```bash
../../.venv/Scripts/python.exe audit_data.py      # optional: re-measure every original
../../.venv/Scripts/python.exe manifest.py        # rebuild split + apply exclusions.csv
../../.venv/Scripts/python.exe train.py           # 10-25 min on this CPU -> model/
# archive a finished run BEFORE the next one overwrites model/ (N = run number):
mkdir -p ../data/build/runN
cp model/report.json model/test_scores.csv model/norm.json ../data/build/manifest.csv ../data/build/train_log.txt ../data/build/runN/
../../.venv/Scripts/python.exe compare_runs.py run1 run2 run3   # add runN
```

**Nothing from this session is committed.** `git status` shows
`prototype/`, the modified `CLAUDE.md`, `demo.md` and `.gitignore`, and the
still-untracked `Human speech/`. Ask before committing. The data repo was not
modified.

### 10.4 Pipeline design — every decision and why

**Features (`features.py`): the parity contract.** The C code must reproduce
this exactly.

| Step | Value |
|---|---|
| Input | int16 16 kHz mono, 16000 samples; `x = s / 32768.0` |
| Framing | frame i = `x[i*160 : i*160+400]`, i = 0..97, no centre padding → 98 frames |
| Window | **periodic Hann**, `0.5 - 0.5*cos(2*pi*n/400)` |
| Spectrum | `abs(FFT_400)^2` → 201 bins |
| Mel | 40 bands, **HTK**, `norm=None`, 20–7600 Hz. The table is built by librosa and will be **exported to C as a table**, not re-derived |
| Log | `10*log10(max(mel, 1e-10))` |
| Normalise | **per-band** mean/std (40 + 40 numbers) in `model/norm.json`. Fit on 1,500 **augmented** training clips, so the statistics match the levels the model really sees |

- **No peak normalisation, unlike the demo.** The demo normalised whole
  recordings; a device listening to an endless stream has none. Level
  robustness comes from gain augmentation instead.
- **Verified:** the NumPy log-mel matches librosa with the same settings to
  within 5e-6 dB, and all 40 bands receive FFT bins.

**Split (`manifest.py`).**
- **Only originals are used;** `aug_*` files are dropped.
- **Groups stay together in one split:**
  - Team files: blocks of 10 consecutive take numbers (`TAKE_BLOCK=10`),
    since consecutive takes are likely the same person and sitting.
  - Speech Commands files: the real speaker ID from the filename.
- **70/15/15 by file count.** Groups are shuffled with seed 42, and the team
  and GSC families are each split separately.
- **The split is computed on ALL originals BEFORE exclusions.** Excluded
  files are then relabelled `excluded` (or `domain_shift`), so excluding a
  file never moves any other file. Runs 1–3 therefore share identical splits
  for every non-excluded file (verified: 0 files moved).
- **Guards:**
  1. No group appears in two of train/val/test.
  2. No identical audio (md5 of samples) appears in two splits.
- **Long recordings** (`data/long/{noise,other}/*.wav`, 16 kHz) are cut into
  1 s windows and split **by time within each file** (70/15/15, 1 s safety
  gaps).
  - Hop: 1 s for noise, 0.5 s for `other`.
  - `other` windows quieter than −45 dBFS are dropped.
- **Honest limit:** speakers are unlabelled, so the test measures *new takes*,
  not *new people*.

**Augmentation (`augment.py`).** Applied fresh each epoch, to training clips
only.
- **Speed perturbation:** p=0.8, rate 0.9–1.15 (tuned in demo §15.4).
- **Random EQ:** p=0.7; ±6 dB tilt, high-pass 50–250 Hz, low-pass 5.5–8 kHz.
- **`safe_shift`: THE BUG FIX from §9.5.** The word's span is measured with
  `active_span`: frames within 20 dB of the loudest. The word is then placed
  within ±0.15 s of its original spot, **clamped so it is never cut**.
  Verified: 20 shifts of every positive kept 100% of the word energy.
- **Gain:** −28 to +6 dB.
- **Overdrive (clipping):** p=0.3, +3 to +12 dB, applied to every class.
- **Background mix:** p=0.8 at 5–30 dB SNR, from the `NoiseBank` (training
  noise only).
- **SpecAugment:** a time mask of ≤10 frames and a band mask of ≤5 bands.
- **Padding uses room tone (Run 2 fix).** Where a shifted or shortened clip
  runs out, the gap is filled with **INMP441 room tone**: a random training
  noise snippet scaled to the clip's 10th-percentile 25 ms block level. Speech
  training uses the **raw, unpadded** audio (`r["raw"]`), so short clips are
  never zero-padded either.
- **Noise-class clips:** random circular shift. Since Run 2 they also get a
  **15% chance of an exact-zero dropout** (25–400 ms). It stays because the
  user required Run 2's configuration for Run 3. After background mixing,
  only about 1% of noise examples end up containing zeros.
- **External noise policy (not active yet; the folder is empty):**
  1. Mixed as background under **all** classes, 50/50 with INMP441 noise.
  2. At most `EXTERNAL_NOISE_CLASS_SHARE = 0.30` of noise-class examples,
     each with real INMP441 floor mixed underneath.
  3. Never used in val/test.

  Reason: otherwise the model could learn "different microphone = noise".
  Download instructions are in `training/README.md`: ESC-50
  (github.com/karolpiczak/ESC-50, CC BY-NC) and DEMAND
  (zenodo.org/record/1227121, 16 kHz files) first; MUSAN noise
  (openslr.org/17) optional. Best option of all: re-record external noise
  through the INMP441 by playing it from a speaker.

**Model (`model.py`).**
- **Note:** the user calls it "the DS-CNN", but it is **not**
  depthwise-separable. It is the demo's standard CNN with a 3-class head.
- **Architecture:** Conv16-BN-ReLU-MaxPool → Conv32-BN-ReLU-MaxPool →
  Conv64-BN-ReLU → GlobalAveragePooling → Dropout 0.3 → Dense 3 → Softmax.
- **Size:** 23,827 parameters, about 24 KB as int8.
- **TFLM compatibility:** every op has an int8 TFLite Micro kernel.
- **Why not a DS-CNN:** switching would be a latency-driven decision, to be
  made only after measuring on-device speed.

**Training (`train.py` + `config.py`).** These values are frozen by the user's
instruction.

| Setting | Value |
|---|---|
| Optimizer | Adam, lr 1e-3 |
| Loss | sparse categorical cross-entropy |
| Batch | 32 |
| Epoch | 4,000 freshly augmented clips (125 steps) |
| Epochs | up to 60 |
| Early stopping | patience 10 on val_loss, restore best weights |
| LR schedule | ReduceLROnPlateau: factor 0.5, patience 4, min 1e-5 |
| Seed | 42 |

**Sampling per epoch, by source** (`SOURCE_MIX`, renormalised over sources
that exist): positive 35%, hard 30%, generic 8%, gsc 7%, conversation 5%
(absent, so dropped), noise 15%. The effective mix is 37/32/8/7/16%. It is
deliberately weighted towards the look-alike boundary (demo §15.4).

**Threshold rule** (on VALIDATION, never test; frozen): take the highest
validation non-keyword score + 0.02, clamped to 0.5–0.95. If that loses more
than 20% of validation keywords, take the best balanced accuracy on a
0.50–0.95 grid instead. It fired on all three runs, and the result was
**0.50** every time.

**Outputs:**
- `report.json`: history, confusion matrix, per-source val/test stats.
- `test_scores.csv`: per clip — p_keyword, p_other, p_noise.

### 10.5 Run 1 — original augmentation

- **Training:** best epoch 29/39 (early stopped). Best val acc 0.861, val
  loss 0.3558.
- **Test:** 125 clips, before cleanup. 3-class accuracy 88.0%.
  - Keyword: 24/29 hit.
  - False accepts: 8/96 (noise 3, hard 2, everyday 3).
- **Warning sign:** `noise_066` scored **0.99** and `noise_067` **0.999** as
  Vektora.
- **Diagnosis:** the old `safe_shift` and speed code padded **speech clips
  only** with exact zeros. The model learned "digital silence = speech".
- **Fix:** room-tone fill, raw-length speech input, and noise-class zero
  dropout (§10.4).

### 10.6 Run 2 — zero-padding fix

- **Training:** best epoch 15/25 (early stopped). Best val acc 0.803, val
  loss 0.3996.
- **Test:** own test 84.8%.
  - Keyword: **14/29**.
  - False accepts: 3/96.
- **`noise_066`/`067` still scored 0.98/0.995**, so the Run 1 explanation was
  incomplete. Claude said so to the user.
- **Plotting showed the real cause:** these clips are **chopped**, with 5
  interior exact-zero gaps each (up to 55 ms). Each gap is a very dark
  vertical stripe on the spectrogram, resembling the /kt/ closure landmark.
- **Where the gaps sit:** 20 originals have exact-zero gaps of ≥10 ms. In
  train, 8 positives and 2 generic clips have them, and **no noise clips**. So
  "chopped" had become a keyword cue.
- **Also:** the whole block `vektora_020`–`029` failed together in test.
- **Interpretation:** exact zeros never occur acoustically. They are most
  likely dropped samples in the recording chain (I2S buffer overrun or the
  PC-side capture). **Check any capture firmware for dropouts before
  recording more data.**

### 10.7 Cleanup — user listened, 14 files excluded

The user confirmed, by ear:
- **Chopped:** `noise_066`, `noise_067`, `vektora_019`.
- **Loud, clipped/chopped, unusable:** `vektora_020`–`029`.
- **No useful keyword:** `vektora_030`.

All 14 are in `training/exclusions.csv`
(`folder,file,action,decided_by,date,reason`; action `exclude` or
`domain_shift`). Files are never deleted. Their original splits:
- **Train:** `019`, `030`.
- **Test:** `020`–`029`, `noise_066`, `noise_067`.

**Corruption audit** (`audit_data.py` → `audit.csv`; categories in
`audit_categories.csv`).

Per-file measurements:
- level: rms, peak
- `clip_frac` (share of samples with |x| ≥ 0.999)
- **interior** exact-zero runs ≥10 ms, with "abrupt" meaning a loud edge
  (|x| > 0.02 within 8 samples)
- noise floor: 10th-percentile 25 ms block, ignoring zero blocks
- HF ratio: 4–7.6 kHz vs 0.3–3.4 kHz
- effective cutoff: within 45 dB of the spectral peak

Category rules:
- **A (clearly corrupted):** ≥2 abrupt gaps, OR peak < 0.02, OR duration
  < 0.5 s. Clipping ≥10% was initially A; Claude moved it to B for noise,
  since loud noise can legitimately clip.
- **B (suspicious):** 1 abrupt gap or any soft gap, clipping ≥5%, level
  outlier (>3.5 robust σ from the folder median), or a speech floor above
  −28 dB.
- **C:** everything else.

The A/B categorisation was computed by an inline script in the session. Its
output is `data/build/audit_categories.csv`; the rules above are enough to
recompute it from `audit.csv`.

Counts: positive 9 A / 20 B / 124 C; hard 0/3/109; generic 2/12/326; noise
3 A (+3 clipping-only) / 2 B / 93.

**Calibration.**
- The rules agree with the user's ears on the chopped files:
  `019`/`021`/`022`/`066`/`067` all came out A.
- **They under-flag the 020–030 session.** Those files only reach B, through
  the high noise floor (−23 to −28 dB). A high speech-clip floor is therefore
  a real warning sign in this data.

**Category A candidates NOT yet excluded (waiting for the user to listen):**

| File | Split | Evidence |
|---|---|---|
| `vektora_032` | train | 3 abrupt gaps up to 143 ms, 6.9% clipped |
| `vektora_034` | train | 2 abrupt gaps, 52 ms |
| `vektora_108` | train | 2 abrupt gaps, 48 ms |
| `vektora_056` | val | 4 abrupt gaps, 99 ms, 7.4% clipped |
| `vektora_057` | val | 3 abrupt gaps, 99 ms, 6.7% clipped |
| `generic_neg_004` | train | 3 abrupt gaps, 85 ms |
| `generic_neg_005` | train | 2 abrupt gaps, 83 ms |
| `noise_068` | test | 3 abrupt gaps, 18 ms |

**B files worth a listen:**
- **Clipping only:** `noise_026` (10.2%), `noise_027` (14.2%), `noise_031`
  (21.2%), `vektora_055` (11.3%).
- **High floor, like the rejected session:** `vektora_031` (−26.0 dB),
  `vektora_033` (−24.0), `vektora_060` (−27.3).
- **One abrupt gap:** `vektora_037`, `039`, `104`, `111`, `hard_neg_ (89)`,
  `noise_065`.
- **Speech Commands clips with gaps:** `gsc_dog_74551073_nohash_1` (10 soft
  gaps, up to 148 ms; a test file and a persistent false accept),
  `gsc_bed_708a9569_nohash_0`, `gsc_go_25132942_nohash_0`,
  `gsc_wow_06f6c194_nohash_0`.

**Evidence the chopping cue is still learned:** the Run 3 model scores the
chopped positives `019` 0.937, `021` 0.988, `022` 0.930 and `056` 1.000,
while clean positives from the same sessions score far lower.

### 10.8 Call-recorded Vektora samples — NOT identified

- **No metadata** (§10.2).
- **No codec signature:**
  - Every positive session's long-term spectrum extends to the full 8 kHz;
    there is no 3.4 kHz or 7 kHz band limit.
  - Speech-frame HF ratios (5–7.6 kHz vs 0.3–3.4 kHz) are −14.6 to −19.2 dB
    across sessions, against −12.0 for hard negatives.
  - Noise-floor shapes match the INMP441 folders.
- **Nothing was moved to `domain_shift`.**

Recording sessions visible in the audio:

| Takes | Clipping | Noise floor | Notes |
|---|---|---|---|
| 004–017 | heavy (0.6–5.4%) | about −41 dB | |
| **019–039** | heavy (up to 7.9%) | **−23 to −34 dB** | most of the chopped files. **Consistent with a call, but not proof** |
| 040–055 | mixed | −32 to −40 dB | |
| 056–081 | heavy | −27 to −35 dB | |
| 083–111 | light | −36 to −43 dB | |
| 113–190 | none | −43 to −56 dB | clean, quiet session; also the whole test set |

**Action waiting on the user:** say which take ranges were recorded over a
call. Then add `domain_shift` rows to `exclusions.csv`, re-run
`manifest.py`, and score that set separately as a robustness set. That
scoring is not built yet: `compare_runs.py` only reads test rows.

### 10.9 Split after cleanup (identical to Runs 1/2 minus the exclusions)

| Source | Train | Val | Test | Excluded |
|---|---|---|---|---|
| positive (Vektora) | 96 | 26 | **19** | 12 |
| hard (look-alikes) | 72 | 20 | 20 | 0 |
| generic (team everyday words) | 59 | 21 | 20 | 0 |
| gsc (Speech Commands) | 168 | 36 | 36 | 0 |
| noise | 62 | 19 | 18 | 2 |
| **Total** | **457** | **122** | **113** | 14 |

Take blocks per split (from `manifest.csv`):

| Source | Train | Val | Test |
|---|---|---|---|
| positive | 001–017, 031–039, 060–068, 071–077, 081–088, 091–099, 100–108, 111–149, 160–169, 190 | 040–048, 050–058, 170–179 | **150–159, 180–189 only**, i.e. only the clean late session |
| hard | 002–029, 040–059, 070–079, 100–113 | 060–069, 080–089 | 030–039, 090–099 |
| generic | 001–019, 030–039, 050–059, 080–099 | 020–029, 060–069, 100 | 040–049, 070–079 |
| noise | 010–019, 030–059, 070–089, 100–101 | 001–009, 020–029 | 060–069 (minus 066/067), 090–099 |

Speech Commands speakers: 152 train / 33 val / 36 test. Both leakage guards
passed.

### 10.10 Run 3 — clean dataset baseline (same config as Run 2)

- **Training:** best epoch **40/50**, early stopping triggered.
  - Best val acc **0.869**, best val loss **0.2957**.
  - Train acc 0.813 at the best epoch, 0.836 at the last. Last val acc 0.844.
- **Threshold:** 0.50. The zero-false-accept cutoff of 0.682 would have lost
  more than 20% of validation keywords.
- **Clean test (113 clips):** 3-class accuracy **93.8%**.
  - Keyword: TP 17 / FN 2 → recall **89.5%**, precision **77.3%**, F1
    **0.829**.
  - False accepts: **5/94 = 5.3% per clip** (noise 0, look-alike 1,
    everyday 4). This is per clip, not per hour.
- **Confusion matrix** (rows true, columns predicted: keyword/other/noise):
  keyword `[17 2 0]`, other `[5 71 0]`, noise `[0 0 18]`.
- **Misses:** `vektora_182` 0.273, `vektora_152` 0.433.
- **False accepts:** `generic_neg_045` 0.510, `generic_neg_072` 0.565,
  `hard_neg_ (92)` 0.688, `gsc_cat_1a4259c3_nohash_0` 0.735,
  `gsc_dog_74551073_nohash_1` 0.916.
- **Validation recall is only 16/26.** Scores for the val positives (Run 3
  model, P(Vektora)):

  | Takes | Scores |
  |---|---|
  | 040–044 | 0.116, 0.286, 0.717, 0.540, 0.352 |
  | 045–048 | 0.617, 0.794, 0.727, 0.822 |
  | 050–058 | 0.623, 0.126, 0.185, 0.452, 0.241, 0.960, 1.000, 0.174, 0.868 |
  | 170–179 | 0.272, 0.460, 0.815, 0.712, 0.691, 0.557, 0.706, 0.532 |

  Misses cluster in the **unclipped** early takes: 040, 041, 044, 051–054.
- **Scores on the excluded positives** (for reference only): 019 0.937,
  020 0.094, 021 0.988, 022 0.930, 023 0.483, 024 0.716, 025 0.682,
  026 0.347, 027 0.383, 028 0.292, 029 0.496, 030 0.016.

### 10.11 Comparison of all runs (`compare_runs.py`)

All three runs are scored on the same **clean** 113-clip test set, each with
its own threshold (0.50 for all).

| | Run 1: original augmentation | Run 2: zero-padding fix | Run 3: corrupted data removed |
|---|---|---|---|
| Vektora recall | 16/19 (84.2%) | 12/19 (63.2%) | **17/19 (89.5%)** |
| Precision / F1 | 72.7% / 0.780 | 92.3% / 0.750 | 77.3% / 0.829 |
| False accepts | 6/94 | **1/94** | 5/94 |
| Noise FA | 1 | 0 | 0 |
| Look-alike FA | 2 | 0 | 1 |
| Everyday FA | 3 | 1 | 4 |
| Best val acc | 0.861 | 0.803 | 0.869 |
| Test 3-class acc | 90.3% | 92.0% | 93.8% |
| Val recall at threshold | 17/26 | 6/26 | 16/26 |
| Own test set | 24/29 hit, 8/96 FA | 14/29 hit, 3/96 FA | = clean |

Recurring offenders across runs:
- **Misses:** `vektora_182` and `vektora_152` (all runs), `vektora_181`
  (Runs 1–2).
- **False accepts:** `gsc_dog_74551073_nohash_1` (all runs; 0.58, 0.90,
  0.92), `hard_neg_ (92)` (Runs 1 and 3), `generic_neg_072` and
  `generic_neg_045` (Runs 1 and 3).

### 10.12 Verdict, and exactly where to resume

**Verdict given to the user: the dataset is NOT yet clean enough for
model/threshold tuning.**
1. **Run 3's improvement can't be credited to the cleanup.** Its training set
   differs from Run 2's by only 2 files (`vektora_019`, `030`); the other 12
   exclusions were test files. Yet Run 2 → 3 moved recall 63% → 89% and false
   accepts 1 → 5. That is **run-to-run variance**. With 19 test positives,
   one clip moves recall by 5.3 points.
2. **Chopped files remain** (8 A candidates, 3 of them positives in train),
   and the model still rewards chopping.
3. **Call-recorded positives are unidentified.**
4. **The test positives come from one session only** (the clean late one).
   Validation, which covers older sessions, shows much weaker recall.

**Waiting on the user or team (ask about these first):**
1. **Listen to the 8 A candidates** (§10.7), plus B files `vektora_031`,
   `033`, `055`, `060` and `noise_026`/`027`/`031`. Decide exclude / keep for
   each.
2. **Which positive take ranges were recorded over a call?** Those become
   `domain_shift`.
3. **Which firmware or script recorded the data, and does it drop samples?**
   If it does, fix it before any new recording.
4. **INMP441 background recordings** (the user is recording now) go into
   `prototype/data/long/noise/` as 16 kHz WAV, 2–10 min per condition, one
   file per condition. External noise goes into
   `prototype/data/external_noise/`. Neither exists yet.
5. **Should Run 2's 15% noise-class zero dropout stay?** It is still active.
   The user said "no dropout augmentation yet", but also asked for Run 2's
   exact configuration. Claude kept Run 2's configuration and flagged the
   conflict.

**Then Claude does, in order (each needs the user's go-ahead):**
1. Add the user's decisions to `exclusions.csv` and re-run `manifest.py`.
2. **Re-split once** so val and test contain every recording session, for
   example by stratifying take blocks by session. This deliberately ends
   comparability with Runs 1–3; run1–3 are already archived.
3. **Multi-run evaluation:** 5 seeds or group-folds per configuration,
   reporting mean ± spread. Single runs can't measure the changes being
   considered. Each full training takes 10–25 min here, so 5 runs take one to
   two hours; run them in the background.
4. Only then tune: LR schedule (Run 2 decayed early), augmentation strength,
   dropout in all classes (the Run 2 proposal), and the threshold and
   post-processing.
5. **Later build steps:**
   - int8 quantization, checking that int8 scores match float.
   - Export the mel table, window and norm constants to C.
   - C feature extraction, with an element-by-element parity test against
     `features.py`.
   - TFLite Micro on the ESP32 (§5: plain ESP32, expect 2–3× slower than S3
     figures).
   - The INMP441 capture firmware (§9.6), which is also needed to check the
     dropout question.

### 10.13 Session gotchas (so they aren't rediscovered)

- **Background training:** run `train.py` in the background and wait for the
  completion notification. The Bash tool here blocks `sleep N; ...` chains.
- **A run takes 25–50 epochs** at 12–43 s each (10–25 min). Epoch time varies
  with machine load.
- **`train.py` overwrites `training/model/`.** Archive to
  `data/build/runN/` before the next run; `compare_runs.py` reads from there.
- **`test_scores.csv` stores basenames plus source.** `compare_runs.py`
  matches on (basename, source) against the current manifest's test rows.
- **Excluded rows stay in `manifest.csv`** with `split=excluded` and a
  `note` giving the original split and the reason. `train.py` loads their
  audio but never uses them.
- **Speed perturbation can stretch a natural near-zero run past 10 ms**
  (about 1 in 5,000 augmentations). This is source-derived, not padding.
- **`demo.md` claimed the early positives were "close-mic studio"
  recordings.** In practice they're heavily clipped, with a high noise floor;
  treat that description as unverified.
- **Bash heredocs containing backticks and quotes failed** in this shell
  (a quoting error). Use the Write/Edit tools for large text.
- **Claude project memory** (`prototype-dataset.md`) holds a short summary of
  the dataset facts above.

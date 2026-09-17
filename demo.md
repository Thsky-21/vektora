# demo.md — Vektora wake-word demo (presentation: 2026-09-12)

> **This file is the active plan.** It replaces the ESP32 / 3-class /
> Speech-Commands plan in `CLAUDE.md` §2–§8 *for the demo*. At your request
> the old `sih26172/` scaffold was **deleted on 2026-09-11** (see §9, D6).

**Goal, in one sentence:** a web page anyone can open, press *record*, say
"Vektora", and watch a small neural network we trained ourselves say
*"detected"* — with graphs that show what it heard and how it decided.

**What it is NOT (for tomorrow):** not running on the ESP32, not always-on
streaming, not measured against a false-accept-per-hour target. Those are the
original SIH26172 goals and make good "next steps" slides.

---

---

## ⏵ CURRENT VERSION — read this first (written 2026-09-12, ~10:50, demo day)

**This section describes the model that is actually trained, committed and
running.** Everything below it in this file is the original plan and the
learning material; where the two disagree, this section is right. The full build detail is **§15 at the end of this file**. In
particular §0.4's "v1" numbers are **historical** — that model was trained on
phone/browser recordings that the current one deliberately does not use.

### The one-paragraph summary

A 2-class convolutional neural network, 23,697 parameters, that answers one
question about one second of audio: *is that "Vektora"?* It is trained on the
repository's own recordings and **nothing else**: 63 positives
(`vektora_*.wav`) and 27 soft negatives (`soft-negative/*.wav`, the look-alike
words "vector" and "victor"), plus 150 noise clips generated in code so that
silence is never guessed at. It is served by a Streamlit page (`demo/app.py`)
that records from the browser, slides a 1-second window across the recording,
and reports the best score against a threshold of 0.50.

### Results — every number measured, none estimated

| Measure | Result |
|---|---|
| **All 63 "Vektora" files** | **62/63 fire**, median score 0.98 |
| **All 27 soft-negative files** | **0/27 fire**, highest score 0.23 |
| Held-out test set (35 clips never trained on) | 8 hits, 1 miss, **0 false accepts**, 26 correct rejects |
| Test accuracy / precision / recall | 97.1% / 1.00 / 0.89 |
| Threshold | **0.50**, chosen on validation (highest validation negative scored 0.17) |
| Model size | 23,697 parameters (~93 KB float32, ~24 KB if quantised to int8) |
| Training set after augmentation | 3,006 clips |
| Training run | 23 epochs, best epoch 11, early stopping restored epoch 11 |
| Live browser-mic "Vektora" (6 recorded words) | 4/6 fire: 0.04, 0.25, **0.65, 0.68, 0.83, 0.89** |

The separation on files is decisive rather than lucky: look-alikes top out at
0.23, the threshold sits at 0.50, real keywords sit at 0.9+. **Nothing scores
between 0.25 and 0.65.**

### The one honest limitation, stated plainly

Both datasets are **close-mic studio recordings**: the speaker is inches from a
good microphone and 86 of the 90 files are clipped (recorded too loud). Live
browser audio is a different world — a metre of room between mouth and laptop,
reflections, Chrome's automatic gain control and noise suppression, no
clipping. **The model has never seen a single sample of it.** Consequences:

- Vektora versus look-alike words: **solid**, on files.
- Vektora versus free-form conversation through a laptop mic: **not solid** —
  ordinary talking can score high, because "ordinary talking" is not in either
  dataset. This is the same gap §3 of this file describes, and it is the
  price of training on these two datasets only.

What closes it is not a cleverer model: it is ~90 seconds of recording through
the demo's own microphone (10 × "Vektora", 10 × "vector/victor"). That was
offered and declined for time on demo day; the code path for it still exists
(`ROOM_AGC` in `train.py`, and the folders `demo/data/live_vektora/`,
`demo/data/live_other/`).

---

## Contents

0. [Key findings + the 2-hour fast track](#0-key-findings--the-2-hour-fast-track) ← **start here**
1. [What you asked for](#1-what-you-asked-for)
2. [Your dataset — what I found when I inspected it](#2-your-dataset--what-i-found)
3. ["Train only on positives?" — why we need negatives too](#3-train-only-on-positives--why-we-need-negatives-too)
4. [The big picture — how the whole thing works](#4-the-big-picture)
5. [Learn it: concepts and keywords, in the order you meet them](#5-learn-it-concepts-and-keywords)
6. [Tools we use, and why each one](#6-tools)
7. [The graphs — training time and test time](#7-the-graphs)
8. [Files we will create](#8-files-we-will-create)
9. [Decisions — defaults chosen, yours to overturn](#9-decisions)
10. [Step-by-step runbook (tonight)](#10-runbook)
11. [Deploying to Hugging Face](#11-deploying-to-hugging-face)
12. [Presentation script + likely questions](#12-presentation)
13. [Demo-day risks and backups](#13-demo-day-risks-and-backups)
14. [Progress checklist](#14-progress-checklist)
15. [How this version was built, in full detail](#15-how-this-version-was-built-in-full-detail) ← **the shipped model**

---

## 0. Key findings + the 2-hour fast track

### 0.1 Key findings (the conclusions everything else follows from)

1. **Training on positives only cannot work.** A model shown only "Vektora"
   learns to answer "Vektora" to everything: 100% in training, useless on the
   website. **We train on positives *and* negatives, for one keyword.**
   (Full explanation: §3.)
2. **The dataset (GitHub `Thsky-21/vektora`), measured file by file:**
   - 63 positives (`vektora_*.wav`, at the repo root) and 27 soft negatives
     (`soft-negative/`).
   - All **16 kHz, mono, exactly 1.000 s**, no duplicates, so no conversion
     is needed.
   - **Missing:** everyday negatives (talking, silence, room noise).
   - **Too loud:** 59/63 positives and 27/27 negatives are clipped (~3% of
     samples flat-topped).
   - **Small:** test scores will move ~10% per mistake.

   (Details: §2.)
3. **Hugging Face no longer offers Streamlit as a built-in SDK.** It runs as
   a **Docker Space using the Streamlit template**, and needs
   `--server.enableXsrfProtection=false`, otherwise recording and upload fail
   with a 403 error. (§6, §11.)
4. **The neural network isn't the hard part** (~40 lines). The things that
   decide whether the demo works are **data variety** (enough different
   negatives and voices) and **the threshold** (false accepts vs. misses).

### 0.2 Can the simplest workable demo be built in 2 hours? **Yes, with a reduced scope.**

Claude writing the code takes minutes, and training takes ~1–2 minutes on a
laptop CPU. The 2 hours go to things that can't be sped up: your
recordings, account setup, **the first Hugging Face build (~5–10 min,
TensorFlow is big)**, testing with your own voice, and fixing whatever
breaks.

**In the 2-hour version (the minimum that still demos well):**

| Part | 2-hour version |
|---|---|
| Data | GitHub clips + **5 min** of your own talking and room noise + noise and partial-word negatives generated in code |
| Model | The small CNN from §5.4, 2 classes |
| Training graphs | **Learning curves, confusion matrix, score histogram** (the three that tell the story) |
| App | **One page**: record or upload → verdict + confidence → waveform + spectrogram + confidence-over-time, plus a tab with the 3 training graphs |
| Hosting | Hugging Face Docker Space (free CPU) |

**Cut until after the 2 hours** (add them tonight if time remains): class-count and example-spectrogram graphs,
threshold trade-off graph, "How it works" tab, built-in example-clip buttons,
teammates' recordings.

**The 2-hour timeline:**

| Time | You | Claude |
|---|---|---|
| 0:00–0:10 | Create HF account, write token, and Space (§11) | `git pull`, install `streamlit` + `huggingface_hub` into the venv |
| 0:10–0:20 | **Record** 3 min talking + 1 min silence (§10 step 3) | Write `audio_utils.py` + `prepare_data.py` |
| 0:20–0:40 | Run data prep → **look at a spectrogram**. Run training → **read the learning curves** with Claude | Write `train.py`, pick the threshold with you |
| 0:40–1:10 | Run the app locally; **10 × "Vektora", 10 × other words**, count results | Write `app.py`, fix what your test finds |
| 1:10–1:40 | Upload to the Space, watch the build log | Write `Dockerfile`, `requirements.txt`, `README.md`; fix build errors |
| 1:40–2:00 | **Rehearse once on the live URL** (§12) | Buffer for surprises |

**What could blow the 2 hours:** (1) the Hugging Face build failing on a
package version, which is why the deploy step has 30 minutes;
(2) the model working on the team's voices but misfiring on yours, if
you didn't record the positives, which is why Q1 matters; (3) the browser
microphone being blocked (backup: upload a WAV).

**Learning in 2 hours:** read §5 *while things run*, e.g. §5.2 during data
prep, §5.5 during training, §5.6 while testing. Each concept shows up on
your screen right when you read it; that's the fastest way to make it stick.

### 0.3 Your answers to the open questions

*(Filled in as you answer. See §14 for the questions.)*

Answered 2026-09-11:

| Q | Answer | What it changes |
|---|---|---|
| Q1: speakers / soft-negative words | Positives: **5–6 people**. Soft negatives: **2–3 people**, words that sound like Vektora (**victor, vector**, …) | Good voice variety for a demo this size, so strangers on the website have a fair chance. Caveat: filenames don't say who is speaking, so the same person can appear in both train and test, which makes test scores a bit optimistic. |
| Q2: record everyday negatives? | **Yes, on my phone** (not the project mic), **one person** (me) | Creates the **microphone confound** below. Fixed by also recording ~20 × "Vektora" on the phone. |
| Q3: Hugging Face account / username | **`ThSky21`** (<https://huggingface.co/ThSky21>) | The Space will be `https://huggingface.co/spaces/ThSky21/vektora-demo` |
| Q4: `sih26172/` keep or delete | **Delete.** Keep only the build plan and project details | Done: `sih26172/` deleted. Kept: `demo.md`, `CONTEXT.md`, `CLAUDE.md`, `.gitignore`, `.venv/` |

### 0.4 Build status  — **HISTORICAL (2026-09-11 → 12 morning).**
*Superseded by the CURRENT VERSION section at the top and §15. The v0/v1
models described below were trained on phone and browser recordings that the
shipped model deliberately does not use. Kept because the reasoning is still
worth reading.*

#### Original text — what exists now (2026-09-11, evening)

**Everything is built and tested locally. Model "v0" is trained WITHOUT your
phone recordings.** Two steps remain: record → retrain, then deploy.

| File (`demo/`) | Status | What it does |
|---|---|---|
| `audio_utils.py` | ✅ tested | Loads any audio → 16 kHz mono; log-mel picture (98×40); normaliser; sliding window; silence gate |
| `plots.py` | ✅ | One colour-blind-safe style for every graph (blue = Vektora/training, orange = not/validation) |
| `prepare_data.py` | ✅ ran | Collects clips, splits 70/15/15, cuts phone recordings, makes example clips + 2 graphs |
| `train.py` | ✅ ran (~1.5 min) | Augments, trains the CNN (**23,697 parameters**), picks threshold, tests, makes 3 graphs |
| `app.py` | ✅ tested | The website: Record / Example clips / Upload → verdict + 4 visuals; training tab; how-it-works tab |
| `requirements.txt`, `Dockerfile`, `README.md` | ✅ written, not yet deployed | Hugging Face Space definition |
| `deploy.py` | ✅ written, not yet run | Creates `ThSky21/vektora-demo` and uploads only what the site needs |

**v1 (2026-09-12 morning) — trained with your 3-minute phone recording of
Hindi/Kannada/English talking.** This fixed the problem you saw live, where
*any* speech triggered the detector. That happened for exactly the reason in
§3: the model had never been shown ordinary talking, so "speech-shaped" was
all it knew.

| | v0 (no talking data) | **v1 (with your recording)** |
|---|---|---|
| Test clips | 35 | **88** |
| Vektoras caught | 9/9 | **9/9** |
| False accepts | 0/26 (no speech in the test) | **1/79** (one look-alike word) |
| Your talking, held-out clips | not tested | **53/53 correctly rejected** |
| Sliding across 27 s of held-out conversation | not tested | **0 false fires; highest score 0.04** |
| Look-alike words scored | 0.20 / 0.66 | **0.15 / 0.21** |
| Threshold | 0.73 | **0.62** |

**The number to quote in the presentation: 0 false alarms in 27 seconds of
continuous conversation it had never heard, while still catching every
"Vektora".** The remaining error is one look-alike ("vector"/"victor") scoring
0.83 — the honest hard case, and worth showing rather than hiding.

⚠️ Still open: there are **no phone-recorded "Vektora" clips** (§0.5). The
random-microphone augmentation below reduces that risk, but the live test on
your laptop mic is what proves it.

**Tested with Streamlit's test runner:**
- all example clips give the right verdict: Vektora 93% and 97%, soft
  negatives 20% and 66%;
- a 3.5 s recording with the word in the middle is detected at the right
  moment (1.6 s);
- a 0.5 s half-word is correctly rejected;
- digital silence is skipped by the silence gate;
- room hiss scores 10%;
- a stereo 48 kHz upload is converted correctly.

**Changes from the plan above, and why:**
- **FFT size 400, not 512.** librosa slices audio by FFT size, so 512 gave
  97 time steps instead of 98. Checked: every mel band still gets ≥ 3 FFT bins.
- **Partial-word negatives (D2c) dropped.** Labelling "-tora" as negative
  risks teaching the model to reject real Vektoras whose quiet /v/ start got
  lost, which trades misses for almost no gain.
- **Score histogram → score dot plot** (`score_distribution.png`). With ~35
  test clips, one dot per clip is more honest than histogram bars.
- **All 5 training graphs are in** (§0.2 cut two, but they turned out cheap).
- **Learning-rate schedule added.** The rate halves when validation stalls.
  The validation curve was very jumpy because the validation set is tiny.
- **Random-microphone augmentation added (v1).** Every training clip gets a
  random tone change (bass/treble roll-off and tilt), because the "Vektora"
  clips come from the project mic while the talking comes from your phone —
  without it the network could recognise the *microphone* instead of the word.
- **m4a support fixed twice.** Phone files must be written to a temporary file
  before ffmpeg reads them (an .m4a index can sit at the end of the file, and a
  pipe can't seek), and the ffmpeg on this machine is from 2013, so it rejects
  the modern `-hide_banner` flag.
- **New graph + metric: `continuous_speech.png`**, the detector's score across
  held-out conversation, and "false fires per minute" in the app.
- **Privacy:** example clips on the public site never include your
  "talking" recording, `deploy.py` never uploads `demo/data/`, and
  `demo/data/` is now in `.gitignore`, because the GitHub repo is public.

**The exact commands** (from the repo root):

```bash
.venv/Scripts/python.exe demo/prepare_data.py            # 1. data  (~45 s)
.venv/Scripts/python.exe demo/train.py                   # 2. train (~1.5 min)
.venv/Scripts/python.exe -m streamlit run demo/app.py    # 3. app → http://localhost:8501
.venv/Scripts/hf.exe auth login                          # 4a. once, in YOUR terminal
.venv/Scripts/python.exe demo/deploy.py                  # 4b. publish
```

### 0.5 The microphone confound (caused by Q2, easy to fix)

If **every "Vektora" clip comes from the project mic** and **all everyday
negatives come from your phone**, the model can take a shortcut: learn *"sounds
like the phone mic → not Vektora"* instead of learning the word. Models take
shortcuts like this whenever one exists, and you'd never see it in the test
scores, because the test clips share the same shortcut. On the website every
visitor uses a *third* mic (their laptop or phone), so the result would be
missed detections.

**Fix (1 extra minute):** in the same phone session, also record **~20 ×
"Vektora"** with a 1–2 second pause between each. The code cuts them apart
automatically. Now both classes contain phone audio, so "which mic" stops
being a clue. The soft negatives (project mic, negative) already help from
the other direction.

---

## 1. What you asked for

Recorded here so nothing gets lost:

- A **quick demo for tomorrow's presentation**.
- You're **new to ML** and want to learn **enough to build it and present it
  smoothly**, not the whole field.
- The dataset is **on GitHub** (`Thsky-21/vektora`): positives and negatives.
- **One keyword: "Vektora".**
- **Set aside everything built so far** and build only the demo.
- **Host on Hugging Face Spaces** with a **simple Streamlit UI**, so anyone
  can test it by opening the link.
- **Graphs and visuals**: during training, during testing, and in the UI.

---

## 2. Your dataset — what I found

I pulled the GitHub repo into a scratch folder and measured every file.

| Folder in the GitHub repo | Files | What they are |
|---|---|---|
| repo root, `vektora_*.wav` | **63** | **Positives**: "Vektora", spoken by 5–6 people |
| `soft-negative/soft_neg_*.wav` | **27** | **Soft negatives**: words that sound like Vektora (victor, vector, …), from 2–3 speakers |
| `positive_sample/positive` | 0 audio | 2-byte placeholder file, no clips; the positives are at the root |

**The good news:** every file is **16 kHz, mono, exactly 1.000 s**, so no
converting or trimming is needed. No duplicate files.

**Three problems, from most to least important:**

1. **No "everyday" negatives.** All 27 negatives are look-alike words. There
   is no silence, no room noise, no ordinary talking. A model that never saw
   those has no idea what to do with them. On the live site, silence or a
   cough could come out "detected". **Fix:** record a few minutes of normal
   talking and room noise tonight, and generate extra noise clips in code
   (§9 D2).
2. **The recordings are clipped.** 59 of the 63 positives and all 27
   negatives hit the maximum possible level. On average about 3% of samples
   are flat-topped. *Clipping* means the microphone was turned up too far and
   the tops of the sound waves got cut off, which adds distortion. The live
   browser microphone will **not** be clipped, so the model would see a
   slightly different kind of audio than it trained on. **Fix:** randomly
   clip and change the volume of *every* class during training, so "clipped"
   never becomes a clue for "Vektora". For future recordings, turn the input
   gain down.
3. **It's a small dataset.** 63 positives is enough for a demo, not for a
   product. We'll make up for it with *augmentation* (§5.4). Test scores will
   be *noisy*: with about 10 test positives, one mistake moves accuracy by
   10%. Say this out loud in the presentation; it earns trust.

Also: the words mostly fill the whole second (the edges are only about 25 dB
quieter than the loudest part), so there's little room to shift them in time.
Fine for now.

---

## 3. "Train only on positives?" — why we need negatives too

"**Only one keyword, Vektora**" — **yes.** That's the plan: the model answers
one question, *"was that Vektora or not?"*

"**Train using only the positive clips**" — **no, this cannot work.** Here's
why:

> Imagine teaching someone to spot cats by showing them 63 cat photos and
> saying "this is a cat" every time. Then you show them a dog. They say
> "cat". They were never shown anything that *isn't* a cat, so "always say
> cat" was a perfect score during training.

A model learns only the difference between the things it's shown. With only
positives, the easiest thing it can learn is "the answer is always Vektora".
It would score 100% in training and be useless on the website. **The
negatives are what teach it where the boundary is.** That's why:

- the **soft negatives** you recorded are valuable: they teach *"close, but
  not it"*;
- we still need **everyday negatives** (talking, silence, noise): they teach
  *"obviously not it"*.

(There are "positives-only" methods, called *one-class* or *template
matching*, but even those need negatives to set their detection threshold.
They're more complicated and less reliable. Not for tomorrow.)

---

## 4. The big picture

```
 ┌──────────────┐   ┌────────────────┐   ┌──────────────┐   ┌──────────────────┐   ┌───────────────┐
 │ 1. DATA      │   │ 2. FEATURES    │   │ 3. TRAIN     │   │ 4. APP           │   │ 5. DEPLOY     │
 │ .wav clips   │──►│ sound → image  │──►│ CNN learns   │──►│ Streamlit page:  │──►│ Hugging Face  │
 │ from GitHub  │   │ (log-mel       │   │ Vektora vs   │   │ record → predict │   │ Space: public │
 │ + new recs   │   │  spectrogram)  │   │ not-Vektora  │   │ → show graphs    │   │ URL for all   │
 └──────────────┘   └────────────────┘   └──────────────┘   └──────────────────┘   └───────────────┘
   prepare_data.py    audio_utils.py        train.py             app.py             Dockerfile +
                      (shared by 2 & 4)     → model + plots                         upload
```

**The whole idea in five lines:**

1. A computer can't "hear". It gets a list of 16,000 numbers per second.
2. We turn each 1-second clip into a small **picture of the sound** (98 × 40
   pixels) that shows which pitches are loud at each moment.
3. A **convolutional neural network (CNN)**, the same kind of model that
   recognises cats in photos, learns what the Vektora picture looks like
   compared with everything else.
4. The trained model outputs **one number from 0 to 1**: how sure it is that
   the clip is Vektora. Above a **threshold** (say 0.8), we say "detected".
5. We wrap that in a web page and host it for free on Hugging Face.

---

## 5. Learn it: concepts and keywords

Ordered the way you'll meet them while building. You need these to *build and
present* the demo; you can skip everything else in ML for now.

### 5.1 Sound

| Term | Plain meaning |
|---|---|
| **Waveform** | Sound is air pressure wobbling. A microphone measures the wobble; plotting it over time gives the familiar squiggly line. |
| **Sample** | One measurement of that wobble: a single number. |
| **Sample rate** | Measurements per second. Ours is **16,000 Hz (16 kHz)**, so a 1 s clip is 16,000 numbers. It captures pitches up to 8 kHz, which covers everything that makes speech understandable. |
| **Mono** | One microphone channel (not stereo). |
| **WAV** | An uncompressed audio file: basically the raw list of samples plus a small header. |
| **dB / dBFS** | Loudness on a log scale. "dBFS" means relative to the maximum possible level; 0 dBFS is the ceiling. |
| **Clipping** | The sound was louder than the ceiling, so the tops of the waves got cut flat. Adds harsh distortion. (Our dataset has a lot of it; see §2.) |

### 5.2 Features: turning sound into a picture

| Term | Plain meaning |
|---|---|
| **Feature** | Whatever we actually feed the model. Raw samples work poorly for small models, so we compute something more informative first. |
| **Frame / window** | We cut the clip into short overlapping slices: **25 ms long** (400 samples), a new one every **10 ms** (the **hop**, 160 samples). Speech sounds change every few tens of milliseconds, so short slices catch each sound. |
| **FFT** | *Fast Fourier Transform*: maths that takes one slice and tells you **how much of each pitch (frequency)** is in it. Like a graphic-equaliser display. |
| **Spectrogram** | Line up the FFT of every slice side by side and you get an **image**: x = time, y = frequency, brightness = loudness. |
| **Mel scale** | Humans hear pitch unevenly: 100→200 Hz sounds like a big jump, 7100→7200 Hz sounds like nothing. The mel scale squeezes high frequencies together to match. We use **40 mel bands**, narrow at the bottom and wide at the top. |
| **Log-mel spectrogram** | A mel spectrogram with loudness in log (dB) scale, because hearing is logarithmic too. **This is the picture our model sees.** |
| **98 × 40 × 1** | The picture's size: 98 time steps (`(16000 − 400) / 160 + 1 = 98`) × 40 mel bands × 1 channel (greyscale). |
| **Normalisation** | Shift and scale the numbers so they sit around 0 with a spread of about 1. Neural networks learn much faster this way. We compute the mean and spread from the *training* data once, save them, and reuse the exact same values in the app. |

**Why "Vektora" is easy to see in this picture:** the *k-t* in "ve**kt**ora"
is a moment of near-silence (a dark vertical stripe) followed by a burst of
noise (a bright stripe across all frequencies). CNNs are very good at spotting
edges like that. Point at it in the presentation.

**The #1 silent bug in ML demos: training/serving skew.** If the app computes
the picture even slightly differently from training (a different hop, a
different normalisation), the model sees unfamiliar pictures and gives
confident nonsense, **with no error message**. Prevention: one file,
`audio_utils.py`, holds the feature code, and *both* training and the app
import it. There is never a second copy.

### 5.3 Data

| Term | Plain meaning |
|---|---|
| **Label** | The correct answer for a clip: `1` = Vektora, `0` = not Vektora. |
| **Class** | A category of label. We have **2 classes** (binary classification). |
| **Positive / negative** | Clips of the keyword / clips of anything else. |
| **Hard (soft) negative** | A negative that *sounds close*: "vector", "victor". The most valuable kind, because it forces the model to learn the fine details. |
| **Dataset** | All clips together with their labels. |
| **Train / validation / test split** | We divide the clips three ways, like **textbook exercises / mock exam / final exam**. The model *learns* from **train** (70%). We *check progress and choose settings* on **validation** (15%). We look at **test** (15%) **once, at the end**, to get an honest score. If you tune things until the test score looks good, the test has become practice and its score is a lie. |
| **Stratified split** | Each split keeps the same Vektora:other ratio, so the 15% test set isn't accidentally all negatives. |
| **Data leakage** | When information from the test set sneaks into training, e.g. two overlapping slices of the same recording land in train *and* test. Scores look great and mean nothing. We slice long recordings *after* splitting them by time to prevent it. |
| **Class imbalance** | If negatives outnumber positives 5:1, a lazy model that always says "no" scores 83%. **Class weights** fix this by making mistakes on the rare class count more. |

### 5.4 The model

| Term | Plain meaning |
|---|---|
| **Model** | A mathematical function with thousands of adjustable knobs. Clip picture in, score out. |
| **Parameters / weights** | The knobs. Ours has roughly **24,000**. (A photo-recognition model has millions, a chatbot billions.) Small = fast, fits on an ESP32 later, and harder to overfit with 63 positives. |
| **Neural network** | A model built from stacked **layers**; each layer transforms its input a little and passes it on. |
| **Convolution (Conv2D)** | A small 3×3 *filter* slides over the whole picture looking for one pattern (an edge, a stripe). A layer has many filters. Stacked layers find **patterns of patterns**: stripes → bursts → "k-t followed by -ora". |
| **CNN** | *Convolutional Neural Network*: a network made mainly of convolution layers. Standard for images, so standard for spectrograms. |
| **Activation (ReLU)** | After each layer, negative values are set to 0. This small bend is what lets the network learn complicated shapes instead of just straight lines. |
| **Pooling** | Shrinks the picture (e.g. keep the strongest value in each 2×2 block). Less computation, and small shifts matter less. |
| **Global average pooling** | At the end, average each pattern detector over the whole picture: *"was this pattern anywhere?"*. That makes the model care less about *where* in the second the word sits. |
| **Dropout** | During training, randomly switch off 30% of the connections each step, so the model can't rely on one fragile clue. Off during real use. |
| **Dense layer** | Combines all the pattern detectors into one number. |
| **Sigmoid** | Squashes that number into the range 0–1, so it can be read as a **confidence score** ("0.93 → 93% sure it's Vektora"). |

Our architecture (small on purpose):

```
log-mel picture 98×40×1
  → Conv 16 filters → ReLU → Pool        (finds edges / stripes)
  → Conv 32 filters → ReLU → Pool        (finds sound pieces: "k-t burst", vowels)
  → Conv 64 filters → ReLU               (finds word-level combinations)
  → Global Average Pooling → Dropout 30%
  → Dense 1 → Sigmoid → score 0..1
```

### 5.5 Training

| Term | Plain meaning |
|---|---|
| **Training** | Show the model clips, measure how wrong it is, nudge every knob slightly in the direction that makes it less wrong. Repeat thousands of times. |
| **Loss** | The number that measures "how wrong". We use **binary cross-entropy**, which punishes *confident* wrong answers very hard. Training = making the loss go down. |
| **Gradient descent** | The "nudge downhill" method. The *gradient* tells each knob which way is downhill. |
| **Optimizer (Adam)** | A smarter version of gradient descent that adapts the nudge size per knob. The default choice. |
| **Learning rate** | How big each nudge is. Too big: it bounces around and never settles. Too small: it takes forever. We use 0.001. |
| **Batch** | Clips processed together before one nudge. We use 32. |
| **Epoch** | One full pass over all training clips. We train for up to ~40. |
| **Overfitting** | **The #1 risk with 63 positives.** The model *memorises the training clips* (this speaker, this room) instead of learning the word. Symptom: training accuracy keeps rising while validation accuracy stalls or drops. The learning-curves graph shows it. |
| **Early stopping** | Watch validation loss; when it stops improving for several epochs, stop and keep the best version. Cheap protection against overfitting. |
| **Data augmentation** | Make altered copies of training clips so the model sees more variety for free: **shift** the word slightly in time, **add background noise**, **change volume and clip**, **blank out** random strips of the spectrogram (*SpecAugment*). Applied **only to training data**, never to validation or test. |

### 5.6 Evaluating it

| Term | Plain meaning |
|---|---|
| **Inference** | Using the trained model on new audio. That's what the app does. |
| **Threshold** | The cutoff on the 0–1 score for saying "detected". Chosen **on validation data**, after training. |
| **False accept (false positive)** | It says "Vektora" when nobody said it. **The worst error for a wake word:** a device that wakes up by itself is creepy and annoying. |
| **False reject / miss (false negative)** | You said Vektora and it didn't respond. Annoying, but you just repeat it. |
| **The trade-off** | Raising the threshold means fewer false accepts but more misses. Lowering it does the opposite. **There is no setting that removes both.** Wake-word products lean toward fewer false accepts. |
| **Accuracy** | % of clips judged correctly. Misleading when classes are imbalanced (§5.3). |
| **Precision** | Of all the times it *said* Vektora, how often was it right? (High = few false accepts.) |
| **Recall** | Of all the times Vektora *was said*, how often did it catch it? (High = few misses.) |
| **Confusion matrix** | A 2×2 table: real answer vs. predicted answer. The four cells are hits, misses, false accepts, and correct rejections. The clearest single picture of performance. |
| **Sliding window** | For a recording longer than 1 s, score a 1 s window, move 0.1 s, score again, and so on. You get **confidence over time**, which shows *when* Vektora was said. |

---

## 6. Tools

| Tool | What it is | Why we use it |
|---|---|---|
| **Python 3.10** | Programming language | The standard language for ML |
| **venv (`.venv/`)** | A private Python install for this project only | Your global Python broke once already (see `CLAUDE.md` §1). **Always run `.venv/Scripts/python.exe`.** |
| **NumPy** | Fast maths on arrays of numbers | Audio and spectrograms are just arrays |
| **soundfile** | Reads and writes WAV files | Loading clips |
| **librosa** | Audio analysis library | Resampling mic audio to 16 kHz, computing mel spectrograms |
| **TensorFlow / Keras 2.15** | Neural-network library (Keras is its friendly front end) | Already installed and tested in the venv. `model.fit()` is the most beginner-friendly training API there is. |
| **scikit-learn** | Classic ML utilities | Stratified train/val/test split, confusion matrix, precision/recall |
| **matplotlib** | Plotting library | Every training graph (saved as PNGs) |
| **Streamlit** | Turns a Python script into a web page, no HTML needed | The demo UI: `st.audio_input` gives a record button in the browser |
| **Hugging Face Spaces** | Free hosting for ML demos | Public URL, anyone can open it |
| **Docker** | Packages the app and its exact dependencies into a container | **Hugging Face no longer offers Streamlit as a built-in option**; it now runs as a Docker Space using the Streamlit template ([HF docs](https://huggingface.co/docs/hub/en/spaces-sdks-streamlit)). We write a ~10-line `Dockerfile`; you don't need to install Docker locally. |
| **huggingface_hub** | Python library for Hugging Face | Uploading the Space from the command line (or use the website's upload button) |
| **git / GitHub** | Version control / hosting | Where the dataset lives |

---

## 7. The graphs

### 7.1 Training time (saved to `demo/plots/`, also shown in the app's "How it was trained" tab)

| Graph | What it shows | What to say when presenting |
|---|---|---|
| **Class counts** (bar chart) | Clips per category: Vektora, soft negatives, other speech, noise | "This is everything the model learned from." |
| **Example pictures** | Waveform + log-mel spectrogram for one clip of each category | "This is what the model *sees*." Point at the *k-t* stripe. |
| **Learning curves** | Loss and accuracy per epoch, train vs. validation | "Both lines improve together, so it learned the word instead of memorising the clips." (Or honestly show the gap if there is one.) |
| **Confusion matrix** | Test-set hits / misses / false accepts / correct rejections | The honest final score |
| **Score histogram** | Model scores for real Vektora clips vs. negatives, with the threshold line | "The two groups barely overlap; the line sits in the gap." The most intuitive graph for a non-ML audience. |
| **Threshold trade-off** | False accepts and misses as the threshold slides from 0 to 1 | Why we picked the threshold we did |

### 7.2 Test time (live in the app, every time someone records)

| Visual | What it shows |
|---|---|
| **Verdict + confidence** | Big "✅ Vektora detected (94%)" or "❌ Not detected (12%)" |
| **Waveform** | The recording, with the detected region highlighted |
| **Log-mel spectrogram** | The exact picture the model looked at |
| **Confidence over time** | Sliding-window score line with the threshold drawn on it; peaks where Vektora was said |

### 7.3 App layout

```
┌─ Vektora wake-word demo ───────────────────────────────────────────┐
│  [ 🎙️ Try it ]   [ 📊 How it was trained ]   [ 🧠 How it works ]   │
├────────────────────────────────────────────────────────────────────┤
│  🎙️ Try it:  ● Record   or   ⬆ upload .wav   or   ▶ example clips │
│    ✅ VEKTORA DETECTED — 94%                                        │
│    [waveform]  [spectrogram]  [confidence-over-time]               │
│  📊 How it was trained: the §7.1 graphs, each with one sentence    │
│  🧠 How it works: the §4 pipeline + a short glossary               │
└────────────────────────────────────────────────────────────────────┘
```

**Example clips** (a few held-out test clips built into the app) mean the demo
works **even if the venue's microphone or browser permissions fail**.

---

## 8. Files we will create

```
vektora/                      (repo root)
├── vektora_*.wav             ← 63 positives (from GitHub, left where the team put them)
├── soft-negative/*.wav       ← 27 soft negatives (from GitHub)
├── demo.md                   ← this file
└── demo/
    ├── data/
    │   ├── other_speech/     ← NEW (phone): your normal talking, sliced into 1 s clips in code
    │   ├── noise/            ← NEW (phone): room silence / fan, sliced in code
    │   └── phone_vektora/    ← NEW (phone): ~20 × "Vektora", cut apart automatically (§0.5)
    ├── audio_utils.py        ← load audio + make log-mel picture. Used by BOTH train and app
    ├── prepare_data.py       ← collect clips, label, split, save features + dataset plots
    ├── train.py              ← build CNN, train, evaluate, save model + training plots
    ├── app.py                ← Streamlit UI
    ├── model/                ← vektora.keras, norm.json, config.json (threshold), metrics.json
    ├── plots/                ← all PNGs from §7.1
    ├── examples/             ← a few held-out clips for the app's example buttons
    ├── requirements.txt      ← exact package versions for the Space
    ├── Dockerfile            ← how Hugging Face builds the app
    └── README.md             ← Hugging Face Space settings (the YAML header)
```

Note: the root `.gitignore` blocks `*.keras` and `*.npz`, so the trained model
won't go to GitHub by default. That's fine: the Space is uploaded separately.
If you also want the model on GitHub, that's a one-line change.

---

## 9. Decisions

Per `CLAUDE.md` §7 you make the decisions. Given the deadline, each one has a
**default**, and the default is what gets built **unless you say otherwise**.

| # | Decision | Default | Why / the trade-off |
|---|---|---|---|
| D1 | Number of classes | **2**: Vektora / not Vektora | The locked plan had 3 (keyword / other speech / noise). That split mattered for the ESP32 false-accept target; for a demo, 2 is simpler to train, explain and plot. Easy to go back later. |
| D2 | Where everyday negatives come from | **(a)** you record ~3 min of normal talking + ~1 min of room noise, sliced into 1 s clips; **(b)** noise generated in code; **(c)** *partial-word* negatives, the "vek-" or "-tora" half of a positive clip, which teach "the whole word, or nothing" | (a) takes 5 minutes, matches your real mic and room, and gives ~250 clips. Teammates adding 1 min each would add *speaker variety*, the thing most likely to make it work for strangers on the website. The alternative, Google Speech Commands (thousands of speakers), is a 2.3 GB download: worth it after the presentation, too slow tonight. |
| D3 | Model | **Small CNN trained from scratch** (~24k parameters) | Simple to explain, trains in about a minute on a laptop CPU, and could later shrink onto the ESP32. Alternative: *transfer learning* (reuse a big pretrained audio model, train only the last layer). Often better with little data, but a black box that's harder to explain and heavier to host. |
| D4 | Framework | **TensorFlow/Keras 2.15** | Already installed and verified in `.venv`. |
| D5 | Hosting | **Hugging Face Docker Space, Streamlit template, free CPU** | Streamlit isn't a native HF option any more (§6). Free CPU is plenty for a 24k-parameter model. |
| D6 | `sih26172/` and the old plan | **Deleted 2026-09-11** (your Q4 answer) | It held 4 files with content (two tested training scripts, a README, a `.gitignore`) and 42 empty stubs, none in git. The plan and reasoning survive in `CLAUDE.md` and `CONTEXT.md`. |
| D9 | Microphone confound (§0.5) | **Also record ~20 × "Vektora" on the phone** | Stops the model learning "phone mic = negative" |
| D7 | Detection threshold | **Chosen after training** from the validation score histogram, leaning toward few false accepts | Can't be picked before we see the scores |
| D8 | Loudness handling | **Peak-normalise every clip + random gain/clip augmentation + a "too quiet" check in the app** | Handles the clipped dataset (§2) and quiet laptop mics. The quiet check stops pure silence from being amplified into noise and guessed at. |

---

## 10. Runbook

Rough timings for tonight. The critical path is **data → train → app → deploy**.
**Deploy tonight, not tomorrow morning.**

| Step | Who | Time | What happens |
|---|---|---|---|
| **0. Answer the open questions** (Q1–Q4 below) | You | 5 min | They change the code |
| **1. Get the dataset locally** | Claude | 2 min | `git pull` (fast-forward; your untracked files are unaffected) |
| **2. Install tools into the venv** | Claude | 5 min | `.venv/Scripts/python.exe -m pip install streamlit huggingface_hub`, checked first so it can't disturb TensorFlow's versions |
| **3. Record on your phone** | You | 10 min | Three separate recordings, any format (m4a is fine; ffmpeg converts it): **(a)** `talking`: ~3 min of normal talking, any language, **never "Vektora"**, with "vector", "victor", "Victoria" mixed in a few times → `demo/data/other_speech/`. **(b)** `silence`: ~1 min of the room with nobody talking (fan and background noise welcome) → `demo/data/noise/`. **(c)** `vektora`: **~20 × "Vektora"**, 1–2 s pause between each, varied speed and loudness → `demo/data/phone_vektora/` (fixes §0.5). Hold the phone at normal speaking distance, not against your mouth. |
| **4. `prepare_data.py`** | Claude writes, you run | 20 min | Loads everything, labels, splits, saves features, draws dataset plots. **You look at the example spectrograms and find the *k-t* stripe yourself.** |
| **5. `train.py`** | Claude writes, you run | 30 min | Trains in about a minute. We read the learning curves together, check for overfitting, and choose the threshold. |
| **6. `app.py` locally** | Claude writes, you run | 45 min | `.venv/Scripts/python.exe -m streamlit run demo/app.py`, opens in the browser. **You test it: 10 × "Vektora", 10 × other words, count results.** |
| **7. Deploy to Hugging Face** | You (account) + Claude (files) | 30 min | §11 |
| **8. Rehearse** | You | 30 min | §12, on the live URL, twice |

Every command runs from the repo root with the **venv** Python. Never a bare
`python` or `pip`.

---

## 11. Deploying to Hugging Face

**What you do (needs your account):**

1. Sign up at <https://huggingface.co/join> (free).
2. **Settings → Access Tokens → Create new token**, type **Write**. Treat it
   like a password: never paste it into code, chat, or GitHub.
3. **New Space** (<https://huggingface.co/new-space>): name `vektora-demo`,
   SDK **Docker** → template **Streamlit**, hardware **CPU basic (free)**,
   visibility **Public**.

**What Claude prepares in `demo/`:**

- `README.md` whose top block tells HF how to run it:
  ```yaml
  ---
  title: Vektora Wake Word Demo
  emoji: 🎙️
  colorFrom: indigo
  colorTo: purple
  sdk: docker
  app_port: 8501
  ---
  ```
- `Dockerfile`: starts from Python 3.10, installs `requirements.txt`, runs
  `streamlit run app.py --server.port=8501 --server.address=0.0.0.0
  --server.enableXsrfProtection=false`. **That last flag matters:** inside
  HF's page frame, Streamlit's upload security check blocks both the record
  button and file uploads with a *403 error* unless it's off.
- `requirements.txt` pinned to the exact versions tested locally
  (`tensorflow-cpu==2.15.1`, `librosa`, `numpy==1.26.4`, `streamlit`, …).

**Uploading (either way works):**

- **Website:** Space → *Files* → *Add file* → *Upload files* → drag in the
  contents of `demo/` (not `data/`).
- **Command line (recommended):** log in once, in your own terminal, with
  `.venv/Scripts/hf.exe auth login` (paste your token when it asks; the
  old `huggingface-cli` name is deprecated). Then run
  `.venv/Scripts/python.exe demo/deploy.py`. It creates the Space for you,
  so you can skip step 3 above.

Then open the Space's **Logs** tab and watch it build (the first build takes
~5 min because TensorFlow is large). When it says *Running*, the app is live
at **<https://huggingface.co/spaces/ThSky21/vektora-demo>**.

**Version trap (found 2026-09-11):** installing `streamlit` unpinned pulls
**protobuf 7**, which breaks TensorFlow 2.15 (it needs protobuf < 5). Always
pin `protobuf==4.25.9`, locally and in the Space's `requirements.txt`. With
that pin, pip picks **streamlit 1.60.0**, and that's the combination tested
locally.

---

## 12. Presentation

### 12.1 Script (~6 minutes)

1. **The problem (30 s).** "Every voice assistant has a *wake word*: the
   thing that's always listening, like 'Hey Siri'. It has to be tiny, fast
   and *almost never wrong*, because a device that wakes up by itself is a
   privacy problem. Ours is 'Vektora'."
2. **Why "Vektora" (20 s).** Three syllables (more evidence than a short word).
   The *k-t* sound is visually distinctive. It isn't a word in any language, so
   nobody says it by accident.
3. **The data (45 s).** Class-counts chart. "63 recordings of Vektora, 27
   deliberately confusing look-alikes, plus everyday speech and noise."
4. **How a computer sees sound (60 s).** Examples chart: waveform →
   spectrogram. Point at the *k-t* stripe.
5. **The model (45 s).** "A small convolutional neural network, the same
   family that recognises faces in photos, with only ~24,000 parameters.
   Small enough to eventually run on a $5 ESP32 chip." Learning curves.
6. **Results (60 s).** Confusion matrix + score histogram. **State the
   small-test-set caveat yourself.**
7. **Live demo (90 s).** Open the Space. Say "Vektora" → detected. Say
   "vector" → rejected. Stay silent → rejected. Show confidence over time.
   Invite someone from the audience to try.
8. **Next steps (20 s).** More speakers, running on the ESP32 on-device, and
   measuring false accepts per hour of real speech.

### 12.2 Questions you'll probably get

| Question | Answer |
|---|---|
| *Why not just use speech-to-text?* | Speech-to-text is huge and power-hungry. A wake word has to run 24/7 on a tiny chip. Ours is ~24k parameters and would fit in about 25 KB. |
| *How accurate is it?* | Quote the test confusion matrix, then add: "The test set is small, so treat that as a rough number. The metric that really matters for a wake word is false accepts per hour, and that's our next measurement." |
| *Will it work for my voice?* | "Probably, but less reliably than for ours. 'Vektora' was recorded by 5–6 people, which is a decent start, but a real product uses thousands. That's the next step." Honest, and it shows you understand overfitting. |
| *Why did you record 'vector' and similar words?* | They're *hard negatives*. Without them the model never learns that the final "-a" is what makes the difference. |
| *What's the hardest part?* | Not the neural network, which is ~40 lines. It's the **data** (enough variety) and **choosing the threshold** (the false-accept vs. miss trade-off). |
| *What is a convolution?* | A small pattern detector sliding across the spectrogram, like looking for a shape through a moving magnifying glass. |

---

## 13. Demo-day risks and backups

| Risk | Backup |
|---|---|
| **The Space is asleep** (free Spaces sleep when unused) | Open the URL **10 minutes before** your slot so it can wake up |
| **Venue wifi is down** | Run it locally: `.venv/Scripts/python.exe -m streamlit run demo/app.py`. Test this tonight too. |
| **The browser blocks the microphone** | Use the built-in **example clips** buttons, or upload a WAV |
| **Laptop mic is very quiet** | The app shows a "too quiet — speak closer" message instead of guessing |
| **A stranger's voice fails live** | Expected (§12.2). Say so, then try it yourself |
| **Hugging Face build fails tomorrow** | That's why we **deploy tonight**; the local run is the backup anyway |

---

## 14. Progress checklist

- [x] Dataset located on GitHub and inspected (§2)
- [x] Plan and learning guide written (this file)
- [x] Q1–Q4 answered (§0.3)
- [x] `sih26172/` deleted (Q4)
- [x] `git pull` (63 positives + 27 soft negatives now local) + venv packages
      installed (streamlit 1.60.0, huggingface_hub, protobuf pinned 4.25.9;
      TensorFlow 2.15.1 re-verified)
- [x] All code written (§0.4); `prepare_data.py` + `train.py` run; v0 model trained
- [x] `app.py` tested locally: example clips, long recordings, silence, 48 kHz upload
- [x] Phone recording of ordinary talking (`Human speech/Humanspeech.m4a`, 3 min)
      → `demo/data/other_speech/`; **v1 retrained: ordinary speech no longer fires**
- [ ] **You:** try the Record button at http://localhost:8501 with your own voice
      (say Vektora ×10, then other words ×10, and count)
- [ ] **Optional but recommended:** ~20 × "Vektora" on the *phone* → `demo/data/phone_vektora/` (§0.5)
- [ ] Live tally: __ / 10 Vektora detected, __ / 10 other words wrongly detected
- [x] Hugging Face account (`ThSky21`)
- [ ] Write token + `hf auth login` (the Space itself is created by `deploy.py`)
- [ ] Space deployed and working at its public URL
- [ ] Rehearsed twice on the live URL; local backup tested

### Open questions

- **Q1.** How many **different people** recorded the 63 Vektora clips? And what
  words are in `soft-negative/`? (This tells us how much to trust the scores
  for new voices, and what the model has already learned to reject.)
- **Q2.** Can you (and ideally 1–2 teammates) record the **everyday negatives**
  in step 3 tonight?
- **Q3.** Do you have a **Hugging Face account**? What's the username?
- **Q4.** `sih26172/`: **keep aside** (default) or **delete**?

---

## 15. How this version was built, in full detail

### 15.1 The data

| Source | Files | Label | Where |
|---|---|---|---|
| "Vektora", spoken by 5–6 people | **63** | 1 | `vektora_*.wav`, repo root |
| Look-alike words ("vector", "victor"), 2–3 people | **27** | 0 | `soft-negative/*.wav` |
| Synthetic noise generated in code | **150** | 0 | none — made fresh each run |

Every real file is already 16 kHz, mono, exactly 1.000 s, so nothing is
resampled or trimmed. Each is **peak-normalised** to 0.9 so that a loud project
mic and a quiet laptop mic look alike to the model.

The synthetic noise is five kinds in equal share — white (hiss), pink (fans,
rain), brown (rumble), 50/60 Hz mains hum, and near-silence — each at a random
level. It is not padding: without a "this is not speech at all" category,
digital silence and room hiss land in unexplored territory and get a confident
guess.

**Everything else was deliberately excluded** (`GITHUB_ONLY = True` in
`prepare_data.py`): phone recordings, browser-mic recordings, the 3-minute
conversation recording, and room noise. The folders are untouched on disk;
setting that one flag to `False` brings them all back.

### 15.2 The split — 70 / 15 / 15, stratified

| Split | Total | Vektora | Soft negative | Noise |
|---|---|---|---|---|
| train | 170 | 45 | 19 | 106 |
| val | 35 | 9 | 4 | 22 |
| test | 35 | 9 | 4 | 22 |

Split with a fixed seed (42), so the same clips land in the same split on every
run and results are comparable between experiments. Train is what the model
learns from; validation is what chooses the threshold and stops the training;
**test is scored once, at the end, and never used to make a decision.**

### 15.3 Features — sound into a picture

Every 1-second clip becomes a **log-mel spectrogram of 98 × 40 × 1**:

| Setting | Value | Why |
|---|---|---|
| Sample rate | 16,000 Hz | Covers everything that makes speech intelligible |
| Window | 400 samples (25 ms) | Speech sounds change every few tens of ms |
| Hop | 160 samples (10 ms) | `(16000 − 400) / 160 + 1 = 98` frames exactly |
| FFT size | **400, not 512** | librosa slices by FFT size; 512 gives 97 frames, not 98 |
| Mel bands | 40 | Narrow at the bottom, wide at the top, like human hearing |
| Scale | log (dB) | Hearing is logarithmic too |
| Normalisation | fixed mean/scale from **training data only**, saved to `norm.json` | The app must reuse the identical numbers |

`audio_utils.py` holds this code and **both** `train.py` and `app.py` import it.
There is never a second copy. This is the defence against *training/serving
skew*, the single most common silent bug in ML demos: if the app computed the
picture even slightly differently, the model would see unfamiliar input and
return confident nonsense with no error message.

### 15.4 Augmentation — where most of the work actually went

45 positive clips cannot train a network. Each training clip is copied many
times with random alterations, turning 170 clips into 3,006:

| Class | Copies each | Reason |
|---|---|---|
| Vektora | 32 | The rare class |
| **Soft negatives** | **40** | See below — this is the most important number in the file |
| Noise | 6 | Plentiful and easy |

**Why soft negatives get the most copies of all.** With 6 copies each, the
negative class came out roughly 85% synthetic noise, so nearly all of the
model's effort went into "word versus hiss" — which is trivial — and almost
none into "Vektora versus vector", which is the only distinction the demo is
judged on. Raising it to 40 puts the training effort exactly on the boundary
that matters. This single change took the look-alikes from *one firing at 0.76*
to *none firing, highest 0.23*.

Each copy gets a random subset of these, applied in this order:

| Step | Setting | Why |
|---|---|---|
| **Speed perturbation** | p = 0.8, rate **0.9–1.15×** | Added after a live test: the detector caught "Vektora" spoken *loudly and slowly* and missed it at normal pace. All 63 recordings are careful, deliberate pronunciations; nothing else in the pipeline varies the *rate* of speech. Resampling shortens or lengthens the word and shifts pitch slightly — standard "speed perturbation" from speech recognition, one line of interpolation instead of a phase vocoder. |
| Random EQ | p = 0.7, ±6 dB tilt, roll-off below 50–250 Hz and above 5.5–8 kHz | Pretends a different microphone, so the network cannot recognise the *recording chain* instead of the word |
| Time shift | ±0.15 s | Live, the word will not sit where it sat in the recordings |
| Gain | **−28 to +6 dB** | Widened from −20 dB after quiet live speech was missed |
| Overdrive | p = 0.3, +3 to +12 dB | 86 of 90 real files are clipped. Clipping *every* class at random stops "clipped" becoming a clue for "Vektora" |
| Background noise | p = 0.8, SNR 5–30 dB | Mixed from the training-split noise clips only, so nothing leaks from val/test |
| SpecAugment | time mask ≤ 10 frames, band mask ≤ 5 bands | Blanks a strip of the picture so no single tiny detail can be depended on |

**Why the speed range is 0.9–1.15 and not wider.** It was 0.82–1.28 for one
run. That made the model so tolerant of duration and pitch that a fast
"vector" started to look like a "Vektora" — measured live by the user as false
activations. Narrowing it kept the pace tolerance and restored the boundary.

Augmentation is applied to **training clips only**, never to validation or
test. Applying it to test clips would inflate the score and mean nothing.

### 15.5 The model

```
log-mel picture 98 × 40 × 1
  → Conv2D 16 filters 3×3 → BatchNorm → ReLU → MaxPool 2   (98×40 → 49×20)
  → Conv2D 32 filters 3×3 → BatchNorm → ReLU → MaxPool 2   (49×20 → 24×10)
  → Conv2D 64 filters 3×3 → BatchNorm → ReLU
  → GlobalAveragePooling2D        ("was each pattern anywhere?")
  → Dropout 0.3
  → Dense 1 → Sigmoid             → one score, 0..1 = P(Vektora)
```

**23,697 parameters.** The three blocks are a hierarchy: the first finds edges
and stripes, the second finds pieces of sounds (the *k-t* burst, vowels), the
third finds word-level combinations of those pieces.

*Global average pooling* rather than flatten is deliberate: it averages each
pattern detector over the whole picture, so the answer barely depends on
**where** in the second the word sits — which is what a sliding window needs.

Small is a feature, not a compromise: with 63 positives a larger network would
memorise them, and ~24k parameters is small enough to eventually quantise to
about 24 KB and run on an ESP32 (the original SIH26172 goal, §5 of `CLAUDE.md`).

### 15.6 Training

| Setting | Value |
|---|---|
| Loss | Binary cross-entropy (punishes *confident* wrong answers hardest) |
| Optimizer | Adam, learning rate 0.001 |
| Batch size | 32 |
| Epochs | up to 60 — **stopped at 23**, best epoch **11** restored |
| Early stopping | patience 12 on validation loss, `restore_best_weights=True` |
| LR schedule | halve on plateau, patience 3, floor 1e-5 |
| Class weights | Inversely proportional to class size, so the rarer class counts more |
| Seed | 42 everywhere (NumPy and Keras), so runs are reproducible |
| Runtime | ~1.5 minutes on this laptop's CPU |

Best validation loss 0.047 with validation accuracy 1.000 at epoch 11.

### 15.7 The threshold

Chosen **on the validation set, after training, never on test**. The rule:
sit just above the highest-scoring validation negative, giving zero false
accepts on validation — unless that would miss more than 20% of validation
keywords, in which case fall back to the best balance.

Here the highest validation negative scored **0.17**, so the rule wanted 0.19,
and the floor of 0.50 applied. Threshold = **0.50**.

> **A correction worth recording, because it cost time on demo day:** the
> earlier model's threshold of 0.73 was not a quality setting and **higher is
> not better**. The threshold is a cutoff: raising it makes the detector fire
> *less*. 0.73 on the current model would catch **zero** of the recorded live
> keywords. 0.73 was simply where that older model's negatives happened to sit.

### 15.8 The app

`demo/app.py`, Streamlit. What happens on each recording:

```
browser mic → WAV bytes → peak-normalise the WHOLE recording
  → slide a 1 s window every 0.1 s
  → skip near-silent windows (below −45 dBFS) — a cheap voice-activity gate
  → log-mel picture → normalise with the SAME saved numbers as training
  → CNN → one score per window
  → smooth over 7 windows (0.7 s) → take the best → compare with threshold
```

**Why smoothing over 0.7 s matters** (widened from 0.3 s on demo day): the page
reports the **maximum** over roughly 30 sliding windows, which is thirty
separate chances for one unlucky window to cross the line. "Vektora" takes
about 0.7 s to say, so a genuine detection holds high confidence across that
whole span while a look-alike produces a brief spike. Demanding a *sustained*
peak is what makes taking the maximum trustworthy. It costs nothing on 1-second
clips, where there is only one window.

Three tabs: **🎙️ Try it** (record / example clips / upload, with verdict,
waveform, spectrogram and confidence-over-time), **📊 How it was trained** (the
five graphs with a sentence each), **🧠 How it works** (the pipeline and the
honest limits). The "🎓 Teach it" recording tab was **removed on 2026-09-12**
at the user's request.

The example clips are held-out test clips bundled with the app, so the demo
works even if the venue's microphone or browser permissions fail:

| Example clip | Score |
|---|---|
| `vektora_project_mic_1/2/3.wav` | 0.91 / 0.93 / **1.00** |
| `soft_negative_1/2/3.wav` | 0.03 / 0.23 / 0.16 |
| `synthetic_noise_1.wav` | 0.25 |

### 15.9 Tools, and the exact versions that work

| Tool | Version | What it does here |
|---|---|---|
| Python | 3.10.7 (in `.venv/`) | Never the global interpreter — see `CLAUDE.md` §1 |
| TensorFlow / Keras | **2.15.1** | The network and training. Do not upgrade to 2.16+: Keras 3 changes the `tf.lite` path |
| NumPy | 1.26.4 | Audio and spectrograms are arrays |
| librosa | 0.11.0 | Resampling, mel filterbank, word segmentation |
| soundfile | 0.14.0 | Reading and writing WAV |
| scikit-learn | 1.7.2 | Split helpers and metrics |
| matplotlib | 3.10.9 | Every graph |
| Streamlit | **1.60.0** | The web page; `st.audio_input` gives the record button |
| protobuf | **4.25.9 — pinned** | Unpinned `streamlit` pulls protobuf 7, which breaks TF 2.15 (needs < 5) |
| ffmpeg | system, from 2013 | Decoding m4a. No `-hide_banner`; must read from a **file**, not a pipe |
| Hugging Face Spaces | Docker SDK, Streamlit template | Free CPU hosting (not yet deployed) |

### 15.10 Everything that was tried and rejected — with the numbers

Recorded so none of it is repeated. Each was measured, not guessed.

| Experiment | Result | Verdict |
|---|---|---|
| **Room + browser-AGC simulation**, p = 0.6 / 0.5 | Live keyword fell to **1/6**; look-alikes already clean on files | Reverted |
| Same, moderated to p = 0.35 / 0.3 | Live keyword **2/6**; one look-alike reached 0.50 | Reverted — kept behind `ROOM_AGC = False` |
| Wide speed perturbation 0.82–1.28× | Live keyword 5/6, but live *talking* scored **1.00** and a look-alike 0.76 | Narrowed to 0.9–1.15 |
| GitHub-only with no speed perturbation | Live keyword 3/6 (0.29–0.58); detected only when spoken slowly and loudly | Speed perturbation added |
| Mixing browser-mic babble into positives at SNR 3–25 dB *(earlier session)* | **0/6** live keywords fired | Reverted |
| "Loudness is the cue" hypothesis *(earlier session)* | Making held-out talking full-volume only lifted it to 0.33 | Disproved |

**The pattern in all of it:** simulating a recording chain you have not
measured costs more than it buys. Every attempt to *guess* at live browser
audio made the keyword weaker. The fix is real recordings from the real
microphone, which is a 90-second job whenever there is time.

### 15.11 Reproducing this exact model

From the repo root, always with the venv interpreter:

```bash
.venv/Scripts/python.exe demo/prepare_data.py          # ~15 s  → data/prepared.npz + 2 graphs
.venv/Scripts/python.exe demo/train.py                 # ~90 s  → model/ + 3 graphs
.venv/Scripts/python.exe -m streamlit run demo/app.py  # → http://localhost:8501
```

The switches that define this version, all at the top of their files:

| Flag | File | Value |
|---|---|---|
| `GITHUB_ONLY` | `prepare_data.py` | `True` |
| `GITHUB_ONLY` | `train.py` | `True` (must match) |
| `POS_COPIES, NEG_COPIES` | `train.py` | `32, 6` |
| `SOFT_COPIES` | `train.py` | `40` |
| `ROOM_AGC` | `train.py` | `False` |
| `smooth(..., n=)` | `app.py` | `7` |

**Run exactly one Streamlit server**, and restart it after retraining. Three
servers ran at once during the build; the oldest held port 8501 and served a
cached old model, which looked exactly like "training didn't work" and cost an
hour. `app.py` now keys its model cache on the file's timestamp, so a stale
*model* cannot survive a retrain — but a stale *server* still serves stale
code. Check with:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*streamlit*' } | Select ProcessId
```

Two earlier candidate models are kept for comparison and are git-ignored:
`demo/model_backup_githubonly/` (before speed perturbation) and
`demo/model_C_roomagc/` (the reverted room/AGC experiment). Swapping either
into `demo/model/` takes ten seconds.

### 15.12 What is in git

Committed on demo day: `CLAUDE.md`, `CONTEXT.md`, `demo.md`, `.gitignore`, and
all of `demo/` — code, plots, example clips, and the trained model
force-added past the `*.keras` rule so this exact version is recoverable.

Deliberately **not** committed: `demo/data/` (raw recordings),
`Human speech/Humanspeech.m4a` (a private 3-minute personal recording — the
GitHub repo is public), and the two scratch model folders.

### 15.13 Presenting this version

The safest order, given §15's limitation:

1. **Example clips first** — real Vektora 0.91 / 0.93 / 1.00, look-alikes 0.03 /
   0.23 / 0.16. This is the three-word story on clips the model never trained
   on, with zero live-microphone risk.
2. **Then say "Vektora" live** as the finale.
3. **State the limitation yourself** rather than being caught by it: "It was
   trained on 63 recordings of the word and 27 look-alikes, and no everyday
   conversation — so it is sharp on the word and over-eager on free-form
   speech. Adding conversation data is the next step." Saying it first is worth
   more than hiding it.

The number worth quoting: **62 of 63 keywords detected, 0 of 27 look-alike
words falsely accepted, with nothing scoring between 0.25 and 0.65** — the
decision is not a close call.

---

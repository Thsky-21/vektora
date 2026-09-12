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

## ⏵ RESUME HERE (handoff written 2026-09-12, ~09:00)

**Everything is built and working locally. Not deployed yet. The one unsolved
problem is live detection through the browser microphone.**

### State in one paragraph

The demo is a 2-class CNN ("Vektora" vs everything else) in `demo/`, trained on
the team's GitHub clips plus the user's own recordings. On held-out *files* it
is excellent (10/10 keywords caught, 0 false alarms across 32 s of continuous
talking). Through the **live browser mic** it is weak: of 6 recorded live
"Vektora"s only **2 fire at Balanced, 4 at Sensitive**. Cause: only 6 live
examples exist, versus 63 project-mic ones. **Fix: the user records 3–5 more
batches in the app's "🎓 Teach it" tab, then re-run `prepare_data.py` +
`train.py`.** Nothing else is blocking; deployment is a 10-minute step whenever
the user is happy.

### Immediate next actions

1. **Ask the user to record more live keywords** — "🎓 Teach it" tab at
   http://localhost:8501, "✅ me saying Vektora", 5–10 repetitions per batch,
   3–5 batches, varying distance/loudness/speed. Target ≥ 30 live words.
   (One batch of 30–60 s of live talking already exists; more is optional.)
2. **Retrain:** `.venv/Scripts/python.exe demo/prepare_data.py` then
   `demo/train.py` (~3 min). Then re-score the live words with the snippet in
   "Diagnostics worth repeating" below. Expect most to clear the threshold.
3. **Deploy** (§11): user runs `.venv/Scripts/hf.exe auth login` in their own
   terminal with a Write token, then `.venv/Scripts/python.exe demo/deploy.py`
   → <https://huggingface.co/spaces/ThSky21/vektora-demo>. First build ≈ 5–10 min.
4. **Rehearse** on the live URL (§12). Presentation is **today**.

### Hard-won lessons — do not rediscover these

- **Run exactly ONE Streamlit server, and restart it after retraining.** Three
  servers were running at once; the oldest held port 8501 and served a
  **cached v0 model**, which looked exactly like "training didn't work" and
  cost an hour. Check with:
  `powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*streamlit*' } | Select ProcessId"`
  (`app.py` now keys its cache on the model file's timestamp, so a stale model
  can't survive a retrain, but a stale *server* still serves stale code.)
- **Streamlit runs every tab in one pass.** An exception in one tab blanks all
  the others — that's why the "Teach it" tab appeared to be missing. The
  risky call is now wrapped in try/except.
- **The user's live-mic complaint was measured, not guessed.** All 25 live
  captures in `demo/data/live_captures/` score ≤ 0.45 with the current model,
  i.e. ordinary speech does **not** fire it. Whether any of those captures were
  the user *saying* "Vektora" is **still unanswered** — worth asking.
- **Two dead ends, already tried — don't repeat:**
  1. *"Loudness is the cue"* — disproved: making held-out talking full-volume
     only lifts it to 0.33.
  2. *Mixing browser-mic babble into positives at SNR 3–25 dB* — made live
     detection **worse** (0/6 fired). Reverted; see the comment in `train.py`.

### Diagnostics worth repeating after each retrain

```bash
cd /c/Users/User/documents/vektora
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe - <<'EOF'
import sys, glob, numpy as np
sys.path.insert(0, "demo")
import app, audio_utils as au
model, norm, metrics = app.load_model(); thr = metrics["threshold"]
seg = sorted(glob.glob("demo/data/_vektora_word_segments/live_vektora_*.wav"))
s = np.array([app.analyse(au.load_audio(f), model, norm, thr)["best_score"] for f in seg])
print(f"thr {thr:.2f} | live words: " + " ".join(f"{v:.2f}" for v in s) + f" -> {(s>=thr).sum()}/{len(s)} fire")
caps = np.array([app.analyse(au.load_audio(f), model, norm, thr)["best_score"]
                 for f in sorted(glob.glob("demo/data/live_captures/*.wav"))])
print(f"live captures (should NOT fire): max {caps.max():.2f}, {(caps>=thr).sum()} fire")
EOF
```

### Current numbers (model trained 2026-09-12 ~08:55, threshold 0.52)

| Measure | Value |
|---|---|
| Test clips | 97 (10 Vektora, 87 other) |
| Hits / misses | **10 / 0** |
| False accepts | **1** (a soft negative: "vector"/"victor") |
| Held-out talking, clip by clip | 53/53 laptop + 8/8 browser-mic rejected |
| Sliding across 32 s of held-out talking | **0 false fires**, top score 0.14 |
| Live browser "Vektora" (6 words) | 0.29 0.29 0.60 0.38 0.46 0.54 → **2/6 Balanced, 4/6 Sensitive** |
| Model | 23,697 parameters, 3,294 training clips after augmentation |

### Where things live

- `demo.md` (this file): §0.4 status, §5 the teaching material, §10 runbook,
  §11 deploy, §12 presentation script, §13 demo-day backups.
- `demo/`: `audio_utils.py` (shared audio→picture), `plots.py` (chart style),
  `prepare_data.py`, `train.py`, `app.py`, `deploy.py`, `Dockerfile`,
  `requirements.txt`, `README.md` (Space config), `model/`, `plots/`, `examples/`.
- `demo/data/` (git-ignored, never uploaded): `other_speech/` (3-min laptop
  m4a), `live_vektora/` + `live_other/` (browser-mic, from the Teach-it tab),
  `live_captures/` (auto-saved live tests), `_vektora_word_segments/` (the cut
  words — listen to these to check the cutting).
- Raw team data: `vektora_*.wav` (63) and `soft-negative/` (27) in the repo root.
- **Nothing is committed to git yet** (`git status` shows everything untracked).

### Environment traps (also in `CLAUDE.md` §1)

- Always `.venv/Scripts/python.exe`; never bare `python`/`pip`.
- `protobuf` must stay pinned at **4.25.9** (streamlit wants 7, TF 2.15 needs < 5).
- ffmpeg on this machine is from **2013**: no `-hide_banner`; m4a must be
  decoded from a temp **file**, not a pipe (both handled in `audio_utils.py`).
- Feature FFT size is **400**, not 512, or the picture is 97 frames, not 98.

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

### 0.4 Build status — what exists now (2026-09-11, evening)

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

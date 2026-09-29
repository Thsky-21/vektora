# Vektora — Project Context

**Problem statement:** SIH26172 — Low Latency & Efficient Voice Activator for Edge Devices.

This file describes *what we are building and why*. Architecture, model layers, wiring
diagrams and code live elsewhere — this is the shared understanding of the product.

---

## The one-sentence version

A battery-powered device that listens continuously for a single spoken keyword, wakes up
the instant it hears it, and streams the speech that follows to a server for
transcription — while proving, with measured numbers, that it stays cheap enough on
power and CPU to run untethered.

---

## The actual problem being solved

Voice assistants normally solve this by streaming *everything* to the cloud, or by
running a large model that needs a phone-class processor. Both are bad on an edge device:

- **Streaming everything** costs bandwidth, costs battery, and sends every private
  conversation in the room to a server.
- **Running a big model continuously** drains the battery in hours and needs hardware
  that costs more than the rest of the device.

So the device must decide *locally* when something is worth transmitting. Everything
else in the project follows from that one constraint.

The answer is a cascade: cheap checks run always, expensive checks run rarely.

1. **Always on, almost free** — is there any sound energy at all? Silence is rejected
   without doing real work. This is the majority of the device's life.
2. **Only when there is sound** — is that sound our keyword? A small neural network
   runs, but only on audio that passed step 1.
3. **Only when the keyword is heard** — open the network connection and stream the
   speech that follows to the server for full transcription.

Each stage is more expensive and runs far less often than the one before it. That is the
entire idea. The power measurements exist to prove the cascade is real.

---

## What the device does, end to end

A person walks up and says: **"<keyword>, what is the weather tomorrow?"**

- The device has been listening the whole time, drawing minimal current.
- It hears the keyword, lights an LED, and begins streaming.
- Crucially, it also sends the **half second of audio it recorded *before* it decided** —
  otherwise the beginning of the sentence is lost, because detection is never
  instantaneous. This buffer of the recent past is what makes the interaction feel
  natural rather than clipped.
- The command is transcribed on the server and appears on a laptop screen.
- When the person stops talking, the device notices the silence, stops streaming,
  and drops back to low-power listening on its own.

The device must do this on battery, with nothing plugged in but the microphone.

---

## What "good" means (the numbers we must hit)

These are the claims we make, and every one has to be *measured*, not estimated. This is
the difference between a demo and a science project.

| What | Target | Why it matters |
|---|---|---|
| Keyword-end → transcription starts | < 300 ms | Below this, the interaction feels instant |
| False activations | < 1 per hour | A device that wakes up randomly is unusable |
| Idle power vs active power | Idle must be dramatically lower | This is the proof the cascade works |
| Model size | Small enough to fit on the chip | Constraint, not a goal in itself |
| Untethered runtime | Hours on one cell | Proves it is genuinely an edge device |

The false-activation number is the hard one. It is easy to build something that detects
the keyword when you say it. It is difficult to build something that *doesn't* detect it
during eight hours of ordinary conversation, TV, and fan noise. Most of the real
engineering effort goes here.

---

## What we are building, concretely

Three separate deliverables that have to meet in the middle:

**1. The keyword detector (Python, trained on a laptop)**
A small neural network trained to recognise one specific word. Its job is to learn the
difference between our keyword, other speech, and background noise. It gets compressed
so it can run on a microcontroller. This is the piece that has to be *trained*, and it
cannot exist without recorded examples.

**2. The device firmware (C, running on the microcontroller)**
Captures audio from the mic, runs the cascade, and streams when triggered. It also
measures its own power draw and reports it. This is where the low-latency claim is
actually won or lost.

**3. The server (Python, on a laptop)**
Receives the streamed audio, transcribes it, and displays the result along with the
measured latency. This is what the judges look at.

---

## The hardware

| Part | What it is there for |
|---|---|
| ESP32-WROOM-32 DevKit V1 | The device itself — everything runs here |
| INMP441 microphone | Its ear. Digital, always listening |
| INA219 current sensor | Measures the device's own power draw live. **This is the proof.** Without it, our efficiency claims are just words on a slide |
| TP4056 + 18650 Li-ion cell | Untethered power. Proves it is not a USB-powered toy |
| 10 µF capacitor | Steadies the mic's supply so the first audio frames aren't corrupted |
| Two LEDs | Visible feedback — one for "heard the keyword", one for "streaming now" |

The INA219 deserves emphasis. It turns *"this is efficient"* into a live graph of
milliamps on a laptop, changing in real time as the device moves between idle,
detecting, and streaming. That is the single most persuasive thing in the demo.

---

## The dependency that gates everything

**The detector cannot be trained without recorded audio, and we have none.**

The model needs to learn three things apart:
- our keyword,
- other words that are *not* our keyword (especially words that sound similar),
- background noise and silence.

The third and second categories are freely available. The first has to be recorded by
the team. Everything downstream — quantization, firmware integration, threshold tuning,
false-activation testing — is blocked behind having those recordings. This is the
critical path.

We are unblocking it the well-trodden way: build and validate the entire pipeline
against Google Speech Commands using **"visual"** as a stand-in word, then swap in our
own "Vektora" recordings once they exist. This means no recording session can be wasted
by a pipeline bug, because the pipeline will already be proven.

---

## Current state (2026-09-10)

- Single scaffold at `sih26172/`. **Nearly every source file is still zero bytes** —
  directory structure only.
- No audio data, no trained model, no firmware written. `model_dscnn.py` is
  written and runs: 23,747 params, ~23 KB as int8, well under the 40 KB budget.
- Python 3.10.7 in an isolated `.venv/`, with TensorFlow 2.15.1, Keras 2.15.0,
  librosa 0.11.0 and numpy 1.26.4 installed and verified importable (`tf.lite`
  resolves, so the jax/TFLite trap is not active in the venv). No NVIDIA GPU
  (training on CPU — acceptable, the model is small).
- Nothing in the ESP-IDF toolchain verified yet.

## Locked decisions

All three founding decisions were closed on 2026-09-10:

- **Keyword: "Vektora"**. Three syllables, a distinctive /kt/ stop-burst, and not a real
  word in any language — so ordinary conversation essentially cannot contain it.
  Recording may now begin. Two known traps: "vector" is a near-collision and must be
  trained against as a hard negative, and the /v/ onset is quiet enough to risk failing
  the stage-1 energy gate.
- **Data strategy: bootstrap first.** Build and validate the entire pipeline against
  Google Speech Commands using **"visual"** as the stand-in word, then swap in real
  clips. "visual" was chosen over the other 34 words because it shares the /v/ onset.
- **Canonical scaffold: `sih26172/`.** `voice_activator/` has been deleted; all 39 of
  its files were 0 bytes, so nothing was lost.

## Resolved

The `esp32s3` instruction in `sih26172/README.md` was wrong for our plain
ESP32-WROOM-32 and has been corrected to `esp32`. The distinction still matters when
reading benchmarks: the S3 has vector instructions that accelerate neural network
inference and our board does not, so any latency figure borrowed from S3 documentation
will be optimistic for us — expect roughly 2-3x slower.

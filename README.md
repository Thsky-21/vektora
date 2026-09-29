# Vektora: low-latency, radio-dark voice activator for edge devices

**Smart India Hackathon 2026 · ISRO · PS SIH26172** "Low Latency and Efficient Voice Activator for Edge Devices" (Hardware)

An ESP32-S3 with an INMP441 microphone listens continuously **with its radio
off**. A two-stage gate (energy, then voice) wakes a 30 KB int8 keyword model
only while someone is speaking. On the keyword **"Vektora"**:

1. the LED lights,
2. the OLED shows the device's own measured stats,
3. Wi-Fi comes up.

A ring buffer holds the keyword and everything said during the connect, so
no audio is lost. It streams over UDP to a local FastAPI + Vosk server,
which sends the transcript back to the device and to a web console. Then
Wi-Fi goes off again. Everything is open source (see `docs/licences.md`).

```
 INMP441 ─I2S─► capture (core 1) ─► ring (2 s) ─► energy gate ─► voice gate ─► log-mel ─► int8 CNN (core 0)
                                        │                                                   │ P(Vektora) ≥ 0.5
                                        │                                  LED ◄────────────┤
                                        └──── pre-roll + connect bridge ──► Wi-Fi on ─► UDP "VAD1" ─► server (Vosk) ─► console
                                                                                                  └─ TRANSCRIPT ─► OLED
```

## Status: what is measured, and what is only built

Rule 0.1 of this project: **a number counts only if the physical board
produced it**. Everything else shows as `—`. The full, dated list is in
`docs/results.md`.

| Part | Built | Proven on the ESP32-S3 |
|---|---|---|
| Keyword model (3-class CNN, int8, 30,536 B) + C/Python feature parity | ✅ | Parity PASS on the laptop (C vs Python, 62 clips). Offline test: 82 % keyword recall, 8.9 % per-clip false accepts |
| Two-stage gate + backfill | ✅ | Tuned offline on dataset clips only |
| Telemetry (CPU per core, RAM, funnel) at 2 Hz | ✅ | Boots and reports on the S3, but so far only with a **dead mic** |
| OLED self-audit screens (T5) | ✅ new | Compiled, not yet run |
| UDP link: on-demand Wi-Fi, clock sync, endpointing (T7) | ✅ new | Compiled, not yet run (needs `secrets.h`) |
| Stride + posterior smoothing (T4) | ✅ new | Compiled, untuned |
| Server (FastAPI + Vosk, UDP + HTTP) | ✅ | Loopback only (`test_sender.py`), no board |
| Web console: Live, Latency, Ledger, Evidence, Before/After, Replay | ✅ | UI only |
| M1–M8 measurement runs | tools ready (`serial_logger.py`, `analyze.py`) | **none run yet**: the S3's INMP441 sends no data (wiring fault) |

**Current blocker:** the microphone on the S3 board. The boot `mic probe`
reports floating noise on both I²S slots. Check the physical wiring (VDD,
GND, SD/SCK/WS jumpers, L/R) before any measurement. `challenges.md` has the
full story.

## Repository layout

| Folder | What |
|---|---|
| `firmware/` | ESP-IDF 5.5.5 project for the ESP32-S3: the submission firmware. [firmware/README.md](firmware/README.md) |
| `training/` | Dataset audit → manifest → train → int8 quantise → C export → parity check. `model/` holds the current trained model; `snapshots/run3-baseline/` the rollback model; `runs/` every training run's report, test scores and logs. |
| `dataset/` | All training audio (2,511 WAVs, including the pre-made augmented copies) + a README of what's what. [dataset/README.md](dataset/README.md) |
| `server/` | FastAPI: UDP `VAD1` + HTTP `/asr` in, Vosk streaming ASR, `/ws/events` out, `/mic` laptop-mic emulator, `test_sender.py` device emulator. |
| `console/` | React + Vite + TypeScript web console (Live, Latency, Ledger, Evidence card, Before/After, Replay). |
| `tools/` | `serial_logger.py` (USB telemetry → JSONL), `analyze.py` (runs → `docs/results.md`), gate tuning scripts. |
| `sessions/` | Recorded runs. `dev-tests-*` = simulated (test_sender). `*-micN` = laptop-mic emulation. **Neither is device evidence.** |
| `docs/` | `current_state.md` (T0 discovery + retrain write-up), `protocol.md`, `results.md`, `licences.md`, `ps.md`, SIH 2026 guidelines, `history/` (older plans). |
| `extras/` | Recorder tool, OLED CPU test, the demo stand-in firmware (**not evidence**), Streamlit tester, the v0 demo. [extras/README.md](extras/README.md) |
| `CLAUDE.md` | The engineering plan (constraints, architecture, protocols M1–M8, task status). |
| `challenges.md` | Everything that went wrong and how it was handled, session by session. |

## Quick start

**Server + console** (laptop, Python 3.12, Node 20+):

```powershell
py -3.12 -m venv server\.venv
server\.venv\Scripts\pip install -r requirements.txt
# Vosk model (55 MB, Apache-2.0), not in git:
#   https://alphacephei.com/vosk/models/vosk-model-small-en-in-0.4.zip  ->  server\models\vosk-model-small-en-in-0.4\
cd console; npm ci; npm run build; cd ..
cd server; .venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000 --no-access-log
# open http://localhost:8000   (Windows Firewall: allow TCP 8000 + UDP 5005)
```

Try it with no hardware: `server\.venv\Scripts\python.exe server\test_sender.py <16 kHz wav>`
(the run is tagged "simulated").

**Firmware:** see [firmware/README.md](firmware/README.md). Copy
`main/secrets.example.h` to `main/secrets.h` first.

**Training** (Python 3.10 + TensorFlow 2.15, separate venv):

```bash
cd training
../.venv/Scripts/python.exe manifest.py && ../.venv/Scripts/python.exe train.py
../.venv/Scripts/python.exe quantize.py && ../.venv/Scripts/python.exe export_c.py && ../.venv/Scripts/python.exe parity_check.py
```

**Measurement runs** (the board on USB):

```powershell
server\.venv\Scripts\python.exe tools\serial_logger.py --port COM10 --forward --out sessions\m2_quiet.jsonl
server\.venv\Scripts\python.exe tools\analyze.py sessions\m2_quiet.jsonl --run M2 --room "quiet lab" --append
```

## Licence note

No project licence file has been chosen yet; pick one before publishing.
The third-party licences are listed in `docs/licences.md`.

# CLAUDE.md — SIH26172 Edge Voice Activator (ESP32-S3 + INMP441)

Smart India Hackathon 2026 · ISRO · PS **SIH26172** "Low Latency and Efficient Voice Activator for Edge Devices" · Hardware.
Idea-submission deadline: **30 Sep 2026**. Today is build day. Optimise for "measured and working" over "complete".

> **2026-09-28 — this file replaces the previous CLAUDE.md wholesale.** The
> full narrative of every prior session (dataset audit, three training runs,
> the 2026-09-18 hardware bring-up on the WROOM-32, the 2026-09-24 SIH
> portal research, the ESP32-S3 recorder tool) is **not repeated here** — it
> lives in `git log -p CLAUDE.md`. Read it if you need the reasoning behind
> a past decision.
>
> **Before doing anything else, read `docs/current_state.md`.** It is
> Task T0 (§13), already done as of this rewrite: framework, exact model
> and feature-extraction params (with values, not placeholders), what
> firmware already works, and — **the one thing that must be resolved
> before Task T2** — a real conflict between two boards used in past
> sessions. The firmware currently flashed and measured (564 ms inference,
> 80,656 B arena) targets a **plain ESP32-WROOM-32**
> (`CONFIG_IDF_TARGET="esp32"` in `prototype/firmware/sdkconfig`), not the
> ESP32-S3 this file's hardware section (§3) assumes. 564 ms is 5.6x over
> any real-time budget and will not meet the latency/idle-CPU metrics (C2,
> C5) on that chip. **Confirm the final board with the team before
> continuing.** Everything else below — a trained int8 model, verified
> feature parity, a two-core audio/inference split, a locked keyword with
> curated hard negatives — is real and already done; extend it, don't
> rebuild it (Rule 0.2).
>
> **2026-09-29 update: board decided = ESP32-S3** (user's call). The
> firmware is retargeted to `esp32s3`, flashed on COM10 and running. The
> WROOM-32 notes above are history now. The current blocker is the **S3's
> mic wiring**: the INMP441 sends no data (see "Next session").
>
> **2026-09-29 (late): the project now lives in this folder,
> `SIH26172-Vektora/`**, a clean, self-contained tree for GitHub. It has
> firmware, training (+ model + every run), dataset, server, console,
> tools, sessions, docs and extras. The old `prototype/` paths mentioned
> below map to this layout: `prototype/firmware` → `firmware/`,
> `prototype/training` → `training/`, `prototype/data/build` →
> `training/runs/`, the external dataset repo → `dataset/`,
> `prototype/{recorder,oled_test,demo_fw,ui}` → `extras/…`, and root `demo/`
> → `extras/streamlit_demo_v0/`. **Work here, not in the old tree.** The
> firmware build dir is `C:\Users\User\esp\vkc` (the old `vkb` belongs to
> the old tree).
>
> **▶ NEW SESSION? Jump to "Next session — start here" at the very end of
> this file** (updated 2026-09-29). It lists what is done, what is
> half-done, what is running (nothing), and the next steps in order.
> **First action: ask whether the mic has been rewired, then reset the board
> and read the `mic probe` boot lines.**

---

## 0. Rules for Claude (read first, apply always)

1. **Never fabricate a measurement.** RAM, CPU, latency, TPR, FA/hr: only values produced by code running on the physical board. Anything unmeasured is shown as `—` in UI, OLED, docs. Estimates in this file are labelled `ESTIMATE` and must be replaced by measurements in `docs/results.md`.
2. **Extend, don't rewrite.** A trained DS-CNN and working firmware already exist. Run Task 0 first. Do not switch framework (Arduino ↔ ESP-IDF) without asking.
3. **Feature extraction on device must match training exactly** (sample rate, window, stride, FFT size, mel bins, MFCC count, log/scale, quantisation params). Read them from the training code; never guess. Mismatch is the #1 silent accuracy killer.
4. **Hot path (audio + gate + features + inference):** no malloc, no printf/logging, no blocking I/O, static buffers only, pinned to core 1.
5. **Open-source only.** No ESP-SR/WakeNet (closed binaries), no Picovoice, no Edge Impulse hosted pipeline, no cloud ASR API. Record every dependency + licence in `docs/licences.md`.
6. **Measure before and after every performance change.** Append a row to `docs/results.md` (date, git hash, config, metric, value, method).
7. All tunables live in `firmware/main/config.h`. Secrets (Wi-Fi SSID/pass, server IP) in `firmware/main/secrets.h` (gitignored) with a committed `secrets.example.h`.
8. After each task: commit, update the **Status** section at the bottom of this file, list what is measured vs not.
9. Keep answers and code comments short. Explain *why* for non-obvious decisions (the team presents this to judges).
10. **Keep `challenges.md` (next to this file) current, session by session.** Before ending a session, add a `## <date> — <session>` section: every challenge hit (bugs, dead ends, hardware/tooling problems, trade-offs), its impact, and its resolution or status (✅ / ⚠️ / ❌). Flip old ❌ entries to ✅ with a date when they get fixed. It is the engineering story for the judges.

---

## 1. What we are building (one paragraph)

An ESP32-S3 listens continuously with the **radio fully off**. A two-stage gate (energy → voice) decides whether sound is speech; only then does the DS-CNN keyword spotter run, which keeps idle CPU < 10%. On the keyword: an **LED lights instantly**, the **OLED shows the live inference and compliance stats**, and **only then does Wi-Fi connect**. A rolling audio buffer captured the keyword and everything said during Wi-Fi connection, so **no audio is lost**. Audio streams over UDP to a local **FastAPI + Vosk** server, which streams the live transcript to a **web console**. When speech ends, Wi-Fi is switched off again.

---

## 2. Hard constraints (from the PS) and how we measure them

> TODO(team): paste the exact PS text from the SIH portal into `docs/ps.md` and correct this table if wording differs. The official page could not be retrieved; this table reflects the constraints consistently cited across multiple SIH26172 team repos.
>
> Note: the official PS text **was** fetched into this repo on 2026-09-24
> (see `git log -p CLAUDE.md` for the old §13.1 — organisation ISRO,
> category Hardware, theme Smart Automation, evaluation on efficiency /
> accuracy / latency). It should already answer most of the TODO above;
> someone still needs to write it into `docs/ps.md` in the format this
> table expects and confirm no wording drifted since 09-24.
>
> **Correction (2026-09-28, checked):** that 09-24 PS text is **not** in git
> history. It was never committed (`git show` of every CLAUDE.md revision
> has no "Smart Automation"/PS body). `docs/ps.md` therefore still does not
> exist. Re-fetch the PS from the SIH portal; don't write it from memory.

| # | Requirement | Our definition (state this on slides) | How measured |
|---|---|---|---|
| C1 | RAM < 256 KB | **Internal SRAM used = static (.data+.bss from `idf.py size`) + runtime heap used.** Report **idle** and **peak (during streaming)** separately. PSRAM is not used for the pipeline; if PSRAM is used for anything, report it separately. | `heap_caps_get_total_size / get_free_size / get_minimum_free_size (MALLOC_CAP_INTERNAL)` + size report |
| C2 | Idle CPU < 10% | CPU busy % = 100 − idle-task share, **per core and average**, over 1 s windows. "Idle" = listening, no keyword. Report two conditions: **quiet room** and **ambient non-keyword speech**. | FreeRTOS run-time stats (`CONFIG_FREERTOS_GENERATE_RUN_TIME_STATS=y`), IDLE0/IDLE1 deltas. Fallback: idle-hook counter calibrated at boot. |
| C3 | Open-source only | Every component has an OSI licence | `docs/licences.md` |
| C4 | High TPR, near-zero false activations | TPR at 0.3 / 1 / 2 m; **false accepts per hour** on long negative audio | Protocols M4, M5 (§11) |
| C5 | Low latency keyword-end → ASR | Stage-wise waterfall L1–L4 (§7) with stated clock-sync method | Protocol M6 |

CPU clock fixed at 240 MHz, dynamic frequency scaling off (so CPU% is well-defined). State this.

---

## 3. Hardware

| Part | Notes |
|---|---|
| ESP32-S3 DevKitC-1 (TODO: module variant, e.g. N8R8/N16R8) | Dual LX7 @240 MHz, 512 KB internal SRAM, vector instructions (ESP-NN) |
| INMP441 I²S MEMS mic | 24-bit data left-justified in 32-bit slots. L/R pin → GND = left channel |
| SSD1306 OLED 128×64, I²C 0x3C | 21 chars × 8 lines at 6×8 font |
| LED + 330 Ω | Wake indicator (onboard RGB optional: GPIO48 on v1.0, GPIO38 on v1.1) |

Default pins (TODO(team): replace with actual wiring in `config.h`). Avoid GPIO 0/3/45/46 (strapping), 19/20 (USB), 26–32 (flash), 33–37 (octal PSRAM).

```
I2S_BCLK=GPIO4  I2S_WS=GPIO5  I2S_SD=GPIO6
OLED_SDA=GPIO8  OLED_SCL=GPIO9  (I2C 400 kHz, try 800 kHz)
LED_WAKE=GPIO2
```

> **These are template defaults from this plan, not verified wiring.** Two
> boards have real, different, previously-used pinouts — see
> `docs/current_state.md` "Hardware conflict" before wiring anything:
> WROOM-32 (flashed, working, measured): SD=GPIO32, WS=GPIO25, SCK=GPIO26,
> LED=GPIO4. A separate ESP32-S3 (recorder tool only): SCK=GPIO6, WS=GPIO5,
> SD=GPIO4. Its mic read all-zero samples in an earlier test.
>
> **2026-09-29: the S3 is final.** Firmware uses **SCK=GPIO4, WS=GPIO5,
> SD=GPIO6, L/R→3V3 (right slot)**, LED=GPIO2, OLED SDA=8/SCL=9
> (`sdkconfig.defaults` + `config.h`; corrected 2026-09-29 late, the earlier
> "SD=4/SCK=6" text was stale). Module: QFN56 rev v0.2,
> 8 MB embedded PSRAM, 16 MB flash (read with esptool). The mic is still
> not delivering data on this wiring (floating SD).

---

## 4. Architecture

```
 CORE 1 (real-time)                                   CORE 0 (everything else)
 ─────────────────                                    ───────────────────────
 INMP441 ─I2S DMA(6×20ms)─► audio_task                 ui_task (OLED, low prio)
                             │ 32→16 bit, gain, DC-HPF telemetry_task (USB JSON 2 Hz)
                             │ write ─► RING (PCM16)   link_task (Wi-Fi + UDP, only after wake)
                             │ Stage0: energy gate     IDF Wi-Fi/lwIP tasks (pinned core 0)
                             │ Stage1: voice gate
                             ▼ (speech only)
                           kws_task
                             │ streaming MFCC (+backfill from RING on gate open)
                             │ DS-CNN int8 (TFLM + ESP-NN)
                             │ posterior smoothing, threshold, refractory
                             ▼ DETECT
                LED on (µs) ─┼─► ui_task: DETECT screen
                             └─► link_task: radio on → fast connect → UDP stream RING from t_detect−PREROLL
                                                      │
                                   Wi-Fi (LAN, local AP)
                                                      ▼
                         FastAPI server: UDP rx → reorder → Vosk streaming → /ws/events → Web console
                                         └─ TRANSCRIPT packet back to device → OLED
```

### Device state machine
`LISTEN` → (gate open) `SPEECH` → (score ≥ thr) `DETECTED` → `LINKING` → `STREAMING` → (endpoint: 700 ms silence or 8 s max) `CLOSING` (Wi-Fi stop) → `LISTEN`.
`LINKING` timeout 3 s → `LINK_FAIL` (log, OLED, LED blink) → `LISTEN`. KWS is paused during `STREAMING`. Refractory 1 s after `CLOSING`.

---

## 5. Firmware design

Layout (adapt to existing project; keep existing model files):
```
firmware/main/
  app_main.c      config.h  secrets.h(gitignored)
  audio_in.c      I2S config, sample conversion, DC high-pass, peak/RMS
  ring.c          lock-free SPSC ring of int16 with absolute sample index
  gate.c          stage0 + stage1, hangover, funnel counters
  features.c      streaming MFCC (must match training)
  kws.cc          TFLM interpreter, ESP-NN kernels, smoothing
  link.c          on-demand Wi-Fi, UDP streaming, clock sync, endpointing
  ui_oled.c       SSD1306 screens
  led.c
  telemetry.c     USB serial JSON
  metrics.c       CPU%, RAM, latency stamps (esp_timer_get_time µs)
```

### 5.1 Audio front end
- I2S std mode, 16 kHz, 32-bit slot, mono left, DMA 6 descriptors × 320 frames (20 ms each). `i2s_channel_read` blocks → CPU idles while waiting.
- Conversion: `int16 = clip((raw32 >> MIC_SHIFT) * MIC_GAIN)`. Start `MIC_SHIFT=14`, tune 11–14 while watching peak/RMS in telemetry. Current range is short (~15–20 cm), so this is the first fix. **Retrain/fine-tune only if the new gain changes the feature distribution noticeably; otherwise keep the model.**
- One-pole DC high-pass before everything else.

### 5.2 Ring buffer (pre-roll + Wi-Fi bridge)
- One circular buffer, written continuously by `audio_task`, tagged with absolute sample index → timestamps derivable.
- Serves three consumers: MFCC backfill on gate open, pre-roll on wake, and **the bridge during Wi-Fi connect**.
- Size: `RING_MS = PREROLL_MS + p95(wifi_connect_ms) + 300 ms margin`. Start `PREROLL_MS=700` (contains the keyword → helps server verification and guarantees the first command syllable), `RING_MS=2000` → 64 KB PCM16. Resize after measuring connect time.
- If RAM is tight: `RING_CODEC=ULAW` (G.711 μ-law, 8 bit) halves RAM and uplink bandwidth. Default `PCM16` until everything works; verify ASR accuracy before switching.
- Overflow (writer laps streamer) is counted and shown; it must be 0 in demos.

### 5.3 Two-stage gate (the idle-CPU mechanism)
Per 20 ms frame:
- **Stage 0, energy (always on, ~µs):** frame RMS vs adaptive noise floor (fast-down / slow-up tracker). Pass if RMS > floor × `GATE0_RATIO` (start +9 dB) for `GATE0_FRAMES` (start 2) consecutive frames.
- **Stage 1, voice check (only when Stage 0 passes):** reuse the MFCC FFT. Speech-band energy ratio (300–3400 Hz / total) > `GATE1_BAND_RATIO` AND spectral flatness < `GATE1_FLATNESS` (noise is flat, voice is harmonic). Optional ZCR sanity check.
- **Open → hangover** `GATE_HANGOVER_MS=400` so the keyword isn't chopped.
- **Backfill:** on gate open, compute MFCC for the last ~300 ms from the ring once, so the KWS window contains the keyword onset the gate needed time to confirm.
- **Funnel counters** (since boot and rolling 60 s): frames heard / passed S0 / passed S1 / inferences run / detections. DS-CNN duty cycle % = inference time ÷ wall time.
- `GATE_ENABLED=0` must be supported, for the before/after baseline.
- Alternative if the custom gate is weak: port WebRTC VAD (BSD-3). Decide by FA/hr and CPU, not taste.

### 5.4 KWS
- TFLM with **ESP-NN optimised kernels** (esp-tflite-micro component). Static tensor arena; after `AllocateTensors()` shrink the arena to `arena_used_bytes() + 1 KB`.
- Streaming features: compute MFCC only for new frames and shift the feature matrix; never recompute the whole 1 s window.
- Inference stride `KWS_STRIDE_FRAMES` (start 2 = 40 ms). Smoothing: mean of last `KWS_SMOOTH_N=3` posteriors ≥ `KWS_THRESHOLD`. Refractory 1 s.
- Timestamps (µs): `t_win_end` = capture time of the last sample in the triggering window (our definition of keyword end), `t_detect` = decision time, `infer_us`.

### 5.5 LED + OLED
- LED: `gpio_set_level` is the **first** action after detection (before logging, OLED, Wi-Fi). LED on while streaming; 3 short blinks on `LINK_FAIL`.
- OLED lives in `ui_task` on core 0, low priority, never blocks core 1. Full refresh only on state change; in `LISTEN`, refresh at `OLED_IDLE_REFRESH_MS=1000` (measure CPU with OLED on vs off; option `OLED_OFF_IN_IDLE`).
- Screens (fit 21×8):
```
LISTEN                      DETECT / STREAM                 RESULT (from server)
SIH26172  LISTEN  RF:OFF    WAKE  score 0.97                "open the log file"
CPU  3.1%  <10  PASS        infer 14ms  L1 38ms              L2 wifi  512ms
RAM 168/256KB   PASS        wifi 512ms  RF:ON               first-byte  561ms
DSCNN duty 1.8%             stream 1.2s  lost 0             lost 0  ovf 0
Gate 5.1%>1.9%>det 0        ring ovf 0                      back to LISTEN
FA since boot 0             sent 38KB
up 01:12:45
```
Show `—` for anything not yet computed.

### 5.6 Link (Wi-Fi only after the keyword)
Policy flag `WIFI_POLICY`:
- `ON_DEMAND` (default, our story): radio never started while idle. On detect: `esp_wifi_start` → connect → stream → `esp_wifi_stop`.
- `WARM` (comparison only): stay associated with modem-sleep. Exists to **measure the trade-off**, not to ship.

Fast-connect for `ON_DEMAND`:
- `esp_wifi_init` at boot vs on demand: flag `WIFI_INIT_AT_BOOT`; measure RAM and connect time for both, pick with data.
- Cache AP BSSID + channel in RAM/NVS after first connect; `bssid_set=1`, `channel=<cached>`, `scan_method=WIFI_FAST_SCAN`, WPA2-PSK, `WIFI_STORAGE_RAM`.
- **Static IP** (stop DHCP client) to skip DHCP.
- `WIFI_PS_NONE` while streaming.
- Dedicated 2.4 GHz AP on a fixed channel (laptop hotspot or cheap router), server on the same LAN. No internet needed.
- Tune Wi-Fi RAM: static RX buffers low, dynamic TX, lwIP UDP only. Measure peak RAM.

Streaming:
- UDP, 20 ms of audio per packet. Send `HELLO` first, then **backlog burst** from `t_detect − PREROLL_MS` at up to ~4× real-time with light pacing, then real-time. No TCP handshake, no WebSocket upgrade.
- Clock sync: 3 `PING`s interleaved with audio right after link-up; server uses the min-RTT sample: `offset = t_srv − (t_dev + RTT/2)`, error bound ±RTT/2.
- Endpoint: gate closed for 700 ms after speech, or 8 s cap → `END` → Wi-Fi stop.
- Record `t_wifi_start, t_assoc, t_ip, t_sock_ready, t_first_tx`.

Plan B if `ON_DEMAND` connect time is too slow (decide after measuring, not before): (a) present `WARM` as a selectable mode with both measured; (b) stretch: ESP-NOW to a second ESP32 gateway on the laptop USB (no association step).

### 5.7 Telemetry (USB serial, 921600 baud, newline JSON)
Radio is off while idle, so evidence flows over USB (and the OLED).
```json
{"type":"tick","t_us":0,"state":"LISTEN","cpu0":0.0,"cpu1":0.0,"cpu_avg":0.0,
 "ram_static":0,"ram_heap_used":0,"ram_used":0,"ram_peak":0,"arena_used":0,"model_bytes":0,
 "mic_peak":0,"mic_rms":0,"noise_floor":0,"score":0.0,
 "funnel":{"frames":0,"s0":0,"s1":0,"infer":0,"det":0},"dscnn_duty":0.0,"ring_ovf":0}
{"type":"detect","t_win_end_us":0,"t_detect_us":0,"score":0.0,"infer_us":0}
{"type":"link","t_wifi_start_us":0,"t_assoc_us":0,"t_ip_us":0,"t_sock_us":0,"t_first_tx_us":0,"ok":true}
{"type":"stream_end","bytes":0,"packets":0,"ring_ovf":0}
```
Tick at 2 Hz, events as they happen. `firmware_hash` and `config_hash` printed at boot.

---

## 6. Memory budget (ESTIMATE — replace with measured)

| Item | ESTIMATE |
|---|---|
| Tensor arena | 20–60 KB (model dependent) |
| Ring 2.0 s PCM16 | 64 KB (32 KB if μ-law) |
| I2S DMA 6×320×4 B | ~8 KB |
| FFT/MFCC buffers + feature matrix | ~10 KB |
| Task stacks | ~20 KB |
| OLED framebuffer | 1 KB |
| IDF/FreeRTOS baseline | measure |
| Wi-Fi + lwIP when active | ~40–80 KB, tunable |

Idle RAM excludes the Wi-Fi stack (another benefit of radio-dark idle). **Peak** during streaming is the risk: if it exceeds 256 KB, apply in order: shrink arena → μ-law ring → reduce Wi-Fi buffers → shorten ring.

> Note: the tensor arena is **not actually an estimate** — it was measured
> at **80,656 bytes** on the WROOM-32 build (`docs/current_state.md`). If
> the final device is also an int8 model of this size, start the arena
> budget from that number, not from the 20–60 KB range above.

---

## 7. Latency definitions (put this box on the slide)

| Stage | Definition | Clock |
|---|---|---|
| L1 detect | `t_detect − t_win_end` | device |
| L2 wake-to-link | `t_sock_ready − t_detect` (radio start + association + socket) | device |
| L3 uplink | server receipt of first packet − `t_sock_ready` | synced (±RTT/2) |
| L4 ASR first text | first Vosk partial − server receipt | server |
| **Total** keyword-end → server receipt | L1 + L2 + L3 | synced |
| Zero-loss | samples lost between `t_detect − PREROLL` and END (seq gaps + ring overflow) | both |

Reduce each stage: L1 (ESP-NN, stride, smoothing N), L2 (fast connect, static IP, init-at-boot), L3 (UDP, backlog burst), L4 (Vosk chunk size ~100 ms, small model). Report median and p90 over ≥20 trials per policy.

> L1 today is **~564 ms measured on the WROOM-32** — see the hardware
> conflict at the top of this file. This single number is why the board
> decision blocks everything else in this section.

---

## 8. UDP protocol (little-endian, packed)

```
header: magic u32 'VAD1' | session u16 | type u8 | codec u8 | seq u32 | t_capture_us u64 | len u16
types device→server: 1 HELLO (payload JSON: score,t_win_end_us,t_detect_us,t_sock_us,wifi_ms,fw,cfg,codec)
                     2 AUDIO (payload: 20 ms PCM16 or μ-law)
                     3 END   4 PING (payload t_dev_us)
types server→device: 5 PONG (t_dev_us, t_srv_us)   6 TRANSCRIPT (≤60 chars UTF-8)
port 5005
```
Document in `docs/protocol.md`; server and firmware must share it verbatim.

---

## 9. Server (`server/`, Python 3.11)

- `app.py`: FastAPI. Lifespan starts an asyncio UDP endpoint on :5005. Serves built console at `/`, `GET /api/sessions`, `GET /api/sessions/{id}`, WebSocket `/ws/events`.
- Per session: reorder by `seq` (≤60 ms wait), decode, feed Vosk `KaldiRecognizer` in ~100 ms chunks on a worker thread (Vosk is blocking). Broadcast `server_rx_first`, `asr_partial`, `asr_final`, `pkt_loss`, `clock_sync` events. Append everything to `sessions/<session>.jsonl` with `time.monotonic_ns()` and wall time.
- On `asr_final`: send `TRANSCRIPT` back to the device.
- Vosk model: `vosk-model-small-en-in` (Indian English small); make the model path configurable.
- `test_sender.py`: emulates the device from a WAV (HELLO, backlog burst, real-time, END) so server + console can be built without hardware.
- Stretch — **server-side keyword re-verification**: run a grammar-restricted Vosk recognizer on the pre-roll; if the keyword is absent, mark the wake `server_rejected` (two-stage verification, same idea as Apple's "Hey Siri" cascade). Only valid if the keyword exists in the Vosk vocabulary; otherwise skip.
- No database, no cloud APIs.

---

## 10. Web console (`console/`, React + Vite + TypeScript)

Style: white background, navy `#0B2545` headers, one saffron accent `#F28C28` for keyword events, green/red only for pass/fail, Inter + monospace numbers, no gradients/emoji. Header: `SIH26172 · ISRO · <PRODUCT_NAME> · build <hash> · session <date>`.

Data sources: Web Serial (device telemetry, Chrome/Edge) + `/ws/events` (server). Mode toggle **Live / Replay** (Replay loads a recorded session JSON; used for the Vercel deploy).

Pages, in priority order:
1. **Live** — gauges with ISRO limit lines (RAM vs 256 KB, CPU vs 10%), gate funnel bars, score trace with threshold, state strip, radio state (OFF/ON), live transcript (partials in grey, finals in black).
2. **Latency** — per-wake stacked bars L1–L4, table of last 20 with median/p90, "Measurement method" box. Skip incomplete wakes and count them.
3. **Ledger** — FA soak (timer, negative-audio description, count, FA/hr, threshold) + positive TPR table by distance + confusable-word checklist.
4. **Evidence** — 1600×900 Compliance Card (Metric | ISRO requirement | Measured | Method | Status), export PNG + CSV/JSON.
5. **Before/After** — CPU and DS-CNN duty with `GATE_ENABLED=0` vs `1`.

`—` for anything not received. Replay banner: "Recorded session from physical ESP32-S3, <date>".

---

## 11. Measurement protocols (each run → `sessions/`, summarised by `tools/analyze.py` into `docs/results.md`)

`tools/serial_logger.py` logs USB telemetry to JSONL (more reliable than a browser tab for multi-hour runs).

| ID | Run | Duration | Output |
|---|---|---|---|
| M1 | Baseline: `GATE_ENABLED=0`, quiet room | 10 min | CPU per core, RAM |
| M2 | Gate on, quiet room | 10 min | idle CPU, RAM idle |
| M3 | Gate on, non-keyword speech at 1 m | 10 min | CPU under ambient speech, funnel, DS-CNN duty |
| M4 | FA soak: multilingual talk radio/podcasts (Hindi/Kannada/English) + fan noise | ≥2 h, overnight if possible | FA/hr at the fixed threshold |
| M5 | Positive: 0.3 / 1 / 2 m × ≥20 utterances × ≥3 speakers | — | TPR per distance |
| M6 | Latency: ≥20 wakes per `WIFI_POLICY` | — | L1–L4 median/p90 |
| M7 | Zero-loss: keyword immediately followed by "one two three four five" | 10 trials | transcript contains "one"; seq gaps = 0; ring_ovf = 0 |
| M8 | Peak RAM during streaming | 10 wakes | peak internal RAM |

Every run records firmware hash, config hash, threshold, mic distance, room description.

---

## 12. Differentiators (what judges must *see*)

Context: nearly every visible SIH26172 team uses ESP32-S3 + INMP441 + DS-CNN; at least one already shows transcripts on an OLED, and one claims <300 ms trigger-to-ASR using a pre-opened socket. Stack and OLED alone are not differentiators. These are:

1. **Radio-dark idle with zero-loss wake.** The device emits no RF and holds no Wi-Fi memory while listening; it connects only after the keyword, and the ring buffer bridges the connection gap so the transcript starts with the first word. Demo proof: laptop-hotspot client list shows the device appear only after the keyword; OLED `RF:OFF → RF:ON`; M7 shows 0 lost samples. Relevance: RF-sensitive labs, receivers/observatories, battery nodes, privacy (0 bytes leave while idle).
2. **Self-auditing device.** The OLED shows the device's own measured compliance (CPU vs 10%, RAM vs 256 KB, PASS/FAIL, last latency) with no laptop attached. Judges read the claim off the hardware.
3. **Visible cascade.** Gate funnel: "frames heard → passed energy gate → passed voice gate → DS-CNN ran X% of the time → 0 false wakes", plus the measured before/after (gate off vs on) CPU. This makes the <10% claim self-explanatory.
4. **Engineered latency trade-off.** Stage-wise waterfall with a stated clock-sync method, measured for both `ON_DEMAND` and `WARM` policies. We show the cost of radio-dark idle in ms and why we chose it.
5. **Measured, not claimed.** Competitor repos list "<256 KB / <10%" as targets marked unmeasured. Our Compliance Card traces every number to a session file.
6. Stretch: server-side keyword re-verification (two-stage, lowers FA/hr).

---

## 13. Task plan (in order; each has an acceptance test)

| # | Task | Acceptance | Time box |
|---|---|---|---|
| T0 | **Discovery.** Read existing firmware + training code. Write `docs/current_state.md`: framework, IDF/Arduino version, model input shape, exact MFCC params, arena size, keyword, pins, current mic conversion, what works. Change nothing. | File exists, feature params listed with source file/line | 20 min |
| T1 | **Metrics + telemetry first.** CPU per core, RAM breakdown, JSON tick at 2 Hz, `tools/serial_logger.py`. Run **M1 baseline** on the current firmware. | Valid JSON (`python -m json.tool`), baseline row in `docs/results.md` | 60 min |
| T2 | **Mic front end + ring.** Shift/gain constants, DC-HPF, peak/RMS; SPSC ring with absolute index. | Speaking at 1 m gives clearly higher RMS than before; ring unit test passes | 45 min |
| T3 | **Gate** (Stage 0 + 1, hangover, backfill, funnel, `GATE_ENABLED`). Run M2, M3. | Idle CPU measured; keyword still detected at 0.3 m with gate on | 75 min |
| T4 | **KWS tuning:** ESP-NN on, streaming MFCC, stride, smoothing, refractory, timestamps. | `detect` events with `infer_us`, L1 measured | 45 min |
| T5 | **LED + OLED** (§5.5). Measure CPU with OLED on vs off. | LED on within 1 ms of `t_detect`; OLED screens render | 45 min |
| T6 | **Server + test_sender** (can run in parallel with T2–T5). | WAV → live partials on `/ws/events`, JSONL written | 75 min |
| T7 | **Link:** on-demand Wi-Fi fast connect, UDP streaming, clock sync, endpoint, Wi-Fi stop, `WIFI_POLICY`. | Speak → transcript on server; M7 passes; L2 measured | 120 min |
| T8 | **Console** Live → Latency → Ledger → Evidence → Before/After, Replay mode. | Live page works with board + server; Replay works offline | 150 min |
| T9 | **Measurement runs** M4 (start as soon as T4 is stable, run unattended), M5, M6, M8. `tools/analyze.py` fills `docs/results.md`. | Every Compliance Card cell filled or `—` | parallel |
| T10 | Replay build to Vercel (static, no backend calls). | URL opens in incognito on a phone | 30 min |
| S1 | Stretch: server keyword re-verification | `server_rejected` counter | — |
| S2 | Stretch: `RING_CODEC=ULAW` | RAM and bandwidth halved, M7 still passes | — |

**Hard cutoffs.** After T1: if run-time stats won't work, use the idle-hook counter and label it. After T7: if Wi-Fi streaming isn't stable, stream audio over USB serial to the server and label it honestly. **Firmware freeze tonight (git tag `sub-v1`)**; 29 Sep = measurements, console polish, video, PDF; submit on the morning of 30 Sep.

**Cut order if behind:** S1/S2 → Before/After page → Ledger UI (use a spreadsheet) → Latency UI (table in slide). **Never cut:** T1 telemetry, gate, LED/OLED, M4 soak, Compliance Card.

---

## 14. Repo layout and commands

```
<repo>/
  CLAUDE.md
  firmware/   ESP-IDF (or existing framework) project
  server/     app.py, asr.py, udp_rx.py, protocol.py, test_sender.py, requirements.txt
  console/    React + Vite + TS
  tools/      serial_logger.py, analyze.py
  sessions/   *.jsonl (runs), demo_session.json (replay)
  docs/       ps.md, current_state.md, protocol.md, results.md, licences.md
```

> **2026-09-29 late:** this layout now exists as `SIH26172-Vektora/` (see
> the top of this file). It also has `training/` (with `runs/` and
> `snapshots/`), `dataset/`, and `extras/`. The manifest stores
> repo-relative paths (`dataset/...`). The commands below are written for
> this folder; for firmware use the `-B C:\Users\User\esp\vkc\build -D
> SDKCONFIG=C:\Users\User\esp\vkc\sdkconfig` form from `firmware/README.md`.

Commands (verified working on this laptop, 2026-09-28):
```
# ESP-IDF lives at C:\Users\User\esp\esp-idf-v5.5.5 (NOT C:\Espressif). PowerShell:
. C:\Users\User\esp\esp-idf-v5.5.5\export.ps1
cd prototype/firmware; idf.py build; idf.py -p COMx flash monitor     # idf.py size for static RAM
#   Build dirs must have SHORT paths (Windows 260-char limit): a build under the
#   Claude scratchpad fails with "opening dependency file ... No such file".

# server: its own venv, Python 3.12 (NOT the training .venv, which is 3.10 + TF)
cd server; .venv/Scripts/python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000 --no-access-log
cd server; .venv/Scripts/python.exe test_sender.py <16k mono wav> [--http] [--trials 5] [--loss 0.03 --reorder 0.05 --skew-ms 5000]
#   re-create venv if missing: py -3.12 -m venv server/.venv ; server/.venv/Scripts/pip install -r server/requirements.txt
#   Vosk model: server/models/vosk-model-small-en-in-0.4 (gitignored; from alphacephei.com/vosk/models)

# console (React+Vite+TS, Node 24). Dev proxies /api and /ws to :8000.
cd console; npm i; npm run dev            # http://localhost:5173 ; npm run build -> console/dist (FastAPI serves it at /)
#   VITE_MODE=replay npm run build  -> static Replay-only build for Vercel (needs public/demo_session.json)

# logger (board on USB; --forward relays JSON to the console via the server)
server/.venv/Scripts/python.exe tools/serial_logger.py --list
server/.venv/Scripts/python.exe tools/serial_logger.py --port COMx --forward --out sessions/<run>.jsonl
```
Windows Firewall must allow inbound TCP 8000 + UDP 5005 for the server's python.exe, or the board can't reach it.

> Target is now **`esp32s3`** (2026-09-29) in both `sdkconfig.defaults` and
> `C:/Users/User/esp/vkb/sdkconfig`. The console baud (921600) only takes
> effect with `CONFIG_ESP_CONSOLE_UART_CUSTOM=y`. Without it IDF silently
> keeps 115200 and the logger reads garbage.

---

## 15. Dependencies and licences (verify, then record in `docs/licences.md`)
ESP-IDF (Apache-2.0) · esp-tflite-micro / TFLM (Apache-2.0) · ESP-NN (Apache-2.0) · ESP-DSP (Apache-2.0) · Vosk (Apache-2.0) · FastAPI (MIT) · React/Vite (MIT) · OLED driver: esp_lcd SSD1306 (IDF) or U8g2 (BSD-2). Not allowed: ESP-SR, Picovoice, Edge Impulse hosted training, cloud ASR.

---

## 16. TODO(team) before starting

- [ ] **Keyword: `Vektora`.** Already locked and trained (not a TODO — filled
      in from prior sessions). Curated hard negatives already recorded for
      the near-collisions (vector/victor/Vectra/spectra/sector), `/kt/`
      words, `/v/`-onset words, same-rhythm words, and partial-keyword
      fragments. See `docs/current_state.md`.
- [ ] Product name: `<PRODUCT_NAME>` still open (avoid names close to
      existing SIH26172 repos, e.g. "EdgeVani").
- [x] **Final board: ESP32-S3** (decided 2026-09-29). Flashed and running;
      mic wiring still to fix.
- [ ] Team name / ID — not recorded in this repo; ask the SPOC.
- [ ] Exact PS text into `docs/ps.md` — was fetched 2026-09-24 (see git
      history), just needs transcribing into that file's format.
- [ ] Actual pins: firmware uses SD=4/WS=5/SCK=6/LED=2. Module variant is
      known (8 MB PSRAM, 16 MB flash). **Mic wiring still needs checking**
      (floating SD, 2026-09-29).
- [ ] Wi-Fi AP (2.4 GHz, fixed channel), server static IP, device static IP.

---

## Status (Claude updates after each task)
| Task | State | Measured values | Notes |
|---|---|---|---|
| T0 | **done** | See `docs/current_state.md` in full. Key facts: ESP-IDF v5.5.5 targeting plain `esp32` (not s3); feature parity verified to <0.0001 dB / near-zero int8 diffs; arena 80,656 B measured; inference 564 ms measured (WROOM-32); features 4.10 ms/frame measured. | Discovery surfaced one blocking conflict (board identity). **Resolved 2026-09-29: ESP32-S3.** |
| **T0.5 (not in the original plan — model retrain, 2026-09-28)** | **done** | New data folded in: `positive_akash` (77 clips, user's own "Vektora") + `negative_akash` (101 clips, user's own look-alikes). New model: int8 30,536 B, `sha256 75997aadce3e9f7c...` (was `e9e10d52a17832fb...`), exported into `prototype/firmware/main/`. Test (152 clips): 88.2% 3-class acc, keyword recall 82.1% (32/39), **akash_pos recall 80% (16/20)**, false-accept rate 8.85% (10/113, up from Run 3's 5.3%). Parity re-verified: PASS. | Full writeup incl. 3 mix-tuning attempts and the known limitation (31.6% false-accept on the user's own look-alike words) in `docs/current_state.md` §"2026-09-28 retrain". Model is flash-ready but NOT yet tagged/snapshotted or committed. This retrain happened before T1–T9. (Later the same day a parallel session built T6/T8 and part of T1/T7 — rows below.) The board-identity conflict was resolved 2026-09-29 (S3). |
| T6 server + test_sender | **done** (uncommitted) | Loopback only, no board, no Wi-Fi: END→TRANSCRIPT median 30.5 ms (UDP, n=5); HTTP last byte→reply median 41.3 ms (n=3); server decode lag ≈43 ms median; Vosk RTF 0.15; clock sync recovered a 5 s skew to 25 µs (±124 µs). Rows in `docs/results.md`. | `server/` (own `.venv`, Py 3.12, Vosk small-en-in). Accepts the **current firmware's HTTP `POST /asr`** (raw PCM16 body, reply `{"text"}`) and the §8 UDP protocol. Latency choices, all measured: model loaded and warmed at startup (cold first decode 2.8 s), pool of pre-built recognizers (144 ms each), decode while audio streams in, 40 ms Vosk chunks (vs 20/100), PONG sent before any other work, QPC clock (Windows `time.monotonic` only ticks every 15.6 ms). PING carries the previous RTT → server computes the offset (`docs/protocol.md`). Sessions → `sessions/<id>.jsonl` + `.wav` (wav gitignored); test runs archived in `sessions/dev-tests-20260928/`. |
| T1 | **code done, flashed + running on the S3** (2026-09-29) | On the S3 (`firmware_hash bf4dc6f63`), taken with a **dead mic** (noise input), so the numbers are provisional: ram_static 51,844 B; idle ram_used ~320 KB (**over 256 KB**), ram_peak 329 KB, heap_free ~68 KB; cpu0 1.3 %, cpu1 15–26 % (noise passes Stage 0 on ~50 % of frames). Boot + 2 Hz ticks verified at 921600. **M1 not run.** | Host: `tools/serial_logger.py` (unchanged). Firmware: new `main/telemetry.c/.h` replaces the old `spec_task`. Task `telem` on core 0, prio 2, prints newline JSON: `boot` (firmware_hash = ELF sha, version, config_hash = FNV of every config.h tunable + model sha + shift, ram_static, arena, model_bytes), **2 Hz `tick`** (state LISTEN/SPEECH/LINKING/STREAMING, radio, cpu0/cpu1/cpu_avg over 1 s windows from FreeRTOS run-time counters in esp_timer µs = 100 − IDLE share, cpu_capture, dscnn_duty = infer task share, ram_static (.data+.bss+.noinit linker symbols) / ram_heap_used / ram_used / ram_peak (min-free watermark) / heap_free / heap_largest, mic_peak/mic_rms dBFS, noise_floor, score, score_max, funnel {frames,s0,s1,opens,infer,det}, backfill, backfill_miss, ring_ovf, ev_dropped), events `detect` (t_win_end_us, t_detect_us, score, infer_us, l1_ms), `link` (t_wake/start/assoc/ip/sock, wifi_ms, l2_ms), `stream_end`, `transcript`. Realtime tasks only enqueue (non-blocking). Console baud now **921600** (`sdkconfig.defaults` + both sdkconfigs) = logger default. CPU prints `null` until the first 1 s window. Field names match the console. **M1 not run.** |
| T3 gate | **code done + offline-tuned, flashed on the S3 (2026-09-29), untestable until the mic works** | Offline only (C gate on dataset clips, `docs/results.md` "Gate — OFFLINE"): 79/80 Vektora clips open the gate, median 185 ms after onset; noise keeps it open 12–14 % of the time. **No device CPU number yet.** | New `main/config.h` (all tunables, Rule 7), `main/gate.c/.h`. Stage 0: frame energy vs adaptive floor (α 0.01 down, ≤2 dB/s up), +12 dB for 2 frames. Stage 1 (only on S0 frames, from the log-mel row, no extra FFT): 300–3400 Hz mel share ≥ 0.6 and flatness ≤ 0.4 for 5 frames. Hangover 400 ms. `GATE_ENABLED 0` = old always-on behaviour (M1 baseline). `main.cc` rewritten around it: capture (core 1) runs features only when S0 passes or the gate is open; rows tagged with their frame number; infer (core 0) sleeps on a task notification while closed and, on open, **backfills** skipped rows from the ASR ring (`asr_ring_read`; ring sample index = frame sample index, priming samples are now fed too), then runs the model back to back. KWS paused while the uploader is busy (`KWS_PAUSE_WHILE_STREAMING`). LED on first, off after 500 ms once the upload is done. Arena now **trimmed to used + 1 KB** after a probe AllocateTensors (was clamped ~128 KB, used 80,656) → more heap for ring/Wi-Fi. Tools: `tools/gate_study.py`, `tools/gate_host_check.py`, `prototype/firmware/host_test/gate_main.c` (gcc at `/c/MinGW/bin/gcc`). Build clean, **0 warnings**. Now built for **esp32s3**: `vektora_kws.bin` 0xfce50 (33 % free). |
| T7 | **partial**: on-demand Wi-Fi in firmware, **compiled, NOT flashed/tested** | — | `prototype/firmware/main/wifi_manager.c/.h`, `asr_uploader.c/.h`, `vk_net_config.h`. Radio OFF while listening. Boot does one probe connect to learn BSSID/channel/DHCP lease, then stops the radio. On each wake, `do_upload()` calls `wifi_link_up()` (fast connect: cached BSSID+channel, lease reused as static IP), streams, then `wifi_link_down()`. Uploader ring is adaptive, 80 KB target / 48 KB min (1 s pre-roll + ~1.5 s connect slack). `VK_WIFI_ON_DEMAND 0` = WARM comparison mode. 2026-09-28 late: the uploader also exposes `asr_state()`, `asr_overruns()`, `asr_ring_read()` and emits `link`/`stream_end`/`transcript` telemetry. Transport is still HTTP; the UDP §8 link is not in firmware yet. **`vk_net_config.h` still has placeholder SSID/password/URL.** |
| Keyword display ("Vektora" in transcript) | **done + tested** (2026-09-28 late) | Loopback test_sender, same 5.59 s TTS clip: UDP n=3 → "Vektora one two three four five open the log file" (mode `timed`, raw "vector …"); UDP with 3 % loss + 5 % reorder + 5 s skew still `timed`; HTTP n=2 → same text (mode `fuzzy`). END→TRANSCRIPT 31–39 ms (unchanged). | New `server/wake_word.py`: timed = words inside the 1 s window ending at `kw_end_s`; fuzzy = first word or adjacent two-word merge in the first 2 s matching a known confusion or difflib ≥ 0.6 (merges ≥ 0.75); otherwise prepend. `asr.py`: `SetWords/SetPartialWords(True)`, result is now `{"text","text_raw","kw"}`. `udp_rx.py`: `kw_end_s = (HELLO.t_win_end_us − capture time of seq 0) / 1e6` (seq-0 time derived from any seq if lost). Events/JSONL/HTTP reply carry `text_raw` + `kw` mode; TRANSCRIPT packet gets the display text. Env `KEYWORD` (default Vektora). `test_sender.py --kw-end-ms`. Console Live page shows "Vosk heard: …" under the transcript (`console/dist` rebuilt). Test sessions archived in `sessions/dev-tests-20260928/`. |
| T8 console | **done** (uncommitted): Live, Latency, Evidence, Replay | — (UI only; tested with synthetic ticks and test_sender, which are **not** evidence) | `console/` React+Vite+TS, inline SVG, no chart library. Evidence card counts only wakes whose HELLO `fw` ≠ `test_sender`. Replay banner reads "Simulated" for test_sender recordings. Ledger and Before/After pages not built (first in the cut order). `public/demo_session.json` must be a **real board recording** (Live page → "Download recording"). |
| T2 (start) mic self-test (2026-09-29) | **done, flashed** | Probe result on the S3: both slots = floating noise (L: 2444/4320 words with low byte ≠ 0, 1876 zeros, AC rms −28 dBFS; R: 2471/4320, −26.9 dBFS). **The mic is not driving SD.** | `mic_probe()` in `main.cc` runs at boot, before `i2s_start()`. It reads both I²S slots for 0.32 s and logs n/zeros/low-byte/DC/RMS per slot. It auto-picks the slot that looks like an INMP441 (low byte always 0, value varies) and falls back to Kconfig `VK_I2S_RIGHT_CHANNEL` if neither passes. Boot-only, not in the hot path. Shift/gain/DC-HPF work of T2 not started. |
| docs | done | — | `docs/protocol.md` (firmware UDP section added), `docs/licences.md` (all rows verified from files on disk 2026-09-29), `docs/results.md`. `docs/ps.md` is a **placeholder**: the portal table is script-rendered and the fetch returned nothing, so paste the text from a browser. |
| **2026-09-29 late: clean repo folder** | **done, uncommitted** | — | `SIH26172-Vektora/`: see the top of this file. Dataset copied from `documents/data` (2,511 WAVs incl. `aug_*`). The root loose WAV folders were MD5-verified duplicates, so not copied twice. Manifests rewritten to repo-relative paths (splits unchanged: they don't depend on path strings). `training/config.py` defaults to `dataset/` and `training/runs/`. Its own `.gitignore` re-includes `*.keras`/`*.tflite`; it excludes build dirs, `secrets.h`, `server/models/` and `sessions/*-mic*.wav` (live room audio). |
| T5 LED + OLED | **code done, compiled, not run** | — | `ui_oled.c/h` + `ssd1306.c/h` (driver from `extras/demo_fw`, verified on SDA=8/SCL=9, plus glyphs J/Q/Z/punctuation and lowercase→uppercase). ui_task core 0, prio 1. Screens LISTEN / DETECT-STREAM / RESULT per §5.5. It only reads `g_metrics`/`g_last`, which the telemetry task publishes each tick/event, so the OLED equals the JSON. Only changed rows go over I2C. `OLED_ENABLED`, `OLED_OFF_IN_IDLE` (CPU A/B), refresh 1000/250 ms. A missing panel is logged and ignored. LED: `vk_led_blink(3)` on LINK_FAIL. **OLED on vs off CPU not measured.** |
| T4 KWS tuning | **code done, compiled, untuned** | — | `KWS_STRIDE_FRAMES 4` (40 ms). `KWS_SMOOTH_N 3`: mean of the last N P(keyword), reset when the gate closes and after a fire. `detect` event now has `score` (smoothed), `score_raw`, `smooth_n`. ESP-NN comes automatically with esp-tflite-micro on S3. **If recall at 0.3 m drops, try N=2 before touching the threshold.** |
| T7 link | **UDP §8 written, compiled, not run** | — | `asr_uploader.c`: `LINK_UDP 1` default (HTTP kept as `LINK_UDP 0`). HELLO (score, t_win_end, t_detect, t_sock, wifi_ms, fw, cfg…), 3 PINGs + 1 before END with previous RTT, backlog 2 pkts/10 ms tick (4× RT), AUDIO t_capture from a (bytes, esp_timer) anchor taken at the wake, endpoint = gate closed 700 ms (≥1.5 s after the wake) or 8 s cap, END with stats, waits ≤3 s for TRANSCRIPT → `telem_transcript` → OLED. `link` event now has `t_first_tx_us`; `stream_end` has `packets`. Ring now `RING_MS 2000` = 64 KB (was 80 KB). Secrets moved to `main/secrets.h` (gitignored) + `secrets.example.h`; server address = `VK_SERVER_IP`. **RAM:** `WIFI_DEINIT_WHEN_IDLE 1` deinitialises the Wi-Fi driver after every link-down (and after the boot probe), so idle RAM shouldn't hold it. Unmeasured: compare idle `ram_used` and `link.wifi_ms` with 0 vs 1. |
| T8 console (rest) | **done, builds** | — | New pages **Ledger** (M4 soak start/stop + FA/h, M5 TPR by distance/speaker with "Said Vektora" markers, confusable-word chips) and **Before/After** (ticks grouped by gate+config_hash, loads extra JSONL logs). Markers are events (`trial`, `soak`) with `rx_ms`, so "Download recording" keeps them. Evidence C4a/C4b now fill from the Ledger. `npm run build` OK. |
| T9 tools/analyze.py | **done, tested on synthetic + recorded files** | — | Reads serial-logger JSONL, server session JSONL, and console recordings. Reports CPU (LISTEN only), RAM idle/peak, funnel, detections or FA/h (`--negative`), L1, inference, Wi-Fi/L2, ring_ovf, server L1–L4, packet loss, Ledger TPR/FA. `--append` writes rows to `docs/results.md`, and **refuses** runs from test_sender (simulated) or laptop-mic (emulated) sessions. |

---

## Next session — start here (updated 2026-09-29)

**2026-09-29 late (newest): everything now lives in `SIH26172-Vektora/`, and the user commits only that folder** (`git add SIH26172-Vektora`; its own `.gitignore` decides what goes in). **Nothing is committed yet; the user commits it themselves.** Don't commit unless asked, and follow memory "No Claude attribution". Nothing is running. The firmware in this folder builds clean for esp32s3 in `C:\Users\User\esp\vkc` but has **not been flashed** (the board still has the older `bf4dc6f63` build from `prototype/`). The new since-last-flash parts (OLED UI, UDP link, stride/smoothing, Wi-Fi deinit) are compiled and untested. First flash: expect to debug them. Read the T4/T5/T7/T8/T9 rows above.

Paths in the older notes below (`prototype/firmware`, `C:\Users\User\esp\vkb`, `vk_net_config.h` credentials) are superseded: use `firmware/`, `vkc`, and `firmware/main/secrets.h`.

### The 5 steps planned for the 2026-09-28 late session: where they stand
| # | Step | State |
|---|---|---|
| 1 | "Vektora" shown in transcript (server) | **Done + tested** (UDP timed, HTTP fuzzy, raw kept as `text_raw`) |
| 2 | Wi-Fi credentials + server IP in `vk_net_config.h` | **Not done: needs the user.** The laptop runs a **Windows Mobile Hotspot at `192.168.137.1`** (interface "Local Area Connection* 2"). Set `VK_ASR_URL "http://192.168.137.1:8000/asr"` and ask the user for that hotspot's SSID/password (don't extract it yourself). Firewall: allow inbound TCP 8000 + UDP 5005 for the `server/.venv` python. |
| 3 | Build + flash, record L2 connect times | **Flashed on the S3 (2026-09-29).** L2 not recorded yet (placeholder creds). |
| 4 | T1 firmware telemetry → M1 | **Running on the S3**; M1 not run (mic dead) |
| 5 | T3 gate | **Flashed**; M2/M3 not run (mic dead) |

### 2026-09-29 session (first S3 flash): what happened
1. The user chose the **ESP32-S3**. The out-of-tree config (`C:/Users/User/esp/vkb/sdkconfig`) was already `esp32s3` with the recorder pins. Built and flashed to COM10.
2. The serial output was garbage at 921600: the baud had stayed at 115200. Fix: `CONFIG_ESP_CONSOLE_UART_CUSTOM=y` (build sdkconfig + `sdkconfig.defaults`). The baud is now really 921600.
3. Added a boot **mic self-test** (`mic_probe()`, T2 row). Result: the mic is not driving SD in either slot. That's a hardware problem.
4. The first S3 telemetry is valid JSON, but RAM is 320 KB > 256 KB at idle (T1 row).
5. `challenges.md` has a 2026-09-29 section. Nothing committed (the user wants one commit at the end). Files touched this session: `prototype/firmware/main/main.cc` (mic_probe), `prototype/firmware/sdkconfig.defaults`, `challenges.md`, `CLAUDE.md`.
6. The user asked whether the project is fully built. Answer given: **no**. Code for most of it exists, but nearly nothing is proven on the board, and OLED (T5), UDP link, T4, all M-runs and T10 are missing.

### BLOCKER (2026-09-29): the S3's mic sends no data
**Board decided: ESP32-S3** (COM10, QFN56 rev v0.2, 8 MB PSRAM, 16 MB flash). Firmware retargeted, built and **flashed and running** (`firmware_hash bf4dc6f63`). Boot, model (arena 80,660 B used), telemetry at 921600 baud and radio-dark idle all work.
- **Mic:** the boot `mic probe` log shows both I²S slots as floating noise (low byte ≠ 0 on ~57 % of words, identical in L and R). The INMP441 is not driving SD. The pins in firmware match the recorder wiring (SD=4, WS=5, SCK=6), so check the hardware: VDD→3V3, GND, L/R→GND, jumpers. Reflash isn't needed; just reset and read the `mic probe` lines. When it says `looks like INMP441`, go on to "After flashing" step 3.
- First S3 numbers (valid, but taken with a dead mic): idle `ram_used` **320 KB > 256 KB** (static 51.8 KB + heap 268 KB); cpu0 1.3 %, cpu1 ~20 % (noise passes Stage 0). See `challenges.md` 2026-09-29.
- Console baud needs `CONFIG_ESP_CONSOLE_UART_CUSTOM=y`, now in `sdkconfig.defaults`. Without it the baud stays 115200.

### Build / flash / log commands (PowerShell)
```
. C:\Users\User\esp\esp-idf-v5.5.5\export.ps1
cd C:\Users\User\documents\vektora\SIH26172-Vektora\firmware
copy main\secrets.example.h main\secrets.h     # once; fill in hotspot SSID/pass + VK_SERVER_IP 192.168.137.1
idf.py -B C:\Users\User\esp\vkc\build -D SDKCONFIG=C:\Users\User\esp\vkc\sdkconfig build
idf.py -B C:\Users\User\esp\vkc\build -D SDKCONFIG=C:\Users\User\esp\vkc\sdkconfig -p COM10 flash
# then log (instead of idf.py monitor; only one process can own the COM port):
..\..\server\.venv\Scripts\python.exe ..\tools\serial_logger.py --port COM10 --forward --out ..\sessions\<run>.jsonl
```
`C:\Users\User\esp\vkc` keeps paths short. It was generated from `sdkconfig.defaults` (921600 baud, S3 pins). The server venv is still the old `vektora\server\.venv` (not copied); recreate it inside the folder if you like (see README).

### After flashing, check in this order
1. The `boot` JSON line appears with firmware_hash/config_hash. The Wi-Fi log shows `boot probe ok` (or `boot probe: no link` while the creds are placeholders, which is harmless), then `radio OFF - listening radio-dark`.
2. Arena trim: the log shows arena used vs size. If boot aborts with "AllocateTensors failed on the trimmed arena", raise the +1 KB slack in `model_start()`.
3. Ticks:
   - `mic_rms` is well above −100 dBFS (the mic is alive), and `noise_floor` settles within a few seconds.
   - Quiet room: `state` = LISTEN and `funnel.s0` barely grows.
   - `cpu0/cpu1` look sane. Idle-task counters update only on context switches, so cross-check with `GATE_ENABLED 0`: core 0 should read ~100 %.
4. Say "Vektora":
   - `opens` increments, then `backfill` > 0, then a `detect` event with `l1_ms`.
   - `backfill_miss` must stay 0; if it doesn't, the ring is too small or the sample index is off.
   - Once Wi-Fi is set: `link`, `stream_end` and `transcript` events.
5. Run **M1** (`GATE_ENABLED 0`, quiet, 10 min), **M2** (gate on, quiet) and **M3** (gate on, speech at 1 m). Write each into `docs/results.md` with the firmware/config hash. Tune `GATE0_RATIO_DB` / `GATE1_*` on the device if needed; the offline values are a starting point, not truth.
6. Record L2 (`link.wifi_ms`, `l2_ms`) over ≥20 wakes. It's the first real board latency number.
   - If `ring_overrun` or `LINK_FAIL` appears, raise `VK_WIFI_CONNECT_TIMEOUT_MS` or `RING_WANT_BYTES` in `asr_uploader.c`.
   - The trimmed arena frees roughly 45 KB of heap for this. That's an estimate; the boot log shows the real figure.

### User decisions already made (don't re-ask)
- ASR = **Vosk** (the whisper mentions in `vk_net_config.h` comments are stale).
- Wi-Fi only after the wake word (radio-dark idle): implemented (T7 row).
- The transcript shows the keyword as "Vektora": implemented (keyword-display row); the raw text stays visible.

### Caveat to keep telling the user
Radio-off alone doesn't give idle CPU < 10 %; **the gate does**. It is now implemented but **unmeasured on a device**. During speech the gate opens and the model runs back to back. On the WROOM-32 that means core 0 is ~100 % busy while anyone talks, which M3 will show. The S3 would shrink that.

### Still open after this (in order; updated 2026-09-29 late)
1. **Mic wiring** (user).
2. Create `firmware/main/secrets.h`, flash the new build, and check the boot line (link udp, deinit_idle 1, oled 1), the OLED LISTEN screen and idle `ram_used`. That measures whether Wi-Fi deinit plus the 64 KB ring got idle RAM under 256 KB (it was 320 KB). Try `WIFI_DEINIT_WHEN_IDLE 0` too (Rule 6: before/after).
3. Verify detection on the S3: `infer_us`, L1, smoothing N (3 vs 2).
4. End-to-end UDP wake: `link` (L2), `stream_end` (packets, ring_ovf 0), TRANSCRIPT on the OLED; console Latency page (L3/L4). **First run of this code: expect bugs.**
5. M1–M3 (`analyze.py --append`), start the M4 soak (Ledger or `--negative`), M5 via the Ledger, M6/M8.
6. OLED on vs off CPU (`OLED_OFF_IN_IDLE`).
7. `docs/ps.md` from the portal (browser), product name, a project licence.
8. Record a real `console/public/demo_session.json`, then the T10 Vercel replay build.

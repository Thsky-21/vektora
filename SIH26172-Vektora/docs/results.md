# Results — measured values only (CLAUDE.md Rule 0.1, 0.6)

One row per measurement. Anything not in this file is unmeasured and must be shown as `—`.
Git hash `786c2bf+wip` = measured on uncommitted work on top of `786c2bf`.

## Device (ESP32)

Carried over from `docs/current_state.md` (WROOM-32, measured 2026-09-18). Protocols M1–M8 have not been run yet.
First ESP32-S3 boot numbers (2026-09-29, `firmware_hash bf4dc6f63`) were taken with a **dead mic** feeding noise, so they aren't listed as results. They are in `CLAUDE.md` (T1 row) and `challenges.md`.

| Date | Git | Config | Metric | Value | Method |
|---|---|---|---|---|---|
| 2026-09-18 | pre-786c2bf | WROOM-32, Run 3 int8, plain-C TFLM | inference time | 564 ms | esp_timer around Invoke() |
| 2026-09-18 | pre-786c2bf | WROOM-32 | feature extraction | 4.10 ms / 10 ms frame | esp_timer |
| 2026-09-18 | pre-786c2bf | WROOM-32 | tensor arena used | 80,656 B | arena_used_bytes() |

## Server (laptop, i5-1235U, Windows 11, Python 3.12.6, vosk-model-small-en-in-0.4)

All over **loopback (127.0.0.1)**, so there is no Wi-Fi in these numbers. Test clip: 5.59 s Windows TTS clip, "Vektora. One two three four five. Open the log file." (`server/test_sender.py`).

| Date | Git | Config | Metric | Value | Method |
|---|---|---|---|---|---|
| 2026-09-28 | 786c2bf+wip | startup | Vosk model load | 240–532 ms | `asr.load()`, perf_counter (2 cold starts) |
| 2026-09-28 | 786c2bf+wip | startup | KaldiRecognizer build | 132–144 ms | same → reason for the pre-built recognizer pool |
| 2026-09-28 | 786c2bf+wip | startup | first decode (cold) | 2,788 ms (first boot), 268 ms (warm file cache) | same → reason for the startup warm-up |
| 2026-09-28 | 786c2bf+wip | UDP, chunk 100 ms, n=5 | END → TRANSCRIPT at sender | median 30.5 ms, p90 32.5 ms | test_sender.py, perf_counter |
| 2026-09-28 | 786c2bf+wip | HTTP /asr, chunk 100 ms, n=3 | last byte → HTTP response at sender | median 41.3 ms | test_sender.py --http |
| 2026-09-28 | 786c2bf+wip | UDP, chunk 100 ms, n=5 | server decode lag, audio in → partial out | median of per-trial medians 42.7 ms, p90 60.8–70.4 ms | `asr_final.decode_lag_*` |
| 2026-09-28 | 786c2bf+wip | UDP, chunk 40 ms, n=5 | same | median of per-trial medians 42.5 ms, p90 61.1–69.5 ms | same |
| 2026-09-28 | 786c2bf+wip | UDP, chunk 20 ms, n=5 | same | median of per-trial medians 50.4 ms, p90 59.1–105.4 ms | same → 20 ms rejected |
| 2026-09-28 | 786c2bf+wip | UDP, chunk 40 ms, n=5 | Vosk real-time factor | 0.149–0.153 | busy time ÷ audio time |
| 2026-09-28 | 786c2bf+wip | UDP, 3% loss + 5% reorder, 5 s clock skew | clock-sync recovery | offset −4,999,975 µs vs true −5,000,000 (±124 µs bound), min RTT 248 µs | `clock_sync` events |
| 2026-09-28 | 786c2bf+wip | same | packet loss handling | 14 lost / 265 received, transcript still has "one two three four five open the log file" (mis-hears "two" in some trials) | `session_end` |

**Chosen:** `ASR_CHUNK_MS=40`. It has the same decode lag as 100 ms, but the chunk-fill wait drops from ≤100 ms to ≤40 ms.

**Note:** Vosk transcribes the keyword "Vektora" as "vector" or "regular", because the word isn't in its vocabulary. Server-side keyword re-verification (stretch goal S1) would need a custom grammar entry, not plain decoding.

## Gate — OFFLINE (laptop, dataset clips; NOT device measurements)

Chooses the §5.3 gate constants in `prototype/firmware/main/config.h`. Stage 1 from `tools/gate_study.py` (Python features); final numbers from `tools/gate_host_check.py` running the **firmware's own `gate.c` + `feature_extract.c`** compiled with MinGW gcc (`prototype/firmware/host_test/gate_main.c`). Device idle CPU is still unmeasured.

| Date | Git | Config | Metric | Value | Method |
|---|---|---|---|---|---|
| 2026-09-28 | 786c2bf+wip | S1 ratio≥0.6, flat≤0.4, 5 frames (Python, per-clip floor) | clips that open the gate | positive_akash 0.99, positive 0.98, hard_negative 0.97, background_negative 0.19 (0.92 with S0 only) | gate_study.py, ≤150 clips/set |
| 2026-09-28 | 786c2bf+wip | first try: floor α 0.3, S0 +9 dB, S1 5 frames | Vektora words opening gate / noise open time | 79/80; noise 22.2 % (scaled ambience 20.3 %) | gate_host_check.py (C gate) |
| 2026-09-28 | 786c2bf+wip | **chosen: floor α 0.01, S0 +12 dB, S1 ratio≥0.6 flat≤0.4 5 frames, hangover 400 ms** | Vektora words opening gate | 79/80, open median 185 ms / p90 487 ms after word onset (backfill covers this) | gate_host_check.py, 80 clips after 6 s ambience at −26 dB |
| 2026-09-28 | 786c2bf+wip | same | gate open time on noise | 13.7 % of 101 s full-level background_negative; 11.8 % on −26 dB ambience | gate_host_check.py |

## 2026-09-29 late — clean tree (`SIH26172-Vektora/`), build + loopback only

Nothing below ran on the board: the S3 microphone still sends no data.

| Date | Git | Config | Metric | Value | Method |
|---|---|---|---|---|---|
| 2026-09-29 | 786c2bf+wip | firmware, esp32s3, LINK_UDP 1, OLED on, WIFI_DEINIT_WHEN_IDLE 1 | static DRAM (.data + .bss) | 20,004 + 33,224 = 53,228 B | `idf.py size` (build-time linker figure, not runtime RAM) |
| 2026-09-29 | 786c2bf+wip | same | app image | 935,127 B (39 % of the app partition free) | `idf.py build` |
| 2026-09-29 | 786c2bf+wip | server from the new tree, loopback, Vosk small-en-in-0.4 | END → TRANSCRIPT (UDP) | median 39.2 ms, n=3; full transcript every trial | `test_sender.py --trials 3` (simulated; sessions in `sessions/dev-tests-20260929/`) |
| 2026-09-29 | 786c2bf+wip | same | last byte → HTTP response | 56.1 ms (server-side 41.6 ms), n=1 | `test_sender.py --http` |

Note: an earlier run the same evening, made while the firmware was compiling on all cores, gave 1 of 2 UDP trials with no transcript and an 8.9 s HTTP decode. That was CPU starvation on the laptop, not a server fault. Those sessions (22:54) are also in `dev-tests-20260929/`. Don't benchmark the server while building.

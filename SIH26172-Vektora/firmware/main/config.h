// config.h — every runtime tunable in one place (CLAUDE.md Rule 7).
// Board wiring stays in Kconfig/sdkconfig.defaults; network in vk_net_config.h.
//
// Changing any value here changes config_hash in the boot line, so every
// session file records which settings produced it.

#pragma once

// --- Two-stage gate (CLAUDE.md §5.3): the idle-CPU mechanism ----------------
// 0 = features + model run on every frame, back to back (the M1 baseline).
#define GATE_ENABLED 1

// Stage 0, energy, every 10 ms frame, a few µs. Adaptive noise floor: falls
// toward quieter frames (tau ~1 s), rises at most 2 dB/s, so speech doesn't
// raise it. Floor alpha 0.01 + 12 dB (not the 0.3 + 9 dB first tried): with a
// fast-falling floor the floor hugs the quietest frames of fluctuating noise,
// and Stage 0 passed 75% of noise frames. tools/gate_host_check.py on the C
// code (2026-09-28): 79/80 Vektora clips open the gate, median 185 ms after
// onset (backfill covers that); scaled-down background clips keep it open 12%
// of the time (was 20%), full-level ones 14% (was 22%). Offline, not device.
#define GATE0_RATIO_DB 12.0f             // frame must be this far above the floor
#define GATE0_FRAMES 2                   // ... for this many consecutive frames
#define GATE0_ABS_MIN_DB (-80.0f)        // and above this (dBFS): ignores digital silence
#define GATE_FLOOR_DOWN_ALPHA 0.01f      // per frame, toward a quieter frame
#define GATE_FLOOR_UP_DB 0.02f           // per frame max rise = 2 dB/s
#define GATE_FLOOR_INIT_DB (-60.0f)

// Stage 1, voice, only on Stage-0 frames, from the log-mel row the model
// needs anyway (no extra FFT). Values chosen offline by tools/gate_study.py
// on the training clips (2026-09-28): speech clips open 97-99%, 1 s noise
// clips open 19% (vs 92% with Stage 0 alone). Re-tune on the device.
#define GATE1_BAND_LO_HZ 300
#define GATE1_BAND_HI_HZ 3400
#define GATE1_BAND_RATIO 0.6f            // speech-band share of mel energy >=
#define GATE1_FLATNESS 0.4f              // spectral flatness <= (noise is flat)
#define GATE1_FRAMES 5                   // consecutive voiced frames to open

// Open -> stay open this long after the last voiced frame, so the keyword's
// quiet tail ("-ra") isn't chopped.
#define GATE_HANGOVER_MS 400

// --- KWS ----------------------------------------------------------------------
// Pause detection while the uploader is linking/streaming (CLAUDE.md §4):
// saves core 0 for WiFi and prevents a second wake mid-command.
#define KWS_PAUSE_WHILE_STREAMING 1

// Inference stride in 10 ms feature frames (CLAUDE.md §5.4): 4 = one decision
// per 40 ms while the gate is open. If Invoke() takes longer than the stride,
// the model simply runs back to back.
#define KWS_STRIDE_FRAMES 4
// Posterior smoothing: fire on the mean of the last N P(keyword) values (fewer
// right after the gate opens). 1 = no smoothing, i.e. exactly the offline test.
// UNTUNED on the device: if recall drops at 0.3 m, try 2 before touching the
// threshold.
#define KWS_SMOOTH_N 3

// --- Mic front end (CLAUDE.md §5.1) -------------------------------------------
// One-pole DC high-pass before the ring. OFF by default: the training clips
// were recorded without it, and Rule 3 (feature parity) wins. The INMP441
// already has its own ~60 Hz high-pass, so the DC it leaves is small.
#define MIC_DC_HPF 0

// --- Link (CLAUDE.md §5.6, §8) --------------------------------------------------
// 1 = UDP "VAD1" protocol (HELLO, backlog burst, PING clock sync, END,
//     TRANSCRIPT back). 0 = HTTP POST /asr (the older, fixed-length upload).
#define LINK_UDP 1
#define PREROLL_MS 1000                  // audio from BEFORE the decision (keyword + margin)
// Ring = pre-roll + WiFi connect bridge + margin (§5.2). 2 s PCM16 = 64 KB.
// Resize after L2 is measured: RING_MS >= PREROLL_MS + p95(connect) + 300.
#define RING_MS 2000
#define STREAM_MIN_MS 1500               // never end sooner than this after the wake
#define ENDPOINT_SILENCE_MS 700          // gate closed this long -> END
#define STREAM_MAX_MS 8000               // hard cap per wake
#define BACKLOG_PKTS_PER_TICK 2          // 2 x 20 ms per 10 ms tick = 4x real time
#define TRANSCRIPT_WAIT_MS 3000          // after END, wait this long for TRANSCRIPT
#define LINK_FAIL_BLINKS 3               // LED blinks on LINK_FAIL (§5.5)

// WiFi driver while listening. 1 = esp_wifi_deinit() after every link-down,
// so the driver's buffers are not held in idle RAM (C1); each wake then pays
// esp_wifi_init() inside L2. 0 = driver stays initialised (radio still OFF).
// Pick with data: measure idle ram_used AND link.wifi_ms both ways.
#define WIFI_DEINIT_WHEN_IDLE 1

// --- OLED (CLAUDE.md §5.5): SSD1306 128x64 on I2C, ui_task on core 0 --------------
#define OLED_ENABLED 1
#define OLED_SDA_GPIO 8
#define OLED_SCL_GPIO 9
#define OLED_ADDR 0x3C
#define OLED_IDLE_REFRESH_MS 1000        // LISTEN screen refresh
#define OLED_BUSY_REFRESH_MS 250         // DETECT / STREAM screen refresh
#define OLED_RESULT_HOLD_MS 8000         // show the transcript this long
#define OLED_OFF_IN_IDLE 0               // 1 = panel dark while listening (CPU A/B test)

// --- Telemetry (CLAUDE.md §5.7) ------------------------------------------------
#define TELEM_TICK_MS 500                // JSON tick at 2 Hz
#define TELEM_CPU_WINDOW_TICKS 2         // CPU % over 2 ticks = 1 s windows

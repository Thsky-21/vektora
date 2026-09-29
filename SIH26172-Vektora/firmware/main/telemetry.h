// telemetry.h — newline JSON over the USB serial console (CLAUDE.md §5.7).
//
// The radio is off while listening, so evidence leaves over USB. One task on
// core 0 prints everything: a 2 Hz "tick" (CPU per core, RAM, gate funnel,
// mic level, score) plus events as they happen (boot, detect, link,
// stream_end, transcript). Realtime code never prints: it drops a small
// record into a queue (non-blocking; a full queue loses the event and counts
// it) and carries on. tools/serial_logger.py stores and forwards the lines.
//
// The same task also publishes g_metrics / g_last for the OLED (ui_oled.c),
// so the screen shows exactly the numbers the JSON carries.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#ifdef __cplusplus
extern "C" {
#endif

// Live values written by the pipeline, read by the tick. Single writer each;
// torn reads cost one tick of display, never a decision.
typedef struct {
    volatile float score;           // last P(keyword), raw
    volatile float score_max;       // max since last tick (reset by the tick)
    volatile uint32_t infer, det;   // since boot
    volatile uint32_t backfill_rows, backfill_miss;  // rows recomputed on gate open
    volatile int64_t infer_us_last;
    volatile bool speech;           // gate open
    size_t arena_used, arena_size, model_bytes;
} vk_live_t;

extern vk_live_t g_live;

// Last 1 s window, written by the telemetry task each tick. NaN = not yet.
typedef struct {
    volatile float cpu0, cpu1, cpu_avg, dscnn_duty;   // %
    volatile uint32_t ram_used, ram_peak, ram_static; // bytes, internal DRAM
    volatile float mic_rms_db, noise_floor_db;
    volatile uint32_t frames, s0, s1;                 // gate funnel since boot
} vk_metrics_t;

extern vk_metrics_t g_metrics;

// Last events, written by the telemetry task as it prints them. `gen` bumps
// on every update so the UI can tell a new wake from an old one.
typedef struct {
    volatile uint32_t gen;
    // detect
    volatile float score;
    volatile float infer_ms, l1_ms;
    volatile int64_t t_detect_us;
    // link (NaN = not reached)
    volatile float wifi_ms, l2_ms;
    volatile bool link_done, link_ok;
    // stream end
    volatile bool stream_done, stream_ok;
    volatile uint32_t bytes, packets, audio_ms;
    volatile int64_t t_end_us;
    char reason[24];
    // transcript (server's display text, <= 60 bytes)
    char text[64];
} vk_last_t;

extern vk_last_t g_last;

// Mic level accumulator, fed by capture_task once per 10 ms hop.
void telem_mic(int peak, uint64_t sumsq, int n);

// Start the telemetry task (core 0). Handles are for per-task CPU shares.
void telem_start(TaskHandle_t capture, TaskHandle_t infer);

// Events. All non-blocking, safe from any task.
void telem_detect(int64_t t_win_end_us, int64_t t_detect_us, float score, float score_raw, int64_t infer_us);
void telem_link(int64_t t_wake_us, int64_t t_wifi_start_us, int64_t t_assoc_us, int64_t t_ip_us,
                int64_t t_sock_us, int64_t t_first_tx_us, bool ok);
void telem_stream_end(uint32_t bytes, uint32_t packets, uint32_t audio_ms, uint32_t total_ms, bool ok,
                      const char *reason);
void telem_transcript(const char *text);

// LISTEN / SPEECH / LINKING / STREAMING, as printed in the tick.
const char *telem_state_name(void);

// Copy g_last.text / g_last.reason consistently (either pointer may be NULL).
void telem_copy_last(char *text, size_t tn, char *reason, size_t rn);

// Identity, for the boot line and the UDP HELLO: ELF sha256 prefix and the
// FNV-1a hash of every tunable in config.h (+ model sha, mic shift).
const char *telem_fw_hash(void);
uint32_t telem_config_hash(void);

#ifdef __cplusplus
}
#endif

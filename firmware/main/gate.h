// gate.h — two-stage speech gate (CLAUDE.md §5.3).
//
// Why: features cost 4.1 ms per 10 ms frame and one inference 564 ms on this
// chip, so running them continuously keeps both cores busy with nobody
// talking. The gate lets them run only while there is voice-like sound:
//
//   Stage 0  energy  every frame, µs      frame >= noise floor + 9 dB
//   Stage 1  voice   Stage-0 frames only  speech-band share high, spectrum not flat
//   open -> features every frame + the model; closes GATE_HANGOVER_MS after
//   the last voiced frame.
//
// Called only from capture_task (core 1), so no locking inside; the counters
// are read by telemetry as plain 32/64-bit loads (a torn read of a counter
// costs one wrong tick, never a wrong decision).

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    // funnel, since boot (CLAUDE.md §5.3 "funnel counters")
    volatile uint32_t frames, s0, s1, opens;
    volatile float floor_db;     // current noise floor, dBFS
    volatile float last_db;      // last frame energy, dBFS
} gate_stats_t;

extern gate_stats_t g_gate;

void gate_init(void);

// Stage 0 on one 10 ms hop. energy_db = frame energy in dBFS. Returns true
// when this frame passes (after GATE0_FRAMES consecutive).
bool gate_stage0(float energy_db);

// Stage 1 on the log-mel row of a Stage-0 frame. Returns true when the gate
// (re)opened on this frame, i.e. after GATE1_FRAMES consecutive voiced frames.
// Call gate_stage1_reset() on frames that failed Stage 0.
bool gate_stage1(const float *mel_db, int64_t frame);
void gate_stage1_reset(void);

// True while frame <= last voiced frame + hangover.
bool gate_is_open(int64_t frame);

#ifdef __cplusplus
}
#endif

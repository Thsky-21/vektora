// feature_extract.h — log-mel features on the device.
//
// PARITY CONTRACT with training/features.py (CLAUDE.md §6 risk 1).
// All constants come from feature_tables.h, which export_c.py generates from
// the Python objects training used. Plain C99 with no ESP-IDF dependencies,
// so the SAME file is compiled on the laptop by parity_check.py and compared
// value-by-value against Python.
#pragma once
#include <stdint.h>

#include "feature_tables.h"

#ifdef __cplusplus
extern "C" {
#endif

// Build the sin/cos tables. Call once before anything else.
void fe_init(void);

// One 25 ms frame: FE_WIN int16 samples -> FE_N_MELS log-mel values in dB
// (un-normalised). Same as one row of features.logmel().
void fe_frame_logmel(const int16_t *samples, float *out_db);

// A whole second: FE_CLIP_SAMPLES samples -> FE_N_FRAMES x FE_N_MELS log-mel,
// row = time. Used by the parity test; the firmware streams frame by frame.
void fe_clip_logmel(const int16_t *pcm, float *out_db);

// Per-band normalisation, in place, for n_frames rows: F = (L - mean) / std.
void fe_normalize(float *feat, int n_frames);

// Normalised feature -> model int8 input: clamp(round(F / scale) + zp).
int8_t fe_quantize(float f);

#ifdef __cplusplus
}
#endif

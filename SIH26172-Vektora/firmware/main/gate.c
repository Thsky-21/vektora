// gate.c — see gate.h.

#include "gate.h"

#include <math.h>

#include "config.h"
#include "feature_tables.h"

gate_stats_t g_gate;

static bool s_in_band[FE_N_MELS];     // mel band centred in 300-3400 Hz
static int s0_run, s1_run;
// Last frame the gate is open for. 32-bit so core 0 reads it in one load
// (frames are 10 ms: wraps after 248 days).
static volatile int32_t s_open_until = -1;

#define HANGOVER_FRAMES (GATE_HANGOVER_MS * FE_SAMPLE_RATE / FE_HOP / 1000)

void gate_init(void)
{
    // Band centre = the FFT bin with the largest weight in each mel filter,
    // read from the exported table so it can't drift from the features.
    const float *w = FE_MEL_W;
    for (int b = 0; b < FE_N_MELS; b++) {
        int best = 0;
        for (int j = 1; j < FE_MEL_COUNT[b]; j++)
            if (w[j] > w[best]) best = j;
        float hz = (float)(FE_MEL_START[b] + best) * FE_SAMPLE_RATE / FE_N_FFT;
        s_in_band[b] = hz >= GATE1_BAND_LO_HZ && hz <= GATE1_BAND_HI_HZ;
        w += FE_MEL_COUNT[b];
    }
    g_gate.floor_db = GATE_FLOOR_INIT_DB;
}

bool gate_stage0(float e_db)
{
    g_gate.frames++;
    g_gate.last_db = e_db;
    float fl = g_gate.floor_db;
    // falls toward quieter frames, rises at most GATE_FLOOR_UP_DB per frame:
    // a noise floor, not an average, so speech barely moves it
    if (e_db < fl) fl += GATE_FLOOR_DOWN_ALPHA * (e_db - fl);
    else fl += fminf(e_db - fl, GATE_FLOOR_UP_DB);
    if (fl < GATE0_ABS_MIN_DB) fl = GATE0_ABS_MIN_DB;
    g_gate.floor_db = fl;

    bool loud = e_db >= fl + GATE0_RATIO_DB && e_db >= GATE0_ABS_MIN_DB;
    s0_run = loud ? s0_run + 1 : 0;
    bool pass = s0_run >= GATE0_FRAMES;
    if (pass) g_gate.s0++;
    return pass;
}

void gate_stage1_reset(void)
{
    s1_run = 0;
}

bool gate_stage1(const float *db, int64_t frame)
{
    // flatness = geometric mean / arithmetic mean of linear mel energy.
    // The geometric mean in dB is just the mean dB: one exp instead of 40 logs.
    float sum = 0.0f, band = 0.0f, mean_db = 0.0f;
    for (int b = 0; b < FE_N_MELS; b++) {
        float lin = expf(db[b] * 0.2302585f);   // 10^(dB/10)
        sum += lin;
        if (s_in_band[b]) band += lin;
        mean_db += db[b];
    }
    mean_db /= FE_N_MELS;
    float mean_lin = sum / FE_N_MELS;
    float flat = mean_lin > 0 ? expf(mean_db * 0.2302585f) / mean_lin : 1.0f;
    float ratio = sum > 0 ? band / sum : 0.0f;

    bool voiced = ratio >= GATE1_BAND_RATIO && flat <= GATE1_FLATNESS;
    if (!voiced) {
        s1_run = 0;
        return false;
    }
    g_gate.s1++;
    if (++s1_run < GATE1_FRAMES) return false;
    bool was_open = (int32_t)frame <= s_open_until;
    s_open_until = (int32_t)frame + HANGOVER_FRAMES;
    if (!was_open) g_gate.opens++;
    return !was_open;
}

bool gate_is_open(int64_t frame)
{
    return (int32_t)frame <= s_open_until;
}

// feature_extract.c — see feature_extract.h. Each step names the line of
// features.py it mirrors.
#include "feature_extract.h"

#include <math.h>

// Why a table-driven DFT and not a library FFT: FE_N_FFT = 400 is not a power
// of two, so the common radix-2 FFTs (including esp-dsp's) can't be used
// without changing the feature definition. A direct DFT over only the bins the
// mel filters actually use is simple, exactly equal to np.fft.rfft, and costs
// about 76k float multiplies per 10 ms frame, which is cheap for the ESP32.
// Replace it with a faster FFT only after measuring, and re-run the parity test.
static float s_cos[FE_N_FFT];
static float s_sin[FE_N_FFT];
static int s_bin_lo, s_bin_hi;                  // bins the mel filters touch

void fe_init(void)
{
    for (int i = 0; i < FE_N_FFT; i++) {
        double a = 2.0 * M_PI * i / FE_N_FFT;
        s_cos[i] = (float)cos(a);
        s_sin[i] = (float)sin(a);
    }
    s_bin_lo = FE_N_BINS;
    s_bin_hi = 0;
    for (int b = 0; b < FE_N_MELS; b++) {
        int lo = FE_MEL_START[b], hi = lo + FE_MEL_COUNT[b] - 1;
        if (lo < s_bin_lo) s_bin_lo = lo;
        if (hi > s_bin_hi) s_bin_hi = hi;
    }
}

void fe_frame_logmel(const int16_t *samples, float *out_db)
{
    float xw[FE_WIN];
    float power[FE_N_BINS];

    // x = s / 32768.0 ; frames * WINDOW
    for (int n = 0; n < FE_WIN; n++)
        xw[n] = ((float)samples[n] / 32768.0f) * FE_WINDOW[n];

    // power = |rfft(frame, n=400)|^2  (only bins the mel filters use)
    for (int k = s_bin_lo; k <= s_bin_hi; k++) {
        float re = 0.0f, im = 0.0f;
        int idx = 0;                            // (k * n) mod N, stepped
        for (int n = 0; n < FE_WIN; n++) {
            re += xw[n] * s_cos[idx];
            im -= xw[n] * s_sin[idx];
            idx += k;
            if (idx >= FE_N_FFT) idx -= FE_N_FFT;
        }
        power[k] = re * re + im * im;
    }

    // mel = power @ MEL_FB.T ; L = 10 * log10(max(mel, 1e-10))
    const float *w = FE_MEL_W;
    for (int b = 0; b < FE_N_MELS; b++) {
        float m = 0.0f;
        int lo = FE_MEL_START[b];
        for (int j = 0; j < FE_MEL_COUNT[b]; j++)
            m += w[j] * power[lo + j];
        w += FE_MEL_COUNT[b];
        out_db[b] = 10.0f * log10f(m > FE_LOG_FLOOR ? m : FE_LOG_FLOOR);
    }
}

void fe_clip_logmel(const int16_t *pcm, float *out_db)
{
    // frame i = x[i*160 : i*160 + 400], i = 0..97, no centre padding
    for (int i = 0; i < FE_N_FRAMES; i++)
        fe_frame_logmel(pcm + i * FE_HOP, out_db + i * FE_N_MELS);
}

void fe_normalize(float *feat, int n_frames)
{
    for (int t = 0; t < n_frames; t++)
        for (int b = 0; b < FE_N_MELS; b++) {
            float *v = &feat[t * FE_N_MELS + b];
            *v = (*v - FE_NORM_MEAN[b]) / FE_NORM_STD[b];
        }
}

int8_t fe_quantize(float f)
{
    // lrintf rounds half-to-even in the default FP mode, like np.round.
    long q = lrintf(f / KWS_INPUT_SCALE) + KWS_INPUT_ZERO_POINT;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    return (int8_t)q;
}

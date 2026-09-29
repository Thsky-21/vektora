// parity_main.c — laptop-side harness for feature_extract.c (not built for the ESP32).
//
// usage: parity_main <in.i16> <out.bin>
//   in.i16   N x 16000 little-endian int16 samples (N clips back to back)
//   out.bin  for each clip: 98x40 float32 raw dB, 98x40 float32 normalised,
//            98x40 int8 quantized model input
// Driven by training/parity_check.py.
#include <stdio.h>
#include <stdlib.h>

#include "feature_extract.h"

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s in.i16 out.bin\n", argv[0]);
        return 2;
    }
    FILE *fi = fopen(argv[1], "rb");
    FILE *fo = fopen(argv[2], "wb");
    if (!fi || !fo) {
        perror("open");
        return 1;
    }
    fe_init();

    const int n = FE_N_FRAMES * FE_N_MELS;
    int16_t pcm[FE_CLIP_SAMPLES];
    float raw[FE_N_FRAMES * FE_N_MELS];
    float nrm[FE_N_FRAMES * FE_N_MELS];
    int8_t q[FE_N_FRAMES * FE_N_MELS];
    int clips = 0;

    while (fread(pcm, sizeof(int16_t), FE_CLIP_SAMPLES, fi) == FE_CLIP_SAMPLES) {
        fe_clip_logmel(pcm, raw);
        for (int i = 0; i < n; i++) nrm[i] = raw[i];
        fe_normalize(nrm, FE_N_FRAMES);
        for (int i = 0; i < n; i++) q[i] = fe_quantize(nrm[i]);
        fwrite(raw, sizeof(float), n, fo);
        fwrite(nrm, sizeof(float), n, fo);
        fwrite(q, sizeof(int8_t), n, fo);
        clips++;
    }
    fclose(fi);
    fclose(fo);
    fprintf(stderr, "%d clips\n", clips);
    return 0;
}

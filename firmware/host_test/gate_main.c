// gate_main.c — laptop-side harness for gate.c (not built for the ESP32).
//
// usage: gate_main <in.i16> <out.csv>
//   in.i16   one long little-endian int16 16 kHz stream
//   out.csv  per 10 ms frame: frame,energy_db,floor_db,s0,open
// Runs exactly the capture_task gate logic (Stage 0 every hop, features +
// Stage 1 only on Stage-0 frames or while open). Driven by tools/gate_host_check.py.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "feature_extract.h"
#include "gate.h"

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s in.i16 out.csv\n", argv[0]);
        return 2;
    }
    FILE *fi = fopen(argv[1], "rb");
    FILE *fo = fopen(argv[2], "w");
    if (!fi || !fo) {
        perror("open");
        return 1;
    }
    fe_init();
    gate_init();

    int16_t window[FE_WIN], hop[FE_HOP];
    float db[FE_N_MELS];
    if (fread(window, sizeof(int16_t), FE_WIN - FE_HOP, fi) != FE_WIN - FE_HOP) return 1;
    long feat_frames = 0;
    fprintf(fo, "frame,energy_db,floor_db,s0,open\n");
    for (int f = 0; fread(hop, sizeof(int16_t), FE_HOP, fi) == FE_HOP; f++) {
        memmove(window, window + FE_HOP, (FE_WIN - FE_HOP) * sizeof(int16_t));
        memcpy(window + FE_WIN - FE_HOP, hop, sizeof(hop));
        unsigned long long sq = 0;
        for (int i = 0; i < FE_HOP; i++) sq += (unsigned long long)((int)hop[i] * hop[i]);
        float e_db = 10.0f * log10f((float)sq / FE_HOP / (32768.0f * 32768.0f) + 1e-12f);
        int s0 = gate_stage0(e_db);
        if (s0 || gate_is_open(f)) {
            fe_frame_logmel(window, db);
            feat_frames++;
            if (s0) gate_stage1(db, f);
            else gate_stage1_reset();
        } else {
            gate_stage1_reset();
        }
        fprintf(fo, "%d,%.2f,%.2f,%d,%d\n", f, e_db, g_gate.floor_db, s0, gate_is_open(f));
    }
    fprintf(stderr, "frames %lu s0 %lu s1 %lu opens %lu feature_frames %ld\n",
            (unsigned long)g_gate.frames, (unsigned long)g_gate.s0, (unsigned long)g_gate.s1,
            (unsigned long)g_gate.opens, feat_frames);
    fclose(fi);
    fclose(fo);
    return 0;
}

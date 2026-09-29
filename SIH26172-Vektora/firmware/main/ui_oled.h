// ui_oled.h — the self-auditing screen (CLAUDE.md §5.5, differentiator 2).
//
// ui_task, core 0, priority 1 (lowest real task): never competes with the
// audio path on core 1. It only reads numbers the telemetry task already
// computed (g_metrics / g_last), so the OLED and the JSON always agree.
// Three screens: LISTEN (compliance: CPU vs 10 %, RAM vs 256 KB, gate funnel),
// DETECT/STREAM (score, infer, L1, WiFi, L2, audio sent), RESULT (the
// server's transcript). Anything not measured yet shows as "-".
// A missing/unwired display is logged and ignored.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void ui_start(void);

#ifdef __cplusplus
}
#endif

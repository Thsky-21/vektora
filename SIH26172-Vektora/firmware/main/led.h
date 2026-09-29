// led.h — the wake LED (CLAUDE.md §5.5). Implemented in main.cc.
//
// vk_led_set() is the FIRST thing the detector does on a wake (before any
// logging, OLED or network). vk_led_blink() is for LINK_FAIL, from the
// uploader task (blocking, ~100 ms per blink, never call it from core 1).

#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void vk_led_set(bool on);
void vk_led_blink(int n);

#ifdef __cplusplus
}
#endif

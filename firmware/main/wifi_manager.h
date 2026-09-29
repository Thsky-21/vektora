// wifi_manager.h — station WiFi, radio OFF while listening, ON only after a wake.
//
// Why on-demand (changed 2026-09-28, was "connect once at boot"): an
// associated radio wakes every DTIM beacon, runs the WiFi/lwIP tasks and
// injects switching noise into the INMP441 centimetres away. Idle should cost
// nothing but listening, and the PS asks for idle CPU < 10%. So while
// listening the radio is stopped (esp_wifi_stop): no RF, no WiFi tasks busy.
//
// The price is connect time after the keyword. Kept small by:
//   * a one-off probe connect at boot (before listening starts) that learns
//     the AP's BSSID + channel and the DHCP lease;
//   * every wake then connects with that BSSID/channel (no scan) and the
//     learned IP as a static IP (no DHCP) — typically a few hundred ms;
//   * the uploader's ring buffer keeps recording through the connect, so no
//     audio is lost (see asr_uploader.c).
//
// VK_WIFI_ON_DEMAND=0 in vk_net_config.h restores the old always-connected
// behaviour, only to MEASURE the trade-off (CLAUDE.md §5.6 "WARM").
//
// Nothing here ever aborts the boot. A missing AP degrades to "LED only".

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int64_t t_start_us;   // esp_wifi_start() called
    int64_t t_assoc_us;   // associated with the AP (0 if never)
    int64_t t_ip_us;      // IP usable (0 if never)
} wifi_timing_t;

// Initialise NVS, netif, event loop and the WiFi driver. With on-demand WiFi
// it does one probe connect (bounded by VK_WIFI_PROBE_TIMEOUT_MS), caches the
// AP and lease, then stops the radio again. Call before listening starts.
esp_err_t wifi_init(void);

// Radio on + connect; blocks until an IP is usable or timeout_ms passes.
// Returns true when up. Fills *t if non-NULL. Call from a non-realtime task.
bool wifi_link_up(uint32_t timeout_ms, wifi_timing_t *t);

// Disconnect and stop the radio (no-op when VK_WIFI_ON_DEMAND=0).
void wifi_link_down(void);

// True while an IP is usable.
bool wifi_is_up(void);

#ifdef __cplusplus
}
#endif

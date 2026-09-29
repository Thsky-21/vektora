// vk_net_config.h — network settings. Credentials and the server address come
// from secrets.h (gitignored, CLAUDE.md Rule 7); copy secrets.example.h to
// secrets.h and edit it.

#pragma once

#if __has_include("secrets.h")
#include "secrets.h"
#else
#pragma message "firmware/main/secrets.h missing: using placeholder WiFi credentials (every wake will LINK_FAIL)"
#include "secrets.example.h"
#endif

// --- WiFi policy (wifi_manager.h) -------------------------------------------
// 1 = ON_DEMAND: radio OFF while listening, connect only after the wake word.
// 0 = WARM: stay associated with modem sleep (only to MEASURE the trade-off).
#define VK_WIFI_ON_DEMAND 1
// Boot-time probe connect: learns BSSID/channel/lease so each wake skips the
// scan and DHCP. Bounded; a missing AP only delays boot by this much.
#define VK_WIFI_PROBE_TIMEOUT_MS 8000
// Per-wake connect budget (CLAUDE.md §4: LINKING timeout 3 s -> LINK_FAIL).
// The ring covers RING_MS - PREROLL_MS of it without losing audio.
#define VK_WIFI_CONNECT_TIMEOUT_MS 3000
// Reuse the boot DHCP lease as a static IP on later wakes (skips DHCP).
// Safe on a dedicated demo hotspot; set 0 on a shared network.
#define VK_WIFI_REUSE_LEASE 1

// --- Server (server/app.py) ---------------------------------------------------
#define VK_SERVER_UDP_PORT 5005                       // §8 protocol (LINK_UDP 1)
#define VK_ASR_URL "http://" VK_SERVER_IP ":8000/asr"  // HTTP fallback (LINK_UDP 0)

// Vosk on the laptop decodes while the audio arrives; the reply is ready tens
// of ms after the last byte (docs/results.md). The timeout is a guard rail.
#define VK_ASR_TIMEOUT_MS 15000

// secrets.example.h — template for secrets.h (CLAUDE.md Rule 7).
//
// Copy this file to secrets.h (same folder) and fill in your values.
// secrets.h is gitignored; this template is committed. Without secrets.h the
// build uses these placeholders: the board boots and detects, but every wake
// ends in LINK_FAIL (LED blinks, no transcript).

#pragma once

// 2.4 GHz AP on a fixed channel (laptop hotspot or a spare router).
#define VK_WIFI_SSID "YOUR_SSID_HERE"
#define VK_WIFI_PASS "YOUR_PASSWORD_HERE"

// The laptop running server/app.py, as seen from the ESP32 (not 127.0.0.1).
// A Windows Mobile Hotspot host is usually 192.168.137.1.
#define VK_SERVER_IP "192.168.137.1"

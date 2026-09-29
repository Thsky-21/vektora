// wifi_manager.c — see wifi_manager.h for the why.

#include "wifi_manager.h"

#include <string.h>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "nvs_flash.h"

#include "config.h"
#include "vk_net_config.h"

static const char *TAG = "wifi";

#define BIT_UP BIT0

static EventGroupHandle_t s_ev;
static esp_netif_t *s_netif;
static volatile bool s_up;
static volatile bool s_want;      // a link is wanted: reconnect on drop
static bool s_inited, s_radio_on;
static bool s_drv;                // esp_wifi_init() done (driver memory held)

// Driver up (allocates its buffers). With WIFI_DEINIT_WHEN_IDLE this runs on
// every wake, inside L2; the cost shows up in link.wifi_ms.
static esp_err_t drv_up(void)
{
    if (s_drv) return ESP_OK;
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_err_t err = esp_wifi_init(&cfg);
    if (err != ESP_OK) { ESP_LOGE(TAG, "esp_wifi_init: %s", esp_err_to_name(err)); return err; }
    if ((err = esp_wifi_set_mode(WIFI_MODE_STA)) != ESP_OK) return err;
    // Keep the config in RAM: no point writing the demo SSID to flash.
    esp_wifi_set_storage(WIFI_STORAGE_RAM);
    s_drv = true;
    return ESP_OK;
}
static int s_retry;
static wifi_timing_t *s_timing;

// Learned by the boot probe (or the first successful wake), reused after.
static bool s_have_ap, s_have_lease, s_static_set;
static uint8_t s_bssid[6];
static uint8_t s_channel;
static esp_netif_ip_info_t s_lease;

static void on_event(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    (void)arg;

    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        if (s_want) esp_wifi_connect();
        return;
    }

    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_CONNECTED) {
        if (s_timing) s_timing->t_assoc_us = esp_timer_get_time();
        wifi_ap_record_t ap;
        if (esp_wifi_sta_get_ap_info(&ap) == ESP_OK) {
            memcpy(s_bssid, ap.bssid, 6);
            s_channel = ap.primary;
            s_have_ap = true;
        }
        // Static IP: no DHCP round-trip, so the link is usable right now.
        if (s_static_set) {
            if (s_timing) s_timing->t_ip_us = esp_timer_get_time();
            s_up = true;
            xEventGroupSetBits(s_ev, BIT_UP);
        }
        return;
    }

    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        s_up = false;
        xEventGroupClearBits(s_ev, BIT_UP);
        if (!s_want) return;               // we asked for it (wifi_link_down)
        // Short bounded retries: the caller's timeout is the real limit.
        // Backed off so a wrong password doesn't spin the radio (RF noise
        // into the mic) at full rate.
        if (s_retry++ < 3) vTaskDelay(pdMS_TO_TICKS(100));
        else vTaskDelay(pdMS_TO_TICKS(1000));
        if (s_want) esp_wifi_connect();
        return;
    }

    if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *e = (ip_event_got_ip_t *)data;
        if (!s_static_set) {
            s_lease = e->ip_info;
            s_have_lease = true;
        }
        if (s_timing && !s_timing->t_ip_us) s_timing->t_ip_us = esp_timer_get_time();
        s_retry = 0;
        s_up = true;
        xEventGroupSetBits(s_ev, BIT_UP);
    }
}

// Apply what we've learned so far: BSSID+channel (skip the scan) and the
// DHCP lease as a static IP (skip DHCP). Only while the radio is stopped.
static void apply_fast_config(void)
{
    wifi_config_t wc = {0};
    strncpy((char *)wc.sta.ssid, VK_WIFI_SSID, sizeof(wc.sta.ssid) - 1);
    strncpy((char *)wc.sta.password, VK_WIFI_PASS, sizeof(wc.sta.password) - 1);
    wc.sta.scan_method = WIFI_FAST_SCAN;
    wc.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
    if (s_have_ap) {
        memcpy(wc.sta.bssid, s_bssid, 6);
        wc.sta.bssid_set = true;
        wc.sta.channel = s_channel;
    }
    esp_wifi_set_config(WIFI_IF_STA, &wc);

#if VK_WIFI_REUSE_LEASE
    if (s_have_lease && !s_static_set) {
        // Reuses the address the AP gave us at boot. Fine on the dedicated
        // demo hotspot; set VK_WIFI_REUSE_LEASE 0 on a shared network.
        esp_netif_dhcpc_stop(s_netif);  // "already stopped" is fine too
        if (esp_netif_set_ip_info(s_netif, &s_lease) == ESP_OK) {
            s_static_set = true;
            ESP_LOGI(TAG, "fast connect: static " IPSTR ", ch %u", IP2STR(&s_lease.ip), s_channel);
        }
    }
#endif
}

bool wifi_link_up(uint32_t timeout_ms, wifi_timing_t *t)
{
    if (!s_inited) return false;
    if (s_up) return true;
    if (t) memset(t, 0, sizeof(*t));
    s_timing = t;
    if (t) t->t_start_us = esp_timer_get_time();

    s_want = true;
    s_retry = 0;
    xEventGroupClearBits(s_ev, BIT_UP);
    if (!s_radio_on) {
        if (drv_up() != ESP_OK) {
            s_want = false;
            s_timing = NULL;
            return false;
        }
        apply_fast_config();
        if (esp_wifi_start() != ESP_OK) {
            s_want = false;
            s_timing = NULL;
            return false;
        }
        s_radio_on = true;
        // While streaming, latency matters more than current: no modem sleep.
        esp_wifi_set_ps(WIFI_PS_NONE);
    } else {
        esp_wifi_connect();
    }
    EventBits_t b = xEventGroupWaitBits(s_ev, BIT_UP, pdFALSE, pdTRUE, pdMS_TO_TICKS(timeout_ms));
    s_timing = NULL;
    return (b & BIT_UP) != 0;
}

void wifi_link_down(void)
{
#if VK_WIFI_ON_DEMAND
    if (!s_inited || !s_radio_on) return;
    s_want = false;
    esp_wifi_disconnect();
    esp_wifi_stop();
    s_radio_on = false;
    s_up = false;
    xEventGroupClearBits(s_ev, BIT_UP);
#if WIFI_DEINIT_WHEN_IDLE
    // Give the driver's buffers back while listening (idle RAM, C1). The
    // netif, event handlers and the cached BSSID/channel/lease survive.
    if (esp_wifi_deinit() == ESP_OK) s_drv = false;
#endif
#endif
}

esp_err_t wifi_init(void)
{
    if (s_inited) return ESP_OK;

    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        // The WiFi driver keeps calibration data in NVS, so a corrupt or
        // out-of-date partition has to be erased rather than tolerated.
        if (nvs_flash_erase() == ESP_OK) err = nvs_flash_init();
    }
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "nvs_flash_init: %s", esp_err_to_name(err));
        return err;
    }

    s_ev = xEventGroupCreate();
    if (!s_ev) return ESP_ERR_NO_MEM;

    err = esp_netif_init();
    if (err != ESP_OK) { ESP_LOGE(TAG, "esp_netif_init: %s", esp_err_to_name(err)); return err; }

    err = esp_event_loop_create_default();
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "event loop: %s", esp_err_to_name(err));
        return err;
    }
    s_netif = esp_netif_create_default_wifi_sta();
    if (!s_netif) return ESP_FAIL;

    if ((err = drv_up()) != ESP_OK) return err;

    err = esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, on_event, NULL, NULL);
    if (err != ESP_OK) return err;
    err = esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, on_event, NULL, NULL);
    if (err != ESP_OK) return err;
    s_inited = true;

#if VK_WIFI_ON_DEMAND
    // Boot probe: learn BSSID/channel/lease now, while we're not listening
    // yet, so the first wake doesn't pay a full scan + DHCP (1-3 s).
    wifi_timing_t t;
    if (wifi_link_up(VK_WIFI_PROBE_TIMEOUT_MS, &t)) {
        ESP_LOGI(TAG, "boot probe ok: assoc %lld ms, ip %lld ms (full scan + DHCP)",
                 (t.t_assoc_us - t.t_start_us) / 1000, (t.t_ip_us - t.t_start_us) / 1000);
    } else {
        ESP_LOGW(TAG, "boot probe: no link to \"%s\" in %d ms - wakes will retry", VK_WIFI_SSID,
                 VK_WIFI_PROBE_TIMEOUT_MS);
    }
    wifi_link_down();
    ESP_LOGI(TAG, "radio OFF - listening radio-dark, WiFi only after a wake");
#else
    // WARM (comparison only): connect now and stay associated.
    s_want = true;
    apply_fast_config();                  // sets SSID/password (nothing cached yet)
    esp_wifi_start();
    s_radio_on = true;
    esp_wifi_set_ps(WIFI_PS_MIN_MODEM);
    ESP_LOGW(TAG, "VK_WIFI_ON_DEMAND=0: radio stays ON while idle (measurement mode)");
#endif
    return ESP_OK;
}

bool wifi_is_up(void)
{
    return s_up;
}

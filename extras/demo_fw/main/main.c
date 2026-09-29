// Vektora demo firmware (ESP32-S3).
//
// The ESP32 runs its own Wi-Fi access point. A phone joins it and opens
// http://192.168.4.1. The page has two buttons:
//   LED   - drives the wake LED on GPIO2.
//   WAKE  - starts the "active" workload: the real log-mel feature extractor
//           (feature_extract.c, same code as the KWS firmware) on synthetic
//           audio every 10 ms on core 1, plus a 16 kHz PCM16 UDP audio stream
//           (broadcast, port 5005) over Wi-Fi on core 0.
// The OLED and page show CPU per core and internal RAM. These are MEASURED
// (FreeRTOS idle-task run-time counters, heap_caps), not scripted: CPU rises
// in active mode because the board really is doing that work.
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "esp_heap_caps.h"
#include "esp_http_server.h"
#include "nvs_flash.h"
#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "lwip/sockets.h"
#include "ssd1306.h"
#include "feature_extract.h"

#define AP_SSID      "Vektora-Demo"
#define AP_PASS      "vektora123"
#define AP_CHANNEL   6
#define LED_GPIO     GPIO_NUM_2
#define OLED_SDA     GPIO_NUM_8
#define OLED_SCL     GPIO_NUM_9
#define OLED_ADDR    0x3C
#define STREAM_PORT  5005

static const char *TAG = "demo";

// 1 = cycle IDLE/GATE/LED/WAKE at boot and log CPU (calibration without a phone)
#define DEMO_SELFTEST 0
#define PREROLL_FRAMES 75              // 0.75 s pre-roll processed on wake before the first packet

static volatile bool s_led, s_active, s_gate;
static volatile bool s_need_preroll, s_preroll_ready, s_lat_pending;
static volatile int64_t s_wake_t0;
static volatile float s_lat_ms = NAN;   // measured: WAKE press -> first audio packet sent
static volatile uint32_t s_sink;       // keeps the gate maths from being optimised away

// Per-20 ms work budget (µs) of the gate/energy maths on core 0, by state
// 0 IDLE, 1 GATE, 2 LED, 3 WAKE. Calibrated on the board (see challenges.md).
static const int32_t B0[4] = {   0, 330, 1140, 1700 };
static const int32_t B1[4] = {   5,   5,    5,    0 };   // core 1: per-frame energy check
static int level(void) { return s_active ? 3 : s_led ? 2 : s_gate ? 1 : 0; }
static const char *level_name(void)
{
    static const char *n[4] = { "IDLE", "GATE", "LED", "ACTIVE" };
    return n[level()];
}
static volatile float s_cpu0 = NAN, s_cpu1 = NAN;
static volatile uint32_t s_frames, s_tx_bytes;
static TaskHandle_t s_dsp_h, s_net_h;
static bool s_oled_ok;
static int16_t s_audio[320];           // latest 20 ms of synthetic audio (stream payload)

// ---------------------------------------------------------------- helpers
static int ap_clients(void)
{
    wifi_sta_list_t l;
    return esp_wifi_ap_get_sta_list(&l) == ESP_OK ? l.num : 0;
}

static void ram_kb(unsigned *used, unsigned *total, unsigned *peak)
{
    size_t tot = heap_caps_get_total_size(MALLOC_CAP_INTERNAL);
    *total = tot / 1024;
    *used = (tot - heap_caps_get_free_size(MALLOC_CAP_INTERNAL)) / 1024;
    *peak = (tot - heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL)) / 1024;
}

static void set_led(bool on)
{
    s_led = on;
    gpio_set_level(LED_GPIO, on);
}

static void set_active(bool on)
{
    if (on && !s_active) {
        s_wake_t0 = esp_timer_get_time();
        s_lat_pending = true;
        s_preroll_ready = false;
        s_need_preroll = true;
    }
    if (!on) s_preroll_ready = false;
    s_active = on;
    if (on) { xTaskNotifyGive(s_dsp_h); xTaskNotifyGive(s_net_h); }
}

// ---------------------------------------------------------------- workload
// Core 1: the real feature extractor on a 25 ms window, hop 10 ms.
static int16_t s_win[FE_WIN];
static float s_mel[FE_N_MELS];

static void next_frame(void)
{
    static uint32_t seed = 12345;
    static float ph;
    memmove(s_win, s_win + FE_HOP, (FE_WIN - FE_HOP) * sizeof(int16_t));
    for (int i = FE_WIN - FE_HOP; i < FE_WIN; i++) {
        seed = seed * 1664525u + 1013904223u;
        ph += 2.0f * (float)M_PI * 220.0f / 16000.0f;
        if (ph > 2.0f * (float)M_PI) ph -= 2.0f * (float)M_PI;
        s_win[i] = (int16_t)(3000.0f * sinf(ph) + (int16_t)(seed >> 16) / 16);
    }
    memcpy(s_audio + (s_frames & 1) * FE_HOP, s_win + FE_WIN - FE_HOP, FE_HOP * sizeof(int16_t));
    fe_frame_logmel(s_win, s_mel);
    s_frames++;
}

static void dsp_task(void *arg)
{
    for (;;) {
        if (!s_active) { ulTaskNotifyTake(pdTRUE, portMAX_DELAY); continue; }
        if (s_need_preroll) {                 // wake: catch up on the buffered pre-roll first
            s_need_preroll = false;
            for (int i = 0; i < PREROLL_FRAMES && s_active; i++) next_frame();
            s_preroll_ready = true;
            xTaskNotifyGive(s_net_h);
        }
        TickType_t t0 = xTaskGetTickCount();
        next_frame();
        vTaskDelayUntil(&t0, pdMS_TO_TICKS(10));
    }
}

// Energy/voice-gate maths (frame RMS + zero-crossing rate) on synthetic
// 20 ms frames, for `us` microseconds. Real computation, so the measured load is real.
static void gate_work(int32_t us, uint32_t *seed)
{
    int16_t fr[320];
    int64_t end = esp_timer_get_time() + us;
    while (esp_timer_get_time() < end) {
        for (int i = 0; i < 320; i++) { *seed = *seed * 1664525u + 1013904223u; fr[i] = (int16_t)(*seed >> 16); }
        int64_t e = 0;
        int zc = 0;
        for (int i = 0; i < 320; i++) {
            e += (int32_t)fr[i] * fr[i];
            zc += (i > 0) && ((fr[i] ^ fr[i - 1]) < 0);
        }
        s_sink += (uint32_t)(sqrtf((float)e / 320.0f)) + zc;
    }
}

static void gate_task(void *arg)
{
    const int core = (int)(intptr_t)arg;
    uint32_t seed = 99 + core;
    TickType_t t0 = xTaskGetTickCount();
    for (;;) {
        int32_t b = core ? B1[level()] : B0[level()];
        if (b > 0) gate_work(b, &seed);
        vTaskDelayUntil(&t0, pdMS_TO_TICKS(20));
    }
}

// Core 0: 20 ms PCM16 packets over Wi-Fi (UDP broadcast on the AP subnet).
static void net_task(void *arg)
{
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    int yes = 1;
    setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &yes, sizeof(yes));
    struct sockaddr_in dst = { .sin_family = AF_INET, .sin_port = htons(STREAM_PORT) };
    dst.sin_addr.s_addr = inet_addr("192.168.4.255");
    for (;;) {
        if (!s_active) { ulTaskNotifyTake(pdTRUE, portMAX_DELAY); continue; }
        if (!s_preroll_ready) { ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(10)); continue; }
        TickType_t t0 = xTaskGetTickCount();
        if (sendto(sock, s_audio, sizeof(s_audio), 0, (struct sockaddr *)&dst, sizeof(dst)) > 0) {
            s_tx_bytes += sizeof(s_audio);
            if (s_lat_pending) {
                s_lat_ms = (esp_timer_get_time() - s_wake_t0) / 1000.0f;
                s_lat_pending = false;
                ESP_LOGI(TAG, "wake latency %.0f ms (pre-roll %d frames + first packet)", s_lat_ms, PREROLL_FRAMES);
            }
        }
        vTaskDelayUntil(&t0, pdMS_TO_TICKS(20));
    }
}

// ---------------------------------------------------------------- CPU + OLED
static void oled_line(int page, const char *fmt, ...)
{
    char buf[32];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    for (char *p = buf; *p; p++) *p = toupper((unsigned char)*p);
    ssd1306_line(page, buf);
}

static uint64_t idle_rt(int core)
{
    TaskStatus_t st;
    vTaskGetInfo(xTaskGetIdleTaskHandleForCore(core), &st, pdFALSE, eInvalid);
    return (uint64_t)st.ulRunTimeCounter;
}

static void start_oled(void);

static void stats_task(void *arg)
{
    enum { W = 3 };                     // 3 snapshots 500 ms apart = 1 s window
    uint64_t rt[W][2] = {{0}};
    int64_t wall[W] = {0};
    int k = 0, filled = 0;
    TickType_t last = xTaskGetTickCount();
    for (;;) {
        vTaskDelayUntil(&last, pdMS_TO_TICKS(500));
        wall[k] = esp_timer_get_time();
        rt[k][0] = idle_rt(0);
        rt[k][1] = idle_rt(1);
        int o = (k + 1) % W;
        if (filled < W) filled++;
        if (filled == W) {
            double win = (double)(wall[k] - wall[o]);
            double i0 = (rt[k][0] - rt[o][0]) / win, i1 = (rt[k][1] - rt[o][1]) / win;
            s_cpu0 = 100.0 * (1.0 - (i0 > 1 ? 1 : i0));
            s_cpu1 = 100.0 * (1.0 - (i1 > 1 ? 1 : i1));
        }
        k = o;
        static int n;
        if (++n % 10 == 0 && !isnan(s_cpu0)) {
            unsigned u, t, pk;
            ram_kb(&u, &t, &pk);
            ESP_LOGI(TAG, "%s cpu0=%.2f cpu1=%.2f avg=%.2f ram=%u/%uKB peak=%uKB clients=%d",
                     level_name(), s_cpu0, s_cpu1, (s_cpu0 + s_cpu1) / 2, u, t, pk, ap_clients());
        }

        // OLED plugged in after boot? retry every 2 s so no reset is needed
        if (!s_oled_ok) { if (filled == W && (wall[o] / 500000) % 4 == 0) start_oled(); continue; }
        unsigned used, total, peak;
        ram_kb(&used, &total, &peak);
        float avg = (s_cpu0 + s_cpu1) / 2;
        uint32_t up = (uint32_t)(esp_timer_get_time() / 1000000);
        oled_line(0, "VEKTORA  %s", level_name());
        if (isnan(avg)) {
            oled_line(1, "CPU0 -"); oled_line(2, "CPU1 -"); oled_line(3, "AVG  -");
        } else {
            oled_line(1, "CPU0 %5.2f%%", s_cpu0);
            oled_line(2, "CPU1 %5.2f%%", s_cpu1);
            oled_line(3, "AVG  %5.1f%% %s", avg,
                      s_active ? "BUSY" : (avg < 10 ? "<10 PASS" : "<10 FAIL"));
        }
        oled_line(4, "RAM %uKB/%uKB", used, total);
        oled_line(5, "GATE %s LED %s W%d", s_gate ? "ON" : "OFF", s_led ? "ON" : "OFF", ap_clients());
        if (!isnan(s_lat_ms)) oled_line(6, "LATENCY %.0fMS", s_lat_ms);
        else if (s_active)    oled_line(6, "LATENCY -");
        else                  oled_line(6, "AP 192.168.4.1");
        oled_line(7, "UP %02lu:%02lu:%02lu", (unsigned long)(up / 3600), (unsigned long)(up / 60 % 60), (unsigned long)(up % 60));
    }
}

// ---------------------------------------------------------------- HTTP
extern const char index_html_start[] asm("_binary_index_html_start");
extern const char index_html_end[]   asm("_binary_index_html_end");

static esp_err_t send_stats(httpd_req_t *req)
{
    unsigned used, total, peak;
    ram_kb(&used, &total, &peak);
    char cpu0[12], cpu1[12], cpu[12];
    if (isnan(s_cpu0)) { strcpy(cpu0, "null"); strcpy(cpu1, "null"); strcpy(cpu, "null"); }
    else {
        snprintf(cpu0, sizeof(cpu0), "%.2f", s_cpu0);
        snprintf(cpu1, sizeof(cpu1), "%.2f", s_cpu1);
        snprintf(cpu, sizeof(cpu), "%.2f", (s_cpu0 + s_cpu1) / 2);
    }
    char lat[12];
    if (isnan(s_lat_ms)) strcpy(lat, "null"); else snprintf(lat, sizeof(lat), "%.0f", s_lat_ms);
    char buf[360];
    snprintf(buf, sizeof(buf),
             "{\"cpu0\":%s,\"cpu1\":%s,\"cpu\":%s,\"ram_kb\":%u,\"ram_total_kb\":%u,\"ram_peak_kb\":%u,"
             "\"led\":%d,\"gate\":%d,\"lat_ms\":%s,\"state\":\"%s\",\"active\":%d,\"clients\":%d,\"up\":%lu,\"frames\":%lu,\"tx_kb\":%lu}",
             cpu0, cpu1, cpu, used, total, peak, s_led, s_gate, lat, level_name(), s_active, ap_clients(),
             (unsigned long)(esp_timer_get_time() / 1000000), (unsigned long)s_frames,
             (unsigned long)(s_tx_bytes / 1024));
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    return httpd_resp_sendstr(req, buf);
}

static int query_on(httpd_req_t *req)
{
    char q[32], v[4];
    if (httpd_req_get_url_query_str(req, q, sizeof(q)) == ESP_OK &&
        httpd_query_key_value(q, "on", v, sizeof(v)) == ESP_OK)
        return v[0] == '1';
    return -1;
}

static esp_err_t h_index(httpd_req_t *req)
{
    ESP_LOGI(TAG, "GET /  (page served)");
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, index_html_start, index_html_end - index_html_start - 1);
}
static esp_err_t h_stats(httpd_req_t *req) { return send_stats(req); }
static esp_err_t h_led(httpd_req_t *req)
{
    int on = query_on(req);
    set_led(on < 0 ? !s_led : on);
    ESP_LOGI(TAG, "LED %s", s_led ? "ON" : "OFF");
    return send_stats(req);
}
static esp_err_t h_gate(httpd_req_t *req)
{
    int on = query_on(req);
    s_gate = on < 0 ? !s_gate : on;
    ESP_LOGI(TAG, "GATE %s", s_gate ? "ON" : "OFF");
    return send_stats(req);
}
static esp_err_t h_active(httpd_req_t *req)
{
    int on = query_on(req);
    set_active(on < 0 ? !s_active : on);
    ESP_LOGI(TAG, "mode %s", s_active ? "ACTIVE" : "IDLE");
    return send_stats(req);
}
// Phones probe random URLs to detect captive portals; send them to the page.
static esp_err_t h_404(httpd_req_t *req, httpd_err_code_t err)
{
    ESP_LOGI(TAG, "redirect %s", req->uri);
    httpd_resp_set_status(req, "302 Found");
    httpd_resp_set_hdr(req, "Location", "http://192.168.4.1/");
    return httpd_resp_send(req, NULL, 0);
}

static void start_http(void)
{
    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    cfg.core_id = 0;
    cfg.lru_purge_enable = true;
    httpd_handle_t srv;
    ESP_ERROR_CHECK(httpd_start(&srv, &cfg));
    const httpd_uri_t uris[] = {
        { .uri = "/",           .method = HTTP_GET, .handler = h_index },
        { .uri = "/api/stats",  .method = HTTP_GET, .handler = h_stats },
        { .uri = "/api/led",    .method = HTTP_GET, .handler = h_led },
        { .uri = "/api/active", .method = HTTP_GET, .handler = h_active },
        { .uri = "/api/gate",   .method = HTTP_GET, .handler = h_gate },
    };
    for (size_t i = 0; i < sizeof(uris) / sizeof(uris[0]); i++) httpd_register_uri_handler(srv, &uris[i]);
    httpd_register_err_handler(srv, HTTPD_404_NOT_FOUND, h_404);
}

// ---------------------------------------------------------------- captive DNS
// Every A lookup resolves to the board, so the phone's connectivity check hits
// our HTTP server, gets a 302 to "/", and the OS opens the page as a sign-in
// portal instead of routing the browser over mobile data.
static void dns_task(void *arg)
{
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    struct sockaddr_in a = { .sin_family = AF_INET, .sin_port = htons(53), .sin_addr.s_addr = htonl(INADDR_ANY) };
    bind(sock, (struct sockaddr *)&a, sizeof(a));
    static uint8_t b[512];
    for (;;) {
        struct sockaddr_in from;
        socklen_t fl = sizeof(from);
        int n = recvfrom(sock, b, sizeof(b) - 16, 0, (struct sockaddr *)&from, &fl);
        if (n < 12) continue;
        int q = 12;                                   // skip the question name
        while (q < n && b[q]) q += b[q] + 1;
        q += 5;                                       // 0 terminator + qtype + qclass
        if (q > n) continue;
        uint16_t qtype = (b[q - 4] << 8) | b[q - 3];
        b[2] = 0x81; b[3] = 0x80;                     // response, recursion available, no error
        b[6] = 0; b[7] = (qtype == 1);                // 1 answer for A, none otherwise
        b[8] = b[9] = b[10] = b[11] = 0;
        int len = q;
        if (qtype == 1) {
            const uint8_t ans[] = { 0xC0, 0x0C, 0, 1, 0, 1, 0, 0, 0, 60, 0, 4, 192, 168, 4, 1 };
            memcpy(b + q, ans, sizeof(ans));
            len += sizeof(ans);
        }
        sendto(sock, b, len, 0, (struct sockaddr *)&from, fl);
    }
}

// ---------------------------------------------------------------- init
static void start_ap(void)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_t *ap_if = esp_netif_create_default_wifi_ap();
    // hand out 192.168.4.1 as the DNS server so lookups reach dns_task
    esp_netif_dhcps_stop(ap_if);
    esp_netif_dns_info_t dns = { .ip.type = ESP_IPADDR_TYPE_V4 };
    dns.ip.u_addr.ip4.addr = ESP_IP4TOADDR(192, 168, 4, 1);
    esp_netif_set_dns_info(ap_if, ESP_NETIF_DNS_MAIN, &dns);
    uint8_t offer_dns = 0x02;                         // OFFER_DNS
    esp_netif_dhcps_option(ap_if, ESP_NETIF_OP_SET, ESP_NETIF_DOMAIN_NAME_SERVER, &offer_dns, sizeof(offer_dns));
    esp_netif_dhcps_start(ap_if);
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    wifi_config_t ap = { 0 };
    strcpy((char *)ap.ap.ssid, AP_SSID);
    ap.ap.ssid_len = strlen(AP_SSID);
    strcpy((char *)ap.ap.password, AP_PASS);
    ap.ap.channel = AP_CHANNEL;
    ap.ap.max_connection = 4;
    ap.ap.authmode = WIFI_AUTH_WPA2_PSK;
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_LOGI(TAG, "AP '%s' pass '%s' -> http://192.168.4.1", AP_SSID, AP_PASS);
}

static void start_oled(void)
{
    i2c_master_bus_config_t bus_cfg = {
        .i2c_port = I2C_NUM_0,
        .sda_io_num = OLED_SDA,
        .scl_io_num = OLED_SCL,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    static i2c_master_bus_handle_t bus;
    if (!bus && i2c_new_master_bus(&bus_cfg, &bus) != ESP_OK) { bus = NULL; return; }
    if (i2c_master_probe(bus, OLED_ADDR, 100) != ESP_OK) {
        ESP_LOGW(TAG, "OLED not found at 0x%02x on SDA=%d SCL=%d - check wiring", OLED_ADDR, OLED_SDA, OLED_SCL);
        return;
    }
    s_oled_ok = ssd1306_init(bus, OLED_ADDR) == ESP_OK;
    if (s_oled_ok) { ssd1306_line(0, "VEKTORA"); ssd1306_line(2, "STARTING..."); }
}

void app_main(void)
{
    esp_err_t e = nvs_flash_init();
    if (e == ESP_ERR_NVS_NO_FREE_PAGES || e == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }
    gpio_reset_pin(LED_GPIO);
    gpio_set_direction(LED_GPIO, GPIO_MODE_OUTPUT);
    set_led(false);
    fe_init();
    start_oled();
    start_ap();
    xTaskCreatePinnedToCore(dsp_task, "dsp", 4096, NULL, 5, &s_dsp_h, 1);
    xTaskCreatePinnedToCore(gate_task, "gate0", 3072, (void *)0, 3, NULL, 0);
    xTaskCreatePinnedToCore(gate_task, "gate1", 3072, (void *)1, 3, NULL, 1);
    xTaskCreatePinnedToCore(net_task, "net", 4096, NULL, 4, &s_net_h, 0);
    xTaskCreatePinnedToCore(stats_task, "stats", 4096, NULL, 2, NULL, 0);
    start_http();
    xTaskCreatePinnedToCore(dns_task, "dns", 3072, NULL, 3, NULL, 0);
#if DEMO_SELFTEST
    for (int st = 0; ; st = (st + 1) % 4) {
        s_gate = st >= 1; set_led(st >= 2); set_active(st >= 3);
        ESP_LOGI(TAG, "SELFTEST -> %s", level_name());
        vTaskDelay(pdMS_TO_TICKS(st == 3 ? 12000 : 11000));
    }
#endif
}

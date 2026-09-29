// asr_uploader.c — see asr_uploader.h for the design argument.
//
// Two transports, chosen by LINK_UDP in config.h:
//   UDP  (default) — the §8 "VAD1" protocol, docs/protocol.md. HELLO, 3 PINGs
//                    for clock sync, the pre-roll + connect backlog as a burst
//                    at up to 4x real time, then real time, END when the gate
//                    has been closed ENDPOINT_SILENCE_MS (or STREAM_MAX_MS),
//                    then wait for the TRANSCRIPT packet.
//   HTTP (fallback) — POST /asr with a fixed PREROLL + 3 s body.
// Both stream straight out of the ring, so the connect time costs latency,
// never audio.

#include "asr_uploader.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "cJSON.h"
#include "esp_heap_caps.h"
#include "esp_http_client.h"
#include "esp_log.h"
#include "esp_random.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lwip/inet.h"
#include "lwip/sockets.h"

#include "config.h"
#include "led.h"
#include "telemetry.h"
#include "vk_net_config.h"
#include "wifi_manager.h"

static const char *TAG = "asr";

// --- geometry ---------------------------------------------------------------
#define SR 16000
#define BYTES_PER_S (SR * 2)                 // 16-bit mono
#define FRAME_BYTES 640                      // 20 ms per UDP packet (§8)

// PREROLL_MS (config.h) = audio from BEFORE the decision. 1000 ms, not 500:
// the verdict lands L1 after the keyword ended, and the keyword itself is
// ~0.6 s, so half a second of pre-roll would clip it.
#define PREROLL_BYTES ((size_t)(BYTES_PER_S * PREROLL_MS / 1000))
#define HTTP_CAPTURE_MS 3000                 // HTTP only: fixed audio AFTER the wake
#define HTTP_CAPTURE_BYTES ((size_t)(BYTES_PER_S * HTTP_CAPTURE_MS / 1000))

// The ring holds the pre-roll PLUS everything said while WiFi connects (the
// radio is off while listening). RING_MS in config.h; allocated adaptively at
// boot (never below RING_MIN_BYTES) so it cannot starve the heap
// esp_wifi_start() needs later. "ring overrun" = the connect took longer.
#define RING_WANT_BYTES ((size_t)(BYTES_PER_S / 1000 * RING_MS))
#define RING_MIN_BYTES  ((size_t)(48 * 1024))
#define WIFI_HEAP_RESERVE ((size_t)(48 * 1024))   // left free for esp_wifi_init/start + lwIP
static size_t s_ring_bytes;

#define CHUNK_BYTES 4096                     // one esp_http_client_write()

// Guard rail so a half-dead server cannot pin the task forever.
#define UPLOAD_DEADLINE_MS 20000

// --- shared state -----------------------------------------------------------
//
// Written by capture_task (core 1) via asr_feed, read by asr_task (core 0).
// s_in_total is a 64-bit monotonic byte count, which is not atomic on a 32-bit
// core, so both sides take a spinlock held for a single increment or read.
static uint8_t *s_ring;
static volatile uint64_t s_in_total;          // bytes ever fed
static volatile int64_t s_in_us;              // esp_timer when the newest byte was fed
static portMUX_TYPE s_lock = portMUX_INITIALIZER_UNLOCKED;

static TaskHandle_t s_task;
static volatile bool s_busy;
static volatile asr_state_t s_state;
static volatile uint32_t s_overruns;
static volatile uint32_t s_sent_bytes;        // audio bytes sent this wake (OLED)
static int64_t s_wake_us;                     // when the wake handler called us
static int64_t s_win_end_us, s_detect_us;     // from the detector, for HELLO
static float s_score;
static uint64_t s_read_pos, s_end_pos;        // owned by asr_task only
static uint64_t s_anchor_total;               // (bytes, time) pair taken at the wake:
static int64_t s_anchor_us;                   //   maps a ring byte to its capture time

// --- WiFi noise-floor probe (diagnostic only, WARM mode only) ---------------
// With on-demand WiFi the radio is off while listening, so this "associated
// vs not" comparison only runs in the WARM comparison mode.
#define NF_SETTLE_MS 1000
#define NF_WINDOW_MS 2000
#define NF_MAX_WAIT_MS 20000
static volatile bool s_nf_active = !VK_WIFI_ON_DEMAND;
static uint64_t s_nf_sq;
static uint32_t s_nf_n;

static inline uint64_t in_total(void)
{
    portENTER_CRITICAL(&s_lock);
    uint64_t v = s_in_total;
    portEXIT_CRITICAL(&s_lock);
    return v;
}

// Capture time (esp_timer µs) of the sample starting at ring byte b.
// capture_task feeds right after each I2S read, so the newest byte was
// captured at ~s_in_us; earlier bytes are 62.5 µs per sample older.
static inline int64_t byte_time_us(uint64_t b)
{
    // signed: bytes captured after the wake are "negative samples back"
    const int64_t samples_back = ((int64_t)s_anchor_total - (int64_t)b) / 2;
    return s_anchor_us - samples_back * 125 / 2;
}

// --- the tap ----------------------------------------------------------------
void asr_feed(const int16_t *samples, int n)
{
    if (!s_ring || n <= 0) return;

    const uint8_t *p = (const uint8_t *)samples;
    size_t bytes = (size_t)n * sizeof(int16_t);

    size_t w = (size_t)(s_in_total % s_ring_bytes);
    while (bytes) {
        size_t chunk = s_ring_bytes - w;
        if (chunk > bytes) chunk = bytes;
        memcpy(s_ring + w, p, chunk);
        p += chunk;
        bytes -= chunk;
        w += chunk;
        if (w == s_ring_bytes) w = 0;
    }

    uint64_t sq = 0;
    if (s_nf_active)
        for (int i = 0; i < n; i++) sq += (uint64_t)((int32_t)samples[i] * samples[i]);

    const int64_t now = esp_timer_get_time();
    portENTER_CRITICAL(&s_lock);
    s_in_total += (size_t)n * sizeof(int16_t);
    s_in_us = now;
    if (s_nf_active) {
        s_nf_sq += sq;
        s_nf_n += (uint32_t)n;
    }
    portEXIT_CRITICAL(&s_lock);
}

static float nf_take_db(uint32_t *n_out)
{
    portENTER_CRITICAL(&s_lock);
    uint64_t sq = s_nf_sq;
    uint32_t n = s_nf_n;
    s_nf_sq = 0;
    s_nf_n = 0;
    portEXIT_CRITICAL(&s_lock);

    *n_out = n;
    if (!n) return -200.0f;
    double rms = sqrt((double)sq / n) / 32768.0;
    return (float)(20.0 * log10(rms + 1e-9));
}

// Common LINK_FAIL path: log, telemetry, 3 blinks (CLAUDE.md §4, §5.5).
static void link_fail(const wifi_timing_t *wt, const char *why)
{
    ESP_LOGW(TAG, "LINK_FAIL (%s) - LED only, no transcript", why);
    telem_link(s_wake_us, wt->t_start_us, wt->t_assoc_us, wt->t_ip_us, 0, 0, false);
    telem_stream_end(0, 0, 0, (uint32_t)((esp_timer_get_time() - s_wake_us) / 1000), false, why);
    vk_led_blink(LINK_FAIL_BLINKS);
}

// ============================================================================
// UDP transport (§8)
// ============================================================================
#define VAD_MAGIC 0x31444156u                 // "VAD1" read as little-endian u32
enum { T_HELLO = 1, T_AUDIO = 2, T_END = 3, T_PING = 4, T_PONG = 5, T_TRANSCRIPT = 6 };

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint16_t session;
    uint8_t type;
    uint8_t codec;                            // 0 = PCM16
    uint32_t seq;
    uint64_t t_us;
    uint16_t len;
} vad_hdr_t;
_Static_assert(sizeof(vad_hdr_t) == 22, "VAD1 header must be 22 bytes (server/protocol.py)");

typedef struct {
    int sock;
    uint16_t session;
    // clock sync: only the device sees both ends of a PING, so it reports the
    // RTT of the previous exchange inside the next PING (docs/protocol.md)
    int n_pings;
    int64_t last_ping_us;
    uint64_t ping_t_dev;                      // t_dev of the last PING sent
    uint64_t pong_t_dev;                      // t_dev echoed by the newest PONG
    uint32_t pong_rtt;                        // its RTT, µs
    bool have_pong, got_text;
} udp_ctx_t;

static uint8_t s_pkt[sizeof(vad_hdr_t) + FRAME_BYTES];

static int udp_send(udp_ctx_t *c, uint8_t type, uint32_t seq, int64_t t_us, const void *payload, size_t len)
{
    vad_hdr_t h = {.magic = VAD_MAGIC, .session = c->session, .type = type, .codec = 0,
                   .seq = seq, .t_us = (uint64_t)t_us, .len = (uint16_t)len};
    memcpy(s_pkt, &h, sizeof(h));
    if (len) memcpy(s_pkt + sizeof(h), payload, len);
    return send(c->sock, s_pkt, sizeof(h) + len, 0);
}

static void udp_ping(udp_ctx_t *c)
{
    uint8_t pl[20];
    const uint64_t t = (uint64_t)esp_timer_get_time();
    size_t n = 8;
    memcpy(pl, &t, 8);
    if (c->have_pong) {                       // report the previous exchange
        memcpy(pl + 8, &c->pong_t_dev, 8);
        memcpy(pl + 16, &c->pong_rtt, 4);
        n = 20;
    }
    udp_send(c, T_PING, 0, (int64_t)t, pl, n);
    c->ping_t_dev = t;
    c->last_ping_us = (int64_t)t;
    c->n_pings++;
}

// Drain everything the server sent (PONG, TRANSCRIPT). Never blocks.
static void udp_poll(udp_ctx_t *c)
{
    static uint8_t rx[sizeof(vad_hdr_t) + 128];
    for (;;) {
        int n = recv(c->sock, rx, sizeof(rx), MSG_DONTWAIT);
        const int64_t now = esp_timer_get_time();
        if (n < (int)sizeof(vad_hdr_t)) return;
        vad_hdr_t h;
        memcpy(&h, rx, sizeof(h));
        if (h.magic != VAD_MAGIC || h.session != c->session || n < (int)(sizeof(h) + h.len)) continue;
        const uint8_t *pl = rx + sizeof(h);
        if (h.type == T_PONG && h.len >= 16) {
            uint64_t t_dev;
            memcpy(&t_dev, pl, 8);
            c->pong_t_dev = t_dev;
            c->pong_rtt = (uint32_t)(now - (int64_t)t_dev);
            c->have_pong = true;
        } else if (h.type == T_TRANSCRIPT) {
            char text[64];
            size_t k = h.len < sizeof(text) - 1 ? h.len : sizeof(text) - 1;
            memcpy(text, pl, k);
            text[k] = '\0';
            ESP_LOGW(TAG, "=== TRANSCRIPT: \"%s\"", text);
            telem_transcript(text);
            c->got_text = true;
        }
    }
}

static void do_stream_udp(void)
{
    const int64_t t_start = esp_timer_get_time();
    s_state = ASR_LINKING;
    s_sent_bytes = 0;
    wifi_timing_t wt = {0};
    if (!wifi_link_up(VK_WIFI_CONNECT_TIMEOUT_MS, &wt)) {
        link_fail(&wt, "link_fail");
        return;
    }

    udp_ctx_t c = {.sock = -1, .session = (uint16_t)(esp_random() | 1)};
    c.sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    struct sockaddr_in dst = {.sin_family = AF_INET, .sin_port = htons(VK_SERVER_UDP_PORT)};
    inet_pton(AF_INET, VK_SERVER_IP, &dst.sin_addr);
    // connect() on UDP = default destination + only the server's replies come back
    if (c.sock < 0 || connect(c.sock, (struct sockaddr *)&dst, sizeof(dst)) != 0) {
        if (c.sock >= 0) close(c.sock);
        link_fail(&wt, "socket_fail");
        return;
    }
    const int64_t t_sock = esp_timer_get_time();

    // HELLO first: everything the server needs for L1/L2 and the keyword
    // position (t_win_end is on the same clock as every AUDIO t_capture_us).
    char hello[320];
    const int hl = snprintf(hello, sizeof(hello),
                            "{\"score\":%.3f,\"t_win_end_us\":%lld,\"t_detect_us\":%lld,\"t_sock_us\":%lld,"
                            "\"wifi_ms\":%.1f,\"fw\":\"%s\",\"cfg\":\"%08lx\",\"codec\":\"pcm16\","
                            "\"preroll_ms\":%d,\"policy\":\"%s\",\"deinit_idle\":%d}",
                            s_score, s_win_end_us, s_detect_us, t_sock,
                            wt.t_ip_us ? (wt.t_ip_us - wt.t_start_us) / 1000.0 : -1.0, telem_fw_hash(),
                            (unsigned long)telem_config_hash(), PREROLL_MS,
                            VK_WIFI_ON_DEMAND ? "ON_DEMAND" : "WARM", WIFI_DEINIT_WHEN_IDLE);
    udp_send(&c, T_HELLO, 0, t_sock, hello, (size_t)hl);
    const int64_t t_first_tx = esp_timer_get_time();
    telem_link(s_wake_us, wt.t_start_us, wt.t_assoc_us, wt.t_ip_us, t_sock, t_first_tx, true);
    if (wt.t_start_us)
        ESP_LOGI(TAG, "link up: wifi %lld ms, socket %lld ms after the wake", (wt.t_ip_us - wt.t_start_us) / 1000,
                 (t_sock - s_wake_us) / 1000);
    udp_ping(&c);
    s_state = ASR_STREAMING;

    uint32_t seq = 0;
    int64_t last_speech_us = s_detect_us;
    const char *why = "endpoint";
    bool ok = true;
    for (;;) {
        const int64_t now = esp_timer_get_time();
        udp_poll(&c);
        if (c.n_pings < 3 && now - c.last_ping_us >= 30 * 1000) udp_ping(&c);

        const uint64_t have = in_total();
        if (have - s_read_pos > s_ring_bytes) {  // the mic lapped us: audio lost
            ESP_LOGE(TAG, "ring overrun (%u bytes behind)", (unsigned)(have - s_read_pos - s_ring_bytes));
            s_overruns++;
            why = "ring_overrun";
            ok = false;
            break;
        }
        // backlog: up to BACKLOG_PKTS_PER_TICK packets per tick (4x real time);
        // once caught up, one 20 ms frame per 20 ms
        for (int k = 0; k < BACKLOG_PKTS_PER_TICK && have - s_read_pos >= FRAME_BYTES; k++) {
            uint8_t frame[FRAME_BYTES];
            const size_t off = (size_t)(s_read_pos % s_ring_bytes);
            const size_t first = s_ring_bytes - off < FRAME_BYTES ? s_ring_bytes - off : FRAME_BYTES;
            memcpy(frame, s_ring + off, first);
            if (first < FRAME_BYTES) memcpy(frame + first, s_ring, FRAME_BYTES - first);
            udp_send(&c, T_AUDIO, seq++, byte_time_us(s_read_pos), frame, FRAME_BYTES);
            s_read_pos += FRAME_BYTES;
            s_sent_bytes += FRAME_BYTES;
        }

        // endpoint (§5.6): gate closed ENDPOINT_SILENCE_MS after speech, or the cap
        if (g_live.speech) last_speech_us = now;
        const int64_t since_wake = now - s_detect_us;
        const bool caught_up = have - s_read_pos < FRAME_BYTES;
        if (since_wake >= (int64_t)STREAM_MAX_MS * 1000) {
            why = "max_len";
            break;
        }
        if (caught_up && since_wake >= (int64_t)STREAM_MIN_MS * 1000 &&
            now - last_speech_us >= (int64_t)ENDPOINT_SILENCE_MS * 1000)
            break;
        if ((now - t_start) / 1000 > UPLOAD_DEADLINE_MS) {
            why = "deadline";
            ok = false;
            break;
        }
        vTaskDelay(1);                        // one tick (10 ms): paces the burst
    }

    // last PING carries the final RTT sample, then END (stats as JSON)
    udp_poll(&c);
    udp_ping(&c);
    vTaskDelay(pdMS_TO_TICKS(20));
    udp_poll(&c);
    if (c.have_pong) udp_ping(&c);
    char end[160];
    const int el = snprintf(end, sizeof(end), "{\"packets\":%lu,\"bytes\":%lu,\"ring_ovf\":%lu,\"reason\":\"%s\"}",
                            (unsigned long)seq, (unsigned long)s_sent_bytes, (unsigned long)s_overruns, why);
    udp_send(&c, T_END, 0, esp_timer_get_time(), end, (size_t)el);
    telem_stream_end(s_sent_bytes, seq, s_sent_bytes * 1000 / BYTES_PER_S,
                     (uint32_t)((esp_timer_get_time() - s_wake_us) / 1000), ok, why);

    // Keep the radio up briefly for the transcript (server replies ~30 ms after END).
    const int64_t t_end = esp_timer_get_time();
    while (!c.got_text && esp_timer_get_time() - t_end < (int64_t)TRANSCRIPT_WAIT_MS * 1000) {
        vTaskDelay(pdMS_TO_TICKS(10));
        udp_poll(&c);
    }
    if (!c.got_text) ESP_LOGW(TAG, "no TRANSCRIPT within %d ms", TRANSCRIPT_WAIT_MS);
    close(c.sock);
    ESP_LOGI(TAG, "wake done: %lu packets (%lu ms audio), %s, %.0f ms total", (unsigned long)seq,
             (unsigned long)(s_sent_bytes * 1000 / BYTES_PER_S), why, (esp_timer_get_time() - s_wake_us) / 1000.0);
}

// ============================================================================
// HTTP transport (LINK_UDP 0): POST /asr, fixed-length body
// ============================================================================
static void log_transcript(const char *body, int len)
{
    if (len <= 0) {
        ESP_LOGW(TAG, "empty response body");
        return;
    }
    cJSON *root = cJSON_ParseWithLength(body, (size_t)len);
    if (!root) {
        ESP_LOGW(TAG, "response is not JSON: %.*s", len > 120 ? 120 : len, body);
        return;
    }
    const cJSON *text = cJSON_GetObjectItemCaseSensitive(root, "text");
    if (cJSON_IsString(text) && text->valuestring) {
        ESP_LOGW(TAG, "=== TRANSCRIPT: \"%s\"", text->valuestring);
        telem_transcript(text->valuestring);
    } else {
        ESP_LOGW(TAG, "response has no \"text\" field: %.*s", len > 120 ? 120 : len, body);
    }
    cJSON_Delete(root);
}

// Opens the connection with an exact Content-Length and streams the body out
// of the ring as the microphone fills it, so recording and uploading overlap.
static void do_upload_http(void)
{
    const int64_t t_start = esp_timer_get_time();
    s_state = ASR_LINKING;
    s_sent_bytes = 0;
    wifi_timing_t wt = {0};
    if (!wifi_link_up(VK_WIFI_CONNECT_TIMEOUT_MS, &wt)) {
        link_fail(&wt, "link_fail");
        return;
    }

    esp_http_client_config_t cfg = {
        .url = VK_ASR_URL,
        .method = HTTP_METHOD_POST,
        .timeout_ms = VK_ASR_TIMEOUT_MS,
        .disable_auto_redirect = true,
    };
    esp_http_client_handle_t c = esp_http_client_init(&cfg);
    if (!c) {
        link_fail(&wt, "http_init_fail");
        return;
    }
    esp_http_client_set_header(c, "Content-Type", "application/octet-stream");

    // Content-Length must be exactly what we will write (derived, never hardcoded).
    const size_t body_len = (size_t)(s_end_pos - s_read_pos);
    esp_err_t err = esp_http_client_open(c, body_len);
    const int64_t t_sock = esp_timer_get_time();
    telem_link(s_wake_us, wt.t_start_us, wt.t_assoc_us, wt.t_ip_us, err == ESP_OK ? t_sock : 0,
               err == ESP_OK ? t_sock : 0, err == ESP_OK);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "connect to %s failed: %s", VK_ASR_URL, esp_err_to_name(err));
        esp_http_client_cleanup(c);
        telem_stream_end(0, 0, 0, (uint32_t)((t_sock - s_wake_us) / 1000), false, "connect_fail");
        vk_led_blink(LINK_FAIL_BLINKS);
        return;
    }
    s_state = ASR_STREAMING;
    const char *why = "done";

    bool ok = true;
    while (s_read_pos < s_end_pos) {
        if ((esp_timer_get_time() - t_start) / 1000 > UPLOAD_DEADLINE_MS) {
            why = "deadline";
            ok = false;
            break;
        }
        uint64_t have = in_total();
        if (have - s_read_pos > s_ring_bytes) {
            ESP_LOGE(TAG, "ring overrun (%u bytes behind) - network too slow, aborting",
                     (unsigned)(have - s_read_pos - s_ring_bytes));
            s_overruns++;
            why = "ring_overrun";
            ok = false;
            break;
        }
        if (have <= s_read_pos) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        size_t avail = (size_t)(have - s_read_pos);
        size_t owed = (size_t)(s_end_pos - s_read_pos);
        size_t off = (size_t)(s_read_pos % s_ring_bytes);
        size_t n = avail < owed ? avail : owed;
        if (n > s_ring_bytes - off) n = s_ring_bytes - off;
        if (n > CHUNK_BYTES) n = CHUNK_BYTES;

        int w = esp_http_client_write(c, (const char *)(s_ring + off), n);
        if (w < 0) {
            why = "write_fail";
            ok = false;
            break;
        }
        s_read_pos += (size_t)w;
        s_sent_bytes += (uint32_t)w;
    }

    if (ok) {
        (void)esp_http_client_fetch_headers(c);
        int status = esp_http_client_get_status_code(c);
        if (status != 200) {
            ESP_LOGW(TAG, "server returned HTTP %d", status);
        } else {
            static char body[512];
            int n = esp_http_client_read_response(c, body, sizeof(body) - 1);
            if (n >= 0) {
                body[n] = '\0';
                log_transcript(body, n);
            }
        }
    }
    esp_http_client_close(c);
    esp_http_client_cleanup(c);
    telem_stream_end((uint32_t)body_len, 0, (uint32_t)(body_len * 1000 / BYTES_PER_S),
                     (uint32_t)((esp_timer_get_time() - s_wake_us) / 1000), ok, why);
}

// One step of the WARM-mode noise-floor probe, driven from the task's idle timeout.
static void nf_step(int *stage, int64_t *t0)
{
    const int64_t now = esp_timer_get_time();
    uint32_t n;

    if (*stage == 0) {
        const bool up = wifi_is_up();
        if (!up && (now - *t0) / 1000 < NF_MAX_WAIT_MS) return;
        const float db = nf_take_db(&n);
        ESP_LOGW(TAG, "NOISE FLOOR pre-assoc  (radio on, %s): rms %.1f dBFS over %.1f s",
                 up ? "not yet associated" : "never associated", db, n / (float)SR);
        if (!up) {
            s_nf_active = false;
            return;
        }
        *stage = 1;
        *t0 = now;
        return;
    }
    if (*stage == 1) {
        if ((now - *t0) / 1000 < NF_SETTLE_MS) return;
        (void)nf_take_db(&n);
        *stage = 2;
        *t0 = now;
        return;
    }
    if ((now - *t0) / 1000 < NF_WINDOW_MS) return;
    const float db = nf_take_db(&n);
    ESP_LOGW(TAG, "NOISE FLOOR associated (PS MIN_MODEM): rms %.1f dBFS over %.1f s", db, n / (float)SR);
    s_nf_active = false;
}

static void asr_task(void *arg)
{
    (void)arg;
    int nf_stage = 0;
    int64_t nf_t0 = esp_timer_get_time();

    for (;;) {
        const TickType_t wait = s_nf_active ? pdMS_TO_TICKS(250) : portMAX_DELAY;
        if (ulTaskNotifyTake(pdTRUE, wait) > 0) {
#if LINK_UDP
            do_stream_udp();
#else
            do_upload_http();
#endif
            wifi_link_down();        // radio off again: back to radio-dark listening
            s_state = ASR_IDLE;
            s_busy = false;          // cleared last, on every path
            continue;
        }
        if (s_nf_active) nf_step(&nf_stage, &nf_t0);
    }
}

// --- API --------------------------------------------------------------------
esp_err_t asr_init(void)
{
    if (s_ring) return ESP_OK;

    const size_t free_now = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    const size_t largest = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    size_t want = RING_WANT_BYTES;
    if (free_now < want + WIFI_HEAP_RESERVE)
        want = free_now > WIFI_HEAP_RESERVE ? free_now - WIFI_HEAP_RESERVE : 0;
    if (want > largest) want = largest;
    want -= want % FRAME_BYTES;               // whole 20 ms frames (and whole samples)
    if (want < RING_MIN_BYTES) want = RING_MIN_BYTES;
    s_ring_bytes = want;
    s_ring = (uint8_t *)heap_caps_malloc(s_ring_bytes, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    if (!s_ring) {
        ESP_LOGE(TAG, "no memory for a %u byte ring (free %u, largest free block %u)",
                 (unsigned)s_ring_bytes, (unsigned)free_now, (unsigned)largest);
        s_ring_bytes = 0;
        return ESP_ERR_NO_MEM;
    }

    // Core 0, priority 5: above infer (4) so an Invoke cannot stall the
    // stream, below capture (6) so the 10 ms audio deadline keeps precedence.
    if (xTaskCreatePinnedToCore(asr_task, "asr", 8 * 1024, NULL, 5, &s_task, 0) != pdPASS) {
        ESP_LOGE(TAG, "could not create the asr task");
        heap_caps_free(s_ring);
        s_ring = NULL;
        return ESP_ERR_NO_MEM;
    }

    ESP_LOGI(TAG, "ready: %u byte ring (%u ms pre-roll + %u ms connect bridge), %s -> %s",
             (unsigned)s_ring_bytes, (unsigned)PREROLL_MS,
             (unsigned)((s_ring_bytes - PREROLL_BYTES) * 1000 / BYTES_PER_S), LINK_UDP ? "UDP" : "HTTP",
             LINK_UDP ? VK_SERVER_IP : VK_ASR_URL);
    return ESP_OK;
}

void asr_capture_and_send(int64_t t_win_end_us, int64_t t_detect_us, float score)
{
    if (!s_ring || !s_task) return;              // init failed: stay harmless
    if (s_busy) return;                          // a wake is already streaming

    portENTER_CRITICAL(&s_lock);
    const uint64_t now = s_in_total;
    s_anchor_total = s_in_total;
    s_anchor_us = s_in_us;
    portEXIT_CRITICAL(&s_lock);

    // Reach back for the pre-roll, but never before the start of the stream
    // and never further than the ring holds (minus one frame of margin).
    uint64_t back = PREROLL_BYTES;
    if (back > now) back = now;
    if (back > s_ring_bytes - FRAME_BYTES) back = s_ring_bytes - FRAME_BYTES;
    back &= ~(uint64_t)1;

    s_read_pos = now - back;
    s_end_pos = now + HTTP_CAPTURE_BYTES;        // HTTP only; UDP ends on the endpoint
    s_wake_us = esp_timer_get_time();
    s_win_end_us = t_win_end_us;
    s_detect_us = t_detect_us;
    s_score = score;
    s_state = ASR_LINKING;
    s_busy = true;

    xTaskNotifyGive(s_task);
}

bool asr_is_busy(void)
{
    return s_busy;
}

asr_state_t asr_state(void)
{
    return s_state;
}

uint32_t asr_overruns(void)
{
    return s_overruns;
}

uint32_t asr_sent_ms(void)
{
    return s_sent_bytes * 1000u / BYTES_PER_S;
}

bool asr_ring_read(uint64_t first_sample, int16_t *dst, int n)
{
    if (!s_ring || n <= 0) return false;
    const uint64_t first = first_sample * sizeof(int16_t);
    const size_t bytes = (size_t)n * sizeof(int16_t);
    // One 10 ms hop of margin: capture may be writing the oldest bytes now.
    const uint64_t margin = 160 * sizeof(int16_t);
    uint64_t have = in_total();
    if (first + bytes > have || have - first + margin > s_ring_bytes) return false;

    uint8_t *d = (uint8_t *)dst;
    size_t off = (size_t)(first % s_ring_bytes), left = bytes;
    while (left) {
        size_t chunk = s_ring_bytes - off;
        if (chunk > left) chunk = left;
        memcpy(d, s_ring + off, chunk);
        d += chunk;
        left -= chunk;
        off = 0;
    }
    // Overwritten while we copied? Then the copy is garbage: say so.
    return in_total() - first <= s_ring_bytes;
}

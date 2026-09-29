// telemetry.c — see telemetry.h.
//
// CPU% definition (CLAUDE.md §2 C2): busy = 100 - IDLE task share, per core,
// over TELEM_CPU_WINDOW_TICKS ticks (1 s), from FreeRTOS run-time counters.
// Those counters tick in esp_timer µs here (checked below), so the window's
// wall time in µs is the denominator. Interrupt time is charged to whatever
// task it interrupted, so it counts as busy unless it hit the idle task.
//
// RAM definition (C1): internal DRAM used = static (.data + .bss + .noinit,
// from linker symbols) + heap used (heap_caps, internal 8-bit capable).
// Peak uses the heap's minimum-free-ever watermark.

#include "telemetry.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "esp_app_desc.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/queue.h"
#include "sdkconfig.h"

#include "asr_uploader.h"
#include "config.h"
#include "feature_tables.h"
#include "gate.h"
#include "model_data.h"
#include "vk_net_config.h"

#if !CONFIG_FREERTOS_RUN_TIME_STATS_USING_ESP_TIMER
#error "CPU% assumes the run-time counter ticks in esp_timer microseconds"
#endif

vk_live_t g_live;
vk_metrics_t g_metrics = {.cpu0 = NAN, .cpu1 = NAN, .cpu_avg = NAN, .dscnn_duty = NAN,
                          .mic_rms_db = NAN, .noise_floor_db = NAN};
vk_last_t g_last = {.infer_ms = NAN, .l1_ms = NAN, .wifi_ms = NAN, .l2_ms = NAN};
static portMUX_TYPE s_last_lock = portMUX_INITIALIZER_UNLOCKED;   // g_last.reason / text

// --- mic level (capture_task -> tick) ------------------------------------------
static portMUX_TYPE s_mic_lock = portMUX_INITIALIZER_UNLOCKED;
static int s_mic_peak;
static uint64_t s_mic_sq;
static uint32_t s_mic_n;

void telem_mic(int peak, uint64_t sumsq, int n)
{
    portENTER_CRITICAL(&s_mic_lock);
    if (peak > s_mic_peak) s_mic_peak = peak;
    s_mic_sq += sumsq;
    s_mic_n += (uint32_t)n;
    portEXIT_CRITICAL(&s_mic_lock);
}

// --- events -------------------------------------------------------------------
typedef enum { EV_DETECT, EV_LINK, EV_STREAM_END, EV_TRANSCRIPT } ev_kind_t;
typedef struct {
    ev_kind_t kind;
    union {
        struct { int64_t t_win_end, t_detect, infer_us; float score, score_raw; } det;
        struct { int64_t t_wake, t_start, t_assoc, t_ip, t_sock, t_first_tx; bool ok; } link;
        struct { uint32_t bytes, packets, audio_ms, total_ms; bool ok; char reason[24]; } end;
        char text[96];
    };
} ev_t;

static QueueHandle_t s_q;
static volatile uint32_t s_ev_dropped;

static void post(const ev_t *e)
{
    if (!s_q || xQueueSend(s_q, e, 0) != pdTRUE) s_ev_dropped++;
}

void telem_detect(int64_t t_win_end_us, int64_t t_detect_us, float score, float score_raw, int64_t infer_us)
{
    ev_t e = {.kind = EV_DETECT};
    e.det.t_win_end = t_win_end_us;
    e.det.t_detect = t_detect_us;
    e.det.score = score;
    e.det.score_raw = score_raw;
    e.det.infer_us = infer_us;
    post(&e);
}

void telem_link(int64_t t_wake_us, int64_t t_wifi_start_us, int64_t t_assoc_us, int64_t t_ip_us,
                int64_t t_sock_us, int64_t t_first_tx_us, bool ok)
{
    ev_t e = {.kind = EV_LINK};
    e.link.t_wake = t_wake_us;
    e.link.t_start = t_wifi_start_us;
    e.link.t_assoc = t_assoc_us;
    e.link.t_ip = t_ip_us;
    e.link.t_sock = t_sock_us;
    e.link.t_first_tx = t_first_tx_us;
    e.link.ok = ok;
    post(&e);
}

void telem_stream_end(uint32_t bytes, uint32_t packets, uint32_t audio_ms, uint32_t total_ms, bool ok,
                      const char *reason)
{
    ev_t e = {.kind = EV_STREAM_END};
    e.end.bytes = bytes;
    e.end.packets = packets;
    e.end.audio_ms = audio_ms;
    e.end.total_ms = total_ms;
    e.end.ok = ok;
    strlcpy(e.end.reason, reason ? reason : "", sizeof(e.end.reason));
    post(&e);
}

void telem_transcript(const char *text)
{
    ev_t e = {.kind = EV_TRANSCRIPT};
    strlcpy(e.text, text ? text : "", sizeof(e.text));
    post(&e);
}

// --- printing (telemetry task only) ---------------------------------------------
static char s_line[768];

static void emit(void)
{
    // One fputs of a whole line: stdout's lock keeps it in one piece even
    // when ESP_LOG prints from another task at the same moment.
    fputs(s_line, stdout);
    fflush(stdout);
}

// JSON string body with quotes, backslashes and control chars escaped.
static void json_str(char *dst, size_t cap, const char *s)
{
    size_t o = 0;
    for (; *s && o + 7 < cap; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') { dst[o++] = '\\'; dst[o++] = (char)c; }
        else if (c < 0x20) o += (size_t)snprintf(dst + o, cap - o, "\\u%04x", c);
        else dst[o++] = (char)c;
    }
    dst[o] = '\0';
}

static void print_event(const ev_t *e)
{
    char buf[200];
    switch (e->kind) {
    case EV_DETECT: {
        const double l1 = (e->det.t_detect - e->det.t_win_end) / 1000.0;
        snprintf(s_line, sizeof(s_line),
                 "{\"type\":\"detect\",\"t_win_end_us\":%lld,\"t_detect_us\":%lld,\"score\":%.3f,"
                 "\"score_raw\":%.3f,\"smooth_n\":%d,\"infer_us\":%lld,\"l1_ms\":%.1f}\n",
                 e->det.t_win_end, e->det.t_detect, e->det.score, e->det.score_raw, KWS_SMOOTH_N,
                 e->det.infer_us, l1);
        // a new wake: forget the previous one's link/stream/transcript
        g_last.score = e->det.score;
        g_last.infer_ms = e->det.infer_us / 1000.0f;
        g_last.l1_ms = (float)l1;
        g_last.t_detect_us = e->det.t_detect;
        g_last.wifi_ms = g_last.l2_ms = NAN;
        g_last.link_done = g_last.link_ok = g_last.stream_done = g_last.stream_ok = false;
        g_last.bytes = g_last.packets = g_last.audio_ms = 0;
        portENTER_CRITICAL(&s_last_lock);
        g_last.text[0] = g_last.reason[0] = '\0';
        portEXIT_CRITICAL(&s_last_lock);
        g_last.gen = g_last.gen + 1;
        break;
    }
    case EV_LINK: {
        const double wifi = e->link.t_ip ? (e->link.t_ip - e->link.t_start) / 1000.0 : -1.0;
        const double l2 = e->link.t_sock ? (e->link.t_sock - e->link.t_wake) / 1000.0 : -1.0;
        snprintf(s_line, sizeof(s_line),
                 "{\"type\":\"link\",\"t_wake_us\":%lld,\"t_wifi_start_us\":%lld,\"t_assoc_us\":%lld,"
                 "\"t_ip_us\":%lld,\"t_sock_us\":%lld,\"t_first_tx_us\":%lld,\"ok\":%s,\"wifi_ms\":%.1f,"
                 "\"l2_ms\":%.1f,\"transport\":\"%s\",\"deinit_idle\":%d}\n",
                 e->link.t_wake, e->link.t_start, e->link.t_assoc, e->link.t_ip, e->link.t_sock,
                 e->link.t_first_tx, e->link.ok ? "true" : "false", wifi, l2, LINK_UDP ? "udp" : "http",
                 WIFI_DEINIT_WHEN_IDLE);
        g_last.wifi_ms = wifi < 0 ? NAN : (float)wifi;
        g_last.l2_ms = l2 < 0 ? NAN : (float)l2;
        g_last.link_ok = e->link.ok;
        g_last.link_done = true;
        g_last.gen = g_last.gen + 1;
        break;
    }
    case EV_STREAM_END:
        snprintf(s_line, sizeof(s_line),
                 "{\"type\":\"stream_end\",\"bytes\":%lu,\"packets\":%lu,\"audio_ms\":%lu,\"total_ms\":%lu,"
                 "\"ok\":%s,\"reason\":\"%s\",\"ring_ovf\":%lu}\n",
                 (unsigned long)e->end.bytes, (unsigned long)e->end.packets, (unsigned long)e->end.audio_ms,
                 (unsigned long)e->end.total_ms, e->end.ok ? "true" : "false", e->end.reason,
                 (unsigned long)asr_overruns());
        g_last.bytes = e->end.bytes;
        g_last.packets = e->end.packets;
        g_last.audio_ms = e->end.audio_ms;
        g_last.stream_ok = e->end.ok;
        g_last.t_end_us = esp_timer_get_time();
        portENTER_CRITICAL(&s_last_lock);
        strlcpy(g_last.reason, e->end.reason, sizeof(g_last.reason));
        portEXIT_CRITICAL(&s_last_lock);
        g_last.stream_done = true;
        g_last.gen = g_last.gen + 1;
        break;
    case EV_TRANSCRIPT:
        json_str(buf, sizeof(buf), e->text);
        snprintf(s_line, sizeof(s_line), "{\"type\":\"transcript\",\"text\":\"%s\"}\n", buf);
        portENTER_CRITICAL(&s_last_lock);
        strlcpy(g_last.text, e->text, sizeof(g_last.text));
        portEXIT_CRITICAL(&s_last_lock);
        g_last.gen = g_last.gen + 1;
        break;
    }
    emit();
}

// --- RAM ------------------------------------------------------------------------
extern int _data_start, _data_end, _bss_start, _bss_end, _noinit_start, _noinit_end;
#define RAM_CAPS (MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)

static size_t ram_static(void)
{
    return (size_t)((char *)&_data_end - (char *)&_data_start) +
           (size_t)((char *)&_bss_end - (char *)&_bss_start) +
           (size_t)((char *)&_noinit_end - (char *)&_noinit_start);
}

// --- identity -------------------------------------------------------------------
// FNV-1a over every tunable that changes behaviour, so a session file says
// exactly which configuration produced it.
uint32_t telem_config_hash(void)
{
    static uint32_t h;
    if (h) return h;
    char s[384];
    snprintf(s, sizeof(s),
             "%d|%.2f|%d|%.1f|%.3f|%.3f|%.1f|%d|%d|%.3f|%.3f|%d|%d|%d|%.3f|%s|%d"
             "|%d|%d|%d|%d|%d|%d|%d|%d|%d|%d|%d|%d|%d|%d|%d",
             GATE_ENABLED, GATE0_RATIO_DB, GATE0_FRAMES, GATE0_ABS_MIN_DB, GATE_FLOOR_DOWN_ALPHA,
             GATE_FLOOR_UP_DB, GATE_FLOOR_INIT_DB, GATE1_BAND_LO_HZ, GATE1_BAND_HI_HZ,
             GATE1_BAND_RATIO, GATE1_FLATNESS, GATE1_FRAMES, GATE_HANGOVER_MS,
             KWS_PAUSE_WHILE_STREAMING, KWS_THRESHOLD, MODEL_SHA256, CONFIG_VK_SAMPLE_SHIFT,
             KWS_STRIDE_FRAMES, KWS_SMOOTH_N, CONFIG_VK_REFRACTORY_MS, MIC_DC_HPF, LINK_UDP, PREROLL_MS,
             RING_MS, STREAM_MIN_MS, ENDPOINT_SILENCE_MS, STREAM_MAX_MS, WIFI_DEINIT_WHEN_IDLE,
             VK_WIFI_ON_DEMAND, OLED_ENABLED, OLED_OFF_IN_IDLE, OLED_IDLE_REFRESH_MS);
    uint32_t x = 2166136261u;
    for (const char *p = s; *p; p++) x = (x ^ (uint8_t)*p) * 16777619u;
    h = x ? x : 1;
    return h;
}

const char *telem_fw_hash(void)
{
    static char fw[12];
    if (!fw[0]) esp_app_get_elf_sha256(fw, sizeof(fw));
    return fw;
}

void telem_copy_last(char *text, size_t tn, char *reason, size_t rn)
{
    portENTER_CRITICAL(&s_last_lock);
    if (text) strlcpy(text, g_last.text, tn);
    if (reason) strlcpy(reason, g_last.reason, rn);
    portEXIT_CRITICAL(&s_last_lock);
}

// --- the task -------------------------------------------------------------------
#define N_TRACK 4                      // idle0, idle1, capture, infer
static TaskHandle_t s_track[N_TRACK];

static uint64_t runtime(TaskHandle_t h)
{
    if (!h) return 0;
    TaskStatus_t st;
    vTaskGetInfo(h, &st, pdFALSE, eInvalid);
    return (uint64_t)st.ulRunTimeCounter;
}

const char *telem_state_name(void)
{
    switch (asr_state()) {
    case ASR_LINKING: return "LINKING";
    case ASR_STREAMING: return "STREAMING";
    default: return g_live.speech ? "SPEECH" : "LISTEN";
    }
}

static void telem_task(void *arg)
{
    (void)arg;
    const char *fw = telem_fw_hash();
    const esp_app_desc_t *app = esp_app_get_description();
    const uint32_t cfg = telem_config_hash();
    const size_t st_ram = ram_static();
    g_metrics.ram_static = (uint32_t)st_ram;

    snprintf(s_line, sizeof(s_line),
             "{\"type\":\"boot\",\"firmware_hash\":\"%s\",\"version\":\"%s\",\"config_hash\":\"%08lx\","
             "\"model_sha256\":\"%.16s\",\"threshold\":%.3f,\"gate\":%d,\"cpu_mhz\":%d,\"ram_static\":%u,"
             "\"heap_total\":%u,\"arena_used\":%u,\"arena_size\":%u,\"model_bytes\":%u,"
             "\"stride_frames\":%d,\"smooth_n\":%d,\"link\":\"%s\",\"wifi_policy\":\"%s\",\"deinit_idle\":%d,"
             "\"oled\":%d,\"preroll_ms\":%d,\"ring_ms\":%d}\n",
             fw, app->version, (unsigned long)cfg, MODEL_SHA256, KWS_THRESHOLD, GATE_ENABLED,
             CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ, (unsigned)st_ram,
             (unsigned)heap_caps_get_total_size(RAM_CAPS), (unsigned)g_live.arena_used,
             (unsigned)g_live.arena_size, (unsigned)g_live.model_bytes, KWS_STRIDE_FRAMES, KWS_SMOOTH_N,
             LINK_UDP ? "udp" : "http", VK_WIFI_ON_DEMAND ? "ON_DEMAND" : "WARM", WIFI_DEINIT_WHEN_IDLE,
             OLED_ENABLED, PREROLL_MS, RING_MS);
    emit();

    // ring of the last TELEM_CPU_WINDOW_TICKS+1 snapshots -> 1 s CPU window
    enum { W = TELEM_CPU_WINDOW_TICKS + 1 };
    uint64_t rt[W][N_TRACK] = {{0}};
    int64_t wall[W] = {0};
    int k = 0, filled = 0;
    TickType_t last_wake = xTaskGetTickCount();

    for (;;) {
        // events first, until the next tick is due
        TickType_t due = last_wake + pdMS_TO_TICKS(TELEM_TICK_MS);
        ev_t e;
        for (;;) {
            TickType_t now = xTaskGetTickCount();
            TickType_t wait = (TickType_t)(due - now);
            if ((int32_t)wait <= 0) break;
            if (xQueueReceive(s_q, &e, wait) == pdTRUE) print_event(&e);
        }
        last_wake = due;

        wall[k] = esp_timer_get_time();
        for (int i = 0; i < N_TRACK; i++) rt[k][i] = runtime(s_track[i]);
        int o = (k + 1) % W;                 // oldest snapshot = window start
        if (filled < W) filled++;
        double cpu[2] = {NAN, NAN}, cap = NAN, inf = NAN;
        double win_us = (double)(wall[k] - wall[o]);
        if (filled == W && win_us > 0) {
            for (int c = 0; c < 2; c++) {
                double idle = (double)(rt[k][c] - rt[o][c]) / win_us;
                cpu[c] = 100.0 * (1.0 - (idle > 1 ? 1 : idle));
            }
            cap = 100.0 * (double)(rt[k][2] - rt[o][2]) / win_us;
            inf = 100.0 * (double)(rt[k][3] - rt[o][3]) / win_us;
        }
        k = (k + 1) % W;

        portENTER_CRITICAL(&s_mic_lock);
        int peak = s_mic_peak;
        uint64_t sq = s_mic_sq;
        uint32_t n = s_mic_n;
        s_mic_peak = 0;
        s_mic_sq = 0;
        s_mic_n = 0;
        portEXIT_CRITICAL(&s_mic_lock);
        double rms_db = n ? 20 * log10(sqrt((double)sq / n) / 32768.0 + 1e-9) : NAN;
        double peak_db = 20 * log10(peak / 32768.0 + 1e-9);

        size_t heap_total = heap_caps_get_total_size(RAM_CAPS);
        size_t heap_free = heap_caps_get_free_size(RAM_CAPS);
        size_t heap_min = heap_caps_get_minimum_free_size(RAM_CAPS);
        float smax = g_live.score_max;
        g_live.score_max = 0;

        // same numbers for the OLED
        g_metrics.cpu0 = (float)cpu[0];
        g_metrics.cpu1 = (float)cpu[1];
        g_metrics.cpu_avg = (float)((cpu[0] + cpu[1]) / 2);
        g_metrics.dscnn_duty = (float)inf;
        g_metrics.ram_used = (uint32_t)(st_ram + heap_total - heap_free);
        g_metrics.ram_peak = (uint32_t)(st_ram + heap_total - heap_min);
        g_metrics.mic_rms_db = isnan(rms_db) ? NAN : (float)rms_db;
        g_metrics.noise_floor_db = g_gate.floor_db;
        g_metrics.frames = g_gate.frames;
        g_metrics.s0 = g_gate.s0;
        g_metrics.s1 = g_gate.s1;

        // NaN is not JSON: print null for "not measured yet" (console shows —)
        char c0[12], c1[12], ca[12], cc[12], ci[12];
#define NUM(buf, v) (isnan(v) ? strcpy(buf, "null") : (snprintf(buf, sizeof(buf), "%.2f", (v)), buf))
        NUM(c0, cpu[0]);
        NUM(c1, cpu[1]);
        NUM(ca, (cpu[0] + cpu[1]) / 2);
        NUM(cc, cap);
        NUM(ci, inf);
#undef NUM
        snprintf(s_line, sizeof(s_line),
                 "{\"type\":\"tick\",\"t_us\":%lld,\"state\":\"%s\",\"radio\":\"%s\","
                 "\"cpu0\":%s,\"cpu1\":%s,\"cpu_avg\":%s,\"cpu_capture\":%s,\"dscnn_duty\":%s,"
                 "\"ram_static\":%u,\"ram_heap_used\":%u,\"ram_used\":%u,\"ram_peak\":%u,\"heap_free\":%u,"
                 "\"heap_largest\":%u,\"arena_used\":%u,\"model_bytes\":%u,"
                 "\"mic_peak\":%.1f,\"mic_rms\":%.1f,\"noise_floor\":%.1f,\"score\":%.3f,\"score_max\":%.3f,"
                 "\"threshold\":%.2f,\"gate\":%d,"
                 "\"funnel\":{\"frames\":%lu,\"s0\":%lu,\"s1\":%lu,\"opens\":%lu,\"infer\":%lu,\"det\":%lu},"
                 "\"backfill\":%lu,\"backfill_miss\":%lu,\"ring_ovf\":%lu,\"ev_dropped\":%lu,"
                 "\"firmware_hash\":\"%s\",\"config_hash\":\"%08lx\"}\n",
                 esp_timer_get_time(), telem_state_name(), asr_state() == ASR_IDLE ? "OFF" : "ON",
                 c0, c1, ca, cc, ci,
                 (unsigned)st_ram, (unsigned)(heap_total - heap_free), (unsigned)(st_ram + heap_total - heap_free),
                 (unsigned)(st_ram + heap_total - heap_min), (unsigned)heap_free,
                 (unsigned)heap_caps_get_largest_free_block(RAM_CAPS), (unsigned)g_live.arena_used,
                 (unsigned)g_live.model_bytes,
                 peak_db, isnan(rms_db) ? -200.0 : rms_db, g_gate.floor_db, g_live.score, smax,
                 KWS_THRESHOLD, GATE_ENABLED,
                 (unsigned long)g_gate.frames, (unsigned long)g_gate.s0, (unsigned long)g_gate.s1,
                 (unsigned long)g_gate.opens, (unsigned long)g_live.infer, (unsigned long)g_live.det,
                 (unsigned long)g_live.backfill_rows, (unsigned long)g_live.backfill_miss,
                 (unsigned long)asr_overruns(), (unsigned long)s_ev_dropped, fw, (unsigned long)cfg);
        emit();
    }
}

void telem_start(TaskHandle_t capture, TaskHandle_t infer)
{
    s_track[0] = xTaskGetIdleTaskHandleForCore(0);
    s_track[1] = xTaskGetIdleTaskHandleForCore(1);
    s_track[2] = capture;
    s_track[3] = infer;
    s_q = xQueueCreate(8, sizeof(ev_t));
    // Core 0, low priority: core 0 is idle while the gate is closed, and
    // printing (~5 ms per tick at 921600 baud) must never delay capture on core 1.
    xTaskCreatePinnedToCore(telem_task, "telem", 4 * 1024, NULL, 2, NULL, 0);
}

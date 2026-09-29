// ui_oled.c — see ui_oled.h.

#include "ui_oled.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "driver/i2c_master.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "asr_uploader.h"
#include "config.h"
#include "ssd1306.h"
#include "telemetry.h"

static const char *TAG = "ui";

#define RAM_LIMIT_KB 256
#define CPU_LIMIT_PCT 10.0f
// Rows are formatted wide and clipped to 21 chars by ssd1306_line(): a long
// value loses its tail on the panel, never corrupts memory.
#define ROW_CHARS 64

typedef enum { SCR_NONE = -1, SCR_LISTEN, SCR_STREAM, SCR_RESULT } screen_t;

static char s_rows[SSD1306_PAGES][ROW_CHARS];    // what is on the panel now

// "-" for anything not measured yet (Rule 0.1 on the device itself)
static const char *num(char *b, size_t n, float v, const char *fmt)
{
    if (isnan(v)) snprintf(b, n, "-");
    else snprintf(b, n, fmt, v);
    return b;
}

static const char *verdict(float v, float limit)
{
    return isnan(v) ? "-" : (v < limit ? "PASS" : "FAIL");
}

static void render_listen(char r[SSD1306_PAGES][ROW_CHARS])
{
    char a[12], b[12];
    const vk_metrics_t *m = &g_metrics;
    const float ram_kb = m->ram_used ? m->ram_used / 1024.0f : NAN;
    const float peak_kb = m->ram_peak ? m->ram_peak / 1024.0f : NAN;
    const float s0 = m->frames ? 100.0f * m->s0 / m->frames : NAN;
    const float s1 = m->frames ? 100.0f * m->s1 / m->frames : NAN;
    const int64_t up = esp_timer_get_time() / 1000000;

    snprintf(r[0], sizeof(r[0]), "VEKTORA %s RF:OFF", g_live.speech ? "SPEECH" : "LISTEN");
    snprintf(r[1], sizeof(r[1]), "CPU %5s%% <10 %s", num(a, sizeof(a), m->cpu_avg, "%.1f"),
             verdict(m->cpu_avg, CPU_LIMIT_PCT));
    snprintf(r[2], sizeof(r[2]), "RAM %s/256KB %s", num(a, sizeof(a), ram_kb, "%.0f"),
             verdict(ram_kb, RAM_LIMIT_KB));
    snprintf(r[3], sizeof(r[3]), "PEAK %sKB DUTY %s%%", num(a, sizeof(a), peak_kb, "%.0f"),
             num(b, sizeof(b), m->dscnn_duty, "%.1f"));
    snprintf(r[4], sizeof(r[4]), "GATE %s%%>%s%%>%lu", num(a, sizeof(a), s0, "%.1f"),
             num(b, sizeof(b), s1, "%.1f"), (unsigned long)g_live.det);
    snprintf(r[5], sizeof(r[5]), "WAKES %lu L1 %sMS", (unsigned long)g_live.det,
             num(a, sizeof(a), g_last.l1_ms, "%.0f"));
    snprintf(r[6], sizeof(r[6]), "MIC %sDB FLR %s", num(a, sizeof(a), m->mic_rms_db, "%.0f"),
             num(b, sizeof(b), m->noise_floor_db, "%.0f"));
    snprintf(r[7], sizeof(r[7]), "UP %02lld:%02lld:%02lld", up / 3600, up / 60 % 60, up % 60);
}

static void render_stream(char r[SSD1306_PAGES][ROW_CHARS])
{
    char a[12], b[12];
    const bool linking = asr_state() == ASR_LINKING;
    snprintf(r[0], sizeof(r[0]), "WAKE SCORE %s", num(a, sizeof(a), g_last.score, "%.2f"));
    snprintf(r[1], sizeof(r[1]), "INFER %sMS L1 %sMS", num(a, sizeof(a), g_last.infer_ms, "%.0f"),
             num(b, sizeof(b), g_last.l1_ms, "%.0f"));
    snprintf(r[2], sizeof(r[2]), "WIFI %sMS RF:ON", num(a, sizeof(a), g_last.wifi_ms, "%.0f"));
    snprintf(r[3], sizeof(r[3]), "L2 %sMS", num(a, sizeof(a), g_last.l2_ms, "%.0f"));
    if (linking) snprintf(r[4], sizeof(r[4]), "LINKING...");
    else snprintf(r[4], sizeof(r[4]), "STREAM %.1fS", asr_sent_ms() / 1000.0f);
    snprintf(r[5], sizeof(r[5]), "RING OVF %lu", (unsigned long)asr_overruns());
    snprintf(r[6], sizeof(r[6]), "LINK %s", LINK_UDP ? "UDP" : "HTTP");
    r[7][0] = '\0';
}

static void render_result(char r[SSD1306_PAGES][ROW_CHARS])
{
    char text[64], reason[24], a[12];
    telem_copy_last(text, sizeof(text), reason, sizeof(reason));
    for (int i = 0; i < SSD1306_PAGES; i++) r[i][0] = '\0';

    if (!g_last.link_ok) {
        snprintf(r[0], sizeof(r[0]), "LINK FAIL");
        snprintf(r[1], sizeof(r[1]), "%s", reason);
        snprintf(r[2], sizeof(r[2]), "LED ONLY, NO TEXT");
    } else if (!text[0]) {
        snprintf(r[0], sizeof(r[0]), "(NO TRANSCRIPT)");
    } else {
        // word-wrap the transcript over rows 0-2
        const char *p = text;
        for (int row = 0; row < 3 && *p; row++) {
            while (*p == ' ') p++;
            int n = (int)strlen(p);
            if (n > SSD1306_COLS) {
                n = SSD1306_COLS;
                while (n > 0 && p[n] != ' ') n--;
                if (n == 0) n = SSD1306_COLS;
            }
            memcpy(r[row], p, (size_t)n);
            r[row][n] = '\0';
            p += n;
        }
    }
    snprintf(r[3], sizeof(r[3]), "L2 WIFI %sMS", num(a, sizeof(a), g_last.wifi_ms, "%.0f"));
    snprintf(r[4], sizeof(r[4]), "SENT %.1fS PKT %lu", g_last.audio_ms / 1000.0f, (unsigned long)g_last.packets);
    snprintf(r[5], sizeof(r[5]), "OVF %lu %s", (unsigned long)asr_overruns(), reason);
    snprintf(r[6], sizeof(r[6]), "SCORE %s", num(a, sizeof(a), g_last.score, "%.2f"));
    snprintf(r[7], sizeof(r[7]), "BACK TO LISTEN");
}

static screen_t pick(void)
{
    const asr_state_t st = asr_state();
    if (st == ASR_LINKING || st == ASR_STREAMING) return SCR_STREAM;
    if (g_last.stream_done && esp_timer_get_time() - g_last.t_end_us < (int64_t)OLED_RESULT_HOLD_MS * 1000)
        return SCR_RESULT;
    return SCR_LISTEN;
}

static void ui_task(void *arg)
{
    (void)arg;
    i2c_master_bus_config_t bus_cfg = {
        .i2c_port = I2C_NUM_0,
        .sda_io_num = OLED_SDA_GPIO,
        .scl_io_num = OLED_SCL_GPIO,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    i2c_master_bus_handle_t bus;
    if (i2c_new_master_bus(&bus_cfg, &bus) != ESP_OK || ssd1306_init(bus, OLED_ADDR) != ESP_OK) {
        ESP_LOGW(TAG, "no SSD1306 at 0x%02x on SDA=%d SCL=%d - running without the OLED", OLED_ADDR,
                 OLED_SDA_GPIO, OLED_SCL_GPIO);
        vTaskDelete(NULL);
        return;
    }
    ESP_LOGI(TAG, "OLED up (SDA=%d SCL=%d addr 0x%02x)", OLED_SDA_GPIO, OLED_SCL_GPIO, OLED_ADDR);

    screen_t shown = SCR_NONE;
    uint32_t gen = 0;
    int64_t last_draw = 0;
    bool panel_on = true;
    static char next[SSD1306_PAGES][ROW_CHARS];

    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(100));
        const screen_t scr = pick();
        const int64_t now = esp_timer_get_time();
        const int64_t every = (scr == SCR_LISTEN ? OLED_IDLE_REFRESH_MS : OLED_BUSY_REFRESH_MS) * 1000LL;
        const bool changed = scr != shown || g_last.gen != gen;
        if (!changed && now - last_draw < every) continue;

        const bool want_on = !(OLED_OFF_IN_IDLE && scr == SCR_LISTEN);
        if (want_on != panel_on) {
            ssd1306_power(want_on);
            panel_on = want_on;
        }
        shown = scr;
        gen = g_last.gen;
        last_draw = now;
        if (!panel_on) continue;

        if (scr == SCR_LISTEN) render_listen(next);
        else if (scr == SCR_STREAM) render_stream(next);
        else render_result(next);
        // only rows that changed go over I2C (~3 ms each at 400 kHz)
        for (int i = 0; i < SSD1306_PAGES; i++) {
            if (strcmp(next[i], s_rows[i]) != 0) {
                ssd1306_line((uint8_t)i, next[i]);
                strcpy(s_rows[i], next[i]);
            }
        }
    }
}

void ui_start(void)
{
#if OLED_ENABLED
    for (int i = 0; i < SSD1306_PAGES; i++) s_rows[i][0] = '\x01';   // force the first draw
    xTaskCreatePinnedToCore(ui_task, "ui", 4 * 1024, NULL, 1, NULL, 0);
#endif
}

// main.cc — Vektora OLED CPU test (ESP32-WROOM-32 DevKit V1).
//
// Standalone rehearsal for CLAUDE.md §13.7 Phase B ("idle CPU" work) and for
// checking that a 0.96" SSD1306 OLED can be driven from this board before
// wiring it into the real firmware. No microphone, no WiFi, no ASR — just:
//
//   the real Run 3 int8 model, invoked back-to-back on core 0
//     -> per-core CPU load, read from the FreeRTOS run-time counters
//        (same technique as firmware's spec_task, §12.8)
//     -> printed once a second over serial, and drawn on the OLED
//
// The model's input tensor is filled once with a fixed pattern (the
// quantized zero point) and never touched again. Invoke() does the same
// amount of work regardless of the input values, so this measures the
// model's real CPU cost without needing a microphone or real audio.
//
// Wiring, confirmed 2026-09-28: OLED GND->GND, VCC->3.3V, SDA->GPIO21,
// SCL->GPIO22 (see main/Kconfig.projbuild to change pins without editing
// this file). The board is the same ESP32-WROOM-32 DevKit V1 used in
// firmware (§12: INMP441 wiring on GPIO 32/25/26 -- unrelated to
// this project, which does not touch the microphone).

#include <stdio.h>
#include <string.h>

#include "driver/i2c_master.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "ssd1306.h"

static const char *TAG = "oled_cpu_test";

// 2 s, not 0.5 s: one Invoke() takes ~568 ms (CLAUDE.md §12.6 measured 564 ms
// on this same board). A sampling period close to that length makes the
// run-time-counter deltas alias against Invoke() boundaries -- a snapshot
// landing mid-invoke can attribute >100% or <0% to a task for that single
// window. A period several invokes long averages that out.
#define DISPLAY_PERIOD_MS 2000

// --- model -------------------------------------------------------------
static tflite::MicroInterpreter *s_interp;
static size_t s_arena_size, s_arena_used;
static TfLiteTensor *s_in;

static void model_start(void)
{
    const tflite::Model *model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "model schema %lu != %d", (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        abort();
    }
    // Exactly the ops the Run 3 model uses (firmware/main.cc).
    static tflite::MicroMutableOpResolver<5> ops;
    ops.AddConv2D();
    ops.AddMaxPool2D();
    ops.AddMean();
    ops.AddFullyConnected();
    ops.AddSoftmax();

    // Same arena-clamping trick as firmware/main.cc model_start():
    // the arena must be ONE contiguous block, and the largest free block on
    // this chip is well under the total free heap (CLAUDE.md §12.2 item 1).
    const size_t want = CONFIG_OLED_ARENA_KB * 1024;
    const size_t largest = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    const size_t margin = 24 * 1024;
    size_t arena_size = want;
    if (largest < want + margin)
        arena_size = largest > margin ? largest - margin : 0;
    ESP_LOGI(TAG, "heap: free %u, largest block %u -> arena %u bytes",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL),
             (unsigned)largest, (unsigned)arena_size);
    uint8_t *arena = arena_size ? (uint8_t *)heap_caps_malloc(arena_size, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL) : NULL;
    if (!arena) {
        ESP_LOGE(TAG, "no memory for a %u byte arena", (unsigned)arena_size);
        abort();
    }
    s_arena_size = arena_size;
    static tflite::MicroInterpreter interp(model, ops, arena, arena_size);
    if (interp.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed (arena too small?)");
        abort();
    }
    s_arena_used = interp.arena_used_bytes();
    s_interp = &interp;
    s_in = interp.input(0);
    ESP_LOGI(TAG, "model %u bytes, sha256 %.16s..., arena used %u / %u bytes",
             g_model_len, MODEL_SHA256, (unsigned)s_arena_used, (unsigned)arena_size);

    // Fixed input: we are measuring the model's CPU cost, not classifying
    // anything, so there is no need for a microphone. zero_point is a valid
    // in-range quantized value for every input this model was trained on.
    memset(s_in->data.int8, s_in->params.zero_point, s_in->bytes);
}

// --- shared state between the two tasks ---------------------------------
static volatile int64_t s_nn_us_sum, s_nn_count, s_nn_us_max;
static portMUX_TYPE s_lock = portMUX_INITIALIZER_UNLOCKED;
static TaskHandle_t s_infer_h;
static bool s_oled_ok;

static void infer_task(void *)
{
    for (;;) {
        int64_t t0 = esp_timer_get_time();
        if (s_interp->Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "Invoke failed");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }
        int64_t dt = esp_timer_get_time() - t0;
        portENTER_CRITICAL(&s_lock);
        s_nn_us_sum += dt;
        s_nn_count++;
        if (dt > s_nn_us_max) s_nn_us_max = dt;
        portEXIT_CRITICAL(&s_lock);
        // Let the scheduler and (on other boards) the watchdog breathe.
        // Same pattern as firmware/main.cc infer_task.
        vTaskDelay(1);
    }
}

// CPU load from the FreeRTOS run-time counters, not from our own timings —
// same technique and same reasoning as firmware's spec_task
// (CLAUDE.md §12.8): both cores run the same wall time, so the sum of every
// task's delta is twice one core's, which self-calibrates the counter.
static void display_task(void *)
{
    const int NCORE = portNUM_PROCESSORS;
    const UBaseType_t CAP = 32;
    TaskStatus_t *st = (TaskStatus_t *)malloc(CAP * sizeof(TaskStatus_t));
    TaskHandle_t idle[2];
    for (int c = 0; c < NCORE; c++) idle[c] = xTaskGetIdleTaskHandleForCore(c);

    uint64_t prev_idle[2] = {0, 0}, prev_inf = 0, prev_total = 0;

    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(DISPLAY_PERIOD_MS));

        UBaseType_t n = uxTaskGetSystemState(st, CAP, NULL);
        uint64_t total = 0, cur_idle[2] = {0, 0}, cur_inf = 0;
        for (UBaseType_t i = 0; i < n; i++) {
            uint64_t rt = (uint64_t)st[i].ulRunTimeCounter;
            total += rt;
            for (int c = 0; c < NCORE; c++)
                if (st[i].xHandle == idle[c]) cur_idle[c] = rt;
            if (st[i].xHandle == s_infer_h) cur_inf = rt;
        }
        double d_total = (double)(total - prev_total);
        double per_core = d_total / NCORE;
        double busy[2] = {0, 0};
        for (int c = 0; c < NCORE; c++)
            busy[c] = per_core > 0 ? 100.0 * (1.0 - (double)(cur_idle[c] - prev_idle[c]) / per_core) : 0;
        double inf_pct = per_core > 0 ? 100.0 * (double)(cur_inf - prev_inf) / per_core : 0;

        portENTER_CRITICAL(&s_lock);
        int64_t nn_us = s_nn_us_sum, nn_n = s_nn_count, nn_max = s_nn_us_max;
        s_nn_us_sum = s_nn_count = s_nn_us_max = 0;
        portEXIT_CRITICAL(&s_lock);

        double avg_ms = nn_n ? nn_us / 1000.0 / nn_n : 0.0;
        double rate = nn_n / (DISPLAY_PERIOD_MS / 1000.0);
        unsigned heap = (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);

        ESP_LOGI(TAG, "cpu: core0 %.1f%% (infer %.1f%%) | core1 %.1f%% | invoke %.0f ms avg (max %.0f) x%.1f/s | heap %u",
                 busy[0], inf_pct, busy[1], avg_ms, nn_max / 1000.0, rate, heap);

        if (s_oled_ok) {
            char line[24];
            double c0 = busy[0] < 0 ? 0 : (busy[0] > 99.9 ? 99.9 : busy[0]);
            double c1 = busy[1] < 0 ? 0 : (busy[1] > 99.9 ? 99.9 : busy[1]);
            ssd1306_line(0, "VEKTORA CPU");
            snprintf(line, sizeof(line), "CORE0 %4.1f%%", c0);
            ssd1306_line(1, line);
            snprintf(line, sizeof(line), "CORE1 %4.1f%%", c1);
            ssd1306_line(2, line);
            snprintf(line, sizeof(line), "INFER %3.0fMS", avg_ms);
            ssd1306_line(3, line);
            snprintf(line, sizeof(line), "RATE %4.1f/S", rate);
            ssd1306_line(4, line);
            snprintf(line, sizeof(line), "HEAP %6u", heap);
            ssd1306_line(5, line);
        }

        for (int c = 0; c < NCORE; c++) prev_idle[c] = cur_idle[c];
        prev_inf = cur_inf;
        prev_total = total;
    }
}

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Vektora OLED CPU test -- SDA=%d SCL=%d addr=0x%02x, arena<=%d KB",
             CONFIG_OLED_I2C_SDA_GPIO, CONFIG_OLED_I2C_SCL_GPIO, CONFIG_OLED_I2C_ADDR, CONFIG_OLED_ARENA_KB);

    i2c_master_bus_config_t bus_cfg = {
        .i2c_port = -1,
        .sda_io_num = (gpio_num_t)CONFIG_OLED_I2C_SDA_GPIO,
        .scl_io_num = (gpio_num_t)CONFIG_OLED_I2C_SCL_GPIO,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags = {.enable_internal_pullup = true},
    };
    i2c_master_bus_handle_t bus;
    ESP_ERROR_CHECK(i2c_new_master_bus(&bus_cfg, &bus));

    esp_err_t oled_err = ssd1306_init(bus, CONFIG_OLED_I2C_ADDR);
    s_oled_ok = (oled_err == ESP_OK);
    if (!s_oled_ok) {
        ESP_LOGE(TAG, "SSD1306 init failed (%s) -- check wiring/address (Kconfig OLED_I2C_ADDR)."
                      " Continuing with serial-only output.", esp_err_to_name(oled_err));
    } else {
        ssd1306_line(0, "VEKTORA CPU");
        ssd1306_line(1, "STARTING...");
    }

    model_start();

    // Core 0: the model, back to back, nothing else. Core 1: the display
    // task, which wakes twice a second -- everything else there is the idle
    // task, so core1's busy% here is essentially "how much this test itself
    // costs", which should stay tiny.
    xTaskCreatePinnedToCore(infer_task, "infer", 8 * 1024, NULL, 5, &s_infer_h, 0);
    xTaskCreatePinnedToCore(display_task, "display", 4 * 1024, NULL, 1, NULL, 1);
}

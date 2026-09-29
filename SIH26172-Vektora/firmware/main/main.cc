// main.cc — Vektora keyword spotter (ESP32-S3; first written for the WROOM-32).
//
//   core 1  capture_task  I2S 10 ms hop -> ASR ring (asr_feed)
//                         -> Stage 0 energy gate (every hop, µs)
//                         -> if Stage 0 passed or the gate is open: log-mel row
//                            (4.1 ms on the WROOM-32) -> Stage 1 voice gate -> feature ring
//   core 0  infer_task    sleeps while the gate is closed. On open: backfill
//                         the feature rows the gate skipped (from the ASR
//                         ring), then run the int8 model every KWS_STRIDE_FRAMES
//                         until the gate closes. Mean of the last KWS_SMOOTH_N
//                         P(keyword) >= threshold -> LED first, then the
//                         detect event, then the uploader (asr_uploader.c).
//           asr_task      on a wake: WiFi on -> UDP stream -> WiFi off
//           telem_task    2 Hz JSON tick + events over USB (telemetry.c)
//           ui_task       OLED (ui_oled.c), lowest priority
//
// Why the gate: features + model running continuously keep both cores busy
// with nobody talking (the model alone was 564 ms per inference on the
// WROOM-32). With the gate, a quiet room costs one energy sum per 10 ms.
//
// Why backfill: the gate needs ~70 ms of voice to open, so the keyword onset
// arrives BEFORE features start. The ASR ring holds the audio anyway, so the
// missing rows are recomputed on core 0 (idle at that point) and the model
// sees the same 1 s window it would have without the gate. Every row is
// tagged with its frame number so a stale row can't be mistaken for a fresh one.
//
// Frame f = samples [160 f, 160 f + 400) since boot (features.py framing);
// the 98 most recent frames are exactly the last second, so streaming rows
// give the same input as features.py on that second.

#include <inttypes.h>
#include <math.h>
#include <string.h>

#include "driver/gpio.h"
#include "driver/i2s_std.h"
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

#include "feature_extract.h"
#include "model_data.h"

#include "asr_uploader.h"
#include "config.h"
#include "gate.h"
#include "led.h"
#include "telemetry.h"
#include "ui_oled.h"
#include "wifi_manager.h"

static const char *TAG = "vektora";

// --- audio ------------------------------------------------------------------
static i2s_chan_handle_t s_rx;
static int s_slot;   // 0 = LEFT, 1 = RIGHT: which half of each stereo frame is the mic

static i2s_std_config_t i2s_cfg(i2s_slot_mode_t mode)
{
    i2s_std_config_t std = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(FE_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, mode),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = (gpio_num_t)CONFIG_VK_I2S_SCK_GPIO,
            .ws = (gpio_num_t)CONFIG_VK_I2S_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din = (gpio_num_t)CONFIG_VK_I2S_SD_GPIO,
            .invert_flags = {.mclk_inv = false, .bclk_inv = false, .ws_inv = false},
        },
    };
    return std;
}

// Boot-time mic self-test. Reads both slots for ~0.3 s and returns the one
// whose top 24 bits carry signal. Falls back to the Kconfig choice when
// neither or both do (both = floating SD line).
static i2s_std_slot_mask_t mic_probe(void)
{
    i2s_std_slot_mask_t fallback =
#if CONFIG_VK_I2S_RIGHT_CHANNEL
        I2S_STD_SLOT_RIGHT;
#else
        I2S_STD_SLOT_LEFT;
#endif
    i2s_chan_handle_t rx;
    i2s_chan_config_t chan = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chan, NULL, &rx));
    i2s_std_config_t std = i2s_cfg(I2S_SLOT_MODE_STEREO);
    std.slot_cfg.slot_mask = I2S_STD_SLOT_BOTH;
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx, &std));
    // Weak pull-up during the probe only: an undriven SD then reads all 1s,
    // which tells "mic not driving" apart from "SD held low".
    gpio_pullup_en((gpio_num_t)CONFIG_VK_I2S_SD_GPIO);
    ESP_ERROR_CHECK(i2s_channel_enable(rx));

    static int32_t buf[2 * 256];
    int64_t n[2] = {0}, lowbits[2] = {0}, zeros[2] = {0};
    double sum[2] = {0}, sum2[2] = {0};
    int32_t first[2] = {0}, varies[2] = {0};
    for (int blk = 0; blk < 20; blk++) {  // 20 x 256 frames = 0.32 s
        size_t got = 0;
        i2s_channel_read(rx, buf, sizeof(buf), &got, pdMS_TO_TICKS(200));
        if (blk < 2) continue;            // skip mic power-up settle
        if (blk == 2 || blk == 12)        // raw words, to diagnose wiring by eye
            for (int i = 0; i < 16; i += 2)
                ESP_LOGI(TAG, "mic raw L %08lx  R %08lx", (unsigned long)buf[i], (unsigned long)buf[i + 1]);
        for (int i = 0; i < (int)(got / sizeof(int32_t)); i++) {
            int c = i & 1;
            int32_t v = buf[i];
            if (n[c] == 0) first[c] = v;
            if (v != first[c]) varies[c] = 1;
            if (v & 0xFF) lowbits[c]++;
            if (v == 0) zeros[c]++;
            double s = (double)(v >> 8);  // 24-bit sample
            sum[c] += s;
            sum2[c] += s * s;
            n[c]++;
        }
    }
    i2s_channel_disable(rx);
    i2s_del_channel(rx);
    gpio_pullup_dis((gpio_num_t)CONFIG_VK_I2S_SD_GPIO);

    // A driven slot carries AC in its top 24 bits; an undriven one is flat
    // there. The low byte is NOT a usable test on the S3: it comes back random
    // even from a working INMP441 (SD tri-states after bit 24). A floating SD
    // line shows as noise in BOTH slots, so "both have signal" = suspect.
    bool ok[2];
    for (int c = 0; c < 2; c++) {
        double mean = n[c] ? sum[c] / n[c] : 0;
        double var = n[c] ? sum2[c] / n[c] - mean * mean : 0;
        double rms_db = 20.0 * log10(sqrt(var > 0 ? var : 0) / 8388608.0 + 1e-12);
        ok[c] = n[c] > 0 && varies[c] && rms_db > -110.0;
        ESP_LOGI(TAG, "mic probe %s: n=%lld zeros=%lld lowbyte!=0 %lld, dc %.0f, ac rms %.1f dBFS -> %s",
                 c ? "RIGHT" : "LEFT", (long long)n[c], (long long)zeros[c], (long long)lowbits[c],
                 mean, rms_db, ok[c] ? "signal" : "silent");
    }
    if (ok[0] && !ok[1]) return I2S_STD_SLOT_LEFT;
    if (ok[1] && !ok[0]) return I2S_STD_SLOT_RIGHT;
    if (ok[0] && ok[1]) ESP_LOGW(TAG, "mic probe: both slots have signal - floating SD? check wiring");
    else ESP_LOGW(TAG, "mic probe: no slot has signal - check wiring/L-R pin");
    return fallback;
}

static void i2s_start(void)
{
    i2s_std_slot_mask_t slot = mic_probe();

    i2s_chan_config_t chan = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chan, NULL, &s_rx));

    // The INMP441 sends 24-bit data in a 32-bit slot, in one channel only
    // (chosen by its L/R pin). Read STEREO and pick the slot in software:
    // on the S3, MONO + slot_mask RIGHT returned the silent LEFT slot
    // (measured 2026-09-29), while stereo reads are unambiguous.
    i2s_std_config_t std = i2s_cfg(I2S_SLOT_MODE_STEREO);
    std.slot_cfg.slot_mask = I2S_STD_SLOT_BOTH;
    s_slot = slot == I2S_STD_SLOT_RIGHT ? 1 : 0;
    ESP_LOGI(TAG, "mic: reading %s slot (stereo read)", s_slot ? "RIGHT" : "LEFT");
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_rx, &std));
    ESP_ERROR_CHECK(i2s_channel_enable(s_rx));
}

// Read exactly n samples, converted to int16 the way the training data was.
static void read_samples(int16_t *out, int n)
{
    static int32_t raw[2 * FE_HOP];   // stereo frames: [L, R, L, R, ...]
#if MIC_DC_HPF
    static float hp_x, hp_y;          // y[n] = x[n] - x[n-1] + 0.995 y[n-1], ~13 Hz corner
#endif
    while (n > 0) {
        int want = n < FE_HOP ? n : FE_HOP;
        size_t got = 0;
        ESP_ERROR_CHECK(i2s_channel_read(s_rx, raw, 2 * want * sizeof(int32_t), &got, portMAX_DELAY));
        int m = (int)(got / (2 * sizeof(int32_t)));
        for (int i = 0; i < m; i++) {
            int32_t v = raw[2 * i + s_slot] >> CONFIG_VK_SAMPLE_SHIFT;
#if MIC_DC_HPF
            const float x = (float)v;
            hp_y = x - hp_x + 0.995f * hp_y;
            hp_x = x;
            v = (int32_t)lrintf(hp_y);
#endif
            out[i] = (int16_t)(v > 32767 ? 32767 : (v < -32768 ? -32768 : v));
        }
        out += m;
        n -= m;
    }
}

// --- model ------------------------------------------------------------------
static tflite::MicroInterpreter *s_interp;
static size_t s_arena_size, s_arena_used;
static TfLiteTensor *s_in, *s_out;

static void model_start(void)
{
    const tflite::Model *model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "model schema %" PRIu32 " != %d", model->version(), TFLITE_SCHEMA_VERSION);
        abort();
    }
    // Exactly the ops quantize.py reported. A missing op fails AllocateTensors.
    static tflite::MicroMutableOpResolver<5> ops;
    ops.AddConv2D();
    ops.AddMaxPool2D();
    ops.AddMean();
    ops.AddFullyConnected();
    ops.AddSoftmax();

    // The arena must be ONE contiguous block. The ESP32's internal RAM is
    // split into regions and partly taken by WiFi/BT buffers and the stack, so
    // the largest free block is well under the total free heap. Ask for the
    // configured size, but never more than the biggest block minus a margin
    // for everything else that still has to allocate.
    const size_t want = CONFIG_VK_ARENA_KB * 1024;
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
    // Probe how much of the arena the model really uses, then hand the rest
    // back to the heap (CLAUDE.md §5.4): the ASR ring and WiFi need it more.
    size_t used;
    {
        tflite::MicroInterpreter probe(model, ops, arena, arena_size);
        if (probe.AllocateTensors() != kTfLiteOk) {
            ESP_LOGE(TAG, "AllocateTensors failed (arena too small?)");
            abort();
        }
        used = probe.arena_used_bytes();
    }
    const size_t tight = (used + 1024 + 15) & ~(size_t)15;   // + alignment slack
    if (tight < arena_size) {
        heap_caps_free(arena);
        arena = (uint8_t *)heap_caps_malloc(tight, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        if (!arena) {
            ESP_LOGE(TAG, "could not re-allocate a %u byte arena", (unsigned)tight);
            abort();
        }
        arena_size = tight;
    }
    s_arena_size = arena_size;
    static tflite::MicroInterpreter interp(model, ops, arena, arena_size);
    if (interp.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed on the %u byte trimmed arena", (unsigned)arena_size);
        abort();
    }
    s_arena_used = interp.arena_used_bytes();
    s_interp = &interp;
    s_in = interp.input(0);
    s_out = interp.output(0);
    ESP_LOGI(TAG, "model %u bytes, sha256 %.16s..., arena used %u / %u bytes",
             g_model_len, MODEL_SHA256, (unsigned)interp.arena_used_bytes(), (unsigned)arena_size);
    ESP_LOGI(TAG, "input scale %f zp %d | output scale %f zp %d",
             s_in->params.scale, (int)s_in->params.zero_point,
             s_out->params.scale, (int)s_out->params.zero_point);
    // The C constants and the model must agree, or features are mis-scaled.
    if (fabsf(s_in->params.scale - KWS_INPUT_SCALE) > 1e-6f ||
        s_in->params.zero_point != KWS_INPUT_ZERO_POINT ||
        s_in->bytes != (size_t)(FE_N_FRAMES * FE_N_MELS)) {
        ESP_LOGE(TAG, "feature_tables.h does not match the model: re-run export_c.py");
        abort();
    }
}

// --- LED --------------------------------------------------------------------
static void led_set(bool on)
{
    if (CONFIG_VK_LED_GPIO >= 0) gpio_set_level((gpio_num_t)CONFIG_VK_LED_GPIO, on);
}

extern "C" void vk_led_set(bool on)
{
    led_set(on);
}

extern "C" void vk_led_blink(int n)
{
    for (int i = 0; i < n; i++) {
        led_set(false); vTaskDelay(pdMS_TO_TICKS(100));
        led_set(true);  vTaskDelay(pdMS_TO_TICKS(100));
    }
    led_set(false);
}

// --- shared state between the two tasks --------------------------------------
//
// Feature ring: 98 rows, row slot = frame % 98, each tagged with the frame it
// holds. capture_task writes rows (only while features run), infer_task reads
// them and backfills missing ones. s_frames (frames completed) and the tags
// change together under s_ring_lock, so a snapshot is always consistent.
// Frame numbers are 32-bit: one load on this core, 248 days at 10 ms.
static int8_t s_ring[FE_N_FRAMES * FE_N_MELS];
static int32_t s_row_frame[FE_N_FRAMES];             // frame held by each slot, -1 = none
static int64_t s_read_us[FE_N_FRAMES];               // when that frame's last hop was read
static volatile int32_t s_frames;                    // frames completed since boot
static portMUX_TYPE s_ring_lock = portMUX_INITIALIZER_UNLOCKED;
static TaskHandle_t s_capture_h, s_infer_h;

static void quantize_row(const float *db, int8_t *row)
{
    for (int b = 0; b < FE_N_MELS; b++)
        row[b] = fe_quantize((db[b] - FE_NORM_MEAN[b]) / FE_NORM_STD[b]);
}

static void capture_task(void *)
{
    static int16_t window[FE_WIN];                   // last 400 samples
    float db[FE_N_MELS];
    int8_t row[FE_N_MELS];
    int16_t hop[FE_HOP];

    // Prime the window. These samples go into the ASR ring too, so ring
    // sample index == frame sample index (frame f starts at sample 160 f).
    read_samples(window, FE_WIN - FE_HOP);
    asr_feed(window, FE_WIN - FE_HOP);

    for (int32_t f = 0;; f++) {
        read_samples(hop, FE_HOP);
        const int64_t t_read = esp_timer_get_time();
        asr_feed(hop, FE_HOP);                       // tap for the ASR uploader (append-only)
        memmove(window, window + FE_HOP, (FE_WIN - FE_HOP) * sizeof(int16_t));
        memcpy(window + FE_WIN - FE_HOP, hop, FE_HOP * sizeof(int16_t));

        uint64_t sq = 0;
        int peak = 0;
        for (int i = 0; i < FE_HOP; i++) {
            int32_t v = hop[i];
            sq += (uint64_t)(v * v);
            int a = v < 0 ? -v : v;
            if (a > peak) peak = a;
        }
        telem_mic(peak, sq, FE_HOP);

        // Stage 0 always (the whole idle cost); features only when needed.
        const float e_db = 10.0f * log10f((float)sq / FE_HOP / (32768.0f * 32768.0f) + 1e-12f);
        const bool s0 = gate_stage0(e_db);
        const bool feat = !GATE_ENABLED || s0 || gate_is_open(f);
        if (feat) {
            fe_frame_logmel(window, db);
#if GATE_ENABLED
            if (s0) {
                if (gate_stage1(db, f)) xTaskNotifyGive(s_infer_h);   // gate just opened
            } else {
                gate_stage1_reset();
            }
#endif
            quantize_row(db, row);
        } else {
            gate_stage1_reset();
        }

        const int slot = f % FE_N_FRAMES;
        portENTER_CRITICAL(&s_ring_lock);
        if (feat) {
            memcpy(s_ring + slot * FE_N_MELS, row, FE_N_MELS);
            s_row_frame[slot] = f;
        }
        s_read_us[slot] = t_read;
        s_frames = f + 1;
        portEXIT_CRITICAL(&s_ring_lock);
        g_live.speech = !GATE_ENABLED || gate_is_open(f);
    }
}

// Recompute rows of the window ending at frame fh that the gate skipped.
// Runs on core 0 while capture keeps going on core 1. Returns rows computed.
static int backfill(int32_t fh)
{
    static int16_t pcm[FE_WIN];
    float db[FE_N_MELS];
    int8_t row[FE_N_MELS];
    int n = 0;
    for (int32_t t = fh - FE_N_FRAMES + 1; t <= fh; t++) {
        const int slot = t % FE_N_FRAMES;
        if (s_row_frame[slot] >= t) continue;        // present (or already newer)
        if (!asr_ring_read((uint64_t)t * FE_HOP, pcm, FE_WIN)) {
            g_live.backfill_miss = g_live.backfill_miss + 1;
            continue;
        }
        fe_frame_logmel(pcm, db);
        quantize_row(db, row);
        portENTER_CRITICAL(&s_ring_lock);
        if (s_row_frame[slot] < t) {                 // capture didn't get there first
            memcpy(s_ring + slot * FE_N_MELS, row, FE_N_MELS);
            s_row_frame[slot] = t;
        }
        portEXIT_CRITICAL(&s_ring_lock);
        n++;
    }
    g_live.backfill_rows = g_live.backfill_rows + n;
    return n;
}

static void infer_task(void *)
{
    int64_t last_fire_us = -(int64_t)CONFIG_VK_REFRACTORY_MS * 1000;
    int64_t led_off_us = 0;
    int32_t last_head = -1000000;                    // window head of the last Invoke
    float hist[KWS_SMOOTH_N];                        // last P(keyword) values
    int n_hist = 0, i_hist = 0;

    for (;;) {
        // LED: on at the wake, off once 500 ms passed AND the upload is over.
        if (led_off_us && esp_timer_get_time() >= led_off_us && !asr_is_busy()) {
            led_set(false);
            led_off_us = 0;
        }

        const int32_t fh = s_frames - 1;
        if (fh < FE_N_FRAMES - 1) {                  // not a full second yet
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }
        if (GATE_ENABLED && !gate_is_open(fh)) {
            g_live.score = 0;
            n_hist = 0;                              // smoothing restarts per utterance
            // Sleep until capture opens the gate. Short timeout only while
            // the LED still has to be switched off.
            ulTaskNotifyTake(pdTRUE, led_off_us ? pdMS_TO_TICKS(50) : pdMS_TO_TICKS(1000));
            continue;
        }
        if (KWS_PAUSE_WHILE_STREAMING && asr_is_busy()) {
            n_hist = 0;
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }
        // Stride: a new decision only once KWS_STRIDE_FRAMES new rows exist.
        // (Right after the gate opens last_head is old, so this passes at once.)
        if (fh - last_head < KWS_STRIDE_FRAMES) {
            vTaskDelay(1);
            continue;
        }

        // Backfill until the latest window is complete (2nd pass catches rows
        // that went stale while the 1st ran; normally 0 rows).
        for (int pass = 0; pass < 3 && backfill(s_frames - 1) > 0; pass++) {
        }

        // Snapshot the newest window, oldest row first.
        int8_t *dst = s_in->data.int8;
        portENTER_CRITICAL(&s_ring_lock);
        const int32_t head = s_frames - 1;
        const int64_t t_win_end = s_read_us[head % FE_N_FRAMES];
        for (int32_t t = head - FE_N_FRAMES + 1; t <= head; t++)
            memcpy(dst + (t - (head - FE_N_FRAMES + 1)) * FE_N_MELS, s_ring + (t % FE_N_FRAMES) * FE_N_MELS,
                   FE_N_MELS);
        portEXIT_CRITICAL(&s_ring_lock);

        last_head = head;

        const int64_t t1 = esp_timer_get_time();
        if (s_interp->Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "Invoke failed");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }
        const int64_t now = esp_timer_get_time();
        g_live.infer = g_live.infer + 1;
        g_live.infer_us_last = now - t1;

        const float pk = KWS_OUTPUT_SCALE * (s_out->data.int8[KWS_KEYWORD_INDEX] - KWS_OUTPUT_ZERO_POINT);
        g_live.score = pk;
        if (pk > g_live.score_max) g_live.score_max = pk;

        // posterior smoothing (§5.4): mean of the last KWS_SMOOTH_N scores
        hist[i_hist] = pk;
        i_hist = (i_hist + 1) % KWS_SMOOTH_N;
        if (n_hist < KWS_SMOOTH_N) n_hist++;
        float smooth = 0;
        for (int i = 0; i < n_hist; i++) smooth += hist[i];
        smooth /= (float)n_hist;

        if (smooth >= KWS_THRESHOLD && now - last_fire_us >= (int64_t)CONFIG_VK_REFRACTORY_MS * 1000) {
            led_set(true);                           // FIRST: before any logging or network
            const int64_t t_detect = esp_timer_get_time();
            last_fire_us = t_detect;
            led_off_us = t_detect + 500 * 1000;
            n_hist = 0;
            g_live.det = g_live.det + 1;
            telem_detect(t_win_end, t_detect, smooth, pk, now - t1);
            asr_capture_and_send(t_win_end, t_detect, smooth);
            ESP_LOGW(TAG, ">>> VEKTORA  p=%.3f (raw %.3f)  L1 %.0f ms", smooth, pk, (t_detect - t_win_end) / 1000.0);
        }
        // Let the idle task run briefly, so the task watchdog stays happy.
        vTaskDelay(1);
    }
}

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Vektora KWS, threshold %.2f, gate %d, mic SCK=%d WS=%d SD=%d shift=%d",
             KWS_THRESHOLD, GATE_ENABLED, CONFIG_VK_I2S_SCK_GPIO, CONFIG_VK_I2S_WS_GPIO,
             CONFIG_VK_I2S_SD_GPIO, CONFIG_VK_SAMPLE_SHIFT);
    if (CONFIG_VK_LED_GPIO >= 0) {
        gpio_reset_pin((gpio_num_t)CONFIG_VK_LED_GPIO);
        gpio_set_direction((gpio_num_t)CONFIG_VK_LED_GPIO, GPIO_MODE_OUTPUT);
        // Boot self-test: 3 blinks so the wiring can be checked by eye.
        for (int i = 0; i < 3; i++) {
            led_set(true);  vTaskDelay(pdMS_TO_TICKS(150));
            led_set(false); vTaskDelay(pdMS_TO_TICKS(150));
        }
    }
    for (int i = 0; i < FE_N_FRAMES; i++) s_row_frame[i] = -1;
    fe_init();
    gate_init();
    model_start();
    g_live.arena_used = s_arena_used;
    g_live.arena_size = s_arena_size;
    g_live.model_bytes = g_model_len;
    i2s_start();
    // After model_start(): the arena is sized first, from the largest free
    // block, so anything that allocates earlier can starve it.
    wifi_init();
    asr_init();
    // Core 1: audio + gate + features, must never fall behind. Core 0: model.
    xTaskCreatePinnedToCore(infer_task, "infer", 8 * 1024, NULL, 4, &s_infer_h, 0);
    xTaskCreatePinnedToCore(capture_task, "capture", 12 * 1024, NULL, 6, &s_capture_h, 1);
    telem_start(s_capture_h, s_infer_h);
    ui_start();
}

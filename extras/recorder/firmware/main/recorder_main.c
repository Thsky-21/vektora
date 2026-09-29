// Vektora recorder firmware.
//
// Job: capture the INMP441 exactly the way the keyword-spotter firmware does
// (16 kHz, left slot, 32-bit I2S sample >> VKR_SAMPLE_SHIFT -> int16), and
// stream it to the PC over USB: the chip's native USB port on the ESP32-S3
// (USB-Serial-JTAG), or UART0 through the board's USB-to-UART bridge. The PC script (../record.py)
// turns the stream into WAV files.
//
// Why a custom packet format instead of just dumping raw bytes:
// the old dataset was damaged by recordings with DROPPED SAMPLES (runs of
// exact zeros, CLAUDE.md §10.6). A raw byte stream cannot tell you when that
// happens. Every packet here therefore carries:
//   - a sequence number    -> the PC detects packets lost on the USB link;
//   - an overflow counter  -> the PC detects audio lost inside the ESP32
//                             (the I2S DMA queue filled up before we read it);
//   - a CRC                -> the PC detects corrupted bytes.
// record.py never joins audio across a gap: a take containing one is thrown
// away and re-recorded. That is how chopped files are prevented, not just
// detected afterwards.
//
// Packet layout (little-endian, 338 bytes, one per 10 ms):
//   0  char[4]  "VKA1"
//   4  u32      seq         packet counter since boot
//   8  u32      ovf         I2S receive-queue overflow events since boot
//  12  u16      n           samples in this packet (always 160)
//  14  u16      flags       bits 0-7: sample shift; bit 8: right channel
//  16  i16[n]   samples     16 kHz mono PCM
//  16+2n u16    crc         CRC-16/CCITT-FALSE over bytes 0 .. 16+2n-1

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s_std.h"
#include "driver/uart.h"
#include "driver/usb_serial_jtag.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_attr.h"
#include "sdkconfig.h"

#define SAMPLE_RATE   16000
#define FRAME         160          // 10 ms, same hop as the feature extractor
#define WARMUP_FRAMES 50           // drop the first 0.5 s: DMA priming + mic start-up
#define UART_PORT     UART_NUM_0   // the port wired to the board's USB-to-UART bridge

static const char *TAG = "vkrec";

// A Kconfig bool that is switched off is not defined at all, so give it a value.
#ifdef CONFIG_VKR_I2S_RIGHT_CHANNEL
#define VKR_RIGHT 1
#else
#define VKR_RIGHT 0
#endif
#ifdef CONFIG_VKR_USE_USB_JTAG
#define VKR_USB 1
#else
#define VKR_USB 0
#endif

typedef struct __attribute__((packed)) {
    char     magic[4];
    uint32_t seq;
    uint32_t ovf;
    uint16_t n;
    uint16_t flags;
    int16_t  pcm[FRAME];
    uint16_t crc;
} packet_t;

static i2s_chan_handle_t s_rx;
static volatile uint32_t s_ovf;

// Called from the I2S interrupt when the driver's receive queue is full, i.e.
// a DMA buffer was overwritten before we read it. That audio is gone.
static bool IRAM_ATTR on_rx_overflow(i2s_chan_handle_t h, i2s_event_data_t *e, void *ctx)
{
    s_ovf++;
    return false;
}

static void i2s_start(void)
{
    i2s_chan_config_t chan = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    // 8 buffers x 256 samples = 128 ms of slack before audio would be lost.
    chan.dma_desc_num = 8;
    chan.dma_frame_num = 256;
    ESP_ERROR_CHECK(i2s_new_channel(&chan, NULL, &s_rx));

    // Identical to the keyword-spotter firmware: the INMP441 sends 24-bit data
    // left-aligned in a 32-bit slot, on the channel chosen by its L/R pin.
    i2s_std_config_t std = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = (gpio_num_t)CONFIG_VKR_I2S_SCK_GPIO,
            .ws = (gpio_num_t)CONFIG_VKR_I2S_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din = (gpio_num_t)CONFIG_VKR_I2S_SD_GPIO,
            .invert_flags = {.mclk_inv = false, .bclk_inv = false, .ws_inv = false},
        },
    };
#if VKR_RIGHT
    std.slot_cfg.slot_mask = I2S_STD_SLOT_RIGHT;
#else
    std.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
#endif
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_rx, &std));

    i2s_event_callbacks_t cbs = {.on_recv_q_ovf = on_rx_overflow};
    ESP_ERROR_CHECK(i2s_channel_register_event_callback(s_rx, &cbs, NULL));
    ESP_ERROR_CHECK(i2s_channel_enable(s_rx));
}

// Read exactly FRAME samples, converted to int16 exactly as main.cc does.
static void read_frame(int16_t *out)
{
    static int32_t raw[FRAME];
    int n = 0;
    while (n < FRAME) {
        size_t got = 0;
        ESP_ERROR_CHECK(i2s_channel_read(s_rx, raw + n, (FRAME - n) * sizeof(int32_t), &got, portMAX_DELAY));
        n += (int)(got / sizeof(int32_t));
    }
    for (int i = 0; i < FRAME; i++) {
        int32_t v = raw[i] >> CONFIG_VKR_SAMPLE_SHIFT;
        out[i] = (int16_t)(v > 32767 ? 32767 : (v < -32768 ? -32768 : v));
    }
}

// CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF). record.py uses the same one.
static uint16_t crc16(const uint8_t *p, size_t len)
{
    uint16_t crc = 0xFFFF;
    while (len--) {
        crc ^= (uint16_t)(*p++) << 8;
        for (int b = 0; b < 8; b++)
            crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
    }
    return crc;
}

void app_main(void)
{
    // Human-readable banner at the boot baud rate (visible in `idf.py monitor`).
    ESP_LOGI(TAG, "Vektora recorder: SCK=%d WS=%d SD=%d, %s channel, shift %d, stream over %s",
             CONFIG_VKR_I2S_SCK_GPIO, CONFIG_VKR_I2S_WS_GPIO, CONFIG_VKR_I2S_SD_GPIO,
             VKR_RIGHT ? "right" : "left", CONFIG_VKR_SAMPLE_SHIFT,
             VKR_USB ? "native USB" : "UART0");
    vTaskDelay(pdMS_TO_TICKS(100));   // let the banner leave the UART FIFO

#if CONFIG_VKR_LED_GPIO >= 0
    gpio_reset_pin((gpio_num_t)CONFIG_VKR_LED_GPIO);
    gpio_set_direction((gpio_num_t)CONFIG_VKR_LED_GPIO, GPIO_MODE_OUTPUT);
#endif

    // From here on the stream port carries binary audio only. Any stray log
    // line would corrupt the stream (the CRC would catch it, but the audio
    // would be lost), so logging is switched off completely.
    esp_log_level_set("*", ESP_LOG_NONE);
#if VKR_USB
    usb_serial_jtag_driver_config_t usb = {.rx_buffer_size = 256, .tx_buffer_size = 16384};
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&usb));
#else
    ESP_ERROR_CHECK(uart_driver_install(UART_PORT, 256, 16384, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_set_baudrate(UART_PORT, CONFIG_VKR_BAUD));
#endif

    i2s_start();

    static packet_t pkt;
    memcpy(pkt.magic, "VKA1", 4);
    pkt.n = FRAME;
    pkt.flags = (uint16_t)(CONFIG_VKR_SAMPLE_SHIFT & 0xFF) | (VKR_RIGHT ? 0x100 : 0);

    // Read into an aligned buffer, then copy into the packed packet.
    static int16_t pcm[FRAME];
    for (int i = 0; i < WARMUP_FRAMES; i++) read_frame(pcm);
    s_ovf = 0;   // overflows during warm-up don't matter: that audio is discarded anyway

    uint32_t seq = 0;
    for (;;) {
        read_frame(pcm);
        memcpy(pkt.pcm, pcm, sizeof(pcm));
        pkt.seq = seq++;
        pkt.ovf = s_ovf;
        pkt.crc = crc16((const uint8_t *)&pkt, offsetof(packet_t, crc));
#if VKR_USB
        // If no program on the PC has the port open, the USB buffer fills and
        // this write times out: the packet is dropped and the audio loop keeps
        // running. The PC sees the jump in seq and never joins across it.
        usb_serial_jtag_write_bytes(&pkt, sizeof(pkt), pdMS_TO_TICKS(5));
#else
        // Blocks only if the 16 KB TX buffer is full, which cannot happen at
        // 921600 baud (the link drains ~3x faster than audio arrives).
        uart_write_bytes(UART_PORT, &pkt, sizeof(pkt));
#endif
#if CONFIG_VKR_LED_GPIO >= 0
        gpio_set_level((gpio_num_t)CONFIG_VKR_LED_GPIO, (seq % 100) < 10);   // 100 ms blink each second
#endif
    }
}

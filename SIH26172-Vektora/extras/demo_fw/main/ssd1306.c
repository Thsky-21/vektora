// ssd1306.c — see ssd1306.h.
#include "ssd1306.h"
#include <string.h>
#include "esp_log.h"

static const char *TAG = "ssd1306";
static i2c_master_dev_handle_t s_dev;

// --- low level ---------------------------------------------------------

static esp_err_t cmd1(uint8_t c)
{
    uint8_t buf[2] = {0x00, c};   // control byte 0x00 = command stream
    return i2c_master_transmit(s_dev, buf, sizeof(buf), 100);
}

static esp_err_t data(const uint8_t *d, size_t n)
{
    // control byte 0x40 = data stream, then the pixel bytes. n is at most
    // SSD1306_WIDTH (128) in this driver, so a 129-byte stack buffer is fine.
    uint8_t buf[SSD1306_WIDTH + 1];
    buf[0] = 0x40;
    memcpy(buf + 1, d, n);
    return i2c_master_transmit(s_dev, buf, n + 1, 100);
}

esp_err_t ssd1306_init(i2c_master_bus_handle_t bus, uint8_t addr7)
{
    i2c_device_config_t dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = addr7,
        .scl_speed_hz = 400000,
    };
    esp_err_t err = i2c_master_bus_add_device(bus, &dev_cfg, &s_dev);
    if (err != ESP_OK) return err;

    static const uint8_t init_seq[] = {
        0xAE,             // display off
        0xD5, 0x80,       // clock divide / oscillator freq
        0xA8, 0x3F,       // multiplex ratio = 64
        0xD3, 0x00,       // display offset = 0
        0x40,             // display start line = 0
        0x8D, 0x14,       // charge pump on (internal Vcc)
        0x20, 0x02,       // memory addressing mode = page
        0xA1,             // segment remap (SEG127 = column 0)
        0xC8,             // COM scan direction, remapped
        0xDA, 0x12,       // COM pin hardware config for 128x64
        0x81, 0xCF,       // contrast
        0xD9, 0xF1,       // pre-charge period
        0xDB, 0x40,       // VCOMH deselect level
        0xA4,             // resume RAM content display (not "all on")
        0xA6,             // normal (not inverted)
        0xAF,             // display on
    };
    for (size_t i = 0; i < sizeof(init_seq); i++) {
        err = cmd1(init_seq[i]);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "init cmd 0x%02x failed at addr 0x%02x: %s",
                     init_seq[i], addr7, esp_err_to_name(err));
            return err;
        }
    }
    ssd1306_clear();
    return ESP_OK;
}

static void set_page_col(uint8_t page, uint8_t col)
{
    cmd1(0xB0 | (page & 0x07));           // set page address
    cmd1(0x00 | (col & 0x0F));            // set lower column nibble
    cmd1(0x10 | ((col >> 4) & 0x0F));     // set higher column nibble
}

void ssd1306_clear(void)
{
    uint8_t zeros[SSD1306_WIDTH] = {0};
    for (uint8_t p = 0; p < SSD1306_PAGES; p++) {
        set_page_col(p, 0);
        data(zeros, SSD1306_WIDTH);
    }
}

// --- 5x7 font ------------------------------------------------------------
//
// Hand-built for exactly the characters main.cc prints (see the readout
// strings there): space . % / the digits, and A C E F H I K M N O P R S T U V.
// Each glyph is 5 columns; column c, bit r (r=0 top .. r=6 bottom) is one
// pixel. That is the native SSD1306 page byte layout, so no transform is
// needed before i2c_master_transmit. Bit 7 is always 0, which is what gives
// text rows a 1px gap from the row below.
struct glyph { char c; uint8_t col[5]; };

static const struct glyph FONT[] = {
    {' ', {0x00, 0x00, 0x00, 0x00, 0x00}},
    {'.', {0x00, 0x00, 0x40, 0x00, 0x00}},
    {':', {0x00, 0x00, 0x12, 0x00, 0x00}},
    {'%', {0x63, 0x13, 0x08, 0x64, 0x62}},
    {'/', {0x40, 0x30, 0x08, 0x06, 0x01}},
    {'0', {0x3E, 0x51, 0x49, 0x45, 0x3E}},
    {'1', {0x00, 0x42, 0x7F, 0x40, 0x00}},
    {'2', {0x42, 0x61, 0x51, 0x49, 0x46}},
    {'3', {0x22, 0x41, 0x49, 0x49, 0x36}},
    {'4', {0x18, 0x14, 0x12, 0x7F, 0x10}},
    {'5', {0x2F, 0x49, 0x49, 0x49, 0x31}},
    {'6', {0x3C, 0x4A, 0x49, 0x49, 0x30}},
    {'7', {0x01, 0x71, 0x09, 0x05, 0x03}},
    {'8', {0x36, 0x49, 0x49, 0x49, 0x36}},
    {'9', {0x06, 0x49, 0x49, 0x29, 0x1E}},
    {'A', {0x7E, 0x09, 0x09, 0x09, 0x7E}},
    {'C', {0x3E, 0x41, 0x41, 0x41, 0x22}},
    {'E', {0x7F, 0x49, 0x49, 0x41, 0x41}},
    {'F', {0x7F, 0x09, 0x09, 0x01, 0x01}},
    {'H', {0x7F, 0x08, 0x08, 0x08, 0x7F}},
    {'I', {0x41, 0x41, 0x7F, 0x41, 0x41}},
    {'K', {0x7F, 0x08, 0x14, 0x22, 0x41}},
    {'M', {0x7F, 0x02, 0x04, 0x02, 0x7F}},
    {'N', {0x7F, 0x02, 0x04, 0x08, 0x7F}},
    {'O', {0x3E, 0x41, 0x41, 0x41, 0x3E}},
    {'P', {0x7F, 0x09, 0x09, 0x09, 0x06}},
    {'R', {0x7F, 0x09, 0x19, 0x29, 0x46}},
    {'S', {0x46, 0x49, 0x49, 0x49, 0x31}},
    {'T', {0x01, 0x01, 0x7F, 0x01, 0x01}},
    {'U', {0x3F, 0x40, 0x40, 0x40, 0x3F}},
    {'V', {0x0F, 0x30, 0x40, 0x30, 0x0F}},
{'B', {0x7F, 0x49, 0x49, 0x49, 0x36}},    {'D', {0x7F, 0x41, 0x41, 0x22, 0x1C}},    {'G', {0x3E, 0x41, 0x49, 0x49, 0x7A}},    {'L', {0x7F, 0x40, 0x40, 0x40, 0x40}},    {'W', {0x3F, 0x40, 0x38, 0x40, 0x3F}},    {'X', {0x63, 0x14, 0x08, 0x14, 0x63}},    {'Y', {0x07, 0x08, 0x70, 0x08, 0x07}},    {'-', {0x08, 0x08, 0x08, 0x08, 0x08}},    {'<', {0x08, 0x14, 0x22, 0x41, 0x00}},    {'>', {0x00, 0x41, 0x22, 0x14, 0x08}},    {'=', {0x14, 0x14, 0x14, 0x14, 0x14}},
};
#define FONT_N (sizeof(FONT) / sizeof(FONT[0]))

static const uint8_t *glyph_for(char c)
{
    for (size_t i = 0; i < FONT_N; i++)
        if (FONT[i].c == c) return FONT[i].col;
    return FONT[0].col;   // unsupported character -> blank, not garbage
}

void ssd1306_line(uint8_t page, const char *text)
{
    if (page >= SSD1306_PAGES) return;
    uint8_t row[SSD1306_WIDTH];
    size_t p = 0;
    for (const char *s = text; *s && p + 6 <= SSD1306_WIDTH; s++) {
        const uint8_t *g = glyph_for(*s);
        memcpy(row + p, g, 5);
        row[p + 5] = 0x00;   // 1px inter-character gap
        p += 6;
    }
    while (p < SSD1306_WIDTH) row[p++] = 0x00;
    set_page_col(page, 0);
    data(row, SSD1306_WIDTH);
}

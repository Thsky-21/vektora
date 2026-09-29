// ssd1306.h — minimal driver for a 128x64 SSD1306 OLED over I2C.
//
// Page-addressing mode only: no framebuffer, text only, one 5x7 glyph per
// character with a blank column of spacing. That is all this test needs —
// a live numeric readout of CPU load. Character set is deliberately small
// (see ssd1306.c): space . % / 0-9 and the ~16 uppercase letters the display
// text in main.cc actually uses. ssd1306_putc() silently blanks any other
// character rather than drawing garbage.
#pragma once
#include "driver/i2c_master.h"
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SSD1306_WIDTH   128
#define SSD1306_PAGES   8   // 64 px tall / 8 px per page

// Probes the display at addr (via i2c_master_bus_add_device) and runs the
// init sequence. Returns the underlying esp_err_t (ESP_OK on success).
esp_err_t ssd1306_init(i2c_master_bus_handle_t bus, uint8_t addr7);

// Fills the whole panel with 0 (blank).
void ssd1306_clear(void);

// Writes one line of text at the given page (0..7), left-aligned at column
// 0, space-padded to the full 128 px width so old text never survives a
// shorter new string in the same row.
void ssd1306_line(uint8_t page, const char *text);

#ifdef __cplusplus
}
#endif

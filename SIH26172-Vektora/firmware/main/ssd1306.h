// ssd1306.h — minimal text driver for a 128x64 SSD1306 OLED over I2C.
//
// Page-addressing mode, no framebuffer: 8 text rows of 21 characters (5x7
// glyph + 1 px gap). Same driver as extras/demo_fw (verified on this board's
// SDA=8/SCL=9 wiring), plus a few glyphs and lowercase folded to uppercase.
// Unknown characters draw as blank, never garbage.
#pragma once
#include <stdbool.h>
#include "driver/i2c_master.h"
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SSD1306_WIDTH   128
#define SSD1306_PAGES   8   // 64 px tall / 8 px per page
#define SSD1306_COLS    21  // characters per row

esp_err_t ssd1306_init(i2c_master_bus_handle_t bus, uint8_t addr7);
void ssd1306_clear(void);
// One row of text (0..7), space-padded to the full width.
void ssd1306_line(uint8_t page, const char *text);
// Panel on/off (0xAF/0xAE). Off keeps RAM contents, costs no I2C traffic.
void ssd1306_power(bool on);

#ifdef __cplusplus
}
#endif

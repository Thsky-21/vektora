# OLED CPU test — ESP32-WROOM-32 DevKit V1

Standalone rehearsal for CLAUDE.md §13.7 Phase B ("idle CPU must be < 10% —
today it is about 60%"). Runs the real Run 3 int8 model back-to-back on core
0 (no microphone, no WiFi) and shows the measured per-core CPU load live on
a 0.96" SSD1306 OLED, with the full numeric readout also on serial. It also
doubles as the first check that this board can drive the OLED at all, before
wiring the display into `prototype/firmware`.

```
main/
  main.cc         model setup, the infer loop, the CPU-load measurement,
                   the OLED readout
  ssd1306.c/h      minimal page-mode SSD1306 driver (I2C, 128x64, 5x7 font
                   for the ~30 characters the readout actually uses)
  model_data.cc/h  COPIED VERBATIM from prototype/firmware/main/ -- the same
                   trained Run 3 model (sha256 e9e10d52a17832fb...). If the
                   model is retrained, re-copy these two files.
  Kconfig.projbuild  OLED pins/address, arena size (idf.py menuconfig)
```

## Wiring (confirmed 2026-09-28)

| SSD1306 0.96" OLED | ESP32-WROOM-32 |
|---|---|
| GND | GND |
| VCC | 3.3V |
| SDA | GPIO 21 |
| SCL | GPIO 22 |

Default I2C address is `0x3C` (the address on almost every 0.96" SSD1306
module). If the display stays blank, check the boot log first -- it logs an
`SSD1306 init failed` line with the esp_err_t if the ACK fails, which is the
signature of a wrong address or a wiring fault, not a code bug. Change the
address with `idf.py menuconfig` -> "Vektora OLED CPU test" -> a few boards
use `0x3D` instead.

This project does not touch the INMP441 pins (GPIO 32/25/26 in
`prototype/firmware`) -- it is a different, unrelated peripheral on the same
board.

## Build and flash

ESP-IDF is installed at `C:\Users\User\esp\esp-idf-v5.5.5`. In PowerShell:

```powershell
C:\Users\User\esp\esp-idf-v5.5.5\export.ps1     # once per terminal
cd prototype\oled_test
idf.py set-target esp32                          # once
idf.py build
idf.py -p COMx flash monitor                     # Ctrl+] leaves the monitor
```

(`COMx` was `COM10` for this board's CP2102 bridge as of §12 in the top-level
`CLAUDE.md` -- check Device Manager if it has changed.)

## What it does, and why this is a fair CPU measurement

- `model_start()` loads the exact Run 3 int8 model (same bytes, same
  sha256, same op set as `prototype/firmware`) and fills its input tensor
  **once**, with the quantized zero point. `Invoke()` does the same amount
  of arithmetic regardless of the input values -- the model doesn't know or
  care that the numbers aren't real audio -- so a tight loop of `Invoke()`
  calls measures the model's real CPU cost without needing a microphone.
- `infer_task` (core 0, priority 5) calls `Invoke()` back to back, forever.
  This is deliberately worse than the real firmware, which only runs
  inference while `capture_task` also owns core 1 -- here core 0 is
  saturated on purpose, which is exactly what Phase B's energy gate is
  supposed to prevent in the final design.
- `display_task` (core 1, priority 1) wakes twice a second, reads the
  FreeRTOS per-task run-time counters (`CONFIG_FREERTOS_GENERATE_RUN_TIME_STATS`,
  same technique as `prototype/firmware`'s `spec_task`, CLAUDE.md §12.8),
  computes each core's busy percentage from the delta since the last wake,
  and writes it to both the serial log and the OLED.
- Expect **core0 near 100%** (the model has no gate, no sleep, nothing to
  wait on) and **core1 in the low single digits** (only this task's own I2C
  writes and bookkeeping). That is the number to compare against once the
  energy gate exists: the whole point of Phase B is to make core0 look like
  core1 does here, while idle.

## What the OLED shows

```
VEKTORA CPU
CORE0 XX.X%
CORE1 XX.X%
INFER XXXMS      <- average Invoke() time this period
RATE  X.X/S      <- inferences per second
HEAP  XXXXXX     <- free internal heap, bytes
```

Serial (`idf.py monitor`) prints the same numbers plus the max Invoke() time
seen in each period, once every 500 ms.

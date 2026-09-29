# Firmware: Vektora edge voice activator (ESP32-S3 + INMP441)

ESP-IDF **v5.5.5**, target `esp32s3`. The device listens with the radio off. A
two-stage gate (energy → voice) wakes the int8 keyword model only while
someone is speaking. On "Vektora": LED first, then Wi-Fi on, then the ring
buffer (pre-roll + connect bridge) streams to the server over UDP. Wi-Fi goes
off again when speech ends. Evidence leaves over USB as JSON and is shown on
the OLED.

```
main/
  main.cc              I2S capture (core 1) → ring → gate → features; infer task (core 0):
                       backfill, stride, smoothing, LED-first detect
  gate.c/h             Stage 0 energy + Stage 1 voice gate, hangover, funnel counters
  feature_extract.c/h  log-mel in plain C (PARITY CONTRACT with training/features.py)
  feature_tables.h     GENERATED: window, mel table, norm, quantization, threshold, model sha
  model_data.cc/h      GENERATED: model_int8.tflite as a byte array
  asr_uploader.c/h     the ring + the link: UDP "VAD1" (§8) or HTTP POST /asr fallback
  wifi_manager.c/h     on-demand Wi-Fi: boot probe, cached BSSID/channel/lease, deinit when idle
  telemetry.c/h        2 Hz JSON tick + boot/detect/link/stream_end/transcript events
  ui_oled.c/h          SSD1306 screens: LISTEN (compliance), DETECT/STREAM, RESULT
  ssd1306.c/h          minimal I2C text driver
  led.h                vk_led_set / vk_led_blink
  config.h             EVERY runtime tunable (Rule 7)
  vk_net_config.h      Wi-Fi policy + server port; includes secrets.h
  secrets.example.h    template → copy to secrets.h (gitignored): SSID, password, server IP
  Kconfig.projbuild    pins, sample shift, LED, arena, refractory (idf.py menuconfig)
host_test/
  parity_main.c        laptop build of feature_extract.c for training/parity_check.py
  gate_main.c          laptop build of gate.c for tools/gate_host_check.py
```

## 1. Secrets

```powershell
copy main\secrets.example.h main\secrets.h   # then edit SSID / password / VK_SERVER_IP
```

Without `secrets.h` the build still works but prints a warning. Every wake
then ends in `LINK_FAIL`: the LED blinks 3 times and no transcript arrives.

## 2. Build and flash (PowerShell)

The build dir must be a **short path** (Windows 260-character limit):

```powershell
. C:\Users\User\esp\esp-idf-v5.5.5\export.ps1
cd firmware
idf.py -B C:\Users\User\esp\vkc\build -D SDKCONFIG=C:\Users\User\esp\vkc\sdkconfig set-target esp32s3   # once
idf.py -B C:\Users\User\esp\vkc\build -D SDKCONFIG=C:\Users\User\esp\vkc\sdkconfig build
idf.py -B C:\Users\User\esp\vkc\build -D SDKCONFIG=C:\Users\User\esp\vkc\sdkconfig -p COM10 flash
..\server\.venv\Scripts\python.exe ..\tools\serial_logger.py --port COM10 --forward --out ..\sessions\<run>.jsonl
```

The console runs at 921600 baud. That only works with
`CONFIG_ESP_CONSOLE_UART_CUSTOM=y`, which is already in `sdkconfig.defaults`.

## 3. Wiring (from `sdkconfig.defaults`, the S3 on COM10)

| Part | Pin | ESP32-S3 |
|---|---|---|
| INMP441 | SCK / WS / SD | GPIO 4 / 5 / 6 |
| INMP441 | L/R | 3V3 (right slot; the boot `mic probe` re-checks) |
| INMP441 | VDD / GND | 3V3 / GND |
| LED + 330 Ω | anode | GPIO 2 |
| SSD1306 OLED | SDA / SCL | GPIO 8 / 9 (0x3C, `config.h`) |

## 4. After retraining the model

From `training/`:

```bash
../.venv/Scripts/python.exe quantize.py      # model.keras -> model_int8.tflite, float vs int8 check
../.venv/Scripts/python.exe export_c.py      # -> firmware/main/model_data.* + feature_tables.h
../.venv/Scripts/python.exe parity_check.py  # C features vs Python, must print PARITY PASS
```

The board prints the model sha256 at boot, and a mismatch with
`feature_tables.h` aborts the boot, so a stale model can't run silently.

## 5. What to check after flashing

1. The `boot` JSON line (firmware_hash, config_hash, link, policy), then `mic probe … signal` on exactly one slot.
2. Ticks at 2 Hz: `mic_rms` well above −100 dBFS; in a quiet room `state` is LISTEN and `funnel.s0` barely moves.
3. Say "Vektora": `opens` goes up, then `backfill`, then a `detect` event (`l1_ms`, `infer_us`), then `link` (`wifi_ms`, `l2_ms`), `stream_end` (`packets`, `ring_ovf` 0), and `transcript`.
4. The OLED shows the same numbers. `-` means not measured yet.

Everything the firmware prints is a measurement. What has actually been
measured on the board so far is listed in `docs/results.md`.

# Vektora OLED CPU test — ESP-IDF project (ESP32-WROOM-32 DevKit V1).
#
# Standalone rehearsal for CLAUDE.md §13.7 Phase B (idle-CPU work): runs the
# real Run 3 int8 model back-to-back on core 0 (no mic, no WiFi) and shows the
# measured per-core CPU load live on a 128x64 SSD1306 OLED, plus a full
# numeric readout over serial. See main/main.cc and README.md.
#
# Build:  idf.py set-target esp32 && idf.py build
cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(vektora_oled_cpu_test)

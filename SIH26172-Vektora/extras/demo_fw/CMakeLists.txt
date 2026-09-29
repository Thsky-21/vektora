# Vektora demo firmware (ESP32-S3): SoftAP + phone web UI, LED on GPIO2,
# SSD1306 OLED with measured CPU/RAM. See main/main.c.
cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(vektora_demo)

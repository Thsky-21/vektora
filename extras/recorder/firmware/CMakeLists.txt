# Vektora recorder — streams raw INMP441 audio to the PC over USB serial.
# Build:  idf.py set-target esp32 && idf.py build
# Flash:  idf.py -p COM10 flash
# The PC side is ../record.py.
cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(vektora_recorder)

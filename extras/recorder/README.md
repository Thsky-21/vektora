# Vektora recorder

Records training audio **through the real INMP441 on the board**, so the data
is captured by exactly the same chain the detector uses: 16 kHz, mono, and
the same 32-bit to 16-bit conversion (`>> 16`).

```
INMP441 --I2S--> ESP32-S3 (firmware/) --USB--> PC (record.py) --> WAV files
```

## Wiring (ESP32-S3, 2026-09-24)

| INMP441 | ESP32-S3 |
|---|---|
| VDD | 3.3 V |
| GND | GND |
| L/R | GND (left channel) |
| SCK | GPIO6 |
| WS | GPIO5 |
| SD | GPIO4 |

Plug the USB cable into the board's **USB** connector (native USB). It shows
up as `USB Serial Device (COMxx)`, VID:PID `303A:1001`.

## 1. Flash the firmware (once)

In PowerShell:

```powershell
. C:\Users\User\esp\esp-idf-v5.5.5\export.ps1
cd prototype\recorder\firmware
idf.py set-target esp32s3      # only the first time
idf.py -p COM11 flash
```

## 2. Record (from the repo root)

```bash
# Check the mic first. Speech at 1 m should read about -20 dBFS rms.
.venv/Scripts/python.exe prototype/recorder/record.py --port COM11 level

# One speaker: ~44 "Vektora" + ~31 hard negatives, guided, about 10 minutes.
.venv/Scripts/python.exe prototype/recorder/record.py --port COM11 clips --speaker S01 --gender F

# Only positives, or only hard negatives:
... clips --speaker S01 --gender F --only pos
... clips --speaker S01 --gender F --only neg

# Room noise / long sessions:
.venv/Scripts/python.exe prototype/recorder/record.py --port COM11 long --name fan_speed3 --kind noise --minutes 10
```

- **Resuming:** if a session stops, run the same command again. Takes that
  are already saved are skipped.
- **Speaker IDs:** give each person one ID and never reuse it for someone
  else. The train/test split will be done by speaker.

## What each take does

1. Press ENTER, wait for `>>> SPEAK <<<`, and say the word once.
2. The script records a 2.5 s window, starting 0.3 s *before* the ENTER.
   Phrases get 3.5 s.
3. **Quality checks.** A take that fails is asked again automatically:
   - **Lost audio:** a lost USB packet, an ESP32 buffer overflow, or 10 ms or
     more of exact silence. This is how the old dataset got its chopped files.
   - **Clipping:** 5 or more samples at full scale.
   - **Timing:** the word touches the start or end of the window.
   - **No speech:** the loudest part is less than 15 dB above the room.
4. **Saving:**
   - The script cuts a **1.0 s clip with the word centred**.
   - Phrases are cut from the start of speech, so the next word sits at the
     end of the clip.
   - A word too long to fit with 150 ms of room tone each side is **kept and
     flagged**, never cut short.

## Output (`prototype/data/`, git-ignored because it contains voices)

```
recordings/<speaker>/clips/pos_S01_1m-normal_001.wav   1.0 s, for training
recordings/<speaker>/raw/pos_S01_1m-normal_001.wav     full take, kept so clips can be re-cut
recordings/takes.csv       every saved take with its QC numbers and sample shift
recordings/speakers.csv    speaker, gender, device, room, date
long/<kind>/<name>_<date>_partN.wav
```

A long recording that hits a gap is split into a new `partN` file instead of
being joined across the gap.

## Stream format

Each packet is 338 bytes and carries 10 ms of audio. The fields are the
magic `VKA1`, a sequence number, an overflow count, the sample count, flags,
160 int16 samples, and a CRC-16. The full layout is in the header comment of
`firmware/main/recorder_main.c`.

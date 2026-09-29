# Challenges log — SIH26172 Vektora

What went wrong or got in the way while building this, **session by session**, and how each was handled. The goal is to have the engineering story ready for judges and to keep future sessions from repeating mistakes.
`CLAUDE.md` says what to do; this file says what bit us.

Format per entry: **Challenge**, then *Impact*, *Resolution / status*.
Status tags: ✅ solved · ⚠️ worked around · ❌ open.

---

## Sessions up to 2026-09-24 (reconstructed from `git log -p CLAUDE.md` and `docs/current_state.md`)

1. **Chopped / zero-gap audio in the team dataset** ✅
   - *Impact:* two training runs learned the artefact instead of the word.
   - *Resolution:* `audit_data.py` flags interior zero-gaps and abrupt cuts; corrupted takes are excluded. Run 3 still rewards the artefact in a handful of clips (cleaner re-split still on the backlog).
2. **Only 706 unique recordings behind 2,334 WAVs** ⚠️
   - *Impact:* the pre-made augmented copies inflate the count and would leak across splits.
   - *Resolution:* those copies are ignored, augmentation happens online, and splits are grouped by original recording.
3. **Tensor arena boot loop on the WROOM-32** ✅
   - *Impact:* the 160 KB arena request couldn't be met as one contiguous block, so the board rebooted endlessly.
   - *Resolution:* the arena is clamped to the largest free block minus a 24 KB margin.
4. **Inference is 564 ms on the plain ESP32** ❌
   - *Impact:* 5.6× over a real-time budget. ESP-NN's optimised kernels are S3-only, and the plain-C TFLM costs ~14 cycles per MAC. This hurts both latency (C5) and CPU (C2).
   - *Status:* depends on the board decision (see 2026-09-28).
5. **Features cost 4.1 ms per 10 ms frame** ⚠️
   - *Impact:* N_FFT = 400 isn't a power of two, so a table-driven DFT is used instead of an FFT. Running features continuously is ~41 % of one core. The gate in the late 2026-09-28 session is the mitigation.
6. **The ESP32-S3 recorder board's mic read all zeros** ❌
   - *Impact:* both I2S slots returned exactly 0, which blocked the guided recording campaign on that board.
   - *Status:* never resolved.
7. **SIH portal PS text fetched on 2026-09-24 but never committed** ❌
   - *Impact:* `docs/ps.md` still doesn't exist.
   - *Status:* the PS must be re-fetched from the portal, not written from memory.

## 2026-09-28 — training session (model retrain, T0.5)

1. **Tuning the hard-negative mix is non-monotonic** ⚠️
   - *Impact:* three mixes (0.20/0.10, 0.15/0.15, 0.18/0.12) gave 80 % / 50 % / 65 % recall on the user's own voice. The v3 mix was worse than v1 on every axis, so there was no smooth trade-off to tune.
   - *Resolution:* kept v1 because demo recall on the user's voice matters most.
2. **31.6 % false accepts on the user's own look-alike words (victor/vector/victoria)** ❌
   - *Impact:* other speakers' look-alikes false-accept only 5 %. The likely cause is too few `negative_akash` clips (61 in training).
   - *Status:* the fix is more `negative_akash` recordings, not more mix tuning.
3. **Test buckets are small (19–20 clips)** ⚠️
   - *Impact:* every recall and false-accept number carries real uncertainty, and there's no multi-seed averaging yet.

## 2026-09-28 — build session (server, console, logger, on-demand Wi-Fi)

1. **Windows 260-character path limit** ✅
   - *Impact:* an IDF build under the Claude scratchpad failed with "opening dependency file … No such file".
   - *Resolution:* build in `C:\Users\User\esp\vkb` (short path).
2. **Windows `time.monotonic` only ticks every 15.6 ms** ✅
   - *Impact:* every server latency number would have been quantised.
   - *Resolution:* use `perf_counter` (QPC, 100 ns).
3. **Vosk cold start** ✅
   - *Impact:* the first decode took 2.8 s, and building each recogniser cost ~144 ms.
   - *Resolution:* the model is loaded and warmed at startup, and a pool of pre-built recognisers is kept ready.
4. **Idle radio vs the <10 % CPU target** ⚠️
   - *Impact:* an associated radio wakes on beacons and injects RF noise into the mic.
   - *Resolution:* on-demand Wi-Fi (radio off while listening) with fast reconnect. The cost is connect time after the wake, which the ring buffer has to cover. Not measured on a board yet.
5. **Vosk doesn't know the word "Vektora"** ✅ (solved in the late session)
   - *Impact:* the small English model hears "vector" or "regular".

## 2026-09-28 — late session (keyword display, T1 telemetry, T3 gate)

1. **Showing "Vektora" without faking evidence** ✅
   - *Impact:* the transcript has to say "Vektora", but Rule 0.1 forbids hiding what Vosk actually heard.
   - *Resolution:* only the display is rewritten. The raw Vosk text is kept as `text_raw`, along with how the keyword was located (`kw.mode`), and the console shows "Vosk heard: …".
2. **Locating the keyword in the transcript** ✅
   - *Impact:* over HTTP there are no timestamps. Over UDP a packet can be lost, including seq 0.
   - *Resolution:*
     - UDP: the device's own timestamps give the keyword's end time; if seq 0 is missing, its time is derived from any other packet.
     - HTTP: fuzzy matching.
3. **Fuzzy matching was too loose** ✅
   - *Impact:* "the doctor" merged into one keyword match.
   - *Resolution:* a two-word merge now needs adjacent words, a 0.75 similarity ratio and both words inside the search window.
4. **Self-inflicted regression in `udp_rx.py`** ✅
   - *Impact:* a scripted edit swallowed the `_on_audio` definition, so every UDP session came back empty.
   - *Resolution:* caught by the end-to-end `test_sender` run, not by reading the code. Lesson: always re-run the loopback test after editing the server.
5. **The gate alone doesn't fix idle CPU if features always run** ✅
   - *Impact:* features cost ~41 % of core 1 whether anyone speaks or not.
   - *Resolution:* features are now gated too, which means the gate opens *after* the keyword starts (median ~185 ms, p90 ~490 ms offline). The skipped rows are backfilled from the audio ring on core 0, and every feature row is tagged with its frame number so a stale row can never be used.
6. **First gate constants kept the gate open ~20 % of the time on noise** ✅
   - *Impact:* the floor fell fast (α 0.3) toward the quietest frames of fluctuating noise, so 75 % of noise frames passed Stage 0.
   - *Resolution:* a sweep through the real C gate code settled on α 0.01 with a +12 dB threshold. Noise open time fell to 12–14 %, and 79/80 keyword clips still open the gate. These are offline numbers; they must be re-tuned on the device.
7. **Duplicate dataset folder** ✅
   - *Impact:* `background_noise/` holds the same files as `background_negative/`, which would have double-counted the noise results.
   - *Resolution:* dropped from the study.
8. **Missing FreeRTOS API in IDF 5.5.5** ✅
   - *Impact:* `ulTaskGetRunTimeCounter(handle)` doesn't exist in this version.
   - *Resolution:* use `vTaskGetInfo`. Known caveat: idle-task counters only update on context switches, so CPU % needs a sanity check against `GATE_ENABLED 0`.
9. **Wasted arena RAM** ✅
   - *Impact:* the arena was clamped to ~128 KB while the model uses 80,656 B.
   - *Resolution:* a probe allocation measures the real need, then the arena is re-allocated at used + 1 KB, freeing heap for the ring and Wi-Fi. Not verified on a board yet.
10. **C++20 `-Wvolatile` warnings** ✅
    - *Impact:* `++` on volatile counters warns under C++20.
    - *Resolution:* explicit `x = x + 1`. The build is now warning-free.
11. **Wrong board on the USB port** ✅ (2026-09-29: user chose the S3; firmware retargeted and flashed)
    - *Impact:* COM10 has the **ESP32-S3**, but the firmware targets the **WROOM-32**, so nothing could be flashed. The user deferred the decision to a new session.
    - *Status:* blocks every device measurement (M1–M3, L2).
12. **Wi-Fi credentials still placeholders** ❌
    - *Impact:* no link test is possible until they're set.
    - *Status:* the laptop hotspot (`192.168.137.1`) is available; the user needs to supply its SSID/password.
13. **Tooling friction on Windows** ⚠️
    - *Impact:* bash heredocs mixing quotes broke long scripted edits.
    - *Resolution:* write the script to a file first, then run it.

## 2026-09-29 — First flash on the ESP32-S3

1. **Console baud silently stayed at 115200** ✅
   - *Impact:* the first capture at 921600 was pure garbage bytes. Editing `CONFIG_ESP_CONSOLE_UART_BAUDRATE` was reverted on reconfigure, because Kconfig only honours it with a *custom* console UART.
   - *Resolution:* `CONFIG_ESP_CONSOLE_UART_CUSTOM=y` in the build sdkconfig and in `sdkconfig.defaults`. Boot log and telemetry now arrive at 921600.
2. **Mic sends no data on the S3** ❌
   - *Impact:* the new boot-time mic probe (`mic_probe()` in `main.cc`) reads both I²S slots: in each one ~43 % of words are exactly 0 and ~57 % have random low bytes. An INMP441 always leaves the low byte at 0, so SD is floating. `mic_peak` sits at 0 dBFS, `mic_rms` about −26 dBFS in a quiet room, and Stage 1 of the gate never passes. No keyword can be detected.
   - *Status:* the pins match the recorder tool (SD=4, WS=5, SCK=6), so this is physical: mic VDD (3V3)/GND, a loose SD/SCK/WS jumper, or a dead mic. Same symptom family as the earlier "all zeros" on this board.
3. **Idle RAM over 256 KB** ❌
   - *Impact:* first real S3 numbers: `ram_used` 320 KB (static 51.8 KB + heap 268 KB), peak 329 KB. Contributors: arena 81.7 KB, uploader ring 80 KB, Wi-Fi driver initialised at boot for the probe connect.
   - *Status:* not tuned yet. Next steps: `wifi` deinit after the probe (or `WIFI_INIT_AT_BOOT 0`), then a smaller ring.
4. **Idle CPU on core 1 is ~20 %** ⚠️
   - *Impact:* measured, but meaningless while the mic feeds noise, because noise passes Stage 0 on ~50 % of frames, so features run on them. Re-measure (M2) once the mic works.

---

*How to update:* at the end of every session, add a `## <date> — <session name>` section. Log what fought back, even small things, with impact and resolution. Flip ❌ to ✅ in earlier sections when something gets fixed, and add a date.

## 2026-09-29 — laptop-mic emulation page
- ⚠️ **S3 mic still dead, demo needed now.** We built `/mic` (`server/static/mic.html` + `server/mic_emu.py`). The browser streams the laptop mic over WebSocket into a host copy of the pipeline: ring → energy gate → voice gate → KWS (a Vosk sound-alike match on "vector/victor…", **not** the DS-CNN) → detect → pre-roll → Vosk streaming → transcript. It reuses `asr.AsrStream`, so sessions are logged to `sessions/*.jsonl` like the board path. The page says the source is the laptop mic. Host numbers (L1, link, CPU) are **not** device evidence.
- ✅ The first KWS version never fired, because Vosk commits a word only after trailing audio and the gate closed first. Fix: flush the KWS recogniser with `FinalResult()` when the gate closes.

## 2026-09-29 — demo firmware (`prototype/demo_fw`)
- ⚠️ Mic still dead on the S3, so for the submission demo we built a separate firmware: ESP32 SoftAP + phone web page (LED button, WAKE button), OLED on SDA=8/SCL=9. KWS is **not** running in this build.
- ✅ Phone page didn't load: the AP had no internet, so Android sent browser traffic over mobile data. Fixed with a captive-portal DNS (every lookup → 192.168.4.1) and 302 redirects.
- ✅ CPU/RAM on the OLED/page are measured (idle-task counters, heap_caps), not scripted. Measured: idle avg 0.4–1.3 %; LED on core0 8.6 % (real log-mel work every 50 ms on core 0); WAKE core1 ~41 %, avg ~23 % (log-mel at 100 frames/s on synthetic audio + UDP audio stream); RAM ~104/337 KB, peak 114 KB.
- ✅ (later) 3 buttons: GATE / LED / WAKE. Each state runs a fixed per-20 ms budget of real energy/ZCR maths on core 0 (`B0` in `main.c`, plus a 5 µs core-1 check), **sized on purpose** to land in the team's target CPU bands. Measured with a phone connected: IDLE 1.8–2.0 % / 0.19–0.20 %, GATE ~3.5 %, LED ~7.6 %, WAKE ~13.5 % / ~40.7 %. lwIP task pinned to core 0.
- ✅ Wake latency is timed on the device: WAKE press → 0.75 s pre-roll log-mel (75 frames) → first UDP packet = 301–304 ms. Caveat for judges: this is a pipeline stand-in on synthetic audio, not keyword-end → ASR.
- ⚠️ (flagged 2026-09-29 late) `demo_fw`'s CPU bands come from a workload **sized to land in the target bands**. They must never be presented as the voice activator's CPU. `extras/README.md` says so.

## 2026-09-29 late — clean repo folder + remaining firmware/console/tools

1. **The project was spread over five places** ✅
   - *Impact:* `prototype/`, root folders, the external `documents/data` repo, root loose WAVs and `demo/`. It couldn't be uploaded cleanly, and the root `.gitignore` hid the trained models.
   - *Resolution:* one self-contained folder, `SIH26172-Vektora/`, with its own `.gitignore` that re-includes `*.keras`/`*.tflite`. MD5 showed that all 456 root loose WAVs (`positive_audio/`, `soft_sample/`, `soft-negative/`, `background_noise/`, `vektora_*.wav`) were byte-identical copies of dataset files, so they're not duplicated.
2. **Manifests held absolute Windows paths** ✅
   - *Impact:* `C:\Users\User\documents\data\…` in 3,707 rows, so a clone couldn't retrain.
   - *Resolution:* the paths are now repo-relative (`dataset/…`), with `config.rel()`/`config.resolve()` in the readers. Splits are unchanged, because they're computed from groups, not path strings.
3. **Stale pin notes in CLAUDE.md** ✅
   - *Impact:* the text said SD=4/SCK=6; `sdkconfig.defaults` and the build config say SCK=4, WS=5, SD=6, right slot. The docs now follow the config.
4. **`-Werror=format-truncation` on the OLED rows** ✅
   - *Impact:* 21-character rows failed the IDF build.
   - *Resolution:* rows are formatted into 64-byte buffers, and the driver clips at the panel edge. The build is now 0 warnings.
5. **Benchmarking the server while the firmware compiled** ✅ (lesson)
   - *Impact:* Vosk was CPU-starved: 1 of 2 UDP trials had no transcript and HTTP took 8.9 s. Re-run on an idle laptop: 39 ms median, all transcripts correct.
6. **A backgrounded `&` server survived the shell** ✅
   - *Impact:* a second server couldn't bind UDP 5005.
   - *Resolution:* use proper background tasks, and check the ports before starting a server.
7. **SIH portal PS page is script-rendered** ❌
   - *Impact:* a plain fetch returned an empty table. `docs/ps.md` is a placeholder; paste the text from a browser.
8. **Still blocked on hardware** ❌
   - The S3 mic (floating SD) blocks every M-run. The OLED UI, UDP link, stride/smoothing and Wi-Fi deinit are compiled but have **never run on the board**. Expect first-run bugs there.

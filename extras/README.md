# extras/: side projects that fed the main build

None of these is the submission firmware (that's `../firmware`). Each one is
kept because it produced something the main build relies on, or because
it's history the judges may ask about.

| Folder | What | Status |
|---|---|---|
| `recorder/` | ESP32-S3 firmware + `record.py`: guided capture of training clips through the real INMP441 over USB, with CRC/sequence checks and a quality gate. Saves into `../dataset/`. | Built 2026-09-24. Blocked by the S3 mic fault (see `challenges.md`). |
| `oled_test/` | WROOM-32 rehearsal that runs the model back to back and shows measured per-core CPU on an SSD1306. | Historical: WROOM-32 pins, Run-3 model. |
| `demo_fw/` | ESP32-S3 **stand-in demo** for the idea-submission video, built while the mic was dead: SoftAP + phone page (GATE / LED / WAKE buttons), OLED, UDP stream of **synthetic** audio. | **Not evidence. Read the warning below.** |
| `streamlit_tester/` | Local Streamlit page: record from the laptop mic, run the trained Keras model, see per-window scores. Captures go to `../dataset/ui_captures/`. | Works on the laptop (training venv). |
| `streamlit_demo_v0/` | The first (2026-09-12) Streamlit/Docker demo and its model (`model/vektora.keras`), before the 3-class pipeline in `../training`. | Historical. |

## ⚠️ `demo_fw` numbers are not measurements of the voice activator

`demo_fw` runs no keyword model and uses no microphone. In each button
state, its core-0 workload is a fixed per-20 ms budget of energy/ZCR maths
(`B0[]` in `main/main.c`). Those budgets were **deliberately sized so the CPU
readings land in the team's target bands**. The CPU/RAM it shows are really
measured (FreeRTOS counters, heap_caps), but only for that synthetic
workload. The 301–304 ms "wake → first packet" figure is the same kind of
stand-in.

Under CLAUDE.md Rule 0.1, none of these numbers may appear on the
Compliance Card, the slides or `docs/results.md` as a result of the voice
activator. If the video shows `demo_fw`, it must say so on screen. The real
numbers come from `../firmware` + `../tools/serial_logger.py` +
`../tools/analyze.py`.

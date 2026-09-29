# prototype/ — the real SIH26172 device

The working prototype: the keyword detector running on the ESP32-WROOM-32,
the cascade around it, and the server that receives the streamed speech.
`demo/` (the 2026-09-12 Streamlit demo) is the reference it starts from.
The plan it follows is `CLAUDE.md` §9.

| Folder | What goes here | Language |
|---|---|---|
| `firmware/` | ESP-IDF project: I2S capture, energy + speech gates, KWS inference, pre-roll ring buffer, streaming | C |
| `training/` | Session recorder, segmenter, dataset build, training, int8 quantization, C-vs-Python feature parity checks | Python |
| `server/` | Streaming receiver, ASR, latency + power dashboard | Python |
| `data/` | All recordings and generated datasets. **Git-ignored** — contains people's voices and the repo is public | — |
| `docs/measurements/` | INA219 power logs, latency and false-accept results | — |

Why this split: it maps one-to-one onto the three deliverables in
`CONTEXT.md` (detector, firmware, server), which meet only at two contracts —
the 98x40 log-mel feature definition (training <-> firmware) and the audio
stream format (firmware <-> server).

Python always runs from the project venv: `../.venv/Scripts/python.exe`.

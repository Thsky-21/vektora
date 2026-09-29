# Dependencies and licences (PS constraint C3: open-source only)

Every component has an OSI-approved licence. There are no closed binaries (no ESP-SR/WakeNet), no hosted training, and no cloud ASR.
Checked 2026-09-29 against the files on disk, unless the last column says otherwise.

| Component | Where | Version | Licence | Verified from |
|---|---|---|---|---|
| ESP-IDF | firmware | v5.5.5 | Apache-2.0 | `LICENSE` in the IDF tree |
| esp-tflite-micro (TFLM) | firmware | 1.4.0 | Apache-2.0 | `managed_components/espressif__esp-tflite-micro/LICENSE`, `dependencies.lock` |
| ESP-NN | firmware | pulled in by esp-tflite-micro (`>=1.1.1`, see `dependencies.lock`) | Apache-2.0 | `managed_components/espressif__esp-nn/LICENSE` |
| cJSON | firmware (HTTP fallback) | bundled with IDF | MIT | IDF `components/json` |
| SSD1306 text driver | firmware | own code (`ssd1306.c`) | this project | — |
| TensorFlow / Keras | training | 2.15.1 / 2.15.0 | Apache-2.0 | installed dist-info |
| librosa | training | 0.11.0 | ISC | installed dist-info |
| soundfile | training | 0.14.0 | BSD-3-Clause | installed dist-info |
| Streamlit | extras/streamlit_tester | 1.60.0 | Apache-2.0 | installed dist-info |
| Vosk (python) | server | 0.3.45 | Apache-2.0 | alphacephei/vosk-api repo (pip metadata says UNKNOWN) |
| vosk-model-small-en-in-0.4 | server | 0.4 | Apache-2.0 | alphacephei.com/vosk/models (the archive README doesn't state it) |
| FastAPI | server | 0.141.1 | MIT | upstream repo |
| Starlette | server | 1.7.0 | BSD-3-Clause | upstream repo |
| Uvicorn | server | 0.54.0 | BSD-3-Clause | upstream repo |
| websockets | server | 17.1 | BSD-3-Clause | upstream repo |
| NumPy | server, training | 2.5.3 (server) | BSD-3-Clause | upstream repo |
| pySerial | tools | 3.5 | BSD-3-Clause | pip metadata: BSD |
| React / React-DOM | console | 19.3.0 | MIT | `console/node_modules/react/package.json` |
| Vite | console | 8.3.1 | MIT | `console/node_modules/vite/package.json` |
| Google Speech Commands (≈240 clips in `dataset/generic_negative/gsc_*`) | dataset | v0.02 | CC-BY-4.0 (data, not code) | dataset README; attribute on the slides/README |

Not used, and not allowed: ESP-SR / WakeNet, Picovoice, Edge Impulse hosted pipeline, any cloud ASR API.

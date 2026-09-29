# Device ↔ server protocol

Source of truth for the server: `server/protocol.py`. The firmware must match this byte for byte.

There are two uplinks. Both feed the same Vosk pipeline.

| Uplink | Used by | Status |
|---|---|---|
| UDP :5005, `VAD1` | firmware default (`LINK_UDP 1`, `firmware/main/asr_uploader.c` `do_stream_udp`) | server tested with `test_sender.py`; firmware side compiles, **not yet run against the server on hardware** |
| HTTP `POST /asr` | firmware fallback (`LINK_UDP 0`, `do_upload_http`) | server tested with `test_sender.py --http` |

Firmware UDP details: HELLO carries `score, t_win_end_us, t_detect_us, t_sock_us, wifi_ms, fw, cfg, codec, preroll_ms, policy, deinit_idle`. There are 3 PINGs 30 ms apart after link-up, then one more before END. The backlog goes out at 2 packets per 10 ms tick (4× real time). AUDIO `t_capture_us` = capture time of the packet's first sample, derived from the ring's (byte count, esp_timer) anchor taken at the wake. The stream ends when the gate has been closed `ENDPOINT_SILENCE_MS` (700 ms, at least `STREAM_MIN_MS` after the wake) or at `STREAM_MAX_MS` (8 s). END carries `{"packets","bytes","ring_ovf","reason"}`. The device then waits up to `TRANSCRIPT_WAIT_MS` for TRANSCRIPT before turning Wi-Fi off.

## 1. HTTP `POST /asr` (current firmware)

- Body: raw PCM16 little-endian, 16 kHz, mono, no header. `Content-Length` = exact byte count. The body may be streamed while recording (the firmware sends 4 KB writes).
- The server decodes the body **as it arrives**, so the transcript is ready about 30 ms after the last byte (measured, see `docs/results.md`).
- Reply `200 application/json`:
  `{"text": "...", "session": "<id>", "upload_ms": .., "final_after_last_byte_ms": .., "audio_ms": .., "first_text_after_request_ms": ..}`
  The firmware reads only `text`.

## 2. UDP `VAD1` (port 5005)

### Header (22 bytes, little-endian, packed)

| Offset | Field | Type | Notes |
|---|---|---|---|
| 0 | magic | u32 | bytes `'V','A','D','1'` (= `0x31444156` read as LE u32) |
| 4 | session | u16 | device-chosen, new for each wake |
| 6 | type | u8 | see below |
| 7 | codec | u8 | 0 = PCM16, 1 = G.711 μ-law |
| 8 | seq | u32 | **AUDIO only**: 0, 1, 2, … per session. 0 for all other types |
| 12 | t_capture_us | u64 | device `esp_timer_get_time()`. AUDIO: capture time of the first sample |
| 20 | len | u16 | payload bytes |

### Types

| # | Direction | Name | Payload |
|---|---|---|---|
| 1 | dev→srv | HELLO | UTF-8 JSON: `score, t_win_end_us, t_detect_us, t_sock_us, wifi_ms, fw, cfg, codec` |
| 2 | dev→srv | AUDIO | 20 ms: 640 B PCM16 or 320 B μ-law |
| 3 | dev→srv | END | empty (optional JSON stats) |
| 4 | dev→srv | PING | `t_dev_us u64` **or** `t_dev_us u64, prev_t_dev_us u64, prev_rtt_us u32` |
| 5 | srv→dev | PONG | `t_dev_us u64` (echoed), `t_srv_us u64` |
| 6 | srv→dev | TRANSCRIPT | ≤ 60 bytes UTF-8 (never split mid-character) |

### Sequence (one wake)

```
HELLO -> PING -> AUDIO backlog burst (from t_detect - PREROLL_MS, up to 4x real time)
      -> PING(prev rtt) -> PING(prev rtt) -> AUDIO real time ... -> END
                                                   <- PONG (each PING)   <- TRANSCRIPT
```

### Clock sync

Only the device sees both ends of a PING, so only the device knows the RTT. It reports each RTT back in the **next** PING (`prev_t_dev_us`, `prev_rtt_us`). The server kept the time it received that earlier PING (`t_srv`) and computes:

`offset = t_srv − (prev_t_dev + RTT/2)`, error bound ±RTT/2. It keeps the sample with the lowest RTT.

Send at least 3 PINGs right after link-up, plus one more before END, so the last RTT gets reported. The server replies with PONG before doing anything else, so its own turnaround doesn't inflate the RTT.

### Server behaviour

- Packets are reordered by `seq`. A gap is held for up to **60 ms**. After that it is declared lost: a `pkt_loss` event is sent, and 20 ms of silence per lost packet is fed to ASR so timing stays aligned. Packets that arrive after that are counted as `late` and dropped.
- If no END arrives within **3 s** of the last packet, the session is finalised anyway (`reason: idle_timeout`).
- On the final transcript: TRANSCRIPT is sent to the device's source address and port, and a `session_end` event is logged with the L1–L4 waterfall.

## 3. Latency fields (CLAUDE.md §7), in `session_end.latency`

| Field | Computed as | Clock |
|---|---|---|
| `L1_ms` | `t_detect − t_win_end` (from HELLO) | device |
| `L2_ms` | `t_sock − t_detect` (from HELLO) | device |
| `L3_ms` (±`L3_error_ms`) | server receipt of first packet − offset − `t_sock` | synced |
| `L4_ms` | first Vosk partial − first AUDIO receipt | server |
| `total_ms` | L1 + L2 + L3 | synced |
| `keyword_end_to_first_text_ms` | first partial − offset − `t_win_end` | synced |

L4 includes *where the speech is* in the stream. To see what the server alone adds, use `asr_partial.lag_ms` (newest audio received → text out) and `asr_final.decode_lag_median_ms` / `decode_lag_p90_ms`.

## 4. Server → console: WebSocket `/ws/events`

One JSON object per message. Every event has `t_mono_us` (server QPC clock) and `wall`. Session events also carry `session`.

`hello_console`, `session_start`, `server_rx_first`, `hello`, `clock_sync`, `pkt_loss`, `asr_partial`, `asr_final`, `asr_error`, `latency` (partial), `session_end`; plus device telemetry (`tick`, `detect`, `link`, `stream_end`, with `source: "device"`) relayed from `tools/serial_logger.py --forward` over UDP 127.0.0.1:5006.

Each session is also written to `sessions/<id>.jsonl`, and the received audio to `sessions/<id>.wav`.

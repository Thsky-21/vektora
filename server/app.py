"""app.py — FastAPI server: device audio in (UDP :5005 and HTTP POST /asr), Vosk
streaming ASR, live events out on /ws/events, every session logged to
sessions/<id>.jsonl + .wav.

Run (from server/):
    .venv/Scripts/python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000 --no-access-log

Two uplinks, same ASR:
  * POST /asr   — what the CURRENT firmware (asr_uploader.c) sends: raw PCM16
                  16 kHz mono, exact Content-Length, streamed while recording.
                  Reply {"text": ...}. Body is decoded as it arrives, not after.
  * UDP :5005   — the §8 protocol for the T7 link (HELLO/AUDIO/END/PING).
"""

import asyncio
import datetime as dt
import itertools
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import asr
import protocol as P
from events import hub, list_sessions, now_us, read_session
from mic_emu import Emu
from udp_rx import TelemetryRelay, UdpRx

TELEMETRY_PORT = int(os.environ.get("TELEMETRY_PORT", "5006"))
_http_ids = itertools.count(1)
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    hub.loop = loop
    info = await loop.run_in_executor(None, asr.load)  # model hot before first packet
    print(f"[asr] {info}", flush=True)
    udp_t, udp = await loop.create_datagram_endpoint(UdpRx, local_addr=("0.0.0.0", P.UDP_PORT))
    tel_t, _ = await loop.create_datagram_endpoint(TelemetryRelay, local_addr=("127.0.0.1", TELEMETRY_PORT))
    state["udp"] = udp
    print(f"[udp] audio on :{P.UDP_PORT}, telemetry relay on 127.0.0.1:{TELEMETRY_PORT}", flush=True)
    yield
    udp_t.close()
    tel_t.close()


app = FastAPI(title="SIH26172 Vektora server", lifespan=lifespan)


@app.post("/asr")
async def http_asr(request: Request):
    sid = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-http{next(_http_ids)}"
    t_req = now_us()
    hub.emit({"type": "session_start", "transport": "http",
              "addr": request.client.host if request.client else None,
              "content_length": request.headers.get("content-length")}, sid)
    stream = asr.AsrStream(sid, lambda ev: hub.emit_threadsafe(ev, sid))
    first, odd, n = True, b"", 0
    async for chunk in request.stream():
        if not chunk:
            continue
        if first:
            first = False
            hub.emit({"type": "server_rx_first", "t_rx_us": now_us()}, sid)
        chunk = odd + chunk  # keep samples 2-byte aligned across TCP reads
        cut = len(chunk) & ~1
        odd = chunk[cut:]
        stream.feed(chunk[:cut])
        n += cut
    t_last = now_us()
    res = await asyncio.wrap_future(stream.finish())
    text = res["text"]
    t_done = now_us()
    lat = {"upload_ms": round((t_last - t_req) / 1000, 1),
           "final_after_last_byte_ms": round((t_done - t_last) / 1000, 1),
           "audio_ms": n * 1000 // (P.SR * 2)}
    if stream.t_first_text_us:
        lat["first_text_after_request_ms"] = round((stream.t_first_text_us - t_req) / 1000, 1)
    hub.emit({"type": "session_end", "reason": "http_done", "text": text,
              "text_raw": res["text_raw"], "kw": res["kw"], "latency": lat}, sid)
    hub.close_session(sid)
    return JSONResponse({"text": text, "text_raw": res["text_raw"], "session": sid, **lat})


@app.get("/api/health")
async def health():
    return {"ok": True, "asr": asr.startup_info, "udp_port": P.UDP_PORT,
            "active_udp_sessions": len(state["udp"].sessions) if "udp" in state else 0}


@app.get("/api/sessions")
async def sessions():
    return list_sessions()


@app.get("/api/sessions/{sid}")
async def session(sid: str):
    data = read_session(sid)
    if data is None:
        raise HTTPException(404, "no such session")
    return data


@app.websocket("/ws/events")
async def ws_events(ws: WebSocket):
    await ws.accept()
    q = hub.subscribe()
    try:
        await ws.send_text(json.dumps({"type": "hello_console", "asr": asr.startup_info}))
        while True:
            await ws.send_text(await q.get())
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.unsubscribe(q)


@app.get("/mic")
async def mic_page():
    """Laptop-mic emulation of the device pipeline (see mic_emu.py)."""
    return FileResponse(Path(__file__).resolve().parent / "static" / "mic.html")


@app.websocket("/ws/mic")
async def ws_mic(ws: WebSocket):
    """Browser mic in (binary PCM16 16 kHz frames, or JSON commands), emulated
    device telemetry + ASR events out on the same socket."""
    await ws.accept()
    out: asyncio.Queue = asyncio.Queue(maxsize=500)
    emu = Emu(asyncio.get_running_loop(), out)

    async def pump():
        while True:
            await ws.send_text(await out.get())
    sender = asyncio.create_task(pump())
    try:
        await ws.send_text(json.dumps({"type": "hello_mic", "asr": asr.startup_info}))
        while True:
            m = await ws.receive()
            if m.get("type") == "websocket.disconnect":
                break
            if m.get("bytes"):
                emu.feed(m["bytes"])
            elif m.get("text"):
                emu.feed(m["text"])
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        emu.close()
        sender.cancel()


_dist = Path(__file__).resolve().parent.parent / "console" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="console")

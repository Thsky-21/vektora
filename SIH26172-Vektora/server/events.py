"""events.py — one place every event goes: WebSocket clients + sessions/<id>.jsonl.

Every event gets two server timestamps: t_mono_us (high-res monotonic, for latency maths)
and wall (ISO time, for humans). Slow WebSocket clients get events dropped
rather than blocking the audio path.
"""

import asyncio
import datetime as dt
import json
import time
from pathlib import Path

SESSIONS_DIR = Path(__file__).resolve().parent.parent / "sessions"


def now_us() -> int:
    # perf_counter, not monotonic: on Windows monotonic ticks every 15.6 ms,
    # which would quantise every latency number. QPC is 100 ns.
    return time.perf_counter_ns() // 1000


class Hub:
    def __init__(self):
        self.clients: set[asyncio.Queue] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self._files: dict[str, object] = {}
        SESSIONS_DIR.mkdir(exist_ok=True)

    def emit(self, ev: dict, session: str | None = None):
        """Call from the event loop thread."""
        ev.setdefault("t_mono_us", now_us())
        ev.setdefault("wall", dt.datetime.now().isoformat(timespec="milliseconds"))
        if session:
            ev["session"] = session
            f = self._files.get(session)
            if f is None:
                f = open(SESSIONS_DIR / f"{session}.jsonl", "a", encoding="utf-8", buffering=1)
                self._files[session] = f
            f.write(json.dumps(ev) + "\n")
        msg = json.dumps(ev)
        for q in list(self.clients):
            if q.full():
                continue  # slow client: drop, never block
            q.put_nowait(msg)

    def emit_threadsafe(self, ev: dict, session: str | None = None):
        """Call from worker threads (Vosk). Stamp now, deliver on the loop."""
        ev.setdefault("t_mono_us", now_us())
        self.loop.call_soon_threadsafe(self.emit, ev, session)

    def close_session(self, session: str):
        f = self._files.pop(session, None)
        if f:
            f.close()

    def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=500)
        self.clients.add(q)
        return q

    def unsubscribe(self, q):
        self.clients.discard(q)


hub = Hub()


def list_sessions():
    out = []
    for p in sorted(SESSIONS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        out.append({"id": p.stem, "bytes": p.stat().st_size, "mtime": p.stat().st_mtime})
    return out


def read_session(sid: str):
    p = SESSIONS_DIR / f"{Path(sid).name}.jsonl"
    if not p.exists():
        return None
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

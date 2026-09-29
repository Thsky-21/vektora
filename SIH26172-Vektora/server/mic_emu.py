"""mic_emu.py — the ESP32-S3 pipeline, emulated on the laptop, fed by the
laptop's microphone over WebSocket /ws/mic (page: /mic).

STAND-IN, NOT THE DEVICE. The INMP441 on the S3 sends no data yet (CLAUDE.md
blocker), so this runs the same pipeline shape on the host so the end-to-end
flow can be shown:

  browser mic 16 kHz PCM16, 20 ms frames
    -> ring (4 s, absolute sample index)
    -> Stage 0 energy gate  (adaptive floor, +GATE0_DB for 2 frames)
    -> Stage 1 voice gate   (300-3400 Hz share + spectral flatness)
    -> KWS only while the gate is open (backfill 300 ms from the ring on open)
       KWS here = a Vosk recogniser matching "Vektora" sound-alikes, NOT the
       DS-CNN (that runs on the board only)
    -> DETECT -> "link up" (open an ASR session) -> pre-roll from the ring
    -> stream to the same asr.AsrStream the UDP/HTTP paths use
    -> endpoint (ENDPOINT_SILENCE_MS of gate-closed, or MAX_STREAM_MS) -> LISTEN

Every number sent to the page is measured on the host by this code.
"""

import asyncio
import datetime as dt
import itertools
import json
import queue
import threading
import time
import traceback

import numpy as np

import asr
from events import hub, now_us

SR = 16000
FRAME = 320                    # 20 ms
FRAME_MS = 20
RING_S = 4
GATE0_DB = 10.0                # frame energy above the noise floor
GATE0_FRAMES = 2
GATE0_ABS_MIN_DB = -58.0       # browser noise suppression drives the floor to ~-100
GATE1_BAND_RATIO = 0.55
GATE1_FLATNESS = 0.45
GATE1_OPEN_FRAMES = 3          # voiced frames (of the last 5) to open
HANGOVER_MS = 400
BACKFILL_MS = 300
PREROLL_BEFORE_KW_MS = 250     # stream from keyword start minus this
ENDPOINT_SILENCE_MS = 800
MIN_AFTER_KW_MS = 1200         # give the user time to start the command
MAX_STREAM_MS = 8000
REFRACTORY_MS = 1000
TICK_MS = 100
MANUAL_MAX_MS = 30000          # push-to-talk cap

# KWS match set: what vosk-model-small-en-in hears for "Vektora". Deliberately
# excludes the hard negatives (doctor, factor, sector...) that wake_word.py
# accepts for display rewriting, so they don't false-trigger here.
KW_WORDS = {"vector", "victor", "vectra", "vectors", "victoria", "vektora"}
KW_MERGES = {"vector a", "victor a", "vic tora", "vector ah", "victor ah"}

_ids = itertools.count(1)
_win = np.hanning(FRAME).astype(np.float32)
_freqs = np.fft.rfftfreq(512, 1 / SR)
_band = (_freqs >= 300) & (_freqs <= 3400)
_all = _freqs >= 80


class Emu:
    """One browser connection. feed() from the event loop; all work on a thread."""

    def __init__(self, loop: asyncio.AbstractEventLoop, out: asyncio.Queue):
        self.loop, self.out = loop, out
        self.q: "queue.Queue[bytes | str | None]" = queue.Queue()
        self.ring = np.zeros(SR * RING_S, np.int16)
        self.ring_rx_us = np.zeros(SR * RING_S // FRAME, np.int64)  # host rx time per frame
        self.n = 0                    # absolute sample index (next to write)
        self.pending = b""
        # gate
        self.floor_db = None
        self.s0_run = 0
        self.s1_hist: list[bool] = []
        self.gate_open = False
        self.last_voice_frame = -10**9
        # funnel (since connect)
        self.f = {"frames": 0, "s0": 0, "s1": 0, "opens": 0, "kws_frames": 0, "det": 0, "manual": 0}
        # kws
        self.kws = None
        self.kws_start = 0            # absolute sample of the KWS segment start
        self.kws_buf = b""
        self.kws_heard = ""
        self.kws_busy_s = 0.0
        self.refractory_until = 0
        # state machine
        self.state = "LISTEN"
        self.stream = None
        self.sid = None
        self.stream_t0_frame = 0
        self.kw_end_frame = 0
        self.wakes = 0
        self.manual = False
        # metering
        self.level_db = -120.0
        self.peak_db = -120.0
        self.s0_now = self.s1_now = False
        self._cpu0 = (time.process_time(), time.perf_counter(), 0.0)
        self.cpu_pct = None
        self.kws_duty = None
        self.t_last_tick = 0.0
        threading.Thread(target=self._run, name="mic-emu", daemon=True).start()

    # -- loop side -------------------------------------------------------------
    def feed(self, data):
        self.q.put(data)

    def close(self):
        self.q.put(None)

    def _send(self, ev: dict, session: str | None = None):
        """Thread-safe: to this page, and (for session events) to the hub/JSONL."""
        ev.setdefault("t_mono_us", now_us())

        def go():
            if session:
                hub.emit(ev, session)  # adds session + wall, logs to sessions/<sid>.jsonl
            if not self.out.full():
                self.out.put_nowait(json.dumps(ev))
        self.loop.call_soon_threadsafe(go)

    # -- worker ----------------------------------------------------------------
    def _run(self):
        try:
            while True:
                item = self.q.get()
                if item is None:
                    break
                if isinstance(item, str):
                    self._cmd(item)
                    continue
                self.pending += item
                while len(self.pending) >= FRAME * 2:
                    fr, self.pending = self.pending[:FRAME * 2], self.pending[FRAME * 2:]
                    self._frame(np.frombuffer(fr, "<i2"), now_us())
                t = time.perf_counter()
                if t - self.t_last_tick >= TICK_MS / 1000:
                    self.t_last_tick = t
                    self._tick()
        except Exception:
            traceback.print_exc()
            self._send({"type": "emu_error", "error": traceback.format_exc(limit=3)})
        finally:
            if self.stream:
                self._end("disconnect")

    def _cmd(self, s: str):
        try:
            c = json.loads(s)
        except ValueError:
            return
        if c.get("cmd") == "wake" and self.state in ("LISTEN", "SPEECH"):
            # push-to-talk (Space on the page): same path as a detection,
            # flagged manual; ends on "end", not on silence
            self.f["manual"] += 1
            self._wake(None, None, "manual", now_us())
        elif c.get("cmd") == "end" and self.state == "STREAMING":
            self._end("key")

    def _frame(self, x: np.ndarray, t_rx: int):
        fi = self.n // FRAME
        i = self.n % self.ring.size
        self.ring[i:i + FRAME] = x
        self.ring_rx_us[fi % self.ring_rx_us.size] = t_rx
        self.n += FRAME
        self.f["frames"] += 1

        xf = x.astype(np.float32) / 32768.0
        ms = float(np.mean(xf * xf)) + 1e-12
        db = 10 * np.log10(ms)
        self.level_db = db
        self.peak_db = max(self.peak_db - 0.6, 20 * np.log10(float(np.max(np.abs(xf))) + 1e-6))

        # Stage 0: adaptive floor, fast down / slow up (<= 2 dB/s)
        if self.floor_db is None:
            self.floor_db = db
        elif db < self.floor_db:
            self.floor_db += 0.1 * (db - self.floor_db)
        else:
            self.floor_db += min(0.04, db - self.floor_db)
        hot = db > self.floor_db + GATE0_DB and db > GATE0_ABS_MIN_DB
        self.s0_run = self.s0_run + 1 if hot else 0
        s0 = self.s0_run >= GATE0_FRAMES
        s1 = False
        if s0:
            self.f["s0"] += 1
            p = np.abs(np.fft.rfft(xf * _win, 512)) ** 2
            tot = float(p[_all].sum()) + 1e-12
            pb = p[_band] + 1e-12
            ratio = float(pb.sum()) / tot
            flat = float(np.exp(np.mean(np.log(pb))) / np.mean(pb))
            s1 = ratio >= GATE1_BAND_RATIO and flat <= GATE1_FLATNESS
            if s1:
                self.f["s1"] += 1
        self.s0_now, self.s1_now = s0, s1
        self.s1_hist = (self.s1_hist + [s1])[-5:]
        if s0:
            self.last_voice_frame = fi

        # gate open/close (hangover from last energetic frame)
        if not self.gate_open and sum(self.s1_hist) >= GATE1_OPEN_FRAMES:
            self.gate_open = True
            self.f["opens"] += 1
            if self.state == "LISTEN":
                self.state = "SPEECH"
        elif self.gate_open and (fi - self.last_voice_frame) * FRAME_MS > HANGOVER_MS:
            self.gate_open = False
            if self.state == "SPEECH":
                self.state = "LISTEN"

        if self.state in ("LISTEN", "SPEECH"):
            self._kws_step(fi, x)
        elif self.state == "STREAMING":
            self.stream.feed(x.tobytes())
            since_kw = (fi - self.kw_end_frame) * FRAME_MS
            silent = (fi - self.last_voice_frame) * FRAME_MS
            dur = (fi - self.stream_t0_frame) * FRAME_MS
            if self.manual:
                if dur >= MANUAL_MAX_MS:
                    self._end("max_len")
            elif dur >= MAX_STREAM_MS:
                self._end("max_len")
            elif since_kw >= MIN_AFTER_KW_MS and silent >= ENDPOINT_SILENCE_MS:
                self._end("silence")

    # -- KWS (runs only while the gate is open) ---------------------------------
    def _kws_step(self, fi: int, x: np.ndarray):
        t_now = now_us()
        if not self.gate_open or t_now < self.refractory_until:
            if self.kws is not None:
                # gate closed: flush the segment. A wake word followed by a
                # pause is decided here, since Vosk commits words late.
                rec, self.kws = self.kws, None
                t0 = time.perf_counter()
                if self.kws_buf:
                    rec.AcceptWaveform(self.kws_buf)
                    self.kws_buf = b""
                words = [w for w in json.loads(rec.FinalResult()).get("result", []) if w.get("word")]
                self.kws_busy_s += time.perf_counter() - t0
                if words:
                    self.kws_heard = " ".join(w["word"] for w in words)[-40:]
                self._check(words)
            return
        if self.kws is None:
            self.kws = asr._take_rec()
            back = min(BACKFILL_MS * SR // 1000, self.n - FRAME)
            self.kws_start = self.n - FRAME - back
            self.kws_buf = self._ring_read(self.kws_start, back)
        self.kws_buf += x.tobytes()
        self.f["kws_frames"] += 1
        if len(self.kws_buf) < asr.CHUNK_BYTES:
            return
        t0 = time.perf_counter()
        done = self.kws.AcceptWaveform(self.kws_buf)
        self.kws_buf = b""
        res = json.loads(self.kws.Result() if done else self.kws.PartialResult())
        self.kws_busy_s += time.perf_counter() - t0
        words = [w for w in res.get("result" if done else "partial_result", []) if w.get("word")]
        if words:
            self.kws_heard = " ".join(w["word"] for w in words)[-40:]
        self._check(words)

    def _check(self, words):
        hit = self._match(words)
        if hit:
            w0, w1 = hit
            start = self.kws_start + int(w0["start"] * SR)
            end = self.kws_start + int(w1["end"] * SR)
            self._wake(start, end, w0["word"] if w0 is w1 else f'{w0["word"]} {w1["word"]}', now_us())

    @staticmethod
    def _match(words):
        for i, w in enumerate(words):
            a = w["word"].lower()
            if i + 1 < len(words) and f'{a} {words[i + 1]["word"].lower()}' in KW_MERGES:
                return w, words[i + 1]
            if a in KW_WORDS:
                return w, w
        return None

    def _ring_read(self, start: int, count: int) -> bytes:
        start = max(start, self.n - self.ring.size)
        idx = np.arange(start, start + count) % self.ring.size
        return self.ring[idx].tobytes()

    # -- wake -> link -> stream -> end ------------------------------------------
    def _wake(self, kw_start, kw_end, word, t_detect):
        self.state = "DETECTED"
        self.wakes += 1
        self.kws = None
        manual = kw_start is None
        self.manual = manual
        if not manual:
            self.f["det"] += 1
        fi_now = self.n // FRAME
        kw_end = kw_end if kw_end is not None else self.n
        kw_start = kw_start if kw_start is not None else self.n
        fi_end = min(kw_end // FRAME, fi_now - 1)
        t_win_end = int(self.ring_rx_us[fi_end % self.ring_rx_us.size])
        self.kw_end_frame = kw_end // FRAME
        self.sid = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-mic{next(_ids)}"
        self._send({"type": "detect", "wake": self.wakes, "manual": manual, "heard": word,
                    "t_win_end_us": t_win_end, "t_detect_us": t_detect,
                    "l1_ms": None if manual else round((t_detect - t_win_end) / 1000, 1)}, self.sid)

        # "link": on the board this is Wi-Fi start + associate + socket. Here it
        # is only opening the ASR session, so it is reported as link_host_ms.
        self.state = "LINKING"
        self._send({"type": "state", "state": "LINKING", "rf": "ON"})
        t0 = now_us()
        sid = self.sid
        self.stream = asr.AsrStream(sid, lambda ev: self._send(ev, sid))
        t_link = now_us()
        pre = max(0, kw_start - PREROLL_BEFORE_KW_MS * SR // 1000)
        pre = max(pre, self.n - self.ring.size + FRAME)
        backlog = self._ring_read(pre, self.n - pre)
        if not manual:
            self.stream.kw_end_s = (kw_end - pre) / SR
        self.stream.feed(backlog)
        self.stream_t0_frame = pre // FRAME
        self._send({"type": "link", "link_host_ms": round((t_link - t0) / 1000, 1),
                    "l2_host_ms": round((t_link - t_detect) / 1000, 1),
                    "preroll_ms": len(backlog) * 1000 // (SR * 2)}, sid)
        self.state = "STREAMING"

    def _end(self, reason: str):
        st, sid = self.stream, self.sid
        self.stream = None
        self.state = "CLOSING"
        audio_ms = st.fed_bytes * 1000 // (SR * 2)
        t_end = now_us()
        self._send({"type": "stream_end", "reason": reason, "audio_ms": audio_ms}, sid)
        fut = st.finish()

        def done(f):
            try:
                self._send({"type": "final_ready", "final_after_end_ms":
                            round((now_us() - t_end) / 1000, 1)}, sid)
            finally:
                self.loop.call_soon_threadsafe(hub.close_session, sid)
        fut.add_done_callback(done)
        self.refractory_until = now_us() + REFRACTORY_MS * 1000
        self.state = "LISTEN"

    # -- 10 Hz telemetry -------------------------------------------------------
    def _tick(self):
        pt, wt, kb = self._cpu0
        pt1, wt1 = time.process_time(), time.perf_counter()
        if wt1 - wt >= 1.0:
            self.cpu_pct = round(100 * (pt1 - pt) / (wt1 - wt), 1)
            self.kws_duty = round(100 * (self.kws_busy_s - kb) / (wt1 - wt), 1)
            self._cpu0 = (pt1, wt1, self.kws_busy_s)
        self._send({"type": "tick", "state": self.state,
                    "rf": "ON" if self.state in ("LINKING", "STREAMING") else "OFF",
                    "level_db": round(self.level_db, 1), "peak_db": round(self.peak_db, 1),
                    "floor_db": round(self.floor_db, 1) if self.floor_db is not None else None,
                    "gate0_db": GATE0_DB, "s0": self.s0_now, "s1": self.s1_now,
                    "gate_open": self.gate_open, "funnel": self.f, "kws_heard": self.kws_heard,
                    "kws_duty": self.kws_duty, "proc_cpu": self.cpu_pct,
                    "stream_ms": ((self.n // FRAME - self.stream_t0_frame) * FRAME_MS
                                  if self.state == "STREAMING" else None),
                    "ring_kb": self.ring.nbytes // 1024, "uptime_s": self.n // SR})

"""udp_rx.py — device audio over UDP (CLAUDE.md §8), plus local telemetry relay.

Per session: reorder by seq (hold a gap <= REORDER_MS, then declare it lost and
insert 20 ms of silence so ASR timing stays aligned), PONG immediately, clock
sync from the min-RTT ping, stream to Vosk, TRANSCRIPT back on final.

Latency notes:
  * PONG is sent inside datagram_received before any other work, so the
    server-side turnaround is microseconds and doesn't inflate RTT.
  * In-order packets go straight to Vosk; only an actual gap waits.
"""

import asyncio
import datetime as dt
import json

import protocol as P
from asr import AsrStream
from events import hub, now_us

REORDER_MS = 60
IDLE_TIMEOUT_S = 3.0  # END lost -> finalise anyway
SILENCE_20MS = bytes(P.FRAME_SAMPLES * 2)


class Session:
    def __init__(self, rx: "UdpRx", addr, dev_session: int):
        self.rx, self.addr, self.dev_session = rx, addr, dev_session
        self.id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-udp{dev_session}"
        self.hello: dict = {}
        self.expected = 0
        self.pending: dict[int, bytes] = {}
        self.gap_timer: asyncio.TimerHandle | None = None
        self.first_rx_us = None
        self.first_audio_rx_us = None
        self.last_rx_us = now_us()
        self.pings: dict[int, int] = {}  # t_dev_us -> t_srv_rx_us
        self.best_rtt_us = None
        self.offset_us = None  # t_srv = t_dev + offset
        self.t_audio0_us = None  # device capture time of AUDIO seq 0
        self.n_audio = self.n_lost = self.n_late = 0
        self.ended = False
        self.asr = AsrStream(self.id, lambda ev: hub.emit_threadsafe(ev, self.id))
        hub.emit({"type": "session_start", "transport": "udp", "addr": f"{addr[0]}:{addr[1]}",
                  "dev_session": dev_session}, self.id)

    # -- packets ---------------------------------------------------------------
    def on_packet(self, ptype, codec, seq, t_cap, payload, t_rx):
        self.last_rx_us = t_rx
        if self.first_rx_us is None:
            self.first_rx_us = t_rx
            hub.emit({"type": "server_rx_first", "t_rx_us": t_rx, "ptype": ptype}, self.id)
        if ptype == P.HELLO:
            try:
                self.hello = json.loads(payload.decode("utf-8") or "{}")
            except ValueError:
                self.hello = {}
            hub.emit({"type": "hello", **self.hello}, self.id)
            self._maybe_latency()
            self._set_kw_end()
        elif ptype == P.AUDIO:
            if self.t_audio0_us is None:  # seq 0 may be lost: frames are 20 ms apart
                self.t_audio0_us = t_cap - seq * P.FRAME_MS * 1000
                self._set_kw_end()
            self._on_audio(seq, P.decode(codec, payload), t_rx)
        elif ptype == P.PING:
            self._on_ping(payload, t_rx)
        elif ptype == P.END:
            self.finish("end")

    def _set_kw_end(self):
        """Keyword end in ASR stream seconds. Both stamps are device clock, so
        no clock sync is involved. Stream time 0 = AUDIO seq 0 (lost packets
        are replaced by silence, so stream time stays aligned)."""
        t_end = self.hello.get("t_win_end_us")
        if t_end is not None and self.t_audio0_us is not None:
            self.asr.kw_end_s = max(0.0, (t_end - self.t_audio0_us) / 1e6)

    def _on_audio(self, seq, pcm, t_rx):
        if self.ended:
            return
        if self.first_audio_rx_us is None:
            self.first_audio_rx_us = t_rx
        self.n_audio += 1
        if seq < self.expected:
            self.n_late += 1  # arrived after we gave up on it, or duplicate
            return
        if seq == self.expected:
            self.asr.feed(pcm)
            self.expected += 1
            self._drain()
        else:
            self.pending[seq] = pcm
            if self.gap_timer is None:
                self.gap_timer = self.rx.loop.call_later(REORDER_MS / 1000, self._gap_expired)

    def _drain(self):
        while self.expected in self.pending:
            self.asr.feed(self.pending.pop(self.expected))
            self.expected += 1
        if not self.pending and self.gap_timer:
            self.gap_timer.cancel()
            self.gap_timer = None

    def _gap_expired(self):
        self.gap_timer = None
        if not self.pending:
            return
        nxt = min(self.pending)
        lost = nxt - self.expected
        if lost > 0:
            self.n_lost += lost
            hub.emit({"type": "pkt_loss", "from_seq": self.expected, "count": lost}, self.id)
            for _ in range(lost):
                self.asr.feed(SILENCE_20MS)
            self.expected = nxt
        self._drain()
        if self.pending:
            self.gap_timer = self.rx.loop.call_later(REORDER_MS / 1000, self._gap_expired)

    def _on_ping(self, payload, t_rx):
        if len(payload) >= P.PING_BASE.size:
            (t_dev,) = P.PING_BASE.unpack_from(payload)
            self.pings[t_dev] = t_rx
        if len(payload) >= P.PING_FULL.size:
            _, prev_t_dev, rtt = P.PING_FULL.unpack_from(payload)
            t_srv = self.pings.get(prev_t_dev)
            if t_srv is not None and (self.best_rtt_us is None or rtt < self.best_rtt_us):
                self.best_rtt_us = rtt
                # the PING left the device at prev_t_dev and reached us ~RTT/2 later
                self.offset_us = t_srv - (prev_t_dev + rtt // 2)
                hub.emit({"type": "clock_sync", "rtt_us": rtt, "offset_us": self.offset_us,
                          "error_bound_us": rtt // 2}, self.id)
                self._maybe_latency()

    # -- latency waterfall (CLAUDE.md §7) ----------------------------------------
    def latency(self) -> dict:
        h, out = self.hello, {}
        if "t_detect_us" in h and "t_win_end_us" in h:
            out["L1_ms"] = (h["t_detect_us"] - h["t_win_end_us"]) / 1000
        if "t_sock_us" in h and "t_detect_us" in h:
            out["L2_ms"] = (h["t_sock_us"] - h["t_detect_us"]) / 1000
        if self.offset_us is not None and "t_sock_us" in h and self.first_rx_us:
            out["L3_ms"] = (self.first_rx_us - self.offset_us - h["t_sock_us"]) / 1000
            out["L3_error_ms"] = self.best_rtt_us / 2000
        if self.asr.t_first_text_us and self.first_audio_rx_us:
            out["L4_ms"] = (self.asr.t_first_text_us - self.first_audio_rx_us) / 1000
        if all(k in out for k in ("L1_ms", "L2_ms", "L3_ms")):
            out["total_ms"] = out["L1_ms"] + out["L2_ms"] + out["L3_ms"]
        if self.offset_us is not None and "t_win_end_us" in h and self.asr.t_first_text_us:
            out["keyword_end_to_first_text_ms"] = (
                self.asr.t_first_text_us - self.offset_us - h["t_win_end_us"]) / 1000
        return {k: round(v, 2) for k, v in out.items()}

    def _maybe_latency(self):
        lat = self.latency()
        if "L3_ms" in lat:
            hub.emit({"type": "latency", "partial": True, **lat}, self.id)

    # -- end -------------------------------------------------------------------
    def finish(self, reason: str):
        if self.ended:
            return
        self.ended = True
        if self.gap_timer:
            self.gap_timer.cancel()
            self.gap_timer = None
        # anything still pending after END: fill gaps as lost, keep order
        for seq in sorted(self.pending):
            if seq > self.expected:
                self.n_lost += seq - self.expected
                for _ in range(seq - self.expected):
                    self.asr.feed(SILENCE_20MS)
            self.asr.feed(self.pending[seq])
            self.expected = seq + 1
        self.pending.clear()
        self.t_end_us = now_us()
        self.rx.loop.create_task(self._finalise(reason))

    async def _finalise(self, reason):
        try:
            res = await asyncio.wrap_future(self.asr.finish())
        except Exception:
            res = {"text": "", "text_raw": "", "kw": None}
        text = res["text"]
        t_final = now_us()
        data = text.encode("utf-8")[:60]
        while data:  # don't cut a multi-byte char in half
            try:
                data.decode("utf-8")
                break
            except UnicodeDecodeError:
                data = data[:-1]
        self.rx.send(self.addr, P.pack(self.dev_session, P.TRANSCRIPT, 0, t_final, data))
        hub.emit({"type": "session_end", "reason": reason, "text": text,
                  "text_raw": res["text_raw"], "kw": res["kw"],
                  "packets": self.n_audio, "lost": self.n_lost, "late": self.n_late,
                  "end_to_final_ms": round((t_final - self.t_end_us) / 1000, 1),
                  "device_stats": self.hello.get("end_stats"),
                  "latency": self.latency()}, self.id)
        hub.close_session(self.id)
        self.rx.sessions.pop((self.addr, self.dev_session), None)


class UdpRx(asyncio.DatagramProtocol):
    def __init__(self):
        self.sessions: dict[tuple, Session] = {}
        self.loop = asyncio.get_running_loop()
        self.transport = None
        self._reaper = self.loop.create_task(self._reap())

    def connection_made(self, transport):
        self.transport = transport

    def send(self, addr, data: bytes):
        if self.transport:
            self.transport.sendto(data, addr)

    def datagram_received(self, data, addr):
        t_rx = now_us()
        pkt = P.unpack(data)
        if pkt is None:
            return
        dev_session, ptype, codec, seq, t_cap, payload = pkt
        if ptype == P.PING and len(payload) >= P.PING_BASE.size:
            # reply FIRST — keeps our turnaround out of the RTT
            (t_dev,) = P.PING_BASE.unpack_from(payload)
            self.send(addr, P.pack(dev_session, P.PONG, seq, t_rx, P.PONG_PAYLOAD.pack(t_dev, t_rx)))
        key = (addr, dev_session)
        s = self.sessions.get(key)
        if s is None:
            if ptype == P.END:
                return  # stray END for a session we already closed
            s = self.sessions[key] = Session(self, addr, dev_session)
        s.on_packet(ptype, codec, seq, t_cap, payload, t_rx)

    def error_received(self, exc):
        pass  # Windows reports ICMP port-unreachable here; harmless for UDP

    async def _reap(self):
        while True:
            await asyncio.sleep(0.5)
            t = now_us()
            for s in list(self.sessions.values()):
                if not s.ended and (t - s.last_rx_us) / 1e6 > IDLE_TIMEOUT_S:
                    s.finish("idle_timeout")


class TelemetryRelay(asyncio.DatagramProtocol):
    """tools/serial_logger.py --forward sends device JSON lines here (localhost),
    so the console gets device telemetry and server events on one WebSocket."""

    def datagram_received(self, data, addr):
        try:
            ev = json.loads(data.decode("utf-8"))
        except ValueError:
            return
        if isinstance(ev, dict):
            ev["source"] = "device"
            hub.emit(ev)

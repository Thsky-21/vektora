"""test_sender.py — plays a WAV at the server exactly like the device would.

UDP (the §8 protocol, default):
    python test_sender.py clip.wav [--host 127.0.0.1] [--preroll-ms 700]
        [--loss 0.02] [--reorder 0.05] [--skew-ms 5000] [--trials 5]
  HELLO -> 3 PINGs interleaved with a 4x-real-time backlog burst of the
  pre-roll -> real-time audio -> END -> waits for TRANSCRIPT.
  --skew-ms offsets the fake device clock, to prove clock sync recovers it.

HTTP (what the current firmware's asr_uploader.c does):
    python test_sender.py clip.wav --http [--url http://127.0.0.1:8000/asr]
  Streams PCM16 in 4 KB chunks at real time with an exact Content-Length.

Prints per-trial end->transcript latency; the server's session_end event (in
sessions/<id>.jsonl and on /ws/events) has the full L1-L4 waterfall.
"""

import argparse
import http.client
import json
import random
import socket
import statistics
import sys
import threading
import time
import urllib.parse
import wave

import numpy as np

import protocol as P


def load_wav(path) -> np.ndarray:
    with wave.open(path, "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        sys.exit(f"{path}: need 16-bit PCM, got {8 * sw}-bit")
    x = np.frombuffer(raw, dtype="<i2").reshape(-1, ch).mean(axis=1)
    if sr != P.SR:
        t = np.arange(0, len(x) / sr, 1 / P.SR)
        x = np.interp(t, np.arange(len(x)) / sr, x)
    return x.astype("<i2")


def dev_clock(skew_us):
    return lambda: time.perf_counter_ns() // 1000 + skew_us  # 100 ns res on Windows


def udp_trial(pcm, a, session_id):
    now = dev_clock(int(a.skew_ms * 1000))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.2)
    dst = (a.host, P.UDP_PORT)
    pongs, got = {}, {}
    stop = threading.Event()

    def rx():
        while not stop.is_set():
            try:
                data, _ = sock.recvfrom(2048)
            except (socket.timeout, OSError):
                continue
            pkt = P.unpack(data)
            if not pkt:
                continue
            _, ptype, _, _, _, payload = pkt
            if ptype == P.PONG:
                t_dev, _ = P.PONG_PAYLOAD.unpack_from(payload)
                pongs[t_dev] = now() - t_dev  # RTT
            elif ptype == P.TRANSCRIPT:
                got["text"] = payload.decode("utf-8", "replace")
                got["t"] = time.perf_counter()

    threading.Thread(target=rx, daemon=True).start()

    frames = [pcm[i:i + P.FRAME_SAMPLES] for i in range(0, len(pcm), P.FRAME_SAMPLES)]
    t_detect = now()
    t_win_end = t_detect - int(a.l1_ms * 1000)
    t_audio0 = t_detect - a.preroll_ms * 1000  # frame 0 was captured this long ago
    if a.kw_end_ms is not None:  # where the keyword really ends in the WAV
        t_win_end = t_audio0 + int(a.kw_end_ms * 1000)
    t_sock = now()
    hello = {"score": 0.97, "t_win_end_us": t_win_end, "t_detect_us": t_detect, "t_sock_us": t_sock,
             "wifi_ms": 0, "fw": "test_sender", "cfg": "-", "codec": "pcm16"}
    sock.sendto(P.pack(session_id, P.HELLO, 0, t_sock, json.dumps(hello).encode()), dst)

    held = None  # for --reorder: send this one after the next
    last_ping, n_pings, prev = 0.0, 0, None
    backlog_rate = 4.0
    t_send0 = time.perf_counter()
    for i, f in enumerate(frames):
        cap_us = t_audio0 + i * P.FRAME_MS * 1000
        # pace: backlog at 4x real time until we've caught up with "now"
        due_backlog = t_send0 + i * P.FRAME_MS / 1000 / backlog_rate
        due_real = t_send0 + (cap_us - t_detect) / 1e6
        wait = max(due_backlog, due_real) - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        if n_pings < 3 and time.perf_counter() - last_ping > 0.03:
            t = now()
            if prev and prev in pongs:
                payload = P.PING_FULL.pack(t, prev, pongs[prev])
            else:
                payload = P.PING_BASE.pack(t)
            sock.sendto(P.pack(session_id, P.PING, 0, t, payload), dst)
            prev, last_ping, n_pings = t, time.perf_counter(), n_pings + 1
        pkt = P.pack(session_id, P.AUDIO, i, cap_us, f.tobytes())
        if random.random() < a.loss:
            continue
        if held is not None:
            sock.sendto(pkt, dst)
            sock.sendto(held, dst)
            held = None
        elif random.random() < a.reorder:
            held = pkt
        else:
            sock.sendto(pkt, dst)
    if held is not None:
        sock.sendto(held, dst)
    # one last ping carrying the final RTT sample
    time.sleep(0.02)
    if prev in pongs:
        t = now()
        sock.sendto(P.pack(session_id, P.PING, 0, t, P.PING_FULL.pack(t, prev, pongs[prev])), dst)
    t_end = time.perf_counter()
    sock.sendto(P.pack(session_id, P.END, 0, now(), b""), dst)
    deadline = t_end + 10
    while "t" not in got and time.perf_counter() < deadline:
        time.sleep(0.005)
    stop.set()
    sock.close()
    if "t" not in got:
        print(f"  session {session_id}: NO TRANSCRIPT within 10 s")
        return None
    ms = (got["t"] - t_end) * 1000
    rtts = sorted(pongs.values())
    print(f"  session {session_id}: END->TRANSCRIPT {ms:6.1f} ms | min RTT {rtts[0] / 1000 if rtts else float('nan'):.2f} ms"
          f" | \"{got['text']}\"")
    return ms


def http_trial(pcm, a):
    u = urllib.parse.urlparse(a.url)
    body = pcm.tobytes()
    t0 = time.perf_counter()
    pre = a.preroll_ms * P.SR * 2 // 1000  # sent immediately, like the firmware's ring

    def gen():
        sent = 0
        while sent < len(body):
            n = min(4096, len(body) - sent)
            due = t0 + max(0, sent + n - pre) / (P.SR * 2)
            w = due - time.perf_counter()
            if w > 0:
                time.sleep(w)
            yield body[sent:sent + n]
            sent += n
        state["t_last"] = time.perf_counter()

    state = {}
    c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=30)
    c.request("POST", u.path or "/asr", body=gen(),
              headers={"Content-Type": "application/octet-stream", "Content-Length": str(len(body))})
    r = c.getresponse()
    data = json.loads(r.read())
    t_resp = time.perf_counter()
    ms = (t_resp - state["t_last"]) * 1000
    print(f"  HTTP {r.status}: last byte->response {ms:6.1f} ms | server says final {data.get('final_after_last_byte_ms')} ms"
          f" | \"{data.get('text')}\"")
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--http", action="store_true")
    ap.add_argument("--url", default="http://127.0.0.1:8000/asr")
    ap.add_argument("--preroll-ms", type=int, default=700)
    ap.add_argument("--l1-ms", type=float, default=20.0, help="fake detect latency put in HELLO")
    ap.add_argument("--kw-end-ms", type=float, default=None,
                    help="keyword end in the WAV (ms); sets HELLO t_win_end_us to match")
    ap.add_argument("--loss", type=float, default=0.0)
    ap.add_argument("--reorder", type=float, default=0.0)
    ap.add_argument("--skew-ms", type=float, default=0.0)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    pcm = load_wav(a.wav)
    print(f"{a.wav}: {len(pcm) / P.SR:.2f} s, {'HTTP ' + a.url if a.http else 'UDP ' + a.host + ':' + str(P.UDP_PORT)}")
    res = []
    for k in range(a.trials):
        ms = http_trial(pcm, a) if a.http else udp_trial(pcm, a, (int(time.time()) + k) & 0xFFFF)
        if ms is not None:
            res.append(ms)
    if len(res) > 1:
        res.sort()
        print(f"median {statistics.median(res):.1f} ms, p90 {res[int(0.9 * (len(res) - 1))]:.1f} ms, n={len(res)}")


if __name__ == "__main__":
    main()

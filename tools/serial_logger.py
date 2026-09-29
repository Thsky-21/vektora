"""serial_logger.py — log the device's USB telemetry (CLAUDE.md §5.7) to JSONL.

    python tools/serial_logger.py --port COM5 --out sessions/m1_baseline.jsonl [--forward]
    python tools/serial_logger.py --list

Every line from the board is stored with host timestamps. JSON lines are kept
as objects; anything else (ESP_LOG output, boot banner) is kept as
{"type":"log","line":...} so a crash mid-soak is still in the record.
--forward also relays JSON lines to the server (127.0.0.1:5006), so the web
console shows device telemetry without Web Serial (one process owns the COM
port, and this one survives a multi-hour M4 soak better than a browser tab).
Reconnects automatically if the board resets or the cable blips.
"""

import argparse
import datetime as dt
import json
import re
import socket
import sys
import time

import serial
from serial.tools import list_ports

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--out")
    ap.add_argument("--forward", action="store_true", help="relay JSON to the server on 127.0.0.1:5006")
    ap.add_argument("--forward-port", type=int, default=5006)
    ap.add_argument("--quiet", action="store_true", help="don't echo tick lines")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list or not a.port:
        for p in list_ports.comports():
            print(f"{p.device:8} {p.description}")
        if not a.port:
            sys.exit(0 if a.list else "need --port (see --list)")
    out = open(a.out or f"sessions/serial-{dt.datetime.now():%Y%m%d-%H%M%S}.jsonl", "a",
               encoding="utf-8", buffering=1)
    fwd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if a.forward else None
    n_json = n_log = n_bad = 0
    t_start = time.perf_counter()
    last_status = 0.0
    print(f"logging {a.port} @ {a.baud} -> {out.name}" + (" (+forward)" if fwd else ""))

    while True:
        try:
            with serial.Serial(a.port, a.baud, timeout=1) as ser:
                out.write(json.dumps({"type": "logger", "event": "open", "port": a.port,
                                      "host_wall": dt.datetime.now().isoformat(timespec="milliseconds")}) + "\n")
                buf = b""
                while True:
                    buf += ser.read(ser.in_waiting or 1)
                    while b"\n" in buf:
                        raw, buf = buf.split(b"\n", 1)
                        line = ANSI.sub("", raw.decode("utf-8", "replace")).strip()
                        if not line:
                            continue
                        host = {"host_us": time.perf_counter_ns() // 1000,
                                "host_wall": dt.datetime.now().isoformat(timespec="milliseconds")}
                        rec = None
                        if line.startswith("{"):
                            try:
                                rec = json.loads(line)
                            except ValueError:
                                n_bad += 1  # torn line (reset mid-print): keep it as text
                        if isinstance(rec, dict):
                            n_json += 1
                            rec.update(host)
                            out.write(json.dumps(rec) + "\n")
                            if fwd:
                                fwd.sendto(json.dumps(rec).encode(), ("127.0.0.1", a.forward_port))
                            if not (a.quiet and rec.get("type") == "tick"):
                                print(line[:160])
                        else:
                            n_log += 1
                            out.write(json.dumps({"type": "log", "line": line, **host}) + "\n")
                            print("  " + line[:160])
                    if time.perf_counter() - last_status > 60:
                        last_status = time.perf_counter()
                        print(f"[logger] {(last_status - t_start) / 60:.0f} min, json {n_json}, log {n_log}, torn {n_bad}",
                              file=sys.stderr)
        except serial.SerialException as e:
            out.write(json.dumps({"type": "logger", "event": "lost", "error": str(e),
                                  "host_wall": dt.datetime.now().isoformat(timespec="milliseconds")}) + "\n")
            print(f"[logger] port lost ({e}); retrying in 1 s", file=sys.stderr)
            time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n[logger] stopped. json {n_json}, log {n_log}, torn {n_bad} -> {out.name}")
            return


if __name__ == "__main__":
    main()

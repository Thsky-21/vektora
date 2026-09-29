"""analyze.py — turn recorded runs into docs/results.md rows (CLAUDE.md §11, T9).

    python tools/analyze.py sessions/m2_quiet.jsonl --run M2 --room "quiet lab, fan off"
    python tools/analyze.py sessions/m4_soak.jsonl --run M4 --negative --append
    python tools/analyze.py sessions/m5_ledger.json --run M5 --append        # console recording with trials
    python tools/analyze.py sessions/2026*-udp*.jsonl --run M6 --append       # server sessions

Input, any mix of:
  * tools/serial_logger.py output (device JSON: boot/tick/detect/link/stream_end/transcript)
  * server sessions/<id>.jsonl (session_start/hello/.../session_end with the L1-L4 latency)
  * a console "Download recording" JSON (array of events; may hold Ledger `trial` markers)

Only numbers present in the files are reported; anything missing prints "—"
(Rule 0.1). Runs whose HELLO came from test_sender.py are tagged SIMULATED
and never written as device evidence. --append adds rows to docs/results.md
under "## Device runs (tools/analyze.py)".
"""

import argparse
import datetime as dt
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results.md"
SECTION = "## Device runs (tools/analyze.py)"
TRIAL_WINDOW_MS = 3000  # a detect within this long after a trial marker counts


def load(paths):
    evs = []
    for p in paths:
        text = Path(p).read_text(encoding="utf-8", errors="replace")
        try:
            j = json.loads(text)
            rows = j if isinstance(j, list) else j.get("events", [])
        except ValueError:
            rows = []
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("{"):
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        pass  # torn line from a reset: skip
        for r in rows:
            if isinstance(r, dict):
                r["_file"] = str(p)
                evs.append(r)
    return evs


def q(xs, p):
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]


def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.median(xs) if xs else None


def f(v, d=1, unit=""):
    return "—" if v is None else f"{v:.{d}f}{unit}"


def git_hash():
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return (h or "?") + ("+wip" if dirty else "")
    except OSError:
        return "?"


def analyze(evs, negative=False):
    # Device lines come from serial_logger.py (no "session") or the server's
    # relay (source=device). The laptop-mic emulator (server/mic_emu.py) writes
    # the same event types into "<id>-micN" sessions: never device evidence.
    kinds = ("boot", "tick", "detect", "link", "stream_end", "transcript")
    emulated = any(e.get("type") in kinds and "-mic" in str(e.get("session", "")) for e in evs)
    dev = [e for e in evs if e.get("type") in kinds and ("session" not in e or e.get("source") == "device")]
    if emulated and not dev:  # show the emulator's numbers, clearly tagged
        dev = [e for e in evs if e.get("type") in kinds]
    ticks = [e for e in dev if e.get("type") == "tick"]
    boots = [e for e in dev if e.get("type") == "boot"]
    detects = [e for e in dev if e.get("type") == "detect"]
    links = [e for e in dev if e.get("type") == "link"]
    ends = [e for e in dev if e.get("type") == "stream_end"]
    hellos = [e for e in evs if e.get("type") == "hello"]
    sess_end = [e for e in evs if e.get("type") == "session_end"]
    trials = [e for e in evs if e.get("type") == "trial"]

    simulated = bool(hellos) and all(h.get("fw") == "test_sender" for h in hellos) and not ticks
    ident = {}
    src = (boots or ticks or [{}])[-1]
    for k in ("firmware_hash", "config_hash", "threshold", "gate"):
        if k in src:
            ident[k] = src[k]
    fws = sorted({h.get("fw") for h in hellos if h.get("fw")})

    out = {"simulated": simulated, "emulated": emulated and not any("session" not in e for e in dev),
           "ident": ident, "hello_fw": fws, "rows": []}
    row = out["rows"].append

    # --- duration --------------------------------------------------------------
    t = [e["t_us"] for e in ticks if isinstance(e.get("t_us"), (int, float))]
    hours = (max(t) - min(t)) / 3.6e9 if len(t) > 1 else None
    out["hours"] = hours
    if hours is not None:
        row(("duration", f"{hours * 60:.1f} min ({len(ticks)} ticks)", "device esp_timer, first→last tick"))

    # --- CPU (C2): LISTEN ticks only, the "idle" definition -------------------
    idle = [e for e in ticks if e.get("state") == "LISTEN"]
    for key, name in (("cpu_avg", "CPU avg"), ("cpu0", "CPU core 0"), ("cpu1", "CPU core 1")):
        v = [e.get(key) for e in idle]
        m = med(v)
        if m is not None:
            row((f"{name}, LISTEN", f"median {f(m, 2)} %, p90 {f(q(v, 0.9), 2)} % (n={sum(x is not None for x in v)})",
                 "FreeRTOS run-time stats, 100 − IDLE share, 1 s windows"))
    allv = [e.get("cpu_avg") for e in ticks]
    if med(allv) is not None and len(idle) != len(ticks):
        row(("CPU avg, all states", f"median {f(med(allv), 2)} %, p90 {f(q(allv, 0.9), 2)} %", "same, incl. SPEECH/LINKING/STREAMING"))
    duty = [e.get("dscnn_duty") for e in ticks]
    if med(duty) is not None:
        row(("DS-CNN duty", f"mean {f(statistics.fmean([x for x in duty if x is not None]), 2)} %, "
             f"max {f(max(x for x in duty if x is not None), 1)} %", "infer task run-time share"))

    # --- RAM (C1) --------------------------------------------------------------
    ru = [e.get("ram_used") for e in idle]
    if med(ru) is not None:
        row(("RAM used, LISTEN", f"median {med(ru) / 1024:.1f} KB (static {f((src.get('ram_static') or 0) / 1024)} KB)",
             "static .data+.bss+.noinit + internal heap used"))
    rp = [e.get("ram_peak") for e in ticks if isinstance(e.get("ram_peak"), (int, float))]
    if rp:
        row(("RAM peak", f"{max(rp) / 1024:.1f} KB", "static + (heap total − min-free-ever watermark)"))

    # --- gate funnel -----------------------------------------------------------
    fun = [e.get("funnel") for e in ticks if isinstance(e.get("funnel"), dict)]
    if len(fun) > 1:
        a, b = fun[0], fun[-1]
        d = {k: (b.get(k, 0) or 0) - (a.get(k, 0) or 0) for k in ("frames", "s0", "s1", "opens", "infer", "det")}
        if d["frames"] > 0:
            row(("gate funnel", f"frames {d['frames']} → S0 {100 * d['s0'] / d['frames']:.2f} % → S1 "
                 f"{100 * d['s1'] / d['frames']:.2f} % → opens {d['opens']} → infer {d['infer']} → det {d['det']}",
                 "tick counters, last − first"))

    # --- detections, FA/hr (C4) -----------------------------------------------
    if ticks or detects:
        n = len(detects)
        if negative and hours:
            row(("false accepts", f"{n} in {hours:.2f} h = {n / hours:.2f} FA/h", "negative-only audio (--negative)"))
        else:
            row(("detections", str(n), "detect events"))
    l1 = [e.get("l1_ms") for e in detects]
    if med(l1) is not None:
        row(("L1 detect", f"median {f(med(l1))} ms, p90 {f(q(l1, 0.9))} ms (n={len(l1)})", "t_detect − t_win_end, device clock"))
    inf = [e.get("infer_us") for e in detects]
    if med(inf) is not None:
        row(("inference time", f"median {med(inf) / 1000:.1f} ms (n={len(inf)})", "esp_timer around Invoke()"))

    # --- link (L2) ---------------------------------------------------------------
    ok = [e for e in links if e.get("ok")]
    if links:
        w = [e.get("wifi_ms") for e in ok if (e.get("wifi_ms") or -1) >= 0]
        l2 = [e.get("l2_ms") for e in ok if (e.get("l2_ms") or -1) >= 0]
        row(("link success", f"{len(ok)}/{len(links)}", "link events"))
        if w:
            row(("WiFi connect", f"median {f(med(w))} ms, p90 {f(q(w, 0.9))} ms (n={len(w)})", "t_ip − t_wifi_start"))
        if l2:
            row(("L2 wake→socket", f"median {f(med(l2))} ms, p90 {f(q(l2, 0.9))} ms (n={len(l2)})", "t_sock − t_wake, device clock"))
    if ends:
        ovf = max((e.get("ring_ovf") or 0) for e in ends)
        row(("stream ends", f"{sum(bool(e.get('ok')) for e in ends)}/{len(ends)} ok, ring_ovf {ovf}",
             "stream_end events"))

    # --- server waterfall (L1-L4) ------------------------------------------------
    lat = [e.get("latency") or {} for e in sess_end if (e.get("latency") or {})]
    for k, name in (("L1_ms", "L1"), ("L2_ms", "L2"), ("L3_ms", "L3"), ("L4_ms", "L4"), ("total_ms", "L1+L2+L3"),
                    ("keyword_end_to_first_text_ms", "keyword end → first text")):
        v = [x.get(k) for x in lat]
        if med(v) is not None:
            row((f"server {name}", f"median {f(med(v))} ms, p90 {f(q(v, 0.9))} ms (n={sum(x is not None for x in v)})",
                 "session_end.latency (PING/PONG clock sync ±RTT/2)"))
    lost = [e.get("lost") for e in sess_end if e.get("lost") is not None]
    if lost:
        row(("UDP packets lost", f"{sum(lost)} over {len(lost)} sessions", "session_end.lost (seq gaps)"))

    # --- Ledger trials (M5 TPR / confusables) -------------------------------------
    if trials:
        det_t = sorted(e["rx_ms"] for e in evs if e.get("type") == "detect" and isinstance(e.get("rx_ms"), (int, float)))
        groups = {}
        for tr in trials:
            t0 = tr.get("rx_ms")
            hit = any(t0 is not None and t0 <= d <= t0 + TRIAL_WINDOW_MS for d in det_t)
            key = (tr.get("kind"), tr.get("label"))
            g = groups.setdefault(key, [0, 0])
            g[0] += hit
            g[1] += 1
        for (kind, label), (h, n) in sorted(groups.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
            what = "TPR" if kind == "positive" else "false accepts"
            row((f"{what} {label}", f"{h}/{n} = {100 * h / n:.0f} %", f"console Ledger trials, detect ≤{TRIAL_WINDOW_MS / 1000:.0f} s after marker"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--run", default="—", help="protocol id, e.g. M2")
    ap.add_argument("--room", default="", help="room / mic distance description")
    ap.add_argument("--negative", action="store_true", help="the audio contained no keyword (M4 soak): detections = FA")
    ap.add_argument("--append", action="store_true", help=f"append rows to {RESULTS.relative_to(ROOT)}")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):  # Windows console: arrows, em dashes
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    evs = load(a.files)
    if not evs:
        sys.exit("no events in those files")
    res = analyze(evs, a.negative)
    idt = res["ident"]
    cfg = f"{a.run}; fw {idt.get('firmware_hash', '—')}, cfg {idt.get('config_hash', '—')}, gate {idt.get('gate', '—')}, thr {idt.get('threshold', '—')}"
    if a.room:
        cfg += f"; {a.room}"
    tag = ("SIMULATED (test_sender, not device evidence)" if res["simulated"]
           else "EMULATED (laptop mic via server/mic_emu.py, not device evidence)" if res["emulated"] else "device")
    print(f"{len(evs)} events from {len(a.files)} file(s) — {tag}")
    print(f"config: {cfg}\n")
    if not res["rows"]:
        print("nothing measurable in these files")
        return
    w = max(len(r[0]) for r in res["rows"])
    for m, v, how in res["rows"]:
        print(f"  {m:<{w}}  {v}")

    if a.append:
        if res["simulated"] or res["emulated"]:
            sys.exit(f"\nrefusing to append: {tag.split(' ')[0].lower()} run is not device evidence (Rule 0.1)")
        date, gh = dt.date.today().isoformat(), git_hash()
        text = RESULTS.read_text(encoding="utf-8") if RESULTS.exists() else "# Results\n"
        if SECTION not in text:
            text = text.rstrip() + f"\n\n{SECTION}\n\nWritten by `tools/analyze.py` from session files; source file in the Method column.\n\n| Date | Git | Config | Metric | Value | Method |\n|---|---|---|---|---|---|\n"
        files = ", ".join(Path(p).name for p in a.files[:3]) + (" …" if len(a.files) > 3 else "")
        rows = "".join(f"| {date} | {gh} | {cfg} | {m} | {v} | {how}; `{files}` |\n" for m, v, how in res["rows"])
        RESULTS.write_text(text.rstrip("\n") + "\n" + rows, encoding="utf-8")
        print(f"\nappended {len(res['rows'])} rows to {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

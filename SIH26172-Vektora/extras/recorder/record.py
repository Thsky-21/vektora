"""
Vektora recorder — PC side. Pairs with firmware/ (flash that to the ESP32 first).

Records the INMP441 through the real board, so the training data is captured by
exactly the chain the detector will use (same mic, same 16 kHz, same 32->16 bit
conversion). Three modes:

  level   live level meter. Use it first to check the mic works and to pick
          a speaking distance. Ctrl-C to stop.

  clips   guided session for ONE speaker: walks through the positive
          ("Vektora") script and the hard-negative word list, one take at a
          time. Each take records a 2.5 s window, runs quality checks, and
          cuts a 1.0 s clip with the word centred. Failed takes are re-asked.

  long    continuous recording to WAV (room noise, a speaker's 2-5 minute
          session). Stops on Ctrl-C or after --minutes.

Run from the repo root with the project venv, e.g.:
  .venv/Scripts/python.exe extras/recorder/record.py level
  .venv/Scripts/python.exe extras/recorder/record.py clips --speaker S01 --gender F
  .venv/Scripts/python.exe extras/recorder/record.py long --name fan_speed3 --kind noise --minutes 10

Output lands under dataset/ (git-ignored, because it contains voices):
  recordings/<speaker>/clips/pos_S01_1m-normal_001.wav   1.0 s training clip
  recordings/<speaker>/raw/pos_S01_1m-normal_001.wav     the full 2.5 s take
  recordings/takes.csv      one row per saved take, with every QC number
  recordings/speakers.csv   one row per speaker (gender, device, room, date)
  long/<kind>/<name>_<date>_partN.wav

WHY THE QUALITY CHECKS (each one traces back to a real problem in the old data,
CLAUDE.md §9.5 and §10.6-10.7):
  * dropped audio  -> any lost packet or ESP32 overflow inside a take rejects it.
                      The old "chopped" files had runs of exact zeros from lost
                      samples and taught the model a fake cue. A take is never
                      stitched across a gap.
  * clipping       -> >= 5 samples at full scale rejects the take.
  * word at edge   -> the word must not touch the start or end of the 2.5 s
                      window (spoke too early / too late).
  * no speech      -> the loudest part must be >= 15 dB above the room floor.
  * word too long  -> a word that cannot fit in 1 s with 150 ms of room tone on
                      each side is KEPT but flagged (never cut short).
"""

import argparse
import csv
import datetime as dt
import os
import struct
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

try:
    import serial
except ImportError:
    sys.exit("pyserial missing: .venv/Scripts/python.exe -m pip install pyserial")

SR = 16000
FRAME = 160
MAGIC = b"VKA1"
PKT = 4 + 4 + 4 + 2 + 2 + 2 * FRAME + 2           # 338 bytes, see firmware header

ROOT = Path(__file__).resolve().parents[2]         # repo root
DATA = ROOT / "dataset"
REC = DATA / "recordings"

# ---------------------------------------------------------------------------
# The per-speaker script (CLAUDE.md §13.6). Condition tags use '-' so that the
# file name  <class>_<speaker>_<condition>_<take>.wav  splits cleanly on '_'.
# ---------------------------------------------------------------------------
POSITIVE_SCRIPT = [
    # (condition tag, instruction shown to the speaker, takes, mode)
    ("1m-normal", "1 m from the mic, normal pace",                  6, "word"),
    ("1m-fast",   "1 m, FAST and casual, like talking to a friend", 4, "word"),
    ("1m-slow",   "1 m, SLOW and deliberate",                       4, "word"),
    ("1m-soft",   "1 m, SOFT voice, almost muttered",               4, "word"),
    ("3m-loud",   "3 m away, calling out across the room",          4, "word"),
    ("3m-normal", "3 m away, NORMAL volume",                        4, "word"),
    ("30cm",      "30 cm from the mic, normal voice",               4, "word"),
    ("1m-away",   "1 m, turned sideways or facing AWAY from the mic", 4, "word"),
    ("1m-phrase", "1 m: 'Vektora, <any command>' e.g. 'Vektora, what time is it'", 6, "phrase"),
]

HARD_NEGATIVES = [
    # (word, takes). Near-collisions get 2 takes: they matter most.
    ("vector", 2), ("victor", 2), ("vectra", 2), ("spectra", 2), ("sector", 2),
    ("doctor", 1), ("actor", 1), ("factor", 1), ("detector", 1), ("projector", 1), ("director", 1),
    ("very", 1), ("video", 1), ("voice", 1), ("victory", 1),
    ("camera", 1), ("banana", 1), ("tomorrow", 1), ("extra", 1),
    # Partial keyword. Frequently forgotten, and the model fires early without them.
    ("vek", 1), ("tora", 1), ("vekto", 1), ("ektora", 1),
    ("model", 1), ("training", 1), ("dataset", 1),
]

# ---------------------------------------------------------------------------
# Serial reader: parses packets, checks CRC / sequence / overflow, keeps the
# last RING_S seconds in a ring buffer and remembers where any gap happened.
# ---------------------------------------------------------------------------
RING_S = 30


# CRC-16/CCITT-FALSE, identical to the firmware's. Table-driven, because the
# bit-by-bit version is slow in Python at 100 packets a second.
_CRC_TAB = []
for _i in range(256):
    _c = _i << 8
    for _ in range(8):
        _c = ((_c << 1) ^ 0x1021) & 0xFFFF if _c & 0x8000 else (_c << 1) & 0xFFFF
    _CRC_TAB.append(_c)


def crc16_fast(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC_TAB[((crc >> 8) ^ b) & 0xFF]
    return crc


class Stream:
    def __init__(self, port: str, baud: int):
        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = baud
        self.ser.timeout = 0.2
        # DTR/RTS low = neither EN nor IO0 pulled: opening the port must NOT
        # reset the board or put it in download mode.
        self.ser.dtr = False
        self.ser.rts = False
        self.ser.open()
        self.ring = np.zeros(RING_S * SR, dtype=np.int16)
        self.total = 0               # samples received since start (absolute index)
        self.gaps = []               # absolute sample indices where audio was lost
        self.lost_packets = 0
        self.bad_crc = 0
        self.flags = None
        self.last_seq = None
        self.last_ovf = None
        self.lock = threading.Lock()
        self.alive = True
        self.err = None
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        buf = bytearray()
        try:
            while self.alive:
                chunk = self.ser.read(4096)
                if not chunk:
                    continue
                buf += chunk
                while True:
                    i = buf.find(MAGIC)
                    if i < 0:
                        del buf[:-3]
                        break
                    if i:
                        del buf[:i]
                    if len(buf) < PKT:
                        break
                    pkt = bytes(buf[:PKT])
                    body, (crc,) = pkt[:-2], struct.unpack("<H", pkt[-2:])
                    if crc16_fast(body) != crc:
                        # Not a real packet boundary (or corrupted): resync one byte on.
                        self.bad_crc += 1
                        del buf[:1]
                        continue
                    del buf[:PKT]
                    seq, ovf, n, flags = struct.unpack("<IIHH", pkt[4:16])
                    pcm = np.frombuffer(pkt[16:16 + 2 * n], dtype="<i2")
                    self._push(seq, ovf, flags, pcm)
        except Exception as e:           # surfaced to the main thread
            self.err = e

    def _push(self, seq, ovf, flags, pcm):
        with self.lock:
            self.flags = flags
            gap = False
            if self.last_seq is not None:
                if seq != (self.last_seq + 1) & 0xFFFFFFFF:
                    gap = True
                    missing = (seq - self.last_seq - 1) & 0xFFFFFFFF
                    self.lost_packets += missing if missing < 10**6 else 1   # huge = board reset
                if ovf != self.last_ovf:
                    gap = True
            if gap:
                self.gaps.append(self.total)
            self.last_seq, self.last_ovf = seq, ovf
            n = len(pcm)
            idx = (self.total + np.arange(n)) % len(self.ring)
            self.ring[idx] = pcm
            self.total += n

    def now(self) -> int:
        with self.lock:
            return self.total

    def get(self, start: int, end: int):
        """Samples [start, end) by absolute index, and whether a gap lies inside."""
        while self.now() < end:
            if self.err:
                raise self.err
            time.sleep(0.02)
        with self.lock:
            if start < self.total - len(self.ring):
                raise RuntimeError("requested audio already left the ring buffer")
            idx = np.arange(start, end) % len(self.ring)
            x = self.ring[idx].copy()
            gap = any(start < g <= end for g in self.gaps)
        return x, gap

    def wait_ready(self, timeout=5.0):
        t0 = time.time()
        while self.now() < SR // 2:
            if self.err:
                raise self.err
            if time.time() - t0 > timeout:
                raise RuntimeError(
                    "No audio packets from the board. Check: recorder firmware flashed "
                    "(firmware/), right COM port, and the port not open in another program "
                    f"(bad CRC count {self.bad_crc}).")
            time.sleep(0.05)

    def close(self):
        self.alive = False
        self.t.join(timeout=1)
        self.ser.close()


# ---------------------------------------------------------------------------
# Audio measurements. Same framing as features.py (400 / 160) and the same
# -20 dB-below-peak endpoint rule that CLAUDE.md §9.7 validated on this data.
# ---------------------------------------------------------------------------
def dbfs(v):
    return 20 * np.log10(np.maximum(v, 1e-9) / 32768.0)


def frame_db(x):
    x = x.astype(np.float64)
    n = 1 + max(0, (len(x) - 400) // 160)
    fr = np.lib.stride_tricks.sliding_window_view(x, 400)[::160][:n]
    return dbfs(np.sqrt(np.mean(fr ** 2, axis=1)))


def zero_runs_ms(x, min_ms=10):
    """Longest run of exact-zero samples, in ms (real audio never has these)."""
    z = np.concatenate([[0], (x == 0).astype(np.int8), [0]])
    d = np.diff(z)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    longest = (ends - starts).max() if len(starts) else 0
    return 1000.0 * longest / SR


def analyse(x, mode):
    """QC one take and choose the 1 s cut. Returns a dict; 'fail' = reason or ''."""
    db = frame_db(x)
    peak_db = float(db.max())
    floor_db = float(np.percentile(db, 10))
    above = np.where(db >= peak_db - 20)[0]
    on = int(above[0] * 160)
    off = int(above[-1] * 160 + 400)
    q = {
        "peak_dbfs": round(float(dbfs(np.abs(x.astype(np.int32)).max())), 1),
        "speech_rms_dbfs": round(float(dbfs(np.sqrt(np.mean(x[on:off].astype(np.float64) ** 2)))), 1),
        "floor_dbfs": round(floor_db, 1),
        "word_s": round((off - on) / SR, 3),
        "clipped": int(np.sum(np.abs(x.astype(np.int32)) >= 32700)),
        "zero_run_ms": round(zero_runs_ms(x), 1),
        "fail": "", "flag": "",
    }
    edge = int(0.05 * SR)
    if peak_db - floor_db < 15:
        q["fail"] = "no clear speech (less than 15 dB above the room) - speak up or move closer"
    elif q["clipped"] >= 5:
        q["fail"] = f"clipped ({q['clipped']} samples at full scale) - speak softer or move back"
    elif q["zero_run_ms"] >= 10:
        q["fail"] = f"{q['zero_run_ms']} ms of exact silence - audio was dropped"
    elif on < edge:
        q["fail"] = "spoke too early (word touches the start) - wait for 'SPEAK'"
    elif off > len(x) - edge and mode == "word":
        q["fail"] = "word still going at the end of the window - start a little sooner"

    # Where to cut the 1.0 s clip.
    if mode == "phrase":
        # The keyword is the START of the phrase: anchor the clip on the speech
        # onset instead of centring on the whole sentence. The next word then
        # sits at the end of the clip, exactly as it will on the device.
        start = on - int(0.2 * SR)
    else:
        start = (on + off) // 2 - SR // 2
    start = int(min(max(start, 0), len(x) - SR))
    q["cut_start_s"] = round(start / SR, 3)
    pre = (on - start) / SR
    post = (start + SR - off) / SR
    if mode == "word" and (pre < 0.15 or post < 0.15) and not q["fail"]:
        q["flag"] = f"long word ({q['word_s']} s): less than 150 ms margin - kept, not cut"
    return q, start


def write_wav(path: Path, x: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(x.astype("<i2").tobytes())


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def cmd_level(st: Stream, args):
    print("Live level (0.5 s blocks). Training clips average about -19 dBFS RMS on speech.")
    print("Ctrl-C to stop.\n")
    t = st.now()
    try:
        while True:
            x, gap = st.get(t, t + SR // 2)
            t += SR // 2
            rms = float(dbfs(np.sqrt(np.mean(x.astype(np.float64) ** 2))))
            pk = float(dbfs(np.abs(x.astype(np.int32)).max()))
            dc = float(np.mean(x))
            bar = "#" * int(max(0, min(60, (rms + 70))))
            print(f"rms {rms:6.1f}  peak {pk:6.1f} dBFS  dc {dc:7.1f}  {'GAP ' if gap else ''}|{bar}")
    except KeyboardInterrupt:
        pass


def load_takes_csv():
    p = REC / "takes.csv"
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


TAKE_FIELDS = ["time", "speaker", "gender", "device", "room", "class", "condition", "word",
               "take", "file", "peak_dbfs", "speech_rms_dbfs", "floor_dbfs", "word_s",
               "clipped", "zero_run_ms", "cut_start_s", "flag", "sample_shift", "lost_packets_session"]


def append_csv(path: Path, fields, row):
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def cmd_clips(st: Stream, args):
    spk = args.speaker
    append_csv(REC / "speakers.csv",
               ["date", "speaker", "gender", "device", "room", "notes"],
               {"date": dt.date.today().isoformat(), "speaker": spk, "gender": args.gender,
                "device": args.device, "room": args.room, "notes": args.notes})

    # Build the to-do list, skipping takes that already exist (resume support).
    todo = []
    if args.only in ("all", "pos"):
        for cond, instr, n, mode in POSITIVE_SCRIPT:
            for k in range(1, n + 1):
                todo.append(("pos", cond, "vektora", k, instr, mode))
    if args.only in ("all", "neg"):
        for word, n in HARD_NEGATIVES:
            for k in range(1, n + 1):
                todo.append(("neg", word, word, k, "1 m from the mic, normal voice", "word"))
    clipdir = REC / spk / "clips"
    todo = [t for t in todo
            if not (clipdir / f"{t[0]}_{spk}_{t[1]}_{t[3]:03d}.wav").exists()]
    if not todo:
        print(f"Speaker {spk}: everything is already recorded.")
        return

    print(f"\nSpeaker {spk}: {len(todo)} takes to record.")
    print("For each take: press ENTER, wait for  >>> SPEAK <<<  then say the word ONCE.")
    print("Type  s + ENTER to skip a take,  q + ENTER to stop (you can resume later).\n")

    shift = (st.flags or 0) & 0xFF
    last_instr = None
    i = 0
    while i < len(todo):
        cls, cond, word, k, instr, mode = todo[i]
        if instr != last_instr:
            print("\n" + "=" * 70)
            print(f"  NOW: {instr}")
            print("=" * 70)
            last_instr = instr
        say = "Vektora" if cls == "pos" else word.upper()
        if mode == "phrase":
            say = "Vektora, <a command>"
        ans = input(f"[{i + 1}/{len(todo)}] say  {say:<22} (take {k})  ENTER> ").strip().lower()
        if ans == "q":
            break
        if ans == "s":
            i += 1
            continue

        # Window: 0.3 s BEFORE the Enter key (so a fast speaker is never cut)
        # to 2.2 s after it; 3.2 s after it for phrases.
        t_enter = st.now()
        dur = 3.5 if mode == "phrase" else 2.5
        start = t_enter - int(0.3 * SR)
        print("        >>> SPEAK <<<", flush=True)
        x, gap = st.get(start, start + int(dur * SR))

        if gap:
            print("   REDO: audio was lost during this take (USB or board overflow). Say it again.")
            continue
        q, cut = analyse(x, mode)
        if q["fail"]:
            print(f"   REDO: {q['fail']}")
            continue

        name = f"{cls}_{spk}_{cond}_{k:03d}.wav"
        write_wav(REC / spk / "raw" / name, x)
        write_wav(clipdir / name, x[cut:cut + SR])
        append_csv(REC / "takes.csv", TAKE_FIELDS, {
            "time": dt.datetime.now().isoformat(timespec="seconds"), "speaker": spk,
            "gender": args.gender, "device": args.device, "room": args.room,
            "class": cls, "condition": cond, "word": word, "take": k, "file": name,
            **{f: q[f] for f in ["peak_dbfs", "speech_rms_dbfs", "floor_dbfs", "word_s",
                                  "clipped", "zero_run_ms", "cut_start_s", "flag"]},
            "sample_shift": shift, "lost_packets_session": st.lost_packets})
        note = f"   ! {q['flag']}" if q["flag"] else ""
        print(f"   ok  word {q['word_s']:.2f} s, speech {q['speech_rms_dbfs']} dBFS, "
              f"peak {q['peak_dbfs']}, room {q['floor_dbfs']}{note}")
        i += 1

    done = [t for t in load_takes_csv() if t["speaker"] == spk]
    print(f"\nSaved so far for {spk}: {sum(t['class'] == 'pos' for t in done)} positives, "
          f"{sum(t['class'] == 'neg' for t in done)} hard negatives -> {clipdir}")


def cmd_long(st: Stream, args):
    outdir = DATA / "long" / args.kind
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    limit = int(args.minutes * 60 * SR) if args.minutes else None
    part, written, t = 1, 0, st.now()
    blk = SR // 2

    def open_part(p):
        path = outdir / f"{args.name}_{stamp}_part{p}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        w = wave.open(str(path), "wb")
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        return w, path

    w, path = open_part(part)
    print(f"Recording to {path}  (Ctrl-C to stop)")
    try:
        while limit is None or written < limit:
            x, gap = st.get(t, t + blk)
            t += blk
            if gap:
                # Never join audio across a gap: close this file, start a new part.
                w.close()
                part += 1
                w, path = open_part(part)
                print(f"\n  audio gap detected -> continuing in a new file {path.name}")
            w.writeframes(x.astype("<i2").tobytes())
            written += blk
            rms = float(dbfs(np.sqrt(np.mean(x.astype(np.float64) ** 2))))
            print(f"\r  {written / SR / 60:5.1f} min   level {rms:6.1f} dBFS   ", end="", flush=True)
    except KeyboardInterrupt:
        pass
    w.close()
    print(f"\nDone: {written / SR / 60:.1f} min in {part} file(s) under {outdir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM10")
    ap.add_argument("--baud", type=int, default=921600)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("level")
    c = sub.add_parser("clips")
    c.add_argument("--speaker", required=True, help="short ID, e.g. S01. Same person = same ID, always")
    c.add_argument("--gender", required=True, choices=["F", "M", "other"])
    c.add_argument("--device", default="inmp441-esp32")
    c.add_argument("--room", default="lab")
    c.add_argument("--notes", default="")
    c.add_argument("--only", default="all", choices=["all", "pos", "neg"])
    lg = sub.add_parser("long")
    lg.add_argument("--name", required=True, help="e.g. fan_speed3, classroom, corridor")
    lg.add_argument("--kind", default="noise", choices=["noise", "other", "session"])
    lg.add_argument("--minutes", type=float, default=0, help="0 = until Ctrl-C")
    args = ap.parse_args()

    st = Stream(args.port, args.baud)
    try:
        st.wait_ready()
        shift = (st.flags or 0) & 0xFF
        print(f"Board streaming on {args.port}: 16 kHz mono, sample shift {shift}.")
        {"level": cmd_level, "clips": cmd_clips, "long": cmd_long}[args.cmd](st, args)
    finally:
        if st.lost_packets or st.bad_crc:
            print(f"(link stats: {st.lost_packets} lost packets, {st.bad_crc} resync bytes)")
        st.close()


if __name__ == "__main__":
    main()

"""protocol.py — the device<->server UDP wire format (CLAUDE.md §8, docs/protocol.md).

Firmware and server must agree on this byte-for-byte. Little-endian, packed.
Header (22 B): magic u32 | session u16 | type u8 | codec u8 | seq u32 | t_capture_us u64 | len u16
"""

import struct

MAGIC = int.from_bytes(b"VAD1", "little")
HDR = struct.Struct("<IHBBIQH")
HDR_LEN = HDR.size  # 22

UDP_PORT = 5005

# device -> server
HELLO, AUDIO, END, PING = 1, 2, 3, 4
# server -> device
PONG, TRANSCRIPT = 5, 6

CODEC_PCM16, CODEC_ULAW = 0, 1

SR = 16000
FRAME_MS = 20
FRAME_SAMPLES = SR * FRAME_MS // 1000  # 320

# PING payload: t_dev_us u64, then optionally the result of the PREVIOUS exchange
# (prev_t_dev_us u64, prev_rtt_us u32) so the server can compute the offset.
# Only the device knows RTT (it sees both ends), so it reports it back.
PING_BASE = struct.Struct("<Q")
PING_FULL = struct.Struct("<QQI")
PONG_PAYLOAD = struct.Struct("<QQ")  # t_dev_us echoed, t_srv_us


def pack(session: int, ptype: int, seq: int, t_us: int, payload: bytes = b"", codec: int = CODEC_PCM16) -> bytes:
    return HDR.pack(MAGIC, session & 0xFFFF, ptype, codec, seq & 0xFFFFFFFF, t_us, len(payload)) + payload


def unpack(data: bytes):
    """Returns (session, type, codec, seq, t_capture_us, payload) or None if malformed."""
    if len(data) < HDR_LEN:
        return None
    magic, session, ptype, codec, seq, t_us, n = HDR.unpack_from(data)
    if magic != MAGIC or len(data) < HDR_LEN + n:
        return None
    return session, ptype, codec, seq, t_us, data[HDR_LEN:HDR_LEN + n]


# G.711 mu-law -> int16, table-driven (256 entries, built once).
def _ulaw_table():
    import numpy as np
    t = np.zeros(256, dtype=np.int16)
    for i in range(256):
        u = ~i & 0xFF
        sign, exp, mant = u & 0x80, (u >> 4) & 0x07, u & 0x0F
        mag = ((mant << 3) + 0x84) << exp
        t[i] = -(mag - 0x84) if sign else (mag - 0x84)
    return t


_ULAW = None


def decode(codec: int, payload: bytes) -> bytes:
    """Always returns PCM16 little-endian bytes."""
    if codec == CODEC_PCM16:
        return payload
    if codec == CODEC_ULAW:
        global _ULAW
        if _ULAW is None:
            _ULAW = _ulaw_table()
        import numpy as np
        return _ULAW[np.frombuffer(payload, dtype=np.uint8)].tobytes()
    raise ValueError(f"unknown codec {codec}")

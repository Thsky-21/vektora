"""wake_word.py — show the wake word as "Vektora" in the transcript.

Vosk's small English model has no "Vektora" in its vocabulary, so it hears
"vector", "regular", "victor a"... The DEVICE already confirmed the keyword
(that's why this audio exists), so the display names it correctly. The raw
Vosk text is always kept alongside as text_raw (CLAUDE.md Rule 0.1: evidence
stays honest; only the display is rewritten).

Two ways to find which recognised words ARE the keyword:
  * timed  — UDP path. The device sends t_win_end_us (end of the window that
             fired) and the capture time of AUDIO seq 0, both on its clock, so
             the keyword end in the stream is known exactly. Words that start
             inside the model's 1 s window before that point are the keyword.
  * fuzzy  — HTTP path (no timestamps). The first word, or two merged words,
             in the first KW_SEARCH_S seconds that sounds like the keyword
             (a known Vosk confusion, or difflib ratio >= KW_MIN_RATIO).
If neither finds anything, "Vektora" is PREPENDED: the device heard it even
if Vosk dropped it.
"""

import difflib
import os

KEYWORD = os.environ.get("KEYWORD", "Vektora")
KW_WINDOW_S = 1.0     # the KWS model's input window (98 frames = 1 s)
KW_SEARCH_S = 2.0     # fuzzy: keyword must start this early (pre-roll is <= 1 s)
KW_MIN_RATIO = 0.6
KW_MERGE_RATIO = 0.75  # stricter for two-word merges ("the doctor" must not match)
KW_MERGE_GAP_S = 0.15  # the two halves must be adjacent in time
# What vosk-model-small-en-in actually returned for "Vektora" in our tests,
# plus the hard negatives the KWS was trained against.
CONFUSIONS = {"vector", "victor", "vectra", "regular", "rector", "sector",
              "doctor", "factor", "victoria", "vectors", "director"}

_kw = KEYWORD.lower()
_TARGETS = {_kw, _kw.replace("k", "c")}  # English spelling of the same sound


def _ratio(s: str) -> float:
    return max(difflib.SequenceMatcher(None, s, t).ratio() for t in _TARGETS)


def _fuzzy(words: list[dict]):
    """-> (index, n_words) of the keyword, or None."""
    for i, w in enumerate(words):
        if w.get("start", 0.0) > KW_SEARCH_S:
            break
        a = w["word"].lower()
        r1 = _ratio(a)
        ok1 = a in CONFUSIONS or a in _TARGETS or (len(a) >= 3 and r1 >= KW_MIN_RATIO)
        if i + 1 < len(words):  # "vector a", "vic tora": two words for one
            b = words[i + 1]
            if b.get("start", 0.0) - w.get("end", 0.0) <= KW_MERGE_GAP_S:
                r2 = _ratio(a + b["word"].lower())
                if r2 >= KW_MERGE_RATIO and r2 > r1:
                    return i, 2
        if ok1:
            return i, 1
    return None


def rewrite(words: list[dict], kw_end_s: float | None = None) -> tuple[str, dict]:
    """words: Vosk word dicts {word,start,end} in stream time (s).
    kw_end_s: keyword end in the same time base, or None if unknown.
    -> (display text, info) where info says what was replaced and how."""
    if not words:
        return "", {"mode": "none"}
    out = [w["word"] for w in words]
    if kw_end_s is not None:
        idx = [i for i, w in enumerate(words)
               if w["start"] < kw_end_s and w["end"] > kw_end_s - KW_WINDOW_S]
        if idx:
            i, j = idx[0], idx[-1] + 1
            info = {"mode": "timed", "replaced": out[i:j], "kw_end_s": round(kw_end_s, 3)}
            return " ".join(out[:i] + [KEYWORD] + out[j:]), info
        # nothing recognised inside the window: insert at the keyword's position
        i = sum(1 for w in words if w["end"] <= kw_end_s - KW_WINDOW_S)
        info = {"mode": "timed_insert", "replaced": [], "kw_end_s": round(kw_end_s, 3)}
        return " ".join(out[:i] + [KEYWORD] + out[i:]), info
    hit = _fuzzy(words)
    if hit:
        i, n = hit
        return " ".join(out[:i] + [KEYWORD] + out[i + n:]), {"mode": "fuzzy", "replaced": out[i:i + n]}
    return " ".join([KEYWORD] + out), {"mode": "prepend", "replaced": []}

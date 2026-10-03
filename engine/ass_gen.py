"""ASS subtitle generation + per-segment re-stamping.

libass has no time-offset option, so each segment gets its own .ass file
with events re-stamped relative to the segment start. Segment boundaries
always fall on caption boundaries (constructed that way), so the clamp is
just a belt-and-braces assertion.
"""
from __future__ import annotations

from pathlib import Path

from engine.util import atomic_write_text, ms_to_ass_time

HEADER = """[Script Info]
Title: {title}
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Narr,{font},{size},{primary},{secondary},{outline},{back},0,0,0,0,100,100,{spacing},0,{borderstyle},{ow},{sh},{align},{ml},{mr},{mv},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

# left-column geometry: figure owns the right ~44% (right-anchored 800px
# overlay + safety), captions center in the remaining left column.
# borderstyle=1 (user call 2026-09-06): NO background card.  A thick pure-
# black outline + semi-opaque drop shadow carry legibility over bright
# landscape footage; every phrase punch-in is animated per event (POP).
DEFAULT_STYLE = dict(
    font="Montserrat ExtraBold", size=76,
    primary="&H00FFFFFF",    # pure white glyphs
    secondary="&H000000FF",
    outline="&H00000000",    # pure black outline, Outline px carries it now
    back="&H60000000",       # BorderStyle 1 -> BackColour = drop-shadow ink
    borderstyle=1, ow=5.0, sh=3.5,   # outline width / shadow offset px
    spacing=1.0,
    align=2, ml=48, mr=840, mv=150,
)

# Phrase punch-in: start at 72% (small), overshoot to 112% by 90 ms, settle
# to 100% by 180 ms; fade out over the last 110 ms.  \fscx/\fscy are visual
# transforms (layout untouched), so the card pops in place; the outline
# scales with it (ScaledBorderAndShadow).  Relative to each event's (already
# re-stamped) start, so it survives segmentation unchanged.
POP = (r"{\fscx72\fscy72\t(0,90,\fscx112\fscy112)"
       r"\t(90,180,\fscx100\fscy100)\fad(0,110)}")


def _esc(text: str) -> str:
    """Make plain text safe for ASS Dialogue field."""
    return text.replace("{", "(").replace("}", ")").replace("\n", "\\N")


def render_ass(events: list[dict], *, title: str = "engine",
               seg_t0_ms: int = 0, seg_t1_ms: int | None = None,
               **style) -> str:
    st = dict(DEFAULT_STYLE)
    st.update(k for k, v in style.items() if v is not None)
    end_all = seg_t1_ms if seg_t1_ms is not None else 24 * 3600 * 1000
    lines = [HEADER.format(title=title.replace("\n", " "), **st)]
    n = 0
    for e in events:
        if e["t1_ms"] <= seg_t0_ms or e["t0_ms"] >= end_all:
            continue
        s = max(e["t0_ms"], seg_t0_ms) - seg_t0_ms
        e_ms = min(e["t1_ms"], end_all) - seg_t0_ms
        assert e_ms > s, f"inverted event {e}"
        lines.append(
            f"Dialogue: 0,{ms_to_ass_time(s)},"
            f"{ms_to_ass_time(e_ms)},Narr,,0,0,0,,{POP}{_esc(e['text'])}")
        n += 1
    return "\n".join(lines) + "\n"


def write_ass_file(path: str | Path, events: list[dict], **kw) -> int:
    text = render_ass(events, **kw)
    atomic_write_text(path, text)
    return text.count("Dialogue:")

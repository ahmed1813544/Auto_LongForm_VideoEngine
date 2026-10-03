"""Pass 2 planning: audio-aligned segments + seeded background scenes.

Segment boundaries fall on caption-event boundaries (so no caption ever
crosses a cut — the ASS re-stamper asserts this), preferring story ends.
Scenes are picked deterministically from the normalized clip pool with a
seeded RNG, so a resumed job plans the identical picture it had before.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

SCENE_MIN_S, SCENE_MAX_S = 8.0, 16.0
RECENT_WINDOW = 6          # don't repeat a clip within last ~3 segments
POOL_HEADROOM = 1.2        # pool seconds must cover 1.2x master duration
POOL_MIN = RECENT_WINDOW + 2


def pool_from_clips(clips: list[dict], norm_paths: dict[str, str | None],
                    duration_key: str = "norm_duration_s") -> list[dict]:
    """Build planner pool dicts from library entries + normalize results.

    clips: [{key, duration_s, ...}] (Clip.to_dict() works).
    norm_paths: {clip_key: norm_path or None} from normalize_all.
    """
    pool = []
    for c in clips:
        path = norm_paths.get(c["key"])
        if not path:
            continue
        pool.append({
            "key": c["key"],
            "path": str(path),
            "duration_s": float(c.get(duration_key)
                                or c.get("duration_s") or 0),
        })
    return pool


@dataclass
class Segment:
    idx: int
    t0_ms: float
    t1_ms: float
    story_first: int
    scenes: list[dict] = field(default_factory=list)  # {key,path,in_s,dur_s}

    @property
    def dur_s(self) -> float:
        return (self.t1_ms - self.t0_ms) / 1000.0

    def to_dict(self) -> dict:
        return {"idx": self.idx, "t0_ms": self.t0_ms, "t1_ms": self.t1_ms,
                "story_first": self.story_first, "dur_s": self.dur_s,
                "scenes": self.scenes}

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(idx=d["idx"], t0_ms=d["t0_ms"], t1_ms=d["t1_ms"],
                   story_first=d["story_first"], scenes=d["scenes"])


def plan_segments(events: list[dict], total_ms: float,
                  seg_max_s: float) -> list[Segment]:
    """Cut the timeline into <= seg_max_s segments ending on event t1s.

    Preference order for each cut: latest story-end within the window,
    else latest event end that still fits.  A runaway single event longer
    than the window is accepted alone (can't split a caption).
    """
    if not events:
        raise ValueError("no caption events to align segments to")
    if seg_max_s <= 0:
        raise ValueError("seg_max_s must be positive")
    seg_max_ms = seg_max_s * 1000.0
    out: list[Segment] = []
    t0 = 0.0
    i = 0
    n = len(events)
    while t0 < total_ms - 1.0:
        last_idx = None
        story_idx = None
        for k in range(i, n):
            e = events[k]
            if e["t1_ms"] - t0 > seg_max_ms:
                break
            last_idx = k
            if e.get("story_end"):
                story_idx = k
        cut_idx = story_idx if story_idx is not None else last_idx
        if cut_idx is None:
            if i >= n:
                # audio runs past the last caption (trailing silence):
                # final segment must cover the whole master.wav
                out.append(Segment(idx=len(out), t0_ms=t0, t1_ms=total_ms,
                                   story_first=events[-1]["story"]))
                break
            cut_idx = i                      # one overlong event, alone
        seg = Segment(idx=len(out), t0_ms=t0,
                      t1_ms=events[cut_idx]["t1_ms"],
                      story_first=events[i]["story"])
        out.append(seg)
        i = cut_idx + 1
        t0 = seg.t1_ms
    return out


def pool_coverage(pool: list[dict], total_s: float) -> tuple[float, bool]:
    """(covered_seconds, enough) for the 1.2x headroom rule."""
    cover = sum(float(p.get("duration_s") or 0) for p in pool)
    return cover, cover >= total_s * POOL_HEADROOM


def plan_scenes(segments: list[Segment], pool: list[dict], *,
                seed: int = 1234) -> list[Segment]:
    """Fill each segment's scenes from pool [{key,path,duration_s}].

    Deterministic for a given (pool order, seed).  Mutates and returns the
    same segment objects.
    """
    usable = [p for p in pool if p.get("path")]
    if not usable:
        raise ValueError("empty clip pool — nothing to plan scenes from")
    if len(usable) < POOL_MIN:
        # tiny pools can deadlock the no-repeat window; drop the window
        window: deque = deque(maxlen=0)
    else:
        window = deque(maxlen=RECENT_WINDOW)
    rng = random.Random(seed)
    order: list[dict] = []
    cursor = 0

    def pick() -> dict:
        nonlocal order, cursor
        while True:
            if cursor >= len(order):
                order = list(usable)
                rng.shuffle(order)
                cursor = 0
            c = order[cursor]
            cursor += 1
            # empty window (tiny pool) never contains the key -> always picks
            if c["key"] not in window:
                window.append(c["key"])
                return c

    for seg in segments:
        remaining = seg.dur_s
        while remaining > 1e-6:
            take = min(rng.uniform(SCENE_MIN_S, SCENE_MAX_S), remaining)
            if take < 0.05:
                # sub-frame stub: leave it for the drift-fix tail below
                # (previously the rounded 0.000 scene deadlocked the loop:
                # MemoryError on a 51 s build, 2026-09-06)
                break
            if take < SCENE_MIN_S * 0.5:
                take = remaining          # fold the tail stub into last scene
            c = pick()
            cdur = float(c.get("duration_s") or 0)
            in_s = 0.0
            dur = take
            if cdur > 0:
                dur = min(take, cdur)     # clip shorter than the scene
                if cdur > dur + 1.0:      # start at a random offset for variety
                    in_s = rng.uniform(0.0, cdur - dur)
            if dur < 0.05:                # pathologically short clip entry
                dur = take
            seg.scenes.append({"key": c["key"], "path": c["path"],
                               "in_s": round(in_s, 3),
                               "dur_s": round(dur, 3)})
            # subtract the EXACT dur, not the rounded stored value, so
            # per-scene rounding drift can never stall the loop
            remaining -= dur
        # make the last scene absorb rounding drift so scenes sum exactly
        acc = sum(s["dur_s"] for s in seg.scenes)
        seg.scenes[-1]["dur_s"] = round(
            seg.scenes[-1]["dur_s"] + (seg.dur_s - acc), 3)
    return segments


def assign_personas(segments: list[Segment], persona_files: list[str]
                    ) -> list[str | None]:
    """One pose per segment, cycled by the segment's first story index."""
    if not persona_files:
        return [None] * len(segments)
    return [persona_files[s.story_first % len(persona_files)]
            for s in segments]

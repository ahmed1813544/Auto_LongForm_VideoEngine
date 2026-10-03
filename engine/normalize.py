"""One-time normalization of library clips to the golden video contract.

Normalized clips carry every encode param of a rendered segment (fps, size,
SAR, pix_fmt, profile/level, timescale), which is what lets the segment
builder slice them with -ss/-t and the final -c copy concat stay seamless.
ffprobe post-normalize fills real durations into the index (Pexels hits
sometimes report duration 0).

Some stock "cinematic" clips are 16:9 files with black letterbox bars baked
into the picture (e.g. 1920x1080 holding 1920x662 of content). The cover-
crop chain alone passes those straight through — scale=increase+crop is a
no-op on an already-16:9 file. So we cropdetect-sample the raw clip and,
when bars are found, strip them via a pre-crop before the cover chain.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from engine.downloader import load_index, norm_path, raw_path, save_index
from engine.render.ffmpeg_cmds import (HEIGHT, WIDTH, GOLDEN_VIDEO,
                                       normalize_cmd)
from engine.stock.base import Clip
from engine.util import ffprobe_duration, ffprobe_video_info, run

_VIDEO_KEYS = ("codec_name", "profile", "width", "height", "r_frame_rate",
               "avg_frame_rate", "pix_fmt", "sample_aspect_ratio", "level")

# cropdetect over-crops anything darker than its limit; bars we accept must
# be real geometry, not a dark frame moment — ignore slivers under ~1%.
_BAR_MIN_W = int(WIDTH * 0.99)    # 1900
_BAR_MIN_H = int(HEIGHT * 0.99)   # 1069


def _probe_crop(path: str | Path, t: float, d: float) -> tuple | None:
    """cropdetect content box at [t, t+d); returns (w,h,x,y) or None."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-ss", f"{t:.2f}", "-t", f"{d:.2f}",
         "-i", str(path), "-vf", "cropdetect=24:2:0", "-an",
         "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    found = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", r.stderr)
    return tuple(int(v) for v in found[-1]) if found else None


def _is_full(w: int, h: int) -> bool:
    return w >= _BAR_MIN_W and h >= _BAR_MIN_H


def _first_clean_start(path: str | Path, dur: float) -> float | None:
    """Earliest time from which the rest of the clip is full-frame.

    cropdetect reports the *minimum* content box over its window, so one
    probe over [t, end] answers "is everything from t on clean". Binary
    search over that monotone predicate.
    """
    if dur - 1.5 < 0:
        return None
    tail = _probe_crop(path, dur - 1.5, 1.5)
    if tail is None or not _is_full(tail[0], tail[1]):
        return None
    lo, hi = 0.0, dur - 1.2          # clean(hi) holds; shrink toward lo
    probes = 0
    while hi - lo > 0.5 and probes < 10:
        mid = (lo + hi) / 2.0
        c = _probe_crop(path, mid, dur - mid)
        probes += 1
        if c is not None and _is_full(c[0], c[1]):
            hi = mid
        else:
            lo = mid
    return hi


def bars_plan(path: str | Path) -> dict:
    """Decide how to make a clip fill the frame, if it doesn't already.

    Samples cropdetect at five points across the clip:
    - all full-frame                     -> {"kind": "none"}
    - all agree on a smaller box         -> {"kind": "static", "crop": box}
      (baked-in letter/pillarbox; strip via pre-crop)
    - starts boxed, ends (and after a
      while stays) full-frame            -> {"kind": "dynamic", "trim_s": t}
      (animated cinematic bars; trim the head instead of cropping, so the
      clean part keeps native resolution)
    - anything else (montage, dark-edge
      content)                           -> {"kind": "none"}  (don't trust)
    """
    plan = {"kind": "none", "crop": None, "trim_s": 0.0}
    try:
        dur = ffprobe_duration(path)
    except RuntimeError:
        return plan
    if dur < 3.0:
        return plan
    times = [min(dur - 1.0, dur * f) for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
    samples = [_probe_crop(path, t, 0.8) for t in times]
    if any(s is None for s in samples):
        return plan
    full = [_is_full(w, h) for w, h, _, _ in samples]
    if all(full):
        return plan
    if not any(full):
        if len(set(samples)) == 1:
            return {"kind": "static", "crop": samples[0], "trim_s": 0.0}
        return plan                       # varying bars: don't trust
    if full[0] or not all(full[-2:]):
        return plan                       # messy content: leave alone
    trim = _first_clean_start(path, dur)
    if trim is None or trim > dur * 0.75:
        return plan                       # no meaningful clean tail
    return {"kind": "dynamic", "crop": None, "trim_s": trim}


def detect_bars(path: str | Path) -> tuple[int, int, int, int] | None:
    """Static baked-in bars only (see bars_plan for the full decision)."""
    p = bars_plan(path)
    return p["crop"] if p["kind"] == "static" else None


def video_problems(path: str | Path) -> list[str]:
    """Violations of the golden *video* contract (audio stream ignored)."""
    info = ffprobe_video_info(path)
    return [f"{k}={info.get(k)!r} want {GOLDEN_VIDEO[k]!r}"
            for k in _VIDEO_KEYS if info.get(k) != GOLDEN_VIDEO[k]]


def normalize_one(cfg, clip: Clip, *, nvenc: bool = False,
                  log=None) -> Path:
    """raw -> norm; idempotent (existing output re-probed before trusting).

    Detects baked-in black bars and fixes them (crop or head-trim —
    see bars_plan). Post-check: the normalized file must be full-frame,
    or the encode is treated as a failure.
    """
    src = raw_path(cfg, clip)
    if not src.exists():
        raise RuntimeError(f"raw clip missing: {src}")
    dst = norm_path(cfg, clip)
    if dst.exists() and dst.stat().st_size > 0 \
            and not video_problems(dst):
        return dst
    plan = bars_plan(src)
    if plan["kind"] != "none" and log:
        log.info(f"bars fix {clip.key}: {plan['kind']} "
                 f"{plan.get('crop') or ('trim %.1fs' % plan['trim_s'])}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(normalize_cmd(src, dst, nvenc=nvenc, pre_crop=plan["crop"],
                      start_s=plan["trim_s"]), log=log)
    probs = video_problems(dst)
    if probs:
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"normalize contract violated for {clip.key}: "
                           + "; ".join(probs))
    if bars_plan(dst)["kind"] != "none":
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"bars survive normalize for {clip.key}")
    return dst


def normalize_all(cfg, clips, *, nvenc: bool = False, log=None) -> dict:
    """Normalize every pending clip; returns {clip.key: norm_path | None}."""
    index = load_index(cfg)
    out: dict[str, str | None] = {}
    done = skip = 0
    since_save = 0
    for clip in clips:
        key = clip.key
        entry = index.get(key) or {}
        cached = entry.get("norm")
        if cached and Path(cached).exists() and not video_problems(cached):
            out[key] = cached
            skip += 1
            continue
        try:
            p = normalize_one(cfg, clip, nvenc=nvenc, log=log)
        except Exception as e:  # noqa: BLE001 — pool survives individual loss
            if log:
                log.warn(f"normalize failed {key}: {e}")
            out[key] = None
            continue
        try:
            dur = ffprobe_duration(p)
        except RuntimeError:
            dur = clip.duration_s
        entry = dict(entry)
        entry.update({"provider": clip.provider, "id": clip.id,
                      "term": entry.get("term") or clip.term,
                      "raw": str(raw_path(cfg, clip)),
                      "norm": str(p), "norm_duration_s": dur})
        # norm duration is the authority the planner slices against —
        # always overwrite (a head-trim for animated bars shrinks it).
        entry["duration_s"] = dur
        index[key] = entry
        out[key] = str(p)
        done += 1
        # Persist progress periodically: a watchdog kill (low-memory guard)
        # mid-wave otherwise discards every index update for the wave, so a
        # resume re-does the whole pass and coverage never advances.
        since_save += 1
        if since_save >= 10:
            save_index(cfg, index)
            since_save = 0
    save_index(cfg, index)
    if log:
        log.info(f"normalize: {done} new, {skip} cached, "
                 f"{len(clips) - done - skip} failed")
    return out

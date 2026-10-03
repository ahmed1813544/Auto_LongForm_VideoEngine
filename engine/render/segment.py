"""Execute a segment render + validate against the golden stream contract."""
from __future__ import annotations

from pathlib import Path

from engine.render.ffmpeg_cmds import GOLDEN_AUDIO, GOLDEN_VIDEO
from engine.util import ffprobe_json, ffprobe_duration


def check_streams(path: str | Path, *, ignore: tuple[str, ...] = ()) -> list[str]:
    """Return list of violations vs the golden dict (empty == OK).

    ignore: keys to skip (e.g. avg_frame_rate on a `-c copy` concatenation,
    where AAC padding shifts the averaged rate by ppm — r_frame_rate is the
    authoritative CFR check there).
    """
    info = ffprobe_json(path, show_streams=True)
    problems: list[str] = []
    v = a = None
    for s in info.get("streams", []):
        if s.get("codec_type") == "video" and v is None:
            v = s
        elif s.get("codec_type") == "audio" and a is None:
            a = s
    if v is None or a is None:
        return ["missing video or audio stream"]
    for k, want in GOLDEN_VIDEO.items():
        if k in ignore:
            continue
        got = v.get(k)
        if got != want:
            problems.append(f"v.{k}={got!r} want {want!r}")
    for k, want in GOLDEN_AUDIO.items():
        if k in ignore:
            continue
        got = a.get(k)
        if got != want:
            problems.append(f"a.{k}={got!r} want {want!r}")
    return problems


def render_segment(cmd: list[str], out: Path, planned_dur: float, log,
                   *, timeout: float | None = None) -> dict:
    """Run one segment; returns {ok, probed_dur, problems, stderr_tail}."""
    from engine.util import run
    proc = run(cmd, log=log, check=False, timeout=timeout)
    res = {"ok": False, "probed_dur": None, "problems": [], "stderr_tail": ""}
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").splitlines()[-40:])
        res["problems"] = [f"exit {proc.returncode}"]
        res["stderr_tail"] = tail
        log.warn(f"segment render failed: {out.name}")
        return res
    try:
        dur = ffprobe_duration(out)
    except Exception as e:  # probe failure => unusable file
        res["problems"] = [f"probe failed: {e}"]
        return res
    res["probed_dur"] = dur
    if abs(dur - planned_dur) > 0.05:
        res["problems"].append(
            f"duration {dur:.3f} != planned {planned_dur:.3f}")
    probs = check_streams(out)
    if probs:
        res["problems"].extend(probs)
    res["ok"] = not res["problems"]
    if not res["ok"]:
        log.warn(f"segment {out.name} problems: {res['problems']}")
    else:
        log.info(f"segment done: {out.name} ({dur:.1f}s)")
    return res

"""Post-build verification for a job: golden streams, decode-clean, math.

`verify --job-id X` checks every finished segment + the final file:
- duration vs plan within tolerance,
- golden stream contract (same params that make -c copy safe),
- -v error decode scan must be silent (deep: per segment too),
- caption count across .ass files == captions.json event count.

Returns a problems list; empty == healthy.
"""
from __future__ import annotations

import json
from pathlib import Path

from engine.render.segment import check_streams
from engine.util import ffprobe_duration, run


def decode_errors(path: str | Path, *, timeout: float = 7200) -> str:
    """Full decode with -v error: anything printed is a defect."""
    proc = run(["ffmpeg", "-hide_banner", "-v", "error", "-i", str(path),
                "-f", "null", "-"], check=False, timeout=timeout)
    return (proc.stderr or "").strip()


def check_final(final: str | Path, planned_total_s: float, *,
                deep: bool = False, log=None) -> list[str]:
    problems: list[str] = []
    final = Path(final)
    if not final.exists():
        return [f"missing: {final}"]
    dur = ffprobe_duration(final)
    if abs(dur - planned_total_s) > 0.25:
        problems.append(
            f"final duration {dur:.2f}s vs planned {planned_total_s:.2f}s")
    # avg_frame_rate drifts ppm-wise on -c copy (AAC padding); r_frame_rate
    # still enforces CFR.
    problems += [f"final: {p}" for p in
                 check_streams(final, ignore=("avg_frame_rate",))]
    errs = decode_errors(final)
    if errs:
        problems.append("final decode errors:\n" + errs[:2000])
    if log:
        log.info(f"final: {dur/3600:.2f} h, "
                 f"{'clean' if not problems else str(len(problems)) + ' problems'}")
    return problems


def verify_job(job_dir: str | Path, cfg, *, deep: bool = False,
               log=None) -> dict:
    """Full health report for a job. Returns {ok, problems, summary}."""
    job_dir = Path(job_dir)
    problems: list[str] = []
    state_p = job_dir / "state.json"
    caps_p = job_dir / "captions.json"
    if not state_p.exists() or not caps_p.exists():
        return {"ok": False, "problems": [f"{job_dir} is not a built job"],
                "summary": {}}
    state = json.loads(state_p.read_text(encoding="utf-8"))
    caps = json.loads(caps_p.read_text(encoding="utf-8"))
    total_s = float(caps["total_ms"]) / 1000.0
    events = caps["events"]

    segs = state.get("segments", [])
    n_done = 0
    for e in segs:
        if e.get("state") != "done":
            problems.append(f"segment {e['idx']}: state={e.get('state')}")
            continue
        p = e.get("path")
        if not p or not Path(p).exists():
            problems.append(f"segment {e['idx']}: file missing ({p})")
            continue
        n_done += 1
        try:
            dur = ffprobe_duration(p)
        except (RuntimeError, OSError) as ex:
            problems.append(f"segment {e['idx']}: probe failed: {ex}")
            continue
        if abs(dur - e["dur_s"]) > 0.1:
            problems.append(
                f"segment {e['idx']}: duration {dur:.2f} vs plan "
                f"{e['dur_s']:.2f}")
        probs = check_streams(p)
        if probs:
            problems.append(f"segment {e['idx']}: " + "; ".join(probs))
        if deep:
            errs = decode_errors(p)
            if errs:
                problems.append(f"segment {e['idx']} decode errors:\n"
                                + errs[:1000])

    # caption conservation: every event rendered exactly once across segs
    ass_total = 0
    for ass in sorted((job_dir / "ass").glob("*.ass")):
        ass_total += ass.read_text(encoding="utf-8").count("Dialogue:")
    if segs and ass_total != len(events):
        problems.append(
            f"captions: {ass_total} .ass Dialogue lines != {len(events)} "
            f"events (segments missing or stale?)")

    out_p = None
    concat = state.get("phases", {}).get("concat", {})
    if concat.get("done"):
        out_p = concat.get("out")
        if out_p:
            problems += check_final(out_p, total_s, deep=deep, log=log)

    summary = {"segments_done": n_done, "segments_planned": len(segs),
               "events": len(events), "ass_lines": ass_total,
               "final": out_p, "duration_h": round(total_s / 3600, 3)}
    if log:
        log.info(f"verify {job_dir.name}: {n_done}/{len(segs)} segments, "
                 f"{'OK' if not problems else str(len(problems)) + ' problem(s)'}")
    return {"ok": not problems, "problems": problems, "summary": summary}

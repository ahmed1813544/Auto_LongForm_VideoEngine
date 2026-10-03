"""Final lossless concat of segments."""
from __future__ import annotations

from pathlib import Path

from engine.render.ffmpeg_cmds import concat_cmd, segments_list_text
from engine.util import run, ffprobe_duration


def build_final(segment_paths: list[Path], out: str | Path, log, *,
                remux_audio: bool = False,
                planned_total_dur: float | None = None) -> dict:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    list_path = out.parent / f"{out.stem}.segments.list"
    list_path.write_text(segments_list_text(list(segment_paths)),
                         encoding="utf-8", newline="\n")
    run(concat_cmd(list_path, out, remux_audio=remux_audio), log=log)
    dur = ffprobe_duration(out)
    res = {"out": out, "duration": dur, "problems": []}
    if planned_total_dur and abs(dur - planned_total_dur) > 0.15:
        res["problems"].append(
            f"final duration {dur:.2f} != planned {planned_total_dur:.2f}")
    log.info(f"final: {out.name} ({dur/3600:.2f} h)")
    return res

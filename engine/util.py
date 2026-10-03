"""Subprocess / hash / path helpers shared by every module.

Windows rules baked in here:
- subprocesses are always list-argv, never shell=True;
- filtergraph paths use forward slashes and single-quote escaping.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


class FFmpegError(RuntimeError):
    def __init__(self, cmd: list[str], returncode: int, stderr_tail: str):
        self.cmd = cmd
        self.returncode = returncode
        self.stderr_tail = stderr_tail
        super().__init__(
            f"ffmpeg/ffprobe exited {returncode}: {' '.join(cmd[:6])} ...\n"
            f"--- stderr tail ---\n{stderr_tail}"
        )


def run(cmd: list[str | os.PathLike], *, log=None, check: bool = True,
        timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run argv list, capture output, never use a shell."""
    argv = [str(c) for c in cmd]
    if log is not None:
        log.info("$ " + subprocess.list2cmdline(argv))
    proc = subprocess.run(argv, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)
    if check and proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout or "").splitlines()[-40:])
        raise FFmpegError(argv, proc.returncode, tail)
    return proc


def ffprobe_json(path: str | Path, *, show_streams: bool = False,
                 show_format: bool = False, select: str | None = None) -> dict:
    cmd = ["ffprobe", "-v", "error"]
    if select:
        cmd += ["-select_streams", select]
    if show_streams:
        cmd.append("-show_streams")
    if show_format:
        cmd.append("-show_format")
    cmd += ["-print_format", "json", str(path)]
    proc = run(cmd)
    return json.loads(proc.stdout or "{}")


def ffprobe_duration(path: str | Path) -> float:
    """Duration in seconds, preferring format duration then first stream."""
    info = ffprobe_json(path, show_streams=True, show_format=True)
    try:
        return float(info["format"]["duration"])
    except (KeyError, TypeError, ValueError):
        pass
    for s in info.get("streams", []):
        if s.get("duration"):
            return float(s["duration"])
    raise RuntimeError(f"ffprobe could not determine duration of {path}")


def ffprobe_video_info(path: str | Path) -> dict:
    """{width,height,duration,r_frame_rate,pix_fmt,sample_aspect_ratio,codec_name, ...}"""
    info = ffprobe_json(path, select="v:0", show_streams=True)
    streams = info.get("streams", [])
    if not streams:
        raise RuntimeError(f"no video stream in {path}")
    s = streams[0]
    out = {k: s.get(k) for k in
           ("codec_name", "width", "height", "pix_fmt", "r_frame_rate",
            "avg_frame_rate", "sample_aspect_ratio", "duration",
            "profile", "level")}
    return out


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ---- ffmpeg filtergraph path escaping ---------------------------------

def ffgraph_path(path: str | Path) -> str:
    """Render an absolute path for use inside a filtergraph argument.

    Two parse levels matter: the graph parser strips single quotes and
    passes their content verbatim; the option parser then splits values on
    `:`.  So we wrap in quotes (protects `[];,=` from the graph parser and
    passes backslashes through literally) AND escape every colon as `\\:`
    so the option parser keeps Windows drive letters inside the value.
    Paths containing `'` are rejected — keep work paths quote-free.
    """
    s = str(path).replace("\\", "/")
    if "'" in s:
        raise ValueError(f"single quotes unsupported in filtergraph path: {s}")
    s = s.replace(":", "\\:")
    return f"'{s}'"


def ffconcat_path(path: str | Path) -> str:
    """Escape a path for a line inside an ffconcat list file:
    wrap in single quotes, internal ' -> '\\'' (list tokenizer rule)."""
    s = str(path).replace("\\", "/")
    s = s.replace("'", "'\\''")
    return f"'{s}'"


def ms_to_ass_time(ms: float) -> str:
    """ASS time stamp h:mm:ss.cc (centiseconds). 10 ms = 1 cs."""
    total_cs = int(round(ms / 10))
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def sec(x: float) -> str:
    """Compact seconds for ffmpeg args."""
    return f"{x:.3f}".rstrip("0").rstrip(".")

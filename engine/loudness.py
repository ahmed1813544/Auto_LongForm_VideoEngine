"""Loudness: measure master.wav once with ebur128 -> one fixed gain in dB.

Per-segment `loudnorm` would pump the noise floor between segments (the
voiceover gain visibly breathing at every seam).  A single linear gain
computed from one measurement of the whole track has no such artifact, and
it rides the `volume=` filter every segment already carries.
"""
from __future__ import annotations

import re
from pathlib import Path

from engine.util import run

TARGET_LUFS = -16.0      # comfortable voice level; YouTube normalizes -14
MAX_GAIN_DB = 12.0
MIN_GAIN_DB = -12.0

_I_RE = re.compile(r"I:\s+(-?\d+(?:\.\d+)?)\s+LUFS")


def parse_integrated_lufs(ebur128_stderr: str) -> float | None:
    """Last 'I: -xx.x LUFS' line == the Summary block value."""
    vals = _I_RE.findall(ebur128_stderr)
    return float(vals[-1]) if vals else None


def measure_lufs(path: str | Path, *, timeout: float = 3600) -> float:
    proc = run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
                "-filter_complex", "ebur128", "-f", "null", "-"],
               check=True, timeout=timeout)
    val = parse_integrated_lufs(proc.stderr or "")
    if val is None:
        raise RuntimeError("ebur128: could not parse integrated loudness")
    return val


def gain_db(target_lufs: float = TARGET_LUFS,
            measured_lufs: float | None = None) -> float:
    """Clamped linear gain from measurement to target (0 dB if unmeasured)."""
    if measured_lufs is None:
        return 0.0
    g = target_lufs - measured_lufs
    return round(max(MIN_GAIN_DB, min(MAX_GAIN_DB, g)), 2)

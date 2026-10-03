"""Per-job state.json: phase completion + per-segment status for resume.

Everything is written atomically.  On resume we re-probe finished segments —
a file that vanished (OneDrive placeholder eviction, manual delete) demotes
its segment back to pending instead of silently corrupting the concat.
"""
from __future__ import annotations

import json
from pathlib import Path

from engine.util import atomic_write_json, ffprobe_duration, sha256_text


def params_hash(params: dict) -> str:
    """Stable hash of the build parameters that invalidate prior work."""
    return sha256_text(json.dumps(params, sort_keys=True,
                                  ensure_ascii=False))


class ParamsMismatch(RuntimeError):
    pass


class JobState:
    VERSION = 1

    def __init__(self, job_dir: Path, data: dict):
        self.job_dir = Path(job_dir)
        self.d = data

    # ---- persistence ------------------------------------------------------
    @property
    def path(self) -> Path:
        return self.job_dir / "state.json"

    @classmethod
    def load_or_new(cls, job_dir: str | Path, *, job_id: str,
                    phash: str, force_new: bool = False) -> "JobState":
        job_dir = Path(job_dir)
        p = job_dir / "state.json"
        if p.exists() and not force_new:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                raise ParamsMismatch(f"{p} is corrupt; rerun with --force-new")
            if data.get("params_hash") != phash:
                raise ParamsMismatch(
                    f"job '{job_id}' was built with different parameters; "
                    "use --force-new to restart it or a new --job-id")
            return cls(job_dir, data)
        data = {"version": cls.VERSION, "job_id": job_id,
                "params_hash": phash, "phases": {}, "segments": [],
                "plan": {}}
        return cls(job_dir, data)

    def save(self) -> None:
        atomic_write_json(self.path, self.d)

    # ---- phases ------------------------------------------------------------
    def phase_done(self, name: str) -> bool:
        return bool(self.d["phases"].get(name, {}).get("done"))

    def phase_output(self, name: str, key: str, default=None):
        return self.d["phases"].get(name, {}).get(key, default)

    def mark_phase(self, name: str, **out) -> None:
        e = self.d["phases"].setdefault(name, {})
        e.update(out)
        e["done"] = True
        self.save()

    # ---- segments -----------------------------------------------------------
    def set_segments(self, planned: list[dict]) -> None:
        """Store the plan; keep existing statuses for matching indices."""
        old = {s["idx"]: s for s in self.d.get("segments", [])}
        self.d["segments"] = [
            {**p, "state": old.get(p["idx"], {}).get("state", "pending"),
             "path": old.get(p["idx"], {}).get("path"),
             "problems": old.get(p["idx"], {}).get("problems")}
            for p in planned]
        self.save()

    def segments(self) -> list[dict]:
        return self.d.get("segments", [])

    def segment(self, idx: int) -> dict | None:
        for s in self.d.get("segments", []):
            if s["idx"] == idx:
                return s
        return None

    def mark_segment(self, idx: int, state: str, *, path: str | None = None,
                     problems: list[str] | None = None) -> None:
        s = self.segment(idx)
        if s is None:
            return
        s["state"] = state
        if path is not None:
            s["path"] = path
        s["problems"] = problems
        self.save()

    def demote_stale_done(self, *, log=None) -> int:
        """Done segments whose file is missing or the wrong length go back
        to pending.  Returns the number demoted."""
        n = 0
        for s in self.d.get("segments", []):
            if s.get("state") != "done":
                continue
            p = s.get("path")
            ok = False
            if p and Path(p).exists():
                try:
                    ok = abs(ffprobe_duration(p) - s["dur_s"]) <= 0.15
                except (RuntimeError, OSError):
                    ok = False
            if not ok:
                s["state"] = "pending"
                s["path"] = None
                n += 1
                if log:
                    log.warn(f"segment {s['idx']} marked done but file is "
                             "missing/short — will re-render")
        if n:
            self.save()
        return n

    # ---- summary -------------------------------------------------------------
    def counts(self) -> dict:
        c: dict[str, int] = {}
        for s in self.d.get("segments", []):
            c[s.get("state", "pending")] = c.get(s.get("state", "pending"),
                                                 0) + 1
        return c

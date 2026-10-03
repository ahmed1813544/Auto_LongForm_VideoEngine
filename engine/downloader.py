"""Stock clip downloader + global clip library index.

Files live under the work dir (outside OneDrive): raw/<provider>_<id>.mp4
and norm/<provider>_<id>.mp4, tracked in library_index.json keyed by
"provider:id".  Downloads stream to a .part file and resume via HTTP Range;
an ffprobe gate rejects corrupt/truncated files before they enter the
library.  The library is shared across jobs — a 3 h build run twice pays
for clips once.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from engine.stock.base import Clip
from engine.util import atomic_write_json, ffprobe_duration, sha256_file


# ---- library index -------------------------------------------------------

def load_index(cfg) -> dict:
    p = Path(cfg.library_index_path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}
    return {}


def save_index(cfg, index: dict) -> None:
    """Merge-write: entries already on disk but absent from ``index``
    survive.  A stale in-memory copy (a second build process left orphaned
    by a session drop, or one loaded before a concurrent writer saved) must
    not orphan entries -- that is how water30 lost 92 fresh clips from the
    index and silently fell back to other jobs' footage (2026-09-15)."""
    merged = load_index(cfg)
    merged.update(index)
    atomic_write_json(cfg.library_index_path, merged)


def load_blocklist(cfg) -> dict:
    """Content-QC blocklist: {"provider:id": reason} of clips banned from
    every pool (urban/industrial/machine/person/indoor/abstract subjects —
    the landscape-only rule).  Blocked keys stay in the library index, so
    search dedupe also never re-downloads them; deleting a norm file is
    NOT needed (and not sufficient for future jobs) to ban a clip.
    """
    if cfg is None:
        return {}
    p = Path(cfg.clip_blocklist_path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("blocked", {})
        except (ValueError, OSError):
            return {}
    return {}


def raw_path(cfg, clip: Clip) -> Path:
    return Path(cfg.clips_raw_dir) / f"{clip.provider}_{clip.id}.mp4"


def norm_path(cfg, clip: Clip) -> Path:
    return Path(cfg.clips_norm_dir) / f"{clip.provider}_{clip.id}.mp4"


# ---- download -------------------------------------------------------------

def download(cfg, clip: Clip, *, log=None) -> Path:
    """Fetch one clip to the raw library (resumable, ffprobe-gated)."""
    dst = raw_path(cfg, clip)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    part = dst.with_name(dst.name + ".part")
    pos = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    r = requests.get(clip.download_url, stream=True, timeout=(15, 120),
                     headers=headers)
    try:
        if r.status_code == 200 and pos:
            pos = 0                       # server ignored Range: restart
        if r.status_code not in (200, 206):
            raise RuntimeError(
                f"HTTP {r.status_code} fetching {clip.key}")
        mode = "ab" if (pos and r.status_code == 206) else "wb"
        with open(part, mode) as f:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)
    except BaseException:
        r.close()
        raise                            # keep .part for resume
    r.close()
    part.replace(dst)
    try:
        dur = ffprobe_duration(dst)
    except Exception as e:
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"integrity gate failed for {clip.key}: {e}")
    if dur < 1.0:
        dst.unlink(missing_ok=True)
        raise RuntimeError(
            f"integrity gate: {clip.key} probes only {dur:.2f}s")
    if log:
        log.info(f"downloaded {clip.key} ({dur:.1f}s, "
                 f"{dst.stat().st_size / 1e6:.1f} MB)")
    return dst


def ensure_clips(cfg, clips, *, log=None, workers: int = 3) -> dict:
    """Parallel download of a clip list.

    Returns {clip.key: raw_path_str | None}; updates + saves the index for
    whatever succeeded (None entries mean the planner must swap in other
    clips).
    """
    cfg.ensure_dirs()
    index = load_index(cfg)
    lock = threading.Lock()

    def one(clip: Clip):
        entry = index.get(clip.key)
        if entry and entry.get("raw") and Path(entry["raw"]).exists():
            return clip.key, str(entry["raw"])
        try:
            p = download(cfg, clip, log=log)
        except Exception as e:  # noqa: BLE001 — one bad clip must not kill pool
            if log:
                log.warn(f"download failed {clip.key}: {e}")
            return clip.key, None
        try:
            digest = sha256_file(p)
        except OSError:
            digest = None
        with lock:
            index[clip.key] = {
                "provider": clip.provider, "id": clip.id,
                "duration_s": clip.duration_s,
                "width": clip.width, "height": clip.height,
                "size_bytes": p.stat().st_size, "sha256": digest,
                "term": clip.term, "raw": str(p), "norm": None,
            }
        return clip.key, str(p)

    results: dict[str, str | None] = {}
    if clips:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for key, path in ex.map(one, clips):
                results[key] = path
    save_index(cfg, index)
    ok = sum(1 for v in results.values() if v)
    if log:
        log.info(f"downloads ready: {ok}/{len(clips)}")
    return results

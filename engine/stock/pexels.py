"""Pexels video API adapter.

GET https://api.pexels.com/videos/search  params: query, per_page, page,
orientation=landscape;  header: Authorization.  (Pexels retired the old
Access-Key header: it now answers 401 "Missing API key" even with a valid
key -- observed 2026-09-08.)
Each hit lists download files under ``video_files[]`` (newer responses may
use ``video_versions[]``).  Real-world quirks this adapter tolerates:
- width/height sometimes null,
- bitrate spelled ``bit_rate`` or ``bitrate``,
- files without a usable link,
- portrait entries slipping past the orientation filter,
- the ``size`` query param (any value) now triggers a misleading 401
  "Missing API key" — do NOT send it; ``_pick_file`` chooses resolution.
"""
from __future__ import annotations

import time
from typing import Sequence

from engine.stock.base import (DEFAULT_MAX_H, DEFAULT_SIZE_CAP, Clip,
                               RETRYABLE, StockProvider, exceeds_cap,
                               request_json)

URL = "https://api.pexels.com/videos/search"
_NO_SIZE = 1 << 62        # unknown size sorts after any known one

# 2026-09 note: the "intermittent, misleading 401 with a correct key" this
# ladder was built for turned out to be Pexels killing the Access-Key
# header (fixed above). 401-as-retryable is kept anyway: real quota windows
# do exist, and a genuinely bad key still ends in a hard failure after the
# ladder (~75 s per request).
REQUEST_GAP_S = 0.7
FLAKY_AUTH = (401,)


def _pick_file(files: list, *, max_h: int = DEFAULT_MAX_H,
               size_cap: int = DEFAULT_SIZE_CAP) -> dict | None:
    """Smallest landscape >=max_h that fits the cap (default 720p/~80 MB —
    the deliverable is 1080p and normalize upscales, so bigger sources only
    cost bytes and decode time); else best landscape below the floor; if
    every sized file is over cap, the smallest of the oversized ones
    (footage beats pool starvation); else a null-dims file (orientation
    trusted via the API param)."""
    known: list[dict] = []
    unknown: list[dict] = []
    for f in files or []:
        link = f.get("link")
        if not link:
            continue
        w, h = f.get("width"), f.get("height")
        if w and h:
            if h > w:
                continue        # portrait despite orientation=landscape
            known.append(f)
        else:
            unknown.append(f)   # null dims: last resort
    fit = [f for f in known if not exceeds_cap(f.get("size"), size_cap)]
    pool = fit or known
    hi = [f for f in pool if (f.get("height") or 0) >= max_h]
    if hi:
        return min(hi, key=lambda f: (f["height"], f["width"],
                                      int(f["size"]) if f.get("size")
                                      else _NO_SIZE))
    if pool:
        return max(pool, key=lambda f: (f.get("height") or 0,
                                        f.get("width") or 0))
    return unknown[0] if unknown else None


def parse_hit(hit: dict, term: str = "", *, max_h: int = DEFAULT_MAX_H,
              size_cap: int = DEFAULT_SIZE_CAP) -> Clip | None:
    """One search hit -> Clip (None if nothing usable). Pure; tested."""
    files = hit.get("video_files") or hit.get("video_versions") or []
    f = _pick_file(files, max_h=max_h, size_cap=size_cap)
    if f is None:
        return None
    dur = hit.get("duration")
    if dur is None:
        dur = f.get("duration")
    try:
        dur = float(dur or 0)
    except (TypeError, ValueError):
        dur = 0.0
    return Clip(
        provider="pexels",
        id=str(hit.get("id")),
        duration_s=dur,           # 0 allowed; downloader ffprobe is the gate
        width=int(f.get("width") or 0),
        height=int(f.get("height") or 0),
        download_url=str(f["link"]),
        bitrate=f.get("bit_rate") or f.get("bitrate"),
        size_bytes=f.get("size"),
        tags="",
        term=term,
        thumb_url=hit.get("image") or "",
    )


class Pexels(StockProvider):
    name = "pexels"

    def search(self, terms: Sequence[str], *, per_term: int = 60,
               pages: int = 2, max_h: int = DEFAULT_MAX_H,
               size_cap: int = DEFAULT_SIZE_CAP) -> list[Clip]:
        headers = {"Authorization": self.key}
        out: list[Clip] = []
        seen: set[str] = set()
        self._n_req = 0
        errors: list[str] = []
        for term in terms:
            # Per-term resilience: the misleading 401 tends to stick to a
            # query's backend path; one bad term must not kill the pool.
            try:
                self._search_term(term, per_term, pages, headers, seen, out,
                                  max_h=max_h, size_cap=size_cap)
            except RuntimeError as e:
                errors.append(str(e)[:120])
                if self.log:
                    self.log.warn(f"pexels: term {term!r} failed, trying "
                                  f"next ({str(e)[:80]})")
        if not out and errors:
            raise RuntimeError("pexels: all terms failed: "
                               + " | ".join(errors)[:300])
        return out

    def _search_term(self, term: str, per_term: int, pages: int,
                     headers: dict, seen: set, out: list, *,
                     max_h: int = DEFAULT_MAX_H,
                     size_cap: int = DEFAULT_SIZE_CAP) -> None:
        for page in range(1, pages + 1):
            if getattr(self, "_n_req", 0):
                time.sleep(REQUEST_GAP_S)
            self._n_req = getattr(self, "_n_req", 0) + 1
            params = {
                "query": term,
                "per_page": max(3, min(int(per_term), 80)),
                "orientation": "landscape", "page": page,
            }
            data = request_json(self.s, URL, headers=headers,
                                params=params, log=self.log,
                                base_delay=15.0,
                                retry_statuses=RETRYABLE + FLAKY_AUTH)
            videos = data.get("videos") or data.get("collection") or []
            if not videos:
                break
            for v in videos:
                c = parse_hit(v, term, max_h=max_h, size_cap=size_cap)
                if c and c.id not in seen:
                    seen.add(c.id)
                    out.append(c)
            total = data.get("total_results")
            if not total or page * params["per_page"] >= int(total):
                break

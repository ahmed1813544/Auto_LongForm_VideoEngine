"""Pixabay video API adapter.

GET https://pixabay.com/api/videos/  params: key, q, per_page, safesearch,
order=popular, page.  Each hit carries renditions under
``videos.{tiny,small,medium,large}`` with width/height/size/url.
Policy: landscape only (reject h>w); pick the smallest rendition >=max_h
tall (default 720 — the deliverable is 1080p and normalize upscales, so
bigger sources only cost bytes and decode time), downgrading files over
the size cap (~80 MB) to a smaller rendition when the API reported one.
"""
from __future__ import annotations

from typing import Sequence

from engine.stock.base import (DEFAULT_MAX_H, DEFAULT_SIZE_CAP, Clip,
                               StockProvider, exceeds_cap, request_json)

URL = "https://pixabay.com/api/videos/"
TIERS = ("large", "medium", "small", "tiny")  # label order (not preference)
_NO_SIZE = 1 << 62        # unknown size sorts after any known one


def _rank(c) -> tuple:
    """Shortest first, then narrowest, then fewest bytes."""
    size = c[2].get("size")
    return (c[0], c[1], int(size) if size else _NO_SIZE)


def _pick_rendition(videos: dict, *, max_h: int = DEFAULT_MAX_H,
                    size_cap: int = DEFAULT_SIZE_CAP) -> dict | None:
    """Smallest landscape rendition >=max_h that fits the cap; else the
    largest landscape one below the floor; if every candidate is over cap,
    the smallest of the oversized ones (footage beats pool starvation)."""
    cands = []
    for tier in TIERS:
        r = videos.get(tier) or {}
        url = r.get("url")
        w = r.get("width") or 0
        h = r.get("height") or 0
        if not url or not w or not h:
            continue          # some tiers ship without url/size fields
        if h > w:
            continue          # portrait clip — wrong shape for 16:9
        cands.append((h, w, r))
    if not cands:
        return None
    fit = [c for c in cands if not exceeds_cap(c[2].get("size"), size_cap)]
    pool = fit or cands
    hi = [c for c in pool if c[0] >= max_h]
    if hi:
        return min(hi, key=_rank)[2]
    return max(pool, key=lambda c: (c[0], c[1]))[2]


def parse_hit(hit: dict, term: str = "", *, max_h: int = DEFAULT_MAX_H,
              size_cap: int = DEFAULT_SIZE_CAP) -> Clip | None:
    """One /videos/ hit -> Clip (None if nothing usable). Pure; tested."""
    r = _pick_rendition(hit.get("videos") or {}, max_h=max_h,
                        size_cap=size_cap)
    if r is None:
        return None
    dur = hit.get("duration") or 0
    if not dur:
        return None
    return Clip(
        provider="pixabay",
        id=str(hit.get("id")),
        duration_s=float(dur),
        width=int(r["width"]),
        height=int(r["height"]),
        download_url=str(r["url"]),
        bitrate=None,                    # Pixabay doesn't report it
        size_bytes=r.get("size"),
        tags=hit.get("tags") or "",
        term=term,
        thumb_url=hit.get("image") or "",
    )


class Pixabay(StockProvider):
    name = "pixabay"

    def search(self, terms: Sequence[str], *, per_term: int = 60,
               pages: int = 2, max_h: int = DEFAULT_MAX_H,
               size_cap: int = DEFAULT_SIZE_CAP) -> list[Clip]:
        out: list[Clip] = []
        seen: set[str] = set()
        for term in terms:
            for page in range(1, pages + 1):
                params = {
                    "key": self.key, "q": term,
                    "per_page": max(3, min(int(per_term), 200)),
                    "safesearch": "true", "order": "popular", "page": page,
                }
                data = request_json(self.s, URL, params=params, log=self.log)
                hits = data.get("hits") or []
                if not hits:
                    break
                for h in hits:
                    c = parse_hit(h, term, max_h=max_h, size_cap=size_cap)
                    if c and c.id not in seen:
                        seen.add(c.id)
                        out.append(c)
                # paginate only while the API says more results exist
                try:
                    total = int(data.get("totalHits") or 0)
                except (TypeError, ValueError):
                    total = 0
                if page * params["per_page"] >= total:
                    break
        return out

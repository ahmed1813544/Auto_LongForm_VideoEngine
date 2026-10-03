"""Clip descriptor + shared HTTP helpers for stock providers.

Every adapter normalizes its provider's JSON into one `Clip`; everything
downstream (planner, downloader) only ever sees Clips, never raw API shapes.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Sequence

import requests

# statuses worth retrying on stock APIs; anything else 4xx is a hard error
# (providers may opt in to more — see Pexels' flaky-401 rate quirk)
RETRYABLE = (429, 500, 502, 503, 504)

# Rendition policy shared by the adapters: the deliverable is 1080p and
# normalize cover-scales every clip anyway, so a 720p source is
# indistinguishable after upscale while downloading ~2-4x fewer bytes and
# decoding faster. Files bigger than SIZE_CAP are downgraded to a smaller
# rendition when the API reported a size. Overridable per search
# (--stock-max-h; 1080 restores the old full-HD-source behavior).
DEFAULT_MAX_H = 720
DEFAULT_SIZE_CAP = 80 * 1024 * 1024


def exceeds_cap(size, cap) -> bool:
    """True only when the API reported a size AND it is over the cap —
    an unknown size is never a disqualifier."""
    if cap is None or size is None:
        return False
    try:
        return int(size) > int(cap)
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class Clip:
    provider: str
    id: str
    duration_s: float
    width: int           # 0 = unknown from API (downloader ffprobes anyway)
    height: int
    download_url: str
    bitrate: int | None = None    # bits/s when the API reported it
    size_bytes: int | None = None
    tags: str = ""
    term: str = ""                # search term that found it
    thumb_url: str = ""           # provider preview still (content QC vetting)

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.id}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["key"] = self.key
        return d


def request_json(session: requests.Session, url: str, *, params=None,
                 headers=None, log=None, attempts: int = 4,
                 timeout: float = 20.0, base_delay: float = 1.0,
                 retry_statuses: tuple = RETRYABLE) -> dict:
    """GET + parse JSON with exp backoff on retryable statuses; honors
    Retry-After. base_delay scales the backoff ladder (pexels rides out
    multi-minute quota windows with base_delay=15)."""
    last = "unknown"
    delay = base_delay
    for n in range(attempts):
        r = None
        try:
            r = session.get(url, params=params, headers=headers,
                            timeout=timeout)
        except requests.RequestException as e:
            last = f"network error: {e}"
        else:
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    last = "non-JSON response"
            else:
                last = f"HTTP {r.status_code}: {(r.text or '')[:120]}"
                if r.status_code not in retry_statuses:
                    break                      # auth/params: retrying won't help
        if n < attempts - 1:
            wait = delay
            if r is not None:
                ra = r.headers.get("Retry-After", "")
                try:
                    wait = max(wait, float(ra))
                except ValueError:
                    pass
            if log:
                log.warn(f"stock API {last!r}; retry {n + 1}/{attempts - 1} "
                         f"in {wait:.0f}s")
            time.sleep(wait)
            delay = min(delay * 2, 30.0)
    raise RuntimeError(f"stock request failed: {url} -> {last}")


class StockProvider:
    """Subclasses implement search(); parsing lives in module-level fns."""
    name = "?"

    def __init__(self, key: str, *, log=None,
                 session: requests.Session | None = None):
        self.key = key
        self.log = log
        self.s = session or requests.Session()

    def search(self, terms: Sequence[str], *, per_term: int = 60,
               pages: int = 2, max_h: int = DEFAULT_MAX_H,
               size_cap: int = DEFAULT_SIZE_CAP) -> list[Clip]:
        raise NotImplementedError


def search_all(providers: Sequence[StockProvider], terms: Sequence[str], *,
               per_term: int = 60, pages: int = 2,
               max_h: int = DEFAULT_MAX_H,
               size_cap: int = DEFAULT_SIZE_CAP,
               log=None) -> list[Clip]:
    """Union of providers for every term; dedupe by provider:id.

    A provider that hard-fails all terms is warned about and skipped —
    the job proceeds on whichever library source still works.
    """
    out: list[Clip] = []
    seen: set[str] = set()
    for p in providers:
        try:
            clips = p.search(terms, per_term=per_term, pages=pages,
                             max_h=max_h, size_cap=size_cap)
        except Exception as e:  # noqa: BLE001 — keep other providers alive
            if log:
                log.warn(f"{p.name}: search failed entirely: {e}")
            continue
        added = 0
        for c in clips:
            if c.key in seen:
                continue
            seen.add(c.key)
            out.append(c)
            added += 1
        if log:
            log.info(f"{p.name}: {added} new clips")
    return out

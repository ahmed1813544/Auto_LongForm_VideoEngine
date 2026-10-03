"""Adapter parse tests against canned API shapes — no network.

Fixtures mirror documented real response shapes, including the known quirks
(portrait hits, null dims, bit_rate|bitrate drift, video_versions).
Rendition policy under test: smallest landscape >=max_h (default 720 —
normalize upscales to the 1080p deliverable) that fits the ~80 MB size cap.
"""
from engine.stock.base import (DEFAULT_MAX_H, DEFAULT_SIZE_CAP, Clip,
                               search_all)
from engine.stock.pexels import Pexels
from engine.stock.pexels import parse_hit as px_parse
from engine.stock.pexels import _pick_file as px_pick
from engine.stock.pixabay import parse_hit as pb_parse
from engine.stock.pixabay import _pick_rendition as pb_pick


# ---- Pixabay --------------------------------------------------------------

PB_HIT = {
    "id": 12345, "pageURL": "https://pixabay.com/videos/x-12345/",
    "tags": "forest, trees, fog", "duration": 14,
    "videos": {
        "large": {"id": 3, "url": "https://cdn/pixabay/large.mp4",
                  "width": 1920, "height": 1080, "size": 5_000_000},
        "medium": {"id": 2, "url": "https://cdn/pixabay/medium.mp4",
                   "width": 1280, "height": 720, "size": 2_000_000},
        "small": {"id": 1, "url": "https://cdn/pixabay/small.mp4",
                  "width": 960, "height": 540, "size": 900_000},
        "tiny": {"id": 0, "url": "https://cdn/pixabay/tiny.mp4",
                 "width": 640, "height": 360, "size": 400_000},
    },
}


def test_pixabay_picks_smallest_720_by_default():
    # 720p source: fewer bytes, faster decode; normalize upscales to 1080p
    c = pb_parse(PB_HIT, "forest")
    assert c.provider == "pixabay" and c.id == "12345"
    assert c.width == 1280 and c.height == 720
    assert c.download_url.endswith("medium.mp4")
    assert c.size_bytes == 2_000_000 and c.duration_s == 14.0
    assert c.term == "forest" and c.key == "pixabay:12345"


def test_pixabay_max_h_1080_restores_full_hd():
    c = pb_parse(PB_HIT, "forest", max_h=1080)
    assert c.width == 1920 and c.height == 1080
    assert c.download_url.endswith("large.mp4")


def test_pixabay_missing_large_falls_back():
    hit = {**PB_HIT, "videos": {k: v for k, v in
                                PB_HIT["videos"].items() if k != "large"}}
    assert pb_pick(hit["videos"])["url"].endswith("medium.mp4")


def test_pixabay_rejects_portrait_and_broken():
    hit = {**PB_HIT, "videos": {
        "large": {"url": "https://c/l.mp4", "width": 1080, "height": 1920},
        "medium": {"url": "", "width": 1280, "height": 720},
        "small": {"url": "https://c/s.mp4"},          # no dims
    }}
    assert pb_parse(hit) is None


def test_pixabay_prefers_1080_over_4k():
    # with the 720 floor, 1080 still beats 4K: bytes + decode cost only
    hit = {**PB_HIT, "videos": {
        "large": {"url": "https://c/4k.mp4", "width": 3840, "height": 2160},
        "medium": {"url": "https://c/1080.mp4", "width": 1920,
                   "height": 1080},
    }}
    assert pb_pick(hit["videos"])["url"].endswith("1080.mp4")


def test_pixabay_prefers_shortest_at_or_above_floor():
    # 2048x858 (cinemascope) is >= the 720 floor and shorter than 1080 —
    # under the small-source policy it wins: fewer bytes to fetch/decode
    hit = {**PB_HIT, "videos": {
        "large": {"url": "https://c/l.mp4", "width": 1920, "height": 1080},
        "medium": {"url": "https://c/m.mp4", "width": 2048, "height": 858},
    }}
    assert pb_pick(hit["videos"])["url"].endswith("m.mp4")


def test_pixabay_size_cap_downgrades():
    # large + medium are over the ~80 MB cap -> the fitting small one wins
    hit = {**PB_HIT, "videos": {
        "large": {"url": "https://c/l.mp4", "width": 1920, "height": 1080,
                  "size": 150_000_000},
        "medium": {"url": "https://c/m.mp4", "width": 1280, "height": 720,
                   "size": 90_000_000},
        "small": {"url": "https://c/s.mp4", "width": 960, "height": 540,
                  "size": 20_000_000},
    }}
    assert pb_pick(hit["videos"])["url"].endswith("s.mp4")


def test_pixabay_all_over_cap_still_returns_footage():
    # pool starvation is worse than one big file: fall back to the
    # smallest-dims oversized rendition
    hit = {**PB_HIT, "videos": {
        "large": {"url": "https://c/4k.mp4", "width": 3840, "height": 2160,
                  "size": 400_000_000},
        "medium": {"url": "https://c/m.mp4", "width": 1920, "height": 1080,
                   "size": 120_000_000},
    }}
    assert pb_pick(hit["videos"])["url"].endswith("m.mp4")


def test_pixabay_unknown_size_not_gated():
    # no size reported -> the cap can't apply; dims decide as before
    hit = {**PB_HIT, "videos": {
        "large": {"url": "https://c/l.mp4", "width": 1920, "height": 1080},
    }}
    assert pb_pick(hit["videos"])["url"].endswith("l.mp4")


def test_pixabay_cap_is_overridable():
    r = pb_pick(PB_HIT["videos"], size_cap=1_000_000)   # cap below medium
    assert r["url"].endswith("small.mp4")               # largest fitting


# ---- Pexels ---------------------------------------------------------------

def _px(files, duration=12):
    return {"id": 999, "duration": duration, "image": "thumb.jpg",
            "video_files": files}


def test_pexels_picks_smallest_720_by_default():
    hit = _px([
        {"id": 1, "bit_rate": 2_000_000, "quality": "hd",
         "extensions": ["mp4"], "link": "https://cdn/px/1080.mp4",
         "width": 1920, "height": 1080},
        {"id": 2, "bit_rate": 800_000, "quality": "sd",
         "extensions": ["mp4"], "link": "https://cdn/px/720.mp4",
         "width": 1280, "height": 720},
    ])
    c = px_parse(hit, "storm")
    assert c.width == 1280 and c.download_url.endswith("720.mp4")
    assert c.bitrate == 800_000 and c.key == "pexels:999"


def test_pexels_max_h_1080_restores_full_hd():
    hit = _px([
        {"link": "https://cdn/px/1080.mp4", "width": 1920, "height": 1080,
         "bit_rate": 2_000_000},
        {"link": "https://cdn/px/720.mp4", "width": 1280, "height": 720},
    ])
    c = px_parse(hit, max_h=1080)
    assert c.download_url.endswith("1080.mp4")
    assert c.bitrate == 2_000_000


def test_pexels_size_cap_downgrades():
    # the 97 MB 1080p file is over cap -> the 25 MB 720p one wins
    hit = _px([
        {"link": "https://cdn/px/1080.mp4", "width": 1920, "height": 1080,
         "size": 97_000_000},
        {"link": "https://cdn/px/720.mp4", "width": 1280, "height": 720,
         "size": 25_000_000},
    ])
    assert px_parse(hit).download_url.endswith("720.mp4")


def test_pexels_unknown_size_not_gated():
    hit = _px([{"link": "https://c/a.mp4", "width": 1920, "height": 1080},
               {"link": "https://c/b.mp4", "width": 1280, "height": 720,
                "size": DEFAULT_SIZE_CAP + 1}])
    # b is over cap, a has no reported size (not gated) -> a wins
    assert px_pick(hit["video_files"])["link"].endswith("a.mp4")


def test_pexels_known_landscape_beats_null_dims():
    # null-dims can't be verified landscape -> prefer a known 720 over it
    hit = _px([{"link": "https://cdn/px/nodate.mp4", "quality": "hd"},
               {"link": "https://cdn/px/720.mp4", "width": 1280,
                "height": 720, "bitrate": 900_000}])
    c = px_parse(hit)
    assert c.download_url.endswith("720.mp4")
    assert c.bitrate == 900_000     # bit-rate key-name drift tolerated


def test_pexels_null_dims_only():
    assert px_pick([{"link": "https://c/x.mp4"}])["link"].endswith("x.mp4")


def test_pexels_prefers_sd_when_no_hd():
    hit = _px([{"link": "https://cdn/px/720.mp4", "width": 1280,
                "height": 720, "bit_rate": 900_000},
               {"link": "https://cdn/px/480.mp4", "width": 854,
                "height": 480}])
    assert px_parse(hit).download_url.endswith("720.mp4")


def test_pexels_rejects_portrait_and_linkless():
    hit = _px([{"link": "https://c/p.mp4", "width": 1080, "height": 1920},
               {"width": 1920, "height": 1080}])          # no link
    assert px_parse(hit) is None


def test_pexels_video_versions_alias():
    hit = _px([])
    hit["video_versions"] = [{"link": "https://c/v.mp4", "width": 1920,
                              "height": 1080}]
    c = px_parse(hit)
    assert c.download_url == "https://c/v.mp4"


def test_pexels_missing_duration_is_zero():
    hit = _px([{"link": "https://c/v.mp4", "width": 1920, "height": 1080}])
    hit["duration"] = None
    assert px_parse(hit).duration_s == 0.0


# ---- search_all ------------------------------------------------------------

class _FakeProvider:
    def __init__(self, name, clips, raises=None):
        self.name = name
        self._clips = clips
        self._raises = raises
        self.kwargs = None

    def search(self, terms, *, per_term=60, pages=2, max_h=DEFAULT_MAX_H,
               size_cap=DEFAULT_SIZE_CAP):
        if self._raises:
            raise self._raises
        self.kwargs = {"per_term": per_term, "pages": pages,
                       "max_h": max_h, "size_cap": size_cap}
        return self._clips


def _clip(prov, id_):
    return Clip(provider=prov, id=str(id_), duration_s=10, width=1920,
                height=1080, download_url=f"https://c/{prov}-{id_}.mp4")


def test_search_all_dedupes_and_survives_failures():
    a = _FakeProvider("pixabay", [_clip("pixabay", 1), _clip("pixabay", 2)])
    b = _FakeProvider("pexels", [_clip("pexels", 1), _clip("pixabay", 1)])
    dead = _FakeProvider("broken", [], raises=RuntimeError("401"))
    out = search_all([a, b, dead], ["forest"])
    keys = [c.key for c in out]
    assert keys == ["pixabay:1", "pixabay:2", "pexels:1"]


def test_search_all_threads_rendition_policy():
    p = _FakeProvider("pixabay", [_clip("pixabay", 1)])
    search_all([p], ["x"])                       # defaults
    assert p.kwargs["max_h"] == DEFAULT_MAX_H
    assert p.kwargs["size_cap"] == DEFAULT_SIZE_CAP
    search_all([p], ["x"], max_h=1080, size_cap=5)
    assert p.kwargs["max_h"] == 1080 and p.kwargs["size_cap"] == 5


# ---- Pexels per-term resilience (flaky-401 windows) ------------------------

class _FakeResp:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text
        self.headers = {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _FakeSession:
    """Serves one canned response per .get() call, in order."""
    def __init__(self, responses):
        self.responses = list(responses)

    def get(self, url, **kw):
        return self.responses.pop(0)


def _payload(video_id, total=1):
    return {"videos": [{"id": video_id, "duration": 30,
                        "video_files": [{"link": f"https://c/{video_id}.mp4",
                                          "width": 1920,
                                          "height": 1080}]}],
            "total_results": total}


_UNAUTH = _FakeResp(401, text='{"message":"Missing API key"}')


def test_pexels_search_survives_dead_term(monkeypatch):
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda s: None)
    # term 1: request_json burns all 4 attempts on 401s -> term caught;
    # term 2: healthy -> its clips come back anyway
    s = _FakeSession([_UNAUTH] * 4 + [_FakeResp(200, _payload(101))])
    out = Pexels("k", session=s).search(["dead term", "good term"],
                                        per_term=80, pages=1)
    assert [c.key for c in out] == ["pexels:101"]


def test_pexels_search_raises_only_when_all_terms_dead(monkeypatch):
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda s: None)
    s = _FakeSession([_UNAUTH] * 8)
    try:
        Pexels("k", session=s).search(["a", "b"], per_term=80, pages=1)
    except RuntimeError as e:
        assert "all terms failed" in str(e)
    else:
        raise AssertionError("expected RuntimeError")

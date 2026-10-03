"""bars_plan / detect_bars: baked-in black bar handling at normalize time.

Regression: cinematic stock clips ship as 16:9 files holding smaller
content boxes. Two shapes seen live (pixabay 162706/162713):
- static bars:   constant content box (e.g. 1920x662) -> pre-crop
- animated bars: starts boxed, later goes full-frame -> head-trim
The cover-crop chain alone is a no-op on already-16:9 files, so bars
shipped into rendered videos.
"""
from engine.normalize import bars_plan, detect_bars, normalize_cmd

FULL = (1920, 1080, 0, 0)


def _fake(monkeypatch, dur=20.0, crops=None, rule=None):
    """Rule crops samples: fixed list of crops, or a function(t)->crop."""
    monkeypatch.setattr("engine.normalize.ffprobe_duration", lambda p: dur)
    seq = iter(crops * 4) if crops is not None else None

    def probe(path, t, d):
        if rule is not None:
            return rule(t)
        return next(seq)

    monkeypatch.setattr("engine.normalize._probe_crop", probe)


def test_static_uniform_bars_detected(monkeypatch):
    _fake(monkeypatch, crops=[(1920, 662, 0, 0)] * 5)
    p = bars_plan("x.mp4")
    assert p["kind"] == "static" and p["crop"] == (1920, 662, 0, 0)
    assert detect_bars("x.mp4") == (1920, 662, 0, 0)


def test_pillarbox_detected(monkeypatch):
    _fake(monkeypatch, crops=[(1440, 1080, 240, 0)] * 5)
    assert detect_bars("x.mp4") == (1440, 1080, 240, 0)


def test_full_frame_none(monkeypatch):
    _fake(monkeypatch, crops=[FULL] * 5)
    assert bars_plan("x.mp4")["kind"] == "none"


def test_varying_boxed_content_not_trusted(monkeypatch):
    # no sample full-frame AND boxes differ (montage of bar heights)
    _fake(monkeypatch, crops=[(1920, 662, 0, 0), (1920, 900, 0, 0),
                              (1920, 700, 0, 0), (1920, 990, 0, 0),
                              (1920, 800, 0, 0)])
    assert bars_plan("x.mp4")["kind"] == "none"
    assert detect_bars("x.mp4") is None


def test_slivers_ignored(monkeypatch):
    _fake(monkeypatch, crops=[(1912, 1076, 4, 2)] * 5)
    assert bars_plan("x.mp4")["kind"] == "none"


def test_dynamic_bars_get_trim(monkeypatch):
    _fake(monkeypatch, dur=53.0,
          rule=lambda t: FULL if t >= 13.5 else (1920, 880, 0, 0))
    p = bars_plan("x.mp4")
    assert p["kind"] == "dynamic"
    assert 13.0 <= p["trim_s"] <= 14.5
    assert detect_bars("x.mp4") is None      # static-only helper


def test_bars_returning_near_end_untouched(monkeypatch):
    # only the very last sample full-frame (bars reappear late) -> the
    # "last two samples must be clean" guard leaves it alone
    _fake(monkeypatch, dur=53.0,
          rule=lambda t: FULL if t >= 45.0 else (1920, 880, 0, 0))
    assert bars_plan("x.mp4")["kind"] == "none"


def test_short_clip_none(monkeypatch):
    _fake(monkeypatch, dur=1.5, crops=[(1920, 662, 0, 0)] * 5)
    assert bars_plan("x.mp4")["kind"] == "none"


def test_pre_crop_and_start_s_in_command():
    cmd = normalize_cmd("in.mp4", "out.mp4",
                        pre_crop=(1920, 662, 0, 0), start_s=13.8)
    vf = cmd[cmd.index("-vf") + 1]
    assert vf.startswith("crop=1920:662:0:0,fps=30,")
    i = cmd.index("-i")
    assert cmd[i - 2:i] == ["-ss", "13.8"]

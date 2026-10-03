"""normalize_all periodic index save — a watchdog kill mid-wave must not
discard every index update for the wave (grove30 2026-09-17: repeated kills
during a 159-clip resume repair re-did the whole pass each time because
save_index only ran at the end)."""
from pathlib import Path

from engine import normalize as nz


class _Clip:
    def __init__(self, key, provider, cid):
        self.key = key
        self.provider = provider
        self.id = cid
        self.duration_s = 20.0
        self.term = "nature"


def test_index_saved_every_10_clips(tmp_path, monkeypatch):
    clips = [_Clip(f"pexels:{i}", "pexels", str(i)) for i in range(25)]
    saves = []
    monkeypatch.setattr(nz, "load_index", lambda cfg: {})
    monkeypatch.setattr(nz, "save_index",
                        lambda cfg, idx: saves.append(dict(idx)))
    monkeypatch.setattr(nz, "raw_path",
                        lambda cfg, c: tmp_path / f"raw_{c.id}.mp4")
    out = tmp_path / "norm"

    def fake_norm(cfg, clip, **k):
        return out / f"norm_{clip.id}.mp4"

    monkeypatch.setattr(nz, "normalize_one", fake_norm)
    monkeypatch.setattr(nz, "ffprobe_duration", lambda p: 20.0)
    nz.normalize_all(None, clips)
    # 25 clips -> saves at 10, 20, and the final flush
    assert [len(s) for s in saves] == [10, 20, 25]


def test_killed_wave_keeps_progress_in_index(tmp_path, monkeypatch):
    """A kill after 12 clips must leave 10 persisted (the last checkpoint),
    so the next resume skips them instead of re-normalizing."""
    clips = [_Clip(f"pexels:{i}", "pexels", str(i)) for i in range(25)]
    saved = {}

    def save(cfg, idx):
        saved.clear()
        saved.update(idx)

    monkeypatch.setattr(nz, "load_index", lambda cfg: {})
    monkeypatch.setattr(nz, "save_index", save)
    monkeypatch.setattr(nz, "raw_path",
                        lambda cfg, c: tmp_path / f"raw_{c.id}.mp4")
    out = tmp_path / "norm"
    n = 0

    class _Killed(BaseException):
        """Process death (watchdog kill) — not caught by `except Exception`."""

    def fake_norm(cfg, clip, **k):
        nonlocal n
        n += 1
        if n == 13:
            raise _Killed()          # watchdog kill mid-wave
        return out / f"norm_{clip.id}.mp4"

    monkeypatch.setattr(nz, "normalize_one", fake_norm)
    monkeypatch.setattr(nz, "ffprobe_duration", lambda p: 20.0)
    try:
        nz.normalize_all(None, clips)
    except _Killed:
        pass
    assert len(saved) == 10          # checkpoint at 10 survived the kill
    assert all(v.get("norm") for v in saved.values())


def test_cached_norms_never_re_saved(tmp_path, monkeypatch):
    """Resume over already-indexed norms: no normalize_one, one final save."""
    norm = tmp_path / "norm_1.mp4"
    norm.write_bytes(b"x")
    clip = _Clip("pexels:1", "pexels", "1")
    idx = {"pexels:1": {"term": "nature", "norm": str(norm),
                        "duration_s": 20.0}}
    saves = []
    monkeypatch.setattr(nz, "load_index", lambda cfg: idx)
    monkeypatch.setattr(nz, "save_index",
                        lambda cfg, i: saves.append(i))
    monkeypatch.setattr(nz, "video_problems", lambda p: [])
    monkeypatch.setattr(nz, "normalize_one",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("cached norm must not re-normalize")))
    out = nz.normalize_all(None, [clip])
    assert out == {"pexels:1": str(norm)}
    assert len(saves) == 1           # only the final flush

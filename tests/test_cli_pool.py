"""_pool_from_index term filtering — the bug: a new job's --mood terms were
ignored because coverage counted the ENTIRE global library (leftover clips
from previous jobs), so no search ran and scenes were built from stale
moods (observed live 2026-09-04 with nature terms vs fog-forest clips)."""
from pathlib import Path

from engine import cli


def _entry(tmp_path, key, term, dur=20.0):
    p = tmp_path / (key.replace(":", "_") + ".mp4")
    p.write_bytes(b"x")
    return {"term": term, "norm": str(p), "duration_s": dur}


def test_pool_filters_to_matching_terms(tmp_path, monkeypatch):
    idx = {
        "pixabay:1": _entry(tmp_path, "pixabay:1", "fog forest morning"),
        "pixabay:2": _entry(tmp_path, "pixabay:2", "nature"),
        "pexels:3": _entry(tmp_path, "pexels:3", "Nature"),
    }
    monkeypatch.setattr(cli, "load_index", lambda cfg: idx)
    out = cli._pool_from_index(None, terms={"nature"})
    assert [p["key"] for p in out] == ["pexels:3", "pixabay:2"]  # sorted, case-insensitive


def test_pool_unfiltered_when_no_terms(tmp_path, monkeypatch):
    idx = {"pixabay:1": _entry(tmp_path, "pixabay:1", "anything")}
    monkeypatch.setattr(cli, "load_index", lambda cfg: idx)
    assert len(cli._pool_from_index(None)) == 1


def test_pool_drops_missing_files_and_short_clips(tmp_path, monkeypatch):
    idx = {
        "pixabay:2": _entry(tmp_path, "pixabay:2", "nature", dur=2.0),  # < POOL_MIN_CLIP_S
        "pixabay:3": {"term": "nature", "norm": str(tmp_path / "gone.mp4"),
                      "duration_s": 30.0},                               # missing file
        "pixabay:4": _entry(tmp_path, "pixabay:4", "nature"),
    }
    monkeypatch.setattr(cli, "load_index", lambda cfg: idx)
    out = cli._pool_from_index(None, terms={"nature"})
    assert [p["key"] for p in out] == ["pixabay:4"]


def test_pool_skips_blocklisted_clips(tmp_path, monkeypatch):
    """Landscape-only rule: content-QC blocklist bans a clip from every
    pool even though its normalized file exists on disk."""
    idx = {
        "pixabay:2": _entry(tmp_path, "pixabay:2", "nature"),
        "pixabay:3": _entry(tmp_path, "pixabay:3", "nature"),
    }
    monkeypatch.setattr(cli, "load_index", lambda cfg: idx)
    monkeypatch.setattr(cli, "load_blocklist",
                        lambda cfg: {"pixabay:3": "urban aerial"})
    out = cli._pool_from_index(None, terms={"nature"})
    assert [p["key"] for p in out] == ["pixabay:2"]


def test_ensure_pool_shortfall_never_backfills_other_moods(tmp_path, monkeypatch):
    """Fresh-clips-only: a thin mood must fail loudly, not inherit other
    jobs' cached clips (water30 got 48 of bus30's delivered clips via the
    old global-library fill, 2026-09-15)."""
    import pytest
    from types import SimpleNamespace

    idx = {"pixabay:9": _entry(tmp_path, "pixabay:9",
                               "some other job's term", dur=900.0)}
    monkeypatch.setattr(cli, "load_index", lambda cfg: idx)
    monkeypatch.setattr(cli, "providers_for", lambda cfg, log=None: [object()])
    monkeypatch.setattr(cli, "search_all", lambda *a, **k: [])

    class _Log:
        def info(self, *a):
            pass

        warn = info

    class _St:
        def mark_phase(self, *a, **k):
            raise AssertionError("shortfall must not pass the stock phase")

    args = SimpleNamespace(mood=["green valley mist"], dry_run=False,
                           nvenc=False, stock_max_h=720)
    with pytest.raises(SystemExit) as e:
        cli._ensure_pool(None, args, _St(), 300.0, _Log())
    assert "shortfall" in str(e.value)


def test_repair_unnormalizes_only_stranded_same_term_clips(tmp_path, monkeypatch):
    """Resume deadlock (grove30 2026-09-17): a killed run indexes downloaded
    raws before normalize; dedupe then counts them 'have' while the pool sees
    no norm. The repair pass must normalize exactly those — same-term, raw on
    disk, norm missing — and touch nothing else."""
    raw_ok = tmp_path / "stranded_raw.mp4"
    raw_ok.write_bytes(b"x")
    raw_other = tmp_path / "other_job_raw.mp4"
    raw_other.write_bytes(b"x")
    norm_done = tmp_path / "done_norm.mp4"
    norm_done.write_bytes(b"x")

    idx = {
        # stranded: right term, raw exists, norm path missing on disk
        "pexels:1": {"provider": "pexels", "id": "1", "term": "pine forest fog aerial",
                     "duration_s": 30.0, "width": 1920, "height": 1080,
                     "raw": str(raw_ok), "norm": str(tmp_path / "never_made.mp4")},
        # complete: norm exists -> skip
        "pexels:2": {"provider": "pexels", "id": "2", "term": "pine forest fog aerial",
                     "duration_s": 30.0, "width": 1920, "height": 1080,
                     "raw": str(norm_done), "norm": str(norm_done)},
        # other job's term (deleted-era footage stays unrecoverable) -> skip
        "pexels:3": {"provider": "pexels", "id": "3", "term": "old job mood",
                     "duration_s": 30.0, "width": 1920, "height": 1080,
                     "raw": str(raw_other), "norm": str(tmp_path / "nope.mp4")},
        # blocklisted -> skip
        "pexels:4": {"provider": "pexels", "id": "4", "term": "pine forest fog aerial",
                     "duration_s": 30.0, "width": 1920, "height": 1080,
                     "raw": str(raw_other), "norm": str(tmp_path / "nope2.mp4")},
        # raw gone (user wipe) -> unrecoverable, skip
        "pexels:5": {"provider": "pexels", "id": "5", "term": "pine forest fog aerial",
                     "duration_s": 30.0, "width": 1920, "height": 1080,
                     "raw": str(tmp_path / "deleted.mp4"), "norm": None},
    }
    monkeypatch.setattr(cli, "load_index", lambda cfg: idx)
    monkeypatch.setattr(cli, "load_blocklist", lambda cfg: {"pexels:4": "junk"})

    got = []
    monkeypatch.setattr(cli, "normalize_all",
                        lambda cfg, clips, **k: got.extend(clips))
    cli._repair_unnormalized(None, {"pine forest fog aerial"}, None)

    assert [(c.provider, c.id) for c in got] == [("pexels", "1")]
    assert got[0].term == "pine forest fog aerial"
    assert got[0].duration_s == 30.0 and got[0].width == 1920


def test_repair_unnormalized_noop_when_nothing_stranded(tmp_path, monkeypatch):
    """Clean resume must not call normalize_all at all."""
    p = _entry(tmp_path, "pixabay:1", "nature")
    idx = {"pixabay:1": dict(p, provider="pixabay", id="1",
                             raw=str(tmp_path / "raw.mp4"))}
    monkeypatch.setattr(cli, "load_index", lambda cfg: idx)
    monkeypatch.setattr(cli, "load_blocklist", lambda cfg: {})
    monkeypatch.setattr(cli, "normalize_all",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("nothing stranded, must not normalize")))
    cli._repair_unnormalized(None, {"nature"}, None)  # no raise


# ---- persona collection (the silent-skip bug: empty dir rendered no
# figure at all instead of failing) ----------------------------------------

def test_collect_personas_errors_on_empty_dir(tmp_path):
    import pytest
    with pytest.raises(SystemExit) as e:
        cli._collect_personas([str(tmp_path)])
    assert "no images" in str(e.value)


def test_collect_personas_accepts_jpg_and_png(tmp_path):
    (tmp_path / "grandpa.png").write_bytes(b"x")
    (tmp_path / "new.jpg").write_bytes(b"x")
    out = cli._collect_personas([str(tmp_path)])
    assert [Path(p).name for p in out] == ["grandpa.png", "new.jpg"]

"""save_index must merge-write. A stale in-memory copy -- a second build
process orphaned by a session drop, or one loaded before a concurrent
writer saved -- must not orphan entries that reached disk after it was
loaded. water30 lost 92 fresh clips exactly this way (2026-09-15), then
silently fell back to other jobs' cached footage."""
from engine import downloader


class _Cfg:
    """Minimal stand-in: save/load_index only touch library_index_path."""

    def __init__(self, path):
        self.library_index_path = path


def test_save_index_keeps_disk_only_entries(tmp_path):
    cfg = _Cfg(tmp_path / "library_index.json")
    downloader.save_index(cfg, {"pixabay:1": {"term": "a"}})
    # process B loaded the index BEFORE pixabay:1 was saved, now saves its
    # own (stale) view with a different key
    downloader.save_index(cfg, {"pixabay:2": {"term": "b"}})
    assert set(downloader.load_index(cfg)) == {"pixabay:1", "pixabay:2"}


def test_save_index_memory_wins_for_shared_keys(tmp_path):
    cfg = _Cfg(tmp_path / "library_index.json")
    downloader.save_index(cfg, {"pixabay:1": {"term": "old"}})
    downloader.save_index(cfg, {"pixabay:1": {"term": "new"}})
    assert downloader.load_index(cfg)["pixabay:1"]["term"] == "new"

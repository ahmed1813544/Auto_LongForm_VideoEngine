"""Edge TTS client: rate mapping, cache keys, mp3->pcm decode, batch order,
retry/backoff — all offline.  Real synthesis is monkeypatched; the ffmpeg
decode test builds a sine MP3 locally (no network).
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from engine import edge_tts as et
from engine.config import SAMPLE_RATE, Config


def make_cfg(tmp_path) -> Config:
    return Config(pixabay_key=None, pexels_key=None, work_dir=tmp_path)


class FakeLog:
    def __init__(self):
        self.warnings = []

    def info(self, msg):
        pass

    def warn(self, msg):
        self.warnings.append(msg)


def test_speed_to_rate():
    assert et.speed_to_rate(1.0) == "+0%"
    assert et.speed_to_rate(1) == "+0%"
    assert et.speed_to_rate(0.9) == "-10%"
    assert et.speed_to_rate(1.25) == "+25%"
    assert et.speed_to_rate(2.0) == "+100%"


def test_cache_sha_props():
    k = et.cache_sha("text", "en-US-ChristopherNeural", 1.0)
    # canonical numerics: 1, 1.0 and "1" share one key
    assert k == et.cache_sha("text", "en-US-ChristopherNeural", 1)
    assert k == et.cache_sha("text", "en-US-ChristopherNeural", "1")
    # every input dimension moves the key
    assert k != et.cache_sha("other", "en-US-ChristopherNeural", 1.0)
    assert k != et.cache_sha("text", "en-US-GuyNeural", 1.0)
    assert k != et.cache_sha("text", "en-US-ChristopherNeural", 1.1)


def test_cache_sha_namespace_distinct_from_deepgram_era():
    """Old deepgram caches must never be picked up as edge audio."""
    from engine.util import sha256_text
    old = sha256_text("text\x1fen-US-ChristopherNeural\x1f1\x1f0\x1f"
                      f"{SAMPLE_RATE}linear16raw")
    assert et.cache_sha("text", "en-US-ChristopherNeural", 1.0) != old


def test_cache_path_layout(tmp_path):
    cfg = make_cfg(tmp_path)
    p = et.cache_path(cfg, "hello", et.DEFAULT_VOICE, 1.0)
    key = et.cache_sha("hello", et.DEFAULT_VOICE, 1.0)
    assert p == cfg.tts_cache_dir / key[:2] / (key + ".pcm")


def test_ensure_pcm_synthesizes_then_caches(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    calls = {"n": 0}

    def fake_synth(text, voice, *, speed=et.DEFAULT_SPEED, attempts=6,
                   log=None):
        calls["n"] += 1
        return b"\x01\x02" * 100

    monkeypatch.setattr(et, "synthesize_bytes", fake_synth)
    p1, n1, cached1 = et.ensure_pcm(cfg, "hello", et.DEFAULT_VOICE,
                                    log=FakeLog())
    p2, n2, cached2 = et.ensure_pcm(cfg, "hello", et.DEFAULT_VOICE,
                                    log=FakeLog())
    assert calls["n"] == 1          # second call served from cache
    assert not cached1 and cached2
    assert p1 == p2 and n1 == n2 == 200


def test_synthesize_all_keeps_input_order(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    texts = [f"sentence number {i}" for i in range(7)]
    # distinct byte-length per text proves ordering survives the pool
    monkeypatch.setattr(
        et, "synthesize_bytes",
        lambda text, voice, **kw: b"\x00" * (20 * (int(text[-1]) + 1)))
    res = et.synthesize_all(cfg, texts, et.DEFAULT_VOICE, FakeLog(),
                            workers=3)
    assert [n for _, n in res] == [20 * (i + 1) for i in range(7)]


def test_synthesize_bytes_retries_transient_then_succeeds(monkeypatch):
    hits = {"n": 0}
    sleeps = []

    def flaky_synth(text, voice, rate, timeout=120.0):
        hits["n"] += 1
        if hits["n"] < 3:
            raise OSError("connection reset by peer")
        return b"fake-mp3"

    log = FakeLog()
    monkeypatch.setattr(et, "_synth_mp3", flaky_synth)
    monkeypatch.setattr(et, "_mp3_to_pcm", lambda mp3: b"\x01\x02" * 50)
    monkeypatch.setattr(et.time, "sleep", sleeps.append)
    monkeypatch.setattr(et, "known_voices", lambda refresh=False: [])
    pcm = et.synthesize_bytes("hello", "en-US-TestNeural", log=log)
    assert pcm == b"\x01\x02" * 50
    assert hits["n"] == 3 and len(sleeps) == 2
    assert len(log.warnings) == 2   # both transients were logged


def test_synthesize_bytes_gives_up_after_attempts(monkeypatch):
    monkeypatch.setattr(et, "known_voices", lambda refresh=False: [])
    monkeypatch.setattr(et.time, "sleep", lambda s: None)

    def always_fail(text, voice, rate, timeout=120.0):
        raise OSError("endpoint down")

    monkeypatch.setattr(et, "_synth_mp3", always_fail)
    with pytest.raises(et.EdgeTTSError, match="after 2 attempts"):
        et.synthesize_bytes("hello", "en-US-TestNeural", attempts=2)


def test_unknown_voice_fails_with_hint(monkeypatch):
    monkeypatch.setattr(et, "known_voices", lambda refresh=False: [
        {"ShortName": "en-US-ChristopherNeural"},
        {"ShortName": "en-US-GuyNeural"},
        {"ShortName": "de-DE-KatjaNeural"},
    ])
    with pytest.raises(et.EdgeTTSError, match="flux-cliff-en"):
        et.synthesize_bytes("hello", "flux-cliff-en")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg needed")
def test_mp3_to_pcm_roundtrip_real_ffmpeg():
    """Local sine -> mp3 (24k mono) -> raw s16le pcm; ~1 s of audio."""
    mp3 = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-ar", "24000", "-ac", "1", "-b:a", "48k", "-f", "mp3", "pipe:1"],
        stdout=subprocess.PIPE, check=True).stdout
    pcm = et._mp3_to_pcm(mp3)
    dur_s = len(pcm) / (SAMPLE_RATE * 2)
    assert 0.9 < dur_s < 1.2
    assert len(pcm) % 2 == 0


def test_mp3_to_pcm_raises_on_garbage():
    with pytest.raises(et.EdgeTTSError, match="mp3->pcm failed"):
        et._mp3_to_pcm(b"not an mp3 at all")

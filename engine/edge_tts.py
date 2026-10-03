"""Microsoft Edge TTS client with content-addressed cache + backoff pool.

Replaces Deepgram (2026-10): edge-tts synthesizes through Microsoft's
Edge read-aloud service — free, no API key.  Output is 24 kHz mono MP3,
decoded locally (ffmpeg pipe) to RAW PCM s16le @24kHz, so the whole
downstream contract is unchanged: duration is pure byte math (48 B/ms)
and the .pcm cache doubles as the timeline's source of truth.

Trade-offs vs a paid API: the endpoint is unofficial (when Microsoft
changes it, `pip install -U edge-tts` is the fix), and big batches can
throttle — so the worker pool is smaller and every transient failure
(websocket reset, DNS, 429-ish silence) is retried with backoff rather
than classified.

Note: `import edge_tts` below resolves to the site-packages package
(absolute import), not this module.
"""
from __future__ import annotations

import asyncio
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import edge_tts

from engine.config import Config, SAMPLE_RATE
from engine.util import sha256_text

DEFAULT_VOICE = "en-US-ChristopherNeural"
DEFAULT_SPEED = 1.0
DEFAULT_WORKERS = 3      # free endpoint: stay politer than Deepgram's 4

# aiohttp (edge-tts's transport) is happiest on the selector loop on
# Windows; the CLI uses asyncio nowhere else, so a process-wide policy
# is safe.
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(
            asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:  # noqa: BLE001
        pass


class EdgeTTSError(RuntimeError):
    pass


def speed_to_rate(speed: float) -> str:
    """Engine --speed multiplier -> edge-tts rate string ('+0%', '-10%')."""
    return f"{round((float(speed) - 1.0) * 100):+d}%"


def _canon(v) -> str:
    """Canonical numeric form so 1, 1.0 and "1" share one cache key."""
    return f"{float(v):g}"


def cache_sha(cleaned_text: str, voice: str,
              speed: float = DEFAULT_SPEED) -> str:
    """The ONE cache-key definition — timeline/chunks.json must import
    this rather than re-spelling the hash (they once drifted silently).
    The namespace differs from the old Deepgram keys on purpose: identical
    text must never resolve to another engine's audio."""
    return sha256_text(
        f"{cleaned_text}\x1f{voice}\x1f{_canon(speed)}\x1f"
        f"{SAMPLE_RATE}s16le-edge")


def cache_path(cfg: Config, cleaned_text: str, voice: str,
               speed: float = DEFAULT_SPEED) -> Path:
    key = cache_sha(cleaned_text, voice, speed)
    return cfg.tts_cache_dir / key[:2] / (key + ".pcm")


def _mp3_to_pcm(mp3: bytes) -> bytes:
    """Decode edge-tts's 24 kHz mono MP3 to raw s16le PCM via ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", "pipe:0", "-f", "s16le", "-acodec", "pcm_s16le",
         "-ar", str(SAMPLE_RATE), "-ac", "1", "pipe:1"],
        input=mp3, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        tail = (proc.stderr or b"")[-400:].decode("utf-8", "replace")
        raise EdgeTTSError(f"ffmpeg mp3->pcm failed: {tail}")
    return proc.stdout


async def _collect_mp3(text: str, voice: str, rate: str) -> bytes:
    audio = bytearray()
    async for chunk in edge_tts.Communicate(text, voice, rate=rate).stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
    if not audio:
        raise edge_tts.exceptions.NoAudioReceived(
            f"no audio chunks for {text[:60]!r}")
    return bytes(audio)


def _synth_mp3(text: str, voice: str, rate: str,
               timeout: float = 120.0) -> bytes:
    return asyncio.run(
        asyncio.wait_for(_collect_mp3(text, voice, rate), timeout))


_voices_cache: list[dict] | None = None


def known_voices(refresh: bool = False) -> list[dict]:
    """Edge voice list (network; cached per process; [] if unavailable)."""
    global _voices_cache
    if _voices_cache is None or refresh:
        try:
            _voices_cache = asyncio.run(edge_tts.list_voices())
        except Exception:  # noqa: BLE001  # offline: never block synthesis
            _voices_cache = []
    return _voices_cache


def _check_voice(voice: str) -> None:
    voices = known_voices()
    if not voices:
        return  # list unavailable — let the synthesis itself report errors
    names = {v["ShortName"] for v in voices}
    if voice in names:
        return
    hints = sorted(n for n in names if n.startswith("en-"))
    raise EdgeTTSError(
        f"unknown voice {voice!r} — list valid names with "
        f"`python -m engine tts-voices`; English voices: {', '.join(hints)}")


def synthesize_bytes(text: str, voice: str, *,
                     speed: float = DEFAULT_SPEED,
                     attempts: int = 6, log=None) -> bytes:
    """One synthesis (mp3 -> pcm) with exponential backoff on transient
    failures. Decode errors (EdgeTTSError) are deterministic — raised
    immediately, never retried."""
    _check_voice(voice)
    rate = speed_to_rate(speed)
    last_err = "unknown"
    for n in range(attempts):
        try:
            pcm = _mp3_to_pcm(_synth_mp3(text, voice, rate))
            if len(pcm) < 2:
                raise EdgeTTSError(f"empty PCM for {text[:60]!r}")
            if len(pcm) % 2:
                pcm += b"\x00"  # s16 alignment paranoia
            return pcm
        except EdgeTTSError:
            raise
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            if n == attempts - 1:
                break
            delay = min(60.0, 1.0 * (2 ** n)) * random.uniform(0.7, 1.3)
            if log:
                log.warn(f"edge-tts {last_err[:160]}; retry in {delay:.1f}s")
            time.sleep(delay)
    raise EdgeTTSError(
        f"edge-tts failed after {attempts} attempts ({last_err}) "
        f"for text={text[:60]!r}")


def ensure_pcm(cfg: Config, text: str, voice: str, *,
               speed: float = DEFAULT_SPEED,
               log=None) -> tuple[Path, int, bool]:
    """Return (path, bytes, was_cached); synthesize into cache if missing."""
    p = cache_path(cfg, text, voice, speed)
    if p.exists() and p.stat().st_size > 0:
        return p, p.stat().st_size, True
    data = synthesize_bytes(text, voice, speed=speed, log=log)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Unique .part name per writer: scripts reuse exact sentences (hook
    # anaphora), and two batch workers hitting the same cache key would
    # otherwise race on one fixed .part path (Windows sharing violation).
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident():x}.part")
    for attempt in range(3):
        try:
            tmp.write_bytes(data)
            break
        except PermissionError:  # transient AV/indexer lock on the new file
            if attempt == 2:
                raise
            time.sleep(0.5 * (attempt + 1))
    try:
        tmp.replace(p)  # atomic on same volume
    except OSError:
        # Destination already exists (a sibling worker/simultaneous job won
        # the race, or it is open for reading).  Cache is content-addressed
        # for identical text+params, so the winner's bytes are as good.
        if p.exists() and p.stat().st_size > 0:
            tmp.unlink(missing_ok=True)
            return p, p.stat().st_size, True
        raise
    return p, len(data), False


def synthesize_all(cfg: Config, texts: list[str], voice: str, log,
                   *, speed: float = DEFAULT_SPEED,
                   workers: int = DEFAULT_WORKERS) -> list[tuple[Path, int]]:
    """Parallel-but-polite batch synthesis. Returns (path, bytes) per input,
    in order. Raises on first permanent failure (cache makes retry cheap)."""
    results: list[tuple[Path, int] | None] = [None] * len(texts)
    done = {"n": 0, "cached": 0, "net": 0, "t0": time.time()}
    t_total = len(texts)

    def work(i_text):
        i, text = i_text
        path, nbytes, cached = ensure_pcm(cfg, text, voice, speed=speed,
                                          log=log)
        results[i] = (path, nbytes)
        done["n"] += 1
        done["cached" if cached else "net"] += 1
        if done["n"] % 50 == 0 or done["n"] == t_total:
            el = time.time() - done["t0"]
            rate = done["n"] / el if el > 0 else 0
            eta = (t_total - done["n"]) / rate if rate > 0 else 0
            log.info(f"tts {done['n']}/{t_total} "
                     f"(net {done['net']}, cached {done['cached']}) "
                     f"{rate:.1f}/s eta {eta/60:.1f}m")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, (i, t)) for i, t in enumerate(texts)]
        for f in futures:
            f.result()  # propagate first error
    return [r for r in results if r is not None]

"""Pass 1: script -> TTS units -> master.wav + captions.json + chunks.json.

Timeline math is milliseconds derived from PCM byte counts (48 bytes per ms
@ 24kHz s16 mono), kept as floats to avoid per-sentence truncation drift.
"""
from __future__ import annotations

import time
import wave
from pathlib import Path

from engine.chunker import build_units
from engine.config import BYTES_PER_SEC, Config, SAMPLE_RATE
from engine.edge_tts import DEFAULT_SPEED, cache_sha, synthesize_all
from engine.script_parser import Story, total_words
from engine.util import atomic_write_json, ffprobe_duration

SILENCE_SENT_MS = 200
SILENCE_STORY_MS = 1200
PREROLL_MS = 800
GROUP_MIN_MS = 900

B_PER_MS = BYTES_PER_SEC // 1000  # 48


def _dur_ms(nbytes: int) -> float:
    """Float ms — integer truncation per sentence would accumulate
    (up to ~1.5 s over 3,000 sentences) and drift captions off the audio."""
    return nbytes / B_PER_MS


def _split_groups(ms: float, weights: list[int]) -> list[float]:
    """Divide one sentence's duration into per-caption-group durations."""
    g = len(weights)
    if g <= 1:
        return [ms]
    mins = [min(GROUP_MIN_MS, ms // g)] * g
    floor_sum = sum(mins)
    if floor_sum >= ms:
        return [ms / g] * g
    tot = sum(weights) or 1
    rem = ms - floor_sum
    alloc = [mins[i] + (rem * w) / tot for i, w in enumerate(weights)]
    alloc[-1] += ms - sum(alloc)  # absorb rounding into the last group
    # monotonic repair: never let a group collapse below 300ms
    for i in range(len(alloc) - 1):
        if alloc[i] < 300:
            take = 300 - alloc[i]
            alloc[i] += take
            alloc[i + 1] -= take
    return alloc


def build_timeline(cfg: Config, job_dir: Path, stories: list[Story],
                   voice: str, log, *,
                   speed: float = DEFAULT_SPEED) -> dict:
    cfg.ensure_dirs()
    audio_dir = job_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    units = build_units(stories)
    if not units:
        raise SystemExit("script produced no sentences")
    log.info(f"{len(stories)} stories, {total_words(stories)} words, "
             f"{len(units)} sentences")

    texts = [u.text for u in units]
    t_start = time.time()
    results = synthesize_all(cfg, texts, voice, log, speed=speed)
    log.info(f"tts pass: {len(results)} units in {time.time() - t_start:.0f}s")

    # ---- timeline -------------------------------------------------------
    chunks, events = [], []
    ev_id = 0
    t_ms = PREROLL_MS
    pcm_frames = []  # (path, nbytes, silence_after) in speech order

    for i, u in enumerate(units):
        path, nbytes = results[i]
        dur_ms = _dur_ms(nbytes)
        sil = SILENCE_STORY_MS if u.ends_story else SILENCE_SENT_MS
        t0s, t1s = t_ms, t_ms + dur_ms
        chunks.append({
            "i": i, "story": u.story_idx, "sent": u.sent_idx,
            "cache_sha": cache_sha(u.text, voice, speed),
            "bytes": nbytes, "dur_ms": dur_ms, "silence_after_ms": sil,
            "t0_ms": t0s, "t1_ms": t1s, "text": u.text,
        })
        # caption groups inside the sentence
        gd = _split_groups(dur_ms, [len(g) for g in u.groups])
        gt = t0s
        for gi, (gtext, gms) in enumerate(zip(u.groups, gd)):
            events.append({
                "id": ev_id, "story": u.story_idx, "sentence": i,
                "group": gi, "text": gtext,
                "t0_ms": gt, "t1_ms": gt + gms,
                "story_end": bool(u.ends_story
                                  and gi == len(u.groups) - 1),
            })
            ev_id += 1
            gt += gms
        pcm_frames.append((path, nbytes, sil))
        t_ms = t1s + sil

    total_ms = t_ms

    # ---- master.wav (pure Python wave: byte-exact) ----------------------
    # The 0.8 s preroll is part of the timeline, so it must be part of the
    # audio too (caption t=0 sits inside it).
    master = audio_dir / "master.wav"
    with wave.open(str(master), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(b"\x00" * (PREROLL_MS * B_PER_MS))
        for path, nbytes, sil in pcm_frames:
            with open(path, "rb") as f:
                w.writeframes(f.read(nbytes))
            if sil:
                w.writeframes(b"\x00" * (sil * B_PER_MS))

    # ---- persist ---------------------------------------------------------
    summary = {
        "version": 1, "sample_rate": SAMPLE_RATE, "encoding": "s16le-mono",
        "voice": voice, "speed": speed,
        "total_ms": total_ms, "words": total_words(stories),
        "sentences": len(chunks), "caption_events": len(events),
        "chunks": chunks,
    }
    atomic_write_json(job_dir / "chunks.json", summary)
    atomic_write_json(job_dir / "captions.json", {
        "version": 1, "fps": 30, "total_ms": total_ms, "events": events,
    })

    probed = ffprobe_duration(master)
    delta = abs(probed - total_ms / 1000)
    if delta > 0.02:
        log.warn(f"master.wav duration delta {delta:.3f}s "
                 f"(ffprobe {probed:.3f} vs bytes {total_ms/1000:.3f})")
    h, rem = divmod(int(total_ms // 1000), 3600)
    m, s = divmod(rem, 60)
    log.info(f"master ready: {h}:{m:02d}:{s:02d} "
             f"({total_ms/1000/3600:.2f} h) - predicted final video length")

    return {"master": master, "total_ms": total_ms, "n_units": len(chunks),
            "n_events": len(events)}

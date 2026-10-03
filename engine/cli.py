"""CLI entry: `python -m engine build ...` runs the full two-pass pipeline.

Resume model: every phase records itself in <job>/state.json; rerunning the
same command continues where it left off.  A params change (voice, moods,
seed, ...) produces a different params_hash and refuses resume unless
--fresh wipes the job.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from pathlib import Path

from engine.ass_gen import write_ass_file
from engine.config import Config, PROJECT_ROOT, SAMPLE_RATE
from engine.edge_tts import DEFAULT_SPEED, DEFAULT_VOICE, ensure_pcm
from engine.doctor import cmd_doctor
from engine.downloader import ensure_clips, load_blocklist, load_index
from engine.logutil import Log
from engine.loudness import TARGET_LUFS, gain_db as compute_gain, measure_lufs
from engine.normalize import normalize_all
from engine.render.concat import build_final
from engine.render.ffmpeg_cmds import segment_cmd
from engine.render.segment import render_segment
from engine.scene_planner import (POOL_HEADROOM, assign_personas,
                                  plan_scenes, plan_segments, pool_coverage)
from engine.script_parser import parse_script
from engine.state import JobState, ParamsMismatch, params_hash
from engine.stock import DEFAULT_MAX_H, providers_for, search_all
from engine.stock.base import Clip
from engine.timeline import build_timeline
from engine.ui import cmd_ui
from engine.util import sha256_file
from engine.verify import verify_job

DEFAULT_TERMS = ["cinematic aerial nature", "rain window city night",
                 "calm ocean waves", "fireplace cozy room"]
POOL_BATCH = 80           # clips per download+normalize wave while topping up
POOL_MIN_CLIP_S = 4.0     # library clips shorter than this are ignored


# ---- helpers ---------------------------------------------------------------

def _read_stories(paths: list[Path], log: Log):
    stories = []
    for p in paths:
        st = parse_script(p)
        if not st:
            raise SystemExit(f"no stories found in {p}")
        log.info(f"{p.name}: {len(st)} stories, "
                 f"{sum(s.word_count for s in st)} words")
        stories += st
    return stories


def _collect_personas(dirs: list[str]) -> list[str]:
    files: list[str] = []
    for d in dirs:
        dp = Path(d)
        if not dp.is_dir():
            raise SystemExit(f"persona dir not found: {d}")
        found = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            found += sorted(dp.glob(ext))
        if not found:
            raise SystemExit(
                f"persona dir {d} has no images (png/jpg/webp) — add "
                f"transparent-background PNGs, or pass no --persona-dir "
                f"for stock-only video")
        files += [str(p) for p in found]
    return files


def _params_obj(cfg, args, stories, personas, script_paths) -> dict:
    obj = {
        "script_hash": [sha256_file(p) for p in script_paths],
        "n_stories": len(stories),
        "voice": args.voice,
        "speed": args.speed,
        "seg_max_s": args.seg_max_min * 60,
        "seed": args.seed,
        "nvenc": args.nvenc,
        "crf": args.crf,
        "preset": args.preset,
        "skip_stock": args.skip_stock,
        "moods": list(args.mood),
        "personas": personas,
        "work": str(cfg.work_dir),
        "gain_db": args.gain_db,
        "target_lufs": args.target_lufs,
    }
    # key rides the hash ONLY when set — jobs built before --eq existed keep
    # their params_hash and remain resumable (land30/story30 unaffected)
    if args.eq:
        obj["eq"] = True
    return obj


def _resolve_gain(cfg, st: JobState, args, master: Path,
                  log) -> float:
    """User override wins; else ebur128 once per job, cached in state."""
    if args.gain_db is not None:
        return args.gain_db
    cached = st.phase_output("loudness", "gain_db")
    if cached is not None:
        log.info(f"loudness cached: gain {cached:+.1f} dB")
        return float(cached)
    measured = st.phase_output("loudness", "measured_lufs")
    if measured is None:
        log.info("measuring master loudness (ebur128)")
        measured = measure_lufs(master)
    g = compute_gain(args.target_lufs, measured)
    st.mark_phase("loudness", measured_lufs=measured, gain_db=g,
                  target_lufs=args.target_lufs)
    log.info(f"loudness: master {measured:.1f} LUFS -> gain {g:+.1f} dB "
             f"(target {args.target_lufs:.0f})")
    return g


# ---- stock pool ------------------------------------------------------------

def _pool_from_index(cfg, terms: set[str] | None = None) -> list[dict]:
    """Normalized, existing clips straight from the global library index
    (shared across jobs).  When ``terms`` is given, keep only clips whose
    originating search term matches, so a job's moods actually shape its
    scenes instead of inheriting the previous job's pool."""
    pool = []
    blocked = load_blocklist(cfg)
    for k, e in sorted(load_index(cfg).items()):
        norm = e.get("norm")
        if not norm or not Path(norm).exists():
            continue
        if k in blocked:
            continue
        if terms is not None and str(e.get("term") or "").lower() not in terms:
            continue
        dur = float(e.get("norm_duration_s") or e.get("duration_s") or 0)
        if dur < POOL_MIN_CLIP_S:
            continue
        pool.append({"key": k, "path": norm, "duration_s": dur})
    return pool


def _repair_unnormalized(cfg, terms: set[str], log) -> None:
    """Resume repair for a killed normalize phase.

    A run killed between ``ensure_clips`` (which indexes downloads) and
    ``normalize_all`` leaves clips INDEXED with no norm file. Search dedupe
    then counts them as ``have`` forever while ``_pool_from_index`` cannot
    see them (no norm) — coverage deadlocks and the pool exits shortfall
    (grove30 2026-09-17: 79 stranded clips). Normalize them now, scoped to
    this job's moods so other jobs' deleted-era footage stays unrecoverable
    and the no-repeat rule holds."""
    blocked = load_blocklist(cfg)
    stranded = []
    for k, e in sorted(load_index(cfg).items()):
        if str(e.get("term") or "").lower() not in terms or k in blocked:
            continue
        raw, norm = e.get("raw"), e.get("norm")
        if not raw or not Path(raw).exists():
            continue
        if norm and Path(norm).exists():
            continue
        stranded.append(Clip(provider=e["provider"], id=e["id"],
                             duration_s=float(e.get("duration_s") or 0),
                             width=int(e.get("width") or 0),
                             height=int(e.get("height") or 0),
                             download_url="", term=e.get("term", "")))
    if not stranded:
        return
    if log:
        log.info(f"resume repair: normalizing {len(stranded)} indexed clips "
                 f"whose normalize phase was killed")
    normalize_all(cfg, stranded, log=log)


def _ensure_pool(cfg, args, st: JobState, total_s: float, log) -> list[dict]:
    """Grow the clip library until it covers POOL_HEADROOM x duration
    with clips matching this job's moods."""
    terms = args.mood or DEFAULT_TERMS
    tset = {t.lower() for t in terms}
    if not args.dry_run:
        _repair_unnormalized(cfg, tset, log)
    pool = _pool_from_index(cfg, terms=tset)
    cover, enough = pool_coverage(pool, total_s)
    if cover == 0:
        # Diagnostic: twice (grove30 --fresh runs, 2026-09-17) the in-process
        # pool read 0s while a separate process computed full coverage from
        # the same files. Log the evidence instead of silently searching.
        idx = load_index(cfg)
        live = [k for k, e in idx.items()
                if e.get("norm") and Path(e["norm"]).exists()
                and str(e.get("term") or "").lower() in tset]
        if log:
            log.warn(f"pool read 0s though index holds {len(live)} live "
                     f"norms for these terms (transient index read?)")
    if not args.dry_run:
        providers = providers_for(cfg, log=log)
        if not providers:
            raise SystemExit(
                "no stock API keys set — put PIXABAY_API_KEY and/or "
                "PEXELS_API_KEY in .env, or run with --skip-stock")
        batch_i = 0
        while not enough:
            log.info(f"pool {cover:.0f}s < {total_s * POOL_HEADROOM:.0f}s "
                     f"needed — searching stock (round {batch_i + 1})")
            clips = search_all(providers, terms, per_term=80,
                               pages=batch_i + 1,
                               max_h=args.stock_max_h, log=log)
            have = set(load_index(cfg)) | set(load_blocklist(cfg))
            fresh = [c for c in clips if c.key not in have]
            if not fresh:
                break
            # 1280-wide = the HD floor under the 720p-source default —
            # download the sharpest candidates first, longest first.
            fresh.sort(key=lambda c: (-(1 if c.width >= 1280 else 0),
                                      -(c.duration_s or 0)))
            # Download only as much as the remaining deficit needs (~15 s
            # per landscape clip), so a 5-min job isn't normalizing 80
            # clips when 25 cover it. Big jobs still move in 80-clip waves.
            i = 0
            while i < len(fresh):
                need = max(0.0, total_s * POOL_HEADROOM - cover)
                wave = max(8, min(POOL_BATCH, int(need / 15.0) + 6))
                chunk = fresh[i:i + wave]
                i += wave
                got = ensure_clips(cfg, chunk, log=log)
                normalize_all(cfg, [c for c in chunk if got.get(c.key)],
                              log=log, nvenc=args.nvenc)
                pool = _pool_from_index(cfg, terms=tset)
                cover, enough = pool_coverage(pool, total_s)
                log.info(f"pool: {cover:.0f}s covered of "
                         f"{total_s * POOL_HEADROOM:.0f}s needed")
                if enough:
                    break
            batch_i += 1
            if batch_i >= 5:
                break
    if not enough:
        # Fresh-clips-only rule: NEVER backfill from other jobs' cached
        # library. That fill is how water30 inherited 48 clips already
        # delivered inside bus30 (2026-09-15). A shortfall means today's
        # moods are too thin at the APIs; fail loudly and let the operator
        # add --mood terms and resume.
        raise SystemExit(
            f"clip pool shortfall: {cover:.0f}s of "
            f"{total_s * POOL_HEADROOM:.0f}s — the APIs served no more "
            f"fresh clips for these moods; add --mood terms and resume "
            f"(cached clips from other jobs are never reused)")
    st.mark_phase("stock", pool_clips=len(pool), cover_s=round(cover, 1))
    return pool


# ---- phases ------------------------------------------------------------------

def _plan_or_load(st: JobState, events: list[dict], total_s: float,
                  personas: list[str], pool: list[dict], args,
                  log) -> list[dict]:
    """Build the segment plan (or reload the stored one on resume)."""
    if st.segments():
        log.info(f"plan loaded: {len(st.segments())} segments")
        return st.segments()

    seg_max_s = args.seg_max_min * 60
    segments = plan_segments(events, total_s * 1000.0, seg_max_s)
    log.info(f"planned {len(segments)} segments "
             f"(<= {seg_max_s / 60:g} min, story-aligned cuts)")
    if not args.skip_stock:
        plan_scenes(segments, pool, seed=args.seed)

    pose = assign_personas(segments, personas) if personas \
        else [None] * len(segments)
    planned = []
    for seg, pp in zip(segments, pose):
        d = seg.to_dict()
        d["persona"] = pp
        planned.append(d)
    st.set_segments(planned)
    return planned


def _render_all(st: JobState, job_dir: Path, events: list[dict],
                args, gain: float, log) -> tuple[list[Path], bool]:
    """Render every pending segment.

    Returns (ordered paths, changed) where changed is True if any segment
    was (re)rendered this run — used to invalidate a cached concat.
    """
    master = Path(st.phase_output("timeline", "master"))
    demoted = st.demote_stale_done(log=log)
    if demoted:
        log.info(f"re-queued {demoted} stale segment(s)")
    fonts_dir = PROJECT_ROOT / "assets" / "fonts"
    seg_paths: list[Path] = []
    changed = bool(demoted)
    total = len(st.segments())
    for entry in st.segments():
        idx = entry["idx"]
        if entry.get("state") == "done" and entry.get("path") \
                and Path(entry["path"]).exists():
            seg_paths.append(Path(entry["path"]))
            continue
        t0, t1 = float(entry["t0_ms"]), float(entry["t1_ms"])
        dur = (t1 - t0) / 1000.0
        ass = job_dir / "ass" / f"seg{idx:04d}.ass"
        # ceil/floor stamps guarantee each caption lands in exactly one file
        # even when the boundary ms is fractional (byte-math durations).
        write_ass_file(ass, events, seg_t0_ms=math.ceil(t0),
                       seg_t1_ms=math.floor(t1))
        bg = [(s["path"], s["in_s"], s["dur_s"]) for s in entry["scenes"]]
        out = job_dir / "segments" / f"seg{idx:04d}.mp4"
        cmd = segment_cmd(
            master_wav=master, seg_t0=t0 / 1000.0, seg_dur=dur,
            bg_clips=bg, persona_img=entry.get("persona"),
            ass_path=ass,
            fonts_dir=fonts_dir if any(fonts_dir.glob("*.ttf")) else None,
            gain_db=gain, out=out,
            nvenc=args.nvenc, crf=args.crf, preset=args.preset,
            skip_stock_testsrc2=args.skip_stock, eq=args.eq)
        log.info(f"[{idx + 1}/{total}] segment {idx}: {dur / 60:.1f} min, "
                 f"{len(entry['scenes'])} scenes"
                 + (" (testsrc2 bg)" if args.skip_stock else ""))
        if args.dry_run:
            continue
        st.mark_segment(idx, "rendering", path=str(out))
        res = render_segment(cmd, out, dur, log)
        if not res["ok"]:
            st.mark_segment(idx, "failed", path=str(out),
                            problems=res["problems"])
            if res.get("stderr_tail"):
                print(res["stderr_tail"], file=sys.stderr)
            log.close()
            sys.exit(f"segment {idx} failed: {res['problems']} — "
                     f"fix the cause and rerun the same command (it resumes)")
        st.mark_segment(idx, "done", path=str(out))
        seg_paths.append(out)
        changed = True
    return seg_paths, changed


# ---- command -----------------------------------------------------------------

def cmd_build(args) -> int:
    cfg = Config.load()
    cfg.ensure_dirs()
    job_id = args.job_id or Path(args.script[0]).stem
    job_dir = cfg.jobs_dir / job_id
    for d in (job_dir / "audio", job_dir / "ass", job_dir / "segments",
              job_dir / "logs"):
        d.mkdir(parents=True, exist_ok=True)
    log = Log(job_dir / "logs" / "build.log")

    script_paths = [Path(p) for p in args.script]
    for p in script_paths:
        if not p.exists():
            raise SystemExit(f"script not found: {p}")
    stories = _read_stories(script_paths, log)
    personas = _collect_personas(args.persona_dir)
    if personas:
        log.info(f"personas: {[Path(p).name for p in personas]}")

    phash = params_hash(_params_obj(cfg, args, stories, personas,
                                    script_paths))
    if args.fresh:
        (job_dir / "state.json").unlink(missing_ok=True)
    try:
        st = JobState.load_or_new(job_dir, job_id=job_id, phash=phash)
    except ParamsMismatch as e:
        raise SystemExit(f"{e} (or pass --fresh to restart the job)")

    # ---- Pass 1: audio + captions --------------------------------------
    caps_p = job_dir / "captions.json"
    master_p = job_dir / "audio" / "master.wav"
    tl_ok = (st.phase_done("timeline")
             and Path(st.phase_output("timeline", "master")).exists()
             and caps_p.exists())
    if not tl_ok and master_p.exists() and caps_p.exists():
        # adopt pass-1 artifacts left by dev scripts / older runs
        st.mark_phase("timeline", master=str(master_p),
                      total_ms=json.loads(
                          caps_p.read_text(encoding="utf-8"))["total_ms"])
        tl_ok = True
    if not tl_ok:
        if args.dry_run:
            raise SystemExit("dry-run: pass 1 (TTS) has not run yet — "
                             "run once without --dry-run")
        log.info("phase: timeline (TTS + captions)")
        info = build_timeline(cfg, job_dir, stories, args.voice, log,
                              speed=args.speed)
        st.mark_phase("timeline", master=str(info["master"]),
                      total_ms=info["total_ms"])
    else:
        log.info("phase: timeline cached")
    caps = json.loads(caps_p.read_text(encoding="utf-8"))
    events, total_ms = caps["events"], float(caps["total_ms"])
    total_s = total_ms / 1000.0

    # ---- Pass 2: pool, plan, render, concat -----------------------------
    gain = _resolve_gain(cfg, st, args, Path(
        st.phase_output("timeline", "master")), log)
    pool = [] if args.skip_stock else _ensure_pool(cfg, args, st, total_s,
                                                   log)
    plan = _plan_or_load(st, events, total_s, personas, pool, args, log)
    if args.dry_run:
        for e in plan[:3]:
            log.info(f"  seg{e['idx']}: {e['dur_s'] / 60:.1f} min, "
                     f"{len(e['scenes'])} scenes, "
                     f"persona={Path(e['persona']).name if e.get('persona') else '-'}")
        if len(plan) > 3:
            log.info(f"  ... {len(plan) - 3} more")

    segs, segs_changed = _render_all(st, job_dir, events, args, gain, log)
    if args.dry_run:
        log.info(f"dry-run: {len(plan)} segments planned, nothing rendered")
        log.close()
        return 0

    out = Path(args.out) if args.out \
        else cfg.work_dir / "output" / f"{job_id}.mp4"
    concat_cached = (st.phase_done("concat")
                     and Path(st.phase_output("concat", "out")).exists()
                     and not segs_changed)
    if concat_cached:
        log.info(f"concat cached: {st.phase_output('concat', 'out')}")
    else:
        res = build_final(segs, out, log, planned_total_dur=total_s)
        if res["problems"]:
            log.warn("final concat: " + "; ".join(res["problems"]))
        st.mark_phase("concat", out=str(res["out"]),
                      duration=res["duration"])
        out = res["out"]
    log.info(f"DONE -> {out}  ({total_s / 3600:.2f} h)")
    log.close()
    return 0


def cmd_verify(args) -> int:
    cfg = Config.load()
    log = Log(quiet=False)
    job_dir = cfg.jobs_dir / args.job_id
    res = verify_job(job_dir, cfg, deep=args.deep, log=log)
    for p in res["problems"]:
        print("PROBLEM:", p)
    s = res["summary"]
    if s:
        print(f"segments {s.get('segments_done')}/{s.get('segments_planned')},"
              f" captions {s.get('ass_lines')}/{s.get('events')},"
              f" final {s.get('final')}")
    log.close()
    return 0 if res["ok"] else 1


def cmd_tts_preview(args) -> int:
    cfg = Config.load()
    cfg.ensure_dirs()
    from engine.tts_text import clean
    text = clean(args.text)
    log = Log()
    path, nbytes, cached = ensure_pcm(cfg, text, args.voice,
                                      speed=args.speed,
                                      log=log)
    out = Path(args.out) if args.out else cfg.work_dir / "tts_preview.wav"
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(path.read_bytes()[:nbytes])
    log.info(f"preview: {out} ({nbytes / 48000:.2f}s, voice={args.voice}, "
             f"speed={args.speed}, {'cached' if cached else 'new'})")
    log.close()
    return 0


def cmd_stock_search(args) -> int:
    cfg = Config.load()
    log = Log()
    providers = providers_for(cfg, log=log)
    if not providers:
        log.warn("no stock API keys set — add PIXABAY_API_KEY / "
                 "PEXELS_API_KEY to .env")
        return 1
    clips = search_all(providers, args.mood, per_term=args.n, pages=1,
                       max_h=args.stock_max_h, log=log)
    clips.sort(key=lambda c: (-(1 if c.width >= 1280 else 0),
                              -c.duration_s))
    log.info(f"{len(clips)} clips (top {args.n}):")
    for c in clips[:args.n]:
        log.info(f"  {c.key:22} {c.width}x{c.height} {c.duration_s:6.1f}s "
                 f"'{c.term}'")
    log.close()
    return 0


def cmd_tts_voices(args) -> int:
    """List Edge TTS voices (free endpoint — no API key involved)."""
    from engine.edge_tts import known_voices

    log = Log()
    voices = known_voices(refresh=True)
    if not voices:
        raise SystemExit("could not fetch the Edge voice list (network?)")
    en = [v for v in voices if v["ShortName"].startswith("en-")]
    rest = [v for v in voices if not v["ShortName"].startswith("en-")]
    log.info(f"{len(voices)} voices ({len(en)} English) — English first; "
             f"use a ShortName with --voice:")
    for v in en + rest:
        print(f"{v['ShortName']:40} {v.get('Gender', '?'):8} "
              f"{v.get('Locale', '')}")
    log.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="engine", description="revenge-story video engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="compile script -> final mp4")
    b.add_argument("--script", nargs="+", required=True,
                   help="one or more .txt story scripts")
    b.add_argument("--mood", nargs="+", default=[],
                   help="stock search terms (default: built-in moods)")
    b.add_argument("--persona-dir", nargs="*", default=[],
                   help="dirs of transparent PNGs to cycle per story")
    b.add_argument("--voice", default=DEFAULT_VOICE,
                   help="Edge TTS voice ShortName (default "
                        f"{DEFAULT_VOICE}); list with tts-voices")
    b.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                   help="TTS rate multiplier (default 1.0)")
    b.add_argument("--out", default=None,
                   help="final mp4 (default: <work>/output/<job-id>.mp4)")
    b.add_argument("--job-id", default=None,
                   help="resume key (default: first script's filename)")
    b.add_argument("--seg-max-min", type=float, default=9.0)
    b.add_argument("--seed", type=int, default=1234)
    b.add_argument("--stock-max-h", type=int, default=DEFAULT_MAX_H,
                   choices=(540, 720, 1080),
                   help="stock source height to prefer (default 720: the "
                        "smallest rendition that still looks right upscaled "
                        "to the 1080p deliverable — downloads and decodes "
                        "faster; files over ~80 MB are downgraded to a "
                        "smaller rendition. 1080 = old full-HD behavior)")
    b.add_argument("--crf", type=int, default=19)
    b.add_argument("--preset", default="veryfast")
    b.add_argument("--gain-db", type=float, default=None,
                   help="override auto loudness gain (dB)")
    b.add_argument("--target-lufs", type=float, default=TARGET_LUFS,
                   help=f"integrated voiceover target (default {TARGET_LUFS})")
    b.add_argument("--nvenc", action="store_true")
    b.add_argument("--eq", action="store_true",
                   help="audio-reactive equalizer bars behind the captions "
                        "(driven by the narration itself)")
    b.add_argument("--skip-stock", action="store_true",
                   help="testsrc2 background (dev / zero-key runs)")
    b.add_argument("--fresh", action="store_true",
                   help="wipe job state and rebuild everything")
    b.add_argument("--dry-run", action="store_true")
    b.set_defaults(func=cmd_build)

    d = sub.add_parser("doctor", help="environment + API health check")
    d.add_argument("--live", action="store_true",
                   help="make one cheap real call per configured API")
    d.add_argument("--persona-dir", nargs="*", default=[],
                   help="also validate these PNG dirs")
    d.set_defaults(func=cmd_doctor)

    v = sub.add_parser("verify", help="check a built job")
    v.add_argument("--job-id", required=True)
    v.add_argument("--deep", action="store_true",
                   help="per-segment decode scans (slow on long jobs)")
    v.set_defaults(func=cmd_verify)

    t = sub.add_parser("tts-preview", help="synthesize one line, save wav")
    t.add_argument("--text", required=True)
    t.add_argument("--voice", default=DEFAULT_VOICE)
    t.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    t.add_argument("--out", default=None, help="wav path (default <work>/tts_preview.wav)")
    t.set_defaults(func=cmd_tts_preview)

    tv = sub.add_parser("tts-voices",
                        help="list Edge TTS voices (free — no key needed)")
    tv.set_defaults(func=cmd_tts_voices)

    s = sub.add_parser("stock-search", help="probe stock search, list clips")
    s.add_argument("--mood", nargs="+", required=True)
    s.add_argument("--n", type=int, default=10)
    s.add_argument("--stock-max-h", type=int, default=DEFAULT_MAX_H,
                   choices=(540, 720, 1080),
                   help="rendition height the pickers should prefer")
    s.set_defaults(func=cmd_stock_search)

    u = sub.add_parser("ui", help="local web UI in your browser "
                                   "(zero extra deps, 127.0.0.1 only)")
    u.add_argument("--port", type=int, default=8000)
    u.add_argument("--no-browser", action="store_true",
                   help="don't auto-open the browser")
    u.set_defaults(func=cmd_ui)

    args = ap.parse_args(argv)
    return args.func(args)

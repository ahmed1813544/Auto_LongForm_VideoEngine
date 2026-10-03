"""`engine doctor` — pre-flight environment + API-shape health check.

Probes are meant to run BEFORE any expensive build: they catch a broken
ffmpeg build, missing keys, a dead disk, and — crucially — API response
shapes that drifted from what the adapters expect (fail loudly now, not
mid-render at minute 170).

`--live` makes real (cheap) calls: one Edge TTS synthesis (free, keyless)
and one search per configured stock provider, validating parse against
reality.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from engine.config import Config
from engine.util import ffprobe_json, run

MIN_FREE_GB = 60
REQUIRED_FILTERS = ("ass", "overlay", "concat", "fade", "scale", "crop",
                    "fps", "setsar", "format", "colorchannelmixer",
                    "ebur128")
REQUIRED_ENCODERS = ("libx264", "aac")


class Report:
    def __init__(self):
        self.lines: list[tuple[str, str]] = []

    def ok(self, label, detail=""):
        self.lines.append(("OK", f"{label}{' — ' + detail if detail else ''}"))

    def warn(self, label, detail=""):
        self.lines.append(("WARN",
                           f"{label}{' — ' + detail if detail else ''}"))

    def fail(self, label, detail=""):
        self.lines.append(("FAIL",
                           f"{label}{' — ' + detail if detail else ''}"))

    @property
    def n_fail(self) -> int:
        return sum(1 for s, _ in self.lines if s == "FAIL")

    @property
    def n_warn(self) -> int:
        return sum(1 for s, _ in self.lines if s == "WARN")

    def print(self, stream=sys.stdout):
        for status, msg in self.lines:
            print(f"[{status:4}] {msg}", file=stream)


def _check_ffmpeg(r: Report) -> None:
    try:
        p = run(["ffmpeg", "-hide_banner", "-version"], check=False)
    except FileNotFoundError:
        r.fail("ffmpeg", "not on PATH")
        return
    first = (p.stdout or "").splitlines()
    r.ok("ffmpeg", first[0].split("Copyright")[0].strip() if first else "?")
    try:
        run(["ffprobe", "-version"], check=True)
        r.ok("ffprobe", "on PATH")
    except Exception as e:  # noqa: BLE001
        r.fail("ffprobe", str(e))

    fl = run(["ffmpeg", "-hide_banner", "-filters"], check=False).stdout or ""
    missing = [f for f in REQUIRED_FILTERS if f" {f} " not in fl]
    if missing:
        r.fail("ffmpeg filters", "missing " + ", ".join(missing))
    else:
        r.ok("ffmpeg filters", f"{len(REQUIRED_FILTERS)} required present")

    enc = run(["ffmpeg", "-hide_banner", "-encoders"],
              check=False).stdout or ""
    miss_e = [e for e in REQUIRED_ENCODERS if e not in enc]
    if miss_e:
        r.fail("ffmpeg encoders", "missing " + ", ".join(miss_e))
    else:
        r.ok("ffmpeg encoders", ", ".join(REQUIRED_ENCODERS))
    if "h264_nvenc" in enc:
        r.ok("nvenc", "h264_nvenc available (optional --nvenc)")
    else:
        r.warn("nvenc", "h264_nvenc not found — --nvenc unavailable")


def _check_config(r: Report, cfg: Config) -> None:
    r.ok("work dir", str(cfg.work_dir))
    if cfg.work_dir.exists():
        free_gb = shutil.disk_usage(cfg.work_dir).free / 1e9
        (r.ok if free_gb >= MIN_FREE_GB else r.fail)(
            "disk free", f"{free_gb:.0f} GB (need >= {MIN_FREE_GB})")
    else:
        r.warn("work dir", "does not exist yet (created on first build)")
    if not cfg.pixabay_key and not cfg.pexels_key:
        r.warn("stock keys", "neither PIXABAY_API_KEY nor PEXELS_API_KEY set "
                             "(--skip-stock still works)")
    else:
        r.ok("stock keys",
             " + ".join(n for n, v in (("pixabay", cfg.pixabay_key),
                                       ("pexels", cfg.pexels_key)) if v))


def _check_fonts_personas(r: Report, persona_dirs: list[str]) -> None:
    fonts = Path(__file__).resolve().parent.parent / "assets" / "fonts"
    ttf = list(fonts.glob("*.ttf")) + list(fonts.glob("*.otf")) if \
        fonts.exists() else []
    if ttf:
        r.ok("fonts", f"{len(ttf)} bundled faces")
    else:
        r.warn("fonts", "none in assets/fonts — Arial Black system fallback "
                        "used")
    for d in persona_dirs:
        dp = Path(d)
        if not dp.is_dir():
            r.fail("persona dir", f"{d} not found")
            continue
        pngs = sorted(dp.glob("*.png"))
        if not pngs:
            r.warn("persona dir", f"{d} has no PNGs")
            continue
        bad = []
        for png in pngs:
            try:
                info = ffprobe_json(png, select="v:0", show_streams=True)
                pf = info["streams"][0].get("pix_fmt", "")
                if "a" not in pf:
                    bad.append(f"{png.name} ({pf}, likely no alpha)")
            except Exception as e:  # noqa: BLE001
                bad.append(f"{png.name}: {e}")
        (r.fail if bad else r.ok)("personas " + dp.name,
                                  f"{len(pngs)} PNGs"
                                  + (f"; suspicious: {bad}" if bad else ""))


def _probe_live(r: Report, cfg: Config) -> None:
    """One cheap real call per provider to validate adapter parse shapes."""
    try:
        from engine.edge_tts import DEFAULT_VOICE, synthesize_bytes
        body = synthesize_bytes("Doctor probe.", DEFAULT_VOICE, attempts=2)
        ms = len(body) / 48.0
        r.ok("edge-tts live",
             f"{DEFAULT_VOICE} -> {len(body)} B raw PCM ({ms:.0f} ms)")
    except Exception as e:  # noqa: BLE001
        r.fail("edge-tts live", str(e)[:200])

    if cfg.pixabay_key:
        try:
            from engine.stock.pixabay import Pixabay
            clips = Pixabay(cfg.pixabay_key).search(["forest"], per_term=5,
                                                    pages=1)
            r.ok("pixabay live", f"{len(clips)} clips parsed"
                                 + (f", best {max(c.width for c in clips)}px"
                                    if clips else ""))
        except Exception as e:  # noqa: BLE001
            r.fail("pixabay live", str(e)[:200])
    if cfg.pexels_key:
        try:
            from engine.stock.pexels import Pexels
            clips = Pexels(cfg.pexels_key).search(["ocean"], per_term=5,
                                                  pages=1)
            r.ok("pexels live", f"{len(clips)} clips parsed")
        except Exception as e:  # noqa: BLE001
            r.fail("pexels live", str(e)[:200])


def cmd_doctor(args) -> int:
    cfg = Config.load()
    r = Report()
    _check_ffmpeg(r)
    _check_config(r, cfg)
    _check_fonts_personas(r, args.persona_dir or [])
    if args.live:
        print("--- running live API probes (one cheap call each) ---")
        _probe_live(r, cfg)
    r.print()
    print(f"\n{r.n_fail} fail, {r.n_warn} warn")
    return 1 if r.n_fail else 0

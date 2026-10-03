"""Local web UI — `python -m engine ui` → http://127.0.0.1:8000

Zero-dependency (stdlib-only) wrapper around the CLI: paste a story,
pick moods/persona/voice, click Build, watch the live log and segment
progress, open the finished mp4.  Builds run as child processes of
`python -m engine build ...`, so caches, resume, blocklist and QC
behavior are identical to a CLI run — mix and match freely.

Security posture (this is a local tool, not a web service):
- binds to 127.0.0.1 only;
- .env values never leave the process (only "is a stock key set" booleans);
- media downloads are restricted to the work dir and to .mp4/.wav;
- persona dirs must live under <project>/assets;
- job control (stop/rerun) only touches processes this server started.
"""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from engine.config import PROJECT_ROOT, Config, SAMPLE_RATE
from engine.stock.base import DEFAULT_MAX_H
from engine.util import sha256_text

WPM_ESTIMATE = 165.0
JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
MEDIA_EXTS = {".mp4": "video/mp4", ".wav": "audio/wav"}
UPLOAD_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
PERSONA_MAX_BYTES = 32 * 1024 * 1024
STOCK_MAX_H_CHOICES = (540, 720, 1080)   # keep in sync with cli.py choices
STATIC_DIR = Path(__file__).parent / "ui_static"

FALLBACK_VOICES = ["en-US-ChristopherNeural", "en-US-GuyNeural",
                   "en-US-EricNeural", "en-GB-RyanNeural"]

_LOCK = threading.Lock()
PROCS: dict[str, subprocess.Popen] = {}   # job_id -> build child process

# Browsers abort media connections all the time (tab close, seek, pause,
# parallel Range probes).  On Windows that surfaces as WinError 10054/10053;
# it is never a server fault, so it must never print a traceback.
CONNECTION_ERRS = (ConnectionResetError, ConnectionAbortedError,
                   BrokenPipeError)


# ---- pure helpers (unit-tested in tests/test_ui.py) -------------------------

def estimate_minutes(words: int) -> float:
    """Rough narration length for the live word-count readout."""
    return words / WPM_ESTIMATE


def valid_job_id(job_id) -> bool:
    return bool(job_id) and bool(JOB_ID_RE.match(str(job_id)))


def build_argv(a: dict) -> list[str]:
    """UI form dict -> the exact CLI argv the child process will run."""
    if not a.get("script_path"):
        raise ValueError("script_path required")
    if not valid_job_id(a.get("job_id")):
        raise ValueError(f"bad job id: {a.get('job_id')!r}")
    argv = [sys.executable, "-m", "engine", "build",
            "--script", str(a["script_path"]),
            "--job-id", str(a["job_id"])]
    for m in a.get("moods") or []:
        m = str(m).strip()
        if m:
            argv += ["--mood", m]
    if a.get("persona_dir"):
        argv += ["--persona-dir", str(a["persona_dir"])]
    if a.get("voice"):
        argv += ["--voice", str(a["voice"])]
    try:
        sp = float(a.get("speed", 1.0))
    except (TypeError, ValueError):
        sp = 1.0
    if abs(sp - 1.0) > 1e-9:
        argv += ["--speed", f"{sp:g}"]
    if a.get("seg_max_min"):
        argv += ["--seg-max-min", str(a["seg_max_min"])]
    if a.get("seed") is not None:
        argv += ["--seed", str(a["seed"])]
    if a.get("stock_max_h") is not None:
        try:
            mh = int(a["stock_max_h"])
        except (TypeError, ValueError):
            raise ValueError(f"bad stock_max_h: {a['stock_max_h']!r}")
        if mh not in STOCK_MAX_H_CHOICES:
            raise ValueError(f"bad stock_max_h: {mh}")
        if mh != DEFAULT_MAX_H:      # default stays out of the argv
            argv += ["--stock-max-h", str(mh)]
    for flag, key in (("--eq", "eq"), ("--skip-stock", "skip_stock"),
                      ("--nvenc", "nvenc"), ("--fresh", "fresh"),
                      ("--dry-run", "dry_run")):
        if a.get(key):
            argv.append(flag)
    return argv


def summarize_state(d: dict) -> dict:
    """state.json dict -> progress summary for the UI."""
    phases = d.get("phases") or {}
    done_phases = [k for k, v in phases.items()
                   if isinstance(v, dict) and v.get("done")]
    segs = d.get("segments") or []
    counts: dict[str, int] = {}
    for s in segs:
        st = s.get("state", "pending")
        counts[st] = counts.get(st, 0) + 1
    total = len(segs)
    if "concat" in done_phases:
        progress = 1.0
    elif total:
        progress = counts.get("done", 0) / total
    else:
        progress = 0.0
    return {"phases_done": done_phases, "segments_total": total,
            "segments": counts, "progress": round(progress, 4)}


def safe_media_path(root: Path, rel: str) -> Path | None:
    """Resolve rel under root; block traversal; whitelist extensions."""
    try:
        p = (root / rel).resolve()
        p.relative_to(root.resolve())
    except (ValueError, OSError):
        return None
    if p.suffix.lower() not in MEDIA_EXTS:
        return None
    return p if p.is_file() else None


# ---- job/process plumbing ----------------------------------------------------

def console_path(cfg: Config, job_id: str) -> Path:
    return cfg.jobs_dir / job_id / "logs" / "console.log"


def proc_info(job_id: str) -> dict | None:
    with _LOCK:
        p = PROCS.get(job_id)
    if p is None:
        return None
    rc = p.poll()
    return {"running": rc is None, "returncode": rc}


def spawn_build(cfg: Config, argv: list[str], job_id: str) -> subprocess.Popen:
    job_dir = cfg.jobs_dir / job_id
    (job_dir / "logs").mkdir(parents=True, exist_ok=True)
    # The child's stdout+stderr (which mirror logs/build.log line-for-line,
    # plus any traceback) go to console.log; the UI tails that file.
    with open(console_path(cfg, job_id), "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            argv, cwd=str(PROJECT_ROOT), stdout=logf,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    with _LOCK:
        PROCS[job_id] = proc
    (job_dir / "ui_meta.json").write_text(
        json.dumps({"argv": argv, "started": time.time()}, indent=1),
        encoding="utf-8")
    return proc


def stop_job(job_id: str) -> bool:
    """Kill the build AND its ffmpeg children (taskkill /T) — an orphaned
    segment ffmpeg keeps writing the file with no state record otherwise."""
    with _LOCK:
        p = PROCS.get(job_id)
    if p is None or p.poll() is not None:
        return False
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                       capture_output=True)
    else:
        p.terminate()
    return True


def list_jobs(cfg: Config) -> list[dict]:
    out = []
    jd = cfg.jobs_dir
    if not jd.is_dir():
        return out
    for d in sorted(jd.iterdir()):
        sj = d / "state.json"
        if not sj.exists():
            continue
        try:
            data = json.loads(sj.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        info = proc_info(d.name) or {}
        mp4 = cfg.work_dir / "output" / f"{d.name}.mp4"
        out.append({
            "job_id": d.name, **summarize_state(data),
            "running": info.get("running", False),
            "returncode": info.get("returncode"),
            "updated": sj.stat().st_mtime,
            "mp4": ({"size": mp4.stat().st_size,
                     "url": f"/media/output/{d.name}.mp4"}
                    if mp4.is_file() else None),
            "ui": (d / "ui_meta.json").exists(),
        })
    out.sort(key=lambda j: j["updated"], reverse=True)
    return out


def scan_personas(cfg: Config) -> list[dict]:
    root = PROJECT_ROOT / "assets"
    out = []
    if root.is_dir():
        for d in sorted(root.iterdir()):
            if not d.is_dir() or d.name in ("fonts", "original_photos"):
                continue
            n = sum(len(list(d.glob(f"*{e}")))
                    for e in ("png", "jpg", "jpeg", "webp"))
            if n:
                out.append({"name": d.name, "path": str(d), "images": n,
                            "source": "assets"})
    up = cfg.work_dir / "ui_personas"
    if up.is_dir():
        for d in sorted(up.iterdir()):
            if d.is_dir() and (d / "storyteller.png").is_file():
                out.append({"name": f"{d.name} (yours)", "path": str(d),
                            "images": 1, "source": "uploaded"})
    return out


PERSONA_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")


def rembg_available() -> bool:
    """True if the optional rembg (AI background remover) is installed —
    checked without importing it (find_spec is cheap, rembg is not)."""
    try:
        return importlib.util.find_spec("rembg") is not None
    except (ImportError, ValueError):
        return False


def save_uploaded_persona(cfg: Config, name: str, data: bytes, *,
                          cut: bool) -> dict:
    """Validate -> (optional rembg cut) -> downscale/convert to PNG under
    <work>/ui_personas/<slug>/storyteller.png.  Never touches the repo."""
    slug = PERSONA_SLUG_RE.sub("-", (name or "").strip()).strip("-")[:40]
    if not slug:
        raise ValueError("give the photo a name (letters/digits)")
    if not data or len(data) > PERSONA_MAX_BYTES:
        raise ValueError("photo empty or larger than 32 MB")
    pdir = cfg.work_dir / "ui_personas" / slug
    pdir.mkdir(parents=True, exist_ok=True)
    out = pdir / "storyteller.png"
    src = pdir / "upload.bin"
    src.write_bytes(data)
    cut_done = False
    if cut:
        if not rembg_available():
            raise ValueError("rembg is not installed — run "
                             "`pip install rembg onnxruntime` or upload "
                             "without background removal")
        from rembg import remove  # lazy: pulls in onnxruntime (~200 MB RAM)
        cut_bytes = remove(data)
        src.write_bytes(cut_bytes)
        cut_done = True
    # ffmpeg: convert anything -> PNG, downscale to <=1600px wide, keep alpha
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(src), "-vf", r"scale='min(1600,iw)':-2",
           "-frames:v", "1", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    src.unlink(missing_ok=True)
    if proc.returncode != 0 or not out.is_file():
        out.unlink(missing_ok=True)
        raise ValueError("not a readable image (ffmpeg said: "
                         + (proc.stderr or "?").strip()[-200:] + ")")
    return {"name": f"{slug} (yours)", "path": str(pdir), "images": 1,
            "source": "uploaded", "cut": cut_done,
            "size": out.stat().st_size}


def ui_config(cfg: Config) -> dict:
    free = (shutil.disk_usage(cfg.work_dir).free / 1e9
            if cfg.work_dir.exists() else None)
    sample = ""
    sp = PROJECT_ROOT / "scripts" / "sample.txt"
    if sp.is_file():
        sample = sp.read_text(encoding="utf-8", errors="replace")[:8000]
    return {
        "work_dir": str(cfg.work_dir),
        "project_root": str(PROJECT_ROOT),
        "stock_keys": {"pixabay": bool(cfg.pixabay_key),
                       "pexels": bool(cfg.pexels_key)},
        "personas": scan_personas(cfg),
        "rembg": rembg_available(),
        "disk_free_gb": round(free, 1) if free else None,
        "sample_script": sample,
        "wpm": WPM_ESTIMATE,
    }


def voice_list() -> list[dict]:
    try:
        from engine.edge_tts import known_voices
        vs = known_voices()
    except Exception:  # noqa: BLE001
        vs = []
    if not vs:
        return [{"ShortName": n, "Gender": "?", "Locale": n[:5]}
                for n in FALLBACK_VOICES]
    en = [v for v in vs if v.get("ShortName", "").startswith("en-")] or vs
    return [{"ShortName": v["ShortName"], "Gender": v.get("Gender", "?"),
             "Locale": v.get("Locale", "")} for v in en]


def make_preview(cfg: Config, text: str, voice: str, speed) -> dict:
    import wave

    from engine.edge_tts import DEFAULT_SPEED, ensure_pcm
    from engine.tts_text import clean
    text = clean(text or "")[:400] or \
        "My name is Gerald. I am seventy-four years old."
    path, nbytes, _ = ensure_pcm(cfg, text, voice,
                                 speed=speed or DEFAULT_SPEED)
    d = cfg.work_dir / "ui_previews"
    d.mkdir(parents=True, exist_ok=True)
    name = sha256_text(f"{text}\x1f{voice}\x1f{speed}")[:12] + ".wav"
    out = d / name
    if not out.exists():
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(path.read_bytes()[:nbytes])
    return {"url": f"/media/preview/{name}",
            "seconds": round(nbytes / 48000.0, 2)}


def read_console(cfg: Config, job_id: str, offset: int) -> dict:
    p = console_path(cfg, job_id)
    if not p.exists():
        p = cfg.jobs_dir / job_id / "logs" / "build.log"  # CLI-started job
    info = proc_info(job_id) or {}
    if not p.exists():
        return {"text": "", "offset": offset, "size": 0,
                "running": info.get("running", False),
                "returncode": info.get("returncode")}
    size = p.stat().st_size
    if offset > size:
        offset = 0  # file was truncated by a rerun
    with open(p, "rb") as f:
        f.seek(offset)
        data = f.read(512 * 1024)
    return {"text": data.decode("utf-8", "replace"),
            "offset": offset + len(data), "size": size,
            "running": info.get("running", False),
            "returncode": info.get("returncode")}


# ---- HTTP server ----------------------------------------------------------------

class UIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, cfg: Config):
        self.cfg = cfg
        super().__init__(addr, Handler)

    def handle_error(self, request, client_address):
        """Silence client-hangup noise; real bugs still print."""
        if isinstance(sys.exc_info()[1], CONNECTION_ERRS):
            return
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    server_version = "VideoEngineUI/1"
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # keep the console quiet
        pass

    # ---- plumbing ----------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str,
              extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, default=str).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > 32 * 1024 * 1024:
            raise ValueError("bad content length")
        return json.loads(self.rfile.read(n).decode("utf-8"))

    # ---- GET -----------------------------------------------------------------
    def do_GET(self):
        u = urlparse(self.path)
        path, q = u.path, parse_qs(u.query)
        cfg = self.server.cfg
        try:
            if path in ("/", "/index.html"):
                html = (STATIC_DIR / "index.html").read_bytes()
                return self._send(200, html, "text/html; charset=utf-8")
            if path == "/api/config":
                return self._json(ui_config(cfg))
            if path == "/api/voices":
                return self._json(voice_list())
            if path == "/api/jobs":
                return self._json(list_jobs(cfg))
            if path == "/api/log":
                job = (q.get("job") or [""])[0]
                if not valid_job_id(job):
                    return self._json({"error": "bad job id"}, 400)
                offset = int((q.get("offset") or ["0"])[0])
                return self._json(read_console(cfg, job, max(0, offset)))
            if path.startswith("/media/"):
                return self._media(cfg, path[len("/media/"):])
            return self._json({"error": "not found"}, 404)
        except CONNECTION_ERRS:
            return   # browser hung up (video seek/close) — normal, stay quiet
        except Exception as e:  # noqa: BLE001
            try:
                return self._json({"error": str(e)[:300]}, 500)
            except CONNECTION_ERRS:
                return

    def _media(self, cfg: Config, rel: str):
        if rel.startswith("output/"):
            p = safe_media_path(cfg.work_dir / "output", rel[len("output/"):])
        elif rel.startswith("preview/"):
            p = safe_media_path(cfg.work_dir / "ui_previews",
                                rel[len("preview/"):])
        else:
            p = None
        if p is None:
            return self._json({"error": "not found"}, 404)
        size = p.stat().st_size
        ctype = MEDIA_EXTS[p.suffix.lower()]
        start, end, code = 0, size - 1, 200
        rng = self.headers.get("Range") or ""
        if rng.startswith("bytes="):
            try:
                a, _, b = rng[6:].partition("-")
                start = int(a) if a else 0
                end = min(int(b), size - 1) if b else size - 1
                if 0 <= start <= end:
                    code = 206
            except ValueError:
                pass
        extra = {"Accept-Ranges": "bytes"}
        if code == 206:
            extra["Content-Range"] = f"bytes {start}-{end}/{size}"
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store")
        for k, v in extra.items():
            self.send_header(k, v)
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(p, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    # ---- POST ------------------------------------------------------------------
    def do_POST(self):
        u = urlparse(self.path)
        cfg = self.server.cfg
        if u.path == "/api/persona":     # raw-bytes upload, not JSON
            try:
                return self._persona_upload(cfg, parse_qs(u.query))
            except CONNECTION_ERRS:
                return
            except Exception as e:  # noqa: BLE001
                try:
                    return self._json({"error": str(e)[:400]}, 500)
                except CONNECTION_ERRS:
                    return
        try:
            a = self._body()
        except Exception as e:  # noqa: BLE001
            return self._json({"error": f"bad JSON body: {e}"}, 400)
        try:
            if u.path == "/api/build":
                return self._build(cfg, a)
            if u.path == "/api/rerun":
                return self._rerun(cfg, a)
            if u.path == "/api/stop":
                job = str(a.get("job_id") or "")
                if not valid_job_id(job):
                    return self._json({"error": "bad job id"}, 400)
                return self._json({"ok": stop_job(job)})
            if u.path == "/api/preview":
                return self._json(make_preview(
                    cfg, a.get("text", ""),
                    str(a.get("voice") or "en-US-ChristopherNeural"),
                    a.get("speed")))
            if u.path == "/api/verify":
                return self._verify(cfg, a)
            return self._json({"error": "not found"}, 404)
        except CONNECTION_ERRS:
            return   # client hung up — normal, stay quiet
        except Exception as e:  # noqa: BLE001
            try:
                return self._json({"error": str(e)[:400]}, 500)
            except CONNECTION_ERRS:
                return

    def _persona_ok(self, persona: str) -> bool:
        if not persona:
            return True
        try:
            p = Path(persona).resolve()
        except OSError:
            return False
        for root in ((PROJECT_ROOT / "assets").resolve(),
                     (self.server.cfg.work_dir / "ui_personas").resolve()):
            try:
                p.relative_to(root)
                return p.is_dir()
            except ValueError:
                continue
        return False

    def _persona_upload(self, cfg: Config, q: dict):
        name = (q.get("name") or ["my-photo"])[0]
        cut = (q.get("cut") or ["0"])[0] in ("1", "true", "yes")
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > PERSONA_MAX_BYTES:
            return self._json({"error": "photo empty or over 32 MB"}, 400)
        data = self.rfile.read(n)
        try:
            info = save_uploaded_persona(cfg, name, data, cut=cut)
        except ValueError as e:
            return self._json({"error": str(e)}, 400)
        return self._json({"ok": True, "persona": info})

    def _spawn(self, cfg: Config, a: dict, script_path: Path):
        job_id = str(a.get("job_id") or "").strip()
        if not valid_job_id(job_id):
            return self._json(
                {"error": "job id must be 1-64 chars: letters, digits, "
                          "underscore, dash"}, 400)
        info = proc_info(job_id)
        if info and info["running"]:
            return self._json({"error": f"job '{job_id}' is already running"},
                              409)
        if a.get("persona_dir") and not self._persona_ok(
                str(a["persona_dir"])):
            return self._json({"error": "persona dir must be a folder under "
                                        "<project>/assets or "
                                        "<work>/ui_personas"}, 400)
        argv = build_argv({**a, "script_path": str(script_path)})
        proc = spawn_build(cfg, argv, job_id)
        return self._json({"ok": True, "job_id": job_id, "pid": proc.pid})

    def _build(self, cfg: Config, a: dict):
        text = str(a.get("script_text") or "")
        if len(text.strip()) < 40:
            return self._json(
                {"error": "script too short — paste at least a few "
                          "sentences (or load the sample)"}, 400)
        job_id = str(a.get("job_id") or "").strip()
        if not valid_job_id(job_id):
            return self._json({"error": "bad job id"}, 400)
        sd = cfg.work_dir / "ui_scripts"
        sd.mkdir(parents=True, exist_ok=True)
        sp = sd / f"{job_id}.txt"
        sp.write_text(text, encoding="utf-8", newline="\n")
        return self._spawn(cfg, a, sp)

    def _rerun(self, cfg: Config, a: dict):
        job_id = str(a.get("job_id") or "")
        if not valid_job_id(job_id):
            return self._json({"error": "bad job id"}, 400)
        meta_p = cfg.jobs_dir / job_id / "ui_meta.json"
        if not meta_p.exists():
            return self._json(
                {"error": "no UI metadata — this job was built from the "
                          "CLI; resume it by rerunning its command"}, 400)
        argv = json.loads(meta_p.read_text(encoding="utf-8"))["argv"]
        sp = None
        for i, tok in enumerate(argv):
            if tok == "--script" and i + 1 < len(argv):
                sp = Path(argv[i + 1])
        if sp is None or not sp.exists():
            return self._json({"error": "original script file is gone; "
                                        "rebuild from the form"}, 400)
        info = proc_info(job_id)
        if info and info["running"]:
            return self._json({"error": f"job '{job_id}' is already running"},
                              409)
        proc = spawn_build(cfg, argv, job_id)
        return self._json({"ok": True, "job_id": job_id, "pid": proc.pid})

    def _verify(self, cfg: Config, a: dict):
        job_id = str(a.get("job_id") or "")
        if not valid_job_id(job_id):
            return self._json({"error": "bad job id"}, 400)
        argv = [sys.executable, "-m", "engine", "verify",
                "--job-id", job_id]
        if a.get("deep"):
            argv.append("--deep")
        proc = subprocess.run(argv, cwd=str(PROJECT_ROOT),
                              capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              timeout=3600)
        out = (proc.stdout or "") + (proc.stderr or "")
        return self._json({"ok": proc.returncode == 0,
                           "output": out[-8000:]})


# ---- entry point -------------------------------------------------------------

def cmd_ui(args) -> int:
    cfg = Config.load()
    cfg.ensure_dirs()
    port, srv = args.port, None
    for attempt in range(10):
        try:
            srv = UIServer(("127.0.0.1", port), cfg)
            break
        except OSError:
            if attempt == 9:
                raise SystemExit(
                    f"no free port in {args.port}..{args.port + 9}")
            port += 1
    url = f"http://127.0.0.1:{port}/"
    print(f"Video Engine UI -> {url}")
    print(f"work dir: {cfg.work_dir}")
    print("Ctrl+C stops the page; running builds keep going and the UI "
          "picks them up when you reopen it.")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        srv.server_close()
    return 0

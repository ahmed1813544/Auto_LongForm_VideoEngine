"""UI server: pure helpers offline + loopback smoke tests.

The integration tests bind 127.0.0.1 on an ephemeral port and point
VIDEOENGINE_WORK at tmp_path (Config.load's load_dotenv(override=False)
means the env var wins over the real .env).  No builds, no network.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

from engine import ui
from engine.config import Config


def make_cfg(tmp_path, monkeypatch) -> Config:
    monkeypatch.setenv("VIDEOENGINE_WORK", str(tmp_path))
    cfg = Config.load()
    cfg.ensure_dirs()
    return cfg


# ---- pure helpers ----------------------------------------------------------

def test_estimate_minutes():
    assert ui.estimate_minutes(165) == pytest.approx(1.0)
    assert ui.estimate_minutes(0) == 0.0


def test_valid_job_id():
    assert ui.valid_job_id("hotel_30-x")
    assert ui.valid_job_id("a")
    assert not ui.valid_job_id("../evil")
    assert not ui.valid_job_id("")
    assert not ui.valid_job_id(None)
    assert not ui.valid_job_id("a b")
    assert not ui.valid_job_id("x" * 65)


def test_build_argv_minimal(tmp_path):
    sp = tmp_path / "s.txt"
    sp.write_text("x", encoding="utf-8")
    argv = ui.build_argv({"script_path": str(sp), "job_id": "j1"})
    assert argv[0] == sys.executable
    assert argv[1:4] == ["-m", "engine", "build"]
    assert "--script" in argv and str(sp) in argv
    assert "--job-id" in argv and "j1" in argv
    # defaults stay out of the argv (CLI defaults apply)
    for flag in ("--eq", "--speed", "--fresh", "--skip-stock", "--nvenc",
                 "--dry-run", "--mood", "--persona-dir", "--stock-max-h"):
        assert flag not in argv


def test_build_argv_full(tmp_path):
    sp = tmp_path / "s.txt"
    sp.write_text("x", encoding="utf-8")
    argv = ui.build_argv({
        "script_path": str(sp), "job_id": "j2",
        "moods": ["green valley", "  river  ", ""],
        "persona_dir": "assets/rowe", "voice": "en-US-GuyNeural",
        "speed": 1.1, "eq": True, "skip_stock": True, "nvenc": False,
        "seg_max_min": 5, "seed": 7, "stock_max_h": 1080})
    assert argv.count("--mood") == 2      # empty dropped
    assert "river" in argv                # trimmed
    assert "--eq" in argv and "--skip-stock" in argv
    assert "--nvenc" not in argv
    assert argv[argv.index("--speed") + 1] == "1.1"
    assert argv[argv.index("--seg-max-min") + 1] == "5"
    assert argv[argv.index("--seed") + 1] == "7"
    assert argv[argv.index("--stock-max-h") + 1] == "1080"


def test_build_argv_stock_max_h_default_and_bad(tmp_path):
    sp = tmp_path / "s.txt"
    sp.write_text("x", encoding="utf-8")
    base = {"script_path": str(sp), "job_id": "j3"}
    # 720 is the CLI default -> stays out of the argv
    assert "--stock-max-h" not in ui.build_argv({**base, "stock_max_h": 720})
    argv = ui.build_argv({**base, "stock_max_h": "540"})   # str coerced
    assert argv[argv.index("--stock-max-h") + 1] == "540"
    with pytest.raises(ValueError):
        ui.build_argv({**base, "stock_max_h": 4320})        # not a choice
    with pytest.raises(ValueError):
        ui.build_argv({**base, "stock_max_h": "tall"})      # not an int


def test_build_argv_rejects_bad(tmp_path):
    with pytest.raises(ValueError):
        ui.build_argv({"script_path": "x.txt", "job_id": "../bad"})
    with pytest.raises(ValueError):
        ui.build_argv({"job_id": "ok"})   # no script_path


def test_summarize_state():
    d = {"phases": {"timeline": {"done": True}, "loudness": {"done": True}},
         "segments": [{"idx": 0, "state": "done"},
                      {"idx": 1, "state": "rendering"},
                      {"idx": 2, "state": "pending"}]}
    s = ui.summarize_state(d)
    assert s["segments_total"] == 3
    assert s["segments"] == {"done": 1, "rendering": 1, "pending": 1}
    assert s["progress"] == pytest.approx(1 / 3, abs=1e-4)  # rounded to 4dp
    assert set(s["phases_done"]) == {"timeline", "loudness"}


def test_summarize_state_concat_wins():
    d = {"phases": {"concat": {"done": True}},
         "segments": [{"idx": 0, "state": "done"}]}
    assert ui.summarize_state(d)["progress"] == 1.0


def test_summarize_state_empty():
    s = ui.summarize_state({})
    assert s["segments_total"] == 0 and s["progress"] == 0.0
    assert s["phases_done"] == []


def test_safe_media_path(tmp_path):
    (tmp_path / "out").mkdir()
    good = tmp_path / "out" / "a.mp4"
    good.write_bytes(b"x")
    assert ui.safe_media_path(tmp_path, "out/a.mp4") == good.resolve()
    assert ui.safe_media_path(tmp_path, "out\\a.mp4") == good.resolve()
    assert ui.safe_media_path(tmp_path, "../secret.mp4") is None
    assert ui.safe_media_path(tmp_path, "out/../../secret.mp4") is None
    assert ui.safe_media_path(tmp_path, "out/a.txt") is None       # ext
    assert ui.safe_media_path(tmp_path, "out/missing.mp4") is None  # absent


def test_voice_list_fallback_offline(monkeypatch):
    import engine.edge_tts as et
    monkeypatch.setattr(et, "known_voices", lambda refresh=False: [])
    vs = ui.voice_list()
    assert vs and all(v["ShortName"].startswith("en-") for v in vs)


# ---- persona uploads ---------------------------------------------------------

def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args],
                   check=True)


@pytest.fixture
def jpg_bytes(tmp_path):
    p = tmp_path / "in.jpg"
    _ffmpeg("-f", "lavfi", "-i", "color=c=red:s=80x60", "-frames:v", "1",
            str(p))
    return p.read_bytes()


@pytest.fixture
def png_bytes(tmp_path):
    p = tmp_path / "in.png"
    _ffmpeg("-f", "lavfi", "-i", "color=c=blue:s=40x40", "-frames:v", "1",
            str(p))
    return p.read_bytes()


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                  reason="ffmpeg needed")


def test_persona_bad_name_rejected(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="name"):
        ui.save_uploaded_persona(cfg, "???", b"x" * 100, cut=False)
    with pytest.raises(ValueError, match="32 MB"):
        ui.save_uploaded_persona(cfg, "ok", b"", cut=False)


def test_persona_cut_requires_rembg(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, monkeypatch)
    monkeypatch.setattr(ui, "rembg_available", lambda: False)
    with pytest.raises(ValueError, match="rembg"):
        ui.save_uploaded_persona(cfg, "me", b"x" * 100, cut=True)


@needs_ffmpeg
def test_persona_plain_photo(tmp_path, monkeypatch, jpg_bytes):
    cfg = make_cfg(tmp_path, monkeypatch)
    info = ui.save_uploaded_persona(cfg, "My Photo!", jpg_bytes, cut=False)
    assert info["name"] == "My-Photo (yours)"
    assert info["cut"] is False and info["source"] == "uploaded"
    pdir = cfg.work_dir / "ui_personas" / "My-Photo"
    assert (pdir / "storyteller.png").is_file()
    assert not (pdir / "upload.bin").exists()       # temp cleaned up
    assert info["path"] == str(pdir)
    # uploaded persona shows up in the scan, assets ones keep working
    names = {p["path"]: p for p in ui.scan_personas(cfg)}
    assert str(pdir) in names
    assert names[str(pdir)]["source"] == "uploaded"


@needs_ffmpeg
def test_persona_cut_with_fake_rembg(tmp_path, monkeypatch, jpg_bytes,
                                     png_bytes):
    import types

    cfg = make_cfg(tmp_path, monkeypatch)
    fake = types.ModuleType("rembg")
    fake.remove = lambda data: png_bytes      # stand-in for the AI cutout
    monkeypatch.setitem(sys.modules, "rembg", fake)
    monkeypatch.setattr(ui, "rembg_available", lambda: True)
    info = ui.save_uploaded_persona(cfg, "guy", jpg_bytes, cut=True)
    assert info["cut"] is True
    assert (cfg.work_dir / "ui_personas" / "guy" / "storyteller.png").is_file()


@needs_ffmpeg
def test_persona_garbage_rejected(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="readable image"):
        ui.save_uploaded_persona(cfg, "junk", b"not-an-image" * 20, cut=False)
    assert not (cfg.work_dir / "ui_personas" / "junk"
                / "storyteller.png").exists()


# ---- process plumbing -------------------------------------------------------

def test_read_console_offset_and_truncation(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, monkeypatch)
    p = ui.console_path(cfg, "j2")
    assert not p.exists()
    r = ui.read_console(cfg, "j2", 0)
    assert r == {**r, "text": "", "size": 0, "running": False}
    p.parent.mkdir(parents=True)
    p.write_text("hello world", encoding="utf-8")
    r = ui.read_console(cfg, "j2", 0)
    assert r["text"] == "hello world" and r["offset"] == 11
    assert ui.read_console(cfg, "j2", 11)["text"] == ""
    p.write_text("short", encoding="utf-8")   # rerun truncated the log
    r = ui.read_console(cfg, "j2", 11)
    assert r["text"] == "short" and r["offset"] == 5


def test_spawn_and_stop(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, monkeypatch)
    argv = [sys.executable, "-c",
            "import time; print('alive', flush=True); time.sleep(60)"]
    proc = ui.spawn_build(cfg, argv, "uitest")
    try:
        time.sleep(1.5)
        assert ui.proc_info("uitest")["running"] is True
        assert ui.stop_job("uitest") is True
        proc.wait(timeout=20)
        assert proc.returncode != 0          # killed, not exited clean
        assert "alive" in ui.console_path(cfg, "uitest").read_text(
            encoding="utf-8")
        meta = json.loads((cfg.jobs_dir / "uitest" / "ui_meta.json")
                          .read_text(encoding="utf-8"))
        assert meta["argv"] == argv
        assert ui.stop_job("uitest") is False  # already dead
    finally:
        if proc.poll() is None:
            proc.kill()
        with ui._LOCK:
            ui.PROCS.pop("uitest", None)


def test_list_jobs(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, monkeypatch)
    jd = cfg.jobs_dir / "jobA"
    jd.mkdir(parents=True)
    (jd / "state.json").write_text(json.dumps(
        {"phases": {"timeline": {"done": True}},
         "segments": [{"idx": 0, "state": "done"},
                      {"idx": 1, "state": "pending"}]}), encoding="utf-8")
    (cfg.work_dir / "output").mkdir(parents=True)
    (cfg.work_dir / "output" / "jobA.mp4").write_bytes(b"x" * 10)
    # a dir without state.json is ignored
    (cfg.jobs_dir / "junk").mkdir()
    jobs = ui.list_jobs(cfg)
    assert [j["job_id"] for j in jobs] == ["jobA"]
    j = jobs[0]
    assert j["segments_total"] == 2 and j["progress"] == 0.5
    assert j["running"] is False and j["returncode"] is None
    assert j["mp4"] == {"size": 10, "url": "/media/output/jobA.mp4"}
    assert j["ui"] is False


# ---- live loopback server ---------------------------------------------------

def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read()


def _get_err(url):
    try:
        return _get(url)
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _post(url, obj):
    req = urllib.request.Request(
        url, data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_server_loopback_smoke(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, monkeypatch)
    srv = ui.UIServer(("127.0.0.1", 0), cfg)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{port}"

        # index page
        code, body = _get(base + "/")
        assert code == 200 and b"<html" in body.lower()
        assert b"Video Engine" in body

        # config: booleans only, never key values
        code, body = _get(base + "/api/config")
        c = json.loads(body)
        assert code == 200
        assert c["work_dir"] == str(tmp_path)
        assert set(c["stock_keys"]) == {"pixabay", "pexels"}
        assert all(isinstance(v, bool) for v in c["stock_keys"].values())
        assert isinstance(c["rembg"], bool)
        assert c["wpm"] == ui.WPM_ESTIMATE

        # jobs: empty, then the one we fake
        assert json.loads(_get(base + "/api/jobs")[1]) == []
        jd = cfg.jobs_dir / "jobZ"
        jd.mkdir(parents=True)
        (jd / "state.json").write_text(json.dumps({"phases": {}, "segments": []}),
                                       encoding="utf-8")
        jobs = json.loads(_get(base + "/api/jobs")[1])
        assert [j["job_id"] for j in jobs] == ["jobZ"]

        # log tail of a nonexistent job: benign empty
        lg = json.loads(_get(base + "/api/log?job=nope&offset=0")[1])
        assert lg["text"] == "" and lg["running"] is False

        # build rejections never spawn anything
        code, body = _post(base + "/api/build",
                           {"job_id": "../evil", "script_text": "x" * 60})
        assert code == 400 and b"job id" in body.lower()
        code, body = _post(base + "/api/build",
                           {"job_id": "tiny", "script_text": "short"})
        assert code == 400 and b"too short" in body.lower()
        assert ui.PROCS == {}

        # persona gate: dir outside assets/ and ui_personas/ -> 400, no spawn
        code, body = _post(base + "/api/build",
                           {"job_id": "ptest", "script_text": "x" * 60,
                            "persona_dir": str(tmp_path)})
        assert code == 400 and b"persona" in body.lower()
        assert ui.PROCS == {}

        # persona upload: empty body -> 400 (no ffmpeg needed for this path)
        req = urllib.request.Request(base + "/api/persona?name=me",
                                     data=b"", method="POST")
        try:
            urllib.request.urlopen(req, timeout=10)
            pytest.fail("expected 400 for empty upload")
        except urllib.error.HTTPError as e:
            assert e.code == 400 and b"empty" in e.read().lower()

        # junk bytes -> readable-image 400, no persona dir left behind
        if shutil.which("ffmpeg"):
            req = urllib.request.Request(base + "/api/persona?name=junk2",
                                         data=b"not-an-image" * 20,
                                         method="POST")
            try:
                urllib.request.urlopen(req, timeout=60)
                pytest.fail("expected 400 for junk upload")
            except urllib.error.HTTPError as e:
                assert e.code == 400 and b"readable" in e.read().lower()
            assert not (tmp_path / "ui_personas" / "junk2"
                        / "storyteller.png").exists()

        # media: traversal blocked, missing 404s, real file served
        assert _get_err(base + "/media/output/../../.env")[0] == 404
        assert _get_err(base + "/media/output/ghost.mp4")[0] == 404
        outdir = cfg.work_dir / "output"
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "jobZ.mp4").write_bytes(b"\x00" * 42)
        code, body = _get(base + "/media/output/jobZ.mp4")
        assert code == 200 and body == b"\x00" * 42
        # Range request -> 206 with the right slice
        req = urllib.request.Request(base + "/media/output/jobZ.mp4",
                                     headers={"Range": "bytes=10-19"})
        with urllib.request.urlopen(req, timeout=10) as r:
            assert r.status == 206 and r.read() == b"\x00" * 10
            assert r.headers["Content-Range"] == "bytes 10-19/42"
        # preview dir is also media-reachable, .txt is not
        pv = cfg.work_dir / "ui_previews"
        pv.mkdir(parents=True, exist_ok=True)
        (pv / "v.wav").write_bytes(b"RIFFxx")
        (pv / "v.txt").write_text("nope", encoding="utf-8")
        assert _get(base + "/media/preview/v.wav")[0] == 200
        assert _get_err(base + "/media/preview/v.txt")[0] == 404

        # rerun without ui_meta -> clear error
        code, body = _post(base + "/api/rerun", {"job_id": "jobZ"})
        assert code == 400 and b"CLI" in body

        # unknown route -> 404 json
        code, body = _get_err(base + "/api/nope")
        assert code == 404 and b"error" in body
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)


def test_media_client_abort_is_quiet(tmp_path, monkeypatch, capsys):
    """Browser hangs up mid-mp4 (seek/close): server stays alive, no
    traceback noise (Windows surfaces this as WinError 10054/10053)."""
    import socket

    cfg = make_cfg(tmp_path, monkeypatch)
    srv = ui.UIServer(("127.0.0.1", 0), cfg)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        outdir = cfg.work_dir / "output"
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "big.mp4").write_bytes(b"\x00" * (8 * 1024 * 1024))
        for _ in range(3):
            s = socket.create_connection(("127.0.0.1", port), timeout=5)
            s.sendall(b"GET /media/output/big.mp4 HTTP/1.1\r\n"
                      b"Host: x\r\n\r\n")
            s.recv(64)      # response started streaming
            s.close()       # slammed mid-body
        time.sleep(0.5)
        # server still healthy...
        assert json.loads(_get(f"http://127.0.0.1:{port}/api/config")[1])[
            "work_dir"] == str(tmp_path)
        # ...and quiet: no traceback escaped to stderr
        assert "Traceback" not in capsys.readouterr().err
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)

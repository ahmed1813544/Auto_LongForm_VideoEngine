"""Segment command builders: plain path regression + --eq filtergraph.

The non-eq render path must stay byte-identical (finished jobs like
land30/story30 resume under it); --eq may only ever ADD to the graph.
"""
from engine.render.ffmpeg_cmds import segment_cmd

KW = dict(
    master_wav="jobs/j/audio/master.wav", seg_t0=0.0, seg_dur=14.0,
    bg_clips=[("library/norm/clip.mp4", 8.0, 14.0)],
    persona_img="assets/persona.png", ass_path="jobs/j/ass/seg0000.ass",
    fonts_dir="assets/fonts", gain_db=-2.0, out="seg0000.mp4",
)


def _fc(cmd):
    return cmd[cmd.index("-filter_complex") + 1]


def test_plain_path_unchanged():
    cmd = segment_cmd(**KW)
    assert cmd.count("-i") == 3                      # wav + bg clip + persona
    assert cmd.count("-f") == 0                      # no lavfi extras yet
    fc = _fc(cmd)
    assert "asplit" not in fc and "color=c=white" not in fc
    assert cmd[cmd.index("-map") + 1] == "[vout]"
    assert "0:a" in cmd and "-af" in cmd             # audio via -af as before
    assert "-map" in cmd and "[aud]" not in cmd


def test_eq_adds_chain_and_moves_audio_into_graph():
    cmd = segment_cmd(**KW, eq=True)
    fc = _fc(cmd)
    # white bar canvas is an extra input
    assert "color=c=white:s=1032x230:r=30" in cmd
    # split + audible branch (replaces the old -af) + vis preamp on top
    assert "[0:a]asplit=2[aud0][visa]" in fc
    assert "[aud0]volume=-2.0dB,aresample=48000[aud]" in fc
    assert "volume=38.0dB,showfreqs" in fc           # -2 + EQ_GAIN_DB 40
    assert ("showfreqs=s=1032x230:rate=30:mode=bar:ascale=sqrt"
            ":fscale=log:win_size=128:colors=white") in fc
    # transparency route: gray -> alpha of white source, dimmed
    assert "[eq]format=gray[eqg]" in fc
    assert "alphamerge,colorchannelmixer=aa=0.35[eqa]" in fc
    assert "overlay=x=48:y=716:eof_action=pass[mv]" in fc
    # bars sit UNDER the captions: ass must come after the overlay
    assert fc.index("overlay=x=48:y=716") < fc.index("ass=filename")
    # audible audio mapped from the graph; -af would be an error now
    assert "-af" not in cmd and "0:a" not in cmd
    i = cmd.index("-map")
    assert cmd[i + 2:i + 4] == ["-map", "[aud]"]


def test_eq_without_persona_overlays_bg_directly():
    kw = dict(KW, persona_img=None)
    fc = _fc(segment_cmd(**kw, eq=True))
    assert "[bg][eqa]overlay" in fc


def test_params_obj_eq_key_only_when_set(tmp_path):
    """Legacy resume contract: without --eq the params dict must be
    structurally identical to the pre-feature one, so finished jobs
    (land30/story30) still accept their stored params_hash on rerun."""
    from types import SimpleNamespace
    from engine.cli import _params_obj
    from engine.state import params_hash

    sp = tmp_path / "s.txt"
    sp.write_text("hi", encoding="utf-8")
    cfg = SimpleNamespace(work_dir=tmp_path)
    base = dict(voice="en-US-ChristopherNeural", speed=1.0,
                seg_max_min=9.0, seed=1234, nvenc=False, crf=19,
                preset="veryfast", skip_stock=False, mood=["x"],
                gain_db=None, target_lufs=-16.0)
    off = _params_obj(cfg, SimpleNamespace(**base, eq=False),
                      ["st"], [], [sp])
    on = _params_obj(cfg, SimpleNamespace(**base, eq=True),
                     ["st"], [], [sp])
    assert "eq" not in off
    assert on["eq"] is True
    assert params_hash(off) != params_hash(on)

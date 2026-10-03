from engine.ass_gen import POP, render_ass, write_ass_file


def ev(i, t0, t1, text="cap"):
    return {"id": i, "story": 0, "sentence": 0, "group": i,
            "text": f"{text} {i}", "t0_ms": t0, "t1_ms": t1}


def test_fractional_boundary_assigns_event_once():
    # byte-math durations are fractional; floor(left t1) + ceil(right t0)
    # must keep each caption in exactly one segment file (regression: the
    # old int() stamping duplicated boundary events as 0.33 ms ghosts).
    import math
    boundary = 29649.333333
    events = [ev(0, 800, boundary), ev(1, boundary, 32649.333333)]
    left = render_ass(events, seg_t0_ms=0,
                      seg_t1_ms=math.floor(boundary))
    right = render_ass(events, seg_t0_ms=math.ceil(boundary),
                       seg_t1_ms=math.ceil(32649.333333))
    assert left.count("Dialogue:") + right.count("Dialogue:") == 2
    assert left.count("cap 0") == 1 and right.count("cap 1") == 1


def test_full_render():
    out = render_ass([ev(0, 800, 2900), ev(1, 2900, 5000)])
    assert out.count("Dialogue:") == 2
    assert "0:00:00.80" in out and "0:00:05.00" in out


def test_segment_restamp_clamp_and_drop():
    events = [ev(0, 0, 1000), ev(1, 1500, 3000), ev(2, 9000, 10000)]
    out = render_ass(events, seg_t0_ms=1000, seg_t1_ms=4000)
    assert out.count("Dialogue:") == 1          # event 1 only
    assert "0:00:00.50" in out and "0:00:02.00" in out
    assert "0:00:09.00" not in out              # absolute time must vanish


def test_segment_clips_straddling_event():
    # shouldn't happen by construction, but clamp must be exact
    out = render_ass([ev(0, 500, 1500)], seg_t0_ms=800, seg_t1_ms=1200)
    assert "0:00:00.00" in out and "0:00:00.40" in out


def test_escapes_braces(tmp_path):
    n = write_ass_file(tmp_path / "x.ass",
                       [ev(0, 0, 1000, text="evil {b} tag")])
    assert n == 1
    t = (tmp_path / "x.ass").read_text()
    evs = t.split("[Events]")[1]
    assert "{b}" not in evs            # user braces neutralized ...
    assert "(b)" in evs                # ... into parentheses
    # the ONLY braces left are the intentional pop override block
    assert evs.count("{") == 1 and evs.count("}") == 1
    assert evs.index("{") < evs.index("evil")   # sits before the text


def test_hour_rollover_format():
    out = render_ass([ev(0, 3_600_000, 3_602_500)])
    assert "1:00:00.00" in out and "1:00:02.50" in out


def test_pop_outline_style():
    # genre look (2026-09-06 redesign): white Montserrat ExtraBold, NO card
    # (BorderStyle=1), thick pure-black outline + semi-opaque black drop
    # shadow; every phrase punches in via a per-event \t scale transform.
    out = render_ass([ev(0, 0, 1000)])
    style = next(l for l in out.splitlines() if l.startswith("Style:"))
    assert "Montserrat ExtraBold" in style
    assert ",1,5.0,3.5," in style              # borderstyle, outline, shadow
    assert "&H00000000" in style               # pure black outline
    assert "&H60000000" in style               # ~63% opaque shadow ink
    assert "&H00FFFFFF" in style               # pure white glyphs
    d = next(l for l in out.splitlines() if l.startswith("Dialogue:"))
    assert d.endswith(POP + "cap 0")          # pop tag prefixes every card
    assert "\\t(0,90,\\fscx112\\fscy112)" in d  # overshoot
    assert "\\fad(0,110)" in d                  # tail fade-out

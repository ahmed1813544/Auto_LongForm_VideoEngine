"""Scene planner + job state tests — pure, no network, no ffmpeg."""
import json

import pytest

from engine.scene_planner import (POOL_HEADROOM, Segment, assign_personas,
                                  plan_scenes, plan_segments,
                                  pool_coverage, pool_from_clips)
from engine.state import JobState, ParamsMismatch, params_hash


def _events(n=60, ev_ms=3000, story_every=10):
    """Uniform 3-s events; story_end every 10th (stories of 30 s)."""
    ev = []
    for i in range(n):
        t0 = 800 + i * ev_ms
        ev.append({"id": i, "story": i // story_every, "sentence": i,
                   "group": 0, "text": f"s{i}",
                   "t0_ms": float(t0), "t1_ms": float(t0 + ev_ms),
                   "story_end": (i % story_every == story_every - 1)})
    return ev


# ---- plan_segments --------------------------------------------------------

def test_segments_fit_window_and_tiling():
    ev = _events(60)  # total ~180 s + 0.8 preroll start
    total = ev[-1]["t1_ms"]
    segs = plan_segments(ev, total, 30.0)
    assert segs[0].t0_ms == 0.0
    assert segs[-1].t1_ms == total
    for s in segs:
        assert s.t1_ms - s.t0_ms <= 30000 + 1
    # perfect tiling: each seg starts where previous ended
    for a, b in zip(segs, segs[1:]):
        assert a.t1_ms == b.t0_ms


def test_cuts_land_on_event_ends_never_straddle():
    ev = _events(40)
    segs = plan_segments(ev, ev[-1]["t1_ms"], 12.0)
    ends = {e["t1_ms"] for e in ev}
    for s in segs:
        assert s.t1_ms in ends
        inside = [e for e in ev
                  if s.t0_ms <= e["t0_ms"] < s.t1_ms]
        # no event starts before a cut and ends after it
        assert all(e["t1_ms"] <= s.t1_ms for e in inside)


def test_story_end_preferred():
    ev = _events(40, story_every=10)  # story ends at 30s, 60s...
    segs = plan_segments(ev, ev[-1]["t1_ms"], 35.0)
    # 35s window from 0: fits events ending <= 35.8s; latest story-end = 30.8
    assert segs[0].t1_ms == ev[9]["t1_ms"]


def test_no_story_end_in_window_cuts_at_last_fit():
    ev = _events(30, ev_ms=3000, story_every=25)  # ends every 75s
    segs = plan_segments(ev, ev[-1]["t1_ms"], 20.0)
    # 20s window from 0: events end at 3.8+3k; last fitting t1 <= 20 is
    # ev[5] at 18.8s (preroll shifts boundaries)
    assert abs(segs[0].dur_s - 18.8) < 0.01


def test_huge_event_tolerated():
    ev = [{"id": 0, "story": 0, "sentence": 0, "group": 0, "text": "x",
           "t0_ms": 0.0, "t1_ms": 120000.0, "story_end": True},
          {"id": 1, "story": 0, "sentence": 1, "group": 0, "text": "y",
           "t0_ms": 120000.0, "t1_ms": 123000.0, "story_end": False}]
    segs = plan_segments(ev, 123000.0, 10.0)
    assert segs[0].t1_ms == 120000.0 and segs[-1].t1_ms == 123000.0


def test_audio_tail_covered_by_final_segment():
    # master.wav runs past the last caption (inter-story silence): the
    # segment plan must tile to total_ms, not stop at the last event
    ev = _events(5)
    last_t1 = ev[-1]["t1_ms"]
    segs = plan_segments(ev, last_t1 + 1200.0, 10.0)
    assert segs[-1].t1_ms == last_t1 + 1200.0
    assert segs[-2].t1_ms == last_t1


# ---- plan_scenes ------------------------------------------------------------

def _pool(n, dur=20.0):
    return [{"key": f"pixabay:{i}", "path": f"C:/lib/p{i}.mp4",
             "duration_s": dur} for i in range(n)]


def test_scenes_sum_exactly_to_segment():
    ev = _events(20)
    segs = plan_segments(ev, ev[-1]["t1_ms"], 30.0)
    plan_scenes(segs, _pool(12))
    for s in segs:
        assert abs(sum(x["dur_s"] for x in s.scenes) - s.dur_s) < 0.01
        # scenes respect the 8-16s band except the folded tail stub
        assert all(0.05 < x["dur_s"] <= 16.5 for x in s.scenes)


def test_scene_planning_deterministic():
    ev = _events(40)
    pool = _pool(10)
    s1 = plan_segments(ev, ev[-1]["t1_ms"], 20.0)
    s2 = plan_segments(ev, ev[-1]["t1_ms"], 20.0)
    plan_scenes(s1, pool, seed=99)
    plan_scenes(s2, pool, seed=99)
    assert [x.scenes for x in s1] == [x.scenes for x in s2]
    plan_scenes(s1, pool, seed=100)
    plan_scenes(s2, pool, seed=101)
    # different seeds must diverge somewhere
    assert [x.scenes for x in s1] != [x.scenes for x in s2]


def test_no_repeat_within_window():
    ev = _events(60)
    segs = plan_segments(ev, ev[-1]["t1_ms"], 30.0)
    plan_scenes(segs, _pool(25), seed=7)
    seq = [sc["key"] for s in segs for sc in s.scenes]
    for i in range(len(seq) - 6):
        assert len(set(seq[i:i + 7])) >= 7 or len(_pool(25)) <= 7


def test_short_clip_cannot_span_scene():
    ev = _events(6)  # 18.8s of events, no story end in range
    segs = plan_segments(ev, ev[-1]["t1_ms"], 24.0)
    assert len(segs) == 1
    plan_scenes(segs, _pool(12, dur=5.0))
    assert all(x["dur_s"] <= 5.0 + 1e-9 for x in segs[0].scenes)
    assert abs(sum(x["dur_s"] for x in segs[0].scenes) - segs[0].dur_s) < 0.01


def test_empty_pool_raises():
    ev = _events(4)
    segs = plan_segments(ev, ev[-1]["t1_ms"], 10.0)
    with pytest.raises(ValueError):
        plan_scenes(segs, [])


def test_tiny_pool_no_deadlock():
    ev = _events(30)
    segs = plan_segments(ev, ev[-1]["t1_ms"], 12.0)
    plan_scenes(segs, _pool(3), seed=5)   # below RECENT_WINDOW+2
    assert all(s.scenes for s in segs)


# ---- pool coverage ----------------------------------------------------------

def test_pool_coverage_headroom():
    pool = _pool(20, dur=24.0)             # 480 s
    cover, ok = pool_coverage(pool, 300.0)  # need 360 s
    assert cover == 480 and ok
    _, ok2 = pool_coverage(pool, 420.0)     # need 504 s
    assert not ok2


def test_pool_from_clips_maps_paths():
    clips = [{"key": "pexels:1", "duration_s": 12.0},
             {"key": "pexels:2", "duration_s": 0.0}]
    paths = {"pexels:1": "C:/lib/n1.mp4", "pexels:2": None}
    pool = pool_from_clips(clips, paths)
    assert pool == [{"key": "pexels:1", "path": "C:/lib/n1.mp4",
                     "duration_s": 12.0}]


# ---- personas ---------------------------------------------------------------

def test_persona_cycled_by_story():
    ev = _events(40, story_every=10)
    segs = plan_segments(ev, ev[-1]["t1_ms"], 35.0)
    imgs = assign_personas(segs, ["a.png", "b.png"])
    assert imgs == [("a.png" if s.story_first % 2 == 0 else "b.png")
                    for s in segs]
    assert assign_personas(segs, []) == [None] * len(segs)


# ---- state --------------------------------------------------------------------

def test_params_hash_stable_and_order_insensitive():
    a = params_hash({"x": 1, "y": [1, 2]})
    b = params_hash({"y": [1, 2], "x": 1})
    assert a == b
    assert a != params_hash({"x": 2, "y": [1, 2]})


def test_state_roundtrip_and_phase_resume(tmp_path):
    ph = params_hash({"v": 1})
    st = JobState.load_or_new(tmp_path, job_id="j1", phash=ph)
    assert not st.phase_done("timeline")
    st.mark_phase("timeline", master="C:/audio/master.wav", total_ms=50000)
    st2 = JobState.load_or_new(tmp_path, job_id="j1", phash=ph)
    assert st2.phase_done("timeline")
    assert st2.phase_output("timeline", "total_ms") == 50000


def test_state_params_mismatch(tmp_path):
    JobState.load_or_new(tmp_path, job_id="j1",
                         phash=params_hash({"v": 1})).save()
    with pytest.raises(ParamsMismatch):
        JobState.load_or_new(tmp_path, job_id="j1",
                             phash=params_hash({"v": 2}))
    # force_new overrides
    st = JobState.load_or_new(tmp_path, job_id="j1",
                              phash=params_hash({"v": 2}), force_new=True)
    assert st.counts() == {}


def test_segment_status_tracking(tmp_path):
    st = JobState.load_or_new(tmp_path, job_id="j1", phash="h")
    st.set_segments([{"idx": 0, "t0_ms": 0, "t1_ms": 10000, "dur_s": 10.0},
                     {"idx": 1, "t0_ms": 10000, "t1_ms": 20000,
                      "dur_s": 10.0}])
    st.mark_segment(0, "done", path="C:/s0.mp4")
    assert st.segment(0)["state"] == "done"
    # re-setting plan (resume) keeps statuses
    st.set_segments([{"idx": 0, "t0_ms": 0, "t1_ms": 10000, "dur_s": 10.0},
                     {"idx": 1, "t0_ms": 10000, "t1_ms": 20000,
                      "dur_s": 10.0}])
    assert st.segment(0)["state"] == "done"


def test_demote_stale_done(tmp_path, capsys):
    st = JobState.load_or_new(tmp_path, job_id="j1", phash="h")
    st.set_segments([{"idx": 0, "t0_ms": 0, "t1_ms": 10000, "dur_s": 10.0}])
    st.mark_segment(0, "done", path=str(tmp_path / "gone.mp4"))
    assert st.demote_stale_done() == 1
    assert st.segment(0)["state"] == "pending"
    assert st.segment(0)["path"] is None


def test_state_file_is_plain_json(tmp_path):
    st = JobState.load_or_new(tmp_path, job_id="j1", phash="h")
    st.save()
    data = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert data["job_id"] == "j1" and data["version"] == JobState.VERSION


# ---- plan_scenes: rounding-drift deadlock regression ----------------------

def test_plan_scenes_drift_stubs_terminate():
    # live bug (2026-09-06): `remaining -= round(dur)` accumulated sub-ms
    # drift until a stub rounded to 0.000; remaining never decreased and
    # the loop appended scenes until MemoryError on a 51 s build.
    pool = [{"key": f"p{i}", "path": f"/x/{i}.mp4", "duration_s": 15.5}
            for i in range(10)]
    for seed in range(40):
        segs = [Segment(0, 0.0, 51_237.0, story_first=0)]
        plan_scenes(segs, pool, seed=seed)
        sc = segs[0].scenes
        assert sc and all(s["dur_s"] >= 0.05 for s in sc)   # ffmpeg-safe
        assert abs(sum(s["dur_s"] for s in sc) - 51.237) < 0.001

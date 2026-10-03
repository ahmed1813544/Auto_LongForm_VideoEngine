"""Pure builders: every ffmpeg invocation as an argv list.

All ffmpeg correctness decisions live here; segment.py only executes.
The encode contract below is shared by normalize + segment render — that
identical contract is what makes the final `-c copy` concat seamless.
"""
from __future__ import annotations

from pathlib import Path

from engine.util import ffgraph_path, sec

FPS = 30
WIDTH, HEIGHT = 1920, 1080
PERSONA_W, PERSONA_H = 800, HEIGHT   # right column for the storyteller

# Golden stream params every rendered segment must match (verify.py).
GOLDEN_VIDEO = {
    "codec_name": "h264", "profile": "High", "width": WIDTH, "height": HEIGHT,
    "r_frame_rate": "30/1", "avg_frame_rate": "30/1", "pix_fmt": "yuv420p",
    "sample_aspect_ratio": "1:1", "time_base": "1/30000", "level": 41,
}
GOLDEN_AUDIO = {
    "codec_name": "aac", "sample_rate": "48000", "channels": 2,
}

# ---- audio-reactive equalizer (--eq) -------------------------------------
# White bars driven by the segment's own narration slice, composited behind
# the captions.  showfreqs has no transparent background, so the render is
# taken through format=gray -> alphamerge (over a white source) ->
# colorchannelmixer=aa (bars translucent; the 5px outline keeps glyphs on
# top).  EQ_GAIN_DB is a visualization-only preamp: raw speech FFT bins sit
# ~1e-3, which on sqrt scale collapses to a hairline; +40 dB fills the
# ~73-230 px strip on loud phrases.  win_size=128 -> 64 bins -> 16 px bars
# across the 1032 px caption column (x 48..1080 = MarginL/MarginR 48/840);
# the strip bottom (y=946) sits just under the text baseline box (MarginV 150).
EQ_GAIN_DB = 40.0
EQ_W, EQ_H = 1032, 230
EQ_X, EQ_Y = 48, HEIGHT - 364
EQ_ALPHA = 0.35


def encode_args(*, nvenc: bool = False, crf: int = 19,
                preset: str = "veryfast") -> list[str]:
    args: list[str] = ["-r", str(FPS)]
    if nvenc:
        args += ["-c:v", "h264_nvenc", "-preset", "p4",
                 "-rc", "cqp", "-cq", "23", "-b:v", "0",
                 "-g", "60", "-strict_gop", "1"]
    else:
        args += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                 "-g", "60", "-keyint_min", "60",
                 "-x264-params", "scenecut=0"]
    args += ["-profile:v", "high", "-level:v", "4.1",
             "-pix_fmt", "yuv420p",
             "-video_track_timescale", "30000",
             "-colorspace", "bt709", "-color_primaries", "bt709",
             "-color_trc", "bt709",
             "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
    return args


# ---- normalization ------------------------------------------------------

NORM_VF = ("fps=30,scale=1920:1080:force_original_aspect_ratio=increase,"
           "crop=1920:1080,setsar=1,format=yuv420p")


def normalize_cmd(src: str | Path, dst: str | Path,
                  *, nvenc: bool = False,
                  pre_crop: tuple[int, int, int, int] | None = None,
                  start_s: float = 0.0,
                  ) -> list[str]:
    """One-time: any stock clip -> 1080p30 yuv420p SAR1:1, no audio.

    pre_crop (w,h,x,y) strips baked-in black bars FIRST — the scale/crop
    cover chain alone can't remove them (a 16:9 file holding 2.9:1 content
    is already "the right shape" and passes straight through).
    start_s trims the head (for animated bars that only open full-frame
    after a while — cropping them would zoom, trimming keeps native res).
    """
    vf = NORM_VF
    if pre_crop is not None:
        w, h, x, y = pre_crop
        vf = f"crop={w}:{h}:{x}:{y},{vf}"
    args = ["ffmpeg", "-hide_banner", "-y"]
    if start_s:
        args += ["-ss", sec(start_s)]
    args += ["-i", str(src), "-an", "-vf", vf]
    return (args + encode_args(nvenc=nvenc)
            + ["-movflags", "+faststart", str(dst)])


# ---- segment render -----------------------------------------------------

def _bg_chain(n_bg: int, start_input: int, dur: float) -> list[str]:
    """Filter fragments for n background clips (already sliced by -ss/-t)."""
    frags = []
    labels = []
    for k in range(n_bg):
        lab = f"[vbg{k}]"
        frags.append(f"[{start_input + k}:v]fps=30,setsar=1,"
                     f"format=yuv420p{lab}")
        labels.append(lab)
    frags.append("".join(labels) + f"concat=n={n_bg}:v=1:a=0,"
                 f"trim=duration={sec(dur)},setpts=PTS-STARTPTS[bg]")
    return frags


def _alpha_bbox(path: str | Path) -> tuple[int, int, int, int] | None:
    """Opaque (w, h, x, y) bounding box of a persona image, or None.

    Trims dead transparent margin before the cover-crop: a wide canvas
    with the figure pushed to one side would otherwise shrink the figure
    into a corner of the column.  Opaque sources (jpg) return the whole
    frame, so the crop is a no-op there.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as im:
            box = im.convert("RGBA").getchannel("A").getbbox()
    except (OSError, ValueError):
        return None
    if not box:
        return None
    x0, y0, x1, y1 = box
    return (x1 - x0, y1 - y0, x0, y0) if x1 > x0 and y1 > y0 else None


def _persona_chain(input_idx: int, *, fade: bool = True,
                   bbox: tuple[int, int, int, int] | None = None) -> list[str]:
    fade_f = ",fade=t=in:st=0:d=0.5:alpha=1" if fade else ""
    trim = f"crop={bbox[0]}:{bbox[1]}:{bbox[2]}:{bbox[3]}," if bbox else ""
    return [
        f"[{input_idx}:v]format=rgba,{trim}"
        f"scale={PERSONA_W}:{PERSONA_H}:force_original_aspect_ratio=increase"
        f":flags=lanczos,"
        # cover the right column top-to-bottom: the vertical crop is
        # top-anchored so the head stays, the horizontal one centers
        f"crop={PERSONA_W}:{PERSONA_H}:(iw-{PERSONA_W})/2:0"
        f"{fade_f}[per]",
        "[bg][per]overlay=x=W-w:y=0[mov]",
    ]


def ass_filter(ass_path: str | Path,
               fonts_dir: str | Path | None = None) -> str:
    s = f"ass=filename={ffgraph_path(ass_path)}"
    if fonts_dir:
        s += f":fontsdir={ffgraph_path(fonts_dir)}"
    return s


def _eq_chain(base: str, white_input: int, *, gain_db: float) -> list[str]:
    """Filter fragments for the narration-reactive EQ overlay.

    Splits the segment's [0:a] audio slice in two: the audible branch takes
    over what the segment's -af used to do (loudness gain + 48 kHz), the
    visualization branch gets EQ_GAIN_DB on top for showfreqs.  Both sides
    come from the same slice, so the bars are sample-synced to the voice by
    construction.  Caller maps [aud] (never 0:a) when this chain is present.
    """
    return [
        "[0:a]asplit=2[aud0][visa]",
        f"[aud0]volume={gain_db}dB,aresample=48000[aud]",
        f"[visa]volume={gain_db + EQ_GAIN_DB}dB,"
        f"showfreqs=s={EQ_W}x{EQ_H}:rate={FPS}:mode=bar"
        f":ascale=sqrt:fscale=log:win_size=128:colors=white[eq]",
        "[eq]format=gray[eqg]",
        f"[{white_input}:v][eqg]alphamerge,"
        f"colorchannelmixer=aa={EQ_ALPHA}[eqa]",
        f"[{base}][eqa]overlay=x={EQ_X}:y={EQ_Y}:eof_action=pass[mv]",
    ]


def segment_cmd(*, master_wav: str | Path, seg_t0: float, seg_dur: float,
                bg_clips: list[tuple[str | Path, float, float]],
                persona_img: str | Path | None,
                ass_path: str | Path, fonts_dir: str | Path | None,
                gain_db: float, out: str | Path,
                nvenc: bool = False, crf: int = 19,
                preset: str = "veryfast", skip_stock_testsrc2: bool = False,
                eq: bool = False,
                ) -> list[str]:
    """Render one timeline segment [t0, t0+dur) to an mp4.

    bg_clips: [(normalized_clip_path, in_s, dur_s), ...] summing to seg_dur.
    If skip_stock_testsrc2, the background is a testsrc2 lavfi source
    (developer path for zero-key end-to-end tests).
    eq=True adds the audio-reactive equalizer bars behind the captions
    (--eq); the plain render path stays byte-identical without it.
    """
    cmd: list[str] = ["ffmpeg", "-hide_banner", "-y"]
    # audio slice (sample-exact on PCM)
    cmd += ["-ss", sec(seg_t0), "-t", sec(seg_dur), "-i", str(master_wav)]
    n_in = 1  # input 0 = wav
    if skip_stock_testsrc2:
        cmd += ["-f", "lavfi",
                "-i", f"testsrc2=size={WIDTH}x{HEIGHT}:rate={FPS}"
                      f":duration={sec(seg_dur)}"]
        bg_inputs = 1
        bg_start = n_in
        n_in += 1
    else:
        for path, in_s, dur_s in bg_clips:
            cmd += ["-ss", sec(in_s), "-t", sec(dur_s), "-i", str(path)]
            n_in += 1
        bg_inputs = len(bg_clips)
        bg_start = 1

    # persona PNG (optional)
    persona_input = None
    if persona_img is not None:
        cmd += ["-loop", "1", "-framerate", str(FPS), "-t", sec(seg_dur),
                "-i", str(persona_img)]
        persona_input = n_in
        n_in += 1

    # EQ needs an opaque white source to carry the bar color (showfreqs
    # output is grayed and used as the alpha instead); no -t: overlay ends
    # with its primary, and the output -t bounds everything anyway.
    eq_input = None
    if eq:
        cmd += ["-f", "lavfi",
                "-i", f"color=c=white:s={EQ_W}x{EQ_H}:r={FPS}"]
        eq_input = n_in
        n_in += 1

    parts = _bg_chain(bg_inputs, bg_start, seg_dur)
    if persona_input is not None:
        parts += _persona_chain(persona_input,
                                bbox=_alpha_bbox(persona_img))
    base = "mov" if persona_input is not None else "bg"
    if eq:
        parts += _eq_chain(base, eq_input, gain_db=gain_db)
        base = "mv"
    parts.append(f"[{base}]{ass_filter(ass_path, fonts_dir)}[vout]")
    cmd += ["-filter_complex", ";".join(parts)]
    if eq:
        # audible audio now lives inside the graph ([aud]); -af cannot be
        # combined with a graph that already consumes 0:a.
        cmd += ["-map", "[vout]", "-map", "[aud]", "-t", sec(seg_dur),
                "-ac", "2"]
    else:
        cmd += ["-map", "[vout]", "-map", "0:a", "-t", sec(seg_dur),
                "-af", f"volume={gain_db}dB,aresample=48000",
                "-ac", "2"]
    cmd += encode_args(nvenc=nvenc, crf=crf, preset=preset)
    cmd += ["-movflags", "+faststart", str(out)]
    return cmd


# ---- final concat -------------------------------------------------------

def concat_cmd(segments_list: str | Path, out: str | Path,
               *, remux_audio: bool = False) -> list[str]:
    cmd = ["ffmpeg", "-hide_banner", "-y", "-f", "concat", "-safe", "0",
           "-i", str(segments_list)]
    if remux_audio:
        cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-c", "copy"]
    cmd += ["-fflags", "+genpts", "-movflags", "+faststart", str(out)]
    return cmd


def segments_list_text(paths: list[str | Path]) -> str:
    from engine.util import ffconcat_path
    return "\n".join(f"file {ffconcat_path(p)}" for p in paths) + "\n"

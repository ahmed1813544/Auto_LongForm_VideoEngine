# Video Engine — long-form narrated story videos from plain text

Local CLI that turns your story text files into long-form YouTube narration
videos (30 min – 3 h): free Microsoft Edge TTS voiceover + burned-in
punch-in caption cards + a storyteller persona on the right + muted
licensed stock footage behind — 1080p30, one final mp4. Built and
battle-tested on Windows 11; dozens of ~30-minute videos rendered
end-to-end.

**A ~6,000-word script yields:** a ~30-minute video (~2.6 GB mp4), 800+
caption cards, paragraph-accurate chapter timestamps, and ~8–12 GB peak
scratch disk during the build (3–5 GB kept afterwards).

## How it works (two passes, audio is the master clock)

1. **Pass 1 — timeline:** parse stories → sentenceize → one free Edge TTS
   call per sentence (content-addressed `tts_cache/`, so reruns are free) →
   byte-exact float-ms caption timeline → `master.wav` + `captions.json`.
   Silences: 0.8 s preroll, 0.2 s inter-sentence, 1.2 s inter-story.
   Master is loudness-measured once (ebur128) → one fixed gain to −16 LUFS.
2. **Pass 2 — assembly:** mood search terms → stock search (Pixabay+Pexels,
   dedup, smallest ≥720p rendition — ~80 MB per-file cap) → download to a
   global `clip_library/` (resume-safe, ffprobe
   integrity gate) → normalize to 1920x1080@30 → deterministic scene plan
   (seeded RNG — same seed, same clips) → cut ≤9-min segments at story
   boundaries → render each with an identical encode contract →
   `concat -c copy` (no re-encode) → final mp4.

Every expensive stage is cached and resumable via `jobs/<id>/state.json`; a
fix at minute 25 re-renders one ~9-min segment, not the whole video.

## Requirements

- **Windows** (built/tested on 11; other OSes untested).
- **ffmpeg 8.x full build** on `PATH` — must include `libass`, `libx264`,
  `aac`; `h264_nvenc` optional for `--nvenc`.
- **Python 3.11+**: `pip install -r requirements.txt`
  (only `requests`, `python-dotenv`, `pytest` — everything else is ffmpeg
  subprocess work).
- **Voice needs no API key** — Microsoft Edge TTS (`edge-tts` package),
  free; 322 neural voices, `en-US-ChristopherNeural` by default.
- **Stock keys** in `.env` (git-ignored; copy `.env.example` → `.env`):
  `PIXABAY_API_KEY` and/or `PEXELS_API_KEY`. Without them, use
  `--skip-stock` (test-pattern background) to try the whole pipeline.
- **Disk:** ~10 GB free for a 30-minute video; ≥60 GB for a true 3-hour one.
  The scratch dir defaults to `C:\videoengine_work` — deliberately **outside
  OneDrive** (multi-GB churn in a synced folder causes lock/sync failures).
  Override with `VIDEOENGINE_WORK` in `.env` (keep it ASCII, no spaces).

## Setup

```powershell
git clone https://github.com/<you>/video-engine.git
cd video-engine
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env    # then paste YOUR stock keys into .env — never commit it

# health check: ffmpeg filters/encoders, disk, PNG validity, and with
# --live one cheap real call per API (Edge TTS synthesis + stock probes)
python -m engine doctor --live
```

## Quick start

```powershell
# 0. list the free Edge TTS voices (no key needed)
python -m engine tts-voices

# 1. hear a voice line before committing to a build
#    (default voice is en-US-ChristopherNeural; speed 0.5-2)
python -m engine tts-preview --text "My name is Gerald. I am seventy-four."

# 2. first watchable video, no stock keys needed (~2 min from sample.txt)
python -m engine build --script scripts/sample.txt `
    --persona-dir assets/rowe --skip-stock --job-id sample

# 3. real build: stock landscapes + audio-reactive EQ bars
python -m engine build --script scripts/mystory.txt `
    --mood "mountain mist sunrise aerial" --mood "green valley river drone" `
    --persona-dir assets/rowe --eq --job-id mystory --out output/mystory.mp4

# 4. deep health report on a built job (streams, durations, full-decode
#    scan, caption count)
python -m engine verify --job-id mystory --deep
```

Output lands in `<work>\output\<job>.mp4`. Re-running the same command
**resumes**: completed segments are skipped, corrupt/missing ones
re-rendered only.

## Web UI (browser)

Prefer clicking to typing?

```powershell
python -m engine ui            # opens http://127.0.0.1:8000
```

A local page (loopback-only — nothing leaves your machine, zero extra
deps): paste the script, pick moods / persona / voice (with an audio
preview), press **Build**, watch the live log + segment progress, open
the finished mp4 in one click. The **duration planner** turns a target
length (10 min – 3 h presets) into the word count you need, shows how
long your pasted script will actually run, and estimates disk usage
(warns when free space is too tight — greenery footage encodes ~2× the
bitrate) and render wall-time (CPU vs NVENC). Want your own face as the
storyteller? Upload any portrait photo in the UI — with optional
`pip install rembg onnxruntime` the background is AI-cut to transparency
automatically; uploads land in `<work>\ui_personas\`, never in the repo.
The UI runs the exact same CLI
as a child process — same caches, resume and QC gate — so mixing UI and
terminal builds is fine. Ctrl+C stops the page; running builds keep
going and the UI picks them back up when you reopen it. `--port N` for
another port, `--no-browser` to skip auto-open.

## Script format

Plain UTF-8 text. Multiple stories per file; delimiter is any of
`STORY: Title` / `===` / `---` / `***` (title optional), or the whole file
is one story if no delimiter:

```
STORY: My Own Son Sold My House
My name is Gerald. I am seventy-four years old. ...

---
STORY: The Heiress Returns
...
```

Blank-line-separated paragraphs double as chapter boundaries — keep them
where you'd want a YouTube chapter cut.

**Duration math:** ≈150–180 wpm → a 30-minute video needs ~5–6.5k words;
a true 3-hour video needs ~27k–33k words. The engine prints predicted
duration right after Pass 1, before any render minutes are spent.

**Write your own stories.** Narration scripts must be original work you own
(the sample is a short original excerpt).

## Personas

`--persona-dir` points at a folder of **transparent-background PNGs**
(one file = one pose). All PNGs across all persona dirs (sorted) form one
cycle: story *k* shows PNG `k % n`. Portraits work best full-height; they're
bbox-cropped and cover-scaled into the right-anchored 800×1080 column
(`assets/rowe/` is a working example — a Pexels portrait with the background
removed). Check alpha with:

```powershell
python -m engine doctor --persona-dir assets/rowe
```

## Content rules: the clip blocklist

The engine honors `<work>\clip_blocklist.json`:

```json
{"blocked": {"pexels:1234567": "city skyline", "pixabay:7654321": "person in frame"}}
```

Blocked clips never enter any job's scene pool, on every future run — this
is how a "landscapes only, no people/buildings/vehicles" rule survives
across thousands of downloaded clips. Build yours by contact-sheet vetting:
extract a mid frame from each normalized clip into a grid image, eyeball the
grid, blocklist the misses, and rebuild (clearing the plan keeps all caches).

## Build flags

| Flag | Meaning |
|---|---|
| `--script FILE...` | story text files (multiple files concatenate) |
| `--mood "..."` | search-terms round; repeat for variety (≤5 rounds) |
| `--persona-dir DIR` | transparent-PNG pose folder; repeat per persona |
| `--voice` | Edge TTS voice ShortName (default `en-US-ChristopherNeural` — see `tts-voices`) |
| `--speed` | TTS rate multiplier (default 1.0) |
| `--job-id` | resume identity (default: derived from script name) |
| `--seg-max-min` | max segment minutes (default 9) |
| `--seed` | scene-plan RNG seed — same seed = same clips (default 1234) |
| `--stock-max-h` | stock source height: 540 / 720 / 1080 (default 720 — smallest rendition that still looks right upscaled to the 1080p output; ~half the download bytes and faster decode. Files over ~80 MB get downgraded to a smaller rendition. 1080 = old full-HD-source behavior) |
| `--gain-db` / `--target-lufs` | override loudness gain |
| `--crf` / `--preset` | libx264 knobs (default 19 / veryfast) |
| `--nvenc` | GPU encode (`h264_nvenc` p4 cqp 23) — needs ~2 GB VRAM headroom |
| `--eq` | Audio-reactive equalizer bars behind the captions (driven by the narration itself) |
| `--skip-stock` | testsrc2 background (no stock keys required) |
| `--fresh` | wipe job state and re-render everything |
| `--dry-run` | print commands, render nothing |

Also: `engine stock-search --mood X --n 10` (preview what the APIs return
for a term), `engine clean` (prune scratch), `engine doctor` (health).

## Layout / look constants

- 1920x1080 @ 30 fps CFR, bt709, yuv420p; audio AAC 192k 48 kHz stereo
  (voice only — stock is muted).
- Captions: Montserrat ExtraBold 76 (bundled in `assets/fonts/`), white text,
  **no background box** — thick 5px pure-black outline + 3.5px semi-opaque
  drop shadow (BorderStyle 1). Every phrase punch-in animates: scales 72% →
  112% → settles 100% in the first ~180 ms, fades out over its last 110 ms.
  Full-line cards centered in the left column (`MarginR 840` keeps them
  clear of the figure). ≤8 words / ≤42 chars per card.
- Persona: right-anchored, 800×1080 column, 0.5 s fade-in on pose change.
- EQ (`--eq`): 64 white bars (1032×230 strip at the caption column's base,
  35% alpha) rendered under the captions and driven by the narration's own
  FFT (showfreqs, sqrt/log scale, +40 dB vis-only preamp). Same audio slice
  feeds both, so bars are sample-synced; silent gaps show a flat hairline.

## Troubleshooting

- `verify` complains `captions: N != M` → stale `.ass` from a changed plan;
  rerun the build (or `--fresh`).
- Mid-render failure → fix the cause, rerun the same command; the failed
  segment re-renders and concat rebuilds, completed segments are kept.
- Dense greenery encodes ~2× the bitrate of mixed scenery at crf 19
  (~12 Mbps, ~790 MB per 9-min segment) — budget disk accordingly.
- AAC seam click (rare): `concat_cmd(..., remux_audio=True)` in
  `engine/render/concat.py` re-encodes only the audio across the concat
  (minutes, not hours) — not wired as a CLI flag yet; segments themselves
  are sample-exact.
- Stock shortfall → `python -m engine stock-search --mood "storm night"`
  shows what the APIs return for a term before a build tries to use it.
- Voice synthesis failing en masse → Edge TTS rides an unofficial
  Microsoft endpoint; `pip install -U edge-tts` is the fix when Microsoft
  changes it (transient throttling is handled by retry+backoff).
- Pexels `HTTP 401: "Missing API key"` with a valid key → Pexels retired
  the `Access-Key` header (2026-09); the adapter authenticates with
  `Authorization`.

## Tests

```powershell
python -m pytest tests -q
```

Pure/offline: chunking math, ASS re-stamping, scene-planner determinism,
ffmpeg argv builders, stock-adapter response parsing, resume/state logic.

## Credits & license

- **Code:** MIT — see [LICENSE](LICENSE).
- **Voice:** Microsoft Edge TTS via the [`edge-tts`](https://pypi.org/project/edge-tts/)
  package — free, no API key; it uses Microsoft's read-aloud endpoint.
- **Caption font:** Montserrat ExtraBold — SIL Open Font License 1.1
  ([assets/fonts/OFL.txt](assets/fonts/OFL.txt)).
- **Sample persona:** portrait from [Pexels](https://www.pexels.com/) with
  the background removed, used under the Pexels License.
- **Footage** is downloaded at runtime from Pixabay/Pexels under their
  respective licenses. Both currently permit commercial use without
  attribution, but licenses change and individual clips can carry extra
  restrictions — review each provider's terms for your use case before
  monetizing.

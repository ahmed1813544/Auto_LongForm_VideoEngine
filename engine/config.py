"""Configuration: .env + environment + defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_WORK_DIR = r"C:\videoengine_work"

SAMPLE_RATE = 24000          # Edge TTS mp3 decoded to this; byte math everywhere
BYTES_PER_SEC = SAMPLE_RATE * 2  # s16 mono


@dataclass(frozen=True)
class Config:
    pixabay_key: str | None
    pexels_key: str | None
    work_dir: Path

    # ---- derived paths -------------------------------------------------
    @property
    def tts_cache_dir(self) -> Path:
        return self.work_dir / "tts_cache"

    @property
    def clips_raw_dir(self) -> Path:
        return self.work_dir / "clip_library" / "raw"

    @property
    def clips_norm_dir(self) -> Path:
        return self.work_dir / "clip_library" / "norm"

    @property
    def library_index_path(self) -> Path:
        return self.work_dir / "clip_library" / "library_index.json"

    @property
    def clip_blocklist_path(self) -> Path:
        return self.work_dir / "clip_blocklist.json"

    @property
    def jobs_dir(self) -> Path:
        return self.work_dir / "jobs"

    def ensure_dirs(self) -> None:
        for d in (self.work_dir, self.tts_cache_dir, self.clips_raw_dir,
                  self.clips_norm_dir, self.jobs_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ---- factory -------------------------------------------------------
    @classmethod
    def load(cls, dotenv_path: str | Path | None = None) -> "Config":
        load_dotenv(dotenv_path or PROJECT_ROOT / ".env", override=False)
        # also allow the env var set by the shell to win over a missing .env
        load_dotenv()
        work = os.getenv("VIDEOENGINE_WORK", "").strip() or DEFAULT_WORK_DIR
        return cls(
            pixabay_key=(os.getenv("PIXABAY_API_KEY") or "").strip() or None,
            pexels_key=(os.getenv("PEXELS_API_KEY") or "").strip() or None,
            work_dir=Path(work),
        )

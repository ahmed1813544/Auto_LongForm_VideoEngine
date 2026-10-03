"""Tiny logger: timestamped console lines, optionally mirrored to a job log."""
from __future__ import annotations

import sys
import time
from pathlib import Path


class Log:
    def __init__(self, path: str | Path | None = None, quiet: bool = False):
        self._fh = None
        self.quiet = quiet
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8")
            self._write("---- session ----")

    def _write(self, msg: str) -> None:
        if self._fh:
            self._fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
            self._fh.flush()

    def _emit(self, level: str, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {level} {msg}"
        self._write(msg)
        if not self.quiet:
            print(line, file=sys.stderr if level == "WARN" else sys.stdout,
                  flush=True)

    def info(self, msg: str) -> None:
        self._emit(" ", msg)

    def warn(self, msg: str) -> None:
        self._emit("WARN", msg)

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

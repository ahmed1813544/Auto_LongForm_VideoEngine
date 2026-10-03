"""Story script parsing.

Convention (all tolerant):
    STORY: The Heiress Returns     <- heading optional
    ...paragraphs...
    ---                            <- or bare delimiter (needs content after)
If no delimiter is present the whole file is one story.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_DELIM = re.compile(r"^\s*(?:STORY\s*[:#]|={3,}|-{3,}|\*{3,})\s*(.*)$", re.I)


@dataclass
class Story:
    idx: int
    title: str
    paragraphs: list[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return sum(len(p.split()) for p in self.paragraphs)


def read_text(path: str | Path) -> str:
    """UTF-8 with BOM/CRLF tolerance -> clean \\n text."""
    with open(path, "r", encoding="utf-8-sig", newline=None) as f:
        text = f.read()
    return text.replace("\r", "").replace("\x0b", "\n").replace("\x0c", "\n")


def split_stories(text: str) -> list[Story]:
    stories: list[Story] = []
    title: str | None = None
    blocks: list[str] = []

    def flush():
        if title is None and not any(b.strip() for b in blocks):
            return
        paras = [b.strip() for b in blocks if b.strip()]
        stories.append(Story(idx=len(stories),
                             title=(title or "").strip() or None,
                             paragraphs=paras))

    # split on blank-line paragraphs while watching for delimiter lines
    cur_para: list[str] = []

    def close_para():
        nonlocal cur_para
        if cur_para:
            blocks.append("\n".join(cur_para).strip())
            cur_para = []

    for line in text.split("\n"):
        m = _DELIM.match(line)
        if m and (m.group(1).strip() or blocks or cur_para):
            # a delimiter (with title, or after content) starts a new story
            close_para()
            flush()
            blocks = []
            title = m.group(1).strip() or None
            continue
        if not line.strip():
            close_para()
            continue
        cur_para.append(line)
    close_para()
    flush()

    # No delimiters found -> entire file is one story
    if not stories and text.strip():
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        first_line = text.strip().split("\n", 1)[0].strip()
        stories.append(Story(idx=0, title=first_line[:80] or None,
                             paragraphs=paras))
    # re-index
    for i, s in enumerate(stories):
        s.idx = i
    return stories


def parse_script(path: str | Path) -> list[Story]:
    return split_stories(read_text(path))


def total_words(stories: list[Story]) -> int:
    return sum(s.word_count for s in stories)

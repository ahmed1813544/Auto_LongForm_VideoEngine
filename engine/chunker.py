"""Sentenceizer + caption grouper — pure, deterministic, unit-tested.

Two unit types come out of one split (design decision):
  * AUDIO unit = one (possibly oversize-split) sentence -> one TTS call
  * CAPTION unit = word group inside a sentence, <=8 words / <=42 chars
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# terms whose trailing dot must not end a sentence
ABBREV = {
    "mr", "mrs", "ms", "dr", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "u.s", "u.k", "no", "gen", "hon", "prof", "inc", "ltd", "co", "ave",
    "blvd", "rd", "dept", "approx", "min", "max", "jan", "feb", "mar",
    "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
}

_SENT_END = re.compile(r'[.!?…]["\'”’»\)\]]*$')
_OPEN_STRIP = '"\'“”‘’(—-—–['
MAX_SENT_WORDS = 30
MAX_SENT_CHARS = 200
_MAX_SPLIT_RE = re.compile(r",\s+(?:and|but|so|yet)\b|[;:]", re.I)


@dataclass(frozen=True)
class Sentence:
    """One TTS audio unit."""
    story_idx: int
    sent_idx: int          # index within the story
    text: str


@dataclass(frozen=True)
class Unit:
    """One TTS audio unit plus its caption cards (in speech order)."""
    story_idx: int
    sent_idx: int
    text: str
    groups: list[str]
    ends_story: bool = False


def sentenceize(text: str) -> list[str]:
    """Split one paragraph/clean-text block into sentences."""
    toks = text.split()
    sents: list[str] = []
    cur: list[str] = []
    for i, tok in enumerate(toks):
        cur.append(tok)
        if not _SENT_END.search(tok):
            continue
        if i + 1 >= len(toks):
            break
        nxt = toks[i + 1].lstrip(_OPEN_STRIP)
        first = nxt[:1]
        if not (first.isupper() or first.isdigit()):
            continue
        # abbreviation / initial guard
        w = re.sub(r"[.\"'”’»\]\)]+$", "", tok).lower()
        if w in ABBREV or (len(w) == 1 and w.isalpha()):
            continue
        sents.append(" ".join(cur))
        cur = []
    if cur:
        sents.append(" ".join(cur))
    return sents


def split_oversize(sentence: str) -> list[str]:
    """Long sentences are awkward as one caption group AND as one TTS call;
    cut at the last natural pause (", and" / ";") before the limits."""
    out = [sentence]
    changed = True
    while changed:
        changed = False
        res = []
        for s in out:
            if len(s.split()) <= MAX_SENT_WORDS and len(s) <= MAX_SENT_CHARS:
                res.append(s)
                continue
            cut = None
            for m in _MAX_SPLIT_RE.finditer(s):
                if m.start() >= MAX_SENT_CHARS or \
                        len(s[:m.start()].split()) > MAX_SENT_WORDS:
                    break
                cut = m.start()
            if cut is None or cut < 20:
                # no pause available: hard cut at last space before word limit
                words = s.split()
                half = max(4, min(MAX_SENT_WORDS, len(words) // 2))
                res.append(" ".join(words[:half]).rstrip(" ,;:"))
                res.append(" ".join(words[half:]))
                changed = True
                continue
            res.append(s[:cut].rstrip(" ,;:") + ",")
            res.append(s[cut + 1:].strip())
            changed = True
        out = res
    return [o for o in out if o]


def caption_groups(sentence: str, max_words: int = 8,
                   max_chars: int = 42) -> list[str]:
    """Greedy pack words into caption cards, merging 1-word tails."""
    words = sentence.split()
    groups: list[str] = []
    cur: list[str] = []
    for w in words:
        cand = cur + [w]
        if cur and (len(cand) > max_words or
                    len(" ".join(cand)) > max_chars):
            groups.append(" ".join(cur))
            cur = [w]
        else:
            cur = cand
    if cur:
        groups.append(" ".join(cur))
    # a lonely 1-word tail is worse than a slightly long card -> merge always
    if len(groups) >= 2 and len(groups[-1].split()) == 1:
        groups[-2] = groups[-2] + " " + groups[-1]
        groups.pop()
    return groups


def build_units(stories) -> list["Unit"]:
    """Full mapping: stories -> TTS units, each with its caption groups."""
    from engine import tts_text

    units: list[Unit] = []
    for st in stories:
        sents: list[str] = []
        for para in st.paragraphs:
            for raw in sentenceize(tts_text.clean(para)):
                sents.extend(split_oversize(raw))
        for i, s in enumerate(sents):
            units.append(Unit(
                story_idx=st.idx,
                sent_idx=i,
                text=s,
                groups=caption_groups(s),
                ends_story=(i == len(sents) - 1),
            ))
    return units

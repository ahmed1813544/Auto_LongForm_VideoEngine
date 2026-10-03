"""TTS text normalization (applied BEFORE cache-keying, so keys stay stable)."""
from __future__ import annotations

import re
import unicodedata

_TABLE = {
    "‘": "'", "’": "'", "‚": "'",
    "“": '"', "”": '"', "„": '"',
    "—": ",", "–": ",",  # em/en dash -> comma pause for TTS
    "…": "...",
    " ": " ",
    "​": "", "‌": "", "‍": "",
}


def clean(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    for src, dst in _TABLE.items():
        t = t.replace(src, dst)
    # strip markdown emphasis / headings / backticks
    t = re.sub(r"[*_]{1,3}", "", t)
    t = re.sub(r"^#{1,6}\s+", "", t, flags=re.M)
    t = re.sub(r"`+", "", t)
    # strip html tags
    t = re.sub(r"<[^>\n]{1,80}>", " ", t)
    # collapse comma runs like ", ," created by dash replacement
    t = re.sub(r"(?:,\s*){2,}", ", ", t)
    t = re.sub(r",\s*\.", ".", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

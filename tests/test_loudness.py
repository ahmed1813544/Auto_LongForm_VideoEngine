"""Loudness math tests (pure string parsing + clamping)."""
from engine.loudness import (MAX_GAIN_DB, TARGET_LUFS, gain_db,
                             parse_integrated_lufs)

EBUR_SAMPLE = """
[Parsed_ebur128_0 @ 0x1234] t: 79.9  TARGET:-23 LUFS    M: -21.3 S: -20.8     I: -22.4 LUFS       LRA: 1.2 LU
[Parsed_ebur128_0 @ 0x1234] Summary:
[Parsed_ebur128_0 @ 0x1234]   I:     -18.7 LUFS
[Parsed_ebur128_0 @ 0x1234]   LRA:     5.3 LU
"""


def test_parse_takes_summary_value():
    assert parse_integrated_lufs(EBUR_SAMPLE) == -18.7


def test_parse_none_when_absent():
    assert parse_integrated_lufs("no numbers here") is None


def test_gain_basic_and_clamp():
    assert gain_db(-16.0, -18.7) == 2.7
    assert gain_db(-16.0, -40.0) == MAX_GAIN_DB      # clamped up
    assert gain_db(-16.0, 0.0) == -12.0              # clamped down (MIN)
    assert gain_db(TARGET_LUFS, None) == 0.0         # unmeasured -> untouched

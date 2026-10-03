from engine.timeline import B_PER_MS, _dur_ms, _split_groups


def test_byte_math():
    assert _dur_ms(48000) == 1000
    assert _dur_ms(48 * 1234) == 1234
    assert abs(_dur_ms(49) - 49 / 48) < 1e-9   # float, no truncation


def test_split_exact_partition():
    d = _split_groups(4000, [20, 10])       # 2:1 char weights
    assert sum(d) == 4000 and len(d) == 2


def test_split_single():
    assert _split_groups(3500, [40]) == [3500]


def test_split_floor_on_long_groups():
    d = _split_groups(3000, [30, 30])
    assert sum(d) == 3000
    assert all(x >= 900 for x in d)


def test_split_degenerate_small_sentence():
    # 3 groups in 1500ms: floors must scale down, total preserved,
    # no group below 300ms
    d = _split_groups(1500, [10, 10, 10])
    assert sum(d) == 1500
    assert all(300 <= x <= 700 for x in d)


def test_split_empty_weights():
    assert _split_groups(1000, []) == [1000]

import pytest

from engine.chunker import (caption_groups, sentenceize, split_oversize,
                            build_units)
from engine import script_parser, tts_text


def test_sentenceize_basic():
    out = sentenceize("He came home. She smiled! Did he notice? Yes, he did.")
    assert out == ["He came home.", "She smiled!",
                   "Did he notice?", "Yes, he did."]


def test_sentenceize_abbreviations():
    out = sentenceize("Mr. Smith arrived at 3.5 miles away. He waved.")
    assert out == ["Mr. Smith arrived at 3.5 miles away.", "He waved."]


def test_sentenceize_quotes():
    out = sentenceize('"Get out." He never spoke again.')
    assert out == ['"Get out."', "He never spoke again."]


def test_ellipsis():
    out = sentenceize("I waited... and then she came in. Finally.")
    assert out == ["I waited... and then she came in.", "Finally."]


def test_split_oversize():
    s = ("The lawyer read the entire contract aloud, and every single "
         "relative in the room turned pale, but my sister just laughed "
         "and asked for another copy of the ridiculous document today")
    parts = split_oversize(s)
    assert len(parts) > 1
    assert all(len(p.split()) <= 30 for p in parts)
    # every original word survives the split (punctuation may move)
    orig_words = {w.strip(",;:.!?") for w in s.split()}
    joined_words = {w.strip(",;:.!?") for p in parts for w in p.split()}
    assert orig_words <= joined_words


def test_caption_limits():
    s = ("The day my father handed the company to my brother, "
         "everything I had built quietly disappeared from every "
         "record of our family history")
    groups = caption_groups(s)
    for g in groups:
        assert len(g.split()) <= 8
        assert len(g) <= 42 + 4  # tolerance for punctuation joins
    assert " ".join(groups) == s


def test_caption_single_word_tail_merged():
    groups = caption_groups("One two three four five six seven eight nine")
    assert all(len(g.split()) >= 2 for g in groups)
    assert groups[-1].split()[-1] == "nine"


def test_tts_clean_dashes_quotes():
    assert tts_text.clean("He said — “leave” — and went") == \
        'He said , "leave" , and went'


def test_build_units_and_stories(tmp_path):
    src = tmp_path / "s.txt"
    src.write_text(
        "STORY: First\nA happened. B followed.\n\nMore here.\n"
        "STORY: Second\nC closed the door.\n", encoding="utf-8")
    stories = script_parser.parse_script(src)
    assert [s.title for s in stories] == ["First", "Second"]
    assert stories[0].word_count > 0
    units = build_units(stories)
    assert [u.story_idx for u in units] == [0, 0, 0, 1]
    assert units[2].ends_story and not units[1].ends_story


def test_no_delimiter_is_one_story(tmp_path):
    src = tmp_path / "s.txt"
    src.write_text("Just one long story.\nNo headers here.", encoding="utf-8")
    stories = script_parser.parse_script(src)
    assert len(stories) == 1

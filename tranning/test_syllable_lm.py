"""Tests for pinyin_units.py and syllable_lm.py (tiny hand-made corpora, no
dependence on the 12MB corpus)."""

from pinyin_units import PAUSE, char_syllables, number_to_chinese, strip_tone, to_syllables
from syllable_lm import SyllableLM


def test_numbers_are_read_in_chinese():
    assert number_to_chinese("10") == "十"
    assert number_to_chinese("205") == "二百零五"
    assert number_to_chinese("2026") == "二千零二十六"
    assert number_to_chinese("4.7") == "四點七"
    assert number_to_chinese("10000") == "一萬"


def test_syllables_tones_and_pauses():
    assert to_syllables("你好") == ["ni3", "hao3"]
    assert to_syllables("你好", tones=False) == ["ni", "hao"]
    # latin words are skipped, numbers spoken, punctuation -> one pause, no trailing pause
    assert to_syllables("hi，我有3個！", keep_pauses=True) == ["wo3", "you3", "san1", "ge4"]
    assert to_syllables("你好，再見。", keep_pauses=True) == ["ni3", "hao3", PAUSE, "zai4", "jian4"]
    assert strip_tone("lv4") == "lv"


def test_polyphones_use_context():
    assert dict(char_syllables("銀行"))["行"] == "hang"
    assert dict(char_syllables("行走"))["行"] == "xing"


def test_lm_picks_characters_from_context(tmp_path):
    # 是/事/市 all read "shi": the bigram decides
    lm = SyllableLM.train([("我是學生\n你是誰\n這件事\n台北市\n我是老師", 1)], min_bigram=1)
    assert lm.decode(["wo", "shi", "xue", "sheng"]) == "我是學生"
    assert lm.decode(["zhe", "jian", "shi"]) == "這件事"
    # unknown syllable stays as pinyin instead of a guessed character
    assert lm.decode(["wo", "shi", "zzz"]) == "我是zzz"
    lm.save(tmp_path)
    again = SyllableLM.load(tmp_path)
    assert again.decode(["tai", "bei", "shi"]) == "台北市"


def test_load_returns_none_when_not_built(tmp_path):
    assert SyllableLM.load(tmp_path) is None

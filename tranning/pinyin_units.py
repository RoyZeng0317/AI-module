"""Chinese text <-> pinyin syllables, shared by speech recognition and speech
synthesis.

Why syllables (2026-10-29, chosen by the user): a character-level speech
model has to tell ~3000+ characters apart from sound alone -- many of them
pronounced identically -- which needs far more recordings than this project
has (the first Common Voice run collapsed into printing one character).
Mandarin has only ~400 toneless syllables, so the acoustic model learns
"which syllable did I hear" (a sound question) and a separate statistical
language model (syllable_lm.py) decides "which character is meant" (a text
question, learned from text alone, of which there is plenty).

pypinyin is a pronunciation DICTIONARY (character/phrase -> pinyin lookup
tables), not an AI model, so it is the same kind of dependency as sympy in
calculus_solver.py -- Rule 06 is about not calling other AI models/APIs.

Syllables are written like pypinyin's Style.TONE3 ("ni3", "hao3", neutral
tone = 5); `strip_tone()` gives the toneless unit the recognizer uses.
"""

import re

from pypinyin import Style, lazy_pinyin

try:  # character-mapping dictionary (pip install opencc-python-reimplemented), not an AI model
    from opencc import OpenCC
    _T2S = OpenCC("t2s")
except ImportError:  # still works, just misreads some Traditional polyphones (銀行 -> yin xing)
    _T2S = None

CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_TOKEN_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]+|\d+(?:\.\d+)?|[A-Za-z]+|[^\s]")
PAUSE = "|"          # pause marker produced for punctuation (speech synthesis uses it)
_PUNCT = set("，。！？、；：,.!?;:…—～~()（）「」『』\"'“”‘’")

_DIGITS = "零一二三四五六七八九"
_UNITS = ["", "十", "百", "千"]


def _int_to_chinese(n: int) -> str:
    """0..99999999 the way it is read aloud (10 -> 十, 205 -> 二百零五)."""
    if n == 0:
        return "零"
    if n >= 10 ** 8:
        return "".join(_DIGITS[int(d)] for d in str(n))

    def below_10k(x: int) -> str:
        out, zero = "", False
        for i, d in enumerate(f"{x:04d}"):
            d = int(d)
            if d == 0:
                zero = bool(out)
                continue
            if zero:
                out += "零"
                zero = False
            out += _DIGITS[d] + _UNITS[3 - i]
        return out

    high, low = divmod(n, 10_000)
    text = (below_10k(high) + "萬" if high else "") + (("零" if high and low < 1000 and low else "") + below_10k(low) if low else "")
    if text.startswith("一十"):
        text = text[1:]
    return text


def number_to_chinese(token: str) -> str:
    """"4.7" -> 四點七, "2026" -> 二千零二十六."""
    whole, _, frac = token.partition(".")
    text = _int_to_chinese(int(whole)) if whole else "零"
    if frac:
        text += "點" + "".join(_DIGITS[int(d)] for d in frac)
    return text


def _pinyin(run: str, style) -> list[str]:
    """pypinyin's phrase table (which decides how a polyphone is read, e.g.
    行 in 銀行 vs 行走) is written in Simplified Chinese, so a Traditional run
    is converted to Simplified first, only for this lookup. The conversion
    is character-for-character; if a run ever changes length the original
    text is used so syllables never shift onto the wrong character."""
    lookup = run
    if _T2S is not None:
        simplified = _T2S.convert(run)
        if len(simplified) == len(run):
            lookup = simplified
    return lazy_pinyin(lookup, style=style, neutral_tone_with_five=True, errors="ignore")


def to_syllables(text: str, tones: bool = True, keep_pauses: bool = False) -> list[str]:
    """Pinyin syllables of the Chinese (and number) content of `text`.

    Numbers are read in Chinese. Latin words are skipped (no English
    pronunciation model here). With keep_pauses, punctuation becomes PAUSE.
    pypinyin picks each polyphonic character's reading from phrase context
    ("銀行" -> hang2, "行走" -> xing2), so whole runs are converted at once.
    """
    out: list[str] = []
    for token in _TOKEN_RE.findall(text):
        if token[0].isdigit():
            token = number_to_chinese(token)
        if CJK_RE.match(token):
            sylls = _pinyin(token, Style.TONE3)
            out.extend(s if tones else strip_tone(s) for s in sylls if s)
        elif keep_pauses and token in _PUNCT and out and out[-1] != PAUSE:
            out.append(PAUSE)
    while keep_pauses and out and out[-1] == PAUSE:
        out.pop()
    return out


def strip_tone(syllable: str) -> str:
    return syllable.rstrip("012345")


def char_syllables(text: str) -> list[tuple[str, str]]:
    """(character, toneless syllable) for every Chinese character in `text`,
    readings resolved in context -- what the language model learns from."""
    pairs = []
    for run in re.findall(r"[㐀-䶿一-鿿豈-﫿]+", text):
        sylls = _pinyin(run, Style.NORMAL)
        if len(sylls) == len(run):
            pairs.extend(zip(run, sylls))
    return pairs

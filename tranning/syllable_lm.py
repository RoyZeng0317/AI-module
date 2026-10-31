"""Syllables -> Chinese characters: a self-trained statistical language model.

The speech recognizer (speech_to_text.py) only says WHICH SYLLABLES it heard
("ni hao"); many characters share each syllable, so this model picks the
characters, the way a pinyin input method does. It is trained by counting,
on the project's own Chinese text (data/corpus_zh_starter.txt, the chat
data in data/pairs.json, and optionally the speech transcripts), nothing
downloaded or borrowed -- Rule 06.

    P(characters | syllables) ~ prod  P(c_i | c_{i-1}) * P(syllable_i | c_i)

* bigram P(c_i | c_{i-1}): interpolated with the unigram, so unseen pairs
  are possible but unlikely ("Jelinek-Mercer" smoothing, lambda = 0.8)
* P(syllable | c): how often this character is read as this syllable in
  the corpus (pypinyin's context-aware reading, see pinyin_units.py)
* Viterbi search over the top CANDIDATES characters of every syllable.

Chat sentences (data/pairs.json) are counted CHAT_WEIGHT times: they are what
people actually say to sinco, while the Wikipedia corpus mostly teaches
general word statistics.

Usage:
    python syllable_lm.py --build                          # ~2 min, writes syllable_lm_runs/lm.json.gz
    python syllable_lm.py --build --extra-text train.json  # also count speech transcripts
    python syllable_lm.py --decode "ni hao wo shi"
"""

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from pinyin_units import char_syllables

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DEFAULT_LM_DIR = HERE / "syllable_lm_runs"
DEFAULT_SOURCES = [DATA / "corpus_zh_starter.txt"]
CHAT_SOURCE = DATA / "pairs.json"
CHAT_WEIGHT = 20
LAMBDA = 0.8
CANDIDATES = 40         # "zhi" alone has 30+ common characters; 12 cut out 智
MIN_BIGRAM = 2          # bigrams seen once are dropped from the saved model (size)
BOS = "<s>"


class SyllableLM:
    def __init__(self, unigram: dict, bigram: dict, emission: dict):
        self.unigram = unigram                      # char -> count
        self.bigram = bigram                        # prev -> {char: count}
        self.emission = emission                    # syllable -> {char: count}
        self.total = sum(unigram.values()) or 1
        self.prev_totals = {p: sum(n.values()) for p, n in bigram.items()}
        self.char_totals = Counter()
        for chars in emission.values():
            for c, n in chars.items():
                self.char_totals[c] += n
        self._candidates = {
            s: sorted(chars, key=lambda c: -chars[c])[:CANDIDATES] for s, chars in emission.items()
        }

    # -- training ---------------------------------------------------------
    @classmethod
    def train(cls, texts: list[tuple[str, int]], min_bigram: int = MIN_BIGRAM) -> "SyllableLM":
        """texts: (text, weight) pairs. Bigrams counted fewer than min_bigram
        times are dropped (keeps the saved model small on a big corpus)."""
        unigram, emission = Counter(), defaultdict(Counter)
        bigram = defaultdict(Counter)
        for text, weight in texts:
            for line in text.splitlines():
                pairs = char_syllables(line)
                prev = BOS
                for char, syl in pairs:
                    unigram[char] += weight
                    emission[syl][char] += weight
                    bigram[prev][char] += weight
                    prev = char
        pruned = {p: {c: n for c, n in nxt.items() if n >= min_bigram} for p, nxt in bigram.items()}
        pruned = {p: nxt for p, nxt in pruned.items() if nxt}
        return cls(dict(unigram), pruned, {s: dict(c) for s, c in emission.items()})

    def save(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "lm.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump({"unigram": self.unigram, "bigram": self.bigram, "emission": self.emission}, f,
                      ensure_ascii=False)
        return path

    @classmethod
    def load(cls, out_dir: Path = DEFAULT_LM_DIR) -> "SyllableLM | None":
        path = Path(out_dir) / "lm.json.gz"
        if not path.exists():
            return None
        with gzip.open(path, "rt", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["unigram"], d["bigram"], d["emission"])

    # -- scoring ----------------------------------------------------------
    def _log_transition(self, prev: str, char: str) -> float:
        uni = (self.unigram.get(char, 0) + 1) / (self.total + len(self.unigram) + 1)
        nxt = self.bigram.get(prev)
        bi = nxt.get(char, 0) / self.prev_totals[prev] if nxt else 0.0
        return math.log(LAMBDA * bi + (1 - LAMBDA) * uni)

    def _log_emission(self, syllable: str, char: str) -> float:
        return math.log(self.emission[syllable][char] / self.char_totals[char])

    def decode(self, syllables: list[str]) -> str:
        """Most likely characters for toneless syllables. A syllable the
        model has never seen is kept as pinyin rather than guessed."""
        # Viterbi over (position, character) with back-pointers
        beams = {BOS: (0.0, [])}
        for syl in syllables:
            cands = self._candidates.get(syl)
            if not cands:
                beams = {k: (score, path + [syl]) for k, (score, path) in beams.items()}
                continue
            new = {}
            for c in cands:
                emit = self._log_emission(syl, c)
                best = max(((score + self._log_transition(prev, c) + emit, path)
                            for prev, (score, path) in beams.items()), key=lambda t: t[0])
                new[c] = (best[0], best[1] + [c])
            beams = new
        return "".join(max(beams.values(), key=lambda t: t[0])[1])


def build(sources: list[Path] = DEFAULT_SOURCES, chat_path: Path = CHAT_SOURCE,
          extra_text: list[Path] = (), out_dir: Path = DEFAULT_LM_DIR) -> SyllableLM:
    texts = []
    for src in sources:
        if Path(src).exists():
            texts.append((Path(src).read_text(encoding="utf-8"), 1))
    if Path(chat_path).exists():
        pairs = json.loads(Path(chat_path).read_text(encoding="utf-8"))
        texts.append(("\n".join(f'{p.get("prompt", "")}\n{p.get("reply", "")}' for p in pairs), CHAT_WEIGHT))
    for extra in extra_text:
        records = json.loads(Path(extra).read_text(encoding="utf-8"))
        texts.append(("\n".join(r["text"] for r in records), CHAT_WEIGHT))
    lm = SyllableLM.train(texts)
    lm.save(Path(out_dir))
    return lm


def main():
    ap = argparse.ArgumentParser(description="Build / query the syllable -> character language model")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--extra-text", type=Path, nargs="*", default=[],
                    help='speech manifests [{"audio":..., "text":...}] whose transcripts are also counted')
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_LM_DIR)
    ap.add_argument("--decode", help='space-separated toneless syllables, e.g. "ni hao"')
    args = ap.parse_args()
    if args.build:
        lm = build(extra_text=args.extra_text, out_dir=args.out_dir)
        print(f"built: {len(lm.unigram)} characters, {len(lm.emission)} syllables -> {args.out_dir}")
    if args.decode:
        lm = SyllableLM.load(args.out_dir)
        if lm is None:
            ap.error("no model yet: run --build first")
        print(lm.decode(args.decode.split()))


if __name__ == "__main__":
    main()

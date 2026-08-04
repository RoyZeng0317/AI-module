"""bpe_tokenizer.py — from-scratch byte-pair-encoding (BPE) tokenizer.

Why subword instead of chats.py's plain one-character-per-token split:
pretraining a Transformer needs the corpus to fit through the model as few
tokens as possible per unit of meaning — a character-level sequence for
Chinese/English text runs 2-4x longer than the equivalent subword sequence
(common multi-character words collapse into a single token), which directly
multiplies training time and eats into how much context fits in the
attention window on an 8GB card. Standard BPE (Sennrich et al., 2016),
implemented here from scratch — no `tokenizers`/`sentencepiece` dependency —
consistent with chats.py's own from-scratch tokenizer stance and Rule 06
(no external model/API, and a tokenizer library isn't one, but pulling in a
new heavy dependency for something this project already does by hand
elsewhere would be inconsistent with how every other module here works).

Base vocabulary is individual characters (same universe chats.py's
tokenize() already collects), not raw bytes — keeps behaviour easy to
reason about/debug and matches the rest of the project. Any character not
seen during training falls back to <unk> at encode time (same contract as
chats.py's encode()).

Pre-tokenization groups the input into merge-able "words" before BPE runs
within each word (this mirrors GPT-2's pre-tokenizer, simplified): runs of
Latin letters/digits are one word, runs of CJK characters are one word (so
adjacent Chinese characters like 學/習 CAN merge into 學習 if frequent
enough — treating every CJK character as its own isolated word, which a
naive whitespace-split pre-tokenizer would do, would make Chinese merging
impossible since BPE never merges across word boundaries), and everything
else (punctuation, whitespace, symbols) is kept as single-character words
so it never gets glued to real text.
"""

import json
import re
from collections import Counter
from pathlib import Path

PAD, SOS, EOS, UNK, SEP = 0, 1, 2, 3, 4
SPECIAL_TOKENS = {"<pad>": PAD, "<sos>": SOS, "<eos>": EOS, "<unk>": UNK, "<sep>": SEP}
# <sep> marks the prompt/reply boundary for transformer_chat.py's fine-tuning
# format (prompt_tokens + <sep> + reply_tokens + <eos>) — without an explicit
# boundary token, a causal LM given just a raw prompt at inference has no
# signal to switch from "continuing prompt-like text" to "starting a reply".

_WORD_RE = re.compile(r"[A-Za-z0-9]+|[一-鿿]+|.", re.DOTALL)

Pair = tuple[str, str]


def _pretokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _word_freqs(texts: list[str]) -> Counter:
    freqs: Counter = Counter()
    for text in texts:
        freqs.update(_pretokenize(text))
    return freqs


def _pair_counts(word_symbols: dict[str, tuple[str, ...]], word_freqs: Counter) -> Counter:
    counts: Counter = Counter()
    for word, freq in word_freqs.items():
        symbols = word_symbols[word]
        for a, b in zip(symbols, symbols[1:]):
            counts[(a, b)] += freq
    return counts


def _merge_word(symbols: tuple[str, ...], pair: Pair) -> tuple[str, ...]:
    merged = pair[0] + pair[1]
    out = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    def __init__(self, vocab: dict[str, int], merges: list[Pair]):
        self.vocab = vocab
        self.idx2tok = {i: t for t, i in vocab.items()}
        self.merges = merges
        # lower rank = learned earlier = applied first at encode time, same
        # convention as the reference BPE algorithm.
        self._merge_rank = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def train(cls, texts: list[str], vocab_size: int, min_pair_freq: int = 2) -> "BPETokenizer":
        """Learn merges from `texts` until either `vocab_size` tokens exist
        or no remaining adjacent pair occurs at least `min_pair_freq` times
        (stops the corpus from being forced to "merge" pure noise once the
        genuinely frequent patterns are exhausted).
        """
        vocab = dict(SPECIAL_TOKENS)
        word_freqs = _word_freqs(texts)
        word_symbols: dict[str, tuple[str, ...]] = {w: tuple(w) for w in word_freqs}
        for symbols in word_symbols.values():
            for ch in symbols:
                if ch not in vocab:
                    vocab[ch] = len(vocab)

        merges: list[Pair] = []
        while len(vocab) < vocab_size:
            counts = _pair_counts(word_symbols, word_freqs)
            if not counts:
                break
            best_pair, best_count = counts.most_common(1)[0]
            if best_count < min_pair_freq:
                break
            merged_token = best_pair[0] + best_pair[1]
            word_symbols = {w: _merge_word(s, best_pair) for w, s in word_symbols.items()}
            vocab[merged_token] = len(vocab)
            merges.append(best_pair)

        return cls(vocab, merges)

    def _apply_merges(self, symbols: tuple[str, ...]) -> tuple[str, ...]:
        symbols = list(symbols)
        while len(symbols) > 1:
            ranked = [
                (self._merge_rank[(a, b)], i)
                for i, (a, b) in enumerate(zip(symbols, symbols[1:]))
                if (a, b) in self._merge_rank
            ]
            if not ranked:
                break
            _, i = min(ranked)
            symbols[i:i + 2] = [symbols[i] + symbols[i + 1]]
        return tuple(symbols)

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for word in _pretokenize(text):
            tokens.extend(self._apply_merges(tuple(word)))
        return tokens

    def encode(self, text: str, max_len: int | None = None) -> list[int]:
        ids = [self.vocab.get(t, UNK) for t in self.tokenize(text)]
        if max_len is not None:
            ids = ids[: max_len - 1]
            ids.append(EOS)
            ids += [PAD] * (max_len - len(ids))
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.idx2tok.get(i, "") for i in ids if i not in (PAD, SOS, EOS, SEP))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "bpe_vocab.json").write_text(
            json.dumps(self.vocab, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / "bpe_merges.json").write_text(
            json.dumps(self.merges, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, out_dir: Path) -> "BPETokenizer":
        out_dir = Path(out_dir)
        vocab = json.loads((out_dir / "bpe_vocab.json").read_text(encoding="utf-8"))
        merges_raw = json.loads((out_dir / "bpe_merges.json").read_text(encoding="utf-8"))
        merges = [tuple(pair) for pair in merges_raw]
        return cls(vocab, merges)

"""Tests for bpe_tokenizer.py — BPE training, encode/decode round-trips,
and the save/load contract transformer_chat.py depends on (fine-tuning must
load the exact same vocab/merges the pretrained checkpoint used).
"""

from bpe_tokenizer import PAD, UNK, BPETokenizer


def _train_toy_tokenizer() -> BPETokenizer:
    texts = [
        "我是 sinco,一個自建的聊天模型,目前還在學習中",
        "你好,我是 sinco",
        "hello world, how are you today",
        "sinco sinco sinco",
    ] * 5
    return BPETokenizer.train(texts, vocab_size=150)


def test_train_produces_specials_plus_merges():
    tok = _train_toy_tokenizer()
    assert tok.vocab["<pad>"] == PAD
    assert tok.vocab["<unk>"] == UNK
    assert len(tok.merges) > 0
    assert tok.vocab_size > len(tok.merges)  # specials + base chars also count


def test_encode_decode_roundtrip():
    tok = _train_toy_tokenizer()
    text = "你好,我是 sinco"
    ids = tok.encode(text, max_len=40)
    assert tok.decode(ids) == text


def test_unseen_character_maps_to_unk():
    tok = _train_toy_tokenizer()
    assert tok.encode("龘") == [UNK]


def test_fixed_length_encode_is_padded():
    tok = _train_toy_tokenizer()
    ids = tok.encode("hi", max_len=20)
    assert len(ids) == 20
    assert ids[-1] == PAD


def test_save_and_load_roundtrip(tmp_path):
    tok = _train_toy_tokenizer()
    tok.save(tmp_path)
    loaded = BPETokenizer.load(tmp_path)

    assert loaded.vocab == tok.vocab
    assert loaded.merges == tok.merges
    text = "sinco sinco sinco"
    assert loaded.encode(text) == tok.encode(text)

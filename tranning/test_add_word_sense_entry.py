"""Engineering tests for manually appending word-sense entries; no real BGE weights needed."""
import json
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast
import tranning.add_word_sense_entry as add_entry_module
from tranning.add_word_sense_entry import add_entry, locate_span, next_manual_id


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    train = [dict(id="purchase_0", family="purchase", term="蘋果", text="我買蘋果是為了換掉舊手機。",
                  span=[2, 4], label="apple_phone", reply="指的是蘋果品牌手機。")]
    sft_train = [dict(prompt="請依語境說明「蘋果」的意思：我買蘋果是為了換掉舊手機。", reply="指的是蘋果品牌手機。")]
    (tmp_path / "train.json").write_text(json.dumps(train, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "sft_train.json").write_text(json.dumps(sft_train, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(add_entry_module, "TRAIN", tmp_path / "train.json")
    monkeypatch.setattr(add_entry_module, "SFT_TRAIN", tmp_path / "sft_train.json")
    monkeypatch.setattr(add_entry_module, "VECTORS", tmp_path / "train_vectors.json")
    return tmp_path


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_add_entry_encoder")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "蘋", "果", "水", "手", "機", "店", "買", "鼠"]
    (path / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(path / "vocab.txt"), do_lower_case=False)
    if tokenizer.vocab_size != len(vocab):
        tokenizer = BertTokenizerFast(vocab=str(path / "vocab.txt"))
    tokenizer.save_pretrained(path)
    torch.manual_seed(7)
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                         num_attention_heads=2, intermediate_size=24, max_position_embeddings=32)).save_pretrained(path)
    return path


def test_locate_span_single_occurrence():
    assert locate_span("我買蘋果。", "蘋果", None) == [2, 4]


def test_locate_span_multiple_requires_occurrence():
    with pytest.raises(ValueError, match="occurrence"):
        locate_span("蘋果店賣蘋果", "蘋果", None)
    assert locate_span("蘋果店賣蘋果", "蘋果", 1) == [4, 6]


def test_locate_span_term_not_found():
    with pytest.raises(ValueError, match="not found"):
        locate_span("我買蘋果。", "鼠", None)


def test_locate_span_occurrence_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        locate_span("我買蘋果。", "蘋果", 1)


def test_next_manual_id_counts_only_manual_prefix():
    rows = [dict(id="purchase_0"), dict(id="manual_0"), dict(id="manual_1")]
    assert next_manual_id(rows) == "manual_2"


def test_add_entry_appends_without_overwriting_existing_rows(data_dir):
    add_entry("我領養一隻鼠。", "鼠", "mouse_animal", compute_vector=False)
    rows = json.loads((data_dir / "train.json").read_text(encoding="utf-8"))
    assert len(rows) == 2 and rows[0]["id"] == "purchase_0" and rows[1]["id"] == "manual_0"
    pairs = json.loads((data_dir / "sft_train.json").read_text(encoding="utf-8"))
    assert len(pairs) == 2 and pairs[1]["reply"] == "指的是會活動的鼠類動物。"


def test_add_entry_uses_known_label_reply_by_default(data_dir):
    entry = add_entry("我領養一隻鼠。", "鼠", "mouse_animal", compute_vector=False)
    assert entry["reply"] == "指的是會活動的鼠類動物。"


def test_add_entry_rejects_duplicate_text_and_term(data_dir):
    with pytest.raises(ValueError, match="duplicate"):
        add_entry("我買蘋果是為了換掉舊手機。", "蘋果", "apple_phone", compute_vector=False)


def test_add_entry_rejects_unknown_label_without_new_label_flag(data_dir):
    with pytest.raises(ValueError, match="unknown label"):
        add_entry("我買了一輛腳踏車。", "腳踏車", "bicycle", compute_vector=False)


def test_add_entry_accepts_new_label_with_explicit_reply(data_dir):
    entry = add_entry("我買了一輛腳踏車。", "腳踏車", "bicycle", reply="指的是交通工具。",
                      new_label=True, compute_vector=False)
    assert entry["label"] == "bicycle" and entry["reply"] == "指的是交通工具。"


def test_add_entry_new_label_without_reply_raises(data_dir):
    with pytest.raises(ValueError, match="reply is required"):
        add_entry("我買了一輛腳踏車。", "腳踏車", "bicycle", new_label=True, compute_vector=False)


def test_add_entry_no_vector_skips_vector_file(data_dir):
    add_entry("我領養一隻鼠。", "鼠", "mouse_animal", compute_vector=False)
    assert not (data_dir / "train_vectors.json").exists()


def test_add_entry_computes_and_caches_vector(data_dir, tiny_model_dir):
    entry = add_entry("水果店買蘋果", "蘋果", "apple_fruit", model_dir=str(tiny_model_dir), max_length=16)
    store = json.loads((data_dir / "train_vectors.json").read_text(encoding="utf-8"))
    cached = store[entry["id"]]
    assert cached["term"] == "蘋果" and cached["span"] == entry["span"] and len(cached["vector"]) == 16


def test_add_entry_second_call_appends_to_existing_vector_store(data_dir, tiny_model_dir):
    first = add_entry("水果店買蘋果", "蘋果", "apple_fruit", model_dir=str(tiny_model_dir), max_length=16)
    second = add_entry("我領養一隻鼠。", "鼠", "mouse_animal", model_dir=str(tiny_model_dir), max_length=16)
    store = json.loads((data_dir / "train_vectors.json").read_text(encoding="utf-8"))
    assert set(store) == {first["id"], second["id"]}

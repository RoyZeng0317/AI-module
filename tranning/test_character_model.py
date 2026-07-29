"""Pipeline smoke test for character_model.py.

Does NOT claim any real temperament-profiling accuracy — there is no real
character-description dataset yet. Only proves the training loop (dataset
loading, model, BCE loss, early stopping, checkpoint + vocab/traits/config
output) and the build_character() inference + character-card-saving path
run end-to-end without crashing, using a handful of synthetic descriptions.
"""

import json

from character_model import build_character, train


def _make_synthetic_records():
    return [
        {"name": "阿凱", "description": "個性活潑開朗，講話很直接，偶爾有點衝動。",
         "traits": {"活潑": 0.9, "直接": 0.8, "沉穩": 0.1}},
        {"name": "小雨", "description": "說話輕聲細語，做事謹慎，很少主動開口。",
         "traits": {"溫柔": 0.8, "沉穩": 0.7, "活潑": 0.1}},
        {"name": "老王", "description": "沉穩內斂，遇事冷靜，說話不多但很有份量。",
         "traits": {"沉穩": 0.9, "冷靜": 0.8, "活潑": 0.0}},
        {"name": "小美", "description": "熱情活潑，喜歡開玩笑，跟誰都聊得來。",
         "traits": {"活潑": 0.9, "幽默": 0.7, "沉穩": 0.1}},
    ] * 5  # repeated so each batch has more than one distinct example


def test_training_and_build_character_run_end_to_end(tmp_path):
    records = _make_synthetic_records()
    train_manifest = tmp_path / "train.json"
    val_manifest = tmp_path / "val.json"
    train_manifest.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    val_manifest.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    out_dir = tmp_path / "runs"
    model, vocab, trait_vocab, history = train(
        train_manifest=train_manifest, val_manifest=val_manifest, out_dir=out_dir,
        epochs=3, batch_size=4, embed_size=16, hidden_size=32, lr=1e-2, max_len=40,
        dropout=0.1, weight_decay=1e-4, patience=5,
    )

    assert set(trait_vocab) == {"活潑", "直接", "沉穩", "溫柔", "冷靜", "幽默"}
    assert len(history) == 3
    assert (out_dir / "best_model.pt").exists()
    assert json.loads((out_dir / "vocab.json").read_text(encoding="utf-8")) == vocab
    assert json.loads((out_dir / "traits.json").read_text(encoding="utf-8")) == trait_vocab
    assert all(h["val_loss"] >= 0 for h in history)

    characters_dir = tmp_path / "characters"
    card = build_character("阿凱", "個性活潑開朗，講話很直接。", out_dir=out_dir, characters_dir=characters_dir)

    assert card["status"] is None
    assert set(card["traits"]) == set(trait_vocab)
    assert all(0.0 <= score <= 1.0 for score in card["traits"].values())
    assert isinstance(card["embedding"], list) and len(card["embedding"]) == 32 * 2
    assert card["voice_ref"] is None
    assert (characters_dir / "阿凱.json").exists()
    assert json.loads((characters_dir / "阿凱.json").read_text(encoding="utf-8")) == card


def test_build_character_without_checkpoint_returns_placeholder(tmp_path):
    card = build_character("測試", "隨便一段描述。", out_dir=tmp_path / "no_checkpoint_here")
    assert card["status"] is not None
    assert "尚未訓練" in card["status"]
    assert card["traits"] == {}
    assert card["embedding"] is None

"""Pipeline smoke test for reward_model.py.

Same contract as test_transformer_chat.py/test_train_grpo.py: no claim about
real-world reward quality on a tiny random-initialized model over a handful
of synthetic pairs, just that the pairwise-preference training loop
(load SFT checkpoint -> RewardModel -> Bradley-Terry loss -> checkpoint save)
runs end-to-end without crashing, plus one narrowly-scoped correctness
assertion (score(chosen) > score(rejected) on data designed to be trivially
separable) that this file is allowed to make because the synthetic data
guarantees it, not because it says anything about real preference quality.
"""

import json

import torch

from reward_model import (
    collate_pairs,
    load_pairs,
    load_reward_model,
    score,
    train_reward_model,
)
from transformer_chat import PAD, finetune, pretrain

_TINY_MODEL_KWARGS = dict(
    batch_size=4, block_size=24, n_layer=2, n_embd=16, n_head=2, vocab_size=80,
)


def _make_synthetic_corpus() -> str:
    return (
        "sinco 是一個從零打造的聊天模型。試問(35 + 45) * 12 =? 計算結果為 960。"
    ) * 20


def _make_synthetic_pairs() -> list[dict]:
    return [
        {"prompt": "試問(35 + 45) * 12 =?", "reply": "Final Answer: 960"},
        {"prompt": "hello", "reply": "hi there"},
    ] * 5


def _make_finetune_checkpoint(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(_make_synthetic_corpus(), encoding="utf-8")
    pretrain_dir = tmp_path / "pretrain_runs"
    pretrain(corpus_path=corpus_path, out_dir=pretrain_dir, epochs=1, val_split=0.2,
              patience=5, **_TINY_MODEL_KWARGS)

    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text(json.dumps(_make_synthetic_pairs()), encoding="utf-8")
    finetune_dir = tmp_path / "finetune_runs"
    finetune(data_path=pairs_path, pretrain_dir=pretrain_dir, out_dir=finetune_dir,
              epochs=1, batch_size=4, val_split=0.2, patience=5)
    return finetune_dir


def _make_reward_pairs_file(tmp_path):
    # chosen/rejected 用截然不同的固定字串重複多次，保證在幾個 epoch 內
    # pairwise accuracy 可以真的學起來，不是隨機猜測 -- 跟 test_train_grpo.py
    # 用重複算式當合成資料同一個道理。
    pairs = [
        {"prompt": "試問(35 + 45) * 12 =?", "chosen": "Final Answer: 960", "rejected": "我不知道"},
    ] * 8
    data_path = tmp_path / "reward_pairs.jsonl"
    data_path.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs) + "\n",
                          encoding="utf-8")
    return data_path


def test_load_pairs_reads_jsonl(tmp_path):
    data_path = _make_reward_pairs_file(tmp_path)
    pairs = load_pairs(data_path)
    assert len(pairs) == 8
    assert pairs[0] == {"prompt": "試問(35 + 45) * 12 =?", "chosen": "Final Answer: 960",
                         "rejected": "我不知道"}


def test_collate_pairs_pads_within_batch_and_last_real_token_index_is_correct():
    # 兩列長度不同（3 個 token vs 5 個 token），collate 後應該 pad 到當批最大長度，
    # 且「最後一個真實 token」的索引（reward_model.RewardModel.forward 用的同一
    # 條算式）要正確對到 padding 之前的最後一個真實位置。
    batch = [([1, 2, 3], [4, 5]), ([6, 7, 8, 9, 10], [11, 12, 13])]
    chosen, rejected = collate_pairs(batch)

    assert chosen.shape == (2, 5)
    assert rejected.shape == (2, 5)
    assert chosen[0].tolist() == [1, 2, 3, PAD, PAD]
    assert rejected[1].tolist() == [11, 12, 13, PAD, PAD]

    last_pos = (chosen != PAD).sum(dim=1).clamp(min=1) - 1
    assert last_pos.tolist() == [2, 4]  # row0 真實長度3->index2；row1 真實長度5->index4


def test_train_reward_model_runs_end_to_end(tmp_path):
    finetune_dir = _make_finetune_checkpoint(tmp_path)
    data_path = _make_reward_pairs_file(tmp_path)
    out_dir = tmp_path / "reward_runs"

    result = train_reward_model(
        data_path=data_path, base_dir=finetune_dir, out_dir=out_dir,
        epochs=15, batch_size=4, val_split=0.2, patience=15,
    )

    assert result is not None
    model, tokenizer, history = result
    assert len(history) > 0
    assert all("train_acc" in h and "val_acc" in h for h in history)
    assert (out_dir / "reward_model.pt").exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "bpe_vocab.json").exists()
    assert (out_dir / "bpe_merges.json").exists()
    assert (out_dir / "history.json").exists()


def test_reward_model_ranks_held_out_pair_correctly(tmp_path):
    finetune_dir = _make_finetune_checkpoint(tmp_path)
    data_path = _make_reward_pairs_file(tmp_path)
    out_dir = tmp_path / "reward_runs"

    train_reward_model(data_path=data_path, base_dir=finetune_dir, out_dir=out_dir,
                        epochs=15, batch_size=4, val_split=0.2, patience=15)

    loaded = load_reward_model(out_dir, device="cpu")
    assert loaded is not None
    model, tokenizer, block_size = loaded

    r_chosen = score(model, tokenizer, "試問(35 + 45) * 12 =?", "Final Answer: 960", block_size, "cpu")
    r_rejected = score(model, tokenizer, "試問(35 + 45) * 12 =?", "我不知道", block_size, "cpu")
    assert r_chosen > r_rejected


def test_train_reward_model_without_checkpoint_returns_none(tmp_path):
    data_path = _make_reward_pairs_file(tmp_path)
    result = train_reward_model(data_path=data_path, base_dir=tmp_path / "no_such_run",
                                 out_dir=tmp_path / "reward_runs", epochs=1)
    assert result is None

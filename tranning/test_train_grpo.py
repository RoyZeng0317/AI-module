"""Pipeline smoke test for train_grpo.py.

Same contract as test_transformer_chat.py: no claim about reply quality on a
tiny random-initialized model over a couple of synthetic examples, just that
the GRPO loop (rollout -> group-relative reward -> policy-gradient + KL
update -> checkpoint save) runs end-to-end without crashing on top of a
transformer_chat.py SFT checkpoint.
"""

import json

from reward_model import train_reward_model
from train_grpo import (
    accuracy_reward,
    grpo_train,
    json_format_reward,
    load_prompts,
)
from transformer_chat import finetune, pretrain

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


def _make_sft_agent_data(tmp_path):
    data_path = tmp_path / "sft_agent_data.jsonl"
    record = {
        "messages": [
            {"role": "system", "content": "你是一個支援 Tool Calling 的 AI 助手。"},
            {"role": "user", "content": "試問(35 + 45) * 12 =?"},
            {"role": "tool", "content": "960"},
            {"role": "assistant", "content": "Final Answer: (35 + 45) * 12 的計算結果是 960。"},
        ]
    }
    data_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    return data_path


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


def test_json_format_reward():
    assert json_format_reward('Action: ```json\n{"action": "calc"}\n```') == 1.0
    assert json_format_reward('Action: ```json\n{bad json}\n```') == -0.5
    assert json_format_reward("no tool call here") == -1.0


def test_accuracy_reward():
    assert accuracy_reward("答案是 960", "960") == 2.0
    assert accuracy_reward("答案是 42", "960") == 0.0


def test_load_prompts_reads_jsonl(tmp_path):
    data_path = _make_sft_agent_data(tmp_path)
    examples = load_prompts(data_path)
    assert examples == [{"prompt": "試問(35 + 45) * 12 =?", "target_answer": "960"}]


def test_grpo_train_runs_end_to_end(tmp_path):
    finetune_dir = _make_finetune_checkpoint(tmp_path)
    data_path = _make_sft_agent_data(tmp_path)
    out_dir = tmp_path / "grpo_runs"

    model, tokenizer, history = grpo_train(
        data_path=data_path, base_dir=finetune_dir, out_dir=out_dir,
        steps=3, group_size=2, max_new_tokens=6, save_every=1,
    )

    assert len(history) == 3
    assert all("mean_reward" in h for h in history)
    assert (out_dir / "model.pt").exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "bpe_vocab.json").exists()
    assert (out_dir / "history.json").exists()


def _make_reward_pairs_file(tmp_path):
    pairs = [
        {"prompt": "試問(35 + 45) * 12 =?", "chosen": "Final Answer: 960", "rejected": "我不知道"},
    ] * 8
    data_path = tmp_path / "reward_pairs.jsonl"
    data_path.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs) + "\n",
                          encoding="utf-8")
    return data_path


def test_grpo_train_with_learned_reward_mode_runs_end_to_end(tmp_path):
    """證明 --reward-mode learned 在 CLI/grpo_train() 這一層真的接得起來
    （不是只有 reward_model.py 自己的管線能跑），train_grpo.py 換一顆學習型
    reward model 一樣能跑完整個 GRPO rollout。"""
    finetune_dir = _make_finetune_checkpoint(tmp_path)
    reward_pairs_path = _make_reward_pairs_file(tmp_path)
    reward_model_dir = tmp_path / "reward_runs"
    train_reward_model(data_path=reward_pairs_path, base_dir=finetune_dir, out_dir=reward_model_dir,
                        epochs=3, batch_size=4, val_split=0.2, patience=5)

    data_path = _make_sft_agent_data(tmp_path)
    out_dir = tmp_path / "grpo_runs"

    model, tokenizer, history = grpo_train(
        data_path=data_path, base_dir=finetune_dir, out_dir=out_dir,
        steps=3, group_size=2, max_new_tokens=6, save_every=1,
        reward_mode="learned", reward_model_dir=reward_model_dir,
    )

    assert len(history) == 3


def test_grpo_train_without_checkpoint_returns_none(tmp_path):
    data_path = _make_sft_agent_data(tmp_path)
    result = grpo_train(data_path=data_path, base_dir=tmp_path / "no_such_run",
                         out_dir=tmp_path / "grpo_runs", steps=1)
    assert result is None

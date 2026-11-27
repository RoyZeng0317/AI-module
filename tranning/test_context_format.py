"""多輪串接與訓練契約；不以這些測試主張模型語意正確。"""
import json
from types import SimpleNamespace

import pytest
import torch

from bpe_tokenizer import BPETokenizer, EOS, PAD, SEP
from context_format import FORMAT_VERSION, build_prompt
from context_training import ContextDataset, collate_context, finetune_context
import transformer_chat as transformer


class CharTokenizer:
    def encode(self, text):
        return [ord(ch)+10 for ch in text]
    def decode(self, ids):
        return "".join(chr(i-10) for i in ids if i >= 10)


def test_prompt_crops_complete_old_pairs_without_mutating_history():
    tok = CharTokenizer()
    history = [["很長"*50, "舊答案"], ["我叫小明", "你好"]]
    current = build_prompt(tok, "我叫什麼", [history[-1]], 1000)
    assert build_prompt(tok, "我叫什麼", history, len(current)) == current
    assert history[0][0] == "很長"*50
    assert current[-1] == SEP
    assert "使用者：我叫小明\n助手：你好\n使用者：我叫什麼\n助手：" == tok.decode(current[:-1])


@pytest.mark.parametrize("message,history,budget", [("", None, 100), ("a", [["x"]], 100), ("a", [[1, "x"]], 100), ("a"*200, None, 40), ("a", None, True)])
def test_invalid_or_overlong_prompt_is_explicit(message, history, budget):
    with pytest.raises(ValueError):
        build_prompt(CharTokenizer(), message, history, budget)


def test_dataset_and_inference_prefix_match_mask_and_dynamic_padding(tmp_path, monkeypatch):
    tok, observed = CharTokenizer(), {}
    record = {"history": [["我叫小明", "你好"]], "prompt": "我叫什麼", "reply": "小明"}
    data = ContextDataset([record, dict(record, history=[], prompt="誰")], tok, 128, 16)
    x, y = data[0]
    prefix = build_prompt(tok, record["prompt"], record["history"], 112)
    assert x[:len(prefix)].tolist() == prefix
    assert torch.all(y[:len(prefix)-1] == PAD)
    assert y[len(prefix)-1:].tolist() == tok.encode("小明")+[EOS]
    batch_x, batch_y = collate_context([data[0], data[1]])
    assert batch_x.shape == batch_y.shape and batch_x.shape[1] < 128
    def generate(idx, **kwargs):
        observed.update(prefix=idx[0].tolist(), kwargs=kwargs)
        return torch.cat((idx, torch.tensor([tok.encode("好")], device=idx.device)), dim=1)
    model = SimpleNamespace(generate=generate)
    (tmp_path/"config.json").write_text(json.dumps({"context_format": FORMAT_VERSION, "response_budget": 16}))
    monkeypatch.setattr(transformer, "_load_gpt", lambda *a: (model, tok, 128, "cpu"))
    assert transformer.reply(record["prompt"], out_dir=tmp_path, history=record["history"], max_new_tokens=16) == "好"
    assert observed["prefix"] == prefix and observed["kwargs"]["penalty_generated_only"] is True
    with pytest.raises(ValueError):
        transformer.reply("a", out_dir=tmp_path, max_new_tokens=17)
    (tmp_path/"config.json").write_text(json.dumps({"context_format": FORMAT_VERSION, "response_budget": True}))
    with pytest.raises(ValueError):
        transformer.reply("a", out_dir=tmp_path, max_new_tokens=1)


def test_legacy_reply_prefix_and_kwargs_remain_compatible(tmp_path, monkeypatch):
    tok, seen = CharTokenizer(), {}
    def generate(idx, **kwargs):
        seen.update(prefix=idx[0].tolist(), kwargs=kwargs)
        return torch.cat((idx, torch.tensor([[ord("好")+10]])), dim=1)
    (tmp_path/"config.json").write_text("{}")
    monkeypatch.setattr(transformer, "_load_gpt", lambda *a: (SimpleNamespace(generate=generate), tok, 128, "cpu"))
    assert transformer.reply("你好", out_dir=tmp_path) == "好"
    assert seen["prefix"] == tok.encode("你好")+[SEP]
    assert "penalty_generated_only" not in seen["kwargs"]


def test_penalty_can_repeat_prompt_fact_without_penalizing_it():
    model = transformer.GPT(8, 8, 1, 8, 2, 0)
    def forward(idx, *args, **kwargs):
        logits = torch.full((*idx.shape, 8), -10.)
        logits[:, -1, 5], logits[:, -1, 6] = 2., 1.6
        return logits, None
    model.forward = forward
    prefix = torch.tensor([[5]])
    old = model.generate(prefix, 1, top_k=1, top_p=None, repetition_penalty=2)
    contextual = model.generate(prefix, 1, top_k=1, top_p=None, repetition_penalty=2, penalty_generated_only=True)
    assert old[0, -1].item() == 6 and contextual[0, -1].item() == 5
    repeated = model.generate(prefix, 2, top_k=1, top_p=None, repetition_penalty=2, penalty_generated_only=True)
    assert repeated[0].tolist() == [5, 5, 6]


def test_context_training_tiny_cpu_and_overlap_rejected(tmp_path):
    source = tmp_path/"source"; source.mkdir()
    tokenizer = BPETokenizer.train(["使用者：我叫小明\n助手：你好\n我叫什麼名字小明小安"], vocab_size=80)
    tokenizer.save(source)
    config = {"block_size": 96, "n_layer": 1, "n_embd": 16, "n_head": 2, "dropout": 0.0}
    (source/"config.json").write_text(json.dumps(config))
    torch.save(transformer.GPT(tokenizer.vocab_size, 96, 1, 16, 2, 0).state_dict(), source/"model.pt")
    records = [{"dialogue_id": "train", "history": [["我叫小明", "你好"]], "prompt": "我叫什麼", "reply": "小明"}]
    train, val = tmp_path/"train.json", tmp_path/"val.json"
    train.write_text(json.dumps(records), encoding="utf-8")
    val.write_text(json.dumps([dict(records[0], dialogue_id="val", history=[["我叫小安", "你好"]], reply="小安")]), encoding="utf-8")
    output=tmp_path/"candidate"
    _, _, history = finetune_context(train, val, source, output, epochs=1, batch_size=1,
                                     freeze_layers=0, response_budget=16, device="cpu")
    assert len(history) == 1
    metadata=json.loads((output/"config.json").read_text(encoding="utf-8"))
    assert metadata["context_format"] == FORMAT_VERSION and metadata["response_budget"] == 16
    assert metadata["context_training"]["best_epoch"] == 1
    assert transformer.reply("我叫什麼", out_dir=output, history=[["我叫小安", "你好"]], max_new_tokens=5, device="cpu")
    with pytest.raises(ValueError, match="不可覆寫"):
        finetune_context(train, val, source, output)
    val.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError, match="同一段對話"):
        finetune_context(train, val, source, tmp_path/"bad")

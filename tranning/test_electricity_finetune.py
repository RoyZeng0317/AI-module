"""電學選項判別試驗的工程測試：微型 GPT 與合成 tokenizer，不代表任何電學能力。"""
import json

import pytest

torch = pytest.importorskip("torch")
import electricity_dataset as data
import electricity_finetune as finetune
import electricity_practice as practice
import transformer_chat as transformer
from bpe_tokenizer import BPETokenizer, SEP
from test_electricity_dataset import reviewed_sources  # noqa: F401  (pytest fixture)
from test_electricity_practice import dataset  # noqa: F401  (pytest fixture)


def make_checkpoint(root, dataset_path):
    """用資料集題目文字訓練微型 tokenizer（含 A–D），存成與正式 checkpoint 同結構。"""
    texts = ["ABCD 答案：", "ABCD"] * 5
    for split in data.SPLITS:
        for row in data.load_questions(dataset_path, split)[1]:
            texts += [practice.format_question(row) + finetune.ANSWER_PREFIX] * 3
    tokenizer = BPETokenizer.train(texts, vocab_size=400, verbose=False)
    out = root / "ckpt"
    tokenizer.save(out)
    config = {"block_size": 256, "n_layer": 3, "n_embd": 32, "n_head": 2, "dropout": 0.0}
    (out / "config.json").write_text(json.dumps(config), encoding="utf-8")
    torch.manual_seed(0)
    model = transformer.GPT(tokenizer.vocab_size, 256, 3, 32, 2, 0.0)
    torch.save(model.state_dict(), out / "model.pt")
    return out


@pytest.fixture
def fixture(dataset, tmp_path):  # noqa: F811
    return dataset, make_checkpoint(tmp_path, dataset), tmp_path / "out"


def test_refuses_without_explicit_approval(fixture):
    dataset_path, checkpoint, out = fixture
    with pytest.raises(ValueError, match="核准"):
        finetune.run(approved=False, checkpoint=checkpoint, dataset=dataset_path, out_dir=out, device="cpu", gpu_cap=False)
    assert not out.exists()


def test_prompt_has_no_gold_and_only_fixed_prefix(fixture):
    dataset_path, checkpoint, _ = fixture
    tokenizer = BPETokenizer.load(checkpoint)
    _, questions = data.load_questions(dataset_path, "train")
    ids = finetune.build_prompt_ids(tokenizer, questions[0], SEP)
    assert ids[-1] == SEP
    assert tokenizer.decode(ids[:-1]) == practice.format_question(questions[0]) + finetune.ANSWER_PREFIX
    assert len(set(finetune.letter_ids(tokenizer))) == 4


def test_only_last_block_and_ln_f_train(fixture):
    _, checkpoint, _ = fixture
    model, *_ = transformer._load_gpt(checkpoint, "cpu")
    params = finetune.freeze_all_but_last(model)
    expected = {id(p) for p in list(model.blocks[-1].parameters()) + list(model.ln_f.parameters())}
    assert {id(p) for p in params} == expected
    assert not model.tok_emb.weight.requires_grad and not model.head.weight.requires_grad


def test_run_never_touches_base_or_test_and_reports(fixture, monkeypatch):
    dataset_path, checkpoint, out = fixture
    before = {p.name: data.file_hash(p) for p in checkpoint.iterdir()}
    real_load = data.load_questions
    seen = []
    monkeypatch.setattr(data, "load_questions", lambda d, s: (seen.append(s), real_load(d, s))[1])
    report = finetune.run(approved=True, checkpoint=checkpoint, dataset=dataset_path, out_dir=out, device="cpu", gpu_cap=False,
                          hyper={"lr": 1e-3, "max_epochs": 2})
    assert "test" not in seen
    assert {p.name: data.file_hash(p) for p in checkpoint.iterdir()} == before
    assert (out / "report.json").is_file()
    assert report["baseline"]["dev_n"] == len(real_load(dataset_path, "dev")[1])
    assert len(report["chat_retention"]["before"]) == 3 == len(report["chat_retention"]["after"])
    assert report["published_weights"] == (report["best_epoch"] > 0) == (out / "model.pt").exists()


def test_existing_output_directory_is_not_overwritten(fixture):
    dataset_path, checkpoint, out = fixture
    out.mkdir()
    with pytest.raises(ValueError, match="已存在"):
        finetune.run(approved=True, checkpoint=checkpoint, dataset=dataset_path, out_dir=out, device="cpu", gpu_cap=False)


def test_wrong_expected_hash_stops_before_training(fixture):
    dataset_path, checkpoint, out = fixture
    with pytest.raises(ValueError, match="hash"):
        finetune.run(approved=True, checkpoint=checkpoint, dataset=dataset_path, out_dir=out, device="cpu",
                     gpu_cap=False, expect_model_sha="0" * 64)
    assert not out.exists()

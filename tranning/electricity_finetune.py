"""基本電學選項判別小型試驗（PILOT_PROPOSAL.md）：只更新最後 block 與 ln_f。

輸入只含題目、四候選與固定「答案：」前綴；正確選項只當 4 類 cross-entropy 的 label，
不進 prompt。只讀 train/dev 的評分鍵，test 永不載入。必須明確 --approved 才會更新權重，
輸出到未存在的新目錄，不覆寫或啟用正式模型。
"""
import argparse, json, math, random, shutil
from pathlib import Path

import torch
import torch.nn.functional as F

import electricity_dataset as data
import electricity_practice as practice

DEFAULT_OUT = data.ROOT / "tranning/gpt_electricity_runs/pilot_v1"
ANSWER_PREFIX = "\n答案："
HYPER = {"lr": 1e-5, "weight_decay": .01, "grad_clip": 1.0, "seed": 43, "batch": 2, "accum": 4,
         "max_epochs": 3, "max_steps": 9, "gpu_fraction": .75, "peak_gib": 6.0, "trainable_blocks": 1}
CHAT_PROMPTS = ("你好", "你是誰？", "how are you?")
COPY_FILES = ("config.json", "bpe_vocab.json", "bpe_merges.json")


def build_prompt_ids(tokenizer, question, sep):
    """固定前綴結尾再接 SEP；不含任何候選的標記。"""
    ids = tokenizer.encode(practice.format_question(question) + ANSWER_PREFIX)
    return ids + [sep]


def letter_ids(tokenizer):
    """A–D 必須各自為單一 token，否則受限解碼的 4 類比較沒有意義。"""
    ids = [tokenizer.encode(letter) for letter in "ABCD"]
    if any(len(item) != 1 for item in ids) or len({item[0] for item in ids}) != 4:
        raise ValueError("A–D 無法各自編為不同的單一 token")
    return [item[0] for item in ids]


def option_logits(model, prompts, choices, device):
    """每題最後位置只比較 A–D 四個 token 的 logit；左側補 PAD 以免注意到 padding。"""
    from bpe_tokenizer import PAD
    width = max(len(p) for p in prompts)
    batch = torch.full((len(prompts), width), PAD, dtype=torch.long)
    for row, prompt in enumerate(prompts):
        batch[row, :len(prompt)] = torch.tensor(prompt)
    batch = batch.to(device)
    logits, _ = model(batch)
    last = torch.tensor([len(p) - 1 for p in prompts], device=device)
    picked = logits[torch.arange(len(prompts), device=device), last]
    return picked[:, choices]


def evaluate(model, examples, choices, device):
    """不更新權重；回傳 (答對題數, 平均 loss)。"""
    model.eval()
    correct, total_loss = 0, 0.0
    with torch.no_grad():
        for prompt, label in examples:
            scores = option_logits(model, [prompt], choices, device)
            total_loss += F.cross_entropy(scores, torch.tensor([label], device=device)).item()
            correct += int(scores.argmax(1).item() == label)
    return correct, total_loss / max(1, len(examples))


def freeze_all_but_last(model, trainable_blocks=1):
    for param in model.parameters():
        param.requires_grad = False
    for module in list(model.blocks[-trainable_blocks:]) + [model.ln_f]:
        for param in module.parameters():
            param.requires_grad = True
    # head 與 tok_emb 權重綁定：tok_emb 已凍結，head 因而一併凍結。
    return [p for p in model.parameters() if p.requires_grad]


def load_examples(dataset, split, tokenizer, sep, choices, block_size, keys):
    _, questions = data.load_questions(dataset, split)
    examples = []
    for question in sorted(questions, key=lambda row: row["id"]):
        key = keys[question["id"]]
        if key.get("kind") != "mcq" or key.get("reviewed") is not True or key["question_sha256"] != question["question_sha256"]:
            raise ValueError(f"評分鍵不符或未審核：{question['id']}")
        prompt = build_prompt_ids(tokenizer, question, sep)
        from bpe_tokenizer import UNK
        if UNK in prompt or len(prompt) > block_size:
            raise ValueError(f"題目含未知字元或超過 block_size，不截斷：{question['id']}")
        examples.append((prompt, "ABCD".index(key["correct_option"])))
    return examples


def chat_retention(model, tokenizer, block_size, device, sep, seed):
    """與基準相同的生成參數，保存原文供人工比較；不評分。"""
    outputs = []
    model.eval()
    for index, text in enumerate(CHAT_PROMPTS):
        random.seed(seed + index); torch.manual_seed(seed + index)
        ids = (tokenizer.encode(text) + [sep])[-block_size:]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        params = practice.PARAMETERS
        out = model.generate(idx, max_new_tokens=params["max_new_tokens"], temperature=params["temperature"],
                             top_k=params["top_k"], top_p=params["top_p"], repetition_penalty=params["repetition_penalty"])
        outputs.append({"prompt": text, "raw_completion": tokenizer.decode(out[0, len(ids):].tolist()) or "..."})
    return outputs


def run(*, approved, checkpoint=practice.DEFAULT_CHECKPOINT, dataset=data.DEFAULT_DATASET, out_dir=DEFAULT_OUT,
        expect_dataset_sha=None, expect_model_sha=None, device=None, hyper=None, gpu_cap=True):
    if approved is not True:
        raise ValueError("未取得本次訓練的明確核准（待辦 #43）；拒絕更新權重")
    hp = {**HYPER, **(hyper or {})}
    checkpoint, dataset, out_dir = (data.inside_project(p) for p in (checkpoint, dataset, out_dir))
    if out_dir.exists():
        raise ValueError(f"輸出目錄已存在，拒絕覆寫：{out_dir}")
    if out_dir == checkpoint:
        raise ValueError("輸出不可是正式 checkpoint")
    if expect_dataset_sha and data.file_hash(dataset / "manifest.json") != expect_dataset_sha:
        raise ValueError("資料 manifest hash 不符")
    protected = {name: data.file_hash(checkpoint / name) for name in ("model.pt",) + COPY_FILES}
    if expect_model_sha and protected["model.pt"] != expect_model_sha:
        raise ValueError("基底 model.pt hash 不符")

    import transformer_chat as transformer
    from bpe_tokenizer import SEP
    device = transformer._resolve_device(device)
    if gpu_cap:
        transformer._cap_gpu_memory(device, hp["gpu_fraction"])
    loaded = transformer._load_gpt(checkpoint, device)
    if loaded is None:
        raise ValueError("無法載入基底 GPT")
    model, tokenizer, block_size, device = loaded
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if config.get("context_format") not in (None, "legacy"):
        raise ValueError("本試驗僅支援 legacy GPT")
    random.seed(hp["seed"]); torch.manual_seed(hp["seed"])
    choices = letter_ids(tokenizer)
    keys = practice._load_keys(dataset)  # 只取 train/dev 的鍵；test 不進 load_examples。
    train = load_examples(dataset, "train", tokenizer, SEP, choices, block_size, keys)
    dev = load_examples(dataset, "dev", tokenizer, SEP, choices, block_size, keys)

    params = freeze_all_but_last(model, hp["trainable_blocks"])
    optimizer = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=hp["weight_decay"])
    retention_before = chat_retention(model, tokenizer, block_size, device, SEP, hp["seed"])
    base_correct, base_loss = evaluate(model, dev, choices, device)
    train_base = evaluate(model, train, choices, device)
    best = {"epoch": 0, "dev_correct": base_correct, "dev_loss": base_loss, "state": None}
    history, steps, stop_reason = [], 0, "max_epochs"
    rng = random.Random(hp["seed"])

    def step():
        nonlocal steps
        grad = torch.nn.utils.clip_grad_norm_(params, hp["grad_clip"])
        if not math.isfinite(float(grad)):
            raise FloatingPointError("梯度非有限值，停止")
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        steps += 1
        if device == "cuda" and torch.cuda.max_memory_allocated() > hp["peak_gib"] * 2 ** 30:
            raise MemoryError("GPU 峰值超過上限，停止")

    for epoch in range(1, hp["max_epochs"] + 1):
        model.train()
        order = list(range(len(train))); rng.shuffle(order)
        batches = [order[i:i + hp["batch"]] for i in range(0, len(order), hp["batch"])]
        losses, pending = [], 0
        for batch in batches:
            scores = option_logits(model, [train[i][0] for i in batch], choices, device)
            loss = F.cross_entropy(scores, torch.tensor([train[i][1] for i in batch], device=device))
            if not math.isfinite(loss.item()):
                raise FloatingPointError("loss 非有限值，停止")
            (loss / hp["accum"]).backward()
            losses.append(loss.item()); pending += 1
            if pending == hp["accum"] and steps < hp["max_steps"]:
                step(); pending = 0
        if pending and steps < hp["max_steps"]:
            step()
        correct, dev_loss = evaluate(model, dev, choices, device)
        history.append({"epoch": epoch, "steps": steps, "train_loss": sum(losses) / len(losses),
                        "dev_correct": correct, "dev_loss": dev_loss})
        if correct < best["dev_correct"]:
            stop_reason = "dev_decline"; break
        if (correct, -dev_loss) > (best["dev_correct"], -best["dev_loss"]):
            best = {"epoch": epoch, "dev_correct": correct, "dev_loss": dev_loss,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        if steps >= hp["max_steps"]:
            stop_reason = "max_steps"; break

    published = best["state"] is not None
    if published:
        model.load_state_dict(best["state"])
    retention_after = chat_retention(model, tokenizer, block_size, device, SEP, hp["seed"])
    out_dir.mkdir(parents=True)
    if published:
        torch.save(best["state"], out_dir / "model.pt")
        for name in COPY_FILES:
            shutil.copy2(checkpoint / name, out_dir / name)
    report = {"schema_version": 1, "approved": True, "hyper": hp, "stop_reason": stop_reason,
              "dataset_sha256": data.file_hash(dataset / "manifest.json"), "base_files": protected,
              "baseline": {"dev_correct": base_correct, "dev_n": len(dev), "dev_loss": base_loss,
                           "train_correct": train_base[0], "train_n": len(train), "train_loss": train_base[1]},
              "history": history, "best_epoch": best["epoch"], "published_weights": published,
              "note": "選項判別可行性試驗；非電學解題能力驗收，test 未使用。",
              "chat_retention": {"before": retention_before, "after": retention_after}}
    data.write_json(out_dir / "report.json", report)
    if any(data.file_hash(checkpoint / name) != digest for name, digest in protected.items()):
        raise ValueError("正式 checkpoint 在訓練期間被改動")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved", action="store_true", help="使用者已逐次核准此次訓練")
    parser.add_argument("--checkpoint", type=Path, default=practice.DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=data.DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--expect-dataset-sha")
    parser.add_argument("--expect-model-sha")
    args = parser.parse_args()
    report = run(approved=args.approved, checkpoint=args.checkpoint, dataset=args.dataset, out_dir=args.out_dir,
                 expect_dataset_sha=args.expect_dataset_sha, expect_model_sha=args.expect_model_sha)
    print(json.dumps({k: report[k] for k in ("stop_reason", "baseline", "history", "published_weights")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

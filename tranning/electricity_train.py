"""基本電學四選一 pilot：preflight 唯讀；train 須依 #43 另行核准。"""
import argparse, copy, json, math, random, shutil, sys
from pathlib import Path

import torch
import torch.nn.functional as F
import electricity_dataset as data
import electricity_practice as practice
from bpe_tokenizer import PAD, SEP, UNK

PREFIX = "答案："
DEFAULT_PREFLIGHT = data.ROOT / "output/basic-electricity/classification-baseline"
POLICY = {"lr": 1e-5, "batch_size": 2, "accumulation_steps": 4, "epochs": 3, "max_steps": 9,
          "gpu_mem_fraction": .75, "weight_decay": .01, "clip_grad_norm": 1.0, "seed": 43}
MAX_MEMORY = 6 * 1024 ** 3


class PilotStop(RuntimeError):
    pass


def policy_checked(overrides=None):
    policy = {**POLICY, **(overrides or {})}
    if set(policy) != set(POLICY):
        raise ValueError("不支援的 pilot 參數")
    for name in ("batch_size", "epochs", "max_steps"):
        if type(policy[name]) is not int or not 1 <= policy[name] <= POLICY[name]:
            raise ValueError(f"{name} 超出 pilot 上限")
    if policy["accumulation_steps"] != 4 or type(policy["seed"]) is not int:
        raise ValueError("固定累積4次，seed須為整數")
    for name in ("lr", "gpu_mem_fraction", "weight_decay", "clip_grad_norm"):
        if not isinstance(policy[name], (int, float)) or not math.isfinite(policy[name]) or not 0 < policy[name] <= POLICY[name]:
            raise ValueError(f"{name} 超出 pilot 上限")
    return policy


def load_pilot_data(directory):
    """student 只讀 train/dev；不開 questions/test.jsonl。"""
    directory = data.inside_project(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("test_frozen") is not True:
        raise ValueError("資料 schema／test 凍結狀態不符")
    rows, groups = [], {}
    for split in ("train", "dev"):
        path = directory / f"questions/{split}.jsonl"
        if data.file_hash(path) != manifest["files"].get(f"questions/{split}.jsonl"):
            raise ValueError("train/dev 資料 hash 不符")
        subset = data.read_jsonl(path)
        if not subset:
            raise ValueError("train/dev 不可為空")
        for row in subset:
            if (set(row) - data.QUESTION_FIELDS or row.get("split") != split or row.get("kind") != "mcq"
                    or row.get("question_sha256") != data.question_digest(row) or set(row.get("options", {})) != set("ABCD")):
                raise ValueError("題目 schema／來源／split 不符")
        rows.extend(subset)
        groups[split] = {row["group_id"] for row in subset}
    if groups["train"] & groups["dev"] or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("train/dev 題族或 id 重疊")
    return manifest, rows


def evaluator_labels(directory, rows, manifest):
    """評分端可讀原單一 keys 檔，但只回傳 train/dev 的類別 label。"""
    path = Path(directory) / "evaluator/keys.jsonl"
    if data.file_hash(path) != manifest["files"].get("evaluator/keys.jsonl"):
        raise ValueError("評分鍵 hash 不符")
    selected = {row["id"]: row for row in rows}
    labels = {}
    for key in data.read_jsonl(path):
        if key.get("id") not in selected:
            continue
        row = selected[key["id"]]
        if (key.get("id") in labels or key.get("reviewed") is not True or key.get("kind") != "mcq"
                or key.get("split") != row["split"] or key.get("question_sha256") != row["question_sha256"]
                or key.get("correct_option") not in set("ABCD")):
            raise ValueError("評分鍵無效或與題目不符")
        labels[key["id"]] = "ABCD".index(key["correct_option"])
    if set(labels) != set(selected):
        raise ValueError("缺少 train/dev 評分鍵")
    return labels


def configure_trainable(model):
    if model.head.weight is not model.tok_emb.weight or not len(model.blocks):
        raise ValueError("需要既有 tied-head GPT 架構")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.blocks[-1], model.ln_f):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def encode_questions(rows, tokenizer, block_size):
    choice_ids = [tokenizer.encode(letter) for letter in "ABCD"]
    if any(len(ids) != 1 or ids[0] == UNK or tokenizer.decode(ids) != letter for ids, letter in zip(choice_ids, "ABCD")) or len({ids[0] for ids in choice_ids}) != 4:
        raise ValueError("A-D 必須分別是唯一、可還原的單一 BPE token")
    prefix_ids = tokenizer.encode(PREFIX)
    encoded = []
    for row in rows:
        prompt = practice.format_question(row)
        ids = tokenizer.encode(prompt) + [SEP] + prefix_ids
        if UNK in ids or len(ids) + 1 > block_size:
            raise ValueError(f"完整題目／固定前綴無法編碼或超出上下文：{row['id']}")
        encoded.append({"id": row["id"], "split": row["split"], "question_sha256": row["question_sha256"], "prompt": prompt, "input_ids": ids})
    return encoded, [ids[0] for ids in choice_ids]


def choice_logits(model, rows, choice_ids):
    device = next(model.parameters()).device
    lengths = torch.tensor([len(row["input_ids"]) for row in rows], device=device)
    inputs = torch.full((len(rows), int(lengths.max())), PAD, dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        inputs[index, :lengths[index]] = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
    # labels 永遠不傳給 model.forward；取每列真正最後 token，不取 PAD 的位置。
    logits, _ = model(inputs)
    return logits[torch.arange(len(rows), device=device), lengths - 1][:, choice_ids]


def memory_check():
    if torch.cuda.is_available() and torch.cuda.max_memory_allocated() > MAX_MEMORY:
        raise PilotStop("GPU 峰值超過6GiB")


def _new_adapter(base_dir, policy):
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(policy["gpu_mem_fraction"])
        torch.cuda.reset_peak_memory_stats()
    return practice.GPTAdapter(base_dir)


def _fingerprint(directory, adapter):
    manifest = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
    return {"dataset_sha256": data.file_hash(Path(directory) / "manifest.json"),
            "question_files": {split: manifest["files"][f"questions/{split}.jsonl"] for split in ("train", "dev")},
            "keys_sha256": manifest["files"]["evaluator/keys.jsonl"], "model": adapter.metadata,
            "trainer_sha256": data.file_hash(Path(__file__))}


@torch.no_grad()
def predict_rows(model, encoded, choice_ids):
    model.eval()
    results = []
    for row in encoded:
        scores = choice_logits(model, [row], choice_ids)[0]
        if not torch.isfinite(scores).all():
            raise PilotStop("非有限預測分數")
        results.append({**row, "fixed_prefix": PREFIX, "scores": scores.cpu().tolist(), "prediction": int(scores.argmax())})
        memory_check()
    return results


def evaluate_predictions(predictions, labels):
    result = {}
    for split in ("train", "dev"):
        rows = [row for row in predictions if row["split"] == split]
        correct = sum(row["prediction"] == labels[row["id"]] for row in rows)
        loss = F.cross_entropy(torch.tensor([row["scores"] for row in rows]), torch.tensor([labels[row["id"]] for row in rows])).item()
        result[split] = {"eligible": len(rows), "correct": correct, "accuracy": correct / len(rows), "class_loss": loss}
    return result


def _chat(adapter):
    return [{"prompt": prompt, "raw_completion": adapter.generate(prompt, practice.PARAMETERS, 43 + index)}
            for index, prompt in enumerate(("你好", "你是誰？", "how are you?"))]


def preflight(data_dir, base_dir, out_dir, *, policy=None, adapter=None):
    policy = policy_checked(policy)
    out_dir = data.inside_project(out_dir)
    if out_dir.exists():
        raise ValueError("preflight 輸出已存在，不可覆寫")
    manifest, rows = load_pilot_data(data_dir)
    adapter = adapter or _new_adapter(base_dir, policy)
    trainable = configure_trainable(adapter.model)
    encoded, choice_ids = encode_questions(rows, adapter.tokenizer, adapter.block_size)
    fingerprint = _fingerprint(data_dir, adapter)
    out_dir.mkdir(parents=True)
    predictions = []
    with (out_dir / "blind_predictions.jsonl").open("x", encoding="utf-8") as stream:
        for row in encoded:
            prediction = predict_rows(adapter.model, [row], choice_ids)[0]
            practice._persist(stream, prediction)
            predictions.append(prediction)
    # 全18 train與3 dev已持久落地，這之前沒有開評分鍵。
    labels = evaluator_labels(data_dir, rows, manifest)
    baseline = evaluate_predictions(predictions, labels)
    data.write_json(out_dir / "baseline_grade.json", baseline)
    data.write_json(out_dir / "chat_before.json", _chat(adapter))
    adapter.verify_unchanged()
    contract = {"schema_version": 1, "status": "preflight_complete", "optimizer_steps": 0,
                "fingerprint": fingerprint, "policy": policy, "prefix": PREFIX, "choice_ids": choice_ids,
                "trainable_names": trainable, "trainable_parameters": sum(p.numel() for p in adapter.model.parameters() if p.requires_grad),
                "total_parameters": sum(p.numel() for p in adapter.model.parameters()), "block_size": adapter.block_size,
                "token_range": [min(len(row["input_ids"]) for row in encoded), max(len(row["input_ids"]) for row in encoded)],
                "memory": {"peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0, "limit_bytes": MAX_MEMORY},
                "environment": {"torch": torch.__version__, "python": sys.version.split()[0], "device": str(next(adapter.model.parameters()).device)},
                "artifacts": {name: data.file_hash(out_dir / name) for name in ("blind_predictions.jsonl", "baseline_grade.json", "chat_before.json")}}
    data.write_json(out_dir / "contract.json", contract)
    return contract


def validate_preflight(directory, data_dir, adapter, encoded, choice_ids, policy):
    directory = data.inside_project(directory)
    contract = json.loads((directory / "contract.json").read_text(encoding="utf-8"))
    if (contract.get("status") != "preflight_complete" or contract.get("optimizer_steps") != 0
            or contract.get("fingerprint") != _fingerprint(data_dir, adapter) or contract.get("policy") != policy
            or contract.get("prefix") != PREFIX or contract.get("choice_ids") != choice_ids):
        raise ValueError("preflight 的資料／模型／程式／配方已改變，必須重新唯讀檢查")
    for name, digest in contract["artifacts"].items():
        if name not in ("blind_predictions.jsonl", "baseline_grade.json", "chat_before.json") or data.file_hash(directory / name) != digest:
            raise ValueError("preflight 作答／基準 hash 不符")
    predictions = data.read_jsonl(directory / "blind_predictions.jsonl")
    if len(predictions) != len(encoded) or any(any(saved.get(key) != expected[key] for key in expected) for saved, expected in zip(predictions, encoded)):
        raise ValueError("preflight 題目／token 序列不一致")
    return contract, predictions


def state_copy(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def optimization_epoch(model, rows, labels, choice_ids, optimizer, policy, remaining):
    """每個4-microbatch視窗以實際樣本數平均；最後不足4批仍正確更新一次。"""
    model.train()
    batches = [rows[index:index + policy["batch_size"]] for index in range(0, len(rows), policy["batch_size"])]
    logs = []
    for start in range(0, len(batches), policy["accumulation_steps"]):
        if len(logs) >= remaining:
            break
        window = batches[start:start + policy["accumulation_steps"]]
        count, total_loss = sum(map(len, window)), 0.0
        optimizer.zero_grad(set_to_none=True)
        for batch in window:
            scores = choice_logits(model, batch, choice_ids)
            target = torch.tensor([labels[row["id"]] for row in batch], device=scores.device)
            loss = F.cross_entropy(scores, target)
            if not torch.isfinite(loss):
                raise PilotStop("非有限訓練 loss")
            (loss * len(batch) / count).backward()
            total_loss += loss.item() * len(batch) / count
            memory_check()
        parameters = [p for p in model.parameters() if p.requires_grad]
        try:
            norm = torch.nn.utils.clip_grad_norm_(parameters, policy["clip_grad_norm"], error_if_nonfinite=True)
        except RuntimeError as error:
            raise PilotStop("非有限梯度") from error
        optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in parameters):
            raise PilotStop("更新後出現非有限參數")
        logs.append({"loss": total_loss, "examples": count, "microbatches": len(window), "gradient_norm": float(norm)})
    return logs


def train(data_dir, base_dir, out_dir, *, preflight_dir=DEFAULT_PREFLIGHT, policy=None, adapter=None):
    policy = policy_checked(policy)
    out_dir, base_dir = data.inside_project(out_dir), data.inside_project(base_dir)
    if out_dir.exists() or out_dir.is_relative_to(base_dir) or base_dir.is_relative_to(out_dir):
        raise ValueError("輸出須為全新且與正式 checkpoint 分離的目錄")
    manifest, rows = load_pilot_data(data_dir)
    adapter = adapter or _new_adapter(base_dir, policy)
    trainable = configure_trainable(adapter.model)
    encoded, choice_ids = encode_questions(rows, adapter.tokenizer, adapter.block_size)
    contract, predictions = validate_preflight(preflight_dir, data_dir, adapter, encoded, choice_ids, policy)
    labels = evaluator_labels(data_dir, rows, manifest)
    baseline = evaluate_predictions(predictions, labels)
    saved_baseline = json.loads((Path(preflight_dir) / "baseline_grade.json").read_text(encoding="utf-8"))
    if baseline != saved_baseline:
        raise ValueError("preflight 評分不一致")
    out_dir.mkdir(parents=True)
    best_state, best_correct, best_epoch, best_step = state_copy(adapter.model), baseline["dev"]["correct"], 0, 0
    optimizer = torch.optim.AdamW([p for p in adapter.model.parameters() if p.requires_grad], lr=policy["lr"], weight_decay=policy["weight_decay"])
    history, steps, stop_reason = [], 0, "recipe_complete"
    try:
        for epoch in range(1, policy["epochs"] + 1):
            if steps >= policy["max_steps"]:
                stop_reason = "max_steps"
                break
            shuffled = [row for row in encoded if row["split"] == "train"]
            random.Random(policy["seed"] + epoch).shuffle(shuffled)
            torch.manual_seed(policy["seed"] + epoch)
            updates = optimization_epoch(adapter.model, shuffled, labels, choice_ids, optimizer, policy, policy["max_steps"] - steps)
            steps += len(updates)
            metrics = evaluate_predictions(predict_rows(adapter.model, encoded, choice_ids), labels)
            history.append({"epoch": epoch, "optimizer_steps": steps, "updates": updates, "metrics": metrics})
            if metrics["dev"]["correct"] < best_correct:
                stop_reason = "dev_dropped_one_or_more"
                break
            if metrics["dev"]["correct"] > best_correct:
                best_state, best_correct, best_epoch, best_step = state_copy(adapter.model), metrics["dev"]["correct"], epoch, steps
            if _fingerprint(data_dir, adapter) != contract["fingerprint"]:
                raise PilotStop("訓練期間資料／基底／程式 hash 改變")
    except (PilotStop, RuntimeError) as error:
        if not isinstance(error, PilotStop) and "out of memory" not in str(error).lower():
            raise
        stop_reason = f"stopped: {error}"
    finally:
        adapter.model.load_state_dict(best_state)
        adapter.model.eval()
    adapter.verify_unchanged()
    data.write_json(out_dir / "history.json", {"policy": policy, "baseline": baseline, "history": history, "optimizer_steps": steps,
                    "stop_reason": stop_reason, "best_epoch": best_epoch, "best_step": best_step, "best_dev_correct": best_correct,
                    "trainable_names": trainable, "preflight_contract_sha256": data.file_hash(Path(preflight_dir) / "contract.json"), "auto_promoted": False})
    torch.save(best_state, out_dir / "model.pt")
    config = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    config["electricity_pilot"] = {"dataset_sha256": contract["fingerprint"]["dataset_sha256"], "decoder": "four_class_next_token", "prefix": PREFIX, "best_epoch": best_epoch}
    data.write_json(out_dir / "config.json", config)
    for name in ("bpe_vocab.json", "bpe_merges.json"):
        shutil.copyfile(base_dir / name, out_dir / name)
    for name in ("contract.json", "baseline_grade.json", "blind_predictions.jsonl", "chat_before.json"):
        shutil.copyfile(Path(preflight_dir) / name, out_dir / name)
    after = _chat(adapter)
    data.write_json(out_dir / "chat_after.json", after)
    before = json.loads((out_dir / "chat_before.json").read_text(encoding="utf-8"))
    data.write_json(out_dir / "chat_review.json", {"changed_prompts": [a["prompt"] for a, b in zip(before, after) if a["raw_completion"] != b["raw_completion"]],
                    "status": "manual review required; no automatic chat quality claim", "auto_promoted": False})
    return json.loads((out_dir / "history.json").read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "train"))
    parser.add_argument("--data-dir", type=Path, default=data.DEFAULT_DATASET)
    parser.add_argument("--base-dir", type=Path, default=practice.DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, default=DEFAULT_PREFLIGHT)
    for name, default in POLICY.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    args = parser.parse_args()
    policy = {name: getattr(args, name) for name in POLICY}
    if args.command == "preflight":
        result = preflight(args.data_dir, args.base_dir, args.out_dir, policy=policy)
        print(json.dumps({"status": result["status"], "optimizer_steps": 0, "tokens": result["token_range"]}, ensure_ascii=False))
    else:
        result = train(args.data_dir, args.base_dir, args.out_dir, preflight_dir=args.preflight_dir, policy=policy)
        print(json.dumps({key: result[key] for key in ("optimizer_steps", "stop_reason", "best_epoch", "auto_promoted")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

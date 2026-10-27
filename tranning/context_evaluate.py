"""直接評估Transformer原始生成；評分不參與runtime回答，也不呼叫外部模型。"""
import argparse, hashlib, json, random, unicodedata
from collections import defaultdict
from pathlib import Path

from context_format import FORMAT_VERSION


def normalized(text):
    return unicodedata.normalize("NFKC", text).strip().rstrip("。.!！").strip()


def score(row, reply):
    """Exact加少數完整句型別名；不以contains接受混有錯誤答案的文字。"""
    expected, actual = normalized(row["reply"]), normalized(reply)
    exact = actual == expected
    allowed = {expected}
    if row.get("turn_index", 0) > 0:
        wrappers = {"name": ("你叫{}", "你的名字是{}", "你目前叫{}"),
                    "object": ("它是{}", "目前的物品是{}", "你說的是{}"),
                    "preference": ("你喜歡{}", "你現在喜歡{}", "你目前喜歡的顏色是{}")}
        allowed.update(pattern.format(expected) for pattern in wrappers.get(row["task"], ()))
    return {"exact": exact, "task_success": actual in allowed}


def _summary(rows):
    tasks = {}
    for task in sorted({row["task"] for row in rows}):
        selected = [row for row in rows if row["task"] == task]
        tasks[task] = {"count": len(selected), "exact": sum(row["exact"] for row in selected)/len(selected),
                       "task_success": sum(row["task_success"] for row in selected)/len(selected)}
    return {"count": len(rows), "by_task": tasks,
            "macro_exact": sum(item["exact"] for item in tasks.values())/max(1, len(tasks)),
            "macro_task_success": sum(item["task_success"] for item in tasks.values())/max(1, len(tasks))}


def evaluate_records(records, predictor, counterfactual=True):
    """predictor只接受prompt/history並回傳模型文字；方便用假模型測評估契約。"""
    groups = defaultdict(list)
    for row in records:
        if row["history"]:
            groups[(row["task"], row["prompt"])].append(row)
    cache, rows = {}, []
    def predict(prompt, history):
        key = json.dumps([prompt, history], ensure_ascii=False)
        if key not in cache:
            cache[key] = predictor(prompt, history)
        return cache[key]
    for index, record in enumerate(records):
        generated = predict(record["prompt"], record["history"])
        row = dict(record, generated=generated, **score(record, generated))
        if counterfactual and record["history"]:
            empty_output = predict(record["prompt"], [])
            empty = dict(record, reply=record["no_history_reply"], turn_index=0)
            row["history_removed"] = {"expected": empty["reply"], "generated": empty_output, **score(empty, empty_output)}
            partner = next((other for other in groups[(record["task"], record["prompt"])] if other["reply"] != record["reply"]), None)
            if partner is not None:
                swapped_output = predict(record["prompt"], partner["history"])
                swapped = dict(record, reply=partner["reply"])
                row["history_swapped"] = {"history": partner["history"], "expected": partner["reply"],
                                           "generated": swapped_output, **score(swapped, swapped_output)}
                row["contrastive_success"] = row["task_success"] and row["history_swapped"]["task_success"] and normalized(generated) != normalized(swapped_output)
        rows.append(row)
    summary = _summary(rows)
    contrastive = [row for row in rows if "contrastive_success" in row]
    removed = [row for row in rows if "history_removed" in row]
    summary["contrastive_pairs"] = len(contrastive)
    summary["contrastive_success"] = sum(row["contrastive_success"] for row in contrastive)/max(1, len(contrastive))
    summary["missing_history_success"] = sum(row["history_removed"]["task_success"] for row in removed)/max(1, len(removed))
    unique = {}
    for row in rows:
        key = json.dumps([row["history"], row["prompt"]], ensure_ascii=False)
        if key in unique and unique[key]["reply"] != row["reply"]:
            raise ValueError("同一模型輸入有衝突的期待答案。")
        unique.setdefault(key, row)
    summary["unique_inputs"] = len(unique)
    summary["repeated_inputs"] = len(rows)-len(unique)
    return {"summary": summary, "unique_input_summary": _summary(list(unique.values())), "rows": rows}


def evaluate_file(data, model_dir, report_path, device=None, seed=42, limit=None,
                  counterfactual=True, legacy=False, max_new_tokens=None):
    import torch
    import transformer_chat
    data, model_dir, report_path = Path(data), Path(model_dir), Path(report_path)
    root = Path(__file__).resolve().parents[1]
    report_path = report_path.resolve()
    if not report_path.is_relative_to(root):
        raise ValueError("評估報告必須寫在專案內。")
    raw = data.read_bytes()
    records = json.loads(raw)
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit必須為正整數。")
        records = records[:limit]
    if not records:
        raise ValueError("評估資料不可為空。")
    random.seed(seed); torch.manual_seed(seed)
    max_new_tokens = max_new_tokens if max_new_tokens is not None else (60 if legacy else 20)
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or not 1 <= max_new_tokens <= 60:
        raise ValueError("max_new_tokens必須為1至60。")
    kwargs = {"out_dir": model_dir, "device": device, "max_new_tokens": max_new_tokens,
              "temperature": 1.0, "top_k": 1, "top_p": 1.0, "repetition_penalty": 1.0}
    def predict(prompt, history):
        if legacy:
            return transformer_chat.reply(prompt, **kwargs)
        return transformer_chat.reply(prompt, history=history, context_format=FORMAT_VERSION, **kwargs)
    result = evaluate_records(records, predict, counterfactual=counterfactual and not legacy)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    result["metadata"] = {"data": str(data.resolve()), "data_sha256": hashlib.sha256(raw).hexdigest(),
                           "model_dir": str(model_dir.resolve()),
                           "model_sha256": hashlib.sha256((model_dir / "model.pt").read_bytes()).hexdigest(),
                           "tokenizer_sha256": hashlib.sha256((model_dir / "bpe_vocab.json").read_bytes()).hexdigest(),
                           "tokenizer_merges_sha256": hashlib.sha256((model_dir / "bpe_merges.json").read_bytes()).hexdigest(),
                           "config_sha256": hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest(),
                           "seed": seed, "mode": "single_turn_regression" if legacy else "context",
                           "format": config.get("context_format", "legacy") if legacy else FORMAT_VERSION,
                           "generation": {key: value for key, value in kwargs.items() if key not in ("out_dir", "device")},
                           "limit": limit, "counterfactual": counterfactual and not legacy,
                           "is_full_test": data.name == "test.json" and limit is None,
                           "evaluation_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "scoring": "NFKC+outer whitespace/terminal punctuation only; exact or predeclared full-sentence aliases, never contains"}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--no-counterfactual", action="store_true")
    parser.add_argument("--legacy", action="store_true", help="單輪回歸；不強制format，沿用checkpoint格式")
    args = parser.parse_args()
    result = evaluate_file(args.data, args.model_dir, args.report, args.device, args.seed, args.limit,
                           not args.no_counterfactual, args.legacy, args.max_new_tokens)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))

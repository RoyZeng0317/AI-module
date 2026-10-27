"""只給題目直接呼叫自建 GPT；作答檔落地後，另一個指令才讀評分鍵。無權重更新。"""
import argparse, json, os, random, re, uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import electricity_dataset as data

DEFAULT_CHECKPOINT = data.ROOT / "tranning/gpt_chat_runs"
PARAMETERS = {"max_new_tokens": 64, "temperature": 0.0, "top_k": 30, "top_p": .85, "repetition_penalty": 1.3}
FEEDBACK = {
    "correct": "本次格式與選項通過核對。",
    "incorrect": "選項未通過核對；請重新檢查題目條件、單位與計算。",
    "unparseable": "請只保留一個明確答案，最後一行以「答案：」接你選的一個代號。",
    "needs_manual_review": "此題型尚無可靠自動評分器，需人工核對。",
}


def format_question(question):
    prompt = "請作答下列基本電學選擇題。最後一行以「答案：」接你選的一個代號（A、B、C或D）。\n" + question["question"]
    if question.get("diagram_description"):
        prompt += "\n圖形描述：" + question["diagram_description"]
    return prompt + "\n" + "\n".join(f"({letter}) {question['options'][letter]}" for letter in "ABCD")


class GPTAdapter:
    """沿用正式 reply 介面，先擋住舊介面的左截斷與未知字元。"""
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device=None):
        import transformer_chat as transformer
        self.transformer, self.checkpoint, self.device = transformer, data.inside_project(checkpoint), device
        required = ("model.pt", "config.json", "bpe_vocab.json", "bpe_merges.json")
        if any(not (self.checkpoint / name).is_file() for name in required):
            raise ValueError("正式 checkpoint／tokenizer 不完整；不接受未訓練提示當作答")
        self.metadata = {"checkpoint": self.checkpoint.relative_to(data.ROOT).as_posix(),
                         "files": {name: data.file_hash(self.checkpoint / name) for name in required},
                         "code": {name: data.file_hash(Path(__file__).parent / name) for name in
                                  ("electricity_practice.py", "electricity_dataset.py", "transformer_chat.py", "bpe_tokenizer.py", "context_format.py")}}
        config = json.loads((self.checkpoint / "config.json").read_text(encoding="utf-8"))
        if config.get("context_format") not in (None, "legacy"):
            raise ValueError("本階段 runner 僅核對正式 legacy GPT；其他上下文格式需先另驗證")
        # 此 runner 是獨立程序；避免沿用其他呼叫留下的舊權重快取。
        transformer._loaded_gpt.pop((str(self.checkpoint), transformer._resolve_device(device)), None)
        loaded = transformer._load_gpt(self.checkpoint, device)
        if loaded is None:
            raise ValueError("無法載入正式 GPT")
        self.model, self.tokenizer, self.block_size, self.actual_device = loaded
        self.model.eval()

    def inspect(self, prompt, max_new_tokens):
        from bpe_tokenizer import UNK
        ids = self.tokenizer.encode(prompt)
        unknown = ids.count(UNK)
        status = "ready"
        if unknown:
            status = "not_attempted_encoding"
        elif len(ids) + 1 + max_new_tokens > self.block_size:
            status = "not_attempted_context_limit"
        return {"status": status, "prompt_tokens_with_sep": len(ids) + 1, "block_size": self.block_size,
                "unknown_tokens": unknown, "unknown_chars": sorted({char for char in prompt if char not in self.tokenizer.vocab})}

    def generate(self, prompt, parameters, seed):
        import torch
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return self.transformer.reply(prompt, out_dir=self.checkpoint, history=None, context_format="legacy",
                                      device=self.actual_device, **parameters)

    def verify_unchanged(self):
        if any(data.file_hash(self.checkpoint / name) != digest for name, digest in self.metadata["files"].items()):
            raise ValueError("推論期間 checkpoint／tokenizer 被修改；本輪不可作有效基準")


def _parameters(parameters):
    parameters = {**PARAMETERS, **(parameters or {})}
    if set(parameters) != set(PARAMETERS) or type(parameters["max_new_tokens"]) is not int or parameters["max_new_tokens"] < 1:
        raise ValueError("生成參數或 token 上限無效")
    if not 0 <= parameters["temperature"] <= 2 or not 0 < parameters["top_p"] <= 1 or parameters["top_k"] < 1 or parameters["repetition_penalty"] < 1:
        raise ValueError("生成參數超出範圍")
    return parameters


def _persist(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def attempt(dataset, split, output, *, checkpoint=DEFAULT_CHECKPOINT, limit=12, seed=43, parameters=None, adapter=None, device=None):
    dataset, output = data.inside_project(dataset), data.inside_project(output)
    manifest, questions = data.load_questions(dataset, split)  # 不開 evaluator/keys.jsonl。
    if not questions or type(limit) is not int or not 1 <= limit <= 12:
        raise ValueError("本階段需非空題目集合，且每輪最多十二題")
    if split == "test" and not manifest["heldout_groups_available"]:
        raise ValueError("來源群不足，沒有有效 heldout test")
    parameters = _parameters(parameters)
    adapter = adapter or GPTAdapter(checkpoint, device)
    selected = sorted(questions, key=lambda row: row["id"])[:limit]
    if split == "test" and any((dataset / "test_attempts" / f"{row['id']}.json").exists() for row in selected):
        raise ValueError("test 題已有首次作答紀錄，不可重試")
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with output.open("x", encoding="utf-8") as stream:
        for index, question in enumerate(selected):
            prompt = format_question(question)
            coverage = adapter.inspect(prompt, parameters["max_new_tokens"])
            row = {"schema_version": data.SCHEMA_VERSION, "attempt_id": uuid.uuid4().hex, "question_id": question["id"],
                   "question_sha256": question["question_sha256"], "dataset_sha256": data.file_hash(dataset / "manifest.json"),
                   "split": split, "source_sha256": question["source"]["sha256"], "model": adapter.metadata,
                   "seed": seed + index, "generation": parameters, "prompt": prompt, "coverage": coverage,
                   "created_at": datetime.now(timezone.utc).isoformat(), "status": coverage["status"], "raw_completion": None}
            if coverage["status"] == "ready":
                if split == "test":
                    registry = dataset / "test_attempts" / f"{question['id']}.json"
                    registry.parent.mkdir(parents=True, exist_ok=True)
                    with registry.open("x", encoding="utf-8") as reservation:
                        _persist(reservation, {"attempt_id": row["attempt_id"], "output": output.relative_to(data.ROOT).as_posix()})
                try:
                    completion = adapter.generate(prompt, parameters, seed + index)
                    if not isinstance(completion, str) or not completion.strip():
                        raise ValueError("模型回覆不是有效非空字串")
                    row["raw_completion"], row["status"] = completion, "completed"
                except Exception as error:
                    row["status"], row["error"] = "generation_failed", f"{type(error).__name__}: {error}"
            _persist(stream, row)  # 此處已 fsync；評分必須另一個命令才可讀 keys。
            records.append(row)
    adapter.verify_unchanged()
    return records


def parse_mcq(completion):
    """只接受唯一末行宣告；列舉、矛盾或多次答案皆不給分。"""
    lines = [line.strip() for line in completion.splitlines() if line.strip()]
    if not lines:
        return None
    pattern = r"(?:答案|最終答案|final answer|answer)\s*[:：]\s*(?:\(\s*([A-D1-4])\s*\)|([A-D1-4]))[.。]?"
    final = re.fullmatch(pattern, lines[-1], re.I)
    if not final:
        return None
    claim = r"(?:答案|answer|我選|應選|選擇|I choose)\s*(?:是|為|is)?\s*[:：]?\s*[（(]?[A-D1-4]"
    enumeration = r"[A-D]\s*(?:[,、/| ]\s*[A-D]){3}|[A-D]\s*(?:or|或|/)\s*[A-D]"
    if any(re.search(claim, line, re.I) or re.search(enumeration, line, re.I)
           or re.search(r"[A-D]\s*(?:is\s+(?:the\s+)?correct|才?是\s*(?:正確)?答案)", line, re.I) for line in lines[:-1]):
        return None
    answer = (final[1] or final[2]).upper()
    return dict(zip("1234", "ABCD")).get(answer, answer)


def score_completion(completion, key):
    # 尚無原檔核對過的自由數值／推導題評分契約；不做子字串或相似度猜分。
    if key.get("kind") != "mcq":
        return {"verdict": "needs_manual_review", "error_type": "unsupported_scoring_contract"}
    if key.get("reviewed") is not True or not isinstance(key.get("correct_option"), str) or key["correct_option"] not in set("ABCD"):
        raise ValueError("評分鍵未審核或不是唯一 A-D")
    answer = parse_mcq(completion)
    if answer is None:
        return {"verdict": "unparseable", "error_type": "missing_or_ambiguous_final_answer"}
    correct = answer == key["correct_option"]
    return {"verdict": "correct" if correct else "incorrect", "error_type": None if correct else "option_mismatch", "parsed_answer": answer}


def _load_keys(dataset):
    data.verify_dataset(dataset, include_keys=True)
    keys = data.read_jsonl(Path(dataset) / "evaluator/keys.jsonl")
    if len({row["id"] for row in keys}) != len(keys):
        raise ValueError("評分鍵 id 重複")
    return {row["id"]: row for row in keys}


def grade(dataset, attempts_path, output, feedback=None):
    dataset, attempts_path = data.inside_project(dataset), data.inside_project(attempts_path)
    output = data.inside_project(output)
    if feedback and data.inside_project(feedback) == output:
        raise ValueError("評分與糾錯輸出不可使用同一路徑")
    if output.exists() or (feedback and data.inside_project(feedback).exists()):
        raise ValueError("評分／糾錯紀錄已存在，拒絕覆寫")
    attempts = data.read_jsonl(attempts_path)
    if not attempts:
        raise ValueError("作答檔為空；不可產生假零分")
    manifest_hash = data.file_hash(dataset / "manifest.json")
    questions = {row["id"]: row for split in data.SPLITS for row in data.load_questions(dataset, split)[1]}
    seen = set()
    for row in attempts:
        question = questions.get(row.get("question_id"))
        if (not question or row.get("schema_version") != data.SCHEMA_VERSION or row.get("dataset_sha256") != manifest_hash
                or row.get("question_sha256") != question["question_sha256"] or row.get("split") != question["split"]
                or row.get("source_sha256") != question["source"]["sha256"] or row.get("prompt") != format_question(question)
                or row.get("question_id") in seen or not row.get("attempt_id")):
            raise ValueError("作答 id／來源／版本／題目／split／prompt 不符，或重複作答")
        if row.get("status") not in ("completed", "generation_failed", "not_attempted_context_limit", "not_attempted_encoding"):
            raise ValueError("作答狀態不受支援")
        if row["status"] == "completed" and not isinstance(row.get("raw_completion"), str):
            raise ValueError("缺少原始模型回覆")
        seen.add(row["question_id"])
    if feedback and any(row["split"] != "train" for row in attempts):
        raise ValueError("只有 train 可提供受限糾錯；dev/test 不可提供回饋")
    keys = _load_keys(dataset)  # 完整作答檔存在且核對完成後，才第一次讀取 keys。
    results, corrections = [], []
    for row in attempts:
        key = keys.get(row["question_id"])
        if (not key or key.get("question_sha256") != row["question_sha256"] or key.get("split") != row["split"]
                or key.get("schema_version") != data.SCHEMA_VERSION):
            raise ValueError("題目與評分鍵不匹配")
        result = score_completion(row["raw_completion"], key) if row["status"] == "completed" else {"verdict": row["status"], "error_type": row["status"]}
        results.append({"attempt_id": row["attempt_id"], "question_id": row["question_id"], "split": row["split"], **result})
        if feedback and result["verdict"] in FEEDBACK:
            corrections.append({"attempt_id": row["attempt_id"], "question_id": row["question_id"], "split": row["split"],
                                "verdict": result["verdict"], "error_type": result["error_type"], "hint": FEEDBACK[result["verdict"]]})
    counts = Counter(row["verdict"] for row in results)
    attempted = sum(row["status"] == "completed" for row in attempts)
    summary = {"eligible": len(attempts), "attempted": attempted, **{name: counts[name] for name in
               ("correct", "incorrect", "unparseable", "needs_manual_review", "not_attempted_context_limit", "not_attempted_encoding", "generation_failed")},
               "first_pass_accuracy": counts["correct"] / attempted if attempted else None,
               "coverage": attempted / len(attempts), "denominator": "all completed first attempts, including unparseable/manual-review outputs"}
    report = {"schema_version": data.SCHEMA_VERSION, "dataset_sha256": manifest_hash, "attempts_sha256": data.file_hash(attempts_path), "summary": summary, "results": results}
    data.write_json(output, report)
    if feedback:
        data.write_jsonl(feedback, corrections)
    return report


def chat_baseline(output, *, checkpoint=DEFAULT_CHECKPOINT, seed=43, adapter=None):
    output = data.inside_project(output)
    adapter = adapter or GPTAdapter(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for index, prompt in enumerate(("你好", "你是誰？", "how are you?")):
            coverage = adapter.inspect(prompt, PARAMETERS["max_new_tokens"])
            if coverage["status"] != "ready":
                raise ValueError("一般聊天基準無法完整編碼／容納")
            completion = adapter.generate(prompt, PARAMETERS, seed + index)
            _persist(stream, {"prompt": prompt, "raw_completion": completion, "seed": seed + index,
                              "generation": PARAMETERS, "model": adapter.metadata, "coverage": coverage,
                              "purpose": "retention baseline only; no quality score or training"})
    adapter.verify_unchanged()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("attempt")
    run.add_argument("--dataset", type=Path, default=data.DEFAULT_DATASET)
    run.add_argument("--split", choices=data.SPLITS, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    run.add_argument("--limit", type=int, default=12)
    run.add_argument("--seed", type=int, default=43)
    run.add_argument("--max-new-tokens", type=int, default=64)
    evaluate = commands.add_parser("grade")
    evaluate.add_argument("--dataset", type=Path, default=data.DEFAULT_DATASET)
    evaluate.add_argument("--attempts", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--feedback", type=Path)
    chat = commands.add_parser("chat-baseline")
    chat.add_argument("--output", type=Path, required=True)
    chat.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if args.command == "attempt":
        rows = attempt(args.dataset, args.split, args.output, checkpoint=args.checkpoint, limit=args.limit,
                       seed=args.seed, parameters={"max_new_tokens": args.max_new_tokens})
        print(json.dumps(dict(Counter(row["status"] for row in rows)), ensure_ascii=False))
    elif args.command == "grade":
        print(json.dumps(grade(args.dataset, args.attempts, args.output, args.feedback)["summary"], ensure_ascii=False))
    else:
        chat_baseline(args.output, checkpoint=args.checkpoint)
        print("已保存三筆一般聊天原始基準；未更新權重。")


if __name__ == "__main__":
    main()

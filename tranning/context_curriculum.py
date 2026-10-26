"""本地詞元跨度複述練習；補強開放字詞引用，不修改凍結val/test。"""
import argparse, hashlib, json, random, re
from pathlib import Path

from bpe_tokenizer import BPETokenizer, UNK
from context_format import RESPONSE_BUDGET, build_prompt

ROOT = Path(__file__).resolve().parents[1]


def make_curriculum(source, output, tokenizer_dir, count=2048, seed=43):
    source, output = Path(source), Path(output).resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError("補強資料須寫入專案內的新檔案。")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count必須為正整數。")
    tokenizer = BPETokenizer.load(Path(tokenizer_dir))
    vocabulary = json.loads((Path(tokenizer_dir) / "bpe_vocab.json").read_text(encoding="utf-8"))
    words = sorted(word for word in vocabulary if re.fullmatch(r"[A-Za-z0-9\u4e00-\u9fff]{1,6}", word))
    if len(words) < 2:
        raise ValueError("詞表沒有足夠可用字詞。")
    rng, additions, seen = random.Random(seed), [], set()
    while len(additions) < count:
        first, second = ("".join(rng.choices(words, k=rng.randint(1, 3))) for _ in range(2))
        if first == second or (first, second) in seen:
            continue
        i = len(additions)
        seen.add((first, second))
        history = [(f"請記住這段指定文字：{first}。", "知道了。")]
        answer = first
        if i % 2:
            history.append((f"更正，指定文字改成「{second}」，不要再用之前那段。", "以最新指定文字為準。"))
            answer = second
        history.append(("我們等一下再聊其他事情。", "好的。"))
        prompt = ["請只回答剛才指定的文字。", "把目前指定文字原樣寫出來，不加說明。", "目前應該保留的文字是什麼？只給文字。", "照前面的更正，回答最新指定文字。"][i % 4]
        if i % 5 == 0:
            history, prompt, answer = [], f"請原樣輸出「{first}」，不要加其他文字。", first
        row = {"dialogue_id": f"copy:{seed}:{i}", "task": "span_copy", "template_id": f"copy:{i%5}",
               "entity_group": hashlib.sha256(f"{first}|{second}".encode()).hexdigest(),
               "history": history, "prompt": prompt, "reply": answer}
        ids = build_prompt(tokenizer, prompt, history, 512-RESPONSE_BUDGET)
        reply = tokenizer.encode(answer)
        if UNK in ids or UNK in reply or len(reply)+1 > RESPONSE_BUDGET:
            continue
        additions.append(row)
    records = json.loads(source.read_text(encoding="utf-8")) + additions
    content = json.dumps(records, ensure_ascii=False, indent=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    manifest = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "seed": seed,
                "copy_records": count, "total_records": len(records),
                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "scope": "訓練開放字詞複述；原val/test不變。字詞可與評測共用，不宣稱詞彙隔離。"}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "tranning/gpt_chat_runs")
    parser.add_argument("--count", type=int, default=2048)
    args = parser.parse_args()
    print(json.dumps(make_curriculum(args.source, args.out, args.tokenizer_dir, args.count), ensure_ascii=False))

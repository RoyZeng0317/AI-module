"""本地產生可重現的多輪訓練／驗證／測試資料，不呼叫外部模型。"""
import argparse, hashlib, itertools, json, random
from collections import Counter
from pathlib import Path

from bpe_tokenizer import BPETokenizer, UNK
from context_format import FORMAT_VERSION, RESPONSE_BUDGET, build_prompt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOLDER = ROOT / "data/context_dialogues"
POOLS = {
    "name": ["小明", "小美", "小華", "小文", "小安", "小林", "小芳", "小強", "阿明", "阿美", "阿華", "阿文", "阿安", "阿林", "阿芳", "阿強"],
    "object": ["橡皮擦", "鉛筆", "直尺", "剪刀", "耳機", "鑰匙", "杯子", "書本", "手機", "電腦", "桌子", "椅子", "紙", "筆", "書", "尺"],
    "preference": ["紅色", "藍色", "綠色", "黃色", "白色", "黑色", "紫色", "粉紅色", "淺藍色", "深藍色", "深綠色", "淺綠色", "深紅色", "淺紅色", "深黃色", "淺黃色"],
}
POOLS["format"] = list(POOLS["object"])
QUESTIONS = {
    "name": ["我叫什麼名字？", "我的名字是什麼？", "請只回答我的名字。", "你記得我的名字嗎？只答名字。", "前面提到的我的姓名是？", "請說出我目前的名字。", "現在應該怎麼稱呼我？只給名字。"],
    "object": ["它是什麼物品？只答名稱。", "剛才說的是什麼東西？", "請只說目前那個物品的名稱。", "你記得我正在說的物品嗎？只答名稱。", "前文的它是指什麼？", "現在談的是哪個物品？", "我目前提到的東西叫什麼？"],
    "preference": ["我現在喜歡什麼顏色？", "請只回答我喜歡的顏色。", "我目前最喜歡的顏色是？", "依照前文，我的顏色偏好是什麼？", "現在我偏好哪一種顏色？", "請回想我最後說喜歡的顏色。", "我現在選的是什麼顏色？只答顏色。"],
    "format": ["照剛才的格式回答它的名稱。", "請沿用前面的格式回答。", "依約定格式輸出物品名稱。", "保持剛才格式，回答目前的物品。", "使用之前要求的格式回答它。", "請按目前的格式給出物品名稱。", "沿用最後指定的格式回答物品。"],
}
MISSING = {"name": "你還沒有告訴我名字。", "object": "請先說明你指的物品。", "preference": "你還沒有告訴我喜歡的顏色。", "format": "請先提供物品與回答格式。"}
PREFIXES = ["答案：", "名稱：", "物品：", "結果："]
SPLITS = {"train": (range(0, 8), range(0, 3)), "val": (range(8, 12), range(3, 5)), "test": (range(12, 16), range(5, 7))}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _facts(task, variant, first, second, index):
    if task == "name":
        opening = [f"我叫{first}，請記住。", f"我的名字是{first}。", f"請叫我{first}。", f"先告訴你，我的姓名是{first}。", f"這次對話請稱呼我{first}。", f"我的稱呼是{first}，之後會問你。", f"我是{first}，這是我的名字。"][variant]
        change = f"更正，我不叫{first}，請改叫我{second}。"
        return opening, change, first, second
    if task == "object":
        opening = [f"我正在說的物品是{first}。", f"桌上這個東西叫{first}。", f"接下來說的它是{first}。", f"目前討論的是{first}這個物品。", f"請記得，這次提到的東西是{first}。", f"我手上的物品叫作{first}。", f"這段對話的它代表{first}。"][variant]
        change = f"現在換一個物品，不是{first}了，接下來它是{second}。"
        return opening, change, first, second
    if task == "preference":
        opening = [f"我喜歡{first}。", f"我最喜歡的顏色是{first}。", f"請記住我的顏色偏好是{first}。", f"在這次對話，我選擇{first}。", f"我偏好{first}這種顏色。", f"我現在喜歡的是{first}。", f"目前我的顏色選擇為{first}。"][variant]
        change = f"我的偏好改了，我現在不喜歡{first}，喜歡的是{second}。"
        return opening, change, first, second
    old, new = PREFIXES[index % len(PREFIXES)], PREFIXES[(index+1) % len(PREFIXES)]
    opening = [f"物品是{first}。接下來只用「{old}物品名稱」回答，不加其他文字。", f"我們說的是{first}，回答名稱時請加上「{old}」。", f"請記住物品{first}；之後格式固定為「{old}」接名稱。", f"這次物品是{first}，後續答案只寫「{old}物品名稱」。", f"目前談{first}。請把「{old}」放在回答名稱之前。", f"對話中的物品為{first}。以後回答只給「{old}」加名稱。", f"請採用「{old}物品名稱」格式；目前物品是{first}。"][variant]
    change = f"物品改成{second}。回答格式也改成「{new}物品名稱」，不要再使用「{old}」。"
    return opening, change, old+first, new+second


def _record(dialogue, task, variant, group, history, prompt, reply, turn):
    return {"dialogue_id": dialogue, "task": task, "template_id": f"{task}:v{variant}",
            "entity_group": group, "history": history, "prompt": prompt, "reply": reply,
            "turn_index": turn, "no_history_reply": MISSING[task]}


def generate_records(seed=42):
    result = {}
    for split, (entity_indices, variants) in SPLITS.items():
        records = []
        for task, pool in POOLS.items():
            pairs = list(itertools.permutations([pool[i] for i in entity_indices], 2))
            for variant in variants:
                question = QUESTIONS[task][variant]
                for index, (first, second) in enumerate(pairs):
                    group = f"{task}:{digest([first, second])[:16]}"
                    dialogue = f"{group}:v{variant}"
                    opening, correction, answer, changed_answer = _facts(task, variant, first, second, index)
                    fact = [opening, "知道了。"]
                    history = [fact, ["我們先聊一下今天的天氣。", "好的。"]]
                    records.append(_record(dialogue, task, variant, group, history, question, answer, 1))
                    history = [fact, [question, answer], [correction, "明白，以最新的資訊為準。"], ["等一下再聊其他事情。", "可以。"]]
                    records.append(_record(dialogue, task, variant, group, history, question, changed_answer, 2))
                group = f"missing:{task}:v{variant}"
                records.append(_record(group, task, variant, group, [], question, MISSING[task], 0))
        random.Random(seed + len(split)).shuffle(records)
        result[split] = records
    return result


def validate_splits(splits):
    keys = ("dialogue_id", "template_id", "entity_group")
    seen = {key: {} for key in keys}
    inputs = {}
    for split, records in splits.items():
        if not records:
            raise ValueError(f"{split} 不可為空。")
        for row in records:
            for key in keys:
                value = row[key]
                owner = seen[key].setdefault(value, split)
                if owner != split:
                    raise ValueError(f"{key} 跨資料集洩漏：{value}")
            fingerprint = digest([row["history"], row["prompt"]])
            owner = inputs.setdefault(fingerprint, split)
            if owner != split:
                raise ValueError("完全相同的模型輸入跨資料集。")
    return {key: {split: len({row[key] for row in rows}) for split, rows in splits.items()} for key in keys}


def make_dataset(folder=DEFAULT_FOLDER, seed=42, tokenizer_dir=ROOT / "tranning/gpt_chat_runs"):
    folder = Path(folder).resolve()
    if not folder.is_relative_to(ROOT) or folder == ROOT:
        raise ValueError("資料輸出必須在專案內。")
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("輸出資料夾非空；請使用新位置，避免改動已凍結測試集。")
    tokenizer = BPETokenizer.load(Path(tokenizer_dir))
    splits = generate_records(seed)
    groups = validate_splits(splits)
    total_tokens = unknown = 0
    max_prompt = max_reply = 0
    for rows in splits.values():
        for row in rows:
            ids = build_prompt(tokenizer, row["prompt"], row["history"], max_tokens=512-RESPONSE_BUDGET)
            reply_ids = tokenizer.encode(row["reply"])
            unknown += ids.count(UNK) + reply_ids.count(UNK)
            total_tokens += len(ids) + len(reply_ids)
            max_prompt, max_reply = max(max_prompt, len(ids)), max(max_reply, len(reply_ids)+1)
    if unknown:
        raise ValueError(f"資料含{unknown}個UNK；請先調整實體／模板，不能用未知字測試記憶。")
    if max_reply > RESPONSE_BUDGET:
        raise ValueError("回答超過共同response_budget。")
    pairs = json.loads((ROOT / "data/pairs.json").read_text(encoding="utf-8"))
    regression = [dict(row, history=[], task="legacy_regression", dialogue_id=f"legacy:{i}") for i, row in enumerate(pairs[:20])]
    replay = [dict(row, history=[], task="legacy_replay", dialogue_id=f"legacy:{i+20}") for i, row in enumerate(pairs[20:])]
    heldout_inputs = {row["prompt"] for row in regression}
    if any(row["prompt"] in heldout_inputs for rows in splits.values() for row in rows):
        raise ValueError("單輪回歸問題與多輪訓練資料重疊。")
    folder.mkdir(parents=True)
    files = {}
    for name, rows in dict(splits, regression=regression, replay=replay).items():
        content = json.dumps(rows, ensure_ascii=False, indent=2)
        target = folder / f"{name}.json"
        target.write_text(content, encoding="utf-8")
        files[target.name] = {"sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "records": len(rows), "tasks": dict(Counter(row["task"] for row in rows))}
    manifest = {"format": FORMAT_VERSION, "response_budget": RESPONSE_BUDGET, "seed": seed,
                "files": files, "groups": groups, "cross_split_intersections": 0,
                "tokenizer_sha256": hashlib.sha256((Path(tokenizer_dir) / "bpe_vocab.json").read_bytes()).hexdigest(),
                "unk_tokens": unknown, "total_tokens": total_tokens, "max_prompt_tokens": max_prompt, "max_reply_tokens_with_eos": max_reply,
                "limitations": ["deterministic模板任務，不代表一般語意理解", "實體及具體模板variant分組隔離；字元和任務類型仍會共用", "回歸資料由既有pairs前20筆預先固定，不是新獨立世界知識測試"]}
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FOLDER)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = make_dataset(args.out_dir, args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))

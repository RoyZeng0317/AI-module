"""多輪上下文資料 v2（to-do #50）：修正 v1 的切分偏差並加大多樣性，不呼叫外部模型。

v1（context_dialogues.py）用名單索引區段切 train/val/test，名單又剛好依字形排列，
造成 test 的答案字形在訓練中從未出現：
  顏色 train=紅色/藍色…（單色）   test=深紅色/淺黃色…（深淺＋色）
  物品 train=剪刀/杯子…（2–3 字） test=紙/筆/書/尺（1 字）
  名字 train=小X                   test=阿X
所以 v1 test 混測了「記住上下文」與「生出沒見過的字形」，偏好／格式 0% 無法分辨是
哪一個失敗。v2 的做法：

  - 實體依「字形」分層，每種字形都按 3:1:1 分到 train/val/test；實體本身仍完全不重疊
    （測的是「把前文出現過的新詞照抄回來」，不是背答案）。
  - 實體池從每類 16 個擴大到 30–53 個；開頭／提問各 10 種句型、更正 5 種、閒聊 12 種，
    句型同樣依 split 隔離（test 句型訓練沒看過）。
  - 對話長度與結構隨機：0–3 輪閒聊、另一任務的干擾事實（要挑對欄位）、
    「有歷史但沒提過」→ 必須回答缺資訊（避免亂猜）。

v1 的 val/test 保留不動，仍可用 context_evaluate.py 對照；v2 輸出到新資料夾。
限制：仍是模板任務，只能證明「抄回前文指定欄位」這個窄能力，不代表一般語意理解。

用法：
    python context_dialogues_v2.py --out-dir ../data/context_dialogues_v2
    python context_dialogues_v2.py --out-dir ../data/context_dialogues_v2 --tokenizer-dir gpt_chat_runs
"""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from bpe_tokenizer import BPETokenizer, UNK
from context_dialogues import MISSING, digest, validate_splits
from context_format import FORMAT_VERSION, RESPONSE_BUDGET, build_prompt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOLDER = ROOT / "data/context_dialogues_v2"
DEFAULT_TOKENIZER = ROOT / "tranning/gpt_chat_runs"
SPLIT_NAMES = ("train", "val", "test")
ROUND_ROBIN = ("train", "train", "train", "val", "test")  # 每種字形 3:1:1

FORMS = {
    "name": {
        "小X": "小明 小美 小華 小文 小安 小林 小芳 小強 小玲 小龍 小婷 小傑".split(),
        "阿X": "阿明 阿美 阿華 阿文 阿安 阿林 阿芳 阿強 阿玲 阿德 阿忠 阿慧".split(),
        "老X": "老王 老李 老張 老陳 老林 老黃".split(),
        "全名": "王小明 李大華 張美玲 陳志強 林雅婷 黃建國 吳家豪 劉淑芬 周子瑜 許文傑".split(),
        "雙字名": "志明 春嬌 家豪 雅婷 建國 美玲 怡君 俊傑".split(),
    },
    "object": {
        "1字": "紙 筆 書 尺 碗 盤 刀 傘 鞋 帽 錶 燈 球 瓶 袋 盒".split(),
        "2字": ("剪刀 耳機 鑰匙 杯子 手機 電腦 桌子 椅子 書本 直尺 鉛筆 枕頭 "
                "毛巾 雨傘 背包 錢包 眼鏡 手錶 檯燈 水壺 筷子 湯匙 盤子 鏡子").split(),
        "3字以上": "橡皮擦 筆記本 計算機 保溫瓶 吹風機 遙控器 行李箱 充電器 電風扇 滑鼠墊 原子筆 膠帶台 螺絲起子".split(),
    },
    "preference": {
        "單色": "紅色 藍色 綠色 黃色 白色 黑色 紫色 灰色 橙色 棕色 金色 銀色 橘色 青色".split(),
        "複合色": ("粉紅色 淺藍色 深藍色 深綠色 淺綠色 深紅色 淺紅色 深黃色 "
                  "淺黃色 淺紫色 深紫色 淺灰色 深灰色 天藍色 米白色 咖啡色").split(),
    },
}
FORMS["format"] = FORMS["object"]
PREFIXES = ["答案：", "名稱：", "物品：", "結果：", "回答：", "輸出："]
TASKS = ("name", "object", "preference", "format")

# 句型：索引 0-5 train、6-7 val、8-9 test（更正 0-2/3/4、閒聊 0-7/8-9/10-11）
VARIANT_SPLITS = {"train": range(0, 6), "val": range(6, 8), "test": range(8, 10)}
CORRECTION_SPLITS = {"train": range(0, 3), "val": range(3, 4), "test": range(4, 5)}
SMALLTALK_SPLITS = {"train": range(0, 8), "val": range(8, 10), "test": range(10, 12)}

OPENINGS = {
    "name": ["我叫{a}，請記住。", "我的名字是{a}。", "請叫我{a}。", "先告訴你，我的姓名是{a}。",
             "這次對話請稱呼我{a}。", "我是{a}，之後會問你。", "大家都叫我{a}。", "自我介紹一下，我是{a}。",
             "你可以叫我{a}就好。", "記一下，我的名字叫{a}。"],
    "object": ["我正在說的物品是{a}。", "桌上這個東西叫{a}。", "接下來說的它是{a}。", "目前討論的是{a}這個物品。",
               "請記得，這次提到的東西是{a}。", "我手上的物品叫作{a}。", "我剛買了一個{a}。", "等一下要問的東西是{a}。",
               "我們現在聊的物品是{a}。", "注意，這裡的它指的是{a}。"],
    "preference": ["我喜歡{a}。", "我最喜歡的顏色是{a}。", "請記住我的顏色偏好是{a}。", "在這次對話，我選擇{a}。",
                   "我偏好{a}這種顏色。", "我現在喜歡的是{a}。", "比起其他顏色，我更喜歡{a}。", "我最近很迷{a}。",
                   "說到顏色，我選{a}。", "如果要挑顏色，我會選{a}。"],
    "format": ["物品是{a}。接下來只用「{p}物品名稱」回答，不加其他文字。", "我們說的是{a}，回答名稱時請加上「{p}」。",
               "請記住物品{a}；之後格式固定為「{p}」接名稱。", "這次物品是{a}，後續答案只寫「{p}物品名稱」。",
               "目前談{a}。請把「{p}」放在回答名稱之前。", "對話中的物品為{a}。以後回答只給「{p}」加名稱。",
               "請採用「{p}物品名稱」格式；目前物品是{a}。", "物品叫{a}，回答時開頭固定寫「{p}」。",
               "之後回答請用「{p}」開頭再接名稱，現在的物品是{a}。", "規則：先寫「{p}」再寫名稱。物品是{a}。"],
}
QUESTIONS = {
    "name": ["我叫什麼名字？", "我的名字是什麼？", "請只回答我的名字。", "你記得我的名字嗎？只答名字。",
             "前面提到的我的姓名是？", "請說出我目前的名字。", "我剛才說我叫什麼？", "請問我的稱呼是什麼？只答名字。",
             "現在應該怎麼稱呼我？只給名字。", "回想一下，我叫什麼？"],
    "object": ["它是什麼物品？只答名稱。", "剛才說的是什麼東西？", "請只說目前那個物品的名稱。", "你記得我正在說的物品嗎？只答名稱。",
               "前文的它是指什麼？", "現在談的是哪個物品？", "我說的那個東西叫什麼？", "請回答目前物品的名稱。",
               "我目前提到的東西叫什麼？", "那個物品是什麼？只說名稱。"],
    "preference": ["我現在喜歡什麼顏色？", "請只回答我喜歡的顏色。", "我目前最喜歡的顏色是？", "依照前文，我的顏色偏好是什麼？",
                   "現在我偏好哪一種顏色？", "請回想我最後說喜歡的顏色。", "我選的是哪個顏色？", "我比較喜歡什麼顏色？只答顏色。",
                   "我現在選的是什麼顏色？只答顏色。", "你記得我喜歡的顏色嗎？只答顏色。"],
    "format": ["照剛才的格式回答它的名稱。", "請沿用前面的格式回答。", "依約定格式輸出物品名稱。", "保持剛才格式，回答目前的物品。",
               "使用之前要求的格式回答它。", "請按目前的格式給出物品名稱。", "用說好的格式回答物品。", "依照規則回答目前物品的名稱。",
               "沿用最後指定的格式回答物品。", "請以約定格式說出物品名稱。"],
}
CORRECTIONS = {
    "name": ["更正，我不叫{a}，請改叫我{b}。", "剛才說錯了，我的名字是{b}，不是{a}。", "我改名了，以後叫我{b}，不要再叫{a}。",
             "抱歉，我其實叫{b}，不是{a}。", "名字更新：{a}不對，正確是{b}。"],
    "object": ["現在換一個物品，不是{a}了，接下來它是{b}。", "我換了，桌上的東西改成{b}，不再是{a}。", "更正：它不是{a}，而是{b}。",
               "剛剛說錯了，物品其實是{b}，不是{a}。", "物品更新：從{a}換成{b}。"],
    "preference": ["我的偏好改了，我現在不喜歡{a}，喜歡的是{b}。", "我改變主意了，現在比較喜歡{b}，不是{a}。",
                   "更正一下，我喜歡的是{b}，{a}不是。", "其實我現在喜歡{b}，不再是{a}。", "顏色偏好更新：{a}改成{b}。"],
    "format": ["物品改成{b}。回答格式也改成「{q}物品名稱」，不要再使用「{p}」。", "現在物品是{b}，格式換成「{q}」開頭，不用「{p}」了。",
               "更正：物品換成{b}，之後請用「{q}」而不是「{p}」。", "規則更新：物品為{b}，開頭改寫「{q}」，「{p}」作廢。",
               "改一下，物品是{b}；回答開頭從「{p}」改成「{q}」。"],
}
CORRECTION_ACKS = ["明白，以最新的資訊為準。", "好的，已更新。", "了解，之後用新的。"]
SMALLTALK = [("我們先聊一下今天的天氣。", "好的。"), ("等一下再聊其他事情。", "可以。"), ("今天有點累。", "辛苦了。"),
             ("我剛吃完午餐。", "好的。"), ("外面在下雨。", "記得帶傘。"), ("你覺得週末要做什麼？", "可以出去走走。"),
             ("我等一下要去開會。", "好的。"), ("最近天氣變冷了。", "要注意保暖。"),
             ("我昨天看了一部電影。", "聽起來不錯。"), ("今天交通很塞。", "辛苦了。"),
             ("我在準備考試。", "加油。"), ("晚點要去買東西。", "好的。")]
FACT_ACK = "知道了。"


def _split_entities(seed):
    """依字形分層，每種字形以 3:1:1 輪流分配；回傳 {task: {split: [(entity, form)]}}。"""
    result = {}
    for task, forms in FORMS.items():
        by_split = {name: [] for name in SPLIT_NAMES}
        for form, entities in forms.items():
            shuffled = sorted(entities)
            random.Random(f"{seed}:{task if task != 'format' else 'object'}:{form}").shuffle(shuffled)
            for i, entity in enumerate(shuffled):
                by_split[ROUND_ROBIN[i % len(ROUND_ROBIN)]].append((entity, form))
        result[task] = by_split
    return result


def _fact(task, variant, entity, prefix=None):
    return OPENINGS[task][variant].format(a=entity, p=prefix)


def _answer(task, entity, prefix=None):
    return prefix + entity if task == "format" else entity


def _smalltalk(rng, split, k):
    pool = [SMALLTALK[i] for i in SMALLTALK_SPLITS[split]]
    return [list(rng.choice(pool)) for _ in range(k)]


def _interference(rng, task, split, entities):
    """另一任務的事實（不同欄位），模型要挑對問的那一項。format 與 object 共用欄位，互不當干擾。"""
    others = [t for t in ("name", "object", "preference") if t != task and not (task == "format" and t == "object")]
    other = rng.choice(others)
    entity, _ = rng.choice(entities[other][split])
    variant = rng.choice(list(VARIANT_SPLITS[split]))
    return [_fact(other, variant, entity), FACT_ACK]


def _record(dialogue, task, kind, template, group, history, prompt, reply, turn, form):
    return {"dialogue_id": dialogue, "task": task, "kind": kind, "template_id": template,
            "entity_group": group, "entity_form": form, "history": history, "prompt": prompt,
            "reply": reply, "turn_index": turn, "no_history_reply": MISSING[task]}


DIALOGUES_PER_TASK = {"train": 450, "val": 40, "test": 40}
IRRELEVANT_PER_TASK = {"train": 120, "val": 18, "test": 18}


def generate_records(seed=42):
    entities = _split_entities(seed)
    result = {}
    for split in SPLIT_NAMES:
        rng = random.Random(f"{seed}:{split}")
        records, seen = [], set()

        def add(row):
            key = json.dumps([row["history"], row["prompt"]], ensure_ascii=False)
            if key in seen:
                return False
            seen.add(key)
            records.append(row)
            return True

        variants = list(VARIANT_SPLITS[split])
        corrections = list(CORRECTION_SPLITS[split])
        for task in TASKS:
            pool = entities[task][split]
            made, attempts = 0, 0
            while made < DIALOGUES_PER_TASK[split] and attempts < 50 * DIALOGUES_PER_TASK[split]:
                attempts += 1
                (first, form1), (second, form2) = rng.sample(pool, 2)
                o, q, c = rng.choice(variants), rng.choice(variants), rng.choice(corrections)
                p_old, p_new = rng.sample(PREFIXES, 2) if task == "format" else (None, None)
                group = f"{task}:{digest([first, second, p_old, p_new])[:16]}"
                dialogue = f"{group}:o{o}q{q}c{c}"
                template = f"{task}:o{o}:q{q}:c{c}"
                question = QUESTIONS[task][q]
                ans1, ans2 = _answer(task, first, p_old), _answer(task, second, p_new)
                fact = [_fact(task, o, first, p_old), FACT_ACK]

                before = _smalltalk(rng, split, rng.randint(0, 1))
                after = _smalltalk(rng, split, rng.randint(0, 2))
                recall_hist = before + [fact] + after
                if rng.random() < 0.5:
                    recall_hist.insert(rng.randint(0, len(recall_hist)), _interference(rng, task, split, entities))
                correction = [CORRECTIONS[task][c].format(a=first, b=second, p=p_old, q=p_new), rng.choice(CORRECTION_ACKS)]
                corrected_hist = recall_hist + [[question, ans1], correction] + _smalltalk(rng, split, rng.randint(0, 2))
                ok1 = add(_record(dialogue, task, "recall", template, group, recall_hist, question, ans1, 1, form1))
                ok2 = add(_record(dialogue, task, "corrected", template, group, corrected_hist, question, ans2, 2, form2))
                made += ok1 and ok2

            for i in range(IRRELEVANT_PER_TASK[split]):
                q = rng.choice(variants)
                hist = _smalltalk(rng, split, rng.randint(1, 3))
                if rng.random() < 0.6:
                    hist.insert(rng.randint(0, len(hist)), _interference(rng, task, split, entities))
                group = f"irrelevant:{split}:{task}:{i}"
                add(_record(group, task, "missing_irrelevant", f"{task}:q{q}:irrelevant", group, hist,
                            QUESTIONS[task][q], MISSING[task], 0, None))
            for q in variants:
                group = f"missing:{split}:{task}:q{q}"
                add(_record(group, task, "missing_empty", f"{task}:q{q}:empty", group, [],
                            QUESTIONS[task][q], MISSING[task], 0, None))
        rng.shuffle(records)
        result[split] = records
    return result


def check_entity_isolation(splits):
    """答案實體不可跨 split，且每個 split 都要涵蓋該任務的所有字形。"""
    owners, forms = {}, defaultdict(lambda: defaultdict(set))
    for split, rows in splits.items():
        for row in rows:
            if row["kind"] not in ("recall", "corrected"):
                continue
            entity = row["reply"]
            if row["task"] == "format":
                entity = next(entity[len(p):] for p in PREFIXES if entity.startswith(p))
            key = (row["task"] if row["task"] != "format" else "object", entity)
            owner = owners.setdefault(key, split)
            if owner != split:
                raise ValueError(f"實體跨資料集洩漏：{key} 同時在 {owner} 與 {split}")
            forms[row["task"]][split].add(row["entity_form"])
    coverage = {}
    for task in TASKS:
        expected = set(FORMS[task])
        for split in SPLIT_NAMES:
            missing = expected - forms[task][split]
            if missing:
                raise ValueError(f"{split} 的 {task} 缺少字形 {sorted(missing)}，又會變成 v1 的分佈偏差")
        coverage[task] = {split: sorted(forms[task][split]) for split in SPLIT_NAMES}
    return coverage


def make_dataset(folder=DEFAULT_FOLDER, seed=42, tokenizer_dir=DEFAULT_TOKENIZER):
    folder = Path(folder).resolve()
    if not folder.is_relative_to(ROOT) or folder == ROOT:
        raise ValueError("資料輸出必須在專案內。")
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("輸出資料夾非空；請使用新位置，避免改動已凍結測試集。")
    tokenizer = BPETokenizer.load(Path(tokenizer_dir))
    splits = generate_records(seed)
    groups = validate_splits(splits)
    coverage = check_entity_isolation(splits)

    unknown = total_tokens = max_prompt = max_reply = 0
    for rows in splits.values():
        for row in rows:
            ids = build_prompt(tokenizer, row["prompt"], row["history"], max_tokens=512 - RESPONSE_BUDGET)
            reply_ids = tokenizer.encode(row["reply"])
            unknown += ids.count(UNK) + reply_ids.count(UNK)
            total_tokens += len(ids) + len(reply_ids)
            max_prompt, max_reply = max(max_prompt, len(ids)), max(max_reply, len(reply_ids) + 1)
    if unknown:
        raise ValueError(f"資料含{unknown}個UNK；請先調整實體／模板，不能用未知字測試記憶。")
    if max_reply > RESPONSE_BUDGET:
        raise ValueError("回答超過共同response_budget。")

    # 單輪保留能力：與 v1 相同的 regression（pairs 前 20 筆）與 replay，結果可跨版本比較
    pairs = json.loads((ROOT / "data/pairs.json").read_text(encoding="utf-8"))
    regression = [dict(row, history=[], task="legacy_regression", dialogue_id=f"legacy:{i}") for i, row in enumerate(pairs[:20])]
    replay = [dict(row, history=[], task="legacy_replay", dialogue_id=f"legacy:{i+20}") for i, row in enumerate(pairs[20:])]
    heldout = {row["prompt"] for row in regression}
    if any(row["prompt"] in heldout for rows in splits.values() for row in rows):
        raise ValueError("單輪回歸問題與多輪資料重疊。")

    folder.mkdir(parents=True)
    files = {}
    for name, rows in dict(splits, regression=regression, replay=replay).items():
        target = folder / f"{name}.json"
        target.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        files[target.name] = {"sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "records": len(rows),
                              "tasks": dict(Counter(row["task"] for row in rows)),
                              "kinds": dict(Counter(row.get("kind", row["task"]) for row in rows))}
    manifest = {"format": FORMAT_VERSION, "version": "context_dialogues_v2", "response_budget": RESPONSE_BUDGET,
                "seed": seed, "files": files, "groups": groups, "entity_form_coverage": coverage,
                "tokenizer_sha256": hashlib.sha256((Path(tokenizer_dir) / "bpe_vocab.json").read_bytes()).hexdigest(),
                "unk_tokens": unknown, "total_tokens": total_tokens, "max_prompt_tokens": max_prompt,
                "max_reply_tokens_with_eos": max_reply,
                "fixes_vs_v1": ["實體依字形分層切分，test 不再只含訓練沒見過的字形",
                                "實體池與句型擴大，句型依 split 隔離",
                                "加入閒聊、干擾事實與『有歷史但沒提過』的缺資訊樣本"],
                "limitations": ["仍是模板任務，只證明抄回前文指定欄位，不代表一般語意理解",
                                "字元與任務類型仍跨 split 共用；實體與句型不共用",
                                "regression 與 v1 相同（pairs 前 20 筆），不是新的世界知識測試"]}
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FOLDER)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER,
                        help="用來檢查 UNK 的 tokenizer（需與要微調的 checkpoint 相同）")
    args = parser.parse_args()
    report = make_dataset(args.out_dir, args.seed, args.tokenizer_dir)
    print(json.dumps({k: report[k] for k in ("files", "entity_form_coverage", "unk_tokens", "max_prompt_tokens")},
                     ensure_ascii=False, indent=2))

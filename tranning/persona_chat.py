"""persona_chat.py — 讓 /character 選到的角色真的「照使用者描述的性格」說話。

為什麼需要這支（2026-09-30，to_do_list.md #45）：
1. **真的 bug**：2026-09-10 一般聊天分支從 GRU 換成 transformer_chat.reply()
   之後，chats.smart_reply_traced() 的一般聊天分支完全沒用到 out_dir——
   /character 選了「周柯宇」，GUI 標籤雖然顯示周柯宇，實際回覆卻還是 sinco
   一般模型，角色卡的描述/氣質分數/對話範例一個都沒被用到。
2. 就算接回舊的 character_chat_runs（GRU，21 筆範例死記），它也只會背那
   21 句，任何沒背過的問法都會變成亂碼，而且所有角色共用同一顆 checkpoint，
   新角色只寫描述是完全沒有作用的。

這裡的做法（不呼叫任何外部 AI/API，符合 Rule 06；不用重新訓練就能生效）：
  (a) **角色範例優先**：使用者自己寫的角色對話（角色卡的 "prompt" 欄位、
      "examples" 欄位、或 data/character.json）用字面相似度比對，夠接近就
      直接回使用者寫的原句——那是最像「真人」的答案，模型生不出更好的。
      比對前會先把角色名字/暱稱拿掉（「柯宇在幹嘛呢?」≈「你在幹嘛」）。
  (b) **沒有範例時**：請 sinco 一般模型取樣產生多個候選回覆，用角色的
      氣質（卡片上的 traits 分數 + 從自由描述文字裡抓到的性格關鍵字）替
      每個候選打分，挑最符合性格的一個，再做「說話風格」改寫：把 sinco/AI
      身分句換掉或拿掉、只留口語的前一兩句、依性格補語尾助詞/稱呼/表情符號
      （高冷/冷淡這類性格則反過來刪掉語助詞、只留一句）。
  (c) **模型還沒訓練或完全生不出東西**：退回依性格寫好的短句（誠實標示在
      trace 裡），不回傳「模型尚未訓練」這種出戲訊息。

老實講清楚限制：(b) 的內容本身仍來自 sinco 一般模型（1746 萬參數、742 筆
對話），這層只能讓「語氣、用字、自稱」貼近角色，沒辦法讓模型真的理解一段
任意描述後自己推演出這個人會說什麼——那需要大量「角色描述→回覆」的訓練
資料（見 to_do_list.md #45 的後續建議）。要讓角色更像，最有效的方法仍是
在角色卡多寫幾組對話範例。

角色卡（tranning/characters/<名字>.json）可用欄位，全部選填（除了 name）：
    name          角色名字
    description   自由文字的性格描述（會被拿來抓性格關鍵字）
    traits        {"溫柔": 0~1, ...}（character_model.py / prompt_traits.py 產生）
    prompt        {"使用者說的話": "角色回覆" | ["回覆1", "回覆2"] | ""}（空字串略過）
    examples      [{"prompt": ..., "reply": ...}, ...]
    aliases       ["柯宇", "阿祖"]，比對範例前會從訊息裡拿掉的暱稱
    call_user     角色對使用者的稱呼，例如 "寶"（寵溺/深情型角色偶爾會帶上）

Usage:
    python persona_chat.py new --name 小晴 --description "個性活潑開朗，講話很直，有點毒舌但很體貼"
    python persona_chat.py chat --name 周柯宇
"""

import argparse
import difflib
import json
import random
import re
import sys
from pathlib import Path
from typing import Callable

if getattr(sys, "frozen", False):
    _PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHARACTERS_DIR = _PROJECT_ROOT / "tranning" / "characters"
LEGACY_EXAMPLES_PATH = _PROJECT_ROOT / "data" / "character.json"

RETRIEVAL_THRESHOLD = 0.7
DEFAULT_CANDIDATES = 6
_DESCRIPTION_TRAIT_STRENGTH = 0.8  # 只在描述裡被提到、沒有滑桿分數的性格，給這個強度

# 每種性格：reply 裡出現就加分的用字、偏好的語尾、可選的表情、完全沒有候選時的備用短句。
# 標籤名稱跟 prompt_traits.py / character_traits_train.json 既有的標籤一致，另外補上
# 常見的相反型性格（高冷/沉穩/傲嬌…），讓不是「寵溺男友」型的角色也有依據。
TRAIT_STYLES: dict[str, dict] = {
    "溫柔": {"words": ["慢慢", "沒關係", "我在", "陪你", "好嗎", "別擔心", "辛苦了"],
             "endings": ["呢", "喔"], "fallback": ["嗯，我在聽，你慢慢說就好", "沒關係的，我陪你"]},
    "體貼": {"words": ["記得", "照顧", "休息", "吃飯", "身體", "累", "早點睡"],
             "endings": ["喔"], "fallback": ["今天有沒有好好休息？別太累了喔"]},
    "寵溺": {"words": ["寶", "乖", "都聽你", "想你", "抱抱", "我的", "依你"],
             "endings": ["呀", "~"], "fallback": ["乖~都依你", "好好好，都聽你的"]},
    "俏皮": {"words": ["嘿嘿", "哼", "才不", "猜猜", "偷偷"],
             "endings": ["啦", "~"], "fallback": ["嘿嘿，你猜猜看啊"]},
    "深情": {"words": ["一直", "永遠", "身邊", "想你", "喜歡你", "只有你"],
             "endings": ["呢"], "fallback": ["不管怎樣，我都會一直在你身邊"]},
    "保護慾": {"words": ["保護", "別怕", "有我在", "誰欺負你", "交給我"],
               "endings": ["！"], "fallback": ["別怕，有我在"]},
    "可愛": {"words": ["嘻嘻", "好耶", "嗚嗚", "好好"],
             "endings": ["啦", "~"], "emoji": ["🥰", "✨", "😆"], "fallback": ["嘻嘻，好耶~"]},
    "撒嬌": {"words": ["嘛", "人家", "陪我", "抱抱", "不管"],
             "endings": ["嘛", "~"], "emoji": ["🥺"], "fallback": ["人家想你了嘛~"]},
    "帥氣": {"words": ["交給我", "沒問題", "小事", "走吧"],
             "endings": [], "fallback": ["小事，交給我"]},
    "幽默": {"words": ["哈哈", "笑死", "開玩笑", "好啦"],
             "endings": ["哈哈"], "emoji": ["😂"], "fallback": ["哈哈，你也太好笑了吧"]},
    "性感": {"words": ["過來", "靠近", "想要"],
             "endings": [], "fallback": ["過來，靠近一點"]},
    "活潑": {"words": ["哇", "超", "好耶", "走", "一起"],
             "endings": ["！", "啦"], "emoji": ["😆"], "fallback": ["哇！聽起來超好玩的！"]},
    "高冷": {"words": ["嗯", "隨便", "還好", "喔"],
             "endings": [], "terse": True, "fallback": ["嗯。", "還好。"]},
    "沉穩": {"words": ["我覺得", "其實", "先", "慢慢來"],
             "endings": [], "fallback": ["先別急，我們一步一步來"]},
    "理性": {"words": ["因為", "所以", "其實", "建議"],
             "endings": [], "fallback": ["我覺得可以先想清楚原因再決定"]},
    "傲嬌": {"words": ["哼", "才不是", "勉強", "又不是"],
             "endings": ["哼"], "fallback": ["哼，才不是特地關心你的"]},
    "毒舌": {"words": ["笨", "拜託", "你喔", "真是"],
             "endings": [], "fallback": ["拜託，你喔……真拿你沒辦法"]},
    "害羞": {"words": ["那個", "嗯…", "其實"],
             "endings": ["…"], "fallback": ["那、那個……嗯"]},
}

# 描述文字裡常見的說法 → 對應到上面的標籤（描述是自由文字，使用者不會剛好寫標籤名）。
TRAIT_SYNONYMS: dict[str, str] = {
    "陽光": "活潑", "開朗": "活潑", "外向": "活潑", "熱情": "活潑", "元氣": "活潑",
    "壞學生": "俏皮", "調皮": "俏皮", "痞": "俏皮", "愛鬧": "俏皮",
    "抱抱": "撒嬌", "黏人": "撒嬌", "愛撒嬌": "撒嬌",
    "寵": "寵溺", "疼人": "寵溺",
    "細心": "體貼", "貼心": "體貼", "照顧人": "體貼",
    "暖": "溫柔", "溫和": "溫柔",
    "專情": "深情", "痴情": "深情",
    "搞笑": "幽默", "有趣": "幽默", "逗": "幽默",
    "冷淡": "高冷", "冷漠": "高冷", "話少": "高冷", "酷": "高冷",
    "穩重": "沉穩", "成熟": "沉穩", "冷靜": "沉穩",
    "理智": "理性",
    "嘴硬": "傲嬌",
    "嘴賤": "毒舌", "講話很直": "毒舌", "直接": "毒舌",
    "內向": "害羞", "靦腆": "害羞",
}

# sinco 一般模型的自我介紹/AI 身分用語——出現在角色回覆裡就「出戲」。
_IDENTITY_WORDS = ["sinco", "Sinco", "SINCO", "人工智慧", "語言模型", "聊天模型", "AI", "機器人", "程式"]
_NOT_TRAINED_MARKERS = ("尚未訓練", "not trained")
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?~])")
_STRIP_FOR_MATCH = re.compile(r"[\s，,。．.！!？?~～、…:：;；\"'「」()（）]+")
_ENDING_PUNCT = "。！？!?~～…"


# --- 角色卡 ---------------------------------------------------------------

def _safe_filename(name: str) -> str:
    # 跟 character_model._safe_filename() 同一套規則（那支會 import torch，這裡不依賴它）
    cleaned = re.sub(r"[^\w\-一-鿿]", "_", name).strip("_")
    return cleaned or "character"


def card_path(name: str, characters_dir: Path = CHARACTERS_DIR) -> Path:
    return characters_dir / f"{_safe_filename(name)}.json"


def load_card(name: str, characters_dir: Path = CHARACTERS_DIR) -> dict | None:
    """依角色卡裡存的 "name" 欄位找卡，不拿 name 去拼檔案路徑：
    (1) character_model.py 存檔時會把名字轉成安全檔名（「Alice Smith」→
    Alice_Smith.json），拿顯示名稱當檔名會找不到；(2) name 可能來自網頁
    /api/chat 的使用者輸入，拼路徑會讓「../../data/xxx」讀到資料夾外的檔案。"""
    if not name or not characters_dir.exists():
        return None
    for path in sorted(characters_dir.glob("*.json")):
        try:
            card = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(card, dict) and card.get("name") == name:
            return card
    return None


def resolve_traits(card: dict) -> dict[str, float]:
    """角色卡的滑桿分數 + 從描述文字抓到的性格（取兩者較大值）。"""
    traits: dict[str, float] = {}
    for name, score in (card.get("traits") or {}).items():
        try:
            traits[name] = max(0.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            continue
    description = card.get("description") or ""
    found = {tag for tag in TRAIT_STYLES if tag in description}
    found |= {tag for phrase, tag in TRAIT_SYNONYMS.items() if phrase in description}
    for tag in found:
        traits[tag] = max(traits.get(tag, 0.0), _DESCRIPTION_TRAIT_STRENGTH)
    return traits


def _aliases(card: dict) -> list[str]:
    name = card["name"]
    aliases = [name] + list(card.get("aliases") or [])
    if len(name) == 3:  # 中文三字姓名：「周柯宇」也常被叫「柯宇」
        aliases.append(name[1:])
    return sorted({a for a in aliases if a}, key=len, reverse=True)


def card_examples(card: dict, legacy_path: Path = LEGACY_EXAMPLES_PATH) -> list[tuple[str, list[str]]]:
    """[(使用者說的話, [可用的角色回覆...])]。空字串回覆（使用者還沒寫完的草稿）略過。"""
    raw: list[tuple[str, object]] = list((card.get("prompt") or {}).items())
    raw += [(e.get("prompt", ""), e.get("reply", "")) for e in card.get("examples") or []]
    # data/character.json 是 to-do #14 時期的共用範例檔（沒有標記屬於哪個角色）——
    # 只有裡面真的提到這個角色的名字/暱稱時才當成它的範例，避免新角色
    # 被套上別人的台詞。
    if legacy_path.exists():
        try:
            legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            legacy = []
        text = json.dumps(legacy, ensure_ascii=False)
        if any(alias in text for alias in _aliases(card)):
            raw += [(e.get("prompt", ""), e.get("reply", "")) for e in legacy]

    merged: dict[str, list[str]] = {}
    for prompt, reply in raw:
        replies = reply if isinstance(reply, list) else [reply]
        replies = [r.strip() for r in replies if isinstance(r, str) and r.strip()]
        if prompt and replies:
            merged.setdefault(prompt.strip(), [])
            merged[prompt.strip()] += [r for r in replies if r not in merged[prompt.strip()]]
    return list(merged.items())


def _normalize(text: str, aliases: list[str]) -> str:
    for alias in aliases:
        text = text.replace(alias, "")
    return _STRIP_FOR_MATCH.sub("", text)


def match_example(message: str, card: dict, examples: list[tuple[str, list[str]]],
                  threshold: float = RETRIEVAL_THRESHOLD) -> tuple[str, list[str], float] | None:
    aliases = _aliases(card)
    query = _normalize(message, aliases)
    best, best_score = None, 0.0
    for prompt, replies in examples:
        target = _normalize(prompt, aliases)
        if not query and not target:
            score = 1.0  # 只叫了名字（「周柯宇」「柯宇」）
        elif not query or not target:
            continue
        else:
            score = difflib.SequenceMatcher(None, query, target).ratio()
        if score > best_score:
            best, best_score = (prompt, replies), score
    if best is None or best_score < threshold:
        return None
    return best[0], best[1], best_score


# --- 候選打分與說話風格 ------------------------------------------------------

def score_candidate(text: str, traits: dict[str, float]) -> float:
    score = 0.0
    for tag, strength in traits.items():
        style = TRAIT_STYLES.get(tag)
        if not style:
            continue
        score += strength * sum(text.count(w) for w in style["words"])
        if any(text.rstrip().endswith(e) for e in style["endings"]):
            score += 0.5 * strength
    score -= 3.0 * sum(text.count(w) for w in _IDENTITY_WORDS)
    if len(text) > 40:
        score -= (len(text) - 40) / 20
    if len(text.strip()) < 2:
        score -= 2.0
    return score


def _top_trait(traits: dict[str, float], key: str) -> tuple[str, float] | None:
    ranked = [(t, s) for t, s in sorted(traits.items(), key=lambda kv: -kv[1])
              if TRAIT_STYLES.get(t, {}).get(key)]
    return ranked[0] if ranked else None


def humanize(text: str, card: dict, traits: dict[str, float], rng: random.Random) -> str:
    """把一般模型的回覆改成角色的說話方式（不改變內容本身，只改語氣/自稱/長度）。"""
    name = card["name"]
    for word in ("sinco", "Sinco", "SINCO"):
        text = text.replace(word, name)
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    in_character = [s for s in sentences if not any(w in s for w in _IDENTITY_WORDS)]
    sentences = in_character or sentences
    terse = any(TRAIT_STYLES.get(t, {}).get("terse") and s >= 0.6 for t, s in traits.items())
    sentences = sentences[:1] if terse else sentences[:2]
    text = "".join(sentences).strip()
    if not text:
        return text

    if terse:
        return re.sub(r"[~～呀啦嘛喔呢]+$", "", text).rstrip("！!") or text

    ending = _top_trait(traits, "endings")
    if ending and ending[1] >= 0.5 and rng.random() < ending[1]:
        particle = rng.choice(TRAIT_STYLES[ending[0]]["endings"])
        core = text.rstrip(_ENDING_PUNCT)
        if not core.endswith(particle):
            text = core + particle

    call_user = card.get("call_user")
    affection = max(traits.get("寵溺", 0.0), traits.get("深情", 0.0))
    if call_user and affection >= 0.6 and call_user not in text and rng.random() < 0.4:
        core = text.rstrip(_ENDING_PUNCT)
        text = f"{core}，{call_user}{text[len(core):]}"

    emoji = _top_trait(traits, "emoji")
    if emoji and emoji[1] >= 0.7 and rng.random() < 0.3:
        text += rng.choice(TRAIT_STYLES[emoji[0]]["emoji"])
    return text


def fallback_reply(traits: dict[str, float], rng: random.Random) -> str:
    top = _top_trait(traits, "fallback")
    if top is None:
        return "嗯，我在聽，你說"
    return rng.choice(TRAIT_STYLES[top[0]]["fallback"])


def _default_generate(message: str) -> str:
    import transformer_chat  # 延遲載入：測試/沒裝 torch 的環境不需要它
    return transformer_chat.reply(message)


def persona_reply(message: str, card: dict, generate: Callable[[str], str] | None = None,
                  n_candidates: int = DEFAULT_CANDIDATES, rng: random.Random | None = None,
                  examples: list[tuple[str, list[str]]] | None = None) -> tuple[str, str]:
    """回傳 (trace, reply)，trace 如實說明這句是怎麼來的（範例原句／挑選改寫／備用短句）。"""
    rng = rng or random.Random()
    generate = generate or _default_generate
    traits = resolve_traits(card)
    examples = card_examples(card) if examples is None else examples
    top = "、".join(t for t, _ in sorted(traits.items(), key=lambda kv: -kv[1])[:3]) or "未設定"

    matched = match_example(message, card, examples)
    if matched is not None:
        prompt, replies, score = matched
        return (f"角色「{card['name']}」→ 命中你寫的角色範例「{prompt}」（相似度 {score:.0%}，直接使用範例原句）",
                rng.choice(replies))

    candidates = []
    for _ in range(max(1, n_candidates)):
        try:
            text = generate(message)
        except Exception:  # 模型載入失敗之類——退回備用短句，不讓整個聊天噴錯
            break
        if not text or any(m in text for m in _NOT_TRAINED_MARKERS):
            break
        candidates.append(text)

    if not candidates:
        return (f"角色「{card['name']}」（性格：{top}）→ 沒有範例可用、一般模型也沒有產生回覆，"
                f"改用依性格寫好的備用短句", fallback_reply(traits, rng))

    scored = sorted(((score_candidate(humanize(c, card, traits, random.Random(0)), traits), i, c)
                     for i, c in enumerate(candidates)), key=lambda t: (-t[0], t[1]))
    _, index, chosen = scored[0]
    reply = humanize(chosen, card, traits, rng) or fallback_reply(traits, rng)
    return (f"角色「{card['name']}」（性格：{top}）→ 沒有相近的角色範例，sinco 產生 {len(candidates)} 個候選，"
            f"依性格挑選第 {index + 1} 個並改寫成角色語氣", reply)


def create_card(name: str, description: str, call_user: str | None = None,
                characters_dir: Path = CHARACTERS_DIR) -> Path:
    """只用一段性格描述建立角色卡（性格分數由描述自動推得，之後可用 prompt_traits.py 微調）。
    已存在的角色卡不覆蓋（使用者可能手動寫過範例）。"""
    path = card_path(name, characters_dir)
    if path.exists() or load_card(name, characters_dir) is not None:
        raise FileExistsError(f"{path} 已存在，不覆蓋；請直接編輯該檔案")
    card = {"name": name, "description": description}
    card["traits"] = resolve_traits(card)
    card["prompt"] = {}
    if call_user:
        card["call_user"] = call_user
    characters_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="讓角色依照描述的性格說話（不需重新訓練）")
    sub = parser.add_subparsers(dest="command", required=True)
    p_new = sub.add_parser("new", help="用一段性格描述建立新角色卡")
    p_new.add_argument("--name", required=True)
    p_new.add_argument("--description", required=True)
    p_new.add_argument("--call-user", default=None, help="角色對你的稱呼，例如 寶")
    p_chat = sub.add_parser("chat", help="跟指定角色對話")
    p_chat.add_argument("--name", required=True)
    args = parser.parse_args()

    if args.command == "new":
        path = create_card(args.name, args.description, args.call_user)
        card = json.loads(path.read_text(encoding="utf-8"))
        print(f"已建立 {path}，推得性格：{card['traits'] or '（描述裡沒有抓到已知的性格詞）'}")
        return
    card = load_card(args.name)
    if card is None:
        print(f"找不到名字是「{args.name}」的角色卡（{CHARACTERS_DIR}）")
        return
    while True:
        text = input("You: ")
        if text.strip().lower() in {"exit", "quit"}:
            break
        trace, reply = persona_reply(text, card)
        print(f"  ({trace})\n{card['name']}: {reply}")


if __name__ == "__main__":
    main()

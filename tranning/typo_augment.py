"""typo_augment.py — 讓聊天/程式碼訓練資料多幾份「帶錯字」的版本，一起餵給
chats.py 訓練，讓 sinco/sinco-code 對輸入端的錯字/口誤更耐噪（往「像 GPT/
Gemini 那樣看得懂帶錯字的口語輸入」這個方向靠近）。

老實說清楚這是什麼、不是什麼：這是純統計層級的資料擴增——訓練資料裡同一個
正確答案配上好幾種「打歪的」輸入版本，模型學到的是「這些變形都對應同一個
答案」的記憶式關聯，不是真正解析語意去猜你打錯字前想打什麼。字元級 GRU
seq2seq 從零訓練、資料量只有幾十到幾百筆等級，做不到 GPT/Gemini 那種語言
模型級別的拼寫容錯；這支腳本能做到的，是讓現有資料集裡「同一題」多幾種
常見輸入噪聲的版本，藉此提高模型對這些噪聲的覆蓋率。

只對 prompt（使用者輸入端）加噪，reply（正確答案）維持原樣不動——訓練目標
永遠是「不管輸入怎麼被打歪，都要輸出同一個正確答案」。

噪聲種類（逐字元決定要不要套用，--rate 是每個字元的觸發機率）：
  刪除     把這個字元整個拿掉（漏打）
  重複     把這個字元多打一次（連擊）
  相鄰互換 跟下一個字元互換位置（手滑打反）
  替換     換成別的字元——拉丁字母用 QWERTY 鍵盤相鄰鍵，其餘字元（含中文）
           從同一份訓練資料的字元詞彙表隨機抽一個（沒有真正的同音字/字形
           相似表，純隨機替換，涵蓋率有限，是刻意的簡化，避免引入額外的
           詞典資源）

Usage:
    python typo_augment.py --data data/pairs.json --out data/pairs.json --variants 2 --rate 0.15
    python typo_augment.py --data data/code_pairs.json --out data/code_pairs.json --variants 2 --rate 0.1
    python typo_augment.py --data data/pairs.json --out data/pairs_aug.json --variants 3 --rate 0.2 --val-ratio 0.2
"""

import argparse
import json
import random
import string
from pathlib import Path

from dataset_import import write_manifest

_QWERTY_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]


def _build_keyboard_neighbors() -> dict[str, list[str]]:
    neighbors: dict[str, list[str]] = {}
    for row in _QWERTY_ROWS:
        for i, ch in enumerate(row):
            options = []
            if i > 0:
                options.append(row[i - 1])
            if i < len(row) - 1:
                options.append(row[i + 1])
            neighbors[ch] = options
    return neighbors


_KEYBOARD_NEIGHBORS = _build_keyboard_neighbors()


def _build_char_pool(entries: list[dict], field: str) -> list[str]:
    pool = {ch for e in entries for ch in e.get(field, "") if not ch.isspace()}
    return sorted(pool) or list(string.ascii_lowercase)


def inject_typos(text: str, rate: float, rng: random.Random, char_pool: list[str]) -> str:
    """回傳 text 的一個帶錯字版本。rate 是每個字元觸發噪聲的機率，char_pool
    是「替換」操作用的候選字元池（來自整份訓練資料的詞彙，涵蓋中英文）。
    """
    chars = list(text)
    out: list[str] = []
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch.isspace() or rng.random() >= rate:
            out.append(ch)
            i += 1
            continue
        op = rng.choice(("delete", "duplicate", "swap", "substitute"))
        if op == "delete":
            i += 1
        elif op == "duplicate":
            out.append(ch)
            out.append(ch)
            i += 1
        elif op == "swap" and i + 1 < len(chars):
            out.append(chars[i + 1])
            out.append(ch)
            i += 2
        else:
            lower = ch.lower()
            if lower in _KEYBOARD_NEIGHBORS:
                repl = rng.choice(_KEYBOARD_NEIGHBORS[lower])
                out.append(repl.upper() if ch.isupper() else repl)
            else:
                out.append(rng.choice(char_pool))
            i += 1
    result = "".join(out).strip()
    return result or text  # 全部被刪光時退回原文，避免產生空字串當訓練資料


def augment_pairs(entries: list[dict], variants: int, rate: float, seed: int = 42) -> list[dict]:
    """回傳「原始 entries + 每筆額外 variants 個 prompt 帶錯字版本」的清單。
    prompt/reply 完全相同的重複版本（噪聲剛好沒生效，或不同次隨機剛好產生
    一樣的結果）會被去重，避免同一筆資料在訓練集裡被灌水。
    """
    rng = random.Random(seed)
    char_pool = _build_char_pool(entries, "prompt")
    augmented: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        key = (entry["prompt"], entry["reply"])
        if key not in seen:
            seen.add(key)
            augmented.append(entry)
        for _ in range(variants):
            noisy_prompt = inject_typos(entry["prompt"], rate, rng, char_pool)
            key = (noisy_prompt, entry["reply"])
            if key in seen:
                continue
            seen.add(key)
            augmented.append({"prompt": noisy_prompt, "reply": entry["reply"]})
    return augmented


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help='原始 [{"prompt", "reply"}, ...] JSON')
    parser.add_argument("--out", type=Path, required=True, help="輸出路徑（可以跟 --data 相同，直接就地擴增）")
    parser.add_argument("--variants", type=int, default=2, help="每筆額外產生幾個錯字版本")
    parser.add_argument("--rate", type=float, default=0.15, help="每個字元觸發噪聲的機率")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    args = parser.parse_args()

    entries = json.loads(args.data.read_text(encoding="utf-8"))
    augmented = augment_pairs(entries, variants=args.variants, rate=args.rate, seed=args.seed)

    if args.out.resolve() == args.data.resolve():
        # write_manifest() 會讀 out 既有內容再合併——如果 out 就是 --data
        # 本身，等於把 augmented（已經含原始資料）跟自己合併一次，結果不變
        # 但白白多做一次 I/O，這裡直接整份覆寫。
        args.out.write_text(json.dumps(augmented, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"共 {len(augmented)} 筆（含原始 {len(entries)} 筆）就地寫入 {args.out}")
        return

    summary = write_manifest(augmented, args.out, val_ratio=args.val_ratio, seed=args.seed)
    print(f"共 {summary['total']} 筆（含原始 {len(entries)} 筆）")
    if summary["val_path"]:
        print(f"  train: {summary['train_path']}（{summary['train_count']} 筆）")
        print(f"  val:   {summary['val_path']}（{summary['val_count']} 筆）")
    else:
        print(f"  寫入: {summary['train_path']}")


if __name__ == "__main__":
    main()

"""code_retrieval.py — 檢索式程式碼助手：先從 data/code_pairs.json 找語意最
接近的既有範例，接近到一定程度就直接回傳那筆範例的 reply，而不是靠
sinco-code（資料量小的字元級 seq2seq）憑空生成。這是「像 Claude Code 一樣
寫程式」的務實版本——Claude Code 的高成功率來自幾千億參數的預訓練模型，
從零訓練、資料量幾十到幾百筆等級的 sinco-code 做不到那個等級；檢索式能做到
的是「資料庫裡有覆蓋到的題目，回答會是完全正確、能跑的既有程式碼」，資料庫
沒覆蓋到的題目，才退回 sinco-code 生成（品質仍受限於小模型，見 chats.py
開頭的架構說明）。

相似度用 difflib.SequenceMatcher（標準庫，不需要額外套件或 embedding 模型）
比較新的 prompt 跟資料庫裡每一筆 prompt 的字元序列相似度，取最高分那筆。
這是字面相似度，不是語意相似度——「寫一個氣泡排序法」跟「寫一個氣泡排序」
分數會很高，但「寫一個氣泡排序法」跟「寫一個由小到大排列的函式」分數會
很低，即使兩者其實是在問同一件事，這是刻意的簡化（避免引入額外的
embedding 模型或外部 API，維持 Rule 06「全部自建、輕量」的邊界）。

呼叫端（chats.smart_reply_traced()）的實際用法：只有 retrieve() 相似度低於
threshold（回傳 None）才會退回 mc_chat_reply() 用 sinco-code 生成；一旦命中，
回傳的是資料庫裡的原文，不經過任何模型，正確性等於資料本身的正確性。

Usage（CLI 快速測試用，正式呼叫路徑是 chats.smart_reply_traced()）：
    python code_retrieval.py "用 Python 寫一個氣泡排序法"
"""

import difflib
import json
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    _PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent

CODE_PAIRS_PATH = _PROJECT_ROOT / "data" / "code_pairs.json"

DEFAULT_THRESHOLD = 0.6


def load_code_pairs(path: Path = CODE_PAIRS_PATH) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def best_match(message: str, pairs: list[dict]) -> tuple[dict | None, float]:
    """回傳 (最相似的那筆 pair, 相似度 0~1)；pairs 是空清單就回傳 (None, 0.0)。"""
    best_pair = None
    best_score = 0.0
    for pair in pairs:
        score = difflib.SequenceMatcher(None, message, pair["prompt"]).ratio()
        if score > best_score:
            best_score = score
            best_pair = pair
    return best_pair, best_score


def retrieve(message: str, threshold: float = DEFAULT_THRESHOLD,
             path: Path = CODE_PAIRS_PATH) -> tuple[str, float, str] | None:
    """相似度 >= threshold 才回傳 (reply, 相似度, 命中的原始 prompt)；
    否則回傳 None（呼叫端退回 sinco-code 生成）。
    """
    pairs = load_code_pairs(path)
    pair, score = best_match(message, pairs)
    if pair is None or score < threshold:
        return None
    return pair["reply"], score, pair["prompt"]


def main():
    if len(sys.argv) < 2:
        print('Usage: python code_retrieval.py "<prompt>"')
        return
    message = sys.argv[1]
    result = retrieve(message)
    if result is None:
        print("沒有找到夠相似的既有範例（低於門檻），實際使用時會退回 sinco-code 生成。")
        return
    reply, score, matched_prompt = result
    print(f"命中「{matched_prompt}」（相似度 {score:.0%}）\n\n{reply}")


if __name__ == "__main__":
    main()

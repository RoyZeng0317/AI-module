"""news_keyword_import.py — 把一個資料夾底下的新聞 .txt 文章，透過 jieba 斷詞 +
sklearn TF-IDF 抽每篇文章的關鍵字，轉成 chats.py 用的 prompt/reply pairs manifest，
沿用 dataset_import.py 的 write_manifest()，同一套「原始檔案 -> 訓練資料」慣例
（CLAUDE.md 需求 #04：手上沒有現成 JSON，但有一般文字檔案）。

這是 jieba/main.py 練習腳本（jieba_TFIDF()/read_news()）的正式版：來源資料夾改成
--source 參數（不再寫死 C:\\news），並多一步把「檔名+關鍵字」轉成
{"prompt": "這篇新聞的關鍵字是什麼：\\n<原文>", "reply": "詞1、詞2、..."} 這種
sinco 可以吃的格式，而不是只印出關鍵字。

輸出預設是 data/pairs_news_keywords_draft.json —— draft 檔，等你人工看過關鍵字
品質沒問題後，再自行合併進 data/pairs.json 重新訓練（跟 data/pairs_*_draft.json
既有慣例一樣，不會自動覆蓋主要訓練檔）。

Usage:
    python news_keyword_import.py --source C:\\news --out ../data/pairs_news_keywords_draft.json
    python news_keyword_import.py --source C:\\news --topk 8 --stopwords stopwords_zh.txt
"""

import argparse
from pathlib import Path

import jieba
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from dataset_import import write_manifest

DEFAULT_STOPWORDS = Path(__file__).with_name("stopwords_zh.txt")
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "pairs_news_keywords_draft.json"


def load_stopwords(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def read_news_folder(source: Path) -> tuple[list[str], list[str]]:
    """讀 source 底下所有檔案內容，回傳 (articles, filenames)，依檔名排序，
    順序穩定（跟 jieba/main.py 的 read_news() 用 list.append 順序不同）。"""
    files = sorted(p for p in source.iterdir() if p.is_file())
    articles = [p.read_text(encoding="utf-8-sig") for p in files]
    return articles, [p.name for p in files]


def extract_keywords_per_doc(articles: list[str], stopwords: set[str], topk: int = 5) -> list[list[tuple[str, float]]]:
    """對整個語料庫算 TF-IDF，回傳每篇文章依權重排序後的前 topk 個 (詞, 權重)。"""
    segmented = [" ".join(w for w in jieba.lcut(a, cut_all=False) if w.strip() and w not in stopwords) for a in articles]
    vectorizer = CountVectorizer()
    counts = vectorizer.fit_transform(segmented)
    vocab = vectorizer.get_feature_names_out()
    tfidf = TfidfTransformer().fit_transform(counts)
    assert isinstance(tfidf, csr_matrix)
    weights = tfidf.toarray()
    return [sorted(zip(vocab, row), key=lambda kv: kv[1], reverse=True)[:topk] for row in weights]


def build_pairs(articles: list[str], keywords_per_doc: list[list[tuple[str, float]]]) -> list[dict]:
    """把 (原文, 該篇關鍵字) 轉成 chats.py 吃的 prompt/reply。權重為 0 的詞（該篇
    其實沒有代表性關鍵字，例如全文都被停用詞濾光）不列入，這種文章整篇跳過。"""
    pairs = []
    for article, keywords in zip(articles, keywords_per_doc):
        words = [w for w, weight in keywords if weight > 0]
        if not words:
            continue
        pairs.append({"prompt": f"這篇新聞的關鍵字是什麼：\n{article.strip()}", "reply": "、".join(words)})
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="新聞 .txt 檔案所在資料夾")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stopwords", type=Path, default=DEFAULT_STOPWORDS)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    articles, filenames = read_news_folder(args.source)
    if not articles:
        print(f"{args.source} 底下沒有任何檔案，中止。")
        return

    keywords_per_doc = extract_keywords_per_doc(articles, load_stopwords(args.stopwords), topk=args.topk)
    for name, keywords in zip(filenames, keywords_per_doc):
        print(name, keywords)

    summary = write_manifest(build_pairs(articles, keywords_per_doc), args.out, val_ratio=args.val_ratio, seed=args.seed)
    print(summary)


if __name__ == "__main__":
    main()

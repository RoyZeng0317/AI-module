"""
最小可用 RAG（知識庫查詢）：資料夾 → 切片 → 向量 → NumPy 找最相近 → 組成提示詞

安裝：pip install numpy sentence-transformers requests
用法：
  1) 建索引：python rag_mini.py build  C:\\Users\\Roy\\knowledge
  2) 查詢  ：python rag_mini.py query  "INMP441 要接哪幾條線"
  3) 查詢並讓本機模型回答（需先安裝 Ollama 並下載模型）：
            python rag_mini.py ask    "INMP441 要接哪幾條線"
"""
import json
import sys
from pathlib import Path

import numpy as np

INDEX_DIR = Path("rag_index")
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 小、支援中文
OLLAMA_MODEL = "qwen2.5:7b"                                                  # 換成你裝的模型
CHUNK_SIZE, OVERLAP, TOP_K = 300, 50, 3

_embedder = None


def embed(texts):
    """把文字變成向量。意思越接近，向量方向越接近。"""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    vecs = _embedder.encode(texts, convert_to_numpy=True)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)   # 長度統一成 1，方便比較


# ---------- 1. 切片：長文章切成小段，每段稍微重疊，避免一句話被切斷 ----------
def split_text(text):
    text = text.strip()
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - OVERLAP
    return [c for c in chunks if c.strip()]


# ---------- 2. 建索引：所有片段轉成向量存起來 ----------
def build(folder):
    records = []
    for f in sorted(Path(folder).rglob("*")):
        if f.suffix.lower() in {".txt", ".md"}:
            for chunk in split_text(f.read_text(encoding="utf-8", errors="ignore")):
                records.append({"source": f.name, "text": chunk})
    if not records:
        sys.exit("資料夾裡沒有 .txt 或 .md 檔")

    vecs = embed([r["text"] for r in records])
    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", vecs)
    (INDEX_DIR / "chunks.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print(f"完成：{len(records)} 個片段，向量形狀 {vecs.shape}")


# ---------- 3. 搜尋：問題也轉成向量，用內積找最像的片段 ----------
def search(question, k=TOP_K):
    vecs = np.load(INDEX_DIR / "vectors.npy")
    records = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    q = embed([question])[0]
    scores = vecs @ q                      # 一次算完全部片段的相似度（-1 ~ 1）
    top = np.argsort(scores)[::-1][:k]     # 分數由高到低取前 k 個
    return [(float(scores[i]), records[i]) for i in top]


# ---------- 4. 組提示詞：把找到的資料塞給模型 ----------
def build_prompt(question, hits, min_score=0.3):
    refs = [f"[筆記: {r['source']}] {r['text']}" for s, r in hits if s >= min_score]
    ref_text = "\n".join(refs) if refs else "（沒有找到相關資料）"
    system = ("你是 Roy 的個人助手。回答時優先使用【參考資料】；資料裡沒有的，就說不確定。\n\n"
              f"【參考資料】\n{ref_text}")
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def ask(question):
    import requests
    messages = build_prompt(question, search(question))
    r = requests.post("http://localhost:11434/api/chat",
                      json={"model": OLLAMA_MODEL, "messages": messages, "stream": False}, timeout=300)
    print(r.json()["message"]["content"])


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cmd, arg = sys.argv[1], sys.argv[2]
    if cmd == "build":
        build(arg)
    elif cmd == "query":
        for score, r in search(arg):
            print(f"{score:.3f}  [{r['source']}]  {r['text'][:80]}...")
        print("\n--- 會送給模型的提示詞 ---")
        print(build_prompt(arg, search(arg))[0]["content"])
    elif cmd == "ask":
        ask(arg)
    else:
        sys.exit(__doc__)

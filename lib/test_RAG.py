"""Pipeline smoke test for lib/RAG.py -- proves ingest_paths() -> retrieve()
runs end-to-end against a real (but tiny, untrained-to-convergence)
SentenceEncoder, and that re-ingesting the same file doesn't duplicate
chunks. Does NOT claim real semantic-retrieval quality -- a one-epoch
encoder's vectors are close to random (same contract as
tranning/test_embedding_model.py).
"""

from pathlib import Path

import lib.RAG as rag
from embedding_model import train


def _train_tiny_encoder(tmp_path: Path) -> Path:
    embed_dir = tmp_path / "embed_runs"
    sentences = [
        "sinco 是一個從零打造的聊天模型。",
        "hello world, sinco is a small language model.",
        "今天天氣真好，適合出門散步。",
        "the weather is nice today, good for a walk.",
    ]
    train(sentences=sentences, out_dir=embed_dir, epochs=1,
          batch_size=4, max_len=16, n_layer=2, n_embd=16, n_head=2, vocab_size=80)
    return embed_dir


def test_ingest_and_retrieve_round_trip(tmp_path):
    embed_dir = _train_tiny_encoder(tmp_path)
    store_dir = tmp_path / "rag_store"
    rag._collection_cache.clear()

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text(
        "Sinco 是本專案自己從零訓練的聊天模型，不使用外部雲端 API。\n\n"
        "RAG 的檢索步驟使用自己訓練的 embedding 模型，存放在本地的向量資料庫裡。",
        encoding="utf-8",
    )

    written = rag.ingest_paths([doc_path], store_dir=store_dir, embed_out_dir=embed_dir)
    assert written > 0

    hits = rag.retrieve("Sinco 是什麼模型", store_dir=store_dir, embed_out_dir=embed_dir)
    assert len(hits) > 0
    assert hits[0]["source"] == str(doc_path)


def test_reingest_same_file_does_not_duplicate(tmp_path):
    embed_dir = _train_tiny_encoder(tmp_path)
    store_dir = tmp_path / "rag_store"
    rag._collection_cache.clear()

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("Sinco 是本專案自己從零訓練的聊天模型。", encoding="utf-8")

    first = rag.ingest_paths([doc_path], store_dir=store_dir, embed_out_dir=embed_dir)
    second = rag.ingest_paths([doc_path], store_dir=store_dir, embed_out_dir=embed_dir)

    collection = rag.get_collection(store_dir=store_dir, embed_out_dir=embed_dir)
    assert collection.count() == first == second


def test_retrieve_returns_empty_list_when_store_is_empty(tmp_path):
    embed_dir = _train_tiny_encoder(tmp_path)
    store_dir = tmp_path / "rag_store"
    rag._collection_cache.clear()

    assert rag.retrieve("hello", store_dir=store_dir, embed_out_dir=embed_dir) == []


def test_sinco_embedding_function_raises_clear_error_without_checkpoint(tmp_path):
    embedding_function = rag.SincoEmbeddingFunction(out_dir=tmp_path / "no_such_run")
    try:
        embedding_function(["hello"])
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "尚未訓練" in str(exc)

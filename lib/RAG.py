"""lib/RAG.py -- retrieval-augmented generation pipeline. Loads/splits
documents, embeds chunks with tranning/embedding_model.py's self-trained
sentence encoder, stores/queries them in a local chromadb collection, and
hands retrieved chunks to tranning/chats.py's smart_reply_traced() (sinco)
for the final reply. No external embedding/LLM API involved (Rule 06)."""

import hashlib
import sys
from pathlib import Path

import chromadb
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

if getattr(sys, "frozen", False):
    _PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TRANNING_DIR = str(_PROJECT_ROOT / "tranning")
if _TRANNING_DIR not in sys.path:
    sys.path.insert(0, _TRANNING_DIR)

from chats import DEFAULT_OUT_DIR as CHAT_OUT_DIR, smart_reply_traced  # noqa: E402
from embedding_model import DEFAULT_OUT_DIR as EMBED_OUT_DIR, embed_texts, load_encoder  # noqa: E402
# 調整區域
DEFAULT_STORE_DIR = _PROJECT_ROOT / "data" / "rag_store"  # chromadb 本地持久化資料夾路徑
DEFAULT_COLLECTION_NAME = "rag_chunks"  # chromadb 集合（collection）名稱
DEFAULT_CHUNK_SIZE = 500  # 文件切塊時每塊的字元數上限
DEFAULT_CHUNK_OVERLAP = 50  # 相鄰切塊之間重疊的字元數，避免語意被切斷


class SincoEmbeddingFunction(EmbeddingFunction[Documents]):
    """chromadb embedding function backed by embedding_model.py's
    self-trained sentence encoder -- the only model this pipeline's
    retrieval step uses, trained from scratch on this project's own corpora."""

    def __init__(self, out_dir: Path = EMBED_OUT_DIR):
        self.out_dir = Path(out_dir)
        self._loaded = None

    def _ensure_loaded(self):
        if self._loaded is None:
            loaded = load_encoder(self.out_dir)
            if loaded is None:
                raise RuntimeError(
                    f"embedding 模型尚未訓練，請先執行 python tranning/embedding_model.py train"
                    f"（checkpoint 應位於 {self.out_dir}）"
                )
            self._loaded = loaded
        return self._loaded

    def __call__(self, input: Documents) -> Embeddings:
        encoder, tokenizer, _max_len, _device = self._ensure_loaded()
        return embed_texts(list(input), encoder, tokenizer)

    @staticmethod
    def name() -> str:
        return "sinco-embedding"

    def get_config(self) -> dict:
        return {"out_dir": str(self.out_dir)}

    @staticmethod
    def build_from_config(config: dict) -> "SincoEmbeddingFunction":
        return SincoEmbeddingFunction(Path(config["out_dir"]))


_collection_cache: dict = {}


def get_collection(store_dir: Path = DEFAULT_STORE_DIR, collection_name: str = DEFAULT_COLLECTION_NAME,
                    embed_out_dir: Path = EMBED_OUT_DIR):
    key = (str(store_dir), collection_name, str(embed_out_dir))
    if key in _collection_cache:
        return _collection_cache[key]
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(store_dir))
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=SincoEmbeddingFunction(embed_out_dir),
    )
    _collection_cache[key] = collection
    return collection


def load_and_split(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE,
                    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    docs = TextLoader(str(path), encoding="utf-8").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [chunk.page_content for chunk in splitter.split_documents(docs)]


def ingest_paths(paths: list[Path], store_dir: Path = DEFAULT_STORE_DIR,
                  collection_name: str = DEFAULT_COLLECTION_NAME,
                  chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
                  embed_out_dir: Path = EMBED_OUT_DIR) -> int:
    """Loads + splits each file in `paths`, embeds the chunks, and upserts
    them into the local chroma collection. Deterministic ids (source path +
    chunk index) mean re-ingesting the same file updates its chunks instead
    of duplicating them. Returns the number of chunks written."""
    collection = get_collection(store_dir, collection_name, embed_out_dir)
    total = 0
    for path in paths:
        path = Path(path)
        chunks = load_and_split(path, chunk_size, chunk_overlap)
        if not chunks:
            continue
        ids = [hashlib.sha1(f"{path}::{i}".encode("utf-8")).hexdigest() for i in range(len(chunks))]
        metadatas = [{"source": str(path), "chunk": i} for i in range(len(chunks))]
        collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
        total += len(chunks)
    return total


def retrieve(query: str, k: int = 4, store_dir: Path = DEFAULT_STORE_DIR,
             collection_name: str = DEFAULT_COLLECTION_NAME,
             embed_out_dir: Path = EMBED_OUT_DIR) -> list[dict]:
    collection = get_collection(store_dir, collection_name, embed_out_dir)
    if collection.count() == 0:
        return []
    result = collection.query(query_texts=[query], n_results=min(k, collection.count()))
    hits = []
    for text, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        hits.append({"text": text, "source": meta.get("source", ""), "distance": distance})
    return hits


def rag_reply(query: str, k: int = 4, store_dir: Path = DEFAULT_STORE_DIR,
              collection_name: str = DEFAULT_COLLECTION_NAME, chat_out_dir: Path = CHAT_OUT_DIR,
              embed_out_dir: Path = EMBED_OUT_DIR, force_mode: str = "auto") -> tuple[str, str]:
    """Retrieves relevant chunks then hands them + the query to
    smart_reply_traced() (sinco) for the final reply. Honest caveat: sinco
    is a small character-level model, so its ability to actually make use
    of a long retrieved context is limited (see transformer_chat.py's
    "honest scope" note) -- this wires retrieval into the existing
    generation path, it doesn't give sinco new reading-comprehension
    ability on its own."""
    hits = retrieve(query, k, store_dir, collection_name, embed_out_dir)
    if not hits:
        trace, reply = smart_reply_traced(query, out_dir=chat_out_dir, force_mode=force_mode)
        return f"[RAG] 向量庫目前是空的，沒有檢索到相關片段，直接交給 sinco 回答 -> {trace}", reply

    context = "\n\n".join(f"[來源 {i + 1}: {h['source']}]\n{h['text']}" for i, h in enumerate(hits))
    augmented = f"參考資料：\n{context}\n\n問題：{query}"
    trace, reply = smart_reply_traced(augmented, out_dir=chat_out_dir, force_mode=force_mode)
    sources = "、".join(sorted({h["source"] for h in hits}))
    return f"[RAG] 檢索到 {len(hits)} 個相關片段（來源：{sources}） -> {trace}", reply

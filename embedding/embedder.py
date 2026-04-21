import os
import re
import json
import hashlib
from numbers import Number
import requests
import chromadb
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

from ingestion.models import ArticleChunk
from config import (
    EMBEDDING_MODEL_NAME as MODEL_NAME,
    OLLAMA_API_BASE,
    CHROMA_DIR,
    COLLECTION_NAME,
    INSTRUCTION_DOC,
    INSTRUCTION_QUERY,
    CHROMA_KEEP_SESSIONS,
    DATA_DIR,
)

import logging
logger = logging.getLogger(__name__)


def _sanitize_metadata_list_for_chroma(values: list[object]) -> list[str] | list[int] | list[float] | list[bool] | None:
    """Normalize list metadata to Chroma-supported homogeneous primitive lists."""
    primitives = [item for item in values if item is not None and isinstance(item, (str, int, float, bool))]
    if not primitives:
        return None

    # Keep pure-bool lists as bool for downstream filters.
    if all(isinstance(item, bool) for item in primitives):
        return [bool(item) for item in primitives]

    # Normalize numeric lists to one numeric type.
    if all(isinstance(item, Number) and not isinstance(item, str) for item in primitives):
        if any(isinstance(item, float) for item in primitives):
            return [float(item) for item in primitives]
        return [int(item) for item in primitives]

    # If there is any mixed primitive type, stringify to keep list type uniform.
    if not all(isinstance(item, str) for item in primitives):
        return [str(item) for item in primitives]
    return [str(item) for item in primitives]


def _sanitize_metadata_for_chroma(metadata: dict) -> dict:
    """Drop/normalize unsupported metadata values before writing to ChromaDB."""
    sanitized: dict = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, list):
            normalized_list = _sanitize_metadata_list_for_chroma(value)
            if normalized_list is not None:
                sanitized[str(key)] = normalized_list
            continue
        if isinstance(value, (str, int, float, bool)):
            sanitized[str(key)] = value
            continue

        # Keep extra context by stringifying unsupported structured values.
        try:
            sanitized[str(key)] = json.dumps(value, ensure_ascii=False)
        except TypeError:
            sanitized[str(key)] = str(value)
    return sanitized

# ── Embedding cache (disk-backed) ─────────────────────────────────────────────
_EMBED_CACHE_PATH = Path(DATA_DIR) / "embedding_cache.json"
_embed_cache: dict[str, list[float]] = {}

def _load_embed_cache() -> None:
    global _embed_cache
    if _EMBED_CACHE_PATH.exists():
        try:
            with open(_EMBED_CACHE_PATH, "r", encoding="utf-8") as f:
                _embed_cache = json.load(f)
            logger.info(f"[EmbedCache] Loaded {len(_embed_cache)} cached embeddings")
        except Exception as e:
            logger.warning(f"[EmbedCache] Failed to load cache: {e}")
            _embed_cache = {}

def _save_embed_cache() -> None:
    try:
        _EMBED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_EMBED_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_embed_cache, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"[EmbedCache] Failed to save cache: {e}")

def _cache_key(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}||{text}".encode("utf-8")).hexdigest()[:16]

def _get_cached_embeddings(texts: list[str], model: str) -> tuple[list[list[float]], list[str]]:
    """Trả về (cached_embeddings, missing_texts_indices)."""
    if not _embed_cache:
        _load_embed_cache()
    embeddings: list[list[float] | None] = [None] * len(texts)
    missing_indices: list[int] = []
    for i, text in enumerate(texts):
        key = _cache_key(text, model)
        if key in _embed_cache:
            embeddings[i] = _embed_cache[key]
        else:
            missing_indices.append(i)
    return embeddings, missing_indices

def _cache_embeddings(texts: list[str], embeddings: list[list[float]], indices: list[int], model: str) -> None:
    for idx, emb in zip(indices, embeddings):
        key = _cache_key(texts[idx], model)
        _embed_cache[key] = emb
    _save_embed_cache()

# ── Helper: Goi Ollama Embedding API ───────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _get_single_batch(batch: list[str]) -> list[list[float]]:
    payload = {
        "model": MODEL_NAME,
        "input": batch
    }
    try:
        response = requests.post(f"{OLLAMA_API_BASE}/embed", json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get('embeddings', [])
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Không thể kết nối Ollama tại {OLLAMA_API_BASE}. "
            "Hãy chắc chắn Ollama đang chạy (ollama serve)."
        )

def _get_batch_embeddings(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    all_embeddings = []
    detected_dim = None
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            batch_embeds = _get_single_batch(batch)
            if batch_embeds:
                if detected_dim is None:
                    detected_dim = len(batch_embeds[0])
                all_embeddings.extend(batch_embeds)
        except Exception as e:
            logger.error(f"[Embedding] Batch {i//batch_size} fail after retries: {e}")
            raise
            
    if len(all_embeddings) != len(texts):
        raise RuntimeError(f"Embedding mismatch: expected {len(texts)} vectors, got {len(all_embeddings)}")
        
    return all_embeddings

def get_model():
    return None

def get_chroma_client() -> chromadb.ClientAPI:
    abs_dir = os.path.abspath(CHROMA_DIR)
    os.makedirs(abs_dir, exist_ok=True)
    return chromadb.PersistentClient(path=abs_dir)


def _cleanup_old_collections(
    client: chromadb.ClientAPI,
    base_name: str,
    keep: int = CHROMA_KEEP_SESSIONS,
) -> None:
    """
    Xoa các collection cũ và dọn dẹp rác vật lý trong thư mục chroma_db.
    Hỗ trợ dọn dẹp các thư mục UUID rác mà ChromaDB tự tạo ra trên Windows.
    """
    import shutil
    import time

    # 1. Dọn dẹp Collection trong DB
    pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)$")
    all_cols = client.list_collections()
    matched: list[tuple[int, str]] = []
    
    for col in all_cols:
        m = pattern.match(col.name)
        if m:
            matched.append((int(m.group(1)), col.name))

    matched.sort(key=lambda x: x[0])
    if len(matched) > keep:
        to_delete = matched[:-keep]
        for ts, name in to_delete:
            try:
                client.delete_collection(name)
                logger.info(f"[ChromaDB] Deleted old collection: '{name}'")
            except Exception as e:
                logger.warning(f"[ChromaDB] Failed to delete collection '{name}': {e}")
    try:
        abs_chroma_path = os.path.abspath(CHROMA_DIR)
        current_time = time.time()
        
        if os.path.exists(abs_chroma_path):
            for item in os.listdir(abs_chroma_path):
                item_path = os.path.join(abs_chroma_path, item)
                
                if os.path.isdir(item_path):
                    mtime = os.path.getmtime(item_path)
                    if len(item) == 36 and (current_time - mtime) > 3600:
                        try:
                            shutil.rmtree(item_path)
                            logger.info(f"[ChromaDB] Removed physical junk dir: {item}")
                        except Exception:
                            pass
    except Exception as e:
        logger.warning(f"[ChromaDB] Physical cleanup failed: {e}")


# ── Main Embedding Functions ──────────────────────────────────────────────────
def embed_chunks(
    chunks: list[ArticleChunk],
    batch_size: int = 100,
    instruction: str = INSTRUCTION_DOC,
) -> list[list[float]]:
    """Chuyen tat ca chunks thanh vectors (Batch Mode) với disk cache."""
    if not chunks:
        return []

    # Chuan bi toan bo text voi instruction
    texts = [f"{instruction}: {chunk.content}" for chunk in chunks]

    # Kiem tra cache
    cached_embeddings, missing_indices = _get_cached_embeddings(texts, MODEL_NAME)
    hit_count = len(texts) - len(missing_indices)

    if hit_count > 0:
        logger.info(f"[Embedding] Cache hit: {hit_count}/{len(texts)} chunks ({100*hit_count/len(texts):.0f}%)")

    if not missing_indices:
        # Tất cả đã có trong cache
        return cached_embeddings

    # Embed chỉ chunks chưa có trong cache
    missing_texts = [texts[i] for i in missing_indices]
    logger.info(f"[Embedding] Encoding {len(missing_texts)} new chunks via Ollama (batch mode)...")

    new_embeddings = _get_batch_embeddings(missing_texts, batch_size=batch_size)

    # Gộp cache + mới embed
    result: list[list[float]] = []
    new_idx = 0
    for i in range(len(texts)):
        if cached_embeddings[i] is not None:
            result.append(cached_embeddings[i])
        else:
            result.append(new_embeddings[new_idx])
            new_idx += 1

    # Lưu mới vào cache
    _cache_embeddings(texts, new_embeddings, missing_indices, MODEL_NAME)

    actual_dim = len(result[0])
    logger.info(f"[Embedding] Done. Vector dim: {actual_dim}")
    return result

def store_in_chromadb(
    chunks: list[ArticleChunk],
    embeddings: list[list[float]],
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """Luu vao ChromaDB. Tao collection moi cho moi phien so sanh (de khong xoa data cu)."""
    import time
    client = get_chroma_client()

    # Dung timestamp suffix de moi lan so sanh co collection rieng biet,
    # tranh xoa du lieu cu va cho phep incremental comparison sau nay
    session_name = f"{collection_name}_{int(time.time())}"

    collection = client.get_or_create_collection(
        name=session_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [chunk.chunk_id() for chunk in chunks]
    documents = [chunk.content for chunk in chunks]

    def _build_meta(chunk: ArticleChunk) -> dict:
        """Trích xuất và chuẩn hóa metadata để lưu vào ChromaDB.
        Đảm bảo các trường quan trọng luôn tồn tại và ở định dạng string.
        """
        # Khởi tạo từ metadata gốc
        m = dict(chunk.metadata)
        
        # Ghi đè/Bổ sung các trường định danh vị trí
        m["raw_index"] = chunk.raw_index
        m["sub_index"] = chunk.sub_index
        
        # Ép kiểu string cho các trường quan trọng (ChromaDB best practice cho filtering)
        m["doc_label"]      = str(m.get("doc_label", chunk.doc_label))
        m["breadcrumb"]     = str(m.get("breadcrumb", ""))
        m["article_number"] = str(m.get("article_number", chunk.article_number or ""))
        
        # Xử lý các trường nullable
        m["page"]  = str(m.get("page")) if m.get("page") is not None else ""
        m["title"] = str(m.get("title")) if m.get("title") is not None else ""

        return _sanitize_metadata_for_chroma(m)

    metadatas = [_build_meta(chunk) for chunk in chunks]

    # Luu vao ChromaDB
    BATCH = 100
    for i in range(0, len(ids), BATCH):
        end = min(i + BATCH, len(ids))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )

    logger.info(f"[ChromaDB] Saved {len(ids)} vectors to collection '{session_name}'")
    _cleanup_old_collections(client, collection_name, keep=CHROMA_KEEP_SESSIONS)
    return collection

def embed_and_store(
    chunks_a: list[ArticleChunk],
    chunks_b: list[ArticleChunk],
    collection_name: str = COLLECTION_NAME,
) -> tuple[chromadb.Collection, list[list[float]], list[list[float]]]:
    """Pipeline Buoc 3 dung Ollama Batch mode."""
    all_chunks = chunks_a + chunks_b

    logger.info(f"[Embedding] === STEP 3: EMBEDDING + VECTOR DB (Ollama Batch) ===")
    logger.info(f"[Embedding] Model: {MODEL_NAME} | Total chunks: {len(all_chunks)}")

    embeddings = embed_chunks(all_chunks)
    embeds_a   = embeddings[:len(chunks_a)]
    embeds_b   = embeddings[len(chunks_a):]

    collection = store_in_chromadb(all_chunks, embeddings, collection_name)
    return collection, embeds_a, embeds_b

def query_similar(
    query_text: str,
    collection: chromadb.Collection | None = None,
    collection_name: str = COLLECTION_NAME,
    n_results: int = 5,
    doc_label_filter: str | None = None,
) -> dict:
    """Truy van ChromaDB dung Ollama embedding.
    
    Uu tien dung 'collection' object neu duoc truyen vao (tranh bug ten collection sai).
    Neu khong co, fall back sang 'collection_name' (dung cho truong hop doc lap).
    """
    query_text_with_inst = f"{INSTRUCTION_QUERY}: {query_text}"
    
    # Lay embedding cho query (van dung ham batch nhung truyen 1 item)
    query_embedding_list = _get_batch_embeddings([query_text_with_inst])
    
    if not query_embedding_list:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        
    query_embedding = query_embedding_list[0]

    if collection is None:
        client = get_chroma_client()
        collection = client.get_collection(collection_name)

    where_filter = None
    if doc_label_filter:
        where_filter = {"doc_label": doc_label_filter}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    return results

def print_query_results(results: dict) -> None:
    """In ket qua truy van."""
    if not results["ids"] or not results["ids"][0]:
        print("  Khong tim thay ket qua.")
        return

    for i, (doc_id, doc, meta, dist) in enumerate(zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        similarity = 1 - dist
        label = meta.get("doc_label", "?")
        article = meta.get("article_number", "?")
        title = meta.get("title", "")

        print(f"\n  [{i+1}] {label} | {article} – {title}")
        print(f"      Similarity: {similarity:.4f}")
        print(f"      Noi dung  : {doc[:150].replace(chr(10), ' ')}...")


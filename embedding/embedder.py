import os
import requests
import chromadb
import re
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
)

import logging
logger = logging.getLogger(__name__)

# ── Helper: Goi Ollama Embedding API ───────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _get_single_batch(batch: list[str]) -> list[list[float]]:
    """Goi API cho mot batch duy nhat với retry."""
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
    """Goi API voi retry tung batch va theo doi dimension."""
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
    """Chuyen tat ca chunks thanh vectors (Batch Mode)."""
    if not chunks:
        return []

    logger.info(f"[Embedding] Encoding {len(chunks)} chunks via Ollama (batch mode)...")
    
    # Chuan bi toan bo text voi instruction linh hoat
    texts = [f"{instruction}: {chunk.content}" for chunk in chunks]
    
    # Goi API lay toan bo vectors
    embeddings = _get_batch_embeddings(texts, batch_size=batch_size)
    
    if not embeddings:
        raise RuntimeError(f"Lỗi: Không thể lấy embedding từ model {MODEL_NAME}. Check Ollama log.")

    actual_dim = len(embeddings[0])
    logger.info(f"[Embedding] Done. Vector dim: {actual_dim} (detected dynamically)")
    return embeddings

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
        m = dict(chunk.metadata)
        m["raw_index"] = chunk.raw_index
        m.setdefault("sub_index", chunk.sub_index)
        
        # Use empty string for nullable fields (ChromaDB best practice)
        m["page"] = str(m.get("page")) if m.get("page") is not None else ""
        m["title"] = str(m.get("title")) if m.get("title") is not None else ""
        
        return m

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


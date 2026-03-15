from pathlib import Path
import logging
from collections import Counter

logger = logging.getLogger(__name__)

from ingestion.loader import load_document
from ingestion.chunker import structure_document
from ingestion.models import ArticleChunk


def process_two_documents(
    file_path_a: str,
    file_path_b: str,
) -> tuple[list[ArticleChunk], list[ArticleChunk]]:
    doc_id_a = Path(file_path_a).name
    doc_id_b = Path(file_path_b).name

    logger.info(f"[Bước 1] Đọc tài liệu A: {doc_id_a}")
    paragraphs_a = load_document(file_path_a)
    logger.info(f"         → {len(paragraphs_a)} đoạn thô")

    logger.info(f"[Bước 1] Đọc tài liệu B: {doc_id_b}")
    paragraphs_b = load_document(file_path_b)
    logger.info(f"         → {len(paragraphs_b)} đoạn thô")

    logger.info("[Bước 2] Chuẩn hóa & nhận diện cấu trúc...")
    chunks_a = structure_document(paragraphs_a, doc_id=doc_id_a, doc_label="doc_A")
    chunks_b = structure_document(paragraphs_b, doc_id=doc_id_b, doc_label="doc_B")
    tops = Counter(c.metadata.get("top_level","") for c in chunks_a + chunks_b)
    mids = Counter(c.metadata.get("mid_level","") for c in chunks_a + chunks_b)
    logger.info(f"Phân bố Top-level: {dict(tops)}")
    logger.info(f"Phân bố Mid-level: {dict(mids)}")
    sub_a = sum(1 for c in chunks_a if c.sub_index > 0)
    sub_b = sum(1 for c in chunks_b if c.sub_index > 0)
    logger.info(f"         → doc_A: {len(chunks_a)} chunk (trong đó {sub_a} sub-chunk)")
    logger.info(f"         → doc_B: {len(chunks_b)} chunk (trong đó {sub_b} sub-chunk)")

    return chunks_a, chunks_b


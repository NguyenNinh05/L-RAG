"""
src/config.py
=============
Configuration loader — đọc từ configs/*.yaml và cung cấp interface
tương thích ngược với config.py cũ ở root level.

Usage:
    from src.config import get_config, EMBEDDING_DIM, MATCH_THRESHOLD

    cfg = get_config()
    print(cfg["alignment"]["match_threshold"])  # 0.65
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Đường dẫn gốc đến configs/
_CONFIG_DIR = Path(__file__).parent.parent / "configs"


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """
    Load và cache toàn bộ config từ pipeline_config.yaml và model_config.yaml.

    Returns:
        dict chứa toàn bộ config với keys: 'ingestion', 'alignment', 'comparison',
        'embedding', 'llm'.
    """
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        logger.warning("PyYAML chưa cài. Dùng fallback defaults. Chạy: pip install pyyaml")
        return _get_defaults()

    cfg: dict[str, Any] = {}

    for yaml_file in ["pipeline_config.yaml", "model_config.yaml"]:
        yaml_path = _CONFIG_DIR / yaml_file
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
                for key, val in loaded.items():
                    if isinstance(val, dict) and key in cfg and isinstance(cfg[key], dict):
                        cfg[key].update(val)
                    else:
                        cfg[key] = val
        else:
            logger.warning("Config file không tồn tại: %s. Dùng defaults.", yaml_path)

    base = _get_defaults()
    for key, val in cfg.items():
        if isinstance(val, dict) and key in base and isinstance(base[key], dict):
            base[key].update(val)
        else:
            base[key] = val
            
    return base


def _get_defaults() -> dict[str, Any]:
    """Fallback defaults khi YAML không load được."""
    return {
        "ingestion": {
            "max_chunk_chars": 2000,
            "overlap_chars": 200,
            "kuzu_db_path": "./data/processed/graph_db",
            "chroma_db_path": "./data/processed/chroma_db",
            "chroma_collection_name": "legal_documents",
            "confidence_threshold": 0.75,
        },
        "alignment": {
            "w_semantic": 0.6,
            "w_jaro_winkler": 0.3,
            "w_ordinal": 0.1,
            "match_threshold": 0.65,
            "split_merge_threshold": 0.80,
            "embed_batch_size": 32,
        },
        "comparison": {
            "fuzzy_match_threshold": 0.85,
            "min_evidence_length": 5,
            "strict_numerical": True,
            "max_concurrency": 4,
            "min_confidence_to_include": 0.4,
        },
        "embedding": {
            "model_name": "BAAI/bge-m3",
            "use_fp16": True,
            "batch_size": 16,
            "max_length": 1024,
            "embedding_dim": 1024,
        },
        "llm": {
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "base_url": "http://localhost:8000/v1",
            "temperature_acu": 0.05,
            "temperature_summary": 0.3,
            "max_tokens_acu": 4096,
            "max_tokens_summary": 1024,
        },
    }


# ---------------------------------------------------------------------------
# Backward-compatible constants (tương thích với config.py cũ ở root)
# ---------------------------------------------------------------------------

def _cfg_val(section: str, key: str, default: Any = None) -> Any:
    try:
        return get_config().get(section, {}).get(key, default)
    except Exception:
        return default


# Ingestion
CONFIDENCE_THRESHOLD: float = _cfg_val("ingestion", "confidence_threshold", 0.75)
MAX_CHUNK_CHARS: int = _cfg_val("ingestion", "max_chunk_chars", 2000)
OVERLAP_CHARS: int = _cfg_val("ingestion", "overlap_chars", 200)
KUZU_DB_PATH: str = _cfg_val("ingestion", "kuzu_db_path", "./data/processed/graph_db")
CHROMA_DB_PATH: str = _cfg_val("ingestion", "chroma_db_path", "./data/processed/chroma_db")

# Alignment
MATCH_THRESHOLD: float = _cfg_val("alignment", "match_threshold", 0.65)
SPLIT_MERGE_THRESHOLD: float = _cfg_val("alignment", "split_merge_threshold", 0.80)

# Embedding
EMBEDDING_DIM: int = _cfg_val("embedding", "embedding_dim", 1024)
DEFAULT_MODEL_NAME: str = _cfg_val("embedding", "model_name", "BAAI/bge-m3")

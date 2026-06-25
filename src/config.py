"""
src/config.py
=============
Configuration loader — đọc từ configs/*.yaml và cung cấp interface
tương thích ngược với config.py cũ ở root level.

Usage:
    from src.config import get_config, get_llm_config, EMBEDDING_DIM, MATCH_THRESHOLD

    cfg = get_config()
    print(cfg["alignment"]["match_threshold"])  # 0.65

    llm_cfg = get_llm_config()                  # resolved LLM config by provider
    llm_cfg = get_llm_config(provider="deepseek")  # force DeepSeek
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Đường dẫn gốc đến configs/
_CONFIG_DIR = Path(__file__).parent.parent / "configs"

# Load .env ngay khi module import (trước mọi config resolution)
try:
    from dotenv import load_dotenv as _load_dotenv
    _ENV_PATH = Path(__file__).parent.parent / ".env"
    if _ENV_PATH.exists():
        _load_dotenv(dotenv_path=_ENV_PATH, override=False)
        logger.debug("Loaded .env from %s", _ENV_PATH)
except ImportError:
    logger.debug("python-dotenv chưa cài — dùng biến môi trường hệ thống.")
except Exception:
    logger.debug("Không load được .env — dùng biến môi trường hệ thống.")


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
            "w_semantic": 0.5,
            "w_jaro_winkler": 0.25,
            "w_ordinal": 0.1,
            "w_sparse": 0.15,
            "match_threshold": 0.60,
            "split_merge_threshold": 0.80,
            "embed_batch_size": 32,
        },
        "comparison": {
            "fuzzy_match_threshold": 0.85,
            "min_evidence_length": 5,
            "strict_numerical": True,
            "max_concurrency": 4,
            "min_confidence_to_include": 0.2,
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
# LLM Provider Resolution — hỗ trợ "local" (Qwen) và "deepseek" (API)
# ---------------------------------------------------------------------------


def get_llm_provider() -> str:
    """
    Xác định LLM provider hiện tại.

    Thứ tự ưu tiên:
        1. Biến môi trường LLM_PROVIDER
        2. Config YAML (llm.provider)
        3. Mặc định "local"
    """
    env_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if env_provider in ("local", "deepseek"):
        return env_provider

    try:
        cfg_provider = get_config().get("llm", {}).get("provider", "local")
        if cfg_provider in ("local", "deepseek"):
            return cfg_provider
    except Exception:
        pass

    return "local"


def get_llm_config(provider: str | None = None) -> dict[str, Any]:
    """
    Trả về LLM config đã resolve theo provider.

    Với provider="local": dùng config từ model_config.yaml (Qwen/Ollama/vLLM).
    Với provider="deepseek": merge preset từ provider_presets.deepseek với biến
    môi trường (DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL).

    Args:
        provider: "local", "deepseek", hoặc None (auto-detect từ env/config).

    Returns:
        dict với các key: model_name, base_url, api_key, temperature_acu,
        temperature_summary, max_tokens_acu, max_tokens_summary,
        timeout_seconds, max_retries, provider.

    Raises:
        ValueError: Nếu provider="deepseek" và DEEPSEEK_API_KEY không được set.
    """
    if provider is None:
        provider = get_llm_provider()

    cfg = get_config()

    if provider == "local":
        llm = dict(cfg.get("llm", {}))
        llm["provider"] = "local"
        return llm

    if provider == "deepseek":
        # Lấy preset từ config YAML
        presets = cfg.get("provider_presets", {}).get("deepseek", {})

        # Override từ biến môi trường
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY không được set. "
                "Tạo file .env với DEEPSEEK_API_KEY=<key> hoặc export biến môi trường."
            )

        base_url = os.getenv("DEEPSEEK_BASE_URL", "").strip() or presets.get(
            "base_url", "https://api.deepseek.com/v1"
        )
        model_name = os.getenv("DEEPSEEK_MODEL", "").strip() or presets.get(
            "model_name", "deepseek-v4-flash"
        )

        return {
            "provider": "deepseek",
            "model_name": model_name,
            "base_url": base_url,
            "api_key": api_key,
            "temperature_acu": float(presets.get("temperature_acu", 0.0)),
            "temperature_summary": float(presets.get("temperature_summary", 0.2)),
            "max_tokens_acu": int(presets.get("max_tokens_acu", 8192)),
            "max_tokens_summary": int(presets.get("max_tokens_summary", 1024)),
            "timeout_seconds": float(presets.get("timeout_seconds", 180.0)),
            "max_retries": int(presets.get("max_retries", 3)),
        }

    raise ValueError(f"Unknown LLM provider: {provider!r}. Use 'local' or 'deepseek'.")


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

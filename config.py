from __future__ import annotations
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

EMBEDDING_MODEL_NAME  = "qwen3-embedding:0.6b"   
OLLAMA_API_BASE       = "http://localhost:11434/api"

INSTRUCTION_DOC   = "Represent this legal document for retrieval"
INSTRUCTION_QUERY = "Represent this query for retrieving legal documents"

CHROMA_DIR       = str(BASE_DIR / "chroma_db")
COLLECTION_NAME  = "legal_chunks"
# Số session ChromaDB tối đa giữ lại; các session cũ hơn sẽ bị xóa tự động
CHROMA_KEEP_SESSIONS = int(os.getenv("CHROMA_KEEP_SESSIONS", "3"))

OLLAMA_LLM_MODEL   = "hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M"
OLLAMA_CHAT_URL    = "http://localhost:11434/api/chat"
LLM_MAX_TOKENS     = 1500
LLM_TEMPERATURE    = 0.5
LLM_TOP_P          = 0.9
LLM_NUM_CTX        = 4096
LLM_PRESENCE_PENALTY = 1.2

UNCHANGED_THRESHOLD  = 0.95   
MODIFIED_THRESHOLD   = 0.75   
GAP_PENALTY          = 0.40   
TEXT_UNCHANGED_RATIO = 0.99   

# CORS: danh sách origin được phép, đọc từ env var khi deploy production
# VD: ALLOWED_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"
ALLOWED_ORIGINS: list[str] = os.getenv(
	"ALLOWED_ORIGINS", "http://localhost:8000,http://localhost:3000"
).split(",")

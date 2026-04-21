import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
SESSION_DB_PATH = str(DATA_DIR / "sessions.db")

EMBEDDING_MODEL_NAME  = "qwen3-embedding:0.6b"   
OLLAMA_API_BASE       = "http://localhost:11434/api"

INSTRUCTION_DOC   = "Represent this legal document for retrieval"
INSTRUCTION_QUERY = "Represent this query for retrieving legal documents"

CHROMA_DIR       = str(BASE_DIR / "chroma_db")
COLLECTION_NAME  = "legal_chunks"
# Số session ChromaDB tối đa giữ lại; các session cũ hơn sẽ bị xóa tự động
CHROMA_KEEP_SESSIONS = int(os.getenv("CHROMA_KEEP_SESSIONS", "5"))

OLLAMA_LLM_MODEL   = "hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M"
OLLAMA_CHAT_URL    = "http://localhost:11434/api/chat"
LLM_MAX_TOKENS     = 4096
LLM_TEMPERATURE    = 0.2
LLM_TOP_P          = 0.9
LLM_NUM_CTX        = 4096
LLM_PRESENCE_PENALTY = 1.2
LLM_ENABLE_REPORT = os.getenv("LLM_ENABLE_REPORT", "1").lower() not in {"0", "false", "no"}
LLM_MAX_CONCURRENT = int(os.getenv("LLM_MAX_CONCURRENT", "4"))  # Giới hạn concurrent LLM calls

UNCHANGED_THRESHOLD  = 0.95   
MODIFIED_THRESHOLD   = 0.75   
GAP_PENALTY          = 0.40   
TEXT_UNCHANGED_RATIO = 0.998   
NEAR_UNCHANGED_BAND  = float(os.getenv("NEAR_UNCHANGED_BAND", "0.02"))
CLAUSE_HINT_ALPHA    = float(os.getenv("CLAUSE_HINT_ALPHA", "0.08"))
CLAUSE_HINT_MIN_GAP  = int(os.getenv("CLAUSE_HINT_MIN_GAP", "3"))

# CORS: danh sách origin được phép, đọc từ env var khi deploy production
# VD: ALLOWED_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"
ALLOWED_ORIGINS: list[str] = os.getenv(
	"ALLOWED_ORIGINS", "http://localhost:8000,http://localhost:3000"
).split(",")

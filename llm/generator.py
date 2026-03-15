"""
llm/generator.py
================
Sinh bao cao so sanh hop dong bang LLM local qua Ollama API.
Dung Qwen3-4B (GGUF Q4_K_M) chay qua Ollama de toi uu VRAM.
"""

from __future__ import annotations
import asyncio
import nest_asyncio
nest_asyncio.apply()  # Cho phep asyncio.run() hoat dong trong thread co san event loop (api.py dung run_in_executor)
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm_asyncio
from retrieval.matcher import ComparedPair
from config import (
    OLLAMA_LLM_MODEL   as OLLAMA_MODEL,
    OLLAMA_CHAT_URL    as OLLAMA_API_URL,
    LLM_MAX_TOKENS     as MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_NUM_CTX,
    LLM_PRESENCE_PENALTY,
)

import logging
logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Bạn là chuyên gia so sánh hợp đồng tại một hãng luật hàng đầu. 
Nhiệm vụ của bạn là lập báo cáo so sánh chi tiết giữa hai phiên bản hợp đồng.

**Tài liệu so sánh:**
- Hợp đồng gốc: {FILE_A}
- Hợp đồng mới: {FILE_B}

ĐỊNH DẠNG BÁO CÁO BẮT BUỘC:
1. **Nội dung thay đổi:** Mô tả ngắn gọn bản chất sự thay đổi.

2. **Bảng so sánh chi tiết:**
| Hạng mục | {FILE_A} | {FILE_B} |
| :--- | :--- | :--- |
| [Tên điều khoản] | [Nội dung gốc] | [Nội dung mới] |

3. **Phân tích tác động:**
- **Rủi ro/Lợi ích:** Phân tích phiên bản mới mang lại lợi ích hay rủi ro so với phiên bản gốc.
- **Điểm cần lưu ý:** Các điều kiện đi kèm hoặc hệ quả pháp lý.

NGUYÊN TẮC:
- Dùng ngôn ngữ pháp lý trang trọng, chuẩn xác.
- CHỈ phân tích dựa trên text cung cấp. Nếu thiếu thông tin, ghi "Không có dữ liệu".
- Mọi trích dẫn phải trùng khớp với văn bản.
- Trả lời bằng tiếng Việt có dấu."""

# ── Call LLM via Ollama ───────────────────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
async def _call_llm_async(session, prompt: str, file_a: str, file_b: str, semaphore, i: int, label: str) -> str:
    """Goi Ollama API de chat completion (Async). Co retry khi that bai."""
    
    system = SYSTEM_PROMPT.replace("{FILE_A}", file_a).replace("{FILE_B}", file_b)
    
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature":       LLM_TEMPERATURE,
            "presence_penalty":  LLM_PRESENCE_PENALTY,
            "num_predict":       MAX_TOKENS,
            "top_p":             LLM_TOP_P,
            "num_ctx":           LLM_NUM_CTX,
        }
    }

    async with semaphore:
        try:
            timeout = aiohttp.ClientTimeout(total=90)
            async with session.post(OLLAMA_API_URL, json=payload, timeout=timeout) as response:
                response.raise_for_status()
                result = await response.json()
                return result['message']['content'].strip()
        except aiohttp.ClientConnectorError:
            raise RuntimeError(
                f"Không thể kết nối Ollama LLM tại {OLLAMA_API_URL}. "
                "Hãy chắc chắn Ollama đang chạy (ollama serve)."
            )
        except asyncio.TimeoutError:
            logger.warning(f"[LLM] Timeout mục {i+1}: {label}, retrying...")
            raise
        except Exception as e:
            logger.error(f"[LLM] Lỗi mục {i+1}: {label} - {e}")
            raise

# ── Build prompt ──────────────────────────────────────────────────────────────
def _build_prompt(pair: ComparedPair, file_a: str = "v1", file_b: str = "v2") -> str:
    label     = pair.chunk_a.article_number or "N/A"
    content_a = pair.chunk_a.content
    content_b = pair.chunk_b.content

    return (
        f"Hay so sanh su khac biet cua dieu khoan: **{label}**\n\n"
        f"--- NGUON DU LIEU ---\n"
        f"[{file_a}]:\n{content_a}\n\n"
        f"[{file_b}]:\n{content_b}\n\n"
        f"Hay tao bang so sanh va phan tich tac dong theo dung dinh dang yeu cau."
    )


def get_llm_model():
    return None

from datetime import datetime

# ── Main report function ──────────────────────────────────────────────────────
def generate_comparison_report(
    pairs: list[ComparedPair],
    file_a: str = "v1",
    file_b: str = "v2",
) -> str:
    """Sinh bao cao Markdown chuyen nghiep, phan tich TOAN BO thay doi."""
    modified = [p for p in pairs if p.match_type == "MODIFIED"]
    added    = [p for p in pairs if p.match_type == "ADDED"]
    deleted  = [p for p in pairs if p.match_type == "DELETED"]

    # Header
    lines = [
        f"# BÁO CÁO SO SÁNH CHI TIẾT VĂN BẢN PHÁP LÝ",
        "",
        f"**Ngày lập:** {datetime.now().strftime('%Y-%m-%d')}",
        f"**Đối tượng so sánh:**",
        f"- Gốc (v1): `{file_a}`",
        f"- Mới (v2): `{file_b}`",
        "",
        "## I. TỔNG QUAN THAY ĐỔI",
        f"Hệ thống đã phát hiện tổng cộng **{len(modified) + len(added) + len(deleted)}** điểm khác biệt đáng chú ý:",
        f"- **{len(modified)}** điều khoản bị sửa đổi nội dung.",
        f"- **{len(added)}** điều khoản/phụ lục được thêm mới.",
        f"- **{len(deleted)}** điều khoản bị loại bỏ.",
        "",
        "---",
        ""
    ]

    # Phan 1: Sua doi
    if modified:
        lines += ["## II. NỘI DUNG SỬA ĐỔI CHI TIẾT", ""]
        logger.info(f"[LLM] Analyzing {len(modified)} modified clauses...")

        async def _run_all_llm_requests(modified_pairs, batch_size=5):
            results = []
            semaphore = asyncio.Semaphore(2) 
            async with aiohttp.ClientSession() as session:
                for i in range(0, len(modified_pairs), batch_size):
                    batch = modified_pairs[i:i+batch_size]
                    tasks = []
                    for j, pair in enumerate(batch):
                        label = pair.chunk_a.article_number or "N/A"
                        prompt = _build_prompt(pair, file_a, file_b)
                        tasks.append(_call_llm_async(session, prompt, file_a, file_b, semaphore, i+j, label))
                    
                    batch_results = await tqdm_asyncio.gather(*tasks, desc=f"Tiến trình LLM (Batch {i//batch_size+1})")
                    results.extend(batch_results)
                    
                    # Memory cleanup
                    await asyncio.sleep(0.1)
                    
                return results

        analysis_results = asyncio.run(_run_all_llm_requests(modified))

        for i, (pair, analysis) in enumerate(zip(modified, analysis_results)):
            label = pair.chunk_a.article_number or "N/A"
            
            # Format lai page info net hon
            page_a = f" (trang {pair.chunk_a.page})" if pair.chunk_a.page else ""
            page_b = f" (trang {pair.chunk_b.page})" if pair.chunk_b.page else ""

            lines += [
                f"### {i+1}. {label}",
                f"*Do tuong dong ngu nghia: {pair.similarity:.2%}*",
                "",
                analysis,
                "",
                f"> **Nguon:** `{file_a}`{page_a} → `{file_b}`{page_b}",
                "",
                "---",
                ""
            ]

    # Phan 2: Them moi
    if added:
        lines += ["## III. ĐIỀU KHOẢN/PHỤ LỤC THÊM MỚI", ""]
        lines.append("| STT | Điều khoản | Vị trí (v2) | Nội dung tóm tắt |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for i, p in enumerate(added):
            label = p.chunk_b.article_number or "N/A"
            page = p.chunk_b.page or "-"
            # Thay vì cắt 150 ký tự, lấy toàn bộ nội dung và bỏ dấu xuống dòng để không làm vỡ bảng
            preview = p.chunk_b.content.replace("\n", " <br> ").replace("|", " ")
            lines.append(f"| {i+1} | {label} | Trang {page} | {preview} |")
        lines.append("")

    # Phan 3: Xoa bo
    if deleted:
        lines += ["## IV. ĐIỀU KHOẢN BỊ LOẠI BỎ", ""]
        lines.append("| STT | Điều khoản | Vị trí (v1) | Nội dung gốc |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for i, p in enumerate(deleted):
            label = p.chunk_a.article_number or "N/A"
            page = p.chunk_a.page or "-"
            # Tương tự như trên, giữ toàn bộ nội dung
            preview = p.chunk_a.content.replace("\n", " <br> ").replace("|", " ")
            lines.append(f"| {i+1} | {label} | Trang {page} | {preview} |")
        lines.append("")

    lines += [
        "",
        "**Ghi chu:**",
        "- Báo cáo này được hỗ trợ bởi công nghệ RAG (Retrieval-Augmented Generation) và LLM Qwen3.",
        "- Các phân tích mang tính chất tham khảo, cần được kiểm chứng bởi bộ phận chuyên môn."
    ]

    return "\n".join(lines)

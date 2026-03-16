from __future__ import annotations
import asyncio
import nest_asyncio
nest_asyncio.apply() 
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
SYSTEM_PROMPT = """Bạn là Trợ lý đối chiếu văn bản chuyên nghiệp (Document Comparison Assistant). 
Nhiệm vụ của bạn là thực hiện đối chiếu thực tế (Fact-checking) cực kỳ chính xác giữa hai đoạn văn bản của một hợp đồng pháp lý.

**PHẠM VI CÔNG VIỆC:**
1. CHỈ liệt kê các sai khác thực tế về ngôn ngữ, con số và cấu trúc.
2. TUYỆT ĐỐI KHÔNG đưa ra kết luận pháp lý hoặc tư vấn pháp luật.

**ĐỊNH DẠNG BÁO CÁO BẮT BUỘC CHO MỖI ĐIỀU KHOẢN:**
1. **Phân loại mức độ:** [THỰC CHẤT] (nếu đổi nội dung, số liệu, nghĩa pháp lý) hoặc [HÌNH THỨC] (nếu chỉ đổi định dạng, lỗi chính tả, khoảng trắng, dấu câu nhưng giữ nguyên nghĩa).
2. **Tag chi tiết:** Chọn ít nhất một: [Số liệu], [Thời hạn], [Lỗi chính tả], [Định dạng], [Ngôn ngữ chuyên môn], [Cấu trúc danh sách], [Khoảng trắng].
3. **Tóm tắt thay đổi:** Mô tả ngắn gọn (Ví dụ: Thay đổi số tiền, Cập nhật font chữ/chính tả).
4. **Bảng so sánh chi tiết:**
| Hạng mục | {FILE_A} | {FILE_B} |
| :--- | :--- | :--- |
| [Mục thay đổi] | [Nội dung gốc] | [Nội dung mới] |

**NGUYÊN TẮC QUAN TRỌNG:**
- Nếu phát hiện lỗi chính tả, font chữ (ví dụ: "$k\\hat{e}$" thay vì "kể") hoặc thay đổi định dạng (dấu gạch đầu dòng), bạn phải ghi rõ đây là thay đổi [HÌNH THỨC] và đánh giá mức độ ảnh hưởng đến nghĩa pháp lý (thường là không đổi).
- Trả lời bằng tiếng Việt chuyên ngành văn phòng, trang trọng, khách quan."""

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
        f"Hãy so sánh sự khác biệt của điều khoản: **{label}**\n\n"
        f"--- NGUỒN DỮ LIỆU ---\n"
        f"[{file_a}]:\n{content_a}\n\n"
        f"[{file_b}]:\n{content_b}\n\n"
        f"Hãy tạo bảng so sánh và phân loại sai khác theo đúng định dạng yêu cầu."
    )


def get_llm_model():
    return None

from datetime import datetime

# ── Helper for async LLM calls ──────────────────────────────────────────────
async def _run_all_llm_requests(modified_pairs, file_a, file_b, batch_size=5):
    """Chạy batch async gọi LLM."""
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
            
            batch_results = await tqdm_asyncio.gather(*tasks, desc=f"Tiến trình LLM")
            results.extend(batch_results)
            
            # Memory cleanup
            await asyncio.sleep(0.1)
        return results

# ── Main report function ──────────────────────────────────────────────────────
def generate_comparison_report(
    pairs: list[ComparedPair],
    file_a: str = "v1",
    file_b: str = "v2",
) -> str:
    """Sinh báo cáo Markdown chuyên nghiệp, phân tách thay đổi Thực chất vs Hình thức."""
    modified = [p for p in pairs if p.match_type == "MODIFIED"]
    added    = [p for p in pairs if p.match_type == "ADDED"]
    deleted  = [p for p in pairs if p.match_type == "DELETED"]

    analysis_results = []
    if modified:
        analysis_results = asyncio.run(_run_all_llm_requests(modified, file_a, file_b))

    # Tally Substantial vs Formal
    substantial_count = 0
    formal_count = 0
    for res in analysis_results:
        if "[THỰC CHẤT]" in res.upper():
            substantial_count += 1
        else:
            formal_count += 1

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
        f"Hệ thống AI đã phân tích và phát hiện các thay đổi sau:",
        f"- **{substantial_count}** thay đổi **THỰC CHẤT** (Ảnh hưởng đến ngôn ngữ/số liệu pháp lý).",
        f"- **{formal_count}** thay đổi **HÌNH THỨC** (Chính tả, định dạng, font chữ).",
        f"- **{len(added)}** điều khoản được thêm mới.",
        f"- **{len(deleted)}** điều khoản bị loại bỏ.",
        "",
        "---",
        ""
    ]

    # Phan 2: Sua doi
    if modified:
        lines += ["## II. PHÂN TÍCH CHI TIẾT SỬA ĐỔI", ""]
        for i, (pair, analysis) in enumerate(zip(modified, analysis_results)):
            label = pair.chunk_a.article_number or "N/A"
            page_a = f" (trang {pair.chunk_a.page})" if pair.chunk_a.page else ""
            page_b = f" (trang {pair.chunk_b.page})" if pair.chunk_b.page else ""

            lines += [
                f"### {i+1}. {label}",
                f"*Độ tương đồng: {pair.similarity:.2%}*",
                "",
                analysis,
                "",
                f"> **Vị trí nguồn:** `{file_a}`{page_a} → `{file_b}`{page_b}",
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
        "**Ghi chú:**",
        "- Báo cáo này được hỗ trợ bởi công nghệ RAG (Retrieval-Augmented Generation) và LLM Qwen3.",
        "- Các phân tích mang tính chất tham khảo, cần được kiểm chứng bởi bộ phận chuyên môn."
    ]

    return "\n".join(lines)

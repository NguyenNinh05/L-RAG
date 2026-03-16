"""
api.py — FastAPI Backend cho Legal RAG Comparison System
=========================================================
Endpoint chính:
  POST /api/compare
    - Nhận 2 file PDF/DOCX qua multipart/form-data
    - Stream progress qua Server-Sent Events (SSE)
    - Trả về Markdown report cuối cùng

  GET /
    - Serve trang chat UI

Usage:
  uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ── Logging setup (áp dụng cho toàn dự án) ────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Internal pipeline imports ──────────────────────────────────────────────────
from ingestion import process_two_documents
from embedding import embed_and_store
from retrieval import build_comparison_pairs
from llm import generate_comparison_report
from config import ALLOWED_ORIGINS

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Legal RAG Comparison API",
    description="API so sánh văn bản pháp lý dùng RAG + LLM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)

# Serve static UI
UI_DIR = Path(__file__).parent / "ui"
UI_DIR.mkdir(exist_ok=True)

# Mount static files (MUST be after other routes or handled carefully)
# We will mount it at the end of the file instead to avoid shadowing /api routes
# Or just mount it for specific extensions.
# Actually, the best way is to keep the / route and mount static files separately if they are in a subfolder, 
# but here they are in the same folder as index.html.


# ── SSE helper ─────────────────────────────────────────────────────────────────
def _sse(event: str, data: dict | str) -> str:
    """Format một SSE frame."""
    if isinstance(data, dict):
        payload = json.dumps(data, ensure_ascii=False)
    else:
        payload = json.dumps({"message": data}, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"



def _sanitize_filename(name: str) -> str:
    """Lấy basename và loại bỏ ký tự có thể gây Markdown/prompt injection.
    Chỉ giữ lại chữ cái, số, dấu gạch ngang, dấu chấm, dấu gạch dưới và khoảng trắng.
    """
    name = Path(name).name          # bỏ đường dẫn, chỉ lấy tên file
    name = re.sub(r"[^\w\-\. ]", "_", name)  # thay ký tự đặc biệt bằng _
    name = name.strip()
    return name[:120] or "document"


# ── Main streaming pipeline ────────────────────────────────────────────────────
async def _run_comparison_pipeline(
    path_a: str,
    path_b: str,
    name_a: str,
    name_b: str,
    preview_url_a: str | None = None,
    preview_url_b: str | None = None,
    temp_paths: list[str] | None = None,
) -> AsyncIterator[str]:
    """
    Generator async: yield SSE frames trong suốt pipeline.
    """
    loop = asyncio.get_event_loop()

    try:
        # Gửi URLs của tệp PDF để UI chuẩn bị iframe
        if preview_url_a and preview_url_b:
            yield _sse("previews", {
                "url_a": preview_url_a,
                "url_b": preview_url_b,
                "name_a": name_a,
                "name_b": name_b
            })

        # ── Bước 1: Ingestion ──────────────────────────────────────────────────
        yield _sse("progress", {
            "step": 1, "total": 5,
            "status": "running",
            "title": "Đọc & phân tích cấu trúc tài liệu",
            "detail": f"Đang tải '{name_a}' và '{name_b}'...",
        })
        await asyncio.sleep(0)  # yield control to event loop

        chunks_a, chunks_b = await loop.run_in_executor(
            None, process_two_documents, path_a, path_b
        )

        yield _sse("progress", {
            "step": 1, "total": 5,
            "status": "done",
            "title": "Đọc & phân tích cấu trúc tài liệu",
            "detail": f"Đã trích xuất {len(chunks_a)} chunks từ '{name_a}', {len(chunks_b)} chunks từ '{name_b}'.",
        })

        # ── Bước 2: Embedding ──────────────────────────────────────────────────
        yield _sse("progress", {
            "step": 2, "total": 5,
            "status": "running",
            "title": "Nhúng vector (Embedding)",
            "detail": f"Đang embedding {len(chunks_a) + len(chunks_b)} chunks qua Ollama...",
        })
        await asyncio.sleep(0)

        start = time.time()
        collection, embeds_a, embeds_b = await loop.run_in_executor(
            None, embed_and_store, chunks_a, chunks_b
        )
        elapsed = time.time() - start

        yield _sse("progress", {
            "step": 2, "total": 5,
            "status": "done",
            "title": "Nhúng vector (Embedding)",
            "detail": f"Hoàn thành embedding trong {elapsed:.1f}s. Đã lưu vào ChromaDB.",
        })

        # ── Bước 3: Cross-Matching ─────────────────────────────────────────────
        yield _sse("progress", {
            "step": 3, "total": 5,
            "status": "running",
            "title": "So sánh ngữ nghĩa (Semantic Matching)",
            "detail": "Đang chạy thuật toán Anchor + Needleman-Wunsch...",
        })
        await asyncio.sleep(0)

        pairs = await loop.run_in_executor(
            None,
            lambda: build_comparison_pairs(
                chunks_a, chunks_b,
                embeds_a=embeds_a,
                embeds_b=embeds_b,
            ),
        )

        modified_count = sum(1 for p in pairs if p.match_type == "MODIFIED")
        added_count    = sum(1 for p in pairs if p.match_type == "ADDED")
        deleted_count  = sum(1 for p in pairs if p.match_type == "DELETED")
        unchanged_count = sum(1 for p in pairs if p.match_type == "UNCHANGED")

        yield _sse("progress", {
            "step": 3, "total": 5,
            "status": "done",
            "title": "So sánh ngữ nghĩa (Semantic Matching)",
            "detail": (
                f"Phát hiện {modified_count} sửa đổi · "
                f"{added_count} thêm mới · "
                f"{deleted_count} bị xóa · "
                f"{unchanged_count} không đổi."
            ),
        })

        # ── Bước 4: LLM Analysis ───────────────────────────────────────────────
        if modified_count == 0:
            yield _sse("progress", {
                "step": 4, "total": 5,
                "status": "skipped",
                "title": "Phân tích LLM",
                "detail": "Không có điều khoản sửa đổi → Bỏ qua bước phân tích LLM.",
            })
            report_md = await loop.run_in_executor(
                None,
                lambda: generate_comparison_report(pairs, file_a=name_a, file_b=name_b),
            )
        else:
            yield _sse("progress", {
                "step": 4, "total": 5,
                "status": "running",
                "title": "Phân tích điều khoản bằng AI (LLM)",
                "detail": f"Đang phân tích {modified_count} điều khoản sửa đổi qua Ollama Qwen3...",
            })
            await asyncio.sleep(0)

            report_md = await loop.run_in_executor(
                None,
                lambda: generate_comparison_report(pairs, file_a=name_a, file_b=name_b),
            )

            yield _sse("progress", {
                "step": 4, "total": 5,
                "status": "done",
                "title": "Phân tích điều khoản bằng AI (LLM)",
                "detail": f"Đã hoàn thành phân tích {modified_count} điều khoản sửa đổi.",
            })

        # ── Bước cuối: Lưu file và Gửi phản hồi ───────────────────────────────
        # Lấy thông tin folder dự án để lưu file chắc chắn
        project_dir = Path(__file__).parent.resolve()
        report_path = project_dir / "comparison_report.md"

        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_md)
            logger.info(f"[API] Đã lưu báo cáo mới tại: {report_path}")
        except Exception as e:
            logger.error(f"[API] Lỗi khi lưu file Markdown: {e}")

        # ── Bước 5: Done ───────────────────────────────────────────────────────
        yield _sse("progress", {
            "step": 5, "total": 5,
            "status": "done",
            "title": "Hoàn tất",
            "detail": f"Báo cáo đã sẵn sàng và được lưu tại {report_path.name}",
        })

        # Gửi summary stats
        yield _sse("stats", {
            "modified":  modified_count,
            "added":     added_count,
            "deleted":   deleted_count,
            "unchanged": unchanged_count,
        })

        # Gửi citations (nội dung gốc để highlight)
        citation_list = []
        for p in pairs:
            if p.match_type in ("MODIFIED", "ADDED", "DELETED"):
                citation_list.append({
                    "id": p.label,
                    "type": p.match_type,
                    "similarity": round(p.similarity, 4),
                    "text_a": p.chunk_a.content if p.chunk_a else None,
                    "text_b": p.chunk_b.content if p.chunk_b else None,
                    "page_a": p.chunk_a.page if p.chunk_a else None,
                    "page_b": p.chunk_b.page if p.chunk_b else None,
                })
        yield _sse("citations", {"items": citation_list})

        # Gửi report Markdown
        yield _sse("report", {"markdown": report_md})

        yield _sse("done", {"message": "Hoàn thành"})

    except Exception as e:
        yield _sse("error", {
            "message": f"Lỗi trong quá trình xử lý: {str(e)}",
            "detail":  str(type(e).__name__),
        })
    finally:
        # Cleanup temp files sau khi stream kết thúc (hoặc lỗi)
        if temp_paths:
            for p in temp_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve chat UI."""
    ui_file = UI_DIR / "index.html"
    if not ui_file.exists():
        raise HTTPException(status_code=404, detail="UI not found. Cần tạo ui/index.html")
    return ui_file.read_text(encoding="utf-8")


@app.post("/api/compare")
async def compare_documents(
    file_a: UploadFile = File(..., description="Văn bản pháp lý phiên bản gốc (v1)"),
    file_b: UploadFile = File(..., description="Văn bản pháp lý phiên bản mới (v2)"),
):
    """
    So sánh 2 văn bản pháp lý.
    Response là Server-Sent Events (SSE) stream với các event:
    - progress: cập nhật tiến trình từng bước
    - stats: thống kê số lượng thay đổi
    - report: nội dung báo cáo Markdown
    - done: tín hiệu kết thúc
    - error: lỗi (nếu có)
    """
    # Validate file type
    allowed = {".pdf", ".docx"}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
    for f in [file_a, file_b]:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Chỉ hỗ trợ file .pdf và .docx. Nhận được: '{f.filename}'",
            )

    # Validate file size (đọc vào bộ nhớ để kiểm tra, rồi reuse)
    content_a = await file_a.read()
    content_b = await file_b.read()
    if len(content_a) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File '{file_a.filename}' vượt quá giới hạn 50MB ({len(content_a)//1024//1024}MB).",
        )
    if len(content_b) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File '{file_b.filename}' vượt quá giới hạn 50MB ({len(content_b)//1024//1024}MB).",
        )

    # Lưu tạm vào /tmp
    suffix_a = Path(file_a.filename or "a").suffix.lower()
    suffix_b = Path(file_b.filename or "b").suffix.lower()

    tmp_a = tempfile.NamedTemporaryFile(delete=False, suffix=suffix_a)
    tmp_b = tempfile.NamedTemporaryFile(delete=False, suffix=suffix_b)

    try:
        tmp_a.write(content_a)
        tmp_a.flush()
        tmp_b.write(content_b)
        tmp_b.flush()
        tmp_a.close()
        tmp_b.close()

        # ── Preview: chỉ hỗ trợ file PDF (trình duyệt không mở được DOCX) ──
        preview_dir = UI_DIR / "previews"
        preview_dir.mkdir(exist_ok=True)
        preview_files: list[str] = []  # track để cleanup sau

        def _prepare_preview(tmp_path: str, suffix: str) -> str | None:
            """Copy file PDF vào thư mục preview, trả về URL. Bỏ qua nếu là DOCX."""
            if suffix == ".pdf":
                import shutil
                dest = preview_dir / Path(tmp_path).name
                shutil.copy2(tmp_path, dest)
                preview_files.append(str(dest))
                return f"/previews/{dest.name}"
            return None

        url_a = _prepare_preview(tmp_a.name, suffix_a)
        url_b = _prepare_preview(tmp_b.name, suffix_b)

        # Truyền file GỐC vào pipeline (load_document sẽ tự chọn loader phù hợp)
        return StreamingResponse(
            _run_comparison_pipeline(
                path_a=tmp_a.name,
                path_b=tmp_b.name,
                name_a=_sanitize_filename(file_a.filename or "Doc A"),
                name_b=_sanitize_filename(file_b.filename or "Doc B"),
                preview_url_a=url_a,
                preview_url_b=url_b,
                temp_paths=[tmp_a.name, tmp_b.name] + preview_files,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # Tắt buffering ở nginx nếu deploy
            },
        )
    except Exception:
        # Lỗi xảy ra trước khi stream bắt đầu → cleanup ngay
        for p in [tmp_a.name, tmp_b.name]:
            try:
                os.unlink(p)
            except OSError:
                pass
        raise


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}

# Mount static files last so they don't override /api routes
app.mount("/", StaticFiles(directory=UI_DIR), name="ui")

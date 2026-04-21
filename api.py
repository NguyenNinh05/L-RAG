from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chat_service import answer_session_question
from comparison import build_comparison_result
from config import ALLOWED_ORIGINS, LLM_ENABLE_REPORT, SESSION_DB_PATH
from embedding import embed_and_store
from ingestion import process_two_documents
from llm import generate_comparison_report
from retrieval import build_comparison_pairs
from session_store import SessionStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Legal RAG Comparison API",
    description="API so sánh văn bản pháp lý dùng RAG + LLM",
    version="1.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)

session_store = SessionStore(SESSION_DB_PATH)

UI_DIR = Path(__file__).parent / "ui"
UI_DIR.mkdir(exist_ok=True)


class ChatRequest(BaseModel):
    question: str


def _sse(event: str, data: dict | str) -> str:
    payload = json.dumps(data if isinstance(data, dict) else {"message": data}, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _iter_answer_chunks(text: str, target_chars: int = 32) -> list[str]:
    if not text:
        return []
    parts = re.findall(r"\n+|[^\s]+\s*", text)
    chunks: list[str] = []
    buffer = ""
    for part in parts:
        if buffer and len(buffer) + len(part) > target_chars and not part.startswith("\n"):
            chunks.append(buffer)
            buffer = part
            continue
        buffer += part
        if len(buffer) >= target_chars or part.startswith("\n\n"):
            chunks.append(buffer)
            buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks


def _sanitize_filename(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^\w\-\. ]", "_", name)
    name = name.strip()
    return name[:120] or "document"


def _hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


async def _check_disconnect(request: Request) -> None:
    if await request.is_disconnected():
        raise asyncio.CancelledError("Client disconnected")


def _write_report_copy(report_markdown: str) -> None:
    report_path = Path(__file__).parent.resolve() / "comparison_report.md"
    try:
        report_path.write_text(report_markdown, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - non-critical local artifact
        logger.warning("Could not write comparison_report.md: %s", exc)


async def _run_comparison_pipeline(
    request: Request,
    session_id: str,
    path_a: str,
    path_b: str,
    name_a: str,
    name_b: str,
    preview_url_a: str | None = None,
    preview_url_b: str | None = None,
    temp_paths: list[str] | None = None,
) -> AsyncIterator[str]:
    loop = asyncio.get_event_loop()
    chunks_a = []
    chunks_b = []
    try:
        yield _sse("session", {"session_id": session_id, "status": "processing"})
        if preview_url_a and preview_url_b:
            yield _sse(
                "previews",
                {
                    "url_a": preview_url_a,
                    "url_b": preview_url_b,
                    "name_a": name_a,
                    "name_b": name_b,
                },
            )

        session_store.update_session_step(session_id, "ingestion")
        yield _sse(
            "progress",
            {
                "step": 1,
                "total": 5,
                "status": "running",
                "title": "Đọc & phân tích cấu trúc tài liệu",
                "detail": f"Đang tải '{name_a}' và '{name_b}'...",
            },
        )
        chunks_a, chunks_b = await loop.run_in_executor(None, process_two_documents, path_a, path_b)
        await _check_disconnect(request)
        yield _sse(
            "progress",
            {
                "step": 1,
                "total": 5,
                "status": "done",
                "title": "Đọc & phân tích cấu trúc tài liệu",
                "detail": f"Đã trích xuất {len(chunks_a)} chunks từ '{name_a}', {len(chunks_b)} chunks từ '{name_b}'.",
            },
        )

        session_store.update_session_step(session_id, "chunking")
        yield _sse(
            "progress",
            {
                "step": 2,
                "total": 5,
                "status": "running",
                "title": "Nhúng vector (Embedding)",
                "detail": f"Đang embedding {len(chunks_a) + len(chunks_b)} chunks qua Ollama...",
            },
        )
        _, embeds_a, embeds_b = await loop.run_in_executor(None, embed_and_store, chunks_a, chunks_b)
        await _check_disconnect(request)
        yield _sse(
            "progress",
            {
                "step": 2,
                "total": 5,
                "status": "done",
                "title": "Nhúng vector (Embedding)",
                "detail": "Hoàn thành embedding và lưu vào vector store.",
            },
        )

        session_store.update_session_step(session_id, "retrieval")
        yield _sse(
            "progress",
            {
                "step": 3,
                "total": 5,
                "status": "running",
                "title": "So sánh ngữ nghĩa (Semantic Matching)",
                "detail": "Đang chạy cross-matching theo clause...",
            },
        )
        pairs = await loop.run_in_executor(
            None,
            lambda: build_comparison_pairs(chunks_a, chunks_b, embeds_a=embeds_a, embeds_b=embeds_b),
        )
        comparison_result = await loop.run_in_executor(
            None,
            lambda: build_comparison_result(pairs, file_a=name_a, file_b=name_b),
        )
        await _check_disconnect(request)
        stats = comparison_result.stats
        yield _sse(
            "progress",
            {
                "step": 3,
                "total": 5,
                "status": "done",
                "title": "So sánh ngữ nghĩa (Semantic Matching)",
                "detail": (
                    f"Phát hiện {stats.modified} sửa đổi · {stats.replaced} thay thế · "
                    f"{stats.added} thêm mới · {stats.deleted} bị xóa · {stats.unchanged} không đổi."
                ),
            },
        )

        session_store.update_session_step(session_id, "comparison")
        yield _sse(
            "progress",
            {
                "step": 4,
                "total": 5,
                "status": "running",
                "title": "LLM per-clause + deterministic post-check",
                "detail": (
                    "Đang sinh báo cáo grounded trên evidence deterministic..."
                    if stats.clauses_affected
                    else "Không có điều khoản thay đổi → báo cáo sẽ được dựng theo deterministic layer."
                ),
            },
        )
        report_md = await loop.run_in_executor(
            None,
            lambda: generate_comparison_report(
                comparison_result,
                file_a=name_a,
                file_b=name_b,
                enable_llm=LLM_ENABLE_REPORT,
            ),
        )
        await _check_disconnect(request)
        yield _sse(
            "progress",
            {
                "step": 4,
                "total": 5,
                "status": "done",
                "title": "LLM per-clause + deterministic post-check",
                "detail": f"Đã dựng báo cáo cho {stats.clauses_affected} điều khoản/phụ lục bị ảnh hưởng.",
            },
        )

        session_store.update_session_step(session_id, "persisting")
        persisted = await loop.run_in_executor(
            None,
            lambda: session_store.complete_session(
                session_id=session_id,
                comparison_result=comparison_result,
                report_markdown=report_md,
                chunks_a=chunks_a,
                chunks_b=chunks_b,
            ),
        )
        _write_report_copy(report_md)
        await _check_disconnect(request)

        yield _sse(
            "progress",
            {
                "step": 5,
                "total": 5,
                "status": "done",
                "title": "Hoàn tất",
                "detail": "Báo cáo, citations và session history đã được lưu local.",
            },
        )
        yield _sse("stats", stats.to_dict())
        yield _sse("analysis", comparison_result.to_dict())
        yield _sse("citations", {"items": persisted["clause_citations"]})
        yield _sse("report", {"markdown": report_md})
        yield _sse("done", {"message": "Hoàn thành", "session_id": session_id})

    except asyncio.CancelledError:
        session_store.fail_session(session_id, "stream_interrupted", "Client disconnected before comparison completed.")
        raise
    except Exception as exc:
        session_store.fail_session(session_id, type(exc).__name__, str(exc))
        yield _sse(
            "error",
            {
                "message": f"Lỗi trong quá trình xử lý: {exc}",
                "detail": type(exc).__name__,
                "session_id": session_id,
            },
        )
    finally:
        if temp_paths:
            for path in temp_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass


@app.on_event("startup")
async def startup_event() -> None:
    session_store.initialize()
    recovered = session_store.recover_interrupted_sessions()
    if recovered:
        logger.info("Recovered %s interrupted processing sessions.", recovered)


@app.get("/", response_class=HTMLResponse)
async def serve_ui() -> str:
    ui_file = UI_DIR / "index.html"
    if not ui_file.exists():
        raise HTTPException(status_code=404, detail="UI not found. Cần tạo ui/index.html")
    return ui_file.read_text(encoding="utf-8")


@app.post("/api/compare")
async def compare_documents(
    request: Request,
    file_a: UploadFile = File(..., description="Văn bản pháp lý phiên bản gốc (v1)"),
    file_b: UploadFile = File(..., description="Văn bản pháp lý phiên bản mới (v2)"),
):
    allowed = {".pdf", ".docx"}
    max_file_size = 50 * 1024 * 1024

    for file in (file_a, file_b):
        ext = Path(file.filename or "").suffix.lower()
        if ext not in allowed:
            raise HTTPException(status_code=400, detail=f"Chỉ hỗ trợ file .pdf và .docx. Nhận được: '{file.filename}'")

    content_a = await file_a.read()
    content_b = await file_b.read()
    if len(content_a) > max_file_size:
        raise HTTPException(status_code=413, detail=f"File '{file_a.filename}' vượt quá giới hạn 50MB.")
    if len(content_b) > max_file_size:
        raise HTTPException(status_code=413, detail=f"File '{file_b.filename}' vượt quá giới hạn 50MB.")

    safe_name_a = _sanitize_filename(file_a.filename or "Doc A")
    safe_name_b = _sanitize_filename(file_b.filename or "Doc B")
    session_id = session_store.create_session(
        file_a_name=safe_name_a,
        file_b_name=safe_name_b,
        file_a_hash=_hash_bytes(content_a),
        file_b_hash=_hash_bytes(content_b),
    )

    suffix_a = Path(file_a.filename or "a").suffix.lower()
    suffix_b = Path(file_b.filename or "b").suffix.lower()
    tmp_a = tempfile.NamedTemporaryFile(delete=False, suffix=suffix_a)
    tmp_b = tempfile.NamedTemporaryFile(delete=False, suffix=suffix_b)
    try:
        tmp_a.write(content_a)
        tmp_b.write(content_b)
        tmp_a.flush()
        tmp_b.flush()
        tmp_a.close()
        tmp_b.close()

        preview_dir = UI_DIR / "previews"
        preview_dir.mkdir(exist_ok=True)
        preview_files: list[str] = []

        def _prepare_preview(tmp_path: str, suffix: str) -> str | None:
            if suffix != ".pdf":
                return None
            import shutil

            dest = preview_dir / Path(tmp_path).name
            shutil.copy2(tmp_path, dest)
            preview_files.append(str(dest))
            return f"/previews/{dest.name}"

        return StreamingResponse(
            _run_comparison_pipeline(
                request=request,
                session_id=session_id,
                path_a=tmp_a.name,
                path_b=tmp_b.name,
                name_a=safe_name_a,
                name_b=safe_name_b,
                preview_url_a=_prepare_preview(tmp_a.name, suffix_a),
                preview_url_b=_prepare_preview(tmp_b.name, suffix_b),
                temp_paths=[tmp_a.name, tmp_b.name, *preview_files],
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except Exception as exc:
        session_store.fail_session(session_id, type(exc).__name__, str(exc))
        for path in (tmp_a.name, tmp_b.name):
            try:
                os.unlink(path)
            except OSError:
                pass
        raise


@app.get("/api/sessions")
async def list_sessions() -> dict[str, list[dict[str, object]]]:
    return {"items": session_store.list_completed_sessions()}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


@app.get("/api/sessions/{session_id}/citations/{citation_id}")
async def get_citation(session_id: str, citation_id: str):
    citation = session_store.get_citation(session_id, citation_id)
    if not citation:
        raise HTTPException(status_code=404, detail="Citation not found.")
    return citation


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    session_store.delete_session(session_id)
    return {"deleted": True, "session_id": session_id}


@app.delete("/api/sessions")
async def delete_all_sessions():
    session_store.delete_all_sessions()
    return {"deleted": True}


@app.post("/api/sessions/{session_id}/chat")
async def chat_session(session_id: str, payload: ChatRequest):
    status = session_store.get_session_status(session_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    if status != "completed":
        return JSONResponse(
            status_code=409,
            content={"detail": "Session chưa hoàn tất so sánh nên chưa thể hỏi đáp."},
        )

    async def _chat_stream() -> AsyncIterator[str]:
        try:
            answer = await answer_session_question(session_store, session_id, payload.question)
            session_store.save_chat_exchange(
                session_id=session_id,
                question=payload.question,
                answer_markdown=answer["answer_markdown"],
                citation_ids=answer.get("used_citation_ids", []),
                answer_type=answer.get("answer_type", "diff_answer"),
            )
            for chunk in _iter_answer_chunks(answer.get("answer_markdown", "")):
                yield _sse("delta", {"content": chunk})
                await asyncio.sleep(0.012)
            yield _sse("message", answer)
            yield _sse("done", {"session_id": session_id})
        except Exception as exc:
            yield _sse("error", {"message": f"Không thể trả lời câu hỏi: {exc}"})

    return StreamingResponse(
        _chat_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "version": "1.1.0"}


app.mount("/", StaticFiles(directory=UI_DIR), name="ui")

#!/usr/bin/env python3
"""Drive the LegalDiff backend end-to-end and save stage outputs."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import redis

PROJECT_ROOT = Path("/home/haipd/L-RAG")
DOC_V1 = PROJECT_ROOT / "data_test" / "01-tand_signed_v1.docx"
DOC_V2 = PROJECT_ROOT / "data_test" / "01-tand_signed_v2.docx"

API_BASE_URL = os.getenv("LEGALDIFF_API_URL", "http://localhost:8001").rstrip("/")
LLM_BASE_URL = os.getenv("LEGALDIFF_LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
LLM_MODEL_NAME = os.getenv("LEGALDIFF_LLM_MODEL_NAME", "llama3:70b")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
TIMEOUT_SECONDS = int(os.getenv("LEGALDIFF_E2E_TIMEOUT_SECONDS", "3600"))
POLL_SECONDS = float(os.getenv("LEGALDIFF_E2E_POLL_SECONDS", "3"))
MAX_CONCURRENCY = int(os.getenv("LEGALDIFF_E2E_MAX_CONCURRENCY", "1"))
MAX_COMPARISON_PAIRS = int(os.getenv("LEGALDIFF_E2E_MAX_COMPARISON_PAIRS", "1"))
MAX_TOKENS_ACU = int(os.getenv("LEGALDIFF_E2E_MAX_TOKENS_ACU", "512"))
MAX_TOKENS_SUMMARY = int(os.getenv("LEGALDIFF_E2E_MAX_TOKENS_SUMMARY", "256"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LEGALDIFF_E2E_LLM_TIMEOUT_SECONDS", "300"))
LLM_MAX_RETRIES = int(os.getenv("LEGALDIFF_E2E_LLM_MAX_RETRIES", "0"))

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(
    os.getenv(
        "LEGALDIFF_E2E_OUTPUT_DIR",
        str(PROJECT_ROOT / "tests" / "e2e_backend" / "outputs" / RUN_ID),
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(message, flush=True)


def save_json(name: str, data: Any) -> Path:
    path = OUTPUT_DIR / name
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def save_text(name: str, data: str) -> Path:
    path = OUTPUT_DIR / name
    path.write_text(data, encoding="utf-8")
    return path


def append_jsonl(name: str, data: Any) -> None:
    path = OUTPUT_DIR / name
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")


def request_json(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    token: str | None = None,
    expected: set[int] | None = None,
    **kwargs: Any,
) -> tuple[int, Any]:
    headers = kwargs.pop("headers", {})
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = client.request(method, f"{API_BASE_URL}{path}", headers=headers, **kwargs)
    expected = expected or {200}
    try:
        payload: Any = response.json()
    except Exception:
        payload = response.text
    if response.status_code not in expected:
        raise RuntimeError(
            f"{method} {path} returned {response.status_code}, expected {sorted(expected)}: {payload}"
        )
    return response.status_code, payload


def mask_login_response(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "token_type": data.get("token_type"),
        "expires_in": data.get("expires_in"),
        "has_access_token": bool(data.get("access_token")),
        "has_refresh_token": bool(data.get("refresh_token")),
        "access_token_prefix": (data.get("access_token") or "")[:16],
    }


def start_progress_listener(job_id: str, stop_event: threading.Event) -> threading.Thread:
    channel = f"job:{job_id}:progress"

    def run() -> None:
        try:
            r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            pubsub = r.pubsub(ignore_subscribe_messages=True)
            pubsub.subscribe(channel)
            log(f"[progress] subscribed redis channel={channel}")
            while not stop_event.is_set():
                message = pubsub.get_message(timeout=1.0)
                if not message:
                    continue
                raw = message.get("data")
                try:
                    event = json.loads(raw)
                except Exception:
                    event = {"raw": raw}
                event["captured_at"] = datetime.now().isoformat(timespec="seconds")
                append_jsonl("10_progress_events.jsonl", event)
                log(
                    "[progress] "
                    f"{event.get('progress_pct', '?')}% "
                    f"{event.get('current_phase', '?')} - "
                    f"{event.get('message', event)}"
                )
        except Exception as exc:
            append_jsonl("10_progress_events.jsonl", {"listener_error": str(exc)})
            log(f"[progress] listener error: {exc}")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def main() -> int:
    started_at = datetime.now().isoformat(timespec="seconds")
    save_json(
        "00_config.json",
        {
            "started_at": started_at,
            "api_base_url": API_BASE_URL,
            "llm_base_url": LLM_BASE_URL,
            "llm_model_name": LLM_MODEL_NAME,
            "redis_url": REDIS_URL,
            "doc_v1": str(DOC_V1),
            "doc_v2": str(DOC_V2),
            "output_dir": str(OUTPUT_DIR),
            "timeout_seconds": TIMEOUT_SECONDS,
            "poll_seconds": POLL_SECONDS,
            "max_concurrency": MAX_CONCURRENCY,
            "max_comparison_pairs": MAX_COMPARISON_PAIRS,
            "max_tokens_acu": MAX_TOKENS_ACU,
            "max_tokens_summary": MAX_TOKENS_SUMMARY,
            "llm_timeout_seconds": LLM_TIMEOUT_SECONDS,
            "llm_max_retries": LLM_MAX_RETRIES,
        },
    )

    log(f"[output] {OUTPUT_DIR}")
    if not DOC_V1.exists() or not DOC_V2.exists():
        raise FileNotFoundError(f"Missing test docs: {DOC_V1}, {DOC_V2}")

    with httpx.Client(timeout=httpx.Timeout(120.0)) as client:
        log("[stage 01] health")
        _, health = request_json(client, "GET", "/api/health")
        save_json("01_health.json", health)
        log(json.dumps(health, ensure_ascii=False))

        log("[stage 02] readiness")
        _, ready = request_json(client, "GET", "/api/health/ready")
        save_json("02_ready.json", ready)
        log(json.dumps(ready, ensure_ascii=False))

        log("[stage 03] llm models")
        llm_response = client.get(f"{LLM_BASE_URL}/models")
        llm_payload = llm_response.json()
        save_json("03_llm_models.json", llm_payload)
        log(json.dumps(llm_payload, ensure_ascii=False))

        username = f"e2e_{RUN_ID}_{uuid.uuid4().hex[:8]}"
        email = f"{username}@example.local"
        password = "password123"

        log("[stage 04] register")
        _, register = request_json(
            client,
            "POST",
            "/api/auth/register",
            expected={201},
            json={
                "email": email,
                "username": username,
                "password": password,
                "full_name": "Backend E2E Test",
            },
        )
        save_json("04_register.json", register)
        log(json.dumps(register, ensure_ascii=False))

        log("[stage 05] login")
        login = None
        for attempt in range(1, 6):
            try:
                _, login = request_json(
                    client,
                    "POST",
                    "/api/auth/login",
                    json={"username": username, "password": password},
                )
                break
            except RuntimeError as exc:
                if attempt == 5 or "returned 401" not in str(exc):
                    raise
                log(f"[stage 05] login retry {attempt}/5 after transient 401")
                time.sleep(1)
        assert login is not None
        token = login["access_token"]
        masked_login = mask_login_response(login)
        save_json("05_login.json", masked_login)
        log(json.dumps(masked_login, ensure_ascii=False))

        log("[stage 06] me")
        _, me = request_json(client, "GET", "/api/auth/me", token=token)
        save_json("06_me.json", me)
        log(json.dumps(me, ensure_ascii=False))

        log("[stage 07] upload V1")
        with DOC_V1.open("rb") as f:
            _, upload_v1 = request_json(
                client,
                "POST",
                "/api/documents/upload",
                token=token,
                expected={201},
                files={"file": (DOC_V1.name, f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            )
        save_json("07_upload_v1.json", upload_v1)
        log(json.dumps(upload_v1, ensure_ascii=False))

        log("[stage 08] upload V2")
        with DOC_V2.open("rb") as f:
            _, upload_v2 = request_json(
                client,
                "POST",
                "/api/documents/upload",
                token=token,
                expected={201},
                files={"file": (DOC_V2.name, f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            )
        save_json("08_upload_v2.json", upload_v2)
        log(json.dumps(upload_v2, ensure_ascii=False))

        log("[stage 09] create comparison job")
        job_body = {
            "document_v1_id": upload_v1["id"],
            "document_v2_id": upload_v2["id"],
            "skip_phase3": False,
            "config_overrides": {
                "llm_base_url": LLM_BASE_URL,
                "llm_model_name": LLM_MODEL_NAME,
                "max_concurrency": MAX_CONCURRENCY,
                "max_comparison_pairs": MAX_COMPARISON_PAIRS,
                "max_tokens_acu": MAX_TOKENS_ACU,
                "max_tokens_summary": MAX_TOKENS_SUMMARY,
                "llm_timeout_seconds": LLM_TIMEOUT_SECONDS,
                "llm_max_retries": LLM_MAX_RETRIES,
                "output_dir": str(OUTPUT_DIR / "pipeline_outputs"),
                "qdrant_path": str(OUTPUT_DIR / "qdrant_db"),
                "collection_name": f"e2e_{RUN_ID}",
            },
        }
        save_json("09_create_job_request.json", job_body)
        _, job = request_json(
            client,
            "POST",
            "/api/jobs",
            token=token,
            expected={201},
            json=job_body,
        )
        save_json("09_create_job_response.json", job)
        log(json.dumps(job, ensure_ascii=False))

        job_id = job["id"]
        stop_event = threading.Event()
        progress_thread = start_progress_listener(job_id, stop_event)

        log("[stage 10] poll job status")
        deadline = time.monotonic() + TIMEOUT_SECONDS
        last_key: tuple[Any, ...] | None = None
        final_status: dict[str, Any] | None = None
        try:
            while time.monotonic() < deadline:
                _, status = request_json(client, "GET", f"/api/jobs/{job_id}/status", token=token)
                status["captured_at"] = datetime.now().isoformat(timespec="seconds")
                append_jsonl("10_status_poll.jsonl", status)
                key = (
                    status.get("status"),
                    status.get("current_phase"),
                    status.get("progress_pct"),
                    status.get("error_message"),
                )
                if key != last_key:
                    log(
                        "[status] "
                        f"{status.get('status')} "
                        f"{status.get('current_phase')} "
                        f"{status.get('progress_pct')}% "
                        f"{status.get('error_message') or ''}"
                    )
                    last_key = key
                if status.get("status") in {"completed", "failed", "cancelled"}:
                    final_status = status
                    break
                time.sleep(POLL_SECONDS)
        finally:
            stop_event.set()
            progress_thread.join(timeout=2)

        if final_status is None:
            final_status = {"status": "timeout", "job_id": job_id, "timeout_seconds": TIMEOUT_SECONDS}
            save_json("11_final_job_status.json", final_status)
            log(json.dumps(final_status, ensure_ascii=False))
            return 3

        save_json("11_final_job_status.json", final_status)
        log(f"[stage 11] final job status: {json.dumps(final_status, ensure_ascii=False)}")

        log("[stage 12] job detail")
        _, job_detail = request_json(client, "GET", f"/api/jobs/{job_id}", token=token)
        save_json("12_job_detail.json", job_detail)

        log("[stage 13] catalog")
        _, catalog = request_json(client, "GET", f"/api/jobs/{job_id}/catalog", token=token)
        save_json("13_catalog.json", catalog)

        log("[stage 14] reports list")
        _, reports = request_json(client, "GET", f"/api/jobs/{job_id}/reports", token=token)
        save_json("14_reports_list.json", reports)

        report_items = reports.get("items", []) if isinstance(reports, dict) else []
        for idx, report in enumerate(report_items, start=1):
            report_id = report["id"]
            _, detail = request_json(client, "GET", f"/api/reports/{report_id}", token=token)
            save_json(f"15_report_{idx:02d}_{report_id}.json", detail)
            md_response = client.get(
                f"{API_BASE_URL}/api/reports/{report_id}/markdown",
                headers={"Authorization": f"Bearer {token}"},
            )
            md_response.raise_for_status()
            save_text(f"16_report_{idx:02d}_{report_id}.md", md_response.text)

        summary = {
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "output_dir": str(OUTPUT_DIR),
            "job_id": job_id,
            "final_status": final_status,
            "catalog_summary_keys": sorted(catalog.keys()) if isinstance(catalog, dict) else [],
            "report_count": len(report_items),
        }
        save_json("99_summary.json", summary)
        log(f"[summary] {json.dumps(summary, ensure_ascii=False)}")
        return 0 if final_status.get("status") == "completed" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        save_json(
            "99_error.json",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "output_dir": str(OUTPUT_DIR),
            },
        )
        log(f"[error] {type(exc).__name__}: {exc}")
        raise

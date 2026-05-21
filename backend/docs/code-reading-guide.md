# Backend LegalDiff — Thứ tự đọc hiểu code

Đây là hướng dẫn đọc code backend theo từng lớp, từ nền tảng đến nghiệp vụ. Mỗi lớp xây dựng trên lớp trước — đọc đúng thứ tự sẽ hiểu toàn bộ kiến trúc mà không bị "gà mờ".

---

## Tổng quan kiến trúc backend

```
[Browser / Frontend]
       │
       ▼
[FastAPI :8001] ─── REST API + WebSocket
       │
       ├── [PostgreSQL] — metadata (users, documents, jobs, reports)
       ├── [Redis]      — Celery broker + pub/sub progress
       └── [Celery Worker :GPU]
              │
              ├── GPU Lock (Redis mutex)
              └── src/pipeline.py (3-phase AI)
                       │
                       └── [llama.cpp :8000] (LLM Qwen2.5-7B)
```

**5 services trong docker-compose.yml:** `postgres`, `redis`, `llm-server`, `web` (FastAPI), `worker` (Celery).

---

## Lớp 1 — Nền tảng: Config & Database

> Đây là 2 file đầu tiên phải đọc. Không hiểu chúng thì mọi thứ phía trên vô nghĩa.

| # | File | Mục đích đọc |
|---|------|-------------|
| 1 | `config.py` | **Singleton config toàn backend.** Pattern `@lru_cache` giống hệt `src/config.py`. Đọc từ biến môi trường (`.env`), có fallback defaults. Export `BackendConfig` dataclass (frozen) chứa: `database_url`, `redis_url`, `jwt_secret_key`, `llm_base_url`, `storage_root`,... |
| 2 | `database.py` | **SQLAlchemy async engine + session factory.** 2 tầng: `create_async_engine()` cho FastAPI (asyncpg, pool_size=10), và `create_engine()` sync cho Celery worker (NullPool). Export `get_db()` — FastAPI dependency yield AsyncSession, và `get_sync_session()` cho Celery task. |

**Hiểu gì sau lớp 1:** Backend config từ đâu ra, database connect như thế nào, tại sao có 2 engine (async cho web, sync cho worker).

---

## Lớp 2 — Data Models (ORM)

> 6 file định nghĩa toàn bộ schema PostgreSQL. Mọi API endpoint đều đọc/ghi qua các model này.

| # | File | Mục đích đọc |
|---|------|-------------|
| 3 | `models/base.py` | **DeclarativeBase + mixins.** Định nghĩa `Base` (SQLAlchemy ORM base), `TimestampMixin` (created_at + updated_at tự động), helper `uuid_pk()`. Tất cả model khác kế thừa từ đây. |
| 4 | `models/user.py` | **Bảng `users`.** id (UUID PK), email (unique), username (unique), password_hash (bcrypt), full_name, is_active, is_superuser. Có relationships → documents, jobs. |
| 5 | `models/document.py` | **Bảng `documents`.** Mỗi row = 1 file PDF/DOCX đã upload. Trường quan trọng: `storage_path` (đường dẫn file), `content_hash_sha256` (dedup), `doc_title`/`doc_number`/`parties` (điền sau khi Phase 1 parse xong), `is_processed`. Relationships → user (owner), jobs_as_v1, jobs_as_v2. |
| 6 | `models/comparison_job.py` | **Bảng `comparison_jobs` — trung tâm của backend.** Mỗi row = 1 lần so sánh 2 documents. Trường chính: `status` (pending→processing→completed/failed/cancelled), `current_phase` (queued→ingestion→alignment→comparison→done), `progress_pct` (0-100), `catalog` (JSONB — toàn bộ DiffPairCatalog), `celery_task_id`, các trường denormalized (total_pairs, matched_count,...), `config_snapshot` (audit trail). Enum: `JobStatus`, `JobPhase`. |
| 7 | `models/comparison_report.py` | **Bảng `comparison_reports`.** Mỗi row = 1 ComparisonReport cho 1 cặp article đã matched. Trường chính: `pair_id`, `match_type`, `location_context`, stats (total_acus, passed_acus, rejected_acus, hallucination_rate), `executive_summary` (JSONB), `markdown_report` (TEXT), `verified_acus` (JSONB), `rejected_acus_detail` (JSONB). |
| 8 | `models/__init__.py` | Re-export tất cả models + Base + mixins để import gọn: `from backend.models import User, Document, ComparisonJob,...` |

**Hiểu gì sau lớp 2:** Toàn bộ schema DB, mối quan hệ giữa các bảng (User → Documents → ComparisonJob → ComparisonReport), cách audit trail được lưu (catalog JSONB, config_snapshot).

---

## Lớp 3 — Schemas (Pydantic API Models)

> 8 file định nghĩa request/response schema cho từng nhóm API. Tách biệt hoàn toàn với ORM models.

| # | File | Mục đích đọc |
|---|------|-------------|
| 9 | `schemas/auth.py` | RegisterRequest, LoginRequest, TokenResponse, RefreshRequest, RefreshResponse. Validate email regex, password min length. |
| 10 | `schemas/user.py` | UserResponse — trả về cho client (không lộ password_hash). |
| 11 | `schemas/document.py` | DocumentUploadResponse (sau upload), DocumentResponse (chi tiết). |
| 12 | `schemas/job.py` | CreateJobRequest (có validator bắt v1≠v2), JobResponse (đầy đủ), JobStatusResponse (gọn nhẹ cho polling). |
| 13 | `schemas/report.py` | ReportSummaryResponse (danh sách), ReportDetailResponse (đầy đủ ACUs + markdown). |
| 14 | `schemas/common.py` | PaginatedResponse[T] (generic pagination), ErrorResponse, HealthResponse. |
| 15 | `schemas/ws.py` | WSProgressMessage — cấu trúc message gửi qua WebSocket. |
| 16 | `schemas/__init__.py` | Re-export tất cả schemas. |

**Hiểu gì sau lớp 3:** API nhận gì, trả gì, validate ra sao. Pagination dùng chung kiểu `PaginatedResponse[T]`.

---

## Lớp 4 — Services (Business Logic)

> 4 file chứa toàn bộ logic nghiệp vụ. API endpoints chỉ là "vỏ" — gọi thẳng vào services.

| # | File | Mục đích đọc |
|---|------|-------------|
| 17 | `services/auth_service.py` | **Xác thực.** `hash_password()` / `verify_password()` (bcrypt), `create_access_token()` / `create_refresh_token()` (JWT HS256), `verify_token()` (decode + validate). Token có trường `type`: "access" hoặc "refresh" để phân biệt. |
| 18 | `services/storage.py` | **Quản lý file trên disk.** `FileStorageManager` quản lý 3 thư mục: `uploads/` (file gốc), `processed/` (graph DB, vector DB), `reports/` (báo cáo). `store_upload()` — lưu file với tên unique `{uuid}_{filename}`, trả về (storage_path, sha256, file_size). `get_absolute_path()` — resolve relative→absolute. `delete_file()` — xóa file. |
| 19 | `services/document_service.py` | **Nghiệp vụ document.** `upload()` — validate extension (.pdf/.docx), check file size, SHA256 dedup (cùng hash + cùng user → trả về document cũ), lưu DB. `list_by_user()` — paginated query. `delete()` — kiểm tra không có active job tham chiếu rồi mới xóa. |
| 20 | `services/job_service.py` | **Nghiệp vụ job — quan trọng nhất.** `create()` — validate 2 documents tồn tại + thuộc về user, tạo ComparisonJob, resolve absolute file paths, `celery_app.send_task("run_pipeline", ...)` dispatch vào Celery worker, lưu celery_task_id. `cancel()` — revoke Celery task nếu đang chạy. `delete()` — chỉ xóa nếu không đang processing. |
| 21 | `services/__init__.py` | Re-export tất cả services + functions. |

**Hiểu gì sau lớp 4:** Toàn bộ flow nghiệp vụ: upload file → dedup → tạo job → dispatch Celery → cancel/delete. Đây là lớp quan trọng nhất để hiểu backend hoạt động thế nào.

---

## Lớp 5 — API Endpoints (Route Handlers)

> 8 file FastAPI route handlers. Mỗi file = 1 nhóm endpoint. Code rất mỏng — chỉ validate + gọi service + return.

| # | File | Mục đích đọc |
|---|------|-------------|
| 22 | `api/deps.py` | **Dependency injection — đọc đầu tiên trong nhóm này.** 3 dependencies: `get_current_user()` — giải mã JWT Bearer token → query DB → trả về User object (401 nếu sai/thiếu token), `get_db()` — yield AsyncSession (re-export từ database.py), `get_storage()` — trả về FileStorageManager singleton. |
| 23 | `api/auth.py` | `/api/auth/*`. 4 endpoints: `POST /register` (tạo user, check duplicate email/username), `POST /login` (verify password → trả về JWT pair), `POST /refresh` (đổi refresh token lấy access mới), `GET /me` (trả về user hiện tại). |
| 24 | `api/documents.py` | `/api/documents/*`. 4 endpoints: `POST /upload` (multipart file → DocumentService.upload), `GET /` (list paginated), `GET /{id}` (detail), `DELETE /{id}` (xóa nếu không có active job). |
| 25 | `api/jobs.py` | `/api/jobs/*` — nhiều endpoint nhất. `POST /` (tạo job từ 2 document ids), `GET /` (list paginated), `GET /{id}` (detail), `GET /{id}/status` (lightweight poll), `POST /{id}/cancel`, `DELETE /{id}`, `GET /{id}/reports` (list reports của job), `GET /{id}/catalog` (DiffPairCatalog JSON). |
| 26 | `api/reports.py` | `/api/reports/*`. 2 endpoints: `GET /{id}` (full detail + ACUs + markdown), `GET /{id}/markdown` (raw markdown text/plain). Có authorization check join qua ComparisonJob → User. |
| 27 | `api/health.py` | `/api/health` (return "ok"), `/api/health/ready` — kiểm tra toàn bộ dependencies: PostgreSQL (SELECT 1), Redis (PING), Celery worker (GPU lock check), LLM server (GET /v1/models). Trả về trạng thái từng thành phần + status tổng. |
| 28 | `api/router.py` | Gom tất cả router con vào 1 `api_router` với prefix `/api`. Mỗi router thêm prefix riêng: `/auth`, `/documents`, `/jobs`, `/reports`, `/health`. |
| 29 | `api/__init__.py` | Re-export api_router + dependencies. |

**Hiểu gì sau lớp 5:** Toàn bộ REST API surface, cách auth hoạt động (JWT Bearer → Depends(get_current_user)), cách response chuẩn hóa (PaginatedResponse, ErrorResponse).

---

## Lớp 6 — Async Workers (Celery + GPU)

> 3 file — nơi pipeline AI thực sự chạy. Đây là trái tim của backend.

| # | File | Mục đích đọc |
|---|------|-------------|
| 30 | `celery_app.py` | **Celery app singleton.** Broker + backend = Redis. Config quan trọng: `worker_concurrency=1` (chỉ 1 pipeline chạy GPU 1 lúc), `task_acks_late=True` (re-deliver nếu worker crash), `task_track_started=True`. Autodiscover tasks từ `backend.workers`. |
| 31 | `workers/gpu_lock.py` | **Redis mutex — đảm bảo 1 GPU pipeline duy nhất.** `GPULock.acquire(job_id)` — non-blocking, dùng Redis `SET NX EX`. `release(job_id)` — atomic check-and-delete bằng Lua script (chỉ xóa nếu job_id khớp). `GPUBusyError` — exception để Celery retry. Timeout lock: 15 phút. |
| 32 | `workers/pipeline_task.py` | **Core Celery task — file phức tạp nhất backend.** Flow: (1) Acquire GPU lock → (2) Update job status processing → (3) Import `LegalDiffPipeline`, tạo `PipelineRunConfig` với `progress_callback` → (4) Gọi `pipeline.run()` → (5) `_store_results()` lưu catalog JSONB + từng ComparisonReport vào PostgreSQL → (6) Update job completed → (7) Release GPU lock. Progress callback publish qua Redis pub/sub kênh `job:{id}:progress`. Có `on_failure` handler: mark job failed, release lock. |
| 33 | `workers/__init__.py` | Re-export run_pipeline + GPULock. |

**Hiểu gì sau lớp 6:** Pipeline AI chạy trong Celery worker như thế nào, GPU lock hoạt động ra sao, progress được publish qua Redis pub/sub, kết quả được lưu vào PostgreSQL, error handling (retry, mark failed).

---

## Lớp 7 — WebSocket (Real-time Progress)

> 2 file — push progress về frontend không cần polling.

| # | File | Mục đích đọc |
|---|------|-------------|
| 34 | `ws/manager.py` | **ConnectionManager — Redis pub/sub bridge.** Duy trì `dict[job_id, set[WebSocket]]` trong memory. `connect(job_id, ws)` — accept connection, thêm vào set. `listen_redis(job_id)` — background task subscribe kênh `job:{id}:progress`, khi có message → broadcast đến tất cả WebSocket local đang theo dõi job đó. Xử lý dead connection (tự dọn). |
| 35 | `ws/handlers.py` | **WebSocket endpoint `/ws/jobs/{job_id}?token=xxx`.** Xác thực qua query param `token` (WebSocket không gửi được Authorization header). Authorization: kiểm tra user sở hữu job. Connect → start Redis listener background → loop nhận ping/pong → cleanup khi disconnect. |
| 36 | `ws/__init__.py` | Re-export ConnectionManager + ws_router. |

**Hiểu gì sau lớp 7:** Cách real-time progress hoạt động: Celery task → Redis pub/sub → ConnectionManager.listen_redis → broadcast WebSocket → frontend cập nhật progress bar.

---

## Lớp 8 — Entry Point & Deployment

> 4 file — khởi động app và deploy.

| # | File | Mục đích đọc |
|---|------|-------------|
| 37 | `main.py` | **FastAPI app factory.** `create_app()` — khởi tạo FastAPI với CORS middleware (allow all origins cho local dev), include api_router + ws_router. `lifespan` context manager: startup → tạo storage directories, shutdown → cleanup. |
| 38 | `api/health.py` | *(đã đọc ở lớp 5)* |
| 39 | `Dockerfile` | Multi-stage build: python:3.12-slim → cài dependencies → copy code → entrypoint chạy alembic migrate + uvicorn. |
| 40 | `docker-compose.yml` | 5 services: `postgres` (16-alpine, healthcheck pg_isready), `redis` (7-alpine), `llm-server` (llama.cpp CUDA, mount models readonly), `web` (FastAPI 2 workers, mount src/ + storage volume), `worker` (Celery concurrency=1, GPU reservation). |

---

## Lớp 9 — Database Migrations (Alembic)

| # | File | Mục đích đọc |
|---|------|-------------|
| 41 | `alembic.ini` | Config Alembic: script_location, sqlalchemy.url fallback. |
| 42 | `alembic/env.py` | Migration environment: import Base metadata từ `backend.models`, override url từ `DATABASE_URL_SYNC` env, config offline + online mode. |
| 43 | `alembic/versions/001_initial_migration.py` | Migration đầu tiên: tạo 4 bảng `users`, `documents`, `comparison_jobs`, `comparison_reports` với đầy đủ columns, foreign keys, indexes. Có `downgrade()` để rollback. |

---

## Tóm tắt thứ tự đọc (rút gọn)

```
1. config.py              ← Config từ env
2. database.py            ← SQLAlchemy engine
3. models/base.py         ← ORM base
4. models/user.py         ← Bảng users
5. models/document.py     ← Bảng documents
6. models/comparison_job.py ← Bảng jobs (trung tâm)
7. models/comparison_report.py ← Bảng reports
8. models/__init__.py
9-16. schemas/*.py        ← Pydantic request/response
17. services/auth_service.py ← JWT + bcrypt
18. services/storage.py   ← File system
19. services/document_service.py ← Upload + dedup
20. services/job_service.py ← Tạo job + dispatch Celery
22. api/deps.py           ← Dependency injection
23. api/auth.py           ← Auth endpoints
24. api/documents.py      ← Document endpoints
25. api/jobs.py           ← Job endpoints (nhiều nhất)
26. api/reports.py        ← Report endpoints
27. api/health.py         ← Health check
28. api/router.py         ← Router aggregation
30. celery_app.py         ← Celery config
31. workers/gpu_lock.py   ← GPU mutex
32. workers/pipeline_task.py ← Core worker (phức tạp nhất)
34. ws/manager.py         ← WebSocket manager
35. ws/handlers.py        ← WebSocket endpoint
37. main.py               ← FastAPI app
39. Dockerfile
40. docker-compose.yml
41-43. alembic/*          ← Migrations
```

---

## Luồng dữ liệu qua các file (1 request hoàn chỉnh)

```
POST /api/auth/login {username, password}
  → api/auth.py → services/auth_service.py (verify_password, create_access_token)
  → return JWT

POST /api/documents/upload (multipart file)
  → api/documents.py → services/document_service.py (validate ext, check size, SHA256 dedup)
  → services/storage.py (store_upload → disk)
  → models/document.py (INSERT)
  → return document_id

POST /api/jobs {document_v1_id, document_v2_id}
  → api/jobs.py → services/job_service.py (validate docs, create ComparisonJob)
  → celery_app.send_task("run_pipeline", ...)
  → return job_id

[Celery Worker]
  → workers/pipeline_task.py (acquire GPU lock → LegalDiffPipeline.run() → store results)
  → Redis pub/sub "job:{id}:progress"
  → PostgreSQL UPDATE job status, INSERT reports

[Browser WebSocket]
  → ws/handlers.py → ws/manager.py (subscribe Redis → broadcast to client)
  → Frontend updates progress bar real-time

GET /api/jobs/{id}/reports
  → api/jobs.py → SELECT FROM comparison_reports WHERE job_id = ...
  → return PaginatedResponse[ReportSummaryResponse]

GET /api/reports/{id}
  → api/reports.py → SELECT + join ComparisonJob (authorization)
  → return ReportDetailResponse (full ACUs, evidence, markdown)
```

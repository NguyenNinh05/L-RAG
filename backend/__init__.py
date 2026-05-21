"""
backend — LegalDiff Web Application

REST API + Celery worker cho hệ thống so sánh văn bản pháp lý.

Services:
    - FastAPI web server (CPU)
    - Celery worker (GPU — sequential pipeline execution)
    - PostgreSQL (metadata)
    - Redis (message broker + pub/sub)
"""

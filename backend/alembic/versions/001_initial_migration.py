"""initial migration — create all tables

Revision ID: 001
Revises:
Create Date: 2026-05-21
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(), primary_key=True),
        sa.Column("email", sa.String(255), unique=True, nullable=False, index=True),
        sa.Column("username", sa.String(100), unique=True, nullable=False, index=True),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255), nullable=True),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("is_superuser", sa.Boolean(), default=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(), primary_key=True),
        sa.Column("user_id", postgresql.UUID(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("original_filename", sa.String(255), nullable=False),
        sa.Column("storage_path", sa.String(512), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("mime_type", sa.String(50), nullable=True),
        sa.Column("content_hash_sha256", sa.String(64), nullable=True, index=True),
        sa.Column("doc_title", sa.String(512), nullable=True),
        sa.Column("doc_number", sa.String(128), nullable=True),
        sa.Column("signing_date", sa.String(64), nullable=True),
        sa.Column("parties", postgresql.JSONB(), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("article_count", sa.Integer(), nullable=True),
        sa.Column("is_processed", sa.Boolean(), default=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    op.create_table(
        "comparison_jobs",
        sa.Column("id", postgresql.UUID(), primary_key=True),
        sa.Column("user_id", postgresql.UUID(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("document_v1_id", postgresql.UUID(), sa.ForeignKey("documents.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("document_v2_id", postgresql.UUID(), sa.ForeignKey("documents.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("status", sa.String(20), default="pending", index=True),
        sa.Column("current_phase", sa.String(20), default="queued"),
        sa.Column("progress_pct", sa.Integer(), default=0),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("catalog", postgresql.JSONB(), nullable=True),
        sa.Column("v1_doc_id", sa.String(64), nullable=True),
        sa.Column("v2_doc_id", sa.String(64), nullable=True),
        sa.Column("total_pairs", sa.Integer(), nullable=True),
        sa.Column("matched_count", sa.Integer(), nullable=True),
        sa.Column("added_count", sa.Integer(), nullable=True),
        sa.Column("deleted_count", sa.Integer(), nullable=True),
        sa.Column("split_count", sa.Integer(), nullable=True),
        sa.Column("merge_count", sa.Integer(), nullable=True),
        sa.Column("config_snapshot", postgresql.JSONB(), nullable=True),
        sa.Column("celery_task_id", sa.String(255), nullable=True, index=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    op.create_table(
        "comparison_reports",
        sa.Column("id", postgresql.UUID(), primary_key=True),
        sa.Column("job_id", postgresql.UUID(), sa.ForeignKey("comparison_jobs.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("pair_id", sa.String(64), nullable=False),
        sa.Column("match_type", sa.String(20), nullable=False),
        sa.Column("location_context", sa.String(512), default=""),
        sa.Column("total_acus", sa.Integer(), default=0),
        sa.Column("passed_acus", sa.Integer(), default=0),
        sa.Column("rejected_acus", sa.Integer(), default=0),
        sa.Column("hallucination_rate", sa.Float(), default=0.0),
        sa.Column("executive_summary", postgresql.JSONB(), nullable=True),
        sa.Column("markdown_report", sa.Text(), nullable=True),
        sa.Column("verified_acus", postgresql.JSONB(), nullable=True),
        sa.Column("rejected_acus_detail", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("comparison_reports")
    op.drop_table("comparison_jobs")
    op.drop_table("documents")
    op.drop_table("users")

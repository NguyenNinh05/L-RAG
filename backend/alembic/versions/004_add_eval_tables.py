"""add eval_runs + eval_pairs tables

Revision ID: 004
Revises: 003
Create Date: 2026-06-25
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "eval_runs",
        sa.Column("id", postgresql.UUID(), primary_key=True),
        sa.Column("slug", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=255), nullable=False),
        sa.Column("source_path", sa.String(length=512), nullable=False),
        sa.Column("num_pairs", sa.Integer(), nullable=False),
        sa.Column("aggregate", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("markdown_report", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
        sa.UniqueConstraint("slug", name="uq_eval_runs_slug"),
    )
    op.create_index("ix_eval_runs_slug", "eval_runs", ["slug"], unique=True)

    op.create_table(
        "eval_pairs",
        sa.Column("id", postgresql.UUID(), primary_key=True),
        sa.Column(
            "run_id",
            postgresql.UUID(),
            sa.ForeignKey("eval_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("pair_name", sa.String(length=128), nullable=False),
        sa.Column("v1_file", sa.String(length=512), nullable=True),
        sa.Column("v2_file", sa.String(length=512), nullable=True),
        sa.Column("gt_file", sa.String(length=512), nullable=True),
        sa.Column("phase1", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("phase2", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("phase3", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("ground_truth", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("change_detection", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("per_category", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("per_category_semantic", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("acus", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("errors", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("total_time", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
        sa.UniqueConstraint("run_id", "pair_name", name="uq_eval_pairs_run_pair"),
    )
    op.create_index("ix_eval_pairs_run_id", "eval_pairs", ["run_id"])
    op.create_index("ix_eval_pairs_pair_name", "eval_pairs", ["pair_name"])


def downgrade() -> None:
    op.drop_index("ix_eval_pairs_pair_name", table_name="eval_pairs")
    op.drop_index("ix_eval_pairs_run_id", table_name="eval_pairs")
    op.drop_table("eval_pairs")
    op.drop_index("ix_eval_runs_slug", table_name="eval_runs")
    op.drop_table("eval_runs")

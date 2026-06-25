"""
backend/scripts/ingest_eval.py — Ingest a benchmark evaluation run into the DB.

Usage (from project root, with the l_rag env active):
    PYTHONPATH=/home/haipd/L-RAG python backend/scripts/ingest_eval.py \
        --dir evaluation/outputs/eval_results_v3 --slug v3 --label "v3 — Qwen2.5 ACU split"

Idempotent: upserts EvalRun by `slug`, then replaces that run's EvalPair rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

# Ensure project root is importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.database import _sync_engine  # noqa: E402
from backend.models.eval import EvalPair, EvalRun  # noqa: E402

logger = logging.getLogger("ingest_eval")


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ingest(dir_path: Path, slug: str, label: str) -> None:
    agg = _load_json(dir_path / "aggregate_metrics.json")
    pairs = _load_json(dir_path / "per_pair_results.json")
    md_path = dir_path / "evaluation_report.md"
    markdown = md_path.read_text(encoding="utf-8") if md_path.exists() else None

    engine = _sync_engine()
    with Session(engine) as session:
        run = session.execute(select(EvalRun).where(EvalRun.slug == slug)).scalar_one_or_none()
        if run is None:
            run = EvalRun(slug=slug)
            session.add(run)
        run.label = label
        run.source_path = str(dir_path)
        run.num_pairs = len(pairs)
        run.aggregate = agg
        run.markdown_report = markdown
        session.flush()
        run_id = run.id

        # Replace this run's pairs (idempotent re-runs).
        session.execute(delete(EvalPair).where(EvalPair.run_id == run_id))

        for name, p in pairs.items():
            session.add(
                EvalPair(
                    run_id=run_id,
                    pair_name=str(name),
                    v1_file=p.get("v1_file"),
                    v2_file=p.get("v2_file"),
                    gt_file=p.get("gt_file"),
                    phase1=p.get("phase1", {}) or {},
                    phase2=p.get("phase2", {}) or {},
                    phase3=p.get("phase3", {}) or {},
                    ground_truth=p.get("ground_truth", {}) or {},
                    change_detection=p.get("change_detection", {}) or {},
                    per_category=p.get("per_category", {}) or {},
                    per_category_semantic=p.get("per_category_semantic", {}) or {},
                    acus=p.get("acus", []) or [],
                    errors=p.get("errors", []) or [],
                    total_time=float(p.get("total_time", 0.0) or 0.0),
                )
            )

        session.commit()

    print(
        f"OK: run '{slug}' ({label}) -> {len(pairs)} pairs ingested "
        f"(f1={agg.get('change_f1')}, f1_sem={agg.get('change_f1_semantic')}, "
        f"halluc={agg.get('hallucination_rate')})"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Ingest an evaluation run into the DB.")
    ap.add_argument("--dir", required=True, help="Path to the eval output directory")
    ap.add_argument("--slug", required=True, help="Unique run slug, e.g. v3")
    ap.add_argument("--label", default="", help="Human-readable label")
    args = ap.parse_args()

    dir_path = Path(args.dir)
    if not dir_path.is_absolute():
        dir_path = _PROJECT_ROOT / dir_path
    if not (dir_path / "aggregate_metrics.json").exists():
        sys.exit(f"Not an eval dir (no aggregate_metrics.json): {dir_path}")

    ingest(dir_path, args.slug, args.label)


if __name__ == "__main__":
    main()

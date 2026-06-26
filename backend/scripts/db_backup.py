"""
backend/scripts/db_backup.py — Dump and load L-RAG database to/from a JSON file.

Usage:
    # To export/backup:
    PYTHONPATH=. python backend/scripts/db_backup.py --action export --file db_backup.json

    # To import/restore:
    PYTHONPATH=. python backend/scripts/db_backup.py --action import --file db_backup.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import uuid
from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.orm import Session
from sqlalchemy.inspection import inspect

# Ensure project root is importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.database import _sync_engine  # noqa: E402
from backend.models.user import User  # noqa: E402
from backend.models.user_settings import UserSettings  # noqa: E402
from backend.models.document import Document  # noqa: E402
from backend.models.comparison_job import ComparisonJob  # noqa: E402
from backend.models.comparison_report import ComparisonReportModel  # noqa: E402
from backend.models.eval import EvalRun, EvalPair  # noqa: E402

MODELS = [
    ("User", User),
    ("UserSettings", UserSettings),
    ("Document", Document),
    ("ComparisonJob", ComparisonJob),
    ("ComparisonReportModel", ComparisonReportModel),
    ("EvalRun", EvalRun),
    ("EvalPair", EvalPair),
]

def export_db(file_path: Path) -> None:
    engine = _sync_engine()
    backup_data = {}

    with Session(engine) as session:
        for name, model_class in MODELS:
            mapper = inspect(model_class)
            columns = [c.key for c in mapper.column_attrs]
            rows = session.query(model_class).all()
            
            rows_list = []
            for row in rows:
                d = {}
                for col in columns:
                    val = getattr(row, col)
                    if isinstance(val, (datetime.datetime, datetime.date)):
                        d[col] = val.isoformat()
                    elif isinstance(val, uuid.UUID):
                        d[col] = str(val)
                    else:
                        d[col] = val
                rows_list.append(d)
            
            backup_data[name] = rows_list
            print(f"Exported {len(rows_list)} rows from table '{name}'")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(backup_data, f, indent=2, ensure_ascii=False)
    print(f"Database exported successfully to: {file_path}")

def import_db(file_path: Path) -> None:
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        backup_data = json.load(f)

    engine = _sync_engine()
    with Session(engine) as session:
        # Clear existing tables in reverse order of foreign key dependency
        print("Clearing existing database tables...")
        for name, model_class in reversed(MODELS):
            count = session.query(model_class).delete()
            print(f"Deleted {count} rows from '{name}'")
        session.flush()

        # Import tables in forward order of dependencies
        print("Importing data...")
        for name, model_class in MODELS:
            mapper = inspect(model_class)
            columns = [c.key for c in mapper.column_attrs]
            rows_data = backup_data.get(name, [])
            
            for row_data in rows_data:
                kwargs = {}
                for col in columns:
                    val = row_data.get(col)
                    if val is None:
                        kwargs[col] = None
                        continue
                    
                    if col in ("created_at", "updated_at", "started_at", "completed_at"):
                        if isinstance(val, str):
                            if val.endswith("Z"):
                                val = val[:-1] + "+00:00"
                            kwargs[col] = datetime.datetime.fromisoformat(val)
                    elif col in ("id", "user_id", "document_v1_id", "document_v2_id", "job_id", "run_id"):
                        if isinstance(val, str):
                            kwargs[col] = uuid.UUID(val)
                    else:
                        kwargs[col] = val
                
                session.add(model_class(**kwargs))
            print(f"Imported {len(rows_data)} rows into table '{name}'")
        
        session.commit()
    print("Database imported/restored successfully!")

def main() -> None:
    ap = argparse.ArgumentParser(description="Backup and restore L-RAG database to/from JSON.")
    ap.add_argument("--action", required=True, choices=["export", "import"], help="Action: export or import")
    ap.add_argument("--file", required=True, help="Path to backup JSON file")
    args = ap.parse_args()

    file_path = Path(args.file)
    
    if args.action == "export":
        export_db(file_path)
    elif args.action == "import":
        import_db(file_path)

if __name__ == "__main__":
    main()

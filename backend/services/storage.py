"""
backend/services/storage.py — Local filesystem storage manager.

Manages: uploads/, processed/, reports/ under the configured storage root.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from pathlib import Path


class FileStorageManager:
    def __init__(self, storage_root: str = "./backend/storage") -> None:
        self._root = Path(storage_root).resolve()
        self._uploads = self._root / "uploads"
        self._processed = self._root / "processed"
        self._reports = self._root / "reports"

    @property
    def root(self) -> Path:
        return self._root

    @property
    def uploads_dir(self) -> Path:
        return self._uploads

    @property
    def processed_dir(self) -> Path:
        return self._processed

    @property
    def reports_dir(self) -> Path:
        return self._reports

    def store_upload(self, content: bytes, original_filename: str) -> tuple[str, str, int]:
        """
        Save uploaded file to disk with a unique name.

        Returns:
            (storage_path, sha256_hash, file_size_bytes)
        """
        month_dir = self._uploads / datetime.now().strftime("%Y-%m")
        month_dir.mkdir(parents=True, exist_ok=True)

        unique_name = f"{uuid.uuid4().hex[:16]}_{original_filename}"
        file_path = month_dir / unique_name
        file_path.write_bytes(content)

        sha256 = hashlib.sha256(content).hexdigest()
        rel_path = str(file_path.relative_to(self._root))

        return rel_path, sha256, len(content)

    def get_absolute_path(self, storage_path: str) -> Path:
        return self._root / storage_path

    def delete_file(self, storage_path: str) -> bool:
        path = self._root / storage_path
        if path.exists():
            path.unlink()
            return True
        return False

    def ensure_directories(self) -> None:
        for d in [self._uploads, self._processed, self._reports]:
            d.mkdir(parents=True, exist_ok=True)

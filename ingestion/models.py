from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ArticleChunk:
    doc_label: str
    doc_id: str
    article_number: str
    title: str | None
    content: str
    page: int | None = None
    page_end: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    raw_index: int = 0
    sub_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_metadata = dict(self.metadata or {})
        normalized_metadata.setdefault("doc_label", self.doc_label)
        normalized_metadata.setdefault("doc_id", self.doc_id)
        normalized_metadata.setdefault("article_number", self.article_number)
        normalized_metadata.setdefault("title", self.title or "")
        normalized_metadata.setdefault("page", self.page)
        normalized_metadata.setdefault(
            "page_end",
            self.page_end if self.page_end is not None else normalized_metadata.get("page_last"),
        )
        normalized_metadata.setdefault("line_start", self.line_start)
        normalized_metadata.setdefault("line_end", self.line_end)
        normalized_metadata.setdefault("char_start", self.char_start)
        normalized_metadata.setdefault("char_end", self.char_end)
        normalized_metadata.setdefault("raw_index", self.raw_index)
        normalized_metadata.setdefault("sub_index", self.sub_index)
        normalized_metadata.setdefault("content_fingerprint", self._content_fingerprint())
        normalized_metadata.setdefault("legacy_chunk_id", self._legacy_chunk_id())
        object.__setattr__(self, "metadata", normalized_metadata)

    @property
    def breadcrumb(self) -> str:
        return str(self.metadata.get("breadcrumb") or self.article_number)

    def _content_fingerprint(self) -> str:
        normalized = " ".join(self.content.split()).lower()
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        return digest[:12]

    def _legacy_chunk_id(self) -> str:
        return f"{self.doc_id}__{self.breadcrumb.replace(' > ', '_')}__idx{self.raw_index}__sub{self.sub_index}"

    def chunk_id(self) -> str:
        if self.metadata.get("chunk_id"):
            return str(self.metadata["chunk_id"])
        fingerprint = str(self.metadata.get("content_fingerprint") or self._content_fingerprint())
        return f"{self.doc_id}__{self.breadcrumb.replace(' > ', '_')}__fp{fingerprint}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_label": self.doc_label,
            "doc_id": self.doc_id,
            "article_number": self.article_number,
            "title": self.title,
            "content": self.content,
            "page": self.page,
            "page_end": self.page_end,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "raw_index": self.raw_index,
            "sub_index": self.sub_index,
            "metadata": dict(self.metadata),
        }

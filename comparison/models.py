from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SourceRef:
    file: str
    page: int | None = None
    page_end: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    chunk_id: str | None = None
    context_snippet: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiffSnippet:
    old: str | None = None
    new: str | None = None
    format: str = "diff"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EntityChange:
    entity_key: str
    field: str
    before: str | None = None
    after: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChangeRecord:
    clause_id: str
    change_kind: str
    impact_level: str
    semantic_effect: str = "SUBSTANTIVE"
    review_status: str = "AUTO"
    semantic_source: str = "deterministic"
    tags: list[str] = field(default_factory=list)
    meta_tags: list[str] = field(default_factory=list)
    confidence_score: float = 0.0
    confidence_band: str = "SUSPECT"
    diff_snippet: DiffSnippet = field(default_factory=DiffSnippet)
    source_a: SourceRef | None = None
    source_b: SourceRef | None = None
    entity_changes: list[EntityChange] = field(default_factory=list)
    llm_notes: str | None = None
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "clause_id": self.clause_id,
            "change_kind": self.change_kind,
            "impact_level": self.impact_level,
            "semantic_effect": self.semantic_effect,
            "review_status": self.review_status,
            "semantic_source": self.semantic_source,
            "tags": list(self.tags),
            "meta_tags": list(self.meta_tags),
            "confidence_score": round(self.confidence_score, 4),
            "confidence_band": self.confidence_band,
            "diff_snippet": self.diff_snippet.to_dict(),
            "source_a": self.source_a.to_dict() if self.source_a else None,
            "source_b": self.source_b.to_dict() if self.source_b else None,
            "entity_changes": [change.to_dict() for change in self.entity_changes],
            "llm_notes": self.llm_notes,
            "summary": self.summary,
        }


@dataclass
class ClauseResult:
    clause_id: str
    clause_change_kind: str
    citation_type: str
    semantic_similarity: float
    source_a: SourceRef | None = None
    source_b: SourceRef | None = None
    text_a: str | None = None
    text_b: str | None = None
    records: list[ChangeRecord] = field(default_factory=list)
    llm_notes: str | None = None
    llm_analysis: dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "clause_id": self.clause_id,
            "clause_change_kind": self.clause_change_kind,
            "citation_type": self.citation_type,
            "semantic_similarity": round(self.semantic_similarity, 4),
            "source_a": self.source_a.to_dict() if self.source_a else None,
            "source_b": self.source_b.to_dict() if self.source_b else None,
            "text_a": self.text_a,
            "text_b": self.text_b,
            "records": [record.to_dict() for record in self.records],
            "llm_notes": self.llm_notes,
            "llm_analysis": dict(self.llm_analysis),
            "summary": self.summary,
        }


@dataclass
class ComparisonStats:
    modified: int = 0
    added: int = 0
    deleted: int = 0
    unchanged: int = 0
    substantive: int = 0
    formal: int = 0
    replaced: int = 0
    clauses_affected: int = 0
    atomic_changes: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "modified": self.modified,
            "added": self.added,
            "deleted": self.deleted,
            "unchanged": self.unchanged,
            "substantive": self.substantive,
            "formal": self.formal,
            "replaced": self.replaced,
            "clauses_affected": self.clauses_affected,
            "atomic_changes": self.atomic_changes,
        }


@dataclass
class ComparisonResult:
    stats: ComparisonStats
    changes: list[ChangeRecord] = field(default_factory=list)
    clauses: list[ClauseResult] = field(default_factory=list)
    executive_summary: dict[str, Any] = field(default_factory=dict)
    report_markdown: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stats": self.stats.to_dict(),
            "changes": [change.to_dict() for change in self.changes],
            "clauses": [clause.to_dict() for clause in self.clauses],
            "executive_summary": dict(self.executive_summary),
            "report_markdown": self.report_markdown,
        }

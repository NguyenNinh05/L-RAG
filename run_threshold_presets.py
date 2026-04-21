from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from comparison import build_comparison_result
from embedding import embed_chunks
from ingestion import process_two_documents
from retrieval import build_comparison_pairs

ROOT = Path(__file__).resolve().parent
DATA_TEST_DIR = ROOT / "docs_test" / "data_test"
OUTPUT_PATH = ROOT / "data" / "matching_threshold_presets.json"


PRESETS = {
    "conservative": {
        "unchanged_threshold": 0.96,
        "modified_threshold": 0.78,
        "near_unchanged_band": 0.015,
        "clause_hint_alpha": 0.06,
        "clause_hint_min_gap": 3,
    },
    "balanced": {
        "unchanged_threshold": 0.95,
        "modified_threshold": 0.75,
        "near_unchanged_band": 0.02,
        "clause_hint_alpha": 0.08,
        "clause_hint_min_gap": 3,
    },
    "aggressive": {
        "unchanged_threshold": 0.94,
        "modified_threshold": 0.72,
        "near_unchanged_band": 0.03,
        "clause_hint_alpha": 0.10,
        "clause_hint_min_gap": 2,
    },
}


@dataclass
class CaseMetrics:
    case_name: str
    gt_changes: int
    predicted_atomic_changes: int
    clause_recall: float
    number_recall: float
    volume_alignment: float


def _discover_cases() -> list[tuple[str, Path, Path, Path]]:
    cases: list[tuple[str, Path, Path, Path]] = []
    for folder in sorted(DATA_TEST_DIR.glob("Test_*_Difficult")):
        gt_path = folder / "ground_truth.json"
        if not gt_path.exists():
            continue
        v1_candidates = sorted(folder.glob("*_v1.docx"))
        v2_candidates = sorted(folder.glob("*_v2.docx"))
        if not v1_candidates or not v2_candidates:
            continue
        cases.append((folder.name, v1_candidates[0], v2_candidates[0], gt_path))
    return cases


def _extract_clause_like_tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    values = re.findall(r"(Điều\s+\d+|Phụ lục\s+[A-Z0-9]+)", text, flags=re.IGNORECASE)
    return {" ".join(v.lower().split()) for v in values}


def _extract_number_tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", text))


def _deterministic_guards(report_markdown: str, ground_truth: dict[str, Any], stats_dict: dict[str, Any]) -> tuple[float, float, float]:
    gt_changes = ground_truth.get("changes", [])
    report_text = report_markdown or ""

    gt_clause_tokens: set[str] = set()
    gt_number_tokens: set[str] = set()
    for change in gt_changes:
        gt_clause_tokens |= _extract_clause_like_tokens(change.get("location"))
        gt_clause_tokens |= _extract_clause_like_tokens(change.get("v1_content"))
        gt_clause_tokens |= _extract_clause_like_tokens(change.get("v2_content"))
        gt_number_tokens |= _extract_number_tokens(change.get("v1_content"))
        gt_number_tokens |= _extract_number_tokens(change.get("v2_content"))

    report_clause_tokens = _extract_clause_like_tokens(report_text)
    report_number_tokens = _extract_number_tokens(report_text)

    clause_recall = (
        len(gt_clause_tokens & report_clause_tokens) / len(gt_clause_tokens)
        if gt_clause_tokens
        else 1.0
    )
    number_recall = (
        len(gt_number_tokens & report_number_tokens) / len(gt_number_tokens)
        if gt_number_tokens
        else 1.0
    )

    predicted_atomic = int(stats_dict.get("atomic_changes", 0))
    expected_atomic = len(gt_changes)
    volume_alignment = 1.0
    if expected_atomic > 0:
        volume_alignment = max(0.0, 1.0 - abs(predicted_atomic - expected_atomic) / expected_atomic)

    return clause_recall, number_recall, volume_alignment


def _simple_report_text(result_dict: dict[str, Any]) -> str:
    parts: list[str] = []
    for clause in result_dict.get("clauses", []):
        cid = clause.get("clause_id") or ""
        parts.append(str(cid))
        for rec in clause.get("records", []):
            parts.append(str(rec.get("summary") or ""))
            diff = rec.get("diff_snippet") or {}
            parts.append(str(diff.get("old") or ""))
            parts.append(str(diff.get("new") or ""))
    return "\n".join(parts)


def _evaluate_preset(name: str, preset: dict[str, Any], cases: list[tuple[str, Path, Path, Path]]) -> dict[str, Any]:
    case_metrics: list[CaseMetrics] = []

    for case_name, file_v1, file_v2, gt_path in cases:
        ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))

        chunks_a, chunks_b = process_two_documents(str(file_v1), str(file_v2))
        embeds_a = embed_chunks(chunks_a)
        embeds_b = embed_chunks(chunks_b)

        pairs = build_comparison_pairs(
            chunks_a,
            chunks_b,
            embeds_a=embeds_a,
            embeds_b=embeds_b,
            unchanged_threshold=float(preset["unchanged_threshold"]),
            modified_threshold=float(preset["modified_threshold"]),
            near_unchanged_band=float(preset["near_unchanged_band"]),
            clause_hint_alpha=float(preset["clause_hint_alpha"]),
            clause_hint_min_gap=int(preset["clause_hint_min_gap"]),
        )
        comparison_result = build_comparison_result(pairs, file_a=file_v1.name, file_b=file_v2.name)

        result_dict = comparison_result.to_dict()
        pseudo_report = _simple_report_text(result_dict)
        clause_recall, number_recall, volume_alignment = _deterministic_guards(
            pseudo_report,
            ground_truth,
            comparison_result.stats.to_dict(),
        )

        case_metrics.append(
            CaseMetrics(
                case_name=case_name,
                gt_changes=len(ground_truth.get("changes", [])),
                predicted_atomic_changes=comparison_result.stats.atomic_changes,
                clause_recall=round(clause_recall, 4),
                number_recall=round(number_recall, 4),
                volume_alignment=round(volume_alignment, 4),
            )
        )

    avg_clause = sum(item.clause_recall for item in case_metrics) / len(case_metrics)
    avg_number = sum(item.number_recall for item in case_metrics) / len(case_metrics)
    avg_volume = sum(item.volume_alignment for item in case_metrics) / len(case_metrics)

    # Weighted score prioritizes factual coverage, then volume sanity.
    composite = 0.45 * avg_clause + 0.45 * avg_number + 0.10 * avg_volume

    return {
        "preset": name,
        "params": preset,
        "summary": {
            "avg_clause_recall": round(avg_clause, 4),
            "avg_number_recall": round(avg_number, 4),
            "avg_volume_alignment": round(avg_volume, 4),
            "composite_score": round(composite, 4),
        },
        "cases": [
            {
                "case_name": item.case_name,
                "gt_changes": item.gt_changes,
                "predicted_atomic_changes": item.predicted_atomic_changes,
                "clause_recall": item.clause_recall,
                "number_recall": item.number_recall,
                "volume_alignment": item.volume_alignment,
            }
            for item in case_metrics
        ],
    }


def main() -> None:
    cases = _discover_cases()
    if not cases:
        raise RuntimeError("No difficult test cases found in docs_test/data_test")

    results = [_evaluate_preset(name, preset, cases) for name, preset in PRESETS.items()]
    best = max(results, key=lambda item: item["summary"]["composite_score"])

    payload = {
        "total_cases": len(cases),
        "presets": results,
        "recommended": {
            "preset": best["preset"],
            "composite_score": best["summary"]["composite_score"],
            "params": best["params"],
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload["recommended"], ensure_ascii=False, indent=2))
    print(f"Saved preset report to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

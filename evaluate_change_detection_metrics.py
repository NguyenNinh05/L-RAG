from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from comparison import build_comparison_result
from embedding import embed_chunks
from ingestion import process_two_documents
from retrieval import build_comparison_pairs

ROOT = Path(__file__).resolve().parent
DATA_TEST_DIR = ROOT / "docs_test" / "data_test"
OUTPUT_JSON = ROOT / "data" / "change_detection_metrics.json"
OUTPUT_MD = ROOT / "data" / "change_detection_metrics.md"

MATCH_THRESHOLD = 0.08


@dataclass(slots=True)
class MatchOutcome:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    decomposed = unicodedata.normalize("NFKD", value.lower())
    no_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    no_marks = no_marks.replace("đ", "d").replace("Đ", "d")
    no_marks = re.sub(r"[^\w\s]", " ", no_marks)
    return re.sub(r"\s+", " ", no_marks).strip()


def _extract_clause_anchors(value: str | None) -> set[str]:
    if not value:
        return set()
    normalized = _normalize_text(value)
    anchors = set(re.findall(r"(?:dieu\s+\d+[a-z]?|phu\s+luc\s+[a-z0-9]+|chuong\s+[ivxlcdm0-9]+)", normalized))
    return anchors


def _extract_numbers(value: str | None) -> set[str]:
    if not value:
        return set()
    return set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", value))


def _change_kind_hint(change_type: str) -> set[str]:
    lowered = _normalize_text(change_type)
    hints: set[str] = set()
    if any(keyword in lowered for keyword in ("numerical", "number", "so lieu")):
        hints.add("Số liệu")
    if any(keyword in lowered for keyword in ("typo", "spelling", "unaccented")):
        hints.add("Chính tả")
    if any(keyword in lowered for keyword in ("format", "formatting", "table", "structural", "swap", "merge", "split")):
        hints.add("Định dạng")
    if any(keyword in lowered for keyword in ("paraphrase", "semantic", "logic")):
        hints.add("SUBSTANTIVE")
    if any(keyword in lowered for keyword in ("silent deletion", "deletion", "addition", "removed", "removed")):
        hints.add("SUBSTANTIVE")
    return hints


def _pred_text(change: Any) -> str:
    diff_snippet = getattr(change, "diff_snippet", None)
    old_text = getattr(diff_snippet, "old", None) if diff_snippet else None
    new_text = getattr(diff_snippet, "new", None) if diff_snippet else None
    return " ".join(
        part
        for part in [
            getattr(change, "clause_id", ""),
            getattr(change, "summary", ""),
            getattr(change, "change_kind", ""),
            getattr(change, "impact_level", ""),
            getattr(change, "semantic_effect", ""),
            old_text or "",
            new_text or "",
            " ".join(getattr(change, "tags", []) or []),
            " ".join(getattr(change, "meta_tags", []) or []),
        ]
        if part
    )


def _gt_text(change: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in [
            str(change.get("location") or ""),
            str(change.get("type") or ""),
            str(change.get("v1_content") or ""),
            str(change.get("v2_content") or ""),
        ]
        if part
    )


def _pair_score(pred_change: Any, gt_change: dict[str, Any]) -> float:
    pred_text = _normalize_text(_pred_text(pred_change))
    gt_text = _normalize_text(_gt_text(gt_change))
    if not pred_text or not gt_text:
        return 0.0

    text_sim = SequenceMatcher(None, pred_text, gt_text, autojunk=False).ratio()

    pred_anchor = _extract_clause_anchors(_pred_text(pred_change))
    gt_anchor = _extract_clause_anchors(_gt_text(gt_change))
    anchor_bonus = 0.0
    if pred_anchor and gt_anchor and pred_anchor & gt_anchor:
        anchor_bonus = 0.18

    gt_numbers = _extract_numbers(gt_change.get("v1_content")) | _extract_numbers(gt_change.get("v2_content"))
    pred_numbers = _extract_numbers(_pred_text(pred_change))
    if gt_numbers and pred_numbers:
        number_overlap = len(gt_numbers & pred_numbers) / len(gt_numbers | pred_numbers)
    else:
        number_overlap = 0.0

    gt_location = _normalize_text(str(gt_change.get("location") or ""))
    pred_location = _normalize_text(str(getattr(pred_change, "clause_id", "") or getattr(pred_change, "summary", "")))
    location_sim = SequenceMatcher(None, pred_location, gt_location, autojunk=False).ratio() if pred_location and gt_location else 0.0

    gt_hint = _change_kind_hint(str(gt_change.get("type") or ""))
    pred_hint = set(_normalize_text(_pred_text(pred_change)).split())
    hint_bonus = 0.0
    if gt_hint:
        if "Số liệu" in gt_hint and ("so" in pred_hint or "số" in pred_hint):
            hint_bonus += 0.05
        if "Chính tả" in gt_hint and any(token in pred_hint for token in {"typo", "chinh", "ta"}):
            hint_bonus += 0.05
        if "Định dạng" in gt_hint and any(token in pred_hint for token in {"format", "table", "structural"}):
            hint_bonus += 0.05
        if "SUBSTANTIVE" in gt_hint and getattr(pred_change, "impact_level", "") == "substantive":
            hint_bonus += 0.06

    score = 0.46 * text_sim + 0.26 * location_sim + 0.16 * number_overlap + 0.12 * anchor_bonus + hint_bonus
    return min(score, 1.0)


def _match_case(predicted_changes: list[Any], ground_truth_changes: list[dict[str, Any]]) -> tuple[list[tuple[int, int, float]], MatchOutcome]:
    scored_pairs: list[tuple[float, int, int]] = []
    for pred_index, pred_change in enumerate(predicted_changes):
        for gt_index, gt_change in enumerate(ground_truth_changes):
            score = _pair_score(pred_change, gt_change)
            if score >= MATCH_THRESHOLD:
                scored_pairs.append((score, pred_index, gt_index))

    scored_pairs.sort(reverse=True, key=lambda item: item[0])
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for score, pred_index, gt_index in scored_pairs:
        if pred_index in matched_pred or gt_index in matched_gt:
            continue
        matched_pred.add(pred_index)
        matched_gt.add(gt_index)
        matches.append((pred_index, gt_index, score))

    tp = len(matches)
    fp = len(predicted_changes) - tp
    fn = len(ground_truth_changes) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    outcome = MatchOutcome(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)
    return matches, outcome


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _evaluate_case(case_name: str, file_v1: Path, file_v2: Path, gt_path: Path) -> dict[str, Any]:
    ground_truth = _read_json(gt_path)

    chunks_a, chunks_b = process_two_documents(str(file_v1), str(file_v2))
    embeds_a = embed_chunks(chunks_a)
    embeds_b = embed_chunks(chunks_b)
    pairs = build_comparison_pairs(chunks_a, chunks_b, embeds_a=embeds_a, embeds_b=embeds_b)
    comparison_result = build_comparison_result(pairs, file_a=file_v1.name, file_b=file_v2.name)

    predicted_changes = list(comparison_result.changes)
    ground_truth_changes = list(ground_truth.get("changes", []))
    matches, outcome = _match_case(predicted_changes, ground_truth_changes)

    return {
        "case_name": case_name,
        "file_v1": file_v1.name,
        "file_v2": file_v2.name,
        "gt_changes": len(ground_truth_changes),
        "predicted_changes": len(predicted_changes),
        "tp": outcome.tp,
        "fp": outcome.fp,
        "fn": outcome.fn,
        "precision": round(outcome.precision, 4),
        "recall": round(outcome.recall, 4),
        "f1": round(outcome.f1, 4),
        "matched_pairs": [
            {
                "pred_index": pred_index,
                "gt_index": gt_index,
                "score": round(score, 4),
                "pred_clause_id": getattr(predicted_changes[pred_index], "clause_id", ""),
                "gt_location": ground_truth_changes[gt_index].get("location"),
                "gt_type": ground_truth_changes[gt_index].get("type"),
            }
            for pred_index, gt_index, score in matches
        ],
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_tp = sum(item["tp"] for item in results)
    total_fp = sum(item["fp"] for item in results)
    total_fn = sum(item["fn"] for item in results)
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    macro_precision = sum(item["precision"] for item in results) / len(results) if results else 0.0
    macro_recall = sum(item["recall"] for item in results) / len(results) if results else 0.0
    macro_f1 = sum(item["f1"] for item in results) / len(results) if results else 0.0

    return {
        "total_cases": len(results),
        "total_gt": sum(item["gt_changes"] for item in results),
        "total_predicted": sum(item["predicted_changes"] for item in results),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
    }


def _render_markdown(results: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Change Detection TP/FP/FN Metrics",
        "",
        f"Match threshold: {MATCH_THRESHOLD}",
        "",
        "| Case | GT | Pred | TP | FP | FN | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        lines.append(
            f"| {item['case_name']} | {item['gt_changes']} | {item['predicted_changes']} | {item['tp']} | {item['fp']} | {item['fn']} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} |"
        )
    lines.extend(
        [
            f"| **Total** | **{summary['total_gt']}** | **{summary['total_predicted']}** | **{summary['tp']}** | **{summary['fp']}** | **{summary['fn']}** | **{summary['precision']:.4f}** | **{summary['recall']:.4f}** | **{summary['f1']:.4f}** |",
            "",
            f"Macro Precision: {summary['macro_precision']:.4f}",
            f"Macro Recall: {summary['macro_recall']:.4f}",
            f"Macro F1: {summary['macro_f1']:.4f}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    case_results: list[dict[str, Any]] = []
    for case_name, file_v1, file_v2, gt_path in _discover_cases():
        case_results.append(_evaluate_case(case_name, file_v1, file_v2, gt_path))

    summary = _aggregate(case_results)
    payload = {
        "match_threshold": MATCH_THRESHOLD,
        "cases": case_results,
        "summary": summary,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(_render_markdown(case_results, summary), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved metrics to: {OUTPUT_JSON}")
    print(f"Saved table to: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
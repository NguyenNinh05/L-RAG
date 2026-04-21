from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from comparison import build_comparison_result
from config import OLLAMA_CHAT_URL, OLLAMA_LLM_MODEL
from embedding import embed_chunks
from ingestion import process_two_documents
from llm import generate_comparison_report
from retrieval import build_comparison_pairs

ROOT = Path(__file__).resolve().parent
DATA_TEST_DIR = ROOT / "docs_test" / "data_test"
OUTPUT_PATH = ROOT / "data" / "matching_baseline_report.json"
USE_LLM_JUDGE = os.getenv("USE_LLM_JUDGE", "0").lower() in {"1", "true", "yes"}


@dataclass
class CaseResult:
    case_name: str
    file_v1: str
    file_v2: str
    gt_changes: int
    predicted_atomic_changes: int
    deterministic_scores: dict[str, float]
    llm_score: float | None
    llm_notes: str | None


def _extract_clause_like_tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    values = re.findall(r"(Điều\s+\d+|Phụ lục\s+[A-Z0-9]+)", text, flags=re.IGNORECASE)
    return {" ".join(v.lower().split()) for v in values}


def _extract_number_tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", text))


def _deterministic_guards(report_markdown: str, ground_truth: dict[str, Any], result_stats: dict[str, Any]) -> dict[str, float]:
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

    predicted_atomic = int(result_stats.get("atomic_changes", 0))
    expected_atomic = len(gt_changes)
    volume_alignment = 1.0
    if expected_atomic > 0:
        volume_alignment = max(0.0, 1.0 - abs(predicted_atomic - expected_atomic) / expected_atomic)

    return {
        "clause_recall": round(clause_recall, 4),
        "number_recall": round(number_recall, 4),
        "volume_alignment": round(volume_alignment, 4),
    }


async def _llm_judge(report_markdown: str, ground_truth: dict[str, Any]) -> tuple[float | None, str | None]:
    prompt = {
        "task": "Judge how well comparison report matches ground truth.",
        "instruction": "Return strict JSON with keys: score (0..10), notes.",
        "ground_truth": ground_truth,
        "report_markdown": report_markdown,
    }
    payload = {
        "model": OLLAMA_LLM_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an evaluator. Compare report against ground truth. "
                    "Be strict. Output valid JSON only: {\"score\": number, \"notes\": string}."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        "options": {"temperature": 0.0, "num_predict": 256},
    }

    try:
        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(OLLAMA_CHAT_URL, json=payload) as response:
                response.raise_for_status()
                body = await response.json()
        content = (body.get("message") or {}).get("content", "").strip()
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            return None, "LLM output not JSON"
        parsed = json.loads(match.group(0))
        score = float(parsed.get("score")) if parsed.get("score") is not None else None
        notes = str(parsed.get("notes")) if parsed.get("notes") is not None else None
        if score is not None:
            score = max(0.0, min(10.0, score))
        return score, notes
    except asyncio.CancelledError as exc:  # pragma: no cover - runtime dependent
        return None, f"LLM judge cancelled: {exc}"
    except Exception as exc:  # pragma: no cover - runtime dependent
        return None, f"LLM judge unavailable: {exc}"


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


async def run_baseline() -> dict[str, Any]:
    case_results: list[CaseResult] = []

    for case_name, file_v1, file_v2, gt_path in _discover_cases():
        ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))

        chunks_a, chunks_b = process_two_documents(str(file_v1), str(file_v2))
        embeds_a = embed_chunks(chunks_a)
        embeds_b = embed_chunks(chunks_b)
        pairs = build_comparison_pairs(chunks_a, chunks_b, embeds_a=embeds_a, embeds_b=embeds_b)
        comparison_result = build_comparison_result(pairs, file_a=file_v1.name, file_b=file_v2.name)
        report_markdown = generate_comparison_report(
            comparison_result,
            file_a=file_v1.name,
            file_b=file_v2.name,
            enable_llm=False,
        )

        det = _deterministic_guards(report_markdown, ground_truth, comparison_result.stats.to_dict())
        llm_score, llm_notes = (None, "LLM judge disabled")
        if USE_LLM_JUDGE:
            llm_score, llm_notes = await _llm_judge(report_markdown, ground_truth)

        case_results.append(
            CaseResult(
                case_name=case_name,
                file_v1=file_v1.name,
                file_v2=file_v2.name,
                gt_changes=len(ground_truth.get("changes", [])),
                predicted_atomic_changes=comparison_result.stats.atomic_changes,
                deterministic_scores=det,
                llm_score=llm_score,
                llm_notes=llm_notes,
            )
        )

    avg_llm = None
    llm_scores = [item.llm_score for item in case_results if item.llm_score is not None]
    if llm_scores:
        avg_llm = round(sum(llm_scores) / len(llm_scores), 4)

    output = {
        "cases": [
            {
                "case_name": item.case_name,
                "file_v1": item.file_v1,
                "file_v2": item.file_v2,
                "gt_changes": item.gt_changes,
                "predicted_atomic_changes": item.predicted_atomic_changes,
                "deterministic_scores": item.deterministic_scores,
                "llm_score": item.llm_score,
                "llm_notes": item.llm_notes,
            }
            for item in case_results
        ],
        "summary": {
            "total_cases": len(case_results),
            "avg_llm_score": avg_llm,
            "avg_clause_recall": round(
                sum(item.deterministic_scores["clause_recall"] for item in case_results) / len(case_results),
                4,
            )
            if case_results
            else None,
            "avg_number_recall": round(
                sum(item.deterministic_scores["number_recall"] for item in case_results) / len(case_results),
                4,
            )
            if case_results
            else None,
            "avg_volume_alignment": round(
                sum(item.deterministic_scores["volume_alignment"] for item in case_results) / len(case_results),
                4,
            )
            if case_results
            else None,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    output = asyncio.run(run_baseline())
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"Saved baseline report to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

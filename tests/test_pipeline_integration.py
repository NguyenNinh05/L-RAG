from __future__ import annotations

import unittest
from pathlib import Path

from comparison import build_comparison_result
from embedding import embed_and_store
from ingestion import process_two_documents
from llm import generate_comparison_report
from retrieval import build_comparison_pairs


class PipelineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        file_a = root / "docs_test" / "v1.pdf"
        file_b = root / "docs_test" / "v2.pdf"

        try:
            chunks_a, chunks_b = process_two_documents(str(file_a), str(file_b))
            _, embeds_a, embeds_b = embed_and_store(chunks_a, chunks_b)
            pairs = build_comparison_pairs(
                chunks_a,
                chunks_b,
                embeds_a=embeds_a,
                embeds_b=embeds_b,
            )
        except Exception as exc:  # pragma: no cover - depends on local Ollama/runtime
            raise unittest.SkipTest(f"Pipeline runtime unavailable: {exc}") from exc

        cls.result = build_comparison_result(pairs, file_a="v1.pdf", file_b="v2.pdf")
        cls.report = generate_comparison_report(
            cls.result,
            file_a="v1.pdf",
            file_b="v2.pdf",
            enable_llm=False,
        )

    def _find_clause(self, name: str):
        for clause in self.result.clauses:
            if clause.clause_id == name:
                return clause
        self.fail(f"Clause not found: {name}")

    def test_appendix_b_is_formal_only(self) -> None:
        clause = self._find_clause("Phụ lục B")
        self.assertTrue(clause.records)
        self.assertTrue(all(record.impact_level == "formal" for record in clause.records))
        tags = {tag for record in clause.records for tag in record.tags}
        self.assertTrue({"Định dạng", "Khoảng trắng"} & tags)

    def test_appendix_a_has_two_replacements_and_one_formal_title_change(self) -> None:
        clause = self._find_clause("Phụ lục A")
        replaced_records = [record for record in clause.records if record.change_kind == "REPLACED"]
        formal_records = [record for record in clause.records if record.impact_level == "formal"]

        self.assertEqual(len(replaced_records), 2)
        self.assertGreaterEqual(len(formal_records), 1)
        self.assertTrue(all("Nhân sự" in record.tags for record in replaced_records))

    def test_article_iii_5_is_location_not_time(self) -> None:
        clause = self._find_clause("Phần III > Điều 5")
        tags = {tag for record in clause.records for tag in record.tags}

        self.assertIn("Địa điểm", tags)
        self.assertNotIn("Thời hạn", tags)

    def test_formal_atomic_count_is_at_least_three(self) -> None:
        self.assertGreaterEqual(self.result.stats.formal, 3)

    def test_every_change_has_citation_and_confidence(self) -> None:
        for change in self.result.changes:
            self.assertTrue(change.confidence_band)
            self.assertTrue(change.source_a or change.source_b)
            source = change.source_a or change.source_b
            self.assertTrue(source.chunk_id)
            self.assertIsNotNone(source.page)

    def test_clause_seven_collapses_to_replaced(self) -> None:
        clause = self._find_clause("Phần II > Điều 7")
        self.assertEqual(clause.clause_change_kind, "REPLACED")

    def test_report_snapshot_shape(self) -> None:
        self.assertIn("## I. EXECUTIVE SUMMARY", self.report)
        self.assertIn("## II. FULL DETAIL", self.report)
        self.assertIn("## III. ĐIỀU KHOẢN/PHỤ LỤC THÊM MỚI", self.report)
        self.assertIn("## IV. ĐIỀU KHOẢN BỊ LOẠI BỎ", self.report)
        self.assertIn("```diff", self.report)
        self.assertIn("Phụ lục A", self.report)

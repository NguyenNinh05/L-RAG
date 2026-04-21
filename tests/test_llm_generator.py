from __future__ import annotations

import unittest

from comparison.models import ChangeRecord, ClauseResult, ComparisonResult, ComparisonStats, DiffSnippet, SourceRef
from llm.generator import _render_record, _validate_executive_summary


class LlmGeneratorRenderTests(unittest.TestCase):
    def test_lexical_equivalent_record_uses_compact_render(self) -> None:
        record = ChangeRecord(
            clause_id="Điều X",
            change_kind="MODIFIED",
            impact_level="substantive",
            semantic_effect="LEXICAL_EQUIVALENT",
            review_status="AUTO",
            semantic_source="llm",
            tags=["Ngôn ngữ chuyên môn"],
            confidence_score=0.88,
            confidence_band="HIGH",
            diff_snippet=DiffSnippet(
                old="Bên A có quyền chấm dứt hợp đồng.",
                new="Bên A có quyền kết thúc hợp đồng.",
                format="diff",
            ),
            source_a=SourceRef(file="v1.pdf", page=1, chunk_id="chunk_a"),
            source_b=SourceRef(file="v2.pdf", page=1, chunk_id="chunk_b"),
            llm_notes="Cụm từ diễn đạt được thay đổi nhưng không làm đổi ngữ nghĩa của mệnh đề.",
            summary="Điều khoản thay đổi nội dung.",
        )

        lines = _render_record(record)
        rendered = "\n".join(lines)

        self.assertIn("[THAY ĐỔI TỪ NGỮ] [KHÔNG ĐỔI NGỮ NGHĨA]", rendered)
        self.assertIn("**Factual summary:**", rendered)
        self.assertIn("**Confidence:**", rendered)
        self.assertIn("**Nguồn:**", rendered)
        self.assertIn("```diff", rendered)
        self.assertNotIn("**Tóm tắt deterministic:**", rendered)
        self.assertNotIn("**Semantic status:**", rendered)


class ExecutiveSummaryGuardrailTests(unittest.TestCase):
    def _make_result(self) -> ComparisonResult:
        change = ChangeRecord(
            clause_id="Phần I > Điều 3",
            change_kind="MODIFIED",
            impact_level="substantive",
            semantic_effect="SUBSTANTIVE",
            tags=["Thời hạn", "Số liệu"],
            confidence_score=0.92,
            confidence_band="HIGH",
            diff_snippet=DiffSnippet(old="24 tháng", new="36 tháng", format="diff"),
            source_a=SourceRef(file="v1.pdf", page=2, chunk_id="a_3"),
            source_b=SourceRef(file="v2.pdf", page=2, chunk_id="b_3"),
            summary="Thời hạn hợp đồng tăng từ 24 tháng lên 36 tháng.",
        )

        clause_anchor = ClauseResult(
            clause_id="Phần II > Điều 7",
            clause_change_kind="REPLACED",
            citation_type="REPLACED",
            semantic_similarity=0.81,
            source_a=SourceRef(file="v1.pdf", page=5, chunk_id="a_7"),
            source_b=SourceRef(file="v2.pdf", page=5, chunk_id="b_7"),
            text_a="Điều 7: Cam kết mức độ dịch vụ (SLA)",
            text_b="Điều 7: Bảo mật và bảo vệ dữ liệu",
            records=[change],
            summary="Điều 7 được thay nội dung.",
        )

        clause_ref = ClauseResult(
            clause_id="Phần II > Điều 1",
            clause_change_kind="MODIFIED",
            citation_type="MODIFIED",
            semantic_similarity=0.9,
            source_a=SourceRef(file="v1.pdf", page=4, chunk_id="a_1"),
            source_b=SourceRef(file="v2.pdf", page=4, chunk_id="b_1"),
            text_a="- f) Bảo trì và hỗ trợ kỹ thuật theo cam kết SLA.",
            text_b="- f) Bảo trì và hỗ trợ kỹ thuật theo cam kết SLA.",
            records=[change],
            summary="Điều 1 cập nhật phạm vi dịch vụ.",
        )

        stats = ComparisonStats(
            modified=1,
            added=0,
            deleted=0,
            unchanged=0,
            substantive=1,
            formal=0,
            replaced=1,
            clauses_affected=2,
            atomic_changes=1,
        )
        return ComparisonResult(stats=stats, changes=[change], clauses=[clause_anchor, clause_ref])

    def test_validate_executive_summary_drops_ungrounded_key_points(self) -> None:
        result = self._make_result()
        raw_summary = {
            "overview": "Các thay đổi tập trung vào điều khoản thời hạn và phạm vi áp dụng.",
            "key_points": [
                "Thời hạn hợp đồng được kéo dài.",
                "Phụ lục A bị thay thế hoàn toàn.",
            ],
            "review_alerts": [],
        }

        validated = _validate_executive_summary(result, raw_summary)

        self.assertIn("Thời hạn hợp đồng được kéo dài", " ".join(validated["key_points"]))
        self.assertFalse(any("Phụ lục A" in item for item in validated["key_points"]))
        self.assertIn("key_points_evidence", validated)
        self.assertTrue(validated["key_points_evidence"])

    def test_validate_executive_summary_adds_sla_anomaly_alert(self) -> None:
        result = self._make_result()

        validated = _validate_executive_summary(result, None)

        joined_alerts = "\n".join(validated["review_alerts"])
        self.assertIn("CRITICAL ANOMALY", joined_alerts)
        self.assertIn("SLA", joined_alerts)

    def test_validate_executive_summary_adds_cross_clause_metric_conflict(self) -> None:
        change_uptime = ChangeRecord(
            clause_id="Phần II > Điều 7",
            change_kind="MODIFIED",
            impact_level="substantive",
            semantic_effect="SUBSTANTIVE",
            tags=["Số liệu"],
            confidence_score=0.9,
            confidence_band="HIGH",
            diff_snippet=DiffSnippet(old="Uptime cam kết: 99.5%", new="Uptime cam kết: 99.9%", format="diff"),
            source_a=SourceRef(file="v1.pdf", page=5, chunk_id="a_7"),
            source_b=SourceRef(file="v2.pdf", page=5, chunk_id="b_7"),
            summary="Uptime thay đổi.",
        )
        change_appendix = ChangeRecord(
            clause_id="Phụ lục B",
            change_kind="MODIFIED",
            impact_level="substantive",
            semantic_effect="SUBSTANTIVE",
            tags=["Số liệu"],
            confidence_score=0.88,
            confidence_band="HIGH",
            diff_snippet=DiffSnippet(old="Uptime > 99.5%/tháng", new="Uptime > 99.5%/tháng", format="diff"),
            source_a=SourceRef(file="v1.pdf", page=10, chunk_id="a_b"),
            source_b=SourceRef(file="v2.pdf", page=10, chunk_id="b_b"),
            summary="Phụ lục B vẫn yêu cầu uptime 99.5%.",
        )

        clause_uptime = ClauseResult(
            clause_id="Phần II > Điều 7",
            clause_change_kind="MODIFIED",
            citation_type="MODIFIED",
            semantic_similarity=0.91,
            source_a=change_uptime.source_a,
            source_b=change_uptime.source_b,
            text_a="Điều 7: Uptime cam kết: 99.5%/tháng.",
            text_b="Điều 7: Uptime cam kết: 99.9%/tháng.",
            records=[change_uptime],
            summary="Điều 7 cập nhật uptime.",
        )
        clause_appendix = ClauseResult(
            clause_id="Phụ lục B",
            clause_change_kind="MODIFIED",
            citation_type="MODIFIED",
            semantic_similarity=0.87,
            source_a=change_appendix.source_a,
            source_b=change_appendix.source_b,
            text_a="Phụ lục B: Uptime > 99.5%/tháng.",
            text_b="Phụ lục B: Uptime > 99.5%/tháng.",
            records=[change_appendix],
            summary="Phụ lục B không đổi ngưỡng uptime.",
        )
        stats = ComparisonStats(
            modified=2,
            added=0,
            deleted=0,
            unchanged=0,
            substantive=2,
            formal=0,
            replaced=0,
            clauses_affected=2,
            atomic_changes=2,
        )
        result = ComparisonResult(
            stats=stats,
            changes=[change_uptime, change_appendix],
            clauses=[clause_uptime, clause_appendix],
        )

        validated = _validate_executive_summary(result, None)
        joined_alerts = "\n".join(validated["review_alerts"])

        self.assertIn("CRITICAL ANOMALY", joined_alerts)
        self.assertIn("uptime", joined_alerts.lower())
        self.assertIn("99.9", joined_alerts)
        self.assertIn("99.5", joined_alerts)

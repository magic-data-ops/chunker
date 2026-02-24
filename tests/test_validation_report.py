"""Tests for qa_generation/validate_qa_dataset.py — report building and printing."""
from __future__ import annotations

import io
import sys

import pytest
from qa_generation import validate_qa_dataset as val


class TestBuildReport:
    def test_empty_list_returns_error(self):
        report = val.build_report([])
        assert "error" in report
        assert "total_chains" not in report

    def test_aggregation(self, sample_chain):
        chains = [sample_chain, {**sample_chain, "chain_id": "test-002", "approved": False,
                                  "category_suitability_score": 0.5,
                                  "answer_completeness_score": 0.4}]
        report = val.build_report(chains)
        assert report["total_chains"] == 2
        assert report["total_approved"] == 1
        targets = report["targets"]
        assert targets["overall_approval_rate"]["value"] == pytest.approx(0.5)
        # Mean suitability = (0.9 + 0.5) / 2 = 0.7
        assert targets["mean_category_suitability"]["value"] == pytest.approx(0.7)
        # Mean completeness = (0.85 + 0.4) / 2 = 0.625
        assert targets["mean_answer_completeness"]["value"] == pytest.approx(0.625)

    def test_per_category_breakdown(self, sample_chain):
        report = val.build_report([sample_chain])
        assert "long_context_citation" in report["per_category"]
        cat = report["per_category"]["long_context_citation"]
        assert cat["total"] == 1
        assert cat["approved"] == 1

    def test_multiturn_metrics_present(self, sample_chain):
        report = val.build_report([sample_chain])
        assert "multiturn_metrics" in report
        mt = report["multiturn_metrics"]
        assert "total_multiturn" in mt
        assert "total_single_turn" in mt
        assert "turn_count_distribution" in mt

    def test_multiturn_metrics_with_multiturn_chain(self, sample_multiturn_chain):
        chain = {
            **sample_multiturn_chain,
            "approved": True,
            "category_suitability_score": 0.9,
            "answer_completeness_score": 0.85,
            "conversation_quality_score": 0.8,
            "answer_leakage_detected": False,
        }
        report = val.build_report([chain])
        mt = report["multiturn_metrics"]
        assert mt["total_multiturn"] == 1
        assert mt["total_single_turn"] == 0
        assert mt["mean_conversation_quality"] == pytest.approx(0.8)
        assert mt["answer_leakage_count"] == 0
        assert mt["answer_leakage_rate"] == pytest.approx(0.0)


class TestPrintReport:
    def test_error_dict_handled(self, capsys):
        """Bug 3: print_report should not crash on error dict."""
        report = val.build_report([])
        val.print_report(report)  # Should not raise
        captured = capsys.readouterr()
        assert "No chains found" in captured.out

    def test_normal_report(self, sample_chain, capsys):
        report = val.build_report([sample_chain])
        val.print_report(report)
        captured = capsys.readouterr()
        assert "VALIDATION REPORT" in captured.out
        assert "Total chains" in captured.out

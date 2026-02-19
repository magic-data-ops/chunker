#!/usr/bin/env python3
"""Aggregate validation metrics from qa_chains_validated.json and write a report.

Usage:
    python validate_qa_dataset.py --input qa_chains_validated.json
    python validate_qa_dataset.py --input qa_chains_validated.json --output validation_report.json

All scores come from contractor output in contractor_polish.py — this script only aggregates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import List


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pass_rate(flags: List[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def _histogram(values: List[int]) -> dict:
    c = Counter(values)
    return {str(k): c[k] for k in sorted(c)}


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def build_report(chains: List[dict]) -> dict:
    total = len(chains)
    if total == 0:
        return {"error": "No chains found"}

    approved = [c for c in chains if c.get("approved") is True]
    approval_rate = len(approved) / total

    # Per-category breakdown
    by_category: dict = defaultdict(list)
    for c in chains:
        by_category[c.get("category", "unknown")].append(c)

    per_category: dict = {}
    for cat, cat_chains in by_category.items():
        cat_approved = [c for c in cat_chains if c.get("approved") is True]
        suitability_scores = [c["category_suitability_score"] for c in cat_chains
                              if "category_suitability_score" in c]
        completeness_scores = [c["answer_completeness_score"] for c in cat_chains
                               if "answer_completeness_score" in c]
        per_category[cat] = {
            "total": len(cat_chains),
            "approved": len(cat_approved),
            "approval_rate": len(cat_approved) / len(cat_chains),
            "mean_category_suitability": _mean(suitability_scores),
            "mean_answer_completeness": _mean(completeness_scores),
        }

    # Overall score means
    all_suitability = [c["category_suitability_score"] for c in chains
                       if "category_suitability_score" in c]
    all_completeness = [c["answer_completeness_score"] for c in chains
                        if "answer_completeness_score" in c]
    mean_suitability = _mean(all_suitability)
    mean_completeness = _mean(all_completeness)

    # Threshold pass rate: both scores >= 0.8
    threshold_passes = [
        c.get("category_suitability_score", 0.0) >= 0.8
        and c.get("answer_completeness_score", 0.0) >= 0.8
        for c in chains
    ]
    threshold_pass_rate = _pass_rate(threshold_passes)

    # Hop distribution
    hop_counts = [c.get("hop_count", 0) for c in chains]
    hop_histogram = _histogram(hop_counts)

    # Termination breakdown
    termination_counts = Counter(c.get("termination_reason", "unknown") for c in chains)
    termination_breakdown = dict(termination_counts)

    # Targets
    targets = {
        "overall_approval_rate": {"value": approval_rate, "target": 0.80,
                                   "pass": approval_rate >= 0.80},
        "mean_category_suitability": {"value": mean_suitability, "target": 0.80,
                                       "pass": mean_suitability >= 0.80},
        "mean_answer_completeness": {"value": mean_completeness, "target": 0.80,
                                      "pass": mean_completeness >= 0.80},
        "threshold_pass_rate": {"value": threshold_pass_rate, "target": 0.80,
                                 "pass": threshold_pass_rate >= 0.80},
    }

    return {
        "total_chains": total,
        "total_approved": len(approved),
        "targets": targets,
        "per_category": per_category,
        "hop_distribution": hop_histogram,
        "termination_breakdown": termination_breakdown,
        "informational": {
            "mean_hop_count": _mean(hop_counts),
            "single_answer_heuristic_count": sum(
                1 for c in chains if c.get("single_answer_heuristic")
            ),
        },
    }


def print_report(report: dict) -> None:
    if "error" in report:
        print(f"\n{report['error']}\n")
        return
    print("\n" + "=" * 60)
    print("Q&A DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"Total chains:    {report['total_chains']}")
    print(f"Total approved:  {report['total_approved']}")
    print()

    print("--- Targets (> 80%) ---")
    for metric, info in report["targets"].items():
        status = "PASS" if info["pass"] else "FAIL"
        print(f"  [{status}] {metric}: {info['value']:.1%}  (target: {info['target']:.0%})")

    print()
    print("--- Per-Category ---")
    for cat, stats in report.get("per_category", {}).items():
        print(f"  {cat}:")
        print(f"    total={stats['total']}  approved={stats['approved']}  "
              f"approval={stats['approval_rate']:.1%}")
        print(f"    suitability={stats['mean_category_suitability']:.2f}  "
              f"completeness={stats['mean_answer_completeness']:.2f}")

    print()
    print("--- Hop Distribution ---")
    for hops, count in report.get("hop_distribution", {}).items():
        bar = "#" * min(count, 50)
        print(f"  {hops} hops: {bar} ({count})")

    print()
    print("--- Termination Reasons ---")
    for reason, count in report.get("termination_breakdown", {}).items():
        print(f"  {reason}: {count}")

    print()
    info = report.get("informational", {})
    print(f"Mean hop count:            {info.get('mean_hop_count', 0):.2f}")
    print(f"Single-answer heuristic:   {info.get('single_answer_heuristic_count', 0)}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Q&A validation metrics")
    parser.add_argument("--input", default="qa_chains_validated.json",
                        help="Output of contractor_polish.py")
    parser.add_argument("--output", default="validation_report.json")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: '{args.input}' not found. Run contractor_polish.py first.")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        chains: List[dict] = json.load(f)
    print(f"Loaded {len(chains)} validated chains.")

    report = build_report(chains)
    print_report(report)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()

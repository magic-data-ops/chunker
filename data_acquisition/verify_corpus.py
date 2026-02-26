#!/usr/bin/env python3
"""Phase 6: Verify the assembled legal casebook corpus.

Runs quality checks on the final corpus:
1. Token count verification
2. Cross-reference integrity (cited case IDs exist)
3. Entity name consistency across documents
4. Numerical value consistency
5. Topic/jurisdiction/document-type distribution
6. Structural integrity (headers, delimiters)

Usage:
    python data_acquisition/verify_corpus.py \
        --corpus-dir corpus_text/legal_casebook \
        [--corpus-file corpus_text/legal_casebook/legal_casebook_complete.txt] \
        [--strict]

Requires: tqdm
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("Missing dependency: pip install tqdm")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN = 4.0

# Target ranges
TARGET_TOTAL_TOKENS_MIN = 150_000_000
TARGET_TOTAL_TOKENS_MAX = 250_000_000

# Expected document types
EXPECTED_DOC_TYPES = {
    "COURT_OPINION", "PLAINTIFF_BRIEF", "DEFENDANT_BRIEF",
    "MOTION_SUMMARY_JUDGMENT", "MOTION_DISCOVERY", "SETTLEMENT_AGREEMENT",
    "COURT_ORDER", "STATUTORY_TEXT", "COMMITTEE_REPORT", "FLOOR_DEBATE",
    "REGULATORY_GUIDANCE", "LAW_REVIEW_ARTICLE", "PRACTICE_GUIDE",
    "TREATISE_SECTION",
}

# Expected jurisdictions
EXPECTED_JURISDICTIONS = {
    "us", "ca-9", "ca-5", "ny", "tex",
    "us-supreme-court", "9th-circuit", "5th-circuit",
    "ny-court-of-appeals", "tx-supreme-court",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _estimate_tokens(char_count: int) -> int:
    return int(char_count / CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Verification checks
# ---------------------------------------------------------------------------


class VerificationResult:
    """Accumulates pass/fail results for verification checks."""

    def __init__(self):
        self.checks: list[dict] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def add_check(self, name: str, passed: bool, details: str = ""):
        status = "PASS" if passed else "FAIL"
        self.checks.append({"name": name, "status": status, "details": details})
        if not passed:
            self.errors.append(f"{name}: {details}")

    def add_warning(self, message: str):
        self.warnings.append(message)

    @property
    def all_passed(self) -> bool:
        return all(c["status"] == "PASS" for c in self.checks)

    def summary(self) -> str:
        lines = ["\n" + "=" * 60, "VERIFICATION REPORT", "=" * 60]
        passed = sum(1 for c in self.checks if c["status"] == "PASS")
        total = len(self.checks)
        lines.append(f"\nResults: {passed}/{total} checks passed\n")

        for check in self.checks:
            marker = "+" if check["status"] == "PASS" else "X"
            line = f"  [{marker}] {check['name']}"
            if check["details"]:
                line += f" — {check['details']}"
            lines.append(line)

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  ! {w}")

        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"  X {e}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


def parse_corpus_documents(corpus_path: str) -> list[dict]:
    """Parse the assembled corpus file into individual document records."""
    documents = []
    current_doc: dict[str, Any] = {}
    current_text_lines: list[str] = []
    header_pattern = re.compile(r"^=== DOCUMENT (\d+)/(\d+) \| TYPE: (\S+) ===$")

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line_stripped = line.rstrip("\n")
            match = header_pattern.match(line_stripped)

            if match:
                # Save previous document
                if current_doc:
                    current_doc["text"] = "\n".join(current_text_lines)
                    current_doc["char_count"] = len(current_doc["text"])
                    documents.append(current_doc)

                # Start new document
                current_doc = {
                    "doc_num": int(match.group(1)),
                    "total_num": int(match.group(2)),
                    "doc_type": match.group(3),
                    "metadata": {},
                }
                current_text_lines = []
            elif current_doc and line_stripped == "---":
                continue  # Skip delimiter
            elif current_doc and ":" in line_stripped and not current_text_lines:
                # Parse metadata lines (before the main text)
                key, _, value = line_stripped.partition(":")
                key = key.strip()
                value = value.strip()
                if key and value:
                    current_doc["metadata"][key] = value
            else:
                current_text_lines.append(line_stripped)

    # Save last document
    if current_doc:
        current_doc["text"] = "\n".join(current_text_lines)
        current_doc["char_count"] = len(current_doc["text"])
        documents.append(current_doc)

    return documents


def check_token_count(corpus_path: str, result: VerificationResult) -> int:
    """Verify total token count is within target range."""
    file_size = os.path.getsize(corpus_path)
    token_estimate = _estimate_tokens(file_size)

    in_range = TARGET_TOTAL_TOKENS_MIN <= token_estimate <= TARGET_TOTAL_TOKENS_MAX
    result.add_check(
        "Token count in target range",
        in_range,
        f"~{token_estimate:,} tokens "
        f"(target: {TARGET_TOTAL_TOKENS_MIN:,}–{TARGET_TOTAL_TOKENS_MAX:,})",
    )

    if token_estimate < TARGET_TOTAL_TOKENS_MIN:
        shortfall = TARGET_TOTAL_TOKENS_MIN - token_estimate
        result.add_warning(
            f"Token count {shortfall:,} below minimum target. "
            "Consider adding more cases or augmented documents."
        )
    elif token_estimate > TARGET_TOTAL_TOKENS_MAX:
        excess = token_estimate - TARGET_TOTAL_TOKENS_MAX
        result.add_warning(
            f"Token count {excess:,} above maximum target. "
            "Consider filtering the corpus."
        )

    return token_estimate


def check_structural_integrity(documents: list[dict], result: VerificationResult):
    """Verify document headers and structure."""
    # Check sequential numbering
    expected_total = len(documents)
    numbering_ok = True
    for i, doc in enumerate(documents):
        if doc["doc_num"] != i + 1:
            numbering_ok = False
            result.add_warning(f"Document numbering gap: expected {i + 1}, got {doc['doc_num']}")
            break
        if doc["total_num"] != expected_total:
            numbering_ok = False
            result.add_warning(f"Total count mismatch in doc {doc['doc_num']}: "
                               f"header says {doc['total_num']}, actual {expected_total}")
            break

    result.add_check(
        "Document numbering sequential",
        numbering_ok,
        f"{expected_total} documents",
    )

    # Check all documents have content
    empty_docs = [d for d in documents if d["char_count"] < 100]
    result.add_check(
        "All documents have content (>100 chars)",
        len(empty_docs) == 0,
        f"{len(empty_docs)} empty/short documents found" if empty_docs else "OK",
    )

    # Check document types
    found_types = {d["doc_type"] for d in documents}
    unexpected_types = found_types - EXPECTED_DOC_TYPES
    if unexpected_types:
        result.add_warning(f"Unexpected document types: {unexpected_types}")

    result.add_check(
        "Document types recognized",
        len(unexpected_types) == 0,
        f"Found types: {sorted(found_types)}",
    )


def check_cross_references(documents: list[dict], graph: dict | None,
                           result: VerificationResult):
    """Verify that cross-referenced case IDs exist in the corpus."""
    # Collect all case IDs from court opinions
    corpus_case_ids = set()
    for doc in documents:
        if doc["doc_type"] == "COURT_OPINION":
            case_id = doc["metadata"].get("CASE_ID", "")
            if case_id:
                corpus_case_ids.add(case_id)

    if not graph:
        result.add_check(
            "Cross-reference integrity",
            True,
            f"Skipped (no entity graph). {len(corpus_case_ids)} case IDs in corpus.",
        )
        return

    # Check citation edges
    broken_refs = 0
    total_edges = len(graph.get("citation_edges", []))
    for edge in graph.get("citation_edges", []):
        if edge["from"] not in corpus_case_ids:
            broken_refs += 1
        if edge["to"] not in corpus_case_ids:
            broken_refs += 1

    integrity_ok = broken_refs == 0
    result.add_check(
        "Citation graph references resolve",
        integrity_ok,
        f"{broken_refs} broken references out of {total_edges * 2} endpoints"
        if not integrity_ok else f"All {total_edges} citation edges valid",
    )


def check_entity_consistency(documents: list[dict], graph: dict | None,
                             result: VerificationResult):
    """Verify entity names are used consistently across documents."""
    if not graph:
        result.add_check(
            "Entity name consistency",
            True,
            "Skipped (no entity graph)",
        )
        return

    # Check a sample of judges/parties appear in the corpus text
    judge_names = [j["name"] for j in graph.get("judges", [])[:50]]
    found_judges = 0
    total_text = " ".join(d.get("text", "")[:5000] for d in documents[:500])

    for name in judge_names:
        if name in total_text:
            found_judges += 1

    if judge_names:
        judge_rate = found_judges / len(judge_names)
        result.add_check(
            "Judge names appear in corpus",
            judge_rate >= 0.5,
            f"{found_judges}/{len(judge_names)} sampled judge names found "
            f"({judge_rate:.0%})",
        )
    else:
        result.add_check("Judge names appear in corpus", True, "No judges in graph")


def check_numerical_consistency(documents: list[dict], graph: dict | None,
                                result: VerificationResult):
    """Spot-check numerical values are consistent across related documents."""
    if not graph:
        result.add_check(
            "Numerical value consistency",
            True,
            "Skipped (no entity graph)",
        )
        return

    # Check that numerical facts from the graph appear in court opinions
    facts = graph.get("numerical_facts", [])[:100]
    found = 0
    for fact in facts:
        raw_text = fact.get("raw_text", "")
        case_id = fact.get("case_id", "")
        # Find the corresponding court opinion
        for doc in documents:
            if (doc["doc_type"] == "COURT_OPINION"
                    and doc["metadata"].get("CASE_ID") == case_id):
                if raw_text in doc.get("text", ""):
                    found += 1
                break

    if facts:
        consistency_rate = found / len(facts)
        result.add_check(
            "Numerical facts present in source opinions",
            consistency_rate >= 0.7,
            f"{found}/{len(facts)} sampled facts found ({consistency_rate:.0%})",
        )
    else:
        result.add_check(
            "Numerical facts present in source opinions",
            True,
            "No numerical facts in graph",
        )


def check_distribution(documents: list[dict], result: VerificationResult) -> dict:
    """Verify topic/jurisdiction/document-type distribution."""
    type_dist = Counter(d["doc_type"] for d in documents)
    topic_dist = Counter(d["metadata"].get("TOPIC", "unknown") for d in documents)
    jurisdiction_dist = Counter(
        d["metadata"].get("JURISDICTION", "N/A")
        for d in documents
        if d["doc_type"] == "COURT_OPINION"
    )

    # Check we have multiple document types
    result.add_check(
        "Multiple document types present",
        len(type_dist) >= 3,
        f"{len(type_dist)} types: {dict(type_dist.most_common())}",
    )

    # Check we have multiple topics
    result.add_check(
        "Multiple topics present",
        len(topic_dist) >= 2,
        f"{len(topic_dist)} topics",
    )

    # Check we have multiple jurisdictions (for real cases)
    if jurisdiction_dist:
        result.add_check(
            "Multiple jurisdictions present",
            len(jurisdiction_dist) >= 2,
            f"{len(jurisdiction_dist)} jurisdictions: {dict(jurisdiction_dist)}",
        )

    # Check for reasonable balance (no single type > 80% unless it's court opinions)
    total = len(documents)
    for dtype, count in type_dist.items():
        ratio = count / total
        if ratio > 0.80 and dtype != "COURT_OPINION":
            result.add_warning(
                f"Document type '{dtype}' dominates at {ratio:.0%} ({count}/{total}). "
                "Consider generating more documents of other types."
            )

    return {
        "document_type_distribution": dict(type_dist),
        "topic_distribution": dict(topic_dist),
        "jurisdiction_distribution": dict(jurisdiction_dist),
    }


def check_non_chronological(documents: list[dict], result: VerificationResult):
    """Verify that documents are not in strict chronological order."""
    dates = []
    for doc in documents:
        if doc["doc_type"] == "COURT_OPINION":
            date = doc["metadata"].get("DATE", "")
            if date:
                dates.append(date)

    if len(dates) < 10:
        result.add_check(
            "Non-chronological ordering",
            True,
            "Insufficient dated documents to verify",
        )
        return

    # Check if dates are sorted (they shouldn't be)
    is_sorted = all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))
    result.add_check(
        "Non-chronological ordering (temporal_ordering challenge)",
        not is_sorted,
        "Documents are NOT in chronological order (good)"
        if not is_sorted else "Documents are in chronological order (bad — defeats temporal_ordering tests)",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Verify the assembled legal casebook corpus."
    )
    parser.add_argument(
        "--corpus-dir", default="corpus_text/legal_casebook",
        help="Directory containing case files and metadata.",
    )
    parser.add_argument(
        "--corpus-file", default=None,
        help="Path to the assembled corpus file. "
             "Defaults to {corpus-dir}/legal_casebook_complete.txt.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit with non-zero code if any check fails.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for verification report JSON. "
             "Defaults to {corpus-dir}/metadata/verification_report.json.",
    )
    args = parser.parse_args()

    if args.corpus_file is None:
        args.corpus_file = os.path.join(args.corpus_dir, "legal_casebook_complete.txt")
    if args.output is None:
        args.output = os.path.join(args.corpus_dir, "metadata", "verification_report.json")

    if not os.path.exists(args.corpus_file):
        sys.exit(f"Corpus file not found: {args.corpus_file}\n"
                 "Run assemble_corpus.py first.")

    result = VerificationResult()

    # 1. Token count
    logger.info("Checking token count...")
    token_estimate = check_token_count(args.corpus_file, result)

    # 2. Parse corpus documents
    logger.info("Parsing corpus documents...")
    documents = parse_corpus_documents(args.corpus_file)
    logger.info(f"Parsed {len(documents)} documents")

    # 3. Structural integrity
    logger.info("Checking structural integrity...")
    check_structural_integrity(documents, result)

    # 4. Load entity graph (optional)
    graph = None
    graph_path = os.path.join(args.corpus_dir, "metadata", "entity_graph.json")
    if os.path.exists(graph_path):
        with open(graph_path) as f:
            graph = json.load(f)
        logger.info(f"Loaded entity graph: {graph.get('summary', {})}")

    # 5. Cross-reference integrity
    logger.info("Checking cross-references...")
    check_cross_references(documents, graph, result)

    # 6. Entity consistency
    logger.info("Checking entity consistency...")
    check_entity_consistency(documents, graph, result)

    # 7. Numerical consistency
    logger.info("Checking numerical consistency...")
    check_numerical_consistency(documents, graph, result)

    # 8. Distribution
    logger.info("Checking distribution...")
    distribution = check_distribution(documents, result)

    # 9. Non-chronological ordering
    logger.info("Checking non-chronological ordering...")
    check_non_chronological(documents, result)

    # Save report
    report = {
        "corpus_file": args.corpus_file,
        "token_estimate": token_estimate,
        "total_documents": len(documents),
        "checks": result.checks,
        "warnings": result.warnings,
        "errors": result.errors,
        "distribution": distribution,
        "all_passed": result.all_passed,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print(result.summary())
    print(f"\nReport saved to: {args.output}")

    if args.strict and not result.all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
